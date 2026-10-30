//! Live Rust L0 -> Engram -> L1 -> compression/index -> L2 on physical TP4.
//! Trace fixtures remain a correctness oracle, never execution inputs for L1.
use crate::{
    attention::{Candidates, PagedKv},
    blas::Blas,
    checkpoint::Checkpoint,
    compressed_source::{Layer as Source, Metadata},
    config::ModelConfig,
    cuda::Cuda,
    decoder::SwaLayer,
    engram::{PackedShard, RowStore},
    engram_gpu::Rows,
    expert_packing,
    ffn::TokenRows,
    hash_history::{HashMetadata, History},
    kv::RowLocations,
    layer_attention::InputMode,
    layer_probe::{Cases, bf16_bytes, compare, fixture, integers, snapshots},
    layout::RankPlan,
    prelude::{FrequencyTable, Positions},
    roce::Roce,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    rc::Rc,
};
#[derive(Deserialize)]
struct Peer {
    config: PathBuf,
    checkpoint: PathBuf,
    rank: usize,
    core: PathBuf,
    kernels: PathBuf,
    staged0: PathBuf,
    staged1: PathBuf,
    staged2: PathBuf,
    #[serde(default)]
    shared_stages: Vec<PathBuf>,
    #[serde(default)]
    shared_fixtures: Vec<PathBuf>,
    #[serde(default)]
    next_source_stages: Vec<PathBuf>,
    #[serde(default)]
    next_source_fixtures: Vec<PathBuf>,
    #[serde(default)]
    engram_source_stages: Vec<PathBuf>,
    #[serde(default)]
    engram_source_fixtures: Vec<PathBuf>,
    #[serde(default)]
    late_stages: Vec<PathBuf>,
    #[serde(default)]
    late_fixtures: Vec<PathBuf>,
    fixture0: PathBuf,
    fixture1: PathBuf,
    fixture2: PathBuf,
    backing: PathBuf,
    hash_metadata: PathBuf,
    endpoint: String,
    ticket: PathBuf,
}
pub fn run(path: &Path) -> Result<Value> {
    let p: Peer = serde_json::from_slice(&std::fs::read(path)?)?;
    let f0: Cases = serde_json::from_slice(&std::fs::read(p.fixture0.join("index.json"))?)?;
    let f1: Cases = serde_json::from_slice(&std::fs::read(p.fixture1.join("index.json"))?)?;
    let manifest2: Value = serde_json::from_slice(&std::fs::read(p.fixture2.join("index.json"))?)?;
    let f2: Cases = serde_json::from_value(manifest2.clone())?;
    let slots: Vec<i32> = serde_json::from_value(manifest2["slots"].clone())?;
    ensure!(
        (f2.rows, f2.world, f2.layer, f2.frequency_rows) == (6, 4, 2, 1048576)
            && f2.cases.len() == 4
            && slots.len() == 67
            && manifest2["kv_pages"] == 4
            && manifest2["index_pages"] == 2,
        "third layer trace geometry"
    );
    ensure!(
        p.rank < 4
            && (f0.rows, f0.world, f0.layer, f0.frequency_rows) == (6, 4, 0, 131072)
            && (f1.rows, f1.world, f1.layer, f1.frequency_rows) == (6, 4, 1, 131072)
            && f0.cases.len() == 4
            && f1.cases.len() == 4,
        "two-layer trace geometry"
    );
    let cfg = ModelConfig::parse(&std::fs::read(p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let plan = RankPlan::build(&cp, &cfg, p.rank, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let core = cuda.module(&std::fs::read(&p.core)?)?;
    let frequency = Rc::new(FrequencyTable::new(&cuda, &core, &stream, 131072, false)?);
    let mut layers = Vec::new();
    let mut atlas_bytes = Vec::new();
    let mut source_hashes = Vec::new();
    for (layer, staged) in [(0, &p.staged0), (1, &p.staged1)] {
        let mut w =
            Weights::load_staged(&cuda, &cp, &plan, &[&format!("layers.{layer}.")], staged)?;
        atlas_bytes.push(w.bytes());
        source_hashes.push(w.source_sha256.clone());
        let prepared = expert_packing::prepare_layer(&cuda, &core, &mut w, layer)?;
        layers.push(SwaLayer::new_with_mode(
            &cuda,
            &core,
            &stream,
            Rc::new(w),
            prepared,
            &p.kernels,
            frequency.clone(),
            InputMode::Split(p.rank),
        )?);
    }
    let mut layer1 = layers.pop().unwrap();
    let mut layer0 = layers.pop().unwrap();
    let core_bytes = std::fs::read(&p.core)?;
    let frequency2 = Rc::new(FrequencyTable::new(&cuda, &core, &stream, 1048576, true)?);
    let mut w2 = Weights::load_staged(&cuda, &cp, &plan, &["layers.2."], &p.staged2)?;
    atlas_bytes.push(w2.bytes());
    source_hashes.push(w2.source_sha256.clone());
    let prepared2 = expert_packing::prepare_layer(&cuda, &core, &mut w2, 2)?;
    let w2 = Rc::new(w2);
    let mut layer2 = SwaLayer::new_with_mode(
        &cuda,
        &core,
        &stream,
        w2.clone(),
        prepared2,
        &p.kernels,
        frequency2.clone(),
        InputMode::Split(p.rank),
    )?;
    ensure!(
        p.shared_stages.len() == p.shared_fixtures.len() && [0, 5].contains(&p.shared_stages.len()),
        "only the first complete source-sharing group is admitted"
    );
    let mut shared_layers = Vec::new();
    let mut shared_caches = Vec::new();
    let mut shared_cases = Vec::new();
    for (i, (stage, fixture_dir)) in p.shared_stages.iter().zip(&p.shared_fixtures).enumerate() {
        let id = i + 3;
        let f: Cases = serde_json::from_slice(&std::fs::read(fixture_dir.join("index.json"))?)?;
        ensure!(
            (f.rows, f.world, f.layer, f.frequency_rows) == (6, 4, id, 1048576)
                && f.cases.len() == 4,
            "shared layer trace geometry"
        );
        let mut w = Weights::load_staged(&cuda, &cp, &plan, &[&format!("layers.{id}.")], stage)?;
        atlas_bytes.push(w.bytes());
        source_hashes.push(w.source_sha256.clone());
        let prepared = expert_packing::prepare_layer(&cuda, &core, &mut w, id)?;
        shared_layers.push(SwaLayer::new_with_mode(
            &cuda,
            &core,
            &stream,
            Rc::new(w),
            prepared,
            &p.kernels,
            frequency2.clone(),
            InputMode::Split(p.rank),
        )?);
        shared_caches.push(PagedKv::new(&cuda, 4, 64)?);
        shared_cases.push(f);
    }
    let mut source = Source::new(
        &cuda,
        &stream,
        w2,
        Rc::new(Blas::new(&cuda)?),
        2,
        &p.kernels,
        frequency2.clone(),
        &core_bytes,
        4,
        2,
        &slots,
        &[0; 6],
    )?;
    let mut group_fixtures = Vec::new();
    let mut next_groups = Vec::new();
    for (start, stages, fixture_paths) in [
        (8, &p.next_source_stages, &p.next_source_fixtures),
        (14, &p.engram_source_stages, &p.engram_source_fixtures),
    ] {
        ensure!(
            stages.len() == fixture_paths.len() && [0, 6].contains(&stages.len()),
            "source group stage/fixture geometry"
        );
        if stages.is_empty() {
            continue;
        }
        ensure!(
            shared_layers.len() == 5 && (start == 8 || next_groups.len() == 1),
            "source group requires complete predecessor"
        );
        let fixtures = crate::source_group_probe::Fixtures::new(fixture_paths, start)?;
        let group = crate::source_group::Group::new(
            &cuda,
            &core,
            &core_bytes,
            &stream,
            &cp,
            &plan,
            start,
            stages,
            &p.kernels,
            frequency2.clone(),
            fixtures.slots(),
            4,
            2,
            &p.backing,
        )?;
        atlas_bytes.extend_from_slice(&group.atlas_bytes);
        source_hashes.extend_from_slice(&group.source_hashes);
        group_fixtures.push(fixtures);
        next_groups.push(group);
    }
    ensure!(
        p.late_stages.len() == p.late_fixtures.len()
            && [0, 4, 8, 12, 16, 20].contains(&p.late_stages.len()),
        "late target stage/fixture geometry"
    );
    let late_fixtures = if p.late_stages.is_empty() {
        None
    } else {
        ensure!(
            next_groups.len() == 2,
            "late target graph requires complete source14 group"
        );
        Some(crate::source_group_probe::Fixtures::new(
            &p.late_fixtures,
            20,
        )?)
    };
    let mut late_group = if let Some(fixtures) = &late_fixtures {
        let group = crate::late_group::Group::new(
            &cuda,
            &core,
            &core_bytes,
            &stream,
            &cp,
            &plan,
            &p.late_stages,
            &p.kernels,
            frequency2.clone(),
            fixtures.slots(),
            8,
            4,
        )?;
        atlas_bytes.extend_from_slice(&group.atlas_bytes);
        source_hashes.extend_from_slice(&group.source_hashes);
        Some(group)
    } else {
        None
    };
    let mut cache2 = PagedKv::new(&cuda, 4, 64)?;
    let seed2 = std::fs::read(p.fixture2.join("initial_cache.bin"))?;
    let mut cache0 = PagedKv::new(&cuda, 4, 64)?;
    let mut cache1 = PagedKv::new(&cuda, 4, 64)?;
    let raw = std::fs::read(p.fixture0.join("initial_cache.bin"))?;
    cache0.upload(&raw)?;
    cache1.upload(&raw)?;
    let pos = integers(&p.fixture0.join("positions.bin"))?;
    let positions = Positions::upload(&cuda, &pos, 131072)?;
    let locations = RowLocations::upload(
        &cuda,
        &pos,
        &integers(&p.fixture0.join("locations.bin"))?,
        cache0.capacity(),
        131072,
    )?;
    let candidates = Candidates::upload(
        &cuda,
        6,
        128,
        cache0.capacity(),
        &integers(&p.fixture0.join("ids.bin"))?,
        &integers(&p.fixture0.join("lengths.bin"))?,
    )?;
    let store = RowStore::new(
        PackedShard::open(
            &p.backing.join(format!("engram-l1-r{}of4.bin", p.rank)),
            1,
            p.rank,
            4,
            384006168,
        )?,
        2 * 1024 * 1024 * 1024,
    )?;
    let mut rows = Rows::new(&cuda, 6, &store)?;
    let metadata = HashMetadata::parse(&std::fs::read(p.hash_metadata)?)?;
    let mut input = cuda.allocate::<bf16>(6 * 20480)?;
    let mut tokens = TokenRows::new(&cuda, &[0; 6], 6)?;
    let mut comm = Roce::local(&cuda, &p.kernels.join("roce-v0"), p.rank)?;
    comm.attach_gather(&p.kernels.join("roce-gather-v1"))?;
    let all = crate::rendezvous::exchange_blob_with_timeout(
        p.rank,
        p.endpoint.parse()?,
        &std::fs::read(p.ticket)?,
        &comm.blob()?,
        std::time::Duration::from_secs(600),
    )?;
    comm.connect(&all)?;
    let mut reports = Vec::new();
    let mut all_equal = true;
    for ((c0, c1), c2) in f0.cases.iter().zip(&f1.cases).zip(&f2.cases) {
        ensure!(c0.name == c1.name, "trace phases disagree");
        let d0 = p.fixture0.join(&c0.name);
        let d1 = p.fixture1.join(&c1.name);
        ensure!(c2.name == c1.name, "third-layer phase mismatch");
        let d2 = p.fixture2.join(&c2.name);
        let s2: BTreeMap<_, _> = c2
            .source
            .iter()
            .map(|t| Ok((t.name.as_str(), fixture(&d2, t)?)))
            .collect::<Result<_>>()?;
        let i32s = |n: &str| -> Vec<i32> {
            s2[n]
                .chunks_exact(4)
                .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
                .collect()
        };
        let pos2 = i32s("positions");
        let loc2 = i32s("locations");
        let source_lengths = i32s("source_lengths");
        let raw2: Vec<i64> = s2["source_raw"]
            .chunks_exact(8)
            .map(|x| i64::from_le_bytes(x.try_into().unwrap()))
            .collect();
        cache2.upload(&seed2)?;
        source.source_mut().cache_mut().upload(&s2["seed_cache"])?;
        source.keys_mut().upload(&s2["seed_keys"])?;
        source.source_mut().ring_mut().unwrap().upload(
            &s2["seed_ring"]
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect::<Vec<_>>(),
        )?;
        source.source_mut().latent_mut().upload(
            &s2["seed_latent"]
                .chunks_exact(2)
                .map(|x| bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())))
                .collect::<Vec<_>>(),
        )?;
        let positions2 = Positions::upload(&cuda, &pos2, 1048576)?;
        let locations2 = RowLocations::upload(&cuda, &pos2, &loc2, cache2.capacity(), 1048576)?;
        let candidates2 = Candidates::upload(
            &cuda,
            6,
            128,
            cache2.capacity(),
            &i32s("ids"),
            &i32s("lengths"),
        )?;
        let meta2 = Metadata::new(
            &cuda,
            &source,
            &pos2,
            &[0; 6],
            &raw2,
            &slots,
            &source_lengths,
            1048576,
        )?;
        for (i, cases) in shared_cases.iter().enumerate() {
            let c = cases
                .cases
                .iter()
                .find(|c| c.name == c0.name)
                .ok_or_else(|| anyhow::anyhow!("shared layer phase absent"))?;
            let dir = p.shared_fixtures[i].join(&c.name);
            for name in ["positions", "locations", "ids", "lengths"] {
                let t = c
                    .source
                    .iter()
                    .find(|t| t.name == name)
                    .ok_or_else(|| anyhow::anyhow!("shared metadata absent"))?;
                ensure!(
                    fixture(&dir, t)? == s2[name],
                    "shared layer step metadata mismatch"
                );
            }
            shared_caches[i].upload(&seed2)?;
        }
        let group_steps = group_fixtures
            .iter()
            .zip(&mut next_groups)
            .map(|(fixtures, group)| fixtures.prepare(&cuda, group, &c0.name))
            .collect::<Result<Vec<_>>>()?;
        let late_step = match (&late_fixtures, &mut late_group) {
            (Some(fixtures), Some(group)) => Some(fixtures.prepare(&cuda, group, &c0.name)?),
            (None, None) => None,
            _ => anyhow::bail!("late target fixture/graph mismatch"),
        };
        let mut graph = None;
        let s0: BTreeMap<_, _> = c0
            .source
            .iter()
            .map(|t| Ok((t.name.as_str(), fixture(&d0, t)?)))
            .collect::<Result<_>>()?;
        let s1: BTreeMap<_, _> = c1
            .source
            .iter()
            .map(|t| Ok((t.name.as_str(), fixture(&d1, t)?)))
            .collect::<Result<_>>()?;
        ensure!(
            s0["input"].len() == input.bytes() && s0["tokens"].len() == 48 && s0["live"].len() == 4,
            "trace source sizes"
        );
        input.upload(
            &s0["input"]
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
                .collect::<Vec<_>>(),
        )?;
        let ids: Vec<i64> = s0["tokens"]
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let live = i32::from_le_bytes(s0["live"].as_slice().try_into().unwrap());
        ensure!((0..=6).contains(&live), "live token rows");
        tokens.update(&ids, live as usize)?;
        let mut history = History::new(1);
        let prefix = history.prepare(&[21, 42, 84], &metadata)?;
        history.commit(prefix, 3)?;
        let hash = history.prepare(
            &ids.iter().map(|t| *t as u32).collect::<Vec<_>>(),
            &metadata,
        )?;
        let hashes: Vec<_> = hash.rows.iter().map(|row| row[0]).collect();
        let hash_bytes: Vec<u8> = hashes
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        ensure!(
            hash_bytes == s1["hash_ids"],
            "native Engram hash disagrees with layer reference"
        );
        rows.prepare(&hashes, &store, 8)?;
        for (group, step) in next_groups.iter_mut().zip(&group_steps) {
            if let Some(expected) = &step.expected_hashes {
                let hashes14: Vec<_> = hash.rows.iter().map(|row| row[1]).collect();
                let bytes: Vec<u8> = hashes14
                    .iter()
                    .flatten()
                    .flat_map(|x| x.to_le_bytes())
                    .collect();
                ensure!(bytes == *expected, "native Engram14 history mismatch");
                group.prepare_engram(&hashes14)?;
            }
        }
        if graph.is_none() {
            layer0.run(
                &input,
                &positions,
                &locations,
                &mut cache0,
                &candidates,
                &tokens,
                &comm,
                &stream,
            )?;
            layer1.run_from_previous(
                layer0.output(),
                Some(layer0.pre()),
                Some(&mut rows),
                &positions,
                &locations,
                &mut cache1,
                &candidates,
                &tokens,
                &comm,
                &stream,
            )?;
            layer2.run_indexed_from_previous(
                layer1.output(),
                layer1.pre(),
                &positions2,
                &locations2,
                &mut cache2,
                &candidates2,
                &tokens,
                &mut source,
                &meta2,
                &comm,
                &stream,
            )?;
            run_shared_chain(
                &mut shared_layers,
                &mut shared_caches,
                layer2.output(),
                layer2.pre(),
                &positions2,
                &locations2,
                &candidates2,
                &tokens,
                &source,
                &comm,
                &stream,
            )?;
            run_next_groups(
                &mut next_groups,
                &group_steps,
                &shared_layers,
                &tokens,
                &comm,
                &stream,
            )?;
            run_late_group(
                &mut late_group,
                &late_step,
                &next_groups,
                &tokens,
                &comm,
                &stream,
            )?;
            stream.synchronize()?;
            comm.health()?;
            graph = Some(stream.capture(|| {
                layer0.run(
                    &input,
                    &positions,
                    &locations,
                    &mut cache0,
                    &candidates,
                    &tokens,
                    &comm,
                    &stream,
                )?;
                layer1.run_from_previous(
                    layer0.output(),
                    Some(layer0.pre()),
                    Some(&mut rows),
                    &positions,
                    &locations,
                    &mut cache1,
                    &candidates,
                    &tokens,
                    &comm,
                    &stream,
                )?;
                layer2.run_indexed_from_previous(
                    layer1.output(),
                    layer1.pre(),
                    &positions2,
                    &locations2,
                    &mut cache2,
                    &candidates2,
                    &tokens,
                    &mut source,
                    &meta2,
                    &comm,
                    &stream,
                )?;
                run_shared_chain(
                    &mut shared_layers,
                    &mut shared_caches,
                    layer2.output(),
                    layer2.pre(),
                    &positions2,
                    &locations2,
                    &candidates2,
                    &tokens,
                    &source,
                    &comm,
                    &stream,
                )?;
                run_next_groups(
                    &mut next_groups,
                    &group_steps,
                    &shared_layers,
                    &tokens,
                    &comm,
                    &stream,
                )?;
                run_late_group(
                    &mut late_group,
                    &late_step,
                    &next_groups,
                    &tokens,
                    &comm,
                    &stream,
                )
            })?);
        }
        for _ in 0..100 {
            unsafe { graph.as_ref().unwrap().launch(&stream) }?;
        }
        let events = (0..100)
            .map(|_| Ok((cuda.event()?, cuda.event()?)))
            .collect::<Result<Vec<_>>>()?;
        for (a, b) in &events {
            a.record(&stream)?;
            unsafe { graph.as_ref().unwrap().launch(&stream) }?;
            b.record(&stream)?;
        }
        stream.synchronize()?;
        comm.health()?;
        let mut ms = events
            .iter()
            .map(|(a, b)| b.elapsed_since(a))
            .collect::<Result<Vec<_>>>()?;
        ms.sort_by(f32::total_cmp);
        ensure!(
            bf16_bytes(layer0.output())? == s1["input"],
            "live layer0 output does not match canonical layer1 input"
        );
        let mut traces = Vec::new();
        let mut differences = 0usize;
        for (layer, case, dir, cache) in [(&layer0, c0, &d0, &cache0), (&layer1, c1, &d1, &cache1)]
        {
            let (mut local, mut global) = snapshots(&cuda, layer, cache)?;
            if let Some(e) = layer.engram() {
                local.insert("engram_local_embedding".into(), bf16_bytes(rows.local())?);
                local.insert(
                    "engram_local_projection".into(),
                    bf16_bytes(e.projected_local())?,
                );
                for (n, b) in [
                    ("engram_embedding", e.embedding()),
                    ("engram_projected", e.projected()),
                    ("engram_output", e.output()),
                ] {
                    global.insert(n.into(), bf16_bytes(b)?);
                }
            }
            let expected = case
                .ranks
                .iter()
                .find(|r| r.rank == p.rank)
                .ok_or_else(|| anyhow::anyhow!("missing rank trace"))?;
            let a = compare(&dir.join(format!("rank{}", p.rank)), &expected.taps, &local)?;
            let b = compare(dir, &case.global_taps, &global)?;
            differences += a
                .iter()
                .chain(&b)
                .map(|t| t["different_values"].as_u64().unwrap() as usize)
                .sum::<usize>();
            traces.push(json!({"local_taps":a,"global_taps":b}));
        }
        ensure!(
            bf16_bytes(layer1.output())? == s2["input"],
            "live L1->L2 input mismatch"
        );
        ensure!(source.error()? == 0, "third-layer candidate fault");
        let (local, mut global) = snapshots(&cuda, &layer2, &cache2)?;
        for (n, b) in
            crate::compressed_source_probe::snapshots(&source, slots.len(), &source_lengths)?
        {
            global.insert(format!("source_{n}"), b);
        }
        let expected = c2
            .ranks
            .iter()
            .find(|r| r.rank == p.rank)
            .ok_or_else(|| anyhow::anyhow!("third-layer rank trace absent"))?;
        let a = compare(&d2.join(format!("rank{}", p.rank)), &expected.taps, &local)?;
        let b = compare(&d2, &c2.global_taps, &global)?;
        differences += a
            .iter()
            .chain(&b)
            .map(|t| t["different_values"].as_u64().unwrap() as usize)
            .sum::<usize>();
        traces.push(json!({"local_taps":a,"global_taps":b}));
        for (i, layer) in shared_layers.iter().enumerate() {
            let c = shared_cases[i]
                .cases
                .iter()
                .find(|c| c.name == c0.name)
                .ok_or_else(|| anyhow::anyhow!("shared comparison phase absent"))?;
            let dir = p.shared_fixtures[i].join(&c.name);
            let input_tap = c
                .source
                .iter()
                .find(|t| t.name == "input")
                .ok_or_else(|| anyhow::anyhow!("shared comparison input absent"))?;
            let previous_output = if i == 0 {
                layer2.output()
            } else {
                shared_layers[i - 1].output()
            };
            ensure!(
                bf16_bytes(previous_output)? == fixture(&dir, input_tap)?,
                "live shared-layer handoff mismatch"
            );
            let (local, global) = snapshots(&cuda, layer, &shared_caches[i])?;
            let expected = c
                .ranks
                .iter()
                .find(|r| r.rank == p.rank)
                .ok_or_else(|| anyhow::anyhow!("shared rank trace absent"))?;
            let a = compare(&dir.join(format!("rank{}", p.rank)), &expected.taps, &local)?;
            let b = compare(&dir, &c.global_taps, &global)?;
            differences += a
                .iter()
                .chain(&b)
                .map(|t| t["different_values"].as_u64().unwrap() as usize)
                .sum::<usize>();
            traces.push(json!({"local_taps":a,"global_taps":b}));
        }
        for (i, ((fixtures, group), step)) in group_fixtures
            .iter()
            .zip(&next_groups)
            .zip(&group_steps)
            .enumerate()
        {
            let (output, pre) = if i == 0 {
                let previous = shared_layers
                    .last()
                    .ok_or_else(|| anyhow::anyhow!("source group predecessor absent"))?;
                (previous.output(), previous.pre())
            } else {
                (next_groups[i - 1].output(), next_groups[i - 1].pre())
            };
            ensure!(
                bf16_bytes(output)? == step.expected_input,
                "live source-group input mismatch"
            );
            let pre_bytes: Vec<u8> = pre
                .download()?
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            ensure!(
                pre_bytes == step.expected_pre,
                "live source-group coefficient mismatch"
            );
            let (extra_traces, extra_differences) =
                fixtures.compare(&cuda, group, step, &c0.name, p.rank)?;
            traces.extend(extra_traces);
            differences += extra_differences;
        }
        if let (Some(fixtures), Some(group), Some(step)) = (&late_fixtures, &late_group, &late_step)
        {
            let previous = next_groups
                .last()
                .ok_or_else(|| anyhow::anyhow!("late target predecessor absent"))?;
            ensure!(
                bf16_bytes(previous.output())? == step.expected_input,
                "live L19->L20 input mismatch"
            );
            let pre_bytes: Vec<u8> = previous
                .pre()
                .download()?
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            ensure!(
                pre_bytes == step.expected_pre,
                "live L19->L20 coefficient mismatch"
            );
            let (extra_traces, extra_differences) =
                fixtures.compare(&cuda, group, step, &c0.name, p.rank)?;
            traces.extend(extra_traces);
            differences += extra_differences;
        }
        all_equal &= differences == 0;
        eprintln!(
            "rank {} connected {}: different={differences}",
            p.rank, c0.name
        );
        reports.push(
            json!({"case":c0.name,"different_values":differences,"traces":traces,
            "gpu_ms":{"p50":ms[49],"p95":ms[94],"p99":ms[98],"max":ms[99]},"samples_ms":ms}),
        );
    }
    stream.synchronize()?;
    comm.health()?;
    Ok(
        json!({"rank":p.rank,"all_bitwise_equal":all_equal,"cases":reports,"atlas_bytes":atlas_bytes,
        "source_tp_bytes_sha256":source_hashes,"layers":3+shared_layers.len()+6*next_groups.len()+p.late_stages.len(),"ordered_collectives_per_step":11+3*shared_layers.len()+18*next_groups.len()+3*p.late_stages.len()+if p.engram_source_stages.is_empty(){0}else{2},"scope":"Live decoder prefix with Engram/image and shared compression/index/physical candidates, controlled past; not full model, full prefill or session qualification"}),
    )
}

fn run_late_group(
    group: &mut Option<crate::late_group::Group>,
    step: &Option<crate::source_group_probe::Step>,
    preceding: &[crate::source_group::Group],
    tokens: &TokenRows,
    comm: &Roce,
    stream: &crate::cuda::Stream,
) -> Result<()> {
    match (group, step) {
        (Some(group), Some(step)) => {
            let previous = preceding
                .last()
                .ok_or_else(|| anyhow::anyhow!("source20 predecessor absent"))?;
            group.run(
                previous.output(),
                previous.pre(),
                &step.positions,
                &step.locations,
                &step.candidates,
                tokens,
                &step.metadata,
                comm,
                stream,
            )
        }
        (None, None) => Ok(()),
        _ => anyhow::bail!("late target step/graph mismatch"),
    }
}

fn run_next_groups(
    groups: &mut [crate::source_group::Group],
    steps: &[crate::source_group_probe::Step],
    preceding: &[SwaLayer],
    tokens: &TokenRows,
    comm: &Roce,
    stream: &crate::cuda::Stream,
) -> Result<()> {
    ensure!(groups.len() == steps.len(), "source group step count");
    if groups.is_empty() {
        return Ok(());
    }
    let first = preceding
        .last()
        .ok_or_else(|| anyhow::anyhow!("source group predecessor absent"))?;
    let mut residual = first.output();
    let mut pre = first.pre();
    for (group, step) in groups.iter_mut().zip(steps) {
        group.run(
            residual,
            pre,
            &step.positions,
            &step.locations,
            &step.candidates,
            tokens,
            &step.metadata,
            comm,
            stream,
        )?;
        residual = group.output();
        pre = group.pre();
    }
    Ok(())
}

fn run_shared_chain(
    layers: &mut [SwaLayer],
    caches: &mut [PagedKv],
    input: &crate::cuda::Buffer<bf16>,
    pre: &crate::cuda::Buffer<f32>,
    positions: &Positions,
    locations: &RowLocations,
    candidates: &Candidates,
    tokens: &TokenRows,
    source: &Source,
    comm: &Roce,
    stream: &crate::cuda::Stream,
) -> Result<()> {
    ensure!(layers.len() == caches.len(), "shared graph cache count");
    let mut residual = input;
    let mut previous = pre;
    for (layer, cache) in layers.iter_mut().zip(caches) {
        layer.run_shared_from_previous(
            residual, previous, positions, locations, cache, candidates, tokens, source, comm,
            stream,
        )?;
        residual = layer.output();
        previous = layer.pre();
    }
    Ok(())
}
