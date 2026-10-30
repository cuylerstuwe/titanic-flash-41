//! Live Rust L0 -> Engram -> L1 graph with predecessor mHC and physical TP4.
//! Trace fixtures remain a correctness oracle, never execution inputs for L1.
use crate::{
    attention::{Candidates, PagedKv},
    checkpoint::Checkpoint,
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
    fixture0: PathBuf,
    fixture1: PathBuf,
    backing: PathBuf,
    hash_metadata: PathBuf,
    endpoint: String,
    ticket: PathBuf,
}
pub fn run(path: &Path) -> Result<Value> {
    let p: Peer = serde_json::from_slice(&std::fs::read(path)?)?;
    let f0: Cases = serde_json::from_slice(&std::fs::read(p.fixture0.join("index.json"))?)?;
    let f1: Cases = serde_json::from_slice(&std::fs::read(p.fixture1.join("index.json"))?)?;
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
    let core = cuda.module(&std::fs::read(p.core)?)?;
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
    let all = crate::rendezvous::exchange_blob(
        p.rank,
        p.endpoint.parse()?,
        &std::fs::read(p.ticket)?,
        &comm.blob()?,
    )?;
    comm.connect(&all)?;
    let mut graph = None;
    let mut reports = Vec::new();
    let mut all_equal = true;
    for (c0, c1) in f0.cases.iter().zip(&f1.cases) {
        ensure!(c0.name == c1.name, "trace phases disagree");
        let d0 = p.fixture0.join(&c0.name);
        let d1 = p.fixture1.join(&c1.name);
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
        "source_tp_bytes_sha256":source_hashes,"scope":"Live L0 -> Engram/image -> L1/predecessor mHC graph, controlled past; not full model or session qualification"}),
    )
}
