//! Physical TP4 layer2 qualification with live compression/index/candidates.
use crate::{
    attention::{Candidates, PagedKv},
    blas::Blas,
    checkpoint::Checkpoint,
    compressed_source::{Layer as Source, Metadata},
    config::ModelConfig,
    cuda::{Buffer, Cuda},
    decoder::SwaLayer,
    expert_packing,
    ffn::TokenRows,
    kv::RowLocations,
    layer_attention::InputMode,
    layer_probe::{Case, compare, fixture, snapshots},
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
struct Config {
    config: PathBuf,
    checkpoint: PathBuf,
    rank: usize,
    staged: PathBuf,
    core: PathBuf,
    kernels: PathBuf,
    fixture: PathBuf,
    endpoint: String,
    ticket: PathBuf,
}
#[derive(Deserialize)]
struct Cases {
    rows: usize,
    world: usize,
    layer: usize,
    frequency_rows: usize,
    slots: Vec<i32>,
    kv_pages: usize,
    index_pages: usize,
    cases: Vec<Case>,
}
fn ints(b: &[u8]) -> Result<Vec<i32>> {
    ensure!(b.len() % 4 == 0, "integer metadata bytes");
    Ok(b.chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn longs(b: &[u8]) -> Result<Vec<i64>> {
    ensure!(b.len() % 8 == 0, "long metadata bytes");
    Ok(b.chunks_exact(8)
        .map(|v| i64::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn bf(b: &mut Buffer<bf16>, data: &[u8]) -> Result<()> {
    ensure!(b.bytes() == data.len(), "BF16 metadata size");
    b.upload(
        &data
            .chunks_exact(2)
            .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())))
            .collect::<Vec<_>>(),
    )
}
fn fp(b: &mut Buffer<f32>, data: &[u8]) -> Result<()> {
    ensure!(b.bytes() == data.len(), "FP32 metadata size");
    b.upload(
        &data
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>(),
    )
}
pub fn run(path: &Path) -> Result<Value> {
    let p: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    let cases: Cases = serde_json::from_slice(&std::fs::read(p.fixture.join("index.json"))?)?;
    ensure!(
        p.rank < 4
            && (cases.rows, cases.world, cases.layer, cases.frequency_rows) == (6, 4, 2, 1048576)
            && cases.cases.len() == 4
            && cases.slots.len() == 67,
        "layer2 trace geometry"
    );
    let cfg = ModelConfig::parse(&std::fs::read(&p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let plan = RankPlan::build(&cp, &cfg, p.rank, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let core_bytes = std::fs::read(&p.core)?;
    let core = cuda.module(&core_bytes)?;
    let mut weights = Weights::load_staged(&cuda, &cp, &plan, &["layers.2."], &p.staged)?;
    let hash = weights.source_sha256.clone();
    let atlas_bytes = weights.bytes();
    let prepared = expert_packing::prepare_layer(&cuda, &core, &mut weights, 2)?;
    let weights = Rc::new(weights);
    let frequency = Rc::new(FrequencyTable::new(&cuda, &core, &stream, 1048576, true)?);
    let mut layer = SwaLayer::new_with_mode(
        &cuda,
        &core,
        &stream,
        weights.clone(),
        prepared,
        &p.kernels,
        frequency.clone(),
        InputMode::Split(p.rank),
    )?;
    let mut source = Source::new(
        &cuda,
        &stream,
        weights,
        Rc::new(Blas::new(&cuda)?),
        2,
        &p.kernels,
        frequency,
        &core_bytes,
        cases.kv_pages,
        cases.index_pages,
        &cases.slots,
        &[0; 6],
    )?;
    let mut cache = PagedKv::new(&cuda, 4, 64)?;
    let seed = std::fs::read(p.fixture.join("initial_cache.bin"))?;
    let mut input = cuda.allocate::<bf16>(6 * 20480)?;
    let mut previous = cuda.allocate::<f32>(6 * 4)?;
    let mut tokens = TokenRows::new(&cuda, &[0; 6], 6)?;
    let mut comm = Roce::local(&cuda, &p.kernels.join("roce-v0"), p.rank)?;
    comm.attach_gather(&p.kernels.join("roce-gather-v1"))?;
    let blobs = crate::rendezvous::exchange_blob(
        p.rank,
        p.endpoint.parse()?,
        &std::fs::read(&p.ticket)?,
        &comm.blob()?,
    )?;
    comm.connect(&blobs)?;
    let mut reports = Vec::new();
    for c in &cases.cases {
        ensure!(
            c.source.len() == 14 && c.global_taps.len() == 32 && c.ranks.len() == 4,
            "layer2 tap count"
        );
        let dir = p.fixture.join(&c.name);
        let data: BTreeMap<_, _> = c
            .source
            .iter()
            .map(|t| Ok((t.name.as_str(), fixture(&dir, t)?)))
            .collect::<Result<_>>()?;
        bf(&mut input, &data["input"])?;
        fp(&mut previous, &data["previous_pre"])?;
        tokens.update(&longs(&data["tokens"])?, ints(&data["live"])?[0] as usize)?;
        cache.upload(&seed)?;
        source
            .source_mut()
            .cache_mut()
            .upload(&data["seed_cache"])?;
        source.keys_mut().upload(&data["seed_keys"])?;
        fp(source.source_mut().ring_mut().unwrap(), &data["seed_ring"])?;
        bf(source.source_mut().latent_mut(), &data["seed_latent"])?;
        let pos = ints(&data["positions"])?;
        let loc = ints(&data["locations"])?;
        let positions = Positions::upload(&cuda, &pos, 1048576)?;
        let locations = RowLocations::upload(&cuda, &pos, &loc, cache.capacity(), 1048576)?;
        let candidates = Candidates::upload(
            &cuda,
            6,
            128,
            cache.capacity(),
            &ints(&data["ids"])?,
            &ints(&data["lengths"])?,
        )?;
        let source_lengths = ints(&data["source_lengths"])?;
        let meta = Metadata::new(
            &cuda,
            &source,
            &pos,
            &[0; 6],
            &longs(&data["source_raw"])?,
            &cases.slots,
            &source_lengths,
            1048576,
        )?;
        layer.run_indexed_from_previous(
            &input,
            &previous,
            &positions,
            &locations,
            &mut cache,
            &candidates,
            &tokens,
            &mut source,
            &meta,
            &comm,
            &stream,
        )?;
        stream.synchronize()?;
        comm.health()?;
        let graph = stream.capture(|| {
            layer.run_indexed_from_previous(
                &input,
                &previous,
                &positions,
                &locations,
                &mut cache,
                &candidates,
                &tokens,
                &mut source,
                &meta,
                &comm,
                &stream,
            )
        })?;
        for _ in 0..20 {
            unsafe { graph.launch(&stream) }?;
        }
        let a = cuda.event()?;
        let b = cuda.event()?;
        let mut times = Vec::new();
        for _ in 0..100 {
            a.record(&stream)?;
            unsafe { graph.launch(&stream) }?;
            b.record(&stream)?;
            times.push(b.elapsed_since(&a)?);
        }
        times.sort_by(f32::total_cmp);
        stream.synchronize()?;
        comm.health()?;
        ensure!(source.error()? == 0, "candidate producer bounds flag");
        let (local, mut global) = snapshots(&cuda, &layer, &cache)?;
        for (n, b) in
            crate::compressed_source_probe::snapshots(&source, cases.slots.len(), &source_lengths)?
        {
            global.insert(format!("source_{n}"), b);
        }
        let expected = c
            .ranks
            .iter()
            .find(|r| r.rank == p.rank)
            .ok_or_else(|| anyhow::anyhow!("rank trace absent"))?;
        let lt = compare(&dir.join(format!("rank{}", p.rank)), &expected.taps, &local)?;
        let gt = compare(&dir, &c.global_taps, &global)?;
        let exact = lt.iter().chain(&gt).all(|t| t["bitwise_equal"] == true);
        eprintln!(
            "{}",
            json!({"rank":p.rank,"case":c.name,"all_bitwise_equal":exact})
        );
        reports.push(json!({"case":c.name,"all_bitwise_equal":exact,"local_taps":lt,"global_taps":gt,"gpu_ms":{"p50":times[49],"p95":times[94],"p99":times[98],"max":times[99]}}));
    }
    Ok(
        json!({"rank":p.rank,"all_bitwise_equal":reports.iter().all(|r|r["all_bitwise_equal"]==true),"cases":reports,"atlas_bytes":atlas_bytes,"source_checkpoint_tp_sha256":hash,"scope":"Full TP4 layer2, canonical L1 input/pre and controlled past. Compression, index and physical attention live; not complete model or full prefill."}),
    )
}
