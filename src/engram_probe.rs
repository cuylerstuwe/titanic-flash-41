//! Physical peer qualification of canonical Engram state and GPU dataflow.
use crate::{
    checkpoint::Checkpoint,
    config::ModelConfig,
    cuda::Cuda,
    engram::{PackedShard, RowStore},
    engram_gpu::{Layer, Rows},
    ffn::TokenRows,
    layer_probe::{Tap, bf16_bytes, compare, fixture},
    layout::RankPlan,
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
    time::Instant,
};
#[derive(Deserialize)]
struct Peer {
    config: PathBuf,
    checkpoint: PathBuf,
    rank: usize,
    core: PathBuf,
    kernels: PathBuf,
    staged: PathBuf,
    backing: PathBuf,
    fixture: PathBuf,
    endpoint: String,
    ticket: PathBuf,
}
#[derive(Deserialize)]
struct Rank {
    rank: usize,
    taps: Vec<Tap>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    source: Vec<Tap>,
    global_taps: Vec<Tap>,
    ranks: Vec<Rank>,
}
#[derive(Deserialize)]
struct LayerCases {
    layer: usize,
    table_rows: u64,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Cases {
    rows: usize,
    layers: Vec<LayerCases>,
}
pub fn run(path: &Path) -> Result<Value> {
    let p: Peer = serde_json::from_slice(&std::fs::read(path)?)?;
    let cases: Cases = serde_json::from_slice(&std::fs::read(p.fixture.join("index.json"))?)?;
    ensure!(
        p.rank < 4 && cases.rows == 6 && cases.layers.len() == 2,
        "Engram fixture geometry"
    );
    let cfg = ModelConfig::parse(&std::fs::read(p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let plan = RankPlan::build(&cp, &cfg, p.rank, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let core = cuda.module(&std::fs::read(p.core)?)?;
    let mut comm = Roce::local(&cuda, &p.kernels.join("roce-v0"), p.rank)?;
    comm.attach_gather(&p.kernels.join("roce-gather-v1"))?;
    let all = crate::rendezvous::exchange_blob(
        p.rank,
        p.endpoint.parse()?,
        &std::fs::read(p.ticket)?,
        &comm.blob()?,
    )?;
    comm.connect(&all)?;
    let mut report = Vec::new();
    let mut all_equal = true;
    for lc in cases.layers {
        ensure!(
            [1, 14].contains(&lc.layer) && lc.cases.len() == 4,
            "Engram layer fixture"
        );
        let prefix = format!("layers.{}.engram.", lc.layer);
        let weights = Weights::load_staged(
            &cuda,
            &cp,
            &plan,
            &[&prefix],
            &p.staged.join(format!("l{}-r{}", lc.layer, p.rank)),
        )?;
        let weight_bytes = weights.bytes();
        let source_sha = weights.source_sha256.clone();
        let mut engine = Layer::new(
            &cuda,
            &core,
            &stream,
            Rc::new(weights),
            lc.layer,
            6,
            &p.kernels,
            None,
        )?;
        let store = RowStore::new(
            PackedShard::open(
                &p.backing
                    .join(format!("engram-l{}-r{}of4.bin", lc.layer, p.rank)),
                lc.layer as u64,
                p.rank,
                4,
                lc.table_rows,
            )?,
            2 * 1024 * 1024 * 1024,
        )?;
        let mut rows = Rows::new(&cuda, 6, &store)?;
        let mut input = cuda.allocate::<bf16>(6 * 20480)?;
        let mut tokens = TokenRows::new(&cuda, &[0; 6], 6)?;
        let mut graph = None;
        let mut layer_reports = Vec::new();
        for case in lc.cases {
            ensure!(
                ["initial", "changed", "zero", "repeated"].contains(&case.name.as_str())
                    && case.source.len() == 3,
                "Engram case"
            );
            let dir = p
                .fixture
                .join(format!("layer{}", lc.layer))
                .join(&case.name);
            let source: BTreeMap<_, _> = case
                .source
                .iter()
                .map(|t| Ok((t.name.as_str(), fixture(&dir, t)?)))
                .collect::<Result<_>>()?;
            ensure!(
                source["input"].len() == input.bytes() && source["ids"].len() == 6 * 24 * 8,
                "Engram fixture sizes"
            );
            input.upload(
                &source["input"]
                    .chunks_exact(2)
                    .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
                    .collect::<Vec<_>>(),
            )?;
            let ids: Vec<[u64; 24]> = source["ids"]
                .chunks_exact(24 * 8)
                .map(|r| {
                    r.chunks_exact(8)
                        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect();
            ensure!(source["tokens"].len() == 48, "Engram token fixture bytes");
            tokens.update(
                &source["tokens"]
                    .chunks_exact(8)
                    .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>(),
                6,
            )?;
            let a = Instant::now();
            rows.prepare(&ids, &store, 8)?;
            let first_ms = a.elapsed().as_secs_f64() * 1000.;
            let first_stats = store.stats();
            let a = Instant::now();
            rows.prepare(&ids, &store, 8)?;
            let repeat_ms = a.elapsed().as_secs_f64() * 1000.;
            if graph.is_none() {
                engine.run(&input, &mut rows, &tokens, &comm, &stream)?;
                stream.synchronize()?;
                graph = Some(
                    stream.capture(|| engine.run(&input, &mut rows, &tokens, &comm, &stream))?,
                );
            }
            for _ in 0..100 {
                unsafe { graph.as_ref().unwrap().launch(&stream) }?;
            }
            stream.synchronize()?;
            comm.health()?;
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
            let expected = case
                .ranks
                .iter()
                .find(|r| r.rank == p.rank)
                .ok_or_else(|| anyhow::anyhow!("missing Engram rank trace"))?;
            let local = compare(
                &dir.join(format!("rank{}", p.rank)),
                &expected.taps,
                &BTreeMap::from([
                    ("local_embedding".into(), bf16_bytes(rows.local())?),
                    (
                        "local_projection".into(),
                        bf16_bytes(engine.projected_local())?,
                    ),
                ]),
            )?;
            let shared = compare(
                &dir,
                &case.global_taps,
                &BTreeMap::from([
                    ("embedding".into(), bf16_bytes(engine.embedding())?),
                    ("projected".into(), bf16_bytes(engine.projected())?),
                    ("output".into(), bf16_bytes(engine.output())?),
                ]),
            )?;
            let different: usize = local
                .iter()
                .chain(&shared)
                .map(|x| x["different_values"].as_u64().unwrap() as usize)
                .sum();
            all_equal &= different == 0;
            eprintln!(
                "rank {} layer {} {}: different={different}",
                p.rank, lc.layer, case.name
            );
            layer_reports.push(json!({"case":case.name,"different_values":different,"local_taps":local,"global_taps":shared,
                "row_preparation_ms":{"first":first_ms,"repeat":repeat_ms},"first_cache_stats":first_stats,"repeat_cache_stats":store.stats(),
                "gpu_ms":{"p50":ms[49],"p95":ms[94],"p99":ms[98],"max":ms[99]},"samples_ms":ms}));
        }
        report.push(json!({"layer":lc.layer,"cases":layer_reports,"weight_bytes":weight_bytes,
            "engram_staging":rows.input_kind(),
            "source_tp_bytes_sha256":source_sha,"row_cache_payload_and_metadata_bytes":store.resident_bytes,"staging_bytes":rows.staging_bytes()}));
        stream.synchronize()?;
        comm.health()?;
    }
    Ok(
        json!({"rank":p.rank,"all_bitwise_equal":all_equal,"layers":report,
        "scope":"Two Engram components, canonical raw row I/O, bounded cache, physical TP4 graph; not full model or session qualification"}),
    )
}
