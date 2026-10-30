//! Canonical GPU index projection/packing oracle comparison.
use crate::{
    blas::Blas,
    checkpoint::Checkpoint,
    config::ModelConfig,
    cuda::{Buffer, Cuda, DeviceElement},
    index::{IndexProjection, KeyCache, Native, QueryPack, Schedule},
    layer_probe::{Tap, compare, fixture},
    layout::RankPlan,
    prelude::FrequencyTable,
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
    model: PathBuf,
    fixture: PathBuf,
    core: PathBuf,
    primitive: PathBuf,
}
#[derive(Deserialize)]
struct Cases {
    frequency_rows: usize,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    layer: usize,
    rows: usize,
    pages: usize,
    positions: Vec<i32>,
    locations: Vec<i64>,
    seed: Tap,
    phases: Vec<Phase>,
}
#[derive(Deserialize)]
struct Phase {
    name: String,
    sources: Vec<Tap>,
    expected: Vec<Tap>,
}
fn bytes<T: DeviceElement>(b: &Buffer<T>) -> Result<Vec<u8>> {
    let x = b.download()?;
    ensure!(cfg!(target_endian = "little"), "diagnostic byte order");
    Ok(unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), b.bytes()) }.to_vec())
}
fn upload(b: &mut Buffer<bf16>, data: &[u8]) -> Result<()> {
    ensure!(data.len() == b.bytes(), "index input byte count");
    b.upload(
        &data
            .chunks_exact(2)
            .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())))
            .collect::<Vec<_>>(),
    )
}
pub fn run(path: &Path) -> Result<Value> {
    let p: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    let cases: Cases = serde_json::from_slice(&std::fs::read(p.fixture.join("index.json"))?)?;
    ensure!(
        cases.cases.len() == 16 && cases.frequency_rows == 1048576,
        "index qualification coverage"
    );
    let cfg = ModelConfig::parse(&std::fs::read(p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let plan = RankPlan::build(&cp, &cfg, 0, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let weights = Rc::new(Weights::load_prefixes(
        &cuda,
        &p.model,
        &cp,
        &plan,
        &["layers.2.attn.indexer.", "layers.20.attn.indexer."],
    )?);
    let verified = weights.qualify(&cp, &p.model, &plan)?;
    let blas = Rc::new(Blas::new(&cuda)?);
    let native = Native::open(&cuda, &p.primitive)?;
    let stream = cuda.stream()?;
    let core = cuda.module(&std::fs::read(&p.core)?)?;
    let frequency = Rc::new(FrequencyTable::new(
        &cuda,
        &core,
        &stream,
        cases.frequency_rows,
        true,
    )?);
    let mut reports = Vec::new();
    for c in &cases.cases {
        let root = p.fixture.join(&c.name);
        let prefix = format!("layers.{}.attn.indexer", c.layer);
        let key = IndexProjection::new(
            &cuda,
            weights.clone(),
            blas.clone(),
            native.clone(),
            &format!("{prefix}.wk"),
            c.rows,
        )?;
        let heads = IndexProjection::new(
            &cuda,
            weights.clone(),
            blas.clone(),
            native.clone(),
            &format!("{prefix}.weights_proj"),
            c.rows,
        )?;
        let mut cache = KeyCache::new(
            &cuda,
            native.clone(),
            weights.clone(),
            frequency.clone(),
            c.layer,
            c.rows,
            c.pages,
        )?;
        cache.upload(&fixture(&root, &c.seed)?)?;
        let query = QueryPack::new(&cuda, native.clone(), frequency.clone(), c.rows)?;
        let schedule = Schedule::new(
            &cuda,
            cache.capacity(),
            frequency.rows(),
            &c.positions,
            &c.locations,
        )?;
        let mut latent = cuda.allocate::<bf16>(c.rows * 512)?;
        let mut hidden = cuda.allocate::<bf16>(c.rows * 5120)?;
        let mut q = cuda.allocate::<bf16>(c.rows * 4096)?;
        let mut graph = None;
        let mut phases = Vec::new();
        for phase in &c.phases {
            let directory = root.join(&phase.name);
            for tap in &phase.sources {
                let data = fixture(&directory, tap)?;
                match tap.name.as_str() {
                    "latent" => upload(&mut latent, &data)?,
                    "hidden" => upload(&mut hidden, &data)?,
                    "query" => upload(&mut q, &data)?,
                    _ => anyhow::bail!("unknown index source"),
                }
            }
            let run = || {
                key.run(&latent, &stream)?;
                heads.run(&hidden, &stream)?;
                cache.run(key.output(), &schedule, &stream)?;
                query.run(&q, heads.output(), &schedule, &stream)
            };
            if graph.is_none() {
                run()?;
                stream.synchronize()?;
                graph = Some(stream.capture(run)?);
            }
            for _ in 0..20 {
                unsafe { graph.as_ref().unwrap().launch(&stream) }?;
            }
            let a = cuda.event()?;
            let b = cuda.event()?;
            let mut times = Vec::new();
            for _ in 0..100 {
                a.record(&stream)?;
                unsafe { graph.as_ref().unwrap().launch(&stream) }?;
                b.record(&stream)?;
                times.push(b.elapsed_since(&a)?);
            }
            times.sort_by(f32::total_cmp);
            stream.synchronize()?;
            let got = BTreeMap::from([
                ("key_projection".into(), bytes(key.output())?),
                ("head_projection".into(), bytes(heads.output())?),
                ("cache".into(), cache.data().download()?),
                ("q_payload".into(), query.payload().download()?),
                ("q_scales".into(), bytes(query.scales())?),
                ("q_weights".into(), bytes(query.weights())?),
            ]);
            phases.push(json!({"name":phase.name,"taps":compare(&directory,&phase.expected,&got)?,"gpu_ms":{"p50":times[49],"p95":times[94],"p99":times[98],"max":times[99]}}));
        }
        let exact = phases.iter().all(|p| {
            p["taps"]
                .as_array()
                .unwrap()
                .iter()
                .all(|t| t["bitwise_equal"] == true)
        });
        eprintln!("{}", json!({"case":c.name,"all_bitwise_equal":exact}));
        reports.push(json!({"case":c.name,"all_bitwise_equal":exact,"phases":phases}));
    }
    Ok(
        json!({"all_bitwise_equal":reports.iter().all(|r|r["all_bitwise_equal"]==true),"cases":reports,"weights":verified,"scope":"Index projections/K cache/Q pack and weights only; not logits/topK/full model"}),
    )
}
