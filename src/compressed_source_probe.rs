use crate::{
    blas::Blas,
    checkpoint::Checkpoint,
    compressed_source::{Layer, Metadata},
    config::ModelConfig,
    cuda::{Buffer, Cuda, DeviceElement},
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
    kernels: PathBuf,
}
#[derive(Deserialize)]
struct Cases {
    rows: usize,
    layer: usize,
    positions: Vec<i32>,
    requests: Vec<i64>,
    slots: Vec<i32>,
    kv_pages: usize,
    index_pages: usize,
    phases: Vec<Phase>,
}
#[derive(Deserialize)]
struct Phase {
    name: String,
    raw_locations: Vec<i64>,
    lengths: Vec<i32>,
    seeds: Vec<Tap>,
    sources: Vec<Tap>,
    expected: Vec<Tap>,
}
fn bytes<T: DeviceElement>(b: &Buffer<T>) -> Result<Vec<u8>> {
    let x = b.download()?;
    ensure!(cfg!(target_endian = "little"), "diagnostic byte order");
    Ok(unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), b.bytes()) }.to_vec())
}
fn bf(b: &mut Buffer<bf16>, data: &[u8]) -> Result<()> {
    ensure!(data.len() == b.bytes(), "BF16 source bytes");
    b.upload(
        &data
            .chunks_exact(2)
            .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())))
            .collect::<Vec<_>>(),
    )
}
fn fp(b: &mut Buffer<f32>, data: &[u8]) -> Result<()> {
    ensure!(data.len() == b.bytes(), "FP32 source bytes");
    b.upload(
        &data
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>(),
    )
}
fn projection_bytes(layer: &Layer) -> Result<Vec<u8>> {
    match layer.projection() {
        crate::compression::Input::Ratio1(b) => bytes(b),
        crate::compression::Input::Ratio2(b) => bytes(b),
    }
}
pub(crate) fn snapshots(
    layer: &Layer,
    keys: usize,
    lengths: &[i32],
) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut got = BTreeMap::from([
        ("projection".into(), projection_bytes(layer)?),
        ("latent".into(), bytes(layer.source().latent())?),
        ("cache".into(), layer.cache().data().download()?),
        ("wk".into(), bytes(layer.wk())?),
        ("key_cache".into(), layer.keys().data().download()?),
    ]);
    if let Some(ring) = layer.source().ring() {
        got.insert("ring".into(), bytes(ring)?);
    }
    got.extend(query_snapshots(
        layer.query(),
        layer.candidates(),
        keys,
        lengths,
    )?);
    Ok(got)
}

pub(crate) fn query_snapshots(
    q: &crate::query_index::Branch,
    candidates: &crate::attention::Candidates,
    keys: usize,
    lengths: &[i32],
) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut got = BTreeMap::from([
        ("q_projection".into(), bytes(q.query_projection())?),
        ("head_projection".into(), bytes(q.head_projection())?),
        ("q_payload".into(), q.packed_query().payload().download()?),
        ("q_scales".into(), bytes(q.packed_query().scales())?),
        ("q_weights".into(), bytes(q.packed_query().weights())?),
        ("keys".into(), q.gathered_keys().payload.download()?),
        (
            "k_scales".into(),
            bytes(&q.gathered_keys().scales)?[..keys * 4].to_vec(),
        ),
        ("candidates".into(), bytes(candidates.ids())?),
        ("candidate_lengths".into(), bytes(candidates.lengths())?),
    ]);
    let scores = q.scores()?.download()?;
    let mut valid = Vec::new();
    for (row, &n) in lengths.iter().enumerate() {
        ensure!(n >= 0 && n as usize <= keys, "source trace score length");
        for x in &scores[row * q.score_stride()..row * q.score_stride() + n as usize] {
            valid.extend(x.to_le_bytes());
        }
    }
    got.insert("logits".into(), valid);
    Ok(got)
}
pub fn run(path: &Path) -> Result<Value> {
    let p: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    let cases: Cases = serde_json::from_slice(&std::fs::read(p.fixture.join("index.json"))?)?;
    ensure!(
        cases.rows == 6 && [2, 20].contains(&cases.layer) && cases.phases.len() == 4,
        "compressed branch coverage"
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
        &[
            &format!("layers.{}.attn.compressor.", cases.layer),
            &format!("layers.{}.attn.indexer.", cases.layer),
        ],
    )?);
    let verified = weights.qualify(&cp, &p.model, &plan)?;
    let blas = Rc::new(Blas::new(&cuda)?);
    let stream = cuda.stream()?;
    let core = std::fs::read(&p.core)?;
    let module = cuda.module(&core)?;
    let freq = Rc::new(FrequencyTable::new(&cuda, &module, &stream, 1048576, true)?);
    let first = &cases.phases[0];
    let mut layer = Layer::new(
        &cuda,
        &stream,
        weights,
        blas,
        cases.layer,
        &p.kernels,
        freq.clone(),
        &core,
        cases.kv_pages,
        cases.index_pages,
        &cases.slots,
        &first.lengths,
    )?;
    let mut hidden = cuda.allocate::<bf16>(6 * 5120)?;
    let mut qlora = cuda.allocate::<bf16>(6 * 1280)?;
    let mut phases = Vec::new();
    for phase in &cases.phases {
        let directory = p.fixture.join(&phase.name);
        for tap in &phase.seeds {
            let data = fixture(&directory, tap)?;
            match tap.name.as_str() {
                "seed_ring" => fp(layer.source_mut().ring_mut().unwrap(), &data)?,
                "seed_latent" => bf(layer.source_mut().latent_mut(), &data)?,
                "seed_cache" => layer.source_mut().cache_mut().upload(&data)?,
                "seed_keys" => layer.keys_mut().upload(&data)?,
                _ => anyhow::bail!("unknown compressed seed"),
            }
        }
        for tap in &phase.sources {
            let data = fixture(&directory, tap)?;
            match tap.name.as_str() {
                "hidden" => bf(&mut hidden, &data)?,
                "q_lora" => bf(&mut qlora, &data)?,
                _ => anyhow::bail!("unknown compressed input"),
            }
        }
        let meta = Metadata::new(
            &cuda,
            &layer,
            &cases.positions,
            &cases.requests,
            &phase.raw_locations,
            &cases.slots,
            &phase.lengths,
            freq.rows(),
        )?;
        layer.run(&hidden, &qlora, &meta, &stream)?;
        stream.synchronize()?;
        let graph = stream.capture(|| layer.run(&hidden, &qlora, &meta, &stream))?;
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
        ensure!(
            layer.error()? == 0,
            "physical candidate producer flagged invalid addresses"
        );
        let got = snapshots(&layer, cases.slots.len(), &phase.lengths)?;
        let taps = compare(&directory, &phase.expected, &got)?;
        phases.push(json!({"name":phase.name,"taps":taps,"gpu_ms":{"p50":times[49],"p95":times[94],"p99":times[98],"max":times[99]}}));
    }
    Ok(
        json!({"layer":cases.layer,"all_bitwise_equal":phases.iter().all(|p|p["taps"].as_array().unwrap().iter().all(|t|t["bitwise_equal"]==true)),"phases":phases,"weights":verified,"scope":"Live low-ratio source/compression/index/physical candidates, controlled past; actual attention and TP model not yet connected"}),
    )
}
