use crate::{
    blas::Blas,
    checkpoint::Checkpoint,
    config::ModelConfig,
    cuda::{Buffer, Cuda, DeviceElement},
    index::{KeyCache, Native},
    layer_probe::{Tap, compare, fixture},
    layout::RankPlan,
    prelude::FrequencyTable,
    query_index::{Branch, Metadata},
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
    raw: PathBuf,
}
#[derive(Deserialize)]
struct Cases {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    rows: usize,
    key_rows: usize,
    columns: usize,
    pages: usize,
    positions: Vec<i32>,
    slots: Vec<i32>,
    starts: Vec<i32>,
    phases: Vec<Phase>,
}
#[derive(Deserialize)]
struct Phase {
    name: String,
    lengths: Vec<i32>,
    sources: Vec<Tap>,
    expected: Vec<Tap>,
    selection: Tap,
}
fn bytes<T: DeviceElement>(b: &Buffer<T>) -> Result<Vec<u8>> {
    let x = b.download()?;
    ensure!(cfg!(target_endian = "little"), "diagnostic byte order");
    Ok(unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), b.bytes()) }.to_vec())
}
fn upload(b: &mut Buffer<bf16>, data: &[u8]) -> Result<()> {
    ensure!(data.len() == b.bytes(), "query branch input bytes");
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
    ensure!(cases.cases.len() == 3, "query branch coverage");
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
        &["layers.2.attn.indexer."],
    )?);
    let verified = weights.qualify(&cp, &p.model, &plan)?;
    let blas = Rc::new(Blas::new(&cuda)?);
    let native = Native::open(&cuda, &p.kernels.join("index-v1"))?;
    let stream = cuda.stream()?;
    let core = std::fs::read(&p.core)?;
    let module = cuda.module(&core)?;
    let freq = Rc::new(FrequencyTable::new(&cuda, &module, &stream, 1048576, true)?);
    std::fs::create_dir_all(&p.raw)?;
    let mut reports = Vec::new();
    for c in &cases.cases {
        ensure!(c.rows == 6, "query branch rows");
        let root = p.fixture.join(&c.name);
        let mut branch = Branch::new(
            &cuda,
            &stream,
            weights.clone(),
            blas.clone(),
            2,
            &p.kernels,
            freq.clone(),
            &core,
            c.key_rows,
            c.columns,
        )?;
        let mut cache = KeyCache::new(
            &cuda,
            native.clone(),
            weights.clone(),
            freq.clone(),
            2,
            6,
            c.pages,
        )?;
        let mut hidden = cuda.allocate::<bf16>(6 * 5120)?;
        let mut qlora = cuda.allocate::<bf16>(6 * 1280)?;
        let mut phases = Vec::new();
        for phase in &c.phases {
            let directory = root.join(&phase.name);
            for tap in &phase.sources {
                let data = fixture(&directory, tap)?;
                match tap.name.as_str() {
                    "hidden" => upload(&mut hidden, &data)?,
                    "q_lora" => upload(&mut qlora, &data)?,
                    "cache" => cache.upload(&data)?,
                    _ => anyhow::bail!("unknown query branch source"),
                }
            }
            let metadata = Metadata::new(
                &cuda,
                cache.capacity(),
                c.columns,
                &c.positions,
                &c.slots,
                &c.starts,
                &phase.lengths,
                freq.rows(),
            )?;
            branch.run(&hidden, &qlora, &cache, &metadata, &stream)?;
            stream.synchronize()?;
            let graph =
                stream.capture(|| branch.run(&hidden, &qlora, &cache, &metadata, &stream))?;
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
            let mut got = BTreeMap::from([
                ("q_projection".into(), bytes(branch.query_projection())?),
                ("head_projection".into(), bytes(branch.head_projection())?),
                (
                    "q_payload".into(),
                    branch.packed_query().payload().download()?,
                ),
                ("q_scales".into(), bytes(branch.packed_query().scales())?),
                ("q_weights".into(), bytes(branch.packed_query().weights())?),
                ("keys".into(), branch.gathered_keys().payload.download()?),
            ]);
            got.insert(
                "k_scales".into(),
                bytes(&branch.gathered_keys().scales)?[..c.key_rows * 4].to_vec(),
            );
            let scores = branch.scores()?.download()?;
            let mut valid = Vec::new();
            for (row, &length) in phase.lengths.iter().enumerate() {
                for x in &scores
                    [row * branch.score_stride()..row * branch.score_stride() + length as usize]
                {
                    valid.extend(x.to_le_bytes());
                }
            }
            got.insert("logits".into(), valid);
            let taps = compare(&directory, &phase.expected, &got)?;
            let raw = bytes(branch.selected())?;
            let file = p.raw.join(format!("{}-{}.bin", c.name, phase.name));
            std::fs::write(&file, &raw)?;
            let selection = compare(
                &directory,
                std::slice::from_ref(&phase.selection),
                &BTreeMap::from([("selected".into(), raw)]),
            )?;
            phases.push(json!({"name":phase.name,"taps":taps,"selection_order_diagnostic":selection,"raw_file":file,"gpu_ms":{"p50":times[49],"p95":times[94],"p99":times[98],"max":times[99]}}));
        }
        let exact = phases.iter().all(|p| {
            p["taps"]
                .as_array()
                .unwrap()
                .iter()
                .all(|t| t["bitwise_equal"] == true)
        });
        eprintln!(
            "{}",
            json!({"case":c.name,"all_numeric_taps_bitwise_equal":exact})
        );
        reports.push(json!({"case":c.name,"all_numeric_taps_bitwise_equal":exact,"phases":phases}));
    }
    Ok(
        json!({"all_numeric_taps_bitwise_equal":reports.iter().all(|r|r["all_numeric_taps_bitwise_equal"]==true),"cases":reports,"weights":verified,"scope":"Live FP8 Q/head projections, Q pack, paged key gather, bounded MQA and selection. Selection optimality evaluated separately; not compression/attention/full model."}),
    )
}
