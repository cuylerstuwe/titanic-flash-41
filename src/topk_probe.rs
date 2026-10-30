use crate::{
    cuda::{Buffer, Cuda},
    layer_probe::{Tap, compare, fixture},
    topk::{Native, Selection, Windows},
};
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
#[derive(Deserialize)]
struct Config {
    fixture: PathBuf,
    primitive: PathBuf,
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
    width: usize,
    topk: usize,
    starts: Vec<i32>,
    lengths: Vec<i32>,
    offsets: Vec<i32>,
    phases: Vec<Phase>,
}
#[derive(Deserialize)]
struct Phase {
    name: String,
    source: Tap,
    expected: Tap,
    reference_repeat_stable: bool,
}
fn upload(out: &mut Buffer<f32>, b: &[u8]) -> Result<()> {
    ensure!(out.bytes() == b.len(), "topk source bytes");
    out.upload(
        &b.chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>(),
    )
}
pub fn run(path: &Path) -> Result<Value> {
    let p: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    let cases: Cases = serde_json::from_slice(&std::fs::read(p.fixture.join("index.json"))?)?;
    ensure!(cases.cases.len() == 7, "topk qualification coverage");
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let native = Native::open(&cuda, &p.primitive)?;
    let stream = cuda.stream()?;
    std::fs::create_dir_all(&p.raw)?;
    let mut reports = Vec::new();
    for c in &cases.cases {
        let root = p.fixture.join(&c.name);
        let windows = Windows::new(&cuda, c.width, &c.starts, &c.lengths, &c.offsets)?;
        let selection = Selection::new(&cuda, native.clone(), c.rows, c.width, c.width, c.topk)?;
        let mut scores = cuda.allocate::<f32>(c.rows * c.width)?;
        let mut graph = None;
        let mut phases = Vec::new();
        for phase in &c.phases {
            let directory = root.join(&phase.name);
            upload(&mut scores, &fixture(&directory, &phase.source)?)?;
            if graph.is_none() {
                selection.run(&mut scores, &windows, &stream)?;
                stream.synchronize()?;
                graph = Some(stream.capture(|| selection.run(&mut scores, &windows, &stream))?);
            }
            let a = cuda.event()?;
            let b = cuda.event()?;
            let mut samples = Vec::new();
            for _ in 0..100 {
                a.record(&stream)?;
                unsafe { graph.as_ref().unwrap().launch(&stream) }?;
                b.record(&stream)?;
                samples.push(b.elapsed_since(&a)?);
            }
            samples.sort_by(f32::total_cmp);
            stream.synchronize()?;
            let ids = selection.indices().download()?;
            // Integer range/cardinality validation only. Score optimality is
            // checked separately against torch.topk on GPU, because the
            // reference primitive's index order/tie boundary are not stable.
            let metadata_valid = ids.chunks_exact(c.topk).enumerate().all(|(row, ids)| {
                let mut unique = std::collections::BTreeSet::new();
                ids.iter().all(|&id| {
                    id == -1
                        || (id >= c.offsets[row]
                            && (id as i64) < c.offsets[row] as i64 + c.lengths[row] as i64
                            && unique.insert(id))
                }) && unique.len() == c.topk.min(c.lengths[row] as usize)
            });
            let raw: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
            let basename = format!("{}-{}.bin", c.name, phase.name);
            ensure!(
                basename
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || [b'-', b'_', b'.'].contains(&b)),
                "topk diagnostic basename"
            );
            let raw_file = p.raw.join(basename);
            std::fs::write(&raw_file, &raw)?;
            let taps = compare(
                &directory,
                std::slice::from_ref(&phase.expected),
                &BTreeMap::from([("selected".into(), raw)]),
            )?;
            phases.push(json!({"name":phase.name,"taps":taps,"metadata_valid":metadata_valid,"raw_file":raw_file,"reference_repeat_order_stable":phase.reference_repeat_stable,"gpu_ms":{"p50":samples[49],"p95":samples[94],"p99":samples[98],"max":samples[99]}}));
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
        json!({"all_bitwise_equal":reports.iter().all(|r|r["all_bitwise_equal"]==true),"all_metadata_valid":reports.iter().all(|r|r["phases"].as_array().unwrap().iter().all(|p|p["metadata_valid"]==true)),"cases":reports,"scope":"Integer admission and diagnostic outputs; score optimality needs the independent GPU evaluator; reference order/tie selection may be nondeterministic"}),
    )
}
