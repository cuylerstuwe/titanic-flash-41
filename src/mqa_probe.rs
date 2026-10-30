//! Standalone valid-score parity/performance for the native FP4 MQA plan.
use crate::{
    cuda::{Buffer, Cuda, Stream},
    index::QueryRows,
    layer_probe::{Tap, compare, fixture},
    mqa::{Chunked, Dense, KeyMatrix, Native, Ranges},
};
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    rc::Rc,
};
#[derive(Deserialize)]
struct Config {
    fixture: PathBuf,
    primitive: PathBuf,
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
    starts: Vec<i32>,
    lengths: Vec<i32>,
    sources: Vec<Tap>,
    phases: Vec<Phase>,
}
#[derive(Deserialize)]
struct Phase {
    name: String,
    sources: Vec<Tap>,
    expected: Tap,
}
fn i32s(b: &[u8]) -> Result<Vec<i32>> {
    ensure!(b.len() % 4 == 0, "int32 input alignment");
    Ok(b.chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn floats(target: &mut Buffer<f32>, b: &[u8]) -> Result<()> {
    ensure!(b.len() == target.bytes(), "FP32 input bytes");
    target.upload(
        &b.chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>(),
    )
}
enum Scorer {
    Whole(Dense),
    Chunks(Chunked),
}
impl Scorer {
    fn inputs_mut(&mut self) -> Result<(&mut QueryRows, &mut KeyMatrix)> {
        match self {
            Self::Whole(x) => x.inputs_mut(),
            Self::Chunks(x) => x.inputs_mut(),
        }
    }
    fn run(&mut self, ranges: &Ranges, stream: &Stream) -> Result<()> {
        match self {
            Self::Whole(x) => x.run(ranges, stream),
            Self::Chunks(x) => {
                for i in 0..x.len() {
                    x.run_chunk(i, ranges, stream)?;
                }
                Ok(())
            }
        }
    }
    fn score_bytes(&self) -> usize {
        match self {
            Self::Whole(x) => x.output().bytes(),
            Self::Chunks(x) => x.output().bytes(),
        }
    }
    fn chunks(&self) -> usize {
        match self {
            Self::Whole(_) => 1,
            Self::Chunks(x) => x.len(),
        }
    }
    fn valid(&mut self, ranges: &Ranges, stream: &Stream, lengths: &[i32]) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        match self {
            Self::Whole(x) => {
                let all = x.output().download()?;
                for (row, &n) in lengths.iter().enumerate() {
                    for v in &all[row * x.stride()..row * x.stride() + n as usize] {
                        result.extend(v.to_le_bytes());
                    }
                }
            }
            Self::Chunks(x) => {
                for i in 0..x.len() {
                    let rows = x.rows(i)?;
                    x.run_chunk(i, ranges, stream)?;
                    stream.synchronize()?;
                    let all = x.output().download()?;
                    for (local, row) in rows.enumerate() {
                        for v in
                            &all[local * x.stride()..local * x.stride() + lengths[row] as usize]
                        {
                            result.extend(v.to_le_bytes());
                        }
                    }
                }
            }
        }
        Ok(result)
    }
}
pub fn run(path: &Path) -> Result<Value> {
    run_mode(path, false)
}
pub fn run_chunked(path: &Path) -> Result<Value> {
    run_mode(path, true)
}
fn run_mode(path: &Path, chunked: bool) -> Result<Value> {
    let p: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    let cases: Cases = serde_json::from_slice(&std::fs::read(p.fixture.join("index.json"))?)?;
    ensure!(cases.cases.len() == 8, "MQA qualification coverage");
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let native = Native::open(&cuda, &p.primitive)?;
    let stream = cuda.stream()?;
    let mut reports = Vec::new();
    for c in &cases.cases {
        let root = p.fixture.join(&c.name);
        let queries = Rc::new(QueryRows::new(&cuda, c.rows)?);
        let keys = Rc::new(KeyMatrix::new(&cuda, c.key_rows)?);
        let chunk_rows = if c.rows >= 161 { 128 } else { 4 };
        let budget = if chunked {
            chunk_rows * c.columns.div_ceil(128) * 128 * 4
        } else {
            64 * 1024 * 1024
        };
        let mut dense = if chunked {
            Scorer::Chunks(Chunked::new(
                &cuda,
                native.clone(),
                queries,
                keys,
                c.columns,
                budget,
            )?)
        } else {
            Scorer::Whole(Dense::new(
                &cuda,
                native.clone(),
                queries,
                keys,
                c.columns,
                budget,
            )?)
        };
        ensure!(
            dense.score_bytes() <= budget,
            "score arena exceeded its byte budget"
        );
        let ranges = Ranges::new(&cuda, c.key_rows, c.columns, &c.starts, &c.lengths)?;
        {
            let (_, keys) = dense.inputs_mut()?;
            for s in &c.sources {
                let b = fixture(&root, s)?;
                match s.name.as_str() {
                    "keys" => keys.payload.upload(&b)?,
                    "k_scales" => {
                        let mut scales = i32s(&b)?;
                        ensure!(scales.len() == c.key_rows, "key scale source rows");
                        scales.resize(keys.scales.len(), 0);
                        keys.scales.upload(&scales)?;
                    }
                    _ => anyhow::bail!("unknown MQA keys"),
                }
            }
            keys.ready.set(true);
        }
        let mut graph = None;
        let mut phases = Vec::new();
        for phase in &c.phases {
            let directory = root.join(&phase.name);
            {
                let (q, _) = dense.inputs_mut()?;
                for s in &phase.sources {
                    let b = fixture(&directory, s)?;
                    match s.name.as_str() {
                        "q" => q.payload.upload(&b)?,
                        "q_scales" => q.scales.upload(&i32s(&b)?)?,
                        "weights" => floats(&mut q.weights, &b)?,
                        _ => anyhow::bail!("unknown MQA query input"),
                    }
                }
                q.ready.set(true);
            }
            if graph.is_none() {
                dense.run(&ranges, &stream)?;
                stream.synchronize()?;
                graph = Some(stream.capture(|| dense.run(&ranges, &stream))?);
            }
            for _ in 0..20 {
                unsafe { graph.as_ref().unwrap().launch(&stream) }?;
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
            let valid = dense.valid(&ranges, &stream, &c.lengths)?;
            let taps = compare(
                &directory,
                std::slice::from_ref(&phase.expected),
                &BTreeMap::from([("logits".into(), valid)]),
            )?;
            phases.push(json!({"name":phase.name,"taps":taps,"gpu_ms":{"p50":samples[49],"p95":samples[94],"p99":samples[98],"max":samples[99]}}));
        }
        let exact = phases.iter().all(|p| {
            p["taps"]
                .as_array()
                .unwrap()
                .iter()
                .all(|t| t["bitwise_equal"] == true)
        });
        eprintln!("{}", json!({"case":c.name,"all_bitwise_equal":exact}));
        reports.push(json!({"case":c.name,"all_bitwise_equal":exact,"rows":c.rows,"key_rows":c.key_rows,"columns":c.columns,"logits_buffer_bytes":dense.score_bytes(),"admitted_score_bytes":budget,"row_chunks":dense.chunks(),"phases":phases}));
    }
    Ok(
        json!({"all_bitwise_equal":reports.iter().all(|r|r["all_bitwise_equal"]==true),"cases":reports,"chunked":chunked,"scope":"Every valid FP4 weighted-ReLU MQA score; one bounded reusable score arena if chunked, not full prefill/topK/model parity. Diagnostic CPU readbacks excluded from timing."}),
    )
}
