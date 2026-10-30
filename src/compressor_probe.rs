//! GPU vendor-projection qualification for the canonical compression sources.
use crate::{
    bf16_linear::{Output, Projection},
    blas::Blas,
    checkpoint::Checkpoint,
    config::ModelConfig,
    cuda::Cuda,
    layer_probe::{Tap, compare, fixture},
    layout::RankPlan,
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
struct Input {
    config: PathBuf,
    checkpoint: PathBuf,
    model: PathBuf,
    fixture: PathBuf,
}
#[derive(Deserialize)]
struct Phase {
    name: String,
    source: Tap,
    expected: Tap,
}
#[derive(Deserialize)]
struct Case {
    layer: usize,
    rows: usize,
    n: usize,
    k: usize,
    output_type: String,
    phases: Vec<Phase>,
}
#[derive(Deserialize)]
struct Cases {
    cases: Vec<Case>,
}
fn phase<T: Output>(
    cuda: &Cuda,
    weights: Rc<Weights>,
    blas: Rc<Blas>,
    root: &Path,
    c: &Case,
    pair: bool,
    stream: &crate::cuda::Stream,
) -> Result<Value> {
    let projection = if pair {
        Projection::<T>::compressor_pair(cuda, weights, blas, c.layer, c.rows)?
    } else {
        Projection::<T>::new(
            cuda,
            weights,
            blas,
            &format!("layers.{}.attn.compressor.wkv", c.layer),
            c.rows,
        )?
    };
    let mut input = cuda.allocate::<bf16>(c.rows * c.k)?;
    let mut graph = None;
    let mut reports = Vec::new();
    let mut exact = true;
    for p in &c.phases {
        let dir = root.join(&p.name);
        let source = fixture(&dir, &p.source)?;
        ensure!(source.len() == input.bytes(), "compressor input byte count");
        input.upload(
            &source
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
                .collect::<Vec<_>>(),
        )?;
        if graph.is_none() {
            projection.run(&input, &stream)?;
            stream.synchronize()?;
            graph = Some(stream.capture(|| projection.run(&input, &stream))?);
        }
        for _ in 0..20 {
            unsafe { graph.as_ref().unwrap().launch(&stream) }?;
        }
        let a = cuda.event()?;
        let b = cuda.event()?;
        a.record(&stream)?;
        for _ in 0..100 {
            unsafe { graph.as_ref().unwrap().launch(&stream) }?;
        }
        b.record(&stream)?;
        let ms = b.elapsed_since(&a)? / 100.;
        stream.synchronize()?;
        let output = projection.output().download()?;
        // Output is a sealed POD BF16/FP32 type, and CUDA/host are little endian.
        ensure!(
            cfg!(target_endian = "little"),
            "unsupported diagnostic byte order"
        );
        let bytes = unsafe {
            std::slice::from_raw_parts(output.as_ptr().cast::<u8>(), projection.output().bytes())
        }
        .to_vec();
        let taps = compare(
            &dir,
            std::slice::from_ref(&p.expected),
            &BTreeMap::from([("output".into(), bytes)]),
        )?;
        exact &= taps.iter().all(|t| t["bitwise_equal"] == true);
        reports.push(json!({"phase":p.name,"graph_ms":ms,"taps":taps}));
    }
    Ok(
        json!({"layer":c.layer,"rows":c.rows,"n":c.n,"k":c.k,"output_type":c.output_type,"all_bitwise_equal":exact,"phases":reports}),
    )
}
pub fn run(path: &Path) -> Result<Value> {
    let p: Input = serde_json::from_slice(&std::fs::read(path)?)?;
    let cases: Cases = serde_json::from_slice(&std::fs::read(p.fixture.join("index.json"))?)?;
    ensure!(cases.cases.len() == 12, "compressor profile count");
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
        &["layers.2.attn.compressor.", "layers.20.attn.compressor."],
    )?);
    let verified = weights.qualify(&cp, &p.model, &plan)?;
    let blas = Rc::new(Blas::new(&cuda)?);
    let stream = cuda.stream()?;
    let mut reports = Vec::new();
    for c in &cases.cases {
        ensure!(
            [1, 6, 16, 161, 460, 4096].contains(&c.rows) && c.k == 5120 && c.phases.len() == 3,
            "compressor profile geometry"
        );
        let root = p.fixture.join(format!("l{}-m{}", c.layer, c.rows));
        let r = match (c.layer, c.n, c.output_type.as_str()) {
            (2, 1024, "fp32") => phase::<f32>(
                &cuda,
                weights.clone(),
                blas.clone(),
                &root,
                c,
                true,
                &stream,
            )?,
            (20, 512, "bf16") => phase::<bf16>(
                &cuda,
                weights.clone(),
                blas.clone(),
                &root,
                c,
                false,
                &stream,
            )?,
            _ => anyhow::bail!("unqualified compressor profile"),
        };
        eprintln!(
            "{}",
            json!({"layer":c.layer,"rows":c.rows,"exact":r["all_bitwise_equal"]})
        );
        reports.push(r);
    }
    Ok(
        json!({"all_bitwise_equal":reports.iter().all(|r|r["all_bitwise_equal"]==true),"cases":reports,"source":verified,
        "scope":"BF16 input projections only; FP32-output ratio2/ BF16-output ratio1, canonical weights and prefill rows; not compression state or full model"}),
    )
}
