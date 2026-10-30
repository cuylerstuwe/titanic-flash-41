//! Canonical two-consumer shared-scratch GPU qualification for medium WO-A.
use crate::{
    checkpoint::Checkpoint,
    config::ModelConfig,
    cuda::Cuda,
    layout::RankPlan,
    weights::Weights,
    woa_medium::{Grouped, Scratch},
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    rc::Rc,
};
#[derive(Deserialize)]
struct Case {
    rows: usize,
    name: String,
}
#[derive(Deserialize)]
struct Fixture {
    config: PathBuf,
    checkpoint: PathBuf,
    model: PathBuf,
    core: PathBuf,
    root: PathBuf,
    cases: Vec<Case>,
}
fn bytes(b: &crate::cuda::Buffer<bf16>) -> Result<Vec<u8>> {
    Ok(b.download()?
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect())
}
pub fn run(path: &Path) -> Result<Value> {
    let f: Fixture = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(
        (1..=16).contains(&f.cases.len()),
        "medium WO-A fixture cases"
    );
    let config = ModelConfig::parse(&std::fs::read(f.config)?)?;
    let cp = Checkpoint::from_manifest(&f.checkpoint)?;
    let plan = RankPlan::build(&cp, &config, 0, 4)?;
    let cuda = Cuda::open(0)?;
    let stream = cuda.stream()?;
    let core = std::fs::read(f.core)?;
    let weights = Rc::new(Weights::load_prefixes(
        &cuda,
        &f.model,
        &cp,
        &plan,
        &["layers.0.attn.wo_a."],
    )?);
    let source = weights.qualify(&cp, &f.model, &plan)?;
    let mut reports = Vec::new();
    for case in f.cases {
        ensure!(
            [9, 64, 128, 384].contains(&case.rows),
            "medium WO-A fixture rows"
        );
        let root = f.root.join(&case.name);
        let scratch = Scratch::new(&cuda, &core, case.rows)?;
        let scratch_bytes = scratch.borrow().owned_scratch_bytes();
        let mut a = Grouped::new(&cuda, weights.clone(), 0, scratch.clone())?;
        let mut b = Grouped::new(&cuda, weights.clone(), 0, scratch)?;
        let mut inputs = Vec::new();
        let mut expected = Vec::new();
        for name in ["a", "b"] {
            let raw = std::fs::read(root.join(format!("input-{name}.bin")))?;
            ensure!(
                raw.len() == case.rows * 8192 * 2,
                "medium WO-A input byte count"
            );
            let data = raw
                .chunks_exact(2)
                .map(|x| bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())))
                .collect::<Vec<_>>();
            let mut buffer = cuda.allocate(data.len())?;
            buffer.upload(&data)?;
            inputs.push(buffer);
            let raw = std::fs::read(root.join(format!("expected-{name}.bin")))?;
            ensure!(
                raw.len() == case.rows * 2048 * 2,
                "medium WO-A expected byte count"
            );
            expected.push(raw);
        }
        a.run(&inputs[0], &stream)?;
        b.run(&inputs[1], &stream)?;
        stream.synchronize()?;
        let eager = bytes(a.output())? == expected[0] && bytes(b.output())? == expected[1];
        let graph = stream.capture(|| {
            a.run(&inputs[0], &stream)?;
            b.run(&inputs[1], &stream)
        })?;
        for _ in 0..100 {
            unsafe {
                graph.launch(&stream)?;
            }
        }
        stream.synchronize()?;
        let captured = bytes(a.output())? == expected[0] && bytes(b.output())? == expected[1];
        let other = cuda.stream()?;
        let wrong_stream_rejected = a.run(&inputs[0], &other).is_err();
        reports.push(json!({"name":case.name,"rows":case.rows,"two_consumers_share_scratch":true,
            "eager_bitwise_equal":eager,"captured_bitwise_equal":captured,
            "wrong_stream_rejected":wrong_stream_rejected,"replays":100,"owned_shared_scratch_bytes":scratch_bytes}));
    }
    Ok(
        json!({"all_correct":reports.iter().all(|r|r["eager_bitwise_equal"]==true && r["captured_bitwise_equal"]==true && r["wrong_stream_rejected"]==true),
        "cases":reports,"source":source,"scope":"Canonical medium-prefill WO-A component and shared scratch only. No full wider model or performance/memory/reliability acceptance."}),
    )
}
