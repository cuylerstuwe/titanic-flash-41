//! GPU-fixture checks for the independent sampler, without CPU numeric oracle.
use crate::{
    cuda::Cuda,
    sampling::{Greedy, VOCAB},
};
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
#[derive(Deserialize)]
struct Case {
    name: String,
    rows: usize,
    logits: String,
    expected: Vec<i32>,
    invalid: bool,
}
#[derive(Deserialize)]
struct Fixture {
    core: PathBuf,
    root: PathBuf,
    cases: Vec<Case>,
}
pub fn run(path: &Path) -> Result<Value> {
    let f: Fixture = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!((1..=32).contains(&f.cases.len()), "greedy case count");
    let cuda = Cuda::open(0)?;
    let stream = cuda.stream()?;
    let core = std::fs::read(&f.core)?;
    let mut reports = Vec::new();
    for c in f.cases {
        ensure!(
            (1..=4096).contains(&c.rows) && c.expected.len() == c.rows,
            "greedy fixture rows"
        );
        let raw = std::fs::read(f.root.join(&c.logits))?;
        ensure!(
            raw.len() == c.rows * VOCAB * 4,
            "greedy fixture input bytes"
        );
        // Byte decoding/transfer only. No host arithmetic on model logits.
        let data = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let mut logits = cuda.allocate(data.len())?;
        logits.upload(&data)?;
        let mut sampler = Greedy::new(&cuda, &core, c.rows)?;
        sampler.run(&logits, &stream)?;
        let result = sampler.finish(&stream);
        let correct = if c.invalid {
            result.is_err()
        } else {
            result.as_ref().is_ok_and(|ids| *ids == c.expected)
        };
        reports.push(
            json!({"name":c.name,"rows":c.rows,"invalid_input":c.invalid,"correct":correct,
            "actual_ids":result.as_ref().ok(),"rejected":result.is_err()}),
        );
    }
    Ok(
        json!({"all_correct":reports.iter().all(|r|r["correct"]==true),"cases":reports,
        "scope":"Independent GPU greedy sampler only. Stochastic sampling, DSpark and complete generation acceptance remain open."}),
    )
}
