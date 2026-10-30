//! GPU-reference checks of bounded DSpark history and separate query storage.
use crate::{
    checkpoint::Checkpoint,
    config::ModelConfig,
    cuda::{Buffer, Cuda},
    dspark_kv::{State, query_layout},
    layout::RankPlan,
    prelude::FrequencyTable,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};
#[derive(Deserialize)]
struct Operation {
    kind: String,
    input: Option<[PathBuf; 3]>,
    rows: usize,
    live: usize,
    accepted: usize,
    repeat: usize,
    frontier: u64,
    expected: [PathBuf; 3],
    layout: Option<PathBuf>,
}
#[derive(Deserialize)]
struct Input {
    config: PathBuf,
    checkpoint: PathBuf,
    model: PathBuf,
    core: PathBuf,
    kernels: PathBuf,
    frequency_rows: usize,
    operations: Vec<Operation>,
}
pub fn run(path: &Path) -> Result<serde_json::Value> {
    let p: Input = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(
        p.frequency_rows == 1048576 && !p.operations.is_empty() && p.operations.len() <= 128,
        "DSpark KV probe bound"
    );
    let cfg = ModelConfig::parse(&std::fs::read(p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let plan = RankPlan::build(&cp, &cfg, 0, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let foreign = cuda.stream()?;
    let core = cuda.module(&std::fs::read(p.core)?)?;
    let weights = Weights::load_prefixes(
        &cuda,
        &p.model,
        &cp,
        &plan,
        &[
            "mtp.0.attn.kv_norm.",
            "mtp.1.attn.kv_norm.",
            "mtp.2.attn.kv_norm.",
        ],
    )?;
    let frequency = Rc::new(FrequencyTable::new(
        &cuda,
        &core,
        &stream,
        p.frequency_rows,
        false,
    )?);
    let mut state = State::new(&cuda, &stream, &weights, &p.kernels, frequency.clone())?;
    let mut records = Vec::new();
    for (op_index, op) in p.operations.into_iter().enumerate() {
        ensure!(
            op.repeat > 0 && op.repeat <= 8192,
            "DSpark KV repeated commit bound"
        );
        let mut buffers: Vec<Buffer<bf16>> = Vec::new();
        if let Some(inputs) = &op.input {
            for input in inputs {
                let bytes = std::fs::read(input)?;
                ensure!(bytes.len() == op.rows * 512 * 2, "DSpark raw KV input size");
                let words: Vec<_> = bytes
                    .chunks_exact(2)
                    .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                    .collect();
                let mut b = cuda.allocate::<bf16>(words.len())?;
                b.upload(&words)?;
                buffers.push(b);
            }
        }
        let before = Instant::now();
        match op.kind.as_str() {
            "reset" => {
                ensure!(op.repeat == 1, "reset repeat");
                state.reset(&stream)?;
            }
            "commit" => {
                ensure!(buffers.len() == 3, "commit inputs");
                let raw = [&buffers[0], &buffers[1], &buffers[2]];
                let pos = state.position()?;
                ensure!(
                    state
                        .commit(raw, op.live, op.live + 1, pos, &stream)
                        .is_err()
                        && state
                            .commit(raw, op.live, op.accepted, pos + 1, &stream)
                            .is_err()
                        && state
                            .commit(raw, op.live, op.accepted, pos, &foreign)
                            .is_err()
                        && state.position()? == pos,
                    "invalid DSpark commit accepted"
                );
                for _ in 0..op.repeat {
                    let pos = state.position()?;
                    state.commit(raw, op.live, op.accepted, pos, &stream)?;
                }
                ensure!(
                    state.attention_state(0, &stream).is_err(),
                    "commit left stale query state published"
                );
            }
            "query" => {
                ensure!(
                    buffers.len() == 3 && op.rows == 5 && op.repeat == 1,
                    "query inputs"
                );
                state.store_queries(
                    [&buffers[0], &buffers[1], &buffers[2]],
                    state.position()?,
                    &stream,
                )?;
                for stage in 0..3 {
                    state.attention_state(stage, &stream)?;
                }
            }
            "end-bound" => {
                ensure!(
                    buffers.len() == 3 && state.position()? == p.frequency_rows as u64,
                    "end-bound fixture"
                );
                ensure!(
                    state
                        .store_queries(
                            [&buffers[0], &buffers[1], &buffers[2]],
                            state.position()?,
                            &stream
                        )
                        .is_err()
                        && state.position()? == p.frequency_rows as u64,
                    "DSpark end-bound query was admitted or poisoned valid state"
                );
            }
            _ => anyhow::bail!("unknown DSpark KV probe operation"),
        }
        let elapsed_ms = before.elapsed().as_secs_f64() * 1000.;
        ensure!(
            state.position()? == op.frontier,
            "DSpark KV frontier mismatch"
        );
        let mut outputs = Vec::new();
        for stage in 0..3 {
            let actual = state.cache(stage)?.data().download()?;
            let expected = std::fs::read(&op.expected[stage])?;
            let different: Vec<_> = actual
                .iter()
                .zip(&expected)
                .enumerate()
                .filter(|(_, (a, e))| a != e)
                .map(
                    |(offset, (a, e))| serde_json::json!({"offset":offset,"actual":a,"expected":e}),
                )
                .collect();
            if actual != expected {
                std::fs::write(
                    path.parent()
                        .unwrap()
                        .join(format!("actual-op{op_index}-stage{stage}.bin")),
                    &actual,
                )?;
            }
            outputs.push(serde_json::json!({"stage":stage,"bytes":actual.len(),"exact":actual==expected,
                "different_bytes":different.len(),"first_differences":different.iter().take(32).collect::<Vec<_>>(),
                "actual_sha256":format!("{:x}",Sha256::digest(&actual)),"expected_sha256":format!("{:x}",Sha256::digest(&expected))}));
        }
        if let Some(path) = op.layout {
            let expected: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
            let layout = serde_json::to_value(query_layout(state.position()?, frequency.rows())?)?;
            ensure!(
                layout == expected,
                "DSpark bidirectional candidate/position metadata differs from reference"
            );
        }
        records.push(serde_json::json!({"kind":op.kind,"rows":op.rows,"live":op.live,"accepted":op.accepted,"repeat":op.repeat,
            "frontier":op.frontier,"wall_ms":elapsed_ms,"all_exact":outputs.iter().all(|o|o["exact"]==true),"outputs":outputs}));
    }
    Ok(
        serde_json::json!({"all_exact":records.iter().all(|v|v["all_exact"]==true),"records":records,
        "cache_bytes_per_stage":state.cache(0)?.data().bytes(),"stages":3,
        "rotary_table_rows":frequency.rows(),"rotary_table_bytes":frequency.values().bytes(),
        "full_engine_acceptance":false,"draft_model_implemented":false,
        "scope":"Bounded3stage draft KV injection/query slots and shared bidirectional5row metadata only, compared with GPU weighted-norm/RoPE/store reference. Fixed memory through ring rollover and1Mrotary range. No draft attention output/query network/Markov/generation or matched engine performance acceptance."}),
    )
}
