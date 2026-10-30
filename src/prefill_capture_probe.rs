//! First-use capture without rerunning the successful prefill. Each paired
//! trial also replays the resulting graph on a reset, identical history.
use crate::{
    cuda::{Cuda, Stream},
    dspark_generation::{compare_logits, compare_positions, teacher_forced_logits},
    roce::Roce,
    target::Target,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

fn fixture(bytes: &[u8], key: &str) -> Result<Vec<u32>> {
    let value: Value = serde_json::from_slice(bytes)?;
    Ok(serde_json::from_value(if key == "batches" { value[key][0].clone() } else { value[key].clone() })?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda: &Cuda, large: &mut Target, verify: &mut Target,
    wides: &mut [Target], comm: &Roce, stream: &Stream, output: &Path, rank: usize) -> Result<Value> {
    ensure!(large.context_capacity() == 1048576 && large.position() == 0
        && std::env::var("CANADA_SCRATCH_POST_HTTP_CAPTURE").as_deref() == Ok("1"),
        "capture screen needs the retired1M HTTP worker at an empty frontier");
    let prefix = fixture(include_bytes!("../tests/fixtures/target-agentic-python-v1.json"), "batches")?;
    let corpus = fixture(include_bytes!("../tests/fixtures/target-agentic-repository-v1.json"), "batches")?;
    let long = fixture(include_bytes!("../tests/fixtures/target-agentic-long15800-v1.json"), "batches")?;
    let forced = fixture(include_bytes!("../tests/fixtures/fixed-verify-python-v1.json"), "teacher_forced_tokens")?;
    ensure!(prefix.len() == 458 && corpus.len() >= 4096 && long.len() == 15800 && forced.len() >= 12,
        "capture screen fixture bounds");
    let cases: Vec<(&str, &[u32], usize)> = vec![
        ("fresh_small", &[], 44), ("fresh_full", &[], 128),
        ("fresh_wide", &[], 4096), ("retained_small", &prefix, 31),
        ("retained_long", &long, 127),
    ];
    let mut records = Vec::new();
    let mut first = true;
    let mut lineage = 120000;
    for (case, base, delta) in cases {
        for repeat in 0..3 {
            let mut times = [[0.; 2]; 2]; // path, then first-capture/replay
            let mut logits: Vec<Vec<f32>> = vec![Vec::new(); 4];
            let mut following: Vec<Vec<Vec<f32>>> = vec![Vec::new(); 4];
            let mut incoming = vec![Value::Null; 4];
            let mut outgoing = vec![Value::Null; 4];
            let mut phases = vec![Value::Null; 4];
            let mut hashes = vec![String::new(); 4];
            for path in [repeat % 2, 1 - repeat % 2] {
                for replay in 0..2 {
                    if !first {
                        lineage += 1;
                        verify.reset(lineage, comm, stream)?;
                        large.handoff_from(verify)?;
                    }
                    first = false;
                    // Establish the accepted prefix eagerly. It must neither
                    // use nor replace the graph being checked for the delta.
                    if !base.is_empty() {
                        large.prefill_wide_with_policy(wides, base, comm, stream, false, false)?;
                    }
                    if replay == 0 { large.set_prefill_capture_once_for_probe(path == 1, stream)?; }
                    let slot = 2 * path + replay;
                    incoming[slot] = json!({"position":large.position(),"encoder":large.encoder_tail_digest()?,"carry":large.carry_digests()?});
                    let started = Instant::now();
                    large.prefill_wide_with_policy(wides, &corpus[..delta], comm, stream, false, true)?;
                    let anchor = large.greedy_token(stream)?;
                    times[path][replay] = started.elapsed().as_secs_f64();
                    ensure!(large.position() as usize == base.len() + delta, "capture changed accepted frontier");
                    phases[slot] = serde_json::to_value(large.append_metrics()?)?;
                    ensure!(phases[slot]["captured_new_graph"] == (replay == 0), "capture/replay trial did not exercise the intended graph path");
                    logits[slot] = large.last_prompt_logits()?;
                    outgoing[slot] = json!({"position":large.position(),"anchor":anchor,"encoder":large.encoder_tail_digest()?,"carry":large.carry_digests()?});
                    verify.handoff_from(large)?;
                    following[slot] = teacher_forced_logits(cuda, verify, &forced[..12], 2, comm, stream)?;
                    outgoing[slot]["following"] = json!({"position":verify.position(),"encoder":verify.encoder_tail_digest()?,"carry":verify.carry_digests()?});
                    let mut hash = Sha256::new();
                    for value in logits[slot].iter().chain(following[slot].iter().flatten()) {
                        ensure!(value.is_finite(), "nonfinite capture screen logit");
                        hash.update(value.to_le_bytes());
                    }
                    hashes[slot] = format!("{:x}", hash.finalize());
                }
            }
            ensure!(incoming.iter().all(|v| v == &incoming[0]), "capture screen incoming state differs");
            let comparisons: Vec<_> = (1..4).map(|slot| json!({"slot":slot,
                "last_row_logits":compare_logits(&logits[0], &logits[slot]),
                "following":compare_positions(&following[0], &following[slot])})).collect();
            let record = json!({"case":case,"prefix_tokens":base.len(),"new_tokens":delta,
                "repeat":repeat,"warmup":repeat==0,"seconds":times,"phases":phases,
                "incoming_state_equal":true,"outgoing_state_equal":outgoing.iter().all(|v| v == &outgoing[0]),
                "logit_hashes":hashes,"comparisons":comparisons});
            std::fs::write(output.join(format!("prefill-capture-{case}-r{repeat}.json")),serde_json::to_vec_pretty(&record)?)?;
            eprintln!("capture rank{rank} {case} repeat{repeat}: first {:?}, once {:?}", times[0], times[1]);
            records.push(record);
        }
    }
    Ok(json!({"rank":rank,"records":records,"context_capacity":large.context_capacity(),
        "reused_after_http":true,"memory":cuda.admission_memory()?,"full_engine_acceptance":false,
        "scope":"15 paired first-capture trials, each with a repeated graph replay on an identical reset prefix. Five cases, one vendor warmup plus two measured pairs; full FP32 prompt-row logits,12 following distributions and retained state. Both paths record a new graph; candidate uses the successful eager result without replaying it immediately. Canonical target precision unchanged. First graph use is distinct from process startup; no broad outcome/production claim."}))
}
