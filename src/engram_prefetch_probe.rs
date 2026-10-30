//! Paired cache-only lookahead test. Identical numerical profiles and accepted
//! inputs; application-row-cache cold and warm cases are reported separately.
use crate::{
    cuda::{Cuda, Stream},
    dspark_generation::{compare_logits, compare_positions, teacher_forced_logits},
    roce::Roce,
    target::Target,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

fn counter(stats: &Value, name: &str) -> u64 {
    stats.as_array().unwrap().iter().map(|s| s[name].as_u64().unwrap()).sum()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    cuda: &Cuda,
    large: &mut Target,
    verify: &mut Target,
    wides: &mut [Target],
    comm: &Roce,
    stream: &Stream,
    prefix: &[u32],
    output: &Path,
    rank: usize,
) -> Result<Value> {
    let corpus: Value = serde_json::from_slice(include_bytes!("../tests/fixtures/target-agentic-repository-v1.json"))?;
    let tokens: Vec<u32> = serde_json::from_value(corpus["batches"][0].clone())?;
    let continuation: Value = serde_json::from_slice(include_bytes!("../tests/fixtures/fixed-verify-python-v1.json"))?;
    let continuation: Vec<u32> = serde_json::from_value(continuation["teacher_forced_tokens"].clone())?;
    ensure!(prefix.len() + 15800 + 48 <= large.context_capacity() && tokens.len() >= 15800
        && std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref() == Ok("1")
        && std::env::var("CANADA_SCRATCH_ENGRAM_PREFETCH").as_deref() != Ok("1"),
        "paired lookahead requires bounded io_uring, room, and explicit per-call policy");
    large.enable_cached_extend_floor()?;
    let mut records = Vec::new();
    let mut lineage = 60000;
    let mut initial = true;
    for (history_mode, cache_mode) in [("fresh", "application_cold"), ("fresh", "warm"),
        ("retained", "application_cold"), ("retained", "warm")] {
        let actual_prefix = if history_mode == "fresh" { &[][..] } else { prefix };
        for delta in [256, 1024, 4096, 8192, 15800] {
            let mut previous: [Option<Vec<Vec<f32>>>; 2] = [None, None];
            for request in 0..4 {
                let mut times = [0.; 2];
                let mut logits: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
                let mut forced: [Vec<Vec<f32>>; 2] = [Vec::new(), Vec::new()];
                let mut phases = [Value::Null, Value::Null];
                let mut incoming = [Value::Null, Value::Null];
                let mut counters = [Value::Null, Value::Null];
                for path in [request % 2, 1 - request % 2] {
                    if !initial {
                        lineage += 1;
                        verify.reset(lineage, comm, stream)?;
                        large.handoff_from(verify)?;
                    }
                    initial = false;
                    if cache_mode == "application_cold" {
                        large.clear_engram_cache_for_probe(stream)?;
                    }
                    // Both start from the same ordinary prefill procedure.
                    if !actual_prefix.is_empty() {
                        large.prefill_wide_with_prefetch(wides, actual_prefix, comm, stream, false)?;
                        incoming[path] = large.encoder_tail_digest()?.unwrap();
                    } else {
                        ensure!(large.position() == 0, "fresh prefetch probe retained history");
                        incoming[path] = json!({"position":0});
                    }
                    let before = large.engram_cache_stats();
                    let position = large.position();
                    let start = Instant::now();
                    large.prefill_wide_with_prefetch(wides, &tokens[..delta], comm, stream, path == 1)?;
                    let _anchor = large.greedy_token(stream)?;
                    times[path] = start.elapsed().as_secs_f64();
                    ensure!(large.position() == position + delta as u64, "prefetch changed accepted frontier");
                    phases[path] = serde_json::to_value(large.append_metrics()?)?;
                    logits[path] = large.last_prompt_logits()?;
                    let after = large.engram_cache_stats();
                    let calls = counter(&after, "prefetch_calls") - counter(&before, "prefetch_calls");
                    ensure!((path != 0 || calls == 0) && (path != 1 || delta <= 4096 || calls > 0),
                        "paired diagnostic did not exercise the selected lookahead policy");
                    counters[path] = json!({"before":before,"after":after,"prefetch_calls":calls});
                    verify.handoff_from(large)?;
                    forced[path] = teacher_forced_logits(cuda, verify, &continuation, 8, comm, stream)?;
                    ensure!(logits[path].iter().all(|x| x.is_finite())
                        && forced[path].iter().flatten().all(|x| x.is_finite()), "nonfinite paired logits");
                }
                let exact = forced[0].iter().zip(&forced[1]).all(|(a,b)|
                    a.iter().zip(b).all(|(x,y)| x.to_bits() == y.to_bits()));
                let repeat: Vec<_> = (0..2).map(|path| previous[path].as_ref()
                    .map(|old| compare_positions(old, &forced[path]))).collect();
                let record = json!({"history_mode":history_mode,"cache_mode":cache_mode,"prefix_tokens":actual_prefix.len(),"new_tokens":delta,
                    "request":request,"warmup":request==0,"seconds_control":times[0],"seconds_prefetch":times[1],
                    "phases":phases,"row_cache":counters,"incoming_encoder_digest_equal":incoming[0]==incoming[1],
                    "last_row_logits":compare_logits(&logits[0],&logits[1]),
                    "teacher_forced":compare_positions(&forced[0],&forced[1]),
                    "teacher_forced_bitwise_equal":exact,"repeat_teacher_forced":repeat});
                std::fs::write(output.join(format!("engram-prefetch-{history_mode}-{cache_mode}-d{delta}-r{request}.json")),
                    serde_json::to_vec_pretty(&record)?)?;
                previous = forced.map(Some);
                eprintln!("prefetch rank{rank} {history_mode} {cache_mode} prefix{} delta{delta} request{request}: control {:.3}s candidate {:.3}s",
                    actual_prefix.len(),times[0],times[1]);
                records.push(record);
            }
        }
    }
    let all_exact = records.iter().all(|r| r["incoming_encoder_digest_equal"]==true
        && r["last_row_logits"]["bitwise_equal"]==true && r["teacher_forced_bitwise_equal"]==true);
    Ok(json!({"rank":rank,"all_exact":all_exact,"records":records,"memory":cuda.admission_memory()?,
        "full_engine_acceptance":false,"scope":"Fresh and retained inputs use the same wide profiles with exact cache lookahead off/on. Application-cold invalidates raw immutable row-cache tags before the common prefix, not OS page cache or model state. Warm cases reuse the row cache; first paired request per size is warmup. Alternating paired order, complete extension time, 48 subsequent teacher-forced distributions and all raw cache counters. This is an initial signal, not p95/p99, HTTP or successful-outcome qualification."}))
}
