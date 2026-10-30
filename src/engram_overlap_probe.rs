//! Interleaved Verify1..6 control versus exact row-I/O/layer0 overlap.
use crate::{
    cuda::{Cuda, Stream},
    roce::Roce,
    target::Target,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

const WARMUP: usize = 4;
const MEASURED: usize = 32;

fn distribution(values: &[f64]) -> Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let at = |n: usize| sorted[(sorted.len() * n).div_ceil(100) - 1];
    json!({"samples":values.len(),"p50":at(50),"p95":at(95),"p99":at(99),
        "min":sorted[0],"max":sorted[sorted.len()-1],"samples_ms_in_order":values})
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
    let bytes = include_bytes!("../tests/fixtures/fixed-verify-python-v1.json");
    let fixture: Value = serde_json::from_slice(bytes)?;
    let continuation: Vec<u32> = serde_json::from_value(fixture["teacher_forced_tokens"].clone())?;
    ensure!(
        [458, 15800].contains(&prefix.len())
            && continuation.len() == 512
            && large.context_capacity() == 32768
            && std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref() == Ok("1")
            && std::env::var_os("CANADA_SCRATCH_ENGRAM_PREFETCH").is_none()
            && std::env::var_os("CANADA_SCRATCH_ENGRAM_PREFETCH_AB").is_none(),
        "Engram overlap probe requires the isolated fixed 32K io_uring profile"
    );
    large.enable_cached_extend_floor()?;
    let begin = Instant::now();
    large.prefill_wide(wides, prefix, comm, stream)?;
    let prefill_seconds = begin.elapsed().as_secs_f64();
    verify.handoff_from(large)?;
    let mut records = Vec::new();
    for offset in [54, 182, 310, 438] {
        let frontier = prefix.len() + offset;
        verify.set_engram_overlap_for_probe(false)?;
        while (verify.position() as usize) < frontier {
            let from = verify.position() as usize - prefix.len();
            verify.append(&continuation[from..(from + 6).min(offset)], comm, stream)?;
        }
        for mode in ["application_cold", "warm"] {
            for live in 1..=6 {
                let block = &continuation[offset..offset + live];
                let carry = verify.carry_digests()?;
                let encoder = verify.encoder_tail_digest()?;
                let mut samples: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
                let mut host: [Vec<[f64; 2]>; 2] = [Vec::new(), Vec::new()];
                let mut logits: [Vec<String>; 2] = [Vec::new(), Vec::new()];
                let mut predictions: [Vec<Vec<u32>>; 2] = [Vec::new(), Vec::new()];
                let mut stages = [Value::Null, Value::Null];
                let mut exercised = 0usize;
                for trial in 0..WARMUP + MEASURED {
                    let measured = trial >= WARMUP;
                    let hash = trial == WARMUP - 1 || trial == WARMUP + MEASURED - 1;
                    for path in [trial % 2, 1 - trial % 2] {
                        verify.set_engram_overlap_for_probe(path == 1)?;
                        if mode == "application_cold" {
                            verify.clear_engram_cache_for_probe(stream)?;
                        }
                        let start = Instant::now();
                        verify.begin_verification(block, comm, stream)?;
                        let ids = verify.verification_greedy(stream)?;
                        let ms = start.elapsed().as_secs_f64() * 1000.;
                        let host_ms = verify.verification_host_ms()?;
                        if hash {
                            let (values, rows) = verify.pending_logits()?;
                            ensure!(rows == live, "overlap live-logit geometry");
                            let raw = unsafe { cuda.read_device(values.ptr(), live * 129280 * 4) }?;
                            logits[path].push(format!("{:x}", Sha256::digest(raw)));
                            stages[path] = serde_json::to_value(verify.verification_metrics()?)?;
                        }
                        if measured {
                            if path == 1 {
                                let metrics = serde_json::to_value(verify.verification_metrics()?)?;
                                ensure!(
                                    metrics["stages_ms"]
                                        .as_array()
                                        .unwrap()
                                        .iter()
                                        .any(|stage| stage[0] == "engram_upload_order"),
                                    "measured candidate did not defer row I/O across layer0"
                                );
                                exercised += 1;
                            }
                            samples[path].push(ms);
                            host[path].push(host_ms);
                            predictions[path].push(ids);
                        }
                        verify.finish_verification(0, comm, stream)?;
                        ensure!(
                            verify.position() as usize == frontier,
                            "zero-acceptance overlap trial advanced the accepted frontier"
                        );
                    }
                }
                ensure!(
                    carry == verify.carry_digests()? && encoder == verify.encoder_tail_digest()?,
                    "overlap trial changed accepted carry/encoder bytes"
                );
                let paired: Vec<_> = samples[0]
                    .iter()
                    .zip(&samples[1])
                    .map(|(a, b)| b - a)
                    .collect();
                let record = json!({"frontier":frontier,"cache_mode":mode,"live_rows":live,
                    "control_ms":distribution(&samples[0]),"overlap_ms":distribution(&samples[1]),
                    "paired_overlap_minus_control_ms":distribution(&paired),"host_ms":host,
                    "sample_stages":stages,"deferred_row_iterations":exercised,
                    "logits_sha256_first_last":logits,"logits_byte_identical":logits[0]==logits[1],
                    "repeat_logits_byte_identical":logits.iter().all(|v|v[0]==v[1]),
                    "predictions_equal":predictions[0]==predictions[1],
                    "accepted_carry_and_encoder_exact":true});
                std::fs::write(
                    output.join(format!("engram-overlap-f{frontier}-{mode}-m{live}.json")),
                    serde_json::to_vec_pretty(&record)?,
                )?;
                eprintln!(
                    "Engram overlap rank{rank} frontier{frontier} {mode} M{live}: control {}ms overlap {}ms exact {}",
                    record["control_ms"]["p50"],
                    record["overlap_ms"]["p50"],
                    record["logits_byte_identical"]
                );
                records.push(record);
            }
        }
    }
    verify.set_engram_overlap_for_probe(false)?;
    let all_exact = records.iter().all(|r| {
        r["logits_byte_identical"] == true
            && r["repeat_logits_byte_identical"] == true
            && r["predictions_equal"] == true
    });
    Ok(
        json!({"rank":rank,"prefix_tokens":prefix.len(),"context_capacity":32768,
        "fixture_sha256":format!("{:x}",Sha256::digest(bytes)),"prefill_seconds":prefill_seconds,
        "all_exact":all_exact,"records":records,"full_engine_acceptance":false,
        "scope":"Same resident Verify1..6 owner and complete incoming accepted state; alternate whole-graph and two-graph schedules with identical row bytes and arithmetic. Four warmups and32 measured pairs per frontier/row count/application-cache mode. Zero-acceptance carry/encoder rollback is exact. OS cache is uncontrolled; this is a verifier latency screen, not natural generation or production tail-latency qualification."}),
    )
}
