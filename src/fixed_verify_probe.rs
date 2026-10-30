//! Fixed accepted-prefix timing/layout diagnostic. All model work remains on
//! GPU. Repeating the same verifier block separates schedule latency from a
//! speculative drafter's changing acceptance rate and autoregressive path.
use crate::{
    cuda::{Cuda, Stream},
    roce::Roce,
    target::Target,
};
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

#[derive(Deserialize)]
struct Fixture {
    case: String,
    teacher_forced_tokens: Vec<u32>,
    frontiers: Vec<usize>,
    warmup_iterations: usize,
    measured_iterations: usize,
}
fn distribution(values: Vec<f64>) -> Value {
    let mut sorted = values.clone();
    sorted.sort_by(f64::total_cmp);
    let at = |percent: usize| sorted[(sorted.len() * percent).div_ceil(100) - 1];
    json!({"samples":values.len(),"p50":at(50),"p95":at(95),"p99":at(99),
        "min":sorted[0],"max":sorted[sorted.len()-1],"samples_ms_in_order":values})
}

/// One timed zero-acceptance Verify6 on `target` at its current frontier.
struct Trial {
    milliseconds: f64,
    rollback_milliseconds: f64,
    ids: Vec<u32>,
    logits: Option<Vec<u8>>,
}
fn trial(
    cuda: &Cuda,
    target: &mut Target,
    block: &[u32],
    hash: bool,
    trace: Option<&Path>,
    comm: &Roce,
    stream: &Stream,
) -> Result<Trial> {
    let start = Instant::now();
    target.begin_verification(block, comm, stream)?;
    let ids = target.verification_greedy(stream)?;
    let milliseconds = start.elapsed().as_secs_f64() * 1000.;
    // Readback/hashing is diagnostic and outside the measured interval.
    let logits = if hash {
        let (logits, live) = target.pending_logits()?;
        ensure!(
            live == 6 && logits.bytes() >= 6 * 129280 * 4,
            "fixed logits bounds"
        );
        Some(unsafe { cuda.read_device(logits.ptr(), 6 * 129280 * 4) }?)
    } else {
        None
    };
    let rollback = Instant::now();
    target.finish_verification(0, comm, stream)?;
    let rollback_milliseconds = rollback.elapsed().as_secs_f64() * 1000.;
    // The zero-acceptance rollback restores causal carry, but preserves layer
    // workspaces. Snapshot only after the writer is healthy and outside both
    // measured intervals. These are real forward outputs, never fixture input.
    if let Some(path) = trace {
        target.dump_layer_trace(path)?;
    }
    Ok(Trial {
        milliseconds,
        rollback_milliseconds,
        ids,
        logits,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    cuda: &Cuda,
    large: &mut Target,
    verify: &mut Target,
    mut candidate: Option<(&mut Target, &[String])>,
    comm: &Roce,
    stream: &Stream,
    prefix: &[u32],
    fixture_path: &Path,
    output: &Path,
    rank: usize,
    trace_layers: bool,
) -> Result<Value> {
    let bytes = std::fs::read(fixture_path)?;
    let f: Fixture = serde_json::from_slice(&bytes)?;
    ensure!(
        f.case == "python-http-retry-fix"
            && prefix.len() == 458
            && large.context_capacity() == 1024
            && f.teacher_forced_tokens.len() == 512
            && f.teacher_forced_tokens.iter().all(|&id| id < 129280)
            && f.frontiers == [512, 640, 768, 896]
            && f.warmup_iterations == 2
            && f.measured_iterations == 100,
        "fixed verification diagnostic fixture/profile"
    );
    let started = Instant::now();
    large.prefill(prefix, comm, stream)?;
    let fresh_prefill_seconds = started.elapsed().as_secs_f64();
    verify.handoff_from(large)?;
    let mut records = Vec::new();
    for frontier in f.frontiers {
        while (verify.position() as usize) < frontier {
            let begin = verify.position() as usize - prefix.len();
            let end = (begin + 6).min(frontier - prefix.len());
            verify.append(&f.teacher_forced_tokens[begin..end], comm, stream)?;
        }
        ensure!(verify.position() as usize == frontier, "fixed frontier");
        let block = &f.teacher_forced_tokens[frontier - prefix.len()..frontier - prefix.len() + 6];
        let carry_before = verify.carry_digests()?;
        let encoder_before = verify.encoder_tail_digest()?;
        let mut samples = Vec::new();
        let mut rollback_samples = Vec::new();
        let mut logits_hashes = Vec::new();
        let mut predictions = Vec::new();
        let mut candidate_samples = Vec::new();
        let mut candidate_hashes = Vec::new();
        let mut candidate_predictions = Vec::new();
        let mut paired = Vec::new();
        for t in 0..f.warmup_iterations + f.measured_iterations {
            let measured = t >= f.warmup_iterations;
            let hash =
                t == f.warmup_iterations || t + 1 == f.warmup_iterations + f.measured_iterations;
            // Alternate which schedule runs first so order/thermal effects cancel.
            let candidate_first = t % 2 == 1;
            let mut control = None;
            let mut other = None;
            for slot in 0..2 {
                if (slot == 0) == candidate_first {
                    if let Some((c, _)) = candidate.as_mut() {
                        c.handoff_from(verify)?;
                        let trace = (trace_layers && hash).then(|| {
                            output.join(format!("frontier{frontier}-trial{t}-candidate-layers"))
                        });
                        other = Some(trial(cuda, c, block, hash, trace.as_deref(), comm, stream)?);
                        verify.handoff_from(c)?;
                    }
                } else {
                    let trace = (trace_layers && hash)
                        .then(|| output.join(format!("frontier{frontier}-trial{t}-layers")));
                    control = Some(trial(
                        cuda,
                        verify,
                        block,
                        hash,
                        trace.as_deref(),
                        comm,
                        stream,
                    )?);
                }
            }
            let control = control.unwrap();
            ensure!(
                verify.position() as usize == frontier,
                "zero acceptance advanced history"
            );
            if let Some(raw) = &control.logits {
                if rank == 0 {
                    std::fs::write(
                        output.join(format!("frontier{frontier}-trial{t}-logits.bin")),
                        raw,
                    )?;
                }
                logits_hashes.push(format!("{:x}", Sha256::digest(raw)));
            }
            if measured {
                samples.push(control.milliseconds);
                rollback_samples.push(control.rollback_milliseconds);
                predictions.push(control.ids.clone());
            }
            if let Some(o) = other {
                if let Some(raw) = &o.logits {
                    candidate_hashes.push(format!("{:x}", Sha256::digest(raw)));
                }
                if measured {
                    paired.push(o.milliseconds - control.milliseconds);
                    candidate_samples.push(o.milliseconds);
                    candidate_predictions.push(o.ids);
                }
            }
        }
        let carry_after = verify.carry_digests()?;
        let encoder_after = verify.encoder_tail_digest()?;
        ensure!(
            carry_before == carry_after && encoder_before == encoder_after,
            "repeated zero-acceptance changed accepted carry/encoder state"
        );
        eprintln!("fixed verifier rank{rank} frontier{frontier} completed100 measured iterations");
        let mut record = json!({"frontier":frontier,"block":block,"live_rows":6,
            "target_and_greedy_host_ms":distribution(samples),"rollback_host_ms":distribution(rollback_samples),
            "logits_sha256_first_last":logits_hashes,"repeated_logits_byte_identical":logits_hashes[0]==logits_hashes[1],
            "predictions":predictions,"accepted_carry_exact":true,"accepted_encoder_tail_exact":true,
            "accepted_encoder_tail":encoder_before});
        if candidate.is_some() {
            let agree = candidate_predictions
                .iter()
                .zip(record["predictions"].as_array().unwrap())
                .filter(|(c, p)| serde_json::to_value(c).ok().as_ref() == Some(*p))
                .count();
            record["candidate"] = json!({"target_and_greedy_host_ms":distribution(candidate_samples),
                "candidate_minus_control_paired_ms":distribution(paired),
                "logits_sha256_first_last":candidate_hashes,
                "logits_byte_identical_to_control":candidate_hashes==logits_hashes,
                "predictions_equal_to_control":agree});
        }
        records.push(record);
    }
    Ok(
        json!({"rank":rank,"fixture_sha256":format!("{:x}",Sha256::digest(bytes)),
        "fresh_prefill_seconds":fresh_prefill_seconds,"swa_storage":verify.swa_storage_report()?,
        "ffn_overlap":std::env::var("CANADA_SCRATCH_FFN_OVERLAP").as_deref()==Ok("1"),
        "l2_prefetch":crate::l2_prefetch::enabled(),"mhc_overlap":crate::mhc::overlap_enabled(),
        "fp8_projection_backend":if crate::fp8_linear::pdl_enabled(){"target_small_qb_flashinfer_static_pdl"}else{"b12x_words"},
        "candidate_flags":candidate.as_ref().map(|(_, f)| f.to_vec()),
        "layer_traces_first_last":trace_layers,
        "records":records,"full_engine_acceptance":false,"quality_parity_proven":false,
        "scope":"Fixed teacher-forced accepted prefixes, target Verify6 and greedy readout.100measured repeats/frontier after2warmups, followed by zero-acceptance rollback. Excludes readback/hash and rollback from verification interval; no drafter, HTTP, transport, generation throughput or broad prefill consistency claim. Byte variability is reported, not a universal correctness/speed gate. Accepted carry and encoder rollback remain exact requirements."}),
    )
}
