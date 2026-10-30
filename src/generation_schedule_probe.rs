//! Short sampled generation screen for accumulated, separately checked schedules.
//! Same seeds and canonical target head; this is not an outcome-quality benchmark.
use crate::{
    checkpoint::Checkpoint,
    cuda::{Cuda, Stream},
    dspark_generation::GreedySession,
    layout::RankPlan,
    roce::Roce,
    target::Target,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

const OUTPUTS: usize = 256;
const WARMUP: usize = 1;
const MEASURED: usize = 6;
const NAMES: [&str; 3] = ["control", "tail_and_engram", "tail_engram_and_query"];

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    cuda: &Cuda,
    core: &[u8],
    stream: &Stream,
    cp: &Checkpoint,
    plan: &RankPlan,
    large: &mut Target,
    verify: &mut Target,
    candidate: &mut Target,
    wides: &mut [Target],
    comm: &Roce,
    prefix: &[u32],
    staged: &Path,
    kernels: &Path,
    output: &Path,
) -> Result<Value> {
    ensure!(
        prefix.len() == 458
            && large.context_capacity() == 32768
            && verify.query_overlap_layers() == 0
            && candidate.query_overlap_layers() == 40
            && std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref() == Ok("1")
            && std::env::var("CANADA_SCRATCH_STOCHASTIC_DRAFT").as_deref() == Ok("1"),
        "generation schedule screen requires the admitted 32K sampled profiles"
    );
    let seed = std::env::var("CANADA_SCRATCH_TARGET_SEED")?.parse::<u64>()?;
    let long: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/target-agentic-long15800-v1.json"
    ))?;
    let mut long: Vec<u32> = serde_json::from_value(long["batches"][0].clone())?;
    let teacher: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/fixed-verify-python-v1.json"
    ))?;
    let teacher: Vec<u32> = serde_json::from_value(teacher["teacher_forced_tokens"].clone())?;
    ensure!(
        long.len() == 15800 && teacher.len() == 512,
        "generation screen fixtures"
    );
    long.extend_from_slice(&teacher);
    long.extend_from_slice(&teacher);
    large.enable_cached_extend_floor()?;
    let mut session: Option<GreedySession> = None;
    let mut records = Vec::new();
    let mut comparisons = Vec::new();
    let mut lineage = 90000;
    for (prompt_index, prompt) in [prefix, long.as_slice()].into_iter().enumerate() {
        for repeat in 0..WARMUP + MEASURED {
            let request_seed = seed.wrapping_add((prompt_index * 100 + repeat) as u64);
            let mut paired: [Value; 3] = std::array::from_fn(|_| Value::Null);
            // Each measured path appears twice in each order position.
            for slot in 0..3 {
                let variant = (repeat + slot) % 3;
                let started = Instant::now();
                large.prefill_wide_with_policy(wides, prompt, comm, stream, false, variant != 0)?;
                let mut anchor = large.sample_token(request_seed, stream)?;
                let prefill_seconds = started.elapsed().as_secs_f64();
                let setup = Instant::now();
                if let Some(s) = &mut session {
                    s.reset_from_prefill(large, cuda, core, stream)?;
                } else {
                    session = Some(GreedySession::from_prefill(
                        large, cuda, core, stream, cp, plan, staged, kernels,
                    )?);
                }
                let s = session.as_mut().unwrap();
                // No EOS stopping: exactly256 diagnostic outputs, including reasoning.
                s.configure_sampling(request_seed, None)?;
                let lookup_bytes =
                    if std::env::var("CANADA_SCRATCH_HISTORY_LOOKUP").as_deref() == Ok("1") {
                        s.configure_history_lookup(large, prompt)?
                    } else {
                        0
                    };
                let t = if variant == 2 {
                    &mut *candidate
                } else {
                    &mut *verify
                };
                t.handoff_from(large)?;
                t.set_engram_overlap_for_probe(variant != 0)?;
                let setup_seconds = setup.elapsed().as_secs_f64();
                if repeat == 0 {
                    // Capture every possible verifier row count before timed
                    // decoding. Rollback leaves the accepted frontier unchanged.
                    for live in 1..=6 {
                        t.begin_verification(&teacher[..live], comm, stream)?;
                        t.verification_greedy(stream)?;
                        t.finish_verification(0, comm, stream)?;
                    }
                }
                let captures_before = t.capture_statistics().1;
                let begin = Instant::now();
                let mut emitted = Vec::with_capacity(OUTPUTS);
                emitted.push(anchor);
                let mut steps = Vec::with_capacity(OUTPUTS);
                while emitted.len() < OUTPUTS {
                    let step = s.step(t, anchor, (OUTPUTS - emitted.len()).min(6), comm, stream)?;
                    ensure!(step.tokens[0] == anchor, "schedule screen pending anchor");
                    emitted.extend_from_slice(&step.tokens[1..]);
                    emitted.push(step.next_anchor);
                    anchor = step.next_anchor;
                    steps.push(step);
                }
                let decode_seconds = begin.elapsed().as_secs_f64();
                let signature: Vec<_> = steps.iter().map(|step| json!({"tokens":step.tokens,
                    "next_anchor":step.next_anchor,"draft_tokens":step.draft_tokens,
                    "matching_proposals":step.matching_proposals,"verified_rows":step.verified_rows,
                    "proposal_source":step.proposal_source,"lookup_source_end":step.lookup_source_end,
                    "position":step.position})).collect();
                ensure!(
                    emitted.len() == OUTPUTS && t.position() == (prompt.len() + OUTPUTS - 1) as u64,
                    "schedule screen accepted frontier"
                );
                let new_captures = t.capture_statistics().1 - captures_before;
                ensure!(
                    repeat < WARMUP || new_captures == 0,
                    "measured schedule recaptured"
                );
                let state = json!({"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?,
                    "frontier":t.position(),"pending_anchor":anchor});
                paired[variant] = json!({"output_ids":emitted,"signature":signature,"state":state});
                let record = json!({"rank":plan.rank,"prompt_index":prompt_index,
                    "prompt_tokens":prompt.len(),"repeat":repeat,"warmup":repeat<WARMUP,
                    "variant":variant,"name":NAMES[variant],"slot":slot,"seed":request_seed,
                    "output_ids":emitted,"steps":steps,"final_state":state,
                    "prefill_seconds":prefill_seconds,"setup_seconds":setup_seconds,
                    "decode_seconds":decode_seconds,"interval_tokens":OUTPUTS-1,
                    "window_seconds":prefill_seconds+setup_seconds+decode_seconds,
                    "decode_tokens_per_second":(OUTPUTS-1)as f64/decode_seconds,
                    "new_verifier_captures":new_captures,"draft_head":s.draft_head_kind(),
                    "history_lookup_workspace_bytes":lookup_bytes,
                    "target_head":"canonical_bf16_weights_fp32_output"});
                std::fs::write(
                    output.join(format!(
                        "generation-p{prompt_index}-r{repeat}-v{variant}.json"
                    )),
                    serde_json::to_vec_pretty(&record)?,
                )?;
                eprintln!(
                    "generation schedule rank{} prompt{} repeat{repeat} {}: {:.2}tok/s",
                    plan.rank,
                    prompt.len(),
                    NAMES[variant],
                    (OUTPUTS - 1) as f64 / decode_seconds
                );
                records.push(record);
                lineage += 1;
                t.reset(lineage, comm, stream)?;
                large.handoff_from(t)?;
            }
            comparisons.push(json!({"prompt_index":prompt_index,"repeat":repeat,
                "tokens_equal":paired.iter().all(|p|p["output_ids"]==paired[0]["output_ids"]),
                "steps_equal":paired.iter().all(|p|p["signature"]==paired[0]["signature"]),
                "state_equal":paired.iter().all(|p|p["state"]==paired[0]["state"])}));
        }
    }
    let all_exact = comparisons
        .iter()
        .all(|c| c["tokens_equal"] == true && c["steps_equal"] == true && c["state_equal"] == true);
    Ok(
        json!({"rank":plan.rank,"names":NAMES,"records":records,"comparisons":comparisons,
        "all_exact":all_exact,"context_capacity":32768,"temperature":1.,"top_p":1.,
        "outputs_per_trial":OUTPUTS,"warmup_repeats":WARMUP,"measured_repeats":MEASURED,
        "memory":cuda.admission_memory()?,"full_engine_acceptance":false,
        "scope":"Short paired sampled generation: two context lengths, three cumulative scheduling variants, one warmup and six balanced measured repeats with identical seeds. Exact output, speculative acceptance and final carry/encoder state comparisons. Every verifier row count is prewarmed; target precision is unchanged. Truncated256-output windows are a scheduling screen, not completed-outcome speed, task quality or production qualification."}),
    )
}
