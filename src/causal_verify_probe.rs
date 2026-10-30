//! Short causal verification policy screens, isolated from HTTP serving.
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

const OUTPUTS: usize = 128;
const WARMUP: usize = 1;
const MEASURED: usize = 3;
const NAMES: [&str; 3] = ["fixed_six", "causal_three_threshold_0.20", "causal_three_threshold_0.35"];
const THRESHOLDS: [Option<f32>; 3] = [None, Some(0.20), Some(0.35)];

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    cuda: &Cuda,
    core: &[u8],
    stream: &Stream,
    cp: &Checkpoint,
    plan: &RankPlan,
    large: &mut Target,
    verify: &mut Target,
    wides: &mut [Target],
    comm: &Roce,
    prefix: &[u32],
    staged: &Path,
    kernels: &Path,
    output: &Path,
    session: &mut Option<GreedySession>,
    reuse_ready_profiles: bool,
) -> Result<Value> {
    if std::env::var("CANADA_SCRATCH_TREE_SCREEN").as_deref()==Ok("1") {
        ensure!(!reuse_ready_profiles,"tree screen is isolated from serving");
        return crate::tree_opportunity_probe::run(cuda,core,stream,cp,plan,large,
            verify,wides,comm,prefix,staged,kernels,output,session);
    }
    let algorithm_screen=std::env::var("CANADA_SCRATCH_ALGORITHM_SCREEN").as_deref()==Ok("1");
    let algorithm_followup=std::env::var("CANADA_SCRATCH_ALGORITHM_FOLLOWUP").as_deref()==Ok("1");
    let conditioning_ab=std::env::var("CANADA_SCRATCH_CONDITIONING_AB").as_deref()==Ok("1");
    ensure!(!algorithm_followup || algorithm_screen,"algorithm follow-up requires algorithm screen");
    let depth_screen = std::env::var("CANADA_SCRATCH_CAUSAL_DEPTH_SCREEN").as_deref() == Ok("1");
    ensure!(!algorithm_screen || !depth_screen, "conflicting sampled screens");
    ensure!(!conditioning_ab || !(algorithm_screen||depth_screen),"conditioning comparison is isolated");
    let names:Vec<&str> = if conditioning_ab {vec!["replicated_conditioning","split_conditioning"]} else if algorithm_screen {
        vec!["token_t1", "block_t1", "token_t0.7", "block_t0.7"]
    } else if depth_screen {
        vec!["causal_three_threshold_0.20", "causal_two_threshold_0.20", "causal_four_threshold_0.20"]
    } else { NAMES.to_vec() };
    let thresholds = if algorithm_screen || depth_screen || conditioning_ab { vec![Some(0.20);names.len()] } else { THRESHOLDS.to_vec() };
    let depths = if depth_screen { vec![3, 2, 4] } else { vec![3;names.len()] };
    let outputs=if algorithm_followup {256} else if algorithm_screen || conditioning_ab {96} else {OUTPUTS};
    let measured=if algorithm_followup {4} else if algorithm_screen || conditioning_ab {2} else {MEASURED};
    ensure!(!depth_screen || (reuse_ready_profiles && large.context_capacity()==1048576),
        "depth comparison requires the isolated post-HTTP1M screen");
    ensure!(
        prefix.len() == 511
            && [32768, 1048576].contains(&large.context_capacity())
            && verify.query_overlap_layers() == 40
            && std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref() == Ok("1")
            && std::env::var("CANADA_SCRATCH_STOCHASTIC_DRAFT").as_deref() == Ok("1"),
        "causal depth screen requires admitted combined-schedule stochastic profiles"
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
    // The optional HTTP follow-up reuses its already-configured profiles and
    // session after the listener has stopped and the target has reset to zero.
    ensure!(large.position() == 0 && (!reuse_ready_profiles || session.is_some()),
        "causal screen requires a fresh frontier and a prepared reused session");
    if !reuse_ready_profiles {
        large.enable_cached_extend_floor()?;
    }
    let mut records = Vec::new();
    let mut lineage = 110000;
    for (prompt_index, prompt) in [prefix, long.as_slice()].into_iter().enumerate() {
        for repeat in 0..WARMUP + measured {
            let request_seed = seed.wrapping_add((prompt_index * 100 + repeat) as u64);
            // Rotate variant order; the four-way rapid screen has only two measured repeats.
            for slot in 0..names.len() {
                let variant = (repeat + slot) % names.len();
                let started = Instant::now();
                large.prefill_wide(wides, prompt, comm, stream)?;
                let mut anchor = large.sample_token(request_seed, stream)?;
                let prefill_seconds = started.elapsed().as_secs_f64();
                let setup = Instant::now();
                if let Some(s) = session.as_mut() {
                    s.reset_from_prefill(large, cuda, core, stream)?;
                } else {
                    *session = Some(GreedySession::from_prefill(
                        large, cuda, core, stream, cp, plan, staged, kernels,
                    )?);
                }
                let s = session.as_mut().unwrap();
                // No EOS stopping: a fixed diagnostic output count, including reasoning.
                s.configure_sampling(request_seed, None)?;
                if conditioning_ab {s.set_conditioning_split_for_probe(variant==1,stream)?;}
                if algorithm_screen {
                    s.configure_algorithm_for_probe(variant%2==1, if variant>=2 {0.7}else{1.},stream)?;
                }
                if depth_screen {
                    s.set_causal_depth_for_probe(depths[variant], thresholds[variant])?;
                } else {
                    s.set_causal_three_threshold_for_probe(thresholds[variant])?;
                }
                let lookup_bytes =
                    if std::env::var("CANADA_SCRATCH_HISTORY_LOOKUP").as_deref() == Ok("1") {
                        s.configure_history_lookup(large, prompt)?
                    } else {
                        0
                    };
                let t = &mut *verify;
                t.handoff_from(large)?;
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
                let mut emitted = Vec::with_capacity(outputs);
                emitted.push(anchor);
                let mut steps = Vec::with_capacity(outputs);
                while emitted.len() < outputs {
                    let step = s.step(t, anchor, (outputs - emitted.len()).min(6), comm, stream)?;
                    ensure!(step.tokens[0] == anchor, "schedule screen pending anchor");
                    emitted.extend_from_slice(&step.tokens[1..]);
                    emitted.push(step.next_anchor);
                    anchor = step.next_anchor;
                    steps.push(step);
                }
                let decode_seconds = begin.elapsed().as_secs_f64();
                ensure!(
                    emitted.len() == outputs && t.position() == (prompt.len() + outputs - 1) as u64,
                    "schedule screen accepted frontier"
                );
                let new_captures = t.capture_statistics().1 - captures_before;
                ensure!(
                    repeat < WARMUP || new_captures == 0,
                    "measured schedule recaptured"
                );
                let state = json!({"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?,
                    "frontier":t.position(),"pending_anchor":anchor,
                    "draft_kv":if conditioning_ab {Some(s.conditioning_state_digest_for_probe(comm,stream)?)}else{None}});
                let record = json!({"rank":plan.rank,"prompt_index":prompt_index,
                    "prompt_tokens":prompt.len(),"repeat":repeat,"warmup":repeat<WARMUP,
                    "variant":variant,"name":names[variant],"block_verification":algorithm_screen && variant%2==1,"draft_temperature":if algorithm_screen && variant>=2 {0.7}else{1.},"threshold":thresholds[variant],"short_rows":depths[variant],"slot":slot,"seed":request_seed,
                    "output_ids":emitted,"steps":steps,"final_state":state,
                    "prefill_seconds":prefill_seconds,"setup_seconds":setup_seconds,
                    "decode_seconds":decode_seconds,"interval_tokens":outputs-1,
                    "window_seconds":prefill_seconds+setup_seconds+decode_seconds,
                    "decode_tokens_per_second":(outputs-1)as f64/decode_seconds,
                    "new_verifier_captures":new_captures,"draft_head":s.draft_head_kind(),"conditioning_split":s.conditioning_split(),
                    "history_lookup_workspace_bytes":lookup_bytes,
                    "target_head":"canonical_bf16_weights_fp32_output"});
                std::fs::write(
                    output.join(format!(
                        "causal-p{prompt_index}-r{repeat}-v{variant}.json"
                    )),
                    serde_json::to_vec_pretty(&record)?,
                )?;
                eprintln!(
                    "causal depth rank{} prompt{} repeat{repeat} {}: {:.2}tok/s",
                    plan.rank,
                    prompt.len(),
                    names[variant],
                    (outputs - 1) as f64 / decode_seconds
                );
                records.push(record);
                lineage += 1;
                t.reset(lineage, comm, stream)?;
                large.handoff_from(t)?;
            }
        }
    }
    Ok(
        json!({"rank":plan.rank,"names":names,"depth_screen":depth_screen,"algorithm_screen":algorithm_screen,"algorithm_followup":algorithm_followup,"conditioning_ab":conditioning_ab,"records":records,"context_capacity":large.context_capacity(),"reused_after_http":reuse_ready_profiles,"temperature":1.,"top_p":1.,
        "outputs_per_trial":outputs,"warmup_repeats":WARMUP,"measured_repeats":measured,
        "memory":cuda.admission_memory()?,"full_engine_acceptance":false,
        "scope":if conditioning_ab {
            "Short paired replicated versus TP4-sharded draft conditioning screen. Target precision/settings, sampler, history lookup and causal verification unchanged. Two contexts,96 outputs,one warmup/two order-balanced measured seeds. Compare tokens, proposals, decisions and accepted state, while measuring full decode and conditioning/commit time. No completed-outcome or broad production qualification."
        } else if algorithm_followup {
            "Focused four-way block-verification and draft-temperature follow-up after finite HTTP checks. Two contexts,256 outputs,one warmup and four order-balanced measured seeds per variant. Target canonical precision/T1/topP1 and causal3 policy unchanged. This measures isolated decode intervals, not completed outcomes or broad production readiness."
        } else if algorithm_screen {
            "Short four-way block-verification and draft-temperature screen. Canonical target BF16 weights/FP32 logits/T1/topP1 unchanged; same causal3 policy. Actual draft q uses its sampling temperature; block rounds use independently keyed random streams and explicit accepted prefix. Two contexts,96 outputs,one warmup,two order-rotated measured repeats. This screen does not establish outcome quality or production readiness."
        } else if depth_screen {
            "Short causal-depth hypothesis screen: two context lengths, current three-row stopping versus two/four-row stopping at threshold0.20, one warmup and three order-balanced measured repeats,128 outputs each. A short block of n rows contains the anchor and proposals1..n-1; confidence n precedes proposal n and is conditioned only on the included prefix. Later confidences are ignored. Exact standard rejection sampling and canonical target precision are unchanged. Different valid stochastic trajectories are expected; cross-rank token/decision/state agreement remains mandatory. No completed-outcome, broad quality or production qualification."
        } else {
            "Short causal-depth hypothesis screen: two context lengths, fixed6 versus two thresholds, one warmup and three balanced measured repeats,128 outputs each. Both paths include the first two proposals; confidence3 precedes proposal3 and depends only on included proposals1/2. Later confidences are ignored. Exact standard rejection sampling and canonical target precision are unchanged. Different valid stochastic trajectories are expected; cross-rank token/decision/state agreement remains mandatory. No completed-outcome, broad quality or production qualification."
        }}),
    )
}
