//! Brief sampled long-context continuation and adaptive graph transition screen.
//! Public fixed corpus, canonical target precision, paired seeds under one load.
use crate::{checkpoint::Checkpoint, cuda::{Cuda, Stream}, dspark_generation::GreedySession,
    layout::RankPlan, roce::Roce, target::Target};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{path::Path, time::Instant};

const OUTPUTS: usize = 128;
const MEASURED: usize = 2;
const NAMES: [&str; 2] = ["dense_index", "adaptive_index"];

fn warm_rows(target: &mut Target, teacher: &[u32], comm: &Roce, stream: &Stream) -> Result<()> {
    for live in 1..=6 {
        target.begin_verification(&teacher[..live], comm, stream)?;
        target.verification_greedy(stream)?;
        target.finish_verification(0, comm, stream)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda: &Cuda, core: &[u8], stream: &Stream, cp: &Checkpoint,
    plan: &RankPlan, large: &mut Target, verify: &mut Target, candidate: &mut Target,
    wides: &mut [Target], comm: &Roce, staged: &Path, kernels: &Path, output: &Path,
) -> Result<Value> {
    ensure!(large.context_capacity() == 262144 && verify.sparse_index_layers() == 0
        && candidate.adaptive_sparse_index_layers() == 4
        && verify.query_overlap_layers() == 40 && candidate.query_overlap_layers() == 40
        && std::env::var("CANADA_SCRATCH_SPARSE_GENERATION_AB").as_deref() == Ok("1")
        && std::env::var("CANADA_SCRATCH_STOCHASTIC_DRAFT").as_deref() == Ok("1")
        && std::env::var("CANADA_SCRATCH_DRAFT_MAIN_SPLIT").as_deref() == Ok("1"),
        "long sparse generation requires admitted paired profiles and split drafting");
    let corpus: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/long-context-verification-corpus-v1.json"))?;
    ensure!(corpus["source_sha256"] == "9502643d4f0f1e5ed712073c8596757afad14ed8186ae354ff627a76983ea1fa",
        "long corpus provenance");
    let corpus: Vec<u32> = serde_json::from_value(corpus["tokens"].clone())?;
    ensure!(corpus.len() == 131584, "long corpus geometry");
    let teacher: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/fixed-verify-python-v1.json"))?;
    let teacher: Vec<u32> = serde_json::from_value(teacher["teacher_forced_tokens"].clone())?;
    let seed = std::env::var("CANADA_SCRATCH_TARGET_SEED")?.parse::<u64>()?;
    let mut barrier_in = cuda.allocate::<f32>(4)?;
    barrier_in.upload(&[0.; 4])?;
    let mut barrier_out = cuda.allocate::<f32>(16)?;
    large.enable_cached_extend_floor()?;
    let mut session: Option<GreedySession> = None;
    let mut lineage = 130000;
    let mut records = Vec::new();
    let mut comparisons = Vec::new();
    // The first prompt deliberately crosses 65536 during real sampled commits.
    for (prompt_index, length) in [65520, 131072].into_iter().enumerate() {
        let prompt = &corpus[..length];
        for repeat in 0..=MEASURED {
            let request_seed = seed.wrapping_add((prompt_index * 100 + repeat) as u64);
            let mut paired = [Value::Null, Value::Null];
            for slot in 0..2 {
                let variant = (repeat + slot) % 2;
                let started = Instant::now();
                large.prefill_wide(wides, prompt, comm, stream)?;
                let mut anchor = large.sample_token(request_seed, stream)?;
                let prefill_seconds = started.elapsed().as_secs_f64();
                let setup = Instant::now();
                if let Some(s) = session.as_mut() {
                    s.reset_from_prefill(large, cuda, core, stream)?;
                } else {
                    session = Some(GreedySession::from_prefill(large, cuda, core, stream,
                        cp, plan, staged, kernels)?);
                }
                let s = session.as_mut().unwrap();
                s.configure_sampling(request_seed, None)?;
                s.set_causal_three_threshold_for_probe(Some(0.20))?;
                let lookup_bytes = s.configure_history_lookup(large, prompt)?;
                ensure!(s.conditioning_split() && !s.block_verification() && s.draft_temperature() == 1.,
                    "sparse screen must retain serving draft policy");
                let t = if variant == 0 { &mut *verify } else { &mut *candidate };
                t.handoff_from(large)?;
                let setup_seconds = setup.elapsed().as_secs_f64();
                if repeat == 0 { warm_rows(t, &teacher, comm, stream)?; }
                stream.join_default()?;
                unsafe { comm.gather_f32(&barrier_in, &mut barrier_out, 4, stream)?; }
                stream.synchronize()?;
                comm.health()?;
                let captures_before = t.capture_statistics().1;
                let begin = Instant::now();
                let mut emitted = vec![anchor];
                let mut steps = Vec::new();
                let mut sparse_modes = Vec::new();
                let mut crossed_warm = length >= 65536;
                while emitted.len() < OUTPUTS {
                    if repeat == 0 && !crossed_warm && t.position() >= 65536 {
                        // Warm every row count in the other bank during the
                        // unmeasured trial; later trials must not recapture.
                        warm_rows(t, &teacher, comm, stream)?;
                        crossed_warm = true;
                    }
                    let position = t.position();
                    let step = s.step(t, anchor, (OUTPUTS - emitted.len()).min(6), comm, stream)?;
                    ensure!(step.tokens[0] == anchor, "pending anchor mismatch");
                    let active = t.active_sparse_index_layers();
                    ensure!(active == if variant == 1 && position >= 65536 {4} else {0},
                        "wrong sparse bank after sampled commit at {position}");
                    sparse_modes.push(json!({"position":position,"active_layers":active}));
                    emitted.extend_from_slice(&step.tokens[1..]);
                    emitted.push(step.next_anchor);
                    anchor = step.next_anchor;
                    steps.push(step);
                }
                let decode_seconds = begin.elapsed().as_secs_f64();
                let new_captures = t.capture_statistics().1 - captures_before;
                ensure!(repeat == 0 || new_captures == 0, "measured generation recaptured");
                ensure!(emitted.len() == OUTPUTS && t.position() == (length + OUTPUTS - 1) as u64,
                    "sampled committed frontier mismatch");
                if variant == 1 && prompt_index == 0 {
                    ensure!(sparse_modes.iter().any(|m|m["active_layers"]==0)
                        && sparse_modes.iter().any(|m|m["active_layers"]==4),
                        "sampled generation did not exercise both graph banks");
                }
                let state = json!({"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?,
                    "frontier":t.position(),"pending_anchor":anchor,
                    "draft_kv":s.conditioning_state_digest_for_probe(comm,stream)?});
                let signature: Vec<_> = steps.iter().map(|step|json!({"tokens":step.tokens,
                    "next_anchor":step.next_anchor,"draft_tokens":step.draft_tokens,
                    "matching_proposals":step.matching_proposals,"verified_rows":step.verified_rows,
                    "draft_confidence":step.draft_confidence,"lookup_match_tokens":step.lookup_match_tokens,
                    "proposal_source":step.proposal_source,"lookup_source_end":step.lookup_source_end,
                    "position":step.position})).collect();
                paired[variant] = json!({"output_ids":emitted,"signature":signature,"state":state});
                let record = json!({"rank":plan.rank,"prompt_index":prompt_index,"prompt_tokens":length,
                    "repeat":repeat,"warmup":repeat==0,"variant":variant,"name":NAMES[variant],
                    "slot":slot,"seed":request_seed,"output_ids":emitted,"steps":steps,
                    "final_state":state,"sparse_modes":sparse_modes,"prefill_seconds":prefill_seconds,
                    "setup_seconds":setup_seconds,"decode_seconds":decode_seconds,
                    "window_seconds":prefill_seconds+setup_seconds+decode_seconds,
                    "interval_tokens":OUTPUTS-1,"decode_tokens_per_second":(OUTPUTS-1)as f64/decode_seconds,
                    "new_verifier_captures":new_captures,"graph_captures":t.capture_statistics().1,
                    "draft_head":s.draft_head_kind(),"conditioning_split":s.conditioning_split(),
                    "history_lookup_workspace_bytes":lookup_bytes,"target_head":"canonical_bf16_weights_fp32_output"});
                std::fs::write(output.join(format!("sparse-generation-p{prompt_index}-r{repeat}-v{variant}.json")),
                    serde_json::to_vec_pretty(&record)?)?;
                eprintln!("sparse generation rank{} prompt{length} repeat{repeat} {}: {:.2}tok/s",
                    plan.rank,NAMES[variant],(OUTPUTS-1)as f64/decode_seconds);
                records.push(record);
                lineage += 1;
                t.reset(lineage, comm, stream)?;
                large.handoff_from(t)?;
            }
            comparisons.push(json!({"prompt_index":prompt_index,"repeat":repeat,
                "tokens_equal":paired[0]["output_ids"]==paired[1]["output_ids"],
                "steps_equal":paired[0]["signature"]==paired[1]["signature"],
                "state_equal":paired[0]["state"]==paired[1]["state"]}));
        }
    }
    let all_exact = comparisons.iter().all(|c|c["tokens_equal"]==true && c["steps_equal"]==true && c["state_equal"]==true);
    Ok(json!({"rank":plan.rank,"names":NAMES,"records":records,"comparisons":comparisons,
        "all_exact":all_exact,"context_capacity":large.context_capacity(),"temperature":1.,"top_p":1.,
        "outputs_per_trial":OUTPUTS,"warmup_repeats":1,"measured_repeats":MEASURED,
        "memory":cuda.admission_memory()?,"full_engine_acceptance":false,
        "scope":"Dense/adaptive sampled continuation at65520 and131072 actual tokens;128outputs,one warmup and two order-balanced measured seeds. Cross the64K graph threshold during commits. Canonical target T1/topP1, FP8 split drafter, actual-q rejection, history lookup and causal3 policy. Compare outputs/proposals/acceptance and target/draft state. This truncated continuation screen does not measure successful outcomes or establish SGLang parity."}))
}
