//! Retained target/drafter continuations, including the pending emitted anchor.
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
) -> Result<Value> {
    const DELTAS: [usize; 8] = [1, 6, 32, 127, 128, 256, 1024, 4096];
    const OUTPUTS: usize = 64;
    ensure!(
        prefix.len() + DELTAS.iter().sum::<usize>() + OUTPUTS * 9 <= large.context_capacity(),
        "retained generation probe context"
    );
    let corpus: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/target-agentic-repository-v1.json"
    ))?;
    let corpus: Vec<u32> = serde_json::from_value(corpus["batches"][0].clone())?;
    let seed = std::env::var("CANADA_SCRATCH_TARGET_SEED")?.parse::<u64>()?;
    large.enable_cached_extend_floor()?;
    let mut session: Option<GreedySession> = None;
    let mut records = Vec::new();
    for repeat in 0..2 {
        if repeat > 0 {
            verify.reset(30001, comm, stream)?;
            large.handoff_from(verify)?;
        }
        let mut committed = prefix.to_vec();
        large.prefill_wide(wides, prefix, comm, stream)?;
        let mut anchor = large.sample_token(seed, stream)?;
        if let Some(s) = &mut session {
            s.reset_from_prefill(large, cuda, core, stream)?;
        } else {
            session = Some(GreedySession::from_prefill(
                large, cuda, core, stream, cp, plan, staged, kernels,
            )?);
        }
        let s = session.as_mut().unwrap();
        s.configure_sampling(seed, None)?;
        if std::env::var("CANADA_SCRATCH_HISTORY_LOOKUP").as_deref() == Ok("1") {
            s.configure_history_lookup(large, prefix)?;
        }
        verify.handoff_from(large)?;
        for turn in 0..=DELTAS.len() {
            let mut extend_seconds = 0.;
            let mut delta = 0;
            if turn > 0 {
                delta = DELTAS[turn - 1];
                let previous = verify.position();
                ensure!(
                    previous == committed.len() as u64,
                    "retained accepted token history"
                );
                // The prior emitted token is pending, not already in KV.
                let mut new = vec![anchor];
                new.extend_from_slice(&corpus[..delta - 1]);
                large.handoff_from(verify)?;
                let begin = Instant::now();
                large.prefill_wide(wides, &new, comm, stream)?;
                s.resume_after_extend(large, previous, &new, stream)?;
                anchor = large.sample_token(seed, stream)?;
                verify.handoff_from(large)?;
                extend_seconds = begin.elapsed().as_secs_f64();
                committed.extend_from_slice(&new);
            }
            let begin = Instant::now();
            let mut emitted = vec![anchor];
            let mut steps = Vec::new();
            while emitted.len() < OUTPUTS {
                let step = s.step(
                    verify,
                    anchor,
                    (OUTPUTS - emitted.len()).min(6),
                    comm,
                    stream,
                )?;
                ensure!(step.tokens[0] == anchor, "step lost its pending anchor");
                committed.extend_from_slice(&step.tokens);
                emitted.extend_from_slice(&step.tokens[1..]);
                emitted.push(step.next_anchor);
                anchor = step.next_anchor;
                steps.push(step);
            }
            ensure!(
                committed.len() as u64 == verify.position() && emitted.len() == OUTPUTS,
                "retained target/draft emission frontier"
            );
            let record = json!({"repeat":repeat,"turn":turn,"new_tokens":delta,
                "extend_seconds":extend_seconds,"decode_seconds":begin.elapsed().as_secs_f64(),
                "output_ids":emitted,"committed_frontier":verify.position(),"pending_anchor":anchor,
                "accepted_token_count":committed.len(),"steps":steps});
            std::fs::write(
                output.join(format!("retained-generation-r{repeat}-t{turn}.json")),
                serde_json::to_vec_pretty(&record)?,
            )?;
            eprintln!(
                "retained generation rank{} repeat{repeat} turn{turn} delta{delta} position{}",
                plan.rank,
                verify.position()
            );
            records.push(record);
        }
    }
    Ok(
        json!({"rank":plan.rank,"records":records,"memory":cuda.admission_memory()?,
        "history_lookup_enabled":std::env::var("CANADA_SCRATCH_HISTORY_LOOKUP").as_deref()==Ok("1"),
        "full_engine_acceptance":false,"scope":"Two repeated nine-turn native target/drafter sequences:64 diagnostic outputs per turn atT1/topP1; retained increments1/6/32/127/128/256/1024/4096 explicitly include the prior pending emitted anchor. Small EXTEND preserves existing draft KV and injects only new target features;128+ replaces the complete draft window. Exact integer frontier checks, full token/step evidence and rank comparison required. This truncated state diagnostic is not natural-completion, task-quality, HTTP or SGLang performance qualification."}),
    )
}
