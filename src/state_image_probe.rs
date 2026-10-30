//! Complete same-owner save/replace/restore qualification. Fixed-duration
//! diagnostic continuations, never a natural outcome or serving benchmark.
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

const BUDGET: usize = 2 * 1024usize.pow(3);

/// Reuse an already loaded long-context continuation for a small state-cache
/// feasibility screen. No HTTP cache is enabled by this probe.
#[allow(clippy::too_many_arguments)]
pub(crate) fn probe_long_boundary(
    cuda:&Cuda, core:&[u8], stream:&Stream, large:&mut Target, verify:&mut Target,
    wides:&mut[Target], comm:&Roce, session:&mut GreedySession, mut anchor:u32,
    seed:u64, lineage:&mut u64, corpus:&[u32], output:&Path,
) -> Result<Value> {
    ensure!(std::env::var("CANADA_SCRATCH_LONG_STATE_IMAGE_SCREEN").as_deref()==Ok("1")
        && std::env::var("CANADA_SCRATCH_COMPACT_STATE_IMAGE").as_deref()==Ok("1")
        && verify.context_capacity()==1048576 && verify.position()>=262144
        && verify.profile_is_current() && session.block_verification()
        && session.history_lookup_metadata().is_some(),"compact long state screen admission");
    let budget=768*1024usize.pow(2);
    let other:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/target-agentic-rust-v1.json"))?;
    let other:Vec<u32>=serde_json::from_value(other["batches"][0].clone())?;
    let mut records=Vec::new();
    for delta in [0usize,32,8193] {
        let frontier=verify.position();
        let began=Instant::now();
        let saved=session.save_state_image(verify,anchor,budget,comm,stream)?;
        let save_seconds=began.elapsed().as_secs_f64();
        let saved_digest=saved.digests()?;
        let manifest=saved.manifest();
        ensure!(manifest["target"]["compact"]==true,"long state image was not compact");
        let branch=|session:&mut GreedySession,large:&mut Target,verify:&mut Target,
                    wides:&mut[Target],anchor:&mut u32|->Result<Value>{
            let began=Instant::now();
            if delta>0 {
                let before=verify.position();
                let mut ids=vec![*anchor];ids.extend_from_slice(&corpus[..delta-1]);
                large.handoff_from(verify)?;
                large.prefill_wide(wides,&ids,comm,stream)?;
                session.resume_after_extend(large,before,&ids,stream)?;
                *anchor=large.sample_token(seed,stream)?;
                verify.handoff_from(large)?;
            }
            let extend_seconds=began.elapsed().as_secs_f64();
            let continuation=advance(session,verify,anchor,24,comm,stream)?;
            Ok(json!({"continuation":continuation,"carry":verify.carry_digests()?,
                "encoder":verify.encoder_tail_digest()?,"extend_seconds":extend_seconds}))
        };
        let control=branch(session,large,verify,wides,&mut anchor)?;
        *lineage+=1;verify.reset(*lineage,comm,stream)?;large.handoff_from(verify)?;
        large.prefill_wide(wides,&other,comm,stream)?;
        let changed_seed=seed.wrapping_add(delta as u64+10000);
        let mut other_anchor=large.sample_token(changed_seed,stream)?;
        session.reset_from_prefill(large,cuda,core,stream)?;
        session.configure_sampling(changed_seed,None)?;verify.handoff_from(large)?;
        advance(session,verify,&mut other_anchor,12,comm,stream)?;
        *lineage+=1;
        let began=Instant::now();
        anchor=session.restore_state_image(verify,&saved,*lineage,comm,stream)?;
        let restore_seconds=began.elapsed().as_secs_f64();
        ensure!(!large.profile_is_current() && verify.logits().is_err(),"compact restore exposed stale outputs");
        let restored_digest=session.save_state_image(verify,anchor,budget,comm,stream)?.digests()?;
        ensure!(restored_digest==saved_digest,"compact saved ranges or sampler metadata changed");
        let resumed=branch(session,large,verify,wides,&mut anchor)?;
        ensure!(resumed["continuation"]==control["continuation"] && resumed["carry"]==control["carry"]
            && resumed["encoder"]==control["encoder"],"compact restoration changed subsequent model execution");
        let record=json!({"frontier":frontier,"retained_increment":delta,"save_seconds":save_seconds,
            "restore_seconds":restore_seconds,"manifest":manifest,"saved_ranges_and_metadata_exact":true,
            "continuation_and_carry_exact":true,"control":control,"resumed":resumed,
            "admitted_memory":cuda.admission_memory()?});
        std::fs::write(output.join(format!("compact-state-d{delta}.json")),serde_json::to_vec_pretty(&record)?)?;
        eprintln!("compact state rank{} delta{delta}: save{save_seconds:.4}s restore{restore_seconds:.4}s exact",comm.rank());
        records.push(record);
    }
    Ok(json!({"records":records,"budget_bytes":budget,"all_exact":true,"http_cache_enabled":false,
        "scope":"Three current262K state boundaries: plain24-input continuation, retained32 and8193 plus24-input continuations. Reset and execute an unrelated request before restoration. Compare every captured byte and sampler metadata, subsequent tokens/logits/carry/encoder. Unpublished pages outside the compact image are not retained. Small feasibility screen, not a production cache or recovery qualification."}))
}

fn advance(
    session: &mut GreedySession,
    target: &mut Target,
    anchor: &mut u32,
    inputs: usize,
    comm: &Roce,
    stream: &Stream,
) -> Result<Value> {
    let mut ids = Vec::new();
    let mut decisions = Vec::new();
    while ids.len() < inputs {
        let step = session.step(target, *anchor, (inputs - ids.len()).min(6), comm, stream)?;
        ensure!(
            step.tokens[0] == *anchor,
            "state image probe pending anchor"
        );
        ids.extend_from_slice(&step.tokens);
        *anchor = step.next_anchor;
        decisions.push(
            json!({"draft":step.draft_tokens,"accepted":step.matching_proposals,
            "verified":step.verified_rows,"confidence":step.draft_confidence,
            "source":step.proposal_source,"lookup_match":step.lookup_match_tokens,
            "lookup_end":step.lookup_source_end,"position":step.position}),
        );
    }
    stream.synchronize()?;
    comm.health()?;
    Ok(
        json!({"committed_inputs":ids,"pending_anchor":anchor,"decisions":decisions,
        "final_outputs":target.valid_output_digests()?}),
    )
}

#[allow(clippy::too_many_arguments)]
fn continuation(
    session: &mut GreedySession,
    large: &mut Target,
    verify: &mut Target,
    wides: &mut [Target],
    anchor: &mut u32,
    extension: &[u32],
    seed: u64,
    comm: &Roce,
    stream: &Stream,
) -> Result<Value> {
    if !extension.is_empty() {
        let before = verify.position();
        let mut ids = vec![*anchor];
        ids.extend_from_slice(extension);
        large.handoff_from(verify)?;
        large.prefill_wide(wides, &ids, comm, stream)?;
        session.resume_after_extend(large, before, &ids, stream)?;
        *anchor = large.sample_token(seed, stream)?;
        verify.handoff_from(large)?;
    }
    advance(session, verify, anchor, 96, comm, stream)
}

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
    ensure!(
        [458, 15800].contains(&prefix.len())
            && [32768, 524288].contains(&large.context_capacity())
            && std::env::var("CANADA_SCRATCH_STOCHASTIC_DRAFT").as_deref() == Ok("1")
            && std::env::var_os("CANADA_SCRATCH_ENGRAM_OVERLAP_AB").is_none()
            && std::env::var_os("CANADA_SCRATCH_ENGRAM_PREFETCH_AB").is_none(),
        "state image probe requires the isolated stochastic target/draft profile"
    );
    let seed = std::env::var("CANADA_SCRATCH_TARGET_SEED")?.parse::<u64>()?;
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/fixed-verify-python-v1.json"
    ))?;
    let teacher: Vec<u32> = serde_json::from_value(fixture["teacher_forced_tokens"].clone())?;
    let other: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/target-agentic-rust-v1.json"
    ))?;
    let other: Vec<u32> = serde_json::from_value(other["batches"][0].clone())?;
    ensure!(teacher.len() == 512, "state image teacher fixture");
    large.enable_cached_extend_floor()?;
    let mut session: Option<GreedySession> = None;
    let mut lineage = 50000;
    let mut records = Vec::new();
    for round in 0..2 {
        if round > 0 {
            lineage += 1;
            verify.reset(lineage, comm, stream)?;
            large.handoff_from(verify)?;
        }
        let mut prompt = prefix.to_vec();
        if round == 1 {
            // For the long-prefix run this crosses the source20 all-candidate
            // boundary, exercising retained encoder block-mask restoration.
            prompt.extend_from_slice(&teacher);
            prompt.extend_from_slice(&teacher);
        }
        large.prefill_wide(wides, &prompt, comm, stream)?;
        let request_seed = seed.wrapping_add(round);
        let mut anchor = large.sample_token(request_seed, stream)?;
        if let Some(session) = &mut session {
            session.reset_from_prefill(large, cuda, core, stream)?;
        } else {
            session = Some(GreedySession::from_prefill(
                large, cuda, core, stream, cp, plan, staged, kernels,
            )?);
        }
        let session = session.as_mut().unwrap();
        session.configure_sampling(request_seed, None)?;
        if round == 1 {
            session.configure_history_lookup(large, &prompt)?;
        }
        verify.handoff_from(large)?;
        advance(session, verify, &mut anchor, 96, comm, stream)?;
        ensure!(
            session
                .save_state_image(verify, anchor, 1, comm, stream)
                .is_err()
                && verify.profile_is_current(),
            "image budget refusal invalidated target"
        );
        let start = Instant::now();
        let saved = session.save_state_image(verify, anchor, BUDGET, comm, stream)?;
        let save_ms = start.elapsed().as_secs_f64() * 1000.;
        let saved_digest = saved.digests()?;
        let extension = if round == 1 {
            &teacher[..31]
        } else {
            &teacher[..0]
        };
        let control = continuation(
            session,
            large,
            verify,
            wides,
            &mut anchor,
            extension,
            request_seed,
            comm,
            stream,
        )?;
        let control_end = session
            .save_state_image(verify, anchor, BUDGET, comm, stream)?
            .digests()?;
        let mut trials = Vec::new();
        for repeat in 0..3 {
            lineage += 1;
            verify.reset(lineage, comm, stream)?;
            large.handoff_from(verify)?;
            large.prefill_wide(wides, &other, comm, stream)?;
            let changed_seed = request_seed.wrapping_add(100 + repeat);
            let mut other_anchor = large.sample_token(changed_seed, stream)?;
            session.reset_from_prefill(large, cuda, core, stream)?;
            session.configure_sampling(changed_seed, Some(1))?;
            verify.handoff_from(large)?;
            advance(session, verify, &mut other_anchor, 32, comm, stream)?;
            lineage += 1;
            let start = Instant::now();
            anchor = session.restore_state_image(verify, &saved, lineage, comm, stream)?;
            let restore_ms = start.elapsed().as_secs_f64() * 1000.;
            ensure!(
                !large.profile_is_current() && verify.logits().is_err(),
                "restored image exposed stale execution output/profile"
            );
            let restored_digest = session
                .save_state_image(verify, anchor, BUDGET, comm, stream)?
                .digests()?;
            let resumed = continuation(
                session,
                large,
                verify,
                wides,
                &mut anchor,
                extension,
                request_seed,
                comm,
                stream,
            )?;
            let resumed_end = session
                .save_state_image(verify, anchor, BUDGET, comm, stream)?
                .digests()?;
            let record = json!({"repeat":repeat,"restore_ms":restore_ms,
                "all_saved_bytes_and_metadata_exact":restored_digest == saved_digest,
                "continuation_exact":resumed == control,"final_state_exact":resumed_end == control_end,
                "resumed":resumed,"restored_digest":restored_digest,"final_digest":resumed_end});
            std::fs::write(
                output.join(format!("state-image-r{round}-trial{repeat}.json")),
                serde_json::to_vec_pretty(&record)?,
            )?;
            eprintln!(
                "state image rank{} round{round} repeat{repeat}: restore {restore_ms:.3}ms state {} continuation {}",
                plan.rank,
                record["all_saved_bytes_and_metadata_exact"],
                record["continuation_exact"]
            );
            trials.push(record);
        }
        records.push(json!({"round":round,"prefix_tokens":prompt.len(),"history_lookup":round == 1,
            "retained_increment":if round == 1 {32} else {0},"save_ms":save_ms,"manifest":saved.manifest(),
            "saved_digest":saved_digest,"control":control,"control_end":control_end,"trials":trials}));
    }
    let all_exact = records.iter().all(|r| {
        r["trials"].as_array().unwrap().iter().all(|t| {
            t["all_saved_bytes_and_metadata_exact"] == true
                && t["continuation_exact"] == true
                && t["final_state_exact"] == true
        })
    });
    Ok(
        json!({"rank":plan.rank,"context_capacity":large.context_capacity(),"prefix_tokens":prefix.len(),
        "records":records,"all_exact":all_exact,"memory":cuda.admission_memory()?,"full_engine_acceptance":false,
        "scope":"Two committed checkpoints, each restored three times after an unrelated prompt, changed seed/stop policy and different decoding. Plain sampled continuation and retained32-token EXTEND with text-lookup history. Complete target/draft device bytes and causal host metadata compared; new monotonic lineage is intentionally excluded. Lossless device-state prototype only: no HTTP scheduler caching, SSD parking, process recovery, natural-outcome or latency-tail qualification."}),
    )
}
