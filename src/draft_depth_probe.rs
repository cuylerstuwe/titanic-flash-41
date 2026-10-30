//! Revisit verification length under the accumulated block/T1.2 candidate.
use crate::{checkpoint::Checkpoint,cuda::{Cuda,Stream},dspark_generation::GreedySession,
    history_policy_probe::{advance,decision_signature},layout::RankPlan,roce::Roce,target::Target};
use anyhow::{ensure,Context,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};

fn physical_read_bytes()->Result<u64> {
    std::fs::read_to_string("/proc/self/io")?.lines()
        .find_map(|line|line.strip_prefix("read_bytes:"))
        .context("missing physical I/O counter")?.trim().parse().map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,core:&[u8],stream:&Stream,cp:&Checkpoint,plan:&RankPlan,
    large:&mut Target,verify:&mut Target,wides:&mut[Target],comm:&Roce,
    staged:&Path,kernels:&Path,output:&Path)->Result<Value> {
    let head_screen=std::env::var("CANADA_SCRATCH_DRAFT_HEAD_SCREEN").as_deref()==Ok("1");
    let markov_screen=std::env::var("CANADA_SCRATCH_MARKOV_HEAD_SCREEN").as_deref()==Ok("1");
    let private_gemv_screen=std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_GEMV_SCREEN").as_deref()==Ok("1");
    let confidence_screen=std::env::var("CANADA_SCRATCH_CONFIDENCE_FUSION_SCREEN").as_deref()==Ok("1");
    let history_match_screen=std::env::var("CANADA_SCRATCH_HISTORY_MATCH_SCREEN").as_deref()==Ok("1");
    ensure!(!confidence_screen || (!history_match_screen && !head_screen && !markov_screen && !private_gemv_screen),"isolated confidence screen");
    ensure!(!history_match_screen || (!head_screen && !markov_screen && !private_gemv_screen),"isolated history maximum screen");
    ensure!(!private_gemv_screen || (!head_screen && !markov_screen),"one isolated head hypothesis");
    let selector_capture=std::env::var("CANADA_SCRATCH_SELECTOR_CAPTURE").as_deref()==Ok("1");
    ensure!(!markov_screen || !head_screen,"separate Markov and M5 hypotheses");
    ensure!(!selector_capture || markov_screen,"selector capture accompanies Markov screen");
    let head_repeat=std::env::var("CANADA_SCRATCH_DRAFT_HEAD_REPEAT").as_deref()==Ok("1");
    let head_gemv_screen=std::env::var("CANADA_SCRATCH_DRAFT_HEAD_GEMV_SCREEN").as_deref()==Ok("1");
    ensure!(!head_gemv_screen || (head_screen && !head_repeat),"M5 screen uses two seeds and three warmed policies");
    let warm_trajectory=head_repeat || head_gemv_screen || markov_screen || private_gemv_screen || history_match_screen || confidence_screen;
    let replay_trace=std::env::var("CANADA_SCRATCH_DRAFT_REPLAY_TRACE").as_deref()==Ok("1");
    ensure!(!replay_trace || head_gemv_screen,"replay trace follows the s447 three-head sequence");
    ensure!(!head_repeat || head_screen,"trajectory warmup requires the head screen");
    ensure!((std::env::var("CANADA_SCRATCH_DRAFT_DEPTH_SCREEN").as_deref()==Ok("1") || head_screen || markov_screen || private_gemv_screen || history_match_screen || confidence_screen)
        && std::env::var("CANADA_SCRATCH_COMPACT_STATE_IMAGE").as_deref()==Ok("1")
        && std::env::var("CANADA_SCRATCH_CURRENT_CONTINUATION").as_deref()==Ok("1")
        && std::env::var("CANADA_SCRATCH_HISTORY_TABLE_SLOTS").as_deref()==Ok("262144")
        && large.context_capacity()==1048576 && wides.len()==5
        && verify.head_gemv_enabled() && verify.fused_compact_publication_layers()==4
        && verify.engram_input_kind()=="mapped" && verify.paired_engram_io_enabled(),
        "draft-depth screen requires the complete accumulated candidate");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture_sha=format!("{:x}",Sha256::digest(raw));
    ensure!(fixture_sha=="107ceccbefc53247fc827cdee7eb3908f7b89a907d8c7a2d3de3ef6b95057fbb","depth fixture hash");
    let fixture:Value=serde_json::from_slice(raw)?;
    let corpus:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let teacher:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/fixed-verify-python-v1.json"))?;
    let teacher:Vec<u32>=serde_json::from_value(teacher["teacher_forced_tokens"].clone())?;
    let prefix_length=if history_match_screen {131072}else{262144};
    let seed=std::env::var("CANADA_SCRATCH_TARGET_SEED")?.parse::<u64>()?+if history_match_screen {100}else{200};
    let mut barrier_in=cuda.allocate::<f32>(4)?;barrier_in.upload(&[0.;4])?;
    let mut barrier_out=cuda.allocate::<f32>(16)?;
    large.enable_cached_extend_floor()?;
    let began=Instant::now();large.prefill_wide(wides,&corpus[..prefix_length],comm,stream)?;
    let mut anchor=large.sample_token(seed,stream)?;let prefill_seconds=began.elapsed().as_secs_f64();
    let mut session=GreedySession::from_prefill(large,cuda,core,stream,cp,plan,staged,kernels)?;
    if confidence_screen {session.set_fused_confidence_for_probe(false,stream)?;}
    if markov_screen {session.set_fused_markov_head_for_probe(false,stream)?;}
    if private_gemv_screen {session.set_private_head_gemv_for_probe(false,stream)?;}
    session.configure_sampling(seed,None)?;session.configure_algorithm_for_probe(true,1.2,stream)?;
    session.set_causal_three_threshold_for_probe(Some(0.20))?;
    session.configure_history_lookup(large,&corpus[..prefix_length])?;verify.handoff_from(large)?;
    for live in 1..=6 {
        verify.begin_verification(&teacher[..live],comm,stream)?;
        verify.verification_greedy(stream)?;verify.finish_verification(0,comm,stream)?;
    }
    let warmup=advance(&mut session,verify,&mut anchor,24,comm,stream)?;
    let (image_bytes,_)=session.compact_state_image_bytes(verify)?;
    ensure!(image_bytes<=768*1024usize.pow(2),"depth snapshot budget");
    let saved=session.save_state_image(verify,anchor,image_bytes,comm,stream)?;
    ensure!(saved.manifest()["target"]["compact"]==true,"depth compact snapshot");
    let frontier=verify.position();let mut records=Vec::new();let mut control=None;
    // Two seeds, opposite candidate orders, plus an exact repeated control.
    // Full verification graphs and the drafter are warmed above. There is no
    // separate512-token warm trajectory for every unproven policy.
    let order:&[(usize,u64)]=if markov_screen || private_gemv_screen || history_match_screen || confidence_screen {&[(0,0),(1,0),(1,1),(0,1),(0,0)]}
        else if head_gemv_screen {&[(0,0),(1,0),(2,0),(2,1),(1,1),(0,1),(0,0)]}
        else if head_repeat {&[(0,0),(1,0),(1,1),(0,1),(0,2),(1,2),(1,3),(0,3),(0,0)]}
        else if head_screen {&[(0,0),(1,0),(1,1),(0,1),(0,0)]}
        else {&[(0,0),(1,0),(2,0),(2,1),(1,1),(0,1),(0,0)]};
    for (slot,&(policy,sample)) in order.iter().enumerate() {
        anchor=session.restore_state_image(verify,&saved,750000+3*slot as u64,comm,stream)?;
        ensure!(verify.position()==frontier,"depth restored frontier");
        let request_seed=seed+1+sample;
        session.configure_sampling(request_seed,None)?;
        let threshold=if head_screen || markov_screen || private_gemv_screen || history_match_screen || confidence_screen {Some(0.20)} else {
            match policy {0=>Some(0.20),1=>Some(0.10),2=>None,_=>unreachable!()}};
        session.set_causal_three_threshold_for_probe(threshold)?;
        if confidence_screen {session.set_fused_confidence_for_probe(policy==1,stream)?;}
        if markov_screen {session.set_fused_markov_head_for_probe(policy==1,stream)?;}
        let history_maximum=if policy==0 {16}else{64};
        let history_workspace_bytes=if history_match_screen {Some(session.configure_history_maximum_for_probe(history_maximum)?)}else{None};
        let head_warmup=if head_screen || private_gemv_screen {
            if private_gemv_screen {session.set_private_head_gemv_for_probe(policy==1,stream)?;}
            else {
                session.set_private_draft_head_for_probe(policy==0,stream)?;
                if head_gemv_screen {session.set_draft_head_gemv_for_probe(policy==2,stream)?;}
            }
            // Head changes invalidate only the draft query graph. Warm it
            // directly so a history lookup cannot skip the selected head.
            // Then restore all state before measuring the real policy.
            let began=Instant::now();
            session.warm_draft_head_for_probe(anchor,comm,stream)?;
            let warm=json!({"direct_draft_query_warmed":true,"seconds":began.elapsed().as_secs_f64()});
            anchor=session.restore_state_image(verify,&saved,750001+3*slot as u64,comm,stream)?;
            session.configure_sampling(request_seed,None)?;
            session.set_causal_three_threshold_for_probe(threshold)?;
            ensure!(verify.position()==frontier,"head warmup restore frontier");
            Some(warm)
        }else{None};
        let trajectory_warmup=if warm_trajectory {
            let before=physical_read_bytes()?;
            let warm=if replay_trace && slot==5 {crate::history_policy_probe::advance_traced(
                &mut session,verify,&mut anchor,256,comm,stream,&output.join("replay-warm"))?}
                else{advance(&mut session,verify,&mut anchor,256,comm,stream)?};
            let reads=physical_read_bytes()?.checked_sub(before).context("warm I/O counter regressed")?;
            anchor=session.restore_state_image(verify,&saved,750002+3*slot as u64,comm,stream)?;
            session.configure_sampling(request_seed,None)?;
            session.set_causal_three_threshold_for_probe(threshold)?;
            ensure!(verify.position()==frontier,"trajectory warmup restore frontier");
            if history_match_screen {session.configure_history_maximum_for_probe(history_maximum)?;}
            Some(json!({"physical_read_bytes":reads,"continuation":warm}))
        }else{None};
        ensure!(session.block_verification() && session.draft_temperature()==1.2,"depth fixed proposal law");
        stream.join_default()?;
        unsafe{comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)?;}
        stream.synchronize()?;comm.health()?;
        let before=physical_read_bytes()?;
        let continuation=if replay_trace && slot==5 {crate::history_policy_probe::advance_traced(
            &mut session,verify,&mut anchor,256,comm,stream,&output.join("replay-measured"))?}
            else{advance(&mut session,verify,&mut anchor,256,comm,stream)?};
        let physical_reads=physical_read_bytes()?.checked_sub(before).context("timed I/O counter regressed")?;
        let signature=decision_signature(&continuation);
        let trajectory_warmup_exact=trajectory_warmup.as_ref()
            .is_none_or(|warm|decision_signature(&warm["continuation"])==signature);
        if slot==0 {control=Some(signature.clone());}
        let repeated_control=slot+1==order.len();
        let control_exact=!repeated_control || control.as_ref()==Some(&signature);
        let name=if confidence_screen {["confidence_vendor","confidence_fused_hidden_latent"][policy]}
            else if history_match_screen {["history_max16","history_max64"][policy]}
            else if private_gemv_screen {["private_fp8_tensor","private_fp8_gemv"][policy]}
            else if markov_screen {["markov_vendor","markov_fused_n16"][policy]}
            else if head_screen {["draft_fp8","draft_bf16","draft_bf16_gemv_m5"][policy]}
            else {["causal3_t02","causal3_t01","fixed6"][policy]};
        let record=json!({"rank":plan.rank,"slot":slot,"policy":policy,
            "name":name,"sample":sample,"head_kind":session.draft_head_kind(),"head_warmup":head_warmup,
            "fused_markov_head":session.fused_markov_head(),"fused_confidence":session.fused_confidence(),
            "history_maximum":history_match_screen.then_some(history_maximum),"history_workspace_bytes":history_workspace_bytes,
            "trajectory_warmup":trajectory_warmup,"trajectory_warmup_exact":trajectory_warmup_exact,
            "physical_read_bytes":physical_reads,
            "seed":request_seed,"start_frontier":frontier,"threshold":threshold,
            "repeated_control":repeated_control,"control_exact":control_exact,
            "block_verification":session.block_verification(),"draft_temperature":session.draft_temperature(),
            "continuation":continuation});
        let prefix=if confidence_screen {"confidence-fusion"}else if history_match_screen {"history-match"}else if private_gemv_screen {"private-head"}else if markov_screen {"markov-head"}else if head_screen {"draft-head"}else{"draft-depth"};
        std::fs::write(output.join(format!("{prefix}-slot{slot}.json")),serde_json::to_vec_pretty(&record)?)?;
        if !control_exact {
            std::fs::write(output.join(format!("{prefix}-repeat-failure.json")),
                serde_json::to_vec_pretty(&json!({"first":control,"repeated":signature}))?)?;
        }
        // Retain observed trajectory differences instead of confusing seeded
        // reproducibility with correctness. Cross-rank consistency and graph
        // reuse remain mandatory; the summary reports any unmatched repeats.
        ensure!((markov_screen || private_gemv_screen || history_match_screen || confidence_screen || (control_exact && trajectory_warmup_exact)) && continuation["new_verifier_captures"]==0,
            "depth repeated state, warmed trajectory, or graph reuse");
        eprintln!("draft depth rank{} slot{slot} policy{policy}: {:.2}tok/s",plan.rank,
            continuation["tokens_per_second"].as_f64().unwrap());
        records.push(record);
    }
    let selector_continuation=if selector_capture {
        anchor=session.restore_state_image(verify,&saved,750100,comm,stream)?;
        session.configure_sampling(seed+2,None)?;
        session.set_fused_markov_head_for_probe(false,stream)?;
        session.set_causal_three_threshold_for_probe(Some(0.20))?;
        Some(crate::history_policy_probe::advance_selector_capture(&mut session,verify,&mut anchor,160,
            comm,stream,(plan.rank==0).then_some(output.join("selector-step39")).as_deref())?)
    }else{None};
    let repeats_exact=records.iter().all(|r|r["control_exact"]==true && r["trajectory_warmup_exact"]==true);
    Ok(json!({"rank":plan.rank,"draft_depth_screen":!head_screen && !markov_screen && !private_gemv_screen && !history_match_screen && !confidence_screen,"draft_head_screen":head_screen,
        "history_match_screen":history_match_screen,"confidence_fusion_screen":confidence_screen,
        "private_head_gemv_screen":private_gemv_screen,
        "markov_head_screen":markov_screen,"selector_capture":selector_capture,"selector_continuation":selector_continuation,
        "head_repeat":head_repeat,"head_gemv_screen":head_gemv_screen,"replay_trace":replay_trace,"samples_per_policy":if head_repeat {4}else{2},"fixture_sha256":fixture_sha,
        "records":records,"prefill_seconds":prefill_seconds,"warmup":warmup,"image_bytes":image_bytes,
        "start_frontier":frontier,"context_capacity":1048576,"inputs_per_trial":256,
        "all_repeated_controls_exact":repeats_exact,"full_engine_acceptance":false,
        "scope":if confidence_screen {"One262K prefix, two warmed256-input seed pairs plus repeated native control. Fuse canonical confidence hidden projection and causal previous-token embedding contribution, FP32 output/sigmoid; target math and actual-q unchanged. Carry max64 history matches and fused Markov on both sides. Small native scheduling integration, not outcome/production qualification."}
        else if history_match_screen {"One131K saved prefix, two256-input seed pairs in opposite orders plus repeated16-token-match control. Add32/64-token accepted-history indexes, unchanged5-token point-mass proposals and target rejection. Each trajectory warmed before restored timing; proposed suffixes are never indexed. Fused Markov/current FP8 tensor head retained in both. Small algorithm screen; no outcome or production qualification."}
        else if private_gemv_screen {"One262K saved prefix, two256-input seed pairs in opposite orders plus repeated original control. Current private FP8 quantization/BF16 dequantized operands, FP32 output: original tensor kernel versus direct GEMV. Fused Markov head retained in both; target precision and actual-q rejection unchanged. Each trajectory warmed before restored timing. Small component-to-model screen, not outcome/quality qualification."}
        else if markov_screen {"One262K saved prefix, two256-input seed pairs in opposite orders plus repeated vendor control. Each trajectory warmed before restored timing. Fused canonical BF16 Markov projection with FP32 output/base add versus vendor projection and separate add. Actual-q rejection and target math unchanged. Optional selector capture occurs after all timing; no general outcome/quality claim."}
        else if head_gemv_screen {"One262K saved prefix, two256-input seeds in opposite orders: FP8, vendor BF16, and M5 GEMV BF16 draft heads plus repeated FP8 control. Each short trajectory warmed then restored exactly before timing. Canonical target path and actual-q rejection unchanged. Small component-to-engine screen, not outcome or production qualification."}
        else if head_repeat {"Follow-up to s444 bracket drift: one262K saved prefix, FP8 versus BF16 draft head, four seeds in opposite orders plus FP8 control. Each256-input trajectory is warmed, state restored, and repeated for timing; full token/decision/state signatures must match, and physical read bytes are recorded. Target canonical precision and actual-q block rejection atT1.2 unchanged. Small warmed continuation screen, not a general speed/outcome claim."}
        else if head_screen {"One262K saved prefix: FP8 versus canonical BF16 drafter head, two seeds in opposite orders plus repeated FP8 control,256committed inputs each. Direct draft-query warmup before each trial then exact state restore; no graph capture inside timing. Target canonical BF16/FP32 unchanged; block verification uses actual proposal q at draftT1.2. A small hypothesis screen, not outcome or production qualification."}
        else {"One262K prefix, one compact snapshot after24warm inputs: existing causal3/6 threshold0.2, threshold0.1, and fixed6. Two seeds in opposite policy orders plus one exact repeated control,256committed inputs each. Same block verification, actual-q draftT1.2,256K history tables, targetT1/topP1 and canonical precision. A short policy screen; trajectories differ and this is not outcome or production qualification."}}))
}
