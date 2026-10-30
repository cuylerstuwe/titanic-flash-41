//! A few proposal-source variants from identical saved accepted frontiers.
//! Snapshot reuse avoids another long prefill for each scheduling hypothesis.
use crate::{checkpoint::Checkpoint, cuda::{Cuda, Stream},
    dspark_generation::GreedySession, layout::RankPlan, roce::Roce, target::Target};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

pub(crate) fn advance(session:&mut GreedySession, target:&mut Target, anchor:&mut u32,
    inputs:usize, comm:&Roce, stream:&Stream)->Result<Value> {
    advance_inner(session,target,anchor,inputs,comm,stream,None,None)
}
pub(crate) fn advance_traced(session:&mut GreedySession,target:&mut Target,anchor:&mut u32,
    inputs:usize,comm:&Roce,stream:&Stream,trace:&Path)->Result<Value>{
    ensure!(std::env::var("CANADA_SCRATCH_DRAFT_REPLAY_TRACE").as_deref()==Ok("1"),"replay trace admission");
    std::fs::create_dir(trace)?;
    advance_inner(session,target,anchor,inputs,comm,stream,Some(trace),None)
}
pub(crate) fn advance_selector_capture(session:&mut GreedySession,target:&mut Target,anchor:&mut u32,
    inputs:usize,comm:&Roce,stream:&Stream,root:Option<&Path>)->Result<Value>{
    ensure!(std::env::var("CANADA_SCRATCH_SELECTOR_CAPTURE").as_deref()==Ok("1"),"selector capture admission");
    advance_inner(session,target,anchor,inputs,comm,stream,None,root)
}
fn advance_inner(session:&mut GreedySession,target:&mut Target,anchor:&mut u32,
    inputs:usize,comm:&Roce,stream:&Stream,trace:Option<&Path>,selector:Option<&Path>)->Result<Value>{
    let start=target.position();let captures=target.capture_statistics().1;
    let began=Instant::now();let mut ids=Vec::new();let mut steps=Vec::new();
    while ids.len()<inputs {
        let step=session.step(target,*anchor,(inputs-ids.len()).min(6),comm,stream)?;
        ensure!(step.tokens[0]==*anchor,"history policy pending anchor");
        if let Some(root)=selector.filter(|_|steps.len()==39) {
            ensure!(step.verified_rows==6,"selector capture requires Verify6");
            target.dump_selector_trace(root,stream)?;
        }
        if let Some(trace)=trace.filter(|_|steps.len()<=45) {
            let slot=steps.len();let raw=trace.join(format!("raw-step{slot}"));
            let state=session.replay_digests(target,(38..=41).contains(&slot).then_some(raw.as_path()),stream)?;
            std::fs::write(trace.join(format!("step{slot}.json")),serde_json::to_vec_pretty(&json!({"step":step,"state":state}))?)?;
        }
        ids.extend_from_slice(&step.tokens);*anchor=step.next_anchor;steps.push(step);
    }
    stream.synchronize()?;comm.health()?;
    let seconds=began.elapsed().as_secs_f64();
    ensure!(target.position()==start+inputs as u64,"history policy committed frontier");
    Ok(json!({"seconds":seconds,"inputs":inputs,"tokens_per_second":inputs as f64/seconds,
        "committed_inputs":ids,"pending_anchor":anchor,"steps":steps,
        "new_verifier_captures":target.capture_statistics().1-captures,
        "frontier":target.position(),"final_outputs":target.valid_output_digests()?,
        "carry":target.carry_digests()?,"encoder":target.encoder_tail_digest()?}))
}

pub(crate) fn decision_signature(record:&Value)->Value {
    json!({"committed_inputs":record["committed_inputs"],"pending_anchor":record["pending_anchor"],
        "steps":record["steps"].as_array().unwrap().iter().map(|s|json!({
            "tokens":s["tokens"],"next_anchor":s["next_anchor"],"draft_tokens":s["draft_tokens"],
            "matching_proposals":s["matching_proposals"],"draft_confidence":s["draft_confidence"],
            "lookup_match_tokens":s["lookup_match_tokens"],"verified_rows":s["verified_rows"],
            "proposal_source":s["proposal_source"],"lookup_source_end":s["lookup_source_end"],
            "position":s["position"]})).collect::<Vec<_>>(),
        "frontier":record["frontier"],"final_outputs":record["final_outputs"],
        "carry":record["carry"],"encoder":record["encoder"]})
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,core:&[u8],stream:&Stream,cp:&Checkpoint,plan:&RankPlan,
    large:&mut Target,verify:&mut Target,wides:&mut[Target],comm:&Roce,
    staged:&Path,kernels:&Path,output:&Path)->Result<Value> {
    let profiles=wides.iter().filter_map(Target::wide_report).collect::<Vec<_>>();
    let history_screen=std::env::var("CANADA_SCRATCH_HISTORY_POLICY_SCREEN").as_deref()==Ok("1");
    let temperature_screen=std::env::var("CANADA_SCRATCH_DRAFT_TEMPERATURE_SCREEN").as_deref()==Ok("1");
    let table_screen=std::env::var("CANADA_SCRATCH_HISTORY_TABLE_SCREEN").as_deref()==Ok("1");
    let warm_repeats=temperature_screen || table_screen;
    ensure!(history_screen!=temperature_screen
        && (!table_screen || history_screen)
        && std::env::var("CANADA_SCRATCH_COMPACT_STATE_IMAGE").as_deref()==Ok("1")
        && std::env::var("CANADA_SCRATCH_ACCEPTANCE_SHADOW").as_deref()!=Ok("1")
        && std::env::var("CANADA_SCRATCH_HISTORY_OPPORTUNITY").as_deref()!=Ok("1")
        && large.context_capacity()==1048576 && verify.head_gemv_enabled()
        && verify.engram_input_kind()=="mapped" && verify.paired_engram_io_enabled()
        && verify.adaptive_sparse_index_layers()==4 && verify.index_key_splits()==[24;8]
        && profiles.iter().map(|p|p["rows"].as_u64().unwrap_or(0)).collect::<Vec<_>>()
            ==vec![8192,4096,2048,1024,512],"history policy needs the current full candidate");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture_sha=format!("{:x}",Sha256::digest(raw));
    ensure!(fixture_sha=="107ceccbefc53247fc827cdee7eb3908f7b89a907d8c7a2d3de3ef6b95057fbb",
        "history policy fixture hash");
    let fixture:Value=serde_json::from_slice(raw)?;
    let corpus:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let teacher:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/fixed-verify-python-v1.json"))?;
    let teacher:Vec<u32>=serde_json::from_value(teacher["teacher_forced_tokens"].clone())?;
    let seed=std::env::var("CANADA_SCRATCH_TARGET_SEED")?.parse::<u64>()?;
    let mut barrier_in=cuda.allocate::<f32>(4)?;barrier_in.upload(&[0.;4])?;
    let mut barrier_out=cuda.allocate::<f32>(16)?;
    let mut session:Option<GreedySession>=None;let mut lineage=190000;let mut records=Vec::new();
    let mut contexts=Vec::new();large.enable_cached_extend_floor()?;
    let context_points:&[(usize,usize)]=if warm_repeats {&[(2,262144)]}
        else {&[(1,131072),(2,262144)]};
    for &(index,length) in context_points {
        let warm_seed=seed.wrapping_add((index*100)as u64);let request_seed=warm_seed.wrapping_add(1);
        let began=Instant::now();large.prefill_wide(wides,&corpus[..length],comm,stream)?;
        let mut anchor=large.sample_token(warm_seed,stream)?;
        let prefill_seconds=began.elapsed().as_secs_f64();
        if let Some(s)=session.as_mut(){s.reset_from_prefill(large,cuda,core,stream)?;}
        else{session=Some(GreedySession::from_prefill(large,cuda,core,stream,cp,plan,staged,kernels)?);}
        let s=session.as_mut().unwrap();s.configure_sampling(warm_seed,None)?;
        s.configure_algorithm_for_probe(true,1.,stream)?;
        s.set_causal_three_threshold_for_probe(Some(0.20))?;
        s.configure_history_lookup(large,&corpus[..length])?;verify.handoff_from(large)?;
        ensure!(s.conditioning_split()&&s.block_verification()&&s.draft_temperature()==1.,"history policy algorithm");
        // Record all live verification graphs, and execute a short real prefix
        // to establish the drafter transport and graph before saving the image.
        for live in 1..=6 {
            verify.begin_verification(&teacher[..live],comm,stream)?;
            verify.verification_greedy(stream)?;verify.finish_verification(0,comm,stream)?;
        }
        let warmup=advance(s,verify,&mut anchor,24,comm,stream)?;
        s.configure_sampling(request_seed,None)?;
        let (image_bytes,_)=s.compact_state_image_bytes(verify)?;
        ensure!(image_bytes<=768*1024usize.pow(2),"history policy snapshot exceeds bounded budget");
        let began=Instant::now();let saved=s.save_state_image(verify,anchor,image_bytes,comm,stream)?;
        let save_seconds=began.elapsed().as_secs_f64();
        ensure!(saved.manifest()["target"]["compact"]==true,"history policy saved a full image");
        let frontier=verify.position();let mut control:Option<Value>=None;
        let order=if table_screen {
            vec![(Some(8),1.,"slots65536",true),(Some(8),1.,"slots262144",true),
                (Some(8),1.,"slots1048576",true),(Some(8),1.,"slots65536",false),
                (Some(8),1.,"slots1048576",false),(Some(8),1.,"slots262144",false),
                (Some(8),1.,"slots65536",false)]
        }else if temperature_screen {
            // Warm each complete trajectory, then bracket the two measured
            // candidates with controls. Snapshot restoration includes the RNG
            // seed and accepted history; target probabilities are unchanged.
            vec![(Some(8),1.,"draft_t1",true),(Some(8),0.7,"draft_t07",true),
                (Some(8),1.2,"draft_t12",true),(Some(8),1.,"draft_t1",false),
                (Some(8),1.2,"draft_t12",false),(Some(8),0.7,"draft_t07",false),
                (Some(8),1.,"draft_t1",false)]
        }else{
            let minima=if index==1 {[Some(8),Some(12),None,Some(8)]}
                else {[Some(8),None,Some(12),Some(8)]};
            minima.into_iter().map(|n|(n,1.,match n{Some(8)=>"history8",Some(12)=>"history12",None=>"neural_only",_=>unreachable!()},false)).collect()
        };
        let mut warm_signatures=std::collections::BTreeMap::new();
        for (slot,(minimum,temperature,name,warm_trial)) in order.into_iter().enumerate() {
            lineage+=1;let began=Instant::now();
            anchor=s.restore_state_image(verify,&saved,lineage,comm,stream)?;
            let restore_seconds=began.elapsed().as_secs_f64();
            ensure!(verify.position()==frontier,"history policy restored wrong frontier");
            s.configure_algorithm_for_probe(true,temperature,stream)?;
            s.configure_history_policy_for_probe(minimum)?;
            let table_slots=if table_screen {
                let slots=name.strip_prefix("slots").unwrap().parse::<usize>()?;
                s.configure_history_tables_for_probe(slots)?;slots
            }else{65536};
            stream.join_default()?;
            unsafe{comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)?;}
            stream.synchronize()?;comm.health()?;
            let continuation=advance(s,verify,&mut anchor,512,comm,stream)?;
            ensure!(continuation["new_verifier_captures"]==0,"history policy measured path captured");
            let signature=decision_signature(&continuation);
            let mut repeat_checks=None;
            if warm_repeats {
                if warm_trial {warm_signatures.insert(name,signature.clone());}
                else {
                    let reference=&warm_signatures[name];
                    repeat_checks=Some(json!(["committed_inputs","pending_anchor","steps","frontier","final_outputs","carry","encoder"]
                        .into_iter().map(|key|(key.to_owned(),json!(reference[key]==signature[key])))
                        .collect::<serde_json::Map<String,Value>>()));
                    if reference!=&signature {
                        std::fs::write(output.join(format!("history-policy-repeat-failure-slot{slot}.json")),
                            serde_json::to_vec_pretty(&json!({"name":name,"slot":slot,"checks":repeat_checks,
                                "warm_signature":reference,"measured_signature":signature}))?)?;
                    }
                }
            }
            let first_control=if warm_repeats{3}else{0};
            let last_control=if warm_repeats{6}else{3};
            if slot==first_control {control=Some(signature.clone());}
            let control_exact=slot!=last_control || control.as_ref()==Some(&signature);
            let record=json!({"rank":plan.rank,"prompt_index":index,"prompt_tokens":length,
                "start_frontier":frontier,"seed":request_seed,"slot":slot,"name":name,
                "minimum_match":minimum,"table_slots":table_slots,"repeated_control":slot==last_control,"warmup":warm_trial,
                "restore_seconds":restore_seconds,"repeat_checks":repeat_checks,"control_exact":control_exact,
                "block_verification":s.block_verification(),"draft_temperature":s.draft_temperature(),
                "continuation":continuation});
            std::fs::write(output.join(format!("history-policy-p{index}-slot{slot}.json")),serde_json::to_vec_pretty(&record)?)?;
            ensure!(repeat_checks.as_ref().is_none_or(|checks|checks.as_object().unwrap().values().all(|v|v==true)),
                "policy warm/measured replay checks: {repeat_checks:?}");
            ensure!(control_exact,"history policy repeated control changed tokens or retained state");
            eprintln!("history policy rank{} prompt{length} slot{slot} {name}: {:.2}tok/s",plan.rank,
                continuation["tokens_per_second"].as_f64().unwrap());
            records.push(record);
        }
        contexts.push(json!({"prompt_tokens":length,"prefill_seconds":prefill_seconds,
            "saved_frontier":frontier,"image_bytes":image_bytes,"save_seconds":save_seconds,
            "warmup":warmup,"repeated_control_exact":true,"memory":cuda.admission_memory()?}));
        drop(saved);lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;
    }
    Ok(json!({"rank":plan.rank,"history_policy_screen":history_screen,
        "draft_temperature_screen":temperature_screen,"history_table_screen":table_screen,"records":records,"contexts":contexts,
        "fixture_sha256":fixture_sha,"context_capacity":large.context_capacity(),"temperature":1.,"top_p":1.,
        "inputs_per_trial":512,"encoder_profiles":profiles,"index_key_splits":verify.index_key_splits(),
        "target_head":"canonical_bf16_weights_fp32_output","all_repeated_controls_exact":true,
        "full_engine_acceptance":false,
        "scope":if table_screen {"Three exact-token history table sizes at262K from one compact snapshot:65536,262144,1048576slots per match length. Each512-input trajectory runs warm, then measured control/largest/middle/control. Only accepted-token index collision retention changes; minimum match8, block/T1 and target precision stay unchanged. Each measured path must replay its warm path exactly. Different valid trajectories, one prefix/seed, no outcome or production qualification."}
        else if temperature_screen {"Three drafter temperatures at262K from one identical compact snapshot after24warmup inputs:1.0,0.7,1.2. Each512-input trajectory runs once warm, followed by measuredT1/T1.2/T0.7/T1. Only the proposal sampler/actual-q temperature changes; target remainsT1/topP1 with canonical precision and block rejection. Every measured trajectory must match its own warm run exactly. Internal fixed-length timing on one accepted prefix/seed, not completed outcomes or production qualification."}
        else {"Three history-source choices at128K/262K from identical compact snapshots after24warmup inputs: matchminimum8,minimum12,and neural-only. One512-input continuation per candidate plus repeated8-token control at each context. Current block/T1,actual-q rejection and unchanged canonical target precision. History suppression uses accepted prefix only; no target lookahead. Internal fixed-length timing, not completed outcomes, a matched SGLang comparison, or production qualification."}}))
}
