//! First two-owner target prototype: separate causal state, shared weights.
//! No batching or serving policy is enabled by this finite ownership screen.
use super::*;

#[path = "paired_generation_probe.rs"]
mod pair_generation;
#[path = "paired_serving_profile_probe.rs"]
mod pair_serving;
#[path = "paired_endpoint_probe.rs"]
mod pair_endpoints;
#[path = "paired_endpoint_matched_probe.rs"]
mod pair_endpoints_matched;
#[path = "encoder_import_probe.rs"]
mod encoder_import;
#[path = "bank_encoder_probe.rs"]
mod bank_encoder;

fn logits(target:&mut Target,ids:&[u32],accepted:usize,cuda:&Cuda,
    comm:&Roce,stream:&Stream)->Result<String> {
    target.begin_verification(ids,comm,stream)?;
    let(y,live)=target.pending_local_logits_for_probe()?;
    ensure!(live==ids.len(),"independent owner logit extent");
    let hash=format!("{:x}",Sha256::digest(unsafe{cuda.read_device(y.ptr(),live*32320*4)}?));
    target.finish_verification(accepted,comm,stream)?;
    Ok(hash)
}

fn state(target:&mut Target,teacher:&[u32],cuda:&Cuda,comm:&Roce,stream:&Stream)->Result<Value> {
    let following=logits(target,teacher,0,cuda,comm,stream)?;
    Ok(json!({"position":target.position(),"following_logits":following,
        "carry":target.carry_digests()?,"encoder":target.encoder_tail_digest()?,
        "regions":target.repeat_state_digests(stream)?}))
}

fn step(target:&mut Target,ids:&[u32],accepted:usize,teacher:&[u32],
    cuda:&Cuda,comm:&Roce,stream:&Stream)->Result<Value> {
    let forward=logits(target,ids,accepted,cuda,comm,stream)?;
    Ok(json!({"forward_logits":forward,"accepted":accepted,
        "state":state(target,teacher,cuda,comm,stream)?}))
}

pub(super) fn run(w:&mut Worker,rank:usize,kernels:&Path,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_INDEPENDENT_OWNER_SCREEN").as_deref()==Ok("1")
        && w.current.is_none() && w.large.position()==0
        && Target::independent_probe_context_admitted(w.large.context_capacity()),
        "finite independent target owner admission");
    let before=w.cuda.admission_memory()?;
    let module=w.cuda.module(&w.core)?;
    let loan=std::env::var("CANADA_SCRATCH_NARROW_ARENA_LOAN_SCREEN").as_deref()==Ok("1");
    let donor=if loan{Some(w.wides.iter().max_by_key(|t|t.physical_rows()).context("complete peer donor bank")?)}else{None};
    let mut other=w.large.independent_owner_for_probe(&module,&w.core,&w.stream,kernels,100,donor)?;
    let mut verify=if let Some(donor)=donor {
        other.fork_profile_lending_prefill_for_probe(&module,&w.core,&w.stream,kernels,6,Phase::Verify,donor)?
    }else{other.fork_profile(&module,&w.core,&w.stream,kernels,6,Phase::Verify)?};
    verify.set_graph_replay(true)?;
    verify.apply_candidate_schedule(&w.stream)?;
    let ownership=w.large.independent_state_report(&other)?;
    ensure!(ownership["independent_causal_owners"]==true
        && other.shared_state_report(&verify)["all_weights_and_causal_storage_shared"]==true,
        "independent state or same-owner profile identity");
    let after=w.cuda.admission_memory()?;
    ensure!(w.control.all_admit(after.admitted_available>=8*1024usize.pow(3))?,
        "two target owners consumed physical reserve");
    if std::env::var("CANADA_SCRATCH_BANK_ENCODER_SCREEN").as_deref()==Ok("1") {
        return bank_encoder::run(w,&mut other,&mut verify,rank,out,ownership);
    }
    if std::env::var("CANADA_SCRATCH_ENCODER_PREFIX_IMPORT_SCREEN").as_deref()==Ok("1") {
        return encoder_import::run(w,&mut other,&mut verify,rank,out,ownership);
    }
    if std::env::var("CANADA_SCRATCH_PAIRED_FFN_SCREEN").as_deref()==Ok("1") {
        return paired_ffn(w,&mut other,&mut verify,rank,kernels,out,ownership);
    }
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let prompts=[&tokens[..257],&tokens[19000..19389]];
    let teacher=&tokens[250000..250006];
    let inputs=[vec![(&tokens[21000..21006],3),(&tokens[21020..21026],0),(&tokens[21040..21046],6)],
        vec![(&tokens[22000..22006],2),(&tokens[22020..22023],0),(&tokens[22040..22046],6)]];
    let mut controls=Vec::new();
    // Obtain both controls through the original request owner and schedule.
    for owner in 0..2 {
        w.large.reset(200+owner as u64,&w.comm,&w.stream)?;
        w.large.prefill_wide(&mut [],prompts[owner],&w.comm,&w.stream)?;
        w.verify.handoff_from(&w.large)?;
        let initial=state(&mut w.verify,teacher,&w.cuda,&w.comm,&w.stream)?;
        let mut steps=Vec::new();
        for &(ids,accepted) in &inputs[owner] {
            steps.push(step(&mut w.verify,ids,accepted,teacher,&w.cuda,&w.comm,&w.stream)?);
        }
        controls.push(json!({"initial":initial,"steps":steps}));
        w.large.handoff_from(&w.verify)?;
    }
    w.large.reset(300,&w.comm,&w.stream)?;
    other.reset(301,&w.comm,&w.stream)?;
    w.large.prefill_wide(&mut [],prompts[0],&w.comm,&w.stream)?;
    w.verify.handoff_from(&w.large)?;
    let first=state(&mut w.verify,teacher,&w.cuda,&w.comm,&w.stream)?;
    let unchanged=w.verify.repeat_state_digests(&w.stream)?;
    other.prefill_wide(&mut [],prompts[1],&w.comm,&w.stream)?;
    verify.handoff_from(&other)?;
    let second=state(&mut verify,teacher,&w.cuda,&w.comm,&w.stream)?;
    ensure!(w.control.all_admit(first==controls[0]["initial"]&&second==controls[1]["initial"]
        && w.verify.repeat_state_digests(&w.stream)?==unchanged)?,
        "independent prefill changed its peer or matched-control result");
    let mut records=Vec::new();
    for index in 0..3 {
        // Reverse execution order in the middle; test both ownership directions.
        for owner in if index==1 {[1usize,0]}else{[0,1]} {
            let(active,peer)=if owner==0 {(&mut w.verify,&mut verify)}else{(&mut verify,&mut w.verify)};
            let untouched=peer.repeat_state_digests(&w.stream)?;
            let(ids,accepted)=inputs[owner][index];
            let result=step(active,ids,accepted,teacher,&w.cuda,&w.comm,&w.stream)?;
            let exact=result==controls[owner]["steps"][index];
            let isolated=untouched==peer.repeat_state_digests(&w.stream)? && peer.profile_is_current();
            records.push(json!({"owner":owner,"step":index,"exact":exact,"isolated":isolated,"result":result}));
            std::fs::write(out.join("independent-owner-progress.json"),serde_json::to_vec_pretty(&records)?)?;
            ensure!(w.control.all_admit(exact&&isolated)?,"independent partial commit or rollback changed model state");
        }
    }
    // Cross-owner restore must still fail before mutating either owner.
    let saved=w.verify.save_state_image(256<<20,&w.comm,&w.stream)?;
    let untouched=verify.repeat_state_digests(&w.stream)?;
    let refused=verify.restore_state_image(&saved,500,&w.comm,&w.stream).is_err();
    ensure!(w.control.all_admit(refused && untouched==verify.repeat_state_digests(&w.stream)?
        &&verify.profile_is_current()&&w.verify.profile_is_current())?,"foreign snapshot crossed owner boundary");
    let first=w.verify.repeat_state_digests(&w.stream)?;
    verify.reset(501,&w.comm,&w.stream)?;
    ensure!(w.control.all_admit(first==w.verify.repeat_state_digests(&w.stream)?)?,
        "reset changed the other live request");
    std::fs::write(out.join("independent-owner-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"ownership":ownership,"memory_before":before,"memory_after":after,
        "prompt_tokens":[prompts[0].len(),prompts[1].len()],"controls":controls,"records":records,
        "all_exact":true,"foreign_restore_refused":true,"peer_reset_isolated":true,
        "scope":"Two independent target owners over shared canonical weights,32K allocation. Distinct public257/389token prefixes, six interleaved partial/full/rejected verifier blocks, full local vocabulary on all four ranks, causal regions and followingM6. Not paired computation, drafter ownership, image execution,1M memory admission or serving qualification."
    }))?)?;
    Ok(())
}

fn paired_ffn(w:&mut Worker,other:&mut Target,verify:&mut Target,rank:usize,
    kernels:&Path,out:&Path,ownership:Value)->Result<()> {
    let full_model=std::env::var("CANADA_SCRATCH_PAIRED_MODEL_SCREEN").as_deref()==Ok("1");
    let variable=std::env::var("CANADA_SCRATCH_PAIRED_VARIABLE_SCREEN").as_deref()==Ok("1");
    ensure!(!variable||full_model,"variable pairing requires whole-model screen");
    let label=if full_model {"paired-model"}else{"paired-ffn"};
    let module=w.cuda.module(&w.core)?;
    let mut wides=if pair_serving::enabled() {
        other.fork_peer_wide_bank_for_probe(&module,&w.core,&w.stream,kernels,&w.large,&w.wides)?
    }else{other.fork_wide_bank(&module,&w.core,&w.stream,kernels,&[512],true)?};
    let pooled=crate::ffn::Paired::new(&w.cuda,&w.core,kernels)?;
    let memory=w.cuda.admission_memory()?;
    ensure!(w.control.all_admit(memory.admitted_available>=8*1024usize.pow(3))?,
        "paired FFN prototype consumed physical reserve");
    let prompts:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/inherited-route-pair-v1.json"))?;
    let frontiers:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/paired-ffn-frontiers-v1.json"))?;
    let mut cases=Vec::new();
    for (prompt,frontier)in prompts["cases"].as_array().context("public prompt cases")?.iter()
        .zip(frontiers["cases"].as_array().context("public frontier cases")?) {
        ensure!(prompt["id"]==frontier["id"],"public paired FFN fixture identity");
        let mut ids:Vec<u32>=serde_json::from_value(prompt["input_ids"].clone())?;
        let tail:Vec<u32>=serde_json::from_value(frontier["accepted_tail"].clone())?;
        let next:Vec<u32>=serde_json::from_value(frontier["verify_ids"].clone())?;
        ensure!(tail.len()==32&&next.len()==6,"bounded paired FFN frontier");
        ids.extend(tail);cases.push((ids,next));
    }
    ensure!(cases.len()==3,"three paired FFN public histories");
    if std::env::var("CANADA_SCRATCH_PAIRED_GENERATION_SCREEN").as_deref()==Ok("1") {
        if std::env::var("CANADA_SCRATCH_PAIRED_LOOKUP_SCREEN").as_deref()==Ok("1") {
            ensure!(std::env::var("CANADA_SCRATCH_PAIRED_LONG_CONTEXT_SCREEN").as_deref()!=Ok("1"),
                "separate small lookup and long-context screens");
            let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/history-short-match-frontiers-v1.json"))?;
            for case in fixture["cases"].as_array().context("public copy frontiers")? {
                let tokens:Vec<u32>=serde_json::from_value(case["input_ids"].clone())?;
                let teacher:Vec<u32>=serde_json::from_value(case["teacher_ids"].clone())?;
                ensure!(tokens.len()<32768&&teacher.len()>=6,"bounded public copy frontier");
                cases.push((tokens,teacher[..6].to_vec()));
            }
            ensure!(cases.len()==5,"two public copy frontiers");
        }
        if std::env::var("CANADA_SCRATCH_PAIRED_LONG_CONTEXT_SCREEN").as_deref()==Ok("1") {
            let long:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/paired-generation-long-v1.json"))?;
            let teacher=cases[0].1.clone();
            for case in long["cases"].as_array().context("long paired public histories")? {
                let tokens:Vec<u32>=serde_json::from_value(case["input_ids"].clone())?;
                ensure!((128000..=132000).contains(&tokens.len())
                    &&tokens.len()+1024<w.large.context_capacity(),"long pair fixture/context");
                cases.push((tokens,teacher.clone()));
            }
            ensure!(cases.len()==5,"two long pair public histories");
        }
        return pair_generation::run(w,other,verify,&mut wides,pooled,&cases,rank,kernels,out,ownership);
    }
    if std::env::var("CANADA_SCRATCH_PAIRED_ACTUAL_SCREEN").as_deref()==Ok("1") {
        return paired_actual(w,other,verify,&mut wides,pooled,&cases,rank,out,ownership,json!(memory));
    }
    let mut records=Vec::new();
    for (index,(name,peer))in [("inherited",1usize),("independent",2)].into_iter().enumerate() {
        w.large.reset(600+index as u64*2,&w.comm,&w.stream)?;
        other.reset(601+index as u64*2,&w.comm,&w.stream)?;
        w.large.prefill_wide(&mut w.wides,&cases[0].0,&w.comm,&w.stream)?;
        w.verify.handoff_from(&w.large)?;
        other.prefill_wide(&mut wides,&cases[peer].0,&w.comm,&w.stream)?;
        verify.handoff_from(other)?;
        let widths:Vec<[usize;2]>=if variable {vec![[6,6],[3,6],[6,3],[3,3]]}else{vec![[6,6]]};
        for [a_live,b_live] in widths {
        let before=[(w.verify.carry_digests()?,w.verify.encoder_tail_digest()?),
            (verify.carry_digests()?,verify.encoder_tail_digest()?)];
        w.verify.begin_verification(&cases[0].1[..a_live],&w.comm,&w.stream)?;
        verify.begin_verification(&cases[peer].1[..b_live],&w.comm,&w.stream)?;
        let component=if full_model {
            w.verify.paired_model_screen(verify,&pooled,&w.comm,&w.stream)?
        }else{w.verify.paired_ffn_screen(verify,&pooled,&w.stream)?};
        w.verify.finish_verification(0,&w.comm,&w.stream)?;
        verify.finish_verification(0,&w.comm,&w.stream)?;
        let after=[(w.verify.carry_digests()?,w.verify.encoder_tail_digest()?),
            (verify.carry_digests()?,verify.encoder_tail_digest()?)];
        ensure!(w.control.all_admit(before==after&&component["all_exact"]==true)?,
            "paired FFN or subsequent zero-accept rollback changed request state");
        records.push(json!({"pair":name,"prompt_tokens":[cases[0].0.len(),cases[peer].0.len()],
            "live_rows":[a_live,b_live],"component":component,"rollback_exact":true}));
        std::fs::write(out.join(format!("{label}-progress.json")),serde_json::to_vec_pretty(&records)?)?;
        }
        w.large.handoff_from(&w.verify)?;other.handoff_from(verify)?;
    }
    std::fs::write(out.join(format!("{label}-summary.json")),serde_json::to_vec_pretty(&json!({
        "rank":rank,"all_exact":true,"full_model":full_model,"ownership":ownership,"memory":memory,"records":records,
        "scope":"Bounded paired FFN or whole-target integration screen on all four TP ranks, two distinct inherited or independent public request states. Teacher-forced reasoning frontiers from prior T1/P1/high runs. No sampler, serving throughput or1Mcapacity claim."
    }))?)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn paired_actual(w:&mut Worker,other:&mut Target,verify:&mut Target,wides:&mut [Target],
    pooled:crate::ffn::Paired,cases:&[(Vec<u32>,Vec<u32>)],rank:usize,out:&Path,
    ownership:Value,memory:Value)->Result<()> {
    let mut executor=crate::target::PairExecutor::new(&w.verify,verify,pooled,&w.comm,&w.stream)?;
    let widths=[[6usize,6],[3,6],[6,3],[3,3],[6,6],[3,6]];
    let accepted=[[3usize,2],[0,6],[6,0],[3,1],[6,6],[1,3]];
    let mut records=Vec::new();
    for (case,(name,peer))in [("inherited",1usize),("independent",2)].into_iter().enumerate() {
        let mut controls=Vec::new();let mut control_times=Vec::new();let mut paired_steps=Vec::new();
        for arm in 0..2 {
            let lineage=700+(case*4+arm*2)as u64;
            w.large.reset(lineage,&w.comm,&w.stream)?;other.reset(lineage+1,&w.comm,&w.stream)?;
            w.large.prefill_wide(&mut w.wides,&cases[0].0,&w.comm,&w.stream)?;
            w.verify.handoff_from(&w.large)?;
            other.prefill_wide(wides,&cases[peer].0,&w.comm,&w.stream)?;
            verify.handoff_from(other)?;
            for (index,([a_live,b_live],[a_accept,b_accept]))in widths.into_iter().zip(accepted).enumerate() {
                // Public teacher-forced tokens change at each frontier. This
                // checks state/metadata and commits, not generated-task quality.
                let a:Vec<_>=cases[0].1.iter().copied().cycle().skip(index).take(a_live).collect();
                let b:Vec<_>=cases[peer].1.iter().copied().cycle().skip(index+1).take(b_live).collect();
                // Exercise a real solo->pair scheduling transition, including
                // one owner retaining the M3 ahead-projection selection while
                // its peer last used M6. Rejection preserves the tested frontier.
                // Keep this outside forward timings and use the same primer
                // for the separate and pooled controls.
                if std::env::var("CANADA_SCRATCH_PAIR_SCHEDULE_TRANSITION_SCREEN").as_deref()==Ok("1") {
                    let prior=if index%2==0 {[3usize,6]}else{[6usize,3]};
                    for (target,teacher,live) in [(&mut w.verify,&cases[0].1,prior[0]),
                        (&mut *verify,&cases[peer].1,prior[1])] {
                        target.begin_verification(&teacher[..live],&w.comm,&w.stream)?;
                        target.finish_verification(0,&w.comm,&w.stream)?;
                    }
                }
                let started=std::time::Instant::now();
                let mut timing=if arm==0 {
                    if index%2==0 {
                        w.verify.begin_verification(&a,&w.comm,&w.stream)?;
                        verify.begin_verification(&b,&w.comm,&w.stream)?;
                    }else{
                        verify.begin_verification(&b,&w.comm,&w.stream)?;
                        w.verify.begin_verification(&a,&w.comm,&w.stream)?;
                    }
                    json!({"total_ms":started.elapsed().as_secs_f64()*1000.})
                }else{executor.begin(&mut w.verify,&a,verify,&b,&w.comm,&w.stream)?};
                let forward=[w.verify.paired_completed_taps()?,verify.paired_completed_taps()?];
                if arm==1&&[0,3].contains(&index)
                    &&std::env::var("CANADA_SCRATCH_PAIRED_DENSE_TIMING_SCREEN").as_deref()==Ok("1") {
                    timing["dense_attribution"]=executor.dense_timing(&mut w.verify,verify,&w.comm,&w.stream)?;
                }
                w.verify.finish_verification(a_accept,&w.comm,&w.stream)?;
                verify.finish_verification(b_accept,&w.comm,&w.stream)?;
                let after=[state(&mut w.verify,&cases[0].1,&w.cuda,&w.comm,&w.stream)?,
                    state(verify,&cases[peer].1,&w.cuda,&w.comm,&w.stream)?];
                let signature=json!({"forward":forward,"state":after,"accepted":[a_accept,b_accept]});
                let exact=arm==0||signature==controls[index];
                std::fs::write(out.join("paired-actual-progress.json"),serde_json::to_vec_pretty(&json!({
                    "pair":name,"arm":arm,"step":index,"exact":exact,"timing":timing,
                    "actual":signature,"expected":if arm==0{None}else{controls.get(index)}
                }))?)?;
                ensure!(w.control.all_admit(exact)?,"fresh paired execution changed forward, committed state or following logits");
                if arm==0 {controls.push(signature);control_times.push(timing);}
                else{paired_steps.push(json!({"step":index,"live_rows":[a_live,b_live],"all_exact":true,
                    "timing":timing,"control_timing":control_times[index],"result":signature}));}
            }
            w.large.handoff_from(&w.verify)?;other.handoff_from(verify)?;
        }
        records.push(json!({"pair":name,"prompt_tokens":[cases[0].0.len(),cases[peer].0.len()],
            "all_exact":true,"steps":paired_steps}));
    }
    drop(executor);
    std::fs::write(out.join("paired-actual-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"all_exact":true,"ownership":ownership,"memory":memory,"records":records,
        "scope":"Fresh paired verification, with both histories prepared before one forward. Two public prefix pairs, six changing blocks each,3/6live rows, full/partial/zero commits, original local vocabulary and live residuals, all retained regions/carry/encoder and followingM6. Graphs reused across advancing positions and reset to a different prefix. Timings include actual eager Engram preparation and first capture where marked; no drafter/sampler or completed-generation/1M capacity claim."
    }))?)?;
    Ok(())
}
