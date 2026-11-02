//! Fill two full-capacity owners and compare terminal fallback with ordinary
//! execution. EOS is deliberately not a stop in this finite capacity screen.
use super::*;
use crate::dspark_generation::GreedySession;

pub(super) fn endpoint(target:&Target,session:&GreedySession,stream:&Stream)->Result<Value> {
    ensure!(target.position()==1048576&&target.profile_is_current(),"full endpoint not reached");
    // A following-row diagnostic would itself exceed the advertised context.
    Ok(json!({"position":target.position(),"carry":target.carry_digests()?,
        "encoder":target.encoder_tail_digest()?,"regions":target.repeat_state_digests(stream)?,
        "draft":session.paired_draft_digest(stream)?}))
}

/// Four successive million-token prefills are a correctness comparison, not
/// a throughput workload. Let the retained model cool between the two arms
/// instead of weakening the independent90C observer or changing model math.
fn thermal_rest(w:&mut Worker)->Result<Value> {
    if std::env::var("CANADA_SCRATCH_ENDPOINT_THERMAL_REST").as_deref()!=Ok("1") {
        return Ok(Value::Null);
    }
    w.stream.synchronize()?;
    let began=std::time::Instant::now();let mut samples=Vec::new();
    loop {
        let mut temperatures=Vec::new();
        for zone in std::fs::read_dir("/sys/class/thermal")? {
            let zone=zone?;
            if zone.file_name().to_string_lossy().starts_with("thermal_zone") {
                temperatures.push(std::fs::read_to_string(zone.path().join("temp"))?.trim().parse::<i64>()?);
            }
        }
        let max=temperatures.into_iter().max().context("no thermal zones during endpoint pause")?;
        samples.push(json!({"seconds":began.elapsed().as_secs_f64(),"maximum_milliC":max}));
        if w.control.all_admit(max<=80_000)?{break;}
        ensure!(began.elapsed().as_secs()<180,"endpoint comparison cooling deadline");
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    Ok(json!({"seconds":began.elapsed().as_secs_f64(),"samples":samples,
        "scope":"Comparison-only gap; no operating-temperature requirement or throughput claim."}))
}

pub(super) fn load_prompts(out:&Path)->Result<Vec<Vec<u32>>> {
    let path=out.parent().and_then(Path::parent).context("finite output root")?
        .join("private/paired-endpoints.json");
    ensure!(std::fs::metadata(&path)?.len()<32*1024*1024,"bounded endpoint fixture");
    let fixture:Value=serde_json::from_slice(&std::fs::read(path)?)?;
    let mut prompts=Vec::new();
    for case in fixture["cases"].as_array().context("two endpoint inputs")? {
        let ids:Vec<u32>=serde_json::from_value(case["input_ids"].clone())?;
        let remain=case["remaining"].as_u64().context("endpoint remaining")? as usize;
        ensure!([43,59].contains(&remain)&&ids.len()+remain==1048576
            &&ids.iter().all(|&id|id<129280),"canonical endpoint input extent");
        prompts.push(ids);
    }
    ensure!(prompts.len()==2&&prompts[0].len()!=prompts[1].len(),"asymmetric endpoint pair");
    Ok(prompts)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(w:&mut Worker,other:&mut Target,verify:&mut Target,wides:&mut [Target],
    pooled:crate::ffn::Paired,first:PreparedSession,second:PreparedSession,
    profiles:&mut pair_serving::Profiles,rank:usize,out:&Path,ownership:Value)->Result<()> {
    let prompts=load_prompts(out)?;
    let mut prepared=[Some(first),Some(second)];
    let mut sessions:[Option<GreedySession>;2]=[None,None];
    let mut controls=[Vec::new(),Vec::new()];let mut finals=Vec::new();let mut final_actual=Vec::new();
    let mut copy_control=Value::Null;
    let mut copy_repeats=Vec::new();let mut copy_repeat_exact=true;
    let mut executor=crate::target::PairExecutor::new(&w.verify,verify,pooled,&w.comm,&w.stream)?;
    let mut paired_steps=0;let mut separate_steps=0;let mut single_survivor_steps=0;
    let seeds=[9371u64,9372];let teacher=[2u32;6];
    let mut thermal_rests=Vec::new();
    for arm in 0..2 {
        let progress=|phase:&str,frontiers:[u64;2]|->Result<()> {
            std::fs::write(out.join("paired-endpoint-progress.json"),serde_json::to_vec_pretty(
                &json!({"rank":rank,"arm":arm,"phase":phase,"frontiers":frontiers}))?)?;Ok(())
        };
        if arm==1 {
            progress("between_arm_thermal_rest",[w.large.position(),other.position()])?;
            thermal_rests.push(thermal_rest(w)?);
        }
        w.large.reset(1200+arm*2,&w.comm,&w.stream)?;
        other.reset(1201+arm*2,&w.comm,&w.stream)?;
        progress("prefill_first",[0,0])?;
        w.large.prefill_wide(&mut w.wides,&prompts[0],&w.comm,&w.stream)?;
        let untouched=w.large.repeat_state_digests(&w.stream)?;
        progress("prefill_second",[w.large.position(),0])?;
        other.prefill_wide(wides,&prompts[1],&w.comm,&w.stream)?;
        ensure!(w.control.all_admit(w.large.repeat_state_digests(&w.stream)?==untouched)?,
            "full peer prefill corrupted its owner");
        for (i,target)in [(0,&w.large),(1,&*other)] {
            if let Some(session)=sessions[i].as_mut(){session.reset_from_prefill(target,&w.cuda,&w.core,&w.stream)?;}
            else{sessions[i]=Some(prepared[i].take().context("endpoint prepared draft")?.attach(target,&w.stream)?);}
            let session=sessions[i].as_mut().unwrap();
            session.configure_sampling(seeds[i],None)?;
            session.configure_history_lookup(target,&prompts[i])?;
        }
        let mut anchors=[w.large.sample_token(seeds[0],&w.stream)?,other.sample_token(seeds[1],&w.stream)?];
        w.verify.handoff_from(&w.large)?;verify.handoff_from(other)?;
        let copy=profiles.check(w,verify,&teacher)?;
        // Exact top-k ties can change copy-profile repetition. Preserve both
        // observations without conflating that diagnostic with strict rollback
        // or paired terminal-step/state equivalence.
        if arm!=0 {copy_repeat_exact=w.control.all_admit(copy==copy_control)?;}
        copy_repeats.push(copy.clone());
        if arm==0{copy_control=copy;}
        let (left,right)=sessions.split_at_mut(1);
        let a=left[0].as_mut().unwrap();let b=right[0].as_mut().unwrap();
        if arm==0 {
            for (owner,session,target)in [(0,&mut *a,&mut w.verify),(1,&mut *b,&mut *verify)] {
                while target.position()<1048576 {
                    ensure!(controls[owner].len()<128,"bounded endpoint control steps");
                    let limit=(1048576-target.position()as usize).min(6);
                    let step=session.step(target,anchors[owner],limit,&w.comm,&w.stream)?;
                    anchors[owner]=step.next_anchor;
                    controls[owner].push(pair_generation::semantic_step(&step)?);
                }
                finals.push(endpoint(target,session,&w.stream)?);
            }
            std::fs::write(out.join("paired-endpoint-control.json"),serde_json::to_vec_pretty(&json!({
                "rank":rank,"endpoints":finals,"control_steps":controls,"copy_profiles":copy_control,
                "scope":"Separate-control arm only. Paired equivalence remains unproven until the final summary."}))?)?;
        }else{
            let mut indices=[0usize;2];
            while w.verify.position()<1048576||verify.position()<1048576 {
                ensure!(indices.iter().all(|&n|n<128),"bounded paired endpoint steps");
                let remaining=[1048576-w.verify.position()as usize,1048576-verify.position()as usize];
                if remaining.iter().all(|&n|n>=6) {
                    let first=a.prepare_pair_proposal(&w.verify,anchors[0],&w.comm,&w.stream)?;
                    let second=b.prepare_pair_proposal(verify,anchors[1],&w.comm,&w.stream)?;
                    ensure!(first.matches_target(&w.verify)&&second.matches_target(verify)
                        &&a.paired_probabilities_disjoint(&first,b,&second)?,"endpoint proposal ownership");
                    executor.begin(&mut w.verify,first.inputs(),verify,second.inputs(),&w.comm,&w.stream)?;
                    for(owner,step)in [(0,a.finish_pair_proposal(&mut w.verify,first,&w.comm,&w.stream)?),
                        (1,b.finish_pair_proposal(verify,second,&w.comm,&w.stream)?)] {
                        let actual=pair_generation::semantic_step(&step)?;
                        let same=controls[owner].get(indices[owner])==Some(&actual);
                        if !same {std::fs::write(out.join(format!("endpoint-repeat-mismatch-{owner}-{}.json",indices[owner])),
                            serde_json::to_vec_pretty(&json!({"owner":owner,"index":indices[owner],
                                "expected":controls[owner].get(indices[owner]),"actual":actual}))?)?;}
                        ensure!(w.control.all_admit(same)?,"paired endpoint sample changed");
                        indices[owner]+=1;anchors[owner]=step.next_anchor;
                    }
                    paired_steps+=1;
                }else{
                    for(owner,session,target)in [(0,&mut *a,&mut w.verify),(1,&mut *b,&mut *verify)] {
                        if remaining[owner]==0{continue;}
                        let step=session.step(target,anchors[owner],remaining[owner].min(6),&w.comm,&w.stream)?;
                        ensure!(w.control.all_admit(controls[owner].get(indices[owner])
                            ==Some(&pair_generation::semantic_step(&step)?))?,"terminal fallback sample changed");
                        indices[owner]+=1;anchors[owner]=step.next_anchor;separate_steps+=1;
                        single_survivor_steps+=usize::from(remaining[1-owner]==0);
                    }
                }
                progress("paired_generation",[w.verify.position(),verify.position()])?;
            }
            final_actual=vec![endpoint(&w.verify,a,&w.stream)?,endpoint(verify,b,&w.stream)?];
            ensure!(w.control.all_admit(final_actual==finals&&indices==[controls[0].len(),controls[1].len()])?,
                "full endpoint retained state changed");
        }
        w.large.handoff_from(&w.verify)?;other.handoff_from(verify)?;
    }
    ensure!(paired_steps>0&&separate_steps>0&&single_survivor_steps>0,"missing terminal pair/fallback coverage");
    let memory=w.cuda.admission_memory()?;
    ensure!(w.control.all_admit(memory.admitted_available>=8*1024usize.pow(3))?,"endpoint physical reserve");
    drop(executor);
    std::fs::write(out.join("paired-endpoint-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"all_exact":copy_repeat_exact,"paired_execution_exact":true,
        "copy_profile_repeat_exact":copy_repeat_exact,"copy_profile_repeats":copy_repeats,
        "ownership":ownership,"memory":memory,"full_serving_profiles":profiles.memory,
        "prompt_tokens":prompts.iter().map(Vec::len).collect::<Vec<_>>(),"endpoints":final_actual,
        "control_steps":controls,"copy_profiles":copy_control,"paired_steps":paired_steps,
        "separate_steps":separate_steps,"single_survivor_steps":single_survivor_steps,
        "thermal_rests":thermal_rests,
        "scope":"Two actual1048576-token endpoints, complete serving profiles and private neural/history/draft state. Ordinary separate control versus pairedM3/M6 with normal short-budget/context-tail fallback and an independently continuing survivor. EOS deliberately not a stopping condition in this finite storage/edge probe; repeated public-source capacity prompts. Not HTTP lifecycle, real task-quality or throughput qualification."
    }))?)?;Ok(())
}
