//! Small sampled continuation comparison; no HTTP admission or throughput
//! qualification. Independent target/drafter/sampler owners share only weights.
use super::*;
use crate::dspark_generation::{GreedySession,Step};

pub(super) fn semantic_step(step:&Step)->Result<Value> {
    let mut value=serde_json::to_value(step)?;
    let object=value.as_object_mut().context("step object")?;
    object.remove("phase_ms");object.remove("verify_host_ms");
    Ok(value)
}

fn inspect_state(index:usize)->bool {
    // Full1M state hashing reads gigabytes and can perturb the following timed
    // step even when its own duration is excluded. Prior screens covered every
    // fourth step; the endpoint plus each emitted proposal/acceptance is enough
    // for subsequent small integration screens without that timing disturbance.
    index==15
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(w:&mut Worker,other:&mut Target,verify:&mut Target,wides:&mut [Target],
    pooled:crate::ffn::Paired,cases:&[(Vec<u32>,Vec<u32>)],rank:usize,kernels:&Path,out:&Path,ownership:Value)->Result<()> {
    let mut first_prepared=w.prepared_session.take();
    let mut second_prepared=Some(first_prepared.as_ref().context("prepared first drafter")?
        .independent_for_probe(other,&w.core,&w.stream,kernels)?);
    let mut serving=if pair_serving::enabled(){Some(pair_serving::Profiles::new(
        w,other,second_prepared.as_mut().unwrap(),kernels,out)?)}else{None};
    let memory=w.cuda.admission_memory()?;
    ensure!(w.control.all_admit(memory.admitted_available>=8*1024usize.pow(3))?,
        "second sampled draft owner consumed physical reserve");
    if std::env::var("CANADA_SCRATCH_PAIRED_ENDPOINT_SCREEN").as_deref()==Ok("1") {
        // Dispatch before entering the legacy comparison's large session
        // frame. Nesting both endpoint harnesses exceeded the8MiB main stack.
        if std::env::var("CANADA_SCRATCH_PAIRED_ENDPOINT_MATCHED_SCREEN").as_deref()==Ok("1") {
            return pair_endpoints_matched::run(w,other,verify,wides,pooled,
                first_prepared.take().context("endpoint first drafter")?,
                second_prepared.take().context("endpoint second drafter")?,
                serving.as_mut().context("endpoints require all serving profiles")?,rank,out,ownership);
        }
        return pair_endpoints::run(w,other,verify,wides,pooled,
            first_prepared.take().context("endpoint first drafter")?,
            second_prepared.take().context("endpoint second drafter")?,
            serving.as_mut().context("endpoints require all serving profiles")?,rank,out,ownership);
    }
    let mut first_session:Option<GreedySession>=None;let mut second_session:Option<GreedySession>=None;
    let mut executor=crate::target::PairExecutor::new(&w.verify,verify,pooled,&w.comm,&w.stream)?;
    let mut records=Vec::new();
    let long=std::env::var("CANADA_SCRATCH_PAIRED_LONG_CONTEXT_SCREEN").as_deref()==Ok("1");
    let lookup=std::env::var("CANADA_SCRATCH_PAIRED_LOOKUP_SCREEN").as_deref()==Ok("1");
    let blocked=std::env::var("CANADA_SCRATCH_PAIRED_BLOCKED_CONTROL").as_deref()==Ok("1");
    let cost_screen=std::env::var("CANADA_SCRATCH_PAIR_COST_SCREEN").as_deref()==Ok("1");
    let pairs=if lookup {vec![("inherited",0usize,1usize),("copy-first",3,2),("copy-second",0,4)]}
        else if long {vec![("inherited",0,1),("long-independent",3,4)]}
        else {vec![("inherited",0,1),("independent",0,2)]};
    for (case,(name,primary,peer))in pairs.into_iter().enumerate() {
        let seeds=[7161+case as u64*2,7162+case as u64*2];
        let mut controls=Vec::new();let mut control_times=Vec::new();let mut control_captures=Vec::new();let mut steps=Vec::new();
        let mut profile_control=Value::Null;
        let mut profile_repeats=Vec::new();let mut copy_repeat_exact=true;
        let mut route_captures=Vec::new();let mut cost=Value::Null;
        for arm in 0..2 {
            let lineage=900+(case*4+arm*2)as u64;
            w.large.reset(lineage,&w.comm,&w.stream)?;other.reset(lineage+1,&w.comm,&w.stream)?;
            w.large.prefill_wide(&mut w.wides,&cases[primary].0,&w.comm,&w.stream)?;
            let isolated=if serving.is_some(){Some(w.large.repeat_state_digests(&w.stream)?)}else{None};
            other.prefill_wide(wides,&cases[peer].0,&w.comm,&w.stream)?;
            if let Some(before)=isolated {
                ensure!(w.control.all_admit(w.large.repeat_state_digests(&w.stream)?==before)?,
                    "peer wide prefill changed the first causal owner");
            }
            for (session,prepared,target,seed,prefix)in [
                (&mut first_session,&mut first_prepared,&w.large,seeds[0],&cases[primary].0),
                (&mut second_session,&mut second_prepared,&*other,seeds[1],&cases[peer].0)] {
                if let Some(active)=session.as_mut() {active.reset_from_prefill(target,&w.cuda,&w.core,&w.stream)?;}
                else{*session=Some(prepared.take().context("fresh prepared draft")?.attach(target,&w.stream)?);}
                session.as_mut().unwrap().configure_sampling(seed,None)?;
                if lookup && case>0 {session.as_mut().unwrap().configure_history_lookup(target,prefix)?;}
            }
            let mut anchors=[w.large.sample_token(seeds[0],&w.stream)?,other.sample_token(seeds[1],&w.stream)?];
            if lookup {
                // Fixed accepted public frontiers isolate real lookup hits;
                // this is conditional model execution, not an outcome sample.
                if primary>=3 {anchors[0]=cases[primary].1[0];}
                if peer>=3 {anchors[1]=cases[peer].1[0];}
            }
            w.verify.handoff_from(&w.large)?;verify.handoff_from(other)?;
            if let Some(profiles)=serving.as_mut() {
                let current=profiles.check(w,verify,&cases[primary].1)?;
                if arm!=0&&current!=profile_control {
                    eprintln!("copy_profile_repeat {}",json!({"rank":rank,"pair":name,
                        "before":profile_control,"after":current}));
                }
                // Source20's exact top-k contract permits different identities
                // at an equal-score cutoff. The captured M32 divergence was
                // reproduced with identical scores and valid top2048 choices
                // on CUDA (production/two-stream-copy-cutoff-20261101.json).
                // Record repetition separately; check() still strictly checks
                // published state, carry, encoder and following logits after
                // every rollback. Paired sampled execution remains exact below.
                if arm!=0 {copy_repeat_exact=w.control.all_admit(current==profile_control)?;}
                profile_repeats.push(current.clone());
                if arm==0{profile_control=current;}
            }
            let a=first_session.as_mut().unwrap();let b=second_session.as_mut().unwrap();
            if arm==0&&crate::config::paired_draft_head_requested() {
                let proof=a.probe_pooled_query(b,anchors,&w.comm,&w.stream)?;
                std::fs::write(out.join(format!("paired-query-head-{case}.json")),serde_json::to_vec_pretty(&proof)?)?;
            }
            if arm==0 && blocked {
                // An uninterrupted block per owner is a stronger compute-only
                // comparator than alternating cold request weights each step.
                // It excludes serving state-exchange cost, conservatively.
                let mut separate=[Vec::new(),Vec::new()];
                for (owner,session,target,teacher) in [
                    (0,&mut *a,&mut w.verify,&cases[primary].1),
                    (1,&mut *b,&mut *verify,&cases[peer].1)] {
                    for index in 0..16 {
                        let capture_before=target.capture_statistics().1;
                        let began=std::time::Instant::now();
                        let step=session.step(target,anchors[owner],6,&w.comm,&w.stream)?;
                        let ms=began.elapsed().as_secs_f64()*1000.;anchors[owner]=step.next_anchor;
                        let captured=target.capture_statistics().1!=capture_before;
                        let (target_state,draft_state)=if inspect_state(index) {
                            (state(target,teacher,&w.cuda,&w.comm,&w.stream)?,session.paired_draft_digest(&w.stream)?)
                        }else{(Value::Null,Value::Null)};
                        separate[owner].push((semantic_step(&step)?,target_state,draft_state,ms,captured));
                    }
                }
                for index in 0..16 {
                    let(x,y)=(&separate[0][index],&separate[1][index]);
                    controls.push(json!({"steps":[x.0,y.0],"state":if inspect_state(index) {
                        json!({"target":[x.1,y.1],"draft":[x.2,y.2]})
                    }else{Value::Null}}));
                    control_times.push(x.3+y.3);
                    control_captures.push([x.4,y.4]);
                }
                w.large.handoff_from(&w.verify)?;other.handoff_from(verify)?;
                continue;
            }
            for index in 0..16 {
                let mut capture_seconds=0.;
                let capture_before=[w.verify.capture_statistics().1,verify.capture_statistics().1];
                let began=std::time::Instant::now();
                let (first,second,target_timing)=if arm==0 {
                    let first=a.step(&mut w.verify,anchors[0],6,&w.comm,&w.stream)?;
                    let second=b.step(verify,anchors[1],6,&w.comm,&w.stream)?;
                    (first,second,Value::Null)
                }else{
                    let(first,second,head_pooled)=a.prepare_pair_proposals(&w.verify,anchors[0],b,verify,anchors[1],true,&w.comm,&w.stream)?;
                    ensure!(first.matches_target(&w.verify)&&second.matches_target(verify)
                        &&!first.matches_target(verify)&&!second.matches_target(&w.verify),
                        "prepared proposal crossed target ownership");
                    ensure!(a.paired_probabilities_disjoint(&first,b,&second)?,"paired proposals aliased q");
                    let mut timing=executor.begin(&mut w.verify,first.inputs(),verify,second.inputs(),&w.comm,&w.stream)?;
                    timing["pooled_draft_head"]=json!(head_pooled);
                    if cost_screen&&index>=2&&route_captures.len()<2
                        &&first.inputs().len()==6&&second.inputs().len()==6 {
                        let at=Instant::now();
                        let directory=out.join(format!("paired-cost-routes/{name}/step{index}"));
                        let captures=[w.verify.capture_expert_components(&directory.join("owner0"),&w.stream)?,
                            verify.capture_expert_components(&directory.join("owner1"),&w.stream)?];
                        w.control.agree(b"pair-cost-routes-captured")?;
                        capture_seconds=at.elapsed().as_secs_f64();
                        route_captures.push(json!({"step":index,"owners":captures,"seconds":capture_seconds}));
                    }
                    let first=a.finish_pair_proposal(&mut w.verify,first,&w.comm,&w.stream)?;
                    let second=b.finish_pair_proposal(verify,second,&w.comm,&w.stream)?;
                    (first,second,timing)
                };
                let elapsed_ms=(began.elapsed().as_secs_f64()-capture_seconds)*1000.;
                anchors=[first.next_anchor,second.next_anchor];
                let captured=[w.verify.capture_statistics().1!=capture_before[0],verify.capture_statistics().1!=capture_before[1]];
                let state=if inspect_state(index) {json!({
                    "target":[state(&mut w.verify,&cases[primary].1,&w.cuda,&w.comm,&w.stream)?,
                        state(verify,&cases[peer].1,&w.cuda,&w.comm,&w.stream)?],
                    "draft":[a.paired_draft_digest(&w.stream)?,b.paired_draft_digest(&w.stream)?]
                })}else{Value::Null};
                let signature=json!({"steps":[semantic_step(&first)?,semantic_step(&second)?],"state":state});
                let exact=arm==0||signature==controls[index];
                std::fs::write(out.join("paired-generation-progress.json"),serde_json::to_vec_pretty(&json!({
                    "pair":name,"arm":arm,"step":index,"exact":exact,"actual":signature,
                    "expected":if arm==0{None}else{controls.get(index)},"elapsed_ms":elapsed_ms,"target":target_timing
                }))?)?;
                ensure!(w.control.all_admit(exact)?,"paired sampled step changed proposals, acceptance, output or retained state");
                if arm==0 {controls.push(signature);control_times.push(elapsed_ms);control_captures.push(captured);}
                else{steps.push(json!({"step":index,"all_exact":true,"result":signature,
                    "emitted_tokens":first.tokens.len()+second.tokens.len(),"target":target_timing,
                    "control_ms":control_times[index],"control_captured":control_captures[index],"paired_ms":elapsed_ms,
                    "route_capture_seconds_excluded":capture_seconds}));}
            }
            if cost_screen&&arm==1 {
                ensure!(!route_captures.is_empty(),"cost screen did not capture a real6+6 pair");
                cost=super::super::pair_step_profile_probe::measure_pair_cost(w,name,&cases[primary].1,out)?;
            }
            w.large.handoff_from(&w.verify)?;other.handoff_from(verify)?;
        }
        records.push(json!({"pair":name,"prompt_tokens":[cases[primary].0.len(),cases[peer].0.len()],
            "seeds":seeds,"all_exact":copy_repeat_exact,"paired_execution_exact":true,
            "copy_profile_repeat_exact":copy_repeat_exact,"steps":steps,"serving_profiles":profile_control,
            "serving_profile_repeats":profile_repeats,"route_captures":route_captures,"cost":cost}));
    }
    drop(executor);
    let mut neural_steps=0;let mut lookup_steps=0;let mut mixed_steps=0;
    for record in &records {
        for step in record["steps"].as_array().context("paired recorded steps")? {
            let source=&step["result"]["steps"];
            let copied=[source[0]["proposal_source"]=="history_point_mass",source[1]["proposal_source"]=="history_point_mass"];
            lookup_steps+=copied.into_iter().filter(|c|*c).count();
            neural_steps+=copied.into_iter().filter(|c|!*c).count();
            mixed_steps+=usize::from(copied[0]!=copied[1]);
        }
    }
    ensure!(!lookup||w.control.all_admit(lookup_steps>0&&mixed_steps>0)?,
        "lookup pairing screen did not exercise point-mass and mixed proposals");
    std::fs::write(out.join("paired-generation-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"all_exact":records.iter().all(|r|r["all_exact"]==true),
        "paired_execution_exact":true,"copy_profile_repeat_exact":records.iter().all(|r|r["copy_profile_repeat_exact"]==true),
        "ownership":ownership,"memory":memory,"records":records,
        "context_capacity_each":w.large.context_capacity(),"blocked_control":blocked,
        "full_serving_profiles":serving.as_ref().map(|p|&p.memory),
        "primary_wide_bank":w.wides.iter().filter_map(Target::wide_report).collect::<Vec<_>>(),
        "peer_wide_bank":wides.iter().filter_map(Target::wide_report).collect::<Vec<_>>(),
        "proposal_coverage":{"neural_steps":neural_steps,"lookup_steps":lookup_steps,"mixed_steps":mixed_steps},
        "repeat_policy":"Copy-profile repetition is diagnostic: exact score ties may choose different candidate identities. Published-state rollback and paired proposals/acceptance/output/final state remain strict checks. all_exact includes repetition; paired_execution_exact reports the paired execution comparison separately.",
        "scope":if serving.is_some() {
            "Two full1M-capacity independent causal and draft owners, complete512..8192wide banks sharing only sequential transient arenas, bothM16/M32copy profiles and conditioning, rank3 vision retained. Public19K/inherited and129.5K/independent prefixes; peer prefill isolation, both copy-profile zero-accept rollback/draws/logits, sampledM3/M6 paired generation versus separate execution. Full state at16-step endpoint. Does not fill1M or qualify serving/terminal/cancel behavior or completed outcomes."
        }else if lookup {
            "Independent mixed neural and point-mass history proposals, T1/P1 with the existing request-local target draw laws. Public fixed reasoning/answer copy frontiers plus ordinary public coding prefixes. Proposal/acceptance/output equality each step and full target/draft/history checks at the16-step endpoint against separate execution. Both control and paired captures marked. Not completed outcomes, wideM16/M32 copy, terminal/context-edge, serving throughput or full1M growth qualification."
        }else{
            "Two independent neural DSpark sessions sharing immutable weights, T1/P1 target block rejection with actual private q and causal3/6lengths. Sixteen sampled steps for each public pair and arm; proposals/acceptance/outputs match ordinary separate steps, with full retained target state/followingM6 and accepted draftKV at the16-step endpoint. Both control and paired captures marked. Sequential control/candidate, capped reasoning diagnostics; not completed-task quality, history-copy/edge behavior, serving throughput or1M memory qualification."
        }
    }))?)?;
    Ok(())
}
