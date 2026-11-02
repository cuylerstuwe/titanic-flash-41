//! Actual point-mass copy generation, with M32/M16/M6 causal fallbacks.
//! This finite mechanism screen does not change serving's M16 selection.
use super::*;

pub(super) fn execute(w:&mut Worker,c:Command)->Result<Value> {
    let bytes=c.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(bytes.as_slice()))?;
    ensure!(wire==bytes,"history32 generation command mismatch");w.execute(c)
}
pub(super) fn state(w:&mut Worker)->Result<Value> {
    Ok(json!({"position":w.verify.position(),"carry":w.verify.carry_digests()?,
        "encoder":w.verify.encoder_tail_digest()?,
        "draft":w.session.as_mut().unwrap().conditioning_state_digest_for_probe(&w.comm,&w.stream)?,
        "lookup":w.session.as_ref().unwrap().history_lookup_digest_for_probe()?}))
}
pub(super) fn restore(w:&mut Worker,image:&crate::dspark_generation::StateImage,anchor:u32,expected:&Value)->Result<()> {
    w.target_lineage=w.target_lineage.checked_add(1).context("history32 generation lineage exhausted")?;
    ensure!(w.session.as_mut().unwrap().restore_state_image(&mut w.verify,image,
        w.target_lineage,&w.comm,&w.stream)?==anchor,"history32 saved anchor");
    ensure!(&state(w)?==expected,"history32 generation restoration changed accepted state");
    Ok(())
}

pub(super) fn run(w:&mut Worker,rank:usize,kernels:&Path,out:&Path)->Result<()> {
    ensure!(crate::config::history32_probe_enabled()
        &&std::env::var("CANADA_SCRATCH_HISTORY32_GENERATION_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none()&&w.long_verify.is_some(),"finite M32 generation admission");
    let before=w.cuda.admission_memory()?;let module=w.cuda.module(&w.core)?;
    let mut wider=w.large.fork_profile(&module,&w.core,&w.stream,kernels,32,Phase::Verify)?;
    wider.set_graph_replay(true)?;
    let serving_compact=Target::compact_history_sampling_requested()?;
    if serving_compact {
        ensure!(w.long_verify.as_ref().unwrap().compact_history_sampling_enabled(),
            "serving M16 compact constructor was not selected");
        wider.enable_compact_history_sampling(&w.core,&w.stream)?;
        for target in [w.long_verify.as_mut().unwrap(),&mut wider] {
            target.set_compact_sampling_for_probe(false,&w.core,&w.stream)?;
        }
    }
    let compact_index_layers=wider.compact_index_layers();
    if std::env::var("CANADA_SCRATCH_GROUPED_SPARSE32").as_deref()==Ok("1") {
        ensure!(compact_index_layers==4,"M32 compact serving profile missing consumers");
    }
    ensure!(w.large.shared_state_report(&wider)["all_weights_and_causal_storage_shared"]==true,
        "M32 generation duplicated canonical model state");
    let mut buffers16=LongHistoryBuffers::new(&w.cuda,&w.core)?;
    let mut buffers32=LongHistoryBuffers::with_rows(&w.cuda,&w.core,32)?;
    w.prepared_session.as_mut().context("M32 screen needs fresh prepared session")?
        .prepare_copy_conditioning(&buffers32,&w.comm,&w.stream)?;
    let conditioning=std::env::var("CANADA_SCRATCH_COPY_CONDITIONING_GENERATION_SCREEN").as_deref()==Ok("1");
    let commit=std::env::var("CANADA_SCRATCH_COPY_COMMIT_GENERATION_SCREEN").as_deref()==Ok("1");
    let compact_sample=std::env::var("CANADA_SCRATCH_SHARDED_CATEGORICAL_MODEL_SCREEN").as_deref()==Ok("1");
    ensure!(usize::from(conditioning)+usize::from(commit)+usize::from(compact_sample)<=1,
        "isolate conditioning, commit metadata and compact sampling");
    let after=w.cuda.admission_memory()?;
    ensure!(w.control.all_admit(after.admitted_available>=9*1024usize.pow(3))?,"M32 generation snapshot reserve");
    let fixtures:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/history-copy-width32-v1.json"))?;
    let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    let selected:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/history-strong-backup-frontiers-v1.json"))?;
    let tools:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-history-score-v1.json"))?;
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let mut records=Vec::new();
    for case in [1usize,2] {
        let c=&fixtures["cases"][case];let ordinal=c["frontier_index"].as_u64().unwrap()as usize;
        let prompt:Vec<u32>=if c["kind"]=="repository" {
            let f=&selected["cases"][ordinal];let mut p=tokens[..f["corpus_tokens"].as_u64().unwrap()as usize].to_vec();
            p.extend(serde_json::from_value::<Vec<u32>>(f["suffix_ids"].clone())?);p
        }else{ensure!(c["kind"]=="tool","M32 fixture kind");serde_json::from_value(tools["cases"][ordinal]["prompt_ids"].clone())?};
        let n=prompt.len();let end=c["source_end"].as_u64().unwrap()as usize;
        let anchor=c["anchor"].as_u64().unwrap()as u32;
        ensure!(n==c["prefix_tokens"].as_u64().unwrap()as usize&&end>=63&&end+32<=n
            &&prompt[end]==anchor&&prompt[end-63..end]==prompt[n-63..],"M32 committed source");
        let mut teacher=vec![anchor];teacher.extend_from_slice(&prompt[end+1..end+32]);
        let begin=execute(w,Command::Begin{images:Vec::new(),id:7_860_000+case as u64,
            seed:2026110300+case as u64,greedy:false,limit:1,tokens:prompt})?;
        ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.,"M32 target sampler");
        let (bytes,cpu)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
        ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
            >=8*1024usize.pow(3)+bytes+cpu)?,"M32 generation bounded snapshot");
        let image=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
        let original=state(w)?;
        // Prime the neural fallback, then restore the entire accepted session.
        // Graph scratch may stay warm; accepted target/draft/history may not.
        w.session.as_mut().unwrap().configure_history_policy_for_probe(None)?;
        w.session.as_mut().unwrap().step(&mut w.verify,anchor,6,&w.comm,&w.stream)?;
        restore(w,&image,anchor,&original)?;
        for live in 1..=6 {
            w.verify.begin_verification(&teacher[..live],&w.comm,&w.stream)?;
            w.verify.finish_verification(0,&w.comm,&w.stream)?;
        }
        for rows in [16usize,32] {
            let target=if rows==16 {w.long_verify.as_mut().unwrap()}else{&mut wider};
            target.handoff_from(&w.verify)?;
            target.begin_verification(&teacher[..rows],&w.comm,&w.stream)?;
            target.finish_verification(0,&w.comm,&w.stream)?;w.verify.handoff_from(target)?;
        }
        // The M32 attention precision policy is new at full-model scope.
        // Check that altered future candidates cannot change accepted logits.
        let mut causal_prefix=None;
        for changed in [false,true] {
            let mut inputs=teacher.clone();
            if changed {for id in &mut inputs[17..] {*id=2+(*id+123)%120000;}}
            wider.handoff_from(&w.verify)?;
            wider.begin_verification(&inputs,&w.comm,&w.stream)?;
            let (logits,live)=wider.pending_logits()?;ensure!(live==32,"M32 causal prefix geometry");
            let bytes=unsafe{w.cuda.read_device(logits.ptr(),17*129280*4)}?;
            let hash=format!("{:x}",Sha256::digest(bytes));
            if let Some(expected)=&causal_prefix {ensure!(expected==&hash,"M32 future suffix changed causal logits");}
            else {causal_prefix=Some(hash);}
            wider.finish_verification(0,&w.comm,&w.stream)?;w.verify.handoff_from(&wider)?;
        }
        let mut append_checks=Vec::new();
        if compact_sample&&serving_compact {
            // Regression mechanism: M16 owns both full-logit retained EXTEND
            // and compact-logit copy graphs. Exercise their shared IO in both
            // directions, including a committed append and rolled-back copy.
            let live=if case==1{7}else{16};
            for enabled in [false,true] {
                restore(w,&image,anchor,&original)?;
                let target=w.long_verify.as_mut().unwrap();target.handoff_from(&w.verify)?;
                target.set_compact_sampling_for_probe(enabled,&w.core,&w.stream)?;
                target.set_tiny_extend_for_probe(true,&w.stream)?;
                target.begin_verification(&teacher[..live],&w.comm,&w.stream)?;
                let (logits,count)=target.pending_logits()?;ensure!(count==live,"append full logits extent");
                let raw=unsafe{w.cuda.read_device(logits.ptr(),live*129280*4)}?;
                let logits_sha256=format!("{:x}",Sha256::digest(raw));
                target.finish_verification(live,&w.comm,&w.stream)?;
                let greedy=target.greedy_token(&w.stream)?;
                let sampled=target.sample_token(2026111300+case as u64,&w.stream)?;
                let output_digests=target.valid_output_digests()?;
                w.session.as_mut().unwrap().resume_after_medium_extend(target,&mut buffers16,
                    n as u64,&teacher[..live],&w.comm,&w.stream)?;
                target.set_tiny_extend_for_probe(false,&w.stream)?;w.verify.handoff_from(target)?;
                let appended=state(w)?;
                let target=w.long_verify.as_mut().unwrap();target.handoff_from(&w.verify)?;
                let mut following=teacher[live..live+16].to_vec();following[0]=sampled;
                target.begin_verification(&following,&w.comm,&w.stream)?;
                let predictions=target.verification_history_sample(2026111400+case as u64,&w.comm,&w.stream)?;
                ensure!(target.pending_logits().is_err()==enabled,"copy full-head visibility contract");
                target.finish_verification(0,&w.comm,&w.stream)?;w.verify.handoff_from(target)?;
                ensure!(state(w)?==appended,"copy rollback changed committed prompt append");
                append_checks.push(json!({"enabled":enabled,"live":live,"logits_sha256":logits_sha256,
                    "greedy":greedy,"sampled":sampled,"output_digests":output_digests,
                    "accepted_state":appended,"following_copy_predictions":predictions,"rollback_exact":true}));
            }
            for key in ["logits_sha256","greedy","sampled","output_digests","accepted_state","following_copy_predictions"] {
                ensure!(w.control.all_admit(append_checks[0][key]==append_checks[1][key])?,
                    "compact sampling changed retained append {key}");
            }
            restore(w,&image,anchor,&original)?;
        }
        let mut trials=Vec::new();
        // One short warm pair followed by a measured pair in reverse order.
        // Both begin at the same saved public conditional frontier.
        for trial in 0..4 {
            let width=if conditioning||commit||compact_sample{32}else{[16,32,32,16][trial]};
            let wide_conditioning=conditioning&&[false,true,true,false][trial];
            if conditioning {buffers32.set_wide_for_probe(wide_conditioning)?;}
            let full_commit_metadata=commit&&[false,true,true,false][trial];
            let maximum=if trial<2 {96}else{192};
            restore(w,&image,anchor,&original)?;
            let compact_enabled=compact_sample&&[false,true,true,false][trial];
            if compact_sample {
                // Warm the new communication topology and its sampler outside
                // timing, then restore every accepted target/draft/history byte.
                for rows in [16usize,32] {
                    let target=if rows==16{w.long_verify.as_mut().unwrap()}else{&mut wider};
                    target.handoff_from(&w.verify)?;
                    target.set_compact_sampling_for_probe(compact_enabled,&w.core,&w.stream)?;
                    target.begin_verification(&teacher[..rows],&w.comm,&w.stream)?;
                    target.verification_history_sample(2026111200,&w.comm,&w.stream)?;
                    if compact_enabled {ensure!(target.pending_logits().is_err(),"compact path exposed stale full logits");}
                    target.finish_verification(0,&w.comm,&w.stream)?;w.verify.handoff_from(target)?;
                }
                restore(w,&image,anchor,&original)?;
            }
            if commit {w.session.as_mut().unwrap().configure_full_commit_metadata_for_probe(full_commit_metadata,&w.stream)?;}
            let seed=2026110400+case as u64*10+(trial/2)as u64;
            w.session.as_mut().unwrap().configure_sampling(seed,Some(w.cfg.eos_token_id))?;
            let captures=w.verify.capture_statistics().1+w.long_verify.as_ref().unwrap().capture_statistics().1
                +wider.capture_statistics().1;
            let mut emitted=vec![anchor];let mut pending=anchor;let mut steps=Vec::new();let mut forced=false;
            let started=Instant::now();
            while emitted.len()<maximum&&pending!=w.cfg.eos_token_id {
                let remaining=maximum-emitted.len();let force=!forced&&emitted.len()>=maximum/2;
                let s=w.session.as_mut().unwrap();
                if force {s.configure_history_policy_for_probe(None)?;}
                let mut long=if width==32&&!force {
                    s.try_long_history_step(&mut w.verify,&mut wider,&mut buffers32,pending,remaining,&w.comm,&w.stream)?
                }else{None};
                if long.is_none()&&!force {
                    long=s.try_long_history_step(&mut w.verify,w.long_verify.as_mut().unwrap(),
                        &mut buffers16,pending,remaining,&w.comm,&w.stream)?;
                }
                let step=match long {Some(step)=>step,None=>s.step(&mut w.verify,pending,remaining.min(6),&w.comm,&w.stream)?};
                if force {
                    ensure!(step.proposal_source=="dspark","M32 neural fallback was not used");
                    forced=true;s.configure_history_policy_for_probe(Some(8))?;
                }
                ensure!(step.tokens[0]==pending&&step.tokens.len()<=remaining,"M32 visible output extent");
                emitted.extend_from_slice(&step.tokens[1..]);emitted.push(step.next_anchor);pending=step.next_anchor;
                steps.push(step);
            }
            let seconds=started.elapsed().as_secs_f64();
            ensure!(emitted.len()<=maximum&&w.verify.position()==(n+emitted.len()-1)as u64
                &&w.session.as_ref().unwrap().history_lookup_metadata().unwrap().0==w.verify.position(),"M32 final frontiers");
            ensure!(steps.iter().any(|s|s.verified_rows==width),"requested history width was never exercised");
            if conditioning {
                ensure!(steps.iter().any(|s|s.verified_rows==32&&s.tokens.len()>=19),"wide copy conditioning not exercised");
            }
            let wide_active=if conditioning{wide_conditioning}else{LongHistoryBuffers::wide_conditioning_requested()?};
            let full_six_chunks:usize=steps.iter().map(|s|
                if wide_active&&s.verified_rows==32&&s.tokens.len()>=19 {0}else{s.tokens.len()/6}).sum();
            if commit {ensure!(full_six_chunks>0,"full-six commit metadata branch not exercised");}
            trials.push(json!({"trial":trial,"warmup":trial<2,"width":width,"seed":seed,
                "wide_conditioning":wide_active,"full_commit_metadata":full_commit_metadata,
                "compact_sampling":compact_enabled,
                "full_six_commit_chunks":full_six_chunks,
                "output_tokens":emitted.len(),"decode_seconds":seconds,"decode_tps":(emitted.len()-1)as f64/seconds,
                "forced_neural_fallback":forced,"output_ids":emitted,"steps":steps,"final_state":state(w)?,
                "new_target_graph_captures":w.verify.capture_statistics().1+w.long_verify.as_ref().unwrap().capture_statistics().1
                    +wider.capture_statistics().1-captures}));
            eprintln!("rank{rank} M32 generation case{case} trial{trial} width{width}: {:.2}tps",(emitted.len()-1)as f64/seconds);
        }
        if conditioning||commit||compact_sample {
            // The component was byte-identical. Verify actual accepted KV,
            // neural fallbacks and causal publication keep that property.
            for pair in trials.chunks_exact(2) {
                let same=pair[0]["output_ids"]==pair[1]["output_ids"]&&pair[0]["final_state"]==pair[1]["final_state"];
                if !same {std::fs::write(out.join(format!("copy-conditioning-mismatch-case{case}.json")),serde_json::to_vec_pretty(&trials)?)?;}
                ensure!(w.control.all_admit(same)?,"copy conditioning/commit/sampler changed generation or accepted state");
            }
        }
        restore(w,&image,anchor,&original)?;
        ensure!(w.history.frontier()==n,"M32 finite generation touched worker request history");
        records.push(json!({"case":case,"name":c["name"],"kind":c["kind"],"prompt_tokens":n,
            "trials":trials,"append_checks":append_checks,"restoration_exact":true,"future_suffix_causal_prefix_sha256":causal_prefix}));
        execute(w,Command::Finish(Finish::Length))?;
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("history32-generation-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"before_m32":before,"after_m32":after,"records":records,"all_restorations_exact":true,
        "compact_index_layers":compact_index_layers,
        "copy_conditioning_screen":conditioning,
        "copy_commit_screen":commit,
        "compact_sampling_screen":compact_sample,
        "serving_compact_construction":serving_compact,
        "scope":"Two public conditional copy frontiers, repository and tool text; one96-token warm pair and one192-token measured pair each. Target T1/P1; M32 then M16 then M6 causal fallbacks, actual independent target sampling and accepted-feature draft injection. Complete step cost included, exact full-session restore. Narrow opportunity screen, not completed-task quality or general workload speed. No serving policy change."
    }))?)?;Ok(())
}
