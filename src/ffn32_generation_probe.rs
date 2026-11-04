//! Same-load M32 scheduling comparison using the ordinary copy/drafter owners.
use super::*;
use super::history32_generation_probe::{execute,state,restore};
fn composition()->bool {std::env::var("CANADA_SCRATCH_SOLO_COMPOSITION_SCREEN").as_deref()==Ok("1")}
fn schedule(w:&mut Worker,mask:u8)->Result<()> {
    if composition() {
        let t=w.long_verify.as_mut().unwrap();t.handoff_from(&w.verify)?;
        t.set_verify16_schedule_for_probe(mask,&w.stream)?;
        ensure!(t.ffn_overlap_layers()==if mask==3{(40,40)}else{(0,0)},"M16 composed FFN schedule");
        w.verify.handoff_from(t)?;
        w.verify.set_cache_ca_for_probe(mask==3,&w.stream)?;
    }
    let t=w.copy32_verify.as_mut().unwrap();t.handoff_from(&w.verify)?;
    t.set_verify16_schedule_for_probe(mask,&w.stream)?;
    ensure!(t.ffn_overlap_layers()==if mask==3 {(40,40)}else{(0,0)},"M32 FFN schedule");
    w.verify.handoff_from(t)
}
fn captures(w:&Worker)->usize {
    (w.verify.capture_statistics().1+w.long_verify.as_ref().unwrap().capture_statistics().1
        +w.copy32_verify.as_ref().unwrap().capture_statistics().1)as usize
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    let restore_profile=std::env::var("CANADA_SCRATCH_WIDE_RESTORE_PROFILE").as_deref()==Ok("1");
    ensure!(!restore_profile||composition(),"wide restore profile requires composed candidate");
    ensure!(w.current.is_none()&&std::env::var("CANADA_SCRATCH_FFN32_GENERATION_SCREEN").as_deref()==Ok("1")
        &&w.copy32_verify.as_ref().context("M32 owner")?.ffn_overlap_layers()==(40,40)
        &&w.long_verify.as_ref().context("M16 owner")?.ffn_overlap_layers()==(40,40),
        "generation requires both carried constructor defaults");
    let fixtures:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/history-copy-width32-v1.json"))?;
    let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    let selected:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/history-strong-backup-frontiers-v1.json"))?;
    let tools:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-history-score-v1.json"))?;
    let mut buffers16=LongHistoryBuffers::new(&w.cuda,&w.core)?;
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let mut records=Vec::new();
    for case in if restore_profile{vec![1usize]}else{vec![1usize,2]} {
        let c=&fixtures["cases"][case];let ordinal=c["frontier_index"].as_u64().unwrap()as usize;
        let prompt:Vec<u32>=if c["kind"]=="repository" {
            let f=&selected["cases"][ordinal];let mut p=tokens[..f["corpus_tokens"].as_u64().unwrap()as usize].to_vec();
            p.extend(serde_json::from_value::<Vec<u32>>(f["suffix_ids"].clone())?);p
        }else{serde_json::from_value(tools["cases"][ordinal]["prompt_ids"].clone())?};
        let n=prompt.len();let end=c["source_end"].as_u64().unwrap()as usize;
        let anchor=c["anchor"].as_u64().unwrap()as u32;
        ensure!(end+32<=n&&prompt[end]==anchor,"copy source frontier");
        let teacher=prompt[end..end+32].to_vec();
        let begun=execute(w,Command::Begin{images:Vec::new(),id:8_020_600+case as u64,
            seed:2026110220+case as u64,greedy:false,limit:1,tokens:prompt})?;
        ensure!(begun["target_temperature"]==1.&&begun["target_top_p"]==1.,"canonical sampler settings");
        let (bytes,cpu)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
        ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
            >=8*1024usize.pow(3)+bytes+cpu)?,"generation snapshot reserve");
        let mut alternate=if composition() {
            let mut control=crate::rejection::Rejection::new_for_graph_probe(&w.cuda,&w.core,false)?;
            w.session.as_mut().unwrap().swap_sampler_for_graph_probe(&mut control,&w.stream)?;
            Some(crate::rejection::Rejection::new_for_graph_probe(&w.cuda,&w.core,true)?)
        }else{None};
        let mut active=false;
        let image=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
        let original=state(w)?;
        w.session.as_mut().unwrap().configure_history_policy_for_probe(None)?;
        w.session.as_mut().unwrap().step(&mut w.verify,anchor,6,&w.comm,&w.stream)?;
        restore(w,&image,anchor,&original)?;
        let mut trials=Vec::new();
        if restore_profile {
            w.stream.synchronize()?;w.control.agree(b"wide-restore-profile-ready")?;
            w.cuda.profiler_start()?;w.control.agree(b"wide-restore-profile-active")?;
        }
        let masks=if restore_profile{vec![3u8,3]}else{vec![0u8,3,3,0]};
        for (trial,mask) in masks.into_iter().enumerate() {
            restore(w,&image,anchor,&original)?;schedule(w,mask)?;
            if let Some(other)=alternate.as_mut() {
                if active!=(mask==3) {w.session.as_mut().unwrap().swap_sampler_for_graph_probe(other,&w.stream)?;active=mask==3;}
                for live in 1..=6 {
                    restore(w,&image,anchor,&original)?;
                    w.session.as_mut().unwrap().step(&mut w.verify,anchor,live,&w.comm,&w.stream)?;
                }
                restore(w,&image,anchor,&original)?;
            }
            // Prime both target and compact categorical graphs outside timing.
            for live in 1..=6 {
                w.verify.begin_verification(&teacher[..live],&w.comm,&w.stream)?;
                w.verify.finish_verification(0,&w.comm,&w.stream)?;
            }
            for rows in [16usize,32] {
                let t=if rows==16 {w.long_verify.as_mut().unwrap()}else{w.copy32_verify.as_mut().unwrap()};
                t.handoff_from(&w.verify)?;
                t.begin_verification(&teacher[..rows],&w.comm,&w.stream)?;
                t.verification_history_sample(2026110230,&w.comm,&w.stream)?;
                t.finish_verification(0,&w.comm,&w.stream)?;w.verify.handoff_from(t)?;
            }
            restore(w,&image,anchor,&original)?;
            let seed=2026110240+case as u64*10+(trial/2)as u64;
            w.session.as_mut().unwrap().configure_sampling(seed,Some(w.cfg.eos_token_id))?;
            let maximum=if restore_profile{128}else{256};let mut emitted=vec![anchor];let mut pending=anchor;let mut steps=Vec::new();let mut forced=false;
            let before=captures(w);
            // State-digest downloads above are a finite correctness check,
            // not serving work. A delayed rank must not charge that check to
            // another rank's first verification collective (s1214 trace).
            w.control.agree(b"ffn32-measurement-ready")?;
            let at=Instant::now();
            while emitted.len()<maximum&&pending!=w.cfg.eos_token_id {
                let remaining=maximum-emitted.len();let force=!forced&&emitted.len()>=maximum/2;
                let s=w.session.as_mut().unwrap();if force{s.configure_history_policy_for_probe(None)?;}
                // Reuse the original buffer addresses: the prepared wide
                // conditioning graph is bound to this owner, not a new buffer.
                let mut long=if !force {s.try_long_history_step(&mut w.verify,w.copy32_verify.as_mut().unwrap(),
                    w.copy32_buffers.as_mut().unwrap(),pending,remaining,&w.comm,&w.stream)?}else{None};
                if long.is_none()&&!force {long=s.try_long_history_step(&mut w.verify,w.long_verify.as_mut().unwrap(),
                    &mut buffers16,pending,remaining,&w.comm,&w.stream)?;}
                let step=match long {Some(step)=>step,None=>s.step(&mut w.verify,pending,remaining.min(6),&w.comm,&w.stream)?};
                if force{ensure!(step.proposal_source=="dspark","forced neural fallback");forced=true;
                    s.configure_history_policy_for_probe(Some(8))?;}
                ensure!(step.tokens[0]==pending&&step.tokens.len()<=remaining,"generation output extent");
                emitted.extend_from_slice(&step.tokens[1..]);emitted.push(step.next_anchor);pending=step.next_anchor;steps.push(step);
            }
            let seconds=at.elapsed().as_secs_f64();
            ensure!(w.verify.position()==(n+emitted.len()-1)as u64,"generation target frontier");
            ensure!(steps.iter().any(|s|s.verified_rows==32),"M32 verification not exercised");
            trials.push(json!({"trial":trial,"mask":mask,"seed":seed,"seconds":seconds,
                "output_tokens":emitted.len(),"decode_tps":(emitted.len()-1)as f64/seconds,
                "output_ids":emitted,"steps":steps,"state":state(w)?,"forced_neural_fallback":forced,
                "new_target_graph_captures":captures(w)-before}));
            std::fs::write(out.join(format!("ffn32-generation-case{case}-partial.json")),
                serde_json::to_vec_pretty(&json!({"name":c["name"],"trials":trials}))?)?;
            eprintln!("M32 FFN generation rank{rank} case{case} trial{trial} mask{mask} complete");
        }
        if restore_profile {
            w.stream.synchronize()?;w.cuda.profiler_stop()?;w.control.agree(b"wide-restore-profile-finished")?;
        }
        for pair in trials.chunks_exact(2) {ensure!(w.control.all_admit(pair[0]["output_ids"]==pair[1]["output_ids"]
            &&pair[0]["state"]==pair[1]["state"])?,"M32 FFN changed emitted output or accepted state");}
        restore(w,&image,anchor,&original)?;schedule(w,3)?;
        records.push(json!({"name":c["name"],"prompt_tokens":n,"trials":trials,"exact":true}));
        std::fs::write(out.join("ffn32-generation-partial.json"),serde_json::to_vec_pretty(&records)?)?;
        execute(w,Command::Finish(Finish::Length))?;
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("ffn32-generation-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"all_exact":true,"default_overlap_layers":40,"overnight_composition":composition(),
        "wide_restore_profile":restore_profile,
        "scope":"Same-load reversed pairs; two public copy frontiers and two paired seeds. Optional overnight composition compares both wide FFN schedules, live6 expert cache policy, sampler graphs/staging and FP64 residual kernels together. TargetT1/P1, actual-q acceptance unchanged, full-session restore, neural fallbacks. Capped256-token scheduling diagnostics, not completed-task outcome quality."}))?)?;
    Ok(())
}
