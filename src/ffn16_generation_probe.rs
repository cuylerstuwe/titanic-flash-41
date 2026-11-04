//! Targeted generation and cached-append qualification of the FFN16 schedule.
use super::*;
use super::history32_generation_probe::{execute,state,restore};
fn schedule(w:&mut Worker,mask:u8)->Result<()> {
    let t=w.long_verify.as_mut().unwrap();t.handoff_from(&w.verify)?;
    t.set_verify16_schedule_for_probe(mask,&w.stream)?;
    ensure!(t.ffn_overlap_layers()==if mask==3 {(40,40)}else{(0,0)},"FFN schedule selection");
    Ok(())
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&std::env::var("CANADA_SCRATCH_FFN16_GENERATION_SCREEN").as_deref()==Ok("1")
        &&w.long_verify.as_ref().context("wide verifier")?.ffn_overlap_layers()==(40,40),
        "generation requires carried constructor default");
    let fixtures:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/history-copy-width32-v1.json"))?;
    let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    let selected:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/history-strong-backup-frontiers-v1.json"))?;
    let tools:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-history-score-v1.json"))?;
    let phases:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json"))?;
    let mut cases=Vec::new();
    for case in [1usize,2] {
        let c=&fixtures["cases"][case];let ordinal=c["frontier_index"].as_u64().unwrap()as usize;
        let prompt:Vec<u32>=if c["kind"]=="repository" {
            let f=&selected["cases"][ordinal];let mut p=tokens[..f["corpus_tokens"].as_u64().unwrap()as usize].to_vec();
            p.extend(serde_json::from_value::<Vec<u32>>(f["suffix_ids"].clone())?);p
        }else{serde_json::from_value(tools["cases"][ordinal]["prompt_ids"].clone())?};
        let end=c["source_end"].as_u64().unwrap()as usize;
        cases.push((c["name"].as_str().unwrap().to_owned(),prompt.clone(),c["anchor"].as_u64().unwrap()as u32,
            prompt[end..end+16].to_vec(),true));
    }
    let c=phases["cases"].as_array().unwrap().iter().find(|c|c["id"]=="offline-product-ideation-answer-2").unwrap();
    let anchor=c["anchor"].as_u64().unwrap()as u32;
    let mut teacher=vec![anchor];teacher.extend_from_slice(&tokens[9000..9015]);
    cases.push((c["id"].as_str().unwrap().to_owned(),serde_json::from_value(c["input_ids"].clone())?,anchor,teacher,false));
    let mut buffers=LongHistoryBuffers::new(&w.cuda,&w.core)?;
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let mut records=Vec::new();
    for (ci,(name,prompt,anchor,teacher,copy_case)) in cases.into_iter().enumerate() {
        let n=prompt.len();let begun=execute(w,Command::Begin{images:Vec::new(),id:8_020_500+ci as u64,
            seed:2026110200+ci as u64,greedy:false,limit:1,tokens:prompt})?;
        ensure!(begun["target_temperature"]==1.&&begun["target_top_p"]==1.,"canonical sampler settings");
        let (bytes,cpu)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
        ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
            >=8*1024usize.pow(3)+bytes+cpu)?,"generation snapshot reserve");
        let image=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
        let original=state(w)?;let mut append_checks=Vec::new();
        // First-use graph readiness changes whether a restored cold session
        // must query the neural drafter before it may use history proposals.
        // Warm that fallback once, then restore all accepted session state.
        w.session.as_mut().unwrap().configure_history_policy_for_probe(None)?;
        w.session.as_mut().unwrap().step(&mut w.verify,anchor,6,&w.comm,&w.stream)?;
        restore(w,&image,anchor,&original)?;
        if ci==0 {
            for live in [7usize,16] {
                for mask in [0u8,3,3,0] {
                    restore(w,&image,anchor,&original)?;schedule(w,mask)?;
                    // Warm the append graph without publishing any accepted token.
                    let t=w.long_verify.as_mut().unwrap();t.set_tiny_extend_for_probe(true,&w.stream)?;
                    t.begin_verification(&teacher[..live],&w.comm,&w.stream)?;
                    t.finish_verification(0,&w.comm,&w.stream)?;
                    let at=Instant::now();t.begin_verification(&teacher[..live],&w.comm,&w.stream)?;
                    t.finish_verification(live,&w.comm,&w.stream)?;
                    let seconds=at.elapsed().as_secs_f64();let outputs=t.valid_output_digests()?;
                    w.session.as_mut().unwrap().resume_after_medium_extend(t,&mut buffers,n as u64,
                        &teacher[..live],&w.comm,&w.stream)?;
                    t.set_tiny_extend_for_probe(false,&w.stream)?;w.verify.handoff_from(t)?;
                    append_checks.push(json!({"live":live,"mask":mask,"seconds":seconds,
                        "outputs":outputs,"state":state(w)?}));
                }
                let group=&append_checks[append_checks.len()-4..];
                ensure!(w.control.all_admit(group.iter().all(|v|v["outputs"]==group[0]["outputs"]
                    &&v["state"]==group[0]["state"]))?,"FFN cached append changed outputs/state");
            }
        }
        let mut trials=Vec::new();
        for (trial,mask) in [0u8,3,3,0].into_iter().enumerate() {
            restore(w,&image,anchor,&original)?;schedule(w,mask)?;
            // The schedule switch invalidates graphs. Capture every possible
            // verification tail before timing so first capture is not a gain.
            for live in 1..=6 {w.verify.begin_verification(&teacher[..live],&w.comm,&w.stream)?;
                w.verify.finish_verification(0,&w.comm,&w.stream)?;}
            let t=w.long_verify.as_mut().unwrap();t.handoff_from(&w.verify)?;
            for live in 7..=16 {t.begin_verification(&teacher[..live],&w.comm,&w.stream)?;
                t.finish_verification(0,&w.comm,&w.stream)?;}
            w.verify.handoff_from(t)?;
            let seed=2026110210+ci as u64*10+(trial/2)as u64;
            w.session.as_mut().unwrap().configure_sampling(seed,Some(w.cfg.eos_token_id))?;
            let maximum=256;let mut emitted=vec![anchor];let mut pending=anchor;let mut steps=Vec::new();let mut forced=false;
            let captures=w.verify.capture_statistics().1+w.long_verify.as_ref().unwrap().capture_statistics().1;
            let at=Instant::now();
            while emitted.len()<maximum&&pending!=w.cfg.eos_token_id {
                let remaining=maximum-emitted.len();let force=!forced&&emitted.len()>=maximum/2;
                let s=w.session.as_mut().unwrap();if force{s.configure_history_policy_for_probe(None)?;}
                let long=if !force {s.try_long_history_step(&mut w.verify,w.long_verify.as_mut().unwrap(),
                    &mut buffers,pending,remaining,&w.comm,&w.stream)?}else{None};
                let step=match long {Some(step)=>step,None=>s.step(&mut w.verify,pending,remaining.min(6),&w.comm,&w.stream)?};
                if force{ensure!(step.proposal_source=="dspark","forced neural fallback");forced=true;
                    s.configure_history_policy_for_probe(Some(8))?;}
                ensure!(step.tokens[0]==pending&&step.tokens.len()<=remaining,"generation output extent");
                emitted.extend_from_slice(&step.tokens[1..]);emitted.push(step.next_anchor);pending=step.next_anchor;steps.push(step);
            }
            let seconds=at.elapsed().as_secs_f64();
            ensure!(w.verify.position()==(n+emitted.len()-1)as u64,"generation target frontier");
            if copy_case{ensure!(steps.iter().any(|s|s.verified_rows==16),"wide verification not exercised");}
            trials.push(json!({"trial":trial,"mask":mask,"seed":seed,"seconds":seconds,
                "output_tokens":emitted.len(),"decode_tps":(emitted.len()-1)as f64/seconds,
                "output_ids":emitted,"steps":steps,"state":state(w)?,
                "new_target_graph_captures":w.verify.capture_statistics().1+w.long_verify.as_ref().unwrap().capture_statistics().1-captures}));
            std::fs::write(out.join(format!("ffn16-generation-case{ci}-partial.json")),
                serde_json::to_vec_pretty(&json!({"name":name,"trials":trials,"append_checks":append_checks}))?)?;
            eprintln!("FFN generation rank{rank} case{ci} trial{trial} mask{mask} complete");
        }
        for pair in trials.chunks_exact(2) {ensure!(w.control.all_admit(pair[0]["output_ids"]==pair[1]["output_ids"]
            &&pair[0]["state"]==pair[1]["state"])?,"FFN generation changed emitted output or accepted state");}
        restore(w,&image,anchor,&original)?;schedule(w,3)?;w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;
        records.push(json!({"name":name,"prompt_tokens":n,"trials":trials,"append_checks":append_checks,"exact":true}));
        std::fs::write(out.join("ffn16-generation-partial.json"),serde_json::to_vec_pretty(&records)?)?;
        execute(w,Command::Finish(Finish::Length))?;
    }
    if std::env::var("CANADA_SCRATCH_SOLO_FFN_COMBINED_SCREEN").as_deref()!=Ok("1") {
        execute(w,Command::Shutdown)?;
    }
    std::fs::write(out.join("ffn16-generation-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"all_exact":true,"default_overlap_layers":40,
        "scope":"Same-load reversed pairs; two public coding/copy frontiers and ideation answer. TargetT1/P1, unchanged actual-q acceptance; full-session restore, neural fallbacks, both append and generation graph banks. Capped256-token diagnostics, not completed-task outcome quality."}))?)?;
    Ok(())
}
