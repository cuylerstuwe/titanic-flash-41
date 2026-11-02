//! Finite causal proposal-policy screen:64/64,32/64,32/32 for M16/M32.
use super::*;
use super::history32_generation_probe::{execute,state,restore};

pub(super) fn run(w:&mut Worker,rank:usize,kernels:&Path,out:&Path)->Result<()> {
    ensure!(crate::config::history32_probe_enabled()
        &&std::env::var("CANADA_SCRATCH_HISTORY_MATCH_GENERATION_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none()&&w.long_verify.is_some(),"finite history minimum screen");
    let before=w.cuda.admission_memory()?;let module=w.cuda.module(&w.core)?;
    let mut wider=w.large.fork_profile(&module,&w.core,&w.stream,kernels,32,Phase::Verify)?;
    wider.set_graph_replay(true)?;
    ensure!(w.large.shared_state_report(&wider)["all_weights_and_causal_storage_shared"]==true,
        "history minimum screen duplicated canonical storage");
    let mut buffers16=LongHistoryBuffers::new(&w.cuda,&w.core)?;
    let mut buffers32=LongHistoryBuffers::with_rows(&w.cuda,&w.core,32)?;
    w.prepared_session.as_mut().context("minimum screen needs prepared session")?
        .prepare_copy_conditioning(&buffers32,&w.comm,&w.stream)?;
    let after=w.cuda.admission_memory()?;
    ensure!(w.control.all_admit(after.admitted_available>=9*1024usize.pow(3))?,"minimum screen snapshot reserve");
    let fresh=std::env::var("CANADA_SCRATCH_HISTORY_MATCH_FRESH_SCREEN").as_deref()==Ok("1");
    let short=std::env::var("CANADA_SCRATCH_HISTORY_MATCH_SHORT_SCREEN").as_deref()==Ok("1");
    let adaptive=std::env::var("CANADA_SCRATCH_HISTORY_MATCH_ADAPTIVE_SCREEN").as_deref()==Ok("1");
    let moderate_fresh=std::env::var("CANADA_SCRATCH_HISTORY_MATCH_MODERATE_FRESH_SCREEN").as_deref()==Ok("1");
    ensure!(usize::from(fresh)+usize::from(short)+usize::from(adaptive)<=1,"separate history gate screens");
    ensure!(!moderate_fresh||fresh,"moderate fresh gate requires fresh mode");
    let fixtures:Value=serde_json::from_slice(if moderate_fresh{include_bytes!("../tests/fixtures/history-moderate-fresh-v1.json")}
        else if fresh{include_bytes!("../tests/fixtures/history-minimum-fresh-v1.json")}
        else if adaptive{include_bytes!("../tests/fixtures/history-adaptive-gates-v1.json")}
        else if short{include_bytes!("../tests/fixtures/history-short-match-frontiers-v1.json")}
        else{include_bytes!("../tests/fixtures/history-minimum-frontiers-v1.json")})?;
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let mut records=Vec::new();
    for (case,c) in fixtures["cases"].as_array().context("minimum cases")?.iter().enumerate() {
        let narrow_gate=short||(adaptive&&c["gate"]=="short");
        let moderate=moderate_fresh||(adaptive&&c["gate"]=="moderate");
        ensure!(!adaptive||narrow_gate||moderate,"adaptive gate identity");
        let prompt:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
        let seed=c["seed"].as_u64().context("seed")?;
        let n=prompt.len();ensure!(n>=32,"minimum prompt extent");
        let mut teacher:Vec<u32>=if fresh{prompt[n-32..].to_vec()}
            else{serde_json::from_value(c["teacher_ids"].clone())?};
        if !fresh {
            let anchor=c["anchor"].as_u64().context("conditional anchor")? as u32;
            let end=c["source_end"].as_u64().context("source end")? as usize;
            let proof=c["prefix_match"].as_u64().context("literal prefix proof")? as usize;
            ensure!([16,24,32].contains(&proof)&&teacher.len()==32&&teacher[0]==anchor&&end+32<=n&&end+1>=proof
                &&prompt[end+1-proof..end]==prompt[n+1-proof..]&&prompt[end..end+32]==teacher,
                "minimum frontier requires exact committed past prefix and source");
        }
        let begin=execute(w,Command::Begin{images:Vec::new(),id:7_910_000+case as u64,
            seed,greedy:false,limit:1,tokens:prompt})?;
        ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.,"minimum target sampler");
        let anchor=if fresh{begin["pending"].as_u64().context("fresh sampled anchor")? as u32}
            else{c["anchor"].as_u64().unwrap() as u32};
        if fresh{ensure!(begin["output_ids"]==json!([anchor]),"fresh minimum initial output");teacher[0]=anchor;}
        let (bytes,cpu)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
        ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
            >=8*1024usize.pow(3)+bytes+cpu)?,"minimum bounded snapshot");
        let image=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
        let original=state(w)?;
        w.session.as_mut().unwrap().configure_history_policy_for_probe(None)?;
        w.session.as_mut().unwrap().step(&mut w.verify,anchor,6,&w.comm,&w.stream)?;
        restore(w,&image,anchor,&original)?;
        for live in 1..=6 {
            w.verify.begin_verification(&teacher[..live],&w.comm,&w.stream)?;
            w.verify.finish_verification(0,&w.comm,&w.stream)?;
        }
        for rows in [16usize,32] {
            let target=if rows==16{w.long_verify.as_mut().unwrap()}else{&mut wider};
            target.handoff_from(&w.verify)?;target.begin_verification(&teacher[..rows],&w.comm,&w.stream)?;
            target.finish_verification(0,&w.comm,&w.stream)?;w.verify.handoff_from(target)?;
        }
        let mut trials=Vec::new();
        let order:Vec<usize>=if fresh{serde_json::from_value(c["order"].clone())?}
            else if moderate{vec![0,1,2,3,3,2,1,0]}else{vec![0,1,2,2,1,0]};
        for (trial,policy) in order.into_iter().enumerate() {
            ensure!(policy<=if moderate{3}else{2},"minimum policy index");
            restore(w,&image,anchor,&original)?;
            let (min16,min32)=if narrow_gate{[(32,32),(24,32),(16,32)][policy]}
                else if moderate{[(64,64),(32,64),(32,48),(32,32)][policy]}
                else{[(64,64),(32,64),(32,32)][policy]};
            buffers16.set_minimum_for_probe(min16)?;buffers32.set_minimum_for_probe(min32)?;
            w.session.as_mut().unwrap().configure_sampling(seed,Some(w.cfg.eos_token_id))?;
            let captures=w.verify.capture_statistics().1+w.long_verify.as_ref().unwrap().capture_statistics().1
                +wider.capture_statistics().1;
            let maximum=if fresh||adaptive{c["max_new_tokens"].as_u64().context("screen cap")? as usize}else{192};
            ensure!((32..=if moderate_fresh{12288}else{4096}).contains(&maximum),"minimum bounded continuation");
            let mut emitted=vec![anchor];let mut pending=anchor;let mut steps=Vec::new();
            w.control.agree(&[policy as u8])?;
            let started=Instant::now();
            while emitted.len()<maximum&&pending!=w.cfg.eos_token_id {
                let remaining=maximum-emitted.len();let s=w.session.as_mut().unwrap();
                let mut long=s.try_long_history_step(&mut w.verify,&mut wider,&mut buffers32,pending,remaining,&w.comm,&w.stream)?;
                if long.is_none(){long=s.try_long_history_step(&mut w.verify,w.long_verify.as_mut().unwrap(),
                    &mut buffers16,pending,remaining,&w.comm,&w.stream)?;}
                let step=match long{Some(step)=>step,None=>s.step(&mut w.verify,pending,remaining.min(6),&w.comm,&w.stream)?};
                ensure!(step.tokens[0]==pending&&step.tokens.len()<=remaining,"minimum visible output extent");
                if !fresh&&steps.is_empty() {
                    if narrow_gate {
                        let proof=c["prefix_match"].as_u64().unwrap() as usize;
                        let width=if proof>=min16{16}else{6};
                        ensure!(step.verified_rows==width&&step.lookup_match_tokens==Some(if width==16{min16}else{16}),
                            "first step did not exercise the intended shorter-match branch");
                    }else if moderate {ensure!(step.lookup_match_tokens==Some(32)&&step.verified_rows==[6,16,16,32][policy],
                        "first step did not exercise the intermediate M32 gate");
                    }else{ensure!(step.lookup_match_tokens==Some(32)&&step.verified_rows==[6,16,32][policy],
                        "first step did not exercise the intended32-only proposal branch");}
                }
                emitted.extend_from_slice(&step.tokens[1..]);emitted.push(step.next_anchor);pending=step.next_anchor;steps.push(step);
            }
            let seconds=started.elapsed().as_secs_f64();
            ensure!(emitted.len()<=maximum&&w.verify.position()==(n+emitted.len()-1)as u64
                &&w.session.as_ref().unwrap().history_lookup_metadata().unwrap().0==w.verify.position(),"minimum final frontiers");
            let hash=Sha256::digest(serde_json::to_vec(&emitted)?);w.control.agree(&hash)?;
            trials.push(json!({"trial":trial,"policy":policy,"minimum16":min16,"minimum32":min32,"seed":seed,
                "finish":if pending==w.cfg.eos_token_id{"eos"}else{"length"},"output_ids":emitted,
                "output_tokens":emitted.len(),"decode_seconds":seconds,"decode_tps":(emitted.len()-1)as f64/seconds,
                "steps":steps,"final_state":state(w)?,
                "new_target_graph_captures":w.verify.capture_statistics().1+w.long_verify.as_ref().unwrap().capture_statistics().1
                    +wider.capture_statistics().1-captures}));
            // Finite diagnostics only. Preserve a completed arm if a later arm
            // is stopped by the independent thermal or lifecycle watchdog.
            // This write is outside the timed generation region and is not a
            // successful-screen marker: restoration is checked below.
            let partial=out.join(format!("history-minimum-case{case}-trial{trial}.json"));
            let temporary=partial.with_extension("tmp");
            std::fs::write(&temporary,serde_json::to_vec_pretty(&json!({
                "rank":rank,"case":case,"name":c["name"],"prompt_tokens":n,
                "trial":trials.last(),"screen_complete":false,"restoration_checked":false,
            }))?)?;
            std::fs::rename(temporary,partial)?;
            eprintln!("rank{rank} minimum case{case} trial{trial} min16={min16} min32={min32}: {:.2}tps",(emitted.len()-1)as f64/seconds);
        }
        restore(w,&image,anchor,&original)?;
        ensure!(w.history.frontier()==n,"minimum screen touched worker history");
        records.push(json!({"case":case,"name":c["name"],"phase":c["phase"],"gate":c["gate"],"prompt_tokens":n,
            "trials":trials,"restoration_exact":true}));execute(w,Command::Finish(Finish::Length))?;
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("history-minimum-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"before_m32":before,"after_m32":after,"records":records,"all_restorations_exact":true,"fresh":fresh,"short":short,"adaptive":adaptive,"moderate_fresh":moderate_fresh,
        "scope":if moderate_fresh{"Original held-out fresh byte-cache repair T1/P1/high, actual sampled anchor.32/48 versus64/64, cap12288 to permit completion. Same load, one pair. Grade final code and report token mix separately; no general quality or speed claim."}
        else if fresh{"Two predetermined original fresh T1/P1/high prompts, actual sampled anchor. Baseline64/64 versus32/32 eligibility, order reversed across tasks. Code repair cap4096; ideation cap1024. Report completion and token mix separately; no general quality or speed claim."}
        else if adaptive{"Four128-token conditional public windows under one load. Two compare64/64,32/64,32/48,32/32 gates; two compare M16 gates32/24/16 with M32 held32. Literal extra past-prefix checks require no model query or extra table. Reversed policies; targetT1/P1 unchanged. Not fresh outcomes or a general performance claim."}
        else if short{"Two conditional192-token public continuations: first16-key source with24 exact past tokens in reasoning, first16-key source failing24 proof in answer. Compare32/24/16 M16 gates, M32 held at32. Selection uses past tokens only; reversed policies, targetT1/P1 unchanged. Not fresh outcomes."}
        else{"Same-load192-token conditional continuations, three causal long-copy eligibility policies in reversed order. First eligible32-only public reasoning/answer prefixes, selected without future acceptance. Target T1/P1 unchanged. Different verification widths may alter floating-point logits; no exact cross-policy requirement or completed-task quality claim."}
    }))?)?;Ok(())
}
