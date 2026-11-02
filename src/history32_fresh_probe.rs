//! Fresh-request screen for the finite M32 proposal path. No serving switch.
use super::*;
use super::history32_generation_probe::{execute,state,restore};

pub(super) fn run(w:&mut Worker,rank:usize,kernels:&Path,out:&Path)->Result<()> {
    ensure!(crate::config::history32_probe_enabled()
        &&std::env::var("CANADA_SCRATCH_HISTORY32_GENERATION_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_HISTORY32_FRESH_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none()&&w.long_verify.is_some(),"finite fresh M32 admission");
    let before=w.cuda.admission_memory()?;let module=w.cuda.module(&w.core)?;
    let mut wider=w.large.fork_profile(&module,&w.core,&w.stream,kernels,32,Phase::Verify)?;
    wider.set_graph_replay(true)?;
    ensure!(w.large.shared_state_report(&wider)["all_weights_and_causal_storage_shared"]==true,
        "fresh M32 duplicated canonical state");
    let mut buffers16=LongHistoryBuffers::new(&w.cuda,&w.core)?;
    let mut buffers32=LongHistoryBuffers::with_rows(&w.cuda,&w.core,32)?;
    let after=w.cuda.admission_memory()?;
    ensure!(w.control.all_admit(after.admitted_available>=9*1024usize.pow(3))?,"fresh M32 snapshot reserve");
    let fixtures:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/history32-fresh-generation-v1.json"))?;
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let mut records=Vec::new();
    for (case,c) in fixtures["cases"].as_array().context("fresh M32 cases")?.iter().enumerate() {
        let prompt:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
        let maximum=c["max_new_tokens"].as_u64().context("fresh M32 cap")? as usize;
        let seed=c["seed"].as_u64().context("fresh M32 seed")?;
        ensure!(prompt.len()>=32&&(32..=4096).contains(&maximum),"fresh M32 input bounds");
        let n=prompt.len();let mut teacher=prompt[n-32..].to_vec();
        let begin=execute(w,Command::Begin{images:Vec::new(),id:7_870_000+case as u64,
            seed,greedy:false,limit:1,tokens:prompt})?;
        ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.,"fresh M32 sampler");
        let anchor=begin["pending"].as_u64().context("fresh M32 sampled anchor")? as u32;
        ensure!(begin["output_ids"]==json!([anchor]),"fresh M32 initial output");teacher[0]=anchor;
        let (bytes,cpu)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
        ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
            >=8*1024usize.pow(3)+bytes+cpu)?,"fresh M32 bounded snapshot");
        let image=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
        let original=state(w)?;
        // Warm only graph scratch. Restore the sampled anchor and every accepted
        // state component, including seeds and history policy, before each arm.
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
        let mut trials=Vec::new();
        for (trial,width) in serde_json::from_value::<Vec<usize>>(c["order"].clone())?.into_iter().enumerate() {
            ensure!([16,32].contains(&width),"fresh M32 width");
            restore(w,&image,anchor,&original)?;
            let captures=w.verify.capture_statistics().1+w.long_verify.as_ref().unwrap().capture_statistics().1
                +wider.capture_statistics().1;
            let mut emitted=vec![anchor];let mut pending=anchor;let mut steps=Vec::new();
            let started=Instant::now();
            while emitted.len()<maximum&&pending!=w.cfg.eos_token_id {
                let remaining=maximum-emitted.len();let s=w.session.as_mut().unwrap();
                let mut long=if width==32 {
                    s.try_long_history_step(&mut w.verify,&mut wider,&mut buffers32,pending,remaining,&w.comm,&w.stream)?
                }else{None};
                if long.is_none() {
                    long=s.try_long_history_step(&mut w.verify,w.long_verify.as_mut().unwrap(),
                        &mut buffers16,pending,remaining,&w.comm,&w.stream)?;
                }
                let step=match long {Some(step)=>step,None=>s.step(&mut w.verify,pending,remaining.min(6),&w.comm,&w.stream)?};
                ensure!(step.tokens[0]==pending&&step.tokens.len()<=remaining,"fresh M32 output extent");
                emitted.extend_from_slice(&step.tokens[1..]);emitted.push(step.next_anchor);pending=step.next_anchor;
                steps.push(step);
            }
            let seconds=started.elapsed().as_secs_f64();
            ensure!(emitted.len()<=maximum&&w.verify.position()==(n+emitted.len()-1)as u64
                &&w.session.as_ref().unwrap().history_lookup_metadata().unwrap().0==w.verify.position(),"fresh M32 final frontiers");
            // A no-opportunity result is useful evidence, not a failed test.
            trials.push(json!({"trial":trial,"width":width,"seed":seed,"finish":if pending==w.cfg.eos_token_id{"eos"}else{"length"},
                "output_tokens":emitted.len(),"decode_seconds":seconds,"decode_tps":(emitted.len()-1)as f64/seconds,
                "output_ids":emitted,"steps":steps,"final_state":state(w)?,
                "new_target_graph_captures":w.verify.capture_statistics().1+w.long_verify.as_ref().unwrap().capture_statistics().1
                    +wider.capture_statistics().1-captures}));
            eprintln!("rank{rank} fresh M32 {} trial{trial} width{width}: {:.2}tps, {}tokens",
                c["name"],(emitted.len()-1)as f64/seconds,emitted.len());
        }
        restore(w,&image,anchor,&original)?;
        ensure!(w.history.frontier()==n,"fresh M32 touched worker history");
        records.push(json!({"case":case,"name":c["name"],"prompt_tokens":n,
            "begin":begin,"trials":trials,"restoration_exact":true}));
        execute(w,Command::Finish(Finish::Length))?;
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("history32-fresh-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"before_m32":before,"after_m32":after,"records":records,"all_restorations_exact":true,
        "scope":"Three fresh original T1/P1/high requests, actual sampled initial anchor. One same-load pair each; two coding tasks capped at3072 and ideation capped at768. Natural M32/M16/M6 selection, no forced copy frontier or fallback. Conditional task diagnostics, not a general speed or quality conclusion."
    }))?)?;Ok(())
}
