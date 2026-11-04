//! Finite real-model check of sampler dispatch, transfer and residual CDF paths.
use super::*;
use super::history32_generation_probe::{execute,state,restore};
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none(),"sampler generation starts idle");
    let fixtures:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json"))?;
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let mut records=Vec::new();
    for (ci,name) in ["rust-config-parser-reasoning-0","offline-product-ideation-reasoning-1",
        "offline-product-ideation-answer-2"].into_iter().enumerate() {
        let c=fixtures["cases"].as_array().unwrap().iter().find(|c|c["id"]==name).unwrap();
        let prompt:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
        let n=prompt.len();let anchor=c["anchor"].as_u64().unwrap()as u32;
        let begun=execute(w,Command::Begin{images:Vec::new(),id:8_021_400+ci as u64,
            seed:2026110270+ci as u64,greedy:false,limit:1,tokens:prompt})?;
        ensure!(begun["target_temperature"]==1.&&begun["target_top_p"]==1.,"canonical target sampling");
        let (bytes,cpu)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
        ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
            >=8*1024usize.pow(3)+bytes+cpu+64*1024*1024)?,"sampler snapshot reserve");
        let mut saved=crate::rejection::Rejection::new_for_graph_probe(&w.cuda,&w.core,false)?;
        w.session.as_mut().unwrap().swap_sampler_for_graph_probe(&mut saved,&w.stream)?;
        drop(saved);
        let mut alternate=crate::rejection::Rejection::new_for_graph_probe(&w.cuda,&w.core,true)?;
        let image=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
        let original=state(w)?;let mut active=false;
        // Warm ordinary execution and graph replay at the same accepted prefix.
        for candidate in [false,true] {
            if candidate!=active {w.session.as_mut().unwrap().swap_sampler_for_graph_probe(&mut alternate,&w.stream)?;active=candidate;}
            restore(w,&image,anchor,&original)?;
            let mut pending=anchor;
            for _ in 0..12 {
                let step=w.session.as_mut().unwrap().step(&mut w.verify,pending,6,&w.comm,&w.stream)?;
                pending=step.next_anchor;if pending==w.cfg.eos_token_id{break;}
            }
        }
        let mut trials=Vec::new();
        for (trial,candidate) in [false,true,true,false].into_iter().enumerate() {
            restore(w,&image,anchor,&original)?;
            if candidate!=active {w.session.as_mut().unwrap().swap_sampler_for_graph_probe(&mut alternate,&w.stream)?;active=candidate;}
            let seed=2026110280+ci as u64*10+(trial/2)as u64;
            w.session.as_mut().unwrap().configure_sampling(seed,Some(w.cfg.eos_token_id))?;
            let maximum=192;let mut emitted=vec![anchor];let mut pending=anchor;let mut steps=Vec::new();
            let at=Instant::now();
            while emitted.len()<maximum&&pending!=w.cfg.eos_token_id {
                let remaining=(maximum-emitted.len()).min(6);
                let step=w.session.as_mut().unwrap().step(&mut w.verify,pending,remaining,&w.comm,&w.stream)?;
                ensure!(step.tokens[0]==pending&&step.tokens.len()<=remaining,"generation output extent");
                emitted.extend_from_slice(&step.tokens[1..]);emitted.push(step.next_anchor);
                pending=step.next_anchor;steps.push(step);
            }
            let seconds=at.elapsed().as_secs_f64();
            ensure!(w.verify.position()==(n+emitted.len()-1)as u64&&steps.iter().any(|s|s.proposal_source=="dspark"),
                "sampler frontier/neural coverage");
            trials.push(json!({"trial":trial,"candidate":candidate,"seed":seed,"seconds":seconds,
                "output_tokens":emitted.len(),"decode_tps":(emitted.len()-1)as f64/seconds,
                "output_ids":emitted,"steps":steps,"state":state(w)?}));
            std::fs::write(out.join(format!("sampler-generation-case{ci}-partial.json")),serde_json::to_vec_pretty(&trials)?)?;
            eprintln!("Sampler generation rank{rank} case{ci} trial{trial} complete");
        }
        for pair in trials.chunks_exact(2) {
            ensure!(w.control.all_admit(pair[0]["output_ids"]==pair[1]["output_ids"]&&pair[0]["state"]==pair[1]["state"])?,
                "sampler path changed emitted output or accepted state");
        }
        restore(w,&image,anchor,&original)?;
        records.push(json!({"name":name,"prompt_tokens":n,"trials":trials,"exact":true}));
        execute(w,Command::Finish(Finish::Length))?;
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("sampler-generation-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,
        "records":records,"all_exact":true,
        "fused_block_finish_screen":std::env::var("CANADA_SCRATCH_BLOCK_FUSED_SCREEN").as_deref()==Ok("1"),
        "parallel_block_finish_screen":std::env::var("CANADA_SCRATCH_BLOCK_FINISH_SCREEN").as_deref()==Ok("1"),
        "mapped_input_screen":std::env::var("CANADA_SCRATCH_SAMPLER_MAPPED_SCREEN").as_deref()==Ok("1"),
        "graph_readout_screen":std::env::var("CANADA_SCRATCH_SAMPLER_GRAPH_READOUT_SCREEN").as_deref()==Ok("1"),
        "scope":"Three public reasoning/answer frontiers, reversed paired seeds at T1/P1. Identical full-session restores, generated tokens and accepted state. Capped192-token integration diagnostics; component timing carries the microsecond performance claim. Screen booleans identify the comparison, not the underlying kernel defaults."}))?)?;
    Ok(())
}
