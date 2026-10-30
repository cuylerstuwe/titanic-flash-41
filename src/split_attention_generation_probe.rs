//! Short foreground generation screen; differing arithmetic may change tokens.
use super::*;
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_SPLIT_ATTENTION_GENERATION_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none()&&w.history.frontier()==0&&w.session_cache_budget==0
        &&!w.prompt_checkpoint&&!w.prompt_cache_logits,"split attention generation admission");
    let outcome=std::env::var("CANADA_SCRATCH_SPLIT_ATTENTION_OUTCOME_SCREEN").as_deref()==Ok("1");
    let raw:&[u8]=if outcome{include_bytes!("../tests/fixtures/resident-small-append-outcomes-v1.json")}
        else{include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json")};
    let fixture:Value=serde_json::from_slice(raw)?;
    let mut cases=Vec::new();
    if outcome {
        // Use the canonical high-reasoning task prompts, with no artificial
        // retained suffix. A larger cap avoids the earlier4096-token censoring.
        for (id,first) in [("select-jobs",0usize),("resolve-aliases",4)] {
            let mut ids:Vec<u32>=serde_json::from_value(fixture[first]["input_ids"].clone())?;
            ids.extend(serde_json::from_value::<Vec<u32>>(fixture[first+1]["append_tokens"].clone())?);
            ensure!(ids.last()==Some(&128821),"expected canonical thinking opener");
            cases.push((id.to_owned(),ids));
        }
    }else{
    for id in ["rust-config-parser-reasoning-1","offline-product-ideation-answer-3"] {
        let c=fixture["cases"].as_array().unwrap().iter().find(|c|c["id"]==id).unwrap();
        cases.push((id.to_owned(),serde_json::from_value::<Vec<u32>>(c["input_ids"].clone())?));
    }
    let long_raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let long:Value=serde_json::from_slice(long_raw)?;
    let mut ids:Vec<u32>=serde_json::from_value(long["tokens"].clone())?;
    ensure!(ids.len()>131072,"long attention generation fixture extent");
    ids.truncate(131072);cases.push(("long-agentic-context".into(),ids));
    }
    let cap=if outcome{12288}else{320};let repeats=if outcome{1}else{2};
    let mut records=Vec::new();let mut request_id=7_860_000;
    for (case,(id,prompt)) in cases.iter().enumerate() {
        for repeat in 0..repeats {
            let seed=if outcome{202679800+case as u64}else{202678600+case as u64*2+repeat as u64};
            let control_first=if outcome{case%2==0}else{repeat==0};
            for path in if control_first {[0u8,2]}else{[2,0]} {
                w.control.agree(&[case as u8,repeat,path])?;
                request_id+=1;
                let request_at=Instant::now();
                let begin=w.execute(Command::Begin{images: Vec::new(),id:request_id,seed,greedy:false,limit:cap,tokens:prompt.clone()})?;
                ensure!(begin["cached_tokens"]==0&&begin["target_temperature"]==1.
                    &&begin["target_top_p"]==1.&&begin["block_verification"]==true,
                    "attention screen changed target sampling or reused a prefix");
                // Begin has transferred the complete owner to Verify6. This
                // finite pre-warmup screen never creates tiny-append graph banks.
                w.verify.set_split_attention_for_probe(path,&w.stream)?;
                let mut output_ids:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;
                let mut steps=Vec::new();let mut measured_at=None;
                let mut measured_start=0;let mut measured_step_start=0;
                let all_at=Instant::now();
                while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
                    let step=w.execute(Command::Step)?;
                    output_ids.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);
                    steps.push(step);
                    if measured_at.is_none()&&w.emitted>=64 {
                        w.control.agree(b"split-attention-generation-window")?;
                        measured_start=output_ids.len();measured_step_start=steps.len();
                        measured_at=Some(Instant::now());
                    }
                }
                let measured_seconds=measured_at.map(|t|t.elapsed().as_secs_f64());
                let all_decode_seconds=all_at.elapsed().as_secs_f64();
                let stop=w.history.pending()==Some(w.cfg.eos_token_id);
                let finish=w.execute(Command::Finish(if stop{Finish::Stop}else{Finish::Length}))?;
                let total_request_seconds=request_at.elapsed().as_secs_f64();
                ensure!(w.history.frontier()==prompt.len()+output_ids.len()-1
                    &&w.history.pending()==output_ids.last().copied()&&output_ids.len()==w.emitted
                    &&(stop||w.emitted==cap),"attention generation committed frontier");
                let measured_tokens=if measured_seconds.is_some(){output_ids.len()-measured_start}else{0};
                records.push(json!({"case":case,"id":id,"repeat":repeat,"path":path,"seed":seed,
                    "prompt_tokens":prompt.len(),"output_ids":output_ids,"cap":cap,"begin":begin,"finish":finish,
                    "total_request_seconds":total_request_seconds,
                    "measured_tokens":measured_tokens,"measured_step_start":measured_step_start,
                    "measured_seconds":measured_seconds,"all_decode_seconds":all_decode_seconds,"steps":steps,
                    "conditioning_split":w.session.as_ref().unwrap().conditioning_split()}));
                std::fs::write(out.join("split-generation-partial.json"),serde_json::to_vec(&records)?)?;
                eprintln!("split generation rank{rank} {id} repeat{repeat} path{path}");
            }
        }
    }
    w.execute(Command::Shutdown)?;
    std::fs::write(out.join("split-generation-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"all_committed_frontiers_checked":true,
        "fixture_sha256":if outcome{vec![format!("{:x}",Sha256::digest(raw))]}else{vec![format!("{:x}",Sha256::digest(raw)),format!("{:x}",Sha256::digest(include_bytes!("../tests/fixtures/long-context-continuation-v2.json")))]},
        "outcome_screen":outcome,"case_count":cases.len(),"repeats":repeats,
        "fp32_partials":std::env::var("CANADA_SCRATCH_SPLIT_ATTENTION_FP32").as_deref()==Ok("1"),
        "scope":if outcome{"Two small coding tasks, one fresh seed per task, order counterbalanced across tasks, T1/P1/high reasoning. FP8 versus splitBF16QK/FP32partials generation;12288-token cap, complete answers graded separately. Total time includes prefill, initial graph work and all reasoning; warm TPS excludes first64outputs. Four samples are a functional/outcome screen, not general quality or performance evidence; short contexts only."}else{"Rust reasoning, ideation answer and131Kreal agentic context; FP8 versus splitQKBF16 verification, two reversed seeds,T1/P1,320-token caps. Exclude first64outputs for graph warmup; original target weights and drafter/rejection policies. Numerical paths may produce different trajectories. Foreground-only mechanism screen, not completed-task quality or production qualification."}}))?)?;
    Ok(())
}
