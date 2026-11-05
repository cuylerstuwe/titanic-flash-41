//! Finite private-head comparison: fixed target policy and actual draft q.
use super::*;
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path,startup:Value)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_NVFP4_SCREEN").as_deref()==Ok("1")
        &&crate::layer_attention::generation_fp32_attention_enabled()?
        &&w.current.is_none()&&w.history.frontier()==0
        &&w.cached_sessions.is_empty()&&w.history_lookup_enabled,
        "private NVFP4 finite foreground admission");
    let raw=include_bytes!("../tests/fixtures/source-prior-transfer-tokens-v1.json");
    let fresh:Value=serde_json::from_slice(raw)?;
    ensure!(fresh["thinking_mode"]=="thinking"&&fresh["reasoning_effort"]=="high","canonical reasoning settings");
    let saved_raw=include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json");
    let saved:Value=serde_json::from_slice(saved_raw)?;
    let tool_raw=include_bytes!("../tests/fixtures/long-history-score-v1.json");
    let tool:Value=serde_json::from_slice(tool_raw)?;
    let mut cases=Vec::new();
    for id in ["rust-incremental-frames","museum-audio-prototype"] {
        let c=fresh["cases"].as_array().unwrap().iter().find(|c|c["id"]==id).unwrap();
        cases.push((id.to_string(),"reasoning",serde_json::from_value::<Vec<u32>>(c["input_ids"].clone())?));
    }
    let c=saved["cases"].as_array().unwrap().iter().find(|c|c["id"]=="offline-product-ideation-answer-2").unwrap();
    let mut ids:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;ids.push(c["anchor"].as_u64().unwrap()as u32);
    cases.push(("ideation-answer".into(),"answer",ids));
    let c=&tool["cases"][0];let mut ids:Vec<u32>=serde_json::from_value(c["prompt_ids"].clone())?;
    ids.push(c["anchor"].as_u64().unwrap()as u32);cases.push(("coding-tool-output".into(),"tool",ids));
    let mut request_id=1;let mut records=Vec::new();
    let mut progress=std::fs::OpenOptions::new().create_new(true).write(true).open(out.join("private-nvfp4-records.jsonl"))?;
    for(ordinal,(name,phase,prompt))in cases.iter().enumerate() {
        ensure!((128..=16384).contains(&prompt.len()),"bounded private-head fixture");
        if *phase!="reasoning" {
            ensure!(prompt.iter().rposition(|&x|x==128822)>prompt.iter().rposition(|&x|x==128821),"saved output phase");
        }
        // Each arm gets one short excluded trajectory on the first prompt.
        for warmup in if ordinal==0{vec![true,false]}else{vec![false]} {
            let seed=2026110300+ordinal as u64;let cap=if warmup{32}else{256};
            let order=if ordinal%2==0{[0,3]}else{[3,0]};
            for kind in order {
                let nvfp4=kind==3;
                let begin=w.execute(Command::Begin{images: Vec::new(),id:request_id,seed,greedy:false,limit:cap,tokens:prompt.clone()})?;request_id+=1;
                ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.
                    &&begin["generation_fp32_attention"]==true&&begin["block_verification"]==true,"target policy unchanged");
                let s=w.session.as_mut().unwrap();s.set_private_head_precision_for_probe(kind,&w.stream)?;
                let head=s.draft_head_kind();ensure!(head.contains("private_nvfp4")==nvfp4&&!head.contains("lossless"),"private-head dispatch identity");
                let at=Instant::now();s.warm_draft_head_for_probe(w.history.pending().unwrap(),&w.comm,&w.stream)?;
                let preparation_seconds=at.elapsed().as_secs_f64();
                let mut ids:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;let mut steps=Vec::new();
                w.control.agree(b"private-nvfp4-measure")?;let started=Instant::now();
                while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
                    let step=w.execute(Command::Step)?;ids.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);steps.push(step);
                }
                let seconds=started.elapsed().as_secs_f64();
                w.session.as_mut().unwrap().set_private_head_precision_for_probe(0,&w.stream)?;
                let stop=w.history.pending()==Some(w.cfg.eos_token_id);
                let finish=w.execute(Command::Finish(if stop{Finish::Stop}else{Finish::Length}))?;
                ensure!(w.history.frontier()==prompt.len()+ids.len()-1&&w.history.pending()==ids.last().copied()
                    &&ids.len()==w.emitted&&(stop||w.emitted==cap),"private NVFP4 committed frontier");
                let record=json!({"case":name,"phase":phase,"nvfp4":nvfp4,"kind":kind,"head_kind":head,"seed":seed,
                    "warmup":warmup,"cap":cap,"prompt_tokens":prompt.len(),"begin":begin,"finish":finish,
                    "output_ids":ids,"decode_seconds":seconds,"preparation_seconds":preparation_seconds,
                    "steps":steps,"original_head_restored":true});
                writeln!(progress,"{}",record)?;progress.flush()?;records.push(record);
                eprintln!("private NVFP4 rank{rank} {name} kind={kind} warmup={warmup}");
            }
        }
    }
    w.execute(Command::Shutdown)?;
    std::fs::write(out.join("private-nvfp4-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,"records":records,
        "all_committed_frontiers_checked":true,"startup_warmup":startup,"target_temperature":1.,"target_top_p":1.,
        "fixtures_sha256":[format!("{:x}",Sha256::digest(raw)),format!("{:x}",Sha256::digest(saved_raw)),format!("{:x}",Sha256::digest(tool_raw))],
        "serving_policy_changed":false,"scope":"Four public fronts: fresh Rust and ideation reasoning, saved ideation answer and coding tool output. Fixed target T1/P1/high, FP32 generation attention, split drafter and actual-q block rejection. FP8 versus private NVFP4 E2M1 with E4M3 scales/16, Marlin BF16 operands/output and FP32 reduction/publication; one seed per front,256-token caps,alternating arm order. Two excluded32-token warmups, graph preparation reported separately. Existing history-copy and width policies retained. All-rank outputs/frontiers checked; no completed-task quality or generalization claim."}))?)?;Ok(())
}
