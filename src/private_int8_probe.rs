//! Finite private-head comparison: fixed target policy and actual draft q.
use super::*;
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path,startup:Value)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_INT8_SCREEN").as_deref()==Ok("1")
        &&crate::layer_attention::generation_fp32_attention_enabled()?
        &&w.current.is_none()&&w.history.frontier()==0&&w.session_cache_budget==0
        &&w.cached_sessions.is_empty()&&!w.prompt_checkpoint&&!w.prompt_cache_logits,
        "private INT8 finite foreground admission");
    let three=std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_LOSSLESS_SCREEN").as_deref()==Ok("1");
    let raw=include_bytes!("../tests/fixtures/source-prior-transfer-tokens-v1.json");
    let fresh:Value=serde_json::from_slice(raw)?;
    ensure!(fresh["thinking_mode"]=="thinking"&&fresh["reasoning_effort"]=="high","canonical reasoning settings");
    let saved_raw=include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json");
    let saved:Value=serde_json::from_slice(saved_raw)?;
    let tool_raw=include_bytes!("../tests/fixtures/long-history-score-v1.json");
    let tool:Value=serde_json::from_slice(tool_raw)?;
    let mut cases=Vec::new();
    for id in if three{["typescript-singleflight-cache","build-queue-investigation"]}else{["rust-incremental-frames","museum-audio-prototype"]} {
        let c=fresh["cases"].as_array().unwrap().iter().find(|c|c["id"]==id).unwrap();
        cases.push((id.to_string(),"reasoning",serde_json::from_value::<Vec<u32>>(c["input_ids"].clone())?));
    }
    let c=saved["cases"].as_array().unwrap().iter().find(|c|c["id"]==if three{"offline-product-ideation-answer-3"}else{"offline-product-ideation-answer-2"}).unwrap();
    let mut ids:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;ids.push(c["anchor"].as_u64().unwrap()as u32);
    cases.push(("ideation-answer".into(),"answer",ids));
    let c=&tool["cases"][0];let mut ids:Vec<u32>=serde_json::from_value(c["prompt_ids"].clone())?;
    ids.push(c["anchor"].as_u64().unwrap()as u32);cases.push(("coding-tool-output".into(),"tool",ids));
    let mut request_id=1;let mut records=Vec::new();
    let mut progress=std::fs::OpenOptions::new().create_new(true).write(true).open(out.join("private-int8-records.jsonl"))?;
    for(ordinal,(name,phase,prompt))in cases.iter().enumerate() {
        ensure!((128..=16384).contains(&prompt.len()),"bounded private-head fixture");
        if *phase!="reasoning" {
            ensure!(prompt.iter().rposition(|&x|x==128822)>prompt.iter().rposition(|&x|x==128821),"saved output phase");
        }
        // Each arm gets one short excluded trajectory on the first prompt.
        for warmup in if ordinal==0{vec![true,false]}else{vec![false]} {
            let seed=(if three{202681800}else{202681600})+ordinal as u64;let cap=if warmup{64}else{384};
            let order=if three{match ordinal%3{0=>vec![0,1,2],1=>vec![2,0,1],_=>vec![1,2,0]}}else if ordinal%2==0{vec![0,1]}else{vec![1,0]};
            for kind in order {
                let integer=kind==1;
                let begin=w.execute(Command::Begin{images: Vec::new(),id:request_id,seed,greedy:false,limit:cap,tokens:prompt.clone()})?;request_id+=1;
                ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.
                    &&begin["generation_fp32_attention"]==true&&begin["block_verification"]==true,"target policy unchanged");
                let s=w.session.as_mut().unwrap();s.set_private_head_precision_for_probe(kind,&w.stream)?;
                let head=s.draft_head_kind();ensure!(head.contains("private_int8")==integer&&head.contains("lossless")== (kind==2),"private-head dispatch identity");
                let at=Instant::now();s.warm_draft_head_for_probe(w.history.pending().unwrap(),&w.comm,&w.stream)?;
                let preparation_seconds=at.elapsed().as_secs_f64();
                let mut ids:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;let mut steps=Vec::new();
                w.control.agree(b"private-int8-measure")?;let started=Instant::now();
                while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
                    let step=w.execute(Command::Step)?;ids.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);steps.push(step);
                }
                let seconds=started.elapsed().as_secs_f64();
                w.session.as_mut().unwrap().set_private_head_precision_for_probe(0,&w.stream)?;
                let stop=w.history.pending()==Some(w.cfg.eos_token_id);
                let finish=w.execute(Command::Finish(if stop{Finish::Stop}else{Finish::Length}))?;
                ensure!(w.history.frontier()==prompt.len()+ids.len()-1&&w.history.pending()==ids.last().copied()
                    &&ids.len()==w.emitted&&(stop||w.emitted==cap),"private INT8 committed frontier");
                let record=json!({"case":name,"phase":phase,"integer":integer,"kind":kind,"head_kind":head,"seed":seed,
                    "warmup":warmup,"cap":cap,"prompt_tokens":prompt.len(),"begin":begin,"finish":finish,
                    "output_ids":ids,"decode_seconds":seconds,"preparation_seconds":preparation_seconds,
                    "steps":steps,"original_head_restored":true});
                writeln!(progress,"{}",record)?;progress.flush()?;records.push(record);
                eprintln!("private INT8 rank{rank} {name} kind={kind} warmup={warmup}");
            }
        }
    }
    w.execute(Command::Shutdown)?;
    std::fs::write(out.join("private-int8-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,"records":records,
        "all_committed_frontiers_checked":true,"startup_warmup":startup,"target_temperature":1.,"target_top_p":1.,
        "fixtures_sha256":[format!("{:x}",Sha256::digest(raw)),format!("{:x}",Sha256::digest(saved_raw)),format!("{:x}",Sha256::digest(tool_raw))],
        "serving_policy_changed":false,"three_heads":three,"scope":if three{"Four public fronts: fresh TypeScript and build-queue reasoning, second saved ideation answer and coding tool output with a fresh seed. Fixed target T1/P1/high, FP32 generation attention, split drafter and actual-q block rejection. FP8, private INT8group128 and lossless canonical BF16 head; one seed/front,384-token caps,rotating arm order. Three excluded64-token warmups; preparation separate. No completed-task quality or generalization claim."}else{"Four public fronts: fresh Rust and ideation reasoning, saved ideation answer and coding tool output. Fixed target T1/P1/high, FP32 generation attention, split drafter and actual-q block rejection. FP8 versus private INT8group128 head; one seed per front,384-token caps,alternating arm order. Two excluded64-token warmups, graph preparation reported separately. All-rank outputs/frontiers checked; no completed-task quality or generalization claim."}}))?)?;Ok(())
}
