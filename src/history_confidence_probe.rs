//! Finite shadow confidence screen. No new proposal-selection policy yet.
use super::*;

fn execute(w:&mut Worker,c:Command)->Result<Value>{
    let bytes=c.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(bytes.as_slice()))?;
    ensure!(wire==bytes,"history confidence command mismatch");w.execute(c)
}
fn readout(w:&mut Worker,teacher:&[u32])->Result<String>{
    w.verify.begin_verification(teacher,&w.comm,&w.stream)?;
    let(y,n)=w.verify.pending_logits()?;ensure!(n==6,"following readout geometry");
    let bytes=unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?;
    w.verify.finish_verification(0,&w.comm,&w.stream)?;
    Ok(format!("{:x}",Sha256::digest(bytes)))
}
fn advance(w:&mut Worker,mut anchor:u32,count:usize,shadow:bool)->Result<Value>{
    let start=w.verify.position();let mut ids=Vec::new();let mut steps=Vec::new();
    w.control.agree(b"history-confidence-start")?;let began=Instant::now();
    while ids.len()<count && anchor!=w.cfg.eos_token_id {
        let left=count-ids.len();let s=w.session.as_mut().unwrap();
        let observation=if shadow{s.observe_long_history_for_probe(&w.verify,anchor,left,&w.comm,&w.stream)?}else{None};
        let long=s.try_long_history_step(&mut w.verify,w.long_verify.as_mut().unwrap(),
            w.long_history_buffers.as_mut().unwrap(),anchor,left,&w.comm,&w.stream)?;
        ensure!(observation.is_none() || long.is_some(),"shadow changed copy eligibility");
        let step=match long{Some(s)=>s,None=>s.step(&mut w.verify,anchor,left.min(6),&w.comm,&w.stream)?};
        ensure!(step.tokens.first()==Some(&anchor)&&step.tokens.len()<=left,"confidence frontier");
        ids.extend_from_slice(&step.tokens);anchor=step.next_anchor;
        steps.push(json!({"step":step,"observation":observation}));
    }
    w.stream.synchronize()?;w.comm.health()?;
    ensure!(w.verify.position()==start+ids.len()as u64
        &&w.session.as_ref().unwrap().history_lookup_metadata().unwrap().0==w.verify.position(),
        "confidence target/history frontiers disagree");
    Ok(json!({"shadow":shadow,"seconds":began.elapsed().as_secs_f64(),
        "committed_inputs":ids,"pending_anchor":anchor,"steps":steps,"frontier":w.verify.position()}))
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_HISTORY_CONFIDENCE_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none()&&w.long_verify.is_some()&&w.long_history_buffers.is_some()
        &&w.history_lookup_enabled,"finite history confidence admission");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture:Value=serde_json::from_slice(raw)?;let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let frontiers=include_bytes!("../tests/fixtures/history-strong-backup-frontiers-v1.json");
    let selected:Value=serde_json::from_slice(frontiers)?;
    let mut cases=Vec::new();
    for c in selected["cases"].as_array().unwrap(){
        let n=c["corpus_tokens"].as_u64().unwrap()as usize;let mut prompt=tokens[..n].to_vec();
        prompt.extend(serde_json::from_value::<Vec<u32>>(c["suffix_ids"].clone())?);
        cases.push((c["name"].as_str().unwrap().to_string(),prompt));
    }
    cases.push(("unselected-tokio-131072".to_string(),tokens[..131072].to_vec()));
    let phases=include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json");
    let phase:Value=serde_json::from_slice(phases)?;
    let c=phase["cases"].as_array().unwrap().iter()
        .find(|c|c["id"]=="offline-product-ideation-answer-2").context("ideation frontier")?;
    cases.push((c["id"].as_str().unwrap().to_string(),serde_json::from_value(c["input_ids"].clone())?));
    ensure!(cases.len()==4,"bounded confidence fixture set");
    let teacher=&tokens[262500..262516];let mut records=Vec::new();
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    for(case,(name,prompt))in cases.into_iter().enumerate(){
        let length=prompt.len();
        let begin=execute(w,Command::Begin{images:Vec::new(),id:7_820_000+case as u64,
            seed:2026110100+case as u64,greedy:false,limit:1,tokens:prompt})?;
        ensure!(begin["target_temperature"]==1. && begin["target_top_p"]==1.
            &&begin["block_verification"]==true
            &&w.session.as_ref().unwrap().draft_temperature()==0.7f32,
            "confidence screen sampling contract");
        let anchor=w.history.pending().unwrap();let frontier=w.verify.position();
        let(bytes,host)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
        ensure!(bytes+host<=1024usize.pow(3),"bounded confidence snapshot");
        ensure!(w.control.all_admit(w.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+bytes+host)?,
            "confidence snapshot admission");
        let saved=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
        let original=readout(w,&teacher[..6])?;
        for live in 1..=6{w.verify.begin_verification(&teacher[..live],&w.comm,&w.stream)?;
            w.verify.finish_verification(0,&w.comm,&w.stream)?;}
        let long=w.long_verify.as_mut().unwrap();long.handoff_from(&w.verify)?;
        long.begin_verification(teacher,&w.comm,&w.stream)?;long.finish_verification(0,&w.comm,&w.stream)?;
        w.verify.handoff_from(long)?;
        let _warm=advance(w,anchor,32,false)?;
        let mut pair=Vec::new();
        for shadow in if case%2==0{[false,true]}else{[true,false]} {
            w.target_lineage=w.target_lineage.checked_add(1).context("confidence lineage exhausted")?;
            let anchor=w.session.as_mut().unwrap().restore_state_image(&mut w.verify,&saved,
                w.target_lineage,&w.comm,&w.stream)?;
            ensure!(w.verify.position()==frontier&&readout(w,&teacher[..6])?==original,
                "confidence trial did not restore target logits");
            w.session.as_mut().unwrap().configure_sampling(2026110200+case as u64,Some(w.cfg.eos_token_id))?;
            let result=advance(w,anchor,256,shadow)?;
            let following=readout(w,&teacher[..6])?;
            let record=json!({"case":case,"name":name,"prefix_tokens":length,"result":result,
                "following_logits_sha256":following,"carry":w.verify.carry_digests()?,
                "encoder":w.verify.encoder_tail_digest()?});
            pair.push(record.clone());records.push(record);
            std::fs::write(out.join("history-confidence-partial.json"),serde_json::to_vec_pretty(&records)?)?;
        }
        for key in ["committed_inputs","pending_anchor","frontier"]{
            ensure!(pair[0]["result"][key]==pair[1]["result"][key],"shadow changed {key}");
        }
        for key in ["following_logits_sha256","carry","encoder"]{
            ensure!(pair[0][key]==pair[1][key],"shadow changed retained {key}");
        }
        eprintln!("history confidence rank{rank} case{case}: shadow/no-shadow tokens and retained state exact");
        w.target_lineage=w.target_lineage.checked_add(1).context("confidence lineage exhausted")?;
        w.session.as_mut().unwrap().restore_state_image(&mut w.verify,&saved,w.target_lineage,&w.comm,&w.stream)?;
        ensure!(w.verify.position()==w.history.frontier()as u64&&readout(w,&teacher[..6])?==original,
            "confidence final restoration");
        execute(w,Command::Finish(Finish::Length))?;
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("history-confidence-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"all_shadow_pairs_exact":true,
        "fixture_sha256":[format!("{:x}",Sha256::digest(raw)),format!("{:x}",Sha256::digest(frontiers)),format!("{:x}",Sha256::digest(phases))],
        "scope":"Four public continuation frontiers, two selected for strong history opportunities plus unselected repository and ideation controls. Paired256-token runs price five teacher-forced copy probabilities at would-be M16 copies without changing proposal policy. All q values are computed on GPU. An extra normal Markov query validates rows with the same predecessor; this diagnostic-only reference cost is excluded from observation_ms but included in continuation seconds. This is a causal-gating opportunity screen, not a qualified threshold or completed-task quality result."}))?)?;Ok(())
}
