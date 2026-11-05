//! Finite real-worker packet-policy comparison on one immutable model allocation.
use super::*;

fn execute(w:&mut Worker,c:Command)->Result<Value>{history32_generation_probe::execute(w,c)}
fn begin(w:&mut Worker,id:&mut u64,tokens:&[u32],seed:u64,limit:usize)->Result<Value>{
    let out=execute(w,Command::Begin{images:Vec::new(),id:*id,seed,greedy:false,limit,tokens:tokens.to_vec()})?;
    *id+=1;
    ensure!(out["target_temperature"]==1.&&out["target_top_p"]==1.&&out["block_verification"]==true,
        "packet workload requires the recommended target sampler");
    Ok(out)
}
fn state(w:&Worker)->Result<Value>{Ok(json!({"position":w.verify.position(),
    "carry":w.verify.carry_digests()?,"encoder":w.verify.encoder_tail_digest()?}))}
fn finish(w:&mut Worker)->Result<Value>{
    execute(w,Command::Finish(if w.history.pending()==Some(w.cfg.eos_token_id){Finish::Stop}else{Finish::Length}))
}
fn generate(w:&mut Worker,id:&mut u64,case:&Value,limit:usize)->Result<Value>{
    let prompt:Vec<u32>=serde_json::from_value(case["input_ids"].clone())?;
    // Each arm starts with the same cold input; opportunistic snapshots remain
    // enabled during the measured request and forward turns below.
    w.cached_sessions.clear();
    let at=Instant::now();let first=begin(w,id,&prompt,2026110305,limit)?;
    let begin_seconds=at.elapsed().as_secs_f64();
    let mut ids:Vec<u32>=serde_json::from_value(first["output_ids"].clone())?;
    let decode=Instant::now();let mut steps=Vec::new();
    while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id){
        let at=Instant::now();let event=execute(w,Command::Step)?;
        let seconds=at.elapsed().as_secs_f64();ids.extend(serde_json::from_value::<Vec<u32>>(event["output_ids"].clone())?);
        steps.push(json!({"seconds":seconds,"event":event}));
    }
    let decode_seconds=decode.elapsed().as_secs_f64();let end=finish(w)?;
    ensure!(w.history.frontier()==prompt.len()+ids.len()-1&&w.history.pending()==ids.last().copied(),
        "packet generation frontier accounting");
    Ok(json!({"case":case["id"],"family":case["family"],"phase":case["phase"],
        "prompt_tokens":prompt.len(),"limit":limit,"seed":2026110305u64,"begin":first,
        "begin_seconds":begin_seconds,"decode_seconds":decode_seconds,"steps":steps,"finish":end,
        "output_ids":ids,"state":state(w)?,"frontier_checked":true}))
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_ADAPTIVE_RDMA").as_deref()==Ok("1")
        &&w.current.is_none()&&!w.progress_next&&w.history.frontier()==0,"isolated packet model admission");
    let cutoff=std::env::var("CANADA_SCRATCH_PACKET_MODEL_CUTOFF").unwrap_or_else(|_|"327680".into()).parse::<u32>()?;
    ensure!([98304,327680].contains(&cutoff),"packet model cutoff must come from the component screen");
    let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;let prefix=131073;
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/confidence-economics-v1.json"))?;
    let cases=fixture["cases"].as_array().context("packet workload cases")?;ensure!(cases.len()==3,"packet case count");
    let mut id=8_030_000;w.comm.set_small_packet_bytes_for_probe(0,&w.stream)?;
    begin(w,&mut id,&tokens[..prefix],2026110304,1)?;finish(w)?;
    generate(w,&mut id,&cases[0],48)?;
    let mut records=Vec::new();let mut expected_fixed:Option<(Value,Value)>=None;
    for (phase,limit)in [0,cutoff,cutoff,0].into_iter().enumerate(){
        w.stream.synchronize()?;w.control.agree(&[0xc9,phase as u8])?;
        w.comm.set_small_packet_bytes_for_probe(limit,&w.stream)?;
        // Put a short unrelated live request in front of the cold control and
        // clear optional snapshots outside timing. Active forward reuse is then
        // tested normally, with no forced eviction between continuation turns.
        begin(w,&mut id,&[1,2,3,4],2026110304,1)?;finish(w)?;w.cached_sessions.clear();
        let at=Instant::now();let cold=begin(w,&mut id,&tokens[..prefix],2026110304,1)?;
        let cold_seconds=at.elapsed().as_secs_f64();ensure!(cold["cached_tokens"]==0,"cold packet arm unexpectedly reused state");
        let before=state(w)?;let mut verification=Vec::new();let mut checked=None;
        let mut proposal=vec![w.history.pending().context("packet pending anchor")?];proposal.extend_from_slice(&tokens[prefix..prefix+5]);
        for trial in 0..8u8{
            w.control.agree(&[0xca,phase as u8,trial])?;
            let at=Instant::now();w.verify.begin_verification(&proposal,&w.comm,&w.stream)?;
            let seconds=at.elapsed().as_secs_f64();
            if trial==1||trial==7{
                let(y,live)=w.verify.pending_logits()?;ensure!(live==6,"packet verification rows");
                let value=json!({"fp32_logits_sha256":format!("{:x}",Sha256::digest(
                    unsafe{w.cuda.read_device(y.ptr(),live*129280*4)}?))});
                if let Some(previous)=&checked{ensure!(previous==&value,"packet replay logits changed");}else{checked=Some(value);}
            }
            w.verify.finish_verification(0,&w.comm,&w.stream)?;
            if trial>=2{verification.push(seconds);}
        }
        let checked=checked.unwrap();ensure!(state(w)?==before,"packet abort changed state");
        if let Some(expected)=&expected_fixed{ensure!(expected==&(checked.clone(),before.clone()),"packet policy changed fixed-prefix logits/state");}
        else{expected_fixed=Some((checked.clone(),before.clone()));}
        finish(w)?;
        let mut forward=Vec::new();let mut offset=prefix;
        for added in [1usize,32,257]{
            let previous=w.history.frontier();let mut prompt=w.history.committed().to_vec();
            prompt.push(w.history.pending().context("packet forward anchor")?);
            prompt.extend_from_slice(&tokens[offset..offset+added]);offset+=added;
            let at=Instant::now();let event=begin(w,&mut id,&prompt,2026110304,1)?;
            let seconds=at.elapsed().as_secs_f64();
            ensure!(event["cached_tokens"].as_u64().is_some_and(|n|n as usize>=previous),"packet forward cache was lost");
            let saved=state(w)?;let end=finish(w)?;
            forward.push(json!({"added_tokens":added,"seconds":seconds,"begin":event,"finish":end,"state":saved}));
        }
        let mut generations=Vec::new();for case in cases{generations.push(generate(w,&mut id,case,192)?);}
        let record=json!({"phase":phase,"small_packet_threshold_bytes":limit,"cold":cold,"cold_seconds":cold_seconds,
            "verification_seconds":verification,"verification_output":checked,"verification_state":before,
            "forward":forward,"generation":generations,"packet_counts":w.comm.packet_counts_for_probe()?});
        std::fs::write(out.join(format!("packet-workload-phase{phase}.json")),serde_json::to_vec_pretty(&record)?)?;
        records.push(record);
    }
    w.comm.set_small_packet_bytes_for_probe(0,&w.stream)?;execute(w,Command::Shutdown)?;
    std::fs::write(out.join("packet-workload-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"prefix_tokens":prefix,"fixed_logits_and_state_exact":true,
        "scope":"One real-worker allocation, bulk/small/small/bulk.131K cold prefill, six-row fixed-prefix verification, three forward cached increments, three192-token T1/P1 generation families per arm. Normal worker command path, no HTTP/network upload measurement, no completed-task quality claim. No target arithmetic or precision changes."}))?)?;
    Ok(())
}
