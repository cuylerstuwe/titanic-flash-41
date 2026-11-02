//! Finite clock-only comparison on one allocation of the accumulated candidate.
//! Host barriers own temporary caps and independent per-device reset timers.
use super::*;

fn execute(w:&mut Worker,c:Command)->Result<Value> {
    history32_generation_probe::execute(w,c)
}
fn state(t:&Target)->Result<Value> {
    Ok(json!({"position":t.position(),"carry":t.carry_digests()?,
        "encoder":t.encoder_tail_digest()?}))
}
fn begin(w:&mut Worker,id:&mut u64,prompt:&[u32],seed:u64,limit:usize)->Result<Value> {
    let event=execute(w,Command::Begin{images:Vec::new(),id:*id,seed,
        greedy:false,limit,tokens:prompt.to_vec()})?;*id+=1;
    ensure!(event["target_temperature"]==1.&&event["target_top_p"]==1.
        &&event["block_verification"]==true,"clock workload target law");
    Ok(event)
}
fn generation(w:&mut Worker,id:&mut u64,case:&Value,limit:usize)->Result<Value> {
    let prompt:Vec<u32>=serde_json::from_value(case["input_ids"].clone())?;
    let start=Instant::now();
    let first=begin(w,id,&prompt,2026111901,limit)?;
    let begin_seconds=start.elapsed().as_secs_f64();
    let mut ids:Vec<u32>=serde_json::from_value(first["output_ids"].clone())?;
    let mut steps=Vec::new();let decode=Instant::now();
    while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
        let at=Instant::now();let event=execute(w,Command::Step)?;
        let seconds=at.elapsed().as_secs_f64();
        ids.extend(serde_json::from_value::<Vec<u32>>(event["output_ids"].clone())?);
        steps.push(json!({"seconds":seconds,"event":event}));
    }
    let decode_seconds=decode.elapsed().as_secs_f64();
    let stopped=w.history.pending()==Some(w.cfg.eos_token_id);
    let finish=execute(w,Command::Finish(if stopped{Finish::Stop}else{Finish::Length}))?;
    ensure!(w.history.frontier()==prompt.len()+ids.len()-1
        &&w.history.pending()==ids.last().copied()&&w.emitted==ids.len()
        &&(stopped||ids.len()==limit),"clock workload frontier accounting");
    Ok(json!({"case":case["id"],"family":case["family"],"prompt_tokens":prompt.len(),
        "limit":limit,"seed":2026111901u64,"begin_seconds":begin_seconds,
        "decode_seconds":decode_seconds,"begin":first,"steps":steps,"finish":finish,
        "output_ids":ids,"frontier_checked":true}))
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.history.frontier()==0&&w.long_verify.is_some()
        &&w.copy32_verify.is_some()&&w.session_cache_budget==0
        &&!w.prompt_checkpoint&&!w.prompt_cache_logits,"finite clock workload admission");
    let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    let prefix=131073;ensure!(tokens.len()>=prefix+32,"clock workload corpus extent");
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/confidence-economics-v1.json"))?;
    let cases=fixture["cases"].as_array().unwrap();ensure!(cases.len()==3,"clock workload cases");
    let mut id=1;
    // Warm long-prefill graphs and ordinary draft graphs outside all clock arms.
    begin(w,&mut id,&tokens[..prefix],2026111900,1)?;
    execute(w,Command::Finish(Finish::Length))?;
    generation(w,&mut id,&cases[0],48)?;
    let mut expected:Vec<Option<(Value,Value)>>=vec![None;4];
    let mut records=Vec::new();
    for (phase,cap) in [0u32,2350,2350,0].into_iter().enumerate() {
        w.stream.synchronize()?;w.control.agree(&[0xd1,phase as u8])?;
        let requested=json!({"phase":phase,"maximum_clock_MHz":cap,"rank":rank});
        let ready=out.join(format!("clock-ready-{phase}.json"));let temp=ready.with_extension("tmp");
        std::fs::write(&temp,serde_json::to_vec(&requested)?)?;std::fs::rename(temp,ready)?;
        let go=out.join(format!("clock-go-{phase}.json"));let wait=Instant::now();
        while !go.exists() {
            ensure!(wait.elapsed()<Duration::from_secs(90),"clock workload release deadline");
            std::thread::sleep(Duration::from_millis(10));
        }
        ensure!(serde_json::from_slice::<Value>(&std::fs::read(go)?)?==requested,"clock release identity");
        w.control.agree(&[0xd2,phase as u8])?;
        let pre_at=Instant::now();let prefill=begin(w,&mut id,&tokens[..prefix],2026111900,1)?;
        let begin_seconds=pre_at.elapsed().as_secs_f64();let mut verification=Vec::new();
        for (slot,rows) in [3usize,6,16,32].into_iter().enumerate() {
            if rows==16 {w.long_verify.as_mut().unwrap().handoff_from(&w.verify)?;}
            if rows==32 {w.copy32_verify.as_mut().unwrap().handoff_from(&w.verify)?;}
            let target=match rows {16=>w.long_verify.as_mut().unwrap(),32=>w.copy32_verify.as_mut().unwrap(),_=>&mut w.verify};
            let before=state(target)?;let mut times=Vec::new();let mut checked=None;
            for trial in 0..10u8 {
                w.control.agree(&[0xd3,phase as u8,rows as u8,trial])?;
                let at=Instant::now();target.begin_verification(&tokens[prefix..prefix+rows],&w.comm,&w.stream)?;
                let seconds=at.elapsed().as_secs_f64();
                if trial==2||trial==9 {
                    let result=if rows<16 {
                        let(y,n)=target.pending_logits()?;ensure!(n==rows,"clock logit row extent");
                        json!({"full_fp32_logits_sha256":format!("{:x}",Sha256::digest(
                            &unsafe{w.cuda.read_device(y.ptr(),rows*129280*4)}?))})
                    }else {
                        ensure!(target.pending_logits().is_err(),"clock copy path must retain compact head");
                        json!({"compact_sample_ids":target.verification_history_sample(2026111902,&w.comm,&w.stream)?})
                    };
                    if let Some(previous)=&checked {ensure!(previous==&result,"clock replay output changed");}
                    else{checked=Some(result);}
                }
                target.finish_verification(0,&w.comm,&w.stream)?;
                if trial>=2 {times.push(seconds);}
            }
            let result=checked.unwrap();ensure!(state(target)?==before,"clock rollback changed state");
            if let Some((output,saved))=&expected[slot] {
                ensure!(output==&result&&saved==&before,"clock cap changed fixed-prefix result/state");
            }else{expected[slot]=Some((result.clone(),before.clone()));}
            verification.push(json!({"rows":rows,"seconds":times,"checked_output":result,"state":before}));
            if rows==16 {w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;}
            if rows==32 {w.verify.handoff_from(w.copy32_verify.as_ref().unwrap())?;}
        }
        execute(w,Command::Finish(Finish::Length))?;
        let mut generations=Vec::new();
        for case in cases {generations.push(generation(w,&mut id,case,512)?);}
        let record=json!({"phase":phase,"cap_MHz":cap,"prefill":prefill,"begin_seconds":begin_seconds,
            "verification":verification,"generation":generations});
        std::fs::write(out.join(format!("clock-workload-phase{phase}.json")),serde_json::to_vec_pretty(&record)?)?;
        records.push(record);
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("clock-workload-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"prefix_tokens":prefix,"fixed_prefix_checks_exact":true,
        "scope":"One accumulated-candidate allocation, stock/2350/2350/stock. Warm131K prefill; M3/M6 full FP32 logit hashes, M16/M32 production compact sampling outputs and all retained-state checks. Three512-token coding/ideation/answer continuations with unchanged T1/P1 sampler and history policies. Capped diagnostics, not completed-task quality or lifetime thermal qualification."}))?)?;
    Ok(())
}
