//! Bounded same-allocation checks for packet policy, owner CPU placement and QKV fences.
use super::*;

#[cfg(target_os="linux")]
fn cpus()->Result<Vec<usize>>{
    let mut mask:libc::cpu_set_t=unsafe{std::mem::zeroed()};
    ensure!(unsafe{libc::sched_getaffinity(0,std::mem::size_of_val(&mask),&mut mask)}==0,"read owner affinity");
    Ok((0..libc::CPU_SETSIZE as usize).filter(|&c|unsafe{libc::CPU_ISSET(c,&mask)}).collect())
}
#[cfg(not(target_os="linux"))]
fn cpus()->Result<Vec<usize>>{anyhow::bail!("Linux placement probe")}
#[cfg(target_os="linux")]
fn pin(cpus:&[usize])->Result<()>{
    ensure!(!cpus.is_empty()&&cpus.iter().all(|&c|c<libc::CPU_SETSIZE as usize),"CPU mask bounds");
    let mut mask:libc::cpu_set_t=unsafe{std::mem::zeroed()};
    for &c in cpus{unsafe{libc::CPU_SET(c,&mut mask)}}
    ensure!(unsafe{libc::sched_setaffinity(0,std::mem::size_of_val(&mask),&mask)}==0,"set owner affinity");Ok(())
}
#[cfg(not(target_os="linux"))]
fn pin(_: &[usize])->Result<()>{anyhow::bail!("Linux placement probe")}
fn state(t:&Target)->Result<Value>{Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))}
fn logits(t:&Target,cuda:&Cuda)->Result<String>{
    // Every rank checks all its vocabulary columns, including the compact
    // M16 path that intentionally does not gather a full-vocabulary buffer.
    let(y,live)=t.pending_local_logits_for_probe()?;
    Ok(format!("{:x}",Sha256::digest(unsafe{cuda.read_device(y.ptr(),live*32320*4)}?)))
}
fn policy(w:&Worker,mode:u8,original:&[usize],performance:&[usize])->Result<()>{
    w.stream.synchronize()?;w.comm.set_small_packet_bytes_for_probe(if mode&1!=0{327680}else{0},&w.stream)?;
    pin(if mode&2!=0{performance}else{original})?;w.comm.health()
}
fn generation(w:&mut Worker,id:u64,case:&Value,limit:usize,warm:bool)->Result<Value>{
    generation_with_prefix_policy(w,id,case,limit,warm,None)
}
fn generation_with_prefix_policy(w:&mut Worker,id:u64,case:&Value,limit:usize,warm:bool,prefix_stop:Option<bool>)->Result<Value>{
    let tokens:Vec<u32>=serde_json::from_value(case["input_ids"].clone())?;
    w.cached_sessions.clear();w.progress_next=true;
    let start=Instant::now();let begin=Command::Begin{images:Vec::new(),id,seed:2026110311,greedy:false,limit,tokens:tokens.clone()}.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(begin.as_slice()))?;
    let first=w.execute(Command::decode(&wire)?)?;let mut next=w.agree_progress(&first,false)?;
    let begin_seconds=start.elapsed().as_secs_f64();let mut ids:Vec<u32>=serde_json::from_value(first["output_ids"].clone())?;
    if warm{
        for live in 1..=6{w.verify.begin_verification(&tokens[..live],&w.comm,&w.stream)?;w.verify.finish_verification(0,&w.comm,&w.stream)?;}
    }
    if let Some(enabled)=prefix_stop {
        w.session.as_mut().context("prefix policy needs a session")?.configure_prefix_stop_for_probe(enabled,&w.stream)?;
    }
    let captures=w.verify.capture_statistics().1;let start=Instant::now();let mut steps=Vec::new();
    let finish=loop{
        let at=Instant::now();let event=w.execute(Command::decode(&next)?)?;
        if event["type"]=="finish"{break event}
        ensure!(event["type"]=="tokens","placement generation event");
        let agree=Instant::now();next=w.agree_progress(&event,false)?;let agreement_seconds=agree.elapsed().as_secs_f64();
        let seconds=at.elapsed().as_secs_f64();ids.extend(serde_json::from_value::<Vec<u32>>(event["output_ids"].clone())?);
        steps.push(json!({"seconds":seconds,"agreement_seconds":agreement_seconds,"event":event}));
    };
    let decode_seconds=start.elapsed().as_secs_f64();w.progress_next=false;
    ensure!(w.history.frontier()==tokens.len()+ids.len()-1&&w.history.pending()==ids.last().copied(),"placement frontier accounting");
    let retained=if prefix_stop.is_some(){history32_generation_probe::state(w)?}else{state(&w.verify)?};
    Ok(json!({"case":case["id"],"begin":first,"begin_seconds":begin_seconds,"decode_seconds":decode_seconds,
        "steps":steps,"output_ids":ids,"finish":finish,"new_verifier_captures":w.verify.capture_statistics().1-captures,"state":retained}))
}

pub(super) fn run_sharded_prefix(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.history.frontier()==0
        &&std::env::var("CANADA_SCRATCH_SHARDED_PREFIX_NATIVE_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_NATIVE_PREFIX_STOP_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_HIERARCHICAL_DRAFT").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_CONTROL_BUSY_POLL_US").as_deref()==Ok("500")
        &&std::env::var("CANADA_SCRATCH_PREFIX_DRAFT_STOP_GPU").as_deref()!=Ok("1"),
        "fresh sharded host-prefix screen with current control policy");
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/confidence-economics-v1.json"))?;
    let cases=fixture["cases"].as_array().context("sharded prefix cases")?;
    let selected=[&cases[0],&cases[2]];let mut id=8_090_000;
    // Warm both Markov graph paths and all target widths before the short
    // ABBA screen. Begin, captures and retained-state hashing are not timed
    // as decode; ordinary execute/progress agreement and Finish are timed.
    for enabled in [false,true] {
        for case in selected {generation_with_prefix_policy(w,id,case,192,true,Some(enabled))?;id+=1;}
    }
    let mut records=Vec::new();let mut expected:Vec<Option<Value>>=vec![None,None];
    for (trial,enabled)in [false,true,true,false].into_iter().enumerate(){
        w.control.agree(&[0xd7,trial as u8])?;
        for(case_index,case)in selected.into_iter().enumerate(){
            let result=generation_with_prefix_policy(w,id,case,192,false,Some(enabled))?;id+=1;
            ensure!(result["new_verifier_captures"]==0,"sharded prefix recaptured verifier");
            let decisions:Vec<Value>=result["steps"].as_array().context("prefix steps")?.iter().map(|step|{
                let e=&step["event"];
                json!({"ids":e["output_ids"],"verified_rows":e["verified_rows"],
                    "accepted":e["accepted_draft_tokens"],"committed":e["committed_model_rows"],
                    "source":e["proposal_source"],"lookup":e["lookup_match_tokens"]})
            }).collect();
            let signature=json!({"ids":result["output_ids"],"state":result["state"],"decisions":decisions});
            let exact=expected[case_index].as_ref().is_none_or(|value|value==&signature);
            records.push(json!({"trial":trial,"prefix_stop":enabled,"same_outputs_state_and_decisions":exact,"result":result}));
            std::fs::write(out.join("sharded-prefix-native-partial.json"),serde_json::to_vec_pretty(&records)?)?;
            ensure!(w.control.all_admit(exact)?,"sharded prefix changed outputs, retained state or decisions");
            if expected[case_index].is_none(){expected[case_index]=Some(signature);}
        }
    }
    w.session.as_mut().unwrap().configure_prefix_stop_for_probe(false,&w.stream)?;
    history32_generation_probe::execute(w,Command::Shutdown)?;
    std::fs::write(out.join("sharded-prefix-native-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"all_exact":true,"prefix_stop_restored_off":true,
        "scope":"Same-allocation off/on/on/off host early-draft stopping with current sharded q, confidence policy and500us socket polling. Two public192-token T1/P1 coding/ideation continuations. Exact outputs, decisions and retained target/drafter/lookup state on all ranks. No sampling, verification-width, prefill or target-math policy change. Capped mechanism screen, not completed-task or HTTP measurement."}))?)?;Ok(())
}

pub(super) fn run_kernel_poll(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.history.frontier()==0
        &&std::env::var("CANADA_SCRATCH_CONTROL_KERNEL_POLL_SCREEN").as_deref()==Ok("1"),
        "fresh finite control kernel polling screen");
    let original=cpus()?;
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/confidence-economics-v1.json"))?;
    let cases=fixture["cases"].as_array().context("control polling cases")?;
    let selected=[&cases[0],&cases[2]];
    let mut id=8_070_000;
    w.control.set_busy_poll_for_probe(0)?;
    for case in selected {generation(w,id,case,192,true)?;id+=1;}
    let mut records=Vec::new();let mut expected:Vec<Option<Value>>=vec![None,None];
    for (trial,poll_us)in [0,500,500,0].into_iter().enumerate(){
        let napi=w.control.set_busy_poll_for_probe(poll_us)?;
        w.control.agree(&[0xd6,trial as u8])?;
        for (case_index,case)in selected.into_iter().enumerate(){
            let result=generation(w,id,case,192,false)?;id+=1;
            ensure!(result["new_verifier_captures"]==0,"control polling recaptured verifier");
            let signature=json!({"ids":result["output_ids"],"state":result["state"]});
            let exact=expected[case_index].as_ref().is_none_or(|value|value==&signature);
            records.push(json!({"trial":trial,"poll_us":poll_us,"napi_ids":napi,
                "same_outputs_and_state":exact,"result":result}));
            std::fs::write(out.join("control-kernel-poll-partial.json"),serde_json::to_vec_pretty(&records)?)?;
            ensure!(w.control.all_admit(exact)?,"control polling changed model outputs or retained state");
            if expected[case_index].is_none(){expected[case_index]=Some(signature);}
        }
    }
    w.control.set_busy_poll_for_probe(0)?;
    ensure!(cpus()?==original,"control polling changed owner affinity");
    history32_generation_probe::execute(w,Command::Shutdown)?;
    std::fs::write(out.join("control-kernel-poll-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"all_exact":true,"polling_restored_zero":true,
        "scope":"Same allocation off/500us/500us/off per-socket kernel polling. Two192-token T1/P1 public coding/ideation continuations after every verifier width warms. Ordinary progress-next framing, all-rank identical output and retained-state checks. No global settings, affinity, protocol or model-math change; no completed-task or HTTP claim."}))?)?;
    Ok(())
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path,kernels:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_QKV_PRODUCER_FENCE").as_deref()==Ok("0"),
        "paired QKV probe must start with its explicit consumer control");
    ensure!(w.current.is_none()&&w.history.frontier()==0&&w.long_verify.is_some()
        &&std::env::var("CANADA_SCRATCH_ADAPTIVE_RDMA").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_QKV_FENCE_SCREEN").as_deref()==Ok("1"),"isolated transport/fence probe");
    let original=cpus()?;let capacities=original.iter().map(|&c|Ok((c,std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{c}/cpu_capacity"))?.trim().parse::<u32>()?))).collect::<Result<Vec<_>>>()?;
    let maximum=capacities.iter().map(|x|x.1).max().context("CPU capacities")?;
    let performance:Vec<_>=capacities.iter().filter(|x|x.1*100>=maximum*90).map(|x|x.0).collect();
    ensure!(!performance.is_empty()&&performance.len()<original.len(),"owner needs both CPU classes");
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;let prefix=131073;
    history32_generation_probe::execute(w,Command::Begin{images:Vec::new(),id:8_040_000,seed:2026110310,greedy:false,limit:1,tokens:tokens[..prefix].to_vec()})?;
    let mut fixed=Vec::new();let mut fences=Vec::new();
    for physical in [6usize,16]{
        if physical==16{w.long_verify.as_mut().unwrap().handoff_from(&w.verify)?;}
        for live in if physical==6{vec![3usize,6]}else{vec![16]}{
            let mut reference=None;
            for (trial,mode)in [0u8,1,3,2,2,3,1,0].into_iter().enumerate(){
                policy(w,mode,&original,&performance)?;w.control.agree(&[0xcf,physical as u8,live as u8,trial as u8])?;
                let t=if physical==6{&mut w.verify}else{w.long_verify.as_mut().unwrap()};let before=state(t)?;
                for _ in 0..3{t.begin_verification(&tokens[prefix..prefix+live],&w.comm,&w.stream)?;t.finish_verification(0,&w.comm,&w.stream)?;}
                let captures=t.capture_statistics().1;let mut samples=Vec::new();
                for _ in 0..8{let at=Instant::now();t.begin_verification(&tokens[prefix..prefix+live],&w.comm,&w.stream)?;samples.push(at.elapsed().as_secs_f64());t.finish_verification(0,&w.comm,&w.stream)?;}
                ensure!(t.capture_statistics().1==captures,"timed placement recapture");
                let mut checks=Vec::new();for offset in [prefix,prefix+16]{t.begin_verification(&tokens[offset..offset+live],&w.comm,&w.stream)?;checks.push(logits(t,&w.cuda)?);t.finish_verification(0,&w.comm,&w.stream)?;}
                ensure!(checks[0]!=checks[1]&&state(t)?==before,"placement changed input/state check");
                if let Some(expected)=&reference{ensure!(&checks==expected,"placement changed logits");}else{reference=Some(checks.clone());}
                fixed.push(json!({"physical_rows":physical,"live":live,"trial":trial,"mode":mode,"seconds":samples,"checks":checks,"state_exact":true}));
                std::fs::write(out.join("transport-placement-fixed.json"),serde_json::to_vec_pretty(&json!({"fixed":fixed,"fences":fences}))?)?;
            }
        }
        policy(w,0,&original,&performance)?;
        let t=if physical==6{&mut w.verify}else{w.long_verify.as_mut().unwrap()};
        let mut other=t.prepare_qkv_fence_alternates_for_probe(kernels)?;let mut active=false;
        let before=state(t)?;let mut reference=None;
        for (trial,producer)in [false,true,true,false].into_iter().enumerate(){
            if active!=producer{t.swap_qkv_fence_for_probe(&mut other,&w.stream)?;active=producer;}
            for _ in 0..6{t.begin_verification(&tokens[prefix..prefix+physical],&w.comm,&w.stream)?;t.finish_verification(0,&w.comm,&w.stream)?;}
            let captures=t.capture_statistics().1;let mut samples=Vec::new();
            for _ in 0..16{let at=Instant::now();t.begin_verification(&tokens[prefix..prefix+physical],&w.comm,&w.stream)?;samples.push(at.elapsed().as_secs_f64());t.finish_verification(0,&w.comm,&w.stream)?;}
            ensure!(captures==t.capture_statistics().1,"timed QKV recapture");
            let mut checks=Vec::new();for offset in [prefix,prefix+16]{t.begin_verification(&tokens[offset..offset+physical],&w.comm,&w.stream)?;checks.push(logits(t,&w.cuda)?);t.finish_verification(0,&w.comm,&w.stream)?;}
            ensure!(checks[0]!=checks[1]&&state(t)?==before,"QKV changed input/state check");
            if let Some(expected)=&reference{ensure!(expected==&checks,"QKV fence changed logits");}else{reference=Some(checks.clone());}
            fences.push(json!({"physical_rows":physical,"trial":trial,"producer":producer,"seconds":samples,"checks":checks,"state_exact":true}));
            std::fs::write(out.join("transport-placement-fixed.json"),serde_json::to_vec_pretty(&json!({"fixed":fixed,"fences":fences}))?)?;
        }
        ensure!(!active,"original QKV restoration");drop(other);
        if physical==16{w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;}
    }
    history32_generation_probe::execute(w,Command::Finish(Finish::Length))?;
    std::fs::write(out.join("transport-placement-fixed.json"),serde_json::to_vec_pretty(&json!({"fixed":fixed,"fences":fences}))?)?;
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/confidence-economics-v1.json"))?;
    let cases=&fixture["cases"].as_array().context("placement cases")?;
    let selected=[&cases[0],&cases[2]];let mut id=8_040_010;
    for case in selected{generation(w,id,case,128,true)?;id+=1;}
    let mut generations=Vec::new();
    for (trial,mode)in [0u8,1,3,2,2,3,1,0].into_iter().enumerate(){
        policy(w,mode,&original,&performance)?;w.control.agree(&[0xd0,trial as u8])?;
        for case in selected{let result=generation(w,id,case,128,false)?;id+=1;ensure!(result["new_verifier_captures"]==0,"placement generation had unwarmed verifier");generations.push(json!({"trial":trial,"mode":mode,"result":result}));}
    }
    policy(w,0,&original,&performance)?;ensure!(cpus()?==original,"owner affinity restoration");
    history32_generation_probe::execute(w,Command::Shutdown)?;
    std::fs::write(out.join("transport-placement-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"fixed":fixed,"fences":fences,"generations":generations,"original_cpus":original,"performance_cpus":performance,
        "modes":["original","selective-packets","owner-performance-pool","owner-performance-pool-and-selective-packets"],
        "all_fixed_logits_and_state_exact":true,"affinity_restored":true,
        "scope":"Same allocation.131K fixed prefix M3/M6/M16 packet/CPU factorial; original and producer QKV fence. Reversed128-token T1/P1 coding/ideation using ordinary progress-next protocol and cold prompts. No HTTP/upload measurement or completed-task quality claim. Original QKV, affinity and bulk policy restored."}))?)?;Ok(())
}
