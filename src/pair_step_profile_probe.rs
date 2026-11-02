//! Two warm target spans on one fixed prefix. Profiling, not sampled throughput.
use super::*;
use super::history32_generation_probe::execute;

/// Same-load cost bound only. A twelve-token *single-owner* forward measures
/// the existing M16 kernels; it is not a fused causal two-owner implementation.
/// Called after a sampled pair, while both real owners and all profiles remain
/// resident. Capture and state reads occur outside every timed forward.
pub(super) fn measure_pair_cost(w:&mut Worker,case:&str,teacher:&[u32],out:&Path)->Result<Value> {
    ensure!(std::env::var("CANADA_SCRATCH_PAIR_COST_SCREEN").as_deref()==Ok("1")
        &&w.long_verify.is_some()&&teacher.len()==6,"paired cost diagnostic admission");
    let ids:Vec<u32>=teacher.iter().cycle().take(12).copied().collect();
    let original=(w.verify.carry_digests()?,w.verify.encoder_tail_digest()?);
    let position=w.verify.position();
    let mut records=Vec::new();
    for (rows,live)in [(6usize,6usize),(16,12)] {
        if rows==16 {w.long_verify.as_mut().unwrap().handoff_from(&w.verify)?;}
        let target=if rows==6{&mut w.verify}else{w.long_verify.as_mut().unwrap()};
        let mut expected=None;let mut times=Vec::new();let mut profile=Value::Null;
        let mut capture=Value::Null;
        // One capture/warmup, four uninstrumented timings, one traced replay.
        for trial in 0..6u8 {
            w.stream.synchronize()?;w.control.agree(&[0xa7,rows as u8,live as u8,trial])?;
            let tracing=trial==5;
            if tracing {
                ensure!(target.verification_graph_is_warm(live),"cost trace requires warmed shape");
                w.cuda.profiler_start()?;w.control.agree(&[0xa8,rows as u8,live as u8])?;
            }
            let at=Instant::now();target.begin_verification(&ids[..live],&w.comm,&w.stream)?;
            w.stream.synchronize()?;let seconds=at.elapsed().as_secs_f64();
            if tracing{w.cuda.profiler_stop()?;}
            let(y,n)=target.pending_local_logits_for_probe()?;
            ensure!(n==live,"cost diagnostic local logit extent");
            let hash=format!("{:x}",Sha256::digest(unsafe{w.cuda.read_device(y.ptr(),live*32320*4)}?));
            if let Some(first)=&expected {ensure!(first==&hash,"cost replay changed fixed-prefix logits");}
            else{expected=Some(hash.clone());}
            if trial==0 {
                let path=out.join(format!("pair-cost-routes/{case}/M{rows}-live{live}"));
                capture=target.capture_expert_components(&path,&w.stream)?;
            }else if tracing {
                profile=json!({"seconds":seconds,"metrics":target.verification_metrics()?,"local_logits":hash});
            }else{times.push(seconds);}
            target.finish_verification(0,&w.comm,&w.stream)?;
            ensure!((target.carry_digests()?,target.encoder_tail_digest()?)==original,
                "cost probe changed causal rollback");
        }
        records.push(json!({"rows":rows,"live_rows":live,"warm_seconds":times,
            "profile":profile,"routes":capture,"logits_repeat_exact":true,"rollback_exact":true}));
        if rows==16 {w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;}
    }
    let report=json!({"case":case,"rank":w.comm.rank(),"position":position,"records":records,
        "scope":"Same resident load and accepted public prefix, M6/live6 and M16/live12; four warm timings and one traced replay each. M16 rows are one causal sequence, not two owners. Real paired routes captured separately. No fused throughput or quality claim."});
    std::fs::write(out.join(format!("pair-cost-{case}.json")),serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_PAIR_STEP_PROFILE").as_deref()==Ok("1")
        &&w.current.is_none()&&w.long_verify.is_some(),"finite pair-step profile admission");
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let ids:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let prefix=131072;
    let begin=execute(w,Command::Begin{images:Vec::new(),id:7_950_000,seed:20261031,
        greedy:false,limit:1,tokens:ids[..prefix].to_vec()})?;
    let original=(w.verify.carry_digests()?,w.verify.encoder_tail_digest()?);
    let mut records=Vec::new();
    for rows in [6usize,16] {
        if rows==16 {w.long_verify.as_mut().unwrap().handoff_from(&w.verify)?;}
        let target=if rows==6 {&mut w.verify}else{w.long_verify.as_mut().unwrap()};
        let mut reference=None;let mut times=Vec::new();let mut profiled=Value::Null;
        for trial in 0..6u8 {
            let tracing=trial==3;
            w.stream.synchronize()?;w.control.agree(&[0x95,rows as u8,trial])?;
            if tracing {
                ensure!(target.verification_graph_is_warm(rows),"profile requires current warm graph");
                w.cuda.profiler_start()?;w.control.agree(&[0x96,rows as u8])?;
            }
            let at=Instant::now();
            target.begin_verification(&ids[prefix..prefix+rows],&w.comm,&w.stream)?;
            w.stream.synchronize()?;let seconds=at.elapsed().as_secs_f64();
            if tracing {w.cuda.profiler_stop()?;}
            let(y,n)=target.pending_local_logits_for_probe()?;ensure!(n==rows,"profile logit extent");
            let hash=format!("{:x}",Sha256::digest(unsafe{w.cuda.read_device(y.ptr(),rows*32320*4)}?));
            if let Some(expected)=&reference {ensure!(expected==&hash,"warm profile changed fixed-prefix logits");}
            else{reference=Some(hash.clone());}
            if tracing {profiled=json!({"seconds":seconds,"metrics":target.verification_metrics()?,"local_logits":hash});}
            else if trial>0 {times.push(seconds);}
            target.finish_verification(0,&w.comm,&w.stream)?;
            ensure!((target.carry_digests()?,target.encoder_tail_digest()?)==original,
                "profile rollback changed causal state");
        }
        records.push(json!({"rows":rows,"warm_unprofiled_seconds":times,"profile":profiled,
            "local_logits_exact":true,"rollback_exact":true}));
        if rows==16 {w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;}
    }
    execute(w,Command::Finish(Finish::Length))?;execute(w,Command::Shutdown)?;
    std::fs::write(out.join("pair-step-profile.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"prompt_tokens":prefix,"begin":begin,"records":records,
        "memory":w.cuda.admission_memory()?,"history32":w.copy32_admission,
        "scope":"One public131Kprefix, warm M6 and M16 using actual compact head. One instrumented target step each; four unprofiled warm samples per shape. All local vocabulary columns and causal rollback checked. Kernel sums overlap. No batched execution, acceptance, task-quality or throughput claim."}))?)?;Ok(())
}
