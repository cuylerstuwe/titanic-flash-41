//! Same-allocation target completion polling. No model arithmetic changes.
use super::*;

fn cpu_seconds()->Result<f64> {
    let mut ts=libc::timespec{tv_sec:0,tv_nsec:0};
    ensure!(unsafe{libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID,&mut ts)}==0,
        "read verification host-thread CPU clock");
    Ok(ts.tv_sec as f64+ts.tv_nsec as f64*1e-9)
}
fn state(target:&Target)->Result<Value> {
    Ok(json!({"position":target.position(),"carry":target.carry_digests()?,
        "encoder":target.encoder_tail_digest()?}))
}
fn screen(target:&mut Target,cuda:&Cuda,comm:&Roce,stream:&Stream,
    tokens:&[u32],rank:usize)->Result<Vec<Value>> {
    let original=state(target)?;let mut expected=None;let mut records=Vec::new();
    for (trial,poll_us) in [0,25,0,25,25,0].into_iter().enumerate() {
        target.set_verification_poll_for_probe(poll_us,stream)?;
        let repetitions=if trial<2{2}else{8};
        let captures=target.capture_statistics().1;
        let start=Instant::now();let cpu=cpu_seconds()?;let mut samples=Vec::new();
        for _ in 0..repetitions {
            let at=Instant::now();
            target.begin_verification(tokens,comm,stream)?;
            target.finish_verification(0,comm,stream)?;
            samples.push(at.elapsed().as_secs_f64()*1000.);
        }
        let cpu_ms=(cpu_seconds()?-cpu)*1000.;let wall_ms=start.elapsed().as_secs_f64()*1000.;
        let new_captures=target.capture_statistics().1-captures;
        ensure!(trial<2||new_captures==0,"timed verification recaptured a graph");
        // Read the actual target logits after the tested completion boundary,
        // outside timing. No host model or probability implementation.
        target.begin_verification(tokens,comm,stream)?;
        let (logits,live)=target.pending_logits()?;ensure!(live==tokens.len(),"wait-probe live extent");
        let raw=unsafe{cuda.read_device(logits.ptr(),live*129280*4)}?;
        let hash=format!("{:x}",Sha256::digest(raw));
        let ids=target.verification_sample(2026111500,stream)?;
        target.finish_verification(0,comm,stream)?;
        ensure!(state(target)?==original,"completion polling changed retained state/rollback");
        if let Some((old_hash,old_ids))=&expected {
            ensure!(old_hash==&hash&&old_ids==&ids,"completion polling changed target logits/tokens");
        }else{expected=Some((hash.clone(),ids.clone()));}
        records.push(json!({"physical_rows":target.physical_rows(),"live":tokens.len(),"trial":trial,
            "warmup":trial<2,"poll_us":poll_us,"repetitions":repetitions,"wall_ms":wall_ms,
            "thread_cpu_ms":cpu_ms,"samples_ms":samples,"logits_sha256":hash,"sampled_ids":ids,
            "new_graph_captures":new_captures,"state_and_rollback_exact":true}));
        eprintln!("rank{rank} verify-wait rows{} live{} poll{poll_us} trial{trial}: {:.4}ms cpu{:.4}ms",
            target.physical_rows(),tokens.len(),wall_ms/repetitions as f64,cpu_ms/repetitions as f64);
    }
    target.set_verification_poll_for_probe(0,stream)?;Ok(records)
}
pub(super) fn run(w:&mut Worker,rank:usize,kernels:&Path,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.history.frontier()==0&&w.large.position()==0
        &&w.long_verify.is_some()&&!Target::compact_history_sampling_requested()?,
        "isolated wait screen needs fresh full-logit profiles");
    let module=w.cuda.module(&w.core)?;
    let mut wider=w.large.fork_profile(&module,&w.core,&w.stream,kernels,32,Phase::Verify)?;
    wider.set_graph_replay(true)?;
    ensure!(w.control.all_admit(w.cuda.admission_memory()?.admitted_available>=9*1024usize.pow(3))?,
        "verification wait screen reserve");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let prefix=65793;ensure!(tokens.len()>=prefix+32,"wait screen public fixture extent");
    w.large.prefill_wide(&mut w.wides,&tokens[..prefix],&w.comm,&w.stream)?;
    w.verify.handoff_from(&w.large)?;let mut records=Vec::new();
    for live in [3,6] {
        records.extend(screen(&mut w.verify,&w.cuda,&w.comm,&w.stream,&tokens[prefix..prefix+live],rank)?);
    }
    for target in [w.long_verify.as_mut().unwrap(),&mut wider] {
        target.handoff_from(&w.verify)?;let rows=target.physical_rows();
        records.extend(screen(target,&w.cuda,&w.comm,&w.stream,&tokens[prefix..prefix+rows],rank)?);
        w.verify.handoff_from(target)?;
    }
    w.control.agree(&[0x76,0x64])?;
    std::fs::write(out.join("verify-wait-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"prefix_tokens":prefix,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "records":records,"all_logits_tokens_state_exact":true,
        "scope":"One meaningful65Kpublic prefix, same weights and allocations. Live3/6and16/32copy profiles. Two warmups thenABBA,8verification/rollback repetitions per arm; only host completion wait differs. Full logits/targetT1samples and retained state exact. No clock,timer-slack,priority,arithmetic or serving policy change. Resource/latency screen, not task quality or thermal steady state."
    }))?)?;Ok(())
}
