//! Same-load encoder completion-wait screen. No arithmetic or sampler changes.
use super::*;

fn thread_cpu_seconds()->Result<f64> {
    let mut ts=libc::timespec {tv_sec:0,tv_nsec:0};
    ensure!(unsafe {libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID,&mut ts)}==0,
        "read model host-thread CPU clock");
    Ok(ts.tv_sec as f64+ts.tv_nsec as f64*1e-9)
}
fn digest(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}
fn state(t:&Target)->Result<Value> {
    Ok(json!({"position":t.position(),"carry":t.carry_digests()?,
        "encoder":t.encoder_tail_digest()?}))
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none() && w.history.frontier()==0,
        "completion-wait screen needs an isolated target");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    // Full wide chunks plus a mixed partial tail; all trials use identical IDs.
    let prefix=65793;
    ensure!(tokens.len()>=prefix+6,"completion-wait fixture extent");
    let flags=w.cuda.context_flags()?;
    let mut records=Vec::new();
    let mut expected:Option<(String,Value)>=None;
    for (trial,blocking) in [false,true,false,true,true,false].into_iter().enumerate() {
        w.control.agree(&[trial as u8,u8::from(blocking)])?;
        w.target_lineage=w.target_lineage.checked_add(1).context("wait-probe lineage exhausted")?;
        if w.verify.profile_is_current() {
            w.verify.reset(w.target_lineage,&w.comm,&w.stream)?;
            w.large.handoff_from(&w.verify)?;
        } else {
            ensure!(w.large.profile_is_current(),"completion-wait target owner absent");
            w.large.reset(w.target_lineage,&w.comm,&w.stream)?;
        }
        w.large.set_prefill_blocking_for_probe(blocking,&w.stream)?;
        for target in &mut w.wides {
            target.set_prefill_blocking_for_probe(blocking,&w.stream)?;
        }
        let began=Instant::now();let cpu=thread_cpu_seconds()?;
        w.large.prefill_wide(&mut w.wides,&tokens[..prefix],&w.comm,&w.stream)?;
        let cpu_seconds=thread_cpu_seconds()?-cpu;
        let seconds=began.elapsed().as_secs_f64();
        let prompt=w.large.snapshot_prompt_logits(&w.stream)?;
        let mut logits=unsafe {w.cuda.read_device(prompt.ptr(),129280*4)}?;
        w.verify.handoff_from(&w.large)?;
        let before=state(&w.verify)?;
        w.verify.begin_verification(&tokens[prefix..prefix+6],&w.comm,&w.stream)?;
        let(y,rows)=w.verify.pending_logits()?;ensure!(rows==6,"completion-wait following extent");
        logits.extend(unsafe {w.cuda.read_device(y.ptr(),6*129280*4)}?);
        w.verify.finish_verification(0,&w.comm,&w.stream)?;
        ensure!(state(&w.verify)?==before,"completion-wait rollback differs");
        let hash=digest(&logits);
        if let Some((old_hash,old_state))=&expected {
            ensure!(hash==*old_hash && before==*old_state,
                "completion-wait changed prompt/following logits or retained state");
        } else {expected=Some((hash.clone(),before.clone()));}
        records.push(json!({"trial":trial,"blocking":blocking,"warmup":trial<2,
            "prefill_tokens":prefix,"prefill_seconds":seconds,"thread_cpu_seconds":cpu_seconds,
            "logits_sha256":hash,"state":before,"rollback_exact":true}));
        std::fs::write(out.join("prefill-wait-progress.json"),serde_json::to_vec_pretty(&records)?)?;
        eprintln!("rank{rank} prefill-wait trial{trial} blocking={blocking} seconds={seconds:.6} cpu={cpu_seconds:.6}");
    }
    w.control.agree(&[0x73,0x28])?;
    std::fs::write(out.join("prefill-wait-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"context_flags":flags,"fixture_sha256":digest(raw),"records":records,
        "all_logits_and_state_exact":true,
        "scope":"Same model allocations, identical 65793-token public prefix, two warmups plus ABBA. Only host completion waits after encoder chunks of at least 1024 rows differ. Prompt and next-six logits, retained state and rollback checked; no outcome-quality claim."
    }))?)?;
    Ok(())
}
