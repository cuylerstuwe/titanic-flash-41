//! One real-UMA cache-admission screen. No HTTP path or artificial memory vote.
use super::*;

fn execute(w:&mut Worker,c:Command)->Result<Value> {
    let bytes=c.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(bytes.as_slice()))?;
    ensure!(wire==bytes,"cache pressure command replicas disagree");
    w.execute(Command::decode(&wire)?)
}
fn observe(w:&mut Worker,ids:&[u32])->Result<Value> {
    let state=|w:&mut Worker|->Result<Value>{Ok(json!({
        "frontier":w.verify.position(),"carry":w.verify.carry_digests()?,
        "encoder":w.verify.encoder_tail_digest()?,
        "draft":w.session.as_mut().unwrap().conditioning_state_digest_for_probe(&w.comm,&w.stream)?,
        "history":w.session.as_ref().unwrap().history_lookup_digest_for_probe()?}))};
    let before=state(w)?;
    w.verify.begin_verification(ids,&w.comm,&w.stream)?;
    let(y,n)=w.verify.pending_logits()?;ensure!(n==6,"pressure screen following shape");
    let bytes=unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?;
    w.verify.finish_verification(0,&w.comm,&w.stream)?;
    ensure!(state(w)?==before,"pressure screen read-only verification changed state");
    Ok(json!({"state":before,"following_sha256":format!("{:x}",Sha256::digest(bytes))}))
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.suspended_image_count==0&&w.encoder_prompt_checkpoint,
        "cache pressure finite admission");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    ensure!(tokens.len()>=200006,"cache pressure fixture extent");
    w.cached_sessions.clear();w.encoder_active_frontier=false;
    w.encoder_upgrade_reconstructed=false;w.prompt_cache_logits=false;
    w.cache_pressure_reclaim=false;
    let begin=execute(w,Command::Begin{images:Vec::new(),id:7_810_001,
        seed:20261030,greedy:false,limit:1,tokens:tokens[..65792].to_vec()})?;
    ensure!(begin["target_temperature"]==1.0&&begin["target_top_p"]==1.0
        &&begin["prompt_checkpoint"]["saved"]==true&&w.cached_sessions.len()==1,
        "pressure screen needs one real encoder parent at T1/P1");
    execute(w,Command::Finish(Finish::Length))?;
    let prefix=w.history.committed().to_vec();
    let media=w.history.media().through(prefix.len());
    let pending=w.history.pending().ok_or_else(||anyhow::anyhow!("pressure screen anchor absent"))?;
    let reference=observe(w,&tokens[200000..200006])?;
    let (device,host)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
    let charge=device+host+prefix.len()*4+media.metadata_bytes();
    let reclaimed=w.cached_sessions.iter().map(|c|c.charge).sum::<usize>();
    let reserve=8*1024usize.pow(3)+16*1024usize.pow(2);
    let required=reserve+charge;
    let (_,slots)=w.cache_capacity(0,0);
    w.cache_pressure_reclaim=true;
    let impossible=w.admit_cache_save(true,Some(Vec::new()),&prefix,&media,256*1024usize.pow(3),slots,&[])?;
    ensure!(!impossible.admitted&&impossible.forecast_refused&&impossible.evicted==0
        &&w.cached_sessions.len()==1,"infeasible forecast must preserve the old cache");
    // Only rank zero is constrained. Keep the full existing reserve, and
    // choose an interval that is too small for the successor but large enough
    // after releasing its real predecessor. This is allocation, not a mocked
    // admission value; all ranks still make the normal collective decision.
    let margin=(reclaimed/2).min(charge/2);
    ensure!(margin>=32*1024*1024,"pressure interval too small for a useful physical screen");
    let before=w.cuda.admission_memory()?;
    let ballast=if rank==0 {
        let target=required-margin;
        ensure!(target>reserve&&before.admitted_available>target+64*1024*1024,
            "insufficient natural headroom for controlled pressure");
        let bytes=(before.admitted_available-target)/65536*65536;
        ensure!(bytes<=8*1024usize.pow(3),"pressure ballast exceeds finite bound");
        let mut buffer=w.cuda.allocate::<u8>(bytes)?;buffer.zero()?;w.stream.synchronize()?;
        Some(buffer)
    }else{None};
    w.control.agree(&[1])?;
    let pressure=w.cuda.admission_memory()?;
    ensure!(w.control.all_admit(pressure.admitted_available>=reserve)?,"pressure screen violated reserve");
    ensure!(w.control.all_admit(rank!=0||(pressure.admitted_available<required
        &&pressure.admitted_available.saturating_add(reclaimed)>=required))?,
        "natural memory movement obscured the intended pressure interval");
    w.cache_pressure_reclaim=false;
    let control=w.retain_cached_prefix(&prefix,pending,&media)?;
    ensure!(control["saved"]==false&&control["memory_refused"]==true
        &&control["evicted_sessions"]==0&&w.cached_sessions.len()==1,
        "pressure control did not retain/refuse as expected");
    ensure!(observe(w,&tokens[200000..200006])?==reference,"refusal changed live model state");
    w.cache_pressure_reclaim=true;
    let candidate=w.retain_cached_prefix(&prefix,pending,&media)?;
    ensure!(candidate["saved"]==true&&candidate["pressure_evicted_sessions"]==1
        &&w.cached_sessions.len()==1&&w.cached_sessions[0].prefix==prefix,
        "pressure candidate did not replace the obsolete parent");
    ensure!(observe(w,&tokens[200000..200006])?==reference,"pressure eviction/save changed live state");
    let after=w.cuda.admission_memory()?;
    ensure!(w.control.all_admit(after.admitted_available>=reserve)?,"saved image exhausted reserve");
    drop(ballast);w.control.agree(&[2])?;
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("cache-pressure-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"begin":begin,"charge":charge,"reclaimed":reclaimed,"margin":margin,
        "before":before,"under_pressure":pressure,"after_save":after,
        "control":control,"candidate":candidate,"live_state_and_logits_exact":true,
        "infeasible_forecast_preserved_cache":true,"reference":reference,
        "fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "scope":"One65Kforeground state; real rank-zero CUDA ballast under unchanged8GiB+16MiB reserve. Logical budget already fits. Compare ordinary refusal with optional physical-pressure eviction. All four ranks agree; live state and6following FP32 rows remain exact. No throughput or production claim."}))?)?;
    Ok(())
}
