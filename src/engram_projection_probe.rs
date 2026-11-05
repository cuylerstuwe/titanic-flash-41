//! Same-allocation screen for overlapping the independent Engram14 projection.
use super::*;

fn state(t:&Target)->Result<Value>{Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))}
fn logits(t:&Target,cuda:&Cuda)->Result<String>{
    let(y,live)=t.pending_local_logits_for_probe()?;
    Ok(format!("{:x}",Sha256::digest(unsafe{cuda.read_device(y.ptr(),live*32320*4)}?)))
}
pub(super) fn run(w:&mut Worker,rank:usize,kernels:&Path,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.history.frontier()==0
        &&std::env::var("CANADA_SCRATCH_ENGRAM_PROJECTION_SCREEN").as_deref()==Ok("1"),
        "isolated selective Engram projection probe");
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;let prefix=131073;
    history32_generation_probe::execute(w,Command::Begin{images:Vec::new(),id:8_050_100,
        seed:2026110312,greedy:false,limit:1,tokens:tokens[..prefix].to_vec()})?;
    let mut records=Vec::new();let mut reference=std::collections::BTreeMap::new();
    for(trial,mode)in [0u8,2,2,0].into_iter().enumerate(){
        let t=&mut w.verify;
        t.set_engram_projection_ahead_for_probe(mode,kernels,&w.stream)?;
        let before=state(t)?;
        // Capture both live widths under the same policy. Later switches must
        // retain both graphs, and only M3 may select the auxiliary projection.
        for _ in 0..3{for live in [3usize,6]{
            t.begin_verification(&tokens[prefix..prefix+live],&w.comm,&w.stream)?;
            t.finish_verification(0,&w.comm,&w.stream)?;
        }}
        let captures=t.capture_statistics().1;
        let mut samples=std::collections::BTreeMap::<usize,Vec<f64>>::new();
        w.control.agree(&[0xd3,mode,trial as u8])?;
        for iteration in 0..12{for live in if iteration%2==0{[3usize,6]}else{[6,3]}{
            let at=Instant::now();t.begin_verification(&tokens[prefix..prefix+live],&w.comm,&w.stream)?;
            samples.entry(live).or_default().push(at.elapsed().as_secs_f64());
            t.finish_verification(0,&w.comm,&w.stream)?;
        }}
        ensure!(t.capture_statistics().1==captures,"mixed M3/M6 projection recaptured");
        for live in [3usize,6]{
            let mut checks=Vec::new();
            for offset in [prefix,prefix+16]{t.begin_verification(&tokens[offset..offset+live],&w.comm,&w.stream)?;
                checks.push(logits(t,&w.cuda)?);t.finish_verification(0,&w.comm,&w.stream)?;}
            let logits_exact=reference.get(&live).is_none_or(|expected|expected==&checks);
            reference.entry(live).or_insert_with(||checks.clone());
            let state_exact=state(t)?==before;
            // One asymmetric application-cache miss per width and arm also
            // checks that rank-local I/O cannot change collective ordering.
            if rank==3{t.clear_engram_cache_for_probe(&w.stream)?;}
            w.control.agree(&[0xd4,mode,trial as u8,live as u8])?;
            let at=Instant::now();t.begin_verification(&tokens[prefix..prefix+live],&w.comm,&w.stream)?;
            let cold_seconds=at.elapsed().as_secs_f64();let cold_exact=logits(t,&w.cuda)?==checks[0];
            t.finish_verification(0,&w.comm,&w.stream)?;let cold_state_exact=state(t)?==before;
            ensure!(checks[0]!=checks[1]&&logits_exact&&state_exact&&cold_exact&&cold_state_exact,
                "selective Engram projection changed outputs or rollback state");
            records.push(json!({"physical_rows":6,"live":live,"trial":trial,"mode":mode,"ahead":mode==2&&live==3,
                "seconds":samples[&live],"checks":checks,"logits_exact":logits_exact,"state_exact":state_exact,
                "asymmetric_cold_seconds":cold_seconds,"cold_logits_exact":cold_exact,"cold_state_exact":cold_state_exact,
                "capture_statistics":t.capture_statistics(),"mixed_widths_no_recapture":true}));
        }
        std::fs::write(out.join("engram-projection-partial.json"),serde_json::to_vec_pretty(&records)?)?;
    }
    history32_generation_probe::execute(w,Command::Finish(Finish::Length))?;
    history32_generation_probe::execute(w,Command::Shutdown)?;
    std::fs::write(out.join("engram-projection-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"all_exact":true,"projection_ahead_restored_off":true,
        "scope":"Same allocation,131K public prefix;off/M3-only/M3-only/off. Mixed M3/M6 with distinct mutable projection buffers sharing immutable weights;both graph keys retained. Changed-input local vocabulary logits,carry/encoder rollback and asymmetric rank3 row-cache misses. Canonical operators and existing row-fetch schedule. No generation or production claim."}))?)?;
    Ok(())
}
