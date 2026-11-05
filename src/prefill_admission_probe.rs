//! Bounded four-rank pressure, refusal, retry and late-preview regression.
use super::*;
use super::history32_generation_probe::{execute,state};
use cache_admission_fault::Point;

fn begin(w:&mut Worker,id:&mut u64,tokens:Vec<u32>)->Result<Value> {
    *id+=1;
    execute(w,Command::Begin{images:Vec::new(),id:*id,seed:20261105,
        greedy:false,limit:1,tokens})
}
fn finish(w:&mut Worker)->Result<()> {execute(w,Command::Finish(Finish::Length))?;Ok(())}
fn following(w:&mut Worker,tokens:&[u32])->Result<Value> {
    let before=state(w)?;
    w.verify.begin_verification(tokens,&w.comm,&w.stream)?;
    let(y,rows)=w.verify.pending_logits()?;ensure!(rows==6,"following verification width");
    let bytes=unsafe{w.cuda.read_device(y.ptr(),rows*129280*4)}?;
    w.verify.finish_verification(0,&w.comm,&w.stream)?;
    ensure!(state(w)?==before,"pressure probe verification changed committed state");
    Ok(json!({"state":before,"following_logits":format!("{:x}",Sha256::digest(&bytes))}))
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.peer_bank.is_none(),"finite single-owner pressure probe");
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let mut id=9_051_300;let mut records=Vec::new();
    // Keep checkpointing enabled to exercise both halves of compound prefill.
    w.prompt_cache_logits=false;w.encoder_active_frontier=false;
    for refusing_rank in [0,3] {
        w.cached_sessions.clear();
        let initial=begin(w,&mut id,tokens[..8892].to_vec())?;
        ensure!(initial["type"]=="tokens","initial pressure prompt");finish(w)?;
        let original=state(w)?;
        w.cache_save_fault.arm(Point::PrefillAfterReclaim,refusing_rank)?;
        let refused=begin(w,&mut id,tokens[..9150].to_vec())?;
        ensure!(refused["type"]=="input_refused"&&w.current.is_none(),"pressure must refuse one request, not exit");
        let fault=w.cache_save_fault.take_observation()?;
        ensure!(state(w)?==original,"refused input changed committed target/drafter/history");
        let retry=begin(w,&mut id,tokens[..9150].to_vec())?;
        ensure!(retry["type"]=="tokens","healthy retry after pressure refusal");finish(w)?;
        // Exercise recoverable pressure at a retained append boundary.
        w.cache_save_fault.arm(Point::PrefillBeforeReclaim,refusing_rank)?;
        let reclaimed=begin(w,&mut id,tokens[..9408].to_vec())?;
        let recovery_fault=w.cache_save_fault.take_observation()?;
        ensure!(reclaimed["type"]=="tokens"&&reclaimed["retained_session_cache"]["prefill_reclaim"].is_object(),
            "bounded reclaim path did not continue the request");finish(w)?;
        records.push(json!({"refusing_rank":refusing_rank,"refused":refused,"fault":fault,
            "retry":retry,"reclaimed":reclaimed,"recovery_fault":recovery_fault}));
    }
    // Demand reads and optional lookahead must yield identical model results.
    let mut control=None;
    for pressure in [false,true,true,false] {
        begin(w,&mut id,tokens[12000..12300].to_vec())?;finish(w)?;
        w.cached_sessions.clear();
        crate::memory::PREFILL_PREVIEW_PRESSURE.set(pressure&&rank==3);
        let event=begin(w,&mut id,tokens[..9409].to_vec())?;
        crate::memory::PREFILL_PREVIEW_PRESSURE.set(false);
        ensure!(event["type"]=="tokens"&&event["cached_tokens"]==0,"fresh late-pressure prefill");
        let proof=following(w,&tokens[10000..10006])?;
        ensure!(w.control.all_admit(control.as_ref().is_none_or(|c|c==&proof))?,
            "rank-local optional prefetch fallback changed model state or following logits");
        control=Some(proof.clone());finish(w)?;
        records.push(json!({"late_preview_pressure":pressure,"proof":proof}));
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("prefill-admission-summary.json"),serde_json::to_vec_pretty(
        &json!({"rank":rank,"passed":true,"injected_pressure":true,"records":records}))?)?;Ok(())
}
