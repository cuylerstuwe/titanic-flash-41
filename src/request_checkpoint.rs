//! Complete unfinished-request images shared by the HTTP scheduler and its finite probe.
use super::*;

pub(super) struct RequestCheckpoint {
    image: crate::dspark_generation::StateImage,
    history: ResidentHistory,
    pending_prefill: Option<PendingPrefill>,
    media: Option<std::rc::Rc<crate::prompt_media::Bindings>>,
    pub(super) id: u64,
    emitted: usize,
    limit: usize,
    pub(super) charge: usize,
}

impl RequestCheckpoint {
    pub(super) fn matches_history(&self,history:&ResidentHistory)->bool {
        self.history.committed()==history.committed() && self.history.pending()==history.pending()
            && self.history.media().matches_prefix(history.media(),history.frontier())
    }
}

pub(super) fn try_checkpoint(w: &mut Worker, other_charge: usize,
    other_count:usize) -> Result<Option<(RequestCheckpoint, Value)>> {
    ensure!(w.current.is_some() && other_count<=1 && w.verify.profile_is_current()
        && w.verify.position() == w.history.frontier() as u64,
        "request checkpoint needs a committed active request and bounded peers");
    let (device, host) = w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
    let charge = device.checked_add(host).and_then(|v|v.checked_add(w.history.owned_token_bytes()))
        .and_then(|v|v.checked_add(w.pending_prefill.as_ref().map_or(0,|p|p.owned_token_bytes())))
        .and_then(|v|v.checked_add(w.active_media.as_ref().map_or(0,|m|m.owned_bytes())))
        .ok_or_else(||anyhow::anyhow!("probe checkpoint byte overflow"))?;
    let entries=w.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
    // Live requests take priority over optional cache entries. Existing paused
    // images are never victims; refusal leaves the current request running.
    let evictions=cache_evictions(&entries,(usize::MAX,charge),
        w.shared_image_budget.saturating_sub(other_charge),true,4usize.saturating_sub(other_count));
    let within_budget=evictions.is_some();
    let mut evicted=0;
    if let Some(mut indices)=evictions {
        indices.sort_unstable();evicted=indices.len();
        for i in indices.into_iter().rev(){w.cached_sessions.remove(i);}
    }
    let reserve = 8*1024usize.pow(3)+16*1024usize.pow(2);
    if !w.control.all_admit(within_budget && w.cuda.admission_memory()?.admitted_available >= reserve+charge)? {
        return Ok(None);
    }
    w.stream.synchronize()?;
    let began=Instant::now();
    let Some(image)=w.try_save_session_image(w.history.pending().unwrap(),device)? else {
        return Ok(None);
    };
    w.stream.synchronize()?;
    ensure!(image.device_bytes()==device,"checkpoint allocation estimate");
    let saved=RequestCheckpoint{image,history:w.history.clone(),pending_prefill:w.pending_prefill.clone(),media:w.active_media.clone(),id:w.current.unwrap(),
        emitted:w.emitted,limit:w.limit,charge};
    let receipt=json!({"request_id":saved.id,"frontier":saved.history.frontier(),
        "emitted":saved.emitted,"limit":saved.limit,"charge_bytes":charge,
        "held_bytes":other_charge+charge,"evicted_cache_entries":evicted,
        "cached_bytes":w.cached_sessions.iter().map(|c|c.charge).sum::<usize>(),
        "save_seconds":began.elapsed().as_secs_f64(),
        "manifest":saved.image.manifest()});
    w.agree(&[])?;
    Ok(Some((saved,receipt)))
}

pub(super) fn suspend(w: &mut Worker) -> Result<Value> {
    ensure!(w.current.is_some(),"probe suspend without active request");
    w.stream.synchronize()?;
    let id=w.current.unwrap();
    w.history.suspend()?;
    w.current=None;w.emitted=0;w.limit=0;w.pending_prefill=None;
    w.install_media(None)?;
    // Keep loaded token history so the next Begin sees the correct GPU owner.
    w.agree(&[])?;
    Ok(json!({"request_id":id,"frontier":w.history.frontier(),"client_finish_emitted":false}))
}

pub(super) fn restore(w: &mut Worker, saved: &RequestCheckpoint) -> Result<Value> {
    ensure!(w.current.is_none(),"probe restore would replace an active request");
    let reserve=8*1024usize.pow(3)+16*1024usize.pow(2);
    let mut admitted=w.control.all_admit(w.cuda.admission_memory()?.admitted_available>=reserve)?;
    if !admitted {
        w.cached_sessions.clear();
        admitted=w.control.all_admit(w.cuda.admission_memory()?.admitted_available>=reserve)?;
    }
    ensure!(admitted,
        "all-rank restore memory admission refused");
    w.target_lineage=w.target_lineage.checked_add(1)
        .ok_or_else(||anyhow::anyhow!("probe lineage overflow"))?;
    let began=Instant::now();
    let anchor=w.session.as_mut().unwrap().restore_state_image(&mut w.verify,&saved.image,
        w.target_lineage,&w.comm,&w.stream)?;
    w.history.resume(&saved.history)?;
    w.install_media(saved.media.clone())?;
    w.current=Some(saved.id);w.emitted=saved.emitted;w.limit=saved.limit;w.pending_prefill=saved.pending_prefill.clone();
    w.stream.synchronize()?;
    ensure!(Some(anchor)==w.history.pending() && w.verify.position()==w.history.frontier() as u64
        && w.verify.logits().is_err() && !w.large.profile_is_current(),
        "restored request frontier or stale scratch exposure");
    w.agree(&[])?;
    Ok(json!({"request_id":saved.id,"frontier":w.history.frontier(),"emitted":w.emitted,
        "limit":w.limit,"restore_seconds":began.elapsed().as_secs_f64()}))
}

