//! A parked request either owns an old compact image or names its still-live
//! physical bank. A physical marker never clones history, media, KV or RNG.
use super::*;
use super::super::request_checkpoint::RequestCheckpoint;

pub(super) enum Storage {Image(RequestCheckpoint),Resident(u8)}
pub(super) struct ParkedRequest {
    pub(super) id:u64,
    pub(super) charge:usize,
    pub(super) storage:Storage,
}
impl ParkedRequest {
    pub(super) fn image(image:RequestCheckpoint)->Self {
        Self{id:image.id,charge:image.charge,storage:Storage::Image(image)}
    }
    pub(super) fn resident(id:u64,bank:u8)->Self {
        Self{id,charge:0,storage:Storage::Resident(bank)}
    }
    pub(super) fn resident_bank<'a>(&self,w:&'a Worker)->Result<&'a resident_bank::Bank> {
        let Storage::Resident(bank)=self.storage else {anyhow::bail!("image is not a physical bank marker")};
        let peer=w.peer_bank.as_ref().context("parked physical bank missing")?;
        ensure!(peer.id==bank&&peer.current==Some(self.id)&&bank!=w.bank_id&&self.charge==0,
            "parked physical owner identity mismatch");Ok(peer)
    }
    pub(super) fn prefill_remaining(&self,w:&Worker)->Result<Option<usize>> {
        match &self.storage {Storage::Image(p)=>Ok(p.prefill_remaining()),
            Storage::Resident(_)=>Ok(self.resident_bank(w)?.prefill_remaining())}
    }
    pub(super) fn exchangeable(&self)->bool {
        match &self.storage {Storage::Image(p)=>p.exchangeable(),Storage::Resident(_)=>true}
    }
    pub(super) fn matches_history(&self,history:&ResidentHistory)->bool {
        match &self.storage {Storage::Image(p)=>p.matches_history(history),Storage::Resident(_)=>false}
    }
    pub(super) fn needs_prefix(&self,w:&Worker,prefix:&[u32],media:&crate::prompt_media::Identity)->Result<bool> {
        match &self.storage {Storage::Image(p)=>Ok(p.needs_prefix(prefix,media)),
            Storage::Resident(_)=>Ok(self.resident_bank(w)?.history.needs_prefix(prefix,media))}
    }
    pub(super) fn is_image(&self)->bool {matches!(self.storage,Storage::Image(_))}
}

pub(super) fn park(w:&mut Worker,images:&mut Vec<ParkedRequest>)->Result<Value> {
    ensure!(images.is_empty()&&w.peer_bank.as_ref().is_some_and(|p|p.current.is_none()),
        "two physical owners leave exactly one free bank before parking");
    let id=w.current.context("physical park without active request")?;
    let bank=w.bank_id;let began=Instant::now();
    w.switch_resident_bank()?;
    ensure!(w.current.is_none(),"physical park did not select the free owner");
    images.push(ParkedRequest::resident(id,bank));
    Ok(json!({"type":"park","request_id":id,"parked":true,"resident_bank":bank,
        "charge_bytes":0,"held_bytes":0,"evicted_cache_entries":0,
        "protected_entries_reclaimed_for_admitted_owner":0,"seconds":began.elapsed().as_secs_f64()}))
}

pub(super) fn resume(w:&mut Worker,saved:&ParkedRequest)->Result<Value> {
    ensure!(w.current.is_none(),"physical resume would replace an active owner");
    saved.resident_bank(w)?;
    let began=Instant::now();w.switch_resident_bank()?;
    ensure!(w.current==Some(saved.id),"physical resume request mismatch");
    Ok(json!({"type":"resume","request_id":saved.id,"resident_bank":w.bank_id,
        "frontier":w.history.frontier(),"emitted":w.emitted,"limit":w.limit,
        "restore_seconds":began.elapsed().as_secs_f64()}))
}

pub(super) fn discard(w:&mut Worker,saved:&ParkedRequest)->Result<Value> {
    saved.resident_bank(w)?;
    w.switch_resident_bank()?;
    // Retire through the same model/media/error boundary as a live request.
    // No client event is published here: the scheduler already cancelled it.
    w.execute(Command::Finish(Finish::Cancelled))?;
    w.switch_resident_bank()?;
    Ok(json!({"type":"discard","request_id":saved.id,"resident_bank":true}))
}

pub(super) fn exchange(w:&mut Worker,saved:&mut ParkedRequest)->Result<Value> {
    saved.resident_bank(w)?;
    let outgoing=w.current.context("physical exchange without active request")?;
    let bank=w.bank_id;let incoming=saved.id;let began=Instant::now();
    w.switch_resident_bank()?;
    ensure!(w.current==Some(incoming),"physical exchange request mismatch");
    *saved=ParkedRequest::resident(outgoing,bank);
    Ok(json!({"type":"exchange","exchanged":true,"request_id":incoming,
        "outgoing_request_id":outgoing,"resident_bank":w.bank_id,
        "frontier":w.history.frontier(),"emitted":w.emitted,"limit":w.limit,
        "charge_bytes":0,"preparation_extra_bytes":0,"evicted_cache_entries":0,
        "seconds":began.elapsed().as_secs_f64()}))
}
