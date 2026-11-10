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
    pub(super) fn prefill_remaining(&self)->Option<usize> {
        self.pending_prefill.as_ref().map(PendingPrefill::remaining_tokens)
    }
    pub(super) fn exchangeable(&self)->bool {self.image.exchangeable()}
    pub(super) fn probe_digest(&self)->Result<Value>{
        ensure!(std::env::var("CANADA_SCRATCH_WORKER_STATE_EXCHANGE_SCREEN").as_deref()==Ok("1"),"exchange diagnostic admission");
        Ok(json!({"image":self.image.digests()?,"id":self.id,"emitted":self.emitted,"limit":self.limit,
            "pending":self.history.pending(),"history_sha256":format!("{:x}",Sha256::digest(
                self.history.committed().iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()))}))
    }
    pub(super) fn needs_prefix(&self, prefix: &[u32], media: &crate::prompt_media::Identity) -> bool {
        self.history.needs_prefix(prefix, media)
    }
    pub(super) fn matches_history(&self,history:&ResidentHistory)->bool {
        self.history.committed()==history.committed() && self.history.pending()==history.pending()
            && self.history.media().matches_prefix(history.media(),history.frontier())
    }
}

pub(super) fn exchange_enabled()->Result<bool>{
    match std::env::var("CANADA_SCRATCH_STATE_EXCHANGE"){
        Ok(v)if v=="1"=>Ok(true),Ok(v)if v=="0"=>Ok(false),Err(std::env::VarError::NotPresent)=>Ok(false),
        _=>anyhow::bail!("state exchange flag must be0or1"),
    }
}

pub(super) fn try_checkpoint(w: &mut Worker, other_charge: usize,
    other_count:usize) -> Result<Option<(RequestCheckpoint, Value)>> {
    checkpoint(w, other_charge, other_count, None)
}

pub(super) fn try_checkpoint_protected(w: &mut Worker, other_charge: usize,
    other_count: usize, protected: u16) -> Result<Option<(RequestCheckpoint, Value)>> {
    checkpoint(w, other_charge, other_count, Some(protected))
}

fn checkpoint(w: &mut Worker, other_charge: usize, other_count: usize,
    protected: Option<u16>) -> Result<Option<(RequestCheckpoint, Value)>> {
    w.state_pressure_trim=None;
    ensure!(w.current.is_some() && other_count<=1 && w.verify.profile_is_current()
        && w.verify.position() == w.history.frontier() as u64,
        "request checkpoint needs a committed active request and bounded peers");
    let exchange=exchange_enabled()?;
    let (device, host) = if exchange{w.session.as_ref().unwrap().exchange_state_image_bytes(&w.verify)?}
        else{w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?};
    let charge = device.checked_add(host).and_then(|v|v.checked_add(w.history.owned_token_bytes()))
        .and_then(|v|v.checked_add(w.pending_prefill.as_ref().map_or(0,|p|p.owned_token_bytes())))
        .and_then(|v|v.checked_add(w.active_media.as_ref().map_or(0,|m|m.owned_bytes())))
        .ok_or_else(||anyhow::anyhow!("probe checkpoint byte overflow"))?;
    let entries=w.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
    let budget = w.shared_image_budget.saturating_sub(other_charge);
    let slots = if w.parking_extra_slots {w.session_cache_slots+1}
        else {w.session_cache_slots.saturating_sub(other_count)};
    let mut evictions = if let Some(mask) = protected {
        ensure!(cache_protection_mask_valid(mask,entries.len()), "park protection mask extent");
        let protected = (0..entries.len()).map(|i| mask & (1 << i) != 0).collect::<Vec<_>>();
        completed_cache_evictions(&entries, &vec![false; entries.len()], &protected,
            (usize::MAX, charge), budget, true, slots)
    } else {
        // Preserve the legacy probe/serving policy outside two-slot admission.
        cache_evictions(&entries, (usize::MAX, charge), budget, true, slots)
    };
    // The first image makes room for an unstarted peer. Its prefill can
    // recommit host preparation pages that malloc_trim just returned. Keep
    // working room in addition to the unchanged physical reserve; completed
    // cache may lend it, but an unfinished owner is never an eviction target.
    // Exchanges between already prepared owners account their staging below.
    let working_room=if exchange && other_count==0 {256}else{16}*1024usize.pow(2);
    let reserve = crate::memory::floor_bytes()+working_room;
    let mut host_allocator_trim = None;
    if exchange || crate::config::single_bank_requested() {
        // Optional completed checkpoints may borrow unused parking capacity.
        // Retaining unfinished owners takes precedence over rewind/ancestor
        // acceleration, whether parking uses copies or in-place exchange.
        // No live or suspended request state is an eviction target.
        if evictions.is_none(){evictions=cache_evictions(&entries,(usize::MAX,charge),budget,true,slots);}
        if !w.control.all_admit(evictions.is_some())?{return Ok(None);}
        let indices=evictions.as_mut().unwrap();
        let mut policy=vec![entries.len()as u8];policy.extend(indices.iter().map(|&i|i as u8));w.control.agree(&policy)?;
        let mut available=w.cuda.admission_memory()?.admitted_available;
        loop {
            let reclaimed=indices.iter().map(|&i|entries[i].1).sum::<usize>();
            if w.control.all_admit(available.saturating_add(reclaimed)>=reserve+charge)?{break;}
            if w.trim_free_host_pages && host_allocator_trim.is_none() {
                available=trim_for_state(w,reserve+charge)?;
                host_allocator_trim=w.state_pressure_trim.clone();
                continue;
            }
            let candidate=(0..entries.len()).filter(|i|!indices.contains(i))
                .min_by_key(|&i|usize::from(protected.unwrap_or(0)&(1<<i)!=0));
            let Some(i)=candidate else{return Ok(None);};indices.push(i);
        }
    } else if protected.is_some() {
        let reclaimable = evictions.as_ref().map_or(0, |indices| indices.iter().map(|&i| entries[i].1).sum());
        let available = w.cuda.admission_memory()?.admitted_available;
        // Agree before deleting anything. A rank-local refusal must not cause
        // the other devices to spend cache on a park that cannot take place.
        if !w.control.all_admit(evictions.is_some() && available.saturating_add(reclaimable) >= reserve+charge)? {
            return Ok(None);
        }
    }
    let within_budget=evictions.is_some();
    let mut evicted=0;
    let evicted_protected=evictions.as_ref().map_or(0,|v|v.iter().filter(|&&i|protected.unwrap_or(0)&(1<<i)!=0).count());
    if let Some(mut indices)=evictions {
        indices.sort_unstable();evicted=indices.len();
        for i in indices.into_iter().rev(){w.cached_sessions.remove(i);}
    }
    if !w.control.all_admit(within_budget && w.cuda.admission_memory()?.admitted_available >= reserve+charge)? {
        return Ok(None);
    }
    w.stream.synchronize()?;
    let began=Instant::now();
    let image=if exchange {
        let local=match w.session.as_mut().unwrap().save_exchange_state_image(&w.verify,w.history.pending().unwrap(),device,&w.comm,&w.stream){
            Ok(image)=>Some(image),Err(e)if e.downcast_ref::<crate::state_image::CapacityRefused>().is_some()=>None,Err(e)=>return Err(e),
        };
        if w.control.all_admit(local.is_some()&&w.cuda.admission_memory()?.admitted_available>=reserve)?{local}else{None}
    }else{w.try_save_session_image(w.history.pending().unwrap(),device)?};
    let Some(image)=image else {
        return Ok(None);
    };
    w.stream.synchronize()?;
    ensure!(image.device_bytes()==device,"checkpoint allocation estimate");
    let saved=RequestCheckpoint{image,history:w.history.clone(),pending_prefill:w.pending_prefill.clone(),media:w.active_media.clone(),id:w.current.unwrap(),
        emitted:w.emitted,limit:w.limit,charge};
    let receipt=json!({"request_id":saved.id,"frontier":saved.history.frontier(),
        "emitted":saved.emitted,"limit":saved.limit,"charge_bytes":charge,
        "held_bytes":other_charge+charge,"evicted_cache_entries":evicted,
        "protected_entries_reclaimed_for_admitted_owner":evicted_protected,
        "host_allocator_trim":host_allocator_trim,
        "prefill_working_allowance_bytes":working_room,
        "cached_bytes":w.cached_sessions.iter().map(|c|c.charge).sum::<usize>(),
        "save_seconds":began.elapsed().as_secs_f64(),
        "manifest":saved.image.manifest()});
    w.agree(&[])?;
    Ok(Some((saved,receipt)))
}

/// Exchange one live and one inactive owner without a second full snapshot.
/// Existing completed caches may supply growth capacity; unfinished state is
/// never an eviction candidate. All ranks prepare before any live-byte write.
pub(super) fn try_exchange(w:&mut Worker,saved:&mut RequestCheckpoint)->Result<Option<Value>> {
    w.state_pressure_trim=None;
    ensure!(w.current.is_some()&&Some(saved.id)!=w.current&&saved.exchangeable()
        &&w.verify.profile_is_current()&&w.verify.position()==w.history.frontier()as u64,
        "exchange needs two distinct committed owners");
    let(extra,device)=w.session.as_ref().unwrap().exchange_growth(&w.verify,&saved.image,w.shared_image_budget)?;
    let(_,host)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
    let charge=device+host+w.history.owned_token_bytes()
        +w.pending_prefill.as_ref().map_or(0,|p|p.owned_token_bytes())+w.active_media.as_ref().map_or(0,|m|m.owned_bytes());
    let peak_charge=charge.max(saved.charge.checked_add(extra).ok_or_else(||anyhow::anyhow!("exchange peak charge overflow"))?);
    let entries=w.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
    // Unlike optional creation of another parked image, this moves existing
    // admitted owners. Their growth takes precedence over borrowed cache space.
    let evictions=cache_evictions(&entries,(usize::MAX,peak_charge),w.shared_image_budget,true,5);
    let reserve=crate::memory::floor_bytes()+16*1024usize.pow(2);
    if !w.control.all_admit(evictions.is_some())?{return Ok(None);}
    let mut evictions=evictions.unwrap();let mut available=w.cuda.admission_memory()?.admitted_available;
    let mut host_allocator_trim=None;
    let mut policy=vec![entries.len()as u8];policy.extend(evictions.iter().map(|&i|i as u8));
    w.control.agree(&policy)?;
    // Tightest-rank admission chooses a common eviction set. Do not let local
    // physical pressure independently change cache indexing on different ranks.
    loop {
        let reclaimed=evictions.iter().map(|&i|entries[i].1).sum::<usize>();
        if w.control.all_admit(available.saturating_add(reclaimed)>=reserve+extra)?{break;}
        if w.trim_free_host_pages && host_allocator_trim.is_none() {
            available=trim_for_state(w,reserve+extra)?;
            host_allocator_trim=w.state_pressure_trim.clone();
            continue;
        }
        let Some(i)=(0..entries.len()).find(|i|!evictions.contains(i))else{return Ok(None);};
        evictions.push(i);
    }
    evictions.sort_unstable();
    let evicted=evictions.len();
    for i in evictions.into_iter().rev(){w.cached_sessions.remove(i);}
    if !w.control.all_admit(w.cuda.admission_memory()?.admitted_available>=reserve+extra)?{return Ok(None);}
    w.cuda.synchronize_all()?;let began=Instant::now();
    let prepared=w.session.as_mut().unwrap().prepare_exchange(&w.verify,&saved.image,w.shared_image_budget,&w.comm,&w.stream)?;
    if !w.control.all_admit(prepared.is_some()&&w.cuda.admission_memory()?.admitted_available>=reserve)?{return Ok(None);}
    let outgoing=w.current.unwrap();let incoming=saved.id;
    w.target_lineage=w.target_lineage.checked_add(1).ok_or_else(||anyhow::anyhow!("exchange lineage overflow"))?;
    let anchor=w.session.as_mut().unwrap().exchange_state_image(&mut w.verify,&mut saved.image,prepared.unwrap(),
        w.history.pending().unwrap(),w.target_lineage,&w.comm,&w.stream)?;
    w.history.exchange(&mut saved.history)?;
    let outgoing_media=w.active_media.clone();w.install_media(saved.media.clone())?;saved.media=outgoing_media;
    std::mem::swap(&mut w.pending_prefill,&mut saved.pending_prefill);
    std::mem::swap(&mut w.emitted,&mut saved.emitted);std::mem::swap(&mut w.limit,&mut saved.limit);
    w.current=Some(incoming);saved.id=outgoing;saved.charge=charge;
    w.stream.synchronize()?;
    ensure!(Some(anchor)==w.history.pending()&&w.verify.position()==w.history.frontier()as u64
        &&w.verify.logits().is_err()&&!w.large.profile_is_current()&&saved.image.device_bytes()==device,
        "exchanged request frontier, charge or stale scratch exposure");
    w.agree(&[])?;
    Ok(Some(json!({"type":"exchange","request_id":incoming,"outgoing_request_id":outgoing,"frontier":w.history.frontier(),
        "emitted":w.emitted,"limit":w.limit,"charge_bytes":charge,"preparation_extra_bytes":extra,
        "peak_inactive_charge_bytes":peak_charge,
        "host_allocator_trim":host_allocator_trim,
        "evicted_cache_entries":evicted,"seconds":began.elapsed().as_secs_f64()})))
}

fn trim_for_state(w:&mut Worker,required_available:usize)->Result<usize> {
    let before=w.cuda.admission_memory()?;
    let began=Instant::now();
    let released=crate::memory::trim_free_host_pages();
    let seconds=began.elapsed().as_secs_f64();
    let after=w.cuda.admission_memory()?;
    w.state_pressure_trim=Some(json!({"returned_pages":released,"seconds":seconds,
        "required_available":required_available,"memory_before":before,"memory_after":after}));
    Ok(after.admitted_available)
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
    let reserve=crate::memory::floor_bytes()+16*1024usize.pow(2);
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

