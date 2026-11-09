//! Two execution owners, with FIFO admission separate from local priority.
//! Kept behind an explicit experiment flag until router lease coordination
//! and the directly affected state/slow-reader paths have been qualified.
use super::*;
use crate::http_server::PrefillPeer;

fn peer_work(w:&Worker,id:Option<u64>,job:&Job,images:&[ParkedRequest])->Result<PrefillPeer> {
    if let Some(id)=id {
        let image=images.iter().find(|p|p.id==id).context("missing peer work image")?;
        Ok(image.prefill_remaining(w)?.map_or(PrefillPeer::Decode,PrefillPeer::Prompt))
    }else{Ok(PrefillPeer::Prompt(estimated_peer_remaining(job,&w.cached_sessions,None)))}
}

fn publish_peer_work(w:&Worker,queue:&Queue,background:bool,ready:&VecDeque<(Option<u64>,Job)>,
    images:&[ParkedRequest],turns:&Turns,refused:bool)->Result<()> {
    let peer=ready.iter().find(|(_,j)|runnable_peer(background,j.background)
        &&!j.cancelled()&&!j.output_backpressured());
    queue.set_prefill_peer_ready(!refused&&peer.is_some());
    let work=if turns.enabled&&!refused {peer.map(|(id,j)|peer_work(w,*id,j,images)).transpose()?}else{None};
    queue.set_prefill_work(turns.enabled,work,turns.prefill_protection());
    Ok(())
}

// Priority and eligibility are settled first. Within that class, repay a
// passed-over owner, then start the next admitted newcomer. Two fresh owners
// choose less uncached work, with stable FIFO ties.
fn selection_key(turns:&Turns,id:Option<u64>,http:&str,remaining:usize,index:usize)->(u8,usize,usize) {
    if !turns.enabled{return (0,0,index);}
    if turns.owes(http){(0,0,index)}
    else if id.is_none(){(1,remaining,index)}else{(2,0,index)}
}

fn occupied(active: bool, ready: &VecDeque<(Option<u64>, Job)>) -> usize {
    // Parking, priority holds and backpressure never surrender ownership.
    // Only retiring the request removes it from active/ready.
    usize::from(active) + ready.len()
}

pub(super) fn choose(w:&mut Worker, queue:&Queue, active:&mut Option<Job>,
    ready:&mut VecDeque<(Option<u64>,Job)>, next_id:&mut u64,
    quantum_start:usize, prefill_turn_done:bool, refused:bool, images:&[ParkedRequest],
    protection:&mut Protection,turns:&mut Turns,idle_policy:&mut IdlePolicy)->Result<Scheduled> {
    let paused_count=images.len();
    loop {
        queue.set_prefill_peer_ready(false);
        queue.set_prefill_work(turns.enabled,None,None);
        // Poll the existing terminal stall timeout even for parked owners;
        // a temporary slow reader retains both its state and admission.
        for (_,job) in ready.iter() {job.output_backpressured();}
        if let Some(i)=ready.iter().position(|(_,j)|j.cancelled()||queue.stopping()) {
            let(id,job)=ready.remove(i).unwrap();
            job.emit(BackendEvent::Failed("request cancelled before completion".into()));
            turns.retired();
            if let Some(id)=id{return Ok(Scheduled::Discard(id));}
            continue;
        }
        if !queue.stopping() {
            if let Some(snapshot)=queue.priority_pending(){return Ok(Scheduled::Priority(snapshot));}
            let target=queue.input_reclaim_target();
            if target>0{return Ok(Scheduled::Native(Command::ReclaimInputs{headroom:target}));}
            let mut used=occupied(active.is_some(),ready);
            ensure!(used<=2,"two-stream execution admission exceeded");
            if used>0 {protection.begin_epoch();}
            while used<2 {
                let Some(job)=queue.try_fifo()? else {break;};
                protection.begin_epoch();
                ready.push_back((None,job));used+=1;
            }
            queue.set_execution_owners(used)?;
        }
        turns.refresh(active.as_ref().map(|j|j.id.as_str()),ready,w.pending_prefill.is_some(),w.emitted.saturating_sub(quantum_start));
        let handoff_wait=turns.take_handoff_wait();
        if !handoff_wait.is_zero()&&!queue.stopping()&&w.current.is_none()&&active.is_none()
            &&ready.len()==1&&ready[0].0.is_some()&&!turns.owes(&ready[0].1.id)
            &&queue.input_pending_for_vacancy() {
            // A released permit wakes raw FIFO input asynchronously. Starting
            // an encoder chunk now can postpone a nearly ready newcomer by
            // seconds. Wait at most once, and never ahead of an owed owner.
            if let Some(job)=queue.next_fifo(handoff_wait)? {ready.push_back((None,job));}
            // Recheck cancellations, router holds and slot counts after waiting.
            continue;
        }
        if w.current.is_some() {
            let job=active.as_ref().ok_or_else(||anyhow::anyhow!("lost admitted HTTP owner"))?;
            publish_peer_work(w,queue,job.background,ready,images,turns,refused)?;
            let blocked=job.output_backpressured();let prefill=w.pending_prefill.is_some();
            let next=if prefill {
                if job.cancelled()||queue.stopping(){Scheduled::Native(Command::Finish(Finish::Cancelled))}
                else{Scheduled::Prefill}
            }else{Scheduled::Native(w.active_command(job.cancelled()||queue.stopping())?)};
            let work=matches!(next,Scheduled::Native(Command::Step)|Scheduled::Prefill);
            let held=job.background&&queue.router_hold();
            if work && refused && job.background {
                // A global foreground hold must not strand both the retained
                // background owner and a newcomer when parking was refused.
                // Only unstarted foreground jobs may fail admission; existing
                // paused owners always retain their complete state and permit.
                let mut removed=false;
                ready.retain(|(id,peer)| {
                    if id.is_none() && !peer.background {
                        peer.emit(BackendEvent::Failed("foreground capacity unavailable: background progress could not be retained".into()));
                        removed=true;false
                    }else{true}
                });
                if removed {turns.retired();continue;}
            }
            if work&&!prefill&&!held&&!blocked {
                if let Some((Some(id),peer))=ready.iter().find(|(_,p)|p.background==job.background
                    &&!p.cancelled()&&!p.output_backpressured()) {
                    if w.resident_peer_terminal(*id) {return Ok(Scheduled::Exchange(*id));}
                    if w.resident_pair_can_step(*id)? {
                        job.set_executing(true);peer.set_executing(true);
                        return Ok(Scheduled::PairStep(*id));
                    }
                }
            }
            // A foreground conversion can finish just after an older
            // background owner is restored. Repayment/first-turn protection
            // applies within a priority class, never across this boundary.
            let foreground_peer=job.background&&ready.iter().any(|(_,p)|!p.background
                &&!p.cancelled()&&!p.output_backpressured());
            let protected=turns.protected(prefill,w.emitted.saturating_sub(quantum_start));
            let new_peer=ready.iter().any(|(id,p)|id.is_none()&&runnable_peer(job.background,p.background)
                &&!p.cancelled()&&!p.output_backpressured());
            let decode_turn=!prefill&&!turns.first_grant()&&!protected
                &&(peer_turn_due(w.emitted,quantum_start)||(turns.enabled&&new_peer));
            if work&&!refused&&paused_count<2
                &&(held||foreground_peer||blocked||(prefill&&prefill_turn_done)||decode_turn) {
                let peer=ready.iter().find(|(_,p)|runnable_peer(job.background,p.background)&&(!held||!p.background)
                    &&!p.output_backpressured());
                if let Some((id,_))=peer {
                    if let Some(id)=id.filter(|id|images.iter().any(|p|p.id==*id&&p.exchangeable())){
                        return Ok(Scheduled::Exchange(id));
                    }
                    return Ok(Scheduled::Park(Some(protection.mask(w,queue,ready,images)?)));
                }
            }
            if work&&(held||blocked) {
                job.set_executing(false);
                std::thread::sleep(Duration::from_millis(100));
                if let Some(s)=queue.priority_pending(){return Ok(Scheduled::Priority(s));}
                return Ok(if held{Scheduled::Hold}else{Scheduled::OutputWait});
            }
            job.set_executing(true);
            return Ok(next);
        }
        if queue.stopping() {
            ensure!(ready.is_empty(),"shutdown left admitted owners");
            return Ok(Scheduled::Native(Command::Shutdown));
        }
        let has_foreground=ready.iter().any(|(_,j)|!j.background);
        let selected=ready.iter().enumerate().filter(|(_,(_,j))|!j.output_backpressured()
            &&(!j.background||(!has_foreground&&!queue.router_hold())))
            .min_by_key(|(index,(id,j))|selection_key(turns,*id,&j.id,
                if turns.enabled&&id.is_none(){estimated_peer_remaining(j,&w.cached_sessions,None)}else{0},*index))
            .map(|(index,_)|index);
        if let Some((id,mut job))=selected.and_then(|i|ready.remove(i)) {
            let command=if let Some(id)=id{turns.activate(id,&job.id,false,None);Scheduled::Resume(id)}else{
                let id=*next_id;*next_id=next_id.checked_add(1)
                    .ok_or_else(||anyhow::anyhow!("HTTP request counter exhausted"))?;
                let peer=ready.iter().find(|(_,peer)|runnable_peer(job.background,peer.background)
                    &&!peer.cancelled()&&!peer.output_backpressured());
                let peer=peer.map(|(id,peer)|->Result<_>{Ok((peer.id.as_str(),id.is_none()
                    ||images.iter().find(|p|Some(p.id)==*id).map(|p|p.prefill_remaining(w))
                        .transpose()?.flatten().is_some()))}).transpose()?;
                turns.activate(id,&job.id,true,peer);
                if !job.images.is_empty(){w.media.as_mut().ok_or_else(||anyhow::anyhow!("image request without native vision"))?
                    .set_job_inputs(&mut job)?;}
                let c=Command::Begin{images:std::mem::take(&mut job.images),id,seed:job.seed,
                    greedy:job.greedy,limit:job.limit,tokens:std::mem::take(&mut job.tokens)};
                if job.background&&w.background_prefill_enabled{Scheduled::BeginBackground(c)}else{Scheduled::Native(c)}
            };
            publish_peer_work(w,queue,job.background,ready,images,turns,false)?;
            job.set_executing(true);
            *active=Some(job);return Ok(command);
        }
        if let Some(s)=queue.priority_pending(){return Ok(Scheduled::Priority(s));}
        if queue.router_hold()||!ready.is_empty() {
            std::thread::sleep(Duration::from_millis(100));
            if let Some(s)=queue.priority_pending(){return Ok(Scheduled::Priority(s));}
            return Ok(if queue.router_hold(){Scheduled::Hold}else{Scheduled::OutputWait});
        }
        // Raw queued inputs and admitted CPU conversions are still part of
        // this fan-out, even during a gap with no model-ready Job.
        if !queue.has_outstanding_requests() {protection.end_epoch();}
        if idle_policy.take()&&w.idle_anchor_can_prepare() {
            // A full logical cache cannot admit this prediction. Avoid even
            // the distributed command/physical-memory query in that case.
            // Let the completed SSE response retire its ingress permit. This
            // waits only while idle and wakes for a ready request or reclaim.
            // A raw upload/conversion also vetoes preparation, even before it
            // has produced a model-ready Job. Do not retry a missed idle gap.
            if let Some(job)=queue.next_fifo(Duration::from_millis(2))? {
                ready.push_back((None,job));continue;
            }
            if queue.stopping(){continue;}
            if let Some(s)=queue.priority_pending(){return Ok(Scheduled::Priority(s));}
            if !queue.router_hold()&&!queue.has_outstanding_requests()
                &&queue.input_reclaim_target()==0&&images.is_empty() {
                return Ok(Scheduled::IdleAnchor);
            }
        }
        // Wake immediately when a request arrives, just like the solo path;
        // polling sleep here would add avoidable first-token latency.
        if let Some(job)=queue.next_fifo(Duration::from_millis(500))? {
            ready.push_back((None,job));continue;
        }
        return Ok(Scheduled::Native(Command::Idle));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn same_class_selection_repays_an_owner_before_another_newcomer() {
        let mut turns=Turns::for_test();
        assert!(selection_key(&turns,None,"a",20,1)<selection_key(&turns,None,"b",40,0));
        assert!(selection_key(&turns,None,"b",20,0)<selection_key(&turns,None,"a",20,1));
        assert!(selection_key(&turns,None,"a",40,1)<selection_key(&turns,Some(1),"long",0,0));
        turns.activate(2,"short",true,Some(("long",true)));
        assert!(selection_key(&turns,Some(1),"long",0,1)<selection_key(&turns,None,"a",1,0));
        turns.enabled=false;
        assert!(selection_key(&turns,Some(1),"long",0,0)<selection_key(&turns,None,"a",1,1));
    }
}
