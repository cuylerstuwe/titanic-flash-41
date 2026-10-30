//! Bounded HTTP scheduling. Peers alternate after64 emitted tokens at a
//! committed decode boundary; foreground work holds every background owner.
//! Up to two inactive images share2GiB. A priority switch that cannot retain
//! background progress rejects the incoming foreground request explicitly.
//! Background prefill can yield at encoder chunks under an explicit policy;
//! router holds drain all ranks and retain active state in place until needed.
use super::*;
use super::request_checkpoint::{RequestCheckpoint,try_checkpoint,suspend,restore};
use crate::{http_server::{Job,Queue},protocol::BackendEvent};
use deepseek_recipe::stream::InferenceFinishReason;
use std::collections::VecDeque;
use crate::router_priority::Snapshot as PrioritySnapshot;

const QUANTUM: usize=64;
enum Scheduled {
    Native(Command),
    Park,
    BeginBackground(Command),
    Prefill,
    Resume(u64),
    Discard(u64),
    Priority(PrioritySnapshot),
    Hold,
    OutputWait,
}
impl Scheduled {
    fn follows_progress(&self,paused_ids:&[u64],remaining:usize)->bool {
        match self {
            Self::Native(Command::Step|Command::Finish(_)|Command::ReclaimInputs{..})=>true,
            Self::Priority(_)|Self::Hold|Self::OutputWait=>true,
            Self::Park=>paused_ids.len()<2 && remaining>0,
            Self::Discard(id)=>paused_ids.contains(id),
            _=>false,
        }
    }
    fn encode(&self)->Result<Vec<u8>> {
        match self {
            Self::Native(command)=>command.encode(),
            Self::BeginBackground(command)=>{
                ensure!(matches!(command,Command::Begin{..}),"background start wire command");
                let mut bytes=vec![10];bytes.extend(command.encode()?);Ok(bytes)
            }
            Self::Prefill=>Ok(vec![11]),
            Self::Priority(snapshot)=>{
                let mut bytes=vec![14];bytes.extend(snapshot.revision.to_le_bytes());bytes.push(u8::from(snapshot.hold));Ok(bytes)
            }
            Self::Hold=>Ok(vec![15]),
            Self::OutputWait=>Ok(vec![16]),
            Self::Park=>Ok(vec![7]),
            Self::Resume(id)|Self::Discard(id)=>{
                let mut bytes=vec![if matches!(self,Self::Resume(_)){8}else{9}];
                bytes.extend_from_slice(&id.to_le_bytes());Ok(bytes)
            }
        }
    }
    fn decode(bytes:&[u8])->Result<Self> {
        match bytes {
            [7]=>Ok(Self::Park),
            [11]=>Ok(Self::Prefill),
            [14,..] if bytes.len()==10 && bytes[9]<=1=>Ok(Self::Priority(PrioritySnapshot{
                revision:u64::from_le_bytes(bytes[1..9].try_into().unwrap()),hold:bytes[9]!=0})),
            [15]=>Ok(Self::Hold),
            [16]=>Ok(Self::OutputWait),
            [10,rest @ ..]=>{
                let command=Command::decode(rest)?;
                ensure!(matches!(command,Command::Begin{..}),"invalid background start frame");
                Ok(Self::BeginBackground(command))
            }
            [8|9,..]if bytes.len()==9=>{
                let id=u64::from_le_bytes(bytes[1..].try_into().unwrap());
                ensure!(id>0,"invalid scheduled request identity");
                Ok(if bytes[0]==8 {Self::Resume(id)}else{Self::Discard(id)})
            }
            _=>Ok(Self::Native(Command::decode(bytes)?)),
        }
    }
}

fn execute(w:&mut Worker,images:&mut Vec<RequestCheckpoint>,command:Scheduled,
    queue:Option<&Queue>,job:Option<&Job>,priority:&mut PrioritySnapshot)->Result<Value> {
    update_reservations(w,images)?;
    w.loaded_request_is_suspended=matches!(command,Scheduled::Native(Command::Begin{..})|Scheduled::BeginBackground(_))
        && images.iter().any(|p|p.matches_history(&w.history));
    let result=match command {
        Scheduled::Priority(snapshot)=>{
            ensure!(snapshot.revision>=priority.revision,"priority revision went backwards");
            w.cuda.synchronize_all()?;
            let mut proof=w.agreement_state(&[]);
            proof.extend_from_slice(b"priority-drained-v1");proof.extend(snapshot.revision.to_le_bytes());proof.push(u8::from(snapshot.hold));
            w.control.agree(&proof)?;
            *priority=snapshot;
            if let Some(queue)=queue {queue.priority_applied(snapshot)?;}
            Ok(json!({"type":"priority","revision":snapshot.revision,"hold":snapshot.hold,
                "request_id":w.current,"frontier":w.history.frontier(),"emitted_tokens":w.emitted,"all_ranks_drained":true}))
        }
        Scheduled::Hold=>{
            ensure!(priority.hold,"background hold without an applied lease");
            w.agree(&[])?;
            Ok(json!({"type":"hold"}))
        }
        Scheduled::OutputWait=>{
            ensure!(w.current.is_some() || !images.is_empty(), "output wait without retained request state");
            w.agree(&[])?;
            Ok(json!({"type":"output_wait"}))
        }
        Scheduled::BeginBackground(command)=>w.start_background(command,queue,job),
        Scheduled::Prefill=>w.continue_background(queue,job),
        Scheduled::Native(command)=>{
            ensure!(!matches!(command,Command::Shutdown)||images.is_empty(),
                "shutdown cannot strand suspended requests");
            w.execute_http(command,queue,job)
        }
        Scheduled::Park=>{
            ensure!(images.len()<2 && !images.iter().any(|p|Some(p.id)==w.current),
                "bounded scheduler image ownership");
            let id=w.current.ok_or_else(||anyhow::anyhow!("park without active request"))?;
            let began=Instant::now();
            let saved=try_checkpoint(w,images.iter().map(|p|p.charge).sum(),images.len())?;
            let admitted=saved.is_some();
            let mut charge=0;
            let mut evicted=0;
            if let Some((saved,receipt))=saved {
                evicted=receipt["evicted_cache_entries"].as_u64().unwrap();
                charge=saved.charge;images.push(saved);suspend(w)?;
            }
            Ok(json!({"type":"park","request_id":id,"parked":admitted,
                "charge_bytes":charge,"held_bytes":images.iter().map(|p|p.charge).sum::<usize>(),
                "evicted_cache_entries":evicted,
                "seconds":began.elapsed().as_secs_f64()}))
        }
        Scheduled::Resume(id)|Scheduled::Discard(id)=>{
            let index=images.iter().position(|p|p.id==id)
                .ok_or_else(||anyhow::anyhow!("unknown suspended request"))?;
            let saved=images.remove(index);
            if matches!(command,Scheduled::Resume(_)) {
                let mut event=restore(w,&saved)?;
                event["type"]=json!("resume");Ok(event)
            }else{
                w.agree(&[])?;
                Ok(json!({"type":"discard","request_id":id}))
            }
        }
    };
    update_reservations(w,images)?;
    let mut event=result?;
    event["suspended_image_bytes"]=json!(w.suspended_image_bytes);
    event["cached_image_bytes"]=json!(w.cached_sessions.iter().map(|p|p.charge).sum::<usize>());
    event["shared_image_bytes"]=json!(w.suspended_image_bytes+w.cached_sessions.iter().map(|p|p.charge).sum::<usize>());
    event["shared_image_count"]=json!(images.len()+w.cached_sessions.len());
    event["shared_image_budget_bytes"]=json!(w.shared_image_budget);
    Ok(event)
}

fn update_reservations(w:&mut Worker,images:&[RequestCheckpoint])->Result<()> {
    w.suspended_image_bytes=images.iter().map(|p|p.charge).sum();
    w.suspended_image_count=images.len();
    ensure!(w.suspended_image_bytes+w.cached_sessions.iter().map(|p|p.charge).sum::<usize>()<=w.shared_image_budget
        && w.suspended_image_count+w.cached_sessions.len()<=4,"shared image budget exceeded");
    Ok(())
}

fn peer_turn_due(emitted:usize,quantum_start:usize)->bool {
    emitted.saturating_sub(quantum_start)>=QUANTUM
}
fn runnable_peer(active_background:bool,peer_background:bool)->bool {
    active_background || !peer_background
}

fn choose(w:&mut Worker,queue:&Queue,active:&mut Option<Job>,ready:&mut VecDeque<(Option<u64>,Job)>,
    next_id:&mut u64,quantum_start:usize,refused:bool,paused_count:usize)->Result<Scheduled> {
    loop {
        // Paused clients retain their bounded output queue. Advance the stall
        // deadline even while another request owns the GPU.
        for (_,job) in ready.iter() {job.output_backpressured();}
        // CPU-only cancellation can be drained before any wire command. A
        // paused request also needs an ordered command to release its image.
        if let Some(index)=ready.iter().position(|(_,job)|job.cancelled()||queue.stopping()) {
            let (id,job)=ready.remove(index).unwrap();
            job.emit(BackendEvent::Failed("request cancelled before completion".into()));
            if let Some(id)=id {return Ok(Scheduled::Discard(id));}
            continue;
        }
        if !queue.stopping() {
            if let Some(snapshot)=queue.priority_pending() {return Ok(Scheduled::Priority(snapshot));}
            let headroom=queue.input_reclaim_target();
            if headroom>0 {return Ok(Scheduled::Native(Command::ReclaimInputs{headroom}));}
        }
        if w.current.is_some() {
            let job=active.as_ref().ok_or_else(||anyhow::anyhow!("lost active HTTP owner"))?;
            let output_blocked=job.output_backpressured();
            let prefill=w.pending_prefill.is_some();
            let next=if prefill {
                if job.cancelled()||queue.stopping(){Scheduled::Native(Command::Finish(Finish::Cancelled))}
                else{Scheduled::Prefill}
            }else{Scheduled::Native(w.active_command(job.cancelled()||queue.stopping())?)};
            let can_work=matches!(next,Scheduled::Native(Command::Step)|Scheduled::Prefill);
            let held=job.background&&queue.router_hold();
            if job.background && refused && can_work
                && (held||prefill||peer_turn_due(w.emitted,quantum_start)) {
                while let Some(job)=queue.try_foreground()? {
                    job.emit(BackendEvent::Failed("foreground capacity unavailable: background progress could not be retained".into()));
                }
            }
            if can_work && !refused && paused_count<2
                && (held||output_blocked||prefill||peer_turn_due(w.emitted,quantum_start)) {
                // A foreground arrival may supersede both background peers.
                // Save the second background too, within the same image bound;
                // further foreground peers queue if both images remain held.
                if !ready.iter().any(|(_,j)|!j.background && !j.output_backpressured()) {
                    if let Some(job)=queue.try_foreground()? {ready.push_front((None,job));}
                }
                if ready.is_empty() && job.background && !held {
                    if let Some(job)=queue.try_next()? {ready.push_back((None,job));}
                }
                if ready.iter().any(|(_,peer)|runnable_peer(job.background,peer.background)
                    && (!held||!peer.background) && !peer.output_backpressured()) {
                    return Ok(Scheduled::Park);
                }
            }
            if can_work && (held||output_blocked) {
                // No state copy is needed when foreground work is on another
                // backend. Local foreground still takes the normal Park path.
                std::thread::sleep(Duration::from_millis(100));
                if let Some(snapshot)=queue.priority_pending() {return Ok(Scheduled::Priority(snapshot));}
                return Ok(if held {Scheduled::Hold}else{Scheduled::OutputWait});
            }
            return Ok(next);
        }
        if queue.stopping() {
            ensure!(ready.is_empty(),"shutdown queue not drained");
            return Ok(Scheduled::Native(Command::Shutdown));
        }
        // Never resume background between two queued foreground requests.
        if !ready.iter().any(|(_,job)|!job.background && !job.output_backpressured()) {
            if let Some(job)=queue.try_foreground()? {ready.push_front((None,job));}
        }
        if ready.is_empty() {
            if let Some(job)=queue.next(Duration::from_millis(500))? {ready.push_back((None,job));}
        }else if !queue.router_hold() && !ready.iter().any(|(_,job)|!job.background)
            && ready.iter().all(|(_,job)|job.output_backpressured()) {
            if let Some(job)=queue.try_next()? {ready.push_back((None,job));}
        }
        let selected=ready.iter().position(|(_,job)|!job.background && !job.output_backpressured())
            .or_else(|| if !queue.router_hold() && !ready.iter().any(|(_,job)|!job.background) {
                ready.iter().position(|(_,job)|!job.output_backpressured())
            }else{None});
        if let Some((id,mut job))=selected.and_then(|i|ready.remove(i)) {
            let command=if let Some(id)=id {Scheduled::Resume(id)}else{
                let id=*next_id;*next_id=next_id.checked_add(1)
                    .ok_or_else(||anyhow::anyhow!("HTTP request counter exhausted"))?;
                if !job.images.is_empty() {
                    w.media.as_mut().ok_or_else(|| anyhow::anyhow!("image request without native vision"))?
                        .set_inputs(&std::mem::take(&mut job.image_inputs))?;
                }
                let command=Command::Begin{images: std::mem::take(&mut job.images),id,seed:job.seed,greedy:job.greedy,
                    limit:job.limit,tokens:std::mem::take(&mut job.tokens)};
                if job.background&&w.background_prefill_enabled {Scheduled::BeginBackground(command)}
                else{Scheduled::Native(command)}
            };
            *active=Some(job);return Ok(command);
        }
        // A lease may arrive while next() waits for new HTTP work.
        if let Some(snapshot)=queue.priority_pending() {return Ok(Scheduled::Priority(snapshot));}
        if queue.router_hold() || ready.iter().any(|(_,job)|job.output_backpressured()) {
            std::thread::sleep(Duration::from_millis(100));
            if let Some(snapshot)=queue.priority_pending() {return Ok(Scheduled::Priority(snapshot));}
            return Ok(if queue.router_hold() {Scheduled::Hold}else{Scheduled::OutputWait});
        }
        return Ok(Scheduled::Native(Command::Idle));
    }
}

pub(super) fn run(w:&mut Worker,queue:Option<&Queue>,events:&mut crate::event_log::EventLog)->Result<()> {
    let mut images=Vec::new();
    let mut active:Option<Job>=None;
    // None is a newly admitted HTTP job; Some(id) owns a saved unfinished one.
    let mut ready:VecDeque<(Option<u64>,Job)>=VecDeque::new();
    let mut next_id=1u64;let mut quantum_start=0;let mut refused=false;
    let mut authorized_next=None;
    let mut priority=PrioritySnapshot::default();
    let result=(||->Result<()>{
        loop {
            let began=Instant::now();
            let bytes=if authorized_next.is_some(){None}else if let Some(queue)=queue {
                Some(choose(w,queue,&mut active,&mut ready,&mut next_id,quantum_start,refused,images.len())?.encode()?)
            }else{None};
            let wire=if let Some(next)=authorized_next.take(){next}else{w.control.command(bytes.as_deref())?};
            let command=Scheduled::decode(&wire)?;
            let started=matches!(command,Scheduled::Native(Command::Begin{..})|Scheduled::BeginBackground(_));
            let resumed=matches!(command,Scheduled::Resume(_));
            let finished=matches!(command,Scheduled::Native(Command::Finish(_)));
            let idle=matches!(command,Scheduled::Native(Command::Idle)|Scheduled::Hold|Scheduled::OutputWait);
            let shutdown=matches!(command,Scheduled::Native(Command::Shutdown));
            let discarded=matches!(command,Scheduled::Discard(_));
            let mut event=execute(w,&mut images,command,queue,active.as_ref(),&mut priority)?;
            if started {event["request_started"]=json!(true);}
            let input_refused=event["type"]=="input_refused";
            if started && !input_refused {event["vision"]=w.media_receipt.clone();}
            if started||resumed {quantum_start=if started {0}else{w.emitted};refused=false;}
            if w.progress_next && event["type"]=="tokens" {
                // Exactly the normal fast path for an uncontended stream.
                // The bounded next frame may also authorize a park or release
                // an already paused request, never a new prompt or restoration.
                let agreement_began=Instant::now();
                let next=queue.map(|queue|choose(w,queue,&mut active,&mut ready,&mut next_id,quantum_start,refused,images.len())
                    .and_then(|command|command.encode())).transpose()?;
                let tokens:Vec<u32>=serde_json::from_value(event["output_ids"].clone())?;
                let wire=w.control.agree_and_next(&w.agreement_state(&tokens),next.as_deref())?;
                let paused_ids=images.iter().map(|image|image.id).collect::<Vec<_>>();
                ensure!(Scheduled::decode(&wire)?.follows_progress(&paused_ids,w.history.step_limit()?),
                    "progress agreement cannot admit a new prompt or restore an image");
                authorized_next=Some(wire);
                event["progress_agreement_seconds"]=json!(agreement_began.elapsed().as_secs_f64());
            }
            if let Some(job)=active.as_ref().filter(|_|!discarded) {
                if !input_refused && (started || (job.background && event["background_prefill"]==true)) {
                    job.record_prefill_timing(&event);
                }
                if started && !input_refused {
                    // Capture this request's media receipt before another
                    // request can replace the worker's active receipt.
                    job.emit(BackendEvent::Ready{fingerprint:w.identity.clone(),
                        cached_tokens:event["cached_tokens"].as_u64().unwrap()as usize});
                    event["queue_and_prefill_seconds"]=json!(job.enqueued.elapsed().as_secs_f64());
                }
                if input_refused {
                    job.emit(BackendEvent::Failed(event["message"].as_str()
                        .ok_or_else(||anyhow::anyhow!("input refusal message missing"))?.into()));
                }else if event["type"]=="tokens" {
                    job.emit(BackendEvent::Tokens(serde_json::from_value(event["output_ids"].clone())?));
                }else if finished {
                    job.emit(match event["reason"].as_str(){
                        Some("stop")=>BackendEvent::Finish(InferenceFinishReason::Stop),
                        Some("length")=>BackendEvent::Finish(InferenceFinishReason::Length),
                        Some("cancelled")=>BackendEvent::Failed("request cancelled before completion".into()),
                        _=>anyhow::bail!("unknown scheduled completion"),
                    });
                }
                event["http_request_id"]=json!(job.id);
                event["request_mode"]=json!(if job.background {"background"}else{"foreground"});
            }
            if event["type"]=="park" {
                if event["parked"]==true {
                    if let Some(job)=active.take(){ready.push_back((event["request_id"].as_u64(),job));}
                }else{
                    refused=true;
                    if active.as_ref().is_some_and(|job|job.background) {
                        // Serial fallback is fine for equal peers. It is not a
                        // successful priority handoff: reject new foreground
                        // work rather than cancel/replay the background owner.
                        let mut index=0;
                        while index<ready.len() {
                            if !ready[index].1.background {
                                let (id,job)=ready.remove(index).unwrap();
                                ensure!(id.is_none(),"foreground image unexpectedly held behind background");
                                job.emit(BackendEvent::Failed("foreground capacity unavailable: background progress could not be retained".into()));
                            }else{index+=1;}
                        }
                    }
                }
            }
            if finished || input_refused {active=None;}
            if !idle {
                redact_http_event(&mut event);
                event["http_interleave"]=json!(true);event["quantum_tokens"]=json!(QUANTUM);
                event["progress_next"]=json!(w.progress_next);
                event["control_and_execution_seconds"]=json!(began.elapsed().as_secs_f64());
                events.record(&event)?;
            }
            if shutdown {break;}
        }
        Ok(())
    })();
    // Local images are dropped when this function returns, before Worker::drop.
    // Retire CPU control first so peers leave blocked reads before any GPU
    // image deallocation or collective teardown can delay the failed group.
    w.history.retire();
    w.control.retire();
    if result.is_err() {events.worker_failed(w.current);}
    if let Some(job)=active {job.emit(BackendEvent::Failed("native worker group failed".into()));}
    for (_,job)in ready {job.emit(BackendEvent::Failed("native worker group failed".into()));}
    if let Some(queue)=queue {queue.fail();}
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scheduler_commands_preserve_ids_and_reject_truncation() {
        for hold in [false,true] {
            let wire=Scheduled::Priority(PrioritySnapshot{revision:17,hold}).encode().unwrap();
            assert_eq!(Scheduled::decode(&wire).unwrap().encode().unwrap(),wire);
            assert!(Scheduled::decode(&wire[..9]).is_err());
            let mut invalid=wire;invalid[9]=2;assert!(Scheduled::decode(&invalid).is_err());
        }
        assert_eq!(Scheduled::decode(&[15]).unwrap().encode().unwrap(),[15]);
        assert!(matches!(Scheduled::decode(&[16]).unwrap(),Scheduled::OutputWait));
        assert!(Scheduled::decode(&[16,0]).is_err());
        assert!(Scheduled::OutputWait.follows_progress(&[],6));
        for id in [1,1u64<<40,u64::MAX] {
            for command in [Scheduled::Resume(id),Scheduled::Discard(id)] {
                let wire=command.encode().unwrap();
                assert_eq!(Scheduled::decode(&wire).unwrap().encode().unwrap(),wire);
                assert!(Scheduled::decode(&wire[..8]).is_err());
            }
        }
        assert!(Scheduled::decode(&[8,0,0,0,0,0,0,0,0]).is_err());
        assert_eq!(Scheduled::decode(&[7]).unwrap().encode().unwrap(),[7]);
    }
    #[test]
    fn progress_cannot_start_or_restore_a_request_or_discard_an_unknown_owner() {
        let reclaim=Scheduled::Native(Command::ReclaimInputs{headroom:9usize<<30});
        assert!(reclaim.follows_progress(&[2,3],0));
        assert_eq!(Scheduled::decode(&reclaim.encode().unwrap()).unwrap().encode().unwrap(),reclaim.encode().unwrap());
        assert!(Scheduled::Park.follows_progress(&[2],6));
        assert!(!Scheduled::Park.follows_progress(&[2],0));
        assert!(!Scheduled::Park.follows_progress(&[2,3],6));
        assert!(Scheduled::Discard(2).follows_progress(&[2],6));
        assert!(!Scheduled::Discard(3).follows_progress(&[2],6));
        assert!(!Scheduled::Resume(2).follows_progress(&[2],6));
        assert!(!Scheduled::Native(Command::Shutdown).follows_progress(&[2],6));
        assert!(!Scheduled::Native(Command::Begin{images: Vec::new(),id:3,seed:7,greedy:false,limit:1,tokens:vec![42]})
            .follows_progress(&[2],6));
    }
}
