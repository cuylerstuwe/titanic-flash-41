//! Batched decode admission (CANADA_SCRATCH_BATCHED_DECODE=1; worker side in batched_serving.rs).
//! One HTTP owner per serving region, admitted FIFO: region 0's conversation plus one for each
//! region 1..R-1. Region 0 runs prefills and, while no region is occupied, today's ordinary
//! decode steps. A decoding region-0 conversation moves into a free region as soon as another
//! conversation is waiting, so the newcomer's prefill can start; every decode step is batched
//! while any region holds a conversation. A prefill yields to the region conversations at each
//! slice (CANADA_SCRATCH_BATCHED_PREFILL_SLICE_MS, default 500 ms): one batched step of theirs,
//! then the prefill continues. A region-0 conversation that cannot move (longer than a region, or
//! no region free) takes turns in region 0 with waiting ones through parked images, as the
//! two-owner scheduler does (one 64-token turn, at most two images; a conversation in its first
//! run yields to a new one after FIRST_TURN tokens, so the newcomer's prefill starts early).
//! Region 0 has no upper bound of its own in the compressed address space it shares with the
//! regions (Worker::region0_admits): a conversation that can reach past one region's capacity
//! takes region 0 only while every region is empty. A waiting one of that kind is not made room
//! for by moving region 0's conversation into a region (it takes turns with it instead), and when
//! regions are occupied it waits, first in line, until their conversations have finished and
//! moved back; nothing else starts ahead of it.
use super::*;
use crate::http_server::PrefillPeer;

pub(super) fn prefill_slice() -> Result<Duration> {
    Ok(Duration::from_millis(match std::env::var("CANADA_SCRATCH_BATCHED_PREFILL_SLICE_MS") {
        Ok(v) => { let ms: u64 = v.parse()?; ensure!((100..=5000).contains(&ms), "batched prefill slice"); ms }
        Err(std::env::VarError::NotPresent) => 500,
        Err(e) => return Err(e.into()),
    }))
}

/// Which waiting conversation takes a free region 0: the first in the order of `ready` (paused and new ones
/// alike; a conversation that was just paused is at the back), skipping background ones while the router
/// holds background work. `background`: each waiting conversation's mode, in that order.
fn next_for_region0(background: impl Iterator<Item = bool>, held: bool) -> Option<usize> {
    background.enumerate().find(|&(_, background)| !(held && background)).map(|(i, _)| i)
}

/// The conversation next in line for region 0 can reach past one serving region and a region is occupied:
/// it waits for the regions to empty (see the module text). `next_fits` is true when nothing is in line.
fn waits_for_empty_regions(next_fits: bool, occupied: bool) -> bool {
    !next_fits && occupied
}

/// Region 0's conversation cannot move and another is in line: it yields region 0 through a parked image when
/// its 64-token turn is over, and at once when the router holds it (a held background conversation makes no
/// progress, so its turn would never end) while one that may run is in line, as the two-owner scheduler does.
fn yields_region0(turn_due: bool, active_held: bool, runnable_in_line: bool) -> bool {
    turn_due || (active_held && runnable_in_line)
}

/// Tokens a conversation in its first run produces before it yields region 0 to a NEW conversation in line.
const FIRST_TURN: usize = 16;

/// Region 0's conversation is in its first run (begun, never paused: its turn counts from its first token), a
/// conversation that has not started is next in line, and FIRST_TURN tokens are out: it yields now instead of
/// after a whole 64-token turn, so the newcomer's prefill starts early, as the two-owner scheduler starts a
/// newcomer at once. Until Nov 15 2026 the second of two conversations that started together waited for the
/// first one's 64 tokens (first token 0.93 s against production's 0.61-0.72 s; production comparison, jobs
/// 11835c and 11840). A resumed conversation keeps its whole turn (`quantum_start` is then its position at
/// the resume), so a stream of newcomers cannot starve a paused conversation; and a few ordinary steps have
/// run before the pause, as at the end of a whole turn.
fn first_turn_over(emitted: usize, quantum_start: usize, next_is_new: bool) -> bool {
    quantum_start == 0 && next_is_new && emitted >= FIRST_TURN
}

/// The waiting conversation `entry` can share the owner with the regions: the positions it can reach (its prompt
/// or paused frontier plus its tokens, or the next lookahead positions: config::region_reach) fit one region.
fn reach_fits(entry: &(Option<u64>, Job), images: &[ParkedRequest], capacity: usize) -> bool {
    let reach = match entry.0 {
        Some(id) => images.iter().find(|p| p.id == id).and_then(|p| p.reach().ok().flatten()),
        None => entry.1.reach().ok(),
    };
    // Unknown (a physical bank, an unreadable setting): the old bound, prompt plus every token it may produce.
    super::super::batched_serving::fits_one_region(reach.unwrap_or_else(|| entry.1.extent()), capacity)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn choose(w: &mut Worker, queue: &Queue, active: &mut Option<Job>,
    ready: &mut VecDeque<(Option<u64>, Job)>, region_jobs: &[(u64, Job)], next_id: &mut u64,
    prefill_turn_done: &mut bool, quantum_start: usize, refused: bool, images: &[ParkedRequest]) -> Result<Scheduled> {
    let regions = crate::config::batched_regions()?;
    let capacity = w.region_capacity()?;
    loop {
        // Paused readers keep their bounded output queue; advance their stall deadlines.
        for job in ready.iter().map(|(_, job)| job).chain(region_jobs.iter().map(|(_, job)| job)) {
            job.output_backpressured();
        }
        // Cancelled waiting requests: a parked one releases its image through an ordered command.
        if let Some(i) = ready.iter().position(|(_, j)| j.cancelled() || queue.stopping()) {
            let (id, job) = ready.remove(i).unwrap();
            job.emit(BackendEvent::Failed("request cancelled before completion".into()));
            if let Some(id) = id { return Ok(Scheduled::Discard(id)); }
            continue;
        }
        // A region conversation that is done or cancelled ends first and frees its region.
        for (id, job) in region_jobs.iter() {
            if let Some(reason) = w.peer_finish_reason(*id, job.cancelled() || queue.stopping())? {
                return Ok(Scheduled::PeerFinish(*id, match reason { Finish::Stop => 0, Finish::Length => 1, Finish::Cancelled => 2 }));
            }
        }
        if !queue.stopping() {
            if let Some(snapshot) = queue.priority_pending() { return Ok(Scheduled::Priority(snapshot)); }
            let target = queue.input_reclaim_target();
            if target > 0 { return Ok(Scheduled::Native(Command::ReclaimInputs { headroom: target })); }
            let mut used = usize::from(active.is_some()) + ready.len() + region_jobs.len();
            while used < regions {
                let Some(job) = queue.try_fifo()? else { break; };
                ready.push_back((None, job));
                used += 1;
            }
        }
        let peers = !region_jobs.is_empty();
        // Region conversations whose readers are behind sit out steps until their queues drain.
        // So do background conversations while the router holds background work.
        let held = queue.router_hold();
        let mut skip = 0u8;
        for (id, job) in region_jobs.iter() {
            // A region conversation at its region's end (lookahead admission) sits out too.
            if job.output_backpressured() || (held && job.background) || w.peer_at_boundary(*id).unwrap_or(false) {
                if let Some(r) = w.peer_region(*id) { skip |= 1u8 << r; }
            }
        }
        let stepping = region_jobs.iter().any(|(id, _)| w.peer_region(*id).is_some_and(|r| skip & (1u8 << r) == 0));
        // A prefill yields to waiting region conversations each slice, and otherwise runs to its end.
        queue.set_prefill_peer_ready(peers);
        queue.set_prefill_work(true, peers.then_some(PrefillPeer::Decode),
            (!peers).then(|| Instant::now() + Duration::from_secs(3600)));
        if w.current.is_some() {
            let job = active.as_ref().ok_or_else(|| anyhow::anyhow!("lost active HTTP owner"))?;
            if w.pending_prefill.is_some() {
                if job.cancelled() || queue.stopping() {
                    return Ok(Scheduled::Native(Command::Finish(Finish::Cancelled)));
                }
                if std::mem::take(prefill_turn_done) && stepping { return Ok(Scheduled::Batched(false, skip)); }
                return Ok(Scheduled::Prefill);
            }
            let next = w.active_command(job.cancelled() || queue.stopping())?;
            if matches!(next, Command::Finish(_)) { return Ok(Scheduled::Native(next)); }
            // Lookahead admission: region 0's conversation reached the first region's rows while regions are occupied.
            // It sits out while they step; with nothing stepping it parks as an image (the regions' finished
            // conversations can then move back) and waits for the regions to empty.
            if w.active_at_boundary()? {
                if stepping { return Ok(Scheduled::Batched(false, skip)); }
                if !refused && images.len() < 2 { return Ok(Scheduled::Park(None)); }
                if let Some(id) = w.finished_peer() { return Ok(Scheduled::PeerFinish(id, 2)); }
                std::thread::sleep(Duration::from_millis(100));
                return Ok(Scheduled::OutputWait);
            }
            let boundary = w.boundary_peer()?;
            // A waiting conversation takes region 0 once this one moves to a free region; when only
            // finished conversations (waiting for region 0) hold the regions, the oldest is dropped.
            // A region conversation at its region's end waits for region 0: nothing new starts or moves meanwhile.
            if !ready.is_empty() && boundary.is_none() {
                // Room is made in region 0 for the conversation next in line only if it can share the owner
                // with the regions: until Nov 14 2026 a short request was moved to a region to resume a paused
                // conversation of ~950,000 tokens beside it (release stress panel, s1582).
                let next = next_for_region0(ready.iter().map(|(_, j)| j.background), held);
                let next_fits = next.is_none_or(|i| reach_fits(&ready[i], images, capacity));
                if next_fits {
                    if w.free_region()?.is_none() {
                        if let Some(id) = w.finished_peer() { return Ok(Scheduled::PeerFinish(id, 2)); }
                    }
                    if w.can_detach()? {
                        if let Some(region) = w.free_region()? { return Ok(Scheduled::Detach(u8::try_from(region)?)); }
                    }
                }
                // It cannot move, or the one in line cannot share the owner: take turns in region 0 through a
                // parked image. Not for one that waits for the regions to empty: it could not start.
                let next_is_new = next.is_some_and(|i| ready[i].0.is_none());
                let turn_over = super::peer_turn_due(w.emitted, quantum_start) || first_turn_over(w.emitted, quantum_start, next_is_new);
                if !waits_for_empty_regions(next_fits, w.regions_occupied()) && !refused && images.len() < 2
                    && yields_region0(turn_over, held && job.background, next.is_some()) {
                    return Ok(Scheduled::Park(None));
                }
            }
            let active_held = held && job.background;
            let active_behind = job.output_backpressured() || active_held;
            if stepping { return Ok(Scheduled::Batched(!active_behind, skip)); }
            if active_behind {
                std::thread::sleep(Duration::from_millis(100));
                return Ok(if active_held { Scheduled::Hold } else { Scheduled::OutputWait });
            }
            return Ok(Scheduled::Native(Command::Step));
        }
        if queue.stopping() && !peers {
            ensure!(ready.is_empty(), "shutdown queue not drained");
            return Ok(Scheduled::Native(Command::Shutdown));
        }
        // Region 0 is free: the waiting conversations take it in the order of `ready`, paused and new
        // ones alike (background ones not while the router holds background work). A conversation
        // that was just paused went to the back of `ready`, so the one it was paused for goes first
        // and two paused conversations alternate. Until Nov 14 2026 any paused conversation resumed
        // before a new one could start: a request that arrived during the answer of a conversation
        // too long to move saw that conversation paused and resumed every 64 tokens and started only
        // when the answer was complete (the release stress panel's "foreground did not preempt
        // background"). A paused conversation's restore replaces region 0's state, so no finished
        // region conversation moves back just before it; one moves back before a new conversation
        // starts, and when nothing waits. Else the region conversations step.
        // One that can reach past a serving region waits while a region is occupied: the finished region
        // conversations move back (one below, each turn of this loop), the decoding ones step to their end.
        // Lookahead admission: a region conversation at its region's end takes region 0 first (finished ones move
        // back before it, other generating ones step on to their ends); nothing else starts ahead of it.
        if let Some(id) = w.boundary_peer()? {
            if let Some(finished) = w.finished_peer() { return Ok(Scheduled::SavePeer(finished)); }
            let others = w.generating_peers() > 1;
            if others && stepping { return Ok(Scheduled::Batched(false, skip)); }
            if !others || (!refused && images.len() < 2) { return Ok(Scheduled::AttachPeer(id)); }
            std::thread::sleep(Duration::from_millis(100));
            return Ok(Scheduled::Native(Command::Idle));
        }
        let next = next_for_region0(ready.iter().map(|(_, j)| j.background), held);
        let next_fits = next.is_none_or(|i| reach_fits(&ready[i], images, capacity));
        let next = next.filter(|_| !waits_for_empty_regions(next_fits, w.regions_occupied()));
        if let Some(i) = next.filter(|&i| ready[i].0.is_some()) {
            let (saved, job) = ready.remove(i).unwrap();
            *active = Some(job);
            return Ok(Scheduled::Resume(saved.unwrap()));
        }
        if let Some(id) = w.finished_peer() { return Ok(Scheduled::SavePeer(id)); }
        if let Some((_, mut job)) = next.and_then(|i| ready.remove(i)) {
            let id = *next_id;
            *next_id = next_id.checked_add(1).ok_or_else(|| anyhow::anyhow!("HTTP request counter exhausted"))?;
            if !job.images.is_empty() {
                w.media.as_mut().ok_or_else(|| anyhow::anyhow!("image request without native vision"))?
                    .set_job_inputs(&mut job)?;
            }
            let command = Command::Begin { images: std::mem::take(&mut job.images), id, seed: job.seed,
                greedy: job.greedy, limit: job.limit, tokens: std::mem::take(&mut job.tokens) };
            *active = Some(job);
            return Ok(Scheduled::Native(command));
        }
        if stepping { return Ok(Scheduled::Batched(false, skip)); }
        if peers {
            // Every region conversation's reader is behind: wait for them.
            std::thread::sleep(Duration::from_millis(100));
            return Ok(Scheduled::Native(Command::Idle));
        }
        if let Some(job) = queue.next_fifo(w.tier_idle_wait(Duration::from_millis(500)))? { ready.push_back((None, job)); continue; }
        return Ok(Scheduled::Native(Command::Idle));
    }
}

#[cfg(test)]
mod tests {
    use super::{first_turn_over, next_for_region0, waits_for_empty_regions, yields_region0, FIRST_TURN};
    #[test]
    fn a_conversation_in_its_first_run_yields_early_to_a_new_one_only() {
        // Begun, never paused (its turn counts from 0), a new conversation next in line: after FIRST_TURN tokens.
        assert!(!first_turn_over(FIRST_TURN - 1, 0, true) && first_turn_over(FIRST_TURN, 0, true));
        // A paused conversation next in line: the whole 64-token turn, as two paused conversations alternate.
        assert!(!first_turn_over(FIRST_TURN, 0, false) && !first_turn_over(63, 0, false));
        // A resumed conversation (its turn counts from its position at the resume) keeps its whole turn, so
        // newcomers cannot starve it.
        assert!(!first_turn_over(40 + FIRST_TURN, 40, true) && !first_turn_over(40 + 63, 40, true));
    }
    #[test]
    fn a_held_background_conversation_that_cannot_move_yields_at_once() {
        // Mid-turn, not held: it keeps region 0 until its turn is over.
        assert!(!yields_region0(false, false, true) && yields_region0(true, false, true));
        // Held by the router with a runnable conversation in line (next_for_region0 skips held ones): at once.
        assert!(yields_region0(false, true, true));
        // Held, and only held conversations in line: a pause would start nothing.
        assert!(!yields_region0(false, true, false));
    }
    use crate::resident_worker::batched_serving::fits_one_region;
    #[test]
    fn lookahead_admission_and_region_ends() {
        use crate::resident_worker::batched_serving::{at_region_end, fits_one_region};
        let capacity = 262_144;
        // Admission by reach (config::region_reach_with): a 16K conversation with max_tokens 262,144 fits a region
        // with a 16,384-position lookahead, not without one.
        assert!(!fits_one_region(crate::config::region_reach_with(16_000, 262_144, 0), capacity));
        assert!(fits_one_region(crate::config::region_reach_with(16_000, 262_144, 16_384), capacity));
        assert_eq!(crate::config::region_reach_with(16_000, 100, 16_384), 16_100);
        // The end: one more step (at most 16 tokens) plus the 64-position margin must fit.
        assert!(!at_region_end(capacity - 64 - 16, 1_000, capacity) && at_region_end(capacity - 64 - 15, 1_000, capacity));
        // A conversation that finishes within the margin may still step.
        assert!(!at_region_end(capacity - 64 - 5, 5, capacity) && at_region_end(capacity - 64 - 5, 6, capacity));
        // Admission with any lookahead of at least 16 leaves room for a step.
        for lookahead in [256, 16_384] {
            let position = capacity - 64 - lookahead;
            assert!(fits_one_region(crate::config::region_reach_with(position, 1 << 20, lookahead), capacity));
            assert!(!at_region_end(position, 1 << 20, capacity));
        }
    }
    #[test]
    fn a_conversation_longer_than_a_region_waits_for_empty_regions() {
        // One region of a 1,048,576-token context split four ways holds 262,144 positions.
        let capacity = 262144;
        // The stress panel's pair: a 950,000-token conversation with 512 tokens to produce, and a short request.
        assert!(!fits_one_region(950_000 + 512, capacity));
        assert!(fits_one_region(1_200 + 64, capacity));
        // The last 64 positions of a region are margin.
        assert!(fits_one_region(capacity - 64, capacity) && !fits_one_region(capacity - 63, capacity));
        assert!(!fits_one_region(usize::MAX, capacity));
        // Full serving regions (Nov 18 2026): any conversation fits whatever its max_tokens, since its extent never
        // passes the context; equal shares keep their 64-position margin.
        use crate::resident_worker::batched_serving::fits_one_region_with;
        let full = crate::config::MODEL_CONTEXT;
        assert!(fits_one_region_with(full, full, true) && !fits_one_region_with(full + 1, full, true));
        assert!(!fits_one_region_with(full, full, false) && fits_one_region_with(full - 64, full, false));
        // It waits only while a region is occupied; with the regions empty it takes region 0 like any other.
        assert!(waits_for_empty_regions(false, true));
        assert!(!waits_for_empty_regions(false, false));
        assert!(!waits_for_empty_regions(true, true) && !waits_for_empty_regions(true, false));
    }
    #[test]
    fn the_first_waiting_conversation_takes_region0_whether_paused_or_new() {
        // [a new foreground request, the background conversation just paused for it]: the new one starts.
        assert_eq!(next_for_region0([false, true].into_iter(), false), Some(0));
        // Two paused conversations alternate: the one paused earlier is in front.
        assert_eq!(next_for_region0([false, false].into_iter(), false), Some(0));
        // While the router holds background work, background conversations (paused or new) wait.
        assert_eq!(next_for_region0([true, true, false].into_iter(), true), Some(2));
        assert_eq!(next_for_region0([true, true].into_iter(), true), None);
        assert_eq!(next_for_region0([true, false].into_iter(), false), Some(0));
        assert_eq!(next_for_region0(std::iter::empty(), false), None);
    }
}
