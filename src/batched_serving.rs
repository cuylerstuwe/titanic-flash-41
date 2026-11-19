//! Batched decode across serving regions (CANADA_SCRATCH_BATCHED_DECODE=1, with
//! CANADA_SCRATCH_BATCHED_REGIONS=R >= 2 and CANADA_SCRATCH_RESET_KEEP_CACHES=1).
//!
//! Region 0 keeps today's single-conversation path: prefill, ordinary decode steps and the
//! session-cache image at the end of a turn. When another prompt is waiting, a decoding region-0
//! conversation moves into a free region 1..R-1 (Target::relocate_session) with its drafter,
//! request history and target history, and region 0 is reset for the next prompt. While any
//! conversation lives in a region, decode steps are batched: one verification over every decoding
//! conversation (the 16-row owner for two or more, rows / conversations rows each, at most six).
//! The region-0 conversation is part 0 of that step and its commit also advances the owner's own
//! history and encoder tail, exactly as an ordinary step does.
//!
//! A conversation that ends in a region (stop or length) gets its finish at once and waits there
//! until region 0 is free; it then moves back (with its encoder tail, which batched commits keep
//! current) and becomes region 0's resident finished conversation, exactly as if it had finished
//! there: its next turn continues from the live state, or the next prompt saves its cache image.
//! When a region is needed and only finished conversations hold them, the oldest is dropped
//! without an image. A conversation with media inputs does not move.
use super::*;
use crate::dspark_generation::BatchedSession;
use crate::target::SessionBase;

/// A conversation that can reach `extent` positions stays inside one serving region of `capacity` positions
/// (64 positions of margin: a step's rows and the page a region's last rows share). Full regions
/// (config::full_regions) need no margin: every region is the whole model context, starts on a page boundary and
/// no conversation's rows pass the context, so any conversation fits.
pub(crate) fn fits_one_region(extent: usize, capacity: usize) -> bool {
    fits_one_region_with(extent, capacity, crate::config::full_regions().unwrap_or(false))
}
pub(crate) fn fits_one_region_with(extent: usize, capacity: usize, full: bool) -> bool {
    extent.saturating_add(if full { 0 } else { 64 }) <= capacity
}
/// Tokens one batched step can add to a conversation at most (a step verifies at most six rows of a conversation;
/// sixteen keeps a margin): with lookahead admission (config::region_lookahead) a conversation closer than this to
/// its region's end sits out.
const BOUNDARY_STEP: usize = 16;
/// A generating conversation at `position` with `remaining` tokens to produce cannot take another batched step
/// inside a region of `capacity` positions (only reachable with lookahead admission).
pub(crate) fn at_region_end(position: usize, remaining: usize, capacity: usize) -> bool {
    !fits_one_region(position.saturating_add(remaining.min(BOUNDARY_STEP)), capacity)
}
pub(crate) fn enabled() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_BATCHED_DECODE") {
        Ok(v) => { ensure!(v == "0" || v == "1", "invalid batched decode policy"); Ok(v == "1") }
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Owner rows for a two-conversation step (CANADA_SCRATCH_BATCHED_PAIR_ROWS: 6, the default, or 16).
fn pair_rows() -> Result<usize> {
    match std::env::var("CANADA_SCRATCH_BATCHED_PAIR_ROWS") {
        Ok(v) => { let n: usize = v.parse()?; ensure!(n == 6 || n == 16, "batched pair rows must be 6 or 16"); Ok(n) }
        Err(std::env::VarError::NotPresent) => Ok(6),
        Err(e) => Err(e.into()),
    }
}

/// Startup validation of the batched decode serving profile.
pub(crate) fn validate() -> Result<()> {
    if !enabled()? {
        ensure!(!crate::config::full_regions()?, "full serving regions need batched decode");
        return Ok(());
    }
    ensure!(crate::config::batched_regions()? >= 2 && crate::config::reset_keeps_caches()?
        && crate::config::batch_screen_sessions()? == 1,
        "batched decode needs CANADA_SCRATCH_BATCHED_REGIONS >= 2 and the kept-cache reset");
    ensure!(std::env::var("CANADA_SCRATCH_SELECTED_GROUPED16_STORAGE_SCREEN").as_deref() != Ok("1"),
        "batched decode needs the 16-row owner's dense index scorer (selected-only M16 storage off)");
    ensure!(std::env::var("CANADA_SCRATCH_COMPACT_STATE_IMAGE").as_deref() == Ok("1")
        && std::env::var("CANADA_SCRATCH_HTTP_INTERLEAVE").as_deref() == Ok("1"),
        "batched decode needs compact (region-0) state images and the HTTP interleave scheduler");
    // The warm-up runs every batched pattern at the regions' empty frontiers before Ready, before the image pool
    // exists: with elastic regions it maps regions 1.. from a temporary pool released before the real one
    // (resident_worker::warm_before_ready).
    crate::dspark_generation::batched_gate_from_env()?;
    ensure!(std::env::var("CANADA_SCRATCH_RESIDENT_BANKS").as_deref() != Ok("1")
        && std::env::var("CANADA_SCRATCH_STATE_EXCHANGE").as_deref() != Ok("1"),
        "batched decode replaces resident banks and state exchange");
    Ok(())
}

/// Batched step timing by conversations per step (index 1..=4), printed every 100 steps and when a
/// region conversation ends: step wall, its batched-verification part, the agreement, the period
/// between consecutive batched steps (scheduling and control included), tokens and graph captures.
#[derive(Default)]
pub(crate) struct Timing {
    steps: [usize; 5], tokens: [usize; 5], step_ms: [f64; 5], core_ms: [f64; 5], agree_ms: [f64; 5],
    period_ms: [f64; 5], periods: [usize; 5], last_end: Option<Instant>, since_report: usize,
    /// Steps that captured a new graph (their count and wall), the other steps' walls (the most
    /// recent 2048 to 4096), and the summed mean history position of each step's conversations.
    capture_steps: [usize; 5], capture_ms: [f64; 5], steady: [Vec<f64>; 5], positions: [f64; 5],
}
impl Timing {
    fn report(&mut self, captures: usize) {
        if self.since_report == 0 { return; }
        let mut parts = Vec::new();
        for n in 1..=4 {
            let s = self.steps[n];
            if s == 0 { continue; }
            let f = s as f64;
            let mut steady = self.steady[n].clone();
            steady.sort_by(|a, b| a.total_cmp(b));
            let pick = |q: f64| if steady.is_empty() { 0. } else { steady[((steady.len() - 1) as f64 * q).round() as usize] };
            parts.push(format!("{n} conversations: {s} steps, {:.2} tokens/step, step {:.2} ms (verify+draft {:.2}, agree {:.2}), period {:.2} ms, {:.1} tok/s; {} steps captured graphs ({:.0} ms); other steps median {:.2} ms, p90 {:.2}, max {:.2}; mean position {:.0}",
                self.tokens[n] as f64 / f, self.step_ms[n] / f, self.core_ms[n] / f, self.agree_ms[n] / f,
                if self.periods[n] > 0 { self.period_ms[n] / self.periods[n] as f64 } else { 0. },
                if self.periods[n] > 0 { self.tokens[n] as f64 / (self.period_ms[n] / 1000.) * self.periods[n] as f64 / f } else { 0. },
                self.capture_steps[n], self.capture_ms[n], pick(0.5), pick(0.9), pick(1.), self.positions[n] / f));
        }
        let phases = crate::dspark_generation::take_batched_phases();
        let phases = if phases.as_array().is_some_and(|a| !a.is_empty()) { format!("; rows (and synced phases) since the last report {phases}") } else { String::new() };
        let forms = if crate::config::draft_head_subset().unwrap_or(0) != 0 || crate::config::draft_head_pool().unwrap_or(false) {
            let take = |i: usize| crate::dspark::BATCHED_HEAD_FORMS[i].swap(0, std::sync::atomic::Ordering::Relaxed);
            format!("; batched-body draft head forms since the last report: whole {} subset {} (pooled passes {})", take(0), take(1),
                crate::dspark::BATCHED_HEAD_POOLED.swap(0, std::sync::atomic::Ordering::Relaxed))
        } else { String::new() };
        eprintln!("batched decode timing: {}; owner graph captures {captures}{forms}{phases}", parts.join("; "));
        // CANADA_SCRATCH_BATCHED_TRACE=1: the steps' gate trace since the last report (diagnostic).
        let trace = crate::dspark_generation::take_batched_trace();
        if !trace.is_empty() { eprintln!("batched trace {}", serde_json::Value::Array(trace)); }
        self.since_report = 0;
    }
}

/// One conversation decoding in serving region `region` (1..R-1).
pub(crate) struct Peer {
    pub(crate) id: u64,
    pub(crate) region: usize,
    base: SessionBase,
    session: GreedySession,
    history: crate::hash_history::History,
    resident: ResidentHistory,
    pub(crate) emitted: usize,
    pub(crate) limit: usize,
    /// Generation ended (stop or length; its client already has its finish): waits in its region
    /// for region 0 (save_peer).
    pub(crate) finished: Option<Finish>,
}
impl Peer {
    fn position(&self) -> usize { self.history.position as usize }
}

impl Worker {
    /// At load, before the first drafter attaches: one spare drafter per extra region, so a
    /// conversation that moves out of region 0 takes its own drafter along and region 0 still has one.
    pub(crate) fn prepare_batched_drafters(&mut self, kernels: &Path) -> Result<Value> {
        if !enabled()? { return Ok(Value::Null); }
        validate()?;
        let regions = crate::config::batched_regions()?;
        let before = self.cuda.admission_memory()?;
        let prepared = self.prepared_session.as_ref().context("batched decode needs the prepared drafter")?;
        for _ in 1..regions {
            // Forks share the prepared drafter's image identity: cached session images restore into
            // whichever drafter region 0 holds.
            self.spare_sessions.push(prepared.serving_fork(&self.verify, &self.core, &self.stream, kernels)?);
        }
        let after = self.cuda.admission_memory()?;
        ensure!(after.admitted_available >= crate::memory::floor_bytes(), "batched drafters consume the runtime reserve");
        Ok(json!({"regions":regions,"spare_drafters":self.spare_sessions.len(),
            "drafter_bytes":before.admitted_available.saturating_sub(after.admitted_available)}))
    }
    /// A serving region holds a conversation: decoding, or finished and waiting to move back to region 0.
    pub(crate) fn regions_occupied(&self) -> bool { !self.peers.is_empty() }
    /// Positions one serving region holds: with full regions a whole conversation, so every conversation fits
    /// whatever its `max_tokens` (Nov 18 2026), else an equal share of the one context.
    pub(crate) fn region_capacity(&self) -> Result<usize> {
        let capacity = SessionBase::serving_capacity(self.verify.context_capacity(), crate::config::batched_regions()?)?;
        Ok(crate::config::region_capacity_test()?.map_or(capacity, |test| test.min(capacity)))
    }
    /// The first generating region conversation that has reached its region's end (lookahead admission): it sits out
    /// and moves back to region 0 once it is the only conversation left (attach_peer).
    pub(crate) fn boundary_peer(&self) -> Result<Option<u64>> {
        let capacity = self.region_capacity()?;
        Ok(self.peers.iter().filter(|p| p.finished.is_none()
            && at_region_end(p.position(), p.limit.saturating_sub(p.emitted), capacity)).map(|p| p.id).min())
    }
    pub(crate) fn peer_at_boundary(&self, id: u64) -> Result<bool> {
        let capacity = self.region_capacity()?;
        let peer = self.peers.iter().find(|p| p.id == id).context("unknown region conversation")?;
        Ok(peer.finished.is_none() && at_region_end(peer.position(), peer.limit.saturating_sub(peer.emitted), capacity))
    }
    /// Region 0's generating conversation has reached the first region's rows while regions are occupied: it sits out
    /// until they are empty (lookahead admission).
    pub(crate) fn active_at_boundary(&self) -> Result<bool> {
        Ok(!self.peers.is_empty() && self.current.is_some() && self.pending_prefill.is_none()
            && at_region_end(self.verify.position() as usize, self.limit.saturating_sub(self.emitted), self.region_capacity()?))
    }
    /// Region conversations still generating (not finished).
    pub(crate) fn generating_peers(&self) -> usize { self.peers.iter().filter(|p| p.finished.is_none()).count() }
    /// Region 0 shares the owner's compressed address space with the serving regions: region k's rows start at
    /// k times a region's capacity, and region 0's at zero, without an upper bound of their own. A region-0
    /// conversation that reaches past one region's capacity therefore covers the regions' rows, and so does its
    /// image when it is restored: it may only start, resume or continue while every region is empty. `extent`:
    /// the positions it can reach (prompt or frontier, plus the tokens it may still produce); the margin is
    /// can_detach's. The scheduler (batched_admission) keeps such a conversation waiting until the regions are
    /// empty and takes turns in region 0 with it through paused images; this is the same rule, checked where
    /// the rows are written.
    pub(crate) fn region0_admits(&self, extent: usize) -> Result<bool> {
        Ok(self.peers.is_empty() || !enabled()? || fits_one_region(extent, self.region_capacity()?))
    }
    pub(crate) fn free_region(&self) -> Result<Option<usize>> {
        let regions = crate::config::batched_regions()?;
        Ok((1..regions).find(|&r| !self.peers.iter().any(|p| p.region == r)))
    }
    /// The decoding region-0 conversation can move: past its prefill, generating, no media, room
    /// for the rest of its output in a region, and a spare drafter for region 0.
    pub(crate) fn can_detach(&self) -> Result<bool> {
        let Some(_) = self.current else { return Ok(false) };
        // Media: the bindings of a prefill still running (active_media), and any image in the conversation's
        // history. The bindings end with the prefill (publish_begin), so until Nov 14 2026 a conversation with
        // images moved once it was decoding, against this module's rule and untested; it now stays in region 0
        // and takes turns there.
        if self.pending_prefill.is_some() || self.prefill_cancelled || self.active_media.is_some()
            || !self.history.media().is_empty() || self.spare_sessions.is_empty() || self.free_region()?.is_none() || self.emitted >= self.limit
            || self.history.pending().is_none_or(|a| a == self.cfg.eos_token_id)
            || !self.verify.profile_is_current() {
            return Ok(false);
        }
        // Elastic regions: the free region's spans must fit the image pool's free chunks now (rank 0 decides; every rank
        // confirms in detach_active).
        // Elastic regions: detach_active makes room by evicting images when it must; a conversation for which even that
        // failed takes turns until a region frees up (elastic_declined is cleared then) or another one is current.
        if crate::config::elastic_regions()? && self.elastic_declined.is_some() && self.elastic_declined == self.current {
            return Ok(false);
        }
        Ok(fits_one_region(crate::config::region_reach(self.verify.position() as usize, self.limit - self.emitted)?,
            self.region_capacity()?))
    }
    /// Move the decoding region-0 conversation into `region` and reset region 0 for the next prompt.
    pub(crate) fn detach_active(&mut self, region: usize) -> Result<Value> {
        let began = Instant::now();
        ensure!(enabled()? && self.can_detach()? && self.free_region()? .is_some()
            && !self.peers.iter().any(|p| p.region == region)
            && (1..crate::config::batched_regions()?).contains(&region), "batched detach eligibility");
        // Elastic regions: map the region's cache spans first (the kept images below also take pool chunks). Every
        // rank must have the room rank 0 saw; a rank that does not fails all of them alike instead of diverging.
        let elastic = crate::config::elastic_regions()?;
        let mut evicted = 0;
        if elastic {
            let (room, gave_way) = self.reclaim_for_region(region)?;
            evicted = gave_way;
            if !room {
                // Nothing more may give way: this conversation stays in region 0 and takes turns, as without regions.
                self.elastic_declined = self.current;
                return Ok(json!({"type":"detach_declined","request_id":self.current,"region":region,
                    "evicted_images":evicted,"seconds":began.elapsed().as_secs_f64()}));
            }
            ensure!(self.map_region(region)?, "elastic region map lost its room");
        }
        let id = self.current.unwrap();
        // Cached images of this conversation are sparse: their prefix lives in region 0's bank, on the lineage the
        // reset below ends. Copy out the newest one (the conversation's next turn restores it) and every
        // shared-prefix image (other conversations restore those), as a newcomer's prompt does for a conversation
        // it displaces in region 0. Without this a turn that follows a batched one started from an older
        // standalone image (s1503 mi3: 276-343 fresh rows a join where pico prefilled 31).
        let kept = if crate::config::batched_detach_keeps_images()? {
            let current = self.history.committed().to_vec();
            self.materialize_displaced_checkpoint(&[], &current, None)?
        } else { Value::Null };
        let regions = crate::config::batched_regions()?;
        let context = self.verify.context_capacity();
        let base = SessionBase::serving_region(region, regions, context, self.verify.swa_layout_for_probe())?;
        let position = self.verify.position() as usize;
        let moved = self.verify.relocate_session(SessionBase::default(), base, position, &self.stream)?;
        let history = self.verify.history_for_probe();
        let mut fresh = ResidentHistory::new(self.large.context_capacity(), self.cfg.eos_token_id)?;
        let mut resident = std::mem::replace(&mut self.history, ResidentHistory::new(self.large.context_capacity(), self.cfg.eos_token_id)?);
        fresh.synchronize_request_ids(&mut resident)?;
        self.history = fresh;
        let mut spare = self.spare_sessions.pop().context("spare drafter")?;
        // Region 0's next conversation may arrive as an image (a paused one resuming, a cached one's turn).
        spare.clear_for_region0(&self.stream)?;
        let session = self.session.replace(spare).context("active drafter")?;
        self.peers.push(Peer { id, region, base, session, history, resident, emitted: self.emitted, limit: self.limit, finished: None });
        self.current = None;
        self.emitted = 0;
        self.limit = 0;
        // Region 0 starts clean: a new lineage (with serving regions the reset clears region 0 only).
        self.target_lineage = self.target_lineage.checked_add(1).context("target lineage exhausted")?;
        self.verify.reset(self.target_lineage, &self.comm, &self.stream)?;
        self.control.agree(&serde_json::to_vec(&json!({"batched_detach_v1":[id, region as u64, position as u64]}))?)?;
        Ok(json!({"type":"detach","request_id":id,"region":region,"position":position,"relocation":moved,
            "kept_images":kept,"peers":self.peers.len(),"elastic":elastic,"evicted_images":evicted,
            "seconds":began.elapsed().as_secs_f64()}))
    }
    /// Elastic regions: evict cached images in the cache's pressure order (as the displaced reclaim does), one at a
    /// time, until every rank's image pool can map serving region `region`. Ranks agree on each victim, so all evict
    /// the same entries. Returns (room, images evicted); no room once nothing evictable is left.
    /// Elastic regions: whether the image pool's free chunks can map `region`'s cache spans and, with full regions, the
    /// 16-row owner's key rows of that region.
    /// Elastic regions: the regions mapping `region` needs, 1..=region minus those already mapped. Every region up to the
    /// highest occupied one stays mapped (release_regions): an index scorer bounds each block of query rows by the union
    /// of its rows' key ranges, so a block holding rows of region 0 and region 3 also reads regions 1 and 2 (Nov 18,
    /// fqa4: an MMU read fault in region 1's index key rows after its conversation left while region 3's stayed).
    fn regions_to_map(&self, region: usize) -> Vec<usize> {
        (1..=region).filter(|&j| self.elastic_mapped & (1u8 << j) == 0).collect()
    }
    fn region_room(&self, region: usize) -> Result<bool> {
        let mut need = 0;
        for j in self.regions_to_map(region) {
            need += self.verify.serving_region_missing_chunks(j)?;
            for owner in self.region_key_owners() { need += owner.region_keys_missing_chunks(j)?; }
        }
        Ok(need <= self.cuda.image_pool_free_chunks().unwrap_or(0))
    }
    /// Every owner whose index key matrices may span the regions: the 6-row, 16-row and 32-row owners (an owner without
    /// region matrices maps nothing).
    fn region_key_owners(&self) -> Vec<&crate::target::Target> {
        [Some(&self.verify), self.long_verify.as_ref(), self.copy32_verify.as_ref()].into_iter().flatten().collect()
    }
    /// Map one region's cache spans and every owner's region key rows, all or nothing (Ok(false) without room).
    fn map_one_region(&self, region: usize) -> Result<bool> {
        if !self.verify.map_serving_region(region)? { return Ok(false); }
        let owners = self.region_key_owners();
        for (i, owner) in owners.iter().enumerate() {
            if !owner.map_region_keys(region)? {
                for done in &owners[..i] { done.unmap_region_keys(region)?; }
                self.verify.unmap_serving_region(region)?;
                return Ok(false);
            }
        }
        Ok(true)
    }
    fn unmap_one_region(&self, region: usize) -> Result<()> {
        for owner in self.region_key_owners() { owner.unmap_region_keys(region)?; }
        self.verify.unmap_serving_region(region)
    }
    /// Map `region` and every unmapped region below it, all or nothing (Ok(false), nothing changed, without room).
    fn map_region(&mut self, region: usize) -> Result<bool> {
        if !self.region_room(region)? { return Ok(false); }
        let mut done = Vec::new();
        for j in self.regions_to_map(region) {
            if !self.map_one_region(j)? {
                for &d in done.iter().rev() { self.unmap_one_region(d)?; self.elastic_mapped &= !(1u8 << d); }
                return Ok(false);
            }
            self.elastic_mapped |= 1u8 << j;
            done.push(j);
        }
        Ok(true)
    }
    /// After a region conversation left (its peer already removed): unmap every mapped region above the highest region
    /// still occupied, highest first. A vacated region below an occupied one stays mapped (see regions_to_map).
    fn release_regions(&mut self) -> Result<()> {
        let high = self.peers.iter().map(|p| p.region).max().unwrap_or(0);
        for j in (high + 1..crate::config::batched_regions()?).rev() {
            if self.elastic_mapped & (1u8 << j) != 0 {
                self.unmap_one_region(j)?;
                self.elastic_mapped &= !(1u8 << j);
            }
        }
        Ok(())
    }
    fn reclaim_for_region(&mut self, region: usize) -> Result<(bool, usize)> {
        let tokens = self.verify.position() as usize;
        let mut evicted = 0;
        loop {
            let room = self.region_room(region)?;
            if self.control.all_admit(room)? { return Ok((true, evicted)); }
            let entries = self.cached_sessions.iter().map(|c| (c.prefix.len(), c.charge)).collect::<Vec<_>>();
            let optional = self.optional_entries(tokens);
            let superseded = vec![false; entries.len()];
            let victim = pressure_cache_victim(&entries, &optional, &superseded, tokens, self.image_entry_limit(),
                self.pressure_long_first(tokens), &[]);
            self.control.agree(&serde_json::to_vec(&("region-reclaim-v1", region, victim, &entries))?)?;
            let Some(victim) = victim else { return Ok((false, evicted)) };
            self.stream.synchronize()?;
            self.evict_cached(victim);
            evicted += 1;
        }
    }
    /// The oldest finished region conversation waiting for region 0.
    pub(crate) fn finished_peer(&self) -> Option<u64> {
        self.peers.iter().filter(|p| p.finished.is_some()).min_by_key(|p| p.id).map(|p| p.id)
    }
    /// Why a decoding region conversation ends now, if it does.
    pub(crate) fn peer_finish_reason(&self, id: u64, cancelled: bool) -> Result<Option<Finish>> {
        let peer = self.peers.iter().find(|p| p.id == id).context("unknown region conversation")?;
        if peer.finished.is_some() { return Ok(None); }
        Ok(if peer.resident.pending() == Some(self.cfg.eos_token_id) { Some(Finish::Stop) }
            else if peer.emitted >= peer.limit { Some(Finish::Length) }
            else if cancelled { Some(Finish::Cancelled) }
            else { None })
    }
    /// End a region conversation's generation. Stop and length keep it in its region until region 0
    /// takes it back (save_peer); a cancelled one, or an already finished one being dropped to free
    /// its region, releases the region now without a cache image.
    pub(crate) fn finish_peer(&mut self, id: u64, reason: Finish) -> Result<Value> {
        let index = self.peers.iter().position(|p| p.id == id).context("unknown region conversation")?;
        self.stream.synchronize()?;
        let label = match reason { Finish::Stop => "stop", Finish::Length => "length", Finish::Cancelled => "cancelled" };
        let (region, emitted, frontier) = { let p = &self.peers[index]; (p.region, p.emitted, p.position()) };
        if let Some(earlier) = self.peers[index].finished {
            ensure!(reason == Finish::Cancelled, "a finished region conversation can only be dropped");
            let mut peer = self.peers.remove(index);
            peer.resident.finish(earlier)?;
            self.history.synchronize_request_ids(&mut peer.resident)?;
            self.spare_sessions.push(peer.session);
            if crate::config::elastic_regions()? { self.release_regions()?; self.elastic_declined = None; }
            self.control.agree(&serde_json::to_vec(&json!({"batched_drop_v1":[id, region as u64]}))?)?;
            return Ok(json!({"type":"peer_dropped","request_id":id,"region":region,"frontier":frontier}));
        }
        let captures = self.long_verify.as_ref().map_or(0, |t| t.graph_capture_count()) + self.verify.graph_capture_count();
        if self.comm.rank() == 0 { self.batched_timing.report(captures); }
        if reason == Finish::Cancelled {
            let mut peer = self.peers.remove(index);
            peer.resident.finish(reason)?;
            self.history.synchronize_request_ids(&mut peer.resident)?;
            self.spare_sessions.push(peer.session);
            if crate::config::elastic_regions()? { self.release_regions()?; self.elastic_declined = None; }
        } else {
            self.peers[index].finished = Some(reason);
        }
        self.control.agree(&serde_json::to_vec(&json!({"batched_finish_v1":[id, region as u64, u64::from(reason == Finish::Cancelled)]}))?)?;
        Ok(json!({"type":"finish","peer":true,"request_id":id,"reason":label,"region":region,
            "completion_tokens":emitted,"frontier":frontier,"awaits_region0":reason != Finish::Cancelled}))
    }
    /// Region 0 is free: the finished region conversation `id` moves back and becomes region 0's
    /// resident finished conversation. A resident finished conversation already there first keeps
    /// its cache image, as a newcomer's prompt would have saved it.
    pub(crate) fn save_peer(&mut self, id: u64) -> Result<Value> {
        let began = Instant::now();
        ensure!(enabled()? && self.current.is_none() && self.pending_prefill.is_none(), "save needs a free region 0");
        let index = self.peers.iter().position(|p| p.id == id && p.finished.is_some()).context("unknown finished region conversation")?;
        let mut displaced = Value::Null;
        if self.session.is_some() && !self.history.committed().is_empty() && self.session_cache_budget > 0 {
            let prefix = self.history.committed().to_vec();
            let pending = self.history.pending().context("resident conversation lacks its pending anchor")?;
            let media = self.history.media().through(prefix.len());
            displaced = self.retain_cached_prefix_reserved(&prefix, pending, media, 0, 0, self.displaced_snapshot)?;
            // The resident conversation's sparse images end with region 0's lineage below, as in detach_active.
            if crate::config::batched_detach_keeps_images()? {
                let kept = self.materialize_displaced_checkpoint(&[], &prefix, None)?;
                if !kept.is_null() && displaced.is_object() { displaced["kept_images"] = kept; }
            }
        }
        let mut peer = self.peers.remove(index);
        let position = peer.position();
        self.batched_timing.last_end = None;
        self.target_lineage = self.target_lineage.checked_add(1).context("target lineage exhausted")?;
        self.verify.reset(self.target_lineage, &self.comm, &self.stream)?;
        let moved = self.verify.relocate_session(peer.base, SessionBase::default(), position, &self.stream)?;
        if crate::config::elastic_regions()? { self.release_regions()?; self.elastic_declined = None; }
        self.verify.adopt_moved_history(peer.history.clone())?;
        peer.resident.finish(peer.finished.unwrap())?;
        peer.resident.synchronize_request_ids(&mut self.history)?;
        self.history = peer.resident;
        let spare = self.session.replace(peer.session);
        if let Some(spare) = spare { self.spare_sessions.push(spare); }
        self.emitted = 0;
        self.limit = 0;
        self.control.agree(&serde_json::to_vec(&json!({"batched_save_v1":[id, peer.region as u64, position as u64]}))?)?;
        Ok(json!({"type":"peer_saved","request_id":id,"region":peer.region,"position":position,
            "relocation":moved,"displaced_resident":displaced,"seconds":began.elapsed().as_secs_f64()}))
    }
    /// Lookahead admission (config::region_lookahead): the region conversation `id`, at its region's end, moves back to
    /// a free region 0 as its generating conversation (the inverse of detach_active; the moves are save_peer's). With
    /// no other region occupied it continues there without bound; otherwise it is region 0's conversation at the
    /// boundary and parks as an image (batched_admission) until the regions are empty. A resident finished
    /// conversation in region 0 first keeps its cache image, as in save_peer.
    pub(crate) fn attach_peer(&mut self, id: u64) -> Result<Value> {
        let began = Instant::now();
        ensure!(enabled()? && self.current.is_none() && self.pending_prefill.is_none() && !self.prefill_cancelled
            && self.peer_at_boundary(id)?, "attach needs a free region 0 and a region conversation at its region's end");
        let mut displaced = Value::Null;
        if self.session.is_some() && !self.history.committed().is_empty() && self.session_cache_budget > 0 {
            let prefix = self.history.committed().to_vec();
            let pending = self.history.pending().context("resident conversation lacks its pending anchor")?;
            let media = self.history.media().through(prefix.len());
            displaced = self.retain_cached_prefix_reserved(&prefix, pending, media, 0, 0, self.displaced_snapshot)?;
            if crate::config::batched_detach_keeps_images()? {
                let kept = self.materialize_displaced_checkpoint(&[], &prefix, None)?;
                if !kept.is_null() && displaced.is_object() { displaced["kept_images"] = kept; }
            }
        }
        let index = self.peers.iter().position(|p| p.id == id).context("unknown region conversation")?;
        let peer = self.peers.remove(index);
        let position = peer.position();
        self.batched_timing.last_end = None;
        self.target_lineage = self.target_lineage.checked_add(1).context("target lineage exhausted")?;
        self.verify.reset(self.target_lineage, &self.comm, &self.stream)?;
        // Region 0's rows [0, position) stay below region 1's: the conversation stopped short of its own region's end.
        let moved = self.verify.relocate_session(peer.base, SessionBase::default(), position, &self.stream)?;
        if crate::config::elastic_regions()? { self.release_regions()?; self.elastic_declined = None; }
        self.verify.adopt_moved_history(peer.history.clone())?;
        let mut resident = peer.resident;
        resident.synchronize_request_ids(&mut self.history)?;
        self.history = resident;
        let spare = self.session.replace(peer.session);
        if let Some(spare) = spare { self.spare_sessions.push(spare); }
        self.current = Some(id);
        self.emitted = peer.emitted;
        self.limit = peer.limit;
        self.control.agree(&serde_json::to_vec(&json!({"batched_attach_v1":[id, peer.region as u64, position as u64]}))?)?;
        Ok(json!({"type":"attach","request_id":id,"region":peer.region,"position":position,"emitted":self.emitted,
            "limit":self.limit,"relocation":moved,"displaced_resident":displaced,"seconds":began.elapsed().as_secs_f64()}))
    }
    /// The region of a region conversation.
    pub(crate) fn peer_region(&self, id: u64) -> Option<usize> {
        self.peers.iter().find(|p| p.id == id).map(|p| p.region)
    }
    /// One decode step for every decoding conversation: the region-0 conversation (when
    /// `include_active`) as part 0, then the region conversations in region order, except regions
    /// in `skip` (bit r: region r's reader is behind; it sits out until its output queue drains).
    pub(crate) fn batched_decode_step(&mut self, include_active: bool, skip: u8) -> Result<Value> {
        use crate::dspark_generation::GreedySession as Session;
        let began = Instant::now();
        ensure!(enabled()? && (include_active || !self.peers.is_empty()), "batched step without conversations");
        if include_active {
            ensure!(self.current.is_some() && self.pending_prefill.is_none() && !self.prefill_cancelled
                && self.emitted < self.limit && self.history.pending().is_some_and(|a| a != self.cfg.eos_token_id),
                "batched step needs a generating region-0 conversation");
        }
        self.peers.sort_by_key(|p| p.region);
        // Lookahead admission: a conversation at its region's end never takes a step (the scheduler also leaves it out).
        let capacity = self.region_capacity()?;
        if include_active {
            ensure!(self.peers.is_empty() || !at_region_end(self.verify.position() as usize,
                self.limit.saturating_sub(self.emitted), capacity), "region-0 conversation at the regions' rows");
        }
        let stepping = |p: &Peer| p.finished.is_none() && skip & (1u8 << p.region) == 0
            && !at_region_end(p.position(), p.limit.saturating_sub(p.emitted), capacity);
        let decoding = self.peers.iter().filter(|p| stepping(p)).count();
        ensure!(include_active || decoding > 0, "batched step without decoding conversations");
        let parts_n = decoding + usize::from(include_active);
        // One conversation keeps the 6-row owner. Two take 3 rows each on it as well (101-103 tok/s in
        // the Nov 10 screens, against 93 at 6 rows each on the 16-row owner;
        // CANADA_SCRATCH_BATCHED_PAIR_ROWS=16 selects the latter). Three or four share the 16-row owner.
        // Full serving regions: only the 16-row owner's index branches hold the regions' keys (Target::key_regions_for),
        // so every step with a region conversation runs there; region 0 alone keeps the 6-row owner.
        let rows = if decoding > 0 && crate::config::full_regions()? { 16 }
            else { match parts_n { 1 => 6, 2 => pair_rows()?, _ => 16 } };
        let per = crate::target::batched_block(rows, parts_n).min(6);
        let active_id = self.current;
        let mut ids = Vec::with_capacity(parts_n);
        let mut group: Vec<BatchedSession> = Vec::with_capacity(parts_n);
        // Guarded draft-head subset: every conversation's verdict from its own committed tokens, before the step.
        let subset_guarded = crate::config::draft_head_subset()? == 1;
        if include_active {
            let limit = per.min(self.history.step_limit_for_rows(if rows == 16 { 16 } else { 6 })?);
            ensure!(limit > 0, "region-0 conversation has no step budget");
            if subset_guarded {
                let ok = crate::config::draft_head_subset_guard(self.history.committed(), self.history.pending().unwrap());
                self.session.as_mut().context("active drafter")?.set_head_subset_ok(ok);
            }
            group.push(BatchedSession { session: self.session.as_mut().context("active drafter")?,
                base: SessionBase::default(), history: self.verify.history_for_probe(),
                anchor: self.history.pending().unwrap(), limit, trace: None });
            ids.push(active_id.unwrap());
        }
        for p in self.peers.iter_mut().filter(|p| stepping(p)) {
            let limit = per.min(p.resident.step_limit_for_rows(if rows == 16 { 16 } else { 6 })?);
            ensure!(limit > 0, "region conversation has no step budget");
            if subset_guarded {
                let ok = crate::config::draft_head_subset_guard(p.resident.committed(),
                    p.resident.pending().context("region conversation anchor")?);
                p.session.set_head_subset_ok(ok);
            }
            ids.push(p.id);
            group.push(BatchedSession { session: &mut p.session, base: p.base, history: p.history.clone(),
                anchor: p.resident.pending().context("region conversation anchor")?, limit, trace: None });
        }
        let captures_before = self.long_verify.as_ref().map_or(0, |t| t.graph_capture_count()) + self.verify.graph_capture_count();
        let mean_position = group.iter().map(|p| p.history.position as f64).sum::<f64>() / parts_n as f64;
        let core_began = Instant::now();
        let steps = {
            let target: &mut Target = if rows == 16 {
                let longer = self.long_verify.as_mut().context("batched decode needs the 16-row owner")?;
                longer.handoff_from(&self.verify)?;
                longer.ensure_batched_engram_overlap()?;
                longer
            } else { &mut self.verify };
            target.set_batched_owner_part(include_active)?;
            Session::batched_step(&mut group, target, &self.comm, &self.stream)?
        };
        let core_ms = core_began.elapsed().as_secs_f64() * 1000.;
        let histories: Vec<_> = group.iter().map(|p| p.history.clone()).collect();
        drop(group);
        if rows == 16 { self.verify.handoff_from(self.long_verify.as_ref().unwrap())?; }
        // Commit to the request histories and publish.
        let mut conversations = Vec::with_capacity(parts_n);
        let mut proof = Vec::with_capacity(parts_n);
        for (k, (step, history)) in steps.iter().zip(histories).enumerate() {
            let (id, output, frontier, pending, emitted) = if include_active && k == 0 {
                ensure!(self.verify.history_for_probe() == history, "region-0 commit did not reach the owner");
                let output = self.history.committed_step(&step.tokens, step.next_anchor, history.position as usize)?;
                self.emitted += output.len();
                ensure!(self.emitted <= self.limit, "batched output exceeded the request limit");
                (ids[0], output, self.history.frontier(), self.history.pending(), self.emitted)
            } else {
                let peer = self.peers.iter_mut().find(|p| p.id == ids[k]).context("region conversation lost")?;
                peer.history = history;
                let output = peer.resident.committed_step(&step.tokens, step.next_anchor, peer.history.position as usize)?;
                peer.emitted += output.len();
                ensure!(peer.emitted <= peer.limit, "batched output exceeded a region request limit");
                (peer.id, output, peer.resident.frontier(), peer.resident.pending(), peer.emitted)
            };
            crate::dspark_generation::log_step_confidence("batched", step, output.len());
            proof.push(json!([id, output, frontier, pending, emitted]));
            conversations.push(json!({"request_id":id,"output_ids":output,"frontier":frontier,"pending":pending,
                "completion_tokens_so_far":emitted,"verified_rows":step.verified_rows,"step_limit":per,
                "accepted_draft_tokens":step.matching_proposals,"proposal_source":step.proposal_source,
                "draft_confidence":step.draft_confidence,
                "region":if include_active && k == 0 {0} else {self.peers.iter().find(|p| p.id == id).map_or(0, |p| p.region)}}));
        }
        let agree_began = Instant::now();
        self.control.agree(&serde_json::to_vec(&json!({"batched_committed_v1":proof}))?)?;
        let agree_ms = agree_began.elapsed().as_secs_f64() * 1000.;
        let emitted_tokens: usize = conversations.iter().map(|c| c["output_ids"].as_array().map_or(0, |a| a.len())).sum();
        let captured = self.long_verify.as_ref().map_or(0, |t| t.graph_capture_count()) + self.verify.graph_capture_count() > captures_before;
        let step_total = began.elapsed().as_secs_f64() * 1000.;
        let t = &mut self.batched_timing;
        if captured { t.capture_steps[parts_n] += 1; t.capture_ms[parts_n] += step_total; }
        else {
            if t.steady[parts_n].len() >= 4096 { t.steady[parts_n].drain(..2048); }
            t.steady[parts_n].push(step_total);
        }
        t.positions[parts_n] += mean_position;
        t.steps[parts_n] += 1; t.tokens[parts_n] += emitted_tokens; t.core_ms[parts_n] += core_ms;
        t.agree_ms[parts_n] += agree_ms; t.step_ms[parts_n] += began.elapsed().as_secs_f64() * 1000.;
        if let Some(last) = t.last_end { t.period_ms[parts_n] += last.elapsed().as_secs_f64() * 1000.; t.periods[parts_n] += 1; }
        t.last_end = Some(Instant::now());
        t.since_report += 1;
        if t.since_report >= 100 && self.comm.rank() == 0 {
            let captures = self.long_verify.as_ref().map_or(0, |t| t.graph_capture_count()) + self.verify.graph_capture_count();
            self.batched_timing.report(captures);
        }
        Ok(json!({"type":"batched_tokens","conversations":conversations,"owner_rows":rows,
            "rows_per_conversation":per,"region0":include_active,"seconds":began.elapsed().as_secs_f64()}))
    }
}
