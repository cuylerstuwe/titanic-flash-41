//! HTTP adapter for the native resident worker. HTTP threads own only
//! protocol conversion and bounded queues; the main rank thread owns all GPU
//! state. Sixteen live/queued requests are admitted without a batching timer;
//! background admission reserves one of those entries for foreground.
use crate::protocol::{BackendEvent, Frontend, MODEL_NAME};
#[path = "http_timing.rs"]
mod timing;
#[path = "http_admission.rs"]
mod admission;
#[path = "http_raw_input.rs"]
mod raw_input;
#[path = "http_context.rs"]
mod context;
#[path = "input_reclaim_ack.rs"]
mod reclaim_ack;
use timing::{PrefillTiming, WithTimings};
use anyhow::{Result, ensure};
use axum::{
    Json, Router,
    body::{Body, Bytes, to_bytes},
    extract::{Path as RoutePath, Request, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use deepseek_recipe::{
    openai::chat_completion::response::ChatCompletionResponse,
    response::{Accumulate, ProtocolResponse},
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    cell::{Cell, RefCell},
    collections::{HashMap, VecDeque},
    io::Read,
    net::SocketAddr,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    task::{Context, Poll},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use tokio_stream::{Stream, StreamExt};

const MAX_BODY: usize = 128 * 1024 * 1024;
// Queued jobs own CPU request data, not an extra live GPU state. The independent
// four-entry/2 GiB parked-state budget still governs execution and restoration.
const QUEUE_DEPTH: usize = 15;
const OUTPUT_CHUNKS: usize = 64;
// The scheduler decides the next operation before publishing this operation's
// Ready/Tokens events. Keep room for those events and a terminal notification.
const OUTPUT_RESERVE: usize = 3;
const LOADING: u8 = 0;
const READY: u8 = 1;
const FAILED: u8 = 2;
// Post-Ready work must keep completing bounded control exchanges, including
// idle/priority holds and each encoder chunk. Loading is not timed here.
const WORKER_PROGRESS_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Default)]
struct ProgressWatch {
    sequence: u64,
    last: Option<Instant>,
    maximum_gap: Duration,
}
impl ProgressWatch {
    fn observe(&mut self, ready: bool, sequence: u64, now: Instant) -> Duration {
        if !ready { self.last = None; return Duration::ZERO; }
        let gap = self.last.map_or(Duration::ZERO, |last| now.saturating_duration_since(last));
        self.maximum_gap = self.maximum_gap.max(gap);
        if self.last.is_none() || self.sequence != sequence {
            self.sequence = sequence; self.last = Some(now); Duration::ZERO
        } else { gap }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServerConfig {
    pub bind: SocketAddr,
    pub tokenizer: PathBuf,
    pub bearer_file: PathBuf,
    #[serde(default)]
    pub priority_file: Option<PathBuf>,
    /// Explicit opt-in for an authenticated listener on a fleet interface.
    #[serde(default)]
    pub allow_remote: bool,
    #[serde(default = "default_output_stall_seconds")]
    pub output_stall_seconds: u64,
    /// Optional immutable input-message cache; zero keeps transport disabled.
    #[serde(default)]
    pub context_cache_mib: usize,
}
fn default_output_stall_seconds() -> u64 { 60 }

impl ServerConfig {
    fn validate_listener(&self) -> Result<()> {
        let ip = self.bind.ip();
        let fleet = match ip {
            std::net::IpAddr::V4(v) => v.is_private()
                || (v.octets()[0] == 100 && (64..=127).contains(&v.octets()[1])),
            std::net::IpAddr::V6(v) => v.is_unique_local(),
        };
        ensure!(self.bind.port() != 0 && self.bind.port() != 18042,
            "native HTTP candidate requires a distinct nonzero port");
        ensure!(ip.is_loopback() || (self.allow_remote && fleet && self.priority_file.is_some()),
            "remote native HTTP requires explicit fleet binding and scheduler authentication");
        ensure!((1..=300).contains(&self.output_stall_seconds), "output stall timeout must be 1..300 seconds");
        ensure!(self.context_cache_mib <= 128, "context transport cache is bounded at128MiB");
        Ok(())
    }
}

#[cfg(test)]
#[test]
fn listener_requires_explicit_authenticated_fleet_binding() {
    let config = |address: &str, allow_remote: bool, priority: bool| ServerConfig {
        bind: address.parse().unwrap(), tokenizer: "unused".into(), bearer_file: "unused".into(),
        priority_file: priority.then(|| PathBuf::from("scheduler.key")), allow_remote,
        output_stall_seconds: 60, context_cache_mib: 0,
    };
    assert!(config("127.0.0.1:18046", false, false).validate_listener().is_ok());
    assert!(config("[::1]:18046", false, false).validate_listener().is_ok());
    for address in ["192.168.1.2:18046", "10.0.0.2:18046", "100.64.0.2:18046", "[fd00::2]:18046"] {
        assert!(config(address, false, true).validate_listener().is_err());
        assert!(config(address, true, false).validate_listener().is_err());
        assert!(config(address, true, true).validate_listener().is_ok());
    }
    for address in ["0.0.0.0:18046", "[::]:18046", "8.8.8.8:18046", "100.128.0.2:18046",
        "127.0.0.1:0", "127.0.0.1:18042"] {
        assert!(config(address, true, true).validate_listener().is_err());
    }
}

struct Shared {
    sender: mpsc::SyncSender<Job>,
    // Separate, coalesced notifications do not consume request admission.
    // Input pressure can need the owner before a model-ready Job exists.
    wake_sender: mpsc::SyncSender<()>,
    frontend: Arc<Frontend>,
    capacity: usize,
    body_limit: usize,
    admission: Arc<Semaphore>,
    background_admission: Arc<Semaphore>,
    decode_priority: AtomicBool,
    background_prefill: AtomicBool,
    execution_activity: AtomicBool,
    execution_admission: Arc<Semaphore>,
    waiting_inputs: AtomicUsize,
    waiting_image_budget: AtomicUsize,
    waiting_raw_pool: Mutex<Option<Arc<raw_input::Pool>>>,
    context_cache: Option<Mutex<crate::context_transport::Cache>>,
    health: AtomicU8,
    progress: Arc<AtomicU64>,
    progress_gap_ms: AtomicU64,
    progress_maximum_gap_ms: AtomicU64,
    progress_timed_out: AtomicBool,
    failure: tokio::sync::watch::Sender<bool>,
    shutdown: AtomicBool,
    notify_shutdown: Notify,
    identity: Mutex<Option<String>>,
    requests: Mutex<HashMap<String, Weak<AtomicBool>>>,
    authorization: [u8; 32],
    nonce: String,
    sequence: AtomicU64,
    priority: Option<crate::router_priority::Control>,
    priority_authorization: Option<[u8; 32]>,
    control_admission: Semaphore,
    output_stall_timeout: Duration,
    output_backpressure_events: AtomicU64,
    output_stall_timeouts: AtomicU64,
    input_reclaim_target: AtomicUsize,
    input_reclaim_generation: AtomicU64,
    input_reclaim_ack: reclaim_ack::Ack,
}

impl Shared {
    fn wake_worker(&self) {
        // Full means a wake is already pending. Disconnected means the owner
        // has retired; the existing health/timeout path handles that case.
        let _ = self.wake_sender.try_send(());
    }
    fn fail(&self) {
        self.health.store(FAILED, Ordering::Release);
        self.shutdown.store(true, Ordering::Release);
        self.execution_admission.close();
        self.wake_worker();
        // Wake streams even when the sole model owner is stuck in CUDA/NCCL
        // and cannot drop their backend senders. Keep HTTP health available
        // until the external supervisor retires this complete worker group.
        self.failure.send_replace(true);
    }
}

fn fail_on_worker_failure<T: Send + 'static>(
    source: impl Stream<Item = Result<T>> + Send + 'static,
    mut failure: tokio::sync::watch::Receiver<bool>,
) -> impl Stream<Item = Result<T>> + Send + 'static {
    async_stream::stream! {
        let mut source = std::pin::pin!(source);
        loop {
            let failed = *failure.borrow();
            if failed { yield Err(anyhow::anyhow!("native worker group failed")); return; }
            tokio::select! {
                biased;
                changed = failure.changed() => {
                    if changed.is_err() || *failure.borrow() {
                        yield Err(anyhow::anyhow!("native worker group failed")); return;
                    }
                }
                item = source.next() => match item {
                    Some(item) => yield item,
                    None => return,
                }
            }
        }
    }
}

/// SSE comments are transport metadata, separate from model deltas and PP.
/// Native admission/pause transitions let a router exclude waiting time from
/// its request budget. Closing the watch must not discard buffered final data.
fn with_execution_activity(
    source: impl Stream<Item = std::io::Result<Event>> + Send + 'static,
    activity: Option<tokio::sync::watch::Receiver<bool>>,
) -> impl Stream<Item = std::io::Result<Event>> + Send + 'static {
    async_stream::stream! {
        let mut source = std::pin::pin!(source);
        let Some(mut activity) = activity else {
            while let Some(event) = source.next().await { yield event; }
            return;
        };
        let comment = |running| Event::default().comment(if running {
            "native-execution running"
        } else { "native-execution waiting" });
        let initial = *activity.borrow_and_update();
        yield Ok(comment(initial));
        let mut open = true;
        loop {
            tokio::select! {
                biased;
                changed = activity.changed(), if open => {
                    if changed.is_ok() {
                        let running = *activity.borrow_and_update();
                        yield Ok(comment(running));
                    } else { open = false; }
                }
                event = source.next() => match event {
                    Some(event) => yield event,
                    None => return,
                }
            }
        }
    }
}

pub(crate) struct Job {
    pub id: String,
    pub tokens: Vec<u32>,
    pub images: Vec<crate::image_cache::Descriptor>,
    pub image_inputs: Vec<Arc<crate::worker_media::PatchInput>>,
    pub seed: u64,
    pub greedy: bool,
    pub limit: usize,
    pub enqueued: Instant,
    pub background: bool,
    cancelled: Arc<AtomicBool>,
    output: tokio::sync::mpsc::Sender<BackendEvent>,
    prompt_tokens: usize,
    prefill_timing: Arc<OnceLock<PrefillTiming>>,
    output_stalled_since: Cell<Option<Instant>>,
    activity: Option<tokio::sync::watch::Sender<bool>>,
    executing: Cell<bool>,
    shared: Arc<Shared>,
    _permit: OwnedSemaphorePermit,
    _background_permit: Option<OwnedSemaphorePermit>,
    _execution_permit: Option<OwnedSemaphorePermit>,
}
impl Job {
    /// Transitions only: no timer, lock or notification on an unchanged step.
    pub(crate) fn set_executing(&self, executing: bool) {
        if self.executing.replace(executing) != executing {
            if let Some(activity) = &self.activity { activity.send_replace(executing); }
        }
    }

    pub(crate) fn needs_cached_prefix(&self, prefix: &[u32], media: &crate::prompt_media::Identity) -> bool {
        // Every entry belongs to this worker's one immutable encoder identity.
        // Encoded input identity and full span geometry must still match; equal
        // placeholder tokens alone cannot identify an image-bearing prefix.
        self.tokens.starts_with(prefix) && self.images.iter().take_while(|image| image.start < prefix.len())
            .map(|image| (image.start, image.geometry.tokens(), image.input))
            .eq(media.placements().iter().take_while(|image| image.start < prefix.len())
                .map(|image| (image.start, image.tokens, image.key.input)))
    }
    /// Publish only a completed measured phase, once per request. The slot
    /// travels with a parked Job; another request cannot overwrite its timing.
    pub fn record_prefill_timing(&self, event: &serde_json::Value) {
        if let Some(timing) = PrefillTiming::from_worker(event, self.prompt_tokens) {
            let _ = self.prefill_timing.set(timing);
        }
    }
    pub fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire) || self.output.is_closed()
    }
    pub fn emit(&self, event: BackendEvent) {
        // The interleaving scheduler holds or parks before this queue fills.
        // A closed or unexpectedly full queue remains a fail-closed fallback;
        // dropping this sender makes the frontend report incomplete output.
        if self.output.try_send(event).is_err() {
            self.cancelled.store(true, Ordering::Release);
        }
    }
    /// Only the root GPU owner accesses this state. Normal progress pays one
    /// channel-capacity read; no timers, locks or CUDA work are added to it.
    pub fn output_backpressured(&self) -> bool {
        if self.output.capacity() >= OUTPUT_RESERVE {
            self.output_stalled_since.set(None);
            return false;
        }
        self.output_backpressured_at(Instant::now())
    }
    fn output_backpressured_at(&self, now: Instant) -> bool {
        if self.cancelled() {return false;}
        if self.output.capacity() >= OUTPUT_RESERVE {
            self.output_stalled_since.set(None);
            return false;
        }
        let started = if let Some(started) = self.output_stalled_since.get() {started} else {
            self.output_stalled_since.set(Some(now));
            self.shared.output_backpressure_events.fetch_add(1, Ordering::Relaxed);
            now
        };
        if now.saturating_duration_since(started) >= self.shared.output_stall_timeout {
            self.shared.output_stall_timeouts.fetch_add(1, Ordering::Relaxed);
            self.cancelled.store(true, Ordering::Release);
            return false;
        }
        true
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        self.set_executing(false);
        self.shared.requests.lock().unwrap().remove(&self.id);
    }
}

#[derive(Clone,Copy)]
pub(crate) enum PrefillPeer { Decode, Prompt(usize) }
fn prefill_work_yield(remaining:usize,peer:PrefillPeer,slice_due:bool)->bool {
    match peer {
        PrefillPeer::Decode=>slice_due,
        // Hysteresis prevents two comparable prefills exchanging repeatedly.
        // The current owner keeps equal work; initial selection is FIFO on ties.
        PrefillPeer::Prompt(other)=>other.checked_mul(2).is_some_and(|n|n<remaining),
    }
}
fn prefill_turn_due(elapsed:Duration,normal:Duration,peer:Option<PrefillPeer>,short:Option<Duration>)->bool {
    // Shorten only a prefill delaying a ready decoder. Prompt/prompt ordering,
    // the newcomer's first turn and repayment protection are separate rules.
    let limit=if matches!(peer,Some(PrefillPeer::Decode)){short.unwrap_or(normal).min(normal)}else{normal};
    elapsed>=limit
}
pub(crate) struct Queue {
    receiver: mpsc::Receiver<Job>,
    wake_receiver: mpsc::Receiver<()>,
    pending: RefCell<VecDeque<Job>>,
    execution_owners: Cell<Option<usize>>,
    // Set by the two-owner scheduler, not inferred from raw third requests.
    prefill_peer_ready: Cell<bool>,
    work_aware_prefill: Cell<bool>,
    prefill_peer_work: Cell<Option<PrefillPeer>>,
    prefill_protected_until: Cell<Option<Instant>>,
    decode_prefill_slice: Cell<Option<Duration>>,
    shared: Arc<Shared>,
}
impl Queue {
    pub fn try_next(&self) -> Result<Option<Job>> {
        self.try_selected(false, false)
    }
    /// Admission order is independent of execution priority. In particular a
    /// third foreground request cannot acquire a third execution slot.
    pub(crate) fn try_fifo(&self) -> Result<Option<Job>> {
        self.try_selected(false, true)
    }
    pub fn foreground_can_join(&self) -> Result<bool> {
        // A third foreground cannot interrupt an already admitted pair's
        // prefill. Peeking must never move it ahead of earlier FIFO requests.
        if self.execution_owners.get().is_some_and(|n| n >= 2) { return Ok(false); }
        self.buffer_pending()?;
        Ok(self.pending.borrow().iter().any(|job| !job.background))
    }
    pub(crate) fn set_execution_owners(&self, owners: usize) -> Result<()> {
        ensure!(self.execution_owners.get().is_some() && owners <= 2, "two-stream admission accounting");
        self.execution_owners.set(Some(owners)); Ok(())
    }
    pub(crate) fn set_prefill_peer_ready(&self, ready: bool) {
        self.prefill_peer_ready.set(ready);
    }
    pub(crate) fn set_prefill_work(&self,enabled:bool,peer:Option<PrefillPeer>,protected:Option<Instant>) {
        self.work_aware_prefill.set(enabled);self.prefill_peer_work.set(peer);
        self.prefill_protected_until.set(protected);
    }
    pub(crate) fn set_decode_prefill_slice(&self,slice:Option<Duration>) {self.decode_prefill_slice.set(slice);}
    pub(crate) fn prefill_slice_due(&self,elapsed:Duration,normal:Duration)->bool {
        prefill_turn_due(elapsed,normal,self.prefill_peer_work.get(),
            if self.work_aware_prefill.get(){self.decode_prefill_slice.get()}else{None})
    }
    pub(crate) fn prefill_peer_should_yield(&self,background:bool,remaining:usize,slice_due:bool,
        mut estimate:impl FnMut(&Job)->usize)->Result<bool> {
        if !self.work_aware_prefill.get(){return Ok(slice_due&&self.prefill_peer_can_run(background)?);}
        if self.prefill_protected_until.get().is_some_and(|end|Instant::now()<end){return Ok(false);}
        if let Some(peer)=self.prefill_peer_work.get(){return Ok(prefill_work_yield(remaining,peer,slice_due));}
        let Some(owners)=self.execution_owners.get()else{return Ok(false);};
        if owners>=2{return Ok(false);}
        self.buffer_pending()?;
        Ok(self.pending.borrow().front().filter(|peer|(background||!peer.background)
            &&!peer.cancelled()&&!peer.output_backpressured())
            .is_some_and(|peer|prefill_work_yield(remaining,PrefillPeer::Prompt(estimate(peer)),slice_due)))
    }
    pub(crate) fn prefill_peer_can_run(&self, background: bool) -> Result<bool> {
        if self.prefill_peer_ready.get() { return Ok(true); }
        let Some(owners)=self.execution_owners.get() else { return Ok(false); };
        if owners>=2 { return Ok(false); }
        self.buffer_pending()?;
        // Only the first FIFO entrant can own the vacant second slot. A
        // foreground owner does not yield solely for a background entrant.
        Ok(self.pending.borrow().front().is_some_and(|peer|background||!peer.background))
    }
    pub fn try_foreground(&self) -> Result<Option<Job>> {
        self.try_selected(true, false)
    }
    fn buffer_pending(&self) -> Result<()> {
        let mut pending = self.pending.borrow_mut();
        pending.retain(|job| !job.cancelled());
        loop {
            match self.receiver.try_recv() {
                Ok(job) if job.cancelled() => continue,
                Ok(job) => pending.push_back(job),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(error) => return Err(error.into()),
            }
        }
        ensure!(pending.len() <= QUEUE_DEPTH + 1, "HTTP admission ownership bound");
        Ok(())
    }
    fn try_selected(&self, foreground_only: bool, fifo: bool) -> Result<Option<Job>> {
        self.buffer_pending()?;
        let mut pending = self.pending.borrow_mut();
        let selected = if fifo { (!pending.is_empty()).then_some(0) } else {
            pending.iter().position(|job| !job.background)
                .or_else(|| (!foreground_only && !pending.is_empty()).then_some(0))
        };
        Ok(selected.and_then(|i| pending.remove(i)))
    }
    pub(crate) fn pending_needs_prefix(&self, prefix: &[u32], media: &crate::prompt_media::Identity) -> Result<bool> {
        // Raw queued inputs have not been tokenized yet. Protect every cache
        // entry until their exact prefix identity is available at admission.
        if self.shared.waiting_inputs.load(Ordering::Acquire) > 0 { return Ok(true); }
        self.buffer_pending()?;
        Ok(self.pending.borrow().iter().any(|job| job.needs_cached_prefix(prefix, media)))
    }
    pub fn next(&self, wait: Duration) -> Result<Option<Job>> {
        self.next_selected(wait, false)
    }
    pub(crate) fn next_fifo(&self, wait: Duration) -> Result<Option<Job>> {
        self.next_selected(wait, true)
    }
    fn next_selected(&self, wait: Duration, fifo: bool) -> Result<Option<Job>> {
        let deadline = Instant::now() + wait;
        loop {
            if let Some(job) = self.try_selected(false, fifo)? { return Ok(Some(job)); }
            // The owner performs reclaim at its ordered scheduling boundary.
            // Never do device work from an HTTP thread or inside this wait.
            if self.shared.input_reclaim_target.load(Ordering::Acquire) > 0 || self.stopping() {
                return Ok(None);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            match self.wake_receiver.recv_timeout(remaining) {
                Ok(()) => continue,
                Err(mpsc::RecvTimeoutError::Timeout) => return Ok(None),
                Err(error) => return Err(error.into()),
            }
        }
    }
    pub fn progress_counter(&self) -> Arc<AtomicU64> { self.shared.progress.clone() }
    pub fn ready(&self, identity: &str) {
        *self.shared.identity.lock().unwrap() = Some(identity.into());
        self.shared.health.store(READY, Ordering::Release);
    }
    pub(crate) fn enable_two_stream_admission(&self) {
        self.execution_owners.set(Some(0));
        self.shared.execution_activity.store(true, Ordering::Release);
    }
    pub(crate) fn reserve_waiting_image_capacity(&self, bytes: usize) -> Result<()> {
        assert!(bytes <= 2usize << 30);
        let mut pool = self.shared.waiting_raw_pool.lock().unwrap();
        if pool.is_none() {
            // Four ordinary text uploads, physically backed before readiness.
            // Larger or additional bodies retain the future-growth guard.
            *pool = Some(raw_input::Pool::new(4, 8 << 20)?);
        }
        self.shared.waiting_image_budget.store(bytes, Ordering::Release);
        Ok(())
    }
    pub fn enable_background_prefill(&self) {
        self.shared.background_prefill.store(true,Ordering::Release);
    }
    pub fn enable_decode_priority(&self) {
        self.shared.decode_priority.store(true, Ordering::Release);
    }
    pub fn stopping(&self) -> bool {
        self.shared.shutdown.load(Ordering::Acquire)
    }
    pub(crate) fn has_outstanding_requests(&self) -> bool {
        self.shared.admission.available_permits() != QUEUE_DEPTH + 1
    }
    pub(crate) fn input_pending_for_vacancy(&self) -> bool {
        let Some(owners)=self.execution_owners.get().filter(|n|*n<2)else{return false;};
        // The FIFO waiter can be waking up, or already hold the newly freed
        // lifetime permit while uploading/converting. Neither is a model Job
        // yet. An idle/uncontended stream must never wait for imaginary work.
        self.shared.waiting_inputs.load(Ordering::Acquire)>0
            ||self.shared.execution_admission.available_permits()<2-owners
    }
    pub fn input_reclaim_target(&self) -> usize {
        let target = self.shared.input_reclaim_target.load(Ordering::Acquire);
        if target > 0 { context::reclaim(&self.shared); }
        target
    }
    pub fn input_reclaimed(&self, target: usize) {
        // A larger request arriving during reclamation needs another pass.
        let _ = self.shared.input_reclaim_target.fetch_update(Ordering::AcqRel, Ordering::Acquire,
            |pending| (pending <= target).then_some(0));
        self.shared.input_reclaim_ack.publish(&self.shared.input_reclaim_generation);
    }
    pub fn router_hold(&self) -> bool {
        self.shared.priority.as_ref().is_some_and(|p| p.hold())
    }
    pub fn priority_pending(&self) -> Option<crate::router_priority::Snapshot> {
        self.shared.priority.as_ref().and_then(|p| p.pending())
    }
    pub fn priority_applied(&self, snapshot: crate::router_priority::Snapshot) -> Result<()> {
        self.shared.priority.as_ref().ok_or_else(|| anyhow::anyhow!("priority is disabled"))?.acknowledge(snapshot)
    }
    pub fn fail(&self) {
        self.shared.fail();
        for job in self.pending.borrow_mut().drain(..) {
            job.emit(BackendEvent::Failed("native worker group failed".into()));
        }
        while let Ok(job) = self.receiver.try_recv() {
            job.emit(BackendEvent::Failed("native worker group failed".into()));
        }
        self.shared.notify_shutdown.notify_one();
    }
}
impl Drop for Queue {
    fn drop(&mut self) {
        self.fail();
    }
}

fn random_bytes<const N: usize>() -> Result<[u8; N]> {
    let mut bytes = [0; N];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes)
}
fn auth_digest(path: &Path) -> Result<[u8; 32]> {
    use std::os::unix::fs::PermissionsExt;
    let metadata = std::fs::metadata(path)?;
    ensure!(
        metadata.is_file() && metadata.len() <= 256 && metadata.permissions().mode() & 0o077 == 0,
        "HTTP bearer file must be a bounded owner-only file"
    );
    let text = std::fs::read_to_string(path)?;
    let token = text.trim();
    ensure!(
        (32..=128).contains(&token.len())
            && token
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b)),
        "HTTP bearer token format"
    );
    Ok(Sha256::digest(format!("Bearer {token}").as_bytes()).into())
}
fn authorized(shared: &Shared, headers: &HeaderMap) -> bool {
    let Some(value) = headers.get("authorization") else {
        return false;
    };
    let digest = Sha256::digest(value.as_bytes());
    digest
        .iter()
        .zip(shared.authorization)
        .fold(0u8, |n, (&a, b)| n | (a ^ b))
        == 0
}
fn error(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(json!({"error":{"type":code,"code":code,"message":message}})),
    )
        .into_response()
}
fn check(shared: &Shared, headers: &HeaderMap) -> Option<Response> {
    (!authorized(shared, headers)).then(|| {
        error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid bearer token",
        )
    })
}
async fn health(State(shared): State<Arc<Shared>>, headers: HeaderMap) -> Response {
    if let Some(error) = check(&shared, &headers) {
        return error;
    }
    let state = shared.health.load(Ordering::Acquire);
    (
        if state == READY {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(
            json!({"status":match state {READY=>"ready",FAILED=>"failed",_=>"loading"},
            "model":MODEL_NAME,"context_capacity":shared.capacity,
            "worker_progress":{"timeout_seconds":WORKER_PROGRESS_TIMEOUT.as_secs(),
                "completed_control_sequence":shared.progress.load(Ordering::Acquire),
                "gap_ms":shared.progress_gap_ms.load(Ordering::Relaxed),
                "maximum_observed_gap_ms":shared.progress_maximum_gap_ms.load(Ordering::Relaxed),
                "timed_out":shared.progress_timed_out.load(Ordering::Acquire)},
            "active_or_queued":QUEUE_DEPTH+1-shared.admission.available_permits(),
            "waiting_for_execution_slot":shared.waiting_inputs.load(Ordering::Acquire),
            "output_flow":{"pause_enabled":shared.decode_priority.load(Ordering::Acquire),
                "stall_timeout_seconds":shared.output_stall_timeout.as_secs(),
                "backpressure_events":shared.output_backpressure_events.load(Ordering::Relaxed),
                "stall_timeouts":shared.output_stall_timeouts.load(Ordering::Relaxed)},
            "image_input_cache":shared.frontend.image_cache_metadata().unwrap_or_else(|_|Some(json!({"unavailable":true}))),
            "native_priority":{"decode":shared.decode_priority.load(Ordering::Acquire),
                "partial_prefill":shared.background_prefill.load(Ordering::Acquire),
                "router_leases":shared.priority.is_some() && shared.decode_priority.load(Ordering::Acquire),
                "router":shared.priority.as_ref().map(|p|p.metadata())},
            "production_qualified":false}),
        ),
    )
        .into_response()
}
async fn models(State(shared): State<Arc<Shared>>, headers: HeaderMap) -> Response {
    if let Some(error) = check(&shared, &headers) {
        return error;
    }
    Json(json!({"object":"list","data":[{"id":MODEL_NAME,"object":"model","owned_by":"deepseek"}]}))
        .into_response()
}
async fn diagnostics(State(shared): State<Arc<Shared>>, headers: HeaderMap) -> Response {
    if let Some(error) = check(&shared, &headers) {
        return error;
    }
    let mut response = Json(crate::event_log::snapshot()).into_response();
    response.headers_mut().insert("cache-control", HeaderValue::from_static("no-store"));
    response
}
async fn cancel(
    State(shared): State<Arc<Shared>>,
    RoutePath(id): RoutePath<String>,
    headers: HeaderMap,
) -> Response {
    if let Some(error) = check(&shared, &headers) {
        return error;
    }
    let cancellation = shared
        .requests
        .lock()
        .unwrap()
        .get(&id)
        .and_then(Weak::upgrade);
    match cancellation {
        Some(flag) => {
            flag.store(true, Ordering::Release);
            Json(json!({"id":id,"cancellation_requested":true})).into_response()
        }
        None => error(
            StatusCode::NOT_FOUND,
            "request_not_active",
            "request is not active or queued",
        ),
    }
}
async fn shutdown(State(shared): State<Arc<Shared>>, headers: HeaderMap) -> Response {
    if let Some(error) = check(&shared, &headers) {
        return error;
    }
    shared.health.store(LOADING, Ordering::Release);
    shared.shutdown.store(true, Ordering::Release);
    shared.execution_admission.close();
    shared.wake_worker();
    Json(json!({"shutdown_requested":true})).into_response()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PriorityUpdate { owner: String, epoch: u64, foreground: bool, lease_seconds: f64 }
async fn priority_control(State(shared): State<Arc<Shared>>, request: Request<Body>) -> Response {
    let supplied = request.headers().get_all("x-scheduler-token").iter().collect::<Vec<_>>();
    let valid = supplied.len() == 1 && shared.priority_authorization.is_some_and(|expected| {
        let mut hash = Sha256::new(); hash.update(b"Bearer "); hash.update(supplied[0].as_bytes());
        hash.finalize().iter().zip(expected).fold(0u8, |n, (&a,b)| n | (a ^ b)) == 0
    });
    if !valid { return error(StatusCode::UNAUTHORIZED, "authentication_error", "invalid scheduler token"); }
    if shared.health.load(Ordering::Acquire) != READY || !shared.decode_priority.load(Ordering::Acquire) {
        return error(StatusCode::SERVICE_UNAVAILABLE, "worker_not_ready", "native priority is not ready");
    }
    let Ok(_permit) = shared.control_admission.try_acquire() else {
        return error(StatusCode::TOO_MANY_REQUESTS, "priority_busy", "priority control capacity exceeded");
    };
    let body = match tokio::time::timeout(Duration::from_secs(2), to_bytes(request.into_body(), 4096)).await {
        Ok(Ok(body)) => body,
        _ => return error(StatusCode::BAD_REQUEST, "invalid_priority", "priority request body is incomplete or too large"),
    };
    let update: PriorityUpdate = match serde_json::from_slice(&body) {
        Ok(update) => update,
        Err(_) => return error(StatusCode::BAD_REQUEST, "invalid_priority", "invalid priority request"),
    };
    let priority = shared.priority.as_ref().unwrap();
    let revision = match priority.update(&update.owner, update.epoch, update.foreground, update.lease_seconds) {
        Ok(revision) => revision,
        Err(reason) => return error(StatusCode::CONFLICT, "invalid_priority", &reason.to_string()),
    };
    let deadline = Instant::now() + Duration::from_secs(25);
    loop {
        if shared.health.load(Ordering::Acquire) != READY || shared.shutdown.load(Ordering::Acquire) {
            return error(StatusCode::SERVICE_UNAVAILABLE, "worker_failed", "native priority owner stopped");
        }
        match priority.receipt(&update.owner, update.epoch, update.foreground, revision) {
            crate::router_priority::Receipt::Applied { hold } => return Json(json!({
                "owner":update.owner,"epoch":update.epoch,"foreground":update.foreground,
                "revision":revision,"background_held":hold,"applied":true})).into_response(),
            crate::router_priority::Receipt::Expired => return error(StatusCode::CONFLICT, "lease_expired", "priority lease expired before acknowledgement"),
            crate::router_priority::Receipt::Superseded => return error(StatusCode::CONFLICT, "lease_superseded", "priority lease was superseded"),
            crate::router_priority::Receipt::Pending => {},
        }
        if Instant::now() >= deadline { return error(StatusCode::SERVICE_UNAVAILABLE, "priority_timeout", "native priority drain was not acknowledged"); }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
struct GuardedStream<S> {
    inner: Pin<Box<S>>,
    _cancel: CancelOnDrop,
}
impl<S: Stream> Stream for GuardedStream<S> {
    type Item = S::Item;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().inner.as_mut().poll_next(cx)
    }
}

fn request_background(headers: &HeaderMap) -> Result<bool> {
    let values = headers.get_all("x-localai-mode").iter().collect::<Vec<_>>();
    ensure!(values.len() <= 1, "duplicate request mode");
    match values.first().map(|v| v.to_str()).transpose()?.unwrap_or("").trim().to_ascii_lowercase().as_str() {
        "" | "foreground" => Ok(false),
        "background" => Ok(true),
        _ => anyhow::bail!("X-LocalAI-Mode must be foreground or background"),
    }
}

/// Admission before reading or converting a body must be able to ask the GPU
/// owner to discard optional retained histories, just like image conversion.
/// Otherwise a transient low-memory observation can strand an idle server:
/// every subsequent request is refused before reaching its existing reclaimer.
/// The ordinary reservation still decides admission, with the same reserve.
async fn reserve_with_input_reclaim<T>(
    mut reserve: impl FnMut() -> Result<T>,
    target: &AtomicUsize,
    generation: &AtomicU64,
    ack: &reclaim_ack::Ack,
    ready: impl Fn() -> bool,
    wake: impl Fn(),
    timeout: Duration,
) -> Result<T> {
    let error = match reserve() {
        Ok(value) => return Ok(value),
        Err(error) => error,
    };
    let required = error.downcast_ref::<crate::memory::InputCapacityError>()
        .map_or(0, |error| error.1);
    if required == 0 || !ready() { return Err(error); }
    let before = generation.load(Ordering::Acquire);
    target.fetch_max(required, Ordering::AcqRel);
    wake();
    ack.wait_async(generation,before,&ready,timeout).await;
    if ready() && generation.load(Ordering::Acquire) != before {
        reserve()
    } else {
        Err(error)
    }
}

async fn reserve_input_workspace(shared: &Shared, bytes: usize)
    -> Result<crate::memory::HostWorkspaceReservation> {
    reserve_with_input_reclaim(
        || context::reserve_after_reclaim(shared, || crate::memory::reserve_host_workspace(bytes)),
        &shared.input_reclaim_target, &shared.input_reclaim_generation, &shared.input_reclaim_ack,
        || shared.health.load(Ordering::Acquire) == READY && !shared.shutdown.load(Ordering::Acquire),
        || shared.wake_worker(),
        Duration::from_secs(5),
    ).await
}

async fn chat(State(shared): State<Arc<Shared>>, request: Request<Body>) -> Response {
    chat_input(shared, request, false).await
}
async fn context_chat(State(shared): State<Arc<Shared>>, request: Request<Body>) -> Response {
    chat_input(shared, request, true).await
}
async fn chat_input(shared: Arc<Shared>, request: Request<Body>, incremental: bool) -> Response {
    if let Some(error) = check(&shared, request.headers()) {
        return error;
    }
    if incremental {
        if shared.context_cache.is_none() {
            return error(StatusCode::NOT_FOUND, "context_unsupported", "incremental context transport is disabled");
        }
        let versions = request.headers().get_all("x-titanic-context").iter().collect::<Vec<_>>();
        if versions.len() != 1 || versions[0] != "v1" {
            return error(StatusCode::BAD_REQUEST, "context_version", "X-Titanic-Context: v1 is required");
        }
    }
    let wire_limit = if incremental { shared.body_limit * 2 + 65536 } else { shared.body_limit };
    if shared.health.load(Ordering::Acquire) != READY {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "worker_not_ready",
            "native worker is not ready",
        );
    }
    let background = match request_background(request.headers()) {
        Ok(value) => value,
        Err(reason) => return error(StatusCode::BAD_REQUEST, "invalid_priority", &reason.to_string()),
    };
    let background_permit = if background {
        if !shared.decode_priority.load(Ordering::Acquire) {
            return error(StatusCode::BAD_REQUEST, "priority_unavailable", "this worker does not support background requests");
        }
        match shared.background_admission.clone().try_acquire_owned() {
            Ok(permit) => Some(permit),
            Err(_) => return error(StatusCode::TOO_MANY_REQUESTS, "background_queue_full", "foreground admission is reserved"),
        }
    } else { None };
    let Ok(permit) = shared.admission.clone().try_acquire_owned() else {
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "queue_full",
            "native request queue is full",
        );
    };
    if request
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_none_or(|v| v.split(';').next().unwrap_or("").trim() != "application/json")
    {
        return error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_content_type",
            "application/json is required",
        );
    }
    // An available lifetime slot may be claimed before upload, preserving
    // solo recovery from low headroom. A waiting upload cannot ask the model
    // to reclaim cache; its bounded CPU body must fit without GPU work.
    let upload_execution = if shared.execution_activity.load(Ordering::Acquire) {
        shared.execution_admission.clone().try_acquire_owned().ok()
    } else { None };
    let waiting_upload = shared.execution_activity.load(Ordering::Acquire) && upload_execution.is_none();
    let declared = request.headers().get("content-length").and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    let pooled_upload = if waiting_upload {
        declared.and_then(|bytes| shared.waiting_raw_pool.lock().unwrap().as_ref()
            .and_then(|pool| pool.take(bytes)))
    } else { None };
    let upload_workspace = if shared.body_limit == MAX_BODY && pooled_upload.is_none() {
        let bytes = request.headers().get("content-length").and_then(|v|v.to_str().ok())
            .and_then(|v|v.parse::<usize>().ok()).unwrap_or(wire_limit);
        if bytes > wire_limit { return error(StatusCode::PAYLOAD_TOO_LARGE, "body_limit", "request body exceeds its bound"); }
        match if waiting_upload {
            // A waiting upload may reserve bounded CPU memory, but cannot
            // trigger GPU cache reclamation before it owns a lifetime slot.
            context::reserve_after_reclaim(&shared, || crate::memory::reserve_waiting_host_workspace(bytes.max(1)*2,
                shared.waiting_image_budget.load(Ordering::Acquire)))
        } else { reserve_input_workspace(&shared, bytes.max(1)*2).await } {
            Ok(reservation) => Some(reservation),
            Err(_) => return context::capacity_error(incremental, "input upload cannot retain the memory reserve"),
        }
    } else { None };
    let body = match tokio::time::timeout(
        Duration::from_secs(15),
        async {
            match pooled_upload {
                Some(input) => input.read(request.into_body()).await,
                None => to_bytes(request.into_body(), wire_limit).await.map_err(anyhow::Error::from),
            }
        },
    )
    .await
    {
        Ok(Ok(body)) => body,
        Ok(Err(_)) => {
            return error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "body_limit",
                "request body exceeds its bound or is incomplete",
            );
        }
        Err(_) => {
            return error(
                StatusCode::REQUEST_TIMEOUT,
                "body_timeout",
                "request body upload timed out",
            );
        }
    };
    let (body, permit, background_permit, upload_workspace, upload_execution) = if incremental {
        // The upload slot must stay owned while blocking validation runs, even
        // if the client disconnects. No materialization or inference happens here.
        let validation = shared.clone();
        match tokio::task::spawn_blocking(move || {
            let prepared = context::prepare(&validation, &body);
            (prepared, permit, background_permit, upload_workspace, upload_execution)
        }).await {
            Ok((Ok(prepared), permit, background_permit, workspace, execution)) =>
                (context::Input::Incremental(prepared), permit, background_permit, workspace, execution),
            Ok((Err(response), ..)) => return response,
            Err(_) => return error(StatusCode::INTERNAL_SERVER_ERROR, "context_failure", "context preparation failed"),
        }
    } else { (context::Input::Raw(body), permit, background_permit, upload_workspace, upload_execution) };
    let receipt = body.receipt();
    let response = admission::dispatch(shared,body,permit,background_permit,upload_workspace,background,upload_execution).await;
    context::confirm(response, receipt.as_ref())
}

#[allow(clippy::too_many_arguments)]
async fn finish_chat(shared: Arc<Shared>, body: context::Input, permit: OwnedSemaphorePermit,
    background_permit: Option<OwnedSemaphorePermit>,
    upload_workspace: Option<crate::memory::HostWorkspaceReservation>, background: bool,
    execution_permit: Option<OwnedSemaphorePermit>, mut registration: admission::Registered) -> Response {
    if shared.health.load(Ordering::Acquire) != READY || shared.shutdown.load(Ordering::Acquire) {
        return error(StatusCode::SERVICE_UNAVAILABLE, "worker_not_ready", "native worker is stopping");
    }
    // Conversion can simultaneously own the raw body, parsed/normalized
    // request, canonical conversation, rendering and tokenizer buffers. Hold
    // this conservative size-based allowance through the blocking task, even
    // if its HTTP caller disconnects. Decoder scratch has its own reservation.
    let conversion_workspace = if shared.body_limit == MAX_BODY {
        match reserve_input_workspace(&shared, body.expanded_len()*16+(64<<20)).await {
            Ok(reservation) => Some(reservation),
            Err(_) => return error(StatusCode::SERVICE_UNAVAILABLE, "input_capacity", "input conversion cannot retain the memory reserve"),
        }
    } else { None };
    drop(upload_workspace);
    let frontend = shared.frontend.clone();
    let capacity = shared.capacity;
    let cancellation = registration.cancellation.clone();
    let guard = CancelOnDrop(cancellation.clone());
    let preprocessing_cancel = cancellation.clone();
    let preprocessing_shared = shared.clone();
    // A disconnected caller cannot release admission while its blocking
    // conversion still owns a potentially large body/decoder workspace.
    let (mut prepared, permit, background_permit, execution_permit) =
        match tokio::task::spawn_blocking(move || {
            let cancelled = || preprocessing_cancel.load(Ordering::Acquire)
                || preprocessing_shared.shutdown.load(Ordering::Acquire);
            let prepared = body.with_bytes(conversion_workspace, &preprocessing_shared, |body, references| {
                let convert = || match references {
                    None => frontend.prepare_chat_cancellable(body, capacity, &cancelled),
                    Some(references) => frontend.prepare_chat_referenced(body, capacity, references, &cancelled),
                };
                let mut prepared = convert();
                let required = prepared.as_ref().err()
                    .and_then(|error| error.downcast_ref::<crate::memory::InputCapacityError>())
                    .map_or(0, |error| error.1);
                if required > 0 && !cancelled() {
                    // Only the model owner can release CUDA-backed cache. Request
                    // one ordered pass, then retry the ordinary guarded conversion.
                    // Refusal remains explicit when active/paused owners need the
                    // space or the owner cannot service it within this short bound.
                    let generation = preprocessing_shared.input_reclaim_generation.load(Ordering::Acquire);
                    preprocessing_shared.input_reclaim_target.fetch_max(required, Ordering::AcqRel);
                    preprocessing_shared.wake_worker();
                    preprocessing_shared.input_reclaim_ack.wait_sync(
                        &preprocessing_shared.input_reclaim_generation,generation,
                        ||!cancelled()&&preprocessing_shared.health.load(Ordering::Acquire)==READY,
                        Duration::from_secs(5));
                    if !cancelled() && preprocessing_shared.health.load(Ordering::Acquire) == READY
                        && preprocessing_shared.input_reclaim_generation.load(Ordering::Acquire) != generation {
                        prepared = convert();
                    }
                }
                prepared
            });
            (prepared, permit, background_permit, execution_permit)
        }).await {
            Ok((Ok(prepared), permit, background_permit, execution_permit)) => (prepared, permit, background_permit, execution_permit),
            Ok((Err(reason), _, _, _)) => {
                if reason.downcast_ref::<crate::memory::InputCapacityError>().is_some() {
                    return error(StatusCode::SERVICE_UNAVAILABLE, "input_capacity", &reason.to_string());
                }
                return error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    &reason.to_string(),
                );
            }
            Err(_) => {
                return error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "protocol_failure",
                    "request conversion failed",
                );
            }
        };
    if shared.health.load(Ordering::Acquire) != READY || shared.shutdown.load(Ordering::Acquire) {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "worker_not_ready",
            "native worker is stopping",
        );
    }
    let seed = match prepared.converted.inference_options.seed {
        Some(seed) => seed,
        None => match random_bytes::<8>() {
            Ok(bytes) => u64::from_le_bytes(bytes) & ((1u64 << 63) - 1),
            Err(_) => {
                return error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "entropy_failure",
                    "could not initialize the request seed",
                );
            }
        },
    };
    let id = registration.id.clone();
    let (output, incoming) = tokio::sync::mpsc::channel(OUTPUT_CHUNKS);
    let prefill_timing = Arc::new(OnceLock::new());
    let activity_enabled = shared.execution_activity.load(Ordering::Acquire);
    let (activity, activity_rx) = if activity_enabled {
        let (tx, rx) = tokio::sync::watch::channel(false); (Some(tx), Some(rx))
    } else { (None, None) };
    let stream = prepared.converted.stream;
    let prompt_tokens = prepared.tokens.len();

    let job = Job {
        id: id.clone(),
        tokens: prepared.tokens.clone(),
        images: std::mem::take(&mut prepared.images),
        image_inputs: std::mem::take(&mut prepared.image_inputs),
        seed,
        greedy: prepared.converted.inference_options.temperature == Some(0.),
        limit: prepared.max_new_tokens as usize,
        enqueued: registration.enqueued,
        background,
        cancelled: cancellation,
        output,
        prompt_tokens,
        prefill_timing: prefill_timing.clone(),
        output_stalled_since: Cell::new(None),
        activity, executing: Cell::new(false),
        shared: shared.clone(),
        _permit: permit,
        _background_permit: background_permit,
        _execution_permit: execution_permit,
    };
    registration.transferred = true;
    match shared.sender.try_send(job) {
        Ok(()) => shared.wake_worker(),
        Err(mpsc::TrySendError::Full(_)) => {
            return error(
                StatusCode::TOO_MANY_REQUESTS,
                "queue_full",
                "native request queue is full",
            );
        }
        Err(mpsc::TrySendError::Disconnected(_)) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "worker_not_ready",
                "native worker has stopped",
            );
        }
    }
    let completion = shared
        .frontend
        .live_completion(prepared, id.clone(), incoming);
    let completion = fail_on_worker_failure(completion, shared.failure.subscribe());
    let mut response = if stream {
        let events = async_stream::stream! {
            let mut completion=std::pin::pin!(completion);
            let mut timings_sent=false;
            while let Some(chunk)=completion.next().await {
                match chunk {
                    Ok(chunk)=>match if !timings_sent && let Some(timing)=prefill_timing.get() {
                        timings_sent=true;
                        Event::default().json_data(WithTimings {response:chunk,timings:timing})
                    }else{Event::default().json_data(chunk)} {
                        Ok(event)=>yield Ok::<Event,std::io::Error>(event),
                        Err(_)=>{yield Err(std::io::Error::other("response serialization failed"));return;}
                    },
                    Err(reason)=>{
                        // An SSE error is explicit and never followed by a
                        // successful finish or [DONE]. Dropping the body also
                        // cancels any remaining native execution.
                        yield Ok(Event::default().event("error").json_data(
                            json!({"error":{"type":"inference_error","message":reason.to_string()}})).unwrap());
                        return;
                    }
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };
        Sse::new(GuardedStream {
            inner: Box::pin(with_execution_activity(events, activity_rx)),
            _cancel: guard,
        })
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(5)))
        .into_response()
    } else {
        let _guard = guard;
        let mut complete = ChatCompletionResponse::new(
            id.clone(),
            MODEL_NAME.into(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            prompt_tokens,
            0,
        );
        let mut completion = std::pin::pin!(completion);
        while let Some(chunk) = completion.next().await {
            match chunk {
                Ok(chunk) => complete.accumulate(chunk),
                Err(reason) => {
                    return error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "inference_error",
                        &reason.to_string(),
                    );
                }
            }
        }
        if let Some(timing)=prefill_timing.get() {
            Json(WithTimings {response:complete,timings:timing}).into_response()
        }else{Json(complete).into_response()}
    };
    if activity_enabled {
        response.headers_mut().insert("x-native-execution-timing", HeaderValue::from_static("v1"));
    }
    response
        .headers_mut()
        .insert("x-request-id", HeaderValue::from_str(&id).unwrap());
    response
        .headers_mut()
        .insert("cache-control", HeaderValue::from_static("no-store"));
    response
}

pub(crate) fn start(config: &ServerConfig, capacity: usize, images: bool) -> Result<Queue> {
    config.validate_listener()?;
    let authorization = auth_digest(&config.bearer_file)?;
    let priority_authorization = config.priority_file.as_ref().map(|p|auth_digest(p)).transpose()?;
    ensure!(priority_authorization != Some(authorization), "scheduler and inference keys must be distinct");
    let frontend = Arc::new(Frontend::load_with_images(&config.tokenizer, images)?);
    let listener = std::net::TcpListener::bind(config.bind)?;
    // Bounded fault injection for real TCP backpressure qualification. Accepted
    // sockets inherit this Linux option. Production uses the OS default.
    #[cfg(target_os = "linux")]
    if let Ok(raw) = std::env::var("CANADA_SCRATCH_HTTP_SNDBUF") {
        use std::os::fd::AsRawFd;
        let bytes = raw.parse::<libc::c_int>()?;
        ensure!(config.bind.ip().is_loopback() && (4096..=65536).contains(&bytes),
            "bounded HTTP send-buffer probe requires loopback and 4..64 KiB");
        let result = unsafe {libc::setsockopt(listener.as_raw_fd(), libc::SOL_SOCKET, libc::SO_SNDBUF,
            (&bytes as *const libc::c_int).cast(), std::mem::size_of_val(&bytes) as libc::socklen_t)};
        ensure!(result == 0, "HTTP send-buffer probe setup failed: {}", std::io::Error::last_os_error());
    }
    listener.set_nonblocking(true)?;
    let (sender, receiver) = mpsc::sync_channel(QUEUE_DEPTH + 1);
    let (wake_sender, wake_receiver) = mpsc::sync_channel(1);
    let shared = Arc::new(Shared {
        sender,
        wake_sender,
        frontend,
        capacity,
        body_limit: if images { MAX_BODY } else { 8 << 20 },
        admission: Arc::new(Semaphore::new(QUEUE_DEPTH + 1)),
        background_admission: Arc::new(Semaphore::new(QUEUE_DEPTH)),
        decode_priority: AtomicBool::new(false),
        background_prefill: AtomicBool::new(false),
        execution_activity: AtomicBool::new(false),
        execution_admission: Arc::new(Semaphore::new(2)),
        waiting_inputs: AtomicUsize::new(0),
        waiting_image_budget: AtomicUsize::new(0),
        waiting_raw_pool: Mutex::new(None),
        context_cache: (config.context_cache_mib > 0).then(|| Mutex::new(crate::context_transport::Cache::new(
            config.context_cache_mib << 20, 16, 30 * 60 * 1000))),
        health: AtomicU8::new(LOADING),
        progress: Arc::new(AtomicU64::new(0)),
        progress_gap_ms: AtomicU64::new(0),
        progress_maximum_gap_ms: AtomicU64::new(0),
        progress_timed_out: AtomicBool::new(false),
        failure: tokio::sync::watch::channel(false).0,
        shutdown: AtomicBool::new(false),
        notify_shutdown: Notify::new(),
        identity: Mutex::new(None),
        requests: Mutex::new(HashMap::new()),
        authorization,
        nonce: random_bytes::<12>()?
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
        sequence: AtomicU64::new(1),
        priority: priority_authorization.map(|_|crate::router_priority::Control::new()),
        priority_authorization,
        control_admission: Semaphore::new(8),
        output_stall_timeout: Duration::from_secs(config.output_stall_seconds),
        output_backpressure_events: AtomicU64::new(0),
        output_stall_timeouts: AtomicU64::new(0),
        input_reclaim_target: AtomicUsize::new(0),
        input_reclaim_generation: AtomicU64::new(0),
        input_reclaim_ack: reclaim_ack::Ack::default(),
    });
    let serving = shared.clone();
    std::thread::Builder::new()
        .name("native-http".into())
        .spawn(move || {
            let result = (|| -> Result<()> {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .max_blocking_threads(2)
                    .enable_all()
                    .build()?;
                runtime.block_on(async {
                    let listener = tokio::net::TcpListener::from_std(listener)?;
                    let clock = serving.clone();
                    tokio::spawn(async move {
                        let mut progress = ProgressWatch::default();
                        while !clock.shutdown.load(Ordering::Acquire) {
                            let gap = progress.observe(clock.health.load(Ordering::Acquire) == READY,
                                clock.progress.load(Ordering::Acquire), Instant::now());
                            clock.progress_gap_ms.store(gap.as_millis() as u64, Ordering::Relaxed);
                            clock.progress_maximum_gap_ms.store(progress.maximum_gap.as_millis() as u64, Ordering::Relaxed);
                            if gap >= WORKER_PROGRESS_TIMEOUT {
                                clock.progress_timed_out.store(true, Ordering::Release);
                                eprintln!("native worker completed no control exchange for 60 seconds; retiring group");
                                clock.fail(); break;
                            }
                            if clock.priority.as_ref().is_some_and(|p|p.tick().is_err()) {
                                clock.fail(); break;
                            }
                            tokio::time::sleep(Duration::from_millis(100)).await;
                        }
                    });
                    let app = Router::new()
                        .route("/health", get(health))
                        .route("/_native/diagnostics", get(diagnostics))
                        .route("/v1/models", get(models))
                        .route("/v1/chat/completions", post(chat))
                        .route("/v1/context/chat/completions", post(context_chat))
                        .route("/v1/requests/{id}/cancel", post(cancel))
                        .route("/_native/shutdown", post(shutdown))
                        .route("/_native/priority", post(priority_control))
                        .with_state(serving.clone());
                    let stop = serving.clone();
                    axum::serve(listener, app)
                        .with_graceful_shutdown(async move {
                            stop.notify_shutdown.notified().await;
                        })
                        .await?;
                    Ok::<(), anyhow::Error>(())
                })
            })();
            if result.is_err() {
                serving.fail();
                eprintln!("native HTTP listener failed");
            }
        })?;
    Ok(Queue { receiver, wake_receiver, pending: RefCell::new(VecDeque::new()), execution_owners: Cell::new(None), prefill_peer_ready:Cell::new(false),
            work_aware_prefill: Cell::new(false),
            prefill_peer_work: Cell::new(None),
            prefill_protected_until: Cell::new(None), decode_prefill_slice: Cell::new(None), shared })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn work_aware_prefill_preserves_hysteresis_and_decoder_slices() {
        assert!(!prefill_work_yield(100,PrefillPeer::Prompt(100),true));
        assert!(!prefill_work_yield(100,PrefillPeer::Prompt(50),true));
        assert!(prefill_work_yield(100,PrefillPeer::Prompt(49),false));
        assert!(!prefill_work_yield(0,PrefillPeer::Prompt(0),true));
        assert!(!prefill_work_yield(usize::MAX,PrefillPeer::Prompt(usize::MAX),true));
        assert!(!prefill_work_yield(100,PrefillPeer::Decode,false));
        assert!(prefill_work_yield(100,PrefillPeer::Decode,true));
    }
    #[test]
    fn shorter_prefill_turns_apply_only_to_a_waiting_decoder() {
        let normal=Duration::from_secs(12);let short=Duration::from_secs(2);
        assert!(prefill_turn_due(short,normal,Some(PrefillPeer::Decode),Some(short)));
        assert!(!prefill_turn_due(short-Duration::from_millis(1),normal,Some(PrefillPeer::Decode),Some(short)));
        assert!(!prefill_turn_due(short,normal,Some(PrefillPeer::Prompt(1)),Some(short)));
        assert!(!prefill_turn_due(short,normal,None,Some(short)));
        assert!(!prefill_turn_due(short,normal,Some(PrefillPeer::Decode),None));
        assert!(prefill_turn_due(normal,normal,Some(PrefillPeer::Decode),Some(normal*2)));
    }
    use std::io::Write;
    #[test]
    fn progress_watch_ignores_loading_and_requires_completed_work() {
        let now = Instant::now(); let mut watch = ProgressWatch::default();
        assert_eq!(watch.observe(false, 0, now + Duration::from_secs(600)), Duration::ZERO);
        assert_eq!(watch.observe(true, 17, now), Duration::ZERO);
        // HTTP polling and output keepalives cannot change the control sequence.
        assert_eq!(watch.observe(true, 17, now + WORKER_PROGRESS_TIMEOUT), WORKER_PROGRESS_TIMEOUT);
        assert_eq!(watch.observe(true, 18, now + WORKER_PROGRESS_TIMEOUT), Duration::ZERO);
        assert_eq!(watch.observe(true, 18, now + WORKER_PROGRESS_TIMEOUT + Duration::from_secs(1)), Duration::from_secs(1));
        assert_eq!(watch.maximum_gap, WORKER_PROGRESS_TIMEOUT);
        watch.observe(false, 18, now + Duration::from_secs(120));
        assert_eq!(watch.observe(true, 18, now + Duration::from_secs(900)), Duration::ZERO);
    }
    #[test]
    fn group_failure_wakes_open_stream_and_rejects_late_subscriber() {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let (failure, rx) = tokio::sync::watch::channel(false);
            let stream = fail_on_worker_failure(tokio_stream::pending::<Result<u8>>(), rx);
            let mut stream = std::pin::pin!(stream);
            let fail = async { tokio::task::yield_now().await; failure.send_replace(true); };
            let (item, ()) = tokio::join!(stream.next(), fail);
            assert!(item.unwrap().is_err()); assert!(stream.next().await.is_none());
            let stream = fail_on_worker_failure(tokio_stream::iter([Ok(1u8)]), failure.subscribe());
            let mut stream = std::pin::pin!(stream);
            assert!(stream.next().await.unwrap().is_err()); assert!(stream.next().await.is_none());
        });
    }
    #[test]
    fn healthy_completion_is_unchanged_by_failure_guard() {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let (_failure, rx) = tokio::sync::watch::channel(false);
            let stream = fail_on_worker_failure(tokio_stream::iter([Ok(7u8), Ok(9)]), rx);
            let mut stream = std::pin::pin!(stream);
            assert_eq!(stream.next().await.unwrap().unwrap(), 7);
            assert_eq!(stream.next().await.unwrap().unwrap(), 9);
            assert!(stream.next().await.is_none());
        });
    }
    #[test]
    fn execution_activity_comments_preserve_transitions_and_final_data() {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            async fn text(event: Event) -> String {
                let body = Sse::new(tokio_stream::iter([Ok::<_, std::io::Error>(event)]))
                    .into_response().into_body();
                String::from_utf8(to_bytes(body, 1024).await.unwrap().to_vec()).unwrap()
            }
            let (activity, rx) = tokio::sync::watch::channel(false);
            let (output, incoming) = tokio::sync::mpsc::channel(2);
            let source = tokio_stream::wrappers::ReceiverStream::new(incoming);
            let mut stream = std::pin::pin!(with_execution_activity(source, Some(rx)));
            assert!(text(stream.next().await.unwrap().unwrap()).await.contains("native-execution waiting"));
            activity.send_replace(true);
            assert!(text(stream.next().await.unwrap().unwrap()).await.contains("native-execution running"));
            activity.send_replace(false);
            assert!(text(stream.next().await.unwrap().unwrap()).await.contains("native-execution waiting"));
            output.send(Ok(Event::default().data("[DONE]"))).await.unwrap();
            drop(activity); drop(output);
            assert!(text(stream.next().await.unwrap().unwrap()).await.contains("data: [DONE]"));
            assert!(stream.next().await.is_none());
            let source = tokio_stream::iter([Ok(Event::default().data("unchanged"))]);
            let mut legacy = std::pin::pin!(with_execution_activity(source, None));
            assert_eq!(text(legacy.next().await.unwrap().unwrap()).await, "data: unchanged\n\n");
            assert!(legacy.next().await.is_none());
        });
    }
    #[test]
    fn foreground_is_default_and_ambiguous_modes_are_rejected() {
        let mut headers=HeaderMap::new();
        assert!(!request_background(&headers).unwrap());
        headers.insert("x-localai-mode",HeaderValue::from_static(" BACKGROUND "));
        assert!(request_background(&headers).unwrap());
        headers.append("x-localai-mode",HeaderValue::from_static("foreground"));
        assert!(request_background(&headers).is_err());
        headers.insert("x-localai-mode",HeaderValue::from_static("urgent"));
        assert!(request_background(&headers).is_err());
    }
    #[test]
    #[ignore = "finite latency screen; requires CANADA_TEST_TOKENIZER"]
    fn idle_reclaim_latency_screen() {
        let (queue, shared, _rt) = test_http_fixture();
        for delay_ms in [5, 37, 149, 5, 37, 149] {
            let shared = shared.clone();
            let (start, started) = mpsc::sync_channel(0);
            let client = std::thread::spawn(move || {
                started.recv().unwrap();
                std::thread::sleep(Duration::from_millis(delay_ms));
                let at = Instant::now();
                let attempts = Cell::new(0);
                let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
                let result = rt.block_on(reserve_with_input_reclaim(|| {
                    attempts.set(attempts.get() + 1);
                    if attempts.get() == 1 {
                        Err(crate::memory::InputCapacityError("latency screen", 8192).into())
                    } else { Ok(17) }
                }, &shared.input_reclaim_target, &shared.input_reclaim_generation, &shared.input_reclaim_ack,
                    || true, || shared.wake_worker(), Duration::from_secs(2)));
                assert_eq!(result.unwrap(), 17);
                (at, at.elapsed().as_secs_f64() * 1000.)
            });
            start.send(()).unwrap();
            assert!(queue.next_fifo(Duration::from_millis(500)).unwrap().is_none());
            let observed = Instant::now();
            assert_eq!(queue.input_reclaim_target(), 8192);
            queue.input_reclaimed(8192);
            let (arrival, client_ms) = client.join().unwrap();
            println!("{}", serde_json::json!({"screen":"idle_reclaim", "delay_ms":delay_ms,
                "owner_notice_ms":observed.duration_since(arrival).as_secs_f64()*1000.,
                "client_ms":client_ms}));
        }
        // Coalesce many notifications, including one posted before waiting.
        // A partial acknowledgement must not lose a larger pending request.
        shared.input_reclaim_target.store(32768, Ordering::Release);
        for _ in 0..4096 { shared.wake_worker(); }
        let at = Instant::now();
        assert!(queue.next_fifo(Duration::from_secs(1)).unwrap().is_none());
        assert!(at.elapsed() < Duration::from_millis(200));
        queue.input_reclaimed(8192);
        assert_eq!(shared.input_reclaim_target.load(Ordering::Acquire), 32768);
        queue.input_reclaimed(32768);
        assert_eq!(shared.input_reclaim_target.load(Ordering::Acquire), 0);
        let at = Instant::now();
        assert!(queue.next_fifo(Duration::from_millis(5)).unwrap().is_none());
        assert!(at.elapsed() >= Duration::from_millis(5), "stale wake returned spurious idle work");
    }
    #[test]
    fn input_workspace_reclaims_once_then_rechecks_the_unchanged_reservation() {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let target = AtomicUsize::new(0);
        let generation = AtomicU64::new(0);
        let ack = reclaim_ack::Ack::default();
        let attempts = Cell::new(0);
        let capacity = Cell::new(false);
        rt.block_on(async {
            let admission = reserve_with_input_reclaim(|| {
                attempts.set(attempts.get() + 1);
                if capacity.get() { Ok(17) } else {
                    Err(crate::memory::InputCapacityError("reserve unchanged", 8192).into())
                }
            }, &target, &generation, &ack, || true, || {}, Duration::from_secs(1));
            let owner = async {
                tokio::task::yield_now().await;
                assert_eq!(target.load(Ordering::Acquire), 8192);
                capacity.set(true);
                ack.publish(&generation);
            };
            let (result, ()) = tokio::join!(admission, owner);
            assert_eq!(result.unwrap(), 17);
        });
        assert_eq!(attempts.get(), 2);
        // Success and fixed per-input limits must not request GPU work.
        target.store(0, Ordering::Release);
        assert_eq!(rt.block_on(reserve_with_input_reclaim(|| Ok(23), &target,
            &generation, &ack, || true, || {}, Duration::ZERO)).unwrap(), 23);
        let fixed: Result<()> = rt.block_on(reserve_with_input_reclaim(||
            Err(crate::memory::InputCapacityError("fixed input cap", 0).into()),
            &target, &generation, &ack, || true, || {}, Duration::ZERO));
        assert!(fixed.is_err());
        assert_eq!(target.load(Ordering::Acquire), 0);
        // No owner response: refuse rather than allocate or retry indefinitely.
        attempts.set(0);
        target.store(16384, Ordering::Release);
        let unavailable: Result<()> = rt.block_on(reserve_with_input_reclaim(|| {
            attempts.set(attempts.get() + 1);
            Err(crate::memory::InputCapacityError("reserve unchanged", 8192).into())
        }, &target, &generation, &ack, || true, || {}, Duration::ZERO));
        assert!(unavailable.is_err());
        assert_eq!(attempts.get(), 1);
        assert_eq!(target.load(Ordering::Acquire), 16384);
    }
    #[test]
    fn dropping_an_unpolled_response_cancels_without_gpu_work() {
        let flag = Arc::new(AtomicBool::new(false));
        let stream = GuardedStream {
            inner: Box::pin(tokio_stream::pending::<()>()),
            _cancel: CancelOnDrop(flag.clone()),
        };
        assert!(!flag.load(Ordering::Acquire));
        drop(stream);
        assert!(flag.load(Ordering::Acquire));
    }
    pub(super) fn test_http_fixture() -> (Queue, Arc<Shared>, tokio::runtime::Runtime) {test_http_fixture_images(false)}
    pub(super) fn test_http_fixture_images(images:bool) -> (Queue, Arc<Shared>, tokio::runtime::Runtime) {
        test_http_fixture_limits(images, 8 << 20)
    }
    pub(super) fn test_http_fixture_limits(images:bool, body_limit:usize) -> (Queue, Arc<Shared>, tokio::runtime::Runtime) {
        let mut frontend=Frontend::load(Path::new(&std::env::var("CANADA_TEST_TOKENIZER").unwrap())).unwrap();
        if images {frontend.enable_test_images();}
        let frontend=Arc::new(frontend);
        let (sender, receiver) = mpsc::sync_channel(QUEUE_DEPTH + 1);
        let (wake_sender, wake_receiver) = mpsc::sync_channel(1);
        let shared = Arc::new(Shared {
            sender,
            wake_sender,
            frontend,
            capacity: 4096,
            body_limit,
            admission: Arc::new(Semaphore::new(QUEUE_DEPTH + 1)),
            background_admission: Arc::new(Semaphore::new(QUEUE_DEPTH)),
            decode_priority: AtomicBool::new(false),
            background_prefill: AtomicBool::new(false),
            execution_activity: AtomicBool::new(false),
            execution_admission: Arc::new(Semaphore::new(2)),
            waiting_inputs: AtomicUsize::new(0),
            waiting_image_budget: AtomicUsize::new(0),
            waiting_raw_pool: Mutex::new(None),
            context_cache: Some(Mutex::new(crate::context_transport::Cache::new(1 << 20, 16, 30 * 60 * 1000))),
            health: AtomicU8::new(READY),
            progress: Arc::new(AtomicU64::new(0)),
            progress_gap_ms: AtomicU64::new(0),
            progress_maximum_gap_ms: AtomicU64::new(0),
            progress_timed_out: AtomicBool::new(false),
            failure: tokio::sync::watch::channel(false).0,
            shutdown: AtomicBool::new(false),
            notify_shutdown: Notify::new(),
            identity: Mutex::new(None),
            requests: Mutex::new(HashMap::new()),
            authorization: Sha256::digest(b"Bearer test-token").into(),
            priority: None,
            priority_authorization: None,
            control_admission: Semaphore::new(8),
            output_stall_timeout: Duration::from_secs(60),
            output_backpressure_events: AtomicU64::new(0),
            output_stall_timeouts: AtomicU64::new(0),
            input_reclaim_target: AtomicUsize::new(0),
            input_reclaim_generation: AtomicU64::new(0),
            input_reclaim_ack: reclaim_ack::Ack::default(),
            nonce: "test".into(),
            sequence: AtomicU64::new(1),
        });
        let queue = Queue {
            receiver,
            wake_receiver,
            pending: RefCell::new(VecDeque::new()),
            execution_owners: Cell::new(None),
            prefill_peer_ready: Cell::new(false),
            work_aware_prefill: Cell::new(false),
            prefill_peer_work: Cell::new(None),
            prefill_protected_until: Cell::new(None),
            decode_prefill_slice: Cell::new(None),
            shared: shared.clone(),
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        (queue,shared,rt)
    }
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn admission_cancel_backpressure_and_bare_eof_are_bounded() {
        let (queue,shared,rt)=test_http_fixture();
        let request = || {
            Request::builder().method("POST").uri("/v1/chat/completions")
            .header("authorization","Bearer test-token").header("content-type","application/json")
            .body(Body::from(r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":64,"seed":0,"stream":true}"#)).unwrap()
        };
        let first = rt.block_on(chat(State(shared.clone()), request()));
        assert_eq!(first.status(), StatusCode::OK);
        let job = queue.try_next().unwrap().unwrap();
        assert!(queue.try_next().unwrap().is_none());
        assert_eq!(job.seed, 0);
        assert_eq!(job.limit, 64);
        let mut queued = Vec::new();
        for _ in 0..QUEUE_DEPTH {
            let response = rt.block_on(chat(State(shared.clone()), request()));
            assert_eq!(response.status(), StatusCode::OK);
            queued.push(response);
        }
        let extra = rt.block_on(chat(State(shared.clone()), request()));
        assert_eq!(extra.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(shared.requests.lock().unwrap().len(), QUEUE_DEPTH + 1);
        for _ in 0..OUTPUT_CHUNKS {
            job.emit(BackendEvent::Tokens(vec![42]));
        }
        assert!(!job.cancelled());
        job.emit(BackendEvent::Tokens(vec![42]));
        assert!(job.cancelled());
        drop(first);
        drop(job);
        drop(queued);
        assert!(queue.try_next().unwrap().is_none());
        assert_eq!(shared.requests.lock().unwrap().len(), 0);
        assert_eq!(shared.admission.available_permits(), QUEUE_DEPTH + 1);

        // A brief stall keeps every accepted event; draining the actual
        // bounded channel clears the deadline. Only a continuous full period
        // expires, independently of how many scheduler polls occurred.
        let response = rt.block_on(chat(State(shared.clone()), request()));
        let mut job = queue.try_next().unwrap().unwrap();
        let (output, mut incoming) = tokio::sync::mpsc::channel(4);
        job.output = output;
        let now = Instant::now();
        for id in [41,42] {job.emit(BackendEvent::Tokens(vec![id]));}
        assert!(job.output_backpressured_at(now));
        assert!(!job.cancelled());
        assert!(matches!(incoming.try_recv().unwrap(),BackendEvent::Tokens(ids) if ids==[41]));
        assert!(!job.output_backpressured_at(now+Duration::from_secs(59)));
        job.emit(BackendEvent::Tokens(vec![43]));
        assert!(job.output_backpressured_at(now+Duration::from_secs(60)));
        assert!(job.output_backpressured_at(now+Duration::from_secs(119)));
        assert!(!job.cancelled());
        assert!(!job.output_backpressured_at(now+Duration::from_secs(120)));
        assert!(job.cancelled());
        assert_eq!(shared.output_backpressure_events.load(Ordering::Relaxed),2);
        assert_eq!(shared.output_stall_timeouts.load(Ordering::Relaxed),1);
        for id in [42,43] {
            assert!(matches!(incoming.try_recv().unwrap(),BackendEvent::Tokens(ids) if ids==[id]));
        }
        drop(job);drop(response);
        assert!(queue.try_next().unwrap().is_none());

        let response = rt.block_on(chat(State(shared.clone()), request()));
        let job = queue.next(Duration::from_millis(10)).unwrap().unwrap();
        job.emit(BackendEvent::Ready {
            fingerprint: "test".into(),
            cached_tokens: 0,
        });
        job.emit(BackendEvent::Tokens(vec![42]));
        drop(job);
        let body = rt
            .block_on(to_bytes(response.into_body(), 1024 * 1024))
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("event: error"));
        assert!(!body.contains("[DONE]"));
        assert!(!body.contains("\"finish_reason\":\"stop\""));

        let background_request = || {
            let mut r=request();
            r.headers_mut().insert("x-localai-mode",HeaderValue::from_static("background"));
            r
        };
        assert_eq!(rt.block_on(chat(State(shared.clone()),background_request())).status(),StatusCode::BAD_REQUEST);
        queue.enable_decode_priority();
        let mut responses=Vec::new();
        for _ in 0..QUEUE_DEPTH {
            let r=rt.block_on(chat(State(shared.clone()),background_request()));
            assert_eq!(r.status(),StatusCode::OK); responses.push(r);
        }
        assert_eq!(rt.block_on(chat(State(shared.clone()),background_request())).status(),StatusCode::TOO_MANY_REQUESTS);
        let foreground=rt.block_on(chat(State(shared.clone()),request()));
        assert_eq!(foreground.status(),StatusCode::OK,"background queue must leave foreground capacity");
        let job=queue.try_next().unwrap().unwrap();
        assert!(!job.background,"later foreground must bypass earlier background peers");
        assert!(queue.try_foreground().unwrap().is_none());
        drop(job);drop(foreground);
        let first=queue.try_next().unwrap().unwrap();
        let second=queue.try_next().unwrap().unwrap();
        assert!(first.background&&second.background);
        assert!(first.enqueued<=second.enqueued,"equal priority remains FIFO at admission");
        drop(first);drop(second);drop(responses);
        assert!(queue.try_next().unwrap().is_none());
        assert_eq!(shared.admission.available_permits(),QUEUE_DEPTH + 1);
        assert_eq!(shared.background_admission.available_permits(),QUEUE_DEPTH);
        assert!(shared.requests.lock().unwrap().is_empty());

        // FIFO execution admission is deliberately different from the
        // legacy foreground-priority dequeue above. A later foreground may
        // win execution only after it owns one of the two slots.
        let bg1=rt.block_on(chat(State(shared.clone()),background_request()));
        let bg2=rt.block_on(chat(State(shared.clone()),background_request()));
        let fg=rt.block_on(chat(State(shared.clone()),request()));
        queue.enable_two_stream_admission();
        queue.set_execution_owners(2).unwrap();
        assert!(!queue.foreground_can_join().unwrap(), "queued foreground cannot interrupt two owners");
        assert!(!queue.prefill_peer_can_run(false).unwrap(),"third request cannot interrupt foreground prefill");
        assert!(!queue.prefill_peer_can_run(true).unwrap(),"third foreground cannot interrupt two background owners");
        queue.set_prefill_work(true,None,None);
        assert!(!queue.prefill_peer_should_yield(true,100,true,|_|1).unwrap(),"work estimates cannot admit a third owner");
        queue.set_prefill_work(true,Some(PrefillPeer::Prompt(49)),None);
        assert!(queue.prefill_peer_should_yield(false,100,false,|_|100).unwrap());
        queue.set_prefill_work(true,Some(PrefillPeer::Prompt(49)),Some(Instant::now()+Duration::from_secs(12)));
        assert!(!queue.prefill_peer_should_yield(false,100,true,|_|1).unwrap(),"repay an interrupted owner's turn");
        queue.set_prefill_work(false,None,None);
        queue.set_prefill_peer_ready(true);
        assert!(queue.prefill_peer_can_run(false).unwrap(),"admitted runnable peer can receive an encoder boundary");
        queue.set_prefill_peer_ready(false);
        queue.set_execution_owners(1).unwrap();
        assert!(queue.foreground_can_join().unwrap());
        assert!(!queue.prefill_peer_can_run(false).unwrap(),"foreground must not yield for a background FIFO head");
        assert!(queue.prefill_peer_can_run(true).unwrap(),"background peer can fill the vacant slot");
        queue.set_prefill_work(true,None,None);
        assert!(!queue.prefill_peer_should_yield(false,100,true,|_|1).unwrap(),"background FIFO head cannot interrupt foreground");
        assert!(queue.prefill_peer_should_yield(true,100,false,|_|1).unwrap());
        queue.set_prefill_work(false,None,None);
        let cached=queue.pending.borrow().front().unwrap().tokens.clone();
        assert!(queue.pending_needs_prefix(&cached,&crate::prompt_media::Identity::default()).unwrap());
        let mut different=cached.clone();different.push(129279);
        assert!(!queue.pending_needs_prefix(&different,&crate::prompt_media::Identity::default()).unwrap());
        let first=queue.try_fifo().unwrap().unwrap();
        let second=queue.try_fifo().unwrap().unwrap();
        let third=queue.try_fifo().unwrap().unwrap();
        assert!(first.background&&second.background&&!third.background);
        assert!(first.enqueued<=second.enqueued&&second.enqueued<=third.enqueued);
        drop((first,second,third,bg1,bg2,fg));
        let cancelled=rt.block_on(chat(State(shared.clone()),background_request()));
        let live=rt.block_on(chat(State(shared.clone()),request()));
        drop(cancelled);
        let job=queue.try_fifo().unwrap().unwrap();assert!(!job.background);
        assert!(queue.try_fifo().unwrap().is_none());drop((job,live));
        assert!(shared.requests.lock().unwrap().is_empty());
        queue.enable_two_stream_admission();
        let response=rt.block_on(chat(State(shared.clone()),request()));
        assert_eq!(response.headers()["x-native-execution-timing"], "v1");
        let job=queue.try_fifo().unwrap().unwrap();
        job.set_executing(true);
        job.emit(BackendEvent::Ready{fingerprint:"test".into(),cached_tokens:0});
        job.emit(BackendEvent::Tokens(vec![42,1]));
        job.emit(BackendEvent::Finish(deepseek_recipe::stream::InferenceFinishReason::Stop));
        drop(job);
        let body=String::from_utf8(rt.block_on(to_bytes(response.into_body(),1024*1024)).unwrap().to_vec()).unwrap();
        assert!(body.contains("native-execution waiting"));
        assert!(body.contains("[DONE]"), "{body}");
        assert!(body.contains("\"finish_reason\":\"stop\""));
        assert!(shared.requests.lock().unwrap().is_empty());
    }
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn raw_fifo_waits_without_conversion_and_releases_only_terminal_slots() {
        let (queue,shared,rt)=test_http_fixture();
        queue.enable_decode_priority();queue.enable_two_stream_admission();
        assert!(!queue.input_pending_for_vacancy(),"idle serving does not wait for work");
        let request=|stream:bool,valid:bool,background:bool| {
            let messages=if valid {json!([{"role":"user","content":"hi"}])} else {json!(42)};
            Request::builder().method("POST").uri("/v1/chat/completions")
                .header("authorization","Bearer test-token").header("content-type","application/json")
                .header("x-localai-mode",if background {"background"}else{"foreground"})
                .body(Body::from(json!({"messages":messages,"max_tokens":16,"stream":stream}).to_string())).unwrap()
        };
        let finish=|job:Job| {
            job.emit(BackendEvent::Ready{fingerprint:"test".into(),cached_tokens:0});
            job.emit(BackendEvent::Tokens(vec![42,1]));
            job.emit(BackendEvent::Finish(deepseek_recipe::stream::InferenceFinishReason::Stop));
            drop(job);
        };
        let text=|response:Response|String::from_utf8(rt.block_on(to_bytes(response.into_body(),1<<20)).unwrap().to_vec()).unwrap();
        let a=rt.block_on(chat(State(shared.clone()),request(true,true,true)));
        let b=rt.block_on(chat(State(shared.clone()),request(true,true,true)));
        let first=queue.try_fifo().unwrap().unwrap();let second=queue.try_fifo().unwrap().unwrap();
        assert!(first.background&&second.background);
        assert_eq!(shared.execution_admission.available_permits(),0);
        queue.set_execution_owners(2).unwrap();

        // A third foreground request is acknowledged but not converted or
        // dispatched. It cannot borrow either paused owner's permit.
        let third=rt.block_on(chat(State(shared.clone()),request(true,true,false)));
        let third_id=third.headers()["x-request-id"].to_str().unwrap().to_owned();
        assert_eq!(shared.waiting_inputs.load(Ordering::Acquire),1);
        assert!(!queue.input_pending_for_vacancy(),"a full pair has no vacancy to wait for");
        assert!(queue.try_fifo().unwrap().is_none());
        assert!(queue.has_outstanding_requests());
        assert!(queue.pending_needs_prefix(&[129279],&crate::prompt_media::Identity::default()).unwrap(),
            "unknown queued prefixes conservatively protect cache");
        first.set_executing(false);second.set_executing(false);
        assert_eq!(shared.execution_admission.available_permits(),0);
        assert_eq!(shared.input_reclaim_target.load(Ordering::Acquire),0);
        let mut body=third.into_body().into_data_stream();
        let first_chunk=rt.block_on(body.next()).unwrap().unwrap();
        assert!(std::str::from_utf8(&first_chunk).unwrap().contains("native-execution waiting"));
        assert!(queue.try_fifo().unwrap().is_none());
        finish(first);
        queue.set_execution_owners(1).unwrap();
        assert!(queue.input_pending_for_vacancy(),"terminal release exposes the queued FIFO entrant");
        rt.block_on(async {tokio::time::timeout(Duration::from_secs(2),body.next()).await}).unwrap().unwrap().unwrap();
        let admitted=queue.try_fifo().unwrap().unwrap();assert_eq!(admitted.id,third_id);
        assert!(!admitted.background);assert_eq!(shared.waiting_inputs.load(Ordering::Acquire),0);
        assert_eq!(shared.execution_admission.available_permits(),0);
        assert!(queue.input_pending_for_vacancy(),"claimed lifetime permit bridges conversion and model admission");
        queue.set_execution_owners(2).unwrap();assert!(!queue.input_pending_for_vacancy());
        finish(second);finish(admitted);
        let remaining=rt.block_on(async {let mut out=Vec::new();while let Some(c)=body.next().await{out.extend_from_slice(&c.unwrap());}out});
        assert!(std::str::from_utf8(&remaining).unwrap().contains("[DONE]"));
        assert!(text(a).contains("[DONE]")&&text(b).contains("[DONE]"));
        assert_eq!(shared.execution_admission.available_permits(),2);
        queue.set_execution_owners(0).unwrap();assert!(!queue.input_pending_for_vacancy());

        // Queued cancellation works by request ID, even before tokenization.
        let a=rt.block_on(chat(State(shared.clone()),request(true,true,false)));
        let b=rt.block_on(chat(State(shared.clone()),request(true,true,false)));
        let first=queue.try_fifo().unwrap().unwrap();let second=queue.try_fifo().unwrap().unwrap();
        let cancelled=rt.block_on(chat(State(shared.clone()),request(true,true,false)));
        let id=cancelled.headers()["x-request-id"].to_str().unwrap().to_owned();
        let mut headers=HeaderMap::new();headers.insert("authorization",HeaderValue::from_static("Bearer test-token"));
        assert_eq!(rt.block_on(cancel(State(shared.clone()),RoutePath(id.clone()),headers)).status(),StatusCode::OK);
        let cancelled=text(cancelled);
        assert!(cancelled.contains("request_cancelled")&&!cancelled.contains("[DONE]"));
        assert_eq!(shared.waiting_inputs.load(Ordering::Acquire),0);
        assert!(!shared.requests.lock().unwrap().contains_key(&id));
        assert!(queue.try_fifo().unwrap().is_none());
        assert_eq!(shared.execution_admission.available_permits(),0);
        let abandoned=rt.block_on(chat(State(shared.clone()),request(true,true,false)));
        drop(abandoned);
        assert_eq!(shared.waiting_inputs.load(Ordering::Acquire),0,"unpolled queued body releases its ticket");
        assert_eq!(shared.requests.lock().unwrap().len(),2);

        // Invalid message conversion happens only after admission. A late
        // error is explicit SSE and cannot masquerade as successful output.
        let invalid=rt.block_on(chat(State(shared.clone()),request(true,false,false)));
        assert_eq!(invalid.status(),StatusCode::OK);
        assert_eq!(shared.waiting_inputs.load(Ordering::Acquire),1);
        assert!(queue.try_fifo().unwrap().is_none());
        finish(first);
        let invalid=text(invalid);assert!(invalid.contains("event: error")&&invalid.contains("invalid_request"));
        assert!(!invalid.contains("[DONE]"));
        assert!(queue.try_fifo().unwrap().is_none());
        assert_eq!(shared.execution_admission.available_permits(),1);
        let invalid=rt.block_on(chat(State(shared.clone()),request(true,false,false)));
        assert_eq!(invalid.status(),StatusCode::BAD_REQUEST,"uncontended validation preserves HTTP status");
        finish(second);drop((a,b));
        assert!(shared.requests.lock().unwrap().is_empty());
        assert_eq!(shared.admission.available_permits(),QUEUE_DEPTH+1);
        assert_eq!(shared.execution_admission.available_permits(),2);

        // Direct nonstream calls wait too, while preserving their final HTTP
        // status and ordinary JSON response. No cancellation/replay is used.
        let a=rt.block_on(chat(State(shared.clone()),request(true,true,false)));
        let b=rt.block_on(chat(State(shared.clone()),request(true,true,false)));
        let first=queue.try_fifo().unwrap().unwrap();let second=queue.try_fifo().unwrap().unwrap();
        rt.block_on(async {
            let pending=chat(State(shared.clone()),request(false,true,false));tokio::pin!(pending);
            assert!(tokio::time::timeout(Duration::from_millis(20),&mut pending).await.is_err());
            assert_eq!(shared.waiting_inputs.load(Ordering::Acquire),1);
            assert!(queue.try_fifo().unwrap().is_none());finish(first);
            let worker=async {
                let until=Instant::now()+Duration::from_secs(2);
                loop {
                    if let Some(job)=queue.try_fifo().unwrap(){finish(job);break;}
                    assert!(Instant::now()<until);tokio::time::sleep(Duration::from_millis(1)).await;
                }
            };
            let (response,())=tokio::join!(pending,worker);assert_eq!(response.status(),StatusCode::OK);
            let value:serde_json::Value=serde_json::from_slice(&to_bytes(response.into_body(),1<<20).await.unwrap()).unwrap();
            assert_eq!(value["choices"][0]["finish_reason"],"stop");
        });
        finish(second);drop((a,b));
        assert_eq!(shared.waiting_inputs.load(Ordering::Acquire),0);
        assert_eq!(shared.execution_admission.available_permits(),2);
        assert!(shared.requests.lock().unwrap().is_empty());
        assert!(!queue.has_outstanding_requests());
        // Worker failure closes raw waiters as well as already prepared jobs.
        let a=rt.block_on(chat(State(shared.clone()),request(true,true,false)));
        let b=rt.block_on(chat(State(shared.clone()),request(true,true,false)));
        let first=queue.try_fifo().unwrap().unwrap();let second=queue.try_fifo().unwrap().unwrap();
        let waiting=rt.block_on(chat(State(shared.clone()),request(true,true,false)));
        shared.fail();
        let failed=text(waiting);assert!(failed.contains("worker_not_ready")&&!failed.contains("[DONE]"));
        drop((first,second,a,b));
        assert_eq!(shared.waiting_inputs.load(Ordering::Acquire),0);
        assert_eq!(shared.execution_admission.available_permits(),2);
        assert!(shared.requests.lock().unwrap().is_empty());
    }
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn real_http_stream_finishes_and_disconnect_cancels() {
        use std::os::unix::fs::OpenOptionsExt;
        let reservation = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = reservation.local_addr().unwrap();
        drop(reservation);
        let key = std::env::temp_dir().join(format!(
            "canada-http-test-{}-{}",
            std::process::id(),
            u64::from_le_bytes(random_bytes().unwrap())
        ));
        struct Remove(PathBuf);
        impl Drop for Remove {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let _remove = Remove(key.clone());
        let token = "test-native-HTTP-credential-0000000000000000";
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&key)
            .unwrap();
        file.write_all(token.as_bytes()).unwrap();
        drop(file);
        let priority_key=key.with_extension("priority");
        let _remove_priority=Remove(priority_key.clone());
        let scheduler_token="test-native-scheduler-credential-00000000000";
        let mut file=std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&priority_key).unwrap();
        file.write_all(scheduler_token.as_bytes()).unwrap();drop(file);
        let queue = start(
            &ServerConfig {
                bind: address,
                tokenizer: std::env::var("CANADA_TEST_TOKENIZER").unwrap().into(),
                bearer_file: key,
                priority_file: Some(priority_key),
                allow_remote: false,
                output_stall_seconds: 60, context_cache_mib: 0,
            },
            4096,
            false,
        )
        .unwrap();
        let send = move |request: String| -> String {
            let mut socket = std::net::TcpStream::connect(address).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            socket.write_all(request.as_bytes()).unwrap();
            let mut output = String::new();
            socket.read_to_string(&mut output).unwrap();
            output
        };
        let unauth =
            send("GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".into());
        assert!(unauth.starts_with("HTTP/1.1 401"));
        let health = || {
            format!(
                "GET /health HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {token}\r\nConnection: close\r\n\r\n"
            )
        };
        assert!(send(health()).starts_with("HTTP/1.1 503"));
        queue.ready("test-worker");
        assert!(send(health()).starts_with("HTTP/1.1 200"));
        queue.enable_decode_priority();
        let control_request=|secret:&str,epoch:u64,foreground:bool,ttl:u64| {
            let body=json!({"owner":"test-router","epoch":epoch,"foreground":foreground,"lease_seconds":ttl}).to_string();
            format!("POST /_native/priority HTTP/1.1\r\nHost: localhost\r\nX-Scheduler-Token: {secret}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len())
        };
        assert!(send(control_request(token,1,true,1)).starts_with("HTTP/1.1 401"));
        let request_control=control_request(scheduler_token,1,true,1);
        let controlling=std::thread::spawn(move||send(request_control));
        let deadline=Instant::now()+Duration::from_secs(3);
        let snapshot=loop {
            if let Some(snapshot)=queue.priority_pending(){break snapshot;}
            assert!(Instant::now()<deadline);std::thread::sleep(Duration::from_millis(5));
        };
        assert!(snapshot.hold && !controlling.is_finished(),"receipt must wait for the model owner");
        // This CPU adapter test simulates the owner callback. The live GPU
        // qualification separately proves the all-rank drain and progress hold.
        queue.priority_applied(snapshot).unwrap();
        assert!(controlling.join().unwrap().starts_with("HTTP/1.1 200"));
        assert!(send(control_request(scheduler_token,0,false,1)).starts_with("HTTP/1.1 409"));
        let deadline=Instant::now()+Duration::from_secs(3);
        while queue.router_hold() {assert!(Instant::now()<deadline);std::thread::sleep(Duration::from_millis(10));}
        let expired=queue.priority_pending().unwrap();assert!(!expired.hold);
        queue.priority_applied(expired).unwrap();
        assert!(send(control_request(scheduler_token,1,false,1)).starts_with("HTTP/1.1 409"),"expired owner keeps its epoch tombstone");
        let body = r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":16,"stream":true}"#;
        let request = || {
            format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
        };
        let first = request();
        let client = std::thread::spawn(move || send(first));
        let job = queue.next(Duration::from_secs(5)).unwrap().unwrap();
        job.emit(BackendEvent::Ready {
            fingerprint: "test-worker".into(),
            cached_tokens: 0,
        });
        job.emit(BackendEvent::Tokens(vec![42, 1]));
        job.emit(BackendEvent::Finish(
            deepseek_recipe::stream::InferenceFinishReason::Stop,
        ));
        drop(job);
        let output = client.join().unwrap();
        assert!(output.starts_with("HTTP/1.1 200"));
        assert!(output.contains("[DONE]"));
        assert!(output.contains("\"finish_reason\":\"stop\""));
        assert!(!output.contains("\"timings\""), "missing phase must not become fabricated timing");
        for (stream, include_usage, cached, late) in [
            (true, false, false, false), (true, true, true, false),
            (true, true, true, true), (false, false, true, false),
        ] {
            let mut body=json!({"messages":[{"role":"user","content":"hi"}],
                "max_tokens":16,"stream":stream});
            if include_usage {body["stream_options"]=json!({"include_usage":true});}
            let body=body.to_string();
            let request=format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len());
            let client=std::thread::spawn(move||send(request));
            let mut job=queue.next(Duration::from_secs(5)).unwrap().unwrap();
            let count=job.tokens.len();
            // Dispatch transfers tokens to the native owner before Begin
            // returns. The HTTP metadata must retain its immutable count.
            let dispatched=std::mem::take(&mut job.tokens);
            assert_eq!(dispatched.len(),count);
            assert!(job.tokens.is_empty());
            let cached=if cached {count-1}else{0};
            let mut measured=json!({"type":"tokens","prompt_tokens":count,"cached_tokens":cached,
                "begin_phases_seconds":{"target_prefill":0.25},"seconds":12.,"queue_and_prefill_seconds":15.});
            if !late {job.record_prefill_timing(&measured);}
            job.emit(BackendEvent::Ready{fingerprint:"test-worker".into(),cached_tokens:cached});
            if late {
                // A Ready chunk can precede completed background prefill. Its
                // final accumulated phase must still reach the same response.
                std::thread::sleep(Duration::from_millis(15));
                measured["background_prefill"]=json!(true);
                job.record_prefill_timing(&measured);
            }
            // A later event must not overwrite the request's first phase.
            measured["begin_phases_seconds"]["target_prefill"]=json!(99.);
            job.record_prefill_timing(&measured);
            job.emit(BackendEvent::Tokens(vec![42,1]));
            job.emit(BackendEvent::Finish(deepseek_recipe::stream::InferenceFinishReason::Stop));
            drop(job);
            let output=client.join().unwrap();
            assert!(output.starts_with("HTTP/1.1 200"));
            let metadata=if stream {
                let chunks=output.lines().filter_map(|line|line.strip_prefix("data: "))
                    .filter_map(|line|serde_json::from_str::<serde_json::Value>(line).ok()).collect::<Vec<_>>();
                assert!(output.contains("[DONE]"));
                let timed=chunks.iter().filter(|chunk|chunk.get("timings").is_some()).collect::<Vec<_>>();
                assert_eq!(timed.len(),1,"one timing object per response");
                assert!(chunks.iter().any(|c|c["choices"][0]["finish_reason"]=="stop"));
                timed[0]["timings"].clone()
            }else{
                let complete:serde_json::Value=serde_json::from_str(output.split_once("\r\n\r\n").unwrap().1).unwrap();
                assert_eq!(complete["choices"][0]["finish_reason"],"stop");
                complete["timings"].clone()
            };
            assert_eq!(metadata,json!({"prompt_n":count-cached,"cache_n":cached,"prompt_ms":250.,
                "prompt_per_second":(count-cached)as f64/0.25,
                "scope":if late {"server_prefill_active"}else{"server_prefill_phase"}}));
            println!("TIMING_RECEIPT {}",json!({"stream":stream,"include_usage":include_usage,"timings":metadata}));
        }
        let second = request();
        let (dispatched, may_disconnect) = mpsc::sync_channel(0);
        let disconnected = std::thread::spawn(move || {
            let mut socket = std::net::TcpStream::connect(address).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            socket.write_all(second.as_bytes()).unwrap();
            let mut first = [0u8; 4096];
            assert!(socket.read(&mut first).unwrap() > 0);
            // Headers can precede owner dispatch. This case specifically
            // checks cancellation AFTER dispatch; an earlier disconnect is
            // legitimately discarded before Queue::next returns a Job.
            may_disconnect.recv().unwrap();
            // Client closes without a backend finish.
        });
        let job = queue.next(Duration::from_secs(5)).unwrap().unwrap();
        job.emit(BackendEvent::Ready {
            fingerprint: "test-worker".into(),
            cached_tokens: 0,
        });
        dispatched.send(()).unwrap();
        disconnected.join().unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        while !job.cancelled() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            job.cancelled(),
            "TCP disconnect did not release the stream cancellation guard"
        );
        drop(job);
        queue.fail();
    }
}
