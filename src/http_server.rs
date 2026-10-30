//! HTTP adapter for the native resident worker. HTTP threads own only
//! protocol conversion and bounded queues; the main rank thread owns all GPU
//! state. Sixteen live/queued requests are admitted without a batching timer;
//! background admission reserves one of those entries for foreground.
use crate::protocol::{BackendEvent, Frontend, MODEL_NAME};
#[path = "http_timing.rs"]
mod timing;
use timing::{PrefillTiming, WithTimings};
use anyhow::{Result, ensure};
use axum::{
    Json, Router,
    body::{Body, to_bytes},
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
        Ok(())
    }
}

#[cfg(test)]
#[test]
fn listener_requires_explicit_authenticated_fleet_binding() {
    let config = |address: &str, allow_remote: bool, priority: bool| ServerConfig {
        bind: address.parse().unwrap(), tokenizer: "unused".into(), bearer_file: "unused".into(),
        priority_file: priority.then(|| PathBuf::from("scheduler.key")), allow_remote,
        output_stall_seconds: 60,
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
    frontend: Arc<Frontend>,
    capacity: usize,
    body_limit: usize,
    admission: Arc<Semaphore>,
    background_admission: Arc<Semaphore>,
    decode_priority: AtomicBool,
    background_prefill: AtomicBool,
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
}

impl Shared {
    fn fail(&self) {
        self.health.store(FAILED, Ordering::Release);
        self.shutdown.store(true, Ordering::Release);
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
    shared: Arc<Shared>,
    _permit: OwnedSemaphorePermit,
    _background_permit: Option<OwnedSemaphorePermit>,
}
impl Job {
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
        self.shared.requests.lock().unwrap().remove(&self.id);
    }
}

pub(crate) struct Queue {
    receiver: mpsc::Receiver<Job>,
    pending: RefCell<VecDeque<Job>>,
    shared: Arc<Shared>,
}
impl Queue {
    pub fn try_next(&self) -> Result<Option<Job>> {
        self.try_selected(false)
    }
    pub fn foreground_waiting(&self) -> Result<bool> {
        if let Some(job)=self.try_foreground()? {
            self.pending.borrow_mut().push_front(job);
            Ok(true)
        }else{Ok(false)}
    }
    pub fn try_foreground(&self) -> Result<Option<Job>> {
        self.try_selected(true)
    }
    fn try_selected(&self, foreground_only: bool) -> Result<Option<Job>> {
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
        let selected = pending.iter().position(|job| !job.background)
            .or_else(|| (!foreground_only && !pending.is_empty()).then_some(0));
        Ok(selected.and_then(|i| pending.remove(i)))
    }
    pub fn next(&self, wait: Duration) -> Result<Option<Job>> {
        let deadline = Instant::now() + wait;
        loop {
            if let Some(job) = self.try_next()? { return Ok(Some(job)); }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            match self.receiver.recv_timeout(remaining) {
                Ok(job) if job.cancelled() => continue,
                Ok(job) => self.pending.borrow_mut().push_back(job),
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
    pub fn enable_background_prefill(&self) {
        self.shared.background_prefill.store(true,Ordering::Release);
    }
    pub fn enable_decode_priority(&self) {
        self.shared.decode_priority.store(true, Ordering::Release);
    }
    pub fn stopping(&self) -> bool {
        self.shared.shutdown.load(Ordering::Acquire)
    }
    pub fn input_reclaim_target(&self) -> usize {
        self.shared.input_reclaim_target.load(Ordering::Acquire)
    }
    pub fn input_reclaimed(&self, target: usize) {
        // A larger request arriving during reclamation needs another pass.
        let _ = self.shared.input_reclaim_target.fetch_update(Ordering::AcqRel, Ordering::Acquire,
            |pending| (pending <= target).then_some(0));
        self.shared.input_reclaim_generation.fetch_add(1, Ordering::Release);
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
    ready: impl Fn() -> bool,
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
    let deadline = Instant::now() + timeout;
    while generation.load(Ordering::Acquire) == before && ready() && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    if ready() && generation.load(Ordering::Acquire) != before {
        reserve()
    } else {
        Err(error)
    }
}

async fn reserve_input_workspace(shared: &Shared, bytes: usize)
    -> Result<crate::memory::HostWorkspaceReservation> {
    reserve_with_input_reclaim(
        || crate::memory::reserve_host_workspace(bytes),
        &shared.input_reclaim_target, &shared.input_reclaim_generation,
        || shared.health.load(Ordering::Acquire) == READY && !shared.shutdown.load(Ordering::Acquire),
        Duration::from_secs(5),
    ).await
}

async fn chat(State(shared): State<Arc<Shared>>, request: Request<Body>) -> Response {
    if let Some(error) = check(&shared, request.headers()) {
        return error;
    }
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
    let upload_workspace = if shared.body_limit == MAX_BODY {
        let bytes = request.headers().get("content-length").and_then(|v|v.to_str().ok())
            .and_then(|v|v.parse::<usize>().ok()).unwrap_or(shared.body_limit);
        if bytes > shared.body_limit { return error(StatusCode::PAYLOAD_TOO_LARGE, "body_limit", "request body exceeds its bound"); }
        match reserve_input_workspace(&shared, bytes.max(1)*2).await {
            Ok(reservation) => Some(reservation),
            Err(_) => return error(StatusCode::SERVICE_UNAVAILABLE, "input_capacity", "input upload cannot retain the memory reserve"),
        }
    } else { None };
    let body = match tokio::time::timeout(
        Duration::from_secs(15),
        to_bytes(request.into_body(), shared.body_limit),
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
    // Conversion can simultaneously own the raw body, parsed/normalized
    // request, canonical conversation, rendering and tokenizer buffers. Hold
    // this conservative size-based allowance through the blocking task, even
    // if its HTTP caller disconnects. Decoder scratch has its own reservation.
    let conversion_workspace = if shared.body_limit == MAX_BODY {
        match reserve_input_workspace(&shared, body.len()*16+(64<<20)).await {
            Ok(reservation) => Some(reservation),
            Err(_) => return error(StatusCode::SERVICE_UNAVAILABLE, "input_capacity", "input conversion cannot retain the memory reserve"),
        }
    } else { None };
    drop(upload_workspace);
    let frontend = shared.frontend.clone();
    let capacity = shared.capacity;
    let cancellation = Arc::new(AtomicBool::new(false));
    let guard = CancelOnDrop(cancellation.clone());
    let preprocessing_cancel = cancellation.clone();
    let preprocessing_shared = shared.clone();
    // A disconnected caller cannot release admission while its blocking
    // conversion still owns a potentially large body/decoder workspace.
    let (mut prepared, permit, background_permit) =
        match tokio::task::spawn_blocking(move || {
            let _workspace = conversion_workspace;
            let cancelled = || preprocessing_cancel.load(Ordering::Acquire)
                || preprocessing_shared.shutdown.load(Ordering::Acquire);
            let mut prepared = frontend.prepare_chat_cancellable(&body, capacity, &cancelled);
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
                let deadline = Instant::now() + Duration::from_secs(5);
                while preprocessing_shared.input_reclaim_generation.load(Ordering::Acquire) == generation
                    && Instant::now() < deadline && !cancelled()
                    && preprocessing_shared.health.load(Ordering::Acquire) == READY {
                    std::thread::sleep(Duration::from_millis(10));
                }
                if !cancelled() && preprocessing_shared.health.load(Ordering::Acquire) == READY
                    && preprocessing_shared.input_reclaim_generation.load(Ordering::Acquire) != generation {
                    prepared = frontend.prepare_chat_cancellable(&body, capacity, &cancelled);
                }
            }
            (prepared, permit, background_permit)
        }).await {
            Ok((Ok(prepared), permit, background_permit)) => (prepared, permit, background_permit),
            Ok((Err(reason), _, _)) => {
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
    let id = format!(
        "chatcmpl-{}-{}",
        shared.nonce,
        shared.sequence.fetch_add(1, Ordering::Relaxed)
    );
    let (output, incoming) = tokio::sync::mpsc::channel(OUTPUT_CHUNKS);
    let prefill_timing = Arc::new(OnceLock::new());
    let stream = prepared.converted.stream;
    let prompt_tokens = prepared.tokens.len();
    shared
        .requests
        .lock()
        .unwrap()
        .insert(id.clone(), Arc::downgrade(&cancellation));
    let job = Job {
        id: id.clone(),
        tokens: prepared.tokens.clone(),
        images: std::mem::take(&mut prepared.images),
        image_inputs: std::mem::take(&mut prepared.image_inputs),
        seed,
        greedy: prepared.converted.inference_options.temperature == Some(0.),
        limit: prepared.max_new_tokens as usize,
        enqueued: Instant::now(),
        background,
        cancelled: cancellation,
        output,
        prompt_tokens,
        prefill_timing: prefill_timing.clone(),
        output_stalled_since: Cell::new(None),
        shared: shared.clone(),
        _permit: permit,
        _background_permit: background_permit,
    };
    match shared.sender.try_send(job) {
        Ok(()) => {}
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
            inner: Box::pin(events),
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
    let shared = Arc::new(Shared {
        sender,
        frontend,
        capacity,
        body_limit: if images { MAX_BODY } else { 8 << 20 },
        admission: Arc::new(Semaphore::new(QUEUE_DEPTH + 1)),
        background_admission: Arc::new(Semaphore::new(QUEUE_DEPTH)),
        decode_priority: AtomicBool::new(false),
        background_prefill: AtomicBool::new(false),
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
    Ok(Queue { receiver, pending: RefCell::new(VecDeque::new()), shared })
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn input_workspace_reclaims_once_then_rechecks_the_unchanged_reservation() {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let target = AtomicUsize::new(0);
        let generation = AtomicU64::new(0);
        let attempts = Cell::new(0);
        let capacity = Cell::new(false);
        rt.block_on(async {
            let admission = reserve_with_input_reclaim(|| {
                attempts.set(attempts.get() + 1);
                if capacity.get() { Ok(17) } else {
                    Err(crate::memory::InputCapacityError("reserve unchanged", 8192).into())
                }
            }, &target, &generation, || true, Duration::from_secs(1));
            let owner = async {
                tokio::task::yield_now().await;
                assert_eq!(target.load(Ordering::Acquire), 8192);
                capacity.set(true);
                generation.fetch_add(1, Ordering::Release);
            };
            let (result, ()) = tokio::join!(admission, owner);
            assert_eq!(result.unwrap(), 17);
        });
        assert_eq!(attempts.get(), 2);
        // Success and fixed per-input limits must not request GPU work.
        target.store(0, Ordering::Release);
        assert_eq!(rt.block_on(reserve_with_input_reclaim(|| Ok(23), &target,
            &generation, || true, Duration::ZERO)).unwrap(), 23);
        let fixed: Result<()> = rt.block_on(reserve_with_input_reclaim(||
            Err(crate::memory::InputCapacityError("fixed input cap", 0).into()),
            &target, &generation, || true, Duration::ZERO));
        assert!(fixed.is_err());
        assert_eq!(target.load(Ordering::Acquire), 0);
        // No owner response: refuse rather than allocate or retry indefinitely.
        attempts.set(0);
        target.store(16384, Ordering::Release);
        let unavailable: Result<()> = rt.block_on(reserve_with_input_reclaim(|| {
            attempts.set(attempts.get() + 1);
            Err(crate::memory::InputCapacityError("reserve unchanged", 8192).into())
        }, &target, &generation, || true, Duration::ZERO));
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
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn admission_cancel_backpressure_and_bare_eof_are_bounded() {
        let frontend = Arc::new(
            Frontend::load(Path::new(&std::env::var("CANADA_TEST_TOKENIZER").unwrap())).unwrap(),
        );
        let (sender, receiver) = mpsc::sync_channel(QUEUE_DEPTH + 1);
        let shared = Arc::new(Shared {
            sender,
            frontend,
            capacity: 4096,
            body_limit: 8 << 20,
            admission: Arc::new(Semaphore::new(QUEUE_DEPTH + 1)),
            background_admission: Arc::new(Semaphore::new(QUEUE_DEPTH)),
            decode_priority: AtomicBool::new(false),
        background_prefill: AtomicBool::new(false),
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
            nonce: "test".into(),
            sequence: AtomicU64::new(1),
        });
        let queue = Queue {
            receiver,
            pending: RefCell::new(VecDeque::new()),
            shared: shared.clone(),
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
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
                output_stall_seconds: 60,
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
        let disconnected = std::thread::spawn(move || {
            let mut socket = std::net::TcpStream::connect(address).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            socket.write_all(second.as_bytes()).unwrap();
            let mut first = [0u8; 4096];
            assert!(socket.read(&mut first).unwrap() > 0);
            // Client closes without a backend finish.
        });
        let job = queue.next(Duration::from_secs(5)).unwrap().unwrap();
        job.emit(BackendEvent::Ready {
            fingerprint: "test-worker".into(),
            cached_tokens: 0,
        });
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
