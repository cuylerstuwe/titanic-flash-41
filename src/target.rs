//! Fixture-free target model owner. Canonical embedding -> forty decoder layers
//! -> final normalization -> vocabulary logits, with retained causal state.
//!
//! Execution profiles are bounded: six-row retained or128-row encoder chunks,
//! one sequence, with admitted experimental contexts through4096 tokens. This is an integration profile, not the
//! qualified full-context/large-prefill/DSpark engine or a performance claim.
use crate::{
    attention::{Candidates, PagedKv},
    blas::Blas,
    checkpoint::Checkpoint,
    compressed_source::{Layer as Source, Metadata, Phase},
    cuda::{Buffer, Cuda, Graph, Module, Stream},
    decoder::SwaLayer,
    engram::{PackedShard, RowStore},
    engram_gpu::Rows,
    expert_packing,
    ffn::TokenRows,
    hash_history::{HashBatch, HashMetadata, History},
    kv::RowLocations,
    layer_attention::InputMode,
    layout::RankPlan,
    model_io::ModelIo,
    prelude::{FrequencyTable, Positions},
    roce::Roce,
    sampling::Greedy,
    weights::Weights,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::{
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};

#[path = "paired_model_probe.rs"]
mod paired_model_probe;
pub(crate) use paired_model_probe::PairExecutor;
#[path = "published_state_probe.rs"]
mod published_state_probe;

thread_local! {
    static PREFILL_BOUNDARY_TRACE: std::cell::RefCell<Option<Vec<serde_json::Value>>>
        = const {std::cell::RefCell::new(None)};
    static RESTORE_INITIAL_LOGITS: std::cell::RefCell<Option<Vec<u8>>>
        = const {std::cell::RefCell::new(None)};
}
pub(crate) fn take_restore_initial_logits_for_probe()->Result<Vec<u8>> {
    ensure!(std::env::var("CANADA_SCRATCH_RESTORE_INITIAL_SCREEN").as_deref()==Ok("1"),
        "initial logits require the finite restore diagnostic");
    RESTORE_INITIAL_LOGITS.with(|cell|cell.borrow_mut().take()
        .context("initial diagnostic did not observe a target draw"))
}
pub(crate) fn begin_prefill_boundary_trace_for_probe()->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_PEER_PREFILL_TRACE").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_PEER_PREFILL_NUMERICS").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_PEER_PREFILL_SCREEN").as_deref()==Ok("1"),
        "finite prefill boundary trace admission");
    PREFILL_BOUNDARY_TRACE.with(|cell|->Result<()>{
        let mut value=cell.borrow_mut();ensure!(value.is_none(),"nested prefill trace");
        *value=Some(Vec::new());Ok(())
    })
}
pub(crate) fn take_prefill_boundary_trace_for_probe()->Result<Vec<serde_json::Value>> {
    PREFILL_BOUNDARY_TRACE.with(|cell|cell.borrow_mut().take()
        .ok_or_else(||anyhow::anyhow!("prefill trace not started")))
}

pub(crate) struct PrefillProgress {
    pub done: usize,
    pub remaining_chunks: Vec<usize>,
}

#[derive(serde::Serialize)]
pub struct AppendMetrics {
    pub(crate) live_rows: usize,
    physical_rows: usize,
    phase: String,
    captured_new_graph: bool,
    engram_split14: bool,
    engram_split14_adaptive: bool,
    encoder_graph_chunks: usize,
    #[serde(skip_serializing_if="Vec::is_empty")]
    lookahead: Vec<LookaheadMetrics>,
    #[serde(skip_serializing_if="Vec::is_empty")]
    stall_counters: Vec<serde_json::Value>,
    #[serde(skip_serializing_if="Option::is_none")]
    native_fp32_rows: Option<usize>,
    #[serde(skip_serializing_if="Option::is_none")]
    layer20_deferred_chunks: Option<usize>,
    pub(crate) stages_ms: Vec<(&'static str, f64)>,
    total_ms: f64,
}

#[derive(serde::Serialize)]
struct LookaheadMetrics {
    next_rows: usize,
    worker_ms: f64,
    gpu_wait_ms: f64,
    join_wait_ms: f64,
}

pub const ROWS: usize = 6;
/// Same-owner lossless device image. It does not retain weights, communicator
/// sequence numbers, output buffers or pending speculative transactions.
pub(crate) struct StateImage {
    image: crate::state_image::Storage,
    owner: crate::session_lease::Lease,
    history: History,
    context: usize,
    rank: usize,
    stream: usize,
    compact: bool,
}
/// Finite replay diagnostics also compare whole final pages. Preserve their
/// not-yet-published slack across arms; normal serving correctly omits it.
pub(crate) struct RepeatPhysicalImage {
    image: crate::state_image::Image,
    frontier: usize,
    extent: usize,
}
/// An encoder prefix is deliberately not a complete target/session image.
/// Restoring it either evaluates a suffix of at least128 tokens or explicitly
/// reconstructs the bounded decoder tail before returning a complete owner.
pub(crate) struct EncoderPrefixImage(StateImage, Option<bank_encoder_lineage::Binding>);
#[path = "bank_encoder_lineage.rs"]
mod bank_encoder_lineage;
pub(crate) use bank_encoder_lineage::Witness as BankEncoderWitness;
#[path = "bank_encoder_checkpoint.rs"]
mod bank_encoder_checkpoint;
impl EncoderPrefixImage {
    /// Disk tier: the image's storage (None for exchange storage).
    pub(crate) fn images_mut(&mut self)->Vec<&mut crate::state_image::Image> {self.0.image.image_mut().into_iter().collect()}
    pub(crate) fn images(&self)->Vec<&crate::state_image::Image> {self.0.image.image().into_iter().collect()}
    pub(crate) fn bytes(&self)->usize {self.0.bytes()}
    pub(crate) fn reusable_storage(&self,bytes:usize)->bool {
        self.bytes()==bytes&&matches!(&self.0.image,crate::state_image::Storage::Copy(_))
    }
    pub(crate) fn fused_copy(&self)->bool {
        matches!(&self.0.image,crate::state_image::Storage::Copy(image)if image.fused_copy())
    }
    /// Consumes a checkpoint already selected for ordinary eviction. None of
    /// its model/history/lineage metadata survives; only charged byte storage.
    pub(crate) fn into_retired_storage(self)->Result<crate::state_image::Image> {
        match self.0.image {crate::state_image::Storage::Copy(image)=>Ok(image),
            _=>anyhow::bail!("mutable exchange storage cannot become an encoder snapshot")}
    }
    pub(crate) fn position(&self)->usize {self.0.history.position as usize}
    pub(crate) fn lineage(&self)->u64 {self.0.history.lineage}
    /// Payload stays immutable; the binding identifies the bank lineage that
    /// still owns the omitted prefix. Restoring this image can rebind it;
    /// reset or an unrelated rewind cannot.
    pub(crate) fn bank_retained(&self)->bool {self.1.is_some()}
    pub(crate) fn binding_bytes()->usize {
        if crate::config::bank_encoder_checkpoint_requested(){bank_encoder_lineage::Binding::HEAP_BYTES}else{0}
    }
    pub(crate) fn region_digests_for_probe(&self)->Result<Vec<(String,usize,String)>> {
        match &self.0.image {crate::state_image::Storage::Copy(image)=>image.region_digests_for_probe(),
            _=>anyhow::bail!("region digests need an ordinary copied image")}
    }
    pub(crate) fn digest_for_import_probe(&self)->Result<String> {
        ensure!(std::env::var("CANADA_SCRATCH_ENCODER_PREFIX_IMPORT_SCREEN").as_deref()==Ok("1"),
            "encoder image digest requires finite import screen");
        self.0.digest()
    }
    #[cfg(feature = "storage-fault-injection")]
    pub(crate) fn retention_digest(&self)->Result<serde_json::Value> {
        Ok(serde_json::json!({"bytes":self.0.bytes(),"image":self.0.digest()?,"causal_history":self.0.causal_history()}))
    }
}
/// The caller keeps the immutable parent through the entire save attempt.
/// Never release it before a successor exists: optional allocation can refuse.
/// What a prefill chunk overwrites, captured as of an earlier frontier inside that chunk
/// (the inline encoder checkpoint): the pair sources' carry rings, the 128-row encoder tail
/// and the host history at that frontier. The caches are append-only by position, so their
/// pages at the frontier are still read from the live buffers when the image is saved.
pub(crate) struct CheckpointAux {
    tail: crate::decoder_tail::Cache,
    rings: [crate::pair_prefill::Ring; 3],
    captured: std::cell::RefCell<Option<History>>,
}
impl CheckpointAux {
    /// The image regions this state replaces, under the live regions' names.
    fn image_sources(&self, cuda: &Cuda) -> Result<Vec<crate::state_image::Region<'_>>> {
        let mut sources = self.tail.image_regions()?;
        // The same extent as the live carry regions: request slot 0 when the owner has several.
        let slots = crate::config::session_slots()?;
        for (layer, ring) in [2, 8, 14].into_iter().zip(&self.rings) {
            sources.push(if slots > 1 {
                crate::state_image::Region::byte_range(cuda, format!("compressed.{layer}.carry"), ring.values(),
                    0..ring.values().bytes() / slots)?
            } else {
                crate::state_image::Region::new(cuda, format!("compressed.{layer}.carry"), ring.values())?
            });
        }
        Ok(sources)
    }
}
pub(crate) enum EncoderPrefixRestore<'a> {
    Borrowed(&'a EncoderPrefixImage,u64),
    Imported(&'a EncoderPrefixImage,&'a Target,u64),
}
impl EncoderPrefixRestore<'_> {
    fn image_and_lineage(&self)->(&EncoderPrefixImage,u64) {
        match self {Self::Borrowed(image,lineage)|Self::Imported(image,_,lineage)=>(*image,*lineage)}
    }
}
fn encoder_checkpoint_enabled()->bool {
    std::env::var("CANADA_SCRATCH_ENCODER_CHECKPOINT_SCREEN").as_deref()==Ok("1")
        ||std::env::var("CANADA_SCRATCH_ENCODER_PROMPT_CHECKPOINT").as_deref()==Ok("1")
}
fn experimental_prefill_flag(name:&str,default:bool)->Result<bool> {
    match std::env::var(name) {
        Ok(v)=>{ensure!(v=="0"||v=="1","invalid {name}");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(default),Err(e)=>Err(e.into()),
    }
}
fn short_prefill_rows_enabled(name:&str)->Result<bool> {
    match std::env::var(name) {
        Ok(v)if v=="0"=>Ok(false),Ok(v)if v=="1"=>Ok(true),
        Err(std::env::VarError::NotPresent)=>Ok(true),
        _=>anyhow::bail!("{name} must be0or1"),
    }
}
pub(crate) fn prefill_upper_rows_requested()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_PREFILL_UPPER_ROWS") {
        Ok(v)=>{ensure!(v=="0"||v=="1","invalid upper prefill row policy");Ok(v=="1")},
        // s1320 same-allocation native ABBA: exact logits, state and following
        // verification at65/96/127 rows;0.6–2.9% faster cached appends.
        Err(std::env::VarError::NotPresent)=>Ok(true),Err(e)=>Err(e.into()),
    }
}
fn verification_mhc_live_row_limit()->Result<usize> {
    // s973: M1 improved in both orders; M3/M5 did not retain the component
    // benefit. Keep the broader experiment explicitly selectable, with the
    // measured singleton path as the default and 0 as the complete off switch.
    match std::env::var("CANADA_SCRATCH_MHC_LIVE_ROWS") {
        Err(std::env::VarError::NotPresent)=>Ok(1),
        Ok(v)if v=="0"=>Ok(0),Ok(v)if v=="1"=>Ok(6),
        _=>anyhow::bail!("CANADA_SCRATCH_MHC_LIVE_ROWS must be0or1"),
    }
}
pub(crate) fn defer_checkpoint_decoder128()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_ENCODER_CHECKPOINT_DEFER128") {
        Err(std::env::VarError::NotPresent)=>Ok(true),
        Ok(v)if v=="0"=>Ok(false),Ok(v)if v=="1"=>Ok(true),
        _=>anyhow::bail!("encoder checkpoint defer128 must be0or1"),
    }
}
impl StateImage {
    /// Disk tier: the copied image (None for exchange storage).
    pub(crate) fn image_mut(&mut self)->Option<&mut crate::state_image::Image> {self.image.image_mut()}
    pub(crate) fn image(&self)->Option<&crate::state_image::Image> {self.image.image()}
    pub(crate) fn exchangeable(&self)->bool {matches!(self.image,crate::state_image::Storage::Exchange(_))}
    pub(crate) fn manifest(&self) -> serde_json::Value {
        let mut value=self.image.manifest();value["compact"]=serde_json::json!(self.compact);value
    }
    pub(crate) fn digest(&self) -> Result<String> { self.image.digest() }
    pub(crate) fn position(&self) -> u64 { self.history.position }
    pub(crate) fn bytes(&self) -> usize { self.image.bytes() }
    pub(crate) fn causal_history(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(&self.history).unwrap();
        value.as_object_mut().unwrap().remove("lineage");
        value
    }
}
/// Two parity arenas hold every wide layer's row-proportional transients
/// (mHC residual boundaries dominate; Engram layers1/14 need the most).
// Sequence-parallel profiles retain the same offsets and lifetimes with a
// smaller unused tail. Every allocation still checks the actual arena bound;
// a future layer-layout increase fails construction before any model work.
/// Encoder-tail publication reads only rows [live-128, live): gather just that window
/// (single-owner rows summed exactly) instead of every SP row. 2 = unresolved.
static TAIL_WINDOW: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
thread_local! {
    /// Probe only (replay_exact_probe): the fresh 128-row prefill's mode for the next calls instead of the environment's.
    static CANONICAL_PREFILL_FOR_PROBE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}
pub(crate) fn set_canonical_prefill_for_probe(mode: Option<bool>) { CANONICAL_PREFILL_FOR_PROBE.with(|c| c.set(mode)); }
pub(crate) fn tail_window_enabled() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match TAIL_WINDOW.load(Relaxed) {
        2 => { let on = std::env::var("CANADA_SCRATCH_PREFILL_TAIL_WINDOW").as_deref() == Ok("1");
            TAIL_WINDOW.store(on as u8, Relaxed); on }
        v => v == 1,
    }
}
/// Eager wide encoder chunks: hash and stage this chunk's Engram rows on the host while the
/// GPU runs the embedding and layer 0 (which read no Engram input), instead of before them.
/// Same rows, kernels and GPU order. 2 = unresolved (CANADA_SCRATCH_PREFILL_ENGRAM_BEHIND_L0).
static ENGRAM_BEHIND_L0: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
/// With ENGRAM_BEHIND_L0: stage layer-1 rows behind layer 0 and layer-14 rows behind layers
/// 1-13 (their only consumer is layer 14). 2 = unresolved (CANADA_SCRATCH_PREFILL_ENGRAM_SPLIT14).
static ENGRAM_SPLIT14: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
pub(crate) fn engram_split14_enabled() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match ENGRAM_SPLIT14.load(Relaxed) {
        2 => { let on = std::env::var("CANADA_SCRATCH_PREFILL_ENGRAM_SPLIT14").as_deref() == Ok("1");
            ENGRAM_SPLIT14.store(on as u8, Relaxed); on }
        v => v == 1,
    }
}
pub(crate) fn engram_behind_l0_enabled() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match ENGRAM_BEHIND_L0.load(Relaxed) {
        2 => { let on = std::env::var("CANADA_SCRATCH_PREFILL_ENGRAM_BEHIND_L0").as_deref() == Ok("1");
            ENGRAM_BEHIND_L0.store(on as u8, Relaxed); on }
        v => v == 1,
    }
}
/// CANADA_SCRATCH_PREFILL_LAYER20_DEFER=1: in eager wide prefill, run only layer
/// 20's projection and source20 keys for chunks before the request's final 128
/// rows; the decoder replay reads layer 20's output only for those final rows.
/// Every value later read is produced by the same kernels in the same order.
pub(crate) fn layer20_defer_enabled() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_PREFILL_LAYER20_DEFER") {
        Ok(v) => { ensure!(v == "0" || v == "1", "layer20 deferral must be0or1"); Ok(v == "1") }
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(e) => Err(e.into()),
    }
}
/// A full chunk [done, done+live) whose rows all precede the final 128 rows of
/// this `total`-row append. Partial chunks (always the last) never defer.
pub(crate) fn layer20_projection_only(done: usize, live: usize, rows: usize, total: usize) -> bool {
    live == rows && done + live + 128 <= total
}
fn wide_arena_bytes_per_row(sequence_parallel: bool) -> usize {
    crate::config::wide_arena_kib_per_row().unwrap_or(if sequence_parallel { 384 } else { 448 }) * 1024
}
pub const CONTEXT: usize = 128;
const SWA_OFFSET: usize = crate::swa_layout::OFFSET;
const COMPRESSED_OFFSET: usize = 32;
/// Preserve existing bounded layouts. Larger contexts need exactly enough
/// pages: rounding1M+32 to a power of two would double storage and exceed the
/// canonical index-cache address domain. Only tokens/page must be a power of2.
fn compressed_pages(context: usize, ratio: usize, page_tokens: usize, minimum: usize) -> usize {
    let pages = (COMPRESSED_OFFSET + context / ratio).div_ceil(page_tokens);
    minimum.max(if context <= 65536 { pages.next_power_of_two() } else { pages })
}

/// Exact page coverage for causal SWA state, including the padding page.
/// Keep two complete windows to cover both normal decoding and bounded
/// 128-row decoder replay. New wide rows are written before attention reads.
fn saved_swa_pages(layout:crate::swa_layout::Layout,position:usize)->Vec<(usize,usize)> {
    let mut pages=std::collections::BTreeSet::from([0usize]);
    let retained=crate::swa_layout::WINDOW+crate::swa_layout::MAX_ROWS;
    for p in position.saturating_sub(retained)..position {pages.insert(layout.slot(p)/64);}
    let mut ranges:Vec<(usize,usize)>=Vec::new();
    for page in pages {
        if let Some(last)=ranges.last_mut().filter(|last|last.1==page){last.1+=1;}
        else{ranges.push((page,page+1));}
    }
    ranges
}

/// Graph slots: one per live-row count for a single session (slot = live rows),
/// then one per executed-row count for each batched session count 2..=4. A step's
/// verify compressor block size (rows / sessions) is a launch argument, so steps
/// with different session counts never share a capture.
const MAX_BATCHED_SESSIONS: usize = 4;
/// One slot group per batched session count, plus one for one-session region steps: a step
/// with batched rows runs in region mode (dense index scorers, full logits; set_region_step),
/// so it never shares a capture with today's single-history step of the same live count.
/// Steps of several sessions with padding routes (ca_route_padding_rows in every layer's routing)
/// take a second copy of these slots.
/// Tight blocks (config::batched_tight_blocks) add, after those, one slot per (padding routes, session count
/// from two, block, executed rows): the block sets the compressors' grid, so it is part of a capture.
fn graph_slots(rows: usize) -> usize {
    // Only the decode owners (at most 32 rows) batch conversations; wide profiles keep the plain slots.
    let tight = if rows <= 32 { 2 * (MAX_BATCHED_SESSIONS - 1) * (rows + 1) * (rows + 1) } else { 0 };
    // The last rows + 1 slots: config::index_short_rows steps (short_index_slot).
    2 * (rows + 1) * (MAX_BATCHED_SESSIONS + 1) + tight + (rows + 1)
}
/// config::index_short_rows: the slot of a single-conversation step of `executed` rows whose rows all have at most
/// 512 index keys (its indexed layers enqueue no scoring, so it never shares a capture with an ordinary step).
fn short_index_slot(rows: usize, executed: usize) -> usize {
    debug_assert!(executed <= rows);
    graph_slots(rows) - (rows + 1) + executed
}
/// Clears the index branches' short-rows switches when a step's enqueue ends, on every path out of it.
struct ShortIndexRows(Vec<Rc<std::cell::Cell<bool>>>);
impl Drop for ShortIndexRows {
    fn drop(&mut self) { for flag in &self.0 { flag.set(false); } }
}
/// `block`: 0 for the fixed layout (rows / sessions); a tight layout's block otherwise.
fn graph_slot(rows: usize, sessions: usize, executed: usize, region: bool, padded: bool, block: usize) -> usize {
    debug_assert!((1..=MAX_BATCHED_SESSIONS).contains(&sessions) && executed <= rows && (region || sessions == 1)
        && (!padded || (region && sessions > 1)) && block <= rows && (block == 0 || (region && sessions > 1)));
    if block > 0 {
        return 2 * (rows + 1) * (MAX_BATCHED_SESSIONS + 1)
            + ((usize::from(padded) * (MAX_BATCHED_SESSIONS - 1) + (sessions - 2)) * (rows + 1) + block) * (rows + 1) + executed;
    }
    let slot = if region && sessions == 1 { MAX_BATCHED_SESSIONS * (rows + 1) + executed }
        else { (sessions - 1) * (rows + 1) + executed };
    if padded { slot + (rows + 1) * (MAX_BATCHED_SESSIONS + 1) } else { slot }
}
/// Rows a batched step executes (Plan::batched_extent).
pub(crate) fn batched_extent(lives: &[usize], rows: usize) -> usize { Plan::batched_extent(lives, rows) }
/// Rows per session block of a batched step (Plan::batched_block).
pub(crate) fn batched_block(rows: usize, sessions: usize) -> usize { Plan::batched_block(rows, sessions) }
/// Bumped whenever the shared key caches may change other than by appends at a conversation's
/// frontier: image restores, resets, relocations, moved histories, and state handed over from a
/// prefill owner. Region gathers (CANADA_SCRATCH_REGION_GATHER_CACHE=1) re-gather fully after any.
static INDEX_CACHE_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
/// Per owner: how far each serving region's pair and single index keys are gathered, as of an epoch.
#[derive(Clone, Copy, Default)]
struct RegionGather { epoch: u64, pair: [i32; MAX_BATCHED_SESSIONS], single: [i32; MAX_BATCHED_SESSIONS] }
/// Key rows below a region's gathered end that a frontier step may rewrite (rejected rows, blocks
/// still filling): re-gathered every step.
const REGION_GATHER_MARGIN: i32 = 32;
/// Batched steps of several sessions route their padding rows to a live row's experts
/// (CANADA_SCRATCH_BATCHED_PADDING_ROUTES=1): padding rows then read no expert of their own.
fn padding_routes_requested() -> bool {
    std::env::var("CANADA_SCRATCH_BATCHED_PADDING_ROUTES").as_deref() == Ok("1")
}
/// Where one batched session lives in a verify owner's caches. Sessions occupy
/// disjoint linear regions: window slots, pair and single raw addresses are
/// offset, and since every key mapping is linear, a session's first visible
/// logical keys are raw_pair / 2 and raw_single. `request` selects the
/// session's pair-compression ring. All zero is today's single sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SessionBase {
    pub(crate) request: i64,
    pub(crate) window: usize,
    pub(crate) raw_pair: usize,
    pub(crate) raw_single: usize,
    /// The region's window ring (None: the owner's layout).
    pub(crate) layout: Option<crate::swa_layout::Layout>,
}
impl SessionBase {
    /// Serving region `index` of `regions` (CANADA_SCRATCH_BATCHED_REGIONS): region 0 is today's
    /// layout; region k >= 1 has a small ring after region 0's ring and the earlier small rings,
    /// request slot k and the k-th share of the compressed address space.
    pub(crate) fn serving_region(index: usize, regions: usize, context: usize,
        layout: crate::swa_layout::Layout) -> Result<Self> {
        Self::serving_region_with(index, regions, context, layout, crate::config::full_regions()?)
    }
    pub(crate) fn serving_region_with(index: usize, regions: usize, context: usize,
        layout: crate::swa_layout::Layout, full: bool) -> Result<Self> {
        ensure!(index < regions && regions <= 4 && layout.ring().is_some(), "serving region geometry");
        if index == 0 { return Ok(Self::default()); }
        let capacity = Self::serving_capacity_with(context, regions, full)?;
        let small = crate::swa_layout::Layout::Ring;
        Ok(Self { request: index as i64,
            window: layout.pages(context)? * 64 + (index - 1) * small.pages(context)? * 64,
            raw_pair: index * capacity, raw_single: index * capacity, layout: Some(small) })
    }
    /// Positions one serving region holds: an equal share of the context, rounded down to 128
    /// positions so every region's compressed and key addresses start on a 64-slot page.
    pub(crate) fn region_capacity(context: usize, regions: usize) -> usize {
        context / regions.max(1) / 128 * 128
    }
    /// Positions one serving region holds in this process: a whole conversation with full regions
    /// (config::full_regions: the caches span regions x context), else region_capacity's equal share.
    pub(crate) fn serving_capacity(context: usize, regions: usize) -> Result<usize> {
        Self::serving_capacity_with(context, regions, crate::config::full_regions()?)
    }
    pub(crate) fn serving_capacity_with(context: usize, regions: usize, full: bool) -> Result<usize> {
        if !full { return Ok(Self::region_capacity(context, regions)); }
        ensure!(crate::config::compressed_address_context_with(context, regions, true)? == context * regions,
            "full serving regions need the full context in every region");
        Ok(context)
    }
    /// Session `index` of `sessions`, each with `capacity` positions, where one
    /// window region spans `window_slots` slots including the shared offset.
    pub(crate) fn region(index: usize, capacity: usize, window_slots: usize) -> Result<Self> {
        ensure!(capacity % 2 == 0 && window_slots > 0, "batched session region geometry");
        Ok(Self { request: index as i64, window: index * window_slots, raw_pair: index * capacity,
            raw_single: index * capacity, layout: None })
    }
}

/// Integer scheduling only; numeric model execution remains on GPU.
#[derive(Debug)]
struct Plan {
    swa_layout: crate::swa_layout::Layout,
    positions: Vec<i32>,
    locations: Vec<i32>,
    raw_pair: Vec<i64>,
    raw_single: Vec<i64>,
    pair_lengths: Vec<i32>,
    single_lengths: Vec<i32>,
    swa_ids: Vec<i32>,
    swa_lengths: Vec<i32>,
    /// Per row: the session's compressor request slot and first visible logical
    /// pair/single keys. Zero for every row of a single sequence.
    requests: Vec<i64>,
    pair_starts: Vec<i32>,
    single_starts: Vec<i32>,
    /// Rows per verify compressor block: all rows for one session; a batched
    /// step gives each session its own block of rows / sessions rows.
    block: usize,
    /// Sessions of a batched step (1 for a single sequence): parts of `block` rows each.
    sessions: usize,
}
impl Plan {
    #[cfg(test)]
    fn append(position: u64, live: usize) -> Result<Self> {
        Self::append_rows(position, live, ROWS)
    }
    #[cfg(test)]
    fn append_rows(position: u64, live: usize, rows: usize) -> Result<Self> {
        Self::append_in(position, live, rows, CONTEXT)
    }
    #[cfg(test)]
    fn append_in(position: u64, live: usize, rows: usize, context: usize) -> Result<Self> {
        Self::append_with_layout(
            position,
            live,
            rows,
            context,
            crate::swa_layout::Layout::Absolute,
        )
    }
    fn append_with_layout(
        position: u64,
        live: usize,
        rows: usize,
        context: usize,
        swa_layout: crate::swa_layout::Layout,
    ) -> Result<Self> {
        Self::append_session(position, live, rows, context, swa_layout, SessionBase::default())
    }
    /// Rows of several sessions in one verification step, in part order. Part k
    /// (base, frontier position, live rows) occupies the block of rows
    /// [k * block, k * block + live), block = rows / parts, laid out exactly as that
    /// session's own append: the pair compressor's verify kernel runs one request
    /// per block of `block` rows, with padding (no-write rows) as a block suffix.
    /// Padding rows repeat their block's last live position and request.
    /// Rows per session block of a batched step: rows / sessions, at most six (a session verifies
    /// its anchor and at most five proposals), so two sessions on 16 rows take blocks of 6, not 8.
    pub(crate) fn batched_block(rows: usize, sessions: usize) -> usize {
        if sessions <= 1 { rows } else { (rows / sessions).min(crate::config::batched_block_cap()) }
    }
    /// The block a batched step is laid out in: the fixed share of the rows, or (tight blocks) the widest
    /// conversation of this step, at least two rows and at most that share.
    fn batched_layout_block(lives: &[usize], rows: usize) -> usize {
        let share = Self::batched_block(rows, lives.len());
        if lives.len() >= 2 && rows <= 32 && crate::config::batched_tight_blocks() {
            lives.iter().copied().max().unwrap_or(share).clamp(2, share.max(2))
        } else { share }
    }
    fn batched(parts: &[(SessionBase, u64, usize)], rows: usize, context: usize,
        swa_layout: crate::swa_layout::Layout) -> Result<Self> {
        // Blocks need not divide the rows (three sessions on 16 rows: blocks of 5); rows past
        // the last block are padding, never executed (batched_extent) nor compressed.
        ensure!(!parts.is_empty() && rows / parts.len() >= 1, "batched session block count");
        let block = Self::batched_layout_block(&parts.iter().map(|p| p.2).collect::<Vec<_>>(), rows);
        ensure!((parts.len() == 1 || (2..=16).contains(&block))
            && parts.iter().all(|p| (1..=block).contains(&p.2)), "batched session rows");
        let mut requests = std::collections::BTreeSet::new();
        ensure!(parts.iter().all(|p| requests.insert(p.0.request)), "one part per session");
        let mut o = Self { swa_layout, positions: vec![0; rows], locations: vec![-1; rows],
            raw_pair: vec![0; rows], raw_single: vec![0; rows], pair_lengths: vec![0; rows],
            single_lengths: vec![0; rows], swa_ids: vec![-1; rows * 128], swa_lengths: vec![0; rows],
            requests: vec![0; rows], pair_starts: vec![0; rows], single_starts: vec![0; rows], block,
            sessions: parts.len() };
        for (k, &(base, position, live)) in parts.iter().enumerate() {
            let p = Self::append_session(position, live, rows, context, swa_layout, base)?;
            for row in 0..block {
                let d = k * block + row;
                if row >= live {
                    o.positions[d] = p.positions[live - 1]; o.requests[d] = p.requests[live - 1];
                    // Empty key ranges at the part's own base: index scorers bound each block of query
                    // rows by its rows' ranges, so a padding row at key 0 would widen a region's block
                    // to span every region below it.
                    o.pair_starts[d] = p.pair_starts[live - 1]; o.single_starts[d] = p.single_starts[live - 1];
                    continue;
                }
                o.positions[d] = p.positions[row]; o.locations[d] = p.locations[row];
                o.raw_pair[d] = p.raw_pair[row]; o.raw_single[d] = p.raw_single[row];
                o.pair_lengths[d] = p.pair_lengths[row]; o.single_lengths[d] = p.single_lengths[row];
                o.swa_ids[d * 128..(d + 1) * 128].copy_from_slice(&p.swa_ids[row * 128..(row + 1) * 128]);
                o.swa_lengths[d] = p.swa_lengths[row]; o.requests[d] = p.requests[row];
                o.pair_starts[d] = p.pair_starts[row]; o.single_starts[d] = p.single_starts[row];
            }
        }
        // Rows past the last block (three sessions on 16 rows: row 15) are never executed, but the index scorers still
        // score every row: an empty range at key 0 widened the last scorer block (four query rows, bounded by the union
        // of their ranges) from the last session's region down to region 0, reading every region between (Nov 19 2026).
        // An empty range at the last session's own base keeps that block inside its region.
        if let Some(last) = parts.len().checked_sub(1) {
            let (pair, single) = (o.pair_starts[last * block], o.single_starts[last * block]);
            for d in parts.len() * block..rows {
                o.pair_starts[d] = pair; o.single_starts[d] = single;
            }
        }
        Ok(o)
    }
    /// Rows a batched step executes: through the last part's live rows.
    fn batched_extent(lives: &[usize], rows: usize) -> usize {
        (lives.len() - 1) * Self::batched_layout_block(lives, rows) + lives[lives.len() - 1]
    }
    fn append_session(
        position: u64,
        live: usize,
        rows: usize,
        context: usize,
        swa_layout: crate::swa_layout::Layout,
        base: SessionBase,
    ) -> Result<Self> {
        ensure!(base.raw_pair % 2 == 0 && base.request >= 0
            && (base == SessionBase::default() || swa_layout.ring().is_some()),
            "batched session base parity/layout");
        // A serving region may use its own (small) ring; slots are computed here only.
        let swa_layout = base.layout.unwrap_or(swa_layout);
        ensure!(
            crate::config::target_rows(rows)
                && rows <= swa_layout.max_rows()
                && crate::config::EXPERIMENTAL_CONTEXT_CAPACITIES.contains(&context)
                && (1..=rows).contains(&live),
            "target append row count"
        );
        let position = usize::try_from(position)?;
        let end = position
            .checked_add(live)
            .ok_or_else(|| anyhow::anyhow!("target position overflow"))?;
        ensure!(
            end <= context,
            "target integration profile context exceeded"
        );
        let mut p = Self {
            swa_layout,
            positions: vec![end as i32 - 1; rows],
            locations: vec![-1; rows],
            raw_pair: vec![0; rows],
            raw_single: vec![0; rows],
            pair_lengths: vec![0; rows],
            single_lengths: vec![0; rows],
            swa_ids: vec![-1; rows * 128],
            swa_lengths: vec![0; rows],
            requests: vec![0; rows],
            pair_starts: vec![0; rows],
            single_starts: vec![0; rows],
            block: rows,
            sessions: 1,
        };
        for row in 0..live {
            let pos = position + row;
            p.positions[row] = pos as i32;
            p.locations[row] = (base.window + swa_layout.slot(pos)) as i32;
            // Pair raw addresses retain absolute-position parity. Slot0 is
            // reserved by the compression kernel as the no-write sentinel.
            p.raw_pair[row] = (base.raw_pair + SWA_OFFSET + pos) as i64;
            p.raw_single[row] = (base.raw_single + COMPRESSED_OFFSET + pos) as i64;
            p.pair_lengths[row] = ((pos + 1) / 2) as i32;
            p.single_lengths[row] = (pos + 1) as i32;
            p.requests[row] = base.request;
            p.pair_starts[row] = i32::try_from(base.raw_pair / 2)?;
            p.single_starts[row] = i32::try_from(base.raw_single)?;
            let start = (pos + 1).saturating_sub(128);
            p.swa_lengths[row] = (pos + 1 - start) as i32;
            // Qualified BuildCausalSwaPageIndices visits newest to oldest.
            // Although the same keys are visible in either order, permuting
            // the attention reduction can change BF16 ties and downstream
            // routing. Keep the actual numeric-order contract.
            for (col, previous) in (start..=pos).rev().enumerate() {
                p.swa_ids[row * 128 + col] = (base.window + swa_layout.slot(previous)) as i32;
            }
        }
        Ok(p)
    }
    fn floor_swa_at(&mut self, floor: usize) -> Result<()> {
        ensure!(
            floor <= self.positions[0] as usize,
            "SWA floor after first live row"
        );
        self.swa_ids.fill(-1);
        self.swa_lengths.fill(0);
        for row in 0..self.positions.len() {
            if self.locations[row] < 0 {
                continue;
            }
            let pos = self.positions[row] as usize;
            let start = floor.max((pos + 1).saturating_sub(128));
            self.swa_lengths[row] = (pos + 1 - start) as i32;
            for (col, previous) in (start..=pos).rev().enumerate() {
                self.swa_ids[row * 128 + col] = self.swa_layout.slot(previous) as i32;
            }
        }
        Ok(())
    }
}

/// The caller supplies only the forty immutable Rust TP stages and IO stage.
/// There are no reference fixture paths or intermediate hidden-state inputs.
struct Step {
    mapped_metadata_bytes: usize,
    staged_metadata: Option<Rc<crate::cuda::MetadataStaging>>,
    positions: Positions,
    compressed_positions: Positions,
    locations: RowLocations,
    compressed_locations: RowLocations,
    candidates: Candidates,
    m2: Metadata,
    m8: Metadata,
    m14: Metadata,
    m20: Metadata,
}
impl Step {
    fn new(
        cuda: &Cuda,
        source2: &Source,
        source8: &Source,
        source14: &Source,
        source20: &Source,
        pair_slots: &[i32],
        single_slots: &[i32],
        rows: usize,
        context: usize,
        swa_capacity: usize,
        swa_layout: crate::swa_layout::Layout,
        phase: Phase,
    ) -> Result<Self> {
        let requested=match std::env::var("CANADA_SCRATCH_MAPPED_METADATA") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid mapped metadata policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(e)=>return Err(e.into()),
        };
        let staged=match std::env::var("CANADA_SCRATCH_STAGED_METADATA") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid staged metadata policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(e)=>return Err(e.into()),
        };
        ensure!(!(requested && staged),"exclusive metadata transport policies");
        // Only this constructor's small integer control buffers use mapping.
        // No allocator override escapes into a model-value constructor.
        let allocator=if requested && rows==6 && phase==Phase::Verify {
            unsafe {cuda.mapped_metadata_allocator()?}
        }else if staged && rows==6 && phase==Phase::Verify {
            unsafe {cuda.staged_metadata_allocator()?}
        }else{cuda.clone()};
        let cuda=&allocator;
        let p = Plan::append_with_layout(
            0,
            if phase == Phase::Decode { 1 } else { rows },
            rows,
            context,
            swa_layout,
        )?;
        let pair = |s: &Source| {
            Metadata::new(
                cuda,
                s,
                &p.positions,
                &vec![0; p.positions.len()],
                &p.raw_pair,
                pair_slots,
                &p.pair_lengths,
                1048576,
            )
        };
        let mut result=Self {
            mapped_metadata_bytes: 0,
            staged_metadata: allocator.metadata_staging(),
            positions: Positions::upload(cuda, &p.positions, context.max(131072))?,
            compressed_positions: Positions::upload(cuda, &p.positions, 1048576)?,
            locations: RowLocations::upload(
                cuda,
                &p.positions,
                &p.locations,
                swa_capacity,
                context.max(131072),
            )?,
            compressed_locations: RowLocations::upload(
                cuda,
                &p.positions,
                &p.locations,
                swa_capacity,
                1048576,
            )?,
            candidates: Candidates::upload(
                cuda,
                rows,
                128,
                swa_capacity,
                &p.swa_ids,
                &p.swa_lengths,
            )?,
            m2: pair(source2)?,
            m8: pair(source8)?,
            m14: pair(source14)?,
            m20: Metadata::new(
                cuda,
                source20,
                &p.positions,
                &vec![0; p.positions.len()],
                &p.raw_single,
                single_slots,
                &p.single_lengths,
                1048576,
            )?,
        };
        result.mapped_metadata_bytes=allocator.mapped_metadata_bytes();
        if let Some(staging)=&result.staged_metadata {
            // Constructor setters wrote the pinned source; publish all initial
            // metadata before the new profile becomes available to a graph.
            unsafe {staging.flush(None)?;}
        }
        Ok(result)
    }
    fn update(&mut self, p: &Plan, stream:&Stream) -> Result<()> {
        self.update_with(p, None, stream)
    }
    /// As update(), with region steps' per-row index gather starts (pair, single).
    fn update_with(&mut self, p: &Plan, gathers: Option<(&[i32], &[i32])>, stream:&Stream) -> Result<()> {
        // Mapped control storage may be read by captured kernels. A single
        // completion boundary prevents CPU writes racing any previous replay.
        if self.mapped_metadata_bytes>0 || self.staged_metadata.is_some() {stream.synchronize()?;}
        self.positions.update(&p.positions)?;
        self.compressed_positions.update(&p.positions)?;
        self.locations.update(&p.positions, &p.locations)?;
        self.compressed_locations
            .update(&p.positions, &p.locations)?;
        self.candidates.update(&p.swa_ids, &p.swa_lengths)?;
        for m in [&mut self.m2, &mut self.m8, &mut self.m14] {
            m.update_ranged_blocks(
                &p.positions,
                &p.requests,
                &p.raw_pair,
                &p.pair_starts,
                &p.pair_lengths,
                p.block,
            )?;
        }
        self.m20.update_ranged(
            &p.positions,
            &p.requests,
            &p.raw_single,
            &p.single_starts,
            &p.single_lengths,
        )?;
        if let Some((pair, single)) = gathers {
            for m in [&mut self.m2, &mut self.m8, &mut self.m14] { m.set_gather_starts(pair)?; }
            self.m20.set_gather_starts(single)?;
        }
        let parts: Vec<(usize, usize)> = if p.sessions > 1 { (0..p.sessions).map(|k| (k * p.block, p.block)).collect() } else { Vec::new() };
        for m in [&mut self.m2, &mut self.m8, &mut self.m14, &mut self.m20] { m.set_parts(&parts)?; }
        if let Some(staging)=&self.staged_metadata {
            unsafe {staging.flush(Some(stream))?;}
        }
        Ok(())
    }
}
struct PendingVerification {
    hash_batch: HashBatch,
    start: u64,
    live: usize,
}
// Probe-only second graph set. Graphs precede their captured metadata owner in
// drop order; all model weights, causal state and activation buffers stay shared.
struct TinyAppendGraphs {
    graphs: Vec<Option<Graph>>,
    engram_graphs: Vec<Option<Vec<Graph>>>,
    alternate_index_graphs: Vec<Option<Graph>>,
    alternate_index_engram_graphs: Vec<Option<Vec<Graph>>>,
    decoder_graph: Option<Graph>,
    adaptive_sparse_active: bool,
    decoder: Option<Step>,
}
pub struct Target {
    graph_retention: crate::graph_retention::Policy,
    graph_evictions: usize,
    rows: usize,
    source_phase: Phase,
    tiny_native_reduce: bool,
    tiny_prefix_sum: bool,
    tiny_prefix_gather: bool,
    prefill_dense_live_rows: bool,
    prefill_moe_live_rows: bool,
    prefill_upper_rows: bool,
    prefill_woa_live: bool,
    prefill_partial_dense_rows: bool,
    prefill_partial_moe_rows: bool,
    // 1: qualified dense bounds; 2 also enables the experimental expert bound.
    prefill_partial1024_rows: u8,
    prefill_partial2048_dense: bool,
    // Partial chunks on the 4,096/8,192-row profiles: bit 0 dense, bit 1 routed experts.
    prefill_partial_wide: u8,
    /// The selected L2 hint queue plan of the six-row decode layers (config::l2_queue_plans), shared with them.
    hint_queue_selected: Rc<std::cell::Cell<Option<usize>>>,
    /// Screens only: the routed experts' front-end kernel launches this many blocks in the verification graphs.
    moe_front_grid: Option<u32>,
    /// The routed experts' phase kernels (FC1 and FC2; 96 blocks each as exported) launch (FC1, FC2) blocks in
    /// the verification graphs, with this shared-memory carve-out preference when given (cuda::Regrid).
    moe_phase_grid: Option<(u32, u32, Option<i32>)>,
    /// Kernel nodes the grid rules matched in the last verification graph capture.
    moe_regridded: usize,
    /// Exported front-end nodes replaced by the native kernel in the last verification graph capture (screens).
    moe_swapped: usize,
    /// dspark_generation::verify_sample_ahead: the next verification returns without waiting for the GPU or reading
    /// its fault words; the caller queues its sampler behind the forward and then calls
    /// complete_deferred_verification (`completion_owed` until it does).
    defer_completion: bool,
    completion_owed: std::cell::Cell<bool>,
    /// The configured live-row floors of the experts' variants, remembered by the first screen toggle.
    moe_live_floors: Option<(usize, usize, bool)>,
    /// config::l2_queue_engram_lead: the switches of the Engram projection hints begun by layers 0 and 13 (shared
    /// with those layers' queues) and the wanted state; layer 14's also follows the step's projection schedule.
    engram_lead: Option<(Rc<std::cell::Cell<bool>>, Rc<std::cell::Cell<bool>>)>,
    engram_lead_on: bool,
    prefill_wide_mhc_rows: bool,
    prefill_wide_mhc_norm: bool,
    prefill_short_merge: bool,
    prefill_mhc_live_rows: bool,
    prefill_mhc_live_norm: bool,
    prefill_blocking_wait: bool,
    verification_poll_us: u64,
    verification_mhc_live_rows: usize,
    // Destroy captured executables before their owned pointer storage. Each
    // live-row count has a distinct collective capture identity. Retain those
    // bounded variants through tool gaps/resets instead of recapturing when
    // an accepted increment changes size. All variants use the same buffers.
    graphs: Vec<Option<Graph>>,
    engram_graphs: Vec<Option<Vec<Graph>>>,
    // Adaptive consumer indexing changes graph topology. Keep a second bank
    // instead of replaying a dense graph after switching to sparse, or paying
    // capture again when a conversation resets below the long-context cutoff.
    alternate_index_graphs: Vec<Option<Graph>>,
    alternate_index_engram_graphs: Vec<Option<Vec<Graph>>>,
    adaptive_sparse_active: bool,
    /// config::index_short_rows: the eight index branches' switches (collected at the first step), whether the
    /// worker allows the short path for the current request (the alternate screen), and steps taken on it.
    short_index_flags: Vec<Rc<std::cell::Cell<bool>>>,
    short_index_allowed: bool,
    short_index_steps: u64,
    engram_overlap: bool,
    engram_projection_ahead: bool,
    engram_projection_m3_only: bool,
    engram_projection_active: bool,
    engram_split14: bool,
    engram_split14_m3_only: bool,
    engram_split14_adaptive: bool,
    paired_engram_io: bool,
    paired_engram32: bool,
    paired_engram32_adaptive: bool,
    engram32_worker: Option<rayon::ThreadPool>,
    engram32_worker_enabled: bool,
    graph_captures: usize,
    // Validated first-use optimization. Recording a graph does not execute
    // it: retain the successful eager prefill instead of immediately replaying.
    prefill_capture_once: bool,
    // Opt-in first-use verification experiment. Keep the successful eager
    // result pending while recording its future replay, without executing twice.
    verification_capture_once: bool,
    carry_snapshot: Option<crate::carry_state::Snapshot>,
    carry_saved_start: Option<u64>,
    repeatable_frontier_measurement: bool,
    encoder_graph: Option<Graph>,
    encoder_graph_live: Option<usize>,
    decoder_graph: Option<Graph>,
    tiny_append_graphs: Option<TinyAppendGraphs>,
    encoder_cache: Option<Rc<crate::decoder_tail::Cache>>,
    /// Batched serving: the encoder tails of serving regions 1.. (region 0's is encoder_cache).
    region_tails: Vec<Rc<crate::decoder_tail::Cache>>,
    // Inline encoder checkpoint: side state (decoder-tail owner only) and, for the finite
    // probe, the chunk plan of the next checkpoint prefill.
    checkpoint_aux: Option<Rc<CheckpointAux>>,
    // A second inline capture in the same last chunk (CANADA_SCRATCH_ENCODER_EXTRA_NEAR_END): its side state, the
    // caller's request for the next prefill (rows from the restore point to the extra position, image bytes), and
    // the image that prefill saved.
    checkpoint_aux_extra: Option<Rc<CheckpointAux>>,
    extra_checkpoint: Option<(usize, usize)>,
    extra_image: Option<EncoderPrefixImage>,
    checkpoint_plan_for_probe: Option<Vec<usize>>,
    checkpoint_inline_for_probe: Option<bool>,
    replay: Option<crate::decoder_tail::Replay>,
    graph_replay: bool,
    prefill_last_head: bool,
    // Deferred layer 20 (CANADA_SCRATCH_PREFILL_LAYER20_DEFER): set by the wide
    // prefill loop for one eager chunk; pending until finished or abandoned.
    layer20_projection_only: bool,
    layer20_pending: bool,
    append_last_gather: bool,
    append_encoded_head: bool,
    comm_instance: Option<u64>,
    // Its sampling graph references IO's local head, so drop before IO.
    compact_categorical: Option<crate::sharded_categorical_probe::CompactSampler>,
    compact_sampling_enabled: bool,
    local_target_enabled: bool,
    // Native row transport fixes the live count inside each capture. A tail
    // captured for three rows cannot publish a later six-row distribution.
    local_target_gather: Vec<Option<Graph>>,
    last_head_local_only: bool,
    cuda: Cuda,
    io: ModelIo,
    sampler: Greedy,
    categorical: crate::categorical::Categorical,
    last_live: usize,
    last_append_metrics: Option<AppendMetrics>,
    pending: Option<PendingVerification>,
    /// Finite batched screen (CANADA_SCRATCH_BATCH_SCREEN=1) only: the next
    /// execute_rows lays out these sessions' rows in order instead of this
    /// owner's single history; appends hand the advanced histories back and a
    /// batched verification is only ever aborted (full carry restore).
    batched_rows: Option<Vec<(SessionBase, History, usize)>>,
    batched_done: Option<Vec<History>>,
    batched_pending: bool,
    /// The current step has batched rows and runs in region mode (set_region_step).
    region_step: bool,
    /// Batched serving: part 0 of the next batched verification is this owner's own conversation
    /// (region 0); its commit also advances the owner's history and encoder tail.
    batched_owner_part: bool,
    /// The sessions of an in-flight batched verification, with their own hash batches.
    batched_inflight: Option<Vec<(SessionBase, History, usize, HashBatch)>>,
    /// While one session of a batched verification samples: its rows were moved
    /// to row 0 and the logits accessors report its live count.
    batched_staged: Option<(usize, usize, u64)>,
    /// Rows per session block of the in-flight batched verification.
    batched_block: usize,
    /// Batched steps of several sessions use padding routes (see graph_slot).
    padding_routes: bool,
    /// Region-mode index gathers already done by this owner (CANADA_SCRATCH_REGION_GATHER_CACHE=1).
    region_gather: RegionGather,
    early: Vec<SwaLayer>,
    early_caches: Vec<PagedKv>,
    source2: Source,
    group8: crate::source_group::Group,
    group14: crate::source_group::Group,
    late: crate::late_group::Group,
    step: Step,
    cached_decoder: Option<Step>,
    /// The next short restored suffix runs as the qualified EXTEND (CANADA_SCRATCH_ENCODER_SHORT_SUFFIX=2).
    short_restore_extend: bool,
    tokens: TokenRows,
    row_store1: Rc<RowStore>,
    rows1: Rows,
    hashes: HashMetadata,
    history: History,
    lease: crate::session_lease::Lease,
    context: usize,
    swa_layout: crate::swa_layout::Layout,
    ordinary: Rc<FrequencyTable>,
    compressed: Rc<FrequencyTable>,
    vendor_blas: Rc<Blas>,
    // Present only on the encoder-only wide prefill profile.
    wide: Option<Rc<crate::decoder::WideBuild>>,
    // Sequence-parallel wide profile: full-row residual/pre gathered from the
    // row shards for the full-row layer 14 and the encoder-tail publication.
    sp_full: Option<(Buffer<bf16>, Buffer<f32>)>,
    rank: usize,
    stream: usize,
    // Profiles share transient scores only; graph pointers stay stable and
    // the existing revision lease still permits exactly one target writer.
    index_score_arena: Option<Rc<Buffer<f32>>>,
    index_score_logical_bytes: usize,
    pub atlas_bytes: Vec<usize>,
    pub source_hashes: Vec<String>,
}
impl Target {
    pub(crate) fn draft_resources(
        &self,
        core: &[u8],
        stream: &Stream,
        kernels: &Path,
    ) -> Result<(ModelIo, Rc<FrequencyTable>)> {
        ensure!(
            self.lease.is_current()
                && self.cuda.owns_stream(stream)
                && self.stream == stream.raw() as usize,
            "draft resources require healthy current target owner"
        );
        Ok((
            self.io.fork_profile(&self.cuda, core, stream, kernels, 5)?,
            self.ordinary.clone(),
        ))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        cuda: &Cuda,
        core: &Module,
        core_bytes: &[u8],
        stream: &Stream,
        cp: &Checkpoint,
        plan: &RankPlan,
        stages: &[PathBuf],
        io_stage: &Path,
        kernels: &Path,
        backing: &Path,
        hashes: HashMetadata,
        lineage: u64,
        source_phase: Phase,
        rows: usize,
        context: usize,
    ) -> Result<Self> {
        ensure!(
            stages.len() == 40
                && plan.world == 4
                && plan.rank < 4
                && lineage != 0
                && crate::config::EXPERIMENTAL_CONTEXT_CAPACITIES.contains(&context)
                && [6, 128].contains(&rows)
                && (rows == 6 || source_phase == Phase::Extend),
            "complete TP4 target construction"
        );
        hashes.validate()?;
        cuda.qualify_gb10()?;
        ensure!(cuda.owns_stream(stream), "target stream context");
        // An external parent has already physically allocated the target
        // atlases. Validate and import them before crediting any such bytes
        // in admission. The environment flag alone is not an admission credit.
        let mut reserved_weights=if crate::weights::external_target_pool_path()?.is_some() {
            Some(crate::weights::StagedReservation::target_layers(cuda,cp,plan,stages)?)
        }else{None};
        if rows > 8 && crate::woa_medium::cached_policy()? {
            let derived = 40 * 2048 * 4096 * 2usize;
            let remaining=(plan.device_weight_bytes as usize)
                .checked_sub(reserved_weights.as_ref().map_or(0,|r|r.already_allocated_bytes()))
                .context("imported target weights exceed model weight forecast")?;
            ensure!(cuda.admission_memory()?.admitted_available
                > remaining + derived + crate::memory::floor_bytes(),
                "target weights plus exact WO-A cache would consume reserved headroom");
        }
        if reserved_weights.is_none() && crate::weights::early_layer_reservation_requested()? {
            reserved_weights=Some(crate::weights::StagedReservation::target_layers(cuda,cp,plan,stages)?);
        }
        let swa_layout = if std::env::var("CANADA_SCRATCH_SWA_RING").as_deref() == Ok("wide16384") {
            crate::swa_layout::Layout::WideRing16384
        } else if std::env::var("CANADA_SCRATCH_SWA_RING").as_deref() == Ok("wide8192") {
            crate::swa_layout::Layout::WideRing8192
        } else if std::env::var("CANADA_SCRATCH_SWA_RING").as_deref() == Ok("wide") {
            crate::swa_layout::Layout::WideRing
        } else if std::env::var("CANADA_SCRATCH_SWA_RING").as_deref() == Ok("1") {
            crate::swa_layout::Layout::Ring
        } else {
            crate::swa_layout::Layout::Absolute
        };
        ensure!(
            context <= 65536
                || (std::env::var("CANADA_SCRATCH_FULL_CONTEXT").as_deref() == Ok("1")
                    && matches!(swa_layout, crate::swa_layout::Layout::WideRing
                        | crate::swa_layout::Layout::WideRing8192 | crate::swa_layout::Layout::WideRing16384)),
            "contexts above64K require the explicit full-context experiment and bounded wide SWA ring"
        );
        let ordinary = Rc::new(FrequencyTable::new(cuda, core, stream, context.max(131072), false)?);
        let compressed = Rc::new(FrequencyTable::new(cuda, core, stream, 1048576, true)?);
        let pair_slots = (COMPRESSED_OFFSET..COMPRESSED_OFFSET + context / 2)
            .map(|s| s as i32)
            .collect::<Vec<_>>();
        let single_slots = (COMPRESSED_OFFSET..COMPRESSED_OFFSET + context)
            .map(|s| s as i32)
            .collect::<Vec<_>>();
        // A finite batched screen keeps one window region per session (see SessionBase).
        let sessions = crate::config::batch_screen_sessions()?;
        ensure!(sessions == 1 || swa_layout.ring().is_some(), "batched sessions need a ring window layout");
        let regions = crate::config::batched_regions()?;
        ensure!(regions == 1 || swa_layout.ring().is_some(), "batched regions need a ring window layout");
        let swa_pages = swa_layout.pages(context)? * sessions
            + (regions - 1) * crate::swa_layout::Layout::Ring.pages(context)?;
        let page_tokens = if source_phase == Phase::Extend {
            64
        } else {
            32
        };
        // Full serving regions (config::full_regions): the compressed caches and their key caches hold every
        // region's whole conversation; positions, slots and the frequency tables stay per conversation.
        let address = crate::config::compressed_address_context(context)?;
        let pair_pages = compressed_pages(address, 2, page_tokens, 4);
        let single_pages = compressed_pages(address, 1, page_tokens, 8);
        let pair_key_pages = compressed_pages(address, 2, 64, 2);
        let single_key_pages = compressed_pages(address, 1, 64, 4);
        // Reject an inconsistent retained-state geometry before loading all
        // forty weight atlases, rather than at later cache activation.
        crate::decoder_tail::Cache::validate_capacity(context, single_pages * page_tokens)?;
        let vendor_blas = Rc::new(Blas::new_on(cuda,stream)?);
        let medium_woa = if rows == 128 {
            Some(crate::woa_medium::Scratch::new_on(cuda, core_bytes, rows,stream)?)
        } else {
            None
        };
        let mut early = Vec::new();
        let mut early_caches = Vec::new();
        let mut source2 = None;
        let mut atlas_bytes = Vec::new();
        let mut source_hashes = Vec::new();
        for (layer, stage) in stages[..8].iter().enumerate() {
            let mut w =
                Weights::load_staged_reserved(cuda, cp, plan, &[&format!("layers.{layer}.")], stage, reserved_weights.as_mut())?;
            atlas_bytes.push(w.bytes());
            source_hashes.push(w.source_sha256.clone());
            let prepared = expert_packing::prepare_layer(cuda, core, &mut w, layer)?;
            let w = Rc::new(w);
            early.push(SwaLayer::new_rows(
                cuda,
                core,
                stream,
                w.clone(),
                prepared,
                kernels,
                if layer < 2 {
                    ordinary.clone()
                } else {
                    compressed.clone()
                },
                InputMode::Split(plan.rank),
                rows,
                Some(vendor_blas.clone()),
                medium_woa.clone(),
            )?);
            early_caches.push(PagedKv::new_layout(cuda, swa_pages, 64, crate::attention::kv_format()?.window)?);
            if layer == 2 {
                source2 = Some(Source::new_rows(
                    cuda,
                    stream,
                    w,
                    vendor_blas.clone(),
                    2,
                    kernels,
                    compressed.clone(),
                    core_bytes,
                    pair_pages,
                    pair_key_pages,
                    &pair_slots,
                    &vec![0; rows],
                    rows,
                    source_phase,
                )?);
            }
            eprintln!("rank{} target loaded layer{layer}", plan.rank);
        }
        let group8 = crate::source_group::Group::new_rows(
            cuda,
            core,
            core_bytes,
            stream,
            cp,
            plan,
            8,
            &stages[8..14],
            kernels,
            compressed.clone(),
            &pair_slots,
            pair_pages,
            pair_key_pages,
            backing,
            source_phase,
            rows,
            Some(vendor_blas.clone()),
            medium_woa.clone(),
            swa_pages,
            reserved_weights.as_mut(),
        )?;
        let group14 = crate::source_group::Group::new_rows(
            cuda,
            core,
            core_bytes,
            stream,
            cp,
            plan,
            14,
            &stages[14..20],
            kernels,
            compressed.clone(),
            &pair_slots,
            pair_pages,
            pair_key_pages,
            backing,
            source_phase,
            rows,
            Some(vendor_blas.clone()),
            medium_woa.clone(),
            swa_pages,
            reserved_weights.as_mut(),
        )?;
        let late = crate::late_group::Group::new_rows(
            cuda,
            core,
            core_bytes,
            stream,
            cp,
            plan,
            &stages[20..40],
            kernels,
            compressed.clone(),
            &single_slots,
            single_pages,
            single_key_pages,
            source_phase,
            rows,
            Some(vendor_blas.clone()),
            medium_woa.clone(),
            swa_pages,
            reserved_weights.as_mut(),
        )?;
        if let Some(reservation)=reserved_weights {
            eprintln!("rank{} early layer weight reservation consumed: {} bytes, individually owned buffers, lossless_compression={}",
                plan.rank,reservation.finish()?,crate::weights::compressed_target_weights_requested()?);
        }
        for (bytes, hashes) in [
            (&group8.atlas_bytes, &group8.source_hashes),
            (&group14.atlas_bytes, &group14.source_hashes),
            (&late.atlas_bytes, &late.source_hashes),
        ] {
            atlas_bytes.extend_from_slice(bytes);
            source_hashes.extend_from_slice(hashes);
        }
        let quarter = crate::config::engram_quarter(plan.rank)?;
        let row_store1 = Rc::new(RowStore::new(
            PackedShard::open(
                &backing.join(format!("engram-l1-r{quarter}of4.bin")),
                1,
                quarter,
                4,
                384006168,
            )?,
            crate::engram::target_cache_bytes()?,
        )?);
        let rows1 = Rows::new(cuda, rows, &row_store1)?;
        let mut io = ModelIo::new(cuda, core_bytes, stream, cp, plan, io_stage, kernels, rows)?;
        let append_encoded_head=source_phase==Phase::Verify&&matches!(rows,16|32)
            &&crate::config::append_encoded_head_requested()?;
        if append_encoded_head {
            ensure!(crate::config::append_last_gather_requested()?,"encoded append head requires final-only output");
            io.prepare_append_encoded(stream)?;
        }
        let source2 = source2.ok_or_else(|| anyhow::anyhow!("source2 absent"))?;
        let step = Step::new(
            cuda,
            &source2,
            group8.source(),
            group14.source(),
            late.source(),
            &pair_slots,
            &single_slots,
            rows,
            context,
            swa_pages * 64,
            swa_layout,
            source_phase,
        )?;
        stream.synchronize()?;
        let mut target = Self {
            rows,
            source_phase,
            tiny_native_reduce: false,
            prefill_dense_live_rows: short_prefill_rows_enabled("CANADA_SCRATCH_PREFILL_DENSE_LIVE_ROWS")?,
            prefill_moe_live_rows: short_prefill_rows_enabled("CANADA_SCRATCH_PREFILL_MOE_LIVE_ROWS")?,
            prefill_upper_rows: prefill_upper_rows_requested()?,
            prefill_woa_live: crate::woa_medium::prefill_live_policy()?,
            prefill_partial_dense_rows: short_prefill_rows_enabled("CANADA_SCRATCH_PREFILL_PARTIAL_DENSE_ROWS")?,
            prefill_partial_moe_rows: short_prefill_rows_enabled("CANADA_SCRATCH_PREFILL_PARTIAL_MOE_ROWS")?,
            prefill_partial1024_rows: match std::env::var("CANADA_SCRATCH_PREFILL_PARTIAL1024_ROWS").as_deref() {
                Err(std::env::VarError::NotPresent)|Ok("1")=>1,Ok("0")=>0,Ok("2")=>2,
                _=>anyhow::bail!("M1024 partial rows must be 0,1 or2"),
            },
            prefill_partial2048_dense: experimental_prefill_flag("CANADA_SCRATCH_PREFILL_PARTIAL2048_DENSE",true)?,
            prefill_partial_wide: crate::config::partial_wide_rows_requested()?,
            hint_queue_selected: Rc::new(std::cell::Cell::new(None)),
            moe_front_grid: None,
            moe_phase_grid: None,
            moe_regridded: 0,
            moe_swapped: 0,
            defer_completion: false,
            completion_owed: std::cell::Cell::new(false),
            moe_live_floors: None,
            engram_lead: None,
            engram_lead_on: false,
            prefill_wide_mhc_rows: experimental_prefill_flag("CANADA_SCRATCH_PREFILL_WIDE_MHC_ROWS",true)?,
            prefill_wide_mhc_norm: experimental_prefill_flag("CANADA_SCRATCH_PREFILL_WIDE_MHC_NORM",true)?,
            prefill_short_merge: experimental_prefill_flag("CANADA_SCRATCH_PREFILL_SHORT_MERGE",true)?,
            prefill_mhc_live_rows: short_prefill_rows_enabled("CANADA_SCRATCH_PREFILL_MHC_LIVE_ROWS")?,
            prefill_mhc_live_norm: short_prefill_rows_enabled("CANADA_SCRATCH_PREFILL_MHC_LIVE_NORM")?,
            prefill_blocking_wait: match std::env::var("CANADA_SCRATCH_PREFILL_BLOCKING_WAIT").as_deref() {
                Err(std::env::VarError::NotPresent)|Ok("0")=>false,Ok("1")=>true,
                _=>anyhow::bail!("prefill blocking wait must be0or1"),
            },
            verification_poll_us: Self::verification_poll_us_requested()?,
            verification_mhc_live_rows: verification_mhc_live_row_limit()?,
            tiny_prefix_sum: source_phase==Phase::Verify && [6,16].contains(&rows)
                && std::env::var("CANADA_SCRATCH_LIVE_ROW_TRANSPORT").as_deref()==Ok("1"),
            tiny_prefix_gather: source_phase==Phase::Verify && [6,16].contains(&rows)
                && std::env::var("CANADA_SCRATCH_LIVE_ROW_TRANSPORT").as_deref()==Ok("1"),
            graph_retention: crate::graph_retention::Policy::new(rows,crate::config::execution_graph_cache_rows()?),
            graph_evictions: 0,
            graphs: (0..graph_slots(rows)).map(|_| None).collect(),
            engram_graphs: (0..graph_slots(rows)).map(|_| None).collect(),
            alternate_index_graphs: (0..graph_slots(rows)).map(|_| None).collect(),
            alternate_index_engram_graphs: (0..graph_slots(rows)).map(|_| None).collect(),
            adaptive_sparse_active: false,
            short_index_flags: Vec::new(),
            short_index_allowed: true,
            short_index_steps: 0,
            engram_overlap: false,
            engram_projection_ahead: false,
            engram_projection_m3_only: false,
            engram_projection_active: false,
            engram_split14: false,
            engram_split14_m3_only: false,
            engram_split14_adaptive: false,
            paired_engram_io: match std::env::var("CANADA_SCRATCH_ENGRAM_PAIRED_IO") {
                Ok(value)=>{ensure!((value=="0" || value=="1") && (value=="0" ||
                    (cfg!(target_os="linux") && std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref()==Ok("1"))),
                    "paired Engram I/O requires Linux io_uring");value=="1"},
                // s416 byte/error checks and s417 retained-model integration pass.
                Err(std::env::VarError::NotPresent)=>cfg!(target_os="linux")
                    && std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref()==Ok("1"),
                Err(error)=>return Err(error.into()),
            },
            paired_engram32: false,
            paired_engram32_adaptive: false,
            engram32_worker: None,
            engram32_worker_enabled: false,
            graph_captures: 0,
            prefill_capture_once: match std::env::var("CANADA_SCRATCH_PREFILL_CAPTURE_ONCE") {
                Ok(value) => {
                    ensure!(value == "0" || value == "1", "invalid prefill capture policy");
                    value == "1"
                }
                Err(std::env::VarError::NotPresent) => true,
                Err(error) => return Err(error.into()),
            },
            verification_capture_once: match std::env::var("CANADA_SCRATCH_VERIFY_CAPTURE_ONCE") {
                Ok(value) => {
                    ensure!(value == "0" || value == "1", "invalid verification capture policy");
                    value == "1"
                }
                Err(std::env::VarError::NotPresent) => false,
                Err(error) => return Err(error.into()),
            },
            carry_snapshot: None,
            carry_saved_start: None,
            repeatable_frontier_measurement: false,
            encoder_graph: None,
            encoder_graph_live: None,
            decoder_graph: None,
            tiny_append_graphs: None,
            encoder_cache: None,
            region_tails: Vec::new(),
            checkpoint_aux: None,
            checkpoint_plan_for_probe: None,
            checkpoint_inline_for_probe: None,
            checkpoint_aux_extra: None,
            extra_checkpoint: None,
            extra_image: None,
            replay: None,
            graph_replay: false,
            prefill_last_head: false,
            layer20_projection_only: false,
            layer20_pending: false,
            append_last_gather: source_phase==Phase::Verify && [6,16,32].contains(&rows)
                && crate::config::append_last_gather_requested()?,
            append_encoded_head,
            comm_instance: None,
            cuda: cuda.clone(),
            io,
            sampler: Greedy::new(cuda, core_bytes, rows)?,
            categorical: crate::categorical::Categorical::new(cuda, core_bytes, rows)?,
            compact_categorical: None,
            compact_sampling_enabled: false,
            local_target_enabled: rows==6&&source_phase==Phase::Verify
                &&crate::sampler_transport::Policy::from_env()?.local_target,
            local_target_gather: (0..=rows).map(|_|None).collect(),
            last_head_local_only: false,
            last_live: 0,
            last_append_metrics: None,
            pending: None,
            batched_rows: None,
            batched_done: None,
            batched_pending: false,
            region_step: false,
            batched_owner_part: false,
            batched_inflight: None,
            batched_staged: None,
            batched_block: 0,
            padding_routes: padding_routes_requested(),
            region_gather: RegionGather::default(),
            early,
            early_caches,
            source2,
            group8,
            group14,
            late,
            step,
            cached_decoder: None,
            short_restore_extend: false,
            tokens: TokenRows::new(cuda, &vec![2; rows], 0)?,
            row_store1,
            rows1,
            hashes,
            history: History::new(lineage),
            lease: crate::session_lease::Lease::new(),
            context,
            swa_layout,
            ordinary,
            compressed,
            vendor_blas,
            wide: None,
            sp_full: None,
            rank: plan.rank,
            stream: stream.raw() as usize,
            index_score_arena: None,
            index_score_logical_bytes: 0,
            atlas_bytes,
            source_hashes,
        };
        target.share_index_scores_before_capture(cuda,stream)?;
        target.share_narrow_moe_before_capture(cuda,stream)?;
        target.enable_verify16_ffn_before_capture(stream)?;
        target.enable_engram_projection_before_capture(kernels)?;
        target.install_communication_hints(core, stream)?;
        target.install_hint_queues(core, stream)?;
        target.install_source_keys_fork(stream)?;
        target.install_index_fork(stream)?;
        target.install_narrow_qkv(kernels)?;
        Ok(target)
    }

    /// Build private execution workspace over the same canonical atlases and
    /// causal storage. Only one revision-current profile may enqueue a write.
    pub fn fork_profile(
        &self,
        core: &Module,
        core_bytes: &[u8],
        stream: &Stream,
        kernels: &Path,
        rows: usize,
        phase: Phase,
    ) -> Result<Self> {
        ensure!([6, 128].contains(&rows) || ([16,32].contains(&rows) && phase == Phase::Verify
            && crate::config::verification_rows(rows)
            && crate::config::verification_carry_rows() > rows), "shared execution row profile");
        self.fork_profile_in(core, core_bytes, stream, kernels, rows, phase, None, None)
    }
    /// This arena holds only wide-prefill local/combined/expanded embeddings.
    /// Terminal logits, causal state, narrow profiles and metadata are separate.
    /// The caller must run image encoding between completed prefill chunks.
    pub(crate) fn image_scratch_arena(&self,stream:&Stream)->Result<Option<Rc<crate::cuda::RowArena>>> {
        ensure!(self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,
            "image scratch donor stream mismatch");
        Ok(self.wide.as_ref().and_then(|w|w.input_arena()).cloned())
    }
    /// All narrow profiles of one owner run sequentially. The two owners use
    /// different prefill parity arenas, so pairedM6 keeps disjoint transients.
    /// Causal/constant storage uses allocate(), outside allocate_rows() loans.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fork_profile_lending_prefill_for_probe(
        &self,core:&Module,core_bytes:&[u8],stream:&Stream,kernels:&Path,
        rows:usize,phase:Phase,donor:&Self,
    )->Result<Self> {
        ensure!(std::env::var("CANADA_SCRATCH_NARROW_ARENA_LOAN_SCREEN").as_deref()==Ok("1")
            &&crate::config::shared_owner_profiles_requested()
            &&[6,16,32].contains(&rows)&&phase==Phase::Verify
            &&self.position()==0&&donor.position()==0&&self.stream==donor.stream
            &&self.context==donor.context&&self.rank==donor.rank,
            "finite narrow arena loan needs fresh same-stream profiles");
        let shared=self.shared_state_report(donor);
        ensure!(shared["all40layer_atlases_shared"]==true&&shared["IO_atlas_shared"]==true,
            "narrow arena loan model identity");
        let slot=usize::from(!self.lease.shares_owner(&donor.lease));
        let arena=donor.wide.as_ref().context("narrow arena donor")?.transient_arena_for_probe(slot)?;
        self.cuda.with_row_arena(&arena,||self.fork_profile(core,core_bytes,stream,kernels,rows,phase))
    }
    pub(crate) fn independent_probe_context_admitted(context:usize)->bool {
        context<=65536 || (context<=crate::config::MODEL_CONTEXT
            &&(std::env::var("CANADA_SCRATCH_PAIRED_LONG_CONTEXT_SCREEN").as_deref()==Ok("1")
                ||crate::config::resident_banks_requested()))
    }
    /// Finite two-owner prototype. Only immutable weights, rotary tables and
    /// key-addressed Engram row stores are shared. Neither causal storage nor
    /// prompt-media bindings, history or revision authority cross the boundary.
    /// The small-context gate is diagnostic admission, never a serving limit.
    pub(crate) fn independent_owner_for_probe(
        &self, core: &Module, core_bytes: &[u8], stream: &Stream,
        kernels: &Path, lineage: u64, prefill_arena:Option<&Self>,
    ) -> Result<Self> {
        ensure!((std::env::var("CANADA_SCRATCH_INDEPENDENT_OWNER_SCREEN").as_deref()==Ok("1")
                ||crate::config::resident_banks_requested())
            && self.rows==128 && self.source_phase==Phase::Extend && self.wide.is_none()
            && self.position()==0 && self.pending.is_none() && self.encoder_cache.is_some()
            && Self::independent_probe_context_admitted(self.context) && lineage>self.history.lineage,
            "independent owner requires an admitted fresh bounded prototype");
        let causal_bytes=self.state_image_regions(1,false)?.iter().map(|r|r.bytes()).sum::<usize>();
        // Includes private row scratch and transient construction before the
        // new owner coalesces its index arenas. Keep the physical floor intact.
        ensure!(self.cuda.admission_memory()?.admitted_available
            > causal_bytes + 4*1024usize.pow(3) + crate::memory::floor_bytes(),
            "independent owner construction would consume reserved headroom");
        let build=||self.fork_profile_in(core,core_bytes,stream,kernels,128,Phase::Extend,None,Some(lineage));
        let mut target=if let Some(donor)=prefill_arena {
            ensure!(crate::config::shared_owner_profiles_requested()
                &&self.shared_state_report(donor)["all_weights_and_causal_storage_shared"]==true
                &&donor.position()==0&&self.stream==donor.stream,"independent construction arena authority");
            let arena=donor.wide.as_ref().context("independent prefill arena donor")?.transient_arena_for_probe(1)?;
            self.cuda.with_row_arena(&arena,build)?
        }else{build()?};
        target.enable_decoder_tail(core_bytes)?;
        if self.cached_decoder.is_some() {target.enable_cached_extend_floor()?;}
        target.set_graph_replay(self.graph_replay)?;
        target.set_prefill_last_head(self.prefill_last_head)?;
        if self.io.prefill_encoded_enabled() {target.set_prefill_encoded_head(true,stream)?;}
        if self.io.prefill_row_pruned() {target.set_prefill_row_head(true,stream)?;}
        if self.io.sparse_publish_enabled() {target.set_prefill_sparse_publish(true,stream)?;}
        ensure!(self.cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes(),
            "independent owner consumed physical reserve");
        ensure!(self.independent_state_report(&target)?["independent_causal_owners"]==true,
            "independent owner construction retained shared mutable state");
        Ok(target)
    }
    /// Encoder-only wide EXTEND profile over the same atlases, causal storage,
    /// Engram stores and encoder-tail ring. It runs layers0..20 over up to
    /// `rows` prompt rows per forward; this profile's owner runs the 128-row
    /// decoder tail. Row-sized transients alternate between two layer-parity
    /// arenas, and one routed-expert workspace is shared by its layers.
    pub fn fork_wide(
        &self,
        core: &Module,
        core_bytes: &[u8],
        stream: &Stream,
        kernels: &Path,
        rows: usize,
        sequence_parallel: bool,
    ) -> Result<Self> {
        self.fork_wide_with_arenas(core, core_bytes, stream, kernels, rows, sequence_parallel, None)
    }
    /// Build a sequential profile bank. Optional arena sharing constructs the
    /// largest profile first, then lends only its transient parity arenas to
    /// smaller profiles. All profiles retain the same single-writer lease.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fork_wide_bank(
        &self, core: &Module, core_bytes: &[u8], stream: &Stream, kernels: &Path,
        widths: &[usize], sequence_parallel: bool,
    ) -> Result<Vec<Self>> {
        let shared = match std::env::var("CANADA_SCRATCH_SHARED_WIDE_ARENAS") {
            Ok(value) => {
                ensure!(value == "0" || value == "1", "invalid wide arena sharing policy");
                value == "1"
            }
            Err(std::env::VarError::NotPresent) => false,
            Err(error) => return Err(error.into()),
        };
        ensure!(widths.iter().all(|&rows| crate::config::wide_rows(rows))
            && widths.iter().collect::<std::collections::BTreeSet<_>>().len() == widths.len(),
            "wide profile bank widths must be valid and unique");
        let mut widths = widths.to_vec();
        if shared { widths.sort_unstable_by(|a, b| b.cmp(a)); }
        let mut bank = Vec::with_capacity(widths.len());
        for rows in widths {
            // config::wide16_elastic: the 16,384-row profile's arenas keep only their 8,192-row halves resident, so every
            // narrower profile must carve inside those halves (it would fault on an unmapped chunk otherwise).
            let prefixes = self.cuda.wide16_resident_prefixes();
            for (arena, _) in &prefixes { arena.reset_profile_high(); }
            let target = self.fork_wide_with_arenas(core, core_bytes, stream, kernels,
                rows, sequence_parallel, if shared {bank.first()} else {None})?;
            if rows == 16384 && self.cuda.wide16_configured() {
                // Its construction ran with the extension mapped from a temporary pool (decoder::WideBuild::new).
                self.cuda.wide16_end_temporary()?;
                eprintln!("wide16 extension {}", serde_json::json!({"rank":self.rank,"bytes":self.cuda.wide16_bytes(),
                    "arenas":self.cuda.wide16_resident_prefixes().iter().map(|(a,r)|serde_json::json!({"bytes":a.bytes(),
                        "resident":r,"carved":a.profile_high()})).collect::<Vec<_>>()}));
            } else {
                for (arena, resident) in &prefixes {
                    ensure!(arena.profile_high() <= *resident,
                        "16K extension: the {rows}-row profile carves {} bytes of an arena whose resident half is {resident}",
                        arena.profile_high());
                }
            }
            bank.push(target);
        }
        Ok(bank)
    }
    /// Optional sharing is restricted to sequential profiles of this same
    /// model/state owner. It never aliases persistent state. The separately
    /// gated wide-MoE experiment also lends route scratch with a layout-change
    /// clear before each encoder capture/replay.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fork_wide_with_arenas(
        &self, core: &Module, core_bytes: &[u8], stream: &Stream, kernels: &Path,
        rows: usize, sequence_parallel: bool, share: Option<&Self>,
    ) -> Result<Self> {
        // The profile's rows decide which family its width-named bundles belong to (config::building_wide_profile).
        // config::shared_padded_inputs: a sequential profile (never a peer bank) is one padded-input share scope.
        crate::config::building_wide_profile(rows,||self.cuda.with_padded_share_scope(crate::config::shared_padded_inputs(),
            ||self.fork_wide_with_arenas_in(core,core_bytes,stream,kernels,rows,sequence_parallel,share,false)))
    }
    /// Two private causal owners still prefill sequentially. Lend only the
    /// existing transient parity/candidate arenas, with one ordered stream;
    /// the separately gated same-width MoE loan keeps that sequential contract.
    /// This finite experiment keeps both complete width banks available.
    pub(crate) fn fork_peer_wide_bank_for_probe(
        &self,core:&Module,core_bytes:&[u8],stream:&Stream,kernels:&Path,owner:&Self,peer:&[Self],
    )->Result<Vec<Self>> {
        ensure!(crate::config::shared_owner_profiles_requested()
            &&self.context==crate::config::MODEL_CONTEXT,"finite peer wide-bank admission");
        let largest=peer.iter().max_by_key(|t|t.rows).context("peer wide bank")?;
        // Complete owners expose every causal region; a wide profile exposes
        // only the encoder's execution layers and cannot be image-audited.
        ensure!(self.independent_state_report(owner)?["independent_causal_owners"]==true
            &&owner.shared_state_report(largest)["all_weights_and_causal_storage_shared"]==true,
            "peer arena donor must belong to the audited independent owner");
        let mut widths=peer.iter().map(|t|t.rows).collect::<Vec<_>>();widths.sort_unstable();
        ensure!(widths==[512,1024,2048,4096,8192],"preserve the complete serving width bank");
        widths.reverse();
        widths.into_iter().map(|rows| {
            let donor=if std::env::var("CANADA_SCRATCH_PEER_WIDE_MOE_SHARE_SCREEN").as_deref()==Ok("1") {
                peer.iter().find(|p|p.rows==rows).context("matching peer MoE geometry")?
            }else{largest};
            crate::config::building_wide_profile(rows,||self.fork_wide_with_arenas_in(core,core_bytes,stream,kernels,rows,true,Some(donor),true))
        }).collect()
    }
    #[allow(clippy::too_many_arguments)]
    fn fork_wide_with_arenas_in(
        &self, core: &Module, core_bytes: &[u8], stream: &Stream, kernels: &Path,
        rows: usize, sequence_parallel: bool, share: Option<&Self>, peer_owner:bool,
    ) -> Result<Self> {
        let share_candidates = match std::env::var("CANADA_SCRATCH_SHARED_CANDIDATE_SCRATCH") {
            Ok(value) => {
                ensure!(value == "0" || value == "1", "invalid candidate scratch sharing policy");
                value == "1"
            }
            Err(std::env::VarError::NotPresent) => false,
            Err(error) => return Err(error.into()),
        };
        ensure!(!share_candidates
            || std::env::var("CANADA_SCRATCH_SHARED_WIDE_ARENAS").as_deref() == Ok("1"),
            "candidate scratch sharing requires a sequential shared wide bank");
        ensure!(
            self.rows == 128
                && crate::config::wide_rows(rows)
                && self.wide.is_none()
                && self.source_phase == Phase::Extend
                && self.encoder_cache.is_some()
                && self.context >= rows
                && rows <= self.swa_layout.max_rows(),
            "wide prefill needs a decoder-tail owner and enough logical/physical SWA capacity"
        );
        if let Some(prior) = share {
            let shared=self.shared_state_report(prior);
            let ownership=if peer_owner {
                crate::config::shared_owner_profiles_requested()
                    &&self.context==prior.context&&!self.lease.shares_owner(&prior.lease)
                    &&shared["all40layer_atlases_shared"]==true&&shared["IO_atlas_shared"]==true
                    &&shared["Engram_row_stores_shared"]==true&&shared["encoder_tail_cache_shared"]==false
            }else{shared["all_weights_and_causal_storage_shared"]==true};
            ensure!(std::env::var("CANADA_SCRATCH_SHARED_WIDE_ARENAS").as_deref()==Ok("1")
                && prior.wide.is_some() && prior.rows >= rows
                && self.rank==prior.rank && self.stream==prior.stream
                && self.history.position == 0 && prior.history.position == 0
                && prior.encoder_graph.is_none() && prior.graphs.iter().all(Option::is_none)
                && ownership,
                "shared wide arenas require admitted sequential construction");
        }else{
            ensure!(!peer_owner,"peer wide arena source missing");
        }
        if self.context > 65536 {
            // Conservative preflight for the largest new arenas, in addition
            // to the unchanged atlas/cgroup guards. Lazy vendor scratch and
            // the draft network still need reserve after profile construction.
            let candidates=if crate::candidates::compact_construction_requested() {
                ensure!(sequence_parallel&&self.context==1048576
                    &&std::env::var("CANADA_SCRATCH_LOCAL_CANDIDATE_SCORES").as_deref()==Ok("1")
                    &&std::env::var("CANADA_SCRATCH_ROW_SHARDED_PUBLISHER").as_deref()!=Ok("0"),
                    "compact candidate preflight geometry");
                // Two compact score/mask allocations coexist briefly while
                // finalizing the retained layout, plus selectors and indices.
                // Full-profile v5 exposed additional non-candidate storage
                // beyond the compact buffers. Retain2GiB for that construction
                // until its actual immutable/transient duplication is removed.
                256*1024usize.pow(2)+2*1024usize.pow(3)
            }else{rows * self.context.div_ceil(8) * 5};
            let planned = if share.is_some() {0}else{rows * wide_arena_bytes_per_row(sequence_parallel) * 2}
                + candidates
                + self.context * 170 + 4 * 64 * 1024 * 1024;
            ensure!(self.cuda.admission_memory()?.admitted_available > planned + crate::memory::floor_bytes(),
                "full-context wide profile would consume reserved host/device headroom");
        }
        let wide = Rc::new(crate::decoder::WideBuild::new(
            &self.cuda,
            kernels,
            rows,
            rows * wide_arena_bytes_per_row(sequence_parallel),
            sequence_parallel.then_some(self.rank),
            share.and_then(|t|t.wide.as_deref()),
        )?);
        let mut target = self.fork_profile_in(
            core,
            core_bytes,
            stream,
            kernels,
            rows,
            Phase::Extend,
            Some(wide.clone()),
            None,
        )?;
        if share_candidates && let Some(prior) = share {
            // fork_profile_in synchronizes initialization before the private
            // buffers are replaced. Keep preflight conservative: construction
            // temporarily needs those buffers before this sharing releases them.
            target.late.source_mut().share_candidate_scratch_from(prior.late.source())?;
        }
        if sequence_parallel {
            let build=||->Result<_>{Ok((self.cuda.allocate_rows(rows*20480)?,self.cuda.allocate_rows(rows*4)?))};
            target.sp_full=Some(if let Some(arena)=wide.gather_arena() {
                self.cuda.with_row_arena(arena,build)?
            }else{build()?});
        }
        target.prefill_last_head = true;
        self.cuda.report_allocation_sites(&format!("rank{} wideM{rows} peer={peer_owner}",self.rank))?;
        Ok(target)
    }
    /// Engram RAM row-cache counters (shared by every profile of this target).
    pub(crate) fn engram_cache_stats(&self) -> serde_json::Value {
        serde_json::json!([self.row_store1.stats(), self.group14.row_store_stats()])
    }
    pub fn wide_report(&self) -> Option<serde_json::Value> {
        self.wide.as_ref().map(|w| {
            let mut report = w.report();
            report["index_row_sharded_sources"] = serde_json::json!(
                [(2, self.source2.query()), (8, self.group8.source().query()),
                    (14, self.group14.source().query()), (20, self.late.source().query())]
                .into_iter().filter_map(|(layer, branch)| branch.row_sharding_enabled().then_some(layer))
                .collect::<Vec<_>>());
            report["index_key_splits"]=serde_json::json!(self.index_key_splits());
            report["candidate_scratch"]=serde_json::json!(self.late.source().query().candidate_scratch_report());
            report
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn fork_profile_in(
        &self,
        core: &Module,
        core_bytes: &[u8],
        stream: &Stream,
        kernels: &Path,
        rows: usize,
        phase: Phase,
        wide: Option<Rc<crate::decoder::WideBuild>>,
        fresh_lineage: Option<u64>,
    ) -> Result<Self> {
        ensure!(
            self.lease.is_current()
                && crate::config::target_rows(rows)
                && crate::config::wide_rows(rows) == wide.is_some()
                && (crate::config::verification_rows(rows) || phase == Phase::Extend)
                && self.cuda.owns_stream(stream)
                && stream.raw() as usize == self.stream,
            "shared execution profile admission/stream"
        );
        ensure!(
            rows == 6 || self.source2.cache().page_tokens() == 64,
            "medium prefill requires64-token shared compressed pages"
        );
        let independent=fresh_lineage.is_some();
        ensure!(!independent || (rows==128 && phase==Phase::Extend && wide.is_none()
            && self.position()==0 && fresh_lineage.unwrap()>self.history.lineage),
            "independent profile geometry and lineage");
        let cuda = &self.cuda;
        crate::memory::trace_profile_memory(cuda, rows, "start");
        let early_caches=if independent {self.early_caches.iter()
            .map(|c|PagedKv::new_layout(cuda,c.pages(),c.page_tokens(),c.layout())).collect::<Result<Vec<_>>>()?}
            else {self.early_caches.clone()};
        // Full serving regions (config::full_regions): the 16-row verify owner runs every batched region step, so its
        // index branches hold every region's keys (region k's slots k x context / ratio + 32 onward, contiguous over the
        // regions); key matrices of the other regions are mapped while occupied (map_region_keys).
        let key_regions = Self::key_regions_for(rows, phase, self.context)?;
        let pair_slots = (COMPRESSED_OFFSET..COMPRESSED_OFFSET + key_regions * self.context / 2)
            .map(|s| s as i32)
            .collect::<Vec<_>>();
        let single_slots = (COMPRESSED_OFFSET..COMPRESSED_OFFSET + key_regions * self.context)
            .map(|s| s as i32)
            .collect::<Vec<_>>();
        let medium_woa = if rows > 8 {
            let build = || crate::woa_medium::Scratch::new_on(cuda, core_bytes, rows,stream);
            Some(match wide.as_deref() { Some(w) => w.scratch(cuda, build)?, None => build()? })
        } else {
            None
        };
        crate::memory::trace_profile_memory(cuda, rows, "after medium WO-A scratch");
        let share_narrow_during_build=match std::env::var("CANADA_SCRATCH_NARROW_MOE_CONSTRUCTION") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid construction-time MoE sharing policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
        } && wide.is_none() && SwaLayer::narrow_workspace_sharing_enabled(rows);
        let mut narrow_moe=None;
        let mut early = Vec::new();
        let mut source2 = None;
        for (id, old) in self.early.iter().enumerate() {
            let weights = old.shared_weights();
            let build = |moe: Option<Rc<crate::moe::Workspace>>, sp: Option<usize>| {
                SwaLayer::new_rows_in(
                    cuda,
                    core,
                    stream,
                    weights.clone(),
                    old.prepared_layout(),
                    kernels,
                    if id < 2 {
                        self.ordinary.clone()
                    } else {
                        self.compressed.clone()
                    },
                    if wide.is_some() {
                        InputMode::Fused
                    } else {
                        InputMode::Split(self.rank)
                    },
                    rows,
                    Some(self.vendor_blas.clone()),
                    medium_woa.clone(),
                    moe,
                    sp,
                )
            };
            early.push(match &wide {
                Some(w) => w.layer(cuda, id, |moe, sp| build(Some(moe), sp))?,
                None => build(narrow_moe.clone(), None)?,
            });
            if id==0 && share_narrow_during_build {
                narrow_moe=Some(early[0].narrow_workspace());
            }
            if id == 2 {
                crate::memory::trace_profile_memory(cuda, rows, "before source 2");
                let build_source = || Source::new_rows_owner(
                    cuda,
                    stream,
                    weights,
                    self.vendor_blas.clone(),
                    2,
                    kernels,
                    self.compressed.clone(),
                    core_bytes,
                    self.source2.cache().pages(),
                    self.source2.keys().capacity() / 64,
                    &pair_slots,
                    &vec![0; rows],
                    rows,
                    phase,
                    (!independent).then_some(&self.source2),
                );
                source2 = Some(match &wide { Some(w) => w.source(cuda, build_source)?, None => build_source()? });
            }
        }
        crate::memory::trace_profile_memory(cuda, rows, "after early layers and source 2");
        let source2 = source2.ok_or_else(|| anyhow::anyhow!("shared source2 missing"))?;
        let group8 = self.group8.fork_profile(
            cuda,
            core,
            core_bytes,
            stream,
            kernels,
            self.rank,
            self.compressed.clone(),
            &pair_slots,
            phase,
            rows,
            self.vendor_blas.clone(),
            medium_woa.clone(),
            wide.as_deref(),
            narrow_moe.clone(),
            independent,
        )?;
        let group14 = self.group14.fork_profile(
            cuda,
            core,
            core_bytes,
            stream,
            kernels,
            self.rank,
            self.compressed.clone(),
            &pair_slots,
            phase,
            rows,
            self.vendor_blas.clone(),
            medium_woa.clone(),
            wide.as_deref(),
            narrow_moe.clone(),
            independent,
        )?;
        let late = self.late.fork_profile(
            cuda,
            core,
            core_bytes,
            stream,
            kernels,
            self.rank,
            self.compressed.clone(),
            &single_slots,
            phase,
            rows,
            self.vendor_blas.clone(),
            medium_woa,
            wide.as_deref(),
            narrow_moe,
            independent,
        )?;
        crate::memory::trace_profile_memory(cuda, rows, "after groups 8, 14 and late");
        let rows1 = match wide.as_deref() {
            Some(w) => w.scratch(cuda, || Rows::new(cuda, rows, &self.row_store1))?,
            None => Rows::new(cuda, rows, &self.row_store1)?,
        };
        let io_build=||if independent {self.io.fork_owner(cuda,core_bytes,stream,kernels,rows)}
            else {self.io.fork_profile(cuda,core_bytes,stream,kernels,rows)};
        let mut io=if let Some(arena)=wide.as_ref().and_then(|w|w.input_arena()) {
            cuda.with_row_arena(arena,io_build)?
        }else{cuda.without_row_arena(io_build)?};
        let append_encoded_head=phase==Phase::Verify&&matches!(rows,16|32)
            &&crate::config::append_encoded_head_requested()?;
        if append_encoded_head {
            ensure!(crate::config::append_last_gather_requested()?,"encoded append head requires final-only output");
            io.prepare_append_encoded(stream)?;
        }
        crate::memory::trace_profile_memory(cuda, rows, "after io");
        // The step's pair-pool outputs are chunk scratch; its other buffers are unaffected.
        let build_step = || Step::new(
            cuda,
            &source2,
            group8.source(),
            group14.source(),
            late.source(),
            &pair_slots,
            &single_slots,
            rows,
            self.context,
            self.early_caches[0].capacity(),
            self.swa_layout,
            phase,
        );
        let step = match wide.as_deref() { Some(w) => w.scratch(cuda, build_step)?, None => build_step()? };
        let lease=if independent {crate::session_lease::Lease::new()} else {
            let mut lease=self.lease.fork();lease.adopt(&self.lease)?;lease
        };
        stream.synchronize()?;
        let mut target = Self {
            rows,
            source_phase: phase,
            tiny_native_reduce: false,
            prefill_dense_live_rows: self.prefill_dense_live_rows,
            prefill_moe_live_rows: self.prefill_moe_live_rows,
            prefill_upper_rows: self.prefill_upper_rows,
            prefill_woa_live: self.prefill_woa_live,
            prefill_partial_dense_rows: self.prefill_partial_dense_rows,
            prefill_partial_moe_rows: self.prefill_partial_moe_rows,
            prefill_partial1024_rows: self.prefill_partial1024_rows,
            prefill_partial2048_dense: self.prefill_partial2048_dense,
            prefill_partial_wide: self.prefill_partial_wide,
            hint_queue_selected: Rc::new(std::cell::Cell::new(None)),
            moe_front_grid: None,
            moe_phase_grid: None,
            moe_regridded: 0,
            moe_swapped: 0,
            defer_completion: false,
            completion_owed: std::cell::Cell::new(false),
            moe_live_floors: None,
            engram_lead: None,
            engram_lead_on: false,
            prefill_wide_mhc_rows: self.prefill_wide_mhc_rows,
            prefill_wide_mhc_norm: self.prefill_wide_mhc_norm,
            prefill_short_merge: self.prefill_short_merge,
            prefill_mhc_live_rows: self.prefill_mhc_live_rows,
            prefill_mhc_live_norm: self.prefill_mhc_live_norm,
            prefill_blocking_wait: self.prefill_blocking_wait,
            verification_poll_us: self.verification_poll_us,
            verification_mhc_live_rows: self.verification_mhc_live_rows,
            tiny_prefix_sum: phase==Phase::Verify && [6,16].contains(&rows)
                && std::env::var("CANADA_SCRATCH_LIVE_ROW_TRANSPORT").as_deref()==Ok("1"),
            tiny_prefix_gather: phase==Phase::Verify && [6,16].contains(&rows)
                && std::env::var("CANADA_SCRATCH_LIVE_ROW_TRANSPORT").as_deref()==Ok("1"),
            graph_retention: crate::graph_retention::Policy::new(rows,crate::config::execution_graph_cache_rows()?),
            graph_evictions: 0,
            graphs: (0..graph_slots(rows)).map(|_| None).collect(),
            engram_graphs: (0..graph_slots(rows)).map(|_| None).collect(),
            alternate_index_graphs: (0..graph_slots(rows)).map(|_| None).collect(),
            alternate_index_engram_graphs: (0..graph_slots(rows)).map(|_| None).collect(),
            adaptive_sparse_active: false,
            short_index_flags: Vec::new(),
            short_index_allowed: true,
            short_index_steps: 0,
            engram_overlap: false,
            engram_projection_ahead: false,
            engram_projection_m3_only: false,
            engram_projection_active: false,
            engram_split14: false,
            engram_split14_m3_only: false,
            engram_split14_adaptive: false,
            paired_engram_io: self.paired_engram_io,
            paired_engram32: self.paired_engram32,
            paired_engram32_adaptive: self.paired_engram32_adaptive,
            engram32_worker: None,
            engram32_worker_enabled: false,
            graph_captures: 0,
            prefill_capture_once: self.prefill_capture_once,
            verification_capture_once: self.verification_capture_once,
            carry_snapshot: None,
            carry_saved_start: None,
            repeatable_frontier_measurement: false,
            encoder_graph: None,
            encoder_graph_live: None,
            decoder_graph: None,
            tiny_append_graphs: None,
            encoder_cache: if independent {None}else{self.encoder_cache.clone()},
            region_tails: if independent {Vec::new()}else{self.region_tails.clone()},
            checkpoint_aux: None,
            checkpoint_plan_for_probe: None,
            checkpoint_inline_for_probe: None,
            checkpoint_aux_extra: None,
            extra_checkpoint: None,
            extra_image: None,
            replay: None,
            graph_replay: false,
            prefill_last_head: false,
            layer20_projection_only: false,
            layer20_pending: false,
            append_last_gather: phase==Phase::Verify && [6,16,32].contains(&rows)
                && crate::config::append_last_gather_requested()?,
            append_encoded_head,
            comm_instance: self.comm_instance,
            cuda: cuda.clone(),
            io,
            sampler: Greedy::new(cuda, core_bytes, rows)?,
            categorical: crate::categorical::Categorical::new(cuda, core_bytes, rows)?,
            compact_categorical: None,
            compact_sampling_enabled: false,
            local_target_enabled: rows==6&&phase==Phase::Verify
                &&crate::sampler_transport::Policy::from_env()?.local_target,
            local_target_gather: (0..=rows).map(|_|None).collect(),
            last_head_local_only: false,
            last_live: 0,
            last_append_metrics: None,
            pending: None,
            batched_rows: None,
            batched_done: None,
            batched_pending: false,
            region_step: false,
            batched_owner_part: false,
            batched_inflight: None,
            batched_staged: None,
            batched_block: 0,
            padding_routes: padding_routes_requested(),
            region_gather: RegionGather::default(),
            early,
            early_caches,
            source2,
            group8,
            group14,
            late,
            step,
            cached_decoder: None,
            short_restore_extend: false,
            tokens: TokenRows::new(cuda, &vec![2; rows], 0)?,
            row_store1: self.row_store1.clone(),
            rows1,
            hashes: self.hashes.clone(),
            history: fresh_lineage.map_or_else(||self.history.clone(),History::new),
            lease,
            context: self.context,
            swa_layout: self.swa_layout,
            ordinary: self.ordinary.clone(),
            compressed: self.compressed.clone(),
            vendor_blas: self.vendor_blas.clone(),
            wide,
            sp_full: None,
            rank: self.rank,
            stream: self.stream,
            index_score_arena: if independent {None}else{self.index_score_arena.clone()},
            index_score_logical_bytes: 0,
            atlas_bytes: self.atlas_bytes.clone(),
            source_hashes: self.source_hashes.clone(),
        };
        target.share_index_scores_before_capture(cuda,stream)?;
        target.share_narrow_moe_before_capture(cuda,stream)?;
        target.enable_verify16_ffn_before_capture(stream)?;
        target.enable_engram_projection_before_capture(kernels)?;
        target.install_communication_hints(core, stream)?;
        target.install_hint_queues(core, stream)?;
        target.install_source_keys_fork(stream)?;
        target.install_index_fork(stream)?;
        target.install_narrow_qkv(kernels)?;
        cuda.report_allocation_sites(&format!("rank{} forkM{rows} independent={independent}",self.rank))?;
        Ok(target)
    }
    /// Opt-in accumulated candidate. The M3 graph owns separate mutable WKV
    /// scratch; its ordinary M6 peer and all bulk-prefill profiles stay intact.
    /// s1274 checked mixed-width captures and asymmetric immutable-row misses.
    fn enable_engram_projection_before_capture(&mut self,kernels:&Path)->Result<()> {
        let enabled=match std::env::var("CANADA_SCRATCH_ENGRAM_PROJECTION_M3").as_deref() {
            Err(std::env::VarError::NotPresent)|Ok("0")=>false,Ok("1")=>true,
            _=>anyhow::bail!("Engram M3 projection policy must be0or1"),
        };
        if !enabled||self.rows!=6||self.source_phase!=Phase::Verify{return Ok(());}
        // config::engram_split14_decode reads the layer-14 rows behind layers 1-13; a projection hoisted to layer 1
        // would need them first. The split wins; the projection then runs at layer 14 as in every other step.
        if crate::config::engram_split14_decode(){return Ok(());}
        ensure!(self.graph_captures==0&&!self.engram_split14&&!self.engram_split14_adaptive,
            "Engram M3 policy must precede graph capture");
        // Only row-sized values/scales/output and stream/module bookkeeping are
        // new. Canonical weights remain shared. Keep margin above the normal
        // allocation floor; this does not borrow an optional conversation slot.
        ensure!(self.cuda.admission_memory()?.admitted_available>=
            self.cuda.allocation_reserve_bytes()+8*1024*1024,
            "Engram projection workspace would consume the physical reserve");
        self.group14.set_engram_projection_ahead_for_probe(true,kernels)?;
        self.engram_projection_ahead=true;self.engram_projection_m3_only=true;
        Ok(())
    }
    fn enable_verify16_ffn_before_capture(&mut self,stream:&Stream)->Result<()> {
        // Wide verification can overlap the independent shared-expert branch.
        // Configure it before capture; the owner also serves tiny retained
        // appends. Bulk-prefill profiles retain their own schedule.
        if ![16,32].contains(&self.rows) || self.source_phase!=Phase::Verify
            || !crate::ffn::wide_overlap_enabled(self.rows)? {return Ok(());}
        ensure!(self.graph_captures==0&&self.pending.is_none()&&self.stream==stream.raw()as usize,
            "M16 FFN overlap must precede capture/execution");
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.enable_verify16_ffn_before_capture(stream)?;
        }
        ensure!(self.layer_owners().iter().all(|l|l.ffn_overlap_flags()==(true,true)),
            "all forty Verify16 FFN schedules must agree");
        Ok(())
    }
    fn share_index_scores_before_capture(&mut self,cuda:&Cuda,stream:&Stream)->Result<()> {
        if (crate::config::sparse_union_share() || crate::config::sparse_union_share_screen()) && self.rows==6
            && self.source_phase==Phase::Verify && self.adaptive_sparse_index_layers()==4 && self.late.sparse_union_linked()==0 {
            ensure!(self.graph_captures==0 && self.pending.is_none() && self.position()==0
                && self.stream==stream.raw()as usize && cuda.owns_stream(stream),
                "candidate union sharing must precede target execution/capture");
            ensure!(self.late.share_sparse_union(crate::config::sparse_union_share())?==3,"candidate union followers");
            eprintln!("rank{} M6 consuming layers 28, 32, 36 linked to layer 24's candidate union and gathered keys (reuse {})",
                self.rank,crate::config::sparse_union_share());
        }
        let requested=match std::env::var("CANADA_SCRATCH_SHARED_INDEX_SCORES") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid shared index scores policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
        };
        if !requested {ensure!(self.index_score_arena.is_none(),"index score policy changed between profiles");return Ok(());}
        ensure!(self.graph_captures==0 && self.pending.is_none() && self.position()==0
            && self.stream==stream.raw()as usize && cuda.owns_stream(stream),
            "index score sharing must precede target execution/capture");
        stream.synchronize()?;
        let storage=match &self.index_score_arena {
            Some(storage)=>storage.clone(),
            None=>Rc::new(cuda.allocate::<f32>(64*1024*1024/4)?),
        };
        let mut released=0;
        for source in [&mut self.source2,self.group8.source_mut(),
            self.group14.source_mut(),self.late.source_mut()] {
            released+=source.share_score_scratch(storage.clone())?;
        }
        released+=self.late.share_consumer_score_scratch(storage.clone())?;
        self.index_score_logical_bytes=released;
        self.index_score_arena=Some(storage);
        eprintln!("rank{} M{} shared index score scratch: logical={} backing={} memory={:?}",
            self.rank,self.rows,released,64*1024*1024,cuda.admission_memory()?);
        Ok(())
    }
    fn share_narrow_moe_before_capture(&mut self,cuda:&Cuda,stream:&Stream)->Result<()> {
        let enabled=SwaLayer::narrow_workspace_sharing_enabled(self.rows);
        if !enabled {return Ok(());}
        ensure!(self.graph_captures==0 && self.pending.is_none() && self.position()==0,
            "scratch sharing must precede execution/capture");
        let before=cuda.admission_memory()?;
        let workspace=self.early[0].narrow_workspace();
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.share_narrow_workspace(cuda,workspace.clone(),self.rows)?;
        }
        stream.synchronize()?;
        ensure!(self.layer_owners().len()==40 && self.layer_owners().iter()
            .all(|layer|layer.routed().ptr()==workspace.output().ptr()),"all forty narrow layers share scratch");
        eprintln!("rank{} shared M{} MoE workspace: before={before:?}, after={:?}",self.rank,self.rows,cuda.admission_memory()?);
        Ok(())
    }
    fn install_communication_hints(&mut self, core: &Module, stream: &Stream) -> Result<()> {
        if !crate::l2_prefetch::enabled_for_rows(self.rows,4)? {
            return Ok(());
        }
        let inputs: Vec<_> = self
            .layer_owners()
            .iter()
            .map(|l| l.input_matrix())
            .collect();
        let cuda = self.cuda.clone();
        let layers = self
            .early
            .iter_mut()
            .chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut())
            .chain(self.late.layers_mut());
        for (id, layer) in layers.enumerate() {
            // Engram executes before the next QKV at1/14. Avoid hinting a later
            // consumer through that different weight working set.
            if let Some(next) = inputs.get(id + 1).filter(|_| ![1, 14].contains(&(id + 1))) {
                layer.install_next_input_prefetch(&cuda, core, stream, next.clone())?;
            }
        }
        stream.synchronize()?;
        Ok(())
    }
    /// config::source_keys_fork (or its plan screen): the verify owners' source layers get their key-emission side
    /// stream before any capture; the flag switches it on, the screen toggles it.
    fn install_source_keys_fork(&mut self, stream: &Stream) -> Result<()> {
        let on = crate::config::source_keys_fork();
        if !(on || crate::config::source_keys_fork_screen()) || ![6, 16].contains(&self.rows)
            || self.source_phase != Phase::Verify { return Ok(()); }
        // Not usable as built (screen s1480 fork1): the vendor projection's handle draws on the cuBLAS workspace
        // pooled per ordered stream, so it cannot run on a side stream without a private handle and workspace.
        // config::l2_queue_index_lead reaches the same weights with hints instead. Refuse rather than fail mid-run.
        anyhow::bail!("CANADA_SCRATCH_SOURCE_KEYS_FORK needs a private vendor workspace for the forked projection (not built); use CANADA_SCRATCH_L2_QUEUE_INDEX_LEAD");
        #[allow(unreachable_code)]
        ensure!(self.graph_captures == 0 && self.pending.is_none(), "the source key fork precedes execution/capture");
        let cuda = self.cuda.clone();
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.enable_source_keys_fork(&cuda, stream, on)?;
        }
        Ok(())
    }
    /// config::qkv_narrow_prequant (or its plan screen): the six-row verify owner's layers load the narrow lowering
    /// of QKV-A beside the default one, before any capture; the flag makes it the active one.
    fn install_narrow_qkv(&mut self, kernels: &Path) -> Result<()> {
        let on = crate::config::qkv_narrow_prequant();
        if !(on || crate::config::qkv_narrow_prequant_screen()) || self.rows != 6
            || self.source_phase != Phase::Verify { return Ok(()); }
        ensure!(self.graph_captures == 0 && self.pending.is_none(), "the narrow QKV lowering precedes execution/capture");
        let mut installed = 0;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            if layer.install_narrow_qkv(kernels, on)? { installed += 1; }
        }
        eprintln!("rank{} M{} narrow QKV-A lowering: {installed} layers, active {on}", self.rank, self.rows);
        Ok(())
    }
    /// Plan screen (CANADA_SCRATCH_QKV_NARROW_PREQUANT_SCREEN=1): QKV-A on the narrow lowering (on) or the default
    /// one (off); graphs recaptured. Returns the number of layers switched.
    pub(crate) fn set_narrow_qkv_for_probe(&mut self, on: bool, stream: &Stream) -> Result<usize> {
        ensure!(crate::config::l2_queue_screen() && self.rows == 6 && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle narrow QKV screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        let mut switched = 0;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            if layer.set_narrow_qkv_for_probe(on)? { switched += 1; }
        }
        Ok(switched)
    }
    /// config::index_fork (or its plan screen): the verify owners' consuming layers get the side stream of their
    /// index work before any capture, and source 20's key cache admits that stream as a reader.
    fn install_index_fork(&mut self, stream: &Stream) -> Result<()> {
        let on = crate::config::index_fork();
        if !(on || crate::config::index_fork_screen()) || ![6, 16].contains(&self.rows)
            || self.source_phase != Phase::Verify { return Ok(()); }
        ensure!(self.graph_captures == 0 && self.pending.is_none(), "the index fork precedes execution/capture");
        let cuda = self.cuda.clone();
        let mut installed = 0;
        for layer in self.late.layers_mut() {
            if layer.enable_index_fork(&cuda, stream, on)? { installed += 1; }
        }
        ensure!(installed == 0 || installed == 4, "index fork on some consuming layers only");
        if installed == 4 {
            let side = cuda.hint_stream(stream)?;
            self.late.source().keys().admit_follower(side.as_ref())?;
        }
        eprintln!("rank{} M{} index fork: {installed} consuming layers, start at the query fork {on}", self.rank, self.rows);
        Ok(())
    }
    /// Plan screen (CANADA_SCRATCH_INDEX_FORK_SCREEN=1): the consuming layers' index work started where Q-B forks
    /// (on) or behind the key store (off); graphs recaptured. Returns the number of layers switched.
    pub(crate) fn set_index_fork_for_probe(&mut self, on: bool, stream: &Stream) -> Result<usize> {
        ensure!(crate::config::l2_queue_screen() && [6, 16].contains(&self.rows) && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle index fork screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        let mut switched = 0;
        for layer in self.late.layers_mut() {
            if layer.set_index_fork_for_probe(on) { switched += 1; }
        }
        Ok(switched)
    }
    /// Plan screen (CANADA_SCRATCH_SOURCE_KEYS_FORK_SCREEN=1): the source layers' key-emission fork on or off;
    /// graphs recaptured. Returns the number of layers switched.
    pub(crate) fn set_source_keys_fork_for_probe(&mut self, on: bool, stream: &Stream) -> Result<usize> {
        ensure!(crate::config::l2_queue_screen() && [6, 16].contains(&self.rows) && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle source key fork screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        let mut switched = 0;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            if layer.set_source_keys_fork_for_probe(on) { switched += 1; }
        }
        // The projections change streams with the switch; everything they ran on is synchronized above.
        for source in [&self.source2, self.group8.source(), self.group14.source(), self.late.source()] {
            source.rebind_projection_stream_for_probe();
        }
        Ok(switched)
    }
    /// CANADA_SCRATCH_L2_QUEUE: the six-row decode owner's layers hint QKV-A, wq_b, wo_a and wo_b as one queue
    /// over three windows (layer_attention::HintQueue). One plan is selected at once; serving selects its
    /// only plan, the plan screen starts on the existing hints. Before any capture.
    fn install_hint_queues(&mut self, core: &Module, stream: &Stream) -> Result<()> {
        crate::moe::install_native_front(core)?;
        let (plans, ffn_plans) = crate::config::l2_queue_plans_for(self.rows);
        if plans.is_empty() || !crate::l2_prefetch::enabled_for_rows(self.rows, 2)? {
            return Ok(());
        }
        crate::config::validate_l2_queue()?;
        ensure!(self.graph_captures == 0 && self.pending.is_none(), "hint queues precede execution/capture");
        let selected = self.hint_queue_selected.clone();
        // config::l2_queue_index_lead (or its screen): what each indexed layer's index work reads cold before the
        // attention needs wq_b: the compressor's projection weights in the source layers, then the index query's.
        let mut leads: Vec<(Vec<(u64, usize)>, Vec<Rc<crate::blockscaled::Matrix>>)> = (0..40).map(|_| (Vec::new(), Vec::new())).collect();
        if crate::config::l2_queue_index_lead() || crate::config::l2_queue_index_lead_screen() {
            let weights: Vec<Rc<Weights>> = self.early.iter().chain(self.group8.layers()).chain(self.group14.layers())
                .chain(self.late.layers()).map(|layer| layer.shared_weights()).collect();
            ensure!(weights.len() == 40, "index lead layer weights");
            let branches = self.index_queries();
            for (slot, id) in [2usize, 8, 14, 20, 24, 28, 32, 36].into_iter().enumerate() {
                let Some(branch) = branches[slot] else { continue };
                let matrix = branch.query_matrix();
                let mut regions = Vec::new();
                let key = format!("layers.{id}.attn.compressor.wkv.weight");
                if id <= 20 && let Ok(view) = weights[id].view(&key) {
                    let mut bytes = view.bytes;
                    if let Ok(gate) = weights[id].view(&format!("layers.{id}.attn.compressor.wgate.weight"))
                        && view.offset + view.bytes == gate.offset { bytes += gate.bytes; }
                    let address = weights[id].ptr(&key)?;
                    if address % 16 == 0 && bytes >= 16 { regions.push((address, bytes / 16 * 16)); }
                }
                regions.extend(matrix.immutable_regions()?);
                leads[id] = (regions, vec![matrix]);
            }
        }
        if self.rank == 0 && leads.iter().any(|l| !l.0.is_empty()) {
            eprintln!("rank0 M{} index lead bytes by indexed layer: {:?}", self.rows,
                [2usize, 8, 14, 20, 24, 28, 32, 36].map(|id| leads[id].0.iter().map(|r| r.1).sum::<usize>()));
        }
        // config::l2_queue_engram_lead (or its screen): the Engram projections of layers 1 and 14 as the first
        // window of those layers, begun by layers 0 and 13 while their gate is set.
        let mut engram_matrices: [Option<Rc<crate::blockscaled::Matrix>>; 2] = [None, None];
        if crate::config::l2_queue_engram_lead() || crate::config::l2_queue_engram_lead_screen() {
            engram_matrices[0] = self.early.get(1).and_then(|layer| layer.engram()).map(|e| e.projection_matrix());
            engram_matrices[1] = self.group14.engram().map(|(e, _)| e.projection_matrix());
            if engram_matrices.iter().all(Option::is_some) {
                self.engram_lead_on = crate::config::l2_queue_engram_lead();
                self.engram_lead = Some((Rc::new(std::cell::Cell::new(self.engram_lead_on)),
                    Rc::new(std::cell::Cell::new(self.engram_lead_on))));
            } else { engram_matrices = [None, None]; }
        }
        let mut queues = Vec::with_capacity(40);
        for (id, layer) in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()).enumerate() {
            let engram = match (id, &self.engram_lead) {
                (1, Some((gate, _))) => engram_matrices[0].clone().map(|m| (m, gate.clone())),
                (14, Some((_, gate))) => engram_matrices[1].clone().map(|m| (m, gate.clone())),
                _ => None,
            };
            // Engram runs before the QKV of layers 1 and 14: no W1 through that working set (as the QKV-A hint).
            queues.push(layer.build_hint_queues(core, stream, plans, ffn_plans, selected.clone(),
                id > 0 && ![1, 14].contains(&id), &leads[id].0, &leads[id].1, engram)?);
        }
        ensure!(queues.len() == 40, "hint queues cover the forty layers");
        let engram_lead = self.engram_lead.is_some();
        for (id, layer) in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()).enumerate() {
            if id + 1 < 40 && (engram_lead || ![1, 14].contains(&(id + 1))) { layer.install_next_hint_queues(queues[id + 1].clone())?; }
        }
        stream.synchronize()?;
        if plans.len() == 1 && !crate::config::l2_queue_screen() { selected.set(Some(0)); }
        eprintln!("rank{} M{} L2 hint queues: {} plan(s), selected {:?}", self.rank, self.rows, plans.len(), selected.get());
        Ok(())
    }
    /// Plan screen: select a hint queue plan (None: the existing separate hints). Graphs are recaptured.
    pub(crate) fn set_l2_queue_for_probe(&mut self, plan: Option<usize>, stream: &Stream) -> Result<()> {
        ensure!(crate::config::l2_queue_screen()
            && plan.is_none_or(|p| p < crate::config::l2_queue_plans_for(self.rows).0.len())
            && self.source_phase == Phase::Verify && self.pending.is_none()
            && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle L2 queue screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        self.hint_queue_selected.set(plan);
        Ok(())
    }
    /// Plan screen: the native WO-A partial kernel on or off in every layer of this owner. Graphs are recaptured.
    /// Plan screen only: the verification graphs captured from now on launch the routed experts' front-end
    /// kernel (48 cooperating blocks by default) with `grid` blocks; None restores the exported grid.
    pub(crate) fn set_moe_front_grid_for_probe(&mut self, grid: Option<u32>, stream: &Stream) -> Result<()> {
        ensure!(crate::config::l2_queue_screen() && grid.is_none_or(|g| (1..=48).contains(&g))
            && [6, 16].contains(&self.rows) && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle front-end grid screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        self.moe_front_grid = grid;
        Ok(())
    }
    /// Plan screen only: the verification graphs captured from now on launch the routed experts' phase kernels
    /// with (FC1, FC2) blocks and that carve-out preference; None leaves the graphs as exported. A control arm that
    /// follows a carve-out arm passes (96, 96, Some(-1)) so the function attribute is restored at its capture.
    pub(crate) fn set_moe_phase_grid_for_probe(&mut self, grid: Option<(u32, u32, Option<i32>)>, stream: &Stream) -> Result<()> {
        ensure!(crate::config::l2_queue_screen() && grid.is_none_or(|(up, down, _)| (1..=1024).contains(&up) && (1..=1024).contains(&down))
            && [6, 16].contains(&self.rows) && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle expert grid screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        self.moe_phase_grid = grid;
        Ok(())
    }
    /// Kernel nodes the grid rules matched in the last verification graph capture (screens).
    pub(crate) fn moe_regridded(&self) -> usize { self.moe_regridded }
    /// Plan screen (CANADA_SCRATCH_MOE_NATIVE_FRONT_SCREEN=1): the verification graphs captured from now on run
    /// the native expert front-end in place of the exported one, or not. Graphs are recaptured.
    pub(crate) fn set_moe_native_front_for_probe(&mut self, on: bool, stream: &Stream) -> Result<()> {
        ensure!(crate::config::l2_queue_screen() && [6, 16].contains(&self.rows) && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle native front-end screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        crate::moe::set_native_front_for_probe(on)
    }
    /// Plan screen (CANADA_SCRATCH_MOE_ROUTE_FRONT_SCREEN=1): the verification graphs captured from now on leave
    /// the router's top-k to the native expert front-end, or not. Graphs are recaptured.
    pub(crate) fn set_moe_route_front_for_probe(&mut self, on: bool, stream: &Stream) -> Result<()> {
        ensure!(crate::config::l2_queue_screen() && [6, 16].contains(&self.rows) && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle fused routing screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        crate::moe::set_route_front_for_probe(on)
    }
    /// Exported front-end nodes the last verification graph capture replaced (screens).
    pub(crate) fn moe_swapped(&self) -> usize { self.moe_swapped }
    /// Plan screen (CANADA_SCRATCH_SPARSE_UNION_SHARE_SCREEN=1): layers 28, 32 and 36 reuse layer 24's candidate
    /// union and gathered keys, or recompute them. Graphs are recaptured. Ok(false) on an owner without the link.
    pub(crate) fn set_sparse_union_share_for_probe(&mut self, on: bool, stream: &Stream) -> Result<bool> {
        ensure!(crate::config::l2_queue_screen() && [6, 16].contains(&self.rows) && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle candidate union screen");
        if self.late.sparse_union_linked() != 3 { return Ok(false); }
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        self.late.set_sparse_union_follow_for_probe(on)?;
        Ok(true)
    }
    /// Plan screen (CANADA_SCRATCH_BOUNDED_MASK_SCREEN=1): the candidate consumers of layers 24..36 with the bounded
    /// mask kernel on or off; graphs recaptured. Returns how many consumers were switched.
    pub(crate) fn set_bounded_mask_for_probe(&mut self, on: bool, stream: &Stream) -> Result<usize> {
        ensure!(crate::config::l2_queue_screen() && [6, 16].contains(&self.rows) && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle bounded mask screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        let mut switched = 0;
        for branch in self.index_queries().into_iter().flatten() {
            if branch.set_bounded_mask_for_probe(on)? { switched += 1; }
        }
        Ok(switched)
    }
    /// Plan screen (CANADA_SCRATCH_L2_QUEUE_ENGRAM_LEAD_SCREEN): the Engram projection hints on or off; graphs recaptured.
    pub(crate) fn set_l2_engram_lead_for_probe(&mut self, on: bool, stream: &Stream) -> Result<()> {
        ensure!(crate::config::l2_queue_screen() && [6, 16].contains(&self.rows) && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none() && self.engram_lead.is_some(),
            "idle Engram lead hint screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        self.engram_lead_on = on;
        if let Some((first, second)) = &self.engram_lead { first.set(on); second.set(on); }
        Ok(())
    }
    /// Plan screen (CANADA_SCRATCH_L2_QUEUE_INDEX_LEAD_SCREEN): the indexed layers' second window with the index
    /// work's cold weights first; graphs recaptured.
    pub(crate) fn set_l2_index_lead_for_probe(&mut self, on: bool, stream: &Stream) -> Result<()> {
        ensure!(crate::config::l2_queue_screen() && [6, 16].contains(&self.rows) && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle index lead hint screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut()).chain(self.group14.layers_mut())
            .chain(self.late.layers_mut()) {
            layer.set_index_lead_for_probe(on);
        }
        Ok(())
    }
    /// Plan screen (CANADA_SCRATCH_L2_QUEUE_INDEX_EARLY_SCREEN): the third window before the index work; graphs recaptured.
    pub(crate) fn set_l2_index_early_for_probe(&mut self, on: bool, stream: &Stream) -> Result<()> {
        ensure!(crate::config::l2_queue_screen() && [6, 16].contains(&self.rows) && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle early index hint screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut()).chain(self.group14.layers_mut())
            .chain(self.late.layers_mut()) {
            layer.attention().set_index_early_for_probe(on);
        }
        Ok(())
    }
    /// Plan screen (CANADA_SCRATCH_MHC_STATS_LATE_SCREEN): move every layer's statistics forks; graphs are recaptured.
    pub(crate) fn set_mhc_stats_late_for_probe(&mut self, points: Option<(u8, u8)>, stream: &Stream) -> Result<()> {
        ensure!(crate::config::l2_queue_screen() && [6, 16].contains(&self.rows) && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle mHC late statistics screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        let (attention, ffn) = points.map_or(crate::mhc::stats_late_requested(self.rows)?, |p| p);
        for layer in self.early.iter_mut().chain(self.group8.layers_mut()).chain(self.group14.layers_mut())
            .chain(self.late.layers_mut()) {
            layer.set_mhc_stats_late_for_probe(attention, ffn)?;
        }
        Ok(())
    }
    /// Plan screen (CANADA_SCRATCH_KNOCKOUT_SCREEN): the kernel groups whose launches the next captures leave out
    /// (crate::knockout; 0 = none). Graphs are recaptured.
    pub(crate) fn set_knockout_for_probe(&mut self, mask: u32, stream: &Stream) -> Result<()> {
        ensure!(crate::config::l2_queue_screen() && self.rows == 6 && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle knockout screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        crate::knockout::set(mask);
        Ok(())
    }
    /// Plan screen (CANADA_SCRATCH_WQB_DIRECT_SCREEN): every layer's wq_b on the direct-read GEMV or back on the
    /// exported kernel; graphs are recaptured.
    pub(crate) fn set_wqb_direct_for_probe(&mut self, on: Option<bool>, stream: &Stream) -> Result<()> {
        ensure!((crate::config::wqb_direct_screen() || crate::config::qkva_direct_screen()) && self.rows == 6 && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle direct wq_b screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut()).chain(self.group14.layers_mut())
            .chain(self.late.layers_mut()) {
            layer.attention().set_wqb_direct_for_probe(on)?;
        }
        Ok(())
    }
    /// Plan screen (CANADA_SCRATCH_INJECT_SCREEN; crate::inject): the injection of the graphs captured from now on.
    pub(crate) fn set_injection_for_probe(&mut self, arm: Option<crate::inject::Arm>, stream: &Stream) -> Result<()> {
        ensure!(crate::config::l2_queue_screen() && crate::inject::armed() && self.rows == 6 && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle injection screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        crate::inject::set(arm);
        Ok(())
    }
    /// crate::head4_rescore on this six-row verify owner (before any graph is captured).
    pub(crate) fn install_head4_rescore(&mut self, head: Rc<crate::nvfp4_head::Head>, cap: usize, window: f32, on: bool) -> Result<()> {
        ensure!(self.rows == 6 && self.source_phase == Phase::Verify && self.pending.is_none() && self.graph_captures == 0,
            "4-bit head rescoring installs on an idle six-row verify owner before its captures");
        self.io.install_head4(head, cap, window, on)
    }
    /// Plan screen (CANADA_SCRATCH_HEAD4_RESCORE_SCREEN): the head by 4-bit copy + rescoring, or the exact head;
    /// graphs are recaptured.
    pub(crate) fn set_head4_for_probe(&mut self, on: bool, stream: &Stream) -> Result<()> {
        ensure!(crate::config::head4_rescore_screen() && self.rows == 6 && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle 4-bit head screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        self.io.set_head4(on)
    }
    /// Plan screen: the rows this rank rescored for each live row of the last completed pass, and the settings.
    pub(crate) fn head4_counts_for_probe(&self, live: usize, stream: &Stream) -> Result<(Vec<i32>, Option<(usize, f32)>)> {
        stream.synchronize()?;
        Ok((self.io.head4_counts(live)?, self.io.head4_settings()))
    }
    /// Plan screen (CANADA_SCRATCH_MHC_QUARTER_STATS_SCREEN): every installed attention boundary's statistics by
    /// rank quarter or back on the whole-residual kernel; graphs are recaptured. Returns the layers switched.
    pub(crate) fn set_quarter_stats_for_probe(&mut self, on: bool, stream: &Stream) -> Result<usize> {
        ensure!(crate::config::mhc_quarter_stats_screen() && self.rows == 6 && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle quarter statistics screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        let mut switched = 0;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut()).chain(self.group14.layers_mut())
            .chain(self.late.layers_mut()) {
            switched += usize::from(layer.attention().set_quarter_stats_for_probe(on)?);
        }
        Ok(switched)
    }
    /// Plan screen: the first layer's QKV-A and wq_b outputs after a completed step (its input is the embedding,
    /// the same in every arm), so two arms' bytes compare the projection kernels alone.
    pub(crate) fn direct_outputs_for_probe(&self, stream: &Stream) -> Result<[Vec<u8>; 2]> {
        stream.synchronize()?;
        self.early.first().context("no early layer")?.attention().direct_outputs_for_probe()
    }
    pub(crate) fn set_woa_native_for_probe(&mut self, on: bool, stream: &Stream) -> Result<()> {
        ensure!(crate::config::woa_native_screen() && [6, 16].contains(&self.rows) && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none(), "idle native WO-A screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut()).chain(self.group14.layers_mut())
            .chain(self.late.layers_mut()) {
            layer.attention().set_woa_native_for_probe(on)?;
        }
        Ok(())
    }
    /// Finite component screen over two completed, independently owned M6
    /// forwards. Attention, reductions and causal publication remain unchanged.
    pub(crate) fn paired_ffn_screen(&self,other:&Self,pooled:&crate::ffn::Paired,
        stream:&Stream)->Result<serde_json::Value> {
        use sha2::{Digest,Sha256};
        ensure!(std::env::var("CANADA_SCRATCH_PAIRED_FFN_SCREEN").as_deref()==Ok("1")
            && self.rows==6 && other.rows==6 && self.source_phase==Phase::Verify
            && other.source_phase==Phase::Verify && self.lease.is_busy_revision()
            && other.lease.is_busy_revision() && !self.lease.shares_owner(&other.lease)
            && self.pending.as_ref().is_some_and(|p|p.live==6)
            && other.pending.as_ref().is_some_and(|p|p.live==6)
            && self.cuda.owns_stream(stream) && stream.raw()as usize==self.stream
            && self.stream==other.stream,"paired FFN requires two independent pending M6 owners");
        stream.synchronize()?;
        let left=self.layer_owners();let right=other.layer_owners();
        let mut reports=Vec::new();
        for index in [0,28] {
            let a=left[index];let b=right[index];
            let read=||->Result<Vec<Vec<u8>>>{
                [a,b].iter().map(|layer|{
                    let out=layer.ffn().output();
                    unsafe{self.cuda.read_device(out.ptr(),out.bytes())}
                }).collect()
            };
            let expected=read()?;
            let run=|mode:usize|->Result<()>{
                let (a,at,b,bt)=if mode<2 {(a,&self.tokens,b,&other.tokens)}
                    else {(b,&other.tokens,a,&self.tokens)};
                if mode%2==0 {
                    for (layer,tokens)in [(a,at),(b,bt)] {
                        layer.ffn().run_quantized_input(&layer.narrow_workspace(),
                            layer.ffn_boundary().input(),layer.ffn_boundary().quantized_input(),tokens,stream)?;
                    }
                    Ok(())
                } else {
                    pooled.run(a.ffn(),a.ffn_boundary().input(),a.ffn_boundary().quantized_input(),at,
                        b.ffn(),b.ffn_boundary().input(),b.ffn_boundary().quantized_input(),bt,stream)
                }
            };
            let mut graphs=Vec::new();
            for mode in 0..4 {
                run(mode)?;stream.synchronize()?;
                ensure!(read()?==expected,"paired FFN layer{index} mode{mode} differs from original target forward");
                graphs.push(stream.capture(||run(mode))?);
            }
            let mut records=Vec::new();
            for repeat in 0..4 {
                let order=if repeat%2==0 {[0,1,2,3]}else{[3,2,1,0]};
                let mut milliseconds=[0f32;4];
                for mode in order {
                    for _ in 0..2 {unsafe{graphs[mode].launch(stream)?;}}
                    stream.synchronize()?;
                    let start=self.cuda.event()?;let end=self.cuda.event()?;start.record(stream)?;
                    for _ in 0..8 {unsafe{graphs[mode].launch(stream)?;}}
                    end.record(stream)?;milliseconds[mode]=end.elapsed_since(&start)?/8.;
                    ensure!(read()?==expected,"paired FFN layer{index} replay mode{mode} changed output");
                }
                records.push(serde_json::json!({"repeat":repeat,"order":order,"milliseconds":milliseconds}));
            }
            reports.push(serde_json::json!({"layer":index,"all_exact":true,
                "original_output_sha256":expected.iter().map(|v|format!("{:x}",Sha256::digest(v))).collect::<Vec<_>>(),
                "prequantized_shared_inputs":[a.ffn_boundary().quantized_input().is_some(),b.ffn_boundary().quantized_input().is_some()],
                "records":records}));
        }
        Ok(serde_json::json!({"all_exact":true,"layers":reports,
            "modes":["separate_ab","pooled_ab","separate_ba","pooled_ba"],
            "scope":"Two real pending target M6 forwards. Original routers and shared experts; only routed expert work pooled into twelve live M16 rows. Includes packing and split addition. All local FFN output bits match the original whole-model forward. Excludes attention, TP4 reductions, residual boundaries and weight lookahead; not whole-model throughput."}))
    }
    /// Read-only (weight read census): the four sources' compressed KV caches and index key caches as
    /// (name, address, bytes). Elastic caches list only what exists (the resident span and mapped regions).
    pub(crate) fn cache_arenas(&self) -> Vec<(String, u64, usize)> {
        let mut out = Vec::new();
        for source in [&self.source2, self.group8.source(), self.group14.source(), self.late.source()] {
            let id = source.layer_id();
            for (name, data) in [(format!("kv{id}"), source.cache().data()), (format!("keys{id}"), source.keys().data())] {
                match data.elastic_mapped_extents() {
                    Ok(extents) => for (i, (at, bytes)) in extents.into_iter().enumerate() { out.push((format!("{name}.{i}"), at, bytes)); },
                    Err(_) => out.push((name, data.ptr(), data.bytes())),
                }
            }
        }
        out
    }
    /// Regions whose keys a profile's index branches hold: every serving region for the batched (16-row verify) owner
    /// under full serving regions, else one.
    pub(crate) fn key_regions_for(rows: usize, phase: Phase, context: usize) -> Result<usize> {
        if rows == 16 && phase == Phase::Verify && crate::config::full_regions()? {
            ensure!(context == crate::config::MODEL_CONTEXT, "full serving regions need the model context");
            return crate::config::batched_regions();
        }
        Ok(1)
    }
    /// Full serving regions: image-pool chunks that mapping region `k`'s rows of this profile's index key matrices
    /// would take (0 unless they are elastic region matrices).
    pub(crate) fn region_keys_missing_chunks(&self, k: usize) -> Result<usize> {
        let mut chunks = 0;
        for q in self.index_queries().into_iter().flatten() { chunks += q.region_key_missing_chunks(k)?; }
        Ok(chunks)
    }
    /// Map region `k`'s rows of every index key matrix of this profile, all or nothing: Ok(false), with nothing mapped,
    /// without room. Call before the region's first step.
    pub(crate) fn map_region_keys(&self, k: usize) -> Result<bool> {
        if self.region_keys_missing_chunks(k)? > self.cuda.image_pool_free_chunks().unwrap_or(0) { return Ok(false); }
        let queries: Vec<_> = self.index_queries().into_iter().flatten().collect();
        for (i, q) in queries.iter().enumerate() {
            if !q.map_region_keys(k)? {
                for done in &queries[..i] { done.unmap_region_keys(k)?; }
                return Ok(false);
            }
        }
        Ok(true)
    }
    /// Release region `k`'s key rows (its conversation left; the unmap drains the context first).
    pub(crate) fn unmap_region_keys(&self, k: usize) -> Result<()> {
        for q in self.index_queries().into_iter().flatten() { q.unmap_region_keys(k)?; }
        Ok(())
    }
    /// Keys by reach (config::region_keys_by_reach): the same for region `k`'s key rows of positions [from, to).
    pub(crate) fn region_keys_span_missing_chunks(&self, k: usize, from: usize, to: usize) -> Result<usize> {
        let mut chunks = 0;
        for q in self.index_queries().into_iter().flatten() { chunks += q.region_key_span_missing_chunks(k, from, to)?; }
        Ok(chunks)
    }
    pub(crate) fn map_region_keys_span(&self, k: usize, from: usize, to: usize) -> Result<bool> {
        let queries: Vec<_> = self.index_queries().into_iter().flatten().collect();
        for (i, q) in queries.iter().enumerate() {
            if !q.map_region_key_span(k, from, to)? {
                for done in &queries[..i] { done.unmap_region_key_span(k, from, to)?; }
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub(crate) fn unmap_region_keys_span(&self, k: usize, from: usize, to: usize) -> Result<()> {
        for q in self.index_queries().into_iter().flatten() { q.unmap_region_key_span(k, from, to)?; }
        Ok(())
    }
    /// Pages [first, last) that serving region `k` of whole conversations (full regions) occupies in a cache of
    /// `ratio` with `page_tokens` slots a page: its slots are k x context / ratio + 32 onward, one conversation long.
    /// Pages of serving region `k` (one source of compression `ratio`, pages of `page_tokens` rows) that hold its
    /// conversation's positions [from, to). The whole region is [0, MODEL_CONTEXT).
    fn serving_region_span_pages(k: usize, ratio: usize, page_tokens: usize, from: usize, to: usize) -> (usize, usize) {
        let context = crate::config::MODEL_CONTEXT;
        let first = k * context / ratio + COMPRESSED_OFFSET;
        let (a, b) = (first + from / ratio, first + to.min(context).div_ceil(ratio));
        (a / page_tokens, b.div_ceil(page_tokens))
    }
    /// Elastic regions (config::elastic_regions): image-pool chunks that mapping serving region `k` would take now.
    pub(crate) fn serving_region_missing_chunks(&self, k: usize) -> Result<usize> {
        self.serving_region_span_missing_chunks(k, 0, crate::config::MODEL_CONTEXT)
    }
    /// As serving_region_missing_chunks, for the positions [from, to) of the region's conversation only (map by reach,
    /// config::region_map_by_reach).
    pub(crate) fn serving_region_span_missing_chunks(&self, k: usize, from: usize, to: usize) -> Result<usize> {
        ensure!(k >= 1 && crate::config::elastic_regions()? && from < to, "elastic serving region");
        let mut chunks = 0;
        for source in [&self.source2, self.group8.source(), self.group14.source(), self.late.source()] {
            let ratio = if source.layer_id() == 20 { 1 } else { 2 };
            let (a, b) = Self::serving_region_span_pages(k, ratio, source.cache().page_tokens(), from, to);
            chunks += source.cache().missing_chunks(a, b)?;
            let (a, b) = Self::serving_region_span_pages(k, ratio, 64, from, to);
            chunks += source.keys().missing_chunks(a, b)?;
        }
        Ok(chunks)
    }
    /// Whether the image pool's free chunks can map serving region `k` now.
    pub(crate) fn serving_region_room(&self, k: usize) -> Result<bool> {
        Ok(self.serving_region_missing_chunks(k)? <= self.cuda.image_pool_free_chunks().unwrap_or(0))
    }
    /// Map serving region `k`'s spans of the four sources' compressed and key caches from the image pool, all or
    /// nothing: Ok(false), with nothing mapped, without room. Call before anything is copied into the region.
    pub(crate) fn map_serving_region(&self, k: usize) -> Result<bool> {
        self.map_serving_region_span(k, 0, crate::config::MODEL_CONTEXT)
    }
    /// As map_serving_region, for the positions [from, to) of the region's conversation. Every span mapped must later
    /// be released with unmap_serving_region_span over the same positions (chunks are reference counted).
    pub(crate) fn map_serving_region_span(&self, k: usize, from: usize, to: usize) -> Result<bool> {
        if self.serving_region_span_missing_chunks(k, from, to)? > self.cuda.image_pool_free_chunks().unwrap_or(0) {
            return Ok(false);
        }
        for source in [&self.source2, self.group8.source(), self.group14.source(), self.late.source()] {
            let ratio = if source.layer_id() == 20 { 1 } else { 2 };
            let (a, b) = Self::serving_region_span_pages(k, ratio, source.cache().page_tokens(), from, to);
            ensure!(source.cache().map_pages(a, b)?, "elastic region map lost its room");
            let (a, b) = Self::serving_region_span_pages(k, ratio, 64, from, to);
            ensure!(source.keys().map_pages(a, b)?, "elastic region map lost its room");
        }
        Ok(true)
    }
    /// Release serving region `k`'s spans (after its conversation left; the unmap drains the context first).
    pub(crate) fn unmap_serving_region(&self, k: usize) -> Result<()> {
        self.unmap_serving_region_span(k, 0, crate::config::MODEL_CONTEXT)
    }
    pub(crate) fn unmap_serving_region_span(&self, k: usize, from: usize, to: usize) -> Result<()> {
        ensure!(k >= 1 && crate::config::elastic_regions()? && from < to, "elastic serving region");
        for source in [&self.source2, self.group8.source(), self.group14.source(), self.late.source()] {
            let ratio = if source.layer_id() == 20 { 1 } else { 2 };
            let (a, b) = Self::serving_region_span_pages(k, ratio, source.cache().page_tokens(), from, to);
            source.cache().unmap_pages(a, b)?;
            let (a, b) = Self::serving_region_span_pages(k, ratio, 64, from, to);
            source.keys().unmap_pages(a, b)?;
        }
        Ok(())
    }
    pub(crate) fn layer_owners(&self) -> Vec<&SwaLayer> {
        self.early
            .iter()
            .chain(self.group8.layers())
            .chain(self.group14.layers())
            .chain(self.late.layers())
            .collect()
    }
    pub(crate) fn repeat_layer_sources(&self,live:usize,compact:bool)->Result<Vec<(String,u64,usize)>> {
        ensure!(std::env::var("CANADA_SCRATCH_READOUT_LAYER_TRACE").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_SAMPLER_READOUT_SCREEN").as_deref()==Ok("1")
            &&self.rows==6&&self.source_phase==Phase::Verify
            &&self.pending.as_ref().is_some_and(|p|p.live==live),
            "repeat layer capture requires a completed pending Verify6");
        let deep=std::env::var("CANADA_SCRATCH_READOUT_DEEP_TRACE").as_deref()==Ok("1");
        ensure!(!compact||deep,"compact peer sources require deep tracing");
        let mut sources=Vec::with_capacity(if compact {160}else if deep {560}else{80});
        for (index,layer) in self.layer_owners().into_iter().enumerate() {
            for (name,boundary,local,reduced) in [
                ("attn",layer.attn_boundary(),layer.attention().output(),layer.attn_reduced()),
                ("ffn",layer.ffn_boundary(),layer.ffn().output(),layer.ffn_reduced())] {
                if compact {
                    ensure!(local.len()==6*5120&&reduced.len()==6*5120,
                        "compact peer source geometry");
                    sources.push((format!("layer{index:02}-{name}-local"),local.ptr(),live*5120*2));
                    sources.push((format!("layer{index:02}-{name}-reduced"),reduced.ptr(),live*5120*2));
                    continue;
                }
                let input=boundary.input();
                ensure!(input.len()==6*5120,"repeat layer input geometry");
                sources.push((format!("layer{index:02}-{name}"),input.ptr(),live*5120*2));
                if deep {
                    // All of these buffers belong to this layer and survive
                    // the forward. The shared expert scratch is excluded.
                    ensure!(local.len()==6*5120&&reduced.len()==6*5120
                        &&boundary.output().len()==6*20480&&boundary.pre().len()==6*4
                        &&boundary.post().len()==6*4&&boundary.comb().len()==6*16,
                        "deep repeat layer storage geometry");
                    for (suffix,ptr,bytes) in [
                        ("local",local.ptr(),live*5120*2),
                        ("reduced",reduced.ptr(),live*5120*2),
                        ("residual",boundary.output().ptr(),live*20480*2),
                        ("pre",boundary.pre().ptr(),live*4*4),
                        ("post",boundary.post().ptr(),live*4*4),
                        ("comb",boundary.comb().ptr(),live*16*4)] {
                        sources.push((format!("layer{index:02}-{name}-{suffix}"),ptr,bytes));
                    }
                }
            }
        }
        ensure!(sources.len()==if compact {160}else if deep {560}else{80},"repeat layer coverage");Ok(sources)
    }
    pub(crate) fn repeat_state_digests(&self,stream:&Stream)->Result<serde_json::Value> {
        ensure!((std::env::var("CANADA_SCRATCH_READOUT_LAYER_TRACE").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_SAMPLER_READOUT_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_ATTENTION16_COMMIT_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_INDEPENDENT_OWNER_SCREEN").as_deref()==Ok("1"))
            &&self.lease.is_current()&&self.pending.is_none()
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,
            "repeat state digest requires a committed owner");
        stream.synchronize()?;
        let digests=self.state_image_regions(self.history.position as usize,true)?
            .iter().map(|r|r.repeat_digest()).collect::<Result<Vec<_>>>()?;
        Ok(serde_json::json!({"position":self.history.position,"regions":digests}))
    }
    pub(crate) fn save_repeat_physical_image(&self,comm:&Roce,stream:&Stream)
        ->Result<RepeatPhysicalImage> {
        ensure!(std::env::var("CANADA_SCRATCH_ATTENTION16_COMMIT_SCREEN").as_deref()==Ok("1"),
            "physical replay snapshot is diagnostic only");
        self.state_image_guard(comm,stream)?;
        let frontier=self.history.position as usize;
        let extent=frontier.checked_add(64).context("diagnostic future extent")?;
        let regions=self.state_image_regions(extent,true)?;
        let mut image=crate::state_image::Image::try_allocate(&self.cuda,&regions,1<<30,stream)?
            .ok_or(crate::state_image::CapacityRefused)?;
        stream.synchronize()?;comm.health()?;self.lease.begin()?;
        image.save(&regions,stream)?;comm.health()?;self.lease.finish_read()?;
        Ok(RepeatPhysicalImage{image,frontier,extent})
    }
    pub(crate) fn restore_repeat_physical_image(&mut self,saved:&RepeatPhysicalImage,
        comm:&Roce,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_ATTENTION16_COMMIT_SCREEN").as_deref()==Ok("1")
            &&self.history.position as usize==saved.frontier,
            "restore ordinary causal frontier before diagnostic page slack");
        self.state_image_guard(comm,stream)?;
        let regions=self.state_image_regions(saved.extent,true)?;
        saved.image.validate(&regions,stream)?;
        stream.synchronize()?;comm.health()?;self.lease.begin()?;
        unsafe{saved.image.restore(&regions,stream)}?;
        drop(regions);comm.health()?;self.lease.finish_write()?;
        Ok(())
    }
    pub(crate) fn repeat_ffn_sources(&self,live:usize)->Result<Vec<(String,u64,usize)>> {
        ensure!(self.rows==6&&self.source_phase==Phase::Verify
            &&self.pending.as_ref().is_some_and(|p|p.live==live),"repeat FFN completed geometry");
        let mut sources=Vec::with_capacity(320);
        for (index,layer) in self.layer_owners().into_iter().enumerate() {
            for (name,ptr,bytes) in layer.ffn().repeat_owned_sources(live)? {
                sources.push((format!("layer{index:02}-ffn-{name}"),ptr,bytes));
            }
            // These are the actual operands consumed by the shared projection,
            // produced by the preceding fused normalization on the main stream.
            // Capture full swizzled scales, including persistent padding lanes.
            let (values,scales)=layer.ffn_boundary().quantized_input()
                .ok_or_else(||anyhow::anyhow!("detailed FFN trace requires fused input quantization"))?;
            ensure!(values.len()==6*5120&&scales.len()==20480,"FFN quantized trace shape");
            sources.push((format!("layer{index:02}-ffn-shared-input-values"),values.ptr(),live*5120));
            sources.push((format!("layer{index:02}-ffn-shared-input-scales"),scales.ptr(),scales.bytes()));
        }
        ensure!(sources.len()==320,"repeat FFN coverage");Ok(sources)
    }
    fn causal_storage_ids(&self) -> Vec<u64> {
        let mut ids = self
            .early_caches
            .iter()
            .chain(self.group8.caches())
            .chain(self.group14.caches())
            .chain(self.late.caches())
            .map(|c| c.data().ptr())
            .collect::<Vec<_>>();
        for source in [
            &self.source2,
            self.group8.source(),
            self.group14.source(),
            self.late.source(),
        ] {
            ids.extend([
                source.cache().data().ptr(),
                source.keys().data().ptr(),
                source.source().ring().map_or(0, |r| r.ptr()),
            ]);
        }
        ids
    }
    fn state_image_regions(&self, position:usize, compact:bool) -> Result<Vec<crate::state_image::Region<'_>>> {
        self.state_image_regions_for_layers(position,compact,false)
    }
    fn state_image_regions_for_layers(&self, position:usize, compact:bool,encoder_only:bool)
        -> Result<Vec<crate::state_image::Region<'_>>> {
        self.state_image_regions_for_layers_mode(position,compact,encoder_only,false)
    }
    fn state_image_regions_for_layers_mode(&self,position:usize,compact:bool,encoder_only:bool,
        bank_retained:bool)->Result<Vec<crate::state_image::Region<'_>>> {
        use crate::state_image::Region;
        ensure!(!bank_retained||(compact&&encoder_only),"bank snapshot is encoder-only and compact");
        ensure!(self.wide.is_none() && self.layer_owners().len() == 40,
            "state image requires a complete target owner");
        let mut regions = Vec::new();
        ensure!(position>0 && position<=self.context,"state image saved frontier");
        let swa_pages=compact.then(||saved_swa_pages(self.swa_layout,position));
        for (layer, cache) in self.early_caches.iter()
            .chain(self.group8.caches()).chain(self.group14.caches()).chain(self.late.caches())
            .take(if encoder_only {21}else{40}).enumerate() {
            if let Some(pages)=&swa_pages {
                ensure!(cache.page_tokens()==64,"compact SWA page geometry");
                for &(first,end) in pages {
                    regions.push(Region::byte_range(&self.cuda,format!("swa.{layer}.pages{first}-{end}"),
                        cache.data(),first*cache.page_bytes()..end*cache.page_bytes())?);
                }
            }else{regions.push(Region::new(&self.cuda, format!("swa.{layer}"), cache.data())?);}
        }
        for (layer, source) in [2, 8, 14, 20].into_iter().zip([
            &self.source2, self.group8.source(), self.group14.source(), self.late.source(),
        ]) {
            if compact {
                // Include the offset/padding page and every published compressed
                // row. Whole final pages also preserve packing/scales and slack.
                let ratio=if layer==20 {1}else{2};
                let slots=COMPRESSED_OFFSET+position.div_ceil(ratio);
                let pages=slots.div_ceil(64);
                ensure!(source.cache().page_tokens()==64 && pages<=source.cache().pages()
                    && pages*64<=source.keys().capacity(),"compact compressed page geometry");
                // Append/verification metadata writes at or beyond the
                // published frontier. Keep its whole final page, including
                // a half-complete C2 pair and packed per-row scales/slack.
                // Earlier pages remain in this exact bank and lineage.
                let first=if bank_retained{pages-1}else{0};
                regions.push(Region::byte_range(&self.cuda,format!("compressed.{layer}.kv"),
                        source.cache().data(),first*source.cache().page_bytes()..pages*source.cache().page_bytes())?.exchange_prefix());
                regions.push(Region::byte_range(&self.cuda,format!("compressed.{layer}.keys"),
                        source.keys().data(),first*64*68..pages*64*68)?.exchange_prefix());
            }else{
                regions.push(Region::new(&self.cuda, format!("compressed.{layer}.kv"), source.cache().data())?);
                regions.push(Region::new(&self.cuda, format!("compressed.{layer}.keys"), source.keys().data())?);
            }
            if let Some(ring) = source.source().ring() {
                // With several request slots (serving regions, screens) an image is region 0's
                // conversation: its carry is request slot 0 (the ring is request-major).
                let slots = crate::config::session_slots()?;
                if slots > 1 {
                    regions.push(Region::byte_range(&self.cuda, format!("compressed.{layer}.carry"), ring,
                        0..ring.bytes() / slots)?);
                } else {
                    regions.push(Region::new(&self.cuda, format!("compressed.{layer}.carry"), ring)?);
                }
            }
        }
        let encoder = self.encoder_cache.as_ref()
            .ok_or_else(|| anyhow::anyhow!("complete state image requires retained encoder history"))?;
        regions.extend(encoder.image_regions()?);
        Ok(regions)
    }
    fn state_image_guard(&self, comm: &Roce, stream: &Stream) -> Result<()> {
        ensure!(self.lease.is_current() && self.pending.is_none()
            && self.history.position <= self.context as u64
            && comm.rank() == self.rank && self.comm_instance == Some(comm.instance())
            && self.cuda.owns_stream(stream) && stream.raw() as usize == self.stream,
            "state image needs a healthy committed target and its original transport/stream");
        ensure!(self.encoder_cache.as_ref().is_some_and(|c| c.position() as u64 == self.history.position),
            "state image encoder/target frontier disagreement");
        Ok(())
    }
    pub(crate) fn compact_state_image_bytes(&self)->Result<usize> {
        ensure!(self.lease.is_current() && self.pending.is_none(),"state size needs a committed owner");
        Ok(self.state_image_regions(self.history.position as usize,true)?.iter().map(|r|r.bytes()).sum())
    }
    pub(crate) fn exchange_state_image_bytes(&self)->Result<usize> {
        ensure!(self.lease.is_current()&&self.pending.is_none(),"exchange size needs a committed owner");
        crate::state_image::exchange::Image::estimate(&self.state_image_regions(self.history.position as usize,true)?,2<<30)
    }
    pub(crate) fn save_exchange_state_image(&self,budget:usize,comm:&Roce,stream:&Stream)->Result<StateImage>{
        self.state_image_guard(comm,stream)?;
        let regions=self.state_image_regions(self.history.position as usize,true)?;
        let mut image=crate::state_image::exchange::Image::try_allocate(&self.cuda,&regions,budget,stream)?
            .ok_or(crate::state_image::CapacityRefused)?;
        stream.synchronize()?;comm.health()?;self.lease.begin()?;
        ensure!(self.encoder_cache.as_ref().unwrap().error()?==0,"encoder error before exchange image");
        image.save(&regions,stream)?;comm.health()?;self.lease.finish_read()?;
        Ok(StateImage{image:crate::state_image::Storage::Exchange(image),owner:self.lease.fork(),
            history:self.history.clone(),context:self.context,rank:self.rank,stream:self.stream,compact:true})
    }
    fn exchange_guard(&self,saved:&StateImage,comm:&Roce,stream:&Stream)->Result<()> {
        self.state_image_guard(comm,stream)?;
        ensure!(self.lease.shares_owner(&saved.owner)&&self.context==saved.context&&self.rank==saved.rank
            &&self.stream==saved.stream&&saved.compact&&saved.exchangeable(),"exchange owner identity");Ok(())
    }
    pub(crate) fn exchange_growth(&self,saved:&StateImage,budget:usize)->Result<(usize,usize)> {
        ensure!(self.lease.is_current()&&self.pending.is_none(),"exchange growth needs committed state");
        let crate::state_image::Storage::Exchange(image)=&saved.image else{anyhow::bail!("nonexchange image");};
        image.growth(&self.state_image_regions(self.history.position as usize,true)?,budget)
    }
    pub(crate) fn prepare_exchange(&self,saved:&StateImage,budget:usize,comm:&Roce,stream:&Stream)
        ->Result<Option<crate::state_image::exchange::Prepared>> {
        self.exchange_guard(saved,comm,stream)?;
        let crate::state_image::Storage::Exchange(image)=&saved.image else{unreachable!()};
        image.validate(&self.state_image_regions(saved.history.position as usize,true)?,stream)?;
        image.try_prepare(&self.state_image_regions(self.history.position as usize,true)?,budget,stream)
    }
    pub(crate) fn exchange_state_image(&mut self,saved:&mut StateImage,prepared:crate::state_image::exchange::Prepared,
        lineage:u64,comm:&Roce,stream:&Stream)->Result<()> {
        self.exchange_guard(saved,comm,stream)?;
        ensure!(lineage>self.history.lineage&&lineage>saved.history.lineage,"exchange lineage must advance");
        let current=self.state_image_regions(self.history.position as usize,true)?;
        let previous=self.state_image_regions(saved.history.position as usize,true)?;
        let crate::state_image::Storage::Exchange(image)=&mut saved.image else{unreachable!()};
        stream.synchronize()?;comm.health()?;self.lease.begin()?;
        unsafe{image.exchange(prepared,&current,&previous,self.io.core_module(),stream)}?;
        drop(current);drop(previous);
        self.encoder_cache.as_ref().unwrap().restore_image_frontier(saved.history.position as usize,stream)?;
        comm.health()?;std::mem::swap(&mut self.history,&mut saved.history);self.history.lineage=lineage;
        self.invalidate_gather_caches();self.last_live=0;self.last_append_metrics=None;self.carry_saved_start=None;
        self.lease.finish_write()?;Ok(())
    }
    /// Snapshot only at a fully committed boundary. Budget refusal occurs
    /// before acquiring the writer lease; device failures poison all profiles.
    pub(crate) fn save_state_image(&self, budget: usize, comm: &Roce, stream: &Stream) -> Result<StateImage> {
        self.state_image_guard(comm, stream)?;
        ensure!(self.history.position > 0, "empty target state image");
        let compact=match std::env::var("CANADA_SCRATCH_COMPACT_STATE_IMAGE") {
            Err(std::env::VarError::NotPresent)=>false,
            Ok(value)if value=="0"=>false,
            Ok(value)if value=="1"=>true,
            _=>anyhow::bail!("compact state image must be0or1"),
        };
        let regions = self.state_image_regions(self.history.position as usize,compact)?;
        let mut image = crate::state_image::Image::allocate(&self.cuda, &regions, budget, stream)?;
        image.configure_copy_kernel(self.io.core_module())?;
        stream.synchronize()?;
        comm.health()?;
        self.lease.begin()?;
        ensure!(self.encoder_cache.as_ref().unwrap().error()? == 0, "encoder error before state image");
        image.save(&regions, stream)?;
        comm.health()?;
        self.lease.finish_read()?;
        Ok(StateImage { image:image.into(), owner: self.lease.fork(), history: self.history.clone(),
            context: self.context, rank: self.rank, stream: self.stream, compact })
    }
    /// Finite mechanism screen only. An incomplete decoder owner never escapes
    /// this method: checkpoint creation is immediately followed by its suffix.
    #[allow(clippy::too_many_arguments)]
    /// Finite inline-checkpoint screen: force the policy (None: the environment's) and the
    /// chunk plan of the next checkpoint prefill's first pass (the prefix, or prefix and
    /// suffix when inline).
    pub(crate) fn set_inline_checkpoint_for_probe(&mut self,inline:Option<bool>,plan:Option<Vec<usize>>)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_ENCODER_CHECKPOINT_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_INLINE_CHECKPOINT_SCREEN").as_deref()==Ok("1")
            &&self.rows==128&&self.wide.is_none(),"inline checkpoint probe admission");
        self.checkpoint_inline_for_probe=inline;self.checkpoint_plan_for_probe=plan;Ok(())
    }
    pub(crate) fn prefill_encoder_checkpoint_for_probe(&mut self,wides:&mut[Self],prefix:&[u32],
        suffix:&[u32],budget:usize,comm:&Roce,stream:&Stream)
        ->Result<(EncoderPrefixImage,serde_json::Value)> {
        ensure!(std::env::var("CANADA_SCRATCH_ENCODER_CHECKPOINT_SCREEN").as_deref()==Ok("1")
            && prefix.len()>=256,"encoder checkpoint finite probe admission");
        let(saved,details)=self.prefill_with_encoder_checkpoint(wides,None,prefix,suffix,Some(budget),comm,stream)?;
        Ok((saved.ok_or_else(||anyhow::anyhow!("requested encoder image missing"))?,details))
    }
    pub(crate) fn prefill_encoder_prefix_for_restore_probe(&mut self,wides:&mut[Self],
        prefix:&[u32],comm:&Roce,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_RESTORE_INITIAL_SCREEN").as_deref()==Ok("1")
            &&self.position()==0&&prefix.len()==18688,"finite matched-chunk encoder prefix admission");
        let (encoder,tiny)=self.default_encoder_graph_policy()?;
        let lookahead=std::env::var("CANADA_SCRATCH_ENGRAM_PREFETCH").as_deref()==Ok("1");
        let progress=self.prefill_wide_interruptible_policy(wides,prefix,comm,stream,lookahead,
            self.graph_replay,encoder,tiny,true,None,&mut |_|Ok(false))?;
        ensure!(progress.done==prefix.len(),"matched-chunk prefix incomplete");Ok(())
    }
    pub(crate) fn encoder_checkpoint_image_bytes(&self,position:usize)->Result<usize> {
        ensure!(encoder_checkpoint_enabled()&&self.lease.is_current()&&self.pending.is_none(),
            "encoder checkpoint size requires a healthy admitted owner");
        self.encoder_checkpoint_image_bytes_by_layout(position)
    }
    /// The same size from the owner's storage layout alone (state_image_regions_for_layers_mode reads only buffer
    /// geometry, never history). For room reservations made before this profile adopts the history: k16b (Nov 20),
    /// config::wide16_elastic's borrow ran before the prefill owner's handoff and refused every 16K request.
    pub(crate) fn encoder_checkpoint_image_bytes_by_layout(&self,position:usize)->Result<usize> {
        ensure!(encoder_checkpoint_enabled(),"encoder checkpoint size requires encoder checkpoints");
        Ok(self.state_image_regions_for_layers_mode(position,true,true,
            crate::config::bank_encoder_checkpoint_requested())?.iter().map(|r|r.bytes()).sum())
    }
    /// One indivisible prefill operation. Optionally restore an encoder prefix,
    /// extend it to a new checkpoint, and finish the complete request. All-rank
    /// budget admission belongs to the caller; None skips the optional image.
    /// The >=128-token suffix replaces every omitted decoder/drafter feature.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_with_encoder_checkpoint(&mut self,wides:&mut[Self],
        restore:Option<EncoderPrefixRestore<'_>>,prefix:&[u32],suffix:&[u32],budget:Option<usize>,
        comm:&Roce,stream:&Stream)->Result<(Option<EncoderPrefixImage>,serde_json::Value)> {
        let (image,details,progress)=self.prefill_with_encoder_checkpoint_interruptible(wides,
            restore,prefix,suffix,budget,None,comm,stream,&mut |_|Ok(false),&mut |admit|Ok(admit),&mut |_|Ok(()))?;
        ensure!(progress.done==prefix.len()+suffix.len(),"uninterrupted encoder checkpoint extent");
        Ok((image,details))
    }
    /// Keep the ordinary encoder/checkpoint/suffix plan until cancellation.
    /// An interrupted prefix completes its bounded decoder tail before return;
    /// the caller can then retain a complete, valid partial-prompt owner.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_with_encoder_checkpoint_interruptible(&mut self,wides:&mut[Self],
        restore:Option<EncoderPrefixRestore<'_>>,prefix:&[u32],suffix:&[u32],budget:Option<usize>,
        recycled:Option<crate::state_image::Image>,
        comm:&Roce,stream:&Stream,should_yield:&mut impl FnMut(usize)->Result<bool>,
        admit_image:&mut impl FnMut(bool)->Result<bool>,
        prefilled:&mut impl FnMut(&mut Self)->Result<()>,
    )->Result<(Option<EncoderPrefixImage>,serde_json::Value,PrefillProgress)> {
        // `prefilled` runs once the whole request is prefilled (never after an interruption): with the inline
        // checkpoint before the image is saved, so the caller can sample and expose the first token first.
        let recycled_bytes=recycled.as_ref().map_or(0,crate::state_image::Image::bytes);
        ensure!(recycled.is_none()||budget==Some(recycled_bytes),"encoder recycled storage extent");
        // The caller's request for a second (near-end) image; only the inline path below serves it.
        let extra_request=self.extra_checkpoint.take();self.extra_image=None;
        let previous=restore.as_ref().map_or(self.history.position as usize,|r|r.image_and_lineage().0.position());
        // A short suffix after a restored image, saving no image (CANADA_SCRATCH_ENCODER_SHORT_SUFFIX): one
        // ragged 128-row chunk whose decoder tail replays the image's retained encoder ring.
        let short_restore=suffix.len()<128&&restore.is_some()&&prefix.is_empty()&&budget.is_none()
            &&crate::config::encoder_short_suffix()?;
        // A near-end checkpoint (CANADA_SCRATCH_ENCODER_NEAR_END): fewer than 128 rows follow it, so it is
        // captured inline in the last encoder chunk, which then completes the request (checked below).
        let near_end=suffix.len()<128&&!short_restore;
        ensure!(encoder_checkpoint_enabled()&&!suffix.is_empty()&&previous+prefix.len()>=128
            && (!near_end||(crate::config::encoder_near_end_checkpoint()?>0
                && (restore.is_some()||crate::config::near_end_after_restore()?)))
            && prefix.len()+suffix.len()<=self.context.saturating_sub(previous)
            && prefix.iter().chain(suffix).all(|&id|id<129280),
            "encoder checkpoint requires a complete replacement suffix");
        let started=Instant::now();
        if let Some(restore)=&restore {
            let (image,lineage)=restore.image_and_lineage();
            match restore {
                EncoderPrefixRestore::Borrowed(..)=>self.restore_encoder_prefix(image,lineage,comm,stream)?,
                EncoderPrefixRestore::Imported(_,source,_)=>self.import_encoder_prefix(image,source,lineage,comm,stream)?,
            }
        }
        // Source lifetime is held by the caller, including asynchronous copies
        // and either late allocation refusal. Preserve the diagnostic field.
        let consumed_parent=false;
        let restore_seconds=started.elapsed().as_secs_f64();
        let lookahead=std::env::var("CANADA_SCRATCH_ENGRAM_PREFETCH").as_deref()==Ok("1");
        let(encoder_graph,tiny_encoder_graph)=self.default_encoder_graph_policy()?;
        let prefix_at=Instant::now();
        let forced_plan=self.checkpoint_plan_for_probe.take();
        let inline_plan=self.inline_checkpoint_plan(wides,prefix.len(),suffix.len(),forced_plan.clone())?
            .filter(|_|!encoder_graph);
        ensure!(!near_end||inline_plan.is_some(),"near-end encoder checkpoint needs its inline capture plan");
        if let Some(plan)=inline_plan {
            // One prefill of prefix and suffix: its last encoder chunk runs through the
            // checkpoint and captures the state as of it; the image is saved from that.
            // A yield before the last chunk returns as an interrupted prefix does, and
            // the resumed prefill takes the ordinary prefix/checkpoint/suffix path.
            // Without an admitted image (a duplicate or a refused save) nothing is captured.
            let aux=if budget.is_some(){Some(self.checkpoint_aux()?)}else{None};
            let tail_graph=match std::env::var("CANADA_SCRATCH_PREFILL_TAIL_GRAPH") {
                Ok(value)=>{ensure!(value=="0"||value=="1","invalid prefill tail graph policy");value=="1"},
                Err(std::env::VarError::NotPresent)=>self.graph_replay,
                Err(error)=>return Err(error.into()),
            };
            let all=[prefix,suffix].concat();
            // The extra image lies in the same last chunk, after the checkpoint, 128 rows or more into the chunk.
            let last_start:usize=plan[..plan.len()-1].iter().sum();
            let extra=extra_request.filter(|&(rows,_)|aux.is_some()&&rows>prefix.len()&&rows<=all.len()&&rows>=last_start+128);
            let aux_extra=if extra.is_some(){Some(self.checkpoint_aux_extra()?)}else{None};
            let progress=self.prefill_wide_interruptible_checkpoint(wides,&all,comm,stream,lookahead,tail_graph,
                encoder_graph,tiny_encoder_graph,false,Some(plan),aux.clone().map(|aux|(aux,prefix.len())),
                aux_extra.clone().zip(extra).map(|(aux,(rows,_))|(aux,rows)),should_yield)?;
            if progress.done<all.len() {
                return Ok((None,serde_json::json!({"interrupted":true,"restore_seconds":restore_seconds,
                    "consumed_parent":consumed_parent,"target_total_seconds":started.elapsed().as_secs_f64(),
                    "partial_phases":self.append_metrics()?}),progress));
            }
            let prefix_seconds=prefix_at.elapsed().as_secs_f64();
            prefilled(self)?;
            let save_at=Instant::now();
            let saved=if let (Some(budget),Some(aux))=(budget,&aux) {
                self.save_inline_checkpoint(aux,previous+prefix.len(),budget,recycled,comm,stream,admit_image)?
            }else{None};
            let extra_at=Instant::now();
            if let (Some((rows,bytes)),Some(aux))=(extra,&aux_extra) {
                self.extra_image=self.save_inline_checkpoint(aux,previous+rows,bytes,None,comm,stream,admit_image)?;
            }
            let extra_seconds=extra_at.elapsed().as_secs_f64();
            let save_seconds=save_at.elapsed().as_secs_f64();
            return Ok((saved,serde_json::json!({"restore_seconds":restore_seconds,"consumed_parent":consumed_parent,
                "prefix_seconds":prefix_seconds,"save_seconds":save_seconds-extra_seconds,"poisoned_decoder_bytes":0,
                "extra_checkpoint":extra.map(|(rows,_)|serde_json::json!({"position":previous+rows,
                    "saved":self.extra_image.is_some(),"seconds":extra_seconds})),
                "finite_swa_poison":false,"inline_checkpoint":true,
                "target_total_seconds":started.elapsed().as_secs_f64(),"suffix_phases":self.append_metrics()?}),progress));
        }
        // The complete final suffix replaces every omitted decoder/MTP row.
        // s788/s796 each exercised the128 omission twice (the other requests
        // were controls). The one s785 sampled divergence remains unexplained.
        let defer128=defer_checkpoint_decoder128()?;
        let progress=if prefix.len()>=256 || (defer128&&prefix.len()>=128&&previous>0) {
            self.prefill_wide_interruptible_policy(wides,prefix,comm,stream,lookahead,
                self.graph_replay,encoder_graph,tiny_encoder_graph,true,forced_plan,should_yield)?
        }else if !prefix.is_empty(){
            // Retain the established small-increment numerical path. Its
            // intermediate decoder result is unused and the final suffix
            // still replaces the entire decoder/drafter window.
            self.prefill_wide_interruptible(wides,prefix,comm,stream,None,should_yield)?
        }else{PrefillProgress{done:0,remaining_chunks:Vec::new()}};
        if progress.done<prefix.len() {
            return Ok((None,serde_json::json!({"interrupted":true,"restore_seconds":restore_seconds,
                "consumed_parent":consumed_parent,"target_total_seconds":started.elapsed().as_secs_f64(),
                "partial_phases":self.append_metrics()?}),progress));
        }
        self.state_image_guard(comm,stream)?;
        let prefix_seconds=prefix_at.elapsed().as_secs_f64();
        let save_at=Instant::now();
        let saved=if let Some(budget)=budget {
            // Prefill may capture graphs or load a lazy module after the
            // caller's initial cache admission. Recheck at allocation time;
            // the checkpoint is optional and never owns the live model state.
            let fits=self.cuda.admission_memory()?.admitted_available
                >=crate::memory::floor_bytes()+16*1024usize.pow(2)+budget-recycled_bytes;
            if admit_image(fits)? {
                let bank_retained=crate::config::bank_encoder_checkpoint_requested();
                let regions=self.state_image_regions_for_layers_mode(self.history.position as usize,true,true,bank_retained)?;
                let mut allocation=match recycled {
                    Some(retired)=>retired.try_recycle(&regions,budget,stream)?,
                    None=>crate::state_image::Image::try_allocate(&self.cuda,&regions,budget,stream)?,
                };
                let fits=allocation.is_some() && self.cuda.admission_memory()?.admitted_available
                    >=crate::memory::floor_bytes()+16*1024usize.pow(2);
                if admit_image(fits)? {
                    let mut image=allocation.take().ok_or_else(||anyhow::anyhow!("all-rank image admission differs"))?;
                    image.configure_copy_kernel(self.io.core_module())?;
                    stream.synchronize()?;comm.health()?;self.lease.begin()?;
                    ensure!(self.encoder_cache.as_ref().unwrap().error()?==0,"encoder checkpoint cache fault");
                    image.save(&regions,stream)?;comm.health()?;self.lease.finish_read()?;
                    Some(EncoderPrefixImage(StateImage{image:image.into(),owner:self.lease.fork(),history:self.history.clone(),
                        context:self.context,rank:self.rank,stream:self.stream,compact:true},
                        bank_retained.then(||bank_encoder_lineage::Binding::new(self.history.lineage))))
                }else{None}
            }else{None}
        }else{None};
        let save_seconds=save_at.elapsed().as_secs_f64();
        let poison_decoder=std::env::var("CANADA_SCRATCH_ENCODER_POISON_DECODER").as_deref()==Ok("1")
            &&defer128&&previous>0&&(128..256).contains(&prefix.len());
        let mut poisoned_decoder_bytes=0;
        let finite_swa_poison=std::env::var("CANADA_SCRATCH_ENCODER_POISON_FINITE_SWA").as_deref()==Ok("1");
        if poison_decoder {
            ensure!(std::env::var("CANADA_SCRATCH_ENCODER_CACHE_POISON_SCREEN").as_deref()==Ok("1")
                &&self.late.caches().len()==20,"isolated omitted-decoder poison");
            self.state_image_guard(comm,stream)?;stream.synchronize()?;self.lease.begin()?;
            // Layer20 and retained encoder rings are inputs to the suffix and
            // remain intact. Layers21..39 SWA is discarded by the tail's floor.
            for cache in self.late.caches().iter().skip(1) {
                // Masked KV entries may still enter a zero-weight XV product.
                // 0x40 is finite in FP8/BF16/FP32 and a legal scale exponent;
                // NaN there is not a faithful test of stale finite state.
                let b=cache.data();unsafe{self.cuda.poison_device_for_probe(b.ptr(),b.bytes(),if finite_swa_poison{64}else{255},stream)?;}
                poisoned_decoder_bytes+=b.bytes();
            }
            let replay=self.replay.as_ref().unwrap();
            for (pointer,bytes) in [(self.late.output().ptr(),self.late.output().bytes()),
                (self.late.pre().ptr(),self.late.pre().bytes()),
                (replay.hidden.ptr(),replay.hidden.bytes()),(replay.pre.ptr(),replay.pre.bytes())] {
                unsafe{self.cuda.poison_device_for_probe(pointer,bytes,255,stream)?;}
                poisoned_decoder_bytes+=bytes;
            }
            stream.synchronize()?;comm.health()?;self.lease.finish_write()?;
        }
        // Report progress over the whole compound append, including the
        // already completed prefix. Scheduling uses the uncached remainder.
        // A short restored suffix replays the image's ring (level 1), or runs as the qualified EXTEND over its new
        // rows when the caller kept the drafter tail (level 2, short_restore_extend).
        let extend=short_restore&&std::mem::take(&mut self.short_restore_extend);
        let mut progress=self.prefill_wide_interruptible(wides,suffix,comm,stream,(short_restore&&!extend).then(||vec![128]),
            &mut |done|should_yield(prefix.len()+done))?;
        progress.done+=prefix.len();
        if progress.done==prefix.len()+suffix.len() {prefilled(self)?;}
        Ok((saved,serde_json::json!({"restore_seconds":restore_seconds,"consumed_parent":consumed_parent,"prefix_seconds":prefix_seconds,"save_seconds":save_seconds,
            "short_restore":short_restore,"short_restore_extend":extend,
            "poisoned_decoder_bytes":poisoned_decoder_bytes,
            "finite_swa_poison":finite_swa_poison,
            "target_total_seconds":started.elapsed().as_secs_f64(),"suffix_phases":self.append_metrics()?}),progress))
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn restore_encoder_checkpoint_for_probe(&mut self,wides:&mut[Self],saved:&EncoderPrefixImage,
        lineage:u64,suffix:&[u32],comm:&Roce,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_ENCODER_CHECKPOINT_SCREEN").as_deref()==Ok("1")
            && suffix.len()>=128 && suffix.len()<=self.context.saturating_sub(saved.0.history.position as usize)
            && suffix.iter().all(|&id|id<129280),
            "encoder checkpoint restore requires a complete replacement suffix");
        self.restore_encoder_prefix(saved,lineage,comm,stream)?;
        self.prefill_wide(wides,suffix,comm,stream)
    }
    /// Reconstruct the complete prefix when a future request has a short
    /// suffix. The caller can rebuild the drafter from these128 exact features
    /// and then use the ordinary short-append path.
    pub(crate) fn restore_encoder_checkpoint_full(&mut self,saved:&EncoderPrefixImage,
        lineage:u64,comm:&Roce,stream:&Stream)->Result<()> {
        ensure!(encoder_checkpoint_enabled()&&self.rows==128 && saved.0.history.position>=128,
            "encoder checkpoint reconstruction admission");
        self.restore_encoder_prefix(saved,lineage,comm,stream)?;
        self.reconstruct_encoder_checkpoint_tail(comm,stream)
    }
    pub(crate) fn encoder_prefix_belongs_to(&self,saved:&EncoderPrefixImage)->bool {
        self.lease.shares_owner(&saved.0.owner)
    }
    pub(crate) fn encoder_prefix_reusable_from(&self,saved:&EncoderPrefixImage)->bool {
        self.encoder_prefix_belongs_to(saved)&&saved.1.as_ref().is_none_or(|lineage|
            self.lease.is_current()&&self.history.lineage==lineage.get()
                &&self.history.position>=saved.position()as u64)
    }
    /// The source owns the image; the destination remains an independent
    /// request. Only canonical encoder state is imported, never a drafter,
    /// sampler, pending transaction, media binding or request authority.
    pub(crate) fn import_encoder_checkpoint_full(&mut self,saved:&EncoderPrefixImage,source:&Self,
        lineage:u64,comm:&Roce,stream:&Stream)->Result<()> {
        ensure!(encoder_checkpoint_enabled()&&self.rows==128&&saved.position()>=128,
            "encoder import reconstruction admission");
        self.import_encoder_prefix(saved,source,lineage,comm,stream)?;
        self.reconstruct_encoder_checkpoint_tail(comm,stream)
    }
    fn reconstruct_encoder_checkpoint_tail(&mut self,comm:&Roce,stream:&Stream)->Result<()> {
        let end=self.history.position as usize;let start=end-128;
        let cache=self.encoder_cache.as_ref().unwrap().clone();
        self.lease.begin()?;
        cache.prepare_committed(self.replay.as_mut().unwrap(),end)?;
        let mut tail=Plan::append_with_layout(start as u64,128,128,self.context,self.swa_layout)?;
        tail.floor_swa_at(start)?;self.step.update(&tail,stream)?;
        comm.select_decode_reduction(false,stream)?;comm.select_prefill_rows(128,128,stream)?;
        stream.join_default()?;self.execute_tail(comm,stream,self.graph_replay)?;
        stream.synchronize()?;comm.health()?;self.check_candidates()?;
        ensure!(cache.error()?==0,"encoder checkpoint reconstructed tail fault");
        self.last_live=128;self.lease.finish_write()?;Ok(())
    }
    fn import_encoder_prefix(&mut self,saved:&EncoderPrefixImage,source:&Self,lineage:u64,
        comm:&Roce,stream:&Stream)->Result<()> {
        ensure!(crate::config::encoder_prefix_import_requested(),"encoder prefix import requires explicit admission");
        self.state_image_guard(comm,stream)?;source.state_image_guard(comm,stream)?;
        ensure!(source.encoder_prefix_reusable_from(saved),"encoder import donor prefix is no longer resident");
        let bank_retained=saved.bank_retained();
        let copies=if bank_retained{self.bank_encoder_prefix_copy_plan(source,saved.position())?}else{Vec::new()};
        let saved=&saved.0;
        ensure!(encoder_checkpoint_enabled()&&self.rows==128&&saved.history.position>=128
            &&source.lease.shares_owner(&saved.owner)&&!self.lease.shares_owner(&saved.owner)
            &&self.context==saved.context&&source.context==saved.context&&self.swa_layout==source.swa_layout
            &&self.rank==saved.rank&&source.rank==saved.rank&&self.stream==saved.stream&&source.stream==saved.stream
            &&lineage>self.history.lineage&&lineage>saved.history.lineage,
            "encoder import owner/context/lineage");
        ensure!(self.independent_state_report(source)?["independent_causal_owners"]==true,
            "encoder import requires the same immutable model and independent causal storage");
        let original=source.state_image_regions_for_layers_mode(saved.history.position as usize,true,true,bank_retained)?;
        let destination=self.state_image_regions_for_layers_mode(saved.history.position as usize,true,true,bank_retained)?;
        saved.image.validate_import(&original,&destination,stream)?;
        stream.synchronize()?;comm.health()?;self.lease.begin()?;
        // The source bank owns the omitted immutable pages. Copy them without
        // allocating another full image or mutating the donor's authority.
        if !copies.is_empty() {
            let copied=(||->Result<()>{for(dst,src,bytes)in copies{
                unsafe{self.cuda.copy_device(dst,src,bytes,stream)}?;
            }Ok(())})();
            let completed=stream.synchronize();copied.and(completed)?;
        }
        unsafe{saved.image.import(&original,&destination,stream)}?;
        drop(original);drop(destination);
        self.encoder_cache.as_ref().unwrap().restore_image_frontier(saved.history.position as usize,stream)?;
        comm.health()?;self.history=saved.history.clone();self.history.lineage=lineage;
        self.invalidate_gather_caches();self.last_live=0;self.last_append_metrics=None;self.carry_saved_start=None;
        self.lease.finish_write()?;Ok(())
    }
    // Only compound operations above may expose this restored encoder owner;
    // each completes its decoder before returning successfully.
    fn restore_encoder_prefix(&mut self,saved:&EncoderPrefixImage,lineage:u64,
        comm:&Roce,stream:&Stream)->Result<()> {
        self.state_image_guard(comm,stream)?;
        let bank_retained=saved.bank_retained();
        ensure!(self.encoder_prefix_reusable_from(saved),
            "bank-retained encoder prefix was invalidated by reset or rewind");
        let binding=saved.1.as_ref();
        let saved=&saved.0;
        ensure!(self.lease.shares_owner(&saved.owner)&&self.context==saved.context
            && self.rank==saved.rank&&self.stream==saved.stream&&lineage>self.history.lineage
            && lineage>saved.history.lineage,"encoder checkpoint owner/lineage");
        let regions=self.state_image_regions_for_layers_mode(saved.history.position as usize,true,true,bank_retained)?;
        saved.image.validate(&regions,stream)?;
        stream.synchronize()?;comm.health()?;self.lease.begin()?;
        unsafe{saved.image.restore(&regions,stream)}?;
        drop(regions);
        self.encoder_cache.as_ref().unwrap().restore_image_frontier(saved.history.position as usize,stream)?;
        comm.health()?;self.history=saved.history.clone();self.history.lineage=lineage;
        self.invalidate_gather_caches();self.last_live=0;self.last_append_metrics=None;self.carry_saved_start=None;
        self.lease.finish_write()?;
        if let Some(binding)=binding{binding.set(lineage);}
        Ok(())
    }
    /// Restore storage into the same loaded model owner, assigning a new
    /// lineage and revision. Older execution profiles must explicitly adopt
    /// this owner again; snapshots never rewind collective counters.
    pub(crate) fn restore_state_image(&mut self, saved: &StateImage, lineage: u64,
        comm: &Roce, stream: &Stream) -> Result<()> {
        self.state_image_guard(comm, stream)?;
        ensure!(self.lease.shares_owner(&saved.owner) && self.context == saved.context
            && self.rank == saved.rank && self.stream == saved.stream
            && lineage > self.history.lineage && lineage > saved.history.lineage,
            "state image owner identity or new lineage invalid");
        // Restore descriptors at the saved frontier, never the unrelated
        // request's current frontier. Unpublished future pages are inaccessible
        // under the restored causal lengths and overwritten before publication.
        let regions = self.state_image_regions(saved.history.position as usize,saved.compact)?;
        saved.image.validate(&regions, stream)?;
        stream.synchronize()?;
        comm.health()?;
        self.lease.begin()?;
        unsafe { saved.image.restore(&regions, stream) }?;
        drop(regions);
        self.encoder_cache.as_ref().unwrap().restore_image_frontier(saved.history.position as usize, stream)?;
        comm.health()?;
        self.history = saved.history.clone();
        self.history.lineage = lineage;
        self.invalidate_gather_caches();
        self.last_live = 0;
        self.last_append_metrics = None;
        self.carry_saved_start = None;
        self.lease.finish_write()?;
        Ok(())
    }
    pub(crate) fn independent_state_report(&self,other:&Self)->Result<serde_json::Value> {
        let shared=self.shared_state_report(other);
        let left=self.state_image_regions(1,false)?;
        let right=other.state_image_regions(1,false)?;
        let disjoint=left.iter().all(|a|right.iter().all(|b|!a.overlaps(b)));
        let immutable=shared["all40layer_atlases_shared"]==true
            &&shared["IO_atlas_shared"]==true&&shared["Engram_row_stores_shared"]==true;
        let private=!self.lease.shares_owner(&other.lease)&&!self.io.shares_media_owner(&other.io)
            &&shared["encoder_tail_cache_shared"]==false&&disjoint;
        Ok(serde_json::json!({"shared":shared,"all_causal_regions_disjoint":disjoint,
            "causal_regions":[left.len(),right.len()],"prompt_media_owners_disjoint":!self.io.shares_media_owner(&other.io),
            "independent_causal_owners":immutable&&private}))
    }
    pub(crate) fn shares_lossless_target_head(&self,other:&Self)->bool {
        self.io.shares_lossless_head(&other.io)
    }
    pub fn shared_state_report(&self, other: &Self) -> serde_json::Value {
        // The wide encoder profile owns layers0..20 only; compare that prefix.
        let layers = |t: &Self| if t.wide.is_some() { 21 } else { 40 };
        let left = self.layer_owners();
        let right = other.layer_owners();
        let all_atlases = left.len() == layers(self)
            && right.len() == layers(other)
            && left
                .iter()
                .zip(right)
                .all(|(a, b)| Rc::ptr_eq(&a.shared_weights(), &b.shared_weights()));
        let causal = self.causal_storage_ids() == other.causal_storage_ids()
            && self.swa_layout == other.swa_layout;
        let row_store = Rc::ptr_eq(&self.row_store1, &other.row_store1)
            && self.group14.shares_row_store(&other.group14);
        let io = self.io.shares_atlas(&other.io);
        let same_lease = self.lease.shares_owner(&other.lease);
        let encoder_tail = match (&self.encoder_cache, &other.encoder_cache) {
            (Some(a), Some(b)) => Rc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        serde_json::json!({"all40layer_atlases_shared":all_atlases,"causal_cache_and_ring_storage_shared":causal,
            "Engram_row_stores_shared":row_store,"IO_atlas_shared":io,"same_revision_owner":same_lease,"encoder_tail_cache_shared":encoder_tail,
            "all_weights_and_causal_storage_shared":all_atlases&&causal&&row_store&&io&&same_lease&&encoder_tail})
    }
    /// Adopt the current accepted history after another profile executed. No GPU
    /// state copy or replay is involved; stale writers fail closed until adoption.
    pub fn handoff_from(&mut self, previous: &Self) -> Result<()> {
        ensure!(
            self.rank == previous.rank
                && self.stream == previous.stream
                && self.context == previous.context
                && self.shared_state_report(previous)["all_weights_and_causal_storage_shared"]
                    == true,
            "profile handoff changed model/state/stream ownership"
        );
        self.lease.adopt(&previous.lease)?;
        self.history = previous.history.clone();
        // The single-sequence cached gather restarts after any handoff. Region gathers survive
        // handoffs between the verify owners (their steps only append at frontiers); state from a
        // prefill owner may have rewritten anything, so it starts a new epoch.
        for q in self.index_queries().into_iter().flatten() {q.invalidate_gather_cache();}
        if !matches!(previous.rows, 6 | 16) {
            INDEX_CACHE_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.comm_instance = previous.comm_instance;
        self.last_live = 0;
        self.last_append_metrics = None;
        self.carry_saved_start = None;
        Ok(())
    }
    pub fn profile_is_current(&self) -> bool {
        self.lease.is_current()
    }
    pub(crate) fn prefill_capture_once(&self) -> bool {
        self.prefill_capture_once
    }
    /// Force a new prefill capture for each paired first-use trial. Ordinary
    /// serving cannot call this diagnostic cache invalidation operation.
    pub(crate) fn set_prefill_capture_once_for_probe(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_POST_HTTP_CAPTURE").as_deref() == Ok("1")
            && self.rows == 128 && self.source_phase == Phase::Extend && self.wide.is_none()
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay
            && self.cuda.owns_stream(stream) && stream.raw() as usize == self.stream,
            "capture-once screen requires the idle graph-enabled prefill owner");
        stream.synchronize()?;
        for graph in &mut self.graphs { *graph = None; }
        self.decoder_graph = None;
        self.prefill_capture_once = enabled;
        Ok(())
    }
    /// Force first-use capture without changing weights, state allocations or
    /// numerical policy. Only the finite verification diagnostic may call this.
    pub(crate) fn set_verification_capture_once_for_probe(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_VERIFY_CAPTURE_SCREEN").as_deref() == Ok("1")
            && [6,16].contains(&self.rows) && self.source_phase == Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.graph_replay
            && self.wide.is_none() && self.tiny_append_graphs.is_none()
            && self.cuda.owns_stream(stream) && stream.raw() as usize == self.stream,
            "verification capture screen requires an idle graph-enabled verifier");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.verification_capture_once = enabled;
        Ok(())
    }
    pub(crate) fn query_overlap_layers(&self) -> usize {
        self.layer_owners().iter().filter(|l| l.attention().query_overlap_enabled()).count()
    }
    pub(crate) fn grouped16_index_layers(&self) -> usize {
        self.index_queries().into_iter().flatten().filter(|q|q.grouped16_enabled()).count()
    }
    pub(crate) fn compact_index_layers(&self) -> usize {
        self.index_queries().into_iter().flatten().filter(|q|q.compact_index_enabled()).count()
    }
    pub(crate) fn attention_coefficient_overlap_layers(&self) -> usize {
        self.layer_owners().iter().filter(|l| l.attention_coefficients_overlap()).count()
    }
    pub(crate) fn ffn_overlap_layers(&self)->(usize,usize) {
        let owners=self.layer_owners();
        (owners.iter().filter(|l|l.ffn_overlap_flags().0).count(),
            owners.iter().filter(|l|l.ffn_overlap_flags().1).count())
    }
    pub(crate) fn cached_woa_allocations(&self) -> Vec<(u64, usize)> {
        self.layer_owners().iter().filter_map(|l|l.attention().cached_woa_allocation()).collect()
    }
    pub(crate) fn set_verify16_overlap_for_probe(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        self.set_verify16_schedule_for_probe(if enabled {3}else{0},stream)
    }
    pub(crate) fn set_split_attention_for_probe(&mut self,path:u8,stream:&Stream)->Result<()> {
        ensure!(((self.rows==6&&std::env::var("CANADA_SCRATCH_SPLIT_ATTENTION_SCREEN").as_deref()==Ok("1"))
            ||(self.rows==16&&std::env::var("CANADA_SCRATCH_SPARSE_PRECISION_WIDE_SCREEN").as_deref()==Ok("1")
                &&!crate::layer_attention::generation_fp32_attention16_enabled()?))
            && path<=2 && self.source_phase==Phase::Verify && self.pending.is_none()
            && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw()as usize==self.stream && self.tiny_append_graphs.is_none()&&self.cached_decoder.is_none(),
            "split attention comparison requires idle verifier before append-bank warmup");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) {*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) {*graph=None;}
        self.decoder_graph=None;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_split_attention_for_probe(path,stream)?;
        }
        Ok(())
    }
    pub(crate) fn set_source20_precision_for_probe(&mut self, enabled:bool, stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_SOURCE20_PRECISION_SCREEN").as_deref()==Ok("1")
            && !crate::compressed_source::source20_fp32_reduction_enabled()?
            && self.rows==6 && self.source_phase==Phase::Verify && self.pending.is_none()
            && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw()as usize==self.stream && self.tiny_append_graphs.is_none()
            && self.cached_decoder.is_none(),
            "source20 precision comparison requires idle verifier before append-bank warmup");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) {*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) {*graph=None;}
        self.decoder_graph=None;
        self.late.source_mut().set_source20_precision_for_probe(enabled)
    }
    /// Isolate only the first two SWA operators using the already allocated
    /// BF16 append workspaces. This finite screen never creates an append bank.
    pub(crate) fn set_swa_precision_for_probe(&mut self, enabled:bool, stream:&Stream)->Result<()> {
        ensure!((std::env::var("CANADA_SCRATCH_SWA_PRECISION_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_SWA_PRECISION_WIDE_SCREEN").as_deref()==Ok("1"))
            && !crate::layer_attention::generation_swa_bf16_enabled()?
            && [6,16].contains(&self.rows) && self.source_phase==Phase::Verify && self.pending.is_none()
            && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw()as usize==self.stream && self.tiny_append_graphs.is_none()
            && self.cached_decoder.is_none() && self.early.len()>=2,
            "SWA precision comparison requires an idle verifier before append-bank warmup");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) {*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) {*graph=None;}
        self.decoder_graph=None;
        for layer in self.early.iter_mut().take(2) {
            layer.set_append_bf16_for_probe(enabled,stream)?;
        }
        Ok(())
    }
    pub(crate) fn set_mhc_two_mma_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MHC_TWO_MMA_SCREEN").as_deref()==Ok("1")
            &&[6,16].contains(&self.rows)&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.tiny_append_graphs.is_none(),
            "two-MMA comparison requires idle verifier before append-bank warmup");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_mhc_two_mma_for_probe(enabled,stream)?;
        }
        Ok(())
    }
    pub(crate) fn set_mhc_live_rows_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MHC_LIVE_ROWS_SCREEN").as_deref()==Ok("1")
            &&self.rows==6&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.tiny_append_graphs.is_none(),
            "mHC live-row comparison requires the finite idle M6 verifier");
        let limit=if enabled{6}else{0};
        if limit==self.verification_mhc_live_rows{return Ok(())}
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.verification_mhc_live_rows=limit;Ok(())
    }
    pub(crate) fn set_append_split_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_APPEND_SPLIT_ATTENTION_SCREEN").as_deref()==Ok("1")
            &&[6,16].contains(&self.rows)&&self.source_phase==Phase::Verify&&self.cached_decoder.is_none()
            &&self.pending.is_none()&&self.lease.is_current()&&self.graph_replay
            &&self.cuda.owns_stream(stream)&&stream.raw()as usize==self.stream,
            "append split comparison requires ordinary idle verifier");
        stream.synchronize()?;
        // Only the inactive append bank contains the replaced attention calls.
        // The ordinary generation bank and numerical kernels stay unchanged.
        self.tiny_append_graphs=None;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_append_split_for_probe(enabled,stream)?;
        }
        Ok(())
    }
    pub(crate) fn set_lossless_head_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_LOSSLESS_HEAD_SCREEN").as_deref()==Ok("1")
            && self.rows==6 && self.source_phase==Phase::Verify && self.pending.is_none()
            && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw()as usize==self.stream && self.tiny_append_graphs.is_none(),
            "lossless head comparison requires idle verifier before append-bank warmup");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) {*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) {*graph=None;}
        self.decoder_graph=None;
        self.io.set_lossless_head_for_probe(enabled)
    }
    pub(crate) fn set_verify16_schedule_for_probe(&mut self, mask: u8, stream: &Stream) -> Result<()> {
        ensure!(crate::ffn::wide_overlap_probe(self.rows)
            && self.source_phase==Phase::Verify && self.pending.is_none()
            && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize==self.stream && mask<=3 && self.cached_decoder.is_none(),
            "idle generation-bank wide overlap screen");
        stream.synchronize()?;
        // Destroy every graph referencing the old event schedule before it is
        // replaced. No causal storage or immutable operands are changed.
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph=None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph=None; }
        self.decoder_graph=None;
        self.tiny_append_graphs=None;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_verify16_schedule_for_probe(mask,stream)?;
        }
        Ok(())
    }
    pub(crate) fn set_wide_epilogues_for_probe(&mut self,enabled:bool,kernels:&Path,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_WIDE_EPILOGUE_SCREEN").as_deref()==Ok("1")
            &&[16,32].contains(&self.rows)&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.cached_decoder.is_none(),
            "wide epilogue switch requires idle uncached verifier");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;self.tiny_append_graphs=None;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_wide_epilogues_for_probe(enabled,kernels)?;
        }
        Ok(())
    }
    pub(crate) fn set_compact32_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_COMPACT32_INDEX_SCREEN").as_deref()==Ok("1")
            &&self.rows==32&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.tiny_append_graphs.is_none()
            &&self.adaptive_sparse_index_layers()==0
            &&self.index_queries().into_iter().flatten().all(|q|!q.has_key_sharded())
            &&self.index_queries().into_iter().flatten().filter(|q|q.has_compact32()).count()==4,
            "idle M32 compact-index comparison");
        let active=self.index_queries().into_iter().flatten().filter(|q|q.compact32_active()).count();
        ensure!(active==0||active==4,"mixed M32 compact consumers");
        if (active==4)==enabled {return Ok(())}
        stream.synchronize()?;
        // M32 has no adaptive M6 or key-partition topology. Reuse its otherwise
        // empty second graph bank, retaining both warmed comparison arms.
        std::mem::swap(&mut self.graphs,&mut self.alternate_index_graphs);
        std::mem::swap(&mut self.engram_graphs,&mut self.alternate_index_engram_graphs);
        self.decoder_graph=None;
        for query in self.index_queries().into_iter().flatten() {query.set_compact32_for_probe(enabled)?;}
        Ok(())
    }
    pub(crate) fn set_grouped16_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_GROUPED_SPARSE16_SCREEN").as_deref()==Ok("1")
            && self.rows==16 && self.source_phase==Phase::Verify && self.pending.is_none()
            && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw()as usize==self.stream
            && self.index_queries().into_iter().flatten().filter(|q|q.has_grouped16()).count()==4,
            "idle graph-enabled M16 compact-index comparison");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) {*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) {*graph=None;}
        self.decoder_graph=None;
        for query in self.index_queries().into_iter().flatten() {query.set_grouped16_for_probe(enabled);}
        if std::env::var("CANADA_SCRATCH_GROUPED16_SERIAL_ATTENTION").as_deref()==Ok("1") {
            // Compare the previously correct compact/serial schedule against
            // the current dense/overlapped default under one model load.
            for layer in self.early.iter_mut().chain(self.group8.layers_mut())
                .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
                layer.set_verify16_attention_for_probe(!enabled,stream)?;
            }
        } else if std::env::var("CANADA_SCRATCH_GROUPED16_SERIAL_QUERY").as_deref()==Ok("1") {
            // Compare current dense+query-overlap against compact+serial-query.
            // Keep all coefficient schedules and causal buffers unchanged.
            for layer in self.early.iter_mut().chain(self.group8.layers_mut())
                .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
                layer.set_verify16_query_for_probe(!enabled,stream)?;
            }
        }
        Ok(())
    }
    pub(crate) fn prepare_qkv_alternates_for_probe(&self,kernels:&Path)->Result<Vec<crate::fp8_linear::Linear>> {
        ensure!(std::env::var("CANADA_SCRATCH_QKV_SCHEDULE_SCREEN").as_deref()==Ok("1")
            && self.rows==16 && self.pending.is_none() && self.lease.is_current()
            && self.cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+64*1024usize.pow(2),
            "alternate QKV screen memory/ownership");
        self.layer_owners().into_iter().map(|l|l.attention().legacy_qkv_for_probe(kernels)).collect()
    }
    pub(crate) fn prepare_qkv_fence_alternates_for_probe(&self,kernels:&Path)->Result<Vec<crate::fp8_linear::Linear>> {
        ensure!(std::env::var("CANADA_SCRATCH_QKV_FENCE_SCREEN").as_deref()==Ok("1")
            && matches!(self.rows,6|16)&&self.pending.is_none()&&self.lease.is_current()
            &&self.cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+64*1024usize.pow(2),
            "QKV fence alternate admission/ownership");
        self.layer_owners().into_iter().map(|l|l.attention().producer_qkv_for_probe(kernels)).collect()
    }
    pub(crate) fn set_shared_down_flash_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_SHARED_DOWN_FLASH_SCREEN").as_deref()==Ok("1")
            &&matches!(self.rows,6|16)&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.cached_decoder.is_none(),
            "idle shared-down graph owner");
        stream.synchronize()?;
        for g in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*g=None;}
        for g in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*g=None;}
        self.decoder_graph=None;self.tiny_append_graphs=None;
        for layer in self.layer_owners(){layer.ffn().set_shared_down_flash_for_probe(enabled)?;}
        Ok(())
    }
    pub(crate) fn set_index_query_flash_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_INDEX_QUERY_FLASH_SCREEN").as_deref()==Ok("1")
            &&self.rows==16&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.cached_decoder.is_none(),
            "idle index query graph owner");
        stream.synchronize()?;
        for g in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*g=None;}
        for g in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*g=None;}
        self.decoder_graph=None;self.tiny_append_graphs=None;
        let branches=self.index_queries();
        ensure!(branches.iter().all(Option::is_some),"index query source count");
        for branch in branches.into_iter().flatten(){branch.set_query_flash_for_probe(enabled)?;}
        Ok(())
    }
    pub(crate) fn swap_qkv_fence_for_probe(&mut self,other:&mut[crate::fp8_linear::Linear],stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_QKV_FENCE_SCREEN").as_deref()==Ok("1")
            &&matches!(self.rows,6|16)&&other.len()==40&&self.source_phase==Phase::Verify
            &&self.pending.is_none()&&self.lease.is_current()&&self.graph_replay
            &&self.cuda.owns_stream(stream)&&stream.raw()as usize==self.stream
            &&self.cached_decoder.is_none(),"idle QKV fence graph owner");
        stream.synchronize()?;
        for g in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*g=None;}
        for g in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*g=None;}
        self.decoder_graph=None;self.tiny_append_graphs=None;
        for(layer,alternate)in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()).zip(other){
            layer.swap_qkv_fence_for_probe(alternate)?;
        }
        Ok(())
    }
    pub(crate) fn set_moe_input_address_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MOE_INPUT_ADDRESS_SCREEN").as_deref()==Ok("1")
            &&self.rows==6&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.tiny_append_graphs.is_none(),
            "idle pre-warmup expert input-address screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;
        for layer in self.layer_owners(){layer.narrow_workspace().set_input_address_for_probe(enabled,stream)?;}
        Ok(())
    }
    pub(crate) fn set_moe_fc_pdl_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MOE_FC_PDL_SCREEN").as_deref()==Ok("1")
            &&self.rows==6&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.tiny_append_graphs.is_none(),
            "idle FC PDL screen ownership");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;
        for layer in self.layer_owners(){layer.narrow_workspace().set_fc_pdl_for_probe(enabled,stream)?;}
        Ok(())
    }
    pub(crate) fn set_late_fc_pdl_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MOE_LATE_PDL_SCREEN").as_deref()==Ok("1")
            &&self.rows==6&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.tiny_append_graphs.is_none(),"idle late FC PDL screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;
        for layer in self.layer_owners(){layer.narrow_workspace().set_late_fc_pdl_for_probe(enabled,stream)?;}
        Ok(())
    }
    pub(crate) fn ffn_hint_chunks(&self)->std::collections::BTreeMap<u32,usize> {
        let mut counts=std::collections::BTreeMap::new();
        for layer in self.layer_owners() {
            if let Some(chunk)=layer.ffn().prefetch_chunk() {*counts.entry(chunk).or_default()+=1;}
        }
        counts
    }
    pub(crate) fn set_ffn_hint_chunk_for_probe(&mut self,chunk:u32,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_FFN_HINT_MODEL_SCREEN").as_deref()==Ok("1")
            &&[2048,16384].contains(&chunk)&&self.rows==6&&self.source_phase==Phase::Verify
            &&self.pending.is_none()&&self.lease.is_current()&&self.graph_replay
            &&self.cuda.owns_stream(stream)&&stream.raw()as usize==self.stream,
            "idle finite FFN hint model screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;self.tiny_append_graphs=None;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_ffn_prefetch_chunk_for_probe(chunk,stream)?;
        }
        Ok(())
    }
    pub(crate) fn set_wide_hints_for_probe(&mut self,mask:u8,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_WIDE_L2_SCREEN").as_deref()==Ok("1")
            &&mask<=7&&matches!(self.rows,16|32)&&self.source_phase==Phase::Verify
            &&self.pending.is_none()&&self.lease.is_current()&&self.lease.shared_idle()
            &&self.graph_replay&&self.cuda.owns_stream(stream)&&stream.raw()as usize==self.stream,
            "idle finite wide L2 hint verifier");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;self.tiny_append_graphs=None;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_wide_hints_for_probe(mask,stream)?;
        }
        Ok(())
    }
    pub(crate) fn set_ffn_sum_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_FFN_SUM_SHARED_SCREEN").as_deref()==Ok("1")
            &&matches!(self.rows,6|16|32)&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.tiny_append_graphs.is_none(),"idle fused FFN sum screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;
        for layer in self.layer_owners(){layer.ffn().set_fused_sum_for_probe(enabled)?;}
        Ok(())
    }
    pub(crate) fn set_moe_wide_transfer_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!((std::env::var("CANADA_SCRATCH_MOE_WIDE_TRANSFER_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_PREFILL_BOUNDS_SCREEN").as_deref()==Ok("1"))
            &&matches!(self.rows,16|128|512|1024)&&self.pending.is_none()&&self.lease.shared_idle()
            &&(self.graph_replay||self.wide.is_some())&&self.cuda.owns_stream(stream)&&stream.raw()as usize==self.stream,
            "idle wide expert transfer screen");
        if self.layer_owners().iter().all(|l|l.narrow_workspace().wide_transfer_enabled()==enabled){return Ok(());}
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;self.tiny_append_graphs=None;
        self.encoder_graph=None;self.encoder_graph_live=None;
        for layer in self.layer_owners(){layer.narrow_workspace().set_wide_transfer_for_probe(enabled,stream)?;}
        Ok(())
    }
    pub(crate) fn set_prefill_upper_rows_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_PREFILL_BOUNDS_SCREEN").as_deref()==Ok("1")
            &&self.rows==128&&self.source_phase==Phase::Extend&&self.pending.is_none()
            &&self.lease.shared_idle()&&self.cuda.owns_stream(stream)&&stream.raw()as usize==self.stream,
            "idle upper prefill row screen");
        if self.prefill_upper_rows==enabled{return Ok(())}
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;self.prefill_upper_rows=enabled;Ok(())
    }
    pub(crate) fn set_prefill_woa_rows_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_WOA_PREFILL_SCREEN").as_deref()==Ok("1")
            &&self.rows==128&&self.source_phase==Phase::Extend&&self.pending.is_none()
            &&self.lease.is_current()&&self.lease.shared_idle()
            &&self.cuda.owns_stream(stream)&&stream.raw()as usize==self.stream,
            "idle WO-A prefill row screen");
        if self.prefill_woa_live==enabled{return Ok(())}
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;self.prefill_woa_live=enabled;Ok(())
    }
    pub(crate) fn set_cooperative_epilogues_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MOE_EPILOGUE_SCREEN").as_deref()==Ok("1")
            &&self.rows==6&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.tiny_append_graphs.is_none(),"idle cooperative epilogue screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;
        for layer in self.layer_owners(){layer.narrow_workspace().set_cooperative_epilogues_for_probe(enabled,stream)?;}
        Ok(())
    }
    /// Plan screen (CANADA_SCRATCH_MOE_LIVE_VARIANTS_SCREEN=1): from how many live rows the six-row owner's experts
    /// take the cache-ca plan and the fused FFN sum; graphs recaptured. None restores the configured floors.
    pub(crate) fn set_moe_live_variants_for_probe(&mut self,floors:Option<(usize,usize,bool)>,stream:&Stream)->Result<()> {
        ensure!(crate::config::l2_queue_screen()&&self.rows==6&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.tiny_append_graphs.is_none(),"idle live variant screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;
        let configured=*self.moe_live_floors.get_or_insert_with(||self.early[0].narrow_workspace().live_variant_floors());
        let (cache_ca,fuse,fc_pdl_other)=floors.unwrap_or(configured);
        for layer in self.layer_owners(){layer.narrow_workspace().set_live_variant_floors_for_probe(cache_ca,fuse,fc_pdl_other,stream)?;}
        Ok(())
    }
    pub(crate) fn set_cache_ca_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MOE_CACHE_CA_SCREEN").as_deref()==Ok("1")
            &&self.rows==6&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.tiny_append_graphs.is_none(),"idle cache-ca screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;
        for layer in self.layer_owners(){layer.narrow_workspace().set_cache_ca_for_probe(enabled,stream)?;}
        Ok(())
    }
    pub(crate) fn set_woa_prefetch_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_WOA_PREFETCH_SCREEN").as_deref()==Ok("1")
            &&self.rows==6&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.tiny_append_graphs.is_none(),"idle WO-A hint screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_woa_prefetch_for_probe(enabled)?;
        }
        Ok(())
    }
    pub(crate) fn set_expert_prefetch_for_probe(&mut self,selected:Option<&[usize]>,core:&[u8],stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_EXPERT_PREFETCH_SCREEN").as_deref()==Ok("1")
            &&selected.is_none_or(|s|s.len()==40&&s.iter().all(|&e|e<384))
            &&self.rows==6&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream,"idle expert hint screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;
        // Ordinary warmup can already own an append bank. Drop it alongside
        // the verifier banks before replacing any captured hint descriptor.
        self.tiny_append_graphs=None;
        let module=self.cuda.module(core)?;
        for (index,layer) in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()).enumerate() {
            layer.set_expert_prefetch_for_probe(&module,stream,selected.map(|s|s[index]))?;
        }
        Ok(())
    }
    pub(crate) fn set_key_sharded_index_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        let publisher_only=std::env::var("CANADA_SCRATCH_KEY_SHARDED_PUBLISHER_SCREEN").as_deref()==Ok("1");
        ensure!(std::env::var("CANADA_SCRATCH_KEY_SHARDED_INDEX_SCREEN").as_deref()==Ok("1")
            &&[6,16].contains(&self.rows)&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.tiny_append_graphs.is_none()
            &&self.index_queries().into_iter().flatten().filter(|q|q.has_key_sharded()).count()==if publisher_only{4}else{3},
            "idle key-shard screen ownership");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;
        let automatic_screen=std::env::var("CANADA_SCRATCH_KEY_SHARD_AUTO_SCREEN").as_deref()==Ok("1");
        let automatic=enabled&&automatic_screen;
        for query in self.index_queries().into_iter().flatten(){
            if publisher_only&&!automatic_screen&&!query.has_key_sharded_publisher(){continue;}
            query.set_key_sharded_auto(automatic);
            query.set_key_sharded_for_probe(enabled&&(!automatic||query.key_sharded_eligible_at(self.history.position)));
        }
        Ok(())
    }
    pub(crate) fn active_key_sharded_layers(&self)->usize {
        self.index_queries().into_iter().flatten().filter(|q|q.key_sharded_active()).count()
    }
    pub(crate) fn set_moe_scale_codes_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MOE_SCALE_CODE_SCREEN").as_deref()==Ok("1")
            &&self.rows==6&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.tiny_append_graphs.is_none(),
            "idle scale-code screen ownership");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;
        for layer in self.layer_owners(){layer.narrow_workspace().set_scale_codes_for_probe(enabled,stream)?;}
        Ok(())
    }
    pub(crate) fn prepare_wo_b_alternates_for_probe(&self,kernels:&Path)->Result<Vec<crate::fp8_linear::Linear>> {
        ensure!(crate::fp8_linear::output_projection_probe(self.rows)
            &&self.pending.is_none()&&self.lease.is_current()
            &&self.cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+64*1024usize.pow(2),
            "alternate WO-B screen memory/ownership");
        self.layer_owners().into_iter().map(|l|l.attention().wo_b_pdl_for_probe(kernels)).collect()
    }
    pub(crate) fn set_wo_b_pdl_for_probe(&mut self,mode:u8,other:&mut[crate::fp8_linear::Linear],stream:&Stream)->Result<()> {
        ensure!(crate::fp8_linear::output_projection_probe(self.rows)
            &&self.source_phase==Phase::Verify&&other.len()==40&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.tiny_append_graphs.is_none(),
            "idle pre-warmup WO-B projection screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;
        for(layer,alternate)in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()).zip(other){
            layer.set_wo_b_pdl_for_probe(mode,alternate)?;
        }
        Ok(())
    }
    pub(crate) fn set_wide_mhc_for_probe(&mut self,mode:u8,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_WIDE_MHC_SCREEN").as_deref()==Ok("1")
            &&self.rows==16&&mode<=3&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.tiny_append_graphs.is_none(),
            "idle M16 mHC comparison owner");
        stream.synchronize()?;
        for g in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*g=None;}
        for g in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*g=None;}
        self.decoder_graph=None;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_wide_mhc_for_probe(mode)?;
        }
        Ok(())
    }
    pub(crate) fn set_mhc_pdl_pair_for_probe(&mut self, mode: u8, stream: &Stream) -> Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MHC_PDL_PAIR_SCREEN").as_deref() == Ok("1")
            && mode <= 3 && self.rows == 6 && self.source_phase == Phase::Verify && self.pending.is_none()
            && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream && self.tiny_append_graphs.is_none()
            && self.cached_decoder.is_none(), "idle M6 PDL graph owner");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph = None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph = None; }
        self.decoder_graph = None;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_mhc_pdl_pair_for_probe(mode)?;
        }
        Ok(())
    }
    pub(crate) fn prepare_shared_up_alternates_for_probe(&self,kernels:&Path)->Result<Vec<crate::fp8_linear::Linear>> {
        ensure!(std::env::var("CANADA_SCRATCH_LEGACY16_FFN_W13").as_deref()==Ok("1")
            && self.rows==16 && self.pending.is_none() && self.lease.is_current()
            && self.cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+64*1024usize.pow(2),
            "alternate W13 screen memory/ownership");
        self.layer_owners().into_iter().map(|l|l.ffn().legacy_shared_up_for_probe(kernels)).collect()
    }
    pub(crate) fn set_legacy16_ffn_projection_for_probe(&mut self,legacy:bool,mask:u8,
        other:&mut[crate::fp8_linear::Linear],stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_LEGACY16_FFN_W13").as_deref()==Ok("1")
            && other.len()==40,"W13 schedule screen geometry");
        // Join old auxiliary work and destroy every graph before replacing
        // projection buffers. Both variants share the canonical matrix owner.
        self.set_verify16_schedule_for_probe(mask,stream)?;
        for(layer,alternate)in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()).zip(other) {
            if layer.ffn().narrow_projection()==legacy {layer.swap_shared_up_for_probe(alternate)?;}
        }
        Ok(())
    }
    pub(crate) fn set_qkv_schedule_for_probe(&mut self,legacy:bool,coefficients:bool,
        other:&mut[crate::fp8_linear::Linear],stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_QKV_SCHEDULE_SCREEN").as_deref()==Ok("1")
            && other.len()==40 && (!coefficients||legacy),"QKV schedule screen geometry");
        // This synchronizes and destroys graphs before any bound pointer or
        // event changes. Both QKV allocations remain owned for all variants.
        self.set_grouped16_for_probe(true,stream)?;
        for(layer,alternate)in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()).zip(other) {
            if layer.attention().narrow_projection()==legacy {layer.swap_qkv_for_probe(alternate)?;}
            layer.set_verify16_query_for_probe(false,stream)?;
            layer.set_attention_coefficients_for_probe(coefficients,stream)?;
        }
        Ok(())
    }
    pub(crate) fn set_cached_index16_for_probe(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_CACHED_INDEX16_SCREEN").as_deref()==Ok("1")
            && self.rows==16 && self.source_phase==Phase::Verify && self.pending.is_none()
            && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize==self.stream,"idle graph-enabled M16 index-cache screen");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph=None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph=None; }
        self.decoder_graph=None;
        for query in self.index_queries().into_iter().flatten() {query.set_cached16_for_probe(enabled)?;}
        Ok(())
    }
    /// Diagnostic only: rebuild this idle verifier's graphs after changing the
    /// exact-operand binding. Causal storage and immutable weights are unchanged.
    pub(crate) fn set_woa_cache_for_probe(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_WOA_CACHE_SCREEN").as_deref()==Ok("1")
            && self.rows==16 && self.source_phase==Phase::Verify && self.pending.is_none()
            && self.lease.is_current() && self.graph_replay && self.cuda.owns_stream(stream)
            && stream.raw() as usize==self.stream && self.cached_woa_allocations().len()==40,
            "WO-A cache screen requires the idle prepared M16 verifier");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) { *graph=None; }
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) { *graph=None; }
        self.decoder_graph=None;
        for layer in self.layer_owners() { layer.attention().set_cached_woa_for_probe(enabled)?; }
        Ok(())
    }
    pub(crate) fn set_woa_compact_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        self.set_woa_compact_mode_for_probe(u8::from(enabled),stream)
    }
    pub(crate) fn set_woa_compact_mode_for_probe(&mut self,mode:u8,stream:&Stream)->Result<()> {
        ensure!(mode<=2&&(mode!=2||std::env::var("CANADA_SCRATCH_WOA_MID_REFERENCE_SCREEN").as_deref()==Ok("1")),
            "finite WO-A reference admission");
        ensure!(std::env::var("CANADA_SCRATCH_WOA_MID_SCREEN").as_deref()==Ok("1")
            &&matches!(self.rows,16|32)&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&self.cuda.owns_stream(stream)
            &&stream.raw()as usize==self.stream&&self.lease.shared_idle(),"idle compact WO-A verifier");
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;self.tiny_append_graphs=None;
        for layer in self.layer_owners(){layer.attention().set_compact_woa_for_probe(mode)?;}
        Ok(())
    }
    pub(crate) fn micro_moe_layers(&self) -> usize {
        self.layer_owners().iter().filter(|l| l.micro_moe_fp32()).count()
    }
    pub(crate) fn live_moe_layers(&self) -> usize {
        self.layer_owners().iter().filter(|l| l.live_moe_bounds()).count()
    }
    pub(crate) fn scale_reuse_moe_layers(&self) -> usize {
        self.layer_owners().iter().filter(|l| l.moe_scale_reuse()).count()
    }
    pub(crate) fn task_order_moe_layers(&self) -> usize {
        self.layer_owners().iter().filter(|l| l.moe_task_order()).count()
    }
    pub(crate) fn task_order_m3_layers(&self)->usize {
        self.layer_owners().iter().filter(|l|l.moe_task_order_m3()).count()
    }
    pub(crate) fn narrow_projection_layers(&self) -> usize {
        self.layer_owners().iter().filter(|l| l.narrow_projections()).count()
    }
    pub(crate) fn fused_projection_layers(&self)->usize {
        self.layer_owners().iter().filter(|l|l.fused_projection()).count()
    }
    pub(crate) fn query_norm_quant_layers(&self)->usize {
        self.layer_owners().iter().filter(|l|l.attention().query_norm_quant_enabled()).count()
    }
    pub(crate) fn input_hint_layers(&self)->usize { self.layer_owners().iter().filter(|l|l.input_hint()).count() }
    pub(crate) fn large_input_hint_layers(&self)->usize {
        self.layer_owners().iter().filter(|l|l.large_input_hint()).count()
    }
    pub(crate) fn norm_quant_boundaries(&self)->usize {
        self.layer_owners().iter().map(|l|l.norm_quant_boundaries()).sum()
    }
    pub(crate) fn fused_shared_quantization_layers(&self)->usize {
        self.layer_owners().iter().filter(|l|l.ffn().fused_shared_quantization()).count()
    }
    pub(crate) fn schedule_counts_for_probe(&self)->[usize;4] {
        let mut counts=[0;4];
        for layer in self.layer_owners() {
            for (total,count) in counts.iter_mut().zip(layer.schedule_counts()) {*total+=count;}
        }
        counts
    }
    pub(crate) fn sparse_index_layers(&self)->usize {
        [24,28,32,36].into_iter().filter(|&id|
            self.late.index(id).is_some_and(|index|index.query().uses_sparse6())).count()
    }
    pub(crate) fn packed_index_gather_layers(&self)->usize {
        [self.source2.query().packed_gather_enabled(),self.group8.source().query().packed_gather_enabled(),
            self.group14.source().query().packed_gather_enabled(),self.late.source().query().packed_gather_enabled()]
            .into_iter().filter(|x|*x).count()
            +[24,28,32,36].into_iter().filter(|&id|
                self.late.index(id).is_some_and(|index|index.query().packed_gather_enabled())).count()
    }
    pub(crate) fn index_queries(&self)->[Option<&crate::query_index::Branch>;8] {
        [Some(self.source2.query()),Some(self.group8.source().query()),
            Some(self.group14.source().query()),Some(self.late.source().query()),
            self.late.index(24).map(|i|i.query()),self.late.index(28).map(|i|i.query()),
            self.late.index(32).map(|i|i.query()),self.late.index(36).map(|i|i.query())]
    }
    pub(crate) fn cached_index_gather_layers(&self)->usize {
        self.index_queries().into_iter().flatten().filter(|q|q.cached_gather_enabled()).count()
    }
    pub(crate) fn mapped_metadata_bytes(&self)->usize {self.step.mapped_metadata_bytes}
    pub(crate) fn staged_metadata_bytes(&self)->usize {
        self.step.staged_metadata.as_ref().map_or(0,|s|s.bytes())
    }
    pub(crate) fn staged_metadata_transfer_bytes(&self)->usize {
        self.step.staged_metadata.as_ref().map_or(0,|s|s.transfer_bytes())
    }
    /// Batched screen diagnostic: drop the incremental index-gather caches, which assume
    /// one contiguous growing sequence (CANADA_SCRATCH_BATCH_SCREEN_INVALIDATE=1).
    pub(crate) fn invalidate_gather_caches_for_probe(&self) -> Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_BATCH_SCREEN").as_deref() == Ok("1") && self.pending.is_none(),
            "gather cache invalidation needs the idle batched screen owner");
        self.invalidate_gather_caches(); Ok(())
    }
    /// Enter or leave region mode for a step (every index branch; query_index::set_region_mode).
    fn set_region_step(&mut self, region: bool) -> Result<()> {
        if region == self.region_step { return Ok(()); }
        for q in self.index_queries().into_iter().flatten() { q.set_region_mode(region)?; }
        self.region_step = region;
        Ok(())
    }
    fn invalidate_gather_caches(&self) {
        INDEX_CACHE_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        for q in self.index_queries().into_iter().flatten() {q.invalidate_gather_cache();}
    }
    /// Per-row index gather starts of a region step (CANADA_SCRATCH_REGION_GATHER_CACHE=1): each row
    /// gathers its region's keys from where this owner last gathered them, less a margin of rows a
    /// frontier step may rewrite; everything after any cache change (INDEX_CACHE_EPOCH).
    fn region_gather_starts(&mut self, p: &Plan, parts: &[(SessionBase, History, usize)]) -> Result<(Vec<i32>, Vec<i32>)> {
        let epoch = INDEX_CACHE_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if self.region_gather.epoch != epoch { self.region_gather = RegionGather { epoch, ..Default::default() }; }
        let (mut pair, mut single) = (p.pair_starts.clone(), p.single_starts.clone());
        let (mut next_pair, mut next_single) = (self.region_gather.pair, self.region_gather.single);
        for (k, (base, _, live)) in parts.iter().enumerate() {
            let r = usize::try_from(base.request)?;
            ensure!(r < MAX_BATCHED_SESSIONS, "region gather region index");
            for row in 0..*live {
                let d = k * p.block + row;
                let (ps, pe) = (p.pair_starts[d], p.pair_starts[d] + p.pair_lengths[d]);
                let (ss, se) = (p.single_starts[d], p.single_starts[d] + p.single_lengths[d]);
                pair[d] = (self.region_gather.pair[r] - REGION_GATHER_MARGIN).clamp(ps, pe.max(ps));
                single[d] = (self.region_gather.single[r] - REGION_GATHER_MARGIN).clamp(ss, se.max(ss));
                next_pair[r] = next_pair[r].max(pe);
                next_single[r] = next_single[r].max(se);
            }
        }
        self.region_gather.pair = next_pair;
        self.region_gather.single = next_single;
        Ok((pair, single))
    }
    fn prepare_gather_caches(&self,stream:&Stream)->Result<()> {
        for q in self.index_queries().into_iter().flatten() {q.prepare_gather_cache(stream)?;}
        Ok(())
    }
    pub(crate) fn packed_sparse_gather_layers(&self)->usize {
        [24,28,32,36].into_iter().filter(|&id|
            self.late.index(id).is_some_and(|index|index.query().packed_sparse_gather_enabled())).count()
    }
    pub(crate) fn head_gemv_enabled(&self)->bool {
        self.io.head_gemv_enabled()
    }
    pub(crate) fn index_key_splits(&self)->[u32;8] {
        [self.source2.query().key_splits(), self.group8.source().query().key_splits(),
            self.group14.source().query().key_splits(),
            self.late.source().query().key_splits(),
            self.late.index(24).map_or(0,|i|i.query().key_splits()),
            self.late.index(28).map_or(0,|i|i.query().key_splits()),
            self.late.index(32).map_or(0,|i|i.query().key_splits()),
            self.late.index(36).map_or(0,|i|i.query().key_splits())]
    }
    pub(crate) fn adaptive_sparse_index_layers(&self)->usize {
        [24,28,32,36].into_iter().filter(|&id|
            self.late.index(id).is_some_and(|index|index.query().adaptive_sparse6())).count()
    }
    pub(crate) fn active_sparse_index_layers(&self)->usize {
        [24,28,32,36].into_iter().filter(|&id|
            self.late.index(id).is_some_and(|index|index.query().sparse6_active())).count()
    }
    pub(crate) fn cached_grouped_union_layers(&self)->usize {
        [24,28,32,36].into_iter().filter(|&id|
            self.late.index(id).is_some_and(|index|index.query().cached_grouped_union())).count()
    }
    pub(crate) fn compact_sparse_index_layers(&self)->usize {
        [24,28,32,36].into_iter().filter(|&id|
            self.late.index(id).is_some_and(|index|index.query().compact_sparse6())).count()
    }
    pub(crate) fn fused_compact_publication_layers(&self)->usize {
        [24,28,32,36].into_iter().filter(|&id|
            self.late.index(id).is_some_and(|index|index.query().fused_compact_publication())).count()
    }
    pub(crate) fn enable_compact_sparse_for_probe(&mut self,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_COMPACT_INDEX_AB").as_deref()==Ok("1")
            && self.source_phase==Phase::Verify && self.rows==6 && self.history.position==0
            && self.pending.is_none() && self.graph_captures==0 && self.adaptive_sparse_index_layers()==4
            && self.graphs.iter().chain(&self.alternate_index_graphs).all(Option::is_none)
            && self.engram_graphs.iter().chain(&self.alternate_index_engram_graphs).all(Option::is_none)
            && self.cuda.owns_stream(stream) && self.stream==stream.raw()as usize,
            "compact index screen requires a fresh uncaptured adaptive verifier");
        stream.synchronize()?;
        self.late.enable_compact_sparse6()?;
        ensure!(self.compact_sparse_index_layers()==4,"compact consumer count");
        Ok(())
    }
    fn select_index_graph_bank(&mut self,stream:&Stream)->Result<()> {
        if self.index_queries().into_iter().flatten().any(|q|q.key_sharded_transition(self.history.position).is_some()) {
            // Switching this early index changes captured graph topology. Drop
            // both consumer-index banks and the inactive tiny-append bank; none
            // may retain a graph using the old mode after a long/short restore.
            stream.synchronize()?;
            for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
            for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
            self.decoder_graph=None;
            if let Some(bank)=&mut self.tiny_append_graphs {
                for graph in bank.graphs.iter_mut().chain(&mut bank.alternate_index_graphs){*graph=None;}
                for graph in bank.engram_graphs.iter_mut().chain(&mut bank.alternate_index_engram_graphs){*graph=None;}
                bank.decoder_graph=None;
            }
            for query in self.index_queries().into_iter().flatten(){
                if let Some(enabled)=query.key_sharded_transition(self.history.position){query.set_key_sharded_for_probe(enabled);}
            }
        }
        let layers=self.adaptive_sparse_index_layers();
        ensure!(layers==0 || (layers==4 && self.rows==6 && (self.source_phase==Phase::Verify
            || (self.source_phase==Phase::Extend
                && std::env::var("CANADA_SCRATCH_TINY_APPEND_PHASE_SCREEN").as_deref()==Ok("1")))),
            "adaptive indexing requires all four Verify6 consumer layers");
        if layers==0 {return Ok(());}
        let sparse=self.history.position>=crate::config::adaptive_sparse_min()?;
        if sparse!=self.adaptive_sparse_active {
            self.late.set_adaptive_sparse6(sparse);
            std::mem::swap(&mut self.graphs,&mut self.alternate_index_graphs);
            std::mem::swap(&mut self.engram_graphs,&mut self.alternate_index_engram_graphs);
            self.adaptive_sparse_active=sparse;
        }
        ensure!(self.active_sparse_index_layers()==if sparse {4}else{0},"adaptive graph/scorer mismatch");
        Ok(())
    }
    /// Bounded read-only capture after a completed finite Verify6/Verify16. Each layer
    /// owns normalized input and router output buffers; only expert workspace
    /// is shared. Commit restores carry and publishes encoder rows without
    /// recomputing these buffers. Caller excludes all capture IO from timing.
    pub(crate) fn capture_expert_components(&self,root:&Path,stream:&Stream)->Result<serde_json::Value> {
        use sha2::{Digest,Sha256};
        let cost_pending=std::env::var("CANADA_SCRATCH_PAIR_COST_SCREEN").as_deref()==Ok("1")
            &&[6,16].contains(&self.rows)&&self.pending.is_some()&&self.lease.is_busy_revision();
        let admitted=cost_pending||(self.rows==6&&std::env::var("CANADA_SCRATCH_POOLED_ROUTE_CAPTURE").as_deref()==Ok("1"))
            ||(self.rows==16&&std::env::var("CANADA_SCRATCH_HISTORY_COPY_ROUTE_SCREEN").as_deref()==Ok("1"));
        let live=if cost_pending{self.pending.as_ref().unwrap().live}else{self.last_live};
        ensure!(admitted&&self.source_phase==Phase::Verify
            &&(cost_pending||(self.lease.is_current()&&self.pending.is_none()))&&live>0&&live<=self.rows
            &&self.cuda.owns_stream(stream)&&stream.raw()as usize==self.stream&&!root.exists(),
            "expert component capture requires its finite committed verification screen");
        stream.synchronize()?;
        let owners=self.layer_owners();ensure!(owners.len()==40,"pooled capture layer count");
        let unique_inputs:std::collections::BTreeSet<_>=owners.iter().map(|l|l.ffn_boundary().input().ptr()).collect();
        let unique_routes:std::collections::BTreeSet<_>=owners.iter().map(|l|l.ffn().router().ids().ptr()).collect();
        ensure!(unique_inputs.len()==40&&unique_routes.len()==40,"pooled capture needs per-layer owned input/routes");
        if self.rank==3{std::fs::create_dir_all(root)?;}
        let mut reports=Vec::new();
        for (layer,l)in owners.iter().enumerate() {
            let ids=l.ffn().router().ids().download()?;
            let routed_rows=if cost_pending{live}else{self.rows};
            ensure!(ids.len()==self.rows*6&&ids[..routed_rows*6].iter().all(|id|(0..384).contains(id)),
                "invalid live router rows at layer{layer}, physical={}, live={routed_rows}",self.rows);
            if routed_rows<self.rows {
                let weights=l.ffn().router().routing().download()?;
                ensure!(ids[routed_rows*6..].iter().all(|&id|id == -1)
                    &&weights.len()==self.rows*6&&weights[routed_rows*6..].iter().all(|&v|v == 0.0),
                    "nonempty padded router rows at layer{layer}");
            }
            let mut taps=Vec::new();
            if [0,28].contains(&layer) {
                let input=l.ffn_boundary().input();let weights=l.ffn().router().routing();
                ensure!(input.len()==self.rows*5120&&weights.len()==self.rows*6,"expert capture tensor geometry");
                for (name,ptr,bytes)in [("input",input.ptr(),input.bytes()),
                    ("ids",l.ffn().router().ids().ptr(),self.rows*6*4),("routing",weights.ptr(),self.rows*6*4)] {
                    let raw=unsafe{self.cuda.read_device(ptr,bytes)}?;let file=format!("layer{layer}-{name}.bin");
                    if self.rank==3{std::fs::write(root.join(&file),&raw)?;}
                    taps.push(serde_json::json!({"name":name,"file":file,"bytes":bytes,"sha256":format!("{:x}",Sha256::digest(&raw))}));
                }
            }
            reports.push(serde_json::json!({"layer":layer,"ids":ids,"taps":taps}));
        }
        let index=serde_json::json!({"rank":self.rank,"rows":self.rows,"position":self.history.position,
            "accepted_rows":self.last_live,"live_rows":live,"pending_verification":cost_pending,
            "layers":reports,"owned_inputs_and_routes":true,
            "scope":"Public-task fixed-input verification buffers; pending/committed and physical/live rows recorded explicitly. Pending padding must contain id=-1 and weight=0. Only canonical input/ids/routing for layers0,28 saved onrank3. No expert workspace or causalcache used as a component oracle."});
        if self.rank==3{std::fs::write(root.join("index.json"),serde_json::to_vec_pretty(&index)?)?;}
        Ok(index)
    }
    /// Read only per-layer route IDs after a finite neural verification commits.
    /// `last_live` is the accepted prefix, whereas `verified_rows` also includes
    /// rejected proposals. Do not confuse those extents when pricing prefetch.
    pub(crate) fn capture_consecutive_routes_for_probe(&self,verified_rows:usize,stream:&Stream)
        ->Result<serde_json::Value> {
        ensure!((std::env::var("CANADA_SCRATCH_SOLO_ECONOMICS_SCREEN").as_deref()==Ok("1")
                ||std::env::var("CANADA_SCRATCH_EXPERT_PREFETCH_SCREEN").as_deref()==Ok("1"))
            &&self.rows==6&&self.source_phase==Phase::Verify&&self.lease.is_current()
            &&self.pending.is_none()&&(1..=self.rows).contains(&verified_rows)
            &&self.last_live>0&&self.last_live<=verified_rows
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,
            "consecutive route capture needs a completed finite neural step");
        stream.synchronize()?;
        let mut layers=Vec::new();
        for (layer,owner) in self.layer_owners().iter().enumerate() {
            let ids=owner.ffn().router().ids().download()?;
            ensure!(ids.len()==self.rows*6&&ids[..verified_rows*6].iter().all(|id|(0..384).contains(id)),
                "invalid live route IDs");
            layers.push(serde_json::json!({"layer":layer,"ids":&ids[..verified_rows*6]}));
        }
        ensure!(layers.len()==40,"consecutive route layer count");
        Ok(serde_json::json!({"rank":self.rank,"position":self.history.position,
            "accepted_rows":self.last_live,"verified_rows":verified_rows,"layers":layers}))
    }
    pub(crate) fn expert_routes_for_probe(&self) -> Result<serde_json::Value> {
        // Verification does not publish last_live until its accepted prefix
        // commits. A zero-acceptance timing trial therefore leaves last_live
        // zero even though the next pending verification has live router rows.
        let (_, pending_live) = self.pending_logits()?;
        ensure!((std::env::var("CANADA_SCRATCH_MOE_MICRO_AB").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_MOE_LIVE_AB").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_MOE_SCALE_AB").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_COMPUTE_AB").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_SCHEDULE_ABLATION").as_deref()==Ok("1"))
            && self.rows==6 && pending_live>0 && pending_live<=self.rows,
            "route diagnostics require a completed pending micro comparison");
        let mut reports=Vec::new();
        for (layer,owner) in self.layer_owners().iter().enumerate() {
            let ids=owner.ffn().router().ids().download()?;
            let live=&ids[..pending_live*6];
            ensure!(live.iter().all(|id| (0..384).contains(id)),"invalid live expert route");
            let mut counts=std::collections::BTreeMap::new();
            for id in live {*counts.entry(*id).or_insert(0usize)+=1;}
            reports.push(serde_json::json!({"layer":layer,"pairs":live.len(),
                "distinct_experts":counts.len(),"reuse_counts":counts.values().collect::<Vec<_>>() }));
        }
        Ok(serde_json::json!(reports))
    }
    pub(crate) fn set_sort_shuffle_for_probe(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_SORT_SELECTED_AB").as_deref() == Ok("1")
            && self.lease.is_current() && self.history.position == 0 && self.pending.is_none()
            && self.source_phase == Phase::Verify && self.rows == 6 && self.graph_captures == 0
            && self.graphs.iter().all(Option::is_none) && self.engram_graphs.iter().all(Option::is_none)
            && self.cuda.owns_stream(stream) && self.stream == stream.raw() as usize,
            "sort comparison requires a fresh uncaptured verifier");
        stream.synchronize()?;
        self.source2.set_sort_shuffle_before_capture(enabled)?;
        self.group8.source_mut().set_sort_shuffle_before_capture(enabled)?;
        self.group14.source_mut().set_sort_shuffle_before_capture(enabled)?;
        self.late.set_sort_shuffle_before_capture(enabled)
    }
    /// Eager paired prefill experiment only. Shared state may belong to another
    /// profile, but no execution can be pending on this single ordered stream.
    pub(crate) fn set_prefill_norm_fusion(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!(self.rows >= 128 && self.pending.is_none() && self.graph_captures == 0
            && self.encoder_graph.is_none() && self.decoder_graph.is_none()
            && self.graphs.iter().all(Option::is_none) && self.engram_graphs.iter().all(Option::is_none)
            && self.cuda.owns_stream(stream) && self.stream == stream.raw() as usize,
            "prefill norm policy cannot change a captured or pending execution");
        stream.synchronize()?;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_norm_fusion(enabled)?;
        }
        Ok(())
    }
    /// The completion fence is outside every capture. It can change between
    /// isolated trials while preserving captured kernels and causal storage.
    pub(crate) fn set_prefill_blocking_for_probe(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_PREFILL_WAIT_SCREEN").as_deref()==Ok("1")
            && self.source_phase==Phase::Extend && self.pending.is_none()
            && self.cuda.owns_stream(stream) && self.stream==stream.raw() as usize,
            "prefill completion-wait probe owner");
        stream.synchronize()?;
        self.prefill_blocking_wait=enabled;
        Ok(())
    }
    fn wait_encoder_chunk(&self, stream: &Stream)->Result<()> {
        if self.prefill_blocking_wait && self.rows>=1024 {
            stream.synchronize_blocking()
        } else {stream.synchronize()}
    }
    pub(crate) fn set_verification_poll_for_probe(&mut self,microseconds:u64,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_VERIFY_WAIT_SCREEN").as_deref()==Ok("1")
            &&[0,25,50].contains(&microseconds)&&self.source_phase==Phase::Verify
            &&self.pending.is_none()&&self.lease.is_current()&&self.cached_decoder.is_none()
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,
            "isolated verification completion-polling owner");
        stream.synchronize()?;self.verification_poll_us=microseconds;Ok(())
    }
    /// Screen only (CANADA_SCRATCH_INDEX_SHORT_ROWS_ALTERNATE=1): whether the next steps may take the short index
    /// path (config::index_short_rows), set by the worker between requests.
    pub(crate) fn set_short_index_allowed(&mut self, allowed: bool) -> Result<()> {
        ensure!(self.pending.is_none(), "alternating short index rows");
        self.short_index_allowed = allowed; Ok(())
    }
    /// config::index_short_rows: steps taken on the short path since the last call.
    pub(crate) fn take_short_index_steps(&mut self) -> u64 { std::mem::take(&mut self.short_index_steps) }
    /// Screen only (CANADA_SCRATCH_VERIFY_POLL_ALTERNATE=1): the completion wait of the next verify passes, set by the
    /// worker between requests (1 = spin on the event query, 25 = sleep 25 us between queries).
    pub(crate) fn set_verification_poll_alternate(&mut self,microseconds:u64)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_VERIFY_POLL_ALTERNATE").as_deref()==Ok("1")&&[1,25].contains(&microseconds)
            &&self.pending.is_none(),"alternating verification poll");
        self.verification_poll_us=microseconds;Ok(())
    }
    pub(crate) fn verification_poll_us_requested()->Result<u64> {
        match std::env::var("CANADA_SCRATCH_VERIFY_POLL_US") {
            // "1": query the completion event in a spin loop (no sleep); "25": sleep 25 us between queries.
            Ok(value)=>match value.as_str(){"0"=>Ok(0),"1"=>Ok(1),"25"=>Ok(25),
                _=>anyhow::bail!("verification polling interval must be 0, 1 (spin) or 25 microseconds")},
            Err(std::env::VarError::NotPresent)=>Ok(0),Err(error)=>Err(error.into()),
        }
    }
    pub(crate) fn fullk_boundaries(&self) -> usize {
        self.layer_owners().iter().map(|layer|layer.fullk_boundaries()).sum()
    }
    pub(crate) fn quant_gather_layers(&self) -> usize {
        self.layer_owners().iter().filter(|layer|layer.quant_gather_active()).count()
    }
    pub(crate) fn set_prefill_quant_gather(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!(self.rows >= 128 && self.pending.is_none() && self.graph_captures == 0
            && self.encoder_graph.is_none() && self.decoder_graph.is_none()
            && self.graphs.iter().all(Option::is_none) && self.engram_graphs.iter().all(Option::is_none)
            && self.cuda.owns_stream(stream) && self.stream == stream.raw() as usize,
            "quantized gather policy cannot change a captured or pending execution");
        stream.synchronize()?;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_quant_gather(enabled);
        }
        Ok(())
    }
    pub(crate) fn local_qkv_layers(&self) -> usize {
        self.layer_owners().iter().filter(|layer|layer.local_qkv_active()).count()
    }
    pub(crate) fn set_prefill_local_qkv(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!(self.rows >= 128 && self.pending.is_none() && self.graph_captures == 0
            && self.encoder_graph.is_none() && self.decoder_graph.is_none()
            && self.graphs.iter().all(Option::is_none) && self.engram_graphs.iter().all(Option::is_none)
            && self.cuda.owns_stream(stream) && self.stream == stream.raw() as usize,
            "local QKV policy cannot change a captured or pending execution");
        stream.synchronize()?;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_local_qkv(enabled);
        }
        Ok(())
    }
    pub(crate) fn moe_gather_layers(&self) -> usize {
        self.layer_owners().iter().filter(|layer|layer.moe_gather_active()).count()
    }
    pub(crate) fn set_prefill_moe_gather(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!(self.rows >= 128 && self.pending.is_none() && self.graph_captures == 0
            && self.encoder_graph.is_none() && self.decoder_graph.is_none()
            && self.graphs.iter().all(Option::is_none) && self.engram_graphs.iter().all(Option::is_none)
            && self.cuda.owns_stream(stream) && self.stream == stream.raw() as usize,
            "MoE gather policy cannot change a captured or pending execution");
        stream.synchronize()?;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_moe_gather(enabled);
        }
        Ok(())
    }
    pub(crate) fn set_prefill_hc_fullk(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!(self.rows >= 128 && self.pending.is_none() && self.graph_captures == 0
            && self.encoder_graph.is_none() && self.decoder_graph.is_none()
            && self.graphs.iter().all(Option::is_none) && self.engram_graphs.iter().all(Option::is_none)
            && self.cuda.owns_stream(stream) && self.stream == stream.raw() as usize,
            "full-K policy cannot change a captured or pending execution");
        stream.synchronize()?;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_fullk(enabled)?;
        }
        Ok(())
    }
    /// Carry s269/s272's checked schedules into ordinary candidate generation.
    /// Diagnostic A/B profiles keep their explicit policies. Configure only the
    /// fresh Verify6 owner, so stream ownership cannot change under a graph.
    pub(crate) fn apply_candidate_schedule(&mut self, stream: &Stream) -> Result<&'static str> {
        ensure!(self.lease.is_current() && self.history.position == 0 && self.pending.is_none()
            && self.source_phase == Phase::Verify && self.rows == 6 && self.wide.is_none()
            && self.graph_replay && self.graph_captures == 0
            && self.graphs.iter().all(Option::is_none) && self.engram_graphs.iter().all(Option::is_none)
            && self.cuda.owns_stream(stream) && self.stream == stream.raw() as usize,
            "candidate schedules require a fresh graph-enabled Verify6 owner");
        let policy = match std::env::var("CANADA_SCRATCH_CANDIDATE_SCHEDULE") {
            Ok(v) => v,
            Err(std::env::VarError::NotPresent) => "combined".to_owned(),
            Err(e) => return Err(e.into()),
        };
        ensure!(["control", "engram", "combined"].contains(&policy.as_str()),
            "unknown candidate schedule policy");
        // The accepted screen used io_uring; preserve other I/O profiles until
        // their performance is measured rather than changing them incidentally.
        if std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref() != Ok("1") {
            return Ok("existing_io_profile");
        }
        let query = policy == "combined";
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_query_overlap(query, stream)?;
        }
        self.engram_overlap = policy != "control";
        // config::engram_split14_decode: three graphs a slot (layer 0 | layers 1-13 | layers 14-39); the step reads
        // table 1's rows behind the first and table 14's behind the second.
        if self.engram_overlap && crate::config::engram_split14_decode() {
            ensure!(!self.engram_projection_ahead && !self.engram_split14_adaptive, "split Engram reads exclude the hoisted projection");
            self.engram_split14 = true;
            self.engram_split14_m3_only = false;
        }
        // config::index_fork follows the query fork, which this owner gets here rather than at construction.
        if query { self.install_index_fork(stream)?; }
        Ok(if query { "combined" } else if self.engram_overlap { "engram" } else { "control" })
    }
    pub fn swa_storage_report(&self) -> Result<serde_json::Value> {
        self.swa_layout.report(self.context)
    }
    pub fn physical_rows(&self) -> usize {
        self.rows
    }
    pub(crate) fn engram_input_kind(&self) -> &'static str {
        self.rows1.input_kind()
    }
    pub(crate) fn paired_engram_io_enabled(&self)->bool {
        self.paired_engram_io && (self.rows<=16 ||
            (self.rows==32&&self.source_phase==Phase::Verify&&self.paired_engram32))
    }
    pub(crate) fn adaptive_engram32_requested()->Result<bool> {
        match std::env::var("CANADA_SCRATCH_ENGRAM32_ADAPTIVE_OVERLAP") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid M32 Engram schedule");Ok(value=="1")},
            Err(std::env::VarError::NotPresent)=>Ok(false),
            Err(error)=>Err(error.into()),
        }
    }
    /// The M32 owner uses paired warm reads,
    /// parallel misses, and overlaps those immutable reads with layer0.
    /// M6 and M16 keep their separately qualified schedules.
    pub(crate) fn apply_adaptive_engram32(&mut self,enabled:bool)->Result<()> {
        if !enabled {return Ok(());}
        ensure!(self.lease.is_current()&&self.position()==0&&self.pending.is_none()
            &&self.source_phase==Phase::Verify&&self.rows==32&&self.wide.is_none()
            &&self.graph_replay&&self.graph_captures==0&&self.paired_engram_io
            &&self.graphs.iter().all(Option::is_none)&&self.engram_graphs.iter().all(Option::is_none),
            "M32 Engram schedule requires a fresh graph-enabled paired-I/O owner");
        self.paired_engram32=true;
        self.paired_engram32_adaptive=true;
        self.engram_overlap=true;
        Ok(())
    }
    pub(crate) fn set_paired_engram32_for_probe(&mut self,enabled:bool)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_WIDE_ENGRAM_SCREEN").as_deref()==Ok("1")
            &&self.lease.is_current()&&self.pending.is_none()&&self.rows==32
            &&self.source_phase==Phase::Verify&&self.paired_engram_io,
            "paired32 I/O requires the idle wide Engram probe owner");
        self.paired_engram32=enabled;self.paired_engram32_adaptive=false;Ok(())
    }
    pub(crate) fn set_adaptive_engram32_for_probe(&mut self,enabled:bool)->Result<()> {
        self.set_paired_engram32_for_probe(enabled)?;
        self.paired_engram32_adaptive=enabled;Ok(())
    }
    pub(crate) fn set_engram32_worker_for_probe(&mut self,enabled:bool)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_ENGRAM32_WORKER_SCREEN").as_deref()==Ok("1")
            &&self.lease.is_current()&&self.pending.is_none()&&self.rows==32
            &&self.source_phase==Phase::Verify&&self.paired_engram32_adaptive,
            "Engram helper screen needs idle adaptive M32 verification");
        if enabled&&self.engram32_worker.is_none() {
            self.engram32_worker=Some(rayon::ThreadPoolBuilder::new().num_threads(1).stack_size(512*1024)
                .thread_name(|_|"engram32-reader".to_owned()).build()?);
        }
        self.engram32_worker_enabled=enabled;Ok(())
    }
    pub(crate) fn encoder_tail_digest(&self) -> Result<Option<serde_json::Value>> {
        ensure!(
            self.lease.is_current() && self.pending.is_none(),
            "encoder tail accepted diagnostic"
        );
        self.encoder_cache
            .as_ref()
            .map(|cache| cache.digest())
            .transpose()
    }
    /// Opt-in diagnostic on the CUDA owner thread. Hash only retained encoder
    /// windows/rings, after already-drained operations; no serving trace writes.
    pub(crate) fn trace_prefill_boundary_for_probe(&self,stage:&str)->Result<()> {
        if !PREFILL_BOUNDARY_TRACE.with(|cell|cell.borrow().is_some()){return Ok(());}
        use sha2::{Digest,Sha256};
        ensure!(self.profile_is_current()&&self.pending.is_none(),"prefill trace owner");
        let position=self.history.position as usize;
        let pages=saved_swa_pages(self.swa_layout,position);
        let mut swa=Vec::new();
        for cache in self.early_caches.iter().chain(self.group8.caches())
            .chain(self.group14.caches()).chain(self.late.caches()).take(21) {
            let mut hash=Sha256::new();
            for &(first,end) in &pages {
                let bytes=(end-first)*cache.page_bytes();
                hash.update(unsafe{self.cuda.read_device(cache.data().ptr()+(first*cache.page_bytes())as u64,bytes)}?);
            }
            swa.push(format!("{:x}",hash.finalize()));
        }
        let value=serde_json::json!({"stage":stage,"rows":self.rows,"position":position,
            "encoder":self.encoder_tail_digest()?,"carry":self.carry_digests()?,"swa":swa});
        PREFILL_BOUNDARY_TRACE.with(|cell|cell.borrow_mut().as_mut().unwrap().push(value));
        Ok(())
    }
    pub(crate) fn capture_statistics(&self) -> (Vec<usize>, usize) {
        (
            self.graphs
                .iter()
                .enumerate()
                .filter_map(|(live, graph)| (graph.is_some() || self.engram_graphs[live].is_some()
                    || self.alternate_index_graphs[live].is_some()
                    || self.alternate_index_engram_graphs[live].is_some()).then_some(live))
                .collect(),
            self.graph_captures+self.compact_categorical.as_ref().map_or(0,|s|s.captures()),
        )
    }
    /// Inspect the bank that the next append will select, not the union of
    /// dense and sparse captures retained for other context lengths.
    pub(crate) fn verification_graph_is_warm(&self, live:usize)->bool {
        if !self.graph_replay || live==0 || live>self.rows {return false;}
        if self.index_queries().into_iter().flatten().any(|q|q.key_sharded_transition(self.history.position).is_some()){return false;}
        let alternate=self.adaptive_sparse_index_layers()>0
            && (self.history.position>=crate::config::adaptive_sparse_min().unwrap_or(65536))!=self.adaptive_sparse_active;
        if self.engram_overlap {
            if alternate {self.alternate_index_engram_graphs[live].is_some()}
            else {self.engram_graphs[live].is_some()}
        }else if alternate {self.alternate_index_graphs[live].is_some()}
        else {self.graphs[live].is_some()}
    }
    /// Experimental schedule only. Immutable row lookup may overlap layer0,
    /// which consumes no Engram rows. Both later layers still see the same
    /// complete uploads through an explicit default-to-model stream event.
    pub(crate) fn set_engram_overlap_for_probe(&mut self, enabled: bool) -> Result<()> {
        ensure!(self.lease.is_current() && self.pending.is_none()
            && self.source_phase == Phase::Verify && [6,16,32].contains(&self.rows)
            && self.graph_replay && self.wide.is_none(),
            "Engram overlap requires an idle graph-enabled verification owner");
        self.engram_overlap = enabled;
        Ok(())
    }
    /// Batched serving (CANADA_SCRATCH_BATCHED_ENGRAM_OVERLAP=1): the 16-row owner, which only runs
    /// batched steps, takes the Engram row overlap as the 6-row owner's ordinary steps do.
    pub(crate) fn ensure_batched_engram_overlap(&mut self) -> Result<()> {
        if crate::config::batched_engram_overlap() && !self.engram_overlap && self.rows == 16 {
            self.set_engram_overlap_for_probe(true)?;
        }
        Ok(())
    }
    pub(crate) fn set_engram_projection_ahead_for_probe(&mut self,mode:u8,kernels:&Path,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_ENGRAM_PROJECTION_SCREEN").as_deref()==Ok("1")
            &&mode<=2
            &&matches!(self.rows,6|16)&&self.source_phase==Phase::Verify
            &&self.pending.is_none()&&self.lease.is_current()&&self.graph_replay
            &&self.cuda.owns_stream(stream)&&stream.raw()as usize==self.stream
            &&self.cached_decoder.is_none()&&!self.engram_split14&&!self.engram_split14_adaptive,
            "idle Engram projection requires the common two-part schedule");
        stream.synchronize()?;
        for g in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*g=None;}
        for g in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*g=None;}
        self.decoder_graph=None;self.tiny_append_graphs=None;
        self.group14.set_engram_projection_ahead_for_probe(mode!=0,kernels)?;
        self.engram_projection_ahead=mode!=0;self.engram_projection_m3_only=mode==2;
        self.engram_projection_active=false;Ok(())
    }
    /// Use distinct layer0, layers1..13, and layers14..39 captures so the
    /// second immutable Engram read can overlap the middle graph.
    /// Claude same-allocation A/B screen switch (mode 0 = control). Unknown names fail.
    pub(crate) fn set_claude_experiment_for_probe(&mut self,name:&str,mode:u8,stream:&Stream)->Result<()> {
        match name {
            // Ordering-only toggle for profiles built with the prefill shared branch.
            "prefill_shared_overlap" => {
                ensure!(mode<=1 && crate::ffn::prefill_shared_overlap_requested()?, "prefill shared overlap A/B admission");
                stream.synchronize()?;
                crate::ffn::set_prefill_shared_concurrent_for_probe(mode==1);
                Ok(())
            }
            // Identical-value toggle: wide RoPE in place on the Q-B output (no Q copy).
"early_quant" => {
                ensure!(mode<=1,"early quantized gather A/B mode"); stream.synchronize()?;
                for layer in self.early.iter_mut().take(2) { layer.set_quant_gather(mode==1); }
                Ok(())
            }
            "wob_exchange" => { ensure!(mode<=1,"WO-B exchange A/B mode"); stream.synchronize()?; crate::layer_attention::set_wob_exchange_for_probe(mode==1); Ok(()) }
            "local_shared" => { ensure!(mode<=1,"row-local shared A/B mode"); stream.synchronize()?; crate::ffn::set_local_shared_for_probe(mode==1); Ok(()) }
            "local_shared_overlap" => { ensure!(mode<=1,"row-local shared overlap A/B mode"); stream.synchronize()?; crate::ffn::set_local_shared_overlap_for_probe(mode==1); Ok(()) }
            "local_shared_late" => { ensure!(mode<=1,"row-local shared late-start A/B mode"); stream.synchronize()?; crate::ffn::set_local_shared_late_for_probe(mode==1); Ok(()) }
            "tail_window" => { ensure!(mode<=1,"tail window A/B mode"); stream.synchronize()?;
                TAIL_WINDOW.store((mode==1) as u8, std::sync::atomic::Ordering::Relaxed); Ok(()) }
"mhc_overlap" => {
                ensure!(mode<=1,"wide coefficient overlap A/B mode"); stream.synchronize()?;
                for layer in self.early.iter_mut().chain(self.group8.layers_mut())
                    .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
                    layer.set_wide_stats_overlap(mode==1, stream)?;
                }
                Ok(())
            }
            "wob8" => { ensure!(mode<=1,"MXFP8 WO exchange A/B mode"); stream.synchronize()?; crate::layer_attention::set_wob8_for_probe(mode==1); Ok(()) }
            "wob8_sum4" => { ensure!(mode<=1,"WOB8 four-part sum A/B mode"); stream.synchronize()?; crate::layer_attention::set_wob8_sum4_for_probe(mode==1); Ok(()) }
            "engram_behind_l0" => { ensure!(mode<=1,"Engram staging behind layer 0 A/B mode"); stream.synchronize()?;
                ENGRAM_BEHIND_L0.store((mode==1) as u8, std::sync::atomic::Ordering::Relaxed); Ok(()) }
            "engram_split14" => { ensure!(mode<=1,"split Engram staging A/B mode"); stream.synchronize()?;
                ENGRAM_SPLIT14.store((mode==1) as u8, std::sync::atomic::Ordering::Relaxed); Ok(()) }
                        "q_inplace" => { ensure!(mode<=1,"Q in-place A/B mode"); stream.synchronize()?; crate::layer_attention::set_q_inplace_for_probe(mode==1); Ok(()) }
            // Ordering/layout-only toggle: per-group GemmEx writing token-major WO-A output.
            "woa_token_major" => {
                ensure!(mode<=1,"WO-A token-major A/B admission");
                stream.synchronize()?;
                crate::woa_medium::set_token_major_for_probe(mode==1);
                Ok(())
            }
            _ => anyhow::bail!("unknown Claude experiment {name} (mode {mode})"),
        }
    }
    pub(crate) fn set_engram_split14_for_probe(&mut self,enabled:bool)->Result<()> {
        ensure!((std::env::var("CANADA_SCRATCH_ENGRAM_SPLIT14_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_VERIFICATION_FUSION_SCREEN").as_deref()==Ok("1"))
            && self.lease.is_current() && self.pending.is_none() && self.rows==6
            && self.source_phase==Phase::Verify && self.graph_replay && self.engram_overlap && !self.engram_projection_ahead
            && self.engram_graphs.iter().chain(&self.alternate_index_engram_graphs).all(Option::is_none),
            "Engram split14 requires an idle uncaptured verification profile");
        self.engram_split14=enabled;
        self.engram_split14_m3_only=false;
        self.engram_split14_adaptive=false;
        Ok(())
    }
    pub(crate) fn set_engram_adaptive_split14_for_probe(&mut self)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_ENGRAM_ADAPTIVE_SPLIT14_SCREEN").as_deref()==Ok("1"),
            "adaptive Engram split requires the isolated comparison");
        self.set_engram_split14_for_probe(true)?;
        self.engram_split14_adaptive=true;
        Ok(())
    }
    pub(crate) fn set_engram_split14_m3_for_probe(&mut self)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_VERIFICATION_FUSION_SCREEN").as_deref()==Ok("1"),
            "M3 Engram split requires the isolated comparison");
        self.set_engram_split14_for_probe(true)?;
        self.engram_split14_m3_only=true;
        Ok(())
    }
    /// Match qualified EXTEND's decoder floor at the current extend start.
    /// Cached prefixes are not recomputed; only the new rows enter21..39.
    /// This is separate from fresh-history128-row replay and decode/verify.
    pub fn enable_cached_extend_floor(&mut self) -> Result<()> {
        ensure!(
            [6, 128].contains(&self.rows)
                && self.source_phase == Phase::Extend
                && self.history.position == 0
                && self.graphs.iter().all(Option::is_none)
                && self.cached_decoder.is_none()
                && (self.encoder_cache.is_none() || self.rows == 128
                    || std::env::var("CANADA_SCRATCH_TINY_APPEND_PHASE_SCREEN").as_deref()==Ok("1")),
            "cached EXTEND policy requires a fresh admitted extend profile"
        );
        let pair_slots = (COMPRESSED_OFFSET..COMPRESSED_OFFSET + self.context / 2)
            .map(|s| s as i32)
            .collect::<Vec<_>>();
        let single_slots = (COMPRESSED_OFFSET..COMPRESSED_OFFSET + self.context)
            .map(|s| s as i32)
            .collect::<Vec<_>>();
        self.cached_decoder = Some(Step::new(
            &self.cuda,
            &self.source2,
            self.group8.source(),
            self.group14.source(),
            self.late.source(),
            &pair_slots,
            &single_slots,
            self.rows,
            self.context,
            self.early_caches[0].capacity(),
            self.swa_layout,
            Phase::Extend,
        )?);
        Ok(())
    }

    /// Isolated retained-append experiment: reuse the M6 numerical path while
    /// giving decoder layers21..39 the ordinary EXTEND request floor. Encoder
    /// history remains absolute. Graphs containing distinct decoder metadata
    /// pointers are either invalidated or kept in a separate bounded bank.
    pub(crate) fn set_tiny_extend_for_probe(&mut self, enabled:bool, stream:&Stream)->Result<()> {
        ensure!(self.rows==6 || std::env::var("CANADA_SCRATCH_TINY_APPEND_RETAIN_GRAPHS").as_deref()==Ok("1"),
            "M16 append requires the retained metadata/attention graph bank");
        ensure!((std::env::var("CANADA_SCRATCH_TINY_APPEND_SCREEN").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_NATIVE_ROW_TRANSPORT_SCREEN").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_RESIDENT_SMALL_APPEND").as_deref()==Ok("1"))
            && (self.rows==6 || (self.rows==16
                && std::env::var("CANADA_SCRATCH_RESIDENT_MEDIUM_APPEND").as_deref()==Ok("1"))
                ||(self.rows==32&&crate::config::append32_workspace_enabled()))
            && self.source_phase==Phase::Verify && self.pending.is_none()
            && self.lease.is_current() && self.encoder_cache.is_some()
            && self.cuda.owns_stream(stream) && self.stream==stream.raw()as usize,
            "tiny append requires its isolated current M6 owner");
        if enabled==self.cached_decoder.is_some() {return Ok(())}
        stream.synchronize()?;
        if std::env::var("CANADA_SCRATCH_TINY_APPEND_RETAIN_GRAPHS").as_deref()==Ok("1") {
            ensure!(self.encoder_graph.is_none(),"M6 append graph bank has no wide encoder graph");
            if self.tiny_append_graphs.is_none() {
                ensure!(enabled && self.cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+32*1024usize.pow(2),
                    "tiny append graph metadata reserve");
                let pairs=(COMPRESSED_OFFSET..COMPRESSED_OFFSET+self.context/2).map(|x|x as i32).collect::<Vec<_>>();
                let singles=(COMPRESSED_OFFSET..COMPRESSED_OFFSET+self.context).map(|x|x as i32).collect::<Vec<_>>();
                let decoder=Step::new(&self.cuda,&self.source2,self.group8.source(),self.group14.source(),
                    self.late.source(),&pairs,&singles,self.rows,self.context,self.early_caches[0].capacity(),self.swa_layout,Phase::Extend)?;
                self.tiny_append_graphs=Some(TinyAppendGraphs {
                    graphs:(0..graph_slots(self.rows)).map(|_|None).collect(),engram_graphs:(0..graph_slots(self.rows)).map(|_|None).collect(),
                    alternate_index_graphs:(0..graph_slots(self.rows)).map(|_|None).collect(),
                    alternate_index_engram_graphs:(0..graph_slots(self.rows)).map(|_|None).collect(),
                    decoder_graph:None,adaptive_sparse_active:false,decoder:Some(decoder),
                });
            }
            if std::env::var("CANADA_SCRATCH_APPEND_BF16_ONLY").as_deref()==Ok("1") {
                // Both primitive workspaces stay alive for their respective
                // graph banks; switch them with the decoder-window metadata.
                for layer in self.early.iter_mut().chain(self.group8.layers_mut())
                    .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
                    layer.set_append_bf16_for_probe(enabled,stream)?;
                }
            }
            let other=self.tiny_append_graphs.as_mut().unwrap();
            std::mem::swap(&mut self.graphs,&mut other.graphs);
            std::mem::swap(&mut self.engram_graphs,&mut other.engram_graphs);
            std::mem::swap(&mut self.alternate_index_graphs,&mut other.alternate_index_graphs);
            std::mem::swap(&mut self.alternate_index_engram_graphs,&mut other.alternate_index_engram_graphs);
            std::mem::swap(&mut self.decoder_graph,&mut other.decoder_graph);
            std::mem::swap(&mut self.adaptive_sparse_active,&mut other.adaptive_sparse_active);
            std::mem::swap(&mut self.cached_decoder,&mut other.decoder);
            if self.adaptive_sparse_index_layers()>0 {self.late.set_adaptive_sparse6(self.adaptive_sparse_active);}
            ensure!(self.cached_decoder.is_some()==enabled,"tiny append graph/metadata mode");
            return Ok(());
        }
        for g in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) {*g=None;}
        for g in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) {*g=None;}
        self.decoder_graph=None;
        if enabled {
            ensure!(self.cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+32*1024usize.pow(2),
                "tiny append metadata must preserve the operating reserve");
            let pairs=(COMPRESSED_OFFSET..COMPRESSED_OFFSET+self.context/2).map(|x|x as i32).collect::<Vec<_>>();
            let singles=(COMPRESSED_OFFSET..COMPRESSED_OFFSET+self.context).map(|x|x as i32).collect::<Vec<_>>();
            self.cached_decoder=Some(Step::new(&self.cuda,&self.source2,self.group8.source(),
                self.group14.source(),self.late.source(),&pairs,&singles,6,self.context,
                self.early_caches[0].capacity(),self.swa_layout,Phase::Extend)?);
        } else {self.cached_decoder=None;}
        Ok(())
    }

    /// Reclaim plans only. Causal buffers, decoder metadata, logits and
    /// sampler/drafter state keep their storage and contents. Row ages are
    /// shared across phase/index banks; their capture identities remain separate.
    fn bound_execution_graphs(&mut self,live:usize,stream:&Stream)->Result<()> {
        if self.graph_retention.limit()==0 {return Ok(());}
        self.graph_retention.touch(live);
        if self.graphs[live].is_some()||self.engram_graphs[live].is_some(){return Ok(());}
        let mut present=(1..=self.rows).filter(|&i|self.graphs[i].is_some()||self.engram_graphs[i].is_some()).collect::<Vec<_>>();
        if self.graph_retention.victim(&present,live).is_none(){return Ok(());}
        stream.synchronize()?;
        while let Some(victim)=self.graph_retention.victim(&present,live) {
            self.graphs[victim]=None;self.engram_graphs[victim]=None;
            present.retain(|&i|i!=victim);self.graph_evictions+=1;
        }
        Ok(())
    }
    pub(crate) fn graph_retention_report(&self)->serde_json::Value {
        let count=|a:&Vec<Option<Graph>>,b:&Vec<Option<Vec<Graph>>>|a.iter().zip(b).filter(|(a,b)|a.is_some()||b.is_some()).count();
        serde_json::json!({"row_limit":self.graph_retention.limit(),"evictions":self.graph_evictions,
            "current":count(&self.graphs,&self.engram_graphs),"alternate":count(&self.alternate_index_graphs,&self.alternate_index_engram_graphs),
            "other_phase":self.tiny_append_graphs.as_ref().map(|b|[count(&b.graphs,&b.engram_graphs),count(&b.alternate_index_graphs,&b.alternate_index_engram_graphs)])})
    }
    pub(crate) fn set_graph_retention_for_probe(&mut self,limit:usize,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_GRAPH_RETENTION_SCREEN").as_deref()==Ok("1")
            &&limit<=32&&self.pending.is_none()&&self.cuda.owns_stream(stream),"finite graph retention control");
        if self.graph_retention.limit()==limit {return Ok(());}
        stream.synchronize()?;
        // Start each control/candidate arm with the same cold executable bank.
        // Preserve every buffer and the alternate-phase metadata ownership.
        for g in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*g=None;}
        for g in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*g=None;}
        if let Some(b)=&mut self.tiny_append_graphs {
            for g in b.graphs.iter_mut().chain(&mut b.alternate_index_graphs){*g=None;}
            for g in b.engram_graphs.iter_mut().chain(&mut b.alternate_index_engram_graphs){*g=None;}
        }
        self.graph_retention=crate::graph_retention::Policy::new(self.rows,limit);
        Ok(())
    }
    pub(crate) fn append_last_gather_enabled(&self)->bool {self.append_last_gather}
    pub(crate) fn append_encoded_head_enabled(&self)->bool {self.append_encoded_head}
    pub(crate) fn set_append_encoded_head_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_APPEND_HEAD_SCREEN").as_deref()==Ok("1")
            &&matches!(self.rows,16|32)&&self.source_phase==Phase::Verify&&self.append_last_gather
            &&self.pending.is_none()&&self.lease.is_current()&&self.cached_decoder.is_none()
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,
            "finite append head needs idle final-only profile");
        stream.synchronize()?;
        if enabled {self.io.prepare_append_encoded(stream)?;}
        if self.append_encoded_head!=enabled {self.tiny_append_graphs=None;self.append_encoded_head=enabled;}
        Ok(())
    }
    pub(crate) fn append_head_normalized_for_probe(&self)->Result<Vec<u8>> {
        ensure!(std::env::var("CANADA_SCRATCH_APPEND_HEAD_SCREEN").as_deref()==Ok("1")
            &&self.pending.is_none()&&self.last_live>0&&self.lease.is_current(),"finite final-head input capture");
        unsafe{self.cuda.read_device(self.io.normalized().ptr()+((self.last_live-1)*5120*2)as u64,5120*2)}
    }
    pub(crate) fn set_append_last_gather_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_APPEND_LAST_GATHER_SCREEN").as_deref()==Ok("1")
            && [6,16,32].contains(&self.rows) && self.source_phase==Phase::Verify
            && self.pending.is_none() && self.lease.is_current() && self.cached_decoder.is_none()
            && self.cuda.owns_stream(stream) && self.stream==stream.raw()as usize,
            "last-logit append comparison needs an idle ordinary verifier");
        if self.append_last_gather==enabled{return Ok(())}
        stream.synchronize()?;
        // Only the inactive retained-append bank contains this transport tail.
        // Ordinary verification continues publishing every requested row.
        self.tiny_append_graphs=None;
        self.append_last_gather=enabled;
        Ok(())
    }

    /// Isolate reduction arithmetic from row width in the tiny append screen.
    /// Rebinding a collective requires fresh graphs; no normal policy changes.
    pub(crate) fn set_tiny_native_reduce_for_probe(&mut self, enabled:bool, stream:&Stream)->Result<()> {
        ensure!((std::env::var("CANADA_SCRATCH_TINY_APPEND_NATIVE128").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_NATIVE_ROW_TRANSPORT_SCREEN").as_deref()==Ok("1"))
            && self.rows==128 && self.source_phase==Phase::Extend && self.wide.is_none()
            && self.pending.is_none() && self.lease.is_current() && self.cached_decoder.is_some()
            && self.cuda.owns_stream(stream) && self.stream==stream.raw()as usize,
            "tiny native reduction requires its isolated current M128 owner");
        if enabled==self.tiny_native_reduce {return Ok(())}
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) {*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) {*graph=None;}
        self.decoder_graph=None;
        self.tiny_native_reduce=enabled;
        Ok(())
    }

    pub(crate) fn set_dense_live_rows_for_probe(&mut self,dense:bool,experts:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_DENSE_LIVE_ROWS_SCREEN").as_deref()==Ok("1")
            && self.rows==128 && self.source_phase==Phase::Extend && self.wide.is_none()
            && self.pending.is_none() && self.lease.is_current()
            && self.cuda.owns_stream(stream) && self.stream==stream.raw()as usize,
            "dense row comparison requires an idle current M128 EXTEND owner");
        if dense==self.prefill_dense_live_rows && experts==self.prefill_moe_live_rows{return Ok(())}
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.prefill_dense_live_rows=dense;self.prefill_moe_live_rows=experts;
        Ok(())
    }

    pub(crate) fn set_partial_dense_rows_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_DENSE_PARTIAL_SCREEN").as_deref()==Ok("1")
            && self.rows==512 && self.wide.is_some() && self.pending.is_none()
            && self.cuda.owns_stream(stream) && self.stream==stream.raw()as usize,
            "partial dense comparison requires the finite M512 encoder screen");
        // Partial chunks are eager. Full-width encoder captures never use the
        // shorter bound and remain valid across this diagnostic-only switch.
        stream.synchronize()?;self.prefill_partial_dense_rows=enabled;Ok(())
    }
    pub(crate) fn set_partial_moe_rows_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MOE_PARTIAL_SCREEN").as_deref()==Ok("1")
            && self.rows==512 && self.wide.is_some() && self.pending.is_none()
            && self.cuda.owns_stream(stream) && self.stream==stream.raw()as usize,
            "partial expert comparison requires the finite M512 encoder screen");
        // Only partial eager encoder chunks are affected. Full-width captured
        // graphs and their allocation/collective identities are unchanged.
        stream.synchronize()?;self.prefill_partial_moe_rows=enabled;Ok(())
    }
    pub(crate) fn set_short_wide_rows_for_probe(&mut self,dense:bool,mhc:bool,norm:bool,stream:&Stream)->Result<(bool,bool,bool)> {
        ensure!(std::env::var("CANADA_SCRATCH_SHORT_WIDE_ROWS_SCREEN").as_deref()==Ok("1")
            &&[512,1024,2048,8192].contains(&self.rows)&&self.wide.is_some()&&self.pending.is_none()
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,
            "short wide-row comparison requires an isolated encoder profile");
        // Affects only eager partial encoder chunks. Full-width captures and
        // the separate decoder-tail profile retain their complete row extent.
        stream.synchronize()?;
        let previous=(self.prefill_partial2048_dense,self.prefill_wide_mhc_rows,self.prefill_wide_mhc_norm);
        self.prefill_partial2048_dense=dense;self.prefill_wide_mhc_rows=mhc;self.prefill_wide_mhc_norm=norm;Ok(previous)
    }
    /// Plan-comparison screen only: bound partial 4,096/8,192-row chunks to live rows.
    pub(crate) fn set_partial_wide_rows_for_probe(&mut self,mask:u8,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_FLEX_WIDTH_SCREEN").as_deref()==Ok("1")
            && mask<=3 && self.wide.is_some() && self.pending.is_none()
            && self.cuda.owns_stream(stream) && self.stream==stream.raw()as usize,
            "partial wide rows require the finite plan-comparison screen");
        // Only eager partial chunks read this; full-width captures are unaffected.
        stream.synchronize()?;self.prefill_partial_wide=mask;Ok(())
    }
    pub(crate) fn set_partial1024_rows_for_probe(&mut self,mode:u8,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_PARTIAL1024_SCREEN").as_deref()==Ok("1")
            && mode<=2 && self.rows==1024 && self.wide.is_some() && self.pending.is_none()
            && self.cuda.owns_stream(stream) && self.stream==stream.raw()as usize,
            "partial1024 comparison requires the finite M1024 encoder screen");
        stream.synchronize()?;self.prefill_partial1024_rows=mode;Ok(())
    }
    pub(crate) fn set_prefill_mhc_rows_for_probe(&mut self,enabled:bool,normalize:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_PREFILL_MHC_ROWS_SCREEN").as_deref()==Ok("1")
            &&[128,512].contains(&self.rows)&&self.source_phase==Phase::Extend&&self.pending.is_none()
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,
            "prefill mHC comparison requires the finite M128/M512 screen");
        ensure!(!normalize||enabled,"prefill normalization requires the admitted row bound");
        if enabled==self.prefill_mhc_live_rows&&normalize==self.prefill_mhc_live_norm{return Ok(())}
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.decoder_graph=None;self.prefill_mhc_live_rows=enabled;self.prefill_mhc_live_norm=normalize;Ok(())
    }
    pub(crate) fn set_tiny_prefix_transport_for_probe(&mut self, sum:bool, gather:bool, stream:&Stream)->Result<()> {
        ensure!((std::env::var("CANADA_SCRATCH_TINY_APPEND_PREFIX_SUM").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_NATIVE_ROW_TRANSPORT_SCREEN").as_deref()==Ok("1"))
            && ((self.rows==128 && self.source_phase==Phase::Extend)
                || ([6,16].contains(&self.rows) && self.source_phase==Phase::Verify))
            && self.wide.is_none() && self.pending.is_none() && self.lease.is_current()
            && self.cuda.owns_stream(stream) && self.stream==stream.raw()as usize,
            "tiny prefix sum requires an isolated current target");
        ensure!(!gather || std::env::var("CANADA_SCRATCH_TINY_APPEND_PREFIX_GATHER").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_NATIVE_ROW_TRANSPORT_SCREEN").as_deref()==Ok("1"),"prefix gather admission");
        if sum==self.tiny_prefix_sum && gather==self.tiny_prefix_gather {return Ok(())}
        stream.synchronize()?;
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) {*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) {*graph=None;}
        self.decoder_graph=None;
        if let Some(bank)=&mut self.tiny_append_graphs {
            for graph in bank.graphs.iter_mut().chain(&mut bank.alternate_index_graphs) {*graph=None;}
            for graph in bank.engram_graphs.iter_mut().chain(&mut bank.alternate_index_engram_graphs) {*graph=None;}
            bank.decoder_graph=None;
        }
        self.tiny_prefix_sum=sum;self.tiny_prefix_gather=gather;
        Ok(())
    }

    /// Diagnostic replay must use the same incoming compressor rings on every
    /// launch. These private copies are not session parking or rollback state.
    pub fn enable_repeatable_frontier_measurement(&mut self) -> Result<()> {
        ensure!(
            self.lease.is_current()
                && self.history.position == 0
                && self.graphs.iter().all(Option::is_none),
            "replay measurement policy needs fresh state"
        );
        self.repeatable_frontier_measurement = true;
        Ok(())
    }
    fn new_carry_snapshot(&self) -> Result<crate::carry_state::Snapshot> {
        let rings = [
            &self.source2,
            self.group8.source(),
            self.group14.source(),
            self.late.source(),
        ]
        .into_iter()
        .filter_map(|s| s.source().ring())
        .collect();
        crate::carry_state::Snapshot::with_requests(&self.cuda, rings, crate::config::session_slots()?)
    }

    /// Opt-in integration experiment until changed-metadata replay is proven.
    /// Production may select this only after qualification for its row profile.
    pub fn set_graph_replay(&mut self, enabled: bool) -> Result<()> {
        ensure!(
            self.lease.is_current() && self.history.position == 0,
            "execution mode changes require healthy fresh causal state"
        );
        self.graph_replay = enabled;
        Ok(())
    }
    pub fn set_prefill_last_head(&mut self, enabled: bool) -> Result<()> {
        ensure!(
            self.history.position == 0 && self.graphs.iter().all(Option::is_none),
            "output mode changes require a fresh target owner"
        );
        self.prefill_last_head = enabled;
        Ok(())
    }
    pub(crate) fn prefill_encoded_head_enabled(&self)->bool {self.io.prefill_encoded_enabled()}
    /// Startup selection, or an isolated same-allocation comparison. Every
    /// graph that can contain the final head is invalidated before switching.
    pub(crate) fn set_prefill_encoded_head(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(self.rows==128&&self.source_phase==Phase::Extend&&self.prefill_last_head
            &&self.pending.is_none()&&self.lease.shared_idle()
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize
            &&(self.position()==0||std::env::var("CANADA_SCRATCH_PREFILL_HEAD_SCREEN").as_deref()==Ok("1")),
            "prefill head selection requires startup or finite idle comparison");
        if self.io.prefill_encoded_enabled()==enabled{return Ok(());}
        stream.synchronize()?;
        for g in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs) {*g=None;}
        for g in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs) {*g=None;}
        self.decoder_graph=None;self.tiny_append_graphs=None;
        self.io.set_prefill_encoded(enabled,stream)
    }
    pub(crate) fn prefill_head_input_for_probe(&self)->Result<Vec<u8>> {
        ensure!(std::env::var("CANADA_SCRATCH_PREFILL_HEAD_SCREEN").as_deref()==Ok("1")
            &&self.rows==128&&self.pending.is_none()&&self.last_live>0&&self.lease.is_current(),
            "finite prefill head input capture");
        unsafe {self.cuda.read_device(self.io.prefill_normalized_ptr(self.last_live)?,5120*2)}
    }
    pub(crate) fn prefill_row_head_enabled(&self)->bool {self.io.prefill_row_pruned()}
    pub(crate) fn set_prefill_row_head(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(self.rows==128&&self.source_phase==Phase::Extend&&self.prefill_last_head
            &&self.pending.is_none()&&self.lease.shared_idle()
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize
            &&(self.position()==0||std::env::var("CANADA_SCRATCH_PREFILL_HEAD_SCREEN").as_deref()==Ok("1")),
            "last-row normalization requires startup or finite idle comparison");
        if self.io.prefill_row_pruned()==enabled{return Ok(());}
        stream.synchronize()?;
        for g in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*g=None;}
        for g in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*g=None;}
        self.decoder_graph=None;self.tiny_append_graphs=None;
        self.io.set_prefill_row_pruned(enabled,stream)
    }
    pub(crate) fn prefill_sparse_publish_enabled(&self)->bool {self.io.sparse_publish_enabled()}
    pub(crate) fn set_prefill_sparse_publish(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(self.rows==128&&self.source_phase==Phase::Extend&&self.prefill_last_head
            &&self.pending.is_none()&&self.lease.shared_idle()
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize
            &&(self.position()==0||std::env::var("CANADA_SCRATCH_PREFILL_HEAD_SCREEN").as_deref()==Ok("1")),
            "sparse logit publication requires startup or finite idle comparison");
        if self.io.sparse_publish_enabled()==enabled{return Ok(());}
        stream.synchronize()?;
        for g in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*g=None;}
        for g in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*g=None;}
        self.decoder_graph=None;self.tiny_append_graphs=None;
        self.io.set_sparse_publish(enabled,stream)
    }

    pub(crate) fn compact_history_sampling_requested()->Result<bool> {
        match std::env::var("CANADA_SCRATCH_COMPACT_HISTORY_SAMPLING") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid compact history sampling policy");Ok(value=="1")},
            Err(std::env::VarError::NotPresent)=>Ok(false),Err(error)=>Err(error.into()),
        }
    }
    pub(crate) fn enable_compact_history_sampling(&mut self,core:&[u8],stream:&Stream)->Result<()> {
        ensure!(Self::compact_history_sampling_requested()?&&[16,32].contains(&self.rows)
            &&self.source_phase==Phase::Verify&&self.history.position==0&&self.lease.is_current()
            &&self.pending.is_none()&&self.graph_replay&&!self.prefill_last_head
            &&self.cached_decoder.is_none()&&self.tiny_append_graphs.is_none()
            &&self.graphs.iter().chain(&self.alternate_index_graphs).all(Option::is_none)
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,
            "compact history sampling requires a fresh copy verifier");
        ensure!(self.cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+64*1024usize.pow(2),
            "compact sampler preserves physical reserve");
        self.compact_categorical=Some(crate::sharded_categorical_probe::CompactSampler::new(
            &self.cuda,core,self.rows,self.rank)?);
        self.compact_sampling_enabled=true;Ok(())
    }
    pub(crate) fn compact_history_sampling_enabled(&self)->bool {self.compact_sampling_enabled}

    /// Finite transport screen. Clear the affected captures when changing the
    /// head's communication topology; accepted causal storage is untouched.
    pub(crate) fn set_compact_sampling_for_probe(&mut self,enabled:bool,core:&[u8],stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_SHARDED_CATEGORICAL_MODEL_SCREEN").as_deref()==Ok("1")
            &&[16,32].contains(&self.rows)&&self.source_phase==Phase::Verify
            &&self.pending.is_none()&&self.lease.is_current()&&self.graph_replay
            &&!self.prefill_last_head&&self.cached_decoder.is_none()
            &&self.tiny_append_graphs.as_ref().is_none_or(|bank|bank.decoder.is_some())
            &&self.encoder_graph.is_none()&&self.decoder_graph.is_none()
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,
            "finite compact sampler requires an idle full-row copy profile");
        if enabled==self.compact_sampling_enabled{return Ok(())}
        stream.synchronize()?;
        if enabled&&self.compact_categorical.is_none() {
            ensure!(self.cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+64*1024usize.pow(2),
                "compact sampler preserves physical reserve");
            self.compact_categorical=Some(crate::sharded_categorical_probe::CompactSampler::new(
                &self.cuda,core,self.rows,self.rank)?);
        }
        for graph in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*graph=None;}
        for graph in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*graph=None;}
        self.compact_sampling_enabled=enabled;Ok(())
    }
    fn compact_sampling_active(&self)->bool {
        // The alternate M16 EXTEND bank still publishes full prompt logits.
        // Its graph topology remains independent of ordinary copy verification.
        // A region step samples every session from full logits (its drafted rows need them).
        ((self.compact_sampling_enabled&&!self.region_step)||(self.local_target_enabled&&self.rows==6
            &&self.source_phase==Phase::Verify&&!self.prefill_last_head))&&self.cached_decoder.is_none()
    }
    pub(crate) fn local_target_enabled(&self)->bool {self.local_target_enabled}
    pub(crate) fn set_local_target_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_LOCAL_TARGET_MODEL_SCREEN").as_deref()==Ok("1")
            &&self.rows==6&&self.source_phase==Phase::Verify&&self.pending.is_none()
            &&self.lease.is_current()&&self.graph_replay&&!self.prefill_last_head
            &&self.cached_decoder.is_none()&&self.encoder_graph.is_none()&&self.decoder_graph.is_none()
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,
            "finite local target requires idle narrow verification");
        if enabled==self.local_target_enabled{return Ok(())}
        stream.synchronize()?;
        for g in self.graphs.iter_mut().chain(&mut self.alternate_index_graphs){*g=None;}
        for g in self.engram_graphs.iter_mut().chain(&mut self.alternate_index_engram_graphs){*g=None;}
        for graph in &mut self.local_target_gather {*graph=None;}
        self.local_target_enabled=enabled;Ok(())
    }

    /// Start a new sequence without reloading immutable weights or discarding
    /// immutable Engram row-cache entries. Complete causal KV, compression
    /// rings, index keys and hash history are cleared together. Captured
    /// pointers remain valid. This is single-owner reset, not branch/COW or
    /// lossless parking; a poisoned owner must still be dropped and reloaded.
    pub fn reset(&mut self, lineage: u64, comm: &Roce, stream: &Stream) -> Result<()> {
        ensure!(
            self.lease.is_current() && lineage > self.history.lineage,
            "target reset needs a healthy owner and newer lineage"
        );
        ensure!(
            comm.rank() == self.rank
                && self.cuda.owns_stream(stream)
                && stream.raw() as usize == self.stream
                && (self.comm_instance.is_none() || self.comm_instance == Some(comm.instance())),
            "target reset transport/stream owner changed"
        );
        // With serving regions a reset clears region 0 only (carry slot 0; the caches are kept and
        // overwritten by the next prefill), so the whole-cache clear is not available.
        let keep_caches = crate::config::reset_keeps_caches()?;
        ensure!(keep_caches || crate::config::batched_regions()? == 1,
            "serving regions need the kept-cache reset (CANADA_SCRATCH_RESET_KEEP_CACHES=1)");
        stream.synchronize()?;
        comm.health()?;
        self.lease.begin()?;
        self.invalidate_gather_caches();
        if !keep_caches {
            for cache in &mut self.early_caches {
                cache.clear(&self.cuda, stream)?;
            }
        }
        self.source2.clear_state(stream, keep_caches)?;
        self.group8.clear_state(&self.cuda, stream, keep_caches)?;
        self.group14.clear_state(&self.cuda, stream, keep_caches)?;
        self.late.clear_state(&self.cuda, stream, keep_caches)?;
        if let Some(cache) = &self.encoder_cache {
            cache.clear(stream)?;
        }
        self.io.update_tokens(&mut self.tokens, self.history.position as usize,&vec![2; self.rows], 0)?;
        self.step.update(&Plan::append_with_layout(
            0,
            if self.source_phase == Phase::Decode {
                1
            } else {
                self.rows
            },
            self.rows,
            self.context,
            self.swa_layout,
        )?, stream)?;
        if let Some(step) = &mut self.cached_decoder {
            step.update(&Plan::append_with_layout(
                0,
                self.rows,
                self.rows,
                self.context,
                self.swa_layout,
            )?, stream)?;
        }
        stream.synchronize()?;
        comm.health()?;
        self.history = History::new(lineage);
        self.last_live = 0;
        self.last_append_metrics = None;
        self.carry_saved_start = None;
        self.lease.finish_write()?;
        Ok(())
    }

    /// Append already accepted input tokens. Any execution failure poisons the
    /// shared owner; only the explicit completed verification can commit/abort.
    /// Install immutable image spans for the next accepted prompt. The IO
    /// holder is shared by M6/M16/M128/wide profiles; captured row buffers stay
    /// profile-local and are refreshed at each execution frontier.
    pub fn set_prompt_media(&mut self, media: Option<Rc<crate::prompt_media::Bindings>>, stream: &Stream) -> Result<()> {
        ensure!(self.pending.is_none() && self.lease.is_current()
            && self.cuda.owns_stream(stream) && self.stream == stream.raw() as usize,
            "media switch requires the idle current target owner");
        ensure!(media.as_ref().is_none_or(|m| m.identity().placements().iter()
            .all(|p| p.start + p.tokens <= self.context)), "media exceeds target context");
        self.io.set_prompt_media(media, stream)
    }
    /// A stale profile can release its row-address owner only when the shared
    /// writer is idle. This changes no causal state or captured allocation.
    pub(crate) fn clear_prompt_media_rows(&mut self, stream: &Stream) -> Result<()> {
        ensure!(self.pending.is_none() && self.lease.shared_idle()
            && self.cuda.owns_stream(stream) && self.stream == stream.raw() as usize,
            "media row release requires an idle shared owner");
        stream.synchronize()?;
        self.tokens.clear_images()
    }
    pub fn append(&mut self, ids: &[u32], comm: &Roce, stream: &Stream) -> Result<()> {
        self.execute_rows(ids, comm, stream, false, false)
    }
    /// Evaluate a speculative block without advancing accepted history. This
    /// holds the shared writer busy until finish_verification; another profile
    /// cannot adopt it or publish speculative state. Drop/error fails closed.
    pub fn begin_verification(&mut self, ids: &[u32], comm: &Roce, stream: &Stream) -> Result<()> {
        self.begin_verification_mode(ids,comm,stream,false)
    }
    pub(crate) fn begin_local_verification(&mut self,ids:&[u32],comm:&Roce,stream:&Stream)->Result<()> {
        ensure!(self.local_target_enabled&&self.rows==6&&self.cached_decoder.is_none()
            &&(2..=6).contains(&ids.len()),"local verification eligibility");
        self.begin_verification_mode(ids,comm,stream,true)
    }
    fn begin_verification_mode(&mut self,ids:&[u32],comm:&Roce,stream:&Stream,local_only:bool)->Result<()> {
        ensure!(
            self.source_phase == Phase::Verify
                && crate::config::verification_rows(self.rows)
                && !self.prefill_last_head
                && (self.cached_decoder.is_none() || ((self.rows==6 || (self.rows==16
                    && std::env::var("CANADA_SCRATCH_RESIDENT_MEDIUM_APPEND").as_deref()==Ok("1"))
                    ||(self.rows==32&&crate::config::append32_workspace_enabled()))
                    && (std::env::var("CANADA_SCRATCH_TINY_APPEND_SCREEN").as_deref()==Ok("1")
                        || std::env::var("CANADA_SCRATCH_NATIVE_ROW_TRANSPORT_SCREEN").as_deref()==Ok("1")
                        || std::env::var("CANADA_SCRATCH_RESIDENT_SMALL_APPEND").as_deref()==Ok("1")))),
            "verification execution policy"
        );
        self.execute_rows(ids, comm, stream, true, local_only)
    }
    fn execute_rows(
        &mut self,
        ids: &[u32],
        comm: &Roce,
        stream: &Stream,
        verification: bool,
        local_only: bool,
    ) -> Result<()> {
        ensure!(
            self.pending.is_none() && self.lease.is_current(),
            "stale or poisoned target profile; adopt current history or reload"
        );
        ensure!(
            self.wide.is_none(),
            "the wide encoder profile executes only through prefill_wide"
        );
        // A serving load with the layer clock armed marks the six-row verifier's forwards only (no effect otherwise).
        let _clock_live = crate::layer_clock::live_scope(self.rows == 6 && self.source_phase == Phase::Verify && verification);
        ensure!(
            comm.rank() == self.rank
                && self.cuda.owns_stream(stream)
                && stream.raw() as usize == self.stream,
            "target rank/ordered stream mismatch"
        );
        ensure!(
            self.comm_instance.is_none() || self.comm_instance == Some(comm.instance()),
            "target transport owner changed; captured collective pointers are not reusable"
        );
        ensure!(
            ids.iter().all(|&id| id < 129280),
            "target token outside vocabulary"
        );
        ensure!(
            self.source_phase != Phase::Decode || ids.len() == 1,
            "single-stream decode requires exactly one accepted token"
        );
        ensure!(
            crate::config::verification_rows(self.rows) || self.history.position == 0 || self.cached_decoder.is_some(),
            "M128 retained input requires explicit cached EXTEND policy"
        );
        ensure!(
            self.encoder_cache.is_none() || self.rows == 128 || self.source_phase == Phase::Verify
                || (self.rows==6 && self.source_phase==Phase::Extend
                    && std::env::var("CANADA_SCRATCH_TINY_APPEND_PHASE_SCREEN").as_deref()==Ok("1")),
            "cached EXTEND increments require decoder-tail replay"
        );
        self.last_live = 0;
        self.last_append_metrics = None;
        self.carry_saved_start = None;
        let begin = Instant::now();
        // A step with batched rows (region mode) runs without the Engram overlap unless
        // CANADA_SCRATCH_BATCHED_ENGRAM_OVERLAP=1: then its rows are looked up while layer 0 runs,
        // as in ordinary steps, with split graphs kept per graph slot (an ordinary step's slot is
        // its live count).
        let overlap = self.engram_overlap && (self.batched_rows.is_none() || crate::config::batched_engram_overlap());
        // Live count is agreed by all ranks and is already a graph-cache key.
        // Keep both projections bound to their own stream so changing widths
        // does not destroy a warmed graph or rebind mutable scratch.
        if self.engram_projection_ahead {
            self.engram_projection_active=overlap&&self.graph_replay
                &&(!self.engram_projection_m3_only||ids.len()==3);
            self.group14.select_engram_projection_ahead(self.engram_projection_active)?;
            // A projection that already runs ahead has read its weights before layer 13's FFN sum.
            if let Some((_, gate14)) = &self.engram_lead { gate14.set(self.engram_lead_on && !self.engram_projection_active); }
        }
        self.select_index_graph_bank(stream)?;
        // A batched step of several sessions replays graphs from its own slots
        // (graph_slot) when CANADA_SCRATCH_BATCHED_GRAPHS=1; otherwise it runs eagerly.
        let sessions = self.batched_rows.as_ref().map_or(1, |p| p.len());
        // Region mode: any step with batched rows, one session included, has rows whose keys
        // may start at a region base. It runs the dense index scorers and gathers full logits,
        // from its own graph slots; the Engram overlap's per-live-count graphs are not used.
        let region = self.batched_rows.is_some();
        let padded_routes = region && sessions >= 2 && self.padding_routes;
        ensure!(sessions <= MAX_BATCHED_SESSIONS && (!region || !overlap || crate::config::batched_engram_overlap()),
            "batched sessions/Engram overlap");
        self.set_region_step(region)?;
        let graph_replay = self.graph_replay && (sessions == 1 || crate::config::batched_graphs_requested());
        if graph_replay && !region {self.bound_execution_graphs(ids.len(),stream)?;}
        let mut marks = vec![("begin", begin)];
        let mut captured_new_graph = false;
        let mut split_engram14=self.engram_split14 && (!self.engram_split14_m3_only || ids.len()==3);
        let retain_warm_prefill = self.prefill_capture_once && !verification
            && self.rows == 128 && self.source_phase == Phase::Extend;
        let retain_warm_result = retain_warm_prefill || (self.verification_capture_once
            && verification && [6,16].contains(&self.rows) && self.source_phase == Phase::Verify);
        let batched = self.batched_rows.take();
        let p = if let Some(parts) = &batched {
            // A shared encoder-tail cache belongs to the owner's own history; batched
            // rows never publish into it or advance it.
            ensure!(crate::config::batched_rows_admitted()
                && self.cached_decoder.is_none()
                && parts.iter().map(|x| x.2).sum::<usize>() == ids.len(),
                "batched screen rows/owner");
            Plan::batched(&parts.iter().map(|(b, h, live)| (*b, h.position, *live)).collect::<Vec<_>>(),
                self.rows, self.context, self.swa_layout)?
        } else {
            Plan::append_with_layout(
                self.history.position,
                ids.len(),
                self.rows,
                self.context,
                self.swa_layout,
            )?
        };
        // The rows a batched step executes are its blocks through the last part's
        // live rows: the concatenated ids are placed at each part's block, with
        // padding token 2 in between (no cache writes; see Plan::batched).
        let concatenated = ids;
        let expanded = batched.as_ref().filter(|parts| parts.len() > 1).map(|parts| {
            let lives = parts.iter().map(|x| x.2).collect::<Vec<_>>();
            let mut v = vec![2u32; Plan::batched_extent(&lives, self.rows)];
            let mut at = 0;
            for (k, live) in lives.iter().enumerate() {
                v[k * p.block..k * p.block + live].copy_from_slice(&concatenated[at..at + live]);
                at += live;
            }
            v
        });
        let ids: &[u32] = expanded.as_deref().unwrap_or(concatenated);
        let tight = if sessions >= 2 && self.rows <= 32 && crate::config::batched_tight_blocks() { p.block } else { 0 };
        let slot = graph_slot(self.rows, sessions, ids.len(), region, padded_routes, tight);
        // config::index_short_rows: an ordinary step whose rows all have at most 512 index keys enqueues no index
        // scoring (query_index::Branch::short_step) and replays captures of its own slots.
        let short_index = crate::config::index_short_rows() && self.short_index_allowed && self.rows == 6
            && sessions == 1 && !region && self.source_phase == Phase::Verify && !self.adaptive_sparse_active
            && p.pair_lengths.iter().chain(&p.single_lengths).all(|&l| (0..=512).contains(&l));
        let slot = if short_index { short_index_slot(self.rows, ids.len()) } else { slot };
        let _short_index_rows = if short_index {
            if self.short_index_flags.is_empty() {
                self.short_index_flags = self.index_queries().into_iter().flatten().map(|q| q.short_rows_flag()).collect();
            }
            for flag in &self.short_index_flags { flag.set(true); }
            self.short_index_steps += 1;
            Some(ShortIndexRows(self.short_index_flags.clone()))
        } else { None };
        let mut padded = vec![2u32; self.rows];
        padded[..ids.len()].copy_from_slice(ids);
        // A batched step hashes each session's rows against its own history;
        // padding rows repeat a valid row, as prepare_padded does for one.
        let mut part_batches = Vec::new();
        let batch = if let Some(parts) = &batched {
            let mut at = 0;
            let mut rows = Vec::with_capacity(self.rows);
            for (k, (_, h, live)) in parts.iter().enumerate() {
                if let Some(&last) = rows.last() { rows.resize(k * p.block, last); }
                let mut part = vec![2u32; *live];
                part.copy_from_slice(&concatenated[at..at + live]);
                let b = h.prepare_padded(&part, *live, &self.hashes)?;
                rows.extend_from_slice(&b.rows[..*live]);
                part_batches.push(b);
                at += live;
            }
            let last = *rows.last().unwrap();
            rows.resize(self.rows, last);
            let mut combined = self.history.prepare_padded(&padded, ids.len(), &self.hashes)?;
            combined.rows = rows;
            combined
        } else {
            self.history.prepare_padded(&padded, ids.len(), &self.hashes)?
        };
        marks.push(("history_hash", Instant::now()));
        let mut engram_gap_recorded = false;
        ensure!(!overlap || (verification && self.graph_replay && [6,16,32].contains(&self.rows)),
            "experimental Engram overlap is verification-only");
        let deferred_rows = overlap && self.engram_graphs[slot].is_some();
        if self.engram_split14_adaptive {
            // Both capture paths execute identical collectives in identical
            // stream order. Ranks may choose different graph boundaries based
            // on their own immutable shard's cache. Capture both paths together
            // before allowing any rank-local choice: a miss must never trigger
            // an extra eager warmup or a new collective on just one rank.
            split_engram14 = deferred_rows && self.group14.engram_store()?
                .misses_any(batch.rows.iter().flat_map(|row| row[1]))?;
            marks.push(("engram_cache_hint", Instant::now()));
        }
        if !deferred_rows {
            self.prepare_engram_rows(&batch.rows)?;
            marks.push(("engram_rows", Instant::now()));
        } else if crate::config::engram_early_submit() && self.rows == 6 && self.source_phase == Phase::Verify
            && self.batched_rows.is_none() && self.paired_engram_io_enabled() {
            // The missing rows' reads go out now; the lookup behind layer 0 publishes them (read_ahead_submit).
            self.submit_engram_rows(&batch.rows, split_engram14)?;
            marks.push(("engram_submit", Instant::now()));
        }
        let token_ids: Vec<_> = padded.iter().copied().map(i64::from).collect();
        if region && verification {
            // Each region supplies generated anchors and proposals. A concurrent prefill
            // can own the shared prompt's images at these same numerical positions.
            // Updating ordinary token rows also clears image pointers from prior use.
            self.tokens.update(&token_ids, ids.len())?;
        } else {
            self.io.update_tokens(&mut self.tokens, self.history.position as usize, &token_ids, ids.len())?;
        }
        if padded_routes {
            // Rows between a session's live rows and the next block take that session's last live
            // row's experts with zero weights: they read no expert of their own.
            let parts = batched.as_ref().ok_or_else(|| anyhow::anyhow!("padding routes without batched parts"))?;
            let mut sources = vec![-1i32; self.rows];
            for (k, (_, _, live)) in parts.iter().enumerate() {
                let start = k * p.block;
                for source in sources.iter_mut().take(((k + 1) * p.block).min(ids.len())).skip(start + live) {
                    *source = (start + live - 1) as i32;
                }
            }
            self.tokens.set_padding_sources(&sources)?;
        }
        marks.push(("token_upload", Instant::now()));
        let gathers = if region && crate::config::region_gather_cache() {
            let parts = batched.as_ref().ok_or_else(|| anyhow::anyhow!("region step without batched parts"))?;
            Some(self.region_gather_starts(&p, parts)?)
        } else { None };
        self.step.update_with(&p, gathers.as_ref().map(|(a, b)| (a.as_slice(), b.as_slice())), stream)?;
        // The two-row case benefits from both smaller launches on the held verifier.
        // Wider ordinary graphs, shared-region graphs and append use the usual kernels.
        if verification && self.source_phase == Phase::Verify && self.rows == 6 && !region && ids.len() == 2 {
            self.tokens.use_live_gate_rows();
            for metadata in [&mut self.step.m2, &mut self.step.m8, &mut self.step.m14] {
                metadata.select_two_row_solo_index();
            }
        }
        if let Some(step) = &mut self.cached_decoder {
            let mut decoder = Plan::append_with_layout(
                self.history.position,
                ids.len(),
                self.rows,
                self.context,
                self.swa_layout,
            )?;
            decoder.floor_swa_at(self.history.position as usize)?;
            step.update(&decoder, stream)?;
        }
        marks.push(("causal_metadata", Instant::now()));
        self.comm_instance = Some(comm.instance());
        // Hold the M6 arithmetic schedule constant when isolating EXTEND
        // compression from Verify. Ordinary prefill policy is unchanged.
        let tiny_phase_native=self.rows==6 && self.source_phase==Phase::Extend
            && std::env::var("CANADA_SCRATCH_TINY_APPEND_PHASE_SCREEN").as_deref()==Ok("1");
        // M128 short extensions are latency-bound. Keep this choice a function
        // of immutable launch policy and graph live-row key, including fresh
        // short prompts, so a cached graph can never retain the wrong reduction.
        // Full encoder chunks and decoder-tail replay keep their existing path.
        let medium_native=self.rows==128 && self.source_phase==Phase::Extend && ids.len()<=64
            && std::env::var("CANADA_SCRATCH_MEDIUM_NATIVE_TRANSPORT").as_deref()==Ok("1");
        // Graphs are keyed by live rows, not position. Use the same bound for
        // fresh and retained prompts so reset/reuse cannot change the policy
        // of an existing capture. Decoder-tail replay remains outside scope.
        let bounded_prefill=self.rows==128 && self.source_phase==Phase::Extend
            && ((17..=64).contains(&ids.len())||(self.prefill_upper_rows&&(65..128).contains(&ids.len())));
        let _dense_rows=crate::fp8_linear::LivePrefillRowsScope::new(
            self.prefill_dense_live_rows && bounded_prefill,self.prefill_moe_live_rows && bounded_prefill,
            ids.len(),self.rows,stream)?;
        let _woa_rows=crate::woa_medium::LiveRowsScope::new(
            self.prefill_woa_live&&self.rows==128&&self.source_phase==Phase::Extend
                // Leave near-full tiles alone:127 rows was flat in the native
                // screen. Retain at least16 avoided rows to pay for a new plan.
                &&(33..=112).contains(&ids.len()),ids.len(),self.rows,stream)?;
        let _mhc_rows=crate::mhc::ActiveRowsScope::new(
            (ids.len()<=self.verification_mhc_live_rows&&self.rows==6&&self.source_phase==Phase::Verify)
                ||self.prefill_mhc_live_rows&&bounded_prefill,
            ids.len(),self.rows,stream,self.prefill_mhc_live_norm&&bounded_prefill)?;
        comm.select_decode_reduction(self.source_phase != Phase::Extend || tiny_phase_native || self.tiny_native_reduce || medium_native, stream)?;
        comm.select_prefill_rows(ids.len(), self.rows, stream)?;
        let _native_rows=comm.native_row_scope(self.tiny_prefix_sum||medium_native,self.tiny_prefix_gather||medium_native,ids.len(),self.rows,stream)?;
        // Never reuse a capture's NCCL count for a different accepted size.
        // Metadata pointers are stable, and the transport/context/stream
        // identity is checked above, so a prior matching variant is reusable.
        marks.push(("reduction_policy", Instant::now()));
        // Host metadata uses legacy-default-stream copies. The model stream
        // is nonblocking; a pageable HtoD return alone is not a DMA dependency.
        // Queue the event outside capture before reading the new row state.
        stream.join_default()?;
        marks.push(("upload_order", Instant::now()));
        self.prepare_gather_caches(stream)?;
        self.lease.begin()?;
        if verification
            || self.repeatable_frontier_measurement
            || (graph_replay && self.graphs[slot].is_none())
        {
            if self.carry_snapshot.is_none() {
                self.carry_snapshot = Some(self.new_carry_snapshot()?);
            }
            unsafe {
                self.carry_snapshot.as_mut().unwrap().save(stream)?;
            }
            self.carry_saved_start = Some(self.history.position);
            marks.push(("carry_snapshot", Instant::now()));
        }
        if graph_replay {
            let missing = if overlap { self.engram_graphs[slot].is_none() }
                else { self.graphs[slot].is_none() };
            if missing {
                // Warm AOT/vendor paths before capture. Six-row frontiers fit
                // within ring8. M128 is restricted above to fresh pos0 full
                // chunks, whose odd rows consume only in-chunk partners.
                // This is not a retained long-chunk replay guarantee.
                captured_new_graph = true;
                self.enqueue(comm, stream)?;
                stream.synchronize()?;
                comm.health()?;
                marks.push(("graph_warmup", Instant::now()));
                // The control restores incoming carry before replaying. The
                // candidate instead retains the successful eager output:
                // recording a graph performs no model execution. Verification
                // keeps the incoming snapshot for partial acceptance/rollback.
                if !retain_warm_result {
                    unsafe {
                        self.carry_snapshot.as_ref().unwrap().restore(stream)?;
                    }
                    stream.synchronize()?;
                }
                let mut rules = Vec::new();
                if let Some(to) = self.moe_front_grid {
                    rules.push(crate::cuda::Regrid { name: "MoEDynamicKernelSilu", from: 48, to, grow: false, carveout: None });
                }
                // A screen's setting, else the owner's configured grids (verify owners of 6 and 16 rows).
                let phase_grid = self.moe_phase_grid.or_else(|| (self.source_phase == Phase::Verify)
                    .then(|| crate::config::moe_phase_grid(self.rows)).flatten());
                if let Some((up, down, carveout)) = phase_grid {
                    rules.push(crate::cuda::Regrid { name: "W4A8MaterializedPhase1Kernel", from: 96, to: up, grow: true, carveout });
                    rules.push(crate::cuda::Regrid { name: "W4A8MaterializedPhase2Kernel", from: 96, to: down, grow: true, carveout });
                }
                let regrid = if rules.is_empty() { None } else { Some(crate::cuda::graph_regrid_scope(&rules)?) };
                if overlap {
                    let initial = stream.capture(|| self.enqueue_initial(comm, stream))?;
                    let parts=if self.engram_split14_adaptive {
                        let combined=stream.capture(||self.enqueue_after_initial(comm,stream))?;
                        let middle=stream.capture(||self.enqueue_before_engram14(comm,stream))?;
                        let remaining=stream.capture(||self.enqueue_after_engram14(comm,stream))?;
                        vec![initial,combined,middle,remaining]
                    }else if split_engram14 {
                        let middle=stream.capture(||self.enqueue_before_engram14(comm,stream))?;
                        let remaining=stream.capture(||self.enqueue_after_engram14(comm,stream))?;
                        vec![initial,middle,remaining]
                    }else{
                        let remaining=stream.capture(||self.enqueue_after_initial(comm,stream))?;
                        vec![initial,remaining]
                    };
                    self.graph_captures += parts.len();
                    self.engram_graphs[slot] = Some(parts);
                } else {
                    self.graphs[slot] = Some(stream.capture(|| self.enqueue(comm, stream))?);
                    self.graph_captures += 1;
                }
                if regrid.is_some() {
                    drop(regrid);
                    self.moe_regridded = crate::cuda::graph_regridded();
                    ensure!(self.moe_front_grid.is_none() || self.moe_regridded >= 40, "front-end grid rule missed the experts' kernels");
                }
                self.moe_swapped = crate::cuda::graph_swapped();
                marks.push(("graph_capture", Instant::now()));
            }
            if captured_new_graph && retain_warm_result {
                // All Engram rows were prepared before the eager first use;
                // only existing graph replays may defer those uploads. Both
                // ordinary and split graphs now describe the next execution.
                ensure!(!deferred_rows, "first capture unexpectedly deferred Engram rows");
            } else if overlap {
                unsafe { self.engram_graphs[slot].as_ref().unwrap()[0].launch(stream)?; }
                if engram_gap_probe() {
                    // Fires when the initial graph (embedding and layer 0) completes.
                    ENGRAM_GAP.with(|g| -> Result<()> {
                        let mut g = g.borrow_mut();
                        if g.is_none() { *g = Some(EngramGap { first: self.cuda.event()?, second: self.cuda.event()?,
                            count: 0, total_ms: 0., max_ms: 0. }); }
                        g.as_ref().unwrap().first.record(stream)
                    })?;
                    engram_gap_recorded = true;
                }
                marks.push(("initial_enqueue", Instant::now()));
                if deferred_rows {
                    // Layer0 runs while the CPU gathers exact row bytes. No
                    // Engram buffer belongs to that graph; uploads therefore
                    // cannot overwrite data it is still consuming.
                    if split_engram14 {
                        let first:Vec<_>=batch.rows.iter().map(|r|r[0]).collect();
                        self.rows1.prepare(&first,&self.row_store1,8)?;
                    }else{self.prepare_engram_rows(&batch.rows)?;}
                    marks.push(("engram_rows", Instant::now()));
                    stream.join_default()?;
                    marks.push(("engram_upload_order", Instant::now()));
                }
                let middle_index=if self.engram_split14_adaptive && split_engram14 {2}else{1};
                if engram_gap_recorded {
                    // Fires when the stream reaches the rest of the forward: later than the first event only if
                    // the row lookup or its upload outlasted layer 0.
                    ENGRAM_GAP.with(|g| g.borrow().as_ref().unwrap().second.record(stream))?;
                }
                unsafe { self.engram_graphs[slot].as_ref().unwrap()[middle_index].launch(stream)?; }
                if split_engram14 {
                    marks.push(("middle_enqueue",Instant::now()));
                    if deferred_rows {
                        let second:Vec<_>=batch.rows.iter().map(|r|r[1]).collect();
                        let (rows,store)=self.group14.engram_rows()?;
                        rows.prepare(&second,store,8)?;
                        marks.push(("engram_rows",Instant::now()));
                        stream.join_default()?;
                        marks.push(("engram_upload_order",Instant::now()));
                    }
                    let tail_index=if self.engram_split14_adaptive {3}else{2};
                    unsafe{self.engram_graphs[slot].as_ref().unwrap()[tail_index].launch(stream)?;}
                }
            } else {
                unsafe { self.graphs[slot].as_ref().unwrap().launch(stream)?; }
            }
        } else {
            self.enqueue(comm, stream)?;
        }
        marks.push(("forward_enqueue", Instant::now()));
        // Narrow local-target graphs omit the vocabulary gather. Ordinary
        // callers, including point-mass history, greedy and accepted append,
        // still receive complete logits through a small tail graph per live
        // count. Native row transport captures that count as a fixed extent.
        // This avoids retaining a duplicate forty-layer graph bank.
        let gathered_tail=self.local_target_enabled&&self.compact_sampling_active()&&!local_only;
        if gathered_tail {
            if graph_replay {
                if self.local_target_gather[ids.len()].is_none() {
                    self.io.gather_logits(comm,stream)?;stream.synchronize()?;comm.health()?;
                    self.local_target_gather[ids.len()]=Some(stream.capture(||self.io.gather_logits(comm,stream))?);
                    self.graph_captures+=1;
                }else{unsafe{self.local_target_gather[ids.len()].as_ref().unwrap().launch(stream)?;}}
            }else{self.io.gather_logits(comm,stream)?;}
        }
        // verify_sample_ahead: the caller queues its sampler behind this forward and runs the completion below after
        // the sampler's own readout (complete_deferred_verification): no host wake-up between the head and the sampler.
        let deferred = verification && self.defer_completion && batched.is_none() && !engram_gap_recorded
            && self.cached_decoder.is_none() && !gathered_tail;
        self.completion_owed.set(deferred);
        if !deferred {
            if verification&&self.verification_poll_us>0&&self.cached_decoder.is_none() {
                stream.synchronize_polling(self.verification_poll_us)?;
            }else{stream.synchronize()?;}
            comm.health()?;
        }
        marks.push(("gpu_completion", Instant::now()));
        if engram_gap_recorded {
            let rank = self.rank;
            ENGRAM_GAP.with(|g| -> Result<()> {
                let mut g = g.borrow_mut();
                let s = g.as_mut().unwrap();
                let ms = s.second.elapsed_since(&s.first)?;
                s.count += 1; s.total_ms += f64::from(ms); s.max_ms = s.max_ms.max(ms);
                if s.count % 64 == 0 {
                    eprintln!("rank{rank} engram gap: mean {:.1} us over {} verifications (max {:.1} us)",
                        1000. * s.total_ms / s.count as f64, s.count, 1000. * s.max_ms);
                }
                Ok(())
            })?;
        }
        if !deferred {
            self.check_candidates()?;
            if let Some(cache) = &self.encoder_cache {
                ensure!(cache.error()? == 0, "encoder tail publication fault");
            }
        }
        marks.push(("candidate_checks", Instant::now()));
        self.last_head_local_only=self.compact_sampling_active()&&!gathered_tail;
        if let Some(parts) = batched {
            if verification {
                self.pending = Some(PendingVerification { hash_batch: batch, start: self.history.position, live: ids.len() });
                self.batched_pending = true;
                self.batched_block = p.block;
                self.batched_inflight = Some(parts.into_iter().zip(part_batches)
                    .map(|((base, h, live), b)| (base, h, live, b)).collect());
            } else {
                let mut done = Vec::with_capacity(parts.len());
                for ((_, mut h, live), b) in parts.into_iter().zip(part_batches) {
                    h.commit(b, live)?;
                    done.push(h);
                }
                self.batched_done = Some(done);
                self.last_live = ids.len();
                self.lease.finish_write()?;
                self.set_region_step(false)?;
            }
        } else if verification {
            self.pending = Some(PendingVerification {
                hash_batch: batch,
                start: self.history.position,
                live: ids.len(),
            });
        } else {
            if self.source_phase == Phase::Verify {
                self.publish_accepted_encoder(ids.len(), stream)?;
                stream.synchronize()?;
            }
            self.history.commit(batch, ids.len())?;
            if let Some(cache) = &self.encoder_cache {
                cache.commit(self.history.position as usize)?;
            }
            self.last_live = ids.len();
            self.lease.finish_write()?;
        }
        marks.push(("history_commit", Instant::now()));
        self.last_append_metrics = Some(AppendMetrics {
            live_rows: ids.len(),
            physical_rows: self.rows,
            phase: format!("{:?}", self.source_phase),
            captured_new_graph,
            engram_split14: split_engram14,
            engram_split14_adaptive: self.engram_split14_adaptive,
            encoder_graph_chunks: 0,
            lookahead: Vec::new(),
            stall_counters: Vec::new(),
            native_fp32_rows: medium_native.then_some(ids.len()),
            layer20_deferred_chunks: None,
            stages_ms: marks
                .windows(2)
                .map(|pair| {
                    (
                        pair[1].0,
                        pair[1].1.duration_since(pair[0].1).as_secs_f64() * 1000.,
                    )
                })
                .collect(),
            total_ms: marks.last().unwrap().1.duration_since(begin).as_secs_f64() * 1000.,
        });
        Ok(())
    }
    /// Read every local vocabulary column without changing the compact head's
    /// graph or adding an all-gather. The finite screen compares all four ranks.
    pub(crate) fn pending_local_logits_for_probe(&self) -> Result<(&Buffer<f32>, usize)> {
        ensure!(std::env::var("CANADA_SCRATCH_INDEX_SCORE_MODEL_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_SHARED_DOWN_FLASH_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_FFN_HINT_MODEL_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_FFN_SUM_SHARED_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_PAIR_STEP_PROFILE").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_ATTENTION16_REFRESH_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_INDEPENDENT_OWNER_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_WIDE_MHC_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_OUTPUT_PROJECTION_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_WOA_MID_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_WIDE_L2_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_FFN32_REFRESH_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_WIDE_EPILOGUE_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_WIDE_ENGRAM_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_FFN16_REFRESH_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_TRANSPORT_PLACEMENT_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_ENGRAM_PROJECTION_SCREEN").as_deref()==Ok("1")
            ||crate::config::l2_queue_screen(),
            "local logit readout requires finite score storage screen");
        let pending=self.pending.as_ref().context("no completed verification")?;
        ensure!(self.lease.is_busy_revision(),"local logit readout lost writer revision");
        Ok((self.io.local_logits(),pending.live))
    }
    pub fn pending_logits(&self) -> Result<(&Buffer<f32>, usize)> {
        ensure!(!self.last_head_local_only,"compact verification has no gathered full-vocabulary logits");
        ensure!(!(self.append_last_gather&&self.cached_decoder.is_some()),
            "retained append publishes only its final distribution");
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no completed verification"))?;
        ensure!(
            self.lease.is_busy_revision(),
            "pending verification lost writer revision"
        );
        ensure!(!self.batched_pending || self.batched_staged.is_some(), "batched logits need a staged session");
        Ok((self.io.logits(), self.batched_staged.map_or(pending.live, |s| s.1)))
    }

    pub(crate) fn gathered_rows_proof_for_probe(&self)->Result<serde_json::Value> {
        use sha2::{Digest,Sha256};
        ensure!(std::env::var("CANADA_SCRATCH_GATHER_ROWS_SCREEN").as_deref()==Ok("1")
            &&self.rows==6&&self.local_target_enabled,"finite gathered-row diagnostic");
        let(full,live)=self.pending_logits()?;
        let shard_bytes=32320*4;let row_bytes=4*shard_bytes;
        let local=unsafe{self.cuda.read_device(self.io.local_logits().ptr(),live*shard_bytes)}?;
        let gathered=unsafe{self.cuda.read_device(full.ptr(),live*row_bytes)}?;
        let exact=(0..live).all(|r|local[r*shard_bytes..(r+1)*shard_bytes]
            ==gathered[r*row_bytes+self.rank*shard_bytes..r*row_bytes+(self.rank+1)*shard_bytes]);
        Ok(serde_json::json!({"live":live,"rank":self.rank,"exact":exact,
            "local_sha256":format!("{:x}",Sha256::digest(&local)),
            "gathered_sha256":format!("{:x}",Sha256::digest(&gathered))}))
    }
    /// Diagnostic only (CANADA_SCRATCH_CLAUDE_LOGIT_EXPORT): this rank's vocabulary shard of the
    /// pending rows, which every head mode computes before any gather or sampler.
    pub(crate) fn pending_local_rows_for_export(&self)->Result<(&Buffer<f32>,usize)> {
        ensure!(crate::logit_export::dir().is_some(),"local row export requires the logit export diagnostic");
        let pending=self.pending.as_ref().context("no completed verification")?;
        ensure!(self.lease.is_busy_revision(),"logit export lost writer revision");
        Ok((self.io.local_logits(),pending.live))
    }
    pub(crate) fn rank_for_export(&self)->usize {self.rank}
    /// Diagnostic only (CANADA_SCRATCH_CLAUDE_HEAD4_EXPORT=2): the installed 4-bit-head scheme's logits of the
    /// pending verification's head input (this rank's shard of the first `live` rows) and the rows it rescored.
    pub(crate) fn head4_scheme_for_export(&self, live: usize, stream: &Stream) -> Result<Option<(Vec<u8>, Vec<i32>)>> {
        ensure!(crate::logit_export::dir().is_some() && self.pending.is_some() && self.rows == 6, "4-bit scheme export owner");
        self.io.head4_export(live.min(6), stream)
    }
    /// Diagnostic only (CANADA_SCRATCH_CLAUDE_HEAD4_EXPORT): the head's input rows of the pending verification.
    pub(crate) fn head_input_for_export(&self)->Result<&Buffer<bf16>> {
        ensure!(crate::logit_export::dir().is_some()&&self.pending.is_some(),"head input export requires the logit export diagnostic");
        Ok(self.io.normalized())
    }
    pub(crate) fn pending_local_target_logits(&self)->Result<(&Buffer<f32>,usize)> {
        let pending=self.pending.as_ref().context("no local target verification")?;
        ensure!(self.local_target_enabled&&self.rows==6&&self.last_head_local_only
            &&self.lease.is_busy_revision(),"local target logit ownership");
        ensure!(!self.batched_pending||self.batched_staged.is_some(),"batched logits need a staged session");
        Ok((self.io.local_logits(),self.batched_staged.map_or(pending.live,|s|s.1)))
    }
    pub fn verification_greedy(&mut self, stream: &Stream) -> Result<Vec<u32>> {
        let (_, live) = self.pending_logits()?;
        ensure!(
            self.cuda.owns_stream(stream) && stream.raw() as usize == self.stream,
            "pending sampler ordered stream"
        );
        let result = (|| {
            self.sampler.run(self.io.logits(), stream)?;
            let ids = self.sampler.finish(stream)?;
            Ok(ids[..live].iter().map(|&id| id as u32).collect())
        })();
        if result.is_err() {
            self.pending = None;
        } // Keep lease poisoned.
        result
    }
    /// Startup's aborted retained appends publish only the final distribution.
    /// Warm that actual serving consumer without reopening the all-row API or
    /// committing any synthetic token/state. Ordinary policy keeps its prior warmup.
    pub(crate) fn warm_pending_append_sampler(&mut self,stream:&Stream)->Result<()> {
        if !self.append_last_gather {self.verification_greedy(stream)?;return Ok(())}
        let pending=self.pending.as_ref().context("append sampler warmup needs pending rows")?;
        ensure!(self.cached_decoder.is_some() && self.lease.is_busy_revision()
            && !self.last_head_local_only && pending.live>0 && self.cuda.owns_stream(stream)
            && stream.raw()as usize==self.stream,"append sampler warmup ownership");
        self.categorical.sample(self.io.logits(),pending.live-1,1,1.,0,
            crate::categorical::TARGET_DOMAIN,pending.start+pending.live as u64,stream)?;
        Ok(())
    }
    /// Sample each verified row from the target at T=1, top_p=1. With a
    /// deterministic proposal, matching these independent target samples until
    /// the first mismatch is exactly rejection sampling for q=delta(proposal).
    pub(crate) fn verification_sample(&mut self, seed: u64, stream: &Stream) -> Result<Vec<u32>> {
        let (_, live) = self.pending_logits()?;
        ensure!(self.cuda.owns_stream(stream) && stream.raw() as usize == self.stream,
            "pending stochastic sampler ordered stream");
        // A staged batched session samples at its own frontier.
        let position = self.batched_staged.map_or(self.history.position, |s| s.2);
        let result = self.categorical.sample(self.io.logits(), 0, live, 1., seed,
            crate::categorical::TARGET_DOMAIN, position + 1, stream);
        if result.is_err() { self.pending = None; } // Keep shared owner poisoned.
        result
    }
    pub(crate) fn verification_history_sample(&mut self,seed:u64,comm:&Roce,stream:&Stream)->Result<Vec<u32>> {
        if !self.last_head_local_only{return self.verification_sample(seed,stream)}
        let pending=self.pending.as_ref().ok_or_else(||anyhow::anyhow!("no completed compact verification"))?;
        ensure!(self.compact_sampling_active()&&self.lease.is_busy_revision()&&pending.live==self.rows&&[16,32].contains(&self.rows)
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize
            &&self.comm_instance==Some(comm.instance()),"compact pending verification ownership");
        let result=self.compact_categorical.as_mut().ok_or_else(||anyhow::anyhow!("compact sampler storage absent"))?
            .sample(self.io.local_logits(),seed,self.history.position+1,comm,stream);
        if result.is_err(){self.pending=None;} // Leave the causal writer poisoned.
        result
    }
    /// Commit only the accepted input prefix (anchor included). Future KV and
    /// index entries stay beyond logical lengths; subsequent causal metadata
    /// cannot expose them. Restore rejected C2 carry writes before publication.
    /// This is bounded speculation, not branching or full session parking.
    /// Finite batched screen: lay out the next append or verification as these
    /// sessions' rows (in order), each from its own history and cache region.
    pub(crate) fn set_batched_rows_for_probe(&mut self, parts: Vec<(SessionBase, History, usize)>) -> Result<()> {
        ensure!(crate::config::batched_rows_admitted()
            && self.batched_rows.is_none() && !self.batched_pending && self.pending.is_none()
            && !parts.is_empty() && parts.len() <= crate::config::session_slots()?,
            "batched screen owner/sessions");
        self.batched_rows = Some(parts);
        Ok(())
    }
    /// The histories advanced by the last batched append, in part order.
    pub(crate) fn take_batched_histories_for_probe(&mut self) -> Result<Vec<History>> {
        self.batched_done.take().context("no completed batched append")
    }
    /// Discard a batched verification: every compressor ring returns to its
    /// saved state and no session's history advances.
    pub(crate) fn abort_batched_verification_for_probe(&mut self, stream: &Stream) -> Result<()> {
        ensure!(self.batched_pending && self.pending.is_some() && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream, "batched verification abort owner");
        self.pending = None;
        self.batched_pending = false;
        self.batched_inflight = None;
        self.batched_staged = None;
        self.batched_owner_part = false;
        unsafe { self.carry_snapshot.as_ref().context("batched carry snapshot")?.restore(stream)?; }
        stream.synchronize()?;
        self.carry_saved_start = None;
        self.last_live = 0;
        self.lease.finish_write()?;
        self.set_region_step(false)?;
        Ok(())
    }
    /// Commit each session's accepted prefix of a batched verification, in part
    /// order: its rejected compressor-ring rows are restored and only its own
    /// history advances. The owner's own history and encoder tail are untouched.
    /// Batched serving: the next batched verification's part 0 is this owner's own conversation.
    pub(crate) fn set_batched_owner_part(&mut self, enabled: bool) -> Result<()> {
        ensure!(self.batched_rows.is_none() && !self.batched_pending && self.pending.is_none(),
            "batched owner part is set between steps");
        self.batched_owner_part = enabled;
        Ok(())
    }
    pub(crate) fn finish_batched_verification(&mut self, accepted: &[usize], stream: &Stream) -> Result<Vec<History>> {
        ensure!(self.batched_pending && self.pending.is_some() && self.cuda.owns_stream(stream)
            && stream.raw() as usize == self.stream, "batched verification commit owner");
        let parts = self.batched_inflight.take().context("batched verification sessions")?;
        ensure!(parts.len() == accepted.len()
            && parts.iter().zip(accepted).all(|(p, &a)| a <= p.2), "batched accepted prefixes");
        let snapshot = self.carry_snapshot.as_ref().context("batched carry snapshot")?;
        for ((base, h, live, _), &a) in parts.iter().zip(accepted) {
            unsafe { snapshot.restore_rejected_request(usize::try_from(base.request)?, h.position, *live, a, stream)?; }
        }
        stream.synchronize()?;
        // The owner's own conversation (part 0, region 0, at the owner's frontier) commits exactly
        // as an ordinary verification does: its accepted rows (block 0 starts at row 0) enter the
        // owner's encoder tail and its history becomes the owner's.
        let owner = std::mem::take(&mut self.batched_owner_part) && parts.first()
            .is_some_and(|p| p.0 == SessionBase::default() && p.1 == self.history);
        if owner { self.publish_accepted_encoder(accepted[0], stream)?; }
        // Region conversations keep their own encoder tails current (batched serving).
        let mut region_tail_commits = Vec::new();
        if !self.region_tails.is_empty() {
            for (k, ((base, h, _, _), &a)) in parts.iter().zip(accepted).enumerate() {
                if base.request == 0 { continue; }
                let tail = self.tail_of(*base)?.clone();
                ensure!(tail.position() == h.position as usize, "region encoder tail frontier");
                if a > 0 {
                    tail.publish_prefix_at(self.late.encoder_output(), self.late.encoder_pre(),
                        self.late.source().candidates(), &self.tokens, &self.step.compressed_positions,
                        self.late.source().query().candidate_mask(), k * self.batched_block, a, stream)?;
                }
                region_tail_commits.push((tail, h.position as usize + a));
            }
            stream.synchronize()?;
            for (tail, _) in &region_tail_commits { ensure!(tail.error()? == 0, "region encoder tail publication fault"); }
        }
        self.pending = None;
        self.batched_pending = false;
        self.batched_staged = None;
        self.carry_saved_start = None;
        let mut out = Vec::with_capacity(parts.len());
        for ((_, mut h, _, b), &a) in parts.into_iter().zip(accepted) {
            h.commit(b, a)?;
            out.push(h);
        }
        if owner {
            self.history = out[0].clone();
            if let Some(cache) = &self.encoder_cache { cache.commit(self.history.position as usize)?; }
        }
        for (tail, end) in region_tail_commits { tail.commit(end)?; }
        self.last_live = accepted.iter().sum();
        self.lease.finish_write()?;
        self.set_region_step(false)?;
        Ok(out)
    }
    /// Batched step: move one session's logits rows (gathered, or the rank-local
    /// shard when the compact head ran) to row 0, where that session's captured
    /// sampler reads them, and report its live count until the next staging.
    /// Sessions are staged in part order, so rows already sampled are free;
    /// rows are copied in increasing order, which is safe for overlap.
    pub(crate) fn stage_batched_session(&mut self, offset: usize, live: usize, stream: &Stream) -> Result<()> {
        let total = self.pending.as_ref().map(|p| p.live).context("no batched verification")?;
        ensure!(self.batched_pending && live > 0 && offset + live <= total
            && self.batched_staged.is_none_or(|(o, l, _)| o + l <= offset)
            && self.cuda.owns_stream(stream) && stream.raw() as usize == self.stream, "batched session staging order");
        // The staged rows must be exactly one session's rows (or, for a whole-step read, all of them);
        // its frontier seeds the owner-side samplers instead of this owner's own history.
        let mut start = None;
        for (k, (_, h, part_live, _)) in self.batched_inflight.as_ref().context("batched verification sessions")?.iter().enumerate() {
            if k * self.batched_block == offset && (*part_live == live || (offset == 0 && live == total)) { start = Some(h.position); }
        }
        let start = start.context("staged rows are not one session's rows")?;
        if offset > 0 {
            let (base, row) = if self.last_head_local_only { (self.io.local_logits().ptr(), 32320) }
                else { (self.io.logits().ptr(), 129280) };
            for r in 0..live {
                unsafe { self.cuda.copy_device(base + (r * row * 4) as u64, base + ((offset + r) * row * 4) as u64, row * 4, stream)?; }
            }
        }
        self.batched_staged = Some((offset, live, start));
        Ok(())
    }
    /// Move one conversation's causal state, committed through `position`, from session region
    /// `from` to `to` of these shared caches (CANADA_SCRATCH_BATCHED_REGIONS): the window entries of
    /// its last 128 positions (whole 64-slot pages; both rings keep a position's offset in its page),
    /// the four sources' compressed entries and index keys (a page-aligned shift of the region's
    /// compressed addresses), and its pair-carry rings (request slot). `from`'s storage is left as is.
    /// Not moved: the 128-row encoder tail (read only by a later prefill) and the drafter, which
    /// belongs to the conversation's session. The caller owns ordering against other work.
    pub(crate) fn relocate_session(&mut self, from: SessionBase, to: SessionBase, position: usize,
        stream: &Stream) -> Result<serde_json::Value> {
        ensure!(self.lease.is_current() && self.pending.is_none() && self.batched_rows.is_none()
            && position >= 1 && position <= self.context
            && self.cuda.owns_stream(stream) && stream.raw() as usize == self.stream,
            "relocation needs an idle current owner");
        let (fl, tl) = (from.layout.unwrap_or(self.swa_layout), to.layout.unwrap_or(self.swa_layout));
        ensure!(fl.ring().is_some() && tl.ring().is_some() && from.window % 64 == 0 && to.window % 64 == 0,
            "relocation window layouts");
        let mut bytes = 0usize;
        let mut copies = 0usize;
        // Window: the pages holding positions [position - 128, position).
        let first_block = position.saturating_sub(128) / 64;
        let last_block = (position - 1) / 64;
        let windows: Vec<&PagedKv> = self.early_caches.iter().chain(self.group8.caches())
            .chain(self.group14.caches()).chain(self.late.caches()).collect();
        ensure!(windows.len() == 40, "relocation window cache count");
        for cache in &windows {
            ensure!(cache.page_tokens() == 64, "relocation window page geometry");
            let page = cache.page_bytes();
            for block in first_block..=last_block {
                let (src, dst) = ((from.window + fl.slot(block * 64)) / 64, (to.window + tl.slot(block * 64)) / 64);
                ensure!(src < cache.pages() && dst < cache.pages(), "relocation window page bounds");
                unsafe { self.cuda.copy_device(cache.data().ptr() + (dst * page) as u64,
                    cache.data().ptr() + (src * page) as u64, page, stream)?; }
                bytes += page; copies += 1;
            }
        }
        // Compressed entries and index keys: slots [base, base + OFFSET + ceil(position / ratio)).
        for (layer, source) in [2usize, 8, 14, 20].into_iter().zip([
            &self.source2, self.group8.source(), self.group14.source(), self.late.source(),
        ]) {
            let ratio = if layer == 20 { 1 } else { 2 };
            let (from_slot, to_slot) = if ratio == 2 { (from.raw_pair / 2, to.raw_pair / 2) } else { (from.raw_single, to.raw_single) };
            let cache = source.cache();
            let tokens = cache.page_tokens();
            ensure!(from_slot % tokens == 0 && to_slot % tokens == 0 && from_slot % 64 == 0 && to_slot % 64 == 0,
                "relocation compressed alignment");
            let slots = COMPRESSED_OFFSET + position.div_ceil(ratio);
            let pages = slots.div_ceil(tokens);
            ensure!((to_slot / tokens + pages) <= cache.pages() && (from_slot / tokens + pages) <= cache.pages(),
                "relocation compressed page bounds");
            let page = cache.page_bytes();
            unsafe { self.cuda.copy_device(cache.data().ptr() + (to_slot / tokens * page) as u64,
                cache.data().ptr() + (from_slot / tokens * page) as u64, pages * page, stream)?; }
            bytes += pages * page; copies += 1;
            let keys = source.keys();
            let key_slots = slots.div_ceil(64) * 64;
            ensure!(to_slot + key_slots <= keys.capacity() && from_slot + key_slots <= keys.capacity(),
                "relocation key bounds");
            unsafe { self.cuda.copy_device(keys.data().ptr() + (to_slot * 68) as u64,
                keys.data().ptr() + (from_slot * 68) as u64, key_slots * 68, stream)?; }
            bytes += key_slots * 68; copies += 1;
            if let Some(ring) = source.source().ring() {
                let requests = crate::config::session_slots()?;
                ensure!(ring.len() % requests == 0 && (from.request as usize) < requests && (to.request as usize) < requests,
                    "relocation carry ring geometry");
                let per = ring.len() / requests * 4;
                unsafe { self.cuda.copy_device(ring.ptr() + (to.request as usize * per) as u64,
                    ring.ptr() + (from.request as usize * per) as u64, per, stream)?; }
                bytes += per; copies += 1;
            }
        }
        // Region tails (batched serving): the conversation's 128-row encoder tail moves along.
        let tail_moved = if !self.region_tails.is_empty() && from.request != to.request {
            let (source, destination) = (self.tail_of(from)?.clone(), self.tail_of(to)?.clone());
            ensure!(source.position() == position, "moved encoder tail frontier");
            // Tail candidate ids are physical slots of the single-ratio source (layer 20).
            destination.copy_from(&source, to.raw_single as i64 - from.raw_single as i64, stream)?;
            true
        } else { false };
        self.invalidate_gather_caches();
        stream.synchronize()?;
        Ok(serde_json::json!({"position": position, "copies": copies, "bytes": bytes,
            "window_blocks": last_block + 1 - first_block, "encoder_tail_moved": tail_moved}))
    }
    /// The DSpark feature taps of one batched session's rows (row-offset pointers).
    pub(crate) fn batched_feature_inputs(&self, offset: usize, live: usize, stream: &Stream) -> Result<[u64; 3]> {
        let (inputs, total) = self.dspark_feature_inputs(stream)?;
        ensure!(self.batched_pending && live > 0 && offset + live <= total, "batched feature rows");
        Ok(inputs.map(|b| b.ptr() + (offset * 20480 * 2) as u64))
    }
    pub(crate) fn history_for_probe(&self) -> History { self.history.clone() }
    pub(crate) fn graph_capture_count(&self) -> usize { self.graph_captures }
    /// Batched serving: a conversation moved back into region 0 (relocate_session, which also moved
    /// its encoder tail) becomes this owner's own history.
    pub(crate) fn adopt_moved_history(&mut self, history: History) -> Result<()> {
        ensure!(self.lease.is_current() && self.pending.is_none() && self.batched_rows.is_none()
            && !self.batched_pending && !self.region_tails.is_empty()
            && self.encoder_cache.as_ref().is_some_and(|c| c.position() as u64 == history.position),
            "moved history needs an idle region-0 owner and its encoder tail");
        self.history = history;
        self.invalidate_gather_caches();
        Ok(())
    }
    pub(crate) fn batched_pending_for_probe(&self) -> bool { self.batched_pending }
    /// Screens: padding routes for the following batched steps of several sessions (their graphs
    /// use their own slots, so both settings stay captured).
    pub(crate) fn set_padding_routes_for_probe(&mut self, on: bool) -> Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_BATCH_SCREEN").as_deref() == Ok("1")
            && self.batched_rows.is_none() && !self.batched_pending, "padding routes screen toggle");
        self.padding_routes = on;
        Ok(())
    }
    /// Row offset of part `k` of the in-flight batched verification.
    pub(crate) fn batched_offset(&self, k: usize) -> Result<usize> {
        let parts = self.batched_inflight.as_ref().context("no batched verification")?;
        ensure!(self.batched_pending && k < parts.len(), "batched part index");
        Ok(k * self.batched_block)
    }
    /// Finite batched screen: replace this owner's own history (to run the
    /// ordinary single-session path at a session's frontier) and get it back.
    pub(crate) fn swap_history_for_probe(&mut self, history: History) -> Result<History> {
        ensure!(std::env::var("CANADA_SCRATCH_BATCH_SCREEN").as_deref() == Ok("1")
            && self.pending.is_none() && self.batched_rows.is_none(), "batched screen history swap");
        Ok(std::mem::replace(&mut self.history, history))
    }
    pub(crate) fn swa_layout_for_probe(&self) -> crate::swa_layout::Layout { self.swa_layout }
    /// verify_sample_ahead: whether the next verification defers its completion (see `defer_completion`).
    pub(crate) fn defer_verification_completion(&mut self, on: bool) { self.defer_completion = on; }
    /// The completion a deferred verification skipped: wait for the stream (already idle after the sampler's
    /// readout), then the transport and fault checks, in the order execute_rows runs them. A no-op otherwise.
    pub(crate) fn complete_deferred_verification(&self, comm: &Roce, stream: &Stream) -> Result<()> {
        if !self.completion_owed.replace(false) { return Ok(()); }
        stream.synchronize()?;
        comm.health()?;
        self.check_candidates()?;
        if let Some(cache) = &self.encoder_cache {
            ensure!(cache.error()? == 0, "encoder tail publication fault");
        }
        Ok(())
    }
    pub fn finish_verification(
        &mut self,
        accepted: usize,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<usize> {
        ensure!(!self.batched_pending, "a batched screen verification is aborted, never committed");
        ensure!(!self.completion_owed.get(), "a deferred verification was not completed before its commit");
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no completed verification"))?;
        ensure!(
            self.lease.is_busy_revision()
                && accepted <= pending.live
                && pending.start == self.history.position
                && self.cuda.owns_stream(stream)
                && stream.raw() as usize == self.stream
                && self.comm_instance == Some(comm.instance())
                && comm.rank() == self.rank,
            "verification commit owner/prefix/stream"
        );
        ensure!(!(self.append_last_gather&&self.cached_decoder.is_some())
            || accepted==0 || accepted==pending.live,
            "last-logit append must commit all rows or abort");
        let pending = self.pending.take().unwrap();
        let restored = unsafe {
            self.carry_snapshot
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("verification carry snapshot absent"))?
                .restore_rejected(pending.start, pending.live, accepted, stream)?
        };
        self.publish_accepted_encoder(accepted, stream)?;
        stream.synchronize()?;
        comm.health()?;
        self.history.commit(pending.hash_batch, accepted)?;
        if let Some(cache) = &self.encoder_cache {
            cache.commit(self.history.position as usize)?;
        }
        self.last_live = accepted;
        self.last_append_metrics = None;
        self.carry_saved_start = None;
        self.lease.finish_write()?;
        Ok(restored)
    }
    fn publish_accepted_encoder(&self, accepted: usize, stream: &Stream) -> Result<()> {
        if let Some(cache) = &self.encoder_cache {
            ensure!(
                cache.position() == self.history.position as usize,
                "encoder tail commit frontier"
            );
            if accepted > 0 {
                cache.publish_prefix(
                    self.late.encoder_output(),
                    self.late.encoder_pre(),
                    self.late.source().candidates(),
                    &self.tokens,
                    &self.step.compressed_positions,
                    self.late.source().query().candidate_mask(),
                    accepted,
                    stream,
                )?;
            }
            stream.synchronize()?;
            ensure!(
                cache.error()? == 0,
                "accepted encoder tail publication fault"
            );
        }
        Ok(())
    }
    /// Byte-only diagnostic of the current accepted carry; no numeric CPU model.
    pub fn carry_digests(&self) -> Result<Vec<String>> {
        ensure!(
            self.lease.is_current() && self.pending.is_none(),
            "carry diagnostic requires accepted state"
        );
        let rings = [
            &self.source2,
            self.group8.source(),
            self.group14.source(),
            self.late.source(),
        ];
        rings
            .into_iter()
            .filter_map(|s| s.source().ring())
            .map(|r| {
                let raw = unsafe { self.cuda.read_device(r.ptr(), r.bytes()) }?;
                Ok(format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(raw)))
            })
            .collect()
    }
    /// Probe only: the raw rows the drafter's taps read after a prompt (three buffers, `live` rows of 20,480 BF16).
    pub(crate) fn dspark_feature_rows_for_probe(&self, stream: &Stream) -> Result<Vec<Vec<u8>>> {
        let (inputs, live) = self.dspark_feature_inputs(stream)?;
        stream.synchronize()?;
        inputs.iter().map(|b| unsafe { self.cuda.read_device(b.ptr(), live * 20480 * 2) }).collect()
    }
    /// Canonical DSpark tap inputs, ordered by target attention layers37..39.
    /// A completed layer36..38 output is the raw residual at the next layer's
    /// attention input. Pending rows remain private to this writer transaction.
    pub(crate) fn dspark_feature_inputs(
        &self,
        stream: &Stream,
    ) -> Result<([&Buffer<bf16>; 3], usize)> {
        ensure!(
            self.cuda.owns_stream(stream)
                && stream.raw() as usize == self.stream
                && self.late.layers().len() == 20,
            "DSpark target tap owner/stream"
        );
        let live = if let Some(pending) = &self.pending {
            ensure!(
                self.lease.is_busy_revision(),
                "DSpark pending tap lost writer"
            );
            pending.live
        } else {
            ensure!(
                self.lease.is_current() && self.last_live > 0,
                "DSpark tap needs completed target rows"
            );
            self.last_live
        };
        let layers = self.late.layers();
        Ok((
            [
                layers[16].output(),
                layers[17].output(),
                layers[18].output(),
            ],
            live,
        ))
    }
    /// The last execute_rows' host stages, also while a batched verification is pending (diagnostic).
    pub(crate) fn last_stages_ms(&self) -> Option<&[(&'static str, f64)]> {
        self.last_append_metrics.as_ref().map(|m| m.stages_ms.as_slice())
    }
    pub fn append_metrics(&self) -> Result<Option<&AppendMetrics>> {
        ensure!(
            self.lease.is_current() && self.last_live > 0,
            "stale append metrics"
        );
        Ok(self.last_append_metrics.as_ref())
    }
    /// Diagnostic metadata for the completed but not-yet-committed verifier.
    /// Normal accepted-output readers must continue rejecting this revision.
    /// Host milliseconds before GPU completion was awaited, and the Engram
    /// row-lookup part of it, for the pending verification.
    pub(crate) fn verification_host_ms(&self) -> Result<[f64; 2]> {
        let m = self.verification_metrics()?;
        let gpu = m.stages_ms.iter().position(|(name, _)| *name == "gpu_completion");
        let host = m.stages_ms[..gpu.unwrap_or(m.stages_ms.len())].iter().map(|(_, ms)| ms).sum();
        let engram = m.stages_ms.iter().filter(|(name, _)| *name == "engram_rows").map(|(_, ms)| ms).sum();
        Ok([host, engram])
    }
    pub(crate) fn verification_metrics(&self) -> Result<&AppendMetrics> {
        ensure!(
            self.lease.is_busy_revision() && self.pending.is_some(),
            "no current pending verification metrics"
        );
        self.last_append_metrics
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("verification metrics absent"))
    }
    pub fn decoder_tail_bytes(&self) -> Option<usize> {
        self.encoder_cache.as_ref().map(|c| c.bytes())
    }
    pub fn enable_decoder_tail(&mut self, core: &[u8]) -> Result<()> {
        ensure!(
            self.rows == 128
                && self.history.position == 0
                && self.graphs.iter().all(Option::is_none)
                && self.encoder_cache.is_none(),
            "decoder-tail initialization needs fresh M128 owner"
        );
        let cache = Rc::new(crate::decoder_tail::Cache::new(
            &self.cuda,
            core,
            self.context,
            self.late.source().cache().capacity(),
        )?);
        self.replay = Some(cache.new_replay()?);
        self.encoder_cache = Some(cache);
        Ok(())
    }
    /// Batched serving (CANADA_SCRATCH_BATCHED_REGIONS > 1): one encoder tail per serving region
    /// 1..R-1, of region 0's shape, on the M128 owner before its profiles fork (they share them).
    pub(crate) fn enable_region_tails(&mut self, regions: usize) -> Result<()> {
        ensure!(self.rows == 128 && (2..=4).contains(&regions) && self.region_tails.is_empty()
            && self.graphs.iter().all(Option::is_none), "region tails need a fresh M128 owner");
        let tail = self.encoder_cache.as_ref().context("region tails follow region 0's encoder tail")?;
        for _ in 1..regions { self.region_tails.push(Rc::new(tail.twin()?)); }
        Ok(())
    }
    fn tail_of(&self, base: SessionBase) -> Result<&Rc<crate::decoder_tail::Cache>> {
        if base.request == 0 { return self.encoder_cache.as_ref().context("region 0 encoder tail"); }
        self.region_tails.get(usize::try_from(base.request)? - 1).context("serving region encoder tail")
    }
    /// Fresh bounded128-row prefill. The default qualified-recipe path runs
    /// encoder chunks then one decoder tail; the explicit canonical experiment
    /// advances every layer over every prompt token. Accepted encoder state
    /// remains reusable. This path enqueues eagerly, outside graph timing.
    pub(crate) fn prefill(&mut self, ids: &[u32], comm: &Roce, stream: &Stream) -> Result<()> {
        ensure!(
            self.rows == 128
                && self.source_phase == Phase::Extend
                && self.history.position == 0
                && self.pending.is_none()
                && self.lease.is_current()
                && self.prefill_last_head
                && self.cached_decoder.is_none()
                && self
                    .encoder_cache
                    .as_ref()
                    .is_some_and(|c| c.position() == 0)
                && self.replay.is_some()
                && !ids.is_empty()
                && ids.len() <= self.context
                && ids.iter().all(|&id| id < 129280)
                && comm.rank() == self.rank
                && self.cuda.owns_stream(stream)
                && stream.raw() as usize == self.stream
                && self.comm_instance.is_none_or(|v| v == comm.instance()),
            "fresh chunked prefill owner/context/stream"
        );
        if ids.len() <= 128 {
            return self.append(ids, comm, stream);
        }
        self.last_live = 0;
        self.last_append_metrics = None;
        self.carry_saved_start = None;
        self.comm_instance = Some(comm.instance());
        let canonical = CANONICAL_PREFILL_FOR_PROBE.with(|c| c.get())
            .unwrap_or_else(|| std::env::var("CANADA_SCRATCH_CANONICAL_PREFILL").as_deref() == Ok("1"));
        // The full reference advances every layer over every prompt token.
        // Finish with128 rows so the draft receives its complete accepted tail,
        // including prompts whose length is not a multiple of128.
        let chunks: Vec<&[u32]> = if canonical {
            let boundary = ids.len() - 128;
            ids[..boundary]
                .chunks(128)
                .chain(std::iter::once(&ids[boundary..]))
                .collect()
        } else {
            ids.chunks(128).collect()
        };
        let cache = self.encoder_cache.as_ref().unwrap().clone();
        self.lease.begin()?;
        for chunk in chunks {
            let previous = self.history.position as usize;
            let end = previous + chunk.len();
            let p = Plan::append_with_layout(
                previous as u64,
                chunk.len(),
                128,
                self.context,
                self.swa_layout,
            )?;
            let mut padded = vec![2u32; 128];
            padded[..chunk.len()].copy_from_slice(chunk);
            let batch = self.history.prepare_padded(&padded, chunk.len(), &self.hashes)?;
            self.prepare_engram_rows(&batch.rows)?;
            self.io.update_tokens(&mut self.tokens, self.history.position as usize,
                &padded.iter().copied().map(i64::from).collect::<Vec<_>>(),
                chunk.len(),
            )?;
            self.step.update(&p, stream)?;
            if !canonical && end == ids.len() {
                cache.prepare(self.replay.as_mut().unwrap(), previous, end)?;
            }
            comm.select_decode_reduction(false, stream)?;
            comm.select_prefill_rows(chunk.len(), 128, stream)?;
            stream.join_default()?;
            if canonical {
                self.enqueue(comm, stream)?;
            } else {
                self.enqueue_encoder(comm, stream)?;
            }
            stream.synchronize()?;
            comm.health()?;
            self.check_candidates()?;
            ensure!(cache.error()? == 0, "prefill encoder publication fault");
            self.history.commit(batch, chunk.len())?;
            cache.commit(end)?;
        }
        if canonical {
            self.last_live = 128;
            self.lease.finish_write()?;
            return Ok(());
        }
        let start = ids.len() - 128;
        let mut tail =
            Plan::append_with_layout(start as u64, 128, 128, self.context, self.swa_layout)?;
        tail.floor_swa_at(start)?;
        self.step.update(&tail, stream)?;
        comm.select_prefill_rows(128, 128, stream)?;
        stream.join_default()?;
        self.enqueue_tail(comm, stream)?;
        stream.synchronize()?;
        comm.health()?;
        self.check_candidates()?;
        ensure!(cache.error()? == 0, "prefill decoder tail fault");
        self.last_live = 128;
        self.lease.finish_write()?;
        Ok(())
    }
    /// Fresh or retained prefill through wide encoder profiles: layers0..20 over up
    /// to a profile's width per forward, so each routed expert is read once
    /// per layer per chunk instead of once per 128 rows, then this profile's
    /// qualified 128-row decoder tail. Chunk widths (including this 128-row
    /// profile's own encoder) are chosen from measured per-chunk cost, since
    /// every kernel runs over its profile's full width. Each chunk is one
    /// write transaction on the profile that executes it; authority moves by
    /// handoff over the same causal state and encoder-tail ring.
    pub(crate) fn prefill_wide(
        &mut self,
        wides: &mut [Self],
        ids: &[u32],
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        let lookahead = std::env::var("CANADA_SCRATCH_ENGRAM_PREFETCH").as_deref() == Ok("1");
        self.prefill_wide_with_prefetch(wides, ids, comm, stream, lookahead)
    }
    /// Explicit per-call policy permits an in-process paired I/O diagnostic
    /// without mutating process environment or numerical execution policy.
    pub(crate) fn prefill_wide_with_prefetch(
        &mut self,
        wides: &mut [Self],
        ids: &[u32],
        comm: &Roce,
        stream: &Stream,
        lookahead: bool,
    ) -> Result<()> {
        // Carry the exact s270 gain into the working candidate. Profiles with
        // explicit graph admission reuse the bounded tail by default; retain
        // an eager control and reject malformed policy rather than guessing.
        let tail_graph = match std::env::var("CANADA_SCRATCH_PREFILL_TAIL_GRAPH") {
            Ok(value) => {
                ensure!(value == "0" || value == "1", "invalid prefill tail graph policy");
                value == "1"
            }
            Err(std::env::VarError::NotPresent) => self.graph_replay,
            Err(error) => return Err(error.into()),
        };
        let (encoder_graph,tiny_encoder_graph)=self.default_encoder_graph_policy()?;
        let progress=self.prefill_wide_interruptible_policy(wides,ids,comm,stream,lookahead,
            tail_graph,encoder_graph,tiny_encoder_graph,false,None,&mut |_|Ok(false))?;
        ensure!(progress.done==ids.len(),"uninterrupted prefill must consume its complete input");
        Ok(())
    }
    // s551: full128-row encoder graphs preserve exact model/state results and
    // save a small warm increment latency. Wide/partial chunks keep their planner.
    fn default_encoder_graph_policy(&self)->Result<(bool,bool)> {
        match std::env::var("CANADA_SCRATCH_PREFILL_ENCODER_GRAPH") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid encoder graph policy");Ok((value=="1",false))},
            Err(std::env::VarError::NotPresent)=>Ok((false,self.graph_replay && self.context==1048576)),
            Err(error)=>Err(error.into()),
        }
    }
    /// Bounded decoder-tail graph capture. Wide encoder chunks remain eager.
    /// The explicit policy supports
    /// an in-process paired screen with identical pointers and collectives.
    pub(crate) fn prefill_wide_with_policy(
        &mut self,
        wides: &mut [Self],
        ids: &[u32],
        comm: &Roce,
        stream: &Stream,
        lookahead: bool,
        tail_graph: bool,
    ) -> Result<()> {
        self.prefill_wide_with_graph_policy(wides, ids, comm, stream, lookahead, tail_graph, false)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_wide_with_graph_policy(
        &mut self,
        wides: &mut [Self],
        ids: &[u32],
        comm: &Roce,
        stream: &Stream,
        lookahead: bool,
        tail_graph: bool,
        encoder_graph: bool,
    ) -> Result<()> {
        let progress=self.prefill_wide_interruptible_policy(wides,ids,comm,stream,lookahead,
            tail_graph,encoder_graph,false,false,None,&mut |_|Ok(false))?;
        ensure!(progress.done==ids.len(),"uninterrupted prefill must consume its complete input");
        Ok(())
    }
    /// Cooperative scheduling/cancellation hook. The callback runs at an already
    /// committed encoder-chunk boundary. If it requests a yield, materialize
    /// the existing bounded decoder tail before returning a snapshotable owner.
    pub(crate) fn prefill_wide_interruptible(
        &mut self,wides:&mut [Self],ids:&[u32],comm:&Roce,stream:&Stream,
        resume_chunks:Option<Vec<usize>>,
        should_yield:&mut impl FnMut(usize)->Result<bool>,
    )->Result<PrefillProgress> {
        let lookahead=std::env::var("CANADA_SCRATCH_ENGRAM_PREFETCH").as_deref()==Ok("1");
        let tail_graph=match std::env::var("CANADA_SCRATCH_PREFILL_TAIL_GRAPH") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid prefill tail graph policy");value=="1"},
            Err(std::env::VarError::NotPresent)=>self.graph_replay,
            Err(error)=>return Err(error.into()),
        };
        let (encoder_graph,tiny_encoder_graph)=self.default_encoder_graph_policy()?;
        self.prefill_wide_interruptible_policy(wides,ids,comm,stream,lookahead,
            tail_graph,encoder_graph,tiny_encoder_graph,false,resume_chunks,should_yield)
    }
    #[allow(clippy::too_many_arguments)]
    fn prefill_wide_interruptible_policy(
        &mut self,wides:&mut [Self],ids:&[u32],comm:&Roce,stream:&Stream,
        lookahead:bool,tail_graph:bool,encoder_graph:bool,tiny_encoder_graph:bool,defer_decoder:bool,
        resume_chunks:Option<Vec<usize>>,
        should_yield:&mut impl FnMut(usize)->Result<bool>,
    )->Result<PrefillProgress> {
        self.prefill_wide_interruptible_checkpoint(wides,ids,comm,stream,lookahead,tail_graph,encoder_graph,
            tiny_encoder_graph,defer_decoder,resume_chunks,None,None,should_yield)
    }
    /// As prefill_wide_interruptible_policy. With `checkpoint` = (side state, offset into
    /// `ids`), the last encoder chunk also captures the state as of that offset
    /// (capture_checkpoint_as_of); the offset must lie in that chunk, 128 rows or more in.
    #[allow(clippy::too_many_arguments)]
    fn prefill_wide_interruptible_checkpoint(
        &mut self,wides:&mut [Self],ids:&[u32],comm:&Roce,stream:&Stream,
        lookahead:bool,tail_graph:bool,encoder_graph:bool,tiny_encoder_graph:bool,defer_decoder:bool,
        resume_chunks:Option<Vec<usize>>,
        checkpoint:Option<(Rc<CheckpointAux>,usize)>,
        extra:Option<(Rc<CheckpointAux>,usize)>,
        should_yield:&mut impl FnMut(usize)->Result<bool>,
    )->Result<PrefillProgress> {
        ensure!(checkpoint.is_none() || (!defer_decoder && resume_chunks.is_some() && !encoder_graph),
            "inline encoder checkpoint needs its admitted eager chunk plan");
        ensure!(extra.is_none() || checkpoint.is_some(), "an extra inline capture accompanies the checkpoint's");
        ensure!(!defer_decoder || (encoder_checkpoint_enabled() && (ids.len()>=256
            || (ids.len()>=128&&self.history.position>0
                &&defer_checkpoint_decoder128()?))),
            "deferred decoder is confined to the encoder-checkpoint transaction");
        ensure!(!(tail_graph || encoder_graph || tiny_encoder_graph) || self.graph_replay,
            "prefill graphs require explicit graph admission");
        ensure!(
            self.rows == 128
                && self.source_phase == Phase::Extend
                && self.pending.is_none()
                && self.lease.is_current()
                && self.prefill_last_head
                && self
                    .encoder_cache
                    .as_ref()
                    .is_some_and(|c| c.position() == self.history.position as usize)
                && self.replay.is_some()
                && !ids.is_empty()
                && ids.len() <= self.context.saturating_sub(self.history.position as usize)
                && ids.iter().all(|&id| id < 129280)
                && comm.rank() == self.rank
                && self.cuda.owns_stream(stream)
                && stream.raw() as usize == self.stream
                && self.comm_instance.is_none_or(|v| v == comm.instance())
                && wides
                    .iter()
                    .all(|w| w.wide.is_some() && w.context == self.context),
            "wide prefill owner/context/stream"
        );
        // A ragged final chunk of an interrupted wide plan must keep that
        // plan's encoder/tail path, even when fewer than128 tokens remain.
        if resume_chunks.is_none() && (ids.len() < 128 || (self.history.position == 0 && ids.len() == 128)) {
            ensure!(self.history.position == 0 || self.cached_decoder.is_some(),
                "small retained EXTEND needs a stable decoder-floor profile");
            // The new request has fewer than128 rows. Its decoder begins at
            // this request's first row, not at the previous128-row history tail.
            // The persistent cached Step is allocated before graph capture.
            self.append(ids, comm, stream)?;
            return Ok(PrefillProgress{done:ids.len(),remaining_chunks:Vec::new()});
        }
        let previous = self.history.position as usize;
        let begin = Instant::now();
        let mut marks = vec![("begin", begin)];
        self.last_live = 0;
        self.last_append_metrics = None;
        self.carry_saved_start = None;
        self.comm_instance = Some(comm.instance());
        let cache = self.encoder_cache.as_ref().unwrap().clone();
        // None is this 128-row profile; Some(i) indexes `wides`.
        let mut holder: Option<usize> = None;
        let mut widths = Vec::new();
        let mut done = 0;
        let mut captured_new_graph = false;
        let mut encoder_graph_chunks = 0;
        let mut lookahead_stages = Vec::new();
        // Captured encoder graphs always contain the whole layer 20.
        let layer20_defer=layer20_defer_enabled()? && !encoder_graph;
        let mut layer20_deferred=0usize;
        let stall_mode = std::env::var("CANADA_SCRATCH_PREFILL_STALL_COUNTERS").ok();
        let profile_first_wide = stall_mode.as_deref() == Some("profile-first-wide")
            && crate::cuda::SERVING_STARTED.load(std::sync::atomic::Ordering::Relaxed);
        let trace_phases = matches!(stall_mode.as_deref(), Some("phases" | "profile-first-wide"));
        let record_stalls = trace_phases || stall_mode.as_deref() == Some("1");
        let mut stall_counters=Vec::new();
        let available: Vec<usize> = self.prefill_widths(wides);
        let lookahead=if lookahead {
            ensure!(std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref() == Ok("1"),
                "Engram lookahead requires the bounded io_uring reader");
            // At8K rows the preview briefly holds two chunks plus split-off
            // hashes (~9MiB); flat IDs and two4096-row byte buffers fit below
            // the16MiB allowance. Preserve the8GiB operating reserve.
            let memory=self.cuda.admission_memory()?;
            // This is admission for optional host-side lookahead, not for the
            // already allocated target state. A changing MemAvailable estimate
            // must not kill one rank here (including between checkpoint prefix
            // and suffix). Serving admits/reclaims at the committed boundary;
            // every later allocation still enforces its own physical floor.
            let available=memory.admitted_available;
            #[cfg(feature="storage-fault-injection")]
            let available=crate::memory::prefill_preview_available(available);
            let fits=available>=crate::memory::floor_bytes()+16*1024usize.pow(2);
            if !fits {
                // This task warms immutable local row caches only. Every rank
                // executes the same GPU work and collective sequence when it
                // is absent; demand reads provide exactly the same row bytes.
                eprintln!("Engram lookahead skipped at reserved memory boundary: {} bytes",memory.admitted_available);
            }
            fits
        }else{false};
        // Diagnostic capture only: select one ordinary eager encoder chunk
        // without changing chunk planning, math, scheduling or cache ownership.
        let profile_start = std::env::var("CANADA_SCRATCH_PROFILE_ENCODER_CHUNK")
            .ok().map(|v|v.parse::<usize>()).transpose()?;
        // Any full 8K chunk boundary (e.g. 131072, or 229376 near the end of a 245K prompt).
        ensure!(profile_start.is_none() || (!encoder_graph && profile_start.is_some_and(|p| p >= 8192 && p % 8192 == 0)),
            "bounded encoder profile requires the current eager path at an 8K chunk boundary");
        let chunks=if let Some(chunks)=resume_chunks {
            crate::config::validate_encoder_chunks(ids.len(),&chunks,&available)?;chunks
        }else{
            let merge=previous>0&&self.prefill_short_merge&&wides.iter().any(|w|
                w.rows==2048&&w.prefill_partial2048_dense&&w.prefill_wide_mhc_rows&&w.prefill_wide_mhc_norm);
            crate::config::plan_encoder_append(ids.len(),&available,merge)
        };
        for (chunk_index, &rows) in chunks.iter().enumerate() {
            ensure!(rows != 16384 || !self.cuda.wide16_configured() || self.cuda.wide16_mapped(),
                "a 16,384-row chunk needs its elastic extension mapped");
            let profile = wides.iter().position(|w| w.rows == rows);
            let remaining = ids.len() - done;
            let chunk = &ids[done..done + remaining.min(rows)];
            let next = if lookahead {
                chunks.get(chunk_index + 1).map(|&capacity| {
                    let start = done + chunk.len();
                    let live = (ids.len() - start).min(capacity);
                    let mut tokens = vec![2u32; capacity];
                    tokens[..live].copy_from_slice(&ids[start..start + live]);
                    tokens
                })
            } else { None };
            match (profile, holder) {
                (None, Some(h)) => self.handoff_from(&wides[h])?,
                (Some(i), None) => wides[i].handoff_from(self)?,
                (Some(i), Some(h)) if i != h => {
                    let (a, b) = if i < h {
                        let (l, r) = wides.split_at_mut(h);
                        (&mut l[i], &r[0])
                    } else {
                        let (l, r) = wides.split_at_mut(i);
                        (&mut r[0], &l[h])
                    };
                    a.handoff_from(b)?
                }
                _ => {}
            }
            let target = match profile {
                None => &mut *self,
                Some(i) => &mut wides[i],
            };
            let bounded_8k = profile_start == Some(previous + done);
            let first_wide = profile_first_wide && previous + done == 0 && rows == 16384
                && chunk.len() == rows && !encoder_graph;
            let profile_this_chunk = bounded_8k || first_wide;
            if profile_this_chunk {
                ensure!(!bounded_8k || (rows == 8192 && chunk.len() == 8192),
                    "bounded profile must contain one full8K encoder chunk");
                stream.synchronize()?;target.cuda.profiler_start()?;
            }
            let chunk_graph=encoder_graph || (tiny_encoder_graph && rows==128 && chunk.len()==128);
            // Deferred layer 20: an eager full chunk whose rows all precede this
            // request's final 128 rows (the only layer-20 rows the decoder reads).
            target.layer20_pending=false;
            target.layer20_projection_only=layer20_defer && !chunk_graph
                && layer20_projection_only(done,chunk.len(),rows,ids.len())
                && target.late.encoder_deferral_supported();
            let counters=record_stalls.then(crate::prefill_stall::Snapshot::read);
            let chunk_start=Instant::now();
            let history_before=checkpoint.as_ref().map(|_|target.history.clone());
            let chunk_result=target.encoder_chunk(chunk, next.as_deref(), &cache, comm, stream, chunk_graph, trace_phases);
            target.layer20_projection_only=false;
            if let Some(counters)=counters {
                stall_counters.push(counters.finish(previous+done,chunk.len(),rows,chunk_start.elapsed().as_secs_f64()*1000.));
            }
            if profile_this_chunk {
                // Stop even if the encoder returns an error; no later model
                // checks or decoder-tail work belong in this trace.
                target.cuda.profiler_stop()?;
                eprintln!("bounded_encoder_profile rank={} start={} rows={}",target.rank,previous+done,rows);
            }
            let (prepared, captured, lookahead_metrics) = chunk_result?;
            if let Some(metrics)=lookahead_metrics {lookahead_stages.push(metrics);}
            captured_new_graph |= captured;
            encoder_graph_chunks += usize::from(chunk_graph && chunk.len() == rows && !captured);
            holder = profile;
            widths.push(rows);
            done += chunk.len();
            // Host preparation (hashing, Engram row lookup, metadata upload)
            // precedes the GPU work of the same chunk.
            marks.push(("encoder_host", prepared));
            marks.push(("encoder_chunk", Instant::now()));
            if let Some((aux,offset))=&checkpoint && done==ids.len() {
                // The admitted plan put the checkpoint in this last chunk.
                let start=done-chunk.len();
                ensure!(profile.is_some() && !chunk_graph && *offset>=start+128 && *offset<=done,
                    "inline encoder checkpoint lies outside the last chunk");
                target.capture_checkpoint_as_of(aux,history_before.as_ref().unwrap(),chunk,*offset-start,comm,stream)?;
                marks.push(("checkpoint_capture", Instant::now()));
            }
            if let Some((aux,offset))=&extra && done==ids.len() {
                // A second capture from the same untouched chunk state, as of a later row (the near-end image).
                let start=done-chunk.len();
                ensure!(profile.is_some() && !chunk_graph && *offset>=start+128 && *offset<=done,
                    "extra inline checkpoint lies outside the last chunk");
                target.capture_checkpoint_as_of(aux,history_before.as_ref().unwrap(),chunk,*offset-start,comm,stream)?;
                marks.push(("checkpoint_capture_extra", Instant::now()));
            }
            if done<ids.len() && should_yield(done)? {
                // This chunk now ends the turn, so its final rows feed the decoder
                // tail: issue the deferred layer-20 work unchanged before stopping.
                if target.layer20_pending {
                    target.finish_deferred_layer20_now(comm, stream, &cache)?;
                }
                break;
            }
            if target.layer20_pending {
                target.layer20_pending=false;
                layer20_deferred+=1;
            }
        }
        ensure!(!self.layer20_pending && wides.iter().all(|w| !w.layer20_pending),
            "deferred layer20 left pending");
        let end=previous+done;
        if let Some(h) = holder {
            self.handoff_from(&wides[h])?;
        }
        if !defer_decoder || done<ids.len() {
        self.lease.begin()?;
        // EXTEND's decoder sees only the newest min(128,new_rows) rows. Here
        // retained increments are >=128, so the whole tail belongs to this
        // request. Encoder history remains absolute and includes the prefix.
        let start = end - 128;
        cache.prepare_committed(self.replay.as_mut().unwrap(), end)?;
        let mut tail =
            Plan::append_with_layout(start as u64, 128, 128, self.context, self.swa_layout)?;
        tail.floor_swa_at(start)?;
        self.step.update(&tail, stream)?;
        comm.select_decode_reduction(false, stream)?;
        comm.select_prefill_rows(128, 128, stream)?;
        stream.join_default()?;
        captured_new_graph |= self.execute_tail(comm, stream, tail_graph)?;
        stream.synchronize()?;
        comm.health()?;
        self.check_candidates()?;
        ensure!(cache.error()? == 0, "prefill decoder tail fault");
        marks.push(("decoder_tail", Instant::now()));
        self.last_live = 128;
        self.lease.finish_write()?;
        self.trace_prefill_boundary_for_probe("decoder_tail")?;
        }
        self.last_append_metrics = Some(AppendMetrics {
            live_rows: done,
            physical_rows: *widths.iter().max().unwrap_or(&128),
            phase: format!("WideExtend{widths:?}"),
            captured_new_graph,
            engram_split14: false,
            engram_split14_adaptive: false,
            encoder_graph_chunks,
            lookahead: lookahead_stages,
            stall_counters,
            native_fp32_rows: None,
            layer20_deferred_chunks: (layer20_deferred>0).then_some(layer20_deferred),
            stages_ms: marks
                .windows(2)
                .map(|pair| {
                    (
                        pair[1].0,
                        pair[1].1.duration_since(pair[0].1).as_secs_f64() * 1000.,
                    )
                })
                .collect(),
            total_ms: marks.last().unwrap().1.duration_since(begin).as_secs_f64() * 1000.,
        });
        Ok(PrefillProgress{done,remaining_chunks:chunks[widths.len()..].to_vec()})
    }
    /// One committed encoder chunk (layers0..20 plus encoder-tail publication)
    /// on this extend profile, as its own write transaction.
    /// Both Engram layers' row lookups (NVMe-bound on cache misses) run
    /// concurrently; uploads stay on this CUDA-context thread.
    fn prepare_engram_rows(&mut self, hashes: &[[[u64; 24]; 2]]) -> Result<()> {
        #[cfg(target_os="linux")]
        let paired=self.paired_engram_io_enabled();
        #[cfg(target_os="linux")]
        let adaptive=paired&&self.rows==32&&self.paired_engram32_adaptive;
        #[cfg(not(target_os="linux"))]
        let paired=false;
        let first: Vec<[u64; 24]> = hashes.iter().map(|r| r[0]).collect();
        let second: Vec<[u64; 24]> = hashes.iter().map(|r| r[1]).collect();
        let (rows14, store14) = self.group14.engram_rows()?;
        let (flat14, staging14) = rows14.stage(&second, store14)?;
        let (flat1, staging1) = self.rows1.stage(&first, &self.row_store1)?;
        let store1 = &*self.row_store1;
        #[cfg(target_os="linux")]
        if paired {
            if adaptive {
                let helper=if self.engram32_worker_enabled {self.engram32_worker.as_ref()}else{None};
                RowStore::lookup_pair_warm_or_parallel_on(helper,store1,&flat1,staging1,store14,&flat14,staging14)?;
            }
            else {RowStore::lookup_pair_into(store1,&flat1,staging1,store14,&flat14,staging14)?;}
        }
        if !paired {
            let (looked1, looked14) = std::thread::scope(|scope| {
                let other = scope.spawn(|| store14.lookup_into(&flat14, staging14, 8));
                let here = store1.lookup_into(&flat1, staging1, 8);
                (here, other.join())
            });
            looked1?;
            looked14.map_err(|_| anyhow::anyhow!("Engram14 lookup thread panicked"))??;
        }
        self.rows1.upload_staged(&flat1)?;
        self.group14.engram_rows()?.0.upload_staged(&flat14)
    }
    /// Layer-1 half of prepare_engram_rows (non-paired wide path): same lookup and publication.
    fn prepare_engram_rows1(&mut self, hashes: &[[[u64; 24]; 2]]) -> Result<()> {
        ensure!(!self.paired_engram_io_enabled(), "split Engram staging is for wide chunks");
        let first: Vec<[u64; 24]> = hashes.iter().map(|r| r[0]).collect();
        let (flat1, staging1) = self.rows1.stage(&first, &self.row_store1)?;
        self.row_store1.lookup_into(&flat1, staging1, 8)?;
        self.rows1.upload_staged(&flat1)
    }
    /// Layer-14 half of prepare_engram_rows (non-paired wide path): same lookup and publication.
    fn prepare_engram_rows14(&mut self, hashes: &[[[u64; 24]; 2]]) -> Result<()> {
        ensure!(!self.paired_engram_io_enabled(), "split Engram staging is for wide chunks");
        let second: Vec<[u64; 24]> = hashes.iter().map(|r| r[1]).collect();
        let (rows14, store14) = self.group14.engram_rows()?;
        let (flat14, staging14) = rows14.stage(&second, store14)?;
        store14.lookup_into(&flat14, staging14, 8)?;
        self.group14.engram_rows()?.0.upload_staged(&flat14)
    }
    /// config::engram_early_submit: request, without waiting, every table row of this verification that the row
    /// cache does not hold (table 14's too unless its rows are read behind layers 1-13). One submission a table;
    /// a read-ahead still pending on a store (the anchor's) is published first.
    fn submit_engram_rows(&mut self, hashes: &[[[u64; 24]; 2]], first_table_only: bool) -> Result<()> {
        let first: Vec<u64> = hashes.iter().flat_map(|r| r[0]).collect();
        self.row_store1.read_ahead_submit(&first)?;
        if first_table_only { return Ok(()); }
        let second: Vec<u64> = hashes.iter().flat_map(|r| r[1]).collect();
        self.group14.engram_store()?.read_ahead_submit(&second)
    }
    /// config::engram_anchor_prefetch: the six-row verify owner's next verification starts at `anchor`, whose table
    /// rows depend only on the committed history and that token. Request them now, without waiting; the lookup of
    /// the verification finds them in the row cache. A hint: nothing here changes what the lookup returns.
    pub(crate) fn prefetch_anchor_engram(&mut self, anchor: u32) -> Result<()> {
        if !self.engram_overlap || self.rows != 6 || self.source_phase != Phase::Verify || self.pending.is_some()
            || self.batched_rows.is_some() || !self.paired_engram_io_enabled() { return Ok(()); }
        // At the very end of the context window the hash preparation refuses a row; the lookup will say so itself.
        let Ok(batch) = self.history.prepare_padded(&[anchor], 1, &self.hashes) else { return Ok(()); };
        let row = batch.rows[0];
        self.row_store1.read_ahead_submit(&row[0])?;
        self.group14.engram_store()?.read_ahead_submit(&row[1])
    }
    /// Diagnostic cache reset only: invalidate immutable weight-row cache tags,
    /// never KV/model state. Every lookahead task is scoped and already joined.
    pub(crate) fn clear_engram_cache_for_probe(&self, stream: &Stream) -> Result<()> {
        ensure!(self.pending.is_none() && self.lease.is_current()
            && self.cuda.owns_stream(stream) && stream.raw() as usize == self.stream,
            "Engram cache reset requires the idle current owner");
        stream.synchronize()?;
        self.row_store1.clear_for_probe()?;
        self.group14.engram_store()?.clear_for_probe()
    }
    #[cfg(target_os="linux")]
    pub(crate) fn evict_engram_file_pages_for_probe(&self,stream:&Stream)->Result<()> {
        ensure!(self.pending.is_none() && self.lease.is_current()
            && self.cuda.owns_stream(stream) && stream.raw() as usize==self.stream,
            "Engram file-cache eviction requires the idle current owner");
        stream.synchronize()?;
        self.row_store1.evict_file_pages_for_probe()?;
        self.group14.engram_store()?.evict_file_pages_for_probe()
    }
    /// Returns when host preparation finished and GPU enqueue began.
    fn encoder_chunk(
        &mut self,
        chunk: &[u32],
        lookahead: Option<&[u32]>,
        cache: &crate::decoder_tail::Cache,
        comm: &Roce,
        stream: &Stream,
        graph: bool,
        trace_phases: bool,
    ) -> Result<(Instant, bool, Option<LookaheadMetrics>)> {
        use crate::prefill_stall::PhaseTrace;
        let mut trace = trace_phases.then(PhaseTrace::new);
        let rows = self.rows;
        ensure!(
            crate::config::extend_rows(rows) && !chunk.is_empty() && chunk.len() <= rows
                && self.comm_instance == Some(comm.instance())
                && self.cuda.owns_stream(stream) && self.stream == stream.raw() as usize,
            "encoder chunk width"
        );
        // One bounded capture per profile; partial chunks retain their eager
        // live-row collective count. No arbitrary-length graph cache grows.
        let graph = graph && chunk.len() == rows;
        ensure!(!graph || self.encoder_graph_live.is_none_or(|n| n == rows),
            "encoder graph collective identity changed");
        let captured = graph && self.encoder_graph.is_none();
        self.lease.begin()?;
        let previous = self.history.position as usize;
        let end = previous + chunk.len();
        let p = Plan::append_with_layout(previous as u64, chunk.len(), rows, self.context, self.swa_layout)?;
        let mut padded = vec![2u32; rows];
        padded[..chunk.len()].copy_from_slice(chunk);
        // Diagnostic split of the serial host preparation (CANADA_SCRATCH_CLAUDE_HOST_TIMING=1).
        // 1 = one line a chunk on every rank; 2 = also wait for layer 0 once this chunk's layer-1 rows are staged and
        // print how long that took (0 = the GPU was already idle: the reads were exposed). Mode 2 delays the next
        // launches by that wait, so it is a probe, not a serving setting.
        let host_timing_mode=std::env::var("CANADA_SCRATCH_CLAUDE_HOST_TIMING").ok().and_then(|v|v.parse::<u8>().ok()).unwrap_or(0);
        let host_timing=host_timing_mode>=1;
        let (misses1_before,misses14_before)=if host_timing {
            (self.row_store1.misses(),self.group14.engram_store()?.misses())
        } else {(0,0)};
        let (mut rows14_ms,mut layer0_wait_ms)=(0.0f64,-1.0f64);
        let behind_l0 = engram_behind_l0_enabled() && !(graph && !captured);
        let mut batch = None;
        let host_started=Instant::now();
        PhaseTrace::mark(&mut trace, "plan");
        if !behind_l0 {
            let prepared_batch = self.history.prepare_padded(&padded, chunk.len(), &self.hashes)?;
            PhaseTrace::mark(&mut trace, "history");
            self.prepare_engram_rows(&prepared_batch.rows)?;
            PhaseTrace::mark(&mut trace, "engram_both");
            batch = Some(prepared_batch);
        }
        let host_engram=Instant::now();
        self.io.update_tokens(&mut self.tokens, self.history.position as usize,
            &padded.iter().copied().map(i64::from).collect::<Vec<_>>(),
            chunk.len(),
        )?;
        self.step.update(&p, stream)?;
        comm.select_decode_reduction(false, stream)?;
        comm.select_prefill_rows(chunk.len(), rows, stream)?;
        stream.join_default()?;
        if let Some(wide)=&self.wide {wide.prepare_moe_scratch(stream)?;}
        let partial_wide=crate::config::partial_wide_width(rows)&&chunk.len()<rows;
        let _partial_rows=crate::fp8_linear::LivePrefillRowsScope::new(
            self.prefill_partial_dense_rows && rows==512 && (129..512).contains(&chunk.len())
                ||self.prefill_partial1024_rows>0&&rows==1024&&(513..1024).contains(&chunk.len())
                ||self.prefill_partial2048_dense&&rows==2048&&(1025..2048).contains(&chunk.len())
                ||self.prefill_partial_wide&1!=0&&partial_wide,
            self.prefill_partial_moe_rows && rows==512 && (129..512).contains(&chunk.len())
                ||self.prefill_partial1024_rows>1&&rows==1024&&(513..1024).contains(&chunk.len())
                ||self.prefill_partial_wide&2!=0&&partial_wide,
            chunk.len(),rows,stream)?;
        let _mhc_rows=crate::mhc::ActiveRowsScope::new(
            self.prefill_mhc_live_rows&&rows==512&&(129..512).contains(&chunk.len())
                ||self.prefill_wide_mhc_rows&&((rows==1024&&(513..1024).contains(&chunk.len()))
                    ||(rows==2048&&(1025..2048).contains(&chunk.len()))),
            // Normalization still reduces every channel of each live token.
            // Sequence-parallel shards, full graphs and decoder replay retain
            // their original extent because their capacity/phase do not match.
            chunk.len(),rows,stream,self.prefill_mhc_live_norm&&rows==512
                ||self.prefill_wide_mhc_norm&&[1024,2048].contains(&rows))?;
        let prepared = Instant::now();
        PhaseTrace::mark(&mut trace, "setup_and_upload");
        let mut behind_ms = 0.0;
        #[cfg(feature="storage-fault-injection")]
        let layer0_trace=end==16896&&std::env::var("CANADA_SCRATCH_PREFILL_LAYER0_TRACE").as_deref()==Ok("1");
        #[cfg(not(feature="storage-fault-injection"))]
        let layer0_trace=false;
        if layer0_trace {
            // A bounded eager diagnostic captures row-arena values before
            // layer1 can overwrite them. Later compare final logits with the
            // ordinary captured path to detect diagnostic perturbation.
            if batch.is_none() {
                let prepared_batch = self.history.prepare_padded(&padded, chunk.len(), &self.hashes)?;
                self.prepare_engram_rows(&prepared_batch.rows)?;
                batch = Some(prepared_batch);
            }
            self.enqueue_initial(comm,stream)?;
            #[cfg(feature="storage-fault-injection")]
            self.dump_prefill_layer0_trace(previous,chunk,stream)?;
            self.enqueue_encoder_after_initial(comm,stream)?;
        }else if graph && !captured {
            unsafe { self.encoder_graph.as_ref().unwrap().launch(stream) }?;
        } else if behind_l0 {
            // Embedding and layer 0 read no Engram input; the host stages this chunk's rows
            // (mapped inputs, published before any consumer launch) while they execute.
            self.enqueue_initial(comm, stream)?;
            PhaseTrace::mark(&mut trace, "enqueue_initial");
            let behind_started = Instant::now();
            let prepared_batch = self.history.prepare_padded(&padded, chunk.len(), &self.hashes)?;
            PhaseTrace::mark(&mut trace, "history");
            if engram_split14_enabled() && !self.engram_projection_active {
                // Layer-14 rows have no consumer before layer 14: stage them behind layers 1-13.
                self.prepare_engram_rows1(&prepared_batch.rows)?;
                PhaseTrace::mark(&mut trace, "engram1");
                behind_ms = behind_started.elapsed().as_secs_f64()*1000.;
                if host_timing_mode>=2 {
                    let waiting=Instant::now();
                    stream.synchronize()?;
                    layer0_wait_ms=waiting.elapsed().as_secs_f64()*1000.;
                }
                self.enqueue_before_engram14(comm, stream)?;
                PhaseTrace::mark(&mut trace, "enqueue_before_14");
                let rows14_started=Instant::now();
                self.prepare_engram_rows14(&prepared_batch.rows)?;
                PhaseTrace::mark(&mut trace, "engram14");
                rows14_ms=rows14_started.elapsed().as_secs_f64()*1000.;
                batch = Some(prepared_batch);
                self.enqueue_encoder_from_engram14(comm, stream)?;
            } else {
                self.prepare_engram_rows(&prepared_batch.rows)?;
                PhaseTrace::mark(&mut trace, "engram_both");
                batch = Some(prepared_batch);
                behind_ms = behind_started.elapsed().as_secs_f64()*1000.;
                self.enqueue_encoder_after_initial(comm, stream)?;
            }
        } else {
            self.enqueue_encoder(comm, stream)?;
        }
        let batch = batch.context("encoder chunk Engram batch")?;
        PhaseTrace::mark(&mut trace, "enqueue_remaining");
        if host_timing {
            eprintln!("claude host rank{} rows={rows} live={} serial_engram_ms={:.3} rest_ms={:.3} behind_l0_engram_ms={:.3} rows14_ms={:.3} layer0_wait_ms={:.3} misses1={} misses14={}",
                self.rank, chunk.len(), host_engram.duration_since(host_started).as_secs_f64()*1000.,
                prepared.duration_since(host_engram).as_secs_f64()*1000., behind_ms, rows14_ms, layer0_wait_ms,
                self.row_store1.misses()-misses1_before, self.group14.engram_store()?.misses()-misses14_before);
        }
        let lookahead_metrics=if let Some(next) = lookahead {
            // Clone only handles on the CUDA thread. Worker closures borrow
            // the Sync stores/metadata, never an Rc, CUDA context or model state.
            let store1 = self.row_store1.clone();
            let store14 = self.group14.engram_store()?;
            let hashes = &self.hashes;
            let history = self.history.clone();
            let (store1, store14) = (&*store1, &*store14);
            let started=Instant::now();
            Some(std::thread::scope(|scope| -> Result<LookaheadMetrics> {
                let worker = scope.spawn(move || -> Result<f64> {
                    let worker_started=Instant::now();
                    let rows = history.preview_known_input_rows(chunk, next, hashes)?;
                    let first: Vec<u64> = rows.iter().flat_map(|row| row[0]).collect();
                    let second: Vec<u64> = rows.iter().flat_map(|row| row[1]).collect();
                    std::thread::scope(|scope| -> Result<()> {
                        let other = scope.spawn(|| store14.prefetch(&second));
                        let first_result = store1.prefetch(&first);
                        let second_result = other.join().map_err(|_| anyhow::anyhow!("Engram14 prefetch panicked"));
                        first_result?;
                        second_result??;
                        Ok(())
                    })?;
                    Ok(worker_started.elapsed().as_secs_f64()*1000.)
                });
                let executed = (|| -> Result<()> {
                    self.wait_encoder_chunk(stream)?;
                    PhaseTrace::mark(&mut trace, "wait_stream");
                    comm.health()?;
                    self.check_candidates()?;
                    ensure!(cache.error()? == 0, "prefill encoder publication fault");
                    Ok(())
                })();
                PhaseTrace::mark(&mut trace, "check_gpu");
                // Drain the bounded I/O task even if the GPU failed. Neither
                // path may leave background work behind a released writer.
                let gpu_wait_ms=started.elapsed().as_secs_f64()*1000.;
                let join_started=Instant::now();
                let warmed = worker.join().map_err(|_| anyhow::anyhow!("Engram prefetch panicked"));
                let join_wait_ms=join_started.elapsed().as_secs_f64()*1000.;
                PhaseTrace::mark(&mut trace, "join_lookahead");
                executed?;
                let worker_ms=warmed??;
                Ok(LookaheadMetrics {next_rows:next.len(),worker_ms,gpu_wait_ms,join_wait_ms})
            })?)
        } else {
            self.wait_encoder_chunk(stream)?;
            PhaseTrace::mark(&mut trace, "wait_stream");
            comm.health()?;
            self.check_candidates()?;
            ensure!(cache.error()? == 0, "prefill encoder publication fault");
            PhaseTrace::mark(&mut trace, "check_gpu");
            None
        };
        self.io.audit_input_images(&self.tokens, previous, chunk, stream)?;
        if captured {
            // Record after one successful eager execution, for FUTURE chunks.
            // Do not rerun this chunk: a warmup/replay pair could overwrite the
            // incoming SWA prefix or compression carry needed by its first rows.
            // Capture records operations without executing them or committing
            // history; all buffers and vendor paths are already initialized.
            self.encoder_graph = Some(stream.capture(|| self.enqueue_encoder(comm, stream))?);
            self.encoder_graph_live = Some(rows);
        }
        self.history.commit(batch, chunk.len())?;
        cache.commit(end)?;
        self.lease.finish_write()?;
        self.trace_prefill_boundary_for_probe("encoder_chunk")?;
        PhaseTrace::mark(&mut trace, "commit");
        if let Some(trace) = trace {
            trace.emit(self.rank, previous, chunk.len(), rows);
        }
        Ok((prepared, captured, lookahead_metrics))
    }
    /// Diagnostic copy of the last prompt row's full FP32 vocabulary logits.
    pub(crate) fn last_prompt_logit_digest(&self)->Result<String> {
        use sha2::{Digest,Sha256};
        let logits=self.logits()?;
        ensure!(logits.len()>=self.last_live*129280,"last-row logit geometry");
        let bytes=unsafe{self.cuda.read_device(logits.ptr()+((self.last_live-1)*129280*4)as u64,129280*4)}?;
        Ok(format!("{:x}",Sha256::digest(bytes)))
    }
    /// R2 diagnostic only, outside all prefill/verification timing and captures.
    pub(crate) fn tail_mask_statistics_for_probe(&self,module:&Module,stream:&Stream)->Result<serde_json::Value> {
        ensure!(std::env::var("CANADA_SCRATCH_LONG_RETAINED_PROBE").as_deref()==Ok("1")
            && self.lease.is_current() && self.pending.is_none() && self.rows==128
            && self.source_phase==Phase::Extend && (1..=128).contains(&self.last_live)
            && self.history.position>=65536 && self.cuda.owns_stream(stream)
            && self.stream==stream.raw()as usize,"tail mask diagnostic owner");
        let metrics=self.append_metrics()?.ok_or_else(||anyhow::anyhow!("missing append metrics"))?;
        let replayed=metrics.phase.starts_with("WideExtend");
        let mask=if replayed {self.replay.as_ref().and_then(|r|r.candidate_mask.as_ref())}
            else{self.late.source().query().candidate_mask()}
            .ok_or_else(||anyhow::anyhow!("tail candidate mask absent"))?;
        let blocks=self.context.div_ceil(crate::candidates::BLOCK);
        ensure!(mask.len()==128*blocks,"tail mask geometry");
        let mut stats=self.cuda.allocate::<u32>(162)?;stats.zero()?;
        let(mut mask_ptr,mut block_count,mut live,mut start,mut output)=
            (mask.ptr(),blocks as i32,self.last_live as i32,(self.history.position-self.last_live as u64)as i32,stats.ptr());
        stream.join_default()?;
        for(name,grid)in[("ca_sparse_mask_union_stats",blocks.div_ceil(256)),("ca_sparse_mask_row_stats",self.last_live)] {
            let mut args:[*mut std::ffi::c_void;5]=[
                (&mut mask_ptr as *mut u64).cast(),(&mut block_count as *mut i32).cast(),
                (&mut live as *mut i32).cast(),(&mut start as *mut i32).cast(),(&mut output as *mut u64).cast()];
            unsafe{module.function(name)?.launch([grid as u32,1,1],[256,1,1],0,stream,&mut args)?;}
        }
        let mut cursor=132;let mut groups=Vec::new();
        for size in [8usize,16,32,64] {
            let count=self.last_live.div_ceil(size);let mut group_rows=size as i32;
            let mut out=stats.ptr()+(cursor*4)as u64;
            let mut args:[*mut std::ffi::c_void;6]=[
                (&mut mask_ptr as *mut u64).cast(),(&mut block_count as *mut i32).cast(),
                (&mut live as *mut i32).cast(),(&mut start as *mut i32).cast(),
                (&mut group_rows as *mut i32).cast(),(&mut out as *mut u64).cast()];
            unsafe{module.function("ca_sparse_mask_group_unions")?.launch(
                [blocks.div_ceil(256)as u32,count as u32,1],[256,1,1],0,stream,&mut args)?;}
            groups.push((size,cursor,count));cursor+=count;
        }
        stream.synchronize()?;
        let counts=stats.download()?;
        ensure!(counts[2]==0 && counts[3]==0 && counts[1]==self.last_live as u32*2048
            && counts[4..4+self.last_live].iter().all(|&n|n==2048)
            && counts[4+self.last_live..132].iter().all(|&n|n==0)
            && (2048..=counts[1]).contains(&counts[0])
            && counts[0]as usize<=(self.history.position as usize).div_ceil(8),"tail mask statistics invariants: {counts:?}");
        let mut group_records=Vec::new();
        for(size,offset,count)in groups {
            let unions=&counts[offset..offset+count];
            ensure!(unions.iter().all(|&n|(2048..=counts[0]).contains(&n)),"subgroup union bounds");
            group_records.push(serde_json::json!({"group_rows":size,"union_blocks":unions}));
        }
        Ok(serde_json::json!({"live_rows":self.last_live,"replayed_tail":replayed,"context_capacity":self.context,
            "frontier":self.history.position,"union_blocks":counts[0],"row_block_counts":&counts[4..4+self.last_live],
            "selected_row_block_pairs":counts[1],"invalid_cells":counts[2],"outside_window_cells":counts[3],"subgroup_unions":group_records,
            "scope":"GPU reductions of existing source20 masks; no model execution or changed selection."}))
    }
    /// Untimed, bounded localization for the isolated tiny encoder diagnostic.
    /// Last-row intermediates distinguish encoder state from decoder execution;
    /// relevant compressed pages and retained masks are included explicitly.
    pub(crate) fn tiny_encoder_trace(&self)->Result<serde_json::Value> {
        use sha2::{Digest,Sha256};
        ensure!(std::env::var("CANADA_SCRATCH_TINY_ENCODER_DIAGNOSTIC").as_deref()==Ok("1")
            && self.rows==128 && self.position()==586 && self.last_live==128,
            "tiny encoder trace needs the exact diagnostic frontier");
        let hash=|ptr:u64,bytes:usize|->Result<String>{
            Ok(format!("{:x}",Sha256::digest(unsafe{self.cuda.read_device(ptr,bytes)}?)))
        };
        let mut layers=Vec::new();
        for (id,layer)in self.layer_owners().iter().enumerate() {
            let mut fields=serde_json::Map::new();
            for(name,ptr,bytes)in [
                ("attn_input",layer.attn_boundary().input().ptr(),128*5120*2),
                ("qnorm",layer.attention().normalized_lora().ptr(),128*1280*2),
                ("attn_reduced",layer.attn_reduced().ptr(),128*5120*2),
                ("ffn_input",layer.ffn_boundary().input().ptr(),128*5120*2),
                ("routed",layer.routed().ptr(),layer.routed().bytes()),
                ("residual",layer.output().ptr(),128*20480*2),
                ("pre",layer.pre().ptr(),128*4*4)] {
                fields.insert(name.into(),serde_json::json!(hash(ptr+(127*(bytes/128))as u64,bytes/128)?));
                fields.insert(format!("all_{name}"),serde_json::json!(hash(ptr,bytes)?));
            }
            if (21..=25).contains(&id) {
                for(name,ptr,bytes,stride)in layer.attention().taps() {
                    let value=if stride==0 {hash(ptr,bytes)?}else{
                        let width=bytes/128;
                        let raw=unsafe{self.cuda.read_device(ptr,127*stride+width)}?;
                        let mut digest=Sha256::new();
                        for row in 0..128 {digest.update(&raw[row*stride..row*stride+width]);}
                        format!("{:x}",digest.finalize())
                    };
                    fields.insert(format!("attention_{name}"),serde_json::json!(value));
                }
                let cache=&self.late.caches()[id-20];
                fields.insert("swa_cache".into(),serde_json::json!(hash(cache.data().ptr(),cache.data().bytes())?));
                if let Some(index)=self.late.index(id) {
                    fields.insert("index_ids".into(),serde_json::json!(hash(index.candidates().ids().ptr(),index.candidates().ids().bytes())?));
                    fields.insert("index_lengths".into(),serde_json::json!(hash(index.candidates().lengths().ptr(),index.candidates().lengths().bytes())?));
                }
            }
            layers.push(serde_json::json!({"layer":id,"fields":fields}));
        }
        let source=self.late.source().cache();
        let pages=(COMPRESSED_OFFSET+586).div_ceil(source.page_tokens());
        let source20=hash(source.data().ptr(),pages*source.page_bytes())?;
        let replay=self.replay.as_ref().ok_or_else(||anyhow::anyhow!("tiny diagnostic replay missing"))?;
        let mask=replay.candidate_mask.as_ref().map(|m|hash(m.ptr(),m.bytes())).transpose()?;
        Ok(serde_json::json!({"layers":layers,"source20_pages":source20,"replay_mask":mask,
            "replay_hidden":hash(replay.hidden.ptr(),replay.hidden.bytes())?,
            "replay_pre":hash(replay.pre.ptr(),replay.pre.bytes())?,
            "replay_ids":hash(replay.candidates.ids().ptr(),replay.candidates.ids().bytes())?,
            "replay_lengths":hash(replay.candidates.lengths().ptr(),replay.candidates.lengths().bytes())?,
            "step_swa_ids":hash(self.step.candidates.ids().ptr(),self.step.candidates.ids().bytes())?,
            "step_swa_lengths":hash(self.step.candidates.lengths().ptr(),self.step.candidates.lengths().bytes())?,
            "logits":hash(self.io.logits().ptr()+127*129280*4,129280*4)?}))
    }
    pub(crate) fn last_prompt_logits(&self) -> Result<Vec<f32>> {
        let logits = self.logits()?;
        let width = 129280;
        ensure!(logits.len() >= self.last_live * width, "logit row geometry");
        let bytes = unsafe {
            self.cuda.read_device(
                logits.ptr() + ((self.last_live - 1) * width * 4) as u64,
                width * 4,
            )
        }?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    /// New encoder rows plus a fresh qualified128-row decoder tail. The output
    /// becomes publishable on `medium`; no accepted history is committed early.
    pub fn append_replayed(
        &mut self,
        medium: &mut Self,
        ids: &[u32],
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            self.rows == 6
                && medium.rows == 128
                && self.source_phase == Phase::Extend
                && medium.source_phase == Phase::Extend
                && self.lease.is_current()
                && self.history.position >= 128
                && self.prefill_last_head
                && medium.prefill_last_head
                && self.shared_state_report(medium)["all_weights_and_causal_storage_shared"]
                    == true
                && self.cuda.owns_stream(stream)
                && stream.raw() as usize == self.stream
                && comm.rank() == self.rank
                && self.comm_instance == Some(comm.instance())
                && medium.comm_instance == Some(comm.instance()),
            "decoder replay profile/state/stream admission"
        );
        ensure!(
            (1..=6).contains(&ids.len()) && ids.iter().all(|&v| v < 129280),
            "decoder replay token domain"
        );
        let cache = self
            .encoder_cache
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("encoder cache not enabled"))?
            .clone();
        ensure!(
            medium
                .encoder_cache
                .as_ref()
                .is_some_and(|c| Rc::ptr_eq(c, &cache))
                && medium.replay.is_some(),
            "decoder replay cache owner"
        );
        self.last_live = 0;
        self.last_append_metrics = None;
        self.carry_saved_start = None;
        medium.last_live = 0;
        medium.last_append_metrics = None;
        let p = Plan::append_with_layout(
            self.history.position,
            ids.len(),
            self.rows,
            self.context,
            self.swa_layout,
        )?;
        let mut padded = vec![2u32; self.rows];
        padded[..ids.len()].copy_from_slice(ids);
        let batch = self.history.prepare_padded(&padded, ids.len(), &self.hashes)?;
        self.prepare_engram_rows(&batch.rows)?;
        self.io.update_tokens(&mut self.tokens, self.history.position as usize,
            &padded.iter().copied().map(i64::from).collect::<Vec<_>>(),
            ids.len(),
        )?;
        self.step.update(&p, stream)?;
        let end = self.history.position as usize + ids.len();
        let start = end - 128;
        let mut tail =
            Plan::append_with_layout(start as u64, 128, 128, self.context, self.swa_layout)?;
        tail.floor_swa_at(start)?;
        medium.step.update(&tail, stream)?;
        cache.prepare(
            medium.replay.as_mut().unwrap(),
            self.history.position as usize,
            end,
        )?;
        comm.select_decode_reduction(false, stream)?;
        comm.select_prefill_rows(ids.len(), self.rows, stream)?;
        if self.encoder_graph_live != Some(ids.len()) {
            self.encoder_graph = None;
            self.encoder_graph_live = None;
        }
        stream.join_default()?;
        self.lease.begin()?;
        if self.graph_replay {
            if self.encoder_graph.is_none() {
                self.enqueue_encoder(comm, stream)?;
                stream.synchronize()?;
                comm.health()?;
                self.encoder_graph = Some(stream.capture(|| self.enqueue_encoder(comm, stream))?);
                self.encoder_graph_live = Some(ids.len());
            }
            unsafe { self.encoder_graph.as_ref().unwrap().launch(stream) }?;
        } else {
            self.enqueue_encoder(comm, stream)?;
        }
        comm.select_prefill_rows(128, 128, stream)?;
        medium.execute_tail(comm, stream, medium.graph_replay)?;
        stream.synchronize()?;
        comm.health()?;
        self.check_candidates()?;
        medium.check_candidates()?;
        ensure!(cache.error()? == 0, "encoder tail replay address/tag fault");
        self.history.commit(batch, ids.len())?;
        cache.commit(end)?;
        self.last_live = 0;
        self.last_append_metrics = None;
        self.carry_saved_start = None;
        self.lease.finish_write()?;
        medium.handoff_from(self)?;
        medium.last_live = 128;
        Ok(())
    }
    /// The sole tail-graph owner has stable128-row buffers, NCCL row count,
    /// communicator and stream. Reset/handoff changes metadata in those buffers.
    /// Warmup rewrites only this tail's SWA slots; it does not commit history or
    /// advance encoder compression. Capture itself executes no model work.
    fn execute_tail(&mut self, comm: &Roce, stream: &Stream, graph: bool) -> Result<bool> {
        ensure!(self.rows == 128 && self.source_phase == Phase::Extend
            && self.prefill_last_head && self.replay.is_some()
            && self.comm_instance == Some(comm.instance())
            && self.cuda.owns_stream(stream) && self.stream == stream.raw() as usize
            && (!graph || self.graph_replay), "decoder-tail graph owner contract");
        // These host choices are capture identity, never device-variable shape.
        comm.select_decode_reduction(false, stream)?;
        comm.select_prefill_rows(128, 128, stream)?;
        if !graph {
            self.enqueue_tail(comm, stream)?;
            return Ok(false);
        }
        let captured = self.decoder_graph.is_none();
        if captured {
            self.enqueue_tail(comm, stream)?;
            stream.synchronize()?;
            comm.health()?;
            self.check_candidates()?;
            ensure!(self.encoder_cache.as_ref().unwrap().error()? == 0,
                "decoder-tail warmup address/tag fault");
            self.decoder_graph = Some(stream.capture(|| self.enqueue_tail(comm, stream))?);
            if self.prefill_capture_once {
                return Ok(true);
            }
        }
        unsafe { self.decoder_graph.as_ref().unwrap().launch(stream) }?;
        Ok(captured)
    }
    fn enqueue_tail(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        let cache = self
            .encoder_cache
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decoder cache absent"))?;
        let replay = self
            .replay
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decoder replay plan absent"))?;
        cache.gather(replay, stream)?;
        self.late.run_decoder(
            Some((
                &replay.hidden,
                &replay.pre,
                &replay.candidates,
                replay.candidate_mask.as_ref(),
            )),
            &self.step.compressed_positions,
            &self.step.compressed_locations,
            &self.step.candidates,
            &replay.tokens,
            &self.step.m20,
            comm,
            stream,
        )?;
        self.io
            .run_head_last(self.late.output(), self.late.pre(), &replay.tokens, stream)?;
        self.io.gather_logits_last(comm, &replay.tokens, stream)
    }
    pub fn measure_replayed(
        &mut self,
        medium: &mut Self,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<serde_json::Value> {
        ensure!(
            self.lease.is_current()
                && medium.lease.is_current()
                && self.encoder_graph.is_some()
                && medium.decoder_graph.is_some()
                && self.comm_instance == Some(comm.instance())
                && medium.comm_instance == Some(comm.instance())
                && self.shared_state_report(medium)["all_weights_and_causal_storage_shared"]
                    == true
                && stream.raw() as usize == self.stream,
            "replay measurement admission"
        );
        let before = medium.valid_output_digests()?;
        let events = (0..100)
            .map(|_| Ok((self.cuda.event()?, self.cuda.event()?)))
            .collect::<Result<Vec<_>>>()?;
        medium.lease.begin()?;
        if let Some(wide)=&self.wide {wide.prepare_moe_scratch(stream)?;}
        let encoder = self.encoder_graph.as_ref().unwrap();
        let decoder = medium.decoder_graph.as_ref().unwrap();
        for _ in 0..100 {
            unsafe {
                encoder.launch(stream)?;
                decoder.launch(stream)?;
            }
        }
        for (a, b) in &events {
            a.record(stream)?;
            unsafe {
                encoder.launch(stream)?;
                decoder.launch(stream)?;
            }
            b.record(stream)?;
        }
        stream.synchronize()?;
        comm.health()?;
        self.check_candidates()?;
        medium.check_candidates()?;
        ensure!(
            self.encoder_cache.as_ref().unwrap().error()? == 0,
            "replay benchmark cache fault"
        );
        let mut samples = events
            .iter()
            .map(|(a, b)| b.elapsed_since(a))
            .collect::<Result<Vec<_>>>()?;
        samples.sort_by(f32::total_cmp);
        medium.lease.finish_read()?;
        let after = medium.valid_output_digests()?;
        Ok(
            serde_json::json!({"encoder_rows":6,"decoder_rows":128,"valid_outputs_unchanged":before==after,"output_before":before,"output_after":after,"samples":100,"warmup_replays":100,
          "gpu_ms":{"p50":samples[49],"p95":samples[94],"p99":samples[98],"max":samples[99]},
          "scope":"Unchanged encoder increment plus qualified bounded decoder-tail replay. Excludes host preparation and SGLang performance acceptance."}),
        )
    }
    fn check_candidates(&self) -> Result<()> {
        ensure!(
            self.source2.error()? == 0
                && self.group8.source().error()? == 0
                && self.group14.source().error()? == 0
                && self.late.source().error()? == 0,
            "target source candidate fault"
        );
        // The encoder-only wide profile owns no decoder read-index layers.
        for id in [24, 28, 32, 36].into_iter().filter(|_| self.wide.is_none()) {
            ensure!(
                self.late
                    .index(id)
                    .ok_or_else(|| anyhow::anyhow!("target index absent"))?
                    .error()?
                    == 0,
                "target read-index candidate fault"
            );
        }
        Ok(())
    }

    fn enqueue_initial_inner(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        self.io.run_input(&self.tokens, comm, stream)?;
        // A sequence-parallel layer 0 takes its rank's rows of the full embedding.
        let shard;
        let input = match self.early[0].sp_rank() {
            Some(rank) => {
                let local = self.io.input().len() / (4 * 20480);
                shard = unsafe { self.io.input().view(rank * local * 20480, local * 20480) }?;
                &shard
            }
            None => self.io.input(),
        };
        self.early[0].run(
            input,
            &self.step.positions,
            &self.step.locations,
            &mut self.early_caches[0],
            &self.step.candidates,
            &self.tokens,
            comm,
            stream,
        )?;
        Ok(())
    }
    fn enqueue_encoder(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        self.enqueue_initial(comm, stream)?;
        self.enqueue_encoder_after_initial(comm, stream)
    }
    fn enqueue_encoder_after_initial(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        self.enqueue_before_engram14(comm,stream)?;
        self.enqueue_encoder_from_engram14(comm,stream)
    }
    fn enqueue_before_engram14_inner(&mut self, comm:&Roce, stream:&Stream)->Result<()> {
        if self.engram_projection_active {
            // Moving this reduction earlier changes the collective order.
            // Rank-local split-I/O graph choices therefore cannot coexist
            // with this schedule. Both row sets must already be prepared.
            ensure!(!self.engram_split14&&!self.engram_split14_adaptive,
                "Engram projection requires a common collective schedule");
            self.group14.prepare_engram_projection_ahead(comm,stream)?;
        }
        let (first, rest) = self.early.split_first_mut().unwrap();
        let (_, caches) = self.early_caches.split_first_mut().unwrap();
        let mut previous_output = first.output();
        let mut previous_pre = first.pre();
        for (offset, (layer, cache)) in rest.iter_mut().zip(caches).enumerate() {
            let id = offset + 1;
            // Sequence-parallel layer 2 takes this rank's rows of layer 1.
            let shard = layer.shard_input(previous_output, previous_pre)?;
            let (input, input_pre) =
                shard.as_ref().map_or((previous_output, previous_pre), |(r, p)| (r, p));
            if id == 1 {
                layer.run_from_previous(
                    input,
                    Some(input_pre),
                    Some(&mut self.rows1),
                    &self.step.positions,
                    &self.step.locations,
                    cache,
                    &self.step.candidates,
                    &self.tokens,
                    comm,
                    stream,
                )?;
            } else if id == 2 {
                layer.run_indexed_from_previous(
                    input,
                    input_pre,
                    &self.step.compressed_positions,
                    &self.step.compressed_locations,
                    cache,
                    &self.step.candidates,
                    &self.tokens,
                    &mut self.source2,
                    &self.step.m2,
                    comm,
                    stream,
                )?;
            } else {
                layer.run_shared_from_previous(
                    input,
                    input_pre,
                    &self.step.compressed_positions,
                    &self.step.compressed_locations,
                    cache,
                    &self.step.candidates,
                    &self.tokens,
                    &self.source2,
                    comm,
                    stream,
                )?;
            }
            previous_output = layer.output();
            previous_pre = layer.pre();
        }
        self.group8.run(
            previous_output,
            previous_pre,
            &self.step.compressed_positions,
            &self.step.compressed_locations,
            &self.step.candidates,
            &self.tokens,
            &self.step.m8,
            comm,
            stream,
        )?;
        Ok(())
    }
    fn enqueue_encoder_from_engram14(&mut self,comm:&Roce,stream:&Stream)->Result<()> {
        // Layer 14 (Engram14 and source14) runs on full rows.
        let (input14, pre14) = match self.sp_full.as_mut() {
            Some((residual, pre)) if self.group8.sequence_parallel()
                && !self.group14.first_sequence_parallel() => {
                unsafe {
                    comm.gather_rows(self.group8.output(), residual, stream)?;
                    comm.gather_rows(self.group8.pre(), pre, stream)?;
                }
                (&*residual, &*pre)
            }
            _ => (self.group8.output(), self.group8.pre()),
        };
        self.group14.run(
            input14,
            pre14,
            &self.step.compressed_positions,
            &self.step.compressed_locations,
            &self.step.candidates,
            &self.tokens,
            &self.step.m14,
            comm,
            stream,
        )?;
        if self.layer20_projection_only {
            // Rows before the request's final 128: keep only the state that later
            // rows, the decoder and decode read; layer 20's output rows are unused.
            ensure!(!self.layer20_pending && self.source_phase != Phase::Verify,
                "deferred layer20 owner");
            self.late.run_encoder_projection(
                self.group14.output(),
                self.group14.pre(),
                &self.step.compressed_positions,
                &self.step.compressed_locations,
                &self.step.m20,
                comm,
                stream,
            )?;
            self.layer20_pending = true;
            return Ok(());
        }
        self.late.run_encoder(
            self.group14.output(),
            self.group14.pre(),
            &self.step.compressed_positions,
            &self.step.compressed_locations,
            &self.step.candidates,
            &self.tokens,
            &self.step.m20,
            comm,
            stream,
        )?;
        self.publish_encoder_tail(comm, stream)
    }
    /// Issue the rest of a deferred layer 20 exactly as run_encoder would have
    /// (same retained buffers, same order), then publish its tail rows.
    fn finish_deferred_layer20(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        ensure!(self.layer20_pending, "no deferred layer20 to finish");
        self.late.finish_encoder_deferred(
            self.group14.output(),
            &self.step.compressed_positions,
            &self.step.candidates,
            &self.tokens,
            &self.step.m20,
            comm,
            stream,
        )?;
        self.layer20_pending = false;
        self.publish_encoder_tail(comm, stream)
    }
    /// A deferred chunk that now ends a turn (cancel or yield): its final rows
    /// feed the decoder tail at this boundary.
    fn finish_deferred_layer20_now(&mut self, comm: &Roce, stream: &Stream,
        cache: &crate::decoder_tail::Cache) -> Result<()> {
        self.lease.begin()?;
        self.finish_deferred_layer20(comm, stream)?;
        stream.synchronize()?;
        comm.health()?;
        self.check_candidates()?;
        ensure!(cache.error()? == 0, "deferred layer20 publication fault");
        self.lease.finish_write()
    }
    /// Inline encoder checkpoint, on the wide profile that has just run `chunk` (all live
    /// rows) from the history `before`. Capture into `aux` what a chunk ending at its row
    /// `end` would have left behind: the pair carry rings, the candidate blocks and encoder
    /// tail of rows [end-128, end), and the host history. Every input is still in place:
    /// the pair projections, layer 20's output, the queries and the gathered keys are
    /// per-chunk buffers and nothing has run since the chunk. No live state changes.
    fn capture_checkpoint_as_of(&mut self, aux: &CheckpointAux, before: &History, chunk: &[u32], end: usize,
        comm: &Roce, stream: &Stream) -> Result<()> {
        ensure!(self.wide.is_some() && self.source_phase == Phase::Extend && self.pending.is_none()
            && self.lease.is_current() && !self.layer20_pending
            && self.late.encoder_sequence_parallel() && tail_window_enabled()
            && self.encoder_cache.is_some() && self.tokens.active_rows() == chunk.len()
            && (128..=chunk.len()).contains(&end)
            && self.history.position == before.position + chunk.len() as u64
            && comm.rank() == self.rank && self.cuda.owns_stream(stream) && stream.raw() as usize == self.stream,
            "inline encoder checkpoint capture admission");
        *aux.captured.borrow_mut() = None;
        self.lease.begin()?;
        aux.tail.clear(stream)?;
        self.source2.publish_ring_as_of(&mut self.step.m2, &aux.rings[0], end, stream)?;
        self.group8.source().publish_ring_as_of(&mut self.step.m8, &aux.rings[1], end, stream)?;
        self.group14.source().publish_ring_as_of(&mut self.step.m14, &aux.rings[2], end, stream)?;
        aux.tail.set_as_of_rows(end, stream)?;
        // A compact mask holds only the newest 128 live rows' blocks: publish this window's.
        let retained = self.late.source().query().candidate_mask_is_compact();
        if retained {
            self.late.source_mut().republish_retained_window(&self.step.m20, &aux.tail.as_of_rows(), end, comm, stream)?;
        }
        {
            let (residual, pre) = self.sp_full.as_mut()
                .context("inline encoder checkpoint needs the full-row gather storage")?;
            let rows = residual.len() / 20480;
            ensure!(rows % 4 == 0 && end <= rows && pre.len() == rows * 4, "inline encoder checkpoint tail geometry");
            let (w0, w1) = (end - 128, end);
            let local = rows / 4;
            let rank = comm.rank();
            let (a, b) = (w0.max(rank * local), w1.min((rank + 1) * local));
            unsafe {
                self.cuda.zero_device(residual.ptr() + (w0 * 40960) as u64, (w1 - w0) * 40960, stream)?;
                if a < b {
                    self.cuda.copy_device(residual.ptr() + (a * 40960) as u64,
                        self.late.encoder_output().ptr() + ((a - rank * local) * 40960) as u64, (b - a) * 40960, stream)?;
                }
                let window = residual.view(w0 * 20480, (w1 - w0) * 20480)?;
                let mut summed = residual.view(w0 * 20480, (w1 - w0) * 20480)?;
                comm.sum_window_bf16(&window, &mut summed, stream)?;
            }
            // `pre` still holds every row from the chunk's own publication.
            aux.tail.publish_as_of(&*residual, &*pre, self.late.source().candidates(), &self.tokens,
                &self.step.compressed_positions, self.late.source().query().candidate_mask(),
                self.late.source().query().candidate_mask_is_compact(), stream)?;
        }
        if retained {
            // Leave the publisher as the chunk left it: the blocks of its own newest rows.
            self.late.source_mut().republish_retained_window(&self.step.m20, self.tokens.live(), chunk.len(), comm, stream)?;
        }
        stream.synchronize()?;
        comm.health()?;
        ensure!(aux.tail.error()? == 0, "inline encoder checkpoint tail fault");
        let mut history = before.clone();
        let batch = history.prepare(&chunk[..end], &self.hashes)?;
        history.commit(batch, end)?;
        *aux.captured.borrow_mut() = Some(history);
        self.lease.finish_read()
    }
    /// The decoder-tail owner's side state for inline encoder checkpoints, built on first use.
    fn checkpoint_aux(&mut self) -> Result<Rc<CheckpointAux>> {
        if self.checkpoint_aux.is_none() {
            ensure!(self.rows == 128 && self.wide.is_none(), "inline checkpoint state belongs to the decoder-tail owner");
            let ring = |source: &Source| source.ring_twin()?.context("pair source without a carry ring");
            self.checkpoint_aux = Some(Rc::new(CheckpointAux {
                tail: self.encoder_cache.as_ref().context("inline checkpoint needs the encoder tail")?.twin()?,
                rings: [ring(&self.source2)?, ring(self.group8.source())?, ring(self.group14.source())?],
                captured: std::cell::RefCell::new(None),
            }));
        }
        Ok(self.checkpoint_aux.clone().unwrap())
    }
    /// A second set of the same side state, for the extra near-end capture of one request.
    fn checkpoint_aux_extra(&mut self) -> Result<Rc<CheckpointAux>> {
        if self.checkpoint_aux_extra.is_none() {
            ensure!(self.rows == 128 && self.wide.is_none(), "inline checkpoint state belongs to the decoder-tail owner");
            let ring = |source: &Source| source.ring_twin()?.context("pair source without a carry ring");
            self.checkpoint_aux_extra = Some(Rc::new(CheckpointAux {
                tail: self.encoder_cache.as_ref().context("inline checkpoint needs the encoder tail")?.twin()?,
                rings: [ring(&self.source2)?, ring(self.group8.source())?, ring(self.group14.source())?],
                captured: std::cell::RefCell::new(None),
            }));
        }
        Ok(self.checkpoint_aux_extra.clone().unwrap())
    }
    /// Ask the next prefill_with_encoder_checkpoint for a second image, `rows` rows past its restore point (inside
    /// the last chunk, 128 rows or more in), of `bytes` bytes; consumed by that call whatever path it takes.
    pub(crate) fn set_extra_checkpoint(&mut self, request: Option<(usize, usize)>) -> Result<()> {
        ensure!(request.is_none() || (self.rows == 128 && self.wide.is_none()), "extra checkpoint belongs to the decoder-tail owner");
        self.extra_checkpoint = request; self.extra_image = None; Ok(())
    }
    /// The second image of the last prefill, if it was asked for, captured and admitted.
    pub(crate) fn take_extra_checkpoint_image(&mut self) -> Option<EncoderPrefixImage> { self.extra_image.take() }
    /// Save an inline-captured image from its side state (the caches' pages are read from the live buffers).
    #[allow(clippy::too_many_arguments)]
    fn save_inline_checkpoint(&mut self, aux: &CheckpointAux, position: usize, budget: usize,
        recycled: Option<crate::state_image::Image>, comm: &Roce, stream: &Stream,
        admit_image: &mut impl FnMut(bool) -> Result<bool>) -> Result<Option<EncoderPrefixImage>> {
        let recycled_bytes = recycled.as_ref().map_or(0, crate::state_image::Image::bytes);
        let history = aux.captured.borrow_mut().take().context("inline encoder checkpoint was not captured")?;
        ensure!(history.position as usize == position, "inline encoder checkpoint frontier");
        self.state_image_guard(comm, stream)?;
        let fits = self.cuda.admission_memory()?.admitted_available
            >= crate::memory::floor_bytes() + 16 * 1024usize.pow(2) + budget - recycled_bytes;
        if !admit_image(fits)? { return Ok(None); }
        let bank_retained = crate::config::bank_encoder_checkpoint_requested();
        let regions = self.state_image_regions_for_layers_mode(history.position as usize, true, true, bank_retained)?;
        let mut allocation = match recycled {
            Some(retired) => retired.try_recycle(&regions, budget, stream)?,
            None => crate::state_image::Image::try_allocate(&self.cuda, &regions, budget, stream)?,
        };
        let fits = allocation.is_some() && self.cuda.admission_memory()?.admitted_available
            >= crate::memory::floor_bytes() + 16 * 1024usize.pow(2);
        if !admit_image(fits)? { return Ok(None); }
        let mut image = allocation.take().ok_or_else(|| anyhow::anyhow!("all-rank image admission differs"))?;
        image.configure_copy_kernel(self.io.core_module())?;
        stream.synchronize()?; comm.health()?; self.lease.begin()?;
        ensure!(self.encoder_cache.as_ref().unwrap().error()? == 0, "encoder checkpoint cache fault");
        let sources = aux.image_sources(&self.cuda)?;
        image.save_with_sources(&regions, &sources, stream)?; comm.health()?; self.lease.finish_read()?;
        Ok(Some(EncoderPrefixImage(StateImage { image: image.into(), owner: self.lease.fork(), history,
            context: self.context, rank: self.rank, stream: self.stream, compact: true },
            bank_retained.then(|| bank_encoder_lineage::Binding::new(self.history.lineage)))))
    }
    /// The chunk plan of an inline encoder checkpoint for `prefix` + `suffix` rows, if the
    /// policy is on and the plan's last chunk can capture the checkpoint: a sequence-parallel
    /// wide chunk that starts 128 rows or more before the checkpoint.
    fn inline_checkpoint_plan(&self, wides: &[Self], prefix: usize, suffix: usize,
        forced: Option<Vec<usize>>) -> Result<Option<Vec<usize>>> {
        self.inline_checkpoint_plan_at(wides, prefix, suffix, forced, self.history.position > 0)
    }
    /// Whether a near-end checkpoint `prefix` rows past a restored encoder image, with `suffix` (< 128) rows
    /// after it, is captured inline. The worker chooses a near-end position only then: without the inline
    /// capture the checkpoint transaction needs a complete 128-row replacement suffix.
    /// The next short restored suffix of prefill_with_encoder_checkpoint runs as the qualified EXTEND
    /// (CANADA_SCRATCH_ENCODER_SHORT_SUFFIX=2); consumed by that call.
    pub(crate) fn set_short_restore_extend(&mut self, extend: bool) -> Result<()> {
        ensure!(!extend || (self.rows == 128 && self.cached_decoder.is_some()), "short restore EXTEND needs the cached floor profile");
        self.short_restore_extend = extend; Ok(())
    }
    pub(crate) fn near_end_checkpoint_admitted(&self, wides: &[Self], prefix: usize, suffix: usize) -> Result<bool> {
        Ok(!self.default_encoder_graph_policy()?.0
            && self.inline_checkpoint_plan_at(wides, prefix, suffix, None, true)?.is_some())
    }
    /// Whether a request of `total` rows past its restore point (`retained`: there is one) with its checkpoint
    /// `prefix` rows in takes the inline path and can also capture a second image `extra` rows in: both in the
    /// plan's last chunk, 128 rows or more into it. The worker asks before it makes room for that image.
    pub(crate) fn extra_checkpoint_admitted(&self, wides: &[Self], prefix: usize, extra: usize, total: usize,
        retained: bool) -> Result<bool> {
        if self.default_encoder_graph_policy()?.0 || extra <= prefix || extra > total || total <= prefix { return Ok(false); }
        Ok(self.inline_checkpoint_plan_at(wides, prefix, total - prefix, None, retained)?
            .is_some_and(|plan| extra >= plan[..plan.len() - 1].iter().sum::<usize>() + 128))
    }
    /// The chunk widths a prefill may plan with: 128 and every wide profile, except the 16,384-row profile while its
    /// elastic extension is unmapped (config::wide16_elastic; its upper arena halves have no memory then).
    fn prefill_widths(&self, wides: &[Self]) -> Vec<usize> {
        let lent = !self.cuda.wide16_configured() || self.cuda.wide16_mapped();
        std::iter::once(128).chain(wides.iter().map(|w| w.rows).filter(|&r| r != 16384 || lent)).collect()
    }
    fn inline_checkpoint_plan_at(&self, wides: &[Self], prefix: usize, suffix: usize,
        forced: Option<Vec<usize>>, retained: bool) -> Result<Option<Vec<usize>>> {
        let requested = match self.checkpoint_inline_for_probe {
            Some(inline) => inline,
            None => crate::config::encoder_checkpoint_inline_requested()?,
        };
        if !requested || prefix < 128 || !tail_window_enabled() {
            return Ok(None);
        }
        let total = prefix + suffix;
        let available: Vec<usize> = self.prefill_widths(wides);
        let plan = match forced {
            Some(plan) => { crate::config::validate_encoder_chunks(total, &plan, &available)?; plan }
            None => {
                let merge = retained && self.prefill_short_merge && wides.iter().any(|w|
                    w.rows == 2048 && w.prefill_partial2048_dense && w.prefill_wide_mhc_rows && w.prefill_wide_mhc_norm);
                crate::config::plan_encoder_append(total, &available, merge)
            }
        };
        let last = *plan.last().context("empty encoder plan")?;
        let start: usize = plan[..plan.len() - 1].iter().sum();
        let admitted = prefix >= start + 128 && wides.iter().any(|w| w.rows == last
            && w.late.encoder_sequence_parallel() && w.sp_full.is_some());
        Ok(admitted.then_some(plan))
    }
    fn publish_encoder_tail(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        let publishing = self.encoder_cache.is_some() && self.source_phase != Phase::Verify;
        let (output20, pre20) = match self.sp_full.as_mut() {
            Some((residual, pre)) if self.late.encoder_sequence_parallel() && tail_window_enabled() => {
                // The bounded publication reads x rows [live-128, live) only; pre is small.
                let rows = residual.len() / 20480;
                let live = self.tokens.active_rows();
                ensure!(rows % 4 == 0 && live <= rows && pre.len() == rows * 4
                    && crate::decoder_tail::bounded_copy_enabled()?, "tail window publication geometry");
                if publishing && live > 0 {
                    let (w0, w1) = (live.saturating_sub(128), live);
                    let local = rows / 4;
                    let rank = comm.rank();
                    let (a, b) = (w0.max(rank * local), w1.min((rank + 1) * local));
                    unsafe {
                        self.cuda.zero_device(residual.ptr() + (w0 * 40960) as u64, (w1 - w0) * 40960, stream)?;
                        if a < b {
                            self.cuda.copy_device(residual.ptr() + (a * 40960) as u64,
                                self.late.encoder_output().ptr() + ((a - rank * local) * 40960) as u64, (b - a) * 40960, stream)?;
                        }
                        let window = residual.view(w0 * 20480, (w1 - w0) * 20480)?;
                        let mut summed = residual.view(w0 * 20480, (w1 - w0) * 20480)?;
                        comm.sum_window_bf16(&window, &mut summed, stream)?;
                        comm.gather_rows(self.late.encoder_pre(), pre, stream)?;
                    }
                }
                (&*residual, &*pre)
            }
            Some((residual, pre)) if self.late.encoder_sequence_parallel() => {
                unsafe {
                    comm.gather_rows(self.late.encoder_output(), residual, stream)?;
                    comm.gather_rows(self.late.encoder_pre(), pre, stream)?;
                }
                (&*residual, &*pre)
            }
            _ => (self.late.encoder_output(), self.late.encoder_pre()),
        };
        if let Some(cache) = &self.encoder_cache
            && publishing
        {
            cache.publish(
                output20,
                pre20,
                self.late.source().candidates(),
                &self.tokens,
                &self.step.compressed_positions,
                self.late.source().query().candidate_mask(),
                self.late.source().query().candidate_mask_is_compact(),
                stream,
            )?;
        }
        Ok(())
    }
    fn enqueue(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        self.enqueue_initial(comm, stream)?;
        self.enqueue_after_initial(comm, stream)
    }
    // The three parts of a forward, each between the layer clock's first and last mark of that part
    // (crate::layer_clock: no effect unless a screen armed it).
    fn enqueue_initial(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        crate::layer_clock::part(stream, 0)?;
        self.enqueue_initial_inner(comm, stream)?;
        crate::layer_clock::end(stream)
    }
    fn enqueue_before_engram14(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        crate::layer_clock::part(stream, 1)?;
        self.enqueue_before_engram14_inner(comm, stream)?;
        crate::layer_clock::end(stream)
    }
    fn enqueue_after_engram14(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        crate::layer_clock::part(stream, 2)?;
        self.enqueue_after_engram14_inner(comm, stream)?;
        crate::layer_clock::end(stream)
    }
    fn enqueue_after_initial(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        self.enqueue_before_engram14(comm,stream)?;
        self.enqueue_after_engram14(comm,stream)
    }
    fn enqueue_after_engram14_inner(&mut self,comm:&Roce,stream:&Stream)->Result<()> {
        self.enqueue_encoder_from_engram14(comm,stream)?;
        let decoder = self.cached_decoder.as_ref().unwrap_or(&self.step);
        // The append bank changes only decoder SWA visibility. Its source20
        // index metadata has identical values, but separately allocated slot
        // IDs would violate the cached key gather's immutable pointer binding.
        // Keep that source metadata on the ordinary narrow owner across both banks.
        let decoder_source=if [6,16].contains(&self.rows) {&self.step.m20}else{&decoder.m20};
        self.late.run_decoder(
            None,
            &decoder.compressed_positions,
            &decoder.compressed_locations,
            &decoder.candidates,
            &self.tokens,
            decoder_source,
            comm,
            stream,
        )?;
        if self.prefill_last_head {
            self.io
                .run_head_last(self.late.output(), self.late.pre(), &self.tokens, stream)?;
            self.io.gather_logits_last(comm, &self.tokens, stream)?;
        } else {
            if self.append_encoded_head&&self.cached_decoder.is_some() {
                ensure!(self.append_last_gather,"final encoded head needs final-only publication");
                self.io.run_append_encoded_last(self.late.output(),self.late.pre(),&self.tokens,stream)?;
            } else if self.rows == 6 && self.io.head4_on() {
                // crate::head4_rescore (an option, off by default; it approximates the model's own output head, see its
                // module documentation): the 4-bit copy of the head, exact rescoring of the largest rows.
                self.io.run_head4(self.late.output(), self.late.pre(), self.tokens.active_rows(), stream)?;
            } else {self.io.run_head(self.late.output(), self.late.pre(), stream)?;}
            if self.append_last_gather&&self.cached_decoder.is_some()&&self.tokens.active_rows()>1 {
                self.io.gather_existing_last_logits(comm,&self.tokens,stream)?;
            } else if !self.compact_sampling_active() {self.io.gather_logits(comm, stream)?;}
        }
        Ok(())
    }

    /// All append rows are valid in verify mode. In last-head prefill mode,
    /// only the most recent append's last row is valid; all others are zero.
    pub fn valid_output_digests(&self) -> Result<serde_json::Value> {
        use sha2::{Digest, Sha256};
        ensure!(
            self.lease.is_current() && self.last_live > 0 && !self.last_head_local_only,
            "output digest needs a completed active result"
        );
        let tensors = [
            (
                "residual",
                self.late.output().ptr(),
                self.last_live * 20480 * 2,
            ),
            ("pre", self.late.pre().ptr(), self.last_live * 4 * 4),
            (
                "logits",
                self.io.logits().ptr() + ((self.last_live - 1) * 129280 * 4) as u64,
                129280 * 4,
            ),
        ];
        let mut result = serde_json::Map::new();
        for (name, ptr, bytes) in tensors {
            let raw = unsafe { self.cuda.read_device(ptr, bytes) }?;
            result.insert(
                name.into(),
                serde_json::json!({"bytes":bytes,"sha256":format!("{:x}",Sha256::digest(raw))}),
            );
        }
        Ok(serde_json::json!({"valid_rows":self.last_live,"taps":result}))
    }
    pub(crate) fn context_capacity(&self) -> usize {
        self.context
    }
    pub fn logits(&self) -> Result<&Buffer<f32>> {
        ensure!(
            self.lease.is_current() && self.history.position > 0 && self.last_live > 0 && !self.last_head_local_only,
            "target logits are not publishable"
        );
        Ok(self.io.logits())
    }
    /// Greedy next-token choice only. Stochastic target sampling and DSpark
    /// remain separate gates. Sampling does not commit a token to history;
    /// the caller must append an accepted token through the normal state path.
    /// Diagnostic only (CANADA_SCRATCH_CLAUDE_LOGIT_DIGEST=1): rank 0 logs the
    /// SHA-256 of the full FP32 logits row a prompt's first token is drawn from.
    fn log_prompt_logit_digest(&self, stream: &Stream) -> Result<()> {
        if self.rank == 0 && std::env::var("CANADA_SCRATCH_CLAUDE_LOGIT_DIGEST").as_deref() == Ok("1") {
            stream.synchronize()?;
            eprintln!("claude prompt logits position={} sha256={}", self.history.position,
                self.last_prompt_logit_digest()?);
        }
        Ok(())
    }
    pub fn greedy_token(&mut self, stream: &Stream) -> Result<u32> {
        ensure!(
            self.lease.is_current()
                && self.history.position > 0
                && self.last_live > 0
                && !self.last_head_local_only
                && self.cuda.owns_stream(stream)
                && stream.raw() as usize == self.stream,
            "target greedy result is not publishable on this stream"
        );
        self.log_prompt_logit_digest(stream)?;
        self.lease.begin()?;
        self.sampler.run(self.io.logits(), stream)?;
        let ids = self.sampler.finish(stream)?;
        self.lease.finish_read()?;
        Ok(ids[self.last_live - 1] as u32)
    }
    /// Sample the next uncommitted token at the model card's T=1, top_p=1.
    /// Position is the absolute predicted output position, including prefill.
    pub(crate) fn sample_token(&mut self, seed: u64, stream: &Stream) -> Result<u32> {
        ensure!(self.lease.is_current() && self.history.position > 0 && self.last_live > 0 && !self.last_head_local_only
            && self.cuda.owns_stream(stream) && stream.raw() as usize == self.stream,
            "target stochastic result is not publishable on this stream");
        self.log_prompt_logit_digest(stream)?;
        self.lease.begin()?;
        let ids = self.categorical.sample(self.io.logits(), self.last_live - 1, 1, 1., seed,
            crate::categorical::TARGET_DOMAIN, self.history.position, stream)?;
        self.lease.finish_read()?;
        self.record_restore_initial_logits_for_probe(self.io.logits(),self.last_live-1)?;
        Ok(ids[0])
    }
    fn record_restore_initial_logits_for_probe(&self,logits:&Buffer<f32>,row:usize)->Result<()> {
        if std::env::var("CANADA_SCRATCH_RESTORE_INITIAL_SCREEN").as_deref()!=Ok("1"){return Ok(());}
        ensure!(self.lease.is_current()&&self.cuda.owns_buffer(logits)
            &&(row+1)*crate::sampling::VOCAB<=logits.len(),"initial diagnostic row/context");
        // The draw has already completed. Retain one bounded CPU byte image;
        // no GPU arithmetic or cache admission is changed. These captures
        // are not performance measurements.
        let bytes=unsafe{self.cuda.read_device(logits.ptr()+(row*crate::sampling::VOCAB*4)as u64,
            crate::sampling::VOCAB*4)}?;
        RESTORE_INITIAL_LOGITS.with(|cell|*cell.borrow_mut()=Some(bytes));Ok(())
    }
    /// Copy the current prompt's final canonical FP32 logit row. The caller
    /// must first admit this optional allocation together with its state image.
    pub(crate) fn snapshot_prompt_logits(&mut self, stream:&Stream)->Result<Buffer<f32>> {
        ensure!(self.lease.is_current() && self.pending.is_none() && self.history.position>0
            && self.last_live>0 && !self.last_head_local_only && self.cuda.owns_stream(stream) && stream.raw()as usize==self.stream,
            "prompt logits require a committed current owner");
        let output=self.cuda.allocate_rows(crate::sampling::VOCAB)?;
        self.lease.begin()?;
        unsafe {self.cuda.copy_device(output.ptr(),
            self.io.logits().ptr()+((self.last_live-1)*crate::sampling::VOCAB*4)as u64,
            output.bytes(),stream)?;}
        stream.synchronize()?;
        self.lease.finish_read()?;
        Ok(output)
    }
    /// A restored prompt image invalidates ordinary scratch logits. Its
    /// separately owned FP32 row is sampled explicitly, at the restored
    /// absolute position, with the new request's seed and normal target domain.
    pub(crate) fn sample_prompt_logits(&mut self,logits:&Buffer<f32>,position:u64,
        seed:u64,stream:&Stream)->Result<u32> {
        ensure!(self.lease.is_current() && self.pending.is_none() && position>0
            && self.history.position==position && self.cuda.owns_buffer(logits)
            && logits.len()==crate::sampling::VOCAB && self.cuda.owns_stream(stream)
            && stream.raw()as usize==self.stream,"cached prompt logit admission");
        self.lease.begin()?;
        let ids=self.categorical.sample(logits,0,1,1.,seed,crate::categorical::TARGET_DOMAIN,position,stream)?;
        self.lease.finish_read()?;
        self.record_restore_initial_logits_for_probe(logits,0)?;
        Ok(ids[0])
    }
    pub fn position(&self) -> u64 {
        self.history.position
    }
    /// Integration measurement only. Restore the actual incoming carry rings
    /// before each replay, and verify all outgoing ring bytes afterward.
    /// This does not supply speculative rollback or complete session parking.
    pub fn measure_current(
        &mut self,
        comm: &Roce,
        stream: &Stream,
        profile_decode: bool,
    ) -> Result<serde_json::Value> {
        ensure!(
            self.lease.is_current()
                && self.history.position > 0
                && self.graphs.get(self.last_live).is_some_and(Option::is_some)
                && self.repeatable_frontier_measurement
                && self
                    .carry_saved_start
                    .is_some_and(|start| start.checked_add(self.last_live as u64)
                        == Some(self.history.position))
                && self.comm_instance == Some(comm.instance())
                && stream.raw() as usize == self.stream,
            "target frontier is not admitted for captured measurement"
        );
        let before = self.valid_output_digests()?;
        let carry = self
            .carry_snapshot
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("incoming carry snapshot absent"))?;
        let carry_before = carry.live_digests()?;
        let events = (0..100)
            .map(|_| Ok((self.cuda.event()?, self.cuda.event()?)))
            .collect::<Result<Vec<_>>>()?;
        self.lease.begin()?;
        let graph = self.graphs[self.last_live].as_ref().unwrap();
        for _ in 0..100 {
            unsafe {
                carry.restore(stream)?;
                graph.launch(stream)?;
            }
        }
        if profile_decode {
            ensure!(
                self.source_phase == Phase::Decode && self.last_live == 1,
                "kernel profiling is limited to the explicit one-token decode diagnostic"
            );
            stream.synchronize()?;
            self.cuda.profiler_start()?;
        }
        for (start, end) in &events {
            unsafe {
                carry.restore(stream)?;
            }
            start.record(stream)?;
            unsafe {
                graph.launch(stream)?;
            }
            end.record(stream)?;
        }
        stream.synchronize()?;
        if profile_decode {
            self.cuda.profiler_stop()?;
        }
        comm.health()?;
        self.check_candidates()?;
        let mut samples = events
            .iter()
            .map(|(a, b)| b.elapsed_since(a))
            .collect::<Result<Vec<_>>>()?;
        samples.sort_by(f32::total_cmp);
        self.lease.finish_read()?;
        let after = self.valid_output_digests()?;
        let carry_after = carry.live_digests()?;
        Ok(
            serde_json::json!({"rows":self.rows,"external_kernel_profile":profile_decode,"valid_outputs_unchanged":before==after,"incoming_carry_restored_before_each_replay":true,"carry_snapshot_bytes":carry.bytes(),"outgoing_carry_unchanged":carry_before==carry_after,"outgoing_carry_before":carry_before,"outgoing_carry_after":carry_after,"output_before":before,"output_after":after,"samples":100,"warmup_replays":100,"gpu_ms":{"p50":samples[49],"p95":samples[94],"p99":samples[98],"max":samples[99]},
            "scope":"Unchanged frontier replay in admitted row/context profile. Excludes uploads, Engram row I/O, complete-turn latency and qualified SGLang performance acceptance."}),
        )
    }
    pub fn residual(&self) -> Result<&Buffer<bf16>> {
        ensure!(
            self.lease.is_current() && self.last_live > 0,
            "stale target residual is not publishable"
        );
        Ok(self.late.output())
    }
    pub fn pre(&self) -> Result<&Buffer<f32>> {
        ensure!(
            self.lease.is_current() && self.last_live > 0,
            "stale target pre coefficients are not publishable"
        );
        Ok(self.late.pre())
    }
    pub fn dump_swa_row(&self, position: usize, path: &Path) -> Result<()> {
        use sha2::{Digest, Sha256};
        ensure!(
            self.lease.is_current()
                && self.last_live > 0
                && position < self.history.position as usize
                && !path.exists(),
            "SWA row audit admission"
        );
        let caches = self
            .early_caches
            .iter()
            .chain(self.group8.caches())
            .chain(self.group14.caches())
            .chain(self.late.caches())
            .collect::<Vec<_>>();
        ensure!(caches.len() == 40, "SWA audit layer count");
        ensure!(
            self.swa_layout
                .retained(position, self.history.position as usize),
            "SWA audit row was evicted from the retained window"
        );
        let slot = self.swa_layout.slot(position);
        let mut records = Vec::new();
        for (layer, cache) in caches.into_iter().enumerate() {
            ensure!(cache.layout() == crate::attention::KvLayout::V4, "the SWA row audit reads the V4 layout only");
            ensure!(slot < cache.capacity(), "SWA audit position outside cache");
            let tokens = cache.page_tokens();
            let base = cache.data().ptr() + ((slot / tokens) * cache.page_bytes()) as u64;
            let payload = unsafe {
                self.cuda
                    .read_device(base + ((slot % tokens) * 576) as u64, 576)
            }?;
            let scales = unsafe {
                self.cuda
                    .read_device(base + (tokens * 576 + (slot % tokens) * 8) as u64, 7)
            }?;
            records.push(serde_json::json!({"layer":layer,"position":position,"physical_slot":slot,
                "payload_sha256":format!("{:x}",Sha256::digest(&payload)),"scales_sha256":format!("{:x}",Sha256::digest(&scales)),
                "payload_nonzero_bytes":payload.iter().filter(|&&v| v!=0).count(),"scales_nonzero_bytes":scales.iter().filter(|&&v| v!=0).count()}));
        }
        std::fs::write(
            path,
            serde_json::to_vec_pretty(
                &serde_json::json!({"records":records,"scope":"Read-only observable V4 SWA data/scales, excludes footer padding"}),
            )?,
        )?;
        Ok(())
    }
    /// Compact read-only localization of a retained EXTEND discrepancy. The
    /// encoder contributes only new valid rows; the decoder contributes its
    /// entire replay tail. No reference tensors enter execution.
    pub fn dump_tail_boundaries(&self, encoder: &Self, new_rows: usize, root: &Path) -> Result<()> {
        use sha2::{Digest, Sha256};
        ensure!(
            self.rows == 128
                && encoder.rows == 6
                && self.last_live == 128
                && self.lease.is_current()
                && encoder.lease.is_current()
                && self.history.position == encoder.history.position
                && self.shared_state_report(encoder)["all_weights_and_causal_storage_shared"]
                    == true
                && (1..=6).contains(&new_rows)
                && !root.exists(),
            "tail boundary trace admission"
        );
        let replay = self
            .replay
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("tail trace requires prepared replay"))?;
        std::fs::create_dir_all(root)?;
        let mut entries = Vec::new();
        let mut save = |layer: usize,
                        key: &str,
                        ptr: u64,
                        bytes: usize,
                        rows: usize|
         -> Result<()> {
            let raw = unsafe { self.cuda.read_device(ptr, bytes) }?;
            let file = format!("L{layer}-{key}.bin");
            std::fs::write(root.join(&file), &raw)?;
            entries.push(serde_json::json!({"layer":layer,"key":key,"rows":rows,"file":file,"bytes":bytes,"sha256":format!("{:x}",Sha256::digest(&raw))}));
            Ok(())
        };
        for (id, layer) in encoder.layer_owners().into_iter().enumerate().take(21) {
            save(
                id,
                "Rout",
                layer.output().ptr(),
                new_rows * 20480 * 2,
                new_rows,
            )?;
            save(id, "pre", layer.pre().ptr(), new_rows * 4 * 4, new_rows)?;
        }
        save(21, "Rin", replay.hidden.ptr(), 128 * 20480 * 2, 128)?;
        save(21, "pre_in", replay.pre.ptr(), 128 * 4 * 4, 128)?;
        for (id, layer) in self.layer_owners().into_iter().enumerate().skip(21) {
            save(id, "Rout", layer.output().ptr(), 128 * 20480 * 2, 128)?;
            save(id, "pre", layer.pre().ptr(), 128 * 4 * 4, 128)?;
        }
        std::fs::write(
            root.join("index.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"position":self.history.position,"new_rows":new_rows,"decoder_rows":128,"records":entries,"full_engine_acceptance":false,"scope":"Valid-row boundary trace before benchmark replays; no fixture injection."}),
            )?,
        )?;
        Ok(())
    }
    /// Read-only differential trace after an accepted synthetic append. These
    /// owned layer buffers survive the whole forward; no fixture is fed back.
    pub(crate) fn dump_selector_trace(&self,root:&Path,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_SELECTOR_CAPTURE").as_deref()==Ok("1")
            && self.lease.is_current() && self.rows==6,"bounded selector trace state");
        stream.join_default()?;stream.synchronize()?;std::fs::create_dir(root)?;
        let mut branches=vec![(2,self.source2.query()),(8,self.group8.source().query()),
            (14,self.group14.source().query()),(20,self.late.source().query())];
        for id in [24,28,32,36] {
            branches.push((id,self.late.index(id).ok_or_else(||anyhow::anyhow!("missing index"))?.query()));
        }
        let mut records=Vec::new();
        for (id,q) in branches {
            let mut record=q.capture_selector(&self.cuda,&root.join(format!("layer{id}")))?;
            record["layer"]=serde_json::json!(id);records.push(record);
        }
        let record=serde_json::json!({"rows":self.rows,"position":self.history.position,"branches":records,
            "scope":"Untimed read-only selector scores and IDs after one real Verify6 step; raw padding remains present. No KV/model copies or CPU numeric analysis."});
        std::fs::write(root.join("index.json"),serde_json::to_vec_pretty(&record)?)?;
        Ok(())
    }
    /// Bounded read-only six-row trace for chunk-plan sensitivity. Full source
    /// cache dumps scale with configured context; these committed layer taps
    /// are enough to locate the first changed stage without copying that cache.
    #[cfg(feature = "storage-fault-injection")]
    pub(crate) fn dump_chunk_plan_trace(&self,root:&Path)->Result<()> {
        use sha2::{Digest,Sha256};
        ensure!(self.rows==6&&self.lease.is_busy_revision()&&self.pending.is_some()
            &&!root.exists(),"chunk-plan trace needs pending six-row verification");
        std::fs::create_dir_all(root)?;
        let mut records=Vec::new();
        for(id,layer)in self.layer_owners().iter().enumerate() {
            let mut taps=Vec::new();
            for(name,buffer)in [
                ("attn_input",layer.attn_boundary().input()),
                ("attn_reduced",layer.attn_reduced()),
                ("ffn_input",layer.ffn_boundary().input()),
                ("residual",layer.output())] {
                ensure!(buffer.bytes()<=6*20480*2,"bounded chunk-plan tap size");
                let raw=unsafe{self.cuda.read_device(buffer.ptr(),buffer.bytes())}?;
                let file=format!("layer{id}-{name}.bf16");
                std::fs::write(root.join(&file),&raw)?;
                taps.push(serde_json::json!({"name":name,"file":file,"bytes":raw.len(),
                    "sha256":format!("{:x}",Sha256::digest(&raw))}));
            }
            records.push(serde_json::json!({"layer":id,"taps":taps}));
        }
        std::fs::write(root.join("index.json"),serde_json::to_vec_pretty(&serde_json::json!({
            "rows":6,"position":self.history.position,"layers":records,
            "scope":"Read-only layer BF16 taps during identical six-token verification after different prefill chunk plans; no fixture injection."}))?)?;
        Ok(())
    }
    #[cfg(feature="storage-fault-injection")]
    fn dump_prefill_layer0_trace(&self,previous:usize,chunk:&[u32],stream:&Stream)->Result<()> {
        use sha2::{Digest,Sha256};
        ensure!(self.source_phase==Phase::Extend&&self.rows>=512&&chunk.len()>=128
            &&previous+chunk.len()==16896&&self.lease.is_busy_revision(),
            "bounded layer0 prefill trace geometry/owner");
        stream.synchronize()?;
        let root=Path::new("/native/evidence").join(format!("rank{}/prefill-layer0",self.rank));
        std::fs::create_dir_all(&root)?;
        let number=(0..16).find(|n|!root.join(format!("trace-{n}")).exists())
            .ok_or_else(||anyhow::anyhow!("layer0 trace bound exhausted"))?;
        let root=root.join(format!("trace-{number}"));std::fs::create_dir(&root)?;
        let layer=&self.early[0];ensure!(layer.sp_rank().is_none(),"layer0 trace expects full rows");
        let mut taps=Vec::new();
        let mut capture=|name:&str,ptr:u64,bytes:usize,element:usize|->Result<()> {
            ensure!(bytes%self.rows==0&&bytes/self.rows<=20480*4,"prefill trace row extent");
            let width=bytes/self.rows;let offset=(chunk.len()-128)*width;
            let raw=unsafe{self.cuda.read_device(ptr+offset as u64,128*width)}?;
            let file=format!("{name}.bin");std::fs::write(root.join(&file),&raw)?;
            taps.push(serde_json::json!({"name":name,"file":file,"rows":128,"element_bytes":element,
                "bytes":raw.len(),"sha256":format!("{:x}",Sha256::digest(&raw))}));Ok(())
        };
        for(name,b)in [("embedding",self.io.input()),("attn_input",layer.attn_boundary().input()),
            ("attn_local",layer.attention().output()),("attn_reduced",layer.attn_reduced()),
            ("attn_residual",layer.attn_boundary().output()),("ffn_input",layer.ffn_boundary().input()),
            ("shared_up",layer.ffn().shared_up()),("shared_down",layer.ffn().shared_down()),
            ("routed",layer.routed()),("ffn_local",layer.ffn().output()),
            ("ffn_reduced",layer.ffn_reduced()),("residual",layer.output())] {
            capture(name,b.ptr(),b.bytes(),2)?;
        }
        let router=layer.ffn().router();
        capture("router_logits",router.logits().ptr(),router.logits().bytes(),4)?;
        capture("router_ids",router.ids().ptr(),router.ids().bytes(),4)?;
        capture("router_weights",router.routing().ptr(),router.routing().bytes(),4)?;
        std::fs::write(root.join("index.json"),serde_json::to_vec_pretty(&serde_json::json!({
            "rank":self.rank,"previous":previous,"live":chunk.len(),"physical_rows":self.rows,
            "first_absolute_row":previous+chunk.len()-128,"last_absolute_row":previous+chunk.len()-1,
            "tail_token_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&chunk[chunk.len()-128..])?)),
            "taps":taps,"scope":"Immediate layer0 prefill taps, before row-arena reuse. Eager diagnostic interrupts enqueue; compare downstream logits against the prior ordinary graph path."}))?)?;
        Ok(())
    }
    /// Bounded first-divergence taps for the finite M16/M32 screen. Read only
    /// current forward workspaces, without downloading the full causal caches.
    pub(crate) fn dump_history32_trace(&self, root:&Path)->Result<()> {
        use anyhow::Context;
        use sha2::{Digest,Sha256};
        ensure!(crate::config::history32_probe_enabled() && [16,32].contains(&self.rows)
            // Logits are complete, but acceptance has not committed the writer.
            // Match pending_logits(): the completed speculative revision is busy.
            && self.source_phase==Phase::Verify && self.lease.is_busy_revision()
            && self.pending.is_some() && self.history.position>0 && !root.exists(),
            "bounded history32 trace ownership");
        std::fs::create_dir_all(root)?;
        let detailed_start=match std::env::var("CANADA_SCRATCH_HISTORY32_TRACE_DETAIL_START") {
            Ok(value)=>value.parse::<usize>().context("history32 trace detail start")?,
            Err(std::env::VarError::NotPresent)=>0,
            Err(error)=>return Err(error.into()),
        };
        ensure!(detailed_start<=36,"history32 trace detail range");
        let layers=self.layer_owners();
        ensure!(layers.len()==40,"history32 complete layer ownership");
        // Hash only the identical first16 causal rows at every layer boundary.
        // The hashes locate the first differing stage without writing hundreds
        // of megabytes of activation files or reading mutable MoE scratch.
        let mut boundaries=Vec::new();
        for (id,layer) in layers.iter().enumerate() {
            let mut taps=Vec::new();
            for (name,b) in [("attn_input",layer.attn_boundary().input()),
                ("attn_reduced",layer.attn_reduced()),("ffn_input",layer.ffn_boundary().input()),
                ("ffn_reduced",layer.ffn_reduced()),("residual",layer.output())] {
                ensure!(b.bytes()%self.rows==0&&b.bytes()/self.rows<=20480*2,
                    "history32 boundary row extent");
                let bytes=unsafe{self.cuda.read_device(b.ptr(),16*(b.bytes()/self.rows))}?;
                taps.push(serde_json::json!({"name":name,"common_rows":16,
                    "bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}));
            }
            boundaries.push(serde_json::json!({"layer":id,"taps":taps}));
        }
        let mut entries=Vec::new();
        for (id,layer) in layers.iter().enumerate().skip(detailed_start).take(4) {
            let dir=root.join(format!("layer{id}"));std::fs::create_dir(&dir)?;
            let mut taps=Vec::new();
            let mut save=|name:&str,dtype:&str,bytes:Vec<u8>|->Result<()> {
                let element=match dtype {"u8"=>1,"bf16"=>2,_=>4};
                ensure!(!bytes.is_empty() && bytes.len()%(self.rows*element)==0
                    && bytes.len()<=self.rows*20480*4,"bounded row-major history32 tap");
                let file=format!("{name}.bin");std::fs::write(dir.join(&file),&bytes)?;
                taps.push(serde_json::json!({"name":name,"dtype":dtype,"file":file,
                    "row_values":bytes.len()/self.rows/element,"bytes":bytes.len(),
                    "sha256":format!("{:x}",Sha256::digest(&bytes))}));Ok(())
            };
            let input=if id==0 {self.io.input()}else if id==1 {
                layer.engram().ok_or_else(||anyhow::anyhow!("history32 Engram1 absent"))?.output()
            }else if id==14 {
                self.group14.engram().ok_or_else(||anyhow::anyhow!("history32 Engram14 absent"))?.0.output()
            }else{layers[id-1].output()};
            save("input_residual","bf16",crate::layer_probe::bf16_bytes(input)?)?;
            for (name,boundary) in [("attn",layer.attn_boundary()),("ffn",layer.ffn_boundary())] {
                for (suffix,b) in [("input",boundary.input()),("output",boundary.output())] {
                    save(&format!("mhc_{name}_{suffix}"),"bf16",crate::layer_probe::bf16_bytes(b)?)?;
                }
                for (suffix,b) in [("pre",boundary.pre()),("post",boundary.post()),("comb",boundary.comb())] {
                    save(&format!("mhc_{name}_{suffix}"),"f32",unsafe{self.cuda.read_device(b.ptr(),b.bytes())}?)?;
                }
            }
            for (name,ptr,bytes,stride) in layer.attention().taps() {
                if !["qa","kv","normalized","qb","q_rope","inverse_attention","woa","output"].contains(&name) {continue;}
                let raw=if stride==0 {unsafe{self.cuda.read_device(ptr,bytes)}?}else{
                    ensure!(bytes%self.rows==0,"history32 strided tap");let mut raw=Vec::with_capacity(bytes);
                    for row in 0..self.rows {raw.extend(unsafe{self.cuda.read_device(ptr+(row*stride)as u64,bytes/self.rows)}?);}
                    raw
                };
                save(&format!("attn_{name}"),"bf16",raw)?;
            }
            let f=layer.ffn();
            // Routed MoE scratch is shared across layers. Its final contents
            // belong to the last layer, so only layer-owned results are tapped.
            for (name,b) in [("attn_reduced",layer.attn_reduced()),("ffn_reduced",layer.ffn_reduced()),
                ("ffn_shared_up",f.shared_up()),("ffn_activation",f.activation()),
                ("ffn_shared_down",f.shared_down()),("ffn_output",f.output())] {
                save(name,"bf16",crate::layer_probe::bf16_bytes(b)?)?;
            }
            for (name,b) in [("ffn_logits",f.router().logits()),("ffn_routing",f.router().routing())] {
                save(name,"f32",unsafe{self.cuda.read_device(b.ptr(),b.bytes())}?)?;
            }
            let ids=f.router().ids();save("ffn_ids","i32",unsafe{self.cuda.read_device(ids.ptr(),ids.bytes())}?)?;
            if id==20 {
                // Source20 owns these projections and selected/remapped IDs.
                // Publisher scratch and full padded score matrices are shared
                // or layout-dependent, so do not interpret them after forward.
                let source=self.late.source();
                let crate::compression::Input::Ratio1(projection)=source.projection() else {
                    anyhow::bail!("history32 source20 projection kind");
                };
                save("source_projection","bf16",crate::layer_probe::bf16_bytes(projection)?)?;
                for (name,b) in [("source_latent",source.source().latent()),("source_wk",source.wk()),
                    ("index_query",source.query().query_projection()),("index_heads",source.query().head_projection())] {
                    save(name,"bf16",crate::layer_probe::bf16_bytes(b)?)?;
                }
                let q=source.query().packed_query();
                for (name,dtype,ptr,bytes) in [
                    ("index_payload","u8",q.payload().ptr(),q.payload().bytes()),
                    ("index_scales","i32",q.scales().ptr(),q.scales().bytes()),
                    ("index_weights","f32",q.weights().ptr(),q.weights().bytes()),
                    ("index_selected","i32",source.query().selected().ptr(),source.query().selected().bytes()),
                    ("index_mapped","i32",source.candidates().ids().ptr(),source.candidates().ids().bytes()),
                    ("index_lengths","i32",source.candidates().lengths().ptr(),source.candidates().lengths().bytes())] {
                    save(name,dtype,unsafe{self.cuda.read_device(ptr,bytes)}?)?;
                }
            }
            entries.push(serde_json::json!({"layer":id,"taps":taps}));
        }
        ensure!(entries.len()==4,"history32 four-layer detailed trace");
        std::fs::write(root.join("index.json"),serde_json::to_vec_pretty(&serde_json::json!({
            "rows":self.rows,"position":self.history.position,"layers":entries,
            "detailed_start":detailed_start,"boundaries":boundaries,
            "scope":"Common-first16 boundary hashes for all40 layers and four selected detailed layers of one ordinary forward, recorded after logits. No injected intermediates, cache dump, or serving policy. Hash differences locate arithmetic changes, not quality loss."}))?)?;
        Ok(())
    }
    /// Probe only (CANADA_SCRATCH_MIDPASS_DUMP, Nov 12): the rows' embeddings (layer 0's input) and the residual
    /// after layers 13, 25 and 39 of the forward that just ran; every layer keeps its output until the next
    /// forward. Read-only; for the offline study of how early a draft's rejection shows in the target's state.
    pub(crate) fn midpass_states_for_probe(&self) -> Result<Vec<Vec<u8>>> {
        let mut layers = self.early.iter().collect::<Vec<_>>();
        layers.extend(self.group8.layers().iter());
        layers.extend(self.group14.layers().iter());
        layers.extend(self.late.layers().iter());
        ensure!(self.lease.is_current() && layers.len() == 40, "mid-pass probe needs the complete current target");
        let mut out = vec![crate::layer_probe::bf16_bytes(self.io.input())?];
        for id in [13usize, 25, 39] { out.push(crate::layer_probe::bf16_bytes(layers[id].output())?); }
        Ok(out)
    }
    /// Probe only (restore layer diff, Nov 13): the residual after each listed layer of the forward that just ran
    /// (every layer keeps its output until the next forward). Read-only.
    pub(crate) fn layer_outputs_for_probe(&self, ids: &[usize]) -> Result<Vec<Vec<u8>>> {
        let mut layers = self.early.iter().collect::<Vec<_>>();
        layers.extend(self.group8.layers().iter());
        layers.extend(self.group14.layers().iter());
        layers.extend(self.late.layers().iter());
        ensure!(self.lease.is_current() && layers.len() == 40 && ids.iter().all(|&id| id < 40),
            "layer output probe needs the complete current target");
        ids.iter().map(|&id| crate::layer_probe::bf16_bytes(layers[id].output())).collect()
    }
    pub fn dump_layer_trace(&self, root: &Path) -> Result<serde_json::Value> {
        use sha2::{Digest, Sha256};
        ensure!(
            self.lease.is_current() && self.history.position > 0 && !root.exists(),
            "target trace state/path"
        );
        std::fs::create_dir_all(root)?;
        let mut layers = self
            .early
            .iter()
            .zip(&self.early_caches)
            .enumerate()
            .map(|(id, (layer, cache))| (id, layer, cache))
            .collect::<Vec<_>>();
        for (start, group) in [(8, &self.group8), (14, &self.group14)] {
            layers.extend(
                group
                    .layers()
                    .iter()
                    .zip(group.caches())
                    .enumerate()
                    .map(|(i, (l, c))| (start + i, l, c)),
            );
        }
        layers.extend(
            self.late
                .layers()
                .iter()
                .zip(self.late.caches())
                .enumerate()
                .map(|(i, (l, c))| (20 + i, l, c)),
        );
        ensure!(layers.len() == 40, "complete target trace geometry");
        let mut entries = Vec::new();
        for (id, layer, cache) in &layers {
            let (local, mut global) = crate::layer_probe::snapshots(&self.cuda, layer, cache)?;
            let residual = match id {
                0 => self.io.input(),
                1 => layer
                    .engram()
                    .ok_or_else(|| anyhow::anyhow!("Engram1 trace absent"))?
                    .output(),
                14 => self
                    .group14
                    .engram()
                    .ok_or_else(|| anyhow::anyhow!("Engram14 trace absent"))?
                    .0
                    .output(),
                _ => layers[id - 1].1.output(),
            };
            global.insert(
                "input_residual".into(),
                crate::layer_probe::bf16_bytes(residual)?,
            );
            let dir = root.join(format!("layer{id}"));
            std::fs::create_dir(&dir)?;
            let mut taps = Vec::new();
            for (scope, values) in [("local", local), ("global", global)] {
                for (name, bytes) in values {
                    let file = format!("{scope}-{name}.bin");
                    std::fs::write(dir.join(&file), &bytes)?;
                    taps.push(serde_json::json!({"scope":scope,"name":name,"file":file,"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}));
                }
            }
            entries.push(serde_json::json!({"layer":id,"taps":taps}));
        }
        // Source taps distinguish attention arithmetic from changed indexed
        // candidate membership/order. Scores may contain unused padded columns;
        // compare only each row's causal prefix in diagnostics.
        let mut sources = Vec::new();
        for source in [&self.source2, self.group8.source(), self.group14.source(), self.late.source()] {
            let id = source.layer_id();
            let dir = root.join(format!("source{id}"));
            std::fs::create_dir(&dir)?;
            let q = source.query();
            let mut taps = Vec::new();
            for (name, ptr, bytes) in [
                ("scores", q.scores()?.ptr(), q.scores()?.bytes()),
                ("selected", q.selected().ptr(), q.selected().bytes()),
                ("candidates", source.candidates().ids().ptr(), source.candidates().ids().bytes()),
                ("candidate_lengths", source.candidates().lengths().ptr(), source.candidates().lengths().bytes()),
                ("compressed_kv", source.cache().data().ptr(), source.cache().data().bytes()),
                ("index_keys", source.keys().data().ptr(), source.keys().data().bytes()),
                ("compressed_latent", source.source().latent().ptr(), source.source().latent().bytes()),
                ("wk", source.wk().ptr(), source.wk().bytes()),
            ] {
                ensure!(bytes <= 64 * 1024 * 1024, "source trace exceeds diagnostic read bound");
                let raw = unsafe { self.cuda.read_device(ptr, bytes) }?;
                let file = format!("{name}.bin");
                std::fs::write(dir.join(&file), &raw)?;
                taps.push(serde_json::json!({"name":name,"file":file,"bytes":bytes,"sha256":format!("{:x}",Sha256::digest(&raw))}));
            }
            sources.push(serde_json::json!({"layer":id,"score_stride":q.score_stride(),"taps":taps}));
        }
        let index = serde_json::json!({"rows":self.rows,"position":self.history.position,"layers":entries,"sources":sources,"scope":"Read-only real token-input full-layer trace; no intermediate fixture injection"});
        std::fs::write(root.join("index.json"), serde_json::to_vec_pretty(&index)?)?;
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn graph_slots_are_unique_per_session_count_and_rows() {
        for rows in [6usize, 16, 32, 128] {
            let mut seen = std::collections::BTreeSet::new();
            for sessions in 1..=MAX_BATCHED_SESSIONS {
                for region in [false, true] {
                    for padded in [false, true] {
                        if (sessions > 1 && !region) || (padded && !(region && sessions > 1)) { continue; }
                        for executed in 0..=rows {
                            let slot = graph_slot(rows, sessions, executed, region, padded, 0);
                            assert!(slot < graph_slots(rows) && seen.insert(slot));
                            if sessions >= 2 && rows <= 32 {
                                for block in 2..=rows / sessions {
                                    let slot = graph_slot(rows, sessions, executed, region, padded, block);
                                    assert!(slot < graph_slots(rows) && seen.insert(slot), "{rows} {sessions} {executed} {block}");
                                }
                            }
                        }
                    }
                }
            }
            // One session keeps today's slot: the live-row count.
            assert!((0..=rows).all(|live| graph_slot(rows, 1, live, false, false, 0) == live));
        }
    }
    #[test]
    fn serving_regions_are_disjoint_and_steps_stay_inside_them() {
        use crate::swa_layout::Layout;
        let context = 1048576;
        // Relocation copies whole 64-slot pages: every ring keeps a position's offset in its page.
        for layout in [Layout::Ring, Layout::WideRing, Layout::WideRing8192, Layout::WideRing16384] {
            assert!((0..300000).step_by(37).all(|pos| layout.slot(pos) % 64 == pos % 64));
        }
        for layout in [Layout::WideRing, Layout::WideRing8192, Layout::WideRing16384] {
            for regions in 2..=4 {
                let pages = layout.pages(context).unwrap() + (regions - 1) * Layout::Ring.pages(context).unwrap();
                let (mut windows, mut shares) = (Vec::new(), Vec::new());
                for k in 0..regions {
                    let b = SessionBase::serving_region(k, regions, context, layout).unwrap();
                    assert_eq!(b.request as usize, k);
                    let span = b.layout.unwrap_or(layout).pages(context).unwrap() * 64;
                    windows.push((b.window, b.window + span));
                    shares.push((b.raw_pair, b.raw_pair + SessionBase::region_capacity(context, regions)));
                    for position in [0u64, 1, 127, 128, 1000, 65535, (SessionBase::region_capacity(context, regions) - 16) as u64] {
                        let p = Plan::batched(&[(b, position, 6)], 16, context, layout).unwrap();
                        for row in 0..6 {
                            let inside = |slot: i32| (b.window..b.window + span).contains(&(slot as usize));
                            assert!(inside(p.locations[row]));
                            let reads = p.swa_lengths[row] as usize;
                            assert!(reads >= 1 && p.swa_ids[row * 128..row * 128 + reads].iter().all(|&id| inside(id)));
                            assert_eq!(p.pair_starts[row] as usize, b.raw_pair / 2);
                            assert_eq!(p.requests[row], k as i64);
                        }
                    }
                }
                assert!(windows.windows(2).all(|w| w[0].1 <= w[1].0) && windows.last().unwrap().1 <= pages * 64);
                assert!(shares.windows(2).all(|w| w[0].1 <= w[1].0) && shares.last().unwrap().1 <= context);
            }
        }
    }
    #[test]
    fn full_serving_regions_hold_whole_conversations() {
        use crate::swa_layout::Layout;
        let context = crate::config::MODEL_CONTEXT;
        for regions in 2..=4 {
            let address = crate::config::compressed_address_context_with(context, regions, true).unwrap();
            assert_eq!(address, context * regions);
            assert_eq!(crate::config::compressed_address_context_with(context, regions, false).unwrap(), context);
            assert!(crate::config::compressed_address_context_with(65536, regions, true).is_err());
            assert!(crate::config::compressed_address_context_with(context, 1, true).is_err());
            assert_eq!(SessionBase::serving_capacity_with(context, regions, true).unwrap(), context);
            assert_eq!(SessionBase::serving_capacity_with(context, regions, false).unwrap(),
                SessionBase::region_capacity(context, regions));
            // Every cache spans every region's whole conversation; key pages stay inside the admitted index domain
            // (config::index_key_capacity_limit with full regions: address + 4096 rows).
            let single = compressed_pages(address, 1, 64, 8) * 64;
            let pair = compressed_pages(address, 2, 64, 4) * 64;
            for ratio in [1, 2] {
                let pages = compressed_pages(address, ratio, 64, 4);
                assert!(pages * 64 >= 32 + address / ratio && pages <= (address + 4096) / 64);
            }
            assert!(crate::decoder_tail::Cache::validate_capacity(context, single).is_ok());
            let layout = Layout::WideRing8192;
            let mut shares = Vec::new();
            for k in 0..regions {
                let b = SessionBase::serving_region_with(k, regions, context, layout, true).unwrap();
                assert_eq!((b.raw_pair, b.raw_single, b.request), (k * context, k * context, k as i64));
                shares.push((b.raw_single, b.raw_single + context));
                for position in [0u64, 1, 128, 262144, 524288, (context - 6) as u64] {
                    let p = Plan::batched(&[(b, position, 6)], 16, context, layout).unwrap();
                    for row in 0..6 {
                        let (raw_single, pair_slot) = (p.raw_single[row] as usize, p.raw_pair[row] as usize / 2);
                        assert!(raw_single >= COMPRESSED_OFFSET + k * context && raw_single < COMPRESSED_OFFSET + (k + 1) * context);
                        assert!(raw_single < single && pair_slot < pair);
                        assert_eq!(p.single_starts[row] as usize, k * context);
                        assert_eq!(p.pair_starts[row] as usize, k * context / 2);
                    }
                }
            }
            assert!(shares.windows(2).all(|w| w[0].1 <= w[1].0) && shares.last().unwrap().1 <= address);
        }
    }
    #[test]
    fn batched_sessions_are_offset_copies_of_their_own_appends() {
        use crate::swa_layout::Layout;
        let layout = Layout::Ring;
        let window = crate::swa_layout::OFFSET + layout.ring().unwrap();
        // A single sequence at base zero is exactly today's plan.
        for (pos, live) in [(0u64, 1usize), (5, 3), (129, 6), (4095, 6)] {
            let legacy = Plan::append_with_layout(pos, live, 6, 262144, layout).unwrap();
            let based = Plan::append_session(pos, live, 6, 262144, layout, SessionBase::default()).unwrap();
            assert_eq!(format!("{legacy:?}"), format!("{based:?}"));
            assert!(legacy.requests.iter().chain(&legacy.requests).all(|&r| r == 0));
            assert!(legacy.pair_starts.iter().chain(&legacy.single_starts).all(|&s| s == 0));
        }
        // Two sessions of 131072 positions in one 262144-position owner.
        let a = SessionBase::region(0, 131072, window).unwrap();
        let b = SessionBase::region(1, 131072, window).unwrap();
        assert_eq!(a, SessionBase::default());
        let batch = Plan::batched(&[(a, 300, 3), (b, 4000, 3)], 6, 262144, layout).unwrap();
        let alone_a = Plan::append_session(300, 3, 6, 262144, layout, a).unwrap();
        let alone_b = Plan::append_session(4000, 3, 6, 262144, layout, b).unwrap();
        for row in 0..3 {
            assert_eq!(batch.positions[row], alone_a.positions[row]);
            assert_eq!(batch.locations[row], alone_a.locations[row]);
            assert_eq!(batch.positions[row + 3], alone_b.positions[row]);
            assert_eq!(batch.locations[row + 3], alone_b.locations[row]);
            assert_eq!(batch.raw_pair[row + 3], alone_b.raw_pair[row]);
            assert_eq!(batch.swa_ids[(row + 3) * 128..(row + 4) * 128], alone_b.swa_ids[row * 128..(row + 1) * 128]);
            // Same visible counts, disjoint regions: B's addresses are A's formulas plus its bases.
            assert_eq!(batch.locations[row + 3] as usize, window + layout.slot(4000 + row));
            assert_eq!(batch.raw_pair[row + 3] as usize, 131072 + SWA_OFFSET + 4000 + row);
            assert_eq!(batch.raw_single[row + 3] as usize, 131072 + COMPRESSED_OFFSET + 4000 + row);
            assert_eq!(batch.pair_lengths[row + 3], ((4000 + row + 1) / 2) as i32);
            assert_eq!((batch.requests[row], batch.requests[row + 3]), (0, 1));
            assert_eq!((batch.pair_starts[row + 3], batch.single_starts[row + 3]), (65536, 131072));
        }
        // B's key region [65536, 65536 + len) and A's [0, len) never overlap within capacity.
        assert!(alone_a.pair_lengths.iter().all(|&l| (l as usize) < 65536));
        // One part per session, total rows bounded, absolute layout refused for offsets.
        assert!(Plan::batched(&[(a, 1, 3), (a, 9, 3)], 6, 262144, layout).is_err());
        assert!(Plan::batched(&[(a, 1, 4), (b, 9, 3)], 6, 262144, layout).is_err());
        assert!(Plan::append_session(1, 1, 6, 262144, Layout::Absolute, b).is_err());
        // Short parts keep their blocks: A at rows [0, 2), B at row 3; the step
        // executes rows [0, 4). Each block is one request with consecutive
        // positions and no-write padding as a suffix (the pair verify kernel's rule).
        let short = Plan::batched(&[(a, 300, 2), (b, 4000, 1)], 6, 262144, layout).unwrap();
        assert_eq!((short.block, Plan::batched_extent(&[2, 1], 6)), (3, 4));
        for row in 0..2 { assert_eq!(short.raw_pair[row], alone_a.raw_pair[row]); }
        assert_eq!(short.raw_pair[3], alone_b.raw_pair[0]);
        assert_eq!((short.raw_pair[2], short.locations[2], short.positions[2], short.requests[2]), (0, -1, 301, 0));
        assert_eq!((short.raw_pair[4], short.raw_pair[5], short.positions[5], short.requests[5]), (0, 0, 4000, 1));
        for plan in [&batch, &short] {
            for start in (0..6).step_by(plan.block) {
                let active = (0..plan.block).take_while(|i| plan.raw_pair[start + i] != 0).count();
                assert!(active > 0 && plan.raw_pair[start + active..start + plan.block].iter().all(|&l| l == 0));
                for i in 0..active {
                    assert_eq!(plan.requests[start + i], plan.requests[start]);
                    assert_eq!(plan.positions[start + i], plan.positions[start] + i as i32);
                }
            }
        }
        // One session keeps one block of every row; blocks must divide the rows.
        assert_eq!(Plan::batched(&[(b, 9, 5)], 6, 262144, layout).unwrap().block, 6);
        assert!(Plan::batched(&[(a, 1, 1), (b, 9, 1), (SessionBase::region(2, 65536, window).unwrap(), 5, 1),
            (SessionBase::region(3, 65536, window).unwrap(), 5, 1)], 6, 262144, layout).is_err());
    }
    #[test]
    fn region_cache_granules_tile_the_whole_region() {
        let g = crate::config::REGION_MAP_GRANULE;
        let context = crate::config::MODEL_CONTEXT;
        for k in 1..4 {
            for (ratio, page) in [(1usize, 64usize), (2, 64), (2, 128), (1, 256)] {
                let whole = Target::serving_region_span_pages(k, ratio, page, 0, context);
                let mut at = 0;
                let mut last_end = whole.0;
                while at < context {
                    let (a, b) = Target::serving_region_span_pages(k, ratio, page, at, at + g);
                    // Consecutive granules meet or share one page, never leave a gap.
                    assert!(a <= last_end && a + 1 >= last_end && b > a, "k {k} ratio {ratio} page {page} at {at}");
                    last_end = b;
                    at += g;
                }
                assert_eq!(last_end, whole.1);
                // A granule's pages hold every position in it.
                let (a, b) = Target::serving_region_span_pages(k, ratio, page, g, 2 * g);
                let first = k * context / ratio + COMPRESSED_OFFSET;
                assert!(a * page <= first + g / ratio && (first + (2 * g) / ratio) <= b * page);
            }
        }
    }
    #[test]
    fn rows_past_the_last_block_keep_the_scorer_inside_the_last_region() {
        use crate::swa_layout::Layout;
        let layout = Layout::Ring;
        let window = crate::swa_layout::OFFSET + layout.ring().unwrap();
        let base = |k| SessionBase::region(k, 65536, window).unwrap();
        // Three sessions on 16 rows: blocks of five, row 15 past the last block.
        let p = Plan::batched(&[(base(0), 300, 5), (base(1), 4000, 3), (base(2), 900, 4)], 16, 262144, layout).unwrap();
        assert_eq!(p.block, 5);
        let (pair2, single2) = (p.pair_starts[10], p.single_starts[10]);
        assert!(pair2 > 0 && single2 > 0);
        assert_eq!((p.pair_starts[15], p.single_starts[15], p.pair_lengths[15], p.single_lengths[15]), (pair2, single2, 0, 0));
        // The MQA scorer's last block of four query rows (12-15) now spans only the third session's keys.
        let lo = (12..16).map(|r| p.pair_starts[r]).min().unwrap();
        let hi = (12..16).map(|r| p.pair_starts[r] + p.pair_lengths[r]).max().unwrap();
        assert_eq!(lo, pair2);
        assert_eq!(hi - lo, 452); // positions 900-903: (903 + 1) / 2 pair keys at most
        // Two sessions on 16 rows: blocks of six; rows 12-15 are all past the last block and stay empty.
        let q = Plan::batched(&[(base(0), 300, 6), (base(1), 4000, 6)], 16, 262144, layout).unwrap();
        assert!((12..16).all(|r| q.pair_starts[r] == q.pair_starts[6] && q.pair_lengths[r] == 0));
    }
    #[test]
    fn compact_state_pages_cover_wraps_and_padding_without_aliases() {
        use crate::swa_layout::Layout;
        for layout in [Layout::Absolute,Layout::Ring,Layout::WideRing,Layout::WideRing8192,Layout::WideRing16384] {
            for frontier in [1,127,128,255,256,257,8191,8192,8193,16383,16384,16385,262655,1048576] {
                let ranges=saved_swa_pages(layout,frontier);
                assert_eq!(ranges[0].0,0);
                assert!(ranges.windows(2).all(|r|r[0].1<r[1].0));
                assert!(ranges.iter().all(|&(a,b)|a<b && b<=layout.pages(1048576).unwrap()));
                assert!(ranges.iter().map(|&(a,b)|b-a).sum::<usize>()<=6);
                for p in frontier.saturating_sub(256)..frontier {
                    let page=layout.slot(p)/64;
                    assert!(ranges.iter().any(|&(a,b)|a<=page && page<b));
                }
            }
        }
    }
    #[test]
    fn every_sequence_parallel_shard_is_an_admitted_mhc_geometry() {
        for &w in crate::config::WIDE_PROFILES {
            assert!(w % 4 == 0 && crate::config::mhc_rows(w / 4) && crate::config::mhc_rows(w));
        }
    }
    #[test]
    fn layer20_deferral_covers_exactly_the_rows_before_the_final_128() {
        let widths = [128, 512, 1024, 2048, 4096, 8192];
        for total in (128..=40000).step_by(37).chain([128, 129, 255, 256, 8191, 8192, 8320, 8321, 145039]) {
            let plan = crate::config::plan_encoder_chunks(total, &widths);
            let mut done = 0;
            for (i, &rows) in plan.iter().enumerate() {
                let live = (total - done).min(rows);
                let deferred = super::layer20_projection_only(done, live, rows, total);
                // Deferred chunks never hold any of the final 128 rows, and never
                // the last chunk; every chunk holding one of them runs layer 20.
                assert_eq!(deferred, live == rows && done + live <= total - 128, "{total} {plan:?} {i}");
                if deferred { assert!(i + 1 < plan.len()); }
                if done + live > total - 128 { assert!(!deferred); }
                done += live;
            }
            assert_eq!(done, total);
        }
        // 145039 rows: 17 full 8K chunks defer, the 4096/1024/512/512 tail runs layer 20
        // only where it holds one of the final 128 rows.
        let plan = crate::config::plan_encoder_chunks(145039, &widths);
        let mut done = 0;
        let deferred = plan.iter().filter(|&&rows| {
            let live = (145039 - done).min(rows);
            let d = super::layer20_projection_only(done, live, rows, 145039);
            done += live; d
        }).count();
        assert!(deferred >= 17, "{plan:?}");
    }
    #[test]
    fn encoder_chunk_plan_minimizes_estimated_cost() {
        let all = [128, 512, 1024, 2048, 4096];
        assert_eq!(crate::config::plan_encoder_chunks(458, &all), vec![512]);
        assert_eq!(crate::config::plan_encoder_chunks(900, &all), vec![1024]);
        assert_eq!(crate::config::plan_encoder_chunks(2039, &all), vec![2048]);
        assert_eq!(crate::config::plan_encoder_chunks(8000, &all), vec![4096, 4096]);
        assert_eq!(crate::config::plan_encoder_chunks(200, &all), vec![512]);
        assert!(crate::config::plan_encoder_chunks(0, &all).is_empty());
        assert_eq!(crate::config::plan_encoder_chunks(200, &[128, 1024]), vec![128, 128]);
        assert_eq!(crate::config::plan_encoder_chunks(27319, &all), vec![4096, 4096, 4096, 4096, 4096, 4096, 2048, 1024]);
        assert_eq!(crate::config::plan_encoder_chunks(2039, &[128, 1024, 4096]), vec![1024, 1024]);
        for len in [129, 1500, 5000, 27319, 65536] {
            let plan = crate::config::plan_encoder_chunks(len, &all);
            let covered: usize = plan.iter().sum();
            assert!(covered >= len && covered - plan.last().unwrap() < len);
        }
    }
    #[test]
    fn ring_addresses_preserve_absolute_positions_and_compression_metadata() {
        use crate::swa_layout::Layout;
        for start in [129, 253, 256, 383, 512, 3968] {
            for live in [1, 6, 128] {
                let absolute = Plan::append_in(start, live, 128, 4096).unwrap();
                let mut ring =
                    Plan::append_with_layout(start, live, 128, 4096, Layout::Ring).unwrap();
                assert_eq!(ring.positions, absolute.positions);
                assert_eq!(ring.raw_pair, absolute.raw_pair);
                assert_eq!(ring.raw_single, absolute.raw_single);
                assert_eq!(ring.pair_lengths, absolute.pair_lengths);
                assert_eq!(ring.single_lengths, absolute.single_lengths);
                assert_eq!(ring.swa_lengths, absolute.swa_lengths);
                for row in 0..live {
                    let pos = start as usize + row;
                    assert_eq!(ring.locations[row], Layout::Ring.slot(pos) as i32);
                    assert_eq!(
                        &ring.swa_ids[row * 128..(row + 1) * 128],
                        &(pos - 127..=pos)
                            .rev()
                            .map(|p| Layout::Ring.slot(p) as i32)
                            .collect::<Vec<_>>()
                    );
                }
                assert!(ring.locations[live..].iter().all(|&x| x == -1));
                ring.floor_swa_at(start as usize).unwrap();
                for row in 0..live {
                    assert_eq!(ring.swa_lengths[row], (row + 1) as i32);
                    assert_eq!(
                        ring.swa_ids[row * 128 + row],
                        Layout::Ring.slot(start as usize) as i32
                    );
                }
            }
        }
    }
    #[test]
    fn retained_full_chunk_preserves_absolute_carry_and_decoder_floor() {
        let mut p = Plan::append_in(129, 128, 128, 512).unwrap();
        assert_eq!(p.positions, (129..257).collect::<Vec<_>>());
        assert_eq!(p.raw_pair[0], 193);
        assert_eq!(p.pair_lengths[0], 65);
        assert_eq!(p.pair_lengths[127], 128);
        assert_eq!(p.swa_lengths[0], 128);
        p.floor_swa_at(129).unwrap();
        assert_eq!(p.swa_lengths[0], 1);
        assert_eq!(p.swa_ids[0], 193);
        assert_eq!(p.swa_ids[1], -1);
        assert_eq!(p.swa_lengths[127], 128);
        assert_eq!(p.swa_ids[127 * 128], 320);
        assert_eq!(p.swa_ids[128 * 128 - 1], 193);
        assert!(Plan::append_in(385, 128, 128, 512).is_err());
    }
    #[test]
    fn extend_floor_separates_new_chunk_from_retained_encoder_window() {
        for start in 0..256usize {
            for live in 1..=6.min(256 - start) {
                let mut decoder = Plan::append_in(start as u64, live, 6, 256).unwrap();
                assert!(decoder.floor_swa_at(start + 1).is_err());
                decoder.floor_swa_at(start).unwrap();
                for row in 0..6 {
                    assert_eq!(
                        decoder.swa_lengths[row],
                        if row < live { (row + 1) as i32 } else { 0 }
                    );
                    for col in 0..128 {
                        let expected = if row < live && col <= row {
                            (SWA_OFFSET + start + row - col) as i32
                        } else {
                            -1
                        };
                        assert_eq!(decoder.swa_ids[row * 128 + col], expected);
                    }
                }
            }
        }
        for start in 0..=128usize {
            let mut tail = Plan::append_in(start as u64, 128, 128, 256).unwrap();
            tail.floor_swa_at(start).unwrap();
            for row in 0..128 {
                assert_eq!(tail.swa_lengths[row], (row + 1) as i32);
                assert_eq!(tail.swa_ids[row * 128 + row], (SWA_OFFSET + start) as i32);
            }
        }
    }
    #[test]
    fn retained_windows_preserve_causal_bounds_after_profile_handoff() {
        for start in 0..256usize {
            for live in 1..=6.min(256 - start) {
                let p = Plan::append_in(start as u64, live, 6, 256).unwrap();
                for row in 0..live {
                    let position = start + row;
                    let visible = (position + 1).min(128);
                    let ids = &p.swa_ids[row * 128..(row + 1) * 128];
                    assert_eq!(p.swa_lengths[row] as usize, visible);
                    assert_eq!(ids[0], (64 + position) as i32);
                    assert_eq!(ids[visible - 1], (64 + position + 1 - visible) as i32);
                    assert!(ids[visible..].iter().all(|&id| id == -1));
                    assert_eq!(p.raw_pair[row] / 2, (32 + position / 2) as i64);
                    assert_eq!(p.raw_single[row], (32 + position) as i64);
                }
                assert!(p.locations[live..].iter().all(|&p| p == -1));
                assert!(p.raw_pair[live..].iter().all(|&p| p == 0));
            }
        }
        assert!(Plan::append_in(128, 128, 128, 256).is_ok());
        assert!(Plan::append_in(129, 128, 128, 256).is_err());
        // Larger admitted context: absolute addresses must survive the last
        // SWA window and reject an append crossing the4096-token boundary.
        for start in [2038, 2047, 3968, 4090, 4095] {
            let live = 6.min(4096 - start);
            let p = Plan::append_in(start as u64, live, 6, 4096).unwrap();
            for row in 0..live {
                let position = start + row;
                let ids = &p.swa_ids[row * 128..(row + 1) * 128];
                assert_eq!(ids[0], (SWA_OFFSET + position) as i32);
                assert_eq!(ids[127], (SWA_OFFSET + position - 127) as i32);
                assert_eq!(p.single_lengths[row], (position + 1) as i32);
            }
            assert!(p.locations[live..].iter().all(|&id| id == -1));
        }
        assert!(Plan::append_in(4091, 6, 6, 4096).is_err());
        for &context in crate::config::EXPERIMENTAL_CONTEXT_CAPACITIES {
            let capacity = compressed_pages(context, 1, 64, 8) * 64;
            assert!(crate::decoder_tail::Cache::validate_capacity(context, capacity).is_ok());
            assert!(crate::decoder_tail::Cache::validate_capacity(context, context + 31).is_err());
            assert!(capacity <= 1052672);
            for ratio in [1,2] {
                let key_pages = compressed_pages(context,ratio,64,4);
                assert!(key_pages <= 16448 && key_pages*64 >= 32+context/ratio);
            }
        }
        // Beyond the admitted experimental contexts.
        assert!(crate::decoder_tail::Cache::validate_capacity(1048577, 1048640).is_err());
    }
    #[test]
    fn wide_plan_keeps_visible_prefix_and_padding_separate() {
        for live in 1..=128 {
            let p = Plan::append_rows(0, live, 128).unwrap();
            assert_eq!(p.positions.len(), 128);
            for row in 0..live {
                let visible = &p.swa_ids[row * 128..(row + 1) * 128];
                assert_eq!(visible.iter().filter(|&&v| v >= 0).count(), row + 1);
                assert!(
                    visible[..row + 1]
                        .iter()
                        .all(|&v| (64..=64 + row as i32).contains(&v))
                );
                assert_eq!(p.raw_pair[row] as usize / 2, 32 + row / 2);
                assert_eq!(p.pair_lengths[row] as usize, (row + 1) / 2);
            }
            assert!(p.raw_pair[live..].iter().all(|&v| v == 0));
            assert!(p.swa_lengths[live..].iter().all(|&v| v == 0));
        }
        assert!(Plan::append_rows(1, 128, 128).is_err());
        assert!(Plan::append_rows(0, 129, 128).is_err());
        assert!(Plan::append_rows(0, 1, 127).is_err());
    }
    #[test]
    fn causal_append_boundaries_and_padding() {
        // Exercise every admitted position/append length, including odd carried
        // pairs and the final page. No future or padded address may be visible.
        for pos in 0..CONTEXT {
            for live in 1..=ROWS.min(CONTEXT - pos) {
                let p = Plan::append(pos as u64, live).unwrap();
                for row in 0..live {
                    let last = pos + row;
                    assert_eq!(p.swa_lengths[row], (last + 1) as i32);
                    assert!(
                        p.swa_ids[row * 128..row * 128 + last + 1]
                            .iter()
                            .enumerate()
                            .all(|(i, &a)| a == (64 + last - i) as i32)
                    );
                    assert_eq!(p.raw_pair[row] % 2, p.positions[row] as i64 % 2);
                    assert_eq!(p.pair_lengths[row], ((last + 1) / 2) as i32);
                    assert_eq!(p.single_lengths[row], (last + 1) as i32);
                }
                for row in live..ROWS {
                    assert_eq!(p.locations[row], -1);
                    assert_eq!(
                        (
                            p.raw_pair[row],
                            p.raw_single[row],
                            p.pair_lengths[row],
                            p.single_lengths[row]
                        ),
                        (0, 0, 0, 0)
                    );
                    assert!(
                        p.swa_ids[row * 128..(row + 1) * 128]
                            .iter()
                            .all(|&a| a == -1)
                    );
                }
            }
        }
        assert!(Plan::append(0, 0).is_err());
        assert!(Plan::append(0, 7).is_err());
        assert!(Plan::append(128, 1).is_err());
        assert!(Plan::append(u64::MAX, 1).is_err());
    }
}

impl Target {
    pub(crate) fn set_coalesced_codec(&mut self,enabled:bool,kernels:&Path,stream:&Stream)->Result<()> {
        ensure!(self.pending.is_none() && self.graph_captures==0 && self.encoder_graph.is_none()
            && self.decoder_graph.is_none() && self.graphs.iter().all(Option::is_none)
            && self.engram_graphs.iter().all(Option::is_none)
            && self.cuda.owns_stream(stream) && self.stream==stream.raw() as usize,
            "codec replacement cannot change a captured or pending execution");
        stream.synchronize()?;
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.set_coalesced_codec(enabled,kernels)?;
        }
        Ok(())
    }
    pub(crate) fn coalesced_codec_layers(&self)->usize {
        self.layer_owners().iter().filter(|layer|layer.coalesced_codec_active()).count()
    }
}

/// CANADA_SCRATCH_ENGRAM_GAP_PROBE=1 (diagnostic, default off): time the GPU idles between a verification's
/// initial graph (embedding and layer 0) and the rest of its forward, which is enqueued after the Engram row
/// lookup. Two timing events a step; each rank prints its running mean every 64 verifications.
fn engram_gap_probe() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("CANADA_SCRATCH_ENGRAM_GAP_PROBE").as_deref() == Ok("1"))
}
struct EngramGap { first: crate::cuda::Event, second: crate::cuda::Event, count: u64, total_ms: f64, max_ms: f32 }
thread_local! {
    static ENGRAM_GAP: std::cell::RefCell<Option<EngramGap>> = const { std::cell::RefCell::new(None) };
}
