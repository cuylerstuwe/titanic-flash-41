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
    live_rows: usize,
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
    stages_ms: Vec<(&'static str, f64)>,
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
fn wide_arena_bytes_per_row(sequence_parallel: bool) -> usize {
    (if sequence_parallel { 384 } else { 448 }) * 1024
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
        };
        for row in 0..live {
            let pos = position + row;
            p.positions[row] = pos as i32;
            p.locations[row] = swa_layout.slot(pos) as i32;
            // Pair raw addresses retain absolute-position parity. Slot0 is
            // reserved by the compression kernel as the no-write sentinel.
            p.raw_pair[row] = (SWA_OFFSET + pos) as i64;
            p.raw_single[row] = (COMPRESSED_OFFSET + pos) as i64;
            p.pair_lengths[row] = ((pos + 1) / 2) as i32;
            p.single_lengths[row] = (pos + 1) as i32;
            let start = (pos + 1).saturating_sub(128);
            p.swa_lengths[row] = (pos + 1 - start) as i32;
            // Qualified BuildCausalSwaPageIndices visits newest to oldest.
            // Although the same keys are visible in either order, permuting
            // the attention reduction can change BF16 ties and downstream
            // routing. Keep the actual numeric-order contract.
            for (col, previous) in (start..=pos).rev().enumerate() {
                p.swa_ids[row * 128 + col] = swa_layout.slot(previous) as i32;
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
            m.update(
                &p.positions,
                &vec![0; p.positions.len()],
                &p.raw_pair,
                &p.pair_lengths,
            )?;
        }
        self.m20.update(
            &p.positions,
            &vec![0; p.positions.len()],
            &p.raw_single,
            &p.single_lengths,
        )?;
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
    replay: Option<crate::decoder_tail::Replay>,
    graph_replay: bool,
    prefill_last_head: bool,
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
    early: Vec<SwaLayer>,
    early_caches: Vec<PagedKv>,
    source2: Source,
    group8: crate::source_group::Group,
    group14: crate::source_group::Group,
    late: crate::late_group::Group,
    step: Step,
    cached_decoder: Option<Step>,
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
                > remaining + derived + 8 * 1024usize.pow(3),
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
        let swa_pages = swa_layout.pages(context)?;
        let page_tokens = if source_phase == Phase::Extend {
            64
        } else {
            32
        };
        let pair_pages = compressed_pages(context, 2, page_tokens, 4);
        let single_pages = compressed_pages(context, 1, page_tokens, 8);
        let pair_key_pages = compressed_pages(context, 2, 64, 2);
        let single_key_pages = compressed_pages(context, 1, 64, 4);
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
            early_caches.push(PagedKv::new(cuda, swa_pages, 64)?);
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
        let row_store1 = Rc::new(RowStore::new(
            PackedShard::open(
                &backing.join(format!("engram-l1-r{}of4.bin", plan.rank)),
                1,
                plan.rank,
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
            graphs: (0..=rows).map(|_| None).collect(),
            engram_graphs: (0..=rows).map(|_| None).collect(),
            alternate_index_graphs: (0..=rows).map(|_| None).collect(),
            alternate_index_engram_graphs: (0..=rows).map(|_| None).collect(),
            adaptive_sparse_active: false,
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
            replay: None,
            graph_replay: false,
            prefill_last_head: false,
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
            early,
            early_caches,
            source2,
            group8,
            group14,
            late,
            step,
            cached_decoder: None,
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
            > causal_bytes + 4*1024usize.pow(3) + 8*1024usize.pow(3),
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
        ensure!(self.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3),
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
            let target = self.fork_wide_with_arenas(core, core_bytes, stream, kernels,
                rows, sequence_parallel, if shared {bank.first()} else {None})?;
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
        self.fork_wide_with_arenas_in(core,core_bytes,stream,kernels,rows,sequence_parallel,share,false)
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
            self.fork_wide_with_arenas_in(core,core_bytes,stream,kernels,rows,true,Some(donor),true)
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
            ensure!(self.cuda.admission_memory()?.admitted_available > planned + 8 * 1024usize.pow(3),
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
        let early_caches=if independent {self.early_caches.iter()
            .map(|c|PagedKv::new(cuda,c.pages(),c.page_tokens())).collect::<Result<Vec<_>>>()?}
            else {self.early_caches.clone()};
        let pair_slots = (COMPRESSED_OFFSET..COMPRESSED_OFFSET + self.context / 2)
            .map(|s| s as i32)
            .collect::<Vec<_>>();
        let single_slots = (COMPRESSED_OFFSET..COMPRESSED_OFFSET + self.context)
            .map(|s| s as i32)
            .collect::<Vec<_>>();
        let medium_woa = if rows > 8 {
            Some(crate::woa_medium::Scratch::new_on(cuda, core_bytes, rows,stream)?)
        } else {
            None
        };
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
                source2 = Some(Source::new_rows_owner(
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
                )?);
            }
        }
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
        let rows1 = Rows::new(cuda, rows, &self.row_store1)?;
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
        let step = Step::new(
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
        )?;
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
            prefill_mhc_live_rows: self.prefill_mhc_live_rows,
            prefill_mhc_live_norm: self.prefill_mhc_live_norm,
            prefill_blocking_wait: self.prefill_blocking_wait,
            verification_poll_us: self.verification_poll_us,
            verification_mhc_live_rows: self.verification_mhc_live_rows,
            tiny_prefix_sum: phase==Phase::Verify && [6,16].contains(&rows)
                && std::env::var("CANADA_SCRATCH_LIVE_ROW_TRANSPORT").as_deref()==Ok("1"),
            tiny_prefix_gather: phase==Phase::Verify && [6,16].contains(&rows)
                && std::env::var("CANADA_SCRATCH_LIVE_ROW_TRANSPORT").as_deref()==Ok("1"),
            graphs: (0..=rows).map(|_| None).collect(),
            engram_graphs: (0..=rows).map(|_| None).collect(),
            alternate_index_graphs: (0..=rows).map(|_| None).collect(),
            alternate_index_engram_graphs: (0..=rows).map(|_| None).collect(),
            adaptive_sparse_active: false,
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
            replay: None,
            graph_replay: false,
            prefill_last_head: false,
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
            early,
            early_caches,
            source2,
            group8,
            group14,
            late,
            step,
            cached_decoder: None,
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
                regions.push(Region::new(&self.cuda, format!("compressed.{layer}.carry"), ring)?);
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
            restore,prefix,suffix,budget,None,comm,stream,&mut |_|Ok(false),&mut |admit|Ok(admit))?;
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
    )->Result<(Option<EncoderPrefixImage>,serde_json::Value,PrefillProgress)> {
        let recycled_bytes=recycled.as_ref().map_or(0,crate::state_image::Image::bytes);
        ensure!(recycled.is_none()||budget==Some(recycled_bytes),"encoder recycled storage extent");
        let previous=restore.as_ref().map_or(self.history.position as usize,|r|r.image_and_lineage().0.position());
        ensure!(encoder_checkpoint_enabled()&&suffix.len()>=128&&previous+prefix.len()>=128
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
        // The complete final suffix replaces every omitted decoder/MTP row.
        // s788/s796 each exercised the128 omission twice (the other requests
        // were controls). The one s785 sampled divergence remains unexplained.
        let defer128=defer_checkpoint_decoder128()?;
        let progress=if prefix.len()>=256 || (defer128&&prefix.len()>=128&&previous>0) {
            self.prefill_wide_interruptible_policy(wides,prefix,comm,stream,lookahead,
                self.graph_replay,encoder_graph,tiny_encoder_graph,true,None,should_yield)?
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
                >=8*1024usize.pow(3)+16*1024usize.pow(2)+budget-recycled_bytes;
            if admit_image(fits)? {
                let bank_retained=crate::config::bank_encoder_checkpoint_requested();
                let regions=self.state_image_regions_for_layers_mode(self.history.position as usize,true,true,bank_retained)?;
                let mut allocation=match recycled {
                    Some(retired)=>retired.try_recycle(&regions,budget,stream)?,
                    None=>crate::state_image::Image::try_allocate(&self.cuda,&regions,budget,stream)?,
                };
                let fits=allocation.is_some() && self.cuda.admission_memory()?.admitted_available
                    >=8*1024usize.pow(3)+16*1024usize.pow(2);
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
        let mut progress=self.prefill_wide_interruptible(wides,suffix,comm,stream,None,
            &mut |done|should_yield(prefix.len()+done))?;
        progress.done+=prefix.len();
        Ok((saved,serde_json::json!({"restore_seconds":restore_seconds,"consumed_parent":consumed_parent,"prefix_seconds":prefix_seconds,"save_seconds":save_seconds,
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
        self.invalidate_gather_caches();
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
            && self.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+64*1024usize.pow(2),
            "alternate QKV screen memory/ownership");
        self.layer_owners().into_iter().map(|l|l.attention().legacy_qkv_for_probe(kernels)).collect()
    }
    pub(crate) fn prepare_qkv_fence_alternates_for_probe(&self,kernels:&Path)->Result<Vec<crate::fp8_linear::Linear>> {
        ensure!(std::env::var("CANADA_SCRATCH_QKV_FENCE_SCREEN").as_deref()==Ok("1")
            && matches!(self.rows,6|16)&&self.pending.is_none()&&self.lease.is_current()
            &&self.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+64*1024usize.pow(2),
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
            &&self.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+64*1024usize.pow(2),
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
            && self.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+64*1024usize.pow(2),
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
    fn invalidate_gather_caches(&self) {
        for q in self.index_queries().into_iter().flatten() {q.invalidate_gather_cache();}
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
        let sparse=self.history.position>=65536;
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
    pub(crate) fn verification_poll_us_requested()->Result<u64> {
        match std::env::var("CANADA_SCRATCH_VERIFY_POLL_US") {
            Ok(value)=>match value.as_str(){"0"=>Ok(0),"25"=>Ok(25),
                _=>anyhow::bail!("verification polling interval must be0or25microseconds")},
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
            && (self.history.position>=65536)!=self.adaptive_sparse_active;
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
                ensure!(enabled && self.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+32*1024usize.pow(2),
                    "tiny append graph metadata reserve");
                let pairs=(COMPRESSED_OFFSET..COMPRESSED_OFFSET+self.context/2).map(|x|x as i32).collect::<Vec<_>>();
                let singles=(COMPRESSED_OFFSET..COMPRESSED_OFFSET+self.context).map(|x|x as i32).collect::<Vec<_>>();
                let decoder=Step::new(&self.cuda,&self.source2,self.group8.source(),self.group14.source(),
                    self.late.source(),&pairs,&singles,self.rows,self.context,self.early_caches[0].capacity(),self.swa_layout,Phase::Extend)?;
                self.tiny_append_graphs=Some(TinyAppendGraphs {
                    graphs:(0..=self.rows).map(|_|None).collect(),engram_graphs:(0..=self.rows).map(|_|None).collect(),
                    alternate_index_graphs:(0..=self.rows).map(|_|None).collect(),
                    alternate_index_engram_graphs:(0..=self.rows).map(|_|None).collect(),
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
            ensure!(self.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+32*1024usize.pow(2),
                "tiny append metadata must preserve the operating reserve");
            let pairs=(COMPRESSED_OFFSET..COMPRESSED_OFFSET+self.context/2).map(|x|x as i32).collect::<Vec<_>>();
            let singles=(COMPRESSED_OFFSET..COMPRESSED_OFFSET+self.context).map(|x|x as i32).collect::<Vec<_>>();
            self.cached_decoder=Some(Step::new(&self.cuda,&self.source2,self.group8.source(),
                self.group14.source(),self.late.source(),&pairs,&singles,6,self.context,
                self.early_caches[0].capacity(),self.swa_layout,Phase::Extend)?);
        } else {self.cached_decoder=None;}
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
        crate::carry_state::Snapshot::new(&self.cuda, rings)
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
        ensure!(self.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+64*1024usize.pow(2),
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
            ensure!(self.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+64*1024usize.pow(2),
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
        (self.compact_sampling_enabled||(self.local_target_enabled&&self.rows==6
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
        stream.synchronize()?;
        comm.health()?;
        self.lease.begin()?;
        self.invalidate_gather_caches();
        for cache in &mut self.early_caches {
            cache.clear(&self.cuda, stream)?;
        }
        self.source2.clear_state(stream)?;
        self.group8.clear_state(&self.cuda, stream)?;
        self.group14.clear_state(&self.cuda, stream)?;
        self.late.clear_state(&self.cuda, stream)?;
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
        // Live count is agreed by all ranks and is already a graph-cache key.
        // Keep both projections bound to their own stream so changing widths
        // does not destroy a warmed graph or rebind mutable scratch.
        if self.engram_projection_ahead {
            self.engram_projection_active=self.engram_overlap&&self.graph_replay
                &&(!self.engram_projection_m3_only||ids.len()==3);
            self.group14.select_engram_projection_ahead(self.engram_projection_active)?;
        }
        self.select_index_graph_bank(stream)?;
        let mut marks = vec![("begin", begin)];
        let mut captured_new_graph = false;
        let mut split_engram14=self.engram_split14 && (!self.engram_split14_m3_only || ids.len()==3);
        let retain_warm_prefill = self.prefill_capture_once && !verification
            && self.rows == 128 && self.source_phase == Phase::Extend;
        let retain_warm_result = retain_warm_prefill || (self.verification_capture_once
            && verification && [6,16].contains(&self.rows) && self.source_phase == Phase::Verify);
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
        marks.push(("history_hash", Instant::now()));
        ensure!(!self.engram_overlap || (verification && self.graph_replay && [6,16,32].contains(&self.rows)),
            "experimental Engram overlap is verification-only");
        let deferred_rows = self.engram_overlap && self.engram_graphs[ids.len()].is_some();
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
        }
        self.io.update_tokens(&mut self.tokens, self.history.position as usize,
            &padded.iter().copied().map(i64::from).collect::<Vec<_>>(),
            ids.len(),
        )?;
        marks.push(("token_upload", Instant::now()));
        self.step.update(&p, stream)?;
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
            || (self.graph_replay && self.graphs[ids.len()].is_none())
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
        if self.graph_replay {
            let missing = if self.engram_overlap { self.engram_graphs[ids.len()].is_none() }
                else { self.graphs[ids.len()].is_none() };
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
                if self.engram_overlap {
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
                    self.engram_graphs[ids.len()] = Some(parts);
                } else {
                    self.graphs[ids.len()] = Some(stream.capture(|| self.enqueue(comm, stream))?);
                    self.graph_captures += 1;
                }
                marks.push(("graph_capture", Instant::now()));
            }
            if captured_new_graph && retain_warm_result {
                // All Engram rows were prepared before the eager first use;
                // only existing graph replays may defer those uploads. Both
                // ordinary and split graphs now describe the next execution.
                ensure!(!deferred_rows, "first capture unexpectedly deferred Engram rows");
            } else if self.engram_overlap {
                unsafe { self.engram_graphs[ids.len()].as_ref().unwrap()[0].launch(stream)?; }
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
                unsafe { self.engram_graphs[ids.len()].as_ref().unwrap()[middle_index].launch(stream)?; }
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
                    unsafe{self.engram_graphs[ids.len()].as_ref().unwrap()[tail_index].launch(stream)?;}
                }
            } else {
                unsafe { self.graphs[ids.len()].as_ref().unwrap().launch(stream)?; }
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
            if self.graph_replay {
                if self.local_target_gather[ids.len()].is_none() {
                    self.io.gather_logits(comm,stream)?;stream.synchronize()?;comm.health()?;
                    self.local_target_gather[ids.len()]=Some(stream.capture(||self.io.gather_logits(comm,stream))?);
                    self.graph_captures+=1;
                }else{unsafe{self.local_target_gather[ids.len()].as_ref().unwrap().launch(stream)?;}}
            }else{self.io.gather_logits(comm,stream)?;}
        }
        if verification&&self.verification_poll_us>0&&self.cached_decoder.is_none() {
            stream.synchronize_polling(self.verification_poll_us)?;
        }else{stream.synchronize()?;}
        comm.health()?;
        marks.push(("gpu_completion", Instant::now()));
        self.check_candidates()?;
        if let Some(cache) = &self.encoder_cache {
            ensure!(cache.error()? == 0, "encoder tail publication fault");
        }
        marks.push(("candidate_checks", Instant::now()));
        self.last_head_local_only=self.compact_sampling_active()&&!gathered_tail;
        if verification {
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
            ||std::env::var("CANADA_SCRATCH_ENGRAM_PROJECTION_SCREEN").as_deref()==Ok("1"),
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
        Ok((self.io.logits(), pending.live))
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
    pub(crate) fn pending_local_target_logits(&self)->Result<(&Buffer<f32>,usize)> {
        let pending=self.pending.as_ref().context("no local target verification")?;
        ensure!(self.local_target_enabled&&self.rows==6&&self.last_head_local_only
            &&self.lease.is_busy_revision(),"local target logit ownership");
        Ok((self.io.local_logits(),pending.live))
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
        let result = self.categorical.sample(self.io.logits(), 0, live, 1., seed,
            crate::categorical::TARGET_DOMAIN, self.history.position + 1, stream);
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
    pub fn finish_verification(
        &mut self,
        accepted: usize,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<usize> {
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
        let canonical = std::env::var("CANADA_SCRATCH_CANONICAL_PREFILL").as_deref() == Ok("1");
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
        let record_stalls=std::env::var("CANADA_SCRATCH_PREFILL_STALL_COUNTERS").as_deref()==Ok("1");
        let mut stall_counters=Vec::new();
        let available: Vec<usize> = std::iter::once(128).chain(wides.iter().map(|w| w.rows)).collect();
        let lookahead=if lookahead {
            ensure!(std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref() == Ok("1"),
                "Engram lookahead requires the bounded io_uring reader");
            // At8K rows the preview briefly holds two chunks plus split-off
            // hashes (~9MiB); flat IDs and two4096-row byte buffers fit below
            // the16MiB allowance. Preserve the8GiB operating reserve.
            let memory=self.cuda.admission_memory()?;
            ensure!(memory.admitted_available>=8*1024usize.pow(3),
                "prefill is below reserved UMA headroom: {} bytes",memory.admitted_available);
            let fits=memory.admitted_available>=8*1024usize.pow(3)+16*1024usize.pow(2);
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
        ensure!(profile_start.is_none() || (!encoder_graph && profile_start==Some(131072)),
            "bounded encoder profile requires the current eager path at131072");
        let chunks=if let Some(chunks)=resume_chunks {
            crate::config::validate_encoder_chunks(ids.len(),&chunks,&available)?;chunks
        }else{crate::config::plan_encoder_chunks(ids.len(), &available)};
        for (chunk_index, &rows) in chunks.iter().enumerate() {
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
            let profile_this_chunk=profile_start==Some(previous+done);
            if profile_this_chunk {
                ensure!(rows==8192 && chunk.len()==8192,"bounded profile must contain one full8K encoder chunk");
                stream.synchronize()?;target.cuda.profiler_start()?;
            }
            let chunk_graph=encoder_graph || (tiny_encoder_graph && rows==128 && chunk.len()==128);
            let counters=record_stalls.then(crate::prefill_stall::Snapshot::read);
            let chunk_start=Instant::now();
            let chunk_result=target.encoder_chunk(chunk, next.as_deref(), &cache, comm, stream, chunk_graph);
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
            if done<ids.len() && should_yield(done)? { break; }
        }
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
    ) -> Result<(Instant, bool, Option<LookaheadMetrics>)> {
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
        let batch = self.history.prepare_padded(&padded, chunk.len(), &self.hashes)?;
        self.prepare_engram_rows(&batch.rows)?;
        self.io.update_tokens(&mut self.tokens, self.history.position as usize,
            &padded.iter().copied().map(i64::from).collect::<Vec<_>>(),
            chunk.len(),
        )?;
        self.step.update(&p, stream)?;
        comm.select_decode_reduction(false, stream)?;
        comm.select_prefill_rows(chunk.len(), rows, stream)?;
        stream.join_default()?;
        if let Some(wide)=&self.wide {wide.prepare_moe_scratch(stream)?;}
        let _partial_rows=crate::fp8_linear::LivePrefillRowsScope::new(
            self.prefill_partial_dense_rows && rows==512 && (129..512).contains(&chunk.len())
                ||self.prefill_partial1024_rows>0&&rows==1024&&(513..1024).contains(&chunk.len()),
            self.prefill_partial_moe_rows && rows==512 && (129..512).contains(&chunk.len())
                ||self.prefill_partial1024_rows>1&&rows==1024&&(513..1024).contains(&chunk.len()),
            chunk.len(),rows,stream)?;
        let _mhc_rows=crate::mhc::ActiveRowsScope::new(
            self.prefill_mhc_live_rows&&rows==512&&(129..512).contains(&chunk.len()),
            chunk.len(),rows,stream,self.prefill_mhc_live_norm)?;
        let prepared = Instant::now();
        #[cfg(feature="storage-fault-injection")]
        let layer0_trace=end==16896&&std::env::var("CANADA_SCRATCH_PREFILL_LAYER0_TRACE").as_deref()==Ok("1");
        #[cfg(not(feature="storage-fault-injection"))]
        let layer0_trace=false;
        if layer0_trace {
            // A bounded eager diagnostic captures row-arena values before
            // layer1 can overwrite them. Later compare final logits with the
            // ordinary captured path to detect diagnostic perturbation.
            self.enqueue_initial(comm,stream)?;
            #[cfg(feature="storage-fault-injection")]
            self.dump_prefill_layer0_trace(previous,chunk,stream)?;
            self.enqueue_encoder_after_initial(comm,stream)?;
        }else if graph && !captured {
            unsafe { self.encoder_graph.as_ref().unwrap().launch(stream) }?;
        } else {
            self.enqueue_encoder(comm, stream)?;
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
                    comm.health()?;
                    self.check_candidates()?;
                    ensure!(cache.error()? == 0, "prefill encoder publication fault");
                    Ok(())
                })();
                // Drain the bounded I/O task even if the GPU failed. Neither
                // path may leave background work behind a released writer.
                let gpu_wait_ms=started.elapsed().as_secs_f64()*1000.;
                let join_started=Instant::now();
                let warmed = worker.join().map_err(|_| anyhow::anyhow!("Engram prefetch panicked"));
                let join_wait_ms=join_started.elapsed().as_secs_f64()*1000.;
                executed?;
                let worker_ms=warmed??;
                Ok(LookaheadMetrics {next_rows:next.len(),worker_ms,gpu_wait_ms,join_wait_ms})
            })?)
        } else {
            self.wait_encoder_chunk(stream)?;
            comm.health()?;
            self.check_candidates()?;
            ensure!(cache.error()? == 0, "prefill encoder publication fault");
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

    fn enqueue_initial(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        self.io.run_input(&self.tokens, comm, stream)?;
        self.early[0].run(
            self.io.input(),
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
    fn enqueue_before_engram14(&mut self, comm:&Roce, stream:&Stream)->Result<()> {
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
            Some((residual, pre)) if self.group8.sequence_parallel() => {
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
        let (output20, pre20) = match self.sp_full.as_mut() {
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
            && self.source_phase != Phase::Verify
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
    fn enqueue_after_initial(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        self.enqueue_before_engram14(comm,stream)?;
        self.enqueue_after_engram14(comm,stream)
    }
    fn enqueue_after_engram14(&mut self,comm:&Roce,stream:&Stream)->Result<()> {
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
