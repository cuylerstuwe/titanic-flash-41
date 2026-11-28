//! Independent Rust MLA attention dataflow. This first supported geometry is
//! TP4 with six verify rows. It emits this rank's output before the collective
//! and mHC post-combine; scheduler/retained-state integration is separate.
use crate::{
    attention::{Candidates, Execution, PagedKv},
    blockscaled::Matrix,
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, Event, Module, Stream},
    fp8_linear::Linear,
    kv::{RowLocations, Store},
    prelude::{FrequencyTable, Native, Positions, QStore},
    roce::Roce,
    weights::Weights,
    woa::Grouped,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::{cell::{Cell, RefCell}, path::Path, rc::Rc};

pub(crate) fn wide_epilogues_enabled()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_WIDE_EPILOGUES") {
        Ok(v)=>{ensure!(v=="0"||v=="1","invalid wide epilogue policy");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(false),Err(e)=>Err(e.into()),
    }
}

pub(crate) fn generation_fp32_attention_enabled()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_GENERATION_FP32_ATTENTION") {
        Err(std::env::VarError::NotPresent)=>Ok(false),
        Ok(v)if v=="0"=>Ok(false),Ok(v)if v=="1"=>Ok(true),
        _=>anyhow::bail!("generation FP32 attention must be0or1"),
    }
}

pub(crate) fn generation_swa_bf16_enabled()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_GENERATION_SWA_BF16") {
        Err(std::env::VarError::NotPresent)=>Ok(false),
        Ok(v)if v=="0"=>Ok(false),Ok(v)if v=="1"=>Ok(true),
        _=>anyhow::bail!("generation SWA BF16 attention must be0or1"),
    }
}

pub(crate) fn generation_fp32_attention16_enabled()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_GENERATION_FP32_ATTENTION16") {
        Err(std::env::VarError::NotPresent)=>Ok(false),
        Ok(v)if v=="0"=>Ok(false),Ok(v)if v=="1"=>Ok(true),
        _=>anyhow::bail!("generation M16 FP32 attention must be0or1"),
    }
}

fn history32_fp32_attention_enabled()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_HISTORY32_FP32_ATTENTION") {
        Err(std::env::VarError::NotPresent)=>Ok(false),Ok(v)if v=="0"=>Ok(false),
        Ok(v)if v=="1"=>{
            ensure!(crate::config::history32_enabled(),"M32 precision requires its admitted capability");
            Ok(true)
        },
        _=>anyhow::bail!("M32 FP32 attention must be0or1"),
    }
}

fn query_overlap16_enabled() -> Result<bool> {
    if !crate::mhc::attention16_overlap_enabled()?
        || std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref() != Ok("1") {
        return Ok(false);
    }
    // The resident candidate sets the M6 query schedule after construction;
    // its environment intentionally has no ATTENTION_Q_OVERLAP override.
    // Apply the same candidate policy to the fresh M16 owner before capture.
    let policy = match std::env::var("CANADA_SCRATCH_CANDIDATE_SCHEDULE") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => "combined".to_owned(),
        Err(error) => return Err(error.into()),
    };
    ensure!(["control", "engram", "combined"].contains(&policy.as_str()),
        "unknown candidate schedule policy");
    Ok(policy == "combined")
}

#[derive(Clone, Copy, Debug)]
pub enum InputMode {
    Separate,
    Shared,
    Fused,
    Split(usize),
}
impl InputMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "separate" => Ok(Self::Separate),
            "shared" => Ok(Self::Shared),
            "fused" => Ok(Self::Fused),
            _ => anyhow::bail!("unknown attention input mode"),
        }
    }
}
/// A six-row decode layer's L2 hint queue under one plan (config::QueuePlan): QKV-A, wq_b, wo_a and wo_b in
/// read order. W1 runs beside the previous layer's FFN sum (that layer begins it), W2 beside this layer's QKV
/// gather, W3 from the attention's start; all are joined after wo_b.
pub(crate) struct HintQueue {
    pub(crate) w1: Option<crate::l2_prefetch::Window>,
    /// When present, the previous layer begins `w1` only while this is set (config::l2_queue_engram_lead: the
    /// Engram projection hint of layers 1 and 14, switched per step and by the plan screen).
    pub(crate) w1_gate: Option<Rc<std::cell::Cell<bool>>>,
    w2: Option<crate::l2_prefetch::Window>,
    /// config::l2_queue_index_lead: W2 with the index work's cold weights ahead of wq_b (indexed layers only).
    w2_lead: Option<crate::l2_prefetch::Window>,
    w3: Option<crate::l2_prefetch::Window>,
    /// The KV pages of the attention's window, hinted just before W2 (plan spec `k`).
    kv: Option<crate::l2_prefetch::Window>,
    /// Plan spec `t`: the previous layer starts `w1` in its experts' tail (mhc::tail_point).
    pub(crate) w1_tail: bool,
    /// Plan spec `e`: the previous layer starts `w1` when its routed experts end (mhc::routed_end_point).
    pub(crate) w1_routed_end: bool,
}
/// Where a registered first window begins: at the gate on the shared branch, or when the routed experts end.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TailKind { Gate, RoutedEnd, Any }
thread_local! {
    static TAIL_W1: std::cell::RefCell<Vec<(usize, TailKind, Rc<HintQueue>)>> = const { std::cell::RefCell::new(Vec::new()) };
}
/// The next layer's first hint window waits for a point inside the FFN of the layer running on `parent`.
pub(crate) fn tail_w1_register(parent: &Stream, queue: Rc<HintQueue>) {
    let kind = if queue.w1_tail { TailKind::Gate } else { TailKind::RoutedEnd };
    TAIL_W1.with(|t| t.borrow_mut().push((parent.raw() as usize, kind, queue)));
}
/// The windows registered for `parent` and `kind`, removed from the list (taken by their point, or, with
/// `TailKind::Any`, dropped by the layer's FFN finish when the point never came).
pub(crate) fn tail_w1_take(parent: &Stream, kind: TailKind) -> Vec<Rc<HintQueue>> {
    TAIL_W1.with(|t| {
        let mut t = t.borrow_mut();
        let (hit, rest): (Vec<_>, Vec<_>) = t.drain(..)
            .partition(|x| x.0 == parent.raw() as usize && (kind == TailKind::Any || x.1 == kind));
        *t = rest;
        hit.into_iter().map(|x| x.2).collect()
    })
}
/// The byte range [start, start + bytes) of `regions` taken in order, as (address, size) pairs.
fn range_w2(regions: &[(u64, usize)], start: usize, bytes: usize) -> Result<Vec<(u64, usize)>> {
    let (mut skip, mut left, mut out) = (start / 16 * 16, bytes, Vec::new());
    for &(address, size) in regions {
        if skip >= size { skip -= size; continue; }
        let take = (size - skip).min(left);
        if take > 0 { out.push((address + skip as u64, take)); left -= take; }
        skip = 0;
    }
    Ok(out)
}
pub struct Layer {
    cuda: Cuda,
    rows: usize,
    mode: InputMode,
    qa: Linear,
    /// config::qkv_narrow_prequant: the other lowering of the same product (the one not in `qa`), if installed.
    qa_alternate: Option<Linear>,
    gathered_qkv: Option<Buffer<bf16>>,
    qb: Linear,
    hint_queues: Vec<Rc<HintQueue>>,
    hint_queue_selected: Rc<std::cell::Cell<Option<usize>>>,
    /// CANADA_SCRATCH_L2_QUEUE_INDEX_EARLY: begin the third window before a layer's index work.
    index_early: std::cell::Cell<bool>,
    index_lead: std::cell::Cell<bool>,
    qb_prefetch: Option<crate::l2_prefetch::Window>,
    woa_prefetch: Option<crate::l2_prefetch::Window>,
    woa_prefetch_active: bool,
    kv: Option<Linear>,
    ob: Linear,
    normalized: Buffer<bf16>,
    q_stride: usize,
    qstore: Query,
    /// The owning SP layer exchanges WO-A rows and runs full-K WO-B itself.
    wob_deferrable: bool,
    wide_qrope: Option<crate::triton_ops::Operator>,
    kstore: Store,
    decode: Execution,
    append_decode: Option<Execution>,
    append_decode_active: bool,
    split_decode:Option<Execution>,
    split_decode_active:bool,
    append_split:Option<Execution>,
    append_split_active:bool,
    woa: WoA,
    native: Native,
    frequencies: Rc<FrequencyTable>,
    qnorm_weight: Buffer<bf16>,
    query_norm_quant: Option<(crate::primitives::Primitives, i32)>,
    kvnorm_weight: Buffer<bf16>,
    query_overlap: Option<QueryOverlap>,
    /// config::mhc_quarter_stats: the attention boundary's statistics partials ride this layer's QKV gather.
    quarter: Option<Rc<crate::mhc::QuarterLink>>,
}
struct QueryOverlap {
    side: Rc<Stream>,
    ready: Event,
    done: Event,
    parent: usize,
    pending: Cell<bool>,
}
enum Query {
    Small(QStore),
    Medium(Buffer<bf16>),
}
/// Wide prefill may apply RoPE in place on the Q-B output and feed it to attention,
/// skipping the full Q copy (identical values; wide profiles run eagerly).
/// 2 = unresolved (CANADA_SCRATCH_PREFILL_Q_INPLACE), else 0/1.
static Q_INPLACE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
pub(crate) fn q_inplace_enabled() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match Q_INPLACE.load(Relaxed) {
        2 => { let on = std::env::var("CANADA_SCRATCH_PREFILL_Q_INPLACE").as_deref() == Ok("1");
            Q_INPLACE.store(on as u8, Relaxed); on }
        v => v == 1,
    }
}
/// Sequence-parallel WO-B exchange: construction (CANADA_SCRATCH_PREFILL_WOB_EXCHANGE=1)
/// stages full WO-B and buffers; this runtime selection (default on) lets one
/// allocation compare it with the reduce-scatter path. 2 = unresolved.
static WOB_EXCHANGE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
pub(crate) fn wob_exchange_constructed() -> bool {
    std::env::var("CANADA_SCRATCH_PREFILL_WOB_EXCHANGE").as_deref() == Ok("1")
}
pub(crate) fn wob_exchange_active() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match WOB_EXCHANGE.load(Relaxed) {
        2 => { let on = wob_exchange_constructed(); WOB_EXCHANGE.store(on as u8, Relaxed); on }
        v => v == 1,
    }
}
pub(crate) fn set_wob_exchange_for_probe(enabled: bool) {
    WOB_EXCHANGE.store(enabled as u8, std::sync::atomic::Ordering::Relaxed);
}
/// MXFP8 WO-A exchange with per-shard WO-B (CANADA_SCRATCH_PREFILL_WOB8=1 constructs;
/// this runtime selection defaults on when constructed). 2 = unresolved.
static WOB8: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
pub(crate) fn wob8_constructed() -> bool {
    std::env::var("CANADA_SCRATCH_PREFILL_WOB8").as_deref() == Ok("1")
}
pub(crate) fn wob8_active() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match WOB8.load(Relaxed) {
        2 => { let on = wob8_constructed(); WOB8.store(on as u8, Relaxed); on }
        v => v == 1,
    }
}
/// WOB8 rotation sum as one ordered four-part kernel (ca_wob8_sum4, same per-step arithmetic as
/// the three ca_ffn_add launches). 2 = unresolved (CANADA_SCRATCH_PREFILL_WOB8_SUM4).
static WOB8_SUM4: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
pub(crate) fn wob8_sum4_active() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match WOB8_SUM4.load(Relaxed) {
        2 => { let on = std::env::var("CANADA_SCRATCH_PREFILL_WOB8_SUM4").as_deref() == Ok("1"); WOB8_SUM4.store(on as u8, Relaxed); on }
        v => v == 1,
    }
}
pub(crate) fn set_wob8_sum4_for_probe(enabled: bool) {
    WOB8_SUM4.store(enabled as u8, std::sync::atomic::Ordering::Relaxed);
}
pub(crate) fn set_wob8_for_probe(enabled: bool) {
    WOB8.store(enabled as u8, std::sync::atomic::Ordering::Relaxed);
}
pub(crate) fn set_q_inplace_for_probe(enabled: bool) {
    Q_INPLACE.store(enabled as u8, std::sync::atomic::Ordering::Relaxed);
}
impl Query {
    fn output(&self) -> &Buffer<bf16> {
        match self {
            Self::Small(v) => v.output(),
            Self::Medium(v) => v,
        }
    }
}
enum WoA {
    Small(Grouped),
    Medium(crate::woa_medium::Grouped),
}
impl Layer {
    pub(crate) fn set_wide_epilogues_for_probe(&mut self,enabled:bool,kernels:&Path)->Result<()> {
        ensure!([16,32].contains(&self.rows)&&matches!(self.qstore,Query::Medium(_))
            &&std::env::var("CANADA_SCRATCH_WIDE_EPILOGUE_SCREEN").as_deref()==Ok("1"),
            "isolated wide query-store screen");
        self.wide_qrope=if enabled {
            Some(crate::triton_ops::Operator::load(&self.cuda,&kernels.join("q-rope-cuda-v1"),"q_rope_small")?)
        }else{None};Ok(())
    }
    pub(crate) fn cached_woa_allocation(&self) -> Option<(u64, usize)> {
        match &self.woa { WoA::Medium(v) => v.cached_allocation(), _ => None }
    }
    /// Plan screen: the native WO-A partial kernel on or off (config::woa_native_screen).
    pub(crate) fn set_index_early_for_probe(&self, on: bool) { self.index_early.set(on); }
    pub(crate) fn set_index_lead_for_probe(&self, on: bool) { self.index_lead.set(on); }
    /// A layer whose index work sits between the query projection and the attention: begin the selected plan's
    /// third window now (when enabled) so the output projections' weights arrive while that work runs.
    pub(crate) fn begin_output_hint_early(&self, stream: &Stream) -> Result<()> {
        if !self.index_early.get() { return Ok(()); }
        if let Some(w) = self.selected_hint_queue().and_then(|q| q.w3.as_ref()) {
            if !w.is_open() { w.begin(stream)?; }
        }
        Ok(())
    }
    /// Plan screen: every installed direct-read projection (wq_b, QKV-A) on or off; `None` puts each back on its
    /// configured setting.
    pub(crate) fn set_wqb_direct_for_probe(&self, on: Option<bool>) -> Result<()> {
        if self.qb.direct_installed() {
            let v = if crate::config::wqb_direct_screen() {
                on.unwrap_or_else(crate::config::wqb_direct)
            } else {
                crate::config::wqb_direct()
            };
            self.qb.set_direct_for_probe(v)?;
        }
        if self.qa.direct_installed() {
            let v = if crate::config::qkva_direct_screen() {
                on.unwrap_or_else(crate::config::qkva_direct)
            } else {
                crate::config::qkva_direct()
            };
            self.qa.set_direct_for_probe(v)?;
        }
        if self.ob.direct_installed() {
            let v = if crate::config::wob_direct_screen() {
                on.unwrap_or_else(crate::config::wob_direct)
            } else {
                crate::config::wob_direct()
            };
            self.ob.set_direct_for_probe(v)?;
        }
        if let WoA::Small(v) = &self.woa {
            if v.fused_direct_installed() {
                let flag = if crate::config::woa_fused_direct_screen() {
                    on.unwrap_or_else(crate::config::woa_fused_direct)
                } else {
                    crate::config::woa_fused_direct()
                };
                v.set_fused_direct_for_probe(flag)?;
            }
        }
        Ok(())
    }
    /// Plan screen: the QKV-A and wq_b outputs of this layer (bytes), for a kernel-against-kernel comparison on a
    /// layer whose input does not depend on the kernels compared.
    pub(crate) fn direct_outputs_for_probe(&self) -> Result<[Vec<u8>; 2]> {
        Ok([self.qa.output_bytes_for_probe()?, self.qb.output_bytes_for_probe()?])
    }
    pub(crate) fn set_woa_native_for_probe(&self, on: bool) -> Result<()> {
        match &self.woa { WoA::Small(v) => v.set_native_for_probe(on),
            WoA::Medium(v) => v.set_native_for_probe(on),
            _ => anyhow::bail!("native WO-A covers the 2..8-row and the compact 16-row profiles") }
    }
    pub(crate) fn set_cached_woa_for_probe(&self, enabled: bool) -> Result<()> {
        match &self.woa { WoA::Medium(v) => v.set_cached_for_probe(enabled),
            _ => anyhow::bail!("WO-A cache probe requires a medium profile") }
    }
    pub(crate) fn set_compact_woa_for_probe(&self,mode:u8)->Result<()> {
        match &self.woa {WoA::Medium(v)=>v.set_compact_mode_for_probe(mode),
            _=>anyhow::bail!("compact WO-A requires a medium profile")}
    }
    /// The fresh target profile owns this policy; call before any graph capture.
    pub(crate) fn set_query_overlap(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!((self.rows <= 8 || (self.rows==16
                && std::env::var("CANADA_SCRATCH_ATTENTION16_SCREEN").as_deref()==Ok("1")))
            && self.cuda.owns_stream(stream)
            && self.query_overlap.as_ref().is_none_or(|q|
                !q.pending.get() && q.parent == stream.raw() as usize),
            "query schedule needs an idle small-row owner and its parent stream");
        if enabled && self.query_overlap.is_none() {
            self.query_overlap = Some(QueryOverlap {
                side: self.cuda.query_stream(stream)?, ready: self.cuda.fence_event()?,
                done: self.cuda.fence_event()?, parent: stream.raw() as usize,
                pending: Cell::new(false),
            });
        } else if !enabled {
            self.query_overlap = None;
        }
        Ok(())
    }
    pub(crate) fn set_verify16_query_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(self.rows==16 && (std::env::var("CANADA_SCRATCH_VERIFY16_ATTN_ONLY").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_GROUPED16_SERIAL_QUERY").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_GROUPED16_SERIAL_ATTENTION").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_QKV_SCHEDULE_SCREEN").as_deref()==Ok("1")),
            "idle M16 attention schedule screen");
        // Target joined old side work and destroyed its graphs before rebinding.
        unsafe {self.qb.release_verify16_probe_binding()?;}
        self.set_query_overlap(enabled,stream)
    }
    pub fn new(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: Rc<Weights>,
        layer: usize,
        kernels: &Path,
        frequencies: Rc<FrequencyTable>,
        mode: InputMode,
    ) -> Result<Self> {
        Self::new_rows(
            cuda,
            module,
            stream,
            weights,
            layer,
            kernels,
            frequencies,
            mode,
            6,
            None,
        )
    }
    pub fn new_rows(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: Rc<Weights>,
        layer: usize,
        kernels: &Path,
        frequencies: Rc<FrequencyTable>,
        mode: InputMode,
        rows: usize,
        medium_scratch: Option<Rc<RefCell<crate::woa_medium::Scratch>>>,
    ) -> Result<Self> {
        Self::new_profile(
            cuda,
            module,
            stream,
            weights,
            crate::layer_id::LayerId::target(layer)?,
            kernels,
            frequencies,
            mode,
            rows,
            medium_scratch,
        )
    }
    pub fn new_draft(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: Rc<Weights>,
        stage: usize,
        kernels: &Path,
        frequencies: Rc<FrequencyTable>,
    ) -> Result<Self> {
        Self::new_profile(
            cuda,
            module,
            stream,
            weights,
            crate::layer_id::LayerId::draft(stage)?,
            kernels,
            frequencies,
            InputMode::Shared,
            5,
            None,
        )
    }
    fn new_profile(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: Rc<Weights>,
        id: crate::layer_id::LayerId,
        kernels: &Path,
        frequencies: Rc<FrequencyTable>,
        mode: InputMode,
        rows: usize,
        medium_scratch: Option<Rc<RefCell<crate::woa_medium::Scratch>>>,
    ) -> Result<Self> {
        let draft = id.is_draft();
        let layer = id.key();
        let mode=if !draft && rows==6
            && std::env::var("CANADA_SCRATCH_FUSED_QKV_VERIFY").as_deref()==Ok("1") {
            ensure!(std::env::var("CANADA_SCRATCH_FUSED_QKV_AB").as_deref()==Ok("1")
                && matches!(mode,InputMode::Split(_)),"fused Verify6 requires isolated split-control comparison");
            InputMode::Fused
        } else {mode};
        ensure!(
            (draft && rows == 5 && matches!(mode, InputMode::Shared | InputMode::Separate))
                || (!draft
                    && crate::config::target_rows(rows)
                    && (rows == 6
                        || matches!(mode, InputMode::Split(_))
                        // Wide encoders compute the full Q-A/KV projection
                        // locally instead of gathering four 448-column shards.
                        || (crate::config::wide_rows(rows) && matches!(mode, InputMode::Fused)))),
            "attention execution profile"
        );
        ensure!(
            frequencies.compressed() == (!draft && layer >= 2)
                && cuda.owns_buffer(frequencies.values()),
            "attention layer canonical rotary-table contract"
        );
        stream.join_default()?;
        let attn = format!("{}.attn", id.prefix());
        let projection = |name: &str, directory: &str| -> Result<Linear> {
            let matrix =
                Matrix::prepare(cuda, module, stream, &weights, &format!("{attn}.{name}"))?;
            Linear::new(cuda, &kernels.join(directory), matrix)
        };
        let qa = if let InputMode::Split(rank) = mode {
            Linear::new(
                cuda,
                &kernels.join(if rows == 6 && crate::fp8_linear::narrow_enabled() {
                    "fp8-flashinfer-narrow-qkv-m6-v1".into()
                } else { format!("fp8-split-qkv-m{rows}-v1") }),
                Matrix::prepare_qkv_split(cuda, module, stream, &weights, layer, rank)?,
            )?
        } else if matches!(mode, InputMode::Fused) {
            Linear::new(
                cuda,
                &kernels.join(format!("fp8-canonical-wqkv_a-r0-m{rows}-v0")),
                Matrix::prepare_qkv(cuda, module, stream, &weights, layer)?,
            )?
        } else {
            projection(
                "wq_a",
                if draft {
                    "fp8-dspark-q-a-m5-v1"
                } else {
                    "fp8-canonical-q-a-m6-v0"
                },
            )?
        };
        let mut qa = qa;
        if !draft && rows == 6 && matches!(mode, InputMode::Split(_))
            && (crate::config::qkva_direct() || crate::config::qkva_direct_screen()) {
            qa.install_direct(crate::config::qkva_direct())?;
        }
        let mut qb = projection(
            "wq_b",
            &if draft {
                "fp8-dspark-wq_b-r0-m5-v1".into()
            } else if rows == 6 && crate::fp8_linear::pdl_enabled() {
                "fp8-flashinfer-pdl-q-b-m6-rust-v2".into()
            } else {
                format!("fp8-canonical-wq_b-r0-m{rows}-v0")
            },
        )?;
        // Stage 1 of the layer build: the six-row wq_b through the direct-read GEMV (config::wqb_direct).
        if !draft && rows == 6 && (crate::config::wqb_direct() || crate::config::wqb_direct_screen()) {
            qb.install_direct(crate::config::wqb_direct())?;
        }
        let kv = if matches!(mode, InputMode::Fused | InputMode::Split(_)) {
            None
        } else {
            Some(projection(
                "wkv",
                if draft {
                    "fp8-dspark-query-kv-m5-v1"
                } else {
                    "fp8-canonical-wkv-r0-m6-v0"
                },
            )?)
        };
        let mut ob = projection(
            "wo_b",
            &if draft {
                "fp8-dspark-wo_b-r0-m5-v1".into()
            } else {
                format!("fp8-canonical-wo_b-r0-m{rows}-v0")
            },
        )?;
        if !draft && rows == 6 && (crate::config::wob_direct() || crate::config::wob_direct_screen()) {
            ob.install_direct(crate::config::wob_direct())?;
        }
        let norm_weight = |name: &str, width: usize| -> Result<Buffer<bf16>> {
            let key = format!("{attn}.{name}.weight");
            let v = weights.view(&key)?;
            ensure!(
                v.dtype == Dtype::Bf16 && v.bytes == width * 2 && v.shape == [width as u64],
                "attention norm weight shape"
            );
            let b = cuda.allocate::<bf16>(width)?;
            unsafe { cuda.copy_device(b.ptr(), weights.ptr(&key)?, b.bytes(), stream) }?;
            Ok(b)
        };
        let qnorm_weight = norm_weight("q_norm", 1280)?;
        let kvnorm_weight = norm_weight("kv_norm", 512)?;
        let q_stride = if matches!(mode, InputMode::Fused | InputMode::Split(_)) {
            1792
        } else {
            1280
        };
        let normalized = cuda.allocate_rows::<bf16>(rows * 1280)?;
        // s592: the same dynamic-row kernel preserves BF16, FP8 and scale
        // bytes at M16. Only admit the compact FlashInfer Q-B profile there.
        let small16 = rows==16
            && std::env::var("CANADA_SCRATCH_VERIFY16_SMALL_KERNELS").as_deref()==Ok("1")
            && std::env::var("CANADA_SCRATCH_VERIFY16_COMPACT_DENSE").as_deref()==Ok("1");
        let query_norm_quant = if !draft && (rows==6 || small16)
            && std::env::var("CANADA_SCRATCH_QUERY_NORM_QUANT_FUSION").as_deref()==Ok("1") {
            let (q,s)=qb.query_quantization_buffers()?;
            ensure!(q.len()==rows*1280 && s.len()==5120,"query RMS quantization output geometry");
            let root=kernels.join("query-norm-quant-v1");
            let library=crate::primitives::Primitives::open(cuda,
                &root.join("libcanada_primitives.so"),&root.join("bindings.json"))?;
            let id=format!("query.rms_quant.s{q_stride}.r1.async0");
            let entries:Vec<_>=library.bindings.entries.iter().filter(|e|e.kernel_id==id).collect();
            ensure!(entries.len()==1 && entries[0].arguments.iter().map(|a|a.kind.as_str()).collect::<Vec<_>>()
                ==["pointer","pointer","pointer","pointer","pointer","i64","f32"],"query RMS quantization ABI");
            let index=entries[0].index;
            Some((library,index))
        } else {None};
        let qstore = if rows <= 8 {
            Query::Small(QStore::new(cuda, &kernels.join("q-rope-cuda-v1"), rows)?)
        } else {
            Query::Medium(cuda.allocate_rows(rows * 8192)?)
        };
        // Target caches follow the process KV format (CANADA_SCRATCH_KV_FORMAT); the drafter's stay V4.
        let kv_v41 = !draft && crate::attention::kv_format()?.window == crate::attention::KvLayout::V41;
        let kstore = Store::open(cuda, &kernels.join(if kv_v41 {"kv-v41-v1"} else {"kv-v2"}))?;
        // Isolated target-append experiment: preserve the BF16 prefill
        // arithmetic at M6 instead of inheriting decode's Q/P FP8 quantization.
        // The drafter and ordinary serving profile retain their own policy.
        let small_bf16=!draft && rows==6
            && std::env::var("CANADA_SCRATCH_SMALL_BF16_ATTENTION").as_deref()==Ok("1");
        let append_bf16=!draft && (rows==6 || (rows==16
            && std::env::var("CANADA_SCRATCH_RESIDENT_MEDIUM_APPEND").as_deref()==Ok("1"))
            ||(rows==32&&crate::config::append32_workspace_enabled()))
            && std::env::var("CANADA_SCRATCH_APPEND_BF16_ONLY").as_deref()==Ok("1");
        ensure!(!(small_bf16 && append_bf16),"choose one small BF16 attention experiment");
        let precision32=!draft&&rows==32&&history32_fp32_attention_enabled()?;
        let generation_fp32=!draft&&layer>=2&&((rows==6&&generation_fp32_attention_enabled()?)
            ||(rows==16&&generation_fp32_attention16_enabled()?)||precision32);
        let sparse16_screen=!draft&&rows==16&&layer>=2
            &&std::env::var("CANADA_SCRATCH_SPARSE_PRECISION_WIDE_SCREEN").as_deref()==Ok("1");
        ensure!(!sparse16_screen||!generation_fp32,"M16 precision screen needs the FP8 control constructor");
        let generation_swa_bf16=!draft&&layer<2
            &&(([6,16].contains(&rows)&&generation_swa_bf16_enabled()?)||precision32);
        ensure!(!generation_swa_bf16||precision32||(append_bf16&&!small_bf16
            &&std::env::var("CANADA_SCRATCH_TINY_APPEND_RETAIN_GRAPHS").as_deref()==Ok("1")),
            "generation SWA BF16 requires the separate BF16 append graph bank");
        ensure!(!generation_fp32||precision32||(append_bf16&&!small_bf16
            &&std::env::var("CANADA_SCRATCH_TINY_APPEND_RETAIN_GRAPHS").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_SPLIT_ATTENTION_SCREEN").as_deref()!=Ok("1")
            &&std::env::var("CANADA_SCRATCH_APPEND_SPLIT_ATTENTION_SCREEN").as_deref()!=Ok("1")),
            "generation FP32 attention requires the separate BF16 append graph bank");
        // s797 native verification and s798 completed-task checks support the
        // higher-precision generation candidate. s799 changes only stores and
        // preserves FP32 intermediates exactly. Append keeps its existing bank.
        // Same-load screen 2026-11-10 (check/bench_attn.py): the adapted pico kernel halves V4.1 attention from 16
        // rows up (8K-row prefill 7.7 vs 13.9 ms, 32-row decode 0.030 vs 0.058 ms); at 6 rows FlashInfer's split
        // decode stays faster. CANADA_SCRATCH_PICO_ATTENTION=0 keeps FlashInfer's kernels everywhere.
        let pico=pico_attention_enabled()? && rows>8;
        let dsv41_prefill=|rows:usize,extra:usize|->Result<Execution>{Ok(Execution::Prefill(
            crate::attention::Prefill::new_dsv41(cuda,&kernels.join(if pico {"attention-pico-prefill-v1"}
                else {"attention-dsv41-prefill-v1"}),rows,extra)?))};
        let dsv41_split = |rows, extra, splits| -> Result<Execution> {
            // Select once during construction so graph replay has no additional branch.
            let bundle = match (rows, extra, splits) {
                (6, 512, 8) => match std::env::var("CANADA_SCRATCH_PREFETCH_SPARSE_PAGES").as_deref() {
                    Ok("1") => "attention-dsv41-decode-prefetch-v1",
                    Ok("2") => "attention-dsv41-decode-prefetch-v2",
                    Ok("0") | Err(std::env::VarError::NotPresent) => "attention-dsv41-decode-v1",
                    _ => anyhow::bail!("CANADA_SCRATCH_PREFETCH_SPARSE_PAGES must be 0, 1 or 2"),
                },
                _ => "attention-dsv41-decode-v1",
            };
            Ok(Execution::Prefill(crate::attention::Prefill::new_dsv41_split(
                cuda, &kernels.join(bundle), rows, extra, splits,
            )?))
        };
        let decode = if kv_v41 {
            // V4.1 caches: FlashInfer v0.7.3's DSv4.1 kernels in the roles of today's primitives. Layers 0-1 read the
            // window only; split decode keeps today's split counts and fp32 partials.
            if generation_swa_bf16 { dsv41_prefill(rows,0)? }
            // Same-load screen 2026-11-10 (check/bench_attn.py): split 8 at 6 rows and split 3 at 16 rows are fastest;
            // at 32 rows the unsplit kernel is (0.058 vs 0.066 ms per layer). Splitting only reorders fp32 sums.
            else if generation_fp32 { if rows==32 || pico { dsv41_prefill(rows,512)? } else { dsv41_split(rows,512,if rows==6 {8}else{3})? } }
            else { dsv41_prefill(rows,if layer>=2 {512}else{0})? }
        }else if generation_swa_bf16 {
            // First-two-layer SWA: canonical BF16 Q/K, FP32 accumulation and
            // BF16 output, using the already qualified append primitive. Its
            // separate append workspace keeps both graph banks independent.
            Execution::Prefill(crate::attention::Prefill::new_small_bf16(cuda,
                &kernels.join("attention-small-bf16-s729"),rows,128,0)?)
        }else if generation_fp32 {
            Execution::Prefill(crate::attention::Prefill::new_split_bf16(cuda,kernels,rows,512,
                if rows==6 {8}else{3},true,true)?)
        }else if small_bf16 {
            Execution::Prefill(crate::attention::Prefill::new_small_bf16(cuda,
                &kernels.join("attention-small-bf16-s729"),rows,128,if layer>=2 {512}else{0})?)
        }else{Execution::new(
            cuda,
            &kernels.join(if rows <= 64 {
                "attention-v1"
            } else if rows > 8192 {
                "attention-prefill-m16384-v1"
            } else if rows > crate::config::DEFAULT_WIDE_ROWS {
                "attention-prefill-m8192-v1"
            } else {
                "attention-prefill-v1"
            }),
            rows,
            if draft { 192 } else { 128 },
            if !draft && layer >= 2 { 512 } else { 0 },
        )?};
        let append_decode=if append_bf16 && kv_v41 {
            Some(dsv41_prefill(rows,if layer>=2 {512}else{0})?)
        }else if append_bf16 {
            Some(Execution::Prefill(crate::attention::Prefill::new_small_bf16(cuda,
                &kernels.join("attention-small-bf16-s729"),rows,128,if layer>=2 {512}else{0})?))
        }else{None};
        let fp32_partials=std::env::var("CANADA_SCRATCH_SPLIT_ATTENTION_FP32").as_deref()==Ok("1");
        let split_decode=if sparse16_screen||(!draft && rows==6 && layer>=2
            && std::env::var("CANADA_SCRATCH_SPLIT_ATTENTION_SCREEN").as_deref()==Ok("1")) {
            ensure!(append_decode.is_some(),"split attention screen needs unsplit BF16 control");
            ensure!(!kv_v41,"the V4 split-attention screens read only V4 caches");
            Some(Execution::Prefill(crate::attention::Prefill::new_split_bf16(cuda,kernels,rows,512,
                if sparse16_screen {3}else{8},sparse16_screen||fp32_partials,sparse16_screen)?))
        }else{None};
        let append_split=if append_bf16&&layer>=2
            &&std::env::var("CANADA_SCRATCH_APPEND_SPLIT_ATTENTION_SCREEN").as_deref()==Ok("1") {
            ensure!(split_decode.is_none()||fp32_partials,"combined split screens require FP32 partial experiment");
            ensure!(!kv_v41,"the V4 split-attention screens read only V4 caches");
            Some(Execution::Prefill(crate::attention::Prefill::new_split_bf16(cuda,kernels,rows,512,
                if rows==16{if fp32_partials{3}else{2}}else{8},fp32_partials,false)?))
        }else{None};
        let woa_prefetch=if !draft && rows==6
            &&std::env::var("CANADA_SCRATCH_WOA_PREFETCH_SCREEN").as_deref()==Ok("1") {
            Some(crate::l2_prefetch::Window::new_woa(cuda,module,stream,weights.clone(),id)?)
        }else{None};
        let mut woa = if rows <= 8 {
            WoA::Small(Grouped::new_named(
                cuda,
                &kernels.join("woa-v1"),
                weights,
                id,
                rows,
            )?)
        } else {
            WoA::Medium(crate::woa_medium::Grouped::new(
                cuda,
                weights,
                layer,
                medium_scratch.ok_or_else(|| {
                    anyhow::anyhow!("medium attention requires shared WO-A scratch")
                })?,
            )?)
        };
        if crate::config::woa_native() || crate::config::woa_native_screen() {
            if let WoA::Small(v) = &mut woa { v.install_native(module, crate::config::woa_native())?; }
        }
        if !draft && rows == 6 && (crate::config::woa_fused_direct() || crate::config::woa_fused_direct_screen()) {
            if let WoA::Small(v) = &mut woa { v.install_fused_direct(crate::config::woa_fused_direct())?; }
        }
        if crate::woa_medium::cached_policy()? {
            if let WoA::Medium(v) = &mut woa { v.prepare_cached(stream)?; }
        }
        if matches!(rows,16|32) {
            let compact=crate::woa_medium::compact_policy()?;
            if compact||std::env::var("CANADA_SCRATCH_WOA_MID_SCREEN").as_deref()==Ok("1") {
                if let WoA::Medium(v)=&woa {v.prepare_compact(kernels,compact,stream)?;}
            }
        }
        let native =
            Native::open_with_rms(cuda, &kernels.join("prelude-v1"), &kernels.join("q-rms-v1"))?;
        let qb_prefetch =
            if matches!(mode, InputMode::Split(_)) && crate::l2_prefetch::enabled_for_rows(rows,2)? {
                Some(crate::l2_prefetch::Window::new(
                    cuda,
                    module,
                    stream,
                    vec![qb.immutable_matrix()],
                    None,
                )?)
            } else {
                None
            };
        stream.synchronize()?;
        Ok(Self {
            cuda: cuda.clone(),
            rows,
            mode,
            qa,
            qa_alternate: None,
            gathered_qkv: if matches!(mode, InputMode::Split(_)) {
                Some(cuda.allocate_rows::<bf16>(rows * 1792)?)
            } else {
                None
            },
            qb,
            hint_queues: Vec::new(),
            hint_queue_selected: Rc::new(std::cell::Cell::new(None)),
            index_early: std::cell::Cell::new(crate::config::l2_queue_index_early()),
            index_lead: std::cell::Cell::new(crate::config::l2_queue_index_lead()),
            qb_prefetch,
            woa_prefetch,
            woa_prefetch_active: false,
            kv,
            ob,
            normalized,
            q_stride,
            qstore,
            wob_deferrable: false,
            wide_qrope:if [16,32].contains(&rows)&&!draft&&wide_epilogues_enabled()? {
                Some(crate::triton_ops::Operator::load(cuda,&kernels.join("q-rope-cuda-v1"),"q_rope_small")?)
            }else{None},
            kstore,
            decode,
            append_decode,
            append_decode_active:false,
            split_decode,split_decode_active:false,
            append_split,append_split_active:false,
            woa,
            native,
            frequencies,
            qnorm_weight,
            query_norm_quant,
            kvnorm_weight,
            query_overlap: if !draft && ((rows <= 8
                && std::env::var("CANADA_SCRATCH_ATTENTION_Q_OVERLAP").as_deref() == Ok("1"))
                || (rows == 16 && query_overlap16_enabled()?)) {
                Some(QueryOverlap { side: cuda.query_stream(stream)?, ready: cuda.fence_event()?,
                    done: cuda.fence_event()?, parent: stream.raw() as usize, pending: Cell::new(false) })
            } else { None },
            quarter: None,
        })
    }
    /// config::mhc_quarter_stats: give this layer's QKV gather a row block of its own (pseudo-rows for the
    /// boundary's statistics partials, then the QKV-A rows, which the direct kernel now writes there) and return
    /// the link the boundary shares. None when this layer's profile does not take it.
    pub(crate) fn install_quarter_link(&mut self, on: bool) -> Result<Option<Rc<crate::mhc::QuarterLink>>> {
        use crate::mhc::{QUARTER_COLUMNS, QUARTER_HEAD, QuarterLink};
        let InputMode::Split(rank) = self.mode else { return Ok(None) };
        if self.rows != 6 || self.gathered_qkv.is_none() || !self.qa.direct_installed() || self.kv.is_some()
            || self.quarter.is_some() { return Ok(None); }
        let mut xin: Buffer<bf16> = self.cuda.allocate((QUARTER_HEAD + self.rows) * QUARTER_COLUMNS)?;
        let mut xout: Buffer<bf16> = self.cuda.allocate((QUARTER_HEAD + self.rows) * QUARTER_COLUMNS * 4)?;
        xin.zero()?;
        xout.zero()?;
        self.qa.set_direct_output(xin.ptr() + (QUARTER_HEAD * QUARTER_COLUMNS * 2) as u64)?;
        // The gathered QKV rows keep one fixed place (behind the pseudo-rows); every consumer reads this view.
        self.gathered_qkv = Some(unsafe { xout.view(QUARTER_HEAD * QUARTER_COLUMNS * 4, self.rows * QUARTER_COLUMNS * 4) }?);
        let link = Rc::new(QuarterLink { rank, on: Cell::new(on), xin, xout, armed: Cell::new(None),
            partials_ready: self.cuda.fence_event()?, gathered: self.cuda.fence_event()? });
        self.quarter = Some(link.clone());
        Ok(Some(link))
    }
    /// Plan screen: the boundary's statistics by rank quarter on or off (the gather follows the boundary).
    pub(crate) fn set_quarter_stats_for_probe(&self, on: bool) -> Result<bool> {
        match &self.quarter {
            Some(link) => { ensure!(link.armed.get().is_none() && (!on || self.qa.direct_on()), "idle quarter statistics link"); link.on.set(on); Ok(true) }
            None => Ok(false),
        }
    }
    pub fn output(&self) -> &Buffer<bf16> {
        self.ob.output()
    }
    /// Token-major WO-A rows [rows, 2048] (this rank's two head groups).
    pub(crate) fn woa_output(&self) -> Result<&Buffer<bf16>> {
        match &self.woa { WoA::Medium(w) => Ok(w.output()), WoA::Small(_) => anyhow::bail!("WO-A exchange requires a medium profile") }
    }
    /// Quantize the token-major WO-A rows with WO-B's own MXFP8 quantizer (all rows).
    pub(crate) fn quantize_wob_input(&self, stream: &Stream) -> Result<()> {
        match &self.woa { WoA::Medium(w) => self.ob.quantize_only(w.output(), stream),
            WoA::Small(_) => anyhow::bail!("WO-B quantization requires a medium profile") }
    }
    /// WO-B's MXFP8 operands for all rows: (values [rows, 2048], MMA scales).
    pub(crate) fn wob_quantized(&self) -> (&Buffer<u8>, &Buffer<u8>) { self.ob.quantized_input() }
    pub(crate) fn wob_matrix(&self) -> std::rc::Rc<crate::blockscaled::Matrix> { self.ob.immutable_matrix() }
    pub(crate) fn set_wob_deferrable(&mut self) -> Result<()> {
        ensure!(matches!(self.woa, WoA::Medium(_)) && crate::config::wide_rows(self.rows), "WO-B exchange profile");
        self.wob_deferrable = true; Ok(())
    }
    pub(crate) fn projection_quantized_input(&self) -> (&Buffer<u8>,&Buffer<u8>) { self.qa.quantized_input() }
    pub(crate) fn narrow_projection(&self) -> bool { self.qa.narrow_projection() }
    /// config::qkv_narrow_prequant, before any capture: load the narrow lowering of this layer's six-row split
    /// QKV-A product beside the default one and make it the active one if `active`. Whether it applies here.
    pub(crate) fn install_narrow_qkv(&mut self, kernels: &Path, active: bool) -> Result<bool> {
        if self.rows != 6 || !matches!(self.mode, InputMode::Split(_)) || self.gathered_qkv.is_none()
            || self.kv.is_some() || self.qa.narrow_projection() { return Ok(false); }
        ensure!(self.qa_alternate.is_none() && self.query_overlap.as_ref().is_none_or(|q| !q.pending.get()),
            "narrow QKV installed once on an idle layer");
        let mut alternate = self.qa.narrow_qkv_alternate(kernels)?;
        if active { std::mem::swap(&mut self.qa, &mut alternate); }
        self.qa_alternate = Some(alternate);
        Ok(true)
    }
    /// Plan screen: the narrow lowering active or not (the caller synchronized and dropped the graphs).
    pub(crate) fn set_narrow_qkv_for_probe(&mut self, on: bool) -> Result<bool> {
        ensure!(self.query_overlap.as_ref().is_none_or(|q| !q.pending.get()), "narrow QKV switch on an idle layer");
        let Some(alternate) = self.qa_alternate.as_mut() else { return Ok(false) };
        if self.qa.narrow_projection() != on { std::mem::swap(&mut self.qa, alternate); }
        ensure!(self.qa.narrow_projection() == on, "narrow QKV switch");
        Ok(true)
    }
    pub(crate) fn legacy_qkv_for_probe(&self,kernels:&Path)->Result<Linear> {self.qa.legacy_qkv_for_probe(kernels)}
    pub(crate) fn swap_qkv_for_probe(&mut self,other:&mut Linear)->Result<()> {self.qa.swap_qkv_for_probe(other)}
    pub(crate) fn producer_qkv_for_probe(&self,kernels:&Path)->Result<Linear> {self.qa.producer_qkv_for_probe(kernels)}
    pub(crate) fn swap_qkv_fence_for_probe(&mut self,other:&mut Linear)->Result<()> {self.qa.swap_qkv_fence_for_probe(other)}
    pub(crate) fn wo_b_pdl_for_probe(&self,kernels:&Path)->Result<Linear>{self.ob.wo_b_pdl_for_probe(kernels)}
    pub(crate) fn set_wo_b_pdl_for_probe(&mut self,mode:u8,other:&mut Linear)->Result<()> {
        ensure!(mode<=2,"WO-B mode");
        if (self.ob.wo_b_mode()==0)!=(mode==0){self.ob.swap_wo_b_backend_for_probe(other)?;}
        if mode!=0{self.ob.set_wo_b_dependency_for_probe(mode==2)?;}
        Ok(())
    }
    pub(crate) fn fused_projection(&self) -> bool {matches!(self.mode,InputMode::Fused)}
    pub(crate) fn input_matrix(&self) -> Rc<Matrix> {
        self.qa.immutable_matrix()
    }
    /// Build this layer's hint queue for every plan (see HintQueue). `has_w1`: the previous layer runs W1
    /// (not before an Engram layer or the first layer); without it the queue starts after QKV-A.
    /// `lead`: immutable regions (and the matrices that own them, if not this layer's weights) that an indexed
    /// layer reads cold before it needs wq_b; config::l2_queue_index_lead puts them first in a second W2.
    pub(crate) fn build_hint_queues(&mut self, module: &Module, stream: &Stream, plans: &[crate::config::QueuePlan],
        selected: Rc<std::cell::Cell<Option<usize>>>, has_w1: bool, lead: &[(u64, usize)],
        lead_owners: &[Rc<Matrix>], engram: Option<(Rc<Matrix>, Rc<std::cell::Cell<bool>>)>) -> Result<Vec<Rc<HintQueue>>> {
        ensure!(engram.is_none() || !has_w1, "an Engram layer takes the Engram hint as its first window");
        let (weights, woa_regions) = match &self.woa {
            WoA::Small(woa) => woa.hint_regions(),
            WoA::Medium(woa) => woa.hint_regions()
                .ok_or_else(|| anyhow::anyhow!("the 16-row L2 hint queue needs the compact WO-A"))?,
        };
        ensure!(self.hint_queues.is_empty() && (self.rows <= 8 || self.rows == 16)
            && matches!(self.mode, InputMode::Split(_)) && self.gathered_qkv.is_some(), "L2 hint queue layer profile");
        let matrices = vec![self.qa.immutable_matrix(), self.qb.immutable_matrix(), self.ob.immutable_matrix()];
        let (qa, qb) = (matrices[0].immutable_regions()?, matrices[1].immutable_regions()?);
        let regions: Vec<(u64, usize)> = qa.into_iter().chain(qb).chain(woa_regions)
            .chain(matrices[2].immutable_regions()?).collect();
        let bytes = |r: &[(u64, usize)]| r.iter().map(|x| x.1).sum::<usize>();
        let (after_qa, after_qb) = (bytes(&qa), bytes(&qa) + bytes(&qb));
        let mut queues = Vec::with_capacity(plans.len());
        for plan in plans {
            let window = |start: usize, w: crate::config::QueueWindow| -> Result<Option<crate::l2_prefetch::Window>> {
                if w.bytes == 0 { return Ok(None); }
                crate::l2_prefetch::Window::new_queue(&self.cuda, module, stream, matrices.clone(), weights.clone(),
                    &regions, start / 16 * 16, w.bytes, w.pace)?.map(|x| x.with_warps(w.warps)?.with_wave(w.wave)).transpose()
            };
            // A window never hints what was already read: W2 starts after QKV-A, W3 after wq_b.
            let first = if has_w1 { plan[0].bytes } else { 0 };
            let second = first.max(after_qa);
            let third = (second + plan[1].bytes).max(after_qb);
            // The lead regions first (at most 16 MiB in all), then as much of W2's wq_b range as still fits.
            let w2_lead = if lead.is_empty() || plan[1].bytes == 0 { None } else {
                const CAP: usize = 16 * 1024 * 1024;
                let mut table: Vec<(u64, usize)> = Vec::new();
                let mut left = CAP;
                for &(address, size) in lead {
                    let take = size.min(left) / 16 * 16;
                    if take > 0 { table.push((address, take)); left -= take; }
                }
                let tail = range_w2(&regions, second, plan[1].bytes.min(left) / 16 * 16)?;
                table.extend(tail);
                let total: usize = table.iter().map(|r| r.1).sum();
                let mut owners = matrices.clone();
                owners.extend(lead_owners.iter().cloned());
                crate::l2_prefetch::Window::new_queue(&self.cuda, module, stream, owners, weights.clone(),
                    &table, 0, total, plan[1].pace)?.map(|x| x.with_warps(plan[1].warps)?.with_wave(plan[1].wave)).transpose()?
            };
            // config::l2_queue_engram_lead: 12 MiB of the Engram projection, column slabs of 2 KiB over every row
            // (what a paced hint delivers beside the previous layer's FFN sum and boundary, where it is joined).
            let engram_w1 = match &engram {
                Some((matrix, _)) if plan[0].bytes > 0 => Some(crate::l2_prefetch::Window::new_tiled(&self.cuda, module,
                    stream, matrix.clone(), 2048, 12 * 1024 * 1024, plan[0].pace.unwrap_or(200))?),
                _ => None,
            };
            // Plan prefix `l` on W1 (config::QueueWindow::lead): an indexed layer's lead regions follow QKV-A's
            // bytes in W1 (at most 16 MiB in all), beside the previous layer's FFN sum; W2 is then wq_b alone.
            let lead_in_w1 = plan[0].lead && has_w1 && !lead.is_empty() && plan[0].bytes > 0;
            let w1_lead = if !lead_in_w1 { None } else {
                const CAP: usize = 16 * 1024 * 1024;
                let mut table = range_w2(&regions, 0, plan[0].bytes.min(CAP) / 16 * 16)?;
                let mut left = CAP - table.iter().map(|r| r.1).sum::<usize>();
                for &(address, size) in lead {
                    let take = size.min(left) / 16 * 16;
                    if take > 0 { table.push((address, take)); left -= take; }
                }
                let total: usize = table.iter().map(|r| r.1).sum();
                let mut owners = matrices.clone();
                owners.extend(lead_owners.iter().cloned());
                crate::l2_prefetch::Window::new_queue(&self.cuda, module, stream, owners, weights.clone(),
                    &table, 0, total, plan[0].pace)?.map(|x| x.with_warps(plan[0].warps)?.with_wave(plan[0].wave)).transpose()?
            };
            let w2_lead = if lead_in_w1 { None } else { w2_lead };
            ensure!(!plan[0].tail || crate::config::experts_tail().is_some(),
                "plan prefix t needs CANADA_SCRATCH_EXPERTS_TAIL");
            let w1_tail = plan[0].tail && has_w1 && !lead_in_w1 && engram.is_none();
            let w1_routed_end = plan[0].routed_end && has_w1 && !lead_in_w1 && engram.is_none();
            // A tail window is begun from the shared expert's branch, which runs on the auxiliary stream: its
            // own kernel goes to the hint stream.
            let w1_plain = if has_w1 && !lead_in_w1 {
                match window(0, plan[0])? {
                    Some(w) if w1_tail || w1_routed_end => Some(w.on_stream(self.cuda.hint_stream(stream)?)?),
                    w => w,
                }
            } else { None };
            queues.push(Rc::new(HintQueue {
                w1: if lead_in_w1 { w1_lead } else if has_w1 { w1_plain } else { engram_w1 },
                w1_gate: engram.as_ref().map(|e| e.1.clone()),
                w2: window(second, plan[1])?,
                w2_lead,
                w3: window(third, plan[2])?,
                kv: if plan[1].kv { Some(crate::l2_prefetch::Window::new_kv(&self.cuda, module, stream)?) } else { None },
                w1_tail, w1_routed_end,
            }));
        }
        self.hint_queues = queues.clone();
        self.hint_queue_selected = selected;
        Ok(queues)
    }
    /// The drafter's hint queue (config::draft_hints). Its input mode has no QKV gather, so two windows: W1, the
    /// first `w1` bytes of the QKV-A and KV projections, which the previous stage begins beside its FFN sum
    /// (`has_w1`: not the first stage), and W3, `w3` bytes of wo_a and wo_b from the attention's start. Both are
    /// joined after wo_b like the target layers' queue.
    pub(crate) fn build_draft_hint_queue(&mut self, module: &Module, stream: &Stream, w1: usize, w3: usize, pace: u32,
        has_w1: bool) -> Result<Rc<HintQueue>> {
        ensure!(self.hint_queues.is_empty() && self.rows == 5 && matches!(self.mode, InputMode::Shared)
            && self.kv.is_some() && self.gathered_qkv.is_none(), "draft hint queue layer profile");
        let (weights, woa_regions) = match &self.woa {
            WoA::Small(woa) => woa.hint_regions(),
            WoA::Medium(_) => anyhow::bail!("the draft hint queue needs the small WO-A"),
        };
        let matrices = vec![self.qa.immutable_matrix(), self.kv.as_ref().unwrap().immutable_matrix(),
            self.qb.immutable_matrix(), self.ob.immutable_matrix()];
        let (qa, kv, qb) = (matrices[0].immutable_regions()?, matrices[1].immutable_regions()?, matrices[2].immutable_regions()?);
        let bytes = |r: &[(u64, usize)]| r.iter().map(|x| x.1).sum::<usize>();
        let after_qb = bytes(&qa) + bytes(&kv) + bytes(&qb);
        let regions: Vec<(u64, usize)> = qa.into_iter().chain(kv).chain(qb).chain(woa_regions)
            .chain(matrices[3].immutable_regions()?).collect();
        let total = bytes(&regions);
        let window = |start: usize, size: usize| -> Result<Option<crate::l2_prefetch::Window>> {
            if size == 0 { return Ok(None); }
            crate::l2_prefetch::Window::new_queue(&self.cuda, module, stream, matrices.clone(), weights.clone(),
                &regions, start / 16 * 16, size, Some(pace))
        };
        let queue = Rc::new(HintQueue { w1: if has_w1 { window(0, w1)? } else { None }, w1_gate: None, w2: None,
            w2_lead: None, w3: window(after_qb, w3)?, kv: None, w1_tail: false, w1_routed_end: false });
        if !has_w1 {
            eprintln!("DSpark draft hint queue: stage dense weights {total} bytes (QKV-A, KV, wq_b {after_qb}); W1 {w1} W3 {w3} bytes at {pace} GB/s");
        }
        self.hint_queues = vec![queue.clone()];
        self.hint_queue_selected = Rc::new(std::cell::Cell::new(Some(0)));
        Ok(queue)
    }
    fn selected_hint_queue(&self) -> Option<&Rc<HintQueue>> {
        self.hint_queue_selected.get().and_then(|i| self.hint_queues.get(i))
    }
    fn join_hint_queue(&self, stream: &Stream) -> Result<()> {
        if let Some(q) = self.selected_hint_queue() {
            for w in [&q.w1, &q.w2, &q.w2_lead, &q.w3, &q.kv].into_iter().flatten() {
                if w.is_open() { w.join(stream)?; }
            }
        }
        Ok(())
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn run(
        &mut self,
        input: &Buffer<bf16>,
        positions: &Positions,
        locations: &RowLocations,
        cache: &mut PagedKv,
        candidates: &Candidates,
        sink: &Buffer<f32>,
        stream: &Stream,
    ) -> Result<()> {
        self.run_distributed(
            input, positions, locations, cache, candidates, sink, None, stream,
        )
    }
    pub fn run_distributed(
        &mut self,
        input: &Buffer<bf16>,
        positions: &Positions,
        locations: &RowLocations,
        cache: &mut PagedKv,
        candidates: &Candidates,
        sink: &Buffer<f32>,
        comm: Option<&Roce>,
        stream: &Stream,
    ) -> Result<()> {
        self.run_distributed_quantized(input,None,positions,locations,cache,candidates,sink,comm,stream)
    }
    pub(crate) fn run_distributed_quantized(&mut self, input:&Buffer<bf16>,
        quantized:Option<(&Buffer<u8>,&Buffer<u8>)>, positions:&Positions, locations:&RowLocations,
        cache:&mut PagedKv,candidates:&Candidates,sink:&Buffer<f32>,comm:Option<&Roce>,stream:&Stream) -> Result<()> {
        ensure!(
            !self.frequencies.compressed(),
            "compressed attention requires an explicit indexed cache source"
        );
        self.project_distributed_quantized(input, quantized, positions, locations, cache, comm, stream)?;
        self.finish(positions, cache, candidates, sink, None, stream)
    }
    /// Batched drafter passes (config::batched_draft_interleave): one phase of this drafter's attention sublayer
    /// (1 input projection, 2 KV projection, 4 query norm, wq_b and stores, 8 attention and wo_a, 16 wo_b), the
    /// same kernels on the same inputs as `run_distributed` runs in that order. Several conversations take each
    /// phase back to back, so a phase's weights (4-11 MB) are read from memory once and from L2 after that.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_draft_phase(&mut self, phase: u8, input: &Buffer<bf16>, positions: &Positions,
        locations: &RowLocations, cache: &mut PagedKv, candidates: &Candidates, sink: &Buffer<f32>,
        stream: &Stream) -> Result<()> {
        ensure!(self.rows == 5 && matches!(self.mode, InputMode::Shared) && self.kv.is_some()
            && self.gathered_qkv.is_none() && matches!(self.woa, WoA::Small(_)) && !self.frequencies.compressed(),
            "draft attention phases need the drafter's profile");
        match phase {
            1 | 2 | 4 => self.project_input_phases(input, positions, locations, cache, None, None, None, false, phase, stream),
            8 | 16 => self.finish_phases(positions, cache, candidates, sink, None, phase, stream),
            _ => anyhow::bail!("unknown draft attention phase"),
        }
    }
    /// Main projections/store precede the source/index branch, which consumes
    /// the exact normalized Q LoRA values before sparse attention runs.
    pub fn project_distributed(
        &mut self,
        input: &Buffer<bf16>,
        positions: &Positions,
        locations: &RowLocations,
        cache: &mut PagedKv,
        comm: Option<&Roce>,
        stream: &Stream,
    ) -> Result<()> {
        self.project_distributed_quantized(input,None,positions,locations,cache,comm,stream)
    }
    pub(crate) fn project_distributed_quantized(&mut self,input:&Buffer<bf16>,
        quantized:Option<(&Buffer<u8>,&Buffer<u8>)>,positions:&Positions,locations:&RowLocations,
        cache:&mut PagedKv,comm:Option<&Roce>,stream:&Stream)->Result<()> {
        self.project_input(input, positions, locations, cache, comm, None, quantized, false, stream)
    }
    pub(crate) fn prepare_row_shard(&self, projection_root: Option<&Path>) -> Result<crate::fp8_linear::RowShard> {
        ensure!(matches!(self.mode, InputMode::Fused) && self.kv.is_none()
            && self.gathered_qkv.is_none(), "row gather requires fused QKV");
        self.qa.prepare_row_shard(projection_root)
    }
    pub(crate) fn project_row_shard(&mut self, input: &Buffer<bf16>,
        positions: &Positions, locations: &RowLocations, cache: &mut PagedKv,
        comm: &Roce, shard: &mut crate::fp8_linear::RowShard, project_local: bool, stream: &Stream) -> Result<()> {
        ensure!(matches!(self.mode, InputMode::Fused) && self.kv.is_none()
            && self.gathered_qkv.is_none(), "row gather requires fused QKV");
        self.project_input(input, positions, locations, cache, Some(comm), Some((shard,project_local)), None, false, stream)
    }
    /// Only the admitted paired executor may supply this owner's already
    /// computed QKV projection. All subsequent query/key/attention work is the
    /// same private path used by ordinary execution.
    pub(crate) fn project_precomputed(&mut self,input:&Buffer<bf16>,positions:&Positions,
        locations:&RowLocations,cache:&mut PagedKv,comm:&Roce,stream:&Stream)->Result<()> {
        self.project_input(input,positions,locations,cache,Some(comm),None,None,true,stream)
    }
    pub(crate) fn paired_input_projection(&self)->Result<&Linear> {
        ensure!(self.rows==6&&matches!(self.mode,InputMode::Split(_))
            &&self.kv.is_none()&&self.gathered_qkv.is_some()&&!self.qa.narrow_projection()
            &&self.query_overlap.as_ref().is_none_or(|q|!q.pending.get()),
            "paired QKV requires an idle split M6 projection");
        Ok(&self.qa)
    }
    #[allow(clippy::too_many_arguments)]
    fn project_input(&mut self, input: &Buffer<bf16>, positions: &Positions,
        locations: &RowLocations, cache: &mut PagedKv, comm: Option<&Roce>,
        shard: Option<(&mut crate::fp8_linear::RowShard,bool)>,
        quantized:Option<(&Buffer<u8>,&Buffer<u8>)>, precomputed:bool, stream: &Stream) -> Result<()> {
        self.project_input_phases(input, positions, locations, cache, comm, shard, quantized, precomputed, 7, stream)
    }
    /// `phases`: 1 = the input projection (and its gather), 2 = the KV projection, 4 = the query norm, wq_b and the
    /// query and key stores; 7 = the whole projection in that order. Single phases serve batched drafter passes
    /// (`run_draft_phase`), where several conversations take each phase back to back.
    #[allow(clippy::too_many_arguments)]
    fn project_input_phases(&mut self, input: &Buffer<bf16>, positions: &Positions,
        locations: &RowLocations, cache: &mut PagedKv, comm: Option<&Roce>,
        shard: Option<(&mut crate::fp8_linear::RowShard,bool)>,
        quantized:Option<(&Buffer<u8>,&Buffer<u8>)>, precomputed:bool, phases: u8, stream: &Stream) -> Result<()> {
        if precomputed {
            self.paired_input_projection()?;
            ensure!(shard.is_none()&&quantized.is_none()&&comm.is_some(),"precomputed QKV input path");
        }
        ensure!(quantized.is_none() || (shard.is_none() && matches!(self.rows, 6 | 16)
            && matches!(self.mode,InputMode::Split(_))
            // config::qkv_narrow_prequant: an installed alternate consumes the same quantized input.
            && (!self.qa.narrow_projection() || self.qa_alternate.is_some())),
            "prequantized normalized attention requires the checked legacy QKV quantizer");
        if let Some(q) = &self.query_overlap {
            ensure!(!q.pending.get() && q.parent == stream.raw() as usize,
                "attention query overlap is pending or has a foreign parent");
        }
        ensure!(
            self.gathered_qkv.is_none() || comm.is_some(),
            "split QKV requires TP4 communicator"
        );
        if phases & 1 != 0 {
        if precomputed {
            // The paired normalizer/projector already wrote this private output.
        } else if let Some((shard,project_local)) = shard {
            self.qa.run_row_shard(input, shard, project_local,
                comm.ok_or_else(|| anyhow::anyhow!("row gather communicator absent"))?, stream)?;
        } else if crate::knockout::has(crate::knockout::Site::QkvA) {
            // Plan-screen knockout: the projection's output of the previous run of this step stays.
        } else if let Some((values,scales)) = quantized {
            self.qa.run_quantized(values,scales,stream)?;
        } else {
            self.qa.run(input, stream)?;
        }
        crate::mhc::stats_point(crate::mhc::StatsPoint::AfterInputProjection, stream)?;
        if let Some(output) = self.gathered_qkv.as_mut() {
            if let Some(q) = self.hint_queue_selected.get().and_then(|i| self.hint_queues.get(i)) {
                if let Some(k) = &q.kv {
                    k.begin_kv(stream, locations.locations_ptr(), cache.data().ptr(), cache.page_tokens(),
                        cache.page_bytes(), cache.pages(), self.rows, 128)?;
                }
                let lead = q.w2_lead.as_ref().filter(|_| self.index_lead.get());
                if let Some(w) = lead.or(q.w2.as_ref()) {
                    let stop = crate::l2_prefetch::early_stop_active().then(|| comm.map(Roce::epoch_ptr)).flatten();
                    w.begin_stopping(stream, stop)?;
                }
            } else if let Some(hint) = &self.qb_prefetch {
                hint.begin(stream)?;
            }
            // The rows ride with the QKV-A rows only when the direct kernel wrote those behind the pseudo-rows (not
            // after a pooled / precomputed projection, which writes the projection's own output).
            let rider = self.quarter.as_ref().filter(|_| self.qa.direct_on() && !precomputed);
            if crate::knockout::has(crate::knockout::Site::Gather) {
                // Plan-screen knockout: the gathered rows of the previous run of this step stay.
                ensure!(self.quarter.as_ref().is_none_or(|link| link.armed.get().is_none()), "gather knockout with quarter statistics");
            } else if let Some(link) = self.quarter.as_ref().filter(|link| rider.is_none() && link.armed.get().is_some()) {
                // Partials were issued but this step's QKV-A rows are not in the row block: the ordinary gather,
                // then the pseudo-rows in a gather of their own (correct in every path, one collective more).
                use crate::mhc::{QUARTER_COLUMNS, QUARTER_HEAD};
                let comm = comm.unwrap();
                let pseudo = link.armed.take().unwrap();
                unsafe { comm.gather_bf16(self.qa.output(), output, 448, stream) }?;
                link.partials_ready.wait(stream)?;
                let first = QUARTER_HEAD - pseudo;
                let input = unsafe { link.xin.view(first * QUARTER_COLUMNS, pseudo * QUARTER_COLUMNS) }?;
                let mut gathered = unsafe { link.xout.view(first * QUARTER_COLUMNS * 4, pseudo * QUARTER_COLUMNS * 4) }?;
                unsafe { comm.gather_bf16_exact(&input, &mut gathered, QUARTER_COLUMNS, stream) }?;
                link.gathered.record(stream)?;
            } else if let Some(link) = rider {
                // The direct kernel wrote this step's QKV-A rows behind the pseudo-rows; the boundary's partials
                // (when it issued any) end where those rows begin: one gather takes both.
                use crate::mhc::{QUARTER_COLUMNS, QUARTER_HEAD};
                let comm = comm.unwrap();
                let pseudo = link.armed.take();
                let (first, rows) = (QUARTER_HEAD - pseudo.unwrap_or(0), pseudo.unwrap_or(0) + comm.gather_live_rows(self.rows));
                if pseudo.is_some() { link.partials_ready.wait(stream)?; }
                let input = unsafe { link.xin.view(first * QUARTER_COLUMNS, rows * QUARTER_COLUMNS) }?;
                let mut gathered = unsafe { link.xout.view(first * QUARTER_COLUMNS * 4, rows * QUARTER_COLUMNS * 4) }?;
                unsafe { comm.gather_bf16_exact(&input, &mut gathered, QUARTER_COLUMNS, stream) }?;
                if pseudo.is_some() { link.gathered.record(stream)?; }
            } else {
                unsafe {
                    comm.unwrap()
                        .gather_bf16(self.qa.output(), output, 448, stream)
                }?;
            }
            crate::mhc::stats_point(crate::mhc::StatsPoint::AfterGather, stream)?;
        }
        }
        if phases & 2 != 0 {
        if let Some(kv) = &self.kv {
            if matches!(self.mode, InputMode::Shared) {
                let (a, s) = self.qa.quantized_input();
                kv.run_quantized(a, s, stream)?;
            } else {
                kv.run(input, stream)?;
            }
        }
        }
        if phases & 4 == 0 { return Ok(()); }
        // The qualified FLASHINFER_CUTLASS backend normalizes via CuTe
        // RMSNorm, rounds to BF16, then quantizes for wq_b. Its alternative
        // fused Triton norm has different BF16 ties on real inputs.
        let input=self.gathered_qkv.as_ref().unwrap_or(self.qa.output());
        if let Some((library,index))=&self.query_norm_quant {
            let (q,s)=self.qb.query_quantization_buffers()?;
            // Writes BF16 for the indexer as well as the exact FlashInfer MXFP8
            // representation. Inactive scale lanes retain constructor zeroes.
            unsafe {library.launch_words(*index,&[input.ptr(),self.qnorm_weight.ptr(),
                self.normalized.ptr(),q.ptr(),s.ptr(),self.rows as u64,1e-20f32.to_bits()as u64],stream)?;}
        } else {
            self.native.rms(input,&self.qnorm_weight,&mut self.normalized,
                self.rows,self.q_stride,stream)?;
        }
        // Adapted from SGLang's low-ratio attention preparation: Q and
        // KV/compressor/index work are independent after Q normalization.
        // Keep causal writers on their admitted parent; move only transient
        // Q-B/RoPE to the side and join at the first attention consumer.
        let query_stream = if let Some(q) = &self.query_overlap {
            q.ready.record(stream)?;
            q.ready.wait(&q.side)?;
            q.pending.set(true);
            q.side.as_ref()
        } else { stream };
        if crate::knockout::has(crate::knockout::Site::WqB) {
            // Plan-screen knockout: the query projection of the previous run of this step stays.
        } else if self.query_norm_quant.is_some() {
            let (q,s)=self.qb.query_quantization_buffers()?;
            self.qb.run_quantized(q,s,query_stream)?;
        } else {self.qb.run(&self.normalized, query_stream)?;}
        match &mut self.qstore {
            Query::Small(q) => q.run(
                self.qb.output(),
                self.frequencies.values(),
                positions,
                query_stream,
            )?,
            Query::Medium(q) if crate::config::wide_rows(self.rows) && self.rows >= 512 && q_inplace_enabled() => {
                let mut rotated = unsafe { self.qb.output().view(0, q.len()) }?;
                self.native.rope(&mut rotated, self.rows, self.frequencies.values(), positions, false, query_stream)?;
            }
            Query::Medium(q) => {
                if let Some(op)=&self.wide_qrope {
                    unsafe{op.launch(self.rows as u32,&[self.qb.output().ptr(),q.ptr(),
                        self.frequencies.values().ptr(),positions.values().ptr()],query_stream)?;}
                }else{
                unsafe {
                    self.cuda
                        .copy_device(q.ptr(), self.qb.output().ptr(), q.bytes(), query_stream)
                }?;
                self.native.rope(
                    q,
                    self.rows,
                    self.frequencies.values(),
                    positions,
                    false,
                    query_stream,
                )?;
                }
            }
        }
        if let Some(q) = &self.query_overlap { q.done.record(query_stream)?; }
        let (kv, offset, stride) = if let Some(kv) = &self.kv {
            (kv.output(), 0, 512)
        } else {
            (
                self.gathered_qkv.as_ref().unwrap_or(self.qa.output()),
                1280,
                1792,
            )
        };
        self.kstore.run_at(
            kv,
            offset,
            stride,
            &self.kvnorm_weight,
            self.frequencies.values(),
            locations,
            cache,
            stream,
        )?;
        Ok(())
    }
    pub fn normalized_lora(&self) -> &Buffer<bf16> {
        &self.normalized
    }
    pub(crate) fn set_append_bf16_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(([6,16].contains(&self.rows)||(self.rows==32
            &&crate::config::append32_workspace_enabled()))
            && self.cuda.owns_stream(stream) && self.append_decode.is_some()
            && self.query_overlap.as_ref().is_none_or(|q|!q.pending.get())
            && std::env::var("CANADA_SCRATCH_APPEND_BF16_ONLY").as_deref()==Ok("1"),
            "append attention switch requires an idle admitted narrow layer");
        if self.append_decode_active!=enabled {
            std::mem::swap(&mut self.decode,self.append_decode.as_mut().unwrap());
            self.append_decode_active=enabled;
        }
        Ok(())
    }
    pub(crate) fn set_split_attention_for_probe(&mut self,path:u8,stream:&Stream)->Result<()> {
        ensure!(path<=2 && self.cuda.owns_stream(stream)
            && self.query_overlap.as_ref().is_none_or(|q|!q.pending.get())
            && ((self.rows==6&&std::env::var("CANADA_SCRATCH_SPLIT_ATTENTION_SCREEN").as_deref()==Ok("1"))
                ||(self.rows==16&&std::env::var("CANADA_SCRATCH_SPARSE_PRECISION_WIDE_SCREEN").as_deref()==Ok("1"))),
            "split attention switch admission");
        // SWA-only layers remain identical in all three arms.
        if self.split_decode.is_none(){return Ok(())}
        if self.split_decode_active {std::mem::swap(&mut self.decode,self.split_decode.as_mut().unwrap());self.split_decode_active=false;}
        self.set_append_bf16_for_probe(false,stream)?;
        if path==1 {self.set_append_bf16_for_probe(true,stream)?;}
        if path==2 {std::mem::swap(&mut self.decode,self.split_decode.as_mut().unwrap());self.split_decode_active=true;}
        Ok(())
    }
    pub(crate) fn set_append_split_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!([6,16].contains(&self.rows)&&!self.append_decode_active&&!self.split_decode_active
            &&self.cuda.owns_stream(stream)&&self.query_overlap.as_ref().is_none_or(|q|!q.pending.get())
            &&std::env::var("CANADA_SCRATCH_APPEND_SPLIT_ATTENTION_SCREEN").as_deref()==Ok("1"),
            "append split requires idle ordinary decode mode");
        if let Some(other)=&mut self.append_split {
            if self.append_split_active!=enabled {
                std::mem::swap(self.append_decode.as_mut().unwrap(),other);self.append_split_active=enabled;
            }
        }
        Ok(())
    }
    pub(crate) fn query_overlap_enabled(&self) -> bool { self.query_overlap.is_some() }
    /// config::index_fork: make `side` wait for the fence recorded where the normalized query latent became final
    /// (where Q-B forked in this layer's projection), so other work that reads it can begin there too.
    pub(crate) fn wait_query_ready(&self, side: &Stream) -> Result<()> {
        let q = self.query_overlap.as_ref().ok_or_else(|| anyhow::anyhow!("no query fork to follow"))?;
        ensure!(q.pending.get() && self.cuda.owns_stream(side), "the query fork has not begun in this layer");
        q.ready.wait(side)
    }
    pub(crate) fn query_norm_quant_enabled(&self) -> bool {self.query_norm_quant.is_some()}
    pub(crate) fn schedule_flags(&self)->(bool,bool) {
        (self.qb_prefetch.is_some(),self.qb.pdl_active())
    }
    pub fn quantized_lora(&self) -> (&Buffer<u8>, &Buffer<u8>) {
        self.qb.quantized_input()
    }
    pub(crate) fn set_woa_prefetch_for_probe(&mut self,enabled:bool)->Result<()> {
        ensure!(self.rows==6 && self.woa_prefetch.is_some()
            &&std::env::var("CANADA_SCRATCH_WOA_PREFETCH_SCREEN").as_deref()==Ok("1"),
            "WO-A hint screen owner");
        self.woa_prefetch_active=enabled;Ok(())
    }
    pub(crate) fn set_wide_hint_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(matches!(self.rows,16|32),"wide Q-B hint profile");
        self.qb_prefetch.as_mut().context("wide Q-B hint was not prepared")?
            .set_enabled_for_wide_probe(enabled,stream)
    }
    pub fn finish(
        &mut self,
        positions: &Positions,
        cache: &PagedKv,
        candidates: &Candidates,
        sink: &Buffer<f32>,
        extra: Option<(&PagedKv, &Candidates)>,
        stream: &Stream,
    ) -> Result<()> {
        self.finish_phases(positions, cache, candidates, sink, extra, 24, stream)
    }
    /// `phases`: 8 = the attention and wo_a, 16 = wo_b and the hint joins; 24 = both in that order.
    #[allow(clippy::too_many_arguments)]
    fn finish_phases(
        &mut self,
        positions: &Positions,
        cache: &PagedKv,
        candidates: &Candidates,
        sink: &Buffer<f32>,
        extra: Option<(&PagedKv, &Candidates)>,
        phases: u8,
        stream: &Stream,
    ) -> Result<()> {
        if phases & 8 != 0 {
        if let Some(q) = &self.query_overlap {
            ensure!(q.pending.get() && q.parent == stream.raw() as usize,
                "attention consumer has no matching query fork");
            q.done.wait(stream)?;
            q.pending.set(false);
        }
        crate::layer_clock::fine(stream, crate::layer_clock::FINE_QUERY_JOINED)?;
        if self.woa_prefetch_active {self.woa_prefetch.as_ref().unwrap().begin(stream)?;}
        if let Some(w) = self.selected_hint_queue().and_then(|q| q.w3.as_ref()) {
            if !w.is_open() { w.begin(stream)?; }
        }
        let query = if matches!(self.qstore, Query::Medium(_)) && crate::config::wide_rows(self.rows)
            && self.rows >= 512 && q_inplace_enabled() { self.qb.output() } else { self.qstore.output() };
        if !crate::knockout::has(crate::knockout::Site::Attention) {
        self.decode.run(
            query,
            cache,
            candidates,
            Some(sink),
            extra,
            0.044_194_173_f32,
            stream,
        )?;
        self.native.rope(
            self.decode.output_mut(),
            self.rows,
            self.frequencies.values(),
            positions,
            true,
            stream,
        )?;
        }
        crate::layer_clock::fine(stream, crate::layer_clock::FINE_ATTENDED)?;
        }
        match &mut self.woa {
            WoA::Small(w) => {
                if phases & 8 != 0 && !crate::knockout::has(crate::knockout::Site::WoA) { w.run(self.decode.output(), stream)?; }
                if phases & 8 != 0 { crate::layer_clock::fine(stream, crate::layer_clock::FINE_WOA)?; }
                if phases & 16 != 0 && !crate::knockout::has(crate::knockout::Site::WoB) {
                    self.ob.run_quantized(w.values(), w.scales(), stream)?;
                }
                if phases & 16 != 0 { crate::layer_clock::fine(stream, crate::layer_clock::FINE_WOB)?; }
                Ok(())
            }
            WoA::Medium(w) => {
                ensure!(phases == 24, "attention phases need the small WO-A");
                w.run(self.decode.output(), stream)?;
                if self.wob_deferrable && (wob_exchange_active() || wob8_active()) { Ok(()) } else { self.ob.run(w.output(), stream) }
            }
        }?;
        if phases & 16 == 0 { return Ok(()); }
        // No data dependency: immutable prefetch can finish after its consumer.
        // Join before the next layer so the side stream remains graph-owned.
        if self.woa_prefetch_active {self.woa_prefetch.as_ref().unwrap().join(stream)?;}
        self.join_hint_queue(stream)?;
        // Read-only hints need no consumer dependency. Match the reference's
        // late join at the next communication boundary rather than stalling QB.
        // A deferred layer-20 projection may already have closed its hint.
        if let Some(hint) = self.qb_prefetch.as_ref().filter(|h| h.is_open()) {
            hint.join(stream)?;
        }
        Ok(())
    }
    /// Deferred layer 20: the projection stops before `finish`, so close the
    /// read-only Q-B L2 hint (no data dependency) before the next projection.
    pub(crate) fn join_projection_hint(&self, stream: &Stream) -> Result<()> {
        ensure!(self.query_overlap.is_none(), "deferred projection cannot leave a query fork");
        self.join_hint_queue(stream)?;
        if let Some(hint) = self.qb_prefetch.as_ref().filter(|h| h.is_open()) {
            hint.join(stream)?;
        }
        Ok(())
    }
    /// Diagnostic taps for numerical-boundary qualification only.
    pub(crate) fn local_qkv_replay_taps(&self)->[(&'static str,u64,usize);3] {
        let(values,scales)=self.qa.quantized_input();
        [("qkv-before-gather",self.qa.output().ptr(),self.qa.output().bytes()),
         ("qkv-quant",values.ptr(),values.bytes()),
         ("qkv-scales",scales.ptr(),scales.bytes())]
    }
    /// Diagnostic taps for numerical-boundary qualification only.
    pub fn taps(&self) -> Vec<(&'static str, u64, usize, usize)> {
        let t = |name, b: &Buffer<bf16>| (name, b.ptr(), b.bytes(), 0);
        let qa = (
            "qa",
            self.gathered_qkv.as_ref().unwrap_or(self.qa.output()).ptr(),
            self.rows * 1280 * 2,
            if matches!(self.mode, InputMode::Fused | InputMode::Split(_)) {
                1792 * 2
            } else {
                0
            },
        );
        let kv = if let Some(kv) = &self.kv {
            t("kv", kv.output())
        } else {
            (
                "kv",
                self.gathered_qkv.as_ref().unwrap_or(self.qa.output()).ptr() + 1280 * 2,
                self.rows * 512 * 2,
                1792 * 2,
            )
        };
        let mut taps = vec![
            qa,
            t("normalized", &self.normalized),
            (
                "norm_quant",
                self.qb.quantized_input().0.ptr(),
                self.qb.quantized_input().0.bytes(),
                0,
            ),
            (
                "norm_scales",
                self.qb.quantized_input().1.ptr(),
                self.qb.quantized_input().1.bytes(),
                0,
            ),
            t("qb", self.qb.output()),
            t("q_rope", self.qstore.output()),
            kv,
            t("inverse_attention", self.decode.output()),
            t("output", self.ob.output()),
        ];
        match &self.woa {
            WoA::Small(w) => taps.extend([
                ("woa_partial", w.partial().ptr(), w.partial().bytes(), 0),
                ("woa_quant", w.values().ptr(), w.values().bytes(), 0),
                ("woa_scales", w.scales().ptr(), w.scales().bytes(), 0),
            ]),
            WoA::Medium(w) => taps.extend([
                t("woa", w.output()),
                (
                    "woa_quant",
                    self.ob.quantized_input().0.ptr(),
                    self.ob.quantized_input().0.bytes(),
                    0,
                ),
                (
                    "woa_scales",
                    self.ob.quantized_input().1.ptr(),
                    self.ob.quantized_input().1.bytes(),
                    0,
                ),
            ]),
        }
        taps
    }
}

/// CANADA_SCRATCH_PICO_ATTENTION: unset or 1 = the adapted pico kernel for V4.1 attention above 8 rows; 0 = FlashInfer's.
pub(crate) fn pico_attention_enabled() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_PICO_ATTENTION").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("1") => Ok(true),
        Ok("0") => Ok(false),
        _ => anyhow::bail!("CANADA_SCRATCH_PICO_ATTENTION must be 0 or 1"),
    }
}
