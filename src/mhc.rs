//! Canonical HC=4 residual boundary. Rust owns coefficient/scratch/output state;
//! AOT low-level kernels preserve the reference's BF16 and FMA boundaries.
use crate::{
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, Event, Module, Stream},
    prelude::Native,
    primitives::Primitives,
    triton_ops::Operator,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{cell::{Cell, RefCell}, path::Path, rc::Rc};

thread_local! {
    static ACTIVE_ROWS:Cell<Option<(usize,usize,usize,bool)>>=const{Cell::new(None)};
}
/// Verify-layer stage 1 (config::fused_boundary): the collapsed row a fused boundary sum wrote for the boundary that
/// prepares next. Keys: that boundary's residual (the fused boundary's output stream), the pre it collapsed with, the
/// stream and the capture serial (one graph, or one eager run). The next `prepare` takes it and uses the row only when
/// every key matches; otherwise it collapses the stream itself, as without the flag.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct FusedValue { residual: u64, pre: u64, value: u64, stream: usize, serial: u64 }
thread_local! {
    static FUSED_VALUE: Cell<Option<FusedValue>> = const { Cell::new(None) };
}
/// Where a deferred statistics fork may be released later than its boundary's normalization
/// (CANADA_SCRATCH_MHC_STATS_LATE, default off). The statistics read only the boundary's input residual and are
/// joined before the boundary's post-mix, so any point inside the sublayer gives the same bits; the point decides
/// which kernels the 80-block statistics kernel shares the SMs with. Released at the normalization (today), it
/// starts just ahead of the sublayer's first weight kernel (QKV-A, the router product) and holds it back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum StatsPoint { AfterInputProjection = 1, AfterGather = 2, AfterRouterProduct = 3, AfterRouting = 4, AfterExperts = 5,
    BeforeAttentionSum = 6, ExpertsTail = 7, AfterRoutedExperts = 8 }
struct LateFork { ready: Rc<Event>, parent: usize, point: u8 }
thread_local! {
    static LATE_FORKS: RefCell<Vec<LateFork>> = const { RefCell::new(Vec::new()) };
}
/// A sublayer reached one of its named points on `stream`: release the statistics forks waiting for it.
pub(crate) fn stats_point(point: StatsPoint, stream: &Stream) -> Result<()> {
    crate::layer_clock::fine(stream, 16 + point as u8)?;
    LATE_FORKS.with(|forks| -> Result<()> {
        let mut forks = forks.borrow_mut();
        let mut index = 0;
        while index < forks.len() {
            if forks[index].point == point as u8 && forks[index].parent == stream.raw() as usize {
                forks.swap_remove(index).ready.record(stream)?;
            } else { index += 1; }
        }
        Ok(())
    })
}
/// The routed experts' kernels have just been issued on `stream` (before the shared expert's join and the sums
/// that follow them): release the statistics forks waiting for `StatsPoint::AfterRoutedExperts` and begin the hint
/// windows registered for that point (plan prefix `e`). They start when the routed experts end, a few microseconds
/// (up to ten when the shared expert's FC2 is late) before the points after the FFN's small sums.
pub(crate) fn routed_end_point(stream: &Stream) -> Result<()> {
    stats_point(StatsPoint::AfterRoutedExperts, stream)?;
    for queue in crate::layer_attention::tail_w1_take(stream, crate::layer_attention::TailKind::RoutedEnd) {
        if let Some(window) = queue.w1.as_ref() { window.begin(stream)?; }
    }
    Ok(())
}
/// The shared expert's branch (`source`) passed its activation while the routed experts run on `parent`
/// (CANADA_SCRATCH_EXPERTS_TAIL): launch the gate there and release, on its completion, the statistics forks
/// waiting for `StatsPoint::ExpertsTail` and the hint windows registered for the tail
/// (layer_attention::tail_w1_register). Nothing waits: no gate.
pub(crate) fn tail_point(module: &crate::cuda::Module, shared_bytes: u32, parent: &Stream, source: &Stream) -> Result<()> {
    let forks: Vec<LateFork> = LATE_FORKS.with(|forks| {
        let mut forks = forks.borrow_mut();
        let (hit, rest): (Vec<_>, Vec<_>) = forks.drain(..)
            .partition(|f| f.point == StatsPoint::ExpertsTail as u8 && f.parent == parent.raw() as usize);
        *forks = rest;
        hit
    });
    let windows = crate::layer_attention::tail_w1_take(parent, crate::layer_attention::TailKind::Gate);
    if forks.is_empty() && windows.is_empty() { return Ok(()); }
    let mut args: [*mut std::ffi::c_void; 0] = [];
    unsafe { module.function("ca_tail_gate")?.launch([1, 1, 1], [32, 1, 1], shared_bytes, source, &mut args) }?;
    for fork in forks { fork.ready.record(source)?; }
    for queue in windows {
        if let Some(window) = queue.w1.as_ref() { window.begin_from(parent, source)?; }
    }
    Ok(())
}
/// The release points requested for an owner of `rows` physical rows: (attention boundary, FFN boundary), 0 = at
/// the normalization. CANADA_SCRATCH_MHC_STATS_LATE=<a><f> for the six-row owner, ..._LATE16 for the 16-row one;
/// a in {0, 1, 2, 6}, f in {0, 3, 4, 5, 7, 8} (the `StatsPoint` numbers; 7 needs CANADA_SCRATCH_EXPERTS_TAIL).
pub(crate) fn stats_late_requested(rows: usize) -> Result<(u8, u8)> {
    let name = match rows { 6 => "CANADA_SCRATCH_MHC_STATS_LATE", 16 => "CANADA_SCRATCH_MHC_STATS_LATE16", _ => return Ok((0, 0)) };
    match std::env::var(name) {
        Ok(value) => parse_stats_late(&value),
        Err(std::env::VarError::NotPresent) => Ok((0, 0)), Err(error) => Err(error.into()),
    }
}
pub(crate) fn parse_stats_late(value: &str) -> Result<(u8, u8)> {
    let digits: Vec<u8> = value.bytes().map(|b| b.wrapping_sub(b'0')).collect();
    ensure!(digits.len() == 2 && [0, 1, 2, 6].contains(&digits[0]) && [0, 3, 4, 5, 7, 8].contains(&digits[1]),
        "invalid mHC late statistics points");
    ensure!(digits[1] != 7 || crate::config::experts_tail().is_some(),
        "statistics point 7 needs CANADA_SCRATCH_EXPERTS_TAIL");
    Ok((digits[0], digits[1]))
}
pub(crate) struct ActiveRowsScope(std::marker::PhantomData<Rc<()>>);
impl ActiveRowsScope {
    pub(crate) fn new(enabled:bool,live:usize,capacity:usize,stream:&Stream,normalize:bool)->Result<Self> {
        ACTIVE_ROWS.with(|slot|->Result<()>{
            ensure!(slot.get().is_none(),"nested mHC live-row scope");
            if enabled {
                ensure!(capacity==6&&(1..=6).contains(&live)
                    ||capacity==128&&(17..128).contains(&live)
                    ||capacity==512&&(129..512).contains(&live)
                    ||capacity==1024&&(513..1024).contains(&live)
                    ||capacity==2048&&(1025..2048).contains(&live),"mHC live-row geometry");
                ensure!(!normalize||capacity>=128,"live normalization is prefill-only");
                slot.set(Some((live,capacity,stream.raw()as usize,normalize)));
            }
            Ok(())
        })?;Ok(Self(std::marker::PhantomData))
    }
}
impl Drop for ActiveRowsScope {
    fn drop(&mut self){ACTIVE_ROWS.with(|s|s.set(None));}
}

pub struct Boundary {
    cuda: Cuda,
    weights: Rc<Weights>,
    rows: usize,
    fn_pointer: u64,
    base_pointer: u64,
    scale_pointer: u64,
    norm: Native,
    plain_norm: Option<Native>,
    combined: Option<Buffer<bf16>>,
    norm_weight: Buffer<bf16>,
    stats: Operator,
    stats_two_mma: Option<Operator>,
    use_two_mma: bool,
    reduce: Operator,
    combine: Operator,
    fused_norm: Option<Primitives>,
    use_fused_norm: bool,
    post_op: Operator,
    partial_mix: Buffer<f32>,
    partial_square: Buffer<f32>,
    pre: Buffer<f32>,
    post: Buffer<f32>,
    comb: Buffer<f32>,
    normalized: Buffer<bf16>,
    output: Buffer<bf16>,
    stream: Cell<usize>,
    prepared: Cell<u64>,
    prepared_rows: Cell<usize>,
    prepared_pre: Cell<u64>,
    deferred_stats: Option<DeferredStats>,
    /// 0: the statistics fork is released at the normalization; otherwise a `StatsPoint` of this sublayer.
    late_point: Cell<u8>,
    /// The FFN boundary of its layer (the attention boundary otherwise): which knockout names its statistics.
    ffn_branch: bool,
    stats_pending: Cell<bool>,
    full_stats: Option<FullStats>,
    use_full_stats: bool,
    norm_quant: Option<NormQuant>,
    norm_quant_active: bool,
    quant_ready: bool,
    pdl_pair: bool,
    alternate_pdl_pair: Option<(Operator, Operator)>,
    pdl_stats: bool,
    alternate_pdl_stats: Option<(Operator, Operator)>,
    // CANADA_SCRATCH_PREFILL_MHC_POST_STATS: the core module of a finishing boundary that also
    // forms the next boundary's statistics (finish_with_next_stats), and, on the next boundary,
    // the residual whose statistics partials are already in place.
    post_stats: Option<Module>,
    stats_precomputed: Cell<u64>,
    /// config::mhc_quarter_stats: the link to the attention's QKV gather and the core module (ca_stats_unpack);
    /// and, between a `prepare` that issued this rank's partials and its `finish`, (pseudo-rows, live rows).
    quarter: Option<(Rc<QuarterLink>, Module)>,
    quarter_step: Cell<Option<(usize, usize)>>,
    /// config::fused_boundary (six-row target boundaries with the layout8 norm-quant): the collapsed row this boundary's
    /// fused sum writes for the next boundary, and the norm-quant variant that reads such a row. Boxed: the Boundary
    /// stays small (the engine's main stack is near its limit).
    fused: Option<Box<FusedBoundary>>,
}
/// CANADA_SCRATCH_PREFILL_MHC_POST_STATS=1 (default off): in sequence-parallel prefill layers the
/// attention boundary's post-mix also forms the FFN boundary's mixing statistics in the same pass
/// over the residual (kernels/mhc_post_stats.cuh). The new residual is bit-identical; the
/// statistics are FP32 FMAs instead of tf32x3 tensor-core dots, so the mixes differ in the last bits.
pub(crate) fn prefill_post_stats_requested() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_PREFILL_MHC_POST_STATS") {
        Ok(v) => { ensure!(v == "0" || v == "1", "invalid fused mHC post/statistics policy"); Ok(v == "1") }
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(e) => Err(e.into()),
    }
}
/// Keep both canonical BF16 routing inputs and the projection's exact native
/// MXFP8 representation. These are transient, profile-owned graph buffers.
/// Shared by an attention boundary and its attention's QKV gather when the boundary's statistics run by rank
/// quarter (config::mhc_quarter_stats). `xin` is the gather's local input: QUARTER_HEAD pseudo-rows of 448 BF16
/// for partial bytes (a step's partials end at row QUARTER_HEAD), then the six QKV-A rows; `xout` the same rows
/// gathered (four ranks side by side in each row).
pub(crate) struct QuarterLink {
    pub(crate) rank: usize,
    pub(crate) on: Cell<bool>,
    pub(crate) xin: Buffer<bf16>,
    pub(crate) xout: Buffer<bf16>,
    /// Pseudo-rows the coming gather carries: set by the boundary's `prepare`, taken by the gather.
    pub(crate) armed: Cell<Option<usize>>,
    pub(crate) partials_ready: Event,
    pub(crate) gathered: Event,
}
/// ceil(2,000 bytes of partials a live row x 6 rows / 896 bytes a pseudo-row).
pub(crate) const QUARTER_HEAD: usize = 14;
pub(crate) const QUARTER_COLUMNS: usize = 448;
struct NormQuant {
    op: Operator,
    values: Buffer<u8>,
    row_scales: Buffer<u8>,
    mma_scales: Buffer<u8>,
}
struct FusedBoundary {
    value: Buffer<bf16>,
    quant: Operator,
    /// This boundary's finish is fused (config::fused_boundary's mode names it); every boundary with the box can
    /// consume a collapsed row.
    produce: bool,
}
struct FullStats {
    stats: Operator,
    reduce: Operator,
    mix: Buffer<f32>,
    square: Buffer<f32>,
}
struct DeferredStats {
    side: Rc<Stream>,
    parent: usize,
    ready: Rc<Event>,
    done: Event,
}
pub(crate) fn overlap_enabled() -> bool {
    std::env::var("CANADA_SCRATCH_MHC_OVERLAP").as_deref() == Ok("1")
}
pub(crate) fn pdl_pair_requested() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_MHC_PDL_PAIR") {
        Ok(value) => { ensure!(value == "0" || value == "1", "invalid mHC PDL pair policy"); Ok(value == "1") },
        Err(std::env::VarError::NotPresent) => Ok(false), Err(error) => Err(error.into()),
    }
}
pub(crate) fn pdl_stats_requested() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_MHC_PDL_STATS") {
        Ok(value) => { ensure!(value == "0" || value == "1", "invalid mHC PDL stats policy"); Ok(value == "1") },
        Err(std::env::VarError::NotPresent) => Ok(false), Err(error) => Err(error.into()),
    }
}
fn wide_flag(name:&str)->Result<bool> {
    match std::env::var(name) {
        Ok(v)=>{ensure!(v=="0"||v=="1","invalid {name}");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(false),Err(e)=>Err(e.into()),
    }
}
pub(crate) fn wide_norm_quant_enabled()->Result<bool>{wide_flag("CANADA_SCRATCH_MHC_NORM_QUANT16")}
pub(crate) fn wide_stats_pdl_enabled()->Result<bool>{wide_flag("CANADA_SCRATCH_MHC_STATS_PDL16")}
/// CANADA_SCRATCH_DRAFT_STATS_OVERLAP=1 (default off): the drafter's boundaries fork their mixing statistics to the
/// coefficient stream at the normalization, as the target's small-row boundaries do (they run inline today:
/// ~20 us a boundary, six a draft pass). Same kernels and inputs, joined before the post-mix. Five-row drafters
/// only (the batched body's ten and fifteen rows keep inline statistics).
pub(crate) fn draft_stats_overlap()->Result<bool>{wide_flag("CANADA_SCRATCH_DRAFT_STATS_OVERLAP")}
pub(crate) fn attention16_overlap_enabled() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_ATTENTION16_OVERLAP") {
        Ok(value) => {
            ensure!(value == "0" || value == "1", "invalid M16 attention overlap policy");
            Ok(value == "1")
        }
        Err(std::env::VarError::NotPresent) => Ok(true),
        Err(error) => Err(error.into()),
    }
}
/// Explicit opt-in for compact M16 attention after the dense pipeline fix.
/// Older packages keep the original serial guard and cannot select this by
/// setting the old overlap flag alone.
pub(crate) fn compact16_overlap_enabled() -> Result<bool> {
    let enabled=match std::env::var("CANADA_SCRATCH_ATTENTION16_COMPACT_OVERLAP") {
        Ok(v)=>{ensure!(v=="0"||v=="1","invalid compact M16 overlap policy");v=="1"},
        Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
    };
    if enabled {
        ensure!(std::env::var("CANADA_SCRATCH_DENSE_PIPELINE_FENCE").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_GROUPED_SPARSE16").as_deref()==Ok("1")
            &&attention16_overlap_enabled()?,
            "compact M16 overlap requires the explicit pipeline fence and compact indexing");
    }
    Ok(enabled)
}
pub(crate) fn prefill_norm_fusion_enabled() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_PREFILL_NORM_FUSION") {
        Ok(value) => {
            ensure!(value == "0" || value == "1", "invalid prefill norm fusion policy");
            Ok(value == "1")
        }
        Err(std::env::VarError::NotPresent) => Ok(true),
        Err(error) => Err(error.into()),
    }
}
fn abi(op: &Operator, spec: &[(&str, &str)]) -> Result<()> {
    ensure!(
        op.entry
            .arguments
            .iter()
            .map(|a| (a.name.as_str(), a.kind.as_str()))
            .eq(spec.iter().copied()),
        "mHC primitive ABI drift: {}",
        op.entry.name
    );
    Ok(())
}
impl Boundary {
    pub(crate) fn set_wide_epilogues_for_probe(&mut self,enabled:bool,kernels:&Path)->Result<()> {
        ensure!([16,32].contains(&self.rows)
            &&std::env::var("CANADA_SCRATCH_WIDE_EPILOGUE_SCREEN").as_deref()==Ok("1"),
            "isolated wide residual update screen");
        if self.rows==32 {
            self.post_op=if enabled {Operator::load(&self.cuda,&kernels.join("mhc-post-tiles-v1"),"hc_post_b128_w1")?}
                else{Operator::load(&self.cuda,&kernels.join("mhc-post-v1"),"mhc_post")?};
        }
        Ok(())
    }
    pub(crate) fn enable_verify16_overlap_before_capture(&mut self,stream:&Stream)->Result<()> {
        ensure!(crate::ffn::wide_overlap_enabled(self.rows)?&&self.prepared.get()==0
            &&self.deferred_stats.is_none()&&!self.stats_pending.get()&&self.cuda.owns_stream(stream),
            "uncaptured M16 FFN coefficient overlap admission");
        self.deferred_stats=Some(DeferredStats {side:self.cuda.coefficient_stream(stream)?,
            parent:stream.raw()as usize,ready:Rc::new(self.cuda.fence_event()?),done:self.cuda.fence_event()?});
        Ok(())
    }
    /// Wide sequence-parallel boundaries: run the coefficient statistics on the shared
    /// coefficient stream, joined before the post (same kernels/inputs; schedule only).
    pub(crate) fn set_wide_stats_overlap(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!(!self.stats_pending.get() && self.cuda.owns_stream(stream), "idle wide coefficient overlap");
        if enabled && self.deferred_stats.is_none() {
            self.deferred_stats=Some(DeferredStats { side:self.cuda.coefficient_stream(stream)?,
                parent:stream.raw()as usize,ready:Rc::new(self.cuda.fence_event()?),done:self.cuda.fence_event()? });
        }
        if !enabled { self.deferred_stats=None; }
        Ok(())
    }
    pub(crate) fn set_two_mma_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MHC_TWO_MMA_SCREEN").as_deref()==Ok("1")
            &&[6,16].contains(&self.rows)&&self.stats_two_mma.is_some()
            &&!self.stats_pending.get()&&self.cuda.owns_stream(stream),"idle two-MMA statistics screen");
        self.use_two_mma=enabled;Ok(())
    }
    pub(crate) fn set_verify16_overlap_for_probe(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!((crate::ffn::wide_overlap_probe(self.rows)
            || ((std::env::var("CANADA_SCRATCH_VERIFY16_OVERLAP_SCREEN").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_GROUPED16_SERIAL_ATTENTION").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_QKV_SCHEDULE_SCREEN").as_deref()==Ok("1"))
                && self.rows==16)) && !self.stats_pending.get() && self.cuda.owns_stream(stream),
            "idle wide coefficient schedule screen");
        if enabled && self.deferred_stats.is_none() {
            self.deferred_stats=Some(DeferredStats { side:self.cuda.coefficient_stream(stream)?,
                parent:stream.raw()as usize,ready:Rc::new(self.cuda.fence_event()?),done:self.cuda.fence_event()? });
        }
        if !enabled { self.deferred_stats=None; }
        Ok(())
    }
    pub(crate) fn enable_verify16_overlap_for_probe(&mut self, stream: &Stream) -> Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_VERIFY16_OVERLAP_SCREEN").as_deref()==Ok("1")
            && self.rows==16 && self.prepared.get()==0 && self.deferred_stats.is_none()
            && self.cuda.owns_stream(stream),"isolated uncaptured M16 coefficient overlap admission");
        self.deferred_stats=Some(DeferredStats { side:self.cuda.coefficient_stream(stream)?,
            parent:stream.raw()as usize,ready:Rc::new(self.cuda.fence_event()?),done:self.cuda.fence_event()? });
        Ok(())
    }
    pub fn new(
        cuda: &Cuda,
        stream: &Stream,
        weights: Rc<Weights>,
        layer: usize,
        branch: &str,
        rows: usize,
        kernels: &Path,
    ) -> Result<Self> {
        Self::new_named(
            cuda,
            stream,
            weights,
            crate::layer_id::LayerId::target(layer)?,
            branch,
            rows,
            kernels,
        )
    }
    pub fn new_named(
        cuda: &Cuda,
        stream: &Stream,
        weights: Rc<Weights>,
        id: crate::layer_id::LayerId,
        branch: &str,
        rows: usize,
        kernels: &Path,
    ) -> Result<Self> {
        ensure!(
            ["attn", "ffn"].contains(&branch)
                && crate::config::mhc_rows(rows)
                && weights.in_context(cuda)
                && cuda.owns_stream(stream),
            "mHC branch/context/geometry"
        );
        stream.join_default()?;
        let prefix = format!("{}.hc_{branch}", id.prefix());
        for (suffix, shape) in [
            ("fn", vec![24, 20480]),
            ("base", vec![24]),
            ("scale", vec![3]),
        ] {
            let v = weights.view(&format!("{prefix}_{suffix}"))?;
            ensure!(
                v.dtype == Dtype::F32 && v.shape == shape,
                "canonical mHC coefficient shape/dtype"
            );
        }
        let norm_name = format!("{}.{branch}_norm.weight", id.prefix());
        let nw = weights.view(&norm_name)?;
        ensure!(
            nw.dtype == Dtype::Bf16 && nw.shape == [5120],
            "canonical mHC norm shape/dtype"
        );
        let norm_weight = cuda.allocate::<bf16>(5120)?;
        unsafe {
            cuda.copy_device(
                norm_weight.ptr(),
                weights.ptr(&norm_name)?,
                norm_weight.bytes(),
                stream,
            )
        }?;
        let root = kernels.join("triton-hc-v2");
        let mut stats = Operator::load(
            cuda,
            &root,
            if rows <= 8 {
                "hc_stats_mb8_s2"
            } else if rows < 64 {
                "hc_stats_mb16_s2"
            } else {
                "hc_stats_mb16_s1"
            },
        )?;
        let mut reduce = Operator::load(cuda, &root, "hc_reduce80")?;
        let two_mma_requested=match std::env::var("CANADA_SCRATCH_MHC_TWO_MMA") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid two-MMA policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(e)=>return Err(e.into()),
        };
        let eligible=[6,16].contains(&rows)&&!id.is_draft();
        let stats_two_mma=if eligible&&(two_mma_requested
            ||std::env::var("CANADA_SCRATCH_MHC_TWO_MMA_SCREEN").as_deref()==Ok("1")) {
            let root=if std::env::var("CANADA_SCRATCH_MHC_TWO_MMA_STAGE1").as_deref()==Ok("1")
                {"mhc-small-stats-two-mma-stage1-s806"}else{"mhc-small-stats-two-mma-s806"};
            let op=Operator::load(cuda,&kernels.join(root),
                &format!("hc_stats_two_mma_m{rows}"))?;
            abi(&op,&[("X","*bf16"),("W","*fp32"),("PM","*fp32"),("PS","*fp32")])?;
            ensure!(op.entry.grid.rows_div==rows as u32&&op.entry.grid.y==80,
                "two-MMA physical row specialization");
            Some(op)
        }else{None};
        let combine = if rows <= 8 {
            Operator::load(cuda, &root, "hc_combine_norm_small")?
        } else {
            Operator::load(
                cuda,
                &kernels.join("mhc-combine-plain-v1"),
                "hc_combine_plain",
            )?
        };
        let small16 = rows==16
            && std::env::var("CANADA_SCRATCH_VERIFY16_SMALL_KERNELS").as_deref()==Ok("1");
        let post_tiles = (rows == 32 && !id.is_draft() && crate::layer_attention::wide_epilogues_enabled()?)
            || (rows == 6 || small16) && !id.is_draft()
            && match std::env::var("CANADA_SCRATCH_MHC_POST_TILES") {
                Ok(v) => { ensure!(v == "0" || v == "1", "invalid mHC post tile policy"); v == "1" },
                Err(std::env::VarError::NotPresent) => false,
                Err(e) => return Err(e.into()),
            };
        let mut post_op = if post_tiles {
            Operator::load(cuda, &kernels.join("mhc-post-tiles-v1"), "hc_post_b128_w1")?
        } else {
            Operator::load(cuda, &kernels.join("mhc-post-v1"), "mhc_post")?
        };
        let fusion_ab = std::env::var("CANADA_SCRATCH_PREFILL_NORM_FUSION_AB").as_deref() == Ok("1");
        let use_fused_norm = rows >= 128 && prefill_norm_fusion_enabled()? && !fusion_ab;
        let fused_norm = if rows >= 128 && (use_fused_norm || fusion_ab)
        {
            let root = kernels.join("mhc-combine-norm-cute-v3");
            let op = Primitives::open(cuda, &root.join("libcanada_primitives.so"), &root.join("bindings.json"))?;
            ensure!(op.bindings.abi == 2 && op.bindings.entries.len() == 1
                && op.bindings.entries[0].kernel_id == "mhc.combine_rms.h5120"
                && op.bindings.entries[0].arguments.iter().map(|a|a.kind.as_str())
                    .eq(["pointer","pointer","pointer","pointer","i64","f32"]),
                "prefill combine/RMS primitive ABI");
            Some(op)
        } else {
            None
        };
        abi(
            &stats,
            &[
                ("x_ptr", "*bf16"),
                ("w_ptr", "*fp32"),
                ("part_mix_ptr", "*fp32"),
                ("part_sq_ptr", "*fp32"),
                ("M", "i32"),
                ("K", "i32"),
                ("x_stride_m", "i32"),
                ("w_stride_n", "i32"),
            ],
        )?;
        abi(
            &reduce,
            &[
                ("part_mix_ptr", "*fp32"),
                ("part_sq_ptr", "*fp32"),
                ("scale_ptr", "*fp32"),
                ("base_ptr", "*fp32"),
                ("pre_ptr", "*fp32"),
                ("post_ptr", "*fp32"),
                ("comb_ptr", "*fp32"),
                ("m", "i32"),
                ("inv_k", "fp32"),
                ("rms_eps", "fp32"),
            ],
        )?;
        if rows <= 8 {
            abi(
                &combine,
                &[
                    ("X", "*bf16"),
                    ("P", "*fp32"),
                    ("W", "*bf16"),
                    ("Y", "*bf16"),
                ],
            )?;
        } else {
            abi(
                &combine,
                &[
                    ("x_ptr", "*bf16"),
                    ("pre_ptr", "*fp32"),
                    ("y_ptr", "*bf16"),
                    ("H", "i32"),
                    ("x_stride_m", "i32"),
                    ("pre_stride_m", "i32"),
                    ("pre_stride_k", "i32"),
                    ("y_stride_m", "i32"),
                ],
            )?;
        }
        abi(
            &post_op,
            &[
                ("X", "*bf16"),
                ("R", "*bf16"),
                ("P", "*fp32"),
                ("C", "*fp32"),
                ("Y", "*bf16"),
            ],
        )?;
        let norm = Native::open_with_rms(
            cuda,
            &kernels.join("prelude-v1"),
            &kernels.join("mhc-rms-v1"),
        )?;
        let fullk_requested = match std::env::var("CANADA_SCRATCH_HC_FULLK") {
            Ok(value) => {
                ensure!(value == "0" || value == "1", "invalid full-K HC policy");
                value == "1"
            }
            Err(std::env::VarError::NotPresent) => false,
            Err(error) => return Err(error.into()),
        };
        let fullk_ab = std::env::var("CANADA_SCRATCH_HC_FULLK_AB").as_deref() == Ok("1");
        // The component screen found a regression at <=512 rows. Keep those
        // on the original kernels, including every decode/verification profile.
        let use_full_stats = rows >= 1024 && fullk_requested && !fullk_ab;
        let full_stats = if rows >= 1024 && (fullk_requested || fullk_ab) {
            let root = kernels.join("mhc-fullk-v1");
            let stats = Operator::load(cuda, &root, if rows <= 2048 {
                "hc_fullk_mb16_s4"
            } else if rows <= 3072 {
                "hc_fullk_mb16_s3"
            } else {
                "hc_fullk_mb32_s3"
            })?;
            abi(&stats, &[("x_ptr","*bf16"),("w_ptr","*fp32"),
                ("mix_ptr","*fp32"),("sq_ptr","*fp32"),("M","i32"),
                ("x_stride_m","i32"),("w_stride_n","i32")])?;
            let reduced = Operator::load(cuda, &root, "hc_reduce1")?;
            ensure!(reduced.entry.arguments.iter().map(|a|(&a.name,&a.kind))
                .eq(reduce.entry.arguments.iter().map(|a|(&a.name,&a.kind))),
                "one-slice HC reduction ABI drift");
            Some(FullStats { stats, reduce: reduced,
                mix: cuda.allocate_rows(rows * 24)?, square: cuda.allocate_rows(rows)? })
        } else { None };
        let wide_screen = rows == 16 && !id.is_draft()
            && std::env::var("CANADA_SCRATCH_WIDE_MHC_SCREEN").as_deref() == Ok("1");
        let wide_norm = rows == 16 && !id.is_draft() && wide_norm_quant_enabled()?;
        let mut norm_quant = if (rows == 6 && !id.is_draft()
            && std::env::var("CANADA_SCRATCH_NORM_QUANT_FUSION").as_deref() == Ok("1"))
            || wide_norm || wide_screen {
            ensure!(!crate::fp8_linear::narrow_enabled(), "norm/quant fusion requires the checked legacy input quantizer");
            let layout8 = rows == 16 || std::env::var("CANADA_SCRATCH_NORM_QUANT_LAYOUT8").as_deref() == Ok("1");
            let op = Operator::load(cuda,
                &kernels.join(if layout8 {"mhc-norm-quant-layout8-v1"} else {"mhc-norm-quant-v1"}),
                if layout8 {"hc_norm_quant_p2"} else {"hc_norm_quant_p4"})?;
            abi(&op, &[("X","*bf16"),("P","*fp32"),("W","*bf16"),("Y","*bf16"),
                ("Q","*fp8e4nv"),("SR","*u8"),("SM","*u8")])?;
            // The unused MMA tile lanes keep their initialization across
            // executions; they cannot borrow another profile's row arena.
            let mut mma_scales = cuda.allocate(20480)?;
            mma_scales.zero()?;
            stream.join_default()?;
            Some(NormQuant {op, values:cuda.allocate_rows(rows*5120)?,
                row_scales:cuda.allocate_rows(rows*160)?, mma_scales})
        } else {None};
        let pdl_pair = rows == 6 && !id.is_draft() && pdl_pair_requested()?;
        let pdl_screen = rows == 6 && !id.is_draft()
            && std::env::var("CANADA_SCRATCH_MHC_PDL_PAIR_SCREEN").as_deref() == Ok("1");
        let mut alternate_pdl_pair = if pdl_pair || pdl_screen {
            ensure!(post_tiles && norm_quant.is_some()
                && std::env::var("CANADA_SCRATCH_NORM_QUANT_LAYOUT8").as_deref() == Ok("1"),
                "mHC PDL pair requires the qualified M6 tile and layout8 quantizer");
            let root = kernels.join("mhc-pdl-pair-v1");
            let pair = (Operator::load(cuda, &root, "hc_post_b128_w1")?,
                Operator::load(cuda, &root, "hc_norm_quant_p2")?);
            for (candidate, control) in [(&pair.0, &post_op), (&pair.1, &norm_quant.as_ref().unwrap().op)] {
                ensure!(candidate.entry.programmatic_stream_serialization
                    && candidate.entry.arguments.iter().map(|a| (&a.name, &a.kind))
                        .eq(control.entry.arguments.iter().map(|a| (&a.name, &a.kind))),
                    "mHC PDL operator contract mismatch");
            }
            Some(pair)
        } else { None };
        if pdl_pair {
            let pair = alternate_pdl_pair.as_mut().unwrap();
            std::mem::swap(&mut post_op, &mut pair.0);
            std::mem::swap(&mut norm_quant.as_mut().unwrap().op, &mut pair.1);
            if !pdl_screen { alternate_pdl_pair = None; }
        }
        let pdl_stats = !id.is_draft() && ((rows == 6 && pdl_stats_requested()?)
            || (rows == 16 && wide_stats_pdl_enabled()?));
        let mut alternate_pdl_stats = if pdl_stats || pdl_screen || wide_screen {
            ensure!(!two_mma_requested && !use_full_stats,
                "mHC PDL stats requires the original 80-slice statistics path");
            let root = kernels.join(if rows == 16 {"mhc-pdl-stats16-v1"} else {"mhc-pdl-stats-v1"});
            let pair = (Operator::load(cuda, &root, if rows == 16 {"hc_stats_mb16_s2"} else {"hc_stats_mb8_s2"})?,
                Operator::load(cuda, &root, "hc_reduce80")?);
            for (candidate, control) in [(&pair.0, &stats), (&pair.1, &reduce)] {
                ensure!(candidate.entry.programmatic_stream_serialization
                    && candidate.entry.arguments.iter().map(|a| (&a.name, &a.kind))
                        .eq(control.entry.arguments.iter().map(|a| (&a.name, &a.kind))),
                    "mHC PDL statistics contract mismatch");
            }
            Some(pair)
        } else { None };
        if pdl_stats {
            let pair = alternate_pdl_stats.as_mut().unwrap();
            std::mem::swap(&mut stats, &mut pair.0);
            std::mem::swap(&mut reduce, &mut pair.1);
            if !pdl_screen && !wide_screen { alternate_pdl_stats = None; }
        }
        // Verify-layer stage 1: only where the variant exists (the six-row layout8 quantizer, plain or as the PDL pair's
        // second kernel) and no screen swaps operators.
        let fused_mode = crate::config::fused_boundary()?;
        let fused = if fused_mode != 0 && rows == 6 && !id.is_draft() && norm_quant.is_some()
            && std::env::var("CANADA_SCRATCH_NORM_QUANT_LAYOUT8").as_deref() == Ok("1")
            && alternate_pdl_pair.is_none() && !wide_screen {
            let quant = Operator::load(cuda, &kernels.join("mhc-value-norm-quant-v1"),
                if pdl_pair {"hc_value_norm_quant_p2_pdl"} else {"hc_value_norm_quant_p2"})?;
            abi(&quant, &[("V","*bf16"),("W","*bf16"),("Y","*bf16"),("Q","*fp8e4nv"),("SR","*u8"),("SM","*u8")])?;
            let control = &norm_quant.as_ref().unwrap().op.entry;
            ensure!(quant.entry.programmatic_stream_serialization == control.programmatic_stream_serialization
                && quant.entry.grid.y == control.grid.y && quant.entry.grid.rows_div == control.grid.rows_div
                && quant.entry.metadata.num_warps == control.metadata.num_warps,
                "fused boundary norm-quant variant contract");
            // Rows past a step's live rows keep finite values (the consumer quantizes all six rows, as today).
            let mut value = cuda.allocate_rows::<bf16>(rows * 5120)?;
            value.zero()?;
            stream.join_default()?;
            let produce = fused_mode == 1 || (fused_mode == 2 && branch == "attn") || (fused_mode == 3 && branch == "ffn");
            Some(Box::new(FusedBoundary { value, quant, produce }))
        } else { None };
        let result = Self {
            pdl_stats,
            alternate_pdl_stats,
            pdl_pair,
            alternate_pdl_pair,
            cuda: cuda.clone(),
            fn_pointer: weights.ptr(&format!("{prefix}_fn"))?,
            base_pointer: weights.ptr(&format!("{prefix}_base"))?,
            scale_pointer: weights.ptr(&format!("{prefix}_scale"))?,
            weights,
            rows,
            norm,
            plain_norm: if rows > 8 {
                Some(Native::open(cuda, &kernels.join("prelude-v1"))?)
            } else {
                None
            },
            combined: if rows > 8 {
                Some(cuda.allocate_rows(rows * 5120)?)
            } else {
                None
            },
            norm_weight,
            stats,
            stats_two_mma,
            use_two_mma:eligible&&two_mma_requested,
            reduce,
            combine,
            fused_norm,
            use_fused_norm,
            post_op,
            partial_mix: cuda.allocate_rows(80 * rows * 24)?,
            partial_square: cuda.allocate_rows(80 * rows)?,
            pre: cuda.allocate_rows(rows * 4)?,
            post: cuda.allocate_rows(rows * 4)?,
            comb: cuda.allocate_rows(rows * 16)?,
            normalized: cuda.allocate_rows(rows * 5120)?,
            output: cuda.allocate_rows(rows * 20480)?,
            stream: Cell::new(0),
            prepared: Cell::new(0),
            prepared_rows: Cell::new(rows),
            prepared_pre: Cell::new(0),
            // s673: the attention branch passed paired logits, abort state and
            // changed-history handoff. The independent M16 FFN path stays off.
            deferred_stats: if overlap_enabled() && (!id.is_draft() || draft_stats_overlap()?)
                && (rows <= 8 || (rows == 16 && branch == "attn" && attention16_overlap_enabled()?)) {
                Some(DeferredStats {
                    side: cuda.coefficient_stream(stream)?,
                    parent: stream.raw() as usize,
                    ready: Rc::new(cuda.fence_event()?),
                    done: cuda.fence_event()?,
                })
            } else {
                None
            },
            late_point: Cell::new(if id.is_draft() { 0 } else {
                let (attention, ffn) = stats_late_requested(rows)?;
                if branch == "attn" { attention } else { ffn }
            }),
            ffn_branch: branch != "attn",
            stats_pending: Cell::new(false),
            full_stats,
            use_full_stats,
            norm_quant_active: norm_quant.is_some() && (rows != 16 || wide_norm),
            norm_quant,
            quant_ready: false,
            post_stats: None,
            stats_precomputed: Cell::new(0),
            quarter: None,
            quarter_step: Cell::new(None),
            fused,
        };
        stream.synchronize()?;
        Ok(result)
    }
    /// config::mhc_quarter_stats: this (attention) boundary's statistics by rank quarter over `link`. Needs the
    /// six-row profile's forked 80-slice kernel (not the two-MMA or whole-K forms).
    pub(crate) fn install_quarter(&mut self, link: Rc<QuarterLink>, module: &Module) -> Result<()> {
        ensure!(self.rows == 6 && !self.ffn_branch && self.deferred_stats.is_some() && !self.use_two_mma
            && !self.use_full_stats && link.rank < 4 && self.quarter.is_none() && self.post_stats.is_none(),
            "quarter statistics boundary profile");
        module.function("ca_stats_unpack")?;
        self.quarter = Some((link, module.clone()));
        Ok(())
    }
    /// The plain 80-slice statistics path that finish_with_next_stats can stand in for.
    fn plain_stats(&self) -> bool {
        self.rows >= 64 && !self.use_two_mma && !self.use_full_stats && self.deferred_stats.is_none()
    }
    /// Opt this boundary into finish_with_next_stats (sequence-parallel prefill layers).
    pub(crate) fn enable_post_stats(&mut self, module: &Module, next: &Self) -> Result<()> {
        // The launch itself refuses a module of another context.
        ensure!(self.plain_stats() && next.plain_stats() && next.rows == self.rows && self.rows % 16 == 0,
            "fused mHC post/statistics geometry");
        self.post_stats = Some(module.clone());
        Ok(())
    }
    pub(crate) fn post_stats_enabled(&self) -> bool { self.post_stats.is_some() }
    /// finish(), and in the same pass the mixing statistics of the new residual for `next`, whose
    /// prepare() on that residual then runs only the reduction.
    pub(crate) fn finish_with_next_stats(&self, sublayer: &Buffer<bf16>, residual: &Buffer<bf16>,
        next: &Self, stream: &Stream) -> Result<()> {
        self.guard(residual, stream)?;
        let module = self.post_stats.as_ref().ok_or_else(|| anyhow::anyhow!("fused mHC post/statistics not enabled"))?;
        let rows = self.prepared_rows.get();
        ensure!(self.prepared.get() == residual.ptr() && sublayer.len() == self.rows * 5120
            && self.cuda.owns_buffer(sublayer) && rows == self.rows && next.rows == self.rows
            && next.plain_stats() && !next.stats_pending.get() && next.stats_precomputed.get() == 0,
            "fused mHC post/statistics requires a whole prepared shard and an idle next boundary");
        let (mut x, mut r, mut p, mut c, mut w) =
            (sublayer.ptr(), residual.ptr(), self.post.ptr(), self.comb.ptr(), next.fn_pointer);
        let (mut y, mut mix, mut sq, mut m) =
            (self.output.ptr(), next.partial_mix.ptr(), next.partial_square.ptr(), rows as i32);
        let mut args = [(&mut x as *mut u64).cast(), (&mut r as *mut u64).cast(), (&mut p as *mut u64).cast(),
            (&mut c as *mut u64).cast(), (&mut w as *mut u64).cast(), (&mut y as *mut u64).cast(),
            (&mut mix as *mut u64).cast(), (&mut sq as *mut u64).cast(), (&mut m as *mut i32).cast()];
        unsafe {
            module.function("ca_mhc_post_stats")?.launch(
                [rows.div_ceil(16) as u32, 20, 1], [256, 1, 1], 24 * 256 * 4, stream, &mut args)?;
        }
        if std::env::var("CANADA_SCRATCH_MHC_POST_STATS_CHECK").as_deref() == Ok("1") {
            self.check_post_stats(sublayer, residual, next, stream)?;
        }
        next.stats_precomputed.set(self.output.ptr());
        Ok(())
    }
    /// Diagnostic (CANADA_SCRATCH_MHC_POST_STATS_CHECK=1): rerun the separate post-mix and
    /// statistics kernels on the same inputs into scratch and compare with the fused results:
    /// the new residual must be bit-identical; the statistics partials are reported by their
    /// largest absolute and relative differences. Reports the first 64 boundaries of a process.
    fn check_post_stats(&self, sublayer: &Buffer<bf16>, residual: &Buffer<bf16>, next: &Self,
        stream: &Stream) -> Result<()> {
        static CHECKED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let index = CHECKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if index >= 64 { return Ok(()); }
        let rows = self.rows;
        let y_ref = self.cuda.allocate::<bf16>(rows * 20480)?;
        let mix_ref = self.cuda.allocate::<f32>(80 * rows * 24)?;
        let sq_ref = self.cuda.allocate::<f32>(80 * rows)?;
        unsafe {
            self.post_op.launch(rows as u32,
                &[sublayer.ptr(), residual.ptr(), self.post.ptr(), self.comb.ptr(), y_ref.ptr()], stream)?;
            next.stats.launch(rows as u32,
                &[self.output.ptr(), next.fn_pointer, mix_ref.ptr(), sq_ref.ptr(), rows as u64, 20480, 20480, 20480],
                stream)?;
        }
        stream.synchronize()?;
        let fused_y = self.output.download()?;
        let ref_y = y_ref.download()?;
        let y_differ = fused_y.iter().zip(&ref_y).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let compare = |fused: &Buffer<f32>, reference: &Buffer<f32>| -> Result<(f32, f32, usize)> {
            let (a, b) = (fused.download()?, reference.download()?);
            let mut worst = (0f32, 0f32, 0usize);
            for (x, y) in a.iter().zip(&b) {
                let d = (x - y).abs();
                worst.0 = worst.0.max(d);
                worst.1 = worst.1.max(d / y.abs().max(1e-30));
                if x.to_bits() != y.to_bits() { worst.2 += 1; }
            }
            Ok(worst)
        };
        let mix = compare(&next.partial_mix, &mix_ref)?;
        let sq = compare(&next.partial_square, &sq_ref)?;
        eprintln!("mhc post/stats check {index}: rows {rows} residual bits differing {y_differ}/{}; \
            mix partials max abs {:.3e} max rel {:.3e} differing {}/{}; square partials max abs {:.3e} max rel {:.3e} differing {}/{}",
            ref_y.len(), mix.0, mix.1, mix.2, 80 * rows * 24, sq.0, sq.1, sq.2, 80 * rows);
        ensure!(y_differ == 0, "fused mHC post-mix differs from the separate kernel");
        Ok(())
    }
    fn guard(&self, residual: &Buffer<bf16>, stream: &Stream) -> Result<()> {
        ensure!(
            residual.len() == self.rows * 20480
                && self.cuda.owns_buffer(residual)
                && self.cuda.owns_stream(stream)
                && self.weights.in_context(&self.cuda),
            "mHC residual context/shape"
        );
        let s = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == s,
            "mHC requires one ordered stream"
        );
        self.stream.set(s);
        Ok(())
    }
    /// The Target owner admits this switch only with no captured executions.
    pub(crate) fn overlap_active(&self)->bool { self.deferred_stats.is_some() }
    /// Plan screen only (graphs retired by the caller): move this boundary's statistics fork. No arithmetic change.
    pub(crate) fn set_stats_late_point_for_probe(&self, point: u8) -> Result<()> {
        ensure!(point <= 8 && !self.stats_pending.get(), "idle mHC late statistics point");
        self.late_point.set(if self.deferred_stats.is_some() { point } else { 0 });
        Ok(())
    }
    pub(crate) fn set_norm_fusion(&mut self, enabled: bool) -> Result<()> {
        ensure!(self.rows >= 128 && !self.stats_pending.get()
            && (!enabled || self.fused_norm.is_some()), "unavailable or busy prefill norm fusion");
        self.use_fused_norm = enabled;
        Ok(())
    }
    pub(crate) fn fullk_active(&self) -> bool { self.use_full_stats }
    pub(crate) fn norm_quant_enabled(&self) -> bool { self.norm_quant.is_some() && self.norm_quant_active }
    pub(crate) fn set_pdl_pair_for_probe(&mut self, enabled: bool) -> Result<()> {
        ensure!(self.rows == 6 && self.alternate_pdl_pair.is_some()
            && std::env::var("CANADA_SCRATCH_MHC_PDL_PAIR_SCREEN").as_deref() == Ok("1"),
            "mHC PDL swap requires the isolated M6 screen");
        if self.pdl_pair != enabled {
            let pair = self.alternate_pdl_pair.as_mut().unwrap();
            std::mem::swap(&mut self.post_op, &mut pair.0);
            std::mem::swap(&mut self.norm_quant.as_mut().unwrap().op, &mut pair.1);
            self.pdl_pair = enabled;
        }
        Ok(())
    }
    pub(crate) fn set_pdl_stats_for_probe(&mut self, enabled: bool) -> Result<()> {
        ensure!(self.rows == 6 && !self.use_two_mma && !self.use_full_stats
            && self.alternate_pdl_stats.is_some()
            && std::env::var("CANADA_SCRATCH_MHC_PDL_PAIR_SCREEN").as_deref() == Ok("1"),
            "mHC PDL statistics swap requires the isolated M6 screen");
        if self.pdl_stats != enabled {
            let pair = self.alternate_pdl_stats.as_mut().unwrap();
            std::mem::swap(&mut self.stats, &mut pair.0);
            std::mem::swap(&mut self.reduce, &mut pair.1);
            self.pdl_stats = enabled;
        }
        Ok(())
    }
    pub(crate) fn set_wide_mhc_for_probe(&mut self, mode:u8)->Result<()> {
        ensure!(self.rows==16 && mode<=3 && !self.stats_pending.get()
            && self.norm_quant.is_some() && self.alternate_pdl_stats.is_some()
            && std::env::var("CANADA_SCRATCH_WIDE_MHC_SCREEN").as_deref()==Ok("1"),
            "wide mHC comparison requires idle prepared operators");
        let enabled=mode&1!=0;
        if self.pdl_stats!=enabled {
            let pair=self.alternate_pdl_stats.as_mut().unwrap();
            std::mem::swap(&mut self.stats,&mut pair.0);
            std::mem::swap(&mut self.reduce,&mut pair.1);self.pdl_stats=enabled;
        }
        self.norm_quant_active=mode&2!=0;self.quant_ready=false;Ok(())
    }
    pub(crate) fn quantized_input(&self) -> Option<(&Buffer<u8>, &Buffer<u8>)> {
        self.norm_quant.as_ref().filter(|_|self.quant_ready).map(|q|(&q.values,&q.mma_scales))
    }
    pub(crate) fn set_fullk(&mut self, enabled: bool) -> Result<()> {
        ensure!(!self.stats_pending.get(), "full-K policy while statistics pending");
        if self.rows < 1024 {
            ensure!(!self.use_full_stats, "small-row full-K policy");
            return Ok(());
        }
        ensure!(!enabled || self.full_stats.is_some(), "full-K statistics not prepared");
        self.use_full_stats = enabled;
        Ok(())
    }
    /// `previous_pre=None` is the model's initial copy-0 RMS input. Later
    /// branches consume the predecessor's coefficients, not their own pre-mix.
    pub fn prepare(
        &mut self,
        residual: &Buffer<bf16>,
        previous_pre: Option<&Buffer<f32>>,
        stream: &Stream,
    ) -> Result<()> {
        self.guard(residual, stream)?;
        ensure!(
            !self.stats_pending.get(),
            "mHC prepare before prior stats join"
        );
        self.quant_ready = false;
        // A collapsed row left by the boundary that finished just before (config::fused_boundary), if it is this one's.
        let fused_value = FUSED_VALUE.with(|v| v.take()).filter(|f| f.residual == residual.ptr()
            && previous_pre.is_some_and(|p| p.ptr() == f.pre) && f.stream == stream.raw() as usize
            && f.serial == crate::cuda::capture_serial());
        // Only the dynamic-M stock statistics path may shorten its packed
        // partial stride. The fixed-M two-MMA and full-K paths stay unchanged.
        // A wide profile's M/4 sequence-parallel boundaries do not match its
        // capacity and deliberately keep their existing complete row shard.
        let (live,live_norm)=if [6,128,512,1024,2048].contains(&self.rows)&&!self.use_two_mma&&!self.use_full_stats {
            ACTIVE_ROWS.with(|s|s.get().filter(|(_,capacity,owner,_)|*capacity==self.rows&&*owner==stream.raw()as usize)
                .map_or((self.rows,false),|(rows,_,_,normalize)|(rows,normalize)))
        }else{(self.rows,false)};
        self.prepared_rows.set(live);
        self.prepared_pre.set(previous_pre.map_or(0, |p| p.ptr()));
        // config::mhc_quarter_stats: this rank's slice partials are issued first. They read the residual only, and
        // the QKV gather that carries them comes two short kernels later (s1565: before the collapse and norm).
        let quarter_issued = !self.stats_knocked_out() && self.quarter.as_ref().is_some_and(|(link, _)| link.on.get());
        if quarter_issued { self.issue_quarter_partials(residual, stream)?; }
        if let Some(p) = previous_pre {
            ensure!(
                p.len() == self.rows * 4 && self.cuda.owns_buffer(p),
                "mHC predecessor pre shape/context"
            );
            if let Some(q) = self.norm_quant.as_ref().filter(|_| self.norm_quant_active) {
                if let (Some(f), Some(fused)) = (fused_value, self.fused.as_ref()) {
                    // The same RMS, normalized boundary and MXFP8 bytes from the row the fused sum collapsed.
                    unsafe {fused.quant.launch(self.rows as u32,
                        &[f.value,self.norm_weight.ptr(),self.normalized.ptr(),
                          q.values.ptr(),q.row_scales.ptr(),q.mma_scales.ptr()],stream)?;}
                } else {
                    unsafe {q.op.launch(self.rows as u32,
                        &[residual.ptr(),p.ptr(),self.norm_weight.ptr(),self.normalized.ptr(),
                          q.values.ptr(),q.row_scales.ptr(),q.mma_scales.ptr()],stream)?;}
                }
                self.quant_ready = true;
            } else if self.use_fused_norm {
                unsafe {
                    self.fused_norm.as_ref().unwrap().launch_words(0,
                        &[residual.ptr(), p.ptr(), self.norm_weight.ptr(), self.normalized.ptr(),
                          (if live_norm{live}else{self.rows})as u64, 1.0e-20f32.to_bits() as u64], stream)?;
                }
            } else if let Some(combined) = self.combined.as_ref() {
                // Qualified9..384-row path rounds the HC combine to BF16
                // before its separate CuTe RMSNorm. Any fused implementation
                // must retain that BF16 rounding boundary explicitly.
                unsafe {
                    self.combine.launch(
                        self.rows as u32,
                        &[
                            residual.ptr(),
                            p.ptr(),
                            combined.ptr(),
                            5120,
                            20480,
                            4,
                            1,
                            5120,
                        ],
                        stream,
                    )?;
                }
                self.plain_norm.as_ref().unwrap().rms(
                    combined,
                    &self.norm_weight,
                    &mut self.normalized,
                    self.rows,
                    5120,
                    stream,
                )?;
            } else {
                unsafe {
                    self.combine.launch(
                        self.rows as u32,
                        &[
                            residual.ptr(),
                            p.ptr(),
                            self.norm_weight.ptr(),
                            self.normalized.ptr(),
                        ],
                        stream,
                    )
                }?;
            }
        } else {
            self.norm.rms(
                residual,
                &self.norm_weight,
                &mut self.normalized,
                self.rows,
                20480,
                stream,
            )?;
        }
        if self.stats_knocked_out() {
            // Plan-screen knockout: the coefficients of the previous run of this step stay in place.
        } else if quarter_issued {
            // Issued before the collapse (issue_quarter_partials).
        } else if let Some(branch) = &self.deferred_stats {
            ensure!(
                branch.parent == stream.raw() as usize,
                "mHC stats parent changed"
            );
            // SGLang _hc_combine records the tiny-row fork after normalization.
            // Coefficients consume the residual, independently of attention/FFN.
            let point = self.late_point.get();
            LATE_FORKS.with(|forks| forks.borrow_mut().retain(|f| !Rc::ptr_eq(&f.ready, &branch.ready)));
            if point == 0 { branch.ready.record(stream)?; }
            else {
                // Released by `stats_point` at this sublayer's named point (or by `finish` if it never comes).
                LATE_FORKS.with(|forks| forks.borrow_mut().push(LateFork { ready: branch.ready.clone(), parent: branch.parent, point }));
            }
            self.stats_pending.set(true);
        } else if self.stats_precomputed.get() != 0 {
            ensure!(self.stats_precomputed.get() == residual.ptr() && self.prepared_rows.get() == self.rows
                && self.plain_stats(), "precomputed mHC statistics belong to another residual");
            self.stats_precomputed.set(0);
            self.run_reduce(stream)?;
        } else {
            self.run_stats(residual, stream)?;
        }
        self.prepared.set(residual.ptr());
        crate::layer_clock::fine(stream, crate::layer_clock::FINE_NORMALIZED)?;
        Ok(())
    }
    /// config::mhc_quarter_stats: this rank's 20 slices of the mixing statistics on the statistics stream: the
    /// exported kernel on this rank's stream of the residual and of the projection, writing where the QKV gather
    /// picks the bytes up.
    fn issue_quarter_partials(&self, residual: &Buffer<bf16>, stream: &Stream) -> Result<()> {
        let (link, _) = self.quarter.as_ref().unwrap();
        let branch = self.deferred_stats.as_ref().unwrap();
        let m = self.prepared_rows.get();
        let pseudo = (2000 * m).div_ceil(QUARTER_COLUMNS * 2);
        ensure!(branch.parent == stream.raw() as usize && (1..=self.rows).contains(&m) && pseudo <= QUARTER_HEAD
            && link.armed.get().is_none() && self.quarter_step.get().is_none(), "quarter statistics step");
        LATE_FORKS.with(|forks| forks.borrow_mut().retain(|f| !Rc::ptr_eq(&f.ready, &branch.ready)));
        branch.ready.record(stream)?;
        branch.ready.wait(&branch.side)?;
        let out = link.xin.ptr() + ((QUARTER_HEAD - pseudo) * QUARTER_COLUMNS * 2) as u64;
        unsafe {
            self.stats.launch_y(m as u32, 20, &[residual.ptr() + (5120 * link.rank * 2) as u64,
                self.fn_pointer + (5120 * link.rank * 4) as u64, out, out + (480 * m * 4) as u64,
                m as u64, 20480, 20480, 20480], &branch.side)
        }?;
        link.partials_ready.record(&branch.side)?;
        crate::layer_clock::fine_from(stream, &branch.side, crate::layer_clock::FINE_SIDE_PARTIALS)?;
        link.armed.set(Some(pseudo));
        self.quarter_step.set(Some((pseudo, m)));
        self.stats_pending.set(true);
        Ok(())
    }
    fn stats_knocked_out(&self) -> bool {
        crate::knockout::has(if self.ffn_branch { crate::knockout::Site::StatsFfn } else { crate::knockout::Site::StatsAttn })
    }
    /// The reduction half of the plain statistics path (partials already formed).
    fn run_reduce(&self, stream: &Stream) -> Result<()> {
        let rows = self.prepared_rows.get();
        unsafe {
            self.reduce.launch(
                rows as u32,
                &[
                    self.partial_mix.ptr(),
                    self.partial_square.ptr(),
                    self.scale_pointer,
                    self.base_pointer,
                    self.pre.ptr(),
                    self.post.ptr(),
                    self.comb.ptr(),
                    rows as u64,
                    (1f32 / 20480.).to_bits() as u64,
                    1e-20f32.to_bits() as u64,
                ],
                stream,
            )
        }
    }
    fn run_stats(&self, residual: &Buffer<bf16>, stream: &Stream) -> Result<()> {
        let rows=self.prepared_rows.get();
        ensure!((1..=self.rows).contains(&rows),"mHC prepared live-row extent");
        unsafe {
            if self.use_full_stats {
                let full = self.full_stats.as_ref().unwrap();
                full.stats.launch(rows as u32,
                    &[residual.ptr(), self.fn_pointer, full.mix.ptr(), full.square.ptr(),
                      rows as u64, 20480, 20480], stream)?;
                full.reduce.launch(rows as u32,
                    &[full.mix.ptr(), full.square.ptr(), self.scale_pointer, self.base_pointer,
                      self.pre.ptr(), self.post.ptr(), self.comb.ptr(), rows as u64,
                      (1.0f32/20480.0).to_bits() as u64, 1.0e-20f32.to_bits() as u64], stream)?;
                return Ok(());
            }
            if self.use_two_mma {
                self.stats_two_mma.as_ref().unwrap().launch(rows as u32,
                    &[residual.ptr(),self.fn_pointer,self.partial_mix.ptr(),self.partial_square.ptr()],stream)?;
            } else { self.stats.launch(
                self.prepared_rows.get() as u32,
                &[
                    residual.ptr(),
                    self.fn_pointer,
                    self.partial_mix.ptr(),
                    self.partial_square.ptr(),
                    rows as u64,
                    20480,
                    20480,
                    20480,
                ],
                stream,
            )?; }
            self.reduce.launch(
                rows as u32,
                &[
                    self.partial_mix.ptr(),
                    self.partial_square.ptr(),
                    self.scale_pointer,
                    self.base_pointer,
                    self.pre.ptr(),
                    self.post.ptr(),
                    self.comb.ptr(),
                    rows as u64,
                    (1f32 / 20480.).to_bits() as u64,
                    1e-20f32.to_bits() as u64,
                ],
                stream,
            )?;
        }
        Ok(())
    }
    /// Apply the globally reduced sublayer result. The all-reduce must retain
    /// BF16 output before this FMA-ordered residual mix.
    pub fn finish(
        &self,
        sublayer: &Buffer<bf16>,
        residual: &Buffer<bf16>,
        stream: &Stream,
    ) -> Result<()> {
        self.guard(residual, stream)?;
        ensure!(
            self.prepared.get() == residual.ptr()
                && sublayer.len() == self.rows * 5120
                && self.cuda.owns_buffer(sublayer),
            "mHC finish requires matching prepared residual and reduced BF16 sublayer"
        );
        if let Some((pseudo, m)) = self.quarter_step.take() {
            // The gather brought every rank's partials: unpack them into the reduce kernel's arrays, reduce.
            let (link, module) = self.quarter.as_ref().unwrap();
            let branch = self.deferred_stats.as_ref().unwrap();
            ensure!(self.stats_pending.get() && link.armed.get().is_none() && m == self.prepared_rows.get(),
                "quarter statistics partials were not gathered");
            link.gathered.wait(&branch.side)?;
            let (mut source, mut mix, mut square, mut rows) = (
                link.xout.ptr() + ((QUARTER_HEAD - pseudo) * QUARTER_COLUMNS * 4 * 2) as u64,
                self.partial_mix.ptr(), self.partial_square.ptr(), m as i32);
            let mut args: [*mut std::ffi::c_void; 4] = [(&mut source as *mut u64).cast(), (&mut mix as *mut u64).cast(),
                (&mut square as *mut u64).cast(), (&mut rows as *mut i32).cast()];
            unsafe { module.function("ca_stats_unpack")?.launch([(2000 * m as u32).div_ceil(256), 1, 1], [256, 1, 1], 0,
                &branch.side, &mut args) }?;
            self.run_reduce(&branch.side)?;
            branch.done.record(&branch.side)?;
            crate::layer_clock::fine_from(stream, &branch.side, crate::layer_clock::FINE_SIDE_STATS)?;
            branch.done.wait(stream)?;
            self.stats_pending.set(false);
        } else {
            self.join_deferred_stats(residual, stream)?;
        }
        if crate::knockout::has(crate::knockout::Site::Post) { return Ok(()); }
        unsafe {
            self.post_op.launch(
                self.prepared_rows.get() as u32,
                &[
                    sublayer.ptr(),
                    residual.ptr(),
                    self.post.ptr(),
                    self.comb.ptr(),
                    self.output.ptr(),
                ],
                stream,
            )
        }?;
        crate::layer_clock::fine(stream, crate::layer_clock::FINE_POST)
    }
    /// The deferred statistics branch's join before this boundary consumes its coefficients (finish, finish_fused).
    fn join_deferred_stats(&self, residual: &Buffer<bf16>, stream: &Stream) -> Result<()> {
        if let Some(branch) = self.deferred_stats.as_ref().filter(|_| !self.stats_knocked_out()) {
            ensure!(
                self.stats_pending.get(),
                "mHC finish without deferred stats fork"
            );
            // Match _hc_mix_stats: issue the main sublayer first, then record
            // independent coefficient work immediately before its consuming join.
            // This preserves the main CUDA-graph chain and all numeric kernels.
            // A late fork whose point never came on this path is released here, before its join.
            let unreleased = LATE_FORKS.with(|forks| {
                let mut forks = forks.borrow_mut();
                let before = forks.len();
                forks.retain(|f| !Rc::ptr_eq(&f.ready, &branch.ready));
                forks.len() != before
            });
            if unreleased { branch.ready.record(stream)?; }
            branch.ready.wait(&branch.side)?;
            self.run_stats(residual, &branch.side)?;
            branch.done.record(&branch.side)?;
            crate::layer_clock::fine_from(stream, &branch.side, crate::layer_clock::FINE_SIDE_STATS)?;
            branch.done.wait(stream)?;
            self.stats_pending.set(false);
        }
        Ok(())
    }
    /// Whether this boundary's next finish can be fused into the sublayer's sum (config::fused_boundary): the variant
    /// is loaded, the quantizer is active and nothing replaces the plain sum / post-mix / statistics path.
    pub(crate) fn fusion_possible(&self) -> bool {
        self.fused.as_ref().is_some_and(|f| f.produce) && self.norm_quant_active && self.prepared.get() != 0
            && self.quarter_step.get().is_none() && self.post_stats.is_none() && !self.stats_knocked_out()
            && !crate::knockout::has(crate::knockout::Site::Post) && !crate::knockout::has(crate::knockout::Site::Sums)
            && (self.deferred_stats.is_none() || self.stats_pending.get())
    }
    /// Verify-layer stage 1: `finish` with the sublayer's sum. Joins the statistics, then one kernel sums `partial`
    /// over the four ranks into `reduced` (as the plain sum stores it), writes this boundary's output stream (as the
    /// post-mix does) and its collapse with this boundary's pre (as the next boundary's norm-quant computes it before
    /// its RMS), which the next `prepare` takes. Callers check `fusion_possible` first.
    pub(crate) fn finish_fused(&self, partial: &Buffer<bf16>, reduced: &mut Buffer<bf16>, residual: &Buffer<bf16>,
        comm: &crate::roce::Roce, stream: &Stream) -> Result<()> {
        self.guard(residual, stream)?;
        ensure!(self.fusion_possible() && comm.fused_boundary_ready() && self.prepared.get() == residual.ptr()
            && partial.len() == self.rows * 5120 && reduced.len() == partial.len() && self.cuda.owns_buffer(partial)
            && self.cuda.owns_buffer(reduced), "fused mHC finish requires a prepared fusable boundary");
        let fused = self.fused.as_ref().unwrap();
        self.join_deferred_stats(residual, stream)?;
        // The kernel post-mixes min(prepared rows, rows summed): with a live-row sum scope and no mHC live-row scope the
        // rows past the sum's are padding (the separate post-mix would mix stale sums there).
        unsafe { comm.sum_hc_bf16(partial, reduced, residual, &self.post, &self.comb, &self.pre,
            &self.output, &fused.value, self.prepared_rows.get(), stream) }?;
        FUSED_VALUE.with(|v| v.set(Some(FusedValue { residual: self.output.ptr(), pre: self.pre.ptr(),
            value: fused.value.ptr(), stream: stream.raw() as usize, serial: crate::cuda::capture_serial() })));
        crate::layer_clock::fine(stream, crate::layer_clock::FINE_POST)
    }
    /// Read-only diagnostics after a completed verification. Layer-owned inputs
    /// survive the forward; never inject these snapshots into inference.
    pub(crate) fn norm_quant_diagnostic_inputs(&self) -> Result<Vec<(&'static str, Vec<u8>)>> {
        ensure!(self.rows == 6 && self.prepared.get() != 0
            && std::env::var("CANADA_SCRATCH_NORM_QUANT_DIAGNOSTIC").as_deref() == Ok("1"),
            "norm/quant diagnostic admission");
        let mut result = vec![
            ("residual", unsafe {self.cuda.read_device(self.prepared.get(),self.rows*20480*2)}?),
            ("weight", unsafe {self.cuda.read_device(self.norm_weight.ptr(),self.norm_weight.bytes())}?),
        ];
        if self.prepared_pre.get()!=0 {
            result.push(("pre",unsafe{self.cuda.read_device(self.prepared_pre.get(),self.rows*4*4)}?));
        }
        Ok(result)
    }
    pub fn input(&self) -> &Buffer<bf16> {
        &self.normalized
    }
    pub fn pre(&self) -> &Buffer<f32> {
        assert!(
            !self.stats_pending.get(),
            "mHC coefficients require finish/join"
        );
        &self.pre
    }
    pub fn post(&self) -> &Buffer<f32> {
        &self.post
    }
    pub fn comb(&self) -> &Buffer<f32> {
        &self.comb
    }
    pub fn partial_mix(&self) -> &Buffer<f32> {
        &self.partial_mix
    }
    pub fn partial_square(&self) -> &Buffer<f32> {
        &self.partial_square
    }
    pub fn output(&self) -> &Buffer<bf16> {
        &self.output
    }
}
#[cfg(test)]
mod late_statistics_tests {
    #[test]
    fn release_points_parse() {
        assert_eq!(super::parse_stats_late("00").unwrap(), (0, 0));
        assert_eq!(super::parse_stats_late("13").unwrap(), (1, 3));
        assert_eq!(super::parse_stats_late("25").unwrap(), (2, 5));
        assert_eq!(super::parse_stats_late("65").unwrap(), (6, 5));
        for bad in ["", "1", "31", "12", "16", "56", "130", "a3"] { assert!(super::parse_stats_late(bad).is_err(), "{bad}"); }
    }
}
