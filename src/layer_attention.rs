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
pub struct Layer {
    cuda: Cuda,
    rows: usize,
    mode: InputMode,
    qa: Linear,
    gathered_qkv: Option<Buffer<bf16>>,
    qb: Linear,
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
        let qb = projection(
            "wq_b",
            &if draft {
                "fp8-dspark-wq_b-r0-m5-v1".into()
            } else if rows == 6 && crate::fp8_linear::pdl_enabled() {
                "fp8-flashinfer-pdl-q-b-m6-rust-v2".into()
            } else {
                format!("fp8-canonical-wq_b-r0-m{rows}-v0")
            },
        )?;
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
        let ob = projection(
            "wo_b",
            &if draft {
                "fp8-dspark-wo_b-r0-m5-v1".into()
            } else {
                format!("fp8-canonical-wo_b-r0-m{rows}-v0")
            },
        )?;
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
        let kstore = Store::open(cuda, &kernels.join("kv-v2"))?;
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
        let decode = if generation_swa_bf16 {
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
        let append_decode=if append_bf16 {
            Some(Execution::Prefill(crate::attention::Prefill::new_small_bf16(cuda,
                &kernels.join("attention-small-bf16-s729"),rows,128,if layer>=2 {512}else{0})?))
        }else{None};
        let fp32_partials=std::env::var("CANADA_SCRATCH_SPLIT_ATTENTION_FP32").as_deref()==Ok("1");
        let split_decode=if sparse16_screen||(!draft && rows==6 && layer>=2
            && std::env::var("CANADA_SCRATCH_SPLIT_ATTENTION_SCREEN").as_deref()==Ok("1")) {
            ensure!(append_decode.is_some(),"split attention screen needs unsplit BF16 control");
            Some(Execution::Prefill(crate::attention::Prefill::new_split_bf16(cuda,kernels,rows,512,
                if sparse16_screen {3}else{8},sparse16_screen||fp32_partials,sparse16_screen)?))
        }else{None};
        let append_split=if append_bf16&&layer>=2
            &&std::env::var("CANADA_SCRATCH_APPEND_SPLIT_ATTENTION_SCREEN").as_deref()==Ok("1") {
            ensure!(split_decode.is_none()||fp32_partials,"combined split screens require FP32 partial experiment");
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
            gathered_qkv: if matches!(mode, InputMode::Split(_)) {
                Some(cuda.allocate_rows::<bf16>(rows * 1792)?)
            } else {
                None
            },
            qb,
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
        })
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
    fn project_input(&mut self, input: &Buffer<bf16>, positions: &Positions,
        locations: &RowLocations, cache: &mut PagedKv, comm: Option<&Roce>,
        shard: Option<(&mut crate::fp8_linear::RowShard,bool)>,
        quantized:Option<(&Buffer<u8>,&Buffer<u8>)>, precomputed:bool, stream: &Stream) -> Result<()> {
        if precomputed {
            self.paired_input_projection()?;
            ensure!(shard.is_none()&&quantized.is_none()&&comm.is_some(),"precomputed QKV input path");
        }
        ensure!(quantized.is_none() || (shard.is_none() && matches!(self.rows, 6 | 16)
            && matches!(self.mode,InputMode::Split(_)) && !self.qa.narrow_projection()),
            "prequantized normalized attention requires the checked legacy QKV quantizer");
        if let Some(q) = &self.query_overlap {
            ensure!(!q.pending.get() && q.parent == stream.raw() as usize,
                "attention query overlap is pending or has a foreign parent");
        }
        ensure!(
            self.gathered_qkv.is_none() || comm.is_some(),
            "split QKV requires TP4 communicator"
        );
        if precomputed {
            // The paired normalizer/projector already wrote this private output.
        } else if let Some((shard,project_local)) = shard {
            self.qa.run_row_shard(input, shard, project_local,
                comm.ok_or_else(|| anyhow::anyhow!("row gather communicator absent"))?, stream)?;
        } else if let Some((values,scales)) = quantized {
            self.qa.run_quantized(values,scales,stream)?;
        } else {
            self.qa.run(input, stream)?;
        }
        if let Some(output) = self.gathered_qkv.as_mut() {
            if let Some(hint) = &self.qb_prefetch {
                hint.begin(stream)?;
            }
            unsafe {
                comm.unwrap()
                    .gather_bf16(self.qa.output(), output, 448, stream)
            }?;
        }
        if let Some(kv) = &self.kv {
            if matches!(self.mode, InputMode::Shared) {
                let (a, s) = self.qa.quantized_input();
                kv.run_quantized(a, s, stream)?;
            } else {
                kv.run(input, stream)?;
            }
        }
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
        if self.query_norm_quant.is_some() {
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
        if let Some(q) = &self.query_overlap {
            ensure!(q.pending.get() && q.parent == stream.raw() as usize,
                "attention consumer has no matching query fork");
            q.done.wait(stream)?;
            q.pending.set(false);
        }
        if self.woa_prefetch_active {self.woa_prefetch.as_ref().unwrap().begin(stream)?;}
        let query = if matches!(self.qstore, Query::Medium(_)) && crate::config::wide_rows(self.rows)
            && self.rows >= 512 && q_inplace_enabled() { self.qb.output() } else { self.qstore.output() };
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
        match &mut self.woa {
            WoA::Small(w) => {
                w.run(self.decode.output(), stream)?;
                self.ob.run_quantized(w.values(), w.scales(), stream)
            }
            WoA::Medium(w) => {
                w.run(self.decode.output(), stream)?;
                if self.wob_deferrable && (wob_exchange_active() || wob8_active()) { Ok(()) } else { self.ob.run(w.output(), stream) }
            }
        }?;
        // No data dependency: immutable prefetch can finish after its consumer.
        // Join before the next layer so the side stream remains graph-owned.
        if self.woa_prefetch_active {self.woa_prefetch.as_ref().unwrap().join(stream)?;}
        // Read-only hints need no consumer dependency. Match the reference's
        // late join at the next communication boundary rather than stalling QB.
        if let Some(hint) = &self.qb_prefetch {
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
