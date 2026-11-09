//! Rust-owned canonical FFN: BF16 router -> FP32 VL top-k -> MXFP4 experts,
//! and FP8 shared gate/up -> clipped SwiGLU -> FP8 down -> BF16 addition.
//! Six-row TP4 local output, before the collective and mHC residual boundary.
use crate::{
    bf16_linear::Projection,
    blas::Blas,
    blockscaled::Matrix,
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, Event, Module, Stream},
    expert_packing::Prepared,
    fp8_linear::Linear,
    layer_id::LayerId,
    moe::{Expert, Workspace},
    triton_ops::Operator,
    weights::Weights,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use libloading::Library;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::Cell, ffi::c_void, path::Path, rc::Rc};

#[path = "paired_ffn.rs"]
mod paired;
pub(crate) use paired::Paired;

#[derive(Deserialize)]
struct NativeContract {
    abi: u32,
    architecture: u32,
    physical_sms: u32,
    hidden: usize,
    experts: usize,
    router_max_rows: usize,
    router_n_split: usize,
    shared_intermediate: usize,
    swiglu_limit: f32,
    library_sha256: String,
    pdl: bool,
}
type Logits =
    unsafe extern "C" fn(i32, *const c_void, *const c_void, *mut c_void, *mut c_void) -> i32;
type Unary = unsafe extern "C" fn(i32, *const c_void, *mut c_void, *mut c_void) -> i32;
type Add = unsafe extern "C" fn(i32, *const c_void, *const c_void, *mut c_void, *mut c_void) -> i32;
struct Native {
    _lib: Library,
    logits: Logits,
    swiglu: Unary,
    add: Add,
}
impl Native {
    fn open(root: &Path, id: LayerId) -> Result<Self> {
        let m: NativeContract = serde_json::from_slice(&std::fs::read(root.join("native.json"))?)?;
        ensure!(
            (
                m.abi,
                m.architecture,
                m.physical_sms,
                m.hidden,
                m.experts,
                m.router_max_rows,
                m.router_n_split,
                m.shared_intermediate
            ) == (1, 121, 48, 5120, id.experts(), 16, id.router_split(), 576)
                && m.swiglu_limit == 10.
                && !m.pdl,
            "FFN native contract changed"
        );
        let p = root.join("libcanada_ffn.so");
        ensure!(
            format!("{:x}", Sha256::digest(std::fs::read(&p)?)) == m.library_sha256,
            "FFN native hash mismatch"
        );
        let lib = unsafe { Library::new(p) }?;
        let abi: unsafe extern "C" fn() -> i32 = *unsafe { lib.get(b"ca_ffn_abi\0") }?;
        ensure!(unsafe { abi() } == 1, "FFN native ABI mismatch");
        Ok(Self {
            logits: *unsafe { lib.get(b"ca_router_logits\0") }?,
            swiglu: *unsafe { lib.get(b"ca_shared_swiglu\0") }?,
            add: *unsafe { lib.get(b"ca_ffn_add\0") }?,
            _lib: lib,
        })
    }
}
/// Validated scheduler metadata. The image token uses the stored VL correction
/// bias; padded rows return id=-1 and weight=0 from the GPU router.
pub struct TokenRows {
    ids: Buffer<i64>,
    live: Buffer<i32>,
    image_pointers: Buffer<u64>,
    image_owner: Option<Rc<crate::prompt_media::Bindings>>,
    rows: usize,
    active_rows: usize,
}
impl TokenRows {
    pub fn new(cuda: &Cuda, ids: &[i64], live: usize) -> Result<Self> {
        ensure!(!ids.is_empty() && ids.len() <= crate::config::MAX_WIDE_ROWS, "token row capacity");
        let mut s = Self {
            ids: cuda.allocate(ids.len())?,
            live: cuda.allocate(1)?,
            image_pointers: cuda.allocate(ids.len())?,
            image_owner: None,
            rows: ids.len(),
            active_rows: 0,
        };
        s.image_pointers.upload(&vec![0; ids.len()])?;
        s.update(ids, live)?;
        Ok(s)
    }
    pub fn update(&mut self, ids: &[i64], live: usize) -> Result<()> {
        ensure!(
            ids.len() == self.rows
                && live <= self.rows
                && ids.iter().all(|&t| (0..129280).contains(&t)),
            "router token metadata domain"
        );
        self.ids.upload(ids)?;
        self.live.upload(&[live as i32])?;
        self.clear_images()?;
        self.active_rows = live;
        Ok(())
    }
    pub fn ids(&self) -> &Buffer<i64> {
        &self.ids
    }
    /// Caller has synchronized the execution stream before releasing owners.
    pub(crate) fn clear_images(&mut self) -> Result<()> {
        if self.image_owner.is_some() {
            self.image_pointers.upload(&vec![0; self.rows])?;
            self.image_owner = None;
        }
        Ok(())
    }
    pub(crate) fn bind_images(&mut self, owner: Rc<crate::prompt_media::Bindings>, start: usize,
        ids: &[i64], live: usize) -> Result<()> {
        ensure!(ids.len() == self.rows && live == self.active_rows, "media token row shape");
        if let Some(pointers) = owner.row_pointers(start, ids, live)? {
            self.image_pointers.upload(&pointers)?;
            self.image_owner = Some(owner);
        }
        Ok(())
    }
    pub(crate) fn image_pointers(&self) -> &Buffer<u64> { &self.image_pointers }
    pub(crate) fn live(&self) -> &Buffer<i32> {
        &self.live
    }
    pub(crate) fn active_rows(&self) -> usize { self.active_rows }
}
pub struct Router {
    cuda: Cuda,
    _weights: Rc<Weights>,
    native: Rc<Native>,
    route: Operator,
    bias: u64,
    bias_vl: u64,
    rows: usize,
    scores: Scores,
    ids: Buffer<i32>,
    routing: Buffer<f32>,
    stream: Cell<usize>,
}
enum Scores {
    Tiny { weight: u64, output: Buffer<f32> },
    Vendor(Projection<f32>),
}
impl Router {
    fn new(
        cuda: &Cuda,
        weights: &Rc<Weights>,
        id: LayerId,
        rows: usize,
        native: Rc<Native>,
        root: &Path,
        vendor_blas: Option<Rc<Blas>>,
    ) -> Result<Self> {
        ensure!(
            (1..=16).contains(&rows) || crate::config::target_rows(rows),
            "router row capacity"
        );
        let experts = id.experts();
        let topk = id.topk();
        let split_native = match std::env::var("CANADA_SCRATCH_HISTORY32_SPLIT_ROUTER").as_deref() {
            Ok("1") => true,
            Ok("0") | Err(_) => false,
            _ => anyhow::bail!("history32 split router must be 0 or 1"),
        };
        ensure!(!split_native || crate::config::history32_enabled(),
            "split router requires the admitted native32 capability");
        let p = format!("{}.ffn.gate", id.prefix());
        for (kind, dtype, shape) in [
            ("weight", Dtype::Bf16, vec![experts as u64, 5120]),
            ("bias", Dtype::F32, vec![experts as u64]),
        ] {
            let v = weights.view(&format!("{p}.{kind}"))?;
            ensure!(
                v.dtype == dtype && v.shape == shape,
                "router checkpoint shape/dtype mismatch"
            );
        }
        if !id.is_draft() {
            let v = weights.view(&format!("{p}.bias_vl"))?;
            ensure!(
                v.dtype == Dtype::F32 && v.shape == [experts as u64],
                "VL router bias geometry"
            );
        }
        let route = Operator::load(cuda, root, "routing_vl")?;
        let contract: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let entries = contract["entries"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing routing contract"))?;
        ensure!(entries.len() == 1, "unexpected routing primitive count");
        let c = &entries[0]["constexprs"];
        for (name, value) in [
            ("N", serde_json::json!(experts)),
            ("K", serde_json::json!(topk)),
            ("K_ROUTED", serde_json::json!(topk)),
            ("SCORING_FUNC", serde_json::json!(1)),
            ("routed_scaling_factor", serde_json::json!(1.5)),
            ("RENORMALIZE_EPSILON", serde_json::json!(1e-20)),
            ("BIAS_ALT_TOKEN_ID", serde_json::json!(129264)),
            ("RENORMALIZE", serde_json::json!(true)),
            ("APPLY_SCALE", serde_json::json!(true)),
            ("HAS_TOKEN_BIAS", serde_json::json!(!id.is_draft())),
            ("HAS_PADDING", serde_json::json!(true)),
            ("HAS_PACKED", serde_json::json!(false)),
            ("USE_PDL", serde_json::json!(false)),
        ] {
            ensure!(
                c[name] == value,
                "routing numerical contract changed: {name}"
            );
        }
        let spec = [
            ("scores_ptr", "*fp32"),
            ("bias_ptr", "*fp32"),
            ("bias_alt_ptr", "*fp32"),
            ("input_ids_ptr", "*i64"),
            ("num_token_non_padded_ptr", "*i32"),
            ("out_weights_ptr", "*fp32"),
            ("out_indices_ptr", "*i32"),
            ("M", "i32"),
        ];
        ensure!(
            route
                .entry
                .arguments
                .iter()
                .map(|a| (a.name.as_str(), a.kind.as_str()))
                .eq(spec),
            "router ABI changed"
        );
        Ok(Self {
            cuda: cuda.clone(),
            _weights: weights.clone(),
            native,
            route,
            bias: weights.ptr(&format!("{p}.bias"))?,
            bias_vl: weights.ptr(&format!(
                "{p}.{}",
                if id.is_draft() { "bias" } else { "bias_vl" }
            ))?,
            rows,
            scores: if rows <= 16 || (rows==32 && split_native && !id.is_draft()) {
                Scores::Tiny {
                    weight: weights.ptr(&format!("{p}.weight"))?,
                    output: cuda.allocate_rows(rows * experts)?,
                }
            } else {
                Scores::Vendor(Projection::new(
                    cuda,
                    weights.clone(),
                    vendor_blas.ok_or_else(|| {
                        anyhow::anyhow!("medium router requires shared vendor workspace")
                    })?,
                    &p,
                    rows,
                )?)
            },
            ids: cuda.allocate_rows(rows * topk)?,
            routing: cuda.allocate_rows(rows * topk)?,
            stream: Cell::new(0),
        })
    }
    pub fn run(&self, input: &Buffer<bf16>, tokens: &TokenRows, stream: &Stream) -> Result<()> {
        ensure!(
            input.len() == self.rows * 5120
                && tokens.rows == self.rows
                && self.cuda.owns_buffer(input)
                && self.cuda.owns_buffer(&tokens.ids)
                && self.cuda.owns_buffer(&tokens.live)
                && self.cuda.owns_stream(stream),
            "router shape/context mismatch"
        );
        let s = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == s,
            "router needs one ordered stream"
        );
        self.stream.set(s);
        match &self.scores {
            Scores::Tiny { weight, output } => {
              // The native primitive's arithmetic and <=16-row admission stay
              // unchanged; only the finite M32 profile issues a second call.
              let experts=output.len()/self.rows;
              for offset in (0..self.rows).step_by(16) {
                ensure!(
                unsafe {
                    (self.native.logits)(
                        (self.rows-offset).min(16) as i32,
                        (input.ptr()+(offset*5120*2)as u64) as *const c_void,
                        *weight as *const c_void,
                        (output.ptr()+(offset*experts*4)as u64) as *mut c_void,
                        stream.raw(),
                    )
                } == 0,
                "tiny router GPU launch failed"
                );
              }
            },
            Scores::Vendor(p) => p.run(input, stream)?,
        }
        unsafe {
            self.route.launch(
                self.rows as u32,
                &[
                    self.logits().ptr(),
                    self.bias,
                    self.bias_vl,
                    tokens.ids.ptr(),
                    tokens.live.ptr(),
                    self.routing.ptr(),
                    self.ids.ptr(),
                    self.rows as u64,
                ],
                stream,
            )
        }
    }
    pub fn logits(&self) -> &Buffer<f32> {
        match &self.scores {
            Scores::Tiny { output, .. } => output,
            Scores::Vendor(p) => p.output(),
        }
    }
    pub fn ids(&self) -> &Buffer<i32> {
        &self.ids
    }
    pub fn routing(&self) -> &Buffer<f32> {
        &self.routing
    }
}
pub struct LayerFfn {
    cuda: Cuda,
    native: Rc<Native>,
    router: Router,
    expert: Expert,
    shared_up: Linear,
    shared_down: Linear,
    shared_quant: Option<Module>,
    fused_sum: Option<Module>,
    fused_sum_enabled: Cell<bool>,
    activation: Buffer<bf16>,
    output: Buffer<bf16>,
    rows: usize,
    shared_branch: Option<SharedBranch>,
    prefetch: Option<crate::l2_prefetch::Window>,
    expert_prefetch: Option<crate::l2_prefetch::Window>,
    row_transport: Option<RowTransport>,
    use_row_transport: bool,
    local_shared: Option<LocalShared>,
}
/// Sequence-parallel row-local shared expert (CANADA_SCRATCH_PREFILL_LOCAL_SHARED=1):
/// staged full w13 (rank-blocked [gate_r | up_r]) and full-K w2 on this rank's rows.
/// Only the routed partial is reduce-scattered; the local shared output is added
/// afterwards. With CANADA_SCRATCH_PREFILL_LOCAL_SHARED_OVERLAP=1 it runs on a side
/// stream during the input gather and joins before the routed experts.
struct LocalShared {
    up: Linear,
    down: Linear,
    activation: Buffer<bf16>,
    rows: usize,
    branch: Option<SharedBranch>,
    pending: Cell<bool>,
}
static LOCAL_SHARED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
static LOCAL_SHARED_OVERLAP: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
/// CANADA_SCRATCH_PREFILL_LOCAL_SHARED_LATE=1 (with the overlap branch): start the row-local
/// shared expert when the routed partial is ready, so it runs beside the routed reduce-scatter
/// instead of the input gather. Same kernels, inputs and final addition order.
static LOCAL_SHARED_LATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
fn resolved(flag: &std::sync::atomic::AtomicU8, name: &str) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match flag.load(Relaxed) {
        2 => { let on = std::env::var(name).as_deref() == Ok("1"); flag.store(on as u8, Relaxed); on }
        v => v == 1,
    }
}
pub(crate) fn local_shared_constructed() -> bool {
    std::env::var("CANADA_SCRATCH_PREFILL_LOCAL_SHARED").as_deref() == Ok("1")
}
pub(crate) fn set_local_shared_for_probe(enabled: bool) {
    LOCAL_SHARED.store(enabled as u8, std::sync::atomic::Ordering::Relaxed);
}
pub(crate) fn set_local_shared_overlap_for_probe(enabled: bool) {
    LOCAL_SHARED_OVERLAP.store(enabled as u8, std::sync::atomic::Ordering::Relaxed);
}
pub(crate) fn set_local_shared_late_for_probe(enabled: bool) {
    LOCAL_SHARED_LATE.store(enabled as u8, std::sync::atomic::Ordering::Relaxed);
}
type Codec = unsafe extern "C" fn(*const c_void, *const c_void, *const c_void, *mut c_void, i32, i32, *mut c_void) -> i32;
type Decode = unsafe extern "C" fn(*const c_void, *mut c_void, *mut c_void, *mut c_void, i32, i32, *mut c_void) -> i32;
type LocalLive = unsafe extern "C" fn(*const c_void, *mut c_void, i32, i32, *mut c_void) -> i32;
type PackBf16 = unsafe extern "C" fn(*const c_void, *mut c_void, i32, i32, *mut c_void) -> i32;
struct RowTransport {
    _lib: Library,
    encode: Codec,
    decode: Decode,
    local_live: LocalLive,
    pack_bf16: Option<PackBf16>,
    router: Router,
    intermediates: Option<([Buffer<u8>;3],[Buffer<u8>;3])>,
    packed: Buffer<u8>,
    gathered: Buffer<u8>,
    live: Buffer<i32>,
}
struct SharedBranch {
    stream: Rc<Stream>,
    input_ready: Event,
    output_ready: Event,
}
/// Carried wide-verification schedule. The caller admits this only on the
/// Verify16/32 owners, including Verify16's tiny retained-append bank. Bulk prefill
/// profiles retain their existing schedule.
pub(crate) fn wide_overlap_enabled(rows:usize) -> Result<bool> {
    ensure!([16,32].contains(&rows),"wide FFN overlap profile");
    let fenced=std::env::var("CANADA_SCRATCH_DENSE_PIPELINE_FENCE").as_deref()==Ok("1");
    let base=std::env::var("CANADA_SCRATCH_FFN_OVERLAP").as_deref()==Ok("1")
        && crate::mhc::overlap_enabled();
    let flag=if rows==16 {"CANADA_SCRATCH_FFN16_OVERLAP"}else{"CANADA_SCRATCH_FFN32_OVERLAP"};
    let enabled=match std::env::var(flag) {
        Ok(v)=>{ensure!(v=="0"||v=="1","invalid wide FFN overlap policy");v=="1"},
        Err(std::env::VarError::NotPresent)=>fenced&&base,Err(e)=>return Err(e.into()),
    };
    ensure!(!enabled||(fenced&&base),"wide FFN overlap requires fenced dense kernels and base overlap policies");
    Ok(enabled)
}
/// Prefill (>=512-row) shared-expert branch: the existing dual-stream schedule
/// (routed work issued first on the parent, shared expert on an auxiliary stream,
/// joined before the shared addition). Same kernels and buffers; ordering only.
/// Same-allocation A/B control for the prefill branch: false issues the shared
/// expert on its branch only after routed work (serial order, same stream binding).
static PREFILL_SHARED_CONCURRENT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
pub(crate) fn set_prefill_shared_concurrent_for_probe(enabled:bool) {
    PREFILL_SHARED_CONCURRENT.store(enabled,std::sync::atomic::Ordering::Relaxed);
}
fn prefill_shared_concurrent(rows:usize)->bool {
    rows<512 || PREFILL_SHARED_CONCURRENT.load(std::sync::atomic::Ordering::Relaxed)
}
pub(crate) fn prefill_shared_overlap_requested()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_PREFILL_SHARED_OVERLAP") {
        Ok(v)=>{ensure!(v=="0"||v=="1","invalid prefill shared overlap policy");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(false),Err(e)=>Err(e.into()),
    }
}
impl LayerFfn {
    pub(crate) fn enable_prefill_shared_overlap(&mut self,stream:&Stream)->Result<()> {
        ensure!(self.rows>=512&&self.shared_branch.is_none()&&self.cuda.owns_stream(stream),
            "prefill shared-expert overlap admission");
        self.shared_branch=Some(SharedBranch {stream:self.cuda.auxiliary_stream(stream)?,
            input_ready:self.cuda.fence_event()?,output_ready:self.cuda.fence_event()?});
        Ok(())
    }
    /// Routed packed experts on the parent, shared expert on the branch when installed.
    fn run_packed_and_shared(&self,workspace:&Workspace,gathered:&Buffer<u8>,q:&Buffer<u8>,sm:&Buffer<u8>,
        stream:&Stream)->Result<()> {
        if let Some(branch)=&self.shared_branch {
            let concurrent=prefill_shared_concurrent(self.rows);
            if concurrent {branch.input_ready.record(stream)?;}
            workspace.run_packed(&self.expert,gathered,self.router.ids(),self.router.routing(),stream)?;
            if !concurrent {branch.input_ready.record(stream)?;}
            branch.input_ready.wait(&branch.stream)?;
            self.shared_up.run_quantized(q,sm,&branch.stream)?;
            self.finish_shared(&branch.stream)?;
            branch.output_ready.record(&branch.stream)?;
            branch.output_ready.wait(stream)?;
        } else {
            workspace.run_packed(&self.expert,gathered,self.router.ids(),self.router.routing(),stream)?;
            self.shared_up.run_quantized(q,sm,stream)?;
            self.finish_shared(stream)?;
        }
        self.add_shared(workspace,stream)
    }
    pub(crate) fn enable_verify16_overlap_before_capture(&mut self,stream:&Stream)->Result<()> {
        ensure!(wide_overlap_enabled(self.rows)?&&self.shared_branch.is_none()
            &&self.cuda.owns_stream(stream),"uncaptured M16 shared-expert overlap admission");
        self.shared_branch=Some(SharedBranch {stream:self.cuda.auxiliary_stream(stream)?,
            input_ready:self.cuda.fence_event()?,output_ready:self.cuda.fence_event()?});
        Ok(())
    }
    pub fn new(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: Rc<Weights>,
        layout: Prepared,
        kernels: &Path,
    ) -> Result<Self> {
        Self::new_rows(cuda, module, stream, weights, layout, kernels, 6, None)
    }
    pub fn new_rows(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: Rc<Weights>,
        layout: Prepared,
        kernels: &Path,
        rows: usize,
        vendor_blas: Option<Rc<Blas>>,
    ) -> Result<Self> {
        let id = LayerId::from_key(layout.layer)?;
        ensure!(
            (id.is_draft() && rows == 5) || (!id.is_draft() && crate::config::target_rows(rows)),
            "FFN execution profile rows"
        );
        ensure!(
            layout.experts == id.experts(),
            "FFN namespace/expert geometry"
        );
        let native = Rc::new(Native::open(
            &kernels.join(if id.is_draft() {
                "ffn-native-dspark-v1"
            } else {
                "ffn-native-v1"
            }),
            id,
        )?);
        let router = Router::new(
            cuda,
            &weights,
            id,
            rows,
            native.clone(),
            &kernels.join(if id.is_draft() {
                "routing-dspark-v1"
            } else {
                "routing-v1"
            }),
            vendor_blas,
        )?;
        let up = Matrix::prepare_shared_named(cuda, module, stream, &weights, id)?;
        let down = Matrix::prepare(
            cuda,
            module,
            stream,
            &weights,
            &format!("{}.ffn.shared_experts.w2", id.prefix()),
        )?;
        ensure!(
            (down.n, down.k, down.padded_k) == (5120, 576, 640),
            "shared down TP4 geometry mismatch"
        );
        let shared_up = Linear::new(
            cuda,
            &kernels.join(if rows == 6 && !id.is_draft() && crate::fp8_linear::narrow_enabled() {
                "fp8-flashinfer-narrow-w13-m6-v1".into()
            } else { format!("fp8-{}shared-w13-r0-m{rows}-v1", if id.is_draft() { "dspark-" } else { "" }) }),
            up,
        )?;
        let shared_down = Linear::new(
            cuda,
            &kernels.join(format!(
                "fp8-{}shared-w2-r0-m{rows}-v1",
                if id.is_draft() { "dspark-" } else { "" }
            )),
            down,
        )?;
        let shared_quant_requested = match std::env::var("CANADA_SCRATCH_FUSED_SHARED_QUANT") {
            Ok(value) => { ensure!(value == "0" || value == "1", "fused shared quantization flag"); value == "1" },
            Err(std::env::VarError::NotPresent) => false,
            Err(error) => return Err(error.into()),
        };
        let shared_quant = if shared_quant_requested && rows == 6 && !id.is_draft() {
            shared_down.shared_quantization_buffers()?;
            let root = kernels.join("shared-swiglu-quant-v1");
            let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
            let raw=std::fs::read(root.join("shared.cubin"))?;
            ensure!(manifest["architecture"]==121 && manifest["threads"]==128 && manifest["rows"]==6
                && manifest["input_columns"]==1152 && manifest["activation_columns"]==576
                && manifest["padded_columns"]==640 && manifest["minimum_amax"]==0.0001
                && manifest["source_sha256"]=="11852b0f56e224e032e3dcad404a41fc60f726a9e555ae3568961ae14ddd382c"
                && manifest["artifact_sha256"]=="eda8fb6995f7e595c804b9523037ecb58f936f5b1f2a8826a03ae44a59a8d143"
                && format!("{:x}",Sha256::digest(&raw))==manifest["artifact_sha256"].as_str().unwrap_or(""),
                "shared activation/quantizer artifact contract");
            Some(cuda.module(&raw)?)
        } else { None };
        // Decode/verify-sized batches only (as SGLang's dual-stream MoE): wide
        // and 128-row prefill keep the measured sequential schedule.
        let shared_branch = if rows <= 8 && std::env::var("CANADA_SCRATCH_FFN_OVERLAP").as_deref() == Ok("1") {
            Some(SharedBranch {
                stream: cuda.auxiliary_stream(stream)?,
                input_ready: cuda.fence_event()?,
                output_ready: cuda.fence_event()?,
            })
        } else {
            None
        };
        let prefetch = if !id.is_draft() && crate::l2_prefetch::enabled_for_rows(rows,1)? {
            let router_key = format!("{}.ffn.gate.weight", id.prefix());
            let chunk=if rows==6&&crate::config::ffn_fine_hint_requested()? {2048}else{16384};
            Some(crate::l2_prefetch::Window::new(
                cuda,
                module,
                stream,
                vec![shared_up.immutable_matrix(), shared_down.immutable_matrix()],
                Some((weights.clone(), &router_key)),
            )?.with_chunk(chunk)?)
        } else {
            None
        };
        let fused_sum_requested=match std::env::var("CANADA_SCRATCH_FFN_SUM_SHARED") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid fused FFN sum policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
        };
        let fused_sum=if !id.is_draft()&&matches!(rows,6|16|32)
            &&(fused_sum_requested||std::env::var("CANADA_SCRATCH_FFN_SUM_SHARED_SCREEN").as_deref()==Ok("1")) {
            module.function("ca_ffn_sum_shared")?;Some(module.clone())
        }else{None};
        Ok(Self {
            cuda: cuda.clone(),
            native,
            router,
            expert: Expert::new(weights, layout)?,
            shared_up,
            shared_down,
            shared_quant,
            fused_sum,
            fused_sum_enabled:Cell::new(fused_sum_requested),
            activation: cuda.allocate_rows(rows * 576)?,
            output: cuda.allocate_rows(rows * 5120)?,
            rows,
            shared_branch,
            prefetch,
            expert_prefetch: None,
            row_transport: None,
            use_row_transport: false,
            local_shared: None,
        })
    }
    pub(crate) fn enable_row_transport(&mut self, kernels: &Path, vendor_blas: Option<Rc<Blas>>) -> Result<()> {
        ensure!([8192,16384].contains(&self.rows) && self.row_transport.is_none() && self.shared_branch.is_none(),
            "packed MoE transport requires the wide sequential profile");
        self.shared_up.wide_shared_quantization_buffers()?;
        let fast = std::env::var("CANADA_SCRATCH_PREFILL_MOE_CODEC_FAST").as_deref() == Ok("1");
        let root = kernels.join(if fast {"moe-input-codec-fast-v1".to_owned()} else {format!("moe-runtime-m{}-prequant-v1",self.rows)});
        let path = root.join("libcanada_moe_input_codec.so");
        let raw = std::fs::read(&path)?;
        ensure!(format!("{:x}",Sha256::digest(&raw)) == if fast {
            "ef5e336f5fb1a00f02e7910bb7800db724bda14bc2055804b5b7dba0aa5e837c"
        } else {"c6a55fca38606293a79123d72c369c14ef4116bd9f20b45a6350bc25dba2cb23"},
            "packed MoE codec hash mismatch");
        let lib = unsafe {Library::new(path)}?;
        let id = LayerId::from_key(self.expert.layout_layer())?;
        let quant = || -> Result<[Buffer<u8>;3]> { Ok([
            self.cuda.allocate_rows((self.rows/4)*5120)?,self.cuda.allocate_rows((self.rows/4)*160)?,self.cuda.allocate_rows((self.rows/4)*160)?]) };
        self.row_transport = Some(RowTransport {
            encode: *unsafe {lib.get(b"ca_encode\0") }?,
            decode: *unsafe {lib.get(b"ca_decode\0") }?,
            local_live: *unsafe {lib.get(b"ca_local_live\0") }?,
            pack_bf16: if fast {Some(*unsafe {lib.get(b"ca_pack_bf16\0") }?)} else {None},
            _lib: lib,
            router: Router::new(&self.cuda,&self.router._weights,id,self.rows/4,self.native.clone(),
                &kernels.join("routing-v1"),vendor_blas)?,
            intermediates: if fast {None} else {Some((quant()?,quant()?))},
            packed: self.cuda.allocate_rows((self.rows/4)*5920)?,
            gathered: self.cuda.allocate_rows(self.rows*5920)?,live: self.cuda.allocate_rows(1)?,
        });
        self.use_row_transport = std::env::var("CANADA_SCRATCH_PREFILL_MOE_GATHER").as_deref() == Ok("1")
            && std::env::var("CANADA_SCRATCH_PREFILL_MOE_GATHER_AB").as_deref() != Ok("1");
        Ok(())
    }
    pub(crate) fn row_transport_active(&self) -> bool { self.use_row_transport }
    pub(crate) fn set_row_transport(&mut self, enabled: bool) {
        self.use_row_transport = enabled && self.row_transport.is_some();
    }
    pub(crate) fn run_row_transport(&mut self, workspace: &Workspace, input: &Buffer<bf16>,
        tokens: &TokenRows, rank: usize, comm: &crate::roce::Roce, stream: &Stream) -> Result<()> {
        ensure!(self.use_row_transport && [8192,16384].contains(&self.rows) && rank < 4 && tokens.rows == self.rows
            && input.len() == (self.rows/4)*5120 && self.cuda.owns_buffer(input)
            && self.cuda.owns_buffer(&tokens.ids) && self.cuda.owns_buffer(&tokens.live)
            && self.cuda.owns_stream(stream), "packed MoE transport input geometry/context");
        let local_shared = self.local_shared_active();
        let t = self.row_transport.as_mut().context("packed MoE transport absent")?;
        // Both views refer to persistent owners. The graph reads the live count
        // on-device, so a captured partial chunk never freezes a CPU row count.
        let local = TokenRows {
            ids: unsafe {tokens.ids.view(rank*(self.rows/4),self.rows/4)}?,
            image_pointers: unsafe {tokens.image_pointers.view(rank*(self.rows/4),self.rows/4)}?,
            image_owner: tokens.image_owner.clone(),
            live: unsafe {t.live.view(0,1)}?, rows:(self.rows/4),
            active_rows:tokens.active_rows.saturating_sub(rank*(self.rows/4)).min(self.rows/4),
        };
        ensure!(unsafe {(t.local_live)(tokens.live.ptr() as *const c_void,t.live.ptr() as *mut c_void,
            (self.rows/4) as i32,rank as i32,stream.raw())} == 0, "packed MoE local metadata launch");
        t.router.run(input,&local,stream)?;
        if let Some(pack) = t.pack_bf16 {
            ensure!(unsafe {pack(input.ptr() as *const c_void,t.packed.ptr() as *mut c_void,
                (self.rows/4) as i32,5120,stream.raw())} == 0, "fused packed MoE producer launch");
        } else {
            let (routed,shared) = t.intermediates.as_ref().context("packed MoE quantizer storage")?;
            workspace.quantize_packed_shard(false,input,&routed[0],&routed[1],&routed[2],stream)?;
            workspace.quantize_packed_shard(true,input,&shared[0],&shared[1],&shared[2],stream)?;
            ensure!(unsafe {(t.encode)(routed[0].ptr() as *const c_void,routed[1].ptr() as *const c_void,
                shared[0].ptr() as *const c_void,t.packed.ptr() as *mut c_void,(self.rows/4) as i32,5120,stream.raw())} == 0,
                "packed MoE encode launch");
        }
        unsafe {
            comm.gather_rows(&t.packed,&mut t.gathered,stream)?;
            comm.gather_rows(t.router.ids(),&mut self.router.ids,stream)?;
            comm.gather_rows(t.router.routing(),&mut self.router.routing,stream)?;
        }
        if local_shared {
            // Only routed experts consume the gathered rows; the shared expert ran row-locally.
            let gathered=unsafe{t.gathered.view(0,t.gathered.len())}?;
            self.join_local_shared(stream)?;
            return workspace.run_packed(&self.expert,&gathered,self.router.ids(),self.router.routing(),stream);
        }
        let (q,sr,sm) = self.shared_up.wide_shared_quantization_buffers()?;
        ensure!(unsafe {(t.decode)(t.gathered.ptr() as *const c_void,q.ptr() as *mut c_void,
            sr.ptr() as *mut c_void,sm.ptr() as *mut c_void,self.rows as i32,5120,stream.raw())} == 0,
            "packed MoE shared decode launch");
        let gathered=unsafe{t.gathered.view(0,t.gathered.len())}?;
        self.run_packed_and_shared(workspace,&gathered,q,sm,stream)
    }
    pub(crate) fn prefetch_begin(&self, stream: &Stream) -> Result<()> {
        // Queue the small expert hint first. Its early join must not wait for
        // the existing dense-weight hint's deliberately later consumer.
        if let Some(hint)=&self.expert_prefetch {hint.begin(stream)?;}
        if let Some(hint) = &self.prefetch {
            hint.begin(stream)?;
        }
        Ok(())
    }
    pub(crate) fn set_expert_prefetch_for_probe(&mut self,module:&Module,stream:&Stream,
        selected:Option<usize>)->Result<()> {
        ensure!(self.rows==6&&!self.use_row_transport
            &&std::env::var("CANADA_SCRATCH_EXPERT_PREFETCH_SCREEN").as_deref()==Ok("1"),
            "expert hint requires finite ordinary M6 owner");
        self.expert_prefetch=selected.map(|e|self.expert.prefetch_window_for_probe(&self.cuda,module,stream,e)).transpose()?;
        Ok(())
    }
    pub(crate) fn schedule_flags(&self)->(bool,bool) {
        (self.shared_branch.is_some(),self.prefetch.is_some())
    }
    pub(crate) fn prefetch_chunk(&self)->Option<u32> {
        self.prefetch.as_ref().map(crate::l2_prefetch::Window::chunk)
    }
    pub(crate) fn set_wide_hint_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        self.prefetch.as_mut().context("wide FFN hint was not prepared")?
            .set_enabled_for_wide_probe(enabled,stream)
    }
    pub(crate) fn set_prefetch_chunk_for_probe(&mut self,chunk:u32,stream:&Stream)->Result<()> {
        self.prefetch.as_mut().ok_or_else(||anyhow::anyhow!("FFN hint is not installed"))?
            .set_chunk_for_probe(chunk,stream)
    }
    pub(crate) fn set_prefetch_region_for_probe(&mut self,mode:u32,stream:&Stream)->Result<()> {
        self.prefetch.as_mut().ok_or_else(||anyhow::anyhow!("FFN hint is not installed"))?
            .set_region_plan_for_probe(mode,stream)
    }
    pub(crate) fn set_prefetch_budget_for_probe(&mut self,budget:usize,stream:&Stream)->Result<()> {
        self.prefetch.as_mut().ok_or_else(||anyhow::anyhow!("FFN hint is not installed"))?
            .set_budget_for_probe(budget,stream)
    }
    pub(crate) fn prefetch_join(&self, stream: &Stream) -> Result<()> {
        if let Some(hint) = &self.prefetch {
            hint.join(stream)?;
        }
        Ok(())
    }
    pub fn run(
        &self,
        workspace: &Workspace,
        input: &Buffer<bf16>,
        tokens: &TokenRows,
        stream: &Stream,
    ) -> Result<()> {
        self.run_quantized_input(workspace,input,None,tokens,stream)
    }
    pub(crate) fn run_quantized_input(&self,workspace:&Workspace,input:&Buffer<bf16>,
        quantized:Option<(&Buffer<u8>,&Buffer<u8>)>,tokens:&TokenRows,stream:&Stream)->Result<()> {
        // The measured overlap window is the real attention reduction and
        // residual/normalization work. No hint can extend beyond this join.
        if let Some(hint)=&self.expert_prefetch {hint.join(stream)?;}
        // DeepseekV2MoE.forward_normal_dual_stream: fork before routing, issue
        // routed work first, then issue the shared branch. CUDA graph issue
        // order matters even when the dependency DAG is mathematically equal.
        ensure!(
            self.cuda.owns_buffer(workspace.output()),
            "MoE workspace context mismatch"
        );
        if self.local_shared_active() {
            // Routed experts only (no fused shared sum); the shared expert ran row-locally.
            self.router.run(input, tokens, stream)?;
            self.join_local_shared(stream)?;
            workspace.run_live_for_ffn(&self.expert, input, self.router.ids(), self.router.routing(),
                tokens.active_rows(), false, stream)?;
            return Ok(());
        }
        let concurrent=prefill_shared_concurrent(self.rows);
        if let Some(branch) = self.shared_branch.as_ref().filter(|_| concurrent) {
            branch.input_ready.record(stream)?;
        }
        self.router.run(input, tokens, stream)?;
        let deferred=workspace.run_live_for_ffn(
            &self.expert,
            input,
            self.router.ids(),
            self.router.routing(),
            tokens.active_rows(),
            self.fused_sum.is_some()&&self.fused_sum_enabled.get(),
            stream,
        )?;
        if let Some(branch) = &self.shared_branch {
            if !concurrent {branch.input_ready.record(stream)?;}
            branch.input_ready.wait(&branch.stream)?;
            self.run_shared_quantized_input(input,quantized,&branch.stream)?;
            branch.output_ready.record(&branch.stream)?;
            branch.output_ready.wait(stream)?;
        } else {
            self.run_shared_quantized_input(input,quantized,stream)?;
        }
        if let Some((mut routes,live))=deferred {
            let module=self.fused_sum.as_ref().context("fused sum module absent")?;
            let mut shared=self.shared_down.output().ptr();let mut routed=workspace.output().ptr();
            let mut output=self.output.ptr();let mut n=(self.rows*5120)as i32;let mut live=live as i32;
            unsafe{module.function("ca_ffn_sum_shared")?.launch([(n as u32).div_ceil(128),1,1],[128,1,1],0,stream,
                &mut[(&mut routes as *mut u64).cast(),(&mut shared as *mut u64).cast(),
                    (&mut routed as *mut u64).cast(),(&mut output as *mut u64).cast(),
                    (&mut n as *mut i32).cast(),(&mut live as *mut i32).cast()])?;}
            Ok(())
        }else{self.add_shared(workspace,stream)}
    }
    pub(crate) fn set_fused_sum_for_probe(&self,enabled:bool)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_FFN_SUM_SHARED_SCREEN").as_deref()==Ok("1")
            &&self.fused_sum.is_some(),"finite fused FFN sum control");
        self.fused_sum_enabled.set(enabled);Ok(())
    }
    fn run_shared_quantized_input(&self,input:&Buffer<bf16>,
        quantized:Option<(&Buffer<u8>,&Buffer<u8>)>,stream:&Stream)->Result<()> {
        // The compact M16 shared projection retains its own quantizer's
        // minimum-amax semantics. Reuse the fused BF16 normalization here,
        // but do not substitute legacy MXFP8 operands for that quantizer.
        let quantized = quantized.filter(|_| !(self.rows == 16 && self.shared_up.narrow_projection()));
        if let Some((values,scales))=quantized {
            ensure!(matches!(self.rows,6|16) && !self.shared_up.narrow_projection(),
                "prequantized normalized shared input requires the checked legacy quantizer");
            self.shared_up.run_quantized(values,scales,stream)?;
            self.finish_shared(stream)
        } else {self.run_shared(input,stream)}
    }
    fn add_shared(&self, workspace: &Workspace, stream: &Stream) -> Result<()> {
        ensure!(
            self.cuda.owns_buffer(workspace.output()),
            "MoE workspace context mismatch"
        );
        ensure!(
            unsafe {
                (self.native.add)(
                    self.rows as i32,
                    workspace.output().ptr() as *const c_void,
                    self.shared_down.output().ptr() as *const c_void,
                    self.output.ptr() as *mut c_void,
                    stream.raw(),
                )
            } == 0,
            "FFN shared addition launch failed"
        );
        Ok(())
    }
    fn run_shared(&self, input: &Buffer<bf16>, stream: &Stream) -> Result<()> {
        self.shared_up.run(input, stream)?;
        self.finish_shared(stream)
    }
    fn finish_shared(&self, stream: &Stream) -> Result<()> {
        if self.rows==16 && self.shared_quant.is_some() { return self.finish_shared16(stream); }
        if let Some(module)=&self.shared_quant {
            let (values,rowscales,scales)=self.shared_down.shared_quantization_buffers()?;
            let mut x=self.shared_up.output().ptr();let mut a=self.activation.ptr();
            let mut q=values.ptr();let mut sr=rowscales.ptr();let mut sm=scales.ptr();
            let mut rows=self.rows as i32;
            // All storage is owned by this layer, on the same ordered shared
            // stream. Preserve the BF16 activation boundary before MXFP8.
            unsafe {module.function("ca_shared_swiglu_quant128")?.launch([8,1,1],[128,1,1],0,stream,
                &mut [(&mut x as *mut u64).cast(),(&mut a as *mut u64).cast(),
                    (&mut q as *mut u64).cast(),(&mut sr as *mut u64).cast(),
                    (&mut sm as *mut u64).cast(),(&mut rows as *mut i32).cast()])?;}
            return self.shared_down.run_quantized(values,scales,stream);
        }
        ensure!(
            unsafe {
                (self.native.swiglu)(
                    self.rows as i32,
                    self.shared_up.output().ptr() as *const c_void,
                    self.activation.ptr() as *mut c_void,
                    stream.raw(),
                )
            } == 0,
            "shared SwiGLU launch failed"
        );
        self.shared_down.run(&self.activation, stream)?;
        Ok(())
    }
    pub fn output(&self) -> &Buffer<bf16> {
        &self.output
    }
    /// Owned per-layer inputs/results survive a completed verification. The
    /// routed-expert workspace is shared across layers and is deliberately
    /// absent here; these taps distinguish its path from routing/shared work.
    pub(crate) fn repeat_owned_sources(&self,live:usize)->Result<Vec<(&'static str,u64,usize)>> {
        ensure!(self.rows==6&&(2..=6).contains(&live)
            &&std::env::var("CANADA_SCRATCH_SAMPLER_READOUT_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_READOUT_TRACE_PEER_DETAIL").as_deref()==Ok("1"),
            "detailed FFN tracing requires the finite repeat screen");
        let Scores::Tiny{output:scores,..}=&self.router.scores else{anyhow::bail!("repeat FFN router profile")};
        ensure!(self.shared_up.output().len()==6*1152&&self.activation.len()==6*576
            &&self.shared_down.output().len()==6*5120&&self.router.ids.len()==6*6
            &&self.router.routing.len()==6*6&&scores.len()==6*384,"repeat FFN source geometry");
        Ok(vec![
            ("router-scores",scores.ptr(),live*384*4),
            ("router-ids",self.router.ids.ptr(),live*6*4),
            ("router-weights",self.router.routing.ptr(),live*6*4),
            ("shared-up",self.shared_up.output().ptr(),live*1152*2),
            ("shared-activation",self.activation.ptr(),live*576*2),
            ("shared-down",self.shared_down.output().ptr(),live*5120*2),
        ])
    }
    pub fn router(&self) -> &Router {
        &self.router
    }
    pub(crate) fn projection_quantized_input(&self) -> (&Buffer<u8>,&Buffer<u8>) { self.shared_up.quantized_input() }
    pub(crate) fn narrow_projection(&self) -> bool { self.shared_up.narrow_projection() }
    pub(crate) fn fused_shared_quantization(&self) -> bool { self.shared_quant.is_some() }
    pub fn shared_up(&self) -> &Buffer<bf16> {
        self.shared_up.output()
    }
    pub fn activation(&self) -> &Buffer<bf16> {
        &self.activation
    }
    pub fn shared_down(&self) -> &Buffer<bf16> {
        self.shared_down.output()
    }
}

impl LayerFfn {
    pub(crate) fn prefill_router_for_probe(&self,kernels:&Path,blas:Rc<Blas>)->Result<Router> {
        ensure!(self.rows==8192 && self.expert.layout_layer()==0
            && std::env::var("CANADA_SCRATCH_PREFILL_ROUTER_OVERLAP_SCREEN").as_deref()==Ok("1"),
            "local router construction is restricted to the isolated layer0 probe");
        Router::new(&self.cuda,&self.router._weights,LayerId::target(0)?,self.rows/4,
            self.native.clone(),&kernels.join("routing-v1"),Some(blas))
    }
}

pub(crate) fn wide_overlap_probe(rows:usize)->bool {
    (rows==16&&std::env::var("CANADA_SCRATCH_VERIFY16_OVERLAP_SCREEN").as_deref()==Ok("1"))
        ||(rows==32&&(std::env::var("CANADA_SCRATCH_FFN32_REFRESH_SCREEN").as_deref()==Ok("1")
                ||std::env::var("CANADA_SCRATCH_FFN32_GENERATION_SCREEN").as_deref()==Ok("1"))
            &&std::env::var("CANADA_SCRATCH_DENSE_PIPELINE_FENCE").as_deref()==Ok("1"))
}
// Isolated same-load comparisons against the serial wide schedule.
impl LayerFfn {
    pub(crate) fn enable_verify16_overlap_for_probe(&mut self, stream: &Stream) -> Result<()> {
        ensure!(wide_overlap_probe(self.rows)
            && self.shared_branch.is_none() && self.cuda.owns_stream(stream),
            "isolated wide shared-expert overlap admission");
        self.shared_branch=Some(SharedBranch { stream:self.cuda.auxiliary_stream(stream)?,
            input_ready:self.cuda.fence_event()?,output_ready:self.cuda.fence_event()? });
        Ok(())
    }
    pub(crate) fn set_verify16_overlap_for_probe(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!(wide_overlap_probe(self.rows)
            && self.cuda.owns_stream(stream), "idle wide FFN schedule screen");
        // Target::set_verify16_overlap_for_probe joins and destroys all of this
        // profile's old graphs before changing any layer's event schedule.
        unsafe {
            self.shared_up.release_verify16_probe_binding()?;
            self.shared_down.release_verify16_probe_binding()?;
        }
        if enabled && self.shared_branch.is_none() { self.enable_verify16_overlap_for_probe(stream)?; }
        if !enabled { self.shared_branch=None; }
        Ok(())
    }
}

impl LayerFfn {
    pub(crate) fn enable_shared_quant16(&mut self,kernels:&Path)->Result<()> {
        ensure!(self.rows==16 && self.shared_quant.is_none()
            && shared_quant16_enabled()?, "M16 shared quantization policy");
        self.shared_down.shared_quantization_buffers16()?;
        let root=kernels.join("shared-swiglu-quant16-v1");
        let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let raw=std::fs::read(root.join("shared.cubin"))?;
        ensure!(manifest["architecture"]==121 && manifest["rows"]==16 && manifest["threads"]==64
            && manifest["grid"]==40 && manifest["padding_scale_exponent"]==127
            && manifest["source_sha256"]=="2c7d06a4b86e2c2aea53e11a8645c161d4f3ebbb944df1ae79c78d2bcee2ca11"
            && manifest["artifact_sha256"]=="dddf8b6ae39a3f092cc68ec9b34e8c9e47884ea91dbbd6fc18494192a04b6692"
            && format!("{:x}",Sha256::digest(&raw))==manifest["artifact_sha256"].as_str().unwrap_or(""),
            "M16 shared activation/quantization artifact contract");
        self.shared_quant=Some(self.cuda.module(&raw)?);Ok(())
    }
    fn finish_shared16(&self,stream:&Stream)->Result<()> {
        let module=self.shared_quant.as_ref().context("M16 shared fusion module")?;
        let(values,row_scales,scales)=self.shared_down.shared_quantization_buffers16()?;
        let mut x=self.shared_up.output().ptr();let mut a=self.activation.ptr();
        let mut q=values.ptr();let mut sr=row_scales.ptr();let mut sm=scales.ptr();let mut rows=16i32;
        unsafe {module.function("ca_shared_swiglu_quant16_64")?.launch([40,1,1],[64,1,1],0,stream,
            &mut[(&mut x as *mut u64).cast(),(&mut a as *mut u64).cast(),(&mut q as *mut u64).cast(),
                (&mut sr as *mut u64).cast(),(&mut sm as *mut u64).cast(),(&mut rows as *mut i32).cast()])?;}
        self.shared_down.run_quantized(values,scales,stream)
    }
}

/// The ordinary candidate's shared-activation fusion also covers M16 after
/// s669's exact operand checks and s673's full-model state/handoff screen.
/// Keep a separate override for paired investigations.
pub(crate) fn shared_quant16_enabled() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_FUSED_SHARED_QUANT16") {
        Ok(value) => {
            ensure!(value == "0" || value == "1", "invalid M16 shared quantization policy");
            Ok(value == "1")
        }
        Err(std::env::VarError::NotPresent) =>
            Ok(std::env::var("CANADA_SCRATCH_FUSED_SHARED_QUANT").as_deref() == Ok("1")),
        Err(error) => Err(error.into()),
    }
}

impl LayerFfn {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enable_local_shared(&mut self, cuda: &Cuda, module: &Module, stream: &Stream,
        weights: &Rc<Weights>, kernels: &Path, layer: usize, rank: usize) -> Result<()> {
        ensure!(crate::config::wide_rows(self.rows) && self.rows % 512 == 0 && rank < 4 && self.local_shared.is_none(),
            "row-local shared expert geometry");
        let local = self.rows / 4;
        let p = format!("layers.{layer}.ffn.shared_experts");
        let up = crate::staged_full::load_parts(cuda, module, stream, weights, kernels, "claude-shared-full-v1",
            Some("w13"), &[&format!("{p}.w1"), &format!("{p}.w3")], layer, rank, 4608, 5120, crate::staged_full::Split::Rows)?;
        let down = crate::staged_full::load_parts(cuda, module, stream, weights, kernels, "claude-shared-full-v1",
            Some("w2"), &[&format!("{p}.w2")], layer, rank, 5120, 2304, crate::staged_full::Split::Columns)?;
        let overlap = std::env::var("CANADA_SCRATCH_PREFILL_LOCAL_SHARED_OVERLAP").as_deref() == Ok("1");
        self.local_shared = Some(LocalShared {
            up: Linear::new_rc(cuda, &kernels.join(format!("claude-fp8-shared-w13-full-m{local}-v1")), up)?,
            down: Linear::new_rc(cuda, &kernels.join(format!("claude-fp8-shared-w2-full-m{local}-v1")), down)?,
            activation: cuda.allocate_rows(local * 2304)?,
            rows: local,
            branch: if overlap { Some(SharedBranch { stream: cuda.auxiliary_stream(stream)?,
                input_ready: cuda.fence_event()?, output_ready: cuda.fence_event()? }) } else { None },
            pending: Cell::new(false),
        });
        Ok(())
    }
    /// out = a + b over `rows` BF16 rows of width 5120 (out may alias a).
    pub(crate) fn add_rows_bf16(&self, a: &Buffer<bf16>, b: &Buffer<bf16>, out: &Buffer<bf16>,
        rows: usize, stream: &Stream) -> Result<()> {
        ensure!(a.len() >= rows * 5120 && b.len() >= rows * 5120 && out.len() >= rows * 5120
            && self.cuda.owns_buffer(a) && self.cuda.owns_buffer(b) && self.cuda.owns_buffer(out), "BF16 row add extent");
        ensure!(unsafe { (self.native.add)(rows as i32, a.ptr() as *const c_void, b.ptr() as *const c_void,
            out.ptr() as *mut c_void, stream.raw()) } == 0, "BF16 row add launch");
        Ok(())
    }
    pub(crate) fn local_shared_active(&self) -> bool {
        self.local_shared.is_some() && resolved(&LOCAL_SHARED, "CANADA_SCRATCH_PREFILL_LOCAL_SHARED")
    }
    /// Late start applies only to the overlapped side-stream branch.
    pub(crate) fn local_shared_late(&self) -> bool {
        self.local_shared_active()
            && self.local_shared.as_ref().is_some_and(|s| s.branch.is_some())
            && resolved(&LOCAL_SHARED_OVERLAP, "CANADA_SCRATCH_PREFILL_LOCAL_SHARED_OVERLAP")
            && resolved(&LOCAL_SHARED_LATE, "CANADA_SCRATCH_PREFILL_LOCAL_SHARED_LATE")
    }
    /// Start the row-local shared expert from this rank's normalized FFN rows.
    pub(crate) fn begin_local_shared(&self, input: &Buffer<bf16>, stream: &Stream) -> Result<()> {
        let s = self.local_shared.as_ref().context("row-local shared expert absent")?;
        ensure!(input.len() == s.rows * 5120 && !s.pending.get() && self.cuda.owns_buffer(input),
            "row-local shared input/state");
        let run = |st: &Stream| -> Result<()> {
            s.up.run(input, st)?;
            // Rank-blocked [gate_r | up_r] rows of width 1152: the per-shard SwiGLU over
            // 4x rows writes the activation in w2's K order.
            ensure!(unsafe { (self.native.swiglu)((s.rows * 4) as i32, s.up.output().ptr() as *const c_void,
                s.activation.ptr() as *mut c_void, st.raw()) } == 0, "row-local shared SwiGLU launch");
            s.down.run(&s.activation, st)
        };
        // Linears bind to one ordered stream: with a branch, always run there; the
        // serial selection joins immediately instead of before the routed experts.
        match s.branch.as_ref() {
            Some(b) => {
                b.input_ready.record(stream)?;
                b.input_ready.wait(&b.stream)?;
                run(&b.stream)?;
                b.output_ready.record(&b.stream)?;
                s.pending.set(true);
                if !resolved(&LOCAL_SHARED_OVERLAP, "CANADA_SCRATCH_PREFILL_LOCAL_SHARED_OVERLAP") {
                    self.join_local_shared(stream)?;
                }
            }
            None => run(stream)?,
        }
        Ok(())
    }
    pub(crate) fn join_local_shared(&self, stream: &Stream) -> Result<()> {
        if let Some(s) = self.local_shared.as_ref().filter(|s| s.pending.get()) {
            s.branch.as_ref().context("local shared branch")?.output_ready.wait(stream)?;
            s.pending.set(false);
        }
        Ok(())
    }
    /// `reduced` (this rank's rows of the routed sum) += the row-local shared output.
    pub(crate) fn add_local_shared(&self, reduced: &mut Buffer<bf16>, stream: &Stream) -> Result<()> {
        self.join_local_shared(stream)?;
        let s = self.local_shared.as_ref().context("row-local shared expert absent")?;
        ensure!(reduced.len() == s.rows * 5120 && self.cuda.owns_buffer(reduced), "row-local shared sum extent");
        ensure!(unsafe { (self.native.add)(s.rows as i32, reduced.ptr() as *const c_void,
            s.down.output().ptr() as *const c_void, reduced.ptr() as *mut c_void, stream.raw()) } == 0,
            "row-local shared addition launch");
        Ok(())
    }
    /// Caller must hold the idle target and forbid switching captured graphs.
    /// Both libraries retain the exact packed row layout and quantizers.
    pub(crate) fn set_coalesced_codec(&mut self, enabled: bool, kernels: &Path) -> Result<()> {
        let Some(t)=self.row_transport.as_mut() else {return Ok(())};
        ensure!(t.pack_bf16.is_some() && t.intermediates.is_none(),"coalesced codec requires fused BF16 producer");
        let (name,hash)=if enabled {
            ("moe-input-codec-coalesced-v1","079b3970a6c444cd06df3a48896ef2e4a2c558bdfcd3a88b0197da4e6f8875e9")
        } else {
            ("moe-input-codec-fast-v1","ef5e336f5fb1a00f02e7910bb7800db724bda14bc2055804b5b7dba0aa5e837c")
        };
        let path=kernels.join(name).join("libcanada_moe_input_codec.so");
        ensure!(format!("{:x}",Sha256::digest(std::fs::read(&path)?))==hash,"coalesced codec artifact hash");
        let lib=unsafe {Library::new(path)}?;
        // Resolve everything before replacing the owner of the old symbols.
        let encode=*unsafe {lib.get::<Codec>(b"ca_encode\0")}?;
        let local_live=*unsafe {lib.get::<LocalLive>(b"ca_local_live\0")}?;
        let decode=*unsafe {lib.get::<Decode>(if enabled {b"ca_decode_registered\0"}else{b"ca_decode\0"})}?;
        let pack=*unsafe {lib.get::<PackBf16>(if enabled {b"ca_pack_registered\0"}else{b"ca_pack_bf16\0"})}?;
        t.encode=encode;t.local_live=local_live;t.decode=decode;t.pack_bf16=Some(pack);t._lib=lib;
        Ok(())
    }
    pub(crate) fn coalesced_codec_active(&self)->bool {
        self.use_row_transport && self.row_transport.as_ref().is_some_and(|t| {
            let registered=unsafe {t._lib.get::<PackBf16>(b"ca_pack_registered\0")};
            registered.is_ok_and(|symbol|t.pack_bf16.is_some_and(|pack|pack as usize==*symbol as usize))
        })
    }
}

impl LayerFfn {
    pub(crate) fn pipeline_eligible(&self)->bool {self.rows==8192&&self.use_row_transport}
    pub(crate) fn run_pipelined_row_transport(&mut self,workspace:&Workspace,input:&Buffer<bf16>,
        tokens:&TokenRows,rank:usize,comm:&crate::roce::Roce,stream:&Stream)->Result<()> {
        ensure!(self.pipeline_eligible()&&rank<4&&tokens.rows==8192&&input.len()==2048*5120
            &&self.cuda.owns_buffer(input)&&self.cuda.owns_buffer(&tokens.ids)&&self.cuda.owns_buffer(&tokens.live)
            &&self.cuda.owns_stream(stream),"pipelined MoE input geometry/context");
        let t=self.row_transport.as_mut().context("pipelined MoE storage absent")?;
        ensure!(t.pack_bf16.is_some()&&t.intermediates.is_none(),"pipeline requires fused dual-row input contract");
        let local=TokenRows{ids:unsafe{tokens.ids.view(rank*2048,2048)}?,live:unsafe{t.live.view(0,1)}?,
            image_pointers:unsafe{tokens.image_pointers.view(rank*2048,2048)}?,image_owner:tokens.image_owner.clone(),
            rows:2048,active_rows:tokens.active_rows.saturating_sub(rank*2048).min(2048)};
        ensure!(unsafe{(t.local_live)(tokens.live.ptr()as *const c_void,t.live.ptr()as *mut c_void,
            2048,rank as i32,stream.raw())}==0,"pipeline local metadata launch");
        t.router.run(input,&local,stream)?;
        let(q,sr,sm)=self.shared_up.wide_shared_quantization_buffers()?;
        comm.pipeline_moe_input(input,&t.gathered,q,sr,sm,stream,||unsafe{
            comm.gather_rows(t.router.ids(),&mut self.router.ids,stream)?;
            comm.gather_rows(t.router.routing(),&mut self.router.routing,stream)?;
            Ok(())
        })?;
        let gathered=unsafe{t.gathered.view(0,t.gathered.len())}?;
        self.run_packed_and_shared(workspace,&gathered,q,sm,stream)
    }
}

impl LayerFfn {
    pub(crate) fn set_shared_down_flash_for_probe(&self,enabled:bool)->Result<()> {
        self.shared_down.set_shared_down_flash_for_probe(enabled)
    }
    pub(crate) fn legacy_shared_up_for_probe(&self,kernels:&Path)->Result<Linear> {
        self.shared_up.legacy_shared_up_for_probe(kernels)
    }
    pub(crate) fn swap_shared_up_for_probe(&mut self,other:&mut Linear)->Result<()> {
        // The target has joined auxiliary work and dropped all captured graphs.
        unsafe {
            self.shared_up.release_verify16_probe_binding()?;
            other.release_verify16_probe_binding()?;
        }
        self.shared_up.swap_shared_up_for_probe(other)
    }
}
