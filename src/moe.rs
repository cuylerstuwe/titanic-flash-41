//! Rust-owned routed-expert execution and shared per-geometry workspace.
//! The AOT dependency describes only a GPU-kernel ABI and scratch byte layout.
//! Model/layer ownership, loading, scheduling and graph construction are Rust.
use crate::{
    cuda::{Buffer, Cuda, Stream},
    expert_packing::Prepared,
    primitives::Primitives,
    weights::Weights,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde::Deserialize;
use std::{cell::Cell, collections::BTreeMap, path::Path, rc::Rc};
#[cfg(feature="storage-fault-injection")]
thread_local! { static PREFILL_WIDE_BOUND_CALLS: Cell<usize> = const { Cell::new(0) }; }
pub(crate) fn prefill_wide_bound_calls()->usize {
    #[cfg(feature="storage-fault-injection")]
    { PREFILL_WIDE_BOUND_CALLS.with(Cell::get) }
    #[cfg(not(feature="storage-fault-injection"))]
    { 0 }
}
#[derive(Deserialize, PartialEq)]
struct Pointer {
    role: String,
    offset: usize,
}
#[derive(Deserialize)]
struct Node {
    entry: i32,
    kernel_id: String,
    pointers: BTreeMap<String, Pointer>,
    scalars: BTreeMap<String, i32>,
}
#[derive(Deserialize)]
struct Contract {
    abi: u32,
    rows: usize,
    experts: usize,
    hidden: usize,
    intermediate: usize,
    topk: usize,
    deterministic: bool,
    swiglu_limit: f32,
    scratch_bytes: usize,
    scratch_initializer: String,
    kernel_library_sha256: String,
    nodes: Vec<Node>,
    #[serde(default)]
    projection_dtype: Option<String>,
    #[serde(default)]
    micro_fp32_projection_overlay: Option<serde_json::Value>,
    #[serde(default)]
    fc1_scale_reuse_overlay: Option<serde_json::Value>,
    #[serde(default)]
    task_order_overlay: Option<serde_json::Value>,
    #[serde(default)]
    input_format: Option<String>,
    #[serde(default)]
    fc1_input_address_overlay: Option<serde_json::Value>,
    #[serde(default)]
    input_bytes: usize,
    #[serde(default)]
    routed_quantizer_entry: Option<usize>,
    #[serde(default)]
    prequantized_input_overlay: Option<serde_json::Value>,
}
struct PackedInput {
    primitives: Primitives,
    contract: Contract,
}
pub(crate) fn packed_gather_requested() -> Result<bool> {
    let enabled = match std::env::var("CANADA_SCRATCH_PREFILL_MOE_GATHER") {
        Ok(v) => { ensure!(v == "0" || v == "1", "invalid MoE input gather policy"); v == "1" },
        Err(std::env::VarError::NotPresent) => false,
        Err(e) => return Err(e.into()),
    };
    Ok(enabled || std::env::var("CANADA_SCRATCH_PREFILL_MOE_GATHER_AB").as_deref() == Ok("1"))
}
pub(crate) fn wide_transfer_requested()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_MOE_WIDE_TRANSFER") {
        Ok(v)=>{ensure!(v=="0"||v=="1","invalid wide expert transfer policy");Ok(v=="1")},
        // M16/M128 code-only transfer passed component and same-allocation
        // full-model ABBA screens. Resolve once while constructing the owner;
        // captured graphs keep the chosen kernels with no runtime flag lookup.
        Err(std::env::VarError::NotPresent)=>Ok(true),Err(e)=>Err(e.into()),
    }
}
pub(crate) fn prefill_transfer_requested()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_MOE_PREFILL_TRANSFER") {
        Ok(v)=>{ensure!(v=="0"||v=="1","invalid prefill expert transfer policy");Ok(v=="1")},
        // s1320 full512/1024-row chunks saved0.9/0.5%; partial512 regressed.
        // The partial1024 matched-input mean saved0.19%. Keep partial512 on
        // its original kernel; the pooled median across two input patterns
        // hides the small same-direction benefit for partial1024.
        Err(std::env::VarError::NotPresent)=>Ok(true),Err(e)=>Err(e.into()),
    }
}
pub struct Expert {
    weights: Rc<Weights>,
    layout: Prepared,
}
impl Expert {
    pub(crate) fn prefetch_window_for_probe(&self,cuda:&Cuda,module:&crate::cuda::Module,
        stream:&Stream,expert:usize)->Result<crate::l2_prefetch::Window> {
        ensure!(self.layout.experts==384&&expert<384,"canonical expert hint index");
        let mut regions=Vec::new();
        for role in ["w13","s13"] {
            let(base,bytes)=self.region(role)?;
            ensure!(bytes%384==0,"expert hint stride");
            let stride=bytes/384;
            regions.push((base+u64::try_from(expert*stride)?,stride));
        }
        crate::l2_prefetch::Window::new_expert(cuda,module,stream,self.weights.clone(),&regions)
    }
    pub(crate) fn layout_layer(&self) -> usize { self.layout.layer }
    pub(crate) fn same_layer_weights(&self,other:&Self)->bool {
        self.layout.layer==other.layout.layer && Rc::ptr_eq(&self.weights,&other.weights)
    }
    pub fn new(weights: Rc<Weights>, layout: Prepared) -> Result<Self> {
        ensure!(
            weights.expert_ready(layout.layer),
            "expert layer is not ready"
        );
        ensure!(
            (layout.experts, layout.hidden, layout.intermediate)
                == (
                    crate::layer_id::LayerId::from_key(layout.layer)?.experts(),
                    5120,
                    576
                ),
            "expert geometry mismatch"
        );
        Ok(Self { weights, layout })
    }
    fn region(&self, role: &str) -> Result<(u64, usize)> {
        let l = &self.layout;
        let wb = l.hidden * l.intermediate / 2;
        let sb = l.hidden * l.intermediate / 32;
        let (offset, bytes) = match role {
            "w13" => (l.w13_offset, 2 * l.experts * wb),
            "w2" => (l.w2_offset, l.experts * wb),
            "s13" => (l.s13_offset, 2 * l.experts * sb),
            "s2" => (l.s2_offset, l.experts * sb),
            _ => anyhow::bail!("unknown expert operand"),
        };
        Ok((self.weights.pointer_at(offset, bytes)?, bytes))
    }
}
pub struct Workspace {
    cuda: Cuda,
    primitives: Primitives,
    contract: Contract,
    scratch: Rc<Buffer<u8>>,
    sequential_scratch: Option<Rc<SequentialScratch>>,
    ones: Buffer<f32>,
    output: Rc<Buffer<bf16>>,
    live_bounds: bool,
    single_row_micro: Option<Box<Workspace>>,
    three_row_task_order: Option<Box<Workspace>>,
    packed: Option<PackedInput>,
    input_address: Option<PackedInput>,
    input_address_enabled: Cell<bool>,
    scale_codes: Option<PackedInput>,
    scale_codes_enabled: Cell<bool>,
    fc_pdl: Option<PackedInput>,
    late_fc_pdl: Option<PackedInput>,
    late_fc_pdl_enabled: Cell<bool>,
    cache_ca: Option<PackedInput>,
    cache_ca_enabled: Cell<bool>,
    cooperative_epilogues: [Option<PackedInput>; 2],
    // 0: original; 1: six-row shared quad; 2: both experimental variants.
    // s1285 retained only the six-row variant; three-row register conversion lost.
    cooperative_epilogue_mode: Cell<u8>,
    // 0: ordinary input addressing; 1: all live rows (finite screen);
    // 3: the measured three-row-only policy.
    fc_pdl_mode: Cell<u8>,
    wide_transfer: Option<Primitives>,
    wide_transfer_enabled: Cell<bool>,
    stream: Cell<Option<usize>>,
}
/// Only the complete, sequential wide-prefill bank may lend this storage.
/// Different widths have different counter offsets. A width handoff clears
/// the destination layout before capture/replay; same-width layer reuse keeps
/// the already qualified kernel initialization behavior.
struct SequentialScratch {
    rows: Cell<usize>,
    stream: Cell<usize>,
}
impl Workspace {
    pub fn new(cuda: &Cuda, root: &Path) -> Result<Self> {
        Self::new_in(cuda,root,None,false)
    }
    pub(crate) fn new_sequential(cuda:&Cuda,root:&Path,donor:Option<&Self>)->Result<Self> {
        ensure!(std::env::var("CANADA_SCRATCH_WIDE_MOE_ARENA_SCREEN").as_deref()==Ok("1"),
            "sequential MoE scratch experiment admission");
        Self::new_in(cuda,root,donor,true)
    }
    fn new_in(cuda:&Cuda,root:&Path,donor:Option<&Self>,sequential:bool)->Result<Self> {
        let c: Contract =
            serde_json::from_slice(&std::fs::read(root.join("runtime-contract.json"))?)?;
        ensure!(c.input_format.is_none(), "BF16 MoE workspace cannot consume a packed-input contract");
        ensure!(
            c.abi == 1
                && c.hidden == 5120
                && c.intermediate == 576
                && (((c.experts, c.topk) == (384, 6) && crate::config::target_rows(c.rows))
                    || ((c.experts, c.topk, c.rows) == (128, 3, 5)))
                && c.deterministic
                && c.swiglu_limit == 10.,
            "unsupported MoE precision/geometry contract"
        );
        ensure!(
            c.scratch_initializer == "zero"
                && c.scratch_bytes > 0
                // The wide plan's grouped M64 scratch is one shared ~530 MiB
                // arena per profile, not one per layer.
                && c.scratch_bytes
                    <= if c.rows == 16384 {
                        2 * 1024 * 1024 * 1024
                    } else if crate::config::wide_rows(c.rows) {
                        1024 * 1024 * 1024
                    } else {
                        64 * 1024 * 1024
                    },
            "unsupported MoE scratch contract"
        );
        let primitives = Primitives::open(
            cuda,
            &root.join("libcanada_moe_primitives.so"),
            &root.join("bindings.json"),
        )?;
        ensure!(
            primitives.bindings.abi == 1
                && primitives.bindings.library_sha256 == c.kernel_library_sha256,
            "MoE AOT contract hash/ABI mismatch"
        );
        let sequence: Vec<_> = c.nodes.iter().map(|n| n.kernel_id.as_str()).collect();
        let micro = sequence == ["gemm.mxfp8_quant_cute", "moe.w4a8.compact_micro", "moe.w4a16.topk_sum"];
        ensure!(
            sequence == ["integration.tp_moe.dynamic", "moe.w4a16.topk_sum"] || micro,
            "unsupported MoE kernel sequence"
        );
        if micro {
            let patch = c.micro_fp32_projection_overlay.as_ref()
                .context("micro kernel requires its FP32 projection provenance")?;
            ensure!((c.rows,c.experts,c.topk)==(6,384,6)
                && c.projection_dtype.as_deref()==Some("float32")
                && patch["base_commit"]=="a7d7d29b2ef8869086e0ceaa787321f17544e3c9"
                && patch["files"]["w4a8_compact_micro.py"]["patched_sha256"]=="a84b7d0e9b0596107b424dd1c69d5cebbd2604c0c9bb1ef55108a628da93ae3b"
                && patch["files"]["w4a8_compact_projection.py"]["patched_sha256"]=="0872002b3de07dca54a1bed049027920015f1e2c5f3fd59b45ea1436966c89f5",
                "unqualified micro projection contract");
        }
        if let Some(patch)=&c.fc1_scale_reuse_overlay {
            ensure!(!micro && c.rows==6 && c.experts==384 && c.topk==6 && c.scratch_bytes==4118728
                && c.kernel_library_sha256=="9137d340b5c5042245caea9ad77872e63073e0bf8ef3bac6d20d2785f5c563d2"
                && patch["base_commit"]=="a7d7d29b2ef8869086e0ceaa787321f17544e3c9"
                && patch["reuse_fc1_weight_scales"]==true && patch["stages"]==serde_json::json!([2,2])
                && patch["files"]["w4a8_phase1.py"]["patched_sha256"]=="d932a13b03170b73504a8b34e52612f0e39aa1e07745a42ceea6b018e7f67572"
                && patch["files"]["w4a8_phase2.py"]["patched_sha256"]=="0c9fce39b243676e6754ea1fe111f765b5a993b6d6946ac410301a4663d1c5c3",
                "unqualified FC1 scale-reuse contract");
        }
        if let Some(patch) = &c.task_order_overlay {
            ensure!(!micro && c.fc1_scale_reuse_overlay.is_none()
                && (c.rows, c.experts, c.topk, c.scratch_bytes) == (6, 384, 6, 4118728)
                && c.kernel_library_sha256 == "36b502a2cd3e13f4b1a340b2494d18ddd4afe0260b869929aa2bc275ffa24003"
                && patch["base_commit"] == "a7d7d29b2ef8869086e0ceaa787321f17544e3c9"
                && patch["output_major_tasks"] == true && patch["reuse_fc1_weight_scales"] == false
                && patch["stages"] == serde_json::json!([2,2])
                && patch["files"]["w4a8_phase1.py"]["patched_sha256"] == "c293321bbe32619162413287d9c83d0504cd006734d47da719767acb61eee0a3"
                && patch["files"]["w4a8_phase2.py"]["patched_sha256"] == "9c6c8e813fa5755c3b7ed84910d700407a93da24ec7794dea4e363fd2d9f0e10",
                "unqualified expert task-order contract");
        }
        for node in &c.nodes {
            let e = primitives
                .bindings
                .entries
                .get(node.entry as usize)
                .context("missing MoE primitive")?;
            ensure!(
                e.kernel_id == node.kernel_id
                    && e.pointers.len() == node.pointers.len()
                    && e.i32_scalars.len() == node.scalars.len(),
                "MoE ABI parameter drift"
            );
            for name in &e.pointers {
                ensure!(node.pointers.contains_key(name), "unknown MoE pointer");
            }
            for name in &e.i32_scalars {
                ensure!(node.scalars.contains_key(name), "unknown MoE scalar");
            }
        }
        ensure!(!sequential||crate::config::wide_rows(c.rows),"only wide MoE layouts may share scratch");
        let (scratch,sequential_scratch)=if let Some(prior)=donor {
            ensure!(sequential&&prior.contract.rows>=c.rows&&prior.scratch.len()>=c.scratch_bytes
                &&cuda.owns_buffer(&prior.scratch)&&prior.sequential_scratch.is_some(),
                "wide MoE scratch donor capacity/context");
            (prior.scratch.clone(),prior.sequential_scratch.clone())
        }else{
            let mut scratch=cuda.allocate::<u8>(c.scratch_bytes)?;scratch.zero()?;
            (Rc::new(scratch),sequential.then(||Rc::new(SequentialScratch{
                rows:Cell::new(c.rows),stream:Cell::new(0)})))
        };
        let mut ones = cuda.allocate::<f32>(c.experts)?;
        ones.upload(&vec![1.; c.experts])?;
        let mut output = cuda.allocate::<bf16>(c.rows * c.hidden)?;
        output.zero()?;
        let mut workspace=Self {
            cuda: cuda.clone(),
            primitives,
            contract: c,
            scratch,
            sequential_scratch,
            ones,
            output: Rc::new(output),
            live_bounds: false,
            single_row_micro: None,
            three_row_task_order: None,
            packed: None,
            input_address: None,
            input_address_enabled: Cell::new(false),
            scale_codes: None,
            scale_codes_enabled: Cell::new(false),
            fc_pdl: None,
            late_fc_pdl: None,
            late_fc_pdl_enabled: Cell::new(false),
            cache_ca: None,
            cache_ca_enabled: Cell::new(false),
            cooperative_epilogues: [None, None],
            cooperative_epilogue_mode: Cell::new(0),
            fc_pdl_mode: Cell::new(0),
            wide_transfer:None,
            wide_transfer_enabled:Cell::new(false),
            stream: Cell::new(None),
        };
        if matches!(workspace.contract.rows,16|128|512|1024) {
            let enabled=if workspace.contract.rows<=128{wide_transfer_requested()?}else{prefill_transfer_requested()?};
            if enabled||std::env::var("CANADA_SCRATCH_MOE_WIDE_TRANSFER_SCREEN").as_deref()==Ok("1")
                ||std::env::var("CANADA_SCRATCH_PREFILL_BOUNDS_SCREEN").as_deref()==Ok("1") {
                workspace.enable_wide_transfer(root.parent().context("expert bundle parent")?)?;
                workspace.wide_transfer_enabled.set(enabled);
            }
        }
        Ok(workspace)
    }
    /// Must run outside the encoder CUDA graph before every wide chunk. The
    /// clear is ordered with both previous users and the next graph on the
    /// single shared stream, and is needed only when the row geometry changes.
    pub(crate) fn prepare_sequential_scratch(&self,stream:&Stream)->Result<()> {
        if let Some(shared)=&self.sequential_scratch {
            self.guard_stream(stream)?;
            if shared.rows.get()!=self.contract.rows {
                unsafe{self.cuda.zero_device(self.scratch.ptr(),self.contract.scratch_bytes,stream)?;}
                shared.rows.set(self.contract.rows);
            }
        }
        Ok(())
    }
    pub(crate) fn sequential_scratch_bytes(&self)->usize {
        if self.sequential_scratch.is_some(){self.scratch.len()}else{0}
    }
    pub fn output(&self) -> &Buffer<bf16> {
        &self.output
    }
    /// A second launch plan shares the existing scratch/output allocation.
    /// Packed bytes never enter the ordinary BF16 operand API.
    pub(crate) fn enable_packed(&mut self, root: &Path) -> Result<()> {
        let c: Contract = serde_json::from_slice(&std::fs::read(root.join("runtime-contract.json"))?)?;
        let patch = c.prequantized_input_overlay.as_ref().context("packed MoE provenance absent")?;
        ensure!(self.packed.is_none() && [8192,16384].contains(&self.contract.rows)
            && c.abi == 1 && (c.rows,c.hidden,c.intermediate,c.experts,c.topk) == (self.contract.rows,5120,576,384,6)
            && c.deterministic && c.swiglu_limit == 10. && c.scratch_initializer == "zero"
            && c.scratch_bytes == self.contract.scratch_bytes && c.scratch_bytes<=self.scratch.len()
            && c.input_format.as_deref() == Some("dual_quant_rows_v1")
            && c.input_bytes == c.rows * c.hidden * 37 / 32 && c.routed_quantizer_entry == Some(2)
            && c.kernel_library_sha256 == "d9450a096168c6f6a6d7824e3d5d86c9554cc1fc9476d3af3821cbb9fbfbeb5b"
            && patch["base_commit"] == "a7d7d29b2ef8869086e0ceaa787321f17544e3c9"
            && patch["files"]["b12x_next/moe/_shared/kernels/dynamic.py"]["patched_sha256"] == "4001a6789c18bd9588f0e7e4a4012e13fe067e7d8770bcab3ed9b77a8aa50de1"
            && patch["files"]["b12x_next/moe/fused_moe/_impl.py"]["patched_sha256"] == "2f474f669d3af802d65d697bf54a11713afcd347d017ae16c6e75be5d94ed86d"
            && c.nodes.len() == self.contract.nodes.len(), "unqualified packed MoE contract");
        let primitives = Primitives::open(&self.cuda, &root.join("libcanada_moe_primitives.so"), &root.join("bindings.json"))?;
        ensure!(primitives.bindings.abi == 1 && primitives.bindings.library_sha256 == c.kernel_library_sha256,
            "packed MoE library contract");
        for (a,b) in c.nodes.iter().zip(&self.contract.nodes) {
            ensure!(a.kernel_id == b.kernel_id && a.pointers == b.pointers && a.scalars == b.scalars,
                "packed MoE must retain the original scratch plan and later arithmetic");
            let e = primitives.bindings.entries.get(a.entry as usize).context("packed MoE entry")?;
            ensure!(e.kernel_id == a.kernel_id && e.pointers.len() == a.pointers.len()
                && e.pointers.iter().all(|n| a.pointers.contains_key(n))
                && e.i32_scalars.len() == a.scalars.len() && e.i32_scalars.iter().all(|n| a.scalars.contains_key(n)),
                "packed MoE ABI drift");
        }
        for i in [2,3] {
            let e = primitives.bindings.entries.get(i).context("packed input quantizer")?;
            ensure!(e.kernel_id == "gemm.mxfp8_quant_cute"
                && e.pointers == ["source_ptr","values_ptr","scale_rows_ptr","scale_mma_ptr"]
                && e.i32_scalars == ["m","source_k","grid_x"], "packed quantizer ABI drift");
        }
        self.packed = Some(PackedInput { primitives, contract: c });
        Ok(())
    }
    pub(crate) fn quantize_packed_shard(&self, shared: bool, input: &Buffer<bf16>,
        values: &Buffer<u8>, row_scales: &Buffer<u8>, mma_scales: &Buffer<u8>, stream: &Stream) -> Result<()> {
        let p = self.packed.as_ref().context("packed MoE absent")?;
        self.guard_stream(stream)?;
        ensure!(input.len() == (self.contract.rows/4)*5120 && values.len() == input.len()
            && row_scales.len() == input.len()/32 && mma_scales.len() == row_scales.len()
            && self.cuda.owns_buffer(input) && self.cuda.owns_buffer(values)
            && self.cuda.owns_buffer(row_scales) && self.cuda.owns_buffer(mma_scales), "packed shard quantizer geometry/context");
        unsafe { p.primitives.launch(if shared {3}else{2},
            &[input.ptr(),values.ptr(),row_scales.ptr(),mma_scales.ptr()], &[(self.contract.rows/4)as i32,5120,192], stream) }
    }
    pub(crate) fn run_packed(&self, expert: &Expert, input: &Buffer<u8>, ids: &Buffer<i32>,
        routing: &Buffer<f32>, stream: &Stream) -> Result<()> {
        let p = self.packed.as_ref().context("packed MoE absent")?;
        self.guard_stream(stream)?;
        ensure!(input.len() == p.contract.input_bytes && self.cuda.owns_buffer(input), "packed MoE operand");
        self.launch_plan(&p.contract, &p.primitives, expert, input.ptr(), input.len(), ids, routing, p.contract.rows, false, stream)
    }
    fn guard_stream(&self, stream: &Stream) -> Result<()> {
        let id = stream.raw() as usize;
        ensure!(self.cuda.owns_stream(stream) && self.stream.get().is_none_or(|s|s == id),
            "shared MoE workspace requires one ordered stream");
        self.stream.set(Some(id));
        if let Some(shared)=&self.sequential_scratch {
            ensure!(shared.stream.get()==0||shared.stream.get()==id,
                "sequential MoE scratch requires one ordered stream across widths");
            shared.stream.set(id);
        }
        Ok(())
    }
    pub(crate) fn micro_fp32(&self) -> bool {
        self.contract.projection_dtype.as_deref()==Some("float32")
            && self.contract.nodes.len()==3
    }
    /// One stable output address for both launch families. Graphs are already
    /// keyed by live row count; selection happens while enqueueing/capturing,
    /// with no device conditional or per-replay CPU/GPU synchronization.
    pub(crate) fn new_live(cuda: &Cuda, kernels: &Path) -> Result<Self> {
        let scale=std::env::var("CANADA_SCRATCH_MOE_SCALE_REUSE").as_deref()==Ok("1");
        ensure!(!scale || std::env::var("CANADA_SCRATCH_MOE_SCALE_AB").as_deref()==Ok("1"),
            "scale-reuse candidate requires isolated comparison");
        let task_order = std::env::var("CANADA_SCRATCH_MOE_TASK_ORDER").as_deref() == Ok("1");
        let task_order_m3=std::env::var("CANADA_SCRATCH_MOE_TASK_ORDER_M3").as_deref()==Ok("1");
        ensure!(!task_order_m3 || (!task_order && !scale
            && std::env::var("CANADA_SCRATCH_VERIFICATION_FUSION_SCREEN").as_deref()==Ok("1")),
            "M3 task ordering requires an isolated shape-selective comparison");
        ensure!(!task_order || (!scale && std::env::var("CANADA_SCRATCH_COMPUTE_AB").as_deref() == Ok("1")),
            "task-order candidate requires isolated comparison");
        let mut grouped=Self::new(cuda,&kernels.join(if task_order {"moe-runtime-m6-task-order-v1"}else if scale {"moe-runtime-m6-scale-reuse-v1"}else{"moe-runtime-m6-v0"}))?;
        let mut micro=Self::new(cuda,&kernels.join("moe-runtime-m6-micro-fp32-v1"))?;
        ensure!(grouped.contract.kernel_library_sha256==if task_order {"36b502a2cd3e13f4b1a340b2494d18ddd4afe0260b869929aa2bc275ffa24003"}else if scale {"9137d340b5c5042245caea9ad77872e63073e0bf8ef3bac6d20d2785f5c563d2"}else{"7c4cb9ca260c6bdc22cdde9d2690778d404249d0ad6bbb408c5cf407c94ce207"}
            && micro.contract.kernel_library_sha256=="03b36cae287f80b123c9ed8b82ac96d722ceee11b4010c036a58a836a408c374",
            "live MoE requires the s305-qualified libraries");
        for w in [&grouped,&micro] {
            ensure!((w.contract.rows,w.contract.experts,w.contract.topk)==(6,384,6),"live MoE shape");
            for n in &w.contract.nodes {
                let s=&n.scalars;
                match n.kernel_id.as_str() {
                    "integration.tp_moe.dynamic" => ensure!(s.get("num_tokens")==Some(&6),"grouped live ABI"),
                    "gemm.mxfp8_quant_cute" => ensure!(s.get("m")==Some(&6) && s.get("source_k")==Some(&5120)
                        && s.get("grid_x")==Some(&60),"micro quant live ABI"),
                    "moe.w4a8.compact_micro" => ensure!(s.get("num_pairs")==Some(&36),"micro route live ABI"),
                    "moe.w4a16.topk_sum" => ensure!(s.get("active_m")==Some(&6),"MoE reduction live ABI"),
                    _ => anyhow::bail!("unqualified live MoE kernel"),
                }
            }
        }
        grouped.live_bounds=true;
        micro.live_bounds=true;
        micro.output=grouped.output.clone();
        grouped.single_row_micro=Some(Box::new(micro));
        if task_order_m3 {
            let mut alternative=Self::new(cuda,&kernels.join("moe-runtime-m6-task-order-v1"))?;
            ensure!(alternative.task_order() && alternative.contract.scratch_bytes==grouped.contract.scratch_bytes
                && alternative.contract.nodes.len()==grouped.contract.nodes.len()
                && alternative.contract.nodes.iter().zip(&grouped.contract.nodes).all(|(a,b)|
                    a.kernel_id==b.kernel_id && a.scalars==b.scalars
                    && a.pointers.len()==b.pointers.len()
                    && a.pointers.iter().all(|(key,value)|b.pointers.get(key).is_some_and(|other|
                        value.role==other.role && value.offset==other.offset))),
                "M3 task-order ABI/layout must match current grouped workspace");
            alternative.live_bounds=true;
            alternative.output=grouped.output.clone();
            grouped.three_row_task_order=Some(Box::new(alternative));
        }
        let input_address = match std::env::var("CANADA_SCRATCH_MOE_INPUT_ADDRESS") {
            Ok(v) => { ensure!(v=="0"||v=="1","invalid expert input-address policy");v=="1" },
            Err(std::env::VarError::NotPresent) => !task_order&&!task_order_m3&&!scale,
            Err(e) => return Err(e.into()),
        };
        if input_address || std::env::var("CANADA_SCRATCH_MOE_INPUT_ADDRESS_SCREEN").as_deref()==Ok("1") {
            ensure!(!task_order&&!task_order_m3&&!scale,"expert input-address plan requires ordinary grouped scheduling");
            grouped.enable_input_address(&kernels.join("moe-runtime-m6-input-address-v1"))?;
            grouped.input_address_enabled.set(input_address);
        }
        if std::env::var("CANADA_SCRATCH_MOE_SCALE_CODE_SCREEN").as_deref()==Ok("1") {
            ensure!(input_address&&!task_order&&!task_order_m3&&!scale,
                "scale-code screen requires the current input-address control");
            grouped.enable_scale_codes(&kernels.join("moe-runtime-m6-scale-codes-v5"))?;
        }
        let pdl_m3=match std::env::var("CANADA_SCRATCH_MOE_FC_PDL_M3") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid M3 expert PDL policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
        };
        if pdl_m3 || std::env::var("CANADA_SCRATCH_MOE_FC_PDL_SCREEN").as_deref()==Ok("1") {
            ensure!(input_address&&!task_order&&!task_order_m3&&!scale&&grouped.scale_codes.is_none(),
                "FC PDL screen requires the ordinary input-address control");
            grouped.enable_fc_pdl(&kernels.join("moe-runtime-m6-fc-pdl-v1"))?;
            grouped.fc_pdl_mode.set(if pdl_m3 {3}else{0});
        }
        let cache_ca=match std::env::var("CANADA_SCRATCH_MOE_CACHE_CA") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid expert cache-ca policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>input_address&&!task_order&&!task_order_m3&&!scale&&grouped.scale_codes.is_none(),
            Err(e)=>return Err(e.into()),
        };
        if cache_ca || std::env::var("CANADA_SCRATCH_MOE_CACHE_CA_SCREEN").as_deref()==Ok("1") {
            grouped.enable_cache_ca(&kernels.join("moe-runtime-m6-cache-ca-v1"))?;
            grouped.cache_ca_enabled.set(cache_ca);
        }
        if std::env::var("CANADA_SCRATCH_MOE_LATE_PDL_SCREEN").as_deref()==Ok("1") {
            grouped.enable_late_fc_pdl(&kernels.join("moe-runtime-m6-fc-pdl-late-v1"))?;
        }
        let cooperative=match std::env::var("CANADA_SCRATCH_MOE_COOPERATIVE_EPILOGUE") {
            Ok(v)=>match v.as_str(){"0"=>0,"1"=>1,"2"=>2,
                _=>anyhow::bail!("invalid expert cooperative-epilogue policy")},
            Err(std::env::VarError::NotPresent)=>0,Err(e)=>return Err(e.into()),
        };
        let cooperative_screen=std::env::var("CANADA_SCRATCH_MOE_EPILOGUE_SCREEN").as_deref()==Ok("1");
        if cooperative!=0 || cooperative_screen {
            ensure!(input_address&&pdl_m3&&cache_ca&&grouped.late_fc_pdl.is_none()
                &&grouped.scale_codes.is_none(),"cooperative epilogue requires the current M3/M6 plans");
            grouped.enable_cooperative_epilogues(kernels,cooperative==2||cooperative_screen)?;
            grouped.cooperative_epilogue_mode.set(cooperative);
        }
        Ok(grouped)
    }
    pub(crate) fn live_bounds(&self) -> bool { self.live_bounds }
    // The s825 AOT check preserves every pointer, scratch offset and scalar.
    // Reuse those existing allocations; both modules outlive captured graphs.
    fn enable_input_address(&mut self, root:&Path)->Result<()> {
        let alternate=Self::new(&self.cuda,root)?;
        let c=&alternate.contract;
        let patch=c.fc1_input_address_overlay.as_ref().context("input-address provenance absent")?;
        ensure!(self.input_address.is_none() && self.live_bounds
            && (c.rows,c.experts,c.topk,c.scratch_bytes)==(6,384,6,4118728)
            && c.kernel_library_sha256=="c76ab738eccf40f86106ad2651f883a5e79e66492be61fe09a68a21bf54cc761"
            && c.task_order_overlay.is_none() && c.fc1_scale_reuse_overlay.is_none()
            && patch["base_commit"]=="a7d7d29b2ef8869086e0ceaa787321f17544e3c9"
            && patch["hoisted_input"]==true && patch["output_major_tasks"]==false
            && patch["reuse_fc1_weight_scales"]==false && patch["stages"]==serde_json::json!([2,2])
            && patch["files"]["w4a8_phase1.py"]["patched_sha256"]=="5192ef5daa86b0f8c70c596d2f9be176f6a3496c32046a60685fd8b304b61509"
            && patch["files"]["w4a8_phase2.py"]["patched_sha256"]=="0c9fce39b243676e6754ea1fe111f765b5a993b6d6946ac410301a4663d1c5c3"
            && c.scratch_bytes==self.contract.scratch_bytes && c.nodes.len()==self.contract.nodes.len()
            && c.nodes.iter().zip(&self.contract.nodes).all(|(a,b)|
                a.kernel_id==b.kernel_id && a.scalars==b.scalars && a.pointers==b.pointers),
            "unqualified expert input-address plan");
        self.input_address=Some(PackedInput{primitives:alternate.primitives,contract:alternate.contract});
        Ok(())
    }
    pub(crate) fn set_input_address_for_probe(&self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MOE_INPUT_ADDRESS_SCREEN").as_deref()==Ok("1")
            && self.input_address.is_some() && self.live_bounds && self.contract.rows==6,
            "expert input-address probe admission");
        self.guard_stream(stream)?;
        self.input_address_enabled.set(enabled);
        Ok(())
    }
    fn enable_scale_codes(&mut self,root:&Path)->Result<()> {
        let alternate=Self::new(&self.cuda,root)?;
        let c=&alternate.contract;
        let patch=c.fc1_input_address_overlay.as_ref().context("scale-code provenance absent")?;
        ensure!(self.scale_codes.is_none()&&self.live_bounds&&self.input_address_enabled.get()
            &&(c.rows,c.experts,c.topk,c.scratch_bytes)==(6,384,6,4118728)
            &&c.kernel_library_sha256=="db4df4136bb446dfe5167a908d1fa336a2240c947af931125da37dc6a5b339ca"
            &&patch["base_commit"]=="a7d7d29b2ef8869086e0ceaa787321f17544e3c9"
            &&patch["files"]["w4a8_phase1.py"]["patched_sha256"]=="d8667d3042b024d413ca08dbad62f79784daa9240a7d21a90dfb6afb2b3af3d2"
            &&patch["files"]["w4a8_phase2.py"]["patched_sha256"]=="0c9fce39b243676e6754ea1fe111f765b5a993b6d6946ac410301a4663d1c5c3"
            &&c.nodes.len()==self.contract.nodes.len()
            &&c.nodes.iter().zip(&self.contract.nodes).all(|(a,b)|
                a.kernel_id==b.kernel_id&&a.scalars==b.scalars&&a.pointers==b.pointers),
            "unqualified lossless scale-code kernel/ABI");
        self.scale_codes=Some(PackedInput{primitives:alternate.primitives,contract:alternate.contract});
        Ok(())
    }
    pub(crate) fn set_scale_codes_for_probe(&self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MOE_SCALE_CODE_SCREEN").as_deref()==Ok("1")
            &&self.scale_codes.is_some()&&self.input_address_enabled.get()&&self.live_bounds,
            "scale-code probe admission");
        self.guard_stream(stream)?;self.scale_codes_enabled.set(enabled);Ok(())
    }
    fn enable_fc_pdl(&mut self,root:&Path)->Result<()> {
        let alternate=Self::new(&self.cuda,root)?;
        let c=&alternate.contract;
        let patch=c.fc1_input_address_overlay.as_ref().context("FC PDL provenance absent")?;
        ensure!(self.fc_pdl.is_none()&&self.live_bounds&&self.input_address_enabled.get()
            &&(c.rows,c.experts,c.topk,c.scratch_bytes)==(6,384,6,4118728)
            &&c.kernel_library_sha256=="0ed9ef3842268d289419c63b26ab09ae8049a0f98956ebf581f243b4b0eafcaf"
            &&c.task_order_overlay.is_none()&&c.fc1_scale_reuse_overlay.is_none()
            &&patch["base_commit"]=="a7d7d29b2ef8869086e0ceaa787321f17544e3c9"
            &&patch["fc_pdl"]==true&&patch["hoisted_input"]==true
            &&patch["output_major_tasks"]==false&&patch["reuse_fc1_weight_scales"]==false
            &&patch["stages"]==serde_json::json!([2,2])
            &&patch["files"]["w4a8_phase1.py"]["patched_sha256"]=="b8e8c848bb1462aecfd549de98373739a8c4a43e5ff79b23f8a10d62e9f7d592"
            &&patch["files"]["w4a8_phase2.py"]["patched_sha256"]=="3aff59cbaee697abc3ef33177beb3cc996b11d52d31c928edc2bbba89dedc54f"
            &&c.scratch_bytes==self.contract.scratch_bytes&&c.nodes.len()==self.contract.nodes.len()
            &&c.nodes.iter().zip(&self.contract.nodes).all(|(a,b)|
                a.kernel_id==b.kernel_id&&a.scalars==b.scalars&&a.pointers==b.pointers),
            "unqualified FC PDL kernel/ABI");
        self.fc_pdl=Some(PackedInput{primitives:alternate.primitives,contract:alternate.contract});
        Ok(())
    }
    pub(crate) fn set_fc_pdl_for_probe(&self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MOE_FC_PDL_SCREEN").as_deref()==Ok("1")
            &&self.fc_pdl.is_some()&&self.input_address_enabled.get()&&self.live_bounds
            &&!self.scale_codes_enabled.get(),"FC PDL probe admission");
        self.guard_stream(stream)?;
        self.fc_pdl_mode.set(if !enabled {0}
            else if std::env::var("CANADA_SCRATCH_MOE_FC_PDL_M3").as_deref()==Ok("1") {3}else{1});Ok(())
    }
    fn enable_late_fc_pdl(&mut self,root:&Path)->Result<()> {
        let alternate=Self::new(&self.cuda,root)?;let c=&alternate.contract;
        let patch=c.fc1_input_address_overlay.as_ref().context("late FC PDL provenance absent")?;
        ensure!(self.late_fc_pdl.is_none()&&self.fc_pdl.is_some()&&self.live_bounds
            &&self.input_address_enabled.get()&&self.scale_codes.is_none()
            &&(c.rows,c.experts,c.topk,c.scratch_bytes)==(6,384,6,4118728)
            &&c.kernel_library_sha256=="85a84f6f50286ecc0d716fa11f5a7822126dc8e397ddafff13088d1350461413"
            &&patch["base_commit"]=="a7d7d29b2ef8869086e0ceaa787321f17544e3c9"
            &&patch["fc_pdl"]==true&&patch["late_pdl_trigger"]==true&&patch["hoisted_input"]==true
            &&patch["stages"]==serde_json::json!([2,2])
            &&patch["files"]["w4a8_phase1.py"]["patched_sha256"]=="1707386b4f8345763655734a7e80a3fa26e4ab8d62b589e87aacc219d48cb3d7"
            &&patch["files"]["w4a8_phase2.py"]["patched_sha256"]=="3aff59cbaee697abc3ef33177beb3cc996b11d52d31c928edc2bbba89dedc54f"
            &&c.scratch_bytes==self.contract.scratch_bytes&&c.nodes.len()==self.contract.nodes.len()
            &&c.nodes.iter().zip(&self.contract.nodes).all(|(a,b)|a.kernel_id==b.kernel_id&&a.scalars==b.scalars&&a.pointers==b.pointers),
            "unqualified late FC PDL ABI");
        self.late_fc_pdl=Some(PackedInput{primitives:alternate.primitives,contract:alternate.contract});Ok(())
    }
    pub(crate) fn set_late_fc_pdl_for_probe(&self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MOE_LATE_PDL_SCREEN").as_deref()==Ok("1")
            &&self.late_fc_pdl.is_some()&&self.input_address_enabled.get()&&self.live_bounds
            &&self.fc_pdl_mode.get()==3&&!self.scale_codes_enabled.get(),"late FC PDL screen admission");
        self.guard_stream(stream)?;self.late_fc_pdl_enabled.set(enabled);Ok(())
    }
    fn enable_cache_ca(&mut self,root:&Path)->Result<()> {
        let alternate=Self::new(&self.cuda,root)?;let c=&alternate.contract;
        let patch=c.fc1_input_address_overlay.as_ref().context("MoE cache-ca provenance absent")?;
        ensure!(self.cache_ca.is_none()&&self.live_bounds
            &&self.input_address_enabled.get()&&self.scale_codes.is_none()
            &&(c.rows,c.experts,c.topk,c.scratch_bytes)==(6,384,6,4118728)
            &&c.kernel_library_sha256=="e68e2711f1603e0b2261c939e67650ea00ae1f9dfffb2b344ea49792214554de"
            &&patch["base_commit"]=="a7d7d29b2ef8869086e0ceaa787321f17544e3c9"
            &&patch["fc_pdl"]==false&&patch["async_copy_cache_operator"]=="ca"&&patch["hoisted_input"]==true
            &&patch["stages"]==serde_json::json!([2,2])
            &&patch["files"]["w4a8_phase1.py"]["patched_sha256"]=="5192ef5daa86b0f8c70c596d2f9be176f6a3496c32046a60685fd8b304b61509"
            &&patch["files"]["w4a8_phase2.py"]["patched_sha256"]=="0c9fce39b243676e6754ea1fe111f765b5a993b6d6946ac410301a4663d1c5c3"
            &&patch["files"]["intrinsics.py"]["patched_sha256"]=="673c83d70fd4b6c365738ad84172b6f0d2352baa2326821a01c2abaad6f1db7f"
            &&c.scratch_bytes==self.contract.scratch_bytes&&c.nodes.len()==self.contract.nodes.len()
            &&c.nodes.iter().zip(&self.contract.nodes).all(|(a,b)|a.kernel_id==b.kernel_id&&a.scalars==b.scalars&&a.pointers==b.pointers),
            "unqualified MoE cache-ca ABI");
        self.cache_ca=Some(PackedInput{primitives:alternate.primitives,contract:alternate.contract});Ok(())
    }
    pub(crate) fn set_cache_ca_for_probe(&self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MOE_CACHE_CA_SCREEN").as_deref()==Ok("1")
            &&self.cache_ca.is_some()&&self.input_address_enabled.get()&&self.live_bounds
            &&self.fc_pdl_mode.get()==3&&!self.scale_codes_enabled.get(),"MoE cache-ca screen admission");
        self.guard_stream(stream)?;self.cache_ca_enabled.set(enabled);Ok(())
    }
    fn enable_cooperative_epilogues(&mut self,kernels:&Path,include_three_row:bool)->Result<()> {
        ensure!(self.cooperative_epilogues.iter().all(Option::is_none)&&self.live_bounds,
            "duplicate cooperative epilogue or unbounded workspace");
        let variants=[
            ("moe-runtime-m6-epilogue-register-pdl-v1",
             "794201fb05754f2cacb767296566aa503224cdb6a2f1230c085ee014c83e07f1",
             "7aaea8fb969a174b55a02012d54f9ef32c0a6f2f26a0950535aede2e7effeee5",
             "3aff59cbaee697abc3ef33177beb3cc996b11d52d31c928edc2bbba89dedc54f",true),
            ("moe-runtime-m6-epilogue-quad-ca-v1",
             "b9d555857a80d72e04b63467fb8c0e0604eef3120f946df2c6c9b0c5186feb59",
             "bf679a26a73466f08fb8ac966963ff11025892a451be05f8266e3528c99a0b76",
             "0c9fce39b243676e6754ea1fe111f765b5a993b6d6946ac410301a4663d1c5c3",false),
        ];
        for (index,(name,library,phase1,phase2,pdl)) in variants.into_iter().enumerate() {
            if index==0&&!include_three_row{continue;}
            let alternate=Self::new(&self.cuda,&kernels.join(name))?;let c=&alternate.contract;
            let patch=c.fc1_input_address_overlay.as_ref().context("cooperative epilogue provenance absent")?;
            ensure!((c.rows,c.experts,c.hidden,c.intermediate,c.topk,c.scratch_bytes)==(6,384,5120,576,6,4118728)
                &&c.kernel_library_sha256==library&&c.task_order_overlay.is_none()&&c.fc1_scale_reuse_overlay.is_none()
                &&patch["base_commit"]=="a7d7d29b2ef8869086e0ceaa787321f17544e3c9"
                &&patch["hoisted_input"]==true&&patch["fc_pdl"]==pdl
                &&patch["output_major_tasks"]==false&&patch["reuse_fc1_weight_scales"]==false
                &&patch["stages"]==serde_json::json!([2,2])
                &&patch["files"]["w4a8_phase1.py"]["patched_sha256"]==phase1
                &&patch["files"]["w4a8_phase2.py"]["patched_sha256"]==phase2
                &&c.scratch_bytes==self.contract.scratch_bytes&&c.nodes.len()==self.contract.nodes.len()
                &&c.nodes.iter().zip(&self.contract.nodes).all(|(a,b)|
                    a.kernel_id==b.kernel_id&&a.scalars==b.scalars&&a.pointers==b.pointers),
                "unqualified cooperative epilogue numerical/layout contract");
            ensure!(if pdl {patch["cooperative_epilogue"]=="register"} else {
                patch["cooperative_epilogue"]==true&&patch["async_copy_cache_operator"]=="ca"
                &&patch["files"]["intrinsics.py"]["patched_sha256"]=="673c83d70fd4b6c365738ad84172b6f0d2352baa2326821a01c2abaad6f1db7f"
            },"cooperative epilogue variant provenance");
            self.cooperative_epilogues[index]=Some(PackedInput{primitives:alternate.primitives,contract:alternate.contract});
        }
        Ok(())
    }
    pub(crate) fn set_cooperative_epilogues_for_probe(&self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MOE_EPILOGUE_SCREEN").as_deref()==Ok("1")
            &&self.cooperative_epilogues.iter().all(Option::is_some)&&self.input_address_enabled.get()
            &&self.live_bounds&&self.fc_pdl_mode.get()==3&&self.cache_ca_enabled.get()
            &&!self.scale_codes_enabled.get()&&!self.late_fc_pdl_enabled.get(),"cooperative epilogue screen admission");
        self.guard_stream(stream)?;self.cooperative_epilogue_mode.set(if enabled{2}else{0});Ok(())
    }
    /// s581 qualified the existing M16 library at live widths8/10/12/16.
    /// Only launch bounds change; scratch capacities and arithmetic stay fixed.
    pub(crate) fn new_live16(cuda: &Cuda, kernels: &Path) -> Result<Self> {
        let mut workspace=Self::new(cuda,&kernels.join("moe-runtime-m16-v0"))?;
        let c=&workspace.contract;
        ensure!((c.rows,c.experts,c.hidden,c.intermediate,c.topk)==(16,384,5120,576,6)
            && c.kernel_library_sha256=="7c4cb9ca260c6bdc22cdde9d2690778d404249d0ad6bbb408c5cf407c94ce207"
            && c.nodes.len()==2 && !workspace.micro_fp32()
            && !workspace.task_order() && !workspace.scale_reuse(),"live M16 qualified contract");
        for node in &c.nodes {
            let expected: &[(&str,i32)]=match node.kernel_id.as_str() {
                "integration.tp_moe.dynamic"=>&[("num_tokens",16),("max_rows",96),("scatter_rows",96),
                    ("rows_padded",1616),("max_tasks",505),("max_phys_tiles",101),("max_active_clusters",48)],
                "moe.w4a16.topk_sum"=>&[("active_m",16),("weight_num_experts",0),("route_num_experts",0)],
                _=>anyhow::bail!("unqualified live M16 kernel"),
            };
            ensure!(node.scalars.len()==expected.len()
                && expected.iter().all(|(key,value)|node.scalars.get(*key)==Some(value)),"live M16 launch capacities");
        }
        workspace.live_bounds=true;
        Ok(workspace)
    }
    /// Transfer code only. Keep this workspace's original capacity, offsets,
    /// launch bounds and output storage; M6 metadata is never used as M16/M128.
    fn enable_wide_transfer(&mut self,kernels:&Path)->Result<()> {
        let c=&self.contract;
        let scratch=match c.rows{16=>10912148,128=>50170556,512=>88190076,1024=>138889596,_=>0};
        ensure!(scratch>0&&(c.experts,c.hidden,c.intermediate,c.topk)==(384,5120,576,6)
            &&c.deterministic&&c.swiglu_limit==10.0&&c.nodes.len()==2
            &&c.kernel_library_sha256=="7c4cb9ca260c6bdc22cdde9d2690778d404249d0ad6bbb408c5cf407c94ce207"
            &&c.scratch_bytes==scratch,"wide expert baseline contract");
        let(name,pin)=if c.rows==16 {
            ("moe-runtime-m6-cache-ca-v1","e68e2711f1603e0b2261c939e67650ea00ae1f9dfffb2b344ea49792214554de")
        }else {
            ("moe-runtime-m6-epilogue-quad-ca-v1","b9d555857a80d72e04b63467fb8c0e0604eef3120f946df2c6c9b0c5186feb59")
        };
        let root=kernels.join(name);
        let alternate=Primitives::open(&self.cuda,&root.join("libcanada_moe_primitives.so"),&root.join("bindings.json"))?;
        ensure!(alternate.bindings.library_sha256==pin&&alternate.bindings.abi==1
            &&alternate.bindings.source_commit=="a7d7d29b2ef8869086e0ceaa787321f17544e3c9"
            &&alternate.bindings.entries.len()==self.primitives.bindings.entries.len()
            &&alternate.bindings.entries.iter().zip(&self.primitives.bindings.entries).all(|(a,b)|
                a.index==b.index&&a.kernel_id==b.kernel_id&&a.pointers==b.pointers
                &&a.i32_scalars==b.i32_scalars&&a.arguments.is_empty()&&b.arguments.is_empty()),
            "wide expert transferred kernel ABI/pin");
        self.wide_transfer=Some(alternate);Ok(())
    }
    pub(crate) fn wide_transfer_enabled(&self)->bool {self.wide_transfer_enabled.get()}
    pub(crate) fn set_wide_transfer_for_probe(&self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!((std::env::var("CANADA_SCRATCH_MOE_WIDE_TRANSFER_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_PREFILL_BOUNDS_SCREEN").as_deref()==Ok("1"))
            &&matches!(self.contract.rows,16|128|512|1024)&&self.wide_transfer.is_some(),"wide expert probe admission");
        self.guard_stream(stream)?;self.wide_transfer_enabled.set(enabled);Ok(())
    }
    pub(crate) fn task_order(&self) -> bool { self.contract.task_order_overlay.is_some() }
    pub(crate) fn task_order_m3(&self)->bool {self.three_row_task_order.is_some()}
    pub(crate) fn scale_reuse(&self) -> bool { self.contract.fc1_scale_reuse_overlay.is_some() }
    pub fn run(
        &self,
        expert: &Expert,
        input: &Buffer<bf16>,
        ids: &Buffer<i32>,
        routing_weights: &Buffer<f32>,
        stream: &Stream,
    ) -> Result<()> {
        self.run_live(expert,input,ids,routing_weights,self.contract.rows,stream)
    }
    pub(crate) fn run_live(&self,expert:&Expert,input:&Buffer<bf16>,ids:&Buffer<i32>,
        routing_weights:&Buffer<f32>,live:usize,stream:&Stream)->Result<()> {
        self.run_live_inner(expert,input,ids,routing_weights,live,false,stream)
    }
    /// Route scratch stays owned by this workspace and its stream. Ordinary
    /// callers still receive the fully reduced output; only LayerFfn may defer.
    pub(crate) fn run_live_for_ffn(&self,expert:&Expert,input:&Buffer<bf16>,ids:&Buffer<i32>,
        routing_weights:&Buffer<f32>,live:usize,fuse_sum:bool,stream:&Stream)->Result<Option<(u64,usize)>> {
        let fuse=fuse_sum && matches!(self.contract.rows,6|16|32)
            // The short M3 model screen lost time despite the component
            // saving. Keep its existing shared-expert overlap schedule.
            && (self.contract.rows!=6 || live==6)
            && !(live==1&&self.single_row_micro.is_some())
            && !(live==3&&self.three_row_task_order.is_some());
        let region=if fuse {Some((self.scratch.ptr()+Self::sum_input_offset(&self.contract)? as u64,
            if self.live_bounds {live}else{self.contract.rows}))}else{None};
        self.run_live_inner(expert,input,ids,routing_weights,live,fuse,stream)?;
        Ok(region)
    }
    fn sum_input_offset(c:&Contract)->Result<usize> {
        ensure!(matches!(c.rows,6|16|32)&&c.hidden==5120&&c.topk==6
            && c.projection_dtype.is_none(),"fused FFN sum requires BF16 route outputs");
        let tail=c.nodes.last().context("missing expert sum")?;
        ensure!(tail.kernel_id=="moe.w4a16.topk_sum"
            &&c.nodes.iter().filter(|n|n.kernel_id=="moe.w4a16.topk_sum").count()==1
            &&tail.scalars.get("active_m")==Some(&(c.rows as i32))
            &&tail.scalars.get("weight_num_experts")==Some(&0)
            &&tail.scalars.get("route_num_experts")==Some(&0),"fused expert sum contract");
        let input=tail.pointers.get("fc2_ptr").context("missing expert route pointer")?;
        let output=tail.pointers.get("output_ptr").context("missing expert sum output")?;
        ensure!(input.role=="scratch"&&input.offset+c.rows*6*5120*2<=c.scratch_bytes
            &&output.role=="output"&&output.offset==0,"fused expert route bounds");
        Ok(input.offset)
    }
    fn run_live_inner(
        &self,
        expert: &Expert,
        input: &Buffer<bf16>,
        ids: &Buffer<i32>,
        routing_weights: &Buffer<f32>,
        live: usize,
        defer_sum: bool,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(live<=self.contract.rows && (!self.live_bounds || live>0),"MoE live row domain");
        self.guard_stream(stream)?;
        if live==1 {
            if let Some(micro)=&self.single_row_micro {
                ensure!(!defer_sum,"micro expert output is already reduced");
                return micro.run_live(expert,input,ids,routing_weights,live,stream);
            }
        }
        if live==3 && let Some(alternative)=&self.three_row_task_order {
            ensure!(!defer_sum,"alternate task-order workspace sum cannot be deferred");
            return alternative.run_live(expert,input,ids,routing_weights,live,stream);
        }
        ensure!(input.len() == self.contract.rows * self.contract.hidden && self.cuda.owns_buffer(input),
            "BF16 MoE operand");
        let cooperative=self.cooperative_epilogue_mode.get();
        if (cooperative!=0&&live==6)||(cooperative==2&&live==3) {
            let p=self.cooperative_epilogues[usize::from(live==6)].as_ref()
                .context("cooperative expert epilogue plan absent")?;
            return self.launch_plan(&p.contract,&p.primitives,expert,input.ptr(),input.len()*2,ids,routing_weights,live,defer_sum,stream);
        }
        if self.late_fc_pdl_enabled.get() {
            let p=self.late_fc_pdl.as_ref().context("late expert FC PDL plan absent")?;
            return self.launch_plan(&p.contract,&p.primitives,expert,input.ptr(),input.len()*2,ids,routing_weights,live,defer_sum,stream);
        }
        if self.fc_pdl_mode.get()==1 || (self.fc_pdl_mode.get()==3 && live==3) {
            let p=self.fc_pdl.as_ref().context("expert FC PDL plan absent")?;
            return self.launch_plan(&p.contract,&p.primitives,expert,input.ptr(),input.len()*2,ids,routing_weights,live,defer_sum,stream);
        }
        if self.scale_codes_enabled.get() {
            let p=self.scale_codes.as_ref().context("expert scale-code plan absent")?;
            return self.launch_plan(&p.contract,&p.primitives,expert,input.ptr(),input.len()*2,ids,routing_weights,live,defer_sum,stream);
        }
        if self.cache_ca_enabled.get() && self.input_address_enabled.get() && live==6 {
            let p=self.cache_ca.as_ref().context("expert cache-ca plan absent")?;
            return self.launch_plan(&p.contract,&p.primitives,expert,input.ptr(),input.len()*2,ids,routing_weights,live,defer_sum,stream);
        }
        if self.input_address_enabled.get() {
            let p=self.input_address.as_ref().context("expert input-address plan absent")?;
            return self.launch_plan(&p.contract,&p.primitives,expert,input.ptr(),input.len()*2,ids,routing_weights,live,defer_sum,stream);
        }
        let primitives=if self.wide_transfer_enabled.get()
            && (self.contract.rows!=512 || live==self.contract.rows) {
            self.wide_transfer.as_ref().context("wide expert transfer plan absent")?
        }else{&self.primitives};
        self.launch_plan(&self.contract, primitives, expert, input.ptr(), input.len()*2, ids, routing_weights, live, defer_sum, stream)
    }
    #[allow(clippy::too_many_arguments)]
    fn launch_plan(&self, c: &Contract, primitives: &Primitives, expert: &Expert,
        input: u64, input_bytes: usize, ids: &Buffer<i32>, routing_weights: &Buffer<f32>,
        live: usize, defer_sum: bool, stream: &Stream) -> Result<()> {
        ensure!(self.sequential_scratch.as_ref().is_none_or(|s|s.rows.get()==c.rows),
            "wide MoE scratch layout was not prepared before enqueue/capture");
        ensure!(
            expert.layout.experts == c.experts
                && expert.layout.hidden == c.hidden
                && expert.layout.intermediate == c.intermediate
                && ids.len() == c.rows * c.topk
                && routing_weights.len() == c.rows * c.topk
                && self.cuda.owns_buffer(ids)
                && self.cuda.owns_buffer(routing_weights)
                && self.cuda.owns_stream(stream)
                && expert.weights.in_context(&self.cuda),
            "MoE input shape mismatch"
        );
        let bounded_rows=crate::fp8_linear::prefill_expert_rows(c.rows,stream);
        let bounded128=c.rows==128 && bounded_rows.is_some();
        let bounded512=c.rows==512 && bounded_rows.is_some();
        let bounded1024=c.rows==1024 && bounded_rows.is_some();
        if bounded128 {
            ensure!(bounded_rows==Some(live) && (17..128).contains(&live)
                && c.kernel_library_sha256=="7c4cb9ca260c6bdc22cdde9d2690778d404249d0ad6bbb408c5cf407c94ce207"
                && c.scratch_bytes==50170556 && c.nodes.len()==2,"qualified M128 live expert contract");
            for node in &c.nodes {
                let expected:&[(&str,i32)]=match node.kernel_id.as_str() {
                    "integration.tp_moe.dynamic"=>&[("num_tokens",128),("max_rows",768),("scatter_rows",768),
                        ("rows_padded",6896),("max_tasks",2155),("max_phys_tiles",431),("max_active_clusters",48)],
                    "moe.w4a16.topk_sum"=>&[("active_m",128),("weight_num_experts",0),("route_num_experts",0)],
                    _=>anyhow::bail!("unqualified M128 live expert kernel"),
                };
                ensure!(node.scalars.len()==expected.len() && expected.iter().all(|(k,v)|node.scalars.get(*k)==Some(v)),
                    "M128 expert capacity drift");
            }
        }
        if bounded512 {
            ensure!(bounded_rows==Some(live) && (129..512).contains(&live)
                && c.kernel_library_sha256=="7c4cb9ca260c6bdc22cdde9d2690778d404249d0ad6bbb408c5cf407c94ce207"
                && c.scratch_bytes==88190076 && c.nodes.len()==2,"qualified M512 live expert contract");
            for node in &c.nodes {
                let expected:&[(&str,i32)]=match node.kernel_id.as_str() {
                    "integration.tp_moe.dynamic"=>&[("num_tokens",512),("max_rows",3072),("scatter_rows",3072),
                        ("rows_padded",9200),("max_tasks",2875),("max_phys_tiles",575),("max_active_clusters",48)],
                    "moe.w4a16.topk_sum"=>&[("active_m",512),("weight_num_experts",0),("route_num_experts",0)],
                    _=>anyhow::bail!("unqualified M512 live expert kernel"),
                };
                ensure!(node.scalars.len()==expected.len() && expected.iter().all(|(k,v)|node.scalars.get(*k)==Some(v)),
                    "M512 expert capacity drift");
            }
        }
        if bounded1024 {
            ensure!(bounded_rows==Some(live) && (513..1024).contains(&live)
                && c.kernel_library_sha256=="7c4cb9ca260c6bdc22cdde9d2690778d404249d0ad6bbb408c5cf407c94ce207"
                && c.scratch_bytes==138889596 && c.nodes.len()==2,"qualified M1024 live expert contract");
            for node in &c.nodes {
                let expected:&[(&str,i32)]=match node.kernel_id.as_str() {
                    "integration.tp_moe.dynamic"=>&[("num_tokens",1024),("max_rows",6144),("scatter_rows",6144),
                        ("rows_padded",12272),("max_tasks",3835),("max_phys_tiles",767),("max_active_clusters",48)],
                    "moe.w4a16.topk_sum"=>&[("active_m",1024),("weight_num_experts",0),("route_num_experts",0)],
                    _=>anyhow::bail!("unqualified M1024 live expert kernel"),
                };
                ensure!(node.scalars.len()==expected.len() && expected.iter().all(|(k,v)|node.scalars.get(*k)==Some(v)),
                    "M1024 expert capacity drift");
            }
        }
        #[cfg(feature="storage-fault-injection")]
        if bounded512||bounded1024 { PREFILL_WIDE_BOUND_CALLS.with(|n|n.set(n.get()+1)); }
        let mut roles = BTreeMap::from([
            ("input", (input, input_bytes)),
            ("ids", (ids.ptr(), ids.len() * 4)),
            (
                "routing_weights",
                (routing_weights.ptr(), routing_weights.len() * 4),
            ),
            ("scratch", (self.scratch.ptr(), c.scratch_bytes)),
            ("ones", (self.ones.ptr(), self.ones.len() * 4)),
            ("output", (self.output.ptr(), self.output.len() * 2)),
        ]);
        for role in ["w13", "s13", "w2", "s2"] {
            roles.insert(role, expert.region(role)?);
        }
        if c.kernel_library_sha256=="db4df4136bb446dfe5167a908d1fa336a2240c947af931125da37dc6a5b339ca" {
            let codes=expert.weights.expert_scale_codes(expert.layout.layer)
                .context("scale-code kernel requires prepared exact scale bytes")?;
            ensure!(self.cuda.owns_buffer(codes)&&codes.len()==2*c.experts*576*160,
                "scale-code operand ownership/size");
            roles.insert("s13",(codes.ptr(),codes.len()));
        }
        if defer_sum {
            ensure!(Self::sum_input_offset(c)?==Self::sum_input_offset(&self.contract)?,
                "selected expert plan changed deferred route storage");
        }
        for node in &c.nodes {
            if defer_sum && node.kernel_id=="moe.w4a16.topk_sum" {continue;}
            let entry = &primitives.bindings.entries[node.entry as usize];
            let pointers = entry
                .pointers
                .iter()
                .map(|name| -> Result<u64> {
                    let p = &node.pointers[name];
                    let (base, bytes) = roles
                        .get(p.role.as_str())
                        .context("unknown MoE runtime pointer role")?;
                    ensure!(p.offset < *bytes, "MoE pointer offset outside operand");
                    Ok(*base + p.offset as u64)
                })
                .collect::<Result<Vec<_>>>()?;
            let scalars: Vec<_> = entry
                .i32_scalars
                .iter()
                .map(|name| {
                    if self.live_bounds || bounded128 || bounded512 || bounded1024 {
                        match (node.kernel_id.as_str(),name.as_str()) {
                            ("integration.tp_moe.dynamic","num_tokens")
                            | ("gemm.mxfp8_quant_cute","m")
                            | ("moe.w4a16.topk_sum","active_m") => return live as i32,
                            ("gemm.mxfp8_quant_cute","grid_x") => return (live*10) as i32,
                            ("moe.w4a8.compact_micro","num_pairs") => return (live*6) as i32,
                            _ => (),
                        }
                    }
                    node.scalars[name]
                })
                .collect();
            unsafe {
                primitives
                    .launch(node.entry, &pointers, &scalars, stream)
            }?;
        }
        Ok(())
    }
}

// Isolated component access; no serving execution path calls this helper.
impl Workspace {
    pub(crate) fn chunk_route_region(&self,stream:&Stream)->Result<u64> {
        self.guard_stream(stream)?;
        let c=&self.contract;
        ensure!(std::env::var("CANADA_SCRATCH_FFN_CHUNK_SCREEN").as_deref()==Ok("1")
            &&c.kernel_library_sha256=="0558b822898c6c305ccc276bd8ebbd03d1f5c2c75276deaf9a91d46839df5c0a"
            &&(c.rows,c.hidden,c.topk,c.scratch_bytes)==(6,5120,6,4118728),"isolated FC2 route storage");
        let nodes:Vec<_>=c.nodes.iter().filter(|n|n.kernel_id=="moe.w4a16.topk_sum").collect();
        ensure!(nodes.len()==1,"unique FC2 route consumer");
        let region=nodes[0].pointers.get("fc2_ptr").context("FC2 route operand")?;
        ensure!(region.role=="scratch"&&region.offset==11728
            &&region.offset+6*6*5120*2<=self.scratch.len(),"FC2 route storage bounds");
        Ok(self.scratch.ptr()+region.offset as u64)
    }
}
