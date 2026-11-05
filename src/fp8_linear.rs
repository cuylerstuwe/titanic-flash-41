//! Rust-owned block-MXFP8 projection, workspace and explicit launch binding.
//! Consumes low-level AOT kernels; no captured Python plan is used at runtime.
use crate::{
    blockscaled::{Matrix, scale_bytes},
    cuda::{Buffer, Cuda, Stream},
    primitives::{Entry, Primitives},
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde::Deserialize;
use std::{cell::Cell, collections::BTreeMap, path::Path, rc::Rc};
#[path = "fp8_paired.rs"]
mod paired;
pub(crate) use paired::PairedSharedUp;
pub use paired::native_probe as paired_shared_up_probe;
#[path = "fp8_paired_qkv.rs"]
mod paired_qkv;
pub(crate) use paired_qkv::PairedQkv;
pub use paired_qkv::native_probe as paired_qkv_probe;
#[path = "fp8_prepared_dense.rs"]
mod prepared_dense;
#[derive(Clone,Copy)]
struct PrefillRows { live:usize, capacity:usize, stream:usize, dense:bool, experts:bool }
thread_local! {
    // Enqueue/capture only: graph executables retain their recorded row bound.
    // Restrict this to the owning stream and matching ABI; unrelated profiles and
    // separately replayed 128-row decoder tails keep their complete extent.
    static LIVE_PREFILL_ROWS: Cell<Option<PrefillRows>> = const { Cell::new(None) };
}
pub(crate) struct LivePrefillRowsScope(std::marker::PhantomData<Rc<()>>);
impl LivePrefillRowsScope {
    pub(crate) fn new(dense:bool,experts:bool,live:usize,capacity:usize,stream:&Stream)->Result<Self> {
        ensure!((1..=capacity).contains(&live),"dense live-row scope extent");
        LIVE_PREFILL_ROWS.with(|slot|->Result<()> {
            ensure!(slot.get().is_none(),"nested dense live-row scope");
            if dense || experts {
                ensure!((capacity==128 && (17..128).contains(&live))
                    || (capacity==512 && (129..512).contains(&live))
                    || (capacity==1024 && (513..1024).contains(&live)),
                    "qualified short or partial prefill row geometry");
                slot.set(Some(PrefillRows{live,capacity,stream:stream.raw()as usize,dense,experts}));
            }
            Ok(())
        })?;
        Ok(Self(std::marker::PhantomData))
    }
}
impl Drop for LivePrefillRowsScope {
    fn drop(&mut self){LIVE_PREFILL_ROWS.with(|slot|slot.set(None));}
}
pub(crate) fn prefill_expert_rows(capacity:usize,stream:&Stream)->Option<usize> {
    LIVE_PREFILL_ROWS.with(|slot|slot.get().filter(|p|p.experts && p.capacity==capacity && p.stream==stream.raw()as usize)
        .map(|p|p.live))
}
#[derive(Deserialize)]
struct Quantizer {
    subgroup_width: usize,
    threads: usize,
    grid_ctas_per_sm: usize,
}
#[derive(Deserialize)]
struct Geometry {
    rows: usize,
    in_features: usize,
    out_features: usize,
    block_size: usize,
    quantizer: Option<Quantizer>,
    output_layout: String,
    quantized_c: bool,
    #[serde(default)]
    dense_policy: Option<DensePolicy>,
    #[serde(default)]
    flashinfer: Option<FlashGeometry>,
}
#[derive(Deserialize)]
struct FlashGeometry {
    scale_a_bytes: usize,
    scale_b_bytes: usize,
    mma_tiler_mn: [usize; 2],
    swap_ab: bool,
    tile_k: usize,
    max_active_clusters: usize,
    use_prefetch: bool,
    quantizer_2t: bool,
    quantizer_grid: usize,
}
#[derive(Clone, Copy, Default, Deserialize)]
#[serde(tag = "kind")]
enum Backend {
    #[default]
    #[serde(rename = "b12x_words_v1")]
    B12x,
    #[serde(rename = "flashinfer_static_words_v1")]
    FlashInfer { enable_pdl: bool },
}
#[derive(Deserialize)]
struct DensePolicy {
    split_k_slices: usize,
    split_k_atomic_bf16: bool,
    output_clear: bool,
}
#[derive(Deserialize)]
struct Manifest {
    geometry: Geometry,
    #[serde(default)]
    backend: Backend,
}
pub(crate) fn pdl_enabled() -> bool {
    std::env::var("CANADA_SCRATCH_FP8_PDL").as_deref() == Ok("1")
}
pub(crate) fn narrow_enabled() -> bool {
    std::env::var("CANADA_SCRATCH_FP8_NARROW").as_deref() == Ok("1")
}
pub(crate) fn output_projection_probe(rows:usize)->bool {
    (rows==6 && std::env::var("CANADA_SCRATCH_WO_B_PDL_SCREEN").as_deref()==Ok("1"))
        || ([6,16].contains(&rows)
            &&std::env::var("CANADA_SCRATCH_OUTPUT_PROJECTION_SCREEN").as_deref()==Ok("1"))
}
pub struct Linear {
    cuda: Cuda,
    stream: Cell<usize>,
    matrix: Rc<Matrix>,
    primitives: Primitives,
    geometry: Geometry,
    backend: Backend,
    quant: usize,
    dense: usize,
    prepared_dense: Option<prepared_dense::PreparedDense>,
    values: Buffer<u8>,
    scale_rows: Option<Buffer<u8>>,
    scale_mma: Buffer<u8>,
    alpha: Buffer<f32>,
    output: Buffer<bf16>,
}
/// One row shard, aligned to complete 128-row MMA scale tiles. Concatenating
/// four such buffers in rank order produces the existing full-row operands.
pub(crate) struct RowShard {
    rows: usize,
    values: Buffer<u8>,
    scale_rows: Buffer<u8>,
    scale_mma: Buffer<u8>,
    projection: Option<Linear>,
}
fn find(entries: &[Entry], id: &str) -> Result<usize> {
    let matches: Vec<_> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.kernel_id == id)
        .collect();
    ensure!(matches.len() == 1, "expected exactly one {id} primitive");
    Ok(matches[0].0)
}
fn bind(entry: &Entry, values: &BTreeMap<&str, (u64, &str)>) -> Result<Vec<u64>> {
    ensure!(
        entry.arguments.len() == values.len(),
        "projection ABI parameter count changed"
    );
    entry
        .arguments
        .iter()
        .map(|a| {
            let (v, kind) = values
                .get(a.name.as_str())
                .with_context(|| format!("unknown projection parameter {}", a.name))?;
            ensure!(a.kind == *kind, "projection parameter type mismatch");
            Ok(*v)
        })
        .collect()
}
fn pipeline_flag(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) => {
            ensure!(value == "0" || value == "1", "{name} must be 0 or 1");
            Ok(value == "1")
        }
        Err(std::env::VarError::NotPresent) => Ok(true),
        Err(error) => Err(error.into()),
    }
}
fn pipeline_fenced_root(root: &Path) -> Result<Option<std::path::PathBuf>> {
    // s894 traced a bad output exactly to a future ring-buffer fragment.
    // Fences/join before TMA reuse preserve the quantizer, MMA and precision.
    // Legacy switches are explicit controls for bounded diagnostics only.
    let enabled = pipeline_flag("CANADA_SCRATCH_DENSE_PIPELINE_FENCE")?;
    let name = root.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let enabled = enabled && (name != "fp8-shared-w13-r0-m6-v1"
        || pipeline_flag("CANADA_SCRATCH_SHARED_PIPELINE_FENCE")?);
    if !enabled { return Ok(None); }
    // Qualified only for these two QKV schedules. Preserve the explicit
    // consumer control for paired diagnostics; other dense fences stay put.
    if matches!(name,"fp8-split-qkv-m6-v1"|"fp8-split-qkv-m16-v1")
        &&!pipeline_flag("CANADA_SCRATCH_QKV_PRODUCER_FENCE")? {
        return Ok(Some(root.with_file_name(format!("{name}-pipeline-fence-v1"))));
    }
    static BUNDLES: std::sync::OnceLock<BTreeMap<String, String>> = std::sync::OnceLock::new();
    let bundles = BUNDLES.get_or_init(|| {
        serde_json::from_str(include_str!("dense_pipeline_fence_bundles.json"))
            .expect("compiled dense pipeline fence registry")
    });
    Ok(bundles.get(name).map(|replacement| root.with_file_name(replacement)))
}
impl Linear {
    pub fn new(cuda: &Cuda, root: &Path, matrix: Matrix) -> Result<Self> {
        Self::new_shared(cuda, root, Rc::new(matrix))
    }
    fn new_shared(cuda: &Cuda, root: &Path, matrix: Rc<Matrix>) -> Result<Self> {
        // s589: the same canonical M16 matrices with smaller, exact-output
        // dense tiles. Select only these three construction-time bundles.
        let compact_root=if std::env::var("CANADA_SCRATCH_VERIFY16_COMPACT_DENSE").as_deref()==Ok("1") {
            let kind=match root.file_name().and_then(|n|n.to_str()) {
                Some("fp8-canonical-wq_b-r0-m16-v0")=>Some("q_b"),
                Some("fp8-shared-w13-r0-m16-v1")=>Some("w13"),
                Some("fp8-split-qkv-m16-v1") if std::env::var("CANADA_SCRATCH_VERIFY16_LEGACY_QKV").as_deref()!=Ok("1")=>Some("qkv"),
                _=>None,
            };
            kind.map(|kind|root.with_file_name(format!("fp8-flashinfer-compact-{kind}-m16-v1")))
        }else{None};
        if std::env::var("CANADA_SCRATCH_VERIFY16_LEGACY_QKV").as_deref()==Ok("1") {
            let serving=std::env::var("CANADA_SCRATCH_GROUPED_SPARSE16").as_deref()==Ok("1")
                && (std::env::var("CANADA_SCRATCH_ATTENTION16_OVERLAP").as_deref()==Ok("0")
                    || crate::mhc::compact16_overlap_enabled()?)
                && std::env::var("CANADA_SCRATCH_VERIFY16_COMPACT_DENSE").as_deref()==Ok("1");
            ensure!((std::env::var("CANADA_SCRATCH_GROUPED_SPARSE16_SCREEN").as_deref()==Ok("1") || serving)
                && std::env::var_os("CANADA_SCRATCH_VERIFY16_QKV_NO_PDL").is_none(),
                "legacy QKV requires its isolated probe or the guarded compact serving candidate");
        }
        let root=compact_root.as_deref().unwrap_or(root);
        Self::load_shared(cuda,root,matrix)
    }
    fn load_shared(cuda:&Cuda,root:&Path,matrix:Rc<Matrix>)->Result<Self> {
        // Resolve after compact/backend selection so every qualified dense
        // path, including direct diagnostic loads, gets its ordering fix.
        // Missing artifacts fail normally; never silently use an unfenced one.
        let fenced_root = pipeline_fenced_root(root)?;
        let root = fenced_root.as_deref().unwrap_or(root);
        let mut m: Manifest = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        if std::env::var("CANADA_SCRATCH_VERIFY16_QKV_NO_PDL").as_deref()==Ok("1")
            && m.geometry.rows==16 && (matrix.n,matrix.k)==(448,5120) {
            ensure!(std::env::var("CANADA_SCRATCH_GROUPED_SPARSE16_SCREEN").as_deref()==Ok("1")
                && matches!(m.backend,Backend::FlashInfer{..}),
                "QKV launch-dependency isolation requires the compact16 probe");
            m.backend=Backend::FlashInfer{enable_pdl:false};
        }
        // Isolate PDL from the backend/tiling choice in the bounded schedule
        // screen. Both entries already belong to the same qualified bundle.
        if std::env::var("CANADA_SCRATCH_PROBE_PDL_OFF").as_deref()==Ok("1") {
            ensure!(std::env::var("CANADA_SCRATCH_SCHEDULE_ABLATION").as_deref()==Ok("1"),
                "PDL override is restricted to the schedule probe");
            if matrix.n==8192 && matrix.k==1280 && m.geometry.rows==6 {
                ensure!(matches!(m.backend,Backend::FlashInfer{..}),"PDL ablation needs the same FlashInfer backend");
                m.backend=Backend::FlashInfer{enable_pdl:false};
            }
        }
        let g = m.geometry;
        ensure!(
            g.rows > 0
                && g.rows <= 65536
                && g.in_features == matrix.k
                && g.out_features == matrix.n
                && g.block_size == 32,
            "projection geometry differs from AOT lowering"
        );
        ensure!(
            g.output_layout == "row_major_bf16" && !g.quantized_c,
            "unsupported dense output contract"
        );
        if let Some(p) = &g.dense_policy {
            ensure!(
                [1, 2, 4].contains(&p.split_k_slices)
                    && (p.split_k_slices == 1 || p.split_k_atomic_bf16)
                    && p.output_clear == (p.split_k_slices > 1 && p.split_k_atomic_bf16),
                "unsupported dense reduction/clear contract"
            );
        } else {
            ensure!(
                [
                    (1280, 5120),
                    (512, 5120),
                    (1792, 5120),
                    (448, 5120),
                    (8192, 1280),
                    (5120, 2048),
                    (1152, 5120),
                    (5120, 576)
                ]
                .contains(&(matrix.n, matrix.k)),
                "unqualified legacy dense bundle lacks explicit reduction/clear metadata: M={} N={} K={}",
                g.rows,
                matrix.n,
                matrix.k
            );
        }
        match m.backend {
            Backend::B12x => {
                let q = g
                    .quantizer
                    .as_ref()
                    .context("legacy quantizer metadata absent")?;
                ensure!(
                    (q.subgroup_width, q.threads) == if g.rows <= 8 { (8, 128) } else { (4, 256) }
                        && q.grid_ctas_per_sm == 4
                        && g.flashinfer.is_none(),
                    "unsupported legacy quantizer lowering"
                );
            }
            Backend::FlashInfer { .. } => {
                let f = g
                    .flashinfer
                    .as_ref()
                    .context("FlashInfer geometry absent")?;
                let p = g
                    .dense_policy
                    .as_ref()
                    .context("FlashInfer dense policy absent")?;
                // Admit only the actual GPU-qualified static bundle. New shapes
                // need their own lowering/reference and owned-buffer proof.
                ensure!(
                    [6,16].contains(&g.rows)
                        && ((matrix.n == 8192 && matrix.k == 1280
                            && f.mma_tiler_mn == [16, 128] && f.quantizer_grid == 22)
                            || ([448, 1152].contains(&matrix.n) && matrix.k == 5120
                                && f.mma_tiler_mn == [16, 64] && f.quantizer_grid == 128)
                            || (g.rows == 6 && matrix.n == 1280 && matrix.k == 15360
                                && f.mma_tiler_mn == [16, 64] && f.quantizer_grid == 128)
                            || ([6,16].contains(&g.rows) && matrix.n == 5120 && matrix.k == 2048
                                && f.mma_tiler_mn == [16, 128] && f.quantizer_grid == 32))
                        && matrix.padded_k == matrix.k
                        && g.quantizer.is_none()
                        && f.scale_a_bytes == scale_bytes(g.rows, matrix.k)?
                        && f.scale_b_bytes == scale_bytes(matrix.n, matrix.k)?
                        && !f.swap_ab
                        && f.tile_k == 128
                        && f.max_active_clusters == 48
                        && !f.use_prefetch
                        && !f.quantizer_2t
                        && p.split_k_slices == 1
                        && !p.split_k_atomic_bf16
                        && !p.output_clear,
                    "unqualified FlashInfer static projection geometry"
                );
            }
        }
        let primitives = Primitives::open(
            cuda,
            &root.join("libcanada_primitives.so"),
            &root.join("bindings.json"),
        )?;
        ensure!(
            primitives.bindings.abi == 2,
            "unsupported projection primitive ABI"
        );
        let entries = &primitives.bindings.entries;
        let (quant, dense) = match m.backend {
            Backend::B12x => {
                ensure!(entries.len() == 2, "unsupported legacy projection bundle");
                (
                    find(entries, "gemm.mxfp8_quant_cute")?,
                    find(entries, "gemm.dense")?,
                )
            }
            Backend::FlashInfer { enable_pdl } => {
                ensure!(
                    entries.len() == 4
                        && primitives.bindings.source
                            == "https://github.com/flashinfer-ai/flashinfer",
                    "unsupported FlashInfer projection bundle"
                );
                for mode in [0, 1] {
                    find(entries, &format!("flashinfer.mxfp8.quant.pdl{mode}"))?;
                    find(entries, &format!("flashinfer.mxfp8.dense.pdl{mode}"))?;
                }
                let mode = usize::from(enable_pdl);
                (
                    find(entries, &format!("flashinfer.mxfp8.quant.pdl{mode}"))?,
                    find(entries, &format!("flashinfer.mxfp8.dense.pdl{mode}"))?,
                )
            }
        };
        // A padded-K operand relies on its zeroed pad columns/scales persisting,
        // so only unpadded quantized inputs may alias other wide layers.
        let input_storage = |len: usize| -> Result<Buffer<u8>> {
            if matrix.padded_k == matrix.k {
                cuda.allocate_rows(len)
            } else {
                cuda.allocate(len)
            }
        };
        let mut values = input_storage(g.rows * matrix.padded_k)?;
        values.zero()?;
        let scale_rows = if matches!(m.backend, Backend::B12x) {
            let mut rows = input_storage(g.rows * matrix.padded_k / 32)?;
            rows.zero()?;
            Some(rows)
        } else {
            None
        };
        // Narrow profiles leave padded MMA row lanes initialized. Preserve
        // them even when the active rows borrow a prefill transient arena.
        let mut scale_mma = if g.rows<=128 {
            cuda.allocate(scale_bytes(g.rows, matrix.padded_k)?)?
        }else{input_storage(scale_bytes(g.rows, matrix.padded_k)?)?};
        scale_mma.zero()?;
        let mut alpha = cuda.allocate::<f32>(1)?;
        alpha.upload(&[1.])?;
        let mut output = cuda.allocate_rows::<bf16>(g.rows * matrix.n)?;
        output.zero()?;
        // Only the dense consumer changes. Preserve native padding, quantizer
        // buffers and fused producers. Resolve control at owner construction;
        // replayed graphs contain no flag lookup or host dispatch.
        let prepared_dense=if matches!(g.rows,6|16)
            &&(matrix.n,matrix.k,matrix.padded_k)==(5120,576,640) {
            let enabled=match std::env::var("CANADA_SCRATCH_SHARED_DOWN_FLASH") {
                Ok(v)=>{ensure!(v=="0"||v=="1","prepared shared down flag");v=="1"},
                // Native-producer M6 passed two all-rank screens: short-live
                // verification improved, full-live was essentially flat.
                // M16 showed no meaningful model gain and stays opt-in.
                Err(std::env::VarError::NotPresent)=>g.rows==6&&matches!(m.backend,Backend::B12x),
                Err(e)=>return Err(e.into()),
            };
            if enabled||std::env::var("CANADA_SCRATCH_SHARED_DOWN_FLASH_SCREEN").as_deref()==Ok("1") {
                ensure!(matches!(m.backend,Backend::B12x)
                    // The qualified legacy M6 bundle predates explicit policy
                    // metadata; load_shared already admits its known shape.
                    // Its exact padded operands/outputs passed the M6 screen.
                    &&g.dense_policy.as_ref().map_or(g.rows==6,
                        |p|p.split_k_slices==1&&!p.split_k_atomic_bf16&&!p.output_clear),
                    "prepared shared down needs the unchanged native producer");
                Some(prepared_dense::PreparedDense::open(cuda,root.parent().context("kernel parent")?,g.rows,enabled)?)
            }else{None}
        }else if g.rows==16&&(matrix.n,matrix.k,matrix.padded_k)==(4096,1280,1280)
            &&root.file_name().and_then(|n|n.to_str()).is_some_and(|n|n.starts_with("fp8-index-q-m16-v1")) {
            let enabled=match std::env::var("CANADA_SCRATCH_INDEX_QUERY_FLASH") {
                Ok(v)=>{ensure!(v=="0"||v=="1","prepared index query flag");v=="1"},
                Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
            };
            if enabled||std::env::var("CANADA_SCRATCH_INDEX_QUERY_FLASH_SCREEN").as_deref()==Ok("1") {
                ensure!(matches!(m.backend,Backend::B12x),"index query needs its unchanged native producer");
                Some(prepared_dense::PreparedDense::open_index_query(cuda,root.parent().context("kernel parent")?,enabled)?)
            }else{None}
        }else{None};
        Ok(Self {
            cuda: cuda.clone(),
            stream: Cell::new(0),
            matrix,
            primitives,
            geometry: g,
            backend: m.backend,
            quant,
            dense,
            prepared_dense,
            values,
            scale_rows,
            scale_mma,
            alpha,
            output,
        })
    }
    pub(crate) fn narrow_projection(&self) -> bool {
        matches!(self.backend, Backend::FlashInfer { .. })
            && self.geometry.flashinfer.as_ref().is_some_and(|f| f.mma_tiler_mn == [16, 64])
            && [448, 1152].contains(&self.matrix.n) && self.matrix.k == 5120
    }
    pub(crate) fn legacy_qkv_for_probe(&self,kernels:&Path)->Result<Self> {
        ensure!(std::env::var("CANADA_SCRATCH_QKV_SCHEDULE_SCREEN").as_deref()==Ok("1")
            && self.geometry.rows==16 && (self.matrix.n,self.matrix.k)==(448,5120)
            && self.narrow_projection(),"alternate QKV requires the isolated compact16 comparison");
        let result=Self::load_shared(&self.cuda,&kernels.join("fp8-split-qkv-m16-v1"),self.matrix.clone())?;
        ensure!(matches!(result.backend,Backend::B12x) && result.geometry.rows==16,"legacy QKV geometry");
        Ok(result)
    }
    pub(crate) fn swap_qkv_for_probe(&mut self,other:&mut Self)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_QKV_SCHEDULE_SCREEN").as_deref()==Ok("1")
            && self.geometry.rows==16 && other.geometry.rows==16
            && (self.matrix.n,self.matrix.k)==(448,5120) && Rc::ptr_eq(&self.matrix,&other.matrix)
            && self.narrow_projection()!=other.narrow_projection(),"alternate QKV identity");
        std::mem::swap(self,other);Ok(())
    }
    pub(crate) fn producer_qkv_for_probe(&self,kernels:&Path)->Result<Self> {
        ensure!(std::env::var("CANADA_SCRATCH_QKV_FENCE_SCREEN").as_deref()==Ok("1")
            && matches!(self.geometry.rows,6|16) && (self.matrix.n,self.matrix.k)==(448,5120)
            && matches!(self.backend,Backend::B12x),"isolated QKV producer fence geometry");
        let result=Self::load_shared(&self.cuda,&kernels.join(format!(
            "fp8-split-qkv-m{}-v1-producer-fence-v1",self.geometry.rows)),self.matrix.clone())?;
        ensure!(matches!(result.backend,Backend::B12x)&&result.geometry.rows==self.geometry.rows,
            "QKV fence alternate changed backend/rows");Ok(result)
    }
    pub(crate) fn swap_qkv_fence_for_probe(&mut self,other:&mut Self)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_QKV_FENCE_SCREEN").as_deref()==Ok("1")
            && matches!(self.geometry.rows,6|16)&&self.geometry.rows==other.geometry.rows
            && matches!(self.backend,Backend::B12x)&&matches!(other.backend,Backend::B12x)
            && (self.matrix.n,self.matrix.k)==(448,5120)&&Rc::ptr_eq(&self.matrix,&other.matrix),
            "QKV fence alternate canonical identity");
        // Caller synchronizes and destroys referencing graphs first. Keep the
        // alternate alive so neither library nor workspace can be released early.
        std::mem::swap(self,other);Ok(())
    }
    pub(crate) fn legacy_shared_up_for_probe(&self,kernels:&Path)->Result<Self> {
        ensure!(std::env::var("CANADA_SCRATCH_LEGACY16_FFN_W13").as_deref()==Ok("1")
            && self.geometry.rows==16 && (self.matrix.n,self.matrix.k)==(1152,5120)
            && self.narrow_projection(),"alternate W13 requires the isolated compact16 comparison");
        let result=Self::load_shared(&self.cuda,&kernels.join("fp8-shared-w13-r0-m16-v1"),self.matrix.clone())?;
        ensure!(matches!(result.backend,Backend::B12x) && result.geometry.rows==16,"legacy W13 geometry");
        Ok(result)
    }
    pub(crate) fn swap_shared_up_for_probe(&mut self,other:&mut Self)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_LEGACY16_FFN_W13").as_deref()==Ok("1")
            && self.geometry.rows==16 && other.geometry.rows==16
            && (self.matrix.n,self.matrix.k)==(1152,5120) && Rc::ptr_eq(&self.matrix,&other.matrix)
            && self.narrow_projection()!=other.narrow_projection(),"alternate W13 identity");
        std::mem::swap(self,other);Ok(())
    }
    pub(crate) fn pdl_active(&self)->bool {
        matches!(self.backend,Backend::FlashInfer{enable_pdl:true})
    }
    pub(crate) fn wo_b_mode(&self)->u8 {
        match self.backend {Backend::B12x=>0,Backend::FlashInfer{enable_pdl}=>1+u8::from(enable_pdl)}
    }
    pub(crate) fn set_wo_b_dependency_for_probe(&mut self,enabled:bool)->Result<()> {
        ensure!(output_projection_probe(self.geometry.rows)
            &&(self.matrix.n,self.matrix.k)==(5120,2048)
            &&matches!(self.backend,Backend::FlashInfer{..}),"WO-B dependency isolation");
        let mode=usize::from(enabled);let entries=&self.primitives.bindings.entries;
        self.quant=find(entries,&format!("flashinfer.mxfp8.quant.pdl{mode}"))?;
        self.dense=find(entries,&format!("flashinfer.mxfp8.dense.pdl{mode}"))?;
        self.backend=Backend::FlashInfer{enable_pdl:enabled};Ok(())
    }
    pub(crate) fn wo_b_pdl_for_probe(&self,kernels:&Path)->Result<Self> {
        ensure!(output_projection_probe(self.geometry.rows)
            &&(self.matrix.n,self.matrix.k,self.matrix.padded_k)==(5120,2048,2048)
            &&matches!(self.backend,Backend::B12x),"alternate WO-B geometry");
        let bundle=if std::env::var("CANADA_SCRATCH_OUTPUT_PROJECTION_SCREEN").as_deref()==Ok("1") {
            format!("fp8-flashinfer-wo-b-m{}-fenced-20261102",self.geometry.rows)
        }else{"fp8-flashinfer-wo-b-m6-s809".to_owned()};
        let other=Self::load_shared(&self.cuda,&kernels.join(bundle),self.matrix.clone())?;
        ensure!(other.pdl_active(),"WO-B alternate must be the qualified PDL bundle");Ok(other)
    }
    pub(crate) fn swap_wo_b_backend_for_probe(&mut self,other:&mut Self)->Result<()> {
        ensure!(output_projection_probe(self.geometry.rows)
            &&other.geometry.rows==self.geometry.rows
            &&(self.matrix.n,self.matrix.k,self.matrix.padded_k)==(5120,2048,2048)
            &&Rc::ptr_eq(&self.matrix,&other.matrix)&&(self.wo_b_mode()==0)!=(other.wo_b_mode()==0)
            &&self.values.len()==other.values.len()&&self.scale_mma.len()==other.scale_mma.len()
            &&self.output.len()==other.output.len()
            &&[&self.geometry,&other.geometry].iter().all(|g|g.dense_policy.as_ref().is_none_or(|p|!p.output_clear)),
            "WO-B backend swap changes operand layout or owner");
        // The target synchronized and discarded all graphs first. Keep every
        // live input/output allocation stable; only immutable kernel bindings
        // and their ABI/geometry move. The M6 WO-A producer owns quantization.
        std::mem::swap(&mut self.primitives,&mut other.primitives);
        std::mem::swap(&mut self.geometry,&mut other.geometry);
        std::mem::swap(&mut self.backend,&mut other.backend);
        std::mem::swap(&mut self.quant,&mut other.quant);
        std::mem::swap(&mut self.dense,&mut other.dense);Ok(())
    }
    pub fn quantized_input(&self) -> (&Buffer<u8>, &Buffer<u8>) {
        (&self.values, &self.scale_mma)
    }
    pub(crate) fn prepare_row_shard(&self, projection_root: Option<&Path>) -> Result<RowShard> {
        ensure!(matches!(self.backend, Backend::B12x)
            && self.matrix.n == 1792 && self.matrix.k == 5120
            && self.matrix.padded_k == self.matrix.k
            && crate::config::wide_rows(self.geometry.rows)
            && self.geometry.rows % 512 == 0,
            "quantized row gather requires an aligned wide fused QKV projection");
        let rows = self.geometry.rows / 4;
        let projection = projection_root.map(|root| Self::new_shared(&self.cuda, root, self.matrix.clone())).transpose()?;
        if let Some(local) = &projection {
            ensure!([8192,16384].contains(&self.geometry.rows)
                && local.geometry.rows == rows && matches!(local.backend, Backend::B12x)
                && !local.geometry.dense_policy.as_ref().is_some_and(|p| p.output_clear),
                "local QKV projection requires the screened row geometry and reduction");
        }
        Ok(RowShard {
            rows,
            values: self.cuda.allocate_rows(rows * self.matrix.k)?,
            scale_rows: self.cuda.allocate_rows(rows * self.matrix.k / 32)?,
            scale_mma: self.cuda.allocate_rows(scale_bytes(rows, self.matrix.k)?)?,
            projection,
        })
    }
    /// Move the existing per-row quantization before the all-gather. The
    /// projection, scale rule and BF16 input boundary are unchanged.
    pub(crate) fn run_row_shard(&mut self, input: &Buffer<bf16>, shard: &mut RowShard,
        project_local: bool, comm: &crate::roce::Roce, stream: &Stream) -> Result<()> {
        self.guard_stream(stream)?;
        ensure!(matches!(self.backend, Backend::B12x)
            && self.matrix.n == 1792 && self.matrix.k == self.matrix.padded_k
            && shard.rows % 128 == 0 && shard.rows * 4 == self.geometry.rows
            && input.len() == shard.rows * self.matrix.k
            && shard.values.len() * 4 == self.values.len()
            && shard.scale_mma.len() * 4 == self.scale_mma.len()
            && self.cuda.owns_buffer(input) && self.cuda.owns_buffer(&shard.values)
            && self.cuda.owns_buffer(&shard.scale_rows) && self.cuda.owns_buffer(&shard.scale_mma),
            "quantized row gather geometry/context");
        if project_local {
            let local = shard.projection.as_mut().context("local QKV projection absent")?;
            local.run(input, stream)?;
            // Every row uses the same replicated canonical matrix. Gather
            // complete BF16 output rows before normalization or cache writes.
            return unsafe { comm.gather_rows(local.output(), &mut self.output, stream) };
        }
        self.quantize_legacy(input.ptr(), shard.rows, shard.values.ptr(),
            shard.scale_rows.ptr(), shard.scale_mma.ptr(), stream)?;
        // Each shard contains complete row tiles, so the swizzled scale bytes
        // already have their final order. No scale transpose or requantization.
        unsafe {
            comm.gather_rows(&shard.values, &mut self.values, stream)?;
            comm.gather_rows(&shard.scale_mma, &mut self.scale_mma, stream)?;
            self.launch_dense(self.values.ptr(), self.scale_mma.ptr(), stream)
        }
    }
    /// Existing owned input storage for the qualified six-row shared producer.
    /// Padding and unused scale lanes retain the constructor's zero fill.
    pub(crate) fn shared_quantization_buffers(&self) -> Result<(&Buffer<u8>, &Buffer<u8>, &Buffer<u8>)> {
        ensure!(matches!(self.backend, Backend::B12x)
            && (self.geometry.rows, self.matrix.n, self.matrix.k, self.matrix.padded_k)
                == (6, 5120, 576, 640), "fused shared quantization profile");
        Ok((&self.values, self.scale_rows.as_ref().context("shared row scales absent")?, &self.scale_mma))
    }
    pub(crate) fn shared_quantization_buffers16(&self) -> Result<(&Buffer<u8>, &Buffer<u8>, &Buffer<u8>)> {
        ensure!(matches!(self.backend,Backend::B12x)
            && (self.geometry.rows,self.matrix.n,self.matrix.k,self.matrix.padded_k)==(16,5120,576,640)
            && crate::ffn::shared_quant16_enabled()?,
            "M16 fused shared quantization geometry");
        Ok((&self.values,self.scale_rows.as_ref().context("shared row scales absent")?,&self.scale_mma))
    }
    /// Fused query RMS writes the existing Q-B operands. Preserve the zeroed
    /// inactive scale lanes and the storage used by the compressed indexer.
    pub(crate) fn query_quantization_buffers(&self) -> Result<(&Buffer<u8>, &Buffer<u8>)> {
        ensure!(matches!(self.backend, Backend::FlashInfer { enable_pdl: true })
            && [6,16].contains(&self.geometry.rows)
            && (self.matrix.n,self.matrix.k,self.matrix.padded_k)==(8192,1280,1280),
            "fused query quantization requires the qualified FlashInfer Q-B profile");
        Ok((&self.values,&self.scale_mma))
    }
    pub(crate) fn immutable_matrix(&self) -> Rc<Matrix> {
        self.matrix.clone()
    }
    /// The row-packed FFN codec reconstructs this exact existing quantizer's
    /// full-row operands, including its minimum-amax rounding corrections.
    pub(crate) fn wide_shared_quantization_buffers(&self) -> Result<(&Buffer<u8>, &Buffer<u8>, &Buffer<u8>)> {
        ensure!(matches!(self.backend, Backend::B12x)
            && [8192,16384].contains(&self.geometry.rows)
            && (self.matrix.n,self.matrix.k,self.matrix.padded_k) == (1152,5120,5120),
            "packed shared input projection geometry");
        Ok((&self.values,self.scale_rows.as_ref().context("shared row scales absent")?,&self.scale_mma))
    }
    pub fn output(&self) -> &Buffer<bf16> {
        &self.output
    }
    pub fn run(&self, input: &Buffer<bf16>, stream: &Stream) -> Result<()> {
        ensure!(
            input.len() == self.geometry.rows * self.matrix.k && self.cuda.owns_buffer(input),
            "projection input shape mismatch"
        );
        // All pointers are owned by input/self and dimensions are checked above.
        unsafe { self.launch_pointer(input.ptr(), stream) }
    }
    /// Consume a GPU producer's MXFP8 values/scales without launching another
    /// quantizer. The producer must use the same BF16 rounding and MMA layout.
    pub fn run_quantized(
        &self,
        values: &Buffer<u8>,
        scales: &Buffer<u8>,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            self.cuda.owns_buffer(values)
                && self.cuda.owns_buffer(scales)
                && values.len() == self.geometry.rows * self.matrix.padded_k
                && scales.len() == scale_bytes(self.geometry.rows, self.matrix.padded_k)?,
            "prequantized projection geometry mismatch"
        );
        unsafe { self.launch_dense(values.ptr(), scales.ptr(), stream) }
    }
    fn guard_stream(&self, stream: &Stream) -> Result<()> {
        ensure!(
            self.cuda.owns_stream(stream),
            "projection stream context mismatch"
        );
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == raw,
            "projection workspace needs one ordered stream"
        );
        self.stream.set(raw);
        Ok(())
    }
    /// Share immutable WKV weights while keeping auxiliary-stream quantization
    /// and output storage independent of the ordinary captured graph.
    pub(crate) fn fork_engram_projection_for_probe(&self,kernels:&Path)->Result<Self> {
        ensure!((std::env::var("CANADA_SCRATCH_ENGRAM_PROJECTION_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_ENGRAM_PROJECTION_M3").as_deref()==Ok("1"))
            &&matches!(self.geometry.rows,6|16)&&self.geometry.in_features==6144
            &&self.geometry.out_features==6400,"isolated Engram WKV binding");
        self.cuda.without_row_arena(||Self::load_shared(&self.cuda,
            &kernels.join(format!("fp8-engram-wkv-m{}-v2",self.geometry.rows)),self.matrix.clone()))
    }
    /// The isolated full-model schedule screen has synchronized its old parent
    /// (including joined auxiliary work) and destroyed every referencing graph.
    /// Only then may these same owned buffers be rebound for the next capture.
    pub(crate) unsafe fn release_verify16_probe_binding(&mut self) -> Result<()> {
        ensure!(crate::ffn::wide_overlap_probe(self.geometry.rows)
            || ((std::env::var("CANADA_SCRATCH_VERIFY16_OVERLAP_SCREEN").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_GROUPED16_SERIAL_QUERY").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_GROUPED16_SERIAL_ATTENTION").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_QKV_SCHEDULE_SCREEN").as_deref()==Ok("1"))
            && self.geometry.rows==16),"isolated idle wide projection rebind");
        self.stream.set(0);
        Ok(())
    }
    unsafe fn launch_dense(&self, values: u64, scales: u64, stream: &Stream) -> Result<()> {
        self.guard_stream(stream)?;
        if let Some(dense)=self.prepared_dense.as_ref().filter(|d|d.enabled()) {
            return unsafe{dense.launch(values,self.matrix.values(),scales,self.matrix.scales(),
                self.output.ptr(),self.alpha.ptr(),stream)};
        }
        if self
            .geometry
            .dense_policy
            .as_ref()
            .is_some_and(|p| p.output_clear)
        {
            unsafe {
                self.cuda
                    .zero_device(self.output.ptr(), self.output.bytes(), stream)
            }?;
        }
        let dense = match self.backend {
            Backend::FlashInfer { .. } => BTreeMap::from([
                ("x", (values, "pointer")),
                ("w", (self.matrix.values(), "pointer")),
                ("xs", (scales, "pointer")),
                ("ws", (self.matrix.scales(), "pointer")),
                ("y", (self.output.ptr(), "pointer")),
                ("alpha", (self.alpha.ptr(), "pointer")),
            ]),
            Backend::B12x => BTreeMap::from([
                ("a_ptr", (values, "pointer")),
                ("b_ptr", (self.matrix.values(), "pointer")),
                ("sfa_ptr", (scales, "pointer")),
                ("sfb_ptr", (self.matrix.scales(), "pointer")),
                ("c_ptr", (self.output.ptr(), "pointer")),
                ("alpha_ptr", (self.alpha.ptr(), "pointer")),
                ("quant_c_values_ptr", (16, "pointer")),
                ("quant_c_scale_rows_ptr", (16, "pointer")),
                ("quant_c_scale_mma_ptr", (16, "pointer")),
                ("m", (self.effective_rows(stream) as u64, "i32")),
            ]),
        };
        let da = bind(&self.primitives.bindings.entries[self.dense], &dense)?;
        unsafe { self.primitives.launch_words(self.dense as i32, &da, stream) }
    }
    fn effective_rows(&self,stream:&Stream)->usize {
        if [128,512,1024].contains(&self.geometry.rows) && matches!(self.backend,Backend::B12x)
            && self.geometry.dense_policy.as_ref().is_some_and(|p|p.split_k_slices==1 && !p.output_clear) {
            LIVE_PREFILL_ROWS.with(|slot|slot.get().filter(|p|p.dense && p.capacity==self.geometry.rows
                && p.stream==stream.raw()as usize).map_or(self.geometry.rows,|p|p.live))
        }else{self.geometry.rows}
    }
    pub(crate) fn set_shared_down_flash_for_probe(&self,enabled:bool)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_SHARED_DOWN_FLASH_SCREEN").as_deref()==Ok("1")
            &&matches!(self.geometry.rows,6|16)
            &&(self.matrix.n,self.matrix.k,self.matrix.padded_k)==(5120,576,640),
            "isolated prepared shared down consumer");
        self.prepared_dense.as_ref().context("prepared shared down was not loaded")?.set_for_probe(enabled);
        Ok(())
    }
    pub(crate) fn set_index_query_flash_for_probe(&self,enabled:bool)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_INDEX_QUERY_FLASH_SCREEN").as_deref()==Ok("1")
            &&self.geometry.rows==16&&(self.matrix.n,self.matrix.k,self.matrix.padded_k)==(4096,1280,1280),
            "isolated prepared index query consumer");
        self.prepared_dense.as_ref().context("prepared index query was not loaded")?.set_for_probe(enabled);
        Ok(())
    }
    /// Safety: input addresses exactly rows*K BF16 values; keep them alive and
    /// unmodified until the stream/any captured graph completes.
    pub unsafe fn launch_pointer(&self, input: u64, stream: &Stream) -> Result<()> {
        self.guard_stream(stream)?;
        ensure!(
            input != 0 && input % 16 == 0,
            "projection input pointer alignment"
        );
        if matches!(self.backend, Backend::FlashInfer { .. }) {
            let quant = BTreeMap::from([
                ("x", (input, "pointer")),
                ("y", (self.values.ptr(), "pointer")),
                ("scales", (self.scale_mma.ptr(), "pointer")),
            ]);
            let qa = bind(&self.primitives.bindings.entries[self.quant], &quant)?;
            unsafe { self.primitives.launch_words(self.quant as i32, &qa, stream) }?;
            return unsafe { self.launch_dense(self.values.ptr(), self.scale_mma.ptr(), stream) };
        }
        self.quantize_legacy(input, self.effective_rows(stream), self.values.ptr(),
            self.scale_rows.as_ref().context("legacy row scales absent")?.ptr(),
            self.scale_mma.ptr(), stream)?;
        unsafe { self.launch_dense(self.values.ptr(), self.scale_mma.ptr(), stream) }
    }
    fn quantize_legacy(&self, input: u64, rows: usize, values: u64,
        scale_rows: u64, scale_mma: u64, stream: &Stream) -> Result<()> {
        let q = self.geometry.quantizer.as_ref().context("legacy quantizer absent")?;
        let tasks = rows * (self.matrix.padded_k / 32).div_ceil(32 / q.subgroup_width);
        let grid = tasks
            .div_ceil(q.threads / 32)
            .max(1)
            .min(48 * q.grid_ctas_per_sm);
        let quant = BTreeMap::from([
            ("source_ptr", (input, "pointer")),
            ("values_ptr", (values, "pointer")),
            ("scale_rows_ptr", (scale_rows, "pointer")),
            ("scale_mma_ptr", (scale_mma, "pointer")),
            ("m", (rows as u64, "i32")),
            ("source_k", (self.matrix.k as u64, "i32")),
            ("grid_x", (grid as u64, "i32")),
        ]);
        let qa = bind(&self.primitives.bindings.entries[self.quant], &quant)?;
        unsafe { self.primitives.launch_words(self.quant as i32, &qa, stream) }
    }
}
