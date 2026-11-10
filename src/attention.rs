//! Rust-owned DSV4 paged KV storage and sparse-decode workspace.
//! Numeric execution is a pinned low-level CUDA dependency, with no model,
//! TensorView, Torch or Python dependency. Indices are validated control data.
use crate::cuda::{Buffer, Cuda, Stream};
use anyhow::{Result, ensure};
use half::bf16;
use libloading::Library;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::Cell, ffi::c_void, path::Path, rc::Rc};

pub const HEADS: usize = 16;
pub const WIDTH: usize = 512;
pub const PAGE_TOKENS: usize = 64;
/// Paged KV byte layouts (SGLang's `KVLayout`). A page is `tokens` data rows followed by as many scale rows,
/// padded to the reader's row stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvLayout {
    /// 448 e4m3 + 64 bf16 RoPE, then 7 ue8m0 scales + 1 pad: 584 B/token, 576-aligned. Today's target caches and
    /// always the drafter's.
    V4,
    /// 512 e4m3 (RoPE dims quantized too), then 16 ue8m0 scales: 528 B/token, 512-aligned. model.py's window cache.
    V41,
    /// 512 e2m1 (two per byte, even element in the low nibble), then 32 e4m3 scales: 288 B/token, 256-aligned.
    /// model.py's compressed cache.
    V41Fp4,
}
impl KvLayout {
    pub const fn bytes_per_token(self) -> usize {
        match self { Self::V4 => 584, Self::V41 => 528, Self::V41Fp4 => 288 }
    }
    pub const fn align(self) -> usize {
        match self { Self::V4 => 576, Self::V41 => 512, Self::V41Fp4 => 256 }
    }
    pub const fn name(self) -> &'static str {
        match self { Self::V4 => "v4", Self::V41 => "v41", Self::V41Fp4 => "v41_fp4" }
    }
}
pub fn layout_page_bytes(layout: KvLayout, tokens: usize) -> Result<usize> {
    ensure!(
        (1..=256).contains(&tokens) && tokens.is_power_of_two(),
        "invalid KV page geometry"
    );
    Ok((tokens * layout.bytes_per_token()).div_ceil(layout.align()) * layout.align())
}
pub fn page_bytes(tokens: usize) -> Result<usize> {
    layout_page_bytes(KvLayout::V4, tokens)
}
/// Target cache format of this process. `CANADA_SCRATCH_KV_FORMAT` unset or `v4`: today's caches, exactly as
/// before. `v41`: the reference's formats, a V4.1 window cache and a V4.1-FP4 compressed cache. The drafter's
/// caches stay V4 in both. All ranks must agree (checked at startup).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvFormat {
    pub window: KvLayout,
    pub compressed: KvLayout,
}
pub(crate) const KV_FORMAT_ENV: &str = "CANADA_SCRATCH_KV_FORMAT";
pub(crate) fn parse_kv_format(value: Option<&str>) -> Result<KvFormat> {
    match value {
        None | Some("v4") => Ok(KvFormat { window: KvLayout::V4, compressed: KvLayout::V4 }),
        Some("v41") => Ok(KvFormat { window: KvLayout::V41, compressed: KvLayout::V41Fp4 }),
        Some(other) => anyhow::bail!("invalid {KV_FORMAT_ENV}={other:?}: expected v4 or v41"),
    }
}
pub(crate) fn kv_format() -> Result<KvFormat> {
    static FORMAT: std::sync::OnceLock<std::result::Result<KvFormat, String>> = std::sync::OnceLock::new();
    FORMAT
        .get_or_init(|| parse_kv_format(std::env::var(KV_FORMAT_ENV).ok().as_deref()).map_err(|e| e.to_string()))
        .clone()
        .map_err(anyhow::Error::msg)
}
/// One byte per process for the startup agreement: 0 = v4, 1 = v41.
pub(crate) fn kv_format_code() -> Result<u8> {
    Ok(if kv_format()?.window == KvLayout::V41 { 1 } else { 0 })
}
#[derive(Clone)]
pub struct PagedKv {
    data: Rc<Buffer<u8>>,
    stream: Rc<Cell<usize>>,
    pages: usize,
    page_tokens: usize,
    page_bytes: usize,
    layout: KvLayout,
}
impl PagedKv {
    /// A V4-layout cache (today's target caches; always the drafter's).
    pub fn new(cuda: &Cuda, pages: usize, tokens: usize) -> Result<Self> {
        Self::new_layout(cuda, pages, tokens, KvLayout::V4)
    }
    pub fn new_layout(cuda: &Cuda, pages: usize, tokens: usize, layout: KvLayout) -> Result<Self> {
        let stride = layout_page_bytes(layout, tokens)?;
        let bytes = pages
            .checked_mul(stride)
            .filter(|&x| x > 0 && x <= 16 * 1024 * 1024 * 1024)
            .ok_or_else(|| anyhow::anyhow!("bounded KV allocation required"))?;
        let mut data = cuda.allocate::<u8>(bytes)?;
        data.zero()?;
        Ok(Self {
            data: Rc::new(data),
            stream: Rc::new(Cell::new(0)),
            pages,
            page_tokens: tokens,
            page_bytes: stride,
            layout,
        })
    }
    pub fn layout(&self) -> KvLayout {
        self.layout
    }
    pub(crate) fn bind_stream(&self, stream: &Stream) -> Result<()> {
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == raw,
            "KV storage requires one ordered stream"
        );
        self.stream.set(raw);
        Ok(())
    }
    pub fn pages(&self) -> usize {
        self.pages
    }
    pub fn page_tokens(&self) -> usize {
        self.page_tokens
    }
    pub fn page_bytes(&self) -> usize {
        self.page_bytes
    }
    pub fn capacity(&self) -> usize {
        self.pages * self.page_tokens
    }
    pub fn upload(&mut self, bytes: &[u8]) -> Result<()> {
        Rc::get_mut(&mut self.data)
            .ok_or_else(|| anyhow::anyhow!("shared KV seeding requires an exclusive owner"))?
            .upload(bytes)
    }
    pub fn data(&self) -> &Buffer<u8> {
        &self.data
    }
    pub(crate) fn clear(&mut self, cuda: &Cuda, stream: &Stream) -> Result<()> {
        ensure!(
            cuda.owns_buffer(&self.data) && cuda.owns_stream(stream),
            "KV reset context"
        );
        self.bind_stream(stream)?;
        unsafe { cuda.zero_device(self.data.ptr(), self.data.bytes(), stream) }
    }
}
/// Candidate addresses and lengths originate in the Rust block manager. A GPU
/// indexer will require a separate checked-capacity contract, not a CPU fallback.
pub struct Candidates {
    ids: Buffer<i32>,
    lengths: Buffer<i32>,
    rows: usize,
    width: usize,
    capacity: usize,
    ready: Cell<bool>,
    stream: Cell<usize>,
}
impl Candidates {
    /// Host block-manager metadata only. Keep GPU output candidates on their
    /// producer path; stable addresses allow an admitted model graph to replay.
    pub(crate) fn update(&mut self, ids: &[i32], lengths: &[i32]) -> Result<()> {
        self.ready.set(false);
        ensure!(
            ids.len() == self.rows * self.width
                && lengths.len() == self.rows
                && ids
                    .iter()
                    .all(|&x| x == -1 || (x >= 0 && (x as usize) < self.capacity))
                && lengths
                    .iter()
                    .all(|&l| l >= 0 && (l as usize) <= self.width),
            "candidate update outside shape/capacity"
        );
        self.ids.upload(ids)?;
        self.lengths.upload(lengths)?;
        self.ready.set(true);
        Ok(())
    }
    pub fn upload(
        cuda: &Cuda,
        rows: usize,
        width: usize,
        capacity: usize,
        ids: &[i32],
        lengths: &[i32],
    ) -> Result<Self> {
        ensure!(
            rows > 0
                && rows <= crate::config::MAX_WIDE_ROWS
                && width > 0
                && width <= 1024
                && capacity > 0
                && capacity <= i32::MAX as usize,
            "invalid candidate geometry"
        );
        ensure!(
            ids.len() == rows * width && lengths.len() == rows,
            "candidate dimensions"
        );
        ensure!(
            ids.iter()
                .all(|&x| x == -1 || (x >= 0 && (x as usize) < capacity)),
            "candidate outside KV capacity"
        );
        ensure!(
            lengths.iter().all(|&x| x >= 0 && x as usize <= width),
            "candidate length outside width"
        );
        let mut ib = cuda.allocate::<i32>(ids.len())?;
        ib.upload(ids)?;
        let mut lb = cuda.allocate::<i32>(rows)?;
        lb.upload(lengths)?;
        Ok(Self {
            ids: ib,
            lengths: lb,
            rows,
            width,
            capacity,
            ready: Cell::new(true),
            stream: Cell::new(0),
        })
    }
    pub(crate) fn device_output(
        cuda: &Cuda,
        rows: usize,
        width: usize,
        capacity: usize,
        lengths: &[i32],
    ) -> Result<Self> {
        ensure!(
            (1..=crate::config::MAX_WIDE_ROWS).contains(&rows)
                && (1..=1024).contains(&width)
                && capacity > 0
                && capacity <= i32::MAX as usize
                && lengths.len() == rows
                && lengths.iter().all(|&l| l >= 0 && (l as usize) <= width),
            "GPU candidate admission"
        );
        let mut b = cuda.allocate(rows)?;
        b.upload(lengths)?;
        Ok(Self {
            ids: cuda.allocate(rows * width)?,
            lengths: b,
            rows,
            width,
            capacity,
            ready: Cell::new(false),
            stream: Cell::new(0),
        })
    }
    pub(crate) fn bind_stream(&self, stream: &Stream) -> Result<()> {
        let p = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == p,
            "candidate storage requires one ordered stream"
        );
        self.stream.set(p);
        Ok(())
    }
    pub(crate) fn produced(&self) {
        self.ready.set(true);
    }
    pub(crate) fn ids(&self) -> &Buffer<i32> {
        &self.ids
    }
    pub(crate) fn lengths(&self) -> &Buffer<i32> {
        &self.lengths
    }
}
#[derive(Deserialize)]
struct Manifest {
    abi: u32,
    architecture: u32,
    physical_sms: u32,
    heads: Vec<usize>,
    topk: Vec<usize>,
    decode_max_rows: usize,
    page_tokens: usize,
    layout: String,
    library_sha256: String,
}
type Launch = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    i32,
    f32,
    *const c_void,
    *const c_void,
    usize,
    *const c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    i32,
    i32,
    usize,
    *mut c_void,
) -> i32;
pub struct Decode {
    _library: Library,
    cuda: Cuda,
    launch: Launch,
    rows: usize,
    topk: usize,
    extra_topk: usize,
    splits: usize,
    chunks_per_block: usize,
    stream: Cell<usize>,
    mid_output: Buffer<bf16>,
    mid_lse: Buffer<f32>,
    output: Buffer<bf16>,
    lse: Buffer<f32>,
}
impl Decode {
    pub fn new(
        cuda: &Cuda,
        root: &Path,
        rows: usize,
        topk: usize,
        extra_topk: usize,
    ) -> Result<Self> {
        cuda.qualify_gb10()?;
        let m: Manifest = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(
            (
                m.abi,
                m.architecture,
                m.physical_sms,
                m.decode_max_rows,
                m.page_tokens
            ) == (1, 121, 48, 64, 64)
                && m.heads == [HEADS]
                && m.layout == "dsv4_fp8_448_bf16_rope64_e8m0_footer8",
            "attention bundle contract changed"
        );
        ensure!(
            rows > 0
                && rows <= 64
                && m.topk.contains(&topk)
                && (extra_topk == 0 || (extra_topk <= 1024 && extra_topk % 64 == 0)),
            "unsupported sparse decode geometry"
        );
        let library = root.join("libcanada_attention.so");
        ensure!(
            format!("{:x}", Sha256::digest(std::fs::read(&library)?)) == m.library_sha256,
            "attention library digest mismatch"
        );
        let lib = unsafe { Library::new(&library) }?;
        let abi: unsafe extern "C" fn() -> i32 = *unsafe { lib.get(b"ca_sparse_attention_abi\0") }?;
        ensure!(unsafe { abi() } == 1, "attention library ABI mismatch");
        let launch = *unsafe { lib.get(b"ca_sparse_attention_decode\0") }?;
        let splits = topk.div_ceil(64) + extra_topk.div_ceil(64);
        // Upstream 48-SM heuristic. Grouping also changes intermediate rounding:
        // partial attention outputs are BF16, so this is not numerically invariant.
        let mut cpb = 1;
        let mut best_gap = 2f32;
        for candidate in 1..=splits {
            let active = rows * splits.div_ceil(candidate);
            let waves = active.div_ceil(48);
            if waves > 3 {
                continue;
            }
            let gap = waves as f32 - active as f32 / 48.;
            if gap < best_gap - 1e-6 || (gap < best_gap + 1e-6 && candidate > cpb) {
                best_gap = gap;
                cpb = candidate;
            }
        }
        let match_grouping = match std::env::var("CANADA_SCRATCH_HISTORY32_MATCH_GROUPING").as_deref() {
            Ok("1") => true,
            Ok("0") | Err(_) => false,
            _ => anyhow::bail!("history32 attention grouping must be 0 or 1"),
        };
        ensure!(!match_grouping || crate::config::history32_enabled(),
            "history32 grouping requires its admitted capability");
        // Isolate width from grouping in the first two SWA layers. All other
        // geometries retain their existing algorithm and arithmetic boundaries.
        if match_grouping && rows == 32 && topk == 128 && extra_topk == 0 {
            cpb = 1;
        }
        let mut mid_output = cuda.allocate::<bf16>(rows * HEADS * splits * WIDTH)?;
        mid_output.zero()?;
        let mut mid_lse = cuda.allocate::<f32>(rows * HEADS * splits)?;
        mid_lse.zero()?;
        let mut output = cuda.allocate::<bf16>(rows * HEADS * WIDTH)?;
        output.zero()?;
        let mut lse = cuda.allocate::<f32>(rows * HEADS)?;
        lse.zero()?;
        Ok(Self {
            _library: lib,
            cuda: cuda.clone(),
            launch,
            rows,
            topk,
            extra_topk,
            splits,
            chunks_per_block: cpb,
            stream: Cell::new(0),
            mid_output,
            mid_lse,
            output,
            lse,
        })
    }
    pub fn output_mut(&mut self) -> &mut Buffer<bf16> {
        &mut self.output
    }
    pub fn output(&self) -> &Buffer<bf16> {
        &self.output
    }
    pub fn lse(&self) -> &Buffer<f32> {
        &self.lse
    }
    pub fn workspace_bytes(&self) -> usize {
        self.mid_output.bytes() + self.mid_lse.bytes() + self.output.bytes() + self.lse.bytes()
    }
    pub fn run(
        &self,
        q: &Buffer<bf16>,
        kv: &PagedKv,
        indices: &Candidates,
        sink: Option<&Buffer<f32>>,
        extra: Option<(&PagedKv, &Candidates)>,
        scale: f32,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            q.len() == self.rows * HEADS * WIDTH
                && self.cuda.owns_buffer(q)
                && self.cuda.owns_stream(stream),
            "attention Q/context mismatch"
        );
        ensure!(kv.layout == KvLayout::V4, "the attention-v1 decode primitive reads only V4 caches");
        ensure!(
            kv.page_tokens == PAGE_TOKENS && self.cuda.owns_buffer(&kv.data),
            "main KV requires native 64-token pages in the same context"
        );
        ensure!(
            indices.ready.get()
                && indices.rows == self.rows
                && indices.width == self.topk
                && indices.capacity == kv.capacity()
                && self.cuda.owns_buffer(&indices.ids)
                && self.cuda.owns_buffer(&indices.lengths),
            "main candidate contract mismatch"
        );
        ensure!(scale.is_finite() && scale > 0., "invalid attention scale");
        if let Some(s) = sink {
            ensure!(
                s.len() == HEADS && self.cuda.owns_buffer(s),
                "attention sink dimensions/context"
            );
        }
        let (ep, ei, el, epbs, estride) = if let Some((cache, ids)) = extra {
            ensure!(cache.layout == KvLayout::V4, "the attention-v1 decode primitive reads only V4 caches");
            ensure!(
                ids.ready.get()
                    && self.extra_topk > 0
                    && ids.rows == self.rows
                    && ids.width == self.extra_topk
                    && ids.capacity == cache.capacity()
                    && self.cuda.owns_buffer(&cache.data)
                    && self.cuda.owns_buffer(&ids.ids)
                    && self.cuda.owns_buffer(&ids.lengths),
                "extra candidate contract mismatch"
            );
            (
                cache.data.ptr(),
                ids.ids.ptr(),
                ids.lengths.ptr(),
                cache.page_tokens,
                cache.page_bytes,
            )
        } else {
            ensure!(self.extra_topk == 0, "missing extra KV");
            (0, 0, 0, 0, 0)
        };
        kv.bind_stream(stream)?;
        indices.bind_stream(stream)?;
        if let Some((cache, _)) = extra {
            cache.bind_stream(stream)?;
        }
        if let Some((_, ids)) = extra {
            ids.bind_stream(stream)?;
        }
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == raw,
            "attention workspace requires one ordered stream"
        );
        self.stream.set(raw);
        let status = unsafe {
            (self.launch)(
                self.rows as i32,
                HEADS as i32,
                self.topk as i32,
                self.splits as i32,
                self.chunks_per_block as i32,
                scale,
                q.ptr() as _,
                kv.data.ptr() as _,
                kv.page_bytes,
                indices.ids.ptr() as _,
                self.mid_output.ptr() as _,
                self.mid_lse.ptr() as _,
                self.output.ptr() as _,
                self.lse.ptr() as _,
                indices.lengths.ptr() as _,
                sink.map_or(0, Buffer::ptr) as _,
                ep as _,
                ei as _,
                el as _,
                self.extra_topk as i32,
                epbs as i32,
                estride,
                stream.raw(),
            )
        };
        ensure!(
            status == 0,
            "native sparse attention launch failed: {status}"
        );
        Ok(())
    }
}
#[derive(Deserialize)]
struct PrefillManifest {
    abi: u32,
    architecture: u32,
    physical_sms: u32,
    heads: usize,
    topk: usize,
    min_rows: usize,
    max_rows: usize,
    main_page_tokens: usize,
    max_extra_topk: usize,
    extra_page_tokens: Vec<usize>,
    compute: String,
    lengths: String,
    layout: String,
    library_sha256: String,
}
type PrefillLaunch = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    f32,
    *const c_void,
    *const c_void,
    usize,
    *const c_void,
    *mut c_void,
    *mut c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    usize,
    *const c_void,
    *const c_void,
    *mut c_void,
) -> i32;
/// Distinct prefill numeric phase. It uses the reference's BF16 MG kernel,
/// without decode split/merge workspaces or silently chunking into decode.
type SplitMerge = unsafe extern "C" fn(*const c_void,*const c_void,*mut c_void,*mut c_void,i32,i32,*mut c_void)->i32;
struct SplitWorkspace { output:Buffer<u8>,lse:Buffer<f32>,merge:SplitMerge,parts:usize }
/// `ca_sparse_decode` (attention-dsv41-decode-v1): the `ca_sparse_prefill` arguments, then scratch, its bytes and
/// the split count. One call runs the split kernel (fp32 partials) and the sink-folding merge.
type Dsv41DecodeLaunch = unsafe extern "C" fn(
    i32, i32, i32, f32, *const c_void, *const c_void, usize, *const c_void, *mut c_void, *mut c_void,
    *const c_void, *const c_void, *const c_void, usize, *const c_void, *const c_void, *mut c_void, usize, i32,
    *mut c_void,
) -> i32;
struct Dsv41Split { launch: Dsv41DecodeLaunch, scratch: Buffer<u8>, splits: usize }
pub(crate) const DSV41_MAIN_LAYOUT: &str = "dsv41_fp8_g32_ue8m0_footer16";
pub(crate) const DSV41_EXTRA_LAYOUT: &str = "dsv41_fp4_g16_e4m3_footer32";
pub struct Prefill {
    _library: Library,
    cuda: Cuda,
    launch: PrefillLaunch,
    rows: usize,
    extra_topk: usize,
    stream: Cell<usize>,
    output: Buffer<bf16>,
    lse: Buffer<f32>,
    split: Option<SplitWorkspace>,
    /// Cache layouts this primitive reads: (V4, V4) for the FlashInfer 0.6.18 bundles, (V41, V41Fp4) for DSv4.1.
    main_layout: KvLayout,
    extra_layout: KvLayout,
    dsv41_split: Option<Dsv41Split>,
}
impl Prefill {
    pub fn new(
        cuda: &Cuda,
        root: &Path,
        rows: usize,
        topk: usize,
        extra_topk: usize,
    ) -> Result<Self> {
        Self::new_admitted(cuda,root,rows,topk,extra_topk,65)
    }
    pub(crate) fn new_small_bf16(cuda:&Cuda,root:&Path,rows:usize,topk:usize,extra_topk:usize)->Result<Self> {
        ensure!([1,6,16].contains(&rows)||(rows==32&&crate::config::history32_enabled()),"small BF16 attention geometry");
        Self::new_admitted(cuda,root,rows,topk,extra_topk,1)
    }
    pub(crate) fn new_split_bf16(cuda:&Cuda,kernels:&Path,rows:usize,extra_topk:usize,parts:usize,fp32:bool,vector:bool)->Result<Self> {
        ensure!(([6,16].contains(&rows)
                ||(rows==32&&crate::config::history32_enabled()&&fp32&&vector&&parts==3))
            && ([2,8].contains(&parts)||(fp32&&parts==3))
            && extra_topk==512&&(!vector||fp32),"split attention native geometry");
        let mut this=Self::new_small_bf16(cuda,&kernels.join("attention-small-bf16-s729"),rows,128,extra_topk)?;
        let root=kernels.join(if rows==32{"attention-direct-fp32-wide-v1"}
            else if vector{"attention-direct-fp32-vector-s799"}else if fp32{"attention-direct-fp32-s795"}else{"attention-direct-split-s782"});
        let m:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(m["abi"]==(if fp32{3}else{2}) && (!fp32||m["partial_dtype"]=="float32")
            &&(!vector||m["partial_store_type"]=="float2")
            && m["architecture"]==121 && m["heads"]==16 && m["width"]==512
            && m["main_topk"]==128 && m["extra_page_tokens"]==serde_json::json!([2,64])
            && m["header_sha256"]=="5f005980cff1606fc1a239faeb93d1d697710c92014d549e4425e9db04f675e0",
            "split BF16 primitive contract");
        let part=parts.to_string();let name=format!("libdirect{parts}.so");
        ensure!(m["libraries"][&part]["file"]==name,"split primitive filename");
        let path=root.join(name);
        ensure!(m["libraries"][&part]["sha256"]==format!("{:x}",Sha256::digest(std::fs::read(&path)?)),
            "split BF16 primitive digest");
        let library=unsafe{Library::new(path)}?;
        let abi:unsafe extern "C" fn()->i32=*unsafe{library.get(b"ca_direct_split_abi\0")}?;
        ensure!(unsafe{abi()}==(if fp32{2}else{1}),"split BF16 ABI");
        let split=SplitWorkspace { output:cuda.allocate_rows(rows*parts*HEADS*WIDTH*(if fp32{4}else{2}))?,
            lse:cuda.allocate_rows(rows*parts*HEADS)?,merge:*unsafe{library.get(b"ca_merge_attention_parts\0") }?,parts };
        this.launch=*unsafe{library.get(b"ca_sparse_prefill\0")}?;
        this._library=library;this.split=Some(split);Ok(this)
    }
    /// FlashInfer v0.7.3's SM120 DeepSeek-V4.1 BF16 prefill kernel (attention-dsv41-prefill-v1): any row count
    /// from 1 to 16,384, V4.1 window cache plus optional V4.1-FP4 compressed cache, log2 LSE with the sink folded in.
    pub(crate) fn new_dsv41(cuda:&Cuda,root:&Path,rows:usize,extra_topk:usize)->Result<Self> {
        cuda.qualify_gb10()?;
        let m:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        // Two primitives share this contract: FlashInfer's DSv4.1 BF16 prefill (up to 1024 extra candidates) and the
        // adapted pico prefill kernel (attention-pico-prefill-v1, f16 MMAs on exactly widened codes, up to 512).
        let max_extra=m["max_extra_topk"].as_u64().unwrap_or(0) as usize;
        ensure!(m["abi"]==2 && m["architecture"]==121 && m["heads"]==16 && m["topk"]==128 && m["min_rows"]==1
            && m["max_rows"]==16384 && m["main_page_tokens"]==64 && m["extra_page_tokens"]==serde_json::json!([64])
            && [512,1024].contains(&max_extra) && m["lengths"]=="explicit"
            && (m["compute"]=="BF16" || m["compute"]=="F16_EXACT_WIDENING")
            && m["layout"]==DSV41_MAIN_LAYOUT && m["extra_layout"]==DSV41_EXTRA_LAYOUT && m["lse"]=="log2_with_sink",
            "DSv4.1 prefill primitive contract");
        ensure!((1..=16384).contains(&rows) && extra_topk<=max_extra && extra_topk%64==0,"DSv4.1 prefill execution shape");
        let path=root.join("libcanada_prefill.so");
        ensure!(m["library_sha256"]==format!("{:x}",Sha256::digest(std::fs::read(&path)?)),"DSv4.1 prefill digest");
        let library=unsafe{Library::new(path)}?;
        let abi:unsafe extern "C" fn()->i32=*unsafe{library.get(b"ca_sparse_prefill_abi\0")}?;
        ensure!(unsafe{abi()}==2,"DSv4.1 prefill ABI");
        Ok(Self{launch:*unsafe{library.get(b"ca_sparse_prefill\0")}?,_library:library,cuda:cuda.clone(),rows,extra_topk,
            stream:Cell::new(0),output:cuda.allocate_rows(rows*HEADS*WIDTH)?,lse:cuda.allocate_rows(rows*HEADS)?,
            split:None,main_layout:KvLayout::V41,extra_layout:KvLayout::V41Fp4,dsv41_split:None})
    }
    /// FlashInfer v0.7.3's DeepSeek-V4.1 split decode with fp32 partials (attention-dsv41-decode-v1): `splits` CTAs
    /// per row share its candidate chunks, then a merge folds in the sink. Same layouts and LSE as `new_dsv41`.
    pub(crate) fn new_dsv41_split(cuda:&Cuda,root:&Path,rows:usize,extra_topk:usize,splits:usize)->Result<Self> {
        cuda.qualify_gb10()?;
        let m:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(m["abi"]==1 && m["kind"]=="split_decode_fp32_partials" && m["architecture"]==121 && m["heads"]==16
            && m["topk"]==128 && m["max_rows"]==4096 && m["max_splits"]==64 && m["main_page_tokens"]==64
            && m["extra_page_tokens"]==serde_json::json!([64]) && m["max_extra_topk"]==1024
            && m["partial_dtype"]=="float32" && m["layout"]==DSV41_MAIN_LAYOUT && m["extra_layout"]==DSV41_EXTRA_LAYOUT
            && m["lse"]=="log2_with_sink","DSv4.1 split decode primitive contract");
        ensure!((1..=4096).contains(&rows) && (1..=64).contains(&splits) && extra_topk<=1024 && extra_topk%64==0,
            "DSv4.1 split decode execution shape");
        let path=root.join("libcanada_decode.so");
        ensure!(m["library_sha256"]==format!("{:x}",Sha256::digest(std::fs::read(&path)?)),"DSv4.1 decode digest");
        let library=unsafe{Library::new(path)}?;
        let abi:unsafe extern "C" fn()->i32=*unsafe{library.get(b"ca_sparse_decode_abi\0")}?;
        ensure!(unsafe{abi()}==1,"DSv4.1 decode ABI");
        let scratch_bytes:unsafe extern "C" fn(i32,i32)->usize=*unsafe{library.get(b"ca_sparse_decode_scratch_bytes\0")}?;
        let bytes=unsafe{scratch_bytes(rows as i32,splits as i32)};
        ensure!(bytes==rows*HEADS*splits*(WIDTH+1)*4,"DSv4.1 decode scratch geometry");
        let split=Dsv41Split{launch:*unsafe{library.get(b"ca_sparse_decode\0")}?,scratch:cuda.allocate::<u8>(bytes)?,splits};
        // `launch` is unused on this path; keep a valid symbol so the struct stays uniform.
        Ok(Self{launch:*unsafe{library.get(b"ca_sparse_decode\0")}?,_library:library,cuda:cuda.clone(),rows,extra_topk,
            stream:Cell::new(0),output:cuda.allocate_rows(rows*HEADS*WIDTH)?,lse:cuda.allocate_rows(rows*HEADS)?,
            split:None,main_layout:KvLayout::V41,extra_layout:KvLayout::V41Fp4,dsv41_split:Some(split)})
    }
    fn new_admitted(cuda:&Cuda,root:&Path,rows:usize,topk:usize,extra_topk:usize,min_rows:usize)->Result<Self> {
        cuda.qualify_gb10()?;
        let m: PrefillManifest =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(
            (
                m.abi,
                m.architecture,
                m.physical_sms,
                m.heads,
                m.topk,
                m.min_rows,
                m.main_page_tokens,
                m.max_extra_topk
            ) == (1, 121, 48, 16, 128, min_rows, 64, 1024)
                && [4096, 8192, 16384].contains(&m.max_rows)
                && m.extra_page_tokens == [2, 64]
                && m.compute == "BF16"
                && m.lengths == "explicit"
                && m.layout == "dsv4_fp8_448_bf16_rope64_e8m0_footer8",
            "prefill primitive contract changed"
        );
        ensure!(
            (min_rows..=m.max_rows).contains(&rows)
                && topk == 128
                && extra_topk <= 1024
                && extra_topk % 64 == 0,
            "prefill execution shape"
        );
        let path = root.join("libcanada_prefill.so");
        ensure!(
            format!("{:x}", Sha256::digest(std::fs::read(&path)?)) == m.library_sha256,
            "prefill primitive digest mismatch"
        );
        let library = unsafe { Library::new(path) }?;
        let abi: unsafe extern "C" fn() -> i32 =
            *unsafe { library.get(b"ca_sparse_prefill_abi\0") }?;
        ensure!(unsafe { abi() } == 1, "prefill primitive ABI");
        Ok(Self {
            launch: *unsafe { library.get(b"ca_sparse_prefill\0") }?,
            _library: library,
            cuda: cuda.clone(),
            rows,
            extra_topk,
            stream: Cell::new(0),
            output: cuda.allocate_rows(rows * HEADS * WIDTH)?,
            lse: cuda.allocate_rows(rows * HEADS)?,
            split:None,
            main_layout: KvLayout::V4,
            extra_layout: KvLayout::V4,
            dsv41_split: None,
        })
    }
    pub fn output(&self) -> &Buffer<bf16> {
        &self.output
    }
    pub fn output_mut(&mut self) -> &mut Buffer<bf16> {
        &mut self.output
    }
    pub fn lse(&self) -> &Buffer<f32> {
        &self.lse
    }
    pub fn workspace_bytes(&self) -> usize {
        self.split.as_ref().map_or(0,|s|s.output.bytes()+s.lse.bytes())
            +self.dsv41_split.as_ref().map_or(0,|s|s.scratch.bytes())
    }
    pub fn main_layout(&self) -> KvLayout {
        self.main_layout
    }
    pub fn run(
        &self,
        q: &Buffer<bf16>,
        kv: &PagedKv,
        indices: &Candidates,
        sink: Option<&Buffer<f32>>,
        extra: Option<(&PagedKv, &Candidates)>,
        scale: f32,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            q.len() == self.rows * HEADS * WIDTH
                && self.cuda.owns_buffer(q)
                && self.cuda.owns_stream(stream),
            "prefill Q/context"
        );
        ensure!(kv.layout == self.main_layout, "prefill main cache layout {:?} for a {:?} primitive",
            kv.layout, self.main_layout);
        ensure!(
            kv.page_tokens == 64
                && self.cuda.owns_buffer(&kv.data)
                && indices.ready.get()
                && indices.rows == self.rows
                && indices.width == 128
                && indices.capacity == kv.capacity()
                && self.cuda.owns_buffer(&indices.ids)
                && self.cuda.owns_buffer(&indices.lengths),
            "prefill main cache/candidate contract"
        );
        ensure!(scale.is_finite() && scale > 0., "prefill scale");
        if let Some(s) = sink {
            ensure!(s.len() == HEADS && self.cuda.owns_buffer(s), "prefill sink");
        }
        let (ep, es, ei, el, pbs) = if let Some((cache, ids)) = extra {
            ensure!(cache.layout == self.extra_layout
                    && (self.extra_layout == KvLayout::V4 || cache.page_tokens == 64),
                "prefill secondary cache layout {:?}/{} for a {:?} primitive", cache.layout, cache.page_tokens,
                self.extra_layout);
            ensure!(
                self.extra_topk > 0
                    && [2, 64].contains(&cache.page_tokens)
                    && ids.ready.get()
                    && ids.rows == self.rows
                    && ids.width == self.extra_topk
                    && ids.capacity == cache.capacity()
                    && self.cuda.owns_buffer(&cache.data)
                    && self.cuda.owns_buffer(&ids.ids)
                    && self.cuda.owns_buffer(&ids.lengths),
                "prefill secondary cache/candidate contract"
            );
            (
                cache.data.ptr(),
                cache.page_bytes,
                ids.ids.ptr(),
                ids.lengths.ptr(),
                cache.page_tokens,
            )
        } else {
            ensure!(self.extra_topk == 0, "prefill secondary cache missing");
            (0, 0, 0, 0, 0)
        };
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == raw,
            "prefill requires one ordered stream"
        );
        kv.bind_stream(stream)?;
        indices.bind_stream(stream)?;
        if let Some((cache, ids)) = extra {
            cache.bind_stream(stream)?;
            ids.bind_stream(stream)?;
        }
        self.stream.set(raw);
        if let Some(d) = &self.dsv41_split {
            ensure!(sink.is_some(), "DSv4.1 split decode needs the sink");
            let status = unsafe {
                (d.launch)(
                    self.rows as i32, self.extra_topk as i32, pbs as i32, scale, q.ptr() as _, kv.data.ptr() as _,
                    kv.page_bytes, indices.ids.ptr() as _, self.output.ptr() as _, self.lse.ptr() as _,
                    indices.lengths.ptr() as _, sink.map_or(0, Buffer::ptr) as _, ep as _, es, ei as _, el as _,
                    d.scratch.ptr() as _, d.scratch.bytes(), d.splits as i32, stream.raw(),
                )
            };
            ensure!(status == 0, "DSv4.1 split decode launch failed: {status}");
            return Ok(());
        }
        ensure!(self.split.is_none() || ([2,64].contains(&pbs) && sink.is_some()),"split attention cache/sink");
        let (out,lse)=self.split.as_ref().map_or((self.output.ptr(),self.lse.ptr()),|s|(s.output.ptr(),s.lse.ptr()));
        let status = unsafe {
            (self.launch)(
                self.rows as i32,
                self.extra_topk as i32,
                pbs as i32,
                scale,
                q.ptr() as _,
                kv.data.ptr() as _,
                kv.page_bytes,
                indices.ids.ptr() as _,
                out as _,
                lse as _,
                indices.lengths.ptr() as _,
                sink.map_or(0, Buffer::ptr) as _,
                ep as _,
                es,
                ei as _,
                el as _,
                stream.raw(),
            )
        };
        ensure!(status == 0, "prefill CUDA launch failed: {status}");
        if let Some(s)=&self.split {
            let status=unsafe{(s.merge)(out as _,lse as _,self.output.ptr()as _,self.lse.ptr()as _,self.rows as i32,s.parts as i32,stream.raw())};
            ensure!(status==0,"split attention merge launch: {status}");
        }
        Ok(())
    }
}

/// Explicit numeric phase selection; each phase retains its own admission and
/// workspace contract. Paths name an already qualified matching primitive.
pub enum Execution {
    Decode(Decode),
    Prefill(Prefill),
}
impl Execution {
    pub fn new(
        cuda: &Cuda,
        root: &Path,
        rows: usize,
        topk: usize,
        extra_topk: usize,
    ) -> Result<Self> {
        if rows <= 64 {
            Ok(Self::Decode(Decode::new(
                cuda, root, rows, topk, extra_topk,
            )?))
        } else {
            Ok(Self::Prefill(Prefill::new(
                cuda, root, rows, topk, extra_topk,
            )?))
        }
    }
    pub fn output(&self) -> &Buffer<bf16> {
        match self {
            Self::Decode(v) => v.output(),
            Self::Prefill(v) => v.output(),
        }
    }
    pub fn output_mut(&mut self) -> &mut Buffer<bf16> {
        match self {
            Self::Decode(v) => v.output_mut(),
            Self::Prefill(v) => v.output_mut(),
        }
    }
    pub fn lse(&self) -> &Buffer<f32> {
        match self {
            Self::Decode(v) => v.lse(),
            Self::Prefill(v) => v.lse(),
        }
    }
    pub fn workspace_bytes(&self) -> usize {
        match self {
            Self::Decode(v) => v.workspace_bytes(),
            Self::Prefill(v) => v.workspace_bytes(),
        }
    }
    pub fn run(
        &self,
        q: &Buffer<bf16>,
        kv: &PagedKv,
        indices: &Candidates,
        sink: Option<&Buffer<f32>>,
        extra: Option<(&PagedKv, &Candidates)>,
        scale: f32,
        stream: &Stream,
    ) -> Result<()> {
        match self {
            Self::Decode(v) => v.run(q, kv, indices, sink, extra, scale, stream),
            Self::Prefill(v) => v.run(q, kv, indices, sink, extra, scale, stream),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn v4_footer_layout_matches_qualified_payload() {
        assert_eq!(super::page_bytes(64).unwrap(), 37440);
        assert_eq!(super::page_bytes(256).unwrap(), 149760);
        assert_eq!(super::page_bytes(2).unwrap(), 1728);
        assert!(super::page_bytes(0).is_err());
        assert!(super::page_bytes(63).is_err());
        assert!(super::page_bytes(512).is_err());
    }
    #[test]
    fn v41_layouts_follow_sglang_page_alignment() {
        use super::{KvLayout::*, layout_page_bytes};
        assert_eq!(layout_page_bytes(V41, 64).unwrap(), 33792);
        assert_eq!(layout_page_bytes(V41, 256).unwrap(), 135168);
        assert_eq!(layout_page_bytes(V41Fp4, 64).unwrap(), 18432);
        assert_eq!(layout_page_bytes(V41Fp4, 2).unwrap(), 768);
        assert_eq!(layout_page_bytes(V4, 64).unwrap(), super::page_bytes(64).unwrap());
        assert!(layout_page_bytes(V41Fp4, 3).is_err());
    }
    #[test]
    fn kv_format_defaults_to_v4_and_rejects_unknown_values() {
        use super::{KvLayout::*, parse_kv_format};
        let v4 = parse_kv_format(None).unwrap();
        assert_eq!((v4.window, v4.compressed), (V4, V4));
        assert_eq!(parse_kv_format(Some("v4")).unwrap(), v4);
        let v41 = parse_kv_format(Some("v41")).unwrap();
        assert_eq!((v41.window, v41.compressed), (V41, V41Fp4));
        assert!(parse_kv_format(Some("fp4")).is_err());
        assert!(parse_kv_format(Some("")).is_err());
    }
}
