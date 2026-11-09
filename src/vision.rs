//! Rust-owned V4.1 vision tower and aligner. All model arithmetic stays on GPU.
//! Kept separate from text kernels; HTTP admission remains gated until qualified.
use crate::{
    blas::Blas,
    checkpoint::{Checkpoint, Dtype},
    cuda::{Buffer, Cuda, Module, Stream},
    layout::RankPlan,
    triton_ops::Operator,
    weights::Weights,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use std::{ffi::c_void, path::Path};

pub const PREFIXES: &[&str] = &["vision.", "aligner.", "image_start", "image_newline", "image_end"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Geometry {
    pub height: usize,
    pub width: usize,
}
impl Geometry {
    pub fn validate(self) -> Result<Self> {
        ensure!(
            self.height > 0 && self.width > 0 && self.height <= 3063 && self.width <= 3063,
            "invalid vision patch grid"
        );
        ensure!(
            self.rows() <= 9198 && self.tokens() <= 1024,
            "vision patch/token budget exceeded"
        );
        Ok(self)
    }
    pub fn rows(self) -> usize {
        self.height * self.width
    }
    pub fn feature_rows(self) -> usize {
        self.height.div_ceil(3) * self.width.div_ceil(3)
    }
    pub fn tokens(self) -> usize {
        self.height.div_ceil(3) * (self.width.div_ceil(3) + 1) + 2
    }

    /// Additional bytes while encoding one image. Weights and the explicit
    /// 32MiB cuBLAS workspace are already resident. The attention operator
    /// refuses hidden global scratch at load time.
    pub(crate) fn workspace_bytes(self, include_patches: bool, compact:bool) -> Result<usize> {
        Ok(WorkspacePlan::new(self)?.admission_bytes(include_patches,compact))
    }
    pub(crate) fn scratch_bytes(self, compact:bool)->Result<usize> {
        Ok(WorkspacePlan::new(self)?.scratch_bytes(compact))
    }
    pub(crate) fn arena_admission_bytes(self, include_patches:bool)->Result<usize> {
        Ok(WorkspacePlan::new(self)?.arena_admission_bytes(include_patches))
    }
}

pub(crate) fn compact_workspace()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_VISION_COMPACT_WORKSPACE").as_deref() {
        Err(std::env::VarError::NotPresent)|Ok("1")=>Ok(true),Ok("0")=>Ok(false),
        _=>anyhow::bail!("vision compact workspace must be0or1"),
    }
}

/// Logical lengths and owned allocation accounting share one layout. Aliases
/// are ordered on one stream and never escape the encoder with the final span.
struct WorkspacePlan {
    bf16: [usize; 15],
    fp32: [usize; 3],
    i32: [usize; 2],
    patches: usize,
}
impl WorkspacePlan {
    fn new(g: Geometry) -> Result<Self> {
        let g = g.validate()?;
        let n = g.rows();
        let m = g.feature_rows();
        let h = n * 1024;
        Ok(Self {
            // x, norm, qkv, q, k, v, attention, delta, gate/up, gated,
            // unfolded, align1, gelu, features, completed image span.
            bf16: [h, h, n * 3072, h, h, h, h, h, n * 5632, n * 2816,
                m * 9216, m * 5120, m * 5120, m * 5120, g.tokens() * 5120],
            fp32: [(n * 5632).max(m * 5120), n * 32, n * 32],
            i32: [1, 1],
            patches: n * 588,
        })
    }
    fn backings(&self,compact:bool)->[usize;15] {
        let mut b=self.bf16;
        if compact {
            // gate/up -> QKV, either projection delta, then unfolded features.
            b[8]=b[8].max(b[2]).max(b[7]).max(b[10]);
            // gated -> either pre-projection norm (not concurrently live).
            b[9]=b[9].max(b[1]);
            // Q/K/V die before the aligner begins.
            for (owner,view) in [(3,11),(4,12),(5,13)] {b[owner]=b[owner].max(b[view]);}
            for view in [1,2,7,10,11,12,13] {b[view]=0;}
        }
        b
    }
    fn fp32_backings(&self,compact:bool)->[usize;3] {
        let mut b=self.fp32;
        // Only bias-bearing projections write FP32 scratch. The larger
        // bias-free gate/up projection writes BF16 directly and never uses it.
        if compact {b[0]=self.bf16[2].max(self.bf16[11]);}
        b
    }
    fn scratch_bytes(&self,compact:bool)->usize {
        // Every transient allocation starts at a 256-byte boundary. The final
        // image span is independently owned and must survive arena reuse.
        self.backings(compact)[..14].iter().map(|n|(n*2).next_multiple_of(256)).sum::<usize>()
            + self.fp32_backings(compact).iter().map(|n|(n*4).next_multiple_of(256)).sum::<usize>()
            + self.i32.iter().map(|n|(n*4).next_multiple_of(256)).sum::<usize>()
    }
    fn arena_admission_bytes(&self,include_patches:bool)->usize {
        // Preserve the existing transient driver allowance. Only tensor bytes
        // backed by an already resident scratch arena disappear from admission.
        (self.bf16[14]*2).next_multiple_of(2<<20)
            + if include_patches {(self.patches*2).next_multiple_of(2<<20)}else{0}
            + (512<<20)
    }
    fn admission_bytes(&self, include_patches: bool,compact:bool) -> usize {
        // A live GB10 image allocation burst consumed up to ~430MiB more
        // physical headroom than tensor bytes. The former64MiB allowance was
        // insufficient under pressure. This transient admission allowance is
        // not allocated, does not lower the UMA floor, and is released when
        // encoding finishes. The post-allocation check remains authoritative.
        let rounded = |bytes: usize| bytes.next_multiple_of(2 << 20);
        self.backings(compact).iter().map(|n| rounded(n * 2)).sum::<usize>()
            + self.fp32_backings(compact).iter().map(|n| rounded(n * 4)).sum::<usize>()
            + self.i32.iter().map(|n| rounded(n * 4)).sum::<usize>()
            + if include_patches { rounded(self.patches * 2) } else { 0 }
            + (512 << 20)
    }
}

pub struct Vision {
    cuda: Cuda,
    weights: Weights,
    module: Module,
    attention: Operator,
    blas: Blas,
    /// Optional cuBLASLt BF16 bias epilogue for biased projections (resolved at creation).
    lt_bias: Option<crate::blas_lt::BlasLt>,
    /// Row-split attention (local query rows, full keys), present when the artifact provides it.
    split_attention: Option<Operator>,
    // Fixed at construction; no environment lookup in image/block loops.
    norm_mode: u8,
}

pub struct ImageSpan {
    pub data: Buffer<bf16>,
    pub geometry: Geometry,
}

/// Diagnostic tensors are read-only, valid only during the callback, and
/// ordered on the supplied stream. No numeric reference work belongs on CPU.
pub type Tap<'a> = dyn FnMut(&str, u64, usize, &'static str, &Stream) -> Result<()> + 'a;

impl Vision {
    pub fn new(
        cuda: &Cuda,
        kernel: &[u8],
        attention: &Path,
        checkpoint: &Checkpoint,
        plan: &RankPlan,
        root: &Path,
    ) -> Result<Self> {
        cuda.qualify_gb10()?;
        let weights = Weights::load_prefixes(
            cuda,
            root,
            checkpoint,
            plan,
            PREFIXES,
        )?;
        Self::from_weights(cuda, kernel, attention, weights)
    }
    /// Serving uses a qualified, lossless vision-only atlas on the encoding
    /// rank. Other ranks need only the completed BF16 image spans.
    pub fn new_staged(cuda: &Cuda, kernel: &[u8], attention: &Path,
        checkpoint: &Checkpoint, plan: &RankPlan, stage: &Path) -> Result<Self> {
        cuda.qualify_gb10()?;
        let weights = Weights::load_staged(cuda, checkpoint, plan, PREFIXES, stage)?;
        Self::from_weights(cuda, kernel, attention, weights)
    }
    fn from_weights(cuda: &Cuda, kernel: &[u8], attention: &Path, weights: Weights) -> Result<Self> {
        let shape = |name: &str, expected: &[u64]| -> Result<()> {
            let v = weights.view(name)?;
            ensure!(
                v.dtype == Dtype::Bf16 && v.shape == expected,
                "vision canonical shape/dtype: {name}"
            );
            Ok(())
        };
        shape("vision.patch_embed.proj.weight", &[1024, 588])?;
        shape("vision.patch_embed.proj.bias", &[1024])?;
        shape("vision.norm.weight", &[1024])?;
        for i in 0..32 {
            for (tail, dims) in [
                ("norm1.weight", vec![1024]),
                ("norm2.weight", vec![1024]),
                ("attn.wqkv.weight", vec![3072, 1024]),
                ("attn.wqkv.bias", vec![3072]),
                ("attn.wo.weight", vec![1024, 1024]),
                ("attn.wo.bias", vec![1024]),
                ("mlp.w1.weight", vec![5632, 1024]),
                ("mlp.w2.weight", vec![1024, 2816]),
            ] {
                shape(&format!("vision.blocks.{i}.{tail}"), &dims)?;
            }
        }
        for (name, dims) in [
            ("aligner.w1.weight", vec![5120, 9216]),
            ("aligner.w1.bias", vec![5120]),
            ("aligner.w2.weight", vec![5120, 5120]),
            ("aligner.w2.bias", vec![5120]),
            ("image_start", vec![5120]),
            ("image_newline", vec![5120]),
            ("image_end", vec![5120]),
        ] {
            shape(name, &dims)?;
        }
        let blas = Blas::new(cuda)?;
        blas.require_fp32_reduction()?;
        // Fused normalization keeps BF16 image spans identical with modest
        // small-image gains. Resolve once at encoder creation.
        let norm_mode = match std::env::var("CANADA_SCRATCH_VISION_NORM_MODE").as_deref() {
            Ok("baseline") => 0,
            Ok("packed") => 1,
            Err(std::env::VarError::NotPresent) | Ok("fused") => 2,
            _ => anyhow::bail!("vision norm mode must be baseline, packed or fused"),
        };
        Ok(Self {
            cuda: cuda.clone(),
            weights,
            module: cuda.module(kernel)?,
            attention: Operator::load(cuda, attention, "vision_full_attention")?,
            split_attention: {
                let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(attention.join("manifest.json"))?)?;
                let present = manifest["entries"].as_array().is_some_and(|e| e.iter().any(|x| x["name"] == "vision_split_attention"));
                if present { Some(Operator::load(cuda, attention, "vision_split_attention")?) } else { None }
            },
            blas,
            lt_bias: match std::env::var("CANADA_SCRATCH_VISION_LT_BIAS").as_deref() {
                Err(std::env::VarError::NotPresent) | Ok("0") => None,
                Ok("1") => Some(crate::blas_lt::BlasLt::new(cuda)?),
                _ => anyhow::bail!("vision Lt bias policy must be 0 or 1"),
            },
            norm_mode,
        })
    }
    pub fn source_sha256(&self) -> &str {
        &self.weights.source_sha256
    }
    pub fn weight_bytes(&self) -> usize {
        self.weights.bytes()
    }
    pub(crate) fn set_probe_norm_mode(&mut self, mode: u8) -> Result<()> {
        ensure!(mode <= 2, "vision probe norm mode");
        self.norm_mode = mode;
        Ok(())
    }

    fn launch(&self, name: &str, blocks: usize, values: &[u64], stream: &Stream) -> Result<()> {
        ensure!(
            self.cuda.owns_stream(stream) && blocks > 0 && blocks <= u32::MAX as usize,
            "vision launch context/grid"
        );
        let mut words = values.to_vec();
        let mut args: Vec<*mut c_void> = words.iter_mut().map(|v| (v as *mut u64).cast()).collect();
        unsafe {
            self.module.function(name)?.launch(
                [blocks as u32, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )
        }
    }
    fn elementwise(&self, name: &str, count: usize, values: &[u64], stream: &Stream) -> Result<()> {
        self.launch(name, count.div_ceil(256), values, stream)
    }
    fn linear(
        &self,
        x: &Buffer<bf16>,
        y: &Buffer<bf16>,
        scratch: &Buffer<f32>,
        name: &str,
        m: usize,
        n: usize,
        k: usize,
        bias: bool,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            x.len() >= m * k && y.len() >= m * n && (!bias || scratch.len() >= m * n),
            "vision projection buffer extent"
        );
        let w = self.weights.ptr(&format!("{name}.weight"))?;
        if let (true, Some(lt)) = (bias, &self.lt_bias) {
            lt.prepare_bias(m, n, k)?;
            return unsafe { lt.linear_bias_bf16_raw(x.ptr(), w, self.weights.ptr(&format!("{name}.bias"))?,
                y.ptr(), m, n, k, stream) };
        }
        unsafe {
            self.blas.linear_raw(
                x.ptr(),
                w,
                if bias { scratch.ptr() } else { y.ptr() },
                m,
                n,
                k,
                if bias { 0 } else { 14 },
                stream,
            )?;
        }
        if bias {
            self.elementwise(
                "vision_bias",
                m * n,
                &[
                    scratch.ptr(),
                    self.weights.ptr(&format!("{name}.bias"))?,
                    y.ptr(),
                    (m * n) as u64,
                    n as u64,
                ],
                stream,
            )?;
        }
        Ok(())
    }
    fn norm(
        &self,
        x: &Buffer<bf16>,
        y: &Buffer<bf16>,
        name: &str,
        rows: usize,
        stream: &Stream,
    ) -> Result<()> {
        if self.norm_mode != 0 {
            return self.launch("vision_norm_rows", rows.div_ceil(8),
                &[x.ptr(), self.weights.ptr(name)?, y.ptr(), rows as u64], stream);
        }
        self.launch(
            "vision_norm",
            rows,
            &[x.ptr(), self.weights.ptr(name)?, y.ptr()],
            stream,
        )
    }
    fn residual_norm(&self, x: &Buffer<bf16>, delta: &Buffer<bf16>,
        y: &Buffer<bf16>, weight: &str, rows: usize, stream: &Stream) -> Result<()> {
        self.launch("vision_residual_norm_rows", rows.div_ceil(8),
            &[x.ptr(), delta.ptr(), self.weights.ptr(weight)?, y.ptr(), rows as u64], stream)
    }

    pub(crate) fn split_capable(&self) -> bool { self.split_attention.is_some() }
    /// Rows per rank for a row-split encode: a multiple of 64 (keeps every row slice of the
    /// 1176-byte patch rows 16-byte aligned and matches the attention query tile), leaving
    /// the last rank at least one row. None means every rank uses single-rank encoding.
    pub(crate) fn split_rows(g: Geometry) -> Option<usize> {
        let n = g.rows();
        let pad = n.div_ceil(4).next_multiple_of(64);
        (3 * pad < n).then_some(pad)
    }
    /// Per-rank transient bytes of a row-split encode (plus the existing driver allowance).
    pub(crate) fn split_admission_bytes(g: Geometry, root: bool) -> Result<usize> {
        let g = g.validate()?; let n = g.rows(); let m = g.feature_rows();
        let pad = Self::split_rows(g).context("image too small for a row-split encode")?;
        let r = |bytes: usize| bytes.next_multiple_of(2 << 20);
        let bf16 = [pad*1024, pad*1024, pad*3072, pad*1024, pad*1024, pad*1024, 4*pad*1024, 4*pad*1024,
            pad*1024, pad*1024, pad*5632, pad*2816, 4*pad*1024, g.tokens()*5120];
        let rootb = if root { [m*9216, m*5120, m*5120, m*5120] } else { [0; 4] };
        Ok(bf16.iter().chain(rootb.iter()).map(|n| r(n * 2)).sum::<usize>()
            + r((pad*3072).max(if root { m*5120 } else { 0 }) * 4) + 2 * r(n*32*4) + 2 * r(8) + (512 << 20))
    }
    /// Row-split encode across the four TP ranks. Each rank runs every block on its
    /// contiguous quarter of the patch rows and gathers K/V per block; per-row attention
    /// arithmetic matches vision_full_attention. Only `root` forms the aligned span; the
    /// caller broadcasts it exactly as for single-rank encoding.
    pub(crate) fn encode_split(&self, patches: &Buffer<bf16>, geometry: Geometry, comm: &crate::roce::Roce,
        rank: usize, root: usize, stream: &Stream) -> Result<ImageSpan> {
        let split = self.split_attention.as_ref().context("vision split attention artifact absent")?;
        let g = geometry.validate()?;
        let n = g.rows(); let m = g.feature_rows();
        ensure!(rank < 4 && root < 4 && comm.rank() == rank && self.cuda.owns_stream(stream)
            && self.cuda.owns_buffer(patches) && patches.len() == n * 588, "vision split input/context");
        let pad = Self::split_rows(g).context("image too small for a row-split encode")?;
        let r0 = rank * pad; let nl = n.saturating_sub(r0).min(pad);
        ensure!(nl > 0 && nl <= pad && (r0 * 588 * 2) % 16 == 0, "vision split partition");
        let needed = Self::split_admission_bytes(g, rank == root)?;
        let reserve = self.cuda.allocation_reserve_bytes();
        let memory_before = self.cuda.admission_memory()?;
        ensure!(needed + reserve < memory_before.admitted_available,
            crate::memory::InputCapacityError("vision split workspace exceeds available UMA above reserve", needed + reserve));
        let a = |len: usize| -> Result<Buffer<bf16>> { self.cuda.allocate_input::<bf16>(len) };
        let (x, norm, qkv) = (a(pad*1024)?, a(pad*1024)?, a(pad*3072)?);
        let (q, k, v) = (a(pad*1024)?, a(pad*1024)?, a(pad*1024)?);
        let (mut kf, mut vf) = (a(4*pad*1024)?, a(4*pad*1024)?);
        let (attn, delta) = (a(pad*1024)?, a(pad*1024)?);
        let (gate_up, gated) = (a(pad*5632)?, a(pad*2816)?);
        let mut vit = a(4*pad*1024)?;
        let scratch = self.cuda.allocate_input::<f32>((pad*3072).max(if rank == root { m*5120 } else { 0 }))?;
        let co = self.cuda.allocate_input::<f32>(n*32)?;
        let si = self.cuda.allocate_input::<f32>(n*32)?;
        let mut starts = self.cuda.allocate_input::<i32>(1)?;
        let mut lengths = self.cuda.allocate_input::<i32>(2)?;
        starts.upload(&[0])?;
        lengths.upload(&[n as i32, nl as i32])?;
        let span = self.cuda.allocate_input::<bf16>(g.tokens()*5120)?;
        let root_buffers = if rank == root { Some((a(m*9216)?, a(m*5120)?, a(m*5120)?, a(m*5120)?)) } else { None };
        stream.join_default()?;
        let decode_selected = comm.decode_reduction_selected();
        comm.select_decode_reduction(false, stream)?;
        let run = (|| -> Result<()> {
            self.elementwise("vision_rope_table", n * 32, &[co.ptr(), si.ptr(), n as u64, g.width as u64], stream)?;
            let local_patches = unsafe { patches.view(r0 * 588, nl * 588)? };
            let co_l = unsafe { co.view(r0 * 32, nl * 32)? };
            let si_l = unsafe { si.view(r0 * 32, nl * 32)? };
            let count = nl * 1024;
            self.linear(&local_patches, &x, &scratch, "vision.patch_embed.proj", nl, 1024, 588, true, stream)?;
            for i in 0..32 {
                let prefix = format!("vision.blocks.{i}");
                if self.norm_mode != 2 || i == 0 {
                    self.norm(&x, &norm, &format!("{prefix}.norm1.weight"), nl, stream)?;
                }
                self.linear(&norm, &qkv, &scratch, &format!("{prefix}.attn.wqkv"), nl, 3072, 1024, true, stream)?;
                self.elementwise("vision_split_rope", count,
                    &[qkv.ptr(), co_l.ptr(), si_l.ptr(), q.ptr(), k.ptr(), v.ptr(), nl as u64], stream)?;
                unsafe {
                    comm.gather_rows(&k, &mut kf, stream)?;
                    comm.gather_rows(&v, &mut vf, stream)?;
                    split.launch(nl as u32, &[q.ptr(), kf.ptr(), vf.ptr(), starts.ptr(), lengths.ptr(), attn.ptr()], stream)?;
                }
                self.linear(&attn, &delta, &scratch, &format!("{prefix}.attn.wo"), nl, 1024, 1024, true, stream)?;
                if self.norm_mode == 2 {
                    self.residual_norm(&x, &delta, &norm, &format!("{prefix}.norm2.weight"), nl, stream)?;
                } else {
                    self.elementwise("vision_residual", count, &[x.ptr(), delta.ptr(), count as u64], stream)?;
                    self.norm(&x, &norm, &format!("{prefix}.norm2.weight"), nl, stream)?;
                }
                self.linear(&norm, &gate_up, &scratch, &format!("{prefix}.mlp.w1"), nl, 5632, 1024, false, stream)?;
                self.elementwise("vision_swiglu", nl * 2816, &[gate_up.ptr(), gated.ptr(), (nl * 2816) as u64], stream)?;
                self.linear(&gated, &delta, &scratch, &format!("{prefix}.mlp.w2"), nl, 1024, 2816, false, stream)?;
                if self.norm_mode == 2 {
                    let weight = if i == 31 { "vision.norm.weight".to_owned() }
                        else { format!("vision.blocks.{}.norm1.weight", i + 1) };
                    self.residual_norm(&x, &delta, &norm, &weight, nl, stream)?;
                } else {
                    self.elementwise("vision_residual", count, &[x.ptr(), delta.ptr(), count as u64], stream)?;
                }
            }
            if self.norm_mode != 2 {
                self.norm(&x, &norm, "vision.norm.weight", nl, stream)?;
            }
            unsafe { comm.gather_rows(&norm, &mut vit, stream)?; }
            if let Some((unfolded, align1, gelu, features)) = &root_buffers {
                self.elementwise("vision_unfold", m * 9216,
                    &[vit.ptr(), unfolded.ptr(), g.height as u64, g.width as u64, (m * 9216) as u64], stream)?;
                self.linear(unfolded, align1, &scratch, "aligner.w1", m, 5120, 9216, true, stream)?;
                self.elementwise("vision_gelu", m * 5120, &[align1.ptr(), gelu.ptr(), (m * 5120) as u64], stream)?;
                self.linear(gelu, features, &scratch, "aligner.w2", m, 5120, 5120, true, stream)?;
                self.elementwise("vision_span", g.tokens() * 5120, &[features.ptr(), self.weights.ptr("image_start")?,
                    self.weights.ptr("image_newline")?, self.weights.ptr("image_end")?, span.ptr(),
                    g.width.div_ceil(3) as u64, g.tokens() as u64], stream)?;
            }
            Ok(())
        })();
        let synchronized = stream.synchronize();
        let restored = comm.select_decode_reduction(decode_selected, stream);
        run?;
        synchronized?;
        restored?;
        Ok(ImageSpan { data: span, geometry: g })
    }
    pub fn encode(
        &self,
        patches: &Buffer<bf16>,
        geometry: Geometry,
        stream: &Stream,
    ) -> Result<ImageSpan> {
        self.encode_with_taps(patches, geometry, stream, &mut |_, _, _, _, _| Ok(()))
    }
    pub fn encode_with_taps(
        &self,
        patches: &Buffer<bf16>,
        geometry: Geometry,
        stream: &Stream,
        tap: &mut Tap<'_>,
    ) -> Result<ImageSpan> {
        self.encode_layout(patches,geometry,stream,compact_workspace()?,false,tap)
    }
    pub(crate) fn encode_with_workspace(&self,patches:&Buffer<bf16>,geometry:Geometry,
        stream:&Stream,compact:bool)->Result<ImageSpan> {
        self.encode_with_arena(patches,geometry,stream,compact,None)
    }
    pub(crate) fn encode_with_arena(&self,patches:&Buffer<bf16>,geometry:Geometry,
        stream:&Stream,compact:bool,arena:Option<&std::rc::Rc<crate::cuda::RowArena>>)->Result<ImageSpan> {
        if let Some(arena)=arena {
            ensure!(geometry.scratch_bytes(compact)?<=arena.bytes(),"image scratch arena too small");
            // The worker lends only completed prefill embedding storage. All
            // owners execute serially; drain the prior use before overwriting.
            stream.synchronize()?;
            self.cuda.with_row_arena(arena,||self.encode_layout(patches,geometry,stream,compact,true,&mut |_,_,_,_,_|Ok(())))
        }else {self.encode_layout(patches,geometry,stream,compact,false,&mut |_,_,_,_,_|Ok(()))}
    }
    fn encode_layout(&self,patches:&Buffer<bf16>,geometry:Geometry,stream:&Stream,
        compact:bool,borrowed:bool,tap:&mut Tap<'_>)->Result<ImageSpan> {
        let g = geometry.validate()?;
        let n = g.rows();
        let m = g.feature_rows();
        let count = n * 1024;
        ensure!(
            self.cuda.owns_stream(stream)
                && self.cuda.owns_buffer(patches)
                && patches.len() == n * 588,
            "vision input extent/context"
        );
        let layout = WorkspacePlan::new(g)?;
        let needed = if borrowed {layout.arena_admission_bytes(false)}else{layout.admission_bytes(false,compact)}; // patches are resident
        let reserve = self.cuda.allocation_reserve_bytes();
        let memory_before = self.cuda.admission_memory()?;
        ensure!(
            needed + reserve < memory_before.admitted_available,
            crate::memory::InputCapacityError("vision workspace exceeds available UMA above reserve", needed + reserve)
        );
        let [x, norm, qkv, _q, _k, _v, attn, delta, _gate_up, _gated,
            unfolded, align1, gelu, features, span] = layout.bf16;
        let [scratch, co, si] = layout.fp32_backings(compact);
        let [starts, lengths] = layout.i32;
        let backing=layout.backings(compact);
        macro_rules! scratch_buffer {
            ($ty:ty,$len:expr)=>{if borrowed {self.cuda.allocate_rows::<$ty>($len)?}
                else{self.cuda.allocate_input::<$ty>($len)?}};
        }
        let x = scratch_buffer!(bf16,x);
        let gate_up = scratch_buffer!(bf16,backing[8]);
        let gated = scratch_buffer!(bf16,backing[9]);
        let q = scratch_buffer!(bf16,backing[3]);
        let k = scratch_buffer!(bf16,backing[4]);
        let v = scratch_buffer!(bf16,backing[5]);
        // SAFETY: all accesses below use the same stream, with these lifetimes:
        // norm -> QKV -> split/attention -> delta/residual -> norm -> gate/up
        // -> gated -> delta/residual. The aligner starts after the last block.
        // Taps may inspect only during their callback. Parents remain in this
        // scope through synchronization on success and on execution failure.
        let norm = if compact {unsafe{gated.view(0,norm)?}}else{scratch_buffer!(bf16,norm)};
        let qkv = if compact {unsafe{gate_up.view(0,qkv)?}}else{scratch_buffer!(bf16,qkv)};
        let attn = scratch_buffer!(bf16,attn);
        let delta = if compact {unsafe{gate_up.view(0,delta)?}}else{scratch_buffer!(bf16,delta)};
        let scratch = scratch_buffer!(f32,scratch);
        let co = scratch_buffer!(f32,co);
        let si = scratch_buffer!(f32,si);
        let unfolded = if compact {unsafe{gate_up.view(0,unfolded)?}}else{scratch_buffer!(bf16,unfolded)};
        let align1 = if compact {unsafe{q.view(0,align1)?}}else{scratch_buffer!(bf16,align1)};
        let gelu = if compact {unsafe{k.view(0,gelu)?}}else{scratch_buffer!(bf16,gelu)};
        let features = if compact {unsafe{v.view(0,features)?}}else{scratch_buffer!(bf16,features)};
        let span = self.cuda.allocate_input::<bf16>(span)?;
        let mut starts = scratch_buffer!(i32,starts);
        let mut lengths = scratch_buffer!(i32,lengths);
        starts.upload(&[0])?;
        lengths.upload(&[n as i32])?;
        stream.join_default()?;
        let memory_allocated = self.cuda.admission_memory()?;
        eprintln!("vision workspace admission: {}", serde_json::json!({
            "geometry":g,"compact_workspace":compact,"borrowed_prefill_scratch":borrowed,"admission_bytes":needed,"reserve_bytes":reserve,"memory_before":memory_before,
            "memory_allocated":memory_allocated
        }));
        ensure!(memory_allocated.admitted_available > reserve,
            crate::memory::InputCapacityError("vision allocations consumed the UMA reserve", needed + reserve));
        // Every local tensor/module outlives queued work even if a tap errors.
        let run = (|| -> Result<()> {
            self.elementwise(
                "vision_rope_table",
                n * 32,
                &[co.ptr(), si.ptr(), n as u64, g.width as u64],
                stream,
            )?;
            tap("cos", co.ptr(), n * 32, "F32", stream)?;
            tap("sin", si.ptr(), n * 32, "F32", stream)?;
            self.linear(
                patches,
                &x,
                &scratch,
                "vision.patch_embed.proj",
                n,
                1024,
                588,
                true,
                stream,
            )?;
            tap("patch", x.ptr(), count, "BF16", stream)?;
            for i in 0..32 {
                let prefix = format!("vision.blocks.{i}");
                if self.norm_mode != 2 || i == 0 {
                    self.norm(&x, &norm, &format!("{prefix}.norm1.weight"), n, stream)?;
                }
                if i == 0 {
                    tap("norm1", norm.ptr(), count, "BF16", stream)?;
                }
                self.linear(
                    &norm,
                    &qkv,
                    &scratch,
                    &format!("{prefix}.attn.wqkv"),
                    n,
                    3072,
                    1024,
                    true,
                    stream,
                )?;
                if i == 0 {
                    tap("qkv", qkv.ptr(), n * 3072, "BF16", stream)?;
                }
                self.elementwise(
                    "vision_split_rope",
                    count,
                    &[
                        qkv.ptr(),
                        co.ptr(),
                        si.ptr(),
                        q.ptr(),
                        k.ptr(),
                        v.ptr(),
                        n as u64,
                    ],
                    stream,
                )?;
                if i == 0 {
                    for (name, b) in [("q", &q), ("k", &k), ("v", &v)] {
                        tap(name, b.ptr(), count, "BF16", stream)?;
                    }
                }
                unsafe {
                    self.attention.launch(
                        n as u32,
                        &[
                            q.ptr(),
                            k.ptr(),
                            v.ptr(),
                            starts.ptr(),
                            lengths.ptr(),
                            attn.ptr(),
                        ],
                        stream,
                    )?;
                }
                if i == 0 {
                    tap("attention", attn.ptr(), count, "BF16", stream)?;
                }
                self.linear(
                    &attn,
                    &delta,
                    &scratch,
                    &format!("{prefix}.attn.wo"),
                    n,
                    1024,
                    1024,
                    true,
                    stream,
                )?;
                if i == 0 {
                    tap("attention_projection", delta.ptr(), count, "BF16", stream)?;
                }
                if self.norm_mode == 2 {
                    self.residual_norm(&x, &delta, &norm, &format!("{prefix}.norm2.weight"), n, stream)?;
                } else {
                    self.elementwise("vision_residual", count,
                        &[x.ptr(), delta.ptr(), count as u64], stream)?;
                }
                if i == 0 {
                    tap("attention_residual", x.ptr(), count, "BF16", stream)?;
                }
                if self.norm_mode != 2 {
                    self.norm(&x, &norm, &format!("{prefix}.norm2.weight"), n, stream)?;
                }
                if i == 0 {
                    tap("norm2", norm.ptr(), count, "BF16", stream)?;
                }
                self.linear(
                    &norm,
                    &gate_up,
                    &scratch,
                    &format!("{prefix}.mlp.w1"),
                    n,
                    5632,
                    1024,
                    false,
                    stream,
                )?;
                if i == 0 {
                    tap("gate_up", gate_up.ptr(), n * 5632, "BF16", stream)?;
                }
                self.elementwise(
                    "vision_swiglu",
                    n * 2816,
                    &[gate_up.ptr(), gated.ptr(), (n * 2816) as u64],
                    stream,
                )?;
                if i == 0 {
                    tap("gated", gated.ptr(), n * 2816, "BF16", stream)?;
                }
                self.linear(
                    &gated,
                    &delta,
                    &scratch,
                    &format!("{prefix}.mlp.w2"),
                    n,
                    1024,
                    2816,
                    false,
                    stream,
                )?;
                if i == 0 {
                    tap("mlp_projection", delta.ptr(), count, "BF16", stream)?;
                }
                if self.norm_mode == 2 {
                    let weight = if i == 31 { "vision.norm.weight".to_owned() }
                        else { format!("vision.blocks.{}.norm1.weight", i+1) };
                    self.residual_norm(&x, &delta, &norm, &weight, n, stream)?;
                } else {
                    self.elementwise("vision_residual", count,
                        &[x.ptr(), delta.ptr(), count as u64], stream)?;
                }
                if [0, 7, 15, 23, 31].contains(&i) {
                    tap(&format!("block{i}"), x.ptr(), count, "BF16", stream)?;
                }
            }
            if self.norm_mode != 2 {
                self.norm(&x, &norm, "vision.norm.weight", n, stream)?;
            }
            tap("vit", norm.ptr(), count, "BF16", stream)?;
            self.elementwise(
                "vision_unfold",
                m * 9216,
                &[
                    norm.ptr(),
                    unfolded.ptr(),
                    g.height as u64,
                    g.width as u64,
                    (m * 9216) as u64,
                ],
                stream,
            )?;
            tap("unfolded", unfolded.ptr(), m * 9216, "BF16", stream)?;
            self.linear(
                &unfolded,
                &align1,
                &scratch,
                "aligner.w1",
                m,
                5120,
                9216,
                true,
                stream,
            )?;
            tap("align1", align1.ptr(), m * 5120, "BF16", stream)?;
            self.elementwise(
                "vision_gelu",
                m * 5120,
                &[align1.ptr(), gelu.ptr(), (m * 5120) as u64],
                stream,
            )?;
            tap("gelu", gelu.ptr(), m * 5120, "BF16", stream)?;
            self.linear(
                &gelu,
                &features,
                &scratch,
                "aligner.w2",
                m,
                5120,
                5120,
                true,
                stream,
            )?;
            tap("features", features.ptr(), m * 5120, "BF16", stream)?;
            self.elementwise(
                "vision_span",
                g.tokens() * 5120,
                &[
                    features.ptr(),
                    self.weights.ptr("image_start")?,
                    self.weights.ptr("image_newline")?,
                    self.weights.ptr("image_end")?,
                    span.ptr(),
                    g.width.div_ceil(3) as u64,
                    g.tokens() as u64,
                ],
                stream,
            )?;
            tap("span", span.ptr(), g.tokens() * 5120, "BF16", stream)?;
            Ok(())
        })();
        let synchronized = stream.synchronize();
        run?;
        synchronized?;
        Ok(ImageSpan {
            data: span,
            geometry: g,
        })
    }
}

#[cfg(test)]
mod workspace_tests {
    use super::*;
    #[test]
    fn compact_views_fit_every_supported_geometry_without_increasing_admission() {
        let mut geometries=0;
        for height in 1..=3063 {
            for width in 1..=3063.min(9198/height) {
                let g=Geometry{height,width};if g.validate().is_err(){continue;}
                let p=WorkspacePlan::new(g).unwrap();let b=p.backings(true);
                for (view,owner) in [(1,9),(2,8),(7,8),(10,8),(11,3),(12,4),(13,5)] {
                    assert!(p.bf16[view]<=b[owner],"{g:?}: view{view}/owner{owner}");
                }
                assert!(p.fp32_backings(true)[0]>=p.bf16[2]);
                assert!(p.fp32_backings(true)[0]>=p.bf16[11]);
                assert!(p.admission_bytes(true,true)<=p.admission_bytes(true,false));
                assert!(p.scratch_bytes(true)<=8192*30720*2,"{g:?}: scratch exceeds prefill input arena");
                assert!(p.arena_admission_bytes(true)<=p.admission_bytes(true,true));
                geometries+=1;
            }
        }
        assert!(geometries>10_000);
    }
}
