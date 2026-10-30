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
use anyhow::{Result, ensure};
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
    pub(crate) fn workspace_bytes(self, include_patches: bool) -> Result<usize> {
        Ok(WorkspacePlan::new(self)?.admission_bytes(include_patches))
    }
}

/// Allocation lengths and admission accounting share one layout. All buffers
/// coexist through encoding; no lifetime overlap is assumed to save memory.
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
    fn admission_bytes(&self, include_patches: bool) -> usize {
        // Round every allocation up to 2MiB and keep another 64MiB for
        // driver/allocator overhead. This is additional to the 8GiB UMA floor.
        let rounded = |bytes: usize| bytes.next_multiple_of(2 << 20);
        self.bf16.iter().map(|n| rounded(n * 2)).sum::<usize>()
            + self.fp32.iter().map(|n| rounded(n * 4)).sum::<usize>()
            + self.i32.iter().map(|n| rounded(n * 4)).sum::<usize>()
            + if include_patches { rounded(self.patches * 2) } else { 0 }
            + (64 << 20)
    }
}

pub struct Vision {
    cuda: Cuda,
    weights: Weights,
    module: Module,
    attention: Operator,
    blas: Blas,
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
        Ok(Self {
            cuda: cuda.clone(),
            weights,
            module: cuda.module(kernel)?,
            attention: Operator::load(cuda, attention, "vision_full_attention")?,
            blas,
        })
    }
    pub fn source_sha256(&self) -> &str {
        &self.weights.source_sha256
    }
    pub fn weight_bytes(&self) -> usize {
        self.weights.bytes()
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
            x.len() >= m * k && y.len() >= m * n && scratch.len() >= m * n,
            "vision projection buffer extent"
        );
        let w = self.weights.ptr(&format!("{name}.weight"))?;
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
        self.launch(
            "vision_norm",
            rows,
            &[x.ptr(), self.weights.ptr(name)?, y.ptr()],
            stream,
        )
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
        let needed = layout.admission_bytes(false); // patches are already resident
        let memory_before = self.cuda.admission_memory()?;
        ensure!(
            needed + (8usize << 30) < memory_before.admitted_available,
            "vision workspace exceeds available UMA above reserve"
        );
        let [x, norm, qkv, q, k, v, attn, delta, gate_up, gated,
            unfolded, align1, gelu, features, span] = layout.bf16;
        let [scratch, co, si] = layout.fp32;
        let [starts, lengths] = layout.i32;
        let x = self.cuda.allocate::<bf16>(x)?;
        let norm = self.cuda.allocate::<bf16>(norm)?;
        let qkv = self.cuda.allocate::<bf16>(qkv)?;
        let q = self.cuda.allocate::<bf16>(q)?;
        let k = self.cuda.allocate::<bf16>(k)?;
        let v = self.cuda.allocate::<bf16>(v)?;
        let attn = self.cuda.allocate::<bf16>(attn)?;
        let delta = self.cuda.allocate::<bf16>(delta)?;
        let gate_up = self.cuda.allocate::<bf16>(gate_up)?;
        let gated = self.cuda.allocate::<bf16>(gated)?;
        let scratch = self.cuda.allocate::<f32>(scratch)?;
        let co = self.cuda.allocate::<f32>(co)?;
        let si = self.cuda.allocate::<f32>(si)?;
        let unfolded = self.cuda.allocate::<bf16>(unfolded)?;
        let align1 = self.cuda.allocate::<bf16>(align1)?;
        let gelu = self.cuda.allocate::<bf16>(gelu)?;
        let features = self.cuda.allocate::<bf16>(features)?;
        let span = self.cuda.allocate::<bf16>(span)?;
        let mut starts = self.cuda.allocate::<i32>(starts)?;
        let mut lengths = self.cuda.allocate::<i32>(lengths)?;
        starts.upload(&[0])?;
        lengths.upload(&[n as i32])?;
        stream.join_default()?;
        let memory_allocated = self.cuda.admission_memory()?;
        eprintln!("vision workspace admission: {}", serde_json::json!({
            "geometry":g,"admission_bytes":needed,"memory_before":memory_before,
            "memory_allocated":memory_allocated
        }));
        ensure!(memory_allocated.admitted_available > (8usize << 30),
            "vision allocations consumed the UMA reserve");
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
                self.norm(&x, &norm, &format!("{prefix}.norm1.weight"), n, stream)?;
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
                self.elementwise(
                    "vision_residual",
                    count,
                    &[x.ptr(), delta.ptr(), count as u64],
                    stream,
                )?;
                if i == 0 {
                    tap("attention_residual", x.ptr(), count, "BF16", stream)?;
                }
                self.norm(&x, &norm, &format!("{prefix}.norm2.weight"), n, stream)?;
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
                self.elementwise(
                    "vision_residual",
                    count,
                    &[x.ptr(), delta.ptr(), count as u64],
                    stream,
                )?;
                if [0, 7, 15, 23, 31].contains(&i) {
                    tap(&format!("block{i}"), x.ptr(), count, "BF16", stream)?;
                }
            }
            self.norm(&x, &norm, "vision.norm.weight", n, stream)?;
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
