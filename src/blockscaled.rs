//! Independent conversion from canonical 32x32 UE8M0 scales to the SM12x MMA
//! byte layout. FP8 values/scales stay unchanged; K padding contributes zeros.
use crate::{
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, Module, Stream},
    weights::Weights,
};
use anyhow::{Context, Result, ensure};
use std::ffi::c_void;
use std::rc::Rc;
pub fn physical_k(k: usize) -> Result<usize> {
    ensure!(k > 0 && k % 32 == 0, "K must be a positive multiple of 32");
    Ok(k.checked_add(127).context("K overflow")? & !127)
}
pub fn scale_bytes(n: usize, k: usize) -> Result<usize> {
    ensure!(n > 0, "empty matrix");
    let k = physical_k(k)?;
    (n.checked_add(127).context("N overflow")? & !127)
        .checked_mul(k / 32)
        .context("scale allocation overflow")
}
fn launch_scales(
    cuda: &Cuda,
    module: &Module,
    stream: &Stream,
    source: u64,
    n: usize,
    k: usize,
) -> Result<Buffer<u8>> {
    let padded_k = physical_k(k)?;
    let scales = cuda.allocate::<u8>(scale_bytes(n, k)?)?;
    let mut src = source;
    let mut dst = scales.ptr();
    let mut nr = i32::try_from(n)?;
    let mut kr = i32::try_from(k)?;
    let mut pk = i32::try_from(padded_k)?;
    let mut args: Vec<*mut c_void> = vec![
        (&mut src as *mut u64).cast(),
        (&mut dst as *mut u64).cast(),
        (&mut nr as *mut i32).cast(),
        (&mut kr as *mut i32).cast(),
        (&mut pk as *mut i32).cast(),
    ];
    unsafe {
        module.function("ca_block_scales_mma")?.launch(
            [u32::try_from(scales.len().div_ceil(256))?, 1, 1],
            [256, 1, 1],
            0,
            stream,
            &mut args,
        )
    }?;
    Ok(scales)
}
pub fn pack_scales_gpu(
    source: &[u8],
    n: usize,
    k: usize,
    core: &std::path::Path,
) -> Result<Vec<u8>> {
    ensure!(
        n > 0 && n <= 65536 && k <= 65536 && n % 32 == 0 && source.len() == n / 32 * (k / 32),
        "canonical scale source shape mismatch"
    );
    physical_k(k)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let module = cuda.module(&std::fs::read(core)?)?;
    let stream = cuda.stream()?;
    let mut raw = cuda.allocate::<u8>(source.len())?;
    raw.upload(source)?;
    let packed = launch_scales(&cuda, &module, &stream, raw.ptr(), n, k)?;
    stream.synchronize()?;
    packed.download()
}
pub struct Matrix {
    _source_owner: Rc<Weights>,
    values: Option<Buffer<u8>>,
    scales: Buffer<u8>,
    source_pointer: u64,
    pub n: usize,
    pub k: usize,
    pub padded_k: usize,
}
impl Matrix {
    pub(crate) fn in_context(&self, cuda: &Cuda) -> bool {
        self._source_owner.in_context(cuda)
            && cuda.owns_buffer(&self.scales)
            && self.values.as_ref().is_none_or(|v| cuda.owns_buffer(v))
    }
    pub(crate) fn immutable_regions(&self) -> Result<[(u64, usize); 2]> {
        let bytes = self
            .n
            .checked_mul(self.padded_k)
            .context("FP8 weight size overflow")?;
        if let Some(values) = &self.values {
            ensure!(values.bytes() == bytes, "FP8 padded allocation size");
        }
        Ok([(self.values(), bytes), (self.scales(), self.scales.bytes())])
    }
    pub fn values(&self) -> u64 {
        self.values
            .as_ref()
            .map_or(self.source_pointer, |b| b.ptr())
    }
    pub fn scales(&self) -> u64 {
        self.scales.ptr()
    }
    /// Shared gate/up concatenation over original canonical row blocks.
    /// Source order is [gate; up], distinct from the routed expert layout.
    pub fn prepare_shared(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: &Rc<Weights>,
        layer: usize,
    ) -> Result<Self> {
        Self::prepare_shared_named(
            cuda,
            module,
            stream,
            weights,
            crate::layer_id::LayerId::target(layer)?,
        )
    }
    pub fn prepare_shared_named(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: &Rc<Weights>,
        id: crate::layer_id::LayerId,
    ) -> Result<Self> {
        let p = format!("{}.ffn.shared_experts", id.prefix());
        let aw = weights.view(&format!("{p}.w1.weight"))?;
        let bw = weights.view(&format!("{p}.w3.weight"))?;
        let a = weights.view(&format!("{p}.w1.scale"))?;
        let b = weights.view(&format!("{p}.w3.scale"))?;
        ensure!(
            aw.dtype == Dtype::Fp8
                && bw.dtype == Dtype::Fp8
                && aw.shape == [576, 5120]
                && bw.shape == [576, 5120]
                && aw.offset + aw.bytes == bw.offset
                && a.dtype == Dtype::E8m0
                && b.dtype == Dtype::E8m0
                && a.shape == [18, 160]
                && b.shape == [18, 160]
                && a.offset + a.bytes == b.offset,
            "shared gate/up atlas is not the exact contiguous TP4 layout"
        );
        Ok(Self {
            _source_owner: weights.clone(),
            values: None,
            source_pointer: weights.ptr(&format!("{p}.w1.weight"))?,
            scales: launch_scales(
                cuda,
                module,
                stream,
                weights.ptr(&format!("{p}.w1.scale"))?,
                1152,
                5120,
            )?,
            n: 1152,
            k: 5120,
            padded_k: 5120,
        })
    }
    /// Adjacent Q-A/KV rows form one matrix over the original atlas. No FP8
    /// weights are copied or requantized; only the MMA scale view is prepared.
    pub fn prepare_qkv(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: &Rc<Weights>,
        layer: usize,
    ) -> Result<Self> {
        Self::prepare_qkv_range(cuda, module, stream, weights, layer, None)
    }
    pub fn prepare_qkv_split(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: &Rc<Weights>,
        layer: usize,
        rank: usize,
    ) -> Result<Self> {
        ensure!(rank < 4, "QKV split rank outside TP4");
        Self::prepare_qkv_range(cuda, module, stream, weights, layer, Some(rank))
    }
    fn prepare_qkv_range(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: &Rc<Weights>,
        layer: usize,
        rank: Option<usize>,
    ) -> Result<Self> {
        let p = format!("layers.{layer}.attn");
        let aw = weights.view(&format!("{p}.wq_a.weight"))?;
        let bw = weights.view(&format!("{p}.wkv.weight"))?;
        let a = weights.view(&format!("{p}.wq_a.scale"))?;
        let b = weights.view(&format!("{p}.wkv.scale"))?;
        ensure!(
            aw.dtype == Dtype::Fp8
                && bw.dtype == Dtype::Fp8
                && aw.shape == [1280, 5120]
                && bw.shape == [512, 5120]
                && aw.offset + aw.bytes == bw.offset
                && a.dtype == Dtype::E8m0
                && b.dtype == Dtype::E8m0
                && a.shape == [40, 160]
                && b.shape == [16, 160]
                && a.offset + a.bytes == b.offset,
            "QKV atlas is not the exact contiguous canonical layout"
        );
        let n = if rank.is_some() { 448 } else { 1792 };
        let first = rank.unwrap_or(0) * n;
        let source_pointer = weights.ptr(&format!("{p}.wq_a.weight"))? + (first * 5120) as u64;
        let scales = launch_scales(
            cuda,
            module,
            stream,
            weights.ptr(&format!("{p}.wq_a.scale"))? + (first / 32 * 160) as u64,
            n,
            5120,
        )?;
        Ok(Self {
            _source_owner: weights.clone(),
            values: None,
            scales,
            source_pointer,
            n,
            k: 5120,
            padded_k: 5120,
        })
    }
    /// The source atlas must outlive this view and all captured GPU work.
    pub(crate) fn prepare_draft_main_split(
        cuda: &Cuda, module: &Module, stream: &Stream, weights: &Rc<Weights>, rank: usize,
    ) -> Result<Self> {
        let w = weights.view("mtp.0.main_proj.weight")?;
        let s = weights.view("mtp.0.main_proj.scale")?;
        ensure!(rank < 4 && w.dtype == Dtype::Fp8 && w.shape == [5120,15360]
            && s.dtype == Dtype::E8m0 && s.shape == [160,480], "draft main TP4 source geometry");
        let first = rank * 1280;
        Ok(Self {
            _source_owner: weights.clone(), values: None,
            source_pointer: weights.ptr("mtp.0.main_proj.weight")? + (first * 15360) as u64,
            scales: launch_scales(cuda, module, stream,
                weights.ptr("mtp.0.main_proj.scale")? + (first / 32 * 480) as u64,1280,15360)?,
            n:1280,k:15360,padded_k:15360,
        })
    }
    /// The source atlas must outlive this view and all captured GPU work.
    pub fn prepare(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: &Rc<Weights>,
        prefix: &str,
    ) -> Result<Self> {
        let wn = format!("{prefix}.weight");
        let sn = format!("{prefix}.scale");
        let w = weights.view(&wn)?;
        let s = weights.view(&sn)?;
        ensure!(
            w.dtype == Dtype::Fp8 && s.dtype == Dtype::E8m0 && w.shape.len() == 2,
            "not canonical block-FP8 weight"
        );
        let n = usize::try_from(w.shape[0])?;
        let k = usize::try_from(w.shape[1])?;
        ensure!(
            n % 32 == 0 && s.shape == [n as u64 / 32, k as u64 / 32],
            "scale geometry mismatch"
        );
        let padded_k = physical_k(k)?;
        let source_pointer = weights.ptr(&wn)?;
        let scales = launch_scales(cuda, module, stream, weights.ptr(&sn)?, n, k)?;
        let mut nr = n as i32;
        let mut kr = k as i32;
        let mut pk = padded_k as i32;
        let values = if padded_k == k {
            None
        } else {
            let values = cuda.allocate::<u8>(
                n.checked_mul(padded_k)
                    .context("matrix allocation overflow")?,
            )?;
            let mut src = source_pointer;
            let mut dst = values.ptr();
            let mut args: Vec<*mut c_void> = vec![
                (&mut src as *mut u64).cast(),
                (&mut dst as *mut u64).cast(),
                (&mut nr as *mut i32).cast(),
                (&mut kr as *mut i32).cast(),
                (&mut pk as *mut i32).cast(),
            ];
            unsafe {
                module.function("ca_pad_fp8_columns")?.launch(
                    [values.len().div_ceil(256) as u32, 1, 1],
                    [256, 1, 1],
                    0,
                    stream,
                    &mut args,
                )
            }?;
            Some(values)
        };
        Ok(Self {
            _source_owner: weights.clone(),
            values,
            scales,
            source_pointer,
            n,
            k,
            padded_k,
        })
    }
}
