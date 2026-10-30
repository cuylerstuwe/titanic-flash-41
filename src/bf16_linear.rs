//! Vendor BF16-input projections with explicit output precision and ownership.
//! Compression ratio 2 retains FP32 until pooling; ratio 1 rounds at GEMM.
use crate::{
    blas::Blas,
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, DeviceElement, Stream},
    triton_ops::Operator,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{cell::Cell, path::Path, rc::Rc};
mod sealed {
    pub trait Sealed {}
    impl Sealed for half::bf16 {}
    impl Sealed for f32 {}
}
pub trait Output: DeviceElement + sealed::Sealed {
    const CUDA_TYPE: i32;
}
impl Output for bf16 {
    const CUDA_TYPE: i32 = 14;
}
impl Output for f32 {
    const CUDA_TYPE: i32 = 0;
}
pub struct Projection<T: Output> {
    cuda: Cuda,
    _weights: Rc<Weights>,
    blas: Rc<Blas>,
    pointer: u64,
    rows: usize,
    n: usize,
    k: usize,
    output: Buffer<T>,
    stream: Cell<usize>,
    head_gemv: Option<Operator>,
    head_gemv_active: Cell<bool>,
    lossless_head: Option<crate::lossless_head::Head>,
    lossless_active: Cell<bool>,
}
impl<T: Output> Projection<T> {
    pub fn new(
        cuda: &Cuda,
        weights: Rc<Weights>,
        blas: Rc<Blas>,
        prefix: &str,
        rows: usize,
    ) -> Result<Self> {
        let name = format!("{prefix}.weight");
        let w = weights.view(&name)?;
        ensure!(
            w.dtype == Dtype::Bf16 && w.shape.len() == 2,
            "BF16 projection source dtype/shape"
        );
        let n = w.shape[0] as usize;
        let k = w.shape[1] as usize;
        let pointer = weights.ptr(&name)?;
        Self::view(cuda, weights, blas, pointer, rows, n, k)
    }
    pub fn compressor_pair(
        cuda: &Cuda,
        weights: Rc<Weights>,
        blas: Rc<Blas>,
        layer: usize,
        rows: usize,
    ) -> Result<Self> {
        ensure!(
            T::CUDA_TYPE == 0 && [2, 8, 14].contains(&layer),
            "FP32-output ratio-2 compressor source"
        );
        let p = format!("layers.{layer}.attn.compressor");
        let a = weights.view(&format!("{p}.wkv.weight"))?;
        let b = weights.view(&format!("{p}.wgate.weight"))?;
        ensure!(
            a.dtype == Dtype::Bf16
                && b.dtype == Dtype::Bf16
                && a.shape == [512, 5120]
                && b.shape == a.shape
                && a.offset + a.bytes == b.offset,
            "compressor KV/score sources are not adjacent canonical rows"
        );
        let pointer = weights.ptr(&format!("{p}.wkv.weight"))?;
        Self::view(cuda, weights, blas, pointer, rows, 1024, 5120)
    }
    fn view(
        cuda: &Cuda,
        weights: Rc<Weights>,
        blas: Rc<Blas>,
        pointer: u64,
        rows: usize,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        ensure!(
            weights.in_context(cuda)
                && (1..=crate::config::MAX_WIDE_ROWS).contains(&rows)
                && (1..=32768).contains(&n)
                && (1..=32768).contains(&k),
            "BF16 projection geometry/context"
        );
        Ok(Self {
            cuda: cuda.clone(),
            _weights: weights,
            blas,
            pointer,
            rows,
            n,
            k,
            output: cuda.allocate_rows(rows * n)?,
            stream: Cell::new(0),
            head_gemv: None,
            head_gemv_active: Cell::new(false),
            lossless_head:None,
            lossless_active:Cell::new(false),
        })
    }
    /// Attach only the qualified vocabulary geometry. Other projection shapes
    /// retain their vendor implementation and output boundary.
    pub(crate) fn with_head_gemv(self, root: &Path) -> Result<Self> {
        ensure!(
            T::CUDA_TYPE == 0 && self.n == 32320 && self.k == 5120,
            "head GEMV requires canonical BF16 vocabulary weights and FP32 output"
        );
        if ![1, 3, 6].contains(&self.rows) {
            return Ok(self);
        }
        self.attach_head_gemv(root)
    }
    /// s577 timing and s579 driver/graph ABI check: canonical operands,
    /// single-split tensor-core FP32 accumulation and output.
    pub(crate) fn with_head_tiled16(mut self,root:&Path)->Result<Self> {
        ensure!(T::CUDA_TYPE==0 && self.rows==16 && self.n==32320 && self.k==5120,
            "tiled head requires canonical M16 BF16/FP32 geometry");
        let op=Operator::load(&self.cuda,root,"target_head_m16")?;
        ensure!(op.entry.arguments.iter().map(|a|(a.name.as_str(),a.kind.as_str())).collect::<Vec<_>>()
            ==[("X","*bf16"),("W","*bf16"),("Y","*fp32"),("P","*fp32")]
            && op.entry.grid.rows_div==128 && op.entry.grid.x_mul==1 && op.entry.grid.x_add==0
            && op.entry.grid.y==1 && op.entry.grid.z==1 && op.entry.metadata.num_warps==4
            && op.entry.artifact_sha256=="006a932620a43f4c6c00e4a429a1cb5a766b50cc50c9268bfc0c1b76972c1c9f",
            "tiled head operator geometry/precision ABI");
        self.head_gemv=Some(op);self.head_gemv_active.set(true);Ok(self)
    }
    pub(crate) fn with_draft_head_gemv(self, root: &Path) -> Result<Self> {
        ensure!(T::CUDA_TYPE == 0 && self.rows == 5 && self.n == 32320 && self.k == 5120,
            "draft head GEMV requires the canonical M5 vocabulary geometry");
        self.attach_head_gemv(root)
    }
    fn attach_head_gemv(mut self, root: &Path) -> Result<Self> {
        let op = Operator::load(&self.cuda, root, &format!("target_head_m{}", self.rows))?;
        let (columns, warps) = if self.rows == 1 { (4, 8) } else { (1, 4) };
        ensure!(
            op.entry
                .arguments
                .iter()
                .map(|a| (a.name.as_str(), a.kind.as_str()))
                .collect::<Vec<_>>()
                == [("X", "*bf16"), ("W", "*bf16"), ("Y", "*fp32")]
                && op.entry.grid.rows_div == columns
                && op.entry.grid.x_mul == 1
                && op.entry.grid.x_add == 0
                && op.entry.grid.y == 1
                && op.entry.grid.z == 1
                && op.entry.metadata.num_warps == warps,
            "head GEMV operator geometry/precision ABI"
        );
        self.head_gemv = Some(op);
        self.head_gemv_active.set(true);
        Ok(self)
    }
    pub(crate) fn uses_head_gemv(&self) -> bool {
        self.head_gemv.is_some() && self.head_gemv_active.get()
    }
    pub(crate) fn with_lossless_head(mut self,root:&Path,stream:&Stream)->Result<Self> {
        ensure!(T::CUDA_TYPE==0 && self.rows==6 && self.n==32320 && self.k==5120
            && self.uses_head_gemv(),"lossless head requires canonical M6 BF16/FP32 GEMV");
        self.lossless_head=Some(crate::lossless_head::Head::new(&self.cuda,self.pointer,root,stream)?);
        self.lossless_active.set(true);Ok(self)
    }
    pub(crate) fn set_lossless_for_probe(&self,active:bool)->Result<()> {
        ensure!(self.lossless_head.is_some() && self.rows==6
            && std::env::var("CANADA_SCRATCH_LOSSLESS_HEAD_SCREEN").as_deref()==Ok("1"),
            "lossless head switch requires the isolated loaded comparison");
        self.lossless_active.set(active);Ok(())
    }
    pub(crate) fn set_draft_head_gemv_for_probe(&self, active: bool) -> Result<()> {
        ensure!(self.rows == 5 && self.head_gemv.is_some()
            && std::env::var("CANADA_SCRATCH_DRAFT_HEAD_GEMV_SCREEN").as_deref() == Ok("1"),
            "M5 GEMV switch requires the isolated three-head screen");
        self.head_gemv_active.set(active);
        Ok(())
    }
    pub fn run(&self, input: &Buffer<bf16>, stream: &Stream) -> Result<()> {
        ensure!(
            self.cuda.owns_buffer(input)
                && self.cuda.owns_stream(stream)
                && input.len() == self.rows * self.k,
            "BF16 projection input shape/context"
        );
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == raw,
            "vendor projection needs one ordered stream"
        );
        self.stream.set(raw);
        if self.lossless_active.get() {
            return self.lossless_head.as_ref().unwrap().run(input.ptr(),self.output.ptr(),stream);
        }
        if let Some(op) = self.head_gemv.as_ref().filter(|_| self.head_gemv_active.get()) {
            let args=[input.ptr(),self.pointer,self.output.ptr(),self.output.ptr()];
            // This kernel's grid enumerates vocabulary rows. The batch size
            // is a qualified compile-time constant in the selected artifact.
            return unsafe {
                op.launch(
                    self.n as u32,
                    &args[..if self.rows==16 {4}else{3}],
                    stream,
                )
            };
        }
        unsafe {
            self.blas.linear_raw(
                input.ptr(),
                self.pointer,
                self.output.ptr(),
                self.rows,
                self.n,
                self.k,
                T::CUDA_TYPE,
                stream,
            )
        }
    }
    pub fn output(&self) -> &Buffer<T> {
        &self.output
    }
}
