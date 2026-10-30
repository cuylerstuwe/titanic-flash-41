//! Direct vendor GEMM primitive, independently owned by the Rust executor.
use crate::cuda::{Buffer, Cuda, Stream};
use anyhow::{Context, Result, ensure};
use libloading::Library;
use std::{
    cell::Cell,
    ffi::{c_int, c_void},
};

pub struct Blas {
    _lib: Library,
    handle: *mut c_void,
    _context: Cuda,
    destroy: unsafe extern "C" fn(*mut c_void) -> c_int,
    set_stream: unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int,
    set_workspace: unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> c_int,
    workspace: Buffer<u8>,
    stream: Cell<usize>,
    strided_batched: unsafe extern "C" fn(
        *mut c_void,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        *const c_void,
        *const c_void,
        c_int,
        c_int,
        i64,
        *const c_void,
        c_int,
        c_int,
        i64,
        *const c_void,
        *mut c_void,
        c_int,
        c_int,
        i64,
        c_int,
        c_int,
        c_int,
    ) -> c_int,
    gemm: unsafe extern "C" fn(
        *mut c_void,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        *const c_void,
        *const c_void,
        c_int,
        c_int,
        *const c_void,
        c_int,
        c_int,
        *const c_void,
        *mut c_void,
        c_int,
        c_int,
        c_int,
        c_int,
    ) -> c_int,
}
fn check(code: i32, operation: &str) -> Result<()> {
    ensure!(code == 0, "{operation}: cuBLAS status {code}");
    Ok(())
}
impl Blas {
    /// Require FP32 partial reductions even when the output is BF16. The
    /// compute type alone does not prohibit reduced-precision split-K storage.
    /// Apply before any launches/graph capture on this private handle.
    pub(crate) fn require_fp32_reduction(&self) -> Result<()> {
        ensure!(self.stream.get() == 0, "configure GEMM precision before launch");
        let set: unsafe extern "C" fn(*mut c_void, c_int) -> c_int =
            *unsafe { self._lib.get(b"cublasSetMathMode\0") }?;
        // CUDA13 cublas_api.h: CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION.
        check(unsafe { set(self.handle, 16) }, "require FP32 GEMM reductions")
    }
    /// FP32 confidence dot products. Inputs originate as exact BF16 values
    /// widened on GPU; no reduced output precision or CPU model arithmetic.
    pub(crate) fn confidence_linear(
        &self,
        x: &Buffer<f32>,
        weight: &Buffer<f32>,
        output: &Buffer<f32>,
        step: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.guard_stream(stream)?;
        ensure!(
            x.len() == 5 * 5376
                && step < 5
                && weight.len() == 5376
                && output.len() == 1
                && self._context.owns_buffer(x)
                && self._context.owns_buffer(weight)
                && self._context.owns_buffer(output),
            "FP32 confidence source/shape/context"
        );
        check(
            unsafe { (self.set_stream)(self.handle, stream.raw()) },
            "confidence stream",
        )?;
        check(
            unsafe {
                (self.set_workspace)(
                    self.handle,
                    self.workspace.ptr() as _,
                    self.workspace.bytes(),
                )
            },
            "confidence workspace",
        )?;
        let alpha = 1f32;
        let beta = 0f32;
        check(
            unsafe {
                (self.gemm)(
                    self.handle,
                    1,
                    0,
                    1,
                    1,
                    5376,
                    (&alpha as *const f32).cast(),
                    weight.ptr() as _,
                    0,
                    5376,
                    (x.ptr() + (step * 5376 * 4) as u64) as _,
                    0,
                    5376,
                    (&beta as *const f32).cast(),
                    output.ptr() as _,
                    0,
                    1,
                    68,
                    99,
                )
            },
            "FP32 confidence projection",
        )
    }
    fn guard_stream(&self, stream: &Stream) -> Result<()> {
        ensure!(
            self._context.owns_stream(stream),
            "cuBLAS stream/context mismatch"
        );
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == raw,
            "shared cuBLAS workspace requires one ordered stream"
        );
        self.stream.set(raw);
        Ok(())
    }
    /// Safety: row-major BF16 X[M,K]/W[N,K] and writable output Y[M,N]
    /// belong to this context and live through stream completion. Output type
    /// is explicitly CUDA_R_32F (0) or CUDA_R_16BF (14), always FP32 accumulation.
    pub unsafe fn linear_raw(
        &self,
        x: u64,
        weight: u64,
        y: u64,
        m: usize,
        n: usize,
        k: usize,
        output_type: i32,
        stream: &Stream,
    ) -> Result<()> {
        self.guard_stream(stream)?;
        ensure!(
            [0, 14].contains(&output_type)
                && self._context.owns_stream(stream)
                && [x, weight, y].iter().all(|p| *p != 0 && *p % 16 == 0)
                && [m, n, k].iter().all(|d| *d > 0 && *d <= i32::MAX as usize),
            "vendor linear geometry/context/type"
        );
        check(
            unsafe { (self.set_stream)(self.handle, stream.raw()) },
            "set cuBLAS stream",
        )?;
        check(
            unsafe {
                (self.set_workspace)(
                    self.handle,
                    self.workspace.ptr() as *mut c_void,
                    self.workspace.bytes(),
                )
            },
            "set qualified cuBLAS workspace",
        )?;
        let alpha = 1f32;
        let beta = 0f32;
        check(
            unsafe {
                (self.gemm)(
                    self.handle,
                    1,
                    0,
                    n as i32,
                    m as i32,
                    k as i32,
                    (&alpha as *const f32).cast(),
                    weight as *const c_void,
                    14,
                    k as i32,
                    x as *const c_void,
                    14,
                    k as i32,
                    (&beta as *const f32).cast(),
                    y as *mut c_void,
                    output_type,
                    n as i32,
                    68,
                    99,
                )
            },
            "BF16 vendor projection",
        )
    }
    /// Qualified medium-prefill WO-A einsum lowering: two BF16 strided GEMMs,
    /// FP32 accumulation and group-major BF16 output before the layout copy.
    pub fn grouped_woa(
        &self,
        x: &Buffer<half::bf16>,
        weight: &Buffer<half::bf16>,
        output: &mut Buffer<half::bf16>,
        rows: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.guard_stream(stream)?;
        ensure!(
            ((9..=384).contains(&rows) || crate::config::wide_rows(rows))
                && x.len() == rows * 8192
                && weight.len() == 2 * 1024 * 4096
                && output.len() == rows * 2048
                && self._context.owns_buffer(x)
                && self._context.owns_buffer(weight)
                && self._context.owns_buffer(output),
            "medium WO-A GEMM buffers/context"
        );
        check(
            unsafe { (self.set_stream)(self.handle, stream.raw()) },
            "set grouped GEMM stream",
        )?;
        check(
            unsafe {
                (self.set_workspace)(
                    self.handle,
                    self.workspace.ptr() as _,
                    self.workspace.bytes(),
                )
            },
            "set grouped GEMM workspace",
        )?;
        let alpha = 1f32;
        let beta = 0f32;
        check(
            unsafe {
                (self.strided_batched)(
                    self.handle,
                    1,
                    0,
                    1024,
                    rows as i32,
                    4096,
                    (&alpha as *const f32).cast(),
                    weight.ptr() as _,
                    14,
                    4096,
                    1024 * 4096,
                    x.ptr() as _,
                    14,
                    8192,
                    4096,
                    (&beta as *const f32).cast(),
                    output.ptr() as _,
                    14,
                    1024,
                    (rows * 1024) as i64,
                    2,
                    68,
                    99,
                )
            },
            "medium WO-A strided BF16 GEMM",
        )
    }
    pub fn new(context: &Cuda) -> Result<Self> {
        let lib = unsafe { Library::new("libcublas.so.13") }.context("load CUDA 13 cuBLAS")?;
        let create: unsafe extern "C" fn(*mut *mut c_void) -> c_int =
            *unsafe { lib.get(b"cublasCreate_v2\0") }?;
        let destroy = *unsafe { lib.get(b"cublasDestroy_v2\0") }?;
        let set_stream = *unsafe { lib.get(b"cublasSetStream_v2\0") }?;
        let gemm = *unsafe { lib.get(b"cublasGemmEx\0") }?;
        let set_workspace = *unsafe { lib.get(b"cublasSetWorkspace_v2\0") }?;
        let strided_batched = *unsafe { lib.get(b"cublasGemmStridedBatchedEx\0") }?;
        // The pinned Torch/SGLang GPU reference explicitly binds 32 MiB and
        // GEMM_DEFAULT_TENSOR_OP. Defaults selected a different BF16 algorithm
        // at M460/N512/K5120. One retained buffer also avoids lazy workspace
        // allocation across captured shapes.
        let workspace = context.allocate::<u8>(32 * 1024 * 1024)?;
        let mut handle = std::ptr::null_mut();
        check(unsafe { create(&mut handle) }, "create cuBLAS handle")?;
        Ok(Self {
            _lib: lib,
            handle,
            _context: context.clone(),
            destroy,
            set_stream,
            set_workspace,
            workspace,
            strided_batched,
            stream: Cell::new(0),
            gemm,
        })
    }
    /// Row-major Y[M,N] = X[M,K] W[N,K]^T, FP32 accumulation and BF16 output.
    /// Safety: all buffers belong to this CUDA context and remain valid and
    /// exclusively accessible to queued work through stream completion.
    pub unsafe fn linear_bf16(
        &self,
        x: &Buffer<u16>,
        weight: &Buffer<u16>,
        y: &mut Buffer<u16>,
        m: usize,
        n: usize,
        k: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.guard_stream(stream)?;
        ensure!(
            m > 0
                && n > 0
                && k > 0
                && m <= i32::MAX as usize
                && n <= i32::MAX as usize
                && k <= i32::MAX as usize,
            "GEMM dimension range"
        );
        ensure!(
            x.len() == m.checked_mul(k).context("X size overflow")?
                && weight.len() == n.checked_mul(k).context("W size overflow")?
                && y.len() == m.checked_mul(n).context("Y size overflow")?,
            "GEMM buffer shape mismatch"
        );
        check(
            unsafe { (self.set_stream)(self.handle, stream.raw()) },
            "set cuBLAS stream",
        )?;
        check(
            unsafe {
                (self.set_workspace)(
                    self.handle,
                    self.workspace.ptr() as *mut c_void,
                    self.workspace.bytes(),
                )
            },
            "set qualified cuBLAS workspace",
        )?;
        let alpha = 1f32;
        let beta = 0f32;
        check(
            unsafe {
                (self.gemm)(
                    self.handle,
                    1,
                    0,
                    n as i32,
                    m as i32,
                    k as i32,
                    (&alpha as *const f32).cast(),
                    weight.ptr() as *const c_void,
                    14,
                    k as i32,
                    x.ptr() as *const c_void,
                    14,
                    k as i32,
                    (&beta as *const f32).cast(),
                    y.ptr() as *mut c_void,
                    14,
                    n as i32,
                    68,
                    99,
                )
            },
            "BF16 linear GEMM",
        )
    }
}
impl Drop for Blas {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.handle);
        }
    }
}
