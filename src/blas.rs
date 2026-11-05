//! Direct vendor GEMM primitive, independently owned by the Rust executor.
use crate::cuda::{Buffer, Cuda, Stream};
use anyhow::{Context, Result, ensure};
use libloading::Library;
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    ffi::{c_int, c_void},
    rc::{Rc, Weak},
};

struct Workspace {
    buffer: Buffer<u8>,
    stream_identity: Option<Rc<()>>,
}
impl Workspace {
    fn ptr(&self) -> u64 { self.buffer.ptr() }
    fn bytes(&self) -> usize { self.buffer.bytes() }
}
thread_local! {
    static WORKSPACES: RefCell<BTreeMap<(usize,usize),Weak<Workspace>>> = RefCell::new(BTreeMap::new());
}

pub struct Blas {
    _lib: Library,
    handle: *mut c_void,
    _context: Cuda,
    destroy: unsafe extern "C" fn(*mut c_void) -> c_int,
    set_stream: unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int,
    set_workspace: unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> c_int,
    workspace: Rc<Workspace>,
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

/// Finite GPU component screen: identical private-handle calls, changing
/// operands, graph replay and simultaneous streams. No CPU arithmetic oracle.
pub fn probe_shared_workspace()->Result<serde_json::Value> {
    use sha2::{Digest,Sha256};
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;
    let stream=cuda.stream()?;let other_stream=cuda.stream()?;
    struct Case { m:usize,n:usize,k:usize,ty:i32,x:Buffer<half::bf16>,w:Buffer<half::bf16>,
        a:Buffer<f32>,b:Buffer<f32>,ordinary:Blas,pooled:Blas }
    let mut cases=Vec::new();
    for (i,(m,n,k,ty))in [(460,512,5120,14),(6,5376,5120,0),(128,2048,4096,14),
        (32,512,5120,14)].into_iter().enumerate() {
        let ordinary=Blas::with_stream_policy(&cuda,&stream,false)?;
        let pooled=Blas::with_stream_policy(&cuda,&stream,true)?;
        if i%2==1 {ordinary.require_fp32_reduction()?;pooled.require_fp32_reduction()?;}
        let mut w=cuda.allocate::<half::bf16>(n*k)?;
        w.upload(&(0..n*k).map(|j|half::bf16::from_f32(
            (((j.wrapping_mul(193)+i*71)%1009)as f32-504.)/631.)) .collect::<Vec<_>>())?;
        cases.push(Case{m,n,k,ty,x:cuda.allocate(m*k)?,w,a:cuda.allocate(m*n)?,b:cuda.allocate(m*n)?,ordinary,pooled});
    }
    let distinct=|shared:bool|cases.iter().map(|c|if shared{c.pooled.workspace.ptr()}else{c.ordinary.workspace.ptr()})
        .collect::<std::collections::BTreeSet<_>>().len();
    ensure!(distinct(false)==4&&distinct(true)==1,"scratch sharing identity");
    let other=Blas::with_stream_policy(&cuda,&other_stream,true)?;
    ensure!(other.workspace.ptr()!=cases[0].pooled.workspace.ptr(),"concurrent streams aliased scratch");
    ensure!(cases[0].pooled.guard_stream(&other_stream).is_err(),"foreign stream guard accepted");
    let other_output=cuda.allocate::<f32>(cases[0].m*cases[0].n)?;
    let launch=|c:&Case,shared:bool|->Result<()> {unsafe{
        let (blas,y)=if shared{(&c.pooled,c.b.ptr())}else{(&c.ordinary,c.a.ptr())};
        blas.linear_raw(c.x.ptr(),c.w.ptr(),y,c.m,c.n,c.k,c.ty,&stream)
    }};
    let mut graphs=None;
    let mut records=Vec::new();
    for cycle in 0..4usize {
        for (i,c)in cases.iter_mut().enumerate() {
            c.x.upload(&(0..c.m*c.k).map(|j|half::bf16::from_f32(
                (((j.wrapping_mul(37)+cycle*137+i*23)%503)as f32-251.)/317.)) .collect::<Vec<_>>())?;
        }
        stream.join_default()?;other_stream.join_default()?;
        if cycle==0 {
            for c in &cases {launch(c,false)?;launch(c,true)?;}
            stream.synchronize()?;
            graphs=Some((stream.capture(||{for c in &cases{launch(c,false)?;}Ok(())})?,
                stream.capture(||{for c in &cases{launch(c,true)?;}Ok(())})?));
        }
        let (a,b)=graphs.as_ref().unwrap();
        ensure!(unsafe{b.launch(&other_stream)}.is_err(),"foreign graph stream guard accepted");
        unsafe {
            if cycle%2==0 {a.launch(&stream)?;b.launch(&stream)?;}
            else {b.launch(&stream)?;a.launch(&stream)?;}
            let c=&cases[0];
            other.linear_raw(c.x.ptr(),c.w.ptr(),other_output.ptr(),c.m,c.n,c.k,c.ty,&other_stream)?;
        }
        stream.synchronize()?;other_stream.synchronize()?;
        let mut hashes=Vec::new();
        for c in &cases {
            let bytes=c.m*c.n*if c.ty==0{4}else{2};
            let a=unsafe{cuda.read_device(c.a.ptr(),bytes)}?;
            let b=unsafe{cuda.read_device(c.b.ptr(),bytes)}?;
            ensure!(a==b,"cuBLAS shared scratch altered eager/captured result");
            hashes.push(format!("{:x}",Sha256::digest(&a)));
        }
        let c=&cases[0];let bytes=c.m*c.n*2;
        ensure!(unsafe{cuda.read_device(c.a.ptr(),bytes)}?==unsafe{cuda.read_device(other_output.ptr(),bytes)}?,
            "simultaneous stream result differs");
        records.push(serde_json::json!({"cycle":cycle,"hashes":hashes,"exact":true}));
    }
    drop(graphs);
    let weak=Rc::downgrade(&cases[0].pooled.workspace);
    drop(cases);
    ensure!(weak.upgrade().is_none(),"scratch cache retained final allocation");
    Ok(serde_json::json!({"all_exact":true,"private_workspace_bytes":4*32*1024*1024,
        "pooled_workspace_bytes":32*1024*1024,"distinct_handles":4,"private_math_modes":true,
        "foreign_stream_rejected":true,"foreign_graph_stream_rejected":true,
        "concurrent_streams_isolated":true,"weak_release":true,
        "records":records,"scope":"GPU GEMM scratch ownership component only; not full-model quality or performance qualification."}))
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
        ensure!(self.workspace.stream_identity.as_ref().is_none_or(|id|
            Rc::as_ptr(id) as usize == stream.identity_key()),
            "cuBLAS pooled workspace belongs to a different ordered stream");
        if self.workspace.stream_identity.is_some() {stream.bind_captured_scratch();}
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
        self.grouped_woa_bounded(x,weight,output,rows,rows,stream)
    }
    /// Keep allocation and group strides at their physical extent while
    /// computing only a prefix. Inactive output rows must not be consumed.
    pub(crate) fn grouped_woa_bounded(
        &self,x:&Buffer<half::bf16>,weight:&Buffer<half::bf16>,
        output:&mut Buffer<half::bf16>,rows:usize,compute:usize,stream:&Stream,
    )->Result<()> {
        self.guard_stream(stream)?;
        ensure!(
            ((9..=384).contains(&rows) || crate::config::wide_rows(rows))
                && (compute==rows || (rows==128&&(17..128).contains(&compute)))
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
                    compute as i32,
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
        Self::with_workspace(context, Rc::new(Workspace {
            buffer: context.allocate::<u8>(32 * 1024 * 1024)?, stream_identity: None,
        }))
    }
    /// Same-stream invocations are enqueued by this single GPU-owner thread.
    /// Keep handles/math modes private, while sharing only the 32 MiB scratch.
    /// cuBLAS documents that a call can comprise several kernels; no caller
    /// may interleave those kernels from another host thread or CUDA stream.
    /// https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace
    pub(crate) fn new_on(context:&Cuda,stream:&Stream)->Result<Self> {
        let shared=std::env::var("CANADA_SCRATCH_SHARED_BLAS_WORKSPACE_SCREEN").as_deref()==Ok("1");
        Self::with_stream_policy(context,stream,shared)
    }
    fn with_stream_policy(context:&Cuda,stream:&Stream,shared:bool)->Result<Self> {
        ensure!(context.owns_stream(stream),"cuBLAS scratch stream context");
        if !shared{return Self::new(context);}
        let key=(context.context_key(),stream.identity_key());
        let workspace=WORKSPACES.with(|cache|->Result<_> {
            let mut cache=cache.borrow_mut();
            cache.retain(|_,value|value.strong_count()>0);
            if let Some(hit)=cache.get(&key).and_then(Weak::upgrade) {
                ensure!(context.owns_buffer(&hit.buffer),"cuBLAS scratch cache context");
                return Ok(hit);
            }
            let workspace=Rc::new(Workspace{buffer:context.allocate::<u8>(32*1024*1024)?,
                stream_identity:Some(stream.identity_token())});
            cache.insert(key,Rc::downgrade(&workspace));Ok(workspace)
        })?;
        Self::with_workspace(context,workspace)
    }
    fn with_workspace(context:&Cuda,workspace:Rc<Workspace>)->Result<Self> {
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
