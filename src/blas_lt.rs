//! Shape-cached cuBLASLt plans. Vendor ABI follows the pinned CUDA 13 header.
use crate::cuda::{Buffer, Cuda, Stream};
use anyhow::{Context, Result, ensure};
use libloading::Library;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    ffi::{c_int, c_void},
    ptr,
    rc::Rc,
};
type Handle = *mut c_void;
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Heuristic {
    algo: [u64; 8],
    workspace: usize,
    state: i32,
    waves: f32,
    reserved: [i32; 4],
}
const _: () = assert!(size_of::<Heuristic>() == 96);
struct Api {
    _lib: Library,
    create: unsafe extern "C" fn(*mut Handle) -> c_int,
    destroy: unsafe extern "C" fn(Handle) -> c_int,
    desc_create: unsafe extern "C" fn(*mut Handle, c_int, c_int) -> c_int,
    desc_attr: unsafe extern "C" fn(Handle, c_int, *const c_void, usize) -> c_int,
    desc_destroy: unsafe extern "C" fn(Handle) -> c_int,
    layout_create: unsafe extern "C" fn(*mut Handle, c_int, u64, u64, i64) -> c_int,
    layout_destroy: unsafe extern "C" fn(Handle) -> c_int,
    preference_create: unsafe extern "C" fn(*mut Handle) -> c_int,
    preference_attr: unsafe extern "C" fn(Handle, c_int, *const c_void, usize) -> c_int,
    preference_destroy: unsafe extern "C" fn(Handle) -> c_int,
    heuristic: unsafe extern "C" fn(
        Handle,
        Handle,
        Handle,
        Handle,
        Handle,
        Handle,
        Handle,
        c_int,
        *mut Heuristic,
        *mut c_int,
    ) -> c_int,
    matmul: unsafe extern "C" fn(
        Handle,
        Handle,
        *const c_void,
        *const c_void,
        Handle,
        *const c_void,
        Handle,
        *const c_void,
        *const c_void,
        Handle,
        *mut c_void,
        Handle,
        *const u64,
        *mut c_void,
        usize,
        Handle,
    ) -> c_int,
}
impl Api {
    fn load() -> Result<Self> {
        let lib = unsafe { Library::new("libcublasLt.so.13") }.context("load CUDA 13 cuBLASLt")?;
        macro_rules! sym {
            ($n:literal) => {
                *unsafe { lib.get(concat!($n, "\0").as_bytes()) }?
            };
        }
        Ok(Self {
            create: sym!("cublasLtCreate"),
            destroy: sym!("cublasLtDestroy"),
            desc_create: sym!("cublasLtMatmulDescCreate"),
            desc_attr: sym!("cublasLtMatmulDescSetAttribute"),
            desc_destroy: sym!("cublasLtMatmulDescDestroy"),
            layout_create: sym!("cublasLtMatrixLayoutCreate"),
            layout_destroy: sym!("cublasLtMatrixLayoutDestroy"),
            preference_create: sym!("cublasLtMatmulPreferenceCreate"),
            preference_attr: sym!("cublasLtMatmulPreferenceSetAttribute"),
            preference_destroy: sym!("cublasLtMatmulPreferenceDestroy"),
            heuristic: sym!("cublasLtMatmulAlgoGetHeuristic"),
            matmul: sym!("cublasLtMatmul"),
            _lib: lib,
        })
    }
}
fn check(s: i32, op: &str) -> Result<()> {
    ensure!(s == 0, "{op}: cuBLASLt status {s}");
    Ok(())
}
struct Plan {
    api: Rc<Api>,
    desc: Handle,
    a: Handle,
    b: Handle,
    c: Handle,
    heuristic: Heuristic,
}
impl Drop for Plan {
    fn drop(&mut self) {
        unsafe {
            (self.api.desc_destroy)(self.desc);
            (self.api.layout_destroy)(self.a);
            (self.api.layout_destroy)(self.b);
            (self.api.layout_destroy)(self.c);
        }
    }
}
pub struct BlasLt {
    context: Cuda,
    api: Rc<Api>,
    handle: Handle,
    workspace: Buffer<u8>,
    plans: RefCell<BTreeMap<(usize, usize, usize, bool), Plan>>,
}
impl BlasLt {
    pub fn new(context: &Cuda) -> Result<Self> {
        let api = Rc::new(Api::load()?);
        let mut handle = ptr::null_mut();
        check(unsafe { (api.create)(&mut handle) }, "create cuBLASLt")?;
        let workspace = context.allocate::<u8>(8 * 1024 * 1024)?;
        Ok(Self {
            context: context.clone(),
            api,
            handle,
            workspace,
            plans: RefCell::new(BTreeMap::new()),
        })
    }
    pub fn prepare(&self, m: usize, n: usize, k: usize) -> Result<()> {
        self.prepare_epilogue(m, n, k, false)
    }
    /// BF16 bias epilogue in FP32 compute: D = bf16(acc + bias). Split-K, if any,
    /// may reduce only in the FP32 compute type, never in the BF16 output type.
    pub fn prepare_bias(&self, m: usize, n: usize, k: usize) -> Result<()> {
        self.prepare_epilogue(m, n, k, true)
    }
    fn prepare_epilogue(&self, m: usize, n: usize, k: usize, bias: bool) -> Result<()> {
        ensure!(
            m > 0
                && n > 0
                && k > 0
                && m <= i32::MAX as usize
                && n <= i32::MAX as usize
                && k <= i32::MAX as usize,
            "Lt GEMM dimension range"
        );
        if self.plans.borrow().contains_key(&(m, n, k, bias)) {
            return Ok(());
        }
        let (mut desc, mut a, mut b, mut c, mut pref) = (
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        );
        // Construct a provisional plan immediately so error paths release descriptors.
        check(
            unsafe { (self.api.desc_create)(&mut desc, 68, 0) },
            "create Lt operation descriptor",
        )?;
        let mut plan = Plan {
            api: self.api.clone(),
            desc,
            a,
            b,
            c,
            heuristic: Heuristic::default(),
        };
        let transa = 1i32;
        check(
            unsafe { (self.api.desc_attr)(desc, 3, (&transa as *const i32).cast(), 4) },
            "Lt transpose weight",
        )?;
        if bias {
            // cublasLt.h (CUDA 13): CUBLASLT_MATMUL_DESC_EPILOGUE=7, CUBLASLT_EPILOGUE_BIAS=4.
            // The bias type defaults to the BF16 output type; the pointer is set per call.
            let epilogue = 4i32;
            check(
                unsafe { (self.api.desc_attr)(desc, 7, (&epilogue as *const i32).cast(), 4) },
                "Lt bias epilogue",
            )?;
        }
        check(
            unsafe { (self.api.layout_create)(&mut a, 14, k as u64, n as u64, k as i64) },
            "Lt weight layout",
        )?;
        plan.a = a;
        check(
            unsafe { (self.api.layout_create)(&mut b, 14, k as u64, m as u64, k as i64) },
            "Lt input layout",
        )?;
        plan.b = b;
        check(
            unsafe { (self.api.layout_create)(&mut c, 14, n as u64, m as u64, n as i64) },
            "Lt output layout",
        )?;
        plan.c = c;
        check(
            unsafe { (self.api.preference_create)(&mut pref) },
            "Lt preference",
        )?;
        let workspace = self.workspace.bytes();
        let configured = check(
            unsafe {
                (self.api.preference_attr)(
                    pref,
                    1,
                    (&workspace as *const usize).cast(),
                    size_of::<usize>(),
                )
            },
            "Lt workspace preference",
        );
        // CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK=3; COMPUTE_TYPE=2 only (no output-type reduction).
        let scheme = 2u32;
        let configured = configured.and_then(|_| if bias { check(
            unsafe { (self.api.preference_attr)(pref, 3, (&scheme as *const u32).cast(), 4) },
            "Lt reduction scheme preference") } else { Ok(()) });
        if let Err(e) = configured {
            unsafe {
                (self.api.preference_destroy)(pref);
            }
            return Err(e);
        }
        let mut candidates = [Heuristic::default(); 16];
        let mut count = 0;
        let selected = check(
            unsafe {
                (self.api.heuristic)(
                    self.handle,
                    desc,
                    a,
                    b,
                    c,
                    c,
                    pref,
                    16,
                    candidates.as_mut_ptr(),
                    &mut count,
                )
            },
            "Lt heuristic selection",
        );
        unsafe {
            (self.api.preference_destroy)(pref);
        }
        selected?;
        plan.heuristic = *candidates[..count as usize]
            .iter()
            .find(|a| a.state == 0 && a.workspace <= workspace)
            .context("no admitted Lt GEMM algorithm")?;
        self.plans.borrow_mut().insert((m, n, k, bias), plan);
        Ok(())
    }
    /// Safety: buffers and stream share the owning CUDA context. Call prepare()
    /// before graph capture; pointers/workspace remain valid through completion.
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
        ensure!(
            x.len() == m.checked_mul(k).context("X size overflow")?
                && weight.len() == n.checked_mul(k).context("W size overflow")?
                && y.len() == m.checked_mul(n).context("Y size overflow")?,
            "Lt GEMM buffer shapes"
        );
        unsafe { self.linear_bf16_raw(x.ptr(), weight.ptr(), y.ptr(), m, n, k, stream) }
    }
    /// Safety: live BF16 X[M,K], W[N,K], Y[M,N] ranges share this context;
    /// caller retains all allocations until stream and graph completion.
    pub unsafe fn linear_bf16_raw(
        &self,
        x: u64,
        weight: u64,
        y: u64,
        m: usize,
        n: usize,
        k: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            self.context.owns_stream(stream)
                && [x, weight, y].iter().all(|p| *p != 0 && *p % 16 == 0),
            "Lt linear context/alignment"
        );
        let plans = self.plans.borrow();
        let p = plans
            .get(&(m, n, k, false))
            .context("Lt shape was not prepared before execution")?;
        let alpha = 1f32;
        let beta = 0f32;
        check(
            unsafe {
                (self.api.matmul)(
                    self.handle,
                    p.desc,
                    (&alpha as *const f32).cast(),
                    weight as *const c_void,
                    p.a,
                    x as *const c_void,
                    p.b,
                    (&beta as *const f32).cast(),
                    y as *const c_void,
                    p.c,
                    y as *mut c_void,
                    p.c,
                    p.heuristic.algo.as_ptr(),
                    self.workspace.ptr() as *mut c_void,
                    self.workspace.bytes(),
                    stream.raw(),
                )
            },
            "Lt BF16 linear",
        )
    }
}
impl BlasLt {
    /// Safety: as linear_bf16_raw; bias is a live BF16 [N] vector. Eager use only:
    /// the per-call bias pointer is a descriptor attribute, so no graph capture.
    pub(crate) unsafe fn linear_bias_bf16_raw(&self, x: u64, weight: u64, bias: u64, y: u64,
        m: usize, n: usize, k: usize, stream: &Stream) -> Result<()> {
        ensure!(self.context.owns_stream(stream)
            && [x, weight, bias, y].iter().all(|p| *p != 0 && *p % 16 == 0), "Lt bias linear context/alignment");
        let plans = self.plans.borrow();
        let p = plans.get(&(m, n, k, true)).context("Lt bias shape was not prepared before execution")?;
        check(unsafe { (self.api.desc_attr)(p.desc, 8, (&bias as *const u64).cast(), 8) }, "Lt bias pointer")?;
        let alpha = 1f32;
        let beta = 0f32;
        check(
            unsafe {
                (self.api.matmul)(self.handle, p.desc, (&alpha as *const f32).cast(), weight as *const c_void, p.a,
                    x as *const c_void, p.b, (&beta as *const f32).cast(), y as *const c_void, p.c, y as *mut c_void, p.c,
                    p.heuristic.algo.as_ptr(), self.workspace.ptr() as *mut c_void, self.workspace.bytes(), stream.raw())
            },
            "Lt BF16 bias linear",
        )
    }
}
impl Drop for BlasLt {
    fn drop(&mut self) {
        self.plans.get_mut().clear();
        unsafe {
            (self.api.destroy)(self.handle);
        }
    }
}
