//! Minimal CUDA driver ownership. No PyTorch, Python, or model-engine dependency.
//! All handles stay on their owning OS thread. Buffers/modules keep the context alive.
use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use std::{
    cell::RefCell,
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    marker::PhantomData,
    os::fd::{AsRawFd, OwnedFd},
    ptr,
    rc::{Rc, Weak},
};

type Handle = *mut c_void;
type CuResult = c_int;

#[derive(Debug)]
struct DriverError { code: CuResult, operation: String, message: String }
impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} ({})", self.operation, self.message, self.code)
    }
}
impl std::error::Error for DriverError {}

#[derive(Debug)]
struct AllocationFloorError { bytes: usize, available: usize, required: usize }
impl std::fmt::Display for AllocationFloorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "construction allocation of {} bytes would consume physical reserve; available={}",
            self.bytes, self.available)
    }
}
impl std::error::Error for AllocationFloorError {}
/// True for the allocator's own floor refusal (`Cuda::allocate` with the allocation-floor guard on): the one
/// allocation error that leaves the device, the context and every existing buffer as they were.
pub(crate) fn allocation_floor_refused(error: &anyhow::Error) -> bool {
    error.downcast_ref::<AllocationFloorError>().is_some()
}

/// Only allocation capacity is retryable. In particular, an asynchronous
/// execution error reported by cuMemAlloc must still retire the worker group.
fn input_allocation_error(error: anyhow::Error, required: usize) -> anyhow::Error {
    if let Some(floor) = error.downcast_ref::<AllocationFloorError>() {
        return crate::memory::InputCapacityError("image allocation cannot retain the UMA reserve", floor.required).into();
    }
    if error.downcast_ref::<DriverError>().is_some_and(|e|
        e.code == 2 && e.operation == "allocate device memory") {
        return crate::memory::InputCapacityError("image device allocation is temporarily unavailable", required).into();
    }
    error
}

#[cfg(test)]
mod input_allocation_tests {
    use super::*;
    #[test]
    fn image_allocation_refusal_does_not_mask_execution_or_identity_errors() {
        let floor=input_allocation_error(AllocationFloorError {bytes:3,available:4,required:12}.into(),9);
        assert_eq!(floor.downcast_ref::<crate::memory::InputCapacityError>().unwrap().1,12);
        for (code,op,retryable) in [(2,"allocate device memory",true),(700,"allocate device memory",false),
            (201,"allocate device memory",false),(2,"launch kernel",false)] {
            let e=input_allocation_error(DriverError{code,operation:op.into(),message:"diagnostic".into()}.into(),9);
            assert_eq!(e.downcast_ref::<crate::memory::InputCapacityError>().is_some(),retryable);
            if !retryable {assert_eq!(e.downcast_ref::<DriverError>().unwrap().code,code);}
        }
        let e=input_allocation_error(anyhow::anyhow!("out of memory (2)"),9);
        assert!(e.downcast_ref::<crate::memory::InputCapacityError>().is_none());
    }
}

/// Nov 17 floor work: set once the worker has finished its startup warm-up. With the allocation-site screen on
/// (CANADA_SCRATCH_ALLOCATION_SITES_SCREEN=1), every device allocation or free of 16 MiB or more after that point is
/// logged as a `runtime_allocation` line (site, bytes, host available memory), the inventory of what a serving engine
/// allocates while it runs. Diagnostic only; nothing changes without the screen.
pub(crate) static SERVING_STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
fn trace_runtime_allocation(kind: &str, site: &std::panic::Location<'_>, bytes: usize) {
    if bytes < (16 << 20) || !SERVING_STARTED.load(std::sync::atomic::Ordering::Relaxed) { return; }
    let available = std::fs::read_to_string("/proc/meminfo").ok().and_then(|m| m.lines()
        .find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()));
    let free = std::fs::read_to_string("/proc/meminfo").ok().and_then(|m| m.lines()
        .find(|l| l.starts_with("MemFree:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()));
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    eprintln!("runtime_allocation {}", serde_json::json!({"t": t, "kind": kind, "site": format!("{}:{}", site.file(), site.line()),
        "bytes": bytes, "available_kib": available, "free_kib": free}));
}

// CUDA 13.0 public VMM ABI. Used only for opt-in, individually owned,
// losslessly compressible weight storage. The finite external-pool experiment
// can import allocations from its direct parent in the same memory cgroup.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MemLocation { kind: c_int, id: c_int }
#[repr(C)]
#[derive(Default)]
struct MemAllocationProp {
    kind: c_int,
    handle_types: c_int,
    location: MemLocation,
    windows_metadata: *mut c_void,
    compression: u8,
    rdma: u8,
    usage: u16,
    reserved: [u8; 4],
}
#[repr(C)]
struct MemAccessDesc { location: MemLocation, flags: c_int }
const _: () = {
    assert!(std::mem::size_of::<MemLocation>() == 8);
    assert!(std::mem::size_of::<MemAllocationProp>() == 32);
    assert!(std::mem::offset_of!(MemAllocationProp, location) == 8);
    assert!(std::mem::offset_of!(MemAllocationProp, windows_metadata) == 16);
    assert!(std::mem::offset_of!(MemAllocationProp, compression) == 24);
    assert!(std::mem::size_of::<MemAccessDesc>() == 12);
    assert!(std::mem::offset_of!(MemAccessDesc, flags) == 8);
};

// CUDA's public CUlaunchConfig / CUlaunchAttribute ABI (cuda.h 13.0).
// Only programmatic stream serialization is exposed here. A PDL consumer
// must wait for its predecessor before reading any dependent global memory.
#[repr(C)]
struct LaunchAttribute {
    id: c_int,
    pad: [u8; 4],
    value: [u64; 8],
}
#[repr(C)]
struct LaunchConfig {
    grid: [c_uint; 3],
    block: [c_uint; 3],
    shared: c_uint,
    stream: Handle,
    attrs: *mut LaunchAttribute,
    count: c_uint,
}
const _: () = {
    assert!(std::mem::size_of::<LaunchAttribute>() == 72);
    assert!(std::mem::offset_of!(LaunchAttribute, value) == 8);
    assert!(std::mem::size_of::<LaunchConfig>() == 56);
    assert!(std::mem::offset_of!(LaunchConfig, stream) == 32);
    assert!(std::mem::offset_of!(LaunchConfig, attrs) == 40);
    assert!(std::mem::offset_of!(LaunchConfig, count) == 48);
};

// CUDA's CUDA_KERNEL_NODE_PARAMS_v2 (cuda.h 13.0), for the graph-node grid rule below.
#[repr(C)]
struct KernelNodeParams {
    func: Handle,
    grid: [c_uint; 3],
    block: [c_uint; 3],
    shared: c_uint,
    params: *mut *mut c_void,
    extra: *mut *mut c_void,
    kern: Handle,
    ctx: Handle,
}
const _: () = {
    assert!(std::mem::size_of::<KernelNodeParams>() == 72);
    assert!(std::mem::offset_of!(KernelNodeParams, params) == 40);
};
/// Kernel nodes of the graphs captured inside a `graph_regrid_scope` whose function name contains `name` and whose
/// grid is (1, 1, from) launch with (1, 1, to) blocks instead. For kernels that size their own cooperation from
/// the launch grid (the routed experts' front-end reads its grid for its barriers) `to` is never above `from`,
/// so every block stays resident. `grow` lifts that bound: only for kernels whose blocks stride one list of
/// independent tasks by the launch grid and keep no state outside their own shared memory (the experts' two phase
/// kernels: `task = block; while task < tasks { ..; task += grid }`), where any grid computes every task once.
/// `carveout`: Some(percent) also sets the function's preferred shared-memory carve-out (100 lets as many blocks
/// stay resident on an SM as its registers allow; -1 restores the driver's choice). The attribute belongs to the
/// function, so a later capture must restore it where it should not persist.
#[derive(Clone, Copy)]
pub(crate) struct Regrid { pub name: &'static str, pub from: u32, pub to: u32, pub grow: bool, pub carveout: Option<i32> }
thread_local! {
    static REGRID: std::cell::Cell<[Option<Regrid>; 3]> = const { std::cell::Cell::new([None; 3]) };
    static REGRIDDED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
pub(crate) struct RegridScope(());
impl Drop for RegridScope {
    fn drop(&mut self) { REGRID.with(|r| r.set([None; 3])); }
}
/// Apply `rules` (one to three) to every graph this thread captures until the scope drops; `graph_regridded` then
/// gives the number of kernel nodes they matched.
pub(crate) fn graph_regrid_scope(rules: &[Regrid]) -> Result<RegridScope> {
    ensure!((1..=3).contains(&rules.len()) && rules.iter().all(|rule| rule.to >= 1
            && (rule.to <= rule.from || (rule.grow && rule.to <= 1024))
            && rule.carveout.is_none_or(|c| c == -1 || (0..=100).contains(&c)))
        && REGRID.with(|r| r.get().iter().all(Option::is_none)), "graph node grid rule");
    let mut slots = [None; 3];
    for (slot, rule) in slots.iter_mut().zip(rules) { *slot = Some(*rule); }
    REGRIDDED.with(|n| n.set(0));
    REGRID.with(|r| r.set(slots));
    Ok(RegridScope(()))
}
pub(crate) fn graph_regridded() -> usize { REGRIDDED.with(|n| n.replace(0)) }
/// A kernel node of the graph being captured that is replaced, before the graph is instantiated, by another kernel
/// with its own arguments (`graph_swap`, called by the code that enqueues the node's launch). The node is found by
/// its function name and by its own arguments: it must hold every pointer of `markers`, so a record can never
/// land on another launch's node; records that match the same node must be identical. The replacement's module
/// outlives every graph. Arguments: `pointers` first, then the 32-bit `scalars`.
pub(crate) struct NodeSwap {
    pub name: &'static str,
    pub markers: [u64; 3],
    pub func: usize,
    pub grid: u32,
    pub block: u32,
    pub pointers: Vec<u64>,
    pub scalars: Vec<i32>,
}
thread_local! {
    static CAPTURE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static SWAPS: std::cell::RefCell<Vec<NodeSwap>> = const { std::cell::RefCell::new(Vec::new()) };
    static SWAPPED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static CAPTURE_SERIAL: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}
/// Advances at every `Stream::capture` begin on this thread: work enqueued under one serial belongs to one graph
/// (or, outside captures, to one eager run between captures).
pub(crate) fn capture_serial() -> u64 { CAPTURE_SERIAL.with(|s| s.get()) }
/// Whether this thread is inside exactly one `Stream::capture` body (the only place `graph_swap` is legal).
pub(crate) fn capturing() -> bool { CAPTURE_DEPTH.with(|d| d.get() == 1) }
pub(crate) fn graph_swap(swap: NodeSwap) -> Result<()> {
    ensure!(capturing() && swap.func != 0 && swap.grid >= 1 && (32..=1024).contains(&swap.block) && swap.block % 32 == 0
        && swap.markers.iter().all(|m| *m != 0), "graph node replacement outside a capture");
    SWAPS.with(|s| s.borrow_mut().push(swap));
    Ok(())
}
/// Kernel nodes replaced since the last call (screens).
pub(crate) fn graph_swapped() -> usize { SWAPPED.with(|n| n.replace(0)) }
/// Graph-node editing entry points, loaded on first use: ordinary startup does not depend on them.
struct GraphEdit {
    _library: Library,
    node_type: unsafe extern "C" fn(Handle, *mut c_int) -> CuResult,
    get: unsafe extern "C" fn(Handle, *mut KernelNodeParams) -> CuResult,
    set: unsafe extern "C" fn(Handle, *const KernelNodeParams) -> CuResult,
    name: unsafe extern "C" fn(*mut *const c_char, Handle) -> CuResult,
    attribute: unsafe extern "C" fn(Handle, c_int, c_int) -> CuResult,
    /// cuFuncGetParamInfo (CUDA 12.4): offset and size of one kernel parameter; an index past the end fails.
    param_info: Option<unsafe extern "C" fn(Handle, usize, *mut usize, *mut usize) -> CuResult>,
    /// cuGraphKernelNodeSetAttribute, for clearing a replaced node's cooperative-launch attribute.
    node_attribute: Option<unsafe extern "C" fn(Handle, c_int, *const [u64; 8]) -> CuResult>,
}
fn graph_edit() -> Result<&'static GraphEdit> {
    static EDIT: std::sync::OnceLock<Option<GraphEdit>> = std::sync::OnceLock::new();
    EDIT.get_or_init(|| unsafe {
        let lib = Library::new("libcuda.so.1").ok()?;
        let node_type = *lib.get(b"cuGraphNodeGetType\0").ok()?;
        let get = *lib.get(b"cuGraphKernelNodeGetParams_v2\0").ok()?;
        let set = *lib.get(b"cuGraphKernelNodeSetParams_v2\0").ok()?;
        let name = *lib.get(b"cuFuncGetName\0").ok()?;
        let attribute = *lib.get(b"cuFuncSetAttribute\0").ok()?;
        let param_info = lib.get(b"cuFuncGetParamInfo\0").ok().map(|f| *f);
        let node_attribute = lib.get(b"cuGraphKernelNodeSetAttribute\0").ok().map(|f| *f);
        Some(GraphEdit { _library: lib, node_type, get, set, name, attribute, param_info, node_attribute })
    }).as_ref().context("the CUDA driver lacks the graph kernel-node entry points")
}

macro_rules! api {
    ($($field:ident : $ty:ty = $sym:literal;)+) => {
        struct Api { _library: Library, $($field:$ty,)+ }
        impl Api {
            unsafe fn load() -> Result<Self> {
                let lib=unsafe {Library::new("libcuda.so.1")}.context("load NVIDIA CUDA driver")?;
                $(let $field: $ty = *unsafe { lib.get(concat!($sym,"\0").as_bytes()) }.context($sym)?;)+
                Ok(Self {_library:lib,$($field,)+})
            }
        }
    }
}
api! {
    init: unsafe extern "C" fn(c_uint)->CuResult = "cuInit";
    device_get: unsafe extern "C" fn(*mut c_int,c_int)->CuResult = "cuDeviceGet";
    device_attribute: unsafe extern "C" fn(*mut c_int,c_int,c_int)->CuResult = "cuDeviceGetAttribute";
    primary_retain: unsafe extern "C" fn(*mut Handle,c_int)->CuResult = "cuDevicePrimaryCtxRetain";
    primary_release: unsafe extern "C" fn(c_int)->CuResult = "cuDevicePrimaryCtxRelease_v2";
    context_flags: unsafe extern "C" fn(*mut c_uint)->CuResult = "cuCtxGetFlags";
    set_current: unsafe extern "C" fn(Handle)->CuResult = "cuCtxSetCurrent";
    context_sync: unsafe extern "C" fn()->CuResult = "cuCtxSynchronize";
    alloc: unsafe extern "C" fn(*mut u64,usize)->CuResult = "cuMemAlloc_v2";
    free: unsafe extern "C" fn(u64)->CuResult = "cuMemFree_v2";
    mem_create: unsafe extern "C" fn(*mut u64,usize,*const MemAllocationProp,u64)->CuResult = "cuMemCreate";
    mem_release: unsafe extern "C" fn(u64)->CuResult = "cuMemRelease";
    mem_granularity: unsafe extern "C" fn(*mut usize,*const MemAllocationProp,c_int)->CuResult = "cuMemGetAllocationGranularity";
    mem_properties: unsafe extern "C" fn(*mut MemAllocationProp,u64)->CuResult = "cuMemGetAllocationPropertiesFromHandle";
    mem_import: unsafe extern "C" fn(*mut u64,*mut c_void,c_int)->CuResult = "cuMemImportFromShareableHandle";
    address_reserve: unsafe extern "C" fn(*mut u64,usize,usize,u64,u64)->CuResult = "cuMemAddressReserve";
    address_free: unsafe extern "C" fn(u64,usize)->CuResult = "cuMemAddressFree";
    mem_map: unsafe extern "C" fn(u64,usize,usize,u64,u64)->CuResult = "cuMemMap";
    mem_unmap: unsafe extern "C" fn(u64,usize)->CuResult = "cuMemUnmap";
    mem_access: unsafe extern "C" fn(u64,usize,*const MemAccessDesc,usize)->CuResult = "cuMemSetAccess";
    info: unsafe extern "C" fn(*mut usize,*mut usize)->CuResult = "cuMemGetInfo_v2";
    zero_async: unsafe extern "C" fn(u64,u8,usize,Handle)->CuResult = "cuMemsetD8Async";
    htod: unsafe extern "C" fn(u64,*const c_void,usize)->CuResult = "cuMemcpyHtoD_v2";
    htod_async: unsafe extern "C" fn(u64,*const c_void,usize,Handle)->CuResult = "cuMemcpyHtoDAsync_v2";
    dtoh: unsafe extern "C" fn(*mut c_void,u64,usize)->CuResult = "cuMemcpyDtoH_v2";
    dtod_async: unsafe extern "C" fn(u64,u64,usize,Handle)->CuResult = "cuMemcpyDtoDAsync_v2";
    zero: unsafe extern "C" fn(u64,u8,usize)->CuResult = "cuMemsetD8_v2";
    stream_create: unsafe extern "C" fn(*mut Handle,c_uint)->CuResult = "cuStreamCreate";
    stream_destroy: unsafe extern "C" fn(Handle)->CuResult = "cuStreamDestroy_v2";
    stream_sync: unsafe extern "C" fn(Handle)->CuResult = "cuStreamSynchronize";
    stream_wait_event: unsafe extern "C" fn(Handle,Handle,c_uint)->CuResult = "cuStreamWaitEvent";
    stream_capture_status: unsafe extern "C" fn(Handle,*mut c_int)->CuResult = "cuStreamIsCapturing";
    module_load: unsafe extern "C" fn(*mut Handle,*const c_void)->CuResult = "cuModuleLoadData";
    module_unload: unsafe extern "C" fn(Handle)->CuResult = "cuModuleUnload";
    module_loading_mode: unsafe extern "C" fn(*mut c_int)->CuResult = "cuModuleGetLoadingMode";
    get_function: unsafe extern "C" fn(*mut Handle,Handle,*const c_char)->CuResult = "cuModuleGetFunction";
    launch: unsafe extern "C" fn(Handle,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,c_uint,Handle,*mut *mut c_void,*mut *mut c_void)->CuResult = "cuLaunchKernel";
    launch_ex: unsafe extern "C" fn(*const LaunchConfig,Handle,*mut *mut c_void,*mut *mut c_void)->CuResult = "cuLaunchKernelEx";
    error_string: unsafe extern "C" fn(CuResult,*mut *const c_char)->CuResult = "cuGetErrorString";
    event_create: unsafe extern "C" fn(*mut Handle,c_uint)->CuResult = "cuEventCreate";
    event_destroy: unsafe extern "C" fn(Handle)->CuResult = "cuEventDestroy_v2";
    event_record: unsafe extern "C" fn(Handle,Handle)->CuResult = "cuEventRecord";
    event_sync: unsafe extern "C" fn(Handle)->CuResult = "cuEventSynchronize";
    event_query: unsafe extern "C" fn(Handle)->CuResult = "cuEventQuery";
    event_elapsed: unsafe extern "C" fn(*mut f32,Handle,Handle)->CuResult = "cuEventElapsedTime";
    profiler_start: unsafe extern "C" fn()->CuResult = "cuProfilerStart";
    profiler_stop: unsafe extern "C" fn()->CuResult = "cuProfilerStop";
    capture_begin: unsafe extern "C" fn(Handle,c_int)->CuResult = "cuStreamBeginCapture_v2";
    capture_end: unsafe extern "C" fn(Handle,*mut Handle)->CuResult = "cuStreamEndCapture";
    graph_instantiate: unsafe extern "C" fn(*mut Handle,Handle,u64)->CuResult = "cuGraphInstantiateWithFlags";
    graph_destroy: unsafe extern "C" fn(Handle)->CuResult = "cuGraphDestroy";
    graph_nodes: unsafe extern "C" fn(Handle,*mut Handle,*mut usize)->CuResult = "cuGraphGetNodes";
    graph_dot: unsafe extern "C" fn(Handle,*const c_char,c_uint)->CuResult = "cuGraphDebugDotPrint";
    exec_destroy: unsafe extern "C" fn(Handle)->CuResult = "cuGraphExecDestroy";
    graph_launch: unsafe extern "C" fn(Handle,Handle)->CuResult = "cuGraphLaunch";
    host_alloc: unsafe extern "C" fn(*mut *mut c_void,usize,c_uint)->CuResult = "cuMemHostAlloc";
    host_device: unsafe extern "C" fn(*mut u64,*mut c_void,c_uint)->CuResult = "cuMemHostGetDevicePointer_v2";
    host_free: unsafe extern "C" fn(*mut c_void)->CuResult = "cuMemFreeHost";
    function_attr: unsafe extern "C" fn(Handle,c_int,c_int)->CuResult = "cuFuncSetAttribute";
}
/// Failure-only allocation diagnostics. Driver refusals are otherwise visible
/// only as kernel `NV_ERR_NO_MEMORY` lines without a size or caller.
fn log_allocation_failure(result: &Result<()>, operation: &str, bytes: usize,
    site: &'static std::panic::Location<'static>, cuda_free: Option<(usize, usize)>) {
    let Err(error) = result else { return; };
    let Some(driver) = error.downcast_ref::<DriverError>() else { return; };
    if driver.code != 2 { return; }
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let field = |name: &str| meminfo.lines().find(|l| l.starts_with(name))
        .and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()).map(|kb| kb * 1024);
    eprintln!("allocation_failure {}", serde_json::json!({
        "operation": operation, "bytes": bytes, "site": format!("{}:{}", site.file(), site.line()),
        "cuda_free": cuda_free.map(|f| f.0), "cuda_total": cuda_free.map(|f| f.1),
        "mem_free": field("MemFree:"), "mem_available": field("MemAvailable:"), "cached": field("Cached:"),
        "s_reclaimable": field("SReclaimable:"), "unix_ms": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)}));
}
impl Api {
    fn check(&self, code: CuResult, op: &str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let mut p = ptr::null();
        let s = if unsafe { (self.error_string)(code, &mut p) } == 0 && !p.is_null() {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        } else {
            format!("CUDA error {code}")
        };
        Err(DriverError { code, operation: op.to_owned(), message: s }.into())
    }
}
struct ContextOwner {
    api: Api,
    device: c_int,
    context: Handle,
    auxiliary: RefCell<Option<(usize, Weak<Stream>)>>,
    coefficients: RefCell<Option<(usize, Weak<Stream>)>>,
    queries: RefCell<Option<(usize, Weak<Stream>)>>,
    hints: RefCell<Option<(usize, Weak<Stream>)>>,
    row_arena: RefCell<Option<Rc<RowArena>>>,
    chunk_scratch: RefCell<Option<(Rc<RowArena>, u32)>>,
    // A weak entry avoids a ContextOwner -> ModuleOwner -> ContextOwner cycle.
    // Only explicitly registered, immutable core code is eligible.
    shared_core: RefCell<Option<([u8;32],usize,Weak<ModuleOwner>)>>,
    allocation_floor_screen: bool,
    allocation_floor_pending: std::cell::Cell<usize>,
    allocation_sites: Option<RefCell<std::collections::BTreeMap<u64,(&'static std::panic::Location<'static>,usize)>>>,
    image_pool: RefCell<Option<Rc<ChunkPool>>>,
    /// config::wide16_elastic: the widest (16,384-row) profile's shared arenas, each with its resident prefix length
    /// (the 8,192-row size); the spans above it are mapped from the image pool only while lent (Cuda::wide16_map).
    wide16: RefCell<Vec<(Weak<RowArena>, usize)>>,
    wide16_mapped: std::cell::Cell<bool>,
    /// The image pool was created by wide16_begin_temporary (released by wide16_end_temporary).
    wide16_temporary_pool: std::cell::Cell<bool>,
    /// config::shared_padded_inputs: each padded-K operand's backing (key: canonical weights, scales, buffer kind) and
    /// the last profile scope that took it. Weak: the takers' carves keep a backing alive, never this registry.
    padded_inputs: RefCell<std::collections::HashMap<(u64, u64, u8), (Weak<RowArena>, u64)>>,
    /// The sequential wide profile under construction (Cuda::with_padded_share_scope), if any, and the next scope id.
    padded_scope: std::cell::Cell<Option<u64>>,
    padded_next: std::cell::Cell<u64>,
    /// config::shared_slot_tables: immutable i32 tables by length and SHA-256 of their bytes (Weak, as above).
    immutable_i32: RefCell<std::collections::HashMap<(usize, [u8; 32]), Weak<Buffer<i32>>>>,
}
impl ContextOwner {
    fn current(&self) -> Result<()> {
        self.api.check(
            unsafe { (self.api.set_current)(self.context) },
            "activate context",
        )
    }
}
impl Drop for ContextOwner {
    fn drop(&mut self) {
        // Every pooled image holds this owner, so all chunks are back in the pool by now.
        if let Some(pool) = self.image_pool.borrow_mut().take() && self.current().is_ok() {
            for handle in pool.free.borrow_mut().drain(..) { unsafe { (self.api.mem_release)(handle); } }
        }
        unsafe {
            (self.api.primary_release)(self.device);
        }
    }
}

/// Nov 17 (the owner: "preallocate as much RAM up-front as we reasonably and justifiably expect to need, and then just hold
/// onto this RAM and reuse it"): conversation snapshots (cache entries, paused requests) take their memory from physical
/// chunks created once at startup, while memory is unfragmented, instead of from fresh driver allocations into memory that
/// days of serving have fragmented (on aged memory those large allocations made the driver fall back from large pages and
/// the kernel reclaim and compact, while sorted memory under the same load did neither). Each snapshot maps the chunks it
/// takes into one contiguous address range (CUDA VMM); dropping it unmaps them and returns them. Any free chunk serves any
/// snapshot, so the pool cannot fragment. CANADA_SCRATCH_IMAGE_POOL_MIB sets the size (unset: no pool).
pub(crate) struct ChunkPool {
    chunk: usize,
    count: usize,
    released: std::cell::Cell<usize>,
    free: RefCell<Vec<u64>>,
}
/// Leave hysteresis for simultaneous HTTP conversion and vision work. At most
/// two allocation cushions (and at most one eighth of the startup pool) become
/// ordinary host memory for the rest of this worker's lifetime.
fn idle_pages_to_return(available: usize, required: usize, cushion: usize, chunk: usize,
    free: usize, keep_bytes: usize, initial: usize, released: usize) -> usize {
    if cushion == 0 || available >= required.saturating_add(cushion / 2) { return 0; }
    let cap = (cushion.saturating_mul(2) / chunk).min(initial / 8);
    required.saturating_add(cushion).saturating_sub(available).div_ceil(chunk)
        .min(free.saturating_sub(keep_bytes.div_ceil(chunk))).min(cap.saturating_sub(released))
}

/// Emergency idle-pool return used only during input-capacity reclamation when
/// the soft-capped pass and general-memory eviction still cannot keep the UMA
/// reserve. Ignores the lifetime cap while still leaving mapped chunks and
/// `keep_bytes` untouched.
fn idle_pages_to_return_emergency(available: usize, required: usize, cushion: usize, chunk: usize,
    free: usize, keep_bytes: usize) -> usize {
    if chunk == 0 || available >= required { return 0; }
    let target = required.saturating_add(cushion.min(128 << 20));
    target.saturating_sub(available).div_ceil(chunk)
        .min(free.saturating_sub(keep_bytes.div_ceil(chunk)))
}

/// Incremental idle pool re-hydration: buy back previously released physical
/// chunks once host memory has comfortably recovered above reserve + cushion + 512 MiB.
fn pool_chunks_to_rehydrate(available: usize, reserve: usize, cushion: usize, chunk: usize,
    released: usize, max_chunks: usize) -> usize {
    if chunk == 0 || released == 0 || max_chunks == 0 { return 0; }
    let floor = reserve.saturating_add(cushion).saturating_add(512 << 20);
    if available <= floor { return 0; }
    ((available - floor) / chunk).min(released).min(max_chunks)
}

#[cfg(test)]
#[test]
fn idle_pool_relief_retains_both_the_next_full_image_and_the_preallocated_pool() {
    let mib = 1 << 20;
    let plan = |available, free, keep, released| idle_pages_to_return(available * mib, 3584 * mib,
        512 * mib, 16 * mib, free, keep * mib, 576, released);
    assert_eq!(plan(3513, 414, 553, 0), 37);
    assert_eq!(plan(3513, 37, 553, 0), 2);
    assert_eq!(plan(3513, 34, 553, 0), 0);
    assert_eq!(plan(3800, 414, 553, 63), 1);
    assert_eq!(plan(3500, 414, 553, 64), 0);
    assert_eq!(plan(3840, 414, 553, 0), 0);
}

#[cfg(test)]
#[test]
fn emergency_pool_relief_bypasses_soft_cap_and_rehydrates_with_hysteresis() {
    let mib = 1 << 20;
    // Past the 64-chunk soft cap, emergency return still frees idle chunks to meet required + 128 MiB.
    assert_eq!(idle_pages_to_return_emergency(3500 * mib, 3600 * mib, 512 * mib, 16 * mib, 200, 0), 15);
    assert_eq!(idle_pages_to_return_emergency(3500 * mib, 3600 * mib, 512 * mib, 16 * mib, 10, 0), 10);
    assert_eq!(idle_pages_to_return_emergency(3500 * mib, 3600 * mib, 512 * mib, 16 * mib, 40, 553 * mib), 5);
    assert_eq!(idle_pages_to_return_emergency(3600 * mib, 3600 * mib, 512 * mib, 16 * mib, 200, 0), 0);
    // Re-hydration stays inactive until available exceeds reserve + cushion + 512 MiB (4608 MiB).
    assert_eq!(pool_chunks_to_rehydrate(4608 * mib, 3584 * mib, 512 * mib, 16 * mib, 64, 8), 0);
    assert_eq!(pool_chunks_to_rehydrate(4640 * mib, 3584 * mib, 512 * mib, 16 * mib, 64, 8), 2);
    assert_eq!(pool_chunks_to_rehydrate(6000 * mib, 3584 * mib, 512 * mib, 16 * mib, 64, 8), 8);
    assert_eq!(pool_chunks_to_rehydrate(6000 * mib, 3584 * mib, 512 * mib, 16 * mib, 3, 8), 3);
    assert_eq!(pool_chunks_to_rehydrate(6000 * mib, 3584 * mib, 512 * mib, 16 * mib, 0, 8), 0);
}
/// Snapshots smaller than this keep ordinary allocations (small blocks are never scarce; a chunk would waste most of itself).
pub(crate) const POOLED_IMAGE_MIN: usize = 8 << 20;
/// CANADA_SCRATCH_ELASTIC_TRACE=1 (diagnostic, default off): one line per elastic map and unmap (range, span).
fn elastic_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CANADA_SCRATCH_ELASTIC_TRACE").as_deref() == Ok("1"))
}
/// An elastic range (Nov 18 2026): one reserved address range
/// whose resident prefix has its own memory and whose later chunks are mapped from the image pool only while a span
/// holding them is mapped. Addresses never change, so captured graphs stay valid. A chunk is mapped while any mapped
/// span covers it (reference counts: two serving regions share one page at their boundary). No kernel may touch an
/// unmapped chunk; unmapping drains the context first, as every VMM unmap here does.
struct ElasticAllocation {
    owner: Rc<ContextOwner>,
    ptr: u64,
    span: usize,
    chunk: usize,
    resident: usize,
    resident_handle: u64,
    /// Chunk i >= resident / chunk: Some((pool handle, references)) while mapped.
    chunks: RefCell<Vec<Option<(u64, u32)>>>,
}
impl ElasticAllocation {
    fn chunk_range(&self, offset: usize, len: usize) -> Result<std::ops::Range<usize>> {
        let end = offset.checked_add(len).context("elastic span overflow")?;
        ensure!(len > 0 && end <= self.span, "elastic span outside its range");
        // Chunks wholly inside the resident prefix are always mapped.
        let first = (offset / self.chunk).max(self.resident / self.chunk);
        let last = end.div_ceil(self.chunk);
        Ok(first..last.max(first))
    }
    /// Pool chunks a map of this span would take now.
    fn missing(&self, offset: usize, len: usize) -> Result<usize> {
        let range = self.chunk_range(offset, len)?;
        let base = self.resident / self.chunk;
        let chunks = self.chunks.borrow();
        Ok(range.filter(|&i| chunks[i - base].is_none()).count())
    }
    fn mapped_extents(&self) -> Vec<(u64, usize)> {
        let mut out: Vec<(u64, usize)> = Vec::new();
        if self.resident > 0 { out.push((self.ptr, self.resident)); }
        let base = self.resident / self.chunk;
        for (j, slot) in self.chunks.borrow().iter().enumerate() {
            if slot.is_none() { continue; }
            let at = self.ptr + ((base + j) * self.chunk) as u64;
            match out.last_mut() {
                Some(last) if last.0 + last.1 as u64 == at => last.1 += self.chunk,
                _ => out.push((at, self.chunk)),
            }
        }
        out
    }
}
impl Drop for ElasticAllocation {
    fn drop(&mut self) {
        let pool = self.owner.image_pool.borrow().clone();
        let mut returned = Vec::new();
        if self.owner.current().is_ok() {
            let api = &self.owner.api;
            unsafe {
                (api.context_sync)();
                let base = self.resident / self.chunk;
                for (j, slot) in self.chunks.borrow().iter().enumerate() {
                    if let Some((handle, _)) = slot {
                        (api.mem_unmap)(self.ptr + ((base + j) * self.chunk) as u64, self.chunk);
                        returned.push(*handle);
                    }
                }
                if self.resident > 0 { (api.mem_unmap)(self.ptr, self.resident); }
                if self.ptr != 0 { (api.address_free)(self.ptr, self.span); }
                if self.resident_handle != 0 { (api.mem_release)(self.resident_handle); }
            }
        }
        if let Some(pool) = pool { pool.free.borrow_mut().extend(returned); }
    }
}
struct PooledAllocation {
    owner: Rc<ContextOwner>,
    pool: Rc<ChunkPool>,
    ptr: u64,
    span: usize,
    handles: Vec<u64>,
    mapped: usize,
}
impl Drop for PooledAllocation {
    fn drop(&mut self) {
        if self.owner.current().is_ok() {
            let api = &self.owner.api;
            unsafe {
                // Like any VMM unmap, this does not join GPU users of the range; drain first.
                if self.mapped > 0 { (api.context_sync)(); }
                for i in 0..self.mapped { (api.mem_unmap)(self.ptr + (i * self.pool.chunk) as u64, self.pool.chunk); }
                if self.ptr != 0 { (api.address_free)(self.ptr, self.span); }
            }
        }
        self.pool.free.borrow_mut().extend(self.handles.drain(..));
    }
}

#[derive(Clone)]
pub struct Cuda {
    owner: Rc<ContextOwner>,
    mapped_metadata_bytes: Option<Rc<std::cell::Cell<usize>>>,
    staged_metadata: Option<Rc<MetadataStaging>>,
}
impl Cuda {
    /// Drain every stream owned by this context before an external hold is
    /// acknowledged. Ordinary token generation keeps its stream-local path.
    pub(crate) fn synchronize_all(&self) -> Result<()> {
        self.owner.current()?;
        self.owner.api.check(unsafe { (self.owner.api.context_sync)() }, "drain CUDA context")
    }
    /// A local allocator for small CPU-written scheduling and sampler controls
    /// (IDs and RNG uniforms). Never pass it to model-value, probability or
    /// weight constructors. GPU kernels must not write these buffers.
    /// Safety: all GPU readers must complete before each host upload, and
    /// uploads must finish before the next GPU reader is launched.
    pub(crate) unsafe fn mapped_metadata_allocator(&self) -> Result<Self> {
        self.qualify_gb10()?;
        ensure!(self.attribute(19)? != 0, "mapped host memory unavailable");
        Ok(Self {owner:self.owner.clone(),mapped_metadata_bytes:Some(Rc::new(std::cell::Cell::new(0))),staged_metadata:None})
    }
    pub(crate) fn mapped_metadata_bytes(&self)->usize {
        self.mapped_metadata_bytes.as_ref().map_or(0,|n|n.get())
    }
    /// Safety: scope this allocator to CPU-written scheduling metadata. Flush
    /// after construction and updates, and complete the previous copy before
    /// modifying its pinned source. GPU kernels must never write these buffers.
    pub(crate) unsafe fn staged_metadata_allocator(&self) -> Result<Self> {
        ensure!(self.mapped_metadata_bytes.is_none() && self.staged_metadata.is_none(),
            "metadata allocator cannot be nested");
        let mut device = self.allocate::<u8>(65536)?;
        device.zero()?;
        let staging = Rc::new(MetadataStaging { device, host: self.mapped_host(65536)?,
            extent: std::cell::Cell::new(0), payload: std::cell::Cell::new(0) });
        Ok(Self {owner:self.owner.clone(),mapped_metadata_bytes:None,staged_metadata:Some(staging)})
    }
    pub(crate) fn metadata_staging(&self) -> Option<Rc<MetadataStaging>> {
        self.staged_metadata.clone()
    }
    /// Cache key for context-scoped external primitive handles. The cache must
    /// retain only weak references; owning primitives already retain CUDA.
    pub(crate) fn context_key(&self) -> usize {
        self.owner.context as usize
    }
    /// Diagnostic profiling control only. Call outside graph capture, with the
    /// owning context current and an explicitly selected external profiler.
    pub fn profiler_start(&self) -> Result<()> {
        self.owner.current()?;
        self.owner.api.check(
            unsafe { (self.owner.api.profiler_start)() },
            "start diagnostic profiler",
        )
    }
    pub fn profiler_stop(&self) -> Result<()> {
        self.owner.current()?;
        self.owner.api.check(
            unsafe { (self.owner.api.profiler_stop)() },
            "stop diagnostic profiler",
        )
    }
    /// Safety: this complete writable range belongs to this context and lives
    /// through stream completion. Required preclear for atomic-output kernels.
    pub unsafe fn zero_device(
        &self,
        destination: u64,
        bytes: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            destination != 0 && bytes > 0 && self.owns_stream(stream),
            "device zero range/context"
        );
        self.owner.current()?;
        self.owner.api.check(
            unsafe { (self.owner.api.zero_async)(destination, 0, bytes, stream.raw()) },
            "zero device asynchronously",
        )
    }
    /// Diagnostic byte poison, not tensor arithmetic. Safety: the entire range
    /// is writable in this context and its owner excludes concurrent readers.
    pub(crate) unsafe fn poison_device_for_probe(
        &self,destination:u64,bytes:usize,value:u8,stream:&Stream,
    )->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_ENCODER_CACHE_POISON_SCREEN").as_deref()==Ok("1")
            &&destination!=0&&bytes>0&&[64,255].contains(&value)&&self.owns_stream(stream),"finite decoder poison admission");
        self.owner.current()?;
        self.owner.api.check(unsafe{(self.owner.api.zero_async)(destination,value,bytes,stream.raw())},
            "poison obsolete decoder bytes asynchronously")
    }
    /// config::index_bf16_check (Nov 19 diagnostic): fill a score range with byte `value` (0xFF: NaN floats) before a bf16
    /// pass, to find positions it leaves unwritten. Safety: as poison_device_for_probe.
    pub(crate) unsafe fn fill_device_for_bf16_check(&self, destination: u64, bytes: usize, value: u8, stream: &Stream)
        -> Result<()> {
        ensure!(crate::config::index_bf16_check() && destination != 0 && bytes > 0 && self.owns_stream(stream),
            "bf16 check fill admission");
        self.owner.current()?;
        self.owner.api.check(unsafe { (self.owner.api.zero_async)(destination, value, bytes, stream.raw()) },
            "fill a score range for the bf16 check")
    }
    /// Safety: the range is a live initialized device allocation in this context;
    /// any producer stream has completed. Diagnostic copy, never numeric execution.
    pub unsafe fn read_device(&self, source: u64, bytes: usize) -> Result<Vec<u8>> {
        ensure!(
            source != 0 && bytes > 0 && bytes <= 64 * 1024 * 1024,
            "bounded diagnostic device read"
        );
        self.owner.current()?;
        let mut data = vec![0u8; bytes];
        self.owner.api.check(
            unsafe { (self.owner.api.dtoh)(data.as_mut_ptr().cast(), source, bytes) },
            "diagnostic device read",
        )?;
        Ok(data)
    }

    pub fn owns_buffer<T: DeviceElement>(&self, buffer: &Buffer<T>) -> bool {
        self.owner.context == buffer.owner.context
    }
    pub fn owns_mapped_host(&self, region: &MappedHost) -> bool {
        self.owner.context == region.owner.context
    }
    pub fn owns_stream(&self, stream: &Stream) -> bool {
        self.owner.context == stream.owner.context
    }

    /// Safety: both address ranges are valid in this context, disjoint, and
    /// live through stream completion. Caller exclusively owns destination.
    pub unsafe fn copy_device(
        &self,
        destination: u64,
        source: u64,
        bytes: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.owner, &stream.owner),
            "copy stream belongs to another context"
        );
        ensure!(
            bytes > 0 && destination != 0 && source != 0,
            "invalid device copy"
        );
        self.owner.current()?;
        self.owner.api.check(
            unsafe { (self.owner.api.dtod_async)(destination, source, bytes, stream.raw()) },
            "copy device bytes",
        )
    }
    #[track_caller]
    pub fn mapped_host(&self, bytes: usize) -> Result<MappedHost> {
        ensure!(
            bytes > 0 && bytes <= 1024 * 1024 * 1024,
            "bounded pinned host allocation required"
        );
        self.owner.current()?;
        // CANADA_SCRATCH_ROCE_HOST_NUMA=1: allocate via cuMemCreate(HOST_NUMA) mapped to both
        // DEVICE and HOST_NUMA at one virtual address so GPU staging/reduce reads run at full GB10
        // DRAM bandwidth (~25% faster than cuMemHostAlloc) while remaining ibv_reg_mr-compatible.
        if std::env::var("CANADA_SCRATCH_ROCE_HOST_NUMA").as_deref() == Ok("1") {
            let api = &self.owner.api;
            let prop = MemAllocationProp {
                kind: 1, // CU_MEM_ALLOCATION_TYPE_PINNED
                location: MemLocation { kind: 3, id: 0 }, // CU_MEM_LOCATION_TYPE_HOST_NUMA
                ..MemAllocationProp::default()
            };
            let mut gran = 0usize;
            api.check(unsafe { (api.mem_granularity)(&mut gran, &prop, 0) }, "query host NUMA granularity")?;
            ensure!(gran > 0 && gran.is_power_of_two(), "invalid host NUMA VMM granularity");
            let vmm_size = bytes.div_ceil(gran) * gran;
            let mut handle = 0u64;
            api.check(unsafe { (api.mem_create)(&mut handle, vmm_size, &prop, 0) }, "create host NUMA pinned allocation")?;
            let mut va = 0u64;
            if let Err(e) = api.check(unsafe { (api.address_reserve)(&mut va, vmm_size, gran, 0, 0) }, "reserve host NUMA VA") {
                unsafe { (api.mem_release)(handle); }
                return Err(e);
            }
            if let Err(e) = api.check(unsafe { (api.mem_map)(va, vmm_size, 0, handle, 0) }, "map host NUMA VA") {
                unsafe { (api.address_free)(va, vmm_size); (api.mem_release)(handle); }
                return Err(e);
            }
            let acc = [
                MemAccessDesc { location: MemLocation { kind: 1, id: self.owner.device }, flags: 3 },
                MemAccessDesc { location: MemLocation { kind: 3, id: 0 }, flags: 3 },
            ];
            if let Err(e) = api.check(unsafe { (api.mem_access)(va, vmm_size, acc.as_ptr(), acc.len()) }, "set host NUMA access") {
                unsafe { (api.mem_unmap)(va, vmm_size); (api.address_free)(va, vmm_size); (api.mem_release)(handle); }
                return Err(e);
            }
            let host = va as *mut c_void;
            unsafe { std::ptr::write_bytes(host.cast::<u8>(), 0, bytes); }
            eprintln!("mapped_host: allocated {} bytes via cuMemCreate(HOST_NUMA) at 0x{:x} (vmm_size={})", bytes, va, vmm_size);
            return Ok(MappedHost {
                owner: self.owner.clone(),
                host,
                device: va,
                bytes,
                vmm_handle: handle,
                vmm_size,
            });
        }
        let mut host = ptr::null_mut();
        let result = self.owner.api.check(
            unsafe { (self.owner.api.host_alloc)(&mut host, bytes, 3) },
            "allocate mapped host region",
        );
        if result.is_err() { log_allocation_failure(&result, "allocate mapped host region", bytes, std::panic::Location::caller(), self.memory().ok()); }
        result?;
        let mut region = MappedHost {
            owner: self.owner.clone(),
            host,
            device: 0,
            bytes,
            vmm_handle: 0,
            vmm_size: 0,
        };
        self.owner.api.check(
            unsafe { (self.owner.api.host_device)(&mut region.device, host, 0) },
            "map host region to GPU",
        )?;
        unsafe {
            std::ptr::write_bytes(host.cast::<u8>(), 0, bytes);
        }
        Ok(region)
    }
    pub fn open(ordinal: i32) -> Result<Self> {
        let api = unsafe { Api::load() }?;
        api.check(unsafe { (api.init)(0) }, "initialize CUDA")?;
        let mut device = 0;
        api.check(
            unsafe { (api.device_get)(&mut device, ordinal) },
            "select device",
        )?;
        let mut context = ptr::null_mut();
        api.check(
            unsafe { (api.primary_retain)(&mut context, device) },
            "retain context",
        )?;
        let c = Self {
            owner: Rc::new(ContextOwner {
                api,
                device,
                context,
                auxiliary: RefCell::new(None),
                coefficients: RefCell::new(None),
                queries: RefCell::new(None),
                hints: RefCell::new(None),
                row_arena: RefCell::new(None),
                chunk_scratch: RefCell::new(None),
                shared_core: RefCell::new(None),
                allocation_floor_screen: std::env::var("CANADA_SCRATCH_ALLOCATION_FLOOR_SCREEN").as_deref()==Ok("1"),
                allocation_sites: (std::env::var("CANADA_SCRATCH_ALLOCATION_SITES_SCREEN").as_deref()==Ok("1"))
                    .then(||RefCell::new(std::collections::BTreeMap::new())),
                allocation_floor_pending: std::cell::Cell::new(0),
                image_pool: RefCell::new(None),
                wide16: RefCell::new(Vec::new()),
                wide16_mapped: std::cell::Cell::new(false),
                wide16_temporary_pool: std::cell::Cell::new(false),
                padded_inputs: RefCell::new(std::collections::HashMap::new()),
                padded_scope: std::cell::Cell::new(None),
                padded_next: std::cell::Cell::new(0),
                immutable_i32: RefCell::new(std::collections::HashMap::new()),
            }),
            mapped_metadata_bytes: None,
            staged_metadata: None,
        };
        c.owner.current()?;
        Ok(c)
    }
    pub fn attribute(&self, id: i32) -> Result<i32> {
        let mut v = 0;
        self.owner.api.check(
            unsafe { (self.owner.api.device_attribute)(&mut v, id, self.owner.device) },
            "device attribute",
        )?;
        Ok(v)
    }
    pub(crate) fn context_flags(&self) -> Result<u32> {
        self.owner.current()?;
        let mut flags = 0;
        self.owner.api.check(unsafe { (self.owner.api.context_flags)(&mut flags) },
            "read context scheduling flags")?;
        Ok(flags)
    }
    /// Report the initialized driver's policy rather than assuming that an
    /// inherited environment variable selected the intended loading mode.
    pub fn module_loading_mode(&self) -> Result<&'static str> {
        let mut mode = 0;
        self.owner.api.check(
            unsafe { (self.owner.api.module_loading_mode)(&mut mode) },
            "query CUDA module loading mode",
        )?;
        match mode {
            1 => Ok("EAGER"),
            2 => Ok("LAZY"),
            _ => bail!("unknown CUDA module loading mode {mode}"),
        }
    }
    pub fn qualify_gb10(&self) -> Result<()> {
        // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR and MULTIPROCESSOR_COUNT.
        ensure!(
            self.attribute(75)? == 12 && self.attribute(76)? == 1 && self.attribute(16)? == 48,
            "kernel artifacts require SM121 with 48 SMs"
        );
        Ok(())
    }
    pub fn memory(&self) -> Result<(usize, usize)> {
        self.owner.current()?;
        let (mut free, mut total) = (0, 0);
        self.owner.api.check(
            unsafe { (self.owner.api.info)(&mut free, &mut total) },
            "query memory",
        )?;
        Ok((free, total))
    }
    /// Linux can reclaim clean file pages for device allocations on this
    /// specifically admitted UMA device. Never apply this to discrete GPUs.
    pub fn admission_memory(&self) -> Result<crate::memory::Available> {
        self.qualify_gb10()?;
        ensure!(
            self.attribute(18)? == 1,
            "UMA admission requires an integrated device"
        );
        let (free, total) = self.memory()?;
        crate::memory::Available::gb10(free, total)
    }
    /// Carve the row-sized transient allocations made by `build` into `arena`.
    /// Each invocation restarts at the arena's beginning. Separate invocations
    /// therefore alias and must execute sequentially. Wide layers alternate
    /// parity arenas; a complete narrow profile may instead carve all layers
    /// in one invocation, with a distinct arena for a concurrently paired owner.
    pub fn with_row_arena<R>(&self, arena: &Rc<RowArena>, build: impl FnOnce() -> Result<R>) -> Result<R> {
        ensure!(
            Rc::ptr_eq(&self.owner, &arena.storage.owner)
                && self.owner.row_arena.borrow().is_none(),
            "row arena context or nesting"
        );
        arena.next.set(0);
        *self.owner.row_arena.borrow_mut() = Some(arena.clone());
        let result = build();
        *self.owner.row_arena.borrow_mut() = None;
        result
    }
    /// As with_row_arena, continuing after the arena's current extent instead of
    /// restarting, so several builders of one profile get distinct storage. The
    /// caller rewinds the arena once per profile (RowArena::rewind).
    pub(crate) fn with_row_arena_append<R>(&self, arena: &Rc<RowArena>, build: impl FnOnce() -> Result<R>) -> Result<R> {
        ensure!(
            Rc::ptr_eq(&self.owner, &arena.storage.owner)
                && self.owner.row_arena.borrow().is_none(),
            "row arena context or nesting"
        );
        *self.owner.row_arena.borrow_mut() = Some(arena.clone());
        let result = build();
        *self.owner.row_arena.borrow_mut() = None;
        result
    }
    /// Row-proportional transient storage. Outside a row-arena scope this is an
    /// ordinary owned allocation, so existing profiles are unchanged.
    #[track_caller]
    pub fn allocate_rows<T: DeviceElement>(&self, len: usize) -> Result<Buffer<T>> {
        let arena = self.owner.row_arena.borrow().clone();
        let Some(arena) = arena else {
            return self.allocate(len);
        };
        self.carve(arena, len)
    }
    /// Scratch that a prefill chunk rewrites in full before reading it and that nothing
    /// reads once the chunk has finished. Inside a `with_chunk_scratch` scope it is carved
    /// from that scope's arena, after the arena's current extent; anywhere else it is an
    /// ordinary owned allocation. Only construction sites audited for that lifetime open
    /// such a scope.
    #[track_caller]
    pub(crate) fn allocate_chunk_scratch<T: DeviceElement>(&self, kind: ChunkScratch, len: usize) -> Result<Buffer<T>> {
        let scope = self.owner.chunk_scratch.borrow().clone();
        match scope {
            Some((arena, kinds)) if kinds & kind as u32 != 0 => self.carve(arena, len),
            _ => self.allocate(len),
        }
    }
    /// Lend `arena` to the chunk-scratch allocations of the given `kinds` (a mask of
    /// ChunkScratch values) made by `build`. Independent of the row-arena scope, so a
    /// builder may hold both.
    pub(crate) fn with_chunk_scratch<R>(&self, arena: &Rc<RowArena>, kinds: u32, build: impl FnOnce() -> Result<R>) -> Result<R> {
        ensure!(
            Rc::ptr_eq(&self.owner, &arena.storage.owner)
                && self.owner.chunk_scratch.borrow().is_none(),
            "chunk scratch context or nesting"
        );
        *self.owner.chunk_scratch.borrow_mut() = Some((arena.clone(), kinds));
        let result = build();
        *self.owner.chunk_scratch.borrow_mut() = None;
        result
    }
    fn carve<T: DeviceElement>(&self, arena: Rc<RowArena>, len: usize) -> Result<Buffer<T>> {
        ensure!(len > 0, "zero-size allocation");
        let bytes = len
            .checked_mul(size_of::<T>())
            .context("allocation size overflow")?;
        let start = arena.next.get().next_multiple_of(256);
        let end = start.checked_add(bytes).context("row arena overflow")?;
        ensure!(
            end <= arena.storage.len,
            "row arena capacity {} below layer need {end}",
            arena.storage.len
        );
        arena.next.set(end);
        arena.high_water.set(arena.high_water.get().max(end));
        arena.profile_high.set(arena.profile_high.get().max(end));
        Ok(Buffer {
            owner: self.owner.clone(),
            ptr: arena.storage.ptr + start as u64,
            len,
            storage: Storage::Arena(arena),
            _element: PhantomData,
        })
    }
    /// Keep terminal IO outside a lent transient arena. In particular, a
    /// completed prefill's logits must survive the next owner's prefill.
    pub(crate) fn without_row_arena<R>(&self,build:impl FnOnce()->Result<R>)->Result<R> {
        let previous=self.owner.row_arena.borrow_mut().take();
        let result=build();
        *self.owner.row_arena.borrow_mut()=previous;
        result
    }
    #[track_caller]
    pub fn row_arena(&self, bytes: usize) -> Result<Rc<RowArena>> {
        Ok(Rc::new(RowArena {
            storage: self.allocate(bytes)?,
            next: std::cell::Cell::new(0),
            high_water: std::cell::Cell::new(0),
            profile_high: std::cell::Cell::new(0),
        }))
    }
    /// Run `build` (one sequential wide profile's construction) as a padded-input share scope when `on`
    /// (config::shared_padded_inputs): inside it, Cuda::shared_padded_input lends each padded-K operand's backing once.
    pub(crate) fn with_padded_share_scope<R>(&self, on: bool, build: impl FnOnce() -> Result<R>) -> Result<R> {
        if !on { return build(); }
        ensure!(self.owner.padded_scope.get().is_none(), "padded input share scopes do not nest");
        let id = self.owner.padded_next.get();
        self.owner.padded_next.set(id + 1);
        self.owner.padded_scope.set(Some(id));
        let result = build();
        self.owner.padded_scope.set(None);
        result
    }
    /// config::shared_padded_inputs (Nov 19 2026): the first `len` bytes of padded-K operand `key`'s zeroed backing
    /// (Matrix::canonical_source plus a buffer kind), created at its first taker's size. Sequential wide profiles are
    /// built widest first and run one at a time on one ordered stream, and only that operand's producers write it (the
    /// k columns of each row), so its pad columns stay zero for every taker. None outside a share scope, when the
    /// backing is smaller than `len`, or when this scope already took it: the caller keeps its own buffer then.
    #[track_caller]
    pub(crate) fn shared_padded_input(&self, key: (u64, u64, u8), len: usize) -> Result<Option<Buffer<u8>>> {
        let Some(scope) = self.owner.padded_scope.get() else { return Ok(None) };
        let existing = self.owner.padded_inputs.borrow().get(&key).and_then(|(weak, taken)| Some((weak.upgrade()?, *taken)));
        let arena = match existing {
            Some((arena, taken)) => {
                if arena.bytes() < len || taken == scope { return Ok(None); }
                arena
            }
            None => {
                let arena = self.row_arena(len)?;
                let mut whole = unsafe { arena.storage.view(0, arena.storage.len)? };
                whole.zero()?;
                arena
            }
        };
        self.owner.padded_inputs.borrow_mut().insert(key, (Rc::downgrade(&arena), scope));
        arena.next.set(0);
        self.carve(arena, len).map(Some)
    }
    /// config::shared_slot_tables (Nov 19 2026): an immutable device copy of `data`. With the flag on, every caller with
    /// the same contents gets the same allocation (no caller may write it); otherwise a fresh upload.
    #[track_caller]
    pub(crate) fn shared_immutable_i32(&self, data: &[i32]) -> Result<Rc<Buffer<i32>>> {
        if !crate::config::shared_slot_tables() {
            let mut b = self.allocate(data.len())?; b.upload(data)?; return Ok(Rc::new(b));
        }
        use sha2::Digest;
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) };
        let key = (data.len(), <[u8; 32]>::from(sha2::Sha256::digest(bytes)));
        if let Some(b) = self.owner.immutable_i32.borrow().get(&key).and_then(Weak::upgrade) { return Ok(b); }
        let mut b = self.allocate(data.len())?; b.upload(data)?;
        let b = Rc::new(b);
        let mut map = self.owner.immutable_i32.borrow_mut();
        map.retain(|_, w| w.strong_count() > 0);
        map.insert(key, Rc::downgrade(&b));
        Ok(b)
    }
    /// A row arena on an elastic range (config::wide16_elastic, Nov 19 2026): `resident` bytes have memory now; the
    /// rest is registered as part of the 16K extension and has memory only while Cuda::wide16_map lends it pool chunks.
    /// Carving is unchanged (addresses never move, so captured graphs stay valid); no kernel may run on a carve beyond
    /// the resident prefix while the extension is unmapped (the prefill planner offers 16,384 rows only while mapped).
    pub(crate) fn row_arena_wide16(&self, bytes: usize, resident: usize, label: &str) -> Result<Rc<RowArena>> {
        let storage = self.allocate_elastic(bytes, resident, label)?;
        let resident = storage.elastic_resident().context("elastic arena")?;
        let arena = Rc::new(RowArena {storage, next: std::cell::Cell::new(0), high_water: std::cell::Cell::new(0),
            profile_high: std::cell::Cell::new(0)});
        if resident < arena.storage.len {
            self.owner.wide16.borrow_mut().push((Rc::downgrade(&arena), resident));
        }
        Ok(arena)
    }
    fn wide16_arenas(&self) -> Vec<(Rc<RowArena>, usize)> {
        self.owner.wide16.borrow().iter().filter_map(|(w, r)| w.upgrade().map(|a| (a, *r))).collect()
    }
    /// Whether this process has a 16K extension (an elastic widest profile).
    pub(crate) fn wide16_configured(&self) -> bool { !self.wide16_arenas().is_empty() }
    pub(crate) fn wide16_mapped(&self) -> bool { self.owner.wide16_mapped.get() }
    /// Bytes of the extension (the spans above the resident prefixes).
    pub(crate) fn wide16_bytes(&self) -> usize { self.wide16_arenas().iter().map(|(a, r)| a.storage.len - r).sum() }
    /// The resident prefix every arena of the extension has (for the narrower profiles' carve check).
    pub(crate) fn wide16_resident_prefixes(&self) -> Vec<(Rc<RowArena>, usize)> { self.wide16_arenas() }
    /// Pool chunks wide16_map would take now.
    pub(crate) fn wide16_missing_chunks(&self) -> Result<usize> {
        if self.wide16_mapped() { return Ok(0); }
        self.wide16_arenas().iter().map(|(a, r)| a.storage.elastic_missing(*r, a.storage.len - r)).sum()
    }
    /// Map every arena's extension from the image pool's free chunks (zeroed), all or nothing. Ok(false), changing
    /// nothing, when the pool has too few free chunks.
    pub(crate) fn wide16_map(&self) -> Result<bool> {
        if self.wide16_mapped() { return Ok(true); }
        let arenas = self.wide16_arenas();
        ensure!(!arenas.is_empty(), "no 16K extension to map");
        let mut done: Vec<&(Rc<RowArena>, usize)> = Vec::new();
        for entry in &arenas {
            let (a, r) = entry;
            match a.storage.elastic_map(*r, a.storage.len - r) {
                Ok(true) => done.push(entry),
                outcome => {
                    for (b, s) in done { b.storage.elastic_unmap(*s, b.storage.len - s)?; }
                    return outcome.map(|_| false);
                }
            }
        }
        self.owner.wide16_mapped.set(true);
        Ok(true)
    }
    /// Return the extension's chunks to the pool (after draining the context, as every VMM unmap here does).
    pub(crate) fn wide16_unmap(&self) -> Result<()> {
        if !self.wide16_mapped() { return Ok(()); }
        for (a, r) in self.wide16_arenas() { a.storage.elastic_unmap(r, a.storage.len - r)?; }
        self.owner.wide16_mapped.set(false);
        Ok(())
    }
    /// Map the extension before the real image pool exists (the widest profile's construction, startup warm-ups): a
    /// temporary pool of exactly the missing chunks backs it until wide16_end_temporary releases it, every chunk
    /// returned (the batched warm-up's pattern for its regions). With a pool present, it maps from that pool.
    pub(crate) fn wide16_begin_temporary(&self) -> Result<()> {
        if !self.wide16_configured() || self.wide16_mapped() { return Ok(()); }
        if self.owner.image_pool.borrow().is_none() {
            let chunks = self.wide16_missing_chunks()?;
            self.create_image_pool(chunks.max(1) * self.image_pool_chunk_bytes()?)?;
            self.owner.wide16_temporary_pool.set(true);
        }
        ensure!(self.wide16_map()?, "16K extension: no room to map it for construction / warm-up");
        Ok(())
    }
    pub(crate) fn wide16_end_temporary(&self) -> Result<()> {
        self.wide16_unmap()?;
        if self.owner.wide16_temporary_pool.replace(false) { self.release_image_pool()?; }
        Ok(())
    }
    /// Run `build` between wide16_begin_temporary and wide16_end_temporary (the end runs on failure too).
    pub(crate) fn with_wide16_temporarily<R>(&self, build: impl FnOnce() -> Result<R>) -> Result<R> {
        if !self.wide16_configured() || self.wide16_mapped() { return build(); }
        self.wide16_begin_temporary()?;
        let result = build();
        let ended = self.wide16_end_temporary();
        let value = result?;
        ended?;
        Ok(value)
    }
    /// Dynamic callers must admit a complete workspace above the same floor
    /// that individual allocations enforce. Otherwise a successful preflight
    /// can fail halfway through a distributed image/cache operation.
    pub(crate) fn allocation_reserve_bytes(&self)->usize {
        crate::memory::floor_bytes()+if self.owner.allocation_floor_screen {512*1024usize.pow(2)}else{0}
    }
    fn check_allocation_floor(&self,bytes:usize)->Result<()> {
        if !self.owner.allocation_floor_screen{return Ok(());}
        let pending=self.owner.allocation_floor_pending.get().saturating_add(bytes);
        self.owner.allocation_floor_pending.set(pending);
        if bytes<1024*1024&&pending<16*1024*1024{return Ok(());}
        self.owner.allocation_floor_pending.set(0);
        let available=self.admission_memory()?.admitted_available;
        let required=self.allocation_reserve_bytes().saturating_add(bytes);
        if available<required {
            self.report_allocation_sites("before allocation-floor refusal")?;
        }
        // Keep the physical8GiB floor plus512MiB for untracked host/vendor
        // activity. This opt-in guard also applies to later dynamic allocations;
        // image/cache preflight must therefore include the same allowance.
        ensure!(available>=required, AllocationFloorError { bytes, available, required });
        Ok(())
    }
    pub(crate) fn report_allocation_sites(&self,label:&str)->Result<()> {
        let Some(sites)=&self.owner.allocation_sites else{return Ok(());};
        let mut grouped=std::collections::BTreeMap::<String,(usize,usize)>::new();
        for (site,bytes)in sites.borrow().values() {
            let item=grouped.entry(format!("{}:{}",site.file(),site.line())).or_default();
            item.0+=bytes;item.1+=1;
        }
        let total:usize=grouped.values().map(|v|v.0).sum();
        let mut sorted=grouped.into_iter().collect::<Vec<_>>();sorted.sort_by_key(|(_,v)|std::cmp::Reverse(v.0));
        eprintln!("allocation_sites {}",serde_json::json!({"label":label,"ordinary_live_bytes_ge64KiB":total,
            "top":sorted.into_iter().take(96).map(|(site,(bytes,count))|serde_json::json!({"site":site,"bytes":bytes,"count":count})).collect::<Vec<_>>(),
            "memory":self.admission_memory()?}));Ok(())
    }
    #[track_caller]
    pub fn allocate<T: DeviceElement>(&self, len: usize) -> Result<Buffer<T>> {
        ensure!(len > 0, "zero-size allocation");
        let bytes = len
            .checked_mul(size_of::<T>())
            .context("allocation size overflow")?;
        if let Some(arena) = &self.staged_metadata && bytes <= 4096 {
            let offset = arena.extent.get().div_ceil(256) * 256;
            let end = offset.checked_add(bytes).context("metadata arena overflow")?;
            ensure!(end <= arena.device.len(), "metadata staging arena exhausted");
            arena.extent.set(end);
            arena.payload.set(arena.payload.get() + bytes);
            return Ok(Buffer {owner:self.owner.clone(),ptr:arena.device.ptr()+offset as u64,
                len,storage:Storage::StagedMetadata {arena:arena.clone(),offset},_element:PhantomData});
        }
        if let Some(total)=&self.mapped_metadata_bytes && bytes<=4096 {
            let region=self.mapped_host(bytes)?;
            let ptr=region.device;
            total.set(total.get()+bytes);
            return Ok(Buffer {owner:self.owner.clone(),ptr,len,storage:Storage::MappedMetadata(region),_element:PhantomData});
        }
        self.owner.current()?;
        self.check_allocation_floor(bytes)?;
        let mut ptr = 0;
        let result = self.owner.api.check(
            unsafe { (self.owner.api.alloc)(&mut ptr, bytes) },
            "allocate device memory",
        );
        if result.is_err() { log_allocation_failure(&result, "allocate device memory", bytes, std::panic::Location::caller(), self.memory().ok()); }
        result?;
        if bytes>=65536&&let Some(sites)=&self.owner.allocation_sites {
            sites.borrow_mut().insert(ptr,(std::panic::Location::caller(),bytes));
            trace_runtime_allocation("alloc", std::panic::Location::caller(), bytes);
        }
        Ok(Buffer {
            owner: self.owner.clone(),
            ptr,
            len,
            storage: Storage::Owned,
            _element: PhantomData,
        })
    }
    /// Request-owned image buffers may refuse capacity without invalidating
    /// model state. Distributed callers must agree on success before GPU IO.
    #[track_caller]
    pub(crate) fn allocate_input<T: DeviceElement>(&self, len: usize) -> Result<Buffer<T>> {
        let required = len.checked_mul(size_of::<T>())
            .and_then(|n| n.checked_add(self.allocation_reserve_bytes()))
            .context("input allocation size overflow")?;
        self.allocate(len).map_err(|e| input_allocation_error(e, required))
    }
    /// Create the snapshot chunk pool (see ChunkPool). Called once, after the startup warm-up.
    pub(crate) fn create_image_pool(&self, bytes: usize) -> Result<serde_json::Value> {
        ensure!(self.owner.image_pool.borrow().is_none(), "image pool already exists");
        self.owner.current()?;
        let api = &self.owner.api;
        let prop = MemAllocationProp {kind:1,location:MemLocation {kind:1,id:self.owner.device},..MemAllocationProp::default()};
        let mut granularity = 0usize;
        api.check(unsafe{(api.mem_granularity)(&mut granularity,&prop,0)},"query image pool granularity")?;
        ensure!(granularity.is_power_of_two(),"invalid VMM granularity");
        let chunk = (16usize << 20).max(granularity).div_ceil(granularity) * granularity;
        let count = bytes / chunk;
        ensure!(count > 0, "image pool smaller than one chunk");
        let began = std::time::Instant::now();
        let before = self.admission_memory()?;
        let mut handles = Vec::with_capacity(count);
        for _ in 0..count {
            let mut handle = 0u64;
            let created = api.check(unsafe{(api.mem_create)(&mut handle,chunk,&prop,0)},"create image pool chunk");
            if let Err(error) = created {
                for h in handles { unsafe { (api.mem_release)(h); } }
                return Err(error);
            }
            handles.push(handle);
        }
        *self.owner.image_pool.borrow_mut() = Some(Rc::new(ChunkPool {chunk,count,released:Default::default(),free:RefCell::new(handles)}));
        Ok(serde_json::json!({"chunks":count,"chunk_bytes":chunk,"bytes":count*chunk,"granularity":granularity,
            "seconds":began.elapsed().as_secs_f64(),"memory_before":before,"memory_after":self.admission_memory()?}))
    }
    /// The image pool's chunk size (16 MiB rounded up to the VMM granularity), as create_image_pool and
    /// allocate_elastic compute it.
    pub(crate) fn image_pool_chunk_bytes(&self) -> Result<usize> {
        self.owner.current()?;
        let api = &self.owner.api;
        let prop = MemAllocationProp {kind:1,location:MemLocation {kind:1,id:self.owner.device},..MemAllocationProp::default()};
        let mut granularity = 0usize;
        api.check(unsafe{(api.mem_granularity)(&mut granularity,&prop,0)},"query image pool granularity")?;
        ensure!(granularity.is_power_of_two(),"invalid VMM granularity");
        Ok((16usize << 20).max(granularity).div_ceil(granularity) * granularity)
    }
    /// Release the image pool and its memory (Nov 18 2026: the temporary pool that backs elastic serving regions
    /// during the batched graph warm-up, before the real pool exists). Every chunk must be free and no pooled
    /// snapshot may hold the pool; otherwise nothing changes. Returns the chunks released.
    pub(crate) fn release_image_pool(&self) -> Result<usize> {
        let pool = self.owner.image_pool.borrow_mut().take().context("no image pool to release")?;
        let free = pool.free.borrow().len();
        let count = pool.count - pool.released.get();
        if free != count || Rc::strong_count(&pool) != 1 {
            *self.owner.image_pool.borrow_mut() = Some(pool);
            anyhow::bail!("image pool still in use ({free} of {count} chunks free)");
        }
        self.owner.current()?;
        let api = &self.owner.api;
        api.check(unsafe{(api.context_sync)()},"drain before releasing the image pool")?;
        let handles: Vec<u64> = pool.free.borrow_mut().drain(..).collect();
        for h in handles { api.check(unsafe{(api.mem_release)(h)},"release image pool chunk")?; }
        Ok(count)
    }
    pub(crate) fn image_pool_free_bytes(&self) -> Option<usize> {
        self.owner.image_pool.borrow().as_ref().map(|p| p.free.borrow().len() * p.chunk)
    }
    pub(crate) fn image_pool_free_chunks(&self) -> Option<usize> {
        self.owner.image_pool.borrow().as_ref().map(|p| p.free.borrow().len())
    }
    pub(crate) fn image_pool_released_chunks(&self) -> Option<usize> {
        self.owner.image_pool.borrow().as_ref().map(|p| p.released.get())
    }
    pub(crate) fn image_pool_total_bytes(&self) -> Option<usize> {
        self.owner.image_pool.borrow().as_ref().map(|p| (p.count - p.released.get()) * p.chunk)
    }
    /// Return idle physical pages to the OS at an input or bank handoff. Only the
    /// tight rank needs to act. Never touch a mapped page or spend the pages
    /// reserved for the pending snapshot. The lifetime cap prevents repeated
    /// requests from draining the preallocated pool or churning large allocations.
    pub(crate) fn return_idle_image_pages(&self, available: usize, required: usize, keep: usize) -> Result<usize> {
        let Some(pool) = self.owner.image_pool.borrow().clone() else { return Ok(0) };
        let cushion = self.allocation_reserve_bytes().saturating_sub(crate::memory::floor_bytes());
        let mut free = pool.free.borrow_mut();
        let n = idle_pages_to_return(available, required, cushion, pool.chunk, free.len(), keep,
            pool.count, pool.released.get());
        if n > 0 {
            self.owner.current()?;
            for _ in 0..n {
                let handle = *free.last().expect("planned free pool page");
                self.owner.api.check(unsafe { (self.owner.api.mem_release)(handle) }, "return unused image pool page")?;
                free.pop();
                pool.released.set(pool.released.get() + 1);
            }
        }
        Ok(n * pool.chunk)
    }
    /// Emergency fallback inside input-capacity reclamation: bypasses the soft
    /// 64-chunk lifetime cap when a rank would otherwise refuse a request while
    /// holding free chunks in its pool. Never touches mapped chunks.
    pub(crate) fn return_idle_image_pages_emergency(&self, available: usize, required: usize, keep: usize) -> Result<usize> {
        let Some(pool) = self.owner.image_pool.borrow().clone() else { return Ok(0) };
        let cushion = self.allocation_reserve_bytes().saturating_sub(crate::memory::floor_bytes());
        let mut free = pool.free.borrow_mut();
        let n = idle_pages_to_return_emergency(available, required, cushion, pool.chunk, free.len(), keep);
        if n > 0 {
            self.owner.current()?;
            for _ in 0..n {
                let handle = *free.last().expect("planned free pool page");
                self.owner.api.check(unsafe { (self.owner.api.mem_release)(handle) }, "return unused image pool page (emergency)")?;
                free.pop();
                pool.released.set(pool.released.get() + 1);
            }
        }
        Ok(n * pool.chunk)
    }
    /// Re-hydrate previously released image pool chunks while idle and host memory
    /// has comfortably recovered above the reserve + hysteresis floor. Best-effort:
    /// stops cleanly on driver refusal without failing the worker.
    pub(crate) fn rehydrate_image_pool(&self, max_chunks: usize) -> Result<usize> {
        let Some(pool) = self.owner.image_pool.borrow().clone() else { return Ok(0) };
        let released = pool.released.get();
        if released == 0 || max_chunks == 0 { return Ok(0); }
        let reserve = self.allocation_reserve_bytes();
        let cushion = reserve.saturating_sub(crate::memory::floor_bytes());
        let available = self.admission_memory()?.admitted_available;
        let n = pool_chunks_to_rehydrate(available, reserve, cushion, pool.chunk, released, max_chunks);
        if n == 0 { return Ok(0); }
        self.owner.current()?;
        let api = &self.owner.api;
        let prop = MemAllocationProp { kind: 1, location: MemLocation { kind: 1, id: self.owner.device }, ..MemAllocationProp::default() };
        let mut added = 0usize;
        for _ in 0..n {
            let mut handle = 0u64;
            if api.check(unsafe { (api.mem_create)(&mut handle, pool.chunk, &prop, 0) }, "rehydrate image pool chunk").is_err() {
                break;
            }
            pool.free.borrow_mut().push(handle);
            pool.released.set(pool.released.get() - 1);
            added += 1;
        }
        Ok(added * pool.chunk)
    }
    /// Whether a snapshot of `bytes` would come from the pool right now.
    pub(crate) fn image_pool_fits(&self, bytes: usize) -> bool {
        bytes >= POOLED_IMAGE_MIN && self.owner.image_pool.borrow().as_ref()
            .is_some_and(|p| p.free.borrow().len() >= bytes.div_ceil(p.chunk))
    }
    /// A snapshot from the pool, or None (no pool, too small, or not enough free chunks: the caller allocates normally).
    pub(crate) fn allocate_pooled(&self, bytes: usize) -> Result<Option<Buffer<u8>>> {
        if bytes < POOLED_IMAGE_MIN { return Ok(None); }
        let Some(pool) = self.owner.image_pool.borrow().clone() else { return Ok(None) };
        let k = bytes.div_ceil(pool.chunk);
        let handles = {
            let mut free = pool.free.borrow_mut();
            if free.len() < k { return Ok(None); }
            let n = free.len();
            free.split_off(n - k)
        };
        let span = k * pool.chunk;
        // Owned from here on: an error below unmaps what was mapped and returns every chunk.
        let mut a = PooledAllocation {owner:self.owner.clone(),pool:pool.clone(),ptr:0,span,handles,mapped:0};
        self.owner.current()?;
        let api = &self.owner.api;
        api.check(unsafe{(api.address_reserve)(&mut a.ptr,span,pool.chunk,0,0)},"reserve pooled image addresses")?;
        for i in 0..k {
            api.check(unsafe{(api.mem_map)(a.ptr+(i*pool.chunk) as u64,pool.chunk,0,a.handles[i],0)},"map pooled image chunk")?;
            a.mapped += 1;
        }
        let access = MemAccessDesc {location:MemLocation {kind:1,id:self.owner.device},flags:3};
        api.check(unsafe{(api.mem_access)(a.ptr,span,&access,1)},"set pooled image access")?;
        let ptr = a.ptr;
        Ok(Some(Buffer {owner:self.owner.clone(),ptr,len:bytes,storage:Storage::Pooled(a),_element:PhantomData}))
    }
    /// An elastic range of `bytes` (see ElasticAllocation): `resident` bytes (rounded up to a pool chunk) get their own
    /// memory now and are zeroed; the rest stays unmapped until Buffer::elastic_map. The chunk size is the image
    /// pool's (16 MiB rounded to the VMM granularity), so it can be computed before the pool exists.
    pub(crate) fn allocate_elastic(&self, bytes: usize, resident: usize, label: &str) -> Result<Buffer<u8>> {
        ensure!(bytes > 0 && resident <= bytes, "elastic range geometry");
        self.owner.current()?;
        let api = &self.owner.api;
        let prop = MemAllocationProp {kind:1,location:MemLocation {kind:1,id:self.owner.device},..MemAllocationProp::default()};
        let mut granularity = 0usize;
        api.check(unsafe{(api.mem_granularity)(&mut granularity,&prop,0)},"query elastic granularity")?;
        ensure!(granularity.is_power_of_two(),"invalid VMM granularity");
        let chunk = (16usize << 20).max(granularity).div_ceil(granularity) * granularity;
        if let Some(pool) = self.owner.image_pool.borrow().as_ref() { ensure!(pool.chunk == chunk, "elastic chunk differs from the pool's"); }
        let span = bytes.div_ceil(chunk) * chunk;
        let resident = (resident.div_ceil(chunk) * chunk).min(span);
        let chunks = (span - resident) / chunk;
        let mut a = ElasticAllocation {owner:self.owner.clone(),ptr:0,span,chunk,resident:0,resident_handle:0,
            chunks:RefCell::new(vec![None; chunks])};
        api.check(unsafe{(api.address_reserve)(&mut a.ptr,span,chunk,0,0)},"reserve elastic addresses")?;
        if resident > 0 {
            api.check(unsafe{(api.mem_create)(&mut a.resident_handle,resident,&prop,0)},"create elastic resident memory")?;
            api.check(unsafe{(api.mem_map)(a.ptr,resident,0,a.resident_handle,0)},"map elastic resident memory")?;
            a.resident = resident;
            let access = MemAccessDesc {location:MemLocation {kind:1,id:self.owner.device},flags:3};
            api.check(unsafe{(api.mem_access)(a.ptr,resident,&access,1)},"set elastic resident access")?;
            api.check(unsafe{(api.zero)(a.ptr,0,resident)},"zero elastic resident memory")?;
        }
        let ptr = a.ptr;
        // Nov 18: every elastic range, so a fault address can be traced to its cache (fqa3: an MMU read fault at one
        // address on all four ranks once a region conversation passed 16K positions).
        eprintln!("elastic range {}", serde_json::json!({"label":label,"ptr":format!("{ptr:#x}"),"end":format!("{:#x}",ptr+span as u64),
            "span":span,"resident":resident,"chunk":chunk}));
        Ok(Buffer {owner:self.owner.clone(),ptr,len:bytes,storage:Storage::Elastic(Box::new(a)),_element:PhantomData})
    }
    /// Admission for conversation snapshots only: with the pool, its free chunks are where a snapshot goes, so they count
    /// as room. Every other allocation still answers to host memory through admission_memory().
    pub(crate) fn image_admission_memory(&self) -> Result<crate::memory::Available> {
        let mut memory = self.admission_memory()?;
        if let Some(free) = self.image_pool_free_bytes() {
            memory.admitted_available = memory.admitted_available.saturating_add(free);
        }
        Ok(memory)
    }
    fn compressible_properties(&self) -> Result<MemAllocationProp> {
        self.owner.current()?;
        ensure!(self.attribute(102)? != 0 && self.attribute(107)? != 0,
            "device lacks VMM or generic lossless memory compression");
        Ok(MemAllocationProp {kind:1,location:MemLocation {kind:1,id:self.owner.device},
            compression:1,..MemAllocationProp::default()})
    }
    pub(crate) fn compressible_bytes(&self, bytes: usize) -> Result<usize> {
        ensure!(bytes > 0, "zero-size compressible allocation");
        let prop=self.compressible_properties()?;
        let mut granularity=0usize;
        self.owner.api.check(unsafe{(self.owner.api.mem_granularity)(&mut granularity,&prop,0)},
            "query compressible allocation granularity")?;
        ensure!(granularity.is_power_of_two(),"invalid VMM granularity");
        let rounded=bytes.checked_add(granularity-1).context("compressible size overflow")?;
        Ok(rounded/granularity*granularity)
    }
    #[track_caller]
    pub(crate) fn allocate_compressible<T: DeviceElement>(&self, len: usize) -> Result<Buffer<T>> {
        let bytes=len.checked_mul(size_of::<T>()).context("compressible buffer size overflow")?;
        let mapped_bytes=self.compressible_bytes(bytes)?;
        let prop=self.compressible_properties()?;
        let api=&self.owner.api;
        let mut allocation=VmmAllocation {owner:self.owner.clone(),ptr:0,bytes:mapped_bytes,handle:0,mapped:false};
        let result=api.check(unsafe{(api.mem_create)(&mut allocation.handle,mapped_bytes,&prop,0)},"create compressible memory");
        if result.is_err() { log_allocation_failure(&result, "create compressible memory", mapped_bytes, std::panic::Location::caller(), self.memory().ok()); }
        result?;
        let mut obtained=MemAllocationProp::default();
        api.check(unsafe{(api.mem_properties)(&mut obtained,allocation.handle)},"check compressible memory properties")?;
        ensure!(obtained.compression==1 && obtained.location.kind==1 && obtained.location.id==self.owner.device,
            "requested lossless compression/device was not obtained");
        api.check(unsafe{(api.address_reserve)(&mut allocation.ptr,mapped_bytes,0,0,0)},"reserve compressible address range")?;
        api.check(unsafe{(api.mem_map)(allocation.ptr,mapped_bytes,0,allocation.handle,0)},"map compressible memory")?;
        allocation.mapped=true;
        let access=MemAccessDesc {location:prop.location,flags:3};
        api.check(unsafe{(api.mem_access)(allocation.ptr,mapped_bytes,&access,1)},"set compressible memory access")?;
        Ok(Buffer {owner:self.owner.clone(),ptr:allocation.ptr,len,storage:Storage::Vmm(allocation),_element:PhantomData})
    }
    /// Exclusive writable import: the unchanged loader uploads and prepares
    /// every canonical byte before use. The parent never reads or writes while
    /// this child is alive. Both share one memory cgroup; startup also checks
    /// physical host headroom because GB10 GPU pages are not fully cgroup-charged.
    pub(crate) fn import_compressible(&self, fd: OwnedFd, bytes: usize, mapped_bytes: usize)
        -> Result<Buffer<u8>> {
        ensure!(mapped_bytes==self.compressible_bytes(bytes)?,"external weight mapping geometry");
        let api=&self.owner.api;
        let mut allocation=VmmAllocation {owner:self.owner.clone(),ptr:0,bytes:mapped_bytes,handle:0,mapped:false};
        api.check(unsafe{(api.mem_import)(&mut allocation.handle,fd.as_raw_fd() as isize as *mut c_void,1)},
            "import compressible target weights")?;
        let mut obtained=MemAllocationProp::default();
        api.check(unsafe{(api.mem_properties)(&mut obtained,allocation.handle)},"check imported weight properties")?;
        // GB10's driver reports DEVICE on creation but HOST_NUMA after FD
        // import. Accept only this integrated GPU's closest host NUMA node,
        // or the ordinary device location; never equate NUMA IDs and ordinals.
        let local_location=match obtained.location.kind {
            1=>obtained.location.id==self.owner.device,
            3=>self.attribute(18)?==1 && obtained.location.id>=0
                && obtained.location.id==self.attribute(134)?,
            _=>false,
        };
        ensure!(obtained.kind==1 && obtained.compression==1 && local_location,
            "invalid imported weights: type={} compression={} location={}:{} device={}",
            obtained.kind,obtained.compression,obtained.location.kind,
            obtained.location.id,self.owner.device);
        api.check(unsafe{(api.address_reserve)(&mut allocation.ptr,mapped_bytes,0,0,0)},"reserve imported weight addresses")?;
        api.check(unsafe{(api.mem_map)(allocation.ptr,mapped_bytes,0,allocation.handle,0)},"map imported weights")?;
        allocation.mapped=true;
        // Access is granted to the consuming GPU even when the physical
        // allocation property is reported as host NUMA memory.
        let access=MemAccessDesc {location:MemLocation {kind:1,id:self.owner.device},flags:3};
        api.check(unsafe{(api.mem_access)(allocation.ptr,mapped_bytes,&access,1)},"set exclusive imported weight access")?;
        Ok(Buffer {owner:self.owner.clone(),ptr:allocation.ptr,len:bytes,storage:Storage::Vmm(allocation),_element:PhantomData})
    }
    pub fn stream(&self) -> Result<Stream> {
        self.owner.current()?;
        let mut raw = ptr::null_mut();
        self.owner.api.check(
            unsafe { (self.owner.api.stream_create)(&mut raw, 1) },
            "create nonblocking stream",
        )?;
        let mut stream = Stream {
            owner: self.owner.clone(),
            raw,
            identity: Rc::new(()),
            graph_stream_bound: std::cell::Cell::new(false),
            upload_fence: ptr::null_mut(),
            blocking_fence: std::cell::Cell::new(ptr::null_mut()),
            polling_fence: std::cell::Cell::new(ptr::null_mut()),
        };
        self.owner.api.check(
            unsafe { (self.owner.api.event_create)(&mut stream.upload_fence, 2) },
            "create upload completion fence",
        )?;
        Ok(stream)
    }
    /// One reusable shared/hint stream for this context's single parent stream.
    /// A weak cache avoids an owner/stream reference cycle. Mutable graph
    /// workspaces still belong to one serialized parent, including its forks.
    pub(crate) fn auxiliary_stream(&self, parent: &Stream) -> Result<Rc<Stream>> {
        self.cached_auxiliary(parent, &self.owner.auxiliary)
    }
    /// mHC coefficients have an independent channel, so an immutable weight
    /// hint cannot serialize a coefficient needed on the main critical path.
    /// Auxiliary channel count is bounded independently of layer count.
    pub(crate) fn coefficient_stream(&self, parent: &Stream) -> Result<Rc<Stream>> {
        self.cached_auxiliary(parent, &self.owner.coefficients)
    }
    /// The FFN's L2 hint queue windows (the paced router/shared-expert window and the routed-expert window).
    /// The auxiliary stream also carries the shared expert's branch, which a long hint there would hold back
    /// into the experts' kernels; these hints run beside it instead.
    pub(crate) fn hint_stream(&self, parent: &Stream) -> Result<Rc<Stream>> {
        self.cached_auxiliary(parent, &self.owner.hints)
    }
    /// Independent attention Q work; every layer joins before attention reads
    /// its output. Shared causal cache writes remain on the original parent.
    pub(crate) fn query_stream(&self, parent: &Stream) -> Result<Rc<Stream>> {
        self.cached_auxiliary(parent, &self.owner.queries)
    }
    fn cached_auxiliary(
        &self,
        parent: &Stream,
        cache: &RefCell<Option<(usize, Weak<Stream>)>>,
    ) -> Result<Rc<Stream>> {
        ensure!(self.owns_stream(parent), "auxiliary parent context");
        for slot in [&self.owner.auxiliary, &self.owner.coefficients, &self.owner.queries, &self.owner.hints] {
            if let Some((bound_parent, weak)) = slot.borrow().as_ref()
                && weak.strong_count() > 0
            {
                ensure!(
                    *bound_parent == parent.raw() as usize,
                    "auxiliary channels require one ordered parent"
                );
            }
        }
        if let Some((bound_parent, weak)) = cache.borrow().as_ref()
            && let Some(stream) = weak.upgrade()
        {
            ensure!(
                *bound_parent == parent.raw() as usize,
                "auxiliary stream requires one ordered parent"
            );
            return Ok(stream);
        }
        let stream = Rc::new(self.stream()?);
        *cache.borrow_mut() = Some((parent.raw() as usize, Rc::downgrade(&stream)));
        Ok(stream)
    }
    pub(crate) fn fence_event(&self) -> Result<Event> {
        self.owner.current()?;
        let mut raw = ptr::null_mut();
        self.owner.api.check(
            unsafe { (self.owner.api.event_create)(&mut raw, 2) },
            "create dependency event",
        )?;
        Ok(Event {
            owner: self.owner.clone(),
            raw,
        })
    }
    pub(crate) fn share_immutable_core(&self,artifact:&[u8])->Result<()> {
        use sha2::Digest;
        ensure!(!artifact.is_empty(),"empty shared core");
        let digest:[u8;32]=sha2::Sha256::digest(artifact).into();
        let mut entry=self.owner.shared_core.borrow_mut();
        if let Some((old,bytes,_))=entry.as_ref() {
            ensure!(*old==digest&&*bytes==artifact.len(),"shared core identity changed");
        }else{*entry=Some((digest,artifact.len(),Weak::new()));}
        Ok(())
    }
    pub fn module(&self, artifact: &[u8]) -> Result<Module> {
        ensure!(!artifact.is_empty(), "empty kernel artifact");
        self.owner.current()?;
        let shared={
            use sha2::Digest;
            let entry=self.owner.shared_core.borrow();
            entry.as_ref().is_some_and(|(digest,bytes,_)|*bytes==artifact.len()
                &&sha2::Sha256::digest(artifact).as_slice()==digest)
        };
        if shared&&let Some(inner)=self.owner.shared_core.borrow().as_ref().unwrap().2.upgrade() {
            return Ok(Module{inner});
        }
        let mut data = artifact.to_vec();
        data.push(0); // PTX needs NUL; cubin ignores trailing byte.
        self.check_allocation_floor(16*1024*1024)?;
        let mut raw = ptr::null_mut();
        self.owner.api.check(
            unsafe { (self.owner.api.module_load)(&mut raw, data.as_ptr().cast()) },
            "load CUDA module",
        )?;
        let inner=Rc::new(ModuleOwner {
                owner: self.owner.clone(),
                raw,
            });
        if shared {self.owner.shared_core.borrow_mut().as_mut().unwrap().2=Rc::downgrade(&inner);}
        Ok(Module{inner})
    }
    pub fn event(&self) -> Result<Event> {
        let mut raw = ptr::null_mut();
        self.owner.api.check(
            unsafe { (self.owner.api.event_create)(&mut raw, 0) },
            "create timing event",
        )?;
        Ok(Event {
            owner: self.owner.clone(),
            raw,
        })
    }
}

pub struct MappedHost {
    owner: Rc<ContextOwner>,
    host: *mut c_void,
    device: u64,
    bytes: usize,
    vmm_handle: u64,
    vmm_size: usize,
}
impl MappedHost {
    pub fn host_ptr(&self) -> *mut c_void {
        self.host
    }
    pub fn device_ptr(&self) -> u64 {
        self.device
    }
    pub fn len(&self) -> usize {
        self.bytes
    }
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
    /// Read an aligned coherent status word written by the GPU/native proxy.
    pub fn status_word(&self, offset: usize) -> Result<u32> {
        ensure!(
            offset % 4 == 0 && offset.checked_add(4).is_some_and(|n| n <= self.bytes),
            "host status word outside allocation"
        );
        Ok(unsafe { std::ptr::read_volatile(self.host.cast::<u8>().add(offset).cast::<u32>()) })
    }
}
impl Drop for MappedHost {
    fn drop(&mut self) {
        let _ = self.owner.current();
        unsafe {
            if self.vmm_handle != 0 {
                let _ = (self.owner.api.mem_unmap)(self.device, self.vmm_size);
                let _ = (self.owner.api.address_free)(self.device, self.vmm_size);
                let _ = (self.owner.api.mem_release)(self.vmm_handle);
            } else {
                (self.owner.api.host_free)(self.host);
            }
        }
    }
}

/// Safety: every bit pattern is valid and the representation is a fixed-width
/// CUDA-compatible scalar. No pointers or host references may implement this.
pub unsafe trait DeviceElement: Copy + Default {}
unsafe impl DeviceElement for u8 {}
unsafe impl DeviceElement for u16 {}
// half::bf16 is repr(transparent) over u16; all 65,536 bit patterns are valid.
unsafe impl DeviceElement for half::bf16 {}
unsafe impl DeviceElement for i32 {}
unsafe impl DeviceElement for i64 {}
unsafe impl DeviceElement for u32 {}
unsafe impl DeviceElement for u64 {}
unsafe impl DeviceElement for f32 {}
unsafe impl DeviceElement for f64 {}
pub struct Buffer<T: DeviceElement> {
    owner: Rc<ContextOwner>,
    ptr: u64,
    len: usize,
    storage: Storage,
    _element: PhantomData<T>,
}
enum Storage {
    Owned,
    Vmm(#[allow(dead_code)] VmmAllocation),
    // A conversation snapshot mapped from the startup chunk pool; dropping returns the chunks.
    Pooled(#[allow(dead_code)] PooledAllocation),
    // A reserved range with a resident prefix; later chunks come from the image pool while mapped (ElasticAllocation).
    // Boxed: inline it was the largest variant and grew every Buffer from 88 to 112 bytes, which overflowed the main
    // thread's stack while the wide targets were built (s1604 small probe, Nov 18 2026).
    Elastic(Box<ElasticAllocation>),
    // Created only by the scoped integer-metadata allocator above.
    MappedMetadata(MappedHost),
    StagedMetadata { arena: Rc<MetadataStaging>, offset: usize },
    // A view into a shared row arena; the arena owns the bytes.
    Arena(#[allow(dead_code)] Rc<RowArena>),
    // A borrowed row range of another buffer (see Buffer::view).
    View,
}
struct VmmAllocation {
    owner: Rc<ContextOwner>,
    ptr: u64,
    bytes: usize,
    handle: u64,
    mapped: bool,
}
impl Drop for VmmAllocation {
    fn drop(&mut self) {
        // Unlike cuMemFree, unmapping does not implicitly join GPU users.
        // Draining here also makes partial-constructor/error cleanup safe.
        if self.owner.current().is_err() { return; }
        let api=&self.owner.api;
        unsafe {
            if self.mapped { (api.context_sync)(); (api.mem_unmap)(self.ptr,self.bytes); }
            if self.ptr != 0 { (api.address_free)(self.ptr,self.bytes); }
            if self.handle != 0 { (api.mem_release)(self.handle); }
        }
    }
}
/// Pinned CPU writes are coalesced into one transfer to ordinary device memory.
/// The device allocation drops before the pinned source, including on errors.
pub(crate) struct MetadataStaging {
    device: Buffer<u8>,
    host: MappedHost,
    extent: std::cell::Cell<usize>,
    payload: std::cell::Cell<usize>,
}
impl MetadataStaging {
    pub(crate) fn bytes(&self) -> usize { self.payload.get() }
    pub(crate) fn transfer_bytes(&self) -> usize { self.extent.get() }
    /// Safety: no host writes may race this copy; all GPU consumers are
    /// ordered after it. A synchronous initial flush uses None.
    pub(crate) unsafe fn flush(&self, stream: Option<&Stream>) -> Result<()> {
        let owner = &self.device.owner;
        owner.current()?;
        let status = if let Some(stream) = stream {
            ensure!(Rc::ptr_eq(owner, &stream.owner), "metadata copy stream context");
            unsafe {(owner.api.htod_async)(self.device.ptr(),self.host.host,self.extent.get(),stream.raw())}
        } else {
            unsafe {(owner.api.htod)(self.device.ptr(),self.host.host,self.extent.get())}
        };
        owner.api.check(status, "coalesced metadata upload")
    }
}
/// The audited kinds of chunk scratch (Cuda::allocate_chunk_scratch), as mask bits.
#[derive(Clone, Copy)]
pub(crate) enum ChunkScratch {
    EngramRows = 1,
    MoeOutput = 2,
    WoaGrouped = 4,
    CompressorLatent = 8,
    PairPool = 16,
    QueryRows = 32,
    TopkIndices = 64,
    CandidateIds = 128,
}
/// Layer-parity transient storage for wide prefill profiles.
pub struct RowArena {
    storage: Buffer<u8>,
    next: std::cell::Cell<usize>,
    high_water: std::cell::Cell<usize>,
    /// The largest extent carved since reset_profile_high (one profile's construction).
    profile_high: std::cell::Cell<usize>,
}
impl RowArena {
    pub fn bytes(&self) -> usize {
        self.storage.len
    }
    pub fn high_water(&self) -> usize {
        self.high_water.get()
    }
    /// Restart carving at the arena's beginning (once per profile's construction).
    pub(crate) fn rewind(&self) {
        self.next.set(0);
    }
    pub(crate) fn reset_profile_high(&self) { self.profile_high.set(0); }
    pub(crate) fn profile_high(&self) -> usize { self.profile_high.get() }
    /// The prefix that always has memory (the whole arena unless it is elastic).
    pub(crate) fn resident_bytes(&self) -> usize { self.storage.elastic_resident().unwrap_or(self.storage.len) }
}
impl Buffer<u8> {
    pub fn upload_part(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        ensure!(
            offset
                .checked_add(data.len())
                .is_some_and(|n| n <= self.len),
            "partial upload outside buffer"
        );
        if let Storage::StagedMetadata {arena,offset:base} = &self.storage {
            unsafe {std::ptr::copy_nonoverlapping(data.as_ptr(),
                arena.host.host.cast::<u8>().add(base + offset),data.len());}
            return Ok(());
        }
        self.owner.current()?;
        self.owner.api.check(
            unsafe {
                (self.owner.api.htod)(self.ptr + offset as u64, data.as_ptr().cast(), data.len())
            },
            "partial upload",
        )
    }
    pub fn download_part(&self, offset: usize, bytes: usize) -> Result<Vec<u8>> {
        ensure!(offset.checked_add(bytes).is_some_and(|n|n<=self.len),
            "partial download outside buffer");
        let mut data = vec![0u8; bytes];
        self.download_part_into(offset, &mut data)?;
        Ok(data)
    }
    pub(crate) fn download_part_into(&self, offset: usize, data: &mut[u8]) -> Result<()> {
        ensure!(
            offset.checked_add(data.len()).is_some_and(|n| n <= self.len),
            "partial download outside buffer"
        );
        self.owner.current()?;
        self.owner.api.check(
            unsafe {
                (self.owner.api.dtoh)(data.as_mut_ptr().cast(), self.ptr + offset as u64, data.len())
            },
            "partial download",
        )
    }
}
impl<T: DeviceElement> Buffer<T> {
    /// Non-owning view of elements [offset, offset+len), e.g. one rank's row
    /// shard of a sequence-parallel activation.
    /// Safety: the view must not outlive `self`, and the caller orders every
    /// GPU access to the aliased range on the owning stream.
    pub unsafe fn view(&self, offset: usize, len: usize) -> Result<Buffer<T>> {
        ensure!(
            len > 0 && offset.checked_add(len).is_some_and(|end| end <= self.len),
            "view outside its base buffer"
        );
        Ok(Buffer {
            owner: self.owner.clone(),
            ptr: self.ptr + (offset * size_of::<T>()) as u64,
            len,
            storage: Storage::View,
            _element: PhantomData,
        })
    }
    pub fn ptr(&self) -> u64 {
        self.ptr
    }
    pub fn len(&self) -> usize {
        self.len
    }
    /// Whether this is an elastic range (Cuda::allocate_elastic): only its resident prefix and mapped spans exist.
    pub(crate) fn is_elastic(&self) -> bool { matches!(self.storage, Storage::Elastic(_)) }

    pub(crate) fn owned_pool_chunks(&self) -> usize {
        match &self.storage { Storage::Pooled(allocation) => allocation.handles.len(), _ => 0 }
    }
    /// An elastic range's resident prefix (bytes, a whole number of pool chunks), None for other storage.
    pub(crate) fn elastic_resident(&self) -> Option<usize> {
        match &self.storage { Storage::Elastic(e) => Some(e.resident.min(self.len)), _ => None }
    }
    /// Pool chunks elastic_map(offset, len) would take now (0 for a span already mapped or resident).
    pub(crate) fn elastic_missing(&self, offset: usize, len: usize) -> Result<usize> {
        let Storage::Elastic(e) = &self.storage else { anyhow::bail!("not an elastic range") };
        e.missing(offset, len)
    }
    /// The resident prefix and every mapped chunk, as merged (address, bytes) extents.
    pub(crate) fn elastic_mapped_extents(&self) -> Result<Vec<(u64, usize)>> {
        let Storage::Elastic(e) = &self.storage else { anyhow::bail!("not an elastic range") };
        Ok(e.mapped_extents())
    }
    /// Map a span of an elastic range from the image pool's free chunks (byte offsets). Ok(false), changing nothing, when
    /// the pool has too few free chunks. New chunks are zeroed; every chunk of the span gains a reference.
    pub(crate) fn elastic_map(&self, offset: usize, len: usize) -> Result<bool> {
        let Storage::Elastic(e) = &self.storage else { anyhow::bail!("not an elastic range") };
        self.owner.current()?;
        let range = e.chunk_range(offset, len)?;
        let base = e.resident / e.chunk;
        let pool = self.owner.image_pool.borrow().clone().context("an elastic map needs the image pool")?;
        ensure!(pool.chunk == e.chunk, "elastic chunk differs from the pool's");
        let missing: Vec<usize> = { let c = e.chunks.borrow(); range.clone().filter(|&i| c[i - base].is_none()).collect() };
        let handles = {
            let mut free = pool.free.borrow_mut();
            if free.len() < missing.len() { return Ok(false); }
            let n = free.len();
            free.split_off(n - missing.len())
        };
        let api = &self.owner.api;
        let access = MemAccessDesc {location:MemLocation {kind:1,id:self.owner.device},flags:3};
        let mut mapped: Vec<(usize, u64)> = Vec::new();
        let mut result = Ok(());
        for (&i, &handle) in missing.iter().zip(&handles) {
            let at = e.ptr + (i * e.chunk) as u64;
            result = api.check(unsafe{(api.mem_map)(at,e.chunk,0,handle,0)},"map elastic chunk");
            if result.is_err() { break; }
            mapped.push((i, handle));
            result = api.check(unsafe{(api.mem_access)(at,e.chunk,&access,1)},"set elastic chunk access")
                .and_then(|_| api.check(unsafe{(api.zero)(at,0,e.chunk)},"zero elastic chunk"));
            if result.is_err() { break; }
        }
        if let Err(error) = result {
            unsafe { for &(i, _) in &mapped { (api.mem_unmap)(e.ptr + (i * e.chunk) as u64, e.chunk); } }
            pool.free.borrow_mut().extend(handles);
            return Err(error);
        }
        // The zeroing ran on the legacy stream, unordered with the model's streams: finish it before anything is
        // copied into the new chunks (maps are rare: a conversation moving into a serving region).
        if !mapped.is_empty() { api.check(unsafe{(api.context_sync)()},"finish elastic zeroing")?; }
        let mut chunks = e.chunks.borrow_mut();
        for &(i, handle) in &mapped { chunks[i - base] = Some((handle, 0)); }
        if elastic_trace() {
            eprintln!("elastic map {}", serde_json::json!({"range":format!("{:#x}",e.ptr),"from":format!("{:#x}",e.ptr+offset as u64),
                "to":format!("{:#x}",e.ptr+(offset+len) as u64),"new_chunks":mapped.len()}));
        }
        for i in range { if let Some((_, references)) = chunks[i - base].as_mut() { *references += 1; } }
        Ok(true)
    }
    /// Drop one reference from every chunk of a span mapped with elastic_map; chunks left without references are
    /// unmapped (after draining the context) and return to the pool.
    pub(crate) fn elastic_unmap(&self, offset: usize, len: usize) -> Result<()> {
        let Storage::Elastic(e) = &self.storage else { anyhow::bail!("not an elastic range") };
        if elastic_trace() {
            eprintln!("elastic unmap {}", serde_json::json!({"range":format!("{:#x}",e.ptr),"from":format!("{:#x}",e.ptr+offset as u64),
                "to":format!("{:#x}",e.ptr+(offset+len) as u64)}));
        }
        let range = e.chunk_range(offset, len)?;
        let base = e.resident / e.chunk;
        let release: Vec<(usize, u64)> = {
            let mut chunks = e.chunks.borrow_mut();
            ensure!(range.clone().all(|i| chunks[i - base].is_some_and(|(_, r)| r > 0)), "elastic unmap of an unmapped span");
            let mut out = Vec::new();
            for i in range {
                let slot = &mut chunks[i - base];
                let (handle, references) = slot.as_mut().unwrap();
                *references -= 1;
                if *references == 0 { out.push((i, *handle)); *slot = None; }
            }
            out
        };
        if release.is_empty() { return Ok(()); }
        self.owner.current()?;
        let api = &self.owner.api;
        api.check(unsafe{(api.context_sync)()},"drain before an elastic unmap")?;
        for &(i, _) in &release {
            api.check(unsafe{(api.mem_unmap)(e.ptr + (i * e.chunk) as u64, e.chunk)},"unmap elastic chunk")?;
        }
        if let Some(pool) = self.owner.image_pool.borrow().as_ref() {
            pool.free.borrow_mut().extend(release.iter().map(|&(_, handle)| handle));
        }
        Ok(())
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn bytes(&self) -> usize {
        self.len * size_of::<T>()
    }
    pub fn upload(&mut self, data: &[T]) -> Result<()> {
        ensure!(
            data.len() == self.len,
            "upload length differs from allocation"
        );
        ensure!(!matches!(self.storage, Storage::Elastic(_)), "whole-range upload into an elastic range");
        if let Storage::StagedMetadata {arena,offset} = &self.storage {
            // The scoped allocator's owner orders writes and its explicit flush.
            unsafe {std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(),
                arena.host.host.cast::<u8>().add(*offset),self.bytes());}
            return Ok(());
        }
        if let Storage::MappedMetadata(region)=&self.storage {
            // The scoped allocator's caller has completed prior GPU readers.
            // Keep ordinary checked metadata setters, changing only transport.
            unsafe {std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(),region.host.cast::<u8>(),self.bytes());}
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            return Ok(());
        }
        self.owner.current()?;
        self.owner.api.check(
            unsafe { (self.owner.api.htod)(self.ptr, data.as_ptr().cast(), self.bytes()) },
            "upload buffer",
        )
    }
    pub fn download(&self) -> Result<Vec<T>> {
        ensure!(!matches!(self.storage, Storage::Elastic(_)), "whole-range download of an elastic range");
        self.owner.current()?;
        let mut v = vec![T::default(); self.len];
        self.owner.api.check(
            unsafe { (self.owner.api.dtoh)(v.as_mut_ptr().cast(), self.ptr, self.bytes()) },
            "download buffer",
        )?;
        Ok(v)
    }
    pub fn zero(&mut self) -> Result<()> {
        if let Storage::Elastic(e) = &self.storage {
            self.owner.current()?;
            for (at, bytes) in e.mapped_extents() {
                self.owner.api.check(unsafe { (self.owner.api.zero)(at, 0, bytes) }, "zero elastic extent")?;
            }
            return Ok(());
        }
        if let Storage::StagedMetadata {arena,offset} = &self.storage {
            unsafe {std::ptr::write_bytes(arena.host.host.cast::<u8>().add(*offset),0,self.bytes());}
            return Ok(());
        }
        self.owner.current()?;
        self.owner.api.check(
            unsafe { (self.owner.api.zero)(self.ptr, 0, self.bytes()) },
            "zero buffer",
        )
    }
}
impl<T: DeviceElement> Drop for Buffer<T> {
    fn drop(&mut self) {
        if !matches!(self.storage, Storage::Owned) {
            return;
        }
        if let Some(sites)=&self.owner.allocation_sites {
            if let Some((site,bytes))=sites.borrow_mut().remove(&self.ptr) {trace_runtime_allocation("free",site,bytes);}
        }
        let _ = self.owner.current();
        unsafe {
            (self.owner.api.free)(self.ptr);
        }
    }
}

pub struct Stream {
    owner: Rc<ContextOwner>,
    raw: Handle,
    // A retained identity token prevents scratch caches from confusing a new
    // stream with a destroyed stream whose CUDA handle address was recycled.
    identity: Rc<()>,
    graph_stream_bound: std::cell::Cell<bool>,
    upload_fence: Handle,
    blocking_fence: std::cell::Cell<Handle>,
    polling_fence: std::cell::Cell<Handle>,
}
impl Stream {
    pub(crate) fn identity_token(&self) -> Rc<()> { self.identity.clone() }
    pub(crate) fn identity_key(&self) -> usize { Rc::as_ptr(&self.identity) as usize }
    pub(crate) fn bind_captured_scratch(&self) { self.graph_stream_bound.set(true); }
    /// Join prior legacy-default-stream uploads/zeroes before nonblocking work.
    /// Pageable HtoD APIs may return after staging, before final DMA completes.
    /// Call at the host preparation boundary, outside capture; this queues a
    /// device dependency and does not impose a host-wide synchronization.
    /// CUDA Driver API "API Synchronization behavior", synchronous memcpy1.
    pub fn join_default(&self) -> Result<()> {
        self.owner.current()?;
        let mut capture = 0;
        self.owner.api.check(
            unsafe { (self.owner.api.stream_capture_status)(self.raw, &mut capture) },
            "query upload fence capture status",
        )?;
        ensure!(
            capture == 0,
            "default-stream upload fence must precede graph capture"
        );
        self.owner.api.check(
            unsafe { (self.owner.api.event_record)(self.upload_fence, ptr::null_mut()) },
            "record default-stream upload completion",
        )?;
        self.owner.api.check(
            unsafe { (self.owner.api.stream_wait_event)(self.raw, self.upload_fence, 0) },
            "order nonblocking stream after uploads",
        )
    }
    pub fn raw(&self) -> Handle {
        self.raw
    }
    pub fn synchronize(&self) -> Result<()> {
        self.owner.api.check(
            unsafe { (self.owner.api.stream_sync)(self.raw) },
            "synchronize stream",
        )
    }
    /// Host-only completion wait for long encoder chunks. The event follows
    /// the same stream work as synchronize(); it changes no device dependency.
    /// Keep decode on its existing wait, avoiding the measured ~1ms wake cost.
    pub(crate) fn synchronize_blocking(&self) -> Result<()> {
        self.owner.current()?;
        let mut capture = 0;
        self.owner.api.check(unsafe {
            (self.owner.api.stream_capture_status)(self.raw, &mut capture)
        }, "query blocking fence capture status")?;
        ensure!(capture == 0, "blocking completion fence cannot run during capture");
        let mut fence = self.blocking_fence.get();
        if fence.is_null() {
            // CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING.
            self.owner.api.check(unsafe { (self.owner.api.event_create)(&mut fence, 3) },
                "create blocking completion fence")?;
            self.blocking_fence.set(fence);
        }
        self.owner.api.check(unsafe { (self.owner.api.event_record)(fence, self.raw) },
            "record blocking completion fence")?;
        self.owner.api.check(unsafe { (self.owner.api.event_sync)(fence) },
            "wait blocking completion fence")
    }
    /// Sleep between nonblocking event queries, avoiding the driver's blocking
    /// wake latency. Success retains the CUDA completion/UMA visibility contract:
    /// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html
    /// Stream/context/thread ownership and capture rejection match other fences.
    pub(crate) fn synchronize_polling(&self,microseconds:u64)->Result<()> {
        // 1 = spin: the thread stays awake between queries (CANADA_SCRATCH_VERIFY_POLL_US=1). A 25 us sleep is
        // ~80 us with the default timer slack, and a sleeping thread's core may drop into an idle state.
        ensure!([1,25,50].contains(&microseconds),"bounded completion polling interval");
        self.owner.current()?;
        let mut capture=0;
        self.owner.api.check(unsafe{(self.owner.api.stream_capture_status)(self.raw,&mut capture)},
            "query polling fence capture status")?;
        ensure!(capture==0,"polling completion fence cannot run during capture");
        let mut fence=self.polling_fence.get();
        if fence.is_null() {
            self.owner.api.check(unsafe{(self.owner.api.event_create)(&mut fence,2)},
                "create polling completion fence")?;
            self.polling_fence.set(fence);
        }
        self.owner.api.check(unsafe{(self.owner.api.event_record)(fence,self.raw)},
            "record polling completion fence")?;
        loop {
            let code=unsafe{(self.owner.api.event_query)(fence)};
            if code==0{return Ok(())}
            if code!=600{return self.owner.api.check(code,"query completion fence")}
            if microseconds==1 {std::hint::spin_loop();}
            else {std::thread::sleep(std::time::Duration::from_micros(microseconds));}
        }
    }
    /// Apply grid rules to the captured graph's kernel nodes before it is instantiated; the nodes matched.
    fn regrid(&self, graph: Handle, rules: &[Option<Regrid>; 3]) -> Result<usize> {
        // CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT (cuda.h).
        const CARVEOUT: c_int = 9;
        let edit = graph_edit()?;
        let api = &self.owner.api;
        let mut count = 0usize;
        api.check(unsafe { (api.graph_nodes)(graph, ptr::null_mut(), &mut count) }, "count graph nodes")?;
        let mut nodes: Vec<Handle> = vec![ptr::null_mut(); count];
        api.check(unsafe { (api.graph_nodes)(graph, nodes.as_mut_ptr(), &mut count) }, "list graph nodes")?;
        let mut changed = 0;
        for &node in &nodes[..count.min(nodes.len())] {
            let mut kind: c_int = -1;
            api.check(unsafe { (edit.node_type)(node, &mut kind) }, "graph node type")?;
            if kind != 0 { continue; }   // CU_GRAPH_NODE_TYPE_KERNEL
            let mut p = std::mem::MaybeUninit::<KernelNodeParams>::zeroed();
            api.check(unsafe { (edit.get)(node, p.as_mut_ptr()) }, "kernel node parameters")?;
            let mut p = unsafe { p.assume_init() };
            if p.func.is_null() || !rules.iter().flatten().any(|rule| p.grid == [1, 1, rule.from]) { continue; }
            let mut name: *const c_char = ptr::null();
            api.check(unsafe { (edit.name)(&mut name, p.func) }, "kernel node function name")?;
            if name.is_null() { continue; }
            let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
            let Some(rule) = rules.iter().flatten().find(|rule| p.grid == [1, 1, rule.from] && name.contains(rule.name))
                else { continue };
            if let Some(percent) = rule.carveout {
                api.check(unsafe { (edit.attribute)(p.func, CARVEOUT, percent) }, "set kernel shared-memory carve-out")?;
            }
            if rule.to != rule.from {
                p.grid[2] = rule.to;
                api.check(unsafe { (edit.set)(node, &p) }, "set kernel node grid")?;
            }
            changed += 1;
        }
        Ok(changed)
    }
    /// Replace the captured graph's kernel nodes named by `swaps` (see NodeSwap); the nodes replaced, which must be
    /// one a record.
    fn swap_nodes(&self, graph: Handle, swaps: &[NodeSwap]) -> Result<usize> {
        // CU_LAUNCH_ATTRIBUTE_COOPERATIVE (cuda.h).
        const COOPERATIVE: c_int = 2;
        let edit = graph_edit()?;
        let info = edit.param_info.context("the CUDA driver lacks cuFuncGetParamInfo")?;
        let api = &self.owner.api;
        let mut count = 0usize;
        api.check(unsafe { (api.graph_nodes)(graph, ptr::null_mut(), &mut count) }, "count graph nodes")?;
        let mut nodes: Vec<Handle> = vec![ptr::null_mut(); count];
        api.check(unsafe { (api.graph_nodes)(graph, nodes.as_mut_ptr(), &mut count) }, "list graph nodes")?;
        let mut changed = 0;
        for &node in &nodes[..count.min(nodes.len())] {
            let mut kind: c_int = -1;
            api.check(unsafe { (edit.node_type)(node, &mut kind) }, "graph node type")?;
            if kind != 0 { continue; }   // CU_GRAPH_NODE_TYPE_KERNEL
            let mut p = std::mem::MaybeUninit::<KernelNodeParams>::zeroed();
            api.check(unsafe { (edit.get)(node, p.as_mut_ptr()) }, "kernel node parameters")?;
            let p = unsafe { p.assume_init() };
            if p.func.is_null() || p.params.is_null() { continue; }
            let mut name: *const c_char = ptr::null();
            api.check(unsafe { (edit.name)(&mut name, p.func) }, "kernel node function name")?;
            if name.is_null() { continue; }
            let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
            if !swaps.iter().any(|swap| name.contains(swap.name)) { continue; }
            // The node's own arguments, as 64-bit words (pointers are passed whole, alone or inside a tensor record).
            let mut values = Vec::new();
            for index in 0..1024usize {
                let (mut offset, mut size) = (0usize, 0usize);
                if unsafe { (info)(p.func, index, &mut offset, &mut size) } != 0 { break; }
                let argument = unsafe { *p.params.add(index) };
                if argument.is_null() { continue; }
                for word in 0..size / 8 { values.push(unsafe { argument.cast::<u64>().add(word).read_unaligned() }); }
            }
            let mut matching = swaps.iter().filter(|swap| name.contains(swap.name)
                && swap.markers.iter().all(|marker| values.contains(marker)));
            // A launch of the same kernel that asked for no replacement keeps its node.
            let Some(swap) = matching.next() else { continue };
            ensure!(matching.all(|other| other.func == swap.func && other.grid == swap.grid && other.block == swap.block
                && other.pointers == swap.pointers && other.scalars == swap.scalars), "ambiguous graph node replacement");
            let (mut pointers, mut scalars) = (swap.pointers.clone(), swap.scalars.clone());
            let mut arguments: Vec<*mut c_void> = pointers.iter_mut().map(|v| (v as *mut u64).cast())
                .chain(scalars.iter_mut().map(|v| (v as *mut i32).cast())).collect();
            let replacement = KernelNodeParams { func: swap.func as Handle, grid: [swap.grid, 1, 1], block: [swap.block, 1, 1],
                shared: 0, params: arguments.as_mut_ptr(), extra: ptr::null_mut(), kern: ptr::null_mut(), ctx: p.ctx };
            api.check(unsafe { (edit.set)(node, &replacement) }, "replace kernel node")?;
            // The replaced kernel was launched cooperatively (all blocks resident); the replacement needs no such
            // contract. Clearing is best effort: a few blocks are resident together either way.
            if let Some(set) = edit.node_attribute { let _ = unsafe { set(node, COOPERATIVE, &[0u64; 8]) }; }
            changed += 1;
        }
        ensure!(changed == swaps.len(), "graph node replacement: {changed} nodes for {} launches", swaps.len());
        Ok(changed)
    }
    pub fn capture(&self, f: impl FnOnce() -> Result<()>) -> Result<Graph> {
        static DIAGNOSTIC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let diagnostic = *DIAGNOSTIC.get_or_init(||std::env::var("CANADA_SCRATCH_GRAPH_MEMORY_SCREEN").as_deref()==Ok("1"));
        let before = diagnostic.then(||crate::memory::Available::gb10(usize::MAX,usize::MAX)).transpose()?;
        self.owner.current()?;
        self.owner.api.check(
            unsafe { (self.owner.api.capture_begin)(self.raw, 0) },
            "begin graph capture",
        )?;
        CAPTURE_DEPTH.with(|d| d.set(d.get() + 1));
        CAPTURE_SERIAL.with(|s| s.set(s.get() + 1));
        let body = f();
        CAPTURE_DEPTH.with(|d| d.set(d.get() - 1));
        CAPTURE_SERIAL.with(|s| s.set(s.get() + 1));
        // Replacement records belong to this capture alone, also when it fails.
        let swaps = SWAPS.with(|s| std::mem::take(&mut *s.borrow_mut()));
        let mut graph = ptr::null_mut();
        let ended = self.owner.api.check(
            unsafe { (self.owner.api.capture_end)(self.raw, &mut graph) },
            "end graph capture",
        );
        if body.is_err() || ended.is_err() {
            if !graph.is_null() {
                unsafe {
                    (self.owner.api.graph_destroy)(graph);
                }
            }
            body?;
            ended?;
            unreachable!();
        }
        let mut exec = ptr::null_mut();
        let built = (|| {
            // Opt-in, host-side dependency evidence. Keep normal graph replay
            // unchanged and bound output when a diagnostic creates many graphs.
            if let Some(root) = std::env::var_os("CANADA_SCRATCH_GRAPH_DOT_DIR") {
                let mut count = 0;
                self.owner.api.check(unsafe {
                    (self.owner.api.graph_nodes)(graph, ptr::null_mut(), &mut count)
                }, "count diagnostic graph nodes")?;
                if count >= 1000 {
                    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                    let index = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if index < 16 {
                        let root = std::path::PathBuf::from(root);
                        std::fs::create_dir_all(&root)?;
                        let path = root.join(format!("graph-{index:02}-{count}.dot"));
                        let path = CString::new(path.as_os_str().as_encoded_bytes())?;
                        self.owner.api.check(unsafe {
                            (self.owner.api.graph_dot)(graph, path.as_ptr(), 1)
                        }, "save diagnostic graph dependencies")?;
                    }
                }
            }
            let rules = REGRID.with(|r| r.get());
            if rules.iter().any(Option::is_some) {
                let changed = self.regrid(graph, &rules)?;
                REGRIDDED.with(|n| n.set(n.get() + changed));
            }
            if !swaps.is_empty() {
                let changed = self.swap_nodes(graph, &swaps)?;
                SWAPPED.with(|n| n.set(n.get() + changed));
            }
            self.owner.api.check(
            unsafe { (self.owner.api.graph_instantiate)(&mut exec, graph, 0) },
            "instantiate graph",
            )
        })();
        unsafe {
            (self.owner.api.graph_destroy)(graph);
        }
        built?;
        if let Some(before) = before {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let after=crate::memory::Available::gb10(usize::MAX,usize::MAX)?;
            let rss=std::fs::read_to_string("/proc/self/status").unwrap_or_default().lines()
                .find(|l|l.starts_with("VmRSS:")).unwrap_or_default().to_owned();
            eprintln!("graph_memory {}",serde_json::json!({"capture":NEXT.fetch_add(1,std::sync::atomic::Ordering::Relaxed),
                "before":before,"after":after,"rss":rss}));
        }
        Ok(Graph {
            owner: self.owner.clone(),
            exec,
            required_stream: self.graph_stream_bound.get().then(||self.identity.clone()),
        })
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        let _ = self.synchronize();
        unsafe {
            if !self.upload_fence.is_null() {
                (self.owner.api.event_destroy)(self.upload_fence);
            }
            if !self.blocking_fence.get().is_null() {
                (self.owner.api.event_destroy)(self.blocking_fence.get());
            }
            if !self.polling_fence.get().is_null() {
                (self.owner.api.event_destroy)(self.polling_fence.get());
            }
            (self.owner.api.stream_destroy)(self.raw);
        }
    }
}
#[derive(Clone)]
pub struct Module {
    inner: Rc<ModuleOwner>,
}
struct ModuleOwner {
    owner: Rc<ContextOwner>,
    raw: Handle,
}
impl Module {
    pub fn function(&self, name: &str) -> Result<Function<'_>> {
        let name = CString::new(name)?;
        let mut raw = ptr::null_mut();
        self.inner.owner.api.check(
            unsafe { (self.inner.owner.api.get_function)(&mut raw, self.inner.raw, name.as_ptr()) },
            "get kernel",
        ).with_context(|| format!("CUDA module symbol {}",name.to_string_lossy()))?;
        Ok(Function { module: self, raw })
    }
}
impl Drop for ModuleOwner {
    fn drop(&mut self) {
        let _ = self.owner.current();
        unsafe {
            (self.owner.api.module_unload)(self.raw);
        }
    }
}
pub struct Function<'a> {
    module: &'a Module,
    raw: Handle,
}
impl Function<'_> {
    /// The raw kernel handle, for graph node replacement (cuda::NodeSwap); valid while the module lives.
    pub(crate) fn handle(&self) -> usize { self.raw as usize }
    /// Safety: the ordinary launch ABI/lifetime contract applies. The kernel
    /// additionally executes griddepcontrol.wait before dependent global reads;
    /// every predecessor's programmatic trigger is legal for that dependency.
    pub(crate) unsafe fn launch_pdl(
        &self, grid: [u32; 3], block: [u32; 3], shared: u32,
        stream: &Stream, args: &mut [*mut c_void],
    ) -> Result<()> {
        ensure!(Rc::ptr_eq(&stream.owner, &self.module.inner.owner), "mixed CUDA contexts");
        ensure!(grid.iter().all(|n| *n > 0) && block.iter().all(|n| *n > 0), "empty launch");
        self.module.inner.owner.current()?;
        let mut attribute = LaunchAttribute { id: 6, pad: [0; 4], value: [0; 8] };
        attribute.value[0] = 1; // int programmaticStreamSerializationAllowed, little endian
        let config = LaunchConfig { grid, block, shared, stream: stream.raw,
            attrs: &mut attribute, count: 1 };
        self.module.inner.owner.api.check(unsafe {
            (self.module.inner.owner.api.launch_ex)(&config, self.raw, args.as_mut_ptr(), ptr::null_mut())
        }, "launch programmatic dependent kernel")
    }
    pub fn opt_in_shared_memory(&self, bytes: u32) -> Result<()> {
        self.module.inner.owner.api.check(
            unsafe { (self.module.inner.owner.api.function_attr)(self.raw, 8, bytes as i32) },
            "opt into dynamic shared memory",
        )
    }
    /// Safety: arguments must exactly match this kernel's scalar/pointer ABI;
    /// buffers and their pointed-to data must remain valid until stream completion.
    pub unsafe fn launch(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        stream: &Stream,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&stream.owner, &self.module.inner.owner),
            "mixed CUDA contexts"
        );
        ensure!(
            grid.iter().all(|n| *n > 0) && block.iter().all(|n| *n > 0),
            "empty launch"
        );
        self.module.inner.owner.current()?;
        self.module.inner.owner.api.check(
            unsafe {
                (self.module.inner.owner.api.launch)(
                    self.raw,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared,
                    stream.raw,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "launch kernel",
        )
    }
}
pub struct Event {
    owner: Rc<ContextOwner>,
    raw: Handle,
}
impl Event {
    pub fn record(&self, stream: &Stream) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.owner, &stream.owner),
            "mixed event/stream context"
        );
        self.owner.current()?;
        self.owner.api.check(
            unsafe { (self.owner.api.event_record)(self.raw, stream.raw) },
            "record event",
        )
    }
    pub(crate) fn wait(&self, stream: &Stream) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.owner, &stream.owner),
            "mixed dependency event/stream context"
        );
        self.owner.current()?;
        self.owner.api.check(
            unsafe { (self.owner.api.stream_wait_event)(stream.raw, self.raw, 0) },
            "wait dependency event",
        )
    }
    pub fn elapsed_since(&self, start: &Event) -> Result<f32> {
        ensure!(
            Rc::ptr_eq(&self.owner, &start.owner),
            "mixed event contexts"
        );
        self.owner.api.check(
            unsafe { (self.owner.api.event_sync)(self.raw) },
            "wait event",
        )?;
        let mut ms = 0.0;
        self.owner.api.check(
            unsafe { (self.owner.api.event_elapsed)(&mut ms, start.raw, self.raw) },
            "elapsed event time",
        )?;
        Ok(ms)
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            (self.owner.api.event_destroy)(self.raw);
        }
    }
}
pub struct Graph {
    owner: Rc<ContextOwner>,
    exec: Handle,
    required_stream: Option<Rc<()>>,
}
impl Graph {
    /// Safety: every captured pointer must still be allocated and exclusively
    /// accessible to this graph, including external workspaces/collectives.
    pub unsafe fn launch(&self, stream: &Stream) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.owner, &stream.owner),
            "mixed graph/stream context"
        );
        ensure!(self.required_stream.as_ref().is_none_or(|id|Rc::ptr_eq(id,&stream.identity)),
            "graph with pooled scratch must replay on its capture stream");
        self.owner.api.check(
            unsafe { (self.owner.api.graph_launch)(self.exec, stream.raw) },
            "launch graph",
        )
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            (self.owner.api.exec_destroy)(self.exec);
        }
    }
}

/// Exercise staged integer uploads, deferred visibility, zero/partial setters,
/// allocator lifetime and changed values consumed by an existing GPU graph.
pub fn probe_staged_metadata() -> Result<serde_json::Value> {
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let allocator = unsafe {cuda.staged_metadata_allocator()?};
    let mut a = allocator.allocate::<i32>(32)?;
    let mut b = allocator.allocate::<u64>(17)?;
    let mut c = allocator.allocate::<u8>(257)?;
    let stage = allocator.metadata_staging().unwrap();
    ensure!(stage.bytes() == 32*4 + 17*8 + 257, "staging payload accounting");
    let mut fallback = allocator.allocate::<u8>(4097)?;
    fallback.upload(&vec![19;4097])?;
    ensure!(stage.bytes() == 521 && fallback.download()? == vec![19;4097], "large allocation fallback");
    drop(allocator);
    // Device bytes remain untouched until the owner explicitly publishes them.
    a.upload(&[7;32])?;b.upload(&[9;17])?;b.zero()?;c.zero()?;
    ensure!(a.download()? == vec![0;32], "staged writes were not deferred");
    unsafe {stage.flush(None)?;}
    ensure!(a.download()? == vec![7;32] && b.download()? == vec![0;17], "initial staging flush");
    let observed = cuda.allocate::<u8>(stage.transfer_bytes())?;
    let graph = stream.capture(||unsafe {
        cuda.copy_device(observed.ptr(),stage.device.ptr(),observed.bytes(),&stream)
    })?;
    let mut cycles = Vec::new();
    for cycle in 0..4 {
        stream.synchronize()?;
        let av: Vec<i32> = (0..32).map(|i| i*13-71+cycle).collect();
        let bv: Vec<u64> = (0..17).map(|i| 100000+37*i+cycle as u64).collect();
        a.upload(&av)?;b.upload(&bv)?;c.zero()?;
        c.upload_part(19,&[cycle as u8+23;31])?;
        unsafe {stage.flush(Some(&stream))?;graph.launch(&stream)?;}
        stream.synchronize()?;
        let actual = observed.download()?;
        let mut expected = vec![0u8;stage.transfer_bytes()];
        let offset = |p:u64| (p-stage.device.ptr()) as usize;
        for (index,value) in av.iter().enumerate() {
            expected[offset(a.ptr())+index*4..offset(a.ptr())+(index+1)*4].copy_from_slice(&value.to_le_bytes());
        }
        for (index,value) in bv.iter().enumerate() {
            expected[offset(b.ptr())+index*8..offset(b.ptr())+(index+1)*8].copy_from_slice(&value.to_le_bytes());
        }
        expected[offset(c.ptr())+19..offset(c.ptr())+50].fill(cycle as u8+23);
        ensure!(actual == expected, "captured staged metadata or alignment gaps changed");
        cycles.push(serde_json::json!({"cycle":cycle,"exact":true}));
    }
    Ok(serde_json::json!({"all_correct":true,"cycles":cycles,"payload_bytes":stage.bytes(),
        "transfer_bytes":stage.transfer_bytes(),"deferred_visibility":true,"allocator_drop_safe":true,
        "large_allocation_fallback":true,"full_engine_acceptance":false}))
}

/// Compare duplicate core-module loads with shared immutable code. Buffers,
/// stream arguments and captured graphs retain their ordinary ownership.
pub fn probe_core_module_reuse(core:&std::path::Path)->Result<serde_json::Value> {
    use sha2::Digest;
    let artifact=std::fs::read(core)?;
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;
    let stream=cuda.stream()?;
    ensure!(cuda.admission_memory()?.admitted_available>=16*1024usize.pow(3),"module screen headroom");
    let mut input=cuda.allocate::<u32>(2)?;
    let output=cuda.allocate::<u32>(2)?;
    // Resolve the diagnostic entry once before the alternating load panel.
    let warm=cuda.module(&artifact)?;let _=warm.function("ca_probe_read_word")?;drop(warm);
    let mut arms=Vec::new();
    for (arm,shared)in [false,true,true,false].into_iter().enumerate() {
        *cuda.owner.shared_core.borrow_mut()=None;
        if shared{cuda.share_immutable_core(&artifact)?;}
        let before=cuda.admission_memory()?;
        let began=std::time::Instant::now();
        let modules=(0..64).map(|_|cuda.module(&artifact)).collect::<Result<Vec<_>>>()?;
        let load_ms=began.elapsed().as_secs_f64()*1000.;
        let after=cuda.admission_memory()?;
        let distinct=modules.iter().map(|m|Rc::as_ptr(&m.inner)as usize)
            .collect::<std::collections::BTreeSet<_>>().len();
        ensure!(distinct==if shared{1}else{64},"module reuse identity");
        let keep=[modules[0].clone(),modules[63].clone()];drop(modules);
        let graph=stream.capture(|| {
            for (i,module)in keep.iter().enumerate() {
                let(mut source,mut dest)=(input.ptr()+(i*4)as u64,output.ptr()+(i*4)as u64);
                let mut args=[(&mut source as *mut u64).cast(),(&mut dest as *mut u64).cast()];
                unsafe{module.function("ca_probe_read_word")?.launch([1,1,1],[32,1,1],0,&stream,&mut args)?;}
            }Ok(())
        })?;
        for cycle in 0..3u32 {
            let expected=[91+cycle+arm as u32*11,127-cycle+arm as u32*17];
            input.upload(&expected)?;stream.join_default()?;
            unsafe{graph.launch(&stream)?;}stream.synchronize()?;
            ensure!(output.download()?==expected,"shared module crossed private arguments or graph lifetime");
        }
        drop(graph);drop(keep);
        ensure!(!shared||cuda.owner.shared_core.borrow().as_ref().unwrap().2.upgrade().is_none(),
            "weak module cache retained GPU code after its final owner");
        arms.push(serde_json::json!({"arm":arm,"shared":shared,"copies":64,"distinct_modules":distinct,
            "load_ms":load_ms,"before":before,"after":after,"released":cuda.admission_memory()?,
            "private_arguments_exact":true,"graph_lifetime_exact":true,"weak_release":true}));
    }
    Ok(serde_json::json!({"all_exact":true,"artifact_bytes":artifact.len(),
        "artifact_sha256":format!("{:x}",sha2::Sha256::digest(&artifact)),"arms":arms,
        "scope":"Immutable in-tree core module only; ABBA64loads, changed private pointer data and graph lifetime. Not full-model numerical or performance qualification."}))
}

/// GPU-only diagnostic of the real stream join, with a bounded delayed default
/// producer. Integer sentinels test ordering, not any model computation.
pub fn probe_default_order(core: &std::path::Path) -> Result<serde_json::Value> {
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let module = cuda.module(&std::fs::read(core)?)?;
    let producer = module.function("ca_probe_delayed_word")?;
    let reader = module.function("ca_probe_read_word")?;
    let mut word = cuda.allocate::<u32>(1)?;
    let output = cuda.allocate::<u32>(1)?;
    let mut records = Vec::new();
    for fenced in [false, true] {
        for repeat in 0..8 {
            word.upload(&[0x11111111])?;
            cuda.owner.api.check(
                unsafe { (cuda.owner.api.stream_sync)(ptr::null_mut()) },
                "seed ordering diagnostic",
            )?;
            let mut address = word.ptr();
            let mut value = 0x22222222u32 + repeat;
            let mut cycles = 50000000u64;
            let mut args = [
                (&mut address as *mut u64).cast(),
                (&mut value as *mut u32).cast(),
                (&mut cycles as *mut u64).cast(),
            ];
            cuda.owner.api.check(
                unsafe {
                    (cuda.owner.api.launch)(
                        producer.raw,
                        1,
                        1,
                        1,
                        32,
                        1,
                        1,
                        0,
                        ptr::null_mut(),
                        args.as_mut_ptr(),
                        ptr::null_mut(),
                    )
                },
                "launch diagnostic default producer",
            )?;
            if fenced {
                stream.join_default()?;
            }
            let mut result = output.ptr();
            let mut read_args = [
                (&mut address as *mut u64).cast(),
                (&mut result as *mut u64).cast(),
            ];
            unsafe { reader.launch([1, 1, 1], [32, 1, 1], 0, &stream, &mut read_args) }?;
            stream.synchronize()?;
            let actual = output.download()?[0];
            cuda.owner.api.check(
                unsafe { (cuda.owner.api.stream_sync)(ptr::null_mut()) },
                "drain diagnostic default producer",
            )?;
            records.push(serde_json::json!({"fenced":fenced,"repeat":repeat,"actual":actual,"expected":value,"new_value_visible":actual==value,"old_value_visible":actual==0x11111111}));
        }
    }
    let capture_guard = stream.capture(|| stream.join_default()).is_err();
    stream.join_default()?;
    stream.synchronize()?;
    Ok(
        serde_json::json!({"all_fenced_correct":records.iter().filter(|r|r["fenced"]==true).all(|r|r["new_value_visible"]==true),
        "unfenced_old_value_cases":records.iter().filter(|r|r["fenced"]==false&&r["old_value_visible"]==true).count(),
        "capture_guard":capture_guard,"records":records,"full_engine_acceptance":false,
        "scope":"Real default-to-nonblocking stream event dependency with delayed integer producer. Establishes ordering mechanism and capture guard; does not prove root cause of earlier1M KV mismatch."}),
    )
}
