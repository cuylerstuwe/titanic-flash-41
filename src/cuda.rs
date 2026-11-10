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
    row_arena: RefCell<Option<Rc<RowArena>>>,
    // A weak entry avoids a ContextOwner -> ModuleOwner -> ContextOwner cycle.
    // Only explicitly registered, immutable core code is eligible.
    shared_core: RefCell<Option<([u8;32],usize,Weak<ModuleOwner>)>>,
    allocation_floor_screen: bool,
    allocation_floor_pending: std::cell::Cell<usize>,
    allocation_sites: Option<RefCell<std::collections::BTreeMap<u64,(&'static std::panic::Location<'static>,usize)>>>,
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
        unsafe {
            (self.api.primary_release)(self.device);
        }
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
                row_arena: RefCell::new(None),
                shared_core: RefCell::new(None),
                allocation_floor_screen: std::env::var("CANADA_SCRATCH_ALLOCATION_FLOOR_SCREEN").as_deref()==Ok("1"),
                allocation_sites: (std::env::var("CANADA_SCRATCH_ALLOCATION_SITES_SCREEN").as_deref()==Ok("1"))
                    .then(||RefCell::new(std::collections::BTreeMap::new())),
                allocation_floor_pending: std::cell::Cell::new(0),
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
    /// Row-proportional transient storage. Outside a row-arena scope this is an
    /// ordinary owned allocation, so existing profiles are unchanged.
    #[track_caller]
    pub fn allocate_rows<T: DeviceElement>(&self, len: usize) -> Result<Buffer<T>> {
        let arena = self.owner.row_arena.borrow().clone();
        let Some(arena) = arena else {
            return self.allocate(len);
        };
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
        }))
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
            "top":sorted.into_iter().take(32).map(|(site,(bytes,count))|serde_json::json!({"site":site,"bytes":bytes,"count":count})).collect::<Vec<_>>(),
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
        for slot in [&self.owner.auxiliary, &self.owner.coefficients, &self.owner.queries] {
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
            (self.owner.api.host_free)(self.host);
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
/// Layer-parity transient storage for wide prefill profiles.
pub struct RowArena {
    storage: Buffer<u8>,
    next: std::cell::Cell<usize>,
    high_water: std::cell::Cell<usize>,
}
impl RowArena {
    pub fn bytes(&self) -> usize {
        self.storage.len
    }
    pub fn high_water(&self) -> usize {
        self.high_water.get()
    }
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
        self.owner.current()?;
        let mut v = vec![T::default(); self.len];
        self.owner.api.check(
            unsafe { (self.owner.api.dtoh)(v.as_mut_ptr().cast(), self.ptr, self.bytes()) },
            "download buffer",
        )?;
        Ok(v)
    }
    pub fn zero(&mut self) -> Result<()> {
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
        if let Some(sites)=&self.owner.allocation_sites {sites.borrow_mut().remove(&self.ptr);}
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
        ensure!([25,50].contains(&microseconds),"bounded completion polling interval");
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
            std::thread::sleep(std::time::Duration::from_micros(microseconds));
        }
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
        let body = f();
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
