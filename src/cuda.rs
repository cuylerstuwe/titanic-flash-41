//! Minimal CUDA driver ownership. No PyTorch, Python, or model-engine dependency.
//! All handles stay on their owning OS thread. Buffers/modules keep the context alive.
use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use std::{
    cell::RefCell,
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    marker::PhantomData,
    ptr,
    rc::{Rc, Weak},
};

type Handle = *mut c_void;
type CuResult = c_int;

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
    set_current: unsafe extern "C" fn(Handle)->CuResult = "cuCtxSetCurrent";
    context_sync: unsafe extern "C" fn()->CuResult = "cuCtxSynchronize";
    alloc: unsafe extern "C" fn(*mut u64,usize)->CuResult = "cuMemAlloc_v2";
    free: unsafe extern "C" fn(u64)->CuResult = "cuMemFree_v2";
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
    error_string: unsafe extern "C" fn(CuResult,*mut *const c_char)->CuResult = "cuGetErrorString";
    event_create: unsafe extern "C" fn(*mut Handle,c_uint)->CuResult = "cuEventCreate";
    event_destroy: unsafe extern "C" fn(Handle)->CuResult = "cuEventDestroy_v2";
    event_record: unsafe extern "C" fn(Handle,Handle)->CuResult = "cuEventRecord";
    event_sync: unsafe extern "C" fn(Handle)->CuResult = "cuEventSynchronize";
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
        bail!("{op}: {s} ({code})")
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
    /// A local allocator for small integer scheduling buffers only. Do not
    /// pass this clone to model-value or weight constructors.
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
    pub fn mapped_host(&self, bytes: usize) -> Result<MappedHost> {
        ensure!(
            bytes > 0 && bytes <= 1024 * 1024 * 1024,
            "bounded pinned host allocation required"
        );
        self.owner.current()?;
        let mut host = ptr::null_mut();
        self.owner.api.check(
            unsafe { (self.owner.api.host_alloc)(&mut host, bytes, 3) },
            "allocate mapped host region",
        )?;
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
    /// Construct one layer's row-sized transient storage inside `arena`.
    /// Every `allocate_rows` call made by `build` carves from the arena start,
    /// so layers built in the same arena alias each other's bytes. Callers
    /// must alternate arenas between adjacent layers and only build layers
    /// that execute sequentially on one ordered stream.
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
    pub fn row_arena(&self, bytes: usize) -> Result<Rc<RowArena>> {
        Ok(Rc::new(RowArena {
            storage: self.allocate(bytes)?,
            next: std::cell::Cell::new(0),
            high_water: std::cell::Cell::new(0),
        }))
    }
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
        let mut ptr = 0;
        self.owner.api.check(
            unsafe { (self.owner.api.alloc)(&mut ptr, bytes) },
            "allocate device memory",
        )?;
        Ok(Buffer {
            owner: self.owner.clone(),
            ptr,
            len,
            storage: Storage::Owned,
            _element: PhantomData,
        })
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
            upload_fence: ptr::null_mut(),
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
    pub fn module(&self, artifact: &[u8]) -> Result<Module> {
        ensure!(!artifact.is_empty(), "empty kernel artifact");
        self.owner.current()?;
        let mut data = artifact.to_vec();
        data.push(0); // PTX needs NUL; cubin ignores trailing byte.
        let mut raw = ptr::null_mut();
        self.owner.api.check(
            unsafe { (self.owner.api.module_load)(&mut raw, data.as_ptr().cast()) },
            "load CUDA module",
        )?;
        Ok(Module {
            inner: Rc::new(ModuleOwner {
                owner: self.owner.clone(),
                raw,
            }),
        })
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
    // Created only by the scoped integer-metadata allocator above.
    MappedMetadata(MappedHost),
    StagedMetadata { arena: Rc<MetadataStaging>, offset: usize },
    // A view into a shared row arena; the arena owns the bytes.
    Arena(#[allow(dead_code)] Rc<RowArena>),
    // A borrowed row range of another buffer (see Buffer::view).
    View,
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
        ensure!(
            offset.checked_add(bytes).is_some_and(|n| n <= self.len),
            "partial download outside buffer"
        );
        self.owner.current()?;
        let mut data = vec![0u8; bytes];
        self.owner.api.check(
            unsafe {
                (self.owner.api.dtoh)(data.as_mut_ptr().cast(), self.ptr + offset as u64, bytes)
            },
            "partial download",
        )?;
        Ok(data)
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
        let _ = self.owner.current();
        unsafe {
            (self.owner.api.free)(self.ptr);
        }
    }
}

pub struct Stream {
    owner: Rc<ContextOwner>,
    raw: Handle,
    upload_fence: Handle,
}
impl Stream {
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
    pub fn capture(&self, f: impl FnOnce() -> Result<()>) -> Result<Graph> {
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
        Ok(Graph {
            owner: self.owner.clone(),
            exec,
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
        )?;
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
}
impl Graph {
    /// Safety: every captured pointer must still be allocated and exclusively
    /// accessible to this graph, including external workspaces/collectives.
    pub unsafe fn launch(&self, stream: &Stream) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.owner, &stream.owner),
            "mixed graph/stream context"
        );
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
