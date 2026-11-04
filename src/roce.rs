//! Rust ownership of a specialized low-level RoCE primitive dependency.
//! This module owns admission, mapping, peer exchange, ordering and poisoning;
//! the separately pinned proxy handles verbs and AOT kernels perform GPU math.
use crate::{
    cuda::{Buffer, Cuda, DeviceElement, MappedHost, Stream},
    primitives::Primitives,
};
use anyhow::{Context, Result, ensure};
use libloading::Library;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    cell::{Cell, RefCell},
    ffi::{CStr, CString, c_char, c_void},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReductionRows {
    live: usize,
    capacity: usize,
}
impl ReductionRows {
    fn new(live: usize, capacity: usize) -> Result<Self> {
        ensure!(
            live > 0
                && live <= capacity
                && ([5, 6, 16, 32, 128].contains(&capacity) || crate::config::wide_rows(capacity)),
            "invalid reduction row geometry"
        );
        Ok(Self { live, capacity })
    }
    fn count(self, elements: usize) -> Result<usize> {
        ensure!(
            elements > 0 && elements % self.capacity == 0,
            "reduction is not row-major capacity storage"
        );
        Ok((elements / self.capacity) * self.live)
    }
}
type Handle = *mut c_void;
/// One ordered scratch pair, shared by every prefill layer/profile. It owns
/// no model state and never changes the native decode reduction path.
struct PrecisePrefill {
    module:crate::cuda::Module,
    input:Buffer<f32>,
    output:Buffer<f32>,
}
#[derive(Deserialize)]
struct Layout {
    slot_bytes: u64,
    recv_off: u64,
    flag_off: u64,
    send_off: u64,
    ctrl_off: u64,
    total_bytes: u64,
    flag_stride: u64,
    slots: u64,
}
#[derive(Deserialize)]
struct Manifest {
    layout: Layout,
    proxy_abi: i32,
    proxy_library_sha256: String,
}
pub struct Roce {
    instance: u64,
    moe_pipeline:Option<crate::moe_transport::Pipeline>,
    use_moe_pipeline:Cell<bool>,
    prefill_sum: Option<crate::collective::Communicator>,
    precise_prefill:Option<PrecisePrefill>,
    exchange_prefill:Option<crate::prefill_exchange::Exchange>,
    native_prefill_rs:Option<crate::prefill_rs::ReduceScatter>,
    small_prefill_sum:bool,
    small_prefill_enqueues:Cell<usize>,
    native_sum_selected: Cell<bool>,
    prefill_rows: Cell<Option<ReductionRows>>,
    native_sum_rows: Cell<Option<ReductionRows>>,
    native_gather_rows: Cell<Option<ReductionRows>>,
    column_gather: Option<(crate::cuda::Module, RefCell<Buffer<u16>>)>,
    column_gather_live_rows: Cell<bool>,
    column_gather_prefix_calls: Cell<usize>,
    cuda: Cuda,
    _lib: Library,
    handle: Handle,
    layout: Layout,
    region: MappedHost,
    peak_native_bytes: Cell<usize>,
    counters: Buffer<u32>,
    gather_grid_override: Cell<Option<usize>>,
    sum_grid_override: Cell<Option<usize>>,
    proxy_thread_for_probe: Cell<Option<i32>>,
    tuned_logit_gather: bool,
    tuned_small_sum: bool,
    kernels: Primitives,
    notification_probe_kernels: Vec<Primitives>,
    notification_probe_variant: Cell<usize>,
    gather: Option<Primitives>,
    local_blob: unsafe extern "C" fn(Handle, *mut c_void, u64) -> i32,
    blob_size: usize,
    connect_fn: unsafe extern "C" fn(Handle, *const c_void, u64) -> i32,
    start_fn: unsafe extern "C" fn(Handle) -> i32,
    failed: unsafe extern "C" fn(Handle) -> i32,
    error: unsafe extern "C" fn(Handle) -> *const c_char,
    destroy: unsafe extern "C" fn(Handle),
    poisoned: Cell<bool>,
    stream: Cell<Option<usize>>,
    rank: usize,
    _hca_names: Vec<CString>,
}
/// Temporary target-only prefix policy. Drafts and unrelated collectives must
/// never inherit a target's live-row count, including after an execution error.
pub(crate) struct NativeRowScope<'a> {
    comm: &'a Roce,
    previous_sum: Option<ReductionRows>,
    previous_gather: Option<ReductionRows>,
}
impl Drop for NativeRowScope<'_> {
    fn drop(&mut self) {
        self.comm.native_sum_rows.set(self.previous_sum);
        self.comm.native_gather_rows.set(self.previous_gather);
    }
}
impl Roce {
    pub(crate) fn instance(&self) -> u64 {
        self.instance
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub(crate) fn transport_memory(&self) -> serde_json::Value {
        serde_json::json!({"slot_bytes":self.layout.slot_bytes,
            "mapped_region_bytes":self.layout.total_bytes,
            "maximum_enqueued_native_bytes":self.peak_native_bytes.get(),
            "small_prefill_sum":self.small_prefill_sum,
            "native_prefill_rs":self.native_prefill_rs.is_some(),
            "native_prefill_rs_enqueues":self.native_prefill_rs.as_ref().map_or(0,|p|p.calls()),
            "small_prefill_enqueues":self.small_prefill_enqueues.get()})
    }
    pub fn local(cuda: &Cuda, root: &Path, rank: usize) -> Result<Self> {
        ensure!(rank < 4, "RoCE rank outside TP4");
        let instance = NEXT_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| anyhow::anyhow!("RoCE instance identity exhausted"))?;
        cuda.qualify_gb10()?;
        let mut manifest: Manifest =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let l = &manifest.layout;
        ensure!(
            manifest.proxy_abi == 3
                && l.slot_bytes == 64 * 1024 * 1024
                && l.slots == 2
                && l.flag_stride == 128
                && l.total_bytes <= 1024 * 1024 * 1024,
            "unsupported RoCE layout"
        );
        let path = root.join("libcanada_roce_proxy.so");
        ensure!(
            format!("{:x}", Sha256::digest(std::fs::read(&path)?)) == manifest.proxy_library_sha256,
            "RoCE proxy hash mismatch"
        );
        let lib = unsafe { Library::new(path) }?;
        macro_rules! sym {
            ($s:literal) => {
                *unsafe { lib.get(concat!($s, "\0").as_bytes()) }?
            };
        }
        let abi: unsafe extern "C" fn() -> i32 = sym!("roce_abi_version");
        ensure!(unsafe { abi() } == 3, "RoCE proxy ABI mismatch");
        let layout_fn: unsafe extern "C" fn(i32, u64, *mut u64) -> i32 = sym!("roce_layout");
        let mut layout = [0u64; 7];
        ensure!(
            unsafe { layout_fn(4, l.slot_bytes, layout.as_mut_ptr()) } == 0,
            "RoCE layout query failed"
        );
        ensure!(
            layout
                == [
                    l.recv_off,
                    l.flag_off,
                    l.send_off,
                    l.ctrl_off,
                    l.total_bytes,
                    l.flag_stride,
                    l.slots
                ],
            "RoCE layout drift"
        );
        // The pinned proxy and kernels take offsets/slot size as runtime
        // arguments. Preserve their exported ABI and hashes, deriving only a
        // smaller owned region. Every native launch still enforces its bound.
        // Experimental until full-width messages and model paths are checked.
        let slot_mib = match std::env::var("CANADA_SCRATCH_ROCE_SLOT_MIB") {
            Err(std::env::VarError::NotPresent) => 64u64,
            Ok(value) => value.parse::<u64>()?,
            Err(error) => return Err(error.into()),
        };
        ensure!([16,64].contains(&slot_mib), "RoCE slots must be16 or64MiB");
        if slot_mib != 64 {
            let slot_bytes=slot_mib*1024*1024;
            let mut derived=[0u64;7];
            ensure!(unsafe{layout_fn(4,slot_bytes,derived.as_mut_ptr())}==0,
                "reduced RoCE layout query failed");
            ensure!(derived[0]==0 && derived[5]==l.flag_stride && derived[6]==l.slots
                && derived[4]<l.total_bytes,"reduced RoCE layout invariants");
            manifest.layout=Layout {slot_bytes,recv_off:derived[0],flag_off:derived[1],
                send_off:derived[2],ctrl_off:derived[3],total_bytes:derived[4],
                flag_stride:derived[5],slots:derived[6]};
        }
        let l=&manifest.layout;
        ensure!(cuda.admission_memory()?.admitted_available as u64
            >l.total_bytes+2*1024*1024*1024,"insufficient memory for bounded RoCE region");
        // Deployment-specific rail identities must be explicit. Checking each
        // GID protects against using a management interface or another rank.
        let hcas = std::env::var("TITANIC_ROCE_HCAS")
            .context("set TITANIC_ROCE_HCAS to this rank's two comma-separated HCA names")?;
        let addresses = std::env::var("TITANIC_ROCE_IPV4")
            .context("set TITANIC_ROCE_IPV4 to this rank's two fabric IPv4 addresses")?;
        let gid_index: i32 = std::env::var("TITANIC_ROCE_GID_INDEX")
            .unwrap_or_else(|_| "3".into()).parse()?;
        let (names, expected_addresses) = rail_configuration(&hcas, &addresses, gid_index)?;
        for (name, expected) in names.iter().zip(&expected_addresses) {
            let p = Path::new("/sys/class/infiniband")
                .join(name.to_str()?).join("ports/1");
            ensure!(std::fs::read_to_string(p.join("state"))?.contains("ACTIVE"),
                "inactive RoCE rail");
            ensure!(std::fs::read_to_string(p.join(format!("gid_attrs/types/{gid_index}")))?.trim()
                == "RoCE v2", "selected GID is not RoCEv2");
            let gid: std::net::Ipv6Addr = std::fs::read_to_string(
                p.join(format!("gids/{gid_index}")))?.trim().parse()?;
            ensure!(gid.to_ipv4_mapped() == Some(*expected),
                "selected GID does not identify the configured fabric rail");
        }
        let region = cuda.mapped_host(l.total_bytes as usize)?;
        ensure!(
            region.host_ptr() as u64 == region.device_ptr(),
            "GB10 mapped host/device addresses differ"
        );
        // Original epoch, four stage/tail classes and poison retain their ABI.
        // Four extra words isolate the optional 16/32-block gather classes.
        let tuned_logit_gather=match std::env::var("CANADA_SCRATCH_LOGIT_GATHER_GRID") {
            Err(std::env::VarError::NotPresent)=>true,
            Ok(v)if v=="1"=>true,
            Ok(v)if v=="0"=>false,
            _=>anyhow::bail!("invalid logit gather grid policy"),
        };
        let tuned_small_sum=match std::env::var("CANADA_SCRATCH_SMALL_SUM_GRID") {
            Err(std::env::VarError::NotPresent)=>true,
            Ok(v)if v=="1"=>true,
            Ok(v)if v=="0"=>false,
            _=>anyhow::bail!("invalid small sum grid policy"),
        };
        let mut counters = cuda.allocate::<u32>(14)?;
        counters.zero()?;
        let create: unsafe extern "C" fn(
            i32,
            i32,
            *const *const c_char,
            i32,
            i32,
            *mut c_void,
            u64,
            u64,
            *mut c_char,
            u64,
        ) -> Handle = sym!("roce_create");
        let name_ptrs: Vec<_> = names.iter().map(|n| n.as_ptr()).collect();
        let mut err = [0 as c_char; 1024];
        let handle = unsafe {
            create(
                4,
                rank as i32,
                name_ptrs.as_ptr(),
                2,
                gid_index,
                region.host_ptr(),
                l.total_bytes,
                l.slot_bytes,
                err.as_mut_ptr(),
                err.len() as u64,
            )
        };
        ensure!(
            !handle.is_null(),
            "create RoCE proxy: {}",
            unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy()
        );
        let destroy: unsafe extern "C" fn(Handle) = sym!("roce_destroy");
        let kernels = match Primitives::open(
            cuda,
            &root.join("libcanada_primitives.so"),
            &root.join("bindings.json"),
        ) {
            Ok(k) => k,
            Err(e) => {
                unsafe {
                    destroy(handle);
                }
                return Err(e);
            }
        };
        let blob_bytes: unsafe extern "C" fn() -> u64 = sym!("roce_blob_bytes");
        let blob_size = unsafe { blob_bytes() } as usize;
        ensure!(
            blob_size > 0 && blob_size <= 65536,
            "invalid RoCE endpoint blob size"
        );
        Ok(Self {
            instance,
            moe_pipeline:None,
            use_moe_pipeline:Cell::new(false),
            prefill_sum: None,
            precise_prefill:None,
            exchange_prefill:None,
            native_prefill_rs:None,
            small_prefill_sum:false,
            small_prefill_enqueues:Cell::new(0),
            native_sum_selected: Cell::new(false),
            prefill_rows: Cell::new(None),
            native_sum_rows: Cell::new(None),
            native_gather_rows: Cell::new(None),
            column_gather: None,
            column_gather_live_rows: Cell::new(std::env::var("CANADA_SCRATCH_PREFILL_ENGRAM_GATHER_ROWS").as_deref()!=Ok("0")),
            column_gather_prefix_calls: Cell::new(0),
            cuda: cuda.clone(),
            local_blob: sym!("roce_local_blob"),
            connect_fn: sym!("roce_connect"),
            start_fn: sym!("roce_start"),
            failed: sym!("roce_failed"),
            error: sym!("roce_error"),
            destroy,
            _lib: lib,
            handle,
            layout: manifest.layout,
            region,
            peak_native_bytes: Cell::new(0),
            counters,
            gather_grid_override: Cell::new(None),
            sum_grid_override: Cell::new(None),
            proxy_thread_for_probe: Cell::new(None),
            tuned_logit_gather,
            tuned_small_sum,
            kernels,
            notification_probe_kernels: Vec::new(),
            notification_probe_variant: Cell::new(0),
            gather: None,
            blob_size,
            poisoned: Cell::new(false),
            stream: Cell::new(None),
            rank,
            _hca_names: names,
        })
    }
    fn check(&self, status: i32) -> Result<()> {
        if status != 0 {
            self.poisoned.set(true);
        }
        ensure!(
            status == 0,
            "RoCE operation failed: {}",
            unsafe { CStr::from_ptr((self.error)(self.handle)) }.to_string_lossy()
        );
        Ok(())
    }
    pub fn blob(&self) -> Result<Vec<u8>> {
        let mut b = vec![0; self.blob_size];
        self.check(unsafe {
            (self.local_blob)(self.handle, b.as_mut_ptr().cast(), b.len() as u64)
        })?;
        Ok(b)
    }
    pub fn connect(&self, all: &[u8]) -> Result<()> {
        ensure!(
            all.len() == 4 * self.blob_size,
            "incomplete endpoint exchange"
        );
        self.check(unsafe {
            (self.connect_fn)(self.handle, all.as_ptr().cast(), all.len() as u64)
        })?;
        let affinity_probe=std::env::var("CANADA_SCRATCH_PROXY_AFFINITY_SCREEN").as_deref()==Ok("1");
        let before=if affinity_probe{Some(proxy_thread_ids()?)}else{None};
        self.check(unsafe { (self.start_fn)(self.handle) })?;
        if let Some(before)=before{
            let after=proxy_thread_ids()?;
            let added=after.difference(&before).copied().collect::<Vec<_>>();
            ensure!(added.len()==1,"isolated proxy probe requires exactly one new progress thread");
            self.proxy_thread_for_probe.set(Some(added[0]));
        }
        Ok(())
    }
    pub fn health(&self) -> Result<()> {
        if let Some(rs)=&self.native_prefill_rs {rs.health()?;}
        if let Some(pipeline)=&self.moe_pipeline {pipeline.health()?;}
        let failed = unsafe { (self.failed)(self.handle) };
        let sequence = self.region.status_word(self.layout.ctrl_off as usize + 8)?;
        if failed != 0 || sequence != 0 {
            self.poisoned.set(true);
        }
        ensure!(
            !self.poisoned.get(),
            "RoCE runtime poisoned: proxy_failed={failed} failed_sequence={sequence}"
        );
        if let Some(nccl) = &self.prefill_sum {
            if let Err(error) = nccl.check_async() {
                self.poisoned.set(true);
                return Err(error);
            }
        }
        Ok(())
    }
    /// A fresh/extended prefill is numerically a distinct qualified phase:
    /// its eager PyTorch ProcessGroupNCCL reduction rounds differently from
    /// the fixed-FP32 RoCE decode/verify reduction. Bind this before any GPU
    /// collective/capture; never switch a captured graph's transport in place.
    pub fn attach_prefill_sum(&mut self, comm: crate::collective::Communicator) -> Result<()> {
        ensure!(
            self.prefill_sum.is_none()
                && self.stream.get().is_none()
                && comm.rank == self.rank
                && comm.world == 4,
            "prefill reduction cannot replace a live communicator"
        );
        self.prefill_sum = Some(comm);
        Ok(())
    }
    pub(crate) fn attach_precise_prefill(&mut self,core:&[u8],rows:usize)->Result<()> {
        let small=match std::env::var("CANADA_SCRATCH_PREFILL_NATIVE_SMALL_SUM") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid small prefill reduction policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(e)=>return Err(e.into()),
        };
        ensure!(self.prefill_sum.is_some()&&self.stream.get().is_none(),
            "prefill reduction policy must precede any execution/capture");
        self.small_prefill_sum=small;
        let enabled=match std::env::var("CANADA_SCRATCH_PREFILL_FP32_SUM") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid FP32 prefill reduction policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(e)=>return Err(e.into()),
        };
        let exchange=match std::env::var("CANADA_SCRATCH_PREFILL_EXCHANGE_SUM") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid exchange prefill reduction policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(e)=>return Err(e.into()),
        };
        ensure!(!(enabled&&exchange),"choose one prefill reduction experiment");
        if !enabled&&!exchange{return Ok(());}
        ensure!(self.precise_prefill.is_none()&&self.exchange_prefill.is_none()
            &&self.prefill_sum.is_some()&&self.stream.get().is_none()
            &&crate::config::wide_rows(rows),"FP32 prefill construction state/extent");
        // Hidden5120 sums and the widest6144 Engram embedding both fit.
        let elements=rows.checked_mul(6144).ok_or_else(||anyhow::anyhow!("FP32 prefill extent overflow"))?;
        let reserve=8*1024usize.pow(3)+16*1024usize.pow(2);
        let bytes=if exchange{elements*2+elements/4*2}else{elements*8};
        ensure!(self.cuda.admission_memory()?.admitted_available>reserve+bytes,
            "FP32 prefill scratch would consume the runtime reserve");
        if exchange{self.exchange_prefill=Some(crate::prefill_exchange::Exchange::new(&self.cuda,core,elements)?);}
        else{self.precise_prefill=Some(PrecisePrefill{module:self.cuda.module(core)?,
            input:self.cuda.allocate(elements)?,output:self.cuda.allocate(elements)?});}
        ensure!(self.cuda.admission_memory()?.admitted_available>reserve,
            "FP32 prefill scratch consumed the runtime reserve");
        Ok(())
    }
    unsafe fn precise_prefill_sum<T:DeviceElement>(&self,input:&Buffer<T>,output:&mut Buffer<T>,
        count:usize,scatter:bool,stream:&Stream)->Result<()> {
        let p=self.precise_prefill.as_ref().ok_or_else(||anyhow::anyhow!("FP32 prefill unavailable"))?;
        let n=if scatter{count/4}else{count};
        ensure!(std::mem::size_of::<T>()==2&&count>0&&count<=input.len()&&count<=p.input.len()
            &&n<=output.len()&&(!scatter||count==4*output.len())
            &&(scatter||input.len()==output.len())
            &&self.cuda.owns_buffer(input)&&self.cuda.owns_buffer(output),"FP32 prefill buffer geometry/owner");
        let convert=|name:&str,source:u64,target:u64,count:usize|->Result<()> {
            let(mut a,mut b,mut n)=(source,target,count as i32);
            let mut args=[(&mut a as *mut u64).cast(),(&mut b as *mut u64).cast(),(&mut n as *mut i32).cast()];
            unsafe{p.module.function(name)?.launch([(count as u32).div_ceil(256),1,1],[256,1,1],0,stream,&mut args)}
        };
        convert("ca_bf16_to_f32",input.ptr(),p.input.ptr(),count)?;
        let source=unsafe{p.input.view(0,count)}?;
        let mut reduced=unsafe{p.output.view(0,n)}?;
        let nccl=self.prefill_sum.as_ref().unwrap();
        if scatter{unsafe{nccl.reduce_scatter_f32(&source,&mut reduced,stream)}?;}
        else{unsafe{nccl.sum_f32_to(&source,&mut reduced,stream)}?;}
        convert("ca_f32_to_bf16",reduced.ptr(),output.ptr(),n)?;
        if n<output.len(){unsafe{self.cuda.zero_device(output.ptr()+(n*2)as u64,(output.len()-n)*2,stream)}?;}
        Ok(())
    }
    /// Sequence-parallel prefill: sum every rank's [M, width] partial and keep
    /// this rank's M/4 rows. Prefill NCCL only; same ordering rules as sums.
    pub(crate) unsafe fn scatter_sum_bf16<T: DeviceElement>(
        &self,
        input: &Buffer<T>,
        output: &mut Buffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        self.health()?;
        let nccl = self
            .prefill_sum
            .as_ref()
            .filter(|_| !self.native_sum_selected.get())
            .ok_or_else(|| anyhow::anyhow!("sequence-parallel prefill needs NCCL prefill sums"))?;
        self.bind_prefill_stream(stream)?;
        let result = if let Some(rs)=self.native_prefill_rs.as_ref()
            .filter(|_|crate::prefill_rs::ReduceScatter::handles(input.len())) {
            unsafe{rs.run(input,output,stream)}
        }else if let Some(exchange)=&self.exchange_prefill {
            unsafe{exchange.run(nccl,input,output,input.len(),true,stream)}
        }else if self.precise_prefill.is_some(){
            unsafe{self.precise_prefill_sum(input,output,input.len(),true,stream)}
        }else{unsafe { nccl.reduce_scatter_bf16(input, output, stream) }};
        if result.is_err() {
            self.poisoned.set(true);
        }
        result
    }
    /// Sequence-parallel prefill: gather every rank's row shard, rank-major
    /// (BF16 activations or FP32 mHC coefficients, bit copy).
    pub(crate) unsafe fn gather_rows<T: DeviceElement>(
        &self,
        local: &Buffer<T>,
        global: &mut Buffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        self.health()?;
        let nccl = self
            .prefill_sum
            .as_ref()
            .filter(|_| !self.native_sum_selected.get())
            .ok_or_else(|| anyhow::anyhow!("sequence-parallel prefill needs NCCL prefill sums"))?;
        self.bind_prefill_stream(stream)?;
        let result = unsafe { nccl.gather_rows(local, global, stream) };
        if result.is_err() {
            self.poisoned.set(true);
        }
        result
    }
    /// Rank-major, in-place-capable bit copy of selected index rows.
    pub(crate) unsafe fn gather_index_rows(&self, local: &Buffer<i32>,
        global: &mut Buffer<i32>, stream: &Stream) -> Result<()> {
        self.health()?;
        ensure!(global.len() == local.len() * 4, "index row-gather geometry");
        let nccl = self.prefill_sum.as_ref().filter(|_| !self.native_sum_selected.get())
            .ok_or_else(|| anyhow::anyhow!("index row gather needs NCCL prefill communicator"))?;
        self.bind_prefill_stream(stream)?;
        let result = unsafe { nccl.gather_bytes_prefix(local, global, stream) };
        if result.is_err() {self.poisoned.set(true);}
        result
    }
    fn bind_prefill_stream(&self, stream: &Stream) -> Result<()> {
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.get().is_none_or(|prior| prior == raw),
            "prefill collectives require the same ordered CUDA stream"
        );
        self.stream.set(Some(raw));
        Ok(())
    }
    /// Select the reduction used by subsequent eager enqueue/capture. Already
    /// captured graphs retain their recorded transport. Drain the ordered
    /// stream at a policy transition so two execution profiles cannot overlap.
    pub(crate) fn select_decode_reduction(&self, decode: bool, stream: &Stream) -> Result<()> {
        ensure!(
            self.cuda.owns_stream(stream)
                && self.stream.get().is_none_or(|s| s == stream.raw() as usize),
            "reduction policy stream changed"
        );
        if self.native_sum_selected.get() != decode {
            stream.synchronize()?;
            self.health()?;
            self.native_sum_selected.set(decode);
        }
        Ok(())
    }
    pub(crate) fn native_row_scope(&self, sum:bool, gather:bool, live:usize, capacity:usize,
        stream:&Stream)->Result<NativeRowScope<'_>> {
        ensure!(!sum || std::env::var("CANADA_SCRATCH_TINY_APPEND_PREFIX_SUM").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_NATIVE_ROW_TRANSPORT_SCREEN").as_deref()==Ok("1")
            || (capacity==128 && live<=64 && std::env::var("CANADA_SCRATCH_MEDIUM_NATIVE_TRANSPORT").as_deref()==Ok("1"))
            || ([6,16].contains(&capacity) && std::env::var("CANADA_SCRATCH_LIVE_ROW_TRANSPORT").as_deref()==Ok("1")),
            "native sum prefix requires an admitted target profile");
        ensure!(!gather || std::env::var("CANADA_SCRATCH_TINY_APPEND_PREFIX_GATHER").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_NATIVE_ROW_TRANSPORT_SCREEN").as_deref()==Ok("1")
            || (capacity==128 && live<=64 && std::env::var("CANADA_SCRATCH_MEDIUM_NATIVE_TRANSPORT").as_deref()==Ok("1"))
            || ([6,16].contains(&capacity) && std::env::var("CANADA_SCRATCH_LIVE_ROW_TRANSPORT").as_deref()==Ok("1")),
            "native gather prefix requires an admitted target profile");
        ensure!(self.cuda.owns_stream(stream)
            && self.stream.get().is_none_or(|s|s==stream.raw()as usize),"native prefix stream");
        let rows=if (sum || gather) && live<capacity {Some(ReductionRows::new(live,capacity)?)}else{None};
        let previous_sum=self.native_sum_rows.replace(if sum {rows}else{None});
        let previous_gather=self.native_gather_rows.replace(if gather {rows}else{None});
        Ok(NativeRowScope {comm:self,previous_sum,previous_gather})
    }
    /// Shape is part of NCCL capture identity, independently of pointer capacity.
    /// Target graph owners must select a capture with this accepted row count;
    /// a previously recorded matching variant can be reused.
    pub(crate) fn select_prefill_rows(
        &self,
        live: usize,
        capacity: usize,
        stream: &Stream,
    ) -> Result<()> {
        let rows = ReductionRows::new(live, capacity)?;
        ensure!(
            self.cuda.owns_stream(stream)
                && self.stream.get().is_none_or(|s| s == stream.raw() as usize),
            "prefill row policy stream changed"
        );
        self.prefill_rows.set(Some(rows));
        Ok(())
    }
    /// Attach the independently attributed raw gather kernel to the same
    /// ordered epoch/transport. A communicator never has two independent epochs.
    pub fn attach_gather(&mut self, root: &Path) -> Result<()> {
        ensure!(self.gather.is_none(), "gather already attached");
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(
            manifest["layout"]["world"] == 4
                && manifest["layout"]["slots"] == self.layout.slots
                && manifest["layout"]["flag_stride"] == self.layout.flag_stride
                // Exported offsets are not embedded in these kernels; their
                // slot_bytes argument receives the admitted runtime layout.
                && manifest["layout"]["slot_bytes"] == 64*1024*1024,
            "gather transport layout mismatch"
        );
        let kernels = Primitives::open(
            &self.cuda,
            &root.join("libcanada_primitives.so"),
            &root.join("bindings.json"),
        )?;
        ensure!(
            kernels.bindings.entries.len() == 4,
            "gather requires four entries"
        );
        for (r, entry) in kernels.bindings.entries.iter().enumerate() {
            ensure!(
                entry.kernel_id == format!("canada.roce.row_gather.rank{r}"),
                "unexpected gather primitive"
            );
            ensure!(
                entry.arguments.iter().any(|a| a.name == "row_packs"),
                "gather missing row shape argument"
            );
        }
        self.gather = Some(kernels);
        Ok(())
    }
    /// Safety: input/output, primitives and transport remain alive until stream
    /// completion. All ranks issue an identical collective order and byte count.
    pub unsafe fn sum_bf16<T: DeviceElement>(
        &self,
        input: &Buffer<T>,
        output: &mut Buffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(output.len() == input.len(), "sum output size mismatch");
        if let Some(nccl) = self
            .prefill_sum
            .as_ref()
            .filter(|_| !self.native_sum_selected.get())
        {
            self.health()?;
            let raw = stream.raw() as usize;
            ensure!(
                self.stream.get().is_none_or(|prior| prior == raw),
                "prefill sum requires the same ordered CUDA stream"
            );
            self.stream.set(Some(raw));
            let count = self
                .prefill_rows
                .get()
                .map_or(Ok(input.len()), |rows| rows.count(input.len()))?;
            let result = if self.small_prefill_sum&&count<=128*6144 {
                // s955: native fixed-rank FP32 accumulation wins at M128,
                // but loses on larger prefill payloads. Keep the prefix and
                // padding contract of the NCCL path, using no extra scratch.
                // Captures retain this choice; decode dispatch is unchanged.
                ensure!(std::mem::size_of::<T>()==2&&count>0&&count<=input.len()
                    &&self.cuda.owns_buffer(input)&&self.cuda.owns_buffer(output)
                    &&self.cuda.owns_stream(stream),"small prefill BF16 storage/context");
                if count<output.len(){unsafe{self.cuda.zero_device(
                    output.ptr()+(count*2)as u64,(output.len()-count)*2,stream)}?;}
                let input=unsafe{input.view(0,count)}?;
                let mut output=unsafe{output.view(0,count)}?;
                self.small_prefill_enqueues.set(self.small_prefill_enqueues.get()+1);
                unsafe{self.collect(&input,&mut output,stream,None)}
            }else if let Some(exchange)=&self.exchange_prefill {
                unsafe{exchange.run(nccl,input,output,count,false,stream)}
            }else if self.precise_prefill.is_some(){
                unsafe{self.precise_prefill_sum(input,output,count,false,stream)}
            }else{unsafe { nccl.sum_bf16_prefix_to(input, output, count, stream) }};
            if result.is_err() {
                self.poisoned.set(true);
            }
            return result;
        }
        if let Some(rows)=self.native_sum_rows.get() {
            let count=rows.count(input.len())?;
            ensure!(std::mem::size_of::<T>()==2,"native prefix BF16 storage");
            // Views preserve the original allocation lifetime. Captures retain
            // pointers/counts, not the temporary Rust view objects.
            let input=unsafe{input.view(0,count)}?;
            let mut output=unsafe{output.view(0,count)}?;
            unsafe {self.collect(&input,&mut output,stream,None)}
        }else{unsafe { self.collect(input, output, stream, None) }}
    }
    /// Safety: all ranks agree on root, length and collective order and retain
    /// the span through stream completion. Used outside model graph captures.
    pub(crate) unsafe fn broadcast_image_span(&self, span: &Buffer<half::bf16>, root: usize, stream: &Stream) -> Result<()> {
        ensure!(span.len() <= 1024 * 5120,"image span extent");
        unsafe {self.broadcast_image_buffer(span,root,stream)}
    }
    /// The canonical patch processor bounds images to9198patches of588BF16
    /// values. Placement on a peer copies those bits once before encoding.
    pub(crate) unsafe fn broadcast_image_patches(&self, patches: &Buffer<half::bf16>, root: usize, stream: &Stream) -> Result<()> {
        ensure!(!patches.is_empty() && patches.len() <= 9198 * 588,"image patch extent");
        unsafe {self.broadcast_image_buffer(patches,root,stream)}
    }
    unsafe fn broadcast_image_buffer(&self, span: &Buffer<half::bf16>, root: usize, stream: &Stream) -> Result<()> {
        ensure!(self.cuda.owns_buffer(span) && self.cuda.owns_stream(stream)
            && root < 4, "image broadcast context/root");
        self.health()?;
        let raw = stream.raw() as usize;
        ensure!(self.stream.get().is_none_or(|prior| prior == raw), "image broadcast stream changed");
        self.stream.set(Some(raw));
        let nccl = self.prefill_sum.as_ref().context("image transfer requires the qualified NCCL communicator")?;
        let result = unsafe { nccl.broadcast_bf16(span, root, stream) };
        if result.is_err() { self.poisoned.set(true); }
        result
    }
    /// All-gather [M, local_columns] to [M, 4 * local_columns] without any
    /// arithmetic, preserving signed zeros, NaNs and raw BF16 bits. Same lifetime
    /// and identical-rank collective ordering requirements as sum_bf16.
    pub unsafe fn gather_bf16<T: DeviceElement>(
        &self,
        input: &Buffer<T>,
        output: &mut Buffer<T>,
        local_columns: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            std::mem::size_of::<T>() == 2
                && local_columns > 0
                && local_columns % 8 == 0
                && input.len() % local_columns == 0
                && input.len().checked_mul(4) == Some(output.len()),
            "gather input/output row shape mismatch"
        );
        ensure!(self.gather.is_some(), "gather primitive not attached");
        ensure!(self.cuda.owns_buffer(input) && self.cuda.owns_buffer(output)
            && self.cuda.owns_stream(stream), "column gather context");
        let rows = input.len() / local_columns;
        if local_columns == 6400 && rows >= 512 && !self.native_sum_selected.get()
            && let Some((module, scratch)) = &self.column_gather {
            self.health()?;
            self.bind_prefill_stream(stream)?;
            let nccl = self.prefill_sum.as_ref().ok_or_else(||anyhow::anyhow!("column gather needs NCCL"))?;
            let result = (|| {
                let mut scratch = scratch.try_borrow_mut()?;
                let live=if self.column_gather_live_rows.get()&&rows==512 {
                    self.prefill_rows.get().filter(|p|p.capacity==rows&&(129..512).contains(&p.live))
                        .map_or(rows,|p|p.live)
                }else{rows};
                let active_input=unsafe{input.view(0,live*local_columns)}?;
                unsafe { nccl.gather_bytes_prefix(&active_input, &mut scratch, stream)?; }
                if live<rows {
                    // Keep capacity-shaped consumers' unused suffix defined;
                    // this cost is included in the paired component screen.
                    let count=live*local_columns*4;
                    unsafe{self.cuda.zero_device(output.ptr()+(count*2)as u64,(output.len()-count)*2,stream)}?;
                    self.column_gather_prefix_calls.set(self.column_gather_prefix_calls.get()+1);
                }
                let (mut src, mut dst, mut r, mut c) = (scratch.ptr(), output.ptr(), live as i32, (local_columns/8) as i32);
                let mut args = [(&mut src as *mut u64).cast(),(&mut dst as *mut u64).cast(),
                    (&mut r as *mut i32).cast(),(&mut c as *mut i32).cast()];
                unsafe { module.function("ca_transpose_rank_columns")?.launch(
                    [(live*local_columns*4/8).div_ceil(256) as u32,1,1],[256,1,1],0,stream,&mut args) }
            })();
            if result.is_err() { self.poisoned.set(true); }
            return result;
        }
        unsafe { self.collect_row_prefix(input,output,local_columns,stream,local_columns/8) }
    }
    pub(crate) fn set_column_gather_rows_for_probe(&self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_PREFILL_COLUMN_GATHER_SCREEN").as_deref()==Ok("1")
            &&self.column_gather.is_some()&&self.cuda.owns_stream(stream)
            &&self.stream.get().is_none_or(|s|s==stream.raw()as usize),"finite column-gather row screen admission");
        stream.synchronize()?;self.column_gather_live_rows.set(enabled);Ok(())
    }
    pub(crate) fn column_gather_prefix_calls(&self)->usize {self.column_gather_prefix_calls.get()}
    /// One fixed workspace shared by all wide Engram gathers on this ordered
    /// stream. Must be attached before any target graph captures its address.
    pub(crate) fn attach_prefill_column_gather(&mut self, core: &[u8], rows: usize) -> Result<()> {
        ensure!(self.column_gather.is_none() && self.prefill_sum.is_some() && self.stream.get().is_none(),
            "column gather attaches before first model collective");
        ensure!([4096, 8192, 16384].contains(&rows), "column gather row capacity");
        let elements = rows * 4 * 6400;
        ensure!(self.cuda.admission_memory()?.admitted_available > elements * 2 + 2 * 1024usize.pow(3),
            "column gather workspace headroom");
        self.column_gather = Some((self.cuda.module(core)?, RefCell::new(self.cuda.allocate(elements)?)));
        Ok(())
    }
    /// Bit-preserving row-major FP32 gather for corrected DSpark logits. The
    /// attributed transport copies aligned16-byte packs without arithmetic;
    /// no reinterpretation or BF16 rounding is applied to the payload.
    pub unsafe fn gather_f32(
        &self,
        input: &Buffer<f32>,
        output: &mut Buffer<f32>,
        local_columns: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            local_columns > 0
                && local_columns % 4 == 0
                && input.len() % local_columns == 0
                && input.len().checked_mul(4) == Some(output.len())
                && self.gather.is_some(),
            "FP32 gather row shape/transport"
        );
        unsafe { self.collect_row_prefix(input,output,local_columns,stream,local_columns/4) }
    }
    /// Exact FP64 payload copy through the same aligned16-byte row-gather
    /// transport. No conversion or floating-point arithmetic is performed.
    pub(crate) unsafe fn gather_f64(
        &self,input:&Buffer<f64>,output:&mut Buffer<f64>,columns:usize,stream:&Stream,
    )->Result<()> {
        ensure!(columns>0&&columns%2==0&&input.len()%columns==0
            &&input.len().checked_mul(4)==Some(output.len())&&self.gather.is_some(),
            "FP64 bit-copy gather row shape/transport");
        unsafe{self.collect_row_prefix(input,output,columns,stream,columns/2)}
    }
    /// Bit-copy into a retained FP32 row without an intermediate output copy.
    /// The complete destination range must fit the same owned allocation.
    pub(crate) unsafe fn gather_f32_at(
        &self,
        input: &Buffer<f32>,
        output: &mut Buffer<f32>,
        output_offset: usize,
        local_columns: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(local_columns>0 && local_columns%4==0
            && input.len()%local_columns==0 && output_offset%4==0
            && input.len().checked_mul(4).and_then(|n| output_offset.checked_add(n))
                .is_some_and(|end| end<=output.len()) && self.gather.is_some(),
            "FP32 gather destination range/transport");
        unsafe { self.collect_at(input,output,stream,Some(local_columns/4),output_offset) }
    }
    unsafe fn collect_row_prefix<T: DeviceElement>(&self,input:&Buffer<T>,output:&mut Buffer<T>,
        columns:usize,stream:&Stream,row_packs:usize)->Result<()> {
        // The prefill last-logit head has one row even in an M128 target.
        // Preserve such distinct geometries; crop only the target row domain.
        if let Some(rows)=self.native_gather_rows.get().filter(|r|input.len()/columns==r.capacity) {
            let count=rows.live*columns;
            let input=unsafe{input.view(0,count)}?;
            let mut output=unsafe{output.view(0,count*4)}?;
            unsafe{self.collect(&input,&mut output,stream,Some(row_packs))}
        }else{unsafe{self.collect(input,output,stream,Some(row_packs))}}
    }
    unsafe fn collect<T: DeviceElement>(
        &self,
        input: &Buffer<T>,
        output: &mut Buffer<T>,
        stream: &Stream,
        row_packs: Option<usize>,
    ) -> Result<()> {
        unsafe { self.collect_at(input,output,stream,row_packs,0) }
    }
    unsafe fn collect_at<T: DeviceElement>(
        &self,
        input: &Buffer<T>,
        output: &mut Buffer<T>,
        stream: &Stream,
        row_packs: Option<usize>,
        output_offset: usize,
    ) -> Result<()> {
        self.health()?;
        ensure!(
            (std::mem::size_of::<T>() == 2
                || (row_packs.is_some() && [4,8].contains(&std::mem::size_of::<T>())))
                && self.cuda.owns_buffer(input)
                && self.cuda.owns_buffer(output)
                && self.cuda.owns_stream(stream),
            "RoCE reduction/gather storage/context mismatch"
        );
        let output_ptr=output_offset.checked_mul(std::mem::size_of::<T>())
            .and_then(|offset| output.ptr().checked_add(offset as u64))
            .ok_or_else(|| anyhow::anyhow!("RoCE output offset overflow"))?;
        let bytes = input.bytes();
        ensure!(
            bytes > 0 && bytes % 16 == 0 && bytes as u64 <= self.layout.slot_bytes,
            "RoCE buffer/capacity/alignment contract"
        );
        ensure!(
            input.ptr() % 16 == 0 && output_ptr % 16 == 0,
            "RoCE pointers are unaligned"
        );
        self.peak_native_bytes.set(self.peak_native_bytes.get().max(bytes));
        let raw = stream.raw() as usize;
        if let Some(prior) = self.stream.get() {
            ensure!(
                prior == raw,
                "RoCE runtime requires one ordered CUDA stream"
            );
        } else {
            self.stream.set(Some(raw));
        }
        let default_blocks=(bytes / 16).div_ceil(512).next_power_of_two().min(8);
        let blocks = if row_packs.is_some() {
            self.gather_grid_override.get().unwrap_or_else(|| {
                // s641/s1054: same raw transport, changed grids qualified on
                // all four peers, including mixed class/order changes. M32
                // benefits from16 blocks; retain M1/unmeasured geometries.
                if self.tuned_logit_gather && std::mem::size_of::<T>()==4
                    && row_packs==Some(32320/4) {
                    match bytes {775680|4136960=>16,2068480=>32,_=>default_blocks}
                }else{default_blocks}
            })
        } else { self.sum_grid_override.get().unwrap_or_else(||{
            // s648/s649: 61,440-byte (M6 hidden5120) sums benefit from four
            // blocks. Kernel, rank addition order and protocol are unchanged.
            if self.tuned_small_sum&&bytes==61440{4}else{default_blocks}
        }) };
        let class = blocks.ilog2() as u64;
        ensure!(class <= 5, "RoCE grid counter class");
        let stage_word = if class < 4 { 1 + class } else { 10 + class - 4 };
        let tail_word = if class < 4 { 5 + class } else { 12 + class - 4 };
        let kernels = if row_packs.is_some() {
            self.gather.as_ref().unwrap()
        } else {
            match self.notification_probe_variant.get() {
                0 => &self.kernels,
                n => &self.notification_probe_kernels[n - 1],
            }
        };
        let values = kernels.bindings.entries[self.rank]
            .arguments
            .iter()
            .map(|a| -> Result<u64> {
                Ok(match a.name.as_str() {
                    "input_ptr" | "input" | "inp" => input.ptr(),
                    "output_ptr" | "output" | "out" => output_ptr,
                    "size_packs" => (bytes / 16) as u64,
                    "row_packs" => {
                        row_packs.ok_or_else(|| anyhow::anyhow!("row shape missing"))? as u64
                    }
                    "nbytes" => bytes as u64,
                    "recv_base" => self.region.device_ptr() + self.layout.recv_off,
                    "flag_base" => self.region.device_ptr() + self.layout.flag_off,
                    "send_base" => self.region.device_ptr() + self.layout.send_off,
                    "ctrl_base" => self.region.device_ptr() + self.layout.ctrl_off,
                    "slot_bytes" => self.layout.slot_bytes,
                    "epoch_address" | "epoch_ptr" => self.counters.ptr(),
                    "stage_counter_address" | "stage_counter_ptr" => {
                        self.counters.ptr() + 4 * stage_word
                    }
                    "tail_counter_address" | "tail_counter_ptr" => {
                        self.counters.ptr() + 4 * tail_word
                    }
                    "poison_address" | "poison_ptr" => self.counters.ptr() + 36,
                    "spin_limit" => 20_000_000,
                    "grid_x" => blocks as u64,
                    unknown => {
                        return Err(anyhow::anyhow!("unbound RoCE primitive argument {unknown}"));
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;
        unsafe { kernels.launch_words(self.rank as i32, &values, stream) }?;
        Ok(())
    }
}
impl Roce {
    /// Isolated component variants share the exact existing proxy, counters and
    /// scalar ABI. Selection affects future captures, never an in-flight graph.
    pub(crate) fn attach_notification_probe(&mut self, root:&Path)->Result<()> {
        let mode=std::env::var("CANADA_SCRATCH_NOTIFY_SCREEN");
        ensure!(matches!(mode.as_deref(),Ok("1"|"2"))
            && self.notification_probe_kernels.is_empty(),"notification probe not admitted");
        for variant in if mode.as_deref()==Ok("2"){["control","poll","poll_fence"]}else{["control","final","stage"]} {
            let path=root.join(format!("roce-notify-{variant}-v1"));
            let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(path.join("manifest.json"))?)?;
            ensure!(manifest["variant"]==variant&&manifest["layout"]["world"]==4
                &&manifest["layout"]["slots"]==self.layout.slots
                &&manifest["layout"]["flag_stride"]==self.layout.flag_stride
                &&manifest["layout"]["slot_bytes"]==self.layout.slot_bytes,"notification layout mismatch");
            let kernels=Primitives::open(&self.cuda,&path.join("libcanada_primitives.so"),&path.join("bindings.json"))?;
            ensure!(kernels.bindings.entries.len()==4,"notification entries");
            for (rank,entry) in kernels.bindings.entries.iter().enumerate() {
                let expected=&self.kernels.bindings.entries[rank];
                ensure!(entry.kernel_id==format!("canada.roce.notify_{variant}.rank{rank}")
                    &&entry.arguments.len()==expected.arguments.len()
                    &&entry.arguments.iter().zip(&expected.arguments).all(|(a,b)|a.name==b.name&&a.kind==b.kind),
                    "notification scalar ABI differs");
            }
            self.notification_probe_kernels.push(kernels);
        }
        Ok(())
    }
    pub(crate) fn set_notification_probe(&self, variant:usize, stream:&Stream)->Result<()> {
        ensure!(matches!(std::env::var("CANADA_SCRATCH_NOTIFY_SCREEN").as_deref(),Ok("1"|"2"))
            &&self.notification_probe_kernels.len()==3&&variant<=3,"notification variant not admitted");
        stream.synchronize()?;self.health()?;self.notification_probe_variant.set(variant);Ok(())
    }
    #[cfg(target_os="linux")]
    pub(crate) fn proxy_cpus_for_probe(&self)->Result<Vec<usize>> {
        let tid=self.proxy_thread_for_probe.get().ok_or_else(||anyhow::anyhow!("proxy affinity probe not armed"))?;
        let mut mask:libc::cpu_set_t=unsafe{std::mem::zeroed()};
        ensure!(unsafe{libc::sched_getaffinity(tid,std::mem::size_of_val(&mask),&mut mask)}==0,
            "read proxy affinity: {}",std::io::Error::last_os_error());
        Ok((0..libc::CPU_SETSIZE as usize).filter(|&cpu|unsafe{libc::CPU_ISSET(cpu,&mask)}).collect())
    }
    #[cfg(not(target_os="linux"))]
    pub(crate) fn proxy_cpus_for_probe(&self)->Result<Vec<usize>> {anyhow::bail!("Linux proxy affinity probe")}
    #[cfg(target_os="linux")]
    pub(crate) fn set_proxy_cpus_for_probe(&self,cpus:&[usize],stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_PROXY_AFFINITY_SCREEN").as_deref()==Ok("1")
            && !cpus.is_empty() && cpus.iter().all(|&c|c<(libc::CPU_SETSIZE as usize))
            && self.cuda.owns_stream(stream),"isolated proxy affinity domain");
        stream.synchronize()?;self.health()?;
        let tid=self.proxy_thread_for_probe.get().ok_or_else(||anyhow::anyhow!("missing proxy thread"))?;
        let mut mask:libc::cpu_set_t=unsafe{std::mem::zeroed()};
        unsafe{libc::CPU_ZERO(&mut mask);for &cpu in cpus{libc::CPU_SET(cpu,&mut mask);}}
        ensure!(unsafe{libc::sched_setaffinity(tid,std::mem::size_of_val(&mask),&mask)}==0,
            "set proxy affinity: {}",std::io::Error::last_os_error());
        let mut expected=cpus.to_vec();expected.sort_unstable();expected.dedup();
        ensure!(self.proxy_cpus_for_probe()?==expected,"proxy affinity did not apply");Ok(())
    }
    #[cfg(not(target_os="linux"))]
    pub(crate) fn set_proxy_cpus_for_probe(&self,_cpus:&[usize],_stream:&Stream)->Result<()> {anyhow::bail!("Linux proxy affinity probe")}
    /// Same isolated geometry control for BF16 sums. Arithmetic is unchanged.
    pub(crate) fn set_sum_grid_for_probe(&self, blocks:usize, stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_SUM_GRID_SCREEN").as_deref()==Ok("1")
            && [1,2,4,8,16].contains(&blocks) && self.cuda.owns_stream(stream)
            && self.stream.get().is_none_or(|s|s==stream.raw()as usize),
            "isolated sum grid control");
        stream.synchronize()?;self.health()?;
        self.sum_grid_override.set(Some(blocks));Ok(())
    }
    /// Isolated capture-time gather geometry control. Existing graphs retain
    /// their grid and disjoint counter class; reductions keep their geometry.
    pub(crate) fn set_gather_grid_for_probe(&self, blocks:usize, stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_LOGIT_GATHER_GRID_SCREEN").as_deref()==Ok("1")
            && [8,16,32].contains(&blocks) && self.cuda.owns_stream(stream)
            && self.stream.get().is_none_or(|s|s==stream.raw()as usize),
            "isolated gather grid control");
        stream.synchronize()?;self.health()?;
        self.gather_grid_override.set(Some(blocks));Ok(())
    }
}
fn rail_configuration(hcas: &str, addresses: &str, gid_index: i32)
    -> Result<(Vec<CString>, Vec<std::net::Ipv4Addr>)>
{
    let parts: Vec<_> = hcas.split(',').collect();
    ensure!(parts.len() == 2 && parts[0] != parts[1]
        && parts.iter().all(|p| !p.is_empty() && p.bytes().all(|c|
            c.is_ascii_alphanumeric() || c == b'_' || c == b'-')),
        "two distinct HCA names are required");
    ensure!((0..=255).contains(&gid_index), "GID index outside supported range");
    let ips: Vec<std::net::Ipv4Addr> = addresses.split(',')
        .map(str::parse).collect::<std::result::Result<_, _>>()?;
    ensure!(ips.len() == 2 && ips[0] != ips[1]
        && ips.iter().all(|ip| !ip.is_unspecified() && !ip.is_loopback()
            && !ip.is_multicast() && !ip.is_broadcast()),
        "two distinct fabric IPv4 addresses are required");
    Ok((parts.into_iter().map(CString::new).collect::<std::result::Result<_, _>>()?, ips))
}
fn proxy_thread_ids()->Result<std::collections::BTreeSet<i32>> {
    std::fs::read_dir("/proc/self/task")?.map(|entry|{
        Ok(entry?.file_name().to_str().ok_or_else(||anyhow::anyhow!("non-UTF8 task ID"))?.parse()?)
    }).collect()
}
impl Drop for Roce {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.handle);
        }
    }
}

impl Roce {
    pub(crate) fn attach_native_prefill_rs(&mut self,root:&Path,stream:&Stream)->Result<()> {
        ensure!(self.native_prefill_rs.is_none()&&self.prefill_sum.is_some(),"prefill RS attach state");
        let rs=crate::prefill_rs::ReduceScatter::new(&self.cuda,root,self.rank)?;
        let bytes=rs.blob()?;
        let mut local=self.cuda.allocate::<u8>(bytes.len())?;
        let mut all=self.cuda.allocate::<u8>(4*bytes.len())?;
        local.upload(&bytes)?;stream.join_default()?;
        unsafe{self.gather_rows(&local,&mut all,stream)}?;stream.synchronize()?;
        rs.connect(&all.download()?)?;
        ensure!(self.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3),
            "prefill RS consumed runtime reserve");
        self.native_prefill_rs=Some(rs);Ok(())
    }
    pub(crate) fn attach_moe_pipeline(&mut self,root:&Path,stream:&Stream)->Result<()> {
        ensure!(self.moe_pipeline.is_none()&&self.prefill_sum.is_some(),"pipeline attach state");
        let pipeline=crate::moe_transport::Pipeline::new(&self.cuda,root,self.rank)?;
        let bytes=pipeline.blob()?;
        let mut local=self.cuda.allocate::<u8>(bytes.len())?;
        let mut all=self.cuda.allocate::<u8>(4*bytes.len())?;
        local.upload(&bytes)?;stream.join_default()?;
        unsafe{self.gather_rows(&local,&mut all,stream)}?;stream.synchronize()?;
        pipeline.connect(&all.download()?)?;
        self.moe_pipeline=Some(pipeline);Ok(())
    }
    pub(crate) fn set_moe_pipeline_for_probe(&self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(self.cuda.owns_stream(stream)&&self.moe_pipeline.is_some(),"pipeline policy state");
        stream.synchronize()?;self.moe_pipeline.as_ref().unwrap().health()?;
        self.use_moe_pipeline.set(enabled);Ok(())
    }
    pub(crate) fn moe_pipeline_active(&self)->bool {self.use_moe_pipeline.get()}
    pub(crate) fn pipeline_moe_input(&self,input:&Buffer<half::bf16>,packed:&Buffer<u8>,
        q:&Buffer<u8>,sr:&Buffer<u8>,sm:&Buffer<u8>,stream:&Stream,metadata:impl FnOnce()->Result<()>)->Result<()> {
        ensure!(self.use_moe_pipeline.get()&&!self.native_sum_selected.get(),"pipeline requires selected prefill phase");
        self.health()?;self.bind_prefill_stream(stream)?;
        self.moe_pipeline.as_ref().ok_or_else(||anyhow::anyhow!("pipeline unavailable"))?
            .enqueue(input,packed,q,sr,sm,stream,metadata)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn explicit_rails_reject_invalid_or_ambiguous_configurations() {
        assert!(super::rail_configuration("rail_a,rail_b", "192.0.2.10,198.51.100.10", 3).is_ok());
        for (names, addresses, gid) in [
            ("rail_a,rail_a", "192.0.2.10,198.51.100.10", 3),
            ("../rail_a,rail_b", "192.0.2.10,198.51.100.10", 3),
            ("rail_a,rail_b", "192.0.2.10,192.0.2.10", 3),
            ("rail_a,rail_b", "127.0.0.1,198.51.100.10", 3),
            ("rail_a,rail_b", "192.0.2.10", 3),
            ("rail_a,rail_b", "192.0.2.10,198.51.100.10", -1),
        ] { assert!(super::rail_configuration(names, addresses, gid).is_err()); }
    }

    use super::ReductionRows;
    #[test]
    fn reduction_count_tracks_accepted_rows_and_rejects_invalid_storage() {
        for capacity in [5usize, 6, 16, 32, 128] {
            for live in 1..=capacity {
                let rows = ReductionRows::new(live, capacity).unwrap();
                for width in [1024usize, 5120, 6144] {
                    assert_eq!(rows.count(capacity * width).unwrap(), live * width);
                }
                assert!(rows.count(0).is_err());
                assert!(rows.count(capacity * 5120 + 1).is_err());
            }
        }
        assert!(ReductionRows::new(0, 6).is_err());
        assert!(ReductionRows::new(7, 6).is_err());
        assert!(ReductionRows::new(4, 4).is_err());
    }
}
