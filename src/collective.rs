//! Persistent NCCL communicator. Collective order is part of the model graph.
use crate::cuda::{Buffer, Cuda, DeviceElement, Stream};
use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use std::{
    ffi::{CStr, c_char, c_int, c_void},
    ptr,
};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct UniqueId {
    pub bytes: [u8; 128],
}

pub struct Nccl {
    _lib: Library,
    get_id: unsafe extern "C" fn(*mut UniqueId) -> c_int,
    init: unsafe extern "C" fn(*mut *mut c_void, c_int, UniqueId, c_int) -> c_int,
    abort: unsafe extern "C" fn(*mut c_void) -> c_int,
    all_reduce: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        c_int,
        c_int,
        *mut c_void,
        *mut c_void,
    ) -> c_int,
    all_gather: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        c_int,
        *mut c_void,
        *mut c_void,
    ) -> c_int,
    reduce_scatter: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        c_int,
        c_int,
        *mut c_void,
        *mut c_void,
    ) -> c_int,
    broadcast: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        c_int,
        c_int,
        *mut c_void,
        *mut c_void,
    ) -> c_int,
    get_error: unsafe extern "C" fn(c_int) -> *const c_char,
    async_error: unsafe extern "C" fn(*mut c_void, *mut c_int) -> c_int,
    group_start: unsafe extern "C" fn() -> c_int,
    group_end: unsafe extern "C" fn() -> c_int,
    send:unsafe extern "C" fn(*const c_void,usize,c_int,c_int,*mut c_void,*mut c_void)->c_int,
    recv:unsafe extern "C" fn(*mut c_void,usize,c_int,c_int,*mut c_void,*mut c_void)->c_int,
}
impl Nccl {
    pub fn load() -> Result<Self> {
        Self::load_from("libnccl.so.2")
    }
    /// The qualified eager-prefill ProcessGroupNCCL links this exact image
    /// dependency, not the independently installed system NCCL. Different
    /// versions/topologies may select different BF16 reduction schedules.
    pub fn load_qualified_prefill() -> Result<Self> {
        Self::load_from("/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2")
    }
    fn load_from(path: &str) -> Result<Self> {
        let lib = unsafe { Library::new(path) }.context("load NCCL")?;
        macro_rules! sym {
            ($name:literal) => {
                *unsafe { lib.get(concat!($name, "\0").as_bytes()) }.context($name)?
            };
        }
        Ok(Self {
            get_id: sym!("ncclGetUniqueId"),
            init: sym!("ncclCommInitRank"),
            abort: sym!("ncclCommAbort"),
            all_reduce: sym!("ncclAllReduce"),
            all_gather: sym!("ncclAllGather"),
            reduce_scatter: sym!("ncclReduceScatter"),
            broadcast: sym!("ncclBroadcast"),
            get_error: sym!("ncclGetErrorString"),
            async_error: sym!("ncclCommGetAsyncError"),
            group_start: sym!("ncclGroupStart"),
            group_end: sym!("ncclGroupEnd"),
            send:sym!("ncclSend"),recv:sym!("ncclRecv"),
            _lib: lib,
        })
    }
    fn check(&self, code: i32, op: &str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let p = unsafe { (self.get_error)(code) };
        let text = if p.is_null() {
            "unknown".into()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy()
        };
        bail!("{op}: {text} ({code})")
    }
    pub fn unique_id(&self) -> Result<UniqueId> {
        let mut id = UniqueId { bytes: [0; 128] };
        self.check(unsafe { (self.get_id)(&mut id) }, "NCCL unique ID")?;
        Ok(id)
    }
    pub fn join(
        self,
        cuda: &Cuda,
        id: UniqueId,
        rank: usize,
        world: usize,
    ) -> Result<Communicator> {
        ensure!(world == 4 && rank < world, "expected four peer ranks");
        cuda.qualify_gb10()?;
        let mut raw = ptr::null_mut();
        self.check(
            unsafe { (self.init)(&mut raw, world as i32, id, rank as i32) },
            "join NCCL ranks",
        )?;
        Ok(Communicator {
            api: self,
            raw,
            rank,
            world,
            _context: cuda.clone(),
        })
    }
}

pub struct Communicator {
    api: Nccl,
    raw: *mut c_void,
    pub rank: usize,
    pub world: usize,
    _context: Cuda,
}
impl Communicator {
    /// Byte-preserving all-to-all, the public NCCL grouped send/recv pattern:
    /// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html
    /// Each destination receives its quarter from all four source ranks.
    /// The caller reduces these BF16 contributions in FP32 after group end.
    pub(crate) unsafe fn exchange_bf16_quarters<T:DeviceElement>(&self,input:&Buffer<T>,
        scratch:&mut Buffer<u16>,count:usize,stream:&Stream)->Result<()> {
        ensure!(std::mem::size_of::<T>()==2&&count>0&&count%4==0&&count<=input.len()
            &&count<=scratch.len()&&self._context.owns_buffer(input)&&self._context.owns_buffer(scratch)
            &&self._context.owns_stream(stream),"BF16 quarter exchange geometry/owner");
        let n=count/4;let offset=(self.rank*n*2)as u64;
        self.check_async()?;
        unsafe{self._context.copy_device(scratch.ptr()+offset,input.ptr()+offset,n*2,stream)}?;
        self.api.check(unsafe{(self.api.group_start)()},"exchange group start")?;
        let enqueue=(||->Result<()>{
            for peer in (0..4).filter(|&r|r!=self.rank) {
                self.api.check(unsafe{(self.api.send)((input.ptr()+(peer*n*2)as u64)as *const c_void,
                    n,9,peer as i32,self.raw,stream.raw())},"BF16 quarter send")?;
                self.api.check(unsafe{(self.api.recv)((scratch.ptr()+(peer*n*2)as u64)as *mut c_void,
                    n,9,peer as i32,self.raw,stream.raw())},"BF16 quarter receive")?;
            }Ok(())
        })();
        let closed=self.api.check(unsafe{(self.api.group_end)()},"exchange group end");
        enqueue.and(closed)
    }
    /// Aggregate three independent rank-major gathers without repacking their
    /// payloads. Uses NVIDIA's public group ABI (NCCL user-guide, Group Calls).
    /// Safety: every rank uses the same order and sizes; all six nonoverlapping
    /// buffers and the communicator outlive completion on the same stream.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn gather_rows_grouped<A:DeviceElement,B:DeviceElement,C:DeviceElement>(
        &self,a:&Buffer<A>,ga:&mut Buffer<A>,b:&Buffer<B>,gb:&mut Buffer<B>,
        c:&Buffer<C>,gc:&mut Buffer<C>,stream:&Stream)->Result<()> {
        fn describe<T:DeviceElement>(cuda:&Cuda,world:usize,local:&Buffer<T>,global:&Buffer<T>)
            ->Result<(u64,u64,usize,i32)> {
            let datatype=match std::mem::size_of::<T>(){1=>1,2=>9,4=>7,_=>bail!("grouped gather element width")};
            ensure!(local.len()>0 && local.len().checked_mul(world)==Some(global.len())
                &&cuda.owns_buffer(local)&&cuda.owns_buffer(global),"grouped gather storage/context");
            Ok((local.ptr(),global.ptr(),local.len(),datatype))
        }
        ensure!(self._context.owns_stream(stream),"grouped gather stream context");
        let descriptors=[describe(&self._context,self.world,a,ga)?,
            describe(&self._context,self.world,b,gb)?,describe(&self._context,self.world,c,gc)?];
        self.check_async()?;
        self.api.check(unsafe{(self.api.group_start)()},"NCCL group start")?;
        // Always close a successfully opened group, including enqueue errors.
        // Nothing is synchronized or consumed until the group has been closed.
        let enqueue=(||->Result<()>{
            for (source,destination,count,datatype) in descriptors {
                self.api.check(unsafe{(self.api.all_gather)(source as *const c_void,
                    destination as *mut c_void,count,datatype,self.raw,stream.raw())},"grouped row gather")?;
            }
            Ok(())
        })();
        let finish=self.api.check(unsafe{(self.api.group_end)()},"NCCL group end");
        enqueue.and(finish)
    }
    /// Diagnostic FP32 reduction. Callers preserve BF16 source values exactly
    /// when promoting them, then round only the completed sum back to BF16.
    /// Safety: identical order/count on every rank; all storage and this
    /// communicator outlive stream completion. This is not a serving policy.
    pub(crate) unsafe fn sum_f32_to(
        &self, input: &Buffer<f32>, output: &mut Buffer<f32>, stream: &Stream,
    ) -> Result<()> {
        ensure!(input.len() > 0 && input.len() == output.len()
            && self._context.owns_buffer(input) && self._context.owns_buffer(output)
            && self._context.owns_stream(stream), "NCCL FP32 storage/context mismatch");
        self.check_async()?;
        // ncclFloat32=7, ncclSum=0 in NVIDIA's public nccl.h ABI.
        self.api.check(unsafe { (self.api.all_reduce)(
            input.ptr() as *const c_void, output.ptr() as *mut c_void,
            input.len(), 7, 0, self.raw, stream.raw(),
        ) }, "FP32 all-reduce")
    }
    /// Optional higher-precision prefill counterpart of the BF16 row scatter.
    /// Callers promote canonical BF16 operands exactly and round after the sum.
    pub(crate) unsafe fn reduce_scatter_f32(
        &self,input:&Buffer<f32>,output:&mut Buffer<f32>,stream:&Stream,
    )->Result<()> {
        ensure!(input.len()>0&&input.len()==output.len()*self.world
            &&self._context.owns_buffer(input)&&self._context.owns_buffer(output)
            &&self._context.owns_stream(stream),"NCCL FP32 scatter storage/context mismatch");
        self.check_async()?;
        self.api.check(unsafe{(self.api.reduce_scatter)(input.ptr()as *const c_void,
            output.ptr()as *mut c_void,output.len(),7,0,self.raw,stream.raw())},"FP32 reduce-scatter")
    }
    pub fn check_async(&self) -> Result<()> {
        let mut code = 0;
        self.api.check(
            unsafe { (self.api.async_error)(self.raw, &mut code) },
            "query NCCL asynchronous error",
        )?;
        self.api.check(code, "NCCL asynchronous operation")
    }
    /// Safety: all four ranks must issue this collective in the same sequence,
    /// and buffers must be exclusively owned until stream completion.
    pub unsafe fn sum_bf16(&self, buffer: &mut Buffer<u16>, stream: &Stream) -> Result<()> {
        self.api.check(
            unsafe {
                (self.api.all_reduce)(
                    buffer.ptr() as *const c_void,
                    buffer.ptr() as *mut c_void,
                    buffer.len(),
                    9,
                    0,
                    self.raw,
                    stream.raw(),
                )
            },
            "BF16 all-reduce",
        )
    }
    /// Out-of-place form for independent model-owned BF16 layer buffers.
    /// Safety: all ranks issue identical collective order and the buffers and
    /// communicator outlive queued GPU work. No host numeric fallback.
    pub unsafe fn sum_bf16_to<T: DeviceElement>(
        &self,
        input: &Buffer<T>,
        output: &mut Buffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        unsafe { self.sum_bf16_prefix_to(input, output, input.len(), stream) }
    }
    /// Reduce only the accepted row prefix. NCCL's BF16 reduction schedule can
    /// depend on count; including capacity padding changes valid-row results.
    /// Zero the unused suffix so capacity-shaped downstream kernels never
    /// consume uninitialized padding. Safety and lifetime rules
    /// are identical to sum_bf16_to, with the same count on every rank.
    pub(crate) unsafe fn sum_bf16_prefix_to<T: DeviceElement>(
        &self,
        input: &Buffer<T>,
        output: &mut Buffer<T>,
        count: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            std::mem::size_of::<T>() == 2
                && count > 0
                && count <= input.len()
                && input.len() == output.len()
                && self._context.owns_buffer(input)
                && self._context.owns_buffer(output)
                && self._context.owns_stream(stream),
            "NCCL BF16 storage/context mismatch"
        );
        self.check_async()?;
        if count < output.len() {
            unsafe {
                self._context.zero_device(
                    output.ptr() + (count * 2) as u64,
                    (output.len() - count) * 2,
                    stream,
                )?;
            }
        }
        self.api.check(
            unsafe {
                (self.api.all_reduce)(
                    input.ptr() as *const c_void,
                    output.ptr() as *mut c_void,
                    count,
                    9,
                    0,
                    self.raw,
                    stream.raw(),
                )
            },
            "out-of-place BF16 all-reduce",
        )
    }
    /// Row-sharded sum for sequence-parallel prefill: rank r receives rows
    /// [r*M/W, (r+1)*M/W) of the BF16 sum of every rank's [M, width] input.
    /// Safety: identical order/sizes on every rank; buffers outlive the stream.
    pub(crate) unsafe fn reduce_scatter_bf16<T: DeviceElement>(
        &self,
        input: &Buffer<T>,
        output: &mut Buffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            std::mem::size_of::<T>() == 2
                && input.len() == output.len() * self.world
                && self._context.owns_buffer(input)
                && self._context.owns_buffer(output)
                && self._context.owns_stream(stream),
            "NCCL BF16 reduce-scatter storage/context mismatch"
        );
        self.check_async()?;
        self.api.check(
            unsafe {
                (self.api.reduce_scatter)(
                    input.ptr() as *const c_void,
                    output.ptr() as *mut c_void,
                    output.len(),
                    9,
                    0,
                    self.raw,
                    stream.raw(),
                )
            },
            "BF16 reduce-scatter",
        )
    }
    /// Byte-preserving all-gather into a reusable buffer whose capacity can
    /// exceed this call's payload. Supports activation and selected-ID storage.
    pub(crate) unsafe fn gather_bytes_prefix<T: DeviceElement, G: DeviceElement>(
        &self, local: &Buffer<T>, global: &mut Buffer<G>, stream: &Stream,
    ) -> Result<()> {
        ensure!([2,4].contains(&std::mem::size_of::<T>()) && local.bytes() > 0
            && local.bytes().checked_mul(self.world).is_some_and(|n| n <= global.bytes())
            && self._context.owns_buffer(local) && self._context.owns_buffer(global)
            && self._context.owns_stream(stream), "NCCL byte gather capacity/context");
        self.check_async()?;
        self.api.check(unsafe { (self.api.all_gather)(local.ptr() as *const c_void,
            global.ptr() as *mut c_void,local.bytes(),1,self.raw,stream.raw()) },"byte all-gather")
    }

    /// Row-major all-gather of equally sized 8/16/32-bit row shards (bit copy).
    pub(crate) unsafe fn gather_rows<T: DeviceElement>(
        &self,
        local: &Buffer<T>,
        global: &mut Buffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        // ncclUint8 for quantized bytes, ncclBfloat16 for 16-bit storage,
        // ncclFloat32 for FP32 coefficients. All-gather performs no arithmetic.
        let datatype = match std::mem::size_of::<T>() {
            1 => 1,
            2 => 9,
            4 => 7,
            _ => bail!("row gather element width"),
        };
        ensure!(
            global.len() == local.len() * self.world
                && self._context.owns_buffer(local)
                && self._context.owns_buffer(global)
                && self._context.owns_stream(stream),
            "NCCL row gather storage/context mismatch"
        );
        self.check_async()?;
        self.api.check(
            unsafe {
                (self.api.all_gather)(
                    local.ptr() as *const c_void,
                    global.ptr() as *mut c_void,
                    local.len(),
                    datatype,
                    self.raw,
                    stream.raw(),
                )
            },
            "row all-gather",
        )
    }
    /// Safety: collective order, device context and asynchronous buffer lifetimes
    /// must agree on every rank. The gather output is rank-major.
    pub unsafe fn gather_bf16(
        &self,
        local: &Buffer<u16>,
        global: &mut Buffer<u16>,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            global.len()
                == local
                    .len()
                    .checked_mul(self.world)
                    .context("gather length overflow")?,
            "gather output size"
        );
        self.api.check(
            unsafe {
                (self.api.all_gather)(
                    local.ptr() as *const c_void,
                    global.ptr() as *mut c_void,
                    local.len(),
                    9,
                    self.raw,
                    stream.raw(),
                )
            },
            "BF16 all-gather",
        )
    }
    /// Safety: every rank calls with the same root/length/order; the buffer is
    /// pinned to the communicator's device and outlives queued GPU work.
    pub unsafe fn broadcast_tokens(
        &self,
        tokens: &mut Buffer<i32>,
        root: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(root < self.world, "invalid broadcast root");
        self.api.check(
            unsafe {
                (self.api.broadcast)(
                    tokens.ptr() as *const c_void,
                    tokens.ptr() as *mut c_void,
                    tokens.len(),
                    2,
                    root as i32,
                    self.raw,
                    stream.raw(),
                )
            },
            "broadcast tokens",
        )
    }
    /// Safety: identical root/count/order on every rank, same device and
    /// stream ownership, and storage retained through completion. Broadcast
    /// transports BF16 bits without any arithmetic or quantization.
    pub unsafe fn broadcast_bf16(&self, buffer: &Buffer<half::bf16>, root: usize, stream: &Stream) -> Result<()> {
        ensure!(root < self.world, "invalid BF16 broadcast root");
        self.api.check(unsafe { (self.api.broadcast)(buffer.ptr() as *const c_void,
            buffer.ptr() as *mut c_void, buffer.len(), 9, root as i32, self.raw, stream.raw()) },
            "broadcast BF16 image span")
    }
}
impl Drop for Communicator {
    fn drop(&mut self) {
        unsafe {
            (self.api.abort)(self.raw);
        }
    }
}
