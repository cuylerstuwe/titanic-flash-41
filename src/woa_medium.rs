//! Medium-prefill WO-A through direct vendor BF16 GEMM. Preserve native raw
//! checkpoint bytes; reconstruct exact BF16 operands on GPU into a shared,
//! bounded scratch, with immutable decoded weights shared across profiles.
//! One ordered stream owns temporary scratch across all decoder layers.
use crate::{
    blas::Blas,
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, Module, Stream},
    weights::Weights,
    triton_ops::Operator,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    rc::{Rc, Weak},
    path::Path,
};

// Immutable derived operands retain the canonical owner. Weak cache entries
// cannot extend a model lifetime or alias a later allocation at the same address.
struct DecodedWeight {
    values: Buffer<bf16>,
    _weights: Rc<Weights>,
}
thread_local! {
    static DECODED: RefCell<BTreeMap<(usize, u64, u64), Weak<DecodedWeight>>> = RefCell::new(BTreeMap::new());
}
pub(crate) fn cached_policy() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_WOA_CACHE") {
        Ok(v) => { ensure!(v == "0" || v == "1", "invalid WO-A operand cache policy"); Ok(v == "1") }
        // s655 component and s657 TP4 native pairs preserve exact operands,
        // logits and state; 640MiB/rank buys 3–5% less M8/M16 verification time.
        // Explicit 0 retains the uncached control under tighter memory budgets.
        Err(std::env::VarError::NotPresent) => Ok(true),
        Err(e) => Err(e.into()),
    }
}
/// Direct token-major per-group WO-A GEMMs for full (non-bounded) medium launches.
/// Resolved once from CANADA_SCRATCH_WOA_TOKEN_MAJOR; the Claude A/B probe may override.
static TOKEN_MAJOR: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
pub(crate) fn token_major_enabled()->bool {
    use std::sync::atomic::Ordering::Relaxed;
    match TOKEN_MAJOR.load(Relaxed) {
        0=>false,1=>true,
        _=>{let on=std::env::var("CANADA_SCRATCH_WOA_TOKEN_MAJOR").as_deref()==Ok("1");
            TOKEN_MAJOR.store(u8::from(on),Relaxed);on}
    }
}
pub(crate) fn set_token_major_for_probe(enabled:bool) {
    TOKEN_MAJOR.store(u8::from(enabled),std::sync::atomic::Ordering::Relaxed);
}
pub(crate) fn compact_policy()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_WOA_MID_COMPACT") {
        Ok(v)=>{ensure!(v=="0"||v=="1","invalid compact medium WO-A policy");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(false),Err(e)=>Err(e.into()),
    }
}
pub(crate) fn prefill_live_policy()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_WOA_PREFILL_LIVE") {
        Ok(v)=>{ensure!(v=="0"||v=="1","invalid WO-A prefill row policy");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(false),Err(e)=>Err(e.into()),
    }
}
thread_local! {
    // Enqueue/capture-only bound. A graph owns its recorded geometry; this
    // cannot affect another thread, stream, profile or decoder-tail replay.
    static PREFILL_ROWS: Cell<Option<(usize,usize)>> = const { Cell::new(None) };
}
pub(crate) struct LiveRowsScope(std::marker::PhantomData<Rc<()>>);
impl LiveRowsScope {
    pub(crate) fn new(enabled:bool,live:usize,capacity:usize,stream:&Stream)->Result<Self> {
        PREFILL_ROWS.with(|slot|->Result<()> {
            ensure!(slot.get().is_none(),"nested WO-A prefill row scope");
            if enabled {
                ensure!(capacity==128&&(33..128).contains(&live),"WO-A prefill row geometry");
                // The component screen found cuBLAS's M33 plan slower hot;
                // the aligned64 plan improved both hot and displaced runs.
                let compute=if live<=64 {live.div_ceil(32)*32}else{live};
                slot.set(Some((compute,stream.raw()as usize)));
            }
            Ok(())
        })?;
        Ok(Self(std::marker::PhantomData))
    }
}
impl Drop for LiveRowsScope {
    fn drop(&mut self){PREFILL_ROWS.with(|slot|slot.set(None));}
}
struct Compact {
    partial_op:Operator,
    reduce_op:Operator,
    partial:Buffer<f32>,
    reference_op:Option<Operator>,
}

pub struct Scratch {
    cuda: Cuda,
    module: Module,
    blas: Blas,
    rows: usize,
    decoded: Buffer<bf16>,
    grouped: Buffer<bf16>,
    stream: Cell<Option<usize>>,
    cache_enabled: Cell<bool>,
    compact: Option<Compact>,
    compact_enabled: Cell<bool>,
    reference_enabled: Cell<bool>,
}
impl Scratch {
    pub fn new(cuda: &Cuda, core: &[u8], rows: usize) -> Result<Rc<RefCell<Self>>> {
        Self::with_blas(cuda,core,rows,Blas::new(cuda)?)
    }
    pub(crate) fn new_on(cuda:&Cuda,core:&[u8],rows:usize,stream:&Stream)->Result<Rc<RefCell<Self>>> {
        Self::with_blas(cuda,core,rows,Blas::new_on(cuda,stream)?)
    }
    fn with_blas(cuda:&Cuda,core:&[u8],rows:usize,blas:Blas)->Result<Rc<RefCell<Self>>> {
        ensure!(
            (9..=384).contains(&rows) || crate::config::wide_rows(rows),
            "medium WO-A rows"
        );
        Ok(Rc::new(RefCell::new(Self {
            cuda: cuda.clone(),
            module: cuda.module(core)?,
            blas,
            rows,
            decoded: cuda.allocate(2048 * 4096)?,
            grouped: cuda.allocate(rows * 2048)?,
            stream: Cell::new(None),
            cache_enabled: Cell::new(true),
            compact:None,
            compact_enabled:Cell::new(false),
            reference_enabled:Cell::new(false),
        })))
    }
    pub fn owned_scratch_bytes(&self) -> usize {
        self.decoded.bytes() + self.grouped.bytes() + 32 * 1024 * 1024
            +self.compact.as_ref().map_or(0,|c|c.partial.bytes())
    }
}
pub struct Grouped {
    cuda: Cuda,
    _weights: Rc<Weights>,
    weight: u64,
    scales: u64,
    scratch: Rc<RefCell<Scratch>>,
    output: Buffer<bf16>,
    rows: usize,
    cached: Option<Rc<DecodedWeight>>,
}
impl Grouped {
    pub(crate) fn prepare_compact(&self,kernels:&Path,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(matches!(self.rows,16|32)&&self.cuda.owns_stream(stream),"compact WO-A profile/context");
        let mut s=self.scratch.try_borrow_mut()?;
        ensure!(s.stream.get().is_none(),"prepare compact WO-A before first execution");
        if s.compact.is_none() {
            let root=kernels.join("woa-mid-compact-v1");
            let partial_op=Operator::load(&self.cuda,&root,&format!("woa_mid_partial_m{}",self.rows))?;
            let reduce_op=Operator::load(&self.cuda,&root,&format!("woa_mid_reduce_m{}",self.rows))?;
            ensure!(partial_op.entry.arguments.iter().map(|a|a.kind.as_str()).collect::<Vec<_>>()
                ==["*bf16","*fp8e4nv","*u8","*fp32"]
                &&reduce_op.entry.arguments.iter().map(|a|a.kind.as_str()).collect::<Vec<_>>()
                ==["*fp32","*bf16"],"compact WO-A ABI");
            let sk=if self.rows==16{4}else{2};
            ensure!((partial_op.entry.grid.x_mul,partial_op.entry.grid.x_add,partial_op.entry.grid.y,partial_op.entry.grid.z)
                ==(0,if self.rows==16{16}else{32},2,sk)
                &&(reduce_op.entry.grid.rows_div,reduce_op.entry.grid.x_mul,reduce_op.entry.grid.x_add,reduce_op.entry.grid.y,reduce_op.entry.grid.z)
                ==(1,8,0,1,1),"compact WO-A grid");
            let mut partial=self.cuda.allocate(sk as usize*self.rows*2048)?;partial.zero()?;
            let reference_op=if std::env::var("CANADA_SCRATCH_WOA_MID_REFERENCE_SCREEN").as_deref()==Ok("1") {
                let op=Operator::load(&self.cuda,&kernels.join("woa-mid-reference-v1"),
                    &format!("woa_fp64_m{}",self.rows))?;
                ensure!(op.entry.arguments.iter().map(|a|a.kind.as_str()).collect::<Vec<_>>()
                    ==["*bf16","*bf16","*bf16"],"WO-A diagnostic reference ABI");
                ensure!((op.entry.grid.rows_div,op.entry.grid.x_mul,op.entry.grid.x_add,op.entry.grid.y,op.entry.grid.z)
                    ==(1,256,0,1,1),"WO-A diagnostic reference grid");Some(op)
            }else{None};
            s.compact=Some(Compact{partial_op,reduce_op,partial,reference_op});
        }
        s.compact_enabled.set(enabled);Ok(())
    }
    pub(crate) fn set_compact_mode_for_probe(&self,mode:u8)->Result<()> {
        let s=self.scratch.try_borrow()?;
        ensure!(mode<=2&&s.compact.is_some(),"compact WO-A probe requires prepared shared scratch");
        ensure!(mode!=2||s.compact.as_ref().unwrap().reference_op.is_some(),"WO-A reference must be prepared");
        s.compact_enabled.set(mode==1);s.reference_enabled.set(mode==2);Ok(())
    }
    pub fn new(
        cuda: &Cuda,
        weights: Rc<Weights>,
        layer: usize,
        scratch: Rc<RefCell<Scratch>>,
    ) -> Result<Self> {
        ensure!(
            layer < 40 && weights.in_context(cuda),
            "medium WO-A weight owner"
        );
        let prefix = format!("layers.{layer}.attn.wo_a");
        let weight = weights.view(&format!("{prefix}.weight"))?;
        let scales = weights.view(&format!("{prefix}.scale"))?;
        ensure!(
            weight.dtype == Dtype::Fp8
                && weight.shape == [2048, 4096]
                && scales.dtype == Dtype::E8m0
                && scales.shape == [64, 128],
            "medium WO-A canonical shape/precision"
        );
        let rows = {
            let s = scratch.try_borrow()?;
            ensure!(cuda.owns_buffer(&s.decoded), "medium WO-A scratch context");
            s.rows
        };
        Ok(Self {
            cuda: cuda.clone(),
            weight: weights.ptr(&format!("{prefix}.weight"))?,
            scales: weights.ptr(&format!("{prefix}.scale"))?,
            _weights: weights,
            scratch,
            rows,
            output: cuda.allocate_rows(rows * 2048)?,
            cached: None,
        })
    }
    /// Prepare outside capture. Identical canonical operands share one immutable
    /// allocation across execution profiles; the preparation stream completes
    /// before any other profile may consume it. No arithmetic change in run().
    pub fn prepare_cached(&mut self, stream: &Stream) -> Result<()> {
        ensure!(self.cuda.owns_stream(stream), "WO-A cache preparation context");
        stream.synchronize()?;
        let key = (self.cuda.context_key(), self.weight, self.scales);
        if let Some(value) = DECODED.with(|cache| cache.borrow().get(&key).and_then(Weak::upgrade)) {
            self.cached = Some(value);
            return Ok(());
        }
        let bytes = 2048 * 4096 * 2;
        ensure!(self.cuda.admission_memory()?.admitted_available > bytes + 8 * 1024usize.pow(3),
            "WO-A decoded operand would consume the8GiB reserve");
        let values = self.cuda.allocate(2048 * 4096)?;
        let mut w = self.weight;
        let mut scales = self.scales;
        let mut decoded = values.ptr();
        let mut args = [(&mut w as *mut u64).cast(), (&mut scales as *mut u64).cast(),
            (&mut decoded as *mut u64).cast()];
        unsafe { self.scratch.try_borrow()?.module.function("ca_woa_decode_weight")?.launch(
            [32768, 1, 1], [256, 1, 1], 0, stream, &mut args)?; }
        stream.synchronize()?;
        let value = Rc::new(DecodedWeight { values, _weights: self._weights.clone() });
        DECODED.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.retain(|_, value| value.strong_count() != 0);
            cache.insert(key, Rc::downgrade(&value));
        });
        self.cached = Some(value);
        Ok(())
    }
    pub(crate) fn cached_allocation(&self) -> Option<(u64, usize)> {
        self.cached.as_ref().map(|v| (v.values.ptr(), v.values.bytes()))
    }
    pub(crate) fn set_cached_for_probe(&self, enabled: bool) -> Result<()> {
        ensure!(self.cached.is_some(), "WO-A probe requires prepared immutable operands");
        self.scratch.try_borrow()?.cache_enabled.set(enabled);
        Ok(())
    }
    pub fn run(&mut self, input: &Buffer<bf16>, stream: &Stream) -> Result<()> {
        ensure!(
            input.len() == self.rows * 8192
                && self.cuda.owns_buffer(input)
                && self.cuda.owns_stream(stream),
            "medium WO-A input/context"
        );
        let mut s = self.scratch.try_borrow_mut()?;
        let raw = stream.raw() as usize;
        ensure!(
            s.rows == self.rows && s.cuda.owns_stream(stream),
            "shared WO-A scratch geometry/context changed"
        );
        ensure!(
            s.stream.get().is_none() || s.stream.get() == Some(raw),
            "shared WO-A scratch requires one ordered stream"
        );
        s.stream.set(Some(raw));
        if s.reference_enabled.get() {
            let op=s.compact.as_ref().and_then(|c|c.reference_op.as_ref()).context("WO-A reference absent")?;
            let w=self.cached.as_ref().context("WO-A reference needs canonical BF16 operands")?;
            unsafe{op.launch(self.rows as u32,&[input.ptr(),w.values.ptr(),self.output.ptr()],stream)}?;
            return Ok(());
        }
        if s.compact_enabled.get() {
            let c=s.compact.as_ref().context("compact WO-A plan absent")?;
            unsafe {
                c.partial_op.launch(self.rows as u32,&[input.ptr(),self.weight,self.scales,c.partial.ptr()],stream)?;
                c.reduce_op.launch(self.rows as u32,&[c.partial.ptr(),self.output.ptr()],stream)?;
            }
            return Ok(());
        }
        let mut w = self.weight;
        let mut scales = self.scales;
        let mut decoded = s.decoded.ptr();
        let mut args = [
            (&mut w as *mut u64).cast(),
            (&mut scales as *mut u64).cast(),
            (&mut decoded as *mut u64).cast(),
        ];
        let cached = self.cached.as_ref().filter(|_| s.cache_enabled.get());
        if cached.is_none() { unsafe {
            s.module.function("ca_woa_decode_weight")?.launch(
                [32768, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )?;
        } }
        // Split the borrow so both allocations remain owned by the same arena.
        let Scratch {
            blas,
            decoded,
            grouped,
            ..
        } = &mut *s;
        let operand = cached.map_or(&*decoded, |v| &v.values);
        let compute=PREFILL_ROWS.with(|slot|slot.get()
            .filter(|(_,owner)|self.rows==128&&*owner==stream.raw()as usize)
            .map_or(self.rows,|(rows,_)|rows));
        // Measured on wide prefill only; narrow (graph-captured verify/extend) profiles keep the qualified path.
        if compute==self.rows && self.rows >= 512 && crate::config::wide_rows(self.rows) && token_major_enabled() {
            return blas.grouped_woa_token_major(input,operand,&mut self.output,self.rows,stream);
        }
        blas.grouped_woa_bounded(input, operand, grouped, self.rows, compute, stream)?;
        let mut source = s.grouped.ptr();
        let mut dest = self.output.ptr();
        let mut rows = self.rows as i32;
        if compute<self.rows {
            let mut active=compute as i32;
            let mut args=[(&mut source as *mut u64).cast(),(&mut dest as *mut u64).cast(),
                (&mut rows as *mut i32).cast(),(&mut active as *mut i32).cast()];
            return unsafe{s.module.function("ca_woa_token_major_bounded")?.launch(
                [(self.rows*2048).div_ceil(256)as u32,1,1],[256,1,1],0,stream,&mut args)};
        }
        let mut args = [
            (&mut source as *mut u64).cast(),
            (&mut dest as *mut u64).cast(),
            (&mut rows as *mut i32).cast(),
        ];
        unsafe {
            s.module.function("ca_woa_token_major")?.launch(
                [(self.rows * 2048).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )
        }
    }
    pub fn output(&self) -> &Buffer<bf16> {
        &self.output
    }
}
