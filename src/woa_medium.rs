//! Medium-prefill WO-A through direct vendor BF16 GEMM. Preserve native raw
//! checkpoint bytes; reconstruct exact BF16 operands on GPU into a shared,
//! bounded scratch, with immutable decoded weights shared across profiles.
//! One ordered stream owns temporary scratch across all decoder layers.
use crate::{
    blas::Blas,
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, Module, Stream},
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    rc::{Rc, Weak},
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

pub struct Scratch {
    cuda: Cuda,
    module: Module,
    blas: Blas,
    rows: usize,
    decoded: Buffer<bf16>,
    grouped: Buffer<bf16>,
    stream: Cell<Option<usize>>,
    cache_enabled: Cell<bool>,
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
        })))
    }
    pub fn owned_scratch_bytes(&self) -> usize {
        self.decoded.bytes() + self.grouped.bytes() + 32 * 1024 * 1024
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
        blas.grouped_woa(input, operand, grouped, self.rows, stream)?;
        let mut source = s.grouped.ptr();
        let mut dest = self.output.ptr();
        let mut rows = self.rows as i32;
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
