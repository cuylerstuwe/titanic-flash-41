//! Read-only L2 hints in explicitly chosen communication windows. Rust owns the
//! fixed plan, source allocation lifetimes and CUDA graph fork/join. No learned
//! model-engine execution plan or change to arithmetic/precision is involved.
use crate::{
    blockscaled::Matrix,
    cuda::{Buffer, Cuda, Event, Module, Stream},
    weights::Weights,
};
use anyhow::{Context, Result, ensure};
use std::{cell::Cell, ffi::c_void, rc::Rc};

const BUDGET: usize = 6 * 1024 * 1024;
pub(crate) fn enabled() -> bool {
    std::env::var("CANADA_SCRATCH_L2_PREFETCH").as_deref() == Ok("1")
}

// CPU metadata only: no reading, packing or computation of model values.
fn bounded_regions(regions: &[(u64, usize)]) -> Result<Vec<u64>> {
    let mut left = BUDGET;
    let mut table = Vec::new();
    for &(address, bytes) in regions {
        ensure!(
            address != 0 && address % 16 == 0 && bytes > 0 && bytes % 16 == 0,
            "L2 hint region alignment/size"
        );
        address
            .checked_add(u64::try_from(bytes)?)
            .context("L2 hint address overflow")?;
        let take = bytes.min(left);
        if take > 0 {
            table.extend([address, u64::try_from(take)?]);
            left -= take;
        }
    }
    ensure!(
        !table.is_empty() && table.len() <= 16,
        "L2 hint region count"
    );
    Ok(table)
}

pub(crate) struct Window {
    cuda: Cuda,
    module: Module,
    side: Rc<Stream>,
    parent: usize,
    ready: Event,
    done: Event,
    segments: Buffer<u64>,
    count: u32,
    chunk: u32,
    open: Cell<bool>,
    // Every hinted byte remains owned even if the consumer is dropped first.
    _matrices: Vec<Rc<Matrix>>,
    _router: Option<Rc<Weights>>,
}
impl Window {
    pub(crate) fn new(
        cuda: &Cuda,
        module: &Module,
        parent: &Stream,
        matrices: Vec<Rc<Matrix>>,
        router: Option<(Rc<Weights>, &str)>,
    ) -> Result<Self> {
        ensure!(cuda.owns_stream(parent), "L2 hint parent context");
        let mut regions = Vec::new();
        let router = if let Some((weights, key)) = router {
            ensure!(weights.in_context(cuda), "L2 router context");
            let v = weights.view(key)?;
            ensure!(
                v.dtype == crate::checkpoint::Dtype::Bf16
                    && v.shape == [384, 5120]
                    && v.bytes == 384 * 5120 * 2,
                "L2 router immutable byte contract"
            );
            regions.push((weights.ptr(key)?, v.bytes));
            Some(weights)
        } else {
            None
        };
        for matrix in &matrices {
            ensure!(matrix.in_context(cuda), "L2 matrix context");
            regions.extend(matrix.immutable_regions()?);
        }
        let table = bounded_regions(&regions)?;
        let mut segments = cuda.allocate(table.len())?;
        segments.upload(&table)?;
        // Nonblocking side stream must observe the metadata upload too.
        parent.join_default()?;
        let side = cuda.auxiliary_stream(parent)?;
        module.function("ca_l2_hint")?;
        Ok(Self {
            cuda: cuda.clone(),
            module: module.clone(),
            side,
            parent: parent.raw() as usize,
            ready: cuda.fence_event()?,
            done: cuda.fence_event()?,
            segments,
            count: u32::try_from(table.len() / 2)?,
            chunk: 16384,
            open: Cell::new(false),
            _matrices: matrices,
            _router: router,
        })
    }
    pub(crate) fn with_chunk(mut self, chunk:u32)->Result<Self> {
        ensure!([16384,1048576].contains(&chunk) && !self.open.get(),"L2 hint chunk admission");
        if chunk!=16384 {self.module.function("ca_l2_hint_sized")?;}
        self.chunk=chunk;Ok(self)
    }
    pub(crate) fn chunk(&self)->u32 {self.chunk}
    pub(crate) fn begin(&self, parent: &Stream) -> Result<()> {
        ensure!(
            self.cuda.owns_stream(parent)
                && parent.raw() as usize == self.parent
                && !self.open.get(),
            "L2 hint requires paired ordered begin/join"
        );
        self.ready.record(parent)?;
        self.ready.wait(&self.side)?;
        let mut segments = self.segments.ptr();
        let mut count = self.count;
        let mut chunk = self.chunk;
        let mut args: [*mut c_void;3] = [
            (&mut segments as *mut u64).cast(),
            (&mut count as *mut u32).cast(),
            (&mut chunk as *mut u32).cast(),
        ];
        let (kernel,arity)=if chunk==16384 {("ca_l2_hint",2)}else{("ca_l2_hint_sized",3)};
        // Only immutable, owned, aligned and bounded global-memory ranges.
        unsafe {
            self.module.function(kernel)?.launch(
                [count.min(8), 1, 1],
                [64, 1, 1],
                0,
                &self.side,
                &mut args[..arity],
            )
        }?;
        self.done.record(&self.side)?;
        self.open.set(true);
        Ok(())
    }
    pub(crate) fn join(&self, parent: &Stream) -> Result<()> {
        ensure!(
            self.cuda.owns_stream(parent)
                && parent.raw() as usize == self.parent
                && self.open.get(),
            "L2 hint join without matching begin"
        );
        self.done.wait(parent)?;
        self.open.set(false);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_hint_bounds_and_budget() {
        assert_eq!(
            bounded_regions(&[(0x1000, 4096), (0x2000, BUDGET)]).unwrap(),
            vec![0x1000, 4096, 0x2000, (BUDGET - 4096) as u64]
        );
        for regions in [
            vec![],
            vec![(0, 16)],
            vec![(0x1001, 16)],
            vec![(0x1000, 17)],
            vec![(0x1000, 0)],
            vec![(u64::MAX - 15, 32)],
        ] {
            assert!(bounded_regions(&regions).is_err());
        }
        // Validate even a truncated-away trailing source before admission.
        assert!(bounded_regions(&[(0x1000, BUDGET), (0x2001, 16)]).is_err());
    }
}
