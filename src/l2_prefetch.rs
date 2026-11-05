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
/// Startup-only mask: FFN/router=1, Q-B=2, next input=4. The small-row
/// schedule is unchanged; wider profiles remain off unless explicitly selected.
pub(crate) fn wide_mask() -> Result<u8> {
    match std::env::var("CANADA_SCRATCH_WIDE_L2_PREFETCH") {
        Ok(v)=>{let mask:u8=v.parse()?;ensure!(mask<=7,"wide L2 hint mask");Ok(mask)},
        Err(std::env::VarError::NotPresent)=>Ok(0),Err(e)=>Err(e.into()),
    }
}
pub(crate) fn enabled_for_rows(rows:usize,bit:u8)->Result<bool> {
    ensure!([1,2,4].contains(&bit),"L2 hint stage");
    Ok(enabled()&&(rows<=8 || (matches!(rows,16|32)
        && (wide_mask()?&bit!=0
            ||std::env::var("CANADA_SCRATCH_WIDE_L2_SCREEN").as_deref()==Ok("1")))))
}

// CPU metadata only: no reading, packing or computation of model values.
fn bounded_regions(regions: &[(u64, usize)]) -> Result<Vec<u64>> {
    bounded_regions_with_budget(regions,BUDGET)
}
fn bounded_regions_with_budget(regions:&[(u64,usize)],budget:usize)->Result<Vec<u64>> {
    ensure!(budget>0&&budget%16==0&&budget<=12*1024*1024,"L2 hint budget bounds");
    let mut left = budget;
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
    // Capture-only descriptor slice. The table remains immutable and owned,
    // so distinct graphs retain their own checked pointer/count arguments.
    selected_start: u32,
    selected_count: u32,
    extra_tables: Vec<(usize,Buffer<u64>)>,
    selected_extra: Option<usize>,
    open: Cell<bool>,
    // Every hinted byte remains owned even if the consumer is dropped first.
    _matrices: Vec<Rc<Matrix>>,
    _router: Option<Rc<Weights>>,
}
impl Window {
    /// Finite expert screen: the owner supplies checked contiguous expert
    /// ranges. The descriptor is fixed before capture, without a GPU route
    /// selector; account for that omitted selector when interpreting speed.
    pub(crate) fn new_expert(cuda:&Cuda,module:&Module,parent:&Stream,
        weights:Rc<Weights>,regions:&[(u64,usize)])->Result<Self> {
        ensure!(std::env::var("CANADA_SCRATCH_EXPERT_PREFETCH_SCREEN").as_deref()==Ok("1")
            &&weights.in_context(cuda)&&cuda.owns_stream(parent),"finite expert hint owner");
        let table=bounded_regions(regions)?;
        let mut segments=cuda.allocate(table.len())?;segments.upload(&table)?;
        parent.join_default()?;module.function("ca_l2_hint")?;
        Ok(Self {cuda:cuda.clone(),module:module.clone(),side:cuda.auxiliary_stream(parent)?,
            parent:parent.raw()as usize,ready:cuda.fence_event()?,done:cuda.fence_event()?,
            segments,count:(table.len()/2)as u32,chunk:16384,selected_start:0,
            selected_count:(table.len()/2)as u32,extra_tables:Vec::new(),selected_extra:None,open:Cell::new(false),
            _matrices:Vec::new(),_router:Some(weights)})
    }
    /// Finite WO-A experiment: immutable canonical FP8/E8M0 bytes only.
    /// The checkpoint owner outlives the hint and every captured replay.
    pub(crate) fn new_woa(cuda:&Cuda,module:&Module,parent:&Stream,
        weights:Rc<Weights>,id:crate::layer_id::LayerId)->Result<Self> {
        ensure!(std::env::var("CANADA_SCRATCH_WOA_PREFETCH_SCREEN").as_deref()==Ok("1")
            && !id.is_draft() && cuda.owns_stream(parent) && weights.in_context(cuda),
            "WO-A hint requires the finite target screen");
        let key=format!("{}.attn.wo_a",id.prefix());
        let w=weights.view(&format!("{key}.weight"))?;
        let s=weights.view(&format!("{key}.scale"))?;
        ensure!(w.dtype==crate::checkpoint::Dtype::Fp8 && w.shape==[2048,4096]
            && w.bytes==8*1024*1024 && s.dtype==crate::checkpoint::Dtype::E8m0
            && s.shape==[64,128] && s.bytes==8192,"WO-A immutable hint extent");
        let wp=weights.ptr(&format!("{key}.weight"))?;
        let sp=weights.ptr(&format!("{key}.scale"))?;
        ensure!(wp%16==0 && sp%16==0,"WO-A hint alignment");
        let mut segments=cuda.allocate(4)?;
        segments.upload(&[wp,w.bytes as u64,sp,s.bytes as u64])?;
        parent.join_default()?;
        module.function("ca_l2_hint")?;
        Ok(Self {cuda:cuda.clone(),module:module.clone(),side:cuda.auxiliary_stream(parent)?,
            parent:parent.raw()as usize,ready:cuda.fence_event()?,done:cuda.fence_event()?,
            segments,count:2,chunk:16384,selected_start:0,selected_count:2,
            extra_tables:Vec::new(),selected_extra:None,open:Cell::new(false),_matrices:Vec::new(),_router:Some(weights)})
    }
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
            selected_start: 0,
            selected_count: u32::try_from(table.len()/2)?,
            extra_tables:Vec::new(),selected_extra:None,
            open: Cell::new(false),
            _matrices: matrices,
            _router: router,
        })
    }
    pub(crate) fn with_chunk(mut self, chunk:u32)->Result<Self> {
        ensure!([2048,16384,1048576].contains(&chunk) && !self.open.get(),"L2 hint chunk admission");
        if chunk!=16384 {self.module.function("ca_l2_hint_sized")?;}
        self.chunk=chunk;Ok(self)
    }
    pub(crate) fn chunk(&self)->u32 {self.chunk}
    /// Caller joined all work and retired graph owners before changing capture
    /// selection. No device write, reallocation or arithmetic change.
    pub(crate) fn set_enabled_for_wide_probe(&mut self,enabled:bool,parent:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_WIDE_L2_SCREEN").as_deref()==Ok("1")
            && !self.open.get() && self.selected_extra.is_none()
            && self.cuda.owns_stream(parent) && parent.raw()as usize==self.parent,
            "finite idle wide L2 hint owner");
        self.selected_start=0;self.selected_count=if enabled{self.count}else{0};Ok(())
    }
    pub(crate) fn set_budget_for_probe(&mut self,budget:usize,parent:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_FFN_PREFETCH_BUDGET_SCREEN").as_deref()==Ok("1")
            &&[6,8,10,12].iter().any(|m|budget==m*1024*1024)&&!self.open.get()
            &&self._router.is_some()&&self._matrices.len()==2&&self.count>=2
            &&self.cuda.owns_stream(parent)&&parent.raw()as usize==self.parent,"finite FFN budget owner");
        parent.synchronize()?;
        self.selected_start=0;
        if budget==BUDGET {self.selected_extra=None;self.selected_count=self.count;return Ok(());}
        let index=if let Some(i)=self.extra_tables.iter().position(|(b,_)|*b==budget) {i}else{
            // Original router descriptor is complete; all remaining sources
            // come from the same owned canonical matrices, never a new atlas.
            let table=self.segments.download()?;
            ensure!(table[1]==(384*5120*2)as u64,"FFN router first extent");
            let mut regions=vec![(table[0],table[1]as usize)];
            for matrix in &self._matrices {regions.extend(matrix.immutable_regions()?);}
            let table=bounded_regions_with_budget(&regions,budget)?;
            let mut owner=self.cuda.allocate(table.len())?;owner.upload(&table)?;parent.join_default()?;
            self.extra_tables.push((budget,owner));self.extra_tables.len()-1
        };
        self.selected_count=u32::try_from(self.extra_tables[index].1.len()/2)?;
        self.selected_extra=Some(index);Ok(())
    }
    pub(crate) fn set_region_plan_for_probe(&mut self,mode:u32,parent:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_FFN_PREFETCH_REGION_SCREEN").as_deref()==Ok("1")
            && mode<=3 && !self.open.get() && self._router.is_some() && self._matrices.len()==2
            && self.count>=2 && self.cuda.owns_stream(parent) && parent.raw()as usize==self.parent,
            "finite FFN region selection owner");
        parent.synchronize()?;
        // The constructor puts the complete384x5120 BF16 router first,
        // followed by shared projection regions up to the same6MiB budget.
        (self.selected_start,self.selected_count)=match mode {
            0=>(0,self.count),1=>(0,1),2=>(1,self.count-1),3=>(0,0),_=>unreachable!(),
        };
        Ok(())
    }
    /// Capture-time-only component control; source descriptors and numerical
    /// consumers are unchanged, and old graphs retain their own scalar args.
    pub(crate) fn set_chunk_for_probe(&mut self,chunk:u32,parent:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_FFN_PREFETCH_CHUNK_SCREEN").as_deref()==Ok("1")
            && [2048,4096,8192,16384,65536,262144,1048576].contains(&chunk) && !self.open.get()
            && self.cuda.owns_stream(parent) && parent.raw()as usize==self.parent,
            "finite FFN hint granularity owner");
        parent.synchronize()?;
        self.module.function("ca_l2_hint_sized")?;
        self.chunk=chunk;Ok(())
    }
    pub(crate) fn begin(&self, parent: &Stream) -> Result<()> {
        ensure!(
            self.cuda.owns_stream(parent)
                && parent.raw() as usize == self.parent
                && !self.open.get(),
            "L2 hint requires paired ordered begin/join"
        );
        if self.selected_count==0 {self.open.set(true);return Ok(())}
        self.ready.record(parent)?;
        self.ready.wait(&self.side)?;
        let base=self.selected_extra.map_or(self.segments.ptr(),|i|self.extra_tables[i].1.ptr());
        let mut segments = base+u64::from(self.selected_start)*16;
        let mut count = self.selected_count;
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
        if self.selected_count>0 {self.done.wait(parent)?;}
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
