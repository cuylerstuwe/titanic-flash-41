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
    /// A paced queue window (ca_l2_paced; ca_l2_paced_wide from two warps or with another wave): ns per 8 KiB
    /// piece, the piece budget, the issuing warps and the pieces a warp issues in one poll.
    paced: Option<(u32, u32, u32, u32)>,
    /// A routed expert window (ca_l2_routed).
    routed: Option<Routed>,
    /// A breadth-first window over equal streams of one region (ca_l2_tiled).
    tiled: Option<Tiled>,
    // Every hinted byte remains owned even if the consumer is dropped first.
    _matrices: Vec<Rc<Matrix>>,
    _router: Option<Rc<Weights>>,
}
/// ca_l2_tiled's arguments.
#[derive(Clone, Copy)]
struct Tiled { base: u64, tile: u32, tiles: u32, piece: u32, budget: u32, ns_piece: u32 }
/// ca_l2_routed's arguments: the router's top-k ids, the layer's up/gate expert region, and the pacing.
#[derive(Clone, Copy)]
struct Routed { ids: u64, count: u32, base: u64, experts: u32, pieces: u32, budget: u32, ns_piece: u32 }
/// The bytes [start, start + bytes) of `regions` taken in order, as a hint table (address, bytes pairs).
/// Empty when the range lies past the regions.
fn range_regions(regions: &[(u64, usize)], start: usize, bytes: usize) -> Result<Vec<u64>> {
    ensure!(start % 16 == 0 && bytes % 16 == 0, "L2 queue range alignment");
    let (mut skip, mut left, mut table) = (start, bytes, Vec::new());
    for &(address, size) in regions {
        ensure!(address != 0 && address % 16 == 0 && size > 0 && size % 16 == 0, "L2 queue region alignment/size");
        address.checked_add(u64::try_from(size)?).context("L2 queue address overflow")?;
        if skip >= size { skip -= size; continue; }
        let take = (size - skip).min(left);
        if take > 0 { table.extend([address + skip as u64, take as u64]); left -= take; }
        skip = 0;
    }
    Ok(table)
}
impl Window {
    /// A paced queue window issued by `warps` warps of one block (ca_l2_paced_wide; config::QueueWindow::warps).
    pub(crate) fn with_warps(mut self, warps: u8) -> Result<Self> {
        ensure!((1..=8).contains(&warps) && (warps == 1 || self.paced.is_some()), "wide L2 queue window");
        if warps > 1 { self.module.function("ca_l2_paced_wide")?; }
        if let Some(paced) = self.paced.as_mut() { paced.2 = u32::from(warps); }
        Ok(self)
    }
    /// A paced queue window whose warps issue at most `wave` pieces in one poll (config::QueueWindow::wave).
    pub(crate) fn with_wave(mut self, wave: u8) -> Result<Self> {
        ensure!((1..=32).contains(&wave) && (wave == 32 || self.paced.is_some()), "L2 queue window wave");
        if wave != 32 { self.module.function("ca_l2_paced_wide")?; }
        if let Some(paced) = self.paced.as_mut() { paced.3 = u32::from(wave); }
        Ok(self)
    }
    /// One window of a decode layer's hint queue (CANADA_SCRATCH_L2_QUEUE): `bytes` of `regions` from
    /// `start`, as one burst (ca_l2_hint) or paced at `pace` GB/s (ca_l2_paced). The owners outlive it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_queue(cuda: &Cuda, module: &Module, parent: &Stream, matrices: Vec<Rc<Matrix>>,
        weights: Rc<Weights>, regions: &[(u64, usize)], start: usize, bytes: usize, pace: Option<u32>)
        -> Result<Option<Self>> {
        ensure!(cuda.owns_stream(parent) && weights.in_context(cuda) && matrices.iter().all(|m| m.in_context(cuda))
            && bytes <= 16 * 1024 * 1024 && pace.is_none_or(|p| (50..=400).contains(&p)), "L2 queue window admission");
        let table = range_regions(regions, start, bytes)?;
        if table.is_empty() { return Ok(None); }
        ensure!(table.len() / 2 <= 16, "L2 queue segment count");
        let total: u64 = table.chunks(2).map(|c| c[1]).sum();
        let mut segments = cuda.allocate(table.len())?;
        segments.upload(&table)?;
        parent.join_default()?;
        let paced = match pace {
            // 8 KiB every 8192 / gbps ns; the budget is the whole window.
            Some(gbps) => { module.function("ca_l2_paced")?; Some(((8192 / gbps).max(1), u32::try_from(total.div_ceil(8192))?, 1, 32)) }
            None => { module.function("ca_l2_hint")?; None }
        };
        let count = u32::try_from(table.len() / 2)?;
        Ok(Some(Self { cuda: cuda.clone(), module: module.clone(), side: cuda.auxiliary_stream(parent)?,
            parent: parent.raw() as usize, ready: cuda.fence_event()?, done: cuda.fence_event()?,
            segments, count, chunk: 16384, selected_start: 0, selected_count: count,
            extra_tables: Vec::new(), selected_extra: None, open: Cell::new(false), paced, routed: None, tiled: None,
            _matrices: matrices, _router: Some(weights) }))
    }
    /// The routed experts' first FC1 tiles (ca_l2_routed, config::RoutedWindow): the kernel reads `ids` (the
    /// router's top-k output, alive with the layer) and hints inside `region`, the layer's contiguous up/gate
    /// expert weights (384 experts of two 576 x 5120 FP4 matrices in the compact N-tile layout).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_routed(cuda: &Cuda, module: &Module, parent: &Stream, weights: Rc<Weights>,
        ids: &Buffer<i32>, region: (u64, usize), plan: crate::config::RoutedWindow) -> Result<Self> {
        const MATRIX: usize = 576 * 5120 / 2;
        ensure!(cuda.owns_stream(parent) && weights.in_context(cuda) && cuda.owns_buffer(ids)
            && !ids.is_empty() && ids.len() <= 16 * 6 && region.0 != 0 && region.0 % 16 == 0
            && region.1 == 384 * 2 * MATRIX && (50..=400).contains(&plan.pace) && plan.bytes % 8192 == 0
            && plan.bytes > 0 && (1..=40).contains(&plan.pieces), "routed L2 window admission");
        region.0.checked_add(region.1 as u64).context("routed L2 region overflow")?;
        let mut segments = cuda.allocate(2)?;
        segments.upload(&[region.0, region.1 as u64])?;
        parent.join_default()?;
        module.function("ca_l2_routed")?;
        Ok(Self { cuda: cuda.clone(), module: module.clone(), side: cuda.auxiliary_stream(parent)?,
            parent: parent.raw() as usize, ready: cuda.fence_event()?, done: cuda.fence_event()?,
            segments, count: 1, chunk: 16384, selected_start: 0, selected_count: 1,
            extra_tables: Vec::new(), selected_extra: None, open: Cell::new(false), paced: None,
            routed: Some(Routed { ids: ids.ptr(), count: u32::try_from(ids.len())?, base: region.0, experts: 384,
                pieces: plan.pieces, budget: u32::try_from(plan.bytes / 8192)?, ns_piece: (8192 / plan.pace).max(1) }),
            tiled: None, _matrices: Vec::new(), _router: Some(weights) })
    }
    /// A paced hint of the first `budget` bytes of a row-major matrix taken column slab by column slab: the rows
    /// are the streams (ca_l2_tiled), `piece` bytes of each a round. For a product too large to hint whole whose
    /// blocks sweep K over all their rows together (the Engram projection).
    pub(crate) fn new_tiled(cuda: &Cuda, module: &Module, parent: &Stream, matrix: Rc<Matrix>, piece: usize,
        budget: usize, pace: u32) -> Result<Self> {
        let region = matrix.immutable_regions()?[0];
        ensure!(cuda.owns_stream(parent) && matrix.in_context(cuda) && region.0 != 0 && region.0 % 16 == 0
            && matrix.n > 0 && matrix.padded_k % 16 == 0 && region.1 == matrix.n * matrix.padded_k
            && [1024usize, 2048, 4096, 8192].contains(&piece) && budget >= piece && budget <= 16 * 1024 * 1024
            && budget % piece == 0 && (50..=400).contains(&pace), "tiled L2 window admission");
        region.0.checked_add(region.1 as u64).context("tiled L2 region overflow")?;
        let mut segments = cuda.allocate(2)?;
        segments.upload(&[region.0, region.1 as u64])?;
        parent.join_default()?;
        module.function("ca_l2_tiled")?;
        Ok(Self { cuda: cuda.clone(), module: module.clone(), side: cuda.auxiliary_stream(parent)?,
            parent: parent.raw() as usize, ready: cuda.fence_event()?, done: cuda.fence_event()?,
            segments, count: 1, chunk: 16384, selected_start: 0, selected_count: 1,
            extra_tables: Vec::new(), selected_extra: None, open: Cell::new(false), paced: None, routed: None,
            tiled: Some(Tiled { base: region.0, tile: u32::try_from(matrix.padded_k)?, tiles: u32::try_from(matrix.n)?,
                piece: piece as u32, budget: u32::try_from(budget / piece)?,
                ns_piece: (u32::try_from(piece)? / pace).max(1) }),
            _matrices: vec![matrix], _router: None })
    }
    /// The KV window hint (ca_l2_kv_window): only the side stream and its events; every begin names the cache
    /// it hints (begin_kv), which the caller owns past every replay.
    pub(crate) fn new_kv(cuda: &Cuda, module: &Module, parent: &Stream) -> Result<Self> {
        ensure!(cuda.owns_stream(parent), "KV window hint parent context");
        let mut segments = cuda.allocate(2)?;
        segments.upload(&[0u64, 0])?;
        parent.join_default()?;
        module.function("ca_l2_kv_window")?;
        Ok(Self { cuda: cuda.clone(), module: module.clone(), side: cuda.auxiliary_stream(parent)?,
            parent: parent.raw() as usize, ready: cuda.fence_event()?, done: cuda.fence_event()?,
            segments, count: 1, chunk: 16384, selected_start: 0, selected_count: 1,
            extra_tables: Vec::new(), selected_extra: None, open: Cell::new(false), paced: None, routed: None, tiled: None,
            _matrices: Vec::new(), _router: None })
    }
    /// Hint the pages of `cache` (base, 64-slot pages of page_bytes, `pages` of them) that hold the window before
    /// the step's first row (`locations`, device slots) and its `rows`. Joined like any other window.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_kv(&self, parent: &Stream, locations: u64, base: u64, page_tokens: usize, page_bytes: usize,
        pages: usize, rows: usize, window: usize) -> Result<()> {
        ensure!(self.cuda.owns_stream(parent) && parent.raw() as usize == self.parent && !self.open.get()
            && locations != 0 && base != 0 && base % 16 == 0 && page_tokens > 0 && page_bytes % 16 == 0
            && page_bytes > 0 && page_bytes <= 1 << 20 && pages > 0 && (1..=32).contains(&rows) && window <= 4096,
            "KV window hint extents");
        self.ready.record(parent)?;
        self.ready.wait(&self.side)?;
        let (mut locations, mut base, mut page_tokens, mut page_bytes, mut pages, mut rows, mut window) =
            (locations, base, page_tokens as u32, page_bytes as u32, pages as u32, rows as u32, window as u32);
        let mut args: [*mut c_void; 7] = [(&mut locations as *mut u64).cast(), (&mut base as *mut u64).cast(),
            (&mut page_tokens as *mut u32).cast(), (&mut page_bytes as *mut u32).cast(), (&mut pages as *mut u32).cast(),
            (&mut rows as *mut u32).cast(), (&mut window as *mut u32).cast()];
        unsafe { self.module.function("ca_l2_kv_window")?.launch([1, 1, 1], [32, 1, 1], 0, &self.side, &mut args) }?;
        self.done.record(&self.side)?;
        self.open.set(true);
        Ok(())
    }
    /// Plan screen only: a hint over a caller-owned scratch range (no model bytes), to time what a hint beside
    /// a collective costs it. The caller keeps the range alive past every replay.
    pub(crate) fn new_scratch_for_probe(cuda: &Cuda, module: &Module, parent: &Stream, region: (u64, usize),
        pace: Option<u32>) -> Result<Self> {
        ensure!(crate::config::l2_queue_screen() && cuda.owns_stream(parent) && region.0 != 0 && region.0 % 16 == 0
            && region.1 > 0 && region.1 % 8192 == 0 && region.1 <= 16 * 1024 * 1024
            && pace.is_none_or(|p| (50..=400).contains(&p)), "scratch L2 hint admission");
        let mut segments = cuda.allocate(2)?;
        segments.upload(&[region.0, region.1 as u64])?;
        parent.join_default()?;
        let paced = match pace {
            Some(gbps) => { module.function("ca_l2_paced")?; Some(((8192 / gbps).max(1), u32::try_from(region.1 / 8192)?, 1, 32)) }
            None => { module.function("ca_l2_hint")?; None }
        };
        Ok(Self { cuda: cuda.clone(), module: module.clone(), side: cuda.auxiliary_stream(parent)?,
            parent: parent.raw() as usize, ready: cuda.fence_event()?, done: cuda.fence_event()?,
            segments, count: 1, chunk: 16384, selected_start: 0, selected_count: 1,
            extra_tables: Vec::new(), selected_extra: None, open: Cell::new(false), paced, routed: None, tiled: None,
            _matrices: Vec::new(), _router: None })
    }
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
            paced:None,routed:None,tiled:None,_matrices:Vec::new(),_router:Some(weights)})
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
            extra_tables:Vec::new(),selected_extra:None,open:Cell::new(false),paced:None,routed:None,tiled:None,_matrices:Vec::new(),_router:Some(weights)})
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
            paced: None,
            routed: None, tiled: None,
            _matrices: matrices,
            _router: router,
        })
    }
    /// Run this window's hint on `side` instead of the shared auxiliary stream (before its first use).
    pub(crate) fn on_stream(mut self, side: Rc<Stream>) -> Result<Self> {
        ensure!(!self.open.get() && self.cuda.owns_stream(&side), "L2 hint stream is chosen before use");
        self.side = side;
        Ok(self)
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
    pub(crate) fn begin(&self, parent: &Stream) -> Result<()> { self.begin_on(parent, parent) }
    /// `begin` with the release recorded on `source`, another stream of the same context: the window starts when
    /// the work already issued on `source` has finished, not when `parent` reaches this call. Joined on `parent`
    /// like any other window.
    pub(crate) fn begin_from(&self, parent: &Stream, source: &Stream) -> Result<()> {
        ensure!(self.cuda.owns_stream(source) && self.tiled.is_none() && self.routed.is_none()
            && self.side.raw() != source.raw(), "an L2 queue window begins from another side stream of its context");
        self.begin_on(parent, source)
    }
    fn begin_on(&self, parent: &Stream, source: &Stream) -> Result<()> {
        ensure!(
            self.cuda.owns_stream(parent)
                && parent.raw() as usize == self.parent
                && !self.open.get(),
            "L2 hint requires paired ordered begin/join"
        );
        if self.selected_count==0 {self.open.set(true);return Ok(())}
        self.ready.record(source)?;
        self.ready.wait(&self.side)?;
        if let Some(t) = self.tiled {
            let (mut base, mut tile, mut tiles, mut piece, mut budget, mut ns) =
                (t.base, t.tile, t.tiles, t.piece, t.budget, t.ns_piece);
            let mut args: [*mut c_void; 6] = [(&mut base as *mut u64).cast(), (&mut tile as *mut u32).cast(),
                (&mut tiles as *mut u32).cast(), (&mut piece as *mut u32).cast(), (&mut budget as *mut u32).cast(),
                (&mut ns as *mut u32).cast()];
            // One warp; hints only inside the validated immutable matrix.
            unsafe { self.module.function("ca_l2_tiled")?.launch([1, 1, 1], [32, 1, 1], 0, &self.side, &mut args) }?;
            self.done.record(&self.side)?;
            self.open.set(true);
            return Ok(());
        }
        if let Some(r) = self.routed {
            let (mut ids, mut count, mut base, mut experts, mut pieces, mut budget, mut ns) =
                (r.ids, r.count, r.base, r.experts, r.pieces, r.budget, r.ns_piece);
            let mut args: [*mut c_void; 7] = [(&mut ids as *mut u64).cast(), (&mut count as *mut u32).cast(),
                (&mut base as *mut u64).cast(), (&mut experts as *mut u32).cast(), (&mut pieces as *mut u32).cast(),
                (&mut budget as *mut u32).cast(), (&mut ns as *mut u32).cast()];
            // One warp; reads the routed ids, hints only inside the validated immutable expert region.
            unsafe { self.module.function("ca_l2_routed")?.launch([1, 1, 1], [32, 1, 1], 0, &self.side, &mut args) }?;
            self.done.record(&self.side)?;
            self.open.set(true);
            return Ok(());
        }
        if let Some((ns_piece, budget, warps, wave)) = self.paced {
            let (mut segments, mut count, mut first, mut budget, mut ns, mut warps, mut wave) =
                (self.segments.ptr(), self.count, 0u32, budget, ns_piece, warps, wave);
            if warps > 1 || wave != 32 {
                let mut args: [*mut c_void; 7] = [(&mut segments as *mut u64).cast(), (&mut count as *mut u32).cast(),
                    (&mut first as *mut u32).cast(), (&mut budget as *mut u32).cast(), (&mut ns as *mut u32).cast(),
                    (&mut warps as *mut u32).cast(), (&mut wave as *mut u32).cast()];
                // One block of `warps` warps; the same immutable, owned, aligned and bounded ranges.
                unsafe { self.module.function("ca_l2_paced_wide")?.launch([1, 1, 1], [32 * warps, 1, 1], 0, &self.side, &mut args) }?;
                self.done.record(&self.side)?;
                self.open.set(true);
                return Ok(());
            }
            let mut args: [*mut c_void; 5] = [(&mut segments as *mut u64).cast(), (&mut count as *mut u32).cast(),
                (&mut first as *mut u32).cast(), (&mut budget as *mut u32).cast(), (&mut ns as *mut u32).cast()];
            // One warp; only immutable, owned, aligned and bounded global-memory ranges.
            unsafe { self.module.function("ca_l2_paced")?.launch([1, 1, 1], [32, 1, 1], 0, &self.side, &mut args) }?;
            self.done.record(&self.side)?;
            self.open.set(true);
            return Ok(());
        }
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
    pub(crate) fn is_open(&self) -> bool {
        self.open.get()
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
    #[test]
    fn queue_ranges_follow_the_regions_in_order() {
        let regions = [(0x1000, 4096), (0x10000, 8192), (0x40000, 1024)];
        assert_eq!(range_regions(&regions, 0, 4096).unwrap(), vec![0x1000, 4096]);
        assert_eq!(range_regions(&regions, 2048, 8192).unwrap(), vec![0x1800, 2048, 0x10000, 6144]);
        assert_eq!(range_regions(&regions, 4096 + 8192, 65536).unwrap(), vec![0x40000, 1024]);
        assert!(range_regions(&regions, 4096 + 8192 + 1024, 4096).unwrap().is_empty());
        assert!(range_regions(&regions, 8, 4096).is_err());
        assert!(range_regions(&[(0x1001, 16)], 0, 16).is_err());
    }
}
