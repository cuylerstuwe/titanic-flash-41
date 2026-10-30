//! Two-level index selection over source20's ratio-1 positions, following the
//! canonical model's select_candidate_blocks. The candidate source scores every
//! reachable position, keeps its best 2048 blocks of 8 (always including the
//! block holding the newest position) and publishes a per-row byte mask; later
//! index sources set positions outside it to -inf before their own top-512.
//! Within 16384 reachable positions every block is kept, so the mask is only
//! built for wider contexts. Integer/score selection stays on GPU.
use crate::{
    cuda::{Buffer, Cuda, Module, Stream},
    topk::{self, Selection, Windows},
};
use anyhow::{Result, ensure};
use std::{ffi::c_void, ops::Range, path::Path, rc::Rc};

pub(crate) const BLOCK: usize = 8;
pub(crate) const TOP_BLOCKS: usize = 2048;
/// Contexts within this many positions keep every block.
pub(crate) const WINDOW: usize = BLOCK * TOP_BLOCKS;

pub(crate) fn needed(columns: usize) -> bool {
    columns > WINDOW
}
fn launch(module: &Module, name: &str, threads: usize, args: &mut [*mut c_void], stream: &Stream) -> Result<()> {
    unsafe {
        module.function(name)?.launch(
            [threads.div_ceil(256).max(1) as u32, 1, 1],
            [256, 1, 1],
            0,
            stream,
            args,
        )
    }
}
struct PublisherStorage {
    scores: Buffer<f32>,
    mask: Buffer<u8>,
}
pub(crate) struct Publisher {
    cuda: Cuda,
    module: Module,
    rows: usize,
    score_shards: usize,
    blocks: usize,
    block_stride: usize,
    scores: Buffer<f32>,
    windows: Windows,
    top_lengths: Buffer<i32>,
    selection: Selection,
    mask: Buffer<u8>,
    bounded_scores: bool,
    compact_mask: bool,
    compact_score_windows: Option<Windows>,
    compact_score_first: Option<Buffer<i32>>,
    retained_raw_lengths: Option<Buffer<i32>>,
    compact_retained_ids: Option<Buffer<i32>>,
    // Declared after the non-owning views: storage outlives every view even
    // when the largest profile is dropped before the smaller profiles.
    storage: Rc<PublisherStorage>,
    shared_scratch: bool,
}
impl Publisher {
    pub(crate) fn selector_taps(&self)->Vec<(&'static str,u64,usize)> {
        vec![("block_scores",self.scores.ptr(),self.scores.bytes()),
            ("block_lengths",self.windows.lengths().ptr(),self.windows.lengths().bytes()),
            ("block_selected",self.selection.indices().ptr(),self.selection.indices().bytes()),
            ("block_mask",self.mask.ptr(),self.mask.bytes())]
    }
    pub(crate) fn new(cuda: &Cuda, core: &[u8], kernels: &Path, rows: usize, columns: usize) -> Result<Self> {
        Self::new_with_score_shards(cuda, core, kernels, rows, columns, 1)
    }
    pub(crate) fn new_with_score_shards(cuda: &Cuda, core: &[u8], kernels: &Path,
        rows: usize, columns: usize, score_shards: usize) -> Result<Self> {
        ensure!(
            needed(columns) && columns <= crate::config::MODEL_CONTEXT && (1..=crate::config::MAX_WIDE_ROWS).contains(&rows)
                && (score_shards == 1 || (score_shards == 4 && crate::config::wide_rows(rows))),
            "candidate block publisher geometry"
        );
        let blocks = columns.div_ceil(BLOCK);
        let block_stride = blocks.div_ceil(128) * 128;
        let bounded_scores=match std::env::var("CANADA_SCRATCH_BOUNDED_CANDIDATE_SCORES") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid candidate score policy");value=="1"},
            // s362 per-score bits and s363 four-rank model/state checks pass.
            // Saves live-window work; near-full windows have a small stage cost.
            Err(std::env::VarError::NotPresent)=>true,
            Err(error)=>return Err(error.into()),
        } && columns==1048576;
        let storage = Rc::new(PublisherStorage {
            scores: cuda.allocate(rows / score_shards * block_stride)?,
            mask: cuda.allocate(rows * blocks)?,
        });
        Ok(Self {
            cuda: cuda.clone(),
            module: cuda.module(core)?,
            rows,
            score_shards,
            blocks,
            block_stride,
            scores: unsafe { storage.scores.view(0, rows / score_shards * block_stride)? },
            windows: Windows::new(cuda, blocks, &vec![0; rows], &vec![0; rows], &vec![0; rows])?,
            top_lengths: cuda.allocate(rows)?,
            selection: Selection::new(
                cuda,
                topk::Native::open(cuda, &kernels.join("topk-v2"))?,
                rows,
                blocks,
                block_stride,
                TOP_BLOCKS,
            )?,
            mask: unsafe { storage.mask.view(0, rows * blocks)? },
            bounded_scores,
            compact_mask: false,
            compact_score_windows: None,
            compact_score_first: None,
            retained_raw_lengths: None,
            compact_retained_ids: None,
            storage,
            shared_scratch: false,
        })
    }
    /// Construction-only sharing between sequential wide profiles of the same
    /// target owner. Target checks the common stream, lease and empty history.
    /// Every publication rebuilds the reachable mask rows, and the retained
    /// decoder mask is copied to its separate ring before a profile handoff.
    pub(crate) fn share_scratch_from(&mut self, prior: &Self) -> Result<()> {
        ensure!(self.rows <= prior.rows && self.score_shards == prior.score_shards && self.blocks == prior.blocks
            && self.block_stride == prior.block_stride
            && self.compact_mask == prior.compact_mask
            && self.cuda.owns_buffer(&prior.storage.scores)
            && self.cuda.owns_buffer(&prior.storage.mask), "candidate scratch sharing geometry/owner");
        // Score storage is transient. M512 can retain its ordinary local-row
        // layout while sharing the128-row backing of a compact larger profile.
        let scores = unsafe { prior.storage.scores.view(0, self.score_storage_rows() * self.block_stride)? };
        let mask = unsafe { prior.storage.mask.view(0, self.mask_rows() * self.blocks)? };
        self.scores = scores;
        self.mask = mask;
        self.storage = prior.storage.clone();
        self.shared_scratch = true;
        Ok(())
    }
    pub(crate) fn scratch_report(&self) -> serde_json::Value {
        let backing_bytes = self.storage.scores.bytes() + self.storage.mask.bytes();
        serde_json::json!({"shared":self.shared_scratch,
            "logical_bytes":self.scores.bytes()+self.mask.bytes(),
            "backing_rows":self.storage.scores.len()/self.block_stride*self.score_shards,
            "score_shards":self.score_shards,"score_rows":self.score_storage_rows(),
            "compact_scores":self.compact_score_windows.is_some(),
            "retained_only":self.retained_raw_lengths.is_some(),
            "compact_mask":self.compact_mask,"mask_rows":self.mask_rows(),
            "compact_gather":self.compact_retained_ids.is_some(),
            "compact_gather_bytes":self.compact_retained_ids.as_ref().map_or(0, Buffer::bytes),
            "gather_rows_per_rank":if self.compact_retained_ids.is_some() {128} else {self.rows / self.score_shards},
            "backing_bytes":backing_bytes,
            "owned_bytes":if self.shared_scratch {0}else{backing_bytes}})
    }
    pub(crate) fn mask(&self) -> &Buffer<u8> {
        &self.mask
    }
    fn mask_rows(&self) -> usize { if self.compact_mask {128} else {self.rows} }
    fn score_storage_rows(&self)->usize {if self.compact_score_windows.is_some(){128}else{self.rows/self.score_shards}}
    pub(crate) fn compact_mask(&self) -> bool { self.compact_mask }
    /// Construction-only opt-in by an encoder-only group. Complete layer
    /// groups must still publish every row for their following decoder layers.
    pub(crate) fn enable_retained_publication(&mut self) -> Result<()> {
        let requested = match std::env::var("CANADA_SCRATCH_RETAINED_CANDIDATE_PUBLICATION") {
            Ok(value) => {ensure!(value == "0" || value == "1", "invalid retained publication policy"); value == "1"},
            Err(std::env::VarError::NotPresent) => true,
            Err(error) => return Err(error.into()),
        };
        if requested && crate::config::wide_rows(self.rows) && self.score_shards == 4
            && self.bounded_scores && self.blocks % 16 == 0 && self.mask.ptr() % 16 == 0 {
            self.retained_raw_lengths = Some(self.cuda.allocate(self.rows)?);
            let compact = match std::env::var("CANADA_SCRATCH_COMPACT_CANDIDATE_GATHER") {
                Ok(value) => {ensure!(value == "0" || value == "1", "invalid compact candidate gather policy"); value == "1"},
                Err(std::env::VarError::NotPresent) => true,
                Err(error) => return Err(error.into()),
            };
            // At512rows a full quarter already has128rows; packing would
            // add work without reducing the fixed collective byte count.
            if compact && self.rows > 512 {
                self.compact_retained_ids = Some(self.cuda.allocate(4 * 128 * TOP_BLOCKS)?);
            }
            let compact_mask = match std::env::var("CANADA_SCRATCH_COMPACT_RETAINED_MASK") {
                Ok(value) => {ensure!(value == "0" || value == "1", "invalid compact retained mask policy"); value == "1"},
                // s953: all four ranks, five native cases, exact state/logits.
                Err(std::env::VarError::NotPresent) => true,
                Err(error) => return Err(error.into()),
            };
            if compact_mask {
                // Construction only, before any graph or smaller profile can
                // borrow this backing. Preserve score ownership while dropping
                // the unused full-row mask allocation.
                ensure!(!self.shared_scratch, "cannot compact shared candidate storage");
                let storage = Rc::get_mut(&mut self.storage)
                    .ok_or_else(|| anyhow::anyhow!("candidate mask already shared"))?;
                let mask = self.cuda.allocate(128 * self.blocks)?;
                self.mask = unsafe { mask.view(0, mask.len())? };
                storage.mask = mask;
                self.compact_mask = true;
            }
            let compact_scores=match std::env::var("CANADA_SCRATCH_COMPACT_RETAINED_SCORES") {
                Ok(value)=>{ensure!(value=="0"||value=="1","invalid compact retained score policy");value=="1"},
                // Same s953 native proof for the combined layout; M512 stays
                // on its existing local-row selector. Source20 math unchanged.
                Err(std::env::VarError::NotPresent)=>true,
                Err(error)=>return Err(error.into()),
            };
            // M512 already owns128score rows; preserve its slightly cheaper
            // ordinary selection path while sharing the same physical extent.
            if compact_scores&&self.rows>512 {
                ensure!(!self.shared_scratch,"cannot compact shared candidate scores");
                let storage=Rc::get_mut(&mut self.storage)
                    .ok_or_else(||anyhow::anyhow!("candidate scores already shared"))?;
                let scores=self.cuda.allocate(128*self.block_stride)?;
                self.scores=unsafe{scores.view(0,scores.len())?};storage.scores=scores;
                self.compact_score_windows=Some(Windows::new(&self.cuda,self.blocks,&[0;128],&[0;128],&[0;128])?);
                self.compact_score_first=Some(self.cuda.allocate(1)?);
                self.selection=Selection::new(&self.cuda,self.selection.native(),128,self.blocks,self.block_stride,TOP_BLOCKS)?;
                if self.compact_retained_ids.is_none(){self.compact_retained_ids=Some(self.cuda.allocate(4*128*TOP_BLOCKS)?);}
            }
        }
        Ok(())
    }
    pub(crate) fn prepare_retained_windows(&mut self, windows: &Windows,
        live: Option<&Buffer<i32>>, rank:Option<usize>, stream: &Stream) -> Result<()> {
        let Some(raw) = &self.retained_raw_lengths else {return Ok(())};
        let live = live.ok_or_else(|| anyhow::anyhow!("retained publisher needs live row count"))?;
        ensure!(windows.rows() == self.rows && live.len() == 1 && self.cuda.owns_buffer(live),
            "retained publisher live/window owner");
        let (mut lengths, mut raw, mut blocks, mut top, mut live, mut rows) = (
            windows.lengths().ptr(), raw.ptr(), self.windows.lengths_mut().ptr(),
            self.top_lengths.ptr(), live.ptr(), self.rows as i32);
        let mut args = [(&mut lengths as *mut u64).cast(), (&mut raw as *mut u64).cast(),
            (&mut blocks as *mut u64).cast(), (&mut top as *mut u64).cast(),
            (&mut live as *mut u64).cast(), (&mut rows as *mut i32).cast()];
        launch(&self.module, "ca_retained_candidate_windows", self.rows, &mut args, stream)?;
        if let Some(windows)=&mut self.compact_score_windows {
            let rank=rank.ok_or_else(||anyhow::anyhow!("compact score windows need row owner"))?;
            ensure!(rank<4,"compact score rank");
            let(mut lengths,mut live,mut compact,mut first,mut rows,mut rank)=(
                raw,live,windows.lengths_mut().ptr(),self.compact_score_first.as_ref().unwrap().ptr(),self.rows as i32,rank as i32);
            let mut args=[(&mut lengths as *mut u64).cast(),(&mut live as *mut u64).cast(),
                (&mut compact as *mut u64).cast(),(&mut first as *mut u64).cast(),
                (&mut rows as *mut i32).cast(),(&mut rank as *mut i32).cast()];
            launch(&self.module,"ca_compact_candidate_score_windows",128,&mut args,stream)?;
        }
        Ok(())
    }
    /// Block maxima for `rows`, whose scores start the given (chunk) arena.
    pub(crate) fn score_rows(
        &self,
        scores: &Buffer<f32>,
        stride: usize,
        windows: &Windows,
        rows: Range<usize>,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            !rows.is_empty()
                && rows.end <= self.rows
                && windows.rows() == self.rows
                && self.cuda.owns_buffer(scores)
                && scores.len() >= rows.len() * stride,
            "candidate block score rows"
        );
        let score_rows = self.rows / self.score_shards;
        let first = if self.score_shards == 1 {0} else {rows.start / score_rows * score_rows};
        ensure!(rows.end <= first + score_rows, "candidate score chunk crosses local shard");
        if let Some(first)=&self.compact_score_first {
            let(mut s,mut st,mut starts,mut lengths)=(scores.ptr(),stride as i32,windows.starts_ptr(),
                self.retained_raw_lengths.as_ref().unwrap().ptr());
            let(mut out,mut bs,mut row0,mut count,mut n,mut first)=(self.scores.ptr(),
                self.block_stride as i32,rows.start as i32,rows.len()as i32,self.blocks as i32,first.ptr());
            let mut args=[(&mut s as *mut u64).cast(),(&mut st as *mut i32).cast(),
                (&mut starts as *mut u64).cast(),(&mut lengths as *mut u64).cast(),
                (&mut out as *mut u64).cast(),(&mut bs as *mut i32).cast(),
                (&mut row0 as *mut i32).cast(),(&mut count as *mut i32).cast(),
                (&mut n as *mut i32).cast(),(&mut first as *mut u64).cast()];
            unsafe{self.module.function("ca_candidate_block_scores_compact")?.launch(
                [16,rows.len()as u32,1],[256,1,1],0,stream,&mut args)?;}
            return Ok(());
        }
        // The existing kernel uses row0 both to index metadata and to place
        // output. Rebase metadata to the owned quarter and row0 to that same
        // quarter; its arithmetic and per-row comparison order are unchanged.
        let metadata_offset = (first * std::mem::size_of::<i32>()) as u64;
        let (mut s, mut st, mut starts, mut lengths) =
            (scores.ptr(), stride as i32, windows.starts_ptr() + metadata_offset,
                self.retained_raw_lengths.as_ref().unwrap_or(windows.lengths()).ptr() + metadata_offset);
        let (mut out, mut bs, mut row0, mut count, mut n) = (
            self.scores.ptr(),
            self.block_stride as i32,
            (rows.start - first) as i32,
            rows.len() as i32,
            self.blocks as i32,
        );
        let mut args = [
            (&mut s as *mut u64).cast(),
            (&mut st as *mut i32).cast(),
            (&mut starts as *mut u64).cast(),
            (&mut lengths as *mut u64).cast(),
            (&mut out as *mut u64).cast(),
            (&mut bs as *mut i32).cast(),
            (&mut row0 as *mut i32).cast(),
            (&mut count as *mut i32).cast(),
            (&mut n as *mut i32).cast(),
        ];
        if self.bounded_scores {
            unsafe {self.module.function("ca_candidate_block_scores_bounded")?.launch(
                [16,rows.len()as u32,1],[256,1,1],0,stream,&mut args)?;}
            Ok(())
        }else{launch(&self.module, "ca_candidate_block_scores", rows.len() * self.blocks, &mut args, stream)}
    }
    /// After every row's block maxima: keep the best 2048 blocks per row.
    /// Reachable blocks follow the score windows of the same step on GPU.
    pub(crate) fn publish(&mut self, score_windows: &Windows, stream: &Stream) -> Result<()> {
        self.publish_impl(score_windows,None,None,stream)
    }
    pub(crate) fn publish_sharded(&mut self, score_windows: &Windows, rows: Range<usize>,
        comm: &crate::roce::Roce, live: Option<&Buffer<i32>>, stream: &Stream) -> Result<()> {
        self.publish_impl(score_windows,Some((rows,comm)),live,stream)
    }
    fn publish_impl(&mut self, score_windows: &Windows,
        shard: Option<(Range<usize>,&crate::roce::Roce)>, live: Option<&Buffer<i32>>,
        stream: &Stream) -> Result<()> {
        ensure!(score_windows.rows() == self.rows, "candidate window rows");
        ensure!(self.score_shards == 1 || shard.as_ref().is_some_and(|(rows,_)|
            rows.len() == self.rows / self.score_shards
                && rows.start % rows.len() == 0 && rows.end <= self.rows),
            "local candidate scores need the complete owned row shard");
        let (mut lengths, mut blocks, mut top, mut rows, mut k) = (
            score_windows.lengths().ptr(),
            self.windows.lengths_mut().ptr(),
            self.top_lengths.ptr(),
            self.rows as i32,
            TOP_BLOCKS as i32,
        );
        let mut args = [
            (&mut lengths as *mut u64).cast(),
            (&mut blocks as *mut u64).cast(),
            (&mut top as *mut u64).cast(),
            (&mut rows as *mut i32).cast(),
            (&mut k as *mut i32).cast(),
        ];
        if self.retained_raw_lengths.is_none() {
            launch(&self.module, "ca_candidate_block_windows", self.rows, &mut args, stream)?;
        } else {
            ensure!(shard.is_some() && live.is_some(), "retained publication needs owned rows/live count");
        }
        if let Some((rows,comm))=shard {
            // Exchange only2048 block IDs per row. Rebuild the full byte mask
            // locally, avoiding communication of rows*context/8 mask bytes.
            if let Some(windows)=&self.compact_score_windows {
                self.selection.run(&mut self.scores,windows,stream)?;
            }else{
                let first = if self.score_shards == 1 {rows.start} else {0};
                let mut local_scores=unsafe {self.scores.view(first*self.block_stride,rows.len()*self.block_stride)?};
                self.selection.run_rows(&mut local_scores,&self.windows,rows.clone(),stream)?;
            }
            if let Some(packed) = &mut self.compact_retained_ids {
                ensure!(self.score_shards == 4 && rows.len() == self.rows / 4
                    && rows.start == comm.rank() * rows.len(), "compact candidate gather owner");
                let live = live.ok_or_else(|| anyhow::anyhow!("compact gather requires live count"))?;
                let (mut ids, mut count, mut destination, mut rows, mut rank) = (
                    self.selection.indices().ptr(), live.ptr(), packed.ptr(),
                    self.rows as i32, comm.rank() as i32);
                let mut args = [(&mut ids as *mut u64).cast(), (&mut count as *mut u64).cast(),
                    (&mut destination as *mut u64).cast(), (&mut rows as *mut i32).cast(),
                    (&mut rank as *mut i32).cast()];
                if self.compact_score_windows.is_some() {
                    // The selector already writes128global tail slots; empty
                    // local windows produce-1. Preserve this packet by copy.
                    unsafe{self.cuda.copy_device(packed.ptr()+(comm.rank()*128*TOP_BLOCKS*4)as u64,
                        self.selection.indices().ptr(),128*TOP_BLOCKS*4,stream)?;}
                }else{
                    launch(&self.module, "ca_pack_retained_candidate_ids", 128 * TOP_BLOCKS / 4,
                        &mut args, stream)?;
                }
                // Same rank-major in-place integer byte gather as the full
                // selection. Every packet slot is written before the exchange.
                unsafe {
                    let local = packed.view(comm.rank() * 128 * TOP_BLOCKS, 128 * TOP_BLOCKS)?;
                    comm.gather_index_rows(&local, packed, stream)?;
                }
            } else {
                self.selection.gather_row_shard(rows,comm,stream)?;
            }
        }else{self.selection.run(&mut self.scores, &self.windows, stream)?;}
        if self.retained_raw_lengths.is_some() {
            return self.publish_retained_mask(live.expect("checked retained live"), stream);
        }
        unsafe { self.cuda.zero_device(self.mask.ptr(), self.mask.bytes(), stream) }?;
        let (mut top, mut k, mut lengths, mut mask, mut stride, mut rows) = (
            self.selection.indices().ptr(),
            TOP_BLOCKS as i32,
            self.top_lengths.ptr(),
            self.mask.ptr(),
            self.blocks as i32,
            self.rows as i32,
        );
        let mut args = [
            (&mut top as *mut u64).cast(),
            (&mut k as *mut i32).cast(),
            (&mut lengths as *mut u64).cast(),
            (&mut mask as *mut u64).cast(),
            (&mut stride as *mut i32).cast(),
            (&mut rows as *mut i32).cast(),
        ];
        launch(&self.module, "ca_candidate_mask", self.rows * TOP_BLOCKS, &mut args, stream)
    }
    fn publish_retained_mask(&self, live: &Buffer<i32>, stream: &Stream) -> Result<()> {
        let (mut mask, mut stride, mut rows, mut live) =
            (self.mask.ptr(), self.blocks as i32, self.rows as i32, live.ptr());
        let mut clear = [(&mut mask as *mut u64).cast(), (&mut stride as *mut i32).cast(),
            (&mut rows as *mut i32).cast(), (&mut live as *mut u64).cast()];
        launch(&self.module, if self.compact_mask {"ca_zero_compact_tail_mask"}
            else {"ca_zero_retained_candidate_mask"}, 128 * (self.blocks / 16), &mut clear, stream)?;
        let (mut ids, mut top) = (self.compact_retained_ids.as_ref()
            .unwrap_or(self.selection.indices()).ptr(), self.top_lengths.ptr());
        let mut build = [(&mut ids as *mut u64).cast(), (&mut top as *mut u64).cast(),
            (&mut live as *mut u64).cast(), (&mut mask as *mut u64).cast(),
            (&mut stride as *mut i32).cast(), (&mut rows as *mut i32).cast()];
        let kernel = match (self.compact_mask, self.compact_retained_ids.is_some()) {
            (true, true) => "ca_build_compact_tail_mask_packed",
            (true, false) => "ca_build_compact_tail_mask",
            (false, true) => "ca_build_compact_retained_candidate_mask",
            (false, false) => "ca_build_retained_candidate_mask",
        };
        launch(&self.module, kernel, 128 * TOP_BLOCKS, &mut build, stream)
    }
}
/// Consumer side: sink positions outside the published blocks to -inf.
pub(crate) struct Consumer {
    cuda: Cuda,
    module: Module,
    rows: usize,
    mask_stride: usize,
    bounded_mask: bool,
}
impl Consumer {
    pub(crate) fn new(cuda: &Cuda, core: &[u8], rows: usize, columns: usize) -> Result<Self> {
        ensure!(needed(columns) && columns <= crate::config::MODEL_CONTEXT, "candidate consumer geometry");
        let enabled=match std::env::var("CANADA_SCRATCH_BOUNDED_CANDIDATE_MASK") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid candidate mask policy");value=="1"},
            Err(std::env::VarError::NotPresent)=>true,
            Err(error)=>return Err(error.into()),
        };
        Ok(Self {
            cuda: cuda.clone(),
            module: cuda.module(core)?,
            rows,
            mask_stride: columns.div_ceil(BLOCK),
            bounded_mask: enabled && rows==128 && columns==1048576,
        })
    }
    pub(crate) fn apply(
        &self,
        scores: &mut Buffer<f32>,
        stride: usize,
        windows: &Windows,
        mask: &Buffer<u8>,
        rows: Range<usize>,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            !rows.is_empty()
                && rows.end <= self.rows
                && windows.rows() == self.rows
                && mask.len() == self.rows * self.mask_stride
                && self.cuda.owns_buffer(scores)
                && self.cuda.owns_buffer(mask)
                && scores.len() >= rows.len() * stride,
            "candidate consumer rows/mask"
        );
        let (mut s, mut st, mut starts, mut lengths, mut m, mut ms, mut row0, mut count) = (
            scores.ptr(),
            stride as i32,
            windows.starts_ptr(),
            windows.lengths().ptr(),
            mask.ptr(),
            self.mask_stride as i32,
            rows.start as i32,
            rows.len() as i32,
        );
        let mut args = [
            (&mut s as *mut u64).cast(),
            (&mut st as *mut i32).cast(),
            (&mut starts as *mut u64).cast(),
            (&mut lengths as *mut u64).cast(),
            (&mut m as *mut u64).cast(),
            (&mut ms as *mut i32).cast(),
            (&mut row0 as *mut i32).cast(),
            (&mut count as *mut i32).cast(),
        ];
        if self.bounded_mask {
            // s355: same writes with fewer idle CTAs at long serving capacity.
            // Consumer128 uses bounded16-row score chunks. Keep a control switch.
            unsafe {self.module.function("ca_apply_candidate_mask_bounded")?.launch(
                [16,rows.len()as u32,1],[128,1,1],0,stream,&mut args[..7])?;}
            Ok(())
        }else{
            launch(&self.module, "ca_apply_candidate_mask", rows.len() * stride, &mut args, stream)
        }
    }
}

/// GPU qualification of level one/two against canonical torch selection
/// (tools/candidate-reference.py): byte-exact kept-block masks for distinct
/// scores, and consumer masking that only sinks positions outside them.
pub fn probe(core: &Path, kernels: &Path, fixture: &Path) -> Result<serde_json::Value> {
    #[derive(serde::Deserialize)]
    struct Case {
        rows: usize,
        columns: usize,
        lengths: Vec<i32>,
        directory: String,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    let f: Fixture = serde_json::from_slice(&std::fs::read(fixture.join("index.json"))?)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let core = std::fs::read(core)?;
    let mut reports = Vec::new();
    for c in &f.cases {
        let dir = fixture.join(&c.directory);
        let floats = |name: &str| -> Result<Vec<f32>> {
            Ok(std::fs::read(dir.join(name))?
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect())
        };
        let scores = floats("scores.bin")?;
        let expected = std::fs::read(dir.join("block_mask.bin"))?;
        let blocks = c.columns.div_ceil(BLOCK);
        ensure!(
            scores.len() == c.rows * c.columns
                && expected.len() == c.rows * blocks
                && c.lengths.len() == c.rows,
            "candidate fixture geometry"
        );
        let stride = c.columns.div_ceil(128) * 128;
        let mut padded = vec![f32::NAN; c.rows * stride];
        for r in 0..c.rows {
            padded[r * stride..r * stride + c.columns]
                .copy_from_slice(&scores[r * c.columns..(r + 1) * c.columns]);
        }
        let mut device = cuda.allocate::<f32>(padded.len())?;
        device.upload(&padded)?;
        let windows = Windows::new(&cuda, c.columns, &vec![0; c.rows], &c.lengths, &vec![0; c.rows])?;
        let mut publisher = Publisher::new(&cuda, &core, kernels, c.rows, c.columns)?;
        publisher.score_rows(&device, stride, &windows, 0..c.rows, &stream)?;
        publisher.publish(&windows, &stream)?;
        stream.synchronize()?;
        let mask = publisher.mask().download()?;
        let mask_exact = mask == expected;
        let consumer = Consumer::new(&cuda, &core, c.rows, c.columns)?;
        consumer.apply(&mut device, stride, &windows, publisher.mask(), 0..c.rows, &stream)?;
        stream.synchronize()?;
        let masked = device.download()?;
        let mut consumer_exact = true;
        for r in 0..c.rows {
            for p in 0..c.columns {
                let (before, after) = (padded[r * stride + p], masked[r * stride + p]);
                let keep = (p as i32) >= c.lengths[r] || expected[r * blocks + p / BLOCK] != 0;
                consumer_exact &= if keep {
                    before.to_bits() == after.to_bits()
                } else {
                    after == f32::NEG_INFINITY
                };
            }
        }
        reports.push(serde_json::json!({"case":c.directory,"rows":c.rows,"columns":c.columns,
            "kept_blocks":mask.iter().map(|&b| b as usize).sum::<usize>(),
            "block_mask_bitwise_equal":mask_exact,"consumer_masking_exact":consumer_exact}));
    }
    let all = reports
        .iter()
        .all(|r| r["block_mask_bitwise_equal"] == true && r["consumer_masking_exact"] == true);
    Ok(serde_json::json!({"all_exact":all,"cases":reports,
        "scope":"Candidate-block level one (block max, newest block pinned, top-2048) and level-two masking on GPU versus canonical torch selection with distinct scores; integer selection only, not full-model long-context quality."}))
}
