//! Score only the candidate union of each bounded consumer chunk. This carries
//! the s356 GPU component into the native graph without changing score math.
use crate::{
    candidates::{BLOCK, WINDOW},
    cuda::{Buffer, Cuda, Module, Stream},
    index::{KeyCache, KeySlots, QueryRows},
    mqa::{Chunked, KeyMatrix, Native, Ranges},
    topk::{self, Selection, Windows},
};
use anyhow::{Result, ensure};
use std::{ffi::c_void, rc::Rc};

macro_rules! args {
    ($($v:ident),* $(,)?) => { [$(std::ptr::addr_of_mut!($v).cast::<c_void>()),*] };
}
const ROWS: usize = 128;
const GROUP: usize = 32;
pub(crate) fn key_capacity(columns: usize) -> usize { columns.min(GROUP * WINDOW) }
pub(crate) fn key_capacity16(columns: usize) -> usize { columns.min(16 * WINDOW) }

pub(crate) struct Grouped {
    cuda: Cuda,
    module: Module,
    packed_gather: Option<Module>,
    rows: usize,
    columns: usize,
    keys: Rc<KeyMatrix>,
    scorer: Chunked,
    ranges: Ranges,
    windows: Windows,
    selection: Selection,
    counts: Buffer<i32>,
    ballots: Buffer<u32>,
    cached_ballots: bool,
    ids: Buffer<i32>,
    error: Buffer<u32>,
}
impl Grouped {
    /// Read-only taps after the owning target has joined every stream. These
    /// distinguish changed producer inputs/scores from equal-score top-k ties.
    pub(crate) fn replay_taps(&self) -> Vec<(&'static str,u64,usize)> {
        vec![("scores",self.scorer.output().ptr(),self.scorer.output().bytes()),
            ("union-counts",self.counts.ptr(),self.counts.bytes()),
            ("union-ballots",self.ballots.ptr(),self.ballots.bytes()),
            ("union-ids",self.ids.ptr(),self.ids.bytes()),
            ("union-ends",self.ranges.key_ends().ptr(),self.ranges.key_ends().bytes()),
            ("selection-lengths",self.windows.lengths().ptr(),self.windows.lengths().bytes()),
            ("selection-error",self.error.ptr(),self.error.bytes())]
    }
    pub(crate) fn share_score_scratch(&mut self, storage: Rc<Buffer<f32>>) -> Result<usize> {
        self.scorer.share_score_scratch(storage)
    }
    pub(crate) fn cached_ballots(&self) -> bool { self.cached_ballots }
    pub(crate) fn columns(&self) -> usize { self.columns }
    // The native selector sees the union length, not each original causal
    // length. Its short-row ordered-output shortcut is valid only for this
    // length (a group can cross512 while some member rows remain shorter).
    pub(crate) fn selection_lengths(&self) -> &Buffer<i32> { self.windows.lengths() }
    pub(crate) fn new(cuda: &Cuda, core: &[u8], native: Rc<Native>,
        topk: Rc<topk::Native>, queries: Rc<QueryRows>, keys: Rc<KeyMatrix>,
        columns: usize, budget: usize) -> Result<Self> {
        let rows=queries.rows();let group=if rows==16 {16}else{GROUP};
        ensure!([16,32,ROWS].contains(&rows) && columns == 1048576
            && keys.rows() == columns.min(group*WINDOW), "grouped sparse geometry");
        let cached_ballots = match std::env::var("CANADA_SCRATCH_GROUPED_UNION_CACHE") {
            Ok(value) => { ensure!(value == "0" || value == "1", "invalid grouped union cache policy"); value == "1" },
            Err(std::env::VarError::NotPresent) => true,
            Err(error) => return Err(error.into()),
        };
        let capacity = keys.rows();
        let scorer = Chunked::new(cuda, native, queries, keys.clone(), capacity, budget)?;
        ensure!(scorer.len() == rows / group && scorer.output().bytes() <= budget
            && (0..scorer.len()).all(|i| scorer.rows(i).is_ok_and(|r| r.len() == group)),
            "grouped sparse bounded score schedule");
        Ok(Self {
            cuda: cuda.clone(), module: cuda.module(core)?, packed_gather:None, rows, columns,
            ranges: Ranges::new(cuda, capacity, capacity, &vec![0; rows], &vec![0; rows])?,
            windows: Windows::new(cuda, capacity, &vec![0; rows], &vec![0; rows], &vec![0; rows])?,
            selection: Selection::new(cuda, topk, rows, capacity, scorer.stride(), 512)?,
            counts: cuda.allocate((columns / BLOCK).div_ceil(256) + 1)?,
            // One 16KiB ballot buffer is reused by the four ordered row groups.
            ballots: cuda.allocate((columns / BLOCK).div_ceil(256) * 8)?, cached_ballots,
            ids: cuda.allocate(capacity / BLOCK)?, error: cuda.allocate(1)?,
            keys, scorer,
        })
    }
    pub(crate) fn configure_packed_gather(&mut self,kernels:&std::path::Path)->Result<()> {
        use sha2::{Digest,Sha256};
        ensure!([16,32].contains(&self.rows),"packed grouped verifier gather shape");
        let root=kernels.join("sparse-gather-packed-v1");
        let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let raw=std::fs::read(root.join("gather.cubin"))?;
        // The gather uses the group's single union length, not a query-row
        // extent. s682 checks the same four-lane packed ABI with16-row unions.
        ensure!(manifest["architecture"]==121 && manifest["threads"]==128
            && manifest["grid"]==192 && manifest["query_rows"]==6 && manifest["lanes_per_key"]==4
            && manifest["source_sha256"]=="3e15bcdbb8428d2d4f052b278e91e6bd207aa70d7d72a5a54eec27070d6e8cd6"
            && manifest["artifact_sha256"]=="dc78579e7b22ccc9dd0f7538e50908002df1ee5cace166b8ef32913fd04b7c41"
            && format!("{:x}",Sha256::digest(&raw))==manifest["artifact_sha256"].as_str().unwrap_or(""),
            "packed grouped gather artifact/geometry contract");
        self.packed_gather=Some(self.cuda.module(&raw)?);Ok(())
    }
    fn launch(&self, name: &str, grid: usize, threads: usize,
        arguments: &mut [*mut c_void], stream: &Stream) -> Result<()> {
        unsafe { self.module.function(name)?.launch([grid as u32, 1, 1],
            [threads as u32, 1, 1], 0, stream, arguments) }
    }
    pub(crate) fn run(&mut self, cache: &KeyCache, slots: &KeySlots,
        windows: &Windows, mask: &Buffer<u8>, output: &Selection, stream: &Stream) -> Result<()> {
        let blocks = self.columns / BLOCK;
        ensure!(windows.rows() == self.rows && mask.len() == self.rows * blocks
            && slots.values().len() == self.columns && slots.capacity() == cache.capacity()
            && self.cuda.owns_buffer(cache.data()) && self.cuda.owns_buffer(slots.values())
            && self.cuda.owns_buffer(mask) && self.cuda.owns_buffer(windows.lengths())
            && self.cuda.owns_stream(stream), "grouped sparse cache/mask/windows/context");
        cache.bind(stream)?; self.keys.bind(stream)?;
        unsafe { self.cuda.zero_device(self.error.ptr(), self.error.bytes(), stream)?; }
        for index in 0..self.scorer.len() {
            let rows = self.scorer.rows(index)?;
            let offset = (rows.start * 4) as u64;
            let (mut mask_ptr, mut lengths, mut counts, mut ids, mut error) =
                (mask.ptr() + (rows.start * blocks) as u64, windows.lengths().ptr() + offset,
                self.counts.ptr(), self.ids.ptr(), self.error.ptr());
            let (mut block_count, mut columns, mut tiles, mut max_blocks, mut row_count) =
                (blocks as i32, self.columns as i32, blocks.div_ceil(256) as i32,
                self.ids.len() as i32, rows.len() as i32);
            let mut ends = self.ranges.key_ends_mut().ptr() + offset;
            let mut ballots = self.ballots.ptr();
            if self.cached_ballots {
                self.launch("ca_group_counts_cached", tiles as usize, 256,
                    &mut args![mask_ptr, lengths, counts, ballots, block_count, columns, row_count, error], stream)?;
            } else {
                self.launch("ca_group_counts", tiles as usize, 256,
                    &mut args![mask_ptr, lengths, counts, block_count, columns, row_count, error], stream)?;
            }
            self.launch("ca_group_offsets", 1, 32,
                &mut args![counts, tiles, max_blocks, ends, row_count, error], stream)?;
            if self.cached_ballots {
                self.launch("ca_group_ids_cached", tiles as usize, 256,
                    &mut args![counts, ballots, ids, max_blocks, error], stream)?;
            } else {
                self.launch("ca_group_ids", tiles as usize, 256,
                    &mut args![mask_ptr, lengths, counts, ids, block_count, columns, max_blocks, row_count, error], stream)?;
            }
            let (mut cache_ptr, mut slot_ptr, mut cache_capacity, mut payload, mut scales, mut capacity) =
                (cache.data().ptr(), slots.values().ptr(), cache.capacity() as i32,
                self.keys.payload.ptr(), self.keys.scales.ptr(), self.keys.rows() as i32);
            let mut arguments=args![cache_ptr, slot_ptr, cache_capacity, ids, ends, payload, scales, columns, capacity, error];
            if let Some(module)=&self.packed_gather {
                unsafe {module.function("ca_sparse6_gather_keys_packed4")?.launch(
                    [192,1,1],[128,1,1],0,stream,&mut arguments)?;}
            }else{self.launch("ca_sparse6_gather_keys",192,128,&mut arguments,stream)?;}
            self.keys.ready.set(true);
            self.scorer.run_chunk(index, &self.ranges, stream)?;
            let (mut scores, mut stride, mut compact, mut compact_lengths) =
                (self.scorer.output().ptr(), self.scorer.stride() as i32, 1i32,
                self.windows.lengths_mut().ptr() + offset);
            self.launch("ca_group_mask", 192, 256,
                &mut args![scores, stride, ids, ends, lengths, mask_ptr, block_count, row_count, compact, compact_lengths], stream)?;
            self.selection.run_rows(self.scorer.output_mut(), &self.windows, rows.clone(), stream)?;
            let mut indices = self.selection.indices().ptr() + (rows.start * 512 * 4) as u64;
            self.launch("ca_group_remap", rows.len(), 512,
                &mut args![indices, scores, stride, ids, ends, lengths, error], stream)?;
        }
        output.copy_remapped_from(&self.selection, windows, stream)
    }
    pub(crate) fn poison_selection_on_error(&self, indices: &Buffer<i32>, stream: &Stream) -> Result<()> {
        ensure!(indices.len() == self.rows * 512 && self.cuda.owns_buffer(indices), "grouped selection geometry");
        let (mut error, mut output) = (self.error.ptr(), indices.ptr());
        self.launch("ca_sparse6_poison_selection", 1, 32, &mut args![error, output], stream)
    }
}
