//! GPU index selection. Window bounds and output rebasing are admitted on CPU;
//! all score comparison, tie handling and index selection stay on GPU.
use crate::cuda::{Buffer, Cuda, Stream};
use anyhow::{Result, ensure};
use libloading::Library;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::Cell, ffi::c_void, path::Path, rc::Rc};
type Run = unsafe extern "C" fn(
    i32,
    i64,
    i32,
    *mut f32,
    *const i32,
    *const i32,
    *const i32,
    *mut i32,
    *mut c_void,
) -> i32;
#[derive(Deserialize)]
struct Contract {
    abi: u32,
    architecture: u32,
    physical_sms: usize,
    maximum_topk: usize,
    maximum_rows: usize,
    maximum_width: usize,
    score_type: String,
    index_type: String,
    pdl: bool,
    fast_math: bool,
    mode: String,
    library_sha256: String,
}
pub struct Native {
    _lib: Library,
    run: Run,
}
impl Native {
    pub fn open(cuda: &Cuda, root: &Path) -> Result<Rc<Self>> {
        cuda.qualify_gb10()?;
        let c: Contract = serde_json::from_slice(&std::fs::read(root.join("native.json"))?)?;
        ensure!(
            (
                c.abi,
                c.architecture,
                c.physical_sms,
                c.maximum_topk,
                c.maximum_rows,
                c.maximum_width
            ) == (1, 121, 48, 2048, 4096, 1048576)
                && c.score_type == "f32"
                && c.index_type == "i32"
                && !c.pdl
                && !c.fast_math
                && matches!(c.mode.as_str(), "ragged_one_cta_per_row" | "ragged_exact_overflow"),
            "topk contract"
        );
        let p = root.join("libcanada_topk.so");
        ensure!(
            format!("{:x}", Sha256::digest(std::fs::read(&p)?)) == c.library_sha256,
            "topk library hash"
        );
        let lib = unsafe { Library::new(p) }?;
        let abi: unsafe extern "C" fn() -> i32 = *unsafe { lib.get(b"ca_topk_abi\0") }?;
        ensure!(unsafe { abi() } == 1, "topk ABI");
        Ok(Rc::new(Self {
            run: *unsafe { lib.get(b"ca_topk_ragged\0") }?,
            _lib: lib,
        }))
    }
}
pub struct Windows {
    starts: Buffer<i32>,
    lengths: Buffer<i32>,
    offsets: Buffer<i32>,
    rows: usize,
    width: usize,
}
impl Windows {
    pub fn update(&mut self, starts: &[i32], lengths: &[i32], offsets: &[i32]) -> Result<()> {
        ensure!(
            starts.len() == self.rows
                && lengths.len() == self.rows
                && offsets.len() == self.rows
                && starts
                    .iter()
                    .zip(lengths)
                    .zip(offsets)
                    .all(|((&s, &l), &o)| s >= 0
                        && l >= 0
                        && l <= 1048576
                        && (s as usize) + (l as usize) <= self.width
                        && o >= 0
                        && (o as i64) + (l as i64) <= i32::MAX as i64),
            "topk window update outside range/rebasing"
        );
        self.starts.upload(starts)?;
        self.lengths.upload(lengths)?;
        self.offsets.upload(offsets)
    }
    pub(crate) fn lengths(&self) -> &Buffer<i32> {
        &self.lengths
    }
    /// Lengths written on GPU by a producer that derives these windows.
    pub(crate) fn lengths_mut(&mut self) -> &mut Buffer<i32> {
        &mut self.lengths
    }
    pub(crate) fn starts_ptr(&self) -> u64 {
        self.starts.ptr()
    }
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }
    pub fn new(
        cuda: &Cuda,
        width: usize,
        starts: &[i32],
        lengths: &[i32],
        offsets: &[i32],
    ) -> Result<Self> {
        ensure!(
            (1..=crate::config::MAX_WIDE_ROWS).contains(&starts.len())
                && width > 0
                && width <= 1048580
                && starts.len() == lengths.len()
                && starts.len() == offsets.len(),
            "topk window shape"
        );
        ensure!(
            starts
                .iter()
                .zip(lengths)
                .zip(offsets)
                .all(|((&s, &l), &o)| s >= 0
                    && l >= 0
                    && l <= 1048576
                    && (s as usize) + (l as usize) <= width
                    && o >= 0
                    && (o as i64) + (l as i64) <= i32::MAX as i64),
            "topk window range/rebasing"
        );
        let mut ds = cuda.allocate(starts.len())?;
        let mut dl = cuda.allocate(starts.len())?;
        let mut doff = cuda.allocate(starts.len())?;
        ds.upload(starts)?;
        dl.upload(lengths)?;
        doff.upload(offsets)?;
        Ok(Self {
            starts: ds,
            lengths: dl,
            offsets: doff,
            rows: starts.len(),
            width,
        })
    }
}
pub struct Selection {
    cuda: Cuda,
    native: Rc<Native>,
    rows: usize,
    width: usize,
    stride: usize,
    topk: usize,
    indices: Buffer<i32>,
    stream: Cell<usize>,
    ready: Cell<bool>,
}
impl Selection {
    pub(crate) fn native(&self)->Rc<Native> {self.native.clone()}
    fn bind_stream(&self, stream: &Stream) -> Result<()> {
        ensure!(self.cuda.owns_stream(stream), "selection stream/context");
        let p = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == p,
            "topk requires one ordered stream"
        );
        self.stream.set(p);
        Ok(())
    }
    pub(crate) fn admitted_indices(
        &self,
        windows: &Windows,
        stream: &Stream,
    ) -> Result<&Buffer<i32>> {
        ensure!(
            self.ready.get() && self.rows == windows.rows && self.width == windows.width,
            "candidate selection has not produced matching windows"
        );
        self.bind_stream(stream)?;
        Ok(&self.indices)
    }
    pub fn new(
        cuda: &Cuda,
        native: Rc<Native>,
        rows: usize,
        width: usize,
        stride: usize,
        topk: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=crate::config::MAX_WIDE_ROWS).contains(&rows)
                && width > 0
                && width <= 1048580
                && stride >= width
                && stride <= 1048704
                && stride % 4 == 0
                && (1..=2048).contains(&topk),
            "topk selection dimensions"
        );
        Ok(Self {
            cuda: cuda.clone(),
            native,
            rows,
            width,
            stride,
            topk,
            indices: cuda.allocate(rows * topk)?,
            stream: Cell::new(0),
            ready: Cell::new(false),
        })
    }
    /// This primitive consumes scores: it may mask up to three preceding columns
    /// in each row. The mutable borrow makes that use visible to the graph owner.
    pub fn run(&self, scores: &mut Buffer<f32>, windows: &Windows, stream: &Stream) -> Result<()> {
        ensure!(
            scores.len() >= (self.rows - 1) * self.stride + self.width
                && windows.rows == self.rows
                && windows.width == self.width
                && self.cuda.owns_buffer(scores)
                && self.cuda.owns_buffer(&windows.starts)
                && self.cuda.owns_buffer(&windows.lengths)
                && self.cuda.owns_buffer(&windows.offsets)
                && self.cuda.owns_stream(stream),
            "topk score/window/context"
        );
        // A short-context 8K prefill can fit the full score matrix in the
        // dense arena. The native selector still admits at most 4096 rows
        // per call, just as in the existing chunked-score path. Slice only
        // the row dimension; preserve score strides and absolute windows.
        if self.rows > 4096 {
            for start in (0..self.rows).step_by(4096) {
                let end = (start + 4096).min(self.rows);
                let len = (end - start - 1) * self.stride + self.width;
                // Safety: the checked dense allocation remains live and the
                // ordered slices contain disjoint complete score rows.
                let mut slice = unsafe { scores.view(start * self.stride, len)? };
                self.run_rows(&mut slice, windows, start..end, stream)?;
            }
            return Ok(());
        }
        let p = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == p,
            "topk requires one ordered stream"
        );
        self.stream.set(p);
        let status = unsafe {
            (self.native.run)(
                self.rows as i32,
                self.stride as i64,
                self.topk as i32,
                scores.ptr() as _,
                windows.lengths.ptr() as _,
                windows.starts.ptr() as _,
                windows.offsets.ptr() as _,
                self.indices.ptr() as _,
                stream.raw(),
            )
        };
        ensure!(status == 0, "topk launch: {status}");
        self.ready.set(true);
        Ok(())
    }
    /// Select one bounded row slice whose scores occupy the start of a
    /// chunked score arena. Every slice of the schedule must run, in order on
    /// the same stream, before the indices are consumed.
    pub fn run_rows(
        &self,
        scores: &mut Buffer<f32>,
        windows: &Windows,
        rows: std::ops::Range<usize>,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            !rows.is_empty() && rows.len() <= 4096
                && rows.end <= self.rows
                && scores.len() >= (rows.len() - 1) * self.stride + self.width
                && windows.rows == self.rows
                && windows.width == self.width
                && self.cuda.owns_buffer(scores)
                && self.cuda.owns_stream(stream),
            "topk row-slice score/window/context"
        );
        let p = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == p,
            "topk requires one ordered stream"
        );
        self.stream.set(p);
        let offset = (rows.start * 4) as u64;
        let status = unsafe {
            (self.native.run)(
                rows.len() as i32,
                self.stride as i64,
                self.topk as i32,
                scores.ptr() as _,
                (windows.lengths.ptr() + offset) as _,
                (windows.starts.ptr() + offset) as _,
                (windows.offsets.ptr() + offset) as _,
                (self.indices.ptr() + (rows.start * self.topk * 4) as u64) as _,
                stream.raw(),
            )
        };
        ensure!(status == 0, "topk row-slice launch: {status}");
        self.ready.set(rows.end == self.rows);
        Ok(())
    }
    pub fn indices(&self) -> &Buffer<i32> {
        &self.indices
    }
    /// Called after every chunk of this rank's owned quarter was selected.
    /// NCCL permits this in-place gather: the send view is precisely rank's
    /// offset into the full receive allocation. It copies integer bytes.
    pub(crate) fn gather_row_shard(&mut self, rows: std::ops::Range<usize>,
        comm: &crate::roce::Roce, stream: &Stream) -> Result<()> {
        ensure!(self.rows % 4 == 0 && rows.len() == self.rows / 4
            && rows.start == comm.rank() * rows.len(), "selection row-shard geometry");
        self.bind_stream(stream)?;
        unsafe {
            let local = self.indices.view(rows.start * self.topk, rows.len() * self.topk)?;
            comm.gather_index_rows(&local, &mut self.indices, stream)?;
        }
        self.ready.set(true);
        Ok(())
    }
    /// A compact producer has remapped its selected IDs to these absolute
    /// windows. Preserve one ordered stream and a separate destination owner.
    pub(crate) fn copy_remapped_from(&self, source:&Selection, windows:&Windows,
        stream:&Stream)->Result<()> {
        self.write_remapped_from(source,windows,stream,|destination| {
            unsafe{self.cuda.copy_device(destination,source.indices.ptr(),self.indices.bytes(),stream)?;}
            Ok(())
        })
    }
    /// Admit the final output owner before enqueueing an integer-only remap.
    /// Mark it ready only after that writer has been submitted successfully.
    pub(crate) fn write_remapped_from(&self,source:&Selection,windows:&Windows,
        stream:&Stream,write:impl FnOnce(u64)->Result<()>)->Result<()> {
        ensure!(source.ready.get() && source.rows==self.rows && source.topk==self.topk
            && self.rows==windows.rows && self.width==windows.width
            && self.cuda.owns_buffer(&source.indices) && self.cuda.owns_buffer(windows.lengths())
            && self.indices.ptr()!=source.indices.ptr(),
            "remapped selection publication shape/ownership");
        source.bind_stream(stream)?;self.bind_stream(stream)?;
        self.ready.set(false);
        write(self.indices.ptr())?;
        self.ready.set(true);Ok(())
    }
}
