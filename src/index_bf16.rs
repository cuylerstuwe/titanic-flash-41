//! Index selection with the reference's bf16 arithmetic: `kernels/index_bf16.cu` (adapted from pico's indexer,
//! MIT, Copyright (c) 2026 Andy Shu: licenses/pico-dgx-ds4-MIT.txt), a prebuilt sm_121a cubin under
//! `<kernels>/index-bf16-v1/` with its manifest contract.
//!
//! config::index_bf16 (CANADA_SCRATCH_INDEX_BF16=1, default off): a dense index layer without a candidate
//! publisher or consumer (source layers 2 / 8 / 14) replaces its MQA logits, topk-v2 selection and ascending sort
//! by one score pass that writes 16-bit order-preserving keys of
//! `bf16(sum_h bf16(relu(bf16(q.k)) * w))` (DeepSeek's model.py: einsum, relu, mul by the head weights, sum, each
//! in bf16) and one selection pass per row (exact top-k over those keys, ties at the cut by ascending position,
//! output ascending with -1 padding, the layout the sorted topk-v2 selection has). A numeric change against the
//! f32 scores it replaces (which rows' 512th place falls on a near-tie): qualified by the reference KL gate, not
//! by bit identity. Layer 20 (candidate publisher) and the consumer layers' dense mode take the same scores as
//! f32 values in place of the MQA logits and keep their publisher, mask, topk-v2 and sort (topk-v2's rule over
//! bf16 values, higher first and equal values by lower column, is the same selection). The consumers' sparse
//! long-context scorer is unchanged (still f32).
use crate::cuda::{Buffer, Cuda, Module, Stream};
use crate::index::QueryRows;
use crate::mqa::KeyMatrix;
use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};
use std::ffi::c_void;
use std::path::Path;

/// Column CTAs of the score launch: each loops over its 256-column tiles up to the group's last window end, so
/// one captured launch serves every context length.
const SCORE_GRID_X: u32 = 96;
/// The fused launch (ca_idx16_dense, step 2): rows a launch, CTAs (one an SM), dynamic shared bytes (index_bf16.cu
/// kDenseRows / kDenseSmem).
const DENSE_ROWS: usize = 8;
const DENSE_GRID: u32 = 48;
const DENSE_SMEM: u32 = 8 * 32 * (64 + 4 + 4) + 8 * 1024 * 4;

pub(crate) struct IndexBf16 {
    module: Module,
    rows: usize,
    columns: usize,
    /// config::index_bf16_dense with an artifact that has ca_idx16_dense: its scratch (two counters, then [rows][2048]
    /// histograms; zero, and left zero by every launch).
    dense: Option<Buffer<u32>>,
}

impl IndexBf16 {
    pub(crate) fn new(cuda: &Cuda, kernels: &Path, rows: usize, columns: usize) -> Result<Self> {
        let root = kernels.join("index-bf16-v1");
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let raw = std::fs::read(root.join("index16.cubin"))?;
        let two = serde_json::json!(["ca_idx16_score", "ca_idx16_select"]);
        let three = serde_json::json!(["ca_idx16_score", "ca_idx16_select", "ca_idx16_dense"]);
        ensure!(manifest["schema"] == 1 && manifest["architecture"] == "sm_121a"
            && (manifest["functions"] == two || manifest["functions"] == three)
            && format!("{:x}", Sha256::digest(&raw)) == manifest["artifact_sha256"].as_str().unwrap_or(""),
            "bf16 index artifact contract");
        ensure!((1..=crate::config::MAX_WIDE_ROWS).contains(&rows) && (1..=1048576).contains(&columns),
            "bf16 index geometry");
        let module = cuda.module(&raw)?;
        module.function("ca_idx16_score")?;
        module.function("ca_idx16_select")?;
        // Step 2: one launch scores and selects up to eight rows (same keys and selection, probe-checked).
        // Up to two groups of eight rows (the 16-row owner's region steps): one launch a group, the same scratch.
        let dense = if manifest["functions"] == three && rows <= 2 * DENSE_ROWS && crate::config::index_bf16_dense()? {
            let function = module.function("ca_idx16_dense")?;
            function.opt_in_shared_memory(DENSE_SMEM)?;
            let mut scratch = cuda.allocate::<u32>(2 + DENSE_ROWS * 2048)?;
            scratch.zero()?;
            Some(scratch)
        } else { None };
        Ok(Self { module, rows, columns, dense })
    }

    /// Scores each row's keys into `scratch` (u16 keys relative to the row's key start, `scratch_stride` u16 a row)
    /// and writes each row's selection to `indices` ([rows][kmax], as offset + relative position: the rebased layout
    /// of the topk windows).
    ///
    /// Safety: `starts` are the rows' absolute key starts (the metadata's key ranges: a serving region's rows start
    /// past region 0's), `lengths` / `offsets` the topk windows' lengths and rebasing offsets (rows entries, every
    /// range inside the key matrix, every length at most `columns`); `scratch` holds rows x `scratch_stride` u16 (a
    /// multiple of 16, at least `columns`) and stays alive and unused by others until the stream reaches this work.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn run(&self, query: &QueryRows, keys: &KeyMatrix, starts: u64, lengths: u64, offsets: u64,
        scratch: u64, scratch_stride: usize, indices: &Buffer<i32>, kmax: usize, stream: &Stream) -> Result<()> {
        ensure!(query.rows() == self.rows && keys.rows() >= self.columns
            && scratch_stride % 16 == 0 && scratch_stride >= self.columns
            && indices.len() >= self.rows * kmax && (1..=4096).contains(&kmax),
            "bf16 index launch geometry");
        let (mut q, mut qs, mut w) = (query.payload.ptr(), query.scales.ptr(), query.weights.ptr());
        let (mut k, mut ks) = (keys.payload.ptr(), keys.scales.ptr());
        let (mut s, mut l, mut o) = (starts, lengths, offsets);
        // Key ranges are bounded by the key matrix (a region matrix holds several conversations' rows).
        let (mut rows, mut columns) = (self.rows as i32, keys.rows() as i32);
        let (mut out, mut stride) = (scratch, scratch_stride as i32);
        if let Some(dense) = self.dense.as_ref() {
            let function = self.module.function("ca_idx16_dense")?;
            for g in (0..self.rows).step_by(DENSE_ROWS) {
                let n = DENSE_ROWS.min(self.rows - g);
                // Group g's rows: queries (32 heads of 64 code bytes, one scale word and one weight each), window
                // entries, score rows and selection rows.
                let (mut gq, mut gqs, mut gw) = (q + (g * 32 * 64) as u64, qs + (g * 32 * 4) as u64, w + (g * 32 * 4) as u64);
                let (mut gs, mut gl, mut go) = (s + (g * 4) as u64, l + (g * 4) as u64, o + (g * 4) as u64);
                let (mut grows, mut gout) = (n as i32, out + (g * scratch_stride * 2) as u64);
                let (mut k_max, mut selected, mut out_stride, mut scr) =
                    (kmax as i32, indices.ptr() + (g * kmax * 4) as u64, kmax as i32, dense.ptr());
                let mut args: Vec<*mut c_void> = vec![
                    (&mut gq as *mut u64).cast(), (&mut gqs as *mut u64).cast(), (&mut gw as *mut u64).cast(),
                    (&mut k as *mut u64).cast(), (&mut ks as *mut u64).cast(),
                    (&mut gs as *mut u64).cast(), (&mut gl as *mut u64).cast(), (&mut go as *mut u64).cast(),
                    (&mut grows as *mut i32).cast(), (&mut columns as *mut i32).cast(),
                    (&mut gout as *mut u64).cast(), (&mut stride as *mut i32).cast(), (&mut k_max as *mut i32).cast(),
                    (&mut selected as *mut u64).cast(), (&mut out_stride as *mut i32).cast(), (&mut scr as *mut u64).cast(),
                ];
                unsafe { function.launch([DENSE_GRID, 1, 1], [1024, 1, 1], DENSE_SMEM, stream, &mut args)?; }
            }
            let _ = &mut rows;
            return Ok(());
        }
        let (mut no_f32, mut no_f32_stride) = (0u64, 0i32);
        let mut args: Vec<*mut c_void> = vec![
            (&mut q as *mut u64).cast(), (&mut qs as *mut u64).cast(), (&mut w as *mut u64).cast(),
            (&mut k as *mut u64).cast(), (&mut ks as *mut u64).cast(),
            (&mut s as *mut u64).cast(), (&mut l as *mut u64).cast(),
            (&mut rows as *mut i32).cast(), (&mut columns as *mut i32).cast(),
            (&mut out as *mut u64).cast(), (&mut stride as *mut i32).cast(),
            (&mut no_f32 as *mut u64).cast(), (&mut no_f32_stride as *mut i32).cast(),
        ];
        unsafe {
            self.module.function("ca_idx16_score")?.launch(
                [SCORE_GRID_X, self.rows.div_ceil(8) as u32, 1], [128, 1, 1], 0, stream, &mut args)?;
        }
        let (mut selected, mut k_max, mut out_stride) = (indices.ptr(), kmax as i32, kmax as i32);
        let mut args: Vec<*mut c_void> = vec![
            (&mut out as *mut u64).cast(), (&mut stride as *mut i32).cast(),
            (&mut s as *mut u64).cast(), (&mut l as *mut u64).cast(), (&mut o as *mut u64).cast(),
            (&mut k_max as *mut i32).cast(), (&mut selected as *mut u64).cast(), (&mut out_stride as *mut i32).cast(),
        ];
        unsafe {
            self.module.function("ca_idx16_select")?.launch(
                [self.rows as u32, 1, 1], [1024, 1, 1], 0, stream, &mut args)?;
        }
        Ok(())
    }

    /// The same scores as f32 values (bf16-exact) into `scores` (row stride `stride` floats, relative to each row's
    /// key start, as the f32 scorer writes them), for a layer whose candidate publisher or consumer mask, topk-v2 and
    /// sort then run as today.
    ///
    /// Safety: as `run` for the key ranges; `scores` holds rows x `stride` floats, `stride` at least `columns`.
    pub(crate) unsafe fn run_scores_f32(&self, query: &QueryRows, keys: &KeyMatrix, starts: u64, lengths: u64,
        scores: &Buffer<f32>, stride: usize, stream: &Stream) -> Result<()> {
        ensure!(query.rows() == self.rows && keys.rows() >= self.columns && stride >= self.columns
            && scores.len() >= self.rows * stride, "bf16 index f32 score geometry");
        let (mut q, mut qs, mut w) = (query.payload.ptr(), query.scales.ptr(), query.weights.ptr());
        let (mut k, mut ks) = (keys.payload.ptr(), keys.scales.ptr());
        let (mut s, mut l) = (starts, lengths);
        let (mut rows, mut columns) = (self.rows as i32, keys.rows() as i32);
        let (mut no_keys, mut no_keys_stride) = (0u64, 0i32);
        let (mut f, mut fstride) = (scores.ptr(), stride as i32);
        let mut args: Vec<*mut c_void> = vec![
            (&mut q as *mut u64).cast(), (&mut qs as *mut u64).cast(), (&mut w as *mut u64).cast(),
            (&mut k as *mut u64).cast(), (&mut ks as *mut u64).cast(),
            (&mut s as *mut u64).cast(), (&mut l as *mut u64).cast(),
            (&mut rows as *mut i32).cast(), (&mut columns as *mut i32).cast(),
            (&mut no_keys as *mut u64).cast(), (&mut no_keys_stride as *mut i32).cast(),
            (&mut f as *mut u64).cast(), (&mut fstride as *mut i32).cast(),
        ];
        unsafe {
            self.module.function("ca_idx16_score")?.launch(
                [SCORE_GRID_X, self.rows.div_ceil(8) as u32, 1], [128, 1, 1], 0, stream, &mut args)?;
        }
        Ok(())
    }
}
