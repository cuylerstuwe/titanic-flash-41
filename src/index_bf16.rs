//! Index selection with the reference's bf16 arithmetic: `kernels/index_bf16.cu` (adapted from pico's indexer,
//! MIT, Copyright (c) 2026 Andy Shu: licenses/pico-dgx-ds4-MIT.txt), a prebuilt sm_121a cubin under
//! a source-addressed directory under the kernel root with its manifest contract.
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
/// The manifest must name the complete hash of the kernel source compiled into this engine.
static SOURCE_SHA256: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| format!("{:x}", Sha256::digest(include_bytes!("../kernels/index_bf16.cu"))));

struct DenseLaunch {
    function: &'static str,
    grid: [u32; 3],
    scratch: Buffer<u32>,
}

pub(crate) struct IndexBf16 {
    module: Module,
    rows: usize,
    columns: usize,
    /// config::index_bf16_dense with an artifact that has ca_idx16_dense: its scratch (two counters, then [rows][2048]
    /// histograms; zero, and left zero by every launch).
    dense: Option<DenseLaunch>,
}

impl IndexBf16 {
    pub(crate) fn new(cuda: &Cuda, kernels: &Path, rows: usize, columns: usize) -> Result<Self> {
        // Different serving snapshots can coexist on the same rank. The full source and
        // artifact hashes below also protect against collisions and incomplete installs.
        let directory = format!("index-bf16-{}", &SOURCE_SHA256[..16]);
        let root = kernels.join(&directory);
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let raw = std::fs::read(root.join("index16.cubin"))?;
        let functions = serde_json::json!(["ca_idx16_score", "ca_idx16_select", "ca_idx16_dense", "ca_idx16_dense_pair", "ca_idx16_mask_row", "ca_idx16_dense_solo"]);
        ensure!(manifest["schema"] == 1 && manifest["architecture"] == "sm_121a" && manifest["functions"] == functions
            && format!("{:x}", Sha256::digest(&raw)) == manifest["artifact_sha256"].as_str().unwrap_or(""),
            "bf16 index artifact contract");
        ensure!(manifest["source_sha256"].as_str() == Some(SOURCE_SHA256.as_str()),
            "bf16 index artifact was built from another kernels/index_bf16.cu (manifest {}, engine {}): rebuild {directory}",
            manifest["source_sha256"], *SOURCE_SHA256);
        ensure!((1..=crate::config::MAX_WIDE_ROWS).contains(&rows) && (1..=1048576).contains(&columns),
            "bf16 index geometry");
        let module = cuda.module(&raw)?;
        module.function("ca_idx16_score")?;
        module.function("ca_idx16_select")?;
        module.function("ca_idx16_mask_row")?;
        let dense = if rows <= 2 * DENSE_ROWS && crate::config::index_bf16_dense()? {
            // Each eight-row group owns two completion counters and its histogram.
            // With two groups use half as many column CTAs per group on the 48-SM GB10.
            let groups = rows.div_ceil(DENSE_ROWS);
            let name = if groups == 1 { "ca_idx16_dense" } else { "ca_idx16_dense_pair" };
            module.function(name)?.opt_in_shared_memory(DENSE_SMEM)?;
            if rows == 6 {
                module.function("ca_idx16_dense_solo")?.opt_in_shared_memory(DENSE_SMEM)?;
            }
            let mut scratch = cuda.allocate::<u32>(groups * (2 + DENSE_ROWS * 2048))?;
            scratch.zero()?;
            Some(DenseLaunch { function: name, grid: [DENSE_GRID / groups as u32, groups as u32, 1], scratch })
        } else { None };
        // Once a process, at the first owner that can take the fused launch (<= 16 rows; wide prefill owners never do):
        // which artifact and whether the fused launch is in use (v1's missing kernel fell back silently).
        static ANNOUNCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if rows <= 2 * DENSE_ROWS && !ANNOUNCED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!("bf16 index artifact {}", serde_json::json!({"dir": directory, "source_version": manifest["source_version"],
                "source_sha256": &SOURCE_SHA256[..16], "fused_requested": crate::config::index_bf16_dense()?,
                "fused_in_use": dense.is_some(), "dense_grid": dense.as_ref().map(|d| d.grid), "rows": rows, "columns": columns}));
        }
        Ok(Self { module, rows, columns, dense })
    }

    /// Scores each row's keys into `scratch` (u16 keys relative to the row's key start, `scratch_stride` u16 a row)
    /// and writes each row's selection to `indices` ([rows][kmax], as offset + relative position: the rebased layout
    /// of the topk windows).
    ///
    /// On a six-row workspace, `solo_zero_based` permits a fused launch over one shared key range.
    /// Its live extent is read from the device at replay time.
    ///
    /// Safety: `starts` are the rows' absolute key starts (the metadata's key ranges: a serving region's rows start
    /// past region 0's), `lengths` / `offsets` the topk windows' lengths and rebasing offsets (rows entries, every
    /// range inside the key matrix, every length at most `columns`); `scratch` holds rows x `scratch_stride` u16 (a
    /// multiple of 16, at least `columns`) and stays alive and unused by others until the stream reaches this work.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn run(&self, query: &QueryRows, keys: &KeyMatrix, starts: u64, lengths: u64, offsets: u64,
        scratch: u64, scratch_stride: usize, indices: &Buffer<i32>, kmax: usize, solo_zero_based: bool, stream: &Stream) -> Result<()> {
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
            let name = if self.rows == 6 && solo_zero_based { "ca_idx16_dense_solo" } else { dense.function };
            let function = self.module.function(name)?;
            let (mut limit, mut selected, mut selected_stride, mut scratch) =
                (kmax as i32, indices.ptr(), kmax as i32, dense.scratch.ptr());
            let mut args: Vec<*mut c_void> = vec![
                (&mut q as *mut u64).cast(), (&mut qs as *mut u64).cast(), (&mut w as *mut u64).cast(),
                (&mut k as *mut u64).cast(), (&mut ks as *mut u64).cast(), (&mut s as *mut u64).cast(),
                (&mut l as *mut u64).cast(), (&mut o as *mut u64).cast(), (&mut rows as *mut i32).cast(),
                (&mut columns as *mut i32).cast(), (&mut out as *mut u64).cast(), (&mut stride as *mut i32).cast(),
                (&mut limit as *mut i32).cast(), (&mut selected as *mut u64).cast(),
                (&mut selected_stride as *mut i32).cast(), (&mut scratch as *mut u64).cast(),
            ];
            unsafe { function.launch(dense.grid, [1024, 1, 1], DENSE_SMEM, stream, &mut args)?; }
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
        scores: &Buffer<f32>, stride: usize, mask: Option<(&Buffer<u8>, usize)>, stream: &Stream) -> Result<()> {
        ensure!(query.rows() == self.rows && keys.rows() >= self.columns && stride >= self.columns
            && scores.len() >= self.rows * stride, "bf16 index f32 score geometry");
        let (mut q, mut qs, mut w) = (query.payload.ptr(), query.scales.ptr(), query.weights.ptr());
        let (mut k, mut ks) = (keys.payload.ptr(), keys.scales.ptr());
        let (mut s, mut l) = (starts, lengths);
        let (mut rows, mut columns) = (self.rows as i32, keys.rows() as i32);
        let (mut no_keys, mut no_keys_stride) = (0u64, 0i32);
        let (mut f, mut fstride) = (scores.ptr(), stride as i32);
        let (function, mut mask_ptr, mut mask_stride, arg_count) = match mask {
            Some((mask, bytes_per_row)) => {
                ensure!(matches!(self.rows, 6 | 16) && bytes_per_row == self.columns.div_ceil(8)
                    && mask.len() == self.rows * bytes_per_row, "bf16 mask row stride");
                ("ca_idx16_mask_row", mask.ptr(), bytes_per_row as i32, 15)
            }
            None => ("ca_idx16_score", 0u64, 0i32, 13),
        };
        let mut args: [*mut c_void; 15] = [
            (&mut q as *mut u64).cast(), (&mut qs as *mut u64).cast(), (&mut w as *mut u64).cast(),
            (&mut k as *mut u64).cast(), (&mut ks as *mut u64).cast(),
            (&mut s as *mut u64).cast(), (&mut l as *mut u64).cast(),
            (&mut rows as *mut i32).cast(), (&mut columns as *mut i32).cast(),
            (&mut no_keys as *mut u64).cast(), (&mut no_keys_stride as *mut i32).cast(),
            (&mut f as *mut u64).cast(), (&mut fstride as *mut i32).cast(),
            (&mut mask_ptr as *mut u64).cast(), (&mut mask_stride as *mut i32).cast(),
        ];
        unsafe {
            self.module.function(function)?.launch(
                [SCORE_GRID_X, self.rows.div_ceil(8) as u32, 1], [128, 1, 1], 0, stream, &mut args[..arg_count])?;
        }
        Ok(())
    }
}
