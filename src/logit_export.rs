//! Diagnostic only (CANADA_SCRATCH_CLAUDE_LOGIT_EXPORT=<directory>, unset in production): for each
//! verification, every rank appends its vocabulary shard (32,320 FP32 columns per row) of the target
//! logits of the rows the step commits, and rank 0 appends each request's prompt ids. An offline
//! comparison concatenates the four shards of a row and compares it with a reference forward over the
//! same tokens. Row r of a step has its input at position start + r; its logits predict the next
//! committed token. The rows are copied before the draw, so the sampler cannot have touched them.
use crate::{cuda::{Cuda, Stream}, target::Target};
use anyhow::{ensure, Context, Result};
use std::{io::Write, path::{Path, PathBuf}};

const SHARD_BYTES: usize = 32320 * 4;

pub(crate) fn dir() -> Option<PathBuf> {
    std::env::var_os("CANADA_SCRATCH_CLAUDE_LOGIT_EXPORT").filter(|v| !v.is_empty()).map(PathBuf::from)
}

fn append(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)
        .with_context(|| format!("logit export {}", path.display()))?;
    file.write_all(bytes)?;
    Ok(())
}

/// This rank's shard of all `live` pending rows, read after the target step completes and before
/// any sampler runs.
pub(crate) fn capture(cuda: &Cuda, target: &Target, live: usize, stream: &Stream) -> Result<Vec<u8>> {
    let (local, rows) = target.pending_local_rows_for_export()?;
    ensure!(rows == live && live > 0 && local.len() >= live * 32320, "logit export capture rows");
    stream.synchronize()?;
    unsafe { cuda.read_device(local.ptr(), live * SHARD_BYTES) }
}

/// A batched step (several conversations in one verification): this rank's shard of every pending row,
/// read once after the verification and before any conversation's rows are staged or sampled (staging
/// copies a later conversation's rows over the first rows). Returns the bytes and the row count.
pub(crate) fn capture_all(cuda: &Cuda, target: &Target, stream: &Stream) -> Result<(Vec<u8>, usize)> {
    // The pending count is the conversations' rows together; each conversation's rows sit at its block offset,
    // so read every row slot of the owner's shard buffer.
    let (local, live) = target.pending_local_rows_for_export()?;
    let rows = local.len() / 32320;
    ensure!(live > 0 && rows >= live && local.len() == rows * 32320, "batched logit export capture rows");
    stream.synchronize()?;
    Ok((unsafe { cuda.read_device(local.ptr(), rows * SHARD_BYTES) }?, rows))
}

/// Append the committed rows (the first `inputs.len()` of the captured ones) and their tokens.
pub(crate) fn commit(dir: &Path, rank: usize, start: u64, inputs: &[u32], next: u32, live: usize,
    captured: &[u8]) -> Result<()> {
    ensure!(!inputs.is_empty() && inputs.len() <= live && captured.len() == live * SHARD_BYTES,
        "logit export commit rows");
    append(&dir.join(format!("rank{rank}.bin")), &captured[..inputs.len() * SHARD_BYTES])?;
    let line = serde_json::json!({"start": start, "live": live, "inputs": inputs, "next": next});
    append(&dir.join(format!("rank{rank}.jsonl")), format!("{line}\n").as_bytes())
}

/// Diagnostic only (CANADA_SCRATCH_CLAUDE_HEAD4_EXPORT=1 beside the logit export, Nov 13): the drafter's 4-bit copy of
/// the vocabulary head run on the target's own head input rows of each step (its first five rows), next to the
/// exact logits, for an offline measure of "4-bit head for the whole vocabulary + exact rescoring of each rank's
/// top rows". Serving arithmetic is untouched. Unset in production.
pub(crate) fn head4_enabled() -> bool { std::env::var("CANADA_SCRATCH_CLAUDE_HEAD4_EXPORT").as_deref() == Ok("1") }
/// CANADA_SCRATCH_CLAUDE_HEAD4_EXPORT=2 (with crate::head4_rescore installed, normally switched off by its screen
/// flag): beside the exact logits, each step also stores the installed scheme's logits of the same head input
/// (rank{r}.h4s.bin, one row a committed row) and the rows it rescored (rank{r}.h4s.jsonl).
pub(crate) fn head4_scheme_enabled() -> bool { std::env::var("CANADA_SCRATCH_CLAUDE_HEAD4_EXPORT").as_deref() == Ok("2") }
pub(crate) fn commit_head4_scheme(dir: &Path, rank: usize, start: u64, rows: usize, captured: &[u8], counts: &[i32]) -> Result<()> {
    ensure!(rows > 0 && captured.len() >= rows * SHARD_BYTES, "4-bit scheme export rows");
    append(&dir.join(format!("rank{rank}.h4s.bin")), &captured[..rows * SHARD_BYTES])?;
    let line = serde_json::json!({"start": start, "rows": rows, "rescored": &counts[..rows.min(counts.len())]});
    append(&dir.join(format!("rank{rank}.h4s.jsonl")), format!("{line}\n").as_bytes())
}
/// Append the first `rows` captured 4-bit rows of a step (rank{r}.h4.bin, rank{r}.h4.jsonl).
pub(crate) fn commit_head4(dir: &Path, rank: usize, start: u64, rows: usize, captured: &[u8]) -> Result<()> {
    ensure!(rows > 0 && captured.len() >= rows * SHARD_BYTES, "4-bit head export rows");
    append(&dir.join(format!("rank{rank}.h4.bin")), &captured[..rows * SHARD_BYTES])?;
    let line = serde_json::json!({"start": start, "rows": rows});
    append(&dir.join(format!("rank{rank}.h4.jsonl")), format!("{line}\n").as_bytes())
}
/// Diagnostic only (CANADA_SCRATCH_CLAUDE_Q_EXPORT=1 beside the logit export, Nov 13): the drafter's proposal
/// probabilities next to the target's logits, for an offline count of how often the target's token is the
/// drafter's first, second, third ... choice (the ceiling of a better drafter) and of the acceptance each draft
/// temperature would give. Unset in production.
pub(crate) fn q_enabled() -> bool { std::env::var("CANADA_SCRATCH_CLAUDE_Q_EXPORT").as_deref() == Ok("1") }

/// This rank's shard (32,320 FP64 columns a draft slot) of the proposal probabilities the step drew from.
pub(crate) fn capture_q(cuda: &Cuda, q: &crate::cuda::Buffer<f64>, slots: usize, stream: &Stream) -> Result<Vec<u8>> {
    ensure!((1..=5).contains(&slots) && q.len() >= slots * 32320, "proposal export capture slots");
    stream.synchronize()?;
    unsafe { cuda.read_device(q.ptr(), slots * 32320 * 8) }
}

/// One FP32 row per committed target row, aligned with rank{rank}.bin: row r holds draft slot r's probabilities
/// (the proposal for the token that row r's logits predict), or zeros when that row had no drafter proposal
/// (a lookup or target-only step, or the row after the last draft).
pub(crate) fn commit_q(dir: &Path, rank: usize, start: u64, committed: usize, drafts: &[u32],
    captured: Option<&[u8]>) -> Result<()> {
    let slots = captured.map_or(0, |c| c.len() / (32320 * 8));
    ensure!(committed >= 1 && slots <= 5 && captured.is_none_or(|c| c.len() == slots * 32320 * 8),
        "proposal export commit rows");
    let mut out = vec![0u8; committed * SHARD_BYTES];
    if let Some(c) = captured {
        for (i, v) in c[..committed.min(slots) * 32320 * 8].chunks_exact(8).enumerate() {
            let x = f64::from_le_bytes(v.try_into().unwrap()) as f32;
            out[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
    }
    append(&dir.join(format!("rank{rank}.q.bin")), &out)?;
    let line = serde_json::json!({"start": start, "committed": committed, "slots": slots, "drafts": drafts});
    append(&dir.join(format!("rank{rank}.q.jsonl")), format!("{line}\n").as_bytes())
}

/// Rank 0 records each request's prompt ids as the worker received them.
pub(crate) fn prompt(dir: &Path, rank: usize, request: u64, tokens: &[u32]) -> Result<()> {
    if rank != 0 { return Ok(()); }
    let line = serde_json::json!({"request": request, "tokens": tokens});
    append(&dir.join("prompts.jsonl"), format!("{line}\n").as_bytes())
}
