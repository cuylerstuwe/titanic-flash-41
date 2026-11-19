//! Disk tier for conversation snapshots (CANADA_SCRATCH_DISK_TIER, Nov 18 2026).
//!
//! A snapshot (state_image::Image) is one contiguous device buffer whose descriptors point at this process's live
//! regions, so a byte copy on local NVMe restores exactly within the same engine process (and is worthless after it:
//! the tier directory is emptied at startup). Copies go through a 4 KiB-aligned host staging buffer with O_DIRECT so
//! nothing lands in the page cache (a GiB of dirty pages would press the serving memory floor). Every 16 MiB chunk
//! carries a SHA-256 digest (hashed in parallel), checked on every read; a mismatch is an error and the caller drops
//! the snapshot (re-prefill) rather than restoring wrong bytes.
use crate::cuda::Buffer;
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Digest granularity: 4 MiB, so even a small snapshot's chunks hash in parallel (Nov 18: a 14 MB image was one
/// 16 MiB chunk on one core).
const CHUNK: usize = 4 << 20;
/// Bytes read back per step (reads are on the request's path: larger pieces, fewer calls).
const BATCH: usize = 32 * CHUNK;
const ALIGN: usize = 4096;

static WRITTEN_BYTES: AtomicU64 = AtomicU64::new(0);
static READ_BYTES: AtomicU64 = AtomicU64::new(0);
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Bytes this process has written / read through the tier (the daily write budget reads the first).
pub(crate) fn written_bytes() -> u64 { WRITTEN_BYTES.load(Ordering::Relaxed) }
#[allow(dead_code)]
pub(crate) fn read_bytes() -> u64 { READ_BYTES.load(Ordering::Relaxed) }

/// One snapshot's copy on disk. Dropping the record deletes the file.
pub(crate) struct DiskFile {
    path: PathBuf,
    bytes: usize,
    digests: Vec<[u8; 32]>,
}
impl DiskFile {
    pub(crate) fn bytes(&self) -> usize { self.bytes }
    /// Space the file takes (padded to the direct-I/O block).
    pub(crate) fn disk_bytes(&self) -> usize { self.bytes.div_ceil(ALIGN) * ALIGN }
}
impl Drop for DiskFile {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.path); }
}

/// A 4 KiB-aligned host buffer for direct I/O.
struct Staging { ptr: *mut u8, len: usize }
impl Staging {
    fn new(len: usize) -> Result<Self> {
        let layout = std::alloc::Layout::from_size_align(len, ALIGN)?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        ensure!(!ptr.is_null(), "disk tier staging allocation");
        Ok(Self { ptr, len })
    }
    fn slice(&mut self, n: usize) -> &mut [u8] {
        assert!(n <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr, n) }
    }
}
impl Drop for Staging {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align_unchecked(self.len, ALIGN)) }
    }
}

fn open(path: &Path, write: bool) -> Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    if write { options.write(true).create_new(true); } else { options.read(true); }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_DIRECT).mode(0o600);
    }
    options.open(path).with_context(|| format!("disk tier open {}", path.display()))
}

fn digests(data: &[u8]) -> Vec<[u8; 32]> {
    data.par_chunks(CHUNK).map(|c| Sha256::digest(c).into()).collect()
}

/// A fresh file name under `dir` for one rank's copy of one snapshot.
pub(crate) fn path_for(dir: &Path, kind: &str) -> PathBuf {
    let n = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    dir.join(format!("{}-{:08}-{}.img", std::process::id(), n, kind))
}

/// Empty the tier directory (files of an earlier process are worthless: their descriptors named its regions).
pub(crate) fn reset_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("disk tier directory {}", dir.display()))?;
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "img") { std::fs::remove_file(&path)?; }
    }
    Ok(())
}

/// Bytes free on the filesystem holding `dir`.
pub(crate) fn free_bytes(dir: &Path) -> Result<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let c = std::ffi::CString::new(dir.as_os_str().as_bytes())?;
        let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
        ensure!(unsafe { libc::statvfs(c.as_ptr(), &mut s) } == 0, "statvfs {}", dir.display());
        return Ok(s.f_bavail as u64 * s.f_frsize as u64);
    }
    #[allow(unreachable_code)]
    Ok(0)
}

/// Bytes one idle command writes (eight digest chunks; slices start at multiples of it, so chunk digests line up).
pub(crate) const SLICE: usize = 8 * CHUNK;

/// A copy written one slice at a time (one slice an idle command), so a request that arrives meanwhile waits for at most
/// one slice, not a whole snapshot. Dropping it unfinished deletes the partial file.
pub(crate) struct PendingWrite {
    file: std::fs::File,
    record: DiskFile,
    at: usize,
    staging: Staging,
}
impl PendingWrite {
    pub(crate) fn begin(bytes: usize, path: &Path) -> Result<Self> {
        ensure!(bytes > 0, "disk tier write extent");
        let file = open(path, true)?;
        let record = DiskFile { path: path.to_path_buf(), bytes, digests: Vec::with_capacity(bytes.div_ceil(CHUNK)) };
        Ok(Self { file, record, at: 0, staging: Staging::new(SLICE)? })
    }
    pub(crate) fn bytes(&self) -> usize { self.record.bytes }
    pub(crate) fn done(&self) -> bool { self.at >= self.record.bytes }
    /// Copy the next slice of `buffer` (the device copy the write began from; images are immutable) to the file.
    pub(crate) fn step(&mut self, buffer: &Buffer<u8>) -> Result<()> {
        ensure!(self.record.bytes <= buffer.len() && !self.done(), "disk tier write step");
        let n = (self.record.bytes - self.at).min(SLICE);
        let padded = n.div_ceil(ALIGN) * ALIGN;
        let data = self.staging.slice(padded);
        buffer.download_part_into(self.at, &mut data[..n])?;
        data[n..].fill(0);
        self.record.digests.extend(digests(&data[..n]));
        self.file.write_all(data).with_context(|| format!("disk tier write {}", self.record.path.display()))?;
        self.at += n;
        if self.done() { WRITTEN_BYTES.fetch_add(self.record.disk_bytes() as u64, Ordering::Relaxed); }
        Ok(())
    }
    /// The finished copy's record (the file stays until the record is dropped).
    pub(crate) fn finish(self) -> Result<DiskFile> {
        ensure!(self.done(), "unfinished disk tier write");
        Ok(self.record)
    }
}

/// Seconds a read back spent in each phase: opening and reading the file, checking digests, uploading to the device.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct ReadTiming { pub(crate) read: f64, pub(crate) digest: f64, pub(crate) upload: f64 }

/// Read a copy back into a device buffer of at least its size, checking every chunk's digest first.
pub(crate) fn read(record: &DiskFile, buffer: &mut Buffer<u8>) -> Result<ReadTiming> {
    ensure!(record.bytes <= buffer.len(), "disk tier read extent");
    let mut timing = ReadTiming::default();
    let t = std::time::Instant::now();
    let mut file = open(&record.path, false)?;
    // Staging sized to the copy (at most one batch): a small snapshot does not fault in a whole batch of pages.
    let mut staging = Staging::new(record.disk_bytes().min(BATCH))?;
    timing.read += t.elapsed().as_secs_f64();
    let mut at = 0;
    let mut chunk = 0;
    while at < record.bytes {
        let n = (record.bytes - at).min(BATCH);
        let padded = n.div_ceil(ALIGN) * ALIGN;
        let data = staging.slice(padded);
        let t = std::time::Instant::now();
        file.read_exact(data).with_context(|| format!("disk tier read {}", record.path.display()))?;
        timing.read += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let got = digests(&data[..n]);
        ensure!(got.as_slice() == &record.digests[chunk..chunk + got.len()],
            "disk tier digest mismatch in {} at byte {at}", record.path.display());
        timing.digest += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        buffer.upload_part(at, &data[..n])?;
        timing.upload += t.elapsed().as_secs_f64();
        chunk += got.len();
        at += n;
    }
    READ_BYTES.fetch_add(record.disk_bytes() as u64, Ordering::Relaxed);
    Ok(timing)
}

/// Seconds to prefill `rows` new rows by the prefill planner's chunk table (CANADA_SCRATCH_ENCODER_CHUNK_COSTS): linear
/// between its points, proportional below the first, the last two points' slope above the last.
fn prefill_seconds(rows: usize, costs: &[(usize, f64)]) -> f64 {
    let mut t = costs.to_vec();
    t.sort_by_key(|&(r, _)| r);
    let Some(&(r0, s0)) = t.first() else { return 0.0 };
    let x = rows as f64;
    if rows <= r0 { return s0 * x / r0 as f64; }
    for w in t.windows(2) {
        let ((ra, sa), (rb, sb)) = (w[0], w[1]);
        if rows <= rb { return sa + (sb - sa) * (x - ra as f64) / (rb - ra) as f64; }
    }
    let n = t.len();
    if n < 2 { return s0 * x / r0 as f64; }
    let ((ra, sa), (rb, sb)) = (t[n - 2], t[n - 1]);
    sb + (sb - sa) / (rb - ra) as f64 * (x - rb as f64)
}
/// The decoder pass an exact retry's image (prompt-end logits) saves on top of the rows (Nov 12 small-prefill anatomy).
const DECODER_TAIL_SECONDS: f64 = 0.085;
/// Whether reading a parked snapshot of `bytes` back (at `read_bytes_per_s`, plus `fixed_s` per reload) beats
/// prefilling the `gain` rows it saves over the best resident entry (`exact`: it is an exact retry's image, which also
/// saves the decoder pass). A follow-up whose near-end checkpoint is resident saves only the few hundred rows between it
/// and the prompt end: those usually prefill faster than a disk read, so the parked prompt-end image stays on disk.
/// Nov 18: s1618 reloads of 14K-114K-token snapshots took 0.099-0.118 s; s1620 (sized staging, 4 MiB digest chunks)
/// 35-82 ms = ~25-30 ms fixed + a read at ~2.2 GB/s (config defaults READ_FIXED_MS 30, READ_MBPS 2200).
pub(crate) fn reload_pays(bytes: usize, gain: usize, exact: bool, read_bytes_per_s: f64, fixed_s: f64, costs: &[(usize, f64)]) -> bool {
    let read = bytes as f64 / read_bytes_per_s + fixed_s;
    let prefill = prefill_seconds(gain, costs) + if exact { DECODER_TAIL_SECONDS } else { 0.0 };
    read < prefill
}

#[cfg(test)]
mod tests {
    #[test]
    fn reload_pays_only_when_the_read_beats_the_prefill() {
        let costs = [(128, 0.09), (512, 0.166), (1024, 0.239), (2048, 0.339), (4096, 0.60), (8192, 0.97)];
        assert!((super::prefill_seconds(768, &costs) - 0.2025).abs() < 1e-9);
        assert!((super::prefill_seconds(16384, &costs) - 1.71).abs() < 1e-9);
        let gib = 1usize << 30;
        // A switched-away 250K conversation (0.23 GiB, nothing resident): reload.
        assert!(super::reload_pays(gib * 23 / 100, 250_000, false, 2e9, 0.010, &costs));
        // A follow-up whose near-end checkpoint is resident, 300 rows short of a 0.9 GiB prompt-end image: prefill.
        assert!(!super::reload_pays(gib * 9 / 10, 300, false, 2e9, 0.010, &costs));
        // The same rows of a small image at a fast rate: reload.
        assert!(super::reload_pays(64 << 20, 300, true, 4e9, 0.010, &costs));
        // With the measured fixed cost (~0.1 s), a follow-up 150 rows short of a small image stays on its resident
        // checkpoint (0.094 s of prefill); at 300 rows (0.124 s) the read wins.
        assert!(!super::reload_pays(16 << 20, 150, false, 5e9, 0.100, &costs));
        assert!(super::reload_pays(16 << 20, 300, false, 5e9, 0.100, &costs));
    }
    #[test]
    fn digests_cover_every_chunk_and_differ_on_change() {
        let mut data = vec![7u8; super::CHUNK * 2 + 5];
        let a = super::digests(&data);
        assert_eq!(a.len(), 3);
        data[super::CHUNK + 1] ^= 1;
        let b = super::digests(&data);
        assert_eq!(a[0], b[0]);
        assert_ne!(a[1], b[1]);
        assert_eq!(a[2], b[2]);
    }
    #[test]
    fn reset_dir_removes_only_snapshot_files() {
        let dir = std::env::temp_dir().join(format!("disk-tier-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("1-00000000-x.img"), b"x").unwrap();
        std::fs::write(dir.join("keep.txt"), b"y").unwrap();
        super::reset_dir(&dir).unwrap();
        assert!(!dir.join("1-00000000-x.img").exists() && dir.join("keep.txt").exists());
        assert!(super::free_bytes(&dir).unwrap() > 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
