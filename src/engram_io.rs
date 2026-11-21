//! Bounded Linux I/O for immutable packed Engram rows; no numerical operations.
//! Uses the io-uring0.7.15 crate (MIT/Apache-2.0), https://github.com/tokio-rs/io-uring.
//! Buffers belong to this reader and remain allocated until all reads complete.
use crate::engram::ROW_BYTES;
use anyhow::{Context, Result, ensure};
use io_uring::{IoUring, opcode, types};
use std::{fs::File, os::fd::AsRawFd};

pub(crate) const ENTRIES: usize = 256;
/// Direct reads (CANADA_SCRATCH_ENGRAM_DIRECT=1, see crate::engram::direct_reads) bypass the page cache: a row is
/// read as the one or two 4 KiB blocks that hold it, into a block-aligned area, and copied out. The bytes are the
/// same; no page-cache page is allocated for a miss, so a read cannot wait for memory reclaim.
const BLOCK: usize = 4096;
/// Block start, read length and the row's position inside the read for the row at `offset`.
pub(crate) fn direct_span(offset: u64) -> (u64, usize, usize) {
    let block = offset & !(BLOCK as u64 - 1);
    let inner = (offset - block) as usize;
    (block, if inner + ROW_BYTES > BLOCK { 2 * BLOCK } else { BLOCK }, inner)
}
struct Aligned {
    ptr: *mut u8,
    len: usize,
}
// One owner; the kernel writes into it only between submit and the drained completions of one batch.
unsafe impl Send for Aligned {}
impl Aligned {
    fn new(len: usize) -> Result<Self> {
        let layout = std::alloc::Layout::from_size_align(len, BLOCK).context("Engram direct workspace layout")?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        ensure!(!ptr.is_null(), "Engram direct workspace allocation");
        Ok(Self { ptr, len })
    }
}
impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align(self.len, BLOCK).unwrap()) }
    }
}

/// Reads submitted without waiting (`Reader::submit`): how many, and how each row is taken from its buffer.
struct Submitted {
    count: usize,
    direct: bool,
    need: [i32; ENTRIES],
    inner_of: [u16; ENTRIES],
}
pub(crate) struct Reader {
    // The ring is dropped before its registered storage. Drop drains any outstanding request.
    ring: IoUring,
    rows: Vec<u8>,
    direct: Option<Aligned>,
    fixed_direct: bool,
    active: bool,
    submitted: Option<Box<Submitted>>,
}
impl Reader {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            ring: IoUring::new(ENTRIES as u32)
                .context("explicit Engram io_uring experiment unavailable")?,
            rows: vec![0; ENTRIES * ROW_BYTES],
            direct: None,
            fixed_direct: false,
            active: false,
            submitted: None,
        })
    }
    fn prepare_direct(&mut self) -> Result<()> {
        if self.direct.is_none() {
            let storage = Aligned::new(ENTRIES * 2 * BLOCK)?;
            if std::env::var("CANADA_SCRATCH_ENGRAM_FIXED_DIRECT").as_deref() == Ok("1") {
                let region = libc::iovec { iov_base: storage.ptr.cast(), iov_len: storage.len };
                // The allocation stays at this address until the ring closes. Each SQE references a
                // disjoint slot in buffer zero and the existing completion checks still apply.
                match unsafe { self.ring.submitter().register_buffers(&[region]) } {
                    Ok(()) => {
                        self.fixed_direct = true;
                        static REGISTERED: std::sync::Once = std::sync::Once::new();
                        REGISTERED.call_once(|| eprintln!("Engram direct-read buffer registered ({} bytes)", storage.len));
                    }
                    Err(error) => {
                        // A low memlock allowance can prevent registration. The original O_DIRECT
                        // read uses this same allocation with identical alignment and offsets.
                        static UNREGISTERED: std::sync::Once = std::sync::Once::new();
                        UNREGISTERED.call_once(|| eprintln!("Engram using unregistered direct-read buffer: {error}"));
                    }
                }
            }
            self.direct = Some(storage);
        }
        Ok(())
    }
    fn direct_entry(&self, file: &File, slot: usize, offset: u64, length: usize) -> io_uring::squeue::Entry {
        let ptr = unsafe { self.direct.as_ref().unwrap().ptr.add(slot * 2 * BLOCK) };
        let fd = types::Fd(file.as_raw_fd());
        if self.fixed_direct {
            opcode::ReadFixed::new(fd, ptr, length as u32, 0).offset(offset).build()
        } else {
            opcode::Read::new(fd, ptr, length as u32).offset(offset).build()
        }
    }
    /// Submit reads of one file WITHOUT waiting for them (a read-ahead; config::engram_anchor_prefetch). Nothing else
    /// may use this reader until `finish_submitted` has returned; the store that owns the reader enforces it.
    pub(crate) fn submit(&mut self, file: &File, direct: bool, offsets: &[u64]) -> Result<()> {
        let count = offsets.len();
        ensure!(count > 0 && count <= ENTRIES, "Engram I/O batch geometry");
        ensure!(
            !self.active && self.submitted.is_none() && self.ring.submission().is_empty() && self.ring.completion().is_empty(),
            "Engram I/O queue was not drained"
        );
        if direct { self.prepare_direct()?; }
        let mut state = Box::new(Submitted { count, direct, need: [ROW_BYTES as i32; ENTRIES], inner_of: [0u16; ENTRIES] });
        self.active = true;
        for (i, &offset) in offsets.iter().enumerate() {
            let entry = if direct {
                let (block, len, inner) = direct_span(offset);
                state.need[i] = (inner + ROW_BYTES) as i32;
                state.inner_of[i] = inner as u16;
                self.direct_entry(file, i, block, len)
            } else {
                opcode::Read::new(
                    types::Fd(file.as_raw_fd()),
                    unsafe { self.rows.as_mut_ptr().add(i * ROW_BYTES) },
                    ROW_BYTES as u32,
                )
                .offset(offset).build()
            }
            .user_data(i as u64);
            unsafe {
                self.ring
                    .submission()
                    .push(&entry)
                    .expect("empty bounded Engram queue");
            }
        }
        loop {
            match self.ring.submit() {
                Ok(_) => break,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => fatal(&format!("Engram submit failed with reads queued: {e}")),
            }
        }
        self.submitted = Some(state);
        Ok(())
    }
    pub(crate) fn has_submitted(&self) -> bool { self.submitted.is_some() }
    /// Wait for the reads of `submit` and return their rows in submission order; `None` if any of them was short or
    /// failed (a read-ahead then publishes nothing: the lookup that follows reads the rows itself).
    pub(crate) fn finish_submitted(&mut self) -> Result<Option<&[u8]>> {
        let Some(state) = self.submitted.take() else { return Ok(Some(&[])) };
        let count = state.count;
        let mut done = 0;
        let mut seen = [false; ENTRIES];
        let mut failed = false;
        while done < count {
            match self.ring.submit_and_wait(count - done) {
                Ok(_) => (),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => fatal(&format!("Engram submit/wait failed with reads in flight: {e}")),
            }
            for completion in self.ring.completion() {
                let i = completion.user_data() as usize;
                if i >= count || seen[i] {
                    fatal("invalid or duplicate Engram I/O completion");
                }
                seen[i] = true;
                done += 1;
                let complete = if state.direct { completion.result() >= state.need[i] } else { completion.result() == state.need[i] };
                if !complete { failed = true; }
            }
        }
        self.active = false;
        if failed { return Ok(None); }
        if state.direct {
            let area = self.direct.as_ref().unwrap();
            for i in 0..count {
                let row = unsafe { std::slice::from_raw_parts(area.ptr.add(i * 2 * BLOCK + state.inner_of[i] as usize), ROW_BYTES) };
                self.rows[i * ROW_BYTES..(i + 1) * ROW_BYTES].copy_from_slice(row);
            }
        }
        Ok(Some(&self.rows[..count * ROW_BYTES]))
    }
    pub(crate) fn workspace_bytes(&self) -> usize {
        self.rows.len() + self.direct.as_ref().map_or(0, |d| d.len)
    }
    /// `direct`: `file` was opened with O_DIRECT (crate::engram::PackedShard::io).
    pub(crate) fn read(&mut self, file: &File, direct: bool, offsets: &[u64]) -> Result<&[u8]> {
        self.read_batch(offsets.len(), direct, |i| (file, offsets[i]))
    }
    /// Submit two immutable files together, retaining both handles until every
    /// completion has drained. Output rows preserve first-file/second-file order.
    pub(crate) fn read_pair(&mut self, first: &File, a: &[u64], second: &File, b: &[u64], direct: bool) -> Result<&[u8]> {
        self.read_batch(a.len()+b.len(), direct, |i| {
            if i<a.len() {(first,a[i])} else {(second,b[i-a.len()])}
        })
    }
    fn read_batch<'a>(&mut self, count: usize, direct: bool, request: impl Fn(usize)->(&'a File,u64)) -> Result<&[u8]> {
        ensure!(
            count > 0 && count <= ENTRIES,
            "Engram I/O batch geometry"
        );
        ensure!(
            !self.active && self.submitted.is_none() && self.ring.submission().is_empty() && self.ring.completion().is_empty(),
            "Engram I/O queue was not drained"
        );
        // An explicit diagnostic build can request one real short read at EOF.
        // No checkpoint bytes or file length change. The usual completion drain,
        // error propagation and whole-group recovery remain under test.
        #[cfg(feature = "storage-fault-injection")]
        let fault_offset = injected_read_offset(request(0).0)?;
        // From here until every CQE is consumed, an abnormal drop must fail the
        // process rather than free memory that the kernel could still write.
        if direct { self.prepare_direct()?; }
        // Per request: the bytes a completion must report, and (direct) where the row sits in its blocks.
        let mut need = [ROW_BYTES as i32; ENTRIES];
        let mut inner_of = [0u16; ENTRIES];
        self.active = true;
        for i in 0..count {
            let (file,offset)=request(i);
            #[cfg(feature = "storage-fault-injection")]
            let offset = if i == 0 { fault_offset.unwrap_or(offset) } else { offset };
            let entry = if direct {
                let (block, len, inner) = direct_span(offset);
                need[i] = (inner + ROW_BYTES) as i32;
                inner_of[i] = inner as u16;
                self.direct_entry(file, i, block, len)
            } else {
                opcode::Read::new(
                    types::Fd(file.as_raw_fd()),
                    unsafe { self.rows.as_mut_ptr().add(i * ROW_BYTES) },
                    ROW_BYTES as u32,
                )
                .offset(offset).build()
            }
            .user_data(i as u64);
            // Distinct disjoint rows in one allocation, never resized in flight.
            unsafe {
                self.ring
                    .submission()
                    .push(&entry)
                    .expect("empty bounded Engram queue");
            }
        }
        // Diagnostic (CANADA_SCRATCH_HOST_STAGES=1): time the submission call apart from the wait for completions.
        let timed = crate::config::host_stages().then(std::time::Instant::now);
        if timed.is_some() {
            match self.ring.submit() {
                Ok(_) => (),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
                Err(e) => fatal(&format!("Engram submit failed with reads in flight: {e}")),
            }
        }
        let submitted = timed.map(|t| t.elapsed());
        let mut done = 0;
        let mut seen = [false; ENTRIES];
        let mut failed = None;
        while done < count {
            match self.ring.submit_and_wait(count - done) {
                Ok(_) => (),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => fatal(&format!(
                    "Engram submit/wait failed with reads in flight: {e}"
                )),
            }
            for completion in self.ring.completion() {
                let i = completion.user_data() as usize;
                if i >= count || seen[i] {
                    fatal("invalid or duplicate Engram I/O completion");
                }
                seen[i] = true;
                done += 1;
                // Buffered: exactly the row. Direct: at least through the row's last byte (the file's last block
                // may be short).
                let complete = if direct { completion.result() >= need[i] } else { completion.result() == need[i] };
                if !complete && failed.is_none() {
                    failed = Some((i, completion.result()));
                }
            }
        }
        self.active = false;
        if let (Some(t), Some(s)) = (timed, submitted) {
            use std::sync::atomic::Ordering::Relaxed;
            crate::engram::IO_STATS[2].fetch_add(1, Relaxed);
            crate::engram::IO_STATS[3].fetch_add(count as u64, Relaxed);
            crate::engram::IO_STATS[4].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
            crate::engram::IO_STATS[5].fetch_add(s.as_nanos() as u64, Relaxed);
        }
        // Even a short/error read has drained the entire batch before reporting
        // failure. No caller receives partially filled row data.
        ensure!(failed.is_none(), "Engram short or failed read: {failed:?}");
        if direct {
            let area = self.direct.as_ref().unwrap();
            for i in 0..count {
                let row = unsafe { std::slice::from_raw_parts(area.ptr.add(i * 2 * BLOCK + inner_of[i] as usize), ROW_BYTES) };
                self.rows[i * ROW_BYTES..(i + 1) * ROW_BYTES].copy_from_slice(row);
            }
        }
        Ok(&self.rows[..count * ROW_BYTES])
    }
}

#[cfg(feature = "storage-fault-injection")]
fn consume_failure_marker(path: &std::path::Path) -> Result<bool> {
    // Removal is the one-shot claim when two Engram readers race to consume it.
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).context("consume isolated Engram fault marker"),
    }
}

#[cfg(feature = "storage-fault-injection")]
fn injected_read_offset(file: &File) -> Result<Option<u64>> {
    use std::{path::{Path, PathBuf}, sync::OnceLock};
    static MARKER: OnceLock<Option<PathBuf>> = OnceLock::new();
    let Some(path) = MARKER.get_or_init(||
        std::env::var_os("CANADA_SCRATCH_TEST_ENGRAM_SHORT_READ").map(PathBuf::from))
    else { return Ok(None); };
    ensure!(path.parent() == Some(Path::new("/scratch/private"))
        && path.file_name().and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("fault-engram-")),
        "Engram fault marker must be in the isolated private task directory");
    if !consume_failure_marker(path)? { return Ok(None); }
    let offset = file.metadata().context("Engram fault test file extent")?.len();
    eprintln!("isolated Engram recovery test: one read at EOF; checkpoint unchanged");
    Ok(Some(offset))
}

#[cfg(all(test, feature = "storage-fault-injection"))]
mod fault_tests {
    #[test]
    fn marker_is_claimed_once_across_concurrent_readers() {
        let path = std::env::temp_dir().join(format!("canada-engram-fault-test-{}-{}",
            std::process::id(), std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        assert!(!super::consume_failure_marker(&path).unwrap());
        std::fs::write(&path, b"recovery qualification only").unwrap();
        let claims = std::thread::scope(|scope| {
            let a = scope.spawn(|| super::consume_failure_marker(&path).unwrap());
            let b = scope.spawn(|| super::consume_failure_marker(&path).unwrap());
            usize::from(a.join().unwrap()) + usize::from(b.join().unwrap())
        });
        assert_eq!(claims, 1);
        assert!(!super::consume_failure_marker(&path).unwrap());
    }
}
fn fatal(message: &str) -> ! {
    eprintln!("{message}; refusing to release kernel-owned I/O buffers");
    std::process::abort()
}
impl Drop for Reader {
    fn drop(&mut self) {
        // A read-ahead still in flight writes into this reader's buffers: let it finish before the buffers go.
        if self.submitted.is_some() { let _ = self.finish_submitted(); }
        if self.active {
            fatal("Engram reader dropped with pending reads");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direct_span_covers_the_row() {
        // A row inside one block, a row that ends exactly at a block edge, and rows that straddle an edge.
        for (offset, block, len, inner) in [(4096u64, 4096u64, 4096usize, 0usize), (4096 + 3832, 4096, 4096, 3832),
            (4096 + 3833, 4096, 8192, 3833), (8191, 4096, 8192, 4095), (4096 + 264 * 1_000_003, 264_003_584, 4096, 1304)] {
            let got = direct_span(offset);
            assert_eq!(got, (block, len, inner), "offset {offset}");
            assert!(got.0 % BLOCK as u64 == 0 && got.0 + got.2 as u64 == offset && got.2 + ROW_BYTES <= got.1);
        }
    }
}
