//! Bounded Linux I/O for immutable packed Engram rows; no numerical operations.
//! Uses the io-uring0.7.15 crate (MIT/Apache-2.0), https://github.com/tokio-rs/io-uring.
//! Buffers belong to this reader and remain allocated until all reads complete.
use crate::engram::ROW_BYTES;
use anyhow::{Context, Result, ensure};
use io_uring::{IoUring, opcode, types};
use std::{fs::File, os::fd::AsRawFd};

pub(crate) const ENTRIES: usize = 256;

pub(crate) struct Reader {
    ring: IoUring,
    rows: Vec<u8>,
    active: bool,
}
impl Reader {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            ring: IoUring::new(ENTRIES as u32)
                .context("explicit Engram io_uring experiment unavailable")?,
            rows: vec![0; ENTRIES * ROW_BYTES],
            active: false,
        })
    }
    pub(crate) fn workspace_bytes(&self) -> usize {
        self.rows.len()
    }
    pub(crate) fn read(&mut self, file: &File, offsets: &[u64]) -> Result<&[u8]> {
        self.read_batch(offsets.len(), |i| (file, offsets[i]))
    }
    /// Submit two immutable files together, retaining both handles until every
    /// completion has drained. Output rows preserve first-file/second-file order.
    pub(crate) fn read_pair(&mut self, first: &File, a: &[u64], second: &File, b: &[u64]) -> Result<&[u8]> {
        self.read_batch(a.len()+b.len(), |i| {
            if i<a.len() {(first,a[i])} else {(second,b[i-a.len()])}
        })
    }
    fn read_batch<'a>(&mut self, count: usize, request: impl Fn(usize)->(&'a File,u64)) -> Result<&[u8]> {
        ensure!(
            count > 0 && count <= ENTRIES,
            "Engram I/O batch geometry"
        );
        ensure!(
            !self.active && self.ring.submission().is_empty() && self.ring.completion().is_empty(),
            "Engram I/O queue was not drained"
        );
        // An explicit diagnostic build can request one real short read at EOF.
        // No checkpoint bytes or file length change. The usual completion drain,
        // error propagation and whole-group recovery remain under test.
        #[cfg(feature = "storage-fault-injection")]
        let fault_offset = injected_read_offset(request(0).0)?;
        // From here until every CQE is consumed, an abnormal drop must fail the
        // process rather than free memory that the kernel could still write.
        self.active = true;
        for i in 0..count {
            let (file,offset)=request(i);
            #[cfg(feature = "storage-fault-injection")]
            let offset = if i == 0 { fault_offset.unwrap_or(offset) } else { offset };
            let entry = opcode::Read::new(
                types::Fd(file.as_raw_fd()),
                unsafe { self.rows.as_mut_ptr().add(i * ROW_BYTES) },
                ROW_BYTES as u32,
            )
            .offset(offset)
            .build()
            .user_data(i as u64);
            // Distinct disjoint rows in one allocation, never resized in flight.
            unsafe {
                self.ring
                    .submission()
                    .push(&entry)
                    .expect("empty bounded Engram queue");
            }
        }
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
                if completion.result() != ROW_BYTES as i32 && failed.is_none() {
                    failed = Some((i, completion.result()));
                }
            }
        }
        self.active = false;
        // Even a short/error read has drained the entire batch before reporting
        // failure. No caller receives partially filled row data.
        ensure!(failed.is_none(), "Engram short or failed read: {failed:?}");
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
        if self.active {
            fatal("Engram reader dropped with pending reads");
        }
    }
}
