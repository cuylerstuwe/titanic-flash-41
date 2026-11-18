//! GB10 UMA admission: clean Linux file cache is reclaimable memory, while
//! device allocations, working host memory, and cgroup limits still count.
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
static HOST_WORKSPACE_RESERVED: AtomicUsize = AtomicUsize::new(0);

/// Test-only override of the physical memory floor, in MiB. Unset in production, where every
/// admission keeps [`DEFAULT_FLOOR`] of MemAvailable (see [`Available::gb10`]) free. The
/// memory-floor test sets it to measure how far below 8 GiB the GB10s stay healthy.
pub(crate) const FLOOR_OVERRIDE_ENV: &str = "CANADA_SCRATCH_CLAUDE_MEMORY_FLOOR_MIB";
pub(crate) const DEFAULT_FLOOR: usize = 8 << 30;
const MIN_FLOOR_MIB: usize = 2048;
const MAX_FLOOR_MIB: usize = 65536;

/// The floor an override value selects; `None` (unset) is the 8 GiB default.
pub(crate) fn parse_floor_override(value: Option<&str>) -> Result<usize> {
    let Some(value) = value else { return Ok(DEFAULT_FLOOR) };
    let mib: usize = value
        .trim()
        .parse()
        .with_context(|| format!("{FLOOR_OVERRIDE_ENV} must be a whole number of MiB"))?;
    ensure!(
        (MIN_FLOOR_MIB..=MAX_FLOOR_MIB).contains(&mib),
        "{FLOOR_OVERRIDE_ENV} must be {MIN_FLOOR_MIB}..={MAX_FLOOR_MIB} MiB"
    );
    Ok(mib << 20)
}

/// The serving floor with the snapshot chunk pool (Nov 17), in MiB: 2048..=8192, accepted only when
/// CANADA_SCRATCH_IMAGE_POOL_MIB is set. The pool holds every snapshot from startup, so serving no longer makes the
/// large runtime GPU allocations that fragmented aged memory; that is what lets the floor sit below 8 GiB. Without
/// the pool the 8 GiB default stands. The test override above takes precedence when both are set.
pub(crate) const SERVING_FLOOR_ENV: &str = "CANADA_SCRATCH_MEMORY_FLOOR_MIB";
const MAX_SERVING_FLOOR_MIB: usize = 8192;

/// The floor the two settings select (`test`: the test override, `serving`: the pool's serving floor).
pub(crate) fn parse_floor(test: Option<&str>, serving: Option<&str>, pool: bool) -> Result<usize> {
    if test.is_some() { return parse_floor_override(test); }
    let Some(value) = serving else { return Ok(DEFAULT_FLOOR) };
    ensure!(pool, "{SERVING_FLOOR_ENV} needs the snapshot pool (CANADA_SCRATCH_IMAGE_POOL_MIB)");
    let mib: usize = value
        .trim()
        .parse()
        .with_context(|| format!("{SERVING_FLOOR_ENV} must be a whole number of MiB"))?;
    ensure!(
        (MIN_FLOOR_MIB..=MAX_SERVING_FLOOR_MIB).contains(&mib),
        "{SERVING_FLOOR_ENV} must be {MIN_FLOOR_MIB}..={MAX_SERVING_FLOOR_MIB} MiB"
    );
    Ok(mib << 20)
}

fn floor_setting() -> &'static std::result::Result<usize, String> {
    static FLOOR: std::sync::OnceLock<std::result::Result<usize, String>> = std::sync::OnceLock::new();
    FLOOR.get_or_init(|| {
        let test = std::env::var(FLOOR_OVERRIDE_ENV).ok();
        let serving = std::env::var(SERVING_FLOOR_ENV).ok();
        let pool = std::env::var_os("CANADA_SCRATCH_IMAGE_POOL_MIB").is_some();
        parse_floor(test.as_deref(), serving.as_deref(), pool).map_err(|error| format!("{error:#}"))
    })
}

/// Startup check: refuses an invalid setting, and reports a valid one.
pub(crate) fn validate_floor() -> Result<usize> {
    let floor = floor_setting().clone().map_err(anyhow::Error::msg)?;
    if floor != DEFAULT_FLOOR {
        let kind = if std::env::var_os(FLOOR_OVERRIDE_ENV).is_some() {"test override"} else {"with the snapshot pool"};
        eprintln!("memory floor: {} MiB ({kind})", floor >> 20);
    }
    Ok(floor)
}

/// Nov 17 (CANADA_SCRATCH_RELEASE_LOAD_FILE_CACHE=1): after the startup warm-up, advise the kernel to drop the page cache
/// of every file under `root` (the kernel bundles' buffers and libraries are read once at load; their clean pages then sit
/// in the page cache as inactive memory, ~2.7 GiB on the head). Advisory: a page still mapped (a loaded library) stays,
/// and a file read again later is read from disk once. Returns (files, bytes the cache held before, bytes after).
#[cfg(target_os = "linux")]
pub(crate) fn release_file_cache(root: &Path) -> Result<(usize, usize, usize)> {
    use std::os::fd::AsRawFd;
    let resident = |file: &std::fs::File, size: usize| -> usize {
        if size == 0 { return 0; }
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ, libc::MAP_SHARED, file.as_raw_fd(), 0) };
        if ptr == libc::MAP_FAILED { return 0; }
        let mut vec = vec![0u8; size.div_ceil(page)];
        let ok = unsafe { libc::mincore(ptr, size, vec.as_mut_ptr()) } == 0;
        unsafe { libc::munmap(ptr, size); }
        if ok { vec.iter().filter(|b| **b & 1 != 0).count() * page } else { 0 }
    };
    let (mut files, mut before, mut after) = (0usize, 0usize, 0usize);
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            if kind.is_dir() { stack.push(entry.path()); continue; }
            if !kind.is_file() { continue; }
            let Ok(file) = std::fs::File::open(entry.path()) else { continue };
            let size = file.metadata().map(|m| m.len() as usize).unwrap_or(0);
            before += resident(&file, size);
            let _ = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
            after += resident(&file, size);
            files += 1;
        }
    }
    Ok((files, before, after))
}

/// The physical memory floor every admission keeps free: 8 GiB unless the test-only override
/// is set. An invalid override is refused by [`validate_floor`] at startup.
pub(crate) fn floor_bytes() -> usize {
    floor_setting().as_ref().copied().unwrap_or(DEFAULT_FLOOR)
}
#[cfg(feature="storage-fault-injection")]
thread_local! { pub(crate) static PREFILL_PREVIEW_PRESSURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
/// The finite probe can exercise a late, rank-local memory decline without
/// exhausting physical RAM. Absent from ordinary serving builds.
#[cfg(feature="storage-fault-injection")]
pub(crate) fn prefill_preview_available(actual:usize)->usize {
    if PREFILL_PREVIEW_PRESSURE.get(){0}else{actual}
}
#[path = "memory_image_storage.rs"]
mod image_storage;
pub(crate) use image_storage::{ImageStorageCharge, allocate_image_storage,retained_image_bytes};

/// Release only allocator-free host pages. Live token histories, model state,
/// immutable caches and CUDA allocations retain their storage and addresses.
/// Used by an explicit pressure experiment, never by per-token execution.
pub(crate) fn trim_free_host_pages() -> bool {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    { unsafe { libc::malloc_trim(0) != 0 } }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    { false }
}

/// Back every page of a new zeroed host buffer before admitting later work.
/// A calloc-backed Vec can otherwise defer nearly its entire physical charge
/// until cache insertions, after the runtime has advertised its UMA reserve.
pub(crate) fn committed_zeroed_bytes(bytes: usize) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    data.try_reserve_exact(bytes).context("reserve committed host buffer")?;
    data.resize(bytes, 0);
    commit_zeroed_pages(&mut data)?;
    Ok(data)
}

fn commit_zeroed_pages(data: &mut [u8]) -> Result<()> {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    ensure!(page > 0, "host page size unavailable");
    // Volatile writes cannot be folded into calloc or optimized away. Writing
    // the existing zero forces private backing even when the page was mapped
    // to the shared zero page. The last write covers an unaligned final page.
    for offset in (0..data.len()).step_by(page as usize) {
        unsafe { data.as_mut_ptr().add(offset).write_volatile(0) };
    }
    if let Some(last) = data.last_mut() {
        unsafe { std::ptr::write_volatile(last, 0) };
    }
    Ok(())
}
/// A valid input can be retried after transient host/UMA pressure subsides.
/// Keep this distinct from malformed input when the HTTP frontend reports it.
/// The second field is the required available space, or zero for a fixed
/// workspace/live-input cap that releasing optional CUDA cache cannot fix.
#[derive(Debug)]
pub(crate) struct InputCapacityError(pub &'static str, pub usize);
impl std::fmt::Display for InputCapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.0) }
}
impl std::error::Error for InputCapacityError {}
pub(crate) struct HostWorkspaceReservation(usize);
pub(crate) fn reserve_host_workspace(bytes: usize) -> Result<HostWorkspaceReservation> {
    reserve_host_workspace_with_floor(bytes, crate::memory::floor_bytes())
}
/// Raw inputs waiting for a lifetime slot cannot consume the unallocated
/// part of the admitted owners' state capacity. Existing cache/image bytes
/// already consume physical memory and must not be charged a second time.
pub(crate) fn reserve_waiting_host_workspace(bytes: usize, image_budget: usize) -> Result<HostWorkspaceReservation> {
    if image_budget == 0 { return reserve_host_workspace(bytes); }
    image_storage::with_unallocated_capacity(image_budget, |growth| {
        reserve_host_workspace_with_floor(bytes, crate::memory::floor_bytes() + growth)
    })
}
fn reserve_host_workspace_with_floor(bytes: usize, floor: usize) -> Result<HostWorkspaceReservation> {
    #[cfg(not(target_os="linux"))]
    let _ = floor;
    ensure!(bytes > 0, "empty host workspace reservation");
    HOST_WORKSPACE_RESERVED.fetch_update(Ordering::AcqRel, Ordering::Acquire,
        |current| current.checked_add(bytes).filter(|&total|total <= 4usize << 30))
        .map_err(|_| InputCapacityError("host input workspace capacity exceeded", 0))?;
    #[cfg(target_os="linux")]
    {
        let admitted = Available::gb10(usize::MAX, usize::MAX)
            .map(|m|m.admitted_available >= floor);
        if !matches!(admitted, Ok(true)) {
            HOST_WORKSPACE_RESERVED.fetch_sub(bytes, Ordering::AcqRel);
            ensure!(admitted?, InputCapacityError("host input workspace cannot retain required UMA headroom", floor + bytes));
        }
    }
    Ok(HostWorkspaceReservation(bytes))
}
impl Drop for HostWorkspaceReservation {
    fn drop(&mut self) {
        assert!(HOST_WORKSPACE_RESERVED.fetch_sub(self.0, Ordering::AcqRel) >= self.0);
    }
}
#[derive(Debug, Serialize)]
pub struct Available {
    pub cuda_immediately_free: usize,
    pub cuda_total: usize,
    pub host_available: usize,
    pub cgroup_available: Option<usize>,
    pub host_workspace_reserved: usize,
    pub admitted_available: usize,
}
fn field(text: &str, key: &str, kib: bool) -> Result<usize> {
    let mut matches = text.lines().filter_map(|line| {
        let mut words = line.split_whitespace();
        if words.next()? != key {
            return None;
        }
        Some((words.next(), words.next()))
    });
    let (value, unit) = matches
        .next()
        .with_context(|| format!("missing memory field {key}"))?;
    ensure!(matches.next().is_none(), "duplicate memory field {key}");
    let n = value.context("missing memory value")?.parse::<usize>()?;
    if kib {
        ensure!(unit == Some("kB"), "unexpected memory unit");
        n.checked_mul(1024).context("memory value overflow")
    } else {
        ensure!(unit.is_none(), "unexpected cgroup memory unit");
        Ok(n)
    }
}
fn bounded_headroom(limit: usize, current: usize, inactive: usize) -> usize {
    limit.saturating_sub(current.saturating_sub(inactive.min(current)))
}
/// Failure diagnostics only. These pages are not credited to admission: usable
/// reclamation above the existing reserve must be demonstrated separately.
pub(crate) fn percpu_free_bytes() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/zoneinfo").ok()?;
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 { return None; }
    parse_percpu_free(&text, page as usize)
}
fn parse_percpu_free(text: &str, page: usize) -> Option<usize> {
    let mut count = None;
    for line in text.lines() {
        let mut words = line.split_whitespace();
        if words.next() != Some("count:") { continue; }
        let pages = words.next()?.parse::<usize>().ok()?;
        if words.next().is_some() { return None; }
        count = Some(count.unwrap_or(0usize).checked_add(pages)?);
    }
    count?.checked_mul(page)
}
impl Available {
    pub(crate) fn gb10(cuda_immediately_free: usize, cuda_total: usize) -> Result<Self> {
        let proc = std::fs::read_to_string("/proc/meminfo")?;
        let host_available = field(&proc, "MemAvailable:", true)?;
        let root = Path::new("/sys/fs/cgroup");
        let cgroup_available = if root.join("memory.max").is_file() {
            let cap = std::fs::read_to_string(root.join("memory.max"))?;
            if cap.trim() == "max" {
                None
            } else {
                let limit = cap.trim().parse::<usize>()?;
                let current = std::fs::read_to_string(root.join("memory.current"))?
                    .trim()
                    .parse::<usize>()?;
                let stat = std::fs::read_to_string(root.join("memory.stat"))?;
                Some(bounded_headroom(
                    limit,
                    current,
                    field(&stat, "inactive_file", false)?,
                ))
            }
        } else if root.join("memory/memory.limit_in_bytes").is_file() {
            let root = root.join("memory");
            let limit = std::fs::read_to_string(root.join("memory.limit_in_bytes"))?
                .trim()
                .parse::<usize>()?;
            let current = std::fs::read_to_string(root.join("memory.usage_in_bytes"))?
                .trim()
                .parse::<usize>()?;
            let stat = std::fs::read_to_string(root.join("memory.stat"))?;
            Some(bounded_headroom(
                limit,
                current,
                field(&stat, "total_inactive_file", false)?,
            ))
        } else {
            None
        };
        let host_workspace_reserved = HOST_WORKSPACE_RESERVED.load(Ordering::Acquire);
        let admitted_available = host_available
            .min(cuda_total)
            .min(cgroup_available.unwrap_or(usize::MAX))
            .saturating_sub(host_workspace_reserved);
        Ok(Self {
            cuda_immediately_free,
            cuda_total,
            host_available,
            cgroup_available,
            host_workspace_reserved,
            admitted_available,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn memory_floor_override_is_bounded_and_defaults_to_eight_gib() {
        assert_eq!(parse_floor_override(None).unwrap(), 8usize << 30);
        assert_eq!(DEFAULT_FLOOR, 8usize << 30);
        assert_eq!(parse_floor_override(Some("4096")).unwrap(), 4usize << 30);
        assert_eq!(parse_floor_override(Some(" 6656 ")).unwrap(), 6656usize << 20);
        assert_eq!(parse_floor_override(Some("2048")).unwrap(), 2usize << 30);
        assert_eq!(parse_floor_override(Some("65536")).unwrap(), 64usize << 30);
        for bad in ["2047", "65537", "0", "-1", "4.5", "", "4GiB"] {
            assert!(parse_floor_override(Some(bad)).is_err(), "{bad:?} must be refused");
        }
        // Without either setting, every admission check keeps exactly 8 GiB.
        if std::env::var_os(FLOOR_OVERRIDE_ENV).is_none() && std::env::var_os(SERVING_FLOOR_ENV).is_none() {
            assert_eq!(floor_bytes(), 8usize << 30);
        }
    }
    #[test]
    fn serving_floor_needs_the_pool_and_stays_at_or_below_eight_gib() {
        assert_eq!(parse_floor(None, None, false).unwrap(), 8usize << 30);
        assert_eq!(parse_floor(None, None, true).unwrap(), 8usize << 30);
        assert_eq!(parse_floor(None, Some("3072"), true).unwrap(), 3usize << 30);
        assert_eq!(parse_floor(None, Some(" 4096 "), true).unwrap(), 4usize << 30);
        assert!(parse_floor(None, Some("3072"), false).is_err(), "no pool, no lower floor");
        for bad in ["2047", "8193", "0", "4.5", "", "3GiB"] {
            assert!(parse_floor(None, Some(bad), true).is_err(), "{bad:?} must be refused");
        }
        // the test override wins and keeps its own range
        assert_eq!(parse_floor(Some("2048"), Some("4096"), true).unwrap(), 2usize << 30);
        assert_eq!(parse_floor(Some("16384"), None, false).unwrap(), 16usize << 30);
        assert!(parse_floor(Some("bad"), Some("4096"), true).is_err());
    }
    #[test]
    fn percpu_diagnostic_counts_zones_without_changing_admission() {
        let zones = "Node 0, zone DMA\n pagesets\n cpu: 0\n count: 2\n high: 9\n cpu: 1\n count: 3\nNode 0, zone Normal\n cpu: 0\n count: 5\n";
        assert_eq!(parse_percpu_free(zones, 4096), Some(10 * 4096));
        assert_eq!(parse_percpu_free("count: 0\n", 4096), Some(0));
        assert_eq!(parse_percpu_free("high: 8\n", 4096), None);
        assert_eq!(parse_percpu_free("count: invalid\n", 4096), None);
        assert_eq!(parse_percpu_free("count: 1 extra\n", 4096), None);
        assert_eq!(parse_percpu_free(&format!("count: {}\n", usize::MAX), 4096), None);
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn zero_buffer_commit_backs_unaligned_first_and_last_pages() {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        assert!(page > 32 && page < 1024 * 1024);
        let len = page * 5;
        let pointer = unsafe { libc::mmap(std::ptr::null_mut(), len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
        assert_ne!(pointer, libc::MAP_FAILED);
        struct Mapping(*mut libc::c_void, usize);
        impl Drop for Mapping {
            fn drop(&mut self) { unsafe { libc::munmap(self.0, self.1); } }
        }
        let mapping = Mapping(pointer, len);
        let resident = || {
            let mut pages = vec![0u8; 5];
            assert_eq!(unsafe { libc::mincore(mapping.0, len, pages.as_mut_ptr()) }, 0);
            pages
        };
        assert!(resident().iter().all(|page| page & 1 == 0));
        let data = unsafe { std::slice::from_raw_parts_mut((pointer as *mut u8).add(13), len - 24) };
        commit_zeroed_pages(data).unwrap();
        assert!(resident().iter().all(|page| page & 1 == 1));
        assert!(data.iter().all(|&byte| byte == 0));
        assert!(committed_zeroed_bytes(0).unwrap().is_empty());
    }
    #[test]
    fn cache_can_be_reclaimed_but_limit_and_resident_charge_remain() {
        assert_eq!(bounded_headroom(28, 24, 14), 18);
        assert_eq!(bounded_headroom(28, 24, 0), 4);
        assert_eq!(bounded_headroom(28, 30, 4), 2);
        assert_eq!(bounded_headroom(28, 1, 1000), 28);
    }
    #[test]
    fn malformed_or_missing_memory_stats_fail_closed() {
        assert_eq!(
            field("MemAvailable: 123 kB\n", "MemAvailable:", true).unwrap(),
            125952
        );
        assert_eq!(
            field("inactive_file 12\n", "inactive_file", false).unwrap(),
            12
        );
        assert!(field("MemAvailable: 12 MB\n", "MemAvailable:", true).is_err());
        assert!(field("MemFree: 1000 kB\n", "MemAvailable:", true).is_err());
        assert!(
            field(
                "MemAvailable: 1 kB\nMemAvailable: 2 kB\n",
                "MemAvailable:",
                true
            )
            .is_err()
        );
    }
}

/// Diagnostic only (CANADA_SCRATCH_PROFILE_MEMORY_TRACE=1): host-available memory at one point of a wide
/// profile's construction, to measure what each part holds outside the shared row arenas. Never changes
/// any allocation.
pub(crate) fn trace_profile_memory(cuda: &crate::cuda::Cuda, rows: usize, label: &str) {
    if !crate::config::wide_rows(rows)
        || std::env::var("CANADA_SCRATCH_PROFILE_MEMORY_TRACE").as_deref() != Ok("1") {
        return;
    }
    if let Ok(a) = cuda.admission_memory() {
        eprintln!("profile memory trace rows {rows} {label}: host_available {} cuda_free {}",
            a.host_available, a.cuda_immediately_free);
    }
    // With CANADA_SCRATCH_ALLOCATION_SITES_SCREEN=1: the live ordinary allocations by call
    // site once each profile is complete; consecutive reports show what a profile owns.
    if label == "after io" {
        let _ = cuda.report_allocation_sites(&format!("after wide profile {rows}"));
    }
}
