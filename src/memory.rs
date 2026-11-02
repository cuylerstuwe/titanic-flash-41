//! GB10 UMA admission: clean Linux file cache is reclaimable memory, while
//! device allocations, working host memory, and cgroup limits still count.
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
static HOST_WORKSPACE_RESERVED: AtomicUsize = AtomicUsize::new(0);
#[path = "memory_image_storage.rs"]
mod image_storage;
pub(crate) use image_storage::{ImageStorageCharge, allocate_image_storage};

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
    reserve_host_workspace_with_floor(bytes, 8usize << 30)
}
/// Raw inputs waiting for a lifetime slot cannot consume the unallocated
/// part of the admitted owners' state capacity. Existing cache/image bytes
/// already consume physical memory and must not be charged a second time.
pub(crate) fn reserve_waiting_host_workspace(bytes: usize, image_budget: usize) -> Result<HostWorkspaceReservation> {
    if image_budget == 0 { return reserve_host_workspace(bytes); }
    image_storage::with_unallocated_capacity(image_budget, |growth| {
        reserve_host_workspace_with_floor(bytes, (8usize << 30) + growth)
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
