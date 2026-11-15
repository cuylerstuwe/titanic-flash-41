//! Diagnostic only (CANADA_SCRATCH_ROUTE_CENSUS=<directory>, unset in production): how often each expert of each
//! layer is routed to. Every router adds one count per routed slot of a live row (one small kernel behind its
//! routing, in the captured graphs too), and at each request's end the rank writes its cumulative table to
//! <directory>/route-census-r<rank>.json: 43 rows (40 target layers, then the drafter's three stages) of 384
//! counters (the drafter uses the first 128). For the census of experts a traffic sample never uses (which would
//! not need to be resident). Nothing the model reads is written; outputs are unchanged.
use crate::cuda::{Buffer, Cuda, Module, Stream};
use anyhow::{Result, ensure};
use std::{cell::RefCell, ffi::c_void, path::PathBuf};

const KEYS: usize = 43;
const EXPERTS: usize = 384;
struct State { cuda: Cuda, module: Module, counters: Buffer<u32>, dir: PathBuf }
thread_local! { static STATE: RefCell<Option<State>> = const { RefCell::new(None) }; }

pub(crate) fn dir() -> Option<PathBuf> {
    static DIR: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| std::env::var_os("CANADA_SCRATCH_ROUTE_CENSUS").filter(|v| !v.is_empty()).map(PathBuf::from)).clone()
}
pub(crate) fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| dir().is_some())
}
/// Arm once a process (no-op when the census is not requested or already armed).
pub(crate) fn arm_once(cuda: &Cuda, core: &[u8]) -> Result<()> {
    let Some(dir) = dir() else { return Ok(()) };
    if STATE.with(|s| s.borrow().is_some()) { return Ok(()); }
    let module = cuda.module(core)?;
    module.function("ca_route_census")?;
    let mut counters: Buffer<u32> = cuda.allocate(KEYS * EXPERTS)?;
    counters.zero()?;
    STATE.with(|s| *s.borrow_mut() = Some(State { cuda: cuda.clone(), module, counters, dir }));
    Ok(())
}
/// One router's routed ids of this pass ([rows][topk] i32 on the device, live row count on the device).
pub(crate) fn add(key: usize, experts: usize, ids: u64, live: u64, rows: usize, topk: usize, stream: &Stream) -> Result<()> {
    STATE.with(|s| {
        let guard = s.borrow();
        let Some(state) = guard.as_ref() else { return Ok(()) };
        ensure!(key < KEYS && experts <= EXPERTS && rows > 0 && topk > 0 && state.cuda.owns_stream(stream), "route census operands");
        let (mut ids, mut live, mut rows_, mut topk_) = (ids, live, rows as i32, topk as i32);
        let (mut counters, mut experts_) = (state.counters.ptr() + (key * EXPERTS * 4) as u64, experts as i32);
        let mut args: [*mut c_void; 6] = [(&mut ids as *mut u64).cast(), (&mut live as *mut u64).cast(), (&mut rows_ as *mut i32).cast(),
            (&mut topk_ as *mut i32).cast(), (&mut counters as *mut u64).cast(), (&mut experts_ as *mut i32).cast()];
        unsafe { state.module.function("ca_route_census")?.launch([((rows * topk) as u32).div_ceil(256), 1, 1], [256, 1, 1], 0, stream, &mut args) }
    })
}
/// Write this rank's cumulative table (the caller has synchronized the model stream).
pub(crate) fn dump(rank: usize) -> Result<()> {
    STATE.with(|s| {
        let guard = s.borrow();
        let Some(state) = guard.as_ref() else { return Ok(()) };
        let bytes = unsafe { state.cuda.read_device(state.counters.ptr(), KEYS * EXPERTS * 4) }?;
        let counts: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        let table: Vec<&[u32]> = counts.chunks_exact(EXPERTS).collect();
        std::fs::create_dir_all(&state.dir)?;
        let path = state.dir.join(format!("route-census-r{rank}.json"));
        let temporary = state.dir.join(format!("route-census-r{rank}.json.tmp"));
        std::fs::write(&temporary, serde_json::to_vec(&serde_json::json!({"rank": rank, "keys": KEYS, "experts": EXPERTS, "counts": table}))?)?;
        std::fs::rename(&temporary, &path)?;
        Ok(())
    })
}
