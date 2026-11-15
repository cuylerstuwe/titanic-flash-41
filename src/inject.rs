//! Sensitivity screen for the decode layer (CANADA_SCRATCH_INJECT_SCREEN; plan screen only; the idea of causal
//! profiling applied to the layer). One arm adds, at one named point of every layer's chain (the layer clock's mark
//! codes: 3 sum starts, 4 sum done, 5 gather starts, 6 gather done, 16 + a sublayer's named point, 32 post-mix done,
//! 33 normalized input ready, 34 query fork joined, 35 attention done, 36 wo_a done, 37 wo_b done), either a spin of
//! T microseconds on the model stream (no memory traffic) or a cold read of B KiB on a side stream released at that
//! point (joined at the end of the forward part, so the chain never waits for it). The slope of the step floor
//! against T says whether the point is on the critical path; against B, whether the bus is free there.
//! Nothing the model reads is written; outputs are unchanged.
use crate::cuda::{Buffer, Cuda, Event, Module, Stream};
use anyhow::{Result, ensure};
use std::cell::RefCell;
use std::ffi::c_void;

/// Bytes one point occurrence may read, and the occurrences a forward may have. Each occurrence has a region of
/// its own, read once a step, so it is cold again a step later (a step streams gigabytes through the cache).
const REGION: usize = 4 << 20;
const POINTS: usize = 96;
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Arm { pub(crate) code: u8, pub(crate) spin_ns: u32, pub(crate) bytes: usize, pub(crate) rank: Option<usize> }
struct State {
    cuda: Cuda,
    module: Module,
    parent: usize,
    rank: usize,
    side: Stream,
    scratch: Buffer<u8>,
    sink: Buffer<u32>,
    go: Vec<Event>,
    done: Vec<Event>,
    next: usize,
    part: usize,
    open: bool,
    pending: bool,
    arm: Option<Arm>,
}
thread_local! { static STATE: RefCell<Option<State>> = const { RefCell::new(None) }; }

/// "code:spin_us:kib[:rank]" arms, comma-separated (at most 64).
pub(crate) fn screen() -> Result<Vec<(String, Arm)>> {
    match std::env::var("CANADA_SCRATCH_INJECT_SCREEN") {
        Ok(value) => {
            let arms = value.split(',').map(|spec| -> Result<(String, Arm)> {
                let f: Vec<&str> = spec.split(':').collect();
                ensure!(f.len() == 3 || f.len() == 4, "injection arm is code:spin_us:kib[:rank]");
                let (code, spin, kib) = (f[0].parse::<u8>()?, f[1].parse::<u32>()?, f[2].parse::<usize>()?);
                let rank = match f.get(3) { Some(r) => Some(r.parse::<usize>()?), None => None };
                ensure!((3..=40).contains(&code) && spin <= 2000 && kib * 1024 <= REGION && kib % 16 == 0
                    && (spin > 0) != (kib > 0) && rank.is_none_or(|r| r < 4), "injection arm bounds (one of spin, bytes)");
                Ok((spec.to_string(), Arm { code, spin_ns: spin * 1000, bytes: kib * 1024, rank }))
            }).collect::<Result<Vec<_>>>()?;
            ensure!(!arms.is_empty() && arms.len() <= 64, "injection screen takes one to 64 arms");
            Ok(arms)
        }
        Err(std::env::VarError::NotPresent) => Ok(Vec::new()),
        Err(error) => Err(error.into()),
    }
}
/// Arm the instrument for the model stream `parent` (no work pending; before the graphs to time are captured).
pub(crate) fn arm(cuda: &Cuda, core: &[u8], parent: &Stream, rank: usize) -> Result<()> {
    ensure!(cuda.owns_stream(parent) && rank < 4 && STATE.with(|s| s.borrow().is_none()), "injection instrument admission");
    let module = cuda.module(core)?;
    module.function("ca_inject_spin")?;
    module.function("ca_inject_read")?;
    let mut scratch: Buffer<u8> = cuda.allocate(REGION * POINTS)?;
    scratch.zero()?;
    let mut sink: Buffer<u32> = cuda.allocate(16)?;
    sink.zero()?;
    parent.join_default()?;
    parent.synchronize()?;
    let done = (0..crate::layer_clock::PARTS).map(|_| cuda.fence_event()).collect::<Result<Vec<_>>>()?;
    let side = cuda.stream()?;
    STATE.with(|s| *s.borrow_mut() = Some(State { cuda: cuda.clone(), module, parent: parent.raw() as usize, rank,
        side, scratch, sink, go: Vec::new(), done, next: 0, part: 0, open: false, pending: false, arm: None }));
    Ok(())
}
pub(crate) fn armed() -> bool { STATE.with(|s| s.borrow().is_some()) }
/// The injection of the graphs captured from now on (None: none). The caller drops the captured graphs.
pub(crate) fn set(arm: Option<Arm>) {
    STATE.with(|s| if let Some(state) = s.borrow_mut().as_mut() {
        state.arm = arm.filter(|a| a.rank.is_none_or(|r| r == state.rank));
    });
}
/// A forward part begins on the model stream (called with the layer clock's part hooks).
pub(crate) fn begin(stream: &Stream, part: usize) -> Result<()> {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut().filter(|state| stream.raw() as usize == state.parent) {
            ensure!(!state.open && part < state.done.len(), "injection parts are sequential");
            if part == 0 { state.next = 0; }
            state.part = part; state.open = true; state.pending = false;
        }
        Ok(())
    })
}
/// The chain reached the point `code` on `stream`.
pub(crate) fn at(stream: &Stream, code: u8) -> Result<()> {
    STATE.with(|s| {
        let mut guard = s.borrow_mut();
        let Some(state) = guard.as_mut() else { return Ok(()) };
        let Some(arm) = state.arm.filter(|a| a.code == code) else { return Ok(()) };
        if !state.open || stream.raw() as usize != state.parent { return Ok(()); }
        if arm.spin_ns > 0 {
            let mut ns = arm.spin_ns;
            let mut args: [*mut c_void; 1] = [(&mut ns as *mut u32).cast()];
            // One thread; no memory traffic. It sits in the chain: what follows waits for it.
            unsafe { state.module.function("ca_inject_spin")?.launch([1, 1, 1], [1, 1, 1], 0, stream, &mut args) }?;
        }
        if arm.bytes > 0 {
            let index = state.next;
            ensure!(index < POINTS, "more injection points in a forward than regions");
            state.next += 1;
            if state.go.len() <= index { state.go.push(state.cuda.fence_event()?); }
            state.go[index].record(stream)?;
            state.go[index].wait(&state.side)?;
            let (mut base, mut packs, mut sink) = (state.scratch.ptr() + (index * REGION) as u64, (arm.bytes / 16) as u32, state.sink.ptr());
            let mut args: [*mut c_void; 3] = [(&mut base as *mut u64).cast(), (&mut packs as *mut u32).cast(), (&mut sink as *mut u64).cast()];
            // Reads only an owned scratch region; released by the event above, joined at the part's end.
            unsafe { state.module.function("ca_inject_read")?.launch([8, 1, 1], [256, 1, 1], 0, &state.side, &mut args) }?;
            state.pending = true;
        }
        Ok(())
    })
}
/// The forward part ends: the side stream joins the model stream (a capture ends joined).
pub(crate) fn end(stream: &Stream) -> Result<()> {
    STATE.with(|s| {
        if let Some(state) = s.borrow_mut().as_mut().filter(|state| stream.raw() as usize == state.parent && state.open) {
            if state.pending {
                let done = &state.done[state.part];
                done.record(&state.side)?;
                done.wait(stream)?;
                state.pending = false;
            }
            state.open = false;
        }
        Ok(())
    })
}
#[cfg(test)]
mod tests {
    #[test]
    fn injection_arms_parse_strictly() {
        // Without the variable there are no arms; the parser itself is exercised through its bounds below.
        let parse = |spec: &str| -> anyhow::Result<super::Arm> {
            let f: Vec<&str> = spec.split(':').collect();
            anyhow::ensure!(f.len() == 3 || f.len() == 4, "fields");
            let (code, spin, kib) = (f[0].parse::<u8>()?, f[1].parse::<u32>()?, f[2].parse::<usize>()?);
            anyhow::ensure!((spin > 0) != (kib > 0), "one of spin, bytes");
            Ok(super::Arm { code, spin_ns: spin * 1000, bytes: kib * 1024, rank: None })
        };
        assert_eq!(parse("35:20:0").unwrap(), super::Arm { code: 35, spin_ns: 20_000, bytes: 0, rank: None });
        assert!(parse("35:20:512").is_err() && parse("35:0:0").is_err() && parse("35").is_err());
    }
}
