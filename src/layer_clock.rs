//! GPU-side clock for decode-step anatomy without a profiler (CANADA_SCRATCH_LAYER_CLOCK=1; screens, and since
//! Nov 14 a serving load's six-row verifier: one "claude clock" line a request and rank, see `parts_us`).
//! A one-thread kernel stores the device's global timer (ns) in a slot. Marks run on their own side stream, forked
//! from the model stream by an event: a mark starts when the kernel before it on the model stream ends and delays
//! nothing there. They are issued at the ends of each forward part and before and after every native decode sum and
//! gather, so every rank records when its local work ended and when each collective completed. Hints, arithmetic
//! and every model dependency are unchanged; graphs captured while the clock is armed keep its slots alive (the
//! clock is never disarmed).
use crate::cuda::{Buffer, Cuda, Event, Module, Stream};
use anyhow::{Result, ensure};
use std::cell::RefCell;
use std::ffi::c_void;

/// Slots of one forward part (initial, before Engram 14, after it, or a screen's own graph).
pub(crate) const PART: usize = 2048;
pub(crate) const PARTS: usize = 4;
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(crate) enum Site { Start = 1, End = 2, SumBefore = 3, SumAfter = 4, GatherBefore = 5, GatherAfter = 6 }

struct Clock {
    cuda: Cuda,
    module: Module,
    side: Stream,
    parent: usize,
    slots: Buffer<u64>,
    kinds: Vec<u8>,
    events: Vec<Option<Event>>,
    done: Vec<Event>,
    next: usize,
    limit: usize,
    part: usize,
    open: bool,
}
thread_local! { static CLOCK: RefCell<Option<Clock>> = const { RefCell::new(None) }; }

pub(crate) fn requested() -> bool {
    std::env::var("CANADA_SCRATCH_LAYER_CLOCK").as_deref() == Ok("1")
}
pub(crate) fn armed() -> bool {
    CLOCK.with(|c| c.borrow().is_some())
}
/// Arm the clock for the model stream `parent`. Call with no work pending and before the graphs to observe
/// are captured.
pub(crate) fn arm(cuda: &Cuda, core: &[u8], parent: &Stream) -> Result<()> {
    ensure!(requested() && cuda.owns_stream(parent) && !armed(), "layer clock admission");
    let module = cuda.module(core)?;
    module.function("ca_clock_mark")?;
    let mut slots: Buffer<u64> = cuda.allocate(PART * PARTS)?;
    slots.zero()?;
    parent.join_default()?;
    parent.synchronize()?;
    let done = (0..PARTS).map(|_| cuda.event()).collect::<Result<Vec<_>>>()?;
    let clock = Clock { cuda: cuda.clone(), module, side: cuda.stream()?, parent: parent.raw() as usize, slots,
        kinds: vec![0; PART * PARTS], events: (0..PART * PARTS).map(|_| None).collect(), done,
        next: 0, limit: 0, part: 0, open: false };
    CLOCK.with(|c| *c.borrow_mut() = Some(clock));
    Ok(())
}
impl Clock {
    fn issue(&mut self, parent: &Stream, site: Site) -> Result<()> { self.issue_code(parent, site as u8) }
    fn issue_code(&mut self, parent: &Stream, code: u8) -> Result<()> {
        ensure!(self.next < self.limit, "layer clock part holds more marks than its slots");
        let slot = self.next;
        if self.events[slot].is_none() { self.events[slot] = Some(self.cuda.event()?); }
        let event = self.events[slot].as_ref().unwrap();
        event.record(parent)?;
        event.wait(&self.side)?;
        let mut address = self.slots.ptr() + 8 * slot as u64;
        let mut args: [*mut c_void; 1] = [(&mut address as *mut u64).cast()];
        // One thread; writes one owned, aligned slot.
        unsafe { self.module.function("ca_clock_mark")?.launch([1, 1, 1], [1, 1, 1], 0, &self.side, &mut args) }?;
        self.kinds[slot] = code;
        self.next += 1;
        Ok(())
    }
}
thread_local! {
    /// Serving use (Nov 14): None = screens, every forward on the model stream is marked; Some(live) = a serving
    /// load, where only the six-row verifier's forwards are marked (its step sets `live` for its own duration).
    static SERVING: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}
/// A serving load armed the clock: from now on forwards are marked only inside a `live_scope(true)`.
pub(crate) fn serving_mode() { SERVING.with(|s| s.set(Some(false))); }
pub(crate) struct LiveScope;
impl Drop for LiveScope {
    fn drop(&mut self) { SERVING.with(|s| if s.get().is_some() { s.set(Some(false)); }); }
}
/// Mark the forwards issued while the returned scope lives (serving mode only; a screen is always live).
pub(crate) fn live_scope(on: bool) -> LiveScope {
    SERVING.with(|s| if s.get().is_some() { s.set(Some(on)); });
    LiveScope
}
/// The last execution's parts in microseconds, for a serving summary: [part 0, its collectives, wait before part 1,
/// part 1, its collectives, wait before part 2, part 2, its collectives]. A part is the time between its first and
/// its last mark on this rank; its collectives are the sums and gathers inside it (which include waiting for the
/// slowest rank); a wait is the model stream's idle time between two parts (the host had not launched the next).
/// None unless all three parts were written by one execution.
pub(crate) fn parts_us(marks: &[(usize, u8, u64)]) -> Option<[f64; 8]> {
    let mut out = [0f64; 8];
    let (mut starts, mut ends) = ([0u64; 3], [0u64; 3]);
    for part in 0..3 {
        let own: Vec<(u8, u64)> = marks.iter().filter(|(slot, _, _)| slot / PART == part).map(|(_, kind, t)| (*kind, *t)).collect();
        let &(first, t0) = own.first()?;
        if first != Site::Start as u8 { return None; }
        let live: Vec<(u8, u64)> = own.into_iter().filter(|(_, t)| *t >= t0).collect();
        let end = live.iter().rev().find(|(kind, _)| *kind == Site::End as u8)?.1;
        let (mut collectives, mut open) = (0u64, None);
        for &(kind, t) in &live {
            if kind == Site::SumBefore as u8 || kind == Site::GatherBefore as u8 { open = Some(t); }
            else if kind == Site::SumAfter as u8 || kind == Site::GatherAfter as u8 {
                if let Some(before) = open.take() { collectives += t.saturating_sub(before); }
            }
        }
        starts[part] = t0; ends[part] = end;
        out[part * 3] = (end - t0) as f64 / 1000.;
        out[part * 3 + 1] = collectives as f64 / 1000.;
    }
    if starts[1] < ends[0] || starts[2] < ends[1] { return None; }
    out[2] = (starts[1] - ends[0]) as f64 / 1000.;
    out[5] = (starts[2] - ends[1]) as f64 / 1000.;
    Some(out)
}
fn with(stream: &Stream, f: impl FnOnce(&mut Clock) -> Result<()>) -> Result<()> {
    if SERVING.with(|s| s.get()) == Some(false) { return Ok(()); }
    CLOCK.with(|c| {
        let mut c = c.borrow_mut();
        match c.as_mut() {
            Some(clock) if stream.raw() as usize == clock.parent => f(clock),
            _ => Ok(()),
        }
    })
}
/// Begin part `index` of a forward on the model stream: its first mark. No effect unless armed.
pub(crate) fn part(stream: &Stream, index: usize) -> Result<()> {
    crate::inject::begin(stream, index)?;
    with(stream, |clock| {
        ensure!(index < PARTS && !clock.open, "layer clock parts are sequential");
        clock.part = index;
        clock.next = index * PART;
        clock.limit = (index + 1) * PART;
        for kind in &mut clock.kinds[index * PART..(index + 1) * PART] { *kind = 0; }
        clock.open = true;
        clock.issue(stream, Site::Start)
    })
}
/// A mark inside an open part. No effect unless armed and inside a part.
pub(crate) fn mark(stream: &Stream, site: Site) -> Result<()> {
    crate::inject::at(stream, site as u8)?;
    with(stream, |clock| if clock.open { clock.issue(stream, site) } else { Ok(()) })
}
/// CANADA_SCRATCH_LAYER_CLOCK_FINE=1 (with the clock; screens only): marks inside the sublayers too, so one screen
/// gives an unprofiled timeline of a layer kernel group by kernel group.
pub(crate) fn fine_requested() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("CANADA_SCRATCH_LAYER_CLOCK_FINE").as_deref() == Ok("1"))
}
/// Fine mark codes: 16 + a sublayer's named point (mhc::StatsPoint: 17 QKV-A done, 18 gather done, 19 router
/// product, 20 routing, 21 experts done, 22 before the attention sum, 23 experts' tail, 24 routed experts done), and:
pub(crate) const FINE_POST: u8 = 32;         // a boundary's post-mix is done (after its statistics join)
pub(crate) const FINE_NORMALIZED: u8 = 33;   // a boundary's normalized (and quantized) input is ready
pub(crate) const FINE_QUERY_JOINED: u8 = 34; // the query projection's fork is joined: the attention may start
pub(crate) const FINE_ATTENDED: u8 = 35;     // attention and output rotation done
pub(crate) const FINE_WOA: u8 = 36;          // wo_a done
pub(crate) const FINE_WOB: u8 = 37;          // wo_b done (before the hint queue is joined)
pub(crate) const FINE_SIDE_PARTIALS: u8 = 38; // statistics stream: a quarter's slice partials are ready
pub(crate) const FINE_SIDE_STATS: u8 = 39;    // statistics stream: a boundary's coefficients are ready
/// A fine mark inside an open part: when the work issued so far on the model stream ended. No effect unless armed,
/// inside a part and requested.
pub(crate) fn fine(stream: &Stream, code: u8) -> Result<()> {
    crate::inject::at(stream, code)?;
    if !fine_requested() { return Ok(()); }
    with(stream, |clock| if clock.open { clock.issue_code(stream, code) } else { Ok(()) })
}
/// A fine mark for work on another stream of the open part (the statistics stream): when the work issued so far
/// on `other` ended. `model` finds the clock. Readers place such marks by time, not as a link of the chain.
pub(crate) fn fine_from(model: &Stream, other: &Stream, code: u8) -> Result<()> {
    if !fine_requested() { return Ok(()); }
    with(model, |clock| if clock.open { clock.issue_code(other, code) } else { Ok(()) })
}
/// End the open part: its last mark, and the side stream joins the model stream (a capture ends joined).
pub(crate) fn end(stream: &Stream) -> Result<()> {
    crate::inject::end(stream)?;
    with(stream, |clock| {
        if !clock.open { return Ok(()); }
        clock.issue(stream, Site::End)?;
        let done = &clock.done[clock.part];
        done.record(&clock.side)?;
        done.wait(stream)?;
        clock.open = false;
        Ok(())
    })
}
/// The marks of the last completed execution: (slot, site, device ns), in slot order. The caller has
/// synchronized the model stream. Slots a graph did not write keep an older time; the reader compares
/// against the part's own first mark.
pub(crate) fn read() -> Result<Option<Vec<(usize, u8, u64)>>> {
    CLOCK.with(|c| {
        let c = c.borrow();
        let Some(clock) = c.as_ref() else { return Ok(None) };
        let bytes = unsafe { clock.cuda.read_device(clock.slots.ptr(), PART * PARTS * 8) }?;
        Ok(Some(clock.kinds.iter().enumerate().filter(|(_, kind)| **kind != 0).map(|(slot, kind)| {
            (slot, *kind, u64::from_le_bytes(bytes[slot * 8..slot * 8 + 8].try_into().unwrap()))
        }).collect()))
    })
}
/// One part's marks as JSON: the sites in order and each mark's ns since the part's first mark (marks older
/// than that first mark were not written by this execution and are dropped with their site).
pub(crate) fn part_json(marks: &[(usize, u8, u64)], index: usize) -> serde_json::Value {
    let part: Vec<_> = marks.iter().filter(|(slot, _, _)| slot / PART == index).collect();
    let Some(&&(_, _, t0)) = part.first() else { return serde_json::Value::Null };
    let live: Vec<_> = part.iter().filter(|(_, _, t)| *t >= t0).collect();
    serde_json::json!({"t0_ns": t0, "sites": live.iter().map(|(_, kind, _)| *kind).collect::<Vec<_>>(),
        "ns": live.iter().map(|(_, _, t)| *t - t0).collect::<Vec<_>>()})
}
