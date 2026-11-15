//! OPTION, OFF BY DEFAULT: the model's own output head read from a 4-bit copy, with exact rescoring of the tokens
//! that matter (CANADA_SCRATCH_HEAD4_RESCORE=<cap>[,<window>]; everything below was measured at 512,24).
//! The owner's decision (Nov 14 2026): available as an option, off by default. It is part of no default
//! candidate, and standings and release targets are judged with it off. It is the only switch in this engine
//! that changes the model's output numbers on purpose; turn it on only knowing the two sections below.
//!
//! WHAT IT CHANGES. The head in question is the TARGET model's vocabulary head in a decode step of the six-row
//! verify owner: the head whose logits the output token is sampled from. (The drafter has a 4-bit head of its
//! own, CANADA_SCRATCH_DRAFT_HEAD_NVFP4; that one cannot change the output, because every draft is verified
//! against the target.) With the option on, such a step
//!   1. computes all 129,280 logits from the drafter's 4-bit copy of the head weights (93 MB a rank) instead of
//!      the canonical encoding (253 MB a rank);
//!   2. picks, per live row and per rank, the tokens within `window` (24) of that row's largest 4-bit logit, at
//!      most `cap` (512) a rank, so up to 2,048 a row; 1,765 a row on average (ca_head4_select);
//!   3. recomputes the picked tokens' logits as FP32 dot products with the canonical BF16 weights
//!      (ca_head4_rescore): the exact head's values up to the order of the floating-point sum.
//! Every other token keeps its 4-bit logit. The 16-row owner, the prefill owners and the first token of a turn
//! always use the exact head.
//!
//! NATURE OF THE DIFFERENCE. With the option on, the distribution a token is sampled from is not the exact
//! head's. The difference has two parts:
//!   - the tokens left in 4 bits carry a logit error of 0.24 rms (largest seen 2.6). They are the far tail: their
//!     total probability is 1.7e-5 on average (99th percentile 3.4e-4, largest seen 2.4e-3), and the most likely
//!     one of them has probability 7e-8 on average (largest seen 1.0e-5; exact rank about 400 on average). This
//!     part is about 70 % of the measured difference;
//!   - through the normalisation, the exactly computed tokens then all scale by one common factor, 1 +- 1.1e-6 on
//!     average (1.8e-4 at most).
//! The token the exact head ranks first was among the rescored ones in all 12,766 measured rows. The head is the
//! last operation of a pass, so nothing amplifies the error; its whole effect is that, rarely, another token is
//! drawn, after which the text continues as it would from any other draw.
//!
//! MEASURED SIZE (the option against the exact head on the same head input; build s1569; Nov 13-14 2026).
//!   Real generation, 5,110 rows: prose, code, agentic, JSON and prose after a 128K document
//!   (components/h4p/head4-scheme-v2.json in the measurement records):
//!     total variation a token  mean 2.5e-6 = one draw in ~396,000 differs; 99th percentile 4.8e-5; largest 3.7e-4
//!     KL(exact || option)      mean 6.5e-7 nats a token; 99th percentile 1.3e-5; largest 1.1e-4
//!     by the exact distribution's entropy (total variation): under 1 nat 3.3e-7, 1-2 nats 3.9e-6, 2-4 nats 2.7e-5
//!   A wider range of text, 7,656 rows over twelve prompt kinds from near-certain to very flat
//!   (components/h4w/head4-scheme.json): total variation mean 1.5e-6 = one draw in ~663,000; largest 3.4e-4.
//!   Whole answers at the same prompt and seed, 50 answers of 512 tokens (runs sbs0 / sbs1 / sbs4): exact against
//!   exact 50 of 50 identical; exact against the option 49 of 50 identical, the 50th parting at token ~407 into
//!   another ordinary continuation.
//!   Against DeepSeek's reference, KL(reference || engine) a token beside the reference's own summation-order
//!   yardstick over the same tokens: coding prompt, option on 0.0103 (yardstick 0.0117), off 0.0113 (0.0119);
//!   flat text, on 0.0261 (0.0243), off 0.0290 (0.0267); a whole candidate with the option on 0.0090 (0.0093).
//!   For scale: that yardstick is a total variation of ~0.040 a token, about 16,000 times the option's 2.5e-6.
//!   The 4-bit head WITHOUT rescoring is 8e-3 nats a token from the exact head: not usable on its own.
//!
//! WHAT IT BUYS. -0.94 / -0.76 / -0.60 / -0.41 / -0.44 ms a step at 2-6 verified rows in serving, +1.7...3.2 %
//! tokens/s by text type (the same texts with the option off and on).
//!
//! These figures hold for cap 512 and window 24 on this checkpoint. Any other setting, or another checkpoint,
//! has to be measured again before it is offered: export with CANADA_SCRATCH_CLAUDE_HEAD4_EXPORT=2 and compare
//! with tools/head4_scheme_analyze.py (measurement tools, kept with the measurement records).
use crate::cuda::{Buffer, Cuda, Module, Stream};
use anyhow::{Result, ensure};
use half::bf16;
use std::{cell::Cell, ffi::c_void, rc::Rc};

const N: usize = 32320;
const K: usize = 5120;
pub(crate) const ROWS: usize = 6;
pub(crate) struct Rescore {
    cuda: Cuda,
    module: Module,
    head: Rc<crate::nvfp4_head::Head>,
    canonical: u64,
    counts: Buffer<i32>,
    indices: Buffer<i32>,
    cap: usize,
    window: f32,
    on: Cell<bool>,
    /// Diagnostic export only: the scheme's logits beside the exact head's (allocated on first use).
    export_out: std::cell::RefCell<Option<Buffer<f32>>>,
}
impl Rescore {
    pub(crate) fn new(cuda: &Cuda, module: &Module, head: Rc<crate::nvfp4_head::Head>, canonical: u64,
        cap: usize, window: f32, on: bool) -> Result<Self> {
        ensure!(canonical != 0 && canonical % 256 == 0 && (64..=4096).contains(&cap) && cap % 8 == 0
            && (4.0..=64.0).contains(&window), "4-bit head rescoring geometry");
        module.function("ca_head4_select")?;
        module.function("ca_head4_rescore")?;
        let mut counts: Buffer<i32> = cuda.allocate(ROWS)?;
        counts.zero()?;
        let mut indices: Buffer<i32> = cuda.allocate(ROWS * cap)?;
        indices.zero()?;
        Ok(Self { cuda: cuda.clone(), module: module.clone(), head, canonical, counts, indices, cap, window, on: Cell::new(on),
            export_out: std::cell::RefCell::new(None) })
    }
    pub(crate) fn on(&self) -> bool { self.on.get() }
    pub(crate) fn set(&self, on: bool) { self.on.set(on); }
    pub(crate) fn settings(&self) -> (usize, f32) { (self.cap, self.window) }
    /// The first `live` rows of `normalized` ([6][5120]) to `logits` ([6][32320] FP32, this rank's vocabulary rows).
    pub(crate) fn run(&self, normalized: &Buffer<bf16>, logits: &Buffer<f32>, live: usize, stream: &Stream) -> Result<()> {
        ensure!(self.cuda.owns_buffer(normalized) && self.cuda.owns_buffer(logits) && self.cuda.owns_stream(stream)
            && normalized.len() == ROWS * K && logits.len() == ROWS * N && (1..=ROWS).contains(&live),
            "4-bit head rescoring operands");
        let (q, order) = self.head.project_rows(normalized.ptr(), live, stream)?;
        let (mut q, mut order, mut out, mut counts, mut indices) = (q, order, logits.ptr(), self.counts.ptr(), self.indices.ptr());
        let (mut window, mut cap) = (self.window, self.cap as i32);
        let mut select: [*mut c_void; 7] = [(&mut q as *mut u64).cast(), (&mut order as *mut u64).cast(), (&mut out as *mut u64).cast(),
            (&mut counts as *mut u64).cast(), (&mut indices as *mut u64).cast(), (&mut window as *mut f32).cast(), (&mut cap as *mut i32).cast()];
        unsafe { self.module.function("ca_head4_select")?.launch([live as u32, 1, 1], [1024, 1, 1], 0, stream, &mut select) }?;
        let (mut x, mut w, mut rows) = (normalized.ptr(), self.canonical, live as i32);
        let mut rescore: [*mut c_void; 7] = [(&mut x as *mut u64).cast(), (&mut w as *mut u64).cast(), (&mut counts as *mut u64).cast(),
            (&mut indices as *mut u64).cast(), (&mut out as *mut u64).cast(), (&mut rows as *mut i32).cast(), (&mut cap as *mut i32).cast()];
        unsafe { self.module.function("ca_head4_rescore")?.launch([((live * self.cap) as u32).div_ceil(8), 1, 1], [256, 1, 1], 0, stream, &mut rescore) }
    }
    /// Diagnostic only (CANADA_SCRATCH_CLAUDE_HEAD4_EXPORT=2): this rank's logits of the first `live` rows as this
    /// scheme computes them from `normalized`, into a buffer of its own (the pass's real logits are untouched), and
    /// the rows it rescored. For an offline comparison with the exact head's logits of the same step.
    pub(crate) fn export(&self, normalized: &Buffer<bf16>, live: usize, stream: &Stream) -> Result<(Vec<u8>, Vec<i32>)> {
        let mut slot = self.export_out.borrow_mut();
        if slot.is_none() { *slot = Some(self.cuda.allocate::<f32>(ROWS * N)?); }
        let out = slot.as_ref().unwrap();
        self.run(normalized, out, live, stream)?;
        stream.synchronize()?;
        Ok((unsafe { self.cuda.read_device(out.ptr(), live * N * 4) }?, self.last_counts(live)?))
    }
    /// The rows rescored for each of the first `live` rows by the last completed pass (the caller synchronized).
    pub(crate) fn last_counts(&self, live: usize) -> Result<Vec<i32>> { Ok(self.counts.download()?[..live.min(ROWS)].to_vec()) }
}
