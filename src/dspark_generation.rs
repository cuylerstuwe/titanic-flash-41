//! Native DSpark and transactional target verify. Greedy or T1/topP1 target
//! draws with deterministic or stochastic proposals; serving/session parking remain open. Tokens become
//! visible to the caller only after both accepted target and draft KV commit.
use crate::{
    checkpoint::Checkpoint,
    cuda::{Cuda, Stream},
    dspark::QueryNetwork,
    dspark_features::Features,
    history_lookup::HistoryLookup,
    layout::RankPlan,
    rejection::Rejection,
    roce::Roce,
    target::Target,
};
use anyhow::{Result, ensure};
use std::{path::Path, rc::Rc, time::Instant};

/// Batched step accounting per conversation count: rows verified (live) and executed (through the
/// last part's live rows), always; with CANADA_SCRATCH_BATCHED_PHASE_SYNC=1 (diagnostic) also the
/// phase times: batched steps then synchronize the stream at their phase boundaries (each draft,
/// the verification, the sessions' finishes, the owner's commit). Synchronizing removes host/GPU
/// overlap, so synced phases sum to slightly more than an unsynchronized step.
#[derive(Clone, Copy)]
pub(crate) struct BatchedPhases { steps: usize, synced: usize, live_rows: usize, executed_rows: usize,
    draft_ms: f64, draft_max_ms: f64, verify_ms: f64, finish_ms: f64, commit_ms: f64, groups: usize, lookups: usize }
impl BatchedPhases {
    const ZERO: Self = Self { steps: 0, synced: 0, live_rows: 0, executed_rows: 0, draft_ms: 0., draft_max_ms: 0.,
        verify_ms: 0., finish_ms: 0., commit_ms: 0., groups: 0, lookups: 0 };
}
static BATCHED_PHASES: std::sync::Mutex<[BatchedPhases; 5]> = std::sync::Mutex::new([BatchedPhases::ZERO; 5]);
/// 0: not read yet, 1: off, 2: on.
static PHASE_SYNC: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
fn phase_sync() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match PHASE_SYNC.load(Relaxed) {
        0 => {
            let on = std::env::var("CANADA_SCRATCH_BATCHED_PHASE_SYNC").as_deref() == Ok("1");
            PHASE_SYNC.store(if on { 2 } else { 1 }, Relaxed);
            on
        }
        state => state == 2,
    }
}
/// Screens: phase syncs for the following batched steps only.
pub(crate) fn set_batched_phase_sync_for_probe(on: bool) {
    PHASE_SYNC.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}
fn record_batched_phases(parts: usize, step: BatchedPhases) {
    let Ok(mut all) = BATCHED_PHASES.lock() else { return; };
    let a = &mut all[parts.min(4)];
    a.steps += step.steps; a.synced += step.synced; a.live_rows += step.live_rows; a.executed_rows += step.executed_rows;
    a.draft_ms += step.draft_ms; a.draft_max_ms += step.draft_max_ms; a.verify_ms += step.verify_ms;
    a.finish_ms += step.finish_ms; a.commit_ms += step.commit_ms; a.groups += step.groups; a.lookups += step.lookups;
}
/// The accounting per conversation count since the last call: rows a step, and phase means over
/// the synced steps.
pub(crate) fn take_batched_phases() -> serde_json::Value {
    let Ok(mut all) = BATCHED_PHASES.lock() else { return serde_json::Value::Null; };
    let stages = BATCHED_STAGES.lock().ok().map(|mut g| std::mem::take(&mut *g)).unwrap_or_default();
    let out: Vec<serde_json::Value> = (1..=4).filter(|&n| all[n].steps > 0).map(|n| {
        let p = all[n]; let f = p.steps as f64; let s = p.synced.max(1) as f64;
        let mut v = serde_json::json!({"conversations":n,"steps":p.steps,"live_rows_per_step":p.live_rows as f64 / f,
            "executed_rows_per_step":p.executed_rows as f64 / f,"groups_per_step":p.groups as f64 / f,
            "lookup_drafts_per_step":p.lookups as f64 / f,"synced_steps":p.synced});
        if let Some(st) = stages.get(n).filter(|m| !m.is_empty()) {
            v["host_stages_ms"] = serde_json::json!(st.iter().map(|(k, ms)| (k.to_string(), serde_json::json!(ms / f))).collect::<serde_json::Map<_, _>>());
        }
        if p.synced > 0 {
            v["draft_ms"] = serde_json::json!(p.draft_ms / s); v["draft_max_ms"] = serde_json::json!(p.draft_max_ms / s);
            v["verify_ms"] = serde_json::json!(p.verify_ms / s); v["finish_ms"] = serde_json::json!(p.finish_ms / s);
            v["commit_ms"] = serde_json::json!(p.commit_ms / s);
        }
        v
    }).collect();
    *all = [BatchedPhases::ZERO; 5];
    serde_json::Value::Array(out)
}

/// Batched verification host stages (Target::execute_rows marks) summed per conversation count.
static BATCHED_STAGES: std::sync::Mutex<Vec<std::collections::BTreeMap<&'static str, f64>>> = std::sync::Mutex::new(Vec::new());
fn record_batched_stages(parts: usize, stages: &[(&'static str, f64)]) {
    let Ok(mut all) = BATCHED_STAGES.lock() else { return; };
    if all.len() < 5 { all.resize(5, Default::default()); }
    for &(name, ms) in stages { *all[parts.min(4)].entry(name).or_default() += ms; }
}
/// CANADA_SCRATCH_HOST_STAGES=1: the single-conversation verifier's host stages (the pending verification's
/// `stages_ms`, read while it is still pending) summed since the last take, with the step count.
static SINGLE_STAGES: std::sync::Mutex<(usize, std::collections::BTreeMap<&'static str, f64>)> =
    std::sync::Mutex::new((0, std::collections::BTreeMap::new()));
fn record_single_stages(stages: &[(&'static str, f64)]) {
    let Ok(mut all) = SINGLE_STAGES.lock() else { return; };
    all.0 += 1;
    for &(name, ms) in stages { *all.1.entry(name).or_default() += ms; }
}
/// The sums since the last call (steps, ms by stage name).
pub(crate) fn take_single_stages() -> (usize, std::collections::BTreeMap<&'static str, f64>) {
    SINGLE_STAGES.lock().ok().map(|mut g| std::mem::take(&mut *g)).unwrap_or_default()
}
/// The draft gate of batched steps (two or more conversations; CANADA_SCRATCH_BATCHED_GATE): one
/// survival threshold for every count, or three for two, three and four conversations (e.g.
/// "0.45,off,0.3"); unset or "off" (for every count, or that count) keeps each conversation's own
/// serving gate. With a threshold, a
/// conversation's neural draft verifies draft j while the running product of its confidences
/// through j stays at or above it (always the first draft, at most its row limit). Only confidences
/// of included proposals decide, so it is a causal stopping rule: the sampled distribution is
/// unchanged; only the number of drafts a step verifies is.
static BATCHED_GATE: std::sync::Mutex<Option<[Option<f32>; 5]>> = std::sync::Mutex::new(None);
pub(crate) fn batched_gate_from_env() -> Result<[Option<f32>; 5]> {
    let v = match std::env::var("CANADA_SCRATCH_BATCHED_GATE") {
        Ok(v) => v,
        Err(std::env::VarError::NotPresent) => return Ok([None; 5]),
        Err(e) => return Err(e.into()),
    };
    let t = v.split(',').map(|s| match s.trim() {
        "off" => Ok(None),
        x => x.parse::<f32>().map(Some).map_err(anyhow::Error::from),
    }).collect::<Result<Vec<Option<f32>>>>()?;
    ensure!((t.len() == 1 || t.len() == 3) && t.iter().flatten().all(|x| (0.0..=1.0).contains(x)),
        "batched gate: one threshold, or three (two, three, four conversations), each in [0, 1] or off");
    let mut out = [None; 5];
    for (n, slot) in out.iter_mut().enumerate().skip(2) { *slot = if t.len() == 1 { t[0] } else { t[n - 2] }; }
    Ok(out)
}
fn batched_gate(parts: usize) -> Result<Option<f32>> {
    let mut g = BATCHED_GATE.lock().map_err(|_| anyhow::anyhow!("batched gate lock"))?;
    if g.is_none() { *g = Some(batched_gate_from_env()?); }
    Ok(g.and_then(|t| t[parts.min(4)]))
}
/// Diagnostic trace of batched steps (CANADA_SCRATCH_BATCHED_TRACE=1, or a screen's setter): per step
/// and conversation, its draft confidences, produced proposals, its own gate's rows, the rows
/// verified and the drafts accepted; for offline gate studies. None: off.
static BATCHED_TRACE: std::sync::Mutex<Option<Vec<serde_json::Value>>> = std::sync::Mutex::new(None);
static BATCHED_TRACE_READ: std::sync::Once = std::sync::Once::new();
fn trace_batched_step(entry: impl FnOnce() -> serde_json::Value) {
    BATCHED_TRACE_READ.call_once(|| {
        if std::env::var("CANADA_SCRATCH_BATCHED_TRACE").as_deref() == Ok("1") {
            if let Ok(mut g) = BATCHED_TRACE.lock() { *g = Some(Vec::new()); }
        }
    });
    if let Ok(mut g) = BATCHED_TRACE.lock() {
        if let Some(v) = g.as_mut() { if v.len() < 1 << 14 { v.push(entry()); } }
    }
}
/// Screens: start (true) or stop (false) the batched step trace.
pub(crate) fn set_batched_trace_for_probe(on: bool) {
    BATCHED_TRACE_READ.call_once(|| {});
    if let Ok(mut g) = BATCHED_TRACE.lock() { *g = on.then(Vec::new); }
}
/// The traced entries since the last call (the trace stays on).
pub(crate) fn take_batched_trace() -> Vec<serde_json::Value> {
    BATCHED_TRACE.lock().ok().and_then(|mut g| g.as_mut().map(std::mem::take)).unwrap_or_default()
}
/// Screens: the batched draft gate for the following steps (None: each conversation's own gate).
pub(crate) fn set_batched_gate_for_probe(threshold: Option<f32>) {
    if let Ok(mut g) = BATCHED_GATE.lock() { *g = Some([threshold; 5]); }
}

/// Validated before loading and applied when preparing each serving owner.
/// The explicit control preserves the established0.20recipe. Both admitted
/// cutoffs use the same causal stopping rule and exact rejection sampler.
/// Data collection (Nov 11, default off): one stderr line a neural step with the verified rows, the proposals the
/// target accepted, the tokens emitted and the drafter's five confidences. No token ids.
/// CANADA_SCRATCH_DRAFT_PRELAUNCH=1 (default off): an ordinary step that will be followed by a neural draft launches
/// that draft pass (body and head, a pure function of the drafter's state and the anchor) right after its feature
/// injection, before the host's bookkeeping and the return to the worker; the next step finds it running or done.
/// Proposals, confidences and every random draw stay in the next step. The next step uses it only if it starts at
/// the same frontier and anchor, the drafter still reports that pass as its last operation, and the collective
/// policies are the drafter's; otherwise it drafts again as before.
pub(crate) fn draft_prelaunch() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    STEP_ARM.with(|arm| arm.get()).map_or_else(
        || *FLAG.get_or_init(|| std::env::var("CANADA_SCRATCH_DRAFT_PRELAUNCH").as_deref() == Ok("1")), |arm| arm.0)
}
thread_local! {
    /// Decode profile probe only (CANADA_SCRATCH_CLAUDE_DECODE_ARMS): (draft pre-launch, sampler ahead) for the
    /// steps that follow, replacing the two environment flags, so both settings alternate within one load.
    static STEP_ARM: std::cell::Cell<Option<(bool, bool)>> = const { std::cell::Cell::new(None) };
}
pub(crate) fn set_step_arm_for_probe(arm: Option<(bool, bool)>) { STEP_ARM.with(|cell| cell.set(arm)); }
/// CANADA_SCRATCH_VERIFY_SAMPLE_AHEAD=1 (default off): an ordinary seeded step with the sharded block sampler
/// queues the sampler (its staged inputs and its captured graph) directly behind the verification forward and
/// waits once, at the sampler's readout; the forward's transport and fault checks run right after instead of
/// between the head and the sampler. Same kernels on the same inputs in the same stream order.
pub(crate) fn verify_sample_ahead() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    STEP_ARM.with(|arm| arm.get()).map_or_else(
        || *FLAG.get_or_init(|| std::env::var("CANADA_SCRATCH_VERIFY_SAMPLE_AHEAD").as_deref() == Ok("1")), |arm| arm.1)
}
pub(crate) fn step_confidence_log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CANADA_SCRATCH_STEP_CONFIDENCE_LOG").as_deref() == Ok("1"))
}
/// Diagnostic (CANADA_SCRATCH_DRAFT_SEED_LOG=1, default off): after every prompt, what the drafter's 128-row window is
/// seeded with: per row the L2 norm of its three target taps and the sum of each tap's first 64 values, and a digest
/// of all rows. A cold prefill and a cache restore of the same prompt should print the same rows up to rounding.
pub(crate) fn draft_seed_log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CANADA_SCRATCH_DRAFT_SEED_LOG").as_deref() == Ok("1"))
}
pub(crate) fn log_step_confidence(path: &str, step: &Step, emitted: usize) {
    if let Some(c) = step.draft_confidence.filter(|_| step_confidence_log()) {
        eprintln!("stepconf {path} rows={} accepted={} emitted={} source={} c={:.4},{:.4},{:.4},{:.4},{:.4}",
            step.verified_rows, step.matching_proposals, emitted, step.proposal_source, c[0], c[1], c[2], c[3], c[4]);
    }
}
/// CANADA_SCRATCH_CAUSAL_LADDER=t2,t4,t5 (default off; single-conversation steps): beside the three-row cutoff t3
/// (CANADA_SCRATCH_CAUSAL_THREE_THRESHOLD), thresholds for the other depths, each on the running product of the
/// drafter's confidences. Proposal 1 is always verified; proposal j >= 2 only while every product c1..ci, i <= j, is
/// at least t_i (first failure stops), so a step verifies 2..6 rows instead of 3 or 6. t2 <= t3 is required: the
/// drafter draws proposals 3..5 exactly when c1 c2 c3 >= t3, and that must imply c1 c2 >= t2.
pub(crate) fn serving_causal_ladder() -> Result<Option<[f32; 3]>> {
    match std::env::var("CANADA_SCRATCH_CAUSAL_LADDER") {
        Ok(spec) => {
            let t = spec.split(',').map(|v| Ok(v.trim().parse::<f32>()?)).collect::<Result<Vec<f32>>>()?;
            ensure!(t.len() == 3 && t.iter().all(|v| v.is_finite() && (0. ..=1.).contains(v))
                && t[0] <= serving_causal_three_threshold()?, "causal ladder must be t2,t4,t5 in [0,1] with t2 <= the three-row cutoff");
            Ok(Some([t[0], t[1], t[2]]))
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}
/// Rows of a step under the ladder: `three` is the drafter's own cutoff (proposals 3..5 exist iff c1 c2 c3 >= three),
/// `ladder` = [t2, t4, t5]. Each confidence precedes the proposal it admits and the first failed threshold stops, so
/// whether proposal j is verified depends on confidences 1..j only: later confidences and excluded proposals cannot
/// change the length. A NaN stops.
fn causal_ladder_rows(confidence: &[f32; 5], three: f32, ladder: [f32; 3], limit: usize) -> usize {
    let thresholds = [ladder[0], three, ladder[1], ladder[2]];
    let mut cumulative = confidence[0];
    let mut drafts = 1;
    for (probability, threshold) in confidence[1..].iter().zip(thresholds) {
        cumulative *= probability;
        if !(cumulative >= threshold) { break; }
        drafts += 1;
    }
    limit.min(1 + drafts)
}
pub(crate) fn serving_causal_three_threshold() -> Result<f32> {
    match std::env::var("CANADA_SCRATCH_CAUSAL_THREE_THRESHOLD").as_deref() {
        Ok("0.2" | "0.20") | Err(std::env::VarError::NotPresent) => Ok(0.20),
        Ok("0.25") => Ok(0.25),
        // Data collection only (Nov 11): 0 verifies all six rows in every step, so each step shows how many of the
        // five proposals the target accepts beside the drafter's confidences; 2 always verifies three rows.
        Ok("0") => Ok(0.),
        Ok("2") => Ok(2.),
        _ => anyhow::bail!("serving causal-three threshold must be0.20or0.25(0or2for data collection)"),
    }
}

#[path = "dspark_pair_probe.rs"]
mod pair_probe;
#[path = "matched_path_session.rs"]
mod matched_path_session;
#[path="query_reuse_generation_probe.rs"]
mod query_reuse_probe;
pub(crate) use query_reuse_probe::QueryReuseWorkspace;

/// A short block of n rows contains the anchor and proposals1..n-1.
/// Confidence n precedes proposal n and uses only that included prefix.
/// Later confidences and excluded proposal identities cannot affect stopping.
fn causal_prefix_rows(confidence: &[f32; 5], threshold: f32, short_rows: usize, limit: usize) -> usize {
    assert!((2..=5).contains(&short_rows));
    let cumulative: f32 = confidence[..short_rows].iter().product();
    if cumulative < threshold { limit.min(short_rows) } else { limit }
}

/// M2 uses only confidence0/1, before observing the excluded second proposal.
/// If that boundary passes, retain the established causal M3/M6 choice.
fn causal_two_three_rows(confidence:&[f32;5], two:f32, three:f32, limit:usize)->usize {
    if confidence[0]*confidence[1]<two {limit.min(2)}
    else {causal_prefix_rows(confidence,three,3,limit)}
}

/// Each confidence precedes the proposal it would admit. Stop at the first
/// failed threshold; excluded future confidences must not influence length.
/// Like the deployed recipe, always verify at least the first draft.
fn causal_full_prefix_rows(confidence: &[f32; 5], threshold: f32, limit: usize) -> usize {
    let mut cumulative = 1.;
    let mut drafts = 0;
    for &probability in confidence {
        cumulative *= probability;
        if cumulative < threshold { break; }
        drafts += 1;
    }
    limit.min(1 + drafts.max(1))
}

#[cfg(test)]
mod causal_depth_tests {
    use super::{causal_prefix_rows, causal_full_prefix_rows, causal_two_three_rows, causal_ladder_rows};
    #[test]
    fn ladder_length_uses_only_confidences_before_each_admitted_proposal() {
        let (three, ladder) = (0.2f32, [0.15f32, 0.2, 0.2]);
        let grid = [0.05f32, 0.3, 0.45, 0.6, 0.8, 1.0];
        for a in grid { for b in grid { for c in grid { for d in grid { for e in grid {
            let rows = causal_ladder_rows(&[a, b, c, d, e], three, ladder, 6);
            assert!((2..=6).contains(&rows));
            // Changing any confidence after the last one that admitted a proposal, or after the one that stopped,
            // cannot change the length: confidence j (1-based) decides proposal j, so rows r read confidences 1..r.
            let decided = rows.min(5);
            for tail in [0.0f32, 1.0, f32::NAN] {
                let mut other = [a, b, c, d, e];
                for x in other.iter_mut().skip(decided) { *x = tail; }
                assert_eq!(causal_ladder_rows(&other, three, ladder, 6), rows);
            }
            // The drafter's own stop (c1 c2 c3 < three) never admits a third proposal, and going on implies depth 2.
            if a * b * c < three { assert!(rows <= 3); } else { assert!(rows >= 4); }
            assert_eq!(causal_ladder_rows(&[a, b, c, d, e], three, ladder, 3), rows.min(3));
        }}}}}
        // With every extra threshold at zero the ladder is today's three-or-six rule.
        for c in [[0.9f32, 0.9, 0.9, 0.1, 0.1], [0.5, 0.5, 0.5, 0.9, 0.9], [1.0, 0.2, 1.0, 1.0, 1.0]] {
            assert_eq!(causal_ladder_rows(&c, 0.25, [0., 0., 0.], 6), causal_prefix_rows(&c, 0.25, 3, 6));
        }
        assert_eq!(causal_ladder_rows(&[0.1, 0.9, 0.9, 0.9, 0.9], 0.2, [0.15, 0.2, 0.2], 6), 2);
        assert_eq!(causal_ladder_rows(&[0.6, 0.5, 0.5, 0.9, 0.9], 0.2, [0.15, 0.2, 0.2], 6), 3);
        assert_eq!(causal_ladder_rows(&[0.9, 0.9, 0.9, 0.2, 0.9], 0.2, [0.15, 0.2, 0.2], 6), 4);
        assert_eq!(causal_ladder_rows(&[0.9, 0.9, 0.9, 0.9, 0.3], 0.2, [0.15, 0.2, 0.2], 6), 5);
        assert_eq!(causal_ladder_rows(&[0.9, 0.9, 0.9, 0.9, 0.9], 0.2, [0.15, 0.2, 0.2], 6), 6);
        assert_eq!(causal_ladder_rows(&[0.9, 0.9, 0.9, 0.9, 0.9], 0.2, [0.15, 0.2, 0.2], 1), 1);
    }
    #[test]
    fn two_three_stopping_boundary_ignores_excluded_draws_and_future_confidences() {
        for tail in [[0.,0.,0.],[1.,1.,1.],[f32::NAN,f32::INFINITY,-1.]] {
            for limit in 1..=6 {
                assert_eq!(causal_two_three_rows(&[0.4,0.5,tail[0],tail[1],tail[2]],0.25,0.25,limit),limit.min(2));
            }
        }
        for tail in [[0.,0.],[1.,1.],[f32::NAN,f32::INFINITY]] {
            for limit in 1..=6 {
                assert_eq!(causal_two_three_rows(&[0.8,0.5,0.5,tail[0],tail[1]],0.25,0.25,limit),limit.min(3));
                assert_eq!(causal_two_three_rows(&[0.8,0.8,0.8,tail[0],tail[1]],0.25,0.25,limit),limit);
            }
        }
    }
    #[test]
    fn full_prefix_stops_before_excluded_future_confidences() {
        for tail in [[0., 0., 0.], [1., 1., 1.], [f32::NAN, f32::INFINITY, -1.]] {
            let confidence = [0.5, 0.1, tail[0], tail[1], tail[2]];
            for limit in 1..=6 {
                assert_eq!(causal_full_prefix_rows(&confidence, 0.1, limit), limit.min(2));
            }
        }
        assert_eq!(causal_full_prefix_rows(&[0.6; 5], 0.1, 6), 5);
        assert_eq!(causal_full_prefix_rows(&[0.9; 5], 0.1, 6), 6);
    }
    #[test]
    fn depth_decision_does_not_use_excluded_future_confidences() {
        for first in [[0.5, 0.5, 0.5], [0.9, 0.9, 0.9]] {
            for tail in [[0., 0.], [1., 1.], [f32::NAN, f32::INFINITY]] {
                let confidence = [first[0], first[1], first[2], tail[0], tail[1]];
                for limit in 1..=6 {
                    assert_eq!(causal_prefix_rows(&confidence, 0.35, 3, limit),
                        if first[0] == 0.5 { limit.min(3) } else { limit });
                }
            }
        }
    }
    #[test]
    fn each_probe_depth_ignores_excluded_future_confidences() {
        for depth in 2..=5 {
            for prefix in [0.2, 0.99] {
                let baseline = [prefix; 5];
                let mut changed = baseline;
                changed[depth..].fill(f32::NAN);
                for limit in 1..=6 {
                    let expected = if prefix < 0.5 { limit.min(depth) } else { limit };
                    assert_eq!(causal_prefix_rows(&baseline, 0.20, depth, limit), expected);
                    assert_eq!(causal_prefix_rows(&changed, 0.20, depth, limit), expected);
                }
            }
        }
    }
}

pub(crate) struct GreedySession {
    image_owner: Rc<()>,
    cuda: Cuda,
    profile_verify: bool,
    profiled_verify: bool,
    draft: QueryNetwork,
    features: Features,
    prefill_features: Features,
    /// The 128-row drafter window built after an EXTEND that followed an encoder-image restore.
    extend_window: Option<crate::cuda::Buffer<half::bf16>>,
    poisoned: bool,
    target_seed: Option<u64>,
    rejection: Option<Rejection>,
    stochastic_prefix_schedule: bool,
    causal_three_threshold: Option<f32>,
    causal_two_threshold: Option<f32>,
    causal_full_threshold: Option<f32>,
    causal_short_rows: usize,
    /// Process policy (serving_causal_ladder): not part of session images; applies to a conversation's own steps.
    causal_ladder: Option<[f32; 3]>,
    /// draft_prelaunch: the (frontier, anchor) of a draft pass launched at the end of the previous step.
    prelaunched: Option<(u64, u32)>,
    stop_token: Option<u32>,
    history_lookup: Option<HistoryLookup>,
    /// config::batched_proposal_launch: this step's proposals, already drawn by the batched step's shared launch.
    batched_proposals: Option<[u32; 5]>,
    /// Guarded draft-head subset: the worker's verdict on this step's context (config::draft_head_subset_guard).
    head_subset_ok: bool,
    /// CANADA_SCRATCH_DRAFT_HEAD_SUBSET=1 (read once at construction).
    head_subset_guarded: bool,
    draft_warmed: bool,
    /// Process policy, independent of accepted causal state and cache images.
    skip_unused_draft: bool,
    launch_flags:crate::launch_flags::SessionFlags,
    block_verification: bool,
    local_history_sampling: bool,
    acceptance_temperature_frontiers: usize,
    /// Diagnostic sampler has separate scratch and never supplies emitted tokens.
    reasoning_temperature_probe: Option<Rejection>,
    root_feedback: Option<crate::root_feedback::RootFeedback>,
    encoder_cache_trace:Option<(std::path::PathBuf,u64,Option<u32>)>,
    buffered_draw_trace:Option<Rc<std::cell::RefCell<crate::draw_trace::DrawTrace>>>,
    /// Diagnostic only (logit export in a batched step): this conversation's rows of the rank's logit shard,
    /// captured by the step before staging; finish_phase commits the accepted ones.
    batched_export:Option<Vec<u8>>,
}
/// Static session allocations and loaded CUDA modules, with no accepted prompt
/// or sampled output. Serving can prepare these before advertising readiness;
/// only `attach` can produce a session that may execute model steps.
pub(crate) struct LongHistoryBuffers {
    features: Features,
    chunk: crate::cuda::Buffer<half::bf16>,
    wide: Option<crate::cuda::Buffer<half::bf16>>,
    wide_enabled: bool,
    rows: usize,
    minimum_match: usize,
}
impl LongHistoryBuffers {
    pub(crate) fn new(cuda:&Cuda,core:&[u8])->Result<Self> {
        Self::with_rows(cuda,core,16)
    }
    pub(crate) fn wide_conditioning_requested()->Result<bool> {
        Ok(match std::env::var("CANADA_SCRATCH_COPY_WIDE_CONDITIONING") {
            Ok(v)if v=="1"=>true,Ok(v)if v=="0"=>false,
            Err(std::env::VarError::NotPresent)=>false,
            _=>anyhow::bail!("copy wide conditioning must be0or1"),
        })
    }
    pub(crate) fn minimum_match_requested(rows:usize)->Result<usize> {
        let name=match rows {
            16=>"CANADA_SCRATCH_HISTORY16_MINIMUM_MATCH",
            32=>"CANADA_SCRATCH_HISTORY32_MINIMUM_MATCH",
            _=>anyhow::bail!("long history minimum requires16or32rows"),
        };
        let minimum=match std::env::var(name) {
            Ok(v)=>match v.as_str(){"32"=>32,"48"=>48,"64"=>64,
                _=>anyhow::bail!("invalid long history minimum: {name}")},
            Err(std::env::VarError::NotPresent)=>64,
            Err(error)=>return Err(error.into()),
        };
        // Narrow16/24 and32-row/32-key experiments stay finite-probe-only.
        ensure!((rows==16&&[32,64].contains(&minimum))||(rows==32&&[48,64].contains(&minimum)),
            "unqualified long history minimum for {rows}rows");
        Ok(minimum)
    }
    pub(crate) fn with_rows(cuda:&Cuda,core:&[u8],rows:usize)->Result<Self> {
        let wide=Self::wide_conditioning_requested()?&&rows==32;
        ensure!(std::env::var("CANADA_SCRATCH_LONG_HISTORY_GENERATION").as_deref()==Ok("1")
            && (rows==16 || (rows==32 && crate::config::history32_enabled()))
            // Include both the external padded input and the conditioner's
            // captured-input allocation in the startup construction forecast.
            && cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+(rows+6+if wide{256}else{0})*15360*2,
            "long history scratch admission");
        Ok(Self{features:Features::new(cuda,core,rows)?,chunk:cuda.allocate(6*15360)?,
            wide:if wide{Some(cuda.allocate(128*15360)?)}else{None},wide_enabled:wide,rows,
            minimum_match:Self::minimum_match_requested(rows)?})
    }
    pub(crate) fn set_wide_for_probe(&mut self,enabled:bool)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_COPY_CONDITIONING_GENERATION_SCREEN").as_deref()==Ok("1")
            &&self.rows==32&&self.wide.is_some(),"copy conditioning switch needs isolated owned storage");
        self.wide_enabled=enabled;Ok(())
    }
    pub(crate) fn set_minimum_for_probe(&mut self,minimum:usize)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_HISTORY_MATCH_GENERATION_SCREEN").as_deref()==Ok("1")
            &&[16,24,32,48,64].contains(&minimum),"history minimum switch requires the finite screen");
        self.minimum_match=minimum;Ok(())
    }
}
/// Discrete token metadata only; model probabilities remain on the GPU.
fn history_lookup_geometry()->Result<(usize,usize,usize)> {
    let slots=match std::env::var("CANADA_SCRATCH_HISTORY_TABLE_SLOTS") {
        Ok(v)=>match v.as_str() {"65536"=>1<<16,"262144"=>1<<18,"1048576"=>1<<20,
        _=>anyhow::bail!("unsupported history table size")},
        Err(std::env::VarError::NotPresent)=>1<<16,
        Err(e)=>return Err(e.into()),
    };
    let (maximum,tables)=match std::env::var("CANADA_SCRATCH_HISTORY_MAX_MATCH") {
        Ok(v)=>match v.as_str(){"16"=>(16,3),"32"=>(32,4),"64"=>(64,5),
        _=>anyhow::bail!("unsupported history maximum match")},
        Err(std::env::VarError::NotPresent)=>(16,3),Err(e)=>return Err(e.into()),
    };
    Ok((slots,maximum,tables))
}
fn history_strong_backup_enabled(maximum:usize)->Result<bool> {
    // s965/s968: preserve the primary tables and recover only displaced64-key
    // copies. Reserve the extra256KiB in active and saved session accounting.
    let enabled=match std::env::var("CANADA_SCRATCH_HISTORY_STRONG_BACKUP") {
        Ok(v)if v=="0"=>false,Ok(v)if v=="1"=>true,
        Err(std::env::VarError::NotPresent)=>maximum==64,
        _=>anyhow::bail!("history strong backup must be0or1"),
    };
    ensure!(!enabled||maximum==64,"strong backup requires64-key history");Ok(enabled)
}
pub(crate) fn history_lookup_reserved_bytes(context:usize)->Result<usize> {
    let(slots,maximum,tables)=history_lookup_geometry()?;
    Ok((context+tables*slots)*4+if history_strong_backup_enabled(maximum)?{
        crate::history_lookup::STRONG_BACKUP_BYTES}else{0})
}
pub(crate) fn new_history_lookup(cuda:&Cuda,context:usize,prefix:&[u32])->Result<HistoryLookup> {
    let(slots,maximum,_)=history_lookup_geometry()?;
    let expected=history_lookup_reserved_bytes(context)?;
    let memory=cuda.admission_memory()?;
    ensure!(memory.admitted_available>=crate::memory::floor_bytes()+expected,
        "history lookup allocation must preserve reserved headroom: {memory:?}, required_history_bytes={expected}");
    let lookup=if history_strong_backup_enabled(maximum)?{HistoryLookup::with_strong_backup(context,prefix,slots)?}
        else if maximum==16&&slots==1<<16 {HistoryLookup::new(context,prefix)?}
        else if maximum==16 {HistoryLookup::with_slots(context,prefix,slots)?}
        else {HistoryLookup::with_maximum(context,prefix,slots,maximum)?};
    ensure!(lookup.workspace_bytes()==expected,"history lookup reserved size differs");
    ensure!(cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+expected,
        "history lookup would consume reserved host/device headroom");Ok(lookup)
}

/// Small proposal-policy metadata accompanies encoder-only prompt images.
/// Tensor conditioning is reconstructed; warmup and proposal choices retain
/// the same snapshot semantics as complete session images.
#[derive(Clone)]
pub(crate) struct ProposalPolicyImage {
    owner:Rc<()>,target_seed:Option<u64>,stop_token:Option<u32>,draft_warmed:bool,
    causal_three_threshold:Option<f32>,causal_two_threshold:Option<f32>,causal_full_threshold:Option<f32>,causal_short_rows:usize,
    block_verification:bool,draft_temperature:f32,conditioning_split:bool,acceptance_temperature_frontiers:usize,
}
#[cfg(feature = "storage-fault-injection")]
impl ProposalPolicyImage {
    pub(crate) fn retention_metadata(&self)->serde_json::Value {
        serde_json::json!({"target_seed":self.target_seed,"stop_token":self.stop_token,
            "draft_warmed":self.draft_warmed,"causal_three_threshold":self.causal_three_threshold,"causal_two_threshold":self.causal_two_threshold,
            "causal_full_threshold":self.causal_full_threshold,"causal_short_rows":self.causal_short_rows,
            "block_verification":self.block_verification,"draft_temperature":self.draft_temperature,
            "conditioning_split":self.conditioning_split,"acceptance_temperature_frontiers":self.acceptance_temperature_frontiers})
    }
}
pub(crate) struct PreparedSession(GreedySession);
impl PreparedSession {
    /// The drafter's active 4-bit vocabulary head (for crate::head4_rescore).
    pub(crate) fn nvfp4_head(&self)->Option<std::rc::Rc<crate::nvfp4_head::Head>> {self.0.draft.nvfp4_head_for_export()}
    pub(crate) fn independent_for_probe(&self,target:&Target,core:&[u8],stream:&Stream,kernels:&Path)->Result<Self> {
        Self::new(&self.0.cuda,core,self.0.draft.independent_for_probe(target,core,stream,kernels)?)?
            .with_serving_policy(stream)
    }
    /// Batched serving: a drafter of identical shape that shares this one's image identity, so a
    /// session image saved by any of them restores into any other (region 0's drafter changes when
    /// its conversation moves to a serving region, and cached images must still restore there).
    /// Images carry only drafter-shaped device state and host settings.
    pub(crate) fn serving_fork(&self,target:&Target,core:&[u8],stream:&Stream,kernels:&Path)->Result<GreedySession> {
        let mut fork=self.independent_for_probe(target,core,stream,kernels)?.detached_for_probe()?;
        fork.image_owner=self.0.image_owner.clone();
        // The session identity above admits the image; the drafter's own state inside it needs the same.
        fork.draft.join_image_family(&self.0.draft)?;
        Ok(fork)
    }
    pub(crate) fn prepare_copy_conditioning(&mut self,buffers:&LongHistoryBuffers,comm:&Roce,stream:&Stream)->Result<bool> {
        let Some(input)=&buffers.wide else{return Ok(false)};
        ensure!(buffers.rows==32&&buffers.wide_enabled,"wide copy preparation storage");
        unsafe{self.0.cuda.zero_device(input.ptr(),input.bytes(),stream)?;}
        self.0.draft.prepare_copy_wide(input,comm,stream)?;Ok(true)
    }
    pub(crate) fn release_copy_conditioning(&mut self,stream:&Stream)->Result<()> {
        self.0.draft.release_prepared_copy_wide(stream)
    }
    /// The s280/s283 paired screens admit this conservative stopping policy
    /// for serving. Diagnostic constructors retain their explicit controls.
    /// Only confidences conditioned on included proposals affect the choice;
    /// the target/rejection distributions and their precision are unchanged.
    pub(crate) fn with_serving_policy(mut self, stream: &Stream) -> Result<Self> {
        let enabled = match std::env::var("CANADA_SCRATCH_CAUSAL_VERIFY") {
            Ok(value) => {
                ensure!(value == "0" || value == "1", "invalid serving causal verify policy");
                value == "1"
            }
            Err(std::env::VarError::NotPresent) => true,
            Err(error) => return Err(error.into()),
        };
        ensure!(self.0.rejection.is_some() && !self.0.stochastic_prefix_schedule,
            "serving causal policy requires the admitted categorical sampler");
        self.0.causal_three_threshold = enabled.then_some(serving_causal_three_threshold()?);
        self.0.causal_short_rows = 3;
        self.0.causal_ladder = if enabled { serving_causal_ladder()? } else { None };
        self.0.block_verification=match std::env::var("CANADA_SCRATCH_BLOCK_VERIFY") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid block verification policy");v=="1"},
            // Carry the block candidate forward: s381 long-context timing and
            // s410 same-path expectations support it. The GPU sampler-law
            // screen s326 covers causal lengths; token mode remains a control.
            Err(std::env::VarError::NotPresent)=>true,
            Err(e)=>return Err(e.into()),
        };
        let temperature=match std::env::var("CANADA_SCRATCH_DRAFT_TEMPERATURE") {
            Ok(v)=>match v.as_str() {"1"=>1.,"0.7"=>0.7,"1.2"=>1.2,_=>anyhow::bail!("unsupported serving draft temperature")},
            Err(std::env::VarError::NotPresent)=>1.,
            Err(e)=>return Err(e.into()),
        };
        self.0.draft.set_draft_temperature(temperature,stream)?;
        Ok(self)
    }
    pub(crate) fn new(cuda: &Cuda, core: &[u8], mut draft: QueryNetwork) -> Result<Self> {
        ensure!(draft.position()? == 0, "session preparation needs a fresh drafter");
        draft.use_local_base_logits()?;
        let stochastic_prefix_schedule = std::env::var("CANADA_SCRATCH_STOCHASTIC_PREFIX_SCHEDULE").as_deref() == Ok("1");
        ensure!(!stochastic_prefix_schedule || (std::env::var("CANADA_SCRATCH_STOCHASTIC_DRAFT").as_deref() == Ok("1")
            && std::env::var("CANADA_SCRATCH_COST_AWARE_VERIFY").as_deref() != Ok("1")),
            "stochastic prefix scheduling requires categorical proposals and the causal threshold rule");
        Ok(Self(GreedySession {
            image_owner: Rc::new(()), cuda: cuda.clone(),
            profile_verify: std::env::var("CANADA_SCRATCH_PROFILE_VERIFY").as_deref() == Ok("1"),
            profiled_verify: false, draft,
            features: Features::new(cuda, core, 6)?,
            prefill_features: Features::new(cuda, core, 128)?,
            extend_window: None,
            poisoned: false, target_seed: None, stochastic_prefix_schedule,
            causal_three_threshold: None, causal_two_threshold: None, causal_full_threshold: None,
            causal_short_rows: 3, causal_ladder: None, prelaunched: None, stop_token: None, history_lookup: None,
            batched_proposals: None,
            head_subset_ok: false, head_subset_guarded: crate::config::draft_head_subset()? == 1,
            draft_warmed: false,
            skip_unused_draft: match std::env::var("CANADA_SCRATCH_SKIP_UNUSED_DRAFT") {
                Ok(value) => { ensure!(value=="0"||value=="1","skip-unused-draft flag"); value=="1" },
                Err(std::env::VarError::NotPresent) => false,
                Err(error) => return Err(error.into()),
            },
            launch_flags:crate::launch_flags::SessionFlags::from_env(),
            block_verification: false,
            local_history_sampling:crate::sampler_transport::Policy::from_env()?.local_history,
            acceptance_temperature_frontiers: 0,
            encoder_cache_trace:None,
            buffered_draw_trace:None,
            batched_export:None,
            root_feedback: if std::env::var("CANADA_SCRATCH_ROOT_FEEDBACK_SCREEN").as_deref()==Ok("1") {
                Some(crate::root_feedback::RootFeedback::new(cuda,core)?)
            } else {None},
            reasoning_temperature_probe: if std::env::var("CANADA_SCRATCH_REASONING_DRAFT_SCREEN").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_POST_THINKING_DRAFT_SCREEN").as_deref()==Ok("1") {
                let mut probe=Rejection::new(cuda,core,6)?;
                probe.enable_history_expectation_probe()?;
                Some(probe)
            } else {None},
            rejection: if std::env::var("CANADA_SCRATCH_STOCHASTIC_DRAFT").as_deref() == Ok("1") {
                Some(Rejection::new(cuda, core, 6)?)
            } else { None },
        }))
    }
    /// Finite batched screen: a session whose drafter starts empty. Its prompt is
    /// absorbed append by append (GreedySession::absorb_batched_append_for_probe).
    pub(crate) fn detached_for_probe(self) -> Result<GreedySession> {
        ensure!(crate::config::batched_rows_admitted()
            && self.0.draft.position()? == 0, "detached screen session needs a fresh drafter");
        Ok(self.0)
    }
    pub(crate) fn attach(mut self, target: &Target, stream: &Stream) -> Result<GreedySession> {
        ensure!(target.physical_rows() == 128 && target.position() > 0
            && target.profile_is_current() && self.0.draft.position()? == 0,
            "DSpark attachment requires fresh draft state and a completed prefill");
        self.0.prefill_features.run(target, stream)?;
        let (values, live) = self.0.prefill_features.values()?;
        ensure!(live == target.position().min(128) as usize, "DSpark prefill features");
        self.0.draft.inject_prefill_tail(values, live, target.position(), stream)?;
        Ok(self.0)
    }
}
/// Complete model and sampler state at an accepted input frontier, plus its
/// one pending emitted anchor. HTTP request IDs/budgets and complete prompt
/// identities remain scheduler responsibilities; this is not a serving cache.
pub(crate) struct StateImage {
    owner: Rc<()>,
    target: crate::target::StateImage,
    draft: crate::dspark_kv::StateImage,
    anchor: u32,
    target_seed: Option<u64>,
    stop_token: Option<u32>,
    history_lookup: Option<HistoryLookup>,
    draft_warmed: bool,
    causal_three_threshold: Option<f32>,
    causal_two_threshold: Option<f32>,
    causal_full_threshold: Option<f32>,
    causal_short_rows: usize,
    block_verification: bool,
    draft_temperature: f32,
    conditioning_split: bool,
    acceptance_temperature_frontiers: usize,
}
impl StateImage {
    pub(crate) fn exchangeable(&self)->bool {self.target.exchangeable()}
    pub(crate) fn device_bytes(&self)->usize {self.target.bytes()+self.draft.bytes()}
    /// Disk tier: every copied image of the session (target, then drafter).
    pub(crate) fn images_mut(&mut self)->Vec<&mut crate::state_image::Image> {
        let mut out: Vec<&mut crate::state_image::Image> = self.target.image_mut().into_iter().collect();
        out.push(self.draft.image_mut());
        out
    }
    pub(crate) fn images(&self)->Vec<&crate::state_image::Image> {
        let mut out: Vec<&crate::state_image::Image> = self.target.image().into_iter().collect();
        out.push(self.draft.image());
        out
    }
    pub(crate) fn manifest(&self) -> serde_json::Value {
        serde_json::json!({"position":self.target.position(),"target":self.target.manifest(),
            "draft":self.draft.manifest(),"device_bytes":self.target.bytes()+self.draft.bytes(),
            "block_verification":self.block_verification,"draft_temperature":self.draft_temperature,"conditioning_split":self.conditioning_split,
            "history_lookup_bytes":self.history_lookup.as_ref().map_or(0, HistoryLookup::workspace_bytes)})
    }
    pub(crate) fn digests(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"target":self.target.digest()?,"draft":self.draft.digest()?,
            "causal_history":self.target.causal_history(),"anchor":self.anchor,
            "seed":self.target_seed,"stop_token":self.stop_token,"draft_warmed":self.draft_warmed,
            "causal_three_threshold":self.causal_three_threshold,"causal_two_threshold":self.causal_two_threshold,
            "causal_full_threshold":self.causal_full_threshold,
            "causal_short_rows":self.causal_short_rows,
            "block_verification":self.block_verification,"draft_temperature":self.draft_temperature,"conditioning_split":self.conditioning_split,
            "acceptance_temperature_frontiers":self.acceptance_temperature_frontiers,
            "lookup":self.history_lookup.as_ref().map(HistoryLookup::state_digest)}))
    }
}
pub(crate) struct ExchangePrepared {
    target:crate::state_image::exchange::Prepared,
    draft:crate::dspark_kv::StateImage,
}
/// One session of a batched decode step: its generation state, cache region, owner-side
/// history (Engram hashes and frontier), pending anchor and row limit.
pub(crate) struct BatchedSession<'a> {
    pub(crate) session: &'a mut GreedySession,
    pub(crate) base: crate::target::SessionBase,
    pub(crate) history: crate::hash_history::History,
    pub(crate) anchor: u32,
    pub(crate) limit: usize,
    /// Screen diagnostic (CANADA_SCRATCH_BATCH_SCREEN_TRACE=1): one record per step with
    /// hashes of the staged target logits and the drafter's proposal probabilities as the
    /// sampler is about to read them, then the step's outcome.
    pub(crate) trace: Option<Vec<serde_json::Value>>,
}
/// The state an ordinary or batched step carries from drafting to its finish.
pub(crate) struct Drafted {
    start: u64,
    begin: Instant,
    drafted: Instant,
    round_seed: Option<u64>,
    lookup: Option<crate::history_lookup::Proposal>,
    target_only: bool,
    context_tail: bool,
    draft_tokens: [u32; 5],
    draft_confidence: Option<[f32; 5]>,
    produced_proposals: usize,
    live: usize,
    verify: [u32; 6],
    acceptance_temperature_probe: Option<serde_json::Value>,
    local_history: bool,
    local_target: bool,
}
#[derive(serde::Serialize)]
pub(crate) struct Step {
    pub tokens: Vec<u32>,
    pub next_anchor: u32,
    pub draft_tokens: Vec<u32>,
    pub matching_proposals: usize,
    pub verified_rows: usize,
    pub draft_confidence: Option<[f32; 5]>,
    pub proposal_source: &'static str,
    pub lookup_match_tokens: Option<usize>,
    pub lookup_source_end: Option<usize>,
    pub profiled_verify: bool,
    pub profile_append_metrics: Option<serde_json::Value>,
    pub position: u64,
    pub phase_ms: [f64; 4],
    /// Verifier host time before awaiting the GPU, and its Engram part.
    pub verify_host_ms: [f64; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acceptance_expected_lengths: Option<Vec<[f64; 2]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acceptance_temperature_probe: Option<serde_json::Value>,
}
impl GreedySession {
    /// Explicit saved post-thinking frontier; this does not change serving policy.
    pub(crate) fn probe_post_thinking_temperatures(&mut self,target:&mut Target,anchor:u32,seed:u64,
        comm:&Roce,stream:&Stream)->Result<serde_json::Value> {
        ensure!(std::env::var("CANADA_SCRATCH_POST_THINKING_DRAFT_SCREEN").as_deref()==Ok("1")
            && !self.poisoned && target.profile_is_current() && self.draft.position()?==target.position()
            && self.history_lookup.is_none(),"post-thinking isolated frontier");
        let began=Instant::now();self.draft.run(anchor,comm,stream)?;
        let query_seconds=began.elapsed().as_secs_f64();
        // Query preparation legitimately writes provisional KV slots. Compare
        // the temperature counterfactuals against that same prepared query.
        let before=self.draft.save_state_image(4*1024*1024,comm,stream)?.digest()?;
        let mut result=self.probe_reasoning_temperatures(target,anchor,seed,comm,stream)?;
        let after=self.draft.save_state_image(4*1024*1024,comm,stream)?.digest()?;
        ensure!(before==after,"post-thinking probe changed drafter conditioning");
        result["draft_conditioning_exact"]=serde_json::json!(true);
        result["query_seconds"]=serde_json::json!(query_seconds);
        Ok(result)
    }
    /// Actual-q counterfactuals at eight accepted reasoning frontiers.
    /// The serving path is re-sampled after rollback at its original temperature.
    fn probe_reasoning_temperatures(&mut self,target:&mut Target,anchor:u32,seed:u64,
        comm:&Roce,stream:&Stream)->Result<serde_json::Value> {
        use serde_json::json;
        let post_thinking=std::env::var("CANADA_SCRATCH_POST_THINKING_DRAFT_SCREEN").as_deref()==Ok("1");
        let phase_repeat=std::env::var("CANADA_SCRATCH_PHASE_REPEAT_DRAFT_SCREEN").as_deref()==Ok("1");
        let strength_screen=std::env::var("CANADA_SCRATCH_MARKOV_STRENGTH_SCREEN").as_deref()==Ok("1");
        ensure!((std::env::var("CANADA_SCRATCH_REASONING_DRAFT_SCREEN").as_deref()==Ok("1") || post_thinking)
            && self.reasoning_temperature_probe.is_some() && self.block_verification
            && self.acceptance_temperature_frontiers<(if phase_repeat||strength_screen {24}else{8}) && self.rejection.is_some(),
            "reasoning temperature probe requires isolated current block sampler");
        let start=target.position();let ordinal=self.acceptance_temperature_frontiers;
        ensure!(self.draft.position()?==start && start>=if post_thinking{512}else{262117},"temperature probe frontier");
        let carry=target.carry_digests()?;let encoder=target.encoder_tail_digest()?;
        let history=self.history_lookup.as_ref().map(HistoryLookup::state_digest);
        let original=self.draft.draft_temperature();
        let narrow = std::env::var("CANADA_SCRATCH_REASONING_DRAFT_NARROW").as_deref()==Ok("1");
        let schedules=std::env::var("CANADA_SCRATCH_REASONING_DRAFT_SCHEDULES").as_deref()==Ok("1");
        ensure!(!post_thinking || (!narrow && !schedules),"post-thinking first screen uses ordinary temperatures");
        ensure!(!schedules || (!narrow && original==0.7 && self.draft.draft_schedule()==0
            && target.norm_quant_boundaries()==80 && target.query_norm_quant_layers()==40),
            "depth schedule probe must restore the ordinary0.7 policy");
        let mut temperatures:Vec<(usize,f32)>=if schedules {vec![(0,0.7),(1,0.7),(2,0.7)]}
            else {(if narrow {vec![0.4,0.5,0.6,0.7,0.85]}else{vec![0.7,1.,1.2]}).into_iter().map(|t|(0,t)).collect()};
        let mut temperatures:Vec<(usize,f32,f32)>=if strength_screen {
            ensure!(post_thinking&&!narrow&&!schedules&&original==0.7,"strength screen fixed sampler");
            vec![(0,0.7,1.),(0,0.7,0.75)]
        }else{temperatures.drain(..).map(|(s,t)|(s,t,1.)).collect()};
        let count=temperatures.len();if !strength_screen{temperatures.rotate_left(ordinal%count);}
        if ordinal%2==1 {temperatures.reverse();}
        let mut records=Vec::new();
        for (schedule,temperature,strength) in temperatures {
            if strength_screen{self.draft.set_markov_strength_for_probe(strength,stream)?;}
            if schedules {self.draft.set_draft_schedule(schedule,stream)?;}
            self.draft.set_draft_temperature(temperature,stream)?;
            let began=Instant::now();let tokens=self.draft.stochastic_proposals(seed,comm,stream)?;
            let markov_seconds=began.elapsed().as_secs_f64();
            let(confidence,_)=self.draft.confidence_schedule()?;
            let policies=[0.1,0.2,0.3].map(|threshold|json!({"threshold":threshold,
                "rows":causal_prefix_rows(&confidence,threshold,3,6)}));
            let mut ids=[anchor;6];ids[1..].copy_from_slice(&tokens);
            let began=Instant::now();target.begin_verification(&ids,comm,stream)?;stream.synchronize()?;
            let verify_seconds=began.elapsed().as_secs_f64();
            let(logits,rows)=target.pending_logits()?;ensure!(rows==6,"reasoning probe full block");
            let draw=self.reasoning_temperature_probe.as_mut().unwrap().sample_block(logits,
                self.draft.proposal_probabilities()?,&tokens,6,seed,start+1,stream)?;
            let first_overlap=draw.first_proposal_overlap.ok_or_else(||anyhow::anyhow!("missing marginal overlap"))?;
            let(local_overlaps,rb_sequential_length,overlap_product_score)=draw.local_proposal_fit
                .ok_or_else(||anyhow::anyhow!("missing local proposal fit"))?;
            let expected=draw.acceptance_expected_lengths.ok_or_else(||anyhow::anyhow!("missing GPU expectations"))?;
            target.finish_verification(0,comm,stream)?;
            ensure!(target.position()==start && self.draft.position()?==start
                && target.carry_digests()?==carry && target.encoder_tail_digest()?==encoder
                && self.history_lookup.as_ref().map(HistoryLookup::state_digest)==history,
                "reasoning counterfactual changed committed state");
            records.push(json!({"markov_strength":strength,"temperature":temperature,"schedule":schedule,"step_temperatures":self.draft.draft_temperatures(),"tokens":tokens,"confidence":confidence,
                "causal_policies":policies,"expected_lengths":expected,"first_proposal_overlap":first_overlap,"markov_seconds":markov_seconds,
                "local_proposal_overlaps":local_overlaps,"rb_sequential_length":rb_sequential_length,"overlap_product_score":overlap_product_score,
                "verify_seconds":verify_seconds,"rollback_exact":true}));
        }
        self.draft.set_draft_temperature(original,stream)?;
        if strength_screen{self.draft.set_markov_strength_for_probe(1.,stream)?;}
        if schedules {self.draft.set_draft_schedule(0,stream)?;}
        self.acceptance_temperature_frontiers+=1;
        Ok(json!({"kind":if strength_screen{"fresh_task_markov_strength"}else if std::env::var("CANADA_SCRATCH_FRESH_PHASE_DRAFT_SCREEN").as_deref()==Ok("1"){"fresh_task_matched_temperatures"}else if post_thinking{"post_thinking_matched_temperatures"}else if schedules{"reasoning_matched_schedules"}else{"reasoning_matched_temperatures"},"restored_schedule":self.draft.draft_schedule(),"frontier":start,"ordinal":ordinal,
            "seed":seed,"restored_temperature":original,"records":records,"serving_policy_changed":false}))
    }
    /// Reuse the current query at one accepted frontier, but resample and
    /// target-verify each actual temperature-dependent proposal distribution.
    /// Every counterfactual is rolled back; normal sampling then runs again.
    fn probe_acceptance_temperatures(
        &mut self, target: &mut Target, anchor: u32, seed: u64,
        comm: &Roce, stream: &Stream,
    ) -> Result<serde_json::Value> {
        use serde_json::json;
        ensure!(std::env::var("CANADA_SCRATCH_ACCEPTANCE_TEMPERATURE_SCREEN").as_deref()==Ok("1")
            && std::env::var("CANADA_SCRATCH_ACCEPTANCE_SHADOW").as_deref()==Ok("1")
            && self.acceptance_temperature_frontiers < 24 && self.rejection.is_some()
            && self.draft.draft_temperature()==1. && !self.block_verification,
            "matched temperature screen requires the isolated token/T1 control");
        let start=target.position();
        ensure!(self.draft.position()?==start,"temperature screen frontier");
        let check_state=self.acceptance_temperature_frontiers==0 || self.acceptance_temperature_frontiers==23;
        let before=if check_state {Some((target.carry_digests()?,target.encoder_tail_digest()?))}else{None};
        let candidate=match std::env::var("CANADA_SCRATCH_ACCEPTANCE_DRAFT_TEMPERATURE") {
            Ok(value)=>match value.as_str(){"0.7"=>0.7,"1.2"=>1.2,
                _=>anyhow::bail!("unsupported diagnostic draft temperature")},
            Err(std::env::VarError::NotPresent)=>0.7,
            Err(error)=>return Err(error.into()),
        };
        let order=if self.acceptance_temperature_frontiers%2==0 {[1.,candidate]}else{[candidate,1.]};
        let mut records=Vec::new();
        for temperature in order {
            self.draft.set_draft_temperature(temperature,stream)?;
            let began=Instant::now();
            let tokens=self.draft.stochastic_proposals(seed,comm,stream)?;
            let markov_seconds=began.elapsed().as_secs_f64();
            let (confidence,_)=self.draft.confidence_schedule()?;
            let causal_rows=if let Some(threshold)=self.causal_three_threshold {
                causal_prefix_rows(&confidence,threshold,self.causal_short_rows,6)
            }else{6};
            let mut ids=[anchor;6];ids[1..].copy_from_slice(&tokens);
            let began=Instant::now();
            target.begin_verification(&ids,comm,stream)?;
            stream.synchronize()?;
            let verify_seconds=began.elapsed().as_secs_f64();
            let (logits,rows)=target.pending_logits()?;
            ensure!(rows==6,"temperature probe full proposal block");
            let draw=self.rejection.as_mut().unwrap().sample(logits,
                self.draft.proposal_probabilities()?,&tokens,6,seed,start+1,stream)?;
            let expected=draw.acceptance_expected_lengths
                .ok_or_else(||anyhow::anyhow!("temperature probe requires GPU expectation diagnostic"))?;
            target.finish_verification(0,comm,stream)?;
            ensure!(target.position()==start && self.draft.position()?==start,
                "temperature probe advanced accepted state");
            if let Some((carry,encoder))=&before {
                ensure!(target.carry_digests()?==*carry && target.encoder_tail_digest()?==*encoder,
                    "temperature probe rollback changed carry or retained encoder");
            }
            records.push(json!({"temperature":temperature,"tokens":tokens,"confidence":confidence,"causal_rows":causal_rows,
                "expected_lengths":expected,"markov_seconds":markov_seconds,"verify_seconds":verify_seconds,
                "rollback_frontier_exact":true,"rollback_state_checked":check_state}));
        }
        self.draft.set_draft_temperature(1.,stream)?;
        self.acceptance_temperature_frontiers+=1;
        Ok(json!({"frontier":start,"ordinal":self.acceptance_temperature_frontiers,
            "seed":seed,"candidate_temperature":candidate,"records":records,"serving_policy_changed":false}))
    }
    pub(crate) fn draft_temperature(&self)->f32 {self.draft.draft_temperature()}
    /// Finite foreground intervention only. The harness restores the uniform
    /// policy before any request boundary; nonuniform schedules are not yet a
    /// saved-session policy and must not enter HTTP or background images.
    pub(crate) fn set_root_temperature_for_probe(&mut self, warmer:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_ROOT_TEMPERATURE_SCREEN").as_deref()==Ok("1")
            && !self.poisoned && self.target_seed.is_some() && self.rejection.is_some()
            && self.block_verification && self.draft.draft_temperature()==0.7,
            "root temperature requires isolated current actual-q serving screen");
        self.draft.set_draft_schedule(if warmer{3}else{0},stream)
    }
    pub(crate) fn set_root_feedback_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        self.set_root_temperature_for_probe(false,stream)?;
        self.root_feedback.as_mut().ok_or_else(||anyhow::anyhow!("root feedback not prepared"))?.reset(enabled,stream)
    }
    pub(crate) fn root_feedback_report(&self)->Result<serde_json::Value> {
        Ok(self.root_feedback.as_ref().ok_or_else(||anyhow::anyhow!("root feedback not prepared"))?.report())
    }
    pub(crate) fn set_hierarchical_for_probe(&mut self,active:bool,stream:&Stream)->Result<()> {
        ensure!(!self.poisoned&&self.target_seed.is_some()&&self.rejection.is_some()&&self.block_verification
            &&self.root_feedback.is_none()&&self.buffered_draw_trace.is_none()&&self.encoder_cache_trace.is_none(),
            "hierarchical draft requires isolated sampled block mode without full-q diagnostics");
        self.draft.set_hierarchical_for_probe(active,stream)
    }
    pub(crate) fn set_local_history_for_probe(&mut self,active:bool)->Result<()> {
        ensure!(!self.poisoned&&self.rejection.is_some()
            &&std::env::var("CANADA_SCRATCH_LOCAL_HISTORY_MODEL_SCREEN").as_deref()==Ok("1"),
            "finite local history sampler control");
        self.local_history_sampling=active;Ok(())
    }
    pub(crate) fn block_verification(&self)->bool {self.block_verification}
    /// Probe only (CANADA_SCRATCH_MIDPASS_DUMP): see Rejection::block_weights_for_probe.
    pub(crate) fn midpass_weights_for_probe(&self,drafts:usize)->Result<Option<Vec<f64>>> {
        self.rejection.as_ref().map(|r|r.block_weights_for_probe(drafts)).transpose()
    }
    pub(crate) fn conditioning_split(&self)->bool{self.draft.conditioning_split()}
    pub(crate) fn set_markov_cache_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()>{
        ensure!(!self.poisoned,"Markov cache switch on poisoned session");
        self.draft.set_markov_cache_for_probe(enabled,stream)
    }
    pub(crate) fn reset_markov_cache_counters_for_probe(&self,stream:&Stream)->Result<()>{
        ensure!(!self.poisoned,"Markov cache counters on poisoned session");
        self.draft.reset_markov_cache_counters_for_probe(stream)
    }
    pub(crate) fn markov_cache_counters_for_probe(&self,stream:&Stream)->Result<[u64;2]>{
        ensure!(!self.poisoned,"Markov cache counters on poisoned session");
        self.draft.markov_cache_counters_for_probe(stream)
    }
    pub(crate) fn set_conditioning_kv_reuse_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()>{
        ensure!(!self.poisoned,"conditioning KV reuse switch on poisoned session");
        self.draft.set_conditioning_kv_reuse_for_probe(enabled,stream)
    }
    pub(crate) fn set_conditioning_graph_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()>{
        ensure!(!self.poisoned,"conditioning graph switch on poisoned session");
        self.draft.set_conditioning_graph_for_probe(enabled,stream)
    }
    pub(crate) fn set_conditioning_split_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()>{
        ensure!(!self.poisoned && std::env::var("CANADA_SCRATCH_CONDITIONING_AB").as_deref()==Ok("1"),
            "conditioning override requires isolated comparison");
        self.draft.set_conditioning_split(enabled,stream)
    }
    pub(crate) fn proposal_policy_image(&self)->Result<ProposalPolicyImage> {
        ensure!(!self.poisoned,"proposal policy snapshot needs a healthy owner");
        Ok(ProposalPolicyImage{owner:self.image_owner.clone(),target_seed:self.target_seed,stop_token:self.stop_token,
            draft_warmed:self.draft_warmed,causal_three_threshold:self.causal_three_threshold,
            causal_two_threshold:self.causal_two_threshold,
            causal_full_threshold:self.causal_full_threshold,causal_short_rows:self.causal_short_rows,
            block_verification:self.block_verification,draft_temperature:self.draft.draft_temperature(),
            conditioning_split:self.draft.conditioning_split(),acceptance_temperature_frontiers:self.acceptance_temperature_frontiers})
    }
    pub(crate) fn restore_proposal_policy(&mut self,saved:&ProposalPolicyImage,stream:&Stream)->Result<()> {
        ensure!(!self.poisoned&&Rc::ptr_eq(&self.image_owner,&saved.owner),"proposal policy image owner");
        self.draft.set_draft_temperature(saved.draft_temperature,stream)?;
        self.draft.set_conditioning_split(saved.conditioning_split,stream)?;
        self.target_seed=saved.target_seed;self.stop_token=saved.stop_token;self.draft_warmed=saved.draft_warmed;
        self.causal_three_threshold=saved.causal_three_threshold;self.causal_full_threshold=saved.causal_full_threshold;
        self.causal_two_threshold=saved.causal_two_threshold;
        self.causal_short_rows=saved.causal_short_rows;self.block_verification=saved.block_verification;
        self.acceptance_temperature_frontiers=saved.acceptance_temperature_frontiers;Ok(())
    }
    pub(crate) fn proposal_policy_for_cache_probe(&self)->Result<serde_json::Value> {
        ensure!(!self.poisoned&&(std::env::var("CANADA_SCRATCH_ENCODER_CACHE_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_ACTIVE_ENCODER_CHECKPOINT_SCREEN").as_deref()==Ok("1")),
            "proposal policy report requires encoder cache screen");
        Ok(serde_json::json!({"warmed":self.draft_warmed,"graphs_ready":self.draft.sampled_graphs_ready(),
            "temperature":self.draft.draft_temperature(),"split":self.draft.conditioning_split(),
            "three_threshold":self.causal_three_threshold,"two_threshold":self.causal_two_threshold,"full_threshold":self.causal_full_threshold,
            "short_rows":self.causal_short_rows,"seed":self.target_seed,"stop":self.stop_token,
            "acceptance_ordinal":self.acceptance_temperature_frontiers,"block":self.block_verification}))
    }
    pub(crate) fn set_encoder_cache_trace_for_probe(&mut self,path:Option<std::path::PathBuf>,position:u64,expected_first:Option<u32>)->Result<()> {
        ensure!(!self.poisoned&&std::env::var("CANADA_SCRATCH_ENCODER_CACHE_TRACE").as_deref()==Ok("1"),
            "cache sample trace requires finite diagnostic admission");
        self.encoder_cache_trace=path.map(|p|(p,position,expected_first));Ok(())
    }
    pub(crate) fn set_buffered_draw_trace_for_probe(&mut self,trace:Rc<std::cell::RefCell<crate::draw_trace::DrawTrace>>)->Result<()> {
        ensure!(!self.poisoned&&std::env::var("CANADA_SCRATCH_READOUT_BUFFERED_TRACE").as_deref()==Ok("1"),
            "buffered draws require finite diagnostic admission");
        self.encoder_cache_trace=None;self.buffered_draw_trace=Some(trace);Ok(())
    }
    pub(crate) fn history_lookup_digest_for_probe(&self)->Result<Option<String>> {
        ensure!(!self.poisoned&&(std::env::var("CANADA_SCRATCH_ENCODER_CACHE_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_PREFILL_HEAD_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_FFN_HINT_MODEL_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_SHARDED_PREFIX_NATIVE_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_UNUSED_DRAFT_SCREEN").as_deref()==Ok("1")
            ||std::env::var_os("CANADA_SCRATCH_CACHE_SLOTS_SCREEN").is_some()
            ||std::env::var_os("CANADA_SCRATCH_FORWARD_CACHE_SCREEN").is_some()
            ||std::env::var("CANADA_SCRATCH_IDLE_ANCHOR_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_ENCODER_RECYCLE_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_STATE_COPY_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_ACTIVE_ENCODER_CHECKPOINT_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_INDEX_SCORE_MODEL_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_PEER_PREFILL_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_FFN16_GENERATION_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_FFN32_GENERATION_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_SAMPLER_GENERATION_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_FFN_SUM_SHARED_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_APPEND_LAST_GATHER_SCREEN").as_deref()==Ok("1")
            ||(crate::config::history32_probe_enabled()
                &&std::env::var("CANADA_SCRATCH_HISTORY32_GENERATION_SCREEN").as_deref()==Ok("1"))),
            "history digest requires finite encoder cache screen");
        Ok(self.history_lookup.as_ref().map(HistoryLookup::state_digest))
    }
    pub(crate) fn conditioning_state_digest_for_probe(&mut self,comm:&Roce,stream:&Stream)->Result<String>{
        ensure!(!self.poisoned&&(std::env::var("CANADA_SCRATCH_CONDITIONING_AB").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_PREFILL_HEAD_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_FFN_HINT_MODEL_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_FFN_SUM_SHARED_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_APPEND_LAST_GATHER_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_SHARDED_PREFIX_NATIVE_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_UNUSED_DRAFT_SCREEN").as_deref()==Ok("1")
            ||std::env::var_os("CANADA_SCRATCH_CACHE_SLOTS_SCREEN").is_some()
            ||std::env::var_os("CANADA_SCRATCH_FORWARD_CACHE_SCREEN").is_some()
            ||std::env::var("CANADA_SCRATCH_IDLE_ANCHOR_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_ENCODER_RECYCLE_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_STATE_COPY_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_ENCODER_CHECKPOINT_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_ENCODER_CACHE_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_ACTIVE_ENCODER_CHECKPOINT_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_SPARSE_GENERATION_AB").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_LONG_HISTORY_GENERATION").as_deref()==Ok("1")),
            "conditioning state digest requires isolated comparison");
        self.draft.save_state_image(4*1024*1024,comm,stream)?.digest()
    }
    pub(crate) fn accepted_conditioning_state_for_probe(&self,stream:&Stream)->Result<serde_json::Value>{
        ensure!(!self.poisoned&&(std::env::var("CANADA_SCRATCH_UNUSED_DRAFT_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_QUERY_REUSE_SCREEN").as_deref()==Ok("1")),
            "accepted conditioning diagnostic requires unused-draft screen");
        self.draft.accepted_repeat_digest(stream)
    }
    pub(crate) fn probe_copy_conditioning(&self,core:&[u8],kernels:&Path,comm:&Roce,stream:&Stream)
        ->Result<serde_json::Value>{
        let(features,live)=self.prefill_features.values()?;
        ensure!(!self.poisoned&&live>=64,"copy conditioning requires a full accepted prefill tail");
        self.draft.probe_copy_conditioning(core,kernels,features,comm,stream)
    }
    pub(crate) fn probe_copy_batched_kv(&mut self,comm:&Roce,stream:&Stream,
        control:&mut crate::control_channel::ControlChannel)->Result<serde_json::Value> {
        let(features,live)=self.prefill_features.values()?;
        ensure!(!self.poisoned&&live>=64,"copy KV batching requires accepted prefill features");
        self.draft.probe_copy_batched_kv(features,comm,stream,control)
    }
    /// Read-only branch-opportunity fixture. Each branch uses the existing
    /// serial verifier and rolls back completely; this is not a tree engine.
    /// Only rank3 persists tensor bytes, for subsequent GPU-only analysis.
    pub(crate) fn probe_tree_branches(
        &mut self, target: &mut Target, anchor: u32, seed: u64, rank: usize,
        output: &Path, comm: &Roce, stream: &Stream,
    ) -> Result<serde_json::Value> {
        use sha2::{Digest, Sha256};
        use serde_json::json;
        ensure!(std::env::var("CANADA_SCRATCH_TREE_SCREEN").as_deref()==Ok("1")
            && !self.poisoned && target.profile_is_current()
            && target.position()==self.draft.position()? && rank==comm.rank(),
            "tree fixture requires an isolated current session");
        self.poisoned=true;
        let result=(|| {
            let start=target.position();
            let carry=target.carry_digests()?;
            let encoder=target.encoder_tail_digest()?;
            let began=Instant::now();
            self.draft.run(anchor,comm,stream)?;
            let query_seconds=began.elapsed().as_secs_f64();
            let mut branches=Vec::new();
            for (branch,drafts) in [4usize,2].into_iter().enumerate() {
                let branch_seed=crate::categorical::block_round_seed(seed,2*start+branch as u64);
                let began=Instant::now();
                let tokens=self.draft.stochastic_proposals(branch_seed,comm,stream)?;
                let markov_seconds=began.elapsed().as_secs_f64();
                let q=self.draft.proposal_probabilities()?;
                let q_bytes=drafts*crate::sampling::VOCAB*8;
                ensure!(q_bytes<=q.bytes(),"tree proposal extent");
                let q_raw=unsafe {self.cuda.read_device(q.ptr(),q_bytes)}?;
                let mut ids=vec![anchor];ids.extend_from_slice(&tokens[..drafts]);
                let began=Instant::now();
                target.begin_verification(&ids,comm,stream)?;
                stream.synchronize()?;
                let verify_seconds=began.elapsed().as_secs_f64();
                let (logits,rows)=target.pending_logits()?;
                ensure!(rows==drafts+1,"tree fixture live rows");
                let logits_bytes=rows*crate::sampling::VOCAB*4;
                ensure!(logits_bytes<=logits.bytes(),"tree target extent");
                let logits_raw=unsafe {self.cuda.read_device(logits.ptr(),logits_bytes)}?;
                target.finish_verification(0,comm,stream)?;
                ensure!(target.position()==start && self.draft.position()?==start
                    && target.carry_digests()?==carry && target.encoder_tail_digest()?==encoder,
                    "tree branch rollback changed accepted state");
                let q_file=format!("branch{branch}-q.f64");
                let logits_file=format!("branch{branch}-logits.f32");
                if rank==3 {
                    std::fs::create_dir_all(output)?;
                    std::fs::write(output.join(&q_file),&q_raw)?;
                    std::fs::write(output.join(&logits_file),&logits_raw)?;
                }
                branches.push(json!({"branch":branch,"drafts":drafts,"tokens":tokens,
                    "seed":branch_seed,"q_file":q_file,"logits_file":logits_file,
                    "q_sha256":format!("{:x}",Sha256::digest(&q_raw)),
                    "logits_sha256":format!("{:x}",Sha256::digest(&logits_raw)),
                    "markov_seconds":markov_seconds,"verify_seconds":verify_seconds,
                    "rollback_exact":true}));
            }
            Ok(json!({"frontier":start,"anchor":anchor,"query_seconds":query_seconds,
                "branches":branches,"logical_node_budget":5,"serial_branch_evaluation":true}))
        })();
        if result.is_ok() {self.poisoned=false;}
        result
    }
    pub(crate) fn draft_head_kind(&self) -> &'static str {
        self.draft.head_kind()
    }
    pub(crate) fn fused_markov_head(&self)->bool {self.draft.fused_markov_head()}
    pub(crate) fn fused_confidence(&self)->bool {self.draft.fused_confidence()}
    pub(crate) fn set_fused_confidence_for_probe(&mut self,active:bool,stream:&Stream)->Result<()> {
        ensure!(!self.poisoned,"confidence switch on poisoned session");
        self.draft.set_fused_confidence_for_probe(active,stream)
    }
    pub(crate) fn set_fused_markov_head_for_probe(&mut self, active:bool, stream:&Stream)->Result<()> {
        ensure!(!self.poisoned && self.rejection.is_some(),"Markov screen requires valid sampled session");
        self.draft.set_fused_markov_head_for_probe(active,stream)
    }
    pub(crate) fn set_private_draft_head_for_probe(&mut self, active: bool, stream: &Stream) -> Result<()> {
        ensure!(!self.poisoned && self.rejection.is_some(), "head screen requires valid sampled session");
        self.draft.set_private_head_for_probe(active, stream)
    }
    pub(crate) fn set_draft_head_gemv_for_probe(&mut self, active: bool, stream: &Stream) -> Result<()> {
        ensure!(!self.poisoned && self.rejection.is_some(), "GEMV head screen requires valid sampled session");
        self.draft.set_head_gemv_for_probe(active, stream)
    }
    pub(crate) fn set_private_head_gemv_for_probe(&mut self,active:bool,stream:&Stream)->Result<()> {
        ensure!(!self.poisoned && self.rejection.is_some(),"private GEMV screen requires valid sampled session");
        self.draft.set_private_head_gemv_for_probe(active,stream)
    }
    pub(crate) fn set_sampler_readout_for_probe(&mut self,mode:u8,stream:&Stream)->Result<()> {
        ensure!(!self.poisoned&&self.rejection.is_some(),"sampled readout requires a valid session");
        self.rejection.as_mut().unwrap().set_readout_for_probe(mode,stream)
    }
    pub(crate) fn swap_sampler_for_graph_probe(&mut self,other:&mut Rejection,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_SAMPLER_GENERATION_SCREEN").as_deref()==Ok("1")
            &&!self.poisoned&&self.cuda.owns_stream(stream)&&self.rejection.is_some(),
            "sampler switch only at a completed draw");
        stream.synchronize()?;std::mem::swap(self.rejection.as_mut().unwrap(),other);Ok(())
    }
    pub(crate) fn set_block_progress_for_probe(&mut self,enabled:bool,validate:bool,stream:&Stream)->Result<()> {
        ensure!(!self.poisoned&&self.rejection.is_some()&&self.block_verification,"sampled block progress session");
        self.rejection.as_mut().unwrap().set_block_progress_for_probe(enabled,validate,stream)
    }
    pub(crate) fn observe_block_progress_for_probe(&mut self,rows:usize,seconds:f64,stream:&Stream)
        ->Result<(usize,serde_json::Value)> {
        ensure!(!self.poisoned&&self.rejection.is_some(),"block progress after completed commit only");
        self.rejection.as_mut().unwrap().observe_block_progress_for_probe(rows,seconds,stream)
    }
    pub(crate) fn set_block_rows_for_probe(&mut self,rows:Option<usize>)->Result<()> {
        ensure!(!self.poisoned&&self.block_verification&&self.target_seed.is_some()
            &&std::env::var("CANADA_SCRATCH_BLOCK_PROGRESS_SCREEN").as_deref()==Ok("1")
            &&rows.is_none_or(|r|[3,6].contains(&r)),"isolated next-round block rows");
        self.causal_three_threshold=Some(match rows{Some(3)=>1.,Some(6)=>0.,_=>0.2});
        self.causal_full_threshold=None;self.causal_short_rows=3;Ok(())
    }
    pub(crate) fn set_private_head_precision_for_probe(&mut self,kind:u8,stream:&Stream)->Result<()> {
        ensure!(!self.poisoned&&self.rejection.is_some(),"private precision requires valid actual-q sampled session");
        self.draft.set_private_head_precision_for_probe(kind,stream)
    }
    pub(crate) fn warm_draft_head_for_probe(&mut self, anchor:u32, comm:&Roce, stream:&Stream) -> Result<()> {
        ensure!(!self.poisoned && self.rejection.is_some(),"head warmup requires valid sampled session");
        self.draft.warm_head_for_probe(anchor,comm,stream)
    }
    pub(crate) fn starting_repeat_digest(&self,stream:&Stream)->Result<serde_json::Value>{
        use sha2::{Digest,Sha256};
        ensure!(!self.poisoned&&std::env::var("CANADA_SCRATCH_DRAFT_REPLAY_TRACE").as_deref()==Ok("1"),
            "starting session trace admission");
        stream.synchronize()?;
        // Begin can hand off from a wide profile and attach a not-yet-warmed
        // drafter. Only the owned feature rows and committed draft history are
        // meaningful here; target output and draft query scratch are not.
        let(features,live)=self.features.values()?;
        let raw=unsafe{self.cuda.read_device(features.ptr(),live*15360*2)}?;
        Ok(serde_json::json!({"features":{"live":live,"bytes":raw.len(),
            "sha256":format!("{:x}",Sha256::digest(&raw))},"draft_warmed":self.draft_warmed,
            "draft":self.draft.accepted_repeat_digest(stream)?}))
    }
    pub(crate) fn replay_digests(&self,target:&Target,root:Option<&Path>,stream:&Stream)->Result<serde_json::Value>{
        use sha2::{Digest,Sha256};
        ensure!(!self.poisoned && std::env::var("CANADA_SCRATCH_DRAFT_REPLAY_TRACE").as_deref()==Ok("1"),
            "session replay trace admission");
        stream.synchronize()?;
        if let Some(root)=root {std::fs::create_dir(root)?;}
        let (features,live)=self.features.values()?;
        let raw=unsafe{self.cuda.read_device(features.ptr(),live*15360*2)}?;
        if let Some(root)=root {std::fs::write(root.join("target-features.bin"),&raw)?;}
        Ok(serde_json::json!({"target_outputs":target.valid_output_digests()?,"target_carry":target.carry_digests()?,
            "encoder":target.encoder_tail_digest()?,"features":{"live":live,"bytes":raw.len(),"sha256":format!("{:x}",Sha256::digest(&raw))},
            "draft":self.draft.replay_digests(root,stream)?}))
    }
    /// Compare a history hit with DSpark at the same accepted frontier. Both
    /// target evaluations are discarded; only the ordinary step may publish.
    /// Conditional expectations describe these sampled paths, not permission
    /// to select a serving proposal using future target probabilities.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn probe_history_candidates(
        &mut self, target: &mut Target, anchor: u32, ordinal: usize, core: &[u8],
        comm: &Roce, stream: &Stream,
    ) -> Result<Option<serde_json::Value>> {
        use serde_json::json;
        use crate::sampling::VOCAB;
        ensure!(std::env::var("CANADA_SCRATCH_HISTORY_OPPORTUNITY").as_deref()==Ok("1")
            && std::env::var("CANADA_SCRATCH_ACCEPTANCE_SHADOW").as_deref()!=Ok("1")
            && ordinal<12 && !self.poisoned && self.block_verification
            && self.draft.draft_temperature()==1. && self.rejection.is_some()
            && target.profile_is_current() && target.position()==self.draft.position()?,
            "history opportunity requires isolated current block/T1 session");
        if !self.draft_warmed {return Ok(None);}
        let start=target.position();
        let Some(proposal)=self.history_lookup.as_ref()
            .map(|h|h.propose(start,anchor)).transpose()?.flatten() else {return Ok(None);};
        let seed=crate::categorical::block_round_seed(self.target_seed
            .ok_or_else(||anyhow::anyhow!("history probe needs seeded target"))?,start);
        self.poisoned=true;
        let result=(|| {
            let carry=target.carry_digests()?;
            let encoder=target.encoder_tail_digest()?;
            let history=self.history_lookup.as_ref().unwrap().state_digest();
            let saved_draft=self.draft.save_state_image(4*1024*1024,comm,stream)?;
            let draft_state=saved_draft.digest()?;
            let mut probe=Rejection::new(&self.cuda,core,6)?;
            probe.enable_history_expectation_probe()?;
            // Construct deterministic q on the device from raw constants.
            // No host model/probability arithmetic or dense CPU q is used.
            let point_mass=self.cuda.allocate::<f64>(5*VOCAB)?;
            let mut one=self.cuda.allocate::<f64>(1)?;one.upload(&[1.])?;
            unsafe {
                self.cuda.zero_device(point_mass.ptr(),point_mass.bytes(),stream)?;
                for (row,&id) in proposal.tokens.iter().enumerate() {
                    ensure!((id as usize)<VOCAB,"history proposal vocabulary");
                    self.cuda.copy_device(point_mass.ptr()+((row*VOCAB+id as usize)*8)as u64,
                        one.ptr(),8,stream)?;
                }
            }
            stream.synchronize()?;
            let order=if ordinal%2==0 {[false,true]}else{[true,false]};
            let mut records=Vec::new();
            for neural in order {
                let began=Instant::now();
                let (tokens,confidence,causal_rows)=if neural {
                    self.draft.run(anchor,comm,stream)?;
                    let tokens=self.draft.stochastic_proposals(seed,comm,stream)?;
                    let (confidence,_)=self.draft.confidence_schedule()?;
                    let rows=if let Some(threshold)=self.causal_three_threshold {
                        causal_prefix_rows(&confidence,threshold,self.causal_short_rows,6)
                    }else{6};
                    (tokens,Some(confidence),rows)
                }else{(proposal.tokens,None,6)};
                let proposal_seconds=began.elapsed().as_secs_f64();
                let mut ids=[anchor;6];ids[1..].copy_from_slice(&tokens);
                let began=Instant::now();
                target.begin_verification(&ids,comm,stream)?;stream.synchronize()?;
                let verify_seconds=began.elapsed().as_secs_f64();
                let (logits,rows)=target.pending_logits()?;
                ensure!(rows==6,"history probe full block");
                let q=if neural {self.draft.proposal_probabilities()?}else{&point_mass};
                let expected=probe.sample(logits,q,&tokens,6,seed,start+1,stream)?
                    .acceptance_expected_lengths
                    .ok_or_else(||anyhow::anyhow!("missing history GPU expectations"))?;
                target.finish_verification(0,comm,stream)?;
                // Query KV occupies private slots in the same allocation as
                // accepted history. Restore that scratch too before claiming
                // full snapshot identity or returning to the normal step.
                let draft_query_storage_changed=self.draft.save_state_image(4*1024*1024,comm,stream)?.digest()?!=draft_state;
                self.draft.restore_state_image(&saved_draft,comm,stream)?;
                let checks=json!({"target_frontier":target.position()==start,
                    "draft_frontier":self.draft.position()?==start,"carry":target.carry_digests()?==carry,
                    "encoder":target.encoder_tail_digest()?==encoder,
                    "history":self.history_lookup.as_ref().unwrap().state_digest()==history,
                    "draft_storage":self.draft.save_state_image(4*1024*1024,comm,stream)?.digest()?==draft_state});
                ensure!(checks.as_object().unwrap().values().all(|v|v==&json!(true)),
                    "history counterfactual state checks: {checks}");
                records.push(json!({"source":if neural{"dspark"}else{"history_point_mass"},
                    "tokens":tokens,"confidence":confidence,"causal_rows":causal_rows,
                    "expected_lengths":expected,"proposal_seconds":proposal_seconds,
                    "verify_seconds":verify_seconds,"rollback_exact":true,
                    "draft_query_storage_changed":draft_query_storage_changed,"state_checks":checks}));
            }
            Ok(Some(json!({"ordinal":ordinal,"frontier":start,"anchor":anchor,"seed":seed,
                "lookup_match_tokens":proposal.matched,"lookup_source_end":proposal.source_end,
                "records":records,"serving_policy_changed":false})))
        })();
        if result.is_ok(){self.poisoned=false;}
        result
    }
    pub(crate) fn history_lookup_metadata(&self) -> Option<(u64, usize)> {
        self.history_lookup
            .as_ref()
            .map(|history| (history.position(), history.workspace_bytes()))
    }
    pub(crate) fn configure_history_tables_for_probe(&mut self, slots: usize) -> Result<()> {
        ensure!(!self.poisoned
            && std::env::var("CANADA_SCRATCH_HISTORY_POLICY_SCREEN").as_deref()==Ok("1")
            && std::env::var("CANADA_SCRATCH_HISTORY_TABLE_SCREEN").as_deref()==Ok("1"),
            "history table override requires isolated screen");
        self.history_lookup.as_mut().ok_or_else(||anyhow::anyhow!("history table probe needs lookup"))?
            .resize_tables_for_probe(slots)
    }
    pub(crate) fn configure_history_associativity_for_probe(&mut self,ways:usize)->Result<()> {
        ensure!(!self.poisoned&&std::env::var("CANADA_SCRATCH_HISTORY_ASSOCIATIVITY_SCREEN").as_deref()==Ok("1"),
            "history associativity requires the finite foreground screen");
        ensure!(self.cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+32*1024usize.pow(2),
            "history index rebuild would consume reserved headroom");
        self.history_lookup.as_mut().ok_or_else(||anyhow::anyhow!("history lookup absent"))?
            .resize_associativity_for_probe(ways)
    }
    /// Device and additional host bytes for an optional compact snapshot.
    /// The caller must obtain all-rank admission before any allocation.
    pub(crate) fn compact_state_image_bytes(&self,target:&Target)->Result<(usize,usize)> {
        ensure!(!self.poisoned && target.profile_is_current() && target.position()==self.draft.position()?,
            "compact image size needs a committed session");
        Ok((target.compact_state_image_bytes()?.checked_add(self.draft.state_image_bytes()?)
            .ok_or_else(||anyhow::anyhow!("session image byte overflow"))?,
            self.history_lookup.as_ref().map_or(0,HistoryLookup::reserved_workspace_bytes)))
    }
    pub(crate) fn exchange_state_image_bytes(&self,target:&Target)->Result<(usize,usize)> {
        let(_,host)=self.compact_state_image_bytes(target)?;
        Ok((target.exchange_state_image_bytes()?+self.draft.state_image_bytes()?,host))
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_prefill(
        target: &Target,
        cuda: &Cuda,
        core: &[u8],
        stream: &Stream,
        cp: &Checkpoint,
        plan: &RankPlan,
        staged: &Path,
        kernels: &Path,
    ) -> Result<Self> {
        ensure!(
            target.physical_rows() == 128 && target.position() > 0 && target.profile_is_current(),
            "DSpark initial integration requires a completed prefill"
        );
        let draft =
            QueryNetwork::for_target(target, cuda, core, stream, cp, plan, staged, kernels)?;
        Self::from_loaded_draft(target, cuda, core, stream, draft)
    }
    /// Attach prompt-dependent state to a draft network whose canonical
    /// weights were loaded before the serving worker advertised readiness.
    /// Loading alone cannot initialize accepted features or publish tokens.
    pub(crate) fn from_loaded_draft(
        target: &Target,
        cuda: &Cuda,
        core: &[u8],
        stream: &Stream,
        draft: QueryNetwork,
    ) -> Result<Self> {
        PreparedSession::new(cuda, core, draft)?.attach(target, stream)
    }
    pub(crate) fn configure_sampling(&mut self, seed: u64, stop_token: Option<u32>) -> Result<()> {
        ensure!(!self.poisoned && stop_token.is_none_or(|t| t < 129280), "session sampling configuration");
        if self.target_seed.is_none() { self.draft_warmed = false; }
        self.target_seed = Some(seed);
        self.stop_token = stop_token;
        Ok(())
    }
    pub(crate) fn configure_greedy(&mut self, stop_token: Option<u32>) -> Result<()> {
        ensure!(!self.poisoned && stop_token.is_none_or(|t| t < 129280), "session greedy configuration");
        if self.target_seed.is_some() { self.draft_warmed = false; }
        self.target_seed = None;
        self.stop_token = stop_token;
        Ok(())
    }
    pub(crate) fn configure_algorithm_for_probe(&mut self, block: bool, temperature: f32, stream: &Stream) -> Result<()> {
        let long_policy=(std::env::var("CANADA_SCRATCH_LONG_POLICY_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_HISTORY_OPPORTUNITY").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_HISTORY_POLICY_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_CURRENT_CONTINUATION").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_DRAFT_DEPTH_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_DRAFT_TEMPERATURE_SCREEN").as_deref()==Ok("1"))
            && std::env::var("CANADA_SCRATCH_LONG_CONTINUATION").as_deref()==Ok("1");
        ensure!(!self.poisoned && self.rejection.is_some() && self.target_seed.is_some()
            && (long_policy || std::env::var("CANADA_SCRATCH_LONG_HISTORY_GENERATION").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_POST_THINKING_DRAFT_SCREEN").as_deref()==Ok("1")
                || (std::env::var("CANADA_SCRATCH_ALGORITHM_SCREEN").as_deref()==Ok("1")
            && (std::env::var("CANADA_SCRATCH_CAUSAL_VERIFY_SCREEN").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_POST_HTTP_CAUSAL").as_deref()==Ok("1")))),
            "algorithm override requires the isolated sampled screen");
        self.draft.set_draft_temperature(temperature,stream)?;
        self.block_verification=block;
        Ok(())
    }
    pub(crate) fn causal_verify_threshold(&self) -> Option<f32> {
        self.causal_three_threshold
    }
    pub(crate) fn direct_markov_gather(&self)->bool { self.draft.direct_markov_gather() }
    pub(crate) fn cached_probability_factors(&self)->bool {
        self.draft.cached_probability_factors()
            && self.rejection.as_ref().is_some_and(|r|r.cached_probability_factors())
    }
    #[cfg(feature = "serving")]
    pub(crate) fn arm_verify_profile(&mut self) -> Result<()> {
        ensure!(!self.poisoned && !self.profile_verify && !self.profiled_verify
            && self.target_seed.is_some() && self.rejection.is_some(),
            "verification profile needs an unprofiled sampled session");
        self.profile_verify = true;
        Ok(())
    }
    /// Isolated experiment: an additional causal stop before the second draft.
    /// Keep the existing three/five-proposal generator and actual-q verifier.
    /// Unused later draws are discarded; no future confidence selects M2.
    pub(crate) fn set_causal_two_three_for_probe(&mut self, threshold: Option<f32>) -> Result<()> {
        ensure!(!self.poisoned && self.rejection.is_some() && self.target_seed.is_some()
            && self.block_verification && !self.stochastic_prefix_schedule
            && self.causal_three_threshold==Some(0.25) && self.causal_short_rows==3
            && self.causal_full_threshold.is_none()
            && threshold.is_none_or(|v|v==0.25)
            && std::env::var("CANADA_SCRATCH_CAUSAL_TWO_THREE_SCREEN").as_deref()==Ok("1"),
            "two/three-row policy requires isolated canonical sampled screen");
        self.causal_two_threshold=threshold;
        Ok(())
    }
    pub(crate) fn set_causal_three_threshold_for_probe(&mut self, threshold: Option<f32>) -> Result<()> {
        self.set_causal_depth_for_probe(3, threshold)
    }
    pub(crate) fn set_causal_depth_for_probe(&mut self, rows: usize, threshold: Option<f32>) -> Result<()> {
        ensure!(!self.poisoned && self.rejection.is_some() && !self.stochastic_prefix_schedule
            && (std::env::var("CANADA_SCRATCH_CAUSAL_VERIFY_SCREEN").as_deref() == Ok("1")
                || std::env::var("CANADA_SCRATCH_POST_HTTP_CAUSAL").as_deref() == Ok("1")
                || std::env::var("CANADA_SCRATCH_SPARSE_GENERATION_AB").as_deref() == Ok("1")
                || std::env::var("CANADA_SCRATCH_LONG_CONTINUATION").as_deref() == Ok("1")
                || std::env::var("CANADA_SCRATCH_FRESH_PHASE_DRAFT_SCREEN").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_NATIVE_PREFIX_STOP_SCREEN").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_CONDITIONING_GRAPH_NATIVE_SCREEN").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_LONG_HISTORY_GENERATION").as_deref()==Ok("1"))
            && threshold.is_none_or(|v| v.is_finite() && (0. ..=1.).contains(&v))
            && (2..=5).contains(&rows)
            && (rows==3 || std::env::var("CANADA_SCRATCH_CAUSAL_DEPTH_SCREEN").as_deref()==Ok("1")),
            "causal depth policy requires the isolated stochastic screen");
        self.causal_three_threshold = threshold;
        self.causal_two_threshold = None;
        self.causal_full_threshold = None;
        self.causal_short_rows = rows;
        Ok(())
    }
    pub(crate) fn set_full_causal_prefix_for_probe(&mut self, threshold: f32) -> Result<()> {
        ensure!(!self.poisoned && self.rejection.is_some() && !self.stochastic_prefix_schedule
            && std::env::var("CANADA_SCRATCH_LONG_FULL_CAP_SCREEN").as_deref() == Ok("1")
            && threshold.is_finite() && (0. ..=1.).contains(&threshold),
            "full causal prefix policy requires the isolated screen");
        self.causal_full_threshold = Some(threshold);
        Ok(())
    }
    pub(crate) fn save_state_image(&mut self, target: &Target, anchor: u32, budget: usize,
        comm: &Roce, stream: &Stream) -> Result<StateImage> {
        self.save_image(target,anchor,budget,false,comm,stream)
    }
    pub(crate) fn save_exchange_state_image(&mut self,target:&Target,anchor:u32,budget:usize,
        comm:&Roce,stream:&Stream)->Result<StateImage>{self.save_image(target,anchor,budget,true,comm,stream)}
    fn save_image(&mut self,target:&Target,anchor:u32,budget:usize,exchange:bool,
        comm:&Roce,stream:&Stream)->Result<StateImage>{
        ensure!(!self.poisoned && target.profile_is_current() && anchor < 129280
            && (1..=2 * 1024usize.pow(3)).contains(&budget)
            && target.position() == self.draft.position()?
            && self.history_lookup.as_ref().is_none_or(|h| h.position() == target.position()),
            "session image requires committed target/draft/history frontiers");
        let cpu_bytes = self.history_lookup.as_ref().map_or(0, HistoryLookup::reserved_workspace_bytes);
        ensure!(self.cuda.admission_memory()?.admitted_available >= crate::memory::floor_bytes() + budget + cpu_bytes,
            crate::state_image::CapacityRefused);
        // Snapshot ownership includes the entire session, so altered seeds,
        // lookup-table collision history and warmup policy are restored too.
        let saved_target = if exchange{target.save_exchange_state_image(budget,comm,stream)?}
            else{target.save_state_image(budget, comm, stream)?};
        let remaining = budget.checked_sub(saved_target.bytes())
            .ok_or_else(|| anyhow::anyhow!("session image budget overflow"))?;
        self.poisoned = true;
        let draft = match self.draft.save_state_image(remaining, comm, stream) {
            Ok(draft)=>draft,
            Err(error) if error.downcast_ref::<crate::state_image::CapacityRefused>().is_some()=> {
                // The target snapshot only read live state; the drafter
                // refuses allocation before taking its writer lease. Drop
                // the optional target copy and retain the healthy session.
                self.poisoned=false;
                return Err(error);
            }
            Err(error)=>return Err(error),
        };
        let saved = StateImage { owner: self.image_owner.clone(), target: saved_target, draft,
            anchor, target_seed: self.target_seed, stop_token: self.stop_token,
            history_lookup: self.history_lookup.as_ref().map(HistoryLookup::clone_reserved), draft_warmed: self.draft_warmed,
            causal_three_threshold: self.causal_three_threshold, causal_short_rows: self.causal_short_rows,
            causal_two_threshold: self.causal_two_threshold,
            causal_full_threshold: self.causal_full_threshold,
            block_verification:self.block_verification,draft_temperature:self.draft.draft_temperature(),
            conditioning_split:self.draft.conditioning_split(),
            acceptance_temperature_frontiers:self.acceptance_temperature_frontiers };
        self.poisoned = false;
        Ok(saved)
    }
    /// Additional peak bytes and final inactive-device charge. History lookup
    /// ownership is moved, never cloned, during the exchange itself.
    pub(crate) fn exchange_growth(&self,target:&Target,saved:&StateImage,budget:usize)->Result<(usize,usize)>{
        ensure!(!self.poisoned&&Rc::ptr_eq(&self.image_owner,&saved.owner)
            &&target.profile_is_current()&&target.position()==self.draft.position()?,"session exchange frontier");
        let draft=self.draft.state_image_bytes()?;
        let(extra,capacity)=target.exchange_growth(&saved.target,budget.checked_sub(draft).ok_or(crate::state_image::CapacityRefused)?)?;
        Ok((extra+draft,capacity+draft))
    }
    pub(crate) fn prepare_exchange(&mut self,target:&Target,saved:&StateImage,budget:usize,comm:&Roce,stream:&Stream)
        ->Result<Option<ExchangePrepared>> {
        let(extra,_)=self.exchange_growth(target,saved,budget)?;
        if self.cuda.admission_memory()?.admitted_available<crate::memory::floor_bytes()+extra{return Ok(None);}
        let draft_bytes=self.draft.state_image_bytes()?;
        let Some(target)=target.prepare_exchange(&saved.target,budget-draft_bytes,comm,stream)?else{return Ok(None);};
        let draft=match self.draft.save_state_image(draft_bytes,comm,stream){
            Ok(image)=>image,Err(e)if e.downcast_ref::<crate::state_image::CapacityRefused>().is_some()=>return Ok(None),Err(e)=>return Err(e),
        };
        Ok(Some(ExchangePrepared{target,draft}))
    }
    pub(crate) fn exchange_state_image(&mut self,target:&mut Target,saved:&mut StateImage,prepared:ExchangePrepared,
        anchor:u32,lineage:u64,comm:&Roce,stream:&Stream)->Result<u32>{
        ensure!(!self.poisoned&&Rc::ptr_eq(&self.image_owner,&saved.owner)&&anchor<129280
            &&target.profile_is_current()&&target.position()==self.draft.position()?,"session exchange owner");
        self.poisoned=true;
        target.exchange_state_image(&mut saved.target,prepared.target,lineage,comm,stream)?;
        self.draft.restore_state_image(&saved.draft,comm,stream)?;saved.draft=prepared.draft;
        let temperature=self.draft.draft_temperature();let conditioning=self.draft.conditioning_split();
        self.draft.set_draft_temperature(saved.draft_temperature,stream)?;
        self.draft.set_conditioning_split(saved.conditioning_split,stream)?;
        saved.draft_temperature=temperature;saved.conditioning_split=conditioning;
        std::mem::swap(&mut self.target_seed,&mut saved.target_seed);
        std::mem::swap(&mut self.stop_token,&mut saved.stop_token);
        std::mem::swap(&mut self.history_lookup,&mut saved.history_lookup);
        std::mem::swap(&mut self.draft_warmed,&mut saved.draft_warmed);
        std::mem::swap(&mut self.causal_three_threshold,&mut saved.causal_three_threshold);
        std::mem::swap(&mut self.causal_two_threshold,&mut saved.causal_two_threshold);
        std::mem::swap(&mut self.causal_full_threshold,&mut saved.causal_full_threshold);
        std::mem::swap(&mut self.causal_short_rows,&mut saved.causal_short_rows);
        std::mem::swap(&mut self.block_verification,&mut saved.block_verification);
        std::mem::swap(&mut self.acceptance_temperature_frontiers,&mut saved.acceptance_temperature_frontiers);
        let restored=saved.anchor;saved.anchor=anchor;
        ensure!(target.position()==self.draft.position()?
            &&self.history_lookup.as_ref().is_none_or(|h|h.position()==target.position()),"exchanged session frontiers disagree");
        self.poisoned=false;Ok(restored)
    }
    pub(crate) fn restore_state_image(&mut self, target: &mut Target, saved: &StateImage,
        lineage: u64, comm: &Roce, stream: &Stream) -> Result<u32> {
        ensure!(!self.poisoned && Rc::ptr_eq(&self.image_owner, &saved.owner)
            && target.profile_is_current() && target.position() == self.draft.position()?,
            "session image restore owner/frontier");
        let cpu_bytes = saved.history_lookup.as_ref().map_or(0, HistoryLookup::reserved_workspace_bytes);
        ensure!(self.cuda.admission_memory()?.admitted_available >= crate::memory::floor_bytes() + cpu_bytes,
            "restored lookup exceeds reserved UMA headroom");
        let lookup = saved.history_lookup.as_ref().map(HistoryLookup::clone_reserved);
        self.poisoned = true;
        target.restore_state_image(&saved.target, lineage, comm, stream)?;
        self.draft.restore_state_image(&saved.draft, comm, stream)?;
        self.draft.set_draft_temperature(saved.draft_temperature,stream)?;
        self.draft.set_conditioning_split(saved.conditioning_split,stream)?;
        self.block_verification=saved.block_verification;
        self.target_seed = saved.target_seed;
        self.stop_token = saved.stop_token;
        self.history_lookup = lookup;
        self.draft_warmed = saved.draft_warmed;
        self.causal_three_threshold = saved.causal_three_threshold;
        self.causal_two_threshold = saved.causal_two_threshold;
        self.causal_full_threshold = saved.causal_full_threshold;
        self.causal_short_rows = saved.causal_short_rows;
        self.acceptance_temperature_frontiers = saved.acceptance_temperature_frontiers;
        ensure!(target.position() == self.draft.position()?
            && self.history_lookup.as_ref().is_none_or(|h| h.position() == target.position()),
            "restored session frontiers disagree");
        self.poisoned = false;
        Ok(saved.anchor)
    }
    pub(crate) fn configure_history_lookup(&mut self, target: &Target, prefix: &[u32]) -> Result<usize> {
        ensure!(!self.poisoned && self.history_lookup.is_none() && self.rejection.is_some()
            && target.profile_is_current() && target.position() == prefix.len() as u64
            && self.draft.position()? == target.position(), "history lookup initial accepted history");
        let lookup=new_history_lookup(&self.cuda,target.context_capacity(),prefix)?;
        let bytes=lookup.workspace_bytes();
        self.history_lookup = Some(lookup);
        Ok(bytes)
    }
    pub(crate) fn checkpoint_history_lookup(&self,previous:u64,delta:&[u32])->Result<Option<HistoryLookup>> {
        ensure!(!self.poisoned&&self.draft.position()?==previous
            &&self.history_lookup.as_ref().is_none_or(|h|h.position()==previous),
            "history checkpoint requires the accepted drafter frontier");
        let mut copy=self.history_lookup.as_ref().map(HistoryLookup::clone_reserved);
        if let Some(h)=&mut copy {if !delta.is_empty(){h.commit(previous,delta)?;}}
        Ok(copy)
    }
    pub(crate) fn install_history_lookup(&mut self,target:&Target,lookup:Option<HistoryLookup>)->Result<()> {
        ensure!(!self.poisoned&&self.history_lookup.is_none()&&target.profile_is_current()
            &&self.draft.position()?==target.position()
            &&lookup.as_ref().is_none_or(|h|h.position()==target.position()),
            "history checkpoint installation frontier");
        self.history_lookup=lookup;Ok(())
    }
    pub(crate) fn configure_history_policy_for_probe(&mut self, minimum: Option<usize>) -> Result<()> {
        ensure!(!self.poisoned && self.block_verification && self.target_seed.is_some()
            && (std::env::var("CANADA_SCRATCH_HISTORY_POLICY_SCREEN").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_SOLO_ECONOMICS_SCREEN").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_EXPERT_PREFETCH_SCREEN").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_DRAFT_TEMPERATURE_SCREEN").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_LONG_HISTORY_GENERATION").as_deref()==Ok("1")),
            "history policy override requires the isolated block/T1 screen");
        self.history_lookup.as_mut().ok_or_else(||anyhow::anyhow!("history policy needs a lookup"))?
            .set_minimum_match(minimum)
    }
    pub(crate) fn expert_prefetch_inputs_for_probe(&mut self,target:&Target,anchor:u32,seed:u64,
        comm:&Roce,stream:&Stream)->Result<[u32;6]> {
        ensure!(!self.poisoned&&self.block_verification&&self.target_seed.is_some()
            &&self.draft.position()?==target.position()&&target.profile_is_current()
            &&std::env::var("CANADA_SCRATCH_EXPERT_PREFETCH_SCREEN").as_deref()==Ok("1"),
            "expert hint fixed proposals need committed sampled frontier");
        let before=self.draft.accepted_repeat_digest(stream)?;
        self.draft.run(anchor,comm,stream)?;
        let mut inputs=[anchor;6];
        inputs[1..].copy_from_slice(&self.draft.stochastic_proposals(seed,comm,stream)?);
        ensure!(self.draft.accepted_repeat_digest(stream)?==before,"hint proposals changed accepted draft state");
        Ok(inputs)
    }
    pub(crate) fn configure_history_strong_backup_for_probe(&mut self,enabled:bool)->Result<()> {
        ensure!(!self.poisoned&&self.block_verification&&self.target_seed.is_some()
            &&std::env::var("CANADA_SCRATCH_HISTORY_STRONG_BACKUP_SCREEN").as_deref()==Ok("1"),
            "strong-history backup requires its finite screen");
        ensure!(self.cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+32*1024usize.pow(2),
            "history backup rebuild must preserve reserved headroom");
        self.history_lookup.as_mut().ok_or_else(||anyhow::anyhow!("history lookup absent"))?
            .resize_strong_backup_for_probe(enabled)
    }
    pub(crate) fn configure_full_commit_metadata_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(!self.poisoned,"full-commit metadata needs a healthy session");
        self.draft.set_full_commit_metadata_for_probe(enabled,stream)
    }
    pub(crate) fn configure_history_maximum_for_probe(&mut self,maximum:usize)->Result<usize> {
        ensure!(!self.poisoned && self.block_verification && self.target_seed.is_some()
            && std::env::var("CANADA_SCRATCH_HISTORY_MATCH_SCREEN").as_deref()==Ok("1"),
            "history maximum requires the isolated causal proposal screen");
        ensure!(self.cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+32*1024usize.pow(2),
            "history index rebuild must preserve reserved headroom");
        let lookup=self.history_lookup.as_mut().ok_or_else(||anyhow::anyhow!("history index absent"))?;
        lookup.resize_maximum_for_probe(maximum)?;Ok(lookup.workspace_bytes())
    }
    /// Diagnostic only: price five teacher-forced probabilities before an
    /// otherwise eligible copy. No candidate selection depends on the separate
    /// stochastic query used to validate the conditioned probabilities.
    pub(crate) fn observe_long_history_for_probe(&mut self,target:&Target,anchor:u32,
        limit:usize,comm:&Roce,stream:&Stream)->Result<Option<serde_json::Value>> {
        ensure!(std::env::var("CANADA_SCRATCH_HISTORY_CONFIDENCE_SCREEN").as_deref()==Ok("1")
            && !self.poisoned && target.profile_is_current()
            && target.position()==self.draft.position()? && self.block_verification,
            "history confidence observation requires the isolated causal screen");
        if limit<16 || self.target_seed.is_none()
            || !(self.draft_warmed || self.draft.sampled_graphs_ready()) {return Ok(None)}
        let Some((source,copy))=self.history_lookup.as_ref()
            .map(|h|h.propose_long(target.position(),anchor)).transpose()?.flatten()
            else{return Ok(None)};
        let began=Instant::now();
        self.draft.run(anchor,comm,stream)?;
        let query_ms=began.elapsed().as_secs_f64()*1000.;
        let known:[u32;5]=copy[..5].try_into().unwrap();
        let(probabilities,confidence,first_capture)=self.draft.score_history_for_probe(&known,comm,stream)?;
        let observation_ms=began.elapsed().as_secs_f64()*1000.;
        // Outside the priced path: normal sampled Markov rows are an
        // independent conditioning check wherever predecessor identities match.
        let seed=crate::categorical::block_round_seed(self.target_seed.unwrap(),target.position());
        let neural=self.draft.stochastic_proposals(seed,comm,stream)?;
        let q=self.draft.proposal_probabilities_prefix(5)?;
        let mut checked=0;
        for i in 0..5{
            if i!=0&&neural[i-1]!=copy[i-1]{continue}
            let raw=unsafe{self.cuda.read_device(q.ptr()+((i*129280+copy[i]as usize)*8)as u64,8)}?;
            ensure!(f64::from_le_bytes(raw.try_into().unwrap())==probabilities[i],
                "teacher-forced probability disagrees with the same normal Markov conditioning");checked+=1;
        }
        ensure!(target.position()==self.draft.position()?,"observation advanced accepted state");
        Ok(Some(serde_json::json!({"position":target.position(),"copy_root":copy[0],
            "q_copy_root":probabilities[0],"q_copy_prefix":probabilities,"copy_prefix":known,
            "root_confidence":confidence[0],"copy_confidences":confidence,
            "neural_root":neural[0],"match_tokens":source.matched,
            "source_end":source.source_end,"query_ms":query_ms,"first_capture":first_capture,
            "normal_conditioning_checks":checked,"observation_ms":observation_ms,
            "diagnostic_total_ms":began.elapsed().as_secs_f64()*1000.})))
    }
    /// Select the long profile using only committed-history match metadata.
    /// A point-mass proposal accepts matching independent target draws; the
    /// first mismatch/bonus is a fresh target token. No approximate acceptance.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_long_history_step(&mut self,target:&mut Target,longer:&mut Target,
        buffers:&mut LongHistoryBuffers,anchor:u32,limit:usize,comm:&Roce,stream:&Stream)->Result<Option<Step>> {
        match longer.physical_rows() {
            16=>self.try_history_block::<16>(target,longer,buffers,anchor,limit,comm,stream),
            32 if self.launch_flags.history32=>
                self.try_history_block::<32>(target,longer,buffers,anchor,limit,comm,stream),
            _=>anyhow::bail!("long history target geometry is not enabled"),
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn try_history_block<const ROWS:usize>(&mut self,target:&mut Target,longer:&mut Target,
        buffers:&mut LongHistoryBuffers,anchor:u32,limit:usize,comm:&Roce,stream:&Stream)->Result<Option<Step>> {
        ensure!(self.launch_flags.long_history
            && !self.poisoned && target.physical_rows()==6 && longer.physical_rows()==ROWS && buffers.rows==ROWS
            && target.profile_is_current() && target.position()==self.draft.position()?,
            "long history session frontier/profile");
        if limit<ROWS || self.target_seed.is_none()
            || !(self.draft_warmed || self.draft.sampled_graphs_ready()) {return Ok(None)}
        let begin=Instant::now();let start=target.position();
        let Some((source,inputs))=self.history_lookup.as_ref()
            .map(|h|h.propose_block_minimum::<ROWS>(start,anchor,buffers.minimum_match)).transpose()?.flatten() else{return Ok(None)};
        let proposals=&inputs[1..];
        self.poisoned=true;
        let result=(|| {
            longer.handoff_from(target)?;let drafted=Instant::now();
            longer.begin_verification(&inputs,comm,stream)?;
            // Diagnostic only (CANADA_SCRATCH_CLAUDE_LOGIT_EXPORT): this path's rows join the exported decode chain.
            let exported_rows=if crate::logit_export::dir().is_some() {
                Some(crate::logit_export::capture(&self.cuda,longer,ROWS,stream)?)} else {None};
            let predictions=longer.verification_history_sample(self.target_seed.unwrap(),comm,stream)?;
            let verify_host_ms=longer.verification_host_ms()?;let verified=Instant::now();
            ensure!(predictions.len()==ROWS,"long history target draws");
            let mut matched=0;
            while matched<ROWS-1 && proposals[matched]==predictions[matched] {matched+=1;}
            let accepted=stop_frontier(&predictions,1+matched,self.stop_token);
            if let (Some(rows),Some(dir))=(exported_rows.as_ref(),crate::logit_export::dir()) {
                crate::logit_export::commit(&dir,longer.rank_for_export(),start,&inputs[..accepted],
                    predictions[accepted-1],ROWS,rows)?;
            }
            buffers.features.run(longer,stream)?;
            longer.finish_verification(accepted,comm,stream)?;let committed=Instant::now();
            let (features,live)=buffers.features.values()?;ensure!(live==ROWS,"long feature capture extent");
            // Four or more M6 chunks repaid the replicated wide projection in
            // the finite component screen. Rejected features are never copied
            // into its input or committed to draft KV. Short copies keep M6.
            if buffers.wide_enabled&&accepted>=19 {
                let padded=buffers.wide.as_ref().ok_or_else(||anyhow::anyhow!("wide copy workspace absent"))?;
                unsafe {
                    self.cuda.zero_device(padded.ptr(),padded.bytes(),stream)?;
                    self.cuda.copy_device(padded.ptr(),features.ptr(),accepted*15360*2,stream)?;
                }
                self.draft.inject_copy_wide(padded,accepted,start,comm,stream)?;
            }else{
                for offset in (0..accepted).step_by(6) {
                    let rows=(accepted-offset).min(6);
                    unsafe {
                        self.cuda.zero_device(buffers.chunk.ptr(),buffers.chunk.bytes(),stream)?;
                        self.cuda.copy_device(buffers.chunk.ptr(),features.ptr()+(offset*15360*2)as u64,rows*15360*2,stream)?;
                    }
                    self.draft.inject_distributed(&buffers.chunk,rows,rows,start+offset as u64,comm,stream)?;
                }
            }
            self.history_lookup.as_mut().unwrap().commit(start,&inputs[..accepted])?;
            target.handoff_from(longer)?;
            ensure!(target.position()==start+accepted as u64 && self.draft.position()?==target.position()
                && self.history_lookup.as_ref().unwrap().position()==target.position(),"long history publication frontiers");
            Ok(Step{tokens:inputs[..accepted].to_vec(),next_anchor:predictions[accepted-1],
                draft_tokens:proposals.to_vec(),matching_proposals:accepted-1,verified_rows:ROWS,draft_confidence:None,
                proposal_source:"long_history_point_mass",lookup_match_tokens:Some(source.matched),lookup_source_end:Some(source.source_end),
                profiled_verify:false,profile_append_metrics:None,position:target.position(),verify_host_ms,
                acceptance_expected_lengths:None,acceptance_temperature_probe:None,
                phase_ms:[drafted.duration_since(begin).as_secs_f64()*1000.,verified.duration_since(drafted).as_secs_f64()*1000.,
                    committed.duration_since(verified).as_secs_f64()*1000.,committed.elapsed().as_secs_f64()*1000.]})
        })();
        if result.is_ok(){self.poisoned=false;}result.map(Some)
    }
    /// The worker's verdict for the next single-conversation step (guarded draft-head subset only).
    pub(crate) fn set_head_subset_ok(&mut self, ok: bool) { self.head_subset_ok = ok; }
    /// Draft passes by head form [whole, subset] since the last call (diagnostic).
    pub(crate) fn take_draft_head_passes(&mut self) -> [u64; 2] { self.draft.take_head_passes() }
    pub(crate) fn step(
        &mut self,
        target: &mut Target,
        anchor: u32,
        limit: usize,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<Step> {
        self.step_with_profile_barrier(target, anchor, limit, comm, stream, &mut || Ok(()))
    }
    pub(crate) fn configure_prefix_stop_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()>{
        ensure!(!self.poisoned,"prefix policy switch on poisoned session");
        self.draft.set_prefix_stop_for_probe(enabled,stream)
    }
    pub(crate) fn set_skip_unused_draft_for_probe(&mut self, enabled: bool) -> Result<()> {
        ensure!(!self.poisoned
            && std::env::var("CANADA_SCRATCH_UNUSED_DRAFT_SCREEN").as_deref()==Ok("1"),
            "unused draft override requires finite screen");
        self.skip_unused_draft=enabled;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_with_profile_barrier(
        &mut self,
        target: &mut Target,
        anchor: u32,
        limit: usize,
        comm: &Roce,
        stream: &Stream,
        profile_barrier: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Step> {
        ensure!(
            !self.poisoned
                && (1..=6).contains(&limit)
                && target.profile_is_current()
                && target.position() == self.draft.position()?,
            "DSpark session frontier/visibility"
        );
        let start = target.position();
        self.poisoned = true;
        let result = (|| {
            // draft_prelaunch: the previous step launched this draft pass and nothing touched the drafter since.
            let ran = self.prelaunched.take() == Some((start, anchor)) && self.draft.ready_for(anchor)
                && comm.query_policy_current(crate::dspark_kv::QUERY_ROWS);
            let drafted = self.draft_phase_ran(target, start, anchor, limit, ran, true, comm, stream)?;
            self.finish_phase(target, drafted, anchor, limit, comm, stream, profile_barrier, None).map(|(step, _)| step)
        })();
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }
    /// Everything before verification: text lookup or the drafter, and the causal
    /// row count. `start` is this session's frontier (the owner's own position
    /// for an ordinary step, the session's history for a batched one).
    #[allow(clippy::too_many_arguments)]
    /// Whether draft_phase with these arguments runs the neural drafter (not a target-only step and
    /// no history-lookup proposal); the same decisions draft_phase makes, without side effects.
    /// config::batched_proposal_launch: the round seed and prefix threshold, if `draft_phase_ran` with a pooled
    /// pass already run would take the prefix-stop proposal path for this step (the same conditions, in order).
    fn batched_proposal_plan(&self, target: &Target, start: u64, limit: usize) -> Option<(u64, f32)> {
        let context_tail = target.context_capacity().saturating_sub(start as usize) < crate::dspark_kv::QUERY_ROWS;
        if context_tail || (self.skip_unused_draft && limit == 1) || self.reasoning_temperature_probe.is_some()
            || self.launch_flags.acceptance_temperature_screen || self.rejection.is_none() { return None; }
        let seed = self.target_seed.map(|seed| if self.block_verification {
            crate::categorical::block_round_seed(seed, start) } else { seed })?;
        let threshold = self.causal_three_threshold.filter(|_| self.causal_full_threshold.is_none() && self.causal_short_rows == 3)?;
        Some((seed, threshold))
    }
    fn neural_draft_needed(&self, target: &Target, start: u64, anchor: u32, limit: usize) -> Result<bool> {
        let context_tail = target.context_capacity().saturating_sub(start as usize) < crate::dspark_kv::QUERY_ROWS;
        let target_only = context_tail || (self.skip_unused_draft && limit==1);
        if target_only { return Ok(false); }
        let fallback_warm=self.draft_warmed || (self.target_seed.is_some() && self.draft.sampled_graphs_ready());
        let lookup = if fallback_warm {
            self.history_lookup.as_ref().map(|history| history.propose(start, anchor)).transpose()?.flatten()
        } else { None };
        Ok(lookup.is_none() && self.reasoning_temperature_probe.is_none())
    }
    fn draft_phase(
        &mut self,
        target: &mut Target,
        start: u64,
        anchor: u32,
        limit: usize,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<Drafted> {
        self.draft_phase_ran(target, start, anchor, limit, false, true, comm, stream)
    }
    /// As draft_phase; `ran`: a pooled pass (QueryNetwork::run_paired) already drafted from `anchor`.
    #[allow(clippy::too_many_arguments)]
    fn draft_phase_ran(
        &mut self,
        target: &mut Target,
        start: u64,
        anchor: u32,
        limit: usize,
        ran: bool,
        own_step: bool,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<Drafted> {
        // The fixed five-row bidirectional drafter cannot query past the
        // advertised context. Finish the last four positions with ordinary
        // target draws instead; no draft probabilities are used at this edge.
        let context_tail = target.context_capacity().saturating_sub(start as usize)
            < crate::dspark_kv::QUERY_ROWS;
        // At a one-row boundary no proposal can be accepted. Reuse the
        // context-tail target draw, still committing target features into
        // drafter KV below so a cached continuation has complete causal state.
        let target_only = context_tail || (self.skip_unused_draft && limit==1);
            let begin = Instant::now();
            let round_seed=self.target_seed.map(|seed|if self.block_verification {
                crate::categorical::block_round_seed(seed,start)
            } else { seed });
            // Prepare a genuinely cold fallback before using text lookup.
            // A prompt snapshot may predate capture while the loaded graph
            // bank is already warm. Its current sampled-policy captures can
            // be reused without an otherwise unnecessary neural draft.
            let fallback_warm=self.draft_warmed
                || (self.target_seed.is_some() && self.draft.sampled_graphs_ready());
            // A pooled pass already chose the neural drafter (neural_draft_needed); keep that choice.
            let lookup = if fallback_warm && !target_only && !ran {
                self.history_lookup.as_ref()
                    .map(|history| history.propose(start, anchor)).transpose()?.flatten()
            } else { None };
            let mut acceptance_temperature_probe=None;
            let (draft_tokens, draft_confidence, scheduled_rows, produced_proposals) = if target_only {
                ([anchor; 5], None, 1, 0)
            } else if let Some(proposal) = &lookup {
                // History-only deterministic proposals have point-mass q.
                // No draft result or stale confidence is consumed on this path.
                (proposal.tokens, None, 6, 5)
            } else {
                if ran { ensure!(self.draft.ready_for(anchor), "pooled draft pass for another anchor"); }
                else {
                    if self.head_subset_guarded { self.draft.select_head_subset(self.head_subset_ok, stream)?; }
                    self.draft.run(anchor, comm, stream)?;
                    // config::engram_anchor_prefetch: the verification after this draft pass starts at `anchor`. Its
                    // table rows are requested here, behind the drafter's launch, and read while the drafter runs.
                    if own_step && crate::config::engram_anchor_prefetch() && target.position() == start {
                        target.prefetch_anchor_engram(anchor)?;
                    }
                }
                if self.reasoning_temperature_probe.is_some() && self.acceptance_temperature_frontiers<8
                    && start>=262117+64*self.acceptance_temperature_frontiers as u64 {
                    acceptance_temperature_probe=Some(self.probe_reasoning_temperatures(target,anchor,
                        round_seed.ok_or_else(||anyhow::anyhow!("reasoning probe needs seeded sampling"))?,comm,stream)?);
                }
                if self.acceptance_temperature_frontiers<24
                    && self.launch_flags.acceptance_temperature_screen {
                    acceptance_temperature_probe=Some(self.probe_acceptance_temperatures(target,anchor,
                        round_seed.ok_or_else(||anyhow::anyhow!("temperature probe needs seeded sampling"))?,comm,stream)?);
                }
                let tokens = if let Some(tokens) = self.batched_proposals.take() {
                    ensure!(ran && !own_step, "shared proposal launch outside a batched step");
                    tokens
                } else if self.rejection.is_some() && self.target_seed.is_some() {
                    let seed=round_seed.ok_or_else(||anyhow::anyhow!("stochastic draft needs target seed"))?;
                    if let Some(threshold)=self.causal_three_threshold.filter(|_|
                        self.causal_full_threshold.is_none()&&self.causal_short_rows==3) {
                        self.draft.stochastic_proposals_causal_three(seed,threshold,comm,stream)?
                    }else{self.draft.stochastic_proposals(seed,comm,stream)?}
                } else {
                    self.draft.greedy_proposals(comm, stream)?
                };
                let (confidence, scheduled) = self.draft.confidence_schedule()?;
                if step_confidence_log() {
                    // One line before the step's own: the probability the drafter gave each proposal it drew.
                    let q = self.draft.selected_q_for_log()?;
                    eprintln!("stepq q={:.6},{:.6},{:.6},{:.6},{:.6}", q[0], q[1], q[2], q[3], q[4]);
                }
                self.draft_warmed = true;
                (tokens, Some(confidence), scheduled, self.draft.produced_proposals()?)
            };
            // A global utility argmax over future confidences is not a stopping
            // time and may bias q. The cumulative-threshold rule is causal:
            // confidence[j] uses only anchor/proposals[..j], before draw j.
            // Cumulative products cannot recover after crossing the threshold;
            // inclusion of proposal j is thus decided before observing it.
            // At that stopping boundary, the bonus is a fresh target draw.
            let live = if target_only {
                1
            } else if lookup.is_some() {
                limit
            } else if let Some(threshold) = self.causal_full_threshold {
                let confidence = draft_confidence
                    .ok_or_else(|| anyhow::anyhow!("full causal depth needs current draft confidence"))?;
                causal_full_prefix_rows(&confidence, threshold, limit)
            } else if let Some(threshold) = self.causal_three_threshold {
                let confidence = draft_confidence
                    .ok_or_else(|| anyhow::anyhow!("causal depth needs current draft confidence"))?;
                if let Some(two)=self.causal_two_threshold {
                    causal_two_three_rows(&confidence,two,threshold,limit)
                } else if let Some(ladder) = self.causal_ladder.filter(|l| own_step && self.causal_short_rows == 3
                    && l[0] <= threshold) {
                    // The drafter stopped after two proposals iff c1 c2 c3 < threshold; the ladder then stops at
                    // depth 2 or 3 and never asks for a proposal that was not drawn.
                    causal_ladder_rows(&confidence, threshold, ladder, limit).min(produced_proposals + 1)
                } else {
                    causal_prefix_rows(&confidence, threshold, self.causal_short_rows, limit)
                }
            } else if self.rejection.is_some() && !self.stochastic_prefix_schedule {
                limit
            } else {
                scheduled_rows.min(limit)
            };
            let drafted = Instant::now();
            ensure!(live-1<=produced_proposals,"verification would consume an unproduced draft");
            let mut verify = [anchor; 6];
            verify[1..].copy_from_slice(&draft_tokens);
            let local_history=self.local_history_sampling&&lookup.is_some();
            let local_target=target.local_target_enabled()&&self.draft.hierarchical_active()
                &&self.block_verification&&self.target_seed.is_some()&&(lookup.is_none()||local_history)&&!target_only&&live>1;
            Ok(Drafted { start, begin, drafted, round_seed, lookup, target_only, context_tail, draft_tokens,
                draft_confidence, produced_proposals, live, verify, acceptance_temperature_probe, local_history, local_target })
    }
    /// Verification (unless a batched step already ran it), sampling, acceptance,
    /// feature capture, the owner's commit (ordinary steps only), drafter
    /// injection and the lookup commit. `batched` is this session's first row in
    /// a batched verification; its commit is the owner's finish_batched_verification.
    #[allow(clippy::too_many_arguments)]
    fn finish_phase(
        &mut self,
        target: &mut Target,
        drafted: Drafted,
        anchor: u32,
        limit: usize,
        comm: &Roce,
        stream: &Stream,
        profile_barrier: &mut dyn FnMut() -> Result<()>,
        batched: Option<usize>,
    ) -> Result<(Step, usize)> {
        let Drafted { start, begin, drafted, round_seed, lookup, target_only, context_tail, draft_tokens,
            draft_confidence, produced_proposals, live, verify, acceptance_temperature_probe, local_history, local_target } = drafted;
        let _ = (anchor, round_seed, local_history);
            let profile_now = batched.is_none() && self.profile_verify
                && !self.profiled_verify
                && lookup.is_none()
                && start >= 640
                && live == 6
                && target.verification_graph_is_warm(live);
            if profile_now {
                stream.synchronize()?;
                profile_barrier()?;
                self.cuda.profiler_start()?;
                // A peer must not enqueue its first model collective while
                // another device is still activating the profiler. The
                // resident diagnostic supplies CPU-only rendezvous here.
                if let Err(error) = profile_barrier() {
                    self.cuda.profiler_stop()?;
                    return Err(error);
                }
            }
            let mut block_prefix=None;
            let mut acceptance_expected_lengths=None;
            let mut next_root_temperature=None;
            let mut exported_rows:Option<Vec<u8>>=if batched.is_some() {self.batched_export.take()} else {None};
            let mut exported_head4:Option<Vec<u8>>=None;
            let mut exported_scheme:Option<(Vec<u8>,Vec<i32>)>=None;
            // A batched step already ran one verification over every session's rows
            // and staged this session's rows at row 0 (Target::stage_batched_session).
            // verify_sample_ahead: exactly the path that ends in sample_block_sharded below.
            let ahead = verify_sample_ahead() && batched.is_none() && !profile_now && self.target_seed.is_some()
                && lookup.is_none() && !target_only && live > 1 && self.block_verification
                && self.rejection.is_some() && self.draft.hierarchical_active() && crate::logit_export::dir().is_none();
            if batched.is_none() { target.defer_verification_completion(ahead); }
            let verified_result = if batched.is_some() {
                    Ok(())
                } else if local_target {
                    target.begin_local_verification(&verify[..live],comm,stream)
                }else{target.begin_verification(&verify[..live],comm,stream)}
                .and_then(|_| {
                    // Diagnostic only: copy this rank's logit shard before any sampler runs. (A batched step
                    // captured every conversation's rows once, before staging: batched_export above.)
                    if crate::logit_export::dir().is_some() && batched.is_none() {
                        exported_rows=Some(crate::logit_export::capture(&self.cuda,target,live,stream)?);
                        // The same head input through the drafter's 4-bit head (diagnostic; see logit_export).
                        if crate::logit_export::head4_enabled() && let Some(head)=self.draft.nvfp4_head_for_export() {
                            exported_head4=Some(head.export_logits(target.head_input_for_export()?,stream)?);
                        }
                        if crate::logit_export::head4_scheme_enabled() {
                            exported_scheme=target.head4_scheme_for_export(live,stream)?;
                        }
                    }
                    Ok(())
                })
                .and_then(|_| {
                    if self.target_seed.is_none() {
                        // Greedy p is a point mass: accept exactly the matching
                        // target-argmax prefix, with its first mismatch as bonus.
                        target.verification_greedy(stream)
                    } else if lookup.is_some() || target_only {
                        // An independent categorical target draw either equals
                        // the point-mass proposal or is its conditional residual.
                        // Taking the matching prefix and first mismatch/bonus
                        // therefore preserves p without constructing dense q.
                        let seed=self.target_seed.ok_or_else(||anyhow::anyhow!("lookup needs target sampling"))?;
                        if local_target {
                            let(logits,rows)=target.pending_local_target_logits()?;
                            ensure!(rows==live&&local_history,"point-mass target extent");
                            self.rejection.as_mut().ok_or_else(||anyhow::anyhow!("compact point-mass sampler"))?
                                .sample_point_mass_sharded(logits,live,seed,start+1,comm,stream)
                        }else{target.verification_sample(seed,stream)}
                    } else if let Some(rejection) = &mut self.rejection {
                        let (logits, rows) = if local_target{target.pending_local_target_logits()?}else{target.pending_logits()?};
                        ensure!(rows == live, "stochastic verification rows");
                        let sample = if self.block_verification { Rejection::sample_block } else { Rejection::sample };
                        let draw = if self.draft.hierarchical_active() {
                            ensure!(self.block_verification,"sharded q requires block rejection");
                            let(q,selected)=self.draft.sharded_probabilities(live-1)?;
                            let result=rejection.sample_block_sharded(logits,q,selected,&draft_tokens[..live-1],live,
                                round_seed.unwrap(),start+1,local_target,comm,stream)?;
                            target.complete_deferred_verification(comm,stream)?;
                            if local_target {rejection.capture_root_branch_for_probe(q,live,
                                self.target_seed.unwrap(),start+1,comm.rank())?;}
                            result
                        }else{sample(rejection,
                            logits,
                            self.draft.proposal_probabilities_prefix(live-1)?,
                            &draft_tokens[..live - 1],
                            live,
                            round_seed.unwrap(),
                            start + 1,
                            stream,
                        )?};
                        if let Some(trace)=self.buffered_draw_trace.as_ref().filter(|_|live>1) {
                            let compact=trace.borrow().compact_peer();
                            let detail=trace.borrow().peer_detail();
                            let q=self.draft.proposal_probabilities_prefix(live-1)?;
                            let mut sources:Vec<_>=if compact {Vec::new()}else{
                                rejection.buffered_draw_sources(live,logits.ptr(),q.ptr())?
                                    .into_iter().map(|(name,ptr,bytes)|(name.to_string(),ptr,bytes)).collect()};
                            if self.launch_flags.readout_layer_trace {
                                sources.extend(target.repeat_layer_sources(live,compact&&!detail)?);
                                if detail {sources.extend(target.repeat_ffn_sources(live)?);}
                            }
                            trace.borrow_mut().record(serde_json::json!({
                                "position":start,"anchor":anchor,"live":live,"target_seed":self.target_seed,
                                "round_seed":round_seed,"draft_tokens":&draft_tokens[..live-1],
                                "draw_ids":draw.ids,"accepted":draw.accepted,"accepted_prefix":draw.accepted_prefix,
                                "expected_first_id":null}),&sources,stream)?;
                        }
                        if let Some((path,position,expected_first))=&self.encoder_cache_trace {
                            // Optional failure-only capture reads no GPU tensor
                            // unless the already-completed sample differs. This
                            // avoids the earlier trace perturbing successful arms.
                            if *position==start&&expected_first.is_none_or(|id|draw.ids.first()!=Some(&id)) {
                                use sha2::{Digest,Sha256};
                                let probabilities=self.draft.proposal_probabilities_prefix(live-1)?;
                                let logits=unsafe{self.cuda.read_device(logits.ptr(),live*129280*4)}?;
                                let q=unsafe{self.cuda.read_device(probabilities.ptr(),(live-1)*129280*8)}?;
                                std::fs::create_dir(path)?;
                                std::fs::write(path.join("logits.bin"),&logits)?;
                                std::fs::write(path.join("q.bin"),&q)?;
                                rejection.capture_completed_draw_for_probe(path,live)?;
                                std::fs::write(path.join("metadata.json"),serde_json::to_vec_pretty(&serde_json::json!({
                                    "position":start,"anchor":anchor,"live":live,"target_seed":self.target_seed,
                                    "round_seed":round_seed,"draft_tokens":&draft_tokens[..live-1],
                                    "draw_ids":draw.ids,"accepted":draw.accepted,"accepted_prefix":draw.accepted_prefix,
                                    "expected_first_id":expected_first,
                                    "logits_sha256":format!("{:x}",Sha256::digest(&logits)),
                                    "q_sha256":format!("{:x}",Sha256::digest(&q))}))?)?;
                            }
                        }
                        ensure!(
                            self.block_verification || (0..live - 1)
                                .all(|i| draw.accepted[i] == (draw.ids[i] == draft_tokens[i])),
                            "rejection readout differs from accepted proposal"
                        );
                        block_prefix=draw.accepted_prefix;
                        acceptance_expected_lengths=draw.acceptance_expected_lengths;
                        if let Some(feedback)=self.root_feedback.as_mut().filter(|f|f.enabled()) {
                            next_root_temperature=feedback.observe(rejection.root_probabilities_after_draw_for_probe()?,
                                self.draft.proposal_probabilities_prefix(live-1)?,self.draft.root_logits_for_feedback()?,
                                self.draft.draft_schedule(),start,stream)?;
                        }
                        Ok(draw.ids)
                    } else {
                        match self.target_seed {
                            Some(seed) => target.verification_sample(seed, stream),
                            None => target.verification_greedy(stream),
                        }
                    }
                });
            if profile_now {
                self.cuda.profiler_stop()?;
                self.profiled_verify = true;
            }
            let predictions = verified_result?;
            // verify_sample_ahead: a no-op after the sharded sampler completed it; any other path completes it here.
            if batched.is_none() { target.complete_deferred_verification(comm, stream)?; }
            let profile_append_metrics = if profile_now {
                Some(serde_json::to_value(target.verification_metrics()?)?)
            } else {
                None
            };
            let verify_host_ms = target.verification_host_ms()?;
            if crate::config::host_stages() && batched.is_none() { record_single_stages(&target.verification_metrics()?.stages_ms); }
            ensure!(predictions.len() == live, "DSpark verifier rows");
            let verified = Instant::now();
            let mut matched = 0;
            while block_prefix.is_none() && matched + 1 < live && draft_tokens[matched] == predictions[matched] {
                matched += 1;
            }
            if let Some(prefix)=block_prefix { matched=prefix; }
            let accepted = stop_frontier(&predictions, (1 + matched).min(limit), self.stop_token);
            matched = accepted - 1;
            if let (Some(rows),Some(dir))=(exported_rows.take(),crate::logit_export::dir()) {
                crate::logit_export::commit(&dir,target.rank_for_export(),start,&verify[..accepted],
                    predictions[accepted-1],live,&rows)?;
                if let Some(head4)=exported_head4.take() {
                    crate::logit_export::commit_head4(&dir,target.rank_for_export(),start,accepted.min(live).min(5),&head4)?;
                }
                if let Some((scheme,counts))=exported_scheme.take() {
                    crate::logit_export::commit_head4_scheme(&dir,target.rank_for_export(),start,accepted.min(live).min(6),&scheme,&counts)?;
                }
                if crate::logit_export::q_enabled() && batched.is_none() {
                    // The drafter's rank-local proposal rows of this step are intact until its next pass.
                    let q=if lookup.is_none()&&!target_only&&live>1&&self.block_verification
                        &&self.rejection.is_some()&&self.draft.hierarchical_active() {
                        let (q,_)=self.draft.sharded_probabilities(live-1)?;
                        Some(crate::logit_export::capture_q(&self.cuda,q,live-1,stream)?)
                    } else {None};
                    crate::logit_export::commit_q(&dir,target.rank_for_export(),start,accepted,
                        &draft_tokens[..live-1],q.as_deref())?;
                }
            }
            // Capture before publication; rejected feature rows remain private.
            if let Some(offset) = batched {
                self.features.run_batched(target, offset, live, stream)?;
            } else {
                self.features.run(target, stream)?;
                target.finish_verification(accepted, comm, stream)?;
            }
            // A batched session's frontier is its own; the owner commits every
            // session together after all of them sampled (finish_batched_verification).
            let frontier = if batched.is_some() { start + accepted as u64 } else { target.position() };
            let committed = Instant::now();
            let (features, live) = self.features.values()?;
            self.draft.inject_distributed(features, live, accepted, start, comm, stream)?;
            if let Some(history) = &mut self.history_lookup {
                history.commit(start, &verify[..accepted])?;
                ensure!(history.position() == frontier, "lookup/target commit frontiers disagree");
            }
            ensure!(
                frontier == start + accepted as u64
                    && self.draft.position()? == frontier,
                "DSpark commit frontiers disagree"
            );
            // The just-verified q remains intact through acceptance and commit.
            // Only a later proposal can observe the updated temperature.
            if let Some(warmer)=next_root_temperature {
                self.draft.set_draft_schedule(if warmer{3}else{0},stream)?;
            }
            // draft_prelaunch: the next ordinary step would draft from this anchor at this frontier (no stop token,
            // no context tail, no text-lookup proposal): start that pass now, behind the injection on the stream.
            let next_anchor = predictions[accepted - 1];
            if draft_prelaunch() && batched.is_none() && self.stop_token != Some(next_anchor)
                && self.neural_draft_needed(target, frontier, next_anchor, 6)? {
                self.draft.run(next_anchor, comm, stream)?;
                self.prelaunched = Some((frontier, next_anchor));
            }
            Ok((Step {
                tokens: verify[..accepted].to_vec(),
                next_anchor: predictions[accepted - 1],
                draft_tokens: draft_tokens[..produced_proposals].to_vec(),
                matching_proposals: matched,
                verified_rows: live,
                draft_confidence,
                proposal_source: if context_tail { "context_tail_target" }
                    else if target_only { "one_row_target" }
                    else if lookup.is_some() { "history_point_mass" } else { "dspark" },
                lookup_match_tokens: lookup.as_ref().map(|p| p.matched),
                lookup_source_end: lookup.as_ref().map(|p| p.source_end),
                profiled_verify: profile_now,
                profile_append_metrics,
                verify_host_ms,
                acceptance_expected_lengths,
                acceptance_temperature_probe,
                position: frontier,
                phase_ms: [
                    drafted.duration_since(begin).as_secs_f64() * 1000.,
                    verified.duration_since(drafted).as_secs_f64() * 1000.,
                    committed.duration_since(verified).as_secs_f64() * 1000.,
                    committed.elapsed().as_secs_f64() * 1000.,
                ],
            }, accepted))
    }
    /// Finite batched screen: after a batched append of `tokens` at `start` for this
    /// session, capture that append's feature taps, inject them into this session's
    /// drafter (as a decode commit does) and commit them to its text lookup.
    pub(crate) fn absorb_batched_append_for_probe(&mut self, target: &Target, start: u64, tokens: &[u32],
        comm: &Roce, stream: &Stream) -> Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_BATCH_SCREEN").as_deref() == Ok("1") && !self.poisoned
            && self.draft.position()? == start, "batched screen prompt absorption");
        self.features.run(target, stream)?;
        let (features, live) = self.features.values()?;
        ensure!(live == tokens.len(), "batched screen absorbed rows");
        self.draft.inject_distributed(features, live, live, start, comm, stream)?;
        if let Some(history) = &mut self.history_lookup { history.commit(start, tokens)?; }
        Ok(())
    }
    /// Screen: separate drafter passes of committed sessions against one batched body pass
    /// (QueryNetwork::probe_batched_body). Nothing is proposed or accepted.
    pub(crate) fn probe_batched_draft_body(sessions: &mut [&mut GreedySession], anchors: &[u32], comm: &Roce,
        stream: &Stream) -> Result<serde_json::Value> {
        ensure!(sessions.iter().all(|s| !s.poisoned), "batched drafter screen requires committed sessions");
        let mut nets: Vec<&mut QueryNetwork> = sessions.iter_mut().map(|s| &mut s.draft).collect();
        QueryNetwork::probe_batched_body(&mut nets, anchors, comm, stream)
    }
    /// Finite batched screen: one decode step for several sessions sharing one verify
    /// owner. Each drafts on its own; one verification runs over every session's rows
    /// when their verification heads agree (otherwise one per session); each then
    /// samples its own staged rows with its own seed and drafts, captures and injects
    /// its features, and the owner commits every session of the group together.
    pub(crate) fn batched_step(parts: &mut [BatchedSession<'_>], target: &mut Target, comm: &Roce,
        stream: &Stream) -> Result<Vec<Step>> {
        ensure!(crate::config::batched_rows_admitted() && !parts.is_empty(), "batched decode screen");
        // Each session verifies in its own block of rows / sessions rows.
        let block = crate::target::batched_block(target.physical_rows(), parts.len());
        // Guarded draft-head subset: each conversation's own verdict (set by the worker from its committed tokens
        // before the step). Batched passes run the heads eagerly after the body graph, a conversation's own pass
        // from its own graph of that form; pooled FP8 heads are not used while the NVFP4 copy is active.
        for p in parts.iter_mut() {
            if p.session.head_subset_guarded { p.session.draft.select_head_subset(p.session.head_subset_ok, stream)?; }
        }
        for p in parts.iter() {
            let s = &*p.session;
            ensure!(!s.poisoned && (1..=6).contains(&p.limit) && p.limit <= block && target.profile_is_current()
                && p.history.position == s.draft.position()?
                && !s.profile_verify && s.reasoning_temperature_probe.is_none()
                && !s.launch_flags.acceptance_temperature_screen
                && s.buffered_draw_trace.is_none() && s.encoder_cache_trace.is_none()
                && s.root_feedback.as_ref().is_none_or(|f| !f.enabled()), "batched session frontier/probes");
        }
        // Diagnostics: CANADA_SCRATCH_BATCH_SCREEN_DRAFT_REVERSE=1 drafts the sessions in
        // reverse order; CANADA_SCRATCH_BATCH_SCREEN_INVALIDATE=1 drops the incremental
        // index-gather caches before every verification.
        let reverse = std::env::var("CANADA_SCRATCH_BATCH_SCREEN_DRAFT_REVERSE").as_deref() == Ok("1");
        let invalidate = std::env::var("CANADA_SCRATCH_BATCH_SCREEN_INVALIDATE").as_deref() == Ok("1");
        let sync = phase_sync();
        let mut timing = BatchedPhases::ZERO;
        let mut drafted: Vec<Option<Drafted>> = (0..parts.len()).map(|_| None).collect();
        let order: Vec<usize> = if reverse { (0..parts.len()).rev().collect() } else { (0..parts.len()).collect() };
        // Pooled draft heads (CANADA_SCRATCH_BATCHED_POOLED_HEAD=1): the conversations that will run the
        // neural drafter go in pairs, each pair's bodies and one pooled ten-row head in one pass.
        let mut ran = vec![false; parts.len()];
        // One body pass for the conversations that run the neural drafter (CANADA_SCRATCH_BATCHED_DRAFT_BODY=1):
        // at most three a pass, four as two and two; a single one left over drafts alone as before.
        let batch_bodies = crate::config::batched_draft_body() && parts.len() >= 2 && !reverse;
        if batch_bodies {
            let mut needs = Vec::new();
            for (i, p) in parts.iter().enumerate() {
                if p.session.neural_draft_needed(target, p.history.position, p.anchor, p.limit)? { needs.push(i); }
            }
            let groups: Vec<Vec<usize>> = if crate::config::batched_draft_fixed_groups() {
                // Stable composition (config::batched_draft_fixed_groups): the step's conversations 0-1 and 2-3
                // draft together only when both need the drafter (all three of a three-conversation step when
                // all three do), so each group's graph is captured once. Groups picked from whoever needs a
                // draft change with every text lookup, and every new set captures a graph.
                if parts.len() == 3 && needs.len() == 3 { vec![vec![0, 1, 2]] }
                else { (0..parts.len() / 2).map(|k| vec![2 * k, 2 * k + 1])
                    .filter(|g| g.iter().all(|i| needs.contains(i))).collect() }
            } else if needs.len() == 4 { vec![needs[..2].to_vec(), needs[2..].to_vec()] }
                else { needs.chunks(3).map(<[usize]>::to_vec).collect() };
            for group in groups.into_iter().filter(|g| g.len() >= 2) {
                let first = &parts[group[0]].session.draft;
                let Some(bodies) = first.batch_bodies() else { continue; };
                if !group.iter().all(|&i| parts[i].session.draft.shares_batch_bodies(first)) { continue; }
                let anchors: Vec<u32> = group.iter().map(|&i| parts[i].anchor).collect();
                let draft_began = Instant::now();
                let mut nets: Vec<&mut QueryNetwork> = Vec::with_capacity(group.len());
                for (i, p) in parts.iter_mut().enumerate() {
                    if group.contains(&i) { p.session.poisoned = true; nets.push(&mut p.session.draft); }
                }
                bodies.get().ok_or_else(|| anyhow::anyhow!("batched drafter bodies"))?.borrow_mut().run(&mut nets, &anchors, comm, stream)?;
                if sync {
                    stream.synchronize()?;
                    timing.draft_ms += draft_began.elapsed().as_secs_f64() * 1000.;
                }
                for &i in &group { ran[i] = true; }
            }
        }
        // With fixed body groups, the conversations no group took still pair their heads as before.
        if (!batch_bodies || crate::config::batched_draft_fixed_groups()) && crate::config::batched_pooled_head()
            && parts.len() >= 2 && !reverse {
            let mut needs = Vec::new();
            for (i, p) in parts.iter().enumerate() {
                if !ran[i] && p.session.neural_draft_needed(target, p.history.position, p.anchor, p.limit)? { needs.push(i); }
            }
            for pair in needs.chunks(2).filter(|c| c.len() == 2) {
                let (i, j) = (pair[0], pair[1]);
                let (lo, hi) = parts.split_at_mut(j);
                let (a, b) = (&mut lo[i], &mut hi[0]);
                if !a.session.draft.can_pool_query_head(&b.session.draft) { continue; }
                a.session.poisoned = true; b.session.poisoned = true;
                let draft_began = Instant::now();
                a.session.draft.run_paired(&mut b.session.draft, [a.anchor, b.anchor], comm, stream)?;
                if sync {
                    stream.synchronize()?;
                    timing.draft_ms += draft_began.elapsed().as_secs_f64() * 1000.;
                }
                ran[i] = true; ran[j] = true;
            }
        }
        // config::batched_proposal_launch: the conversations whose pass ran above launch the first segment of
        // their proposals back to back and share one wait; those past the prefix stop launch the rest and share a
        // second one. Each conversation's own draft phase below then finds its proposals drawn.
        if crate::config::batched_proposal_launch() && parts.len() >= 2 && !reverse {
            let draft_began = Instant::now();
            let mut plans: Vec<Option<f32>> = vec![None; parts.len()];
            for (i, p) in parts.iter_mut().enumerate() {
                if !ran[i] { continue; }
                let Some((seed, threshold)) = p.session.batched_proposal_plan(target, p.history.position, p.limit) else { continue; };
                ensure!(p.session.draft.ready_for(p.anchor), "pooled draft pass for another anchor");
                if p.session.draft.launch_causal_three_prefix(seed, comm, stream)? { plans[i] = Some(threshold); }
            }
            if plans.iter().any(Option::is_some) {
                stream.synchronize()?;
                let mut rest = vec![false; parts.len()];
                for (i, p) in parts.iter_mut().enumerate() {
                    let Some(threshold) = plans[i] else { continue; };
                    match p.session.draft.finish_causal_three_prefix(threshold, comm, stream)? {
                        Some(tokens) => p.session.batched_proposals = Some(tokens),
                        None => rest[i] = true,
                    }
                }
                if rest.iter().any(|&r| r) {
                    stream.synchronize()?;
                    for (i, p) in parts.iter_mut().enumerate() {
                        if rest[i] { p.session.batched_proposals = Some(p.session.draft.finish_causal_three_rest(comm)?); }
                    }
                }
                if sync { timing.draft_ms += draft_began.elapsed().as_secs_f64() * 1000.; }
            }
        }
        for i in order {
            let p = &mut parts[i];
            p.session.poisoned = true;
            let draft_began = Instant::now();
            drafted[i] = Some(p.session.draft_phase_ran(target, p.history.position, p.anchor, p.limit, ran[i], false, comm, stream)?);
            ensure!(p.session.batched_proposals.is_none(), "shared proposals were drawn and not used");
            if sync {
                stream.synchronize()?;
                let ms = draft_began.elapsed().as_secs_f64() * 1000.;
                timing.draft_ms += ms;
                timing.draft_max_ms = timing.draft_max_ms.max(ms);
            }
        }
        timing.lookups = drafted.iter().filter(|d| d.as_ref().is_some_and(|d| d.lookup.is_some())).count();
        let step_began = Instant::now();
        let own: Vec<(usize, usize, bool, Option<[f32; 5]>)> = drafted.iter().map(|d| {
            let d = d.as_ref().unwrap(); (d.live, d.produced_proposals, d.lookup.is_some(), d.draft_confidence) }).collect();
        // The batched draft gate: in a step of several conversations a row costs more than in a
        // conversation's own step, so low-survival drafts are left out (causal; see batched_gate).
        if parts.len() >= 2 {
            if let Some(threshold) = batched_gate(parts.len())? {
                for (d, p) in drafted.iter_mut().zip(parts.iter()) {
                    let d = d.as_mut().unwrap();
                    if d.lookup.is_some() || d.target_only { continue; }
                    let Some(confidence) = d.draft_confidence else { continue; };
                    // The drafter may have stopped early under the conversation's own causal rule
                    // (prefix draft stop): never verify a proposal it did not produce. That stop
                    // also used only confidences of included proposals, so the clamp stays causal.
                    let live = causal_full_prefix_rows(&confidence, threshold, p.limit)
                        .min(d.produced_proposals + 1);
                    ensure!(live >= 1, "batched gate rows");
                    d.live = live;
                }
            }
        }
        let lives: Vec<usize> = drafted.iter().map(|d| d.as_ref().unwrap().live).collect();
        timing.steps = 1;
        timing.live_rows = lives.iter().sum();
        timing.executed_rows = crate::target::batched_extent(&lives, target.physical_rows());
        let head = drafted[0].as_ref().unwrap().local_target;
        let groups: Vec<Vec<usize>> = if drafted.iter().all(|d| d.as_ref().unwrap().local_target == head) {
            vec![(0..parts.len()).collect()]
        } else { (0..parts.len()).map(|i| vec![i]).collect() };
        let mut steps: Vec<Option<Step>> = (0..parts.len()).map(|_| None).collect();
        for group in groups {
            let local = drafted[group[0]].as_ref().unwrap().local_target;
            let rows = group.iter().map(|&i| (parts[i].base, parts[i].history.clone(),
                drafted[i].as_ref().unwrap().live)).collect::<Vec<_>>();
            let ids = group.iter().flat_map(|&i| { let d = drafted[i].as_ref().unwrap(); d.verify[..d.live].to_vec() })
                .collect::<Vec<u32>>();
            // First traced step of a pair: the same rows verified alone, as a pair and as
            // the swapped pair (each aborted), hashed per row, before the real step.
            let mut first_step = None;
            // Once per process: the screen hands in a fresh trace vector every step.
            static FIRST_PAIR_CHECKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if group.len() == 2 && parts[group[0]].trace.is_some()
                && !FIRST_PAIR_CHECKED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                use sha2::{Digest, Sha256};
                let cuda = parts[group[0]].session.cuda.clone();
                let mut out = serde_json::Map::new();
                for (name, order) in [("solo_first", vec![0usize]), ("solo_second", vec![1]), ("pair", vec![0, 1]),
                    ("pair_swapped", vec![1, 0]), ("solo_first_again", vec![0])] {
                    let rows = order.iter().map(|&g| (parts[group[g]].base, parts[group[g]].history.clone(),
                        drafted[group[g]].as_ref().unwrap().live)).collect::<Vec<_>>();
                    let ids = order.iter().flat_map(|&g| { let d = drafted[group[g]].as_ref().unwrap(); d.verify[..d.live].to_vec() })
                        .collect::<Vec<u32>>();
                    if invalidate { target.invalidate_gather_caches_for_probe()?; }
                    target.set_batched_rows_for_probe(rows)?;
                    if local { target.begin_local_verification(&ids, comm, stream)?; } else { target.begin_verification(&ids, comm, stream)?; }
                    let mut hashes = Vec::new();
                    for (k, &g) in order.iter().enumerate() {
                        let live = drafted[group[g]].as_ref().unwrap().live;
                        target.stage_batched_session(target.batched_offset(k)?, live, stream)?;
                        stream.synchronize()?;
                        let (logits, n) = if local { target.pending_local_target_logits()? } else { target.pending_logits()? };
                        let row_bytes = if local { 32320 * 4 } else { 129280 * 4 };
                        let bytes = unsafe { cuda.read_device(logits.ptr(), n * row_bytes) }?;
                        hashes.push(bytes.chunks(row_bytes).map(|r| format!("{:x}", Sha256::digest(r))[..12].to_owned()).collect::<Vec<_>>());
                    }
                    target.abort_batched_verification_for_probe(stream)?;
                    out.insert(name.into(), serde_json::json!(hashes));
                }
                first_step = Some(serde_json::Value::Object(out));
            }
            if invalidate { target.invalidate_gather_caches_for_probe()?; }
            let verify_began = Instant::now();
            target.set_batched_rows_for_probe(rows)?;
            if local { target.begin_local_verification(&ids, comm, stream)?; }
            else { target.begin_verification(&ids, comm, stream)?; }
            if sync { stream.synchronize()?; timing.verify_ms += verify_began.elapsed().as_secs_f64() * 1000.; }
            // Diagnostic only (CANADA_SCRATCH_CLAUDE_LOGIT_EXPORT): this rank's shard of every conversation's rows,
            // read once before any conversation's rows are staged; each finish_phase commits its accepted rows.
            let exported = if crate::logit_export::dir().is_some() {
                Some(crate::logit_export::capture_all(&parts[group[0]].session.cuda, target, stream)?) } else { None };
            if let Some(stages) = target.last_stages_ms() { record_batched_stages(parts.len(), stages); }
            let finish_began = Instant::now();
            let mut accepted = Vec::with_capacity(group.len());
            for (k, &i) in group.iter().enumerate() {
                let d = drafted[i].take().unwrap(); let live = d.live;
                let offset = target.batched_offset(k)?;
                target.stage_batched_session(offset, live, stream)?;
                if let Some((rows, total)) = exported.as_ref() {
                    ensure!(offset + live <= *total, "batched logit export rows");
                    parts[i].session.batched_export = Some(rows[offset * 32320 * 4..(offset + live) * 32320 * 4].to_vec());
                }
                let mut record = None;
                if parts[i].trace.is_some() {
                    use sha2::{Digest, Sha256};
                    let s = &*parts[i].session;
                    stream.synchronize()?;
                    let (logits, rows) = if d.local_target { target.pending_local_target_logits()? } else { target.pending_logits()? };
                    let row_bytes = if d.local_target { 32320 * 4 } else { 129280 * 4 };
                    let l = unsafe { s.cuda.read_device(logits.ptr(), rows * row_bytes) }?;
                    let row_hashes = l.chunks(row_bytes).map(|r| format!("{:x}", Sha256::digest(r))[..12].to_owned()).collect::<Vec<_>>();
                    let q = if d.lookup.is_none() && !d.target_only && live > 1 && s.draft.hierarchical_active() {
                        let (q, _) = s.draft.sharded_probabilities(live - 1)?;
                        Some(format!("{:x}", Sha256::digest(unsafe { s.cuda.read_device(q.ptr(), q.len() * 8) }?)))
                    } else { None };
                    record = Some(serde_json::json!({"position": d.start, "group": group.len(), "offset": offset, "live": live,
                        "local_target": d.local_target, "lookup": d.lookup.is_some(), "draft_tokens": &d.draft_tokens[..live.saturating_sub(1)],
                        "draft_confidence": d.draft_confidence, "logits_sha256": format!("{:x}", Sha256::digest(&l)), "q_sha256": q,
                        "logits_rows": row_hashes, "verify": &d.verify[..live],
                        "first_step": if k == 0 { first_step.take() } else { None }}));
                }
                let (step, a) = parts[i].session.finish_phase(target, d, parts[i].anchor, parts[i].limit,
                    comm, stream, &mut || Ok(()), Some(offset))?;
                if let (Some(mut r), Some(trace)) = (record, parts[i].trace.as_mut()) {
                    r["accepted"] = serde_json::json!(a); r["tokens"] = serde_json::json!(step.tokens);
                    r["next_anchor"] = serde_json::json!(step.next_anchor);
                    trace.push(r);
                }
                steps[i] = Some(step); accepted.push(a);
            }
            if sync { stream.synchronize()?; timing.finish_ms += finish_began.elapsed().as_secs_f64() * 1000.; }
            let commit_began = Instant::now();
            let histories = target.finish_batched_verification(&accepted, stream)?;
            if sync { stream.synchronize()?; timing.commit_ms += commit_began.elapsed().as_secs_f64() * 1000.; }
            timing.groups += 1;
            for (&i, h) in group.iter().zip(histories) { parts[i].history = h; parts[i].session.poisoned = false; }
        }
        timing.synced = usize::from(sync);
        record_batched_phases(parts.len(), timing);
        if comm.rank() == 0 { trace_batched_step(|| {
            let parts_json: Vec<serde_json::Value> = parts.iter().zip(&own).zip(&steps).map(|((p, o), s)| {
                let s = s.as_ref().unwrap();
                serde_json::json!({"limit":p.limit,"own_live":o.0,"produced":o.1,"lookup":o.2,"confidence":o.3,
                    "live":s.verified_rows,"accepted":s.matching_proposals,"tokens":s.tokens.len()})
            }).collect();
            serde_json::json!({"conversations":parts.len(),"rows":target.physical_rows(),
                "after_draft_ms":step_began.elapsed().as_secs_f64() * 1000.,"parts":parts_json})
        }); }
        Ok(steps.into_iter().map(|s| s.unwrap()).collect())
    }
    /// Resume the retained drafter after a new target EXTEND request. The
    /// caller includes the previous pending emitted anchor in `ids` exactly
    /// when that token belongs to the next prompt. It was never in target KV.
    pub(crate) fn resume_after_extend(
        &mut self,target:&Target,previous:u64,ids:&[u32],stream:&Stream,
    ) -> Result<()> {
        let added = ids.len();
        ensure!(!self.poisoned && added>0 && target.physical_rows()==128
            && target.profile_is_current() && self.draft.position()?==previous
            && previous.checked_add(added as u64)==Some(target.position()),
            "DSpark retained EXTEND ownership/frontier");
        self.poisoned=true;
        self.prefill_features.run(target,stream)?;
        let (values,live)=self.prefill_features.values()?;
        // An ordinary short EXTEND evaluates only its new decoder rows.
        // A resumed ragged wide chunk instead replays the complete128-row
        // decoder tail, including old rows under the original request floor.
        // In that case those recomputed features replace the entire drafter
        // window even when this interruption added fewer than128 tokens.
        ensure!(live==128 || (added<128 && live==added),
            "DSpark retained EXTEND feature rows");
        if live<128 {
            // Old accepted features still occupy the draft's bounded window.
            // Only this request's new rows were evaluated by the decoder.
            self.draft.inject(values,live,live,previous,stream)?;
        } else {
            // The entire128-token draft window is replaced by the target tail.
            self.draft.reset(stream)?;
            self.draft.inject_prefill_tail(values,live,target.position(),stream)?;
        }
        ensure!(self.draft.position()?==target.position(),"DSpark EXTEND commit frontier");
        if let Some(history) = &mut self.history_lookup {
            history.commit(previous, ids)?;
            ensure!(history.position() == target.position(), "lookup EXTEND commit frontier");
        }
        self.poisoned=false;
        Ok(())
    }
    /// A whole retained request suffix fits the six-row target. Preserve the
    /// old drafter window and use its distributed six-row conditioning path.
    /// The target has committed every supplied row before these taps are read.
    pub(crate) fn resume_after_small_extend(
        &mut self,target:&Target,previous:u64,ids:&[u32],comm:&Roce,stream:&Stream,
    )->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_RESIDENT_SMALL_APPEND").as_deref()==Ok("1")
            && !self.poisoned && (1..=6).contains(&ids.len()) && target.physical_rows()==6
            && target.profile_is_current() && self.draft.position()?==previous
            && previous.checked_add(ids.len()as u64)==Some(target.position()),
            "DSpark retained small EXTEND ownership/frontier");
        self.poisoned=true;
        self.features.run(target,stream)?;
        let (values,live)=self.features.values()?;
        ensure!(live==ids.len(),"DSpark retained small EXTEND feature rows");
        self.draft.inject_distributed(values,live,live,previous,comm,stream)?;
        ensure!(self.draft.position()?==target.position(),"DSpark small EXTEND commit frontier");
        if let Some(history)=&mut self.history_lookup {
            history.commit(previous,ids)?;
            ensure!(history.position()==target.position(),"lookup small EXTEND commit frontier");
        }
        self.poisoned=false;
        Ok(())
    }
    /// A complete7..16token request suffix was committed by the M16 target.
    /// Only drafter feature conditioning is split; target decoder visibility
    /// remains one request window. Reuse the qualified long-history buffers.
    pub(crate) fn resume_after_medium_extend(
        &mut self,target:&Target,buffers:&mut LongHistoryBuffers,previous:u64,
        ids:&[u32],comm:&Roce,stream:&Stream,
    )->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_RESIDENT_MEDIUM_APPEND").as_deref()==Ok("1")
            && !self.poisoned && ((target.physical_rows()==16&&(7..=16).contains(&ids.len()))
                ||(target.physical_rows()==32&&(17..=32).contains(&ids.len())
                    &&crate::config::append32_workspace_enabled()))
            && buffers.rows==target.physical_rows() && target.profile_is_current() && self.draft.position()?==previous
            && previous.checked_add(ids.len()as u64)==Some(target.position()),
            "DSpark retained medium EXTEND ownership/frontier");
        self.poisoned=true;
        buffers.features.run(target,stream)?;
        let(features,live)=buffers.features.values()?;
        ensure!(live==ids.len(),"DSpark retained medium EXTEND feature rows");
        for offset in (0..live).step_by(6) {
            let rows=(live-offset).min(6);
            unsafe {
                self.cuda.zero_device(buffers.chunk.ptr(),buffers.chunk.bytes(),stream)?;
                self.cuda.copy_device(buffers.chunk.ptr(),features.ptr()+(offset*15360*2)as u64,
                    rows*15360*2,stream)?;
            }
            self.draft.inject_distributed(&buffers.chunk,rows,rows,previous+offset as u64,comm,stream)?;
        }
        ensure!(self.draft.position()?==target.position(),"DSpark medium EXTEND commit frontier");
        if let Some(history)=&mut self.history_lookup {
            history.commit(previous,ids)?;
            ensure!(history.position()==target.position(),"lookup medium EXTEND commit frontier");
        }
        self.poisoned=false;
        Ok(())
    }
    /// A copy of the drafter feature rows of the last prefill's 128-row tail (positions [position-128, position)),
    /// kept beside a near-end encoder image (CANADA_SCRATCH_ENCODER_SHORT_SUFFIX=2).
    pub(crate) fn copy_prefill_tail_features(&self, stream: &Stream) -> Result<crate::cuda::Buffer<half::bf16>> {
        let (values, live) = self.prefill_features.values()?;
        ensure!(!self.poisoned && live == 128 && values.len() == 128 * 15360, "prefill tail feature rows");
        let copy = self.cuda.allocate::<half::bf16>(128 * 15360)?;
        unsafe { self.cuda.copy_device(copy.ptr(), values.ptr(), 128 * 15360 * 2, stream) }?;
        Ok(copy)
    }
    /// After the qualified short EXTEND that followed an encoder-image restore (no drafter state survives a
    /// restore): seed the drafter window from the image's saved tail features for the rows before the restore
    /// point and this EXTEND's own features for its rows. The tail's last row (the restore point's old token)
    /// is replaced by the EXTEND's first row. Drafts only; the target is unaffected.
    pub(crate) fn reset_from_extend_with_tail(&mut self, target: &Target, tail: &crate::cuda::Buffer<half::bf16>,
        tail_end: u64, stream: &Stream) -> Result<()> {
        let position = target.position();
        ensure!(!self.poisoned && target.physical_rows() == 128 && target.profile_is_current()
            && tail.len() == 128 * 15360 && tail_end >= 128 && tail_end <= position && position < tail_end + 127,
            "DSpark EXTEND-after-restore window");
        self.poisoned = true;
        self.history_lookup = None;
        self.draft.reset(stream)?;
        self.prefill_features.run(target, stream)?;
        let (values, live) = self.prefill_features.values()?;
        ensure!(live < 128 && tail_end == position - live as u64 + 1, "DSpark EXTEND feature rows");
        let old = 128 - live;
        let first = (position - tail_end) as usize;
        if self.extend_window.is_none() { self.extend_window = Some(self.cuda.allocate::<half::bf16>(128 * 15360)?); }
        let window = self.extend_window.as_ref().unwrap();
        unsafe {
            self.cuda.copy_device(window.ptr(), tail.ptr() + (first * 15360 * 2) as u64, old * 15360 * 2, stream)?;
            self.cuda.copy_device(window.ptr() + (old * 15360 * 2) as u64, values.ptr(), live * 15360 * 2, stream)?;
        }
        self.draft.inject_prefill_tail(window, 128, position, stream)?;
        self.poisoned = false;
        Ok(())
    }
    pub(crate) fn reset_from_prefill(
        &mut self,
        target: &Target,
        cuda: &Cuda,
        core: &[u8],
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            !self.poisoned
                && target.position() > 0
                && target.physical_rows() == 128
                && target.profile_is_current(),
            "DSpark reset prefix owner"
        );
        self.poisoned = true;
        self.history_lookup = None;
        self.draft.reset(stream)?;
        // Retain stable feature storage across requests instead of allocating
        // another CUDA module and buffer on every fresh request.
        let _=(cuda,core);
        self.prefill_features.run(target, stream)?;
        let (values, live) = self.prefill_features.values()?;
        ensure!(
            live == target.position().min(128) as usize,
            "DSpark reset features"
        );
        if draft_seed_log() {
            stream.synchronize()?;
            let raw = unsafe { cuda.read_device(values.ptr(), live * 15360 * 2) }?;
            let value = |b: &[u8]| f64::from(half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32());
            let rows: Vec<String> = raw.chunks_exact(15360 * 2).map(|row| {
                let norm = row.chunks_exact(2).map(|b| value(b) * value(b)).sum::<f64>().sqrt();
                let head = |tap: usize| row[tap * 10240..tap * 10240 + 128].chunks_exact(2).map(value).sum::<f64>();
                format!("{norm:.2}/{:.3}/{:.3}/{:.3}", head(0), head(1), head(2))
            }).collect();
            eprintln!("draftseed end={} live={live} sha={:x} rows={}", target.position(), <sha2::Sha256 as sha2::Digest>::digest(&raw), rows.join(","));
        }
        self.draft
            .inject_prefill_tail(values, live, target.position(), stream)?;
        self.poisoned = false;
        Ok(())
    }
}

#[cfg(feature = "serving")]
impl GreedySession {
    /// Discard only the unpublished public startup trial. Stable allocations
    /// and graph captures survive; the first client prefill reconditions them.
    pub(crate) fn clear_startup_state(&mut self, stream: &Stream) -> Result<bool> {
        // Worker::warm_before_ready owns admission and the unpublished-state
        // checks. Requiring an explicit environment override here would reject
        // its default-enabled path after the public warmup had already run.
        ensure!(!self.poisoned, "startup reset requires a healthy session");
        self.poisoned=true;
        self.draft.reset(stream)?;
        self.history_lookup=None;
        self.draft_warmed=false;
        self.target_seed=None;
        self.stop_token=None;
        stream.synchronize()?;
        ensure!(self.draft.position()?==0,"startup draft reset retained a frontier");
        self.poisoned=false;
        Ok(true)
    }
}

impl GreedySession {
    /// Batched serving: this drafter takes region 0 as its conversation leaves for a serving region. Region 0's
    /// target is reset there, and its drafter starts empty beside it: a spare may still hold the conversation
    /// it last served, while a session image (a paused or cached conversation) restores only where drafter and
    /// target stand at the same frontier. Allocations and captured graphs stay.
    pub(crate) fn clear_for_region0(&mut self, stream: &Stream) -> Result<()> {
        ensure!(!self.poisoned, "a drafter taking region 0 must be healthy");
        self.poisoned=true;
        self.draft.reset(stream)?;
        self.history_lookup=None;
        self.draft_warmed=false;
        self.target_seed=None;
        self.stop_token=None;
        stream.synchronize()?;
        ensure!(self.draft.position()?==0,"region 0's drafter kept a frontier");
        self.poisoned=false;
        Ok(())
    }
}

/// Input commit count when emitting a stop token. predictions[i] is the token
/// after input row i. Keep that stop as the one pending emitted anchor, so no
/// model state after it is published, even when more drafts matched the target.
fn stop_frontier(predictions: &[u32], accepted: usize, stop: Option<u32>) -> usize {
    stop.and_then(|id| predictions[..accepted].iter().position(|&t| t == id))
        .map_or(accepted, |i| i + 1)
}

#[cfg(test)]
mod stop_tests {
    #[test]
    fn eos_truncates_committed_inputs_before_the_pending_anchor() {
        for end in 1..=6 {
            for eos in 0..6 {
                let mut predictions = [7; 6]; predictions[eos] = 1;
                assert_eq!(super::stop_frontier(&predictions,end,Some(1)),end.min(eos+1));
                assert_eq!(super::stop_frontier(&predictions,end,None),end);
            }
        }
    }
}

/// Native greedy reference: identical target verify profile with one accepted
/// input each time, compared to six-row speculative evaluation. This is a
/// correctness/control benchmark, not a qualified SGLang speed comparison.
#[allow(clippy::too_many_arguments)]
pub(crate) fn probe(
    cuda: &Cuda,
    core: &[u8],
    stream: &Stream,
    cp: &Checkpoint,
    plan: &RankPlan,
    large: &mut Target,
    verify: &mut Target,
    comm: &Roce,
    prefix: &[u32],
    staged: &Path,
    kernels: &Path,
) -> Result<serde_json::Value> {
    ensure!(prefix.len() == 128, "DSpark integration prefix");
    let mut session = None;
    let mut lineage = 1000u64;
    let mut records = Vec::new();
    let mut control_all = None;
    for request in 0..5 {
        let mut control = Vec::new();
        let mut spec = Vec::new();
        let mut steps = Vec::new();
        let mut times = [0f64; 2];
        for mode in 0..2 {
            if mode != 0 || request != 0 {
                lineage += 1;
                verify.reset(lineage, comm, stream)?;
                large.handoff_from(verify)?;
            }
            large.append(prefix, comm, stream)?;
            let mut anchor = large.greedy_token(stream)?;
            if mode == 1 {
                if let Some(session) = session.as_mut() {
                    GreedySession::reset_from_prefill(session, large, cuda, core, stream)?;
                } else {
                    session = Some(GreedySession::from_prefill(
                        large, cuda, core, stream, cp, plan, staged, kernels,
                    )?);
                }
            }
            verify.handoff_from(large)?;
            let begin = Instant::now();
            if mode == 0 {
                for _ in 0..64 {
                    control.push(anchor);
                    verify.append(&[anchor], comm, stream)?;
                    anchor = verify.greedy_token(stream)?;
                }
            } else {
                while spec.len() < 64 {
                    let step = session.as_mut().unwrap().step(
                        verify,
                        anchor,
                        (64 - spec.len()).min(6),
                        comm,
                        stream,
                    )?;
                    anchor = step.next_anchor;
                    spec.extend_from_slice(&step.tokens);
                    steps.push(step);
                }
            }
            times[mode] = begin.elapsed().as_secs_f64();
        }
        ensure!(
            control == spec,
            "DSpark greedy sequence differs from one-input target control in request{request}"
        );
        if let Some(first) = &control_all {
            ensure!(
                first == &control,
                "Target greedy control changed across identical requests"
            );
        } else {
            control_all = Some(control.clone());
        }
        records.push(serde_json::json!({"request":request,"warmup":request<2,"tokens":spec,"control_seconds":times[0],
            "speculative_seconds":times[1],"speculative_iterations":steps.len(),"steps":steps,"all64_tokens_exact":true}));
    }
    Ok(
        serde_json::json!({"all_greedy_sequences_exact":true,"requests":records,"warmup_requests":2,"measured_requests":3,
        "tokens_per_request":64,"canonical_native_quantization":true,"full_engine_acceptance":false,
        "qualified_sglang_performance_compared":false,"stochastic_acceptance":false,
        "scope":"Native DSpark query/Markov greedy proposals, transactional target verify, accepted-only feature/KV commits and continuation. Control is the same native target Verify profile evaluated with one accepted input at a time; not an SGLang performance baseline."}),
    )
}

/// Full-vocabulary agreement of two FP32 logit rows from different prefill
/// row profiles. Diagnostic only: admissible rounding can move near-ties.
pub(crate) fn compare_logits(a: &[f32], b: &[f32]) -> serde_json::Value {
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    let top = |v: &[f32], k: usize| {
        let mut ids: Vec<usize> = (0..v.len()).collect();
        ids.sort_by(|&x, &y| v[y].total_cmp(&v[x]));
        ids.truncate(k);
        ids
    };
    let softmax = |v: &[f32]| {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let e: Vec<f64> = v.iter().map(|&x| (x as f64 - m).exp()).collect();
        let s: f64 = e.iter().sum();
        e.into_iter().map(|x| x / s).collect::<Vec<f64>>()
    };
    let (pa, pb) = (softmax(a), softmax(b));
    let tv = 0.5 * pa.iter().zip(&pb).map(|(x, y)| (x - y).abs()).sum::<f64>();
    let kl = pa
        .iter()
        .zip(&pb)
        .filter(|(x, _)| **x > 0.)
        .map(|(x, y)| x * (x / y.max(1e-300)).ln())
        .sum::<f64>();
    let diff: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .collect();
    let (ta, tb) = (top(a, 20), top(b, 20));
    serde_json::json!({"bitwise_equal":a.iter().zip(b).all(|(x,y)|x.to_bits()==y.to_bits()),
        "max_abs":diff.iter().copied().fold(0.,f64::max),
        "rms":(diff.iter().map(|d|d*d).sum::<f64>()/diff.len() as f64).sqrt(),
        "argmax_equal":argmax(a)==argmax(b),"top5_overlap":ta[..5].iter().filter(|i|tb[..5].contains(i)).count(),
        "top20_overlap":ta.iter().filter(|i|tb.contains(i)).count(),"total_variation":tv,"kl_128_to_wide":kl})
}

/// Per-position agreement of two equally long sets of full-vocabulary logits.
pub(crate) fn compare_positions(a: &[Vec<f32>], b: &[Vec<f32>]) -> serde_json::Value {
    let rows: Vec<_> = a.iter().zip(b).map(|(x, y)| compare_logits(x, y)).collect();
    let field = |k: &str| -> Vec<f64> { rows.iter().map(|r| r[k].as_f64().unwrap()).collect() };
    let (mut tv, kl) = (field("total_variation"), field("kl_128_to_wide"));
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (tv_mean, kl_mean) = (mean(&tv), mean(&kl));
    tv.sort_by(f64::total_cmp);
    serde_json::json!({"positions":rows.len(),"argmax_equal":rows.iter().filter(|r|r["argmax_equal"]==true).count(),
        "mean_total_variation":tv_mean,"median_total_variation":tv[tv.len()/2],"max_total_variation":tv[tv.len()-1],
        "mean_kl":kl_mean})
}
/// Teacher-forced next-token logits on the verifier for `blocks` six-token
/// blocks of a fixed continuation, committing each block after scoring it.
pub(crate) fn teacher_forced_logits(
    cuda: &Cuda,
    verify: &mut Target,
    continuation: &[u32],
    blocks: usize,
    comm: &Roce,
    stream: &Stream,
) -> Result<Vec<Vec<f32>>> {
    ensure!(
        continuation.len() >= blocks * 6,
        "teacher-forced continuation length"
    );
    let mut rows = Vec::new();
    for block in continuation.chunks_exact(6).take(blocks) {
        verify.begin_verification(block, comm, stream)?;
        let (logits, live) = verify.pending_logits()?;
        ensure!(live == 6, "teacher-forced verification rows");
        stream.synchronize()?;
        let raw = unsafe { cuda.read_device(logits.ptr(), 6 * 129280 * 4) }?;
        rows.extend(raw.chunks_exact(129280 * 4).map(|row| {
            row.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<f32>>()
        }));
        verify.finish_verification(6, comm, stream)?;
    }
    Ok(rows)
}

/// Admitted real coding prompt comparison. The interval is exactly511 further
/// target steps after the initial output, matching SGLang's512-output stream.
#[allow(clippy::too_many_arguments)]
pub(crate) fn probe_agentic(
    cuda: &Cuda,
    core: &[u8],
    stream: &Stream,
    cp: &Checkpoint,
    plan: &RankPlan,
    large: &mut Target,
    verify: &mut Target,
    comm: &Roce,
    prefix: &[u32],
    staged: &Path,
    kernels: &Path,
    wides: &mut [Target],
    sp_wides: &mut [Target],
) -> Result<serde_json::Value> {
    ensure!(
        crate::config::AGENTIC_PROMPT_TOKENS.contains(&prefix.len())
            && prefix.len() + 512 <= large.context_capacity(),
        "Agentic benchmark needs an admitted fixture and complete bounded output room"
    );
    let mut prefill_comparisons = Vec::new();
    let target_seed = std::env::var("CANADA_SCRATCH_TARGET_SEED")
        .ok()
        .map(|s| s.parse::<u64>())
        .transpose()?;
    // Last-row logits of each prefill path on the previous request: separates
    // a path's own repeat variation from 128-row versus wide differences.
    let mut previous_logits: [Option<Vec<f32>>; 3] = [None, None, None];
    // Optional bounded Nsight capture of one warm prefill path (1 wide, 2 SP).
    let profile_path = std::env::var("CANADA_SCRATCH_PROFILE_WIDE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let path_count = if sp_wides.is_empty() { 2 } else { 3 };
    // Request 0's native greedy output, teacher-forced after every later
    // prefill path: 48 next-token distributions per path and request.
    let mut continuation: Option<Vec<u32>> = None;
    let mut previous_forced: [Option<Vec<Vec<f32>>>; 3] = [None, None, None];
    let mut session = None;
    let mut lineage = 1000u64;
    let mut records = Vec::new();
    let memory_at_start = cuda.admission_memory()?;
    let profile_verify = std::env::var("CANADA_SCRATCH_PROFILE_VERIFY").as_deref() == Ok("1");
    let requests = if profile_verify { 1 } else { 4 };
    for request in 0..requests {
        let mut controls = Vec::new();
        let mut speculative = Vec::new();
        let mut steps = Vec::new();
        let mut seconds = [0.; 2];
        let mut prefills = [0.; 2];
        let mut states = Vec::new();
        let mut memory_after_modes = Vec::new();
        if !wides.is_empty() {
            // Interleaved fresh-prefill A/B on identical reset state: the
            // qualified 128-row chunks, then the wide encoder profile. Both
            // end in the same 128-row decoder tail and last-row logits.
            let mut paths = Vec::new();
            let mut forced = Vec::new();
            for path in 0..path_count {
                lineage += 1;
                verify.reset(lineage, comm, stream)?;
                large.handoff_from(verify)?;
                let profile = profile_path == Some(path) && path > 0 && request == 1;
                if profile {
                    stream.synchronize()?;
                    cuda.profiler_start()?;
                }
                let started = Instant::now();
                match path {
                    0 => large.prefill(prefix, comm, stream)?,
                    1 => large.prefill_wide(wides, prefix, comm, stream)?,
                    _ => large.prefill_wide(sp_wides, prefix, comm, stream)?,
                }
                if profile {
                    stream.synchronize()?;
                    cuda.profiler_stop()?;
                }
                let anchor = large.greedy_token(stream)?;
                let seconds = started.elapsed().as_secs_f64();
                paths.push((
                    seconds,
                    anchor,
                    large.last_prompt_logits()?,
                    large.encoder_tail_digest()?,
                    serde_json::to_value(large.append_metrics()?)?,
                ));
                // The verifier owns resets: return write authority after each path.
                verify.handoff_from(large)?;
                if let Some(c) = &continuation {
                    forced.push(teacher_forced_logits(cuda, verify, c, 8, comm, stream)?);
                }
            }
            let forced_comparison = (forced.len() == path_count).then(|| {
                let json = serde_json::json!({
                    "continuation":"request0 native greedy output, first 48 tokens",
                    "vs_prefill128":(1..path_count).map(|p| compare_positions(&forced[0], &forced[p])).collect::<Vec<_>>(),
                    "repeat_vs_previous_request":(0..path_count).map(|p| previous_forced[p]
                        .as_ref().map(|prev| compare_positions(prev, &forced[p]))).collect::<Vec<_>>()});
                for (p, rows) in forced.drain(..).enumerate() {
                    previous_forced[p] = Some(rows);
                }
                json
            });
            let comparison = compare_logits(&paths[0].2, &paths[1].2);
            let repeats: Vec<_> = (0..path_count)
                .map(|path| {
                    previous_logits[path]
                        .as_ref()
                        .map(|previous| compare_logits(previous, &paths[path].2))
                })
                .collect();
            for (path, entry) in paths.iter().enumerate() {
                previous_logits[path] = Some(entry.2.clone());
            }
            eprintln!(
                "agentic request{request} prefill128 {:.3}s wide {:.3}s{} anchors {} {} {comparison}",
                paths[0].0,
                paths[1].0,
                paths
                    .get(2)
                    .map_or(String::new(), |p| format!(" sp {:.3}s", p.0)),
                paths[0].1,
                paths[1].1
            );
            let mut entry = serde_json::json!({"request":request,"warmup":request==0,
                "prefill128_seconds":paths[0].0,"wide_prefill_seconds":paths[1].0,
                "prefill128_anchor":paths[0].1,"wide_anchor":paths[1].1,"anchors_equal":paths[0].1==paths[1].1,
                "last_row_logits":comparison,"prefill128_repeat_vs_previous_request":repeats[0],
                "wide_repeat_vs_previous_request":repeats[1],"encoder_tail_digest_equal":paths[0].3==paths[1].3,
                "wide_phases":paths[1].4});
            if let Some(f) = forced_comparison {
                entry["teacher_forced"] = f;
            }
            if let Some(sp) = paths.get(2) {
                entry["sequence_parallel"] = serde_json::json!({"prefill_seconds":sp.0,"anchor":sp.1,
                    "anchor_equal_prefill128":sp.1==paths[0].1,
                    "last_row_logits_vs_prefill128":compare_logits(&paths[0].2,&sp.2),
                    "last_row_logits_vs_wide":compare_logits(&paths[1].2,&sp.2),
                    "repeat_vs_previous_request":repeats[2],"phases":sp.4});
            }
            prefill_comparisons.push(entry);
        }
        for mode in 0..2 {
            // The prefill A/B leaves its final prompt state behind; start fresh.
            if mode != 0 || request != 0 || !wides.is_empty() {
                lineage += 1;
                verify.reset(lineage, comm, stream)?;
                large.handoff_from(verify)?;
            }
            let started = Instant::now();
            if !wides.is_empty() {
                large.prefill_wide(wides, prefix, comm, stream)?;
            } else {
                large.prefill(prefix, comm, stream)?;
            }
            let request_seed = target_seed.map(|s| s.wrapping_add(request as u64));
            let mut anchor = match request_seed {
                Some(seed) => large.sample_token(seed, stream)?,
                None => large.greedy_token(stream)?,
            };
            prefills[mode] = started.elapsed().as_secs_f64();
            eprintln!(
                "agentic request{request} mode{mode} prefill {}tokens {:.3}s anchor{anchor}",
                prefix.len(),
                prefills[mode]
            );
            if mode == 1 {
                if let Some(s) = session.as_mut() {
                    GreedySession::reset_from_prefill(s, large, cuda, core, stream)?;
                } else {
                    session = Some(GreedySession::from_prefill(
                        large, cuda, core, stream, cp, plan, staged, kernels,
                    )?);
                }
            }
            if mode == 1 {
                session.as_mut().unwrap().target_seed = request_seed;
            }
            verify.handoff_from(large)?;
            let mut generated = vec![anchor];
            let begin = Instant::now();
            if mode == 0 {
                while generated.len() < 512 {
                    verify.append(&[anchor], comm, stream)?;
                    anchor = match request_seed {
                        Some(seed) => verify.sample_token(seed, stream)?,
                        None => verify.greedy_token(stream)?,
                    };
                    generated.push(anchor);
                }
            } else {
                while generated.len() < 512 {
                    let step = session.as_mut().unwrap().step(
                        verify,
                        anchor,
                        (512 - generated.len()).min(6),
                        comm,
                        stream,
                    )?;
                    generated.extend_from_slice(&step.tokens[1..]);
                    generated.push(step.next_anchor);
                    anchor = step.next_anchor;
                    steps.push(step);
                }
            }
            seconds[mode] = begin.elapsed().as_secs_f64();
            eprintln!(
                "agentic request{request} mode{mode} 512outputs {:.3}s {:.3}tok/s",
                seconds[mode],
                511. / seconds[mode]
            );
            ensure!(
                verify.position() == prefix.len() as u64 + 511,
                "Agentic output/cache frontier"
            );
            states.push(verify.encoder_tail_digest()?);
            memory_after_modes.push(cuda.admission_memory()?);
            if mode == 0 {
                controls = generated;
            } else {
                speculative = generated;
            }
        }
        // Different batch shapes can round differently, including at greedy
        // near-ties. Preserve both trajectories and timings rather than turn
        // sequence identity into a universal model-quality requirement. State
        // causality/commit invariants remain enforced inside each execution.
        if continuation.is_none() {
            continuation = Some(controls.clone());
        }
        let outputs_exact = controls == speculative;
        let first_difference = controls.iter().zip(&speculative).position(|(a, b)| a != b);
        ensure!(
            states.iter().all(Option::is_some),
            "accepted encoder tail absent"
        );
        let tails_exact = states[0] == states[1];
        eprintln!(
            "agentic request{request} output_exact{outputs_exact} first_difference{first_difference:?} encoder_tail_exact{tails_exact}"
        );
        records.push(serde_json::json!({"request":request,"warmup":request==0,"output_ids":speculative,
            "control_output_ids":controls,"first_difference_vs_native_control":first_difference,
            "all512_output_ids_match_native_control":outputs_exact,"accepted_encoder_tail_exact":tails_exact,
            "encoder_tail_same_input_history":outputs_exact,"encoder_tail":states[1],"control_encoder_tail":states[0],
            "control_decode_seconds":seconds[0],"speculative_decode_seconds":seconds[1],
            "control_decode_tokens_per_second":511./seconds[0],"speculative_decode_tokens_per_second":511./seconds[1],
            "control_prefill_seconds":prefills[0],"speculative_prefill_seconds":prefills[1],"memory_after_modes":memory_after_modes,"steps":steps}));
    }
    ensure!(
        !profile_verify || session.as_ref().is_some_and(|s| s.profiled_verify),
        "requested verification profile never reached a warm six-row step"
    );
    Ok(
        serde_json::json!({"prompt_tokens":prefix.len(),"context_capacity":large.context_capacity(),"memory_at_start":memory_at_start,"output_tokens":512,"decode_interval_tokens":511,"records":records,
        "warmup_requests":1,"measured_requests":requests-1,"external_verify_profile":profile_verify,
        "ffn_execution":if std::env::var("CANADA_SCRATCH_FFN_OVERLAP").as_deref()==Ok("1") {"shared_and_routed_overlap_with_event_join"}else{"sequential_shared_and_routed"},
        "l2_prefetch":crate::l2_prefetch::enabled(),
        "fp8_projection_backend":if crate::fp8_linear::pdl_enabled(){"target_small_qb_flashinfer_static_pdl"}else{"b12x_words"},
        "mhc_execution":if crate::mhc::overlap_enabled(){"deferred_independent_stats_stream"}else{"sequential_stats"},
        "swa_storage":large.swa_storage_report()?,
        "verification_schedule":if std::env::var("CANADA_SCRATCH_STOCHASTIC_PREFIX_SCHEDULE").as_deref()==Ok("1") {"stochastic_prefix_stopping_time_confidence_threshold_0.1"}else if std::env::var("CANADA_SCRATCH_STOCHASTIC_DRAFT").as_deref()==Ok("1") {"fixed6_stochastic_proposals"}else if std::env::var("CANADA_SCRATCH_COST_AWARE_VERIFY").as_deref()==Ok("1") {"confidence_expected_inputs_per_measured_complete_step_cost"}else{"confidence_cumulative_threshold_0.1"},
        "prefill_mode":if std::env::var("CANADA_SCRATCH_CANONICAL_PREFILL").as_deref()==Ok("1") {"canonical_all_layers_all_prompt_tokens"}else{"qualified_recipe_decoder_tail_replay"},
        "nccl_proto":std::env::var("NCCL_PROTO").ok(),
        "prefill_encoder_rows":wides.iter().map(|w| w.physical_rows()).collect::<Vec<_>>(),
        "wide_profile":wides.iter().chain(sp_wides.iter()).filter_map(|w| w.wide_report()).collect::<Vec<_>>(),
        "engram_row_cache":large.engram_cache_stats(),
        "prefill_comparisons":prefill_comparisons,
        "profiled_verify_steps":records.iter().flat_map(|r|r["steps"].as_array().unwrap()).filter(|s|s["profiled_verify"]==true).count(),
        "all_native_control_and_drafted_outputs_exact":records.iter().all(|r|r["all512_output_ids_match_native_control"]==true),
        "full_engine_acceptance":false,"stochastic_acceptance":target_seed.is_some(),"qualified_sglang_performance_compared":false,
        "sampling":target_seed.map(|seed|serde_json::json!({"target_temperature":1.0,"target_top_p":1.0,"seed":seed,
            "rng":"SHA256 counter PRF v1, absolute predicted position, independent target/draft/accept/residual domains",
            "proposal":if std::env::var("CANADA_SCRATCH_STOCHASTIC_DRAFT").as_deref()==Ok("1") {"native DSpark T1/topP1 Markov autoregressive categorical"}else{"deterministic native DSpark Markov argmax; q=point mass"},
            "acceptance":if std::env::var("CANADA_SCRATCH_STOCHASTIC_DRAFT").as_deref()==Ok("1") {"min(1,p(y)/q(y)); on rejection sample normalized max(p-q,0); depth fixed or a recorded stopping time in proposal prefixes"}else{"independent target categorical draws; accept matching draft prefix, emit first mismatch/bonus"},
            "proof":"q(x)min(1,p(x)/q(x)) + max(p(x)-q(x),0) = p(x). Deterministic draft matching is the point-mass special case. Draw domains are independent, depth does not select on future random proposals, and the final bonus is sampled from p."})),
        "scope":"Exact admitted frozen real coding prompt and reported bounded context capacity, bounded128row chunks using the reported prefill mode, target-only versus native DSpark proposals, with target sampling specified separately. Both trajectories and tail digests retained; output equality is diagnostic, not a quality gate. Tail equality across different input histories is not expected. No HTTP or broader-context/quality/reliability acceptance."}),
    )
}
