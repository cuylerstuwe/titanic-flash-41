//! Raw full-target integration capture from token IDs only. This probe cannot
//! award SGLang parity: its outputs must be checked against the qualified
//! whole-model oracle and its context/row profile is explicitly limited.
use crate::{
    checkpoint::Checkpoint, config::ModelConfig, cuda::Cuda, hash_history::HashMetadata,
    layout::RankPlan, roce::Roce, target::Target,
};
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Deserialize)]
struct Peer {
    config: PathBuf,
    checkpoint: PathBuf,
    rank: usize,
    core: PathBuf,
    kernels: PathBuf,
    stages: Vec<PathBuf>,
    io_stage: PathBuf,
    backing: PathBuf,
    hash_metadata: PathBuf,
    endpoint: String,
    ticket: PathBuf,
    output: PathBuf,
    batches: Vec<Vec<u32>>,
    #[serde(default)]
    graph_replay: bool,
    #[serde(default)]
    trace_layers: bool,
    #[serde(default)]
    prefill_nccl: bool,
    #[serde(default)]
    prefill_last_head: bool,
    #[serde(default)]
    prefill_nccl_graph_experiment: bool,
    #[serde(default)]
    reset_between_batches: bool,
    #[serde(default)]
    greedy_tokens: bool,
    #[serde(default)]
    trace_all_steps: bool,
    #[serde(default)]
    trace_step: Option<usize>,
    #[serde(default = "default_rows")]
    rows: usize,
    #[serde(default = "default_context")]
    context: usize,
    #[serde(default)]
    profile_handoff: bool,
    #[serde(default)]
    decoder_tail_replay: bool,
    #[serde(default)]
    cached_extend_floor: bool,
    #[serde(default)]
    decode_steps: Vec<usize>,
    #[serde(default)]
    profile_decode_kernels: bool,
    #[serde(default)]
    performance_tokens: usize,
    #[serde(default)]
    verification_transactions: bool,
    #[serde(default)]
    dspark_feature_probe: bool,
    #[serde(default)]
    dspark_generation: bool,
    #[serde(default)]
    dspark_agentic: bool,
    /// Fresh prompts run layers0..20 in one 4096-row encoder forward per
    /// chunk (plus the 128-row decoder tail), with an interleaved A/B against
    /// the 128-row chunked prefill on every request.
    #[serde(default)]
    wide_prefill: bool,
    /// Wide encoder widths to build (default: the widest admitted width).
    #[serde(default)]
    wide_widths: Vec<usize>,
    /// Also build sequence-parallel-mHC wide profiles of the same widths and
    /// add them as a third interleaved path of the in-process prefill A/B.
    #[serde(default)]
    wide_sequence_parallel: bool,
    /// Build only sequence-parallel wide profiles (the production set) and
    /// use them as the wide path of the 128-row versus wide A/B.
    #[serde(default)]
    wide_sequence_parallel_only: bool,
    #[serde(default)]
    agentic_parity_debug: bool,
    #[serde(default)]
    fixed_verify_fixture: Option<PathBuf>,
    /// Scheduling flags (CANADA_SCRATCH_*) enabled only while building a second
    /// Verify6 profile that the fixed-prefix probe interleaves with the control.
    #[serde(default)]
    fixed_verify_candidate: Vec<String>,
    #[serde(default)]
    retained_prefill: bool,
    #[serde(default)]
    oracle_logits: bool,
    #[serde(default)]
    natural_generation: bool,
    #[serde(default)]
    retained_generation: bool,
    #[serde(default)]
    dspark_staged: Option<PathBuf>,
    #[serde(default)]
    trace_decoder_step: Option<usize>,
    #[serde(default)]
    reset_steps: Vec<usize>,
    #[serde(default)]
    source_phase: crate::compressed_source::Phase,
    #[serde(default)]
    eager_from_step: Option<usize>,
}
fn default_context() -> usize {
    128
}
fn default_rows() -> usize {
    6
}
/// Construction-time switches a fixed-verify candidate may enable. Sorting
/// selected indices preserves membership but fixes the attention reduction order.
const SCHEDULE_FLAGS: &[&str] = &[
    "CANADA_SCRATCH_SORT_SELECTED",
    "CANADA_SCRATCH_MHC_OVERLAP",
    "CANADA_SCRATCH_FFN_OVERLAP",
    "CANADA_SCRATCH_L2_PREFETCH",
    "CANADA_SCRATCH_FP8_PDL",
];
/// Field destruction order is a correctness contract: NCCL retains graph
/// callbacks and its abort waits for their release. Drop the target's captured
/// executable before its communicator on success and on every error path.
struct Execution {
    target: Target,
    small: Option<Target>,
    decode: Option<Target>,
    verify: Option<Target>,
    candidate: Option<Target>,
    wides: Vec<Target>,
    sp_wides: Vec<Target>,
    comm: Roce,
}
fn save(root: &Path, name: &str, bytes: &[u8]) -> Result<Value> {
    std::fs::write(root.join(format!("{name}.bin")), bytes)?;
    Ok(json!({"name":name,"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes))}))
}
fn feature_bytes(
    cuda: &Cuda,
    target: &Target,
    features: &mut Option<crate::dspark_features::Features>,
    stream: &crate::cuda::Stream,
) -> Result<Option<Vec<u8>>> {
    if let Some(features) = features {
        features.run(target, stream)?;
        stream.synchronize()?;
        let (values, live) = features.values()?;
        Ok(Some(unsafe {
            cuda.read_device(values.ptr(), live * 15360 * 2)
        }?))
    } else {
        Ok(None)
    }
}
fn transaction_digest(
    cuda: &Cuda,
    target: &Target,
    live: usize,
    features: &mut Option<crate::dspark_features::Features>,
    stream: &crate::cuda::Stream,
) -> Result<Value> {
    let logits = target.logits()?;
    let bytes = live * 129280 * 4;
    ensure!(
        live > 0 && live <= 6 && bytes <= logits.bytes(),
        "transaction logits capture bounds"
    );
    let raw = unsafe { cuda.read_device(logits.ptr(), bytes) }?;
    let feature =
        feature_bytes(cuda, target, features, stream)?.map(|v| format!("{:x}", Sha256::digest(v)));
    Ok(
        json!({"position":target.position(),"valid_outputs":target.valid_output_digests()?,
        "all_live_logits_sha256":format!("{:x}",Sha256::digest(raw)),"carry":target.carry_digests()?,
        "target_feature_sha256":feature}),
    )
}
fn verification_transactions(
    cuda: &Cuda,
    large: &mut Target,
    verify: &mut Target,
    comm: &Roce,
    stream: &crate::cuda::Stream,
    prefix: &[u32],
    output: &Path,
    rank: usize,
    load_seconds: f64,
    core: &[u8],
    capture: bool,
) -> Result<Value> {
    ensure!(prefix.len() == 128, "verification test prefix");
    let mut large_features = if capture {
        Some(crate::dspark_features::Features::new(cuda, core, 128)?)
    } else {
        None
    };
    let mut features = if capture {
        Some(crate::dspark_features::Features::new(cuda, core, 6)?)
    } else {
        None
    };
    let mut records = Vec::new();
    let mut sequence = 0;
    let mut prefill_feature = None;
    let mut graph_variants = Vec::new();
    for frontier in 0..if capture { 2 } else { 1 } {
        let base = 128 + frontier as u64;
        for accepted in 0..=6 {
            let mut speculative = None;
            for path in 0..2 {
                if sequence > 0 {
                    verify.reset(sequence as u64 + 1, comm, stream)?;
                    large.handoff_from(verify)?;
                }
                large.append(prefix, comm, stream)?;
                ensure!(
                    large.greedy_token(stream)? == 271,
                    "verification anchor changed"
                );
                if sequence == 0 {
                    if let Some(raw) = feature_bytes(cuda, large, &mut large_features, stream)? {
                        std::fs::write(output.join("prefill-features.bin"), &raw)?;
                        prefill_feature = Some(format!("{:x}", Sha256::digest(raw)));
                    }
                }
                verify.handoff_from(large)?;
                if frontier == 1 {
                    verify.append(&[271], comm, stream)?;
                }
                let proposals = if frontier == 0 {
                    [271u32, 372, 223, 3442, 9090, 201]
                } else {
                    [372u32, 223, 3442, 9090, 201, 19]
                };
                if frontier == 1 {
                    ensure!(
                        verify.greedy_token(stream)? == proposals[0],
                        "odd frontier anchor changed"
                    );
                }
                let incoming_carry = verify.carry_digests()?;
                let mut blocked = false;
                let mut restored = 0;
                let mut begin_ms = None;
                let mut finish_ms = None;
                let mut predictions = Vec::new();
                let mut pending_features = None;
                if path == 0 {
                    let t = Instant::now();
                    verify.begin_verification(&proposals, comm, stream)?;
                    begin_ms = Some(t.elapsed().as_secs_f64() * 1000.);
                    ensure!(
                        verify.position() == base
                            && !verify.profile_is_current()
                            && !large.profile_is_current(),
                        "pending verification published history"
                    );
                    ensure!(
                        verify.pending_logits()?.1 == 6,
                        "verification row visibility"
                    );
                    predictions = verify.verification_greedy(stream)?;
                    pending_features = feature_bytes(cuda, verify, &mut features, stream)?;
                    blocked = verify.logits().is_err()
                        && large.logits().is_err()
                        && verify.append(&[6000], comm, stream).is_err()
                        && verify.begin_verification(&proposals, comm, stream).is_err()
                        && large.handoff_from(verify).is_err()
                        && verify.reset(sequence as u64 + 100, comm, stream).is_err()
                        && verify.finish_verification(7, comm, stream).is_err();
                    ensure!(
                        blocked && verify.pending_logits()?.1 == 6,
                        "pending or invalid-commit guard damaged state"
                    );
                    let t = Instant::now();
                    restored = verify.finish_verification(accepted, comm, stream)?;
                    finish_ms = Some(t.elapsed().as_secs_f64() * 1000.);
                    ensure!(
                        verify.pending_logits().is_err()
                            && verify.finish_verification(accepted, comm, stream).is_err(),
                        "verification could commit twice"
                    );
                    if accepted > 0 {
                        ensure!(
                            verify.greedy_token(stream)? == predictions[accepted - 1],
                            "accepted prefix did not expose its own next-token result"
                        );
                    }
                } else if accepted > 0 {
                    verify.append(&proposals[..accepted], comm, stream)?;
                }
                ensure!(
                    verify.position() == base + accepted as u64 && verify.profile_is_current(),
                    "verification committed wrong frontier"
                );
                let accepted_features = if accepted > 0 {
                    let raw = feature_bytes(cuda, verify, &mut features, stream)?;
                    if let (Some(pending), Some(actual)) = (&pending_features, &raw) {
                        ensure!(
                            actual == &pending[..accepted * 15360 * 2],
                            "accepted DSpark features changed"
                        );
                    }
                    raw.map(|v| format!("{:x}", Sha256::digest(v)))
                } else {
                    ensure!(
                        verify.carry_digests()? == incoming_carry,
                        "abort did not restore incoming carry"
                    );
                    // Only an actual aborted verification invalidates its
                    // provisional outputs. A clean zero-length path may still
                    // expose the previous, accepted seed token at odd frontier.
                    if path == 0
                        && let Some(features) = &mut features
                    {
                        ensure!(
                            features.run(verify, stream).is_err() && features.values().is_err(),
                            "aborted verification published stale target features"
                        );
                    }
                    None
                };
                // One token leaves most rejected future entries present; they must
                // remain invisible before a larger continuation replaces them.
                verify.append(&[6000 + accepted as u32], comm, stream)?;
                let one = transaction_digest(cuda, verify, 1, &mut features, stream)?;
                let suffix: Vec<_> = (0..6).map(|i| 7000 + accepted as u32 * 11 + i).collect();
                verify.append(&suffix, comm, stream)?;
                let six = transaction_digest(cuda, verify, 6, &mut features, stream)?;
                let equal = speculative
                    .as_ref()
                    .map(|prior| prior == &(one.clone(), six.clone(), accepted_features.clone()));
                records.push(json!({"frontier":base,"accepted":accepted,"path":if path==0 {"speculative"}else{"clean-prefix"},
                "position":verify.position(),"one_token_continuation":one,"six_token_continuation":six,
                "accepted_features":accepted_features,"predictions":predictions,"busy_operations_blocked":blocked,
                "restored_carry_bytes":restored,"begin_ms":begin_ms,"finish_ms":finish_ms,"matches_speculative":equal}));
                std::fs::write(
                    output.join("transaction-progress.json"),
                    serde_json::to_vec_pretty(&records)?,
                )?;
                if path == 0 {
                    speculative = Some((one, six, accepted_features));
                } else {
                    ensure!(
                        equal == Some(true),
                        "speculative prefix differs from clean accepted-prefix continuation"
                    );
                }
                sequence += 1;
            }
        }
        let (live_counts, captures) = verify.capture_statistics();
        if !live_counts.is_empty() {
            ensure!(
                live_counts == [1, 2, 3, 4, 5, 6] && captures == 6,
                "target graph variants were recaptured or lost across live-count changes/resets"
            );
        }
        graph_variants
            .push(json!({"frontier":base,"live_counts":live_counts,"total_captures":captures}));
    }
    Ok(
        json!({"rank":rank,"load_seconds":load_seconds,"records":records,
        "all_cutoffs_0_through_6_exact":true,"both_frontier_parities":capture,"target_features_enabled":capture,
        "prefill_feature_sha256":prefill_feature,"all_busy_and_double_commit_guards_pass":true,
        "graph_variants":graph_variants,
        "full_engine_acceptance":false,"draft_model_implemented":false,
        "scope":"Actual128prefix (also129 with feature probe) ->speculative6 ->commit0..6 ->one-token andsix-token continuation, bytewise versus clean accepted-prefix runs. All live logits, valid hidden/pre, C2 carry and enabled DSpark target features. Rejected future cache entries tested while still present. Own GPU values only. Does not qualify Verify6 versus SGLang, draft proposal network, stochastic acceptance, session branching or full engine parity."}),
    )
}
pub fn run(path: &Path) -> Result<Value> {
    let probe_load_started=std::time::Instant::now();
    let p: Peer = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(
        !p.dspark_generation || (p.verification_transactions && p.dspark_staged.is_some()),
        "DSpark generation requires controlled verifier and MTP atlas"
    );
    ensure!(
        !p.dspark_agentic
            || (p.dspark_generation
                && crate::config::agentic_context(p.context, p.wide_prefill)
                && p.rows == 128
                && p.batches.len() == 1
                && crate::config::AGENTIC_PROMPT_TOKENS.contains(&p.batches[0].len())
                && p.batches[0].len() + 512 <= p.context
                && p.prefill_last_head
                && !p.cached_extend_floor
                && !p.decoder_tail_replay
                && p.reset_steps.is_empty()
                && p.decode_steps.is_empty()),
        "Agentic DSpark probe requires an admitted exact fixture and bounded context"
    );
    ensure!(
        !p.dspark_feature_probe || p.verification_transactions,
        "DSpark feature integration probe needs verification transactions"
    );
    ensure!(
        !(p.graph_replay && p.prefill_nccl) || p.prefill_nccl_graph_experiment,
        "prefill NCCL capture requires explicit isolated experimental admission"
    );
    ensure!(
        !p.prefill_nccl_graph_experiment || (p.graph_replay && p.prefill_nccl),
        "prefill graph experiment requires both graph and qualified NCCL phase"
    );
    ensure!(
        !p.output.exists(),
        "immutable target capture already exists"
    );
    ensure!(
        p.rank < 4
            && [6, 128].contains(&p.rows)
            && (p.rows == 6
                || ((p.reset_between_batches || p.profile_handoff)
                    && p.source_phase == crate::compressed_source::Phase::Extend))
            && (1..=128).contains(&p.batches.len())
            && p.batches.iter().all(|b| ((1..=p.rows).contains(&b.len())
                && (p.rows == 6 || b.len() == 128 || (p.profile_handoff && b.len() <= 6))
                || p.dspark_agentic && crate::config::AGENTIC_PROMPT_TOKENS.contains(&b.len()))
                && b.iter().all(|&id| id < 129280))
            && ([128, 256, 512].contains(&p.context)
                || p.dspark_agentic && crate::config::agentic_context(p.context, p.wide_prefill))
            && (!p.profile_handoff
                || (p.rows == 128
                    && ([256, 512].contains(&p.context)
                        || p.dspark_agentic && crate::config::agentic_context(p.context, p.wide_prefill))))
            && p.reset_steps.iter().all(|&i| i > 0 && i < p.batches.len())
            && (p.profile_handoff
                || p.reset_between_batches
                || p.batches.iter().map(Vec::len).sum::<usize>() <= p.context),
        "bounded target integration batches"
    );
    ensure!(
        !p.decoder_tail_replay
            || (p.profile_handoff
                && p.rows == 128
                && p.context == 256
                && p.prefill_last_head
                && !p.trace_all_steps
                && p.eager_from_step.is_none()),
        "bounded tail replay probe profile"
    );
    ensure!(
        !p.cached_extend_floor
            || (p.profile_handoff
                && p.rows == 128
                && p.source_phase == crate::compressed_source::Phase::Extend
                && p.prefill_last_head
                && !p.decoder_tail_replay),
        "cached EXTEND floor requires an independent retained profile"
    );
    ensure!(
        p.decode_steps.is_empty()
            || (p.cached_extend_floor
                && p.greedy_tokens
                && p.decode_steps
                    .iter()
                    .all(|&i| i > 0 && i < p.batches.len() && p.batches[i].len() == 1)
                && p.decode_steps.windows(2).all(|v| v[0] < v[1])),
        "decode steps require ordered generated-token indices and cached EXTEND profile"
    );
    ensure!(
        p.trace_decoder_step.is_none_or(|i| p.decoder_tail_replay
            && i > 0
            && i < p.batches.len()
            && p.batches[i].len() <= 6
            && !p.trace_layers),
        "decoder trace requires one retained tail step"
    );
    ensure!(
        p.trace_step
            .is_none_or(|i| i < p.batches.len() && !p.decoder_tail_replay),
        "single-step trace requires an ordinary accepted append"
    );
    let cfg = ModelConfig::parse(&std::fs::read(&p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let plan = RankPlan::build(&cp, &cfg, p.rank, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let core_bytes = std::fs::read(&p.core)?;
    let core = cuda.module(&core_bytes)?;
    let load_start = Instant::now();
    let mut target = Target::load(
        &cuda,
        &core,
        &core_bytes,
        &stream,
        &cp,
        &plan,
        &p.stages,
        &p.io_stage,
        &p.kernels,
        &p.backing,
        HashMetadata::parse(&std::fs::read(&p.hash_metadata)?)?,
        1,
        p.source_phase,
        p.rows,
        p.context,
    )?;
    if p.graph_replay && p.performance_tokens == 0 && !p.verification_transactions {
        target.enable_repeatable_frontier_measurement()?;
    }
    target.set_graph_replay(p.graph_replay)?;
    target.set_prefill_last_head(p.prefill_last_head)?;
    if p.cached_extend_floor {
        target.enable_cached_extend_floor()?;
    }
    if p.decoder_tail_replay || p.dspark_agentic {
        target.enable_decoder_tail(&core_bytes)?;
    }
    ensure!(
        !p.wide_prefill || (p.dspark_agentic && p.rows == 128 && p.context >= 4096 && p.prefill_nccl),
        "wide prefill needs the admitted 4096-context agentic profile with NCCL prefill sums"
    );
    let widths = if p.wide_prefill && p.wide_widths.is_empty() {
        vec![crate::config::DEFAULT_WIDE_ROWS]
    } else {
        p.wide_widths.clone()
    };
    ensure!(
        widths.iter().all(|&w| crate::config::wide_rows(w) && w <= p.context)
            && (p.wide_prefill || widths.is_empty()),
        "wide prefill widths"
    );
    ensure!(
        (!p.wide_sequence_parallel && !p.wide_sequence_parallel_only)
            || (p.wide_prefill && !(p.wide_sequence_parallel && p.wide_sequence_parallel_only)),
        "sequence-parallel wide profiles extend the wide prefill A/B"
    );
    let (mut wides, mut sp_wides) = (Vec::new(), Vec::new());
    for sequence_parallel in [false, true] {
        if (sequence_parallel && !(p.wide_sequence_parallel || p.wide_sequence_parallel_only))
            || (!sequence_parallel && p.wide_sequence_parallel_only)
        {
            continue;
        }
        for t in target.fork_wide_bank(&core, &core_bytes, &stream, &p.kernels, &widths, sequence_parallel)? {
            let shared = target.shared_state_report(&t);
            ensure!(
                shared["all_weights_and_causal_storage_shared"] == true,
                "wide profile duplicated model/causal storage: {shared}"
            );
            eprintln!("rank{} wide prefill profile {:?}", p.rank, t.wide_report());
            if sequence_parallel && !p.wide_sequence_parallel_only { &mut sp_wides } else { &mut wides }.push(t);
        }
    }
    let mut small = if p.profile_handoff && !p.verification_transactions {
        let mut t = target.fork_profile(
            &core,
            &core_bytes,
            &stream,
            &p.kernels,
            6,
            crate::compressed_source::Phase::Extend,
        )?;
        if p.graph_replay && p.performance_tokens == 0 {
            t.enable_repeatable_frontier_measurement()?;
        }
        t.set_graph_replay(p.graph_replay)?;
        t.set_prefill_last_head(p.prefill_last_head)?;
        if p.cached_extend_floor {
            t.enable_cached_extend_floor()?;
        }
        Some(t)
    } else {
        None
    };
    let shared_state = small.as_ref().map(|t| target.shared_state_report(t));
    let mut decode = if p.decode_steps.is_empty() || p.verification_transactions {
        None
    } else {
        let mut t = target.fork_profile(
            &core,
            &core_bytes,
            &stream,
            &p.kernels,
            6,
            crate::compressed_source::Phase::Decode,
        )?;
        if p.graph_replay && p.performance_tokens == 0 {
            t.enable_repeatable_frontier_measurement()?;
        }
        t.set_graph_replay(p.graph_replay)?;
        t.set_prefill_last_head(p.prefill_last_head)?;
        Some(t)
    };
    let decode_shared_state = decode.as_ref().map(|t| target.shared_state_report(t));
    let mut verify = if p.verification_transactions {
        ensure!(
            p.profile_handoff
                && p.rows == 128
                && (p.context == 512 || p.dspark_agentic && crate::config::agentic_context(p.context, p.wide_prefill))
                && p.graph_replay
                && p.performance_tokens == 0
                && !p.decoder_tail_replay
                && !p.profile_decode_kernels,
            "bounded verification transaction probe"
        );
        let mut t = target.fork_profile(
            &core,
            &core_bytes,
            &stream,
            &p.kernels,
            6,
            crate::compressed_source::Phase::Verify,
        )?;
        t.set_graph_replay(true)?;
        Some(t)
    } else {
        None
    };
    let rapid_screen = std::env::var("CANADA_SCRATCH_RAPID_SCHEDULE_SCREEN").as_deref() == Ok("1");
    let generation_screen = std::env::var("CANADA_SCRATCH_GENERATION_SCHEDULE_AB").as_deref() == Ok("1");
    let compact_screen=std::env::var("CANADA_SCRATCH_COMPACT_INDEX_AB").as_deref()==Ok("1");
    let sparse_generation=std::env::var("CANADA_SCRATCH_SPARSE_GENERATION_AB").as_deref()==Ok("1");
    let schedule_screen = rapid_screen || generation_screen || sparse_generation;
    let schedule_ablation=std::env::var("CANADA_SCRATCH_SCHEDULE_ABLATION").as_deref()==Ok("1");
    let sort_screen = std::env::var("CANADA_SCRATCH_SORT_SELECTED_AB").as_deref() == Ok("1");
    ensure!(!sort_screen || (rapid_screen && !generation_screen), "sort AB requires rapid screen");
    let micro_screen = std::env::var("CANADA_SCRATCH_MOE_MICRO_AB").as_deref() == Ok("1");
    let live_moe_screen = std::env::var("CANADA_SCRATCH_MOE_LIVE_AB").as_deref() == Ok("1");
    let scale_screen = std::env::var("CANADA_SCRATCH_MOE_SCALE_AB").as_deref() == Ok("1");
    ensure!(!scale_screen || (rapid_screen && !generation_screen && !sort_screen && !micro_screen && !live_moe_screen),"scale MoE requires isolated screen");
    ensure!(!live_moe_screen || (rapid_screen && !generation_screen && !sort_screen && !micro_screen),
        "live MoE AB requires an isolated rapid screen");
    ensure!(!micro_screen || (rapid_screen && !generation_screen && !sort_screen),
        "micro AB requires an isolated rapid screen");
    let compute_screen = std::env::var("CANADA_SCRATCH_COMPUTE_AB").as_deref() == Ok("1");
    let narrow_only = std::env::var("CANADA_SCRATCH_NARROW_ONLY_SCREEN").as_deref() == Ok("1");
    let task_order_only = std::env::var("CANADA_SCRATCH_TASK_ORDER_ONLY_SCREEN").as_deref() == Ok("1");
    ensure!(!task_order_only || (compute_screen && !narrow_only), "task-order-only requires isolated compute comparison");
    ensure!(!narrow_only || compute_screen, "narrow-only screen requires compute comparison");
    let sparse_screen = sparse_generation || std::env::var("CANADA_SCRATCH_SPARSE_INDEX_AB").as_deref() == Ok("1");
    let fused_qkv_screen = std::env::var("CANADA_SCRATCH_FUSED_QKV_AB").as_deref() == Ok("1");
    let adaptive_sparse_screen=sparse_generation || std::env::var("CANADA_SCRATCH_ADAPTIVE_SPARSE_AB").as_deref()==Ok("1");
    ensure!(!sparse_generation || (!rapid_screen && !generation_screen && p.retained_generation
        && p.context==262144 && !p.retained_prefill),"sparse generation requires isolated long sampled screen");
    ensure!(!adaptive_sparse_screen || sparse_screen,"adaptive comparison requires sparse screen");
    ensure!(!fused_qkv_screen || (rapid_screen && !generation_screen && !sort_screen && !micro_screen
        && !live_moe_screen && !scale_screen && !compute_screen && !schedule_ablation && !sparse_screen
        && p.context==32768),"fused QKV comparison requires isolated 32K rapid screen");
    ensure!(!sparse_screen || ((rapid_screen || sparse_generation) && !generation_screen && !sort_screen && !micro_screen
        && !live_moe_screen && !scale_screen && !compute_screen && !schedule_ablation && p.context==262144),
        "sparse comparison requires isolated 256K-capacity rapid screen");
    ensure!(!compute_screen || (rapid_screen && !generation_screen && !sort_screen && !micro_screen
        && !live_moe_screen && !scale_screen), "compute comparison requires isolated rapid screen");
    ensure!(!schedule_ablation ||(rapid_screen && !generation_screen && !sort_screen && !micro_screen
        && !live_moe_screen && !scale_screen && !compute_screen && p.context==32768
        && p.retained_prefill && p.fixed_verify_candidate.is_empty()),"isolated schedule ablation");
    ensure!(!compact_screen || (rapid_screen && p.retained_prefill && p.context==1048576
        && !generation_screen && !sparse_screen && !fused_qkv_screen && !sort_screen
        && !micro_screen && !live_moe_screen && !scale_screen && !compute_screen && !schedule_ablation
        && p.fixed_verify_candidate.is_empty()),"compact index requires isolated 1M rapid screen");
    let candidate_flags = if schedule_ablation || compact_screen {Vec::new()} else if schedule_screen {
        ensure!(!(rapid_screen && generation_screen) && p.fixed_verify_candidate.is_empty()
            && (rapid_screen && p.retained_prefill || (generation_screen || sparse_generation) && p.retained_generation)
            && p.context == if sparse_screen {262144} else {32768}
            && std::env::var_os("CANADA_SCRATCH_ATTENTION_Q_OVERLAP").is_none(),
            "schedule screen needs an unmodified control and one admitted probe");
        let mut flags=vec!["CANADA_SCRATCH_ATTENTION_Q_OVERLAP".to_string()];
        if sparse_screen {flags.push("CANADA_SCRATCH_SPARSE_INDEX".to_string());}
        if adaptive_sparse_screen {flags.push("CANADA_SCRATCH_SPARSE_INDEX_ADAPTIVE".to_string());}
        if fused_qkv_screen {flags.push("CANADA_SCRATCH_FUSED_QKV_VERIFY".to_string());}
        if compute_screen {
            if !task_order_only { flags.push("CANADA_SCRATCH_FP8_NARROW".to_string()); }
            if !narrow_only { flags.push("CANADA_SCRATCH_MOE_TASK_ORDER".to_string()); }
        }
        if scale_screen {flags.push("CANADA_SCRATCH_MOE_SCALE_REUSE".to_string());}
        if micro_screen {flags.push("CANADA_SCRATCH_MOE_MICRO_FP32".to_string());}
        if live_moe_screen {flags.push("CANADA_SCRATCH_MOE_LIVE_CANDIDATE".to_string());}
        flags
    } else { p.fixed_verify_candidate.clone() };
    let mut candidate = if candidate_flags.is_empty() && !compact_screen {
        None
    } else {
        ensure!(
            (schedule_screen || p.fixed_verify_fixture.is_some())
                && verify.is_some()
                && candidate_flags.iter().all(|f| (SCHEDULE_FLAGS.contains(&f.as_str())
                    || (schedule_screen && f == "CANADA_SCRATCH_ATTENTION_Q_OVERLAP")
                    || (micro_screen && f == "CANADA_SCRATCH_MOE_MICRO_FP32")
                    || (live_moe_screen && f == "CANADA_SCRATCH_MOE_LIVE_CANDIDATE")
                    || (scale_screen && f == "CANADA_SCRATCH_MOE_SCALE_REUSE")
                    || (sparse_screen && f == "CANADA_SCRATCH_SPARSE_INDEX")
                    || (adaptive_sparse_screen && f == "CANADA_SCRATCH_SPARSE_INDEX_ADAPTIVE")
                    || (fused_qkv_screen && f == "CANADA_SCRATCH_FUSED_QKV_VERIFY")
                    || (compute_screen && ["CANADA_SCRATCH_FP8_NARROW", "CANADA_SCRATCH_MOE_TASK_ORDER"].contains(&f.as_str())))
                    && std::env::var_os(f).is_none()),
            "fixed-verify candidate needs unset known scheduling flags"
        );
        // Construction-time switches only: set while forking, then cleared so
        // the control and every other profile keep the default schedule.
        for flag in &candidate_flags {
            unsafe { std::env::set_var(flag, "1") };
        }
        let built = target.fork_profile(
            &core,
            &core_bytes,
            &stream,
            &p.kernels,
            6,
            crate::compressed_source::Phase::Verify,
        );
        for flag in &candidate_flags {
            unsafe { std::env::remove_var(flag) };
        }
        let mut t = built?;
        t.set_graph_replay(true)?;
        let shared = target.shared_state_report(&t);
        ensure!(
            shared["all_weights_and_causal_storage_shared"] == true,
            "candidate verifier duplicated model/causal storage: {shared}"
        );
        Some(t)
    };
    if compact_screen {
        let control=verify.as_mut().unwrap();let other=candidate.as_mut().unwrap();
        control.apply_candidate_schedule(&stream)?;other.apply_candidate_schedule(&stream)?;
        other.enable_compact_sparse_for_probe(&stream)?;
        ensure!(control.adaptive_sparse_index_layers()==4 && other.adaptive_sparse_index_layers()==4
            && control.compact_sparse_index_layers()==0 && other.compact_sparse_index_layers()==4
            && control.schedule_counts_for_probe()==other.schedule_counts_for_probe(),
            "compact selection comparison changed another schedule");
    }
    if sparse_screen {
        verify.as_mut().unwrap().apply_candidate_schedule(&stream)?;
        candidate.as_mut().unwrap().apply_candidate_schedule(&stream)?;
        ensure!(verify.as_ref().unwrap().sparse_index_layers()==0
            && candidate.as_ref().unwrap().sparse_index_layers()==4,"sparse screen constructor mismatch");
        ensure!(candidate.as_ref().unwrap().adaptive_sparse_index_layers()==if adaptive_sparse_screen {4}else{0},
            "adaptive sparse constructor mismatch");
    }
    if fused_qkv_screen {
        let control=verify.as_mut().unwrap();let other=candidate.as_mut().unwrap();
        control.apply_candidate_schedule(&stream)?;other.apply_candidate_schedule(&stream)?;
        ensure!(control.fused_projection_layers()==0 && other.fused_projection_layers()==40,
            "fused QKV screen constructor mismatch");
        // The forty Q-B prefetch windows were hidden under the now-removed
        // QKV gathers. Other overlap and PDL settings must remain identical.
        ensure!(control.schedule_counts_for_probe()==[117,40,80,40]
            && other.schedule_counts_for_probe()==[77,40,80,40],"fused QKV scheduling contract");
    }
    if sort_screen {
        let control = verify.as_mut().unwrap();
        control.apply_candidate_schedule(&stream)?;
        control.set_sort_shuffle_for_probe(false, &stream)?;
        let other = candidate.as_mut().unwrap();
        other.apply_candidate_schedule(&stream)?;
        other.set_sort_shuffle_for_probe(true, &stream)?;
    }
    if micro_screen {
        let control=verify.as_mut().unwrap();
        control.apply_candidate_schedule(&stream)?;
        let other=candidate.as_mut().unwrap();
        other.apply_candidate_schedule(&stream)?;
        ensure!(control.micro_moe_layers()==0 && other.micro_moe_layers()==40,
            "micro comparison must change only the candidate's forty routed layers");
    }
    if live_moe_screen {
        let control=verify.as_mut().unwrap();
        control.apply_candidate_schedule(&stream)?;
        let other=candidate.as_mut().unwrap();
        other.apply_candidate_schedule(&stream)?;
        ensure!(control.live_moe_layers()==0 && other.live_moe_layers()==40,
            "live comparison must change only the candidate's forty routed layers");
    }
    if scale_screen {
        let control=verify.as_mut().unwrap();control.apply_candidate_schedule(&stream)?;
        let other=candidate.as_mut().unwrap();other.apply_candidate_schedule(&stream)?;
        ensure!(control.live_moe_layers()==40 && other.live_moe_layers()==40
            && control.scale_reuse_moe_layers()==0 && other.scale_reuse_moe_layers()==40,
            "scale comparison must change only the grouped candidate library");
    }
    if compute_screen {
        let control = verify.as_mut().unwrap(); control.apply_candidate_schedule(&stream)?;
        let other = candidate.as_mut().unwrap(); other.apply_candidate_schedule(&stream)?;
        ensure!(control.live_moe_layers() == 40 && other.live_moe_layers() == 40
            && control.task_order_moe_layers() == 0 && other.task_order_moe_layers() == if narrow_only {0} else {40}
            && control.narrow_projection_layers() == 0 && other.narrow_projection_layers() == if task_order_only {0} else {40},
            "compute comparison must select the requested projection/expert policies in all forty layers");
    }
    if !schedule_screen && !p.retained_prefill && (!p.oracle_logits || std::env::var("CANADA_SCRATCH_QUALITY_REPLAY").as_deref()==Ok("1"))
        && p.fixed_verify_fixture.is_none()
        && std::env::var_os("CANADA_SCRATCH_STATE_IMAGE_PROBE").is_none() {
        if let Some(v) = &mut verify {
            let policy = v.apply_candidate_schedule(&stream)?;
            eprintln!("rank{} candidate verifier schedule: {policy}", p.rank);
        }
    }
    if let Some(v) = &decode_shared_state {
        ensure!(
            v["all_weights_and_causal_storage_shared"] == true,
            "decode duplicated model/causal storage"
        );
    }
    if let Some(v) = &shared_state {
        ensure!(
            v["all_weights_and_causal_storage_shared"] == true,
            "profiles duplicated model/causal storage"
        );
    }
    let load_seconds = load_start.elapsed().as_secs_f64();
    ensure!(p.context <= 65536 || cuda.admission_memory()?.admitted_available >= 8 * 1024usize.pow(3),
        "full-context profiles leave insufficient reserve for draft/capture/runtime work");
    eprintln!("rank{} target loaded in {load_seconds:.3}s", p.rank);
    let mut comm = Roce::local(&cuda, &p.kernels.join("roce-v0"), p.rank)?;
    comm.attach_gather(&p.kernels.join("roce-gather-v1"))?;
    let all = crate::rendezvous::exchange_blob_with_timeout(
        p.rank,
        p.endpoint.parse()?,
        &std::fs::read(&p.ticket)?,
        &comm.blob()?,
        std::time::Duration::from_secs(600),
    )?;
    comm.connect(&all)?;
    if p.prefill_nccl {
        let api = crate::collective::Nccl::load_qualified_prefill()?;
        let local_id = if p.rank == 0 {
            api.unique_id()?.bytes
        } else {
            [0u8; 128]
        };
        let mut endpoint: std::net::SocketAddr = p.endpoint.parse()?;
        endpoint.set_port(
            endpoint
                .port()
                .checked_add(2)
                .ok_or_else(|| anyhow::anyhow!("prefill rendezvous port overflow"))?,
        );
        let ids = crate::rendezvous::exchange_blob_with_timeout(
            p.rank,
            endpoint,
            &std::fs::read(&p.ticket)?,
            &local_id,
            std::time::Duration::from_secs(600),
        )?;
        let id = crate::collective::UniqueId {
            bytes: ids[..128].try_into().unwrap(),
        };
        comm.attach_prefill_sum(api.join(&cuda, id, p.rank, 4)?)?;
    }
    if std::env::var("CANADA_SCRATCH_ENGRAM_NCCL_GATHER").as_deref() == Ok("1") {
        comm.attach_prefill_column_gather(&core_bytes, widths.iter().copied().max().unwrap_or(4096).max(4096))?;
    }
    // The peer exchange uses the first ordered collective. Install every
    // communicator option before that operation binds the execution stream.
    if std::env::var("CANADA_SCRATCH_MOE_PIPELINE_AB").as_deref()==Ok("1") {
        comm.attach_moe_pipeline(&p.kernels.join("moe-transport-pipeline8-v2"),&stream)?;
    }
    let mut execution = Execution {
        target,
        small: small.take(),
        decode: decode.take(),
        verify: verify.take(),
        candidate: candidate.take(),
        wides: std::mem::take(&mut wides),
        sp_wides: std::mem::take(&mut sp_wides),
        comm,
    };
    let Execution {
        target: large,
        small,
        decode,
        verify,
        candidate,
        wides,
        sp_wides,
        comm,
    } = &mut execution;
    let mut active = 0usize;
    let mut stale_writes_rejected = 0usize;
    std::fs::create_dir_all(&p.output)?;
    if sparse_generation {
        return crate::sparse_generation_probe::run(&cuda,&core_bytes,&stream,&cp,&plan,large,
            verify.as_mut().unwrap(),candidate.as_mut().unwrap(),wides,comm,
            p.dspark_staged.as_deref().unwrap(),&p.kernels,&p.output);
    }
    if std::env::var("CANADA_SCRATCH_LONG_CONTINUATION").as_deref()==Ok("1") {
        ensure!(p.retained_generation && !p.oracle_logits && !p.retained_prefill && !sparse_generation,
            "long continuation requires isolated retained generation");
        if std::env::var("CANADA_SCRATCH_DRAFT_DEPTH_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_MARKOV_HEAD_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_GEMV_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_CONFIDENCE_FUSION_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_HISTORY_MATCH_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_DRAFT_HEAD_SCREEN").as_deref()==Ok("1") {
            let load_seconds=probe_load_started.elapsed().as_secs_f64();
            let mut result=crate::draft_depth_probe::run(&cuda,&core_bytes,&stream,&cp,&plan,
                large,verify.as_mut().unwrap(),wides,comm,p.dspark_staged.as_deref().unwrap(),&p.kernels,&p.output)?;
            result["target_load_and_collective_setup_seconds"]=json!(load_seconds);
            return Ok(result);
        }
        if std::env::var("CANADA_SCRATCH_WIDE_BANK_SCREEN").as_deref()==Ok("1") {
            return crate::wide_bank_probe::run(&cuda, &core_bytes, &stream, &cp, &plan,
                large, verify.as_mut().unwrap(), wides, comm,
                p.dspark_staged.as_deref().unwrap(), &p.kernels, &p.output);
        }
        if std::env::var("CANADA_SCRATCH_HISTORY_POLICY_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_DRAFT_TEMPERATURE_SCREEN").as_deref()==Ok("1") {
            return crate::history_policy_probe::run(&cuda,&core_bytes,&stream,&cp,&plan,
                large,verify.as_mut().unwrap(),wides,comm,
                p.dspark_staged.as_deref().unwrap(),&p.kernels,&p.output);
        }
        return crate::long_continuation_probe::run(&cuda,&core_bytes,&stream,&cp,&plan,large,
            verify.as_mut().unwrap(),wides,comm,p.dspark_staged.as_deref().unwrap(),&p.kernels,&p.output);
    }
    if p.retained_generation {
        ensure!(p.dspark_agentic && p.wide_prefill && !p.natural_generation && !p.retained_prefill
            && !p.oracle_logits && p.fixed_verify_fixture.is_none() && !p.agentic_parity_debug
            && !p.wide_sequence_parallel,"retained generation requires one admitted target/draft profile set");
        if std::env::var("CANADA_SCRATCH_CAUSAL_VERIFY_SCREEN").as_deref() == Ok("1") {
            ensure!(!generation_screen && !rapid_screen && !p.natural_generation
                && std::env::var_os("CANADA_SCRATCH_STATE_IMAGE_PROBE").is_none(),
                "causal depth screen requires isolated sampled generation");
            return crate::causal_verify_probe::run(&cuda,&core_bytes,&stream,&cp,&plan,large,
                verify.as_mut().unwrap(),wides,comm,&p.batches[0],
                p.dspark_staged.as_deref().unwrap(),&p.kernels,&p.output,&mut None,false);
        }
        if generation_screen {
            ensure!(std::env::var_os("CANADA_SCRATCH_STATE_IMAGE_PROBE").is_none(),
                "generation schedule screen is isolated from state-image work");
            return crate::generation_schedule_probe::run(&cuda,&core_bytes,&stream,&cp,&plan,large,
                verify.as_mut().unwrap(),candidate.as_mut().unwrap(),wides,comm,&p.batches[0],
                p.dspark_staged.as_deref().unwrap(),&p.kernels,&p.output);
        }
        if std::env::var("CANADA_SCRATCH_STATE_IMAGE_PROBE").as_deref() == Ok("1") {
            return crate::state_image_probe::run(&cuda,&core_bytes,&stream,&cp,&plan,large,
                verify.as_mut().unwrap(),wides,comm,&p.batches[0],p.dspark_staged.as_deref().unwrap(),&p.kernels,&p.output);
        }
        return crate::retained_generation_probe::run(&cuda,&core_bytes,&stream,&cp,&plan,large,
            verify.as_mut().unwrap(),wides,comm,&p.batches[0],p.dspark_staged.as_deref().unwrap(),&p.kernels,&p.output);
    }
    if p.natural_generation {
        ensure!(p.dspark_agentic && p.wide_prefill && !p.retained_prefill && !p.oracle_logits
            && p.fixed_verify_fixture.is_none() && !p.agentic_parity_debug && !p.wide_sequence_parallel,
            "natural generation requires one admitted target/draft profile set");
        return crate::natural_generation_probe::run(&cuda,&core_bytes,&stream,&cfg,&cp,&plan,large,
            verify.as_mut().unwrap(),wides,comm,&p.batches[0],p.dspark_staged.as_deref().unwrap(),&p.kernels,&p.output);
    }
    if p.oracle_logits {
        ensure!(p.dspark_agentic && !p.retained_prefill && p.fixed_verify_fixture.is_none()
            && !p.agentic_parity_debug && !p.wide_sequence_parallel,
            "oracle replay needs one prefill path and the agentic verifier");
        if std::env::var("CANADA_SCRATCH_QUALITY_REPLAY").as_deref()==Ok("1") {
            return crate::quality_replay_probe::run(&cuda,large,verify.as_mut().unwrap(),wides,comm,&stream,&p.output,p.rank);
        }
        return crate::oracle_logits_probe::run(
            &cuda, large, verify.as_mut().unwrap(), wides, comm, &stream, &p.batches[0], &p.output, p.rank,
        );
    }
    if p.retained_prefill {
        ensure!(p.wide_prefill && p.dspark_agentic && !wides.is_empty()
            && !p.wide_sequence_parallel && !p.fixed_verify_fixture.is_some(),
            "retained prefill needs one wide profile set and the agentic verifier");
        if std::env::var("CANADA_SCRATCH_LONG_RETAINED_PROBE").as_deref()==Ok("1") {
            ensure!(!rapid_screen && p.graph_replay && p.context==1048576,
                "long retained probe requires isolated1M profiles");
            verify.as_mut().unwrap().apply_candidate_schedule(&stream)?;
            if std::env::var("CANADA_SCRATCH_POST_THINKING_DRAFT_SCREEN").as_deref()==Ok("1") {
                return crate::post_thinking_draft_probe::run(&cuda,&core_bytes,&stream,&cp,&plan,large,
                    verify.as_mut().unwrap(),wides,comm,p.dspark_staged.as_deref().unwrap(),&p.kernels);
            }
            if std::env::var("CANADA_SCRATCH_TINY_APPEND_SCREEN").as_deref()==Ok("1") {
                large.enable_cached_extend_floor()?;
                let mut extend=if std::env::var("CANADA_SCRATCH_TINY_APPEND_PHASE_SCREEN").as_deref()==Ok("1") {
                    ensure!(cuda.admission_memory()?.admitted_available>=11*1024usize.pow(3),
                        "tiny append phase profile reserve");
                    let mut target=large.fork_profile(&core,&core_bytes,&stream,&p.kernels,6,crate::compressed_source::Phase::Extend)?;
                    target.set_graph_replay(true)?;target.enable_cached_extend_floor()?;
                    ensure!(cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3),
                        "tiny append phase profile consumed reserve");
                    Some(target)
                }else{None};
                return crate::tiny_append_probe::run(&cuda,large,verify.as_mut().unwrap(),wides,
                    comm,&stream,&p.output,p.rank,extend.as_mut(),&std::fs::read(&p.ticket)?);
            }
            if std::env::var("CANADA_SCRATCH_LONG_HISTORY_VERIFY").as_deref()==Ok("1") {
                let mut longer=large.fork_profile(&core,&core_bytes,&stream,&p.kernels,16,crate::compressed_source::Phase::Verify)?;
                longer.set_graph_replay(true)?;
                if std::env::var("CANADA_SCRATCH_LONG_HISTORY_GENERATION").as_deref()==Ok("1") {
                    return crate::long_history_generation_probe::run(&cuda,&core_bytes,&stream,&cp,&plan,large,
                        verify.as_mut().unwrap(),&mut longer,wides,comm,p.dspark_staged.as_deref().unwrap(),&p.kernels);
                }
                return crate::long_history_verify_probe::run(&cuda,large,verify.as_mut().unwrap(),&mut longer,wides,
                    comm,&stream,&p.output,p.rank,&std::fs::read(&p.ticket)?,&p.kernels);
            }
            if std::env::var("CANADA_SCRATCH_LONG_HISTORY_SCORE").as_deref()==Ok("1") {
                return crate::long_history_score_probe::run(&cuda,large,verify.as_mut().unwrap(),wides,
                    comm,&stream,&p.output,p.rank);
            }
            if std::env::var("CANADA_SCRATCH_STAGED_METADATA_SCREEN").as_deref()==Ok("1") {
                ensure!(std::env::var_os("CANADA_SCRATCH_STAGED_METADATA").is_none(),"staged metadata screen requires ordinary control allocation");
                unsafe{std::env::set_var("CANADA_SCRATCH_STAGED_METADATA","1");}
                let built=large.fork_profile(&core,&core_bytes,&stream,&p.kernels,6,crate::compressed_source::Phase::Verify);
                unsafe{std::env::remove_var("CANADA_SCRATCH_STAGED_METADATA");}
                let mut other=built?;other.handoff_from(large)?;other.set_graph_replay(true)?;other.apply_candidate_schedule(&stream)?;
                return crate::staged_metadata_model_probe::run(&cuda,large,verify.as_mut().unwrap(),&mut other,wides,comm,&stream,&p.output,p.rank);
            }
            if std::env::var("CANADA_SCRATCH_MAPPED_METADATA_SCREEN").as_deref()==Ok("1") {
                ensure!(std::env::var_os("CANADA_SCRATCH_MAPPED_METADATA").is_none(),"mapped metadata screen requires ordinary control allocation");
                unsafe{std::env::set_var("CANADA_SCRATCH_MAPPED_METADATA","1");}
                let built=large.fork_profile(&core,&core_bytes,&stream,&p.kernels,6,crate::compressed_source::Phase::Verify);
                unsafe{std::env::remove_var("CANADA_SCRATCH_MAPPED_METADATA");}
                let mut other=built?;other.handoff_from(large)?;other.set_graph_replay(true)?;other.apply_candidate_schedule(&stream)?;
                return crate::mapped_metadata_model_probe::run(&cuda,large,verify.as_mut().unwrap(),&mut other,wides,comm,&stream,&p.output,p.rank);
            }
            if std::env::var("CANADA_SCRATCH_CACHED_GATHER_SCREEN").as_deref()==Ok("1") {
                ensure!(std::env::var_os("CANADA_SCRATCH_CACHED_INDEX_GATHER").is_none(),"cached gather screen needs an uncached control");
                unsafe{std::env::set_var("CANADA_SCRATCH_CACHED_INDEX_GATHER","1");}
                let built=large.fork_profile(&core,&core_bytes,&stream,&p.kernels,6,crate::compressed_source::Phase::Verify);
                unsafe{std::env::remove_var("CANADA_SCRATCH_CACHED_INDEX_GATHER");}
                let mut other=built?;other.handoff_from(large)?;other.set_graph_replay(true)?;other.apply_candidate_schedule(&stream)?;
                return crate::cached_gather_model_probe::run(&cuda,large,verify.as_mut().unwrap(),&mut other,wides,comm,&stream,&p.output,p.rank);
            }
            if std::env::var("CANADA_SCRATCH_TINY_ENCODER_DIAGNOSTIC").as_deref()==Ok("1") {
                large.enable_cached_extend_floor()?;
                return crate::tiny_encoder_probe::diagnose(&cuda,large,verify.as_mut().unwrap(),
                    wides,comm,&stream,&p.output,p.rank);
            }
            if std::env::var("CANADA_SCRATCH_TINY_ENCODER_ONLY").as_deref()==Ok("1") {
                large.enable_cached_extend_floor()?;
                ensure!(large.cached_grouped_union_layers()==4,"grouped union cache did not bind all128-row consumers");
                return Ok(serde_json::json!({"tiny_encoder":crate::tiny_encoder_probe::run(&cuda,large,
                    verify.as_mut().unwrap(),wides,comm,&stream,&p.output,p.rank)?}));
            }
            if std::env::var("CANADA_SCRATCH_INPUT_HINT_SCREEN").as_deref()==Ok("1") {
                large.enable_cached_extend_floor()?;
                ensure!(std::env::var_os("CANADA_SCRATCH_INPUT_HINT_CHUNK").is_none(),
                    "input hint screen needs unchanged control geometry");
                let hint_off=std::env::var("CANADA_SCRATCH_INPUT_HINT_OFF_SCREEN").as_deref()==Ok("1");
                unsafe{std::env::set_var("CANADA_SCRATCH_INPUT_HINT_CHUNK",if hint_off{"0"}else{"1048576"});}
                let built=large.fork_profile(&core,&core_bytes,&stream,&p.kernels,6,
                    crate::compressed_source::Phase::Verify);
                unsafe{std::env::remove_var("CANADA_SCRATCH_INPUT_HINT_CHUNK");}
                let mut other=built?;other.handoff_from(large)?;other.set_graph_replay(true)?;
                other.apply_candidate_schedule(&stream)?;
                let mut report=crate::norm_quant_model_probe::run(&cuda,large,verify.as_mut().unwrap(),&mut other,
                    wides,comm,&stream,&p.output,p.rank)?;
                if std::env::var("CANADA_SCRATCH_TINY_ENCODER_128_SCREEN").as_deref()==Ok("1") {
                    report["tiny_encoder"]=crate::tiny_encoder_probe::run(&cuda,large,verify.as_mut().unwrap(),
                        wides,comm,&stream,&p.output,p.rank)?;
                }
                return Ok(report);
            }
            if std::env::var("CANADA_SCRATCH_QUERY_NORM_QUANT_SCREEN").as_deref()==Ok("1") {
                ensure!(std::env::var_os("CANADA_SCRATCH_QUERY_NORM_QUANT_FUSION").is_none(),
                    "query norm screen requires an unchanged query control");
                unsafe{std::env::set_var("CANADA_SCRATCH_QUERY_NORM_QUANT_FUSION","1");}
                let built=large.fork_profile(&core,&core_bytes,&stream,&p.kernels,6,
                    crate::compressed_source::Phase::Verify);
                unsafe{std::env::remove_var("CANADA_SCRATCH_QUERY_NORM_QUANT_FUSION");}
                let mut other=built?;other.handoff_from(large)?;other.set_graph_replay(true)?;
                other.apply_candidate_schedule(&stream)?;
                return crate::norm_quant_model_probe::run(&cuda,large,verify.as_mut().unwrap(),&mut other,
                    wides,comm,&stream,&p.output,p.rank);
            }
            if std::env::var("CANADA_SCRATCH_NORM_QUANT_SCREEN").as_deref()==Ok("1") {
                if std::env::var("CANADA_SCRATCH_TINY_ENCODER_SCREEN").as_deref()==Ok("1") {
                    large.enable_cached_extend_floor()?;
                }
                ensure!(std::env::var_os("CANADA_SCRATCH_NORM_QUANT_FUSION").is_none(),
                    "norm quant screen requires an unchanged control");
                unsafe{std::env::set_var("CANADA_SCRATCH_NORM_QUANT_FUSION","1");}
                let built=large.fork_profile(&core,&core_bytes,&stream,&p.kernels,6,
                    crate::compressed_source::Phase::Verify);
                unsafe{std::env::remove_var("CANADA_SCRATCH_NORM_QUANT_FUSION");}
                let mut other=built?;other.handoff_from(large)?;other.set_graph_replay(true)?;
                other.apply_candidate_schedule(&stream)?;
                let mut report=crate::norm_quant_model_probe::run(&cuda,large,verify.as_mut().unwrap(),&mut other,
                    wides,comm,&stream,&p.output,p.rank)?;
                if std::env::var("CANADA_SCRATCH_TINY_ENCODER_SCREEN").as_deref()==Ok("1") {
                    report["tiny_encoder"]=crate::tiny_encoder_probe::run(&cuda,large,verify.as_mut().unwrap(),
                        wides,comm,&stream,&p.output,p.rank)?;
                }
                return Ok(report);
            }
            if std::env::var("CANADA_SCRATCH_SHARED_QUANT_SCREEN").as_deref()==Ok("1") {
                ensure!(std::env::var_os("CANADA_SCRATCH_FUSED_SHARED_QUANT").is_none(),
                    "shared fusion screen requires an unchanged control");
                unsafe{std::env::set_var("CANADA_SCRATCH_FUSED_SHARED_QUANT","1");}
                let built=large.fork_profile(&core,&core_bytes,&stream,&p.kernels,6,
                    crate::compressed_source::Phase::Verify);
                unsafe{std::env::remove_var("CANADA_SCRATCH_FUSED_SHARED_QUANT");}
                let mut other=built?;other.handoff_from(large)?;other.set_graph_replay(true)?;
                other.apply_candidate_schedule(&stream)?;
                return crate::shared_quant_model_probe::run(&cuda,large,verify.as_mut().unwrap(),&mut other,
                    wides,comm,&stream,&p.output,p.rank);
            }
            if std::env::var("CANADA_SCRATCH_PACKED_INDEX_GATHER_SCREEN").as_deref()==Ok("1") {
                let sparse=std::env::var("CANADA_SCRATCH_PACKED_SPARSE_GATHER_SCREEN").as_deref()==Ok("1");
                ensure!(std::env::var_os("CANADA_SCRATCH_PACKED_INDEX_GATHER").is_none()
                    && std::env::var_os("CANADA_SCRATCH_PACKED_SPARSE_GATHER").is_none(),
                    "packed gather screen requires an unchanged control");
                unsafe{std::env::set_var("CANADA_SCRATCH_PACKED_INDEX_GATHER","4");}
                if sparse {unsafe{std::env::set_var("CANADA_SCRATCH_PACKED_SPARSE_GATHER","4");}}
                let built=large.fork_profile(&core,&core_bytes,&stream,&p.kernels,6,
                    crate::compressed_source::Phase::Verify);
                unsafe{std::env::remove_var("CANADA_SCRATCH_PACKED_INDEX_GATHER");}
                if sparse {unsafe{std::env::remove_var("CANADA_SCRATCH_PACKED_SPARSE_GATHER");}}
                let mut other=built?;other.handoff_from(large)?;other.set_graph_replay(true)?;
                other.apply_candidate_schedule(&stream)?;
                return crate::index_gather_model_probe::run(&cuda,large,verify.as_mut().unwrap(),&mut other,
                    wides,comm,&stream,&p.output,p.rank);
            }
            if std::env::var("CANADA_SCRATCH_VERIFICATION_FUSION_SCREEN").as_deref()==Ok("1") {
                ensure!(["CANADA_SCRATCH_ENGRAM_COLD_SCREEN","CANADA_SCRATCH_ENGRAM_SPLIT14_SCREEN",
                    "CANADA_SCRATCH_KEYSPLIT_INDEX_AB","CANADA_SCRATCH_COMPACT_REMAP_SORT",
                    "CANADA_SCRATCH_MOE_TASK_ORDER_M3"].iter().all(|f|std::env::var_os(f).is_none()),
                    "verification fusion screen requires an isolated unmodified control");
                let mut variants=Vec::new();
                for flag in [Some("CANADA_SCRATCH_COMPACT_REMAP_SORT"),Some("CANADA_SCRATCH_MOE_TASK_ORDER_M3"),None] {
                    if let Some(flag)=flag {unsafe{std::env::set_var(flag,"1");}}
                    let built=large.fork_profile(&core,&core_bytes,&stream,&p.kernels,6,
                        crate::compressed_source::Phase::Verify);
                    if let Some(flag)=flag {unsafe{std::env::remove_var(flag);}}
                    let mut other=built?;other.handoff_from(large)?;other.set_graph_replay(true)?;
                    other.apply_candidate_schedule(&stream)?;
                    if flag.is_none(){other.set_engram_split14_m3_for_probe()?;}
                    variants.push(other);
                }
                return crate::verification_fusion_probe::run(&cuda,large,verify.as_mut().unwrap(),&mut variants,
                    wides,comm,&stream,&p.output,p.rank);
            }
            if std::env::var("CANADA_SCRATCH_ENGRAM_SPLIT14_SCREEN").as_deref()==Ok("1") {
                ensure!(std::env::var_os("CANADA_SCRATCH_ENGRAM_COLD_SCREEN").is_none()
                    && std::env::var_os("CANADA_SCRATCH_KEYSPLIT_INDEX_AB").is_none(),"isolated Engram split14 screen");
                let mut other=large.fork_profile(&core,&core_bytes,&stream,&p.kernels,6,
                    crate::compressed_source::Phase::Verify)?;
                other.handoff_from(large)?;
                other.set_graph_replay(true)?;other.apply_candidate_schedule(&stream)?;
                other.set_engram_split14_for_probe(true)?;
                if std::env::var("CANADA_SCRATCH_ENGRAM_ADAPTIVE_SPLIT14_SCREEN").as_deref()==Ok("1") {
                    other.set_engram_adaptive_split14_for_probe()?;
                }
                return crate::engram_split_probe::run(&cuda,large,verify.as_mut().unwrap(),&mut other,
                    wides,comm,&stream,&p.output,p.rank);
            }
            if std::env::var("CANADA_SCRATCH_ENGRAM_COLD_SCREEN").as_deref()==Ok("1") {
                return crate::engram_cold_probe::run(&cuda,large,verify.as_mut().unwrap(),
                    wides,comm,&stream,&p.output,p.rank);
            }
            let mut key_candidate=if std::env::var("CANADA_SCRATCH_KEYSPLIT_INDEX_AB").as_deref()==Ok("1") {
                ensure!(std::env::var_os("CANADA_SCRATCH_KEY_SPLIT_MQA").is_none(),
                    "key split comparison needs an unchanged control");
                unsafe {std::env::set_var("CANADA_SCRATCH_KEY_SPLIT_MQA","24");}
                let built=large.fork_profile(&core,&core_bytes,&stream,&p.kernels,6,
                    crate::compressed_source::Phase::Verify);
                unsafe {std::env::remove_var("CANADA_SCRATCH_KEY_SPLIT_MQA");}
                let mut other=built?;other.set_graph_replay(true)?;
                other.apply_candidate_schedule(&stream)?;
                let shared=large.shared_state_report(&other);
                ensure!(shared["all_weights_and_causal_storage_shared"]==true
                    && verify.as_ref().unwrap().index_key_splits()==[0;8]
                    && other.index_key_splits()==[24;8],"paired key split profiles: control={:?} candidate={:?} shared={shared}",
                    verify.as_ref().unwrap().index_key_splits(),other.index_key_splits());
                Some(other)
            }else{None};
            return crate::long_retained_probe::run(&cuda,&core,large,verify.as_mut().unwrap(),
                key_candidate.as_mut(),wides,comm,&stream,&p.output,p.rank);
        }
        if std::env::var("CANADA_SCRATCH_WIDE16K_SCREEN").as_deref()==Ok("1") {
            return crate::wide16k_probe::run(&cuda,large,verify.as_mut().unwrap(),wides,comm,&stream,&p.output,p.rank);
        }
        if std::env::var("CANADA_SCRATCH_MOE_PIPELINE_AB").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_COALESCED_CODEC_AB").as_deref() == Ok("1")
            || std::env::var("CANADA_SCRATCH_PREFILL_NORM_FUSION_AB").as_deref() == Ok("1")
            || std::env::var("CANADA_SCRATCH_HC_FULLK_AB").as_deref() == Ok("1")
            || std::env::var("CANADA_SCRATCH_PREFILL_QUANT_GATHER_AB").as_deref() == Ok("1")
            || std::env::var("CANADA_SCRATCH_PREFILL_MOE_GATHER_AB").as_deref() == Ok("1")
            || std::env::var("CANADA_SCRATCH_PREFILL_LOCAL_QKV_AB").as_deref() == Ok("1") {
            ensure!(!rapid_screen, "norm fusion requires an isolated paired prefill");
            return crate::prefill_norm_probe::run(&cuda, large, verify.as_mut().unwrap(),
                wides, comm, &stream, &p.batches[0], &p.output, p.rank, &p.kernels);
        }
        if std::env::var("CANADA_SCRATCH_PREFILL_TAIL_GRAPH_AB").as_deref() == Ok("1") {
            ensure!(!rapid_screen && p.graph_replay && p.prefill_nccl_graph_experiment,
                "tail graph screen requires isolated explicit NCCL graph admission");
            return crate::prefill_tail_graph_probe::run(&cuda, large, verify.as_mut().unwrap(),
                wides, comm, &stream, &p.batches[0], &p.output, p.rank);
        }
        if rapid_screen {
            if schedule_ablation {
                return crate::rapid_schedule_probe::run_ablations(&cuda,&core,&core_bytes,large,
                    verify.as_mut().unwrap(),wides,comm,&stream,&p.kernels,&p.batches[0],&p.output,p.rank);
            }
            return crate::rapid_schedule_probe::run(&cuda, large, verify.as_mut().unwrap(),
                candidate.as_mut().unwrap(), wides, comm, &stream, &p.batches[0], &p.output, p.rank,None);
        }
        if std::env::var("CANADA_SCRATCH_ENGRAM_OVERLAP_AB").as_deref() == Ok("1") {
            return crate::engram_overlap_probe::run(
                &cuda, large, verify.as_mut().unwrap(), wides, comm, &stream, &p.batches[0], &p.output, p.rank,
            );
        }
        if std::env::var("CANADA_SCRATCH_ENGRAM_PREFETCH_AB").as_deref() == Ok("1") {
            return crate::engram_prefetch_probe::run(
                &cuda, large, verify.as_mut().unwrap(), wides, comm, &stream, &p.batches[0], &p.output, p.rank,
            );
        }
        return crate::retained_prefill_probe::run(
            &cuda, large, verify.as_mut().unwrap(), wides, comm, &stream, &p.batches[0], p.rank,
        );
    }
    if let Some(fixture) = &p.fixed_verify_fixture {
        ensure!(
            p.dspark_agentic && !p.agentic_parity_debug,
            "fixed verifier timing requires the admitted exact coding profile"
        );
        return crate::fixed_verify_probe::run(
            &cuda,
            large,
            verify.as_mut().unwrap(),
            candidate.as_mut().map(|c| (c, p.fixed_verify_candidate.as_slice())),
            comm,
            &stream,
            &p.batches[0],
            fixture,
            &p.output,
            p.rank,
            p.trace_layers,
        );
    }
    if p.agentic_parity_debug {
        ensure!(
            p.dspark_agentic,
            "teacher-forced parity diagnostic requires bounded agentic profile"
        );
        return crate::agentic_parity_probe::run(
            &cuda,
            large,
            verify.as_mut().unwrap(),
            comm,
            &stream,
            &p.batches[0],
            &p.output,
            p.rank,
        );
    }
    if p.dspark_generation && p.dspark_agentic {
        return crate::dspark_generation::probe_agentic(
            &cuda,
            &core_bytes,
            &stream,
            &cp,
            &plan,
            large,
            verify.as_mut().unwrap(),
            comm,
            &p.batches[0],
            p.dspark_staged.as_deref().unwrap(),
            &p.kernels,
            wides,
            sp_wides,
        );
    }
    if p.dspark_generation {
        return crate::dspark_generation::probe(
            &cuda,
            &core_bytes,
            &stream,
            &cp,
            &plan,
            large,
            verify.as_mut().unwrap(),
            comm,
            &p.batches[0],
            p.dspark_staged.as_deref().unwrap(),
            &p.kernels,
        );
    }
    if p.verification_transactions {
        return verification_transactions(
            &cuda,
            large,
            verify.as_mut().unwrap(),
            comm,
            &stream,
            &p.batches[0],
            &p.output,
            p.rank,
            load_seconds,
            &core_bytes,
            p.dspark_feature_probe,
        );
    }
    if p.performance_tokens != 0 {
        ensure!(
            p.performance_tokens == 64
                && p.profile_handoff
                && p.graph_replay
                && p.rows == 128
                && p.context == 512
                && !p.profile_decode_kernels
                && !p.decoder_tail_replay
                && p.cached_extend_floor,
            "bounded target-only continuous performance workload"
        );
        let decode = decode
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("decode profile absent"))?;
        let mut records = Vec::new();
        let mut reference = None;
        for repeat in 0..7 {
            if repeat > 0 {
                decode.reset(repeat as u64 + 1, comm, &stream)?;
                large.handoff_from(decode)?;
            }
            let start = Instant::now();
            large.append(&p.batches[0], comm, &stream)?;
            let first = large.greedy_token(&stream)?;
            let ttft_ms = start.elapsed().as_secs_f64() * 1000.;
            let mut output_ids = vec![first];
            decode.handoff_from(large)?;
            let decode_start = Instant::now();
            let mut latencies = Vec::new();
            for _ in 1..p.performance_tokens {
                let t = Instant::now();
                decode.append(&[*output_ids.last().unwrap()], comm, &stream)?;
                output_ids.push(decode.greedy_token(&stream)?);
                latencies.push(t.elapsed().as_secs_f64() * 1000.);
            }
            let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.;
            let e2e_ms = start.elapsed().as_secs_f64() * 1000.;
            ensure!(output_ids[0] == 271, "continuous prefix result changed");
            if let Some(prior) = &reference {
                ensure!(
                    *prior == output_ids,
                    "continuous greedy generation is not repeatable"
                );
            } else {
                reference = Some(output_ids.clone());
            }
            records.push(json!({"repeat":repeat,"warmup":repeat<2,"profiled":false,
                "prompt_tokens":128,"generated_tokens":p.performance_tokens,
                "output_ids":output_ids,"ttft_ms":ttft_ms,"e2e_ms":e2e_ms,
                "decode_ms_per_token":decode_ms/63.,"decode_tokens_per_s":63000./decode_ms,
                "inter_token_ms":latencies,"final_position":decode.position()}));
            // Request boundaries only: no diagnostic downloads/files/replays
            // between these63 real, sequential, state-advancing decode steps.
            std::fs::write(
                p.output.join("continuous-progress.json"),
                serde_json::to_vec_pretty(&records)?,
            )?;
        }
        return Ok(
            json!({"rank":p.rank,"load_seconds":load_seconds,"context":p.context,
            "no_drafting":true,"single_stream":true,"records":records,
            "all_generated_sequences_identical":true,
            "shared_state":shared_state,"decode_shared_state":decode_shared_state,
            "full_engine_acceptance":false,
            "scope":"Fresh128 ->64 genuine greedy tokens;2warmup+5measured requests. Native append+sampling latency includes host preparation/communication, excludes HTTP transport and diagnostic replay/copies. Same canonical quant/TP4, no drafting. Compare output IDs and actual interval with SGLang before drawing conclusions."}),
        );
    }
    let mut steps = Vec::new();
    for (index, batch) in p.batches.iter().enumerate() {
        if index > 0 && (p.reset_between_batches || p.reset_steps.contains(&index)) {
            match active {
                0 => large.reset(index as u64 + 1, comm, &stream)?,
                1 => small
                    .as_mut()
                    .unwrap()
                    .reset(index as u64 + 1, comm, &stream)?,
                _ => decode
                    .as_mut()
                    .unwrap()
                    .reset(index as u64 + 1, comm, &stream)?,
            }
        }
        let use_decode = p.decode_steps.contains(&index);
        let use_small = !use_decode && p.profile_handoff && batch.len() <= 6;
        let mode = if use_decode {
            2
        } else if use_small {
            1
        } else {
            0
        };
        if use_decode {
            ensure!(
                steps
                    .last()
                    .is_some_and(|s: &Value| s["greedy_token"].as_u64() == Some(batch[0] as u64)),
                "generated-token decode did not match previous native greedy choice"
            );
        }
        if mode != active {
            let (next, prior): (&mut Target, &Target) = match (active, mode) {
                (0, 1) => (small.as_mut().unwrap(), &*large),
                (1, 0) => (&mut *large, small.as_ref().unwrap()),
                (0, 2) => (decode.as_mut().unwrap(), &*large),
                (2, 0) => (&mut *large, decode.as_ref().unwrap()),
                (1, 2) => (decode.as_mut().unwrap(), small.as_ref().unwrap()),
                (2, 1) => (small.as_mut().unwrap(), decode.as_ref().unwrap()),
                _ => unreachable!(),
            };
            ensure!(
                !next.profile_is_current() && next.append(batch, comm, &stream).is_err(),
                "stale execution profile wrote shared state"
            );
            stale_writes_rejected += 1;
            next.handoff_from(prior)?;
            active = mode;
        }
        let use_graph = p.graph_replay && p.eager_from_step.is_none_or(|first| index < first);
        let start = Instant::now();
        let (target, captured_measurement, wall_ms) = if p.decoder_tail_replay && use_small {
            let input = small.as_mut().unwrap();
            input.append_replayed(large, batch, comm, &stream)?;
            let wall_ms = start.elapsed().as_secs_f64() * 1000.;
            if p.trace_decoder_step == Some(index) {
                large.dump_tail_boundaries(
                    input,
                    batch.len(),
                    &p.output.join(format!("partition-step{index}")),
                )?;
            }
            let measure = if use_graph {
                Some(input.measure_replayed(large, comm, &stream)?)
            } else {
                None
            };
            (&mut *large, measure, wall_ms)
        } else {
            let target = if use_decode {
                decode.as_mut().unwrap()
            } else if use_small {
                small.as_mut().unwrap()
            } else {
                &mut *large
            };
            if p.eager_from_step == Some(index) {
                target.set_graph_replay(false)?;
            }
            target.append(batch, comm, &stream)?;
            let wall_ms = start.elapsed().as_secs_f64() * 1000.;
            if p.trace_step == Some(index) {
                target.dump_layer_trace(&p.output.join(format!("step{index}/layers")))?;
            }
            let measure = if use_graph {
                Some(target.measure_current(
                    comm,
                    &stream,
                    p.profile_decode_kernels
                        && use_decode
                        && p.decode_steps.first() == Some(&index),
                )?)
            } else {
                None
            };
            (target, measure, wall_ms)
        };
        let root = p.output.join(format!("step{index}"));
        std::fs::create_dir_all(&root)?;
        if use_decode {
            target.dump_swa_row(target.position() as usize - 1, &root.join("swa-row.json"))?;
        }
        if p.trace_step != Some(index) && p.trace_layers && (index == 0 || p.trace_all_steps) {
            target.dump_layer_trace(&root.join("layers"))?;
        }
        let residual = target
            .residual()?
            .download()?
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let pre = target
            .pre()?
            .download()?
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        // Last-head execution produces one valid next-token vocabulary row.
        // Copy/save that row directly; full capacity downloads introduce large
        // between-rank artifact delays before the following append.
        let output_rows = if p.decoder_tail_replay && use_small {
            target.physical_rows()
        } else {
            batch.len()
        };
        let logits_row_start = if p.prefill_last_head {
            output_rows - 1
        } else {
            0
        };
        let logits = if p.prefill_last_head {
            unsafe {
                cuda.read_device(
                    target.logits()?.ptr() + (logits_row_start * 129280 * 4) as u64,
                    129280 * 4,
                )?
            }
        } else {
            target
                .logits()?
                .download()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let taps = vec![
            save(&root, "residual", &residual)?,
            save(&root, "pre", &pre)?,
            save(&root, "logits", &logits)?,
        ];
        let greedy_token = if p.greedy_tokens {
            Some(target.greedy_token(&stream)?)
        } else {
            None
        };
        eprintln!(
            "rank{} target append{index} position{} live{} wall{wall_ms:.3}ms",
            p.rank,
            target.position(),
            batch.len()
        );
        steps.push(json!({"step":index,"position":target.position(),"tokens":batch,"live_rows":batch.len(),"physical_rows":if mode==0 {p.rows}else{6},"output_rows":target.physical_rows(),"logits_row_start":logits_row_start,"tail_replay":p.decoder_tail_replay&&use_small,"decode":use_decode,"append_wall_ms":wall_ms,"append_phases":target.append_metrics()?,"taps":taps,"captured_measurement":captured_measurement,"greedy_token":greedy_token,"graph_replay":use_graph}));
        // Preserve completed observations even if later execution/teardown
        // fails. This progress file never awards successful experiment exit.
        let progress = p.output.join("progress.json");
        let temporary = progress.with_extension("tmp");
        std::fs::write(
            &temporary,
            serde_json::to_vec_pretty(
                &json!({"rank":p.rank,"steps":steps,"experiment_complete":false}),
            )?,
        )?;
        std::fs::rename(temporary, progress)?;
        ensure!(
            steps.last().unwrap()["captured_measurement"].is_null()
                || (steps.last().unwrap()["captured_measurement"]["valid_outputs_unchanged"]
                    == true
                    && (p.decoder_tail_replay
                        || steps.last().unwrap()["captured_measurement"]["outgoing_carry_unchanged"]
                            == true)),
            "captured replay changed published model outputs; progress evidence saved"
        );
    }
    Ok(
        json!({"rank":p.rank,"load_seconds":load_seconds,"source_phase":p.source_phase,"rows":p.rows,"context":p.context,"layers":40,"steps":steps,"graph_replay":p.graph_replay,"prefill_nccl":p.prefill_nccl,"prefill_last_head":p.prefill_last_head,"prefill_nccl_graph_experiment":p.prefill_nccl_graph_experiment,
        "atlas_bytes":large.atlas_bytes,"source_hashes":large.source_hashes,
        "profile_handoff":p.profile_handoff,"decoder_tail_replay":p.decoder_tail_replay,"cached_extend_floor":p.cached_extend_floor,"decode_steps":p.decode_steps,"decode_shared_state":decode_shared_state,"encoder_tail_bytes":large.decoder_tail_bytes(),"shared_state":shared_state,"stale_writes_rejected":stale_writes_rejected,"reset_steps":p.reset_steps,
        "reset_between_batches":p.reset_between_batches,"greedy_tokens":p.greedy_tokens,
        "full_model_parity_proven":false,"sampling_implemented":false,"greedy_only_implemented":true,
        "scope":"Fixture-free embedding ->40 target layers ->canonical BF16 head operands with FP32 accumulation/output/logits. Bounded one-sequence, 6-row increments or128-row complete chunks in the admitted context; raw captures awaiting qualified whole-model comparison. No complete prefill/DSpark/vision/service/session acceptance."}),
    )
}
