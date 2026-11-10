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

/// Validated before loading and applied when preparing each serving owner.
/// The explicit control preserves the established0.20recipe. Both admitted
/// cutoffs use the same causal stopping rule and exact rejection sampler.
pub(crate) fn serving_causal_three_threshold() -> Result<f32> {
    match std::env::var("CANADA_SCRATCH_CAUSAL_THREE_THRESHOLD").as_deref() {
        Ok("0.2" | "0.20") | Err(std::env::VarError::NotPresent) => Ok(0.20),
        Ok("0.25") => Ok(0.25),
        _ => anyhow::bail!("serving causal-three threshold must be0.20or0.25"),
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
    use super::{causal_prefix_rows, causal_full_prefix_rows, causal_two_three_rows};
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
    poisoned: bool,
    target_seed: Option<u64>,
    rejection: Option<Rejection>,
    stochastic_prefix_schedule: bool,
    causal_three_threshold: Option<f32>,
    causal_two_threshold: Option<f32>,
    causal_full_threshold: Option<f32>,
    causal_short_rows: usize,
    stop_token: Option<u32>,
    history_lookup: Option<HistoryLookup>,
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
    pub(crate) fn independent_for_probe(&self,target:&Target,core:&[u8],stream:&Stream,kernels:&Path)->Result<Self> {
        Self::new(&self.0.cuda,core,self.0.draft.independent_for_probe(target,core,stream,kernels)?)?
            .with_serving_policy(stream)
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
            poisoned: false, target_seed: None, stochastic_prefix_schedule,
            causal_three_threshold: None, causal_two_threshold: None, causal_full_threshold: None,
            causal_short_rows: 3, stop_token: None, history_lookup: None,
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
            let predictions=longer.verification_history_sample(self.target_seed.unwrap(),comm,stream)?;
            let verify_host_ms=longer.verification_host_ms()?;let verified=Instant::now();
            ensure!(predictions.len()==ROWS,"long history target draws");
            let mut matched=0;
            while matched<ROWS-1 && proposals[matched]==predictions[matched] {matched+=1;}
            let accepted=stop_frontier(&predictions,1+matched,self.stop_token);
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
        // The fixed five-row bidirectional drafter cannot query past the
        // advertised context. Finish the last four positions with ordinary
        // target draws instead; no draft probabilities are used at this edge.
        let context_tail = target.context_capacity().saturating_sub(start as usize)
            < crate::dspark_kv::QUERY_ROWS;
        // At a one-row boundary no proposal can be accepted. Reuse the
        // context-tail target draw, still committing target features into
        // drafter KV below so a cached continuation has complete causal state.
        let target_only = context_tail || (self.skip_unused_draft && limit==1);
        self.poisoned = true;
        let result = (|| {
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
            let lookup = if fallback_warm && !target_only {
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
                self.draft.run(anchor, comm, stream)?;
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
                let tokens = if self.rejection.is_some() && self.target_seed.is_some() {
                    let seed=round_seed.ok_or_else(||anyhow::anyhow!("stochastic draft needs target seed"))?;
                    if let Some(threshold)=self.causal_three_threshold.filter(|_|
                        self.causal_full_threshold.is_none()&&self.causal_short_rows==3) {
                        self.draft.stochastic_proposals_causal_three(seed,threshold,comm,stream)?
                    }else{self.draft.stochastic_proposals(seed,comm,stream)?}
                } else {
                    self.draft.greedy_proposals(comm, stream)?
                };
                let (confidence, scheduled) = self.draft.confidence_schedule()?;
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
            let profile_now = self.profile_verify
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
            let local_history=self.local_history_sampling&&lookup.is_some();
            let local_target=target.local_target_enabled()&&self.draft.hierarchical_active()
                &&self.block_verification&&self.target_seed.is_some()&&(lookup.is_none()||local_history)&&!target_only&&live>1;
            let verified_result = if local_target {
                    target.begin_local_verification(&verify[..live],comm,stream)
                }else{target.begin_verification(&verify[..live],comm,stream)}
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
            let profile_append_metrics = if profile_now {
                Some(serde_json::to_value(target.verification_metrics()?)?)
            } else {
                None
            };
            let verify_host_ms = target.verification_host_ms()?;
            ensure!(predictions.len() == live, "DSpark verifier rows");
            let verified = Instant::now();
            let mut matched = 0;
            while block_prefix.is_none() && matched + 1 < live && draft_tokens[matched] == predictions[matched] {
                matched += 1;
            }
            if let Some(prefix)=block_prefix { matched=prefix; }
            let accepted = stop_frontier(&predictions, (1 + matched).min(limit), self.stop_token);
            matched = accepted - 1;
            // Capture before publication; rejected feature rows remain private.
            self.features.run(target, stream)?;
            target.finish_verification(accepted, comm, stream)?;
            let committed = Instant::now();
            let (features, live) = self.features.values()?;
            self.draft.inject_distributed(features, live, accepted, start, comm, stream)?;
            if let Some(history) = &mut self.history_lookup {
                history.commit(start, &verify[..accepted])?;
                ensure!(history.position() == target.position(), "lookup/target commit frontiers disagree");
            }
            ensure!(
                target.position() == start + accepted as u64
                    && self.draft.position()? == target.position(),
                "DSpark commit frontiers disagree"
            );
            // The just-verified q remains intact through acceptance and commit.
            // Only a later proposal can observe the updated temperature.
            if let Some(warmer)=next_root_temperature {
                self.draft.set_draft_schedule(if warmer{3}else{0},stream)?;
            }
            Ok(Step {
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
                position: target.position(),
                phase_ms: [
                    drafted.duration_since(begin).as_secs_f64() * 1000.,
                    verified.duration_since(drafted).as_secs_f64() * 1000.,
                    committed.duration_since(verified).as_secs_f64() * 1000.,
                    committed.elapsed().as_secs_f64() * 1000.,
                ],
            })
        })();
        if result.is_ok() {
            self.poisoned = false;
        }
        result
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
