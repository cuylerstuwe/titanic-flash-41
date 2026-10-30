//! Native resident worker candidate. One GPU-owner thread retains target,
//! drafter and execution profiles; rank zero supplies ordered CPU commands.
//! The finite token-file driver is a qualification interface, not an HTTP API.
use crate::{
    checkpoint::Checkpoint,
    compressed_source::Phase,
    config::ModelConfig,
    control_channel::ControlChannel,
    cuda::{Buffer, Cuda, Stream},
    dspark::QueryNetwork,
    dspark_generation::{GreedySession, PreparedSession, LongHistoryBuffers},
    hash_history::HashMetadata,
    layout::RankPlan,
    resident_history::{Finish, Plan as RequestPlan, ResidentHistory},
    roce::Roce,
    target::Target,
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

#[path = "worker_interleave_probe.rs"]
mod interleave_probe;
#[path = "request_checkpoint.rs"]
mod request_checkpoint;
#[path = "encoder_prompt_cache_probe.rs"]
mod encoder_cache_probe;
#[path = "mask_storage_probe.rs"]
mod mask_storage_probe;
#[path = "history_associativity_probe.rs"]
mod history_associativity_probe;
#[path = "dense_live_rows_probe.rs"]
mod dense_live_rows_probe;
#[cfg(any(test, feature = "storage-fault-injection"))]
#[path = "cache_admission_fault.rs"]
mod cache_admission_fault;
#[cfg(feature = "storage-fault-injection")]
#[path = "cache_retention_probe.rs"]
mod cache_retention_probe;
#[cfg(feature = "storage-fault-injection")]
#[path = "cache_pressure_probe.rs"]
mod cache_pressure_probe;
#[path = "active_encoder_checkpoint_probe.rs"]
mod active_encoder_checkpoint_probe;
#[path = "split_attention_generation_probe.rs"]
mod split_generation_probe;
#[path = "source_prior_probe.rs"]
mod source_prior_probe;
#[path = "sampler_readout_probe.rs"]
mod sampler_readout_probe;
#[path = "fixed_prefix_quality_probe.rs"]
mod fixed_prefix_quality_probe;
#[path = "block_progress_probe.rs"]
mod block_progress_probe;
#[path = "pooled_route_probe.rs"]
mod pooled_route_probe;
#[path = "private_int8_probe.rs"]
mod private_int8_probe;
#[path = "worker_background_prefill.rs"]
mod background_prefill;
use background_prefill::PendingPrefill;
#[cfg(feature = "serving")]
#[path = "worker_interleave_http.rs"]
mod interleave_http;

fn adaptive_sparse_policy(sparse: Option<&str>, adaptive: Option<&str>) -> Result<bool> {
    let flag = |value| -> Result<bool> {
        match value {
            None | Some("0") => Ok(false),
            Some("1") => Ok(true),
            _ => anyhow::bail!("resident sparse policy must be0/1"),
        }
    };
    let sparse = flag(sparse)?;
    ensure!(sparse == flag(adaptive)?, "resident sparse mode requires the paired adaptive policy");
    Ok(sparse)
}

fn redact_http_event(event:&mut Value) {
    event.as_object_mut().unwrap().remove("output_ids");
    event.as_object_mut().unwrap().remove("pending");
    if let Some(cache)=event.get_mut("retained_session_cache").and_then(Value::as_object_mut) {
        if let Some(anchor)=cache.remove("exact_prompt_anchor") {
            cache.insert("exact_prompt_hit".into(),json!(anchor.is_u64()));
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerConfig {
    config: PathBuf,
    checkpoint: PathBuf,
    rank: usize,
    core: PathBuf,
    kernels: PathBuf,
    stages: Vec<PathBuf>,
    io_stage: PathBuf,
    backing: PathBuf,
    hash_metadata: PathBuf,
    dspark_staged: PathBuf,
    endpoint: SocketAddr,
    ticket: PathBuf,
    context: usize,
    wide_widths: Vec<usize>,
    events: PathBuf,
    // Only rank zero reads this file. Future serving supplies the same driver
    // interface through a bounded request queue, without replacing the engine.
    requests: Option<PathBuf>,
    vision: Option<crate::worker_media::Config>,
    #[cfg(feature = "serving")]
    http: Option<crate::http_server::ServerConfig>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    input_ids: Option<Vec<u32>>,
    append_tokens: Option<Vec<u32>>,
    #[serde(default)]
    include_pending: bool,
    max_new_tokens: usize,
    seed: u64,
    #[serde(default)]
    cancel_after_tokens: Option<usize>,
    #[serde(default)]
    images: Vec<crate::worker_media::Fixture>,
}
impl Request {
    fn prompt(&self, history: &ResidentHistory) -> Result<Vec<u32>> {
        ensure!(
            self.input_ids.is_some() != self.append_tokens.is_some(),
            "resident pilot needs either full input IDs or a retained append"
        );
        if let Some(tokens) = &self.input_ids {
            ensure!(
                !self.include_pending,
                "pending anchor applies only to a retained append"
            );
            return Ok(tokens.clone());
        }
        ensure!(
            !history.committed().is_empty(),
            "retained append without a completed request"
        );
        let append = self.append_tokens.as_ref().unwrap();
        ensure!(
            history.frontier() + append.len() + usize::from(self.include_pending)
                <= crate::config::MODEL_CONTEXT,
            "retained pilot input exceeds model context"
        );
        let mut tokens = history.committed().to_vec();
        if self.include_pending {
            tokens.push(
                history
                    .pending()
                    .ok_or_else(|| anyhow::anyhow!("retained pending anchor absent"))?,
            );
        }
        tokens.extend_from_slice(append);
        Ok(tokens)
    }
}

enum Command {
    Begin {
        id: u64,
        seed: u64,
        greedy: bool,
        limit: usize,
        tokens: Vec<u32>,
        images: Vec<crate::image_cache::Descriptor>,
    },
    Step,
    Finish(Finish),
    Shutdown,
    Idle,
    ReclaimInputs { headroom: usize },
}

impl Command {
    fn encode(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        match self {
            Self::Begin {
                id,
                seed,
                greedy,
                limit,
                tokens,
                images,
            } => {
                ensure!(
                    tokens.len() <= crate::config::MODEL_CONTEXT && *limit <= 262144,
                    "resident command bounds"
                );
                out.push(if images.is_empty() {if *greedy {6}else{1}}else{if *greedy {13}else{12}});
                out.extend_from_slice(&id.to_le_bytes());
                out.extend_from_slice(&seed.to_le_bytes());
                out.extend_from_slice(&(*limit as u32).to_le_bytes());
                out.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
                for token in tokens {
                    out.extend_from_slice(&token.to_le_bytes());
                }
                if !images.is_empty() {crate::image_cache::Descriptor::encode_all(images,&mut out)?;}
            }
            Self::Step => out.push(2),
            Self::Finish(reason) => out.extend_from_slice(&[
                3,
                match reason {
                    Finish::Stop => 0,
                    Finish::Length => 1,
                    Finish::Cancelled => 2,
                },
            ]),
            Self::Shutdown => out.push(4),
            Self::Idle => out.push(5),
            Self::ReclaimInputs { headroom } => {
                ensure!(*headroom > 8usize << 30 && *headroom <= 12usize << 30,
                    "input reclamation headroom bounds");
                out.push(17);out.extend_from_slice(&(*headroom as u64).to_le_bytes());
            }
        }
        Ok(out)
    }
    fn decode(bytes: &[u8]) -> Result<Self> {
        match bytes {
            [1 | 6 | 12 | 13, ..] if bytes.len() >= 25 => {
                let n = u32::from_le_bytes(bytes[21..25].try_into().unwrap()) as usize;
                ensure!(
                    n <= crate::config::MODEL_CONTEXT && bytes.len() >= 25 + 4 * n,
                    "resident prompt command length"
                );
                let end=25+4*n;
                let images=if bytes[0]>=12 {
                    let images=crate::image_cache::Descriptor::decode_all(&bytes[end..])?;
                    ensure!(!images.is_empty(),"image command without images");images
                }else{ensure!(bytes.len()==end,"extra text command data");Vec::new()};
                Ok(Self::Begin {
                    id: u64::from_le_bytes(bytes[1..9].try_into().unwrap()),
                    seed: u64::from_le_bytes(bytes[9..17].try_into().unwrap()),
                    greedy: matches!(bytes[0],6|13),
                    limit: u32::from_le_bytes(bytes[17..21].try_into().unwrap()) as usize,
                    tokens: bytes[25..end]
                        .chunks_exact(4)
                        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
                        .collect(),
                    images,
                })
            }
            [2] => Ok(Self::Step),
            [3, 0] => Ok(Self::Finish(Finish::Stop)),
            [3, 1] => Ok(Self::Finish(Finish::Length)),
            [3, 2] => Ok(Self::Finish(Finish::Cancelled)),
            [4] => Ok(Self::Shutdown),
            [5] => Ok(Self::Idle),
            [17, ..] if bytes.len() == 9 => {
                let headroom = usize::try_from(u64::from_le_bytes(bytes[1..].try_into().unwrap()))?;
                ensure!(headroom > 8usize << 30 && headroom <= 12usize << 30,
                    "input reclamation headroom bounds");
                Ok(Self::ReclaimInputs { headroom })
            }
            _ => anyhow::bail!("invalid resident command"),
        }
    }
}

enum CachedState {
    Complete {image:crate::dspark_generation::StateImage,pending:u32},
    Encoder {image:crate::target::EncoderPrefixImage,history:Option<crate::history_lookup::HistoryLookup>,policy:crate::dspark_generation::ProposalPolicyImage},
}
struct CachedSession {
    prefix:Vec<u32>,
    media:crate::prompt_media::Identity,
    image:CachedState,
    charge:usize,
    prompt_logits:Option<Buffer<f32>>,
}
impl CachedSession {
    fn opportunistic_logits(&self)->bool {
        self.prompt_logits.is_some() && self.media.placements().is_empty()
    }
}

/// Keep room for the current conversation's checkpoint and completed turn.
/// Protect up to two longer parents; use oldest-first replacement in the
/// remaining slots. A pure longest-prefix policy lets slightly longer stale
/// completions permanently exclude every new prompt checkpoint.
/// Entries are ordered by insertion/restoration, so cache hits refresh age.
fn cache_evictions(entries:&[(usize,usize)], incoming:(usize,usize), budget:usize,
    long_first:bool,slots:usize)->Option<Vec<usize>> {
    if incoming.1>budget || slots==0 {return None;}
    let mut protected=if long_first {
        (0..entries.len()).filter(|&i|entries[i].0>incoming.0).collect::<Vec<_>>()
    }else{Vec::new()};
    protected.sort_by_key(|&i|std::cmp::Reverse((entries[i].0,i)));
    protected.truncate(2.min(slots.saturating_sub(2)));
    let mut retained=(0..entries.len()).collect::<Vec<_>>();let mut removed=Vec::new();
    while retained.len()>=slots || retained.iter().map(|&i|entries[i].1).sum::<usize>()+incoming.1>budget {
        let victim=*retained.iter().find(|&&i|!protected.contains(&i))?;
        retained.retain(|&i|i!=victim);removed.push(victim);
    }
    Some(removed)
}

/// Image prompts have paid for visual prefill and use the ordinary bounded
/// conversation policy. Text-only exact-logit snapshots remain spare-capacity
/// optimizations, so they never displace an existing conversation.
fn prompt_cache_evictions(entries:&[(usize,usize)],incoming:(usize,usize),budget:usize,
    long_first:bool,slots:usize,has_media:bool)->Option<Vec<usize>> {
    let evictions=cache_evictions(entries,incoming,budget,long_first,slots)?;
    (has_media || evictions.is_empty()).then_some(evictions)
}

/// Optional text-only final-logit images yield before the existing prefix and
/// conversation cache. Image prompts retain ordinary conversation priority.
fn cache_evictions_with_optional(entries:&[(usize,usize)], optional:&[bool],
    incoming:(usize,usize),budget:usize,long_first:bool,slots:usize)->Option<Vec<usize>> {
    if incoming.1>budget || slots==0 || optional.len()!=entries.len() {return None;}
    let mut retained=(0..entries.len()).collect::<Vec<_>>();let mut removed=Vec::new();
    while retained.len()>=slots || retained.iter().map(|&i|entries[i].1).sum::<usize>()+incoming.1>budget {
        let Some(victim)=retained.iter().copied().filter(|&i|optional[i])
            .min_by_key(|&i|(entries[i].0,i)) else {break;};
        retained.retain(|&i|i!=victim);removed.push(victim);
    }
    let rest=retained.iter().map(|&i|entries[i]).collect::<Vec<_>>();
    removed.extend(cache_evictions(&rest,incoming,budget,long_first,slots)?.into_iter().map(|i|retained[i]));
    Some(removed)
}

/// The restored parent is a live reservation, not an eviction candidate.
/// A successor must coexist with it even during allocation. Otherwise a late
/// refusal or cancellation could destroy the only reusable shared prefix.
fn encoder_checkpoint_evictions(entries:&[(usize,usize)], optional:&[bool],
    incoming:(usize,usize),parent_charge:Option<usize>,budget:usize,
    long_first:bool,slots:usize)->Option<Vec<usize>> {
    cache_evictions_with_optional(entries,optional,incoming,
        budget.checked_sub(parent_charge.unwrap_or(0))?,long_first,
        slots.checked_sub(usize::from(parent_charge.is_some()))?)
}

/// Only completed cache entries participate. Borrowed parents and unfinished
/// request images are absent. Prefer obsolete copies of this conversation;
/// preserve the ordinary long-parent protection for unrelated short requests.
fn pressure_cache_victim(entries:&[(usize,usize)], optional:&[bool], superseded:&[bool],
    incoming_tokens:usize,slots:usize,long_first:bool,removed:&[usize])->Option<usize> {
    if entries.len()!=optional.len() || entries.len()!=superseded.len(){return None;}
    let mut protected=if long_first {
        (0..entries.len()).filter(|&i|entries[i].0>incoming_tokens&&!removed.contains(&i))
            .collect::<Vec<_>>()
    }else{Vec::new()};
    protected.sort_by_key(|&i|std::cmp::Reverse((entries[i].0,i)));
    protected.truncate(2.min(slots.saturating_sub(2)));
    (0..entries.len()).filter(|i|!removed.contains(i)
        &&(optional[*i]||superseded[*i]||!protected.contains(i)))
        .min_by_key(|&i|if optional[i] {(0,entries[i].0,i)}
            else if superseded[i] {(1,usize::MAX-entries[i].0,i)}else{(2,0,i)})
}

struct CacheSaveAdmission {
    admitted:bool,available:usize,local_fits:bool,evicted:usize,
    pressure_evicted:usize,forecast_refused:bool,
}

/// Reclaim only completed, optional conversation snapshots. The caller owns
/// the live session and paused-request images separately. Headroom may differ
/// by rank (only rank zero runs vision); all ranks still evict the same entry.
fn reclaim_optional_cache(cuda:&Cuda,control:&mut ControlChannel,
    cached:&mut Vec<CachedSession>,long_first:bool,headroom:usize)->Result<Value> {
    let began=Instant::now();
    cuda.synchronize_all()?;
    let mut proof=b"optional-cache-reclaim-v1".to_vec();
    proof.extend_from_slice(&(cached.len() as u64).to_le_bytes());
    for entry in cached.iter() {
        proof.extend_from_slice(&(entry.prefix.len() as u64).to_le_bytes());
        proof.extend_from_slice(&(entry.charge as u64).to_le_bytes());
        proof.push(u8::from(entry.opportunistic_logits()));
    }
    control.agree(&proof)?;
    let before=cuda.admission_memory()?;
    let mut admitted=control.all_admit(before.admitted_available > headroom)?;
    let mut evicted=Vec::new();
    while !admitted && !cached.is_empty() {
        let index=cached.iter().enumerate().min_by_key(|(i,c)|
            (!c.opportunistic_logits(),if long_first {c.prefix.len()}else{0},*i)).unwrap().0;
        let entry=cached.remove(index);
        evicted.push(json!({"prefix_tokens":entry.prefix.len(),"charge_bytes":entry.charge}));
        drop(entry);
        admitted=control.all_admit(cuda.admission_memory()?.admitted_available > headroom)?;
    }
    Ok(json!({"headroom_bytes":headroom,"admitted":admitted,"evicted":evicted,
        "remaining_cache_entries":cached.len(),"memory_before":before,
        "memory_after":cuda.admission_memory()?,"seconds":began.elapsed().as_secs_f64()}))
}

struct Worker {
    // Captured graph owners must drop before their NCCL/RoCE communicator,
    // including when any request, socket operation or output write fails.
    session: Option<GreedySession>,
    cached_sessions: Vec<CachedSession>,
    #[cfg(feature = "storage-fault-injection")]
    cache_save_fault: cache_admission_fault::Fault,
    session_cache_budget: usize,
    shared_image_budget: usize,
    suspended_image_bytes: usize,
    suspended_image_count: usize,
    loaded_request_is_suspended: bool,
    session_cache_long_first: bool,
    cache_pressure_reclaim: bool,
    prompt_checkpoint: bool,
    encoder_prompt_checkpoint: bool,
    encoder_active_frontier: bool,
    encoder_upgrade_reconstructed: bool,
    prompt_checkpoint_tail: usize,
    prompt_cache_logits: bool,
    retain_restored_checkpoint: bool,
    prepared_session: Option<PreparedSession>,
    session_prepare_seconds: f64,
    nccl_init_seconds: f64,
    target_lineage: u64,
    verify: Target,
    long_verify: Option<Target>,
    long_history_buffers: Option<LongHistoryBuffers>,
    wides: Vec<Target>,
    large: Target,
    comm: Roce,
    control: ControlChannel,
    stream: Stream,
    cuda: Cuda,
    cfg: ModelConfig,
    core: Vec<u8>,
    history: ResidentHistory,
    history_lookup_enabled: bool,
    progress_next: bool,
    current: Option<u64>,
    pending_prefill: Option<PendingPrefill>,
    prefill_cancelled: bool,
    media: Option<crate::worker_media::Media>,
    active_media: Option<std::rc::Rc<crate::prompt_media::Bindings>>,
    media_receipt: Value,
    background_prefill_enabled: bool,
    emitted: usize,
    limit: usize,
    identity: String,
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.history.retire();
        // Close CPU control before destroying collective objects, so peers
        // leave a blocked control read even if communicator teardown is slow.
        self.control.retire();
    }
}

impl Worker {
    fn prompt_identity(&self, images:&[crate::image_cache::Descriptor], tokens:&[u32])
        ->Result<crate::prompt_media::Identity> {
        if images.is_empty() {return crate::prompt_media::Identity::new(Vec::new(),tokens);}
        self.media.as_ref().ok_or_else(||anyhow::anyhow!("vision is not configured"))?.identity(images,tokens)
    }
    fn install_media(&mut self, media:Option<std::rc::Rc<crate::prompt_media::Bindings>>) -> Result<()> {
        if self.active_media.is_none() && media.is_none() {return Ok(());}
        self.stream.synchronize()?;
        self.large.clear_prompt_media_rows(&self.stream)?;
        self.verify.clear_prompt_media_rows(&self.stream)?;
        if let Some(target)=&mut self.long_verify {target.clear_prompt_media_rows(&self.stream)?;}
        for target in &mut self.wides {target.clear_prompt_media_rows(&self.stream)?;}
        if self.verify.profile_is_current() {self.verify.set_prompt_media(media.clone(),&self.stream)?;}
        else {self.large.set_prompt_media(media.clone(),&self.stream)?;}
        self.active_media=media;
        Ok(())
    }
    fn prepare_media(&mut self, images:&[crate::image_cache::Descriptor], plan:&RequestPlan, frontier:usize)
        ->Result<()> {
        self.install_media(None)?;
        self.media_receipt=Value::Null;
        if !images.is_empty() {
            let cuda=&self.cuda;let cached=&mut self.cached_sessions;
            let long_first=self.session_cache_long_first;
            let (bindings,receipt)=self.media.as_mut().ok_or_else(||anyhow::anyhow!("vision is not configured"))?
                .prepare(plan.media.clone(),images,frontier,&mut self.control,&self.comm,&self.stream,
                    &mut |control,headroom|reclaim_optional_cache(cuda,control,cached,long_first,headroom))?;
            self.install_media(Some(bindings))?;
            self.media_receipt=receipt;
        }
        Ok(())
    }
    /// Called only when media preparation refuses capacity, before any new
    /// prompt tokens enter the model. A cache restore may have changed the
    /// retained owner, but its GPU state and committed history must still agree.
    fn refuse_media_input(&mut self,id:u64,error:anyhow::Error)->Result<Value> {
        let Some(capacity)=error.downcast_ref::<crate::memory::InputCapacityError>() else {
            return Err(error);
        };
        let message=capacity.0;let headroom=capacity.1;
        ensure!(self.current.is_none() && self.pending_prefill.is_none()
            && if self.session.is_some() {self.verify.profile_is_current()
                && self.verify.position()==self.history.frontier() as u64}
                else {self.history.frontier()==0},"input refusal requires an unchanged committed owner");
        self.install_media(None)?;
        if let Some(media)=&mut self.media {media.clear_inputs();}
        self.history.finish(Finish::Cancelled)?;
        self.agree(&[])?;
        Ok(json!({"type":"input_refused","request_id":id,"reason":"input_capacity",
            "message":message,"headroom_bytes":headroom,"frontier":self.history.frontier(),
            "remaining_cache_entries":self.cached_sessions.len(),"memory":self.cuda.admission_memory()?}))
    }
    /// Small policy intervention on the real serving worker. Uniform schedules
    /// are restored before Finish/Begin; no saved-session layout is changed.
    fn root_temperature_screen(&mut self,rank:usize,output:&Path,startup:Value)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_ROOT_TEMPERATURE_SCREEN").as_deref()==Ok("1")
            && self.current.is_none() && self.history.frontier()==0
            && self.session_cache_budget==0 && self.cached_sessions.is_empty()
            && !self.prompt_checkpoint && !self.prompt_cache_logits && self.long_verify.is_some(),
            "root-only temperature probe requires empty finite worker without images");
        let raw=include_bytes!("../tests/fixtures/agentic-exploration-tokens-v1.json");
        let fixture:Value=serde_json::from_slice(raw)?;
        ensure!(fixture["thinking_mode"]=="thinking" && fixture["reasoning_effort"]=="high",
            "root-temperature canonical high-reasoning fixtures");
        let mut records=Vec::new();let mut request_id=1u64;
        let mut progress=std::fs::OpenOptions::new().create_new(true).write(true)
            .open(output.join("root-temperature-records.jsonl"))?;
        for (ordinal,id) in ["typescript-abort-race","sqlite-migration-plan","incident-triage-evidence"].into_iter().enumerate() {
            let case=fixture["cases"].as_array().unwrap().iter().find(|c|c["id"]==id)
                .ok_or_else(||anyhow::anyhow!("missing root-temperature fixture"))?;
            let prompt:Vec<u32>=serde_json::from_value(case["input_ids"].clone())?;
            ensure!((128..=512).contains(&prompt.len()),"bounded root-temperature prefix");
            // Two short, excluded warmups capture both actual-q proposal graphs.
            // Two measured seeds then reverse order, without fitting a selector.
            for repeat in 0usize..if ordinal==0 {3}else{2} {
                let warmup=ordinal==0 && repeat==0;
                let measured_repeat=if ordinal==0 {repeat.saturating_sub(1)}else{repeat};
                let seed=202674900+(ordinal*2+measured_repeat)as u64;
                for warmer in if measured_repeat==0 {[false,true]}else{[true,false]} {
                    let cap=if warmup{64}else{256};
                    self.control.agree(&[ordinal as u8,repeat as u8,warmer as u8])?;
                    let begin=self.execute(Command::Begin{images: Vec::new(),id:request_id,seed,greedy:false,limit:cap,tokens:prompt.clone()})?;
                    request_id+=1;
                    ensure!(begin["cached_tokens"]==0 && begin["target_temperature"]==1.
                        && begin["target_top_p"]==1. && begin["block_verification"]==true,
                        "root-temperature must retain fresh prefix and target sampler");
                    self.session.as_mut().unwrap().set_root_temperature_for_probe(warmer,&self.stream)?;
                    self.control.agree(b"root-temperature-timing-start")?;
                    let started=Instant::now();let mut steps=Vec::new();
                    let mut output_ids:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;
                    while self.emitted<self.limit && self.history.pending()!=Some(self.cfg.eos_token_id) {
                        let step=self.execute(Command::Step)?;
                        output_ids.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);
                        steps.push(step);
                    }
                    let decode_seconds=started.elapsed().as_secs_f64();
                    self.session.as_mut().unwrap().set_root_temperature_for_probe(false,&self.stream)?;
                    let stop=self.history.pending()==Some(self.cfg.eos_token_id);
                    let finish=self.execute(Command::Finish(if stop{Finish::Stop}else{Finish::Length}))?;
                    ensure!(self.history.frontier()==prompt.len()+output_ids.len()-1
                        && self.history.pending()==output_ids.last().copied()
                        && output_ids.len()==self.emitted && (stop || self.emitted==cap),
                        "root-temperature committed frontier");
                    let record=json!({"case":id,"family":case["family"],"repeat":measured_repeat,
                        "warmup":warmup,"seed":seed,"warmer_root":warmer,"cap":cap,
                        "draft_temperatures":if warmer{[1.,0.7,0.7,0.7,0.7]}else{[0.7;5]},
                        "conditioning_split":self.session.as_ref().unwrap().conditioning_split(),
                        "prompt_tokens":prompt.len(),"output_ids":output_ids,"decode_seconds":decode_seconds,
                        "begin":begin,"finish":finish,"steps":steps,"uniform_policy_restored":true});
                    writeln!(progress,"{}",record)?;progress.flush()?;
                    eprintln!("root-temperature rank{rank} {id} repeat{measured_repeat} warmer={warmer} warmup={warmup}");
                    records.push(record);
                }
            }
        }
        self.execute(Command::Shutdown)?;
        std::fs::write(output.join("root-temperature-summary.json"),serde_json::to_vec_pretty(&json!({
            "rank":rank,"records":records,"startup_warmup":startup,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
            "all_committed_frontiers_checked":true,"saved_session_layout_changed":false,
            "target_temperature":1.,"target_top_p":1.,"serving_policy_changed":false,
            "scope":"Three public high-reasoning task families, two seeds,256-token caps, actual serving M6/M16 and split drafter. Root T1 versus uniformT.7; later proposals T.7. Two excluded64-token warmups. Foreground-only intervention, not completed-task quality or generalization proof."}))?)?;
        Ok(())
    }

    fn root_feedback_screen(&mut self,rank:usize,output:&Path,startup:Value)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_ROOT_FEEDBACK_SCREEN").as_deref()==Ok("1")
            && self.current.is_none() && self.history.frontier()==0
            && self.session_cache_budget==0 && self.cached_sessions.is_empty()
            && !self.prompt_checkpoint && !self.prompt_cache_logits && self.long_verify.is_some(),
            "root-only temperature probe requires empty finite worker without images");
        let raw=include_bytes!("../tests/fixtures/root-feedback-exploration-tokens-v1.json");
        let fixture:Value=serde_json::from_slice(raw)?;
        ensure!(fixture["thinking_mode"]=="thinking" && fixture["reasoning_effort"]=="high",
            "root-feedback canonical high-reasoning fixtures");
        let mut records=Vec::new();let mut request_id=1u64;
        let mut progress=std::fs::OpenOptions::new().create_new(true).write(true)
            .open(output.join("root-feedback-records.jsonl"))?;
        for (ordinal,id) in ["go-coalescing-cancellation","javascript-incremental-ndjson","library-workshop-tradeoffs"].into_iter().enumerate() {
            let case=fixture["cases"].as_array().unwrap().iter().find(|c|c["id"]==id)
                .ok_or_else(||anyhow::anyhow!("missing root-feedback fixture"))?;
            let prompt:Vec<u32>=serde_json::from_value(case["input_ids"].clone())?;
            ensure!((128..=512).contains(&prompt.len()),"bounded root-feedback prefix");
            // Two short, excluded warmups capture both actual-q proposal graphs.
            // Then compare adaptive feedback against fixed T.7 on two reversed seeds.
            for repeat in 0usize..if ordinal==0 {3}else{2} {
                let warmup=ordinal==0 && repeat==0;
                let measured_repeat=if ordinal==0 {repeat.saturating_sub(1)}else{repeat};
                let seed=202676200+(ordinal*2+measured_repeat)as u64;
                for warmer in if measured_repeat==0 {[false,true]}else{[true,false]} {
                    let cap=if warmup{64}else{384};
                    self.control.agree(&[ordinal as u8,repeat as u8,warmer as u8])?;
                    let begin=self.execute(Command::Begin{images: Vec::new(),id:request_id,seed,greedy:false,limit:cap,tokens:prompt.clone()})?;
                    request_id+=1;
                    ensure!(begin["cached_tokens"]==0 && begin["target_temperature"]==1.
                        && begin["target_top_p"]==1. && begin["block_verification"]==true,
                        "root-feedback must retain fresh prefix and target sampler");
                    if warmup {
                        self.session.as_mut().unwrap().set_root_temperature_for_probe(warmer,&self.stream)?;
                    } else {
                        self.session.as_mut().unwrap().set_root_feedback_for_probe(warmer,&self.stream)?;
                    }
                    self.control.agree(b"root-feedback-timing-start")?;
                    let started=Instant::now();let mut steps=Vec::new();
                    let mut output_ids:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;
                    while self.emitted<self.limit && self.history.pending()!=Some(self.cfg.eos_token_id) {
                        let step=self.execute(Command::Step)?;
                        output_ids.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);
                        steps.push(step);
                    }
                    let decode_seconds=started.elapsed().as_secs_f64();
                    let feedback=self.session.as_ref().unwrap().root_feedback_report()?;
                    self.session.as_mut().unwrap().set_root_feedback_for_probe(false,&self.stream)?;
                    let stop=self.history.pending()==Some(self.cfg.eos_token_id);
                    let finish=self.execute(Command::Finish(if stop{Finish::Stop}else{Finish::Length}))?;
                    ensure!(self.history.frontier()==prompt.len()+output_ids.len()-1
                        && self.history.pending()==output_ids.last().copied()
                        && output_ids.len()==self.emitted && (stop || self.emitted==cap),
                        "root-feedback committed frontier");
                    let record=json!({"case":id,"family":case["family"],"repeat":measured_repeat,
                        "warmup":warmup,"seed":seed,"warmer_root":warmer,"cap":cap,
                        "adaptive_root_feedback":warmer&&!warmup,"feedback":feedback,
                        "conditioning_split":self.session.as_ref().unwrap().conditioning_split(),
                        "prompt_tokens":prompt.len(),"output_ids":output_ids,"decode_seconds":decode_seconds,
                        "begin":begin,"finish":finish,"steps":steps,"uniform_policy_restored":true});
                    writeln!(progress,"{}",record)?;progress.flush()?;
                    eprintln!("root-feedback rank{rank} {id} repeat{measured_repeat} warmer={warmer} warmup={warmup}");
                    records.push(record);
                }
            }
        }
        self.execute(Command::Shutdown)?;
        std::fs::write(output.join("root-feedback-summary.json"),serde_json::to_vec_pretty(&json!({
            "rank":rank,"records":records,"startup_warmup":startup,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
            "all_committed_frontiers_checked":true,"saved_session_layout_changed":false,
            "target_temperature":1.,"target_top_p":1.,"serving_policy_changed":false,
            "scope":"Three public high-reasoning task families, two seeds,384-token caps, actual serving M6/M16 and split drafter. Past-root GPU overlap feedback versus uniformT.7; selection affects only the next round root, later proposals T.7. Two excluded64-token warmups. Foreground-only intervention, not completed-task quality or generalization proof."}))?)?;
        Ok(())
    }

    fn load(p: &WorkerConfig) -> Result<Self> {
        let cache_pressure_reclaim=match std::env::var("CANADA_SCRATCH_CACHE_PRESSURE_RECLAIM") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid cache pressure policy");value=="1"},
            Err(std::env::VarError::NotPresent)=>true,
            Err(error)=>return Err(error.into()),
        };
        let session_cache_budget=match std::env::var("CANADA_SCRATCH_SESSION_CACHE_BYTES") {
            Err(std::env::VarError::NotPresent)=>0,
            Ok(value)=>value.parse::<usize>()?,
            Err(error)=>return Err(error.into()),
        };
        ensure!(session_cache_budget<=2*1024usize.pow(3)
            && (session_cache_budget==0 || std::env::var("CANADA_SCRATCH_COMPACT_STATE_IMAGE").as_deref()==Ok("1")),
            "session cache requires compact images and at most2GiB additional state");
        let shared_image_budget=match std::env::var("CANADA_SCRATCH_SHARED_IMAGE_BYTES") {
            Err(std::env::VarError::NotPresent)=>2*1024usize.pow(3),
            Ok(value)=>value.parse::<usize>()?,
            Err(error)=>return Err(error.into()),
        };
        ensure!((64*1024usize.pow(2)..=2*1024usize.pow(3)).contains(&shared_image_budget),
            "shared inactive image budget must be64MiB..2GiB");
        let session_cache_long_first=match std::env::var("CANADA_SCRATCH_SESSION_CACHE_POLICY") {
            Err(std::env::VarError::NotPresent)=>true,
            Ok(value)if value=="fifo"=>false,
            Ok(value)if value=="long_parents_and_recent" || value=="longest_prefix"=>true,
            _=>anyhow::bail!("session cache policy must be fifo or long_parents_and_recent"),
        };
        let prompt_checkpoint=match std::env::var("CANADA_SCRATCH_PROMPT_CHECKPOINT") {
            Err(std::env::VarError::NotPresent)=>false,
            Ok(value)if value=="0"=>false,
            Ok(value)if value=="1"=>true,
            _=>anyhow::bail!("prompt checkpoint must be0or1"),
        };
        ensure!(!prompt_checkpoint || session_cache_budget>0,
            "prompt checkpoints require the bounded compact session cache");
        let encoder_prompt_checkpoint=match std::env::var("CANADA_SCRATCH_ENCODER_PROMPT_CHECKPOINT") {
            Err(std::env::VarError::NotPresent)=>false,
            Ok(v)if v=="0"=>false,Ok(v)if v=="1"=>true,
            _=>anyhow::bail!("encoder prompt checkpoint must be0or1"),
        };
        ensure!(!encoder_prompt_checkpoint||prompt_checkpoint,"encoder checkpoint needs the prompt cache");
        let encoder_active_frontier=match std::env::var("CANADA_SCRATCH_ENCODER_ACTIVE_FRONTIER") {
            // s787/s792: avoid splitting a true active append merely to move
            // its recovery checkpoint. This matches ordinary unsplit execution;
            // a later branch may replay a longer suffix from the older image.
            Err(std::env::VarError::NotPresent)=>encoder_prompt_checkpoint,
            Ok(v)if v=="0"=>false,Ok(v)if v=="1"=>true,
            _=>anyhow::bail!("encoder active frontier must be0or1"),
        };
        ensure!(!encoder_active_frontier||encoder_prompt_checkpoint,"active frontier requires encoder checkpoints");
        let prompt_checkpoint_tail=match std::env::var("CANADA_SCRATCH_PROMPT_CHECKPOINT_TAIL") {
            Err(std::env::VarError::NotPresent)=>128,
            Ok(value)if value=="128"=>128,
            Ok(value)if value=="1"=>1,
            _=>anyhow::bail!("prompt checkpoint tail must be1or128"),
        };
        ensure!(!encoder_prompt_checkpoint||prompt_checkpoint_tail==128,"encoder checkpoint uses the aligned128-tail policy");
        ensure!(prompt_checkpoint_tail==128 || prompt_checkpoint,
            "one-token checkpoint replay requires prompt checkpoints");
        let prompt_cache_logits=match std::env::var("CANADA_SCRATCH_PROMPT_CACHE_LOGITS") {
            Err(std::env::VarError::NotPresent)=>false,
            Ok(value)if value=="0"=>false,
            Ok(value)if value=="1"=>true,
            _=>anyhow::bail!("prompt logit cache must be0or1"),
        };
        ensure!(!prompt_cache_logits || (prompt_checkpoint && prompt_checkpoint_tail==128),
            "prompt logit cache requires the shared aligned checkpoint policy");
        let retain_restored_checkpoint=match std::env::var("CANADA_SCRATCH_RETAIN_RESTORED_CHECKPOINT") {
            Err(std::env::VarError::NotPresent)=>false,
            Ok(value)if value=="0"=>false,
            Ok(value)if value=="1"=>true,
            _=>anyhow::bail!("restored checkpoint retention must be0or1"),
        };
        ensure!(!retain_restored_checkpoint || prompt_checkpoint,
            "retaining restored images requires prompt checkpoints");
        let encoder_upgrade_reconstructed=match std::env::var("CANADA_SCRATCH_ENCODER_UPGRADE_RECONSTRUCTED") {
            // s794: retain the complete state already reconstructed for a
            // short branch. Saving costs about4ms and avoids a later~84ms
            // decoder reconstruction, within the same cache/reserve budgets.
            Err(std::env::VarError::NotPresent)=>encoder_prompt_checkpoint&&retain_restored_checkpoint,
            Ok(v)if v=="0"=>false,Ok(v)if v=="1"=>true,
            _=>anyhow::bail!("reconstructed checkpoint upgrade must be0or1"),
        };
        ensure!(!encoder_upgrade_reconstructed||(encoder_prompt_checkpoint&&retain_restored_checkpoint),
            "reconstructed upgrade requires retained encoder checkpoints");
        ensure!(
            p.rank < 4
                && crate::config::agentic_context(p.context, true)
                && p.context >= 32768
                && !p.wide_widths.is_empty()
                && p.wide_widths
                    .iter()
                    .all(|&w| crate::config::wide_rows(w) && w <= p.context)
                && p.wide_widths.windows(2).all(|w| w[0] < w[1]),
            "resident worker profile admission"
        );
        ensure!(
            std::env::var("CANADA_SCRATCH_STOCHASTIC_DRAFT").as_deref() == Ok("1"),
            "resident worker requires target-correct stochastic DSpark"
        );
        let history_lookup_enabled = match std::env::var("CANADA_SCRATCH_HISTORY_LOOKUP") {
            Ok(value) => {
                ensure!(value == "0" || value == "1", "invalid resident history lookup policy");
                value == "1"
            }
            Err(std::env::VarError::NotPresent) => false,
            Err(error) => return Err(error.into()),
        };
        let policy_value = |name| -> Result<Option<String>> {
            match std::env::var(name) {
                Ok(value) => Ok(Some(value)),
                Err(std::env::VarError::NotPresent) => Ok(None),
                Err(error) => Err(error.into()),
            }
        };
        if let Some(value)=policy_value("CANADA_SCRATCH_ENGRAM_PREFETCH")? {
            ensure!(["0","1"].contains(&value.as_str()),"invalid resident Engram lookahead policy");
            ensure!(value=="0" || std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref()==Ok("1"),
                "resident lookahead requires admitted io_uring rows");
        }
        let sparse_flag = policy_value("CANADA_SCRATCH_SPARSE_INDEX")?;
        let adaptive_flag = policy_value("CANADA_SCRATCH_SPARSE_INDEX_ADAPTIVE")?;
        let adaptive_sparse = adaptive_sparse_policy(sparse_flag.as_deref(), adaptive_flag.as_deref())?;
        // Neural and exact-token history proposals use the existing target-
        // correct sampler. Other policies require independent qualification.
        for name in [
            "CANADA_SCRATCH_COMPACT_INDEX_AB",
            "CANADA_SCRATCH_LONG_RETAINED_PROBE",
            "CANADA_SCRATCH_ENGRAM_OVERLAP_AB",
            "CANADA_SCRATCH_STATE_IMAGE_PROBE",
            "CANADA_SCRATCH_RAPID_SCHEDULE_SCREEN",
            "CANADA_SCRATCH_SORT_SELECTED_AB",
            "CANADA_SCRATCH_GENERATION_SCHEDULE_AB",
            "CANADA_SCRATCH_CAUSAL_VERIFY_SCREEN",
            "CANADA_SCRATCH_PREFILL_NORM_FUSION_AB",
            "CANADA_SCRATCH_ATTENTION_Q_OVERLAP",
            "CANADA_SCRATCH_STOCHASTIC_PREFIX_SCHEDULE",
            "CANADA_SCRATCH_COST_AWARE_VERIFY",
            "CANADA_SCRATCH_PROFILE_VERIFY",
            "CANADA_SCRATCH_PROFILE_WIDE",
        ] {
            ensure!(
                std::env::var_os(name).is_none(),
                "resident pilot has not qualified {name}"
            );
        }
        let config_bytes = std::fs::read(&p.config)?;
        let manifest_bytes = std::fs::read(&p.checkpoint)?;
        let cfg = ModelConfig::parse(&config_bytes)?;
        let cp = Checkpoint::from_manifest(&p.checkpoint)?;
        let plan = RankPlan::build(&cp, &cfg, p.rank, 4)?;
        let core_bytes = std::fs::read(&p.core)?;
        let hash_metadata = std::fs::read(&p.hash_metadata)?;
        let mut identity = Sha256::new();
        identity.update(Sha256::digest(&config_bytes));
        identity.update(Sha256::digest(&manifest_bytes));
        identity.update(Sha256::digest(&core_bytes));
        identity.update(Sha256::digest(&hash_metadata));
        identity.update(Sha256::digest(std::fs::read(std::env::current_exe()?)?));
        identity.update(serde_json::to_vec(&(p.context, &p.wide_widths))?);
        if let Some(config)=&p.vision {
            identity.update(config.fingerprint()?);
            identity.update((config.cache_bytes as u64).to_le_bytes());
        }
        let mut flags: Vec<_> = std::env::vars()
            .filter(|(key, _)| key.starts_with("CANADA_SCRATCH_"))
            .collect();
        flags.sort();
        identity.update(serde_json::to_vec(&flags)?);
        let identity: [u8; 32] = identity.finalize().into();
        let cuda = Cuda::open(0)?;
        cuda.qualify_gb10()?;
        let stream = cuda.stream()?;
        let core = cuda.module(&core_bytes)?;
        let mut large = Target::load(
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
            HashMetadata::parse(&hash_metadata)?,
            1,
            Phase::Extend,
            128,
            p.context,
        )?;
        large.set_graph_replay(true)?;
        large.set_prefill_last_head(true)?;
        large.enable_decoder_tail(&core_bytes)?;
        large.enable_cached_extend_floor()?;
        let wides = large.fork_wide_bank(&core, &core_bytes, &stream, &p.kernels, &p.wide_widths, true)?;
        eprintln!("rank{} wide scratch admission: {}", p.rank,
            json!(wides.iter().filter_map(Target::wide_report).collect::<Vec<_>>()));
        for t in &wides {
            ensure!(
                large.shared_state_report(t)["all_weights_and_causal_storage_shared"] == true,
                "resident wide profile duplicated state"
            );
        }
        let mut verify =
            large.fork_profile(&core, &core_bytes, &stream, &p.kernels, 6, Phase::Verify)?;
        ensure!(verify.adaptive_sparse_index_layers() == if adaptive_sparse {4} else {0},
            "resident adaptive sparse construction differs from admitted policy");
        let compact=crate::query_index::compact_sparse_index_enabled(p.context,adaptive_sparse)?;
        ensure!(verify.compact_sparse_index_layers()==if compact {4}else{0},
            "resident compact index construction differs from admitted policy");
        ensure!(verify.cached_index_gather_layers()==if std::env::var("CANADA_SCRATCH_CACHED_INDEX_GATHER").as_deref()==Ok("1") {8}else{0},
            "resident gathered-prefix construction differs from admitted policy");
        ensure!(verify.fused_shared_quantization_layers()==if std::env::var("CANADA_SCRATCH_FUSED_SHARED_QUANT").as_deref()==Ok("1") {40}else{0},
            "resident shared fusion construction differs from admitted policy");
        verify.set_graph_replay(true)?;
        let schedule = verify.apply_candidate_schedule(&stream)?;
        eprintln!("rank{} candidate verifier schedule: {schedule}", p.rank);
        ensure!(
            large.shared_state_report(&verify)["all_weights_and_causal_storage_shared"] == true,
            "resident verifier duplicated state"
        );
        let use_long_history=match std::env::var("CANADA_SCRATCH_LONG_HISTORY_GENERATION") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid long history policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
        };
        let (long_verify,long_history_buffers)=if use_long_history {
            ensure!(p.context==1048576 && history_lookup_enabled,"long history needs full-capacity history lookup");
            let mut target=large.fork_profile(&core,&core_bytes,&stream,&p.kernels,16,Phase::Verify)?;
            target.set_graph_replay(true)?;
            ensure!(large.shared_state_report(&target)["all_weights_and_causal_storage_shared"]==true,
                "long history verifier duplicated model state");
            (Some(target),Some(LongHistoryBuffers::new(&cuda,&core_bytes)?))
        }else{(None,None)};
        let prepared_draft = QueryNetwork::for_target(
            &large,
            &cuda,
            &core_bytes,
            &stream,
            &cp,
            &plan,
            &p.dspark_staged,
            &p.kernels,
        )?;
        let prepare_started = Instant::now();
        let prepared_session = PreparedSession::new(&cuda, &core_bytes, prepared_draft)?
            .with_serving_policy(&stream)?;
        stream.synchronize()?;
        let session_prepare_seconds = prepare_started.elapsed().as_secs_f64();
        eprintln!("rank{} allocation boundary before vision: {}",p.rank,json!(cuda.admission_memory()?));
        let media=p.vision.as_ref().map(|config|crate::worker_media::Media::load(&cuda,p.rank,&cfg,config)).transpose()?;
        eprintln!("rank{} allocation boundary after vision: {}",p.rank,json!(cuda.admission_memory()?));
        ensure!(
            cuda.admission_memory()?.admitted_available >= 8 * 1024usize.pow(3),
            "resident profiles and drafter consumed the runtime reserve"
        );
        let ticket = std::fs::read(&p.ticket)?;
        let mut comm = Roce::local(&cuda, &p.kernels.join("roce-v0"), p.rank)?;
        comm.attach_gather(&p.kernels.join("roce-gather-v1"))?;
        let all = crate::rendezvous::exchange_blob_with_timeout(
            p.rank,
            p.endpoint,
            &ticket,
            &comm.blob()?,
            Duration::from_secs(600),
        )?;
        comm.connect(&all)?;
        let api = crate::collective::Nccl::load_qualified_prefill()?;
        let local = if p.rank == 0 {
            api.unique_id()?.bytes
        } else {
            [0u8; 128]
        };
        let mut endpoint = p.endpoint;
        endpoint.set_port(
            endpoint
                .port()
                .checked_add(2)
                .ok_or_else(|| anyhow::anyhow!("resident endpoint overflow"))?,
        );
        let all = crate::rendezvous::exchange_blob_with_timeout(
            p.rank,
            endpoint,
            &ticket,
            &local,
            Duration::from_secs(600),
        )?;
        let nccl_started = Instant::now();
        comm.attach_prefill_sum(api.join(
            &cuda,
            crate::collective::UniqueId {
                bytes: all[..128].try_into().unwrap(),
            },
            p.rank,
            4,
        )?)?;
        let nccl_init_seconds = nccl_started.elapsed().as_secs_f64();
        comm.attach_precise_prefill(&core_bytes,p.wide_widths.iter().copied().max().unwrap_or(4096).max(4096))?;
        if std::env::var("CANADA_SCRATCH_ENGRAM_NCCL_GATHER").as_deref() == Ok("1") {
            comm.attach_prefill_column_gather(&core_bytes, p.wide_widths.iter().copied().max().unwrap_or(4096).max(4096))?;
        }
        endpoint.set_port(
            endpoint
                .port()
                .checked_add(2)
                .ok_or_else(|| anyhow::anyhow!("resident endpoint overflow"))?,
        );
        let rank_ips =
            std::array::from_fn(|rank| IpAddr::V4(Ipv4Addr::new(192, 168, 200, 12 + rank as u8)));
        let mut control = ControlChannel::connect(
            p.rank,
            endpoint,
            rank_ips,
            &ticket,
            &identity,
            Duration::from_secs(600),
        )?;
        control.agree(&(session_cache_budget as u64).to_le_bytes())?;
        control.agree(&(shared_image_budget as u64).to_le_bytes())?;
        control.agree(&[u8::from(session_cache_long_first)])?;
        control.agree(&[u8::from(cache_pressure_reclaim)])?;
        control.agree(&[u8::from(prompt_checkpoint)])?;
        control.agree(&[u8::from(encoder_prompt_checkpoint)])?;
        control.agree(&[u8::from(encoder_active_frontier)])?;
        control.agree(&[u8::from(encoder_upgrade_reconstructed)])?;
        control.agree(&[u8::from(crate::target::defer_checkpoint_decoder128()?)])?;
        control.agree(&(prompt_checkpoint_tail as u64).to_le_bytes())?;
        control.agree(&[u8::from(prompt_cache_logits)])?;
        control.agree(&[u8::from(retain_restored_checkpoint)])?;
        control.agree(&[u8::from(use_long_history)])?;
        #[cfg(feature = "storage-fault-injection")]
        control.agree(&[u8::from(std::env::var("CANADA_SCRATCH_CACHE_RETENTION_SCREEN").as_deref()==Ok("1"))])?;
        let history = ResidentHistory::new(p.context, cfg.eos_token_id)?;
        Ok(Self {
            session: None,
            cached_sessions:Vec::new(),
            #[cfg(feature = "storage-fault-injection")]
            cache_save_fault: cache_admission_fault::Fault::default(),
            session_cache_budget,
            shared_image_budget,
            suspended_image_bytes:0,
            suspended_image_count:0,
            loaded_request_is_suspended:false,
            session_cache_long_first,
            cache_pressure_reclaim,
            prompt_checkpoint,
            encoder_prompt_checkpoint,
            encoder_active_frontier,
            encoder_upgrade_reconstructed,
            prompt_checkpoint_tail,
            prompt_cache_logits,
            retain_restored_checkpoint,
            prepared_session: Some(prepared_session),
            session_prepare_seconds,
            nccl_init_seconds,
            target_lineage: 1,
            verify,
            long_verify,
            long_history_buffers,
            wides,
            large,
            comm,
            control,
            stream,
            cuda,
            cfg,
            core: core_bytes,
            history,
            history_lookup_enabled,
            progress_next: false,
            current: None,
            pending_prefill: None,
            prefill_cancelled: false,
            media,
            active_media: None,
            media_receipt: Value::Null,
            background_prefill_enabled: match std::env::var("CANADA_SCRATCH_BACKGROUND_PREFILL") {
                Ok(v)=>{ensure!(v=="0"||v=="1","invalid background prefill policy");v=="1"},
                Err(std::env::VarError::NotPresent)=>false,Err(error)=>return Err(error.into()),
            },
            emitted: 0,
            limit: 0,
            identity: identity.iter().map(|x| format!("{x:02x}")).collect(),
        })
    }

    fn agreement_state(&self, tokens: &[u32]) -> Vec<u8> {
        let mut state = Vec::with_capacity(56);
        state.extend_from_slice(&self.current.unwrap_or(0).to_le_bytes());
        state.extend_from_slice(&(self.history.frontier() as u64).to_le_bytes());
        state.extend_from_slice(&(self.emitted as u64).to_le_bytes());
        state.extend_from_slice(&self.history.pending().unwrap_or(u32::MAX).to_le_bytes());
        state.push(u8::from(self.pending_prefill.is_some()));
        state.push(u8::from(self.prefill_cancelled));
        for token in tokens {
            state.extend_from_slice(&token.to_le_bytes());
        }
        state
    }
    fn agree(&mut self, tokens: &[u32]) -> Result<()> {
        self.control.agree(&self.agreement_state(tokens))
    }
    fn active_command(&self, cancelled: bool) -> Result<Command> {
        ensure!(self.current.is_some() && self.emitted <= self.limit,
            "next command requires an active bounded request");
        Ok(if self.prefill_cancelled {
            Command::Finish(Finish::Cancelled)
        } else if self.history.pending() == Some(self.cfg.eos_token_id) {
            Command::Finish(Finish::Stop)
        } else if self.emitted == self.limit {
            Command::Finish(Finish::Length)
        } else if cancelled {
            Command::Finish(Finish::Cancelled)
        } else {
            Command::Step
        })
    }
    /// The only commands admitted by a progress acknowledgment are the next
    /// step or a finish at the already agreed frontier. A new request still
    /// requires the full command protocol. Call before exposing any tokens.
    fn agree_progress(&mut self, event: &Value, cancelled: bool) -> Result<Vec<u8>> {
        ensure!(self.progress_next && event["type"] == "tokens",
            "progress agreement requires an unpublished token event");
        let tokens: Vec<u32> = serde_json::from_value(event["output_ids"].clone())?;
        let leader = if self.comm.rank() == 0 {
            Some(self.active_command(cancelled)?.encode()?)
        } else { None };
        let state = self.agreement_state(&tokens);
        let next = self.control.agree_and_next(&state, leader.as_deref())?;
        ensure!(matches!(Command::decode(&next)?, Command::Step | Command::Finish(_)),
            "progress acknowledgment cannot begin or retire a request");
        Ok(next)
    }

    /// Retain only a committed target/drafter/history image. Prompt checkpoints
    /// and displaced conversations share the same four-entry/2GiB budget.
    fn retain_cached_prefix(&mut self,prefix:&[u32],pending:u32,media:&crate::prompt_media::Identity)->Result<Value> {
        self.retain_cached_prefix_reserved(prefix,pending,media.through(prefix.len()),0,0)
    }

    fn cache_capacity(&self,extra_bytes:usize,extra_count:usize)->(usize,usize) {
        (self.session_cache_budget.min(self.shared_image_budget
            .saturating_sub(self.suspended_image_bytes.saturating_add(extra_bytes))),
            4usize.saturating_sub(self.suspended_image_count.saturating_add(extra_count)))
    }

    /// Plan against the tightest physical rank before dropping any optional
    /// image, then recheck actual memory after release. The estimate never
    /// authorizes allocation; ordinary reserve and late-save checks still do.
    #[allow(clippy::too_many_arguments)]
    fn admit_cache_save(&mut self,eligible:bool,evictions:Option<Vec<usize>>,prefix:&[u32],
        media:&crate::prompt_media::Identity,required:usize,slots:usize)->Result<CacheSaveAdmission> {
        let mut removed=if eligible {evictions.expect("eligible cache eviction plan")}else{Vec::new()};
        let budget_evictions=removed.len();
        if self.cache_pressure_reclaim {
            let available=self.cuda.admission_memory()?.admitted_available;
            // Ordinary saves need exactly the existing memory query and one
            // all-rank vote. Enter the fuller eviction agreement only when a
            // rank actually needs reclamation or a logical-budget eviction.
            if self.control.all_admit(eligible&&removed.is_empty()&&available>=required)? {
                return Ok(CacheSaveAdmission{admitted:true,available,local_fits:true,
                    evicted:0,pressure_evicted:0,forecast_refused:false});
            }
            let entries=self.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
            let optional=self.cached_sessions.iter().map(CachedSession::opportunistic_logits).collect::<Vec<_>>();
            let superseded=self.cached_sessions.iter().map(|c|c.prefix.len()<prefix.len()
                &&prefix.starts_with(&c.prefix)&&media.matches_prefix(&c.media,c.prefix.len())).collect::<Vec<_>>();
            self.control.agree(&serde_json::to_vec(&(eligible,&entries,&optional,&superseded,&removed,slots,prefix.len()))?)?;
            loop {
                let reclaim=removed.iter().map(|&i|entries[i].1).sum::<usize>();
                if self.control.all_admit(eligible&&available.saturating_add(reclaim)>=required)?{break;}
                let next=eligible.then(||pressure_cache_victim(&entries,&optional,&superseded,
                    prefix.len(),slots,self.session_cache_long_first,&removed)).flatten();
                let Some(next)=next else {
                    return Ok(CacheSaveAdmission{admitted:false,available,local_fits:available>=required,
                        evicted:0,pressure_evicted:0,forecast_refused:true});
                };
                removed.push(next);
            }
            if !removed.is_empty(){self.stream.synchronize()?;}
        }
        removed.sort_unstable();
        let evicted=removed.len();
        for i in removed.into_iter().rev(){self.cached_sessions.remove(i);}
        let available=self.cuda.admission_memory()?.admitted_available;
        let local_fits=available>=required;
        let admitted=self.control.all_admit(eligible&&local_fits)?;
        Ok(CacheSaveAdmission{admitted,available,local_fits,evicted,
            pressure_evicted:evicted-budget_evictions,forecast_refused:false})
    }

    /// All ranks either retain their read-only snapshot or release it. Only
    /// explicit pre-copy memory refusals are recoverable; CUDA, identity and
    /// transport failures still retire the worker group.
    fn try_save_session_image(&mut self,anchor:u32,budget:usize)
        ->Result<Option<crate::dspark_generation::StateImage>> {
        let image=match self.session.as_mut().unwrap().save_state_image(
            &self.verify,anchor,budget,&self.comm,&self.stream) {
            Ok(image)=>Some(image),
            Err(error) if error.downcast_ref::<crate::state_image::CapacityRefused>().is_some()=>None,
            Err(error)=>return Err(error),
        };
        // Actual allocation/metadata costs can exceed the byte estimate.
        // Keep the ordinary 16MiB preparation allowance after saving too.
        let fits=image.is_some() && self.cuda.admission_memory()?.admitted_available
            >=8*1024usize.pow(3)+16*1024usize.pow(2);
        if self.control.all_admit(fits)? {Ok(image)}else{Ok(None)}
    }

    /// Reclaim optional cache at a committed boundary. The live target/session
    /// and the scheduler's unfinished request images are never candidates.
    fn reclaim_input_space(&mut self, headroom:usize)->Result<Value> {
        let began=Instant::now();
        self.cuda.synchronize_all()?;
        self.agree(&[])?;
        let before=self.cuda.admission_memory()?;
        if !self.control.all_admit(before.admitted_available > headroom)? {
            if let Some(media)=&mut self.media {media.release_warm();}
        }
        let mut event=reclaim_optional_cache(&self.cuda,&mut self.control,&mut self.cached_sessions,
            self.session_cache_long_first,headroom)?;
        self.agree(&[])?;
        event["type"]=json!("input_reclaim");event["request_id"]=json!(self.current);
        event["frontier"]=json!(self.history.frontier());event["suspended_image_bytes"]=json!(self.suspended_image_bytes);
        event["memory_before"]=json!(before);event["seconds"]=json!(began.elapsed().as_secs_f64());
        Ok(event)
    }

    fn retain_cached_prefix_reserved(&mut self,prefix:&[u32],pending:u32,media:crate::prompt_media::Identity,
        extra_bytes:usize,extra_count:usize)->Result<Value> {
        ensure!(!prefix.is_empty() && self.verify.position()==prefix.len()as u64,
            "cached prefix and target frontier disagree");
        if self.cached_sessions.iter().any(|c|c.prefix==prefix && c.media.matches_prefix(&media,prefix.len())) {
            return Ok(json!({"saved":false,"already_cached":true,"saved_seconds":0.,
                "evicted_sessions":0,"capacity_refused":false}));
        }
        let (device_bytes,host_bytes)=self.session.as_ref().unwrap().compact_state_image_bytes(&self.verify)?;
        let charge=device_bytes.checked_add(host_bytes).and_then(|n|n.checked_add(prefix.len()*4+media.metadata_bytes()))
            .ok_or_else(||anyhow::anyhow!("cached session byte overflow"))?;
        let entries=self.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
        let (budget,slots)=self.cache_capacity(extra_bytes,extra_count);
        let optional=self.cached_sessions.iter().map(CachedSession::opportunistic_logits).collect::<Vec<_>>();
        let evictions=cache_evictions_with_optional(&entries,&optional,(prefix.len(),charge),
            budget,self.session_cache_long_first,slots);
        let eligible=device_bytes<=2*1024usize.pow(3) && evictions.is_some();
        // The wanted image remains alive during this check, so temporarily
        // retaining both current and wanted sessions is charged to real UMA.
        let reserve=8*1024usize.pow(3)+16*1024usize.pow(2);
        let admission=self.admit_cache_save(eligible,evictions,prefix,&media,reserve+charge,slots)?;
        let admitted=admission.admitted;
        let mut saved_seconds=0.;
        let mut saved=false;
        if admitted {
            let began=Instant::now();
            if let Some(image)=self.try_save_session_image(pending,device_bytes)? {
                ensure!(image.device_bytes()==device_bytes,"cached image estimate differs from allocation");
                self.cached_sessions.push(CachedSession{prefix:prefix.to_vec(),media,image:CachedState::Complete{image,pending},charge,prompt_logits:None});
                saved=true;
            }
            saved_seconds=began.elapsed().as_secs_f64();
        }
        Ok(json!({"saved":saved,"late_memory_refused":admitted&&!saved,"saved_seconds":saved_seconds,
            "evicted_sessions":admission.evicted,"capacity_refused":!eligible,
            "memory_refused":eligible&&!admitted,"local_memory_refused":eligible&&!admission.local_fits,
            "pressure_evicted_sessions":admission.pressure_evicted,"forecast_refused":admission.forecast_refused,
            "admission_available_bytes":admission.available,"admission_required_bytes":reserve+charge}))
    }

    /// Reserve an exact prompt image under the bounded cache policy. The
    /// logit row is copied while the prefill profile is still current; the
    /// complete causal image is saved just after handing ownership to Verify.
    fn prepare_prompt_logit_cache(&mut self,plan:&RequestPlan,greedy:bool,cache_hit:bool)
        ->Result<(Option<(Buffer<f32>,usize,usize)>,Value)> {
        let started=Instant::now();
        // A short image prompt still paid for visual prefill. Retain its exact
        // prompt state/logits under the conversation cache policy, using the
        // same full media identity and fresh target sampling as long prompts.
        if !self.prompt_cache_logits || greedy
            || (plan.prompt.len()<8192 && plan.media.placements().is_empty()) || (plan.reuse && !cache_hit) {
            return Ok((None,json!({"enabled":self.prompt_cache_logits,"saved":false,
                "active_append_skipped":plan.reuse && !cache_hit,"seconds":0.})));
        }
        if self.cached_sessions.iter().any(|c|c.prefix==plan.prompt && c.media.matches_prefix(&plan.media,plan.prompt.len()) && c.prompt_logits.is_some()) {
            return Ok((None,json!({"enabled":true,"saved":false,"already_cached":true,"seconds":0.})));
        }
        let (device,host)=self.session.as_ref().unwrap().compact_state_image_bytes(&self.large)?;
        let charge=device.checked_add(host).and_then(|n|n.checked_add(plan.prompt.len()*4+plan.media.metadata_bytes()))
            .and_then(|n|n.checked_add(crate::sampling::VOCAB*4))
            .ok_or_else(||anyhow::anyhow!("prompt logit cache size overflow"))?;
        let (budget,slots)=self.cache_capacity(0,0);
        let entries=self.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
        let evictions=prompt_cache_evictions(&entries,(plan.prompt.len(),charge),budget,
            self.session_cache_long_first,slots,!plan.media.placements().is_empty());
        let eligible=evictions.is_some();
        let reserve=8*1024usize.pow(3)+16*1024usize.pow(2);
        let admitted=self.control.all_admit(eligible
            && self.cuda.admission_memory()?.admitted_available>=reserve+charge)?;
        // Admission is conservative while all old allocations still exist.
        // A refused allocation leaves their cache ownership untouched.
        let mut evicted=0;
        let ready=if admitted {
            let mut indices=evictions.unwrap();indices.sort_unstable();evicted=indices.len();
            for i in indices.into_iter().rev(){self.cached_sessions.remove(i);}
            Some((self.large.snapshot_prompt_logits(&self.stream)?,device,charge))
        }else{None};
        Ok((ready,json!({"enabled":true,"saved":false,"admitted":admitted,
            "capacity_refused":!eligible,"evicted_sessions":evicted,"seconds":started.elapsed().as_secs_f64()})))
    }

    /// Retain a displaced conversation and restore a matching parent prefix.
    /// Active exact-prefix reuse takes the existing allocation-free path.
    /// At most four inactive images share one explicit memory budget; every
    /// optional allocation needs all-rank admission above the8GiB reserve.
    fn reuse_cached_session(&mut self,plan:&mut RequestPlan,allow_exact_logits:bool,allow_encoder:bool)->Result<(Value,Option<CachedSession>)> {
        let started=Instant::now();
        if self.session_cache_budget==0 || plan.reuse || self.session.is_none() || plan.previous==0 {
            return Ok((json!({"enabled":self.session_cache_budget>0,"hit":false,"saved":false,
                "active_reuse":plan.reuse,"seconds":started.elapsed().as_secs_f64()}),None));
        }
        ensure!(self.verify.profile_is_current() && self.verify.position()==plan.previous as u64,
            "cache lookup requires the current accepted owner");
        let hit=self.cached_sessions.iter().enumerate()
            .filter(|(_,c)|(allow_encoder||matches!(&c.image,CachedState::Complete{..})) && plan.prompt.starts_with(&c.prefix)
                && plan.media.matches_prefix(&c.media,c.prefix.len())
                && (c.prefix.len()<plan.prompt.len() || (self.prompt_cache_logits && allow_exact_logits
                    && c.prefix.len()==plan.prompt.len() && c.prompt_logits.is_some())))
            .max_by_key(|(_,c)|c.prefix.len()).map(|(i,_)|i);
        let wanted=hit.map(|i|self.cached_sessions.remove(i));
        let current_prefix=self.history.committed().to_vec();
        let current_pending=self.history.pending().ok_or_else(||anyhow::anyhow!("cached session lacks pending anchor"))?;
        // A displaced unfinished request already owns an immutable image.
        // Do not copy that same loaded prefix into the optional cache too.
        // A wanted cache image remains charged while held outside the vector.
        let saved=if self.loaded_request_is_suspended {
            json!({"saved":false,"saved_seconds":0.,"evicted_sessions":0,
                "capacity_refused":false,"already_suspended":true})
        }else{
            self.retain_cached_prefix_reserved(&current_prefix,current_pending,self.history.media().through(current_prefix.len()),
                wanted.as_ref().map_or(0,|c|c.charge),usize::from(wanted.is_some()))?
        };
        let transient_image_bytes=self.suspended_image_bytes
            +self.cached_sessions.iter().map(|c|c.charge).sum::<usize>()
            +wanted.as_ref().map_or(0,|c|c.charge);
        let transient_image_count=self.suspended_image_count+self.cached_sessions.len()+usize::from(wanted.is_some());
        ensure!(transient_image_bytes<=self.shared_image_budget && transient_image_count<=4,
            "restoring cache image exceeded the shared inactive budget");
        let reserve=8*1024usize.pow(3)+16*1024usize.pow(2);
        let mut restored_tokens=0;let mut restore_seconds=0.;let mut restore_refused=false;
        let mut retained_image=false;let mut retention_evictions=0;
        let mut reconstructed_upgrade=false;let mut upgrade_refused=false;let mut upgrade_seconds=0.;
        let mut upgrade_capacity_refused=false;let mut upgrade_memory_refused=false;
        let mut upgrade_late_memory_refused=false;let mut upgrade_evicted_sessions=0;
        let mut encoder_reconstructed=false;
        let mut exact_prompt_anchor=None;let mut sample_seconds=0.;
        if let Some(cached)=wanted {
            let extra_history=match &cached.image {CachedState::Encoder{history,..}=>
                history.as_ref().map_or(0,crate::history_lookup::HistoryLookup::workspace_bytes),_=>0};
            let mut admitted=self.control.all_admit(self.cuda.admission_memory()?.admitted_available>=reserve+extra_history)?;
            if !admitted {
                self.cached_sessions.clear();
                admitted=self.control.all_admit(self.cuda.admission_memory()?.admitted_available>=reserve+extra_history)?;
            }
            if admitted {
                self.target_lineage=self.target_lineage.checked_add(1)
                    .ok_or_else(||anyhow::anyhow!("cached target lineage exhausted"))?;
                let began=Instant::now();
                if matches!(&cached.image,CachedState::Encoder{..}) && plan.prompt.len()-cached.prefix.len()>=128 {
                    // Transfer the charged parent to the compound operation.
                    // It keeps the parent alive through optional successor
                    // allocation; no incomplete decoder owner escapes.
                    return Ok((json!({"enabled":true,"hit":true,"encoder_deferred":true,
                        "restored_tokens":cached.prefix.len(),"saved":saved["saved"],
                        "saved_seconds":saved["saved_seconds"],"evicted_sessions":saved["evicted_sessions"],
                        "capacity_refused":saved["capacity_refused"],"transient_image_bytes":transient_image_bytes,
                        "transient_image_count":transient_image_count,"seconds":started.elapsed().as_secs_f64()}),Some(cached)));
                }
                let pending=match &cached.image {
                    CachedState::Complete{image,pending}=>{
                        let restored=self.session.as_mut().unwrap().restore_state_image(&mut self.verify,image,
                            self.target_lineage,&self.comm,&self.stream)?;
                        ensure!(restored==*pending,"cached anchor differs from session image");restored
                    },
                    CachedState::Encoder{image,history,policy}=>{
                        encoder_reconstructed=true;
                        ensure!(self.encoder_prompt_checkpoint,"encoder cache disabled");
                        self.large.handoff_from(&self.verify)?;
                        self.large.restore_encoder_checkpoint_full(image,self.target_lineage,&self.comm,&self.stream)?;
                        let index=history.as_ref().map(crate::history_lookup::HistoryLookup::clone_reserved);
                        self.attach_prompt_with_index(false,0,&cached.prefix,&cached.prefix,plan.seed,
                            !allow_exact_logits,Some(index))?;
                        let session=self.session.as_mut().unwrap();
                        session.restore_proposal_policy(policy,&self.stream)?;
                        if allow_exact_logits {session.configure_sampling(plan.seed,Some(self.cfg.eos_token_id))?;}
                        else{session.configure_greedy(Some(self.cfg.eos_token_id))?;}
                        let pending=self.large.greedy_token(&self.stream)?;
                        self.verify.handoff_from(&self.large)?;pending
                    },
                };
                ensure!(self.verify.position()==cached.prefix.len()as u64,
                    "cached image and resident prefix disagree");
                restore_seconds=began.elapsed().as_secs_f64();
                if cached.prefix.len()==plan.prompt.len() {
                    ensure!(self.prompt_cache_logits && allow_exact_logits,"exact prompt sampling policy");
                    self.history.restore_cached_media(plan,&cached.prefix,Some(pending),true,&cached.media)?;
                    let sample_at=Instant::now();
                    exact_prompt_anchor=Some(self.verify.sample_prompt_logits(
                        cached.prompt_logits.as_ref().ok_or_else(||anyhow::anyhow!("exact prompt has no logits"))?,
                        cached.prefix.len()as u64,plan.seed,&self.stream)?);
                    sample_seconds=sample_at.elapsed().as_secs_f64();
                    self.session.as_mut().unwrap().configure_sampling(plan.seed,Some(self.cfg.eos_token_id))?;
                }else{
                    self.history.restore_cached_media(plan,&cached.prefix,Some(pending),false,&cached.media)?;
                }
                restored_tokens=cached.prefix.len();
                // A restored image is immutable and remains valid across target
                // lineages. If it is exactly the prefix the next Begin would
                // snapshot again, retain its existing allocation instead. The
                // ordinary four-entry/byte policy still governs admission;
                // a replacement must also charge its still-live original.
                let checkpoint=self.prompt_checkpoint_positions(plan.prompt.len())[0];
                if self.retain_restored_checkpoint && (cached.prefix.len()>=8192
                    || (!cached.media.placements().is_empty() && cached.prompt_logits.is_some()))
                    && checkpoint<=cached.prefix.len() {
                    // A short suffix forced reconstruction of a complete
                    // encoder-only parent. Save that already valid state once
                    // so another short branch does not repeat the decoder pass.
                    let upgrade=self.encoder_upgrade_reconstructed&&matches!(&cached.image,CachedState::Encoder{..});
                    let mut candidate=cached;
                    if upgrade {
                        let at=Instant::now();
                        let (device,host)=self.session.as_ref().unwrap().compact_state_image_bytes(&self.verify)?;
                        let charge=device.checked_add(host).and_then(|n|n.checked_add(candidate.prefix.len()*4+candidate.media.metadata_bytes()))
                            .ok_or_else(||anyhow::anyhow!("reconstructed cache charge overflow"))?;
                        let entries=self.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
                        let optional=self.cached_sessions.iter().map(CachedSession::opportunistic_logits).collect::<Vec<_>>();
                        let (budget,slots)=self.cache_capacity(0,0);
                        // The old encoder image is a reservation until all
                        // ranks have saved/admitted the complete replacement.
                        // Never credit its bytes or slot before that commit.
                        let evictions=encoder_checkpoint_evictions(&entries,&optional,
                            (candidate.prefix.len(),charge),Some(candidate.charge),budget,
                            self.session_cache_long_first,slots);
                        let eligible=device<=2*1024usize.pow(3)&&evictions.is_some();
                        upgrade_capacity_refused=!eligible;
                        let admitted=self.control.all_admit(eligible
                            &&self.cuda.admission_memory()?.admitted_available>=reserve+charge)?;
                        upgrade_memory_refused=eligible&&!admitted;
                        if admitted {
                            let mut indices=evictions.unwrap();indices.sort_unstable();
                            upgrade_evicted_sessions=indices.len();
                            for i in indices.into_iter().rev(){self.cached_sessions.remove(i);}
                            self.stream.synchronize()?;
                            let image=self.try_save_session_image(pending,device)?;
                            // Test builds can decline one rank's vote after the
                            // actual copy/admission, before replacing the parent.
                            #[cfg(feature = "storage-fault-injection")]
                            let image={
                                let fits=self.cache_save_fault.apply(cache_admission_fault::Point::ReconstructedAfterCopy,
                                    self.comm.rank(),image.is_some());
                                if self.control.all_admit(fits)? {image}else{None}
                            };
                            if let Some(image)=image {
                                ensure!(image.device_bytes()==device,"reconstructed cache estimate differs");
                                // save_state_image joins every copy before it
                                // returns. Only a successful all-rank save may
                                // now release the original encoder/history.
                                candidate.image=CachedState::Complete{image,pending};
                                candidate.charge=charge;candidate.prompt_logits=None;
                                reconstructed_upgrade=true;
                            }else{upgrade_refused=true;upgrade_late_memory_refused=true;}
                        }else{upgrade_refused=true;}
                        upgrade_seconds=at.elapsed().as_secs_f64();
                    }
                    if upgrade {
                        // Both refusal and success still own the reservation;
                        // reinsertion cannot silently discard this parent.
                        retained_image=self.return_encoder_parent(Some(candidate))?;
                    }else{
                        let cached=candidate;
                        let entries=self.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
                        let (budget,slots)=self.cache_capacity(0,0);
                        if let Some(mut evictions)=cache_evictions(&entries,(cached.prefix.len(),cached.charge),
                            budget,self.session_cache_long_first,slots) {
                            evictions.sort_unstable();retention_evictions=evictions.len();
                            for i in evictions.into_iter().rev(){self.cached_sessions.remove(i);}
                            self.cached_sessions.push(cached);
                            retained_image=true;
                        }
                    }
                }
            }else{restore_refused=true;}
        }
        Ok((json!({"enabled":true,"hit":restored_tokens>0,"restored_tokens":restored_tokens,
            "exact_prompt_anchor":exact_prompt_anchor,"sample_seconds":sample_seconds,
            "policy":if self.session_cache_long_first {"long_parents_and_recent"}else{"fifo"},
            "evicted_sessions":saved["evicted_sessions"],"capacity_refused":saved["capacity_refused"],
            "saved":saved["saved"],"saved_seconds":saved["saved_seconds"],"restore_seconds":restore_seconds,"restore_refused":restore_refused,
            "already_suspended":saved["already_suspended"],
            "transient_image_bytes":transient_image_bytes,"transient_image_count":transient_image_count,
            "retained_image":retained_image,"retention_evictions":retention_evictions,
            "reconstructed_upgrade":reconstructed_upgrade,"upgrade_refused":upgrade_refused,"upgrade_seconds":upgrade_seconds,
            "upgrade_capacity_refused":upgrade_capacity_refused,"upgrade_memory_refused":upgrade_memory_refused,
            "upgrade_late_memory_refused":upgrade_late_memory_refused,"upgrade_evicted_sessions":upgrade_evicted_sessions,
            "upgrade_parent_retained_on_refusal":upgrade_refused&&retained_image,
            "encoder_reconstructed":encoder_reconstructed,
            "inactive_sessions":self.cached_sessions.len(),"charged_bytes":self.cached_sessions.iter().map(|c|c.charge).sum::<usize>(),
            "budget_bytes":self.session_cache_budget,"seconds":started.elapsed().as_secs_f64()}),None))
    }

    #[allow(clippy::too_many_arguments)]
    fn attach_prompt(&mut self,reuse:bool,previous:usize,ids:&[u32],prefix:&[u32],
        seed:u64,greedy:bool)->Result<()> {
        self.attach_prompt_with_index(reuse,previous,ids,prefix,seed,greedy,None)
    }
    #[allow(clippy::too_many_arguments)]
    fn attach_prompt_with_index(&mut self,reuse:bool,previous:usize,ids:&[u32],prefix:&[u32],
        seed:u64,greedy:bool,index:Option<Option<crate::history_lookup::HistoryLookup>>)->Result<()> {
        ensure!(!reuse||index.is_none(),"active history is already incremental");
        if let Some(session) = &mut self.session {
            if reuse {
                session.resume_after_extend(
                    &self.large,
                    previous as u64,
                    ids,
                    &self.stream,
                )?;
            } else {
                session.reset_from_prefill(
                    &self.large,
                    &self.cuda,
                    &self.core,
                    &self.stream,
                )?;
            }
        } else {
            let prepared = self
                .prepared_session
                .take()
                .ok_or_else(|| anyhow::anyhow!("resident prepared session absent"))?;
            self.session = Some(prepared.attach(&self.large, &self.stream)?);
        }
        if greedy {
            self.session.as_mut().unwrap().configure_greedy(Some(self.cfg.eos_token_id))?;
        } else {
            self.session.as_mut().unwrap().configure_sampling(seed, Some(self.cfg.eos_token_id))?;
        }
        if let Some(index)=index {
            ensure!(index.is_some()==self.history_lookup_enabled,"cached history policy mismatch");
            self.session.as_mut().unwrap().install_history_lookup(&self.large,index)?;
        }else if self.history_lookup_enabled && !reuse {
            self.session.as_mut().unwrap().configure_history_lookup(&self.large,prefix)?;
        }
        Ok(())
    }

    fn prompt_checkpoint_positions(&self, prompt_tokens: usize) -> Vec<usize> {
        // Both policies leave at least one real token to reconstruct scratch
        // logits and sample afresh. Keep the aligned shared prefix as well:
        // replacing it with only a near-end image would lose task-suffix reuse.
        let aligned=(prompt_tokens.saturating_sub(128)/128)*128;
        let mut positions=vec![aligned];
        if self.prompt_checkpoint_tail==1 && prompt_tokens.saturating_sub(1)>aligned {
            positions.push(prompt_tokens.saturating_sub(1));
        }
        positions
    }

    fn publish_begin(&mut self,plan:RequestPlan,anchor:u32,greedy:bool,started:Instant,
        details:Value)->Result<Value> {
        let agreement_at=Instant::now();
        let lookup=self.session.as_ref().unwrap().history_lookup_metadata();
        ensure!(self.verify.profile_is_current() && self.verify.position()==plan.prompt.len()as u64
            && lookup.is_some()==self.history_lookup_enabled
            && lookup.is_none_or(|(position,_)|position==self.verify.position()),
            "resident prompt state/history admission");
        let output=self.history.prefilled(self.verify.position()as usize,anchor)?;
        self.current=Some(plan.id);self.emitted=1;self.limit=plan.limit;
        self.install_media(None)?;
        if !self.progress_next {self.agree(&output)?;}
        let mut event=json!({"type":"tokens","request_id":plan.id,"output_ids":output,
            "prompt_tokens":plan.prompt.len(),"cached_tokens":if plan.reuse {plan.previous}else{0},
            "uncached_tokens":plan.delta().len(),"frontier":self.history.frontier(),
            "pending":self.history.pending(),"seconds":started.elapsed().as_secs_f64(),
            "vision":self.media_receipt,"transport_memory":self.comm.transport_memory(),
            "target_temperature":if greedy {0.}else{1.},"target_top_p":1.,
            "causal_verify_threshold":self.session.as_ref().unwrap().causal_verify_threshold(),
            "draft_head":self.session.as_ref().unwrap().draft_head_kind(),
            "engram_staging":self.verify.engram_input_kind(),"paired_engram_io":self.verify.paired_engram_io_enabled(),
            "draft_temperature":self.session.as_ref().unwrap().draft_temperature(),
            "block_verification":self.session.as_ref().unwrap().block_verification(),
            "conditioning_split":self.session.as_ref().unwrap().conditioning_split(),
            "markov_direct_gather":self.session.as_ref().unwrap().direct_markov_gather(),
            "cached_probability_factors":self.session.as_ref().unwrap().cached_probability_factors(),
            "prefill_norm_fusion":crate::mhc::prefill_norm_fusion_enabled()?,"session_prepared_before_ready":true,
            "generation_fp32_attention":crate::layer_attention::generation_fp32_attention_enabled()?,
            "selected_sort_shuffle":crate::query_index::selected_sort_shuffle_enabled()?,
            "history_lookup_enabled":self.history_lookup_enabled,"history_lookup_position":lookup.map(|(p,_)|p),
            "history_lookup_workspace_bytes":lookup.map_or(0,|(_,bytes)|bytes)});
        for (key,value) in details.as_object().ok_or_else(||anyhow::anyhow!("begin details must be an object"))? {
            ensure!(!event.as_object().unwrap().contains_key(key),"duplicate begin field");
            event[key]=value.clone();
        }
        event["begin_phases_seconds"]["frontier_agreement"]=json!(agreement_at.elapsed().as_secs_f64());
        Ok(event)
    }

    fn begin_small_append(&mut self,plan:RequestPlan,greedy:bool,started:Instant,
        retained_session_cache:Value)->Result<Value> {
        let medium=plan.delta().len()>6;
        ensure!(plan.reuse && (1..=16).contains(&plan.delta().len()) && self.session.is_some()
            && (!medium || (std::env::var("CANADA_SCRATCH_RESIDENT_MEDIUM_APPEND").as_deref()==Ok("1")
                && self.long_verify.is_some() && self.long_history_buffers.is_some()))
            && self.verify.profile_is_current() && self.verify.position()==plan.previous as u64
            && std::env::var("CANADA_SCRATCH_APPEND_BF16_ONLY").as_deref()==Ok("1")
            && std::env::var("CANADA_SCRATCH_TINY_APPEND_RETAIN_GRAPHS").as_deref()==Ok("1"),
            "resident small append requires a complete short suffix and BF16 graph bank");
        let prepared_at=Instant::now();
        // One whole request suffix: the decoder floor must not reset between
        // chunks of a larger prompt. Encoder history stays absolute.
        let target=if medium {
            let longer=self.long_verify.as_mut().unwrap();
            longer.handoff_from(&self.verify)?;
            longer
        } else {&mut self.verify};
        target.set_tiny_extend_for_probe(true,&self.stream)?;
        target.begin_verification(plan.delta(),&self.comm,&self.stream)?;
        target.finish_verification(plan.delta().len(),&self.comm,&self.stream)?;
        let prefilled_at=Instant::now();
        let anchor=if greedy {target.greedy_token(&self.stream)?}
            else {target.sample_token(plan.seed,&self.stream)?};
        let sampled_at=Instant::now();
        let session=self.session.as_mut().unwrap();
        if medium {
            session.resume_after_medium_extend(target,self.long_history_buffers.as_mut().unwrap(),
                plan.previous as u64,plan.delta(),&self.comm,&self.stream)?;
        } else {
            session.resume_after_small_extend(target,plan.previous as u64,plan.delta(),
                &self.comm,&self.stream)?;
        }
        if greedy {session.configure_greedy(Some(self.cfg.eos_token_id))?;}
        else {session.configure_sampling(plan.seed,Some(self.cfg.eos_token_id))?;}
        // Ordinary decode and all saved images must see the normal verifier
        // mode, while both attention workspaces remain alive for graph replay.
        target.set_tiny_extend_for_probe(false,&self.stream)?;
        if medium {self.verify.handoff_from(self.long_verify.as_ref().unwrap())?;}
        let attached_at=Instant::now();
        self.publish_begin(plan,anchor,greedy,started,json!({
            "retained_session_cache":retained_session_cache,
            "small_append":true,
            "small_append_physical_rows":if medium {16}else{6},
            "prompt_checkpoint":{"enabled":self.prompt_checkpoint,"saved":false},
            "checkpoint_passes":[],
            "prompt_logit_cache":{"enabled":self.prompt_cache_logits,"hit":false,"saved":false,
                "small_append_skipped":true},
            "prefill_phases":Value::Null,"prefix_prefill_phases":Value::Null,
            "begin_phases_seconds":{
                "reset_and_handoff":prepared_at.duration_since(started).as_secs_f64(),
                "target_prefill":prefilled_at.duration_since(prepared_at).as_secs_f64(),
                "first_target_sample":sampled_at.duration_since(prefilled_at).as_secs_f64(),
                "draft_attach_and_history":attached_at.duration_since(sampled_at).as_secs_f64()}}))
    }

    /// Remove the unpublished decoder/drafter checkpoint pass. Only the final
    /// >=128-row suffix becomes a complete serving owner. The saved history
    /// index advances incrementally, including its collision replacement state.
    fn begin_encoder_prompt(&mut self,mut plan:RequestPlan,greedy:bool,started:Instant,
        mut retained:Value,wanted:Option<CachedSession>,
        should_cancel:&mut impl FnMut(&mut ControlChannel)->Result<bool>)->Result<Value> {
        use crate::history_lookup::HistoryLookup;
        let active_reuse=plan.reuse&&wanted.is_none();
        let previous=wanted.as_ref().map_or(if plan.reuse{plan.previous}else{0},|c|c.prefix.len());
        // A true active continuation already owns a complete encoder frontier.
        // Snapshot it without splitting the new append into two numerical plans.
        // A cache-restored Complete image also sets plan.reuse; it must retain
        // the ordinary near-end policy, hence the explicit active_reuse marker.
        let live_frontier=self.encoder_active_frontier&&active_reuse
            &&retained["active_reuse"]==true&&previous>=8192;
        let checkpoint=if live_frontier{previous}
            else{self.prompt_checkpoint_positions(plan.prompt.len())[0].max(previous)};
        ensure!(self.encoder_prompt_checkpoint&&checkpoint>=8192&&checkpoint>=previous
            &&plan.prompt.len()-checkpoint>=128,"encoder serving checkpoint extent");
        if self.session.is_some() {
            ensure!(self.verify.profile_is_current()&&self.verify.position()==plan.previous as u64,
                "encoder serving target/history owner");
            if wanted.is_none()&&!plan.reuse {
                self.target_lineage=self.target_lineage.checked_add(1)
                    .ok_or_else(||anyhow::anyhow!("encoder target lineage exhausted"))?;
                self.verify.reset(self.target_lineage,&self.comm,&self.stream)?;
            }
            self.large.handoff_from(&self.verify)?;
        }else{ensure!(!plan.reuse&&wanted.is_none(),"encoder cache before initial session");}
        let prepared_at=Instant::now();
        let prefix=&plan.prompt[..checkpoint];
        let media=plan.media.through(checkpoint);
        let device_bytes=self.large.encoder_checkpoint_image_bytes(checkpoint)?;
        let host_bytes=if self.history_lookup_enabled {
            crate::dspark_generation::history_lookup_reserved_bytes(self.large.context_capacity())?
        }else{0};
        let charge=device_bytes.checked_add(host_bytes).and_then(|x|x.checked_add(prefix.len()*4+media.metadata_bytes()))
            .and_then(|x|x.checked_add(std::mem::size_of::<crate::dspark_generation::ProposalPolicyImage>()))
            .ok_or_else(||anyhow::anyhow!("encoder cache charge overflow"))?;
        let duplicate=self.cached_sessions.iter().chain(wanted.iter()).any(|c|c.prefix==prefix && c.media.matches_prefix(&media,prefix.len()));
        // Retain the shared parent even when this checkpoint advances into a
        // branch-specific suffix. Both images must fit the existing budget;
        // do not credit the parent's bytes before a successor actually exists.
        let (budget,slots)=self.cache_capacity(0,0);
        let entries=self.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
        let optional=self.cached_sessions.iter().map(CachedSession::opportunistic_logits).collect::<Vec<_>>();
        let evictions=encoder_checkpoint_evictions(&entries,&optional,(checkpoint,charge),
            wanted.as_ref().map(|c|c.charge),budget,self.session_cache_long_first,slots);
        let eligible=!duplicate&&device_bytes<=2*1024usize.pow(3)&&evictions.is_some();
        // One extra index can become the new live index while the checkpoint
        // copy remains immutable. Include that temporary allocation in UMA.
        let reserve=8*1024usize.pow(3)+16*1024usize.pow(2);
        let admission_required=reserve+charge+host_bytes;
        let admission=self.admit_cache_save(eligible,evictions,prefix,&media,admission_required,
            slots.saturating_sub(usize::from(wanted.is_some())))?;
        let admitted=admission.admitted;
        let copy_extend=|history:&Option<HistoryLookup>,start:usize,ids:&[u32]|->Result<Option<HistoryLookup>> {
            let mut copy=history.as_ref().map(HistoryLookup::clone_reserved);
            if let Some(h)=&mut copy {
                ensure!(h.position()==start as u64,"encoder history clone frontier");
                if !ids.is_empty(){h.commit(start as u64,ids)?;}
            }
            Ok(copy)
        };
        let index_at=Instant::now();
        let checkpoint_history=if admitted {
            let index=if let Some(cached)=&wanted {
                let CachedState::Encoder{history,..}=&cached.image else{anyhow::bail!("expected encoder cache");};
                copy_extend(history,previous,&plan.prompt[previous..checkpoint])?
            }else if plan.reuse {
                self.session.as_ref().unwrap().checkpoint_history_lookup(previous as u64,&plan.prompt[previous..checkpoint])?
            }else if self.history_lookup_enabled {
                Some(crate::dspark_generation::new_history_lookup(&self.cuda,self.large.context_capacity(),prefix)?)
            }else{None};
            ensure!(index.as_ref().map_or(0,HistoryLookup::workspace_bytes)<=host_bytes,
                "encoder history index exceeds reserved bytes");Some(index)
        }else{None};
        let index_seconds=index_at.elapsed().as_secs_f64();
        let restore=if let Some(cached)=&wanted {
            ensure!(cached.prompt_logits.is_none(),"encoder parent unexpectedly owns prompt logits");
            let CachedState::Encoder{image,..}=&cached.image else{anyhow::bail!("expected encoder cache");};
            Some(crate::target::EncoderPrefixRestore::Borrowed(image,self.target_lineage))
        }else{None};
        // Both callbacks run synchronously on the sole worker owner. Share
        // its ordered CPU control channel without introducing another writer.
        let control=std::cell::RefCell::new(&mut self.control);
        #[cfg(feature = "storage-fault-injection")]
        let fault=&mut self.cache_save_fault;
        #[cfg(feature = "storage-fault-injection")]
        let mut admission_number=0usize;
        let (image,details,done)=self.large.prefill_with_encoder_checkpoint_interruptible(&mut self.wides,restore,
            &plan.prompt[previous..checkpoint],&plan.prompt[checkpoint..],admitted.then_some(device_bytes),
            &self.comm,&self.stream,&mut |_|should_cancel(&mut control.borrow_mut()),
            &mut |fits| {
                #[cfg(feature = "storage-fault-injection")]
                let fits={
                    admission_number+=1;
                    let point=if admission_number==1 {cache_admission_fault::Point::EncoderBeforeAllocate}
                        else {cache_admission_fault::Point::EncoderAfterAllocate};
                    fault.apply(point,self.comm.rank(),fits)
                };
                control.borrow_mut().all_admit(fits)
            })?;
        if previous+done<plan.prompt.len() {
            // A speculative cache image/index may be ahead of the stopped
            // frontier. Release it and rebuild only the committed prefix.
            drop(image);drop(checkpoint_history);
            retained["retained_image"]=json!(self.return_encoder_parent(wanted)?);
            return self.cancel_prefill(plan,greedy,active_reuse,previous,previous+done,started,retained);
        }
        let prefilled_at=Instant::now();
        let anchor=if greedy{self.large.greedy_token(&self.stream)?}else{self.large.sample_token(plan.seed,&self.stream)?};
        let sampled_at=Instant::now();
        let live_index=if active_reuse{None}
            else if let Some(index)=&checkpoint_history {
                Some(copy_extend(index,checkpoint,&plan.prompt[checkpoint..])?)
            }else if let Some(cached)=&wanted {
                let CachedState::Encoder{history,..}=&cached.image else{unreachable!()};
                Some(copy_extend(history,previous,&plan.prompt[previous..])?)
            }else{None};
        self.attach_prompt_with_index(active_reuse,previous,&plan.prompt[previous..],
            &plan.prompt,plan.seed,greedy,live_index)?;
        let restored_policy=if let Some(cached)=&wanted {
            let CachedState::Encoder{policy,..}=&cached.image else{unreachable!()};
            Some(policy)
        }else{None};
        if let Some(policy)=restored_policy {
            let session=self.session.as_mut().unwrap();session.restore_proposal_policy(policy,&self.stream)?;
            if greedy {session.configure_greedy(Some(self.cfg.eos_token_id))?;}
            else{session.configure_sampling(plan.seed,Some(self.cfg.eos_token_id))?;}
        }
        let checkpoint_saved=image.is_some();
        if let Some(image)=image {
            ensure!(image.bytes()==device_bytes&&image.position()==checkpoint,"encoder cache saved extent");
            self.cached_sessions.push(CachedSession{prefix:prefix.to_vec(),media,charge,prompt_logits:None,
                image:CachedState::Encoder{image,history:checkpoint_history.unwrap(),
                    policy:self.session.as_ref().unwrap().proposal_policy_image()?}});
        }
        let restored_prefix=wanted.as_ref().map(|c|(&c.prefix,&c.media));
        if let Some((prefix,media))=restored_prefix {
            self.history.restore_cached_media(&mut plan,prefix,None,false,media)?;
            retained["restore_seconds"]=details["restore_seconds"].clone();
            retained["encoder_restored"]=json!(true);
        }
        let retained_image=self.return_encoder_parent(wanted)?;
        retained["retained_image"]=json!(retained_image);
        ensure!(self.cached_sessions.iter().map(|c|c.charge).sum::<usize>()+self.suspended_image_bytes<=self.shared_image_budget
            &&self.cached_sessions.len()+self.suspended_image_count<=4,"encoder cache exceeded shared budget");
        let (prompt_image,mut prompt_logit_cache)=self.prepare_prompt_logit_cache(&plan,greedy,retained["hit"]==true)?;
        self.verify.handoff_from(&self.large)?;
        if let Some((logits,device,charge))=prompt_image {
            let save_at=Instant::now();
            if let Some(image)=self.try_save_session_image(anchor,device)? {
                ensure!(image.device_bytes()==device,"encoder final prompt image estimate differs");
                self.cached_sessions.push(CachedSession{prefix:plan.prompt.clone(),media:plan.media.clone(),charge,prompt_logits:Some(logits),
                    image:CachedState::Complete{image,pending:anchor}});
                prompt_logit_cache["saved"]=json!(true);
            }else{prompt_logit_cache["late_memory_refused"]=json!(true);}
            prompt_logit_cache["seconds"]=json!(prompt_logit_cache["seconds"].as_f64().unwrap()+save_at.elapsed().as_secs_f64());
        }
        let attached_at=Instant::now();
        self.publish_begin(plan,anchor,greedy,started,json!({"retained_session_cache":retained,
            "prompt_checkpoint":{"enabled":true,"encoder_only":true,"saved":checkpoint_saved,"already_cached":duplicate,
                "active_frontier":live_frontier,
                "evicted_sessions":admission.evicted,"capacity_refused":!eligible&&!duplicate,"prefix_tokens":checkpoint,
                "memory_refused":eligible&&!admitted,"local_memory_refused":eligible&&!admission.local_fits,
                "pressure_evicted_sessions":admission.pressure_evicted,"forecast_refused":admission.forecast_refused,
                "admission_available_bytes":admission.available,"admission_required_bytes":admission_required,
                "late_memory_refused":admitted&&!checkpoint_saved,
                "image_charge_bytes":if checkpoint_saved{charge}else{0},"history_index_seconds":index_seconds,
                "charged_bytes":self.cached_sessions.iter().map(|c|c.charge).sum::<usize>(),
                "inactive_sessions":self.cached_sessions.len()},
            "encoder_checkpoint_phases":details,"checkpoint_passes":[],"prompt_logit_cache":prompt_logit_cache,
            "prefill_phases":details["suffix_phases"],"prefix_prefill_phases":Value::Null,
            "begin_phases_seconds":{"reset_and_handoff":prepared_at.duration_since(started).as_secs_f64(),
                "target_prefill":details["target_total_seconds"],"first_target_sample":sampled_at.duration_since(prefilled_at).as_secs_f64(),
                "draft_attach_and_history":attached_at.duration_since(sampled_at).as_secs_f64()+index_seconds}}))
    }

    fn execute(&mut self, command: Command) -> Result<Value> {
        self.execute_with_prefill_cancel(command,&mut |_|Ok(false))
    }

    /// Reinsert the reservation after success, optional refusal or cancellation.
    /// Admission reserved its slot/bytes, so returning it cannot evict a sibling.
    fn return_encoder_parent(&mut self,parent:Option<CachedSession>)->Result<bool> {
        if let Some(parent)=parent.filter(|_|self.retain_restored_checkpoint) {
            let (budget,slots)=self.cache_capacity(0,0);
            ensure!(self.cached_sessions.len()<slots
                &&self.cached_sessions.iter().map(|c|c.charge).sum::<usize>()+parent.charge<=budget,
                "encoder parent reservation lost");
            self.cached_sessions.push(parent);
            Ok(true)
        }else{Ok(false)}
    }

    /// A cancelled prefill retains its complete partial target/drafter state,
    /// but publishes no temporary anchor. The next wire command must finish it.
    #[allow(clippy::too_many_arguments)]
    fn cancel_prefill(&mut self,plan:RequestPlan,greedy:bool,reuse:bool,previous:usize,
        frontier:usize,started:Instant,retained:Value)->Result<Value> {
        ensure!(frontier>previous&&frontier<plan.prompt.len(),"cancelled prefill frontier");
        let anchor=self.large.greedy_token(&self.stream)?;
        self.attach_prompt(reuse,previous,&plan.prompt[previous..frontier],
            &plan.prompt[..frontier],plan.seed,greedy)?;
        self.verify.handoff_from(&self.large)?;
        self.history.partial_prefilled(frontier,anchor)?;
        self.current=Some(plan.id);self.emitted=0;self.limit=plan.limit;self.prefill_cancelled=true;
        self.install_media(None)?;
        self.agree(&[])?;
        Ok(json!({"type":"prefill_progress","request_id":plan.id,"cancelled":true,
            "frontier":frontier,"prompt_tokens":plan.prompt.len(),"emitted_tokens":0,
            "cached_tokens":if plan.reuse{plan.previous}else{retained["restored_tokens"].as_u64().unwrap_or(0)as usize},
            "seconds":started.elapsed().as_secs_f64(),
            "retained_session_cache":retained}))
    }

    fn execute_with_prefill_cancel(&mut self,command:Command,
        should_cancel:&mut impl FnMut(&mut ControlChannel)->Result<bool>)->Result<Value> {
        let started = Instant::now();
        match command {
            Command::Begin {
                images,
                id,
                seed,
                greedy,
                limit,
                tokens,
            } => {
                ensure!(
                    self.current.is_none(),
                    "cannot replace an active resident request"
                );
                let media=self.prompt_identity(&images,&tokens)?;
                let mut plan = self.history.prepare_with_media(id, tokens, media, limit, seed)?;
                let (retained_session_cache,deferred_encoder)=self.reuse_cached_session(&mut plan,!greedy,true)?;
                let media_frontier=deferred_encoder.as_ref().map_or(if plan.reuse{plan.previous}else{0},|c|c.prefix.len());
                if let Err(error)=self.prepare_media(&images,&plan,media_frontier) {
                    return self.refuse_media_input(id,error);
                }
                if let Some(anchor)=retained_session_cache["exact_prompt_anchor"].as_u64() {
                    ensure!(plan.reuse && plan.previous==plan.prompt.len() && !greedy,
                        "exact prompt restoration did not cover the request");
                    let sample_seconds=retained_session_cache["sample_seconds"].as_f64().unwrap();
                    return self.publish_begin(plan,anchor as u32,greedy,started,json!({
                        "retained_session_cache":retained_session_cache,
                        "prompt_checkpoint":{"enabled":self.prompt_checkpoint,"saved":false,"exact_prompt_hit":true},
                        "prompt_logit_cache":{"enabled":true,"hit":true,"saved":false},"checkpoint_passes":[],
                        "prefill_phases":Value::Null,"prefix_prefill_phases":Value::Null,
                        "begin_phases_seconds":{"reset_and_handoff":started.elapsed().as_secs_f64()-sample_seconds,
                            "target_prefill":0.,"first_target_sample":sample_seconds,"draft_attach_and_history":0.}}));
                }
                let encoder_previous=deferred_encoder.as_ref().map_or(if plan.reuse {plan.previous}else{0},|c|c.prefix.len());
                let encoder_checkpoint=self.prompt_checkpoint_positions(plan.prompt.len())[0].max(encoder_previous);
                if self.encoder_prompt_checkpoint&&encoder_checkpoint>=8192
                    &&plan.prompt.len().saturating_sub(encoder_checkpoint)>=128 {
                    return self.begin_encoder_prompt(plan,greedy,started,retained_session_cache,deferred_encoder,should_cancel);
                }
                ensure!(deferred_encoder.is_none(),"deferred encoder cache requires the compound path");
                if std::env::var("CANADA_SCRATCH_RESIDENT_SMALL_APPEND").as_deref()==Ok("1")
                    && plan.reuse && (1..=if std::env::var("CANADA_SCRATCH_RESIDENT_MEDIUM_APPEND").as_deref()==Ok("1")
                        && std::env::var("CANADA_SCRATCH_RESIDENT_MEDIUM_APPEND_CONTROL").as_deref()!=Ok("1") {16}else{6})
                        .contains(&plan.delta().len()) && self.session.is_some() {
                    return self.begin_small_append(plan,greedy,started,retained_session_cache);
                }
                if self.session.is_some() {
                    ensure!(
                        self.verify.profile_is_current()
                            && self.verify.position() == plan.previous as u64,
                        "resident GPU owner disagrees with accepted history"
                    );
                    if !plan.reuse {
                        self.target_lineage = self.target_lineage.checked_add(1)
                            .ok_or_else(|| anyhow::anyhow!("resident target lineage exhausted"))?;
                        self.verify.reset(self.target_lineage, &self.comm, &self.stream)?;
                    }
                    self.large.handoff_from(&self.verify)?;
                } else {
                    ensure!(!plan.reuse, "cannot reuse state before initial prefill");
                }
                let prepared_at = Instant::now();
                let mut append_start=if plan.reuse {plan.previous}else{0};
                let mut append_reuse=plan.reuse;
                let mut prefix_prefill_seconds=0.;
                let mut prefix_prefill_phases=Value::Null;
                let mut prompt_checkpoint=json!({"enabled":self.prompt_checkpoint,"saved":false});
                let mut checkpoint_passes=Vec::new();
                // The opt-in tail policy adds a near-end image after the shared
                // aligned prefix. Both use the ordinary four-entry/byte budget.
                // Restoring a
                // state image deliberately invalidates scratch logits; replay
                // reconstructs publishable logits and all drafter features.
                let checkpoints=self.prompt_checkpoint_positions(plan.prompt.len());
                for (index,&checkpoint) in checkpoints.iter().enumerate() {
                let checkpoint=if retained_session_cache["hit"]==true && index+1==checkpoints.len()
                    {checkpoint.max(append_start)}else{checkpoint};
                if self.prompt_checkpoint && checkpoint>=8192 && checkpoint>=append_start {
                    let began=Instant::now();
                    let mut pass_prefill_seconds=0.;
                    let anchor=if checkpoint>append_start {
                        let pre=Instant::now();
                        let control=&mut self.control;
                        let done=self.large.prefill_wide_interruptible(&mut self.wides,&plan.prompt[append_start..checkpoint],
                            &self.comm,&self.stream,&mut |_|should_cancel(control))?;
                        if append_start+done<checkpoint {
                            return self.cancel_prefill(plan,greedy,append_reuse,append_start,append_start+done,started,retained_session_cache);
                        }
                        pass_prefill_seconds=pre.elapsed().as_secs_f64();
                        prefix_prefill_seconds+=pass_prefill_seconds;
                        prefix_prefill_phases=serde_json::to_value(self.large.append_metrics()?)?;
                        // This unpublished anchor is metadata only: the real
                        // suffix will replace it before any client emission.
                        let anchor=self.large.greedy_token(&self.stream)?;
                        self.attach_prompt(append_reuse,append_start,&plan.prompt[append_start..checkpoint],
                            &plan.prompt[..checkpoint],plan.seed,greedy)?;
                        self.verify.handoff_from(&self.large)?;
                        anchor
                    } else {
                        // A restored checkpoint may not fit the optional
                        // cache again. Its live target still needs the verifier
                        // lease before a replacement image is considered.
                        self.verify.handoff_from(&self.large)?;
                        self.history.pending().ok_or_else(||anyhow::anyhow!("restored prompt has no anchor"))?
                    };
                    prompt_checkpoint=self.retain_cached_prefix(&plan.prompt[..checkpoint],anchor,&plan.media)?;
                    prompt_checkpoint["enabled"]=json!(true);
                    prompt_checkpoint["tail_policy"]=json!(self.prompt_checkpoint_tail);
                    prompt_checkpoint["prefix_tokens"]=json!(checkpoint);
                    prompt_checkpoint["replayed_suffix_tokens"]=json!(plan.prompt.len()-checkpoint);
                    prompt_checkpoint["charged_bytes"]=json!(self.cached_sessions.iter().map(|c|c.charge).sum::<usize>());
                    prompt_checkpoint["inactive_sessions"]=json!(self.cached_sessions.len());
                    prompt_checkpoint["overhead_seconds"]=json!(began.elapsed().as_secs_f64()-pass_prefill_seconds);
                    checkpoint_passes.push(json!({"checkpoint":prompt_checkpoint,
                        "prefill_seconds":pass_prefill_seconds,
                        "prefill_phases":if pass_prefill_seconds>0. {prefix_prefill_phases.clone()}else{Value::Null}}));
                    self.large.handoff_from(&self.verify)?;
                    append_start=checkpoint;append_reuse=true;
                }
                }
                let suffix_prefill_at=Instant::now();
                let control=&mut self.control;
                let done=self.large.prefill_wide_interruptible(&mut self.wides,&plan.prompt[append_start..],
                    &self.comm,&self.stream,&mut |_|should_cancel(control))?;
                if append_start+done<plan.prompt.len() {
                    return self.cancel_prefill(plan,greedy,append_reuse,append_start,append_start+done,started,retained_session_cache);
                }
                let prefilled_at = Instant::now();
                let prefill_phases = serde_json::to_value(self.large.append_metrics()?)?;
                let anchor = if greedy {self.large.greedy_token(&self.stream)?}
                    else {self.large.sample_token(plan.seed, &self.stream)?};
                let sampled_at = Instant::now();
                self.attach_prompt(append_reuse,append_start,&plan.prompt[append_start..],
                    &plan.prompt,plan.seed,greedy)?;
                let (prompt_image,mut prompt_logit_cache)=self.prepare_prompt_logit_cache(
                    &plan,greedy,retained_session_cache["hit"]==true)?;
                self.verify.handoff_from(&self.large)?;
                if let Some((logits,device,charge))=prompt_image {
                    let save_at=Instant::now();
                    if let Some(image)=self.try_save_session_image(anchor,device)? {
                        ensure!(image.device_bytes()==device,"exact prompt image estimate disagrees");
                        self.cached_sessions.push(CachedSession{prefix:plan.prompt.clone(),media:plan.media.clone(),
                            image:CachedState::Complete{image,pending:anchor},charge,prompt_logits:Some(logits)});
                        prompt_logit_cache["saved"]=json!(true);
                        prompt_logit_cache["image_charge_bytes"]=json!(charge);
                    }else{prompt_logit_cache["late_memory_refused"]=json!(true);}
                    prompt_logit_cache["seconds"]=json!(prompt_logit_cache["seconds"].as_f64().unwrap()
                        +save_at.elapsed().as_secs_f64());
                    prompt_logit_cache["charged_bytes"]=json!(self.cached_sessions.iter().map(|c|c.charge).sum::<usize>());
                    prompt_logit_cache["inactive_sessions"]=json!(self.cached_sessions.len());
                }
                let attached_at = Instant::now();
                self.publish_begin(plan,anchor,greedy,started,json!({
                    "retained_session_cache":retained_session_cache,"prompt_checkpoint":prompt_checkpoint,
                    "checkpoint_passes":checkpoint_passes,"prompt_logit_cache":prompt_logit_cache,
                    "begin_phases_seconds":{
                        "reset_and_handoff":prepared_at.duration_since(started).as_secs_f64(),
                        "target_prefill":prefix_prefill_seconds+prefilled_at.duration_since(suffix_prefill_at).as_secs_f64(),
                        "first_target_sample":sampled_at.duration_since(prefilled_at).as_secs_f64(),
                        "draft_attach_and_history":attached_at.duration_since(sampled_at).as_secs_f64()},
                    "prefill_phases":prefill_phases,"prefix_prefill_phases":prefix_prefill_phases}))
            }
            Command::Step => {
                ensure!(self.current.is_some(), "decode without resident request");
                let limit = if self.long_verify.is_some() {self.history.step_limit_for_rows(16)?}
                    else {self.history.step_limit()?};
                ensure!(limit > 0, "decode after terminal output");
                let long=if let (Some(target),Some(buffers))=(&mut self.long_verify,&mut self.long_history_buffers) {
                    self.session.as_mut().unwrap().try_long_history_step(&mut self.verify,target,buffers,
                        self.history.pending().unwrap(),limit,&self.comm,&self.stream)?
                }else{None};
                let control = &mut self.control;
                let step = if let Some(step)=long {step}else{self.session.as_mut().unwrap().step_with_profile_barrier(
                    &mut self.verify,
                    self.history.pending().unwrap(),
                    limit.min(6),
                    &self.comm,
                    &self.stream,
                    &mut || control.agree(b"warm-verify-profile-rendezvous"),
                )?};
                let output = self.history.committed_step(
                    &step.tokens,
                    step.next_anchor,
                    self.verify.position() as usize,
                )?;
                self.emitted += output.len();
                if !self.progress_next { self.agree(&output)?; }
                let mut event = json!({"type":"tokens","request_id":self.current,"output_ids":output,"frontier":self.history.frontier(),
                    "pending":self.history.pending(),"seconds":started.elapsed().as_secs_f64(),"phase_ms":step.phase_ms,
                    "verified_rows":step.verified_rows,"step_limit":limit,
                    "committed_model_rows":step.tokens.len(),"accepted_draft_tokens":step.matching_proposals,
                    "emitted_tokens":output.len(),"active_sparse_index_layers":self.verify.active_sparse_index_layers(),
                    "compact_sparse_index_layers":self.verify.compact_sparse_index_layers(),
                    "proposal_source":step.proposal_source,"lookup_match_tokens":step.lookup_match_tokens,
                    "history_lookup_position":self.session.as_ref().unwrap().history_lookup_metadata().map(|(position,_)|position)});
                if let Some(expected) = &step.acceptance_expected_lengths {
                    event["acceptance_expected_lengths"] = json!(expected);
                    event["acceptance_shadow_preserved_draw"] = json!(true);
                }
                if let Some(probe) = &step.acceptance_temperature_probe {
                    event["acceptance_temperature_probe"] = probe.clone();
                }
                if step.profiled_verify {
                    event["profiled_verify"] = json!(true);
                    event["profile_start_position"] = json!(step.position - step.tokens.len() as u64);
                    event["profile_append_metrics"] = json!(step.profile_append_metrics);
                    event["verify_host_ms"] = json!(step.verify_host_ms);
                }
                Ok(event)
            }
            Command::Finish(reason) => {
                ensure!(self.current.is_some(), "finish without resident request");
                // Surface deferred device errors before acknowledging a final
                // request boundary. Ordered asynchronous draft/state copies
                // must complete before HTTP can publish successful completion.
                self.stream.synchronize()?;
                self.history.finish(reason)?;
                self.pending_prefill=None;
                self.prefill_cancelled=false;
                self.install_media(None)?;
                self.agree(&[])?;
                let id = self.current.take().unwrap();
                Ok(
                    json!({"type":"finish","request_id":id,"reason":match reason {Finish::Stop=>"stop",Finish::Length=>"length",Finish::Cancelled=>"cancelled"},
                    "completion_tokens":self.emitted,"frontier":self.history.frontier(),"pending":self.history.pending()}),
                )
            }
            Command::Shutdown => {
                ensure!(
                    self.current.is_none(),
                    "shutdown requires completed/cancelled request"
                );
                self.agree(&[])?;
                Ok(json!({"type":"shutdown","frontier":self.history.frontier()}))
            }
            Command::Idle => {
                ensure!(
                    self.current.is_none(),
                    "idle heartbeat during an active request"
                );
                self.agree(&[])?;
                Ok(json!({"type":"idle"}))
            }
            Command::ReclaimInputs { headroom } => self.reclaim_input_space(headroom),
        }
    }
}

#[cfg(feature = "serving")]
impl Worker {
    fn execute_http(&mut self,command:Command,queue:Option<&crate::http_server::Queue>,
        job:Option<&crate::http_server::Job>)->Result<Value> {
        let reclaim=match &command {Command::ReclaimInputs{headroom}=>Some(*headroom),_=>None};
        let event=self.execute_with_prefill_cancel(command,&mut |control| {
            // Only rank zero observes sockets. All ranks stop at the same
            // already committed encoder chunk; ordinary chunk widths and
            // numerical execution are unchanged when cancellation is absent.
            let head=queue.map(|q|[u8::from(q.stopping()||job.is_some_and(|j|j.cancelled()))]);
            let wire=control.command(head.as_ref().map(|v|v.as_slice()))?;
            ensure!(wire.len()==1&&wire[0]<=1,"foreground prefill cancellation command");
            Ok(wire[0]==1)
        })?;
        if let (Some(queue),Some(headroom))=(queue,reclaim) {queue.input_reclaimed(headroom);}
        Ok(event)
    }

    fn warm_before_ready(&mut self, rank: usize) -> Result<Value> {
        ensure!(self.current.is_none() && self.session.is_none()
            && self.history.frontier()==0 && !self.progress_next
            && self.large.context_capacity()>=256,
            "startup warmup needs an unpublished fresh worker");
        let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/target-agentic-python-errors-v2.json"))?;
        let prompt:Vec<u32>=serde_json::from_value(fixture["batches"][0].clone())?;
        let teacher:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/fixed-verify-python-v1.json"))?;
        let teacher:Vec<u32>=serde_json::from_value(teacher["teacher_forced_tokens"].clone())?;
        ensure!(prompt.len()==511 && teacher.len()==512,"startup public fixture identity");
        let started=Instant::now();
        let empty_carry=self.large.carry_digests()?;
        let empty_encoder=self.large.encoder_tail_digest()?;
        let wire=Command::Begin{images: Vec::new(),id:1,seed:0,greedy:false,limit:7,tokens:prompt[..128].to_vec()}.encode()?;
        let command=self.control.command((rank==0).then_some(wire.as_slice()))?;
        self.execute(Command::decode(&command)?)?;
        for live in [1,3,6] {
            self.verify.begin_verification(&teacher[..live],&self.comm,&self.stream)?;
            self.verify.verification_greedy(&self.stream)?;
            self.verify.finish_verification(0,&self.comm,&self.stream)?;
        }
        let mut append_rows=Vec::new();
        if std::env::var("CANADA_SCRATCH_RESIDENT_SMALL_APPEND").as_deref()==Ok("1") {
            // Capture the bounded alternate bank before Ready. Each probe is
            // aborted, and the final startup reset below checks the original
            // empty carry/encoder state. Ordinary decode returns to its own
            // graph and attention owners before any public request can run.
            self.verify.set_tiny_extend_for_probe(true,&self.stream)?;
            for live in 1..=6 {
                self.verify.begin_verification(&teacher[..live],&self.comm,&self.stream)?;
                self.verify.verification_greedy(&self.stream)?;
                self.verify.finish_verification(0,&self.comm,&self.stream)?;
                append_rows.push(live);
            }
            self.verify.set_tiny_extend_for_probe(false,&self.stream)?;
        }
        if let Some(longer)=&mut self.long_verify {
            longer.handoff_from(&self.verify)?;
            longer.begin_verification(&teacher[..16],&self.comm,&self.stream)?;
            longer.verification_greedy(&self.stream)?;
            longer.finish_verification(0,&self.comm,&self.stream)?;
            if std::env::var("CANADA_SCRATCH_RESIDENT_MEDIUM_APPEND").as_deref()==Ok("1") {
                longer.set_tiny_extend_for_probe(true,&self.stream)?;
                for live in 7..=16 {
                    longer.begin_verification(&teacher[..live],&self.comm,&self.stream)?;
                    longer.verification_greedy(&self.stream)?;
                    longer.finish_verification(0,&self.comm,&self.stream)?;
                    append_rows.push(live);
                }
                longer.set_tiny_extend_for_probe(false,&self.stream)?;
            }
            self.verify.handoff_from(longer)?;
        }
        let mut steps=0;
        loop {
            let command=if rank==0 {
                Some(if steps==2 {Command::Finish(Finish::Cancelled)}
                    else {self.active_command(false)?}.encode()?)
            } else {None};
            let wire=self.control.command(command.as_deref())?;
            let event=self.execute(Command::decode(&wire)?)?;
            if event["type"]=="finish" {break;}
            ensure!(event["type"]=="tokens","unexpected startup warmup event");
            steps+=1;
        }
        // HTTP request IDs remain independent of internal GPU generations.
        // Empty both authorities before Ready; the first real Begin performs
        // its normal fresh prefill and injects fresh drafter conditioning.
        self.target_lineage=self.target_lineage.checked_add(1)
            .ok_or_else(||anyhow::anyhow!("startup target lineage exhausted"))?;
        self.verify.reset(self.target_lineage,&self.comm,&self.stream)?;
        let draft_cleared=self.session.as_mut().unwrap().clear_startup_state(&self.stream)?;
        self.cached_sessions.clear();
        let carry_cleared=self.verify.carry_digests()?==empty_carry;
        let encoder_cleared=self.verify.encoder_tail_digest()?==empty_encoder;
        ensure!(carry_cleared && encoder_cleared,
            "startup reset differs from initial empty state: carry={carry_cleared}, encoder={encoder_cleared}");
        self.history=ResidentHistory::new(self.large.context_capacity(),self.cfg.eos_token_id)?;
        self.emitted=0;self.limit=0;
        ensure!(self.current.is_none() && self.history.frontier()==0
            && self.history.pending().is_none() && self.verify.profile_is_current()
            && self.verify.position()==0 && draft_cleared,
            "startup model/session state was not cleared");
        self.agree(&[])?;
        let memory=self.cuda.admission_memory()?;
        ensure!(memory.admitted_available>=8*1024usize.pow(3),"startup warmup consumed runtime reserve");
        Ok(json!({"seconds":started.elapsed().as_secs_f64(),"public_prompt_tokens":128,
            "prewarmed_append_rows":append_rows,
            "decode_steps":steps,"prewarmed_verify_rows":if self.long_verify.is_some() {vec![1,3,6,16]}else{vec![1,3,6]},"target_frontier":0,
            "resident_frontier":0,"draft_frontier":0,"history_lookup_cleared":true,
            "carry_and_encoder_match_initial_empty":true,
            "no_output_published":true,"memory":memory}))
    }
    /// One instrumented warm target step after the HTTP listener shuts down.
    /// Public fixtures reuse the same sampler, model state owner and schedule.
    fn verify_profile_probe(&mut self, rank: usize, output:&Path) -> Result<Value> {
        ensure!(self.current.is_none() && self.large.context_capacity() == 1048576
            && self.progress_next
            && std::env::var("CANADA_SCRATCH_POST_HTTP_VERIFY_PROFILE").as_deref() == Ok("1"),
            "verification profile requires an idle isolated1M worker");
        let long: Value = serde_json::from_slice(include_bytes!("../tests/fixtures/target-agentic-long15800-v1.json"))?;
        let mut prompt: Vec<u32> = serde_json::from_value(long["batches"][0].clone())?;
        let teacher: Value = serde_json::from_slice(include_bytes!("../tests/fixtures/fixed-verify-python-v1.json"))?;
        let teacher: Vec<u32> = serde_json::from_value(teacher["teacher_forced_tokens"].clone())?;
        ensure!(prompt.len() == 15800 && teacher.len() == 512, "verification profile fixture identity");
        prompt.extend_from_slice(&teacher);
        prompt.extend_from_slice(&teacher);
        let long_profile=match std::env::var("CANADA_SCRATCH_PROFILE_VERIFY_262K") {
            Ok(value)=>{ensure!(value=="1","invalid long verification profile flag");true},
            Err(std::env::VarError::NotPresent)=>false,
            Err(error)=>return Err(error.into()),
        };
        if long_profile {
            let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
            ensure!(format!("{:x}",Sha256::digest(raw))=="107ceccbefc53247fc827cdee7eb3908f7b89a907d8c7a2d3de3ef6b95057fbb",
                "long profile fixture identity");
            let corpus:Value=serde_json::from_slice(raw)?;
            prompt=serde_json::from_value(corpus["tokens"].clone())?;
            ensure!(prompt.len()==262656,"long profile corpus extent");
            prompt.truncate(262144);
        }
        self.control.agree(b"post-http-verify-profile-v1")?;
        let begin = Command::Begin {images: Vec::new(), id: 600001, seed: 20261101, greedy:false, limit: 512, tokens: prompt.clone() }.encode()?;
        let wire = self.control.command((rank == 0).then_some(begin.as_slice()))?;
        let first = self.execute(Command::decode(&wire)?)?;
        let mut emitted: Vec<u32> = serde_json::from_value(first["output_ids"].clone())?;
        // Startup warms the dense short-context bank. Warm the actual current
        // bank before arming the profiler; zero acceptance retains causal state.
        let carry=self.verify.carry_digests()?;
        let encoder=self.verify.encoder_tail_digest()?;
        self.verify.begin_verification(&teacher[..6],&self.comm,&self.stream)?;
        self.verify.verification_greedy(&self.stream)?;
        self.verify.finish_verification(0,&self.comm,&self.stream)?;
        ensure!(self.verify.verification_graph_is_warm(6)
            && self.verify.carry_digests()?==carry && self.verify.encoder_tail_digest()?==encoder,
            "current-bank profile warmup changed accepted state");
        self.session.as_mut().unwrap().arm_verify_profile()?;
        let mut next = self.agree_progress(&first, false)?;
        let mut steps = Vec::new();
        let finish_reason = loop {
            ensure!(steps.len() < 512, "verification profile request exceeded its bound");
            let event = self.execute(Command::decode(&next)?)?;
            if event["type"] == "finish" { break event["reason"].clone(); }
            ensure!(event["type"] == "tokens", "unexpected verification profile event");
            if event["profiled_verify"]==true {
                std::fs::write(output.join("verify-profile-step.json"),serde_json::to_vec_pretty(&event)?)?;
            }
            next = self.agree_progress(&event, false)?;
            emitted.extend(serde_json::from_value::<Vec<u32>>(event["output_ids"].clone())?);
            steps.push(event);
        };
        let profiled: Vec<_> = steps.iter().filter(|s|s["profiled_verify"] == true).collect();
        ensure!(profiled.len() == 1 && self.current.is_none() && emitted.len() <= 512,
            "verification profile did not complete exactly one capture");
        let step = profiled[0];
        ensure!(step["verified_rows"] == 6 && step["proposal_source"] == "dspark"
            && step["profile_start_position"].as_u64().is_some_and(|p|p >= prompt.len() as u64)
            && step["profile_append_metrics"]["captured_new_graph"] == false,
            "verification profile captured an unwarmed or unexpected path");
        Ok(json!({"rank":rank,"prompt_tokens":prompt.len(),"seed":20261101,
            "context_capacity":self.large.context_capacity(),"begin":first,"steps":steps,
            "output_ids":emitted,"finish_reason":finish_reason,
            "final_outputs":self.verify.valid_output_digests()?,
            "memory":self.cuda.admission_memory()?,"profiled_verify_steps":1,
            "profile_start_cpu_rendezvous":true,
            "current_graph_bank_warmed":true,
            "performance_result":false,"full_engine_acceptance":false,
            "scope":format!("One warm neural-proposal Verify6 step on a public{}-token fixture, after HTTP shutdown. Same resident candidate, T1/topP1, canonical target precision, causal verification and history policy. CUDA capture covers target verification and rejection readout; excludes draft, prefill, commit and HTTP. Instrumented timing is diagnostic only.",prompt.len())}))
    }
    /// Short matched wire-protocol screen using the actual model-owning worker,
    /// after HTTP shutdown. Public fixture histories never reach the listener.
    fn control_probe(&mut self, output: &Path, rank: usize) -> Result<Value> {
        ensure!(self.current.is_none() && self.large.context_capacity() == 1048576
            && std::env::var("CANADA_SCRATCH_POST_HTTP_CONTROL").as_deref() == Ok("1"),
            "control screen requires an idle isolated1M worker");
        let short: Value = serde_json::from_slice(include_bytes!("../tests/fixtures/target-agentic-python-errors-v2.json"))?;
        let short: Vec<u32> = serde_json::from_value(short["batches"][0].clone())?;
        let long: Value = serde_json::from_slice(include_bytes!("../tests/fixtures/target-agentic-long15800-v1.json"))?;
        let mut long: Vec<u32> = serde_json::from_value(long["batches"][0].clone())?;
        let teacher: Value = serde_json::from_slice(include_bytes!("../tests/fixtures/fixed-verify-python-v1.json"))?;
        let teacher: Vec<u32> = serde_json::from_value(teacher["teacher_forced_tokens"].clone())?;
        ensure!(short.len() == 511 && long.len() == 15800 && teacher.len() == 512,
            "control screen fixture identity");
        long.extend_from_slice(&teacher);
        long.extend_from_slice(&teacher);
        let mut records = Vec::new();
        let mut id = 500000u64;
        for (prompt_index, prompt) in [&short, &long].into_iter().enumerate() {
            for repeat in 0..5 {
                // Each measured seed gets both AB and BA. The first screen
                // had three measured pairs, leaving order confounded with seed.
                let sample = if repeat == 0 { 0 } else { (repeat + 1) / 2 };
                let first_variant = [0, 0, 1, 1, 0][repeat];
                let seed = 20261101 + (prompt_index * 100 + sample) as u64;
                for variant in [first_variant, 1 - first_variant] {
                    self.control.agree(&[prompt_index as u8, repeat as u8, variant as u8])?;
                    self.progress_next = variant == 1;
                    id += 1;
                    let begin = Command::Begin {images: Vec::new(), id, seed, greedy:false, limit: 128, tokens: prompt.clone() }.encode()?;
                    let started = Instant::now();
                    let wire = self.control.command((rank == 0).then_some(begin.as_slice()))?;
                    let event = self.execute(Command::decode(&wire)?)?;
                    let mut authorized_next = if self.progress_next {
                        Some(self.agree_progress(&event, false)?)
                    } else { None };
                    let prefill_seconds = started.elapsed().as_secs_f64();
                    let mut emitted: Vec<u32> = serde_json::from_value(event["output_ids"].clone())?;
                    if repeat == 0 {
                        // Exercise every graph count outside measured decoding.
                        // Zero accepted inputs preserves the agreed frontier.
                        for live in 1..=6 {
                            self.verify.begin_verification(&teacher[..live], &self.comm, &self.stream)?;
                            self.verify.verification_greedy(&self.stream)?;
                            self.verify.finish_verification(0, &self.comm, &self.stream)?;
                        }
                    }
                    let captures_before = self.verify.capture_statistics().1;
                    let started = Instant::now();
                    let mut steps = Vec::new();
                    let finish_reason = loop {
                        let iteration = Instant::now();
                        let wire = if let Some(next) = authorized_next.take() { next } else {
                            let command = if rank == 0 { Some(self.active_command(false)?.encode()?) } else { None };
                            self.control.command(command.as_deref())?
                        };
                        let mut event = self.execute(Command::decode(&wire)?)?;
                        if event["type"] == "finish" { break event["reason"].clone(); }
                        ensure!(event["type"] == "tokens", "unexpected control probe event");
                        if self.progress_next { authorized_next = Some(self.agree_progress(&event, false)?); }
                        event["control_and_execution_seconds"] = json!(iteration.elapsed().as_secs_f64());
                        emitted.extend(serde_json::from_value::<Vec<u32>>(event["output_ids"].clone())?);
                        steps.push(event);
                    };
                    let decode_seconds = started.elapsed().as_secs_f64();
                    ensure!(!emitted.is_empty() && emitted.len() <= 128 && self.current.is_none(),
                        "control probe did not finish its bounded request");
                    let new_captures = self.verify.capture_statistics().1 - captures_before;
                    ensure!(repeat == 0 || new_captures == 0, "control timing recaptured a verifier graph");
                    let final_state = json!({"frontier":self.history.frontier(),"pending":self.history.pending(),
                        "carry":self.verify.carry_digests()?,"encoder":self.verify.encoder_tail_digest()?,
                        "outputs":self.verify.valid_output_digests()?,
                        "history_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(self.history.committed())?))});
                    let record = json!({"rank":rank,"prompt_index":prompt_index,"prompt_tokens":prompt.len(),
                        "repeat":repeat,"warmup":repeat==0,"variant":variant,"first_variant":first_variant,"progress_next":self.progress_next,
                        "seed":seed,"output_ids":emitted,"finish_reason":finish_reason,"steps":steps,
                        "final_state":final_state,"prefill_seconds":prefill_seconds,"decode_seconds":decode_seconds,
                        "interval_tokens":emitted.len()-1,"new_verifier_captures":new_captures});
                    std::fs::write(output.join(format!("control-p{prompt_index}-r{repeat}-v{variant}.json")),serde_json::to_vec_pretty(&record)?)?;
                    eprintln!("control rank{rank} prompt{} repeat{repeat} progress{variant}: {:.2}tok/s",prompt.len(),(emitted.len()-1)as f64/decode_seconds);
                    records.push(record);
                }
            }
        }
        Ok(json!({"rank":rank,"records":records,"control_probe_schema":2,"context_capacity":self.large.context_capacity(),
            "reused_after_http":true,"memory":self.cuda.admission_memory()?,"full_engine_acceptance":false,
            "scope":"20 actual-worker trials: two public prefixes, one warmup pair plus four measured128-output pairs each. Every measured seed receives both AB and BA order. Same canonical target, sampler, seeds, precision and model schedules; only the CPU command protocol changes. Both paths require all-rank committed-state agreement before output. Candidate acknowledgment carries the next step/finish authorization, avoiding a separate command exchange. Compare sampled tokens, decisions, retained state and final logits within each pair and repeat trajectories across both orders. Short transport/timing diagnostic, not outcome-quality or production fault-recovery qualification."}))
    }
}

/// Finite native qualification driver, retaining the same model allocations
/// through fresh/retained requests, natural stops, limits and cancellations.
pub fn run_file(path: &Path) -> Result<()> {
    let retention_probe=std::env::var("CANADA_SCRATCH_CACHE_RETENTION_SCREEN").as_deref()==Ok("1");
    ensure!(!retention_probe || cfg!(feature="storage-fault-injection"),
        "cache refusal screen requires a non-serving fault-injection build");
    let p: WorkerConfig = serde_json::from_slice(&std::fs::read(path)?)?;
    #[cfg(feature = "serving")]
    ensure!(
        p.http.is_none(),
        "file pilot cannot also own an HTTP listener"
    );
    ensure!(
        p.requests.is_some() == (p.rank == 0),
        "only rank zero owns the request source"
    );
    let mut requests = if let Some(path) = &p.requests {
        ensure!(
            std::fs::metadata(path)?.len() <= 32 * 1024 * 1024,
            "resident pilot request file bound"
        );
        let requests: Vec<Request> = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(
            (1..=64).contains(&requests.len()),
            "resident pilot request count"
        );
        for request in &requests {
            ensure!(
                request.input_ids.is_some() != request.append_tokens.is_some()
                    && request.seed < (1u64 << 63)
                    && (1..=262144).contains(&request.max_new_tokens)
                    && request
                        .cancel_after_tokens
                        .is_none_or(|n| n > 0 && n < request.max_new_tokens),
                "invalid resident pilot request controls"
            );
            for ids in [&request.input_ids, &request.append_tokens]
                .into_iter()
                .flatten()
            {
                ensure!(
                    ids.len() <= p.context && ids.iter().all(|&id| id < 129280),
                    "resident pilot token domain"
                );
            }
        }
        requests.into_iter()
    } else {
        Vec::new().into_iter()
    };
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&p.events)?;
    let mut events = std::io::BufWriter::new(file);
    let mut worker = Worker::load(&p)?;
    let history_screen=std::env::var("CANADA_SCRATCH_HISTORY_ASSOCIATIVITY_SCREEN").as_deref()==Ok("1");
    let dense_rows_screen=std::env::var("CANADA_SCRATCH_DENSE_LIVE_ROWS_SCREEN").as_deref()==Ok("1");
    if history_screen || dense_rows_screen {
        // Each finite screen ends at a restored frontier with no active
        // request. Compatible comparisons can share one model load.
        if history_screen {history_associativity_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;}
        if dense_rows_screen {dense_live_rows_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;}
        if std::env::var("CANADA_SCRATCH_MHC_LIVE_ROWS_SCREEN").as_deref()==Ok("1") {
            worker.target_lineage=worker.target_lineage.checked_add(1).context("combined probe lineage exhausted")?;
            worker.verify.reset(worker.target_lineage,&worker.comm,&worker.stream)?;
            worker.large.handoff_from(&worker.verify)?;
            return crate::mhc_live_probe::run(&worker.cuda,&mut worker.large,&mut worker.verify,
                worker.long_verify.as_mut().context("MHC live screen long verifier")?,&mut worker.wides,
                &worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank,worker.target_lineage);
        }
        return Ok(());
    }
    #[cfg(feature = "storage-fault-injection")]
    if std::env::var("CANADA_SCRATCH_CACHE_PRESSURE_SCREEN").as_deref()==Ok("1") {
        return cache_pressure_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_MASK_STORAGE_SCREEN").as_deref()==Ok("1") {
        return mask_storage_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    #[cfg(feature = "storage-fault-injection")]
    if retention_probe {
        return cache_retention_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_ACTIVE_ENCODER_CHECKPOINT_SCREEN").as_deref()==Ok("1") {
        active_encoder_checkpoint_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;
        if std::env::var("CANADA_SCRATCH_ENCODER_CACHE_SCREEN").as_deref()==Ok("1") {
            worker.encoder_prompt_checkpoint=true;worker.encoder_active_frontier=false;worker.prompt_checkpoint=true;
            encoder_cache_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;
        }
        return Ok(());
    }
    if std::env::var("CANADA_SCRATCH_ENCODER_FRONTIER_SCREEN").as_deref()==Ok("1") {
        return crate::encoder_frontier_probe::run(&worker.cuda,&mut worker.large,&mut worker.verify,
            &mut worker.wides,&worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank);
    }
    if std::env::var("CANADA_SCRATCH_SPLIT_ATTENTION_GENERATION_SCREEN").as_deref()==Ok("1") {
        return split_generation_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_ENCODER_CACHE_SCREEN").as_deref()==Ok("1") {
        return encoder_cache_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_ENCODER_CHECKPOINT_SCREEN").as_deref()==Ok("1") {
        return crate::encoder_checkpoint_probe::run(&worker.cuda,&worker.core,&mut worker.large,
            &mut worker.verify,&mut worker.wides,&mut worker.prepared_session,&mut worker.session,
            &worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank);
    }
    if std::env::var("CANADA_SCRATCH_SPLIT_ATTENTION_SCREEN").as_deref()==Ok("1") {
        crate::split_attention_probe::run(&worker.cuda,&mut worker.large,&mut worker.verify,
            &mut worker.wides,&worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank)?;
        // Both probes use an idle target owner. Reuse this model load while
        // starting the append screen with an empty ordinary-precision history.
        if std::env::var("CANADA_SCRATCH_APPEND_SPLIT_ATTENTION_SCREEN").as_deref()==Ok("1") {
            worker.verify.set_split_attention_for_probe(0,&worker.stream)?;
            worker.verify.reset(7_970_000,&worker.comm,&worker.stream)?;
            worker.large.handoff_from(&worker.verify)?;
            return crate::split_attention_probe::run_append(&worker.cuda,&mut worker.large,&mut worker.verify,
                worker.long_verify.as_mut().ok_or_else(||anyhow::anyhow!("append split needs M16"))?,
                &mut worker.wides,&worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank);
        }
        return Ok(());
    }
    if std::env::var("CANADA_SCRATCH_MHC_LIVE_ROWS_SCREEN").as_deref()==Ok("1") {
        return crate::mhc_live_probe::run(&worker.cuda,&mut worker.large,&mut worker.verify,
            worker.long_verify.as_mut().ok_or_else(||anyhow::anyhow!("live-row screen needs the ordinary long verifier"))?,
            &mut worker.wides,&worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank,worker.target_lineage);
    }
    if std::env::var("CANADA_SCRATCH_MHC_TWO_MMA_SCREEN").as_deref()==Ok("1") {
        crate::mhc_stats_probe::run(&worker.cuda,&mut worker.large,&mut worker.verify,
            worker.long_verify.as_mut().ok_or_else(||anyhow::anyhow!("two-MMA screen needs M16"))?,
            &mut worker.wides,&worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank)?;
        if std::env::var("CANADA_SCRATCH_SOURCE_PRIOR_SCREEN").as_deref()!=Ok("1"){return Ok(());}
        // The kernel probe restores the original statistics path on both
        // profiles. Reset its target-only history before the independent policy
        // comparison; the worker/session has not begun a request yet.
        ensure!(worker.current.is_none()&&worker.history.frontier()==0,"combined probe worker history");
        worker.target_lineage=8_080_000;
        worker.verify.reset(worker.target_lineage,&worker.comm,&worker.stream)?;
        worker.large.handoff_from(&worker.verify)?;
    }
    if std::env::var("CANADA_SCRATCH_MOE_INPUT_ADDRESS_SCREEN").as_deref()==Ok("1") {
        return crate::moe_input_address_probe::run(&worker.cuda,&mut worker.large,&mut worker.verify,
            &mut worker.wides,&worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank);
    }
    if std::env::var("CANADA_SCRATCH_WO_B_PDL_SCREEN").as_deref()==Ok("1") {
        return crate::wo_b_probe::run(&worker.cuda,&mut worker.large,&mut worker.verify,
            &mut worker.wides,&worker.comm,&worker.stream,&mut worker.control,&p.kernels,p.events.parent().unwrap(),p.rank);
    }
    if std::env::var("CANADA_SCRATCH_APPEND_SPLIT_ATTENTION_SCREEN").as_deref()==Ok("1") {
        return crate::split_attention_probe::run_append(&worker.cuda,&mut worker.large,&mut worker.verify,
            worker.long_verify.as_mut().ok_or_else(||anyhow::anyhow!("append split needs M16"))?,
            &mut worker.wides,&worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank);
    }
    if std::env::var("CANADA_SCRATCH_LOSSLESS_HEAD_SCREEN").as_deref()==Ok("1") {
        return crate::lossless_head_probe::run(&worker.cuda,&mut worker.large,&mut worker.verify,
            &mut worker.wides,&worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank);
    }
    if std::env::var("CANADA_SCRATCH_APPEND_DISTRIBUTION_SCREEN").as_deref()==Ok("1") {
        return crate::append_distribution_probe::run(&worker.cuda,&mut worker.large,&mut worker.verify,
            worker.long_verify.as_mut().ok_or_else(||anyhow::anyhow!("append diagnostic requires M16"))?,
            &mut worker.wides,&worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank);
    }
    #[cfg(feature = "serving")]
    let startup_warmup = if std::env::var("CANADA_SCRATCH_FILE_STARTUP_WARMUP").as_deref()==Ok("1") {
        Some(worker.warm_before_ready(p.rank)?)
    } else {None};
    #[cfg(not(feature = "serving"))]
    let startup_warmup:Option<Value> = None;
    if std::env::var("CANADA_SCRATCH_POOLED_ROUTE_CAPTURE").as_deref()==Ok("1") {
        return pooled_route_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),
            serde_json::to_value(startup_warmup)?);
    }
    if std::env::var("CANADA_SCRATCH_BLOCK_PROGRESS_SCREEN").as_deref()==Ok("1") {
        return block_progress_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),
            serde_json::to_value(startup_warmup)?);
    }
    if std::env::var("CANADA_SCRATCH_FIXED_PREFIX_QUALITY_SCREEN").as_deref()==Ok("1") {
        return fixed_prefix_quality_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_SAMPLER_READOUT_SCREEN").as_deref()==Ok("1") {
        return sampler_readout_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),
            serde_json::to_value(startup_warmup)?);
    }
    if std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_INT8_SCREEN").as_deref()==Ok("1") {
        return private_int8_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),
            serde_json::to_value(startup_warmup)?);
    }
    if std::env::var("CANADA_SCRATCH_SOURCE_PRIOR_SCREEN").as_deref()==Ok("1") {
        return source_prior_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),
            serde_json::to_value(startup_warmup)?);
    }
    if std::env::var("CANADA_SCRATCH_ROOT_FEEDBACK_SCREEN").as_deref()==Ok("1") {
        return worker.root_feedback_screen(p.rank,p.events.parent().unwrap(),
            serde_json::to_value(startup_warmup)?);
    }
    if std::env::var("CANADA_SCRATCH_ROOT_TEMPERATURE_SCREEN").as_deref()==Ok("1") {
        return worker.root_temperature_screen(p.rank,p.events.parent().unwrap(),
            startup_warmup.unwrap_or(Value::Null));
    }
    serde_json::to_writer(
        &mut events,
        &json!({"type":"ready","rank":p.rank,"identity":worker.identity,
        "startup_warmup":startup_warmup,
        "session_prepare_seconds":worker.session_prepare_seconds,
        "prefill_capture_once":worker.large.prefill_capture_once(),
        "coalesced_moe_codec_layers":worker.wides.iter().map(Target::coalesced_codec_layers).sum::<usize>(),
        "live_moe_layers":worker.verify.live_moe_layers(),
        "adaptive_sparse_index_layers":worker.verify.adaptive_sparse_index_layers(),
        "compact_sparse_index_layers":worker.verify.compact_sparse_index_layers(),
        "packed_index_gather_layers":worker.verify.packed_index_gather_layers(),
        "packed_sparse_gather_layers":worker.verify.packed_sparse_gather_layers(),
        "cached_index_gather_layers":worker.verify.cached_index_gather_layers(),
        "mapped_metadata_bytes":worker.verify.mapped_metadata_bytes(),
        "staged_metadata_bytes":worker.verify.staged_metadata_bytes(),
        "fused_shared_quantization_layers":worker.verify.fused_shared_quantization_layers(),
        "cuda_module_loading":worker.cuda.module_loading_mode()?,
        "transport_memory":worker.comm.transport_memory(),
        "nccl_init_seconds":worker.nccl_init_seconds,
        "nccl_runtime_connect_requested":std::env::var("NCCL_RUNTIME_CONNECT").ok(),
        "context_capacity":p.context,"memory":worker.cuda.admission_memory()?,"production_qualified":false}),
    )?;
    events.write_all(b"\n")?;
    events.flush()?;
    let mut next_id = 1;
    let mut cancel_after = None;
    let outcome = (|| {
        if std::env::var("CANADA_SCRATCH_WORKER_INTERLEAVE_SCREEN").as_deref() == Ok("1") {
            let result = interleave_probe::run(&mut worker, &p)?;
            serde_json::to_writer(&mut events, &result)?;
            events.write_all(b"\n")?;
            events.flush()?;
            return Ok(());
        }
        loop {
            let iteration = Instant::now();
            let bytes = if p.rank == 0 {
                let command = if worker.current.is_some() {
                    if worker.history.pending() == Some(worker.cfg.eos_token_id) {
                        Command::Finish(Finish::Stop)
                    } else if worker.emitted == worker.limit {
                        Command::Finish(Finish::Length)
                    } else if cancel_after.is_some_and(|n| worker.emitted >= n) {
                        Command::Finish(Finish::Cancelled)
                    } else {
                        Command::Step
                    }
                } else if let Some(request) = requests.next() {
                    let tokens = request.prompt(&worker.history)?;
                    let images=request.images.iter().map(|i|i.descriptor.clone()).collect::<Vec<_>>();
                    if !images.is_empty() {
                        worker.media.as_mut().ok_or_else(||anyhow::anyhow!("vision fixture needs a configured encoder"))?
                            .set_inputs(&request.images.iter().map(|i|i.load()).collect::<Result<Vec<_>>>()?)?;
                    }
                    cancel_after = request.cancel_after_tokens;
                    let id = next_id;
                    next_id += 1;
                    Command::Begin {images,
                        id,
                        seed: request.seed,
                        greedy: false,
                        limit: request.max_new_tokens,
                        tokens,
                    }
                } else {
                    Command::Shutdown
                };
                Some(command.encode()?)
            } else {
                None
            };
            let wire = worker.control.command(bytes.as_deref())?;
            let command = Command::decode(&wire)?;
            let shutdown = matches!(command, Command::Shutdown);
            let mut event = worker.execute(command)?;
            event["control_and_execution_seconds"] = json!(iteration.elapsed().as_secs_f64());
            // Only acknowledged committed output reaches this sink. A failed
            // write retires the group; no EOF-to-success conversion exists.
            serde_json::to_writer(&mut events, &event)?;
            events.write_all(b"\n")?;
            events.flush()?;
            if shutdown {
                break;
            }
        }
        Ok(())
    })();
    if outcome.is_err() {
        worker.history.retire();
    }
    outcome
}

/// The same model-owning loop behind a real bounded HTTP request queue. Only
/// rank zero observes network cancellation; it publishes that decision to all
/// ranks between committed steps. Idle heartbeats touch CPU sockets only.
#[cfg(feature = "serving")]
pub fn run_http(path: &Path) -> Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_CACHE_RETENTION_SCREEN").as_deref()!=Ok("1"),
        "cache refusal probe is restricted to the finite file runner");
    use crate::{http_server, protocol::BackendEvent};
    use deepseek_recipe::stream::InferenceFinishReason;
    let p: WorkerConfig = serde_json::from_slice(&std::fs::read(path)?)?;
    // HTTP upload/conversion and image decoding reserve bounded workspace
    // while they own it. Actual retained inputs remain in host/cgroup charge;
    // an idle server need not permanently reserve the largest possible input.
    let post_http_causal = match std::env::var("CANADA_SCRATCH_POST_HTTP_CAUSAL") {
        Ok(value) => {
            ensure!(value == "1" && p.context == 1048576,
                "post-HTTP causal pilot requires the admitted1M profiles");
            true
        }
        Err(std::env::VarError::NotPresent) => false,
        Err(error) => return Err(error.into()),
    };
    let post_http_capture = match std::env::var("CANADA_SCRATCH_POST_HTTP_CAPTURE") {
        Ok(value) => {
            ensure!(value == "1" && p.context == 1048576 && !post_http_causal,
                "post-HTTP capture screen requires an isolated1M worker");
            true
        }
        Err(std::env::VarError::NotPresent) => false,
        Err(error) => return Err(error.into()),
    };
    ensure!(
        p.requests.is_none() && p.http.is_some() == (p.rank == 0),
        "native HTTP source belongs to rank zero only"
    );
    let progress_next = match std::env::var("CANADA_SCRATCH_PROGRESS_NEXT") {
        Ok(value) => {
            ensure!(value == "0" || value == "1", "invalid progress command policy");
            value == "1"
        }
        // s290: all-rank exact paired results, including both orders per seed.
        Err(std::env::VarError::NotPresent) => true,
        Err(error) => return Err(error.into()),
    };
    let http_interleave=match std::env::var("CANADA_SCRATCH_HTTP_INTERLEAVE") {
            Ok(value)=>{ensure!((value=="0" || value=="1") && (value=="0" || p.context==1048576),
                "HTTP interleave policy must be0/1 and enabled scheduling requires1M capacity");value=="1"},
        Err(std::env::VarError::NotPresent)=>false,
        Err(error)=>return Err(error.into()),
    };
    let post_http_control = match std::env::var("CANADA_SCRATCH_POST_HTTP_CONTROL") {
        Ok(value) => {
            ensure!(value == "1" && p.context == 1048576 && !post_http_causal && !post_http_capture,
                "post-HTTP control screen requires an isolated1M worker");
            true
        }
        Err(std::env::VarError::NotPresent) => false,
        Err(error) => return Err(error.into()),
    };
    let post_http_verify_profile = match std::env::var("CANADA_SCRATCH_POST_HTTP_VERIFY_PROFILE") {
        Ok(value) => {
            ensure!(value == "1" && p.context == 1048576 && progress_next
                && !post_http_causal && !post_http_capture && !post_http_control,
                "verification profile requires an isolated1M worker");
            true
        }
        Err(std::env::VarError::NotPresent) => false,
        Err(error) => return Err(error.into()),
    };
    let post_http_long_quality=match std::env::var("CANADA_SCRATCH_POST_HTTP_LONG_QUALITY") {
        Ok(value)=>{
            ensure!(value=="1" && p.context==1048576 && !post_http_causal
                && !post_http_capture && !post_http_control && !post_http_verify_profile,
                "long quality requires an isolated current worker after HTTP shutdown");
            true
        },
        Err(std::env::VarError::NotPresent)=>false,
        Err(error)=>return Err(error.into()),
    };
    let queue = p
        .http
        .as_ref()
        .map(|config| http_server::start(config, p.context, p.vision.is_some()))
        .transpose()?;
    let mut events = crate::event_log::EventLog::from_env(&p.events)?;
    let mut worker = Worker::load(&p)?;
    ensure!(!http_interleave || (!post_http_causal && !post_http_capture && !post_http_control
        && !post_http_verify_profile && !post_http_long_quality),
        "HTTP interleave experiment needs an isolated worker");
    let startup_warmup = match std::env::var("CANADA_SCRATCH_WARMUP_BEFORE_READY") {
        Ok(value) => {
            ensure!(value=="0" || value=="1","invalid startup warmup policy");
            if value=="1" {Some(worker.warm_before_ready(p.rank)?)} else {None}
        }
        // Keep the tiny diagnostic capacity usable; the public warmup needs
        // room for its 128-token prefix and bounded continuation.
        Err(std::env::VarError::NotPresent) if p.context>=256 =>
            Some(worker.warm_before_ready(p.rank)?),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    worker.progress_next = progress_next;
    let ready = json!({"type":"ready","rank":p.rank,"identity":worker.identity,
        "session_cache_budget_bytes":worker.session_cache_budget,"prompt_checkpoint":worker.prompt_checkpoint,
        "encoder_prompt_checkpoint":worker.encoder_prompt_checkpoint,
        "encoder_active_frontier":worker.encoder_active_frontier,
        "encoder_checkpoint_defer128":crate::target::defer_checkpoint_decoder128()?,
        "generation_fp32_attention":crate::layer_attention::generation_fp32_attention_enabled()?,
        "encoder_upgrade_reconstructed":worker.encoder_upgrade_reconstructed,
        "prompt_checkpoint_tail":worker.prompt_checkpoint_tail,
        "prompt_cache_logits":worker.prompt_cache_logits,
        "shared_image_budget_bytes":worker.shared_image_budget,
        "retain_restored_checkpoint":worker.retain_restored_checkpoint,
        "session_cache_policy":if worker.session_cache_long_first {"long_parents_and_recent"}else{"fifo"},
        "engram_staging":worker.verify.engram_input_kind(),
        "paired_engram_io":worker.verify.paired_engram_io_enabled(),
        "session_prepare_seconds":worker.session_prepare_seconds,
        "startup_warmup":startup_warmup,
        "history_verifier":worker.long_verify.as_ref().map(|target|json!({
            "compact_index_layers":target.grouped16_index_layers(),
            "query_overlap_layers":target.query_overlap_layers(),
            "coefficient_overlap_layers":target.attention_coefficient_overlap_layers()})),
        "progress_next":worker.progress_next,"http_interleave":http_interleave,
        "prefill_capture_once":worker.large.prefill_capture_once(),
        "live_moe_layers":worker.verify.live_moe_layers(),
        "adaptive_sparse_index_layers":worker.verify.adaptive_sparse_index_layers(),
        "compact_sparse_index_layers":worker.verify.compact_sparse_index_layers(),
        "packed_index_gather_layers":worker.verify.packed_index_gather_layers(),
        "packed_sparse_gather_layers":worker.verify.packed_sparse_gather_layers(),
        "cached_index_gather_layers":worker.verify.cached_index_gather_layers(),
        "mapped_metadata_bytes":worker.verify.mapped_metadata_bytes(),
        "staged_metadata_bytes":worker.verify.staged_metadata_bytes(),
        "fused_shared_quantization_layers":worker.verify.fused_shared_quantization_layers(),
        "cuda_module_loading":worker.cuda.module_loading_mode()?,
        "transport_memory":worker.comm.transport_memory(),
        "nccl_init_seconds":worker.nccl_init_seconds,
        "nccl_runtime_connect_requested":std::env::var("NCCL_RUNTIME_CONNECT").ok(),
        "context_capacity":p.context,"memory":worker.cuda.admission_memory()?,"production_qualified":false});
    events.record(&ready)?;
    if let Some(queue) = &queue {
        if http_interleave {
            queue.enable_decode_priority();
            if worker.background_prefill_enabled { queue.enable_background_prefill(); }
        }
        worker.control.observe_progress(queue.progress_counter());
        queue.ready(&worker.identity);
    }
    if http_interleave {
        return interleave_http::run(&mut worker,queue.as_ref(),&mut events);
    }
    let mut active: Option<http_server::Job> = None;
    let mut next_id = 1u64;
    let mut authorized_next = None;
    let outcome = (|| -> Result<()> {
        loop {
            let iteration = Instant::now();
            let bytes = if authorized_next.is_some() { None } else if let Some(queue) = &queue {
                let command = if !queue.stopping() && queue.input_reclaim_target()>0 {
                    Command::ReclaimInputs{headroom:queue.input_reclaim_target()}
                } else if worker.current.is_some() {
                    let job = active
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("active native request lost HTTP owner"))?;
                    worker.active_command(job.cancelled() || queue.stopping())?
                } else if queue.stopping() {
                    Command::Shutdown
                } else if let Some(mut job) = queue.next(Duration::from_millis(500))? {
                    let tokens = std::mem::take(&mut job.tokens);
                    let id = next_id;
                    next_id = next_id
                        .checked_add(1)
                        .ok_or_else(|| anyhow::anyhow!("request counter exhausted"))?;
                    if !job.images.is_empty() {
                        worker.media.as_mut().ok_or_else(|| anyhow::anyhow!("image request without native vision"))?
                            .set_inputs(&std::mem::take(&mut job.image_inputs))?;
                    }
                    let command = Command::Begin {images: std::mem::take(&mut job.images),
                        id,
                        seed: job.seed,
                        greedy: job.greedy,
                        limit: job.limit,
                        tokens,
                    };
                    active = Some(job);
                    command
                } else {
                    Command::Idle
                };
                Some(command.encode()?)
            } else {
                None
            };
            let wire = if let Some(next) = authorized_next.take() { next }
                else { worker.control.command(bytes.as_deref())? };
            let command = Command::decode(&wire)?;
            let began = matches!(command, Command::Begin { .. });
            let idle = matches!(command, Command::Idle);
            let shutdown = matches!(command, Command::Shutdown);
            let finished = matches!(command, Command::Finish(_));
            let mut event = worker.execute_http(command,queue.as_ref(),active.as_ref())?;
            if began {event["request_started"]=json!(true);}
            let input_refused=event["type"]=="input_refused";
            if began && !input_refused {event["vision"]=worker.media_receipt.clone();}
            if worker.progress_next && event["type"] == "tokens" {
                let cancelled = active.as_ref().is_some_and(|job| job.cancelled())
                    || queue.as_ref().is_some_and(|queue| queue.stopping());
                let agreement_started = Instant::now();
                authorized_next = Some(worker.agree_progress(&event, cancelled)?);
                event["progress_agreement_seconds"] = json!(agreement_started.elapsed().as_secs_f64());
            }
            event["progress_next"] = json!(worker.progress_next);
            if let Some(job) = &active {
                if began && !input_refused {
                    job.record_prefill_timing(&event);
                    job.emit(BackendEvent::Ready {
                        fingerprint: worker.identity.clone(),
                        cached_tokens: event["cached_tokens"].as_u64().unwrap() as usize,
                    });
                    event["queue_and_prefill_seconds"] =
                        json!(job.enqueued.elapsed().as_secs_f64());
                }
                if input_refused {
                    job.emit(BackendEvent::Failed(event["message"].as_str()
                        .ok_or_else(||anyhow::anyhow!("input refusal message missing"))?.into()));
                } else if event["type"] == "tokens" {
                    job.emit(BackendEvent::Tokens(serde_json::from_value(
                        event["output_ids"].clone(),
                    )?));
                } else if finished {
                    job.emit(match event["reason"].as_str() {
                        Some("stop") => BackendEvent::Finish(InferenceFinishReason::Stop),
                        Some("length") => BackendEvent::Finish(InferenceFinishReason::Length),
                        Some("cancelled") => {
                            BackendEvent::Failed("request cancelled before completion".into())
                        }
                        _ => anyhow::bail!("unknown committed worker completion"),
                    });
                }
                event["http_request_id"] = json!(job.id);
            }
            if !idle {
                // Real API histories are private data. Operational evidence
                // contains counts/timings/identities, never prompt/output IDs.
                redact_http_event(&mut event);
                event["control_and_execution_seconds"] = json!(iteration.elapsed().as_secs_f64());
                events.record(&event)?;
            }
            if finished || input_refused {
                active = None;
            }
            if shutdown {
                break;
            }
        }
        Ok(())
    })();
    if let Some(job) = active.take() {
        job.emit(BackendEvent::Failed("native worker group failed".into()));
    }
    if let Some(queue) = &queue {
        queue.fail();
    }
    if (post_http_control || post_http_verify_profile) && outcome.is_ok() {
        drop(queue);
        let output = p.events.parent().ok_or_else(|| anyhow::anyhow!("HTTP pilot output directory"))?;
        let result = if post_http_verify_profile { worker.verify_profile_probe(p.rank,output) }
            else { worker.control_probe(output, p.rank) };
        worker.history.retire();
        worker.control.retire();
        let result = result?;
        let name = if post_http_verify_profile { "verify-profile-post-http.json" } else { "control-post-http.json" };
        std::fs::write(output.join(name),serde_json::to_vec_pretty(&result)?)?;
        return Ok(());
    }
    worker.history.retire();
    worker.control.retire();
    if outcome.is_err() {events.worker_failed(worker.current);}
    outcome?;
    if post_http_causal {
        // The finite HTTP pilot has shut down. Reuse its allocations for the
        // next public-fixture screen instead of reloading the1M model. No HTTP
        // request can use the reset diagnostic history or altered draft policy.
        drop(queue);
        worker.verify.reset(100000, &worker.comm, &worker.stream)?;
        worker.large.handoff_from(&worker.verify)?;
        let cp = Checkpoint::from_manifest(&p.checkpoint)?;
        let plan = RankPlan::build(&cp, &worker.cfg, p.rank, 4)?;
        let fixture: Value = serde_json::from_slice(include_bytes!(
            "../tests/fixtures/target-agentic-python-errors-v2.json"))?;
        let prefix: Vec<u32> = serde_json::from_value(fixture["batches"][0].clone())?;
        let output = p.events.parent().ok_or_else(|| anyhow::anyhow!("HTTP pilot output directory"))?;
        let result = crate::causal_verify_probe::run(&worker.cuda,&worker.core,&worker.stream,
            &cp,&plan,&mut worker.large,&mut worker.verify,&mut worker.wides,&worker.comm,
            &prefix,&p.dspark_staged,&p.kernels,output,&mut worker.session,true)?;
        std::fs::write(output.join("causal-post-http.json"),serde_json::to_vec_pretty(&result)?)?;
    } else if post_http_capture {
        drop(queue);
        worker.verify.reset(100000, &worker.comm, &worker.stream)?;
        worker.large.handoff_from(&worker.verify)?;
        let output = p.events.parent().ok_or_else(|| anyhow::anyhow!("HTTP pilot output directory"))?;
        let result = crate::prefill_capture_probe::run(&worker.cuda,&mut worker.large,&mut worker.verify,
            &mut worker.wides,&worker.comm,&worker.stream,output,p.rank)?;
        std::fs::write(output.join("prefill-capture-post-http.json"),serde_json::to_vec_pretty(&result)?)?;
    } else if post_http_long_quality {
        drop(queue);
        // Serving is over. Release saved sessions and their budgets before
        // resetting the target to the explicit public diagnostic histories.
        worker.session=None;
        worker.prepared_session=None;
        worker.cached_sessions.clear();
        let output=p.events.parent().ok_or_else(||anyhow::anyhow!("HTTP quality output directory"))?;
        let result=crate::long_tail_quality_probe::run(&worker.cuda,&mut worker.large,&mut worker.verify,
            &mut worker.wides,&worker.comm,&worker.stream,output,p.rank)?;
        std::fs::write(output.join("long-tail-quality-post-http.json"),serde_json::to_vec_pretty(&result)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn request_incident_redaction_keeps_cache_counts_without_output_tokens() {
        let mut event=serde_json::json!({"type":"tokens","request_id":7,
            "request_started":true,"output_ids":[41,42],"pending":43,
            "prompt_tokens":124000,"cached_tokens":123904,
            "retained_session_cache":{"hit":true,"exact_prompt_anchor":44,
                "restored_tokens":123904},
            "prompt_checkpoint":{"saved":false,"late_memory_refused":true}});
        super::redact_http_event(&mut event);
        assert!(event.get("output_ids").is_none() && event.get("pending").is_none());
        assert!(event["retained_session_cache"].get("exact_prompt_anchor").is_none());
        assert_eq!(event["retained_session_cache"]["exact_prompt_hit"],true);
        assert_eq!(event["cached_tokens"],123904);
        assert_eq!(event["prompt_checkpoint"]["late_memory_refused"],true);
        assert_eq!(event["request_started"],true);
    }

    #[test]
    fn advancing_sibling_checkpoints_keep_the_shared_parent_within_existing_limits() {
        let parent=(123904,100);
        let mut entries=vec![parent];
        for checkpoint in [124416,124544,124672,124800] {
            let at=entries.iter().position(|entry|*entry==parent).unwrap();
            let held=entries.remove(at);
            let incoming=(checkpoint,101);
            let mut evictions=encoder_checkpoint_evictions(&entries,&vec![false;entries.len()],
                incoming,Some(held.1),500,true,4).unwrap();
            evictions.sort_unstable();
            for i in evictions.into_iter().rev(){entries.remove(i);}
            // Both immutable images coexist; the shared one is refreshed.
            entries.push(incoming);entries.push(held);
            assert!(entries.contains(&parent));
            assert!(entries.len()<=4&&entries.iter().map(|e|e.1).sum::<usize>()<=500);
        }
    }
    #[test]
    fn parent_reservation_survives_successor_refusal_or_cancellation() {
        let parent=(130816,100);
        let original=vec![(210000,120),(220000,120),(230000,120)];
        for saved in [false,true] {
            let mut entries=original.clone();
            let incoming=(261888,150);
            let mut evictions=encoder_checkpoint_evictions(&entries,&[false;3],incoming,
                Some(parent.1),500,true,4).unwrap();
            evictions.sort_unstable();
            for i in evictions.into_iter().rev(){entries.remove(i);}
            if saved {entries.push(incoming);}
            // Allocation refusal and cancellation never transfer parent ownership.
            entries.push(parent);
            assert!(entries.contains(&parent));
            assert!(entries.len()<=4&&entries.iter().map(|e|e.1).sum::<usize>()<=500);
        }
    }
    #[test]
    fn successor_cannot_credit_parent_bytes_or_reserved_slots() {
        let parent_bytes=930;
        // A 564K parent and near-1M successor exceed the unchanged 2GiB pool.
        assert!(encoder_checkpoint_evictions(&[],&[],(988000,1540),Some(parent_bytes),
            2048,true,4).is_none());
        assert!(encoder_checkpoint_evictions(&[],&[],(124416,100),Some(100),
            500,true,1).is_none());
        // Optional snapshots can yield, but the held parent cannot.
        assert_eq!(encoder_checkpoint_evictions(&[(90000,100),(123904,100)],&[true,false],
            (124416,100),Some(100),300,true,4),Some(vec![0]));
        assert_eq!(encoder_checkpoint_evictions(&[],&[],(124416,100),None,
            100,true,1),Some(vec![]));
    }
    #[test]
    fn reconstructed_upgrade_cannot_spend_the_original_bytes_or_slot() {
        // An eventual same-prefix replacement would fit by itself; its
        // transactional save must first coexist with the immutable original.
        let incoming=(123904,160);
        assert!(encoder_checkpoint_evictions(&[],&[],incoming,Some(100),
            220,true,4).is_none());
        assert!(encoder_checkpoint_evictions(&[],&[],incoming,Some(100),
            500,true,1).is_none());
        assert_eq!(encoder_checkpoint_evictions(&[],&[],incoming,Some(100),
            260,true,2),Some(vec![]));
    }
    #[test]
    fn reconstructed_upgrade_plan_bounds_transient_and_committed_ownership() {
        let original=(123904,100);let complete=(123904,180);
        let others=vec![(90000,50),(91000,60),(92000,70)];
        let mut evictions=encoder_checkpoint_evictions(&others,&[false,true,false],complete,
            Some(original.1),420,true,4).unwrap();
        // Prefer an optional text-logit entry; the held original is not a victim.
        assert_eq!(evictions,vec![1]);
        let mut retained=others;evictions.sort_unstable();
        for i in evictions.into_iter().rev(){retained.remove(i);}
        let transient=retained.iter().map(|e|e.1).sum::<usize>()+original.1+complete.1;
        assert!(transient<=420 && retained.len()+2<=4);
        // Metadata proof only: actual GPU save/refusal/restore is a later check.
        for accepted in [false,true] {
            let mut committed=retained.clone();
            committed.push(if accepted{complete}else{original});
            assert!(committed.iter().map(|e|e.1).sum::<usize>()<=420 && committed.len()<=4);
            assert_eq!(committed.last(),Some(if accepted{&complete}else{&original}));
        }
    }
    #[test]
    fn image_prompt_cache_admission_preserves_long_history_and_shared_limits() {
        let entries=[(131200,500),(14485,100),(431,40),(43,30)];
        // A screenshot can replace a shorter side conversation, without
        // changing the four-slot budget or sacrificing the long parent.
        assert_eq!(prompt_cache_evictions(&entries,(662,50),800,true,4,true),Some(vec![2]));
        assert!(prompt_cache_evictions(&entries,(662,50),800,true,4,false).is_none());
        // Byte pressure can require two short victims; a larger request that
        // would require evicting the parent is rejected transactionally.
        assert_eq!(prompt_cache_evictions(&entries,(662,100),700,true,4,true),Some(vec![2,3]));
        assert!(prompt_cache_evictions(&entries,(662,200),700,true,4,true).is_none());
        assert!(prompt_cache_evictions(&entries,(662,50),800,true,0,true).is_none());
        assert_eq!(prompt_cache_evictions(&entries[..2],(9000,50),800,true,4,false),Some(vec![]));
    }
    #[test]
    fn optional_prompt_logits_yield_before_shared_prefixes_and_conversations() {
        let entries=[(100,100),(110,100),(120,100)];
        assert_eq!(cache_evictions_with_optional(&entries,&[false,true,false],(30,50),350,true,3),Some(vec![1]));
        assert_eq!(cache_evictions_with_optional(&entries,&[false,true,false],(30,50),150,true,3),Some(vec![1,0]));
        assert_eq!(cache_evictions_with_optional(&entries,&[false,false,false],(150,100),300,true,3),
            cache_evictions(&entries,(150,100),300,true,3));
    }
    use super::*;
    #[test]
    fn physical_pressure_prefers_optional_then_longest_superseded_snapshot() {
        let entries=[(8000,30),(200000,70),(210000,80),(100,10)];
        let optional=[false,false,false,true];let superseded=[true,true,true,false];
        let mut removed=Vec::new();
        for expected in [3,2,1,0] {
            let victim=pressure_cache_victim(&entries,&optional,&superseded,220000,4,true,&removed);
            assert_eq!(victim,Some(expected));removed.push(expected);
        }
        assert_eq!(pressure_cache_victim(&entries,&optional,&superseded,220000,4,true,&removed),None);
    }
    #[test]
    fn physical_pressure_keeps_long_parents_for_an_unrelated_short_request() {
        let entries=[(8000,30),(200000,70),(210000,80),(100,10)];
        let optional=[false,false,false,true];let superseded=[false;4];
        assert_eq!(pressure_cache_victim(&entries,&optional,&superseded,100,4,true,&[]),Some(3));
        assert_eq!(pressure_cache_victim(&entries,&optional,&superseded,100,4,true,&[3]),Some(0));
        assert_eq!(pressure_cache_victim(&entries,&optional,&superseded,100,4,true,&[3,0]),None);
        // One external parent reservation reduces available slots to three;
        // the ordinary policy then protects one additional long cache entry.
        assert_eq!(pressure_cache_victim(&entries,&optional,&superseded,100,3,true,&[3,0]),Some(1));
        assert_eq!(pressure_cache_victim(&entries,&optional,&superseded,100,4,false,&[3]),Some(0));
        assert_eq!(pressure_cache_victim(&entries,&optional,&superseded[..3],100,4,true,&[]),None);
    }
    #[test]
    fn cache_retains_long_parent_through_short_requests_and_bounds_every_admission() {
        let mut entries=vec![(60000,130)];
        for _ in 0..8 {
            let incoming=(40,30);
            let mut evictions=cache_evictions(&entries,incoming,220,true,4).unwrap();
            evictions.sort_unstable();
            for i in evictions.into_iter().rev(){entries.remove(i);}
            entries.push(incoming);
            assert!(entries.contains(&(60000,130)));
            assert!(entries.len()<=4 && entries.iter().map(|e|e.1).sum::<usize>()<=220);
        }
        assert_eq!(cache_evictions(&[(60000,130),(40,30),(40,30),(40,30)],(40,30),220,false,4),Some(vec![0]));
        assert!(cache_evictions(&[(60000,130)],(40,30),140,true,4).is_none());
        assert_eq!(cache_evictions(&[(40,30),(40,30),(40,30),(40,30)],(60000,130),220,true,4),Some(vec![0]));
        assert_eq!(cache_evictions(&[(40000,100),(50000,100)],(60000,180),200,true,4),Some(vec![0,1]));
        assert!(cache_evictions(&[],(60000,221),220,true,4).is_none());
        assert_eq!(cache_evictions(&[],(60000,220),220,true,4),Some(vec![]));
    }
    #[test]
    fn similar_length_stale_completions_cannot_exclude_current_prompt() {
        let mut entries=vec![(124200,100),(124240,100),(124300,100),(124340,100)];
        for incoming in [(124032,100),(124160,100)] {
            let mut evictions=cache_evictions(&entries,incoming,500,true,4).unwrap();
            evictions.sort_unstable();
            for index in evictions.into_iter().rev(){entries.remove(index);}
            entries.push(incoming);
        }
        assert!(entries.contains(&(124032,100)));
        // Restoring the prompt refreshes its position. Later completed turns
        // replace older optional completions, not that just-used checkpoint.
        let index=entries.iter().position(|e|e.0==124032).unwrap();
        let prompt=entries.remove(index);entries.push(prompt);
        let evictions=cache_evictions(&entries,(124170,100),500,true,4).unwrap();
        assert_eq!(evictions,vec![2]);
        assert_eq!(entries[2],(124160,100));
    }
    #[test]
    fn a_short_image_can_use_recent_slots_beside_two_long_parents() {
        let entries=[(124200,100),(124240,100),(124300,100),(124340,100)];
        let evictions=prompt_cache_evictions(&entries,(653,20),500,true,4,true).unwrap();
        assert_eq!(evictions,vec![0]);
        assert!(prompt_cache_evictions(&entries,(653,20),500,true,4,false).is_none());
    }
    #[test]
    fn suspended_reservations_reduce_cache_slots_and_take_priority_without_becoming_victims() {
        // One held request consumes60bytes/one slot of a220byte/four-image
        // pool. Saving another live120byte request evicts only optional cache.
        let entries=vec![(100,30),(50000,100)];
        let evictions=cache_evictions(&entries,(usize::MAX,120),220-60,true,4-1).unwrap();
        assert_eq!(evictions,vec![0,1]);
        assert!(cache_evictions(&entries,(usize::MAX,170),220-60,true,3).is_none());
        assert_eq!(entries,vec![(100,30),(50000,100)],"a refused plan cannot mutate cache or held requests");
        assert!(cache_evictions(&[],(usize::MAX,1),220,true,0).is_none());
        assert_eq!(cache_evictions(&[(1,10),(2,10)],(3,10),220,true,2),Some(vec![0]));
    }
    #[test]
    fn resident_admits_adaptive_sparse_but_rejects_unpaired_experiments() {
        for sparse in [None, Some("0")] {
            for adaptive in [None, Some("0")] {
                assert!(!adaptive_sparse_policy(sparse, adaptive).unwrap());
            }
        }
        assert!(adaptive_sparse_policy(Some("1"), Some("1")).unwrap());
        for pair in [(Some("1"), None), (None, Some("1")),
            (Some("1"), Some("0")), (Some("2"), Some("2"))] {
            assert!(adaptive_sparse_policy(pair.0, pair.1).is_err());
        }
    }
    #[test]
    fn input_reclamation_wire_cannot_change_or_claim_a_request() {
        for headroom in [(8usize<<30)+1,9usize<<30,12usize<<30] {
            let bytes=Command::ReclaimInputs{headroom}.encode().unwrap();
            assert_eq!(Command::decode(&bytes).unwrap().encode().unwrap(),bytes);
            for n in 0..bytes.len() {assert!(Command::decode(&bytes[..n]).is_err());}
            let mut extra=bytes;extra.push(0);assert!(Command::decode(&extra).is_err());
        }
        for headroom in [0,8usize<<30,(12usize<<30)+1,usize::MAX] {
            assert!(Command::ReclaimInputs{headroom}.encode().is_err());
            let mut bytes=vec![17];bytes.extend_from_slice(&(headroom as u64).to_le_bytes());
            assert!(Command::decode(&bytes).is_err());
        }
    }
    #[test]
    fn command_token_bytes_are_lossless_and_bounded() {
        let command = Command::Begin {images: Vec::new(),
            id: 91,
            seed: (1u64 << 63) - 1,
            greedy: true,
            limit: 262144,
            tokens: vec![0, 1, 129279],
        };
        let bytes = command.encode().unwrap();
        let Command::Begin {
                images,
            id,
            seed,
            greedy,
            limit,
            tokens,
        } = Command::decode(&bytes).unwrap()
        else {
            panic!()
        };
        assert!(greedy);
        assert!(images.is_empty());
        assert_eq!(
            (id, seed, limit, tokens),
            (91, (1u64 << 63) - 1, 262144, vec![0, 1, 129279])
        );
        assert!(Command::decode(&bytes[..bytes.len() - 1]).is_err());
        let mut bad = bytes;
        bad[21..25].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Command::decode(&bad).is_err());
        assert!(Command::decode(&[3, 4]).is_err());
    }
    #[test]
    fn image_command_roundtrips_full_descriptors_without_aliasing_text_wire() {
        use crate::{image_cache::Descriptor,vision::Geometry};
        let images=vec![Descriptor{start:1,geometry:Geometry{height:3,width:3},input:[19;32]}];
        let command=Command::Begin{id:4,seed:9,greedy:false,limit:17,
            tokens:vec![2,129264,129264,129264,129264,3],images:images.clone()};
        let wire=command.encode().unwrap();
        assert_eq!(wire[0],12);
        let Command::Begin{images:decoded,..}=Command::decode(&wire).unwrap() else {panic!()};
        assert_eq!(images,decoded);
        for cut in 0..wire.len() {assert!(Command::decode(&wire[..cut]).is_err());}
        let mut extra=wire.clone();extra.push(0);assert!(Command::decode(&extra).is_err());
        let mut text=wire;text[0]=1;assert!(Command::decode(&text).is_err());
    }
}
