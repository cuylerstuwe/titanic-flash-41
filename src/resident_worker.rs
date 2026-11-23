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

#[path = "output_projection_probe.rs"]
mod output_projection_probe;
#[path = "verifier_numerics_probe.rs"]
mod verifier_numerics_probe;
#[path = "woa_mid_native_probe.rs"]
mod woa_mid_native_probe;
#[path = "wide_hint_native_probe.rs"]
mod wide_hint_native_probe;
#[path = "ffn32_refresh_probe.rs"]
mod ffn32_refresh_probe;
#[path = "ffn32_generation_probe.rs"]
mod ffn32_generation_probe;
#[path = "sampler_generation_probe.rs"]
mod sampler_generation_probe;
#[path = "wide_epilogue_probe.rs"]
mod wide_epilogue_probe;
#[path = "moe_late_pdl_probe.rs"]
mod moe_late_pdl_probe;
#[path = "moe_cacheca_probe.rs"]
mod moe_cacheca_probe;
#[path = "moe_epilogue_probe.rs"]
mod moe_epilogue_probe;
#[path = "ffn_sum_append_probe.rs"]
mod ffn_sum_append_probe;
#[path = "ffn_hint_model_probe.rs"]
mod ffn_hint_model_probe;
#[path = "graph_retention_probe.rs"]
mod graph_retention_probe;
#[path = "append_last_gather_probe.rs"]
mod append_last_gather_probe;
#[path="gather_rows_probe.rs"]
mod gather_rows_probe;
#[path = "solo_economics_probe.rs"]
mod solo_economics_probe;
#[cfg(feature = "vision-protocol")]
#[path = "state_copy_probe.rs"]
mod state_copy_probe;
#[path = "causal_two_three_probe.rs"]
mod causal_two_three_probe;
#[path = "expert_prefetch_probe.rs"]
mod expert_prefetch_probe;
#[path = "ffn16_refresh_probe.rs"]
mod ffn16_refresh_probe;
#[path = "ffn16_generation_probe.rs"]
mod ffn16_generation_probe;

#[path = "worker_interleave_probe.rs"]
mod interleave_probe;
#[path = "request_checkpoint.rs"]
mod request_checkpoint;
#[path = "encoder_prompt_cache_probe.rs"]
mod encoder_cache_probe;
#[path = "restore_initial_probe.rs"]
mod restore_initial_probe;
#[path = "mask_storage_probe.rs"]
mod mask_storage_probe;
#[path = "index_score_model_probe.rs"]
mod index_score_model_probe;
#[path = "history_associativity_probe.rs"]
mod history_associativity_probe;
#[path = "history_confidence_probe.rs"]
mod history_confidence_probe;
#[path = "dense_live_rows_probe.rs"]
mod dense_live_rows_probe;
#[cfg(any(test, feature = "storage-fault-injection"))]
#[path = "cache_admission_fault.rs"]
mod cache_admission_fault;
#[cfg(feature = "storage-fault-injection")]
#[path = "cache_retention_probe.rs"]
mod cache_retention_probe;
#[cfg(feature = "storage-fault-injection")]
#[path = "prefill_admission_probe.rs"]
mod prefill_admission_probe;
#[cfg(feature = "storage-fault-injection")]
#[path = "cache_pressure_probe.rs"]
mod cache_pressure_probe;
#[path = "active_encoder_checkpoint_probe.rs"]
mod active_encoder_checkpoint_probe;
#[cfg(feature = "vision-protocol")]
#[path = "cache_slots_probe.rs"]
mod cache_slots_probe;
#[cfg(feature = "vision-protocol")]
#[path = "forward_cache_probe.rs"]
mod forward_cache_probe;
#[cfg(feature = "vision-protocol")]
#[path = "idle_anchor_probe.rs"]
mod idle_anchor_probe;
#[cfg(feature = "serving")]
#[path = "idle_policy_probe.rs"]
mod idle_policy_probe;
#[cfg(feature = "vision-protocol")]
#[path = "encoder_recycle_probe.rs"]
mod encoder_recycle_probe;
#[path = "split_attention_generation_probe.rs"]
mod split_generation_probe;
#[path = "source_prior_probe.rs"]
mod source_prior_probe;
#[path = "sampler_readout_probe.rs"]
mod sampler_readout_probe;
#[path = "fixed_prefix_quality_probe.rs"]
mod fixed_prefix_quality_probe;
#[path = "matched_path_probe.rs"]
mod matched_path_probe;
#[path = "chunk_plan_probe.rs"]
mod chunk_plan_probe;
#[path = "chunk_economics_probe.rs"]
mod chunk_economics_probe;
#[path = "flex_width_probe.rs"]
mod flex_width_probe;
#[path = "inline_checkpoint_probe.rs"]
mod inline_checkpoint_probe;
#[path = "prefill_wait_probe.rs"]
mod prefill_wait_probe;
#[path = "verify_wait_probe.rs"]
mod verify_wait_probe;
#[path = "wide_mhc_probe.rs"]
mod wide_mhc_probe;
#[path = "mhc_pdl_probe.rs"]
mod mhc_pdl_probe;
#[path = "clock_verify_probe.rs"]
mod clock_verify_probe;
#[path = "clock_workload_probe.rs"]
mod clock_workload_probe;
#[path = "packet_workload_probe.rs"]
mod packet_workload_probe;
#[path = "transport_placement_probe.rs"]
mod transport_placement_probe;
#[path = "shared_down_flash_probe.rs"]
mod shared_down_flash_probe;
#[path = "prefill_head_probe.rs"]
mod prefill_head_probe;
#[path = "prefill_bounds_probe.rs"]
mod prefill_bounds_probe;
#[path = "claude_profile_probe.rs"]
mod claude_profile_probe;
#[path = "claude_ab_probe.rs"]
mod claude_ab_probe;
#[path = "claude_decode_profile.rs"]
mod claude_decode_profile;
#[path = "claude_small_probe.rs"]
mod claude_small_probe;
#[path = "woa_prefill_probe.rs"]
mod woa_prefill_probe;
#[path="query_reuse_worker_probe.rs"]
mod query_reuse_worker_probe;
#[path = "engram_projection_probe.rs"]
mod engram_projection_probe;
#[path = "unused_draft_probe.rs"]
mod unused_draft_probe;
#[path = "sampler_law_probe.rs"]
mod sampler_law_probe;
#[path = "step_budget_probe.rs"]
mod step_budget_probe;
#[path = "hierarchical_model_probe.rs"]
mod hierarchical_model_probe;
#[path = "wide_engram_probe.rs"]
mod wide_engram_probe;
#[path = "woa_prefetch_probe.rs"]
mod woa_prefetch_probe;
#[path = "l2_queue_probe.rs"]
mod l2_queue_probe;
#[path = "batch_screen_probe.rs"]
mod batch_screen_probe;
#[path = "block_progress_probe.rs"]
mod block_progress_probe;
#[path = "confidence_economics_probe.rs"]
mod confidence_economics_probe;
#[path = "confidence_policy_probe.rs"]
mod confidence_policy_probe;
#[path = "pooled_route_probe.rs"]
mod pooled_route_probe;
#[path = "history_copy_route_probe.rs"]
mod history_copy_route_probe;
#[path = "history32_probe.rs"]
mod history32_probe;
#[path = "history32_generation_probe.rs"]
mod history32_generation_probe;
#[path = "history32_fresh_probe.rs"]
mod history32_fresh_probe;
#[path = "history_minimum_probe.rs"]
mod history_minimum_probe;
#[path = "compact32_index_probe.rs"]
mod compact32_index_probe;
#[path = "swa_precision_probe.rs"]
mod swa_precision_probe;
#[path = "private_int8_probe.rs"]
mod private_int8_probe;
#[path = "private_nvfp4_probe.rs"]
mod private_nvfp4_probe;
#[path = "worker_background_prefill.rs"]
mod background_prefill;
use background_prefill::PendingPrefill;
#[derive(Clone,Copy,PartialEq,Eq)]
enum PrefillDirective { Continue, Cancel, Yield }
#[cfg(feature = "serving")]
#[path = "worker_interleave_http.rs"]
mod interleave_http;
#[path = "batched_serving.rs"]
mod batched_serving;
#[cfg(feature = "serving")]
#[path = "peer_prefill_probe.rs"]
mod peer_prefill_probe;
#[path = "independent_owner_probe.rs"]
mod independent_owner_probe;
#[path = "resident_bank.rs"]
mod resident_bank;
#[path = "pair_step_profile_probe.rs"]
mod pair_step_profile_probe;
#[path = "attention16_refresh_probe.rs"]
mod attention16_refresh_probe;

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
    if let Some(peer)=event.get_mut("paired_peer") {redact_http_event(peer);}
    // Batched decode: every conversation of a step carries its own outputs.
    if let Some(conversations)=event.get_mut("conversations").and_then(Value::as_array_mut) {
        for conversation in conversations.iter_mut().filter(|c|c.is_object()) {redact_http_event(conversation);}
    }
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
    /// config::copy_hints: the next Begin's file-hint tokens, sent in their own frame immediately before it. They
    /// apply only to a Begin whose prompt has `prompt_len` tokens and ends with `tail`.
    CopyHints { prompt_len: usize, tail: Vec<u32>, tokens: Vec<u32> },
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
                ensure!(*headroom > crate::memory::floor_bytes() && *headroom <= 12usize << 30,
                    "input reclamation headroom bounds");
                out.push(17);out.extend_from_slice(&(*headroom as u64).to_le_bytes());
            }
            Self::CopyHints { prompt_len, tail, tokens } => {
                ensure!(*prompt_len <= crate::config::MODEL_CONTEXT && tail.len() <= 64 && tail.len() <= *prompt_len
                    && tokens.len() <= 1 << 18 && tokens.iter().chain(tail).all(|&id| id < 129280), "copy hint command bounds");
                out.push(25);out.extend_from_slice(&(*prompt_len as u32).to_le_bytes());out.push(tail.len() as u8);
                for id in tail { out.extend_from_slice(&id.to_le_bytes()); }
                out.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
                for id in tokens { out.extend_from_slice(&id.to_le_bytes()); }
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
            [25, ..] if bytes.len() >= 6 => {
                let prompt_len = u32::from_le_bytes(bytes[1..5].try_into().unwrap()) as usize;
                let tail_len = bytes[5] as usize;
                ensure!(tail_len <= 64 && bytes.len() >= 6 + 4 * tail_len + 4, "copy hint command length");
                let words = |b: &[u8]| b.chunks_exact(4).map(|x| u32::from_le_bytes(x.try_into().unwrap())).collect::<Vec<_>>();
                let tail = words(&bytes[6..6 + 4 * tail_len]);
                let at = 6 + 4 * tail_len;
                let n = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
                ensure!(n <= 1 << 18 && bytes.len() == at + 4 + 4 * n && prompt_len <= crate::config::MODEL_CONTEXT
                    && tail_len <= prompt_len, "copy hint command length");
                let tokens = words(&bytes[at + 4..]);
                ensure!(tokens.iter().chain(&tail).all(|&id| id < 129280), "copy hint token domain");
                Ok(Self::CopyHints { prompt_len, tail, tokens })
            }
            [17, ..] if bytes.len() == 9 => {
                let headroom = usize::try_from(u64::from_le_bytes(bytes[1..].try_into().unwrap()))?;
                ensure!(headroom > crate::memory::floor_bytes() && headroom <= 12usize << 30,
                    "input reclamation headroom bounds");
                Ok(Self::ReclaimInputs { headroom })
            }
            _ => anyhow::bail!("invalid resident command"),
        }
    }
}

struct CachedCompleteImage {
    session:crate::dspark_generation::StateImage,
    bank_ancestors:Vec<crate::target::BankEncoderWitness>,
}
impl CachedCompleteImage {
    fn device_bytes(&self)->usize {self.session.device_bytes()}
}
enum CachedState {
    Complete {image:CachedCompleteImage,pending:u32},
    Encoder {image:crate::target::EncoderPrefixImage,history:Option<crate::history_lookup::HistoryLookup>,policy:crate::dspark_generation::ProposalPolicyImage},
}
/// The drafter feature rows of a prompt's 128-row tail (positions [end-128, end)), kept beside its near-end encoder
/// image (CANADA_SCRATCH_ENCODER_SHORT_SUFFIX=2) and matched by end position and the tokens before end-1.
struct DraftTail {
    end: u64,
    tokens: Vec<u32>,
    values: crate::cuda::Buffer<half::bf16>,
}
struct CachedSession {
    owner: u8,
    prefix:CachedPrefix,
    media:crate::prompt_media::Identity,
    image:CachedState,
    charge:usize,
    prompt_logits:Option<Buffer<f32>>,
    // Speculative idle work yields to established conversation checkpoints
    // until an authoritative request actually selects that exact prefix.
    speculative:bool,
    // An image at the end of what several conversations begin with (CANADA_SCRATCH_SHARED_PREFIX_CHECKPOINT).
    // Other conversations restore it, so it is kept as a standalone image when an unrelated prompt takes the bank.
    shared:bool,
    // Reuse census (Nov 18 2026, for a disk tier's sizing; read only by CANADA_SCRATCH_CACHE_TRACE): what saved the
    // entry, when, and when a request last checked it out.
    kind:&'static str,
    born:Instant,
    used:Instant,
    // config::cache_keep_latest: the Begin that saved the entry (Worker::cache_serial) and the Begin that last checked
    // it out (0: never restored); keep_latest_earlier reads both.
    serial:u64,
    restored_serial:u64,
}
impl CachedSession {
    /// [prefix tokens, charge, kind, seconds since saved, seconds since last checked out, disk state] for the cache trace.
    fn trace(&self)->Value {
        let disk=if !self.resident_now(){"parked"}else if self.disk_clean(){"clean"}else{"ram"};
        json!([self.prefix.len(),self.charge,self.kind,self.born.elapsed().as_secs(),self.used.elapsed().as_secs(),disk])
    }
    /// Disk tier (config::disk_tier): every copied image of the entry.
    fn images(&self)->Vec<&crate::state_image::Image> {
        match &self.image {CachedState::Complete{image,..}=>image.session.images(),CachedState::Encoder{image,..}=>image.images()}
    }
    fn images_mut(&mut self)->Vec<&mut crate::state_image::Image> {
        match &mut self.image {CachedState::Complete{image,..}=>image.session.images_mut(),CachedState::Encoder{image,..}=>image.images_mut()}
    }
    /// Self-contained snapshots only: every storage a copied image (no exchange storage) and no bank-retained tail (its
    /// omitted prefix lives in the live bank, so its validity depends on state outside the image).
    fn parkable(&self)->bool {
        match &self.image {
            CachedState::Complete{image,..}=>image.session.images().len()==2,
            CachedState::Encoder{image,..}=>!image.bank_retained()&&image.images().len()==1,
        }
    }
    fn disk_clean(&self)->bool {let v=self.images();!v.is_empty()&&v.iter().all(|i|i.has_disk_copy())}
    fn resident_now(&self)->bool {self.images().iter().all(|i|i.resident())}
    fn disk_bytes(&self)->usize {self.images().iter().map(|i|i.disk_bytes()).sum()}
    fn logical_bytes(&self)->usize {self.images().iter().map(|i|i.bytes()).sum()}
    fn park_device(&mut self)->Result<()> {for image in self.images_mut(){image.drop_device()?;} Ok(())}
    /// Ok(false): no memory for every image now (this rank's partial reload is undone).
    /// Ok(None): no memory for every image now (this rank's partial reload is undone). Otherwise the seconds spent
    /// allocating, reading, checking digests and uploading, summed over the images.
    fn reload_device(&mut self)->Result<Option<[f64;4]>> {
        let mut sum=[0.0f64;4];
        let mut ok=true;
        for image in self.images_mut(){
            match image.reload()? {
                Some((alloc,t))=>{sum[0]+=alloc;sum[1]+=t.read;sum[2]+=t.digest;sum[3]+=t.upload;}
                None=>{ok=false;break;}
            }
        }
        if !ok { for image in self.images_mut(){ if image.resident()&&image.has_disk_copy(){image.drop_device()?;} } return Ok(None); }
        Ok(Some(sum))
    }
    fn drop_disk_copies(&mut self) {for image in self.images_mut(){ if image.resident(){let _=image.drop_disk_copy();} }}
    fn opportunistic_logits(&self)->bool {
        self.speculative || (self.prompt_logits.is_some() && self.media.placements().is_empty())
    }
}

/// Eviction cost at a bank handoff. The replacement must survive the reset
/// and match both the full token prefix and its image identities.
struct CheckpointAlternatives {
    lengths: Vec<usize>,
    coverage: Vec<Vec<usize>>,
    independent: Vec<bool>,
    cheap: Vec<bool>,
}

impl CheckpointAlternatives {
    fn new(entries: &[CachedSession]) -> Self {
        let lengths: Vec<_> = entries.iter().map(|e| e.prefix.len()).collect();
        let independent: Vec<_> = entries.iter().map(|e| !e.speculative && e.prompt_logits.is_none()
            && matches!(&e.image, CachedState::Encoder { image, .. } if !image.bank_retained())).collect();
        let cheap = entries.iter().map(|e| e.speculative || (e.parkable() && e.disk_clean())).collect();
        let coverage = entries.iter().enumerate().map(|(i, candidate)| {
            entries.iter().enumerate().map(|(j, other)| {
                let through = lengths[i].min(lengths[j]);
                if i == j || !independent[j] || !candidate.media.matches_prefix(&other.media, through) { return 0; }
                let covered = match lengths[i].cmp(&lengths[j]) {
                    std::cmp::Ordering::Greater => candidate.prefix.extends(&other.prefix),
                    std::cmp::Ordering::Equal => candidate.prefix[..] == other.prefix[..],
                    std::cmp::Ordering::Less => !candidate.shared && other.prefix.extends(&candidate.prefix),
                };
                if covered { through } else { 0 }
            }).collect()
        }).collect();
        Self { lengths, coverage, independent, cheap }
    }

    fn victim(&self, removed: &[bool], protected: &[usize]) -> Option<(usize, usize)> {
        (0..self.lengths.len()).filter(|i| !removed[*i] && !protected.contains(i)).map(|i| {
            let covered = self.coverage[i].iter().zip(removed)
                .filter_map(|(&n, &gone)| (!gone).then_some(n)).max().unwrap_or(0);
            (i, self.lengths[i].saturating_sub(covered))
        }).min_by_key(|&(i, lost)| (!self.cheap[i], self.independent[i] && lost == self.lengths[i], lost, i))
    }
}

#[cfg(test)]
#[test]
fn bank_handoff_keeps_independent_inputs_even_when_a_generated_answer_is_longer() {
    let alternatives = CheckpointAlternatives {
        lengths: vec![1_010_048, 1_010_416, 1_012_224, 1_012_524],
        // The fourth snapshot requires the bank about to be reset. It cannot cover the third.
        coverage: vec![vec![0; 4], vec![1_010_048,0,0,0], vec![0; 4], vec![0,0,1_012_224,0]],
        independent: vec![true,false,true,false], cheap:vec![false;4],
    };
    assert_eq!(alternatives.victim(&[false;4], &[3]), Some((1,368)));
    assert_eq!(alternatives.victim(&[false,true,false,false], &[3]), Some((0,1_010_048)));
}

/// Cache slots kept for checkpoints under 8,192 tokens (CANADA_SCRATCH_SESSION_CACHE_SHORT_RESERVE=R, at most
/// half the slots; default 0). With 0 the s1507-s1510 guard applies: a short checkpoint never takes a long
/// entry's slot and every short checkpoint gives way first to a long image. That left short conversations and
/// short shared prompts with nothing once sixteen long entries filled the cache, which is the steady state
/// (s1514 shp4: capacity refused, every turn of the short chat prefilled cold again). With R > 0 the two kinds
/// have shares instead: a long image never takes the newest R short entries, and a short image takes a long
/// entry only while short entries hold fewer than R slots; otherwise the oldest short one. Set once at startup
/// (every rank agrees); the planners below read it.
static SHORT_RESERVE:std::sync::atomic::AtomicUsize=std::sync::atomic::AtomicUsize::new(0);
fn short_reserve()->usize {SHORT_RESERVE.load(std::sync::atomic::Ordering::Relaxed)}
/// CANADA_SCRATCH_CACHE_LADDER_CAP=K (Nov 17; 0 = off, default; at most 16): a conversation whose earlier checkpoints
/// already hold K or more cache entries pays for a new checkpoint with one of its own (`ladder_victim`: never its two
/// latest rungs) instead of another conversation's least recently used entry. Otherwise eviction is unchanged (least
/// recently used). Found Nov 17: sixteen image requests of one ~50K conversation pushed an idle 974K conversation out
/// of the cache (re-read 115 s) by filling all sixteen slots with its own checkpoints. Cost-based orders that fixed
/// that broke rotating long conversations or thinking-stripped follow-ups (see eviction_cost_order).
static LADDER_CAP:std::sync::atomic::AtomicUsize=std::sync::atomic::AtomicUsize::new(0);
fn ladder_cap()->usize {LADDER_CAP.load(std::sync::atomic::Ordering::Relaxed)}
fn parse_ladder_cap(value:Option<&str>)->Result<usize> {
    let Some(value)=value else {return Ok(0)};
    let cap:usize=value.parse()?;
    ensure!(cap==0||(2..=16).contains(&cap),"cache ladder cap must be 0 or 2..=16");
    Ok(cap)
}
/// CANADA_SCRATCH_CACHE_TRACE=1 (Nov 17; diagnostic, default off): one `cache trace {json}` line on stderr for every
/// cache eviction decision: the path (a save's budget plan and memory-pressure victims, the restored-image
/// retention pass), the incoming entry, every victim's prefix length and charge, the entries held, and for a save the
/// memory available against the memory required. Read-only: it changes no decision. Found Nov 17: the stress panel's
/// idle 974K conversation is lost although the slot planner protects it, so the losing path must be seen.
static CACHE_TRACE:std::sync::atomic::AtomicBool=std::sync::atomic::AtomicBool::new(false);
fn parse_cache_trace(value:Option<&str>)->Result<bool> {
    match value {None=>Ok(false),Some("0")=>Ok(false),Some("1")=>Ok(true),
        Some(_)=>anyhow::bail!("CANADA_SCRATCH_CACHE_TRACE must be 0 or 1")}
}
fn cache_trace(event:impl FnOnce()->Value) {
    if CACHE_TRACE.load(std::sync::atomic::Ordering::Relaxed) {eprintln!("cache trace {}",event());}
}
fn parse_short_reserve(value:Option<&str>,slots:usize)->Result<usize> {
    let Some(value)=value else {return Ok(0)};
    let reserve:usize=value.parse()?;
    ensure!(reserve<=slots/2,"short checkpoint reserve exceeds half the session cache slots");
    Ok(reserve)
}

/// Sixteen slots keep four long conversations' prompt checkpoints and ladder rungs resident at once. The protection mask
/// is 32 bits wide (Nov 20; was 16): CANADA_SCRATCH_SESSION_CACHE_SLOTS=24 leaves room for the parking extra slots
/// (image_entry_limit) under the same mask. Bank-retained checkpoints cannot park, so with many long conversations the
/// slot count binds before the byte budget (the 4 x 256K multibench sat at 16 entries throughout).
const MAX_SESSION_CACHE_SLOTS:usize=32;
fn cache_protection_mask_valid(mask:u32,entries:usize)->bool {
    entries<=MAX_SESSION_CACHE_SLOTS && u64::from(mask)<(1u64<<entries)
}
fn session_cache_slots(value:Option<&str>)->Result<usize> {
    match value {
        None|Some("4")=>Ok(4),Some("8")=>Ok(8),Some("16")=>Ok(16),Some("24")=>Ok(24),
        _=>anyhow::bail!("session cache slots must be 4, 8, 16 or 24"),
    }
}
fn configured_session_cache_slots()->Result<usize> {
    let value=match std::env::var("CANADA_SCRATCH_SESSION_CACHE_SLOTS") {
        Ok(v)=>Some(v),Err(std::env::VarError::NotPresent)=>None,Err(e)=>return Err(e.into()),
    };
    session_cache_slots(value.as_deref())
}

fn idle_anchor_enabled()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_IDLE_ANCHOR") {
        Ok(v)=>{ensure!(v=="0"||v=="1","idle anchor must be0or1");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(false),Err(e)=>Err(e.into()),
    }
}

fn idle_anchor_rolling()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_IDLE_ANCHOR_ROLLING") {
        Ok(v)=>{ensure!(v=="0"||v=="1","rolling idle anchor must be0or1");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(false),Err(e)=>Err(e.into()),
    }
}

fn encoder_duplicate_skip()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_ENCODER_DUPLICATE_SKIP") {
        Ok(v)=>{ensure!(v=="0"||v=="1","encoder duplicate skip must be0or1");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(true),Err(e)=>Err(e.into()),
    }
}
fn encoder_image_recycle()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_ENCODER_IMAGE_RECYCLE") {
        Ok(v)=>{ensure!(v=="0"||v=="1","encoder image recycle must be0or1");Ok(v=="1")},
        // s1264: reuse passed exact same-load eviction/restoration checks.
        // Respect an explicitly disabled reclamation control by default.
        Err(std::env::VarError::NotPresent)=>cache_pressure_reclaim(),Err(e)=>Err(e.into()),
    }
}
fn cache_pressure_reclaim()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_CACHE_PRESSURE_RECLAIM") {
        Ok(value)=>{ensure!(value=="0"||value=="1","invalid cache pressure policy");Ok(value=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(true),Err(error)=>Err(error.into()),
    }
}
fn displaced_snapshot()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_DISPLACED_SNAPSHOT") {
        Ok(v)=>{ensure!(v=="0"||v=="1","displaced snapshot must be0or1");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(false),Err(e)=>Err(e.into()),
    }
}
/// Whether an existing entry for the same prefix already keeps it for a later
/// request. When another conversation is about to reuse the bank
/// (`survivors_only`), a bank-retained encoder image does not: it is invalid as
/// soon as the bank's pages are rewritten.
fn keeps_prefix(bank_retained:bool,survivors_only:bool)->bool {!survivors_only||!bank_retained}
/// Diagnostic only (CANADA_SCRATCH_CLAUDE_DECODE_SUMMARY=1): per-request decode
/// totals, logged by rank 0 when the request finishes.
#[derive(Default)]
struct DecodeSummary {
    steps: usize,
    verified_rows: usize,
    accepted: usize,
    emitted: usize,
    phase_ms: [f64; 4],
    wall_ms: f64,
    by_rows: std::collections::BTreeMap<usize, (usize, f64)>,
    /// By verified rows: proposals accepted, and steps that accepted every proposal (a full block: a wider one
    /// would have gone on). `merge64`: 32-row steps that directly followed a full 32-row step not already counted,
    /// i.e. the steps a 64-row copy block would absorb; `open32` is that pairing's state.
    full_rows: std::collections::BTreeMap<usize, (usize, usize)>,
    merge64: usize,
    open32: bool,
    /// The first steps of the request: (ms since the previous event left the worker, rows, drafts, tokens out,
    /// step ms, phases).
    first: Vec<(f64, usize, usize, usize, f64, [f64; 4])>,
    /// CANADA_SCRATCH_LAYER_CLOCK=1 in a serving load: sums of layer_clock::parts_us over the steps that had them.
    clock: [f64; 8],
    clock_steps: usize,
    /// CANADA_SCRATCH_STEP_CONFIDENCE_LOG=1: whether this request's committed tokens already hold the end of
    /// thinking, and a tool-call marker (the kind of text a step's rows predict).
    kind_ended: bool,
    kind_call: bool,
}
/// CANADA_SCRATCH_MIDPASS_DUMP=<existing directory> (probe only; needs CANADA_SCRATCH_CLAUDE_DECODE_SUMMARY=1 and the
/// always-six data collection): see the hook after the single-conversation step.
fn midpass_dump_dir() -> Option<PathBuf> {
    static DIR: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| std::env::var("CANADA_SCRATCH_MIDPASS_DUMP").ok().map(PathBuf::from).filter(|p| p.is_dir())).clone()
}
fn decode_summary_enabled() -> bool {
    std::env::var("CANADA_SCRATCH_CLAUDE_DECODE_SUMMARY").as_deref() == Ok("1")
}
fn displaced_checkpoint()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_DISPLACED_CHECKPOINT") {
        Ok(v)=>{ensure!(v=="0"||v=="1","displaced checkpoint must be0or1");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(false),Err(e)=>Err(e.into()),
    }
}
/// CANADA_SCRATCH_DISPLACED_RECLAIM=1 (default off): when the conversation an incoming prompt displaces cannot be
/// copied out of the bank for lack of room (cache budget or memory admission on any rank), older inactive images
/// give way first, by the cache's own pressure order, instead of the displaced conversation being dropped. It is
/// the most recently used conversation; found Nov 13 with four 230-260K-token conversations, whose follow-up turns
/// prefilled from scratch (25-34 s) while nine stale images stayed cached.
fn displaced_reclaim()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_DISPLACED_RECLAIM") {
        Ok(v)=>{ensure!(v=="0"||v=="1","displaced reclaim must be0or1");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(false),Err(e)=>Err(e.into()),
    }
}
/// CANADA_SCRATCH_CACHE_COST_ORDER=1 (default off; Nov 17, the owner's queue item "eviction order"): before a cache save
/// plans its evictions, entries are reordered so the planners (which take the oldest unprotected entry first) lose the
/// cheapest entry first: the one whose loss costs the fewest tokens to prefill again. An entry's cost is the distance
/// from the longest other cached entry it extends (its next-best restore point; its whole length when it extends none).
/// A conversation's dense checkpoints and replaced follow-ups are cheap; another conversation's only snapshot costs its
/// whole length. Entries unused for STALE_AFTER saves lose that protection. Found on Nov 17: in production's own release
/// stress panel, sixteen image requests of one ~50K-token conversation pushed an idle 974K-token conversation's snapshots
/// out (re-prefilled in ~116 s) to keep sixteen checkpoints a few hundred tokens apart.
fn cache_cost_order()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_CACHE_COST_ORDER") {
        Ok(v)=>{ensure!(v=="0"||v=="1","cache cost order must be0or1");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(false),Err(e)=>Err(e.into()),
    }
}
/// Room for the cache entries: (bytes, slots). Entries are charged to the cache budget (`session`) and, with parked
/// request images (`suspended_*`), to the shared inactive budget (`shared`). `extra_*` is a restored parent held
/// outside the vector that goes back into it, so it is charged to both. Nov 17 (P3 stress panel, s1590): its bytes
/// were taken from the shared budget only; with the cache budget below the shared one (6 GiB against 7 GiB) the
/// displaced conversation's save filled the parent's room and the parent's return stopped the worker group.
fn cache_capacity_of(session:usize,shared:usize,suspended_bytes:usize,suspended_count:usize,limit:usize,
    extra_bytes:usize,extra_count:usize)->(usize,usize) {
    (session.min(shared.saturating_sub(suspended_bytes)).saturating_sub(extra_bytes),
        limit.saturating_sub(suspended_count.saturating_add(extra_count)))
}

/// Saves after which an entry no longer protects itself by cost (it is then the cheapest).
const STALE_AFTER:usize=32;
/// The order (indices, the first one evicted first) for prefixes given oldest first, chosen one entry at a time against
/// the entries still left:
/// 0. stale entries (STALE_AFTER positions from the newest) first;
/// 1. then a rung with exactly one direct extension among the others (a conversation's older turn under its newer
///    one): only an edit inside the gap would restore it; a continuation restores the newer entry;
/// 2. then everything else: a conversation's newest entry, and a branch point (two or more direct extensions, e.g. a
///    prompt with two regenerations, which serves the next regeneration).
/// The class comes from all entries (a branch point stays one after a sibling goes). Within a class the cheapest loss
/// goes first: the tokens between the entry and the longest entry left that it extends (its whole length when none);
/// oldest first among equal costs. Choosing again after each pick lets an entry whose lower rungs are gone count its
/// whole length.
/// Nov 17 (C13 bar, residency 4 x 980K): the first version ranked every entry by that cost alone, so a conversation's
/// newest turn (a few dozen tokens past its previous one) went first and the next turn restored an older rung
/// (~100 tokens more to prefill, +0.1 s at ~1M tokens).
fn eviction_cost_order(prefixes:&[&[u32]])->Vec<usize> {
    let n=prefixes.len();
    // ext[a][b]: entry a strictly extends entry b (b is a proper prefix of a).
    let ext=(0..n).map(|a|(0..n).map(|b|a!=b&&prefixes[b].len()<prefixes[a].len()&&prefixes[a].starts_with(prefixes[b]))
        .collect::<Vec<_>>()).collect::<Vec<_>>();
    // direct extensions of i: entries that extend i with no entry between them
    let rung=(0..n).map(|i|(0..n).filter(|&j|ext[j][i]&&!(0..n).any(|k|ext[j][k]&&ext[k][i])).count()==1).collect::<Vec<_>>();
    let mut left=(0..n).collect::<Vec<_>>();
    let mut order=Vec::with_capacity(n);
    while !left.is_empty() {
        let key=|i:usize|->(u8,usize,usize) {
            if n-1-i>=STALE_AFTER {return (0,0,i);}
            let covered=left.iter().filter(|&&j|ext[i][j]).map(|&j|prefixes[j].len()).max().unwrap_or(0);
            (if rung[i] {1} else {2},prefixes[i].len()-covered,i)
        };
        let at=(0..left.len()).min_by_key(|&p|key(left[p])).expect("entries left");
        order.push(left.remove(at));
    }
    order
}

fn checkpoint_ladder()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_CHECKPOINT_LADDER") {
        Ok(v)=>{ensure!(v=="0"||v=="1","checkpoint ladder must be0or1");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(false),Err(e)=>Err(e.into()),
    }
}

/// Keep room for the current conversation's checkpoint and completed turn.
/// Protect up to two longer parents; use oldest-first replacement in the
/// remaining slots. A pure longest-prefix policy lets slightly longer stale
/// completions permanently exclude every new prompt checkpoint.
/// Entries are ordered by insertion/restoration, so cache hits refresh age.
fn cache_evictions(entries:&[(usize,usize)], incoming:(usize,usize), budget:usize,
    long_first:bool,slots:usize)->Option<Vec<usize>> {
    cache_evictions_ladder(entries,None,incoming,budget,long_first,slots)
}

/// Earlier checkpoints of the incoming conversation (`ancestors`, entries whose
/// prefix the incoming prefix extends) form a reuse ladder: an edit of early
/// history restores the latest retained checkpoint at or before the edit.
/// Whenever the ordinary policy would evict one of them, the ladder evicts
/// instead the rung whose loss adds the least re-prefill for an edit at a
/// uniformly random position, i.e. the smallest (at-prev)*(next-at). The two
/// latest rungs (append, regenerate) are never ladder victims. Which entry
/// class is evicted (an ancestor or another conversation) and how many are
/// evicted are unchanged; `None` is exactly the ordinary policy.
fn cache_evictions_ladder(entries:&[(usize,usize)], ancestors:Option<&[bool]>, incoming:(usize,usize),
    budget:usize,long_first:bool,slots:usize)->Option<Vec<usize>> {
    cache_evictions_ladder_reserved(entries,ancestors,incoming,budget,long_first,slots,short_reserve(),ladder_cap())
}
/// As cache_evictions_ladder with `reserve` slots kept for checkpoints under 8,192 tokens (0: no shares). The
/// entry the ordinary policy starts from (the oldest unprotected one) is then chosen by kind: an image of 8,192
/// tokens or more skips the newest `reserve` short entries; a shorter image takes the oldest short entry once
/// short entries hold `reserve` slots or more, and otherwise the oldest entry of either kind.
#[allow(clippy::too_many_arguments)]
fn cache_evictions_ladder_reserved(entries:&[(usize,usize)], ancestors:Option<&[bool]>, incoming:(usize,usize),
    budget:usize,long_first:bool,slots:usize,reserve:usize,cap:usize)->Option<Vec<usize>> {
    if incoming.1>budget || slots==0 || ancestors.is_some_and(|a|a.len()!=entries.len()) {return None;}
    let mut protected=if long_first {
        (0..entries.len()).filter(|&i|entries[i].0>incoming.0).collect::<Vec<_>>()
    }else{Vec::new()};
    protected.sort_by_key(|&i|std::cmp::Reverse((entries[i].0,i)));
    protected.truncate(2.min(slots.saturating_sub(2)));
    let mut retained=(0..entries.len()).collect::<Vec<_>>();let mut removed=Vec::new();
    while retained.len()>=slots || retained.iter().map(|&i|entries[i].1).sum::<usize>()+incoming.1>budget {
        let oldest=if reserve==0 {*retained.iter().find(|&&i|!protected.contains(&i))?}else{
            let shorts=retained.iter().copied().filter(|&i|entries[i].0<8192).collect::<Vec<_>>();
            if incoming.0>=8192 {
                let kept=&shorts[shorts.len().saturating_sub(reserve)..];
                *retained.iter().find(|&&i|!protected.contains(&i)&&!kept.contains(&i))?
            }else if shorts.len()>=reserve {*shorts.iter().find(|&&i|!protected.contains(&i))?}
            else{*retained.iter().find(|&&i|!protected.contains(&i))?}
        };
        let victim=match ancestors {
            // See ladder_cap: a conversation holding K or more of its own rungs pays with one of them.
            Some(a) if cap>0&&retained.iter().filter(|&&i|a[i]).count()>=cap=>
                ladder_victim(entries,a,&retained,&protected).unwrap_or(oldest),
            Some(a) if a[oldest]=>ladder_victim(entries,a,&retained,&protected).unwrap_or(oldest),
            _=>oldest,
        };
        retained.retain(|&i|i!=victim);removed.push(victim);
    }
    Some(removed)
}

/// The retained ancestor whose removal merges the two smallest-product gaps.
/// Positions are prefix lengths; the latest two rungs are kept.
fn ladder_victim(entries:&[(usize,usize)],ancestors:&[bool],retained:&[usize],protected:&[usize])->Option<usize> {
    let mut rungs=retained.iter().copied().filter(|&i|ancestors[i]).collect::<Vec<_>>();
    rungs.sort_by_key(|&i|(entries[i].0,i));
    if rungs.len()<3 {return None;}
    (0..rungs.len()-2).filter(|&k|!protected.contains(&rungs[k])).min_by_key(|&k|{
        let at=entries[rungs[k]].0;
        let prev=if k==0{0}else{entries[rungs[k-1]].0};
        let next=entries[rungs[k+1]].0;
        ((at-prev) as u128*(next-at) as u128,rungs[k])
    }).map(|k|rungs[k])
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
    cache_evictions_with_optional_ladder(entries,optional,None,incoming,budget,long_first,slots)
}
fn cache_evictions_with_optional_ladder(entries:&[(usize,usize)], optional:&[bool], ancestors:Option<&[bool]>,
    incoming:(usize,usize),budget:usize,long_first:bool,slots:usize)->Option<Vec<usize>> {
    if incoming.1>budget || slots==0 || optional.len()!=entries.len()
        || ancestors.is_some_and(|a|a.len()!=entries.len()) {return None;}
    let mut retained=(0..entries.len()).collect::<Vec<_>>();let mut removed=Vec::new();
    while retained.len()>=slots || retained.iter().map(|&i|entries[i].1).sum::<usize>()+incoming.1>budget {
        let Some(victim)=retained.iter().copied().filter(|&i|optional[i])
            .min_by_key(|&i|(entries[i].0,i)) else {break;};
        retained.retain(|&i|i!=victim);removed.push(victim);
    }
    let rest=retained.iter().map(|&i|entries[i]).collect::<Vec<_>>();
    let rest_ancestors=ancestors.map(|a|retained.iter().map(|&i|a[i]).collect::<Vec<_>>());
    removed.extend(cache_evictions_ladder(&rest,rest_ancestors.as_deref(),incoming,budget,long_first,slots)?
        .into_iter().map(|i|retained[i]));
    Some(removed)
}

/// A completed continuation can be reconstructed from an already saved parent.
/// Do not replace that parent with another child merely because it is slightly
/// longer: sibling completions cannot serve a regeneration or another branch.
/// This reservation applies only to optional completed-turn saves; advancing
/// prompt checkpoints and mandatory suspended requests keep their own policy.
fn completed_cache_evictions(entries:&[(usize,usize)], optional:&[bool], ancestors:&[bool],
    incoming:(usize,usize),budget:usize,long_first:bool,slots:usize)->Option<Vec<usize>> {
    if entries.len()!=optional.len()||entries.len()!=ancestors.len(){return None;}
    let mut reserved=0usize;let mut count=0;
    let mut retained=Vec::new();
    for (i,&parent) in ancestors.iter().enumerate() {
        if parent{reserved=reserved.checked_add(entries[i].1)?;count+=1;}
        else{retained.push(i);}
    }
    let rest=retained.iter().map(|&i|entries[i]).collect::<Vec<_>>();
    let optional=retained.iter().map(|&i|optional[i]).collect::<Vec<_>>();
    Some(cache_evictions_with_optional(&rest,&optional,incoming,budget.checked_sub(reserved)?,
        long_first,slots.checked_sub(count)?)?.into_iter().map(|i|retained[i]).collect())
}

/// Where a request saves its checkpoint when cached conversations share more of its prompt than the image it
/// starts from (CANADA_SCRATCH_SHARED_PREFIX_CHECKPOINT). `common` is the longest prefix of the prompt that a
/// cached conversation also begins with, `previous` the extent of the restored image (0 for a cold prompt) and
/// `otherwise` the position the ordinary policy picks (aligned, or near the prompt end). An image at the end of
/// the shared part (rounded down to 128) is what the next conversation with that shared part restores: it saves
/// each of them `gain` rows. This conversation's next turn re-prefills the `cost` rows between it and
/// `otherwise` once. Taken when the gain is 1,024 rows or more and at least four times the cost, the cost is at
/// most 4,096 rows, and the position is a legal checkpoint (at or above the minimum, 128 rows or more before the
/// prompt end). An ordinary position at or before the shared end already serves later conversations.
fn shared_prefix_checkpoint(common:usize,previous:usize,otherwise:usize,prompt:usize,minimum:usize)->Option<usize> {
    let at=common/128*128;
    if at<=previous||at>=otherwise||at<minimum||prompt<at+128 {return None;}
    let (gain,cost)=(at-previous,otherwise-at);
    (gain>=1024&&cost<=4096&&gain>=4*cost).then_some(at)
}
/// The same for a cold prompt with nothing shared in the cache (level 2 of the flag): an image at the end of the
/// prompt's system part (`system`, the first user turn's start; rounded down to 128), which is what a later
/// conversation under the same system prompt begins with. Nothing says yet that one will come, so the rule is
/// stricter: the gain is 1,024 rows or more and at least eight times the cost, and the cost is at most 1,024 rows
/// (one next turn of this conversation re-prefills its task: 0.15 s at most).
fn system_prefix_checkpoint(system:usize,otherwise:usize,prompt:usize,minimum:usize)->Option<usize> {
    let at=system/128*128;
    if at>=otherwise||at<minimum||prompt<at+128 {return None;}
    let (gain,cost)=(at,otherwise-at);
    (gain>=1024&&cost<=1024&&gain>=8*cost).then_some(at)
}
/// Whether `near` is a near-end checkpoint of the prompt `prompt` (CANADA_SCRATCH_REDUNDANT_PROMPT_IMAGE): a proper,
/// non-empty prefix of it fewer than 128 tokens shorter, which a repeated prompt restores with one short suffix.
fn near_end_of(prompt:&[u32],near:&[u32])->bool {
    !near.is_empty()&&near.len()<prompt.len()&&prompt.len()-near.len()<128&&prompt.starts_with(near)
}
/// Length of the common prefix of two token sequences. Whole blocks first: prompts reach a million tokens and a
/// cache holds sixteen of them.
fn common_prefix(a:&[u32],b:&[u32])->usize {
    let n=a.len().min(b.len());let mut i=0;
    while i+4096<=n&&a[i..i+4096]==b[i..i+4096] {i+=4096;}
    i+a[i..n].iter().zip(&b[i..n]).take_while(|(x,y)|x==y).count()
}
/// A checkpoint below 8,192 tokens is optional (see CANADA_SCRATCH_PROMPT_CHECKPOINT_MIN): an eviction plan that
/// names a checkpoint of 8,192 tokens or more is refused, and those entries are returned as protected from the
/// pressure reclamation as well. Longer checkpoints keep the ordinary policy unchanged.
/// When the ordinary plan names a long entry, room is made among the short entries instead, oldest first (a short
/// conversation recycles its own earlier checkpoints and other short conversations'); if they do not suffice the
/// save is refused. `incoming` is the new image's charge; `budget` and `slots` are those the ordinary plan used.
fn short_checkpoint_plan(entries:&[(usize,usize)],checkpoint:usize,incoming:usize,budget:Option<usize>,
    slots:Option<usize>,plan:Option<Vec<usize>>)->(Option<Vec<usize>>,Vec<usize>) {
    let reserve=short_reserve();
    if reserve>0 {
        // Shares instead of the guard: the plan already follows them (cache_evictions_ladder_reserved). The
        // second value protects the other kind from memory-pressure reclamation beyond the plan: long entries
        // for a short image, the newest `reserve` short entries for a long one.
        let planned=plan.clone().unwrap_or_default();
        let protected=if checkpoint<8192 {
            (0..entries.len()).filter(|&i|entries[i].0>=8192&&!planned.contains(&i)).collect()
        }else{
            let shorts=(0..entries.len()).filter(|&i|entries[i].0<8192).collect::<Vec<_>>();
            shorts[shorts.len().saturating_sub(reserve)..].iter().copied().filter(|i|!planned.contains(i)).collect()
        };
        return (plan,protected);
    }
    if checkpoint>=8192 {return (plan,Vec::new());}
    let long=(0..entries.len()).filter(|&i|entries[i].0>=8192).collect::<Vec<_>>();
    let plan=match (plan,budget,slots) {
        (Some(p),_,_) if p.iter().all(|i|!long.contains(i))=>Some(p),
        (Some(_),Some(budget),Some(slots)) if slots>0&&incoming<=budget=>{
            let mut retained=(0..entries.len()).collect::<Vec<_>>();let mut removed=Vec::new();
            loop {
                if retained.len()<slots&&retained.iter().map(|&i|entries[i].1).sum::<usize>()+incoming<=budget {
                    break Some(removed);
                }
                let Some(victim)=retained.iter().copied().find(|i|!long.contains(i)) else {break None;};
                retained.retain(|&i|i!=victim);removed.push(victim);
            }
        },
        _=>None,
    };
    (plan,long)
}

/// The restored parent is a live reservation, not an eviction candidate.
/// A successor must coexist with it even during allocation. Otherwise a late
/// refusal or cancellation could destroy the only reusable shared prefix.
fn encoder_checkpoint_evictions(entries:&[(usize,usize)], optional:&[bool],
    incoming:(usize,usize),parent_charge:Option<usize>,budget:usize,
    long_first:bool,slots:usize)->Option<Vec<usize>> {
    encoder_checkpoint_evictions_ladder(entries,optional,None,incoming,parent_charge,budget,long_first,slots)
}
#[allow(clippy::too_many_arguments)]
fn encoder_checkpoint_evictions_ladder(entries:&[(usize,usize)], optional:&[bool], ancestors:Option<&[bool]>,
    incoming:(usize,usize),parent_charge:Option<usize>,budget:usize,
    long_first:bool,slots:usize)->Option<Vec<usize>> {
    cache_evictions_with_optional_ladder(entries,optional,ancestors,incoming,
        budget.checked_sub(parent_charge.unwrap_or(0))?,long_first,
        slots.checked_sub(usize::from(parent_charge.is_some()))?)
}

/// The largest shared inactive image budget for an owner layout. Completed caches, parked requests and a
/// restore in flight share it, and every image allocation is still admitted against measured memory above the
/// floor. One physical owner saves several GiB of permanently allocated state, and with the V4.1 cache formats a
/// ~1M-token image is ~0.94 GB per rank, so one bank allows 8 GiB: four 1M conversations resident with room to
/// park an overlapping request. Two physical banks keep 2 GiB.
/// With the snapshot chunk pool (CANADA_SCRATCH_IMAGE_POOL_MIB) one bank's limit is the pool's size when that is
/// larger: pooled snapshots take no runtime allocation, and the pool holds their device bytes (a charge is never
/// below its device bytes, so a budget within the pool keeps every cached and parked snapshot inside it).
fn shared_image_limit(single_bank:bool,pool_mib:Option<usize>)->usize {
    if single_bank {cache_budget_ceiling(pool_mib)} else {2*1024usize.pow(3)}
}

/// The ceiling of the cache budget (CANADA_SCRATCH_SESSION_CACHE_BYTES): 8 GiB, or the snapshot pool's size when it
/// is larger (see shared_image_limit). Every save is still admitted against measured memory above the floor.
fn cache_budget_ceiling(pool_mib:Option<usize>)->usize {
    (8*1024usize.pow(3)).max(pool_mib.unwrap_or(0)<<20)
}

/// CANADA_SCRATCH_IMAGE_POOL_MIB: the snapshot chunk pool's size in MiB (256..65536; unset: no pool).
fn image_pool_mib()->Result<Option<usize>> {
    match std::env::var("CANADA_SCRATCH_IMAGE_POOL_MIB") {
        Ok(v)=>{let mib:usize=v.parse().context("CANADA_SCRATCH_IMAGE_POOL_MIB must be a whole number of MiB")?;
            ensure!((256..=65536).contains(&mib),"image pool size outside 256..65536 MiB");Ok(Some(mib))},
        Err(std::env::VarError::NotPresent)=>Ok(None),
        Err(error)=>Err(error.into()),
    }
}

fn validate_shared_image_budget(budget:usize,single_bank:bool,exchange:bool,pool_mib:Option<usize>)->Result<()> {
    ensure!((64*1024usize.pow(2)..=shared_image_limit(single_bank,pool_mib)).contains(&budget),
        "shared inactive image budget exceeds the selected owner layout");
    // State exchange lends the budget's unallocated growth to waiting inputs, a reservation sized for at
    // most 2 GiB (HttpQueue::reserve_waiting_image_capacity). Refuse at startup rather than at readiness.
    ensure!(!exchange || budget<=2*1024usize.pow(3),
        "state exchange supports at most a 2 GiB shared inactive image budget");
    Ok(())
}

/// Disk tier: the copy in progress (one slice an idle command): the entry by birth and prefix length, the image being
/// written, the write class, when the entry's copy began, and the open write.
struct DiskPending {
    born: Instant,
    tokens: usize,
    image: usize,
    class: u8,
    began: Instant,
    write: Option<crate::disk_tier::PendingWrite>,
}

impl Worker {
    /// config::cache_keep_latest: which cache entries are earlier checkpoints (keep_latest_earlier_flags over this list).
    fn keep_latest_earlier(&self) -> Vec<bool> {
        let (facts, ext) = keep_latest_inputs(&self.cached_sessions);
        keep_latest_earlier_flags(&facts, &ext)
    }
    /// config::cache_keep_latest: put the cache list in keep-latest order (keep_latest_order) before a request's saves and
    /// evictions, so every planner's "oldest eligible entry" is an earlier checkpoint while one is left and a
    /// conversation's latest image only after that. Called where no index into the list is held (command start).
    fn order_cache_keep_latest(&mut self) {
        if !crate::config::cache_keep_latest() || self.cached_sessions.len() < 2 { return; }
        let order = keep_latest_order(&self.keep_latest_earlier());
        if order.iter().enumerate().all(|(k, &i)| k == i) { return; }
        let mut slots: Vec<Option<CachedSession>> = self.cached_sessions.drain(..).map(Some).collect();
        self.cached_sessions = order.into_iter().map(|i| slots[i].take().expect("each entry once")).collect();
    }
    /// config::cache_superseded_first: the eviction plan `victims` (indices into the cache list) of an image whose prefix
    /// is `prefix`, moved onto the cheapest retained images (prefer_cheaper). The earlier-checkpoint rule is
    /// read now, so the restore this request made counts. Never taken as a replacement: `protected` entries, shared
    /// prefixes, and the incoming image's own ancestors (the planners' ladder and reservations decide about those).
    /// `keep`: victims the planner chose because they are optional stay as they are. Every rank holds the same list and
    /// serials, so every rank returns the same plan.
    fn superseded_first(&self, victims: Vec<usize>, prefix: &[u32], protected: &[usize], keep: Option<&[bool]>) -> Vec<usize> {
        if !crate::config::cache_superseded_first() || victims.is_empty() { return victims; }
        let class = cache_classes(&self.cached_sessions, &self.keep_latest_earlier());
        let none = vec![false; self.cached_sessions.len()];
        let keep = keep.filter(|k| k.len() == none.len()).unwrap_or(&none);
        if victims.iter().all(|&v| v >= class.len() || class[v] == 0 || keep[v]) { return victims; }
        let entries: Vec<(usize, usize)> = self.cached_sessions.iter().map(|c| (c.prefix.len(), c.charge)).collect();
        let excluded: Vec<bool> = self.cached_sessions.iter().enumerate().map(|(i, c)| protected.contains(&i)
            || c.shared || c.kind == "shared_prefix"
            || (c.prefix.len() <= prefix.len() && prefix.starts_with(&c.prefix))).collect();
        let planned = victims.clone();
        let chosen = prefer_cheaper(&entries, &class, &excluded, keep, victims);
        if chosen != planned {
            cache_trace(|| json!({"path":"superseded_first","incoming_tokens":prefix.len(),
                "planned":planned.iter().map(|&i| self.cached_sessions[i].trace()).collect::<Vec<_>>(),
                "chosen":chosen.iter().map(|&i| self.cached_sessions[i].trace()).collect::<Vec<_>>()}));
        }
        chosen
    }
    /// Disk tier (config::disk_tier): evict cache entry `i`, parking it (RAM freed, kept on disk) when it has a disk copy,
    /// else dropping it as before. Every rank holds the same entries and copies, so every rank does the same.
    pub(crate) fn evict_cached(&mut self, i: usize) {
        let mut entry = self.cached_sessions.remove(i);
        if self.disk_tier.is_some() && entry.parkable() && entry.disk_clean() && entry.resident_now()
            && entry.park_device().is_ok() {
            self.push_parked(entry);
        }
    }
    /// Disk tier, save on evict (DiskTierConfig::save_on_evict_tokens; 0 = off): before cache pressure evicts entry `i`,
    /// write its missing disk copies now when it holds at least that many tokens and the tier's budgets allow, so the
    /// eviction parks it (RAM freed, kept on disk) instead of dropping it. The idle-time copier only reaches entries in
    /// quiet gaps; under a burst a long conversation's only snapshot could fall out first and cost a cold re-prefill
    /// later (Nov 18, fqm5: a fourth 254K conversation re-prefilled for 34 s after the regions' maps evicted it). A
    /// 256K-token snapshot is ~0.23 GiB a box, a fraction of a second to write. Every rank writes its own shard; the
    /// choice uses only state every rank shares and every outcome goes through an all-rank vote, so all ranks keep the
    /// same entries. A failed copy leaves the entry as it was (the eviction drops it).
    pub(crate) fn save_before_evict(&mut self, i: usize, last_forward: bool) -> Result<Option<Value>> {
        let Some(tier) = self.disk_tier.clone() else { return Ok(None) };
        if (!last_forward && tier.save_on_evict_tokens == 0) || i >= self.cached_sessions.len() { return Ok(None); }
        let began = Instant::now();
        {
            let c = &self.cached_sessions[i];
            // Nov 19 (r2b): optional entries (prompt-end images with exact logits) are not worth a write in the request
            // path: ten such saves cost 0.09-0.11 s each to keep images that only serve an exact retry.
            if (!last_forward && c.prefix.len() < tier.save_on_evict_tokens) || !c.parkable() || !c.resident_now() || c.disk_clean()
                || c.speculative || c.opportunistic_logits() { return Ok(None); }
        }
        // A copy the idle-time writer has in progress for this entry is abandoned: the loop below writes every missing
        // image whole (the same on every rank: the pending copy is replicated state).
        if self.disk_pending.as_ref().is_some_and(|p| {
            let c = &self.cached_sessions[i]; p.born == c.born && p.tokens == c.prefix.len() }) { self.disk_pending = None; }
        let bytes = self.cached_sessions[i].images().iter().filter(|im| !im.has_disk_copy()).map(|im| im.bytes()).sum::<usize>() as u64;
        while self.disk_tier_bytes() as u64 + bytes > tier.cap_bytes && !self.parked_sessions.is_empty() {
            self.parked_sessions.remove(0);
        }
        let days = (self.disk_tier_started.elapsed().as_secs_f64() / 86400.0).max(1.0);
        let budget = (crate::disk_tier::written_bytes() + bytes) as f64 <= tier.write_bytes_per_day as f64 * days;
        let space = crate::disk_tier::free_bytes(&tier.dir).is_ok_and(|f| f >= tier.free_floor_bytes + bytes);
        let cap = self.disk_tier_bytes() as u64 + bytes <= tier.cap_bytes;
        if !self.control.all_admit(budget && space && cap)? {
            return Ok(Some(json!({"event":"save_on_evict_refused","tokens":self.cached_sessions[i].prefix.len(),
                "budget":budget,"space":space,"cap":cap})));
        }
        self.stream.synchronize()?;
        let mut failed = None;
        let n = self.cached_sessions[i].images().len();
        for k in 0..n {
            if self.cached_sessions[i].images()[k].has_disk_copy() { continue; }
            let path = crate::disk_tier::path_for(&tier.dir, self.cached_sessions[i].kind);
            let result = (|| -> Result<()> {
                let mut w = self.cached_sessions[i].images()[k].disk_write_begin(&path)?;
                while !w.done() { self.cached_sessions[i].images()[k].disk_write_step(&mut w)?; }
                self.cached_sessions[i].images_mut()[k].disk_write_finish(w)
            })();
            if let Err(error) = result { failed = Some(error); break; }
        }
        if let Some(error) = &failed { eprintln!("disk tier save on evict failed: {error:#}"); }
        let tokens = self.cached_sessions[i].prefix.len();
        if !self.control.all_admit(failed.is_none())? {
            self.cached_sessions[i].drop_disk_copies();
            return Ok(Some(json!({"event":"save_on_evict_failed","tokens":tokens})));
        }
        let event = json!({"event":"save_on_evict","kind":self.cached_sessions[i].kind,"tokens":tokens,"bytes":bytes,
            "seconds":began.elapsed().as_secs_f64(),"disk_bytes":self.disk_tier_bytes(),
            "written_total":crate::disk_tier::written_bytes()});
        eprintln!("disk tier {event}");
        Ok(Some(event))
    }
    /// Obtain any indispensable durable copies before a planned eviction changes RAM.
    fn preserve_forward_victims(&mut self, plan: &[(usize, usize)], alternatives: &CheckpointAlternatives) -> Result<bool> {
        for &(i, lost) in plan {
            if !alternatives.independent[i] || alternatives.cheap[i] || lost == 0 || lost != alternatives.lengths[i] { continue; }
            let Some(tier) = self.disk_tier.as_ref() else { continue; };
            let entry = &self.cached_sessions[i];
            let worth_saving = entry.parkable() && crate::disk_tier::reload_pays(
                entry.logical_bytes().saturating_mul(2), lost, false, tier.read_bytes_per_s,
                2. * tier.read_fixed_s, crate::config::encoder_chunk_costs());
            if !self.control.all_admit(worth_saving)? { continue; }
            self.save_before_evict(i, true)?;
            if !self.control.all_admit(self.cached_sessions[i].disk_clean())? { return Ok(false); }
        }
        Ok(true)
    }

    /// evict_cached after save_before_evict: what cache pressure (not policy parking) uses.
    pub(crate) fn evict_cached_saving(&mut self, i: usize) -> Result<()> {
        self.save_before_evict(i, false)?;
        self.evict_cached(i);
        Ok(())
    }
    /// config::wide16_elastic (Nov 19 2026): before a prefill of `uncached` new tokens (a prompt of `total`), lend the
    /// 16,384-row profile its extension from the image pool, so the planner can offer 16,384-row chunks. Free chunks
    /// first, keeping room for this request's own prompt-end and near-end images; then the policy's snapshots give way
    /// (wide16_victim). Every rank decides alike: the cache list is agreed and every room check is an all-rank vote.
    /// Returns a report, Null when not applicable; without the room nothing changes and the prefill uses 8,192 rows.
    fn borrow_wide16(&mut self, uncached: usize, total: usize) -> Result<Value> {
        let policy = crate::config::wide16_elastic()?;
        if policy == 0 || !self.cuda.wide16_configured() || uncached < 16384 { return Ok(Value::Null); }
        if self.cuda.wide16_mapped() { return Ok(json!({"lent":true,"already_mapped":true})); }
        let began = Instant::now();
        let chunk = self.cuda.image_pool_chunk_bytes()?;
        let need = self.cuda.wide16_missing_chunks()?;
        let reserve = (2 * self.large.encoder_checkpoint_image_bytes_by_layout(total.min(self.large.context_capacity()))?).div_ceil(chunk);
        let mut proof = b"wide16-borrow-v1".to_vec();
        for v in [need, reserve, uncached, total, self.cached_sessions.len()] { proof.extend_from_slice(&(v as u64).to_le_bytes()); }
        for c in &self.cached_sessions {
            proof.push(c.owner); proof.extend_from_slice(&(c.prefix.len() as u64).to_le_bytes());
            proof.extend_from_slice(&[u8::from(c.disk_clean()), u8::from(c.resident_now()), u8::from(c.parkable())]);
        }
        self.control.agree(&proof)?;
        let free_before = self.cuda.image_pool_free_chunks().unwrap_or(0);
        let mut gave_way = Vec::new();
        let plan = self.wide16_eviction_plan(policy, total, need + reserve)?;
        let lent = plan.is_some();
        if let Some(indices) = plan {
            for i in indices {
                let c = &self.cached_sessions[i];
                gave_way.push(json!({"tokens":c.prefix.len(),"parked":self.disk_tier.is_some()&&c.parkable()&&c.disk_clean(),
                    "optional":self.optional_entries(total)[i],"bytes":c.logical_bytes()}));
                self.evict_cached(i);
            }
        }
        let mapped = lent && self.cuda.wide16_map()?;
        let agreed = lent && self.control.all_admit(mapped)?;
        if mapped && !agreed { self.cuda.wide16_unmap()?; }
        self.wide16_held = agreed;
        let report = json!({"lent":agreed,"policy":policy,"uncached":uncached,"need_chunks":need,"reserve_chunks":reserve,
            "free_chunks_before":free_before,"free_chunks_after":self.cuda.image_pool_free_chunks().unwrap_or(0),
            "gave_way":gave_way,"seconds":began.elapsed().as_secs_f64()});
        eprintln!("wide16 borrow {report}");
        Ok(report)
    }
    /// Stage potential evictions without freeing GPU storage. Each selection sees the same
    /// remaining entries it would see after a real eviction. Put every entry back in its
    /// original position before returning, including when a rank declines the loan.
    fn wide16_eviction_plan(&mut self, policy: u8, total: usize, required: usize) -> Result<Option<Vec<usize>>> {
        let mut available = self.cuda.image_pool_free_chunks().unwrap_or(0);
        let mut held = Vec::new();
        let fits = loop {
            match self.control.all_admit(available >= required) {
                Ok(true) => break Ok(true),
                Err(error) => break Err(error),
                Ok(false) => {}
            }
            let Some(index) = self.wide16_victim(policy, total) else { break Ok(false) };
            let entry = self.cached_sessions.remove(index);
            available += entry.images().iter().map(|image| image.owned_pool_chunks()).sum::<usize>();
            held.push((index, entry));
        };
        let indices = held.iter().map(|(index, _)| *index).collect();
        for (index, entry) in held.into_iter().rev() { self.cached_sessions.insert(index, entry); }
        Ok(fits?.then_some(indices))
    }

    /// The next snapshot that gives way to the 16K extension (identical on every rank: it reads only the shared cache
    /// list), or None. Policy 1 (approved Nov 18): optional and superseded snapshots only, those with a disk copy first
    /// (they park, nothing is lost), oldest first; policy 2 then parks any resident snapshot that has a disk copy.
    fn wide16_victim(&self, policy: u8, incoming: usize) -> Option<usize> {
        let optional = self.optional_entries(incoming);
        let entries = &self.cached_sessions;
        // Superseded = an earlier checkpoint by the keep-latest rule (keep_latest_earlier: a request has gone past it); a
        // conversation's latest images and the one it last restored stay, since that is what its next turn resumes from.
        let earlier = self.keep_latest_earlier();
        let superseded = |i: usize| earlier[i];
        let parks = |i: usize| self.disk_tier.is_some() && entries[i].parkable() && entries[i].disk_clean();
        let resident: Vec<usize> = (0..entries.len()).filter(|&i| entries[i].resident_now()).collect();
        let allowed = |i: usize| optional[i] || superseded(i);
        resident.iter().copied().find(|&i| allowed(i) && parks(i))
            .or_else(|| resident.iter().copied().find(|&i| allowed(i)))
            .or_else(|| (policy >= 2).then(|| resident.iter().copied().find(|&i| parks(i))).flatten())
    }
    /// Return the 16K extension to the pool once no prefill holds it (no paused prefill of this worker). Every rank
    /// runs the same commands, so every rank returns it at the same point.
    fn release_wide16_if_idle(&mut self) -> Result<()> {
        if self.wide16_held && self.pending_prefill.is_none() {
            self.cuda.wide16_unmap()?;
            self.wide16_held = false;
        }
        Ok(())
    }
    /// Add a parked entry (newest last) and keep the parked list within its count and the disk cap.
    fn push_parked(&mut self, entry: CachedSession) {
        self.parked_sessions.push(entry);
        let Some((max, cap)) = self.disk_tier.as_ref().map(|t| (t.max_parked, t.cap_bytes)) else { return };
        while !self.parked_sessions.is_empty()
            && (self.parked_sessions.len() > max || self.disk_tier_bytes() as u64 > cap) {
            self.parked_sessions.remove(0);
        }
    }
    /// Disk bytes the tier holds: copies of resident entries and parked entries.
    fn disk_tier_bytes(&self) -> usize {
        self.cached_sessions.iter().chain(&self.parked_sessions).map(|c| c.disk_bytes()).sum()
    }
    /// Reload parked entry `j` on every rank; on success it is the newest cache entry. A rank without room makes every
    /// rank undo and evict the oldest resident entry (which parks if it has a copy) before the next try; a read or digest
    /// failure anywhere drops the entry everywhere (the request then prefills).
    fn reload_parked(&mut self, j: usize) -> Result<bool> {
        let began = Instant::now();
        let mut entry = self.parked_sessions.remove(j);
        let (tokens, bytes) = (entry.prefix.len(), entry.logical_bytes());
        let mut evicted = 0usize;
        // Room under the cache's own slot and byte limits first, chosen by the planner a new snapshot uses (victims park
        // when they have a copy). Without room the entry stays parked and the request prefills as it would have.
        let (budget, slots) = self.cache_capacity(0, 0);
        let entries = self.cached_sessions.iter().map(|c| (c.prefix.len(), c.charge)).collect::<Vec<_>>();
        let victims = cache_evictions(&entries, (tokens, entry.charge), budget, self.session_cache_long_first, slots);
        if !self.control.all_admit(victims.is_some())? {
            self.push_parked(entry);
            eprintln!("disk tier {}", json!({"event":"reload_over_budget","tokens":tokens,"bytes":bytes}));
            return Ok(false);
        }
        let mut victims = victims.unwrap_or_default();
        victims.sort_unstable();
        let planned = began.elapsed().as_secs_f64();
        for i in victims.into_iter().rev() { self.evict_cached_saving(i)?; evicted += 1; }
        let evicting = began.elapsed().as_secs_f64() - planned;
        let mut phases = [0.0f64; 4];
        loop {
            let local = match entry.reload_device() {
                Ok(timing) => Some(timing),
                Err(error) => { eprintln!("disk tier reload failed: {error:#}"); None }
            };
            if !self.control.all_admit(local.is_some())? {
                eprintln!("disk tier {}", json!({"event":"reload_dropped","tokens":tokens,"bytes":bytes}));
                return Ok(false);
            }
            let room = local.flatten();
            if let Some(t) = room { for k in 0..4 { phases[k] += t[k]; } }
            if self.control.all_admit(room.is_some())? { break; }
            if entry.resident_now() { entry.park_device()?; }
            if self.cached_sessions.is_empty() {
                self.push_parked(entry);
                eprintln!("disk tier {}", json!({"event":"reload_no_room","tokens":tokens,"bytes":bytes}));
                return Ok(false);
            }
            self.evict_cached(0);
            evicted += 1;
        }
        entry.used = Instant::now();
        self.cached_sessions.push(entry);
        eprintln!("disk tier {}", json!({"event":"reload","tokens":tokens,"bytes":bytes,"evicted":evicted,
            "seconds":began.elapsed().as_secs_f64(),"parked_left":self.parked_sessions.len(),
            "plan_s":planned,"evict_s":evicting,"alloc_s":phases[0],"read_s":phases[1],"digest_s":phases[2],"upload_s":phases[3]}));
        Ok(true)
    }
    /// Write-behind candidate: prompt-end images with logits first, then completed turns of other conversations, then
    /// the rest except the two newest entries (the next turn usually supersedes them: writing them would only wear the
    /// SSD); oldest first within a class. Uses only state every rank shares.
    fn disk_tier_pick(&self) -> Option<(u8, usize)> {
        let live = self.history.committed();
        let n = self.cached_sessions.len();
        self.cached_sessions.iter().enumerate().filter(|(_, c)| c.parkable() && c.resident_now()
            && !c.disk_clean() && !c.speculative)
            .map(|(i, c)| (if c.kind == "prompt_end_logits" {0u8}
                else if matches!(c.image, CachedState::Complete{..}) && !live.starts_with(&c.prefix) {1} else {2}, i))
            .filter(|&(class, i)| class < 2 || n - 1 - i >= 2).min()
    }
    /// How long the leader waits for a request before its next idle command, given its usual wait: while copies wait to be
    /// written and the cluster is quiet, idle commands follow each other closely (each writes one slice).
    pub(crate) fn tier_idle_wait(&self, usual: Duration) -> Duration {
        match &self.disk_tier {
            Some(t) if self.disk_tier_busy && self.disk_tier_backoff == 0
                && self.last_active.elapsed() >= Duration::from_secs(t.idle_secs) => usual.min(Duration::from_millis(20)),
            _ => usual,
        }
    }
    /// One idle-time step (config::disk_tier), run by every rank in lockstep: at most one slice
    /// (disk_tier::SLICE) of the copy in progress, then policy parking. Whatever can differ between ranks (quiet time,
    /// free space, budgets, write success) goes through an all-rank vote; the choices use only state every rank shares
    /// (entry kinds, the cache's oldest-first order, entry identity by birth and length).
    fn disk_tier_idle_step(&mut self) -> Result<Option<Value>> {
        let Some(tier) = self.disk_tier.clone() else { return Ok(None) };
        if !self.control.all_admit(self.last_active.elapsed() >= Duration::from_secs(tier.idle_secs))? { return Ok(None); }
        let mut event = None;
        // The copy in progress belongs to an entry that is still cached and resident; otherwise it is abandoned (the
        // partial file goes with it). The cache is the same on every rank, so is this.
        let at = self.disk_pending.as_ref().and_then(|p| self.cached_sessions.iter()
            .position(|c| c.born == p.born && c.prefix.len() == p.tokens && c.resident_now()));
        if at.is_none() { self.disk_pending = None; }
        if self.disk_pending.is_none() && let Some((class, i)) = self.disk_tier_pick() {
            let bytes = self.cached_sessions[i].logical_bytes() as u64;
            while self.disk_tier_bytes() as u64 + bytes > tier.cap_bytes && !self.parked_sessions.is_empty() {
                self.parked_sessions.remove(0);
            }
            let days = (self.disk_tier_started.elapsed().as_secs_f64() / 86400.0).max(1.0);
            let budget = (crate::disk_tier::written_bytes() + bytes) as f64 <= tier.write_bytes_per_day as f64 * days;
            let space = crate::disk_tier::free_bytes(&tier.dir).is_ok_and(|f| f >= tier.free_floor_bytes + bytes);
            let cap = self.disk_tier_bytes() as u64 + bytes <= tier.cap_bytes;
            if self.control.all_admit(budget && space && cap)? {
                let c = &self.cached_sessions[i];
                self.disk_pending = Some(DiskPending { born: c.born, tokens: c.prefix.len(), image: usize::MAX, class,
                    began: Instant::now(), write: None });
            } else {
                self.disk_tier_backoff = 20;
            }
        }
        if let Some(mut p) = self.disk_pending.take() {
            let i = self.cached_sessions.iter().position(|c| c.born == p.born && c.prefix.len() == p.tokens)
                .context("disk tier pending entry")?;
            // Start the entry's next image without a copy when none is in progress.
            let mut failed = None;
            if p.write.is_none() {
                let c = &self.cached_sessions[i];
                if let Some(k) = c.images().iter().position(|im| !im.has_disk_copy()) {
                    let began = c.images()[k].disk_write_begin(&crate::disk_tier::path_for(&tier.dir, c.kind));
                    match began { Ok(w) => { p.image = k; p.write = Some(w); } Err(e) => failed = Some(e) }
                }
            }
            if self.control.all_admit(failed.is_none())? {
                if let Some(mut w) = p.write.take() {
                    let stepped = self.cached_sessions[i].images()[p.image].disk_write_step(&mut w);
                    let finished = match stepped {
                        Ok(()) if w.done() => self.cached_sessions[i].images_mut()[p.image].disk_write_finish(w).map(|()| true),
                        Ok(()) => { p.write = Some(w); Ok(false) }
                        Err(e) => Err(e),
                    };
                    match finished { Err(e) => failed = Some(e), Ok(_) => {} }
                }
            }
            if let Some(error) = &failed { eprintln!("disk tier write failed: {error:#}"); }
            if !self.control.all_admit(failed.is_none())? {
                drop(p);
                self.cached_sessions[i].drop_disk_copies();
                self.disk_tier_backoff = 20;
                event = Some(json!({"event":"copy_failed","tokens":self.cached_sessions[i].prefix.len()}));
            } else if p.write.is_none() && self.cached_sessions[i].disk_clean() {
                let c = &self.cached_sessions[i];
                event = Some(json!({"event":"copy","kind":c.kind,"class":p.class,"tokens":c.prefix.len(),
                    "bytes":c.logical_bytes(),"seconds":p.began.elapsed().as_secs_f64(),"disk_bytes":self.disk_tier_bytes(),
                    "written_total":crate::disk_tier::written_bytes()}));
            } else {
                self.disk_pending = Some(p);
            }
        }
        // Policy parking: prompt-end images once they have a copy; completed turns of other conversations once
        // `park_after` newer entries exist.
        let keep = tier.park_after_requests as usize;
        let n = self.cached_sessions.len();
        let live = self.history.committed().to_vec();
        let park = self.cached_sessions.iter().enumerate().filter(|(i, c)| c.parkable() && c.disk_clean()
            && c.resident_now() && (c.kind == "prompt_end_logits"
                || (matches!(c.image, CachedState::Complete{..}) && !live.starts_with(&c.prefix) && n - 1 - i >= keep)))
            .map(|(i, _)| i).collect::<Vec<_>>();
        if !park.is_empty() {
            let freed = park.iter().map(|&i| self.cached_sessions[i].logical_bytes()).sum::<usize>();
            for &i in park.iter().rev() { self.evict_cached(i); }
            let parked = json!({"event":"park","entries":park.len(),"bytes_freed":freed,"parked":self.parked_sessions.len()});
            event = Some(match event { Some(e) => json!([e, parked]), None => parked });
        }
        self.disk_tier_busy = self.disk_pending.is_some() || self.disk_tier_pick().is_some();
        Ok(event)
    }
}

/// A checkpoint's token sequence is immutable for its lifetime. Compare a pair
/// exactly once while both checkpoints exist. A weak reference keeps only the
/// Vec header's identity after eviction; dropping the last owner frees its tokens.
struct CachedPrefix {
    ids: std::rc::Rc<Vec<u32>>,
    comparisons: std::cell::RefCell<Vec<(std::rc::Weak<Vec<u32>>, bool)>>,
}
impl From<Vec<u32>> for CachedPrefix {
    fn from(ids: Vec<u32>) -> Self {
        Self { ids: std::rc::Rc::new(ids), comparisons: Default::default() }
    }
}
impl std::ops::Deref for CachedPrefix {
    type Target = Vec<u32>;
    fn deref(&self) -> &Vec<u32> { &self.ids }
}
impl<T: AsRef<[u32]> + ?Sized> PartialEq<T> for CachedPrefix {
    fn eq(&self, other: &T) -> bool { self.ids.as_slice() == other.as_ref() }
}
impl CachedPrefix {
    fn extends(&self, shorter: &Self) -> bool {
        if self.len() <= shorter.len() { return false; }
        let key = std::rc::Rc::as_ptr(&shorter.ids);
        let mut comparisons = self.comparisons.borrow_mut();
        if let Some((_, result)) = comparisons.iter().find(|(other, _)| other.as_ptr() == key) {
            return *result;
        }
        let matches = self.starts_with(shorter);
        comparisons.retain(|(other, _)| other.strong_count() != 0);
        comparisons.push((std::rc::Rc::downgrade(&shorter.ids), matches));
        matches
    }
}
/// config::cache_keep_latest: what the earlier-checkpoint rule reads of an entry: the Begin that saved it, the Begin that
/// last restored it (0: never) and whether it is a prefix several conversations share.
#[derive(Clone, Copy)]
struct EntryFacts { serial: u64, restored: u64, shared: bool }
/// config::cache_keep_latest (v4, Nov 20): entry i is an earlier checkpoint when a request has gone past it: some other
/// entry R that a request restored (and not a shared prefix), restored after i was saved (i.serial < R.restored), either
/// extends i (i is an ancestor of where that conversation went on) or shares a cached ancestor A with i (A not a
/// shared prefix; A == R or R extends A, and i extends A: a branch or answer image the conversation left behind).
/// What a conversation last restored, and every image saved since, stays latest; so does a conversation's only image,
/// however long ago it was used. `ext[j][i]`: entry j extends entry i (strictly longer, same media).
/// k16b / kx (Nov 20): v1 (every extended entry earlier) and v2 (extended by a later request's prompt-side image)
/// evicted the near-end images later turns resumed from: one request's images extend one another, and follow-ups that
/// branch from one base each save images extending it, though none of those images is ever restored.
fn keep_latest_earlier_flags(facts: &[EntryFacts], ext: &[Vec<bool>]) -> Vec<bool> {
    let n = facts.len();
    (0..n).map(|i| (0..n).any(|r| {
        let fr = facts[r];
        r != i && fr.restored > 0 && !fr.shared && facts[i].serial < fr.restored
            && (ext[r][i] || (0..n).any(|a| a != i && !facts[a].shared && ext[i][a] && (a == r || ext[r][a])))
    })).collect()
}
/// What the keep-latest rule reads of a cache list: each entry's facts, and `ext[j][i]`: entry j extends entry i
/// (strictly longer, the same tokens and media through i's length).
fn keep_latest_inputs(c: &[CachedSession]) -> (Vec<EntryFacts>, Vec<Vec<bool>>) {
    let n = c.len();
    let ext: Vec<Vec<bool>> = (0..n).map(|j| (0..n).map(|i| j != i && c[j].prefix.extends(&c[i].prefix)
        && c[j].media.matches_prefix(&c[i].media, c[i].prefix.len())).collect()).collect();
    let facts: Vec<EntryFacts> = c.iter().map(|e| EntryFacts { serial: e.serial, restored: e.restored_serial,
        shared: e.shared || e.kind == "shared_prefix" }).collect();
    (facts, ext)
}
/// config::cache_superseded_first, host-memory reclaim (reclaim_optional_cache): the order in which cached images may
/// give way when an input or a restore lacks host memory. Optional images first, then earlier checkpoints (class 0),
/// then completed-turn images no request has restored (class 1), then images that a later prompt checkpoint of the
/// same conversation extends (`extended`: the conversation's prompt has grown past them, though no request has
/// restored the later checkpoint yet); each group in list order. Shared prefixes never. A conversation's newest
/// prompt checkpoint is in none of the groups.
fn host_reclaim_order(optional: &[bool], class: &[u8], extended: &[bool], shared: &[bool]) -> Vec<usize> {
    let n = optional.len();
    if class.len() != n || extended.len() != n || shared.len() != n { return Vec::new(); }
    let pick = |want: &dyn Fn(usize) -> bool| (0..n).filter(|&i| !shared[i] && want(i)).collect::<Vec<_>>();
    let mut order = pick(&|i| optional[i]);
    order.extend(pick(&|i| !optional[i] && class[i] == 0));
    order.extend(pick(&|i| !optional[i] && class[i] == 1));
    order.extend(pick(&|i| !optional[i] && class[i] == 2 && extended[i]));
    order
}
/// config::cache_superseded_first: `victims` of an eviction plan over `entries` ((prefix tokens, charge) in list order:
/// among latest images the least recently used first), moved onto the cheapest images the plan retained. `class`:
/// 0 an earlier checkpoint (the keep-latest rule), 1 a completed-turn image that no request has restored (the next
/// turn of its conversation normally resumes from the prompt-side checkpoint before it), 2 anything else: a restore
/// point in use, a prompt-side image, an idle conversation's image. A victim of class 1 or 2 that is not `keep` is
/// replaced by retained images of a lower class, class 0 before class 1, each in list order, never `excluded`, as
/// many as free at least the victim's bytes (losing a second cheap image costs nothing a conversation resumes from).
/// Victims are served most recently used first (highest index), so when the cheap images run out the hottest restore
/// points stay and the coldest go. A victim without a replacement stays. The plan's slot count and byte budget
/// still hold: no fewer entries go and no fewer bytes are freed.
/// Nov 22, third four-conversation run: the plan named an idle conversation's checkpoint and another conversation's
/// restore point used 4 s before; the first version served victims in plan order, gave the only earlier checkpoint
/// to the idle one and let the restore point go, with two never-restored completed images of 0.95 GB retained.
fn prefer_cheaper(entries: &[(usize, usize)], class: &[u8], excluded: &[bool], keep: &[bool], mut victims: Vec<usize>) -> Vec<usize> {
    let n = entries.len();
    if class.len() != n || excluded.len() != n || keep.len() != n || victims.iter().any(|&v| v >= n) { return victims; }
    let mut order: Vec<usize> = (0..victims.len()).collect();
    order.sort_by_key(|&k| std::cmp::Reverse(victims[k]));
    for k in order {
        let v = victims[k];
        if keep[v] || class[v] == 0 { continue; }
        let (mut take, mut freed) = (Vec::new(), 0usize);
        'classes: for cheaper in 0..class[v] {
            for s in 0..n {
                if freed >= entries[v].1 { break 'classes; }
                if class[s] == cheaper && !excluded[s] && !keep[s] && !victims.contains(&s) {
                    take.push(s);
                    freed += entries[s].1;
                }
            }
        }
        if freed >= entries[v].1 && !take.is_empty() {
            victims[k] = take[0];
            victims.extend_from_slice(&take[1..]);
        }
    }
    victims
}
/// The class prefer_cheaper and host_reclaim_order read of each cached image (0 earlier checkpoint, 1 completed-turn
/// image never restored, 2 anything else); `earlier` from keep_latest_earlier_flags over the same list.
fn cache_classes(cached: &[CachedSession], earlier: &[bool]) -> Vec<u8> {
    cached.iter().zip(earlier).map(|(c, &e)| if e { 0 } else if c.kind == "completed" && c.restored_serial == 0 { 1 } else { 2 }).collect()
}
/// The keep-latest order of a cache list (config::cache_keep_latest): every earlier checkpoint first, then every latest
/// image; each group keeps its existing order (oldest / least recently used first, restores refresh age). Returns the
/// new order as indices into the old list.
fn keep_latest_order(earlier: &[bool]) -> Vec<usize> {
    let n = earlier.len();
    (0..n).filter(|&i| earlier[i]).chain((0..n).filter(|&i| !earlier[i])).collect()
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

fn displaced_storage_fits(host: usize, reserve: usize, bytes: usize, pool: usize) -> bool {
    crate::state_image::image_storage_fits(host, reserve, bytes,
        bytes >= crate::cuda::POOLED_IMAGE_MIN && pool >= bytes)
}

#[cfg(test)]
#[test]
fn a_displaced_copy_needs_a_whole_allocation_from_one_memory_source() {
    let mib = 1 << 20;
    let reserve = 3584 * mib;
    let host = reserve + 849 * mib;
    let size = 886 * mib;
    assert!(!displaced_storage_fits(host, reserve, size, 112 * mib + 48 * mib));
    assert!(displaced_storage_fits(host, reserve, size, 112 * mib + 784 * mib));
    assert!(displaced_storage_fits(host + 48 * mib, reserve, size, 112 * mib));
    assert!(!displaced_storage_fits(reserve - 1, reserve, size, 896 * mib));
}

struct CacheSaveAdmission {
    admitted:bool,available:usize,local_fits:bool,evicted:usize,
    pressure_evicted:usize,forecast_refused:bool,
    recycled:Option<crate::state_image::Image>,
}

/// Reclaim only completed, optional conversation snapshots. The caller owns
/// the live session and paused-request images separately. Headroom may differ
/// by rank (only rank zero runs vision); all ranks still evict the same entry.
/// `for_image`: the room is for a snapshot (with the startup pool, evicted snapshots free pool chunks, so count them).
/// Room for anything else is host memory, which evicting pooled snapshots does not give back: with a pool, nothing is
/// evicted for it (Nov 17).
fn reclaim_optional_cache(cuda:&Cuda,control:&mut ControlChannel,
    cached:&mut Vec<CachedSession>,long_first:bool,headroom:usize,for_image:bool)->Result<Value> {
    let began=Instant::now();
    cuda.synchronize_all()?;
    let room=|| if for_image {cuda.image_admission_memory()} else {cuda.admission_memory()};
    let pointless=!for_image&&cuda.image_pool_total_bytes().is_some();
    let mut proof=b"optional-cache-reclaim-v1".to_vec();
    proof.extend_from_slice(&(cached.len() as u64).to_le_bytes());
    for entry in cached.iter() {
        proof.push(entry.owner);
        proof.extend_from_slice(&(entry.prefix.len() as u64).to_le_bytes());
        proof.extend_from_slice(&(entry.charge as u64).to_le_bytes());
        proof.push(u8::from(entry.opportunistic_logits()));
    }
    control.agree(&proof)?;
    let before=room()?;
    let mut admitted=control.all_admit(before.admitted_available > headroom)?;
    let mut evicted=Vec::new();
    while !admitted && !cached.is_empty() && !pointless {
        let index=cached.iter().enumerate().min_by_key(|(i,c)|
            (!c.opportunistic_logits(),if long_first {c.prefix.len()}else{0},*i)).unwrap().0;
        let entry=cached.remove(index);
        evicted.push(json!({"prefix_tokens":entry.prefix.len(),"charge_bytes":entry.charge}));
        drop(entry);
        admitted=control.all_admit(room()?.admitted_available > headroom)?;
    }
    // config::cache_superseded_first (Nov 22). With a pool, room for host memory was never made from the cache: pooled
    // snapshots give none back. But a snapshot saved while the pool was full came from general memory and holds it
    // for as long as it lives; four ~1M-token conversations left the head 150 MB under its reserve that way and an
    // image input was refused, with a replaced checkpoint of 0.94 GB still resident. Such images give way in
    // host_reclaim_order, never a conversation's newest prompt-side image. Whether an image sits outside the pool is a
    // rank's own fact, so each candidate is voted on: it goes everywhere when any rank holds it in general memory.
    if !admitted && pointless && crate::config::cache_superseded_first() && !cached.is_empty() {
        let (facts,ext)=keep_latest_inputs(cached);
        let earlier=keep_latest_earlier_flags(&facts,&ext);
        let n=cached.len();
        // A prompt-end image with logits does not count as the later checkpoint: other planners give those up freely,
        // so the image it extends would be the conversation's only durable restore point.
        let checkpoint=cached.iter().map(|c|matches!(c.kind,"prompt_checkpoint"|"near_end")).collect::<Vec<_>>();
        let extended=(0..n).map(|i|(0..n).any(|j|ext[j][i]&&checkpoint[j])).collect::<Vec<_>>();
        let optional=cached.iter().map(|c|c.opportunistic_logits()).collect::<Vec<_>>();
        let shared=facts.iter().map(|f|f.shared).collect::<Vec<_>>();
        let class=cache_classes(cached,&earlier);
        let order=host_reclaim_order(&optional,&class,&extended,&shared);
        let mut slots=cached.drain(..).map(Some).collect::<Vec<Option<CachedSession>>>();
        let outcome=(||->Result<bool>{
            let mut admitted=false;
            for i in order {
                let entry=slots[i].as_ref().expect("each candidate once");
                let local=entry.images().iter().any(|image|image.resident()&&image.device_bytes()>0&&image.owned_pool_chunks()==0);
                if control.all_admit(!local)? {continue;}
                let entry=slots[i].take().unwrap();
                evicted.push(json!({"prefix_tokens":entry.prefix.len(),"charge_bytes":entry.charge,"kind":entry.kind,
                    "general_memory":true,"group":if optional[i]{"optional"}else if class[i]==0{"earlier"}else if class[i]==1{"unrestored_completed"}else{"extended"}}));
                drop(entry);
                admitted=control.all_admit(room()?.admitted_available > headroom)?;
                if admitted {break;}
            }
            Ok(admitted)
        })();
        // The list goes back whatever happened: an error must not empty the cache.
        cached.extend(slots.into_iter().flatten());
        admitted=outcome?;
        if !evicted.is_empty() {
            cache_trace(||json!({"path":"host_reclaim","headroom":headroom,"admitted":admitted,"evicted":evicted,
                "entries":cached.iter().map(|c|c.trace()).collect::<Vec<_>>()}));
        }
    }
    Ok(json!({"headroom_bytes":headroom,"admitted":admitted,"evicted":evicted,"for_image":for_image,
        "remaining_cache_entries":cached.len(),"memory_before":before,
        "memory_after":room()?,"seconds":began.elapsed().as_secs_f64()}))
}

struct Worker {
    // Captured graph owners must drop before their NCCL/RoCE communicator,
    // including when any request, socket operation or output write fails.
    resident_pair: Option<crate::target::PairExecutor>,
    peer_bank: Option<resident_bank::Bank>,
    bank_id: u8,
    session: Option<GreedySession>,
    /// Batched decode (batched_serving.rs): conversations decoding in serving regions 1.., and
    /// fresh drafters for the next region-0 conversation. Dropped with the other drafters.
    peers: Vec<batched_serving::Peer>,
    spare_sessions: Vec<GreedySession>,
    /// Elastic regions: a conversation whose move into a serving region found no room in the image pool even after
    /// evicting images; it takes turns until a region frees up or another conversation is current.
    elastic_declined: Option<u64>,
    batched_timing: batched_serving::Timing,
    cached_sessions: Vec<CachedSession>,
    /// Disk tier (config::disk_tier): snapshots parked on disk (RAM freed), oldest first. Every rank holds the same list.
    parked_sessions: Vec<CachedSession>,
    disk_tier: Option<crate::config::DiskTierConfig>,
    disk_tier_started: Instant,
    /// config::wide16_elastic: this worker lent the 16,384-row profile its extension for a request's prefill.
    wide16_held: bool,
    /// config::cache_keep_latest: counts Begin commands (the same on every rank); cache entries record the one that saved them.
    cache_serial: u64,
    disk_tier_backoff: u32,
    disk_pending: Option<DiskPending>,
    /// Elastic serving regions: bit k set while region k's spans are mapped (batched_serving::map_region / release_regions).
    elastic_mapped: u8,
    /// Map by reach (config::region_map_by_reach): positions of region k's compressed and key caches mapped now, in
    /// granules of config::REGION_MAP_GRANULE (0: none; elastic_mapped then covers the owners' key matrices only).
    region_cache_reach: [usize; 4],
    /// Map by reach: when region k's caches could not grow, the position of its conversation from which a retry is due
    /// (0: none; replicated state only, so every rank retries alike).
    region_grow_retry: [usize; 4],
    /// Copies wait to be written (the leader shortens its idle wait: tier_idle_wait).
    disk_tier_busy: bool,
    draft_tails: std::collections::VecDeque<DraftTail>,
    #[cfg(feature = "storage-fault-injection")]
    cache_save_fault: cache_admission_fault::Fault,
    session_cache_budget: usize,
    session_cache_slots: usize,
    // Finite forward-turn screen only until its maintenance tradeoff is checked.
    skip_replaced_completion: bool,
    shared_image_budget: usize,
    parking_extra_slots: bool,
    suspended_image_bytes: usize,
    suspended_image_count: usize,
    loaded_request_is_suspended: bool,
    session_cache_long_first: bool,
    checkpoint_ladder: bool,
    displaced_snapshot: bool,
    session_cache_shared_ancestors: bool,
    cache_pressure_reclaim: bool,
    prompt_checkpoint: bool,
    encoder_prompt_checkpoint: bool,
    encoder_active_frontier: bool,
    encoder_duplicate_skip: bool,
    encoder_image_recycle: bool,
    idle_anchor_enabled: bool,
    idle_anchor_rolling: bool,
    encoder_upgrade_reconstructed: bool,
    prompt_checkpoint_tail: usize,
    prompt_cache_logits: bool,
    retain_restored_checkpoint: bool,
    // CANADA_SCRATCH_REDUNDANT_PROMPT_IMAGE (see config::redundant_prompt_image).
    redundant_prompt_image: bool,
    prepared_session: Option<PreparedSession>,
    session_prepare_seconds: f64,
    nccl_init_seconds: f64,
    target_lineage: u64,
    verify: Target,
    long_verify: Option<Target>,
    long_history_buffers: Option<LongHistoryBuffers>,
    copy32_verify: Option<Target>,
    copy32_buffers: Option<LongHistoryBuffers>,
    copy32_admission: Value,
    retained_append32: bool,
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
    /// The active job's output while a Begin executes (CANADA_SCRATCH_EARLY_FIRST_TOKEN), and whether its first
    /// token already went out.
    #[cfg(feature = "serving")]
    early_emit: Option<crate::http_server::EarlyEmitter>,
    /// config::copy_hints: hints received for the next Begin (prompt length, prompt tail, hint tokens), and the
    /// built hint sources of the requests that began with hints and have not finished (request id, lookup; at
    /// most four, one per serving region, oldest dropped; at most 10.5 MiB each). A request parked and resumed, or stepping after another began, finds its own.
    pending_copy_hints: Option<(usize, Vec<u32>, Vec<u32>)>,
    copy_hints: Vec<(u64, std::rc::Rc<crate::history_lookup::HintIndex>)>,
    early_first_emitted: bool,
    current: Option<u64>,
    /// Within-load screens: the arm the current request ran (printed with its decode summary).
    alternate_tag: Option<&'static str>,
    pending_prefill: Option<PendingPrefill>,
    prefill_cancelled: bool,
    decode_summary: Option<DecodeSummary>,
    /// config::idle_heartbeat: the setting, the kernel module, when the last request began or ended here, and the
    /// beats issued (rank-local).
    heartbeat: Option<(u64, u32, u64, u32)>,
    heartbeat_module: Option<crate::cuda::Module>,
    last_active: Instant,
    heartbeats: u64,
    /// Diagnostic (decode summary): when the last event of the active request was built, and the begin's own
    /// record (ms of the whole Begin, its phases, uncached tokens, early first token).
    last_event_at: Option<Instant>,
    begin_record: Option<(f64, Value, usize, bool)>,
    displaced_checkpoint_enabled: bool,
    displaced_reclaim: bool,
    cache_cost_order: bool,
    media: Option<crate::worker_media::Media>,
    active_media: Option<std::rc::Rc<crate::prompt_media::Bindings>>,
    media_receipt: Value,
    background_prefill_enabled: bool,
    peer_prefill_enabled: bool,
    peer_prefill_slice: Duration,
    trim_free_host_pages: bool,
    state_pressure_trim: Option<Value>,
    /// Why the last pause (request_checkpoint::checkpoint) was refused: stage, measured and required memory.
    /// Reported once in the scheduler's park event.
    park_refusal: Option<Value>,
    emitted: usize,
    limit: usize,
    identity: String,
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.history.retire();
        if let Some(peer)=&mut self.peer_bank {peer.retire();}
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
        if let Some(target)=&mut self.copy32_verify {target.clear_prompt_media_rows(&self.stream)?;}
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
                    &mut |control,headroom|reclaim_optional_cache(cuda,control,cached,long_first,headroom,false))?;
            self.install_media(Some(bindings))?;
            self.media_receipt=receipt;
        }
        Ok(())
    }
    /// Called only when input preparation refuses capacity, before any new
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
        Roce::validate_startup(&p.kernels.join(crate::config::roce_bundle()?))?;
        let state_copy_kernel=crate::state_image::copy::configure_worker()?;
        let use_copy32=crate::config::history32_generation_enabled()?;
        let peer_prefill_enabled=match std::env::var("CANADA_SCRATCH_PEER_PREFILL") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid peer prefill policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
        };
        ensure!(!peer_prefill_enabled||(std::env::var("CANADA_SCRATCH_TWO_STREAM_ADMISSION").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_BACKGROUND_PREFILL").as_deref()==Ok("1")),
            "peer prefill requires bounded two-owner resumable scheduling");
        // Same-class fairness is deliberately coarse. A short subagent prompt
        // normally finishes without paying another owner's decoding quantum.
        // Cancellation and foreground priority continue to use every boundary.
        let peer_prefill_slice_ms=match std::env::var("CANADA_SCRATCH_PEER_PREFILL_SLICE_MS") {
            Ok(v)=>v.parse::<u64>().context("invalid peer prefill slice")?,
            Err(std::env::VarError::NotPresent)=>12_000,Err(e)=>return Err(e.into()),
        };
        ensure!((1_000..=30_000).contains(&peer_prefill_slice_ms),"peer prefill slice must be1000..30000ms");
        let peer_prefill_slice=Duration::from_millis(peer_prefill_slice_ms);
        let parking_extra_slots=match std::env::var("CANADA_SCRATCH_PARKING_EXTRA_SLOTS") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid parking slot policy");value=="1"},
            // s1076: count-only parking evictions fell2/1→0/0 with the
            // same byte budget and reserve. Keep an explicit control switch.
            Err(std::env::VarError::NotPresent)=>true,
            Err(error)=>return Err(error.into()),
        };
        let cache_pressure_reclaim=cache_pressure_reclaim()?;
        let session_cache_budget=match std::env::var("CANADA_SCRATCH_SESSION_CACHE_BYTES") {
            Err(std::env::VarError::NotPresent)=>0,
            Ok(value)=>value.parse::<usize>()?,
            Err(error)=>return Err(error.into()),
        };
        let session_cache_slots=configured_session_cache_slots()?;
        // Every save is still admitted against real available memory above the floor; this is only the ceiling.
        // After the FP4 cache a 1M conversation's image is about 1 GB, so 8 GiB holds four with their ladders;
        // with the snapshot pool the ceiling is the pool's size when larger (cache_budget_ceiling).
        let pool_mib=image_pool_mib()?;
        ensure!(session_cache_budget<=cache_budget_ceiling(pool_mib)
            && (session_cache_budget==0 || std::env::var("CANADA_SCRATCH_COMPACT_STATE_IMAGE").as_deref()==Ok("1")),
            "session cache requires compact images and at most 8 GiB (or the snapshot pool's size) of additional state");
        let shared_image_budget=match std::env::var("CANADA_SCRATCH_SHARED_IMAGE_BYTES") {
            Err(std::env::VarError::NotPresent)=>2*1024usize.pow(3),
            Ok(value)=>value.parse::<usize>()?,
            Err(error)=>return Err(error.into()),
        };
        validate_shared_image_budget(shared_image_budget,crate::config::single_bank_requested(),
            request_checkpoint::exchange_enabled()?,pool_mib)?;
        let session_cache_long_first=match std::env::var("CANADA_SCRATCH_SESSION_CACHE_POLICY") {
            Err(std::env::VarError::NotPresent)=>true,
            Ok(value)if value=="fifo"=>false,
            Ok(value)if value=="long_parents_and_recent" || value=="longest_prefix"=>true,
            _=>anyhow::bail!("session cache policy must be fifo or long_parents_and_recent"),
        };
        let checkpoint_ladder=checkpoint_ladder()?;
        let layer20_defer=crate::target::layer20_defer_enabled()?;
        let displaced_snapshot=displaced_snapshot()?;
        let session_cache_shared_ancestors=match std::env::var("CANADA_SCRATCH_CACHE_SHARED_ANCESTORS") {
            Err(std::env::VarError::NotPresent)=>false,
            Ok(v)if v=="0"=>false,Ok(v)if v=="1"=>true,
            _=>anyhow::bail!("shared ancestor cache policy must be0or1"),
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
        crate::config::validate_flex_widths()?;
        crate::engram::validate_miss_threads()?;
        crate::engram::validate_direct_reads()?;
        crate::config::draft_head_subset()?;
        ensure!(crate::config::flex_widths().iter().all(|w| p.wide_widths.contains(w)),
            "every flex encoder width needs a configured wide profile");
        // Bundles are resolved by the row count in their names (config::resolve_flex_bundle). A flex width equal
        // to another flex profile's per-rank share (640 beside 2,560; 1,280 beside 5,120) is resolved by the
        // profile under construction (config::building_wide_profile, s1519; such sets were refused before).
        // Beside a native width (256 and 1,024) the native profile's shard bundles exist under their own names
        // and win.
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
        if std::env::var("CANADA_SCRATCH_CORE_MODULE_CACHE_SCREEN").as_deref()==Ok("1") {
            cuda.share_immutable_core(&core_bytes)?;
        }
        let core = cuda.module(&core_bytes)?;
        let startup_padding=crate::allocation_placement_probe::padding(&cuda,&stream)?;
        let fragment_soak=crate::allocation_placement_probe::soak(&cuda,&core,&stream,p.rank)?;
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
        if !startup_padding.is_empty(){
            stream.synchronize()?;
            ensure!(cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes(),"startup padding consumed physical reserve");
            drop(startup_padding);
            eprintln!("startup padding released before profile forks: {}",json!(cuda.admission_memory()?));
        }
        if !fragment_soak.is_empty(){
            stream.synchronize()?;drop(fragment_soak);
            eprintln!("rank{} fragment soak released after the target load: {}",p.rank,json!(cuda.admission_memory()?));
        }
        if crate::allocation_placement_probe::census_requested()? {
            crate::allocation_placement_probe::census(&cuda,&core,&large,&stream,p.rank)?;
        }
        large.set_graph_replay(true)?;
        large.set_prefill_last_head(true)?;
        large.enable_decoder_tail(&core_bytes)?;
        // Batched decode: one encoder tail per serving region, shared by every profile forked below.
        if batched_serving::enabled()? { large.enable_region_tails(crate::config::batched_regions()?)?; }
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
        let loan_narrow=std::env::var("CANADA_SCRATCH_NARROW_ARENA_LOAN_SCREEN").as_deref()==Ok("1");
        let fork_narrow=|rows|->Result<Target> {
            if loan_narrow {large.fork_profile_lending_prefill_for_probe(&core,&core_bytes,&stream,&p.kernels,
                rows,Phase::Verify,wides.iter().max_by_key(|t|t.physical_rows()).context("complete prefill arena bank")?)}
            else{large.fork_profile(&core,&core_bytes,&stream,&p.kernels,rows,Phase::Verify)}
        };
        let mut verify = fork_narrow(6)?;
        ensure!(verify.adaptive_sparse_index_layers() == if adaptive_sparse {4} else {0},
            "resident adaptive sparse construction differs from admitted policy");
        let compact=crate::query_index::compact_sparse_index_enabled(p.context,adaptive_sparse)?;
        ensure!(verify.compact_sparse_index_layers()==if compact {4}else{0},
            "resident compact index construction differs from admitted policy");
        // An owner sized for several sessions gathers without the single-frontier cache (index.rs).
        ensure!(verify.cached_index_gather_layers()==if std::env::var("CANADA_SCRATCH_CACHED_INDEX_GATHER").as_deref()==Ok("1")
            &&crate::config::batch_screen_sessions()?==1 {8}else{0},
            "resident gathered-prefix construction differs from admitted policy");
        ensure!(verify.fused_shared_quantization_layers()==if std::env::var("CANADA_SCRATCH_FUSED_SHARED_QUANT").as_deref()==Ok("1") {40}else{0},
            "resident shared fusion construction differs from admitted policy");
        verify.set_graph_replay(true)?;
        let schedule = verify.apply_candidate_schedule(&stream)?;
        eprintln!("rank{} candidate verifier schedule: {schedule}", p.rank);
        if crate::layer_clock::requested() {
            // Diagnostic (default off): the device clock's part marks in the six-row verifier's graphs of this load.
            crate::layer_clock::arm(&cuda, &core_bytes, &stream)?;
            crate::layer_clock::serving_mode();
            eprintln!("rank{} layer clock armed for the six-row verifier (diagnostic load)", p.rank);
        }
        ensure!(
            large.shared_state_report(&verify)["all_weights_and_causal_storage_shared"] == true,
            "resident verifier duplicated state"
        );
        let use_long_history=match std::env::var("CANADA_SCRATCH_LONG_HISTORY_GENERATION") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid long history policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
        };
        let compact_history_sampling=Target::compact_history_sampling_requested()?;
        ensure!(!compact_history_sampling||use_long_history,"compact sampling needs the long-history verifier");
        let (long_verify,long_history_buffers)=if use_long_history {
            ensure!(p.context==1048576 && history_lookup_enabled,"long history needs full-capacity history lookup");
            let mut target=fork_narrow(16)?;
            target.set_graph_replay(true)?;
            if compact_history_sampling {target.enable_compact_history_sampling(&core_bytes,&stream)?;}
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
        let mut prepared_session = PreparedSession::new(&cuda, &core_bytes, prepared_draft)?
            .with_serving_policy(&stream)?;
        stream.synchronize()?;
        let heartbeat = crate::config::idle_heartbeat()?;
        let heartbeat_module = match heartbeat {
            Some(setting) => {
                let module = cuda.module(&core_bytes)?;
                module.function("ca_inject_spin")?;
                eprintln!("rank{} idle heartbeat: every {} ms a spin of {} us on {} block(s), for {} s after a request", p.rank, setting.0, setting.1, setting.3, setting.2);
                Some(module)
            }
            None => None,
        };
        // crate::head4_rescore (default off; an approximation by design): the verify owner's head through the
        // drafter's 4-bit copy. The screen flag installs it switched off.
        {
            let configured = crate::config::head4_rescore()?;
            let screen = crate::config::head4_rescore_screen();
            if let Some((cap, window)) = configured.or(screen.then_some((512, 24.0))) {
                let head = prepared_session.nvfp4_head()
                    .context("CANADA_SCRATCH_HEAD4_RESCORE needs the drafter's 4-bit head (CANADA_SCRATCH_DRAFT_HEAD_NVFP4=1)")?;
                let on = configured.is_some() && !screen;
                verify.install_head4_rescore(head, cap, window, on)?;
                // The option changes the model's output distribution by design (crate::head4_rescore: nature and measured
                // size); the line says so wherever a log is read.
                eprintln!("rank{} verify head: 4-bit copy with exact rescoring installed (cap {cap} rows a rank, window {window}); active {on}; \
                    an option that is off by default: it approximates the model's own output head (about one draw in 400,000-660,000 \
                    differs at 512,24; see src/head4_rescore.rs)", p.rank);
            }
        }
        let session_prepare_seconds = prepare_started.elapsed().as_secs_f64();
        eprintln!("rank{} allocation boundary before vision: {}",p.rank,json!(cuda.admission_memory()?));
        let mut media=p.vision.as_ref().map(|config|crate::worker_media::Media::load(&cuda,p.rank,&cfg,config)).transpose()?;
        if let Some(media)=&mut media {
            let arena=wides.iter().max_by_key(|t|t.physical_rows())
                .map(|t|t.image_scratch_arena(&stream)).transpose()?.flatten();
            media.lend_prefill_scratch(arena);
        }
        eprintln!("rank{} allocation boundary after vision: {}",p.rank,json!(cuda.admission_memory()?));
        ensure!(
            cuda.admission_memory()?.admitted_available >= crate::memory::floor_bytes(),
            "resident profiles and drafter consumed the runtime reserve"
        );
        let ticket = std::fs::read(&p.ticket)?;
        let mut comm = Roce::local(&cuda, &p.kernels.join(crate::config::roce_bundle()?), p.rank)?;
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
        comm.attach_oneshot_twin(&core_bytes, false)?;
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
        control.agree(&crate::control_channel::kernel_poll_us_requested()?.to_le_bytes())?;
        control.agree(&(session_cache_budget as u64).to_le_bytes())?;
        control.agree(&(session_cache_slots as u64).to_le_bytes())?;
        let reserve=parse_short_reserve(std::env::var("CANADA_SCRATCH_SESSION_CACHE_SHORT_RESERVE").ok().as_deref(),session_cache_slots)?;
        control.agree(&[&[0x7e][..],&(reserve as u64).to_le_bytes()].concat())?;
        SHORT_RESERVE.store(reserve,std::sync::atomic::Ordering::Relaxed);
        let cap=parse_ladder_cap(std::env::var("CANADA_SCRATCH_CACHE_LADDER_CAP").ok().as_deref())?;
        control.agree(&[&[0x7f][..],&(cap as u64).to_le_bytes()].concat())?;
        LADDER_CAP.store(cap,std::sync::atomic::Ordering::Relaxed);
        // Diagnostic only (no decision depends on it), so each rank reads it alone: no agreement needed.
        CACHE_TRACE.store(parse_cache_trace(std::env::var("CANADA_SCRATCH_CACHE_TRACE").ok().as_deref())?,
            std::sync::atomic::Ordering::Relaxed);
        control.agree(&(shared_image_budget as u64).to_le_bytes())?;
        control.agree(&[u8::from(parking_extra_slots)])?;
        control.agree(&[u8::from(session_cache_long_first)])?;
        control.agree(&[u8::from(checkpoint_ladder)])?;
        // Every rank must write and read the same target cache format (CANADA_SCRATCH_KV_FORMAT).
        control.agree(&[crate::attention::kv_format_code()?])?;
        // All ranks must admit against the same physical floor (test override included).
        control.agree(&(crate::memory::validate_floor()? as u64).to_le_bytes())?;
        control.agree(&[u8::from(layer20_defer)])?;
        control.agree(&[u8::from(crate::config::encoder_checkpoint_inline_requested()?)])?;
        control.agree(&[u8::from(displaced_checkpoint()?)])?;
        control.agree(&[0x7d,u8::from(displaced_reclaim()?)])?;
        control.agree(&[0x7b,u8::from(cache_cost_order()?)])?;
        // Nov 12 cache policies: every rank must run the same ones (they decide positions, saves and copies).
        control.agree(&[0x7c,crate::config::shared_prefix_checkpoint()?,u8::from(crate::config::encoder_extra_near_end()?),
            u8::from(crate::config::batched_detach_keeps_images()?)])?;
        // Nov 18: the near-end image after a restore made inside the lookup changes the chunk plan on every rank.
        control.agree(&[0x6e,u8::from(crate::config::near_end_after_restore()?)])?;
        // Nov 18: redundant prompt-end images yield first; it changes eviction victims on every rank.
        control.agree(&[0x6f,u8::from(crate::config::redundant_prompt_image()?)])?;
        // Nov 19: exact window pages move every serving region's window base; every rank must lay them out alike.
        control.agree(&[0x6d,u8::from(crate::config::validate_swa_exact_pages()?)])?;
        // Nov 19: the keep-latest order decides eviction victims on every rank.
        control.agree(&[0x6c,u8::from(crate::config::validate_cache_keep_latest()?)])?;
        // Nov 22: validated here so a bad value fails at startup (the overlap itself is rank-local).
        control.agree(&[0x69,crate::config::validate_engram_media_overlap()?])?;
        // Nov 22: superseded-first changes eviction victims on every rank.
        control.agree(&[0x6a,u8::from(crate::config::validate_cache_superseded_first()?)])?;
        // Nov 19: profile dedup (config::validate_profile_dedup) moves profiles' buffers only; every rank runs one layout.
        control.agree(&[0x6b,crate::config::validate_profile_dedup()?])?;
        // Host waits (Nov 12): the control socket's spin budget and the worker thread's timer slack
        // (CANADA_SCRATCH_TIMER_SLACK_NS, 1000-50000; default: the system's 50 us): a 25 us completion poll
        // sleeps ~80 us with the default slack.
        let slack=crate::config::timer_slack_ns()?;
        control.agree(&[&[0x7f][..],&crate::control_channel::validate_control_spin_us()?.to_le_bytes(),&slack.to_le_bytes()].concat())?;
        #[cfg(target_os="linux")]
        if slack>0 {
            ensure!(unsafe{libc::prctl(libc::PR_SET_TIMERSLACK,slack as libc::c_ulong)}==0,
                "timer slack: {}",std::io::Error::last_os_error());
        }
        control.agree(&[&[0x7d][..],&crate::config::validate_encoder_chunk_costs()?.iter()
            .flat_map(|&(rows,seconds)|[(rows as u64).to_le_bytes(),seconds.to_bits().to_le_bytes()].concat()).collect::<Vec<u8>>()].concat())?;
        control.agree(&[u8::from(displaced_snapshot)])?;
        control.agree(&[u8::from(cache_pressure_reclaim)])?;
        control.agree(&[u8::from(state_copy_kernel)])?;
        control.agree(&[u8::from(prompt_checkpoint)])?;
        control.agree(&[u8::from(encoder_prompt_checkpoint)])?;
        control.agree(&[u8::from(encoder_active_frontier)])?;
        let encoder_duplicate_skip=encoder_duplicate_skip()?;
        control.agree(&[u8::from(encoder_duplicate_skip)])?;
        let encoder_image_recycle=encoder_image_recycle()?;
        ensure!(!encoder_image_recycle||cache_pressure_reclaim,"encoder reuse requires coordinated cache reclamation");
        control.agree(&[u8::from(encoder_image_recycle)])?;
        let idle_anchor_enabled=idle_anchor_enabled()?;
        ensure!(!idle_anchor_enabled||(encoder_active_frontier&&crate::config::bank_encoder_checkpoint_requested()),
            "idle anchors require active bank encoder checkpoints");
        control.agree(&[u8::from(idle_anchor_enabled)])?;
        let idle_anchor_rolling=idle_anchor_rolling()?;
        ensure!(!idle_anchor_rolling||idle_anchor_enabled,"rolling anchors require idle preparation");
        control.agree(&[u8::from(idle_anchor_rolling)])?;
        control.agree(&[u8::from(encoder_upgrade_reconstructed)])?;
        control.agree(&[u8::from(crate::target::defer_checkpoint_decoder128()?)])?;
        control.agree(&(prompt_checkpoint_tail as u64).to_le_bytes())?;
        // CANADA_SCRATCH_PROMPT_CHECKPOINT_MIN (default 8192): the shortest prefix that gets a reusable checkpoint.
        control.agree(&(crate::config::validate_prompt_checkpoint_min()? as u64).to_le_bytes())?;
        // CANADA_SCRATCH_EXPERTS_TAIL: an invalid value refuses the load; ranks agree on the gate.
        control.agree(&(crate::config::validate_experts_tail()?.unwrap_or(0) as u64).to_le_bytes())?;
        // CANADA_SCRATCH_WIDE_ARENA_KIB_PER_ROW: an invalid value refuses the load; ranks agree on the size.
        control.agree(&(crate::config::validate_wide_arena_kib_per_row()?.unwrap_or(0) as u64).to_le_bytes())?;
        control.agree(&[u8::from(prompt_cache_logits)])?;
        control.agree(&[u8::from(retain_restored_checkpoint)])?;
        control.agree(&[u8::from(use_long_history)])?;
        control.agree(&[u8::from(compact_history_sampling)])?;
        control.agree(&Target::verification_poll_us_requested()?.to_le_bytes())?;
        control.agree(crate::config::roce_bundle()?.as_bytes())?;
        control.agree(&[u8::from(crate::mhc::pdl_pair_requested()?)])?;
        control.agree(&[u8::from(crate::mhc::pdl_stats_requested()?)])?;
        let (late6, late16) = (crate::mhc::stats_late_requested(6)?, crate::mhc::stats_late_requested(16)?);
        control.agree(&[late6.0, late6.1, late16.0, late16.1])?;
        control.agree(&[u8::from(crate::weights::early_layer_reservation_requested()?)])?;
        control.agree(&[u8::from(crate::weights::compressed_target_weights_requested()?)])?;
        control.agree(&[u8::from(crate::weights::external_target_pool_path()?.is_some())])?;
        control.agree(&[LongHistoryBuffers::minimum_match_requested(16)? as u8,
            LongHistoryBuffers::minimum_match_requested(32)? as u8])?;
        let native_prefill_rs=match std::env::var("CANADA_SCRATCH_PREFILL_NATIVE_RS") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid native prefill RS policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(error)=>return Err(error.into()),
        };
        control.agree(&[u8::from(native_prefill_rs)])?;
        if native_prefill_rs {comm.attach_native_prefill_rs(&p.kernels.join("prefill-rs-native-v2"),&stream)?;}
        // This optional profile shares canonical weights and causal storage.
        // Admit after vision, drafter and communication allocations, on every
        // rank. Its measured final workspace is~1.9GiB and construction peaks
        // around2.7GiB; 12GiB leaves the8GiB physical reserve plus margin.
        // Refusal keeps M16/M6 serving available under the same memory caps.
        control.agree(&[u8::from(use_copy32)])?;
        let requested_append32=crate::config::resident_append32_requested()?;
        control.agree(&[u8::from(requested_append32)])?;
        control.agree(&[u8::from(crate::config::append_last_gather_requested()?)])?;
        control.agree(&[u8::from(crate::config::append_encoded_head_requested()?)])?;
        control.agree(&[u8::from(crate::config::ffn_fine_hint_requested()?)])?;
        let adaptive_engram32=Target::adaptive_engram32_requested()?;
        control.agree(&[u8::from(adaptive_engram32)])?;
        control.agree(&[u8::from(peer_prefill_enabled)])?;
        control.agree(&peer_prefill_slice_ms.to_le_bytes())?;
        control.agree(&[u8::from(LongHistoryBuffers::wide_conditioning_requested()?)])?;
        let copy32_before=cuda.admission_memory()?;
        let mut copy32_verify=None;let mut copy32_buffers=None;let mut copy_conditioning_prepared=false;
        let copy32_fit=use_copy32&&control.all_admit(copy32_before.admitted_available>=12*1024usize.pow(3))?;
        if copy32_fit {
            ensure!(use_long_history&&long_verify.is_some(),"M32 requires the existing M16 fallback");
            let mut target=fork_narrow(32)?;
            target.set_graph_replay(true)?;
            target.apply_adaptive_engram32(adaptive_engram32)?;
            if compact_history_sampling {target.enable_compact_history_sampling(&core_bytes,&stream)?;}
            ensure!(large.shared_state_report(&target)["all_weights_and_causal_storage_shared"]==true,
                "M32 serving duplicated model state");
            let buffers=LongHistoryBuffers::with_rows(&cuda,&core_bytes,32)?;
            copy_conditioning_prepared=prepared_session.prepare_copy_conditioning(&buffers,&comm,&stream)?;
            stream.synchronize()?;
            if control.all_admit(cuda.admission_memory()?.admitted_available>=(crate::memory::floor_bytes()+1024usize.pow(3)))? {
                copy32_verify=Some(target);copy32_buffers=Some(buffers);
            }else if copy_conditioning_prepared {
                prepared_session.release_copy_conditioning(&stream)?;copy_conditioning_prepared=false;
            }
        }
        let copy32_after=cuda.admission_memory()?;
        ensure!(control.all_admit(copy32_after.admitted_available>=crate::memory::floor_bytes())?,
            "M32 admission consumed physical reserve");
        let copy32_admission=json!({"requested":use_copy32,"admitted":copy32_verify.is_some(),
            "adaptive_engram_overlap":adaptive_engram32&&copy32_verify.is_some(),
            "compact_history_sampling":compact_history_sampling,
            "wide_copy_conditioning_prepared":copy_conditioning_prepared,
            "construction_headroom_bytes":12*1024usize.pow(3),"memory_before":copy32_before,"memory_after":copy32_after});
        let retained_append32=requested_append32&&copy32_verify.is_some()&&copy32_buffers.is_some();
        if use_copy32 {eprintln!("rank{} M32 serving admission: {copy32_admission}",p.rank);}
        #[cfg(feature = "storage-fault-injection")]
        control.agree(&[u8::from(std::env::var("CANADA_SCRATCH_CACHE_RETENTION_SCREEN").as_deref()==Ok("1"))])?;
        let history = ResidentHistory::new(p.context, cfg.eos_token_id)?;
        let trim_free_host_pages = match std::env::var("CANADA_SCRATCH_STATE_PRESSURE_TRIM") {
            Ok(v) => { ensure!(v=="0" || v=="1", "host allocator trim policy"); v=="1" },
            Err(std::env::VarError::NotPresent) => false,
            Err(e) => return Err(e.into()),
        };
        control.agree(&[u8::from(trim_free_host_pages)])?;
        let prefill_encoded_head=crate::config::prefill_encoded_head_requested()?;
        control.agree(&[u8::from(prefill_encoded_head)])?;
        if prefill_encoded_head {large.set_prefill_encoded_head(true,&stream)?;}
        let prefill_row_head=crate::config::prefill_row_head_requested()?;
        control.agree(&[u8::from(prefill_row_head)])?;
        if prefill_row_head {large.set_prefill_row_head(true,&stream)?;}
        control.agree(&[u8::from(crate::moe::wide_transfer_requested()?)])?;
        control.agree(&[u8::from(crate::moe::prefill_transfer_requested()?),
            u8::from(crate::target::prefill_upper_rows_requested()?)])?;
        control.agree(&[u8::from(crate::woa_medium::compact_policy()?),
            u8::from(std::env::var("CANADA_SCRATCH_WOA_MID_SCREEN").as_deref()==Ok("1")),
            u8::from(std::env::var("CANADA_SCRATCH_WOA_MID_REFERENCE_SCREEN").as_deref()==Ok("1"))])?;
        control.agree(&[crate::l2_prefetch::wide_mask()?,
            u8::from(std::env::var("CANADA_SCRATCH_WIDE_L2_SCREEN").as_deref()==Ok("1"))])?;
        control.agree(&[u8::from(crate::woa_medium::prefill_live_policy()?),
            u8::from(std::env::var("CANADA_SCRATCH_WOA_PREFILL_SCREEN").as_deref()==Ok("1"))])?;
        let sparse_publish=crate::config::prefill_sparse_publish_requested()?;
        control.agree(&[u8::from(sparse_publish)])?;
        if sparse_publish {large.set_prefill_sparse_publish(true,&stream)?;}
        Ok(Self {
            resident_pair: None,
            peer_bank: None,
            bank_id: 0,
            session: None,
            peers: Vec::new(),
            spare_sessions: Vec::new(),
            elastic_declined: None,
            batched_timing: Default::default(),
            cached_sessions:Vec::new(),
            parked_sessions:Vec::new(),
            disk_tier:None,
            disk_tier_started:Instant::now(),
            wide16_held:false,
            cache_serial:0,
            disk_tier_backoff:0,
            disk_pending:None,
            elastic_mapped:0,
            region_cache_reach:[0; 4],
            region_grow_retry:[0; 4],
            disk_tier_busy:false,
            draft_tails: Default::default(),
            #[cfg(feature = "storage-fault-injection")]
            cache_save_fault: cache_admission_fault::Fault::default(),
            session_cache_budget,
            session_cache_slots,
            skip_replaced_completion: false,
            shared_image_budget,
            parking_extra_slots,
            suspended_image_bytes:0,
            suspended_image_count:0,
            loaded_request_is_suspended:false,
            session_cache_long_first,
            checkpoint_ladder,
            displaced_snapshot,
            session_cache_shared_ancestors,
            cache_pressure_reclaim,
            prompt_checkpoint,
            encoder_prompt_checkpoint,
            encoder_active_frontier,
            encoder_duplicate_skip,
            encoder_image_recycle,
            idle_anchor_enabled,
            idle_anchor_rolling,
            encoder_upgrade_reconstructed,
            prompt_checkpoint_tail,
            prompt_cache_logits,
            retain_restored_checkpoint,
            redundant_prompt_image: crate::config::redundant_prompt_image()?,
            prepared_session: Some(prepared_session),
            session_prepare_seconds,
            nccl_init_seconds,
            target_lineage: 1,
            verify,
            long_verify,
            long_history_buffers,
            copy32_verify,
            copy32_buffers,
            copy32_admission,
            retained_append32,
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
            trim_free_host_pages,
            state_pressure_trim: None,
            park_refusal: None,
            progress_next: false,
            #[cfg(feature = "serving")]
            early_emit: None,
            pending_copy_hints: None,
            copy_hints: Vec::new(),
            early_first_emitted: false,
            current: None,
            pending_prefill: None,
            prefill_cancelled: false,
            decode_summary: None,
            heartbeat,
            heartbeat_module,
            last_active: Instant::now(),
            heartbeats: 0,
            last_event_at: None,
            begin_record: None,
            displaced_checkpoint_enabled: displaced_checkpoint()?,
            displaced_reclaim: displaced_reclaim()?,
            cache_cost_order: cache_cost_order()?,
            media,
            active_media: None,
            media_receipt: Value::Null,
            background_prefill_enabled: match std::env::var("CANADA_SCRATCH_BACKGROUND_PREFILL") {
                Ok(v)=>{ensure!(v=="0"||v=="1","invalid background prefill policy");v=="1"},
                Err(std::env::VarError::NotPresent)=>false,Err(error)=>return Err(error.into()),
            },
            peer_prefill_enabled,
            peer_prefill_slice,
            emitted: 0,
            alternate_tag: None,
            limit: 0,
            identity: identity.iter().map(|x| format!("{x:02x}")).collect(),
        })
    }

    /// How long the leader waits for a request before its next idle command (config::idle_heartbeat shortens it
    /// while the last request is recent).
    pub(crate) fn idle_wait(&self) -> Duration {
        self.tier_idle_wait(match self.heartbeat {
            Some((period, _, window, _)) if self.last_active.elapsed() < Duration::from_secs(window) => Duration::from_millis(period),
            _ => Duration::from_millis(500),
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

    /// Entries that give way first when an incoming image of `incoming` tokens needs room: the opportunistic
    /// ones and, for an image of 8,192 tokens or more, checkpoints under 8,192 tokens. With a short reserve
    /// (CANADA_SCRATCH_SESSION_CACHE_SHORT_RESERVE) only those beyond its newest R (entries are ordered oldest
    /// first), so a long conversation cannot empty the short entries' share.
    fn optional_entries(&self,incoming:usize)->Vec<bool> {
        let reserve=short_reserve();
        let short=|c:&CachedSession|c.prefix.len()<8192&&!c.opportunistic_logits();
        let mut spare=self.cached_sessions.iter().filter(|c|short(c)).count().saturating_sub(reserve);
        let redundant=self.redundant_prompt_images();
        self.cached_sessions.iter().zip(redundant).map(|(c,r)|r||c.opportunistic_logits()||(incoming>=8192&&short(c)&&{
            let take=reserve==0||spare>0;spare=spare.saturating_sub(1);take})).collect()
    }

    /// CANADA_SCRATCH_REDUNDANT_PROMPT_IMAGE (config::redundant_prompt_image): the prompt-end images with exact logits
    /// whose near-end checkpoint is cached as well (same owner, a proper prefix 1-127 tokens shorter, same media
    /// through it). Every rank holds the same entries, so every rank marks the same ones.
    fn redundant_prompt_images(&self)->Vec<bool> {
        self.cached_sessions.iter().enumerate().map(|(i,c)|self.redundant_prompt_image&&c.prompt_logits.is_some()
            &&self.cached_sessions.iter().enumerate().any(|(j,e)|j!=i&&e.owner==c.owner
                &&near_end_of(&c.prefix,&e.prefix)&&c.media.matches_prefix(&e.media,e.prefix.len()))).collect()
    }

    /// Retain only a committed target/drafter/history image. Prompt checkpoints
    /// and displaced conversations share the bounded entry/2GiB budget.
    /// With the checkpoint ladder, mark this owner's cached prefixes that the
    /// incoming checkpoint extends (same tokens and media up to their length).
    /// Every rank holds the same entries, so all ranks choose the same victims.
    /// See cache_cost_order(). Every rank holds the same entries, so the order is the same everywhere; nothing else holds
    /// an index into the cache across this call. `_incoming` is kept for the call sites (the order does not depend on it).
    fn order_cache_by_lineage(&mut self,_incoming:&[u32]) {
        if !self.cache_cost_order||self.cached_sessions.len()<2 {return;}
        let order=eviction_cost_order(&self.cached_sessions.iter().map(|c|c.prefix.as_slice()).collect::<Vec<_>>());
        if order.iter().enumerate().all(|(k,&i)|k==i) {return;}
        let mut slots=std::mem::take(&mut self.cached_sessions).into_iter().map(Some).collect::<Vec<_>>();
        self.cached_sessions=order.into_iter().map(|i|slots[i].take().unwrap()).collect();
    }

    fn ladder_ancestors(&self,prefix:&[u32],media:&crate::prompt_media::Identity)->Option<Vec<bool>> {
        self.checkpoint_ladder.then(||self.cached_sessions.iter().map(|c|c.owner==self.bank_id
            &&c.prefix.len()<prefix.len()&&prefix.starts_with(&c.prefix)
            &&media.matches_prefix(&c.media,c.prefix.len())).collect())
    }

    fn retain_cached_prefix(&mut self,prefix:&[u32],pending:u32,media:&crate::prompt_media::Identity)->Result<Value> {
        self.retain_cached_prefix_reserved(prefix,pending,media.through(prefix.len()),0,0,false)
    }

    /// See cache_capacity_of(). `extra_bytes`/`extra_count`: a restored parent held outside the vector that goes
    /// back into it (return_encoder_parent).
    fn cache_capacity(&self,extra_bytes:usize,extra_count:usize)->(usize,usize) {
        cache_capacity_of(self.session_cache_budget,self.shared_image_budget,self.suspended_image_bytes,
            self.suspended_image_count,self.image_entry_limit(),extra_bytes,extra_count)
    }

    // The configured cache slots remain available while at most two request images
    // temporarily coexist. The shared physical byte budget is unchanged.
    fn image_entry_limit(&self)->usize {
        self.session_cache_slots+if self.parking_extra_slots {self.suspended_image_count}else{0}
    }

    fn complete_cache_ancestor_bytes(&self)->usize {
        if crate::config::bank_encoder_checkpoint_requested() {
            self.image_entry_limit()*std::mem::size_of::<crate::target::BankEncoderWitness>()
        }else{0}
    }

    /// Memory-pressure victims are taken least recently used first (entries are ordered oldest first; a reused
    /// image is pushed to the end again). The "two longest stay" protection exists so that short unrelated
    /// requests cannot push out a long conversation. With CANADA_SCRATCH_DISPLACED_RECLAIM=1 it applies only to
    /// such short incoming images (under 8,192 tokens): for a long incoming image it kept stale long conversations
    /// and dropped the ones in use (Nov 13, four 250K-token conversations in rotation: `s1548 pmbl2`).
    fn pressure_long_first(&self,incoming:usize)->bool {
        self.session_cache_long_first&&!(self.displaced_reclaim&&incoming>=8192)
    }

    fn cache_reserve(&self)->Result<usize> {
        // Optional images must not spend the working room needed by the next
        // owner's prefill. Keep the legacy policy outside two-state exchange.
        let ordinary=crate::memory::floor_bytes()
            +if self.peer_bank.is_some()||request_checkpoint::exchange_enabled()?{256}else{16}*1024usize.pow(2);
        Ok(ordinary.max(self.cuda.allocation_reserve_bytes()))
    }

    /// Plan against the tightest physical rank before dropping any optional
    /// image, then recheck actual memory after release. The estimate never
    /// authorizes allocation; ordinary reserve and late-save checks still do.
    #[allow(clippy::too_many_arguments)]
    fn admit_cache_save(&mut self,eligible:bool,evictions:Option<Vec<usize>>,prefix:&[u32],
        media:&crate::prompt_media::Identity,required:usize,slots:usize,protected:&[usize],
        recycle_bytes:Option<usize>)->Result<CacheSaveAdmission> {
        let mut removed=if eligible {evictions.expect("eligible cache eviction plan")}else{Vec::new()};
        if crate::config::cache_superseded_first()&&!removed.is_empty() {
            let optional=self.optional_entries(prefix.len());
            removed=self.superseded_first(removed,prefix,protected,Some(&optional));
        }
        ensure!(protected.iter().all(|&i|i<self.cached_sessions.len()&&!removed.contains(&i)),
            "cache admission would evict a protected parent");
        let budget_evictions=removed.len();
        let mut reusable=Vec::new();
        let mut traced_available=None;
        if self.cache_pressure_reclaim {
            let available=self.cuda.image_admission_memory()?.admitted_available;
            traced_available=Some(available);
            // Ordinary saves need exactly the existing memory query and one
            // all-rank vote. Enter the fuller eviction agreement only when a
            // rank actually needs reclamation or a logical-budget eviction.
            if self.control.all_admit(eligible&&removed.is_empty()&&available>=required)? {
                return Ok(CacheSaveAdmission{admitted:true,available,local_fits:true,
                    evicted:0,pressure_evicted:0,forecast_refused:false,recycled:None});
            }
            let entries=self.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
            let optional=self.optional_entries(prefix.len());
            let superseded=self.cached_sessions.iter().map(|c|c.prefix.len()<prefix.len()
                &&prefix.starts_with(&c.prefix)&&media.matches_prefix(&c.media,c.prefix.len())).collect::<Vec<_>>();
            if let Some(bytes)=recycle_bytes {
                reusable=self.cached_sessions.iter().map(|c|
                    matches!(&c.image,CachedState::Encoder{image,..}if image.reusable_storage(bytes))).collect();
            }
            self.control.agree(&serde_json::to_vec(&(eligible,&entries,&optional,&superseded,&removed,
                protected,slots,prefix.len(),recycle_bytes,&reusable))?)?;
            loop {
                let reclaim=removed.iter().map(|&i|entries[i].1).sum::<usize>();
                if self.control.all_admit(eligible&&available.saturating_add(reclaim)>=required)?{break;}
                let excluded=removed.iter().chain(protected).copied().collect::<Vec<_>>();
                let next=eligible.then(||pressure_cache_victim(&entries,&optional,&superseded,
                    prefix.len(),slots,self.pressure_long_first(prefix.len()),&excluded)).flatten();
                let Some(next)=next else {
                    return Ok(CacheSaveAdmission{admitted:false,available,local_fits:available>=required,
                        evicted:0,pressure_evicted:0,forecast_refused:true,recycled:None});
                };
                removed.push(next);
            }
            // The pressure victims too (every rank ran the same loop over the same list).
            if removed.len()>budget_evictions {removed=self.superseded_first(removed,prefix,protected,Some(&optional));}
            if !removed.is_empty(){self.stream.synchronize()?;}
        }
        if !removed.is_empty() {
            let held=|i:usize|self.cached_sessions[i].trace();
            cache_trace(||json!({"path":"save","incoming_tokens":prefix.len(),"required":required,
                "available":traced_available,"slots":slots,"protected":protected,
                "budget_victims":removed[..budget_evictions].iter().map(|&i|held(i)).collect::<Vec<_>>(),
                "pressure_victims":removed[budget_evictions..].iter().map(|&i|held(i)).collect::<Vec<_>>(),
                "entries":(0..self.cached_sessions.len()).map(held).collect::<Vec<_>>()}));
        }
        removed.sort_unstable();
        let evicted=removed.len();
        // Eligibility was included in the existing metadata agreement. The
        // shared ordinary policy chooses the victims; reuse adds no exchange,
        // selects no extra victim, and keeps no old history as an uncharged spare.
        let reuse=removed.iter().copied().find(|&i|reusable.get(i)==Some(&true));
        let mut recycled=None;
        for i in removed.into_iter().rev() {
            let retired=self.cached_sessions.remove(i);
            if Some(i)==reuse {
                let CachedState::Encoder{image,..}=retired.image else{unreachable!()};
                recycled=Some(image.into_retired_storage()?);
            }
        }
        // Retain the normal prefill headroom while this old buffer is still
        // resident. Crediting its bytes here could crowd a lazy graph capture
        // during the prefix. If any rank cannot keep that headroom, drop it
        // everywhere and retry the unchanged ordinary post-eviction admission.
        let mut available=self.cuda.image_admission_memory()?.admitted_available;
        let mut local_fits=available>=required;
        let mut admitted=self.control.all_admit(eligible&&local_fits)?;
        if !admitted&&recycled.is_some() {
            recycled=None;available=self.cuda.image_admission_memory()?.admitted_available;
            local_fits=available>=required;admitted=self.control.all_admit(eligible&&local_fits)?;
        }
        Ok(CacheSaveAdmission{admitted,available,local_fits,evicted,
            pressure_evicted:evicted-budget_evictions,forecast_refused:false,recycled})
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
        // Keep the prefill preparation allowance after saving too.
        let fits=image.is_some() && self.cuda.admission_memory()?.admitted_available
            >=self.cache_reserve()?;
        if self.control.all_admit(fits)? {Ok(image)}else{Ok(None)}
    }

    fn try_save_cached_session_image(&mut self,anchor:u32,budget:usize)
        ->Result<Option<CachedCompleteImage>> {
        let Some(session)=self.try_save_session_image(anchor,budget)?else{return Ok(None);};
        let mut bank_ancestors=Vec::new();
        if crate::config::bank_encoder_checkpoint_requested() {
            bank_ancestors=Vec::with_capacity(self.image_entry_limit());
            ensure!(self.cached_sessions.len()<=bank_ancestors.capacity(),"ancestor metadata capacity");
            for cached in &self.cached_sessions {
                if cached.owner!=self.bank_id {continue;}
                if let CachedState::Encoder{image,..}=&cached.image {
                    if let Some(witness)=self.verify.bank_encoder_witness(image){bank_ancestors.push(witness);}
                }
            }
        }
        Ok(Some(CachedCompleteImage{session,bank_ancestors}))
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
            self.session_cache_long_first,headroom,false)?;
        self.agree(&[])?;
        event["type"]=json!("input_reclaim");event["request_id"]=json!(self.current);
        event["frontier"]=json!(self.history.frontier());event["suspended_image_bytes"]=json!(self.suspended_image_bytes);
        event["memory_before"]=json!(before);event["seconds"]=json!(began.elapsed().as_secs_f64());
        Ok(event)
    }

    /// A concurrent HTTP conversion can temporarily occupy another request's working room.
    /// Its buffers are independent of the model checkpoints. Only a rank under pressure waits.
    /// Keep this off the HTTP reclaim-command path, since that command can unblock a conversion.
    fn recover_request_headroom(&mut self, required: usize) -> Result<Value> {
        let before = self.cuda.admission_memory()?;
        let trimmed = before.admitted_available <= required && crate::memory::trim_free_host_pages();
        let mut available = self.cuda.admission_memory()?;
        let began = Instant::now();
        while available.admitted_available <= required && available.host_workspace_reserved > 0
            && began.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(20));
            available = self.cuda.admission_memory()?;
        }
        let wait_ms = began.elapsed().as_secs_f64() * 1000.;
        let mut event = if self.control.all_admit(available.admitted_available > required)? {
            json!({"evicted":[]})
        } else {
            self.reclaim_input_space(required)?
        };
        event["trimmed_free_host_pages"] = json!(trimmed);
        event["input_wait_ms"] = json!(wait_ms);
        event["before_input_recovery"] = json!(before);
        event["after_input_recovery"] = json!(self.cuda.admission_memory()?);
        Ok(event)
    }

    /// Admission at the committed request boundary, before resetting a live
    /// owner or appending any tokens. Optional caches may have accumulated, or
    /// host/driver memory may have grown since the preceding request. All ranks
    /// make the same reclamation/refusal decision; a tight rank cannot strand
    /// its peers inside prefill. No live or suspended request state is evicted.
    fn admit_prefill_input(&mut self)->Result<Option<Value>> {
        let required=self.cuda.allocation_reserve_bytes()+16*1024usize.pow(2);
        let fits=self.cuda.admission_memory()?.admitted_available>required;
        #[cfg(feature = "storage-fault-injection")]
        let fits=!self.cache_save_fault.armed_for(cache_admission_fault::Point::PrefillAfterReclaim,self.comm.rank())
            &&self.cache_save_fault.apply(cache_admission_fault::Point::PrefillBeforeReclaim,self.comm.rank(),fits);
        if self.control.all_admit(fits)? {return Ok(None);}
        // Return allocator-free host pages before sacrificing reusable model
        // images. This is pressure-only work, never a decode-loop operation.
        let reclaimed=self.recover_request_headroom(required)?;
        let fits=self.cuda.admission_memory()?.admitted_available>required;
        #[cfg(feature = "storage-fault-injection")]
        let fits=self.cache_save_fault.apply(cache_admission_fault::Point::PrefillAfterReclaim,self.comm.rank(),fits);
        ensure!(self.control.all_admit(fits)?,crate::memory::InputCapacityError(
            "prefill cannot retain the UMA reserve after optional cache reclamation",required));
        Ok(Some(reclaimed))
    }

    #[allow(clippy::too_many_arguments)]
    /// CANADA_SCRATCH_DISPLACED_CHECKPOINT=1: when an unrelated conversation is about to reuse
    /// this bank, keep the displaced conversation's newest bank-retained encoder checkpoint as a
    /// standalone image. Its next turn re-renders history differently after the prompt (thinking
    /// is not resent), so only such a checkpoint can serve it; a bank-retained one dies with the
    /// bank. Budget, entry and memory admission are those of an ordinary cache save, agreed by all
    /// ranks; a refusal leaves the ordinary behavior. No arithmetic changes.
    fn materialize_displaced_checkpoint(&mut self,incoming:Option<(&[u32], &crate::prompt_media::Identity)>,current:&[u32],wanted:Option<&CachedSession>)->Result<Value> {
        if !self.displaced_checkpoint_enabled||self.session.is_none()||current.is_empty() {return Ok(Value::Null);}
        let continued = |entry: &CachedSession| incoming.is_some_and(|(tokens, media)|
            tokens.starts_with(&entry.prefix) && media.matches_prefix(&entry.media, entry.prefix.len()));
        let live=|worker:&Self,c:&CachedSession|c.owner==worker.bank_id&&current.starts_with(&c.prefix)
            &&worker.history.media().matches_prefix(&c.media,c.prefix.len())
            &&matches!(&c.image,CachedState::Encoder{image,..}if image.bank_retained()&&worker.verify.encoder_prefix_reusable_from(image));
        let found=self.cached_sessions.iter().enumerate().filter(|(_,c)|live(self,c))
            .max_by_key(|(_,c)|c.prefix.len()).map(|(i,_)|i);
        // Shared-prefix images serve other conversations: every sparse one on this bank that the incoming prompt
        // does not continue is kept too (the newest image alone is the displaced conversation's own).
        let shared=(0..self.cached_sessions.len()).filter(|&i|Some(i)!=found&&self.cached_sessions[i].shared
            &&live(self,&self.cached_sessions[i])&&!continued(&self.cached_sessions[i])).collect::<Vec<_>>();
        // Entries that give way for one copy (CANADA_SCRATCH_DISPLACED_RECLAIM) are removed from the vector, so
        // the indices still to be copied are kept in step.
        let mut shared=shared;
        let mut report=match found {
            None=>json!({"materialized":false,"candidate":false}),
            // The incoming request continues this checkpoint's lineage; the bank keeps serving it.
            Some(index) if continued(&self.cached_sessions[index])=>
                json!({"materialized":false,"candidate":true,"continued":true}),
            Some(index)=>{
                let (report,removed)=self.materialize_bank_image(index,wanted,&shared)?;
                for i in shared.iter_mut() {*i-=removed.iter().filter(|&&r|r<*i).count();}
                report
            },
        };
        if !shared.is_empty() {
            let mut reports=Vec::new();
            for at in 0..shared.len() {
                let keep=shared.clone();
                let (one,removed)=self.materialize_bank_image(shared[at],wanted,&keep)?;
                for i in shared.iter_mut() {*i-=removed.iter().filter(|&&r|r<*i).count();}
                reports.push(one);
            }
            report["shared"]=Value::Array(reports);
        }
        Ok(report)
    }
    /// Replace one sparse (bank-retained) cached image by a standalone copy, under the cache budget and all-rank
    /// memory admission. Indices stay as they are: the entry grows in place.
    /// `keep`: other entries that must stay (images about to be copied themselves). Returns the report and the
    /// indices (before removal, ascending) of the older entries that gave way for this copy.
    fn materialize_bank_image(&mut self,index:usize,wanted:Option<&CachedSession>,keep:&[usize])->Result<(Value,Vec<usize>)> {
        let started=Instant::now();
        let mut index=index;
        let CachedState::Encoder{image,..}=&self.cached_sessions[index].image else {unreachable!()};
        let full=self.verify.materialized_encoder_bytes(image)?;
        let delta=full.saturating_sub(image.bytes());
        // A wanted parent held outside the vector is reinserted later under exactly this budget
        // (return_encoder_parent) and is part of the transient shared-budget check; the copy
        // grows this entry in place, so all three must still fit together.
        let (budget,_)=self.cache_capacity(0,0);
        let charged=self.cached_sessions.iter().map(|c|c.charge).sum::<usize>()
            .saturating_add(wanted.map_or(0,|c|c.charge));
        let reserve=self.cache_reserve()?;
        let available=self.cuda.admission_memory()?.admitted_available;
        let pooled=self.cuda.image_pool_fits(full);
        let fits=full<=2*1024usize.pow(3)&&charged.saturating_add(delta)<=budget
            &&crate::state_image::image_storage_fits(available,reserve,full,pooled);
        let mut removed:Vec<usize>=Vec::new();
        if !self.control.all_admit(fits)? {
            cache_trace(||json!({"path":"displaced_admission","available":available,"reserve":reserve,
                "bytes":full,"pool_free":self.cuda.image_pool_free_bytes(),"pool_fits":pooled,"budget":budget,"charged":charged}));
            let refused=|evicted:usize|json!({"materialized":false,"candidate":true,"refused":true,"bytes":full,
                "charged":charged,"budget":budget,"gave_way":evicted});
            if !(self.displaced_reclaim&&self.cache_pressure_reclaim&&full<=2*1024usize.pow(3)) {return Ok((refused(0),removed));}
            // A pooled image releases pool pages. Its logical charge does not become free
            // host memory. Only count actual backing for the proposed destination.
            let alternatives = CheckpointAlternatives::new(&self.cached_sessions);
            let tokens = alternatives.lengths[index];
            let mut protected = keep.to_vec();
            protected.push(index);
            let mut gone = vec![false; self.cached_sessions.len()];
            let mut plan = Vec::new();
            let mut projected_charge = charged;
            let mut projected_host = available;
            let mut projected_pool = self.cuda.image_pool_free_bytes().unwrap_or(0);
            let chunk_bytes = self.cuda.image_pool_chunk_bytes()?;
            loop {
                let local = projected_charge.saturating_add(delta) <= budget
                    && displaced_storage_fits(projected_host, reserve, full, projected_pool);
                if self.control.all_admit(local)? { break; }
                let victim = alternatives.victim(&gone, &protected);
                self.control.agree(&serde_json::to_vec(&("displaced-reclaim-v2", index, victim, &protected))?)?;
                let Some((i, lost)) = victim else { return Ok((refused(0), Vec::new())); };
                let entry = &self.cached_sessions[i];
                gone[i] = true;
                projected_charge = projected_charge.saturating_sub(entry.charge);
                for image in entry.images() {
                    let chunks = image.owned_pool_chunks();
                    if chunks > 0 { projected_pool += chunks * chunk_bytes; }
                    else if image.resident() { projected_host += image.bytes(); }
                }
                plan.push((i, lost));
                removed.push(i);
            }
            if !self.preserve_forward_victims(&plan, &alternatives)? { return Ok((refused(0), Vec::new())); }
            removed.sort_unstable();
            if !removed.is_empty() {
                cache_trace(||json!({"path":"displaced_reclaim","incoming_tokens":tokens,"available":available,
                    "victims":removed.iter().map(|&i|self.cached_sessions[i].trace()).collect::<Vec<_>>(),
                    "entries":self.cached_sessions.iter().map(|c|c.trace()).collect::<Vec<_>>()}));
            }
            self.stream.synchronize()?;
            for &i in removed.iter().rev() {self.evict_cached_saving(i)?;}
            index-=removed.iter().filter(|&&r|r<index).count();
            // The estimate never authorizes the allocation: recheck actual memory after the release.
            let available=self.cuda.admission_memory()?.admitted_available;
            let fits=crate::state_image::image_storage_fits(available,reserve,full,self.cuda.image_pool_fits(full));
            if !self.control.all_admit(fits)? {return Ok((refused(removed.len()),removed));}
        }
        let CachedState::Encoder{image,..}=&self.cached_sessions[index].image else {unreachable!()};
        // The copy writes only the new image; any refusal or mismatch leaves live state untouched.
        let made=match self.verify.try_materialize_bank_encoder(image,full,&self.comm,&self.stream) {
            Ok(made)=>made,
            Err(error)=>{eprintln!("displaced checkpoint not materialized: {error:#}");None}
        };
        if !self.control.all_admit(made.is_some())? {
            return Ok((json!({"materialized":false,"candidate":true,"late_refused":true,"bytes":full,"gave_way":removed.len()}),removed));
        }
        let entry=&mut self.cached_sessions[index];
        let tokens=entry.prefix.len();
        if let CachedState::Encoder{image,..}=&mut entry.image {*image=made.unwrap();}
        entry.charge+=delta;
        Ok((json!({"materialized":true,"tokens":tokens,"bytes":full,"gave_way":removed.len(),
            "seconds":started.elapsed().as_secs_f64()}),removed))
    }
    fn retain_cached_prefix_reserved(&mut self,prefix:&[u32],pending:u32,media:crate::prompt_media::Identity,
        extra_bytes:usize,extra_count:usize,survivors_only:bool)->Result<Value> {
        ensure!(!prefix.is_empty() && self.verify.position()==prefix.len()as u64,
            "cached prefix and target frontier disagree");
        if self.cached_sessions.iter().any(|c|c.owner==self.bank_id && c.prefix==prefix && c.media.matches_prefix(&media,prefix.len())
            && keeps_prefix(matches!(&c.image,CachedState::Encoder{image,..}if image.bank_retained()),survivors_only)) {
            return Ok(json!({"saved":false,"already_cached":true,"saved_seconds":0.,
                "evicted_sessions":0,"capacity_refused":false}));
        }
        let (device_bytes,host_bytes)=self.session.as_ref().unwrap().compact_state_image_bytes(&self.verify)?;
        let charge=device_bytes.checked_add(host_bytes).and_then(|n|n.checked_add(prefix.len()*4+media.metadata_bytes()))
            .and_then(|n|n.checked_add(self.complete_cache_ancestor_bytes()))
            .ok_or_else(||anyhow::anyhow!("cached session byte overflow"))?;
        self.order_cache_by_lineage(prefix);
        let entries=self.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
        let (budget,slots)=self.cache_capacity(extra_bytes,extra_count);
        let optional=self.optional_entries(prefix.len());
        let ancestors=self.cached_sessions.iter().map(|c|self.session_cache_shared_ancestors
            &&c.prefix.len()>=8192&&c.prefix.len()<prefix.len()&&prefix.starts_with(&c.prefix)
            &&media.matches_prefix(&c.media,c.prefix.len())).collect::<Vec<_>>();
        let protected=ancestors.iter().enumerate().filter_map(|(i,&yes)|yes.then_some(i)).collect::<Vec<_>>();
        let evictions=completed_cache_evictions(&entries,&optional,&ancestors,(prefix.len(),charge),
            budget,self.session_cache_long_first,slots);
        let eligible=device_bytes<=2*1024usize.pow(3) && evictions.is_some();
        // The wanted image remains alive during this check, so temporarily
        // retaining both current and wanted sessions is charged to real UMA.
        let reserve=self.cache_reserve()?;
        let admission=self.admit_cache_save(eligible,evictions,prefix,&media,reserve+charge,slots,&protected,None)?;
        let admitted=admission.admitted;
        let mut saved_seconds=0.;
        let mut saved=false;
        if admitted {
            let began=Instant::now();
            if let Some(image)=self.try_save_cached_session_image(pending,device_bytes)? {
                ensure!(image.device_bytes()==device_bytes,"cached image estimate differs from allocation");
                self.cached_sessions.push(CachedSession{owner:self.bank_id,prefix:prefix.to_vec().into(),media,image:CachedState::Complete{image,pending},charge,prompt_logits:None,kind:"completed",born:Instant::now(),used:Instant::now(),serial:self.cache_serial,restored_serial:0,speculative:false,shared:false});
                saved=true;
            }
            saved_seconds=began.elapsed().as_secs_f64();
        }
        Ok(json!({"saved":saved,"late_memory_refused":admitted&&!saved,"saved_seconds":saved_seconds,
            "evicted_sessions":admission.evicted,"capacity_refused":!eligible,
            "memory_refused":eligible&&!admitted,"local_memory_refused":eligible&&!admission.local_fits,
            "pressure_evicted_sessions":admission.pressure_evicted,"forecast_refused":admission.forecast_refused,
            "protected_ancestors":protected.len(),
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
            || (plan.prompt.len()<8192 && plan.media.placements().is_empty()
                && !sampler_law_probe::enabled()) || (plan.reuse && !cache_hit) {
            return Ok((None,json!({"enabled":self.prompt_cache_logits,"saved":false,
                "active_append_skipped":plan.reuse && !cache_hit,"seconds":0.})));
        }
        if self.cached_sessions.iter().any(|c|c.owner==self.bank_id && c.prefix==plan.prompt && c.media.matches_prefix(&plan.media,plan.prompt.len()) && c.prompt_logits.is_some()) {
            return Ok((None,json!({"enabled":true,"saved":false,"already_cached":true,"seconds":0.})));
        }
        let (device,host)=self.session.as_ref().unwrap().compact_state_image_bytes(&self.large)?;
        let charge=device.checked_add(host).and_then(|n|n.checked_add(plan.prompt.len()*4+plan.media.metadata_bytes()))
            .and_then(|n|n.checked_add(self.complete_cache_ancestor_bytes()))
            .and_then(|n|n.checked_add(crate::sampling::VOCAB*4))
            .ok_or_else(||anyhow::anyhow!("prompt logit cache size overflow"))?;
        self.order_cache_by_lineage(&plan.prompt);
        let (budget,slots)=self.cache_capacity(0,0);
        let entries=self.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
        // Speculative entries yield first; with CANADA_SCRATCH_REDUNDANT_PROMPT_IMAGE so do prompt-end images whose
        // near-end checkpoint is cached (redundant_prompt_images; all false without the flag).
        let speculative=self.cached_sessions.iter().zip(self.redundant_prompt_images())
            .map(|(c,r)|c.speculative||r).collect::<Vec<_>>();
        let evictions=if speculative.iter().any(|&v|v) {
            cache_evictions_with_optional(&entries,&speculative,(plan.prompt.len(),charge),budget,
                self.session_cache_long_first,slots).filter(|removed|
                    !plan.media.placements().is_empty()||removed.iter().all(|&i|speculative[i]))
        }else{prompt_cache_evictions(&entries,(plan.prompt.len(),charge),budget,
            self.session_cache_long_first,slots,!plan.media.placements().is_empty())};
        let evictions=if plan.media.placements().is_empty() {evictions}
            else{evictions.map(|victims|self.superseded_first(victims,&plan.prompt,&[],Some(&speculative)))};
        let eligible=evictions.is_some();
        let reserve=self.cache_reserve()?;
        let admitted=self.control.all_admit(eligible
            && self.cuda.image_admission_memory()?.admitted_available>=reserve+charge)?;
        // Admission is conservative while all old allocations still exist.
        // A refused allocation leaves their cache ownership untouched.
        let mut evicted=0;
        let ready=if admitted {
            let mut indices=evictions.unwrap();indices.sort_unstable();evicted=indices.len();
            if !indices.is_empty() {
                cache_trace(||json!({"path":"prompt_logits_save","incoming_tokens":plan.prompt.len(),"incoming_charge":charge,
                    "budget":budget,"slots":slots,
                    "victims":indices.iter().map(|&i|self.cached_sessions[i].trace()).collect::<Vec<_>>(),
                    "entries":self.cached_sessions.iter().map(|c|c.trace()).collect::<Vec<_>>()}));
            }
            for i in indices.into_iter().rev(){self.evict_cached_saving(i)?;}
            Some((self.large.snapshot_prompt_logits(&self.stream)?,device,charge))
        }else{None};
        Ok((ready,json!({"enabled":true,"saved":false,"admitted":admitted,
            "capacity_refused":!eligible,"evicted_sessions":evicted,"seconds":started.elapsed().as_secs_f64()})))
    }

    /// Retain a displaced conversation and restore a matching parent prefix.
    /// Active exact-prefix reuse takes the existing allocation-free path.
    /// A bounded number of inactive images share one explicit memory budget; every
    /// optional allocation needs all-rank admission above the8GiB reserve.
    fn reuse_cached_session(&mut self,plan:&mut RequestPlan,allow_exact_logits:bool,allow_encoder:bool)->Result<(Value,Option<CachedSession>)> {
        let started=Instant::now();
        self.prune_invalid_bank_images();
        let imports=allow_encoder&&crate::config::encoder_prefix_import_requested()&&self.peer_bank.is_some();
        // Batched decode: after a conversation moves to its region, region 0 is reset with a drafter
        // but no resident history; the newcomer still restores its cached image there.
        let fresh_region0=self.session.is_some()&&plan.previous==0&&batched_serving::enabled()?;
        if self.session_cache_budget==0 || plan.reuse || (!imports&&(self.session.is_none() || (plan.previous==0&&!fresh_region0))) {
            return Ok((json!({"enabled":self.session_cache_budget>0,"hit":false,"saved":false,
                "active_reuse":plan.reuse,"seconds":started.elapsed().as_secs_f64()}),None));
        }
        ensure!(if self.session.is_some(){self.verify.profile_is_current() && self.verify.position()==plan.previous as u64}
            else{plan.previous==0&&self.large.position()==0&&self.large.profile_is_current()},
            "cache lookup requires the current accepted owner");
        // Disk tier: a parked entry that beats every resident one is reloaded (all ranks) and becomes the newest entry.
        if self.disk_tier.is_some() && !self.parked_sessions.is_empty() {
            let usable=|c:&CachedSession| c.owner==self.bank_id
                && (allow_encoder||matches!(&c.image,CachedState::Complete{..})) && plan.prompt.starts_with(&c.prefix)
                && plan.media.matches_prefix(&c.media,c.prefix.len())
                && (c.prefix.len()<plan.prompt.len() || (self.prompt_cache_logits && allow_exact_logits
                    && c.prefix.len()==plan.prompt.len() && c.prompt_logits.is_some()));
            let resident_best=self.cached_sessions.iter().filter(|c|usable(c)).map(|c|c.prefix.len()).max();
            let parked_best=self.parked_sessions.iter().enumerate().filter(|(_,c)|usable(c))
                .max_by_key(|(_,c)|c.prefix.len()).map(|(j,c)|(j,c.prefix.len()));
            if let Some((j,len))=parked_best && resident_best.is_none_or(|r|len>r) {
                // Only when the read beats prefilling the rows it saves (disk_tier::reload_pays); every rank agrees.
                let gain=len-resident_best.unwrap_or(0);
                let bytes=self.parked_sessions[j].logical_bytes();
                let (rate,fixed)=self.disk_tier.as_ref().map_or((1.0,0.0),|t|(t.read_bytes_per_s,t.read_fixed_s));
                let pays=crate::disk_tier::reload_pays(bytes,gain,len==plan.prompt.len(),rate,fixed,crate::config::encoder_chunk_costs());
                if self.control.all_admit(pays)? {self.reload_parked(j)?;}
                else {eprintln!("disk tier {}",json!({"event":"reload_skipped","tokens":len,"gain":gain,"bytes":bytes}));}
            }
        }
        let hit=self.cached_sessions.iter().enumerate()
            .filter(|(_,c)|(c.owner==self.bank_id||(imports&&self.can_import_cached_encoder(c)))
                && (allow_encoder||matches!(&c.image,CachedState::Complete{..})) && plan.prompt.starts_with(&c.prefix)
                && plan.media.matches_prefix(&c.media,c.prefix.len())
                && (c.prefix.len()<plan.prompt.len() || (self.prompt_cache_logits && allow_exact_logits
                    && c.prefix.len()==plan.prompt.len() && c.prompt_logits.is_some())))
            .max_by_key(|(_,c)|c.prefix.len()).map(|(i,_)|i);
        // Which entry a request checks out (it is held outside the vector until it returns or is dropped), and what
        // else the cache holds at that moment.
        cache_trace(||json!({"path":"restore_lookup","prompt_tokens":plan.prompt.len(),
            "hit":hit.map(|i|self.cached_sessions[i].trace()),
            "entries":self.cached_sessions.iter().map(|c|c.trace()).collect::<Vec<_>>(),
            "parked":self.parked_sessions.iter().map(|c|c.trace()).collect::<Vec<_>>()}));
        let wanted=hit.map(|i|{let mut cached=self.cached_sessions.remove(i);cached.speculative=false;cached.used=Instant::now();
            cached.restored_serial=self.cache_serial;cached});
        let current_prefix=self.history.committed().to_vec();
        // A displaced unfinished request already owns an immutable image.
        // Do not copy that same loaded prefix into the optional cache too.
        // A wanted cache image remains charged while held outside the vector.
        let saved=if self.session.is_none()||current_prefix.is_empty() {
            json!({"saved":false,"saved_seconds":0.,"evicted_sessions":0,"capacity_refused":false})
        }else if self.loaded_request_is_suspended {
            json!({"saved":false,"saved_seconds":0.,"evicted_sessions":0,
                "capacity_refused":false,"already_suspended":true})
        }else if (self.skip_replaced_completion && plan.replaces_completed_output)
            || (plan.replaces_completed_turn && crate::config::skip_replaced_turn()) {
            // The incoming authoritative prompt keeps the prior request's
            // input, but not its complete output. Preserve the input anchor
            // and avoid copying that displaced child on this forward turn.
            // Unrelated conversations, unfinished owners and true appends
            // retain their existing paths. GPU reuse still needs exact IDs.
            json!({"saved":false,"saved_seconds":0.,"evicted_sessions":0,
                "capacity_refused":false,"replaced_completion_skipped":true})
        }else{
            // With the displaced snapshot, a conversation displaced by another
            // one keeps a complete image even when its only entry for this
            // prefix is a bank-retained anchor that the newcomer's prefill
            // invalidates. A branch restored from this bank's own lineage
            // (an edit) keeps the ordinary policy.
            let branch=wanted.as_ref().is_some_and(|c|c.owner==self.bank_id
                &&matches!(&c.image,CachedState::Encoder{image,..}if image.bank_retained()));
            self.retain_cached_prefix_reserved(&current_prefix,self.history.pending().context("cached session lacks pending anchor")?,self.history.media().through(current_prefix.len()),
                wanted.as_ref().map_or(0,|c|c.charge),usize::from(wanted.is_some()),self.displaced_snapshot&&!branch)?
        };
        // Before the incoming request touches the bank: keep the displaced conversation's newest
        // bank-retained checkpoint as a standalone image (CANADA_SCRATCH_DISPLACED_CHECKPOINT).
        let displaced=self.materialize_displaced_checkpoint(Some((&plan.prompt,&plan.media)),&current_prefix,wanted.as_ref())?;
        let transient_image_bytes=self.suspended_image_bytes
            +self.cached_sessions.iter().map(|c|c.charge).sum::<usize>()
            +wanted.as_ref().map_or(0,|c|c.charge);
        let transient_image_count=self.suspended_image_count+self.cached_sessions.len()+usize::from(wanted.is_some());
        ensure!(transient_image_bytes<=self.shared_image_budget && transient_image_count<=self.image_entry_limit(),
            "restoring cache image exceeded the shared inactive budget");
        let reserve=self.cache_reserve()?;
        let mut restored_tokens=0;let mut restore_seconds=0.;let mut restore_refused=false;
        let mut retained_image=false;let mut retention_evictions=0;
        let mut reconstructed_upgrade=false;let mut upgrade_refused=false;let mut upgrade_seconds=0.;
        let mut upgrade_capacity_refused=false;let mut upgrade_memory_refused=false;
        let mut upgrade_late_memory_refused=false;let mut upgrade_evicted_sessions=0;
        let mut encoder_reconstructed=false;let mut imported_owner=None;
        let mut exact_prompt_anchor=None;let mut sample_seconds=0.;
        if let Some(cached)=wanted {
            let imported=cached.owner!=self.bank_id;
            if imported {
                ensure!(imports&&self.can_import_cached_encoder(&cached),"cached image belongs to an inadmissible physical owner");
                self.peer_bank.as_ref().unwrap().encoder_import_source(&cached)?;
                imported_owner=Some(cached.owner);
            }
            let extra_history=match &cached.image {
                CachedState::Encoder{history,..}=>history.as_ref().map_or(0,crate::history_lookup::HistoryLookup::reserved_workspace_bytes),
                CachedState::Complete{image,..}=>image.session.restore_host_workspace_bytes(),
            };
            let mut admitted=self.control.all_admit(self.cuda.admission_memory()?.admitted_available>=reserve+extra_history)?;
            if !admitted {
                let recovery=self.recover_request_headroom(reserve+extra_history)?;
                admitted=self.control.all_admit(self.cuda.admission_memory()?.admitted_available>=reserve+extra_history)?;
                cache_trace(||json!({"path":"restore_admission_retry","restoring_tokens":cached.prefix.len(),
                    "required":reserve+extra_history,"recovery":recovery,"admitted":admitted}));
            }
            if admitted {
                let prior=if imported {let CachedState::Encoder{image,..}=&cached.image else{unreachable!()};
                    self.target_lineage.max(image.lineage())}else{self.target_lineage};
                self.target_lineage=prior.checked_add(1)
                    .ok_or_else(||anyhow::anyhow!("cached target lineage exhausted"))?;
                let began=Instant::now();
                // CANADA_SCRATCH_ENCODER_SHORT_SUFFIX: a shorter suffix also takes the compound path, as one
                // ragged chunk replaying the image's retained encoder ring (no reconstruction pass).
                let suffix_rows=plan.prompt.len()-cached.prefix.len();
                if matches!(&cached.image,CachedState::Encoder{..}) && (suffix_rows>=128
                    || (suffix_rows>=1&&cached.prefix.len()>=128&&crate::config::encoder_short_suffix()?)) {
                    // Transfer the charged parent to the compound operation.
                    // It keeps the parent alive through optional successor
                    // allocation; no incomplete decoder owner escapes.
                    let mut report=json!({"enabled":true,"hit":true,"encoder_deferred":true,"imported_owner":imported_owner,
                        "restored_tokens":cached.prefix.len(),"saved":saved["saved"],
                        "saved_seconds":saved["saved_seconds"],"evicted_sessions":saved["evicted_sessions"],
                        "replaced_completion_skipped":saved["replaced_completion_skipped"],
                        "capacity_refused":saved["capacity_refused"],"transient_image_bytes":transient_image_bytes,
                        "transient_image_count":transient_image_count,"seconds":started.elapsed().as_secs_f64()});
                    if !displaced.is_null() {report["displaced_checkpoint"]=displaced;}
                    return Ok((report,Some(cached)));
                }
                let pending=match &cached.image {
                    CachedState::Complete{image,pending}=>{
                        ensure!(!imported,"complete sessions cannot be imported");
                        let restored=self.session.as_mut().context("complete cache needs its prepared owner")?.restore_state_image(&mut self.verify,&image.session,
                            self.target_lineage,&self.comm,&self.stream)?;
                        self.verify.rebind_encoder_ancestors(&image.bank_ancestors)?;
                        ensure!(restored==*pending,"cached anchor differs from session image");restored
                    },
                    CachedState::Encoder{image,history,policy}=>{
                        encoder_reconstructed=true;
                        ensure!(self.encoder_prompt_checkpoint,"encoder cache disabled");
                        let bank_ancestors=self.bank_ancestors_before_restore(&cached)?;
                        if self.session.is_some(){self.large.handoff_from(&self.verify)?;}
                        if imported {
                            let source=self.peer_bank.as_ref().unwrap().encoder_import_source(&cached)?;
                            self.large.import_encoder_checkpoint_full(image,source,self.target_lineage,&self.comm,&self.stream)?;
                        }else{self.large.restore_encoder_checkpoint_full(image,self.target_lineage,&self.comm,&self.stream)?;}
                        self.large.rebind_encoder_ancestors(&bank_ancestors)?;
                        let index=history.as_ref().map(crate::history_lookup::HistoryLookup::clone_reserved);
                        self.attach_prompt_with_index(false,0,&cached.prefix,&cached.prefix,plan.seed,
                            !allow_exact_logits,Some(index))?;
                        let session=self.session.as_mut().unwrap();
                        if !imported {session.restore_proposal_policy(policy,&self.stream)?;}
                        if allow_exact_logits {session.configure_sampling(plan.seed,Some(self.cfg.eos_token_id))?;}
                        else{session.configure_greedy(Some(self.cfg.eos_token_id))?;}
                        let pending=self.large.greedy_token(&self.stream)?;
                        self.verify.handoff_from(&self.large)?;pending
                    },
                };
                self.prune_invalid_bank_images();
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
                if self.retain_restored_checkpoint && (imported||((cached.prefix.len()>=crate::config::prompt_checkpoint_min()
                    || sampler_law_probe::enabled()
                    || (!cached.media.placements().is_empty() && cached.prompt_logits.is_some()))
                    && checkpoint<=cached.prefix.len())) {
                    // A short suffix forced reconstruction of a complete
                    // encoder-only parent. Save that already valid state once
                    // so another short branch does not repeat the decoder pass.
                    // A bank tail deliberately avoids duplicating the whole
                    // compressed prefix. Preserve it on short restores rather
                    // than replacing it with a near-1M complete snapshot.
                    let upgrade=!imported&&self.encoder_upgrade_reconstructed
                        &&matches!(&cached.image,CachedState::Encoder{image,..}if !image.bank_retained());
                    let mut candidate=cached;
                    if upgrade {
                        let at=Instant::now();
                        let (device,host)=self.session.as_ref().unwrap().compact_state_image_bytes(&self.verify)?;
                        let charge=device.checked_add(host).and_then(|n|n.checked_add(candidate.prefix.len()*4+candidate.media.metadata_bytes()))
                            .and_then(|n|n.checked_add(self.complete_cache_ancestor_bytes()))
                            .ok_or_else(||anyhow::anyhow!("reconstructed cache charge overflow"))?;
                        let entries=self.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
                        let optional=self.optional_entries(candidate.prefix.len());
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
                            if !indices.is_empty() {
                                cache_trace(||json!({"path":"restored_image_upgrade","incoming_tokens":candidate.prefix.len(),
                                    "incoming_charge":charge,"budget":budget,"slots":slots,
                                    "victims":indices.iter().map(|&i|json!([entries[i].0,entries[i].1])).collect::<Vec<_>>(),
                                    "entries":entries.iter().map(|e|json!([e.0,e.1])).collect::<Vec<_>>()}));
                            }
                            for i in indices.into_iter().rev(){self.evict_cached_saving(i)?;}
                            self.stream.synchronize()?;
                            let image=self.try_save_cached_session_image(pending,device)?;
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
                    if upgrade||imported {
                        // Both refusal and success still own the reservation;
                        // reinsertion cannot silently discard this parent.
                        retained_image=self.return_encoder_parent(Some(candidate))?;
                    }else{
                        let cached=candidate;
                        self.order_cache_by_lineage(&cached.prefix);
                        let entries=self.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
                        let (budget,slots)=self.cache_capacity(0,0);
                        let ancestors=self.ladder_ancestors(&cached.prefix,&cached.media);
                        if let Some(mut evictions)=cache_evictions_ladder(&entries,ancestors.as_deref(),
                            (cached.prefix.len(),cached.charge),budget,self.session_cache_long_first,slots) {
                            evictions.sort_unstable();retention_evictions=evictions.len();
                            if !evictions.is_empty() {
                                cache_trace(||json!({"path":"restored_image_retention","incoming_tokens":cached.prefix.len(),
                                    "incoming_charge":cached.charge,"budget":budget,"slots":slots,
                                    "victims":evictions.iter().map(|&i|json!([entries[i].0,entries[i].1])).collect::<Vec<_>>(),
                                    "entries":entries.iter().map(|e|json!([e.0,e.1])).collect::<Vec<_>>()}));
                            }
                            for i in evictions.into_iter().rev(){self.evict_cached_saving(i)?;}
                            self.cached_sessions.push(cached);
                            retained_image=true;
                        }
                    }
                }
            }else{
                restore_refused=true;
                // The checked-out checkpoint still owns its reserved slot and bytes. A
                // transient refusal has made no changes to its contents or its bank.
                self.cached_sessions.push(cached);
                retained_image=true;
            }
        }
        Ok((json!({"enabled":true,"hit":restored_tokens>0,"restored_tokens":restored_tokens,
            "exact_prompt_anchor":exact_prompt_anchor,"sample_seconds":sample_seconds,
            "policy":if self.session_cache_long_first {"long_parents_and_recent"}else{"fifo"},
            "evicted_sessions":saved["evicted_sessions"],"capacity_refused":saved["capacity_refused"],
            "saved":saved["saved"],"saved_seconds":saved["saved_seconds"],"restore_seconds":restore_seconds,"restore_refused":restore_refused,
            "already_suspended":saved["already_suspended"],
            "replaced_completion_skipped":saved["replaced_completion_skipped"],
            "transient_image_bytes":transient_image_bytes,"transient_image_count":transient_image_count,
            "retained_image":retained_image,"retention_evictions":retention_evictions,
            "reconstructed_upgrade":reconstructed_upgrade,"upgrade_refused":upgrade_refused,"upgrade_seconds":upgrade_seconds,
            "upgrade_capacity_refused":upgrade_capacity_refused,"upgrade_memory_refused":upgrade_memory_refused,
            "upgrade_late_memory_refused":upgrade_late_memory_refused,"upgrade_evicted_sessions":upgrade_evicted_sessions,
            "upgrade_parent_retained_on_refusal":upgrade_refused&&retained_image,
            "encoder_reconstructed":encoder_reconstructed,"imported_owner":imported_owner,
            "inactive_sessions":self.cached_sessions.len(),"charged_bytes":self.cached_sessions.iter().map(|c|c.charge).sum::<usize>(),
            "budget_bytes":self.session_cache_budget,"seconds":started.elapsed().as_secs_f64()}),None))
            .map(|(mut report,none):(Value,Option<CachedSession>)|{if !displaced.is_null(){report["displaced_checkpoint"]=displaced;}(report,none)})
    }

    #[allow(clippy::too_many_arguments)]
    fn attach_prompt(&mut self,reuse:bool,previous:usize,ids:&[u32],prefix:&[u32],
        seed:u64,greedy:bool)->Result<()> {
        self.attach_prompt_with_index(reuse,previous,ids,prefix,seed,greedy,None)
    }
    #[allow(clippy::too_many_arguments)]
    /// The drafter after the qualified EXTEND that followed an encoder-image restore
    /// (CANADA_SCRATCH_ENCODER_SHORT_SUFFIX=2): its window comes from the image's tail rows and the EXTEND's.
    fn attach_extend_with_tail(&mut self,plan:&RequestPlan,greedy:bool,
        index:Option<Option<crate::history_lookup::HistoryLookup>>,tail:&DraftTail)->Result<()> {
        self.large.trace_prefill_boundary_for_probe("before_draft_attach")?;
        let session=self.session.as_mut().ok_or_else(||anyhow::anyhow!("EXTEND after restore needs an attached drafter"))?;
        session.reset_from_extend_with_tail(&self.large,&tail.values,tail.end,&self.stream)?;
        if greedy {session.configure_greedy(Some(self.cfg.eos_token_id))?;}
        else {session.configure_sampling(plan.seed,Some(self.cfg.eos_token_id))?;}
        if let Some(index)=index {
            ensure!(index.is_some()==self.history_lookup_enabled,"cached history policy mismatch");
            self.session.as_mut().unwrap().install_history_lookup(&self.large,index)?;
        }else if self.history_lookup_enabled {
            self.session.as_mut().unwrap().configure_history_lookup(&self.large,&plan.prompt)?;
        }
        self.large.trace_prefill_boundary_for_probe("after_draft_attach")?;
        Ok(())
    }
    fn attach_prompt_with_index(&mut self,reuse:bool,previous:usize,ids:&[u32],prefix:&[u32],
        seed:u64,greedy:bool,index:Option<Option<crate::history_lookup::HistoryLookup>>)->Result<()> {
        ensure!(!reuse||index.is_none(),"active history is already incremental");
        self.large.trace_prefill_boundary_for_probe("before_draft_attach")?;
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
        self.large.trace_prefill_boundary_for_probe("after_draft_attach")?;
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
        event["history32_verifier"]=self.copy32_admission.clone();
        for (key,value) in details.as_object().ok_or_else(||anyhow::anyhow!("begin details must be an object"))? {
            ensure!(!event.as_object().unwrap().contains_key(key),"duplicate begin field");
            event[key]=value.clone();
        }
        event["begin_phases_seconds"]["frontier_agreement"]=json!(agreement_at.elapsed().as_secs_f64());
        if std::mem::take(&mut self.early_first_emitted) {
            ensure!(output.len()==1,"early first token covers the whole first output");
            event["early_first_token"]=json!(true);
        }
        if decode_summary_enabled() {
            self.begin_record=Some((started.elapsed().as_secs_f64()*1000.,event["begin_phases_seconds"].clone(),
                event["uncached_tokens"].as_u64().unwrap_or(0)as usize,event["early_first_token"]==true));
            self.last_event_at=Some(Instant::now());
        }
        Ok(event)
    }

    fn begin_small_append(&mut self,plan:RequestPlan,greedy:bool,started:Instant,
        retained_session_cache:Value)->Result<Value> {
        let medium=plan.delta().len()>6;
        let extended=plan.delta().len()>16;
        ensure!(plan.reuse && (1..=if self.retained_append32{32}else{16}).contains(&plan.delta().len()) && self.session.is_some()
            && (!extended || (self.retained_append32&&self.copy32_verify.is_some()&&self.copy32_buffers.is_some()))
            && (!medium || (std::env::var("CANADA_SCRATCH_RESIDENT_MEDIUM_APPEND").as_deref()==Ok("1")
                && self.long_verify.is_some() && self.long_history_buffers.is_some()))
            && self.verify.profile_is_current() && self.verify.position()==plan.previous as u64
            && std::env::var("CANADA_SCRATCH_APPEND_BF16_ONLY").as_deref()==Ok("1")
            && std::env::var("CANADA_SCRATCH_TINY_APPEND_RETAIN_GRAPHS").as_deref()==Ok("1"),
            "resident small append requires a complete short suffix and BF16 graph bank");
        let prepared_at=Instant::now();
        // One whole request suffix: the decoder floor must not reset between
        // chunks of a larger prompt. Encoder history stays absolute.
        let target=if extended {
            let longer=self.copy32_verify.as_mut().unwrap();
            longer.handoff_from(&self.verify)?;
            longer
        } else if medium {
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
        if extended {
            session.resume_after_medium_extend(target,self.copy32_buffers.as_mut().unwrap(),
                plan.previous as u64,plan.delta(),&self.comm,&self.stream)?;
        } else if medium {
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
        let final_row_gather=target.append_last_gather_enabled()&&plan.delta().len()>1;
        let encoded_head=target.append_encoded_head_enabled();
        target.set_tiny_extend_for_probe(false,&self.stream)?;
        if extended {self.verify.handoff_from(self.copy32_verify.as_ref().unwrap())?;}
        else if medium {self.verify.handoff_from(self.long_verify.as_ref().unwrap())?;}
        let attached_at=Instant::now();
        self.publish_begin(plan,anchor,greedy,started,json!({
            "retained_session_cache":retained_session_cache,
            "small_append":true,
            "small_append_physical_rows":if extended {32}else if medium {16}else{6},
            "small_append_final_row_gather":final_row_gather,
            "small_append_encoded_head":encoded_head,
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
        should_cancel:&mut impl FnMut(&mut ControlChannel,usize,&[CachedSession],Option<&CachedSession>)->Result<PrefillDirective>)->Result<Value> {
        use crate::history_lookup::HistoryLookup;
        let active_reuse=plan.reuse&&wanted.is_none();
        let previous=wanted.as_ref().map_or(if plan.reuse{plan.previous}else{0},|c|c.prefix.len());
        // A true active continuation already owns a complete encoder frontier.
        // Snapshot it without splitting the new append into two numerical plans.
        // A cache-restored Complete image also sets plan.reuse; it must retain
        // the ordinary near-end policy, hence the explicit active_reuse marker.
        let live_frontier=self.encoder_active_frontier&&active_reuse
            &&retained["active_reuse"]==true&&previous>=crate::config::prompt_checkpoint_min();
        // A short suffix after a restored image (CANADA_SCRATCH_ENCODER_SHORT_SUFFIX) saves no new image.
        let short_suffix=wanted.is_some()&&plan.prompt.len()-previous<128;
        // CANADA_SCRATCH_ENCODER_SHORT_SUFFIX=2: the short suffix runs as the qualified EXTEND when the drafter rows
        // of the image's prompt tail were kept (the image sits one token before that prompt's end). Every rank
        // agrees, since the two paths issue different collectives.
        let extend_tail=if short_suffix&&crate::config::encoder_short_suffix_level()?==2&&previous>=127 {
            self.draft_tails.iter().position(|t|t.end==previous as u64+1
                &&t.tokens.as_slice()==&plan.prompt[previous+1-128..previous])
        }else{None};
        if short_suffix&&crate::config::encoder_short_suffix_level()?==2 {
            self.control.agree(&[0x7a,u8::from(extend_tail.is_some())])?;
        }
        // A turn continuing a restored image saves its next image near its prompt end when that image is
        // captured inline (CANADA_SCRATCH_ENCODER_NEAR_END); otherwise the aligned position as before.
        let near=crate::config::encoder_near_end_checkpoint()?;
        // CANADA_SCRATCH_NEAR_END_AFTER_RESTORE: a prefix restored inside the lookup (a complete image, or an encoder
        // image reconstructed there) is a continuing turn as well; it arrives without `wanted`.
        let restored_here=wanted.is_none()&&plan.reuse&&previous>0&&retained["hit"]==true
            &&retained["restored_tokens"].as_u64()==Some(previous as u64)&&crate::config::near_end_after_restore()?;
        let near_end=if near>0&&(wanted.is_some()||restored_here)&&!live_frontier&&!short_suffix
            &&plan.prompt.len()>=previous+128+near {
            let at=plan.prompt.len()-near;
            self.large.near_end_checkpoint_admitted(&self.wides,at-previous,near)?.then_some(at)
        }else{None};
        let checkpoint=if live_frontier||short_suffix{previous}
            else if let Some(at)=near_end{at}
            else{self.prompt_checkpoint_positions(plan.prompt.len())[0].max(previous)};
        // CANADA_SCRATCH_SHARED_PREFIX_CHECKPOINT: a prompt that starts cold or from a restored image and shares
        // more with a cached conversation than that image (the same system prompt, another task) saves its
        // image at the end of the shared part when that is cheap for its own next turn. Every rank holds the
        // same entries; they agree on the position all the same, since it changes the chunk plan.
        // Level 2: a cold prompt with nothing shared in the cache saves it at the end of its system part.
        let level=crate::config::shared_prefix_checkpoint()?;
        let (checkpoint,near_end,shared_image)=if level>0&&!plan.reuse&&!live_frontier&&!short_suffix {
            let minimum=crate::config::prompt_checkpoint_min();
            let common=self.cached_sessions.iter().map(|c|{
                let n=common_prefix(&c.prefix,&plan.prompt);
                if plan.media.matches_prefix(&c.media,n/128*128){n}else{0}
            }).max().unwrap_or(0);
            let system=if level==2&&wanted.is_none()&&previous==0 {
                plan.prompt.iter().position(|&id|id==crate::config::USER_TURN_TOKEN)
                    .filter(|&end|plan.media.through(end/128*128).is_empty())
            }else{None};
            let seen=shared_prefix_checkpoint(common,previous,checkpoint,plan.prompt.len(),minimum);
            let shared=seen.or_else(||system.and_then(|end|
                system_prefix_checkpoint(end,checkpoint,plan.prompt.len(),minimum)));
            self.control.agree(&[&[0x7b][..],&(shared.unwrap_or(0) as u64).to_le_bytes()].concat())?;
            // The image is one that other conversations begin with: placed by the rule, or a cold prompt's
            // ordinary image that already lies inside the shared (or system) part.
            let inside=wanted.is_none()&&previous==0&&checkpoint>=1024&&checkpoint<=common.max(system.unwrap_or(0));
            retained["shared_prefix"]=json!({"common":common,"system":system,"checkpoint":shared,
                "first_sighting":seen.is_none()&&shared.is_some(),"otherwise":checkpoint,"inside":inside});
            match shared {Some(at)=>(at,None,true),None=>(checkpoint,near_end,inside)}
        }else{(checkpoint,near_end,false)};
        ensure!(self.encoder_prompt_checkpoint&&checkpoint>=crate::config::prompt_checkpoint_min()&&checkpoint>=previous
            &&(plan.prompt.len()-checkpoint>=128||short_suffix||near_end.is_some()),"encoder serving checkpoint extent");
        // CANADA_SCRATCH_ENCODER_EXTRA_NEAR_END: the image above is not near the prompt end (a cold prompt's
        // aligned image, a shared-prefix image), so this conversation's next turn would re-prefill everything
        // after it. Ask for a second image `near` rows before the end, captured from the same last chunk, when the
        // prefill will take the inline path with both positions in that chunk. Every rank takes the same branch
        // (same configuration, prompt and plan).
        let extra_at=(crate::config::encoder_extra_near_end()?&&near>0&&near_end.is_none()&&!live_frontier&&!short_suffix
            &&plan.prompt.len()>=checkpoint+128+near
            &&self.large.extra_checkpoint_admitted(&self.wides,checkpoint-previous,plan.prompt.len()-near-previous,
                plan.prompt.len()-previous,previous>0)?).then(||plan.prompt.len()-near);
        if self.session.is_some() {
            ensure!(self.verify.profile_is_current()&&self.verify.position()==plan.previous as u64,
                "encoder serving target/history owner");
            if wanted.is_none()&&!plan.reuse {
                self.target_lineage=self.target_lineage.checked_add(1)
                    .ok_or_else(||anyhow::anyhow!("encoder target lineage exhausted"))?;
                self.verify.reset(self.target_lineage,&self.comm,&self.stream)?;
            }
            self.large.handoff_from(&self.verify)?;
        }else{ensure!(!plan.reuse&&wanted.as_ref().is_none_or(|c|self.can_import_cached_encoder(c)),
            "initial encoder session requires a fresh prompt or explicit imported parent");}
        self.prune_invalid_bank_images();
        let prepared_at=Instant::now();
        let prefix=&plan.prompt[..checkpoint];
        let media=plan.media.through(checkpoint);
        let device_bytes=self.large.encoder_checkpoint_image_bytes(checkpoint)?;
        let host_bytes=if self.history_lookup_enabled {
            crate::dspark_generation::history_lookup_reserved_bytes(self.large.context_capacity())?
        }else{0};
        let charge=device_bytes.checked_add(host_bytes).and_then(|x|x.checked_add(prefix.len()*4+media.metadata_bytes()))
            .and_then(|x|x.checked_add(crate::target::EncoderPrefixImage::binding_bytes()))
            .and_then(|x|x.checked_add(std::mem::size_of::<crate::dspark_generation::ProposalPolicyImage>()))
            .ok_or_else(||anyhow::anyhow!("encoder cache charge overflow"))?;
        let duplicate=self.cached_sessions.iter().chain(wanted.iter()).any(|c|
            (c.owner==self.bank_id||self.can_import_cached_encoder(c))
            &&c.prefix==prefix&&c.media.matches_prefix(&media,prefix.len()));
        if duplicate {
            for cached in &mut self.cached_sessions {
                if cached.owner==self.bank_id&&cached.prefix==prefix&&cached.media.matches_prefix(&media,prefix.len()) {
                    cached.speculative=false;
                }
            }
        }
        // Retain the shared parent even when this checkpoint advances into a
        // branch-specific suffix. Both images must fit the existing budget;
        // do not credit the parent's bytes before a successor actually exists.
        self.order_cache_by_lineage(prefix);
        let (budget,slots)=self.cache_capacity(0,0);
        let entries=self.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
        let optional=self.optional_entries(checkpoint);
        let ancestors=self.ladder_ancestors(prefix,&media);
        let evictions=encoder_checkpoint_evictions_ladder(&entries,&optional,ancestors.as_deref(),(checkpoint,charge),
            wanted.as_ref().map(|c|c.charge),budget,self.session_cache_long_first,slots);
        // A checkpoint below the long-standing 8,192-token minimum (CANADA_SCRATCH_PROMPT_CHECKPOINT_MIN) is
        // optional: it may take the place of other short checkpoints, its own earlier ones included, but never of
        // a checkpoint of 8,192 tokens or more, neither by the slot and byte policy nor under memory pressure.
        let (evictions,long_entries)=short_checkpoint_plan(&entries,checkpoint,charge,
            budget.checked_sub(wanted.as_ref().map_or(0,|c|c.charge)),
            slots.checked_sub(usize::from(wanted.is_some())),evictions);
        // No room among the short entries either: a short conversation whose restored parent is itself a short
        // checkpoint lets the new image take the parent's place (one slot for the conversation, its latest image)
        // when the cache holds both without the parent's reservation. The parent is dropped only once the new
        // image is saved.
        let replace_parent=evictions.is_none()&&!duplicate&&checkpoint<8192
            &&wanted.as_ref().is_some_and(|c|c.prefix.len()<8192&&c.owner==self.bank_id&&!c.shared)
            &&entries.len()<slots&&entries.iter().map(|e|e.1).sum::<usize>().saturating_add(charge)<=budget;
        let evictions=if replace_parent{Some(Vec::new())}else{evictions};
        let eligible=!duplicate&&device_bytes<=2*1024usize.pow(3)&&evictions.is_some();
        // One extra index can become the new live index while the checkpoint
        // copy remains immutable. Include that temporary allocation in UMA.
        let reserve=self.cache_reserve()?;
        let admission_required=reserve+charge+host_bytes;
        let admission_at=Instant::now();
        let admission_skipped=duplicate&&self.encoder_duplicate_skip;
        let mut admission=if admission_skipped {
            // Retained images already own their charged storage. No allocation
            // or eviction is requested, so pressure admission has no work to do.
            // Keep one explicit agreement. A rank taking ordinary admission
            // instead sends a different frame type and fails closed.
            self.control.agree(b"encoder-checkpoint-already-retained-v1")?;
            CacheSaveAdmission{admitted:false,available:0,local_fits:true,
                evicted:0,pressure_evicted:0,forecast_refused:false,recycled:None}
        }else{self.admit_cache_save(eligible,evictions,prefix,&media,admission_required,
            slots.saturating_sub(usize::from(wanted.is_some())),&long_entries,
            self.encoder_image_recycle.then_some(device_bytes))?};
        let admission_seconds=admission_at.elapsed().as_secs_f64();
        let recycled_device_bytes=admission.recycled.as_ref().map_or(0,crate::state_image::Image::bytes);
        let recycled=admission.recycled.take();
        let admitted=admission.admitted;
        // The extra image goes through the same cache rules as a second save, with the parent and the first image
        // (neither is in the vector now) reserved: (position, device bytes, charge, media).
        let extra=match extra_at.filter(|_|admitted) {
            Some(at)=>{
                let prefix2=&plan.prompt[..at];
                let media2=plan.media.through(at);
                let device2=self.large.encoder_checkpoint_image_bytes(at)?;
                let charge2=device2.checked_add(host_bytes).and_then(|x|x.checked_add(prefix2.len()*4+media2.metadata_bytes()))
                    .and_then(|x|x.checked_add(crate::target::EncoderPrefixImage::binding_bytes()))
                    .and_then(|x|x.checked_add(std::mem::size_of::<crate::dspark_generation::ProposalPolicyImage>()))
                    .ok_or_else(||anyhow::anyhow!("encoder cache charge overflow"))?;
                let duplicate2=self.cached_sessions.iter().chain(wanted.iter()).any(|c|
                    (c.owner==self.bank_id||self.can_import_cached_encoder(c))
                    &&c.prefix==prefix2&&c.media.matches_prefix(&media2,prefix2.len()));
                self.order_cache_by_lineage(prefix2);
                let (budget,slots)=self.cache_capacity(0,0);
                let reserved=wanted.as_ref().map_or(0,|c|c.charge).saturating_add(charge);
                let held=usize::from(wanted.is_some())+1;
                let entries=self.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
                let optional=self.optional_entries(at);
                let ancestors=self.ladder_ancestors(prefix2,&media2);
                let room=budget.checked_sub(reserved).zip(slots.checked_sub(held));
                let plan2=room.and_then(|(budget,slots)|cache_evictions_with_optional_ladder(&entries,&optional,
                    ancestors.as_deref(),(at,charge2),budget,self.session_cache_long_first,slots));
                let (plan2,long2)=short_checkpoint_plan(&entries,at,charge2,room.map(|r|r.0),room.map(|r|r.1),plan2);
                let eligible2=!duplicate2&&device2<=2*1024usize.pow(3)&&plan2.is_some();
                let admission2=self.admit_cache_save(eligible2,plan2,prefix2,&media2,
                    admission_required+charge2+host_bytes,slots.saturating_sub(held),&long2,None)?;
                retained["extra_near_end"]=json!({"position":at,"already_cached":duplicate2,"eligible":eligible2,
                    "admitted":admission2.admitted,"evicted_sessions":admission2.evicted});
                admission2.admitted.then_some((at,device2,charge2,media2))
            },
            None=>None,
        };
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
        let extra_history=match (&extra,&checkpoint_history) {
            (Some((at,..)),Some(index))=>{
                let index=copy_extend(index,checkpoint,&plan.prompt[checkpoint..*at])?;
                ensure!(index.as_ref().map_or(0,HistoryLookup::workspace_bytes)<=host_bytes,
                    "encoder history index exceeds reserved bytes");Some(index)
            },
            _=>None,
        };
        let index_seconds=index_at.elapsed().as_secs_f64();
        let bank_ancestors=match &wanted {
            Some(cached)=>self.bank_ancestors_before_restore(cached)?,None=>Vec::new(),
        };
        let restore=if let Some(cached)=&wanted {
            ensure!(cached.prompt_logits.is_none(),"encoder parent unexpectedly owns prompt logits");
            let CachedState::Encoder{image,..}=&cached.image else{anyhow::bail!("expected encoder cache");};
            if cached.owner==self.bank_id {Some(crate::target::EncoderPrefixRestore::Borrowed(image,self.target_lineage))}
            else{
                let source=self.peer_bank.as_ref().context("encoder import source missing")?.encoder_import_source(cached)?;
                Some(crate::target::EncoderPrefixRestore::Imported(image,source,self.target_lineage))
            }
        }else{None};
        // The first token is sampled as soon as the whole prompt is prefilled, before the image save. With
        // CANADA_SCRATCH_EARLY_FIRST_TOKEN every rank then agrees on it and rank 0 sends Ready and it, ahead of
        // the save and the drafter attach (the begin event later carries early_first_token).
        let early=crate::config::early_first_token();
        let early_state=self.agreement_state(&[]);
        // The callbacks run synchronously on the sole worker owner. Share
        // its ordered CPU control channel without introducing another writer.
        let control=std::cell::RefCell::new(&mut self.control);
        #[cfg(feature = "storage-fault-injection")]
        let fault=&mut self.cache_save_fault;
        #[cfg(feature = "storage-fault-injection")]
        let mut admission_number=0usize;
        let mut directive=PrefillDirective::Continue;
        let mut first:Option<(u32,f64,f64)>=None;
        #[cfg_attr(not(feature = "serving"), allow(unused_mut))]
        let mut early_sent=false;
        self.large.set_short_restore_extend(extend_tail.is_some())?;
        self.large.set_extra_checkpoint(extra.as_ref().filter(|_|extra_history.is_some()).map(|(at,device2,..)|(*at-previous,*device2)))?;
        let prefilled=self.large.prefill_with_encoder_checkpoint_interruptible(&mut self.wides,restore,
            &plan.prompt[previous..checkpoint],&plan.prompt[checkpoint..],admitted.then_some(device_bytes),
            recycled,
            &self.comm,&self.stream,&mut |done|{
                directive=should_cancel(&mut control.borrow_mut(),plan.prompt.len()-previous-done,&self.cached_sessions,wanted.as_ref())?;Ok(directive!=PrefillDirective::Continue)
            },
            &mut |fits| {
                #[cfg(feature = "storage-fault-injection")]
                let fits={
                    admission_number+=1;
                    let point=if admission_number==1 {cache_admission_fault::Point::EncoderBeforeAllocate}
                        else {cache_admission_fault::Point::EncoderAfterAllocate};
                    fault.apply(point,self.comm.rank(),fits)
                };
                control.borrow_mut().all_admit(fits)
            },
            &mut |target| {
                let at=Instant::now();
                let anchor=if greedy{target.greedy_token(&self.stream)?}else{target.sample_token(plan.seed,&self.stream)?};
                let sample_seconds=at.elapsed().as_secs_f64();
                if early {
                    let mut state=early_state.clone();state.extend_from_slice(&anchor.to_le_bytes());
                    control.borrow_mut().agree(&state)?;
                    #[cfg(feature = "serving")]
                    if let Some(emit)=self.early_emit.as_ref() {
                        emit.emit(crate::protocol::BackendEvent::Ready{fingerprint:self.identity.clone(),cached_tokens:previous});
                        emit.emit(crate::protocol::BackendEvent::Tokens(vec![anchor]));
                        early_sent=true;
                    }
                }
                first=Some((anchor,sample_seconds,started.elapsed().as_secs_f64()));
                Ok(())
            });
        self.large.set_short_restore_extend(false)?;
        let extra_image=self.large.take_extra_checkpoint_image();
        self.large.set_extra_checkpoint(None)?;
        let (image,details,progress)=prefilled?;
        self.large.rebind_encoder_ancestors(&bank_ancestors)?;
        retained["rebound_bank_ancestors"]=json!(bank_ancestors.len());
        let done=progress.done;
        if previous+done<plan.prompt.len() {
            // A speculative cache image/index may be ahead of the stopped
            // frontier. Release it and rebuild only the committed prefix.
            drop(image);drop(checkpoint_history);drop(extra_image);drop(extra_history);
            let resume=directive==PrefillDirective::Yield;
            let (index,policy)=if resume {
                if let Some(cached)=&wanted {
                    let CachedState::Encoder{history,policy,..}=&cached.image else{unreachable!()};
                    let index=Some(copy_extend(history,previous,&plan.prompt[previous..previous+done])?);
                    self.history.restore_cached_media(&mut plan,&cached.prefix,None,false,&cached.media)?;
                    (index,(cached.owner==self.bank_id).then(||policy.clone()))
                }else{(None,None)}
            }else{(None,None)};
            retained["retained_image"]=json!(self.return_encoder_parent(wanted)?);
            return self.interrupt_prefill(plan,greedy,active_reuse,previous,previous+done,started,retained,
                resume,details["target_total_seconds"].as_f64().context("interrupted prefill timing")?,index,policy.as_ref(),
                if previous+done<checkpoint{vec![checkpoint]}else{Vec::new()},progress.remaining_chunks);
        }
        // The complete prefill sampled the first token in its callback (see above); with
        // CANADA_SCRATCH_EARLY_FIRST_TOKEN it was also agreed and sent. With CANADA_SCRATCH_PROGRESS_NEXT the begin
        // event's progress agreement still follows (it authorizes the next command).
        let sampled_at=Instant::now();
        let (anchor,sample_seconds,first_at)=first.context("complete encoder prefill did not sample its first token")?;
        self.early_first_emitted=early_sent;
        let live_index=if active_reuse{None}
            else if let Some(index)=&checkpoint_history {
                Some(copy_extend(index,checkpoint,&plan.prompt[checkpoint..])?)
            }else if let Some(cached)=&wanted {
                let CachedState::Encoder{history,..}=&cached.image else{unreachable!()};
                Some(copy_extend(history,previous,&plan.prompt[previous..])?)
            }else{None};
        if let Some(i)=extend_tail {
            let tail=self.draft_tails.remove(i).ok_or_else(||anyhow::anyhow!("draft tail vanished"))?;
            let attached=self.attach_extend_with_tail(&plan,greedy,live_index,&tail);
            self.draft_tails.push_back(tail);
            attached?;
        }else{
            self.attach_prompt_with_index(active_reuse,previous,&plan.prompt[previous..],
                &plan.prompt,plan.seed,greedy,live_index)?;
        }
        let restored_policy=if let Some(cached)=wanted.as_ref().filter(|c|c.owner==self.bank_id) {
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
            self.cached_sessions.push(CachedSession{owner:self.bank_id,prefix:prefix.to_vec().into(),media,charge,prompt_logits:None,speculative:false,shared:shared_image,
                kind:if shared_image {"shared_prefix"} else {"prompt_checkpoint"},born:Instant::now(),used:Instant::now(),serial:self.cache_serial,restored_serial:0,
                image:CachedState::Encoder{image,history:checkpoint_history.unwrap(),
                    policy:self.session.as_ref().unwrap().proposal_policy_image()?}});
        }
        // The extra near-end image, after the first one (it is the newer entry and extends it).
        let extra_saved=extra_image.is_some();
        if let (Some(image),Some((at,device2,charge2,media2)),Some(history))=(extra_image,extra,extra_history) {
            ensure!(image.bytes()==device2&&image.position()==at,"extra encoder cache saved extent");
            self.cached_sessions.push(CachedSession{owner:self.bank_id,prefix:plan.prompt[..at].to_vec().into(),media:media2,
                charge:charge2,prompt_logits:None,speculative:false,shared:false,
                kind:"near_end",born:Instant::now(),used:Instant::now(),serial:self.cache_serial,restored_serial:0,
                image:CachedState::Encoder{image,history,policy:self.session.as_ref().unwrap().proposal_policy_image()?}});
        }
        if retained.get("extra_near_end").is_some() {retained["extra_near_end"]["saved"]=json!(extra_saved);}
        if checkpoint_saved&&near_end.is_some()&&crate::config::encoder_short_suffix_level()?==2 {
            let p=plan.prompt.len();
            let values=self.session.as_ref().unwrap().copy_prefill_tail_features(&self.stream)?;
            self.draft_tails.push_back(DraftTail{end:p as u64,tokens:plan.prompt[p-128..p-1].to_vec(),values});
            while self.draft_tails.len()>8 {self.draft_tails.pop_front();}
        }
        let restored_prefix=wanted.as_ref().map(|c|(&c.prefix,&c.media));
        if let Some((prefix,media))=restored_prefix {
            self.history.restore_cached_media(&mut plan,prefix,None,false,media)?;
            retained["restore_seconds"]=details["restore_seconds"].clone();
            retained["encoder_restored"]=json!(true);
        }
        let retained_image=if replace_parent&&checkpoint_saved {
            // The new image took the short parent's place (see replace_parent above).
            drop(wanted);self.prune_invalid_bank_images();retained["replaced_parent"]=json!(true);false
        }else{self.return_encoder_parent(wanted)?};
        retained["retained_image"]=json!(retained_image);
        ensure!(self.cached_sessions.iter().map(|c|c.charge).sum::<usize>()+self.suspended_image_bytes<=self.shared_image_budget
            &&self.cached_sessions.len()+self.suspended_image_count<=self.image_entry_limit(),"encoder cache exceeded shared budget");
        let (prompt_image,mut prompt_logit_cache)=self.prepare_prompt_logit_cache(&plan,greedy,retained["hit"]==true)?;
        self.verify.handoff_from(&self.large)?;
        if let Some((logits,device,charge))=prompt_image {
            let save_at=Instant::now();
            if let Some(image)=self.try_save_cached_session_image(anchor,device)? {
                ensure!(image.device_bytes()==device,"encoder final prompt image estimate differs");
                self.cached_sessions.push(CachedSession{owner:self.bank_id,prefix:plan.prompt.clone().into(),media:plan.media.clone(),charge,prompt_logits:Some(logits),speculative:false,shared:false,
                    kind:"prompt_end_logits",born:Instant::now(),used:Instant::now(),serial:self.cache_serial,restored_serial:0,
                    image:CachedState::Complete{image,pending:anchor}});
                prompt_logit_cache["saved"]=json!(true);
            }else{prompt_logit_cache["late_memory_refused"]=json!(true);}
            prompt_logit_cache["seconds"]=json!(prompt_logit_cache["seconds"].as_f64().unwrap()+save_at.elapsed().as_secs_f64());
        }
        let attached_at=Instant::now();
        self.publish_begin(plan,anchor,greedy,started,json!({"retained_session_cache":retained,
            "prompt_checkpoint":{"enabled":true,"encoder_only":true,"saved":checkpoint_saved,"already_cached":duplicate,
                "bank_retained":crate::config::bank_encoder_checkpoint_requested(),"device_bytes":device_bytes,
                "active_frontier":live_frontier,
                "evicted_sessions":admission.evicted,"capacity_refused":!eligible&&!duplicate,"prefix_tokens":checkpoint,
                "memory_refused":eligible&&!admitted,"local_memory_refused":eligible&&!admission.local_fits,
                "pressure_evicted_sessions":admission.pressure_evicted,"forecast_refused":admission.forecast_refused,
                "admission_available_bytes":if admission_skipped{Value::Null}else{json!(admission.available)},
                "admission_required_bytes":admission_required,
                "admission_skipped_duplicate":admission_skipped,"admission_seconds":admission_seconds,
                "recycled_device_bytes":recycled_device_bytes,
                "late_memory_refused":admitted&&!checkpoint_saved,
                "image_charge_bytes":if checkpoint_saved{charge}else{0},"history_index_seconds":index_seconds,
                "charged_bytes":self.cached_sessions.iter().map(|c|c.charge).sum::<usize>(),
                "inactive_sessions":self.cached_sessions.len()},
            "encoder_checkpoint_phases":details,"checkpoint_passes":[],"prompt_logit_cache":prompt_logit_cache,
            "prefill_phases":details["suffix_phases"],"prefix_prefill_phases":Value::Null,
            "begin_phases_seconds":{"reset_and_handoff":prepared_at.duration_since(started).as_secs_f64(),
                "target_prefill":details["target_total_seconds"],"first_target_sample":sample_seconds,
                "first_token_ready":first_at,"early_first_token":early,
                "draft_attach_and_history":attached_at.duration_since(sampled_at).as_secs_f64()+index_seconds}}))
    }

    fn execute(&mut self, command: Command) -> Result<Value> {
        self.execute_with_prefill_cancel(command,&mut |_,_,_,_|Ok(PrefillDirective::Continue))
    }

    /// Reinsert the reservation after success, optional refusal or cancellation.
    /// Admission reserved its slot/bytes, so returning it cannot evict a sibling.
    fn return_encoder_parent(&mut self,parent:Option<CachedSession>)->Result<bool> {
        self.prune_invalid_bank_images();
        if let Some(parent)=parent.filter(|_|self.retain_restored_checkpoint) {
            ensure!(self.cached_bank_image_valid(&parent),"encoder parent lost its retained bank prefix");
            if parent.owner!=self.bank_id {
                ensure!(self.can_import_cached_encoder(&parent),"returning an unknown foreign encoder parent");
                self.peer_bank.as_ref().unwrap().encoder_import_source(&parent)?;
            }
            let (budget,slots)=self.cache_capacity(0,0);
            let charged=self.cached_sessions.iter().map(|c|c.charge).sum::<usize>();
            if self.cached_sessions.len()>=slots||charged.saturating_add(parent.charge)>budget {
                // Every save made while the parent is held reserves its slot and bytes, so this is an accounting
                // gap. The worker keeps serving: the cache's own policy makes room (every rank holds the same
                // entries and plans the same victims; nothing is allocated) and the parent is dropped only when
                // that fails. Logged, so the bar sees it.
                self.order_cache_by_lineage(&parent.prefix);
                let entries=self.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
                let plan=cache_evictions(&entries,(parent.prefix.len(),parent.charge),budget,
                    self.session_cache_long_first,slots);
                eprintln!("encoder parent reservation gap: {}",json!({"parent_tokens":parent.prefix.len(),
                    "parent_charge":parent.charge,"entries":entries.len(),"charged":charged,"budget":budget,"slots":slots,
                    "evicted":plan.as_ref().map(Vec::len),"parent_dropped":plan.is_none()}));
                let Some(mut victims)=plan else {return Ok(false);};
                victims.sort_unstable();
                for i in victims.into_iter().rev(){self.evict_cached_saving(i)?;}
            }
            self.cached_sessions.push(parent);
            Ok(true)
        }else{Ok(false)}
    }

    /// A cancelled prefill retains its complete partial target/drafter state,
    /// but publishes no temporary anchor. The next wire command must finish it.
    #[allow(clippy::too_many_arguments)]
    fn interrupt_prefill(&mut self,plan:RequestPlan,greedy:bool,reuse:bool,previous:usize,
        frontier:usize,started:Instant,retained:Value,resume:bool,target_seconds:f64,
        index:Option<Option<crate::history_lookup::HistoryLookup>>,
        policy:Option<&crate::dspark_generation::ProposalPolicyImage>,
        checkpoints:Vec<usize>,remaining_chunks:Vec<usize>)->Result<Value> {
        ensure!(frontier>previous&&frontier<plan.prompt.len(),"cancelled prefill frontier");
        let anchor=self.large.greedy_token(&self.stream)?;
        self.attach_prompt_with_index(reuse,previous,&plan.prompt[previous..frontier],
            &plan.prompt[..frontier],plan.seed,greedy,index)?;
        if let Some(policy)=policy {
            let session=self.session.as_mut().unwrap();session.restore_proposal_policy(policy,&self.stream)?;
            if greedy{session.configure_greedy(Some(self.cfg.eos_token_id))?;}
            else{session.configure_sampling(plan.seed,Some(self.cfg.eos_token_id))?;}
        }
        self.verify.handoff_from(&self.large)?;
        self.history.partial_prefilled(frontier,anchor)?;
        self.current=Some(plan.id);self.emitted=0;self.limit=plan.limit;self.prefill_cancelled=!resume;
        // A yielded request may still contain image rows after this frontier.
        // Its bindings travel with the existing lossless request checkpoint.
        if !resume{self.install_media(None)?;}
        let elapsed=started.elapsed().as_secs_f64();
        let event=json!({"type":"prefill_progress","request_id":plan.id,"cancelled":!resume,
            "frontier":frontier,"prompt_tokens":plan.prompt.len(),"emitted_tokens":0,
            "cached_tokens":if plan.reuse{plan.previous}else{retained["restored_tokens"].as_u64().unwrap_or(0)as usize},
            "seconds":elapsed,"target_prefill_seconds":target_seconds,
            "retained_session_cache":retained});
        if resume {
            self.pending_prefill=Some(PendingPrefill::foreground(plan,greedy,frontier,retained,
                target_seconds,elapsed,self.prompt_checkpoint,checkpoints,remaining_chunks)?);
        }
        self.agree(&[])?;Ok(event)
    }

    fn execute_with_prefill_cancel(&mut self,command:Command,
        should_cancel:&mut impl FnMut(&mut ControlChannel,usize,&[CachedSession],Option<&CachedSession>)->Result<PrefillDirective>)->Result<Value> {
        let result=self.execute_with_prefill_cancel_inner(command,should_cancel);
        // config::wide16_elastic: a request's prefill that finished (or never paused) gives the extension back.
        if result.is_ok() {self.release_wide16_if_idle()?;}
        result
    }
    fn execute_with_prefill_cancel_inner(&mut self,command:Command,
        should_cancel:&mut impl FnMut(&mut ControlChannel,usize,&[CachedSession],Option<&CachedSession>)->Result<PrefillDirective>)->Result<Value> {
        // config::cache_keep_latest: requests save and evict images in Begin and Finish; order the cache first. A Begin's
        // prompt-side images carry its serial.
        if matches!(command, Command::Begin{..}) { self.cache_serial += 1; }
        if matches!(command, Command::Begin{..} | Command::Finish(_)) { self.order_cache_keep_latest(); }
        let started = Instant::now();
        crate::route_census::arm_once(&self.cuda, &self.core)?;
        if !matches!(command, Command::Idle) { self.last_active = started; }
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
                // config::copy_hints: hints sent for exactly this prompt become this request's extra copy source.
                self.copy_hints.retain(|(known, _)| *known != id);
                match self.pending_copy_hints.take() {
                    Some((n, tail, hints)) if n == tokens.len() && tokens.ends_with(&tail) && hints.len() >= 16 => {
                        if self.comm.rank() == 0 { eprintln!("copy hints attached {}", json!({"request_id":id,"hint_tokens":hints.len(),"prompt_tokens":n})); }
                        if self.copy_hints.len() >= 4 { self.copy_hints.remove(0); }
                        self.copy_hints.push((id, std::rc::Rc::new(crate::history_lookup::HintIndex::new(&hints)?)));
                    }
                    Some((n, _, hints)) => {
                        if self.comm.rank() == 0 { eprintln!("copy hints dropped {}", json!({"request_id":id,"hint_tokens":hints.len(),"hint_prompt_tokens":n,"prompt_tokens":tokens.len()})); }
                    }
                    None => {}
                }
                // Batched serving (Worker::region0_admits): the scheduler keeps such a request waiting.
                ensure!(self.region0_admits(crate::config::region_reach(tokens.len(), limit)?)?,
                    "a conversation that reaches past one serving region needs every region empty");
                if let Some(dir)=crate::logit_export::dir() {
                    crate::logit_export::prompt(&dir,self.comm.rank(),id,&tokens)?;
                }
                let media=self.prompt_identity(&images,&tokens)?;
                let mut plan = self.history.prepare_with_media(id, tokens, media, limit, seed)?;
                let (mut retained_session_cache,deferred_encoder)=self.reuse_cached_session(&mut plan,!greedy,true)?;
                let media_frontier=deferred_encoder.as_ref().map_or(if plan.reuse{plan.previous}else{0},|c|c.prefix.len());
                // config::engram_media_overlap: this rank's Engram reads for the first new rows run on a host thread
                // while the images encode; `stop` ends them between 1,024-id batches once encoding has returned.
                let warm=if !images.is_empty()&&plan.prompt.len()>media_frontier&&crate::config::engram_media_overlap(id) {
                    let end=plan.prompt.len().min(media_frontier+2048);
                    Some(self.large.first_rows_engram_ids(&plan.prompt[..media_frontier],&plan.prompt[media_frontier..end])?)
                }else{None};
                let stop=std::sync::atomic::AtomicBool::new(false);
                let (prepared,warmed)=std::thread::scope(|scope|{
                    let task=warm.as_ref().map(|(store1,store14,first,second)|{
                        let (store1,store14,stop)=(&**store1,&**store14,&stop);
                        scope.spawn(move||->Result<(usize,f64)>{
                            let began=Instant::now();let mut done=0usize;
                            for (a,b) in first.chunks(1024).zip(second.chunks(1024)) {
                                if stop.load(std::sync::atomic::Ordering::Relaxed) {break;}
                                store1.prefetch(a)?;store14.prefetch(b)?;done+=a.len();
                            }
                            Ok((done,began.elapsed().as_secs_f64()*1000.))
                        })
                    });
                    let media_began=Instant::now();
                    let prepared=self.prepare_media(&images,&plan,media_frontier);
                    let media_ms=media_began.elapsed().as_secs_f64()*1000.;
                    stop.store(true,std::sync::atomic::Ordering::Relaxed);
                    let joined=Instant::now();
                    let warmed=task.map(|t|(t.join(),media_ms,joined.elapsed().as_secs_f64()*1000.));
                    (prepared,warmed)
                });
                if let (Some((outcome,media_ms,join_ms)),Some((_,_,first,_)))=(warmed,warm.as_ref()) {
                    // A failed hint read is not the request's failure: the prefill's own lookup reads the row again.
                    match outcome {
                        Ok(Ok((done,ms)))=>eprintln!("engram media overlap {}",json!({"rank":self.comm.rank(),"request_id":id,
                            "row_ids":first.len(),"read_ids":done,"read_ms":ms,"media_ms":media_ms,"join_ms":join_ms})),
                        Ok(Err(error))=>eprintln!("engram media overlap read failed: {error:#}"),
                        Err(_)=>eprintln!("engram media overlap thread panicked"),
                    }
                }
                if let Err(error)=prepared {
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
                match self.admit_prefill_input() {
                    Ok(Some(reclaimed))=>{retained_session_cache["prefill_reclaim"]=reclaimed;},
                    Ok(None)=>{},
                    Err(error)=>{
                        // A borrowed encoder parent is still unchanged and its
                        // reservation remains charged; preserve it for retry.
                        self.return_encoder_parent(deferred_encoder)?;
                        return self.refuse_media_input(id,error);
                    },
                }
                let encoder_previous=deferred_encoder.as_ref().map_or(if plan.reuse {plan.previous}else{0},|c|c.prefix.len());
                // config::wide16_elastic: lend the 16,384-row profile its extension before this request plans any chunk
                // (the inline checkpoint's admission plans with the same widths as the prefill).
                let wide16=self.borrow_wide16(plan.prompt.len()-encoder_previous,plan.prompt.len())?;
                if !wide16.is_null() {retained_session_cache["wide16"]=wide16;}
                let encoder_checkpoint=self.prompt_checkpoint_positions(plan.prompt.len())[0].max(encoder_previous);
                let short_deferred=deferred_encoder.is_some()&&plan.prompt.len()-encoder_previous<128;
                if self.encoder_prompt_checkpoint&&encoder_checkpoint>=crate::config::prompt_checkpoint_min()
                    &&(plan.prompt.len().saturating_sub(encoder_checkpoint)>=128||short_deferred) {
                    return self.begin_encoder_prompt(plan,greedy,started,retained_session_cache,deferred_encoder,should_cancel);
                }
                ensure!(deferred_encoder.is_none(),"deferred encoder cache requires the compound path");
                if std::env::var("CANADA_SCRATCH_RESIDENT_SMALL_APPEND").as_deref()==Ok("1")
                    && plan.reuse && (1..=if self.retained_append32{32}else if std::env::var("CANADA_SCRATCH_RESIDENT_MEDIUM_APPEND").as_deref()==Ok("1")
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
                        let mut directive=PrefillDirective::Continue;
                        let progress=self.large.prefill_wide_interruptible(&mut self.wides,&plan.prompt[append_start..checkpoint],
                            &self.comm,&self.stream,None,&mut |done|{directive=should_cancel(control,plan.prompt.len()-append_start-done,&self.cached_sessions,None)?;Ok(directive!=PrefillDirective::Continue)})?;
                        let done=progress.done;
                        if append_start+done<checkpoint {
                            let remaining_checkpoints=std::iter::once(checkpoint)
                                .chain(checkpoints[index+1..].iter().copied().filter(|&c|c>checkpoint)).collect();
                            return self.interrupt_prefill(plan,greedy,append_reuse,append_start,append_start+done,started,
                                retained_session_cache,directive==PrefillDirective::Yield,prefix_prefill_seconds+pre.elapsed().as_secs_f64(),None,None,
                                remaining_checkpoints,progress.remaining_chunks);
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
                let mut directive=PrefillDirective::Continue;
                let progress=self.large.prefill_wide_interruptible(&mut self.wides,&plan.prompt[append_start..],
                    &self.comm,&self.stream,None,&mut |done|{directive=should_cancel(control,plan.prompt.len()-append_start-done,&self.cached_sessions,None)?;Ok(directive!=PrefillDirective::Continue)})?;
                let done=progress.done;
                if append_start+done<plan.prompt.len() {
                    return self.interrupt_prefill(plan,greedy,append_reuse,append_start,append_start+done,started,
                        retained_session_cache,directive==PrefillDirective::Yield,prefix_prefill_seconds+suffix_prefill_at.elapsed().as_secs_f64(),None,None,
                        Vec::new(),progress.remaining_chunks);
                }
                let prefilled_at = Instant::now();
                let prefill_phases = serde_json::to_value(self.large.append_metrics()?)?;
                let anchor = if greedy {self.large.greedy_token(&self.stream)?}
                    else {self.large.sample_token(plan.seed, &self.stream)?};
                let sampled_at = Instant::now();
                // CANADA_SCRATCH_EARLY_FIRST_TOKEN_APPEND: every rank agrees on the sampled token and rank 0 sends
                // Ready and it now, ahead of the drafter's resume, the prompt image and the hand-over (the begin
                // event later carries early_first_token, as on the encoder path).
                #[cfg_attr(not(feature = "serving"), allow(unused_mut))]
                let mut early_sent=false;
                if crate::config::early_first_token_append() {
                    let mut state=self.agreement_state(&[]);state.extend_from_slice(&anchor.to_le_bytes());
                    self.control.agree(&state)?;
                    #[cfg(feature = "serving")]
                    if let Some(emit)=self.early_emit.as_ref() {
                        emit.emit(crate::protocol::BackendEvent::Ready{fingerprint:self.identity.clone(),
                            cached_tokens:if plan.reuse {plan.previous}else{0}});
                        emit.emit(crate::protocol::BackendEvent::Tokens(vec![anchor]));
                        early_sent=true;
                    }
                }
                self.attach_prompt(append_reuse,append_start,&plan.prompt[append_start..],
                    &plan.prompt,plan.seed,greedy)?;
                let (prompt_image,mut prompt_logit_cache)=self.prepare_prompt_logit_cache(
                    &plan,greedy,retained_session_cache["hit"]==true)?;
                self.verify.handoff_from(&self.large)?;
                if let Some((logits,device,charge))=prompt_image {
                    let save_at=Instant::now();
                    if let Some(image)=self.try_save_cached_session_image(anchor,device)? {
                        ensure!(image.device_bytes()==device,"exact prompt image estimate disagrees");
                        self.cached_sessions.push(CachedSession{owner:self.bank_id,prefix:plan.prompt.clone().into(),media:plan.media.clone(),
                            image:CachedState::Complete{image,pending:anchor},charge,prompt_logits:Some(logits),speculative:false,shared:false,
                            kind:"prompt_end_logits",born:Instant::now(),used:Instant::now(),serial:self.cache_serial,restored_serial:0});
                        prompt_logit_cache["saved"]=json!(true);
                        prompt_logit_cache["image_charge_bytes"]=json!(charge);
                    }else{prompt_logit_cache["late_memory_refused"]=json!(true);}
                    prompt_logit_cache["seconds"]=json!(prompt_logit_cache["seconds"].as_f64().unwrap()
                        +save_at.elapsed().as_secs_f64());
                    prompt_logit_cache["charged_bytes"]=json!(self.cached_sessions.iter().map(|c|c.charge).sum::<usize>());
                    prompt_logit_cache["inactive_sessions"]=json!(self.cached_sessions.len());
                }
                let attached_at = Instant::now();
                self.early_first_emitted=early_sent;
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
                let step_started = Instant::now();
                if crate::config::copy_hints() && let Some(session) = self.session.as_mut() {
                    session.use_copy_hints(self.copy_hints.iter().find(|(id, _)| Some(*id) == self.current)
                        .map(|(_, hints)| hints.clone()));
                }
                let limit = if self.copy32_verify.is_some() {self.history.step_limit_for_rows(32)?}
                    else if self.long_verify.is_some() {self.history.step_limit_for_rows(16)?}
                    else {self.history.step_limit()?};
                ensure!(limit > 0, "decode after terminal output");
                let mut long=if let (Some(target),Some(buffers))=(&mut self.copy32_verify,&mut self.copy32_buffers) {
                    self.session.as_mut().unwrap().try_long_history_step(&mut self.verify,target,buffers,
                        self.history.pending().unwrap(),limit,&self.comm,&self.stream)?
                }else{None};
                if long.is_none() {
                    if let (Some(target),Some(buffers))=(&mut self.long_verify,&mut self.long_history_buffers) {
                        long=self.session.as_mut().unwrap().try_long_history_step(&mut self.verify,target,buffers,
                            self.history.pending().unwrap(),limit,&self.comm,&self.stream)?;
                    }
                }
                // Within-load screens: a variant alternates request by request, so one load measures both arms under
                // the same memory layout and temperature (a bench pass with an odd number of requests swaps the arms
                // on its next pass). A request is counted at its first step (its id differs from the last one's).
                let odd_request = {
                    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
                    static REQUESTS: AtomicU64 = AtomicU64::new(0);
                    static LAST_ID: AtomicU64 = AtomicU64::new(u64::MAX);
                    let id = self.current.unwrap_or(u64::MAX);
                    if LAST_ID.swap(id, Relaxed) != id { REQUESTS.fetch_add(1, Relaxed); }
                    REQUESTS.load(Relaxed) % 2 == 1
                };
                if long.is_none() && crate::config::draft_head_subset()? == 1 {
                    // Guarded draft-head subset: decided from the context before the step, the same on every rank.
                    let mut ok = crate::config::draft_head_subset_guard(self.history.committed(), self.history.pending().unwrap());
                    // Screen only (CANADA_SCRATCH_DRAFT_HEAD_SUBSET_ALTERNATE=1): odd requests keep the whole head.
                    if std::env::var("CANADA_SCRATCH_DRAFT_HEAD_SUBSET_ALTERNATE").as_deref() == Ok("1") { ok &= !odd_request; }
                    self.session.as_mut().unwrap().set_head_subset_ok(ok);
                }
                if std::env::var("CANADA_SCRATCH_INDEX_SHORT_ROWS_ALTERNATE").as_deref() == Ok("1") {
                    // Screen only: odd requests keep the indexed layers' full chain, even ones may take the short path.
                    self.verify.set_short_index_allowed(!odd_request)?;
                    self.alternate_tag = Some(if odd_request { "indexfull" } else { "indexshort" });
                }
                if std::env::var("CANADA_SCRATCH_VERIFY_POLL_ALTERNATE").as_deref() == Ok("1") {
                    // Screen only: odd requests spin on the verify pass's completion, even ones sleep 25 us a query.
                    self.verify.set_verification_poll_alternate(if odd_request { 1 } else { 25 })?;
                    self.alternate_tag = Some(if odd_request { "poll1" } else { "poll25" });
                }
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
                if decode_summary_enabled() {
                    let ms = step_started.elapsed().as_secs_f64() * 1000.;
                    crate::dspark_generation::log_step_confidence("single", &step, output.len());
                    // Probe only (Nov 12): rank 0 appends the step's mid-layer states and the block test's ratio
                    // products for an offline study (always-six data collection; nothing in the step changes).
                    if let Some(dir) = midpass_dump_dir().filter(|_| self.comm.rank() == 0 && step.verified_rows == 6
                        && step.draft_tokens.len() == 5 && self.session.as_ref().unwrap().block_verification()) {
                        use std::io::Write;
                        let states = self.verify.midpass_states_for_probe()?;
                        let w = self.session.as_ref().unwrap().midpass_weights_for_probe(5)?.unwrap_or_default();
                        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("midpass-r0.bin"))?;
                        for part in &states { file.write_all(part)?; }
                        eprintln!("stepmid bytes={:?} accepted={} drafts={:?} w={}", states.iter().map(Vec::len).collect::<Vec<_>>(),
                            step.matching_proposals, step.draft_tokens,
                            w.iter().map(|v| format!("{v:.6e}")).collect::<Vec<_>>().join(","));
                    }
                    let gap_ms = self.last_event_at.map_or(-1., |t| step_started.duration_since(t).as_secs_f64() * 1000.);
                    self.last_event_at = Some(Instant::now());
                    let d = self.decode_summary.get_or_insert_with(DecodeSummary::default);
                    if crate::dspark_generation::step_confidence_log() {
                        // Data collection beside the `stepconf` line: the kind of text at the step's first row and after
                        // its last committed row (r = reasoning, a = answer text, c = inside a tool call).
                        let kind = |ended: bool, call: bool| if call { 'c' } else if ended { 'a' } else { 'r' };
                        let mut start = None;
                        for &token in &step.tokens {
                            if token == 128822 { d.kind_ended = true; }
                            if token == 128825 { d.kind_call = true; }
                            if start.is_none() { start = Some(kind(d.kind_ended, d.kind_call)); }
                        }
                        eprintln!("stepkind start={} end={} committed={}", start.unwrap_or('?'), kind(d.kind_ended, d.kind_call), step.tokens.len());
                    }
                    if d.first.len() < 5 { d.first.push((gap_ms, step.verified_rows, step.draft_tokens.len(), output.len(), ms, step.phase_ms)); }
                    d.steps += 1; d.verified_rows += step.verified_rows; d.accepted += step.matching_proposals;
                    d.emitted += output.len(); d.wall_ms += ms;
                    for (total, phase) in d.phase_ms.iter_mut().zip(step.phase_ms) { *total += phase; }
                    let e = d.by_rows.entry(step.verified_rows).or_insert((0, 0.)); e.0 += 1; e.1 += ms;
                    let full = !step.draft_tokens.is_empty() && step.matching_proposals == step.draft_tokens.len();
                    let e = d.full_rows.entry(step.verified_rows).or_insert((0, 0)); e.0 += step.matching_proposals; e.1 += usize::from(full);
                    if step.verified_rows == 32 {
                        if d.open32 { d.merge64 += 1; d.open32 = false; } else { d.open32 = full; }
                    } else { d.open32 = false; }
                    if crate::layer_clock::armed() {
                        if let Some(parts) = crate::layer_clock::read()?.and_then(|marks| crate::layer_clock::parts_us(&marks)) {
                            for (total, value) in d.clock.iter_mut().zip(parts) { *total += value; }
                            d.clock_steps += 1;
                        }
                    }
                }
                if !self.progress_next { self.agree(&output)?; }
                let mut event = json!({"type":"tokens","request_id":self.current,"output_ids":output,"frontier":self.history.frontier(),
                    "pending":self.history.pending(),"seconds":started.elapsed().as_secs_f64(),"phase_ms":step.phase_ms,
                    "verified_rows":step.verified_rows,"step_limit":limit,
                    "committed_model_rows":step.tokens.len(),"accepted_draft_tokens":step.matching_proposals,
                    "draft_confidence":step.draft_confidence,
                    "emitted_tokens":output.len(),"active_sparse_index_layers":self.verify.active_sparse_index_layers(),
                    "compact_sparse_index_layers":self.verify.compact_sparse_index_layers(),
                    "proposal_source":step.proposal_source,"lookup_match_tokens":step.lookup_match_tokens,
                    "history_lookup_position":self.session.as_ref().unwrap().history_lookup_metadata().map(|(position,_)|position)});
                if let Some(expected) = &step.acceptance_expected_lengths {
                    event["acceptance_expected_lengths"] = json!(expected);
                    event["acceptance_shadow_preserved_draw"] = json!(true);
                }
                if confidence_economics_probe::enabled() || causal_two_three_probe::enabled() {
                    event["draft_confidence"] = json!(step.draft_confidence);
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
                self.copy_hints.retain(|(id, _)| Some(*id) != self.current);
                if let Some(session) = self.session.as_mut() { session.use_copy_hints(None); }
                // Surface deferred device errors before acknowledging a final
                // request boundary. Ordered asynchronous draft/state copies
                // must complete before HTTP can publish successful completion.
                self.stream.synchronize()?;
                crate::route_census::dump(self.comm.rank())?;
                self.history.finish(reason)?;
                self.pending_prefill=None;
                self.prefill_cancelled=false;
                if crate::config::host_stages() {
                    // Every rank: mean host time of each stage of the verifier's step in this request (the pending
                    // verification's marks, collected by the session while the verification was pending).
                    let (steps, stages) = crate::dspark_generation::take_single_stages();
                    if steps > 0 {
                        eprintln!("rank{} claude host stages: steps {} us a step: {}", self.comm.rank(), steps,
                            stages.iter().map(|(name, ms)| format!("{name} {:.1}", 1000. * ms / steps as f64)).collect::<Vec<_>>().join(" "));
                    }
                    let io = crate::engram::take_io_stats();
                    if io[0] > 0 || io[2] > 0 {
                        let per = |ns: u64, n: u64| if n > 0 { ns as f64 / 1000. / n as f64 } else { 0. };
                        eprintln!("rank{} claude row reads: lookups {} us each {:.1}; batches {} rows {} us a batch {:.1} submission {:.1}",
                            self.comm.rank(), io[0], per(io[1], io[0]), io[2], io[3], per(io[4], io[2]), per(io[5], io[2]));
                    }
                }
                if let Some(d) = self.decode_summary.as_ref().filter(|d| d.clock_steps > 0) {
                    // Every rank: where the verifier's forward spent its time, a step (layer_clock::parts_us).
                    let n = d.clock_steps as f64;
                    eprintln!("rank{} claude clock: steps {} us a step: part0 {:.1} (collectives {:.1}) wait {:.1} part1 {:.1} ({:.1}) wait {:.1} part2 {:.1} ({:.1})",
                        self.comm.rank(), d.clock_steps, d.clock[0] / n, d.clock[1] / n, d.clock[2] / n, d.clock[3] / n,
                        d.clock[4] / n, d.clock[5] / n, d.clock[6] / n, d.clock[7] / n);
                }
                if let Some(d) = self.decode_summary.take().filter(|_| self.comm.rank() == 0) {
                    let n = d.steps.max(1) as f64;
                    eprintln!("claude decode summary: steps {} tokens {} rows/step {:.2} accepted/step {:.2} step ms {:.2} (draft {:.2} verify {:.2} commit {:.2} after {:.2}) by rows {:?}",
                        d.steps, d.emitted, d.verified_rows as f64 / n, d.accepted as f64 / n, d.wall_ms / n,
                        d.phase_ms[0] / n, d.phase_ms[1] / n, d.phase_ms[2] / n, d.phase_ms[3] / n,
                        d.by_rows.iter().map(|(r, (c, ms))| (*r, *c, (ms / *c as f64 * 100.).round() / 100.)).collect::<Vec<_>>());
                    eprintln!("claude decode widths: [rows, steps, proposals accepted, full steps] {:?} merge64 {}",
                        d.by_rows.iter().map(|(r, (c, _))| { let f = d.full_rows.get(r).copied().unwrap_or((0, 0)); (*r, *c, f.0, f.1) }).collect::<Vec<_>>(),
                        d.merge64);
                    // The request's begin and its first steps (timing fields only): for the first-token itemisation.
                    if let Some((begin_ms, phases, uncached, early)) = self.begin_record.take() {
                        let ms = |key: &str| phases[key].as_f64().map_or(-1., |s| (s * 1e5).round() / 100.);
                        eprintln!("claude first steps: begin ms {:.2} uncached {} early {} (reset {} prefill {} sample {} attach {} agree {}) steps [gap ms, rows, drafts, tokens, step ms, draft, verify, commit, after] {:?}",
                            begin_ms, uncached, early, ms("reset_and_handoff"), ms("target_prefill"), ms("first_target_sample"),
                            ms("draft_attach_and_history"), ms("frontier_agreement"),
                            d.first.iter().map(|(g, r, dr, out, m, p)| format!("[{g:.2}, {r}, {dr}, {out}, {m:.2}, {:.2}, {:.2}, {:.2}, {:.2}]", p[0], p[1], p[2], p[3])).collect::<Vec<_>>());
                    }
                    if crate::config::draft_head_subset()? != 0 {
                        if let Some(session) = self.session.as_mut() {
                            let passes = session.take_draft_head_passes();
                            eprintln!("claude draft head forms: whole {} subset {}", passes[0], passes[1]);
                        }
                    }
                    if crate::config::index_short_rows() {
                        eprintln!("claude short index steps: {}", self.verify.take_short_index_steps());
                    }
                    if let Some(tag) = self.alternate_tag.take() { eprintln!("claude alternate: {tag}"); }
                }
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
                crate::query_index::finish_bf16_checks();
                Ok(json!({"type":"shutdown","frontier":self.history.frontier()}))
            }
            Command::Idle => {
                ensure!(
                    self.current.is_none(),
                    "idle heartbeat during an active request"
                );
                self.agree(&[])?;
                // Disk tier: one write-behind / parking step while the cluster is quiet.
                if self.disk_tier.is_some() {
                    if self.disk_tier_backoff>0 {self.disk_tier_backoff-=1;}
                    else if let Some(event)=self.disk_tier_idle_step()? {eprintln!("disk tier {event}");}
                }
                // config::idle_heartbeat: a short spin on this rank's GPU while the last request is recent.
                if let (Some((_, spin, window, blocks)), Some(module)) = (self.heartbeat, self.heartbeat_module.as_ref()) {
                    if spin > 0 && self.last_active.elapsed() < Duration::from_secs(window) {
                        let mut ns = spin * 1000;
                        let mut args: [*mut std::ffi::c_void; 1] = [(&mut ns as *mut u32).cast()];
                        unsafe { module.function("ca_inject_spin")?.launch([blocks, 1, 1], [32, 1, 1], 0, &self.stream, &mut args) }?;
                        self.heartbeats += 1;
                    }
                }
                Ok(json!({"type":"idle"}))
            }
            Command::ReclaimInputs { headroom } => self.reclaim_input_space(headroom),
            Command::CopyHints { prompt_len, tail, tokens } => {
                ensure!(crate::config::copy_hints(), "copy hints without the startup selection");
                let digest = crate::history_lookup::fingerprint_ids(&tokens);
                self.agree(&[tokens.len() as u32, digest as u32, (digest >> 32) as u32, prompt_len as u32])?;
                let n = tokens.len();
                self.pending_copy_hints = Some((prompt_len, tail, tokens));
                Ok(json!({"type":"copy_hints","tokens":n}))
            }
        }
    }
}

#[cfg(feature = "serving")]
fn estimated_peer_remaining(job:&crate::http_server::Job,cached:&[CachedSession],held:Option<&CachedSession>)->usize {
    // The encoder parent stays reusable while the compound prefill temporarily
    // holds it outside the cache vector. This is only a scheduling estimate;
    // ordinary cache admission and media identity checks still govern reuse.
    let prefix=cached.iter().chain(held).filter(|c|job.needs_cached_prefix(&c.prefix,&c.media)
        &&(c.prefix.len()<job.tokens.len()||c.prompt_logits.is_some()))
        .map(|c|c.prefix.len()).max().unwrap_or(0);
    job.tokens.len().saturating_sub(prefix)
}

#[cfg(feature = "serving")]
impl Worker {
    fn execute_http(&mut self,command:Command,queue:Option<&crate::http_server::Queue>,
        job:Option<&crate::http_server::Job>)->Result<Value> {
        let reclaim=match &command {Command::ReclaimInputs{headroom}=>Some(*headroom),_=>None};
        let peers=self.peer_prefill_enabled;let slice=self.peer_prefill_slice;let began=Instant::now();
        self.early_emit=job.filter(|_|crate::config::early_first_token()&&matches!(command,Command::Begin{..}))
            .map(crate::http_server::Job::early_emitter);
        self.early_first_emitted=false;
        let event=self.execute_with_prefill_cancel(command,&mut |control,remaining,cached,held| {
            // Only rank zero observes sockets. All ranks stop at the same
            // already committed encoder chunk; ordinary chunk widths and
            // numerical execution are unchanged when cancellation is absent.
            let head=queue.map(|q|->Result<[u8;1]>{
                Ok([if q.stopping()||job.is_some_and(|j|j.cancelled()){1}
                    else if peers&&q.prefill_peer_should_yield(job.is_some_and(|j|j.background),remaining,q.prefill_slice_due(began.elapsed(),slice),
                        |peer|estimated_peer_remaining(peer,cached,held))? {2}else{0}])
            }).transpose()?;
            let wire=control.command(head.as_ref().map(|v|v.as_slice()))?;
            ensure!(wire.len()==1&&wire[0]<=2&& (wire[0]!=2||peers),"foreground prefill interruption command");
            Ok(match wire[0]{0=>PrefillDirective::Continue,1=>PrefillDirective::Cancel,_=>PrefillDirective::Yield})
        });
        self.early_emit=None;
        let event=event?;
        if let (Some(queue),Some(headroom))=(queue,reclaim) {queue.input_reclaimed(headroom);}
        Ok(event)
    }

    /// Optional wide-prefill warmup (CANADA_SCRATCH_WIDE_STARTUP_WARMUP=1), after the public
    /// startup warmup: one public 8,320-token Begin, then one Begin of exactly each configured
    /// wide width (each from its own part of the fixture, so no prompt extends another), and for
    /// a 16,384-row widest width one prompt of two such chunks and a tail (its multi-chunk path),
    /// with prompt/encoder checkpoints and the session cache disabled, each cancelled after its first
    /// token and never published. First-use library loads, captures and NCCL connections of
    /// every wide profile then happen before ready instead of in a client's first request.
    /// The reset and empty-state checks are exactly those of warm_before_ready.
    #[cfg(feature = "serving")]
    fn warm_wide_before_ready(&mut self, rank: usize, widths: &[usize], pass: WideWarmupPass) -> Result<Value> {
        ensure!(self.current.is_none() && self.history.frontier()==0 && !self.progress_next
            && self.large.context_capacity()>=16384,
            "wide startup warmup needs an unpublished fresh worker");
        let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
        let corpus:Value=serde_json::from_slice(raw)?;
        let ids:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
        let prompts=wide_warmup_prompts(widths,ids.len()).context("wide startup warmup fixture extent")?;
        let (first_prompt, prompts) = match pass {
            WideWarmupPass::All => (0, prompts.as_slice()),
            WideWarmupPass::BeforePool | WideWarmupPass::RealPool => {
                let last = prompts.len() - 1;
                ensure!(widths.contains(&16384) && prompts[last].1 == 2 * 16384 + 128,
                    "split wide warmup requires the 16K multi-chunk fixture");
                if matches!(pass, WideWarmupPass::RealPool) { (last, &prompts[last..]) }
                else { (0, &prompts[..last]) }
            }
        };
        let started=Instant::now();
        // After the public warmup's reset the verify owner holds the lease; its digests are the
        // initial empty state that warm_before_ready itself verified.
        let empty_carry=self.verify.carry_digests()?;
        let empty_encoder=self.verify.encoder_tail_digest()?;
        let saved=(self.session_cache_budget,self.prompt_checkpoint,self.encoder_prompt_checkpoint,self.prompt_cache_logits);
        self.cached_sessions.clear();
        self.session_cache_budget=0;self.prompt_checkpoint=false;
        self.encoder_prompt_checkpoint=false;self.prompt_cache_logits=false;
        let run=(|| -> Result<usize> {
            let mut events=0;
            for (round,&(at,len)) in prompts.iter().enumerate() {
                let wire=Command::Begin{images:Vec::new(),id:2+(first_prompt+round) as u64,seed:0,greedy:false,limit:1,
                    tokens:ids[at..at+len].to_vec()}.encode()?;
                let command=self.control.command((rank==0).then_some(wire.as_slice()))?;
                self.execute(Command::decode(&command)?)?;
                let mut round_events=0;
                loop {
                    let command=if rank==0 {Some(Command::Finish(Finish::Cancelled).encode()?)} else {None};
                    let wire=self.control.command(command.as_deref())?;
                    let event=self.execute(Command::decode(&wire)?)?;
                    round_events+=1;
                    if event["type"]=="finish" {break;}
                    ensure!(event["type"]=="tokens" && round_events<4,"unexpected wide startup warmup event");
                }
                events+=round_events;
            }
            Ok(events)
        })();
        (self.session_cache_budget,self.prompt_checkpoint,self.encoder_prompt_checkpoint,self.prompt_cache_logits)=saved;
        let events=run?;
        self.target_lineage=self.target_lineage.checked_add(1)
            .ok_or_else(||anyhow::anyhow!("startup target lineage exhausted"))?;
        self.verify.reset(self.target_lineage,&self.comm,&self.stream)?;
        let draft_cleared=self.session.as_mut().context("wide warmup session")?.clear_startup_state(&self.stream)?;
        self.cached_sessions.clear();
        let carry_cleared=self.verify.carry_digests()?==empty_carry;
        let encoder_cleared=self.verify.encoder_tail_digest()?==empty_encoder;
        ensure!(carry_cleared && encoder_cleared,
            "wide startup reset differs from initial empty state: carry={carry_cleared}, encoder={encoder_cleared}");
        self.history=ResidentHistory::new(self.large.context_capacity(),self.cfg.eos_token_id)?;
        self.emitted=0;self.limit=0;
        ensure!(self.current.is_none() && self.history.frontier()==0
            && self.history.pending().is_none() && self.verify.profile_is_current()
            && self.verify.position()==0 && draft_cleared,
            "wide startup model/session state was not cleared");
        self.agree(&[])?;
        let memory=self.cuda.admission_memory()?;
        ensure!(memory.admitted_available>=crate::memory::floor_bytes(),"wide startup warmup consumed runtime reserve");
        Ok(json!({"seconds":started.elapsed().as_secs_f64(),"public_prompts":prompts.iter().map(|p|p.1).collect::<Vec<_>>(),"finish_events":events,
            "checkpoints_disabled_during_warmup":true,"carry_and_encoder_match_initial_empty":true,
            "no_output_published":true,"memory":memory}))
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
        let mut batched_patterns=0usize;
        if std::env::var("CANADA_SCRATCH_RESIDENT_SMALL_APPEND").as_deref()==Ok("1") {
            // Capture the bounded alternate bank before Ready. Each probe is
            // aborted, and the final startup reset below checks the original
            // empty carry/encoder state. Ordinary decode returns to its own
            // graph and attention owners before any public request can run.
            self.verify.set_tiny_extend_for_probe(true,&self.stream)?;
            for live in 1..=6 {
                self.verify.begin_verification(&teacher[..live],&self.comm,&self.stream)?;
                self.verify.warm_pending_append_sampler(&self.stream)?;
                self.verify.finish_verification(0,&self.comm,&self.stream)?;
                append_rows.push(live);
            }
            self.verify.set_tiny_extend_for_probe(false,&self.stream)?;
        }
        if let Some(longer)=&mut self.long_verify {
            longer.handoff_from(&self.verify)?;
            longer.begin_verification(&teacher[..16],&self.comm,&self.stream)?;
            longer.verification_history_sample(0,&self.comm,&self.stream)?;
            longer.finish_verification(0,&self.comm,&self.stream)?;
            if std::env::var("CANADA_SCRATCH_RESIDENT_MEDIUM_APPEND").as_deref()==Ok("1") {
                longer.set_tiny_extend_for_probe(true,&self.stream)?;
                for live in 7..=16 {
                    if crate::config::execution_graph_cache_rows()?>0 && ![7,16].contains(&live){continue;}
                    longer.begin_verification(&teacher[..live],&self.comm,&self.stream)?;
                    longer.warm_pending_append_sampler(&self.stream)?;
                    longer.finish_verification(0,&self.comm,&self.stream)?;
                    append_rows.push(live);
                }
                longer.set_tiny_extend_for_probe(false,&self.stream)?;
            }
            if crate::config::batched_graph_warmup() && crate::config::batched_rows_admitted()
                && crate::config::batched_regions()? == 4 && longer.physical_rows() == 16 {
                // Every (conversations, block, executed rows) slot a batched step can take: all conversations
                // but the last at the block's width, the last at each count up to it.
                let (context, layout) = (longer.context_capacity(), longer.swa_layout_for_probe());
                // Serving's batched steps run this owner's Engram-overlap graphs, which it only switches to at
                // its first batched step (batched_serving): switch first, or the captures land in the other set.
                longer.ensure_batched_engram_overlap()?;
                // Elastic serving regions: the captures write at every region's empty frontier, so regions 1.. need
                // their spans mapped, and the image pool does not exist yet (it takes memory last). A temporary pool
                // backs them for the warm-up and is released, every chunk returned, before the real pool. Addresses
                // never change, so the graphs stay valid when serving maps the regions from the real pool.
                let elastic = crate::config::elastic_regions()?;
                if elastic {
                    let mut chunks = 0usize;
                    // Every owner's region key rows (the 6-row and 32-row owners' too, where they span the regions).
                    for k in 1..4 {
                        chunks += longer.serving_region_missing_chunks(k)? + longer.region_keys_missing_chunks(k)?
                            + self.verify.region_keys_missing_chunks(k)?
                            + self.copy32_verify.as_ref().map_or(Ok(0), |t| t.region_keys_missing_chunks(k))?;
                    }
                    let chunk = self.cuda.image_pool_chunk_bytes()?;
                    let report = self.cuda.create_image_pool(chunks.max(1) * chunk)?;
                    for k in 1..4 {
                        ensure!(longer.map_serving_region(k)? && longer.map_region_keys(k)? && self.verify.map_region_keys(k)?
                            && self.copy32_verify.as_ref().map_or(Ok(true), |t| t.map_region_keys(k))?, "warm-up map of serving region {k}");
                    }
                    eprintln!("batched warm-up regions mapped {}", json!({"chunks":chunks,"pool":report}));
                }
                for sessions in 2..=4usize {
                    let share = crate::target::batched_block(16, sessions);
                    let blocks: Vec<usize> = if crate::config::batched_tight_blocks() { (2..=share).collect() } else { vec![share] };
                    for block in blocks {
                        for last in 1..=block {
                            let mut lives = vec![block; sessions];
                            lives[sessions - 1] = last;
                            let total: usize = lives.iter().sum();
                            let parts = lives.iter().enumerate().map(|(k, &live)| Ok((
                                crate::target::SessionBase::serving_region(k, 4, context, layout)?,
                                crate::hash_history::History::new(0x5741_524d_0000 + k as u64), live)))
                                .collect::<Result<Vec<_>>>()?;
                            longer.set_batched_rows_for_probe(parts)?;
                            longer.begin_verification(&teacher[..total],&self.comm,&self.stream)?;
                            longer.abort_batched_verification_for_probe(&self.stream)?;
                            batched_patterns += 1;
                        }
                    }
                }
                if elastic {
                    self.stream.synchronize()?;
                    for k in 1..4 {
                        longer.unmap_region_keys(k)?;
                        self.verify.unmap_region_keys(k)?;
                        if let Some(t) = self.copy32_verify.as_ref() { t.unmap_region_keys(k)?; }
                        longer.unmap_serving_region(k)?;
                    }
                    let released = self.cuda.release_image_pool()?;
                    eprintln!("batched warm-up regions released {}", json!({"chunks":released}));
                }
            }
            self.verify.handoff_from(longer)?;
        }
        if let Some(longer)=&mut self.copy32_verify {
            longer.handoff_from(&self.verify)?;
            longer.begin_verification(&teacher[..32],&self.comm,&self.stream)?;
            longer.verification_history_sample(0,&self.comm,&self.stream)?;
            longer.finish_verification(0,&self.comm,&self.stream)?;
            if self.retained_append32 {
                longer.set_tiny_extend_for_probe(true,&self.stream)?;
                for live in 17..=32 {
                    if crate::config::execution_graph_cache_rows()?>0 && ![17,32].contains(&live){continue;}
                    longer.begin_verification(&teacher[..live],&self.comm,&self.stream)?;
                    longer.warm_pending_append_sampler(&self.stream)?;
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
        ensure!(memory.admitted_available>=crate::memory::floor_bytes(),"startup warmup consumed runtime reserve");
        Ok(json!({"seconds":started.elapsed().as_secs_f64(),"public_prompt_tokens":128,
            "prewarmed_append_rows":append_rows,"prewarmed_batched_patterns":batched_patterns,
            "decode_steps":steps,"prewarmed_verify_rows":if self.copy32_verify.is_some(){vec![1,3,6,16,32]}
                else if self.long_verify.is_some() {vec![1,3,6,16]}else{vec![1,3,6]},"target_frontier":0,
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
/// Cheap checks shared by the launcher and real entry points. This deliberately
/// opens no model files, sockets, events or CUDA contexts. Extend it when a new
/// configuration-only precondition is introduced; device/state checks stay live.
fn validate_startup_config(p:&WorkerConfig,http:bool)->Result<()> {
    ensure!(!http||std::env::var("CANADA_SCRATCH_PREFILL_HEAD_SCREEN").as_deref()!=Ok("1"),
        "prefill head comparison is a finite native probe");
    crate::config::prefill_row_head_requested()?;
    crate::config::prefill_sparse_publish_requested()?;
    crate::moe::wide_transfer_requested()?;
    crate::moe::prefill_transfer_requested()?;
    crate::target::prefill_upper_rows_requested()?;
    crate::woa_medium::compact_policy()?;
    crate::l2_prefetch::wide_mask()?;
    crate::woa_medium::prefill_live_policy()?;
    if std::env::var("CANADA_SCRATCH_WOA_PREFILL_SCREEN").as_deref()==Ok("1") {
        ensure!(!http&&std::env::var("CANADA_SCRATCH_PREFILL_HEAD_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_PREFILL_BOUNDS_SCREEN").as_deref()!=Ok("1")
            &&std::env::var("CANADA_SCRATCH_WIDE_L2_SCREEN").as_deref()!=Ok("1")
            &&std::env::var("CANADA_SCRATCH_WOA_MID_SCREEN").as_deref()!=Ok("1"),
            "WO-A prefill screen requires isolated finite file execution");
    }
    ensure!(!http||std::env::var("CANADA_SCRATCH_WIDE_L2_SCREEN").as_deref()!=Ok("1"),
        "wide L2 hint comparison is confined to finite file execution");
    if std::env::var("CANADA_SCRATCH_WIDE_L2_SCREEN").as_deref()==Ok("1") {
        ensure!(crate::l2_prefetch::enabled()&&!crate::woa_medium::compact_policy()?
            &&std::env::var("CANADA_SCRATCH_WOA_MID_SCREEN").as_deref()!=Ok("1")
            &&std::env::var("CANADA_SCRATCH_WOA_MID_REFERENCE_SCREEN").as_deref()!=Ok("1"),
            "wide L2 screen requires isolated ordinary arithmetic");
    }
    ensure!(!http||std::env::var("CANADA_SCRATCH_WOA_MID_SCREEN").as_deref()!=Ok("1"),
        "compact WO-A comparison is a finite native probe");
    ensure!(std::env::var("CANADA_SCRATCH_WOA_MID_REFERENCE_SCREEN").as_deref()!=Ok("1")
        ||(!http&&std::env::var("CANADA_SCRATCH_WOA_MID_SCREEN").as_deref()==Ok("1")),
        "FP64 WO-A is confined to the finite native diagnostic");
    if std::env::var("CANADA_SCRATCH_PREFILL_BOUNDS_SCREEN").as_deref()==Ok("1") {
        ensure!(!http&&std::env::var("CANADA_SCRATCH_PREFILL_HEAD_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_SHARED_DOWN_FLASH_SCREEN").as_deref()!=Ok("1")
            &&std::env::var("CANADA_SCRATCH_MOE_WIDE_TRANSFER_SCREEN").as_deref()!=Ok("1"),
            "prefill bounds require the isolated finite file runner");
    }
    if std::env::var("CANADA_SCRATCH_MOE_WIDE_TRANSFER_SCREEN").as_deref()==Ok("1") {
        ensure!(!http&&std::env::var("CANADA_SCRATCH_PREFILL_HEAD_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_SHARED_DOWN_FLASH_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_PREFILL_ROW_HEAD_SCREEN").as_deref()!=Ok("1")
            &&std::env::var("CANADA_SCRATCH_PREFILL_PUBLISH_SCREEN").as_deref()!=Ok("1")
            &&std::env::var("CANADA_SCRATCH_INDEX_QUERY_FLASH_SCREEN").as_deref()!=Ok("1"),
            "wide expert transfer requires isolated grouped prefill/verify screen");
    }
    if std::env::var("CANADA_SCRATCH_PREFILL_PUBLISH_SCREEN").as_deref()==Ok("1") {
        ensure!(!http&&std::env::var("CANADA_SCRATCH_PREFILL_HEAD_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_PREFILL_ROW_HEAD_SCREEN").as_deref()!=Ok("1"),
            "sparse publication comparison requires one finite head variant");
    }
    if std::env::var("CANADA_SCRATCH_PREFILL_ROW_HEAD_SCREEN").as_deref()==Ok("1") {
        ensure!(!http&&std::env::var("CANADA_SCRATCH_PREFILL_HEAD_SCREEN").as_deref()==Ok("1"),
            "last-row normalization comparison requires the finite head runner");
    }
    if crate::config::prefill_encoded_head_requested()? {
        for name in ["CANADA_SCRATCH_LOSSLESS_HEAD","CANADA_SCRATCH_SHARED_TARGET_HEAD"] {
            ensure!(std::env::var(name).as_deref()==Ok("1"),"encoded prefill head requires {name}");
        }
    }
    ensure!(!http||std::env::var("CANADA_SCRATCH_SHARED_DOWN_FLASH_SCREEN").as_deref()!=Ok("1"),
        "shared-down comparison is a finite native probe");
    if std::env::var("CANADA_SCRATCH_INDEX_QUERY_FLASH_SCREEN").as_deref()==Ok("1") {
        ensure!(!http&&std::env::var("CANADA_SCRATCH_SHARED_DOWN_FLASH_SCREEN").as_deref()==Ok("1"),
            "index query comparison requires the finite prepared-dense runner");
    }
    if crate::config::ffn_fine_hint_requested()? {
        ensure!(crate::l2_prefetch::enabled(),"fine FFN hints require L2 hints");
    }
    if crate::config::append_encoded_head_requested()? {
        ensure!(crate::config::append_last_gather_requested()?,"encoded append head requires final-only transport");
        for name in ["CANADA_SCRATCH_LOSSLESS_HEAD","CANADA_SCRATCH_SHARED_TARGET_HEAD"] {
            ensure!(std::env::var(name).as_deref()==Ok("1"),"encoded append head requires {name}");
        }
    }
    if std::env::var("CANADA_SCRATCH_APPEND_HEAD_SCREEN").as_deref()==Ok("1") {
        ensure!(!http&&std::env::var("CANADA_SCRATCH_APPEND_LAST_GATHER_SCREEN").as_deref()==Ok("1"),
            "append head diagnostic requires the finite last-row runner");
    }
    if crate::config::append_last_gather_requested()? {
        for name in ["CANADA_SCRATCH_RESIDENT_SMALL_APPEND","CANADA_SCRATCH_TINY_APPEND_RETAIN_GRAPHS",
            "CANADA_SCRATCH_APPEND_BF16_ONLY"] {
            ensure!(std::env::var(name).as_deref()==Ok("1"),"last-row append transport requires {name}");
        }
    }
    if crate::config::resident_append32_requested()? {
        for name in ["CANADA_SCRATCH_RESIDENT_SMALL_APPEND","CANADA_SCRATCH_RESIDENT_MEDIUM_APPEND",
            "CANADA_SCRATCH_TINY_APPEND_RETAIN_GRAPHS","CANADA_SCRATCH_APPEND_BF16_ONLY"] {
            ensure!(std::env::var(name).as_deref()==Ok("1"),"retained M32 append requires {name}");
        }
        ensure!(std::env::var("CANADA_SCRATCH_RESIDENT_MEDIUM_APPEND_CONTROL").as_deref()!=Ok("1"),
            "retained M32 append conflicts with the medium-append control");
    }
    if std::env::var("CANADA_SCRATCH_FFN_HINT_MODEL_SCREEN").as_deref()==Ok("1") {
        ensure!(!http&&std::env::var("CANADA_SCRATCH_FFN_PREFETCH_CHUNK_SCREEN").as_deref()==Ok("1"),
            "FFN hint model screen requires finite chunk control, never HTTP");
    }
    ensure!(!http||std::env::var("CANADA_SCRATCH_GATHER_ROWS_SCREEN").as_deref()!=Ok("1"),
        "gathered-row diagnostic is unavailable in HTTP serving");
    ensure!(!http||std::env::var("CANADA_SCRATCH_APPEND_LAST_GATHER_SCREEN").as_deref()!=Ok("1"),
        "append last-row diagnostic is unavailable in HTTP serving");
    crate::control_channel::kernel_poll_us_requested()?;
    ensure!(!http||(std::env::var("CANADA_SCRATCH_FFN_SUM_SHARED_SCREEN").as_deref()!=Ok("1")
        &&std::env::var("CANADA_SCRATCH_APPEND32_SCREEN").as_deref()!=Ok("1")),
        "FFN/append component screens are unavailable in HTTP serving");
    ensure!(!http||std::env::var("CANADA_SCRATCH_MOE_EPILOGUE_SCREEN").as_deref()!=Ok("1"),
        "finite cooperative epilogue screen is unavailable in HTTP serving");
    ensure!(!http||std::env::var("CANADA_SCRATCH_QUERY_REUSE_SCREEN").as_deref()!=Ok("1"),
        "query reuse observer is restricted to the finite file runner");
    crate::config::execution_graph_cache_rows()?;
    let single=std::env::var("CANADA_SCRATCH_SINGLE_BANK");
    ensure!(single.is_err()||single.as_deref()==Ok("0")||single.as_deref()==Ok("1"),"invalid single-bank mode policy");
    ensure!(!crate::config::single_bank_requested()||(!crate::config::resident_banks_requested()
        &&!crate::config::paired_serving_requested()&&!crate::config::encoder_prefix_import_requested()
        &&(!crate::config::paired_draft_head_requested()||crate::config::batched_pooled_head())),
        "single-bank mode excludes peer capabilities (batched pooled draft heads excepted)");
    ensure!(!http||std::env::var("CANADA_SCRATCH_GRAPH_RETENTION_SCREEN").as_deref()!=Ok("1"),"graph retention probe is unavailable in HTTP serving");
    idle_anchor_enabled()?;
    ensure!(!idle_anchor_rolling()?||idle_anchor_enabled()?,"rolling anchors require idle preparation");
    encoder_duplicate_skip()?;
    encoder_image_recycle()?;
    ensure!(!http||std::env::var_os("CANADA_SCRATCH_ENCODER_RECYCLE_SCREEN").is_none(),
        "encoder recycle screen is restricted to the finite file runner");
    crate::state_image::copy::configured()?;
    let cache_slots=configured_session_cache_slots()?;
    ensure!(cache_slots<=MAX_SESSION_CACHE_SLOTS,"cache slots exceed protection mask capacity");
    ensure!(!http || std::env::var_os("CANADA_SCRATCH_CACHE_SLOTS_SCREEN").is_none(),
        "cache slot screen is restricted to the finite file runner");
    ensure!(!http || std::env::var_os("CANADA_SCRATCH_FORWARD_CACHE_SCREEN").is_none(),
        "forward cache screen is restricted to the finite file runner");
    ensure!(!http || std::env::var_os("CANADA_SCRATCH_STATE_COPY_SCREEN").is_none(),
        "state copy screen is restricted to the finite file runner");
    ensure!(!http || std::env::var_os("CANADA_SCRATCH_IDLE_ANCHOR_SCREEN").is_none(),
        "idle anchor screen is restricted to the finite file runner");
    crate::sampler_transport::Policy::from_env()?;
    crate::dspark_generation::serving_causal_three_threshold()?;
    crate::dspark_generation::serving_causal_ladder()?;
    let flag=|name:&str|->Result<bool>{match std::env::var(name){
        Ok(v)=>{ensure!(v=="0"||v=="1","{name} must be0or1");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(false),Err(e)=>Err(e.into()),
    }};
    ensure!(!encoder_image_recycle()?||cache_pressure_reclaim()?,
        "encoder reuse requires coordinated cache reclamation");
    ensure!(!idle_anchor_enabled()?||(flag("CANADA_SCRATCH_HTTP_INTERLEAVE")?
        &&flag("CANADA_SCRATCH_TWO_STREAM_ADMISSION")?
        &&flag(if std::env::var_os("CANADA_SCRATCH_ENCODER_ACTIVE_FRONTIER").is_some()
            {"CANADA_SCRATCH_ENCODER_ACTIVE_FRONTIER"}else{"CANADA_SCRATCH_ENCODER_PROMPT_CHECKPOINT"})?
        &&crate::config::bank_encoder_checkpoint_requested()),
        "idle anchor scheduling requires two-owner HTTP and active bank checkpoints");
    // These constraints also apply to narrow-only diagnostic packages: the
    // constructors read the capability flags before choosing a row profile.
    // Reject an incomplete M32-control configuration before any model load.
    crate::config::history32_generation_enabled()?;
    for name in ["CANADA_SCRATCH_HISTORY32_SPLIT_ROUTER",
        "CANADA_SCRATCH_HISTORY32_MATCH_GROUPING","CANADA_SCRATCH_HISTORY32_FP32_ATTENTION"] {
        ensure!(!flag(name)?||crate::config::history32_enabled(),
            "{name} requires the admitted native32 capability");
    }
    ensure!(p.rank<4&&p.stages.len()==40&&p.context>0&&p.context<=1048576,"worker rank/stages/context");
    if http {
        ensure!(p.requests.is_none(),"HTTP worker cannot consume file requests");
        #[cfg(feature="serving")]
        ensure!(p.http.is_some()==(p.rank==0),"HTTP source belongs to rank zero only");
    }else{
        ensure!(p.requests.is_some()==(p.rank==0),"only rank zero owns file requests");
        #[cfg(feature="serving")]
        ensure!(p.http.is_none(),"file worker cannot own an HTTP listener");
    }
    let independent=flag("CANADA_SCRATCH_INDEPENDENT_OWNER_SCREEN")?;
    let ffn=flag("CANADA_SCRATCH_PAIRED_FFN_SCREEN")?;
    let model=flag("CANADA_SCRATCH_PAIRED_MODEL_SCREEN")?;
    let actual=flag("CANADA_SCRATCH_PAIRED_ACTUAL_SCREEN")?;
    let serving=crate::config::paired_serving_requested();
    // Batched decode (serving or its screens) pools pairs of its conversations' draft heads
    // (CANADA_SCRATCH_BATCHED_POOLED_HEAD=1): its drafters are independent forks.
    ensure!(!crate::config::paired_draft_head_requested()||serving
        ||flag("CANADA_SCRATCH_PAIRED_GENERATION_SCREEN")?
        ||(crate::config::batched_pooled_head()&&(batched_serving::enabled()?||crate::config::batched_rows_admitted())),
        "paired draft head requires independent generation owners");
    ensure!(!independent||Target::independent_probe_context_admitted(p.context),
        "independent owner diagnostic context needs explicit admission before model loading");
    ensure!(!actual||(independent&&ffn&&model),"fresh pair screen requires independent/FFN/model admission");
    for name in ["CANADA_SCRATCH_PAIRED_SHARED_UP","CANADA_SCRATCH_PAIRED_QKV","CANADA_SCRATCH_PAIRED_TARGET_HEAD"] {
        flag(name)?;
    }
    ensure!(!(serving||(independent&&ffn))||!flag("CANADA_SCRATCH_PAIRED_SHARED_UP")?||flag("CANADA_SCRATCH_NORM_QUANT_FUSION")?,
        "pooled shared-up requires the normalized prequantized input path");
    ensure!(!flag("CANADA_SCRATCH_PAIRED_DENSE_TIMING_SCREEN")?||(actual
        &&flag("CANADA_SCRATCH_PAIRED_SHARED_UP")?&&flag("CANADA_SCRATCH_PAIRED_QKV")?
        &&flag("CANADA_SCRATCH_PAIRED_TARGET_HEAD")?),"dense timing requires all three pair components");
    if flag("CANADA_SCRATCH_MATCHED_PATH_SCREEN")? {
        ensure!(!http&&!actual&&!independent,"matched path is an isolated file diagnostic");
        matched_path_probe::validate()?;
    }
    ensure!(!flag("CANADA_SCRATCH_CONFIDENCE_TRANSFER_SCREEN")?||(!http
        &&confidence_economics_probe::enabled()&&flag("CANADA_SCRATCH_BLOCK_PROGRESS_SCREEN")?),
        "confidence transfer requires the isolated file-worker economics screen");
    ensure!(!flag("CANADA_SCRATCH_CONFIDENCE_POLICY_SCREEN")?||(!http
        &&!confidence_economics_probe::enabled()&&!flag("CANADA_SCRATCH_BLOCK_PROGRESS_SCREEN")?
        &&flag("CANADA_SCRATCH_LONG_HISTORY_GENERATION")?
        &&crate::compressed_source::source20_fp32_reduction_enabled()?
        &&crate::layer_attention::generation_swa_bf16_enabled()?
        &&crate::dspark_generation::serving_causal_three_threshold()?==0.25),
        "confidence policy requires the isolated file-worker with configured0.25cutoff");
    if http&&flag("CANADA_SCRATCH_HTTP_INTERLEAVE")? {
        ensure!(p.context==1048576,"HTTP scheduling requires full1M capacity");
        #[cfg(feature="serving")]
        interleave_http::validate_startup()?;
    }
    Ok(())
}
pub fn validate_configuration(path:&Path,http:bool)->Result<Value> {
    let p:WorkerConfig=serde_json::from_slice(&std::fs::read(path)?)?;
    validate_startup_config(&p,http)?;
    Ok(json!({"valid":true,"rank":p.rank,"context":p.context,"http":http,
        "scope":"Known host-only flag and profile dependencies; no model/device work"}))
}
pub fn run_file(path: &Path) -> Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_PREFILL_ADMISSION_SCREEN").as_deref()!=Ok("1")
        || cfg!(feature="storage-fault-injection"),"prefill pressure screen requires a fault-injection build");
    let retention_probe=std::env::var("CANADA_SCRATCH_CACHE_RETENTION_SCREEN").as_deref()==Ok("1");
    ensure!(!retention_probe || cfg!(feature="storage-fault-injection"),
        "cache refusal screen requires a non-serving fault-injection build");
    let p: WorkerConfig = serde_json::from_slice(&std::fs::read(path)?)?;
    validate_startup_config(&p,false)?;
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
    if std::env::var("CANADA_SCRATCH_WOA_MID_SCREEN").as_deref()==Ok("1") {
        return woa_mid_native_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_WIDE_L2_SCREEN").as_deref()==Ok("1") {
        return wide_hint_native_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_PREFILL_BOUNDS_SCREEN").as_deref()==Ok("1") {
        return prefill_bounds_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_CLAUDE_PREFILL_PROFILE").as_deref()==Ok("1") {
        return claude_profile_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_CLAUDE_SMALL_PROFILE").as_deref()==Ok("1") {
        return claude_small_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_CLAUDE_DECODE_PROFILE").as_deref()==Ok("1") {
        return claude_decode_profile::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var_os("CANADA_SCRATCH_CLAUDE_AB").is_some() {
        return claude_ab_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_WOA_PREFILL_SCREEN").as_deref()==Ok("1") {
        return woa_prefill_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_PREFILL_HEAD_SCREEN").as_deref()==Ok("1") {
        prefill_head_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;
        if std::env::var("CANADA_SCRATCH_PREFILL_PUBLISH_SCREEN").as_deref()==Ok("1") {
            history32_generation_probe::execute(&mut worker,Command::Shutdown)?;
            return Ok(());
        }
    }
    if std::env::var("CANADA_SCRATCH_SHARED_DOWN_FLASH_SCREEN").as_deref()==Ok("1") {
        return shared_down_flash_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_QUERY_REUSE_SCREEN").as_deref()==Ok("1") {
        return query_reuse_worker_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_SHARDED_PREFIX_NATIVE_SCREEN").as_deref()==Ok("1") {
        return transport_placement_probe::run_sharded_prefix(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_CONTROL_KERNEL_POLL_SCREEN").as_deref()==Ok("1") {
        return transport_placement_probe::run_kernel_poll(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_UNUSED_DRAFT_SCREEN").as_deref()==Ok("1") {
        return unused_draft_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_ENGRAM_PROJECTION_SCREEN").as_deref()==Ok("1") {
        return engram_projection_probe::run(&mut worker,p.rank,&p.kernels,p.events.parent().unwrap());
    }
    #[cfg(feature = "serving")]
    if std::env::var("CANADA_SCRATCH_TRANSPORT_PLACEMENT_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_IDLE_POLICY_SCREEN").as_deref()==Ok("1") {
        ensure!(std::env::var("CANADA_SCRATCH_IDLE_ANCHOR_SCREEN").as_deref()==Ok("1"),
            "combined policy screen needs its finite storage gate");
        transport_placement_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),&p.kernels)?;
        return idle_policy_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    #[cfg(feature = "vision-protocol")]
    if std::env::var("CANADA_SCRATCH_ENCODER_RECYCLE_SCREEN").as_deref()==Ok("1") {
        return encoder_recycle_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    #[cfg(feature = "vision-protocol")]
    if std::env::var("CANADA_SCRATCH_IDLE_ANCHOR_SCREEN").as_deref()==Ok("1") {
        #[cfg(feature = "serving")]
        if std::env::var("CANADA_SCRATCH_IDLE_POLICY_SCREEN").as_deref()==Ok("1") {
            return idle_policy_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
        }
        return idle_anchor_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    #[cfg(feature = "vision-protocol")]
    if std::env::var("CANADA_SCRATCH_STATE_COPY_SCREEN").as_deref()==Ok("1") {
        return state_copy_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    #[cfg(feature = "vision-protocol")]
    if let Ok(path)=std::env::var("CANADA_SCRATCH_CACHE_SLOTS_SCREEN") {
        return cache_slots_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),Path::new(&path));
    }
    #[cfg(feature = "vision-protocol")]
    if let Ok(path)=std::env::var("CANADA_SCRATCH_FORWARD_CACHE_SCREEN") {
        return forward_cache_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),Path::new(&path));
    }
    if std::env::var("CANADA_SCRATCH_INDEPENDENT_OWNER_SCREEN").as_deref()==Ok("1") {
        return independent_owner_probe::run(&mut worker,p.rank,&p.kernels,p.events.parent().unwrap());
    }
    #[cfg(feature="serving")]
    if std::env::var("CANADA_SCRATCH_PEER_PREFILL_SCREEN").as_deref()==Ok("1") {
        peer_prefill_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;
        if std::env::var("CANADA_SCRATCH_POOLED_ROUTE_CAPTURE").as_deref()==Ok("1") {
            // Two bounded diagnostics can share a model load. The capture
            // still begins with ordinary fresh requests and no prompt cache.
            ensure!(worker.current.is_none(),"prefill diagnostic left an active owner");
            worker.cached_sessions.clear();worker.session_cache_budget=0;
            worker.prompt_checkpoint=false;worker.encoder_prompt_checkpoint=false;
            worker.encoder_active_frontier=false;worker.prompt_cache_logits=false;
            return pooled_route_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),
                json!({"preceding_peer_prefill_screen":true}));
        }
        if std::env::var("CANADA_SCRATCH_PAIR_STEP_PROFILE").as_deref()==Ok("1") {
            pair_step_profile_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;
        }
        return Ok(());
    }
    if std::env::var("CANADA_SCRATCH_PAIR_STEP_PROFILE").as_deref()==Ok("1") {
        return pair_step_profile_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_ATTENTION16_REFRESH_SCREEN").as_deref()==Ok("1") {
        return attention16_refresh_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_INDEX_SCORE_MODEL_SCREEN").as_deref()==Ok("1") {
        return index_score_model_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_COPY_KV_BATCH_SCREEN").as_deref()==Ok("1") {
        let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
        let tokens:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
        history32_generation_probe::execute(&mut worker,Command::Begin{images:Vec::new(),id:7_891_001,
            seed:2026111002,greedy:false,limit:1,tokens:tokens[..1013].to_vec()})?;
        let before=worker.session.as_mut().unwrap().conditioning_state_digest_for_probe(&worker.comm,&worker.stream)?;
        let report=worker.session.as_mut().unwrap().probe_copy_batched_kv(&worker.comm,&worker.stream,&mut worker.control)?;
        ensure!(before==worker.session.as_mut().unwrap().conditioning_state_digest_for_probe(&worker.comm,&worker.stream)?,
            "copy KV batching screen changed accepted drafter state");
        std::fs::write(p.events.parent().unwrap().join("copy-kv-batch-summary.json"),serde_json::to_vec_pretty(&report)?)?;
        history32_generation_probe::execute(&mut worker,Command::Finish(Finish::Length))?;
        history32_generation_probe::execute(&mut worker,Command::Shutdown)?;return Ok(());
    }
    if std::env::var("CANADA_SCRATCH_COPY_CONDITIONING_SCREEN").as_deref()==Ok("1") {
        let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
        let tokens:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
        history32_generation_probe::execute(&mut worker,Command::Begin{images:Vec::new(),id:7_890_001,
            seed:2026111001,greedy:false,limit:1,tokens:tokens[..1024].to_vec()})?;
        let before=worker.session.as_mut().unwrap().conditioning_state_digest_for_probe(&worker.comm,&worker.stream)?;
        let mut report=worker.session.as_ref().unwrap().probe_copy_conditioning(&worker.core,&p.kernels,&worker.comm,&worker.stream)?;
        let after=worker.session.as_mut().unwrap().conditioning_state_digest_for_probe(&worker.comm,&worker.stream)?;
        ensure!(before==after,"component probe changed accepted drafter state");
        report["accepted_draft_state_unchanged"]=json!(true);
        std::fs::write(p.events.parent().unwrap().join("copy-conditioning-summary.json"),serde_json::to_vec_pretty(&report)?)?;
        history32_generation_probe::execute(&mut worker,Command::Finish(Finish::Length))?;
        history32_generation_probe::execute(&mut worker,Command::Shutdown)?;return Ok(());
    }
    if std::env::var("CANADA_SCRATCH_CLOCK_WORKLOAD_SCREEN").as_deref()==Ok("1") {
        return clock_workload_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_FFN_HINT_MODEL_SCREEN").as_deref()==Ok("1") {
        ffn_hint_model_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;
        history32_generation_probe::execute(&mut worker,Command::Shutdown)?;return Ok(());
    }
    if std::env::var("CANADA_SCRATCH_FFN_SUM_SHARED_SCREEN").as_deref()==Ok("1") {
        ffn_sum_append_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;
        history32_generation_probe::execute(&mut worker,Command::Shutdown)?;return Ok(());
    }
    if std::env::var("CANADA_SCRATCH_GRAPH_RETENTION_SCREEN").as_deref()==Ok("1") {
        graph_retention_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;
        history32_generation_probe::execute(&mut worker,Command::Shutdown)?;return Ok(());
    }
    if std::env::var("CANADA_SCRATCH_APPEND_LAST_GATHER_SCREEN").as_deref()==Ok("1") {
        append_last_gather_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;
        history32_generation_probe::execute(&mut worker,Command::Shutdown)?;return Ok(());
    }
    if std::env::var("CANADA_SCRATCH_GATHER_ROWS_SCREEN").as_deref()==Ok("1") {
        gather_rows_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;
        history32_generation_probe::execute(&mut worker,Command::Shutdown)?;return Ok(());
    }
    if std::env::var("CANADA_SCRATCH_MOE_EPILOGUE_SCREEN").as_deref()==Ok("1") {
        moe_epilogue_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;
        history32_generation_probe::execute(&mut worker,Command::Shutdown)?;
        return Ok(());
    }
    if std::env::var("CANADA_SCRATCH_TRANSPORT_PLACEMENT_SCREEN").as_deref()==Ok("1") {
        return transport_placement_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),&p.kernels);
    }
    if std::env::var("CANADA_SCRATCH_PACKET_MODEL_SCREEN").as_deref()==Ok("1") {
        return packet_workload_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_CLOCK_VERIFY_SCREEN").as_deref()==Ok("1") {
        return clock_verify_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_VERIFY_WAIT_SCREEN").as_deref()==Ok("1") {
        return verify_wait_probe::run(&mut worker,p.rank,&p.kernels,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_SOLO_FFN_COMBINED_SCREEN").as_deref()==Ok("1") {
        ensure!(std::env::var("CANADA_SCRATCH_FFN16_GENERATION_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_FFN32_REFRESH_SCREEN").as_deref()==Ok("1"),
            "combined FFN qualification requires both finite screens");
        ffn16_generation_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;
        ffn32_refresh_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;
        history32_generation_probe::execute(&mut worker,Command::Shutdown)?;
        return Ok(());
    }
    if std::env::var("CANADA_SCRATCH_FFN16_GENERATION_SCREEN").as_deref()==Ok("1") {
        return ffn16_generation_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_SOLO_COMPOSITION_SCREEN").as_deref()==Ok("1") {
        ensure!(std::env::var("CANADA_SCRATCH_FFN32_GENERATION_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_VERIFY16_OVERLAP_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_MOE_CACHE_CA_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_SAMPLER_GENERATION_SCREEN").as_deref()==Ok("1"),"overnight composition flags");
        return ffn32_generation_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_SOLO_CACHE_SAMPLER_COMBINED_SCREEN").as_deref()==Ok("1") {
        ensure!(std::env::var("CANADA_SCRATCH_MOE_CACHE_CA_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_SAMPLER_GENERATION_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_BLOCK_FUSED_SCREEN").as_deref()==Ok("1"),"combined cache/sampler flags");
        moe_cacheca_probe::run(&mut worker,p.rank,p.events.parent().unwrap())?;
        worker.verify.set_cache_ca_for_probe(true,&worker.stream)?;
        return sampler_generation_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_SAMPLER_GENERATION_SCREEN").as_deref()==Ok("1") {
        return sampler_generation_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_SOLO_MICRO_COMBINED_SCREEN").as_deref()==Ok("1") {
        ensure!(std::env::var("CANADA_SCRATCH_WIDE_EPILOGUE_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_MOE_LATE_PDL_SCREEN").as_deref()==Ok("1"),
            "combined micro screen flags");
        wide_epilogue_probe::run(&mut worker,p.rank,&p.kernels,p.events.parent().unwrap())?;
        return moe_late_pdl_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_WIDE_EPILOGUE_SCREEN").as_deref()==Ok("1") {
        return wide_epilogue_probe::run(&mut worker,p.rank,&p.kernels,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_MOE_LATE_PDL_SCREEN").as_deref()==Ok("1") {
        return moe_late_pdl_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_FFN32_GENERATION_SCREEN").as_deref()==Ok("1") {
        return ffn32_generation_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_FFN32_REFRESH_SCREEN").as_deref()==Ok("1") {
        return ffn32_refresh_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_FFN16_REFRESH_SCREEN").as_deref()==Ok("1") {
        return ffn16_refresh_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_BATCH_SCREEN").as_deref()==Ok("1") {
        return batch_screen_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),&p.kernels);
    }
    if std::env::var("CANADA_SCRATCH_WOA_PREFETCH_SCREEN").as_deref()==Ok("1") {
        return woa_prefetch_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if crate::config::l2_queue_screen() {
        return l2_queue_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_OUTPUT_PROJECTION_SCREEN").as_deref()==Ok("1") {
        return output_projection_probe::run(&mut worker,p.rank,&p.kernels,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_WIDE_MHC_SCREEN").as_deref()==Ok("1") {
        return wide_mhc_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_MHC_PDL_PAIR_SCREEN").as_deref()==Ok("1") {
        return mhc_pdl_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if crate::config::history32_probe_enabled() {
        if std::env::var("CANADA_SCRATCH_HISTORY_MATCH_GENERATION_SCREEN").as_deref()==Ok("1") {
            return history_minimum_probe::run(&mut worker,p.rank,&p.kernels,p.events.parent().unwrap());
        }
        if std::env::var("CANADA_SCRATCH_COMPACT32_INDEX_SCREEN").as_deref()==Ok("1") {
            return compact32_index_probe::run(&mut worker,p.rank,&p.kernels,p.events.parent().unwrap());
        }
        if std::env::var("CANADA_SCRATCH_HISTORY32_FRESH_SCREEN").as_deref()==Ok("1") {
            return history32_fresh_probe::run(&mut worker,p.rank,&p.kernels,p.events.parent().unwrap());
        }
        if std::env::var("CANADA_SCRATCH_HISTORY32_GENERATION_SCREEN").as_deref()==Ok("1") {
            return history32_generation_probe::run(&mut worker,p.rank,&p.kernels,p.events.parent().unwrap());
        }
        return history32_probe::run(&mut worker,p.rank,&p.kernels,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_HISTORY_COPY_ROUTE_SCREEN").as_deref()==Ok("1") {
        return history_copy_route_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_HISTORY_CONFIDENCE_SCREEN").as_deref()==Ok("1") {
        return history_confidence_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_MATCHED_PATH_SCREEN").as_deref()==Ok("1") {
        return matched_path_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
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
    if std::env::var("CANADA_SCRATCH_PREFILL_ADMISSION_SCREEN").as_deref()==Ok("1") {
        return prefill_admission_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
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
    if std::env::var("CANADA_SCRATCH_RESTORE_INITIAL_SCREEN").as_deref()==Ok("1") {
        return restore_initial_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_ENCODER_CACHE_SCREEN").as_deref()==Ok("1") {
        return encoder_cache_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_INLINE_CHECKPOINT_SCREEN").as_deref()==Ok("1") {
        return inline_checkpoint_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_REPLAY_EXACT_SCREEN").as_deref()==Ok("1") {
        return crate::replay_exact_probe::run(&worker.cuda,&mut worker.large,&mut worker.verify,&mut worker.wides,
            &worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank);
    }
    if std::env::var("CANADA_SCRATCH_RESTORE_KL_SCREEN").as_deref()==Ok("1") {
        return crate::restore_kl_probe::run(&mut worker.large,&mut worker.verify,&mut worker.wides,
            &worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank);
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
    if std::env::var("CANADA_SCRATCH_ALLOCATION_PLACEMENT_SCREEN").as_deref()==Ok("1") {
        return crate::allocation_placement_probe::run(&worker.cuda,&std::fs::read(&p.core)?,&mut worker.large,&mut worker.verify,
            &mut worker.wides,&worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank);
    }
    if std::env::var("CANADA_SCRATCH_KEY_SHARDED_INDEX_SCREEN").as_deref()==Ok("1") {
        if std::env::var("CANADA_SCRATCH_KEY_SHARDED16_SCREEN").as_deref()==Ok("1") {
            return crate::key_sharded16_probe::run(&worker.cuda,&mut worker.large,&mut worker.verify,
                worker.long_verify.as_mut().ok_or_else(||anyhow::anyhow!("key partition screen needs M16"))?,
                &mut worker.wides,&worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank);
        }
        return crate::key_sharded_model_probe::run(&worker.cuda,&mut worker.large,&mut worker.verify,
            &mut worker.wides,&worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank);
    }
    if std::env::var("CANADA_SCRATCH_VERIFY_CAPTURE_SCREEN").as_deref()==Ok("1") {
        return crate::verification_capture_probe::run(&worker.cuda,&mut worker.large,&mut worker.verify,
            worker.long_verify.as_mut().ok_or_else(||anyhow::anyhow!("verification capture needs M16"))?,
            &mut worker.wides,&worker.comm,&worker.stream,&mut worker.control,p.events.parent().unwrap(),p.rank);
    }
    if std::env::var("CANADA_SCRATCH_MOE_INPUT_ADDRESS_SCREEN").as_deref()==Ok("1")
        || std::env::var("CANADA_SCRATCH_MOE_SCALE_CODE_SCREEN").as_deref()==Ok("1") {
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
    // This finite operator comparison owns the fresh target and must precede
    // serving warmup, which transfers ownership and builds the append bank.
    if std::env::var("CANADA_SCRATCH_SWA_PRECISION_WIDE_SCREEN").as_deref()==Ok("1")
        ||std::env::var("CANADA_SCRATCH_SPARSE_PRECISION_WIDE_SCREEN").as_deref()==Ok("1") {
        return swa_precision_probe::run_wide(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_SWA_PRECISION_SCREEN").as_deref()==Ok("1")
        ||std::env::var("CANADA_SCRATCH_SOURCE20_PRECISION_SCREEN").as_deref()==Ok("1") {
        return swa_precision_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    #[cfg(feature = "serving")]
    let startup_warmup = if std::env::var("CANADA_SCRATCH_FILE_STARTUP_WARMUP").as_deref()==Ok("1") {
        Some(worker.warm_resident_before_ready(p.rank)?)
    } else {None};
    #[cfg(not(feature = "serving"))]
    let startup_warmup:Option<Value> = None;
    if std::env::var("CANADA_SCRATCH_POOLED_ROUTE_CAPTURE").as_deref()==Ok("1") {
        return pooled_route_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),
            serde_json::to_value(startup_warmup)?);
    }
    if confidence_economics_probe::enabled() {
        return confidence_economics_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),
            serde_json::to_value(startup_warmup)?);
    }
    if std::env::var("CANADA_SCRATCH_SOLO_ECONOMICS_SCREEN").as_deref()==Ok("1") {
        return solo_economics_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),
            serde_json::to_value(startup_warmup)?);
    }
    if causal_two_three_probe::enabled() {
        return causal_two_three_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),
            serde_json::to_value(startup_warmup)?);
    }
    if std::env::var("CANADA_SCRATCH_EXPERT_PREFETCH_SCREEN").as_deref()==Ok("1") {
        return expert_prefetch_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),
            serde_json::to_value(startup_warmup)?);
    }
    if confidence_policy_probe::enabled() {
        return confidence_policy_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),
            serde_json::to_value(startup_warmup)?);
    }
    if std::env::var("CANADA_SCRATCH_BLOCK_PROGRESS_SCREEN").as_deref()==Ok("1") {
        return block_progress_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),
            serde_json::to_value(startup_warmup)?);
    }
    if std::env::var("CANADA_SCRATCH_FIXED_PREFIX_QUALITY_SCREEN").as_deref()==Ok("1") {
        return fixed_prefix_quality_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_CHUNK_ECONOMICS_SCREEN").as_deref()==Ok("1") {
        return chunk_economics_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_FLEX_WIDTH_SCREEN").as_deref()==Ok("1") {
        return flex_width_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_CHUNK_PLAN_SCREEN").as_deref()==Ok("1") {
        return chunk_plan_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_PREFILL_WAIT_SCREEN").as_deref()==Ok("1") {
        return prefill_wait_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_HIERARCHICAL_MODEL_SCREEN").as_deref()==Ok("1") {
        return hierarchical_model_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_WIDE_ENGRAM_SCREEN").as_deref()==Ok("1") {
        return wide_engram_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if sampler_law_probe::enabled() {
        return sampler_law_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_STEP_BUDGET_SCREEN").as_deref()==Ok("1") {
        return step_budget_probe::run(&mut worker,p.rank,p.events.parent().unwrap());
    }
    if std::env::var("CANADA_SCRATCH_SAMPLER_READOUT_SCREEN").as_deref()==Ok("1") {
        return sampler_readout_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),
            serde_json::to_value(startup_warmup)?);
    }
    if std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_NVFP4_SCREEN").as_deref()==Ok("1") {
        return private_nvfp4_probe::run(&mut worker,p.rank,p.events.parent().unwrap(),
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
        "verification_poll_us":Target::verification_poll_us_requested()?,
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
    ensure!(std::env::var("CANADA_SCRATCH_PREFILL_ADMISSION_SCREEN").as_deref()!=Ok("1"),
        "prefill pressure probe is restricted to the finite file runner");
    ensure!(std::env::var("CANADA_SCRATCH_CACHE_RETENTION_SCREEN").as_deref()!=Ok("1"),
        "cache refusal probe is restricted to the finite file runner");
    use crate::{http_server, protocol::BackendEvent};
    use deepseek_recipe::stream::InferenceFinishReason;
    let p: WorkerConfig = serde_json::from_slice(&std::fs::read(path)?)?;
    validate_startup_config(&p,true)?;
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
    if http_interleave { interleave_http::validate_startup()?; }
    ensure!(std::env::var("CANADA_SCRATCH_INDEPENDENT_OWNER_SCREEN").as_deref()!=Ok("1")
        ||Target::independent_probe_context_admitted(p.context),
        "independent owner diagnostic context needs explicit admission before model loading");
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
    worker.control.agree(&crate::config::execution_graph_cache_rows()?.to_le_bytes())?;
    let resident_banks=worker.initialize_resident_banks(&p.kernels)?;
    worker.cuda.report_allocation_sites("resident banks constructed")?;
    let batched_drafters=worker.prepare_batched_drafters(&p.kernels)?;
    ensure!(!http_interleave || (!post_http_causal && !post_http_capture && !post_http_control
        && !post_http_verify_profile && !post_http_long_quality),
        "HTTP interleave experiment needs an isolated worker");
    let startup_warmup = match std::env::var("CANADA_SCRATCH_WARMUP_BEFORE_READY") {
        Ok(value) => {
            ensure!(value=="0" || value=="1","invalid startup warmup policy");
            if value=="1" {Some(worker.warm_resident_before_ready(p.rank)?)} else {None}
        }
        // Keep the tiny diagnostic capacity usable; the public warmup needs
        // room for its 128-token prefix and bounded continuation.
        Err(std::env::VarError::NotPresent) if p.context>=256 =>
            Some(worker.warm_resident_before_ready(p.rank)?),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    #[cfg(feature = "serving")]
    let warm_real_pool = std::env::var("CANADA_SCRATCH_WIDE16_WARM_REAL_POOL").as_deref() == Ok("1");
    #[cfg(feature = "serving")]
    worker.control.agree(&[0xea, u8::from(warm_real_pool)])?;
    #[cfg(feature = "serving")]
    let wide_startup_warmup = if std::env::var("CANADA_SCRATCH_WIDE_STARTUP_WARMUP").as_deref()==Ok("1") {
        ensure!(startup_warmup.is_some(),"wide startup warmup follows the public startup warmup");
        {
            // config::wide16_elastic: the 16,384-row width warms (and captures its graphs) with its extension mapped
            // from a temporary pool, returned before the real pool exists.
            worker.cuda.wide16_begin_temporary()?;
            let pass = if warm_real_pool { WideWarmupPass::BeforePool } else { WideWarmupPass::All };
            let warmed=worker.warm_wide_before_ready(p.rank,&p.wide_widths,pass);
            worker.cuda.wide16_end_temporary()?;
            Some(warmed?)
        }
    } else {None};
    #[cfg(not(feature = "serving"))]
    let wide_startup_warmup:Option<Value> = None;
    worker.cuda.report_allocation_sites("startup warmup complete")?;
    // Nov 17: the snapshot chunk pool (cuda::ChunkPool), created after every startup allocation so it takes memory last;
    // every rank must agree on its size (snapshots are planned identically everywhere).
    let image_pool_mib = image_pool_mib()?;
    worker.control.agree(&[0x7a].iter().copied().chain((image_pool_mib.unwrap_or(0) as u64).to_le_bytes()).collect::<Vec<u8>>())?;
    // Before the pool takes its memory: give back the page cache of the kernel bundles read at load (see
    // memory::release_file_cache). Rank-local and advisory; every rank reports what it released.
    #[cfg(target_os = "linux")]
    if std::env::var("CANADA_SCRATCH_RELEASE_LOAD_FILE_CACHE").as_deref() == Ok("1") {
        let before = worker.cuda.admission_memory()?;
        let (files, cached, left) = crate::memory::release_file_cache(&p.kernels)?;
        eprintln!("load file cache released {}", json!({"rank":p.rank,"files":files,"cached_bytes":cached,"left_bytes":left,
            "memory_before":before,"memory_after":worker.cuda.admission_memory()?}));
    }
    let image_pool = match image_pool_mib {
        Some(mib) => { let report = worker.cuda.create_image_pool(mib << 20)?; eprintln!("image pool {report}"); Some(report) },
        None => None,
    };
    #[cfg(feature = "serving")]
    if warm_real_pool {
        ensure!(wide_startup_warmup.is_some() && worker.cuda.wide16_configured() && image_pool.is_some(),
            "real-pool warmup requires wide startup warmup and an elastic 16K profile");
        let free = worker.cuda.image_pool_free_chunks();
        // This public Begin follows the same borrow/release path as a served request.
        // The final fixture is moved here from the temporary-pool warmup.
        let report = worker.warm_wide_before_ready(p.rank, &p.wide_widths, WideWarmupPass::RealPool)?;
        ensure!(!worker.wide16_held && !worker.cuda.wide16_mapped() && worker.cuda.image_pool_free_chunks() == free,
            "real-pool warmup retained an image or an arena loan");
        eprintln!("wide16 real pool warmup {}", json!({"rank":p.rank, "free_chunks":free, "warmup":report}));
    }
    // Disk tier (config::disk_tier): files of an earlier process are worthless (their descriptors named its regions).
    worker.disk_tier = crate::config::disk_tier()?;
    if let Some(tier) = worker.disk_tier.as_ref() {
        crate::disk_tier::reset_dir(&tier.dir)?;
        worker.disk_tier_started = Instant::now();
        eprintln!("disk tier {}", json!({"event":"enabled","dir":tier.dir,"cap_gib":tier.cap_bytes>>30,
            "free_gib":crate::disk_tier::free_bytes(&tier.dir)?>>30,"park_after":tier.park_after_requests}));
    }
    crate::cuda::SERVING_STARTED.store(true, std::sync::atomic::Ordering::Relaxed);
    worker.progress_next = progress_next;
    let mut ready = json!({"type":"ready","rank":p.rank,"identity":worker.identity,
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
        "session_cache_shared_ancestors":worker.session_cache_shared_ancestors,
        "engram_staging":worker.verify.engram_input_kind(),
        "paired_engram_io":worker.verify.paired_engram_io_enabled(),
        "session_prepare_seconds":worker.session_prepare_seconds,
        "startup_warmup":startup_warmup,
        "history_verifier":worker.long_verify.as_ref().map(|target|json!({
            "compact_history_sampling":target.compact_history_sampling_enabled(),
            "compact_index_layers":target.grouped16_index_layers(),
            "query_overlap_layers":target.query_overlap_layers(),
            "coefficient_overlap_layers":target.attention_coefficient_overlap_layers()})),
        "history32_verifier":worker.copy32_admission,
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
    ready["wide_startup_warmup"]=json!(wide_startup_warmup);
    ready["idle_anchor_enabled"]=json!(worker.idle_anchor_enabled);
    ready["checkpoint_ladder"]=json!(worker.checkpoint_ladder);
    ready["memory_floor_bytes"]=json!(crate::memory::floor_bytes());
    ready["prefill_layer20_defer"]=json!(crate::target::layer20_defer_enabled()?);
    ready["encoder_checkpoint_inline"]=json!(crate::config::encoder_checkpoint_inline_requested()?);
    ready["displaced_checkpoint"]=json!(worker.displaced_checkpoint_enabled);
    ready["displaced_snapshot"]=json!(worker.displaced_snapshot);
    ready["retained_append32"]=json!(worker.retained_append32);
    ready["append_final_row_gather"]=json!(worker.verify.append_last_gather_enabled());
    ready["ffn_hint_chunks"]=json!(worker.verify.ffn_hint_chunks());
    ready["prefill_encoded_head"]=json!(worker.large.prefill_encoded_head_enabled());
    ready["prefill_row_head"]=json!(worker.large.prefill_row_head_enabled());
    ready["moe_wide_transfer"]=json!(crate::moe::wide_transfer_requested()?);
    ready["moe_prefill_transfer"]=json!(crate::moe::prefill_transfer_requested()?);
    ready["prefill_upper_rows"]=json!(crate::target::prefill_upper_rows_requested()?);
    ready["prefill_woa_live"]=json!(crate::woa_medium::prefill_live_policy()?);
    ready["prefill_sparse_publish"]=json!(worker.large.prefill_sparse_publish_enabled());
    ready["append_encoded_head"]=json!({
        "requested":crate::config::append_encoded_head_requested()?,
        "verify16":worker.long_verify.as_ref().is_some_and(|t|t.append_encoded_head_enabled()),
        "verify32":worker.copy32_verify.as_ref().is_some_and(|t|t.append_encoded_head_enabled()),
        "private_input_bytes_per_enabled_profile":6*5120*2,
        "additional_weight_bytes":0});
    ready["control_kernel_poll_us"]=json!(crate::control_channel::kernel_poll_us_requested()?);
    ready["idle_anchor_rolling"]=json!(worker.idle_anchor_rolling);
    ready["encoder_image_recycle"]=json!(worker.encoder_image_recycle);
    ready["session_cache_slots"]=json!(worker.session_cache_slots);
    ready["image_pool"]=json!(image_pool);
    if let Ok(f)=crate::attention::kv_format() {
        ready["kv_format"]=json!({"window":f.window.name(),"compressed":f.compressed.name()});
    }
    ready["state_copy_kernel"]=json!(crate::state_image::copy::enabled());
    ready["bank_encoder_checkpoint"]=json!(crate::config::bank_encoder_checkpoint_requested());
    ready["resident_banks"]=resident_banks;
    ready["batched_decode"]=batched_drafters;
    ready["parking_extra_slots"]=json!(worker.parking_extra_slots);
    ready["physical_execution_banks"]=json!(if worker.peer_bank.is_some(){2}else{1});
    ready["execution_graph_cache_rows"]=json!(crate::config::execution_graph_cache_rows()?);
    if let Ok(fixture)=std::env::var("CANADA_SCRATCH_VERIFIER_NUMERICS_PROBE") {
        verifier_numerics_probe::run(&mut worker,p.rank,&fixture)?;
    }
    events.record(&ready)?;
    if let Some(queue) = &queue {
        if http_interleave {
            // Configure lifetime admission before HTTP can advertise Ready.
            let (two_stream, timing) = interleave_http::http_contract(interleave_http::two_stream_enabled()?, batched_serving::enabled()?);
            if two_stream { queue.enable_two_stream_admission(); }
            if timing { queue.enable_execution_timing(); }
            if request_checkpoint::exchange_enabled()? {
                ensure!(interleave_http::two_stream_enabled()?,"state exchange requires two-owner admission");
                queue.reserve_waiting_image_capacity(worker.shared_image_budget)?;
            }
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
    let mut next_id = worker.history.next_request_id()?;
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
                } else if let Some(mut job) = queue.next(worker.idle_wait())? {
                    let tokens = std::mem::take(&mut job.tokens);
                    let id = next_id;
                    next_id = next_id
                        .checked_add(1)
                        .ok_or_else(|| anyhow::anyhow!("request counter exhausted"))?;
                    if !job.images.is_empty() {
                        worker.media.as_mut().ok_or_else(|| anyhow::anyhow!("image request without native vision"))?
                            .set_job_inputs(&mut job)?;
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
            let early_sent = event["early_first_token"] == true;
            if let Some(job) = &active {
                if began && !input_refused {
                    job.record_prefill_timing(&event);
                    if !early_sent { job.emit(BackendEvent::Ready {
                        fingerprint: worker.identity.clone(),
                        cached_tokens: event["cached_tokens"].as_u64().unwrap() as usize,
                    }); }
                    event["queue_and_prefill_seconds"] =
                        json!(job.enqueued.elapsed().as_secs_f64());
                }
                if input_refused {
                    job.emit(BackendEvent::Failed(event["message"].as_str()
                        .ok_or_else(||anyhow::anyhow!("input refusal message missing"))?.into()));
                } else if event["type"] == "tokens" {
                    if !early_sent { job.emit(BackendEvent::Tokens(serde_json::from_value(
                        event["output_ids"].clone(),
                    )?)); }
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
    fn near_end_of_a_prompt() {
        use super::near_end_of;
        // Nov 18 traced stress panel: the prompt-end image 974,060 and its near-end checkpoint 974,059 (K=1).
        let prompt=(0..974_060u32).map(|i|i%129_000).collect::<Vec<_>>();
        assert!(near_end_of(&prompt,&prompt[..974_059]));
        assert!(near_end_of(&prompt,&prompt[..974_060-127]));
        assert!(!near_end_of(&prompt,&prompt[..974_060-128]),"128 rows or more: not one short suffix");
        assert!(!near_end_of(&prompt,&prompt),"the same prompt is not its own near-end image");
        assert!(!near_end_of(&prompt,&[]));
        assert!(!near_end_of(&prompt[..974_059],&prompt),"longer is never near-end");
        let mut other=prompt[..974_059].to_vec();other[10]+=1;
        assert!(!near_end_of(&prompt,&other),"another conversation of nearly the same length");
    }
    #[test]
    fn shared_prefix_checkpoint_position() {
        use super::{common_prefix,shared_prefix_checkpoint as at};
        // s1509 shp1: a 14,731-token prompt whose first 14,250 tokens open another conversation. Cold: the
        // ordinary image would sit at 14,592, inside the task; the shared end rounds down to 14,208.
        assert_eq!(at(14250,0,14592,14731,512),Some(14208));
        // The third conversation restores that image and keeps the ordinary near-end position.
        assert_eq!(at(14250,14208,14729,14730,512),None);
        // The ordinary position already lies inside the shared part: nothing to change.
        assert_eq!(at(14700,0,14592,14731,512),None);
        assert_eq!(at(14592,0,14592,14731,512),None);
        // Too little shared (under 1,024 rows of gain, or under the checkpoint minimum).
        assert_eq!(at(1000,0,1280,1500,512),None);
        assert_eq!(at(1100,0,1152,1300,512),Some(1024));
        assert_eq!(at(1100,0,1152,1300,8192),None);
        // A long task after a short shared part: the next turn of this conversation would re-prefill too much.
        assert_eq!(at(2100,0,30080,30300,512),None);
        assert_eq!(at(8200,0,10240,10400,512),Some(8192));
        assert_eq!(at(8200,0,10368,10500,512),None);
        // Never more than 4,096 rows of cost, whatever the gain.
        assert_eq!(at(100_000,0,104_064,104_200,512),Some(99_968));
        assert_eq!(at(100_000,0,104_192,104_330,512),None);
        // From a restored image: a deeper shared part (project context after the system prompt) counts by what
        // it adds beyond the image.
        assert_eq!(at(19_400,14_208,19_899,19_900,512),Some(19_328));
        assert_eq!(at(14_900,14_208,19_899,19_900,512),None);
        // A legal checkpoint leaves 128 rows or more after it.
        assert_eq!(at(14_250,13_000,14_300,14_301,512),None);
        assert_eq!(at(14_250,13_000,14_335,14_336,512),Some(14_208));
        // Level 2, a cold prompt alone: the end of its system part, under the stricter rule.
        use super::system_prefix_checkpoint as sys;
        assert_eq!(sys(14_247,14_592,14_731,512),Some(14_208));
        assert_eq!(sys(14_247,15_488,15_700,512),None);     // the task costs more than 1,024 rows
        assert_eq!(sys(14_247,15_232,15_400,512),Some(14_208));
        assert_eq!(sys(2_100,2_304,2_500,512),Some(2_048));  // gain 2,048, cost 256
        assert_eq!(sys(2_100,2_432,2_600,512),None);         // cost 384 is more than an eighth of the gain
        assert_eq!(sys(1,1_920,2_100,512),None);             // no system part
        assert_eq!(sys(900,1_024,1_200,512),None);           // under 1,024 rows of gain
        assert_eq!(sys(9_000,8_960,9_100,512),None);         // the ordinary image already lies inside it
        assert_eq!(sys(14_247,14_592,14_731,8192),Some(14_208));
        assert_eq!(sys(4_200,4_352,4_500,8192),None);        // below the checkpoint minimum
        let a=(0..10_000u32).collect::<Vec<_>>();
        let mut b=a.clone();assert_eq!(common_prefix(&a,&b),10_000);
        assert_eq!(common_prefix(&a,&b[..9000]),9000);assert_eq!(common_prefix(&a[..5],&b),5);
        for at in [0,1,4095,4096,4097,8191,8192,9999] {
            b=a.clone();b[at]=u32::MAX;assert_eq!(common_prefix(&a,&b),at);assert_eq!(common_prefix(&b,&a),at);
        }
        assert_eq!(common_prefix(&[],&a),0);
    }
    #[test]
    fn short_and_long_checkpoints_keep_their_shares() {
        use super::parse_short_reserve;
        // the ladder cap off (0): these cases are about the short-checkpoint shares only
        let plan=|e:&[(usize,usize)],a:Option<&[bool]>,i:(usize,usize),b:usize,l:bool,s:usize,r:usize|
            super::cache_evictions_ladder_reserved(e,a,i,b,l,s,r,0);
        let (long,short)=((15_000usize,40usize),(2_500usize,30usize));
        let full=vec![long;8];
        // No reserve: the ordinary policy (oldest first) whatever the kinds.
        assert_eq!(plan(&full,None,short,10_000,false,8,0),Some(vec![0]));
        // A cache full of long entries: a short image takes the oldest long one while short entries hold
        // fewer than their two slots...
        assert_eq!(plan(&full,None,short,10_000,false,8,2),Some(vec![0]));
        let one=[long,long,short,long,long,long,long,long];
        assert_eq!(plan(&one,None,short,10_000,false,8,2),Some(vec![0]));
        // ...and the oldest short one once they hold both, however old the long entries are.
        let two=[long,long,short,long,long,short,long,long];
        assert_eq!(plan(&two,None,short,10_000,false,8,2),Some(vec![2]));
        // A long image never takes the newest two short entries, even when they are the oldest entries...
        let old_shorts=[short,short,long,long,long,long,long,long];
        assert_eq!(plan(&old_shorts,None,long,10_000,false,8,2),Some(vec![2]));
        assert_eq!(plan(&old_shorts,None,long,10_000,false,8,0),Some(vec![0]));
        // ...but a third, older short entry is beyond the share.
        let three=[short,short,short,long,long,long,long,long];
        assert_eq!(plan(&three,None,long,10_000,false,8,2),Some(vec![0]));
        // Only short entries and nothing else to take: a long image is refused rather than emptying the share.
        assert_eq!(plan(&[short,short],None,long,10_000,false,2,2),None);
        // Bytes: a short image that does not fit takes entries by the same rule until it does (300 bytes held;
        // after the oldest short entry the short ones hold one slot again, so the next is the oldest of any kind).
        assert_eq!(plan(&two,None,(2_500,60),330,false,9,2),Some(vec![2]));
        assert_eq!(plan(&two,None,(2_500,60),320,false,9,2),Some(vec![2,0]));
        assert_eq!(parse_short_reserve(None,16).unwrap(),0);
        assert_eq!(parse_short_reserve(Some("4"),16).unwrap(),4);
        assert_eq!(parse_short_reserve(Some("8"),16).unwrap(),8);
        assert!(parse_short_reserve(Some("9"),16).is_err());
        assert!(parse_short_reserve(Some("x"),16).is_err());
    }
    #[test]
    fn short_checkpoints_never_displace_long_ones() {
        // (prefix tokens, charge): two long conversations and two short checkpoints, oldest first.
        let entries=[(900_000usize,10usize),(4_480,1),(20_000,5),(5_094,1)];
        let plan=|checkpoint,budget,slots,p|super::short_checkpoint_plan(&entries,checkpoint,1,Some(budget),Some(slots),p);
        // A long incoming checkpoint keeps whatever plan the ordinary policy made, and protects nothing extra.
        assert_eq!(plan(8_192,100,5,Some(vec![0,1])),(Some(vec![0,1]),Vec::<usize>::new()));
        // A short incoming checkpoint keeps a plan that names short entries only; the long ones come back protected.
        assert_eq!(plan(5_760,100,5,Some(vec![1,3])),(Some(vec![1,3]),vec![0,2]));
        assert_eq!(plan(5_760,100,5,Some(Vec::new())),(Some(Vec::new()),vec![0,2]));
        // The ordinary plan named a long entry: the oldest short entries make the room instead.
        assert_eq!(plan(5_760,100,4,Some(vec![2])),(Some(vec![1]),vec![0,2]));   // slots: four entries, need three
        assert_eq!(plan(5_760,100,3,Some(vec![0,2])),(Some(vec![1,3]),vec![0,2]));
        assert_eq!(plan(5_760,16,5,Some(vec![2])),(Some(vec![1,3]),vec![0,2]));  // bytes: 17 held, 16 allowed with it
        // Short entries cannot make enough room: refused, nothing evicted.
        assert_eq!(plan(5_760,100,2,Some(vec![0,2])),(None,vec![0,2]));
        assert_eq!(plan(5_760,14,5,Some(vec![2])),(None,vec![0,2]));
        assert_eq!(plan(5_760,100,5,None),(None,vec![0,2]));
        // The ordinary policy with full slots protects one longer entry (three slots) and evicts the oldest of the
        // rest: the 20,000-token conversation here. A short checkpoint is refused instead (no short entry to recycle).
        let ordinary=super::encoder_checkpoint_evictions_ladder(&[(900_000,10),(20_000,5),(12_000,5)],&[false;3],None,
            (5_760,1),None,1_000,true,3);
        assert_eq!(ordinary,Some(vec![1]));
        assert_eq!(super::short_checkpoint_plan(&[(900_000,10),(20_000,5),(12_000,5)],5_760,1,Some(1_000),Some(3),ordinary).0,None);
        // The other direction: a long incoming image takes the short checkpoints first (CachedSession::yields_to
        // marks them optional for it), shortest first, before the ordinary policy touches a long entry.
        let yields=[false,true,false,true];
        assert_eq!(super::cache_evictions_with_optional_ladder(&entries,&yields,None,(30_000,5),1_000,true,4),Some(vec![1]));
        assert_eq!(super::cache_evictions_with_optional_ladder(&entries,&yields,None,(30_000,5),1_000,true,3),Some(vec![1,3]));
        // With a long entry older than the short ones, the ordinary policy alone would evict that long entry
        // (the oldest unprotected); with the marking the shortest short checkpoint goes instead.
        let older=[(900_000usize,10usize),(20_000,5),(4_480,1),(5_094,1)];
        assert_eq!(super::cache_evictions_with_optional_ladder(&older,&[false;4],None,(30_000,5),1_000,true,4),Some(vec![1]));
        assert_eq!(super::cache_evictions_with_optional_ladder(&older,&[false,false,true,true],None,(30_000,5),1_000,true,4),Some(vec![2]));
    }
    #[test]
    fn shared_image_budget_follows_the_owner_layout_and_exchange() {
        let gib=1024usize.pow(3);
        assert_eq!(shared_image_limit(true,None),8*gib);
        assert_eq!(shared_image_limit(false,None),2*gib);
        for ok in [64*1024usize.pow(2),4*gib,7*gib,8*gib] {assert!(validate_shared_image_budget(ok,true,false,None).is_ok(),"{ok}");}
        assert!(validate_shared_image_budget(8*gib+1,true,false,None).is_err());
        assert!(validate_shared_image_budget(32*1024usize.pow(2),true,false,None).is_err());
        assert!(validate_shared_image_budget(2*gib,false,false,None).is_ok());
        assert!(validate_shared_image_budget(3*gib,false,false,None).is_err(),"two banks keep 2 GiB");
        assert!(validate_shared_image_budget(2*gib,true,true,None).is_ok());
        assert!(validate_shared_image_budget(4*gib,true,true,None).is_err(),"exchange lends at most 2 GiB to waiting inputs");
        // the snapshot pool raises one bank's ceiling to its own size, never lowers it, and leaves two banks alone
        assert_eq!(shared_image_limit(true,Some(12*1024)),12*gib);
        assert_eq!(shared_image_limit(true,Some(4*1024)),8*gib);
        assert_eq!(shared_image_limit(false,Some(12*1024)),2*gib);
        assert!(validate_shared_image_budget(11*gib,true,false,Some(12*1024)).is_ok());
        assert!(validate_shared_image_budget(12*gib+1,true,false,Some(12*1024)).is_err());
        assert!(validate_shared_image_budget(4*gib,true,true,Some(12*1024)).is_err(),"exchange still lends at most 2 GiB");
        assert_eq!(cache_budget_ceiling(None),8*gib);
        assert_eq!(cache_budget_ceiling(Some(10*1024)),10*gib);
    }
    #[test]
    fn request_incident_redaction_keeps_cache_counts_without_output_tokens() {
        let mut event=serde_json::json!({"type":"tokens","request_id":7,
            "request_started":true,"output_ids":[41,42],"pending":43,
            "paired_peer":{"request_id":8,"output_ids":[45,46],"pending":47,"frontier":123905},
            "prompt_tokens":124000,"cached_tokens":123904,
            "retained_session_cache":{"hit":true,"exact_prompt_anchor":44,
                "restored_tokens":123904},
            "prompt_checkpoint":{"saved":false,"late_memory_refused":true}});
        super::redact_http_event(&mut event);
        assert!(event.get("output_ids").is_none() && event.get("pending").is_none());
        assert!(event["paired_peer"].get("output_ids").is_none()&&event["paired_peer"].get("pending").is_none());
        assert_eq!(event["paired_peer"]["frontier"],123905);
        assert!(event["retained_session_cache"].get("exact_prompt_anchor").is_none());
        assert_eq!(event["retained_session_cache"]["exact_prompt_hit"],true);
        assert_eq!(event["cached_tokens"],123904);
        assert_eq!(event["prompt_checkpoint"]["late_memory_refused"],true);
        assert_eq!(event["request_started"],true);
        let mut batched=serde_json::json!({"type":"batched_tokens","conversations":[
            {"request_id":7,"output_ids":[41],"pending":42,"frontier":10},
            {"request_id":9,"output_ids":[51,52],"pending":53,"frontier":20}]});
        super::redact_http_event(&mut batched);
        assert!(batched["conversations"].as_array().unwrap().iter()
            .all(|c|c.get("output_ids").is_none()&&c.get("pending").is_none()&&c["frontier"].is_u64()));
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
    #[test]
    fn completed_siblings_cannot_displace_their_only_reusable_parents() {
        // Observed545K case: exact-prompt image is borrowed while restoring a
        // regeneration. Saving the old completed child would evict its encoder
        // parent under the unchanged2GiB budget. Decline that optional copy.
        let encoder=(545024,928409100);let prompt=(545215,934905984);
        let completed=(545355,934723760);let budget=2147483648;
        assert_eq!(cache_evictions_with_optional(&[encoder],&[false],completed,
            budget-prompt.1,true,3),Some(vec![0]));
        assert!(completed_cache_evictions(&[encoder],&[false],&[true],completed,
            budget-prompt.1,true,3).is_none());
        // A later short request must not turn the two retained parents into
        // two mutually divergent completions that both miss the original input.
        assert!(completed_cache_evictions(&[encoder,prompt],&[false,true],&[true,true],
            (545342,934416108),budget,true,4).is_none());
        assert_eq!(completed_cache_evictions(&[encoder,prompt],&[false,true],&[false,false],
            (367,12000000),budget,true,4),Some(vec![]));
    }
    #[test]
    fn displaced_snapshot_ignores_only_bank_retained_entries_when_asked() {
        // Ordinary policy: any same-prefix entry counts as already cached.
        assert!(keeps_prefix(true,false)&&keeps_prefix(false,false));
        // A displacement by another conversation: only entries that survive
        // the bank's reuse (complete or independent encoder images) count.
        assert!(!keeps_prefix(true,true)&&keeps_prefix(false,true));
    }
    #[test]
    fn checkpoint_ladder_spreads_rungs_over_one_growing_conversation() {
        let (charge,budget,slots)=(77usize<<20,2usize<<30,8);
        let grow=|ladder:bool|{
            let mut cache:Vec<(usize,usize)>=Vec::new();
            for k in 1..=150usize {
                let incoming=(k*4096,charge);
                let ancestors=vec![true;cache.len()];
                let mut removed=cache_evictions_ladder(&cache,ladder.then_some(&ancestors[..]),incoming,
                    budget,true,slots).unwrap();
                removed.sort_unstable();
                for i in removed.into_iter().rev(){cache.remove(i);}
                cache.push(incoming);
            }
            let mut positions=cache.iter().map(|e|e.0).collect::<Vec<_>>();positions.sort_unstable();positions
        };
        // The ordinary policy keeps only the eight latest checkpoints, so an
        // edit before them re-prefills everything (mean waste 0.46 of the
        // conversation for a uniform edit). The ladder keeps the same two latest
        // rungs and spreads the rest (mean waste 0.13); values match an
        // independent Python model of this policy.
        assert_eq!(grow(false),(143..=150).map(|k|k*4096).collect::<Vec<_>>());
        assert_eq!(grow(true),vec![40960,106496,172032,270336,532480,606208,610304,614400]);
    }
    #[test]
    fn checkpoint_ladder_changes_only_which_ancestor_is_evicted() {
        let near=[(10000,10),(12000,10),(30000,10),(31000,10),(40000,10)];
        assert_eq!(cache_evictions(&near,(50000,10),1000,true,5),Some(vec![0]));
        assert_eq!(cache_evictions_ladder(&near,None,(50000,10),1000,true,5),Some(vec![0]));
        assert_eq!(cache_evictions_ladder(&near,Some(&[false;5]),(50000,10),1000,true,5),Some(vec![0]));
        // 30000 is nearly covered by 31000; the latest two rungs never go.
        assert_eq!(cache_evictions_ladder(&near,Some(&[true;5]),(50000,10),1000,true,5),Some(vec![2]));
        // The oldest entry belongs to another conversation: ordinary victim.
        let foreign=[(5000,10),(10000,10),(20000,10),(30000,10),(40000,10)];
        assert_eq!(cache_evictions_ladder(&foreign,Some(&[false,true,true,true,true]),(50000,10),1000,true,5),
            Some(vec![0]));
        // A duplicate position costs nothing to drop.
        let duplicate=[(10000,10),(20000,10),(20000,10),(30000,10),(40000,10)];
        assert_eq!(cache_evictions_ladder(&duplicate,Some(&[true;5]),(50000,10),1000,true,5),Some(vec![1]));
        // Fewer than three rungs: ordinary victim. Mismatched masks refuse.
        assert_eq!(cache_evictions_ladder(&[(10000,10),(20000,10)],Some(&[true;2]),(50000,10),1000,true,2),
            Some(vec![0]));
        assert!(cache_evictions_ladder(&near,Some(&[true;4]),(50000,10),1000,true,5).is_none());
        // Longer protected parents stay protected and are never rungs.
        let longer=[(10000,10),(12000,10),(30000,10),(90000,10),(31000,10),(40000,10)];
        let removed=cache_evictions_ladder(&longer,Some(&[true,true,true,false,true,true]),(50000,10),1000,true,6);
        assert_eq!(removed,Some(vec![2]));
    }
    #[test]
    fn completed_parent_reservation_preserves_bounds_and_unrelated_evictions() {
        let entries=[(100000,100),(100200,100),(9000,40)];
        let optional=[false,true,false];let parents=[true,true,false];
        assert_eq!(completed_cache_evictions(&entries,&optional,&parents,(100300,50),
            250,true,3),Some(vec![2]));
        assert!(completed_cache_evictions(&entries,&optional,&parents,(100300,60),
            250,true,3).is_none());
        assert!(completed_cache_evictions(&entries,&optional,&parents,(100300,1),
            300,true,2).is_none());
        assert!(completed_cache_evictions(&entries,&optional,&parents[..2],(1,1),
            300,true,4).is_none());
        assert_eq!(completed_cache_evictions(&entries,&optional,&[false;3],(100300,80),
            300,true,4),cache_evictions_with_optional(&entries,&optional,(100300,80),300,true,4));
        // Physical-pressure planning excludes reserved parents without
        // crediting their bytes as reclaimed memory.
        assert_eq!(pressure_cache_victim(&entries,&optional,&[true,true,true],100300,
            4,true,&[0,1]),Some(2));
        assert_eq!(pressure_cache_victim(&entries,&optional,&[true,true,true],100300,
            4,true,&[0,1,2]),None);
    }
    use super::*;
    #[test]
    fn cost_order_loses_dense_checkpoints_before_a_long_conversation() {
        // oldest first: a long conversation L (shared 45K base, then grown) with two rungs, then an image conversation F
        // over the same base with four checkpoints ~1K apart
        let base=vec![7u32;45_000];
        let mut l1=base.clone();l1.extend(vec![1u32;600_000]);let mut l2=l1.clone();l2.extend(vec![1u32;300_000]);
        let mut f=Vec::new();let mut cur=base.clone();
        for k in 0..4 {cur.extend(vec![2u32+k;1_000]);f.push(cur.clone());}
        let prefixes=[l1.as_slice(),l2.as_slice(),f[0].as_slice(),f[1].as_slice(),f[2].as_slice(),f[3].as_slice()];
        let order=super::eviction_cost_order(&prefixes);
        // F's older checkpoints (rungs: 1K over their predecessor, then F's first 46K) go first, then L's older rung,
        // then the newest entries by cost: F's newest, and L2 (the long conversation's newest) last.
        assert_eq!(order,vec![3,4,2,0,5,1]);
        // replaced follow-ups of A (siblings over A's cached prompt) go before D's only entry
        let a=vec![3u32;15_000];let mut a1=a.clone();a1.extend(vec![4u32;300]);let mut a2=a.clone();a2.extend(vec![5u32;300]);
        let d=vec![9u32;15_000];
        let p=[d.as_slice(),a.as_slice(),a1.as_slice(),a2.as_slice()];
        let o=super::eviction_cost_order(&p);
        assert_eq!(&o[..2],&[2,3]);
        // then D's only entry and A's prompt cost the same (their whole length): the older goes first
        assert!(!o[..2].contains(&0));
        // a stale entry costs nothing
        let many=(0..40).map(|k|vec![k as u32+100;10]).collect::<Vec<_>>();
        let refs=many.iter().map(|v|v.as_slice()).collect::<Vec<_>>();
        assert_eq!(super::eviction_cost_order(&refs)[0],0);
    }

    #[test]
    fn held_parent_is_charged_to_the_cache_budget_too() {
        const G:usize=1<<30;
        // production's budgets: cache 6 GiB, shared 7 GiB, 16 slots; nothing parked
        assert_eq!(super::cache_capacity_of(6*G,7*G,0,0,16,0,0),(6*G,16));
        // a held ~0.9 GiB parent: the room is 6 GiB minus the parent, not min(6, 7 - 0.9) = 6 GiB
        assert_eq!(super::cache_capacity_of(6*G,7*G,0,0,16,G-G/10,1),(6*G-(G-G/10),15));
        // a parked image takes from the shared budget; the parent from what is left
        assert_eq!(super::cache_capacity_of(6*G,7*G,2*G,1,17,G/2,1),(5*G-G/2,15));
        // saturating: nothing is left rather than wrapping
        assert_eq!(super::cache_capacity_of(G,7*G,0,0,1,2*G,2),(0,0));
        // the P3 case: 14 entries (5.24 GiB) with the parent among them; the parent is checked out and the displaced
        // conversation's ~1 GB image is saved before the parent goes back
        let parent=950_000_000usize;
        let mut entries=vec![(974_000usize,parent)];
        for k in 0..13 {entries.push((50_000+k*1000,(5_625_025_028-parent)/13));}
        let held=entries.remove(0);
        let displaced=(974_587usize,1_010_000_000usize);
        let (budget,slots)=super::cache_capacity_of(6*G,7*G,0,0,16,held.1,1);
        let optional=vec![false;entries.len()];let ancestors=vec![false;entries.len()];
        let plan=super::completed_cache_evictions(&entries,&optional,&ancestors,displaced,budget,true,slots).unwrap();
        let kept=(0..entries.len()).filter(|i|!plan.contains(i)).map(|i|entries[i].1).sum::<usize>();
        let count=entries.len()-plan.len()+1;
        // the parent still fits when it returns: one slot and its bytes are left
        assert!(count<16&&kept+displaced.1+held.1<=6*G,"{kept} {count}");
        assert!(!plan.is_empty());
    }

    #[test]
    fn cost_order_keeps_each_conversations_newest_turn_before_older_rungs() {
        // C13 bar, residency 4 x 980K: four long conversations, each with its turn-0 and turn-1 images (turn 1 extends
        // turn 0 by a few dozen tokens), oldest first. The older rungs go first, every conversation's newest turn stays.
        let mut prefixes=Vec::new();
        for c in 0..4u32 {let base=vec![c+1;90_000];let mut t1=base.clone();t1.extend(vec![77u32;45]);prefixes.push(base);prefixes.push(t1);}
        let refs=prefixes.iter().map(|v|v.as_slice()).collect::<Vec<_>>();
        let order=super::eviction_cost_order(&refs);
        assert_eq!(&order[..4],&[0,2,4,6],"turn-0 rungs first");
        assert!(order[4..].iter().all(|i|i%2==1));
        // a branch point keeps its class after one of its two regenerations goes
        let a=vec![3u32;15_000];let mut a1=a.clone();a1.extend(vec![4u32;300]);let mut a2=a.clone();a2.extend(vec![5u32;300]);
        let d=vec![9u32;15_000];
        let o=super::eviction_cost_order(&[d.as_slice(),a.as_slice(),a1.as_slice(),a2.as_slice()]);
        assert_eq!(o,vec![2,3,0,1]);
    }
    #[test]
    fn ladder_cap_makes_a_busy_conversation_pay_with_its_own_rungs() {
        // Oldest first: an idle long conversation's three rungs, then thirteen checkpoints of another conversation
        // (~1K apart); sixteen slots; the incoming fourteenth checkpoint needs one slot.
        let mut entries=vec![(350_000usize,10usize),(650_000,10),(974_000,10)];
        for k in 0..13 {entries.push((50_000+k*1_000,1));}
        let mut ancestors=vec![false;3];ancestors.extend(vec![true;13]);
        let incoming=(63_000,1);
        // Without the cap (least recently used): the idle conversation's oldest rung goes.
        assert_eq!(super::cache_evictions_ladder_reserved(&entries,Some(&ancestors),incoming,1_000,true,16,0,0),Some(vec![0]));
        // With the cap: one of the busy conversation's own rungs, never its two latest.
        let with_cap=super::cache_evictions_ladder_reserved(&entries,Some(&ancestors),incoming,1_000,true,16,0,4).unwrap();
        assert_eq!(with_cap.len(),1);
        assert!(with_cap[0]>=3&&with_cap[0]<14,"{with_cap:?}");
        // Below the cap nothing changes: three rungs of the incoming conversation, the oldest entry goes.
        let few=vec![(350_000usize,10usize),(650_000,10),(974_000,10),(50_000,1),(51_000,1),(52_000,1)];
        let a=vec![false,false,false,true,true,true];
        assert_eq!(super::cache_evictions_ladder_reserved(&few,Some(&a),(53_000,1),1_000,true,6,0,4),Some(vec![0]));
        assert!(super::parse_ladder_cap(None).unwrap()==0&&super::parse_ladder_cap(Some("4")).unwrap()==4);
        assert!(super::parse_ladder_cap(Some("1")).is_err()&&super::parse_ladder_cap(Some("17")).is_err());
    }
    #[test]
    fn superseded_first_replays_the_four_conversation_evictions() {
        use super::{cache_evictions, keep_latest_earlier_flags, prefer_cheaper, EntryFacts};
        // First run (Nov 22): one conversation's fifth turn saves its prompt-end image (1,012,261 tokens)
        // with the cache full. List order as traced: an earlier completed image, a second conversation's only restore
        // point, two more conversations' checkpoints, three completed images of the turn before, the saving
        // conversation's checkpoint (just restored).
        let entries = [(1012562, 946906572), (1012424, 942265484), (1010214, 940251652), (1012523, 942311448),
            (1012251, 946586352), (1012581, 946929432), (1010349, 944892728), (1012146, 941968180)];
        let planned = cache_evictions(&entries, (1012261, 947126296), 8321499136, true, 16).unwrap();
        assert_eq!(planned, vec![1], "the length rule protects the stale image and names another conversation's restore point");
        // Entry 0 was an earlier checkpoint already; entry 4 (the saving conversation's previous completed image)
        // becomes one when this request restores entry 7: saved at Begin 20, entry 7 restored at Begin 24, 4 extends 7.
        let facts = [(12, 0), (4, 21), (8, 22), (9, 23), (20, 0), (22, 0), (23, 0), (8, 24)].map(|(serial, restored)|
            EntryFacts { serial, restored, shared: false });
        let mut ext = vec![vec![false; 8]; 8];
        ext[4][7] = true;
        let earlier = keep_latest_earlier_flags(&facts, &ext);
        assert_eq!(earlier, vec![false, false, false, false, true, false, false, false]);
        // Classes: earlier checkpoints 0 (entries 0 and 4), never-restored completed images 1 (5 and 6), the rest 2.
        let class = [0u8, 2, 2, 2, 0, 1, 1, 2];
        let mut excluded = vec![false; 8];
        excluded[7] = true;
        let none = vec![false; 8];
        assert_eq!(prefer_cheaper(&entries, &class, &excluded, &none, planned.clone()), vec![0]);
        // An optional victim stays; a victim that is already an earlier checkpoint stays.
        let mut keep = none.clone();
        keep[1] = true;
        assert_eq!(prefer_cheaper(&entries, &class, &excluded, &keep, planned.clone()), vec![1]);
        assert_eq!(prefer_cheaper(&entries, &class, &excluded, &none, vec![4]), vec![4]);
        // Nothing cheaper outside the plan, or only an excluded image: unchanged.
        assert_eq!(prefer_cheaper(&entries, &[2u8; 8], &excluded, &none, planned.clone()), vec![1]);
        let lone = [2u8, 2, 2, 2, 2, 2, 2, 0];
        assert_eq!(prefer_cheaper(&entries, &lone, &excluded, &none, planned.clone()), vec![1]);

        // Third run (Nov 22): the plan needs two victims and names an idle conversation's checkpoint (entry 0)
        // and another conversation's restore point used 4 s before (entry 1). Retained: the saving conversation's
        // previous completed image (entry 5, an earlier checkpoint) and two completed images no request has restored
        // (6 and 7). The restore point is served first and takes entry 5; the idle checkpoint takes entry 6.
        let entries = [(621059, 592355448), (1012424, 942265484), (1012522, 942311444), (1010214, 940251652),
            (1010048, 939404652), (1012212, 946586196), (1012493, 946815160), (1012558, 946906556), (1012145, 941968176)];
        let planned = cache_evictions(&entries, (1012183, 947080416), 8321499136, true, 16).unwrap();
        assert_eq!(planned, vec![0, 1]);
        let class = [2u8, 2, 2, 2, 2, 0, 1, 1, 2];
        let mut excluded = vec![false; 9];
        excluded[8] = true;
        let mut chosen = prefer_cheaper(&entries, &class, &excluded, &[false; 9], planned.clone());
        chosen.sort_unstable();
        assert_eq!(chosen, vec![5, 6]);
        // With one cheap image only, the hottest restore point keeps its place and the idle checkpoint goes.
        let scarce = [2u8, 2, 2, 2, 2, 0, 2, 2, 2];
        let mut chosen = prefer_cheaper(&entries, &scarce, &excluded, &[false; 9], planned);
        chosen.sort_unstable();
        assert_eq!(chosen, vec![0, 5]);

        // Smaller cheap images: as many as free the victim's bytes, or none.
        let small = [(100, 30), (200, 100), (300, 40), (400, 45), (500, 50)];
        let cheap = [0u8, 2, 0, 0, 2];
        assert_eq!(prefer_cheaper(&small, &cheap, &[false; 5], &[false; 5], vec![1]), vec![0, 2, 3]);
        assert_eq!(prefer_cheaper(&small, &cheap, &[false, false, true, false, false], &[false; 5], vec![1]), vec![1]);
        // Two victims never take the same replacement; a class-1 victim may move onto a class-0 image.
        let pair = [(10, 50), (20, 50), (30, 50), (40, 50)];
        let mut both = prefer_cheaper(&pair, &[0u8, 2, 2, 0], &[false; 4], &[false; 4], vec![1, 2]);
        both.sort_unstable();
        assert_eq!(both, vec![0, 3]);
        assert_eq!(prefer_cheaper(&pair, &[0u8, 1, 2, 2], &[false; 4], &[false; 4], vec![1]), vec![0]);
    }
    #[test]
    fn host_reclaim_takes_optional_then_cheap_then_extended_images() {
        use super::host_reclaim_order;
        // The cache when an image input was refused (second run): three conversations' new prompt checkpoints, the
        // replaced checkpoint of one of them (entry 3; its newer checkpoint, entry 2, not restored yet), and a fourth
        // conversation's checkpoint. Nothing optional, nothing earlier or unrestored: only entry 3 may go.
        let none = [false; 5];
        let extended = [false, false, false, true, false];
        assert_eq!(host_reclaim_order(&none, &[2u8; 5], &extended, &none), vec![3]);
        // Groups in order, list order within a group; shared prefixes never; an optional entry counts once.
        let optional = [false, true, false, false, true, false, false];
        let class = [0u8, 0, 2, 1, 2, 2, 1];
        let extended = [true, false, true, false, false, true, false];
        let shared = [false, false, false, false, false, true, false];
        assert_eq!(host_reclaim_order(&optional, &class, &extended, &shared), vec![1, 4, 0, 3, 6, 2]);
        assert!(host_reclaim_order(&[false; 3], &[2u8; 3], &[false; 3], &[false; 3]).is_empty());
        assert!(host_reclaim_order(&[true; 2], &[0u8; 3], &[false; 2], &[false; 2]).is_empty());
    }
    #[test]
    fn cache_trace_is_off_by_default_and_parses_strictly() {
        assert!(!super::parse_cache_trace(None).unwrap()&&!super::parse_cache_trace(Some("0")).unwrap());
        assert!(super::parse_cache_trace(Some("1")).unwrap());
        assert!(super::parse_cache_trace(Some("yes")).is_err()&&super::parse_cache_trace(Some("")).is_err());
    }
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
        // Recency order for a long incoming image (CANADA_SCRATCH_DISPLACED_RECLAIM): without the long protection the
        // victims are the optional entry, then the others oldest first, whatever their length.
        let lru=[(250000,10),(259000,10),(227000,10),(100,1)];
        let none=[false,false,false,false];
        assert_eq!(pressure_cache_victim(&lru,&[false,false,false,true],&none,254000,16,false,&[]),Some(3));
        assert_eq!(pressure_cache_victim(&lru,&[false,false,false,true],&none,254000,16,false,&[3]),Some(0));
        assert_eq!(pressure_cache_victim(&lru,&[false,false,false,true],&none,254000,16,false,&[3,0]),Some(1));
        // The protected pair under the old order: the stale 259K entry stays and the recent 227K one goes.
        assert_eq!(pressure_cache_victim(&lru,&[false,false,false,true],&none,254000,16,true,&[3,0]),Some(2));
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
    fn cache_slots_remain_bounded_without_raising_the_byte_budget() {
        assert_eq!(session_cache_slots(None).unwrap(),4);
        for slots in [4,8,16,24] {
            assert_eq!(session_cache_slots(Some(&slots.to_string())).unwrap(),slots);
            let entries=vec![(2700usize,46usize);slots];
            let victims=cache_evictions(&entries,(2700,46),slots*46,true,slots).unwrap();
            assert_eq!(victims.len(),1,"a full slot set still needs replacement");
            assert!(cache_evictions(&entries,(800000,2049),2048,true,slots).is_none());
        }
        for value in ["0","3","5","9","12","17","32","-1",""] {
            assert!(session_cache_slots(Some(value)).is_err());
        }
        // A larger metadata bound cannot spend capacity held by another owner.
        let held=1300;let budget=2048-held;
        let entries=vec![(20000,300),(25000,300)];
        let victims=cache_evictions(&entries,(30000,400),budget,true,8).unwrap();
        let retained=entries.iter().enumerate().filter(|(i,_)|!victims.contains(i))
            .map(|(_,e)|e.1).sum::<usize>();
        assert!(retained+400+held<=2048);
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
    fn park_budget_refuses_to_spend_parent_and_needed_prefixes() {
        // Representative charges in MiB: retaining the parent and the other
        // unfinished owner leaves too little space for another park.
        let entries=[(800_000,1380),(200_000,370)];
        assert!(completed_cache_evictions(&entries,&[false;2],&[true,true],
            (usize::MAX,370),2048-370,true,5).is_none());
        assert_eq!(cache_evictions(&entries,(usize::MAX,370),2048-370,true,5),Some(vec![0]));
        // An unrelated stale entry can still yield; this is not a blanket
        // no-eviction policy that needlessly serializes independent streams.
        let entries=[(800_000,1000),(20_000,100),(40_000,400)];
        assert_eq!(completed_cache_evictions(&entries,&[false;3],&[true,true,false],
            (usize::MAX,300),1600,true,5),Some(vec![2]));
        assert!(completed_cache_evictions(&entries,&[false;3],&[true,true,false],
            (usize::MAX,600),1600,true,5).is_none());
    }
    #[test]
    fn dedicated_parking_slots_preserve_cache_entries_but_not_excess_bytes() {
        let cache=[(100,30),(200,30),(300,30),(400,30)];
        // One60-byte paused owner plus the new60-byte image fit beside all
        // four cache entries in240bytes. Entry count alone must not evict.
        assert_eq!(cache_evictions(&cache,(usize::MAX,60),240-60,true,5),Some(vec![]));
        assert_eq!(cache_evictions(&cache,(usize::MAX,60),240-60,true,3),Some(vec![0,1]));
        // Dedicated slots confer no new byte entitlement. With only180bytes
        // total, exactly two optional entries must still yield; an oversized
        // request remains refused without touching the saved owner.
        assert_eq!(cache_evictions(&cache,(usize::MAX,60),180-60,true,5),Some(vec![0,1]));
        assert!(cache_evictions(&cache,(usize::MAX,121),180-60,true,5).is_none());
        let eight=(1..=8).map(|n|(n*100,30)).collect::<Vec<_>>();
        assert_eq!(cache_evictions(&eight,(usize::MAX,60),360-60,true,9),Some(vec![]));
        assert_eq!(cache_evictions(&eight,(usize::MAX,60),300-60,true,9),Some(vec![0,1]));
        let mut protected=vec![false;8];protected[7]=true;
        let evicted=completed_cache_evictions(&eight,&[false;8],&protected,
            (usize::MAX,60),300-60,true,9).unwrap();
        assert!(!evicted.contains(&7));
        assert!(completed_cache_evictions(&eight,&[false;8],&[true;8],
            (usize::MAX,60),300-60,true,9).is_none());
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

#[cfg(feature = "serving")]
#[derive(Clone, Copy)]
enum WideWarmupPass { All, BeforePool, RealPool }

/// Wide startup warmup prompts as (offset, length) in the fixture: the original 8,320-token prompt at 0, then every
/// width largest first, each at its own offset so no prompt extends another. Offsets are 10,000 apart while that fits
/// the fixture (unchanged for up to 26 widths); a longer list (C22's 26 widths plus 16,384 needs 270,000 of the
/// fixture's 262,656 tokens) steps the spacing down by 500 until every prompt fits. None if no spacing fits.
///
/// A widest width of 16,384 or more (the 16K extension) also gets one prompt of two such chunks and a 128-token tail,
/// at the fixture's end (its own start): the single-width prompts run as one chunk each, so a real prompt's
/// multi-chunk path at that width (a chunk with followers, layer 20 deferred, the next chunk's lookahead, a chunk after
/// a 16K chunk) first ran in a client's first long request: ky, Nov 20, +3.4 / +6.2 s on the first 32K prompt after
/// load (8K: +0.1 s, its multi-chunk path is warmed by the 8,320-token prompt). Lists without such a width are unchanged.
#[cfg(any(feature = "serving", test))]
fn wide_warmup_prompts(widths: &[usize], fixture: usize) -> Option<Vec<(usize, usize)>> {
    let mut widths_desc = widths.to_vec();
    widths_desc.sort_unstable_by(|a, b| b.cmp(a));
    if widths_desc.iter().any(|&w| w == 0) || 8320 > fixture { return None; }
    let mut prompts = (2..=20).rev().map(|half| half * 500).find_map(|step| {
        let prompts: Vec<(usize, usize)> = std::iter::once((0, 8320))
            .chain(widths_desc.iter().enumerate().map(|(k, &w)| (step * (k + 1), w))).collect();
        prompts.iter().all(|&(at, len)| at + len <= fixture).then_some(prompts)
    })?;
    if let Some(&widest) = widths_desc.first().filter(|&&w| w >= 16384) {
        let len = 2 * widest + 128;
        let mut at = fixture.checked_sub(len)?;
        while prompts.iter().any(|&(start, _)| start == at) { at = at.checked_sub(1)?; }
        prompts.push((at, len));
    }
    Some(prompts)
}

#[cfg(test)]
mod wide_warmup_tests {
    use super::wide_warmup_prompts;
    const FIXTURE: usize = 262_656;
    const C22: [usize; 26] = [256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1792, 2048, 2304, 2560, 2816,
        3072, 3328, 3584, 3840, 4096, 4608, 5120, 6656, 7680, 8192];
    /// The 8,320-token prompt, one prompt of each width, and (a widest width of 16,384 or more) the multi-chunk prompt
    /// last; every prompt inside the fixture at its own start.
    fn check(widths: &[usize], prompts: &[(usize, usize)]) {
        let widest = widths.iter().copied().max().unwrap_or(0);
        let multi = usize::from(widest >= 16384);
        assert_eq!(prompts.len(), widths.len() + 1 + multi);
        assert_eq!(prompts[0], (0, 8320));
        assert!(prompts.iter().all(|&(at, len)| len > 0 && at + len <= FIXTURE));
        let mut starts: Vec<usize> = prompts.iter().map(|p| p.0).collect();
        starts.sort_unstable();
        starts.dedup();
        assert_eq!(starts.len(), prompts.len(), "every prompt starts at its own offset");
        let mut lens: Vec<usize> = prompts[1..prompts.len() - multi].iter().map(|p| p.1).collect();
        let mut want = widths.to_vec();
        lens.sort_unstable();
        want.sort_unstable();
        assert_eq!(lens, want, "every width warms once");
        if multi == 1 { assert_eq!(prompts.last().unwrap().1, 2 * widest + 128, "two widest chunks and a tail"); }
    }
    #[test]
    fn lists_that_fit_keep_the_ten_thousand_spacing() {
        let p = wide_warmup_prompts(&C22, FIXTURE).unwrap();
        check(&C22, &p);
        assert_eq!(p[1], (10_000, 8192));
        assert_eq!(p[26], (260_000, 256));
        let short = [256, 512, 1024, 2048, 4096, 8192, 16384];
        let p = wide_warmup_prompts(&short, FIXTURE).unwrap();
        check(&short, &p);
        assert_eq!(p[1], (10_000, 16384));
    }
    #[test]
    fn the_sixteen_k_list_steps_the_spacing_down() {
        let mut w = C22.to_vec();
        w.push(16384);
        let p = wide_warmup_prompts(&w, FIXTURE).unwrap();
        check(&w, &p);
        assert_eq!(p[1], (9_500, 16384));
        assert_eq!(*p.last().unwrap(), (FIXTURE - 32_896, 32_896), "the multi-chunk prompt ends the fixture");
    }
    #[test]
    fn only_a_sixteen_k_list_gets_the_multi_chunk_prompt() {
        let p = wide_warmup_prompts(&C22, FIXTURE).unwrap();
        assert!(p.iter().all(|&(_, len)| len <= 8320), "C22's list warms exactly as before");
        let p = wide_warmup_prompts(&[8192, 16384], FIXTURE).unwrap();
        check(&[8192, 16384], &p);
        assert_eq!(p, vec![(0, 8320), (10_000, 16384), (20_000, 8192), (FIXTURE - 32_896, 32_896)]);
        // A start already taken (a 42,896-token fixture puts the multi-chunk prompt at 10,000, the 16,384 prompt's
        // start) moves it down by one token.
        let p = wide_warmup_prompts(&[16384], 42_896).unwrap();
        assert_eq!(p, vec![(0, 8320), (10_000, 16384), (9_999, 32_896)]);
    }
    #[test]
    fn no_spacing_fits_gives_none() {
        assert!(wide_warmup_prompts(&[300_000], FIXTURE).is_none());
        assert!(wide_warmup_prompts(&[0], FIXTURE).is_none());
        let many: Vec<usize> = (1..=300).map(|k| 256 + k).collect();
        assert!(wide_warmup_prompts(&many, FIXTURE).is_none());
        // The width fits once, but two of it and a tail do not.
        assert!(wide_warmup_prompts(&[140_000], FIXTURE).is_none());
    }
}

#[cfg(test)]
mod keep_latest_tests {
    use super::{cache_evictions, keep_latest_earlier_flags, keep_latest_order, CachedPrefix, EntryFacts};
    fn ext_of(list: &[&Vec<u32>]) -> Vec<Vec<bool>> {
        let prefixes: Vec<CachedPrefix> = list.iter().map(|v| (*v).clone().into()).collect();
        prefixes.iter().map(|longer| prefixes.iter().map(|shorter| longer.extends(shorter)).collect()).collect()
    }
    fn f(serial: u64, restored: u64) -> EntryFacts { EntryFacts { serial, restored, shared: false } }
    #[test]
    fn follow_ups_branching_from_one_base_keep_the_base() {
        // Warm rounds: base E (near-end of request 11) restored by requests 15, 16, 18; each saved its own prompt-end image.
        let e: Vec<u32> = (0..30_000).collect();
        let leaf = |k: u32| -> Vec<u32> { let mut v = e.clone(); v.extend((0..20 * k).map(|x| 900_000 + 1000 * k + x)); v };
        let (p15, p16, p18) = (leaf(1), leaf(2), leaf(3));
        let list = [&e, &p15, &p16, &p18];
        let facts = [f(11, 18), f(15, 0), f(16, 0), f(18, 0)];
        let earlier = keep_latest_earlier_flags(&facts, &ext_of(&list));
        assert_eq!(earlier, vec![false, true, true, false], "the base stays; leaves saved before its last restore go first");
        assert_eq!(keep_latest_order(&earlier), vec![1, 2, 0, 3]);
    }
    #[test]
    fn a_conversation_that_moved_on_yields_before_an_idle_conversations_only_image() {
        // A: E restored by conc2 (Begin 26), which saved N2 and its answer image C2; conc3 (29) restored N2.
        let e: Vec<u32> = (0..30_000).collect();
        let mut c1 = e.clone(); c1.extend(500_000..500_900u32);           // conc1's answer image (reasoning kept)
        let mut n2 = e.clone(); n2.extend(600_000..600_700u32);           // conc2's near-end image
        let mut c2 = n2.clone(); c2.extend(700_000..701_000u32);          // conc2's answer image
        let mut n3 = n2.clone(); n3.extend(800_000..800_400u32);          // conc3's near-end image
        let d: Vec<u32> = (2_000_000..2_025_000).collect();               // D: restored at warm4 (Begin 24) only
        let list = [&d, &e, &c1, &n2, &c2, &n3];
        let facts = [f(14, 24), f(11, 26), f(25, 0), f(26, 29), f(26, 0), f(29, 0)];
        let earlier = keep_latest_earlier_flags(&facts, &ext_of(&list));
        assert_eq!(earlier, vec![false, true, true, false, true, false]);
        // The planner evicts the oldest eligible entry: in keep-latest order A's left-behind images, never D's.
        let order = keep_latest_order(&earlier);
        let entries: Vec<(usize, usize)> = order.iter().map(|&i| (list[i].len(), 10)).collect();
        let victims = cache_evictions(&entries, (31_000, 10), 1_000, false, 4).unwrap();
        let named: Vec<usize> = victims.iter().map(|&k| order[k]).collect();
        assert!(!named.contains(&0) && named.iter().all(|&i| earlier[i]), "{named:?}");
    }
    #[test]
    fn a_shared_prefix_never_links_two_conversations() {
        let s: Vec<u32> = (0..10_000).collect();
        let mut x = s.clone(); x.extend(50_000..60_000u32);
        let mut y = s.clone(); y.extend(70_000..75_000u32);
        let list = [&s, &x, &y];
        let facts = [EntryFacts { serial: 5, restored: 0, shared: true }, f(6, 9), f(7, 0)];
        let earlier = keep_latest_earlier_flags(&facts, &ext_of(&list));
        assert_eq!(earlier, vec![true, false, false], "X's restore passes the shared prefix but never Y");
    }
    #[test]
    fn images_saved_by_the_restoring_request_stay_latest() {
        // One request restores E and saves its checkpoint, near-end, prompt-end and answer images in the same Begin.
        let e: Vec<u32> = (0..30_000).collect();
        let mut k = e.clone(); k.extend(10_000_000..10_000_300u32);
        let mut ne = k.clone(); ne.extend(11_000_000..11_000_099u32);
        let mut pe = ne.clone(); pe.push(12_000_000);
        let mut ans = pe.clone(); ans.extend(13_000_000..13_000_500u32);
        let list = [&e, &k, &ne, &pe, &ans];
        let facts = [f(11, 30), f(30, 0), f(30, 0), f(30, 0), f(30, 0)];
        assert_eq!(keep_latest_earlier_flags(&facts, &ext_of(&list)), vec![false; 5]);
    }
    #[test]
    fn edited_conversations_keep_their_independent_checkpoints() {
        let unrelated = vec![77; 40_000];
        let original: Vec<u32> = (0..65_536).collect();
        let mut edited = original.clone();
        edited[32_768] += 1;
        edited.push(77);
        let list = [&unrelated, &original, &edited];
        let facts = [f(1, 0), f(2, 0), f(3, 4)];
        let earlier = keep_latest_earlier_flags(&facts, &ext_of(&list));
        assert_eq!(earlier, [false; 3]);
        let order = keep_latest_order(&earlier);
        let entries = order.iter().map(|&i| (list[i].len(), 10)).collect::<Vec<_>>();
        let victim = cache_evictions(&entries, (70_000, 10), 30, true, 16).unwrap();
        assert_eq!(order[victim[0]], 0);

        let prefix: CachedPrefix = original.into();
        let mut extended = prefix.to_vec();
        extended.push(77);
        let longer: CachedPrefix = extended.into();
        let fork: CachedPrefix = edited[..65_536].to_vec().into();
        for _ in 0..2 {
            assert!(longer.extends(&prefix));
            assert!(!longer.extends(&fork));
            assert!(!prefix.extends(&longer) && !prefix.extends(&prefix));
        }
        assert_eq!(longer.comparisons.borrow().len(), 2);
        drop(fork);
        assert!(longer.extends(&prefix[..127].to_vec().into()));
        assert_eq!(longer.comparisons.borrow().len(), 2);
    }
}
