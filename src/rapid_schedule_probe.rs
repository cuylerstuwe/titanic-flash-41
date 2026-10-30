//! Short paired schedule screens sharing one loaded model and causal state.
//! Numerical/state equality and per-call timings decide which ideas merit
//! broader generation/quality work; no automatic optimization promotion.
use crate::{
    cuda::{Buffer, Cuda, Module, Stream},
    roce::Roce,
    target::Target,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

const WARMUP: usize = 2;
const MEASURED: usize = 8;
const NAMES: [&str; 4] = ["control", "engram_overlap", "attention_q_overlap", "both"];

/// Reuse the loaded weights and accepted-state storage, retaining only one
/// candidate workspace at a time. Each candidate gets its own balanced A/B
/// trials against the same all-on control at both fixed prefixes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_ablations(
    cuda:&Cuda,core:&Module,core_bytes:&[u8],large:&mut Target,verify:&mut Target,
    wides:&mut[Target],comm:&Roce,stream:&Stream,kernels:&Path,prefix:&[u32],output:&Path,rank:usize,
)->Result<Value>{
    let cases=[("no_l2","CANADA_SCRATCH_L2_PREFETCH"),
        ("no_shared_ffn_overlap","CANADA_SCRATCH_FFN_OVERLAP"),
        ("no_mhc_overlap","CANADA_SCRATCH_MHC_OVERLAP"),
        ("no_pdl_same_backend","CANADA_SCRATCH_PROBE_PDL_OFF")];
    ensure!(std::env::var("CANADA_SCRATCH_SCHEDULE_ABLATION").as_deref()==Ok("1")
        && cases[..3].iter().all(|(_,flag)|std::env::var(flag).as_deref()==Ok("1"))
        && std::env::var("CANADA_SCRATCH_FP8_PDL").as_deref()==Ok("1")
        && std::env::var_os("CANADA_SCRATCH_PROBE_PDL_OFF").is_none(),
        "schedule ablations require the all-on starting profile");
    verify.apply_candidate_schedule(stream)?;
    let counts=verify.schedule_counts_for_probe();
    ensure!(counts==[117,40,80,40],"unexpected all-on schedule objects: {counts:?}");
    large.enable_cached_extend_floor()?;
    let mut results=Vec::new();
    for(index,(name,flag))in cases.into_iter().enumerate(){
        if index>0{
            verify.reset(81000+index as u64*10,comm,stream)?;
            large.handoff_from(verify)?;
        }
        stream.synchronize()?;
        let previous=std::env::var_os(flag);
        // No CPU background model execution observes these switches. The
        // resulting profile owns every selected stream/kernel/graph object.
        unsafe{if index==3{std::env::set_var(flag,"1");}else{std::env::remove_var(flag);}}
        let built=large.fork_profile(core,core_bytes,stream,kernels,6,crate::compressed_source::Phase::Verify);
        unsafe{if let Some(value)=previous{std::env::set_var(flag,value);}else{std::env::remove_var(flag);}}
        let mut candidate=built?;
        candidate.set_graph_replay(true)?;
        candidate.apply_candidate_schedule(stream)?;
        let mut expected=counts;expected[index]=0;
        ensure!(candidate.schedule_counts_for_probe()==expected,"ablation did not isolate {name}");
        let shared=large.shared_state_report(&candidate);
        ensure!(shared["all_weights_and_causal_storage_shared"]==true,"ablation duplicated model storage");
        let directory=output.join(name);std::fs::create_dir(&directory)?;
        let mut result=run(cuda,large,verify,&mut candidate,wides,comm,stream,prefix,&directory,rank,Some(name))?;
        result["schedule_counts"]=json!([counts,expected]);
        result["ablation_case"]=json!(name);
        result["scope"]=json!("One scheduling feature disabled, canonical precision unchanged; PDL off uses the same FlashInfer backend and tile. Two fixed prefixes, M3/6, cold/warm Engram, one warmup and four balanced pairs. Logits, greedy choices, expert reuse counts and rollback checks. No throughput, outcome or production claim.");
        std::fs::write(directory.join("summary.json"),serde_json::to_vec_pretty(&result)?)?;
        results.push(result);
        stream.synchronize()?;
    }
    Ok(json!({"rank":rank,"schedule_ablation":true,"cases":results,
        "all_exact":results.iter().all(|r|r["all_exact"]==true),"full_engine_acceptance":false}))
}

struct Trial {
    ms: f64,
    barrier_ms: f64,
    host: [f64; 2],
    ids: Vec<u32>,
    digest: Option<String>,
    routes: Option<Value>,
}
fn trial(
    cuda: &Cuda,
    target: &mut Target,
    block: &[u32],
    hash: bool,
    cold: bool,
    engram_overlap: bool,
    comm: &Roce,
    stream: &Stream,
    barrier_in: &Buffer<f32>,
    barrier_out: &mut Buffer<f32>,
    dump: Option<&Path>,
) -> Result<Trial> {
    target.set_engram_overlap_for_probe(engram_overlap)?;
    if cold {
        target.clear_engram_cache_for_probe(stream)?;
    }
    // Hash/route diagnostics and cache clearing are deliberately outside the
    // model interval. Let every peer finish them before its timer starts;
    // otherwise an early rank charges another rank's readback to the next
    // verifier's first collective. Retain this wait as separate evidence.
    let barrier_start = Instant::now();
    stream.join_default()?;
    unsafe { comm.gather_f32(barrier_in, barrier_out, 4, stream)?; }
    stream.synchronize()?;
    comm.health()?;
    let barrier_ms = barrier_start.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    target.begin_verification(block, comm, stream)?;
    let ids = target.verification_greedy(stream)?;
    let ms = start.elapsed().as_secs_f64() * 1000.;
    let host = target.verification_host_ms()?;
    let digest = if hash {
        let (logits, live) = target.pending_logits()?;
        ensure!(live == block.len(), "rapid screen logits geometry");
        let raw=unsafe {cuda.read_device(logits.ptr(),live*129280*4)}?;
        if let Some(path)=dump {
            ensure!(!path.exists(),"immutable logit diagnostic already exists");
            let temporary=path.with_extension("tmp");
            std::fs::write(&temporary,&raw)?;
            std::fs::rename(temporary,path)?;
        }
        Some(format!("{:x}",Sha256::digest(raw)))
    } else {
        None
    };
    let routes=if hash && (std::env::var("CANADA_SCRATCH_MOE_MICRO_AB").as_deref()==Ok("1")
        || std::env::var("CANADA_SCRATCH_MOE_LIVE_AB").as_deref()==Ok("1")
        || std::env::var("CANADA_SCRATCH_MOE_SCALE_AB").as_deref()==Ok("1")
        || std::env::var("CANADA_SCRATCH_COMPUTE_AB").as_deref()==Ok("1")
        || std::env::var("CANADA_SCRATCH_SCHEDULE_ABLATION").as_deref()==Ok("1")) {
        Some(target.expert_routes_for_probe()?)
    } else {None};
    target.finish_verification(0, comm, stream)?;
    Ok(Trial {
        ms,
        barrier_ms,
        host,
        ids,
        digest,
        routes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    cuda: &Cuda,
    large: &mut Target,
    verify: &mut Target,
    candidate: &mut Target,
    wides: &mut [Target],
    comm: &Roce,
    stream: &Stream,
    prefix: &[u32],
    output: &Path,
    rank: usize,
    ablation: Option<&str>,
) -> Result<Value> {
    let sort_screen = std::env::var("CANADA_SCRATCH_SORT_SELECTED_AB").as_deref() == Ok("1");
    let micro_screen = std::env::var("CANADA_SCRATCH_MOE_MICRO_AB").as_deref() == Ok("1");
    let live_moe_screen = std::env::var("CANADA_SCRATCH_MOE_LIVE_AB").as_deref() == Ok("1");
    let scale_screen=std::env::var("CANADA_SCRATCH_MOE_SCALE_AB").as_deref()==Ok("1");
    let compute_screen = std::env::var("CANADA_SCRATCH_COMPUTE_AB").as_deref() == Ok("1");
    let narrow_only = std::env::var("CANADA_SCRATCH_NARROW_ONLY_SCREEN").as_deref() == Ok("1");
    let task_order_only = std::env::var("CANADA_SCRATCH_TASK_ORDER_ONLY_SCREEN").as_deref() == Ok("1");
    ensure!(!task_order_only || (compute_screen && !narrow_only), "task-order-only requires isolated compute comparison");
    ensure!(!narrow_only || compute_screen, "narrow-only screen requires compute comparison");
    let compact_screen=std::env::var("CANADA_SCRATCH_COMPACT_INDEX_AB").as_deref()==Ok("1");
    let sparse_screen = std::env::var("CANADA_SCRATCH_SPARSE_INDEX_AB").as_deref() == Ok("1");
    let adaptive_sparse_screen=std::env::var("CANADA_SCRATCH_ADAPTIVE_SPARSE_AB").as_deref()==Ok("1");
    let fused_qkv_screen = std::env::var("CANADA_SCRATCH_FUSED_QKV_AB").as_deref() == Ok("1");
    let compute_numerical=std::env::var("CANADA_SCRATCH_COMPUTE_NUMERICAL").as_deref()==Ok("1");
    ensure!(!compute_numerical || compute_screen,"numerical capture requires compute comparison");
    let paired_screen = compact_screen || sort_screen || micro_screen || live_moe_screen || scale_screen || compute_screen || sparse_screen || fused_qkv_screen || ablation.is_some();
    let ablation_names=["all_on",ablation.unwrap_or("")];
    let names: &[&str] = if compact_screen {&["expanded_selection","compact_selection"]} else if adaptive_sparse_screen {&["dense_index","adaptive_index"]} else if fused_qkv_screen {&["split_qkv","replicated_qkv"]} else if sparse_screen {&["dense_index","candidate_only_index"]} else if ablation.is_some() {&ablation_names} else if task_order_only { &["current_live", "task_order_only"] } else if narrow_only { &["current_live", "narrow_only"] } else if compute_screen { &["current_live", "narrow_and_task_order"] } else if scale_screen { &["current_live", "fc1_scale_reuse"] } else if live_moe_screen { &["grouped", "live_bounds_micro1"] } else if micro_screen { &["grouped", "micro_fp32"] } else if sort_screen { &["original_sort", "native_shuffle_sort"] } else { &NAMES };
    let warmup = if paired_screen { 1 } else { WARMUP };
    let measured = if narrow_only || compact_screen {2} else if paired_screen { 4 } else { MEASURED };
    let offsets: &[usize] = if paired_screen { &[54] } else { &[54, 438] };
    ensure!(
        prefix.len() == 458
            && large.context_capacity() == if compact_screen {1048576} else if sparse_screen {262144} else {32768}
            && verify.query_overlap_layers() == (if paired_screen { 40 } else { 0 })
            && candidate.query_overlap_layers() == 40
            && std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref() == Ok("1")
            && std::env::var_os("CANADA_SCRATCH_ENGRAM_PREFETCH_AB").is_none()
            && std::env::var_os("CANADA_SCRATCH_ENGRAM_OVERLAP_AB").is_none(),
        "rapid screen requires two known schedules sharing the admitted owner"
    );
    ensure!(!micro_screen || (verify.micro_moe_layers()==0 && candidate.micro_moe_layers()==40), "micro workspace comparison");
    ensure!(!live_moe_screen || (verify.live_moe_layers()==0 && candidate.live_moe_layers()==40),"live workspace comparison");
    ensure!(!scale_screen || (verify.live_moe_layers()==40 && candidate.live_moe_layers()==40
        && verify.scale_reuse_moe_layers()==0 && candidate.scale_reuse_moe_layers()==40),"scale workspace comparison");
    ensure!(!compute_screen || (verify.live_moe_layers() == 40 && candidate.live_moe_layers() == 40
        && verify.task_order_moe_layers() == 0 && candidate.task_order_moe_layers() == if narrow_only {0} else {40}
        && verify.narrow_projection_layers() == 0 && candidate.narrow_projection_layers() == if task_order_only {0} else {40}),
        "compute workspace comparison");
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/fixed-verify-python-v1.json"
    ))?;
    let teacher: Vec<u32> = serde_json::from_value(fixture["teacher_forced_tokens"].clone())?;
    let long: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/target-agentic-long15800-v1.json"
    ))?;
    let mut long: Vec<u32> = serde_json::from_value(long["batches"][0].clone())?;
    ensure!(
        teacher.len() == 512 && long.len() == 15800,
        "rapid screen fixtures"
    );
    long.extend_from_slice(&teacher);
    long.extend_from_slice(&teacher);
    // Use the existing SGLang panel's frozen repository-review corpus.
    // Token slicing is scheduler metadata, not CPU model computation.
    let corpus:Vec<u32>=if sparse_screen || compact_screen {
        let data:Value=serde_json::from_slice(if compact_screen {include_bytes!("../tests/fixtures/long-context-continuation-v2.json").as_slice()}else{include_bytes!("../tests/fixtures/long-context-verification-corpus-v1.json").as_slice()})?;
        ensure!(data["source_sha256"]=="9502643d4f0f1e5ed712073c8596757afad14ed8186ae354ff627a76983ea1fa","long corpus provenance");
        let tokens:Vec<u32>=serde_json::from_value(data["tokens"].clone())?;
        ensure!(tokens.len()==if compact_screen {262656}else{131584} && tokens.iter().all(|&v|v<129280),"long corpus token domain");
        tokens
    } else {Vec::new()};
    let prompts:Vec<Vec<u32>>=if sparse_screen || compact_screen {
        let frontiers:&[usize]=if compact_screen {&[65536,131072,262144,16384]} else if adaptive_sparse_screen {&[16384,65536,131072,16384]}else{&[16384,65536,131072]};
        frontiers.iter().map(|&frontier|
            corpus[..frontier-54].to_vec()).collect()
    } else if narrow_only || task_order_only {vec![long]} else {vec![prefix.to_vec(),long]};
    if ablation.is_none() {large.enable_cached_extend_floor()?;}
    let mut barrier_in = cuda.allocate::<f32>(4)?;
    barrier_in.upload(&[0.; 4])?;
    let mut barrier_out = cuda.allocate::<f32>(16)?;
    let mut records = Vec::new();
    let mut prefill_records=Vec::new();
    for (prompt_index, prompt) in prompts.iter().enumerate() {
        let teacher=if sparse_screen || compact_screen {&corpus[prompt.len()..prompt.len()+512]} else {teacher.as_slice()};
        if prompt_index > 0 {
            // Each ablation owns two fresh lineages within the same load.
            // Later cases must not return to the first case's reset number.
            let base=match ablation {
                Some("no_l2")=>80000,
                Some("no_shared_ffn_overlap")=>81010,
                Some("no_mhc_overlap")=>81020,
                Some("no_pdl_same_backend")=>81030,
                _=>70000,
            };
            verify.reset(base + prompt_index as u64, comm, stream)?;
            large.handoff_from(verify)?;
        }
        let prefill_start=Instant::now();
        large.prefill_wide(wides, prompt, comm, stream)?;
        prefill_records.push(json!({"uncached_tokens":prompt.len(),"seconds":prefill_start.elapsed().as_secs_f64()}));
        verify.handoff_from(large)?;
        for &offset in offsets {
            verify.set_engram_overlap_for_probe(false)?;
            let frontier = prompt.len() + offset;
            while (verify.position() as usize) < frontier {
                let begin = verify.position() as usize - prompt.len();
                verify.append(&teacher[begin..(begin + 6).min(offset)], comm, stream)?;
            }
            for &cold in if narrow_only || task_order_only { &[false][..] } else { &[true, false][..] } {
                    let depths: &[usize] = if narrow_only || task_order_only {&[3,6]} else if adaptive_sparse_screen {&[1,3,6]} else if compact_screen || sparse_screen || fused_qkv_screen || ablation.is_some() {&[3,6]} else if live_moe_screen { &[1,2,3,4,5,6] } else if micro_screen { &[1,2,3,4,6] } else { &[1,3,6] };
                for &live in depths {
                    let block = &teacher[offset..offset + live];
                    let carry = verify.carry_digests()?;
                    let encoder = verify.encoder_tail_digest()?;
                    let mut samples: Vec<Vec<f64>> = (0..names.len()).map(|_| Vec::new()).collect();
                    let mut warmup_samples: Vec<Vec<f64>> = (0..names.len()).map(|_| Vec::new()).collect();
                    let mut host: Vec<Vec<[f64; 2]>> = (0..names.len()).map(|_| Vec::new()).collect();
                    let mut barriers: Vec<Vec<f64>> = (0..names.len()).map(|_| Vec::new()).collect();
                    let mut ids: Vec<Vec<Vec<u32>>> = (0..names.len()).map(|_| Vec::new()).collect();
                    let mut hashes: Vec<Vec<String>> = (0..names.len()).map(|_| Vec::new()).collect();
                    let mut routes: Vec<Vec<Value>> = (0..names.len()).map(|_| Vec::new()).collect();
                    let mut diagnostics=Vec::new();
                    for iteration in 0..warmup + measured {
                        let hash = iteration == warmup - 1 || iteration == warmup + measured - 1;
                        // Balanced rotations include each path in every order
                        // position equally often among measured repetitions.
                        for slot in 0..names.len() {
                            let path = (slot + iteration) % names.len();
                            let diagnose=narrow_only || (compute_numerical && prompt_index==0 && !cold && live==3);
                            let dump=(diagnose&&rank==3).then(||output.join(format!("numeric-f{frontier}-m{live}-i{iteration}-v{path}.f32")));
                            let t = if if paired_screen { path == 1 } else { path >= 2 } {
                                candidate.handoff_from(verify)?;
                                let t = trial(
                                    cuda,
                                    candidate,
                                    block,
                                    hash||diagnose,
                                    cold,
                                    paired_screen || path % 2 == 1,
                                    comm,
                                    stream,
                                    &barrier_in,
                                    &mut barrier_out,
                                    dump.as_deref(),
                                )?;
                                verify.handoff_from(candidate)?;
                                t
                            } else {
                                trial(cuda, verify, block, hash||diagnose, cold, paired_screen || path == 1, comm, stream,
                                    &barrier_in, &mut barrier_out,dump.as_deref())?
                            };
                            if diagnose {
                                diagnostics.push(json!({"iteration":iteration,"warmup":iteration<warmup,"variant":path,
                                    "sha256":t.digest,"predictions":t.ids,"expert_routes":t.routes,
                                    "file":format!("numeric-f{frontier}-m{live}-i{iteration}-v{path}.f32")}));
                            }
                            if iteration >= warmup {
                                samples[path].push(t.ms);
                                barriers[path].push(t.barrier_ms);
                                host[path].push(t.host);
                                ids[path].push(t.ids);
                            } else {
                                warmup_samples[path].push(t.ms);
                            }
                            if hash && let Some(digest) = t.digest {
                                hashes[path].push(digest);
                            }
                            if hash && let Some(value) = t.routes {routes[path].push(value);}
                        }
                    }
                    let state_exact = verify.position() as usize == frontier
                        && verify.carry_digests()? == carry
                        && verify.encoder_tail_digest()? == encoder;
                    let exact: Vec<_> = (0..names.len())
                        .map(|p| {
                            hashes[p] == hashes[0]
                                && hashes[p][0] == hashes[p][1]
                                && ids[p] == ids[0]
                                && state_exact
                        })
                        .collect();
                    let record = json!({"prompt_index":prompt_index,"active_sparse_index_layers":[verify.active_sparse_index_layers(),candidate.active_sparse_index_layers()],"graph_captures":[verify.capture_statistics().1,candidate.capture_statistics().1],"prefix_tokens":prompt.len(),"frontier":frontier,"live_rows":live,
                        "cache_mode":if cold {"application_cold"} else {"warm"},"names":names,
                        "samples_ms":samples,"warmup_samples_ms":warmup_samples,"pretrial_barrier_ms":barriers,"host_ms":host,"logits_hashes":hashes,"predictions":ids,"expert_routes":routes,
                        "exact_vs_control":exact,"accepted_state_exact":state_exact,"numerical_diagnostics":diagnostics});
                    std::fs::write(
                        output.join(format!("rapid-p{prompt_index}-f{frontier}-cold{cold}-m{live}.json")),
                        serde_json::to_vec_pretty(&record)?,
                    )?;
                    eprintln!(
                        "rapid rank{rank} frontier{frontier} cold{cold} M{live}: exact {:?}",
                        exact
                    );
                    records.push(record);
                }
            }
        }
    }
    verify.set_engram_overlap_for_probe(false)?;
    let all_exact = records.iter().all(|r| {
        r["exact_vs_control"]
            .as_array()
            .unwrap()
            .iter()
            .all(|b| b == true)
    });
    Ok(
        json!({"rank":rank,"context_capacity":large.context_capacity(),"compact_screen":compact_screen,"compact_sparse_index_layers":[verify.compact_sparse_index_layers(),candidate.compact_sparse_index_layers()],"fused_qkv_screen":fused_qkv_screen,"fused_projection_layers":[verify.fused_projection_layers(),candidate.fused_projection_layers()],"schedule_counts":[verify.schedule_counts_for_probe(),candidate.schedule_counts_for_probe()],"adaptive_sparse_screen":adaptive_sparse_screen,"adaptive_sparse_index_layers":[verify.adaptive_sparse_index_layers(),candidate.adaptive_sparse_index_layers()],"sparse_screen":sparse_screen,"sparse_index_layers":[verify.sparse_index_layers(),candidate.sparse_index_layers()],"prefill":prefill_records,"names":names,"records":records,"all_exact":all_exact,"sort_screen":sort_screen,"micro_screen":micro_screen,"live_moe_screen":live_moe_screen,"scale_screen":scale_screen,"compute_screen":compute_screen,"compute_numerical":compute_numerical,"narrow_only_screen":narrow_only,"task_order_only_screen":task_order_only,"task_order_layers":[verify.task_order_moe_layers(),candidate.task_order_moe_layers()],"narrow_projection_layers":[verify.narrow_projection_layers(),candidate.narrow_projection_layers()],"warmup_iterations":warmup,"measured_iterations":measured,
        "pretrial_peer_barrier":true,"full_engine_acceptance":false,"scope":if compact_screen {"One1M allocation; 64K/128K/256K prefixes and16K dense fallback. Same adaptive sparse scorer and canonical precision; compact top-k avoids full score expansion and full-width selection. M3/6, cold/warm, one warmup and two balanced pairs; full logit hashes and rollback checks. Component evidence s351b2; no generation or outcome claim."} else if adaptive_sparse_screen { "Dense versus adaptive consumer indexing at16K/64K/128K and a reset to16K, M1/3/6 cold/warm and four balanced pairs. Dense below64K accepted tokens, sparse above; separate retained CUDA graph banks. Canonical precision and full logit/rollback checks; no outcome or production qualification." } else if fused_qkv_screen { "One32K load, two accepted contexts, M3/6 cold/warm and four balanced pairs. Replicated canonical QKV-A projection removes40TP gathers and their Q-B prefetch windows; other scheduling and precision unchanged. Full logit hashes, greedy IDs and rollback checks. No outcome or production qualification." } else if sparse_screen { "One256K-capacity load with fixed16K/64K/128K accepted prefixes, M3/6 cold/warm and four balanced pairs. Dense versus candidate-only FP32 scoring at four consumer indexers; canonical keys/queries unchanged. Full live logit hashes, greedy IDs and rollback checks. Frozen repository-review corpus shared with the SGLang prefill panel; no actual-session, outcome or matched SGLang comparison." } else if task_order_only { "One32K-capacity load, public16878-token frontier, warm M3/M6 and four balanced measured pairs after warmup. Only grouped expert task order differs; narrow projections, precision and other schedules unchanged. Full live logits/routes and rollback checks. Small component integration screen; no throughput, long-context or production claim." } else if narrow_only { "One32K-capacity load, public16878-token frontier, warm M3/M6 and two reversed-order measured pairs after warmup. Only the two narrow FP8 projections differ; expert task order and precision unchanged. Full live logits/routes and rollback checks, with rank3 raw logits retained. A small kernel-selection screen, not generation throughput, long-context or production qualification." } else if compute_screen { "One 32K load, two accepted contexts, M1/3/6 cold/warm. Two narrow FP8 projections and grouped expert task order differ; precision unchanged. Four balanced pairs with full live logits/routes and retained state checks. No outcome or production qualification." } else if scale_screen { "One32K load, two accepted contexts, M1/3/6 cold/warm, same live bounds and M1micro. Only grouped FC1scale loading differs. Four balanced pairs, exact complete live logits/routes/retained state. No outcome or production qualification." } else if live_moe_screen { "One 32K load, two accepted contexts, M1..6, cold/warm Engram and identical combined schedules. Full-capacity grouped versus live bounds and FP32 micro only at M1. Four balanced pairs; full live logits, routes and accepted state checks. No outcome or production qualification." } else if micro_screen { "One32K load, two accepted contexts, M1/2/3/4/6 with cold/warm Engram rows and identical combined schedules. Current grouped versus FP32-projection micro routed experts. One warmup and four balanced pairs, full live logit hashes and unchanged accepted state. Canonical weights and activation boundaries; no outcome or production qualification." } else if sort_screen { "One32K load, two accepted contexts, M1/3/6 and cold/warm Engram rows, identical combined schedules. Original integer bitonic versus metadata skip and warp shuffle. One warmup and four balanced pairs, exact full-row logit hashes and unchanged accepted state. Component screen passed47cases. No outcome or production qualification." } else { "One model load, two accepted-context lengths (including source20 block-mask selection), Verify1/3/6, application-cold/warm immutable row lookup, four balanced schedules, two warmups and eight measured repetitions. Full valid-row logit hashes and unchanged accepted carry/encoder history. A rapid hypothesis screen, not tail-latency, stochastic outcome or production qualification." }}),
    )
}
