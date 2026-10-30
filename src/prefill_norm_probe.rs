//! Small paired prefill screen for a BF16-preserving combine/RMS fusion.
//! Compare distributions on identical prefixes; FP32 reduction order may differ.
use crate::{
    cuda::{Cuda, Stream},
    dspark_generation::{compare_logits, compare_positions, teacher_forced_logits},
    roce::Roce,
    target::Target,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

fn policy(large: &mut Target, wides: &mut [Target], enabled: bool, stream: &Stream, kernels: &Path, comm:&Roce) -> Result<()> {
    let pipeline=std::env::var("CANADA_SCRATCH_MOE_PIPELINE_AB").as_deref()==Ok("1");
    let coalesced = std::env::var("CANADA_SCRATCH_COALESCED_CODEC_AB").as_deref() == Ok("1");
    let fullk = std::env::var("CANADA_SCRATCH_HC_FULLK_AB").as_deref() == Ok("1");
    let quant = std::env::var("CANADA_SCRATCH_PREFILL_QUANT_GATHER_AB").as_deref() == Ok("1");
    let moe = std::env::var("CANADA_SCRATCH_PREFILL_MOE_GATHER_AB").as_deref() == Ok("1");
    let local_qkv = std::env::var("CANADA_SCRATCH_PREFILL_LOCAL_QKV_AB").as_deref() == Ok("1");
    if pipeline { large.set_coalesced_codec(true,kernels,stream)?; comm.set_moe_pipeline_for_probe(enabled,stream)?; }
    else if coalesced { large.set_coalesced_codec(enabled,kernels,stream)?; }
    else if local_qkv { large.set_prefill_local_qkv(enabled, stream)?; }
    else if moe { large.set_prefill_moe_gather(enabled, stream)?; }
    else if quant { large.set_prefill_quant_gather(enabled, stream)?; }
    else if fullk { large.set_prefill_hc_fullk(enabled, stream)?; }
    else { large.set_prefill_norm_fusion(enabled, stream)?; }
    for wide in wides {
        if pipeline { wide.set_coalesced_codec(true,kernels,stream)?; }
        else if coalesced { wide.set_coalesced_codec(enabled,kernels,stream)?; }
        else if local_qkv { wide.set_prefill_local_qkv(enabled, stream)?; }
        else if moe { wide.set_prefill_moe_gather(enabled, stream)?; }
        else if quant { wide.set_prefill_quant_gather(enabled, stream)?; }
        else if fullk { wide.set_prefill_hc_fullk(enabled, stream)?; }
        else { wide.set_prefill_norm_fusion(enabled, stream)?; }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    cuda: &Cuda,
    large: &mut Target,
    verify: &mut Target,
    wides: &mut [Target],
    comm: &Roce,
    stream: &Stream,
    prefix: &[u32],
    output: &Path,
    rank: usize,
    kernels: &Path,
) -> Result<Value> {
    let pipeline=std::env::var("CANADA_SCRATCH_MOE_PIPELINE_AB").as_deref()==Ok("1");
    let coalesced = std::env::var("CANADA_SCRATCH_COALESCED_CODEC_AB").as_deref() == Ok("1");
    let fullk = std::env::var("CANADA_SCRATCH_HC_FULLK_AB").as_deref() == Ok("1");
    let quant = std::env::var("CANADA_SCRATCH_PREFILL_QUANT_GATHER_AB").as_deref() == Ok("1");
    let moe = std::env::var("CANADA_SCRATCH_PREFILL_MOE_GATHER_AB").as_deref() == Ok("1");
    let local_qkv = std::env::var("CANADA_SCRATCH_PREFILL_LOCAL_QKV_AB").as_deref() == Ok("1");
    ensure!(usize::from(pipeline)+usize::from(coalesced)+usize::from(fullk)+usize::from(quant)+usize::from(moe)+usize::from(local_qkv) <= 1, "one paired prefill change at a time");
    let label = if pipeline { "moe-pipeline" } else if coalesced { "coalesced-codec" } else if local_qkv { "prefill-local-qkv" } else if moe { "prefill-moe-gather" } else if quant { "prefill-quant-gather" } else if fullk { "prefill-hc-fullk" } else { "prefill-norm" };
    ensure!(
        large.context_capacity() == 32768 && prefix.len() == 458,
        "norm fusion screen requires32K capacity and458-token fixture"
    );
    let corpus: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/target-agentic-repository-v1.json"
    ))?;
    let tokens: Vec<u32> = serde_json::from_value(corpus["batches"][0].clone())?;
    let long: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/target-agentic-long15800-v1.json"
    ))?;
    let long: Vec<u32> = serde_json::from_value(long["batches"][0].clone())?;
    let forced: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/fixed-verify-python-v1.json"
    ))?;
    let continuation: Vec<u32> = serde_json::from_value(forced["teacher_forced_tokens"].clone())?;
    ensure!(
        tokens.len() >= 4096 && long.len() == 15800 && continuation.len() >= 12,
        "norm fusion fixture bounds"
    );
    large.enable_cached_extend_floor()?;
    let cases: Vec<(&str, &[u32], usize)> = if pipeline || coalesced || fullk || quant || moe || local_qkv {
        ensure!(tokens.len() >= 8192, "full-K prefill fixture length");
        vec![("fresh", &[], 8192), ("retained_long", &long, 8192)]
    } else { vec![
        ("fresh", &[], 129),
        ("fresh", &[], 4096),
        ("retained", prefix, 256),
        ("retained_long", &long, 1024),
    ] };
    let mut records = Vec::new();
    let mut initial = true;
    let mut lineage = 100000;
    for (history, base, delta) in cases {
        for request in 0..if pipeline || coalesced || fullk || quant || moe || local_qkv {3} else {5} {
            let mut seconds = [0.; 2];
            let mut logits: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
            let mut forced: [Vec<Vec<f32>>; 2] = [Vec::new(), Vec::new()];
            let mut incoming = [Value::Null, Value::Null];
            let mut outgoing = [Value::Null, Value::Null];
            let mut phases = [Value::Null, Value::Null];
            let mut hashes = [String::new(), String::new()];
            for path in [request % 2, 1 - request % 2] {
                if !initial {
                    lineage += 1;
                    verify.reset(lineage, comm, stream)?;
                    large.handoff_from(verify)?;
                }
                initial = false;
                // Retained trials start from the same control-produced state.
                policy(large, wides, false, stream, kernels, comm)?;
                if !base.is_empty() {
                    large.prefill_wide_with_policy(wides, base, comm, stream, false, false)?;
                }
                incoming[path] = json!({"position":large.position(),
                    "encoder":large.encoder_tail_digest()?,"carry":large.carry_digests()?});
                policy(large, wides, path == 1, stream, kernels, comm)?;
                if pipeline {
                    ensure!(comm.moe_pipeline_active()==(path==1)&&wides.iter().map(Target::coalesced_codec_layers).sum::<usize>()==18,"pipeline activation or baseline codec");
                }
                if coalesced {
                    let count:usize=wides.iter().map(Target::coalesced_codec_layers).sum();
                    ensure!(count==if path==1 {18}else{0},"coalesced codec activation count");
                    ensure!(wides.iter().map(Target::moe_gather_layers).sum::<usize>()==18,"codec changed packed transport");
                }
                if local_qkv {
                    let count: usize = wides.iter().map(Target::local_qkv_layers).sum();
                    ensure!(if path == 1 {count == 15} else {count == 0}, "local QKV policy did not activate");
                    ensure!(wides.iter().map(Target::quant_gather_layers).sum::<usize>()==15*wides.len(), "local QKV changed control transport");
                }
                if moe {
                    let count: usize = wides.iter().map(Target::moe_gather_layers).sum();
                    ensure!(if path == 1 {count == 18} else {count == 0}, "packed MoE policy did not activate");
                }
                if quant {
                    let count: usize = wides.iter().map(Target::quant_gather_layers).sum();
                    ensure!(if path == 1 {count == 15 * wides.len()} else {count == 0}, "quantized gather policy did not activate");
                }
                if fullk {
                    let count: usize = wides.iter().map(Target::fullk_boundaries).sum();
                    ensure!(if path == 1 {count > 0} else {count == 0}, "full-K policy did not activate");
                }
                let started = Instant::now();
                large.prefill_wide_with_policy(
                    wides,
                    &tokens[..delta],
                    comm,
                    stream,
                    false,
                    false,
                )?;
                let anchor = large.greedy_token(stream)?;
                seconds[path] = started.elapsed().as_secs_f64();
                ensure!(
                    large.position() as usize == base.len() + delta,
                    "norm fusion changed accepted history frontier"
                );
                phases[path] = serde_json::to_value(large.append_metrics()?)?;
                ensure!(
                    phases[path]["captured_new_graph"] == false
                        && phases[path]["encoder_graph_chunks"] == 0,
                    "norm policy experiment must not capture a prefill graph"
                );
                logits[path] = large.last_prompt_logits()?;
                outgoing[path] = json!({"position":large.position(),"anchor":anchor,
                    "encoder":large.encoder_tail_digest()?,"carry":large.carry_digests()?});
                verify.handoff_from(large)?;
                forced[path] =
                    teacher_forced_logits(cuda, verify, &continuation[..12], 2, comm, stream)?;
                ensure!(
                    logits[path]
                        .iter()
                        .chain(forced[path].iter().flatten())
                        .all(|x| x.is_finite()),
                    "norm fusion nonfinite model output"
                );
                let mut hash = Sha256::new();
                for value in logits[path].iter().chain(forced[path].iter().flatten()) {
                    hash.update(value.to_le_bytes());
                }
                hashes[path] = format!("{:x}", hash.finalize());
            }
            ensure!(
                incoming[0] == incoming[1],
                "norm pair did not start from identical state"
            );
            let record = json!({"implementation":if pipeline {"pipelined_moe_transport"}else if coalesced {"coalesced_moe_codec_nccl"}else if local_qkv {"project_local_qkv_before_bf16_gather"} else if moe {"dual_quant_row_packed_moe_input"} else if quant {"shard_local_mxfp8_transport"} else if fullk {"knapcio_full_k_statistics"} else {"cute_native_reduction_layout"},"history_mode":history,"prefix_tokens":base.len(),"new_tokens":delta,
                "request":request,"warmup":request==0,"seconds_control":seconds[0],"seconds_fused":seconds[1],
                "phases":phases,"incoming_state_equal":true,"outgoing_state_equal":outgoing[0]==outgoing[1],
                "logit_hashes":hashes,"last_row_logits":compare_logits(&logits[0],&logits[1]),
                "teacher_forced":compare_positions(&forced[0],&forced[1])});
            std::fs::write(
                output.join(format!("{label}-{history}-d{delta}-r{request}.json")),
                serde_json::to_vec_pretty(&record)?,
            )?;
            eprintln!(
                "{label} rank{rank} {history} delta{delta} request{request}: control {:.4}s fused {:.4}s",
                seconds[0], seconds[1]
            );
            records.push(record);
        }
    }
    Ok(
        json!({"rank":rank,"records":records,"all_finite":true,"memory":cuda.admission_memory()?,
        "full_engine_acceptance":false,"scope":if pipeline {
            "Six paired foreground M8192 prefills: current coalesced codec/NCCL versus eight-chunk registered transport with fused ordinary-memory copy. Exact logits/state and12 following positions. No sampler, target precision, background or outcome-speed change."
        } else if coalesced {
            "Six paired foreground prefills, fresh8192 and retained15800+8192, current packed NCCL transport in both arms. Change only pack/decode IO organization. One warmup/two reversed measured pairs per case, exact logits, state and12 following positions. Component tests used pinned GPU reference and both ordinary magnitudes and shared-floor cases. No background tests or outcome-speed claim."
        } else if local_qkv {
            "Six paired foreground eager prefills at32K capacity, fresh8192 and retained15800+8192. Fifteen sequence-parallel attention layers project local rows with shared canonical weights, then gather BF16 outputs. Control gathers quantized inputs before full projection. Eight GPU component cases exact; compare prompt logits/state and12 following teacher-forced positions. Existing packed MoE transport retained. No outcome claim."
        } else if moe {
            "Six paired foreground eager prefills at32K capacity, fresh8192 and retained15800+8192. Eighteen8192-row sequence-parallel FFNs route local BF16 rows and gather packed inputs preserving routed and shared quantizers exactly. Existing scratch, weights, later MoE arithmetic, sampler and priority untouched. One warmup and two reversed measured pairs per case, prompt logits/state and12 following teacher-forced positions; no completed-outcome claim."
        } else if quant {
            "Six paired eager prefills at32K capacity, fresh8192 and retained15800+8192. Fifteen eligible sequence-parallel attention layers quantize each local shard with the existing native kernel and all-gather exact values/scales. Source layers, BF16 boundary and dense projection unchanged. Component layout checks precede this four-device test. Compare full logits, retained state and12 following teacher-forced positions; no completed-outcome claim."
        } else if fullk {
            "One32K model load; six paired eager prefills: fresh8192 and retained15800+8192, one warmup/two balanced measured pairs each. Full-K HC statistics only at >=1024 local rows, otherwise current kernels. Component screen exact across seven geometries/weights. Compare prompt logits, retained state and12 following teacher-forced distributions; no throughput or outcome claim."
        } else {"One32K model load;20 paired eager prefills, four fresh/retained cases, one warmup and four balanced measured pairs. CuTe fusion preserves the BF16 combine boundary and native RMS reduction layout;32 component cases were bitwise equal. Compare full prompt-row logits and12 following teacher-forced distributions with exact incoming state. Prefill capture disabled for both paths; no outcome or production-quality claim."}}),
    )
}
