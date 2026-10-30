//! Short paired fusion screen, with the accumulated copy/schedule candidate.
use crate::{cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,candidate:&mut Target,
    wides:&mut[Target],comm:&Roce,stream:&Stream,output:&Path,rank:usize)->Result<Value> {
    let (warmup_pairs,measured_pairs)=(2,2);
    ensure!(std::env::var("CANADA_SCRATCH_SHARED_QUANT_SCREEN").as_deref()==Ok("1")
        && large.context_capacity()==1048576 && verify.fused_shared_quantization_layers()==0
        && candidate.fused_shared_quantization_layers()==40
        && verify.packed_index_gather_layers()==8 && candidate.packed_index_gather_layers()==8
        && verify.packed_sparse_gather_layers()==4 && candidate.packed_sparse_gather_layers()==4
        && verify.schedule_counts_for_probe()==candidate.schedule_counts_for_probe()
        && verify.fused_compact_publication_layers()==4 && candidate.fused_compact_publication_layers()==4,
        "shared fusion requires otherwise equal accumulated profiles");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    ensure!(format!("{:x}",Sha256::digest(raw))=="107ceccbefc53247fc827cdee7eb3908f7b89a907d8c7a2d3de3ef6b95057fbb",
        "shared fusion fixture identity");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let frontier=262144;
    let mut barrier_in=cuda.allocate::<f32>(4)?;barrier_in.upload(&[0.;4])?;
    let mut barrier_out=cuda.allocate::<f32>(16)?;
    let prefill=Instant::now();large.prefill_wide_with_prefetch(wides,&tokens[..frontier],comm,stream,true)?;
    let prefill_seconds=prefill.elapsed().as_secs_f64();verify.handoff_from(large)?;
    let carry=verify.carry_digests()?;let encoder=verify.encoder_tail_digest()?;
    let mut records=Vec::new();
    for live in [3,6] {
        let mut reference=None;
        for iteration in 0..warmup_pairs+measured_pairs {
            for path in [iteration%2,1-iteration%2] {
                if path==1 {candidate.handoff_from(verify)?;}
                let target=if path==1 {&mut *candidate}else{&mut *verify};
                stream.join_default()?;
                unsafe{comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)?;}
                stream.synchronize()?;comm.health()?;
                let start=Instant::now();target.begin_verification(&tokens[frontier..frontier+live],comm,stream)?;
                let predictions=target.verification_greedy(stream)?;
                let milliseconds=start.elapsed().as_secs_f64()*1000.;
                let phases=serde_json::to_value(target.verification_metrics()?)?;
                let (logits,rows)=target.pending_logits()?;ensure!(rows==live,"shared fusion live logits");
                let signature=json!({"logits":format!("{:x}",Sha256::digest(unsafe{cuda.read_device(logits.ptr(),live*129280*4)}?)),
                    "predictions":predictions});
                let exact=reference.as_ref().is_none_or(|r|r==&signature);
                if reference.is_none(){reference=Some(signature.clone());}
                target.finish_verification(0,comm,stream)?;
                let rollback=target.position()as usize==frontier && target.carry_digests()?==carry
                    && target.encoder_tail_digest()?==encoder;
                let record=json!({"live_rows":live,"iteration":iteration,"warmup":iteration<warmup_pairs,
                    "fused":path==1,"milliseconds":milliseconds,"phases":phases,"signature":signature,
                    "same_signature":exact,"rollback_exact":rollback,"fused_layers":target.fused_shared_quantization_layers(),
                    "capture_statistics":target.capture_statistics()});
                std::fs::write(output.join(format!("shared-quant-m{live}-i{iteration}-v{path}.json")),
                    serde_json::to_vec_pretty(&record)?)?;
                ensure!(exact&&rollback,"shared fusion changed output/state; saved evidence");
                if path==1 {verify.handoff_from(candidate)?;}
                records.push(record);
            }
        }
    }
    Ok(json!({"rank":rank,"shared_quant_screen":true,"all_exact":true,"frontier":frontier,
        "warmup_pairs":warmup_pairs,"measured_pairs":measured_pairs,
        "records":records,"prefill_seconds":prefill_seconds,"memory":cuda.admission_memory()?,
        "full_engine_acceptance":false,"scope":"One262K frontier and accumulated1M candidate. Fuse canonical BF16 shared activation and native MXFP8 quantization. Two excluded warmup pairs and two balanced measured pairs at M3/M6. Full FP32 logits, peer agreement, graph reuse and rollback state. Fixed-input greedy readout isolates kernel timing; intended sampler remains T1/P1/high reasoning."}))
}
