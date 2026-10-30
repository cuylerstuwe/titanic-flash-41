//! One-frontier comparison of checked causal metadata with mapped transport.
use crate::{cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};

fn evaluate(cuda:&Cuda,target:&mut Target,ids:&[u32],accepted:usize,comm:&Roce,stream:&Stream)->Result<Value> {
    let began=Instant::now();target.begin_verification(ids,comm,stream)?;
    let predictions=target.verification_greedy(stream)?;
    let milliseconds=began.elapsed().as_secs_f64()*1000.;
    let phases=serde_json::to_value(target.verification_metrics()?)?;
    let (logits,live)=target.pending_logits()?;ensure!(live==ids.len(),"mapped metadata live logits");
    let signature=json!({"logits":format!("{:x}",Sha256::digest(unsafe{cuda.read_device(logits.ptr(),live*129280*4)}?)),"predictions":predictions});
    target.finish_verification(accepted,comm,stream)?;
    Ok(json!({"milliseconds":milliseconds,"phases":phases,"signature":signature,
        "state":{"position":target.position(),"carry":target.carry_digests()?,"encoder":target.encoder_tail_digest()?}}))
}
fn equal(a:&Value,b:&Value)->bool {a["signature"]==b["signature"] && a["state"]==b["state"]}
fn sequence(cuda:&Cuda,target:&mut Target,tokens:&[u32],comm:&Roce,stream:&Stream)->Result<Vec<Value>> {
    let mut result=Vec::new();
    for (stage,(live,accepted)) in [(6,3),(3,0),(6,1),(3,2)].into_iter().enumerate() {
        let start=target.position() as usize;let mut ids=tokens[start..start+live].to_vec();
        if stage==1 {ids[live-1]=(ids[live-1]+1)%129280;}
        result.push(evaluate(cuda,target,&ids,accepted,comm,stream)?);
    }
    Ok(result)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,candidate:&mut Target,
    wides:&mut[Target],comm:&Roce,stream:&Stream,output:&Path,rank:usize)->Result<Value> {
    ensure!(std::env::var("CANADA_SCRATCH_MAPPED_METADATA_SCREEN").as_deref()==Ok("1")
        && large.context_capacity()==1048576 && verify.mapped_metadata_bytes()==0 && candidate.mapped_metadata_bytes()>0
        && large.mapped_metadata_bytes()==0
        && verify.cached_index_gather_layers()==8 && candidate.cached_index_gather_layers()==8
        && verify.packed_index_gather_layers()==8 && candidate.packed_index_gather_layers()==8
        && verify.packed_sparse_gather_layers()==4 && candidate.packed_sparse_gather_layers()==4
        && verify.fused_shared_quantization_layers()==40 && candidate.fused_shared_quantization_layers()==40
        && verify.schedule_counts_for_probe()==candidate.schedule_counts_for_probe(),
        "mapped metadata requires otherwise equal current profiles");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    ensure!(format!("{:x}",Sha256::digest(raw))=="107ceccbefc53247fc827cdee7eb3908f7b89a907d8c7a2d3de3ef6b95057fbb","mapped metadata fixture identity");
    let fixture:Value=serde_json::from_slice(raw)?;let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let frontier=262144;
    let mut barrier_in=cuda.allocate::<f32>(4)?;barrier_in.upload(&[0.;4])?;let mut barrier_out=cuda.allocate::<f32>(16)?;
    let began=Instant::now();large.prefill_wide_with_prefetch(wides,&tokens[..frontier],comm,stream,true)?;
    let prefill_seconds=began.elapsed().as_secs_f64();verify.handoff_from(large)?;
    let saved=verify.save_state_image(2*1024*1024*1024,comm,stream)?;
    let mut records=Vec::new();let mut signatures=Vec::new();
    const WARMUP_PAIRS:usize=2;
    const MEASURED_PAIRS:usize=2;
    for live in [3,6] {
        let mut reference=None;
        for iteration in 0..WARMUP_PAIRS+MEASURED_PAIRS {
            for path in [iteration%2,1-iteration%2] {
                if path==1 {candidate.handoff_from(verify)?;}
                let target=if path==1 {&mut *candidate}else{&mut *verify};
                // Handoff invalidates scratch. Prime once outside warm timing;
                // this also isolates cold/first-use graph capture cost.
                let prime=evaluate(cuda,target,&tokens[frontier..frontier+live],0,comm,stream)?;
                stream.join_default()?;unsafe{comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)?;}
                stream.synchronize()?;comm.health()?;
                let measured=evaluate(cuda,target,&tokens[frontier..frontier+live],0,comm,stream)?;
                let exact=equal(&prime,&measured) && reference.as_ref().is_none_or(|r|equal(r,&measured));
                if reference.is_none(){reference=Some(measured.clone());}
                let record=json!({"live_rows":live,"iteration":iteration,"warmup":iteration<WARMUP_PAIRS,"mapped":path==1,
                    "prime":prime,"measured":measured,"all_exact":exact,"mapped_bytes":target.mapped_metadata_bytes()});
                std::fs::write(output.join(format!("mapped-metadata-m{live}-i{iteration}-v{path}.json")),serde_json::to_vec_pretty(&record)?)?;
                ensure!(exact,"mapped metadata warm/prime logits or rollback state changed; evidence saved");
                if path==1 {verify.handoff_from(candidate)?;}
                records.push(record);
            }
        }
        signatures.push(reference.unwrap());
    }
    // Record a short accepted/zero-accepted trajectory, then a different short
    // prompt. The cached profile still contains the original long prefix.
    let reference_sequence=sequence(cuda,verify,&tokens,comm,stream)?;
    verify.reset(900001,comm,stream)?;
    let short:Vec<u32>=tokens[100..130].iter().map(|id|(id+17)%129280).collect();
    let mut reset_reference=Vec::new();
    for ids in short[..24].chunks(6){reset_reference.push(evaluate(cuda,verify,ids,6,comm,stream)?);}
    let handoff_reference=evaluate(cuda,verify,&short[24..30],0,comm,stream)?;
    candidate.handoff_from(verify)?;
    let handoff=evaluate(cuda,candidate,&short[24..30],0,comm,stream)?;
    let handoff_exact=equal(&handoff_reference,&handoff);
    // The current gathered bytes now belong to the short prompt. Restoring
    // canonical KV alone cannot make them valid for the original long prompt.
    candidate.restore_state_image(&saved,900002,comm,stream)?;
    let restored=evaluate(cuda,candidate,&tokens[frontier..frontier+6],0,comm,stream)?;
    let restore_exact=equal(&signatures[1],&restored);
    let continued=sequence(cuda,candidate,&tokens,comm,stream)?;
    let partial_acceptance_exact=reference_sequence.iter().zip(&continued).all(|(a,b)|equal(a,b));
    candidate.reset(900003,comm,stream)?;
    let mut reset=Vec::new();
    for ids in short[..24].chunks(6){reset.push(evaluate(cuda,candidate,ids,6,comm,stream)?);}
    let reset_exact=reset_reference.iter().zip(&reset).all(|(a,b)|equal(a,b));
    let lifecycle=json!({"handoff_exact":handoff_exact,"restore_exact":restore_exact,
        "partial_acceptance_exact":partial_acceptance_exact,"reset_exact":reset_exact,
        "handoff":handoff,"restored":restored,"continued":continued,"reset":reset,
        "reference_sequence":reference_sequence,"handoff_reference":handoff_reference,"reset_reference":reset_reference});
    std::fs::write(output.join("mapped-metadata-lifecycle.json"),serde_json::to_vec_pretty(&lifecycle)?)?;
    ensure!(handoff_exact && restore_exact && partial_acceptance_exact && reset_exact,"mapped metadata lifecycle changed logits/state; evidence saved");
    verify.handoff_from(candidate)?;
    Ok(json!({"rank":rank,"mapped_metadata_screen":true,"all_exact":true,"frontier":frontier,
        "warmup_pairs":WARMUP_PAIRS,"measured_pairs":MEASURED_PAIRS,"records":records,"lifecycle":lifecycle,
        "prefill_seconds":prefill_seconds,"state_image_bytes":saved.bytes(),"memory":cuda.admission_memory()?,
        "full_engine_acceptance":false,"scope":"One262K frontier, current1M candidate, M3/M6 mapped scheduling buffers versus device copies with identical checked setters. Two warmups/two measured pairs, stream completion before host writes. Exact FP32 logits and retained state; partial/zero acceptance and changed-prompt handoff, image restore, reset. No sampler or arithmetic change; intended settings remain T1/P1/high reasoning."}))
}
