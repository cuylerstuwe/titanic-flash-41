//! Temporary startup padding experiment and read-only measurements on the
//! actual retained weight allocations. No serving defaults or math changes.
use crate::{control_channel::ControlChannel,cuda::{Buffer,Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{ffi::c_void,path::Path,time::Instant};
const GIB:usize=1024usize.pow(3);
fn enabled()->bool{std::env::var("CANADA_SCRATCH_ALLOCATION_PLACEMENT_SCREEN").as_deref()==Ok("1")}
pub(crate) fn padding(cuda:&Cuda,stream:&Stream)->Result<Vec<Buffer<u8>>> {
    let amount=std::env::var("CANADA_SCRATCH_STARTUP_PADDING_GIB").unwrap_or_else(|_|"0".into()).parse::<usize>()?;
    ensure!([0,20].contains(&amount)&& (amount==0||enabled()),"startup padding is an isolated0/20GiB diagnostic");
    let initial=cuda.admission_memory()?;
    if enabled(){eprintln!("startup placement initial: {}",json!({"padding_GiB":amount,"memory":initial}));}
    let mut held=Vec::new();if amount==0{return Ok(held);}
    // The preceding target load retained >=34GiB available before wide scratch
    // construction. Use20GiB, then release it before those profiles are built.
    // Refuse a host with materially less initial room instead of shrinking the
    // physical8GiB reserve or silently changing the experiment's geometry.
    // GB10 driver allocations and ordinary cgroup-accounted host pages are
    // distinct accounting views.114GiB is a physical-host precondition, not a
    // request to exceed or enlarge the unchanged96GiB container limit.
    ensure!(initial.host_available>=114*GIB&&initial.admitted_available>=28*GIB,"startup padding initial headroom");
    for _ in 0..amount/4 {
        ensure!(cuda.admission_memory()?.admitted_available>12*GIB,"startup padding reserve");
        let buffer=cuda.allocate::<u8>(4*GIB)?;
        unsafe{cuda.zero_device(buffer.ptr(),buffer.bytes(),stream)?;}held.push(buffer);
    }
    stream.synchronize()?;
    eprintln!("startup padding held: {}",json!({"GiB":amount,"memory":cuda.admission_memory()?}));
    Ok(held)
}
fn hash(bytes:&[u8])->String{format!("{:x}",Sha256::digest(bytes))}
fn state(t:&Target)->Result<Value>{Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))}
fn logits(cuda:&Cuda,t:&Target)->Result<Vec<u8>>{let(y,rows)=t.pending_logits()?;unsafe{cuda.read_device(y.ptr(),rows*129280*4)}}
fn bandwidth(cuda:&Cuda,core:&[u8],target:&Target,stream:&Stream)->Result<Value>{
    stream.synchronize()?;let module=cuda.module(core)?;let function=module.function("ca_allocation_probe_read")?;
    let output=cuda.allocate::<u32>(384*256*4)?;let mut records=Vec::new();
    let layers=target.layer_owners();ensure!(layers.len()==40,"allocation probe full target");
    for layer in [0,14,39]{
        let weights=layers[layer].shared_weights();let bytes=weights.bytes();
        ensure!(bytes>64*1024*1024&&bytes%16==0,"allocation probe bounded complete atlas");
        let(mut source,mut vectors,mut result)=(weights.pointer_at(0,bytes)?,(bytes/16)as u64,output.ptr());
        let launch=|source:&mut u64,vectors:&mut u64,result:&mut u64|->Result<()>{
            unsafe{function.launch([384,1,1],[256,1,1],0,stream,&mut [
                std::ptr::from_mut(source).cast::<c_void>(),std::ptr::from_mut(vectors).cast::<c_void>(),std::ptr::from_mut(result).cast::<c_void>()])}
        };
        let mut rates=Vec::new();let mut checksum=None;
        for _ in 0..3 {
            for _ in 0..4{launch(&mut source,&mut vectors,&mut result)?;}
            let start=cuda.event()?;let end=cuda.event()?;start.record(stream)?;
            for _ in 0..12{launch(&mut source,&mut vectors,&mut result)?;}
            end.record(stream)?;let ms=end.elapsed_since(&start)?/12.;
            let digest=hash(&unsafe{cuda.read_device(output.ptr(),output.bytes())}?);
            ensure!(checksum.as_ref().is_none_or(|x|x==&digest),"read-only weight bandwidth checksum changed");
            checksum=Some(digest);rates.push(bytes as f64/(ms as f64*1e6));
        }
        records.push(json!({"layer":layer,"atlas_bytes":bytes,"GB_per_s":rates,"gpu_read_xor_sha256":checksum}));
    }Ok(json!(records))
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,core:&[u8],large:&mut Target,verify:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,control:&mut ControlChannel,out:&Path,rank:usize)->Result<()> {
    ensure!(enabled()&&large.position()==0&&verify.physical_rows()==6,"allocation placement screen admission");
    let before=bandwidth(cuda,core,large,stream)?;
    let raw=include_bytes!("../tests/fixtures/long-history-context-v1.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["cases"][0]["prompt_ids"].clone())?;
    let prefix=131072;ensure!(tokens.len()>prefix+6,"allocation placement public corpus");
    control.agree(b"placement-prefill")?;
    let at=Instant::now();large.prefill_wide(wides,&tokens[..prefix],comm,stream)?;
    let prefill_seconds=at.elapsed().as_secs_f64();verify.handoff_from(large)?;let accepted=state(verify)?;
    let mut records=Vec::new();
    for trial in 0..2 {for live in if trial==0{[3,6]}else{[6,3]} {
        control.agree(&[trial as u8,live as u8])?;
        verify.begin_verification(&tokens[prefix..prefix+live],comm,stream)?;
        let reference=logits(cuda,verify)?;verify.finish_verification(0,comm,stream)?;
        for _ in 0..8 {
            verify.begin_verification(&tokens[prefix..prefix+live],comm,stream)?;
            verify.finish_verification(0,comm,stream)?;
        }
        let mut seconds=Vec::new();let mut metrics=Vec::new();
        for replay in 0..24 {
            let at=Instant::now();verify.begin_verification(&tokens[prefix..prefix+live],comm,stream)?;
            stream.synchronize()?;seconds.push(at.elapsed().as_secs_f64());
            metrics.push(serde_json::to_value(verify.verification_metrics()?)?);
            if replay==23{ensure!(reference==logits(cuda,verify)?,"placement repeated logits changed");}
            verify.finish_verification(0,comm,stream)?;
        }
        ensure!(state(verify)?==accepted,"placement rollback state changed");
        records.push(json!({"trial":trial,"live":live,"seconds":seconds,"metrics":metrics,"logits_sha256":hash(&reference),"state_exact":true}));
    }}
    let after=bandwidth(cuda,core,verify,stream)?;
    for (a,b)in before.as_array().unwrap().iter().zip(after.as_array().unwrap()){
        ensure!(a["gpu_read_xor_sha256"]==b["gpu_read_xor_sha256"],"retained weight checksum changed");
    }
    std::fs::write(out.join("allocation-placement-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,
        "padding_GiB":std::env::var("CANADA_SCRATCH_STARTUP_PADDING_GIB").unwrap_or_else(|_|"0".into()),
        "early_layer_weights":crate::weights::early_layer_reservation_requested()?,
        "compressed_target_weights":crate::weights::compressed_target_weights_requested()?,
        "prefix_tokens":prefix,"fixture_sha256":hash(raw),"prefill_seconds":prefill_seconds,
        "weight_bandwidth_before":before,"weight_bandwidth_after":after,"records":records,"all_exact":true,
        "scope":"Diagnostic startup padding is released before wide profiles/drafter. Actual retained layer0/14/39 atlas read rates before/after public131K prefill; two reversed M3/M6 blocks with8 warm and24 timed steps. Same canonical math; no serving-default change. Cross-load comparisons require repeated controls and recorded allocation/boot history."}))?)?;
    Ok(())
}
