//! Temporary startup padding experiment and read-only measurements on the
//! actual retained weight allocations. No serving defaults or math changes.
use crate::{control_channel::ControlChannel,cuda::{Buffer,Cuda,Function,Module,Stream},roce::Roce,target::Target};
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
const PIECE:usize=256*1024*1024;
/// GB10 memory layout diagnostics (Nov 11). GPU reads that mix the low and the high half of physical memory (the
/// boundary sits near 66 GiB) run up to 8 % slower than reads that stay in one half, and a boot that has loaded
/// models before hands the driver memory from both halves, small fragments first. Both functions below are
/// read-only on model state and default off.
pub(crate) fn soak_mib()->Result<usize>{
    let mib=std::env::var("CANADA_SCRATCH_FRAGMENT_SOAK_MIB").unwrap_or_else(|_|"0".into()).parse::<usize>()?;
    ensure!(mib%256==0&&mib<=16384,"fragment soak is a multiple of256MiB up to16GiB");Ok(mib)
}
pub(crate) fn census_requested()->Result<bool>{
    match std::env::var("CANADA_SCRATCH_WEIGHT_READ_CENSUS").as_deref(){
        Err(std::env::VarError::NotPresent)|Ok("0")=>Ok(false),Ok("1")=>Ok(true),
        _=>anyhow::bail!("weight read census must be0or1"),
    }
}
struct ReadProbe<'a>{wave:Function<'a>,streams:Function<'a>,output:Buffer<u32>}
impl<'a> ReadProbe<'a>{
    fn new(cuda:&Cuda,core:&'a Module)->Result<Self>{
        Ok(Self{wave:core.function("ca_allocation_probe_read")?,streams:core.function("ca_allocation_probe_streams")?,
            output:cuda.allocate::<u32>(384*256*4)?})
    }
    /// GB/s over `passes` timed launches after one untimed launch.
    #[allow(clippy::too_many_arguments)]
    fn time(&self,cuda:&Cuda,stream:&Stream,function:&Function<'_>,block:u32,source:u64,bytes:usize,passes:usize)->Result<f64>{
        let(mut source,mut vectors,mut result)=(source,(bytes/16)as u64,self.output.ptr());
        let mut launch=||->Result<()>{
            unsafe{function.launch([384,1,1],[block,1,1],0,stream,&mut [
                std::ptr::from_mut(&mut source).cast::<c_void>(),std::ptr::from_mut(&mut vectors).cast::<c_void>(),
                std::ptr::from_mut(&mut result).cast::<c_void>()])}
        };
        launch()?;let start=cuda.event()?;let end=cuda.event()?;start.record(stream)?;
        for _ in 0..passes{launch()?;}
        end.record(stream)?;let ms=f64::from(end.elapsed_since(&start)?)/passes as f64;
        Ok(bytes as f64/(ms*1e6))
    }
    /// (one384x256 wavefront over the whole range, median of384x128 line-aligned block-contiguous streams per
    /// ~256MiB slice)
    fn rates(&self,cuda:&Cuda,stream:&Stream,ptr:u64,bytes:usize)->Result<(f64,f64)>{
        let bytes=bytes/16*16;ensure!(bytes>=16*1024*1024,"read probe range");
        let wave=self.time(cuda,stream,&self.wave,256,ptr,bytes,3)?;
        // Every stream must start on a 64-byte line (a stream that starts inside one reads ~5 % slower, which
        // made nine layers with other atlas sizes look slow in s1473): slices are multiples of 384 x 64 bytes.
        let slices=(bytes/PIECE).max(1);let slice=bytes/slices/24576*24576;let mut each=Vec::new();
        for q in 0..slices{each.push(self.time(cuda,stream,&self.streams,128,ptr+(q*slice)as u64,slice,3)?);}
        each.sort_by(f64::total_cmp);
        Ok(((wave*10.).round()/10.,(each[each.len()/2]*10.).round()/10.))
    }
}
/// A throwaway driver allocation made before the target weights and held while they are allocated, so that the
/// fragments the kernel hands out first (scattered over both halves) do not end up under the weights. Each
/// 256 MiB piece is timed, which also shows where the fragments end.
pub(crate) fn soak(cuda:&Cuda,core:&Module,stream:&Stream,rank:usize)->Result<Vec<Buffer<u8>>>{
    let mib=soak_mib()?;let mut held=Vec::new();if mib==0{return Ok(held);}
    let before=cuda.admission_memory()?;
    ensure!(before.admitted_available>=mib*1024*1024+32*GIB,"fragment soak initial headroom");
    let probe=ReadProbe::new(cuda,core)?;let(mut wave,mut streams)=(Vec::new(),Vec::new());
    for _ in 0..mib/256 {
        let buffer=cuda.allocate::<u8>(PIECE)?;let(w,s)=probe.rates(cuda,stream,buffer.ptr(),PIECE)?;
        wave.push(w);streams.push(s);held.push(buffer);
    }
    eprintln!("rank{rank} fragment soak held: {}",json!({"MiB":mib,"wave_GB_per_s":wave,"streams_GB_per_s":streams,
        "memory_before":before,"memory":cuda.admission_memory()?}));
    Ok(held)
}
/// Read rates of all forty retained layer atlases, after the target load and before the profile forks.
pub(crate) fn census(cuda:&Cuda,core:&Module,target:&Target,stream:&Stream,rank:usize)->Result<()>{
    stream.synchronize()?;let probe=ReadProbe::new(cuda,core)?;
    let(mut wave,mut streams,mut mib)=(Vec::new(),Vec::new(),Vec::new());
    for layer in target.layer_owners(){
        let weights=layer.shared_weights();let bytes=weights.bytes();
        let(w,s)=probe.rates(cuda,stream,weights.pointer_at(0,bytes)?,bytes)?;
        wave.push(w);streams.push(s);mib.push(bytes>>20);
    }
    eprintln!("rank{rank} weight read census: {}",json!({"layers":wave.len(),"atlas_MiB":mib,"wave_GB_per_s":wave,
        "streams_GB_per_s":streams,
        "scope":"Read-only. Wave: one384x256 wavefront over each layer atlas. Streams:384x128 block-contiguous streams per256MiB slice, median of the slices."}));
    // Nov 12: the same two read rates over the four sources' compressed KV caches and index key caches (at most
    // the first 2 GiB of each). Long-context steps read these arenas, which the weight census does not cover;
    // step times at 128K differ between loads by more than the weights' placement explains. Read-only.
    let mut arenas=Vec::new();
    for (name,ptr,bytes) in target.cache_arenas(){
        if bytes<64*1024*1024{continue;}
        let(w,s)=probe.rates(cuda,stream,ptr,bytes.min(2*GIB))?;
        arenas.push(json!({"name":name,"MiB":bytes>>20,"wave_GB_per_s":w,"streams_GB_per_s":s}));
    }
    eprintln!("rank{rank} cache read census: {}",json!({"arenas":arenas,
        "scope":"Read-only. Compressed KV caches and index key caches of sources 2, 8, 14, 20; the first 2 GiB of each."}));
    Ok(())
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
