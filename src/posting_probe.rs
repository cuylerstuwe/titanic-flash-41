//! Finite same-allocation CPU submission API screen; canonical GPU transport.
use crate::{cuda::Cuda,roce::Roce};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use std::path::Path;

pub fn run(rank:usize,endpoint:&str,nonce:&[u8],root:&Path,core:&Path)->Result<Value>{
    ensure!(rank<4&&std::env::var("CANADA_SCRATCH_POSTING_API_SCREEN").as_deref()==Ok("1"),
        "finite posting screen admission");
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let module=cuda.module(&std::fs::read(core)?)?;
    let mut comm=Roce::local(&cuda,&root.join("roce-v0"),rank)?;
    comm.attach_gather(&root.join("roce-gather-v1"))?;
    let all=crate::rendezvous::exchange_blob(rank,endpoint.parse()?,nonce,&comm.blob()?)?;
    comm.connect(&all)?;
    let mut epoch=cuda.allocate::<u32>(1)?;epoch.zero()?;
    let mut stats=cuda.allocate::<u32>(7)?;stats.upload(&[0,u32::MAX,u32::MAX,0,0,0,0])?;stream.join_default()?;
    let mut records=Vec::new();
    for bytes in [16usize,32,128,4096,65536,1048576,8388608] {
        let n=bytes/2;let input=cuda.allocate::<u16>(n)?;let mut output=cuda.allocate::<u16>(4*n)?;
        let fill=||->Result<()> {
            let(mut x,mut e,mut count,mut r,mut p,mut j)=(input.ptr(),epoch.ptr(),n as i32,rank as i32,1i32,0i32);
            unsafe{module.function("ca_sequence_fill")?.launch([(n as u32).div_ceil(256),1,1],[256,1,1],0,&stream,
                &mut[(&mut x as *mut u64).cast(),(&mut e as *mut u64).cast(),(&mut count as *mut i32).cast(),
                    (&mut r as *mut i32).cast(),(&mut p as *mut i32).cast(),(&mut j as *mut i32).cast()])?;}Ok(())
        };
        let advance=||->Result<()> {
            let mut e=epoch.ptr();unsafe{module.function("ca_sequence_advance")?.launch([1,1,1],[32,1,1],0,&stream,
                &mut[(&mut e as *mut u64).cast()])?;}Ok(())
        };
        // Pointer-only view keeps the validation closure independent of the
        // mutable Rust output handle used during capture.
        let out_ptr=output.ptr();let stats_ptr=stats.ptr();
        let check=||->Result<()> {
            let(mut y,mut e,mut s,mut count,mut p,mut g)=(out_ptr,epoch.ptr(),stats_ptr,n as i32,1i32,1i32);
            unsafe{module.function("ca_posting_check")?.launch([(4*n as u32).div_ceil(256),1,1],[256,1,1],0,&stream,
                &mut[(&mut y as *mut u64).cast(),(&mut e as *mut u64).cast(),(&mut s as *mut u64).cast(),
                    (&mut count as *mut i32).cast(),(&mut p as *mut i32).cast(),(&mut g as *mut i32).cast()])?;}Ok(())
        };
        fill()?;unsafe{comm.gather_bf16(&input,&mut output,n,&stream)?;}
        check()?;stream.synchronize()?;ensure!(stats.download()?[0]==0,"posting initial GPU bytes");
        advance()?;check()?;stream.synchronize()?;ensure!(stats.download()?[0]>0,"posting stale-data negative control");
        stats.upload(&[0,u32::MAX,u32::MAX,0,0,0,0])?;stream.join_default()?;
        let changes=stream.capture(||{
            for _ in 0..8 {fill()?;unsafe{comm.gather_bf16(&input,&mut output,n,&stream)?;}check()?;advance()?;}
            Ok(())
        })?;
        let timed=stream.capture(||{
            for _ in 0..8 {unsafe{comm.gather_bf16(&input,&mut output,n,&stream)?;}}
            Ok(())
        })?;
        let mut pairs=Vec::new();
        for repeat in 0..4 {
            let order=if repeat%2==0 {[0u32,1,2]}else{[2,1,0]};
            let mut ms=[0f32;3];
            for mode in order {
                comm.set_posting_mode_for_probe(mode,&stream)?;
                // Across all sizes this exceeds SQ/CQ ring capacities with
                // independently changing GPU payloads and alternating slots.
                for _ in 0..8 {unsafe{changes.launch(&stream)?;}}
                fill()?;
                for _ in 0..2 {unsafe{timed.launch(&stream)?;}}
                stream.synchronize()?;
                let start=cuda.event()?;let end=cuda.event()?;start.record(&stream)?;
                for _ in 0..8 {unsafe{timed.launch(&stream)?;}}
                end.record(&stream)?;ms[mode as usize]=end.elapsed_since(&start)?/64.;
                check()?;stream.synchronize()?;comm.health()?;
                ensure!(stats.download()?[0]==0,"posting API changed GPU payload");
            }
            pairs.push(json!({"repeat":repeat,"order":order,"milliseconds":ms}));
        }
        records.push(json!({"bytes_per_rank":bytes,"pairs":pairs,"all_gpu_bytes_exact":true,
            "stale_payload_negative_control":true}));
        eprintln!("rank{rank} posting bytes{bytes} exact");
    }
    comm.set_posting_mode_for_probe(0,&stream)?;
    Ok(json!({"rank":rank,"all_exact":true,"records":records,"modes":["generic WR","builder","builder+inline<=16B"],
        "scope":"One allocation, QP set and captured native GPU graphs; only CPU posting API changes. Four alternating orders, changing GPU payloads, exact all-peer bytes and stale negative control. Same packet sizes, dual rails and precision. No model or serving changes."}))
}
