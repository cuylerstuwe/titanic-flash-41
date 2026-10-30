//! Canonical TP4 attention boundary, with Q and residual-statistics side work.
use crate::{attention::{Candidates,PagedKv},cuda::{Cuda,Module,Stream},
    kv::RowLocations,layer_attention::{Layer,InputMode},mhc::Boundary,
    prelude::{FrequencyTable,Positions},roce::Roce,weights::Weights};
use anyhow::{ensure,Result};
use half::bf16;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,rc::Rc};

pub(crate) fn run(cuda:&Cuda,module:&Module,core:&[u8],stream:&Stream,weights:Rc<Weights>,
    kernels:&Path,rank:usize,endpoint:&str,ticket:&[u8])->Result<Value>{
    ensure!(std::env::var("CANADA_SCRATCH_ATTENTION16_SCREEN").as_deref()==Ok("1"),"M16 attention component admission");
    const M:usize=16;const FREQ:usize=131072;
    let residual=cuda.allocate::<bf16>(M*20480)?;
    let mut previous=cuda.allocate::<f32>(M*4)?;previous.upload(&vec![0.25;M*4])?;
    let frequency=Rc::new(FrequencyTable::new(cuda,module,stream,FREQ,false)?);
    let scratch=crate::woa_medium::Scratch::new(cuda,core,M)?;
    let sink=cuda.allocate::<f32>(16)?;
    unsafe{cuda.copy_device(sink.ptr(),weights.ptr("layers.0.attn.attn_sink")?,sink.bytes(),stream)?;}
    let mut variants=Vec::new();
    for mode in 0..4 {
        let mut attention=Layer::new_rows(cuda,module,stream,weights.clone(),0,kernels,
            frequency.clone(),InputMode::Split(rank),M,Some(scratch.clone()))?;
        attention.set_query_overlap(mode&1!=0,stream)?;
        let mut boundary=Boundary::new(cuda,stream,weights.clone(),0,"attn",M,kernels)?;
        if mode&2!=0 {boundary.enable_verify16_overlap_for_probe(stream)?;}
        variants.push((attention,boundary,PagedKv::new(cuda,4,64)?,cuda.allocate::<bf16>(M*5120)?));
    }
    let mut comm=Roce::local(cuda,&kernels.join("roce-v0"),rank)?;
    comm.attach_gather(&kernels.join("roce-gather-v1"))?;
    let peers=crate::rendezvous::exchange_blob(rank,endpoint.parse()?,ticket,&comm.blob()?)?;
    comm.connect(&peers)?;
    let mut positions=Positions::upload(cuda,&[0;M],FREQ)?;
    let mut locations=RowLocations::upload(cuda,&[0;M],&[-1;M],256,FREQ)?;
    let mut candidates=Candidates::upload(cuda,M,128,256,&[-1;M*128],&[0;M])?;
    let fill=|pattern:i32|->Result<()>{
        let mut dst=residual.ptr();let mut n=residual.len()as i32;let mut seed=pattern;let mut p=0i32;
        unsafe{module.function("ca_collective_fixture")?.launch([(n as u32).div_ceil(256),1,1],[256,1,1],0,stream,
            &mut[(&mut dst as *mut u64).cast(),(&mut n as *mut i32).cast(),
                (&mut seed as *mut i32).cast(),(&mut p as *mut i32).cast()])}
    };
    // Fill a real128-token layer0 KV history with canonical GPU projections.
    // Host work below only supplies causal positions and candidate slot IDs.
    for base in (0..=128).step_by(M) {
        let pos:Vec<i32>=(base..base+M).map(|i|i as i32).collect();
        let loc:Vec<i32>=pos.iter().map(|p|p+1).collect();positions.update(&pos)?;locations.update(&pos,&loc)?;
        let mut ids=vec![-1;M*128];let mut lengths=vec![0;M];
        for row in 0..M {
            let end=base+row+1;let start=end.saturating_sub(128);lengths[row]=(end-start)as i32;
            for(i,t)in(start..end).enumerate(){ids[row*128+i]=(t+1)as i32;}
        }
        candidates.update(&ids,&lengths)?;stream.join_default()?;fill((base/M)as i32)?;
        for(attention,boundary,cache,reduced)in &mut variants {
            boundary.prepare(&residual,Some(&previous),stream)?;
            attention.run_distributed(boundary.input(),&positions,&locations,cache,&candidates,&sink,Some(&comm),stream)?;
            unsafe{comm.sum_bf16(attention.output(),reduced,stream)?;}
            boundary.finish(reduced,&residual,stream)?;
        }
        stream.synchronize()?;comm.health()?;
    }
    let mut graphs=Vec::new();
    for(attention,boundary,cache,reduced)in &mut variants {
        graphs.push(stream.capture(||{
            boundary.prepare(&residual,Some(&previous),stream)?;
            attention.run_distributed(boundary.input(),&positions,&locations,cache,&candidates,&sink,Some(&comm),stream)?;
            unsafe{comm.sum_bf16(attention.output(),reduced,stream)?;}
            boundary.finish(reduced,&residual,stream)
        })?);
    }
    let hash=|p:u64,n:usize|->Result<String>{Ok(format!("{:x}",Sha256::digest(unsafe{cuda.read_device(p,n)}?)))};
    let mut results=Vec::new();
    for pattern in 0..2 {
        fill(pattern+31)?;unsafe{graphs[0].launch(stream)?;}stream.synchronize()?;
        let taps=|mode:usize|->Result<Vec<String>>{
            let(a,b,c,r)=&variants[mode];Ok(vec![hash(a.output().ptr(),a.output().bytes())?,hash(r.ptr(),r.bytes())?,
                hash(b.output().ptr(),b.output().bytes())?,hash(c.data().ptr(),c.data().bytes())?])
        };
        let reference=taps(0)?;let mut pairs=Vec::new();
        for repeat in 0..4 {
            let mut order=vec![0,1,2,3];order.rotate_left(repeat/2);if repeat%2==1{order.reverse();}
            let mut milliseconds=[0f32;4];
            for &mode in &order {
                for _ in 0..4 {unsafe{graphs[mode].launch(stream)?;}}
                stream.synchronize()?;
                let start=cuda.event()?;let end=cuda.event()?;start.record(stream)?;
                for _ in 0..16 {unsafe{graphs[mode].launch(stream)?;}}
                end.record(stream)?;milliseconds[mode]=end.elapsed_since(&start)?/16.;comm.health()?;
                ensure!(taps(mode)?==reference,"M16 attention schedule changed local/global/residual/KV bytes");
            }
            pairs.push(json!({"repeat":repeat,"order":order,"milliseconds":milliseconds}));
        }
        results.push(json!({"pattern":pattern,"all_exact":true,"local_global_residual_kv_sha256":reference,"pairs":pairs}));
    }
    Ok(json!({"rank":rank,"all_exact":true,"all_bitwise_equal":true,"rows":M,
        "variants":["serial","query_side","attn_residual_stats","both"],"results":results,
        "scope":"One actual layer0 attention boundary with128-token GPU-generated KV history and two residual patterns. Same math and TP4 collectives; compressed-index overlap is outside this component."}))
}
