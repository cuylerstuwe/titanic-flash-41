//! Reuse the qualified small-row schedules with M16 kernels, independently.
use crate::{blas::Blas,cuda::{Cuda,Module,Stream},expert_packing::Prepared,
    ffn::{LayerFfn,TokenRows},mhc::Boundary,moe::Workspace,roce::Roce,weights::Weights};
use anyhow::{ensure,Result};
use half::bf16;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,rc::Rc};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,module:&Module,stream:&Stream,weights:Rc<Weights>,prepared:Prepared,
    kernels:&Path,rank:usize,endpoint:&str,ticket:&[u8])->Result<Value>{
    ensure!(std::env::var("CANADA_SCRATCH_VERIFY16_OVERLAP_SCREEN").as_deref()==Ok("1"),"M16 component admission");
    const M:usize=16;
    let residual=cuda.allocate::<bf16>(M*20480)?;
    // Fixed input coefficients for the predecessor boundary, not a CPU model.
    let mut previous=cuda.allocate::<f32>(M*4)?;previous.upload(&vec![0.25;M*4])?;
    let mut tokens=TokenRows::new(cuda,&[1001;M],M)?;
    let workspace=Workspace::new_live16(cuda,kernels)?;
    let mut variants=Vec::new();
    for mode in 0..4 {
        let mut ffn=LayerFfn::new_rows(cuda,module,stream,weights.clone(),prepared.clone(),kernels,M,
            Some(Rc::new(Blas::new(cuda)?)))?;
        let mut boundary=Boundary::new(cuda,stream,weights.clone(),0,"ffn",M,kernels)?;
        if mode&1!=0 {ffn.enable_verify16_overlap_for_probe(stream)?;}
        if mode&2!=0 {boundary.enable_verify16_overlap_for_probe(stream)?;}
        variants.push((ffn,boundary,cuda.allocate::<bf16>(M*5120)?));
    }
    let comm=Roce::local(cuda,&kernels.join("roce-v0"),rank)?;
    let peers=crate::rendezvous::exchange_blob(rank,endpoint.parse()?,ticket,&comm.blob()?)?;
    comm.connect(&peers)?;
    let hash=|b:&crate::cuda::Buffer<bf16>|->Result<String>{
        Ok(format!("{:x}",Sha256::digest(unsafe{cuda.read_device(b.ptr(),b.bytes())}?)))
    };
    let mut results=Vec::new();
    for live in [8,16] {
        let mut ids=[1001;M];ids[0]=129264;tokens.update(&ids,live)?;stream.join_default()?;
        let mut graphs=None;
        for pattern in 0..2 {
            let mut dst=residual.ptr();let mut n=residual.len()as i32;let mut seed=pattern;let mut p=0i32;
            unsafe{module.function("ca_collective_fixture")?.launch([(n as u32).div_ceil(256),1,1],[256,1,1],0,stream,
                &mut[(&mut dst as *mut u64).cast(),(&mut n as *mut i32).cast(),
                    (&mut seed as *mut i32).cast(),(&mut p as *mut i32).cast()])?;}
            if graphs.is_none(){
                let mut captured=Vec::new();
                for (ffn,boundary,reduced)in &mut variants{
                    let mut run=||->Result<()>{
                        boundary.prepare(&residual,Some(&previous),stream)?;
                        ffn.run(&workspace,boundary.input(),&tokens,stream)?;
                        unsafe{comm.sum_bf16(ffn.output(),reduced,stream)?;}
                        boundary.finish(reduced,&residual,stream)
                    };
                    run()?;stream.synchronize()?;comm.health()?;
                    captured.push(stream.capture(run)?);
                }
                graphs=Some(captured);
            }
            let graphs=graphs.as_ref().unwrap();
            unsafe{graphs[0].launch(stream)?;}stream.synchronize()?;comm.health()?;
            let reference=[hash(variants[0].0.output())?,hash(&variants[0].2)?,hash(variants[0].1.output())?];
            let mut pairs=Vec::new();
            for repeat in 0..4 {
                let mut order=vec![0,1,2,3];order.rotate_left(repeat/2);if repeat%2==1{order.reverse();}
                let mut milliseconds=[0f32;4];
                for &mode in &order {
                    for _ in 0..4{unsafe{graphs[mode].launch(stream)?;}}
                    stream.synchronize()?;
                    let start=cuda.event()?;let end=cuda.event()?;start.record(stream)?;
                    for _ in 0..16{unsafe{graphs[mode].launch(stream)?;}}
                    end.record(stream)?;milliseconds[mode]=end.elapsed_since(&start)?/16.;comm.health()?;
                    let observed=[hash(variants[mode].0.output())?,hash(&variants[mode].2)?,hash(variants[mode].1.output())?];
                    ensure!(observed==reference,"M16 overlap changed local FFN/reduction/residual outputs");
                }
                pairs.push(json!({"repeat":repeat,"order":order,"milliseconds":milliseconds}));
            }
            results.push(json!({"live":live,"pattern":pattern,"all_exact":true,"local_global_residual_sha256":reference,"pairs":pairs}));
        }
    }
    stream.synchronize()?;comm.health()?;
    Ok(json!({"rank":rank,"all_exact":true,"all_bitwise_equal":true,"rows":M,"variants":["serial","shared_expert","residual_stats","both"],"results":results,
        "scope":"One canonical TP4 FFN plus its actual residual boundary and native output reduction. Same operators, two GPU input patterns and live8/16. No whole-model or outcome speed claim."}))
}
