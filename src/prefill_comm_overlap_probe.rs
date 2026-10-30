//! Concurrency feasibility for two independent prefill microbatches. This
//! measures an overlap envelope, not a model pipeline or a serving schedule.
use crate::{blas::Blas,collective::Nccl,cuda::{Buffer,Cuda,Module,Stream},
    expert_packing::Prepared,ffn::{LayerFfn,TokenRows},moe::Workspace,weights::Weights};
use anyhow::{ensure,Result};
use half::bf16;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,rc::Rc};

fn digest<T:crate::cuda::DeviceElement>(cuda:&Cuda,b:&Buffer<T>)->Result<String>{
    let mut hash=Sha256::new();
    for offset in (0..b.bytes()).step_by(4*1024*1024) {
        let count=(b.bytes()-offset).min(4*1024*1024);
        hash.update(unsafe{cuda.read_device(b.ptr()+offset as u64,count)}?);
    }
    Ok(format!("{:x}",hash.finalize()))
}
fn fill(module:&Module,b:&Buffer<bf16>,seed:i32,stream:&Stream)->Result<()>{
    let mut ptr=b.ptr();let mut n=i32::try_from(b.len())?;let mut seed=seed;let mut pattern=0i32;
    let mut args=[(&mut ptr as *mut u64).cast(),(&mut n as *mut i32).cast(),
        (&mut seed as *mut i32).cast(),(&mut pattern as *mut i32).cast()];
    unsafe{module.function("ca_collective_fixture")?.launch([(n as u32).div_ceil(256),1,1],[256,1,1],0,stream,&mut args)}
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,module:&Module,stream:&Stream,weights:Rc<Weights>,prepared:Prepared,
    kernels:&Path,rank:usize,endpoint:&str,ticket:&[u8])->Result<Value>{
    ensure!(std::env::var("CANADA_SCRATCH_PREFILL_COMM_OVERLAP_SCREEN").as_deref()==Ok("1"),
        "prefill communication overlap is an isolated component");
    const M:usize=8192;const K:usize=5120;
    let blas=Rc::new(Blas::new(cuda)?);
    let ffn=LayerFfn::new_rows(cuda,module,stream,weights,prepared,kernels,M,Some(blas))?;
    if std::env::var("CANADA_SCRATCH_PREFILL_ROUTER_OVERLAP_SCREEN").as_deref()==Ok("1") {
        return router_overlap(cuda,module,stream,&ffn,kernels,rank,endpoint,ticket);
    }
    let workspace=Workspace::new(cuda,&kernels.join("moe-runtime-m8192-v0"))?;
    let input=cuda.allocate::<bf16>(M*K)?;
    let mut tokens=TokenRows::new(cuda,&vec![1001;M],M)?;
    let api=Nccl::load_qualified_prefill()?;
    let id=crate::rendezvous::join(&api,rank,endpoint.parse()?,ticket)?;
    let comm=api.join(cuda,id,rank,4)?;
    let communication_stream=cuda.auxiliary_stream(stream)?;
    let start_event=cuda.fence_event()?;let done_event=cuda.fence_event()?;
    // A packed input gather from one microbatch, and BF16 output reduction
    // from another. They deliberately do not consume this FFN's current output.
    // Exposing such independence in the real model is a separate prerequisite.
    let packed_local=cuda.allocate::<u8>((M/4)*5920)?;
    let mut packed_full=cuda.allocate::<u8>(M*5920)?;
    let reduction_in=cuda.allocate::<bf16>(M*K)?;
    let mut reduction_out=cuda.allocate::<bf16>((M/4)*K)?;
    let mut barrier_in=cuda.allocate::<u16>(4)?;barrier_in.upload(&[0;4])?;
    let mut barrier_out=cuda.allocate::<u16>(4)?;
    stream.join_default()?;
    // Generate transport bytes on GPU; gather uses the production byte type
    // and preserves every input bit without doing floating-point arithmetic.
    let packed_seed=cuda.allocate::<bf16>(packed_local.len()/2)?;
    fill(module,&packed_seed,rank as i32,stream)?;
    unsafe{cuda.copy_device(packed_local.ptr(),packed_seed.ptr(),packed_seed.bytes(),stream)?;}
    fill(module,&reduction_in,rank as i32,stream)?;
    fill(module,&input,0,stream)?;
    ffn.run(&workspace,&input,&tokens,stream)?;
    stream.synchronize()?;
    for kind in [0,1] {
        if kind==0 {unsafe{comm.gather_rows(&packed_local,&mut packed_full,&communication_stream)?;}}
        else{unsafe{comm.reduce_scatter_bf16(&reduction_in,&mut reduction_out,&communication_stream)?;}}
    }
    communication_stream.synchronize()?;comm.check_async()?;
    let mut records=Vec::new();
    for kind in [0,1] {
        let mut collect=|s:&Stream|->Result<()>{unsafe{
            if kind==0 {comm.gather_rows(&packed_local,&mut packed_full,s)}
            else{comm.reduce_scatter_bf16(&reduction_in,&mut reduction_out,s)}
        }};
        let compute=stream.capture(||ffn.run(&workspace,&input,&tokens,stream))?;
        let transport=stream.capture(||collect(stream))?;
        let serial=stream.capture(||{ffn.run(&workspace,&input,&tokens,stream)?;collect(stream)})?;
        let overlap=stream.capture(||{
            start_event.record(stream)?;start_event.wait(&communication_stream)?;
            collect(&communication_stream)?;
            done_event.record(&communication_stream)?;
            ffn.run(&workspace,&input,&tokens,stream)?;
            done_event.wait(stream)
        })?;
        let graphs=[compute,transport,serial,overlap];
        for pattern in 0..2 {
            fill(module,&input,pattern,stream)?;
            let mut ids=vec![1001;M];if pattern==1 {for i in (0..M).step_by(127){ids[i]=129264;}}
            tokens.update(&ids,M)?;stream.join_default()?;
            unsafe{graphs[2].launch(stream)?;}stream.synchronize()?;comm.check_async()?;
            let reference=digest(cuda,ffn.output())?;
            let network_reference=if kind==0{digest(cuda,&packed_full)?}else{digest(cuda,&reduction_out)?};
            let mut timings=Vec::new();
            for repeat in 0..2 {
                for mode in if repeat==0{[0,1,2,3]}else{[3,2,1,0]} {
                    unsafe{comm.sum_bf16_to(&barrier_in,&mut barrier_out,stream)?;}
                    stream.synchronize()?;
                    let start=cuda.event()?;let end=cuda.event()?;
                    start.record(stream)?;
                    for _ in 0..3 {unsafe{graphs[mode].launch(stream)?;}}
                    end.record(stream)?;
                    let ms=f64::from(end.elapsed_since(&start)?)/3.;comm.check_async()?;
                    ensure!(digest(cuda,ffn.output())?==reference,"prefill overlap changed FFN output");
                    let network=if kind==0{digest(cuda,&packed_full)?}else{digest(cuda,&reduction_out)?};
                    ensure!(network==network_reference,"prefill overlap changed collective output");
                    timings.push(json!({"repeat":repeat,"mode":(["compute","communication","serial","overlap"][mode]),"ms":ms}));
                }
            }
            records.push(json!({"collective":if kind==0{"packed_gather"}else{"bf16_reduce_scatter"},
                "pattern":pattern,"ffn_sha256":reference,"network_sha256":network_reference,
                "all_exact":true,"timings":timings}));
        }
        drop(graphs);
    }
    let grouped=if std::env::var("CANADA_SCRATCH_GROUPED_PREFILL_GATHER_SCREEN").as_deref()==Ok("1") {
        Some(grouped_gather(cuda,module,stream,&comm,rank,M,&packed_local,&mut packed_full)?)
    }else{None};
    stream.synchronize()?;comm.check_async()?;
    Ok(json!({"rank":rank,"prefill_comm_overlap_probe":true,"rows":M,"all_exact":true,"all_bitwise_equal":true,"records":records,
        "grouped_gather":grouped,
        "scope":"One canonical TP4 FFN against independent same-sized prefill collectives. Two input patterns, two opposite-order three-replay pairs. Concurrency envelope only: no dependency pipeline, microbatch overhead or whole-model gain included."}))
}

#[allow(clippy::too_many_arguments)]
fn router_overlap(cuda:&Cuda,module:&Module,stream:&Stream,ffn:&LayerFfn,kernels:&Path,
    rank:usize,endpoint:&str,ticket:&[u8])->Result<Value>{
    use std::ffi::c_void;
    const LOCAL:usize=2048;
    type Pack=unsafe extern "C" fn(*const c_void,*mut c_void,i32,i32,*mut c_void)->i32;
    let path=kernels.join("moe-input-codec-fast-v1/libcanada_moe_input_codec.so");
    ensure!(format!("{:x}",Sha256::digest(std::fs::read(&path)?))
        =="ef5e336f5fb1a00f02e7910bb7800db724bda14bc2055804b5b7dba0aa5e837c",
        "local router probe requires the carried packed-input codec");
    let library=unsafe{libloading::Library::new(path)}?;
    let pack:Pack=*unsafe{library.get(b"ca_pack_bf16\0")}?;
    // Each cuBLAS handle and router owns one stream. These two small owners
    // compare the same canonical operations; neither switches a shared handle.
    let serial_router=ffn.prefill_router_for_probe(kernels,Rc::new(Blas::new(cuda)?))?;
    let parallel_router=ffn.prefill_router_for_probe(kernels,Rc::new(Blas::new(cuda)?))?;
    let input=cuda.allocate::<bf16>(LOCAL*5120)?;
    let packed=cuda.allocate::<u8>(LOCAL*5920)?;
    let mut gathered=cuda.allocate::<u8>(4*packed.len())?;
    let mut all_ids=cuda.allocate::<i32>(4*serial_router.ids().len())?;
    let mut all_weights=cuda.allocate::<f32>(4*serial_router.routing().len())?;
    let mut tokens=TokenRows::new(cuda,&vec![1001;LOCAL],LOCAL)?;
    let side=cuda.auxiliary_stream(stream)?;
    let ready=cuda.fence_event()?;let done=cuda.fence_event()?;
    let api=Nccl::load_qualified_prefill()?;
    let id=crate::rendezvous::join(&api,rank,endpoint.parse()?,ticket)?;
    let comm=api.join(cuda,id,rank,4)?;
    let mut barrier_in=cuda.allocate::<u16>(4)?;barrier_in.upload(&[0;4])?;
    let mut barrier_out=cuda.allocate::<u16>(4)?;
    let produce=||->Result<()>{
        ensure!(unsafe{pack(input.ptr()as*const c_void,packed.ptr()as*mut c_void,
            LOCAL as i32,5120,stream.raw())}==0,"local router probe codec launch");Ok(())
    };
    fill(module,&input,rank as i32,stream)?;stream.join_default()?;
    ready.record(stream)?;ready.wait(&side)?;
    parallel_router.run(&input,&tokens,&side)?;done.record(&side)?;done.wait(stream)?;
    serial_router.run(&input,&tokens,stream)?;produce()?;
    unsafe{comm.gather_rows(&packed,&mut gathered,stream)?;
        comm.gather_rows(serial_router.ids(),&mut all_ids,stream)?;
        comm.gather_rows(serial_router.routing(),&mut all_weights,stream)?;}
    stream.synchronize()?;comm.check_async()?;
    let mut execute=|mode:usize|->Result<()>{
        let router=if mode==2{&parallel_router}else{&serial_router};
        if mode==2 {
            ready.record(stream)?;ready.wait(&side)?;
            router.run(&input,&tokens,&side)?;done.record(&side)?;
            produce()?;
            unsafe{comm.gather_rows(&packed,&mut gathered,stream)?;}
            done.wait(stream)?;
        } else {
            if mode==1{produce()?;}
            router.run(&input,&tokens,stream)?;
            if mode==0{produce()?;}
            unsafe{comm.gather_rows(&packed,&mut gathered,stream)?;}
        }
        unsafe{comm.gather_rows(router.ids(),&mut all_ids,stream)?;
            comm.gather_rows(router.routing(),&mut all_weights,stream)}
    };
    let serial=stream.capture(||execute(0))?;
    let reorder=stream.capture(||execute(1))?;
    let overlap=stream.capture(||execute(2))?;drop(execute);
    let graphs=[serial,reorder,overlap];let mut records=Vec::new();
    for pattern in 0..2 {
        fill(module,&input,(17*pattern+rank)as i32,stream)?;
        let mut ids=vec![1001;LOCAL];
        if pattern==1{for i in (0..LOCAL).step_by(127){ids[i]=129264;}}
        tokens.update(&ids,if pattern==0{LOCAL}else{LOCAL-73})?;stream.join_default()?;
        unsafe{graphs[0].launch(stream)?;}stream.synchronize()?;comm.check_async()?;
        let reference=[digest(cuda,&gathered)?,digest(cuda,&all_ids)?,digest(cuda,&all_weights)?];
        let logits=digest(cuda,serial_router.logits())?;
        let mut timings=Vec::new();
        for repeat in 0..2 {
            for mode in if repeat==0{[0,1,2]}else{[2,1,0]} {
                unsafe{graphs[mode].launch(stream)?;}
                unsafe{comm.sum_bf16_to(&barrier_in,&mut barrier_out,stream)?;}
                stream.synchronize()?;
                let start=cuda.event()?;let end=cuda.event()?;start.record(stream)?;
                for _ in 0..8{unsafe{graphs[mode].launch(stream)?;}}
                end.record(stream)?;let ms=f64::from(end.elapsed_since(&start)?)/8.;
                comm.check_async()?;
                ensure!([digest(cuda,&gathered)?,digest(cuda,&all_ids)?,digest(cuda,&all_weights)?]==reference,
                    "router overlap changed activation or routing bytes");
                let router=if mode==2{&parallel_router}else{&serial_router};
                ensure!(digest(cuda,router.logits())?==logits,"router overlap changed FP32 logits");
                timings.push(json!({"repeat":repeat,"mode":(["serial","pack_first","overlap"][mode]),"ms":ms}));
            }
        }
        records.push(json!({"pattern":pattern,"all_exact":true,"sha256":reference,"logits_sha256":logits,"timings":timings}));
    }
    drop(graphs);stream.synchronize()?;side.synchronize()?;comm.check_async()?;
    Ok(json!({"rank":rank,"prefill_comm_overlap_probe":true,"router_overlap_probe":true,
        "all_exact":true,"all_bitwise_equal":true,"rows":8192,"local_rows":LOCAL,"records":records,
        "payload_bytes_per_rank":[packed.bytes(),serial_router.ids().bytes(),serial_router.routing().bytes()],
        "additional_production_workspace_bytes":32*1024*1024,
        "scope":"Canonical local2048-row router and carried packed-input producer; actual activation/ID/weight gathers on four peers. Two inputs including VL/padding, two reversed-order eight-replay pairs. Same-layer dependency graph only; no full-model gain claim."}))
}

#[allow(clippy::too_many_arguments)]
fn grouped_gather(cuda:&Cuda,module:&Module,stream:&Stream,comm:&crate::collective::Communicator,
    rank:usize,rows:usize,packed:&Buffer<u8>,gathered:&mut Buffer<u8>)->Result<Value>{
    // Match the production8K MoE gather:5920bytes/row plus8 expert IDs and
    // eight FP32 routing weights. Transport is bitwise, so GPU-generated raw
    // words exercise the same operation without another model evaluation.
    let ids=cuda.allocate::<u32>((rows/4)*8)?;
    let weights=cuda.allocate::<u32>((rows/4)*8)?;
    let mut all_ids=cuda.allocate::<u32>(rows*8)?;
    let mut all_weights=cuda.allocate::<u32>(rows*8)?;
    let seed=cuda.allocate::<bf16>(ids.len()*2)?;
    fill(module,&seed,rank as i32+17,stream)?;
    unsafe{cuda.copy_device(ids.ptr(),seed.ptr(),ids.bytes(),stream)?;}
    fill(module,&seed,rank as i32+41,stream)?;
    unsafe{cuda.copy_device(weights.ptr(),seed.ptr(),weights.bytes(),stream)?;}
    let mut launch=|grouped:bool|->Result<()>{unsafe{
        if grouped {comm.gather_rows_grouped(packed,gathered,&ids,&mut all_ids,&weights,&mut all_weights,stream)}
        else {comm.gather_rows(packed,gathered,stream)?;
            comm.gather_rows(&ids,&mut all_ids,stream)?;comm.gather_rows(&weights,&mut all_weights,stream)}
    }};
    // Both paths are warmed before capture and use the same exact buffers.
    launch(false)?;launch(true)?;stream.synchronize()?;comm.check_async()?;
    let separate=stream.capture(||launch(false))?;
    let grouped=stream.capture(||launch(true))?;drop(launch);
    unsafe{separate.launch(stream)?;}stream.synchronize()?;comm.check_async()?;
    let reference=[digest(cuda,gathered)?,digest(cuda,&all_ids)?,digest(cuda,&all_weights)?];
    let mut records=Vec::new();
    for repeat in 0..2 {
        for enabled in if repeat==0{[false,true]}else{[true,false]} {
            unsafe{(if enabled{&grouped}else{&separate}).launch(stream)?;}
            stream.synchronize()?;
            let start=cuda.event()?;let end=cuda.event()?;start.record(stream)?;
            for _ in 0..16 {unsafe{(if enabled{&grouped}else{&separate}).launch(stream)?;}}
            end.record(stream)?;let ms=f64::from(end.elapsed_since(&start)?)/16.;comm.check_async()?;
            ensure!([digest(cuda,gathered)?,digest(cuda,&all_ids)?,digest(cuda,&all_weights)?]==reference,
                "grouped gather changed activation or routing bytes");
            records.push(json!({"repeat":repeat,"grouped":enabled,"ms":ms}));
        }
    }
    Ok(json!({"all_exact":true,"rows":rows,"payload_bytes_per_rank":[packed.bytes(),ids.bytes(),weights.bytes()],
        "sha256":reference,"records":records,
        "scope":"The three8K MoE payload shapes, two opposite-order16-replay pairs, bitwise comparison. No model integration or overall speed claim."}))
}
