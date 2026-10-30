//! Short paired GPU transport screen; never changes the serving reduction path.
use crate::{collective::Nccl, cuda::{Buffer, Cuda, Module, Stream}, roce::Roce};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{net::SocketAddr, path::Path};

fn convert(module: &Module, name: &str, source: u64, target: u64,
    count: usize, stream: &Stream) -> Result<()> {
    let (mut src,mut dst,mut n)=(source,target,count as i32);
    let mut args=[(&mut src as *mut u64).cast(),(&mut dst as *mut u64).cast(),
        (&mut n as *mut i32).cast()];
    unsafe { module.function(name)?.launch([(count as u32).div_ceil(256),1,1],
        [256,1,1],0,stream,&mut args) }
}
fn fixture(module: &Module, input: &mut Buffer<u16>, rank: usize,
    pattern: usize, stream: &Stream) -> Result<()> {
    let (mut dst,mut n,mut r,mut p)=(input.ptr(),input.len() as i32,rank as i32,pattern as i32);
    let mut args=[(&mut dst as *mut u64).cast(),(&mut n as *mut i32).cast(),
        (&mut r as *mut i32).cast(),(&mut p as *mut i32).cast()];
    unsafe { module.function("ca_collective_fixture")?.launch([(input.len() as u32).div_ceil(256),1,1],
        [256,1,1],0,stream,&mut args) }
}
fn check(cuda: &Cuda, module: &Module, output: &Buffer<u16>, raw: &Buffer<f32>,
    pattern: usize, variant: usize, stream: &Stream) -> Result<Value> {
    let mut stats=cuda.allocate::<u32>(5)?;stats.zero()?;stream.join_default()?;
    let (mut src,mut f,mut out,mut n,mut p,mut has)=(output.ptr(),raw.ptr(),stats.ptr(),
        output.len() as i32,pattern as i32,if variant==2{-1}else{variant as i32});
    let mut args=[(&mut src as *mut u64).cast(),(&mut f as *mut u64).cast(),
        (&mut out as *mut u64).cast(),(&mut n as *mut i32).cast(),
        (&mut p as *mut i32).cast(),(&mut has as *mut i32).cast()];
    unsafe { module.function("ca_collective_check")?.launch([(output.len() as u32).div_ceil(256),1,1],
        [256,1,1],0,stream,&mut args)?; }
    stream.synchronize()?;
    let s=stats.download()?;
    ensure!(s[0]==0 && s[1]==0 && s[4]==0 && s[3]>0,
        "transport GPU FP64 bound/finite/nonzero check failed variant{variant} pattern{pattern}: {s:?}");
    ensure!(variant==2 || pattern!=0 || s[2]==0,"narrow-range FP32 sums must equal GPU FP64 oracle");
    let bytes:Vec<u8>=output.download()?.into_iter().flat_map(u16::to_le_bytes).collect();
    Ok(json!({"gpu_fp64_bounds_pass":true,"different_from_fp64_rounded":s[2],
        "nonzero":s[3],"sha256":format!("{:x}",Sha256::digest(bytes))}))
}

pub fn run(rank: usize, endpoint: &str, nonce: &[u8], root: &Path, core: &Path) -> Result<Value> {
    if std::env::var("CANADA_SCRATCH_PREFILL_COLUMN_GATHER_SCREEN").as_deref()==Ok("1") {
        return run_column_gather(rank,endpoint,nonce,root,core);
    }
    if std::env::var("CANADA_SCRATCH_PREFILL_LOCALITY_SCREEN").as_deref()==Ok("1") {
        return run_locality(rank,endpoint,nonce,root,core);
    }
    if std::env::var("CANADA_SCRATCH_PREFILL_SUM_SCREEN").as_deref()==Ok("1") {
        return run_prefill(rank,endpoint,nonce,root,core);
    }
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;
    let stream=cuda.stream()?;let module=cuda.module(&std::fs::read(core)?)?;
    let native=Roce::local(&cuda,&root.join("roce-v0"),rank)?;
    let mut address:SocketAddr=endpoint.parse()?;
    let all=crate::rendezvous::exchange_blob(rank,address,nonce,&native.blob()?)?;
    native.connect(&all)?;
    address.set_port(address.port().checked_add(2).ok_or_else(||anyhow::anyhow!("probe port overflow"))?);
    let api=Nccl::load_qualified_prefill()?;
    let id=crate::rendezvous::join(&api,rank,address,nonce)?;
    let nccl=api.join(&cuda,id,rank,4)?;
    let mut results=Vec::new();
    for (rows,columns) in [(1,5120),(3,5120),(5,5120),(6,5120),(128,5120),(1,6144),(6,6144)] {
        let n=rows*columns;
        let mut input=cuda.allocate::<u16>(n)?;
        let mut output_native=cuda.allocate::<u16>(n)?;
        let output_nccl=cuda.allocate::<u16>(n)?;
        let wide=cuda.allocate::<f32>(n)?;
        let mut reduced=cuda.allocate::<f32>(n)?;
        fixture(&module,&mut input,rank,0,&stream)?;
        let mut enqueue=|variant:usize| -> Result<()> {
            if variant==0 { unsafe { native.sum_bf16(&input,&mut output_native,&stream) } }
            else {
                convert(&module,"ca_bf16_to_f32",input.ptr(),wide.ptr(),n,&stream)?;
                unsafe { nccl.sum_f32_to(&wide,&mut reduced,&stream)?; }
                convert(&module,"ca_probe_f32_to_bf16",reduced.ptr(),output_nccl.ptr(),n,&stream)
            }
        };
        for v in [0,1] {for _ in 0..4 {enqueue(v)?;}stream.synchronize()?;}
        // A graph contains eight complete reductions; conversion overhead is
        // included for NCCL. No recurrence can turn the payload into zeros/NaNs.
        let g0=stream.capture(|| {for _ in 0..8 {enqueue(0)?;}Ok(())})?;
        let g1=stream.capture(|| {for _ in 0..8 {enqueue(1)?;}Ok(())})?;
        drop(enqueue);
        for pattern in 0..3 {
            fixture(&module,&mut input,rank,pattern,&stream)?;
            let mut checks=Vec::new();
            for (v,g) in [(0,&g0),(1,&g1)] {
                unsafe {g.launch(&stream)?;}stream.synchronize()?;
                checks.push(check(&cuda,&module,if v==0{&output_native}else{&output_nccl},&reduced,pattern,v,&stream)?);
            }
            if pattern==0 {ensure!(checks[0]["sha256"]==checks[1]["sha256"],"exact transport comparison failed");}
            let mut pairs=Vec::new();
            for repeat in 0..6 {
                let mut times=[0f32;2];
                for v in [repeat%2,1-repeat%2] {
                    let g=if v==0{&g0}else{&g1};
                    // Warm the selected graph immediately before measurement.
                    for _ in 0..4 {unsafe {g.launch(&stream)?;}}stream.synchronize()?;
                    let start=cuda.event()?;let end=cuda.event()?;
                    start.record(&stream)?;
                    for _ in 0..32 {unsafe {g.launch(&stream)?;}}
                    end.record(&stream)?;times[v]=end.elapsed_since(&start)?/256.;
                    native.health()?;nccl.check_async()?;
                }
                pairs.push(json!({"repeat":repeat,"first_variant":repeat%2,"milliseconds":times}));
            }
            let after=[check(&cuda,&module,&output_native,&reduced,pattern,0,&stream)?,
                check(&cuda,&module,&output_nccl,&reduced,pattern,1,&stream)?];
            ensure!(checks[0]==after[0] && checks[1]==after[1],"transport replay changed output");
            results.push(json!({"rows":rows,"columns":columns,"pattern":pattern,
                "checks":checks,"pairs":pairs}));
        }
    }
    stream.synchronize()?;native.health()?;nccl.check_async()?;
    Ok(json!({"rank":rank,"nccl_proto":std::env::var("NCCL_PROTO").ok(),
        "nccl_graph_stream_ordering":std::env::var("NCCL_GRAPH_STREAM_ORDERING").ok(),
        "results":results,"all_gpu_checks_pass":true,"serving_policy_changed":false,
        "scope":"BF16 sources; native FP32 sum versus exact promotion, NCCL FP32 sum, one BF16 output rounding. GPU-generated fixtures and FP64 oracle. Timing includes both conversion kernels; six balanced pairs, 256 reductions per interval. Component screen only."}))
}

fn run_column_gather(rank:usize,endpoint:&str,nonce:&[u8],root:&Path,core:&Path)->Result<Value>{
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let bytes=std::fs::read(core)?;let module=cuda.module(&bytes)?;
    let api=Nccl::load_qualified_prefill()?;
    let id=crate::rendezvous::join(&api,rank,endpoint.parse()?,nonce)?;
    let nccl=api.join(&cuda,id,rank,4)?;
    let mut comm=Roce::local(&cuda,&root.join("roce-v0"),rank)?;
    let mut address:SocketAddr=endpoint.parse()?;address.set_port(address.port().checked_add(2)
        .ok_or_else(||anyhow::anyhow!("column gather endpoint overflow"))?);
    let all=crate::rendezvous::exchange_blob(rank,address,nonce,&comm.blob()?)?;comm.connect(&all)?;
    comm.attach_gather(&root.join("roce-gather-v1"))?;comm.attach_prefill_sum(nccl)?;
    comm.attach_prefill_column_gather(&bytes,4096)?;
    let capacity=512;let columns=6400;let n=capacity*columns*4;
    let mut input=cuda.allocate::<u16>(capacity*columns)?;
    let mut storage=(0..2).map(|_|cuda.allocate::<u16>(n+128)).collect::<Result<Vec<_>>>()?;
    for b in &mut storage{b.upload(&vec![0xa55au16;n+128])?;}
    stream.join_default()?;let mut records=Vec::new();
    for live in[129,257,385,512] {
        comm.select_prefill_rows(live,capacity,&stream)?;
        let mut graphs=Vec::new();
        for path in[0,1] {
            comm.set_column_gather_rows_for_probe(path==1,&stream)?;
            let mut output=unsafe{storage[path].view(64,n)}?;
            fixture(&module,&mut input,rank,0,&stream)?;
            let mut enqueue=||unsafe{comm.gather_bf16(&input,&mut output,columns,&stream)};
            enqueue()?;stream.synchronize()?;
            let g=stream.capture(||{for _ in 0..8{enqueue()?;}Ok(())})?;graphs.push(g);
        }
        for pattern in[0,1,2] {
            fixture(&module,&mut input,rank,pattern,&stream)?;
            for g in &graphs{unsafe{g.launch(&stream)}?;}stream.synchronize()?;
            let before=storage.iter().map(Buffer::download).collect::<Result<Vec<_>>>()?;
            let active=live*columns*4;
            ensure!(before[0][64..64+active]==before[1][64..64+active],"column gather active bits changed");
            ensure!(before.iter().all(|x|x[..64].iter().chain(&x[n+64..]).all(|v|*v==0xa55a)),"column gather output guard");
            ensure!(live==capacity||before[1][64+active..64+n].iter().all(|v|*v==0),"column gather unused suffix not zero");
            let mut pairs=Vec::new();
            for repeat in 0..4 {
                let mut times=[0f32;2];
                for path in[repeat%2,1-repeat%2] {
                    unsafe{graphs[path].launch(&stream)}?;stream.synchronize()?;
                    let start=cuda.event()?;let end=cuda.event()?;start.record(&stream)?;
                    for _ in 0..8{unsafe{graphs[path].launch(&stream)}?;}
                    end.record(&stream)?;times[path]=end.elapsed_since(&start)?/64.;comm.health()?;
                }
                pairs.push(json!({"repeat":repeat,"milliseconds":times}));
            }
            for (i,b) in storage.iter().enumerate(){ensure!(b.download()?==before[i],"column gather replay changed output");}
            let raw=before[0][64..64+active].iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>();
            records.push(json!({"live":live,"capacity":capacity,"columns":columns,"pattern":pattern,
                "active_bits_exact":true,"guards":true,"active_sha256":format!("{:x}",Sha256::digest(raw)),"pairs":pairs}));
        }
    }
    ensure!(comm.column_gather_prefix_calls()>0,"column gather screen never exercised the row bound");
    Ok(json!({"rank":rank,"records":records,"all_checks_pass":true,
        "scope":"Existing NCCL byte all-gather and rank-column transpose at M512. Physical versus live129/257/385/512, three changed GPU fixtures, guards and defined zero padding. Four reversed-order timing pairs; eight collectives per CUDA graph. No target arithmetic or precision changes; component only."}))
}

/// Distinguish a warm transport-only benefit from a benefit after substantial
/// unrelated GPU writes. Fixture generation and displacement are inside the
/// graphs and included in every timing; an identical no-collective graph records
/// their common cost. This is not a model traffic simulator or serving claim.
fn run_locality(rank:usize,endpoint:&str,nonce:&[u8],root:&Path,core:&Path)->Result<Value>{
    ensure!(std::env::var("CANADA_SCRATCH_SUM_GRID_SCREEN").as_deref()==Ok("1"),
        "locality screen needs isolated geometry control");
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let module=cuda.module(&std::fs::read(core)?)?;
    let api=Nccl::load_qualified_prefill()?;
    let id=crate::rendezvous::join(&api,rank,endpoint.parse()?,nonce)?;
    let nccl=api.join(&cuda,id,rank,4)?;
    let native=Roce::local(&cuda,&root.join("roce-v0"),rank)?;
    let mut address:SocketAddr=endpoint.parse()?;address.set_port(address.port().checked_add(2)
        .ok_or_else(||anyhow::anyhow!("locality endpoint overflow"))?);
    let all=crate::rendezvous::exchange_blob(rank,address,nonce,&native.blob()?)?;native.connect(&all)?;
    let mut displacement=cuda.allocate::<u16>(64*1024*1024/2)?;
    let mut records=Vec::new();
    for columns in [5120,6144] {
        let n=128*columns;let mut input=cuda.allocate::<u16>(n)?;
        let raw=cuda.allocate::<f32>(n)?;
        let outputs=(0..5).map(|_|cuda.allocate::<u16>(n)).collect::<Result<Vec<_>>>()?;
        for pattern in [0,2] {for displaced in [false,true] {
            let mut graphs=Vec::new();let mut checks=Vec::new();
            for (variant,output) in outputs.iter().enumerate() {
                if variant>=2 {native.set_sum_grid_for_probe([4,8,16][variant-2],&stream)?;}
                let mut out=unsafe{output.view(0,n)}?;
                let mut enqueue=||->Result<()> {
                    if displaced {fixture(&module,&mut displacement,rank,0,&stream)?;}
                    // Produce a hot input after displacing the transport's
                    // buffers; ordinary layers also write their own operand.
                    fixture(&module,&mut input,rank,pattern,&stream)?;
                    match variant {0=>Ok(()),1=>unsafe{nccl.sum_bf16_to(&input,&mut out,&stream)},
                        _=>unsafe{native.sum_bf16(&input,&mut out,&stream)}}
                };
                enqueue()?;stream.synchronize()?;
                // Native model graphs enqueue the whole layer schedule with
                // one host launch. Eight in-graph copies keep per-collective
                // host dispatch/skew out of this second locality screen.
                let graph=stream.capture(|| {for _ in 0..8 {enqueue()?;}Ok(())})?;
                unsafe{graph.launch(&stream)}?;stream.synchronize()?;
                if variant>0 {
                    checks.push(json!({"variant":variant,"check":check(&cuda,&module,output,&raw,pattern,
                        if variant==1{2}else{0},&stream)?}));
                }
                graphs.push(graph);
            }
            let mut pairs=Vec::new();
            for repeat in 0..5 {
                let mut milliseconds=[0f32;5];
                for offset in 0..5 {
                    let variant=(repeat+offset)%5;let graph=&graphs[variant];
                    unsafe{graph.launch(&stream)}?;stream.synchronize()?;
                    let start=cuda.event()?;let end=cuda.event()?;start.record(&stream)?;
                    for _ in 0..8 {unsafe{graph.launch(&stream)}?;}
                    end.record(&stream)?;milliseconds[variant]=end.elapsed_since(&start)?/64.;
                    native.health()?;nccl.check_async()?;
                }
                pairs.push(json!({"repeat":repeat,"first_variant":repeat,"milliseconds":milliseconds}));
            }
            records.push(json!({"rows":128,"columns":columns,"pattern":pattern,
                "displacement_bytes":if displaced{displacement.bytes()}else{0},"checks":checks,"pairs":pairs}));
        }}
    }
    stream.synchronize()?;native.health()?;nccl.check_async()?;
    Ok(json!({"rank":rank,"records":records,"in_graph_repetitions":8,"all_candidate_gpu_bounds_pass":true,
        "variants":["common fixture/displacement only","NCCL BF16","native FP32 grid4","native FP32 grid8","native FP32 grid16"],
        "scope":"Hot operands after zero/64MiB unrelated GPU writes. Two M128 payloads, two GPU-generated input patterns, five rotated pairs. Times include common write/fixture cost, separately measured. GPU FP64 oracle with each transport's own intermediate precision bound; native narrow-range sums exact. No serving change or model speed/quality claim."}))
}

fn run_prefill(rank:usize,endpoint:&str,nonce:&[u8],root:&Path,core:&Path)->Result<Value> {
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let bytes=std::fs::read(core)?;let module=cuda.module(&bytes)?;
    let api=Nccl::load_qualified_prefill()?;
    let id=crate::rendezvous::join(&api,rank,endpoint.parse()?,nonce)?;
    let nccl=api.join(&cuda,id,rank,4)?;let mut records=Vec::new();
    let native=Roce::local(&cuda,&root.join("roce-v0"),rank)?;
    let mut address:SocketAddr=endpoint.parse()?;
    address.set_port(address.port().checked_add(2).ok_or_else(||anyhow::anyhow!("prefill screen endpoint overflow"))?);
    let all=crate::rendezvous::exchange_blob(rank,address,nonce,&native.blob()?)?;native.connect(&all)?;
    let small=std::env::var("CANADA_SCRATCH_PREFILL_SUM_SMALL_SCREEN").as_deref()==Ok("1");
    let shapes=if small{vec![(128,6144),(192,5120),(256,5120),(384,5120)]}
        else{vec![(128,5120),(512,5120),(641,5120),(8192,5120)]};
    for (rows,columns) in shapes {
        let n=rows*columns;let mut input=cuda.allocate::<u16>(n)?;
        let wide=cuda.allocate::<f32>(n)?;let reduced=cuda.allocate::<f32>(n)?;
        let exchange=crate::prefill_exchange::Exchange::new(&cuda,&bytes,n)?;
        let outputs=(0..4).map(|_|cuda.allocate::<u16>(n)).collect::<Result<Vec<_>>>()?;
        let mut gathered=cuda.allocate::<u16>(n)?;
        fixture(&module,&mut input,rank,0,&stream)?;
        for scatter in [false,true] {
            if small&&scatter{continue;}
            let count=if scatter{n/4}else{n};let mut graphs=Vec::new();
            for (variant,output) in outputs.iter().enumerate() {
                let mut out=unsafe{output.view(0,count)}?;
                let mut r=unsafe{reduced.view(0,count)}?;
                let mut enqueue=||->Result<()> {
                    match variant {
                        0=>if scatter{unsafe{nccl.reduce_scatter_bf16(&input,&mut out,&stream)}}
                            else{unsafe{nccl.sum_bf16_to(&input,&mut out,&stream)}},
                        1=>{
                            convert(&module,"ca_bf16_to_f32",input.ptr(),wide.ptr(),n,&stream)?;
                            if scatter{unsafe{nccl.reduce_scatter_f32(&wide,&mut r,&stream)}?;}
                            else{unsafe{nccl.sum_f32_to(&wide,&mut r,&stream)}?;}
                            convert(&module,"ca_f32_to_bf16",r.ptr(),out.ptr(),count,&stream)
                        },
                        2=>unsafe{exchange.run(&nccl,&input,&mut out,n,scatter,&stream)},
                        3=>{
                            // Fixed-rank FP32 arithmetic on original BF16 wire
                            // operands. Respect the smallest qualified 16 MiB slot.
                            let chunk=16*1024*1024/2;
                            for first in (0..n).step_by(chunk) {
                                let len=(n-first).min(chunk);
                                let x=unsafe{input.view(first,len)}?;
                                let mut y=unsafe{output.view(first,len)}?;
                                unsafe{native.sum_bf16(&x,&mut y,&stream)}?;
                            }
                            if scatter&&rank!=0{unsafe{cuda.copy_device(out.ptr(),output.ptr()+(rank*count*2)as u64,count*2,&stream)}?;}
                            Ok(())
                        },
                        _=>unreachable!(),
                    }
                };
                for _ in 0..2{enqueue()?;}stream.synchronize()?;
                graphs.push(stream.capture(&mut enqueue)?);
            }
            for pattern in [0,2] {
                fixture(&module,&mut input,rank,pattern,&stream)?;
                let mut checks=Vec::new();
                for (variant,g) in graphs.iter().enumerate() {
                    unsafe{g.launch(&stream)}?;
                    if scatter {
                        let local=unsafe{outputs[variant].view(0,count)}?;
                        unsafe{nccl.gather_rows(&local,&mut gathered,&stream)}?;
                    }
                    let output=if scatter{&gathered}else{&outputs[variant]};
                    let stats=cuda.allocate::<u32>(5)?;
                    unsafe{cuda.zero_device(stats.ptr(),stats.bytes(),&stream)}?;
                    let(mut src,mut raw,mut dst,mut size,mut p,mut has)=(output.ptr(),wide.ptr(),stats.ptr(),n as i32,pattern as i32,0i32);
                    let mut args=[(&mut src as *mut u64).cast(),(&mut raw as *mut u64).cast(),
                        (&mut dst as *mut u64).cast(),(&mut size as *mut i32).cast(),(&mut p as *mut i32).cast(),(&mut has as *mut i32).cast()];
                    unsafe{module.function("ca_collective_check")?.launch([(n as u32).div_ceil(256),1,1],[256,1,1],0,&stream,&mut args)}?;
                    stream.synchronize()?;let s=stats.download()?;
                    ensure!(s[0]==0&&s[3]>0,"prefill sum nonfinite/empty output");
                    if variant>0{ensure!(s[1]==0&&s[4]==0&&(pattern!=0||s[2]==0),"prefill FP32 sum violated GPU FP64 bound: {s:?}");}
                    checks.push(json!({"variant":variant,"nonfinite":s[0],"outside_fp32_bound":s[1],"different_from_fp64_rounded":s[2],"nonzero":s[3]}));
                }
                let mut pairs=Vec::new();
                for repeat in 0..4 {
                    let mut timings=[0f32;4];
                    for offset in 0..4 {
                        let v=(repeat+offset)%4;let g=&graphs[v];
                        for _ in 0..2{unsafe{g.launch(&stream)}?;}stream.synchronize()?;
                        let a=cuda.event()?;let b=cuda.event()?;a.record(&stream)?;
                        for _ in 0..8{unsafe{g.launch(&stream)}?;}b.record(&stream)?;
                        timings[v]=b.elapsed_since(&a)?/8.;nccl.check_async()?;native.health()?;
                    }
                    pairs.push(json!({"repeat":repeat,"first_variant":repeat,"milliseconds":timings}));
                }
                records.push(json!({"rows":rows,"columns":columns,"scatter":scatter,"pattern":pattern,"checks":checks,"pairs":pairs}));
            }
        }
    }
    stream.synchronize()?;nccl.check_async()?;native.health()?;
        Ok(json!({"rank":rank,"variants":["NCCL BF16","NCCL FP32 with conversions","BF16 quarter exchange plus local FP32 sum","native FP32 sum in 16 MiB chunks, then row slice"],"records":records,
        "small_payload_screen":small,"all_candidate_gpu_bounds_pass":true,"scope":"Four payload sizes; full sum (also row scatter in broad mode); changed narrow-range/cancellation inputs through captured graphs; four rotated timing pairs. GPU FP64 oracle. No model/output-quality or serving claim."}))
}
