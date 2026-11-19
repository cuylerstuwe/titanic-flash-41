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
    if std::env::var("CANADA_SCRATCH_ADAPTIVE_RDMA_SCREEN").as_deref()==Ok("1") {
        return run_adaptive_packets(rank,endpoint,nonce,root,core);
    }
    if std::env::var("CANADA_SCRATCH_TWO_OWNER_TRANSPORT_SCREEN").as_deref()==Ok("1") {
        return run_two_owner_pool(rank,endpoint,nonce,root,core);
    }
    if std::env::var("CANADA_SCRATCH_PREFILL_COLUMN_GATHER_SCREEN").as_deref()==Ok("1") {
        return run_column_gather(rank,endpoint,nonce,root,core);
    }
    if std::env::var("CANADA_SCRATCH_PREFILL_LOCALITY_SCREEN").as_deref()==Ok("1") {
        return run_locality(rank,endpoint,nonce,root,core);
    }
    if std::env::var("CANADA_SCRATCH_PREFILL_SUM_SCREEN").as_deref()==Ok("1") {
        return run_prefill(rank,endpoint,nonce,root,core);
    }
    if std::env::var("CANADA_SCRATCH_ONESHOT_TWIN_SCREEN").as_deref()==Ok("1") {
        return run_oneshot_twin(rank,endpoint,nonce,root,core);
    }
    if std::env::var("CANADA_SCRATCH_ONESHOT_HC_SCREEN").as_deref()==Ok("1") {
        return run_oneshot_hc(rank,endpoint,nonce,root,core);
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

/// Same allocation, GPU graph and arithmetic; only CPU-side QP selection varies.
fn run_adaptive_packets(rank:usize,endpoint:&str,nonce:&[u8],root:&Path,core:&Path)->Result<Value> {
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let module=cuda.module(&std::fs::read(core)?)?;
    let comm=Roce::local(&cuda,&root.join("roce-v0"),rank)?;
    let all=crate::rendezvous::exchange_blob(rank,endpoint.parse()?,nonce,&comm.blob()?)?;comm.connect(&all)?;
    let mut results=Vec::new();let thresholds=[0u32,96*1024,512*1024];
    for (rows,columns) in [(1,5120),(6,5120),(8,5120),(12,5120),(16,5120),(32,5120),(128,5120),(6,6144)] {
        let n=rows*columns;let mut input=cuda.allocate::<u16>(n)?;
        let mut output=cuda.allocate::<u16>(n)?;let unused=cuda.allocate::<f32>(n)?;
        fixture(&module,&mut input,rank,0,&stream)?;
        let graph=stream.capture(||{for _ in 0..8 {unsafe{comm.sum_bf16(&input,&mut output,&stream)?;}}Ok(())})?;
        for pattern in 0..3 {
            fixture(&module,&mut input,rank,pattern,&stream)?;
            let mut checks=Vec::new();
            for &limit in &thresholds {
                comm.set_small_packet_bytes_for_probe(limit,&stream)?;
                unsafe{graph.launch(&stream)?;}stream.synchronize()?;
                checks.push(check(&cuda,&module,&output,&unused,pattern,0,&stream)?);
            }
            ensure!(checks.iter().all(|x|x==&checks[0]),"packet policy changed native output");
            let mut pairs=Vec::new();
            for repeat in 0..6 {
                let order=if repeat%2==0 {[0,1,2]}else{[2,1,0]};let mut ms=[0.;3];
                for v in order {
                    comm.set_small_packet_bytes_for_probe(thresholds[v],&stream)?;
                    for _ in 0..4{unsafe{graph.launch(&stream)?;}}stream.synchronize()?;
                    let start=cuda.event()?;let end=cuda.event()?;start.record(&stream)?;
                    for _ in 0..16{unsafe{graph.launch(&stream)?;}}
                    end.record(&stream)?;ms[v]=end.elapsed_since(&start)?/128.;comm.health()?;
                }
                pairs.push(json!({"repeat":repeat,"order":order,"milliseconds":ms}));
            }
            ensure!(check(&cuda,&module,&output,&unused,pattern,0,&stream)?==checks[0],
                "packet replay changed native output");
            results.push(json!({"rows":rows,"columns":columns,"pattern":pattern,"checks":checks,"pairs":pairs}));
        }
    }
    stream.synchronize()?;comm.health()?;let counts=comm.packet_counts_for_probe()?;
    ensure!(counts.iter().all(|&n|n>0),"both packet banks must execute");
    Ok(json!({"rank":rank,"all_gpu_checks_pass":true,"threshold_bytes":thresholds,
        "packet_counts_bulk_small":counts,"results":results,"serving_policy_changed":false,
        "scope":"Single allocation and captured native graph, balanced threshold switches; GPU fixtures/FP64 bound and exact outputs across packet policies. No model or NCCL arithmetic changes."}))
}

/// Price the existing communication kernels before introducing a batched
/// model/state owner. Compare two contiguous row slices with one combined
/// payload; no arithmetic order, target precision or serving path changes.
fn run_two_owner_pool(rank:usize,endpoint:&str,nonce:&[u8],root:&Path,core:&Path)->Result<Value> {
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let module=cuda.module(&std::fs::read(core)?)?;
    let mut comm=Roce::local(&cuda,&root.join("roce-v0"),rank)?;
    comm.attach_gather(&root.join("roce-gather-v1"))?;
    let blobs=crate::rendezvous::exchange_blob(rank,endpoint.parse()?,nonce,&comm.blob()?)?;
    comm.connect(&blobs)?;let mut records=Vec::new();
    for rows in [3usize,6,16] {
        for (kind,columns,gather) in [("residual_sum",5120,false),("qkv_gather",448,true),
            ("engram_gather",6400,true),("logit_raw_gather",64640,true)] {
            // FP32 logit gathers use the same raw16-byte-pack primitive;
            //64640u16 words are32320FP32 words per rank and row.
            let n=2*rows*columns;let out_n=n*if gather{4}else{1};
            let mut input=cuda.allocate::<u16>(n)?;
            let mut storage=(0..2).map(|_|cuda.allocate::<u16>(out_n+128)).collect::<Result<Vec<_>>>()?;
            for x in &mut storage{x.upload(&vec![0xa55a;out_n+128])?;}
            let mut outputs=storage.iter().map(|x|unsafe{x.view(64,out_n)}).collect::<Result<Vec<_>>>()?;
            let raw=cuda.allocate::<f32>(out_n)?;
            fixture(&module,&mut input,rank,0,&stream)?;stream.join_default()?;
            let mut graphs=Vec::new();
            for variant in 0..2 {
                let out=&mut outputs[variant];
                let enqueue=||->Result<()>{
                    let parts=if variant==0{2}else{1};
                    for part in 0..parts {
                        let x=unsafe{input.view(part*n/parts,n/parts)}?;
                        let mut y=unsafe{out.view(part*out_n/parts,out_n/parts)}?;
                        if gather {unsafe{comm.gather_bf16(&x,&mut y,columns,&stream)}?;}
                        else {unsafe{comm.sum_bf16(&x,&mut y,&stream)}?;}
                    }Ok(())
                };
                enqueue()?;stream.synchronize()?;
                graphs.push(stream.capture(||{for _ in 0..8{enqueue()?;}Ok(())})?);
            }
            let mut checks=Vec::new();let mut pairs=Vec::new();
            for pattern in [0,2] {
                fixture(&module,&mut input,rank,pattern,&stream)?;
                for graph in &graphs{unsafe{graph.launch(&stream)}?;}
                stream.synchronize()?;
                let before=storage.iter().map(Buffer::download).collect::<Result<Vec<_>>>()?;
                ensure!(before[0]==before[1],"pooled transport changed {kind} rows{rows} pattern{pattern}");
                ensure!(before.iter().all(|x|x[..64].iter().chain(&x[out_n+64..]).all(|&v|v==0xa55a)),"pooled transport guard overwritten");
                let oracle=if gather {
                    let rank_major=cuda.allocate::<u16>(out_n)?;
                    let expected=cuda.allocate::<u16>(out_n)?;
                    for peer in 0..4 {
                        let mut peer_input=unsafe{rank_major.view(peer*n,n)}?;
                        fixture(&module,&mut peer_input,peer,pattern,&stream)?;
                    }
                    let (mut src,mut dst,mut r,mut c)=(rank_major.ptr(),expected.ptr(),(2*rows)as i32,(columns/8)as i32);
                    let mut args=[(&mut src as *mut u64).cast(),(&mut dst as *mut u64).cast(),
                        (&mut r as *mut i32).cast(),(&mut c as *mut i32).cast()];
                    unsafe {module.function("ca_transpose_rank_columns")?.launch(
                        [(out_n/8).div_ceil(256)as u32,1,1],[256,1,1],0,&stream,&mut args)}?;
                    stream.synchronize()?;
                    ensure!(expected.download()?==before[0][64..out_n+64],"pooled gather differs from GPU-generated expected peers");
                    json!({"gpu_peer_fixture_and_transpose_exact":true})
                }else{check(&cuda,&module,&outputs[1],&raw,pattern,0,&stream)?};
                checks.push(json!({"pattern":pattern,"exact":true,"guards":true,"gpu_oracle":oracle,
                    "sha256":format!("{:x}",Sha256::digest(before[0][64..out_n+64].iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>()))}));
                for repeat in 0..6 {
                    let mut times=[0f32;2];
                    for variant in [repeat%2,1-repeat%2] {
                        for _ in 0..4 {unsafe{graphs[variant].launch(&stream)}?;}
                        stream.synchronize()?;
                        let a=cuda.event()?;let b=cuda.event()?;a.record(&stream)?;
                        for _ in 0..16 {unsafe{graphs[variant].launch(&stream)}?;}
                        b.record(&stream)?;times[variant]=b.elapsed_since(&a)?/128.;comm.health()?;
                    }
                    pairs.push(json!({"pattern":pattern,"repeat":repeat,"first_variant":repeat%2,"milliseconds_per_pair":times}));
                }
                for (i,x) in storage.iter().enumerate(){ensure!(x.download()?==before[i],"pooled transport replay changed bytes");}
            }
            records.push(json!({"kind":kind,"rows_per_owner":rows,"pooled_rows":2*rows,"local_columns_u16":columns,
                "input_bytes_per_owner":n,"checks":checks,"pairs":pairs}));
        }
    }
    stream.synchronize()?;comm.health()?;
    Ok(json!({"rank":rank,"all_exact":true,"records":records,
        "scope":"Two existing native collectives versus one combined row payload on four devices. Twelve shapes, two changed GPU fixtures, output guards, six reversed-order timing pairs per fixture; native sums retain FP32 accumulation and BF16 output. Warm captured-graph transport only, no row-pack cost, competing model traffic or whole-model/serving claim."}))
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

/// Verify-layer stage 1 (a), Nov 18 2026: the CUDA C++ twin of the exported b12x one-shot sum
/// (kernels/roce_oneshot_twin.cuh) against the exported kernel on the same proxy and counters: every shape and
/// fixture must give identical bytes, eager and inside captured graphs, with the two kernels alternating (they share
/// the epoch and counters); then balanced timing pairs of eight-sum graphs.
fn run_oneshot_twin(rank:usize,endpoint:&str,nonce:&[u8],root:&Path,core:&Path)->Result<Value> {
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let core_bytes=std::fs::read(core)?;let module=cuda.module(&core_bytes)?;
    let mut comm=Roce::local(&cuda,&root.join("roce-v0"),rank)?;
    let all=crate::rendezvous::exchange_blob(rank,endpoint.parse()?,nonce,&comm.blob()?)?;comm.connect(&all)?;
    comm.attach_oneshot_twin(&core_bytes,true)?;
    let digest=|b:&Buffer<u16>|->Result<String>{
        Ok(format!("{:x}",Sha256::digest(b.download()?.into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>())))};
    let mut results=Vec::new();
    for (rows,columns) in [(1,5120),(2,5120),(3,5120),(6,5120),(16,5120),(128,5120),(1,6144),(6,6144)] {
        let n=rows*columns;
        let mut input=cuda.allocate::<u16>(n)?;
        let mut exported=cuda.allocate::<u16>(n)?;let mut twin=cuda.allocate::<u16>(n)?;
        let mut checks=Vec::new();
        for pattern in 0..3 {
            fixture(&module,&mut input,rank,pattern,&stream)?;
            // Eager, alternating twice: exported, twin, exported, twin (the epoch passes between the kernels).
            let mut sums=Vec::new();
            for round in 0..2 {
                for twin_on in [false,true] {
                    comm.set_oneshot_twin_for_probe(twin_on,&stream)?;
                    let out=if twin_on {&mut twin} else {&mut exported};
                    out.zero()?;stream.join_default()?;
                    unsafe { comm.sum_bf16(&input,out,&stream)?; }
                    stream.synchronize()?;comm.health()?;
                    sums.push((round,twin_on,digest(if twin_on {&twin} else {&exported})?));
                }
            }
            let first=&sums[0].2;
            ensure!(sums.iter().all(|(_,_,d)|d==first),"one-shot twin differs from the exported kernel: rows{rows} columns{columns} pattern{pattern}: {sums:?}");
            let nonzero=exported.download()?.iter().filter(|&&x|x!=0).count();
            ensure!(nonzero>0,"one-shot sums are all zero");
            checks.push(json!({"pattern":pattern,"identical":true,"sha256":first,"nonzero":nonzero}));
        }
        // Captured graphs of eight sums each; launched alternately, outputs compared after each launch.
        comm.set_oneshot_twin_for_probe(false,&stream)?;
        let g0=stream.capture(||{for _ in 0..8 {unsafe{comm.sum_bf16(&input,&mut exported,&stream)?;}}Ok(())})?;
        comm.set_oneshot_twin_for_probe(true,&stream)?;
        let g1=stream.capture(||{for _ in 0..8 {unsafe{comm.sum_bf16(&input,&mut twin,&stream)?;}}Ok(())})?;
        comm.set_oneshot_twin_for_probe(false,&stream)?;
        for _ in 0..3 {
            unsafe{g0.launch(&stream)?;g1.launch(&stream)?;}stream.synchronize()?;comm.health()?;
            ensure!(digest(&exported)?==digest(&twin)?,"captured one-shot twin differs from the exported kernel: rows{rows} columns{columns}");
        }
        let mut pairs=Vec::new();
        for repeat in 0..6 {
            let mut times=[0f32;2];
            for v in [repeat%2,1-repeat%2] {
                let g=if v==0{&g0}else{&g1};
                for _ in 0..4 {unsafe{g.launch(&stream)?;}}stream.synchronize()?;
                let start=cuda.event()?;let end=cuda.event()?;
                start.record(&stream)?;
                for _ in 0..32 {unsafe{g.launch(&stream)?;}}
                end.record(&stream)?;times[v]=end.elapsed_since(&start)?/256.;
                comm.health()?;
            }
            pairs.push(json!({"repeat":repeat,"first_variant":repeat%2,"milliseconds_exported_twin":times}));
        }
        ensure!(digest(&exported)?==digest(&twin)?,"timed replays changed the comparison");
        results.push(json!({"rows":rows,"columns":columns,"checks":checks,"pairs":pairs}));
    }
    stream.synchronize()?;comm.health()?;
    Ok(json!({"rank":rank,"results":results,"all_identical":true,"serving_policy_changed":false,
        "scope":"Exported b12x one-shot BF16 sum versus its CUDA C++ twin on the same proxy, counters and slots: three GPU fixtures, eight shapes, eager alternation and captured graphs, bytes compared; six balanced timing pairs of eight-sum graphs, 256 sums an interval. Component screen only."}))
}

/// splitmix64 values for the boundary screen's host fixtures (the same seed on every rank: the engine's residual
/// stream and coefficients are replicated; the partial sums differ by rank).
struct Fixture(u64);
impl Fixture {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    /// Roughly normal (sum of four uniforms, centred), times `scale`.
    fn normal(&mut self, scale: f32) -> f32 {
        let s: f32 = (0..4).map(|_| (self.next() >> 40) as f32 / (1u64 << 24) as f32).sum();
        (s - 2.0) * 1.7320508 * scale
    }
}
/// The boundary screen's buffers: shared inputs, then the reference chain's (r_) and the fused chain's (f_) outputs.
struct HcBuffers {
    partial: Buffer<half::bf16>, residual: Buffer<half::bf16>, post: Buffer<f32>, comb: Buffer<f32>, pre: Buffer<f32>,
    weight: Buffer<half::bf16>, value: Buffer<half::bf16>,
    r_sum: Buffer<half::bf16>, f_sum: Buffer<half::bf16>, r_stream: Buffer<half::bf16>, f_stream: Buffer<half::bf16>,
    r_y: Buffer<half::bf16>, f_y: Buffer<half::bf16>, r_q: Buffer<u8>, f_q: Buffer<u8>,
    r_sr: Buffer<u8>, f_sr: Buffer<u8>, r_sm: Buffer<u8>, f_sm: Buffer<u8>,
}
impl HcBuffers {
    fn new(cuda: &Cuda, rows: usize) -> Result<Self> {
        let mut b = Self {
            partial: cuda.allocate(rows * 5120)?, residual: cuda.allocate(rows * 20480)?, post: cuda.allocate(rows * 4)?,
            comb: cuda.allocate(rows * 16)?, pre: cuda.allocate(rows * 4)?, weight: cuda.allocate(5120)?,
            value: cuda.allocate(rows * 5120)?, r_sum: cuda.allocate(rows * 5120)?, f_sum: cuda.allocate(rows * 5120)?,
            r_stream: cuda.allocate(rows * 20480)?, f_stream: cuda.allocate(rows * 20480)?, r_y: cuda.allocate(rows * 5120)?,
            f_y: cuda.allocate(rows * 5120)?, r_q: cuda.allocate(rows * 5120)?, f_q: cuda.allocate(rows * 5120)?,
            r_sr: cuda.allocate(rows * 160)?, f_sr: cuda.allocate(rows * 160)?, r_sm: cuda.allocate(20480)?,
            f_sm: cuda.allocate(20480)?,
        };
        b.clear()?;
        Ok(b)
    }
    fn clear(&mut self) -> Result<()> {
        for x in [&mut self.r_sum, &mut self.f_sum, &mut self.r_stream, &mut self.f_stream, &mut self.r_y, &mut self.f_y,
            &mut self.value] { x.zero()?; }
        for x in [&mut self.r_q, &mut self.f_q, &mut self.r_sr, &mut self.f_sr, &mut self.r_sm, &mut self.f_sm] { x.zero()?; }
        Ok(())
    }
}
/// Production's six-row chain: the exported one-shot sum, the PDL pair's post-mix, its layout8 norm-quant.
fn hc_reference(comm: &Roce, post_op: &crate::triton_ops::Operator, quant: &crate::triton_ops::Operator, b: &mut HcBuffers,
    rows: usize, stream: &Stream) -> Result<()> {
    unsafe {
        comm.sum_bf16(&b.partial, &mut b.r_sum, stream)?;
        post_op.launch(rows as u32, &[b.r_sum.ptr(), b.residual.ptr(), b.post.ptr(), b.comb.ptr(), b.r_stream.ptr()], stream)?;
        quant.launch(rows as u32, &[b.r_stream.ptr(), b.pre.ptr(), b.weight.ptr(), b.r_y.ptr(), b.r_q.ptr(), b.r_sr.ptr(),
            b.r_sm.ptr()], stream)
    }
}
/// The fused chain: the one-shot sum with the boundary epilogue, then the norm-quant variant on the collapsed row.
fn hc_fused(comm: &Roce, variant: &crate::triton_ops::Operator, b: &mut HcBuffers, rows: usize, stream: &Stream) -> Result<()> {
    unsafe {
        comm.sum_hc_bf16(&b.partial, &mut b.f_sum, &b.residual, &b.post, &b.comb, &b.pre, &b.f_stream, &b.value, rows, stream)?;
        variant.launch(rows as u32, &[b.value.ptr(), b.weight.ptr(), b.f_y.ptr(), b.f_q.ptr(), b.f_sr.ptr(), b.f_sm.ptr()], stream)
    }
}
fn hc_compare(b: &HcBuffers, label: &str, rows: usize) -> Result<Value> {
    let b16 = |x: &Buffer<half::bf16>| -> Result<Vec<u8>> { Ok(x.download()?.into_iter().flat_map(|v| v.to_bits().to_le_bytes()).collect()) };
    let pairs = [("sum", b16(&b.r_sum)?, b16(&b.f_sum)?), ("stream", b16(&b.r_stream)?, b16(&b.f_stream)?),
        ("normalized", b16(&b.r_y)?, b16(&b.f_y)?), ("values", b.r_q.download()?, b.f_q.download()?),
        ("row_scales", b.r_sr.download()?, b.f_sr.download()?), ("mma_scales", b.r_sm.download()?, b.f_sm.download()?)];
    let mut digests = serde_json::Map::new();
    for (name, x, y) in &pairs {
        let d = x.iter().zip(y).filter(|(u, v)| u != v).count();
        ensure!(d == 0, "fused boundary differs from production's chain: {label} rows{rows} {name}: {d} of {} bytes", x.len());
        digests.insert(name.to_string(), json!(format!("{:x}", Sha256::digest(x))));
    }
    let nonzero = pairs[2].1.chunks(2).filter(|c| c[0] != 0 || c[1] != 0).count();
    Ok(json!({"case": label, "identical": true, "sha256": digests, "normalized_nonzero": nonzero}))
}
/// Verify-layer stage 1 (b/c/d), Nov 18 2026: the fused boundary sum (kernels/roce_oneshot_hc.cuh) followed by the
/// norm-quant variant reading its collapsed row (primitive-exports/mhc-value-norm-quant-v1, PDL form) against
/// production's six-row chain on the same inputs: the exported one-shot sum, the PDL pair's post-mix and its layout8
/// norm-quant. Every output byte must agree (reduced sum, new stream, normalized row, MXFP8 values, both scale layouts)
/// for six fixtures, four row counts and every fused grid, eager and inside captured graphs; then balanced timing pairs
/// of eight-boundary graphs.
fn run_oneshot_hc(rank:usize,endpoint:&str,nonce:&[u8],root:&Path,core:&Path)->Result<Value> {
    use half::bf16;
    use crate::triton_ops::Operator;
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let core_bytes=std::fs::read(core)?;
    let mut comm=Roce::local(&cuda,&root.join("roce-v0"),rank)?;
    let all=crate::rendezvous::exchange_blob(rank,endpoint.parse()?,nonce,&comm.blob()?)?;comm.connect(&all)?;
    comm.attach_oneshot_twin(&core_bytes,true)?;
    comm.set_oneshot_twin_for_probe(false,&stream)?;
    comm.select_decode_reduction(true,&stream)?;
    let post_op=Operator::load(&cuda,&root.join("mhc-pdl-pair-v1"),"hc_post_b128_w1")?;
    let quant=Operator::load(&cuda,&root.join("mhc-pdl-pair-v1"),"hc_norm_quant_p2")?;
    let variant=Operator::load(&cuda,&root.join("mhc-value-norm-quant-v1"),"hc_value_norm_quant_p2_pdl")?;
    ensure!(post_op.entry.programmatic_stream_serialization && quant.entry.programmatic_stream_serialization
        && variant.entry.programmatic_stream_serialization,"production PDL pair forms");
    let cases=["normal","signed_zero","large","tiny_weight","mixed_scales","cancel"];
    let mut results=Vec::new();
    for rows in [1usize,2,3,6] {
        let mut b=HcBuffers::new(&cuda,rows)?;
        stream.join_default()?;
        let mut checks=Vec::new();
        for (index,case) in cases.iter().enumerate() {
            let mut shared=Fixture(0x5eed_0000+index as u64*977+rows as u64);
            let mut own=Fixture(0xa11_0000+index as u64*131+rows as u64*7+rank as u64*1_000_003);
            let scale=match *case {"large"=>1.0e10f32,_=>1.0};
            let mut p:Vec<f32>=(0..rows*5120).map(|_|own.normal(scale)).collect();
            let mut x:Vec<f32>=(0..rows*20480).map(|_|shared.normal(scale)).collect();
            let mut w:Vec<f32>=(0..5120).map(|_|shared.normal(1.0)).collect();
            match *case {
                "signed_zero"=>{for (i,v) in x.iter_mut().enumerate(){*v=if i%2==1{-0.0}else{0.0};}
                                for (i,v) in p.iter_mut().enumerate(){*v=if i%2==1{-0.0}else{0.0};}}
                "tiny_weight"=>{for v in w.iter_mut(){*v*=1.0e-30;}}
                "mixed_scales"=>{for v in w.iter_mut(){*v=2f32.powi((shared.next()%200) as i32-100);}}
                "cancel"=>{let mut same=Fixture(0xc0ffee+rows as u64);let sign=if rank%2==0{1.0}else{-1.0};
                           for v in p.iter_mut(){*v=sign*same.normal(1.0);}}
                _=>{}
            }
            b.partial.upload(&p.iter().map(|&v|bf16::from_f32(v)).collect::<Vec<_>>())?;
            b.residual.upload(&x.iter().map(|&v|bf16::from_f32(v)).collect::<Vec<_>>())?;
            b.weight.upload(&w.iter().map(|&v|bf16::from_f32(v)).collect::<Vec<_>>())?;
            b.post.upload(&(0..rows*4).map(|_|shared.normal(1.0)).collect::<Vec<_>>())?;
            b.comb.upload(&(0..rows*16).map(|_|shared.normal(0.5)).collect::<Vec<_>>())?;
            b.pre.upload(&(0..rows*4).map(|_|shared.normal(1.0)).collect::<Vec<_>>())?;
            for (poll,blocks) in [false,true].into_iter().flat_map(|p|[4usize,8,16,32].map(move|b|(p,b))) {
                comm.set_fused_poll_for_probe(poll,&stream)?;
                comm.set_fused_blocks_for_probe(blocks,&stream)?;
                b.clear()?;stream.join_default()?;
                hc_reference(&comm,&post_op,&quant,&mut b,rows,&stream)?;
                hc_fused(&comm,&variant,&mut b,rows,&stream)?;
                stream.synchronize()?;comm.health()?;
                let mut record=hc_compare(&b,case,rows)?;
                record["blocks"]=json!(blocks);record["single_poller"]=json!(poll);
                if blocks==16 && !poll { checks.push(record); }
            }
        }
        // Captured: eight boundaries a graph, launched alternately, compared after each launch pair.
        comm.set_fused_blocks_for_probe(16,&stream)?;
        let g_reference=stream.capture(||{for _ in 0..8 {hc_reference(&comm,&post_op,&quant,&mut b,rows,&stream)?;}Ok(())})?;
        let mut graphs=Vec::new();
        for (poll,blocks) in [false,true].into_iter().flat_map(|p|[4usize,8,16,32].map(move|b|(p,b))) {
            comm.set_fused_poll_for_probe(poll,&stream)?;
            comm.set_fused_blocks_for_probe(blocks,&stream)?;
            graphs.push(((poll,blocks),stream.capture(||{for _ in 0..8 {hc_fused(&comm,&variant,&mut b,rows,&stream)?;}Ok(())})?));
        }
        comm.set_fused_poll_for_probe(false,&stream)?;
        for (_,g) in &graphs {
            for _ in 0..2 {
                unsafe{g_reference.launch(&stream)?;g.launch(&stream)?;}
                stream.synchronize()?;comm.health()?;hc_compare(&b,"captured",rows)?;
            }
        }
        let mut timing=Vec::new();
        for ((poll,blocks),g) in &graphs {
            let mut pairs=Vec::new();
            for repeat in 0..6 {
                let mut times=[0f32;2];
                for v in [repeat%2,1-repeat%2] {
                    let graph=if v==0{&g_reference}else{g};
                    for _ in 0..4 {unsafe{graph.launch(&stream)?;}}stream.synchronize()?;
                    let start=cuda.event()?;let end=cuda.event()?;
                    start.record(&stream)?;
                    for _ in 0..32 {unsafe{graph.launch(&stream)?;}}
                    end.record(&stream)?;times[v]=end.elapsed_since(&start)?/256.;
                    comm.health()?;
                }
                pairs.push(json!({"repeat":repeat,"milliseconds_reference_fused":times}));
            }
            stream.synchronize()?;hc_compare(&b,"timed",rows)?;
            timing.push(json!({"blocks":blocks,"single_poller":poll,"pairs":pairs}));
        }
        results.push(json!({"rows":rows,"checks":checks,"timing":timing}));
    }
    stream.synchronize()?;comm.health()?;
    Ok(json!({"rank":rank,"results":results,"all_identical":true,"serving_policy_changed":false,
        "scope":"Fused one-shot sum + mHC post-mix + collapse (every block polling, and one polling block), then the value-reading layout8 norm-quant (PDL), versus the exported sum, the PDL pair's post-mix and norm-quant: six host fixtures, rows 1/2/3/6, fused grids 4-32, eager and captured; six balanced timing pairs of eight-boundary graphs per grid, 256 boundaries an interval. Component screen only."}))
}
