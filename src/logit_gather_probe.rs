//! Isolated raw FP32 logit transport screen. No model arithmetic or serving policy.
use crate::{collective::Nccl, cuda::Cuda, roce::Roce};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{net::SocketAddr, path::Path};

fn pattern(index: usize, rank: usize, phase: usize) -> u32 {
    // Include NaNs, signed zeros, infinities and subnormals as opaque bits.
    const SPECIAL: [u32; 8] = [0, 0x80000000, 0x7fc01234, 0xff801234,
        0x7f800000, 0xff800000, 1, 0x807fffff];
    if index % 17 == 0 { SPECIAL[(index / 17 + rank + phase) % SPECIAL.len()] }
    else { (index as u32).wrapping_mul(0x9e3779b9)
        .wrapping_add((rank as u32 + 1).wrapping_mul(0x6d2b79f5))
        .wrapping_add((phase as u32).wrapping_mul(0x85ebca6b)) }
}

/// Component screen for selecting the final live row from an unchanged head.
pub fn run_last(rank:usize,endpoint:&str,nonce:&[u8],root:&Path)->Result<Value> {
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let mut comm=Roce::local(&cuda,&root.join("roce-v0"),rank)?;
    comm.attach_gather(&root.join("roce-gather-v1"))?;
    let mut address:SocketAddr=endpoint.parse()?;
    let all=crate::rendezvous::exchange_blob(rank,address,nonce,&comm.blob()?)?;
    comm.connect(&all)?;
    address.set_port(address.port().checked_add(1).context("last gather ready port")?);
    ensure!(crate::rendezvous::exchange_blob(rank,address,nonce,b"R")?==b"RRRR",
        "last gather all peers connected");
    let mut sum_input=cuda.allocate::<u16>(8)?;
    sum_input.upload(&vec![half::bf16::from_f32((rank+1)as f32).to_bits();8])?;
    let mut sum_output=cuda.allocate::<u16>(8)?;
    let mut records=Vec::new();
    for rows in [6usize,16,32] {
        const COLS:usize=32320;
        const GUARD:usize=16;
        const CANARY:u32=0x7fc0abcd;
        let mut input=cuda.allocate::<f32>(rows*COLS)?;
        let full=cuda.allocate::<f32>(rows*COLS*4)?;
        let mut guarded=cuda.allocate::<f32>(rows*COLS*4+GUARD*2)?;
        guarded.upload(&vec![f32::from_bits(CANARY);guarded.len()])?;
        // This view lives inside its guarded owner for the whole graph bank.
        let mut last=unsafe{guarded.view(GUARD,rows*COLS*4)}?;
        input.zero()?;stream.join_default()?;
        let mut graphs=Vec::new();
        for live in [1,rows/2+1,rows] {
            let prefix=unsafe{input.view(0,live*COLS)}?;
            let mut full_prefix=unsafe{full.view(0,live*COLS*4)}?;
            unsafe {
                comm.gather_f32(&prefix,&mut full_prefix,COLS,&stream)?;
                comm.gather_f32_last(&input,&mut last,COLS,live,&stream)?;
            }
            stream.synchronize()?;
            let baseline=stream.capture(|| {
                for _ in 0..8 {unsafe{comm.gather_f32(&prefix,&mut full_prefix,COLS,&stream)?;}}
                Ok(())
            })?;
            let selected=stream.capture(|| {
                for _ in 0..8 {unsafe{comm.gather_f32_last(&input,&mut last,COLS,live,&stream)?;}}
                Ok(())
            })?;
            graphs.push((live,[baseline,selected]));
        }
        for phase in 0..2 {
            input.upload(&(0..input.len()).map(|i|f32::from_bits(pattern(i,rank,phase))).collect::<Vec<_>>())?;
            stream.join_default()?;
            for index in if phase==0{[0,1,2]}else{[2,1,0]} {
                let(live,pair)=&graphs[index];
                for g in pair {unsafe{g.launch(&stream)?;}}
                unsafe {comm.sum_bf16(&sum_input,&mut sum_output,&stream)?;}
                stream.synchronize()?;comm.health()?;
                ensure!(sum_output.download()?.iter().all(|&x|x==half::bf16::from_f32(10.).to_bits()),
                    "last gather mixed collective order");
                let actual=guarded.download()?;let baseline=full.download()?;
                ensure!(actual[..GUARD].iter().chain(&actual[actual.len()-GUARD..]).all(|x|x.to_bits()==CANARY),
                    "last gather output canary");
                for row in 0..rows {for source in 0..4 {for col in 0..COLS {
                    let i=(row*4+source)*COLS+col;
                    let expected=if row+1==*live{pattern(row*COLS+col,source,phase)}else{0};
                    ensure!(actual[GUARD+i].to_bits()==expected,"last gather bit/zero contract");
                    if row+1==*live {ensure!(baseline[i].to_bits()==expected,"last/full row mismatch");}
                }}}
                let mut timings=Vec::new();
                for repeat in 0..2 {
                    let order=if repeat==0{[0,1]}else{[1,0]};let mut milliseconds=[0f32;2];
                    for variant in order {
                        unsafe{comm.sum_bf16(&sum_input,&mut sum_output,&stream)?;}
                        for _ in 0..2 {unsafe{pair[variant].launch(&stream)?;}}
                        stream.synchronize()?;
                        let start=cuda.event()?;let end=cuda.event()?;start.record(&stream)?;
                        for _ in 0..8 {unsafe{pair[variant].launch(&stream)?;}}
                        end.record(&stream)?;milliseconds[variant]=end.elapsed_since(&start)?/64.;
                        comm.health()?;
                    }
                    timings.push(json!({"order":order,"milliseconds":milliseconds}));
                }
                records.push(json!({"rows":rows,"live":live,"phase":phase,"all_exact":true,
                    "unused_rows_zero":true,"guards_intact":true,"timings":timings}));
            }
        }
        eprintln!("rank{rank} last-row gather M{rows} exact");
    }
    Ok(json!({"rank":rank,"all_exact":true,"records":records,
        "scope":"Existing raw FP32 head rows; same transport and precision. One-row gather plus full output zeroing versus live-prefix gather; changing live counts and source bytes, guarded output and mixed collective order. Component only."}))
}

/// Keep the identical primitive and protocol, varying only its launch grid.
pub fn run_grid(rank:usize,endpoint:&str,nonce:&[u8],root:&Path)->Result<Value>{
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let mut comm=Roce::local(&cuda,&root.join("roce-v0"),rank)?;
    comm.attach_gather(&root.join("roce-gather-v1"))?;
    let mut endpoint:SocketAddr=endpoint.parse()?;
    let all=crate::rendezvous::exchange_blob(rank,endpoint,nonce,&comm.blob()?)?;
    let connect_delay_ms=match std::env::var("CANADA_SCRATCH_LOGIT_GATHER_CONNECT_DELAY_MS") {
        Ok(v)=>v.parse::<u64>()?,Err(std::env::VarError::NotPresent)=>0,Err(e)=>return Err(e.into()),
    };
    ensure!(connect_delay_ms<=250,"finite gather connect delay bound");
    if connect_delay_ms>0 {std::thread::sleep(std::time::Duration::from_millis(connect_delay_ms));}
    comm.connect(&all)?;
    // Endpoint exchange publishes INIT queues, not ready-to-send queues. The
    // full model has a subsequent NCCL rendezvous; this standalone probe must
    // also wait for every peer's local connect before any native GPU traffic.
    endpoint.set_port(endpoint.port().checked_add(1).ok_or_else(||anyhow::anyhow!("gather ready port overflow"))?);
    ensure!(crate::rendezvous::exchange_blob(rank,endpoint,nonce,b"R")?==b"RRRR",
        "gather peer readiness agreement");
    let mut sum_input=cuda.allocate::<u16>(8)?;
    sum_input.upload(&vec![half::bf16::from_f32((rank+1)as f32).to_bits();8])?;
    let mut sum_output=cuda.allocate::<u16>(8)?;
    let mut results=Vec::new();
    let wide_only=std::env::var("CANADA_SCRATCH_LOGIT_GATHER_GRID32_SCREEN").as_deref()==Ok("1");
    for &rows in if wide_only{&[32usize][..]}else{&[1usize,6,16][..]}{
        let columns=32320usize;
        let mut input=cuda.allocate::<f32>(rows*columns)?;
        let mut output=cuda.allocate::<f32>(rows*columns*4)?;
        input.zero()?;stream.join_default()?;
        let mut graphs=Vec::new();let mut mixed=Vec::new();
        for blocks in [8,16,32]{
            comm.set_gather_grid_for_probe(blocks,&stream)?;
            unsafe{comm.gather_f32(&input,&mut output,columns,&stream)?;}
            stream.synchronize()?;
            graphs.push(stream.capture(||{for _ in 0..8{
                unsafe{comm.gather_f32(&input,&mut output,columns,&stream)?;}
            }Ok(())})?);
            mixed.push(stream.capture(||unsafe{
                comm.sum_bf16(&sum_input,&mut sum_output,&stream)?;
                comm.gather_f32(&input,&mut output,columns,&stream)?;
                comm.sum_bf16(&sum_input,&mut sum_output,&stream)
            })?);
        }
        let mut cases=Vec::new();
        for phase in 0..2{
            input.upload(&(0..input.len()).map(|i|f32::from_bits(pattern(i,rank,phase)))
                .collect::<Vec<_>>())?;stream.join_default()?;
            let mut hashes=Vec::new();
            for variant in [0,1,2,1,0]{
                unsafe{graphs[variant].launch(&stream)?;mixed[variant].launch(&stream)?;}
                stream.synchronize()?;comm.health()?;
                ensure!(sum_output.download()?.iter().all(|&v|v==half::bf16::from_f32(10.).to_bits()),
                    "gather grid mixed sum ordering");
                let actual=output.download()?;
                for row in 0..rows{for source in 0..4{for col in 0..columns{
                    ensure!(actual[(row*4+source)*columns+col].to_bits()==pattern(row*columns+col,source,phase),
                        "gather grid bytes differ: {rows}/{phase}/{variant}/{row}/{source}/{col}");
                }}}
                let mut hash=Sha256::new();for value in actual{hash.update(value.to_bits().to_le_bytes());}
                hashes.push(format!("{:x}",hash.finalize()));
            }
            ensure!(hashes.iter().all(|h|h==&hashes[0]),"gather grid hashes");
            let mut pairs=Vec::new();
            for repeat in 0..2{
                let mut milliseconds=[0f32;3];
                let order=if repeat==0{[0,1,2]}else{[2,1,0]};
                for variant in order{
                    unsafe{comm.sum_bf16(&sum_input,&mut sum_output,&stream)?;}
                    for _ in 0..4{unsafe{graphs[variant].launch(&stream)?;}}
                    stream.synchronize()?;
                    let start=cuda.event()?;let end=cuda.event()?;start.record(&stream)?;
                    for _ in 0..8{unsafe{graphs[variant].launch(&stream)?;}}
                    end.record(&stream)?;milliseconds[variant]=end.elapsed_since(&start)?/64.;comm.health()?;
                }
                pairs.push(json!({"repeat":repeat,"order":order,"milliseconds":milliseconds}));
            }
            cases.push(json!({"phase":phase,"bytes_exact":true,"mixed_collective_ordering_exact":true,
                "sha256":hashes[0],"pairs":pairs}));
        }
        results.push(json!({"rows":rows,"local_columns":columns,"input_bytes":input.bytes(),"cases":cases}));
        eprintln!("rank{rank}: gather grid M{rows} exact");
    }
    stream.synchronize()?;comm.health()?;
    Ok(json!({"rank":rank,"all_exact":true,"results":results,"variants":[8,16,32],
        "all_peers_connected_before_gpu":true,"connect_delay_ms":connect_delay_ms,
        "scope":"Same raw FP32 gather primitive, protocol, bytes and ordered stream. Only8/16/32-block launch geometry differs. Two patterns, two opposite orders; counter classes switched back and forth with mixed native reductions. Component only."}))
}

pub fn run(rank: usize, endpoint: &str, nonce: &[u8], root: &Path,
    core: &Path) -> Result<Value> {
    let cuda = Cuda::open(0)?; cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let module = cuda.module(&std::fs::read(core)?)?;
    let transpose = module.function("ca_transpose_rank_columns")?;
    let mut native = Roce::local(&cuda, &root.join("roce-v0"), rank)?;
    native.attach_gather(&root.join("roce-gather-v1"))?;
    let mut address: SocketAddr = endpoint.parse()?;
    let all = crate::rendezvous::exchange_blob(rank, address, nonce, &native.blob()?)?;
    native.connect(&all)?;
    address.set_port(address.port().checked_add(2)
        .ok_or_else(|| anyhow::anyhow!("logit gather rendezvous port"))?);
    let api = Nccl::load_qualified_prefill()?;
    let id = crate::rendezvous::join(&api, rank, address, nonce)?;
    let nccl = api.join(&cuda, id, rank, 4)?;
    let mut sum_input = cuda.allocate::<u16>(8)?;
    sum_input.upload(&vec![half::bf16::from_f32((rank + 1) as f32).to_bits(); 8])?;
    let mut sum_output = cuda.allocate::<u16>(8)?;
    let mut results = Vec::new();
    for rows in [1usize, 6, 16] {
        let columns = 32320usize;
        let mut input = cuda.allocate::<f32>(rows * columns)?;
        let mut output_native = cuda.allocate::<f32>(rows * columns * 4)?;
        let mut output_nccl = cuda.allocate::<f32>(rows * columns * 4)?;
        let mut scratch = cuda.allocate::<f32>(rows * columns * 4)?;
        input.zero()?; stream.join_default()?;
        let mut enqueue = |variant: usize| -> Result<()> {
            if variant == 0 {
                unsafe { native.gather_f32(&input, &mut output_native, columns, &stream) }
            } else if rows == 1 {
                // One row needs no rank-major -> row-major shuffle.
                unsafe { nccl.gather_bytes_prefix(&input, &mut output_nccl, &stream) }
            } else {
                unsafe { nccl.gather_bytes_prefix(&input, &mut scratch, &stream)?; }
                let (mut src, mut dst, mut r, mut c) =
                    (scratch.ptr(), output_nccl.ptr(), rows as i32, (columns / 4) as i32);
                let mut args = [(&mut src as *mut u64).cast(), (&mut dst as *mut u64).cast(),
                    (&mut r as *mut i32).cast(), (&mut c as *mut i32).cast()];
                unsafe { transpose.launch([(output_nccl.len() / 4).div_ceil(256) as u32, 1, 1],
                    [256, 1, 1], 0, &stream, &mut args) }
            }
        };
        for variant in [0, 1] { enqueue(variant)?; stream.synchronize()?; }
        let g0 = stream.capture(|| { for _ in 0..8 { enqueue(0)?; } Ok(()) })?;
        let g1 = stream.capture(|| { for _ in 0..8 { enqueue(1)?; } Ok(()) })?;
        let mixed = stream.capture(|| {
            unsafe { native.sum_bf16(&sum_input, &mut sum_output, &stream)?; }
            enqueue(1)?;
            unsafe { native.sum_bf16(&sum_input, &mut sum_output, &stream) }
        })?;
        drop(enqueue);
        let mut cases = Vec::new();
        for phase in 0..2 {
            input.upload(&(0..input.len()).map(|i| f32::from_bits(pattern(i, rank, phase)))
                .collect::<Vec<_>>())?;
            stream.join_default()?;
            unsafe { g0.launch(&stream)?; g1.launch(&stream)?; mixed.launch(&stream)?; }
            stream.synchronize()?; native.health()?; nccl.check_async()?;
            ensure!(sum_output.download()?.iter().all(|&v|
                v == half::bf16::from_f32(10.).to_bits()), "mixed collective ordering");
            let mut hashes = Vec::new();
            for output in [&output_native, &output_nccl] {
                let actual = output.download()?;
                for row in 0..rows { for source in 0..4 { for col in 0..columns {
                    ensure!(actual[(row * 4 + source) * columns + col].to_bits()
                        == pattern(row * columns + col, source, phase),
                        "logit gather bytes differ: {rows}/{phase}/{row}/{source}/{col}");
                }}}
                let mut hash = Sha256::new();
                for value in actual { hash.update(value.to_bits().to_le_bytes()); }
                hashes.push(format!("{:x}", hash.finalize()));
            }
            ensure!(hashes[0] == hashes[1], "transport output hash");
            let mut pairs = Vec::new();
            for repeat in 0..2 {
                let mut milliseconds = [0f32; 2];
                for variant in [repeat % 2, 1 - repeat % 2] {
                    // Synchronize peers outside the interval; one ordered stream.
                    unsafe { native.sum_bf16(&sum_input, &mut sum_output, &stream)?; }
                    let graph = if variant == 0 { &g0 } else { &g1 };
                    for _ in 0..4 { unsafe { graph.launch(&stream)?; } }
                    stream.synchronize()?;
                    let start = cuda.event()?; let end = cuda.event()?;
                    start.record(&stream)?;
                    for _ in 0..8 { unsafe { graph.launch(&stream)?; } }
                    end.record(&stream)?;
                    milliseconds[variant] = end.elapsed_since(&start)? / 64.;
                    native.health()?; nccl.check_async()?;
                }
                pairs.push(json!({"repeat":repeat,"first_variant":repeat%2,
                    "milliseconds":milliseconds}));
            }
            cases.push(json!({"phase":phase,"bytes_exact":true,
                "mixed_collective_ordering_exact":true,"sha256":hashes[0],"pairs":pairs}));
        }
        results.push(json!({"rows":rows,"local_columns":columns,"input_bytes":input.bytes(),
            "extra_shuffle":rows>1,"cases":cases}));
        eprintln!("rank {rank}: logit gather M{rows} exact");
    }
    stream.synchronize()?; native.health()?; nccl.check_async()?;
    Ok(json!({"rank":rank,"all_exact":true,"results":results,
        "variants":["RoCEnante raw-copy column gather","NCCL byte gather plus required rank/row shuffle"],
        "scope":"Opaque FP32 payloads, M1/M6/M16, two input patterns and two opposite-order pairs per pattern. Includes shuffle cost and mixed native-sum/NCCL-gather graph ordering. No model or serving policy change."}))
}
