//! Isolated raw FP32 logit transport screen. No model arithmetic or serving policy.
use crate::{collective::Nccl, cuda::Cuda, roce::Roce};
use anyhow::{Result, ensure};
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

/// Keep the identical primitive and protocol, varying only its launch grid.
pub fn run_grid(rank:usize,endpoint:&str,nonce:&[u8],root:&Path)->Result<Value>{
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let mut comm=Roce::local(&cuda,&root.join("roce-v0"),rank)?;
    comm.attach_gather(&root.join("roce-gather-v1"))?;
    let all=crate::rendezvous::exchange_blob(rank,endpoint.parse()?,nonce,&comm.blob()?)?;
    comm.connect(&all)?;
    let mut sum_input=cuda.allocate::<u16>(8)?;
    sum_input.upload(&vec![half::bf16::from_f32((rank+1)as f32).to_bits();8])?;
    let mut sum_output=cuda.allocate::<u16>(8)?;
    let mut results=Vec::new();
    for rows in [1usize,6,16]{
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
