//! Payload-copy/ordering qualification for the native communication primitive.
//! Integer bit patterns are raw transport fixtures, not CPU model computation.
use crate::{cuda::Cuda, roce::Roce};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::path::Path;
fn pattern(i: usize, rank: usize, phase: usize) -> u16 {
    (i as u16)
        .wrapping_mul(13)
        .wrapping_add((rank as u16) * 8191)
        .wrapping_add((phase as u16) * 317)
}
pub fn run(rank: usize, endpoint: &str, nonce: &[u8], root: &Path, column_core: Option<&Path>) -> Result<Value> {
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let mut comm = Roce::local(&cuda, &root.join("roce-v0"), rank)?;
    comm.attach_gather(&root.join("roce-gather-v1"))?;
    let all = crate::rendezvous::exchange_blob(rank, endpoint.parse()?, nonce, &comm.blob()?)?;
    comm.connect(&all)?;
    if let Some(core) = column_core {
        let api=crate::collective::Nccl::load_qualified_prefill()?;
        let mut address:std::net::SocketAddr=endpoint.parse()?;
        address.set_port(address.port().checked_add(2).ok_or_else(||anyhow::anyhow!("gather rendezvous port"))?);
        let id=crate::rendezvous::join(&api,rank,address,nonce)?;
        comm.attach_prefill_sum(api.join(&cuda,id,rank,4)?)?;
        comm.attach_prefill_column_gather(&std::fs::read(core)?, 4096)?;
    }
    let mut sum_input = cuda.allocate::<u16>(8)?;
    sum_input.upload(&vec![half::bf16::from_f32((rank + 1) as f32).to_bits(); 8])?;
    let mut sum_output = cuda.allocate::<u16>(8)?;
    let mut results = Vec::new();
    for (rows, columns) in [
        (1usize, 448usize),
        (6, 448),
        (161, 448),
        (460, 448),
        (1, 6400),
        (6, 6400),
        (161, 6400),
        (460, 6400),
        (512, 6400),
        (1024, 6400),
        (2048, 6400),
        (4096, 6400),
    ] {
        let mut input = cuda.allocate::<u16>(rows * columns)?;
        let mut output = cuda.allocate::<u16>(4 * rows * columns)?;
        input.zero()?;
        stream.join_default()?;
        comm.select_decode_reduction(rows < 512, &stream)?;
        // Initialize NCCL/vendor paths before capturing mixed communication.
        unsafe {
            comm.sum_bf16(&sum_input,&mut sum_output,&stream)?;
            comm.gather_bf16(&input,&mut output,columns,&stream)?;
        }
        stream.synchronize()?;
        // Each graph includes sum -> gather -> sum, sharing one epoch. Changing
        // shape on the next iteration exercises every power-of-two grid class.
        let graph = stream.capture(|| unsafe {
            comm.sum_bf16(&sum_input, &mut sum_output, &stream)?;
            comm.gather_bf16(&input, &mut output, columns, &stream)?;
            comm.sum_bf16(&sum_input, &mut sum_output, &stream)
        })?;
        for phase in 0..2 {
            input.upload(
                &(0..rows * columns)
                    .map(|i| pattern(i, rank, phase))
                    .collect::<Vec<_>>(),
            )?;
            stream.join_default()?;
            for _ in 0..5 {
                unsafe { graph.launch(&stream) }?;
            }
            stream.synchronize()?;
            comm.health()?;
            let actual = output.download()?;
            for row in 0..rows {
                for source in 0..4 {
                    for col in 0..columns {
                        ensure!(
                            actual[(row * 4 + source) * columns + col]
                                == pattern(row * columns + col, source, phase),
                            "gather payload changed: row={row}, source={source}, col={col}, phase={phase}"
                        );
                    }
                }
            }
            ensure!(
                sum_output
                    .download()?
                    .iter()
                    .all(|v| *v == half::bf16::from_f32(10.).to_bits()),
                "sum/gather shared epoch lost ordering"
            );
        }
        // Measure gather alone. Individual graph replay events retain latency
        // tails rather than dividing one aggregate by many launches.
        let gather = stream
            .capture(|| unsafe { comm.gather_bf16(&input, &mut output, columns, &stream) })?;
        for _ in 0..50 {
            unsafe { gather.launch(&stream) }?;
        }
        stream.synchronize()?;
        let events = (0..100)
            .map(|_| Ok((cuda.event()?, cuda.event()?)))
            .collect::<Result<Vec<_>>>()?;
        for (start, end) in &events {
            start.record(&stream)?;
            unsafe { gather.launch(&stream) }?;
            end.record(&stream)?;
        }
        stream.synchronize()?;
        comm.health()?;
        let mut ms = events
            .iter()
            .map(|(a, b)| b.elapsed_since(a))
            .collect::<Result<Vec<_>>>()?;
        ms.sort_by(f32::total_cmp);
        results.push(json!({"rows":rows,"local_columns":columns,"input_bytes":input.bytes(),
            "output_bytes":output.bytes(),"bitwise_equal":true,"payload_phases":2,
            "sum_gather_interleaving_exact":true,"latency_ms":{"p50":ms[49],"p95":ms[94],"p99":ms[98],"max":ms[99]},
            "samples":ms.len(),"samples_ms":ms}));
        eprintln!("rank {rank}: gather {rows}x{columns} exact");
    }
    stream.synchronize()?;
    comm.health()?;
    Ok(
        json!({"rank":rank,"all_bitwise_equal":true,"layout":"[row, rank, local_columns]",
        "source":if column_core.is_some(){"NCCL byte all-gather and native rank/row transpose for wide Engram; RoCEnante otherwise"}else{"pinned RoCEnante with attributed raw-copy gather adapter"},
        "scope":"communication primitive only; includes every u16 pattern, changed inputs, graph replay and interleaved sum",
        "results":results}),
    )
}
