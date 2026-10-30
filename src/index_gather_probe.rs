//! Raw-byte index-key gather gate; synthetic bytes, no CPU model math.
use crate::cuda::{Buffer, Cuda, Module, Stream};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{ffi::c_void, path::Path};
fn launch(
    module: &Module,
    cache: &Buffer<u32>,
    slots: &Buffer<i32>,
    payload: &Buffer<u32>,
    scales: &Buffer<u32>,
    ends: Option<&Buffer<i32>>,
    stream: &Stream,
) -> Result<()> {
    let (mut c, mut s, mut p, mut e) = (cache.ptr(), slots.ptr(), payload.ptr(), scales.ptr());
    let mut rows = slots.len() as i32;
    let mut args: Vec<*mut c_void> = vec![
        (&mut c as *mut u64).cast(),
        (&mut s as *mut u64).cast(),
        (&mut p as *mut u64).cast(),
        (&mut e as *mut u64).cast(),
        (&mut rows as *mut i32).cast(),
    ];
    let mut bound = ends.map_or(0, |e| e.ptr());
    let mut queries = ends.map_or(0, |e| e.len()) as i32;
    if ends.is_some() {
        args.extend([
            (&mut bound as *mut u64).cast(),
            (&mut queries as *mut i32).cast(),
        ]);
    }
    unsafe {
        module
            .function(if ends.is_some() {
                "ca_gather_index_k_bounded"
            } else {
                "ca_gather_index_k"
            })?
            .launch(
                [
                    if ends.is_some() {
                        192
                    } else {
                        slots.len().div_ceil(4) as u32
                    },
                    1,
                    1,
                ],
                [128, 1, 1],
                0,
                stream,
                &mut args,
            )
    }
}
pub fn run(core: &Path) -> Result<Value> {
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let module = cuda.module(&std::fs::read(core)?)?;
    let mut reports = Vec::new();
    for rows in [4096usize, 32768, 1048576] {
        let pages = (rows + 64).div_ceil(64);
        let mut cache = cuda.allocate::<u32>(pages * 4352 / 4)?;
        let mut slots = cuda.allocate::<i32>(rows)?;
        slots.upload(
            &(0..rows)
                .map(|i| ((i * 997) % rows + 64) as i32)
                .collect::<Vec<_>>(),
        )?;
        let reference = cuda.allocate::<u32>(rows * 16)?;
        let reference_scales = cuda.allocate::<u32>(rows)?;
        let mut bounded = cuda.allocate::<u32>(rows * 16)?;
        let mut bounded_scales = cuda.allocate::<u32>(rows)?;
        for queries in [6usize, 128, 4096] {
            let mut ends = cuda.allocate::<i32>(queries)?;
            ends.zero()?;
            stream.join_default()?;
            let full = stream.capture(|| {
                launch(
                    &module,
                    &cache,
                    &slots,
                    &reference,
                    &reference_scales,
                    None,
                    &stream,
                )
            })?;
            let short = stream.capture(|| {
                launch(
                    &module,
                    &cache,
                    &slots,
                    &bounded,
                    &bounded_scales,
                    Some(&ends),
                    &stream,
                )
            })?;
            for (phase, live) in [0usize, 1, 127, 128, 129, 458, rows.min(8000), rows]
                .into_iter()
                .enumerate()
            {
                let bytes: Vec<u32> = (0..cache.len())
                    .map(|i| {
                        (i as u32)
                            .wrapping_mul(0x9e3779b9)
                            .rotate_left(phase as u32)
                            ^ 0xa5b3c7d9
                    })
                    .collect();
                cache.upload(&bytes)?;
                let mut limits = vec![0i32; queries];
                limits[queries / 2] = live as i32;
                limits[queries - 1] = live.saturating_sub(1) as i32;
                ends.upload(&limits)?;
                bounded.upload(&vec![0xdeadbeef; rows * 16])?;
                bounded_scales.upload(&vec![0xdeadbeef; rows])?;
                stream.join_default()?;
                unsafe {
                    full.launch(&stream)?;
                    short.launch(&stream)?;
                }
                stream.synchronize()?;
                let n = live.div_ceil(128).saturating_mul(128).min(rows);
                let a = reference.download()?;
                let b = bounded.download()?;
                let sa = reference_scales.download()?;
                let sb = bounded_scales.download()?;
                ensure!(
                    a[..n * 16] == b[..n * 16] && sa[..n] == sb[..n],
                    "bounded gather changed reachable raw bytes"
                );
                ensure!(
                    b[n * 16..].iter().all(|&x| x == 0xdeadbeef)
                        && sb[n..].iter().all(|&x| x == 0xdeadbeef),
                    "bounded gather touched unreachable output"
                );
                let mut times = [Vec::new(), Vec::new()];
                if queries == 6 && [458, rows.min(8000), rows].contains(&live) {
                    let start = cuda.event()?;
                    let stop = cuda.event()?;
                    for pair in 0..100 {
                        for which in [pair % 2, 1 - pair % 2] {
                            start.record(&stream)?;
                            unsafe {
                                if which == 0 {
                                    full.launch(&stream)?
                                } else {
                                    short.launch(&stream)?
                                }
                            }
                            stop.record(&stream)?;
                            times[which].push(stop.elapsed_since(&start)?);
                        }
                    }
                }
                reports.push(
                    json!({"capacity":rows,"queries":queries,"live_keys":live,"copied_keys":n,
                    "phase":phase,"reachable_bytes_equal":true,"unreachable_output_untouched":true,
                    "native_ms":times[0],"bounded_ms":times[1]}),
                );
            }
        }
    }
    Ok(
        json!({"all_exact":true,"cases":reports,"full_engine_acceptance":false,
        "scope":"Synthetic raw-byte paged key gathers at4K/32K/1M capacities and6/128/4096 query rows. Captured metadata bounds change across replays and arbitrary payload patterns. Every reachable payload/scale bit matches the full-gather control; unreachable output remains untouched. Alternating100 paired GPU timings on six-query cases. This does not qualify full-model attention or long-context performance."}),
    )
}
