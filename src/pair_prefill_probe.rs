//! GPU prefill pair qualification with non-idempotent replay state restored.
use crate::{
    cuda::{Buffer, Cuda, DeviceElement},
    layer_probe::{Tap, compare, fixture},
    pair_prefill::{Pool, Ring, Schedule},
};
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
#[derive(Deserialize)]
struct Config {
    fixture: PathBuf,
    primitive: PathBuf,
}
#[derive(Deserialize)]
struct Cases {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    rows: usize,
    ring: usize,
    request_slots: usize,
    raw_capacity: usize,
    frequency_rows: usize,
    seed: Tap,
    phases: Vec<Phase>,
}
#[derive(Deserialize)]
struct Phase {
    name: String,
    positions: Vec<i32>,
    requests: Vec<i64>,
    locations: Vec<i64>,
    source: Tap,
    expected: Vec<Tap>,
}
fn bytes<T: DeviceElement>(b: &Buffer<T>) -> Result<Vec<u8>> {
    let x = b.download()?;
    ensure!(cfg!(target_endian = "little"), "diagnostic byte order");
    Ok(unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), b.bytes()) }.to_vec())
}
fn upload(b: &mut Buffer<f32>, data: &[u8]) -> Result<()> {
    ensure!(data.len() == b.bytes(), "pair input bytes");
    b.upload(
        &data
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>(),
    )
}
pub fn run(path: &Path) -> Result<Value> {
    let p: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    let cases: Cases = serde_json::from_slice(&std::fs::read(p.fixture.join("index.json"))?)?;
    ensure!(cases.cases.len() == 10, "pair prefill coverage");
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let mut reports = Vec::new();
    for c in &cases.cases {
        let root = p.fixture.join(&c.name);
        let mut state = Ring::new(&cuda, c.request_slots, c.ring)?;
        upload(state.values_mut(), &fixture(&root, &c.seed)?)?;
        let pool = Pool::new(&cuda, &p.primitive, c.rows, c.ring)?;
        let mut input = cuda.allocate::<f32>(c.rows * 1024)?;
        let initial = cuda.allocate::<f32>(state.values().len())?;
        let mut phases = Vec::new();
        for phase in &c.phases {
            let directory = root.join(&phase.name);
            upload(&mut input, &fixture(&directory, &phase.source)?)?;
            let meta = Schedule::new(
                &cuda,
                &phase.positions,
                &phase.requests,
                &phase.locations,
                c.request_slots,
                c.ring,
                c.raw_capacity,
                c.frequency_rows,
            )?;
            unsafe {
                cuda.copy_device(
                    initial.ptr(),
                    state.values().ptr(),
                    initial.bytes(),
                    &stream,
                )
            }?;
            let graph = stream.capture(|| pool.run(&input, &mut state, &meta, &stream))?;
            unsafe { graph.launch(&stream) }?;
            stream.synchronize()?;
            let got = BTreeMap::from([
                ("pooled".into(), bytes(pool.output())?),
                ("ring".into(), bytes(state.values())?),
            ]);
            let taps = compare(&directory, &phase.expected, &got)?;
            let a = cuda.event()?;
            let b = cuda.event()?;
            let mut times = Vec::new();
            for _ in 0..100 {
                // Long chunks overwrite the carried slot; replaying without restoration
                // would test a different history. Copy is deliberately outside timing.
                unsafe {
                    cuda.copy_device(
                        state.values().ptr(),
                        initial.ptr(),
                        initial.bytes(),
                        &stream,
                    )
                }?;
                a.record(&stream)?;
                unsafe { graph.launch(&stream) }?;
                b.record(&stream)?;
                times.push(b.elapsed_since(&a)?);
            }
            times.sort_by(f32::total_cmp);
            stream.synchronize()?;
            let repeated = BTreeMap::from([
                ("pooled".into(), bytes(pool.output())?),
                ("ring".into(), bytes(state.values())?),
            ]);
            let replay_taps = compare(&directory, &phase.expected, &repeated)?;
            phases.push(json!({"name":phase.name,"taps":taps,"replay_taps":replay_taps,"gpu_ms_excludes_fixture_restore":{"p50":times[49],"p95":times[94],"p99":times[98],"max":times[99]}}));
        }
        let exact = phases.iter().all(|p| {
            ["taps", "replay_taps"].iter().all(|k| {
                p[*k]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|t| t["bitwise_equal"] == true)
            })
        });
        eprintln!("{}", json!({"case":c.name,"all_bitwise_equal":exact}));
        reports.push(json!({"case":c.name,"all_bitwise_equal":exact,"phases":phases}));
    }
    Ok(
        json!({"all_bitwise_equal":reports.iter().all(|r|r["all_bitwise_equal"]==true),"cases":reports,"scope":"GPU prefill pooling and persistent ring tails only; not full prefill or model parity"}),
    )
}
