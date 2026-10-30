//! GPU-only qualification of Rust-owned low-ratio compression state.
use crate::{
    checkpoint::Checkpoint,
    compression::{Input, Metadata, Source},
    config::ModelConfig,
    cuda::{Buffer, Cuda, Stream},
    layer_probe::{Tap, compare, fixture},
    layout::RankPlan,
    prelude::FrequencyTable,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    rc::Rc,
};
#[derive(Deserialize)]
struct Config {
    config: PathBuf,
    checkpoint: PathBuf,
    model: PathBuf,
    fixture: PathBuf,
    core: PathBuf,
    primitive: PathBuf,
}
#[derive(Deserialize)]
struct Cases {
    frequency_rows: usize,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    ratio: usize,
    layer: usize,
    rows: usize,
    page_tokens: usize,
    pages: usize,
    request_slots: usize,
    ring: usize,
    draft: usize,
    initial: Vec<Tap>,
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
fn upload_bf16(b: &mut Buffer<bf16>, data: &[u8]) -> Result<()> {
    ensure!(data.len() == b.bytes(), "BF16 fixture byte count");
    b.upload(
        &data
            .chunks_exact(2)
            .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())))
            .collect::<Vec<_>>(),
    )
}
fn upload_f32(b: &mut Buffer<f32>, data: &[u8]) -> Result<()> {
    ensure!(data.len() == b.bytes(), "FP32 fixture byte count");
    b.upload(
        &data
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>(),
    )
}
fn bytes<T: crate::cuda::DeviceElement>(b: &Buffer<T>) -> Result<Vec<u8>> {
    let x = b.download()?;
    ensure!(cfg!(target_endian = "little"), "diagnostic byte order");
    Ok(unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), b.bytes()) }.to_vec())
}
pub fn run(path: &Path) -> Result<Value> {
    let p: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    let cases: Cases = serde_json::from_slice(&std::fs::read(p.fixture.join("index.json"))?)?;
    ensure!(
        cases.frequency_rows == 1048576 && cases.cases.len() == 17,
        "compression reference coverage"
    );
    let cfg = ModelConfig::parse(&std::fs::read(p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let plan = RankPlan::build(&cp, &cfg, 0, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let weights = Rc::new(Weights::load_prefixes(
        &cuda,
        &p.model,
        &cp,
        &plan,
        &[
            "layers.2.attn.compressor.norm.",
            "layers.20.attn.compressor.norm.",
        ],
    )?);
    let verified = weights.qualify(&cp, &p.model, &plan)?;
    let stream = cuda.stream()?;
    let core = cuda.module(&std::fs::read(&p.core)?)?;
    let freq = Rc::new(FrequencyTable::new(
        &cuda,
        &core,
        &stream,
        cases.frequency_rows,
        true,
    )?);
    let mut reports = Vec::new();
    for c in &cases.cases {
        let root = p.fixture.join(&c.name);
        let mut source = Source::new(
            &cuda,
            weights.clone(),
            c.layer,
            c.rows,
            c.request_slots,
            c.ring,
            c.pages,
            c.page_tokens,
            freq.clone(),
            &p.primitive,
        )?;
        for tap in &c.initial {
            let data = fixture(&root.join("seed"), tap)?;
            match tap.name.as_str() {
                "cache" => source.cache_mut().upload(&data)?,
                "latent" => upload_bf16(source.latent_mut(), &data)?,
                "ring" => upload_f32(
                    source
                        .ring_mut()
                        .ok_or_else(|| anyhow::anyhow!("missing ring"))?,
                    &data,
                )?,
                _ => anyhow::bail!("unknown compression seed"),
            }
        }
        let first = c
            .phases
            .first()
            .ok_or_else(|| anyhow::anyhow!("empty compression phases"))?;
        let mut meta = Metadata::new(
            &cuda,
            c.ratio,
            source.cache().capacity(),
            c.request_slots,
            c.ring,
            c.draft,
            freq.rows(),
            &first.positions,
            &first.requests,
            &first.locations,
        )?;
        let mut x1 = cuda.allocate::<bf16>(c.rows * 512)?;
        let mut x2 = cuda.allocate::<f32>(c.rows * 1024)?;
        let mut graph = None;
        let mut phases = Vec::new();
        for phase in &c.phases {
            let directory = root.join(&phase.name);
            let input = fixture(&directory, &phase.source)?;
            meta.update(&phase.positions, &phase.requests, &phase.locations)?;
            if c.ratio == 1 {
                upload_bf16(&mut x1, &input)?
            } else {
                upload_f32(&mut x2, &input)?
            }
            let run = |s: &mut Source, st: &Stream| {
                if c.ratio == 1 {
                    s.run(Input::Ratio1(&x1), &meta, st)
                } else {
                    s.run(Input::Ratio2(&x2), &meta, st)
                }
            };
            if graph.is_none() {
                run(&mut source, &stream)?;
                stream.synchronize()?;
                graph = Some(stream.capture(|| run(&mut source, &stream))?)
            }
            unsafe { graph.as_ref().unwrap().launch(&stream) }?;
            stream.synchronize()?;
            let mut got = BTreeMap::from([
                ("cache".into(), source.cache().data().download()?),
                ("latent".into(), bytes(source.latent())?),
            ]);
            if let Some(r) = source.ring() {
                got.insert("ring".into(), bytes(r)?);
            }
            let taps = compare(&directory, &phase.expected, &got)?;
            // Each schedule is replay-idempotent: completing rows never alter
            // their predecessor ring slot; verify reads preceding input rows.
            let mut samples = Vec::new();
            let a = cuda.event()?;
            let b = cuda.event()?;
            for _ in 0..100 {
                a.record(&stream)?;
                unsafe { graph.as_ref().unwrap().launch(&stream) }?;
                b.record(&stream)?;
                samples.push(b.elapsed_since(&a)?);
            }
            samples.sort_by(f32::total_cmp);
            stream.synchronize()?;
            let mut repeated = BTreeMap::from([
                ("cache".into(), source.cache().data().download()?),
                ("latent".into(), bytes(source.latent())?),
            ]);
            if let Some(r) = source.ring() {
                repeated.insert("ring".into(), bytes(r)?);
            }
            let replay_taps = compare(&directory, &phase.expected, &repeated)?;
            phases.push(json!({"name":phase.name,"taps":taps,"replay_taps":replay_taps,"gpu_ms":{"p50":samples[49],"p95":samples[94],"p99":samples[98],"max":samples[99]}}));
        }
        let exact = phases.iter().all(|r| {
            ["taps", "replay_taps"].iter().all(|k| {
                r[*k]
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
        json!({"all_bitwise_equal":reports.iter().all(|r|r["all_bitwise_equal"]==true),"cases":reports,"weights":verified,"scope":"Persistent c1/c2 decode/verify state; full prefill pooling/session graph not yet qualified"}),
    )
}

/// Composite prefill state proof. Restore complete incoming state before every
/// graph replay: a long chunk overwrites its carried ring slot and is not
/// replay-idempotent by itself.
pub fn run_prefill(path: &Path) -> Result<Value> {
    let p: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    let cases: Cases = serde_json::from_slice(&std::fs::read(p.fixture.join("index.json"))?)?;
    ensure!(
        cases.frequency_rows == 1048576 && [4, 8].contains(&cases.cases.len()),
        "prefill compression coverage"
    );
    let cfg = ModelConfig::parse(&std::fs::read(p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let plan = RankPlan::build(&cp, &cfg, 0, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let weights = Rc::new(Weights::load_prefixes(
        &cuda,
        &p.model,
        &cp,
        &plan,
        &[
            "layers.2.attn.compressor.norm.",
            "layers.20.attn.compressor.norm.",
        ],
    )?);
    let verified = weights.qualify(&cp, &p.model, &plan)?;
    let stream = cuda.stream()?;
    let core_bytes = std::fs::read(&p.core)?;
    let module = cuda.module(&core_bytes)?;
    let freq = Rc::new(FrequencyTable::new(&cuda, &module, &stream, 1048576, true)?);
    let mut reports = Vec::new();
    for c in &cases.cases {
        ensure!(
            ((c.layer == 2 && c.ratio == 2) || (c.layer == 20 && c.ratio == 1)) && c.rows >= 128,
            "prefill ratio/rows"
        );
        let root = p.fixture.join(&c.name);
        let mut source = Source::new(
            &cuda,
            weights.clone(),
            c.layer,
            c.rows,
            c.request_slots,
            c.ring,
            c.pages,
            c.page_tokens,
            freq.clone(),
            &p.primitive,
        )?;
        for tap in &c.initial {
            let data = fixture(&root.join("seed"), tap)?;
            match tap.name.as_str() {
                "cache" => source.cache_mut().upload(&data)?,
                "latent" => upload_bf16(source.latent_mut(), &data)?,
                "ring" => upload_f32(
                    source
                        .ring_mut()
                        .ok_or_else(|| anyhow::anyhow!("seed requires ring"))?,
                    &data,
                )?,
                _ => anyhow::bail!("unknown prefill seed"),
            }
        }
        let mut input2 = cuda.allocate::<f32>(c.rows * 1024)?;
        let mut input1 = cuda.allocate::<bf16>(c.rows * 512)?;
        let ring_copy = source
            .ring()
            .map(|r| cuda.allocate::<u8>(r.bytes()))
            .transpose()?;
        let cache_copy = cuda.allocate::<u8>(source.cache().data().bytes())?;
        let latent_copy = cuda.allocate::<u8>(source.latent().bytes())?;
        let first = c
            .phases
            .first()
            .ok_or_else(|| anyhow::anyhow!("missing prefill phase"))?;
        let mut schedule = crate::compression::PairPrefill::new(
            &cuda,
            &core_bytes,
            p.primitive
                .parent()
                .ok_or_else(|| anyhow::anyhow!("kernel root"))?,
            &source,
            &first.positions,
            &first.requests,
            &first.locations,
        )?;
        let mut graph = None;
        let mut phases = Vec::new();
        for phase in &c.phases {
            let directory = root.join(&phase.name);
            let input = fixture(&directory, &phase.source)?;
            if c.ratio == 2 {
                upload_f32(&mut input2, &input)?;
            } else {
                upload_bf16(&mut input1, &input)?;
            }
            schedule.update(&phase.positions, &phase.requests, &phase.locations)?;
            let mut incoming = vec![
                (
                    cache_copy.ptr(),
                    source.cache().data().ptr(),
                    cache_copy.bytes(),
                ),
                (
                    latent_copy.ptr(),
                    source.latent().ptr(),
                    latent_copy.bytes(),
                ),
            ];
            if let Some(copy) = &ring_copy {
                incoming.push((copy.ptr(), source.ring().unwrap().ptr(), copy.bytes()));
            }
            for &(dst, src, size) in &incoming {
                unsafe { cuda.copy_device(dst, src, size, &stream) }?;
            }
            let restore = || -> Result<()> {
                for &(saved, dst, size) in &incoming {
                    unsafe { cuda.copy_device(dst, saved, size, &stream) }?;
                }
                Ok(())
            };
            let run = |s: &mut Source| -> Result<()> {
                if c.ratio == 2 {
                    s.run_pair_prefill(&input2, &schedule, &stream)
                } else {
                    s.run_single_prefill(&input1, &schedule, &stream)
                }
            };
            if graph.is_none() {
                run(&mut source)?;
                stream.synchronize()?;
                restore()?;
                graph = Some(stream.capture(|| run(&mut source))?);
            }
            let mut times = Vec::new();
            for _ in 0..100 {
                restore()?;
                let start = cuda.event()?;
                let end = cuda.event()?;
                start.record(&stream)?;
                unsafe { graph.as_ref().unwrap().launch(&stream) }?;
                end.record(&stream)?;
                times.push(end.elapsed_since(&start)?);
            }
            stream.synchronize()?;
            let mut got = BTreeMap::from([
                ("cache".into(), source.cache().data().download()?),
                ("latent".into(), bytes(source.latent())?),
            ]);
            if let Some(ring) = source.ring() {
                got.insert("ring".into(), bytes(ring)?);
            }
            let taps = compare(&directory, &phase.expected, &got)?;
            times.sort_by(f32::total_cmp);
            phases.push(json!({"phase":phase.name,"taps":taps,"samples":100,"incoming_state_restored_each_replay":true,
                "graph_reused_across_metadata_updates":true,
                "gpu_ms_excludes_restore":{"p50":times[49],"p95":times[94],"p99":times[98],"max":times[99]}}));
        }
        let exact = phases.iter().all(|r| {
            r["taps"]
                .as_array()
                .unwrap()
                .iter()
                .all(|t| t["bitwise_equal"] == true)
        });
        reports.push(json!({"case":c.name,"ratio":c.ratio,"rows":c.rows,"all_bitwise_equal":exact,"phases":phases}));
    }
    Ok(
        json!({"all_bitwise_equal":reports.iter().all(|r|r["all_bitwise_equal"]==true),"cases":reports,"weights":verified,
        "full_engine_acceptance":false,"scope":"GPU composite prefill pooling/RMS/cache, carried ring and pointer-stable metadata updates only; not full-model acceptance."}),
    )
}
