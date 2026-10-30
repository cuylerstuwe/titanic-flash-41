//! Small exact-operand reuse screen with hot and rotating canonical layers.
use crate::{checkpoint::Checkpoint, config::ModelConfig, cuda::{Buffer, Cuda},
    layout::RankPlan, weights::Weights, woa_medium::{Grouped, Scratch}};
use anyhow::{Result, ensure};
use half::bf16;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::{Path, PathBuf}, rc::Rc};

#[derive(Deserialize)]
struct Fixture { config: PathBuf, checkpoint: PathBuf, model: PathBuf, core: PathBuf, root: PathBuf }
fn bits(b: &Buffer<bf16>) -> Result<Vec<u16>> {
    Ok(b.download()?.iter().map(|v|v.to_bits()).collect())
}
pub fn run(path: &Path) -> Result<Value> {
    let f: Fixture = serde_json::from_slice(&std::fs::read(path)?)?;
    let config = ModelConfig::parse(&std::fs::read(f.config)?)?;
    let cp = Checkpoint::from_manifest(&f.checkpoint)?;
    let plan = RankPlan::build(&cp, &config, 0, 4)?;
    let cuda = Cuda::open(0)?;
    let stream = cuda.stream()?;
    let core = std::fs::read(f.core)?;
    let layers = [0, 4, 8, 12, 16, 20, 24, 28];
    let prefixes = layers.iter().map(|i|format!("layers.{i}.attn.wo_a.")).collect::<Vec<_>>();
    let names = prefixes.iter().map(String::as_str).collect::<Vec<_>>();
    let weights = Rc::new(Weights::load_prefixes(&cuda, &f.model, &cp, &plan, &names)?);
    let source = weights.qualify(&cp, &f.model, &plan)?;
    let mut profiles = Vec::new();
    let mut identities = Vec::new();
    let mut preparation_ms = Vec::new();
    for rows in [16, 128] {
        let scratch = Scratch::new(&cuda, &core, rows)?;
        let mut baseline = Vec::new();
        let mut cached = Vec::new();
        let start = std::time::Instant::now();
        for layer in layers {
            baseline.push(Grouped::new(&cuda, weights.clone(), layer, scratch.clone())?);
            let mut candidate = Grouped::new(&cuda, weights.clone(), layer, scratch.clone())?;
            candidate.prepare_cached(&stream)?;
            cached.push(candidate);
        }
        preparation_ms.push(start.elapsed().as_secs_f64()*1000.);
        identities.push(cached.iter().map(|v|v.cached_allocation().unwrap()).collect::<Vec<_>>());
        profiles.push((rows, baseline, cached));
    }
    ensure!(identities[0] == identities[1], "WO-A profiles duplicated decoded operands");
    ensure!(identities[0].iter().map(|v|v.0).collect::<BTreeSet<_>>().len()==layers.len(),
        "WO-A different layers alias decoded operands");
    let cached_bytes: usize = identities[0].iter().map(|v|v.1).sum();
    let mut cases = Vec::new();
    for (rows, baseline, cached) in &mut profiles {
        for phase in ["a", "b"] {
            let raw = std::fs::read(f.root.join(format!("m{rows}-{phase}.bin")))?;
            ensure!(raw.len()==*rows*8192*2, "WO-A cache input size");
            let data = raw.chunks_exact(2).map(|v|bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap()))).collect::<Vec<_>>();
            ensure!(data.iter().all(|v|v.is_finite()), "WO-A cache finite input");
            let mut input = cuda.allocate(data.len())?;
            input.upload(&data)?;
            for i in 0..layers.len() { baseline[i].run(&input, &stream)?; cached[i].run(&input, &stream)?; }
            stream.synchronize()?;
            for i in 0..layers.len() {
                ensure!(bits(baseline[i].output())?==bits(cached[i].output())?, "WO-A eager operand reuse differs");
            }
            for count in [1, layers.len()] {
                let control = stream.capture(|| {
                    for op in baseline.iter_mut().take(count) { op.run(&input, &stream)?; }
                    Ok(())
                })?;
                let candidate = stream.capture(|| {
                    for op in cached.iter_mut().take(count) { op.run(&input, &stream)?; }
                    Ok(())
                })?;
                let graphs = [control, candidate];
                let mut pairs = Vec::new();
                for repeat in 0..4 {
                    let order = if repeat % 2 == 0 { [0, 1] } else { [1, 0] };
                    let mut milliseconds = [0f32; 2];
                    for variant in order {
                        for _ in 0..8 { unsafe { graphs[variant].launch(&stream)?; } }
                        stream.synchronize()?;
                        let start = cuda.event()?; let end = cuda.event()?;
                        start.record(&stream)?;
                        for _ in 0..32 { unsafe { graphs[variant].launch(&stream)?; } }
                        end.record(&stream)?;
                        milliseconds[variant] = end.elapsed_since(&start)? / (32 * count) as f32;
                    }
                    pairs.push(json!({"repeat":repeat,"order":order,"ms_per_layer":milliseconds}));
                }
                stream.synchronize()?;
                for i in 0..count {
                    ensure!(bits(baseline[i].output())?==bits(cached[i].output())?, "WO-A captured operand reuse differs");
                }
                cases.push(json!({"rows":rows,"phase":phase,"rotating_layers":count,
                    "all_exact":true,"pairs":pairs}));
            }
        }
    }
    let other = cuda.stream()?;
    let input = cuda.allocate::<bf16>(16*8192)?;
    ensure!(profiles[0].2[0].run(&input, &other).is_err(), "WO-A scratch accepted a different stream");
    stream.synchronize()?;
    Ok(json!({"all_exact":true,"layers":layers,"cases":cases,"source":source,
        "same_cached_allocations_across_profiles":true,"cached_weight_bytes":cached_bytes,
        "preparation_and_constructor_ms":preparation_ms,"wrong_stream_rejected":true,
        "scope":"Unchanged BF16 GEMM arithmetic; exact GPU-decoded canonical operands. Component latency only, not outcome speed."}))
}
