//! Bounded single-GPU vision qualification. Reads/writes tensor bytes only;
//! numerical comparison is performed by tools/compare-vision-gpu.py on GPU.
use crate::{
    checkpoint::Checkpoint,
    config::ModelConfig,
    cuda::Cuda,
    layout::RankPlan,
    vision::{Geometry, Vision},
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Deserialize)]
struct Config {
    config: PathBuf,
    checkpoint: PathBuf,
    model: PathBuf,
    kernel: PathBuf,
    attention: PathBuf,
    reference: PathBuf,
    output: PathBuf,
    /// Optional serving vision-only atlas (logical rank-zero plan), used where the
    /// original safetensors are not present. Exactly one of model/stage is used.
    #[serde(default)]
    stage: Option<PathBuf>,
}
#[derive(Deserialize)]
struct Cases {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    geometry: Geometry,
    patches_sha256: String,
}
pub fn run(path: &Path) -> Result<Value> {
    let p: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    let cases: Cases = serde_json::from_slice(&std::fs::read(p.reference.join("index.json"))?)?;
    ensure!(
        !cases.cases.is_empty() && cases.cases.len() <= 8,
        "vision qualification case limit"
    );
    std::fs::create_dir(&p.output)?;
    let cfg = ModelConfig::parse(&std::fs::read(&p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let load = Instant::now();
    let vision = if let Some(stage) = &p.stage {
        let plan = RankPlan::build(&cp, &cfg, 0, 4)?;
        Vision::new_staged(&cuda, &std::fs::read(&p.kernel)?, &p.attention, &cp, &plan, stage)?
    } else {
        let plan = RankPlan::build(&cp, &cfg, 3, 4)?;
        Vision::new(&cuda, &std::fs::read(&p.kernel)?, &p.attention, &cp, &plan, &p.model)?
    };
    let profile = std::env::var("CANADA_SCRATCH_VISION_PROFILE").as_deref() == Ok("1");
    let load_seconds = load.elapsed().as_secs_f64();
    let mut records = Vec::new();
    for case in cases.cases {
        ensure!(
            !case.name.is_empty()
                && case
                    .name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'),
            "vision fixture basename"
        );
        let g = case.geometry.validate()?;
        let raw = std::fs::read(p.reference.join(&case.name).join("patches.bin"))?;
        ensure!(
            raw.len() == g.rows() * 588 * 2
                && format!("{:x}", Sha256::digest(&raw)) == case.patches_sha256,
            "vision input identity"
        );
        let mut patches = cuda.allocate::<bf16>(g.rows() * 588)?;
        patches.upload(
            &raw.chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
                .collect::<Vec<_>>(),
        )?;
        let root = p.output.join(&case.name);
        std::fs::create_dir(&root)?;
        let mut taps = Vec::new();
        let out=vision.encode_with_taps(&patches,g,&stream,&mut |name,ptr,n,dtype,stream|{
            stream.synchronize()?;
            // Large images have gate/up taps above the diagnostic read cap.
            // Stream bounded byte chunks; do not relax the shared CUDA guard
            // or materialize another full tensor in host memory.
            let bytes=n*if dtype=="BF16"{2}else{4};
            let mut file=std::fs::File::create(root.join(format!("{name}.bin")))?;
            let mut digest=Sha256::new();
            for offset in (0..bytes).step_by(32<<20) {
                let part=unsafe{cuda.read_device(ptr+offset as u64,(bytes-offset).min(32<<20))}?;
                file.write_all(&part)?;digest.update(&part);
            }
            taps.push(json!({"name":name,"dtype":dtype,"elements":n,"sha256":format!("{:x}",digest.finalize())}));
            Ok(())
        })?;
        let span = out.data.download()?;
        drop(out);
        let mut times = Vec::new();
        let mut repeat_exact = true;
        for repeat in 0..3 {
            let profiled = profile && repeat == 2;
            if profiled { stream.synchronize()?; cuda.profiler_start()?; }
            let start = cuda.event()?;
            let end = cuda.event()?;
            let wall = Instant::now();
            start.record(&stream)?;
            let value = vision.encode(&patches, g, &stream)?;
            end.record(&stream)?;
            stream.synchronize()?;
            if profiled { cuda.profiler_stop()?; }
            times.push(json!({"event_ms":end.elapsed_since(&start)?,"wall_ms":wall.elapsed().as_secs_f64()*1000.}));
            repeat_exact &= value
                .data
                .download()?
                .iter()
                .map(|x| x.to_bits())
                .eq(span.iter().map(|x| x.to_bits()));
        }
        let record = json!({"name":case.name,"geometry":g,"tokens":g.tokens(),"taps":taps,"timings":times,"repeat_span_exact":repeat_exact});
        println!(
            "{}",
            json!({"vision_case":case.name,"repeat_span_exact":repeat_exact,"timings":times})
        );
        records.push(record);
    }
    let report = json!({"weight_bytes":vision.weight_bytes(),"weight_source_sha256":vision.source_sha256(),"load_seconds":load_seconds,
        "cases":records,"gpu_numerical_comparison_pending":true,"scope":"Native vision tower and aligner; no full language model or HTTP claim"});
    std::fs::write(
        p.output.join("index.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}
