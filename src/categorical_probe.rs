//! Independent GPU-reference and frequency qualification for categorical draws.
use crate::{
    categorical::{Categorical, DRAFT_DOMAIN, TARGET_DOMAIN},
    cuda::{Buffer, Cuda},
    rejection::Rejection,
    sampling::VOCAB,
};
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Deserialize)]
struct Case {
    name: String,
    rows: usize,
    logits: String,
    temperature: f32,
    uniforms: Vec<f64>,
    expected: Vec<u32>,
    invalid: bool,
    sha256: String,
    cdf_reference: Option<CdfReference>,
}
#[derive(Deserialize)]
struct CdfReference {
    file: String,
    sha256: String,
}
#[derive(Deserialize)]
struct Histogram {
    rows: usize,
    iterations: usize,
    logits: String,
    ids: Vec<u32>,
    probabilities: Vec<f64>,
    seed: u64,
}
#[derive(Deserialize)]
struct Fixture {
    core: PathBuf,
    root: PathBuf,
    cases: Vec<Case>,
    histogram: Histogram,
    rejection: Vec<RejectionCase>,
}
#[derive(Deserialize)]
struct RejectionCase {
    name: String,
    rows: usize,
    iterations: usize,
    p: String,
    q: String,
    p_sha256: String,
    q_sha256: String,
    ids: Vec<u32>,
    probabilities: Vec<f64>,
    expected_acceptance: f64,
    uniforms: Vec<Vec<f64>>,
    proposals: Vec<u32>,
    expected: Vec<u32>,
    expected_accepted: Vec<bool>,
}
fn upload(cuda: &Cuda, path: &Path, rows: usize) -> Result<Buffer<f32>> {
    let raw = std::fs::read(path)?;
    ensure!(raw.len() == rows * VOCAB * 4, "categorical fixture shape");
    let v: Vec<_> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut out = cuda.allocate(v.len())?;
    out.upload(&v)?;
    Ok(out)
}
pub fn run(path: &Path) -> Result<Value> {
    let f: Fixture = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(
        (1..=32).contains(&f.cases.len()),
        "categorical fixture cases"
    );
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let core = std::fs::read(&f.core)?;
    let mut cases = Vec::new();
    for c in f.cases {
        let path = f.root.join(&c.logits);
        ensure!(
            format!("{:x}", Sha256::digest(std::fs::read(&path)?)) == c.sha256,
            "categorical fixture hash"
        );
        let logits = upload(&cuda, &path, c.rows)?;
        let mut sampler = Categorical::new(&cuda, &core, c.rows)?;
        let got = sampler.sample_with_uniforms(&logits, 0, c.temperature, &c.uniforms, &stream);
        let cdf = if let Some(reference) = &c.cdf_reference {
            let bytes = std::fs::read(f.root.join(&reference.file))?;
            ensure!(
                c.name == "extreme-range"
                    && bytes.len() == c.rows * VOCAB * 8
                    && format!("{:x}", Sha256::digest(&bytes)) == reference.sha256,
                "FP64 CDF reference"
            );
            Some(
                bytes
                    .chunks_exact(8)
                    .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let exact = got.as_ref().is_ok_and(|v| *v == c.expected);
        let mut underflow_edges = Vec::new();
        let correct = if c.invalid {
            got.is_err()
        } else {
            got.as_ref().is_ok_and(|ids|ids.len()==c.expected.len()&&ids.iter().zip(&c.expected).enumerate().all(|(row,(&actual,&expected))|{
                if actual==expected{return true;}
                if c.uniforms[row]!=0. {return false;}
                let Some(cdf)=&cdf else{return false;};
                let after=cdf[row*VOCAB+actual as usize];
                let before=if actual==0 {0.}else{cdf[row*VOCAB+actual as usize-1]};
                let bounded=before>=0.&&after>before&&after<1e-38;
                underflow_edges.push(json!({"row":row,"reference_cdf_before":before,"reference_cdf_after":after,
                    "allowed_absolute_mass":1e-38,"passed":bounded}));
                bounded
            }))
        };
        let mut actual_q = cuda.allocate(c.rows * VOCAB)?;
        sampler.probabilities(&logits, 0, c.rows, c.temperature, &mut actual_q, 0, &stream)?;
        stream.synchronize()?;
        let q_bytes = unsafe { cuda.read_device(actual_q.ptr(), actual_q.bytes()) }?;
        let actual_q_sha256 = format!("{:x}", Sha256::digest(q_bytes));
        cases.push(json!({"name":c.name,"actual_q_sha256":actual_q_sha256,"correct":correct,"exact_reference_ids":exact,"expected":c.expected,
            "actual":got.as_ref().ok(),"rejected":got.is_err(),"zero_uniform_underflow_edges":underflow_edges}));
    }
    // Integer-only permutation proof for the optional candidate locality sort.
    let original: Vec<i32> = (0..8)
        .flat_map(|r| {
            (0..512).map(move |i| {
                if (i + r) % 31 == 0 {
                    -1
                } else {
                    ((i * 317 + r * 53) % 512) as i32
                }
            })
        })
        .collect();
    let mut indices = cuda.allocate(original.len())?;
    indices.upload(&original)?;
    stream.join_default()?;
    let module = cuda.module(&core)?;
    let mut ptr = indices.ptr();
    let mut args = [(&mut ptr as *mut u64).cast()];
    unsafe {
        module.function("ca_sort_selected512")?.launch(
            [8, 1, 1],
            [512, 1, 1],
            0,
            &stream,
            &mut args,
        )?;
    }
    stream.synchronize()?;
    let sorted = indices.download()?;
    let sort_correct = original.chunks(512).zip(sorted.chunks(512)).all(|(a, b)| {
        let mut expected = a.to_vec();
        expected.sort_by_key(|&x| if x < 0 { i32::MAX } else { x });
        expected == b
    });
    let h = f.histogram;
    ensure!(
        h.rows == 128 && h.iterations == 1024 && h.ids.len() == 4 && h.probabilities.len() == 4,
        "bounded histogram fixture"
    );
    let logits = upload(&cuda, &f.root.join(h.logits), h.rows)?;
    let mut sampler = Categorical::new(&cuda, &core, h.rows)?;
    let mut counts = [0u64; 4];
    let mut other = 0u64;
    let began = Instant::now();
    for batch in 0..h.iterations {
        let ids = sampler.sample(
            &logits,
            0,
            h.rows,
            1.,
            h.seed,
            TARGET_DOMAIN,
            (batch * h.rows) as u64,
            &stream,
        )?;
        for id in ids {
            if let Some(i) = h.ids.iter().position(|&x| x == id) {
                counts[i] += 1;
            } else {
                other += 1;
            }
        }
    }
    let seconds = began.elapsed().as_secs_f64();
    // Diagnostic frequency statistics only. The reference probabilities above
    // were computed by independent FP64 GPU softmax, not a host model oracle.
    let n = (h.rows * h.iterations) as f64;
    let chi_square: f64 = counts
        .iter()
        .zip(&h.probabilities)
        .map(|(&c, &p)| {
            let d = c as f64 - n * p;
            d * d / (n * p)
        })
        .sum();
    let first = sampler.sample(&logits, 0, h.rows, 1., h.seed, TARGET_DOMAIN, 128, &stream)?;
    let repeated = sampler.sample(&logits, 0, h.rows, 1., h.seed, TARGET_DOMAIN, 128, &stream)?;
    let repeat_exact = first == repeated;
    let frequency_pass = other == 0 && chi_square < 25.; // df3, conservative fixed gate.
    let mut rejection_reports = Vec::new();
    ensure!(f.rejection.len() == 4, "rejection fixture case count");
    for c in f.rejection {
        ensure!(
            c.rows == 64
                && c.iterations == 512
                && c.ids.len() == 4
                && c.probabilities.len() == 4
                && c.uniforms.len() == 3,
            "rejection fixture bounds"
        );
        for (file, hash) in [(&c.p, &c.p_sha256), (&c.q, &c.q_sha256)] {
            ensure!(
                format!("{:x}", Sha256::digest(std::fs::read(f.root.join(file))?)) == *hash,
                "rejection fixture hash"
            );
        }
        let p = upload(&cuda, &f.root.join(&c.p), c.rows)?;
        let q = upload(&cuda, &f.root.join(&c.q), c.rows)?;
        let mut q_sampler = Categorical::new(&cuda, &core, c.rows)?;
        let mut q_probs = cuda.allocate(c.rows * VOCAB)?;
        let mut rejection = Rejection::new(&cuda, &core, c.rows)?;
        let proposed = q_sampler.sample_with_uniforms(&q, 0, 1., &c.uniforms[0], &stream)?;
        // The draft path captures categorical draws; changed uniforms are data
        // inputs, never a reason to capture or reuse a stale sampled token.
        let graph = stream.capture(|| q_sampler.enqueue(&q, 0, c.rows, 1., &stream))?;
        unsafe {
            graph.launch(&stream)?;
        }
        let graph_ids = q_sampler.finish(c.rows, &stream)?;
        q_sampler.probabilities(&q, 0, c.rows, 1., &mut q_probs, 0, &stream)?;
        let explicit = rejection.explicit(
            &p,
            &q_probs,
            &proposed,
            &c.uniforms[1],
            &c.uniforms[2],
            &c.uniforms[0],
            &stream,
        )?;
        let explicit_pass = proposed == c.proposals
            && graph_ids == proposed
            && explicit.ids == c.expected
            && explicit.accepted == c.expected_accepted;
        let mut counts = [0u64; 4];
        let mut other = 0u64;
        let mut accepted = 0u64;
        let began = Instant::now();
        for batch in 0..c.iterations {
            let pos = (batch * c.rows) as u64;
            let uniforms: Vec<_> = (0..c.rows)
                .map(|r| crate::categorical::uniform(20261024, DRAFT_DOMAIN, pos + r as u64))
                .collect();
            q_sampler.upload_uniforms(&uniforms)?;
            stream.join_default()?;
            unsafe {
                graph.launch(&stream)?;
            }
            let proposed = q_sampler.finish(c.rows, &stream)?;
            q_sampler.probabilities(&q, 0, c.rows, 1., &mut q_probs, 0, &stream)?;
            let draw = rejection.sample(&p, &q_probs, &proposed, c.rows, 20261024, pos, &stream)?;
            accepted += draw.accepted.iter().filter(|&&a| a).count() as u64;
            for id in draw.ids {
                if let Some(i) = c.ids.iter().position(|&x| x == id) {
                    counts[i] += 1;
                } else {
                    other += 1;
                }
            }
        }
        let seconds = began.elapsed().as_secs_f64();
        let n = (c.rows * c.iterations) as f64;
        let chi: f64 = counts
            .iter()
            .zip(&c.probabilities)
            .map(|(&c, &p)| {
                let d = c as f64 - n * p;
                d * d / (n * p)
            })
            .sum();
        let rate = c.expected_acceptance;
        let acceptance_z =
            (accepted as f64 - n * rate).abs() / (n * rate * (1. - rate)).sqrt().max(1.);
        let passed = explicit_pass && other == 0 && chi < 25. && acceptance_z < 6.;
        rejection_reports.push(
            json!({"name":c.name,"passed":passed,"explicit_reference_exact":explicit_pass,
            "samples":n,"counts":counts,"other":other,"gpu_reference_probabilities":c.probabilities,
            "chi_square_df3":chi,"expected_acceptance":rate,"actual_acceptance":accepted as f64/n,
            "acceptance_z":acceptance_z,"seconds":seconds}),
        );
    }
    let rejection_pass = rejection_reports.iter().all(|r| r["passed"] == true);
    Ok(
        json!({"all_correct":cases.iter().all(|c|c["correct"]==true)&&frequency_pass&&repeat_exact&&sort_correct&&rejection_pass,
        "cases":cases,"histogram":{"samples":n,"ids":h.ids,"counts":counts,"other":other,"gpu_reference_probabilities":h.probabilities,
            "chi_square_df3":chi_square,"threshold":25.,"passed":frequency_pass,"seconds":seconds},
        "cached_probability_factors":sampler.cached_probability_factors(),
        "parallel_finish":sampler.parallel_finish_enabled(),
        "normalized_factor":sampler.normalized_factor_enabled(),
        "rejection":rejection_reports,"sort_selected512_exact":sort_correct,"counter_replay_exact":repeat_exact,"full_engine_acceptance":false,
        "scope":"GPU categorical sampler only, explicit-uniform comparison with independent CUDA FP64 softmax/CDF reference, invalid-input rejection,131072draw frequency test and counter replay. Does not qualify full-model quality or stochastic DSpark performance."}),
    )
}
