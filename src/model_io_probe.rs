//! Four-peer embedding/output checks; residual/pre are controlled inputs.
use crate::{
    checkpoint::Checkpoint,
    config::ModelConfig,
    cuda::Cuda,
    ffn::TokenRows,
    layer_probe::{Case, bf16_bytes, compare, fixture},
    layout::RankPlan,
    model_io::ModelIo,
    roce::Roce,
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
#[derive(Deserialize)]
struct Peer {
    config: PathBuf,
    checkpoint: PathBuf,
    rank: usize,
    core: PathBuf,
    kernels: PathBuf,
    staged: PathBuf,
    fixture: PathBuf,
    endpoint: String,
    ticket: PathBuf,
}
#[derive(Deserialize)]
struct Cases {
    rows: usize,
    world: usize,
    lm_head_output_dtype: String,
    #[serde(default)]
    head_gemv: bool,
    cases: Vec<IoCase>,
}
#[derive(Deserialize)]
struct IoCase {
    #[serde(flatten)]
    taps: Case,
    last_heads: Vec<LastHead>,
}
#[derive(Deserialize)]
struct LastHead {
    live: usize,
    global_taps: Vec<crate::layer_probe::Tap>,
}
fn float_bytes(buffer: &crate::cuda::Buffer<f32>) -> Result<Vec<u8>> {
    Ok(buffer
        .download()?
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect())
}
pub fn run(path: &Path) -> Result<Value> {
    let p: Peer = serde_json::from_slice(&std::fs::read(path)?)?;
    let f: Cases = serde_json::from_slice(&std::fs::read(p.fixture.join("index.json"))?)?;
    ensure!(
        f.rows == 6
            && f.world == 4
            && f.cases.len() == 4
            && p.rank < 4
            && f.lm_head_output_dtype == "FP32",
        "model IO fixture geometry"
    );
    let cfg = ModelConfig::parse(&std::fs::read(p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let plan = RankPlan::build(&cp, &cfg, p.rank, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let core = std::fs::read(&p.core)?;
    let mut io = ModelIo::new(&cuda, &core, &stream, &cp, &plan, &p.staged, &p.kernels, 6)?;
    ensure!(
        io.head_gemv_enabled() == f.head_gemv,
        "head GEMV fixture/backend mismatch"
    );
    let middle = if f.head_gemv {
        Some(io.fork_profile(&cuda, &core, &stream, &p.kernels, 3)?)
    } else {
        None
    };
    let mut middle_input = cuda.allocate::<bf16>(3 * 5120)?;
    middle_input.zero()?;
    let middle_graph = if let Some(middle) = &middle {
        ensure!(
            middle.head_gemv_enabled(),
            "M3 head GEMV was not constructed"
        );
        middle.run_normalized_head(&middle_input, &stream)?;
        stream.synchronize()?;
        Some(stream.capture(|| middle.run_normalized_head(&middle_input, &stream))?)
    } else {
        None
    };
    let mut residual = cuda.allocate::<bf16>(6 * 20480)?;
    let mut pre = cuda.allocate::<f32>(6 * 4)?;
    let mut tokens = TokenRows::new(&cuda, &[0; 6], 6)?;
    let mut comm = Roce::local(&cuda, &p.kernels.join("roce-v0"), p.rank)?;
    comm.attach_gather(&p.kernels.join("roce-gather-v1"))?;
    let all = crate::rendezvous::exchange_blob(
        p.rank,
        p.endpoint.parse()?,
        &std::fs::read(p.ticket)?,
        &comm.blob()?,
    )?;
    comm.connect(&all)?;
    let mut cases = Vec::new();
    let mut all_equal = true;
    for fixture_case in &f.cases {
        let case = &fixture_case.taps;
        let root = p.fixture.join(&case.name);
        let inputs: BTreeMap<_, _> = case
            .source
            .iter()
            .map(|t| Ok((t.name.as_str(), fixture(&root, t)?)))
            .collect::<Result<_>>()?;
        ensure!(
            inputs["tokens"].len() == 48
                && inputs["residual"].len() == residual.bytes()
                && inputs["pre"].len() == pre.bytes(),
            "model IO input byte sizes"
        );
        let ids: Vec<i64> = inputs["tokens"]
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        tokens.update(&ids, 6)?;
        residual.upload(
            &inputs["residual"]
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
                .collect::<Vec<_>>(),
        )?;
        pre.upload(
            &inputs["pre"]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>(),
        )?;
        io.run_input(&tokens, &comm, &stream)?;
        io.run_head(&residual, &pre, &stream)?;
        io.gather_logits(&comm, &stream)?;
        stream.synchronize()?;
        comm.health()?;
        let graph = stream.capture(|| {
            io.run_input(&tokens, &comm, &stream)?;
            io.run_head(&residual, &pre, &stream)?;
            io.gather_logits(&comm, &stream)
        })?;
        let repeats = if f.head_gemv { 5 } else { 100 };
        for _ in 0..if f.head_gemv { 3 } else { 100 } {
            unsafe { graph.launch(&stream) }?;
        }
        let events = (0..repeats)
            .map(|_| Ok((cuda.event()?, cuda.event()?)))
            .collect::<Result<Vec<_>>>()?;
        for (a, b) in &events {
            a.record(&stream)?;
            unsafe { graph.launch(&stream) }?;
            b.record(&stream)?;
        }
        stream.synchronize()?;
        comm.health()?;
        let mut ms = events
            .iter()
            .map(|(a, b)| b.elapsed_since(a))
            .collect::<Result<Vec<_>>>()?;
        ms.sort_by(f32::total_cmp);
        let local = BTreeMap::from([
            ("embedding_local".into(), bf16_bytes(io.local_embedding())?),
            ("logits_local".into(), float_bytes(io.local_logits())?),
        ]);
        let global = BTreeMap::from([
            ("embedding".into(), bf16_bytes(io.embedding())?),
            ("expanded".into(), bf16_bytes(io.input())?),
            ("normalized".into(), bf16_bytes(io.normalized())?),
            (
                "logits_fp32".into(),
                io.logits()
                    .download()?
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect(),
            ),
        ]);
        let expected = case
            .ranks
            .iter()
            .find(|r| r.rank == p.rank)
            .ok_or_else(|| anyhow::anyhow!("model IO rank trace absent"))?;
        let a = compare(
            &root.join(format!("rank{}", p.rank)),
            &expected.taps,
            &local,
        )?;
        let b = compare(&root, &case.global_taps, &global)?;
        let different: usize = a
            .iter()
            .chain(&b)
            .map(|t| t["different_values"].as_u64().unwrap() as usize)
            .sum();
        all_equal &= different == 0;
        let middle_equal = if let (Some(middle), Some(graph)) = (&middle, &middle_graph) {
            unsafe {
                cuda.copy_device(
                    middle_input.ptr(),
                    io.normalized().ptr(),
                    middle_input.bytes(),
                    &stream,
                )?;
                graph.launch(&stream)?;
            }
            stream.synchronize()?;
            let equal =
                float_bytes(middle.local_logits())? == local["logits_local"][..3 * 32320 * 4];
            all_equal &= equal;
            Some(equal)
        } else {
            None
        };
        ensure!(
            fixture_case
                .last_heads
                .iter()
                .map(|h| h.live)
                .collect::<Vec<_>>()
                == [6, 1, 3, 6],
            "changed last-live-row fixture sequence"
        );
        io.run_head_last(&residual, &pre, &tokens, &stream)?;
        io.gather_logits_last(&comm, &tokens, &stream)?;
        stream.synchronize()?;
        let last_graph = stream.capture(|| {
            io.run_head_last(&residual, &pre, &tokens, &stream)?;
            io.gather_logits_last(&comm, &tokens, &stream)
        })?;
        let mut last_results = Vec::new();
        for (index, last) in fixture_case.last_heads.iter().enumerate() {
            tokens.update(&ids, last.live)?;
            stream.join_default()?;
            unsafe {
                last_graph.launch(&stream)?;
            }
            stream.synchronize()?;
            comm.health()?;
            let actual = BTreeMap::from([("logits_fp32".into(), float_bytes(io.logits())?)]);
            let taps = compare(
                &root.join(format!("last{index}")),
                &last.global_taps,
                &actual,
            )?;
            all_equal &= taps.iter().all(|t| t["bitwise_equal"] == true);
            last_results.push(json!({"live":last.live,"taps":taps}));
        }
        eprintln!(
            "rank{} model IO {}: different={different}",
            p.rank, case.name
        );
        let len = ms.len();
        cases.push(json!({"case":case.name,"different_values":different,"local_taps":a,"global_taps":b,"captured_last_head":last_results,"m3_prefix_exact":middle_equal,"timing_samples":len,"gpu_ms":{"p50":ms[(len-1)/2],"p95":ms[(len*95).div_ceil(100)-1],"p99":ms[(len*99).div_ceil(100)-1],"max":ms[len-1]}}));
    }
    Ok(
        json!({"rank":p.rank,"all_bitwise_equal":all_equal,"head_gemv":io.head_gemv_enabled(),"cases":cases,"ordered_collectives_per_step":2,"scope":"Vocabulary-parallel embedding and predecessor-aware final norm; canonical BF16 head operands with FP32 accumulation/output/gather; captured live-row changes. Controlled hidden states, not complete model inference"}),
    )
}
