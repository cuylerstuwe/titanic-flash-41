//! Diagnostic for the connected query network on four physical TP peers.
//! Inputs are GPU-produced target features, never Python model execution plans.
use crate::{
    checkpoint::Checkpoint,
    config::ModelConfig,
    cuda::{Buffer, Cuda},
    dspark::QueryNetwork,
    layout::RankPlan,
    model_io::ModelIo,
    prelude::FrequencyTable,
    roce::Roce,
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};
#[derive(Deserialize)]
struct Input {
    config: PathBuf,
    checkpoint: PathBuf,
    rank: usize,
    core: PathBuf,
    kernels: PathBuf,
    staged: PathBuf,
    io_stage: PathBuf,
    features: PathBuf,
    features_sha256: String,
    endpoint: String,
    ticket: PathBuf,
    output: PathBuf,
}
pub fn run(path: &Path) -> Result<serde_json::Value> {
    let p: Input = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(
        p.rank < 4 && !p.output.exists(),
        "DSpark immutable TP diagnostic"
    );
    let cfg = ModelConfig::parse(&std::fs::read(p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let plan = RankPlan::build(&cp, &cfg, p.rank, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let core = std::fs::read(&p.core)?;
    let module = cuda.module(&core)?;
    let begin = Instant::now();
    let io = ModelIo::new(
        &cuda,
        &core,
        &stream,
        &cp,
        &plan,
        &p.io_stage,
        &p.kernels,
        5,
    )?;
    let frequency = Rc::new(FrequencyTable::new(
        &cuda, &module, &stream, 1048576, false,
    )?);
    let mut net = QueryNetwork::load_staged(
        &cuda, &core, &stream, &cp, &plan, &p.staged, &p.kernels, io, frequency,
    )?;
    let load_seconds = begin.elapsed().as_secs_f64();
    let mut comm = Roce::local(&cuda, &p.kernels.join("roce-v0"), p.rank)?;
    comm.attach_gather(&p.kernels.join("roce-gather-v1"))?;
    let blob = crate::rendezvous::exchange_blob(
        p.rank,
        p.endpoint.parse()?,
        &std::fs::read(p.ticket)?,
        &comm.blob()?,
    )?;
    comm.connect(&blob)?;
    // Raw payload transport qualification, including signed zeros, infinities
    // and NaN payloads. Integer patterns are byte fixtures, not CPU model math.
    let pattern = |i: usize, rank: usize, phase: usize| -> u32 {
        if i % 17 < 6 {
            [
                0u32, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, 0x7fa56789,
            ][i % 17]
        } else {
            (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add((rank as u32) * 104729)
                .wrapping_add((phase as u32) * 65537)
        }
    };
    let mut copy_input = cuda.allocate::<f32>(5 * 32320)?;
    let mut copy_output = cuda.allocate::<f32>(5 * 129280)?;
    let copy_graph = stream
        .capture(|| unsafe { comm.gather_f32(&copy_input, &mut copy_output, 32320, &stream) })?;
    for phase in 0..2 {
        copy_input.upload(
            &(0..5 * 32320)
                .map(|i| f32::from_bits(pattern(i, p.rank, phase)))
                .collect::<Vec<_>>(),
        )?;
        stream.join_default()?;
        for _ in 0..5 {
            unsafe { copy_graph.launch(&stream) }?;
        }
        stream.synchronize()?;
        comm.health()?;
        let words = copy_output.download()?;
        for row in 0..5 {
            for source in 0..4 {
                for col in 0..32320 {
                    ensure!(
                        words[(row * 4 + source) * 32320 + col].to_bits()
                            == pattern(row * 32320 + col, source, phase),
                        "FP32 gather changed payload"
                    );
                }
            }
        }
    }
    drop(copy_graph);
    drop(copy_input);
    drop(copy_output);
    let gather_screen=std::env::var("CANADA_SCRATCH_MARKOV_GATHER_SCREEN").as_deref()==Ok("1");
    let prefix_screen=std::env::var("CANADA_SCRATCH_MARKOV_PREFIX_STOP_SCREEN").as_deref()==Ok("1");
    ensure!(!(gather_screen&&prefix_screen),"one Markov component at a time");
    if gather_screen {
        let mut input=cuda.allocate::<f32>(32320)?;
        let mut output=cuda.allocate::<f32>(7*129280)?;
        for offset in [1,7*129280,usize::MAX] {
            ensure!(unsafe {comm.gather_f32_at(&input,&mut output,offset,32320,&stream)}.is_err(),
                "offset gather accepted an invalid range");
        }
        let sentinel=f32::from_bits(0x7fc0beef);
        for row in 1..=5 {
            input.upload(&(0..32320).map(|i|f32::from_bits(pattern(i,p.rank,row))).collect::<Vec<_>>())?;
            output.upload(&vec![sentinel;7*129280])?;
            stream.join_default()?;
            let graph=stream.capture(||unsafe {comm.gather_f32_at(&input,&mut output,row*129280,32320,&stream)})?;
            for _ in 0..3 {unsafe {graph.launch(&stream)}?;}
            stream.synchronize()?;comm.health()?;
            for (index,value)in output.download()?.iter().enumerate() {
                let expected=if index/129280==row {
                    pattern(index%32320,(index%129280)/32320,row)
                }else{sentinel.to_bits()};
                ensure!(value.to_bits()==expected,"offset gather payload/sentinel changed");
            }
        }
    }
    let raw = std::fs::read(&p.features)?;
    ensure!(
        raw.len() == 128 * 15360 * 2 && format!("{:x}", Sha256::digest(&raw)) == p.features_sha256,
        "DSpark real target feature identity"
    );
    let mut features: Buffer<bf16> = cuda.allocate(128 * 15360)?;
    features.upload(
        &raw.chunks_exact(2)
            .map(|w| bf16::from_bits(u16::from_le_bytes(w.try_into().unwrap())))
            .collect::<Vec<_>>(),
    )?;
    stream.join_default()?;
    std::fs::create_dir(&p.output)?;
    if std::env::var("CANADA_SCRATCH_CONDITIONING_GRAPH_SCREEN").as_deref()==Ok("1") {
        return net.probe_conditioning_graph(&core,&p.kernels,&features,&comm,&stream);
    }
    if std::env::var("CANADA_SCRATCH_CONDITIONING_SPLIT_SCREEN").as_deref()==Ok("1") {
        return net.probe_conditioning_split(&core,&p.kernels,&features,&comm,&stream);
    }
    let initial_guard = net.base_logits().is_err();
    let mut records = Vec::new();
    let mut previous = None;
    for (case, anchor, accepted, reset) in [
        ("prefix128", 1001, 128, false),
        ("anchor_changed", 1234, 0, false),
        ("extended129", 1234, 1, false),
        ("reset_prefix128", 1001, 128, true),
    ] {
        if reset {
            net.reset(&stream)?;
        }
        let start = net.position()?;
        net.inject(&features, 128, accepted, start, &stream)?;
        let output_guard = net.base_logits().is_err();
        let first = Instant::now();
        net.run(anchor, &comm, &stream)?;
        let first_seconds = first.elapsed().as_secs_f64();
        let bytes =
            unsafe { cuda.read_device(net.base_logits()?.ptr(), net.base_logits()?.bytes()) }?;
        let ms = net.measure_current(20, &comm, &stream)?;
        let replay =
            unsafe { cuda.read_device(net.base_logits()?.ptr(), net.base_logits()?.bytes()) }?;
        ensure!(bytes == replay, "DSpark graph replay changed base logits");
        if case == "reset_prefix128" {
            ensure!(
                previous.as_ref() == Some(&bytes),
                "DSpark reset/reinjection changed base logits"
            );
        }
        if case == "prefix128" {
            previous = Some(bytes.clone());
        }
        let trace = if gather_screen||prefix_screen {serde_json::Value::Null}else{net.trace(&p.output.join(case))?};
        ensure!(
            net.dump_proposals(&p.output.join(case)).is_err(),
            "DSpark query exposed stale Markov output"
        );
        let proposals = net.greedy_proposals(&comm, &stream)?;
        let repeated = net.greedy_proposals(&comm, &stream)?;
        ensure!(
            proposals == repeated,
            "DSpark greedy proposal replay changed"
        );
        let markov = if gather_screen||prefix_screen {serde_json::Value::Null}else{net.dump_proposals(&p.output.join(case))?};
        let markov_gather=if gather_screen {Some(net.probe_markov_gather(&comm,&stream)?)}else{None};
        let markov_prefix_stop=if prefix_screen {Some(net.probe_markov_prefix_stop(&comm,&stream)?)}else{None};
        records.push(serde_json::json!({"case":case,"anchor":anchor,"accepted":accepted,"position":net.position()?,
            "first_seconds":first_seconds,"query_graph_ms":ms,"replays":20,"replay_exact":true,"output_guard":output_guard,
            "base_logits_sha256":format!("{:x}",Sha256::digest(&bytes)),"trace":trace,
            "greedy_proposals":proposals,"proposal_replay_exact":true,"markov":markov,"markov_gather":markov_gather,
            "markov_prefix_stop":markov_prefix_stop}));
    }
    comm.health()?;
    Ok(
        serde_json::json!({"rank":p.rank,"load_seconds":load_seconds,"atlas_bytes":net.atlas_bytes(),
        "initial_output_guard":initial_guard,"records":records,"reset_exact":true,
        "fp32_gather_raw_payload_exact":true,
        "full_engine_acceptance":false,"markov_proposals":true,"numeric_oracle_compared":false,
        "scope":"Connected three-stage five-row TP4 DSpark graph and native greedy Markov proposals, retained conditioning, changed anchor, reset/reinjection and replay; no target acceptance or full generation performance claim."}),
    )
}
