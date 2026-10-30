//! Small independent execution probes, never a substitute for model qualification.
use crate::cuda::Cuda;
use anyhow::{Result, ensure};
use half::bf16;
use serde_json::json;
use std::path::Path;

pub fn roce(rank: usize, endpoint: &str, nonce: &[u8], root: &Path) -> Result<serde_json::Value> {
    let c = Cuda::open(0)?;
    c.qualify_gb10()?;
    let stream = c.stream()?;
    let mut comm = crate::roce::Roce::local(&c, root, rank)?;
    let capacity_screen=std::env::var("CANADA_SCRATCH_ROCE_CAPACITY_SCREEN").as_deref()==Ok("1");
    if capacity_screen {comm.attach_gather(&root.parent().unwrap().join("roce-gather-v1"))?;}
    let local = comm.blob()?;
    let all = crate::rendezvous::exchange_blob(rank, endpoint.parse()?, nonce, &local)?;
    comm.connect(&all)?;
    let mut report = Vec::new();
    for rows in [1usize, 6, 161, 460, 1024] {
        let mut input = c.allocate::<u16>(rows * 5120)?;
        let mut output = c.allocate::<u16>(rows * 5120)?;
        input.upload(&vec![
            bf16::from_f32((rank + 1) as f32).to_bits();
            input.len()
        ])?;
        unsafe { comm.sum_bf16(&input, &mut output, &stream) }?;
        stream.synchronize()?;
        comm.health()?;
        ensure!(
            output
                .download()?
                .iter()
                .all(|v| bf16::from_bits(*v).to_f32() == 10.),
            "native RoCE sum mismatch"
        );
        input.zero()?;
        for _ in 0..3 {
            unsafe { comm.sum_bf16(&input, &mut output, &stream) }?;
        }
        stream.synchronize()?;
        comm.health()?;
        let graph = stream.capture(|| unsafe { comm.sum_bf16(&input, &mut output, &stream) })?;
        let start = c.event()?;
        let end = c.event()?;
        start.record(&stream)?;
        for _ in 0..50 {
            unsafe { graph.launch(&stream) }?;
        }
        end.record(&stream)?;
        let ms = end.elapsed_since(&start)? / 50.;
        comm.health()?;
        ensure!(
            output.download()?.iter().all(|v| *v == 0),
            "native RoCE graph replay mismatch"
        );
        report.push(json!({"rows":rows,"bytes":input.bytes(),"graph_ms":ms,"correct":true}));
    }
    stream.synchronize()?;
    comm.health()?;
    if capacity_screen {
        // Constant bit markers test transport layout, not a CPU model oracle.
        for (rows,columns) in [(1usize,32320usize),(6,32320),(16,32320),(128,32320),(8192,448)] {
            let mut input=c.allocate::<u16>(rows*columns)?;
            input.upload(&vec![0x3f80+rank as u16;input.len()])?;
            let mut output=c.allocate::<u16>(input.len()*4)?;
            unsafe{comm.gather_bf16(&input,&mut output,columns,&stream)?;}
            stream.synchronize()?;comm.health()?;
            let bytes=output.download()?;
            for row in bytes.chunks_exact(columns*4) {
                for(peer,part)in row.chunks_exact(columns).enumerate(){
                    ensure!(part.iter().all(|&v|v==0x3f80+peer as u16),"BF16 row-gather layout differs");
                }
            }
            let graph=stream.capture(||unsafe{comm.gather_bf16(&input,&mut output,columns,&stream)})?;
            for _ in 0..8 {unsafe{graph.launch(&stream)}?;}
            stream.synchronize()?;comm.health()?;
            ensure!(output.download()?==bytes,"BF16 row-gather replay differs");
            report.push(json!({"operation":"bf16-row-gather","rows":rows,"columns":columns,"bytes":input.bytes(),"correct":true}));
        }
        for rows in [6usize,16,128] {
            let columns=32320;let mut input=c.allocate::<f32>(rows*columns)?;
            input.upload(&vec![f32::from_bits(0x3f800000+rank as u32);input.len()])?;
            let mut output=c.allocate::<f32>(input.len()*4)?;
            unsafe{comm.gather_f32(&input,&mut output,columns,&stream)?;}
            stream.synchronize()?;comm.health()?;
            let bytes=output.download()?;
            for row in bytes.chunks_exact(columns*4) {
                for(peer,part)in row.chunks_exact(columns).enumerate(){
                    ensure!(part.iter().all(|v|v.to_bits()==0x3f800000+peer as u32),"FP32 row-gather layout differs");
                }
            }
            let graph=stream.capture(||unsafe{comm.gather_f32(&input,&mut output,columns,&stream)})?;
            for _ in 0..8 {unsafe{graph.launch(&stream)}?;}
            stream.synchronize()?;comm.health()?;
            ensure!(output.download()?==bytes,"FP32 row-gather replay differs");
            report.push(json!({"operation":"fp32-row-gather","rows":rows,"columns":columns,"bytes":input.bytes(),"correct":true}));
        }
        let bound=comm.transport_memory()["slot_bytes"].as_u64().unwrap()as usize;
        let input=c.allocate::<u16>((bound+16)/2)?;let mut output=c.allocate::<u16>(input.len())?;
        ensure!(unsafe{comm.sum_bf16(&input,&mut output,&stream)}.is_err(),"oversized native message was admitted");
        comm.health()?;
        report.push(json!({"operation":"oversized-message-rejection","bytes":input.bytes(),"correct":true}));
    }
    Ok(json!({"rank":rank,"transport":"Rust-owned RoCEnante dependency","transport_memory":comm.transport_memory(),"results":report}))
}

pub fn collective(rank: usize, endpoint: &str, nonce: &[u8]) -> Result<serde_json::Value> {
    let c = Cuda::open(0)?;
    c.qualify_gb10()?;
    let stream = c.stream()?;
    let api = crate::collective::Nccl::load()?;
    let id = crate::rendezvous::join(&api, rank, endpoint.parse()?, nonce)?;
    let comm = api.join(&c, id, rank, 4)?;
    let mut results = Vec::new();
    for rows in [1usize, 6, 161, 460, 1024] {
        let mut buffer = c.allocate::<u16>(rows * 5120)?;
        let values = vec![bf16::from_f32((rank + 1) as f32).to_bits(); buffer.len()];
        buffer.upload(&values)?;
        unsafe { comm.sum_bf16(&mut buffer, &stream) }?;
        stream.synchronize()?;
        ensure!(
            buffer
                .download()?
                .iter()
                .all(|v| bf16::from_bits(*v).to_f32() == 10.),
            "four-rank BF16 sum mismatch"
        );
        buffer.zero()?;
        for _ in 0..3 {
            unsafe { comm.sum_bf16(&mut buffer, &stream) }?;
        }
        stream.synchronize()?;
        let graph = stream.capture(|| unsafe { comm.sum_bf16(&mut buffer, &stream) })?;
        let a = c.event()?;
        let b = c.event()?;
        a.record(&stream)?;
        for _ in 0..50 {
            unsafe { graph.launch(&stream) }?;
        }
        b.record(&stream)?;
        let ms = b.elapsed_since(&a)? / 50.;
        comm.check_async()?;
        ensure!(
            buffer.download()?.iter().all(|v| *v == 0),
            "zero-sum graph replay mismatch"
        );
        results.push(json!({"operation":"all_reduce","rows":rows,"bytes":buffer.bytes(),"graph_ms":ms,"correct":true}));
    }
    let mut local = c.allocate::<u16>(32320)?;
    local.upload(&vec![bf16::from_f32(rank as f32).to_bits(); 32320])?;
    let mut global = c.allocate::<u16>(129280)?;
    unsafe { comm.gather_bf16(&local, &mut global, &stream) }?;
    stream.synchronize()?;
    let gathered = global.download()?;
    for (i, part) in gathered.chunks_exact(32320).enumerate() {
        ensure!(
            part.iter()
                .all(|v| bf16::from_bits(*v).to_f32() == i as f32),
            "vocabulary gather rank order mismatch"
        );
    }
    comm.check_async()?;
    results.push(json!({"operation":"vocabulary_gather","local_rows":32320,"global_rows":129280,"correct":true}));
    Ok(json!({"rank":rank,"transport":"NCCL over private four-rank fabric","results":results}))
}

pub fn linear(
    input: &Path,
    weights: &Path,
    output: &Path,
    m: usize,
    n: usize,
    k: usize,
) -> Result<serde_json::Value> {
    fn read(p: &Path) -> Result<Vec<u16>> {
        let b = std::fs::read(p)?;
        ensure!(b.len() % 2 == 0, "BF16 file alignment");
        Ok(b.chunks_exact(2)
            .map(|a| u16::from_le_bytes(a.try_into().unwrap()))
            .collect())
    }
    let c = Cuda::open(0)?;
    c.qualify_gb10()?;
    let stream = c.stream()?;
    let x = read(input)?;
    let w = read(weights)?;
    ensure!(
        x.len() == m * k && w.len() == n * k,
        "linear fixture dimensions"
    );
    let mut dx = c.allocate::<u16>(x.len())?;
    dx.upload(&x)?;
    let mut dw = c.allocate::<u16>(w.len())?;
    dw.upload(&w)?;
    let mut dy = c.allocate::<u16>(m * n)?;
    // Match the effective reference's GEMMEx contract. Lt is an independent
    // candidate backend and cannot silently substitute a heuristic algorithm.
    let blas = crate::blas::Blas::new(&c)?;
    for _ in 0..3 {
        unsafe { blas.linear_bf16(&dx, &dw, &mut dy, m, n, k, &stream) }?;
    }
    stream.synchronize()?;
    let graph =
        stream.capture(|| unsafe { blas.linear_bf16(&dx, &dw, &mut dy, m, n, k, &stream) })?;
    let a = c.event()?;
    let b = c.event()?;
    a.record(&stream)?;
    for _ in 0..50 {
        unsafe { graph.launch(&stream) }?;
    }
    b.record(&stream)?;
    let ms = b.elapsed_since(&a)? / 50.;
    let result = dy.download()?;
    let raw: Vec<u8> = result.iter().flat_map(|a| a.to_le_bytes()).collect();
    std::fs::write(output, raw)?;
    Ok(
        json!({"m":m,"n":n,"k":k,"graph_ms":ms,"output":"BF16","accumulation":"FP32","backend":"cublasGemmEx/32MiB/default_tensor_op"}),
    )
}
/// Real checkpoint projection through the Rust loader, independent scale
/// packing, Rust-owned workspace, and explicit low-level kernel binding.
pub fn fp8_checkpoint(
    config: &std::path::Path,
    manifest: &std::path::Path,
    root: &std::path::Path,
    rank: usize,
    prefix: &str,
    core: &std::path::Path,
    kernels: &std::path::Path,
) -> anyhow::Result<serde_json::Value> {
    use crate::{
        blockscaled::Matrix, checkpoint::Checkpoint, config::ModelConfig, fp8_linear::Linear,
        layout::RankPlan, weights::Weights,
    };
    use half::bf16;
    use sha2::{Digest, Sha256};
    use std::rc::Rc;
    let cfg = ModelConfig::parse(&std::fs::read(config)?)?;
    let cp = Checkpoint::from_manifest(manifest)?;
    let plan = RankPlan::build(&cp, &cfg, rank, 4)?;
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(kernels.join("manifest.json"))?)?;
    anyhow::ensure!(
        metadata["geometry"]["checkpoint_prefix"].as_str() == Some(prefix),
        "oracle checkpoint matrix differs"
    );
    let cuda = crate::cuda::Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let module = cuda.module(&std::fs::read(core)?)?;
    let weights = Rc::new(Weights::load(
        &cuda,
        root,
        &cp,
        &plan,
        &format!("{prefix}."),
    )?);
    let weight_report = weights.qualify(&cp, root, &plan)?;
    let matrix = Matrix::prepare(&cuda, &module, &stream, &weights, prefix)?;
    let input = std::fs::read(kernels.join("input-bf16.bin"))?;
    anyhow::ensure!(input.len() % 2 == 0, "truncated BF16 fixture");
    let data: Vec<_> = input
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect();
    let mut device_input = cuda.allocate::<bf16>(data.len())?;
    device_input.upload(&data)?;
    let linear = Linear::new(&cuda, kernels, matrix)?;
    linear.run(&device_input, &stream)?;
    stream.synchronize()?;
    let expected = std::fs::read(kernels.join("output-expected.bin"))?;
    let actual: Vec<u8> = linear
        .output()
        .download()?
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect();
    anyhow::ensure!(
        actual == expected,
        "canonical Rust projection differs from reference"
    );
    let graph = stream.capture(|| linear.run(&device_input, &stream))?;
    let start = cuda.event()?;
    let end = cuda.event()?;
    start.record(&stream)?;
    for _ in 0..50 {
        unsafe { graph.launch(&stream) }?;
    }
    end.record(&stream)?;
    let ms = end.elapsed_since(&start)? / 50.;
    let replay: Vec<u8> = linear
        .output()
        .download()?
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect();
    anyhow::ensure!(replay == expected, "canonical projection replay differs");
    Ok(
        serde_json::json!({"prefix":prefix,"rank":rank,"weights":weight_report,"outputs_bitwise_equal":true,
      "output_values":expected.len()/2,"graph_ms":ms,"replays":50,
      "input_sha256":format!("{:x}",Sha256::digest(input)),"output_sha256":format!("{:x}",Sha256::digest(expected)),
      "runtime_python_required":false,"captured_python_plan_required":false}),
    )
}

/// GPU-reference validation of canonical MoE weights and Rust-owned execution.
/// Replays one captured graph across changed and zero inputs to catch stale
/// scratch, captured pointers, and route epochs. No CPU MoE/packing oracle.
pub fn moe_checkpoint(
    config: &Path,
    manifest: &Path,
    root: &Path,
    rank: usize,
    layer: usize,
    core: &Path,
    kernels: &Path,
    fixtures: &Path,
) -> Result<serde_json::Value> {
    use crate::{
        checkpoint::Checkpoint,
        config::ModelConfig,
        expert_packing,
        layout::RankPlan,
        moe::{Expert, Workspace},
        weights::Weights,
    };
    use std::{collections::BTreeMap, fs::File, rc::Rc};
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        reference_graph_ms: f64,
    }
    #[derive(serde::Deserialize)]
    struct Cases {
        layer: usize,
        rank: usize,
        rows: usize,
        gpu_prepared_sha256: BTreeMap<String, String>,
        cases: Vec<Case>,
    }
    fn bf16_file(p: &Path) -> Result<Vec<bf16>> {
        let b = std::fs::read(p)?;
        ensure!(b.len() % 2 == 0, "BF16 fixture length");
        Ok(b.chunks_exact(2)
            .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])))
            .collect())
    }
    let cases: Cases = serde_json::from_reader(File::open(fixtures.join("index.json"))?)?;
    ensure!(
        (cases.layer, cases.rank) == (layer, rank)
            && crate::config::target_rows(cases.rows)
            && cases.cases.len() == 3,
        "canonical MoE oracle geometry mismatch"
    );
    let cfg = ModelConfig::parse(&std::fs::read(config)?)?;
    let cp = Checkpoint::from_manifest(manifest)?;
    let plan = RankPlan::build(&cp, &cfg, rank, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let module = cuda.module(&std::fs::read(core)?)?;
    let stream = cuda.stream()?;
    let mut weights = Weights::load(
        &cuda,
        root,
        &cp,
        &plan,
        &format!("layers.{layer}.ffn.experts."),
    )?;
    let source = weights.qualify(&cp, root, &plan)?;
    let prepared = expert_packing::prepare_layer(&cuda, &module, &mut weights, layer)?;
    let hashes = expert_packing::prepared_hashes(&weights, &prepared)?;
    ensure!(
        hashes == cases.gpu_prepared_sha256,
        "canonical expert GPU-prepared bytes differ from reference GPU"
    );
    let expert = Expert::new(Rc::new(weights), prepared.clone())?;
    let workspace = Workspace::new(&cuda, kernels)?;
    let mut input = cuda.allocate::<bf16>(cases.rows * 5120)?;
    let mut ids = cuda.allocate::<i32>(cases.rows * 6)?;
    let mut routing = cuda.allocate::<f32>(cases.rows * 6)?;
    let mut graph = None;
    let mut reports = Vec::new();
    for case in cases.cases {
        ensure!(
            ["initial", "changed", "zero"].contains(&case.name.as_str()),
            "unknown oracle case"
        );
        let base = fixtures.join(&case.name);
        let data = bf16_file(&base.join("input.bin"))?;
        ensure!(data.len() == input.len(), "MoE input fixture length");
        input.upload(&data)?;
        let ib = std::fs::read(base.join("ids.bin"))?;
        let rb = std::fs::read(base.join("routing.bin"))?;
        ensure!(
            ib.len() == cases.rows * 6 * 4 && rb.len() == cases.rows * 6 * 4,
            "MoE routing fixture length"
        );
        let iv: Vec<_> = ib
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let rv: Vec<_> = rb
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(
            iv.iter().all(|&v| (0..384).contains(&v))
                && rv.iter().all(|v| v.is_finite() && *v >= 0.),
            "MoE oracle route domain"
        );
        ids.upload(&iv)?;
        routing.upload(&rv)?;
        if graph.is_none() {
            workspace.run(&expert, &input, &ids, &routing, &stream)?;
            stream.synchronize()?;
            graph =
                Some(stream.capture(|| workspace.run(&expert, &input, &ids, &routing, &stream))?);
        }
        let start = cuda.event()?;
        let end = cuda.event()?;
        start.record(&stream)?;
        for _ in 0..50 {
            unsafe { graph.as_ref().unwrap().launch(&stream) }?;
        }
        end.record(&stream)?;
        let ms = end.elapsed_since(&start)? / 50.;
        let output: Vec<u8> = workspace
            .output()
            .download()?
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        ensure!(
            output == std::fs::read(base.join("output.bin"))?,
            "canonical MoE output differs from reference GPU: {}",
            case.name
        );
        reports.push(json!({"case":case.name,"values":output.len()/2,"bitwise_equal":true,"replays":50,
          "graph_ms":ms,"reference_graph_ms":case.reference_graph_ms,"native_over_reference":ms as f64/case.reference_graph_ms}));
    }
    Ok(
        json!({"layer":layer,"rank":rank,"source":source,"prepared":prepared,"gpu_prepared_sha256":hashes,
      "prepared_bytes_match_reference_gpu":true,"runtime_python_required":false,"cpu_numeric_oracle":false,"cases":reports}),
    )
}

/// Sparse attention directly from Rust-owned pages/workspace, including replay
/// with changed Q. Reference fixtures and expected values were produced on GPU.
pub fn attention(kernels: &Path, fixtures: &Path) -> Result<serde_json::Value> {
    use crate::attention::{Candidates, Execution, HEADS, PagedKv, WIDTH};
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        rows: usize,
        topk: usize,
        pages: usize,
        extra_topk: usize,
        extra_pages: usize,
        extra_page_tokens: usize,
        sink: bool,
        scale: f32,
        reference_graph_ms: f64,
    }
    #[derive(serde::Deserialize)]
    struct Cases {
        cases: Vec<Case>,
    }
    fn bf16_file(path: &Path) -> Result<Vec<bf16>> {
        let b = std::fs::read(path)?;
        ensure!(b.len() % 2 == 0, "BF16 fixture byte length");
        Ok(b.chunks_exact(2)
            .map(|x| bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())))
            .collect())
    }
    fn integers(path: &Path) -> Result<Vec<i32>> {
        let b = std::fs::read(path)?;
        ensure!(b.len() % 4 == 0, "index fixture byte length");
        Ok(b.chunks_exact(4)
            .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
            .collect())
    }
    let cases: Cases = serde_json::from_slice(&std::fs::read(fixtures.join("index.json"))?)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let mut reports = Vec::new();
    for c in cases.cases {
        ensure!(
            c.name
                .chars()
                .all(|v| v.is_ascii_alphanumeric() || v == '-' || v == '_'),
            "unsafe fixture name"
        );
        let p = fixtures.join(&c.name);
        let mut cache = PagedKv::new(&cuda, c.pages, 64)?;
        cache.upload(&std::fs::read(p.join("kv.bin"))?)?;
        let ids = Candidates::upload(
            &cuda,
            c.rows,
            c.topk,
            cache.capacity(),
            &integers(&p.join("ids.bin"))?,
            &integers(&p.join("lengths.bin"))?,
        )?;
        let extra = if c.extra_topk > 0 {
            let mut cache = PagedKv::new(&cuda, c.extra_pages, c.extra_page_tokens)?;
            cache.upload(&std::fs::read(p.join("extra_kv.bin"))?)?;
            let ids = Candidates::upload(
                &cuda,
                c.rows,
                c.extra_topk,
                cache.capacity(),
                &integers(&p.join("extra_ids.bin"))?,
                &integers(&p.join("extra_lengths.bin"))?,
            )?;
            Some((cache, ids))
        } else {
            None
        };
        let sink = if c.sink {
            let raw = std::fs::read(p.join("sink.bin"))?;
            ensure!(raw.len() == HEADS * 4, "sink fixture length");
            let data: Vec<_> = raw
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect();
            ensure!(data.iter().all(|x| x.is_finite()), "nonfinite fixture sink");
            let mut b = cuda.allocate::<f32>(HEADS)?;
            b.upload(&data)?;
            Some(b)
        } else {
            None
        };
        let mut q = cuda.allocate::<bf16>(c.rows * HEADS * WIDTH)?;
        q.upload(&bf16_file(&p.join("q.bin"))?)?;
        let decode = Execution::new(&cuda, kernels, c.rows, c.topk, c.extra_topk)?;
        let run = || {
            decode.run(
                &q,
                &cache,
                &ids,
                sink.as_ref(),
                extra.as_ref().map(|(a, b)| (a, b)),
                c.scale,
                &stream,
            )
        };
        run()?;
        stream.synchronize()?;
        let graph = stream.capture(run)?;
        let mut phases = Vec::new();
        for phase in ["initial", "changed"] {
            if phase == "changed" {
                q.upload(&bf16_file(&p.join("changed_q.bin"))?)?;
            }
            let a = cuda.event()?;
            let b = cuda.event()?;
            a.record(&stream)?;
            for _ in 0..50 {
                unsafe { graph.launch(&stream) }?;
            }
            b.record(&stream)?;
            let ms = b.elapsed_since(&a)? / 50.;
            let prefix = if phase == "initial" { "" } else { "changed_" };
            let actual: Vec<u8> = decode
                .output()
                .download()?
                .iter()
                .flat_map(|x| x.to_bits().to_le_bytes())
                .collect();
            let lse: Vec<u8> = decode
                .lse()
                .download()?
                .iter()
                .flat_map(|x| x.to_bits().to_le_bytes())
                .collect();
            let expected = std::fs::read(p.join(format!("{prefix}output.bin")))?;
            let expected_lse = std::fs::read(p.join(format!("{prefix}lse.bin")))?;
            let diff = actual
                .chunks_exact(2)
                .zip(expected.chunks_exact(2))
                .filter(|(a, b)| a != b)
                .count();
            let ldiff = lse
                .chunks_exact(4)
                .zip(expected_lse.chunks_exact(4))
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "attention {} {phase}: output {diff} unequal BF16, LSE {ldiff} unequal FP32",
                c.name
            );
            ensure!(
                actual.len() == expected.len()
                    && diff == 0
                    && lse.len() == expected_lse.len()
                    && ldiff == 0,
                "attention {} {phase} differs from reference GPU",
                c.name
            );
            phases.push(json!({"phase":phase,"bitwise_equal_output":true,"bitwise_equal_lse":true,"values":actual.len()/2,"replays":50,"graph_ms":ms}));
        }
        reports.push(json!({"name":c.name,"rows":c.rows,"topk":c.topk,"extra_page_tokens":c.extra_page_tokens,
            "workspace_bytes":decode.workspace_bytes(),"reference_graph_ms":c.reference_graph_ms,"phases":phases}));
    }
    Ok(
        json!({"runtime_python":false,"cpu_numeric_oracle":false,"native_page_tokens":64,"cases":reports}),
    )
}

pub fn kv(kernels: &Path, fixtures: &Path) -> Result<serde_json::Value> {
    use crate::{
        attention::PagedKv,
        kv::{RowLocations, Store},
    };
    use sha2::{Digest, Sha256};
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        rows: usize,
        stride: usize,
        pages: usize,
        page_tokens: usize,
        frequency_rows: usize,
        reference_graph_ms: f64,
    }
    #[derive(serde::Deserialize)]
    struct Cases {
        frequency_sha256: String,
        cases: Vec<Case>,
    }
    fn bf16s(p: &Path) -> Result<Vec<bf16>> {
        let b = std::fs::read(p)?;
        ensure!(b.len() % 2 == 0, "BF16 fixture length");
        Ok(b.chunks_exact(2)
            .map(|x| bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())))
            .collect())
    }
    fn ints(p: &Path) -> Result<Vec<i32>> {
        let b = std::fs::read(p)?;
        ensure!(b.len() % 4 == 0, "metadata fixture length");
        Ok(b.chunks_exact(4)
            .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
            .collect())
    }
    let cases: Cases = serde_json::from_slice(&std::fs::read(fixtures.join("index.json"))?)?;
    let fbytes = std::fs::read(fixtures.join("frequencies.bin"))?;
    ensure!(
        format!("{:x}", Sha256::digest(&fbytes)) == cases.frequency_sha256 && fbytes.len() % 4 == 0,
        "frequency fixture hash/length"
    );
    let fdata: Vec<_> = fbytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let mut frequencies = cuda.allocate::<f32>(fdata.len())?;
    frequencies.upload(&fdata)?;
    let store = Store::open(&cuda, kernels)?;
    let mut reports = Vec::new();
    ensure!(
        RowLocations::upload(&cuda, &[0, 1], &[1, 1], 64, 10).is_err(),
        "duplicate KV writes accepted"
    );
    ensure!(
        RowLocations::upload(&cuda, &[10], &[1], 64, 10).is_err(),
        "out of range RoPE accepted"
    );
    for c in cases.cases {
        ensure!(
            c.name
                .chars()
                .all(|v| v.is_ascii_alphanumeric() || v == '-'),
            "unsafe fixture name"
        );
        let p = fixtures.join(&c.name);
        let mut cache = PagedKv::new(&cuda, c.pages, c.page_tokens)?;
        cache.upload(&std::fs::read(p.join("initial_cache.bin"))?)?;
        let rows = RowLocations::upload(
            &cuda,
            &ints(&p.join("positions.bin"))?,
            &ints(&p.join("locations.bin"))?,
            cache.capacity(),
            c.frequency_rows,
        )?;
        let mut input = cuda.allocate::<bf16>(c.rows * c.stride)?;
        input.upload(&bf16s(&p.join("input.bin"))?)?;
        let mut weight = cuda.allocate::<bf16>(512)?;
        weight.upload(&bf16s(&p.join("weight.bin"))?)?;
        store.run(
            &input,
            c.stride,
            &weight,
            &frequencies,
            &rows,
            &mut cache,
            &stream,
        )?;
        stream.synchronize()?;
        let graph = stream.capture(|| {
            store.run(
                &input,
                c.stride,
                &weight,
                &frequencies,
                &rows,
                &mut cache,
                &stream,
            )
        })?;
        let mut phases = Vec::new();
        for phase in ["initial", "changed"] {
            if phase == "changed" {
                input.upload(&bf16s(&p.join("changed_input.bin"))?)?;
            }
            let a = cuda.event()?;
            let b = cuda.event()?;
            a.record(&stream)?;
            for _ in 0..50 {
                unsafe { graph.launch(&stream) }?;
            }
            b.record(&stream)?;
            let ms = b.elapsed_since(&a)? / 50.;
            let prefix = if phase == "initial" { "" } else { "changed_" };
            let output = cache.data().download()?;
            let expected = std::fs::read(p.join(format!("{prefix}output.bin")))?;
            let diff = output.iter().zip(&expected).filter(|(a, b)| a != b).count();
            println!("KV {} {phase}: {diff} unequal cache bytes", c.name);
            ensure!(
                output.len() == expected.len() && diff == 0,
                "KV {} {phase} differs from reference GPU",
                c.name
            );
            phases.push(json!({"phase":phase,"bytes":output.len(),"bitwise_equal":true,"graph_ms":ms,"replays":50}));
        }
        let other_stream = cuda.stream()?;
        ensure!(
            store
                .run(
                    &input,
                    c.stride,
                    &weight,
                    &frequencies,
                    &rows,
                    &mut cache,
                    &other_stream
                )
                .is_err(),
            "cross-stream cache race allowed"
        );
        reports
            .push(json!({"name":c.name,"reference_graph_ms":c.reference_graph_ms,"phases":phases}));
    }
    Ok(
        json!({"runtime_python":false,"cpu_numeric_oracle":false,"frequency_sha256":cases.frequency_sha256,"metadata_rejections_qualified":true,"cases":reports}),
    )
}

pub fn prelude(kernels: &Path, fixtures: &Path) -> Result<serde_json::Value> {
    use crate::{
        prelude::{Native, NormMxfp8, Positions},
        triton_ops::Operator,
    };
    use sha2::{Digest, Sha256};
    #[derive(serde::Deserialize)]
    struct Case {
        op: String,
        name: String,
        rows: usize,
        width: usize,
        stride: usize,
        reference_graph_ms: f64,
    }
    #[derive(serde::Deserialize)]
    struct Cases {
        frequency_rows: usize,
        frequency_sha256: String,
        cases: Vec<Case>,
    }
    fn bf16s(p: &Path) -> Result<Vec<bf16>> {
        let b = std::fs::read(p)?;
        ensure!(b.len() % 2 == 0, "BF16 fixture length");
        Ok(b.chunks_exact(2)
            .map(|x| bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())))
            .collect())
    }
    fn raw(b: &crate::cuda::Buffer<bf16>) -> Result<Vec<u8>> {
        Ok(b.download()?
            .iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect())
    }
    let c: Cases = serde_json::from_slice(&std::fs::read(fixtures.join("index.json"))?)?;
    let fbytes = std::fs::read(fixtures.join("frequencies.bin"))?;
    ensure!(
        fbytes.len() == c.frequency_rows * 256
            && format!("{:x}", Sha256::digest(&fbytes)) == c.frequency_sha256,
        "frequency fixture hash/geometry"
    );
    let fdata: Vec<_> = fbytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let mut frequencies = cuda.allocate::<f32>(fdata.len())?;
    frequencies.upload(&fdata)?;
    let native = Native::open(&cuda, kernels)?;
    let mut reports = Vec::new();
    for case in c.cases {
        ensure!(
            case.name
                .chars()
                .all(|x| x.is_ascii_alphanumeric() || x == '-'),
            "unsafe prelude fixture name"
        );
        let p = fixtures.join(&case.name);
        let data = bf16s(&p.join("input.bin"))?;
        let mut input = cuda.allocate::<bf16>(data.len())?;
        input.upload(&data)?;
        let mut output = cuda.allocate::<bf16>(case.rows * case.width)?;
        output.zero()?;
        if case.op == "rms" || case.op == "norm_mxfp8" {
            let mut weight = cuda.allocate::<bf16>(case.width)?;
            weight.upload(&bf16s(&p.join("weight.bin"))?)?;
            if case.op == "rms" {
                native.rms(
                    &input,
                    &weight,
                    &mut output,
                    case.rows,
                    case.stride,
                    &stream,
                )?;
                stream.synchronize()?;
                ensure!(
                    raw(&output)? == std::fs::read(p.join("output.bin"))?,
                    "RMS differs from reference GPU: {}",
                    case.name
                );
                let graph = stream.capture(|| {
                    native.rms(
                        &input,
                        &weight,
                        &mut output,
                        case.rows,
                        case.stride,
                        &stream,
                    )
                })?;
                let a = cuda.event()?;
                let b = cuda.event()?;
                a.record(&stream)?;
                for _ in 0..50 {
                    unsafe { graph.launch(&stream) }?;
                }
                b.record(&stream)?;
                let ms = b.elapsed_since(&a)? / 50.;
                ensure!(
                    raw(&output)? == std::fs::read(p.join("output.bin"))?,
                    "RMS replay differs"
                );
                reports.push(json!({"case":case.name,"op":case.op,"values":output.len(),"bitwise_equal":true,"graph_ms":ms,"reference_graph_ms":case.reference_graph_ms}));
            } else {
                let norm = NormMxfp8::new(&cuda, kernels, case.rows)?;
                norm.run(&input, &weight, &stream)?;
                stream.synchronize()?;
                let graph = stream.capture(|| norm.run(&input, &weight, &stream))?;
                let a = cuda.event()?;
                let b = cuda.event()?;
                a.record(&stream)?;
                for _ in 0..50 {
                    unsafe { graph.launch(&stream) }?;
                }
                b.record(&stream)?;
                let ms = b.elapsed_since(&a)? / 50.;
                ensure!(
                    raw(norm.output())? == std::fs::read(p.join("output.bin"))?,
                    "fused normalized BF16 differs: {}",
                    case.name
                );
                ensure!(
                    norm.values().download()? == std::fs::read(p.join("values.bin"))?,
                    "fused quantized values differ: {}",
                    case.name
                );
                ensure!(
                    norm.scales().download()? == std::fs::read(p.join("scales.bin"))?,
                    "fused quantized scales differ: {}",
                    case.name
                );
                reports.push(json!({"case":case.name,"op":case.op,"bf16_and_quant_and_scales_bitwise_equal":true,"graph_ms":ms,"reference_graph_ms":case.reference_graph_ms}));
            }
        } else if case.op == "rope" {
            let rb = std::fs::read(p.join("positions.bin"))?;
            ensure!(rb.len() == case.rows * 4, "position fixture bytes");
            let rv: Vec<_> = rb
                .chunks_exact(4)
                .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
                .collect();
            let positions = Positions::upload(&cuda, &rv, c.frequency_rows)?;
            let mut initial = cuda.allocate::<bf16>(data.len())?;
            initial.upload(&data)?;
            for inverse in [false, true] {
                input.upload(&data)?;
                native.rope(
                    &mut input,
                    case.rows,
                    &frequencies,
                    &positions,
                    inverse,
                    &stream,
                )?;
                stream.synchronize()?;
                let name = if inverse {
                    "inverse.bin"
                } else {
                    "forward.bin"
                };
                let expected = std::fs::read(p.join(name))?;
                ensure!(
                    raw(&input)? == expected,
                    "RoPE differs from reference GPU: {} inverse={inverse}",
                    case.name
                );
                let graph = stream.capture(|| {
                    unsafe {
                        cuda.copy_device(input.ptr(), initial.ptr(), initial.bytes(), &stream)
                    }?;
                    native.rope(
                        &mut input,
                        case.rows,
                        &frequencies,
                        &positions,
                        inverse,
                        &stream,
                    )
                })?;
                let a = cuda.event()?;
                let b = cuda.event()?;
                a.record(&stream)?;
                for _ in 0..50 {
                    unsafe { graph.launch(&stream) }?;
                }
                b.record(&stream)?;
                let ms = b.elapsed_since(&a)? / 50.;
                ensure!(raw(&input)? == expected, "RoPE replay differs");
                reports.push(json!({"case":case.name,"op":case.op,"inverse":inverse,"bitwise_equal":true,"graph_ms_including_reset_copy":ms,"reference_forward_ms_including_reset_copy":case.reference_graph_ms}));
            }
            if case.rows <= 8 || case.rows >= 4096 {
                let name = if case.rows <= 8 {
                    "q_rope_small"
                } else {
                    "q_rope_prefill"
                };
                let operator = Operator::load(&cuda, kernels, name)?;
                let mut qout = cuda.allocate::<bf16>(data.len())?;
                qout.zero()?;
                let mut args = vec![initial.ptr(), qout.ptr(), frequencies.ptr()];
                // The validated metadata is independently uploaded for the AOT raw ABI.
                let mut pos = cuda.allocate::<i32>(rv.len())?;
                pos.upload(&rv)?;
                args.push(pos.ptr());
                if case.rows >= 4096 {
                    args.push(case.rows as u64);
                }
                unsafe { operator.launch(case.rows as u32, &args, &stream) }?;
                stream.synchronize()?;
                ensure!(
                    raw(&qout)? == std::fs::read(p.join("forward.bin"))?,
                    "AOT Q RoPE differs: {}",
                    case.name
                );
                reports
                    .push(json!({"case":case.name,"op":"q_rope_aot_store","bitwise_equal":true}));
            }
        } else {
            anyhow::bail!("unknown prelude case");
        }
        println!("prelude {} passed", case.name);
    }
    Ok(json!({"cpu_numeric_oracle":false,"runtime_python":false,"cases":reports}))
}

pub fn frequency(core: &Path, fixtures: &Path) -> Result<serde_json::Value> {
    use sha2::{Digest, Sha256};
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        rows: usize,
        compressed: bool,
        sha256: String,
    }
    #[derive(serde::Deserialize)]
    struct Cases {
        cases: Vec<Case>,
    }
    let cases: Cases = serde_json::from_slice(&std::fs::read(fixtures.join("index.json"))?)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let module = cuda.module(&std::fs::read(core)?)?;
    let stream = cuda.stream()?;
    let mut reports = Vec::new();
    for case in cases.cases {
        ensure!(
            ["compressed", "swa"].contains(&case.name.as_str()),
            "unknown frequency fixture"
        );
        let start = std::time::Instant::now();
        let table = crate::prelude::FrequencyTable::new(
            &cuda,
            &module,
            &stream,
            case.rows,
            case.compressed,
        )?;
        let seconds = start.elapsed().as_secs_f64();
        let raw: Vec<_> = table
            .values()
            .download()?
            .iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect();
        let expected = std::fs::read(fixtures.join(format!("{}.bin", case.name)))?;
        ensure!(
            format!("{:x}", Sha256::digest(&expected)) == case.sha256,
            "reference frequency hash mismatch"
        );
        let diff = raw
            .chunks_exact(4)
            .zip(expected.chunks_exact(4))
            .filter(|(a, b)| a != b)
            .count();
        let mut max = 0f32;
        for (a, b) in raw.chunks_exact(4).zip(expected.chunks_exact(4)) {
            let a = f32::from_le_bytes(a.try_into().unwrap());
            let b = f32::from_le_bytes(b.try_into().unwrap());
            max = max.max((a - b).abs());
        }
        println!(
            "frequency {}: {diff} unequal FP32 values, max_abs={max}",
            case.name
        );
        // Persist diagnostic differences even when the strict equality gate fails.
        let r = json!({"case":case.name,"values":raw.len()/4,"different_values":diff,"max_abs":max,"generation_seconds":seconds,
            "native_sha256":format!("{:x}",Sha256::digest(&raw)),"reference_sha256":case.sha256,"base_frequencies":table.base().download()?});
        reports.push(r);
    }
    Ok(json!({"cpu_numeric_oracle":false,"runtime_python":false,"cases":reports}))
}

pub fn attention_checkpoint(
    config: &Path,
    manifest: &Path,
    model: &Path,
    core: &Path,
    kernels: &Path,
    fixtures: &Path,
    mode: crate::layer_attention::InputMode,
) -> Result<serde_json::Value> {
    use crate::{
        attention::{Candidates, PagedKv},
        checkpoint::{Checkpoint, Dtype},
        config::ModelConfig,
        kv::RowLocations,
        layer_attention::Layer,
        layout::RankPlan,
        prelude::Positions,
        weights::Weights,
    };
    use std::rc::Rc;
    #[derive(serde::Deserialize)]
    struct Case {
        phase: String,
        reference_graph_ms: f64,
    }
    #[derive(serde::Deserialize)]
    struct Cases {
        layer: usize,
        rank: usize,
        rows: usize,
        frequency_rows: usize,
        original_woa_equals_recipe_twin: bool,
        cases: Vec<Case>,
    }
    fn ints(p: &Path) -> Result<Vec<i32>> {
        let b = std::fs::read(p)?;
        ensure!(b.len() % 4 == 0, "integer fixture bytes");
        Ok(b.chunks_exact(4)
            .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
            .collect())
    }
    fn bf16s(p: &Path) -> Result<Vec<bf16>> {
        let b = std::fs::read(p)?;
        ensure!(b.len() % 2 == 0, "BF16 fixture bytes");
        Ok(b.chunks_exact(2)
            .map(|x| bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())))
            .collect())
    }
    let cases: Cases = serde_json::from_slice(&std::fs::read(fixtures.join("index.json"))?)?;
    ensure!(
        (cases.layer, cases.rank, cases.rows) == (0, 0, 6) && cases.original_woa_equals_recipe_twin,
        "attention oracle scope mismatch"
    );
    let cfg = ModelConfig::parse(&std::fs::read(config)?)?;
    let cp = Checkpoint::from_manifest(manifest)?;
    let plan = RankPlan::build(&cp, &cfg, 0, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let module = cuda.module(&std::fs::read(core)?)?;
    let stream = cuda.stream()?;
    let weights = Rc::new(Weights::load(&cuda, model, &cp, &plan, "layers.0.attn.")?);
    let source = weights.qualify(&cp, model, &plan)?;
    let sink = cuda.allocate::<f32>(16)?;
    let sn = "layers.0.attn.attn_sink";
    let sv = weights.view(sn)?;
    ensure!(
        sv.dtype == Dtype::F32 && sv.shape == [16] && sv.bytes == 64,
        "canonical TP4 sink shape"
    );
    unsafe { cuda.copy_device(sink.ptr(), weights.ptr(sn)?, 64, &stream) }?;
    stream.synchronize()?;
    let sr: Vec<_> = sink
        .download()?
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect();
    ensure!(
        sr == std::fs::read(fixtures.join("sink.bin"))?,
        "loaded attention sink differs"
    );
    let mut layer = Layer::new(
        &cuda,
        &module,
        &stream,
        weights,
        0,
        kernels,
        Rc::new(crate::prelude::FrequencyTable::new(
            &cuda,
            &module,
            &stream,
            cases.frequency_rows,
            false,
        )?),
        mode,
    )?;
    let mut cache = PagedKv::new(&cuda, 4, 64)?;
    cache.upload(&std::fs::read(fixtures.join("initial_cache.bin"))?)?;
    let pos = ints(&fixtures.join("positions.bin"))?;
    let loc = ints(&fixtures.join("locations.bin"))?;
    let positions = Positions::upload(&cuda, &pos, cases.frequency_rows)?;
    let locations =
        RowLocations::upload(&cuda, &pos, &loc, cache.capacity(), cases.frequency_rows)?;
    let ids = Candidates::upload(
        &cuda,
        6,
        128,
        cache.capacity(),
        &ints(&fixtures.join("ids.bin"))?,
        &ints(&fixtures.join("lengths.bin"))?,
    )?;
    let mut input = cuda.allocate::<bf16>(6 * 5120)?;
    let mut graph = None;
    let mut reports = Vec::new();
    for case in cases.cases {
        ensure!(
            ["initial", "changed"].contains(&case.phase.as_str()),
            "unknown attention phase"
        );
        let p = fixtures.join(&case.phase);
        input.upload(&bf16s(&p.join("input.bin"))?)?;
        if graph.is_none() {
            layer.run(
                &input, &positions, &locations, &mut cache, &ids, &sink, &stream,
            )?;
            stream.synchronize()?;
            graph = Some(stream.capture(|| {
                layer.run(
                    &input, &positions, &locations, &mut cache, &ids, &sink, &stream,
                )
            })?);
        }
        // Let clock/DRAM state settle before measuring the small fixed graph.
        // Replaying this controlled fixture overwrites the same current KV rows.
        for _ in 0..1000 {
            unsafe { graph.as_ref().unwrap().launch(&stream) }?;
        }
        stream.synchronize()?;
        let a = cuda.event()?;
        let b = cuda.event()?;
        a.record(&stream)?;
        for _ in 0..50 {
            unsafe { graph.as_ref().unwrap().launch(&stream) }?;
        }
        b.record(&stream)?;
        let ms = b.elapsed_since(&a)? / 50.;
        let mut taps = Vec::new();
        for (name, ptr, bytes, stride) in layer.taps() {
            let actual = if stride == 0 {
                unsafe { cuda.read_device(ptr, bytes) }?
            } else {
                let mut data = Vec::with_capacity(bytes);
                for row in 0..6 {
                    data.extend(unsafe {
                        cuda.read_device(ptr + (row * stride) as u64, bytes / 6)
                    }?);
                }
                data
            };
            let expected = std::fs::read(p.join(format!("{name}.bin")))?;
            let diff = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            println!("attention {} {name}: {diff} unequal bytes", case.phase);
            if diff != 0 || actual.len() != expected.len() {
                std::fs::write(
                    fixtures.join(format!("debug-native-{}-{name}.bin", case.phase)),
                    &actual,
                )?;
            }
            ensure!(
                actual.len() == expected.len() && diff == 0,
                "canonical attention {} {name} differs from GPU reference",
                case.phase
            );
            taps.push(json!({"name":name,"bytes":bytes,"bitwise_equal":true}));
        }
        ensure!(
            cache.data().download()? == std::fs::read(p.join("cache.bin"))?,
            "canonical attention cache differs"
        );
        reports.push(json!({"phase":case.phase,"graph_ms":ms,"reference_graph_ms":case.reference_graph_ms,"replays":50,"warmup_replays":1000,"taps":taps,"cache_bitwise_equal":true}));
    }
    Ok(
        json!({"input_mode":format!("{mode:?}"),"layer":0,"rank":0,"rows":6,"source":source,"cpu_numeric_oracle":false,"runtime_python":false,
        "scope":"independent single-rank attention chain, before collective/mHC; controlled past KV and metadata; not full-model or retained-state qualification","phases":reports}),
    )
}

/// Same-context, same-atlas alternating graph timings. Avoids process startup,
/// different weight addresses and short GPU warmup as strategy confounders.
pub fn attention_compare(
    config: &Path,
    manifest: &Path,
    model: &Path,
    core: &Path,
    kernels: &Path,
    fixtures: &Path,
) -> Result<serde_json::Value> {
    use crate::{
        attention::{Candidates, PagedKv},
        checkpoint::Checkpoint,
        config::ModelConfig,
        cuda::Graph,
        kv::RowLocations,
        layer_attention::{InputMode, Layer},
        layout::RankPlan,
        prelude::{FrequencyTable, Positions},
        weights::Weights,
    };
    use std::rc::Rc;
    struct State {
        graph: Graph,
        layer: Layer,
        cache: PagedKv,
        mode: InputMode,
    }
    fn ints(p: &Path) -> Result<Vec<i32>> {
        let b = std::fs::read(p)?;
        ensure!(b.len() % 4 == 0, "metadata bytes");
        Ok(b.chunks_exact(4)
            .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
            .collect())
    }
    fn bf16s(p: &Path) -> Result<Vec<bf16>> {
        let b = std::fs::read(p)?;
        ensure!(b.len() % 2 == 0, "BF16 bytes");
        Ok(b.chunks_exact(2)
            .map(|x| bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())))
            .collect())
    }
    let cfg = ModelConfig::parse(&std::fs::read(config)?)?;
    let cp = Checkpoint::from_manifest(manifest)?;
    let plan = RankPlan::build(&cp, &cfg, 0, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let module = cuda.module(&std::fs::read(core)?)?;
    let weights = Rc::new(Weights::load(&cuda, model, &cp, &plan, "layers.0.attn.")?);
    let source = weights.qualify(&cp, model, &plan)?;
    let freq = Rc::new(FrequencyTable::new(&cuda, &module, &stream, 131072, false)?);
    let sink = cuda.allocate::<f32>(16)?;
    ensure!(
        weights.view("layers.0.attn.attn_sink")?.bytes == 64,
        "sink bytes"
    );
    unsafe {
        cuda.copy_device(
            sink.ptr(),
            weights.ptr("layers.0.attn.attn_sink")?,
            64,
            &stream,
        )
    }?;
    let p = ints(&fixtures.join("positions.bin"))?;
    let loc = ints(&fixtures.join("locations.bin"))?;
    let positions = Positions::upload(&cuda, &p, 131072)?;
    let locations = RowLocations::upload(&cuda, &p, &loc, 256, 131072)?;
    let ids = Candidates::upload(
        &cuda,
        6,
        128,
        256,
        &ints(&fixtures.join("ids.bin"))?,
        &ints(&fixtures.join("lengths.bin"))?,
    )?;
    let mut input = cuda.allocate::<bf16>(6 * 5120)?;
    input.upload(&bf16s(&fixtures.join("initial/input.bin"))?)?;
    let mut states = Vec::new();
    for mode in [InputMode::Separate, InputMode::Shared, InputMode::Fused] {
        let mut layer = Layer::new(
            &cuda,
            &module,
            &stream,
            weights.clone(),
            0,
            kernels,
            freq.clone(),
            mode,
        )?;
        let mut cache = PagedKv::new(&cuda, 4, 64)?;
        cache.upload(&std::fs::read(fixtures.join("initial_cache.bin"))?)?;
        layer.run(
            &input, &positions, &locations, &mut cache, &ids, &sink, &stream,
        )?;
        stream.synchronize()?;
        let graph = stream.capture(|| {
            layer.run(
                &input, &positions, &locations, &mut cache, &ids, &sink, &stream,
            )
        })?;
        states.push(State {
            graph,
            layer,
            cache,
            mode,
        });
    }
    let mut timings = Vec::new();
    let mut quality = Vec::new();
    for phase in ["initial", "changed"] {
        let path = fixtures.join(phase);
        input.upload(&bf16s(&path.join("input.bin"))?)?;
        for s in &states {
            for _ in 0..1000 {
                unsafe { s.graph.launch(&stream) }?;
            }
        }
        stream.synchronize()?;
        for trial in 0..6 {
            for slot in 0..3 {
                let s = &states[(trial + slot) % 3];
                let a = cuda.event()?;
                let b = cuda.event()?;
                a.record(&stream)?;
                for _ in 0..200 {
                    unsafe { s.graph.launch(&stream) }?;
                }
                b.record(&stream)?;
                let ms = b.elapsed_since(&a)? / 200.;
                timings.push(json!({"phase":phase,"trial":trial,"mode":format!("{:?}",s.mode),"milliseconds":ms}));
            }
        }
        for s in &states {
            for (name, ptr, bytes, stride) in s.layer.taps() {
                let actual = if stride == 0 {
                    unsafe { cuda.read_device(ptr, bytes) }?
                } else {
                    let mut v = Vec::with_capacity(bytes);
                    for r in 0..6 {
                        v.extend(unsafe {
                            cuda.read_device(ptr + (r * stride) as u64, bytes / 6)
                        }?);
                    }
                    v
                };
                ensure!(
                    actual == std::fs::read(path.join(format!("{name}.bin")))?,
                    "same-context {:?} {phase} {name} differs",
                    s.mode
                );
            }
            ensure!(
                s.cache.data().download()? == std::fs::read(path.join("cache.bin"))?,
                "same-context cache differs"
            );
            quality.push(json!({"mode":format!("{:?}",s.mode),"phase":phase,"all_twelve_taps_and_cache_bitwise_equal":true}));
        }
    }
    let mut medians = serde_json::Map::new();
    for mode in ["Separate", "Shared", "Fused"] {
        let mut phases = serde_json::Map::new();
        for phase in ["initial", "changed"] {
            let mut values: Vec<_> = timings
                .iter()
                .filter(|v| v["mode"] == mode && v["phase"] == phase)
                .map(|v| v["milliseconds"].as_f64().unwrap())
                .collect();
            values.sort_by(f64::total_cmp);
            phases.insert(phase.into(), json!((values[2] + values[3]) / 2.));
        }
        medians.insert(mode.into(), json!(phases));
    }
    Ok(
        json!({"source":source,"same_context":true,"same_canonical_weight_atlas":true,"shared_rotary_table":true,"warmup_replays_per_phase_strategy":1000,
        "replays_per_timing":200,"alternating_trials_per_phase":6,"median_ms":medians,"timings":timings,"quality":quality,
        "scope":"Single-rank layer-0 six-row controlled attention, before collective/mHC. No claim about full qualified SGLang performance."}),
    )
}

/// Complete canonical per-rank FFN chain. All intermediate buffers are compared
/// with the effective GPU oracle, including image routing and masked rows.
pub fn ffn_checkpoint(
    config: &Path,
    manifest: &Path,
    root: &Path,
    core: &Path,
    kernels: &Path,
    fixtures: &Path,
) -> Result<serde_json::Value> {
    use crate::{
        checkpoint::Checkpoint,
        config::ModelConfig,
        expert_packing,
        ffn::{LayerFfn, TokenRows},
        layout::RankPlan,
        moe::Workspace,
        weights::Weights,
    };
    use std::{collections::BTreeMap, rc::Rc};
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        reference_graph_ms: f64,
    }
    #[derive(serde::Deserialize)]
    struct Cases {
        layer: usize,
        rank: usize,
        rows: usize,
        cases: Vec<Case>,
        gpu_prepared_sha256: BTreeMap<String, String>,
    }
    let cases: Cases = serde_json::from_slice(&std::fs::read(fixtures.join("index.json"))?)?;
    ensure!(
        (cases.layer, cases.rank) == (0, 0)
            && [6, 128].contains(&cases.rows)
            && cases.cases.len() == 4,
        "FFN oracle geometry mismatch"
    );
    let cfg = ModelConfig::parse(&std::fs::read(config)?)?;
    let cp = Checkpoint::from_manifest(manifest)?;
    let plan = RankPlan::build(&cp, &cfg, 0, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let module = cuda.module(&std::fs::read(core)?)?;
    let mut weights = Weights::load(&cuda, root, &cp, &plan, "layers.0.ffn.")?;
    let source = weights.qualify(&cp, root, &plan)?;
    let prepared = expert_packing::prepare_layer(&cuda, &module, &mut weights, 0)?;
    ensure!(
        expert_packing::prepared_hashes(&weights, &prepared)? == cases.gpu_prepared_sha256,
        "FFN GPU prepared expert bytes differ"
    );
    let rows = cases.rows;
    let blas = if rows > 16 {
        Some(Rc::new(crate::blas::Blas::new(&cuda)?))
    } else {
        None
    };
    let ffn = LayerFfn::new_rows(
        &cuda,
        &module,
        &stream,
        Rc::new(weights),
        prepared,
        kernels,
        rows,
        blas,
    )?;
    let workspace = Workspace::new(&cuda, &kernels.join(format!("moe-runtime-m{rows}-v0")))?;
    let mut input = cuda.allocate::<bf16>(rows * 5120)?;
    let mut tokens = TokenRows::new(&cuda, &vec![0; rows], rows)?;
    let warmup = if rows == 6 { 1000 } else { 50 };
    let replays = if rows == 6 { 200 } else { 50 };
    let mut graph = None;
    let mut reports = Vec::new();
    let mut all_equal = true;
    for case in cases.cases {
        ensure!(
            ["initial", "changed", "zero", "padded"].contains(&case.name.as_str()),
            "unknown FFN oracle case"
        );
        let dir = fixtures.join(&case.name);
        let b = std::fs::read(dir.join("input.bin"))?;
        ensure!(b.len() == input.bytes(), "FFN fixture input bytes");
        input.upload(
            &b.chunks_exact(2)
                .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())))
                .collect::<Vec<_>>(),
        )?;
        let tb = std::fs::read(dir.join("tokens.bin"))?;
        let lb = std::fs::read(dir.join("live.bin"))?;
        ensure!(
            tb.len() == rows * 8 && lb.len() == 4,
            "FFN token metadata bytes"
        );
        let ids: Vec<_> = tb
            .chunks_exact(8)
            .map(|v| i64::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let live = i32::from_le_bytes(lb.try_into().unwrap());
        ensure!((0..=rows as i32).contains(&live), "FFN live rows domain");
        tokens.update(&ids, live as usize)?;
        if graph.is_none() {
            ffn.run(&workspace, &input, &tokens, &stream)?;
            stream.synchronize()?;
            graph = Some(stream.capture(|| ffn.run(&workspace, &input, &tokens, &stream))?);
        }
        for _ in 0..warmup {
            unsafe { graph.as_ref().unwrap().launch(&stream) }?;
        }
        let mut times = Vec::new();
        for _ in 0..6 {
            let start = cuda.event()?;
            let end = cuda.event()?;
            start.record(&stream)?;
            for _ in 0..replays {
                unsafe { graph.as_ref().unwrap().launch(&stream) }?;
            }
            end.record(&stream)?;
            times.push(end.elapsed_since(&start)? / replays as f32);
        }
        stream.synchronize()?;
        let mut taps = Vec::new();
        for (name, actual, size) in [
            (
                "logits",
                ffn.router()
                    .logits()
                    .download()?
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
                4,
            ),
            (
                "ids",
                ffn.router()
                    .ids()
                    .download()?
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                4,
            ),
            (
                "routing",
                ffn.router()
                    .routing()
                    .download()?
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                4,
            ),
            (
                "shared_up",
                ffn.shared_up()
                    .download()?
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect(),
                2,
            ),
            (
                "activation",
                ffn.activation()
                    .download()?
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect(),
                2,
            ),
            (
                "shared_down",
                ffn.shared_down()
                    .download()?
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect(),
                2,
            ),
            (
                "routed",
                workspace
                    .output()
                    .download()?
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect(),
                2,
            ),
            (
                "output",
                ffn.output()
                    .download()?
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect(),
                2,
            ),
        ] {
            let expected = std::fs::read(dir.join(format!("{name}.bin")))?;
            ensure!(actual.len() == expected.len(), "FFN {name} fixture length");
            let differences = actual
                .chunks_exact(size)
                .zip(expected.chunks_exact(size))
                .filter(|(a, b)| a != b)
                .count();
            all_equal &= differences == 0;
            let max_abs = if name == "ids" {
                serde_json::Value::Null
            } else {
                let decode = |x: &[u8]| {
                    if size == 2 {
                        bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())).to_f32()
                    } else {
                        f32::from_le_bytes(x.try_into().unwrap())
                    }
                };
                let max = actual
                    .chunks_exact(size)
                    .zip(expected.chunks_exact(size))
                    .map(|(a, b)| (decode(a) - decode(b)).abs())
                    .fold(0f32, f32::max);
                json!(max)
            };
            taps.push(json!({"name":name,"values":actual.len()/size,"different_values":differences,"bitwise_equal":differences==0,"max_abs":max_abs}));
        }
        let mut sorted = times.clone();
        sorted.sort_by(f32::total_cmp);
        let median = (sorted[2] + sorted[3]) / 2.;
        let report = json!({"case":case.name,"live_rows":live,"taps":taps,"graph_ms":median,"trials_ms":times,
            "reference_graph_ms":case.reference_graph_ms,"native_over_reference":median as f64/case.reference_graph_ms});
        println!("{}", report);
        reports.push(report);
    }
    Ok(
        json!({"scope":"TP4 rank-local canonical FFN before collective/mHC; not complete layer/model",
        "source":source,"rows":rows,"all_bitwise_equal":all_equal,"cases":reports,"warmup_replays":warmup,"trial_replays":replays,"trials_per_case":6}),
    )
}

/// Canonical mHC state/rounding qualification with controlled reduced sublayers.
/// This deliberately does not claim attention/collective/FFN layer integration.
pub fn mhc_checkpoint(
    config: &Path,
    manifest: &Path,
    root: &Path,
    kernels: &Path,
    fixtures: &Path,
) -> Result<serde_json::Value> {
    use crate::{
        checkpoint::Checkpoint, config::ModelConfig, layout::RankPlan, mhc::Boundary,
        weights::Weights,
    };
    use std::rc::Rc;
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        resumed: bool,
        reference_graph_ms: f64,
    }
    #[derive(serde::Deserialize)]
    struct Cases {
        layer: usize,
        rows: usize,
        cases: Vec<Case>,
    }
    let cases: Cases = serde_json::from_slice(&std::fs::read(fixtures.join("index.json"))?)?;
    ensure!(
        cases.layer == 0 && (1..=384).contains(&cases.rows) && cases.cases.len() == 4,
        "mHC fixture geometry"
    );
    let cfg = ModelConfig::parse(&std::fs::read(config)?)?;
    let cp = Checkpoint::from_manifest(manifest)?;
    let plan = RankPlan::build(&cp, &cfg, 0, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let weights = Rc::new(Weights::load_prefixes(
        &cuda,
        root,
        &cp,
        &plan,
        &["layers.0.hc_", "layers.0.attn_norm.", "layers.0.ffn_norm."],
    )?);
    let source = weights.qualify(&cp, root, &plan)?;
    let mut attn = Boundary::new(
        &cuda,
        &stream,
        weights.clone(),
        0,
        "attn",
        cases.rows,
        kernels,
    )?;
    let mut ffn = Boundary::new(&cuda, &stream, weights, 0, "ffn", cases.rows, kernels)?;
    let mut residual = cuda.allocate::<bf16>(cases.rows * 20480)?;
    let mut a = cuda.allocate::<bf16>(cases.rows * 5120)?;
    let mut f = cuda.allocate::<bf16>(cases.rows * 5120)?;
    let mut pre = cuda.allocate::<f32>(cases.rows * 4)?;
    let mut graphs = std::collections::BTreeMap::new();
    let mut reports = Vec::new();
    let mut all_equal = true;
    for case in cases.cases {
        ensure!(
            ["initial", "changed", "resumed", "resumed_changed"].contains(&case.name.as_str()),
            "mHC case name"
        );
        let dir = fixtures.join(&case.name);
        for (name, buffer) in [
            ("residual", &mut residual),
            ("attn_sublayer", &mut a),
            ("ffn_sublayer", &mut f),
        ] {
            let b = std::fs::read(dir.join(format!("{name}.bin")))?;
            ensure!(b.len() == buffer.bytes(), "mHC BF16 fixture size");
            buffer.upload(
                &b.chunks_exact(2)
                    .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
                    .collect::<Vec<_>>(),
            )?;
        }
        let b = std::fs::read(dir.join("previous_pre.bin"))?;
        ensure!(b.len() == pre.bytes(), "mHC pre fixture size");
        pre.upload(
            &b.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>(),
        )?;
        let mut run = || -> Result<()> {
            attn.prepare(
                &residual,
                if case.resumed { Some(&pre) } else { None },
                &stream,
            )?;
            attn.finish(&a, &residual, &stream)?;
            ffn.prepare(attn.output(), Some(attn.pre()), &stream)?;
            ffn.finish(&f, attn.output(), &stream)
        };
        if !graphs.contains_key(&case.resumed) {
            run()?;
            stream.synchronize()?;
            graphs.insert(case.resumed, stream.capture(&mut run)?);
        }
        let graph = &graphs[&case.resumed];
        for _ in 0..1000 {
            unsafe { graph.launch(&stream) }?;
        }
        let mut times = Vec::new();
        for _ in 0..6 {
            let start = cuda.event()?;
            let end = cuda.event()?;
            start.record(&stream)?;
            for _ in 0..200 {
                unsafe { graph.launch(&stream) }?;
            }
            end.record(&stream)?;
            times.push(end.elapsed_since(&start)? / 200.);
        }
        stream.synchronize()?;
        let mut taps = Vec::new();
        let mut compare = |name: String, actual: Vec<u8>, size: usize| -> Result<()> {
            let expected = std::fs::read(dir.join(format!("{name}.bin")))?;
            ensure!(actual.len() == expected.len(), "mHC fixture tap size");
            let different = actual
                .chunks_exact(size)
                .zip(expected.chunks_exact(size))
                .filter(|(a, b)| a != b)
                .count();
            all_equal &= different == 0;
            let decode = |b: &[u8]| {
                if size == 2 {
                    bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()
                } else {
                    f32::from_le_bytes(b.try_into().unwrap())
                }
            };
            let max = actual
                .chunks_exact(size)
                .zip(expected.chunks_exact(size))
                .map(|(a, b)| (decode(a) - decode(b)).abs())
                .fold(0f32, f32::max);
            taps.push(json!({"name":name,"values":actual.len()/size,"different_values":different,"bitwise_equal":different==0,"max_abs":max}));
            Ok(())
        };
        for (branch, boundary) in [("attn", &attn), ("ffn", &ffn)] {
            for (suffix, buffer) in [
                ("partial_mix", boundary.partial_mix()),
                ("partial_square", boundary.partial_square()),
                ("pre", boundary.pre()),
                ("post", boundary.post()),
                ("comb", boundary.comb()),
            ] {
                compare(
                    format!("{branch}_{suffix}"),
                    buffer
                        .download()?
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                    4,
                )?;
            }
            compare(
                format!("{branch}_input"),
                boundary
                    .input()
                    .download()?
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect(),
                2,
            )?;
        }
        compare(
            "attn_output".into(),
            attn.output()
                .download()?
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
            2,
        )?;
        compare(
            "output".into(),
            ffn.output()
                .download()?
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
            2,
        )?;
        let mut sorted = times.clone();
        sorted.sort_by(f32::total_cmp);
        let median = (sorted[2] + sorted[3]) / 2.;
        let report = json!({"case":case.name,"resumed":case.resumed,"taps":taps,"graph_ms":median,"trials_ms":times,
            "reference_graph_ms":case.reference_graph_ms,"native_over_reference":median as f64/case.reference_graph_ms});
        println!("{}", report);
        reports.push(report);
    }
    Ok(
        json!({"scope":"canonical mHC attention/FFN boundaries with controlled reduced BF16 sublayers; not complete layer",
        "all_bitwise_equal":all_equal,"source":source,"cases":reports,"warmup_replays":1000,"trial_replays":200,"trials_per_case":6}),
    )
}

/// Distributed canonical FFN. Loads independently qualified source-byte TP
/// atlases, executes real routing/shared/routed paths, and uses native RoCE.
pub fn ffn_peer(config_path: &Path) -> Result<serde_json::Value> {
    use crate::{
        checkpoint::Checkpoint,
        config::ModelConfig,
        expert_packing,
        ffn::{LayerFfn, TokenRows},
        layout::RankPlan,
        moe::Workspace,
        roce::Roce,
        weights::Weights,
    };
    use std::{collections::BTreeMap, path::PathBuf, rc::Rc};
    #[derive(serde::Deserialize)]
    struct Peer {
        config: PathBuf,
        checkpoint: PathBuf,
        rank: usize,
        core: PathBuf,
        kernels: PathBuf,
        staged: PathBuf,
        fixture: PathBuf,
        global_fixture: PathBuf,
        endpoint: String,
        ticket: PathBuf,
    }
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        input_sha256: String,
        tokens_sha256: String,
        live: usize,
    }
    #[derive(serde::Deserialize)]
    struct Cases {
        rows: usize,
        world: usize,
        cases: Vec<Case>,
    }
    #[derive(serde::Deserialize)]
    struct LocalCases {
        gpu_prepared_sha256: BTreeMap<String, String>,
    }
    let p: Peer = serde_json::from_slice(&std::fs::read(config_path)?)?;
    ensure!(p.rank < 4, "FFN peer rank");
    let cases: Cases =
        serde_json::from_slice(&std::fs::read(p.global_fixture.join("index.json"))?)?;
    ensure!(
        cases.rows == 6 && cases.world == 4 && cases.cases.len() == 4,
        "TP4 FFN global fixture geometry"
    );
    let local: LocalCases = serde_json::from_slice(&std::fs::read(p.fixture.join("index.json"))?)?;
    let cfg = ModelConfig::parse(&std::fs::read(p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let plan = RankPlan::build(&cp, &cfg, p.rank, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let module = cuda.module(&std::fs::read(&p.core)?)?;
    let mut weights = Weights::load_staged(&cuda, &cp, &plan, &["layers.0."], &p.staged)?;
    let source_sha = weights.source_sha256.clone();
    let atlas_bytes = weights.bytes();
    let prepared = expert_packing::prepare_layer(&cuda, &module, &mut weights, 0)?;
    ensure!(
        expert_packing::prepared_hashes(&weights, &prepared)? == local.gpu_prepared_sha256,
        "TP4 FFN prepared expert GPU bytes differ"
    );
    if std::env::var("CANADA_SCRATCH_PREFILL_COMM_OVERLAP_SCREEN").as_deref()==Ok("1") {
        return crate::prefill_comm_overlap_probe::run(&cuda,&module,&stream,Rc::new(weights),prepared,
            &p.kernels,p.rank,&p.endpoint,&std::fs::read(&p.ticket)?);
    }
    if std::env::var("CANADA_SCRATCH_VERIFY16_OVERLAP_SCREEN").as_deref()==Ok("1") {
        if std::env::var("CANADA_SCRATCH_ATTENTION16_SCREEN").as_deref()==Ok("1") {
            return crate::attention16_overlap_probe::run(&cuda,&module,&std::fs::read(&p.core)?,&stream,Rc::new(weights),
                &p.kernels,p.rank,&p.endpoint,&std::fs::read(&p.ticket)?);
        }
        return crate::verify16_overlap_probe::run(&cuda,&module,&stream,Rc::new(weights),prepared,
            &p.kernels,p.rank,&p.endpoint,&std::fs::read(&p.ticket)?);
    }
    let ffn = LayerFfn::new(
        &cuda,
        &module,
        &stream,
        Rc::new(weights),
        prepared,
        &p.kernels,
    )?;
    let workspace = Workspace::new(&cuda, &p.kernels.join("moe-runtime-m6-v0"))?;
    let mut input = cuda.allocate::<bf16>(6 * 5120)?;
    let mut tokens = TokenRows::new(&cuda, &[0; 6], 6)?;
    let mut output = cuda.allocate::<bf16>(6 * 5120)?;
    let comm = Roce::local(&cuda, &p.kernels.join("roce-v0"), p.rank)?;
    let blob = comm.blob()?;
    let ticket = std::fs::read(p.ticket)?;
    let all = crate::rendezvous::exchange_blob(p.rank, p.endpoint.parse()?, &ticket, &blob)?;
    comm.connect(&all)?;
    if std::env::var("CANADA_SCRATCH_FFN_CHUNK_SCREEN").as_deref()==Ok("1") {
        return crate::ffn_chunk_probe::run(&cuda,&stream,&ffn,&workspace,&mut input,&mut tokens,
            &mut output,&comm,&p.fixture,&p.global_fixture);
    }
    let mut graph = None;
    let mut reports = Vec::new();
    let mut all_equal = true;
    for case in cases.cases {
        ensure!(
            ["initial", "changed", "zero", "padded"].contains(&case.name.as_str()),
            "TP4 FFN fixture case"
        );
        let dir = p.fixture.join(&case.name);
        let b = std::fs::read(dir.join("input.bin"))?;
        let tb = std::fs::read(dir.join("tokens.bin"))?;
        use sha2::{Digest, Sha256};
        ensure!(
            b.len() == input.bytes()
                && tb.len() == 48
                && format!("{:x}", Sha256::digest(&b)) == case.input_sha256
                && format!("{:x}", Sha256::digest(&tb)) == case.tokens_sha256,
            "TP4 FFN rank inputs mismatch"
        );
        input.upload(
            &b.chunks_exact(2)
                .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())))
                .collect::<Vec<_>>(),
        )?;
        tokens.update(
            &tb.chunks_exact(8)
                .map(|v| i64::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>(),
            case.live,
        )?;
        stream.join_default()?;
        if graph.is_none() {
            ffn.run(&workspace, &input, &tokens, &stream)?;
            unsafe { comm.sum_bf16(ffn.output(), &mut output, &stream) }?;
            stream.synchronize()?;
            comm.health()?;
            graph = Some(stream.capture(|| -> Result<()> {
                ffn.run(&workspace, &input, &tokens, &stream)?;
                unsafe { comm.sum_bf16(ffn.output(), &mut output, &stream) }
            })?);
        }
        for _ in 0..100 {
            unsafe { graph.as_ref().unwrap().launch(&stream) }?;
        }
        let mut times = Vec::new();
        for _ in 0..3 {
            let a = cuda.event()?;
            let b = cuda.event()?;
            a.record(&stream)?;
            for _ in 0..100 {
                unsafe { graph.as_ref().unwrap().launch(&stream) }?;
            }
            b.record(&stream)?;
            times.push(b.elapsed_since(&a)? / 100.);
            comm.health()?;
        }
        stream.synchronize()?;
        comm.health()?;
        let mut taps = Vec::new();
        let mut compare = |name: &str,
                           actual: Vec<u8>,
                           size: usize,
                           expected_path: PathBuf|
         -> Result<()> {
            let expected = std::fs::read(expected_path)?;
            ensure!(actual.len() == expected.len(), "TP4 FFN tap size");
            let different = actual
                .chunks_exact(size)
                .zip(expected.chunks_exact(size))
                .filter(|(a, b)| a != b)
                .count();
            all_equal &= different == 0;
            taps.push(json!({"name":name,"values":actual.len()/size,"different_values":different,"bitwise_equal":different==0}));
            Ok(())
        };
        compare(
            "logits",
            ffn.router()
                .logits()
                .download()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            4,
            dir.join("logits.bin"),
        )?;
        compare(
            "ids",
            ffn.router()
                .ids()
                .download()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            4,
            dir.join("ids.bin"),
        )?;
        compare(
            "routing",
            ffn.router()
                .routing()
                .download()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            4,
            dir.join("routing.bin"),
        )?;
        for (name, buffer) in [
            ("shared_up", ffn.shared_up()),
            ("activation", ffn.activation()),
            ("shared_down", ffn.shared_down()),
            ("routed", workspace.output()),
            ("output", ffn.output()),
        ] {
            compare(
                name,
                buffer
                    .download()?
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect(),
                2,
                dir.join(format!("{name}.bin")),
            )?;
        }
        compare(
            "global_output",
            output
                .download()?
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
            2,
            p.global_fixture.join(&case.name).join("output.bin"),
        )?;
        let mut sorted = times.clone();
        sorted.sort_by(f32::total_cmp);
        let ms = sorted[1];
        let report = json!({"case":case.name,"graph_ms":ms,"trials_ms":times,"taps":taps});
        println!("{}", report);
        reports.push(report);
    }
    Ok(
        json!({"rank":p.rank,"scope":"canonical layer-0 TP4 FFN including real routing/shared/routed outputs and native RoCE; before mHC; not complete layer/model",
        "ffn_overlap_enabled":std::env::var("CANADA_SCRATCH_FFN_OVERLAP").as_deref()==Ok("1"),"all_bitwise_equal":all_equal,"source_checkpoint_tp_bytes_sha256":source_sha,"atlas_bytes":atlas_bytes,
        "source_gpu_bytes_verified":true,"warmup_replays":100,"trial_replays":100,"trials_per_case":3,"cases":reports}),
    )
}
