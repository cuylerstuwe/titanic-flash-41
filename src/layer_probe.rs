//! Distributed canonical layer trace, separate from model serving/scheduling.
use crate::{
    attention::{Candidates, PagedKv},
    checkpoint::Checkpoint,
    config::ModelConfig,
    cuda::{Buffer, Cuda},
    decoder::SwaLayer,
    expert_packing,
    ffn::TokenRows,
    kv::RowLocations,
    layout::RankPlan,
    prelude::{FrequencyTable, Positions},
    roce::Roce,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    rc::Rc,
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
    #[serde(default)]
    split_qkv: bool,
}
#[derive(Deserialize)]
pub(crate) struct Tap {
    pub name: String,
    pub bytes: usize,
    pub element_bytes: usize,
    pub sha256: String,
}
#[derive(Deserialize)]
pub(crate) struct RankTaps {
    pub rank: usize,
    pub taps: Vec<Tap>,
}
#[derive(Deserialize)]
pub(crate) struct Case {
    pub name: String,
    pub source: Vec<Tap>,
    pub global_taps: Vec<Tap>,
    pub ranks: Vec<RankTaps>,
}
#[derive(Deserialize)]
pub(crate) struct Cases {
    pub rows: usize,
    pub world: usize,
    pub layer: usize,
    pub frequency_rows: usize,
    pub cases: Vec<Case>,
}
pub(crate) fn integers(path: &Path) -> Result<Vec<i32>> {
    let b = std::fs::read(path)?;
    ensure!(b.len() % 4 == 0, "integer fixture bytes");
    Ok(b.chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
pub(crate) fn bf16_bytes(buffer: &Buffer<bf16>) -> Result<Vec<u8>> {
    Ok(buffer
        .download()?
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect())
}
fn float_bytes(buffer: &Buffer<f32>) -> Result<Vec<u8>> {
    Ok(buffer
        .download()?
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect())
}
pub(crate) fn fixture(root: &Path, tap: &Tap) -> Result<Vec<u8>> {
    ensure!(
        !tap.name.is_empty()
            && tap
                .name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_'),
        "fixture tap name"
    );
    ensure!(
        [1, 2, 4, 8].contains(&tap.element_bytes)
            && tap.bytes > 0
            && tap.bytes <= 64 * 1024 * 1024
            && tap.bytes % tap.element_bytes == 0,
        "fixture tap size/type"
    );
    let b = std::fs::read(root.join(format!("{}.bin", tap.name)))?;
    ensure!(
        b.len() == tap.bytes && format!("{:x}", Sha256::digest(&b)) == tap.sha256,
        "fixture hash mismatch: {}",
        tap.name
    );
    Ok(b)
}
pub(crate) fn compare(
    root: &Path,
    expected: &[Tap],
    actual: &BTreeMap<String, Vec<u8>>,
) -> Result<Vec<Value>> {
    ensure!(
        expected.len() == actual.len(),
        "layer trace tap count changed"
    );
    expected.iter().map(|tap|->Result<Value>{
        let reference=fixture(root,tap)?;let result=actual.get(&tap.name).ok_or_else(||anyhow::anyhow!("missing actual tap {}",tap.name))?;
        ensure!(result.len()==reference.len(),"layer trace tap byte count: {}",tap.name);
        let different=result.chunks_exact(tap.element_bytes).zip(reference.chunks_exact(tap.element_bytes)).filter(|(a,b)|a!=b).count();
        let first:Vec<_>=result.chunks_exact(tap.element_bytes).zip(reference.chunks_exact(tap.element_bytes))
            .enumerate().filter(|(_, (a,b))|a!=b).take(8)
            .map(|(i,(a,b))|json!({"index":i,"actual_le_hex":a.iter().map(|v|format!("{v:02x}")).collect::<String>(),
                "expected_le_hex":b.iter().map(|v|format!("{v:02x}")).collect::<String>()})).collect();
        Ok(json!({"name":tap.name,"bytes":tap.bytes,"element_bytes":tap.element_bytes,"values":tap.bytes/tap.element_bytes,
            "actual_sha256":format!("{:x}",Sha256::digest(result)),"reference_sha256":tap.sha256,
            "different_values":different,"bitwise_equal":different==0,"first_differences":first}))
    }).collect()
}
pub fn run(config_path: &Path) -> Result<Value> {
    let p: Peer = serde_json::from_slice(&std::fs::read(config_path)?)?;
    ensure!(p.rank < 4, "layer peer rank");
    let cases: Cases = serde_json::from_slice(&std::fs::read(p.fixture.join("index.json"))?)?;
    ensure!(
        (cases.rows, cases.world, cases.layer, cases.frequency_rows) == (6, 4, 0, 131072)
            && cases.cases.len() == 4,
        "layer fixture geometry"
    );
    let cfg = ModelConfig::parse(&std::fs::read(p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let plan = RankPlan::build(&cp, &cfg, p.rank, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let core = cuda.module(&std::fs::read(p.core)?)?;
    let mut weights = Weights::load_staged(&cuda, &cp, &plan, &["layers.0."], &p.staged)?;
    let source_sha = weights.source_sha256.clone();
    let atlas_bytes = weights.bytes();
    let prepared = expert_packing::prepare_layer(&cuda, &core, &mut weights, 0)?;
    let frequency = Rc::new(FrequencyTable::new(
        &cuda,
        &core,
        &stream,
        cases.frequency_rows,
        false,
    )?);
    let mut layer = SwaLayer::new_with_mode(
        &cuda,
        &core,
        &stream,
        Rc::new(weights),
        prepared,
        &p.kernels,
        frequency,
        if p.split_qkv {
            crate::layer_attention::InputMode::Split(p.rank)
        } else {
            crate::layer_attention::InputMode::Fused
        },
    )?;
    let combined = crate::l2_prefetch::enabled()
        && std::env::var("CANADA_SCRATCH_FFN_OVERLAP").as_deref() == Ok("1");
    if combined {
        // Exercise the third communication-window branch with a retained
        // immutable projection; the full target wires the actual next layer.
        layer.install_next_input_prefetch(&cuda, &core, &stream, layer.input_matrix())?;
    }
    let mut cache = PagedKv::new(&cuda, 4, 64)?;
    cache.upload(&std::fs::read(p.fixture.join("initial_cache.bin"))?)?;
    let pos = integers(&p.fixture.join("positions.bin"))?;
    let loc = integers(&p.fixture.join("locations.bin"))?;
    let positions = Positions::upload(&cuda, &pos, cases.frequency_rows)?;
    let locations =
        RowLocations::upload(&cuda, &pos, &loc, cache.capacity(), cases.frequency_rows)?;
    let candidates = Candidates::upload(
        &cuda,
        6,
        128,
        cache.capacity(),
        &integers(&p.fixture.join("ids.bin"))?,
        &integers(&p.fixture.join("lengths.bin"))?,
    )?;
    let mut input = cuda.allocate::<bf16>(6 * 20480)?;
    let mut tokens = TokenRows::new(&cuda, &[0; 6], 6)?;
    let mut comm = Roce::local(&cuda, &p.kernels.join("roce-v0"), p.rank)?;
    if p.split_qkv {
        comm.attach_gather(&p.kernels.join("roce-gather-v1"))?;
    }
    let all = crate::rendezvous::exchange_blob(
        p.rank,
        p.endpoint.parse()?,
        &std::fs::read(&p.ticket)?,
        &comm.blob()?,
    )?;
    comm.connect(&all)?;
    let mut graph = None;
    let mut reports = Vec::new();
    let mut all_equal = true;
    let chunk_screen=std::env::var("CANADA_SCRATCH_FFN_PREFETCH_CHUNK_SCREEN").as_deref()==Ok("1");
    let fine_chunks=std::env::var("CANADA_SCRATCH_FFN_PREFETCH_FINE_SCREEN").as_deref()==Ok("1");
    ensure!(!fine_chunks||chunk_screen,"fine FFN hints require the chunk component screen");
    let chunks=if fine_chunks {[16384u32,8192,4096,2048]}else{[16384u32,65536,262144,1048576]};
    let region_screen=std::env::var("CANADA_SCRATCH_FFN_PREFETCH_REGION_SCREEN").as_deref()==Ok("1");
    let budget_screen=std::env::var("CANADA_SCRATCH_FFN_PREFETCH_BUDGET_SCREEN").as_deref()==Ok("1");
    ensure!([chunk_screen,region_screen,budget_screen].iter().filter(|v|**v).count()<=1,"independent hint component screens");
    let hint_screen=chunk_screen||region_screen||budget_screen;
    let displacement=if hint_screen {
        ensure!(combined,"FFN granularity screen needs ordinary overlap and hints");
        let mut address:std::net::SocketAddr=p.endpoint.parse()?;
        address.set_port(address.port().checked_add(1).ok_or_else(||anyhow::anyhow!("hint ready port"))?);
        ensure!(crate::rendezvous::exchange_blob(p.rank,address,&std::fs::read(&p.ticket)?,b"R")?==b"RRRR",
            "FFN hint all peers connected");
        Some(cuda.allocate::<u16>(64*1024*1024/2)?)
    }else{None};
    for case in cases.cases {
        ensure!(
            ["initial", "changed", "zero", "padded"].contains(&case.name.as_str()),
            "unknown layer fixture phase"
        );
        ensure!(
            case.source.len() == 3 && case.global_taps.len() == 16 && case.ranks.len() == 4,
            "layer fixture trace count"
        );
        let dir = p.fixture.join(&case.name);
        let source: BTreeMap<_, _> = case
            .source
            .iter()
            .map(|tap| Ok((tap.name.as_str(), fixture(&dir, tap)?)))
            .collect::<Result<_>>()?;
        let bytes = &source["input"];
        ensure!(bytes.len() == input.bytes(), "layer input fixture size");
        input.upload(
            &bytes
                .chunks_exact(2)
                .map(|v| bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())))
                .collect::<Vec<_>>(),
        )?;
        let tb = &source["tokens"];
        let lb = &source["live"];
        ensure!(tb.len() == 48 && lb.len() == 4, "layer token metadata size");
        let live = i32::from_le_bytes(lb.as_slice().try_into().unwrap());
        ensure!((0..=6).contains(&live), "layer live rows");
        tokens.update(
            &tb.chunks_exact(8)
                .map(|v| i64::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>(),
            live as usize,
        )?;
        stream.join_default()?;
        if hint_screen {
            for displaced in [false,true] {
                let mut variants=Vec::new();let mut baseline=None;let mut exact_hashes=None;
                for variant in 0..4 {
                    if region_screen {layer.set_ffn_prefetch_region_for_probe(variant,&stream)?;}
                    else if budget_screen {layer.set_ffn_prefetch_budget_for_probe([6usize,8,10,12][variant as usize]*1024*1024,&stream)?;}
                    else {layer.set_ffn_prefetch_chunk_for_probe(chunks[variant as usize],&stream)?;}
                    let mut enqueue=||->Result<()> {
                        if displaced {
                            let d=displacement.as_ref().unwrap();
                            let(mut ptr,mut n,mut rank,mut pattern)=(d.ptr(),d.len()as i32,p.rank as i32,0i32);
                            let mut args=[(&mut ptr as *mut u64).cast(),(&mut n as *mut i32).cast(),
                                (&mut rank as *mut i32).cast(),(&mut pattern as *mut i32).cast()];
                            unsafe{core.function("ca_collective_fixture")?.launch(
                                [d.len().div_ceil(256)as u32,1,1],[256,1,1],0,&stream,&mut args)?;}
                        }
                        layer.run(&input,&positions,&locations,&mut cache,&candidates,&tokens,&comm,&stream)
                    };
                    enqueue()?;stream.synchronize()?;comm.health()?;
                    variants.push(stream.capture(&mut enqueue)?);
                    drop(enqueue);
                    let values=snapshots(&cuda,&layer,&cache)?;
                    if let Some(expected)=&baseline {ensure!(expected==&values,"FFN hint changed GPU bytes");}
                    else {
                        exact_hashes=Some(values.0.iter().chain(values.1.iter()).map(|(k,v)|
                            (k.clone(),format!("{:x}",Sha256::digest(v)))).collect::<BTreeMap<_,_>>());
                        baseline=Some(values);
                    }
                }
                let mut samples=vec![Vec::new();4];
                for repeat in 0..4 {
                    let order=if repeat%2==0{[0,1,2,3]}else{[3,2,1,0]};
                    for variant in order {
                        for _ in 0..4 {unsafe{variants[variant].launch(&stream)?;}}
                        stream.synchronize()?;
                        let start=cuda.event()?;let end=cuda.event()?;start.record(&stream)?;
                        for _ in 0..24 {unsafe{variants[variant].launch(&stream)?;}}
                        end.record(&stream)?;samples[variant].push(end.elapsed_since(&start)?/24.);
                        comm.health()?;
                        ensure!(baseline.as_ref()==Some(&snapshots(&cuda,&layer,&cache)?),
                            "FFN hint graph replay changed GPU bytes");
                    }
                }
                reports.push(json!({"case":case.name,"live":live,"displaced":displaced,
                    "chunks":if chunk_screen {json!(chunks)}else{Value::Null},
                    "regions":if region_screen {json!(["router_and_partial_shared","router_only","partial_shared_only","disabled"])}else{Value::Null},
                    "budgets_mib":if budget_screen {json!([6,8,10,12])}else{Value::Null},
                    "samples_ms":samples,
                    "all_bitwise_equal":true,"gpu_taps_sha256":exact_hashes}));
                eprintln!("rank{} FFN hint {} displaced{displaced} exact",p.rank,case.name);
            }
            continue;
        }
        if graph.is_none() {
            layer.run(
                &input,
                &positions,
                &locations,
                &mut cache,
                &candidates,
                &tokens,
                &comm,
                &stream,
            )?;
            stream.synchronize()?;
            comm.health()?;
            graph = Some(stream.capture(|| {
                layer.run(
                    &input,
                    &positions,
                    &locations,
                    &mut cache,
                    &candidates,
                    &tokens,
                    &comm,
                    &stream,
                )
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
        let (actual, global) = snapshots(&cuda, &layer, &cache)?;
        let expected = case
            .ranks
            .iter()
            .find(|r| r.rank == p.rank)
            .ok_or_else(|| anyhow::anyhow!("missing rank trace"))?;
        ensure!(expected.taps.len() == 21, "rank trace count");
        let local = compare(
            &dir.join(format!("rank{}", p.rank)),
            &expected.taps,
            &actual,
        )?;
        let shared = compare(&dir, &case.global_taps, &global)?;
        let different: usize = local
            .iter()
            .chain(shared.iter())
            .map(|t| t["different_values"].as_u64().unwrap() as usize)
            .sum();
        all_equal &= different == 0;
        let mut sorted = times.clone();
        sorted.sort_by(f32::total_cmp);
        let ms = sorted[1];
        let report = json!({"case":case.name,"graph_ms":ms,"trials_ms":times,"different_values":different,"local_taps":local,"global_taps":shared});
        println!("{}", report);
        reports.push(report);
    }
    stream.synchronize()?;
    comm.health()?;
    if hint_screen {
        return Ok(json!({"rank":p.rank,"all_bitwise_equal":true,"cases":reports,
            "source_checkpoint_tp_bytes_sha256":source_sha,"atlas_bytes":atlas_bytes,
            "scope":"Same layer0 storage and consumers: fixed FFN hint chunk or owned descriptor-subset comparison. Real attention/mHC/FFN/TP4 transport; steady replay and64MiB displacement, original/changed/zero/padded inputs. All37 GPU tap buffers exact to current16KiB/full-descriptor control, not historical fixture precision. Displacement cost included; no whole-model claim."}));
    }
    Ok(
        json!({"rank":p.rank,"all_bitwise_equal":all_equal,"cases":reports,"source_checkpoint_tp_bytes_sha256":source_sha,"atlas_bytes":atlas_bytes,
        "l2_prefetch_enabled":crate::l2_prefetch::enabled(),"split_qkv":p.split_qkv,
        "fp8_pdl_enabled":crate::fp8_linear::pdl_enabled(),
        "ffn_overlap_enabled":std::env::var("CANADA_SCRATCH_FFN_OVERLAP").as_deref()==Ok("1"),
        "mhc_overlap_enabled":crate::mhc::overlap_enabled(),
        "next_input_hint_exercised":combined,
        "scope":"canonical layer 0, six rows, controlled past KV, real four-peer attention/mHC/FFN/two RoCE reductions; not full model",
        "warmup_replays":100,"trial_replays":100,"trials_per_case":3}),
    )
}

pub(crate) fn snapshots(
    cuda: &Cuda,
    layer: &SwaLayer,
    cache: &PagedKv,
) -> Result<(BTreeMap<String, Vec<u8>>, BTreeMap<String, Vec<u8>>)> {
    let mut global = BTreeMap::new();
    for (name, boundary) in [
        ("attn", layer.attn_boundary()),
        ("ffn", layer.ffn_boundary()),
    ] {
        for (suffix, b) in [
            ("partial_mix", boundary.partial_mix()),
            ("partial_square", boundary.partial_square()),
            ("pre", boundary.pre()),
            ("post", boundary.post()),
            ("comb", boundary.comb()),
        ] {
            global.insert(format!("mhc_{name}_{suffix}"), float_bytes(b)?);
        }
        global.insert(format!("mhc_{name}_input"), bf16_bytes(boundary.input())?);
        global.insert(format!("mhc_{name}_output"), bf16_bytes(boundary.output())?);
    }
    global.insert("attn_reduced".into(), bf16_bytes(layer.attn_reduced())?);
    global.insert("ffn_reduced".into(), bf16_bytes(layer.ffn_reduced())?);
    let mut actual = BTreeMap::new();
    for (name, ptr, bytes, stride) in layer.attention().taps() {
        let result = if stride == 0 {
            unsafe { cuda.read_device(ptr, bytes) }?
        } else {
            let rows = layer.attention().rows();
            ensure!(bytes % rows == 0, "strided attention tap row size");
            let row_bytes = bytes / rows;
            let mut v = Vec::with_capacity(bytes);
            for r in 0..rows {
                v.extend(unsafe { cuda.read_device(ptr + (r * stride) as u64, row_bytes) }?);
            }
            v
        };
        actual.insert(format!("attn_{name}"), result);
    }
    let f = layer.ffn();
    actual.insert("ffn_logits".into(), float_bytes(f.router().logits())?);
    actual.insert(
        "ffn_ids".into(),
        f.router()
            .ids()
            .download()?
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    );
    actual.insert("ffn_routing".into(), float_bytes(f.router().routing())?);
    for (name, b) in [
        ("shared_up", f.shared_up()),
        ("activation", f.activation()),
        ("shared_down", f.shared_down()),
        ("routed", layer.routed()),
        ("output", f.output()),
    ] {
        actual.insert(format!("ffn_{name}"), bf16_bytes(b)?);
    }
    actual.insert("cache".into(), cache.data().download()?);
    Ok((actual, global))
}
