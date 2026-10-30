//! Token-input TP4 first-layer integration. No oracle intermediates are execution
//! inputs. Only external trace comparison can qualify this bounded model prefix.
use crate::{
    attention::{Candidates, PagedKv},
    blas::Blas,
    checkpoint::Checkpoint,
    compressed_source::{Layer as Source, Metadata, Phase},
    config::ModelConfig,
    cuda::Cuda,
    decoder::SwaLayer,
    engram::{PackedShard, RowStore},
    engram_gpu::Rows,
    expert_packing,
    ffn::TokenRows,
    hash_history::{HashMetadata, History},
    kv::RowLocations,
    layer_attention::InputMode,
    layout::RankPlan,
    model_io::ModelIo,
    prelude::{FrequencyTable, Positions},
    roce::Roce,
    weights::Weights,
    woa_medium::Scratch,
};
use anyhow::{Result, ensure};
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
    io_stage: PathBuf,
    endpoint: String,
    ticket: PathBuf,
    output: PathBuf,
    input_ids: Vec<i64>,
    #[serde(default)]
    graph_experiment: bool,
    #[serde(default = "default_layers")]
    layers: usize,
    staged1: Option<PathBuf>,
    staged2: Option<PathBuf>,
    backing: Option<PathBuf>,
    hash_metadata: Option<PathBuf>,
}
fn default_layers() -> usize {
    1
}
struct Second {
    layer: SwaLayer,
    cache: PagedKv,
    rows: Rows,
    store: RowStore,
    weight_bytes: usize,
    source_sha256: String,
}
struct Third {
    layer: SwaLayer,
    cache: PagedKv,
    source: Source,
    metadata: Metadata,
    positions: Positions,
    locations: RowLocations,
    weight_bytes: usize,
    source_sha256: String,
}
fn execute(
    io: &mut ModelIo,
    first: &mut SwaLayer,
    cache: &mut PagedKv,
    second: Option<&mut Second>,
    third: Option<&mut Third>,
    positions: &Positions,
    locations: &RowLocations,
    candidates: &Candidates,
    tokens: &TokenRows,
    comm: &Roce,
    stream: &crate::cuda::Stream,
) -> Result<()> {
    io.run_input(tokens, comm, stream)?;
    first.run(
        io.input(),
        positions,
        locations,
        cache,
        candidates,
        tokens,
        comm,
        stream,
    )?;
    if let Some(s) = second {
        s.layer.run_from_previous(
            first.output(),
            Some(first.pre()),
            Some(&mut s.rows),
            positions,
            locations,
            &mut s.cache,
            candidates,
            tokens,
            comm,
            stream,
        )?;
        if let Some(t) = third {
            t.layer.run_indexed_from_previous(
                s.layer.output(),
                s.layer.pre(),
                &t.positions,
                &t.locations,
                &mut t.cache,
                candidates,
                tokens,
                &mut t.source,
                &t.metadata,
                comm,
                stream,
            )?;
        }
    } else {
        ensure!(third.is_none(), "source layer requires live predecessor");
    }
    Ok(())
}
type Snapshot = (BTreeMap<String, Vec<u8>>, BTreeMap<String, Vec<u8>>);
fn snapshots(
    cuda: &Cuda,
    io: &ModelIo,
    first: &SwaLayer,
    cache: &PagedKv,
    second: Option<&Second>,
    third: Option<&Third>,
) -> Result<Vec<Snapshot>> {
    let (local, mut global) = crate::layer_probe::snapshots(cuda, first, cache)?;
    global.insert(
        "input_residual".into(),
        crate::layer_probe::bf16_bytes(io.input())?,
    );
    let mut out = vec![(local, global)];
    if let Some(s) = second {
        let (local, mut global) = crate::layer_probe::snapshots(cuda, &s.layer, &s.cache)?;
        let engram = s
            .layer
            .engram()
            .ok_or_else(|| anyhow::anyhow!("layer1 Engram absent"))?;
        global.insert(
            "input_residual".into(),
            crate::layer_probe::bf16_bytes(engram.output())?,
        );
        out.push((local, global));
        if let Some(t) = third {
            let (mut local, mut global) = crate::layer_probe::snapshots(cuda, &t.layer, &t.cache)?;
            global.insert(
                "input_residual".into(),
                crate::layer_probe::bf16_bytes(s.layer.output())?,
            );
            let mut raw = |name: &str, ptr: u64, size: usize| -> Result<()> {
                local.insert(name.into(), unsafe { cuda.read_device(ptr, size) }?);
                Ok(())
            };
            match t.source.projection() {
                crate::compression::Input::Ratio2(b) => {
                    raw("source_projection", b.ptr(), b.bytes())?
                }
                crate::compression::Input::Ratio1(b) => {
                    raw("source_projection", b.ptr(), b.bytes())?
                }
            }
            for (name, b) in [
                ("source_latent", t.source.source().latent()),
                ("source_wk", t.source.wk()),
                ("index_query", t.source.query().query_projection()),
                ("index_heads", t.source.query().head_projection()),
            ] {
                raw(name, b.ptr(), b.bytes())?;
            }
            raw(
                "source_compressed_cache",
                t.source.cache().data().ptr(),
                t.source.cache().data().bytes(),
            )?;
            raw(
                "source_index_keys",
                t.source.keys().data().ptr(),
                t.source.keys().data().bytes(),
            )?;
            if let Some(b) = t.source.source().ring() {
                raw("source_ring", b.ptr(), b.bytes())?;
            }
            raw(
                "source_index_scores",
                t.source.query().scores()?.ptr(),
                t.source.query().scores()?.bytes(),
            )?;
            raw(
                "source_selected",
                t.source.query().selected().ptr(),
                t.source.query().selected().bytes(),
            )?;
            out.push((local, global));
        }
    }
    Ok(out)
}
pub fn run(path: &Path) -> Result<Value> {
    let p: Peer = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(
        p.rank < 4
            && (1..=3).contains(&p.layers)
            && p.input_ids.len() == 128
            && p.input_ids.iter().all(|&v| (0..129280).contains(&v))
            && !p.output.exists(),
        "prefill prefix probe admission"
    );
    let cfg = ModelConfig::parse(&std::fs::read(p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let plan = RankPlan::build(&cp, &cfg, p.rank, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let core_bytes = std::fs::read(p.core)?;
    let module = cuda.module(&core_bytes)?;
    let mut weights = Weights::load_staged(&cuda, &cp, &plan, &["layers.0."], &p.staged)?;
    let source_hash = weights.source_sha256.clone();
    let weight_bytes = weights.bytes();
    let prepared = expert_packing::prepare_layer(&cuda, &module, &mut weights, 0)?;
    let frequency = Rc::new(FrequencyTable::new(&cuda, &module, &stream, 131072, false)?);
    let blas = Rc::new(Blas::new(&cuda)?);
    let medium_woa = Scratch::new(&cuda, &core_bytes, 128)?;
    let mut layer = SwaLayer::new_rows(
        &cuda,
        &module,
        &stream,
        Rc::new(weights),
        prepared,
        &p.kernels,
        frequency.clone(),
        InputMode::Split(p.rank),
        128,
        Some(blas.clone()),
        Some(medium_woa.clone()),
    )?;
    let mut second = if p.layers >= 2 {
        let staged = p
            .staged1
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer1 stage absent"))?;
        let backing = p
            .backing
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Engram backing absent"))?;
        let hash_metadata = p
            .hash_metadata
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Engram hash metadata absent"))?;
        let mut w = Weights::load_staged(&cuda, &cp, &plan, &["layers.1."], staged)?;
        let bytes = w.bytes();
        let sha = w.source_sha256.clone();
        let prepared = expert_packing::prepare_layer(&cuda, &module, &mut w, 1)?;
        let layer = SwaLayer::new_rows(
            &cuda,
            &module,
            &stream,
            Rc::new(w),
            prepared,
            &p.kernels,
            frequency.clone(),
            InputMode::Split(p.rank),
            128,
            Some(blas.clone()),
            Some(medium_woa.clone()),
        )?;
        let metadata = HashMetadata::parse(&std::fs::read(hash_metadata)?)?;
        let batch = History::new(1).prepare(
            &p.input_ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
            &metadata,
        )?;
        let hashes: Vec<[u64; 24]> = batch.rows.iter().map(|v| v[0]).collect();
        let store = RowStore::new(
            PackedShard::open(
                &backing.join(format!("engram-l1-r{}of4.bin", p.rank)),
                1,
                p.rank,
                4,
                384006168,
            )?,
            64 * 1024 * 1024,
        )?;
        let mut rows = Rows::new(&cuda, 128, &store)?;
        rows.prepare(&hashes, &store, 8)?;
        Some(Second {
            layer,
            cache: PagedKv::new(&cuda, 2, 64)?,
            rows,
            store,
            weight_bytes: bytes,
            source_sha256: sha,
        })
    } else {
        None
    };
    let mut io = ModelIo::new(
        &cuda,
        &core_bytes,
        &stream,
        &cp,
        &plan,
        &p.io_stage,
        &p.kernels,
        128,
    )?;
    let tokens = TokenRows::new(&cuda, &p.input_ids, 128)?;
    let pos: Vec<i32> = (0..128).collect();
    let positions = Positions::upload(&cuda, &pos, 131072)?;
    let mut cache = PagedKv::new(&cuda, 2, 64)?;
    let locations = RowLocations::upload(&cuda, &pos, &pos, cache.capacity(), 131072)?;
    let ids: Vec<i32> = (0..128i32)
        .flat_map(|i| (0..128i32).map(move |j| if j <= i { i - j } else { -1 }))
        .collect();
    let lengths: Vec<i32> = (1..=128).collect();
    let candidates = Candidates::upload(&cuda, 128, 128, cache.capacity(), &ids, &lengths)?;
    let mut third = if p.layers == 3 {
        let staged = p
            .staged2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer2 stage absent"))?;
        let frequency = Rc::new(FrequencyTable::new(&cuda, &module, &stream, 1048576, true)?);
        let mut w = Weights::load_staged(&cuda, &cp, &plan, &["layers.2."], staged)?;
        let weight_bytes = w.bytes();
        let source_sha256 = w.source_sha256.clone();
        let prepared = expert_packing::prepare_layer(&cuda, &module, &mut w, 2)?;
        let w = Rc::new(w);
        let layer = SwaLayer::new_rows(
            &cuda,
            &module,
            &stream,
            w.clone(),
            prepared,
            &p.kernels,
            frequency.clone(),
            InputMode::Split(p.rank),
            128,
            Some(blas.clone()),
            Some(medium_woa.clone()),
        )?;
        let slots: Vec<i32> = (64..128).collect();
        let lengths: Vec<i32> = (0..128).map(|i| (i + 1) / 2).collect();
        let source = Source::new_rows(
            &cuda,
            &stream,
            w,
            blas,
            2,
            &p.kernels,
            frequency,
            &core_bytes,
            4,
            2,
            &slots,
            &lengths,
            128,
            Phase::Extend,
        )?;
        let raw_locations: Vec<i64> = (128..256).collect();
        let metadata = Metadata::new(
            &cuda,
            &source,
            &pos,
            &vec![0; 128],
            &raw_locations,
            &slots,
            &lengths,
            1048576,
        )?;
        Some(Third {
            layer,
            cache: PagedKv::new(&cuda, 2, 64)?,
            source,
            metadata,
            positions: Positions::upload(&cuda, &pos, 1048576)?,
            locations: RowLocations::upload(&cuda, &pos, &pos, 128, 1048576)?,
            weight_bytes,
            source_sha256,
        })
    } else {
        None
    };
    let mut comm = Roce::local(&cuda, &p.kernels.join("roce-v0"), p.rank)?;
    comm.attach_gather(&p.kernels.join("roce-gather-v1"))?;
    let ticket = std::fs::read(p.ticket)?;
    let all = crate::rendezvous::exchange_blob_with_timeout(
        p.rank,
        p.endpoint.parse()?,
        &ticket,
        &comm.blob()?,
        std::time::Duration::from_secs(120),
    )?;
    comm.connect(&all)?;
    let api = crate::collective::Nccl::load_qualified_prefill()?;
    let local = if p.rank == 0 {
        api.unique_id()?.bytes
    } else {
        [0u8; 128]
    };
    let mut endpoint: std::net::SocketAddr = p.endpoint.parse()?;
    endpoint.set_port(
        endpoint
            .port()
            .checked_add(2)
            .ok_or_else(|| anyhow::anyhow!("rendezvous port"))?,
    );
    let ids = crate::rendezvous::exchange_blob_with_timeout(
        p.rank,
        endpoint,
        &ticket,
        &local,
        std::time::Duration::from_secs(120),
    )?;
    comm.attach_prefill_sum(api.join(
        &cuda,
        crate::collective::UniqueId {
            bytes: ids[..128].try_into()?,
        },
        p.rank,
        4,
    )?)?;
    std::fs::create_dir(&p.output)?;
    let mut trials = Vec::new();
    for repeat in 0..2 {
        cache.clear(&cuda, &stream)?;
        if let Some(s) = second.as_mut() {
            s.cache.clear(&cuda, &stream)?;
        }
        if let Some(t) = third.as_mut() {
            t.cache.clear(&cuda, &stream)?;
            t.source.clear_state(&stream)?;
        }
        let start = cuda.event()?;
        let end = cuda.event()?;
        start.record(&stream)?;
        execute(
            &mut io,
            &mut layer,
            &mut cache,
            second.as_mut(),
            third.as_mut(),
            &positions,
            &locations,
            &candidates,
            &tokens,
            &comm,
            &stream,
        )?;
        end.record(&stream)?;
        let gpu_ms = end.elapsed_since(&start)?;
        stream.synchronize()?;
        comm.health()?;
        let root = p.output.join(format!("repeat{repeat}"));
        std::fs::create_dir(&root)?;
        for (layer_id, (mut local, mut global)) in
            snapshots(&cuda, &io, &layer, &cache, second.as_ref(), third.as_ref())?
                .into_iter()
                .enumerate()
        {
            let root = if layer_id == 0 {
                root.clone()
            } else {
                let v = root.join(format!("layer{layer_id}"));
                std::fs::create_dir(&v)?;
                v
            };
            let mut taps = Vec::new();
            for (scope, values) in [("local", &mut local), ("global", &mut global)] {
                for (name, bytes) in values.iter() {
                    let file = format!("{scope}-{name}.bin");
                    std::fs::write(root.join(&file), bytes)?;
                    taps.push(
                        json!({"scope":scope,"name":name,"file":file,"bytes":bytes.len(),
                    "sha256":format!("{:x}",Sha256::digest(bytes))}),
                    );
                }
            }
            let index =
                json!({"rows":128,"layer":layer_id,"rank":p.rank,"repeat":repeat,"taps":taps});
            std::fs::write(root.join("index.json"), serde_json::to_vec_pretty(&index)?)?;
        }
        trials.push(json!({"repeat":repeat,"gpu_ms":gpu_ms}));
    }
    stream.synchronize()?;
    comm.health()?;
    let captured = if p.graph_experiment {
        let expected = snapshots(&cuda, &io, &layer, &cache, second.as_ref(), third.as_ref())?;
        // This graph has no cross-query history beyond its freshly computed
        // 128 rows. Every replay recomputes embedding and all causally visible
        // keys. The graph is scoped after comm and drops before NCCL teardown.
        let graph = stream.capture(|| {
            execute(
                &mut io,
                &mut layer,
                &mut cache,
                second.as_mut(),
                third.as_mut(),
                &positions,
                &locations,
                &candidates,
                &tokens,
                &comm,
                &stream,
            )
        })?;
        for _ in 0..50 {
            unsafe { graph.launch(&stream) }?;
        }
        stream.synchronize()?;
        comm.health()?;
        let mut samples = Vec::new();
        for _ in 0..100 {
            let start = cuda.event()?;
            let end = cuda.event()?;
            start.record(&stream)?;
            unsafe { graph.launch(&stream) }?;
            end.record(&stream)?;
            samples.push(end.elapsed_since(&start)?);
        }
        stream.synchronize()?;
        comm.health()?;
        let actual = snapshots(&cuda, &io, &layer, &cache, second.as_ref(), third.as_ref())?;
        let exact = actual == expected;
        let mut sorted = samples.clone();
        sorted.sort_by(f32::total_cmp);
        let result = json!({"warmups":50,"samples_ms":samples,"p50_ms":sorted[49],
            "p95_ms":sorted[94],"p99_ms":sorted[98],"max_ms":sorted[99],
            "all_local_global_and_cache_bytes_equal_eager":exact,"full_engine_acceptance":false,
            "layers":p.layers,"scope":"Unchanged fresh128token model prefix only, not full model prefill or serving latency."});
        std::fs::write(
            p.output.join("captured.json"),
            serde_json::to_vec_pretty(&result)?,
        )?;
        ensure!(exact, "captured prefill prefix differs; evidence saved");
        Some(result)
    } else {
        None
    };
    Ok(
        json!({"rank":p.rank,"rows":128,"layers":p.layers,"input_ids":p.input_ids,"source_sha256":source_hash,
        "layer_weight_bytes":weight_bytes,"trials":trials,"captured":captured,"tokens_are_only_model_inputs":true,
        "second_layer":second.as_ref().map(|s|json!({"weight_bytes":s.weight_bytes,"source_sha256":s.source_sha256,
            "row_cache_budget_bytes":s.store.resident_bytes,"row_cache":s.store.stats()})),
        "third_layer":third.as_ref().map(|s|json!({"weight_bytes":s.weight_bytes,"source_sha256":s.source_sha256,
            "compression_phase":"extend","mapping_error":s.source.error().ok()})),
        "full_engine_acceptance":false,"scope":"Fresh128-token embedding and bounded decoder prefix only; must compare against actual qualified SGLang."}),
    )
}
