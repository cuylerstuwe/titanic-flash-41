use anyhow::{Result, bail};
use canada_engine::{checkpoint::Checkpoint, config::ModelConfig, layout::RankPlan};
use std::{env, fs::File, path::Path};

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("gpu-image-recycle") if args.len()==4=>{
            let report=canada_engine::image_recycle_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(std::fs::OpenOptions::new().write(true).create_new(true).open(&args[3])?,&report)?;
        }
        Some("gpu-local-sampler-readout") if args.len()==4=>{
            let report=canada_engine::probe_local_sampler_readout(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(std::fs::OpenOptions::new().write(true).create_new(true).open(&args[3])?,&report)?;
        }
        Some("gpu-device-digest") if args.len()==4=>{
            let report=canada_engine::weights::Weights::probe_device_digest(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(std::fs::OpenOptions::new().write(true).create_new(true).open(&args[3])?,&report)?;
        }
        Some("gpu-staged-read-ahead") if args.len()==4 => {
            let report=canada_engine::weights::Weights::probe_read_ahead(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(std::fs::OpenOptions::new().write(true).create_new(true).open(&args[3])?,&report)?;
        }
        Some("gpu-prompt-media") if args.len() == 3 => {
            println!("{}", serde_json::to_string_pretty(&canada_engine::prompt_media_probe::run(std::path::Path::new(&args[2]))?)?);
        }
        Some("gpu-vision") if args.len() == 3 => {
            canada_engine::vision_probe::run(Path::new(&args[2]))?;
        }
        #[cfg(target_os = "linux")]
        Some("engram-pair-io-probe") if args.len() == 5 => {
            let report=canada_engine::engram_pair_probe::run(Path::new(&args[2]),args[3].parse()?)?;
            serde_json::to_writer_pretty(File::create(&args[4])?,&report)?;
        }
        Some("build-provenance") if args.len() == 2 => {
            println!("{}", serde_json::json!({
                "source_manifest_sha256":option_env!("CANADA_BUILD_SOURCE_SHA256"),
                "protocol":cfg!(feature="protocol"),"serving":cfg!(feature="serving"),
                "vision_input":cfg!(feature="vision-input"),"vision_protocol":cfg!(feature="vision-protocol"),
                "storage_fault_injection":cfg!(feature="storage-fault-injection"),
                "package_version":env!("CARGO_PKG_VERSION")
            }));
        }
        #[cfg(feature = "serving")]
        Some("resident-http-peer") if args.len() == 3 => {
            canada_engine::resident_worker::run_http(Path::new(&args[2]))?;
        }
        Some("resident-token-peer") if args.len() == 3 => {
            canada_engine::resident_worker::run_file(Path::new(&args[2]))?;
        }
        Some("worker-config-validate") if args.len()==4 && ["http","file"].contains(&args[3].as_str()) => {
            println!("{}",canada_engine::resident_worker::validate_configuration(Path::new(&args[2]),args[3]=="http")?);
        }
        Some("gpu-primitive-lifetime") if args.len() == 4 => {
            let report=canada_engine::primitives::probe_lifetime(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?,&report)?;
        }
        Some("gpu-sparse-index") if args.len() == 6 => {
            let report=canada_engine::sparse_index::probe(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]))?;
            serde_json::to_writer_pretty(File::create(&args[5])?,&report)?;
            anyhow::ensure!(report["all_exact"]==true,"sparse index differs; evidence saved");
        }
        #[cfg(feature = "protocol")]
        Some("bench-frontend") if args.len() == 5 => {
            let report=canada_engine::protocol::frontend_benchmark(Path::new(&args[2]),Path::new(&args[3]))?;
            serde_json::to_writer_pretty(std::fs::OpenOptions::new().write(true).create_new(true).open(&args[4])?,&report)?;
        }
        #[cfg(feature = "protocol")]
        Some("encode-chat") if args.len() == 5 => {
            let report=canada_engine::protocol::encode_report(Path::new(&args[2]),Path::new(&args[3]))?;
            serde_json::to_writer_pretty(File::create(&args[4])?,&report)?;
        }
        #[cfg(feature = "protocol")]
        Some("decode-tokens") if args.len() == 5 => {
            let report=canada_engine::protocol::decode_report(Path::new(&args[2]),Path::new(&args[3]))?;
            serde_json::to_writer_pretty(File::create(&args[4])?,&report)?;
        }
        #[cfg(feature = "protocol")]
        Some("parse-completion") if args.len() == 6 => {
            let report=canada_engine::protocol::parse_report(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]))?;
            serde_json::to_writer_pretty(File::create(&args[5])?,&report)?;
        }
        Some("gpu-index-gather-probe") if args.len() == 4 => {
            let report=canada_engine::index_gather_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?,&report)?;
        }
        Some("gpu-dspark-query-peer") if args.len() == 4 => {
            let report = canada_engine::dspark_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
        }
        Some("gpu-staged-metadata-probe") if args.len() == 3 => {
            let report = canada_engine::cuda::probe_staged_metadata()?;
            serde_json::to_writer_pretty(File::create(&args[2])?, &report)?;
        }
        Some("gpu-core-module-reuse-probe") if args.len() == 4 => {
            let report=canada_engine::cuda::probe_core_module_reuse(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?,&report)?;
        }
        Some("gpu-blas-workspace-probe") if args.len() == 3 => {
            let report=canada_engine::blas::probe_shared_workspace()?;
            serde_json::to_writer_pretty(File::create(&args[2])?,&report)?;
        }
        Some("gpu-ordering-probe") if args.len() == 4 => {
            let report = canada_engine::cuda::probe_default_order(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_fenced_correct"] == true
                    && report["capture_guard"] == true
                    && report["unfenced_old_value_cases"]
                        .as_u64()
                        .is_some_and(|n| n > 0),
                "stream ordering diagnostic failed; evidence saved"
            );
        }
        Some("gpu-dspark-kv") if args.len() == 4 => {
            let report = canada_engine::dspark_kv_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_exact"] == true,
                "DSpark KV mismatch; evidence saved"
            );
        }
        Some("gpu-dspark-conditioning") if args.len() == 4 => {
            let report = canada_engine::dspark_conditioning::fixture(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_exact"] == true,
                "DSpark conditioning mismatch; evidence saved"
            );
        }
        Some("gpu-dspark-features") if args.len() == 4 => {
            let report = canada_engine::dspark_features::fixture(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_exact"] == true,
                "DSpark target feature mismatch; evidence saved"
            );
        }
        Some("gpu-compression-prefill") if args.len() == 4 => {
            let report = canada_engine::compression_probe::run_prefill(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "prefill compressor differs; evidence saved"
            );
        }
        Some("gpu-prefill-layer-peer") if args.len() == 4 => {
            let report = canada_engine::prefill_layer_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
        }
        Some("gpu-woa-medium") if args.len() == 4 => {
            let report = canada_engine::woa_medium_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_correct"] == true,
                "medium WO-A differs; evidence saved"
            );
        }
        Some("gpu-woa-cache") if args.len() == 4 => {
            let report = canada_engine::woa_cache_probe::run(Path::new(&args[2]))?;
            std::fs::write(&args[3], serde_json::to_vec_pretty(&report)?)?;
        }
        Some("gpu-sampler-graph") if args.len()==4 => {
            let report=canada_engine::probe_sampler_graph(Path::new(&args[2]))?;
            std::fs::write(&args[3],serde_json::to_vec_pretty(&report)?)?;
            anyhow::ensure!(report["all_correct"]==true,"native sampler graph failed");
        }
        Some("gpu-sampler-readout") if args.len()==4 => {
            let report=canada_engine::probe_sampler_readout(Path::new(&args[2]))?;
            std::fs::write(&args[3],serde_json::to_vec_pretty(&report)?)?;
            anyhow::ensure!(report["all_correct"]==true,"native sampler readout failed");
        }
        Some("gpu-categorical") if args.len() == 4 => {
            let report = canada_engine::categorical_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(report["all_correct"] == true, "GPU categorical sampler mismatch; evidence saved");
        }
        Some("gpu-greedy") if args.len() == 4 => {
            let report = canada_engine::sampling_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_correct"] == true,
                "GPU greedy sampler mismatch; evidence saved"
            );
        }
        Some("gpu-target-append-peer") if args.len() == 4 => {
            let report = canada_engine::target_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
        }
        Some("gpu-memory-info") if args.len() == 3 => {
            let cuda = canada_engine::cuda::Cuda::open(0)?;
            let report = serde_json::json!({"memory":cuda.admission_memory()?,"integrated_attribute":cuda.attribute(18)?});
            serde_json::to_writer_pretty(File::create(&args[2])?, &report)?;
        }
        Some("gpu-mqa-score-pool") if args.len() == 5 => {
            let report=canada_engine::mqa_probe::run_score_pool(Path::new(&args[2]),Path::new(&args[3]))?;
            serde_json::to_writer_pretty(File::create(&args[4])?,&report)?;
        }
        Some("gpu-mqa-chunked") if args.len() == 4 => {
            let report = canada_engine::mqa_probe::run_chunked(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "chunked MQA mismatch; diagnostics saved"
            );
        }
        Some("gpu-three-layer-peer") if args.len() == 4 => {
            let report = canada_engine::three_layer_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "live three-layer mismatch; diagnostics saved"
            );
        }
        Some("gpu-candidate-map-check") if args.len() == 4 => {
            let report = canada_engine::candidate_map::probe(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
        }
        Some("gpu-indexed-layer-peer") if args.len() == 4 => {
            let report = canada_engine::indexed_layer_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "indexed layer mismatch; diagnostics saved"
            );
        }
        Some("gpu-compressed-source") if args.len() == 4 => {
            let report = canada_engine::compressed_source_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "compressed source mismatch; diagnostics saved"
            );
        }
        Some("gpu-query-index") if args.len() == 4 => {
            let report = canada_engine::query_index_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_numeric_taps_bitwise_equal"] == true,
                "connected query/index mismatch; diagnostics saved"
            );
        }
        Some("gpu-topk-state") if args.len() == 4 => {
            let report = canada_engine::topk_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_metadata_valid"] == true,
                "topk metadata invalid; diagnostics saved"
            );
        }
        Some("gpu-mqa-state") if args.len() == 4 => {
            let report = canada_engine::mqa_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "MQA scores mismatch; diagnostics saved"
            );
        }
        Some("gpu-prefill-pair") if args.len() == 4 => {
            let report = canada_engine::pair_prefill_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "prefill pair mismatch; diagnostics saved"
            );
        }
        Some("gpu-index-state") if args.len() == 4 => {
            let report = canada_engine::index_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "index state mismatch; diagnostics saved"
            );
        }
        Some("gpu-compression-state") if args.len() == 4 => {
            let report = canada_engine::compression_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "compression state mismatch; diagnostics saved"
            );
        }
        Some("gpu-compressor-projection") if args.len() == 4 => {
            let report = canada_engine::compressor_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "compressor projection differs; diagnostic saved"
            );
        }
        Some("gpu-two-layer-peer") if args.len() == 4 => {
            let report = canada_engine::two_layer_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "connected layer GPU trace mismatch; diagnostic saved"
            );
        }
        Some("gpu-engram-peer") if args.len() == 4 => {
            let report = canada_engine::engram_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "Engram GPU trace mismatch; diagnostic saved"
            );
        }
        Some("gpu-layer0-peer") if args.len() == 4 => {
            let report = canada_engine::layer_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "TP4 layer trace mismatch; diagnostic saved"
            );
        }

        Some("gpu-ffn-peer") if args.len() == 4 => {
            let report = canada_engine::probe::ffn_peer(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "TP4 FFN mismatch; diagnostic saved"
            );
        }

        Some("gpu-draft-head") if args.len() == 4 => {
            let report = canada_engine::draft_head::qualify(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(report["all_exact"] == true, "private draft-head GPU gate failed");
        }
        Some("gpu-model-io-peer") if args.len() == 4 => {
            let report = canada_engine::model_io_probe::run(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "model IO boundary mismatch; diagnostic saved"
            );
        }
        Some("gpu-stage-layer" | "gpu-stage-prefix") if args.len() == 8 => {
            let cfg = ModelConfig::parse(&std::fs::read(&args[2])?)?;
            let cp = Checkpoint::from_manifest(Path::new(&args[3]))?;
            let rank: usize = args[5].parse()?;
            let prefixes = if args[1] == "gpu-stage-prefix" && args[6] == "model-io" {
                vec![
                    "embed.".to_string(),
                    "head.".to_string(),
                    "norm.".to_string(),
                ]
            } else if args[1] == "gpu-stage-prefix" && args[6] == "vision" {
                canada_engine::vision::PREFIXES.iter().map(|s| s.to_string()).collect()
            } else if args[1] == "gpu-stage-prefix" {
                anyhow::ensure!(
                    ["layers.1.engram.", "layers.14.engram.", "mtp."].contains(&args[6].as_str()),
                    "unqualified staging prefix"
                );
                vec![args[6].clone()]
            } else {
                let layer: usize = args[6].parse()?;
                anyhow::ensure!(layer < 40, "staging layer outside model");
                vec![format!("layers.{layer}.")]
            };
            let plan = RankPlan::build(&cp, &cfg, rank, 4)?;
            let cuda = canada_engine::cuda::Cuda::open(0)?;
            cuda.qualify_gb10()?;
            let prefix_refs: Vec<_> = prefixes.iter().map(String::as_str).collect();
            let weights = canada_engine::weights::Weights::load_prefixes(
                &cuda,
                Path::new(&args[4]),
                &cp,
                &plan,
                &prefix_refs,
            )?;
            let report = weights.stage(
                &cp,
                Path::new(&args[4]),
                &plan,
                &prefix_refs,
                Path::new(&args[7]),
            )?;
            println!("{}", report);
        }

        Some("gpu-mhc-checkpoint-probe") if args.len() == 8 => {
            let report = canada_engine::probe::mhc_checkpoint(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
                Path::new(&args[6]),
            )?;
            serde_json::to_writer_pretty(File::create(&args[7])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "mHC GPU reference mismatch; diagnostic report saved"
            );
        }

        Some("gpu-ffn-checkpoint-probe") if args.len() == 9 => {
            let report = canada_engine::probe::ffn_checkpoint(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
                Path::new(&args[6]),
                Path::new(&args[7]),
            )?;
            serde_json::to_writer_pretty(File::create(&args[8])?, &report)?;
            anyhow::ensure!(
                report["all_bitwise_equal"] == true,
                "FFN GPU reference mismatch; diagnostic report saved"
            );
        }

        Some("gpu-attention-compare") if args.len() == 9 => {
            let report = canada_engine::probe::attention_compare(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
                Path::new(&args[6]),
                Path::new(&args[7]),
            )?;
            serde_json::to_writer_pretty(File::create(&args[8])?, &report)?;
            println!("{}", report);
        }

        Some("gpu-attention-checkpoint-probe") if args.len() == 9 || args.len() == 10 => {
            let report = canada_engine::probe::attention_checkpoint(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
                Path::new(&args[6]),
                Path::new(&args[7]),
                canada_engine::layer_attention::InputMode::parse(
                    args.get(9).map(String::as_str).unwrap_or("fused"),
                )?,
            )?;
            serde_json::to_writer_pretty(File::create(&args[8])?, &report)?;
            println!("{}", report);
        }

        Some("gpu-frequency-probe") if args.len() == 5 => {
            let report = canada_engine::probe::frequency(Path::new(&args[2]), Path::new(&args[3]))?;
            serde_json::to_writer_pretty(File::create(&args[4])?, &report)?;
            println!("{}", report);
            anyhow::ensure!(
                report["cases"]
                    .as_array()
                    .is_some_and(|cases| !cases.is_empty()
                        && cases
                            .iter()
                            .all(|c| c["different_values"].as_u64() == Some(0))),
                "native GPU rotary table differs from reference; diagnostic report saved"
            );
        }

        Some("gpu-prelude-probe") if args.len() == 5 => {
            let report = canada_engine::probe::prelude(Path::new(&args[2]), Path::new(&args[3]))?;
            serde_json::to_writer_pretty(File::create(&args[4])?, &report)?;
            println!("{}", report);
        }

        Some("gpu-kv-probe") if args.len() == 5 => {
            let report = canada_engine::probe::kv(Path::new(&args[2]), Path::new(&args[3]))?;
            serde_json::to_writer_pretty(File::create(&args[4])?, &report)?;
            println!("{}", report);
        }

        Some("gpu-attention-probe") if args.len() == 5 => {
            let report = canada_engine::probe::attention(Path::new(&args[2]), Path::new(&args[3]))?;
            serde_json::to_writer_pretty(File::create(&args[4])?, &report)?;
            println!("{}", report);
        }

        Some("gpu-candidate-probe") if args.len() == 6 => {
            let report = canada_engine::candidates::probe(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
            )?;
            serde_json::to_writer_pretty(File::create(&args[5])?, &report)?;
            anyhow::ensure!(report["all_exact"] == true, "candidate blocks differ; evidence saved");
        }
        Some("gpu-moe-checkpoint-probe") if args.len() == 11 => {
            let report = canada_engine::probe::moe_checkpoint(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                args[5].parse()?,
                args[6].parse()?,
                Path::new(&args[7]),
                Path::new(&args[8]),
                Path::new(&args[9]),
            )?;
            serde_json::to_writer_pretty(File::create(&args[10])?, &report)?;
            println!("{}", report);
        }
        Some("gpu-experts-checkpoint-probe") if args.len() == 9 => {
            let cfg = ModelConfig::parse(&std::fs::read(&args[2])?)?;
            let cp = Checkpoint::from_manifest(Path::new(&args[3]))?;
            let root = Path::new(&args[4]);
            let rank = args[5].parse()?;
            let layer = args[6].parse()?;
            let plan = RankPlan::build(&cp, &cfg, rank, 4)?;
            let cuda = canada_engine::cuda::Cuda::open(0)?;
            cuda.qualify_gb10()?;
            let module = cuda.module(&std::fs::read(&args[7])?)?;
            let mut weights = canada_engine::weights::Weights::load(
                &cuda,
                root,
                &cp,
                &plan,
                &format!("layers.{layer}.ffn.experts."),
            )?;
            let source = weights.qualify(&cp, root, &plan)?;
            let start = std::time::Instant::now();
            let prepared =
                canada_engine::expert_packing::prepare_layer(&cuda, &module, &mut weights, layer)?;
            let seconds = start.elapsed().as_secs_f64();
            anyhow::ensure!(
                weights.expert_ready(layer),
                "expert readiness transition failed"
            );
            anyhow::ensure!(
                canada_engine::expert_packing::prepare_layer(&cuda, &module, &mut weights, layer)
                    .is_err(),
                "expert double preparation was allowed"
            );
            let hashes = canada_engine::expert_packing::prepared_hashes(&weights, &prepared)?;
            let report = serde_json::json!({"source":source,"prepared":prepared,"prepare_seconds":seconds,"gpu_prepared_sha256":hashes,"double_preparation_rejected":true});
            serde_json::to_writer_pretty(File::create(&args[8])?, &report)?;
            println!("{}", report);
        }
        Some("pack-expert") if args.len() == 10 => {
            let source = std::fs::read(&args[2])?;
            let n = args[3].parse()?;
            let k = args[4].parse()?;
            let group = args[5].parse()?;
            let scales = std::fs::read(&args[6])?;
            let (w, s) = canada_engine::expert_packing::pack_gpu(
                Path::new(&args[9]),
                &source,
                &scales,
                n,
                k,
                group,
            )?;
            std::fs::write(&args[7], w)?;
            std::fs::write(&args[8], s)?;
        }
        Some("gpu-fp8-checkpoint-probe") if args.len() == 10 => {
            let report = canada_engine::probe::fp8_checkpoint(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                args[5].parse()?,
                &args[6],
                Path::new(&args[7]),
                Path::new(&args[8]),
            )?;
            serde_json::to_writer_pretty(File::create(&args[9])?, &report)?;
            println!("{}", report);
        }
        Some("pack-scales") if args.len() == 7 => {
            let packed = canada_engine::blockscaled::pack_scales_gpu(
                &std::fs::read(&args[2])?,
                args[3].parse()?,
                args[4].parse()?,
                Path::new(&args[6]),
            )?;
            std::fs::write(&args[5], packed)?;
        }
        Some("gpu-weights-probe") if args.len() == 8 => {
            let cfg = ModelConfig::parse(&std::fs::read(&args[2])?)?;
            let cp = Checkpoint::from_manifest(Path::new(&args[3]))?;
            let root = Path::new(&args[4]);
            let rank = args[5].parse()?;
            let plan = RankPlan::build(&cp, &cfg, rank, 4)?;
            let cuda = canada_engine::cuda::Cuda::open(0)?;
            cuda.qualify_gb10()?;
            let start = std::time::Instant::now();
            let weights = canada_engine::weights::Weights::load(&cuda, root, &cp, &plan, &args[6])?;
            let seconds = start.elapsed().as_secs_f64();
            let mut report = weights.qualify(&cp, root, &plan)?;
            report["load_seconds"] = serde_json::json!(seconds);
            serde_json::to_writer_pretty(File::create(&args[7])?, &report)?;
            println!("{}", report);
        }
        Some("gpu-program-probe") if args.len() == 4 => {
            let report = canada_engine::program::qualify(Path::new(&args[2]))?;
            serde_json::to_writer_pretty(File::create(&args[3])?, &report)?;
            println!("{}", report);
        }
        Some("hash-fixtures") if args.len() == 5 => {
            use canada_engine::hash_history::{HashMetadata, History};
            #[derive(serde::Deserialize)]
            struct Step {
                tokens: Vec<u32>,
                accepted: usize,
            }
            #[derive(serde::Deserialize)]
            struct Case {
                id: String,
                prefix: Vec<u32>,
                steps: Vec<Step>,
            }
            let m = HashMetadata::parse(&std::fs::read(&args[2])?)?;
            let cases: Vec<Case> = serde_json::from_reader(File::open(&args[3])?)?;
            let mut output = Vec::new();
            for case in cases {
                let mut h = History::new(1);
                let p = h.prepare(&case.prefix, &m)?;
                h.commit(p, case.prefix.len())?;
                let mut phases = Vec::new();
                for s in case.steps {
                    let p = h.prepare(&s.tokens, &m)?;
                    phases.push(serde_json::json!(p.rows));
                    h.commit(p, s.accepted)?;
                }
                output.push(serde_json::json!({"id":case.id,"rows":phases,"history":h}));
            }
            serde_json::to_writer(File::create(&args[4])?, &output)?;
        }
        Some("gpu-gather-probe") if args.len() == 7 || args.len() == 8 => {
            let r = canada_engine::gather_probe::run(
                args[2].parse()?,
                &args[3],
                &std::fs::read(&args[4])?,
                Path::new(&args[5]),
                args.get(7).map(Path::new),
            )?;
            serde_json::to_writer_pretty(File::create(&args[6])?, &r)?;
        }
        Some("gpu-logit-gather-probe") if args.len() == 8 => {
            let result = canada_engine::logit_gather_probe::run(
                args[2].parse()?, &args[3], &std::fs::read(&args[4])?,
                Path::new(&args[5]), Path::new(&args[6]),
            )?;
            serde_json::to_writer_pretty(File::create(&args[7])?, &result)?;
        }
        Some("gpu-logit-gather-grid-probe") if args.len() == 8 => {
            let result = canada_engine::logit_gather_probe::run_grid(
                args[2].parse()?, &args[3], &std::fs::read(&args[4])?, Path::new(&args[5]),
            )?;
            serde_json::to_writer_pretty(File::create(&args[7])?, &result)?;
        }
        Some("gpu-logit-gather-last-probe") if args.len() == 8 => {
            let result = canada_engine::logit_gather_probe::run_last(
                args[2].parse()?, &args[3], &std::fs::read(&args[4])?, Path::new(&args[5]),
            )?;
            serde_json::to_writer_pretty(File::create(&args[7])?, &result)?;
        }
        Some("gpu-sharded-categorical-probe") if args.len() == 8 => {
            let result = canada_engine::sharded_categorical_probe::run(
                args[2].parse()?, &args[3], &std::fs::read(&args[4])?,
                Path::new(&args[5]), Path::new(&args[6]),
            )?;
            serde_json::to_writer_pretty(File::create(&args[7])?, &result)?;
        }
        Some("gpu-hierarchical-categorical-probe") if args.len()==8=>{
            let result=canada_engine::hierarchical_categorical::run(args[2].parse()?,&args[3],
                &std::fs::read(&args[4])?,Path::new(&args[5]),Path::new(&args[6]))?;
            serde_json::to_writer_pretty(File::create(&args[7])?,&result)?;
        }
        Some("gpu-paired-shared-up-probe") if args.len()==8 => {
            let result=canada_engine::fp8_linear::paired_shared_up_probe(
                Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]),
                Path::new(&args[5]),Path::new(&args[6]))?;
            serde_json::to_writer_pretty(File::create(&args[7])?,&result)?;
        }
        Some("gpu-paired-qkv-probe") if args.len()==8 => {
            let result=canada_engine::fp8_linear::paired_qkv_probe(
                Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]),
                Path::new(&args[5]),Path::new(&args[6]))?;
            serde_json::to_writer_pretty(File::create(&args[7])?,&result)?;
        }
        Some("gpu-sum-grid-probe") if args.len() == 8 => {
            let result = canada_engine::sum_grid_probe::run(
                args[2].parse()?, &args[3], &std::fs::read(&args[4])?, Path::new(&args[5]),
            )?;
            serde_json::to_writer_pretty(File::create(&args[7])?, &result)?;
        }
        Some("gpu-posting-probe") if args.len() == 8 => {
            let result=canada_engine::posting_probe::run(args[2].parse()?,&args[3],
                &std::fs::read(&args[4])?,Path::new(&args[5]),Path::new(&args[6]))?;
            serde_json::to_writer_pretty(File::create(&args[7])?,&result)?;
        }
        Some("gpu-sum-sequence-probe") if args.len() == 8 => {
            let result = canada_engine::sum_grid_probe::run_sequence(
                args[2].parse()?, &args[3], &std::fs::read(&args[4])?,
                Path::new(&args[5]), Path::new(&args[6]),
            )?;
            serde_json::to_writer_pretty(File::create(&args[7])?, &result)?;
        }
        Some("gpu-vision-placement-probe") if args.len() == 3 => {
            let report=canada_engine::worker_media::placement_probe(Path::new(&args[2]))?;
            println!("{}",serde_json::json!({"rank":report["rank"],"placement_exact":report["placement_exact"]}));
        }
        Some("gpu-transport-probe") if args.len() == 8 => {
            let result = canada_engine::collective_probe::run(
                args[2].parse()?, &args[3], &std::fs::read(&args[4])?,
                Path::new(&args[5]), Path::new(&args[6]),
            )?;
            serde_json::to_writer_pretty(File::create(&args[7])?, &result)?;
        }
        Some("gpu-roce-probe") if args.len() == 7 => {
            let r = canada_engine::probe::roce(
                args[2].parse()?,
                &args[3],
                &std::fs::read(&args[4])?,
                Path::new(&args[5]),
            )?;
            serde_json::to_writer_pretty(File::create(&args[6])?, &r)?;
            println!("{}", r);
        }
        Some("gpu-collective-probe") if args.len() == 6 => {
            let result = canada_engine::probe::collective(
                args[2].parse()?,
                &args[3],
                &std::fs::read(&args[4])?,
            )?;
            serde_json::to_writer_pretty(File::create(&args[5])?, &result)?;
            println!("{}", result);
        }
        Some("gpu-linear-probe") if args.len() == 8 => {
            let result = canada_engine::probe::linear(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                args[5].parse()?,
                args[6].parse()?,
                args[7].parse()?,
            )?;
            println!("{}", result);
        }
        Some("scan") if args.len() == 4 => {
            let c = Checkpoint::scan(Path::new(&args[2]))?;
            serde_json::to_writer(File::create(&args[3])?, &c)?;
            println!(
                "{}",
                serde_json::json!({"shards":c.files.len(),"tensors":c.tensors.len()})
            );
        }
        Some("plan") if (4..=5).contains(&args.len()) => {
            let cfg = ModelConfig::parse(&std::fs::read(&args[2])?)?;
            let cp = Checkpoint::from_manifest(Path::new(&args[3]))?;
            let mut plans = Vec::new();
            for rank in 0..4 {
                let p = RankPlan::build(&cp, &cfg, rank, 4)?;
                println!("{}", p.summary());
                plans.push(p);
            }
            if let Some(out) = args.get(4) {
                serde_json::to_writer(File::create(out)?, &plans)?;
            }
        }
        _ => bail!(
            "usage: canada-engine scan MODEL_ROOT MANIFEST | plan CONFIG MANIFEST [PLANS] | gpu-attention-probe KERNEL_ROOT GPU_FIXTURES REPORT"
        ),
    }
    Ok(())
}
