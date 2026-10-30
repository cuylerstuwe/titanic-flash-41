//! Equal-rank tensor parallel layout, in the checkpoint's original packed units.
use crate::{
    checkpoint::{Checkpoint, Dtype, Tensor},
    config::ModelConfig,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Placement {
    Replicated,
    /// A contiguous range of an axis, expressed in STORED elements (including FP4 packing).
    Shard {
        axis: usize,
        start: u64,
        len: u64,
    },
    Omitted {
        reason: String,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightPlan {
    pub placement: Placement,
    pub local_shape: Vec<u64>,
    pub local_bytes: u64,
    pub host_resident: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankPlan {
    pub rank: usize,
    pub world: usize,
    pub weights: BTreeMap<String, WeightPlan>,
    pub device_weight_bytes: u64,
    pub host_engram_bytes: u64,
    pub omitted_bytes: u64,
    pub local_query_heads: usize,
    pub local_output_groups: usize,
    pub local_expert_width: usize,
    pub local_vocabulary_rows: usize,
}

fn split(t: &Tensor, axis: usize, rank: usize, world: usize) -> Result<Placement> {
    let width = *t.shape.get(axis).context("shard axis outside shape")?;
    ensure!(
        width % world as u64 == 0,
        "non-divisible tensor parallel axis"
    );
    Ok(Placement::Shard {
        axis,
        start: width * rank as u64 / world as u64,
        len: width / world as u64,
    })
}

fn classify(
    name: &str,
    t: &Tensor,
    config: &ModelConfig,
    rank: usize,
    world: usize,
) -> Result<Placement> {
    let text = &config.text_config;
    if name == "embed.weight" || name == "head.weight" {
        return split(t, 0, rank, world);
    }
    let layer = name
        .strip_prefix("layers.")
        .and_then(|s| s.split('.').next())
        .and_then(|s| s.parse::<usize>().ok());
    let is_mtp = name.starts_with("mtp.");
    if is_mtp {
        let stage = name
            .split('.')
            .nth(1)
            .and_then(|s| s.parse::<usize>().ok())
            .context("invalid DSpark stage namespace")?;
        ensure!(
            stage < text.num_nextn_predict_layers,
            "unknown DSpark stage"
        );
        // The qualified Markov path follows the shared vocabulary head's TP
        // partition. The previous-token embedding remains replicated, avoiding
        // another latency-sensitive collective for its256-value latent.
        if name == format!("mtp.{stage}.markov_head.head.weight") {
            return split(t, 0, rank, world);
        }
    }
    if let Some(layer) = layer {
        ensure!(layer < text.num_hidden_layers, "unknown target layer");
        if name.contains(".attn.compressor.") && !text.kv_source_layer_ids.contains(&layer) {
            return Ok(Placement::Omitted {
                reason: "non-source compressor is absent from 1017 graph".into(),
            });
        }
        if name.contains(".attn.indexer.") {
            if !text.index_source_layer_ids.contains(&layer)
                || (!text.kv_source_layer_ids.contains(&layer)
                    && (name.contains(".wk.") || name.contains(".k_norm.")))
            {
                return Ok(Placement::Omitted {
                    reason: "shared sparse source owns these keys".into(),
                });
            }
        }
    }
    if name.contains(".engram.embed.") {
        // Unequal last rows are intentional: checkpoint table sizes need not divide by four.
        let rows = t.shape[0];
        let start = rows * rank as u64 / world as u64;
        let end = rows * (rank + 1) as u64 / world as u64;
        return Ok(Placement::Shard {
            axis: 0,
            start,
            len: end - start,
        });
    }
    if layer.is_some() || is_mtp {
        // Qualified replicated-split Engram projection: row-shard the stored
        // FP8 values and their 32-row scale blocks, then bit-preserving gather.
        if name.contains(".engram.wkv.") {
            return split(t, 0, rank, world);
        }
        if name.contains(".attn.wq_b.")
            || name.contains(".attn.wo_a.")
            || name.ends_with(".attn.attn_sink")
        {
            return split(t, 0, rank, world);
        }
        if name.contains(".attn.wo_b.") {
            return split(t, 1, rank, world);
        }
        if name.contains(".ffn.experts.") || name.contains(".ffn.shared_experts.") {
            if name.contains(".w1.") || name.contains(".w3.") {
                return split(t, 0, rank, world);
            }
            if name.contains(".w2.") {
                return split(t, 1, rank, world);
            }
        }
    }
    Ok(Placement::Replicated)
}

impl RankPlan {
    pub fn build(c: &Checkpoint, config: &ModelConfig, rank: usize, world: usize) -> Result<Self> {
        config.validate()?;
        ensure!(
            world == 4 && rank < world,
            "this qualification targets exactly four peer ranks"
        );
        validate_target_weights(c, config)?;
        validate_draft_weights(c, config)?;
        let mut plan = Self {
            rank,
            world,
            weights: BTreeMap::new(),
            device_weight_bytes: 0,
            host_engram_bytes: 0,
            omitted_bytes: 0,
            local_query_heads: 16,
            local_output_groups: 2,
            local_expert_width: 576,
            local_vocabulary_rows: 32320,
        };
        for (name, t) in &c.tensors {
            let placement = classify(name, t, config, rank, world).with_context(|| name.clone())?;
            let mut shape = t.shape.clone();
            let bytes = match placement {
                Placement::Omitted { .. } => 0,
                Placement::Replicated => t.bytes(),
                Placement::Shard { axis, len, .. } => {
                    shape[axis] = len;
                    shape
                        .iter()
                        .try_fold(t.dtype.bytes(), |n, d| n.checked_mul(*d))
                        .context("local size overflow")?
                }
            };
            let host = name.contains(".engram.embed.");
            if matches!(placement, Placement::Omitted { .. }) {
                plan.omitted_bytes += t.bytes();
            } else if host {
                plan.host_engram_bytes += bytes;
            } else {
                plan.device_weight_bytes += bytes;
            }
            plan.weights.insert(
                name.clone(),
                WeightPlan {
                    placement,
                    local_shape: shape,
                    local_bytes: bytes,
                    host_resident: host,
                },
            );
        }
        Ok(plan)
    }
    pub fn summary(&self) -> serde_json::Value {
        let draft_bytes = self
            .weights
            .iter()
            .filter(|(n, _)| n.starts_with("mtp."))
            .map(|(_, v)| v.local_bytes)
            .sum::<u64>();
        serde_json::json!({"rank":self.rank,"world":self.world,
            "device_weight_bytes":self.device_weight_bytes,"host_engram_bytes":self.host_engram_bytes,
            "engram_backing_is_fully_resident":false,
            "baseline_engram_cache_bytes":4u64*1024*1024*1024,
            "resident_weights_and_row_cache_bytes":self.device_weight_bytes+4u64*1024*1024*1024,
            "omitted_global_bytes":self.omitted_bytes,"query_heads":self.local_query_heads,
            "output_groups":self.local_output_groups,"expert_width":self.local_expert_width,
            "vocabulary_rows":self.local_vocabulary_rows,
            "planned_draft_weight_bytes":draft_bytes,"draft_stages_header_validated":3})
    }
}

pub fn validate_target_weights(c: &Checkpoint, config: &ModelConfig) -> Result<()> {
    let d = config.text_config.hidden_size as u64;
    c.require("embed.weight", Dtype::Bf16, &[129280, d])?;
    c.require("head.weight", Dtype::Bf16, &[129280, d])?;
    c.require("norm.weight", Dtype::Bf16, &[d])?;
    for l in 0..40 {
        let p = format!("layers.{l}");
        for branch in ["attn", "ffn"] {
            c.require(&format!("{p}.hc_{branch}_fn"), Dtype::F32, &[24, 4 * d])?;
            c.require(&format!("{p}.hc_{branch}_base"), Dtype::F32, &[24])?;
            c.require(&format!("{p}.hc_{branch}_scale"), Dtype::F32, &[3])?;
            c.require(&format!("{p}.{branch}_norm.weight"), Dtype::Bf16, &[d])?;
        }
        for (name, n, k) in [
            ("wq_a", 1280, d),
            ("wq_b", 32768, 1280),
            ("wkv", 512, d),
            ("wo_a", 8192, 4096),
            ("wo_b", d, 8192),
        ] {
            c.require(&format!("{p}.attn.{name}.weight"), Dtype::Fp8, &[n, k])?;
            c.require(
                &format!("{p}.attn.{name}.scale"),
                Dtype::E8m0,
                &[n / 32, k / 32],
            )?;
        }
        for e in 0..384 {
            for (name, n, k) in [("w1", 2304, d), ("w3", 2304, d), ("w2", d, 2304)] {
                c.require(
                    &format!("{p}.ffn.experts.{e}.{name}.weight"),
                    Dtype::PackedFp4,
                    &[n, k / 2],
                )?;
                c.require(
                    &format!("{p}.ffn.experts.{e}.{name}.scale"),
                    Dtype::E8m0,
                    &[n, k / 32],
                )?;
            }
        }
        for (name, n, k) in [("w1", 2304, d), ("w3", 2304, d), ("w2", d, 2304)] {
            c.require(
                &format!("{p}.ffn.shared_experts.{name}.weight"),
                Dtype::Fp8,
                &[n, k],
            )?;
            c.require(
                &format!("{p}.ffn.shared_experts.{name}.scale"),
                Dtype::E8m0,
                &[n / 32, k / 32],
            )?;
        }
        c.require(&format!("{p}.ffn.gate.weight"), Dtype::Bf16, &[384, d])?;
        c.require(&format!("{p}.ffn.gate.bias"), Dtype::F32, &[384])?;
        c.require(&format!("{p}.ffn.gate.bias_vl"), Dtype::F32, &[384])?;
    }
    for (&l, &n) in config
        .text_config
        .engram_layer_ids
        .iter()
        .zip(&config.text_config.engram_num_embeddings)
    {
        c.require(
            &format!("layers.{l}.engram.embed.weight"),
            Dtype::Fp8,
            &[n as u64, 256],
        )?;
        c.require(
            &format!("layers.{l}.engram.embed.scale"),
            Dtype::E8m0,
            &[n as u64, 8],
        )?;
    }
    Ok(())
}

/// Validate all three canonical DSpark stages before allocation/staging. This
/// checks checkpoint metadata only; numeric packing and execution stay on GPU.
pub fn validate_draft_weights(c: &Checkpoint, config: &ModelConfig) -> Result<()> {
    let t = &config.text_config;
    let d = t.hidden_size as u64;
    for stage in 0..t.num_nextn_predict_layers {
        let p = format!("mtp.{stage}");
        for branch in ["attn", "ffn"] {
            c.require(&format!("{p}.hc_{branch}_fn"), Dtype::F32, &[24, 4 * d])?;
            c.require(&format!("{p}.hc_{branch}_base"), Dtype::F32, &[24])?;
            c.require(&format!("{p}.hc_{branch}_scale"), Dtype::F32, &[3])?;
            c.require(&format!("{p}.{branch}_norm.weight"), Dtype::Bf16, &[d])?;
        }
        for (name, n, k) in [
            ("wq_a", 1280, d),
            ("wq_b", 32768, 1280),
            ("wkv", 512, d),
            ("wo_a", 8192, 4096),
            ("wo_b", d, 8192),
        ] {
            c.require(&format!("{p}.attn.{name}.weight"), Dtype::Fp8, &[n, k])?;
            c.require(
                &format!("{p}.attn.{name}.scale"),
                Dtype::E8m0,
                &[n / 32, k / 32],
            )?;
        }
        c.require(&format!("{p}.attn.attn_sink"), Dtype::F32, &[64])?;
        c.require(&format!("{p}.attn.q_norm.weight"), Dtype::Bf16, &[1280])?;
        c.require(&format!("{p}.attn.kv_norm.weight"), Dtype::Bf16, &[512])?;
        for expert in 0..t.dspark_n_routed_experts {
            for (name, n, k) in [("w1", 2304, d), ("w3", 2304, d), ("w2", d, 2304)] {
                c.require(
                    &format!("{p}.ffn.experts.{expert}.{name}.weight"),
                    Dtype::PackedFp4,
                    &[n, k / 2],
                )?;
                c.require(
                    &format!("{p}.ffn.experts.{expert}.{name}.scale"),
                    Dtype::E8m0,
                    &[n, k / 32],
                )?;
            }
        }
        for (name, n, k) in [("w1", 2304, d), ("w3", 2304, d), ("w2", d, 2304)] {
            c.require(
                &format!("{p}.ffn.shared_experts.{name}.weight"),
                Dtype::Fp8,
                &[n, k],
            )?;
            c.require(
                &format!("{p}.ffn.shared_experts.{name}.scale"),
                Dtype::E8m0,
                &[n / 32, k / 32],
            )?;
        }
        c.require(&format!("{p}.ffn.gate.weight"), Dtype::Bf16, &[128, d])?;
        c.require(&format!("{p}.ffn.gate.bias"), Dtype::F32, &[128])?;
    }
    c.require("mtp.0.main_proj.weight", Dtype::Fp8, &[d, 3 * d])?;
    c.require("mtp.0.main_proj.scale", Dtype::E8m0, &[d / 32, 3 * d / 32])?;
    c.require("mtp.0.main_norm.weight", Dtype::Bf16, &[d])?;
    c.require("mtp.2.norm.weight", Dtype::Bf16, &[d])?;
    for name in ["embed", "head"] {
        c.require(
            &format!("mtp.2.markov_head.{name}.weight"),
            Dtype::Bf16,
            &[129280, 256],
        )?;
    }
    c.require("mtp.2.confidence_head.proj.weight", Dtype::Bf16, &[1, 5376])?;
    Ok(())
}

/// Copy spans for a packed rank shard. Strided column slices stream row by row.
/// Returned offsets are relative to the tensor, never to an unvalidated pointer.
pub fn copy_spans(t: &Tensor, w: &WeightPlan) -> Result<Vec<(u64, u64)>> {
    match w.placement {
        Placement::Omitted { .. } => Ok(Vec::new()),
        Placement::Replicated => Ok(vec![(0, t.bytes())]),
        Placement::Shard { axis, start, len } => {
            let inner = t.shape[axis + 1..].iter().product::<u64>() * t.dtype.bytes();
            let outer = t.shape[..axis].iter().product::<u64>();
            ensure!(start + len <= t.shape[axis], "invalid local shard");
            Ok((0..outer)
                .map(|i| ((i * t.shape[axis] + start) * inner, len * inner))
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn draft_markov_head_matches_vocabulary_shards_and_keeps_latents_local() {
        let c = ModelConfig::parse(include_bytes!("../tests/fixtures/1017-config.json")).unwrap();
        let t = Tensor {
            file: "x.safetensors".into(),
            dtype: Dtype::Bf16,
            shape: vec![129280, 256],
            data_offsets: [0, 129280 * 256 * 2],
        };
        for rank in 0..4 {
            assert_eq!(
                classify("mtp.2.markov_head.head.weight", &t, &c, rank, 4).unwrap(),
                Placement::Shard {
                    axis: 0,
                    start: rank as u64 * 32320,
                    len: 32320
                }
            );
            assert_eq!(
                classify("mtp.2.markov_head.embed.weight", &t, &c, rank, 4).unwrap(),
                Placement::Replicated
            );
        }
        assert!(classify("mtp.3.attn.wq_a.weight", &t, &c, 0, 4).is_err());
    }
    #[test]
    fn packed_down_projection_has_no_padding_or_dropped_scale_group() {
        let c = ModelConfig::parse(include_bytes!("../tests/fixtures/1017-config.json")).unwrap();
        for (dtype, cols) in [(Dtype::PackedFp4, 1152), (Dtype::E8m0, 72)] {
            let t = Tensor {
                file: "x.safetensors".into(),
                dtype,
                shape: vec![5120, cols],
                data_offsets: [0, 5120 * cols],
            };
            let mut ranges = Vec::new();
            for rank in 0..4 {
                let p = classify("layers.0.ffn.experts.0.w2.weight", &t, &c, rank, 4).unwrap();
                let w = WeightPlan {
                    placement: p,
                    local_shape: vec![5120, cols / 4],
                    local_bytes: t.bytes() / 4,
                    host_resident: false,
                };
                let rr = copy_spans(&t, &w).unwrap();
                assert_eq!(rr.iter().map(|(_, n)| n).sum::<u64>(), t.bytes() / 4);
                ranges.extend(rr);
            }
            ranges.sort_unstable();
            let mut end = 0;
            for (s, n) in ranges {
                assert_eq!(s, end);
                end = s + n;
            }
            assert_eq!(end, t.bytes());
        }
    }
}
