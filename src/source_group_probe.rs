//! Source8-group differential diagnostics. Fixture tensors are expected values
//! and controlled initial state, never hidden-state inputs to the live group.
use crate::{
    attention::Candidates,
    compressed_source::Metadata,
    cuda::Cuda,
    kv::RowLocations,
    layer_probe::{Cases, bf16_bytes, compare, fixture, snapshots},
    prelude::Positions,
    source_group::Group,
};
use anyhow::{Result, ensure};
use half::bf16;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};
pub(crate) trait State {
    fn reset_swa(&mut self, seed: &[u8]) -> Result<()>;
    fn source(&self) -> &crate::compressed_source::Layer;
    fn source_mut(&mut self) -> &mut crate::compressed_source::Layer;
    fn layers(&self) -> &[crate::decoder::SwaLayer];
    fn caches(&self) -> &[crate::attention::PagedKv];
    fn engram(&self) -> Option<(&crate::engram_gpu::Layer, &crate::engram_gpu::Rows)> {
        None
    }
    fn index(&self, _layer: usize) -> Option<&crate::compressed_source::ReadIndex> {
        None
    }
}
impl State for Group {
    fn reset_swa(&mut self, seed: &[u8]) -> Result<()> {
        Group::reset_swa(self, seed)
    }
    fn source(&self) -> &crate::compressed_source::Layer {
        Group::source(self)
    }
    fn source_mut(&mut self) -> &mut crate::compressed_source::Layer {
        Group::source_mut(self)
    }
    fn layers(&self) -> &[crate::decoder::SwaLayer] {
        Group::layers(self)
    }
    fn caches(&self) -> &[crate::attention::PagedKv] {
        Group::caches(self)
    }
    fn engram(&self) -> Option<(&crate::engram_gpu::Layer, &crate::engram_gpu::Rows)> {
        Group::engram(self)
    }
}
impl State for crate::late_group::Group {
    fn reset_swa(&mut self, seed: &[u8]) -> Result<()> {
        crate::late_group::Group::reset_swa(self, seed)
    }
    fn source(&self) -> &crate::compressed_source::Layer {
        crate::late_group::Group::source(self)
    }
    fn source_mut(&mut self) -> &mut crate::compressed_source::Layer {
        crate::late_group::Group::source_mut(self)
    }
    fn layers(&self) -> &[crate::decoder::SwaLayer] {
        crate::late_group::Group::layers(self)
    }
    fn caches(&self) -> &[crate::attention::PagedKv] {
        crate::late_group::Group::caches(self)
    }
    fn index(&self, layer: usize) -> Option<&crate::compressed_source::ReadIndex> {
        crate::late_group::Group::index(self, layer)
    }
}
pub(crate) struct Fixtures {
    start: usize,
    roots: Vec<PathBuf>,
    cases: Vec<Cases>,
    slots: Vec<i32>,
    seed: Vec<u8>,
}
pub(crate) struct Step {
    pub positions: Positions,
    pub locations: RowLocations,
    pub candidates: Candidates,
    pub metadata: Metadata,
    source_lengths: Vec<i32>,
    pub expected_input: Vec<u8>,
    pub expected_pre: Vec<u8>,
    pub expected_hashes: Option<Vec<u8>>,
}
fn ints(b: &[u8]) -> Result<Vec<i32>> {
    ensure!(b.len() % 4 == 0, "group integer bytes");
    Ok(b.chunks_exact(4)
        .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
        .collect())
}
impl Fixtures {
    pub fn new(roots: &[PathBuf], start: usize) -> Result<Self> {
        ensure!(
            ([8, 14].contains(&start) && roots.len() == 6)
                || (start == 20 && [4, 8, 12, 16, 20].contains(&roots.len())),
            "ratio2 group fixture count"
        );
        let mut cases = Vec::new();
        let mut slots = Vec::new();
        for (i, root) in roots.iter().enumerate() {
            let manifest: Value = serde_json::from_slice(&std::fs::read(root.join("index.json"))?)?;
            let c: Cases = serde_json::from_value(manifest.clone())?;
            ensure!(
                (c.rows, c.world, c.layer, c.frequency_rows) == (6, 4, i + start, 1048576)
                    && c.cases.len() == 4,
                "source8 group fixture geometry"
            );
            if i == 0 {
                slots = serde_json::from_value(manifest["slots"].clone())?;
                ensure!(
                    (
                        slots.len(),
                        manifest["kv_pages"].as_u64(),
                        manifest["index_pages"].as_u64()
                    ) == if start == 20 {
                        (134, Some(8), Some(4))
                    } else {
                        (67, Some(4), Some(2))
                    },
                    "source8 source fixture layout"
                );
            }
            cases.push(c);
        }
        let seed = std::fs::read(roots[0].join("initial_cache.bin"))?;
        Ok(Self {
            start,
            roots: roots.to_vec(),
            cases,
            slots,
            seed,
        })
    }
    pub fn slots(&self) -> &[i32] {
        &self.slots
    }
    pub fn prepare<G: State>(&self, cuda: &Cuda, group: &mut G, name: &str) -> Result<Step> {
        let c = self.cases[0]
            .cases
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| anyhow::anyhow!("source8 phase absent"))?;
        let root = self.roots[0].join(name);
        let data: BTreeMap<_, _> = c
            .source
            .iter()
            .map(|t| Ok((t.name.as_str(), fixture(&root, t)?)))
            .collect::<Result<_>>()?;
        ensure!(
            data.len()
                == match self.start {
                    14 => 15,
                    20 => 13,
                    _ => 14,
                },
            "source group input metadata count"
        );
        group.reset_swa(&self.seed)?;
        group
            .source_mut()
            .source_mut()
            .cache_mut()
            .upload(&data["seed_cache"])?;
        group.source_mut().keys_mut().upload(&data["seed_keys"])?;
        if let Some(data_ring) = data.get("seed_ring") {
            group.source_mut().source_mut().ring_mut().unwrap().upload(
                &data_ring
                    .chunks_exact(4)
                    .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                    .collect::<Vec<_>>(),
            )?;
        } else {
            ensure!(
                group.source().source().ring().is_none(),
                "ring seed absent from ratio2 source"
            );
        }
        group.source_mut().source_mut().latent_mut().upload(
            &data["seed_latent"]
                .chunks_exact(2)
                .map(|x| bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap())))
                .collect::<Vec<_>>(),
        )?;
        let pos = ints(&data["positions"])?;
        let loc = ints(&data["locations"])?;
        let source_lengths = ints(&data["source_lengths"])?;
        let raw: Vec<i64> = data["source_raw"]
            .chunks_exact(8)
            .map(|x| i64::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let positions = Positions::upload(cuda, &pos, 1048576)?;
        let locations = RowLocations::upload(cuda, &pos, &loc, 256, 1048576)?;
        let candidates = Candidates::upload(
            cuda,
            6,
            128,
            256,
            &ints(&data["ids"])?,
            &ints(&data["lengths"])?,
        )?;
        let metadata = Metadata::new(
            cuda,
            group.source(),
            &pos,
            &[0; 6],
            &raw,
            &self.slots,
            &source_lengths,
            1048576,
        )?;
        // Every consumer must see the same admitted rows and cache locations.
        for (i, cases) in self.cases.iter().enumerate().skip(1) {
            let case = cases
                .cases
                .iter()
                .find(|c| c.name == name)
                .ok_or_else(|| anyhow::anyhow!("source8 consumer phase absent"))?;
            for key in ["positions", "locations", "ids", "lengths"] {
                let tap = case
                    .source
                    .iter()
                    .find(|t| t.name == key)
                    .ok_or_else(|| anyhow::anyhow!("consumer metadata missing"))?;
                ensure!(
                    fixture(&self.roots[i].join(name), tap)? == data[key],
                    "source8 consumer metadata mismatch"
                );
            }
        }
        Ok(Step {
            positions,
            locations,
            candidates,
            metadata,
            source_lengths,
            expected_input: data["input"].clone(),
            expected_pre: data["previous_pre"].clone(),
            expected_hashes: data.get("hash_ids").cloned(),
        })
    }
    pub fn compare<G: State>(
        &self,
        cuda: &Cuda,
        group: &G,
        step: &Step,
        name: &str,
        rank: usize,
    ) -> Result<(Vec<Value>, usize)> {
        ensure!(group.source().error()? == 0, "source8 candidate fault");
        let mut traces = Vec::new();
        let mut different = 0;
        for (i, layer) in group.layers().iter().enumerate() {
            let c = self.cases[i]
                .cases
                .iter()
                .find(|c| c.name == name)
                .ok_or_else(|| anyhow::anyhow!("group trace phase"))?;
            let root = self.roots[i].join(name);
            if i > 0 {
                let input = c
                    .source
                    .iter()
                    .find(|t| t.name == "input")
                    .ok_or_else(|| anyhow::anyhow!("consumer input tap"))?;
                ensure!(
                    bf16_bytes(group.layers()[i - 1].output())? == fixture(&root, input)?,
                    "source8 group live handoff mismatch"
                );
            }
            let (mut local, mut global) = snapshots(cuda, layer, &group.caches()[i])?;
            if i == 0 {
                if let Some((engram, rows)) = group.engram() {
                    local.insert("engram_local_embedding".into(), bf16_bytes(rows.local())?);
                    local.insert(
                        "engram_local_projection".into(),
                        bf16_bytes(engram.projected_local())?,
                    );
                    for (n, b) in [
                        ("engram_embedding", engram.embedding()),
                        ("engram_projected", engram.projected()),
                        ("engram_output", engram.output()),
                    ] {
                        global.insert(n.into(), bf16_bytes(b)?);
                    }
                }
                for (n, b) in crate::compressed_source_probe::snapshots(
                    group.source(),
                    self.slots.len(),
                    &step.source_lengths,
                )? {
                    global.insert(format!("source_{n}"), b);
                }
            }
            if let Some(index) = group.index(self.start + i) {
                ensure!(index.error()? == 0, "read-index candidate fault");
                for (n, b) in crate::compressed_source_probe::query_snapshots(
                    index.query(),
                    index.candidates(),
                    self.slots.len(),
                    &step.source_lengths,
                )? {
                    global.insert(format!("index_{n}"), b);
                }
            }
            let expected = c
                .ranks
                .iter()
                .find(|r| r.rank == rank)
                .ok_or_else(|| anyhow::anyhow!("group rank tap"))?;
            let a = compare(&root.join(format!("rank{rank}")), &expected.taps, &local)?;
            let b = compare(&root, &c.global_taps, &global)?;
            different += a
                .iter()
                .chain(&b)
                .map(|t| t["different_values"].as_u64().unwrap() as usize)
                .sum::<usize>();
            traces.push(json!({"local_taps":a,"global_taps":b}));
        }
        Ok((traces, different))
    }
}
