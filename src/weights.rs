//! Independent checkpoint-to-device loading. One aligned atlas per rank;
//! preserve original FP4/FP8/scales and apply the exact stored-unit TP slice.
use crate::{
    checkpoint::{Checkpoint, Dtype},
    cuda::{Buffer, Cuda},
    layout::{Placement, RankPlan, copy_spans},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    os::unix::fs::{FileExt, MetadataExt},
    path::Path,
    time::Instant,
};
const CHUNK: usize = 16 * 1024 * 1024;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct View {
    pub offset: usize,
    pub bytes: usize,
    pub shape: Vec<u64>,
    pub dtype: Dtype,
}
pub struct Weights {
    atlas: Buffer<u8>,
    pub views: BTreeMap<String, View>,
    pub source_sha256: String,
    expert_states: BTreeMap<usize, bool>,
}
struct Source {
    file: File,
    device: u64,
    inode: u64,
    len: u64,
    modified: i64,
    modified_ns: i64,
}
impl Source {
    fn open(root: &Path, name: &str, checkpoint: &Checkpoint) -> Result<Self> {
        let file = File::open(root.join(name))?;
        let m = file.metadata()?;
        let shard = &checkpoint.files[name];
        ensure!(
            m.len() == shard.size,
            "checkpoint shard size changed: {name}"
        );
        let mut n = [0; 8];
        file.read_exact_at(&mut n, 0)?;
        ensure!(
            u64::from_le_bytes(n) + 8 == shard.data_start,
            "checkpoint header length changed"
        );
        let mut header = vec![0; usize::try_from(shard.data_start - 8)?];
        file.read_exact_at(&mut header, 8)?;
        ensure!(
            format!("{:x}", Sha256::digest(&header)) == shard.header_sha256,
            "checkpoint header changed: {name}"
        );
        Ok(Self {
            file,
            device: m.dev(),
            inode: m.ino(),
            len: m.len(),
            modified: m.mtime(),
            modified_ns: m.mtime_nsec(),
        })
    }
    fn unchanged(&self) -> Result<()> {
        let m = self.file.metadata()?;
        ensure!(
            (m.dev(), m.ino(), m.len(), m.mtime(), m.mtime_nsec())
                == (
                    self.device,
                    self.inode,
                    self.len,
                    self.modified,
                    self.modified_ns
                ),
            "checkpoint changed during loading"
        );
        Ok(())
    }
}
fn align(n: usize) -> Result<usize> {
    Ok(n.checked_add(255).context("atlas overflow")? & !255)
}
// Keep every layer's compact expert operands contiguous, in numeric expert
// order. Up precedes gate in W13. The names still address canonical views;
// subsequent in-place permutation changes no tensor's byte count.
fn allocation_key(name: &str) -> (String, u8, usize, u8, String) {
    // The ratio-2 compressor consumes [KV; score] rows in one FP32-output
    // projection. Keep the original BF16 sources adjacent without a copy.
    if let Some((prefix, tail)) = name.split_once(".attn.compressor.") {
        if let Some(slot) = match tail {
            "wkv.weight" => Some(0),
            "wgate.weight" => Some(1),
            _ => None,
        } {
            return (
                format!("{prefix}.attn.compressor.wkv_gate"),
                0,
                0,
                slot,
                name.into(),
            );
        }
    }
    if let Some((prefix, tail)) = name.split_once(".ffn.shared_experts.") {
        let pair = match tail {
            "w1.weight" => Some((0, 0)),
            "w3.weight" => Some((0, 1)),
            "w1.scale" => Some((1, 0)),
            "w3.scale" => Some((1, 1)),
            _ => None,
        };
        if let Some((kind, slot)) = pair {
            return (
                format!("{prefix}.ffn.shared_experts.w13"),
                kind,
                0,
                slot,
                name.into(),
            );
        }
    }
    // Q-A and KV share their activation. Preserve adjacent canonical row
    // blocks so one fused projection can address them without another copy.
    if let Some((prefix, tail)) = name.split_once(".attn.") {
        let pair = match tail {
            "wq_a.weight" => Some((0, 0)),
            "wkv.weight" => Some((0, 1)),
            "wq_a.scale" => Some((1, 0)),
            "wkv.scale" => Some((1, 1)),
            _ => None,
        };
        if let Some((kind, slot)) = pair {
            return (format!("{prefix}.attn.wqkv_a"), kind, 0, slot, name.into());
        }
    }

    if let Some((prefix, tail)) = name.split_once(".ffn.experts.") {
        let parts: Vec<_> = tail.split('.').collect();
        if let [expert, projection, kind] = parts.as_slice() {
            if let Ok(expert) = expert.parse::<usize>() {
                let key = match (*projection, *kind) {
                    ("w3", "weight") => Some((0, 0)),
                    ("w1", "weight") => Some((0, 1)),
                    ("w2", "weight") => Some((1, 0)),
                    ("w3", "scale") => Some((2, 0)),
                    ("w1", "scale") => Some((2, 1)),
                    ("w2", "scale") => Some((3, 0)),
                    _ => None,
                };
                if let Some((group, slot)) = key {
                    return (
                        format!("{prefix}.ffn.experts"),
                        group,
                        expert,
                        slot,
                        name.into(),
                    );
                }
            }
        }
    }
    (name.into(), 0, 0, 0, String::new())
}
#[derive(Serialize, Deserialize)]
struct AtlasSnapshot {
    version: u32,
    rank: usize,
    world: usize,
    prefixes: Vec<String>,
    checkpoint_sha256: String,
    source_sha256: String,
    atlas_sha256: String,
    bytes: usize,
    views: BTreeMap<String, View>,
}
fn select_views(
    cp: &Checkpoint,
    plan: &RankPlan,
    prefixes: &[&str],
) -> Result<(BTreeMap<String, View>, usize)> {
    ensure!(!prefixes.is_empty(), "empty checkpoint selection");
    cp.validate()?;
    ensure!(plan.world == 4 && plan.rank < 4, "invalid rank plan");
    let mut views = BTreeMap::new();
    let mut total = 0usize;
    let mut names: Vec<_> = plan.weights.keys().collect();
    names.sort_by_cached_key(|name| allocation_key(name));
    for name in names {
        let w = &plan.weights[name];
        if !prefixes.iter().any(|p| name.starts_with(p))
            || w.host_resident
            || matches!(w.placement, Placement::Omitted { .. })
        {
            continue;
        }
        let t = cp.tensor(name)?;
        let spans = copy_spans(t, w)?;
        ensure!(
            spans.iter().map(|s| s.1).sum::<u64>() == w.local_bytes,
            "shard size mismatch"
        );
        // The shared gate/up scale rows have 2880 bytes each. Keep the
        // second block adjacent rather than inserting atlas padding;
        // it remains 64-byte aligned and the pair's start is 256-aligned.
        if !name.ends_with(".ffn.shared_experts.w3.scale") {
            total = align(total)?;
        }
        let bytes = usize::try_from(w.local_bytes)?;
        ensure!(bytes > 0, "empty device weight");
        views.insert(
            name.clone(),
            View {
                offset: total,
                bytes,
                shape: w.local_shape.clone(),
                dtype: t.dtype.clone(),
            },
        );
        total = total.checked_add(bytes).context("atlas overflow")?;
    }
    total = align(total)?;
    ensure!(total > 0, "weight selection is empty");
    Ok((views, total))
}
impl Weights {
    pub(crate) fn in_context(&self, cuda: &Cuda) -> bool {
        cuda.owns_buffer(&self.atlas)
    }
    pub(crate) fn device_range(&self, offset: usize, bytes: usize) -> Result<Vec<u8>> {
        ensure!(
            bytes <= 16 * 1024 * 1024,
            "diagnostic transfer exceeds staging bound"
        );
        self.atlas.download_part(offset, bytes)
    }
    pub(crate) fn pointer_at(&self, offset: usize, bytes: usize) -> Result<u64> {
        ensure!(
            bytes > 0
                && offset
                    .checked_add(bytes)
                    .is_some_and(|end| end <= self.atlas.len()),
            "weight range outside atlas"
        );
        Ok(self.atlas.ptr() + offset as u64)
    }
    pub(crate) fn begin_expert_layer(&mut self, layer: usize) -> Result<()> {
        ensure!(
            !self.expert_states.contains_key(&layer),
            "expert layer was already prepared or tainted"
        );
        self.expert_states.insert(layer, false);
        Ok(())
    }
    pub(crate) fn finish_expert_layer(&mut self, layer: usize) -> Result<()> {
        ensure!(
            self.expert_states.get(&layer) == Some(&false),
            "invalid expert preparation transition"
        );
        self.expert_states.insert(layer, true);
        Ok(())
    }
    pub fn expert_ready(&self, layer: usize) -> bool {
        self.expert_states.get(&layer) == Some(&true)
    }
    pub fn ptr(&self, name: &str) -> Result<u64> {
        let v = self.views.get(name).context("missing loaded weight")?;
        Ok(self.atlas.ptr() + v.offset as u64)
    }
    pub fn view(&self, name: &str) -> Result<&View> {
        self.views.get(name).context("missing loaded weight")
    }
    pub fn bytes(&self) -> usize {
        self.atlas.len()
    }
    /// prefix bounds a vertical-slice qualification. Empty selects every device
    /// weight; exact Engram backing is separately owned by its row cache.
    pub fn load(
        cuda: &Cuda,
        root: &Path,
        cp: &Checkpoint,
        plan: &RankPlan,
        prefix: &str,
    ) -> Result<Self> {
        Self::load_prefixes(cuda, root, cp, plan, &[prefix])
    }
    pub fn load_prefixes(
        cuda: &Cuda,
        root: &Path,
        cp: &Checkpoint,
        plan: &RankPlan,
        prefixes: &[&str],
    ) -> Result<Self> {
        let (views, total) = select_views(cp, plan, prefixes)?;
        let free = cuda.admission_memory()?.admitted_available;
        // Leave explicit operating headroom; the model's admission plan adds
        // cache/workspace/graphs separately before a complete-engine load.
        ensure!(
            total
                .checked_add(8 * 1024 * 1024 * 1024)
                .is_some_and(|v| v < free),
            "weight atlas exceeds available memory with headroom"
        );
        let mut atlas = cuda.allocate::<u8>(total)?;
        atlas.zero()?;
        let mut sources = BTreeMap::new();
        let mut ordering = Vec::new();
        for name in views.keys() {
            let t = cp.tensor(name)?;
            ordering.push((t.file.clone(), t.data_offsets[0], name.clone()));
        }
        ordering.sort_unstable();
        let mut staging = vec![0u8; CHUNK];
        let mut digest = Sha256::new();
        for (file, _, name) in ordering {
            if !sources.contains_key(&file) {
                sources.insert(file.clone(), Source::open(root, &file, cp)?);
            }
            let f = &sources[&file].file;
            let t = cp.tensor(&name)?;
            let w = &plan.weights[&name];
            let view = &views[&name];
            // Validate the compact FP4 kernel's lossless scale domain while the
            // checkpoint bytes are already on the host. This avoids 1152 GPU
            // readbacks/synchronizations per layer during preparation.
            let expert_scale = name.contains(".ffn.experts.") && name.ends_with(".scale");
            let validate_bytes = |data: &[u8]| -> Result<()> {
                ensure!(
                    !expert_scale || data.iter().all(|&x| x <= 247),
                    "canonical expert scale exceeds lossless conversion range: {name}"
                );
                Ok(())
            };
            digest.update((name.len() as u64).to_le_bytes());
            digest.update(name.as_bytes());
            digest.update((view.bytes as u64).to_le_bytes());
            let base = cp.files[&file].data_start + t.data_offsets[0];
            let mut destination = view.offset;
            // Row-sharded matrices are contiguous. Column-sharded matrices are
            // streamed in bounded source windows and compacted on the CPU;
            // this avoids millions of tiny NVMe/NFS reads and CUDA transfers.
            let spans = copy_spans(t, w)?;
            let mut i = 0;
            while i < spans.len() {
                let window_start = spans[i].0;
                let mut window_end = window_start + spans[i].1.min(CHUNK as u64);
                if spans[i].1 <= CHUNK as u64 {
                    let mut j = i + 1;
                    while j < spans.len() && spans[j].0 + spans[j].1 - window_start <= CHUNK as u64
                    {
                        window_end = spans[j].0 + spans[j].1;
                        j += 1;
                    }
                }
                if spans[i].1 > CHUNK as u64 {
                    let (start, len) = spans[i];
                    let mut copied = 0;
                    while copied < len {
                        let n = (len - copied).min(CHUNK as u64) as usize;
                        f.read_exact_at(&mut staging[..n], base + start + copied)?;
                        validate_bytes(&staging[..n])?;
                        digest.update(&staging[..n]);
                        atlas.upload_part(destination, &staging[..n])?;
                        destination += n;
                        copied += n as u64;
                    }
                    i += 1;
                    continue;
                }
                let count = usize::try_from(window_end - window_start)?;
                f.read_exact_at(&mut staging[..count], base + window_start)?;
                let mut compacted = 0;
                while i < spans.len() && spans[i].0 + spans[i].1 <= window_end {
                    let offset = usize::try_from(spans[i].0 - window_start)?;
                    let n = usize::try_from(spans[i].1)?;
                    staging.copy_within(offset..offset + n, compacted);
                    compacted += n;
                    i += 1;
                }
                validate_bytes(&staging[..compacted])?;
                digest.update(&staging[..compacted]);
                atlas.upload_part(destination, &staging[..compacted])?;
                destination += compacted;
            }
            ensure!(
                destination == view.offset + view.bytes,
                "incomplete weight upload"
            );
        }
        for source in sources.values() {
            source.unchanged()?;
        }
        Ok(Self {
            atlas,
            views,
            source_sha256: format!("{:x}", digest.finalize()),
            expert_states: BTreeMap::new(),
        })
    }
    fn device_digest(&self, cp: &Checkpoint) -> Result<String> {
        let mut order: Vec<_> = self.views.keys().collect();
        order.sort_by_key(|name| {
            let t = &cp.tensors[*name];
            (t.file.clone(), t.data_offsets[0])
        });
        let mut digest = Sha256::new();
        for name in order {
            let v = &self.views[name];
            digest.update((name.len() as u64).to_le_bytes());
            digest.update(name.as_bytes());
            digest.update((v.bytes as u64).to_le_bytes());
            for offset in (0..v.bytes).step_by(CHUNK) {
                digest.update(
                    self.atlas
                        .download_part(v.offset + offset, CHUNK.min(v.bytes - offset))?,
                );
            }
        }
        let device_sha = format!("{:x}", digest.finalize());
        Ok(device_sha)
    }
    /// Lossless per-rank source-byte cache for cluster nodes that do not own
    /// the canonical NVMe checkpoint. Export only unprepared original units.
    pub fn stage(
        &self,
        cp: &Checkpoint,
        root: &Path,
        plan: &RankPlan,
        prefixes: &[&str],
        directory: &Path,
    ) -> Result<serde_json::Value> {
        use std::io::Write;
        ensure!(
            self.expert_states.is_empty(),
            "stage only original unprepared checkpoint units"
        );
        let (views, total) = select_views(cp, plan, prefixes)?;
        ensure!(
            views == self.views && total == self.bytes(),
            "staging selection differs from loaded atlas"
        );
        let qualified = self.qualify(cp, root, plan)?;
        std::fs::create_dir_all(directory)?;
        ensure!(
            !directory.join("atlas.json").exists(),
            "immutable staged atlas exists"
        );
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("atlas.bin"))?;
        let mut hash = Sha256::new();
        for offset in (0..self.bytes()).step_by(CHUNK) {
            let b = self
                .atlas
                .download_part(offset, CHUNK.min(self.bytes() - offset))?;
            out.write_all(&b)?;
            hash.update(&b);
        }
        out.sync_all()?;
        let snapshot = AtlasSnapshot {
            version: 1,
            rank: plan.rank,
            world: plan.world,
            prefixes: prefixes.iter().map(|s| s.to_string()).collect(),
            checkpoint_sha256: format!("{:x}", Sha256::digest(serde_json::to_vec(cp)?)),
            source_sha256: self.source_sha256.clone(),
            atlas_sha256: format!("{:x}", hash.finalize()),
            bytes: total,
            views,
        };
        let mut meta = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("atlas.json.tmp"))?;
        meta.write_all(&serde_json::to_vec_pretty(&snapshot)?)?;
        meta.sync_all()?;
        std::fs::rename(
            directory.join("atlas.json.tmp"),
            directory.join("atlas.json"),
        )?;
        File::open(directory)?.sync_all()?;
        Ok(
            serde_json::json!({"source":qualified,"atlas_sha256":snapshot.atlas_sha256,"checkpoint_sha256":snapshot.checkpoint_sha256,
            "rank":plan.rank,"bytes":total,"precision":"original checkpoint bytes; no numeric packing or conversion"}),
        )
    }
    pub fn load_staged(
        cuda: &Cuda,
        cp: &Checkpoint,
        plan: &RankPlan,
        prefixes: &[&str],
        directory: &Path,
    ) -> Result<Self> {
        use std::io::Read;
        let metadata = std::fs::read(directory.join("atlas.json"))?;
        ensure!(
            metadata.len() <= 64 * 1024 * 1024,
            "staged atlas metadata bound"
        );
        let m: AtlasSnapshot = serde_json::from_slice(&metadata)?;
        let (views, total) = select_views(cp, plan, prefixes)?;
        ensure!(
            m.version == 1
                && m.rank == plan.rank
                && m.world == 4
                && m.world == plan.world
                && m.bytes == total
                && m.prefixes == prefixes.iter().map(|s| s.to_string()).collect::<Vec<_>>()
                && m.views == views
                && m.checkpoint_sha256 == format!("{:x}", Sha256::digest(serde_json::to_vec(cp)?)),
            "staged atlas checkpoint/TP/layout contract mismatch"
        );
        let available = cuda.admission_memory()?;
        ensure!(
            total
                .checked_add(8 * 1024 * 1024 * 1024)
                .is_some_and(|n| n < available.admitted_available),
            "staged atlas exceeds memory headroom: requested={total}, available={available:?}"
        );
        let mut input = File::open(directory.join("atlas.bin"))?;
        let before = input.metadata()?;
        ensure!(before.len() == total as u64, "staged atlas size mismatch");
        let scale_ranges: Vec<_> = views
            .iter()
            .filter(|(n, _)| n.contains(".ffn.experts.") && n.ends_with(".scale"))
            .map(|(_, v)| (v.offset, v.offset + v.bytes))
            .collect();
        let mut atlas = cuda.allocate::<u8>(total)?;
        let mut staging = vec![0u8; CHUNK];
        let mut hash = Sha256::new();
        for offset in (0..total).step_by(CHUNK) {
            let n = CHUNK.min(total - offset);
            input.read_exact(&mut staging[..n])?;
            for &(begin, end) in &scale_ranges {
                let begin = begin.max(offset);
                let end = end.min(offset + n);
                if begin < end {
                    ensure!(
                        staging[begin - offset..end - offset]
                            .iter()
                            .all(|&v| v <= 247),
                        "staged expert scale domain"
                    );
                }
            }
            hash.update(&staging[..n]);
            atlas.upload_part(offset, &staging[..n])?;
        }
        let after = input.metadata()?;
        ensure!(
            (
                before.dev(),
                before.ino(),
                before.len(),
                before.mtime(),
                before.mtime_nsec()
            ) == (
                after.dev(),
                after.ino(),
                after.len(),
                after.mtime(),
                after.mtime_nsec()
            ),
            "staged atlas changed during upload"
        );
        ensure!(
            format!("{:x}", hash.finalize()) == m.atlas_sha256,
            "staged atlas data hash mismatch"
        );
        let result = Self {
            atlas,
            views,
            source_sha256: m.source_sha256,
            expert_states: BTreeMap::new(),
        };
        ensure!(
            result.device_digest(cp)? == result.source_sha256,
            "staged GPU bytes differ from qualified checkpoint TP bytes"
        );
        Ok(result)
    }
    pub fn qualify(
        &self,
        cp: &Checkpoint,
        root: &Path,
        plan: &RankPlan,
    ) -> Result<serde_json::Value> {
        let start = Instant::now();
        let mut samples = 0;
        let device_sha = self.device_digest(cp)?;
        ensure!(
            device_sha == self.source_sha256,
            "full device atlas differs from source TP bytes"
        );
        // Verify boundary bytes and an interior row in every real checkpoint
        // tensor, independently reading the original shard's TP copy spans.
        for (name, v) in &self.views {
            let spans = copy_spans(cp.tensor(name)?, &plan.weights[name])?;
            for logical in [0, v.bytes / 2, v.bytes.saturating_sub(64)] {
                let n = 64.min(v.bytes - logical);
                let actual = self.atlas.download_part(v.offset + logical, n)?;
                let mut expected = vec![0; n];
                let mut skip = logical as u64;
                let mut written = 0;
                for (off, len) in &spans {
                    if skip >= *len {
                        skip -= len;
                        continue;
                    }
                    let k = (n - written).min((len - skip) as usize);
                    cp.read_slice(root, name, off + skip, &mut expected[written..written + k])?;
                    written += k;
                    skip = 0;
                    if written == n {
                        break;
                    }
                }
                ensure!(
                    actual == expected,
                    "checkpoint GPU bytes differ: {name} at {logical}"
                );
                samples += 1;
            }
        }
        Ok(
            serde_json::json!({"tensors":self.views.len(),"device_bytes":self.bytes(),"source_sha256":self.source_sha256,
          "device_sha256":device_sha,"every_uploaded_byte_verified":true,
          "verified_boundary_and_interior_samples":samples,"verification_seconds":start.elapsed().as_secs_f64()}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expert_allocation_pairs_up_and_gate_in_numeric_order() {
        let mut names = vec![
            "layers.0.ffn.experts.10.w1.weight",
            "layers.0.ffn.experts.2.w2.weight",
            "layers.0.ffn.experts.2.w1.weight",
            "layers.0.ffn.experts.10.w3.weight",
            "layers.0.ffn.experts.2.w3.weight",
        ];
        names.sort_by_cached_key(|n| allocation_key(n));
        assert_eq!(
            names,
            vec![
                "layers.0.ffn.experts.2.w3.weight",
                "layers.0.ffn.experts.2.w1.weight",
                "layers.0.ffn.experts.10.w3.weight",
                "layers.0.ffn.experts.10.w1.weight",
                "layers.0.ffn.experts.2.w2.weight"
            ]
        );
    }
}
