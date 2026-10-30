//! Connected ratio1/2 source state for canonical six-row verify execution.
//! Precision-specific projection -> persistent compressed cache -> WK/index K -> queries,
//! scores and GPU physical auxiliary-candidate production.
use crate::{
    attention::{Candidates, PagedKv},
    bf16_linear::Projection,
    blas::Blas,
    candidate_map::Map,
    compression::{self, Source},
    cuda::{Buffer, Cuda, Stream},
    index::{self, IndexProjection, KeyCache},
    prelude::FrequencyTable,
    query_index::{self, Branch},
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{
    path::{Path, PathBuf},
    rc::Rc,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Verify,
    Extend,
    Decode,
}
impl Default for Phase {
    fn default() -> Self {
        Self::Verify
    }
}
enum CompressionMetadata {
    Verify(compression::Metadata),
    Extend(compression::PairPrefill),
}
pub struct Metadata {
    rows: usize,
    phase: Phase,
    owner_keys: u64,
    ratio: usize,
    frequency_rows: usize,
    compression: CompressionMetadata,
    keys: index::Schedule,
    query: query_index::Metadata,
}
impl Metadata {
    pub fn update(
        &mut self,
        positions: &[i32],
        requests: &[i64],
        raw_locations: &[i64],
        lengths: &[i32],
    ) -> Result<()> {
        ensure!(
            positions.len() == self.rows
                && raw_locations.len() == self.rows
                && lengths.len() == self.rows
                && positions
                    .iter()
                    .all(|&p| p >= 0 && (p as usize) < self.frequency_rows),
            "compressed source metadata update geometry"
        );
        let key_locations = positions
            .iter()
            .zip(raw_locations)
            .map(|(&p, &l)| {
                if l != 0 && (p as usize + 1) % self.ratio == 0 {
                    l / self.ratio as i64
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        match &mut self.compression {
            CompressionMetadata::Verify(m) => m.update(positions, requests, raw_locations)?,
            CompressionMetadata::Extend(m) => m.update(positions, requests, raw_locations)?,
        }
        self.keys.update(positions, &key_locations)?;
        self.query.update(positions, &vec![0; self.rows], lengths)
    }
    pub fn new(
        cuda: &Cuda,
        source: &Layer,
        positions: &[i32],
        requests: &[i64],
        raw_locations: &[i64],
        slots: &[i32],
        lengths: &[i32],
        freq_rows: usize,
    ) -> Result<Self> {
        ensure!(
            positions.len() == source.rows
                && raw_locations.len() == source.rows
                && lengths.len() == source.rows
                && positions
                    .iter()
                    .all(|&p| p >= 0 && (p as usize) < freq_rows),
            "compressed source rows"
        );
        let ratio = source.ratio;
        let key_locations = positions
            .iter()
            .zip(raw_locations)
            .map(|(&p, &loc)| {
                if loc != 0 && (p as usize + 1) % ratio == 0 {
                    loc / ratio as i64
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        Ok(Self {
            rows: source.rows,
            phase: source.phase,
            owner_keys: source.keys.data().ptr(),
            ratio,
            frequency_rows: freq_rows,
            compression: match source.phase {
                Phase::Verify | Phase::Decode => {
                    CompressionMetadata::Verify(compression::Metadata::new(
                        cuda,
                        ratio,
                        source.source.cache().capacity(),
                        1,
                        source.source.ring_size(),
                        if ratio == 1 || source.phase == Phase::Decode {
                            1
                        } else {
                            source.rows
                        },
                        freq_rows,
                        positions,
                        requests,
                        raw_locations,
                    )?)
                }
                Phase::Extend => {
                    let (core, kernels) = source
                        .prefill_assets
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("prefill source assets missing"))?;
                    CompressionMetadata::Extend(compression::PairPrefill::new(
                        cuda,
                        core,
                        kernels,
                        &source.source,
                        positions,
                        requests,
                        raw_locations,
                    )?)
                }
            },
            keys: index::Schedule::new(
                cuda,
                source.keys.capacity(),
                freq_rows,
                positions,
                &key_locations,
            )?,
            query: query_index::Metadata::new(
                cuda,
                source.keys.capacity(),
                slots.len(),
                positions,
                slots,
                &vec![0; source.rows],
                lengths,
                freq_rows,
            )?,
        })
    }
}
pub struct Layer {
    rows: usize,
    phase: Phase,
    prefill_assets: Option<(Vec<u8>, PathBuf)>,
    layer: usize,
    ratio: usize,
    projection: SourceProjection,
    source: Source,
    wk: IndexProjection,
    keys: KeyCache,
    query: Branch,
    mapping: Map,
}
enum SourceProjection {
    Pair(Projection<f32>),
    Single(Projection<bf16>),
}
impl Layer {
    pub(crate) fn set_sort_shuffle_before_capture(&mut self, enabled: bool) -> Result<()> {
        self.query.set_sort_shuffle_before_capture(enabled)
    }
    pub fn new(
        cuda: &Cuda,
        stream: &Stream,
        weights: Rc<Weights>,
        blas: Rc<Blas>,
        layer: usize,
        kernels: &Path,
        frequency: Rc<FrequencyTable>,
        core: &[u8],
        kv_pages: usize,
        index_pages: usize,
        slots: &[i32],
        lengths: &[i32],
    ) -> Result<Self> {
        Self::new_rows(
            cuda,
            stream,
            weights,
            blas,
            layer,
            kernels,
            frequency,
            core,
            kv_pages,
            index_pages,
            slots,
            lengths,
            6,
            Phase::Verify,
        )
    }
    pub fn new_rows(
        cuda: &Cuda,
        stream: &Stream,
        weights: Rc<Weights>,
        blas: Rc<Blas>,
        layer: usize,
        kernels: &Path,
        frequency: Rc<FrequencyTable>,
        core: &[u8],
        kv_pages: usize,
        index_pages: usize,
        slots: &[i32],
        lengths: &[i32],
        rows: usize,
        phase: Phase,
    ) -> Result<Self> {
        Self::new_rows_owner(
            cuda,
            stream,
            weights,
            blas,
            layer,
            kernels,
            frequency,
            core,
            kv_pages,
            index_pages,
            slots,
            lengths,
            rows,
            phase,
            None,
        )
    }
    pub(crate) fn new_rows_shared(
        cuda: &Cuda,
        stream: &Stream,
        weights: Rc<Weights>,
        blas: Rc<Blas>,
        layer: usize,
        kernels: &Path,
        frequency: Rc<FrequencyTable>,
        core: &[u8],
        kv_pages: usize,
        index_pages: usize,
        slots: &[i32],
        lengths: &[i32],
        rows: usize,
        phase: Phase,
        owner: &Self,
    ) -> Result<Self> {
        Self::new_rows_owner(
            cuda,
            stream,
            weights,
            blas,
            layer,
            kernels,
            frequency,
            core,
            kv_pages,
            index_pages,
            slots,
            lengths,
            rows,
            phase,
            Some(owner),
        )
    }
    fn new_rows_owner(
        cuda: &Cuda,
        stream: &Stream,
        weights: Rc<Weights>,
        blas: Rc<Blas>,
        layer: usize,
        kernels: &Path,
        frequency: Rc<FrequencyTable>,
        core: &[u8],
        kv_pages: usize,
        index_pages: usize,
        slots: &[i32],
        lengths: &[i32],
        rows: usize,
        phase: Phase,
        owner: Option<&Self>,
    ) -> Result<Self> {
        ensure!(
            crate::config::target_rows(rows) && (crate::config::verification_rows(rows) || phase == Phase::Extend),
            "source execution profile/phase"
        );
        ensure!(
            [2, 8, 14, 20].contains(&layer) && !slots.is_empty(),
            "low-ratio KV-source layer"
        );
        let native = index::Native::open(cuda, &kernels.join("index-v1"))?;
        let ratio = if layer == 20 { 1 } else { 2 };
        let projection = if ratio == 1 {
            SourceProjection::Single(Projection::new(
                cuda,
                weights.clone(),
                blas.clone(),
                &format!("layers.{layer}.attn.compressor.wkv"),
                rows,
            )?)
        } else {
            SourceProjection::Pair(Projection::compressor_pair(
                cuda,
                weights.clone(),
                blas.clone(),
                layer,
                rows,
            )?)
        };
        let source = if let Some(owner) = owner {
            ensure!(
                owner.layer == layer && owner.source.matches_static(&weights, &frequency),
                "shared source model/frequency identity changed"
            );
            owner
                .source
                .share_rows(cuda, rows, &kernels.join("compression-v2"))?
        } else {
            Source::new(
                cuda,
                weights.clone(),
                layer,
                rows,
                1,
                crate::config::verification_carry_rows(),
                kv_pages,
                if phase == Phase::Extend { 64 } else { 32 },
                frequency.clone(),
                &kernels.join("compression-v2"),
            )?
        };
        let wk = IndexProjection::new(
            cuda,
            weights.clone(),
            blas.clone(),
            native.clone(),
            &format!("layers.{layer}.attn.indexer.wk"),
            rows,
        )?;
        let keys = if let Some(owner) = owner {
            owner.keys.share_rows(rows)?
        } else {
            KeyCache::new(
                cuda,
                native,
                weights.clone(),
                frequency.clone(),
                layer,
                rows,
                index_pages,
            )?
        };
        let query = Branch::new_rows(
            cuda,
            stream,
            weights,
            blas,
            layer,
            kernels,
            frequency,
            core,
            slots.len(),
            slots.len(),
            rows,
        )?;
        let mapping = Map::new(
            cuda,
            core,
            rows,
            512,
            source.cache().capacity(),
            slots,
            lengths,
        )?;
        Ok(Self {
            rows,
            phase,
            prefill_assets: if phase == Phase::Extend {
                Some((core.to_vec(), kernels.to_owned()))
            } else {
                None
            },
            layer,
            ratio,
            projection,
            source,
            wk,
            keys,
            query,
            mapping,
        })
    }
    pub fn run(
        &mut self,
        hidden: &Buffer<bf16>,
        q_lora: &Buffer<bf16>,
        meta: &Metadata,
        stream: &Stream,
    ) -> Result<()> {
        self.prepare_keys(hidden, meta, stream)?;
        self.query
            .run(hidden, q_lora, &self.keys, &meta.query, stream)?;
        self.mapping
            .run(self.query.selection(), meta.query.windows(), stream)
    }
    pub fn run_quantized(
        &mut self,
        hidden: &Buffer<bf16>,
        q: &Buffer<u8>,
        scales: &Buffer<u8>,
        meta: &Metadata,
        stream: &Stream,
    ) -> Result<()> {
        self.prepare_keys(hidden, meta, stream)?;
        self.query
            .run_quantized(hidden, q, scales, &self.keys, &meta.query, None, stream)?;
        self.mapping
            .run(self.query.selection(), meta.query.windows(), stream)
    }
    pub(crate) fn run_distributed(&mut self, hidden: &Buffer<bf16>, q_lora: &Buffer<bf16>,
        meta: &Metadata, live: &Buffer<i32>, comm: &crate::roce::Roce, stream: &Stream) -> Result<()> {
        self.prepare_keys(hidden, meta, stream)?;
        self.query.run_distributed(hidden, q_lora, &self.keys, &meta.query, live, comm, stream)?;
        self.mapping.run(self.query.selection(), meta.query.windows(), stream)
    }
    pub(crate) fn run_quantized_distributed(&mut self, hidden: &Buffer<bf16>, q: &Buffer<u8>,
        scales: &Buffer<u8>, meta: &Metadata, live: &Buffer<i32>,
        comm: &crate::roce::Roce, stream: &Stream) -> Result<()> {
        self.prepare_keys(hidden, meta, stream)?;
        self.query.run_quantized_distributed(hidden, q, scales, &self.keys, &meta.query, live, comm, stream)?;
        self.mapping.run(self.query.selection(), meta.query.windows(), stream)
    }
    fn prepare_keys(
        &mut self,
        hidden: &Buffer<bf16>,
        meta: &Metadata,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            meta.owner_keys == self.keys.data().ptr()
                && meta.phase == self.phase
                && meta.rows == self.rows,
            "source metadata belongs to another key owner"
        );
        match &self.projection {
            SourceProjection::Pair(p) => {
                p.run(hidden, stream)?;
                match &meta.compression {
                    CompressionMetadata::Verify(m) => {
                        if self.phase == Phase::Decode {
                            self.source.run_single_decode(
                                compression::Input::Ratio2(p.output()),
                                m,
                                stream,
                            )?
                        } else {
                            self.source
                                .run(compression::Input::Ratio2(p.output()), m, stream)?
                        }
                    }
                    CompressionMetadata::Extend(m) => {
                        self.source.run_pair_prefill(p.output(), m, stream)?
                    }
                }
            }
            SourceProjection::Single(p) => {
                p.run(hidden, stream)?;
                match &meta.compression {
                    CompressionMetadata::Verify(m) => {
                        if self.phase == Phase::Decode {
                            self.source.run_single_decode(
                                compression::Input::Ratio1(p.output()),
                                m,
                                stream,
                            )?
                        } else {
                            self.source
                                .run(compression::Input::Ratio1(p.output()), m, stream)?
                        }
                    }
                    CompressionMetadata::Extend(m) => {
                        self.source.run_single_prefill(p.output(), m, stream)?
                    }
                }
            }
        }
        self.wk.run(self.source.latent(), stream)?;
        self.keys.run(self.wk.output(), &meta.keys, stream)
    }
    pub fn cache(&self) -> &PagedKv {
        self.source.cache()
    }
    pub(crate) fn clear_state(&mut self, stream: &Stream) -> Result<()> {
        self.source.clear_state(stream)?;
        self.keys.clear_state(stream)
    }
    pub fn layer_id(&self) -> usize {
        self.layer
    }
    pub fn candidates(&self) -> &Candidates {
        self.mapping.output()
    }
    pub fn error(&self) -> Result<u32> {
        self.mapping.error()
    }
    pub fn projection(&self) -> compression::Input<'_> {
        match &self.projection {
            SourceProjection::Pair(p) => compression::Input::Ratio2(p.output()),
            SourceProjection::Single(p) => compression::Input::Ratio1(p.output()),
        }
    }
    pub fn source(&self) -> &Source {
        &self.source
    }
    pub(crate) fn source_mut(&mut self) -> &mut Source {
        &mut self.source
    }
    pub fn wk(&self) -> &Buffer<bf16> {
        self.wk.output()
    }
    pub fn keys(&self) -> &KeyCache {
        &self.keys
    }
    pub(crate) fn keys_mut(&mut self) -> &mut KeyCache {
        &mut self.keys
    }
    pub fn query(&self) -> &Branch {
        &self.query
    }
    pub(crate) fn share_candidate_scratch_from(&mut self, prior: &Self) -> Result<()> {
        self.query.share_candidate_scratch_from(&prior.query)
    }
    pub(crate) fn enable_retained_publication(&mut self) -> Result<()> {
        self.query.enable_retained_publication()
    }
}

/// A later index source owns queries/selection, while borrowing source20's
/// compressed KV and index keys. It must never allocate a duplicate WK/cache.
/// The initial connected trace admits the all-candidates window only; larger
/// windows require the separately qualified source20 block-mask path.
pub struct ReadIndex {
    layer: usize,
    owner_keys: u64,
    owner_kv: u64,
    query: Branch,
    mapping: Map,
}
impl ReadIndex {
    pub(crate) fn enable_compact_sparse6(&mut self)->Result<()> {
        self.query.enable_compact_sparse6()
    }
    pub(crate) fn set_adaptive_sparse6(&mut self, enabled: bool) {
        self.query.set_adaptive_sparse6(enabled);
    }
    pub(crate) fn set_sort_shuffle_before_capture(&mut self, enabled: bool) -> Result<()> {
        self.query.set_sort_shuffle_before_capture(enabled)
    }
    pub(crate) fn run(&mut self, hidden: &Buffer<bf16>, q_lora: &Buffer<bf16>, owner: &Layer,
        metadata: &Metadata, replay_mask: Option<&Buffer<u8>>, stream: &Stream) -> Result<()> {
        ensure!(owner.layer == 20 && self.owner_keys == owner.keys.data().ptr()
            && self.owner_kv == owner.cache().data().ptr() && metadata.owner_keys == self.owner_keys,
            "read-only index source owner changed");
        self.query.run_with_mask(hidden, q_lora, &owner.keys, &metadata.query,
            replay_mask.or(owner.query.candidate_mask()), stream)?;
        self.mapping.run(self.query.selection(), metadata.query.windows(), stream)
    }
    pub fn new(
        cuda: &Cuda,
        stream: &Stream,
        weights: Rc<Weights>,
        blas: Rc<Blas>,
        layer: usize,
        kernels: &Path,
        frequency: Rc<FrequencyTable>,
        core: &[u8],
        owner: &Layer,
        slots: &[i32],
        lengths: &[i32],
    ) -> Result<Self> {
        ensure!(
            [24, 28, 32, 36].contains(&layer)
                && owner.layer == 20
                && !slots.is_empty()
                && slots.len() <= crate::config::MODEL_CONTEXT,
            "read-only index source requires source20 and an admitted candidate window"
        );
        Ok(Self {
            layer,
            owner_keys: owner.keys.data().ptr(),
            owner_kv: owner.cache().data().ptr(),
            query: Branch::new_rows(
                cuda,
                stream,
                weights,
                blas,
                layer,
                kernels,
                frequency,
                core,
                slots.len(),
                slots.len(),
                owner.rows,
            )?,
            mapping: Map::new(
                cuda,
                core,
                owner.rows,
                512,
                owner.cache().capacity(),
                slots,
                lengths,
            )?,
        })
    }
    pub fn run_quantized(
        &mut self,
        hidden: &Buffer<bf16>,
        q: &Buffer<u8>,
        scales: &Buffer<u8>,
        owner: &Layer,
        metadata: &Metadata,
        replay_mask: Option<&Buffer<u8>>,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            owner.layer == 20
                && self.owner_keys == owner.keys.data().ptr()
                && self.owner_kv == owner.cache().data().ptr()
                && metadata.owner_keys == self.owner_keys,
            "read-only index source owner changed"
        );
        self.query
            .run_quantized(
                hidden,
                q,
                scales,
                &owner.keys,
                &metadata.query,
                // A decoder-tail replay consumes the blocks source20 published
                // for those rows in the encoder; otherwise this step's own.
                replay_mask.or(owner.query.candidate_mask()),
                stream,
            )?;
        self.mapping
            .run(self.query.selection(), metadata.query.windows(), stream)
    }
    pub fn layer_id(&self) -> usize {
        self.layer
    }
    pub fn candidates(&self) -> &Candidates {
        self.mapping.output()
    }
    pub fn error(&self) -> Result<u32> {
        self.mapping.error()
    }
    pub fn query(&self) -> &Branch {
        &self.query
    }
}
