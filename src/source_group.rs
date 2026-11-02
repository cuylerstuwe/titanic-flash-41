//! Rust-owned second ratio2 decoder group. The source8 writer/indexer runs
//! once; layers9..13 retain independent SWA keys and reuse its sparse state.
//! Six-row execution only until wider model plans are qualified.
use crate::{
    attention::{Candidates, PagedKv},
    blas::Blas,
    checkpoint::Checkpoint,
    compressed_source::{Layer as Source, Metadata, Phase},
    cuda::{Buffer, Cuda, Module, Stream},
    decoder::SwaLayer,
    engram::{PackedShard, RowStore},
    engram_gpu::{Layer as Engram, Rows},
    expert_packing,
    ffn::TokenRows,
    kv::RowLocations,
    layer_attention::InputMode,
    layout::RankPlan,
    prelude::{FrequencyTable, Positions},
    roce::Roce,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{
    path::{Path, PathBuf},
    rc::Rc,
};
pub struct Group {
    start: usize,
    layers: Vec<SwaLayer>,
    caches: Vec<PagedKv>,
    source: Source,
    engram: Option<Engram>,
    rows: Option<Rows>,
    row_store: Option<Rc<RowStore>>,
    pub atlas_bytes: Vec<usize>,
    pub source_hashes: Vec<String>,
}
impl Group {
    pub fn new(
        cuda: &Cuda,
        module: &Module,
        core: &[u8],
        stream: &Stream,
        cp: &Checkpoint,
        plan: &RankPlan,
        start: usize,
        stages: &[PathBuf],
        kernels: &Path,
        frequency: Rc<FrequencyTable>,
        slots: &[i32],
        kv_pages: usize,
        index_pages: usize,
        backing: &Path,
    ) -> Result<Self> {
        Self::new_with_phase(
            cuda,
            module,
            core,
            stream,
            cp,
            plan,
            start,
            stages,
            kernels,
            frequency,
            slots,
            kv_pages,
            index_pages,
            backing,
            Phase::Verify,
        )
    }
    pub fn new_with_phase(
        cuda: &Cuda,
        module: &Module,
        core: &[u8],
        stream: &Stream,
        cp: &Checkpoint,
        plan: &RankPlan,
        start: usize,
        stages: &[PathBuf],
        kernels: &Path,
        frequency: Rc<FrequencyTable>,
        slots: &[i32],
        kv_pages: usize,
        index_pages: usize,
        backing: &Path,
        phase: Phase,
    ) -> Result<Self> {
        Self::new_rows(
            cuda,
            module,
            core,
            stream,
            cp,
            plan,
            start,
            stages,
            kernels,
            frequency,
            slots,
            kv_pages,
            index_pages,
            backing,
            phase,
            6,
            None,
            None,
            4,
            None,
        )
    }
    pub(crate) fn new_rows(
        cuda: &Cuda,
        module: &Module,
        core: &[u8],
        stream: &Stream,
        cp: &Checkpoint,
        plan: &RankPlan,
        start: usize,
        stages: &[PathBuf],
        kernels: &Path,
        frequency: Rc<FrequencyTable>,
        slots: &[i32],
        kv_pages: usize,
        index_pages: usize,
        backing: &Path,
        phase: Phase,
        num_rows: usize,
        vendor_blas: Option<Rc<Blas>>,
        medium_woa: Option<Rc<std::cell::RefCell<crate::woa_medium::Scratch>>>,
        swa_pages: usize,
        mut reserved: Option<&mut crate::weights::StagedReservation>,
    ) -> Result<Self> {
        ensure!(
            [8, 14].contains(&start) && stages.len() == 6 && frequency.compressed(),
            "ratio2 source group construction"
        );
        ensure!(
            crate::config::target_rows(num_rows) && (crate::config::verification_rows(num_rows) || phase == Phase::Extend),
            "group row/phase profile"
        );
        let vendor_blas = match vendor_blas {
            Some(b) => b,
            None => Rc::new(Blas::new(cuda)?),
        };
        let mut layers = Vec::new();
        let mut caches = Vec::new();
        let mut source = None;
        let mut atlas_bytes = Vec::new();
        let mut source_hashes = Vec::new();
        let mut engram = None;
        for (i, stage) in stages.iter().enumerate() {
            let id = i + start;
            let mut weights =
                Weights::load_staged_reserved(cuda, cp, plan, &[&format!("layers.{id}.")], stage, reserved.as_deref_mut())?;
            atlas_bytes.push(weights.bytes());
            source_hashes.push(weights.source_sha256.clone());
            let prepared = expert_packing::prepare_layer(cuda, module, &mut weights, id)?;
            let weights = Rc::new(weights);
            layers.push(SwaLayer::new_rows(
                cuda,
                module,
                stream,
                weights.clone(),
                prepared,
                kernels,
                frequency.clone(),
                InputMode::Split(plan.rank),
                num_rows,
                Some(vendor_blas.clone()),
                medium_woa.clone(),
            )?);
            if id == start {
                if start == 14 {
                    engram = Some(Engram::new(
                        cuda,
                        module,
                        stream,
                        weights.clone(),
                        14,
                        num_rows,
                        kernels,
                    )?);
                }
                source = Some(Source::new_rows(
                    cuda,
                    stream,
                    weights,
                    vendor_blas.clone(),
                    id,
                    kernels,
                    frequency.clone(),
                    core,
                    kv_pages,
                    index_pages,
                    slots,
                    &vec![0; num_rows],
                    num_rows,
                    phase,
                )?);
            }
            caches.push(PagedKv::new(cuda, swa_pages, 64)?);
        }
        let row_store = if start == 14 {
            Some(Rc::new(RowStore::new(
                PackedShard::open(
                    &backing.join(format!("engram-l14-r{}of4.bin", plan.rank)),
                    14,
                    plan.rank,
                    4,
                    384016682,
                )?,
                crate::engram::target_cache_bytes()?,
            )?))
        } else {
            None
        };
        let rows = row_store
            .as_ref()
            .map(|store| Rows::new(cuda, num_rows, store))
            .transpose()?;
        Ok(Self {
            start,
            layers,
            caches,
            source: source.ok_or_else(|| anyhow::anyhow!("group source missing"))?,
            engram,
            rows,
            row_store,
            atlas_bytes,
            source_hashes,
        })
    }
    /// Bind a row/phase execution plan to existing model and causal state.
    pub(crate) fn fork_profile(
        &self,
        cuda: &Cuda,
        module: &Module,
        core: &[u8],
        stream: &Stream,
        kernels: &Path,
        rank: usize,
        frequency: Rc<FrequencyTable>,
        slots: &[i32],
        phase: Phase,
        num_rows: usize,
        vendor_blas: Rc<Blas>,
        medium_woa: Option<Rc<std::cell::RefCell<crate::woa_medium::Scratch>>>,
        wide: Option<&crate::decoder::WideBuild>,
        narrow_moe: Option<Rc<crate::moe::Workspace>>,
        independent: bool,
    ) -> Result<Self> {
        ensure!(wide.is_none() || narrow_moe.is_none(),"exclusive wide/narrow shared workspaces");
        ensure!(!independent || (num_rows==128 && phase==Phase::Extend && wide.is_none()
            && self.source.cache().page_tokens()==64),"independent group geometry");
        let mut layers = Vec::new();
        let mut engram = None;
        let mut source = None;
        for (i, old) in self.layers.iter().enumerate() {
            let id = self.start + i;
            let weights = old.shared_weights();
            // Engram14 output is layer14's residual, so it shares that layer's
            // arena scope (distinct offsets) rather than aliasing it.
            let build = |moe: Option<Rc<crate::moe::Workspace>>, sp: Option<usize>| -> Result<(SwaLayer, Option<Engram>)> {
                let layer = SwaLayer::new_rows_in(
                    cuda,
                    module,
                    stream,
                    weights.clone(),
                    old.prepared_layout(),
                    kernels,
                    frequency.clone(),
                    if wide.is_some() { InputMode::Fused } else { InputMode::Split(rank) },
                    num_rows,
                    Some(vendor_blas.clone()),
                    medium_woa.clone(),
                    moe,
                    sp,
                )?;
                let engram = if i == 0 && id == 14 {
                    Some(Engram::new(
                        cuda,
                        module,
                        stream,
                        weights.clone(),
                        14,
                        num_rows,
                        kernels,
                    )?)
                } else {
                    None
                };
                Ok((layer, engram))
            };
            let (layer, layer_engram) = match wide {
                Some(w) => w.layer(cuda, id, |moe, sp| build(Some(moe), sp))?,
                None => build(narrow_moe.clone(), None)?,
            };
            layers.push(layer);
            if layer_engram.is_some() {
                engram = layer_engram;
            }
            if i == 0 {
                source = Some(Source::new_rows_owner(
                    cuda,
                    stream,
                    weights,
                    vendor_blas.clone(),
                    id,
                    kernels,
                    frequency.clone(),
                    core,
                    self.source.cache().pages(),
                    self.source.keys().capacity() / 64,
                    slots,
                    &vec![0; num_rows],
                    num_rows,
                    phase,
                    (!independent).then_some(&self.source),
                )?);
            }
        }
        let row_store = self.row_store.clone();
        let rows = row_store
            .as_ref()
            .map(|s| Rows::new(cuda, num_rows, s))
            .transpose()?;
        Ok(Self {
            start: self.start,
            layers,
            caches: if independent {self.caches.iter()
                .map(|c|PagedKv::new(cuda,c.pages(),c.page_tokens())).collect::<Result<Vec<_>>>()?}
                else {self.caches.clone()},
            source: source.ok_or_else(|| anyhow::anyhow!("shared group source absent"))?,
            engram,
            rows,
            row_store,
            atlas_bytes: self.atlas_bytes.clone(),
            source_hashes: self.source_hashes.clone(),
        })
    }
    pub fn run(
        &mut self,
        input: &Buffer<bf16>,
        pre: &Buffer<f32>,
        positions: &Positions,
        locations: &RowLocations,
        candidates: &Candidates,
        tokens: &TokenRows,
        metadata: &Metadata,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        let mut residual = if let (Some(engram), Some(rows)) = (&mut self.engram, &mut self.rows) {
            engram.run(input, rows, tokens, comm, stream)?;
            engram.output()
        } else {
            input
        };
        let mut previous = pre;
        for (i, (layer, cache)) in self.layers.iter_mut().zip(&mut self.caches).enumerate() {
            // Sequence-parallel layer 15 takes this rank's rows of layer 14.
            let shard = layer.shard_input(residual, previous)?;
            let (input, input_pre) = shard.as_ref().map_or((residual, previous), |(r, p)| (r, p));
            if i == 0 {
                layer.run_indexed_from_previous(
                    input,
                    input_pre,
                    positions,
                    locations,
                    cache,
                    candidates,
                    tokens,
                    &mut self.source,
                    metadata,
                    comm,
                    stream,
                )?;
            } else {
                layer.run_shared_from_previous(
                    input,
                    input_pre,
                    positions,
                    locations,
                    cache,
                    candidates,
                    tokens,
                    &self.source,
                    comm,
                    stream,
                )?;
            }
            residual = layer.output();
            previous = layer.pre();
        }
        Ok(())
    }
    /// One ordinary attention/source stage for the finite two-owner executor.
    /// The caller completes both FFNs before asking for the following layer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_pair_layer(&mut self,index:usize,input:&Buffer<bf16>,pre:&Buffer<f32>,
        positions:&Positions,locations:&RowLocations,candidates:&Candidates,tokens:&TokenRows,
        metadata:&Metadata,comm:&Roce,stream:&Stream)->Result<()> {
        ensure!(index<6&&!self.sequence_parallel(),"paired source group layer domain");
        let input=if index==0 {
            if let (Some(engram),Some(rows))=(&mut self.engram,&mut self.rows) {
                engram.run(input,rows,tokens,comm,stream)?;engram.output()
            }else{input}
        }else{input};
        let (previous,rest)=self.layers.split_at_mut(index);let layer=&mut rest[0];
        let (input,pre)=previous.last().map_or((input,pre),|p|(p.output(),p.pre()));
        if index==0 {
            layer.run_indexed_from_previous(input,pre,positions,locations,&mut self.caches[index],
                candidates,tokens,&mut self.source,metadata,comm,stream)
        }else{
            layer.run_shared_from_previous(input,pre,positions,locations,&mut self.caches[index],
                candidates,tokens,&self.source,comm,stream)
        }
    }
    /// The group's output is this rank's sequence-parallel row shard.
    pub(crate) fn sequence_parallel(&self) -> bool {
        self.layers[5].sp_rank().is_some()
    }
    pub fn output(&self) -> &Buffer<bf16> {
        self.layers[5].output()
    }
    pub fn pre(&self) -> &Buffer<f32> {
        self.layers[5].pre()
    }
    pub fn source(&self) -> &Source {
        &self.source
    }
    pub fn source_mut(&mut self) -> &mut Source {
        &mut self.source
    }
    /// Share only the immutable row store with a bounded prefetch task.
    pub(crate) fn engram_store(&self) -> Result<Rc<RowStore>> {
        ensure!(self.start == 14, "only source14 owns Engram rows");
        self.row_store.clone().ok_or_else(|| anyhow::anyhow!("Engram store absent"))
    }
    /// Engram14 staging rows and their store, for a concurrent lookup.
    pub(crate) fn engram_rows(&mut self) -> Result<(&mut Rows, &RowStore)> {
        ensure!(self.start == 14, "only source14 owns Engram rows");
        Ok((
            self.rows
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("Engram rows absent"))?,
            self.row_store
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("Engram store absent"))?,
        ))
    }
    pub fn prepare_engram(&mut self, hashes: &[[u64; 24]]) -> Result<()> {
        ensure!(self.start == 14, "only source14 owns Engram rows");
        self.rows
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Engram rows absent"))?
            .prepare(
                hashes,
                self.row_store
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Engram store absent"))?,
                8,
            )
    }
    pub fn engram(&self) -> Option<(&Engram, &Rows)> {
        self.engram.as_ref().zip(self.rows.as_ref())
    }
    pub fn reset_swa(&mut self, seed: &[u8]) -> Result<()> {
        for cache in &mut self.caches {
            cache.upload(seed)?;
        }
        Ok(())
    }
    pub(crate) fn clear_state(&mut self, cuda: &Cuda, stream: &Stream) -> Result<()> {
        for cache in &mut self.caches {
            cache.clear(cuda, stream)?;
        }
        self.source.clear_state(stream)
    }
    pub(crate) fn row_store_stats(&self) -> Option<serde_json::Value> {
        self.row_store.as_ref().map(|s| s.stats())
    }
    pub(crate) fn shares_row_store(&self, other: &Self) -> bool {
        match (&self.row_store, &other.row_store) {
            (Some(a), Some(b)) => Rc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        }
    }
    pub(crate) fn layers(&self) -> &[SwaLayer] {
        &self.layers
    }
    pub(crate) fn layers_mut(&mut self) -> &mut [SwaLayer] {
        &mut self.layers
    }
    pub(crate) fn caches(&self) -> &[PagedKv] {
        &self.caches
    }
}
