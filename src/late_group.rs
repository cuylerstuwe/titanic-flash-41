//! Ratio1 target graph: one source20 KV/index-key owner and four independent
//! read-only indexers. Every layer retains its own SWA keys. Initial six-row
//! all-candidates trace geometry; not yet full context or prefill qualification.
use crate::{
    attention::{Candidates, PagedKv},
    blas::Blas,
    checkpoint::Checkpoint,
    compressed_source::{Layer as Source, Metadata, Phase, ReadIndex},
    cuda::{Buffer, Cuda, Module, Stream},
    decoder::SwaLayer,
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
    // Wide prefill profiles own only encoder layer20 and its source; the
    // decoder layers21..39 run in the 128-row tail profile.
    encoder_only: bool,
    layers: Vec<SwaLayer>,
    caches: Vec<PagedKv>,
    source: Source,
    indices: Vec<ReadIndex>,
    pub atlas_bytes: Vec<usize>,
    pub source_hashes: Vec<String>,
}
impl Group {
    pub(crate) fn enable_compact_sparse6(&mut self)->Result<()> {
        for index in &mut self.indices {index.enable_compact_sparse6()?;}
        Ok(())
    }
    pub(crate) fn set_adaptive_sparse6(&mut self, enabled: bool) {
        for index in &mut self.indices {index.set_adaptive_sparse6(enabled);}
    }
    pub(crate) fn set_sort_shuffle_before_capture(&mut self, enabled: bool) -> Result<()> {
        self.source.set_sort_shuffle_before_capture(enabled)?;
        for index in &mut self.indices {
            index.set_sort_shuffle_before_capture(enabled)?;
        }
        Ok(())
    }
    pub fn new(
        cuda: &Cuda,
        module: &Module,
        core: &[u8],
        stream: &Stream,
        cp: &Checkpoint,
        plan: &RankPlan,
        stages: &[PathBuf],
        kernels: &Path,
        frequency: Rc<FrequencyTable>,
        slots: &[i32],
        kv_pages: usize,
        index_pages: usize,
    ) -> Result<Self> {
        Self::new_with_phase(
            cuda,
            module,
            core,
            stream,
            cp,
            plan,
            stages,
            kernels,
            frequency,
            slots,
            kv_pages,
            index_pages,
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
        stages: &[PathBuf],
        kernels: &Path,
        frequency: Rc<FrequencyTable>,
        slots: &[i32],
        kv_pages: usize,
        index_pages: usize,
        phase: Phase,
    ) -> Result<Self> {
        Self::new_rows(
            cuda,
            module,
            core,
            stream,
            cp,
            plan,
            stages,
            kernels,
            frequency,
            slots,
            kv_pages,
            index_pages,
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
        stages: &[PathBuf],
        kernels: &Path,
        frequency: Rc<FrequencyTable>,
        slots: &[i32],
        kv_pages: usize,
        index_pages: usize,
        phase: Phase,
        num_rows: usize,
        vendor_blas: Option<Rc<Blas>>,
        medium_woa: Option<Rc<std::cell::RefCell<crate::woa_medium::Scratch>>>,
        swa_pages: usize,
        mut reserved: Option<&mut crate::weights::StagedReservation>,
    ) -> Result<Self> {
        ensure!(
            [4, 8, 12, 16, 20].contains(&stages.len()) && frequency.compressed(),
            "late target group geometry"
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
        let mut indices = Vec::new();
        let mut atlas_bytes = Vec::new();
        let mut source_hashes = Vec::new();
        let blas = vendor_blas.clone();
        for (i, stage) in stages.iter().enumerate() {
            let id = i + 20;
            let mut w = Weights::load_staged_reserved(cuda, cp, plan, &[&format!("layers.{id}.")], stage, reserved.as_deref_mut())?;
            atlas_bytes.push(w.bytes());
            source_hashes.push(w.source_sha256.clone());
            let prepared = expert_packing::prepare_layer(cuda, module, &mut w, id)?;
            let w = Rc::new(w);
            layers.push(SwaLayer::new_rows(
                cuda,
                module,
                stream,
                w.clone(),
                prepared,
                kernels,
                frequency.clone(),
                InputMode::Split(plan.rank),
                num_rows,
                Some(vendor_blas.clone()),
                medium_woa.clone(),
            )?);
            if id == 20 {
                source = Some(Source::new_rows(
                    cuda,
                    stream,
                    w,
                    blas.clone(),
                    20,
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
            } else if id % 4 == 0 {
                indices.push(ReadIndex::new(
                    cuda,
                    stream,
                    w,
                    blas.clone(),
                    id,
                    kernels,
                    frequency.clone(),
                    core,
                    source
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("source20 construction order"))?,
                    slots,
                    &vec![0; num_rows],
                )?);
            }
            caches.push(PagedKv::new(cuda, swa_pages, 64)?);
        }
        Ok(Self {
            encoder_only: false,
            layers,
            caches,
            source: source.ok_or_else(|| anyhow::anyhow!("source20 absent"))?,
            indices,
            atlas_bytes,
            source_hashes,
        })
    }
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
        ensure!(!self.encoder_only, "fork from a complete late group");
        let mut layers = Vec::new();
        let mut source = None;
        let mut indices = Vec::new();
        let encoder_only = wide.is_some();
        for (i, old) in self.layers.iter().enumerate() {
            let id = 20 + i;
            if encoder_only && i > 0 {
                break;
            }
            let weights = old.shared_weights();
            let build = |moe: Option<Rc<crate::moe::Workspace>>, sp: Option<usize>| {
                SwaLayer::new_rows_in(
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
                )
            };
            layers.push(match wide {
                Some(w) => w.layer(cuda, id, |moe, sp| build(Some(moe), sp))?,
                None => build(narrow_moe.clone(), None)?,
            });
            if i == 0 {
                source = Some(Source::new_rows_owner(
                    cuda,
                    stream,
                    weights,
                    vendor_blas.clone(),
                    20,
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
            } else if id % 4 == 0 {
                indices.push(ReadIndex::new(
                    cuda,
                    stream,
                    weights,
                    vendor_blas.clone(),
                    id,
                    kernels,
                    frequency.clone(),
                    core,
                    source
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("shared source20 absent"))?,
                    slots,
                    &vec![0; num_rows],
                )?);
            }
        }
        let mut source = source.ok_or_else(|| anyhow::anyhow!("shared source20 absent"))?;
        if encoder_only {
            source.enable_retained_publication()?;
        }
        Ok(Self {
            encoder_only,
            layers,
            caches: if independent {self.caches.iter()
                .map(|c|PagedKv::new(cuda,c.pages(),c.page_tokens())).collect::<Result<Vec<_>>>()?}
                else {self.caches.clone()},
            source,
            indices,
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
        ensure!(!self.encoder_only, "complete late run needs decoder layers");
        let mut residual = input;
        let mut previous = pre;
        for (i, (layer, cache)) in self.layers.iter_mut().zip(&mut self.caches).enumerate() {
            if i == 0 {
                layer.run_indexed_from_previous(
                    residual,
                    previous,
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
            } else if i < 4 {
                layer.run_shared_from_previous(
                    residual,
                    previous,
                    positions,
                    locations,
                    cache,
                    candidates,
                    tokens,
                    &self.source,
                    comm,
                    stream,
                )?;
            } else if i % 4 == 0 {
                layer.run_reindexed_from_previous(
                    residual,
                    previous,
                    positions,
                    locations,
                    cache,
                    candidates,
                    tokens,
                    &self.source,
                    &mut self.indices[i / 4 - 1],
                    metadata,
                    None,
                    comm,
                    stream,
                )?;
            } else {
                layer.run_selected_from_previous(
                    residual,
                    previous,
                    positions,
                    locations,
                    cache,
                    candidates,
                    tokens,
                    &self.source,
                    &self.indices[i / 4 - 1],
                    comm,
                    stream,
                )?;
            }
            residual = layer.output();
            previous = layer.pre();
        }
        Ok(())
    }
    pub(crate) fn run_encoder(
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
        self.layers[0].run_indexed_from_previous(
            input,
            pre,
            positions,
            locations,
            &mut self.caches[0],
            candidates,
            tokens,
            &mut self.source,
            metadata,
            comm,
            stream,
        )
    }
    pub(crate) fn encoder_deferral_supported(&self) -> bool {
        self.layers[0].indexed_deferral_supported()
    }
    /// Deferred layer 20: projection and source20 keys only (see SwaLayer).
    pub(crate) fn run_encoder_projection(
        &mut self,
        input: &Buffer<bf16>,
        pre: &Buffer<f32>,
        positions: &Positions,
        locations: &RowLocations,
        metadata: &Metadata,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        self.layers[0].run_indexed_projection(
            input,
            pre,
            positions,
            locations,
            &mut self.caches[0],
            &mut self.source,
            metadata,
            comm,
            stream,
        )
    }
    /// The rest of run_encoder after run_encoder_projection, unchanged.
    pub(crate) fn finish_encoder_deferred(
        &mut self,
        input: &Buffer<bf16>,
        positions: &Positions,
        candidates: &Candidates,
        tokens: &TokenRows,
        metadata: &Metadata,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        self.layers[0].finish_indexed_deferred(
            input,
            positions,
            &self.caches[0],
            candidates,
            tokens,
            &mut self.source,
            metadata,
            comm,
            stream,
        )
    }
    pub(crate) fn encoder_output(&self) -> &Buffer<bf16> {
        self.layers[0].output()
    }
    pub(crate) fn encoder_pre(&self) -> &Buffer<f32> {
        self.layers[0].pre()
    }
    pub(crate) fn encoder_sequence_parallel(&self) -> bool {
        self.layers[0].sp_rank().is_some()
    }
    pub(crate) fn run_decoder(
        &mut self,
        input: Option<(&Buffer<bf16>, &Buffer<f32>, &Candidates, Option<&Buffer<u8>>)>,
        positions: &Positions,
        locations: &RowLocations,
        candidates: &Candidates,
        tokens: &TokenRows,
        metadata: &Metadata,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(!self.encoder_only, "decoder tail needs decoder layers");
        let (encoder, layers) = self
            .layers
            .split_first_mut()
            .ok_or_else(|| anyhow::anyhow!("decoder encoder20 absent"))?;
        let (_, caches) = self.caches.split_first_mut().unwrap();
        let (mut residual, mut pre, source_candidates, replay_mask) =
            input.unwrap_or((encoder.output(), encoder.pre(), self.source.candidates(), None));
        for (i, (layer, cache)) in layers.iter_mut().zip(caches).enumerate() {
            let id = 21 + i;
            if id < 24 {
                layer.run_shared_candidates_from_previous(
                    residual,
                    pre,
                    positions,
                    locations,
                    cache,
                    candidates,
                    tokens,
                    &self.source,
                    source_candidates,
                    comm,
                    stream,
                )?;
            } else if id % 4 == 0 {
                layer.run_reindexed_from_previous(
                    residual,
                    pre,
                    positions,
                    locations,
                    cache,
                    candidates,
                    tokens,
                    &self.source,
                    &mut self.indices[(id - 24) / 4],
                    metadata,
                    replay_mask,
                    comm,
                    stream,
                )?;
            } else {
                layer.run_selected_from_previous(
                    residual,
                    pre,
                    positions,
                    locations,
                    cache,
                    candidates,
                    tokens,
                    &self.source,
                    &self.indices[(id / 4 * 4 - 24) / 4],
                    comm,
                    stream,
                )?;
            }
            residual = layer.output();
            pre = layer.pre();
        }
        Ok(())
    }
    /// Ordinary source20/decoder attention, one layer at a time for a pair.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_pair_layer(&mut self,index:usize,input:&Buffer<bf16>,pre:&Buffer<f32>,
        positions:&Positions,locations:&RowLocations,candidates:&Candidates,tokens:&TokenRows,
        metadata:&Metadata,comm:&Roce,stream:&Stream)->Result<()> {
        ensure!(index<20&&!self.encoder_only&&!self.encoder_sequence_parallel(),"paired late layer domain");
        let (previous,rest)=self.layers.split_at_mut(index);let layer=&mut rest[0];
        let (input,pre)=previous.last().map_or((input,pre),|p|(p.output(),p.pre()));
        let id=index+20;let cache=&mut self.caches[index];
        if index==0 {
            layer.run_indexed_from_previous(input,pre,positions,locations,cache,candidates,tokens,
                &mut self.source,metadata,comm,stream)
        }else if id<24 {
            layer.run_shared_candidates_from_previous(input,pre,positions,locations,cache,candidates,tokens,
                &self.source,self.source.candidates(),comm,stream)
        }else if id%4==0 {
            layer.run_reindexed_from_previous(input,pre,positions,locations,cache,candidates,tokens,
                &self.source,&mut self.indices[(id-24)/4],metadata,None,comm,stream)
        }else{
            layer.run_selected_from_previous(input,pre,positions,locations,cache,candidates,tokens,
                &self.source,&self.indices[(id/4*4-24)/4],comm,stream)
        }
    }
    pub fn output(&self) -> &Buffer<bf16> {
        self.layers.last().unwrap().output()
    }
    pub fn pre(&self) -> &Buffer<f32> {
        self.layers.last().unwrap().pre()
    }
    pub fn source(&self) -> &Source {
        &self.source
    }
    pub(crate) fn source_mut(&mut self) -> &mut Source {
        &mut self.source
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
    pub(crate) fn share_consumer_score_scratch(&mut self, storage: Rc<Buffer<f32>>) -> Result<usize> {
        let mut bytes=0;
        for index in &mut self.indices {bytes+=index.share_score_scratch(storage.clone())?;}
        Ok(bytes)
    }
    pub(crate) fn index(&self, layer: usize) -> Option<&ReadIndex> {
        self.indices.iter().find(|i| i.layer_id() == layer)
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
}
