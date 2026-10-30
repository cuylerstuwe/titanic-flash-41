//! Connected replicated index branch, with retained producer buffers and no
//! Python plan: Q/heads -> FP4 -> paged-key gather -> bounded scores -> top-k.
use crate::{
    blas::Blas,
    blockscaled::Matrix,
    cuda::{Buffer, Cuda, Stream},
    fp8_linear::Linear,
    index::{self, IndexProjection, KeyCache, KeyGather, KeySlots, QueryPack},
    mqa::{self, Chunked, Dense, KeyMatrix, Ranges},
    prelude::FrequencyTable,
    topk::{self, Selection, Windows},
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{cell::Cell, path::Path, rc::Rc};
pub struct Metadata {
    rows: usize,
    zero_based: bool,
    query: index::Schedule,
    ranges: Ranges,
    windows: Windows,
    slots: KeySlots,
}
impl Metadata {
    pub fn update(&mut self, positions: &[i32], starts: &[i32], lengths: &[i32]) -> Result<()> {
        ensure!(
            positions.len() == self.rows && starts.len() == self.rows && lengths.len() == self.rows,
            "connected index metadata update geometry"
        );
        self.query.update(positions, &vec![0; self.rows])?;
        self.ranges.update(starts, lengths)?;
        self.windows.update(&vec![0; self.rows], lengths, starts)?;
        self.zero_based = starts.iter().all(|&s| s == 0);
        Ok(())
    }
    pub(crate) fn windows(&self) -> &Windows {
        &self.windows
    }
    pub fn new(
        cuda: &Cuda,
        cache_capacity: usize,
        columns: usize,
        positions: &[i32],
        slots: &[i32],
        starts: &[i32],
        lengths: &[i32],
        frequency_rows: usize,
    ) -> Result<Self> {
        ensure!(
            crate::config::target_rows(positions.len())
                && starts.len() == positions.len()
                && lengths.len() == positions.len(),
            "connected index metadata profile"
        );
        Ok(Self {
            rows: positions.len(),
            zero_based: starts.iter().all(|&s| s == 0),
            query: index::Schedule::new(
                cuda,
                cache_capacity,
                frequency_rows,
                positions,
                &vec![0; positions.len()],
            )?,
            ranges: Ranges::new(cuda, slots.len(), columns, starts, lengths)?,
            windows: Windows::new(cuda, columns, &vec![0; positions.len()], lengths, starts)?,
            slots: KeySlots::new(cuda, cache_capacity, slots)?,
        })
    }
}
/// Dense keeps the qualified single-arena scorer. Wide rows x long context
/// exceed its budget, so the same scores are produced in bounded row chunks,
/// each selected before the next chunk reuses the arena.
enum Scorer {
    Dense(Dense),
    Chunked(Chunked),
    Grouped(crate::grouped_sparse_index::Grouped),
    Sparse6(crate::sparse_index::Sparse6),
    Adaptive6 { dense: Dense, sparse: crate::sparse_index::Sparse6, active: bool },
}
const SCORE_BUDGET: usize = 64 * 1024 * 1024;
/// s352: compact selection preserves the target and saves warm Verify6 time
/// at the serving allocation size. The paired probe keeps an expanded control.
pub(crate) fn compact_sparse_index_enabled(columns:usize,adaptive:bool)->Result<bool> {
    let requested=match std::env::var("CANADA_SCRATCH_COMPACT_INDEX") {
        Ok(value)=>{ensure!(value=="0"||value=="1","invalid compact index policy");value=="1"},
        Err(std::env::VarError::NotPresent)=>
            std::env::var("CANADA_SCRATCH_COMPACT_INDEX_AB").as_deref()!=Ok("1"),
        Err(error)=>return Err(error.into()),
    };
    Ok(requested && adaptive && columns==1048576)
}
pub(crate) fn selected_sort_shuffle_enabled() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_SORT_SHUFFLE") {
        Ok(value) => {
            ensure!(value == "0" || value == "1", "invalid selected-index sort policy");
            Ok(value == "1")
        }
        Err(std::env::VarError::NotPresent) => Ok(true),
        Err(error) => Err(error.into()),
    }
}
pub struct Branch {
    q: Linear,
    heads: IndexProjection,
    query: QueryPack,
    gather: KeyGather,
    keys: Rc<KeyMatrix>,
    scorer: Scorer,
    grouped16: Option<crate::grouped_sparse_index::Grouped>,
    grouped16_active: Cell<bool>,
    selection: Selection,
    // Beyond 16384 positions: source20 publishes kept blocks, 24..36 consume them.
    publisher: Option<crate::candidates::Publisher>,
    consumer: Option<crate::candidates::Consumer>,
    sort_selected: Option<crate::cuda::Module>,
    sort_shuffle: bool,
    bounded_gather: bool,
    shard_index_rows: bool,
    key_splits: u32,
}
impl Branch {
    pub(crate) fn grouped16_replay_taps(&self)->Vec<(&'static str,u64,usize)> {
        let Some(grouped)=self.grouped16.as_ref().filter(|_|self.grouped16_active.get()) else{return Vec::new();};
        let mut taps=grouped.replay_taps();
        taps.extend([
            ("query-projection",self.q.output().ptr(),self.q.output().bytes()),
            ("query-heads",self.heads.output().ptr(),self.heads.output().bytes()),
            ("query-payload",self.query.payload().ptr(),self.query.payload().bytes()),
            ("query-scales",self.query.scales().ptr(),self.query.scales().bytes()),
            ("query-weights",self.query.weights().ptr(),self.query.weights().bytes()),
            ("selected",self.selection.indices().ptr(),self.selection.indices().bytes())]);
        taps
    }
    pub(crate) fn has_grouped16(&self)->bool {self.grouped16.is_some()}
    pub(crate) fn grouped16_enabled(&self)->bool {self.grouped16_active.get()}
    pub(crate) fn set_grouped16_for_probe(&self,enabled:bool) {
        if self.grouped16.is_some() {
            self.grouped16_active.set(enabled);
            self.gather.invalidate_cache();
        }
    }
    pub(crate) fn packed_gather_enabled(&self)->bool {self.bounded_gather && self.gather.packed_enabled()}
    pub(crate) fn cached_gather_enabled(&self)->bool {self.bounded_gather && self.gather.cached_enabled()}
    pub(crate) fn invalidate_gather_cache(&self) {self.gather.invalidate_cache();}
    pub(crate) fn prepare_gather_cache(&self,stream:&Stream)->Result<()> {self.gather.prepare_cache(stream)}
    pub(crate) fn set_cached16_for_probe(&self,enabled:bool)->Result<()> {self.gather.set_cached16_for_probe(enabled)}
    pub(crate) fn packed_sparse_gather_enabled(&self)->bool {
        match &self.scorer {
            Scorer::Sparse6(s)|Scorer::Adaptive6{sparse:s,..}=>s.packed_gather_enabled(),
            _=>false,
        }
    }
    pub(crate) fn capture_selector(&self,cuda:&Cuda,root:&Path)->Result<serde_json::Value> {
        use sha2::{Digest,Sha256};
        ensure!(std::env::var("CANADA_SCRATCH_SELECTOR_CAPTURE").as_deref()==Ok("1"),"selector capture admission");
        let compact=self.compact_sparse6() && self.sparse6_active();
        let (stride,mut taps)=if compact {
            match &self.scorer {
                Scorer::Sparse6(s)|Scorer::Adaptive6{sparse:s,active:true,..}=>s.selector_taps()?,
                _=>anyhow::bail!("compact selector kind"),
            }
        }else{(self.score_stride(),vec![("scores",self.scores()?.ptr(),self.scores()?.bytes())])};
        taps.push(("selected",self.selected().ptr(),self.selected().bytes()));
        if let Some(p)=&self.publisher {taps.extend(p.selector_taps());}
        std::fs::create_dir(root)?;let mut records=Vec::new();
        for (name,ptr,bytes) in taps {
            let raw=unsafe{cuda.read_device(ptr,bytes)}?;
            let file=format!("{name}.bin");std::fs::write(root.join(&file),&raw)?;
            records.push(serde_json::json!({"name":name,"file":file,"bytes":bytes,"sha256":format!("{:x}",Sha256::digest(&raw))}));
        }
        Ok(serde_json::json!({"score_stride":stride,"global_score_stride":self.score_stride(),
            "compact":compact,"fused_compact_remap_sort":self.fused_compact_publication(),"taps":records}))
    }
    pub(crate) fn key_splits(&self)->u32 {self.key_splits}
    pub(crate) fn row_sharding_enabled(&self) -> bool {self.shard_index_rows}
    pub(crate) fn cached_grouped_union(&self)->bool {
        matches!(&self.scorer, Scorer::Grouped(grouped) if grouped.cached_ballots())
    }
    pub(crate) fn compact_sparse6(&self)->bool {
        match &self.scorer {
            Scorer::Sparse6(s) | Scorer::Adaptive6{sparse:s,..}=>s.compact_selection_enabled(),
            _=>false,
        }
    }
    pub(crate) fn fused_compact_publication(&self)->bool {
        match &self.scorer {
            Scorer::Sparse6(s) | Scorer::Adaptive6{sparse:s,..}=>s.fused_remap_sort_enabled(),
            _=>false,
        }
    }
    pub(crate) fn enable_compact_sparse6(&mut self)->Result<()> {
        match &mut self.scorer {
            Scorer::Sparse6(s) | Scorer::Adaptive6{sparse:s,..}=>s.enable_compact_selection(self.selection.native()),
            _=>anyhow::bail!("compact selection requires a sparse consumer"),
        }
    }
    pub(crate) fn uses_sparse6(&self) -> bool {matches!(self.scorer,Scorer::Sparse6(_) | Scorer::Adaptive6{..})}
    pub(crate) fn adaptive_sparse6(&self) -> bool {matches!(self.scorer,Scorer::Adaptive6{..})}
    pub(crate) fn sparse6_active(&self) -> bool {
        matches!(self.scorer,Scorer::Sparse6(_) | Scorer::Adaptive6{active:true,..})
    }
    /// Only Target may switch this together with its matching CUDA graph bank.
    pub(crate) fn set_adaptive_sparse6(&mut self, enabled: bool) {
        if let Scorer::Adaptive6{active,..}=&mut self.scorer {*active=enabled;}
    }
    pub(crate) fn set_sort_shuffle_before_capture(&mut self, enabled: bool) -> Result<()> {
        ensure!(self.sort_selected.is_some(), "paired sorting requires sorted selection");
        self.sort_shuffle = enabled;
        Ok(())
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
        key_rows: usize,
        columns: usize,
    ) -> Result<Self> {
        Self::new_rows(
            cuda, stream, weights, blas, layer, kernels, frequency, core, key_rows, columns, 6,
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
        key_rows: usize,
        columns: usize,
        rows: usize,
    ) -> Result<Self> {
        ensure!(
            crate::config::target_rows(rows),
            "index execution row profile"
        );
        ensure!(
            [2, 8, 14, 20, 24, 28, 32, 36].contains(&layer),
            "canonical index-source layer"
        );
        let native = index::Native::open(cuda, &kernels.join("index-v1"))?;
        let module = cuda.module(core)?;
        let prefix = format!("layers.{layer}.attn.indexer");
        let matrix = Matrix::prepare(cuda, &module, stream, &weights, &format!("{prefix}.wq_b"))?;
        let q = Linear::new(
            cuda,
            &kernels.join(format!("fp8-index-q-m{rows}-v1")),
            matrix,
        )?;
        let heads = IndexProjection::new(
            cuda,
            weights,
            blas,
            native.clone(),
            &format!("{prefix}.weights_proj"),
            rows,
        )?;
        let query = QueryPack::new(cuda, native, frequency, rows)?;
        let sparse6 = rows == 6 && [24, 28, 32, 36].contains(&layer)
            && columns > crate::candidates::WINDOW
            && std::env::var("CANADA_SCRATCH_SPARSE_INDEX").as_deref() == Ok("1");
        let adaptive6=sparse6 && std::env::var("CANADA_SCRATCH_SPARSE_INDEX_ADAPTIVE").as_deref()==Ok("1");
        let grouped_requested = match std::env::var("CANADA_SCRATCH_GROUPED_SPARSE128") {
            Ok(value) => {ensure!(value == "0" || value == "1", "invalid grouped sparse policy"); value == "1"},
            // s357: four-device retained-prefix/logit/state screen passed.
            // Keep an explicit control for future paired timing.
            Err(std::env::VarError::NotPresent) => true,
            Err(error) => return Err(error.into()),
        };
        let grouped = grouped_requested && rows == 128 && columns == 1048576
            && [24, 28, 32, 36].contains(&layer);
        let shard_requested = match std::env::var("CANADA_SCRATCH_ROW_SHARDED_INDEX") {
            Ok(value) => {ensure!(value == "0" || value == "1", "invalid index row-shard policy"); value == "1"},
            // s361: exact four-rank logits/state against saved s353 controls.
            Err(std::env::VarError::NotPresent) => true,
            Err(error) => return Err(error.into()),
        };
        let shard_publisher = match std::env::var("CANADA_SCRATCH_ROW_SHARDED_PUBLISHER") {
            Ok(value) => {ensure!(value == "0" || value == "1", "invalid publisher row-shard policy"); value == "1"},
            // s363: exact four-rank logits, masks and state; large fresh
            // long-context gain. Retain control for warm/shorter update checks.
            Err(std::env::VarError::NotPresent) => true,
            Err(error) => return Err(error.into()),
        };
        // Source20 additionally gathers its compact block-ID list, then
        // rebuilds the canonical full mask on each rank for later consumers.
        let shard_index_rows = crate::config::wide_rows(rows) &&
            ((shard_requested && columns == 524288 && [2, 8, 14].contains(&layer))
                || (shard_publisher && columns == 1048576 && layer == 20));
        ensure!(!(sparse6 || grouped) || key_rows == columns, "sparse index requires zero-based full key storage");
        let keys = Rc::new(KeyMatrix::new(cuda,
            if grouped { crate::grouped_sparse_index::key_capacity(columns) }
            else if sparse6 && !adaptive6 { crate::sparse_index::key_capacity(columns) } else { key_rows })?);
        let native = mqa::Native::open(cuda, &kernels.join("mqa-v1"))?;
        let schedule=native.clone();
        let topk_native = topk::Native::open(cuda, &kernels.join("topk-v2"))?;
        let dense_bytes = rows.div_ceil(4) * 4 * columns.div_ceil(128) * 128 * 4;
        let mut scorer = if grouped {
            Scorer::Grouped(crate::grouped_sparse_index::Grouped::new(cuda, core, native,
                topk_native.clone(), query.output(), keys.clone(), columns, SCORE_BUDGET)?)
        } else if adaptive6 {
            let compact=crate::sparse_index::key_capacity(columns);
            let sparse_bytes=6*columns*4+8*compact*4;
            ensure!(dense_bytes+sparse_bytes<=SCORE_BUDGET,"adaptive index combined score budget");
            Scorer::Adaptive6 {
                dense:Dense::new(cuda,native.clone(),query.output(),keys.clone(),columns,dense_bytes)?,
                sparse:crate::sparse_index::Sparse6::new(cuda,core,native,query.output(),
                    Rc::new(KeyMatrix::new(cuda,compact)?),columns,sparse_bytes)?,
                active:false,
            }
        } else if sparse6 {
            Scorer::Sparse6(crate::sparse_index::Sparse6::new(
                cuda, core, native, query.output(), keys.clone(), columns, SCORE_BUDGET,
            )?)
        } else if dense_bytes <= SCORE_BUDGET {
            Scorer::Dense(Dense::new(
                cuda,
                native,
                query.output(),
                keys.clone(),
                columns,
                SCORE_BUDGET,
            )?)
        } else {
            Scorer::Chunked(Chunked::new(
                cuda,
                native,
                query.output(),
                keys.clone(),
                columns,
                SCORE_BUDGET,
            )?)
        };
        match &mut scorer {
            Scorer::Sparse6(s)|Scorer::Adaptive6{sparse:s,..}=>s.configure_packed_gather(kernels)?,
            _=>{},
        }
        let key_splits=schedule.key_splits(match &scorer {
            Scorer::Chunked(c)=>c.rows(0)?.len(),
            Scorer::Grouped(_)=>32,
            _=>rows,
        });
        if shard_index_rows {
            let Scorer::Chunked(chunked) = &scorer else {anyhow::bail!("row sharding needs bounded chunks")};
            let quarter = rows / 4;
            for i in 0..chunked.len() {
                let r = chunked.rows(i)?;
                ensure!(r.start / quarter == (r.end - 1) / quarter,
                    "index chunk straddles a rank's owned rows");
            }
        }
        let stride = match &scorer {
            Scorer::Dense(d) => d.stride(),
            Scorer::Chunked(c) => c.stride(),
            Scorer::Grouped(_) => columns, // absolute-ID publication; no dense scores allocated
            Scorer::Sparse6(s) => s.stride(),
            Scorer::Adaptive6{dense,..} => dense.stride(),
        };
        let selection = Selection::new(
            cuda,
            topk_native.clone(),
            rows,
            columns,
            stride,
            512,
        )?;
        let wide = crate::candidates::needed(columns);
        let local_scores = match std::env::var("CANADA_SCRATCH_LOCAL_CANDIDATE_SCORES") {
            Ok(value) => {ensure!(value == "0" || value == "1", "invalid local candidate scores policy"); value == "1"},
            Err(std::env::VarError::NotPresent) => false,
            Err(error) => return Err(error.into()),
        };
        let publisher = (wide && layer == 20)
            .then(|| crate::candidates::Publisher::new_with_score_shards(cuda, core, kernels, rows, columns,
                if local_scores && shard_index_rows {4} else {1}))
            .transpose()?;
        let consumer = (wide && [24, 28, 32, 36].contains(&layer))
            .then(|| crate::candidates::Consumer::new(cuda, core, rows, columns))
            .transpose()?;
        let serving_grouped16=match std::env::var("CANADA_SCRATCH_GROUPED_SPARSE16") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid compact M16 policy");value=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(error)=>return Err(error.into()),
        };
        ensure!(!serving_grouped16 || !crate::mhc::attention16_overlap_enabled()?,
            "compact M16 currently requires the qualified serial attention schedule");
        let grouped16=if rows==16 && columns==1048576 && consumer.is_some()
            && (serving_grouped16 || std::env::var("CANADA_SCRATCH_GROUPED_SPARSE16_SCREEN").as_deref()==Ok("1")) {
            ensure!(matches!(scorer,Scorer::Dense(_)) && !shard_index_rows && key_rows==columns
                && key_splits==12,"M16 compact comparison requires the native dense12-way schedule");
            let mut grouped=crate::grouped_sparse_index::Grouped::new(cuda,core,schedule,
                topk_native,query.output(),Rc::new(KeyMatrix::new(cuda,
                    crate::grouped_sparse_index::key_capacity16(columns))?),columns,SCORE_BUDGET)?;
            grouped.configure_packed_gather16(kernels)?;
            Some(grouped)
        }else{None};
        let grouped16_active=Cell::new(serving_grouped16 && grouped16.is_some());
        let mut branch=Self {
            q,
            heads,
            query,
            gather: KeyGather::new(cuda, core, kernels, rows)?,
            keys,
            scorer,
            grouped16,grouped16_active,
            selection,
            publisher,
            consumer,
            sort_selected: (std::env::var("CANADA_SCRATCH_SORT_SELECTED").as_deref()==Ok("1")).then_some(module),
            sort_shuffle: selected_sort_shuffle_enabled()?,
            bounded_gather: std::env::var("CANADA_SCRATCH_BOUNDED_INDEX_GATHER").as_deref()==Ok("1"),
            shard_index_rows,
            key_splits,
        };
        if compact_sparse_index_enabled(columns,adaptive6)? {branch.enable_compact_sparse6()?;}
        Ok(branch)
    }
    pub fn run(
        &mut self,
        hidden: &Buffer<bf16>,
        q_lora: &Buffer<bf16>,
        cache: &KeyCache,
        meta: &Metadata,
        stream: &Stream,
    ) -> Result<()> {
        self.run_with_mask(hidden, q_lora, cache, meta, None, stream)
    }
    pub(crate) fn run_with_mask(&mut self, hidden: &Buffer<bf16>, q_lora: &Buffer<bf16>,
        cache: &KeyCache, meta: &Metadata, mask: Option<&Buffer<u8>>, stream: &Stream) -> Result<()> {
        self.q.run(q_lora, stream)?;
        self.finish(hidden, cache, meta, mask, None, None, stream)
    }
    pub(crate) fn run_distributed(&mut self, hidden: &Buffer<bf16>, q_lora: &Buffer<bf16>,
        cache: &KeyCache, meta: &Metadata, live: &Buffer<i32>,
        comm: &crate::roce::Roce, stream: &Stream) -> Result<()> {
        self.q.run(q_lora, stream)?;
        self.finish(hidden, cache, meta, None, Some(comm), Some(live), stream)
    }
    /// Main attention already materializes the same normalized MXFP8 input.
    /// This entry avoids a second quantization when connected to that producer.
    pub fn run_quantized(
        &mut self,
        hidden: &Buffer<bf16>,
        q: &Buffer<u8>,
        scales: &Buffer<u8>,
        cache: &KeyCache,
        meta: &Metadata,
        mask: Option<&Buffer<u8>>,
        stream: &Stream,
    ) -> Result<()> {
        self.q.run_quantized(q, scales, stream)?;
        self.finish(hidden, cache, meta, mask, None, None, stream)
    }
    pub(crate) fn run_quantized_distributed(&mut self, hidden: &Buffer<bf16>, q: &Buffer<u8>,
        scales: &Buffer<u8>, cache: &KeyCache, meta: &Metadata, live: &Buffer<i32>,
        comm: &crate::roce::Roce, stream: &Stream) -> Result<()> {
        self.q.run_quantized(q, scales, stream)?;
        self.finish(hidden, cache, meta, None, Some(comm), Some(live), stream)
    }
    /// Candidate blocks this source published in its latest run, if any.
    pub(crate) fn candidate_mask(&self) -> Option<&Buffer<u8>> {
        self.publisher.as_ref().map(|p| p.mask())
    }
    pub(crate) fn candidate_mask_is_compact(&self) -> bool {
        self.publisher.as_ref().is_some_and(|p| p.compact_mask())
    }
    pub(crate) fn share_candidate_scratch_from(&mut self, prior: &Self) -> Result<()> {
        match (&mut self.publisher, &prior.publisher) {
            (Some(p), Some(previous)) => p.share_scratch_from(previous),
            (None, None) => Ok(()),
            _ => anyhow::bail!("candidate scratch sharing requires matching publisher roles"),
        }
    }
    pub(crate) fn candidate_scratch_report(&self) -> Option<serde_json::Value> {
        self.publisher.as_ref().map(|p| p.scratch_report())
    }
    pub(crate) fn enable_retained_publication(&mut self) -> Result<()> {
        if self.shard_index_rows && let Some(p) = &mut self.publisher {
            p.enable_retained_publication()?;
        }
        Ok(())
    }
    fn finish(
        &mut self,
        hidden: &Buffer<bf16>,
        cache: &KeyCache,
        meta: &Metadata,
        mask: Option<&Buffer<u8>>,
        comm: Option<&crate::roce::Roce>,
        live: Option<&Buffer<i32>>,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            self.consumer.is_none() || mask.is_some(),
            "candidate consumer needs source20's published blocks"
        );
        let owned_rows = if self.shard_index_rows {
            let comm = comm.ok_or_else(|| anyhow::anyhow!("row-sharded index requires collective owner"))?;
            ensure!(comm.rank() < 4 && self.consumer.is_none(),
                "row-sharded index ownership");
            let quarter = meta.rows / 4;
            Some(comm.rank() * quarter..(comm.rank() + 1) * quarter)
        } else {None};
        if let Some(p) = &mut self.publisher {
            p.prepare_retained_windows(&meta.windows, live, comm.map(|c|c.rank()), stream)?;
        }
        self.heads.run(hidden, stream)?;
        self.query
            .run(self.q.output(), self.heads.output(), &meta.query, stream)?;
        let grouped16_active=self.grouped16_active.get();
        if !grouped16_active && !self.sparse6_active() && !matches!(self.scorer, Scorer::Grouped(_)) {
            self.gather.run_with_bound(cache, &meta.slots, &self.keys,
                self.bounded_gather.then_some(meta.ranges.key_ends()), stream)?;
        }
        // Block maxima read the raw scores and consumers mask them, both
        // before top-k consumes the row slice.
        let level_one = |scores: &mut Buffer<f32>, stride: usize, rows: std::ops::Range<usize>| -> Result<()> {
            if let Some(p) = &self.publisher {
                p.score_rows(scores, stride, &meta.windows, rows.clone(), stream)?;
            }
            if let (Some(c), Some(m)) = (&self.consumer, mask) {
                c.apply(scores, stride, &meta.windows, m, rows, stream)?;
            }
            Ok(())
        };
        if grouped16_active {
            ensure!(meta.zero_based,"M16 compact requires zero-based ranges");
            self.grouped16.as_mut().expect("prepared M16 compact scorer").run(cache,&meta.slots,&meta.windows,
                mask.ok_or_else(||anyhow::anyhow!("M16 compact source mask missing"))?,&self.selection,stream)?;
        }else{match &mut self.scorer {
            Scorer::Dense(dense) | Scorer::Adaptive6{dense,active:false,..} => {
                dense.run(&meta.ranges, stream)?;
                let stride = dense.stride();
                level_one(dense.output_mut(), stride, 0..meta.windows.rows())?;
                self.selection
                    .run(dense.output_mut(), &meta.windows, stream)?;
            }
            Scorer::Chunked(chunked) => {
                let stride = chunked.stride();
                for index in 0..chunked.len() {
                    let rows = chunked.rows(index)?;
                    if owned_rows.as_ref().is_some_and(|owned| rows.start < owned.start || rows.end > owned.end) {
                        continue;
                    }
                    chunked.run_chunk(index, &meta.ranges, stream)?;
                    level_one(chunked.output_mut(), stride, rows.clone())?;
                    self.selection
                        .run_rows(chunked.output_mut(), &meta.windows, rows, stream)?;
                }
            }
            Scorer::Sparse6(sparse) | Scorer::Adaptive6{sparse,active:true,..} => {
                ensure!(meta.zero_based, "sparse index profile requires zero-based ranges");
                sparse.run(cache, &meta.slots, &meta.windows,
                    mask.ok_or_else(|| anyhow::anyhow!("sparse index missing source mask"))?, stream)?;
                sparse.select(&self.selection, &meta.windows, stream)?;
            }
            Scorer::Grouped(grouped) => {
                ensure!(meta.zero_based, "grouped sparse requires zero-based ranges");
                grouped.run(cache, &meta.slots, &meta.windows,
                    mask.ok_or_else(|| anyhow::anyhow!("grouped sparse missing source mask"))?,
                    &self.selection, stream)?;
            }
        }}
        if let Some(rows) = &owned_rows {
            self.selection.gather_row_shard(rows.clone(), comm.expect("checked row-shard communicator"), stream)?;
        }
        let already_sorted=self.sparse6_active() && self.fused_compact_publication();
        if let Some(module) = self.sort_selected.as_ref().filter(|_|!already_sorted) {
            let mut indices = self.selection.indices().ptr();
            // Only the lengths passed to the native selector establish its
            // <=512 ordered-output guarantee. Compact unions can exceed512
            // while a member causal row is shorter. Skipping its sort leaves
            // native selection order variable. The short-row guarantee must
            // cover the actual producer before we rely on ordered IDs/padding.
            let mut lengths = if grouped16_active {
                self.grouped16.as_ref().unwrap().selection_lengths()
            }else{match &self.scorer {
                Scorer::Grouped(grouped)=>grouped.selection_lengths(),
                Scorer::Sparse6(s)|Scorer::Adaptive6{sparse:s,active:true,..}=>
                    s.compact_selection_lengths().unwrap_or(meta.windows.lengths()),
                _=>meta.windows.lengths(),
            }}.ptr();
            let mut args = vec![(&mut indices as *mut u64).cast()];
            let name = if self.sort_shuffle {
                args.push((&mut lengths as *mut u64).cast());
                "ca_sort_selected512_native"
            } else { "ca_sort_selected512" };
            unsafe {module.function(name)?.launch(
                [meta.windows.rows() as u32,1,1],[512,1,1],0,stream,&mut args)?;}
        }
        if let Scorer::Sparse6(sparse) | Scorer::Adaptive6{sparse,active:true,..} = &self.scorer {
            // Sorting normalizes negative padding to -1; poison afterwards so
            // the downstream mapper sees the deliberately invalid -2 marker.
            sparse.poison_selection_on_error(self.selection.indices(), stream)?;
        }
        if let Scorer::Grouped(grouped) = &self.scorer {
            grouped.poison_selection_on_error(self.selection.indices(), stream)?;
        }
        if grouped16_active {
            self.grouped16.as_ref().unwrap().poison_selection_on_error(self.selection.indices(),stream)?;
        }
        if let Some(p) = &mut self.publisher {
            if let Some(rows)=owned_rows {
                p.publish_sharded(&meta.windows,rows,comm.expect("checked publisher communicator"),live,stream)?;
            }else{p.publish(&meta.windows, stream)?;}
        }
        Ok(())
    }
    pub fn query_projection(&self) -> &Buffer<bf16> {
        self.q.output()
    }
    pub fn head_projection(&self) -> &Buffer<bf16> {
        self.heads.output()
    }
    pub fn packed_query(&self) -> &QueryPack {
        &self.query
    }
    pub fn gathered_keys(&self) -> &KeyMatrix {
        match &self.scorer {
            Scorer::Adaptive6{sparse,active:true,..}=>sparse.gathered_keys(),
            _=>&self.keys,
        }
    }
    /// Diagnostic only: the complete matrix for Dense, the last row chunk otherwise.
    pub fn scores(&self) -> Result<&Buffer<f32>> {
        ensure!(!self.compact_sparse6() || !self.sparse6_active(), "compact selection does not materialize the diagnostic full score matrix");
        Ok(match &self.scorer {
            Scorer::Dense(d) | Scorer::Adaptive6{dense:d,active:false,..} => d.output(),
            Scorer::Chunked(c) => c.output(),
            Scorer::Grouped(_) => anyhow::bail!("grouped sparse does not materialize the full diagnostic scores"),
            Scorer::Sparse6(s) | Scorer::Adaptive6{sparse:s,active:true,..} => s.output(),
        })
    }
    pub fn score_stride(&self) -> usize {
        match &self.scorer {
            Scorer::Dense(d) | Scorer::Adaptive6{dense:d,active:false,..} => d.stride(),
            Scorer::Chunked(c) => c.stride(),
            Scorer::Grouped(grouped) => grouped.columns(),
            Scorer::Sparse6(s) | Scorer::Adaptive6{sparse:s,active:true,..} => s.stride(),
        }
    }
    pub fn selected(&self) -> &Buffer<i32> {
        self.selection.indices()
    }
    pub fn selection(&self) -> &Selection {
        &self.selection
    }
}
