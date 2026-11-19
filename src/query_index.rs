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
    /// Region steps with CANADA_SCRATCH_REGION_GATHER_CACHE=1: each row's first key row to gather.
    gather_starts: Buffer<i32>,
    /// Host copies of each row's key range (the gather check's ranges).
    host_ranges: Vec<(usize, usize)>,
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
        self.host_ranges = starts.iter().zip(lengths).map(|(&s, &l)| (s.max(0) as usize, (s + l).max(0) as usize)).collect();
        Ok(())
    }
    /// Every row's key range holds at most `keys` keys (host copy of the latest update).
    pub(crate) fn rows_within(&self, keys: usize) -> bool {
        self.host_ranges.len() == self.rows && self.host_ranges.iter().all(|&(s, e)| e.saturating_sub(s) <= keys)
    }
    /// Region steps: each row's first key row to gather (its range start for a full gather).
    pub(crate) fn set_gather_starts(&mut self, starts: &[i32]) -> Result<()> {
        ensure!(starts.len() == self.rows && starts.iter().zip(&self.host_ranges)
            .all(|(&g, &(s, e))| g >= 0 && (g as usize) >= s && (g as usize) <= e.max(s)), "region gather starts");
        self.gather_starts.upload(starts)
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
            gather_starts: { let mut b = cuda.allocate::<i32>(positions.len())?; b.upload(starts)?; b },
            host_ranges: starts.iter().zip(lengths).map(|(&s, &l)| (s.max(0) as usize, (s + l).max(0) as usize)).collect(),
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
    key_sharded: Option<crate::key_sharded_index::KeySharded>,
    key_sharded_active: Cell<bool>,
    key_sharded_auto: Cell<bool>,
    key_sharded_min_position: u64,
    compact_verifier: Option<crate::grouped_sparse_index::Grouped>,
    compact_verifier_active: Cell<bool>,
    /// Serving regions: this step's rows have non-zero key starts (see set_region_mode).
    region_mode: Cell<bool>,
    /// config::index_short_rows: set by the owning target around the enqueue of a step whose rows all have at most
    /// 512 keys (Target::begin_verification); see `short_step`.
    short_rows: Rc<Cell<bool>>,
    selection: Selection,
    // Beyond 16384 positions: source20 publishes kept blocks, 24..36 consume them.
    publisher: Option<crate::candidates::Publisher>,
    consumer: Option<crate::candidates::Consumer>,
    sort_selected: Option<crate::cuda::Module>,
    sort_shuffle: bool,
    bounded_gather: bool,
    shard_index_rows: bool,
    key_splits: u32,
    /// config::index_bf16: this dense layer's scores and selection with the reference's bf16 arithmetic.
    index16: Option<crate::index_bf16::IndexBf16>,
}
impl Branch {
    /// Plan screen: this branch's candidate consumer (if any) with the bounded mask kernel on or off.
    pub(crate) fn set_bounded_mask_for_probe(&self, on: bool) -> Result<bool> {
        match &self.consumer { Some(c) => { c.set_bounded_for_probe(on)?; Ok(true) } None => Ok(false) }
    }
    /// The index query projection's immutable weights (hint regions for config::l2_queue_index_lead).
    pub(crate) fn query_matrix(&self) -> Rc<crate::blockscaled::Matrix> { self.q.immutable_matrix() }
    pub(crate) fn set_query_flash_for_probe(&self,enabled:bool)->Result<()> {
        self.q.set_index_query_flash_for_probe(enabled)
    }
    pub(crate) fn has_key_sharded(&self)->bool {self.key_sharded.is_some()}
    pub(crate) fn has_key_sharded_publisher(&self)->bool {self.key_sharded.as_ref().is_some_and(|p|p.block_ids().is_some())}
    pub(crate) fn key_sharded_active(&self)->bool {self.key_sharded_active.get()}
    pub(crate) fn set_key_sharded_auto(&self,enabled:bool) {
        self.key_sharded_auto.set(enabled&&self.key_sharded.is_some());
    }
    pub(crate) fn key_sharded_eligible_at(&self,position:u64)->bool {
        position>=self.key_sharded_min_position
    }
    pub(crate) fn key_sharded_transition(&self,position:u64)->Option<bool> {
        let desired=self.key_sharded_eligible_at(position);
        (self.key_sharded_auto.get()&&desired!=self.key_sharded_active.get()).then_some(desired)
    }
    pub(crate) fn set_key_sharded_for_probe(&self,enabled:bool) {
        if let Some(shard)=&self.key_sharded {
            self.key_sharded_active.set(enabled);shard.invalidate();self.gather.invalidate_cache();
        }
    }
    pub(crate) fn grouped16_replay_taps(&self)->Vec<(&'static str,u64,usize)> {
        let grouped=match &self.scorer {
            Scorer::Grouped(grouped) if self.query.output().rows()==16=>Some(grouped),
            _=>self.compact_verifier.as_ref().filter(|_|self.compact_verifier_active.get()),
        };
        let Some(grouped)=grouped else{return Vec::new();};
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
    pub(crate) fn has_grouped16(&self)->bool {self.query.output().rows()==16&&self.compact_verifier.is_some()}
    pub(crate) fn grouped16_enabled(&self)->bool {self.query.output().rows()==16
        &&(self.compact_verifier_active.get()||matches!(self.scorer,Scorer::Grouped(_)))}
    pub(crate) fn has_compact32(&self)->bool {self.query.output().rows()==32&&self.compact_verifier.is_some()}
    pub(crate) fn compact32_active(&self)->bool {self.has_compact32()&&self.compact_verifier_active.get()}
    pub(crate) fn compact_index_enabled(&self)->bool {
        self.compact_verifier_active.get()||matches!(self.scorer,Scorer::Grouped(_))
    }
    pub(crate) fn set_compact32_for_probe(&self,enabled:bool)->Result<()> {
        self.gather.set_packed32_for_probe(enabled)?;
        if self.has_compact32() {self.compact_verifier_active.set(enabled);}
        Ok(())
    }
    /// Serving regions (target.rs SessionBase): a batched step's rows start at their region's
    /// keys, which only the dense scorer handles. In region mode a step runs the dense
    /// alternative instead of the adaptive sparse, compact (grouped) or key-sharded scorer,
    /// without changing their states; the next ordinary step uses them again. The target
    /// captures region-mode steps in their own graph slots.
    /// Full serving regions with elastic storage: pool chunks mapping region k's key rows would take (0 when this
    /// branch's keys hold one region).
    pub(crate) fn region_key_missing_chunks(&self, k: usize) -> Result<usize> { self.keys.region_missing_chunks(k) }
    pub(crate) fn map_region_keys(&self, k: usize) -> Result<bool> { self.keys.map_region(k) }
    pub(crate) fn unmap_region_keys(&self, k: usize) -> Result<()> { self.keys.unmap_region(k) }
    pub(crate) fn key_regions(&self) -> usize { self.keys.regions() }
    pub(crate) fn set_region_mode(&self,enabled:bool)->Result<()> {
        ensure!(!enabled||!matches!(self.scorer,Scorer::Grouped(_)|Scorer::Sparse6(_)),
            "region mode needs a dense index scorer");
        self.region_mode.set(enabled);Ok(())
    }
    pub(crate) fn set_grouped16_for_probe(&self,enabled:bool) {
        if self.compact_verifier.is_some() {
            self.compact_verifier_active.set(enabled);
            self.gather.invalidate_cache();
        }
    }
    pub(crate) fn packed_gather_enabled(&self)->bool {self.bounded_gather && self.gather.packed_enabled()}
    pub(crate) fn cached_gather_enabled(&self)->bool {self.bounded_gather && self.gather.cached_enabled()}
    pub(crate) fn invalidate_gather_cache(&self) {
        self.gather.invalidate_cache();if let Some(shard)=&self.key_sharded {shard.invalidate();}
    }
    pub(crate) fn prepare_gather_cache(&self,stream:&Stream)->Result<()> {
        self.gather.prepare_cache(stream)?;
        if let Some(shard)=&self.key_sharded {shard.prepare(stream)?;}Ok(())
    }
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
            cuda, stream, weights, blas, layer, kernels, frequency, core, key_rows, columns, 6, None,
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
        key_owner: Option<&Self>,
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
        let serving_grouped32=match std::env::var("CANADA_SCRATCH_GROUPED_SPARSE32") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid compact M32 policy");value=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(error)=>return Err(error.into()),
        };
        let serving_grouped16=match std::env::var("CANADA_SCRATCH_GROUPED_SPARSE16") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid compact M16 policy");value=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(error)=>return Err(error.into()),
        };
        let selected_grouped16_storage=match std::env::var("CANADA_SCRATCH_SELECTED_GROUPED16_STORAGE_SCREEN") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid selected M16 storage policy");value=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(error)=>return Err(error.into()),
        } && serving_grouped16
            &&std::env::var("CANADA_SCRATCH_GROUPED_SPARSE16_SCREEN").as_deref()!=Ok("1");
        let compact32_screen=std::env::var("CANADA_SCRATCH_COMPACT32_INDEX_SCREEN").as_deref()==Ok("1");
        let grouped_requested = match std::env::var("CANADA_SCRATCH_GROUPED_SPARSE128") {
            Ok(value) => {ensure!(value == "0" || value == "1", "invalid grouped sparse policy"); value == "1"},
            // s357: four-device retained-prefix/logit/state screen passed.
            // Keep an explicit control for future paired timing.
            Err(std::env::VarError::NotPresent) => true,
            Err(error) => return Err(error.into()),
        };
        // The selected-only M16 experiment removes an unused dense key matrix,
        // retaining the identical compact scorer. Explicit A/B probes still own
        // both alternatives, so they can switch without replacing graph inputs.
        let grouped = ((grouped_requested&&rows==128)
            ||(rows==16&&selected_grouped16_storage)
            ||(rows==32&&serving_grouped32&&!compact32_screen))
            &&columns==1048576&&[24,28,32,36].contains(&layer);
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
        // Full serving regions (config::full_regions): the batched owner's key storage spans every region (key_rows =
        // regions x columns, region k at k x columns); region 0 stays zero-based, so the scorers that need zero-based
        // storage still see it on single-conversation steps (region steps use the dense scorer).
        ensure!(key_rows % columns == 0 && (key_rows == columns || crate::config::full_regions()?),
            "region key storage needs full serving regions");
        // An unbounded gather walks every slot row, the empty (unmapped) regions' too: region storage needs the bound.
        ensure!(key_rows == columns || std::env::var("CANADA_SCRATCH_BOUNDED_INDEX_GATHER").as_deref() == Ok("1"),
            "region key storage needs the bounded index gather");
        ensure!(!(sparse6 || grouped) || key_rows % columns == 0, "sparse index requires zero-based full key storage");
        let key_capacity=if grouped&&rows==16 {crate::grouped_sparse_index::key_capacity16(columns)}
            else if grouped {crate::grouped_sparse_index::key_capacity(columns)}
            else if sparse6&&!adaptive6 {crate::sparse_index::key_capacity(columns)} else{key_rows};
        let keys=if let Some(owner)=key_owner {
            // The source constructor has already checked this exact causal
            // cache/weight owner. Wide and M128 gathers rebuild all reachable
            // keys on the same ordered stream; no decode gather cache is lent.
            ensure!(std::env::var("CANADA_SCRATCH_WIDE_INDEX_KEYS_SCREEN").as_deref()==Ok("1")
                &&crate::config::wide_rows(rows)&&[2,8,14,20].contains(&layer)
                &&!grouped&&!sparse6&&!owner.gather.cached_enabled()
                &&owner.keys.rows()==key_capacity&&cuda.owns_buffer(&owner.keys.payload),
                "sequential wide gathered-key loan geometry/owner");
            owner.keys.clone()
        }else if key_capacity==key_rows&&key_rows>columns {
            Rc::new(KeyMatrix::new_regions(cuda,columns,key_rows/columns,crate::config::elastic_regions()?)?)
        }else{Rc::new(KeyMatrix::new(cuda,key_capacity)?)};
        // Region key storage (full serving regions) exceeds mqa-v1's 1,048,576-row adapter limit: mqa-v2 is the same
        // scorer with the limit raised to the regions' bound.
        let native = mqa::Native::open(cuda, &kernels.join(if key_rows > 1048576 {"mqa-v2"} else {"mqa-v1"}))?;
        let schedule=native.clone();
        let topk_native = topk::Native::open(cuda, &kernels.join("topk-v2"))?;
        let key_sharded_requested=match std::env::var("CANADA_SCRATCH_KEY_SHARDED_INDEX") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid key-shard policy");value=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(error)=>return Err(error.into()),
        };
        let publisher_screen=std::env::var("CANADA_SCRATCH_KEY_SHARDED_PUBLISHER_SCREEN").as_deref()==Ok("1");
        ensure!(!publisher_screen||std::env::var("CANADA_SCRATCH_KEY_SHARDED_INDEX_SCREEN").as_deref()==Ok("1"),"publisher partition is an isolated index screen");
        let publisher_requested=match std::env::var("CANADA_SCRATCH_KEY_SHARDED_PUBLISHER") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid key publisher policy");value=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(error)=>return Err(error.into()),
        };
        let shard_geometry=(columns==524288&&[2,8,14].contains(&layer))
            ||((publisher_screen||publisher_requested)&&columns==1048576&&layer==20);
        let branch_requested=if layer==20{publisher_requested}else{key_sharded_requested};
        let sixteen_requested=match std::env::var("CANADA_SCRATCH_KEY_SHARDED16") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid sixteen-row key partition policy");value=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(error)=>return Err(error.into()),
        };
        let key_sharded=if (rows==6||(rows==16&&sixteen_requested))&&key_rows%columns==0&&shard_geometry
            &&(branch_requested||std::env::var("CANADA_SCRATCH_KEY_SHARDED_INDEX_SCREEN").as_deref()==Ok("1")) {
            Some(crate::key_sharded_index::KeySharded::new(cuda,core,native.clone(),topk_native.clone(),query.output(),columns)?)
        }else{None};
        let key_sharded_auto=Cell::new(key_sharded.is_some()&&branch_requested
            &&std::env::var("CANADA_SCRATCH_KEY_SHARDED_INDEX_SCREEN").as_deref()!=Ok("1"));
        let key_sharded_active=Cell::new(key_sharded.is_some()&&branch_requested&&!key_sharded_auto.get());
        // The smaller publisher crossover is retained only as a diagnostic:
        // its component gain did not establish a full-model gain at393K.
        let key_sharded_min_position=if layer==20&&publisher_screen
            &&std::env::var("CANADA_SCRATCH_KEY_SHARD_MIDPOINT_SCREEN").as_deref()==Ok("1") {393216}else{524288};
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
            Scorer::Grouped(g) if [16,32].contains(&rows)=>g.configure_packed_gather(kernels)?,
            _=>{},
        }
        let key_splits=schedule.key_splits(match &scorer {
            Scorer::Chunked(c)=>c.rows(0)?.len(),
            Scorer::Grouped(_)=>rows.min(32),
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
        let selection = Selection::new_chunk_scratch(
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
        // Dense scorers (source layers, and the consumers' dense mode): full-width keys, no row sharding.
        let index16 = (crate::config::index_bf16()?
            && matches!(scorer, Scorer::Dense(_) | Scorer::Adaptive6{..}) && key_capacity >= columns
            && !shard_index_rows)
            .then(|| crate::index_bf16::IndexBf16::new(cuda, kernels, rows, columns))
            .transpose()?;
        ensure!(!serving_grouped16 || !crate::mhc::attention16_overlap_enabled()?
            || crate::mhc::compact16_overlap_enabled()?,
            "compact M16 requires serial attention or the pipeline-fenced overlap opt-in");
        let compact_requested=(rows==16&&!selected_grouped16_storage&&(serving_grouped16
            ||std::env::var("CANADA_SCRATCH_GROUPED_SPARSE16_SCREEN").as_deref()==Ok("1")))
            ||(rows==32&&compact32_screen);
        let compact_verifier=if compact_requested && columns==1048576 && consumer.is_some() {
            ensure!(((rows==16&&matches!(scorer,Scorer::Dense(_)))
                ||(rows==32&&matches!(scorer,Scorer::Chunked(_))))
                && !shard_index_rows && key_rows%columns==0 && key_splits==12,
                "compact verifier comparison requires the native16-row12-way dense schedule");
            let mut grouped=crate::grouped_sparse_index::Grouped::new(cuda,core,schedule,
                topk_native,query.output(),Rc::new(KeyMatrix::new(cuda,
                    if rows==16 {crate::grouped_sparse_index::key_capacity16(columns)}
                    else {crate::grouped_sparse_index::key_capacity(columns)})?),columns,SCORE_BUDGET)?;
            grouped.configure_packed_gather(kernels)?;
            Some(grouped)
        }else{None};
        let compact_verifier_active=Cell::new(((rows==16&&serving_grouped16)||(rows==32&&serving_grouped32))&&compact_verifier.is_some());
        let mut branch=Self {
            q,
            heads,
            query,
            gather: KeyGather::new(cuda, core, kernels, rows)?,
            keys,
            scorer,
            key_sharded,key_sharded_active,key_sharded_auto,key_sharded_min_position,
            compact_verifier,compact_verifier_active,region_mode:Cell::new(false),
            short_rows:Rc::new(Cell::new(false)),
            selection,
            publisher,
            consumer,
            sort_selected: (std::env::var("CANADA_SCRATCH_SORT_SELECTED").as_deref()==Ok("1")).then_some(module),
            sort_shuffle: selected_sort_shuffle_enabled()?,
            bounded_gather: std::env::var("CANADA_SCRATCH_BOUNDED_INDEX_GATHER").as_deref()==Ok("1"),
            shard_index_rows,
            key_splits,
            index16,
        };
        if compact_sparse_index_enabled(columns,adaptive6)? {branch.enable_compact_sparse6()?;}
        Ok(branch)
    }
    /// The owning target's switch for config::index_short_rows (it sets it around a short step's enqueue).
    pub(crate) fn short_rows_flag(&self) -> Rc<Cell<bool>> { self.short_rows.clone() }
    /// config::index_short_rows: this enqueue is a step whose rows all have at most 512 keys, on the dense
    /// six-row profile. The selector then writes every key of a row in order without reading a score
    /// (topk-v2's rule for a row no longer than k), so nothing that only feeds the scores is enqueued.
    fn short_step(&self, meta: &Metadata) -> bool {
        self.short_rows.get() && meta.rows_within(512) && meta.zero_based && !self.region_mode.get()
            && !self.key_sharded_active.get() && !self.compact_verifier_active.get() && !self.shard_index_rows
            && matches!(self.scorer, Scorer::Dense(_) | Scorer::Adaptive6{active:false,..})
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
        if !self.short_step(meta) { self.q.run(q_lora, stream)?; }
        self.finish(hidden, cache, meta, mask, None, None, stream)
    }
    pub(crate) fn run_distributed(&mut self, hidden: &Buffer<bf16>, q_lora: &Buffer<bf16>,
        cache: &KeyCache, meta: &Metadata, live: &Buffer<i32>,
        comm: &crate::roce::Roce, stream: &Stream) -> Result<()> {
        if !self.short_step(meta) { self.q.run(q_lora, stream)?; }
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
        if !self.short_step(meta) { self.q.run_quantized(q, scales, stream)?; }
        self.finish(hidden, cache, meta, mask, None, None, stream)
    }
    pub(crate) fn run_quantized_distributed(&mut self, hidden: &Buffer<bf16>, q: &Buffer<u8>,
        scales: &Buffer<u8>, cache: &KeyCache, meta: &Metadata, live: &Buffer<i32>,
        comm: &crate::roce::Roce, stream: &Stream) -> Result<()> {
        if !self.short_step(meta) { self.q.run_quantized(q, scales, stream)?; }
        self.finish(hidden, cache, meta, None, Some(comm), Some(live), stream)
    }
    /// Rerun the retained candidate publication of the latest row-sharded run for another
    /// frontier of the same chunk: the 128 rows before row `end` (`live` holds `end`).
    /// Only the score chunks that hold those rows are recomputed, from the same queries
    /// and gathered keys; the per-row selection of the run is not touched.
    pub(crate) fn republish_retained_window(&mut self, meta: &Metadata, comm: &crate::roce::Roce,
        live: &Buffer<i32>, end: usize, stream: &Stream) -> Result<()> {
        ensure!(self.shard_index_rows && self.consumer.is_none() && self.publisher.is_some()
            && comm.rank() < 4 && end >= 128 && end <= meta.rows && live.len() == 1
            && !self.key_sharded_active.get() && !self.compact_verifier_active.get(),
            "retained window republication admission");
        let quarter = meta.rows / 4;
        let owned = comm.rank() * quarter..(comm.rank() + 1) * quarter;
        let window = end - 128..end;
        self.publisher.as_mut().unwrap().prepare_retained_windows(&meta.windows, Some(live), Some(comm.rank()), stream)?;
        let Scorer::Chunked(chunked) = &mut self.scorer else {
            anyhow::bail!("retained window republication needs the chunked scorer");
        };
        let stride = chunked.stride();
        for index in 0..chunked.len() {
            let rows = chunked.rows(index)?;
            if rows.start < owned.start || rows.end > owned.end
                || rows.end <= window.start || rows.start >= window.end { continue; }
            chunked.run_chunk(index, &meta.ranges, stream)?;
            self.publisher.as_ref().unwrap().score_rows(chunked.output_mut(), stride, &meta.windows, rows, stream)?;
        }
        self.publisher.as_mut().unwrap().publish_sharded(&meta.windows, owned, comm, Some(live), stream)
    }
    /// Candidate blocks this source published in its latest run, if any.
    pub(crate) fn candidate_mask(&self) -> Option<&Buffer<u8>> {
        self.publisher.as_ref().map(|p| p.mask())
    }
    /// Publisher-owned block scores survive later consumer index operations;
    /// the shared full-token score workspace does not.
    pub(crate) fn published_block_taps(&self)->Vec<(&'static str,u64,usize)> {
        self.publisher.as_ref().map_or_else(Vec::new,|p|p.selector_taps())
    }
    pub(crate) fn candidate_mask_is_compact(&self) -> bool {
        self.publisher.as_ref().is_some_and(|p| p.compact_mask())
    }
    /// One active index scorer at a time on the target's ordered stream.
    /// Sparse6 scatter output stays separate: it can read its compact scores
    /// while writing the expanded matrix. Masks/keys/selected IDs also remain
    /// private and retain their original lifetime.
    pub(crate) fn share_score_scratch(&mut self, storage: Rc<Buffer<f32>>) -> Result<usize> {
        let mut released=match &mut self.scorer {
            Scorer::Dense(s)=>s.share_score_scratch(storage.clone())?,
            Scorer::Chunked(s)=>s.share_score_scratch(storage.clone())?,
            Scorer::Grouped(s)=>s.share_score_scratch(storage.clone())?,
            Scorer::Sparse6(s)=>s.share_score_scratch(storage.clone())?,
            Scorer::Adaptive6{dense,sparse,..}=>dense.share_score_scratch(storage.clone())?
                +sparse.share_score_scratch(storage.clone())?,
        };
        if let Some(s)=&mut self.compact_verifier {released+=s.share_score_scratch(storage)?;}
        Ok(released)
    }
    /// config::sparse_union_share: this consumer's sparse scorer reuses `owner`'s candidate union and gathered
    /// keys (same source-20 publication, lengths and key cache in every pass). Before any execution.
    pub(crate) fn share_sparse_union_from(&mut self, owner: &Self, follow: bool) -> Result<()> {
        match (&mut self.scorer, &owner.scorer) {
            (Scorer::Adaptive6{sparse: mine, ..}, Scorer::Adaptive6{sparse: theirs, ..})
            | (Scorer::Sparse6(mine), Scorer::Sparse6(theirs)) => mine.share_union_from(theirs, follow),
            _ => anyhow::bail!("shared candidate union needs two six-row sparse consumers"),
        }
    }
    pub(crate) fn follows_sparse_union(&self) -> bool {
        matches!(&self.scorer, Scorer::Sparse6(s) | Scorer::Adaptive6{sparse: s, ..} if s.follows_union())
    }
    pub(crate) fn sparse_union_linked(&self) -> bool {
        matches!(&self.scorer, Scorer::Sparse6(s) | Scorer::Adaptive6{sparse: s, ..} if s.union_linked())
    }
    pub(crate) fn set_sparse_union_follow_for_probe(&self, on: bool) -> Result<()> {
        match &self.scorer {
            Scorer::Sparse6(s) | Scorer::Adaptive6{sparse: s, ..} => s.set_union_follow_for_probe(on),
            _ => anyhow::bail!("candidate union follower switch on a non-sparse consumer"),
        }
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
        let short = self.short_step(meta);
        if !short {
        self.heads.run(hidden, stream)?;
        self.query
            .run(self.q.output(), self.heads.output(), &meta.query, stream)?;
        }
        let region=self.region_mode.get();
        let compact_verifier_active=self.compact_verifier_active.get()&&!region;
        let key_sharded_active=self.key_sharded_active.get()&&!region;
        let sparse6_active=self.sparse6_active()&&!region;
        if !key_sharded_active && !compact_verifier_active && !sparse6_active && !matches!(self.scorer, Scorer::Grouped(_)) {
            if region && !meta.zero_based {
                // Rows start at their regions' keys: copy only those ranges, not everything below them;
                // with the region gather cache, only from where this owner last gathered each region.
                let cached = crate::config::region_gather_cache();
                let starts = if cached { &meta.gather_starts } else { meta.ranges.key_starts() };
                self.gather.run_ranges(cache, &meta.slots, &self.keys, starts, meta.ranges.key_ends(), stream)?;
                if cached && crate::config::region_gather_check() {
                    let mut by_start: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
                    for &(s, e) in &meta.host_ranges { if e > s { let x = by_start.entry(s).or_insert(e); *x = (*x).max(e); } }
                    let ranges: Vec<(usize, usize)> = by_start.into_iter().collect();
                    self.gather.check_ranges(cache, &meta.slots, &self.keys, meta.ranges.key_starts(),
                        meta.ranges.key_ends(), &ranges, stream)?;
                }
            } else {
                self.gather.run_with_bound(cache, &meta.slots, &self.keys,
                    self.bounded_gather.then_some(meta.ranges.key_ends()), stream)?;
            }
        }
        // Block maxima read the raw scores and consumers mask them, both
        // before top-k consumes the row slice.
        let mut bf16_selected = false;
        let level_one = |scores: &mut Buffer<f32>, stride: usize, rows: std::ops::Range<usize>| -> Result<()> {
            if let Some(p) = &self.publisher {
                p.score_rows(scores, stride, &meta.windows, rows.clone(), stream)?;
            }
            if let (Some(c), Some(m)) = (&self.consumer, mask) {
                c.apply(scores, stride, &meta.windows, m, rows, stream)?;
            }
            Ok(())
        };
        if key_sharded_active {
            ensure!(meta.zero_based&&self.consumer.is_none()
                &&self.publisher.is_some()==self.has_key_sharded_publisher(),"key-shard zero-based source index");
            self.key_sharded.as_mut().expect("prepared key sharding").run(cache,&meta.slots,meta.ranges.key_ends(),
                &meta.windows,&self.selection,comm.ok_or_else(||anyhow::anyhow!("key-shard collective owner missing"))?,stream)?;
        }else if compact_verifier_active {
            ensure!(meta.zero_based,"compact verifier requires zero-based ranges");
            self.compact_verifier.as_mut().expect("prepared compact verifier scorer").run(cache,&meta.slots,&meta.windows,
                mask.ok_or_else(||anyhow::anyhow!("compact verifier source mask missing"))?,&self.selection,stream)?;
        }else{match &mut self.scorer {
            Scorer::Dense(dense) | Scorer::Adaptive6{dense,active:false,..} if short => {
                // Every row is no longer than k: the selector emits each row's keys in order and reads no score.
                self.selection
                    .run(dense.output_mut(), &meta.windows, stream)?;
            }
            Scorer::Dense(dense) | Scorer::Adaptive6{dense,active:false,..} if self.index16.is_some() => {
                let b = self.index16.as_ref().expect("checked bf16 index");
                ensure!(meta.windows.width() <= dense.stride(), "bf16 index window width");
                if self.publisher.is_none() && self.consumer.is_none() {
                    // Scores as 16-bit keys in the f32 score arena's bytes (half of each row), then the selection,
                    // already ascending: no topk-v2, no sort.
                    unsafe {
                        b.run(&self.query.output(), &self.keys, meta.ranges.key_starts().ptr(),
                            meta.windows.lengths().ptr(), meta.windows.offsets_ptr(), dense.output().ptr(),
                            dense.stride() * 2, self.selection.indices(), self.selection.topk(), stream)?;
                    }
                    self.selection.note_written();
                    bf16_selected = true;
                } else {
                    // The publisher / consumer read f32 scores: the same values as f32, then today's path.
                    let stride = dense.stride();
                    unsafe {
                        b.run_scores_f32(&self.query.output(), &self.keys, meta.ranges.key_starts().ptr(),
                            meta.windows.lengths().ptr(), dense.output(), stride, stream)?;
                    }
                    level_one(dense.output_mut(), stride, 0..meta.windows.rows())?;
                    self.selection.run(dense.output_mut(), &meta.windows, stream)?;
                }
            }
            Scorer::Dense(dense) | Scorer::Adaptive6{dense,active:false,..} => {
                dense.run(&meta.ranges, stream)?;
                let stride = dense.stride();
                level_one(dense.output_mut(), stride, 0..meta.windows.rows())?;
                self.selection
                    .run(dense.output_mut(), &meta.windows, stream)?;
            }
            Scorer::Adaptive6{dense,..} if region => {
                let stride = dense.stride();
                if let Some(b) = &self.index16 {
                    ensure!(meta.windows.width() <= stride, "bf16 index window width");
                    unsafe {
                        b.run_scores_f32(&self.query.output(), &self.keys, meta.ranges.key_starts().ptr(),
                            meta.windows.lengths().ptr(), dense.output(), stride, stream)?;
                    }
                } else { dense.run(&meta.ranges, stream)?; }
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
        let already_sorted=(sparse6_active && self.fused_compact_publication()) || bf16_selected;
        if let Some(module) = self.sort_selected.as_ref().filter(|_|!already_sorted) {
            let mut indices = self.selection.indices().ptr();
            // Only the lengths passed to the native selector establish its
            // <=512 ordered-output guarantee. Compact unions can exceed512
            // while a member causal row is shorter. Skipping its sort leaves
            // native selection order variable. The short-row guarantee must
            // cover the actual producer before we rely on ordered IDs/padding.
            let mut lengths = if key_sharded_active {
                self.key_sharded.as_ref().unwrap().selection_lengths()
            }else if compact_verifier_active {
                self.compact_verifier.as_ref().unwrap().selection_lengths()
            }else{match &self.scorer {
                Scorer::Grouped(grouped)=>grouped.selection_lengths(),
                Scorer::Sparse6(s)|Scorer::Adaptive6{sparse:s,active:true,..} if !region=>
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
        if let Scorer::Sparse6(sparse) | Scorer::Adaptive6{sparse,active:true,..} = &self.scorer && sparse6_active {
            // Sorting normalizes negative padding to -1; poison afterwards so
            // the downstream mapper sees the deliberately invalid -2 marker.
            sparse.poison_selection_on_error(self.selection.indices(), stream)?;
        }
        if let Scorer::Grouped(grouped) = &self.scorer {
            grouped.poison_selection_on_error(self.selection.indices(), stream)?;
        }
        if compact_verifier_active {
            self.compact_verifier.as_ref().unwrap().poison_selection_on_error(self.selection.indices(),stream)?;
        }
        // A short step's consumers select every key as well: nothing reads this step's publication.
        if short { return Ok(()); }
        if let Some(p) = &mut self.publisher {
            if key_sharded_active {
                p.publish_selected(self.key_sharded.as_ref().and_then(|s|s.block_ids())
                    .ok_or_else(||anyhow::anyhow!("key partition missing candidate blocks"))?,&meta.windows,stream)?;
            }else if let Some(rows)=owned_rows {
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
        ensure!(!self.key_sharded_active.get(),"key-shard path does not materialize the replicated diagnostic scores");
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
