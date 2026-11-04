//! Independent TP4 decoder graph: predecessor-aware mHC, canonical SWA,
//! real routed/shared FFN, and two BF16 native RoCE reductions. Construction
//! is a model graph milestone; only a complete distributed trace qualifies it.
use crate::{
    attention::{Candidates, PagedKv},
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, Module, Stream},
    engram_gpu::{Layer as Engram, Rows as EngramRows},
    expert_packing::Prepared,
    ffn::{LayerFfn, TokenRows},
    kv::RowLocations,
    layer_attention::{InputMode, Layer as Attention},
    mhc::Boundary,
    moe::Workspace,
    prelude::{FrequencyTable, Positions},
    roce::Roce,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{cell::RefCell, path::Path, rc::Rc};

/// Construction context of the wide encoder profile. Layers alternate between
/// two row arenas by layer parity (layer L's output stays valid while layer
/// L+1 runs) and share one routed-expert workspace, since the 4096-row MoE
/// scratch is too large to replicate per layer. Execution is sequential on
/// one ordered stream; FFN/mHC side-stream overlaps are not used at this width.
pub(crate) struct WideBuild {
    rows: usize,
    // Some(rank): sequence-parallel row-local mHC on this rank's M/4 rows.
    sp_rank: Option<usize>,
    arenas: [Rc<crate::cuda::RowArena>; 2],
    shared_arenas: bool,
    shared_moe: bool,
    moe: Rc<Workspace>,
    // Embedding and full-row gather storage survive individual layer scopes,
    // but not a completed prefill chunk. Keep them separate from layer parity
    // storage and lend them only to sequential wide profiles.
    input_arena: Option<Rc<crate::cuda::RowArena>>,
    gather_arena: Option<Rc<crate::cuda::RowArena>>,
}
/// Engram layers 1/14 and the embedding-fed layer0 keep full rows; every
/// other encoder layer may run its mHC boundaries sequence-parallel.
pub(crate) fn sequence_parallel_layer(layer: usize) -> bool {
    (2..=20).contains(&layer) && layer != 14
}
impl WideBuild {
    pub(crate) fn transient_arena_for_probe(&self,slot:usize)->Result<Rc<crate::cuda::RowArena>> {
        ensure!(std::env::var("CANADA_SCRATCH_NARROW_ARENA_LOAN_SCREEN").as_deref()==Ok("1")
            &&self.rows==8192&&self.sp_rank.is_some()&&slot<2,"finite prefill arena loan");
        Ok(self.arenas[slot].clone())
    }
    pub(crate) fn new(
        cuda: &Cuda,
        kernels: &Path,
        rows: usize,
        arena_bytes: usize,
        sp_rank: Option<usize>,
        share: Option<&Self>,
    ) -> Result<Self> {
        ensure!(
            crate::config::wide_rows(rows) && sp_rank.is_none_or(|r| r < 4),
            "wide build width/rank"
        );
        let arenas = if let Some(prior) = share {
            ensure!(prior.rows >= rows && prior.sp_rank == sp_rank
                && prior.arenas.iter().all(|a| a.bytes() >= arena_bytes),
                "shared wide arenas require the same layout and sufficient capacity");
            // These are explicitly transient layer-parity arenas. Model state
            // and padded-K operands remain separately owned. Same-width peer
            // MoE scratch has its own finite admission below.
            // The caller admits only sequential profiles on one ordered
            // stream; independent request owners may lend these transients
            // only through the separately gated peer-bank construction.
            prior.arenas.clone()
        } else {
            [cuda.row_arena(arena_bytes)?, cuda.row_arena(arena_bytes)?]
        };
        let peer_moe=std::env::var("CANADA_SCRATCH_PEER_WIDE_MOE_SHARE_SCREEN").as_deref()==Ok("1")
            &&share.is_some_and(|p|p.rows==rows);
        let moe=if peer_moe {
            ensure!(crate::config::shared_owner_profiles_requested(),
                "finite peer MoE workspace admission");
            share.unwrap().moe.clone()
        }else{
            let root=kernels.join(format!("moe-runtime-m{rows}-v0"));
            let mut moe=if std::env::var("CANADA_SCRATCH_WIDE_MOE_ARENA_SCREEN").as_deref()==Ok("1") {
                ensure!(crate::config::shared_owner_profiles_requested()&&sp_rank.is_some(),
                    "wide MoE arena loan requires sequential owner profiles");
                Workspace::new_sequential(cuda,&root,share.map(|s|s.moe.as_ref()))?
            }else{Workspace::new(cuda,&root)?};
            if [8192,16384].contains(&rows) && sp_rank.is_some() && crate::moe::packed_gather_requested()? {
                moe.enable_packed(&kernels.join(format!("moe-runtime-m{rows}-prequant-v1")))?;
            }
            Rc::new(moe)
        };
        let io_pool=std::env::var("CANADA_SCRATCH_WIDE_IO_ARENAS_SCREEN").as_deref()==Ok("1");
        ensure!(!io_pool || (sp_rank.is_some()
            &&crate::config::shared_owner_profiles_requested()),
            "wide IO arena loan requires the finite sequential full-profile screen");
        let (input_arena,gather_arena)=if io_pool {
            if let Some(prior)=share {
                (Some(prior.input_arena.as_ref().ok_or_else(||anyhow::anyhow!("missing wide input arena donor"))?.clone()),
                 Some(prior.gather_arena.as_ref().ok_or_else(||anyhow::anyhow!("missing wide gather arena donor"))?.clone()))
            }else{
                (Some(cuda.row_arena(rows*30720*2)?),Some(cuda.row_arena(rows*(20480*2+4*4))?))
            }
        }else{(None,None)};
        Ok(Self {
            rows,
            sp_rank,
            arenas,
            shared_arenas: share.is_some(),
            shared_moe: peer_moe,
            moe,
            input_arena,gather_arena,
        })
    }
    pub(crate) fn input_arena(&self)->Option<&Rc<crate::cuda::RowArena>> {self.input_arena.as_ref()}
    pub(crate) fn gather_arena(&self)->Option<&Rc<crate::cuda::RowArena>> {self.gather_arena.as_ref()}
    pub(crate) fn prepare_moe_scratch(&self,stream:&Stream)->Result<()> {
        self.moe.prepare_sequential_scratch(stream)
    }
    /// Build one layer's transient storage in its parity arena.
    pub(crate) fn layer<R>(
        &self,
        cuda: &Cuda,
        layer: usize,
        build: impl FnOnce(Rc<Workspace>, Option<usize>) -> Result<R>,
    ) -> Result<R> {
        let moe = self.moe.clone();
        let sp = self.sp_rank.filter(|_| sequence_parallel_layer(layer));
        cuda.with_row_arena(&self.arenas[layer % 2], || build(moe, sp))
    }
    pub(crate) fn report(&self) -> serde_json::Value {
        serde_json::json!({"rows":self.rows,"sequence_parallel_mhc":self.sp_rank.is_some(),"arena_bytes":self.arenas[0].bytes(),
            "shared_row_arenas":self.shared_arenas,"shared_moe_workspace":self.shared_moe,
            "sequential_moe_backing_bytes":self.moe.sequential_scratch_bytes(),
            "owned_arena_bytes":if self.shared_arenas {0}else{self.arenas[0].bytes()+self.arenas[1].bytes()},
            "arena_high_water":[self.arenas[0].high_water(),self.arenas[1].high_water()],
            "shared_wide_IO":self.input_arena.is_some(),
            "wide_IO_backing_bytes":self.input_arena.as_ref().map_or(0,|a|a.bytes())
                +self.gather_arena.as_ref().map_or(0,|a|a.bytes())})
    }
}
/// Sequence-parallel sublayer I/O: this rank owns rows [rank*M/4, (rank+1)*M/4)
/// of the residual/mHC state; TP sublayers see all M gathered input rows.
struct Sp {
    rank: usize,
    gathered: Buffer<bf16>,
    quantized: Option<crate::fp8_linear::RowShard>,
    use_quantized: bool,
    local_qkv_prepared: bool,
    use_local_qkv: bool,
}
pub struct SwaLayer {
    sp: Option<Sp>,
    weights: Rc<Weights>,
    prepared: Prepared,
    layer: usize,
    engram: Option<Engram>,
    attention: Attention,
    ffn: LayerFfn,
    moe_workspace: Rc<Workspace>,
    attn_boundary: Boundary,
    ffn_boundary: Boundary,
    attn_reduced: Buffer<bf16>,
    ffn_reduced: Buffer<bf16>,
    sink: Buffer<f32>,
    next_input_prefetch: Option<crate::l2_prefetch::Window>,
    pair_deferred_ffn: bool,
    pair_attention: PairAttention,
}
/// Enqueue/capture-only state. A pooled projection must consume exactly the
/// private normalization prepared from this predecessor, on the same stream.
#[derive(Clone,Copy,PartialEq,Eq)]
struct PairAttentionInput { residual:u64, pre:u64, stream:usize }
#[derive(Clone,Copy,PartialEq,Eq)]
enum PairAttention { Ordinary, Defer, Prepared(PairAttentionInput), Projected(PairAttentionInput) }
impl SwaLayer {
    pub(crate) fn enable_verify16_ffn_before_capture(&mut self,stream:&Stream)->Result<()> {
        self.ffn.enable_verify16_overlap_before_capture(stream)?;
        self.ffn_boundary.enable_verify16_overlap_before_capture(stream)
    }
    pub(crate) fn ffn_overlap_flags(&self)->(bool,bool) {
        (self.ffn.schedule_flags().0,self.ffn_boundary.overlap_active())
    }
    pub(crate) fn set_verify16_schedule_for_probe(&mut self, mask: u8, stream: &Stream) -> Result<()> {
        ensure!(mask<=3,"M16 FFN overlap component mask");
        if std::env::var("CANADA_SCRATCH_VERIFY16_ATTN_ONLY").as_deref()==Ok("1") {
            self.ffn.set_verify16_overlap_for_probe(false,stream)?;
            self.ffn_boundary.set_verify16_overlap_for_probe(false,stream)?;
            self.attention.set_verify16_query_for_probe(mask&1!=0,stream)?;
            return self.attn_boundary.set_verify16_overlap_for_probe(mask&2!=0,stream);
        }
        self.ffn.set_verify16_overlap_for_probe(mask&1!=0,stream)?;
        self.ffn_boundary.set_verify16_overlap_for_probe(mask&2!=0,stream)
    }
    pub(crate) fn narrow_workspace(&self)->Rc<Workspace> {self.moe_workspace.clone()}
    /// Construction only, before graphs/hints. Routed work and its output
    /// consumer execute on one ordered stream. The FFN's shared branch joins
    /// before add_shared writes the layer-owned output; later layers therefore
    /// cannot overwrite an earlier residual or DSpark feature tap.
    pub(crate) fn share_narrow_workspace(&mut self,cuda:&Cuda,workspace:Rc<Workspace>,rows:usize)->Result<()> {
        let old=&*self.moe_workspace;
        ensure!([6,16,32,128].contains(&rows) && self.sp.is_none()
            && old.output().len()==rows*5120 && workspace.output().len()==rows*5120
            && cuda.owns_buffer(old.output()) && cuda.owns_buffer(workspace.output())
            && old.live_bounds()==workspace.live_bounds()
            && [&*self.moe_workspace,&*workspace].iter().all(|w|!w.micro_fp32()
                && !w.task_order() && !w.task_order_m3() && !w.scale_reuse())
            && (rows==6 || rows==16 || !workspace.live_bounds()),
            "shared narrow scratch requires matching ordinary profile contracts");
        self.moe_workspace=workspace;Ok(())
    }
    pub(crate) fn shared_weights(&self) -> Rc<Weights> {
        self.weights.clone()
    }
    pub(crate) fn prepared_layout(&self) -> Prepared {
        self.prepared.clone()
    }

    pub fn new(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: Rc<Weights>,
        prepared: Prepared,
        kernels: &Path,
        frequencies: Rc<FrequencyTable>,
    ) -> Result<Self> {
        Self::new_with_mode(
            cuda,
            module,
            stream,
            weights,
            prepared,
            kernels,
            frequencies,
            InputMode::Fused,
        )
    }
    pub fn new_with_mode(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: Rc<Weights>,
        prepared: Prepared,
        kernels: &Path,
        frequencies: Rc<FrequencyTable>,
        mode: InputMode,
    ) -> Result<Self> {
        Self::new_rows(
            cuda,
            module,
            stream,
            weights,
            prepared,
            kernels,
            frequencies,
            mode,
            6,
            None,
            None,
        )
    }
    /// M128 has an explicit prefill path. Qualification remains per immutable
    /// runtime, exact model prefix and phase; construction does not award parity.
    pub fn new_rows(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: Rc<Weights>,
        prepared: Prepared,
        kernels: &Path,
        frequencies: Rc<FrequencyTable>,
        mode: InputMode,
        rows: usize,
        vendor_blas: Option<Rc<crate::blas::Blas>>,
        medium_woa: Option<Rc<RefCell<crate::woa_medium::Scratch>>>,
    ) -> Result<Self> {
        Self::new_rows_in(
            cuda,
            module,
            stream,
            weights,
            prepared,
            kernels,
            frequencies,
            mode,
            rows,
            vendor_blas,
            medium_woa,
            None,
            None,
        )
    }
    pub(crate) fn narrow_workspace_sharing_enabled(rows:usize)->bool {
        match rows {
            32=>crate::config::history32_enabled(),
            16=>std::env::var("CANADA_SCRATCH_VERIFY16_SHARED_MOE").as_deref()==Ok("1"),
            6|128=>std::env::var("CANADA_SCRATCH_NARROW_SHARED_MOE").as_deref()==Ok("1"),
            _=>false,
        }
    }
    /// `shared_moe` belongs to this execution profile. Wide profiles require it;
    /// qualified narrow profiles may bind their scratch during construction,
    /// avoiding allocation of forty workspaces before the usual consolidation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_rows_in(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: Rc<Weights>,
        prepared: Prepared,
        kernels: &Path,
        frequencies: Rc<FrequencyTable>,
        mode: InputMode,
        rows: usize,
        vendor_blas: Option<Rc<crate::blas::Blas>>,
        medium_woa: Option<Rc<RefCell<crate::woa_medium::Scratch>>>,
        shared_moe: Option<Rc<Workspace>>,
        sp_rank: Option<usize>,
    ) -> Result<Self> {
        ensure!(
            crate::config::target_rows(rows),
            "decoder execution profile not implemented for this layer"
        );
        ensure!(
            if crate::config::wide_rows(rows) {shared_moe.is_some()} else {
                shared_moe.is_none() || (Self::narrow_workspace_sharing_enabled(rows)
                    && std::env::var("CANADA_SCRATCH_NARROW_MOE_CONSTRUCTION").as_deref()==Ok("1"))
            },
            "shared routed-expert workspace requires its admitted profile policy"
        );
        // Ratio2 groups through19 are implemented; later ratio1 sources need
        // a separate graph. Qualification is recorded per immutable runtime.
        ensure!(
            prepared.layer < 40,
            "only target layers0..39 are admitted by the assembled graph"
        );
        let shared_weights = weights.clone();
        let prepared_copy = prepared.clone();
        let layer = prepared.layer;
        let sink_name = format!("layers.{layer}.attn.attn_sink");
        let v = weights.view(&sink_name)?;
        ensure!(
            v.dtype == Dtype::F32 && v.shape == [16],
            "TP4 attention sink shape/dtype"
        );
        let sink = cuda.allocate::<f32>(16)?;
        unsafe { cuda.copy_device(sink.ptr(), weights.ptr(&sink_name)?, sink.bytes(), stream) }?;
        ensure!(
            sp_rank.is_none_or(|r| r < 4 && rows % 4 == 0 && sequence_parallel_layer(layer)),
            "sequence-parallel mHC layer/rank"
        );
        // Sequence-parallel boundaries hold only this rank's quarter of the rows.
        let local_rows = if sp_rank.is_some() { rows / 4 } else { rows };
        let attn_boundary =
            Boundary::new(cuda, stream, weights.clone(), layer, "attn", local_rows, kernels)?;
        let ffn_boundary =
            Boundary::new(cuda, stream, weights.clone(), layer, "ffn", local_rows, kernels)?;
        let attention = Attention::new_rows(
            cuda,
            module,
            stream,
            weights.clone(),
            layer,
            kernels,
            frequencies,
            mode,
            rows,
            medium_woa,
        )?;
        let engram = if layer == 1 {
            Some(Engram::new(
                cuda,
                module,
                stream,
                weights.clone(),
                layer,
                rows,
                kernels,
            )?)
        } else {
            None
        };
        let mut ffn = LayerFfn::new_rows(
            cuda,
            module,
            stream,
            weights,
            prepared,
            kernels,
            rows,
            vendor_blas.clone(),
        )?;
        if rows==16 && crate::ffn::shared_quant16_enabled()? {
            ffn.enable_shared_quant16(kernels)?;
        }
        if [8192,16384].contains(&rows) && sp_rank.is_some() && crate::moe::packed_gather_requested()? {
            ffn.enable_row_transport(kernels,vendor_blas)?;
            // s699 component and s700 whole-model pairs preserve every codec
            // byte, logit and retained-state check. Carry the small M8192 gain.
            if rows==8192 {
                let coalesced=match std::env::var("CANADA_SCRATCH_PREFILL_MOE_CODEC_COALESCED") {
                    Ok(value)=>{ensure!(value=="0"||value=="1","invalid coalesced codec policy");value=="1"},
                    Err(std::env::VarError::NotPresent)=>false,
                    Err(error)=>return Err(error.into()),
                };
                if coalesced {ffn.set_coalesced_codec(true,kernels)?;}
            }
        }
        let micro_moe = rows==6
            && std::env::var("CANADA_SCRATCH_MOE_MICRO_FP32").as_deref()==Ok("1");
        // s305 ABI and s308 four-rank model screens qualify live bounds and
        // the FP32 micro path at M1. Keep A/B controls on the old workspace.
        let live_default = match std::env::var("CANADA_SCRATCH_MOE_LIVE") {
            Ok(v) => { ensure!(v=="0" || v=="1", "invalid live MoE policy"); v=="1" }
            Err(std::env::VarError::NotPresent) => true,
            Err(e) => return Err(e.into()),
        };
        let live_candidate = std::env::var("CANADA_SCRATCH_MOE_LIVE_CANDIDATE").as_deref()==Ok("1");
        let live_ab = std::env::var("CANADA_SCRATCH_MOE_LIVE_AB").as_deref()==Ok("1");
        let micro_ab = std::env::var("CANADA_SCRATCH_MOE_MICRO_AB").as_deref()==Ok("1");
        ensure!(!live_candidate || live_ab, "live MoE comparison flag requires its A/B screen");
        let live_moe = rows==6 && (live_candidate || (live_default && !live_ab && !micro_ab));
        ensure!(!(micro_moe && live_moe),"exclusive MoE experiment modes");
        ensure!(!micro_moe || std::env::var("CANADA_SCRATCH_MOE_MICRO_AB").as_deref()==Ok("1"),
            "micro MoE requires the isolated model comparison");
        if !crate::config::wide_rows(rows) && let Some(workspace)=&shared_moe {
            let live=live_moe || (rows==16
                && std::env::var("CANADA_SCRATCH_VERIFY16_LIVE_MOE").as_deref()==Ok("1"));
            ensure!(sp_rank.is_none() && !micro_moe && workspace.output().len()==rows*5120
                && cuda.owns_buffer(workspace.output()) && workspace.live_bounds()==live
                && !workspace.micro_fp32() && !workspace.task_order()
                && !workspace.task_order_m3() && !workspace.scale_reuse(),
                "construction-time narrow scratch must preserve the ordinary workspace contract");
        }
        let moe_workspace = match shared_moe {
            Some(w) => w,
            None if live_moe => Rc::new(Workspace::new_live(cuda,kernels)?),
            None if rows==16 && std::env::var("CANADA_SCRATCH_VERIFY16_LIVE_MOE").as_deref()==Ok("1")
                => Rc::new(Workspace::new_live16(cuda,kernels)?),
            None => Rc::new(Workspace::new(
                cuda,
                &kernels.join(if micro_moe {"moe-runtime-m6-micro-fp32-v1".to_owned()}
                    else {format!("moe-runtime-m{rows}-v0")}),
            )?),
        };
        let quant_gather = match std::env::var("CANADA_SCRATCH_PREFILL_QUANT_GATHER") {
            Ok(v) => { ensure!(v == "0" || v == "1", "invalid quantized prefill gather policy"); v == "1" },
            Err(std::env::VarError::NotPresent) => false,
            Err(e) => return Err(e.into()),
        };
        let quant_ab = std::env::var("CANADA_SCRATCH_PREFILL_QUANT_GATHER_AB").as_deref() == Ok("1");
        let local_qkv = match std::env::var("CANADA_SCRATCH_PREFILL_LOCAL_QKV") {
            Ok(v) => { ensure!(v == "0" || v == "1", "invalid local QKV policy"); v == "1" },
            Err(std::env::VarError::NotPresent) => false,
            Err(e) => return Err(e.into()),
        };
        let local_qkv_ab = std::env::var("CANADA_SCRATCH_PREFILL_LOCAL_QKV_AB").as_deref() == Ok("1");
        ensure!(!(local_qkv || local_qkv_ab) || (quant_gather && !quant_ab),
            "local QKV requires the current quantized gather control");
        Ok(Self {
            sp: sp_rank
                .map(|rank| -> Result<Sp> {
                    // Source layers also feed BF16 hidden states to their
                    // compressor/indexer. Only shared-source attention can
                    // eliminate this gather without changing another input.
                    let eligible = (3..=19).contains(&layer) && ![8, 14].contains(&layer);
                    let local_qkv_prepared = eligible && [8192,16384].contains(&rows) && (local_qkv || local_qkv_ab);
                    let local_root = kernels.join(format!("fp8-canonical-wqkv_a-r0-m{}-v0",rows/4));
                    Ok(Sp {
                        rank,
                        gathered: cuda.allocate_rows(rows * 5120)?,
                        quantized: (eligible && (quant_gather || quant_ab))
                            .then(|| attention.prepare_row_shard(local_qkv_prepared.then_some(local_root.as_path()))).transpose()?,
                        use_quantized: eligible && quant_gather && !quant_ab,
                        local_qkv_prepared,
                        use_local_qkv: local_qkv_prepared && local_qkv && !local_qkv_ab,
                    })
                })
                .transpose()?,
            weights: shared_weights,
            prepared: prepared_copy,
            layer,
            engram,
            attention,
            ffn,
            moe_workspace,
            attn_boundary,
            ffn_boundary,
            sink,
            next_input_prefetch: None,
            pair_deferred_ffn: false,
            pair_attention: PairAttention::Ordinary,
            attn_reduced: cuda.allocate_rows(local_rows * 5120)?,
            ffn_reduced: cuda.allocate_rows(local_rows * 5120)?,
        })
    }
    pub(crate) fn input_matrix(&self) -> Rc<crate::blockscaled::Matrix> {
        self.attention.input_matrix()
    }
    pub(crate) fn input_hint(&self)->bool { self.next_input_prefetch.is_some() }
    pub(crate) fn large_input_hint(&self)->bool {
        self.next_input_prefetch.as_ref().is_some_and(|h|h.chunk()==1048576)
    }
    pub(crate) fn install_next_input_prefetch(
        &mut self,
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        matrix: Rc<crate::blockscaled::Matrix>,
    ) -> Result<()> {
        ensure!(
            self.next_input_prefetch.is_none(),
            "next input hint already installed"
        );
        let chunk=match std::env::var("CANADA_SCRATCH_INPUT_HINT_CHUNK") {
            Ok(value)=>{ensure!(value=="0"||value=="16384"||value=="1048576","input hint geometry");value.parse()?},
            Err(std::env::VarError::NotPresent)=>16384,
            Err(error)=>return Err(error.into()),
        };
        if chunk==0 { return Ok(()); }
        self.next_input_prefetch = Some(crate::l2_prefetch::Window::new(
            cuda,
            module,
            stream,
            vec![matrix],
            None,
        )?.with_chunk(chunk)?);
        Ok(())
    }
    pub fn run(
        &mut self,
        residual: &Buffer<bf16>,
        positions: &Positions,
        locations: &RowLocations,
        cache: &mut PagedKv,
        candidates: &Candidates,
        tokens: &TokenRows,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        self.run_from_previous(
            residual, None, None, positions, locations, cache, candidates, tokens, comm, stream,
        )
    }
    pub fn run_from_previous(
        &mut self,
        residual: &Buffer<bf16>,
        previous_pre: Option<&Buffer<f32>>,
        engram_rows: Option<&mut EngramRows>,
        positions: &Positions,
        locations: &RowLocations,
        cache: &mut PagedKv,
        candidates: &Candidates,
        tokens: &TokenRows,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            (self.layer == 0 && previous_pre.is_none() && engram_rows.is_none())
                || (self.layer == 1 && previous_pre.is_some() && engram_rows.is_some()),
            "SWA layer predecessor/Engram state mismatch"
        );
        let residual = if let Some(engram) = self.engram.as_mut() {
            engram.run(residual, engram_rows.unwrap(), tokens, comm, stream)?;
            engram.output()
        } else {
            residual
        };
        self.attn_boundary.prepare(residual, previous_pre, stream)?;
        Self::gather_input(&mut self.sp, &self.attn_boundary, comm, stream)?;
        self.attention.run_distributed_quantized(
            match &self.sp { Some(sp) => &sp.gathered, None => self.attn_boundary.input() },
            self.attn_boundary.quantized_input(),
            positions,
            locations,
            cache,
            candidates,
            &self.sink,
            Some(comm),
            stream,
        )?;
        self.ffn.prefetch_begin(stream)?;
        unsafe { Self::reduce(self.sp.is_some(), comm, self.attention.output(), &mut self.attn_reduced, stream) }?;
        self.attn_boundary
            .finish(&self.attn_reduced, residual, stream)?;
        self.ffn_boundary.prepare(
            self.attn_boundary.output(),
            Some(self.attn_boundary.pre()),
            stream,
        )?;
        if self.pair_deferred_ffn {return Ok(());}
        self.run_ffn(tokens, comm, stream)?;
        self.finish_ffn(comm,stream)
    }
    pub fn output(&self) -> &Buffer<bf16> {
        self.ffn_boundary.output()
    }
    /// Some(rank) when this layer's residual/pre are this rank's row shard.
    pub(crate) fn sp_rank(&self) -> Option<usize> {
        self.sp.as_ref().map(|sp| sp.rank)
    }
    /// This layer's view of a predecessor's residual/pre: an SP layer fed by a
    /// full-row layer takes its own rank's rows (no copy or communication).
    pub(crate) fn shard_input(
        &self,
        residual: &Buffer<bf16>,
        pre: &Buffer<f32>,
    ) -> Result<Option<(Buffer<bf16>, Buffer<f32>)>> {
        let Some(sp) = &self.sp else { return Ok(None) };
        let rows = pre.len() / 4;
        if residual.len() == self.attn_boundary_rows() * 20480 {
            return Ok(None);
        }
        ensure!(
            residual.len() == rows * 20480 && rows == 4 * self.attn_boundary_rows(),
            "sequence-parallel input shard geometry"
        );
        let local = rows / 4;
        // Safety: the views alias the predecessor's live output for this call.
        unsafe {
            Ok(Some((
                residual.view(sp.rank * local * 20480, local * 20480)?,
                pre.view(sp.rank * local * 4, local * 4)?,
            )))
        }
    }
    fn attn_boundary_rows(&self) -> usize {
        self.attn_reduced.len() / 5120
    }
    /// SP: gather every rank's normalized sublayer rows before a TP sublayer.
    fn gather_input(
        sp: &mut Option<Sp>,
        boundary: &Boundary,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        if let Some(sp) = sp.as_mut() {
            unsafe { comm.gather_rows(boundary.input(), &mut sp.gathered, stream) }?;
        }
        Ok(())
    }
    /// Sum a TP partial: all rows (all-reduce), or this rank's rows (SP).
    unsafe fn reduce(
        sp: bool,
        comm: &Roce,
        partial: &Buffer<bf16>,
        reduced: &mut Buffer<bf16>,
        stream: &Stream,
    ) -> Result<()> {
        if sp {
            unsafe { comm.scatter_sum_bf16(partial, reduced, stream) }
        } else {
            unsafe { comm.sum_bf16(partial, reduced, stream) }
        }
    }
    pub(crate) fn defer_pair_attention(&mut self)->Result<()> {
        ensure!(self.pair_deferred_ffn&&self.sp.is_none()&&self.engram.is_none()
            &&(2..40).contains(&self.layer)&&self.layer!=14
            &&self.pair_attention==PairAttention::Ordinary,
            "paired QKV stage requires an idle admitted non-Engram layer");
        self.attention.paired_input_projection()?;
        self.pair_attention=PairAttention::Defer;Ok(())
    }
    pub(crate) fn clear_pair_attention(&mut self) {self.pair_attention=PairAttention::Ordinary;}
    /// None ends the first stage; Some(true) resumes after pooled projection.
    /// The ordinary path enqueues exactly its existing boundary preparation.
    fn prepare_attention_input(&mut self,residual:&Buffer<bf16>,pre:&Buffer<f32>,stream:&Stream)
        ->Result<Option<bool>> {
        let identity=PairAttentionInput{residual:residual.ptr(),pre:pre.ptr(),stream:stream.raw()as usize};
        match self.pair_attention {
            PairAttention::Ordinary=>{
                self.attn_boundary.prepare(residual,Some(pre),stream)?;Ok(Some(false))
            },
            PairAttention::Defer=>{
                self.attn_boundary.prepare(residual,Some(pre),stream)?;
                self.pair_attention=PairAttention::Prepared(identity);Ok(None)
            },
            PairAttention::Projected(expected)=>{
                ensure!(expected==identity,"paired attention predecessor or stream changed");
                self.pair_attention=PairAttention::Ordinary;Ok(Some(true))
            },
            PairAttention::Prepared(_)=>anyhow::bail!("paired attention resumed before projection"),
        }
    }
    pub(crate) fn project_pair_attention(&mut self,other:&mut Self,pool:&crate::fp8_linear::PairedQkv,
        live:usize,other_live:usize,stream:&Stream)->Result<()> {
        let(PairAttention::Prepared(a),PairAttention::Prepared(b))=(self.pair_attention,other.pair_attention)
            else{anyhow::bail!("both paired normalizations must be ready");};
        ensure!(self.layer==other.layer&&a.stream==b.stream&&a.stream==stream.raw()as usize
            &&self.attn_boundary.input().ptr()!=other.attn_boundary.input().ptr(),
            "paired attention private input/layer/stream ownership");
        pool.run(self.attention.paired_input_projection()?,self.attn_boundary.input(),
            self.attn_boundary.quantized_input(),live,other.attention.paired_input_projection()?,
            other.attn_boundary.input(),other.attn_boundary.quantized_input(),other_live,stream)?;
        self.pair_attention=PairAttention::Projected(a);other.pair_attention=PairAttention::Projected(b);Ok(())
    }
    fn project_attention_input(&mut self,precomputed:bool,positions:&Positions,locations:&RowLocations,
        cache:&mut PagedKv,comm:&Roce,stream:&Stream)->Result<()> {
        if precomputed {
            ensure!(self.sp.is_none(),"pooled attention cannot consume a sequence shard");
            self.attention.project_precomputed(self.attn_boundary.input(),positions,locations,cache,comm,stream)
        }else{
            Self::gather_input(&mut self.sp,&self.attn_boundary,comm,stream)?;
            self.attention.project_distributed_quantized(
                match &self.sp{Some(sp)=>&sp.gathered,None=>self.attn_boundary.input()},
                self.attn_boundary.quantized_input(),positions,locations,cache,Some(comm),stream)
        }
    }
    pub fn run_indexed_from_previous(
        &mut self,
        residual: &Buffer<bf16>,
        previous_pre: &Buffer<f32>,
        positions: &Positions,
        locations: &RowLocations,
        cache: &mut PagedKv,
        candidates: &Candidates,
        tokens: &TokenRows,
        source: &mut crate::compressed_source::Layer,
        source_metadata: &crate::compressed_source::Metadata,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            [2, 8, 14, 20].contains(&self.layer)
                && source.layer_id() == self.layer
                && self.engram.is_none(),
            "ratio2 source-layer graph contract"
        );
        let Some(precomputed)=self.prepare_attention_input(residual,previous_pre,stream)? else{return Ok(());};
        self.project_attention_input(precomputed,positions,locations,cache,comm,stream)?;
        if self.attention.query_overlap_enabled() {
            // Q-B's quantized workspace is still in flight on its side stream.
            // The index's own canonical quantizer consumes the ready BF16
            // normalized values, keeping its scheduling independent.
            source.run_distributed(
                match &self.sp { Some(sp) => &sp.gathered, None => self.attn_boundary.input() },
                self.attention.normalized_lora(), source_metadata, tokens.live(), comm, stream,
            )?;
        } else {
            let (q, s) = self.attention.quantized_lora();
            source.run_quantized_distributed(
            match &self.sp { Some(sp) => &sp.gathered, None => self.attn_boundary.input() },
            q,
            s,
            source_metadata,
            tokens.live(),
            comm,
            stream,
        )?;
        }
        self.finish_compressed(
            residual, positions, cache, candidates, tokens, source, comm, stream,
        )
    }
    /// Each source-sharing group uses its source's latents/selection for this step.
    /// They still compute and store their own SWA keys and their own Q/output.
    pub fn run_shared_from_previous(
        &mut self,
        residual: &Buffer<bf16>,
        previous_pre: &Buffer<f32>,
        positions: &Positions,
        locations: &RowLocations,
        cache: &mut PagedKv,
        candidates: &Candidates,
        tokens: &TokenRows,
        source: &crate::compressed_source::Layer,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            (((3..=7).contains(&self.layer) && source.layer_id() == 2)
                || ((9..=13).contains(&self.layer) && source.layer_id() == 8)
                || ((15..=19).contains(&self.layer) && source.layer_id() == 14)
                || ((21..=23).contains(&self.layer) && source.layer_id() == 20))
                && self.engram.is_none(),
            "shared compressed-source graph contract"
        );
        let Some(precomputed)=self.prepare_attention_input(residual,previous_pre,stream)? else{return Ok(());};
        if let Some(sp) = self.sp.as_mut().filter(|sp| sp.use_quantized && !precomputed) {
            self.attention.project_row_shard(self.attn_boundary.input(), positions, locations,
                cache, comm, sp.quantized.as_mut().ok_or_else(|| anyhow::anyhow!("quantized shard absent"))?, sp.use_local_qkv, stream)?;
        } else {
            self.project_attention_input(precomputed,positions,locations,cache,comm,stream)?;
        }
        self.finish_compressed(
            residual, positions, cache, candidates, tokens, source, comm, stream,
        )
    }
    /// Decoder replay uses persisted source20 selection for the whole tail.
    pub(crate) fn run_shared_candidates_from_previous(
        &mut self,
        residual: &Buffer<bf16>,
        previous_pre: &Buffer<f32>,
        positions: &Positions,
        locations: &RowLocations,
        cache: &mut PagedKv,
        candidates: &Candidates,
        tokens: &TokenRows,
        source: &crate::compressed_source::Layer,
        source_candidates: &Candidates,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            (21..=23).contains(&self.layer) && source.layer_id() == 20 && self.engram.is_none(),
            "decoder source20 replay contract"
        );
        let Some(precomputed)=self.prepare_attention_input(residual,previous_pre,stream)? else{return Ok(());};
        self.project_attention_input(precomputed,positions,locations,cache,comm,stream)?;
        self.finish_auxiliary(
            residual,
            positions,
            cache,
            candidates,
            tokens,
            source.cache(),
            source_candidates,
            comm,
            stream,
        )
    }
    fn finish_compressed(
        &mut self,
        residual: &Buffer<bf16>,
        positions: &Positions,
        cache: &PagedKv,
        candidates: &Candidates,
        tokens: &TokenRows,
        source: &crate::compressed_source::Layer,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        self.finish_auxiliary(
            residual,
            positions,
            cache,
            candidates,
            tokens,
            source.cache(),
            source.candidates(),
            comm,
            stream,
        )
    }
    /// A later index source reuses source20's KV and keys while computing its
    /// own queries and selected positions from its live normalized input.
    pub fn run_reindexed_from_previous(
        &mut self,
        residual: &Buffer<bf16>,
        previous_pre: &Buffer<f32>,
        positions: &Positions,
        locations: &RowLocations,
        cache: &mut PagedKv,
        candidates: &Candidates,
        tokens: &TokenRows,
        owner: &crate::compressed_source::Layer,
        index: &mut crate::compressed_source::ReadIndex,
        metadata: &crate::compressed_source::Metadata,
        replay_mask: Option<&Buffer<u8>>,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            self.layer == index.layer_id() && owner.layer_id() == 20 && self.engram.is_none(),
            "read-index layer contract"
        );
        let Some(precomputed)=self.prepare_attention_input(residual,previous_pre,stream)? else{return Ok(());};
        self.project_attention_input(precomputed,positions,locations,cache,comm,stream)?;
        if self.attention.query_overlap_enabled() {
            index.run(
                match &self.sp { Some(sp) => &sp.gathered, None => self.attn_boundary.input() },
                self.attention.normalized_lora(), owner, metadata, replay_mask, stream,
            )?;
        } else {
            let (q, scales) = self.attention.quantized_lora();
            index.run_quantized(
            match &self.sp { Some(sp) => &sp.gathered, None => self.attn_boundary.input() },
            q,
            scales,
            owner,
            metadata,
            replay_mask,
            stream,
        )?;
        }
        self.finish_auxiliary(
            residual,
            positions,
            cache,
            candidates,
            tokens,
            owner.cache(),
            index.candidates(),
            comm,
            stream,
        )
    }
    pub fn run_selected_from_previous(
        &mut self,
        residual: &Buffer<bf16>,
        previous_pre: &Buffer<f32>,
        positions: &Positions,
        locations: &RowLocations,
        cache: &mut PagedKv,
        candidates: &Candidates,
        tokens: &TokenRows,
        owner: &crate::compressed_source::Layer,
        index: &crate::compressed_source::ReadIndex,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            (25..=39).contains(&self.layer)
                && self.layer % 4 != 0
                && self.layer / 4 * 4 == index.layer_id()
                && owner.layer_id() == 20
                && self.engram.is_none(),
            "shared read-index layer contract"
        );
        let Some(precomputed)=self.prepare_attention_input(residual,previous_pre,stream)? else{return Ok(());};
        self.project_attention_input(precomputed,positions,locations,cache,comm,stream)?;
        self.finish_auxiliary(
            residual,
            positions,
            cache,
            candidates,
            tokens,
            owner.cache(),
            index.candidates(),
            comm,
            stream,
        )
    }
    fn finish_auxiliary(
        &mut self,
        residual: &Buffer<bf16>,
        positions: &Positions,
        cache: &PagedKv,
        candidates: &Candidates,
        tokens: &TokenRows,
        auxiliary_cache: &PagedKv,
        auxiliary_candidates: &Candidates,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        self.attention.finish(
            positions,
            cache,
            candidates,
            &self.sink,
            Some((auxiliary_cache, auxiliary_candidates)),
            stream,
        )?;
        self.ffn.prefetch_begin(stream)?;
        unsafe { Self::reduce(self.sp.is_some(), comm, self.attention.output(), &mut self.attn_reduced, stream) }?;
        self.attn_boundary
            .finish(&self.attn_reduced, residual, stream)?;
        self.ffn_boundary.prepare(
            self.attn_boundary.output(),
            Some(self.attn_boundary.pre()),
            stream,
        )?;
        if self.pair_deferred_ffn {return Ok(());}
        self.run_ffn(tokens, comm, stream)?;
        self.finish_ffn(comm,stream)
    }
    /// Explicitly admitted paired execution stops after preparing this owner's normalized
    /// FFN input. Its canonical collective and residual boundary resume only
    /// after the pooled or separate expert work has produced its own output.
    pub(crate) fn defer_pair_ffn(&mut self,enabled:bool)->Result<()> {
        ensure!((std::env::var("CANADA_SCRATCH_PAIRED_MODEL_SCREEN").as_deref()==Ok("1")
            || crate::config::paired_serving_requested())
            && self.sp.is_none() && self.ffn.output().len()==6*5120,
            "deferred FFN requires explicitly admitted nonsharded M6 pair");
        self.pair_deferred_ffn=enabled;Ok(())
    }
    pub(crate) fn finish_pair_ffn(&mut self,comm:&Roce,stream:&Stream)->Result<()> {
        ensure!(self.pair_deferred_ffn,"paired FFN was not prepared");
        self.finish_ffn(comm,stream)
    }
    pub(crate) fn finish_pair_ffn_reduced(&mut self,reduced:&Buffer<bf16>,cuda:&Cuda,stream:&Stream)->Result<()> {
        ensure!(self.pair_deferred_ffn&&self.sp.is_none()&&reduced.len()>0
            &&reduced.len()<=self.ffn_reduced.len()&&reduced.len()%5120==0
            &&cuda.owns_buffer(reduced)&&cuda.owns_buffer(&self.ffn_reduced),"paired FFN reduction owner");
        self.ffn.prefetch_join(stream)?;
        if let Some(hint)=&self.next_input_prefetch {hint.begin(stream)?;}
        unsafe{cuda.copy_device(self.ffn_reduced.ptr(),reduced.ptr(),reduced.bytes(),stream)?;}
        self.ffn_boundary.finish(&self.ffn_reduced,self.attn_boundary.output(),stream)?;
        if let Some(hint)=&self.next_input_prefetch {hint.join(stream)?;}
        Ok(())
    }
    fn finish_ffn(&mut self,comm:&Roce,stream:&Stream)->Result<()> {
        self.ffn.prefetch_join(stream)?;
        if let Some(hint) = &self.next_input_prefetch {
            hint.begin(stream)?;
        }
        unsafe { Self::reduce(self.sp.is_some(), comm, self.ffn.output(), &mut self.ffn_reduced, stream) }?;
        self.ffn_boundary
            .finish(&self.ffn_reduced, self.attn_boundary.output(), stream)?;
        if let Some(hint) = &self.next_input_prefetch {
            hint.join(stream)?;
        }
        Ok(())
    }
    pub fn engram(&self) -> Option<&Engram> {
        self.engram.as_ref()
    }
    pub fn pre(&self) -> &Buffer<f32> {
        self.ffn_boundary.pre()
    }
    pub fn attention(&self) -> &Attention {
        &self.attention
    }
    pub(crate) fn set_split_attention_for_probe(&mut self,path:u8,stream:&Stream)->Result<()> {
        self.attention.set_split_attention_for_probe(path,stream)
    }
    pub(crate) fn set_mhc_two_mma_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        self.attn_boundary.set_two_mma_for_probe(enabled,stream)?;
        self.ffn_boundary.set_two_mma_for_probe(enabled,stream)
    }
    pub(crate) fn set_wide_mhc_for_probe(&mut self,mode:u8)->Result<()> {
        self.attn_boundary.set_wide_mhc_for_probe(mode)?;
        self.ffn_boundary.set_wide_mhc_for_probe(mode)
    }
    pub(crate) fn set_wide_epilogues_for_probe(&mut self,enabled:bool,kernels:&Path)->Result<()> {
        self.attention.set_wide_epilogues_for_probe(enabled,kernels)?;
        self.attn_boundary.set_wide_epilogues_for_probe(enabled,kernels)?;
        self.ffn_boundary.set_wide_epilogues_for_probe(enabled,kernels)
    }
    pub(crate) fn set_mhc_pdl_pair_for_probe(&mut self, mode: u8) -> Result<()> {
        ensure!(mode <= 3, "MHC PDL screen mode");
        for boundary in [&mut self.attn_boundary, &mut self.ffn_boundary] {
            boundary.set_pdl_pair_for_probe(mode & 1 != 0)?;
            boundary.set_pdl_stats_for_probe(mode & 2 != 0)?;
        }
        Ok(())
    }
    pub(crate) fn set_append_bf16_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        self.attention.set_append_bf16_for_probe(enabled,stream)
    }
    pub(crate) fn set_woa_prefetch_for_probe(&mut self,enabled:bool)->Result<()> {
        self.attention.set_woa_prefetch_for_probe(enabled)
    }
    pub(crate) fn set_append_split_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        self.attention.set_append_split_for_probe(enabled,stream)
    }
    pub(crate) fn swap_qkv_for_probe(&mut self,other:&mut crate::fp8_linear::Linear)->Result<()> {
        self.attention.swap_qkv_for_probe(other)
    }
    pub(crate) fn set_wo_b_pdl_for_probe(&mut self,mode:u8,other:&mut crate::fp8_linear::Linear)->Result<()> {
        self.attention.set_wo_b_pdl_for_probe(mode,other)
    }
    pub(crate) fn swap_shared_up_for_probe(&mut self,other:&mut crate::fp8_linear::Linear)->Result<()> {
        self.ffn.swap_shared_up_for_probe(other)
    }
    pub(crate) fn set_attention_coefficients_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        self.attn_boundary.set_verify16_overlap_for_probe(enabled,stream)
    }
    pub(crate) fn set_query_overlap(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        self.attention.set_query_overlap(enabled, stream)
    }
    pub(crate) fn set_verify16_query_for_probe(&mut self, enabled:bool, stream:&Stream)->Result<()> {
        self.attention.set_verify16_query_for_probe(enabled,stream)
    }
    pub(crate) fn set_verify16_attention_for_probe(&mut self, enabled:bool, stream:&Stream)->Result<()> {
        self.attention.set_verify16_query_for_probe(enabled,stream)?;
        self.attn_boundary.set_verify16_overlap_for_probe(enabled,stream)
    }
    pub(crate) fn attention_coefficients_overlap(&self)->bool {
        self.attn_boundary.overlap_active()
    }
    pub(crate) fn set_norm_fusion(&mut self, enabled: bool) -> Result<()> {
        self.attn_boundary.set_norm_fusion(enabled)?;
        self.ffn_boundary.set_norm_fusion(enabled)
    }
    pub(crate) fn fullk_boundaries(&self) -> usize {
        usize::from(self.attn_boundary.fullk_active()) + usize::from(self.ffn_boundary.fullk_active())
    }
    pub(crate) fn set_fullk(&mut self, enabled: bool) -> Result<()> {
        self.attn_boundary.set_fullk(enabled)?;
        self.ffn_boundary.set_fullk(enabled)
    }
    fn run_ffn(&mut self, tokens: &TokenRows, comm: &Roce, stream: &Stream) -> Result<()> {
        if self.ffn.row_transport_active() {
            let sp = self.sp.as_ref().ok_or_else(|| anyhow::anyhow!("packed FFN requires SP rows"))?;
            if comm.moe_pipeline_active()&&self.ffn.pipeline_eligible() {
                return self.ffn.run_pipelined_row_transport(&self.moe_workspace,self.ffn_boundary.input(),tokens,sp.rank,comm,stream);
            }
            self.ffn.run_row_transport(&self.moe_workspace,self.ffn_boundary.input(),tokens,sp.rank,comm,stream)
        } else {
            Self::gather_input(&mut self.sp,&self.ffn_boundary,comm,stream)?;
            self.ffn.run_quantized_input(&self.moe_workspace,
                match &self.sp {Some(sp)=>&sp.gathered,None=>self.ffn_boundary.input()},
                self.ffn_boundary.quantized_input(),tokens,stream)
        }
    }
    pub(crate) fn norm_quant_boundaries(&self) -> usize {
        usize::from(self.attn_boundary.norm_quant_enabled()) + usize::from(self.ffn_boundary.norm_quant_enabled())
    }
    pub(crate) fn moe_gather_active(&self) -> bool { self.ffn.row_transport_active() }
    pub(crate) fn set_moe_gather(&mut self, enabled: bool) { self.ffn.set_row_transport(enabled); }
    pub(crate) fn quant_gather_active(&self) -> bool {
        self.sp.as_ref().is_some_and(|sp| sp.use_quantized)
    }
    pub(crate) fn local_qkv_active(&self) -> bool {
        self.sp.as_ref().is_some_and(|sp| sp.use_quantized && sp.use_local_qkv)
    }
    pub(crate) fn set_local_qkv(&mut self, enabled: bool) {
        if let Some(sp) = self.sp.as_mut().filter(|sp| sp.local_qkv_prepared) {
            sp.use_local_qkv = enabled;
        }
    }
    pub(crate) fn set_quant_gather(&mut self, enabled: bool) {
        if let Some(sp) = self.sp.as_mut().filter(|sp| sp.quantized.is_some()) {
            sp.use_quantized = enabled;
        }
    }
    pub fn ffn(&self) -> &LayerFfn {
        &self.ffn
    }
    pub(crate) fn schedule_counts(&self)->[usize;4] {
        let(ffn,hint)=self.ffn.schedule_flags();
        let(qhint,pdl)=self.attention.schedule_flags();
        [usize::from(hint)+usize::from(qhint)+usize::from(self.next_input_prefetch.is_some()),
         usize::from(ffn),usize::from(self.attn_boundary.overlap_active())+usize::from(self.ffn_boundary.overlap_active()),
         usize::from(pdl)]
    }
    pub fn routed(&self) -> &Buffer<bf16> {
        self.moe_workspace.output()
    }
    pub(crate) fn micro_moe_fp32(&self) -> bool {
        self.moe_workspace.micro_fp32()
    }
    pub(crate) fn live_moe_bounds(&self) -> bool { self.moe_workspace.live_bounds() }
    pub(crate) fn moe_task_order(&self) -> bool { self.moe_workspace.task_order() }
    pub(crate) fn moe_task_order_m3(&self)->bool {self.moe_workspace.task_order_m3()}
    pub(crate) fn narrow_projections(&self) -> bool { self.attention().narrow_projection() && self.ffn.narrow_projection() }
    pub(crate) fn fused_projection(&self) -> bool { self.attention().fused_projection() }
    pub(crate) fn moe_scale_reuse(&self) -> bool { self.moe_workspace.scale_reuse() }
    pub fn attn_boundary(&self) -> &Boundary {
        &self.attn_boundary
    }
    pub fn ffn_boundary(&self) -> &Boundary {
        &self.ffn_boundary
    }
    pub fn attn_reduced(&self) -> &Buffer<bf16> {
        &self.attn_reduced
    }
    pub fn ffn_reduced(&self) -> &Buffer<bf16> {
        &self.ffn_reduced
    }
}

impl SwaLayer {
    pub(crate) fn set_coalesced_codec(&mut self,enabled:bool,kernels:&Path)->Result<()> {
        self.ffn.set_coalesced_codec(enabled,kernels)
    }
    pub(crate) fn coalesced_codec_active(&self)->bool {self.ffn.coalesced_codec_active()}
}
