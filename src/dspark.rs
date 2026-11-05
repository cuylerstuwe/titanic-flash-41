//! Independent native-quant DSpark three-stage query graph. Rust owns loading,
//! expert preparation, retained KV, collective ordering and graph execution.
//! Markov correction and greedy proposals are a separate GPU graph. Target-
//! correct stochastic acceptance and integrated generation remain separate work.
use crate::{
    attention::{Candidates, PagedKv},
    checkpoint::{Checkpoint, Dtype},
    cuda::{Buffer, Cuda, Graph, Module, Stream},
    dspark_conditioning::Conditioning,
    dspark_kv::{QUERY_ROWS, State},
    expert_packing::{self, Prepared},
    ffn::{LayerFfn, TokenRows},
    kv::RowLocations,
    layer_attention::Layer as Attention,
    layer_id::LayerId,
    layout::RankPlan,
    mhc::Boundary,
    model_io::ModelIo,
    moe::Workspace,
    prelude::{FrequencyTable, Native, Positions},
    roce::Roce,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{path::Path, rc::Rc};

#[path = "dspark_pair_query.rs"]
mod pair_query;
#[path="dspark_query_reuse_probe.rs"]
mod query_reuse_probe;
pub(crate) use query_reuse_probe::QueryOutputSnapshot;

struct Stage {
    attention: Attention,
    ffn: LayerFfn,
    attn_boundary: Boundary,
    ffn_boundary: Boundary,
    attn_reduced: Buffer<bf16>,
    ffn_reduced: Buffer<bf16>,
    sink: Buffer<f32>,
}
impl Stage {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: Rc<Weights>,
        prepared: Prepared,
        kernels: &Path,
        frequencies: Rc<FrequencyTable>,
        stage: usize,
    ) -> Result<Self> {
        let id = LayerId::draft(stage)?;
        ensure!(
            prepared.layer == id.key() && prepared.experts == 128,
            "DSpark stage expert identity"
        );
        stream.join_default()?;
        let name = format!("{}.attn.attn_sink", id.prefix());
        let v = weights.view(&name)?;
        ensure!(
            v.dtype == Dtype::F32 && v.shape == [16],
            "DSpark TP4 attention sink"
        );
        let sink = cuda.allocate::<f32>(16)?;
        unsafe { cuda.copy_device(sink.ptr(), weights.ptr(&name)?, sink.bytes(), stream) }?;
        Ok(Self {
            attention: Attention::new_draft(
                cuda,
                module,
                stream,
                weights.clone(),
                stage,
                kernels,
                frequencies,
            )?,
            attn_boundary: Boundary::new_named(
                cuda,
                stream,
                weights.clone(),
                id,
                "attn",
                QUERY_ROWS,
                kernels,
            )?,
            ffn_boundary: Boundary::new_named(
                cuda,
                stream,
                weights.clone(),
                id,
                "ffn",
                QUERY_ROWS,
                kernels,
            )?,
            ffn: LayerFfn::new_rows(
                cuda, module, stream, weights, prepared, kernels, QUERY_ROWS, None,
            )?,
            attn_reduced: cuda.allocate(QUERY_ROWS * 5120)?,
            ffn_reduced: cuda.allocate(QUERY_ROWS * 5120)?,
            sink,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        residual: &Buffer<bf16>,
        previous_pre: Option<&Buffer<f32>>,
        positions: &Positions,
        locations: &RowLocations,
        cache: &mut PagedKv,
        candidates: &Candidates,
        tokens: &TokenRows,
        workspace: &Workspace,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        self.attn_boundary.prepare(residual, previous_pre, stream)?;
        self.attention.run_distributed(
            self.attn_boundary.input(),
            positions,
            locations,
            cache,
            candidates,
            &self.sink,
            Some(comm),
            stream,
        )?;
        unsafe { comm.sum_bf16(self.attention.output(), &mut self.attn_reduced, stream) }?;
        self.attn_boundary
            .finish(&self.attn_reduced, residual, stream)?;
        self.ffn_boundary.prepare(
            self.attn_boundary.output(),
            Some(self.attn_boundary.pre()),
            stream,
        )?;
        self.ffn
            .run(workspace, self.ffn_boundary.input(), tokens, stream)?;
        unsafe { comm.sum_bf16(self.ffn.output(), &mut self.ffn_reduced, stream) }?;
        self.ffn_boundary
            .finish(&self.ffn_reduced, self.attn_boundary.output(), stream)
    }
    fn output(&self) -> &Buffer<bf16> {
        self.ffn_boundary.output()
    }
    fn pre(&self) -> &Buffer<f32> {
        self.ffn_boundary.pre()
    }
}

pub struct QueryNetwork {
    // Drop the captured executable before any pointer storage.
    graph: Option<Graph>,
    pair_body_graph:Option<Graph>,
    pair_head:Option<crate::draft_head::Pair>,
    cuda: Cuda,
    module: Module,
    stream: usize,
    rank: usize,
    comm_instance: Option<u64>,
    markov: crate::dspark_markov::Markov,
    last_anchor: Option<u32>,
    weights: Rc<Weights>,
    prepared_layouts: Vec<Prepared>,
    io: ModelIo,
    private_head: Option<crate::draft_head::DraftHead>,
    private_head_active: bool,
    state: State,
    six: Conditioning,
    split_six: Option<Conditioning>,
    conditioning_split: bool,
    conditioning_graph: bool,
    full_commit_device_metadata: bool,
    wide: Conditioning,
    stages: Vec<Stage>,
    workspace: Workspace,
    positions: Positions,
    tokens: TokenRows,
    norm: Native,
    norm_weight: Buffer<bf16>,
    combined: Buffer<bf16>,
    normalized: Buffer<bf16>,
    ready: bool,
    gather_base_logits: bool,
    poisoned: bool,
}
impl QueryNetwork {
    pub(crate) fn independent_for_probe(&self,target:&crate::target::Target,core:&[u8],
        stream:&Stream,kernels:&Path)->Result<Self> {
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_PAIRED_GENERATION_SCREEN").as_deref()==Ok("1")
            ||crate::config::resident_banks_requested(),
            "independent drafter requires finite pairing admission");
        let (io,frequencies)=target.draft_resources(core,stream,kernels)?;
        let mut fork=Self::from_prepared(&self.cuda,core,stream,self.cuda.module(core)?,
            self.weights.clone(),self.prepared_layouts.clone(),self.rank,kernels,io,frequencies)?;
        if std::env::var("CANADA_SCRATCH_SHARED_DRAFT_HEAD").as_deref()==Ok("1") {
            ensure!(fork.graph.is_none()&&fork.pair_body_graph.is_none()&&!fork.ready,
                "shared draft encoding must precede every peer graph");
            let source=self.private_head.as_ref().ok_or_else(||anyhow::anyhow!("shared draft head source absent"))?;
            let destination=fork.private_head.as_mut().ok_or_else(||anyhow::anyhow!("shared draft head destination absent"))?;
            let reclaimed=destination.share_encoding_from(source,stream)?;
            eprintln!("rank{} shared immutable draft-head encoding: reclaimed {} explicit bytes; private query/output/policy retained",self.rank,reclaimed);
        }
        ensure!(Rc::ptr_eq(&self.weights,&fork.weights)&&self.state.independent_from(&fork.state)?
            &&self.io.local_logits().ptr()!=fork.io.local_logits().ptr()
            &&self.normalized.ptr()!=fork.normalized.ptr(),"independent drafter aliased mutable storage");
        Ok(fork)
    }
    pub(crate) fn sampled_graphs_ready(&self) -> bool {
        !self.poisoned && (self.graph.is_some()||self.pair_body_graph.is_some()) && self.markov.sampled_graph_ready()
    }
    #[allow(clippy::too_many_arguments)]
    pub fn for_target(
        target: &crate::target::Target,
        cuda: &Cuda,
        core: &[u8],
        stream: &Stream,
        cp: &Checkpoint,
        plan: &RankPlan,
        staged: &Path,
        kernels: &Path,
    ) -> Result<Self> {
        let (io, frequencies) = target.draft_resources(core, stream, kernels)?;
        Self::load_staged(
            cuda,
            core,
            stream,
            cp,
            plan,
            staged,
            kernels,
            io,
            frequencies,
        )
    }
    /// `io` shares the target's canonical embedding/head atlas. The staged MTP
    /// atlas contains only original checkpoint bytes, never exported activations
    /// or a serialized reference engine. All model preparation happens in Rust.
    #[allow(clippy::too_many_arguments)]
    pub fn load_staged(
        cuda: &Cuda,
        core: &[u8],
        stream: &Stream,
        cp: &Checkpoint,
        plan: &RankPlan,
        staged: &Path,
        kernels: &Path,
        io: ModelIo,
        frequencies: Rc<FrequencyTable>,
    ) -> Result<Self> {
        let weights = Weights::load_staged(cuda, cp, plan, &["mtp."], staged)?;
        Self::from_weights(
            cuda,
            core,
            stream,
            weights,
            plan.rank,
            kernels,
            io,
            frequencies,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn from_weights(
        cuda: &Cuda,
        core: &[u8],
        stream: &Stream,
        mut weights: Weights,
        rank: usize,
        kernels: &Path,
        io: ModelIo,
        frequencies: Rc<FrequencyTable>,
    ) -> Result<Self> {
        ensure!(
            rank < 4
                && io.matches_profile(rank, QUERY_ROWS, cuda)
                && cuda.owns_stream(stream)
                && weights.in_context(cuda)
                && !frequencies.compressed()
                && cuda.owns_buffer(frequencies.values()),
            "DSpark context/geometry"
        );
        stream.join_default()?;
        let module = cuda.module(core)?;
        let mut prepared = Vec::new();
        for stage in 0..3 {
            prepared.push(expert_packing::prepare_draft(
                cuda,
                &module,
                &mut weights,
                stage,
            )?);
        }
        Self::from_prepared(cuda,core,stream,module,Rc::new(weights),prepared,rank,kernels,io,frequencies)
    }
    #[allow(clippy::too_many_arguments)]
    fn from_prepared(cuda:&Cuda,core:&[u8],stream:&Stream,module:Module,weights:Rc<Weights>,
        prepared:Vec<Prepared>,rank:usize,kernels:&Path,io:ModelIo,frequencies:Rc<FrequencyTable>)->Result<Self> {
        ensure!(prepared.len()==3&&rank<4&&weights.in_context(cuda)&&cuda.owns_stream(stream)
            &&io.matches_profile(rank,QUERY_ROWS,cuda)&&!frequencies.compressed()
            &&cuda.owns_buffer(frequencies.values()),"prepared DSpark owner/geometry");
        let prepared_layouts=prepared.clone();
        let state = State::new(cuda, stream, &weights, kernels, frequencies.clone())?;
        let mut stages = Vec::new();
        for (stage, layout) in prepared.into_iter().enumerate() {
            stages.push(Stage::new(
                cuda,
                &module,
                stream,
                weights.clone(),
                layout,
                kernels,
                frequencies.clone(),
                stage,
            )?);
        }
        let v = weights.view("mtp.2.norm.weight")?;
        ensure!(
            v.dtype == Dtype::Bf16 && v.shape == [5120],
            "DSpark final norm"
        );
        let norm_weight = cuda.allocate::<bf16>(5120)?;
        unsafe {
            cuda.copy_device(
                norm_weight.ptr(),
                weights.ptr("mtp.2.norm.weight")?,
                norm_weight.bytes(),
                stream,
            )
        }?;
        let private_head = if std::env::var("CANADA_SCRATCH_DRAFT_HEAD_FP8").as_deref() == Ok("1") {
            let mut head=io.make_private_draft_head(&kernels.join("draft-head-fp8-v1"), stream)?;
            if std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_GEMV").as_deref()==Ok("1") {
                head=head.with_gemv(&kernels.join("draft-head-fp8-gemv-m5-v1"))?;
            }
            if std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_INT8_SCREEN").as_deref()==Ok("1") {
                head=head.with_int8(&kernels.join("draft-head-int8-m5-s815"),stream)?;
                if std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_LOSSLESS_SCREEN").as_deref()==Ok("1") {
                    head=head.with_lossless(&kernels.join("draft-head-lossless-m5-s817"),stream)?;
                }
            }
            if std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_NVFP4_SCREEN").as_deref()==Ok("1") {
                head=head.with_nvfp4(&kernels.join("draft-head-nvfp4-marlin-m5-v1"),stream)?;
            }
            Some(head)
        } else {
            None
        };
        // Only this drafter-owned profile receives M5 GEMV. Target verification
        // profiles and their already-qualified artifact bundle remain unchanged.
        let io = if std::env::var("CANADA_SCRATCH_DRAFT_HEAD_GEMV").as_deref() == Ok("1") {
            io.with_draft_head_gemv(&kernels.join("draft-head-gemv-m5-v1"))?
        } else { io };
        let conditioning_split=match std::env::var("CANADA_SCRATCH_DRAFT_MAIN_SPLIT") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid draft conditioning split policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(e)=>return Err(e.into()),
        };
        let split_six=if conditioning_split || std::env::var("CANADA_SCRATCH_CONDITIONING_AB").as_deref()==Ok("1") {
            Some(Conditioning::new_split(cuda,core,stream,weights.clone(),kernels,rank)?)
        }else{None};
        let conditioning_graph=match std::env::var("CANADA_SCRATCH_CONDITIONING_GRAPH") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid conditioning graph policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(e)=>return Err(e.into()),
        };
        let full_commit_device_metadata=match std::env::var("CANADA_SCRATCH_FULL_COMMIT_DEVICE_METADATA") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid full-commit metadata policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(e)=>return Err(e.into()),
        };
        let result = Self {
            graph: None,
            pair_body_graph:None,
            pair_head:if crate::config::paired_draft_head_requested(){
                ensure!(crate::config::paired_serving_requested()
                    ||std::env::var("CANADA_SCRATCH_PAIRED_GENERATION_SCREEN").as_deref()==Ok("1"),
                    "paired draft head requires independent generation owners");
                Some(crate::draft_head::Pair::new(cuda,&kernels.join("draft-head-pool-m10-v1"),stream)?)
            }else{None},
            cuda: cuda.clone(),
            module,
            stream: stream.raw() as usize,
            rank,
            comm_instance: None,
            markov: crate::dspark_markov::Markov::new(cuda, core, stream, rank, weights.clone(), kernels)?,
            last_anchor: None,
            six: Conditioning::new(cuda, core, stream, weights.clone(), kernels, 6)?,
            split_six,conditioning_split,conditioning_graph,full_commit_device_metadata,
            wide: Conditioning::new(cuda, core, stream, weights.clone(), kernels, 128)?,
            positions: Positions::upload(cuda, &[0, 1, 2, 3, 4], frequencies.rows())?,
            tokens: TokenRows::new(cuda, &[128799; QUERY_ROWS], QUERY_ROWS)?,
            workspace: Workspace::new(cuda, &kernels.join("moe-runtime-dspark-m5-v1"))?,
            norm: Native::open(cuda, &kernels.join("prelude-v1"))?,
            norm_weight,
            combined: cuda.allocate(QUERY_ROWS * 5120)?,
            normalized: cuda.allocate(QUERY_ROWS * 5120)?,
            weights,
            prepared_layouts,
            io,
            private_head_active: private_head.is_some(),
            private_head,
            state,
            stages,
            ready: false,
            gather_base_logits: true,
            poisoned: false,
        };
        stream.join_default()?;
        stream.synchronize()?;
        Ok(result)
    }
    fn guard(&self, stream: &Stream) -> Result<()> {
        ensure!(
            !self.poisoned && self.cuda.owns_stream(stream) && self.stream == stream.raw() as usize,
            "DSpark network poisoned or foreign stream"
        );
        Ok(())
    }
    pub(crate) fn probe_conditioning_split(
        &self,core:&[u8],kernels:&Path,features:&Buffer<bf16>,comm:&Roce,stream:&Stream,
    )->Result<serde_json::Value>{
        self.guard(stream)?;
        crate::dspark_conditioning::probe_split(&self.cuda,core,stream,self.weights.clone(),
            kernels,self.rank,features,comm)
    }
    pub(crate) fn conditioning_split(&self)->bool{self.conditioning_split}
    pub(crate) fn set_full_commit_metadata_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_COPY_COMMIT_GENERATION_SCREEN").as_deref()==Ok("1"),
            "full-commit metadata switch requires finite generation screen");
        self.state.position()?;stream.synchronize()?;
        self.full_commit_device_metadata=enabled;Ok(())
    }
    pub(crate) fn probe_copy_conditioning(
        &self,core:&[u8],kernels:&Path,features:&Buffer<bf16>,comm:&Roce,stream:&Stream,
    )->Result<serde_json::Value>{
        self.guard(stream)?;
        crate::dspark_conditioning::probe_copy_width(&self.cuda,core,stream,self.weights.clone(),
            kernels,self.rank,features,comm)
    }
    pub(crate) fn set_conditioning_kv_reuse_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()>{
        self.guard(stream)?;
        ensure!(!self.conditioning_graph,"isolate KV reuse from the cached-graph variant");
        self.six.set_kv_input_reuse_for_probe(enabled,stream)?;
        if let Some(split)=self.split_six.as_mut(){split.set_kv_input_reuse_for_probe(enabled,stream)?;}
        self.wide.set_kv_input_reuse_for_probe(enabled,stream)
    }
    pub(crate) fn set_conditioning_graph_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()>{
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_CONDITIONING_GRAPH_NATIVE_SCREEN").as_deref()==Ok("1"),
            "conditioning graph override requires the paired screen");
        stream.synchronize()?;
        self.conditioning_graph=enabled;
        Ok(())
    }
    pub(crate) fn probe_conditioning_graph(
        &self,core:&[u8],kernels:&Path,features:&Buffer<bf16>,comm:&Roce,stream:&Stream,
    )->Result<serde_json::Value>{
        self.guard(stream)?;
        crate::dspark_conditioning::probe_graph(&self.cuda,core,stream,self.weights.clone(),
            kernels,self.rank,features,comm)
    }
    pub(crate) fn set_conditioning_split(&mut self,enabled:bool,stream:&Stream)->Result<()>{
        self.guard(stream)?;
        ensure!(!enabled || self.split_six.is_some(),"split conditioning workspace not prepared");
        self.conditioning_split=enabled;
        self.ready=false;self.last_anchor=None;self.markov.invalidate_results();
        Ok(())
    }
    pub(crate) fn head_kind(&self) -> &'static str {
        if self.private_head_active {
            if self.private_head.as_ref().is_some_and(|h|h.nvfp4_enabled()){return "private_nvfp4_marlin_bf16_operands_output_fp32_publication";}
            if self.private_head.as_ref().is_some_and(|h|h.lossless_enabled()){return "canonical_bf16_lossless_weights_fp32_output_m5";}
            if self.private_head.as_ref().is_some_and(|h|h.int8_enabled()){return "private_int8_group128_bf16_operands_fp32_output";}
            if self.private_head.as_ref().is_some_and(|h|h.gemv_enabled()) {
                "private_e4m3_block32_weights_bf16_operands_fp32_output_gemv_m5"
            }else{"private_e4m3_block32_weights_bf16_operands_fp32_output"}
        } else if self.io.draft_head_gemv_enabled() {
            "canonical_bf16_weights_fp32_output_gemv_m5"
        } else {
            "canonical_bf16_weights_fp32_output"
        }
    }
    pub(crate) fn set_head_gemv_for_probe(&mut self, active: bool, stream: &Stream) -> Result<()> {
        self.guard(stream)?;
        if self.io.draft_head_gemv_enabled() != active {
            stream.join_default()?;
            stream.synchronize()?;
            self.graph = None;
            self.io.set_draft_head_gemv_for_probe(active)?;
            self.ready = false;
            self.last_anchor = None;
            self.markov.invalidate_results();
        }
        Ok(())
    }
    pub(crate) fn set_private_head_gemv_for_probe(&mut self,active:bool,stream:&Stream)->Result<()> {
        self.guard(stream)?;
        ensure!(self.private_head_active,"private GEMV screen requires active private head");
        let head=self.private_head.as_ref().ok_or_else(||anyhow::anyhow!("private head absent"))?;
        if head.gemv_enabled()!=active {
            stream.join_default()?;stream.synchronize()?;self.graph=None;
            head.set_gemv_for_probe(active)?;
            self.ready=false;self.last_anchor=None;self.markov.invalidate_results();
        }
        Ok(())
    }
    pub(crate) fn set_private_head_precision_for_probe(&mut self,kind:u8,stream:&Stream)->Result<()> {
        self.guard(stream)?;
        ensure!(self.private_head_active,"private precision screen requires active private head");
        let head=self.private_head.as_ref().ok_or_else(||anyhow::anyhow!("private head absent"))?;
        if head.precision_kind()!=kind {
            stream.join_default()?;stream.synchronize()?;self.graph=None;
            head.set_precision_for_probe(kind)?;
            self.ready=false;self.last_anchor=None;self.markov.invalidate_results();
        }
        Ok(())
    }
    pub(crate) fn set_private_head_for_probe(&mut self, active: bool, stream: &Stream) -> Result<()> {
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_DRAFT_HEAD_SCREEN").as_deref()==Ok("1")
            && self.private_head.is_some(), "head switch requires the isolated two-head screen");
        if self.private_head_active != active {
            // The graph captures this dispatch. Drain and discard it before
            // changing the head; the probe warms its replacement before timing.
            stream.join_default()?;
            stream.synchronize()?;
            self.graph = None;
            self.private_head_active = active;
            self.ready = false;
            self.last_anchor = None;
            self.markov.invalidate_results();
        }
        Ok(())
    }
    pub(crate) fn warm_head_for_probe(&mut self, anchor:u32, comm:&Roce, stream:&Stream) -> Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_DRAFT_HEAD_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_NVFP4_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_INT8_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_GEMV_SCREEN").as_deref()==Ok("1"),
            "direct head warmup requires the isolated screen");
        self.run(anchor,comm,stream)?;
        stream.synchronize()?;
        comm.health()?;
        ensure!(self.graph.is_some(),"draft query graph was not warmed");
        Ok(())
    }
    pub(crate) fn accepted_repeat_digest(&self,stream:&Stream)->Result<serde_json::Value> {
        self.guard(stream)?;stream.synchronize()?;
        self.state.accepted_replay_digests(None)
    }
    pub(crate) fn replay_digests(&self, root:Option<&Path>, stream:&Stream)->Result<serde_json::Value> {
        use sha2::{Digest,Sha256};
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_DRAFT_REPLAY_TRACE").as_deref()==Ok("1"),"draft replay trace admission");
        stream.synchronize()?;
        let mut taps=vec![("draft-normalized".to_string(),self.normalized.ptr(),self.normalized.bytes())];
        for (i,s)in self.stages.iter().enumerate() {
            taps.push((format!("draft-stage{i}-output"),s.output().ptr(),s.output().bytes()));
        }
        let conditioning=if self.conditioning_split {self.split_six.as_ref().unwrap()}else{&self.six};
        let (norm,kv,live)=conditioning.outputs()?;
        taps.push(("draft-conditioning-norm".into(),norm.ptr(),live*5120*2));
        for (i,b)in kv.iter().enumerate(){taps.push((format!("draft-conditioning-kv{i}"),b.ptr(),live*512*2));}
        let mut records=Vec::new();
        for(name,ptr,bytes)in taps {
            let raw=unsafe{self.cuda.read_device(ptr,bytes)}?;
            if let Some(root)=root {std::fs::write(root.join(format!("{name}.bin")),&raw)?;}
            records.push(serde_json::json!({"name":name,"bytes":bytes,"sha256":format!("{:x}",Sha256::digest(&raw))}));
        }
        Ok(serde_json::json!({"head_kind":self.head_kind(),"accepted":self.state.accepted_replay_digests(root)?,"taps":records}))
    }
    /// Append only target-accepted rows. Rejected draft/target suffixes never
    /// enter the retained draft history. Six and128 are physical capacities.
    pub fn inject(
        &mut self,
        features: &Buffer<bf16>,
        live: usize,
        accepted: usize,
        start: u64,
        stream: &Stream,
    ) -> Result<()> {
        self.inject_inner(features,live,accepted,start,None,false,stream)
    }
    pub(crate) fn inject_distributed(
        &mut self,features:&Buffer<bf16>,live:usize,accepted:usize,start:u64,comm:&Roce,stream:&Stream,
    )->Result<()>{
        ensure!(comm.rank()==self.rank&&self.comm_instance.is_none_or(|c|c==comm.instance()),
            "conditioning communicator changed");
        self.inject_inner(features,live,accepted,start,Some(comm),false,stream)
    }
    pub(crate) fn inject_copy_wide(&mut self,features:&Buffer<bf16>,accepted:usize,start:u64,
        comm:&Roce,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_COPY_WIDE_CONDITIONING").as_deref()==Ok("1")
            &&(19..=32).contains(&accepted)&&features.len()==128*15360
            &&comm.rank()==self.rank&&self.comm_instance.is_none_or(|c|c==comm.instance()),
            "wide accepted-copy conditioning admission");
        self.inject_inner(features,accepted,accepted,start,Some(comm),true,stream)
    }
    /// Capture the local wide projection before serving advertises readiness.
    /// This writes projection scratch only: no accepted KV, sampler or query
    /// state is advanced, and the caller owns the zeroed padded input.
    pub(crate) fn prepare_copy_wide(&mut self,features:&Buffer<bf16>,comm:&Roce,stream:&Stream)->Result<()> {
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_COPY_WIDE_CONDITIONING").as_deref()==Ok("1")
            &&self.state.position()?==0&&features.len()==128*15360&&comm.rank()==self.rank,
            "wide copy preparation requires a fresh admitted drafter");
        self.wide.run_cached_distributed(features,32,Some(comm),stream)?;
        stream.synchronize()
    }
    pub(crate) fn release_prepared_copy_wide(&mut self,stream:&Stream)->Result<()> {
        self.guard(stream)?;
        ensure!(self.state.position()?==0,"only a fresh optional graph may be released");
        self.wide.release_replay(stream)
    }
    /// Keep the existing six-row projection arithmetic, but publish accepted
    /// KV once after all chunks. Projection never reads the retained KV ring.
    /// Staging copies share the projection stream, before its next overwrite.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn inject_copy_batched(&mut self,features:&Buffer<bf16>,accepted:usize,start:u64,
        chunk:&Buffer<bf16>,raw:[&Buffer<bf16>;3],comm:&Roce,stream:&Stream)->Result<()> {
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_COPY_KV_BATCH_SCREEN").as_deref()==Ok("1")
            &&(7..=32).contains(&accepted)&&[16*15360,32*15360,128*15360].contains(&features.len())
            &&accepted<=features.len()/15360&&self.cuda.owns_buffer(features)
            &&chunk.len()==6*15360&&self.cuda.owns_buffer(chunk)
            &&raw.iter().all(|b|b.len()==128*512&&self.cuda.owns_buffer(b))
            &&start==self.state.position()?&&comm.rank()==self.rank
            &&self.comm_instance.is_none_or(|c|c==comm.instance()),"batched copy KV admission");
        self.ready=false;self.last_anchor=None;self.markov.invalidate_results();
        let result=(|| {
            for dst in raw {unsafe{self.cuda.zero_device(dst.ptr(),dst.bytes(),stream)?;}}
            let conditioning=if self.conditioning_split {
                self.split_six.as_mut().ok_or_else(||anyhow::anyhow!("split conditioning absent"))?
            }else{&mut self.six};
            for offset in (0..accepted).step_by(6) {
                let live=(accepted-offset).min(6);
                unsafe {
                    self.cuda.zero_device(chunk.ptr(),chunk.bytes(),stream)?;
                    self.cuda.copy_device(chunk.ptr(),features.ptr()+(offset*15360*2)as u64,live*15360*2,stream)?;
                }
                if self.conditioning_graph {conditioning.run_cached_distributed(chunk,live,Some(comm),stream)?;}
                else{conditioning.run_distributed(chunk,live,Some(comm),stream)?;}
                let(_,projected,n)=conditioning.outputs()?;ensure!(n==live,"copy chunk extent");
                for stage in 0..3 {unsafe {
                    self.cuda.copy_device(raw[stage].ptr()+(offset*512*2)as u64,
                        projected[stage].ptr(),live*512*2,stream)?;
                }}
            }
            self.state.commit(raw,accepted,accepted,start,stream)
        })();
        if result.is_err(){self.poisoned=true;self.state.invalidate();}result
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn inject_copy_queued(&mut self,features:&Buffer<bf16>,accepted:usize,start:u64,
        chunk:&Buffer<bf16>,comm:&Roce,stream:&Stream)->Result<()> {
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_COPY_KV_QUEUED_SCREEN").as_deref()==Ok("1")
            &&(1..=32).contains(&accepted)&&[16*15360,32*15360,128*15360].contains(&features.len())
            &&accepted<=features.len()/15360&&self.cuda.owns_buffer(features)
            &&chunk.len()==6*15360&&self.cuda.owns_buffer(chunk)
            &&start==self.state.position()?&&comm.rank()==self.rank
            &&self.comm_instance.is_none_or(|c|c==comm.instance()),"queued copy KV admission");
        self.ready=false;self.last_anchor=None;self.markov.invalidate_results();
        let result=(|| {
            self.state.begin_copy_commit(start,accepted,stream)?;
            let conditioning=if self.conditioning_split {
                self.split_six.as_mut().ok_or_else(||anyhow::anyhow!("split conditioning absent"))?
            }else{&mut self.six};
            for offset in (0..accepted).step_by(6) {
                let live=(accepted-offset).min(6);
                unsafe {
                    self.cuda.zero_device(chunk.ptr(),chunk.bytes(),stream)?;
                    self.cuda.copy_device(chunk.ptr(),features.ptr()+(offset*15360*2)as u64,live*15360*2,stream)?;
                }
                if self.conditioning_graph {conditioning.run_cached_distributed(chunk,live,Some(comm),stream)?;}
                else{conditioning.run_distributed(chunk,live,Some(comm),stream)?;}
                let(_,raw,n)=conditioning.outputs()?;ensure!(n==live,"queued copy chunk extent");
                self.state.enqueue_copy_chunk(raw,live,&self.module,stream)?;
            }
            self.state.finish_copy_commit(stream)
        })();
        if result.is_err(){self.poisoned=true;self.state.invalidate();}result
    }
    /// Finite same-owner comparison including actual weighted norm/RoPE KV
    /// stores and their completion boundary. No target or sampler is executed.
    pub(crate) fn probe_copy_batched_kv(&mut self,features:&Buffer<bf16>,comm:&Roce,stream:&Stream,
        control:&mut crate::control_channel::ControlChannel)
        ->Result<serde_json::Value>{
        use serde_json::json;
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_COPY_KV_BATCH_SCREEN").as_deref()==Ok("1")
            &&features.len()==128*15360,"copy KV batch finite screen");
        let queued=std::env::var("CANADA_SCRATCH_COPY_KV_QUEUED_SCREEN").as_deref()==Ok("1");
        let short=std::env::var("CANADA_SCRATCH_COPY_KV_SHORT_SCREEN").as_deref()==Ok("1");
        ensure!(!short||queued,"short KV metadata screen requires queued stores");
        let original=self.save_state_image(4*1024*1024,comm,stream)?;
        let original_digest=original.digest()?;let start=self.position()?;
        let input=self.cuda.allocate::<bf16>(128*15360)?;
        let chunk=self.cuda.allocate::<bf16>(6*15360)?;
        let raw:[Buffer<bf16>;3]=[self.cuda.allocate(128*512)?,self.cuda.allocate(128*512)?,self.cuda.allocate(128*512)?];
        let mut barrier_input=self.cuda.allocate::<f32>(4)?;
        barrier_input.upload(&[0.;4])?;
        let mut barrier_output=self.cuda.allocate::<f32>(16)?;
        stream.join_default()?;
        let mut records=Vec::new();
        let cases=if short{&[(1usize,0usize),(3,0),(6,32),(7,0),(12,0),(16,0),(18,0),(32,32)][..]}
            else{&[(7usize,0usize),(12,0),(16,0),(18,0),(32,32)][..]};
        for &(accepted,offset) in cases {
            unsafe {
                self.cuda.zero_device(input.ptr(),input.bytes(),stream)?;
                self.cuda.copy_device(input.ptr(),features.ptr()+(offset*15360*2)as u64,
                    (128-offset)*15360*2,stream)?;
            }
            stream.synchronize()?;
            let mut expected=None;let mut trials=Vec::new();
            for (trial,batched) in [false,true,true,false,false,true,true,false,false,true].into_iter().enumerate() {
                self.restore_state_image(&original,comm,stream)?;
                control.agree(&[accepted as u8,offset as u8,trial as u8,u8::from(queued)])?;
                // TCP agreement returns to peers at different instants. Drain
                // one native collective before timing so its wakeup skew is
                // not charged to the first conditioning projection.
                // Raw gather does not inherit the target's active-row bounds,
                // unlike a BF16 model activation reduction on this communicator.
                unsafe{comm.gather_f32(&barrier_input,&mut barrier_output,4,stream)?;}
                stream.synchronize()?;
                let began=std::time::Instant::now();
                if batched&&queued {self.inject_copy_queued(&input,accepted,start,&chunk,comm,stream)?;}
                else if batched {self.inject_copy_batched(&input,accepted,start,&chunk,raw.each_ref(),comm,stream)?;}
                else {
                    for row in (0..accepted).step_by(6) {
                        let live=(accepted-row).min(6);
                        unsafe {
                            self.cuda.zero_device(chunk.ptr(),chunk.bytes(),stream)?;
                            self.cuda.copy_device(chunk.ptr(),input.ptr()+(row*15360*2)as u64,live*15360*2,stream)?;
                        }
                        self.inject_distributed(&chunk,live,live,start+row as u64,comm,stream)?;
                    }
                }
                stream.synchronize()?;let milliseconds=began.elapsed().as_secs_f64()*1000.;
                ensure!(self.position()?==start+accepted as u64,"copy KV batch frontier");
                let digest=self.save_state_image(4*1024*1024,comm,stream)?.digest()?;
                if let Some(previous)=&expected{ensure!(previous==&digest,"copy KV batching changed retained state");}
                else{expected=Some(digest.clone());}
                trials.push(json!({"warmup":trial<2,"batched":batched,"milliseconds":milliseconds,"digest":digest}));
            }
            // The input contains real future rows. Erasing them must leave
            // the accepted result identical, including a partial final chunk.
            self.restore_state_image(&original,comm,stream)?;
            unsafe{self.cuda.zero_device(input.ptr()+(accepted*15360*2)as u64,(128-accepted)*15360*2,stream)?;}
            if queued {self.inject_copy_queued(&input,accepted,start,&chunk,comm,stream)?;}
            else {self.inject_copy_batched(&input,accepted,start,&chunk,raw.each_ref(),comm,stream)?;}
            ensure!(Some(self.save_state_image(4*1024*1024,comm,stream)?.digest()?)==expected,
                "rejected future features affected copy KV");
            records.push(json!({"accepted":accepted,"feature_offset":offset,"trials":trials,"future_independent":true}));
        }
        self.restore_state_image(&original,comm,stream)?;
        ensure!(self.position()?==start&&self.save_state_image(4*1024*1024,comm,stream)?.digest()?==original_digest,
            "copy KV batch screen did not restore original state");
        Ok(json!({"rank":self.rank,"position":start,"records":records,"all_states_exact":true,
            "original_state_restored":true,"ordinary_conditioning_graph":self.conditioning_graph,
            "implementation":if queued{"queued_metadata_and_stores"}else{"staged_raw_kv"},
            "pretrial_peer_barrier":true,"pretrial_device_barrier":true}))
    }
    #[allow(clippy::too_many_arguments)]
    fn inject_inner(
        &mut self,features:&Buffer<bf16>,live:usize,accepted:usize,start:u64,comm:Option<&Roce>,
        copy_wide:bool,stream:&Stream,
    )->Result<()>{
        self.guard(stream)?;
        ensure!(
            start == self.state.position()?
                && live > 0
                && accepted <= live
                && self.cuda.owns_buffer(features)
                && [6 * 15360, 128 * 15360].contains(&features.len())
                && live <= features.len() / 15360,
            "DSpark feature commit"
        );
        self.ready = false;
        self.last_anchor = None;
        self.markov.invalidate_results();
        let result = (|| {
            let conditioning = if features.len() == 6 * 15360 {
                if self.conditioning_split {
                    self.split_six.as_mut().ok_or_else(||anyhow::anyhow!("split conditioning absent"))?
                }else{&mut self.six}
            } else {
                &mut self.wide
            };
            if copy_wide || (self.conditioning_graph && features.len()==6*15360) {
                conditioning.run_cached_distributed(features,live,comm,stream)?;
            }else{conditioning.run_distributed(features, live, comm, stream)?;}
            let (_, raw, n) = conditioning.outputs()?;
            if self.full_commit_device_metadata&&features.len()==6*15360&&n==6&&accepted==6 {
                // s1054: exact full-six-row store, with one GPU metadata write
                // replacing two host uploads. Keep the ordinary completion
                // boundary; wider transaction queuing did not show a gain.
                self.state.begin_copy_commit(start,accepted,stream)?;
                self.state.enqueue_copy_chunk(raw,accepted,&self.module,stream)?;
                self.state.finish_copy_commit(stream)
            }else{self.state.commit(raw,n,accepted,start,stream)}
        })();
        if result.is_err() {
            self.poisoned = true;
            self.state.invalidate();
        }
        result
    }
    pub(crate) fn inject_prefill_tail(
        &mut self,
        features: &Buffer<bf16>,
        live: usize,
        end: u64,
        stream: &Stream,
    ) -> Result<()> {
        self.guard(stream)?;
        ensure!(
            self.state.position()? == 0
                && features.len() == 128 * 15360
                && self.cuda.owns_buffer(features)
                && live > 0
                && live == end.min(128) as usize,
            "DSpark fresh prefill tail"
        );
        self.ready = false;
        self.last_anchor = None;
        self.markov.invalidate_results();
        let result = (|| {
            self.wide.run(features, live, stream)?;
            let (_, raw, n) = self.wide.outputs()?;
            self.state.seed_tail(raw, n, end, stream)
        })();
        if result.is_err() {
            self.poisoned = true;
            self.state.invalidate();
        }
        result
    }
    /// sample_from_anchor=true: one observed anchor and four noise embeddings,
    /// with five bidirectional output rows. This does not sample proposals yet.
    pub fn run(&mut self, anchor: u32, comm: &Roce, stream: &Stream) -> Result<()> {
        self.guard(stream)?;
        ensure!(
            anchor < 129280
                && comm.rank() == self.rank
                && self.comm_instance.is_none_or(|c| c == comm.instance()),
            "DSpark anchor or communicator changed"
        );
        self.ready = false;
        self.last_anchor = None;
        self.markov.invalidate_results();
        let result = (|| {
            let layout = self.state.prepare_query(stream)?;
            self.positions.update(&layout.positions)?;
            self.tokens.update(
                &[i64::from(anchor), 128799, 128799, 128799, 128799],
                QUERY_ROWS,
            )?;
            comm.select_decode_reduction(true, stream)?;
            comm.select_prefill_rows(QUERY_ROWS, QUERY_ROWS, stream)?;
            stream.join_default()?;
            if self.graph.is_none() {
                self.enqueue(comm, stream)?;
                stream.synchronize()?;
                comm.health()?;
                self.graph = Some(stream.capture(|| self.enqueue(comm, stream))?);
                self.comm_instance = Some(comm.instance());
            }
            unsafe { self.graph.as_ref().unwrap().launch(stream) }?;
            self.state.publish_queries(stream)?;
            comm.health()?;
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
            self.state.invalidate();
        } else {
            self.ready = true;
            self.last_anchor = Some(anchor);
        }
        result
    }
    fn enqueue(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        self.enqueue_body(comm,stream)?;
        if self.private_head_active {
            let head = self.private_head.as_ref().ok_or_else(||anyhow::anyhow!("private draft head absent"))?;
            self.io.run_private_draft_head(head, &self.normalized, stream)?;
        } else {
            self.io.run_normalized_head(&self.normalized, stream)?;
        }
        if self.gather_base_logits {
            self.io.gather_logits(comm, stream)?;
        }
        Ok(())
    }
    fn enqueue_body(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        self.io.run_input(&self.tokens, comm, stream)?;
        for stage in 0..3 {
            let (previous, current) = self.stages.split_at_mut(stage);
            let (residual, pre) = if let Some(previous) = previous.last() {
                (previous.output(), Some(previous.pre()))
            } else {
                (self.io.input(), None)
            };
            let (cache, candidates, locations) = self.state.query_parts(stage, stream)?;
            current[0].run(
                residual,
                pre,
                &self.positions,
                locations,
                cache,
                candidates,
                &self.tokens,
                &self.workspace,
                comm,
                stream,
            )?;
        }
        let last = self.stages.last().unwrap();
        let mut input = last.output().ptr();
        let mut pre = last.pre().ptr();
        let mut output = self.combined.ptr();
        let mut args = [
            (&mut input as *mut u64).cast(),
            (&mut pre as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
        ];
        unsafe {
            self.module.function("ca_final_hc_combine")?.launch(
                [QUERY_ROWS as u32, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )
        }?;
        self.norm.rms(
            &self.combined,
            &self.norm_weight,
            &mut self.normalized,
            QUERY_ROWS,
            5120,
            stream,
        )?;
        Ok(())
    }
    /// Generation consumes local FP32 base logits through Markov correction;
    /// full FP32 distributions are gathered there. The earlier base-logit
    /// all-gather is diagnostic-only and would transfer the vocabulary twice.
    pub(crate) fn use_local_base_logits(&mut self) -> Result<()> {
        ensure!(
            self.graph.is_none() && !self.ready && !self.poisoned,
            "DSpark base-logit policy is immutable after capture"
        );
        self.gather_base_logits = false;
        Ok(())
    }
    pub fn base_logits(&self) -> Result<&Buffer<f32>> {
        ensure!(
            self.ready && !self.poisoned && self.gather_base_logits,
            "DSpark logits unavailable"
        );
        Ok(self.io.logits())
    }
    /// Five sequential native Markov-corrected greedy proposals. Requires the
    /// current query's exact anchor and vocabulary pointers. No target state is
    /// accepted or advanced here; the verifier must decide the accepted prefix.
    pub fn greedy_proposals(&mut self, comm: &Roce, stream: &Stream) -> Result<[u32; 5]> {
        self.proposals(None, None, comm, stream)
    }
    pub(crate) fn stochastic_proposals(
        &mut self,
        seed: u64,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<[u32; 5]> {
        let position = self
            .position()?
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("draft position overflow"))?;
        self.proposals(Some((seed, position)), None, comm, stream)
    }
    pub(crate) fn score_history_for_probe(&mut self,copy:&[u32;5],comm:&Roce,stream:&Stream)
        ->Result<([f64;5],[f32;5],bool)> {
        self.guard(stream)?;
        ensure!(self.ready&&self.comm_instance==Some(comm.instance())&&comm.rank()==self.rank,
            "history score query owner");
        let anchor=self.last_anchor.ok_or_else(||anyhow::anyhow!("history score query anchor absent"))?;
        let result=self.markov.score_history_for_probe(self.io.local_logits(),&self.combined,anchor,copy,comm,stream);
        if result.is_err(){self.poisoned=true;self.ready=false;self.state.invalidate();}
        result
    }
    pub(crate) fn stochastic_proposals_causal_three(
        &mut self,seed:u64,threshold:f32,comm:&Roce,stream:&Stream,
    )->Result<[u32;5]>{
        let position=self.position()?.checked_add(1).ok_or_else(||anyhow::anyhow!("draft position overflow"))?;
        self.proposals(Some((seed,position)),Some(threshold),comm,stream)
    }
    pub(crate) fn set_prefix_stop_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()>{
        self.markov.set_prefix_stop_for_probe(enabled,stream)
    }
    pub(crate) fn set_markov_strength_for_probe(&mut self,strength:f32,stream:&Stream)->Result<()>{
        self.guard(stream)?;self.markov.set_strength_for_probe(strength,stream)
    }
    pub(crate) fn set_markov_cache_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()>{
        self.guard(stream)?;self.markov.set_correction_cache_for_probe(enabled,stream)
    }
    pub(crate) fn reset_markov_cache_counters_for_probe(&self,stream:&Stream)->Result<()>{
        self.guard(stream)?;self.markov.reset_cache_counters_for_probe(stream)
    }
    pub(crate) fn markov_cache_counters_for_probe(&self,stream:&Stream)->Result<[u64;2]>{
        self.guard(stream)?;self.markov.cache_counters_for_probe(stream)
    }
    pub(crate) fn hierarchical_active(&self)->bool{self.markov.hierarchical_active()}
    pub(crate) fn set_hierarchical_for_probe(&mut self,active:bool,stream:&Stream)->Result<()>{
        self.guard(stream)?;self.markov.set_hierarchical_for_probe(active,stream)
    }
    pub(crate) fn sharded_probabilities(&self,requested:usize)->Result<(&Buffer<f64>,[u64;5])>{
        self.markov.sharded_probabilities(requested)
    }
    pub(crate) fn produced_proposals(&self)->Result<usize>{self.markov.produced()}
    pub(crate) fn proposal_probabilities_prefix(&self,requested:usize)->Result<&Buffer<f64>>{
        self.markov.probabilities_prefix(requested)
    }
    pub(crate) fn proposal_probabilities(&self) -> Result<&Buffer<f64>> {
        self.markov.probabilities()
    }
    pub(crate) fn root_logits_for_feedback(&self)->Result<&Buffer<f32>> {
        self.markov.root_logits_for_feedback()
    }
    fn proposals(
        &mut self,
        sampling: Option<(u64, u64)>,
        causal_three:Option<f32>,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<[u32; 5]> {
        self.guard(stream)?;
        ensure!(
            self.ready && self.comm_instance == Some(comm.instance()) && comm.rank() == self.rank,
            "DSpark proposal owner"
        );
        let anchor=self.last_anchor.ok_or_else(||anyhow::anyhow!("DSpark query anchor missing"))?;
        let result = if let Some(threshold)=causal_three {
            let(seed,position)=sampling.ok_or_else(||anyhow::anyhow!("causal prefix requires sampled proposals"))?;
            self.markov.run_causal_three(self.io.local_logits(),&self.combined,anchor,seed,position,threshold,comm,stream)
        }else{self.markov.run(
            self.io.local_logits(),
            &self.combined,
            anchor,
            sampling,
            comm,
            stream,
        )};
        if result.is_err() {
            self.poisoned = true;
            self.ready = false;
            self.state.invalidate();
        }
        result
    }
    pub(crate) fn confidence_schedule(&self) -> Result<([f32; 5], usize)> {
        ensure!(
            self.ready && !self.poisoned && self.last_anchor.is_some(),
            "confidence query unavailable"
        );
        self.markov.confidence()
    }
    pub(crate) fn direct_markov_gather(&self)->bool { self.markov.direct_gather_enabled() }
    pub(crate) fn fused_markov_head(&self)->bool { self.markov.fused_head_enabled() }
    pub(crate) fn fused_confidence(&self)->bool {self.markov.fused_confidence()}
    pub(crate) fn set_fused_confidence_for_probe(&mut self,active:bool,stream:&Stream)->Result<()> {
        self.guard(stream)?;
        self.markov.set_fused_confidence_for_probe(active,stream)
    }
    pub(crate) fn set_fused_markov_head_for_probe(&mut self, active:bool, stream:&Stream)->Result<()> {
        self.guard(stream)?;
        self.markov.set_fused_head_for_probe(active,stream)
    }
    pub(crate) fn draft_temperature(&self) -> f32 { self.markov.temperature() }
    pub(crate) fn draft_schedule(&self)->usize {self.markov.draft_schedule()}
    pub(crate) fn draft_temperatures(&self)->[f32;5] {self.markov.temperatures()}
    pub(crate) fn set_draft_schedule(&mut self,schedule:usize,stream:&Stream)->Result<()> {
        self.markov.set_draft_schedule(schedule,stream)
    }
    pub(crate) fn set_draft_temperature(&mut self, temperature: f32, stream: &Stream) -> Result<()> {
        self.markov.set_temperature(temperature, stream)
    }
    pub(crate) fn cached_probability_factors(&self)->bool { self.markov.cached_probability_factors() }
    pub(crate) fn dump_proposals(&self, root: &Path) -> Result<serde_json::Value> {
        ensure!(
            self.ready && !self.poisoned,
            "DSpark proposal dump unavailable"
        );
        self.markov.dump(root)
    }
    pub(crate) fn probe_markov_gather(&mut self,comm:&Roce,stream:&Stream)->Result<serde_json::Value> {
        self.guard(stream)?;
        ensure!(self.ready && self.comm_instance==Some(comm.instance()),"Markov probe query state");
        let position=self.position()?.checked_add(1).ok_or_else(||anyhow::anyhow!("probe position overflow"))?;
        self.markov.probe_direct_gather(self.io.local_logits(),&self.combined,
            self.last_anchor.ok_or_else(||anyhow::anyhow!("probe anchor"))?,position,comm,stream)
    }
    pub(crate) fn probe_markov_prefix_stop(&mut self,comm:&Roce,stream:&Stream)->Result<serde_json::Value> {
        self.guard(stream)?;
        ensure!(self.ready && self.comm_instance==Some(comm.instance()),"Markov prefix probe query state");
        let position=self.position()?.checked_add(1).ok_or_else(||anyhow::anyhow!("probe position overflow"))?;
        self.markov.probe_prefix_stop(self.io.local_logits(),&self.combined,
            self.last_anchor.ok_or_else(||anyhow::anyhow!("probe anchor"))?,position,comm,stream)
    }
    pub fn position(&self) -> Result<u64> {
        ensure!(!self.poisoned, "DSpark network poisoned");
        self.state.position()
    }
    pub(crate) fn save_state_image(&mut self, budget: usize, comm: &Roce, stream: &Stream)
        -> Result<crate::dspark_kv::StateImage> {
        self.guard(stream)?;
        ensure!(comm.rank() == self.rank && self.comm_instance.is_none_or(|c|c==comm.instance()),
            "draft image needs its current communicator");
        comm.health()?;
        // A prompt checkpoint can precede the first neural query. Bind its
        // communicator here using the same initial-binding rule as run();
        // every later query, image save and restore must retain that identity.
        if self.comm_instance.is_none() {
            ensure!(self.graph.is_none(), "captured draft graph lost its communicator");
            self.comm_instance=Some(comm.instance());
        }
        let saved = self.state.save_state_image(budget, stream);
        if saved.as_ref().is_err_and(|e|e.downcast_ref::<crate::state_image::CapacityRefused>().is_none()) {
            self.poisoned = true;
        }
        saved
    }
    pub(crate) fn state_image_bytes(&self)->Result<usize> {self.state.state_image_bytes()}
    pub(crate) fn restore_state_image(&mut self, saved: &crate::dspark_kv::StateImage,
        comm: &Roce, stream: &Stream) -> Result<()> {
        self.guard(stream)?;
        ensure!(comm.rank() == self.rank && self.comm_instance == Some(comm.instance()),
            "draft image restore changed communicator");
        comm.health()?;
        self.poisoned = true;
        self.ready = false;
        self.last_anchor = None;
        self.markov.invalidate_results();
        self.state.restore_state_image(saved, stream)?;
        comm.health()?;
        self.poisoned = false;
        Ok(())
    }
    pub fn atlas_bytes(&self) -> usize {
        self.weights.bytes()
    }
    /// Diagnostic only: repeated query graph with unchanged inputs/history.
    /// Excludes Markov, acceptance and target work; not generation throughput.
    pub(crate) fn measure_current(
        &self,
        replays: usize,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<f32> {
        self.guard(stream)?;
        ensure!(
            self.ready
                && (1..=100).contains(&replays)
                && self.comm_instance == Some(comm.instance())
                && comm.rank() == self.rank,
            "DSpark measurement needs current output/communicator"
        );
        let begin = self.cuda.event()?;
        let end = self.cuda.event()?;
        begin.record(stream)?;
        for _ in 0..replays {
            unsafe { self.graph.as_ref().unwrap().launch(stream) }?;
        }
        end.record(stream)?;
        let ms = end.elapsed_since(&begin)? / replays as f32;
        comm.health()?;
        Ok(ms)
    }
    pub(crate) fn trace(&self, root: &Path) -> Result<serde_json::Value> {
        ensure!(
            self.ready && !self.poisoned && self.gather_base_logits && !root.exists(),
            "DSpark trace unavailable or immutable output exists"
        );
        std::fs::create_dir(root)?;
        let mut taps = vec![(
            "embedding".to_string(),
            self.io.input().ptr(),
            self.io.input().bytes(),
        )];
        for (i, s) in self.stages.iter().enumerate() {
            for (name, pointer, bytes, stride) in s.attention.taps() {
                ensure!(stride == 0, "DSpark query tap must be contiguous");
                taps.push((format!("s{i}_attention_{name}"), pointer, bytes));
            }
            for (name, b) in [
                ("attn_input", s.attn_boundary.input()),
                ("attn_reduced", &s.attn_reduced),
                ("attn_output", s.attn_boundary.output()),
                ("ffn_input", s.ffn_boundary.input()),
                ("ffn_local", s.ffn.output()),
                ("ffn_reduced", &s.ffn_reduced),
                ("output", s.output()),
                ("shared_up", s.ffn.shared_up()),
                ("activation", s.ffn.activation()),
                ("shared_down", s.ffn.shared_down()),
            ] {
                taps.push((format!("s{i}_{name}"), b.ptr(), b.bytes()));
            }
            for (name, b) in [
                ("attn_pre", s.attn_boundary.pre()),
                ("pre", s.pre()),
                ("router_logits", s.ffn.router().logits()),
                ("routing", s.ffn.router().routing()),
            ] {
                taps.push((format!("s{i}_{name}"), b.ptr(), b.bytes()));
            }
            let b = s.ffn.router().ids();
            taps.push((format!("s{i}_router_ids"), b.ptr(), b.bytes()));
        }
        for (name, pointer, bytes) in [
            ("combined", self.combined.ptr(), self.combined.bytes()),
            ("normalized", self.normalized.ptr(), self.normalized.bytes()),
            (
                "local_logits",
                self.io.local_logits().ptr(),
                self.io.local_logits().bytes(),
            ),
            (
                "base_logits",
                self.io.logits().ptr(),
                self.io.logits().bytes(),
            ),
        ] {
            taps.push((name.to_string(), pointer, bytes));
        }
        use sha2::{Digest, Sha256};
        let mut report = Vec::new();
        for (name, pointer, bytes) in taps {
            let raw = unsafe { self.cuda.read_device(pointer, bytes) }?;
            std::fs::write(root.join(format!("{name}.bin")), &raw)?;
            report.push(serde_json::json!({"name":name,"bytes":bytes,"sha256":format!("{:x}",Sha256::digest(raw))}));
        }
        let result = serde_json::json!({"rank":self.rank,"position":self.position()?,"rows":QUERY_ROWS,"taps":report});
        std::fs::write(root.join("index.json"), serde_json::to_vec_pretty(&result)?)?;
        Ok(result)
    }
    pub fn reset(&mut self, stream: &Stream) -> Result<()> {
        // Reset invalidates results and history, but does not recover a failed
        // transport or GPU context; poisoned owners require reconstruction.
        self.guard(stream)?;
        self.ready = false;
        self.last_anchor = None;
        self.markov.invalidate_results();
        self.state.reset(stream)
    }
}
