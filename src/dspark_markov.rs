//! Native checkpoint Markov correction and sequential GPU proposals.
//! Canonical BF16 operands, FP32 base/bias projections and FP32 correction,
//! following DeepSeek's minimal reference head precision. T1/topP1 proposals retain their actual GPU q for
//! target rejection sampling; greedy remains a separate captured path.
use crate::{
    bf16_linear::Projection,
    blas::Blas,
    categorical::{Categorical, DRAFT_DOMAIN, uniform},
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, Graph, Module, Stream},
    roce::Roce,
    sampling::{Greedy, VOCAB},
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use sha2::{Digest,Sha256};
use std::{path::Path, rc::Rc};
const DRAFT_TEMPERATURES: [f32;8] = [1.0,0.8,0.7,0.4,1.2,0.5,0.6,0.85];
const TEMPERATURE_GRAPH_STRIDE: usize = 4*DRAFT_TEMPERATURES.len();
const DRAFT_SCHEDULES: usize = 4;
fn temperature_at(schedule:usize,temperature:f32,step:usize)->f32 {
    match schedule {
        0=>temperature,
        1=>[0.7,0.7,0.5,0.5,0.4][step],
        2=>[0.7,0.85,1.0,1.2,1.2][step],
        3=>[1.0,0.7,0.7,0.7,0.7][step],
        _=>unreachable!("checked draft schedule"),
    }
}
fn proposal_graph_index(sampled: bool, direct: bool, temperature: usize, fused_head: bool, fused_confidence: bool, schedule:usize, correction_cache:bool, scaled:bool) -> usize {
    usize::from(sampled)+2*usize::from(sampled && direct)+4*if sampled {temperature}else{0}
        +TEMPERATURE_GRAPH_STRIDE*(usize::from(fused_head)+2*usize::from(fused_confidence))
        +TEMPERATURE_GRAPH_STRIDE*4*if sampled{schedule}else{0}
        +TEMPERATURE_GRAPH_STRIDE*4*DRAFT_SCHEDULES*usize::from(correction_cache)
        +TEMPERATURE_GRAPH_STRIDE*8*DRAFT_SCHEDULES*usize::from(scaled)
}
pub(crate) struct Markov {
    history_teacher_graph: Option<(usize,Graph)>,
    history_teacher_previous: Option<Buffer<i32>>,
    graphs: [Option<Graph>; TEMPERATURE_GRAPH_STRIDE*16*DRAFT_SCHEDULES],
    prefix_graphs: [Option<(Graph,Graph)>; TEMPERATURE_GRAPH_STRIDE*16*DRAFT_SCHEDULES],
    conditional_prefix_graphs: [Option<(u32,crate::markov_conditional::Conditional)>; TEMPERATURE_GRAPH_STRIDE*16*DRAFT_SCHEDULES],
    correction_cache:Option<crate::markov_cache::CorrectionCache>,
    correction_cache_active:bool,
    conditional_bridge: Option<Rc<crate::markov_conditional::Bridge>>,
    gpu_prefix_stop: bool,
    prefix_stop_enabled: bool,
    produced: usize,
    draft_schedule: usize,
    proposal_temperature: f32,
    temperature_index: usize,
    direct_gather: bool,
    categorical: Vec<Categorical>,
    hierarchical: Option<HierarchicalDraft>,
    hierarchical_active: bool,
    staged_inputs: Option<Rc<crate::cuda::MetadataStaging>>,
    proposal_probabilities: Buffer<f64>,
    sampled: bool,
    cuda: Cuda,
    module: Module,
    stream: usize,
    rank: usize,
    comm_instance: Option<u64>,
    base_pointer: Option<u64>,
    hidden_pointer: Option<u64>,
    weights: Rc<Weights>,
    confidence_blas: Rc<Blas>,
    confidence_weight: Buffer<f32>,
    confidence_features: Buffer<f32>,
    confidence_projection: Buffer<f32>,
    confidence_raw: Buffer<f32>,
    confidence_probability: Buffer<f32>,
    confidence_fused: Option<Module>,
    confidence_fused_active: bool,
    confidence_terms: Buffer<f32>,
    confidence_costs: Option<Buffer<f32>>,
    confidence_local_limit: Buffer<f32>,
    confidence_gathered_limit: Buffer<f32>,
    last_confidence: Option<([f32; 5], usize)>,
    anchor: Buffer<i32>,
    latent: Buffer<bf16>,
    errors: Vec<Buffer<u32>>,
    head: Projection<f32>,
    fused_head: Option<crate::triton_ops::Operator>,
    fused_head_active: bool,
    scaled_head: Option<crate::triton_ops::Operator>,
    scaled_head_active: bool,
    local: Buffer<f32>,
    gathered: Buffer<f32>,
    corrected: Buffer<f32>,
    readout: Buffer<u32>,
    conditional_result: Buffer<u32>,
    conditional_readout: Buffer<u32>,
    samplers: Vec<Greedy>,
    ready: bool,
    poisoned: bool,
}
struct HierarchicalDraft {
    draws: Vec<crate::hierarchical_categorical::Hierarchical>,
    q: Buffer<f64>,
}
impl Markov {
    fn flush_inputs(&self,stream:&Stream)->Result<()>{
        if let Some(staging)=&self.staged_inputs{unsafe{staging.flush(Some(stream))?;}}
        Ok(())
    }
    pub(crate) fn hierarchical_active(&self)->bool { self.hierarchical_active }
    pub(crate) fn set_hierarchical_for_probe(&mut self,active:bool,stream:&Stream)->Result<()> {
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_HIERARCHICAL_MODEL_SCREEN").as_deref()==Ok("1")
            &&self.hierarchical.is_some(),"isolated hierarchical draft control");
        if self.hierarchical_active!=active {
            stream.synchronize()?;
            // Captured kernels refer to either the full-vocab or sharded
            // sampler buffers. Never replay the other mode's graph.
            for g in &mut self.graphs {*g=None;}
            for g in &mut self.prefix_graphs {*g=None;}
            for g in &mut self.conditional_prefix_graphs {*g=None;}
            self.history_teacher_graph=None;
            self.hierarchical_active=active;
        }
        self.invalidate_results();Ok(())
    }
    pub(crate) fn sharded_probabilities(&self,requested:usize)->Result<(&Buffer<f64>,[u64;5])> {
        ensure!(self.hierarchical_active&&self.ready&&!self.poisoned&&self.sampled
            &&requested<=self.produced,"rank-local proposal probabilities unavailable");
        let h=self.hierarchical.as_ref().unwrap();
        Ok((&h.q,std::array::from_fn(|i|h.draws[i].selected_q().ptr())))
    }
    fn upload_proposal_uniforms(&mut self,seed:u64,position:u64)->Result<()> {
        ensure!(position.checked_add(5).is_some(),"draft RNG overflow");
        if self.hierarchical_active {
            for(step,draw)in self.hierarchical.as_mut().unwrap().draws.iter_mut().enumerate(){
                draw.upload_uniforms(seed,position+step as u64)?;
            }
        }else{
            for(step,draw)in self.categorical.iter_mut().enumerate(){
                draw.upload_uniforms(&[uniform(seed,DRAFT_DOMAIN,position+step as u64)])?;
            }
        }
        Ok(())
    }
    fn sampled_output(&self,step:usize)->u64 {
        if self.hierarchical_active{self.hierarchical.as_ref().unwrap().draws[step].output().ptr()}
        else{self.categorical[step].device_output().ptr()}
    }
    fn sampled_errors(&self,step:usize)->u64 {
        if self.hierarchical_active{self.hierarchical.as_ref().unwrap().draws[step].errors().ptr()}
        else{self.categorical[step].device_errors().ptr()}
    }

    pub(crate) fn set_strength_for_probe(&mut self,strength:f32,stream:&Stream)->Result<()>{
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_MARKOV_STRENGTH_SCREEN").as_deref()==Ok("1")
            &&[1.,0.75].contains(&strength)&&self.scaled_head.is_some()&&self.fused_head_active
            &&!self.correction_cache_active,"isolated Markov strength control");
        stream.synchronize()?;self.scaled_head_active=strength==0.75;self.invalidate_results();Ok(())
    }

    pub(crate) fn set_correction_cache_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()>{
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_MARKOV_CACHE_NATIVE_SCREEN").as_deref()==Ok("1")
            &&self.correction_cache.is_some()&&self.fused_head_active,"isolated Markov cache control");
        stream.synchronize()?;self.correction_cache_active=enabled;self.invalidate_results();Ok(())
    }
    pub(crate) fn reset_cache_counters_for_probe(&self,stream:&Stream)->Result<()>{
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_MARKOV_CACHE_NATIVE_SCREEN").as_deref()==Ok("1"),"isolated cache counters");
        self.correction_cache.as_ref().ok_or_else(||anyhow::anyhow!("cache absent"))?.reset_counters(stream)
    }
    pub(crate) fn cache_counters_for_probe(&self,stream:&Stream)->Result<[u64;2]>{
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_MARKOV_CACHE_NATIVE_SCREEN").as_deref()==Ok("1"),"isolated cache counters");
        self.correction_cache.as_ref().ok_or_else(||anyhow::anyhow!("cache absent"))?.counters(stream)
    }
    /// Captures belong to the loaded graph bank, not to a prompt snapshot.
    pub(crate) fn sampled_graph_ready(&self) -> bool {
        self.graphs[proposal_graph_index(true,self.direct_gather,self.temperature_index,
            self.fused_head_active,self.confidence_fused_active,self.draft_schedule,self.correction_cache_active,self.scaled_head_active)].is_some()
    }
    pub(crate) fn fused_confidence(&self)->bool {self.confidence_fused_active}
    pub(crate) fn set_fused_confidence_for_probe(&mut self,active:bool,stream:&Stream)->Result<()> {
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_CONFIDENCE_FUSION_SCREEN").as_deref()==Ok("1")
            && self.confidence_fused.is_some(),"isolated confidence switch requires the qualified module");
        stream.join_default()?;stream.synchronize()?;
        self.confidence_fused_active=active;self.invalidate_results();Ok(())
    }
    pub(crate) fn fused_head_enabled(&self)->bool { self.fused_head_active }
    pub(crate) fn set_fused_head_for_probe(&mut self, active:bool, stream:&Stream)->Result<()> {
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_MARKOV_HEAD_SCREEN").as_deref()==Ok("1")
            && self.fused_head.is_some(),"isolated Markov head switch requires both implementations");
        stream.join_default()?;stream.synchronize()?;
        self.fused_head_active=active;
        self.invalidate_results();
        Ok(())
    }
    pub(crate) fn temperature(&self) -> f32 { self.proposal_temperature }
    pub(crate) fn draft_schedule(&self)->usize {self.draft_schedule}
    pub(crate) fn set_draft_schedule(&mut self,schedule:usize,stream:&Stream)->Result<()> {
        self.guard(stream)?;
        ensure!(schedule<DRAFT_SCHEDULES
            && (std::env::var("CANADA_SCRATCH_REASONING_DRAFT_SCHEDULES").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_ROOT_TEMPERATURE_SCREEN").as_deref()==Ok("1")),
            "draft schedules require the isolated actual-q screen");
        stream.synchronize()?;self.draft_schedule=schedule;self.invalidate_results();Ok(())
    }
    pub(crate) fn temperatures(&self)->[f32;5] {
        std::array::from_fn(|step|temperature_at(self.draft_schedule,self.proposal_temperature,step))
    }
    pub(crate) fn set_temperature(&mut self, temperature: f32, stream: &Stream) -> Result<()> {
        self.guard(stream)?;
        // Each temperature owns its captured sampler/actual-q graph. Existing
        // temperature indices stay distinct from head/confidence graph variants.
        let index = DRAFT_TEMPERATURES.iter().position(|&t|t==temperature)
            .ok_or_else(||anyhow::anyhow!("unsupported draft temperature"))?;
        stream.synchronize()?;
        self.proposal_temperature=temperature;
        self.temperature_index=index;
        self.invalidate_results();
        Ok(())
    }
    pub(crate) fn invalidate_results(&mut self) {
        self.ready = false;
        self.last_confidence = None;
        self.produced = 0;
    }
    pub(crate) fn new(
        cuda: &Cuda,
        core: &[u8],
        stream: &Stream,
        rank: usize,
        weights: Rc<Weights>,
        kernels: &Path,
    ) -> Result<Self> {
        ensure!(
            cuda.owns_stream(stream) && weights.in_context(cuda) && rank < 4,
            "DSpark Markov context"
        );
        let staged=match std::env::var("CANADA_SCRATCH_MARKOV_STAGED_INPUTS"){
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid Markov input staging policy");value=="1"},
            Err(std::env::VarError::NotPresent)=>false,Err(error)=>return Err(error.into()),
        };
        // Only the CPU-written anchor and hierarchical uniforms use this
        // storage. Every run synchronizes before the caller can rewrite it.
        let controls=if staged{unsafe{cuda.staged_metadata_allocator()?}}else{cuda.clone()};
        for (name, shape) in [("embed", vec![129280, 256]), ("head", vec![32320, 256])] {
            let w = weights.view(&format!("mtp.2.markov_head.{name}.weight"))?;
            ensure!(
                w.dtype == Dtype::Bf16 && w.shape == shape,
                "DSpark Markov native geometry"
            );
        }
        stream.join_default()?;
        let blas = Rc::new(Blas::new_on(cuda,stream)?);
        let head = Projection::new(
            cuda,
            weights.clone(),
            blas.clone(),
            "mtp.2.markov_head.head",
            1,
        )?;
        let confidence = weights.view("mtp.2.confidence_head.proj.weight")?;
        ensure!(
            confidence.dtype == Dtype::Bf16 && confidence.shape == [1, 5376],
            "canonical confidence head"
        );
        let module = cuda.module(core)?;
        let confidence_weight = cuda.allocate::<f32>(5376)?;
        let mut source = weights.ptr("mtp.2.confidence_head.proj.weight")?;
        let mut destination = confidence_weight.ptr();
        let mut args = [
            (&mut source as *mut u64).cast(),
            (&mut destination as *mut u64).cast(),
        ];
        unsafe {
            module.function("ca_dspark_confidence_weight")?.launch(
                [21, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )
        }?;
        let confidence_costs =
            if std::env::var("CANADA_SCRATCH_COST_AWARE_VERIFY").as_deref() == Ok("1") {
                let calibration: serde_json::Value = serde_json::from_slice(include_bytes!(
                    "../tests/fixtures/verify-costs-s155-v1.json"
                ))?;
                ensure!(
                    calibration["schema"] == 1
                        && calibration["verified_rows"] == serde_json::json!([3, 4, 5, 6]),
                    "verification cost calibration schema"
                );
                let costs = calibration["complete_step_ms"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("missing measured costs"))?;
                ensure!(costs.len() == 4, "verification cost count");
                let costs = costs
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .map(|x| x as f32)
                            .filter(|x| x.is_finite() && *x > 0. && *x < 1000.)
                            .ok_or_else(|| anyhow::anyhow!("invalid measured cost"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut buffer = cuda.allocate(4)?;
                buffer.upload(&costs)?;
                Some(buffer)
            } else {
                None
            };
        let mut errors = Vec::new();
        let mut samplers = Vec::new();
        let mut categorical = Vec::new();
        for _ in 0..5 {
            errors.push(cuda.allocate::<u32>(1)?);
            samplers.push(Greedy::new(cuda, core, 1)?);
            categorical.push(Categorical::new(cuda, core, 1)?);
        }
        let direct_gather=match std::env::var("CANADA_SCRATCH_MARKOV_DIRECT_GATHER") {
            Ok(v)=>{ensure!(v=="0" || v=="1","invalid Markov gather policy");v=="1"}
            Err(std::env::VarError::NotPresent)=>true,
            Err(e)=>return Err(e.into()),
        };
        // Select once, before any graph capture. The cache hint changes neither
        // canonical BF16 operands nor the FP32 reduction/output boundary.
        let head_cg=match std::env::var("CANADA_SCRATCH_MARKOV_HEAD_CG") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid Markov head cache policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
        };
        let fused_head=match std::env::var("CANADA_SCRATCH_MARKOV_FUSED_HEAD") {
            Ok(value)if value=="1"=>{
                let root=if head_cg {"draft-markov-gemv-cg-v1"}else{"draft-markov-gemv-v1"};
                let op=crate::triton_ops::Operator::load(cuda,&kernels.join(root),"markov_head_m1")?;
                ensure!(op.entry.arguments.iter().map(|a|(a.name.as_str(),a.kind.as_str())).collect::<Vec<_>>()
                    ==[("X","*bf16"),("W","*bf16"),("Base","*fp32"),("Y","*fp32")]
                    && op.entry.grid.rows_div==16 && op.entry.grid.x_mul==1 && op.entry.grid.x_add==0
                    && op.entry.grid.y==1 && op.entry.grid.z==1 && op.entry.metadata.num_warps==4,
                    "fused Markov head canonical geometry/FP32 ABI");
                Some(op)
            },
            Err(std::env::VarError::NotPresent)=>None,
            Ok(value)if value=="0"=>None,
            _=>anyhow::bail!("invalid fused Markov head policy"),
        };
        ensure!(!head_cg||fused_head.is_some(),"Markov cache policy requires fused head");
        let scaled_head=if std::env::var("CANADA_SCRATCH_MARKOV_STRENGTH_SCREEN").as_deref()==Ok("1") {
            ensure!(fused_head.is_some(),"Markov strength requires the fused head");
            let root=kernels.join("draft-markov-scaled-v1");
            let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
            ensure!(manifest["entries"][0]["constexprs"]==serde_json::json!({"N":32320,"BN":16,"STRENGTH":0.75}),
                "Markov strength geometry");
            let op=crate::triton_ops::Operator::load(cuda,&root,"markov_head_scaled_m1")?;
            ensure!(op.entry.arguments.iter().map(|a|(a.name.as_str(),a.kind.as_str())).collect::<Vec<_>>()
                ==[("X","*bf16"),("W","*bf16"),("Base","*fp32"),("Y","*fp32")]
                &&op.entry.grid.rows_div==16&&op.entry.grid.x_mul==1&&op.entry.grid.x_add==0
                &&op.entry.grid.y==1&&op.entry.grid.z==1&&op.entry.metadata.num_warps==4,
                "Markov strength ABI");Some(op)
        }else{None};
        let correction_cache=match std::env::var("CANADA_SCRATCH_MARKOV_CORRECTION_CACHE") {
            Ok(v)if v=="1"=>{ensure!(fused_head.is_some(),"Markov cache requires fused head");
                Some(crate::markov_cache::CorrectionCache::new(cuda,kernels,stream)?)},
            Ok(v)if v=="0"=>None,Err(std::env::VarError::NotPresent)=>None,
            _=>anyhow::bail!("invalid Markov correction cache policy"),
        };
        let confidence_fused=match std::env::var("CANADA_SCRATCH_FUSED_CONFIDENCE") {
            Ok(value)if value=="1"=>{
                let root=kernels.join("confidence-fusion-v1");
                let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
                ensure!(manifest["schema"]==1 && manifest["architecture"]==121 && manifest["block_threads"]==256
                    && manifest["functions"]==serde_json::json!([
                        {"name":"ca_confidence_hidden_terms","pointer_arguments":4,"int_arguments":0},
                        {"name":"ca_confidence_fused_latent","pointer_arguments":9,"int_arguments":1}]),
                    "confidence fusion CUDA ABI contract");
                let bytes=std::fs::read(root.join("confidence.cubin"))?;
                ensure!(format!("{:x}",Sha256::digest(&bytes))==manifest["artifact_sha256"].as_str().unwrap_or(""),
                    "confidence fusion artifact hash");
                Some(cuda.module(&bytes)?)
            },
            Err(std::env::VarError::NotPresent)=>None,Ok(value)if value=="0"=>None,
            _=>anyhow::bail!("invalid fused confidence policy"),
        };
        Ok(Self {
            history_teacher_graph:None,
            history_teacher_previous:if std::env::var("CANADA_SCRATCH_HISTORY_CONFIDENCE_SCREEN").as_deref()==Ok("1") {
                Some(cuda.allocate(5)?)
            }else{None},
            graphs: std::array::from_fn(|_|None),
            prefix_graphs: std::array::from_fn(|_|None),
            conditional_prefix_graphs: std::array::from_fn(|_|None),
            correction_cache_active:correction_cache.is_some(),
            correction_cache,
            conditional_bridge:None,
            gpu_prefix_stop:match std::env::var("CANADA_SCRATCH_PREFIX_DRAFT_STOP_GPU") {
                Ok(v)=>{ensure!(v=="0"||v=="1","invalid GPU prefix stop policy");v=="1"},
                Err(std::env::VarError::NotPresent)=>false,
                Err(e)=>return Err(e.into()),
            },
            prefix_stop_enabled:match std::env::var("CANADA_SCRATCH_PREFIX_DRAFT_STOP") {
                Ok(v)=>{ensure!(v=="0"||v=="1","invalid prefix draft stop policy");v=="1"},
                Err(std::env::VarError::NotPresent)=>false,
                Err(e)=>return Err(e.into()),
            },
            produced:0,
            proposal_temperature: 1.,
            draft_schedule: 0,
            temperature_index: 0,
            direct_gather,
            scaled_head,scaled_head_active:false,
            fused_head_active:fused_head.is_some(),
            fused_head,
            confidence_fused_active:confidence_fused.is_some(),
            confidence_fused,
            confidence_terms:cuda.allocate(5)?,
            categorical,
            hierarchical:if crate::sampler_transport::Policy::from_env()?.hierarchical_draft
                ||std::env::var("CANADA_SCRATCH_HIERARCHICAL_MODEL_SCREEN").as_deref()==Ok("1") {
                Some(HierarchicalDraft{draws:(0..5).map(|_|crate::hierarchical_categorical::Hierarchical::with_uniform_allocator(cuda,core,rank,&controls)).collect::<Result<Vec<_>>>()?,
                    q:cuda.allocate(5*32320)?})
            }else{None},
            hierarchical_active:crate::sampler_transport::Policy::from_env()?.hierarchical_draft,
            staged_inputs:controls.metadata_staging(),
            proposal_probabilities: cuda.allocate(5 * VOCAB)?,
            sampled: false,
            cuda: cuda.clone(),
            module,
            stream: stream.raw() as usize,
            rank,
            comm_instance: None,
            base_pointer: None,
            hidden_pointer: None,
            confidence_blas: blas,
            confidence_weight,
            confidence_features: cuda.allocate(5 * 5376)?,
            confidence_projection: cuda.allocate(1)?,
            confidence_raw: cuda.allocate(5)?,
            confidence_probability: cuda.allocate(5)?,
            confidence_costs,
            confidence_local_limit: cuda.allocate(4)?,
            confidence_gathered_limit: cuda.allocate(16)?,
            last_confidence: None,
            weights,
            anchor: controls.allocate(1)?,
            latent: cuda.allocate(256)?,
            errors,
            head,
            local: cuda.allocate(32320)?,
            gathered: cuda.allocate(VOCAB)?,
            corrected: cuda.allocate(5 * VOCAB)?,
            readout: cuda.allocate(32)?,
            conditional_result:cuda.allocate(4)?,
            conditional_readout:cuda.allocate(36)?,
            samplers,
            ready: false,
            poisoned: false,
        })
    }
    fn guard(&self, stream: &Stream) -> Result<()> {
        ensure!(
            !self.poisoned && self.cuda.owns_stream(stream) && self.stream == stream.raw() as usize,
            "DSpark Markov poisoned/foreign stream"
        );
        Ok(())
    }
    pub(crate) fn run(
        &mut self,
        base: &Buffer<f32>,
        post_hc: &Buffer<bf16>,
        anchor: u32,
        sampling: Option<(u64, u64)>,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<[u32; 5]> {
        self.guard(stream)?;
        ensure!(
            base.len() == 5 * 32320
                && self.cuda.owns_buffer(base)
                && post_hc.len() == 5 * 5120
                && self.cuda.owns_buffer(post_hc)
                && anchor < VOCAB as u32
                && comm.rank() == self.rank
                && self.comm_instance.is_none_or(|i| i == comm.instance())
                && self.base_pointer.is_none_or(|p| p == base.ptr())
                && self.hidden_pointer.is_none_or(|p| p == post_hc.ptr()),
            "DSpark Markov input/collective identity"
        );
        self.ready = false;
        self.last_confidence = None;
        self.produced = 0;
        let result = (|| {
            self.anchor.upload(&[anchor as i32])?;
            if let Some((seed, position)) = sampling {
                ensure!(
                    position.checked_add(5).is_some(),
                    "draft sampling position overflow"
                );
                self.upload_proposal_uniforms(seed,position)?;
            }
            let sampled = sampling.is_some();
            let graph_index = proposal_graph_index(sampled,self.direct_gather,self.temperature_index,
                self.fused_head_active,self.confidence_fused_active,self.draft_schedule,self.correction_cache_active,self.scaled_head_active);
            self.sampled = sampled;
            stream.join_default()?;
            self.flush_inputs(stream)?;
            if self.graphs[graph_index].is_none() {
                self.enqueue(base, post_hc, sampled, comm, stream)?;
                stream.synchronize()?;
                comm.health()?;
                self.graphs[graph_index] =
                    Some(stream.capture(|| self.enqueue(base, post_hc, sampled, comm, stream))?);
                self.comm_instance = Some(comm.instance());
                self.base_pointer = Some(base.ptr());
                self.hidden_pointer = Some(post_hc.ptr());
            }
            unsafe { self.graphs[graph_index].as_ref().unwrap().launch(stream) }?;
            stream.synchronize()?;
            comm.health()?;
            self.read_full_result()
        })();
        if result.is_err() {
            self.poisoned = true;
        } else {
            self.ready = true;
            self.produced = 5;
        }
        result
    }
    fn read_full_result(&mut self)->Result<[u32;5]>{
        let words = self.readout.download()?;
        self.read_full_words(&words)
    }
    /// Finite opportunity screen only. Each Markov row is conditioned on the
    /// known copy predecessor, never a sampled proposal. Discard sampled IDs;
    /// these q rows must not become a stochastic proposal sequence.
    pub(crate) fn score_history_for_probe(&mut self,base:&Buffer<f32>,post_hc:&Buffer<bf16>,
        anchor:u32,copy:&[u32;5],comm:&Roce,stream:&Stream)->Result<([f64;5],[f32;5],bool)> {
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_HISTORY_CONFIDENCE_SCREEN").as_deref()==Ok("1")
            &&self.history_teacher_previous.is_some()&&!self.correction_cache_active
            &&base.len()==5*32320&&self.cuda.owns_buffer(base)
            &&post_hc.len()==5*5120&&self.cuda.owns_buffer(post_hc)
            &&anchor<VOCAB as u32&&copy.iter().all(|&id|id<VOCAB as u32)
            &&comm.rank()==self.rank&&self.comm_instance==Some(comm.instance())
            &&self.base_pointer==Some(base.ptr())&&self.hidden_pointer==Some(post_hc.ptr()),
            "history scoring requires an isolated warmed Markov query");
        self.invalidate_results();
        let result=(||{
            let previous=[anchor as i32,copy[0]as i32,copy[1]as i32,copy[2]as i32,copy[3]as i32];
            self.history_teacher_previous.as_mut().unwrap().upload(&previous)?;
            // Normalization shares the categorical implementation. Draws are
            // unused and do not supply any later row's conditioning token.
            for sampler in &mut self.categorical{sampler.upload_uniforms(&[0.5])?;}
            stream.join_default()?;
            let index=proposal_graph_index(true,self.direct_gather,self.temperature_index,
                self.fused_head_active,self.confidence_fused_active,self.draft_schedule,self.correction_cache_active,self.scaled_head_active);
            let first_capture=self.history_teacher_graph.is_none();
            if first_capture{
                let previous=self.history_teacher_previous.as_ref().unwrap().ptr();
                self.enqueue_segment_previous(base,post_hc,true,comm,stream,0,false,Some(previous))?;
                stream.synchronize()?;comm.health()?;
                let graph=stream.capture(||self.enqueue_segment_previous(base,post_hc,true,comm,stream,0,false,Some(previous)))?;
                self.history_teacher_graph=Some((index,graph));
            }
            let(saved_index,graph)=self.history_teacher_graph.as_ref().unwrap();
            ensure!(*saved_index==index,"history score configuration changed after capture");
            unsafe{graph.launch(stream)?;}stream.synchronize()?;comm.health()?;
            self.read_full_result()?;
            let confidence=self.last_confidence.unwrap().0;
            let mut q=[0.;5];
            for i in 0..5{
                let bytes=unsafe{self.cuda.read_device(self.proposal_probabilities.ptr()+((i*VOCAB+copy[i]as usize)*8)as u64,8)}?;
                q[i]=f64::from_le_bytes(bytes.try_into().unwrap());
                ensure!(q[i].is_finite()&&(0. ..=1.).contains(&q[i]),"invalid history conditional probability");
            }
            Ok((q,confidence,first_capture))
        })();
        self.invalidate_results();
        if result.is_err(){self.poisoned=true;}
        result
    }
    fn read_full_words(&mut self,words:&[u32])->Result<[u32;5]>{
        ensure!(words.len()==32,"full Markov readout size");
        ensure!(
            words[15] == 0x4453504b && words[5..15].iter().all(|&v| v == 0),
            "DSpark proposal/preceding-token validation failed"
        );
        ensure!(
            words[..5].iter().all(|&id| id < VOCAB as u32),
            "DSpark proposal outside vocabulary"
        );
        let confidence: [f32; 5] = std::array::from_fn(|i| f32::from_bits(words[21 + i]));
        ensure!(
            words[31] == 0x434f4e46
                && words[27..31].iter().all(|&v| v == 0)
                && (2..=6).contains(&words[26])
                && confidence
                    .iter()
                    .all(|p| p.is_finite() && *p >= 0. && *p <= 1.),
            "confidence scheduling/readout fault"
        );
        self.last_confidence = Some((confidence, words[26] as usize));
        ensure!(
            words[16..21].iter().all(|&v| f32::from_bits(v).is_finite()),
            "nonfinite confidence projection"
        );
        Ok(words[..5].try_into().unwrap())
    }
    pub(crate) fn probabilities(&self) -> Result<&Buffer<f64>> {
        self.probabilities_prefix(5)
    }
    pub(crate) fn root_logits_for_feedback(&self)->Result<&Buffer<f32>> {
        ensure!(std::env::var("CANADA_SCRATCH_ROOT_FEEDBACK_SCREEN").as_deref()==Ok("1")
            && !self.hierarchical_active && self.ready && !self.poisoned && self.sampled && self.produced>0,
            "completed root draft required for feedback");
        Ok(&self.corrected)
    }
    pub(crate) fn probabilities_prefix(&self,requested:usize) -> Result<&Buffer<f64>> {
        ensure!(
            !self.hierarchical_active && self.ready && !self.poisoned && self.sampled && requested<=self.produced,
            "stochastic draft probabilities unavailable"
        );
        Ok(&self.proposal_probabilities)
    }
    pub(crate) fn produced(&self)->Result<usize>{
        ensure!(self.ready&&!self.poisoned,"draft count unavailable");Ok(self.produced)
    }
    pub(crate) fn set_prefix_stop_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()>{
        self.guard(stream)?;
        ensure!(std::env::var("CANADA_SCRATCH_NATIVE_PREFIX_STOP_SCREEN").as_deref()==Ok("1"),
            "native prefix switch requires isolated admission");
        stream.synchronize()?;self.prefix_stop_enabled=enabled;self.invalidate_results();Ok(())
    }
    /// Same proposal law as run(), but stop before proposal3 when the existing
    /// causal-three verifier will discard it. Only the first produced rows of q
    /// are readable; full-q diagnostic access rejects a partial result.
    pub(crate) fn run_causal_three(
        &mut self,base:&Buffer<f32>,post_hc:&Buffer<bf16>,anchor:u32,
        seed:u64,position:u64,threshold:f32,comm:&Roce,stream:&Stream,
    )->Result<[u32;5]>{
        if !self.prefix_stop_enabled{return self.run(base,post_hc,anchor,Some((seed,position)),comm,stream);}
        self.guard(stream)?;
        ensure!(threshold.is_finite()&&threshold>=0.&&threshold<=1.
            &&base.len()==5*32320&&self.cuda.owns_buffer(base)
            &&post_hc.len()==5*5120&&self.cuda.owns_buffer(post_hc)
            &&anchor<VOCAB as u32&&position.checked_add(5).is_some()
            &&comm.rank()==self.rank&&self.comm_instance.is_none_or(|i|i==comm.instance())
            &&self.base_pointer.is_none_or(|p|p==base.ptr())
            &&self.hidden_pointer.is_none_or(|p|p==post_hc.ptr()),"causal Markov input/identity");
        self.invalidate_results();
        let result=(||{
            if self.gpu_prefix_stop {
                return self.run_causal_three_gpu(base,post_hc,anchor,seed,position,threshold,comm,stream);
            }
            let index=proposal_graph_index(true,self.direct_gather,self.temperature_index,
                self.fused_head_active,self.confidence_fused_active,self.draft_schedule,self.correction_cache_active,self.scaled_head_active);
            if self.prefix_graphs[index].is_none(){
                // Warm every kernel and bind the same immutable graph inputs.
                // Captures survive prompt reset exactly like the full graph.
                self.run(base,post_hc,anchor,Some((seed,position)),comm,stream)?;
                let first=stream.capture(||self.enqueue_segment(base,post_hc,true,comm,stream,0,true))?;
                let rest=stream.capture(||self.enqueue_segment(base,post_hc,true,comm,stream,2,false))?;
                self.prefix_graphs[index]=Some((first,rest));
            }
            self.invalidate_results();self.sampled=true;
            self.anchor.upload(&[anchor as i32])?;
            self.upload_proposal_uniforms(seed,position)?;
            stream.join_default()?;
            self.flush_inputs(stream)?;
            unsafe{self.prefix_graphs[index].as_ref().unwrap().0.launch(stream)}?;
            stream.synchronize()?;comm.health()?;
            let words=self.readout.download()?;
            ensure!(words[15]==0x4453504b&&words[5..15].iter().all(|&x|x==0)
                &&words[..2].iter().all(|&id|id<VOCAB as u32),"causal Markov prefix ID/error");
            let mut confidence=[0.;5];
            for i in 0..3{
                confidence[i]=f32::from_bits(words[27+i]);
                ensure!(words[21+i]==words[27+i]&&confidence[i].is_finite()
                    &&confidence[i]>=0.&&confidence[i]<=1.&&f32::from_bits(words[16+i]).is_finite(),
                    "causal confidence disagrees with rank0/invalid");
            }
            if confidence[..3].iter().product::<f32>()<threshold{
                self.last_confidence=Some((confidence,3));self.produced=2;
                // These placeholders are never verified or published. The
                // produced bound guards q and determines diagnostic ID count.
                Ok([words[0],words[1],0,0,0])
            }else{
                unsafe{self.prefix_graphs[index].as_ref().unwrap().1.launch(stream)}?;
                stream.synchronize()?;comm.health()?;
                let ids=self.read_full_result()?;self.produced=5;Ok(ids)
            }
        })();
        if result.is_err(){self.poisoned=true;self.invalidate_results();}else{self.ready=true;}
        result
    }
    /// The same causal decision stays on the device; one final readout reports
    /// whether two or five proposals were produced. No intermediate host sync.
    #[allow(clippy::too_many_arguments)]
    fn run_causal_three_gpu(&mut self,base:&Buffer<f32>,post_hc:&Buffer<bf16>,anchor:u32,
        seed:u64,position:u64,threshold:f32,comm:&Roce,stream:&Stream)->Result<[u32;5]>{
        let index=proposal_graph_index(true,self.direct_gather,self.temperature_index,
            self.fused_head_active,self.confidence_fused_active,self.draft_schedule,self.correction_cache_active,self.scaled_head_active);
        if !self.conditional_prefix_graphs[index].as_ref().is_some_and(|(bits,_)|*bits==threshold.to_bits()) {
            self.run(base,post_hc,anchor,Some((seed,position)),comm,stream)?;
            if self.conditional_bridge.is_none(){
                let path=std::env::var("CANADA_SCRATCH_MARKOV_CONDITIONAL_LIBRARY")?;
                self.conditional_bridge=Some(crate::markov_conditional::Bridge::open(&self.cuda,Path::new(&path))?);
            }
            let bridge=self.conditional_bridge.as_ref().unwrap().clone();
            let first=bridge.capture(stream,||self.enqueue_segment(base,post_hc,true,comm,stream,0,true))?;
            let rest=bridge.capture(stream,||self.enqueue_segment(base,post_hc,true,comm,stream,2,false))?;
            let graph=unsafe{bridge.graph_readout(&first,&rest,&self.confidence_gathered_limit,
                &mut self.conditional_result,threshold,&self.readout,&mut self.conditional_readout,stream)}?;
            self.conditional_prefix_graphs[index]=Some((threshold.to_bits(),graph));
        }
        self.invalidate_results();self.sampled=true;self.anchor.upload(&[anchor as i32])?;
        self.upload_proposal_uniforms(seed,position)?;
        stream.join_default()?;
        self.flush_inputs(stream)?;
        unsafe{self.conditional_prefix_graphs[index].as_ref().unwrap().1.launch(stream)}?;
        stream.synchronize()?;comm.health()?;
        let words=self.conditional_readout.download()?;
        ensure!(words.len()==36&&words[32]<=1&&words[33]==0&&words[34]<=1
            &&words[32]+words[34]==1&&words[35]==0x4d434f4e,"GPU prefix decision readout");
        if words[32]==1 {
            let ids=self.read_full_words(&words[..32])?;
            ensure!(self.last_confidence.unwrap().0[..3].iter().product::<f32>()>=threshold,
                "GPU continue disagrees with causal verifier");
            self.produced=5;Ok(ids)
        }else{
            ensure!(words[15]==0x4453504b&&words[5..15].iter().all(|&v|v==0)
                &&words[..2].iter().all(|&id|id<VOCAB as u32),"GPU prefix ID/error");
            let mut confidence=[0.;5];
            for i in 0..3 {
                confidence[i]=f32::from_bits(words[27+i]);
                ensure!(words[21+i]==words[27+i]&&confidence[i].is_finite()
                    &&(0. ..=1.).contains(&confidence[i])&&f32::from_bits(words[16+i]).is_finite(),
                    "GPU prefix confidence disagrees with rank0/invalid");
            }
            ensure!(confidence[..3].iter().product::<f32>()<threshold,"GPU stop disagrees with causal verifier");
            self.last_confidence=Some((confidence,3));self.produced=2;
            Ok([words[0],words[1],0,0,0])
        }
    }
    fn enqueue(
        &mut self,
        base: &Buffer<f32>,
        post_hc: &Buffer<bf16>,
        sampled: bool,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        self.enqueue_segment(base, post_hc, sampled, comm, stream, 0, false)
    }
    /// The split path is admitted only by the component probe. Its first
    /// segment stops after confidence3, before drawing proposal3: that
    /// confidence depends only on the two proposals included in three rows.
    fn enqueue_segment(
        &mut self, base: &Buffer<f32>, post_hc: &Buffer<bf16>, sampled: bool,
        comm: &Roce, stream: &Stream, first: usize, prefix_only: bool,
    ) -> Result<()> {
        self.enqueue_segment_previous(base,post_hc,sampled,comm,stream,first,prefix_only,None)
    }
    fn enqueue_segment_previous(
        &mut self, base: &Buffer<f32>, post_hc: &Buffer<bf16>, sampled: bool,
        comm: &Roce, stream: &Stream, first: usize, prefix_only: bool, previous_override:Option<u64>,
    ) -> Result<()> {
        ensure!(!self.hierarchical_active||previous_override.is_none(),"teacher q requires full-vocabulary sampler");
        ensure!((first == 0 || first == 2) && (!prefix_only || first == 0),
            "Markov segment geometry");
        if (first != 0 || prefix_only) && !self.prefix_stop_enabled &&
            std::env::var("CANADA_SCRATCH_MARKOV_PREFIX_STOP_SCREEN").as_deref()!=Ok("1") {
            anyhow::bail!("Markov split requires isolated component admission");
        }
        if first == 0 && self.confidence_fused_active {
            let mut words=[post_hc.ptr(),self.confidence_weight.ptr(),self.confidence_features.ptr(),self.confidence_terms.ptr()];
            let mut args:Vec<*mut std::ffi::c_void>=words.iter_mut().map(|p|(p as *mut u64).cast()).collect();
            unsafe{self.confidence_fused.as_ref().unwrap().function("ca_confidence_hidden_terms")?
                .launch([5,1,1],[256,1,1],0,stream,&mut args)?;}
        }
        for step in first..5 {
            let mut previous = if let Some(previous)=previous_override {
                previous+(step*4)as u64
            } else if step == 0 {
                self.anchor.ptr()
            } else if sampled {
                self.sampled_output(step-1)
            } else {
                self.samplers[step - 1].device_output()?.ptr()
            };
            let mut weight = self.weights.ptr("mtp.2.markov_head.embed.weight")?;
            let mut latent = self.latent.ptr();
            let mut error = self.errors[step].ptr();
            if self.confidence_fused_active {
                let mut words=[previous,weight,self.confidence_weight.ptr(),self.confidence_terms.ptr(),latent,
                    self.confidence_features.ptr(),self.confidence_raw.ptr(),self.confidence_probability.ptr(),error];
                let mut index=step as i32;
                let mut args:Vec<*mut std::ffi::c_void>=words.iter_mut().map(|p|(p as *mut u64).cast()).collect();
                args.push((&mut index as *mut i32).cast());
                unsafe{self.confidence_fused.as_ref().unwrap().function("ca_confidence_fused_latent")?
                    .launch([1,1,1],[256,1,1],0,stream,&mut args)?;}
            }else{
            let mut args = [
                (&mut previous as *mut u64).cast(),
                (&mut weight as *mut u64).cast(),
                (&mut latent as *mut u64).cast(),
                (&mut error as *mut u64).cast(),
            ];
            unsafe {
                self.module.function("ca_dspark_markov_embed")?.launch(
                    [1, 1, 1],
                    [256, 1, 1],
                    0,
                    stream,
                    &mut args,
                )
            }?;
            let mut hidden = post_hc.ptr();
            let mut features = self.confidence_features.ptr();
            let mut index = step as i32;
            let mut args = [
                (&mut hidden as *mut u64).cast(),
                (&mut latent as *mut u64).cast(),
                (&mut features as *mut u64).cast(),
                (&mut index as *mut i32).cast(),
            ];
            unsafe {
                self.module
                    .function("ca_dspark_confidence_features")?
                    .launch([21, 1, 1], [256, 1, 1], 0, stream, &mut args)
            }?;
            self.confidence_blas.confidence_linear(
                &self.confidence_features,
                &self.confidence_weight,
                &self.confidence_projection,
                step,
                stream,
            )?;
            let mut projection = self.confidence_projection.ptr();
            let mut raw = self.confidence_raw.ptr();
            let mut probability = self.confidence_probability.ptr();
            let mut args = [
                (&mut projection as *mut u64).cast(),
                (&mut raw as *mut u64).cast(),
                (&mut probability as *mut u64).cast(),
                (&mut index as *mut i32).cast(),
            ];
            unsafe {
                self.module
                    .function("ca_dspark_confidence_probability")?
                    .launch([1, 1, 1], [32, 1, 1], 0, stream, &mut args)
            }?;
            }
            if prefix_only && step == 2 { break; }
            if self.fused_head_active {
                if self.correction_cache_active {
                    unsafe{self.correction_cache.as_ref().unwrap().enqueue(self.latent.ptr(),
                        self.weights.ptr("mtp.2.markov_head.head.weight")?,base.ptr()+(step*32320*4)as u64,
                        previous,self.local.ptr(),stream)?;}
                }else{
                let op=if self.scaled_head_active{self.scaled_head.as_ref().expect("scaled head loaded")}
                    else{self.fused_head.as_ref().expect("active fused head is loaded")};
                // Draft-only numerical path. Sampling and exported actual q
                // both consume these same FP32 corrected logits below.
                unsafe{op.launch(32320,&[self.latent.ptr(),self.weights.ptr("mtp.2.markov_head.head.weight")?,
                    base.ptr()+(step*32320*4)as u64,self.local.ptr()],stream)?;}
                }
            }else{
            self.head.run(&self.latent, stream)?;
            let mut input = base.ptr();
            let mut bias = self.head.output().ptr();
            let mut output = self.local.ptr();
            let mut index = step as i32;
            let mut args = [
                (&mut input as *mut u64).cast(),
                (&mut bias as *mut u64).cast(),
                (&mut output as *mut u64).cast(),
                (&mut index as *mut i32).cast(),
            ];
            unsafe {
                self.module.function("ca_dspark_markov_local")?.launch(
                    [32320u32.div_ceil(256), 1, 1],
                    [256, 1, 1],
                    0,
                    stream,
                    &mut args,
                )
            }?;
            }
            if sampled&&self.hierarchical_active {
                let temperature=temperature_at(self.draft_schedule,self.proposal_temperature,step);
                let h=self.hierarchical.as_mut().unwrap();
                h.draws[step].enqueue(&self.local,0,&h.q,step,temperature,comm,stream)?;
                continue;
            }
            let direct = sampled && self.direct_gather;
            if direct {
                unsafe { comm.gather_f32_at(&self.local,&mut self.corrected,step*VOCAB,32320,stream) }?;
            } else {
                unsafe { comm.gather_f32(&self.local, &mut self.gathered, 32320, stream) }?;
            }
            if sampled {
                let (logits,first)=if direct {(&self.corrected,step)}else{(&self.gathered,0)};
                let temperature=temperature_at(self.draft_schedule,self.proposal_temperature,step);
                self.categorical[step].enqueue(logits, first, 1, temperature, stream)?;
                self.categorical[step].probabilities(
                    logits,
                    first,
                    1,
                    temperature,
                    &mut self.proposal_probabilities,
                    step,
                    stream,
                )?;
            } else {
                self.samplers[step].run(&self.gathered, stream)?;
            }
            if !direct { unsafe {
                self.cuda.copy_device(
                    self.corrected.ptr() + (step * VOCAB * 4) as u64,
                    self.gathered.ptr(),
                    VOCAB * 4,
                    stream,
                )
            }?; }
        }
        if prefix_only {
            // Four-word gather retains the existing transport geometry. Only
            // the first three probabilities are current; word4 is ignored.
            let input = unsafe { self.confidence_probability.view(0, 4)? };
            unsafe { comm.gather_f32(&input, &mut self.confidence_gathered_limit, 4, stream) }?;
            self.pack_markov_readout(sampled,2,3,stream)?;
            for (offset,source) in [(16,self.confidence_raw.ptr()),(21,self.confidence_probability.ptr()),
                (27,self.confidence_gathered_limit.ptr())] {
                unsafe{self.cuda.copy_device(self.readout.ptr()+offset*4,source,3*4,stream)}?;
            }
            return Ok(());
        }
        self.pack_markov_readout(sampled,5,5,stream)?;
        self.enqueue_confidence_readout(comm,stream)
    }
    fn pack_markov_readout(&self,sampled:bool,produced:usize,confidences:usize,stream:&Stream)->Result<()> {
        ensure!((1..=5).contains(&produced)&&(produced..=5).contains(&confidences),"partial readout geometry");
        let mut pointers = Vec::with_capacity(16);
        for step in 0..5 {
            let step=step.min(produced-1);
            pointers.push(if sampled {
                self.sampled_output(step)
            } else {
                self.samplers[step].device_output()?.ptr()
            });
        }
        for step in 0..5 {
            let step=step.min(produced-1);
            pointers.push(if sampled {
                self.sampled_errors(step)
            } else {
                self.samplers[step].device_errors()?.ptr()
            });
        }
        for step in 0..5 {
            pointers.push(self.errors[step.min(confidences-1)].ptr());
        }
        pointers.push(self.readout.ptr());
        let mut args = pointers
            .iter_mut()
            .map(|v| (v as *mut u64).cast())
            .collect::<Vec<_>>();
        unsafe {
            self.module.function("ca_dspark_markov_readout")?.launch(
                [1, 1, 1],
                [32, 1, 1],
                0,
                stream,
                &mut args,
            )
        }?;
        Ok(())
    }
    fn enqueue_confidence_readout(&mut self,comm:&Roce,stream:&Stream)->Result<()> {
        let mut probability = self.confidence_probability.ptr();
        let mut local_limit = self.confidence_local_limit.ptr();
        if let Some(costs) = &self.confidence_costs {
            let mut costs = costs.ptr();
            let mut args = [
                (&mut probability as *mut u64).cast(),
                (&mut costs as *mut u64).cast(),
                (&mut local_limit as *mut u64).cast(),
            ];
            unsafe {
                self.module
                    .function("ca_dspark_confidence_cost_limit")?
                    .launch([1, 1, 1], [32, 1, 1], 0, stream, &mut args)
            }?;
        } else {
            let mut args = [
                (&mut probability as *mut u64).cast(),
                (&mut local_limit as *mut u64).cast(),
            ];
            unsafe {
                self.module.function("ca_dspark_confidence_limit")?.launch(
                    [1, 1, 1],
                    [32, 1, 1],
                    0,
                    stream,
                    &mut args,
                )
            }?;
        }
        unsafe {
            comm.gather_f32(
                &self.confidence_local_limit,
                &mut self.confidence_gathered_limit,
                4,
                stream,
            )
        }?;
        let mut raw = self.confidence_raw.ptr();
        let mut gathered = self.confidence_gathered_limit.ptr();
        let mut readout = self.readout.ptr();
        let mut args = [
            (&mut raw as *mut u64).cast(),
            (&mut probability as *mut u64).cast(),
            (&mut gathered as *mut u64).cast(),
            (&mut readout as *mut u64).cast(),
        ];
        unsafe {
            self.module
                .function("ca_dspark_confidence_readout")?
                .launch([1, 1, 1], [32, 1, 1], 0, stream, &mut args)
        }?;
        Ok(())
    }
    pub(crate) fn confidence(&self) -> Result<([f32; 5], usize)> {
        ensure!(self.ready && !self.poisoned, "confidence not current");
        self.last_confidence
            .ok_or_else(|| anyhow::anyhow!("confidence unavailable"))
    }
    /// Component economics of a host-visible causal decision. Both branches
    /// are forced and measured; this does not choose or deploy a length policy.
    pub(crate) fn probe_prefix_stop(
        &mut self, base:&Buffer<f32>, post_hc:&Buffer<bf16>, anchor:u32,
        position:u64, comm:&Roce, stream:&Stream,
    ) -> Result<serde_json::Value> {
        ensure!(std::env::var("CANADA_SCRATCH_MARKOV_PREFIX_STOP_SCREEN").as_deref()==Ok("1")
            && self.direct_gather && self.fused_head_active && self.confidence_fused_active,
            "prefix component requires current qualified draft kernels");
        fn digest(cuda:&Cuda, ranges:&[(u64,usize)]) -> Result<String> {
            let mut h=Sha256::new();
            for &(p,n) in ranges { h.update(unsafe {cuda.read_device(p,n)}?); }
            Ok(format!("{:x}",h.finalize()))
        }
        let original_temperature=self.temperature();
        self.set_temperature(0.7,stream)?;
        let outcome=(|| {
            self.run(base,post_hc,anchor,Some((20261027,position)),comm,stream)?;
            self.enqueue_segment(base,post_hc,true,comm,stream,0,true)?;
            self.enqueue_segment(base,post_hc,true,comm,stream,2,false)?;
            stream.synchronize()?;comm.health()?;
            let prefix=stream.capture(||self.enqueue_segment(base,post_hc,true,comm,stream,0,true))?;
            let tail=stream.capture(||self.enqueue_segment(base,post_hc,true,comm,stream,2,false))?;
            let conditional_enabled=std::env::var("CANADA_SCRATCH_MARKOV_CONDITIONAL_SCREEN").as_deref()==Ok("1");
            let mut conditional_result=self.cuda.allocate::<u32>(4)?;
            let mut conditional=Vec::new();
            if conditional_enabled {
                let path=std::env::var("CANADA_SCRATCH_MARKOV_CONDITIONAL_LIBRARY")?;
                let bridge=crate::markov_conditional::Bridge::open(&self.cuda,Path::new(&path))?;
                let first=bridge.capture(stream,||self.enqueue_segment(base,post_hc,true,comm,stream,0,true))?;
                let rest=bridge.capture(stream,||self.enqueue_segment(base,post_hc,true,comm,stream,2,false))?;
                for force in [1,2,0] {
                    conditional.push(unsafe{bridge.graph(&first,&rest,&self.confidence_gathered_limit,
                        &mut conditional_result,force,stream)}?);
                }
                // Child graphs are cloned by CUDA; the source handles may now
                // be released. Captured buffers remain owned by this probe.
            }
            let index=proposal_graph_index(true,true,self.temperature_index,
                self.fused_head_active,self.confidence_fused_active,self.draft_schedule,self.correction_cache_active,self.scaled_head_active);
            let begin=self.cuda.event()?;let end=self.cuda.event()?;
            let mut barrier_input=self.cuda.allocate::<f32>(4)?;
            let mut barrier_output=self.cuda.allocate::<f32>(16)?;
            barrier_input.upload(&[0.;4])?;
            let prefix_ranges=[(self.corrected.ptr(),2*VOCAB*4),
                (self.proposal_probabilities.ptr(),2*VOCAB*8),
                (self.confidence_raw.ptr(),3*4),(self.confidence_probability.ptr(),3*4)];
            let tail_ranges=[(self.corrected.ptr()+(2*VOCAB*4)as u64,3*VOCAB*4),
                (self.proposal_probabilities.ptr()+(2*VOCAB*8)as u64,3*VOCAB*8),
                (self.confidence_raw.ptr()+3*4,2*4),(self.confidence_probability.ptr()+3*4,2*4)];
            let full_ranges=[(self.corrected.ptr(),5*VOCAB*4),
                (self.proposal_probabilities.ptr(),5*VOCAB*8),
                (self.confidence_raw.ptr(),5*4),(self.confidence_probability.ptr(),5*4)];
            let mut records=Vec::new();
            for repeat in 0..5 {
                let seed=20261127+repeat;
                let expected_ids=self.run(base,post_hc,anchor,Some((seed,position)),comm,stream)?;
                let expected_prefix=digest(&self.cuda,&prefix_ranges)?;
                let expected_full=digest(&self.cuda,&full_ranges)?;
                let mut order:Vec<usize>=(0..if conditional_enabled{6}else{3}).collect();
                if repeat%2!=0{order.reverse();}
                for mode in order {
                    // Poison future rows with a different valid draw. The
                    // short segment must leave those excluded rows untouched.
                    self.run(base,post_hc,anchor,Some((seed+10000,position)),comm,stream)?;
                    let unused_before=digest(&self.cuda,&tail_ranges)?;
                    for(step,sampler)in self.categorical.iter_mut().enumerate(){
                        sampler.upload_uniforms(&[uniform(seed,DRAFT_DOMAIN,position+step as u64)])?;
                    }
                    stream.join_default()?;
                    unsafe {comm.gather_f32(&barrier_input,&mut barrier_output,4,stream)}?;
                    stream.synchronize()?;comm.health()?;
                    let started=std::time::Instant::now();begin.record(stream)?;
                    let mut short_decision=false;
                    let mut used=if mode==1{2}else{5};
                    for _ in 0..8 {
                        if mode==0 {
                            unsafe {self.graphs[index].as_ref().unwrap().launch(stream)}?;
                        }else if mode<3{
                            unsafe {prefix.launch(stream)}?;
                            stream.synchronize()?;
                            let confidence=self.confidence_gathered_limit.download()?;
                            ensure!(confidence.chunks_exact(4).all(|p|p[..3].iter().all(|v|
                                v.is_finite()&&*v>=0.&&*v<=1.)),"prefix confidence validity");
                            // As with native scheduling, rank0 owns the choice.
                            short_decision=confidence[..3].iter().product::<f32>()<0.2;
                            if mode==2 {unsafe {tail.launch(stream)}?;}
                        }else{
                            unsafe{conditional[mode-3].launch(stream)}?;
                            stream.synchronize()?;
                            let decision=conditional_result.download()?;
                            ensure!(decision[1]==0&&decision[3]==0x4d434f4e&&decision[0]<=1&&decision[2]<=1,
                                "conditional decision/error readout");
                            if mode==3{ensure!(decision[0]==0,"forced short conditional continued");}
                            if mode==4{ensure!(decision[0]==1,"forced continue conditional stopped");}
                            if mode==5{ensure!(decision[0]+decision[2]==1,"adaptive conditional disagrees with threshold");}
                            used=if decision[0]==0{2}else{5};short_decision=decision[2]!=0;
                        }
                        stream.synchronize()?;comm.health()?;
                        if used==5 {let words=self.readout.download()?;
                            ensure!(words[..5]==expected_ids&&words[5..15].iter().all(|v|*v==0)
                                &&words[15]==0x4453504b,"split replay output/errors");}
                    }
                    end.record(stream)?;
                    let timeline_ms=end.elapsed_since(&begin)?/8.;
                    let wall_ms=started.elapsed().as_secs_f64()*1000./8.;
                    let device_replay_ms=if conditional_enabled&&(mode==0||mode>=3){
                        unsafe{comm.gather_f32(&barrier_input,&mut barrier_output,4,stream)}?;
                        stream.synchronize()?;begin.record(stream)?;
                        for _ in 0..16 {if mode==0 {
                            unsafe{self.graphs[index].as_ref().unwrap().launch(stream)}?;
                        }else{unsafe{conditional[mode-3].launch(stream)}?;}}
                        end.record(stream)?;let ms=end.elapsed_since(&begin)?/16.;comm.health()?;Some(ms)
                    }else{None};
                    ensure!(digest(&self.cuda,&prefix_ranges)?==expected_prefix,"split prefix changed actual-q/logits/confidence");
                    for step in 0..used {
                        ensure!(self.categorical[step].device_output().download()?[0]==expected_ids[step] as i32
                            &&self.categorical[step].device_errors().download()?[0]==0
                            &&self.errors[step].download()?[0]==0,"split sampled ID/error mismatch");
                    }
                    if used==2 {ensure!(digest(&self.cuda,&tail_ranges)?==unused_before,"short path touched excluded rows");}
                    else {ensure!(digest(&self.cuda,&full_ranges)?==expected_full,"split tail changed actual-q/logits/confidence");}
                    records.push(serde_json::json!({"repeat":repeat,"warmup":repeat==0,"seed":seed,
                        "mode":(["full","prefix_only","split_continue","cuda_short","cuda_continue","cuda_adaptive"][mode]),"timeline_ms":timeline_ms,"wall_ms":wall_ms,
                        "replays":8,"device_replay_ms":device_replay_ms,"short_decision_at_0_2":if mode==0{None}else{Some(short_decision)},
                        "prefix_sha256":expected_prefix,"full_sha256":if used==2{None}else{Some(expected_full.clone())},
                        "used_proposals":&expected_ids[..used],"all_exact":true}));
                }
            }
            Ok(serde_json::json!({"all_exact":true,"records":records,"draft_temperature":0.7,
                "conditional_graphs":conditional_enabled,
                "parallel_finish":self.categorical[0].parallel_finish_enabled(),
                "normalized_factor":self.categorical[0].normalized_factor_enabled(),
                "scope":"Forced short and continue branches on identical queries/uniforms. Includes host decision/readout and extra consensus gather; excludes query, verification and outcomes. No length policy deployment."}))
        })();
        self.set_temperature(original_temperature,stream)?;
        self.invalidate_results();
        outcome
    }
    pub(crate) fn direct_gather_enabled(&self)->bool { self.direct_gather }
    pub(crate) fn cached_probability_factors(&self)->bool {
        self.categorical.iter().all(|c|c.cached_probability_factors())
    }
    /// Same query/history and uniforms, alternating graphs. GPU timings omit
    /// diagnostic copies and their peer-arrival skew; exact q is checked too.
    pub(crate) fn probe_direct_gather(
        &mut self, base:&Buffer<f32>, post_hc:&Buffer<bf16>, anchor:u32,
        position:u64, comm:&Roce, stream:&Stream,
    ) -> Result<serde_json::Value> {
        use sha2::{Digest,Sha256};
        ensure!(std::env::var("CANADA_SCRATCH_MARKOV_GATHER_SCREEN").as_deref()==Ok("1"),
            "direct gather probe requires isolated admission");
        let original=self.direct_gather;
        let mut barrier_input=self.cuda.allocate::<f32>(4)?;
        let mut barrier_output=self.cuda.allocate::<f32>(16)?;
        barrier_input.upload(&[0.;4])?;
        let begin=self.cuda.event()?;
        let end=self.cuda.event()?;
        let mut records=Vec::new();
        let outcome=(|| {
            for repeat in 0..5 {
                let order=if repeat>0 && repeat%2==0 {[true,false]}else{[false,true]};
                let mut signature=None;
                for direct in order {
                    self.direct_gather=direct;
                    let seed=20261024+repeat;
                    let started=std::time::Instant::now();
                    let ids=self.run(base,post_hc,anchor,Some((seed,position)),comm,stream)?;
                    let run_seconds=started.elapsed().as_secs_f64();
                    stream.join_default()?;
                    let barrier_start=std::time::Instant::now();
                    unsafe { comm.gather_f32(&barrier_input,&mut barrier_output,4,stream) }?;
                    stream.synchronize()?;
                    comm.health()?;
                    let barrier_seconds=barrier_start.elapsed().as_secs_f64();
                    begin.record(stream)?;
                    for _ in 0..16 {
                        unsafe { self.graphs[proposal_graph_index(true,direct,self.temperature_index,
                            self.fused_head_active,self.confidence_fused_active,self.draft_schedule,self.correction_cache_active,self.scaled_head_active)].as_ref().unwrap().launch(stream) }?;
                    }
                    end.record(stream)?;
                    let graph_ms=end.elapsed_since(&begin)?/16.;
                    comm.health()?;
                    let mut hash=Sha256::new();
                    for (ptr,bytes) in [(self.corrected.ptr(),self.corrected.bytes()),
                        (self.proposal_probabilities.ptr(),self.proposal_probabilities.bytes()),
                        (self.confidence_raw.ptr(),self.confidence_raw.bytes()),
                        (self.confidence_probability.ptr(),self.confidence_probability.bytes())] {
                        hash.update(unsafe {self.cuda.read_device(ptr,bytes)}?);
                    }
                    let words=self.readout.download()?;
                    ensure!(words[..5]==ids && words[5..15].iter().all(|v|*v==0)
                        && words[15]==0x4453504b,"direct gather replay output/errors");
                    let digest=format!("{:x}",hash.finalize());
                    let current=(ids,digest.clone(),self.confidence()?.0.map(f32::to_bits));
                    if let Some(expected)=&signature { ensure!(expected==&current,"direct gather changed proposals/q/confidence/logits"); }
                    else {signature=Some(current);}
                    records.push(serde_json::json!({"repeat":repeat,"warmup":repeat==0,
                        "direct_gather":direct,"seed":seed,"graph_ms":graph_ms,
                        "run_seconds":run_seconds,"pretrial_barrier_seconds":barrier_seconds,
                        "proposals":ids,"logits_q_confidence_sha256":digest,"replays":16}));
                }
            }
            Ok(serde_json::json!({"all_exact":true,"records":records,
                "cached_probability_factors":self.categorical.iter().all(|c|c.cached_probability_factors()),
                "scope":"Four balanced warm pairs; exact corrected logits, actual q, confidence and proposals. Sixteen same-input graph replays/trial; excludes query, target, outcomes and HTTP."}))
        })();
        self.direct_gather=original;
        outcome
    }
    pub(crate) fn dump(&self, root: &Path) -> Result<serde_json::Value> {
        ensure!(
            self.ready && !self.poisoned && self.produced==5,
            "DSpark Markov output not ready"
        );
        let raw = unsafe {
            self.cuda
                .read_device(self.corrected.ptr(), self.corrected.bytes())
        }?;
        std::fs::write(root.join("corrected_logits.bin"), &raw)?;
        for (name, buffer) in [
            ("confidence_features", &self.confidence_features),
            ("confidence_weight", &self.confidence_weight),
            ("confidence_raw", &self.confidence_raw),
            ("confidence_probability", &self.confidence_probability),
        ] {
            let bytes = unsafe { self.cuda.read_device(buffer.ptr(), buffer.bytes()) }?;
            std::fs::write(root.join(format!("{name}.bin")), &bytes)?;
        }
        use sha2::{Digest, Sha256};
        Ok(
            serde_json::json!({"corrected_logits_sha256":format!("{:x}",Sha256::digest(raw)),
            "corrected_logits_dtype":"FP32","markov_projection_dtype":"BF16 operands, FP32 accumulation/output","target_correct_stochastic_sampling":false,"proposal_sampling":if self.sampled {"T1/topP1 categorical, target rejection qualified separately"}else{"greedy"},
            "confidence":self.confidence()?.0,"rank0_scheduled_verify_rows":self.confidence()?.1,
            "confidence_projection_dtype":"FP32; exact BF16 source values widened on GPU; no bias"}),
        )
    }
}

#[cfg(test)]
mod temperature_graph_tests {
    use super::*;
    #[test]
    fn proposal_distributions_never_alias_graphs() {
        let mut seen=std::collections::BTreeSet::new();
        for head in [false,true] { for confidence in [false,true] { for cache in [false,true] { for scaled in [false,true] {
            let greedy=proposal_graph_index(false,false,0,head,confidence,0,cache,scaled);
            assert!(seen.insert(greedy));
            for schedule in 0..DRAFT_SCHEDULES {
                for temperature in 0..DRAFT_TEMPERATURES.len() { for direct in [false,true] {
                    let i=proposal_graph_index(true,direct,temperature,head,confidence,schedule,cache,scaled);
                    assert!(i<TEMPERATURE_GRAPH_STRIDE*16*DRAFT_SCHEDULES && seen.insert(i));
                    assert_eq!(proposal_graph_index(false,direct,temperature,head,confidence,schedule,cache,scaled),greedy);
                }}
            }
        }}}}
        assert_eq!(seen.len(),16*(1+2*DRAFT_TEMPERATURES.len()*DRAFT_SCHEDULES));
    }
    #[test]
    fn depth_schedule_retains_control_and_explicit_temperatures() {
        for t in DRAFT_TEMPERATURES {for step in 0..5 {assert_eq!(temperature_at(0,t,step),t);}}
        assert_eq!(std::array::from_fn::<_,5,_>(|i|temperature_at(1,0.7,i)),[0.7,0.7,0.5,0.5,0.4]);
        assert_eq!(std::array::from_fn::<_,5,_>(|i|temperature_at(2,0.7,i)),[0.7,0.85,1.,1.2,1.2]);
        assert_eq!(std::array::from_fn::<_,5,_>(|i|temperature_at(3,0.7,i)),[1.,0.7,0.7,0.7,0.7]);
    }
}
