//! GPU rejection sampling for a stochastic autoregressive draft. No acceptance
//! shortcut, temperature adjustment, approximate match or host probability math.
use crate::{
    categorical::{ACCEPT_DOMAIN, Categorical, RESIDUAL_DOMAIN, TARGET_DOMAIN, uniform},
    cuda::{Buffer, Cuda, Graph, MappedHost, Module, Stream},
    sampling::VOCAB,
};
use anyhow::{Result, ensure};
#[path="rejection_sharded.rs"]
mod sharded;
pub(crate) struct Rejection {
    // Destroy captured work before any sampler-owned buffers/modules. External
    // input addresses are checked on every replay; session state is not cached.
    graphs: Vec<Option<SamplerGraph>>,
    sharded_graphs: Vec<Option<sharded::ShardedGraph>>,
    sharded: Option<crate::sharded_block::ShardedBlock>,
    sharded_comm: Option<(usize,u64)>,
    bundled_ids: bool,
    root_branch_capture:Option<crate::root_branch_capture::Capture>,
    local_targets: Vec<Option<crate::sharded_categorical_probe::CompactSampler>>,
    graph_replay: bool,
    graph_readout: bool,
    staged_inputs: Option<std::rc::Rc<crate::cuda::MetadataStaging>>,
    cuda: Cuda,
    module: Module,
    capacity: usize,
    target: Categorical,
    p: Buffer<f64>,
    proposals: Buffer<i32>,
    accept_u: Buffer<f64>,
    residual_u: Buffer<f64>,
    state: Buffer<u32>,
    errors: Buffer<u32>,
    output: Buffer<i32>,
    sums: Buffer<f64>,
    tiles: Buffer<i32>,
    fractions: Buffer<f64>,
    block_weights: Buffer<f64>,
    shadow_expected: Option<Buffer<f64>>,
    readout: Option<Readout>,
    progress: Option<crate::block_progress::BlockProgress>,
    parallel_block_decide: bool,
    parallel_block_finish: bool,
    fused_block_finish: bool,
    decision_control: Option<(Buffer<u32>, Buffer<i32>)>,
}
struct SamplerGraph {
    key:(u64,u64,usize,usize,bool,usize,u64),
    graph:Graph,
}
enum Readout { Device(Buffer<u32>), Mapped(MappedHost) }
impl Readout {
    fn pointer(&self)->u64 {match self{Self::Device(b)=>b.ptr(),Self::Mapped(b)=>b.device_ptr()}}
    fn read(&self)->Result<Vec<u32>> {match self {
        Self::Device(b)=>b.download(),
        Self::Mapped(b)=>(0..b.len()/4).map(|i|b.status_word(i*4)).collect(),
    }}
}
pub(crate) struct Draw {
    pub ids: Vec<u32>,
    pub accepted: Vec<bool>,
    /// Block correction may equal a draft token; equality cannot extend its prefix.
    pub accepted_prefix: Option<usize>,
    /// Diagnostic only: each prefix's [token, block] expected committed length.
    pub acceptance_expected_lengths: Option<Vec<[f64; 2]>>,
    /// GPU-integrated first-proposal overlap; only populated by shadow probes.
    pub first_proposal_overlap: Option<f64>,
    /// Shadow-only next-token overlaps at sampled ancestor prefixes, followed
    /// by sequential Rao-Blackwell estimate and a product-overlap heuristic.
    pub local_proposal_fit: Option<(Vec<f64>,f64,f64)>,
}
impl Rejection {
    pub(crate) fn new_for_graph_probe(cuda:&Cuda,core:&[u8],candidate:bool)->Result<Self> {
        ensure!(std::env::var("CANADA_SCRATCH_SAMPLER_GENERATION_SCREEN").as_deref()==Ok("1"),
            "isolated sampler construction");
        if std::env::var("CANADA_SCRATCH_FUSED_PROBABILITY_RESIDUAL_MODEL_SCREEN").as_deref()==Ok("1") {
            let mut sampler=Self::new(cuda,core,6)?;
            ensure!(sampler.bundled_ids && sampler.graph_replay && sampler.graph_readout,
                "probability fusion screen requires current sampler graph and transport");
            sampler.sharded.as_mut().ok_or_else(||anyhow::anyhow!("probability fusion screen needs sharded actual q"))?
                .fused_probabilities=candidate;
            return Ok(sampler);
        }
        let finish_probe=std::env::var("CANADA_SCRATCH_BLOCK_FINISH_SCREEN").as_deref()==Ok("1");
        let fused_probe=std::env::var("CANADA_SCRATCH_BLOCK_FUSED_SCREEN").as_deref()==Ok("1");
        let composition=std::env::var("CANADA_SCRATCH_SOLO_COMPOSITION_SCREEN").as_deref()==Ok("1");
        if std::env::var("CANADA_SCRATCH_SAMPLER_GRAPH_READOUT_SCREEN").as_deref()==Ok("1") {
            let mut sampler=Self::with_staging(cuda,core,6,true)?;
            sampler.graph_replay=true;
            sampler.graph_readout=candidate;
            return Ok(sampler);
        }
        if std::env::var("CANADA_SCRATCH_SAMPLER_MAPPED_SCREEN").as_deref()==Ok("1") {
            let mut sampler=Self::with_controls(cuda,core,6,if candidate{2}else{1})?;
            sampler.graph_replay=true;
            return Ok(sampler);
        }
        let mut sampler=Self::with_staging(cuda,core,6,if composition {candidate}else{finish_probe||fused_probe||candidate})?;
        sampler.graph_replay=finish_probe||fused_probe||candidate;
        if finish_probe {sampler.parallel_block_finish=candidate;}
        if fused_probe {sampler.parallel_block_finish=true;sampler.fused_block_finish=candidate;}
        if composition {sampler.graph_replay=candidate;sampler.parallel_block_finish=candidate;sampler.fused_block_finish=candidate;}
        Ok(sampler)
    }
    pub(crate) fn buffered_draw_sources(&self,live:usize,logits:u64,q:u64)
        ->Result<[(&'static str,u64,usize);9]> {
        ensure!((2..=6).contains(&live)&&live<=self.capacity
            &&std::env::var("CANADA_SCRATCH_READOUT_BUFFERED_TRACE").as_deref()==Ok("1"),
            "finite buffered draw extent");
        Ok([("logits",logits,live*VOCAB*4),("q",q,(live-1)*VOCAB*8),
            ("p",self.p.ptr(),live*VOCAB*8),("accept-u",self.accept_u.ptr(),live*8),
            ("residual-u",self.residual_u.ptr(),live*8),
            ("bonus-u",self.target.uniform_pointer_for_probe()?,live*8),
            ("proposals",self.proposals.ptr(),live*4),("state",self.state.ptr(),live*4),
            ("output",self.output.ptr(),live*4)])
    }
    /// Byte capture after sample() has synchronized its readout. This has no
    /// host probability arithmetic and cannot change the completed draw.
    pub(crate) fn capture_completed_draw_for_probe(&self,path:&std::path::Path,live:usize)->Result<()> {
        ensure!(live>0 && live<=self.capacity
            && std::env::var("CANADA_SCRATCH_ENCODER_CACHE_TRACE").as_deref()==Ok("1"),
            "finite completed-draw capture admission");
        for (name,ptr,bytes) in [
            ("p.bin",self.p.ptr(),live*VOCAB*8),
            ("accept-u.bin",self.accept_u.ptr(),live*8),
            ("residual-u.bin",self.residual_u.ptr(),live*8),
            ("proposals.bin",self.proposals.ptr(),live*4),
            ("state.bin",self.state.ptr(),live*4),
            ("output.bin",self.output.ptr(),live*4),
        ] {
            std::fs::write(path.join(name),unsafe{self.cuda.read_device(ptr,bytes)}?)?;
        }
        std::fs::write(path.join("bonus-u.bin"),self.target.uniform_bytes_for_probe(live)?)?;
        Ok(())
    }
    /// Called immediately after a completed draw, before the next sampler use.
    pub(crate) fn root_probabilities_after_draw_for_probe(&self)->Result<&Buffer<f64>> {
        ensure!(std::env::var("CANADA_SCRATCH_ROOT_FEEDBACK_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_QUERY_REUSE_SCREEN").as_deref()==Ok("1"),
            "root-feedback target probabilities require isolated admission");
        Ok(&self.p)
    }
    /// Enable the existing GPU expectation diagnostic on a separate sampler.
    /// The history screen never enables it on the serving session's sampler.
    pub(crate) fn enable_history_expectation_probe(&mut self) -> Result<()> {
        ensure!((std::env::var("CANADA_SCRATCH_HISTORY_OPPORTUNITY").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_REASONING_DRAFT_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_POST_THINKING_DRAFT_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_MATCHED_PATH_SCREEN").as_deref()==Ok("1"))
            && self.shadow_expected.is_none(), "isolated history expectation probe");
        self.shadow_expected=Some(self.cuda.allocate(self.capacity*3+2)?);
        Ok(())
    }
    pub(crate) fn cached_probability_factors(&self) -> bool {
        self.target.cached_probability_factors()
    }
    pub(crate) fn new(cuda: &Cuda, core: &[u8], capacity: usize) -> Result<Self> {
        let staged=match std::env::var("CANADA_SCRATCH_REJECTION_STAGED_INPUTS") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid rejection input policy");v=="1"},
            // s1189 exact GPU draws/probabilities; s1190 four-rank generated
            // tokens and full accepted state match with unchanged RNG inputs.
            Err(std::env::VarError::NotPresent)=>true,Err(e)=>return Err(e.into()),
        };
        let mapped=match std::env::var("CANADA_SCRATCH_REJECTION_MAPPED_INPUTS") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid mapped rejection input policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
        };
        Self::with_controls(cuda,core,capacity,if mapped{2}else{usize::from(staged)})
    }
    fn with_staging(cuda:&Cuda,core:&[u8],capacity:usize,staged:bool)->Result<Self> {
        Self::with_controls(cuda,core,capacity,usize::from(staged))
    }
    fn with_controls(cuda:&Cuda,core:&[u8],capacity:usize,mode:usize)->Result<Self> {
        ensure!((1..=128).contains(&capacity), "rejection capacity");
        // Only these CPU-written controls use the arena. Each draw completes
        // its copy and GPU readers before the next call can rewrite the source.
        let controls=match mode {
            0=>cuda.clone(),1=>unsafe{cuda.staged_metadata_allocator()?},
            2=>unsafe{cuda.mapped_metadata_allocator()?},
            _=>anyhow::bail!("invalid rejection control storage"),
        };
        let parallel_block_decide = match std::env::var("CANADA_SCRATCH_PARALLEL_BLOCK_DECIDE") {
            Ok(v) => { ensure!(v == "0" || v == "1", "invalid block decision policy"); v == "1" },
            Err(std::env::VarError::NotPresent) => true,
            Err(e) => return Err(e.into()),
        };
        let parallel_block_finish=match std::env::var("CANADA_SCRATCH_PARALLEL_BLOCK_FINISH") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid block finish policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>true,Err(e)=>return Err(e.into()),
        };
        let fused_block_finish=match std::env::var("CANADA_SCRATCH_FUSED_BLOCK_FINISH") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid fused block finish policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>true,Err(e)=>return Err(e.into()),
        };
        let decision_control = if std::env::var("CANADA_SCRATCH_BLOCK_DECIDE_ROWS_VALIDATE").as_deref() == Ok("1") {
            ensure!(std::env::var("CANADA_SCRATCH_SAMPLER_READOUT_SCREEN").as_deref() == Ok("1"),
                "block decision mirror requires isolated native sampler screen");
            Some((cuda.allocate(capacity)?, cuda.allocate(capacity)?))
        } else { None };
        let readout=match std::env::var("CANADA_SCRATCH_SAMPLER_READOUT") {
            Ok(v)=>match v.as_str(){"0"=>None,"1"=>Some(Readout::Device(cuda.allocate(capacity*5)?)),
                "2"=>Some(Readout::Mapped(cuda.mapped_host(capacity*5*4)?)),_=>anyhow::bail!("invalid sampler readout mode")},
            // s820 preserves native integer results/error messages; s822
            // preserves complete outputs and retained state on all four ranks.
            // Shared GB10 memory avoids five tiny DtoH copies (~20us/readout).
            // Mode0 remains the ordinary-copy control.
            Err(std::env::VarError::NotPresent)=>Some(Readout::Mapped(cuda.mapped_host(capacity*5*4)?)),
            Err(e)=>return Err(e.into()),
        };
        let graph_replay=match std::env::var("CANADA_SCRATCH_REJECTION_GRAPHS") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid rejection graph policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>true,Err(e)=>return Err(e.into()),
        };
        let graph_readout=match std::env::var("CANADA_SCRATCH_REJECTION_GRAPH_READOUT") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid rejection graph readout policy");v=="1"},
            // s1199:384 exact native draws, invalid-value recovery and four
            // destination rebindings; all-rank generation/state checks pass.
            Err(std::env::VarError::NotPresent)=>true,Err(e)=>return Err(e.into()),
        };
        Ok(Self {
            graphs:(0..capacity*2).map(|_|None).collect(),graph_replay,graph_readout,
            sharded_graphs:(0..capacity).map(|_|None).collect(),sharded_comm:None,
            bundled_ids:crate::sampler_transport::Policy::from_env()?.bundled_ids,
            root_branch_capture:crate::root_branch_capture::Capture::from_env()?,
            local_targets:(1..=capacity).map(|rows|->Result<_>{
                if capacity==6&&rows>1&&(crate::sampler_transport::Policy::from_env()?.local_target
                    ||std::env::var("CANADA_SCRATCH_LOCAL_TARGET_MODEL_SCREEN").as_deref()==Ok("1")) {
                    let mut sampler=crate::sharded_categorical_probe::CompactSampler::with_uniform_allocator(cuda,core,rows,0,&controls)?;
                    sampler.prepare_local_probabilities()?;Ok(Some(sampler))
                }else{Ok(None)}
            }).collect::<Result<Vec<_>>>()?,
            sharded:if capacity==6&&(crate::sampler_transport::Policy::from_env()?.hierarchical_draft
                ||std::env::var("CANADA_SCRATCH_HIERARCHICAL_MODEL_SCREEN").as_deref()==Ok("1")) {
                Some(crate::sharded_block::ShardedBlock::with_controls(cuda,core,0,capacity,&controls)?)
            }else{None},
            staged_inputs:controls.metadata_staging(),
            cuda: cuda.clone(),
            module: cuda.module(core)?,
            capacity,
            readout,
            parallel_block_decide,parallel_block_finish,fused_block_finish,
            decision_control,
            progress:if capacity==6&&std::env::var("CANADA_SCRATCH_BLOCK_PROGRESS_SCREEN").as_deref()==Ok("1") {
                Some(crate::block_progress::BlockProgress::new(cuda,core)?)
            }else{None},
            target: Categorical::with_uniform_allocator(cuda,core,capacity,&controls)?,
            p: cuda.allocate(capacity * VOCAB)?,
            proposals: controls.allocate(capacity)?,
            accept_u: controls.allocate(capacity)?,
            residual_u: controls.allocate(capacity)?,
            state: cuda.allocate(capacity)?,
            errors: cuda.allocate(capacity)?,
            output: cuda.allocate(capacity)?,
            sums: cuda.allocate(capacity * 127)?,
            tiles: cuda.allocate(capacity)?,
            fractions: cuda.allocate(capacity)?,
            block_weights: cuda.allocate(capacity)?,
            shadow_expected: if std::env::var("CANADA_SCRATCH_ACCEPTANCE_SHADOW").as_deref() == Ok("1") {
                Some(cuda.allocate(capacity * 3 + 2)?)
            } else { None },
        })
    }
    pub(crate) fn set_readout_for_probe(&mut self,mode:u8,stream:&Stream)->Result<()> {
        ensure!((std::env::var("CANADA_SCRATCH_SAMPLER_READOUT_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_SAMPLER_GRAPH_READOUT_SCREEN").as_deref()==Ok("1"))
            && self.cuda.owns_stream(stream),"isolated sampler readout screen");
        stream.synchronize()?;
        for graph in &mut self.graphs{*graph=None;}
        self.readout=match mode {0=>None,1=>Some(Readout::Device(self.cuda.allocate(self.capacity*5)?)),
            2=>Some(Readout::Mapped(self.cuda.mapped_host(self.capacity*20)?)),_=>anyhow::bail!("invalid readout mode")};
        Ok(())
    }
    pub(crate) fn set_block_progress_for_probe(&mut self,enabled:bool,validate:bool,stream:&Stream)->Result<()> {
        self.progress.as_mut().ok_or_else(||anyhow::anyhow!("block progress not prepared"))?.reset(enabled,validate,stream)
    }
    pub(crate) fn observe_block_progress_for_probe(&mut self,rows:usize,seconds:f64,stream:&Stream)
        ->Result<(usize,serde_json::Value)> {
        self.progress.as_mut().ok_or_else(||anyhow::anyhow!("block progress not prepared"))?.observe(rows,seconds,stream)
    }
    fn finish_readout(&self,live:usize,stream:&Stream)->Result<(Vec<i32>,Vec<u32>)> {
        self.finish_readout_inner(live,stream,false)
    }
    fn enqueue_readout(&self,stream:&Stream)->Result<()> {
        self.enqueue_readout_from(self.target.device_errors(),self.target.device_output(),self.capacity,stream)
    }
    fn enqueue_readout_from(&self,target_errors:&Buffer<u32>,target_ids:&Buffer<i32>,
        rows:usize,stream:&Stream)->Result<()> {
        ensure!(rows>0&&rows<=self.capacity&&target_errors.len()>=rows&&target_ids.len()>=rows
            &&self.cuda.owns_buffer(target_errors)&&self.cuda.owns_buffer(target_ids),"readout target ownership/extent");
        let readout=self.readout.as_ref().ok_or_else(||anyhow::anyhow!("packed readout unavailable"))?;
        let mut te=target_errors.ptr();let mut ti=target_ids.ptr();
        let mut re=self.errors.ptr();let mut ids=self.output.ptr();let mut states=self.state.ptr();
        let mut packed=readout.pointer();let mut cap=rows as i32;
        let mut args=[(&mut te as *mut u64).cast(),(&mut ti as *mut u64).cast(),
            (&mut re as *mut u64).cast(),(&mut ids as *mut u64).cast(),
            (&mut states as *mut u64).cast(),(&mut packed as *mut u64).cast(),(&mut cap as *mut i32).cast()];
        unsafe{self.module.function("ca_sampler_readout")?.launch([1,1,1],[128,1,1],0,stream,&mut args)?;}
        Ok(())
    }
    fn finish_readout_inner(&self,live:usize,stream:&Stream,enqueued:bool)->Result<(Vec<i32>,Vec<u32>)> {
        self.finish_readout_from(live,stream,enqueued,None)
    }
    fn finish_readout_from(&self,live:usize,stream:&Stream,enqueued:bool,
        target:Option<(&Buffer<u32>,&Buffer<i32>)>)->Result<(Vec<i32>,Vec<u32>)> {
        ensure!(live>0&&live<=self.capacity&&self.cuda.owns_stream(stream),"rejection readout rows/context");
        ensure!(!enqueued||self.readout.is_some(),"captured readout destination unavailable");
        if let Some((errors,ids))=target{ensure!(errors.len()>=live&&ids.len()>=live
            &&self.cuda.owns_buffer(errors)&&self.cuda.owns_buffer(ids),"readout target ownership/extent");}
        let (output,state)=if let Some(readout)=&self.readout {
            if !enqueued {
                if let Some((errors,ids))=target{self.enqueue_readout_from(errors,ids,live,stream)?;}
                else{self.enqueue_readout(stream)?;}
            }
            // Completion orders all GPU writes before reading coherent host
            // words. The allocation stays alive for this sampler's lifetime.
            stream.synchronize()?;
            let words=readout.read()?;
            ensure!(words[..live*5].chunks_exact(5).all(|r|r[0]==0),"invalid categorical logits or probability mass");
            ensure!(words[..live*5].chunks_exact(5).all(|r|r[1]<VOCAB as u32),"categorical output outside vocabulary");
            ensure!(words[..live*5].chunks_exact(5).all(|r|r[2]==0),"invalid rejection probability mass");
            (words.chunks_exact(5).map(|r|r[3]as i32).collect(),words.chunks_exact(5).map(|r|r[4]).collect())
        }else {
            if let Some((errors,ids))=target{
                stream.synchronize()?;
                ensure!(errors.download()?[..live].iter().all(|&e|e==0),"invalid categorical logits or probability mass");
                ensure!(ids.download()?[..live].iter().all(|&id|(0..VOCAB as i32).contains(&id)),"categorical output outside vocabulary");
            }else{self.target.finish(live,stream)?;}
            ensure!(self.errors.download()?[..live].iter().all(|&e|e==0),"invalid rejection probability mass");
            (self.output.download()?,self.state.download()?)
        };
        ensure!(output[..live].iter().all(|&x|(0..VOCAB as i32).contains(&x))
            &&state[..live].iter().all(|&s|s<=2),"invalid rejection readout");
        Ok((output,state))
    }
    pub(crate) fn sample(
        &mut self,
        logits: &Buffer<f32>,
        q: &Buffer<f64>,
        proposals: &[u32],
        live: usize,
        seed: u64,
        position: u64,
        stream: &Stream,
    ) -> Result<Draw> {
        ensure!(
            position.checked_add(live as u64).is_some(),
            "rejection RNG counter overflow"
        );
        let uniforms = |domain| {
            (0..live)
                .map(|r| uniform(seed, domain, position + r as u64))
                .collect::<Vec<_>>()
        };
        self.explicit(
            logits,
            q,
            proposals,
            &uniforms(ACCEPT_DOMAIN),
            &uniforms(RESIDUAL_DOMAIN),
            &uniforms(TARGET_DOMAIN),
            stream,
        )
    }
    pub(crate) fn sample_block(
        &mut self,
        logits: &Buffer<f32>,
        q: &Buffer<f64>,
        proposals: &[u32],
        live: usize,
        seed: u64,
        position: u64,
        stream: &Stream,
    ) -> Result<Draw> {
        ensure!(
            position.checked_add(live as u64).is_some(),
            "block RNG counter overflow"
        );
        let uniforms = |domain| {
            (0..live)
                .map(|r| uniform(seed, domain, position + r as u64))
                .collect::<Vec<_>>()
        };
        self.explicit_rule(
            logits,
            q,
            proposals,
            &uniforms(ACCEPT_DOMAIN),
            &uniforms(RESIDUAL_DOMAIN),
            &uniforms(TARGET_DOMAIN),
            stream,
            true,
        )
    }
    /// Explicit independently supplied draws support GPU reference/frequency
    /// qualification. A final row without a proposal supplies the bonus token.
    pub(crate) fn explicit(
        &mut self,
        logits: &Buffer<f32>,
        q: &Buffer<f64>,
        proposals: &[u32],
        accept: &[f64],
        residual: &[f64],
        bonus: &[f64],
        stream: &Stream,
    ) -> Result<Draw> {
        self.explicit_rule(logits, q, proposals, accept, residual, bonus, stream, false)
    }
    fn enqueue_rule(&mut self,logits:&Buffer<f32>,q:&Buffer<f64>,live:usize,drafts:usize,
        block:bool,stream:&Stream)->Result<()> {
        self.target.enqueue(logits, 0, live, 1., stream)?;
        self.target
            .probabilities(logits, 0, live, 1., &mut self.p, 0, stream)?;
        unsafe {
            self.cuda
                .zero_device(self.errors.ptr(), self.errors.bytes(), stream)?;
        }
        let mut p = self.p.ptr();
        let mut q = q.ptr();
        let mut proposed = self.proposals.ptr();
        let mut ua = self.accept_u.ptr();
        let mut b = self.target.device_output().ptr();
        let mut state = self.state.ptr();
        let mut out = self.output.ptr();
        let mut errors = self.errors.ptr();
        let mut d = drafts as i32;
        let mut sums = self.sums.ptr();
        let mut weights = self.block_weights.ptr();
        if block {
            let mut args = [
                (&mut p as *mut u64).cast(),
                (&mut q as *mut u64).cast(),
                (&mut proposed as *mut u64).cast(),
                (&mut weights as *mut u64).cast(),
                (&mut errors as *mut u64).cast(),
                (&mut d as *mut i32).cast(),
            ];
            unsafe {
                self.module.function("ca_block_weights")?.launch(
                    [1, 1, 1],
                    [32, 1, 1],
                    0,
                    stream,
                    &mut args,
                )?;
            }
            let mut args = [
                (&mut p as *mut u64).cast(),
                (&mut q as *mut u64).cast(),
                (&mut weights as *mut u64).cast(),
                (&mut sums as *mut u64).cast(),
                (&mut errors as *mut u64).cast(),
                (&mut d as *mut i32).cast(),
            ];
            unsafe {
                self.module.function("ca_block_partials")?.launch(
                    [127, live as u32, 1],
                    [256, 1, 1],
                    0,
                    stream,
                    &mut args,
                )?;
            }
            if self.fused_block_finish && self.parallel_block_finish && self.parallel_block_decide
                &&drafts<32&&self.progress.is_none()&&self.decision_control.is_none() {
                let mut ur=self.residual_u.ptr();let mut tiles=self.tiles.ptr();let mut fractions=self.fractions.ptr();
                let mut args=[(&mut p as *mut u64).cast(),(&mut q as *mut u64).cast(),
                    (&mut weights as *mut u64).cast(),(&mut sums as *mut u64).cast(),
                    (&mut proposed as *mut u64).cast(),(&mut ua as *mut u64).cast(),
                    (&mut b as *mut u64).cast(),(&mut ur as *mut u64).cast(),
                    (&mut state as *mut u64).cast(),(&mut out as *mut u64).cast(),
                    (&mut errors as *mut u64).cast(),(&mut tiles as *mut u64).cast(),
                    (&mut fractions as *mut u64).cast(),(&mut d as *mut i32).cast()];
                unsafe{self.module.function("ca_block_decide_finish_fused")?.launch([1,1,1],[256,1,1],0,stream,&mut args)?;}
                return Ok(());
            }
            let controls=self.progress.as_ref().and_then(|p|p.controls())
                .or_else(||self.decision_control.as_ref().map(|(s,o)|(s.ptr(),o.ptr())));
            if let Some((mut cs,mut co))=controls {
                unsafe {self.module.function("ca_block_decide")?.launch([1,1,1],[32,1,1],0,stream,
                    &mut[(&mut weights as *mut u64).cast(),(&mut sums as *mut u64).cast(),
                        (&mut proposed as *mut u64).cast(),(&mut ua as *mut u64).cast(),
                        (&mut b as *mut u64).cast(),(&mut cs as *mut u64).cast(),(&mut co as *mut u64).cast(),
                        (&mut errors as *mut u64).cast(),(&mut d as *mut i32).cast()])?;}
            }
            let mut progress=self.progress.as_ref().and_then(|p|p.pointer()).unwrap_or(0);
            let mut args = [
                (&mut weights as *mut u64).cast(),
                (&mut sums as *mut u64).cast(),
                (&mut proposed as *mut u64).cast(),
                (&mut ua as *mut u64).cast(),
                (&mut b as *mut u64).cast(),
                (&mut state as *mut u64).cast(),
                (&mut out as *mut u64).cast(),
                (&mut errors as *mut u64).cast(),
                (&mut d as *mut i32).cast(),
                (&mut progress as *mut u64).cast(),
            ];
            let name=if progress!=0 {
                ensure!(drafts<=5,"block progress supports at most five drafts");
                "ca_block_decide_progress"
            }else if self.parallel_block_decide && drafts < 32 {"ca_block_decide_rows"}
            else{"ca_block_decide"};
            unsafe {
                self.module.function(name)?.launch(
                    [1, 1, 1],
                    [32, 1, 1],
                    0,
                    stream,
                    &mut args[..if progress!=0{10}else{9}],
                )?;
            }
            if let Some((mut cs,mut co))=controls {
                let mut n=live as i32;
                unsafe{self.module.function("ca_block_progress_compare")?.launch([1,1,1],[32,1,1],0,stream,
                    &mut[(&mut cs as *mut u64).cast(),(&mut co as *mut u64).cast(),
                        (&mut state as *mut u64).cast(),(&mut out as *mut u64).cast(),
                        (&mut errors as *mut u64).cast(),(&mut n as *mut i32).cast()])?;}
            }
        } else {
            let mut args = [
                (&mut p as *mut u64).cast(),
                (&mut q as *mut u64).cast(),
                (&mut proposed as *mut u64).cast(),
                (&mut ua as *mut u64).cast(),
                (&mut b as *mut u64).cast(),
                (&mut state as *mut u64).cast(),
                (&mut out as *mut u64).cast(),
                (&mut errors as *mut u64).cast(),
                (&mut d as *mut i32).cast(),
            ];
            unsafe {
                self.module.function("ca_reject_decide")?.launch(
                    [live as u32, 1, 1],
                    [32, 1, 1],
                    0,
                    stream,
                    &mut args,
                )?;
            }
            let mut args = [
                (&mut p as *mut u64).cast(),
                (&mut q as *mut u64).cast(),
                (&mut state as *mut u64).cast(),
                (&mut sums as *mut u64).cast(),
                (&mut errors as *mut u64).cast(),
            ];
            unsafe {
                self.module.function("ca_reject_partials")?.launch(
                    [127, live as u32, 1],
                    [256, 1, 1],
                    0,
                    stream,
                    &mut args,
                )?;
            }
        }
        let mut ur = self.residual_u.ptr();
        let mut tiles = self.tiles.ptr();
        let mut fractions = self.fractions.ptr();
        let mut args = [
            (&mut sums as *mut u64).cast(),
            (&mut ur as *mut u64).cast(),
            (&mut state as *mut u64).cast(),
            (&mut tiles as *mut u64).cast(),
            (&mut fractions as *mut u64).cast(),
            (&mut errors as *mut u64).cast(),
        ];
        unsafe {
            self.module.function("ca_reject_tile")?.launch(
                [live as u32, 1, 1],
                [32, 1, 1],
                0,
                stream,
                &mut args,
            )?;
        }
        if block {
            let mut args = [
                (&mut p as *mut u64).cast(),
                (&mut q as *mut u64).cast(),
                (&mut weights as *mut u64).cast(),
                (&mut state as *mut u64).cast(),
                (&mut tiles as *mut u64).cast(),
                (&mut fractions as *mut u64).cast(),
                (&mut out as *mut u64).cast(),
                (&mut errors as *mut u64).cast(),
            ];
            unsafe {
                self.module.function(if self.parallel_block_finish {"ca_block_finish_raking"}else{"ca_block_finish"})?.launch(
                    [live as u32, 1, 1],
                    [256, 1, 1],
                    0,
                    stream,
                    &mut args,
                )?;
            }
        } else {
            let mut args = [
                (&mut p as *mut u64).cast(),
                (&mut q as *mut u64).cast(),
                (&mut state as *mut u64).cast(),
                (&mut tiles as *mut u64).cast(),
                (&mut fractions as *mut u64).cast(),
                (&mut out as *mut u64).cast(),
                (&mut errors as *mut u64).cast(),
            ];
            unsafe {
                self.module.function("ca_reject_finish")?.launch(
                    [live as u32, 1, 1],
                    [256, 1, 1],
                    0,
                    stream,
                    &mut args,
                )?;
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn explicit_rule(
        &mut self,
        logits: &Buffer<f32>,
        q: &Buffer<f64>,
        proposals: &[u32],
        accept: &[f64],
        residual: &[f64],
        bonus: &[f64],
        stream: &Stream,
        block: bool,
    ) -> Result<Draw> {
        let live = bonus.len();
        let drafts = proposals.len();
        ensure!(
            !block || drafts + 1 == live,
            "block verification requires a bonus row"
        );
        ensure!(
            live > 0
                && live <= self.capacity
                && (drafts == live || drafts + 1 == live)
                && logits.len() >= live * VOCAB
                && q.len() >= drafts * VOCAB
                && self.cuda.owns_buffer(logits)
                && self.cuda.owns_buffer(q)
                && self.cuda.owns_stream(stream)
                && accept.len() == live
                && residual.len() == live
                && [accept, residual, bonus]
                    .iter()
                    .all(|a| a.iter().all(|u| u.is_finite() && (0. ..1.).contains(u)))
                && proposals.iter().all(|&x| x < (VOCAB as u32)),
            "rejection shape/uniform/context"
        );
        let mut ids = vec![0; self.capacity];
        for (i, &v) in proposals.iter().enumerate() {
            ids[i] = v as i32;
        }
        let mut a = vec![0.; self.capacity];
        a[..live].copy_from_slice(accept);
        let mut r = vec![0.; self.capacity];
        r[..live].copy_from_slice(residual);
        self.proposals.upload(&ids)?;
        self.accept_u.upload(&a)?;
        self.residual_u.upload(&r)?;
        self.target.upload_uniforms(bonus)?;
        stream.join_default()?;
        if let Some(staging)=&self.staged_inputs {
            unsafe{staging.flush(Some(stream))?;}
        }
        let replay=self.graph_replay && self.progress.is_none() && self.decision_control.is_none()
            && self.shadow_expected.is_none();
        let captured_readout=replay&&self.graph_readout&&self.readout.is_some();
        if replay {
            let slot=(live-1)*2+usize::from(block);
            let readout_key=if captured_readout{self.readout.as_ref().unwrap().pointer()}else{0};
            let key=(logits.ptr(),q.ptr(),live,drafts,block,stream.raw()as usize,readout_key);
            if self.graphs[slot].as_ref().is_none_or(|g|g.key!=key) {
                // Inputs were uploaded and ordered before capture; all work in
                // this closure is GPU-only and uses fixed sampler-owned storage.
                self.graphs[slot]=None;
                let graph=stream.capture(||{
                    self.enqueue_rule(logits,q,live,drafts,block,stream)?;
                    if captured_readout{self.enqueue_readout(stream)?;}
                    Ok(())
                })?;
                self.graphs[slot]=Some(SamplerGraph{key,graph});
            }
            unsafe{self.graphs[slot].as_ref().unwrap().graph.launch(stream)?;}
        }else{self.enqueue_rule(logits,q,live,drafts,block,stream)?;}
        // Shadow diagnostics use the same completed buffers, but never replay
        // a graph because their optional destinations can change independently.
        let mut p=self.p.ptr();let mut q=q.ptr();let mut proposed=self.proposals.ptr();
        let mut weights=self.block_weights.ptr();let mut errors=self.errors.ptr();
        let mut d=drafts as i32;let mut sums=self.sums.ptr();
        let (output,state)=self.finish_readout_inner(live,stream,captured_readout)?;
        let accepted_prefix =
            block.then(|| state[..drafts].iter().take_while(|&&x| x == 1).count());
        ensure!(
            !block
                || state[..drafts]
                    .iter()
                    .enumerate()
                    .all(|(i, &x)| (x == 1) == (i < accepted_prefix.unwrap())),
            "block acceptance is not a prefix"
        );
        // The actual draw has finished. Reuse scratch for a read-only shadow
        // calculation; consume no RNG and never feed these metrics into policy.
        let mut first_proposal_overlap=None;
        let mut local_proposal_fit=None;
        let acceptance_expected_lengths = if let Some(expected) = &self.shadow_expected {
            ensure!(drafts + 1 == live, "acceptance shadow requires a bonus row");
            let mut args = [
                (&mut p as *mut u64).cast(), (&mut q as *mut u64).cast(),
                (&mut proposed as *mut u64).cast(), (&mut weights as *mut u64).cast(),
                (&mut errors as *mut u64).cast(), (&mut d as *mut i32).cast(),
            ];
            unsafe { self.module.function("ca_block_weights")?.launch([1,1,1],[32,1,1],0,stream,&mut args)?; }
            let mut args = [
                (&mut p as *mut u64).cast(), (&mut q as *mut u64).cast(),
                (&mut weights as *mut u64).cast(), (&mut sums as *mut u64).cast(),
                (&mut errors as *mut u64).cast(), (&mut d as *mut i32).cast(),
            ];
            unsafe { self.module.function("ca_block_partials")?.launch([127,live as u32,1],[256,1,1],0,stream,&mut args)?; }
            let mut metrics = expected.ptr();
            let mut args = [
                (&mut p as *mut u64).cast(), (&mut q as *mut u64).cast(),
                (&mut proposed as *mut u64).cast(), (&mut weights as *mut u64).cast(),
                (&mut sums as *mut u64).cast(), (&mut metrics as *mut u64).cast(),
                (&mut errors as *mut u64).cast(), (&mut d as *mut i32).cast(),
            ];
            unsafe { self.module.function("ca_acceptance_expected_lengths")?.launch([1,1,1],[32,1,1],0,stream,&mut args)?; }
            if drafts>0 {
                let mut args=[(&mut p as *mut u64).cast(),(&mut q as *mut u64).cast(),
                    (&mut sums as *mut u64).cast(),(&mut errors as *mut u64).cast(),(&mut d as *mut i32).cast()];
                unsafe{self.module.function("ca_acceptance_local_partials")?.launch([127,drafts as u32,1],[256,1,1],0,stream,&mut args)?;}
            }
            let mut args=[(&mut p as *mut u64).cast(),(&mut q as *mut u64).cast(),
                (&mut proposed as *mut u64).cast(),(&mut sums as *mut u64).cast(),
                (&mut metrics as *mut u64).cast(),(&mut errors as *mut u64).cast(),(&mut d as *mut i32).cast()];
            unsafe{self.module.function("ca_acceptance_local_finish")?.launch([1,1,1],[32,1,1],0,stream,&mut args)?;}
            stream.synchronize()?;
            ensure!(self.errors.download()?[..live].iter().all(|&e| e == 0), "invalid shadow probability mass");
            ensure!(self.output.download()? == output && self.state.download()? == state,
                "acceptance shadow changed the completed draw");
            let metrics=expected.download()?;let overlap=metrics[live*2];
            ensure!(overlap.is_finite()&&(0. ..=1.).contains(&overlap),"invalid marginal proposal overlap");
            first_proposal_overlap=Some(overlap);
            let local=metrics[live*2+1..live*2+1+drafts].to_vec();
            ensure!(local.iter().all(|x|x.is_finite()&&(0. ..=1.).contains(x))
                && local.first().is_none_or(|x|(x-overlap).abs()<1e-10),"local proposal overlap mismatch");
            let rb=metrics[live*2+1+drafts];let product=metrics[live*2+2+drafts];
            ensure!([rb,product].iter().all(|x|x.is_finite()&&(1. ..=live as f64).contains(x)),
                "invalid local proposal score");
            local_proposal_fit=Some((local,rb,product));
            Some(metrics[..live*2].chunks_exact(2).map(|v|[v[0],v[1]]).collect())
        } else { None };
        Ok(Draw {
            accepted_prefix,
            acceptance_expected_lengths,
            first_proposal_overlap,
            local_proposal_fit,
            ids: output[..live].iter().map(|&x| x as u32).collect(),
            accepted: state[..live].iter().map(|&x| x == 1).collect(),
        })
    }
}

/// Integer-only native transfer and error-propagation screen. No probability
/// or model arithmetic is duplicated on the CPU.
pub fn probe_local_readout(core:&std::path::Path)->Result<serde_json::Value>{
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;let core=std::fs::read(core)?;
    let mut sampler=Rejection::new(&cuda,&core,6)?;let mut checks=0;
    for live in [2usize,3,5,6]{
        sampler.local_targets[live-1]=Some(crate::sharded_categorical_probe::CompactSampler::new(&cuda,&core,live,0)?);
        for mode in 0..3{
            sampler.readout=match mode{0=>None,1=>Some(Readout::Device(cuda.allocate(30)?)),
                _=>Some(Readout::Mapped(cuda.mapped_host(120)?))};
            let graph=if mode>0{Some(stream.capture(||sampler.enqueue_sharded_readout(live,true,&stream))?)}else{None};
            for corrupt in 0..6{
                stream.synchronize()?;
                // Stale legacy target state must neither mask a current error
                // nor reject a valid compact draw. Live buffers can be smaller
                // than the surrounding six-row rejection workspace.
                unsafe{sampler.target.device_errors().view(0,6)?}.upload(&vec![if corrupt==0{1}else{0};6])?;
                unsafe{sampler.target.device_output().view(0,6)?}.upload(&vec![if corrupt==0{-1}else{19};6])?;
                let mut errors=vec![0;live];let mut ids=vec![31;live];
                match corrupt{1=>errors[live-1]=8,2=>ids[live-1]=-1,3=>ids[live-1]=VOCAB as i32,_=>{}}
                let target=sampler.local_targets[live-1].as_ref().unwrap();
                unsafe{target.errors().view(0,live)?}.upload(&errors)?;
                unsafe{target.output().view(0,live)?}.upload(&ids)?;
                sampler.errors.upload(&vec![if corrupt==4{4}else{0};6])?;
                sampler.output.upload(&vec![42;6])?;sampler.state.upload(&vec![1;6])?;
                stream.join_default()?;
                if let Some(graph)=&graph{unsafe{graph.launch(&stream)?;}}
                let result=sampler.finish_sharded_readout(live,true,&stream,graph.is_some());
                ensure!(result.is_err()==(1..=4).contains(&corrupt),"compact target readout error source/extent");
                if let Ok((output,state))=result{ensure!(output[..live]==vec![42;live]&&state[..live]==vec![1;live],
                    "compact target readout changed accepted state");}
                checks+=1;
            }
        }
    }
    Ok(serde_json::json!({"all_correct":true,"checks":checks,"live_rows":[2,3,5,6],
        "scope":"Integer-only actual local-target readout wiring; stale legacy errors, current invalid errors/IDs, rejection errors, recovery, narrow buffers and captured copy/mapped readouts. No model arithmetic."}))
}

pub fn probe_readout(core:&std::path::Path)->Result<serde_json::Value> {
    use serde_json::json;
    use std::time::Instant;
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;let core=std::fs::read(core)?;
    let mut reports=Vec::new();
    for capacity in [3usize,6,16] {
        let mut sampler=Rejection::new(&cuda,&core,capacity)?;
        let mut mapped=Some(Readout::Mapped(cuda.mapped_host(capacity*20)?));
        let mut copied=Some(Readout::Device(cuda.allocate(capacity*5)?));
        let mut checks=0;
        for live in [1,capacity] {
            for corrupt in 0..8 {
                stream.synchronize()?;
                let mut te=vec![0u32;capacity];let mut ti:Vec<i32>=(0..capacity).map(|x|x as i32+100).collect();
                let mut re=vec![0u32;capacity];let mut ids:Vec<i32>=(0..capacity).map(|x|x as i32+200).collect();let mut st=vec![1u32;capacity];
                // The last case injects bad data only outside the live range.
                let i=if corrupt==5{capacity-1}else{live-1};
                match corrupt {1=>te[i]=1,2=>ti[i]=-1,3=>re[i]=256,4=>st[i]=99,5=>{te[i]=1;ti[i]=-1;re[i]=2;ids[i]=-1;st[i]=99;},6=>ids[i]=-1,7=>ti[i]=VOCAB as i32,_=>{}}
                unsafe{sampler.target.device_errors().view(0,capacity)?}.upload(&te)?;
                unsafe{sampler.target.device_output().view(0,capacity)?}.upload(&ti)?;
                sampler.errors.upload(&re)?;sampler.output.upload(&ids)?;sampler.state.upload(&st)?;
                sampler.readout=None;let ordinary=sampler.finish_readout(live,&stream);
                for slot in [&mut copied,&mut mapped] {
                    std::mem::swap(&mut sampler.readout,slot);
                    let got=sampler.finish_readout(live,&stream);
                    ensure!(match(&ordinary,&got){(Ok(a),Ok(b))=>a==b,(Err(a),Err(b))=>a.to_string()==b.to_string(),_=>false},"native readout differs");
                    std::mem::swap(&mut sampler.readout,slot);checks+=1;
                }
                ensure!(ordinary.is_err()==(corrupt!=0&&!(corrupt==5&&live<capacity)),"native negative-control admission");
            }
        }
        unsafe{sampler.target.device_errors().view(0,capacity)?}.zero()?;
        unsafe{sampler.target.device_output().view(0,capacity)?}.upload(&vec![11;capacity])?;
        sampler.errors.zero()?;sampler.output.upload(&vec![12;capacity])?;sampler.state.upload(&vec![1;capacity])?;
        let mut timing=[Vec::new(),Vec::new(),Vec::new()];
        for repeat in 0..6 {
            for mode in if repeat%2==0{[0,1,2]}else{[2,1,0]} {
                if mode==1{std::mem::swap(&mut sampler.readout,&mut copied);}
                if mode==2{std::mem::swap(&mut sampler.readout,&mut mapped);}
                for _ in 0..8{sampler.finish_readout(capacity,&stream)?;}
                let began=Instant::now();for _ in 0..256{sampler.finish_readout(capacity,&stream)?;}
                timing[mode].push(began.elapsed().as_secs_f64()*1e6/256.);
                if mode==1{std::mem::swap(&mut sampler.readout,&mut copied);}
                if mode==2{std::mem::swap(&mut sampler.readout,&mut mapped);}
            }
        }
        reports.push(json!({"capacity":capacity,"integer_and_error_checks":checks,"timing_us":timing}));
    }
    Ok(json!({"all_correct":true,"reports":reports,"modes":["five_copies","packed_copy","mapped"],
        "scope":"Native Rust readout with identical integer outputs and error messages, invalid target IDs/errors and rejection flags, live-prefix bounds. Six reversed timing pairs of256readouts; no model arithmetic or whole-model speed claim."}))
}

/// Same GPU kernels, uniforms and exact actual-q correction, with graph replay.
/// Inputs are generated by the existing CUDA fixture, never a CPU probability oracle.
pub fn probe_graph(core:&std::path::Path)->Result<serde_json::Value> {
    use serde_json::json;
    use sha2::{Digest,Sha256};
    use std::time::Instant;
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let core=std::fs::read(core)?;let module=cuda.module(&core)?;
    let logits0=cuda.allocate::<f32>(6*VOCAB)?;
    let logits1=cuda.allocate::<f32>(6*VOCAB)?;
    let q_logits=cuda.allocate::<f32>(6*VOCAB)?;
    let mut q=cuda.allocate::<f64>(6*VOCAB)?;
    let fill=|buffer:&Buffer<f32>,phase:i32|->Result<()> {
        for row in 0..6 {for rank in 0..4 {
            let mut ptr=buffer.ptr()+((row*VOCAB+rank*32320)*4)as u64;
            let mut r=rank as i32;let mut rows=1i32;let mut p=phase;
            unsafe{module.function("ca_sharded_categorical_fixture")?.launch([127,1,1],[256,1,1],0,&stream,
                &mut[(&mut ptr as *mut u64).cast(),(&mut r as *mut i32).cast(),
                    (&mut rows as *mut i32).cast(),(&mut p as *mut i32).cast()])?;}
        }}Ok(())
    };
    let mut draft=Categorical::new(&cuda,&core,6)?;
    let mapped_probe=std::env::var("CANADA_SCRATCH_SAMPLER_MAPPED_SCREEN").as_deref()==Ok("1");
    let readout_probe=std::env::var("CANADA_SCRATCH_SAMPLER_GRAPH_READOUT_SCREEN").as_deref()==Ok("1");
    ensure!(!(mapped_probe&&readout_probe),"one control policy per sampler screen");
    let mut ordinary=Rejection::with_staging(&cuda,&core,6,mapped_probe||readout_probe)?;
    let mut staged=Rejection::with_controls(&cuda,&core,6,if mapped_probe{2}else{1})?;
    if readout_probe{ordinary.graph_readout=false;staged.graph_readout=true;}
    let mut records=Vec::new();
    let digest=|sampler:&Rejection,live:usize|->Result<String> {
        Ok(format!("{:x}",Sha256::digest(unsafe{cuda.read_device(sampler.p.ptr(),live*VOCAB*8)}?)))
    };
    for phase in 0..4 {
        // Alternating pointers exercise replacement of a captured input binding;
        // phase changes also invalidate any accidental reuse of probability data.
        let logits=if phase%2==0{&logits0}else{&logits1};
        fill(logits,phase)?;fill(&q_logits,if phase==2{1}else{0})?;
        let proposals=draft.sample_with_uniforms(&q_logits,0,0.7,&[0.13,0.29,0.47,0.71,0.93,0.61],&stream)?;
        draft.probabilities(&q_logits,0,6,0.7,&mut q,0,&stream)?;
        stream.synchronize()?;
        for live in [1usize,3,5,6] {for block in [false,true] {
            let mut checks=0;
            for seed in [10u64,29,733] {
                let mut baseline=None;
                for mode in [0usize,1,2,3,3] {
                    let sampler=if mode<2{&mut ordinary}else{&mut staged};
                    sampler.graph_replay=mode%2==1;
                    let result=if block{sampler.sample_block(logits,&q,&proposals[..live-1],live,seed,173,&stream)?}
                        else{sampler.sample(logits,&q,&proposals[..live-1],live,seed,173,&stream)?};
                    let got=json!({"ids":result.ids,"accepted":result.accepted,"prefix":result.accepted_prefix,
                        "target_probability_sha256":digest(sampler,live)?});
                    if let Some(expected)=&baseline{ensure!(expected==&got,"rejection graph draw/probability mismatch");checks+=1;}
                    else{baseline=Some(got);}
                }
            }
            let mut timing=Vec::new();
            for mode in [0usize,1,2,3,3,2,1,0] {
                let sampler=if mode<2{&mut ordinary}else{&mut staged};
                sampler.graph_replay=mode%2==1;
                for _ in 0..4 {if block{sampler.sample_block(logits,&q,&proposals[..live-1],live,17,173,&stream)?;}
                    else{sampler.sample(logits,&q,&proposals[..live-1],live,17,173,&stream)?;}}
                let at=Instant::now();
                for seed in 0..64u64 {if block{sampler.sample_block(logits,&q,&proposals[..live-1],live,seed,173,&stream)?;}
                    else{sampler.sample(logits,&q,&proposals[..live-1],live,seed,173,&stream)?;}}
                timing.push(json!({"mode":mode,"enabled":mode%2==1,"staged":mode>=2,"us_per_draw":at.elapsed().as_secs_f64()*1e6/64.}));
            }
            records.push(json!({"phase":phase,"live":live,"block":block,"exact_comparisons":checks,"timing":timing}));
        }}
    }
    let mut invalid=Vec::new();
    for phase in [4,5,6] {
        fill(&logits0,phase)?;stream.synchronize()?;
        for live in [1usize,6] {
            let mut errors=Vec::new();
            for mode in 0usize..4 {
                let sampler=if mode<2{&mut ordinary}else{&mut staged};
                sampler.graph_replay=mode%2==1;
                let result=sampler.sample_block(&logits0,&q,&vec![17;live-1],live,19,177,&stream);
                ensure!(result.is_err(),"invalid logits silently accepted");
                errors.push(result.err().unwrap().to_string());
            }
            ensure!(errors.iter().all(|e|e==&errors[0]),"graph/input error propagation differs");
            invalid.push(json!({"phase":phase,"live":live,"error":errors[0]}));
        }
    }
    // Replace invalid contents at the same address; stale error flags must clear.
    fill(&logits0,0)?;stream.synchronize()?;staged.graph_replay=true;
    staged.sample_block(&logits0,&q,&[17;5],6,21,177,&stream)?;
    let mut readout_rebind_checks=0;
    if readout_probe {
        ordinary.graph_replay=true;
        let expected=ordinary.sample_block(&logits0,&q,&[17;5],6,21,177,&stream)?;
        for mode in [1,0,2,2] {
            staged.set_readout_for_probe(mode,&stream)?;
            let got=staged.sample_block(&logits0,&q,&[17;5],6,21,177,&stream)?;
            ensure!(got.ids==expected.ids&&got.accepted==expected.accepted
                &&got.accepted_prefix==expected.accepted_prefix,"readout rebind changed draw");
            readout_rebind_checks+=1;
        }
    }
    let graphs=ordinary.graphs.iter().flatten().count()+staged.graphs.iter().flatten().count();
    Ok(json!({"all_correct":true,"records":records,"invalid_cases":invalid,"cached_graphs":graphs,"mapped_input_screen":mapped_probe,"graph_readout_screen":readout_probe,"readout_rebind_checks":readout_rebind_checks,
        "input_modes":if mapped_probe{["coalesced_copy","mapped_cpu_controls"]}else if readout_probe{["coalesced_copy","coalesced_copy"]}else{["ordinary_copies","coalesced_copy"]},
        "scope":"Native eager/graph and ordinary/coalesced-input reversed order, fixed allocations; GPU-generated broad, sparse, point-mass and extreme logits.32 geometry/rule cases,384 exact draw+probability comparisons; pointer rebinding and invalid-value recovery. Same kernels and FP64 actual-q correction. Timing includes ordinary metadata uploads and readout; no model-speed claim."}))
}
