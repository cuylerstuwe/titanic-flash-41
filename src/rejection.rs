//! GPU rejection sampling for a stochastic autoregressive draft. No acceptance
//! shortcut, temperature adjustment, approximate match or host probability math.
use crate::{
    categorical::{ACCEPT_DOMAIN, Categorical, RESIDUAL_DOMAIN, TARGET_DOMAIN, uniform},
    cuda::{Buffer, Cuda, MappedHost, Module, Stream},
    sampling::VOCAB,
};
use anyhow::{Result, ensure};
pub(crate) struct Rejection {
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
    decision_control: Option<(Buffer<u32>, Buffer<i32>)>,
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
        ensure!(std::env::var("CANADA_SCRATCH_ROOT_FEEDBACK_SCREEN").as_deref()==Ok("1"),
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
        ensure!((1..=128).contains(&capacity), "rejection capacity");
        let parallel_block_decide = match std::env::var("CANADA_SCRATCH_PARALLEL_BLOCK_DECIDE") {
            Ok(v) => { ensure!(v == "0" || v == "1", "invalid block decision policy"); v == "1" },
            Err(std::env::VarError::NotPresent) => true,
            Err(e) => return Err(e.into()),
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
        Ok(Self {
            cuda: cuda.clone(),
            module: cuda.module(core)?,
            capacity,
            readout,
            parallel_block_decide,
            decision_control,
            progress:if capacity==6&&std::env::var("CANADA_SCRATCH_BLOCK_PROGRESS_SCREEN").as_deref()==Ok("1") {
                Some(crate::block_progress::BlockProgress::new(cuda,core)?)
            }else{None},
            target: Categorical::new(cuda, core, capacity)?,
            p: cuda.allocate(capacity * VOCAB)?,
            proposals: cuda.allocate(capacity)?,
            accept_u: cuda.allocate(capacity)?,
            residual_u: cuda.allocate(capacity)?,
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
        ensure!(std::env::var("CANADA_SCRATCH_SAMPLER_READOUT_SCREEN").as_deref()==Ok("1")
            && self.cuda.owns_stream(stream),"isolated sampler readout screen");
        stream.synchronize()?;
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
        ensure!(live>0&&live<=self.capacity&&self.cuda.owns_stream(stream),"rejection readout rows/context");
        let (output,state)=if let Some(readout)=&self.readout {
            let mut te=self.target.device_errors().ptr();let mut ti=self.target.device_output().ptr();
            let mut re=self.errors.ptr();let mut ids=self.output.ptr();let mut states=self.state.ptr();
            let mut packed=readout.pointer();let mut cap=self.capacity as i32;
            let mut args=[(&mut te as *mut u64).cast(),(&mut ti as *mut u64).cast(),
                (&mut re as *mut u64).cast(),(&mut ids as *mut u64).cast(),
                (&mut states as *mut u64).cast(),(&mut packed as *mut u64).cast(),(&mut cap as *mut i32).cast()];
            unsafe{self.module.function("ca_sampler_readout")?.launch([1,1,1],[128,1,1],0,stream,&mut args)?;}
            // Completion orders all GPU writes before reading coherent host
            // words. The allocation stays alive for this sampler's lifetime.
            stream.synchronize()?;
            let words=readout.read()?;
            ensure!(words[..live*5].chunks_exact(5).all(|r|r[0]==0),"invalid categorical logits or probability mass");
            ensure!(words[..live*5].chunks_exact(5).all(|r|r[1]<VOCAB as u32),"categorical output outside vocabulary");
            ensure!(words[..live*5].chunks_exact(5).all(|r|r[2]==0),"invalid rejection probability mass");
            (words.chunks_exact(5).map(|r|r[3]as i32).collect(),words.chunks_exact(5).map(|r|r[4]).collect())
        }else {
            self.target.finish(live,stream)?;
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
                self.module.function("ca_block_finish")?.launch(
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
        let (output,state)=self.finish_readout(live,stream)?;
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
