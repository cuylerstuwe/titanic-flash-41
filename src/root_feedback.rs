//! Finite foreground experiment: root-only temperature selected from prior
//! target feedback. Target p and emitted tokens are never changed here.
use crate::{categorical::Categorical,cuda::{Buffer,Cuda,Module,Stream},sampling::VOCAB};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use std::time::Instant;

pub(crate) struct RootFeedback {
    cuda:Cuda,module:Module,alternate:Categorical,q:Buffer<f64>,sums:Buffer<f64>,
    metrics:Buffer<f64>,choice:Buffer<u32>,errors:Buffer<u32>,
    enabled:bool,rounds:usize,warmer:bool,records:Vec<Value>,
}
impl RootFeedback {
    pub(crate) fn new(cuda:&Cuda,core:&[u8])->Result<Self> {
        ensure!(std::env::var("CANADA_SCRATCH_ROOT_FEEDBACK_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_QUERY_REUSE_SCREEN").as_deref()==Ok("1"),
            "root feedback requires isolated foreground admission");
        let mut alternate=Categorical::new(cuda,core,1)?;
        // Enqueue computes normalization as well as categorical scratch. Its
        // sampled token is discarded; this fixed dummy uniform consumes no RNG.
        alternate.upload_uniforms(&[0.5])?;
        Ok(Self{cuda:cuda.clone(),module:cuda.module(core)?,alternate,
            q:cuda.allocate(VOCAB)?,sums:cuda.allocate(254)?,metrics:cuda.allocate(3)?,
            choice:cuda.allocate(2)?,errors:cuda.allocate(1)?,
            enabled:false,rounds:0,warmer:false,records:Vec::new()})
    }
    pub(crate) fn reset(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        stream.synchronize()?;
        self.metrics.upload(&[0.;3])?;self.choice.upload(&[0;2])?;
        stream.join_default()?;
        self.enabled=enabled;self.rounds=0;self.warmer=false;self.records.clear();Ok(())
    }
    pub(crate) fn report(&self)->Value {
        json!({"enabled":self.enabled,"neural_rounds":self.rounds,"observations":self.records,
            "interval_rounds":8,"ema_new_weight":0.5,"hysteresis":0.01,
            "policy":"Past-root integrated overlap selects NEXT round root T.7 or T1; suffix T.7, actual-q block rejection unchanged."})
    }
    pub(crate) fn enabled(&self)->bool {self.enabled}
    /// Read-only opportunity screen. Reuse the established GPU overlap
    /// kernels; their policy choice is discarded and never affects a draw.
    pub(crate) fn observe_pair_for_query_probe(&mut self,p:&Buffer<f64>,fresh:&Buffer<f64>,
        reused:&Buffer<f64>,stream:&Stream)->Result<[f64;2]> {
        ensure!(std::env::var("CANADA_SCRATCH_QUERY_REUSE_SCREEN").as_deref()==Ok("1")
            &&[p.len(),fresh.len(),reused.len()].iter().all(|&n|n==VOCAB)
            &&self.cuda.owns_buffer(p)&&self.cuda.owns_buffer(fresh)&&self.cuda.owns_buffer(reused),
            "query root-pair diagnostic operands");
        self.reset(false,stream)?;
        unsafe{self.cuda.zero_device(self.errors.ptr(),self.errors.bytes(),stream)?;}
        let mut pp=p.ptr();let mut errors=self.errors.ptr();let mut rows=1i32;
        for(i,mut q)in [fresh.ptr(),reused.ptr()].into_iter().enumerate(){
            let mut sums=self.sums.ptr()+(i*127*8)as u64;
            unsafe{self.module.function("ca_acceptance_local_partials")?.launch([127,1,1],[256,1,1],0,stream,
                &mut[(&mut pp as *mut u64).cast(),(&mut q as *mut u64).cast(),
                    (&mut sums as *mut u64).cast(),(&mut errors as *mut u64).cast(),(&mut rows as *mut i32).cast()])?;}
        }
        let mut sums=self.sums.ptr();let mut metrics=self.metrics.ptr();let mut choice=self.choice.ptr();
        unsafe{self.module.function("ca_root_feedback_finish")?.launch([1,1,1],[32,1,1],0,stream,
            &mut[(&mut sums as *mut u64).cast(),(&mut metrics as *mut u64).cast(),
                (&mut choice as *mut u64).cast(),(&mut errors as *mut u64).cast()])?;}
        stream.synchronize()?;
        ensure!(self.errors.download()?[0]==0,"query root-pair probability error");
        let m=self.metrics.download()?;
        ensure!(m[..2].iter().all(|x|x.is_finite()&&(0. ..=1.+1e-10).contains(x)),"query root-pair overlap range");
        Ok([m[0],m[1]])
    }
    pub(crate) fn observe(&mut self,p:&Buffer<f64>,actual_q:&Buffer<f64>,logits:&Buffer<f32>,
        schedule:usize,position:u64,stream:&Stream)->Result<Option<bool>> {
        ensure!(self.enabled && [0,3].contains(&schedule) && (schedule==3)==self.warmer
            && [p.len(),actual_q.len(),logits.len()].iter().all(|n|*n>=VOCAB)
            && self.cuda.owns_buffer(p) && self.cuda.owns_buffer(actual_q)
            && self.cuda.owns_buffer(logits) && self.cuda.owns_stream(stream),"root feedback operands");
        self.rounds+=1;
        if self.rounds%8!=1 {return Ok(None);}
        let started=Instant::now();
        let temperature=if self.warmer {0.7}else{1.};
        self.alternate.enqueue(logits,0,1,temperature,stream)?;
        self.alternate.probabilities(logits,0,1,temperature,&mut self.q,0,stream)?;
        let _unused_checked_draw=self.alternate.finish(1,stream)?;
        unsafe {self.cuda.zero_device(self.errors.ptr(),self.errors.bytes(),stream)?;}
        let mut pp=p.ptr();let mut errors=self.errors.ptr();let mut rows=1i32;
        let probabilities=if self.warmer {[self.q.ptr(),actual_q.ptr()]}else{[actual_q.ptr(),self.q.ptr()]};
        for(i,mut q)in probabilities.into_iter().enumerate() {
            let mut sums=self.sums.ptr()+(i*127*8)as u64;
            unsafe {self.module.function("ca_acceptance_local_partials")?.launch([127,1,1],[256,1,1],0,stream,
                &mut[(&mut pp as *mut u64).cast(),(&mut q as *mut u64).cast(),
                    (&mut sums as *mut u64).cast(),(&mut errors as *mut u64).cast(),(&mut rows as *mut i32).cast()])?;}
        }
        let mut sums=self.sums.ptr();let mut metrics=self.metrics.ptr();let mut choice=self.choice.ptr();
        unsafe {self.module.function("ca_root_feedback_finish")?.launch([1,1,1],[32,1,1],0,stream,
            &mut[(&mut sums as *mut u64).cast(),(&mut metrics as *mut u64).cast(),
                (&mut choice as *mut u64).cast(),(&mut errors as *mut u64).cast()])?;}
        stream.synchronize()?;
        ensure!(self.errors.download()?[0]==0,"invalid root-feedback probability mass");
        let decision=self.choice.download()?;let metrics=self.metrics.download()?;
        ensure!(decision[0]<=1 && metrics.iter().all(|x|x.is_finite()),"root-feedback readout");
        let next=decision[0]==1;let changed=next!=self.warmer;
        self.records.push(json!({"round":self.rounds,"position":position,"current_warmer":self.warmer,
            "next_warmer":next,"overlap_cold":metrics[0],"overlap_hot":metrics[1],"ema_delta":metrics[2],
            "observation":decision[1],"seconds":started.elapsed().as_secs_f64()}));
        self.warmer=next;Ok(changed.then_some(next))
    }
}
