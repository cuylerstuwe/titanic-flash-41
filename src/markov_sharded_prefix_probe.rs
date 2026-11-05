//! Finite cost screen for early stopping with the current rank-local sampler.
//! No target execution or serving policy changes. Forced branches bound cost.
use super::*;
use serde_json::{Value,json};
use std::time::Instant;

impl Markov {
    pub(super) fn probe_sharded_prefix_stop(&mut self,base:&Buffer<f32>,post_hc:&Buffer<bf16>,
        anchor:u32,position:u64,comm:&Roce,stream:&Stream)->Result<Value>{
        ensure!(std::env::var("CANADA_SCRATCH_MARKOV_PREFIX_STOP_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_MARKOV_CONDITIONAL_SCREEN").as_deref()==Ok("1")
            &&self.hierarchical_active&&self.fused_head_active&&self.confidence_fused_active,
            "isolated sharded prefix component policy");
        let original=(self.temperature(),self.prefix_stop_enabled,self.gpu_prefix_stop);
        self.set_temperature(0.7,stream)?;
        let result=(||{
            let mut barrier_input=self.cuda.allocate::<f32>(4)?;
            let mut barrier_output=self.cuda.allocate::<f32>(16)?;
            barrier_input.upload(&[0.;4])?;
            let hash=|cuda:&Cuda,p:u64,bytes:usize|->Result<String>{
                Ok(format!("{:x}",Sha256::digest(unsafe{cuda.read_device(p,bytes)}?)))
            };
            let mut records=Vec::new();
            for repeat in 0..6u64 {
                let seed=2026110401+repeat;
                self.prefix_stop_enabled=false;
                let expected=self.run(base,post_hc,anchor,Some((seed,position)),comm,stream)?;
                let qp=self.hierarchical.as_ref().unwrap().q.ptr();
                let q_prefix=hash(&self.cuda,qp,2*32320*8)?;
                let q_full=hash(&self.cuda,qp,5*32320*8)?;
                let confidence_prefix=hash(&self.cuda,self.confidence_probability.ptr(),3*4)?;
                let confidence_full=hash(&self.cuda,self.confidence_probability.ptr(),5*4)?;
                let mut order=[0usize,1,2,3,4];
                order.rotate_left(repeat as usize%5);
                if repeat%2==1{order.reverse();}
                for mode in order {
                    self.prefix_stop_enabled=mode!=0;
                    self.gpu_prefix_stop=mode>=3;
                    let threshold=if mode==1||mode==3{1.}else{0.};
                    // This first call includes any graph creation. Keep it
                    // separate, then compare only warmed replay/readout cost.
                    let started=Instant::now();
                    self.run_causal_three(base,post_hc,anchor,seed,position,threshold,comm,stream)?;
                    let first_ms=started.elapsed().as_secs_f64()*1000.;
                    // Capture warming may run the full path. Poison future q
                    // only after that first use; warmed short replay must
                    // leave those excluded rows untouched and inaccessible.
                    self.run(base,post_hc,anchor,Some((seed+1000,position)),comm,stream)?;
                    let tail_before=hash(&self.cuda,qp+2*32320*8,3*32320*8)?;
                    self.run_causal_three(base,post_hc,anchor,seed,position,threshold,comm,stream)?;
                    let used=self.produced()?;
                    ensure!(used==if mode==1||mode==3{2}else{5},"forced prefix branch did not occur");
                    unsafe{comm.gather_f32(&barrier_input,&mut barrier_output,4,stream)?;}
                    stream.synchronize()?;comm.health()?;
                    let started=Instant::now();
                    let mut ids=[0u32;5];
                    for _ in 0..16 {
                        ids=self.run_causal_three(base,post_hc,anchor,seed,position,threshold,comm,stream)?;
                    }
                    let wall_ms=started.elapsed().as_secs_f64()*1000./16.;
                    ensure!(ids[..used]==expected[..used]&&self.produced()?==used,
                        "sharded prefix changed produced IDs or bounds");
                    ensure!(hash(&self.cuda,qp,2*32320*8)?==q_prefix
                        &&hash(&self.cuda,self.confidence_probability.ptr(),3*4)?==confidence_prefix,
                        "sharded prefix changed retained q/confidence");
                    if used==2 {
                        ensure!(hash(&self.cuda,qp+2*32320*8,3*32320*8)?==tail_before,
                            "short sharded prefix overwrote unused q rows");
                        ensure!(self.sharded_probabilities(3).is_err(),"unproduced q was readable");
                    }else{
                        ensure!(hash(&self.cuda,qp,5*32320*8)?==q_full
                            &&hash(&self.cuda,self.confidence_probability.ptr(),5*4)?==confidence_full,
                            "continued sharded prefix changed full q/confidence");
                    }
                    let _=self.sharded_probabilities(used)?;
                    records.push(json!({"repeat":repeat,"warmup":repeat==0,"mode":
                        (["full","host_short","host_continue","gpu_short","gpu_continue"][mode]),
                        "first_call_ms":first_ms,"wall_ms":wall_ms,"replays":16,
                        "used":used,"ids":&ids[..used],"q_prefix_sha256":q_prefix,
                        "q_full_sha256":if used==5{Some(q_full.clone())}else{None},"all_exact":true}));
                }
            }
            Ok(json!({"all_exact":true,"records":records,"hierarchical_draft":true,
                "temperature":0.7,"scope":"Five forced branch modes at identical real query features and sampled RNG; warmed native Markov calls with current sharded q. Full query/target verification excluded. Short/continue costs do not establish policy or whole-model benefit. Unused q stays private and inaccessible."}))
        })();
        self.prefix_stop_enabled=original.1;self.gpu_prefix_stop=original.2;
        self.set_temperature(original.0,stream)?;self.invalidate_results();result
    }
}
