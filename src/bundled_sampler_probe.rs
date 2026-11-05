//! Isolate combining the target-ID and residual-ID exchanges, preserving both.
use crate::{categorical::{uniform,TARGET_DOMAIN,ACCEPT_DOMAIN,RESIDUAL_DOMAIN},
    cuda::{Buffer,Cuda,Stream},hierarchical_categorical::Hierarchical,roce::Roce,
    sharded_block::ShardedBlock,sharded_categorical_probe::CompactSampler};
use anyhow::{Result,ensure};
use serde_json::{Value,json};
const COLS:usize=32320;const VOCAB:usize=COLS*4;
struct Lane{target:CompactSampler,block:ShardedBlock,p:Buffer<f64>,w:Buffer<f64>,
    state:Buffer<u32>,output:Buffer<i32>,errors:Buffer<u32>}
impl Lane{
    fn new(cuda:&Cuda,core:&[u8],rank:usize,live:usize)->Result<Self>{
        let mut target=CompactSampler::new(cuda,core,live,rank)?;target.prepare_local_probabilities()?;
        Ok(Self{target,block:ShardedBlock::new(cuda,core,rank,6)?,p:cuda.allocate(live*VOCAB)?,
            w:cuda.allocate(6)?,state:cuda.allocate(6)?,output:cuda.allocate(6)?,errors:cuda.allocate(6)?})
    }
    #[allow(clippy::too_many_arguments)]
    fn enqueue(&mut self,local:&Buffer<f32>,q:&Buffer<f64>,props:&Buffer<i32>,a:&Buffer<f64>,
        u:&Buffer<f64>,live:usize,deferred:bool,comm:&Roce,stream:&Stream)->Result<()>{
        if deferred{
            if self.block.fused_probabilities{self.target.enqueue_deferred_factors(local,props,&self.p,comm,stream)?;}
            else{self.target.enqueue_deferred_probabilities(local,props,&self.p,1.,comm,stream)?;}
            let probabilities=if self.block.fused_probabilities{Some(self.target.deferred_probabilities(local)?)}else{None};
            self.block.enqueue_deferred(&self.p,q,props,a,u,self.target.output(),&self.w,
                &self.state,&self.output,&self.errors,live,comm,stream,self.target.local_output(),self.target.errors(),probabilities)
        }else{
            self.target.enqueue_local_probabilities(local,props,&self.p,1.,comm,stream)?;
            self.block.enqueue(&self.p,q,props,a,u,self.target.output(),&self.w,
                &self.state,&self.output,&self.errors,live,comm,stream)
        }
    }
}
pub(crate) fn run(cuda:&Cuda,core:&[u8],q:&Buffer<f64>,draws:&[Hierarchical],comm:&Roce,stream:&Stream)->Result<Value>{
    let module=cuda.module(core)?;let rank=comm.rank();let mut results=Vec::new();
    let probability_fusion=std::env::var("CANADA_SCRATCH_FUSED_PROBABILITY_RESIDUAL_SCREEN").as_deref()==Ok("1");
    let residual_fusion=std::env::var("CANADA_SCRATCH_FUSED_RESIDUAL_LOCAL_SCREEN").as_deref()==Ok("1");
    let finish=probability_fusion||std::env::var("CANADA_SCRATCH_FUSED_TARGET_FINISH_SCREEN").as_deref()==Ok("1");
    let fused=finish||std::env::var("CANADA_SCRATCH_FUSED_TARGET_TRANSPORT_SCREEN").as_deref()==Ok("1");
    let mut pointers=[0u64;5];let mut ids=vec![0i32;6];
    for i in 0..5{pointers[i]=draws[i].selected_q().ptr();ids[i]=draws[i].output().download()?[0];}
    let mut props=cuda.allocate(6)?;props.upload(&ids)?;
    let mut accept=cuda.allocate(6)?;let mut residual=cuda.allocate(6)?;
    for live in if probability_fusion{vec![2usize,3,6]}else if fused{vec![2usize,3,4,5,6]}else{vec![3usize,6]}{
        let local=cuda.allocate::<f32>(live*COLS)?;
        let mut lanes=[Lane::new(cuda,core,rank,live)?,Lane::new(cuda,core,rank,live)?];
        if fused{lanes[1].target.set_fused_controls_for_probe(true)?;}
        if finish{
            lanes[0].target.set_fused_controls_for_probe(true)?;
            lanes[0].target.set_fused_finish_for_probe(false)?;
            lanes[1].target.set_fused_finish_for_probe(true)?;
        }
        if residual_fusion {
            ensure!(finish,"residual fusion comparison uses the accepted target finish in both arms");
            for(i,lane)in lanes.iter_mut().enumerate(){
                lane.target.set_fused_finish_for_probe(true)?;
                lane.block.fused_local=i==1;
            }
        }
        if probability_fusion {
            for(i,lane)in lanes.iter_mut().enumerate(){
                lane.target.set_fused_controls_for_probe(true)?;
                lane.target.set_fused_finish_for_probe(true)?;
                lane.block.fused_local=true;
                lane.block.fused_probabilities=i==1;
            }
        }
        for lane in &mut lanes{lane.block.selected.upload(&pointers)?;}
        let fill=|phase:i32|->Result<()>{
            let(mut out,mut rank,mut rows,mut phase)=(local.ptr(),rank as i32,live as i32,phase);
            let mut args=[(&mut out as *mut u64).cast(),(&mut rank as *mut i32).cast(),
                (&mut rows as *mut i32).cast(),(&mut phase as *mut i32).cast()];
            unsafe{module.function("ca_sharded_categorical_fixture")?.launch([local.len().div_ceil(256)as u32,1,1],
                [256,1,1],0,stream,&mut args)}
        };
        let mut cases=Vec::new();
        for phase in 0..8{for trial in 0..3{
            fill(phase)?;
            let a:Vec<_>=(0..6).map(|i|match trial{0=>0.,1=>1.-1./9007199254740992.,_=>uniform(2026110315,ACCEPT_DOMAIN,i)}).collect();
            let u:Vec<_>=(0..6).map(|i|uniform(2026110315+trial,RESIDUAL_DOMAIN,i)).collect();
            let bonus:Vec<_>=(0..live).map(|i|match trial{0=>0.,1=>1.-1./9007199254740992.,_=>uniform(2026110315,TARGET_DOMAIN,i as u64)}).collect();
            accept.upload(&a)?;residual.upload(&u)?;
            for lane in &mut lanes{lane.target.upload_uniforms(&bonus)?;lane.block.uniforms(2026110315+trial,0,live)?;}
            stream.join_default()?;
            for(i,lane)in lanes.iter_mut().enumerate(){lane.enqueue(&local,q,&props,&accept,&residual,live,fused||i==1,comm,stream)?;}
            stream.synchronize()?;comm.health()?;
            let errors=lanes[0].target.errors().download()?;let target=lanes[0].target.output().download()?;
            ensure!(errors==lanes[1].target.errors().download()?&&target==lanes[1].target.output().download()?,"bundled target IDs/errors");
            let a=lanes[0].p.download()?;let b=lanes[1].p.download()?;
            ensure!((0..live).all(|row|(rank*COLS..(rank+1)*COLS).chain(std::iter::once(ids[row]as usize))
                .all(|id|{let i=row*VOCAB+id;a[i].to_bits()==b[i].to_bits()||(a[i].is_nan()&&b[i].is_nan())})),"bundled actual target probabilities");
            let ea=lanes[0].errors.download()?;let eb=lanes[1].errors.download()?;
            // Categorical failure can be detected locally before or after its
            // deferred exchange. Both must reject; valid paths stay exact.
            let valid=errors.iter().all(|&e|e==0);
            if valid{
                ensure!(ea[..live]==eb[..live]&&ea[..live].iter().all(|&e|e==0),"bundled residual errors");
                ensure!(lanes[0].w.download()?[..live]==lanes[1].w.download()?[..live]
                    &&lanes[0].state.download()?[..live]==lanes[1].state.download()?[..live]
                    &&lanes[0].output.download()?[..live]==lanes[1].output.download()?[..live],"bundled draw/state/weights");
            }
            cases.push(json!({"phase":phase,"trial":trial,"valid":valid,"target_ids":target,"target_errors":errors,"exact":true}));
        }}
        fill(0)?;stream.synchronize()?;
        let graphs=lanes.iter_mut().enumerate().map(|(i,lane)|stream.capture(||{
            for _ in 0..8{lane.enqueue(&local,q,&props,&accept,&residual,live,fused||i==1,comm,stream)?;}Ok(())
        })).collect::<Result<Vec<_>>>()?;
        let mut pairs=Vec::new();
        for repeat in 0..6{
            let mut ms=[0.;2];let order=if repeat%2==0{[0,1]}else{[1,0]};
            for variant in order{
                for _ in 0..2{unsafe{graphs[variant].launch(stream)?;}}stream.synchronize()?;
                let start=cuda.event()?;let end=cuda.event()?;start.record(stream)?;
                for _ in 0..8{unsafe{graphs[variant].launch(stream)?;}}
                end.record(stream)?;ms[variant]=end.elapsed_since(&start)?/64.;comm.health()?;
            }
            ensure!(lanes[0].output.download()?[..live]==lanes[1].output.download()?[..live]
                &&lanes[0].state.download()?[..live]==lanes[1].state.download()?[..live],"bundled replay output/state");
            pairs.push(json!({"repeat":repeat,"order":order,"milliseconds":ms}));
        }
        results.push(json!({"live":live,"cases":cases,"pairs":pairs}));
    }
    Ok(json!({"all_exact":true,"results":results,"fused_target_controls":fused,"fused_target_finish":finish,"fused_residual_local":residual_fusion,"fused_probability_residual":probability_fusion,
        "scope":if probability_fusion{"Both arms use current fused target/residual controls and bundled IDs. Candidate forms exact full local p inside unchanged residual partial reductions, after selected p determines block weights. Identical actual p/q, RNG, target IDs/errors, weights, acceptance and correction with changed inputs and invalid logits. Actual four-rank transport included; no model speed claim."}
        else if residual_fusion{"Both arms use qualified fused target sampling and bundled IDs. Candidate combines residual tile selection, unchanged FP64 CUB local CDF and wire packing into one CTA. Identical actual p/q, RNG, target IDs/errors, block weights, acceptance and correction; changed-input replay. Component only."}
        else if finish{"Both arms use the qualified fused target controls and bundled IDs. Candidate reads the selected tile directly into the original FP64 CUB raking CDF without staging through scratch. Identical actual p/q, RNG, target IDs/errors, block weights, acceptance and correction; changed-input replay. Component only."}
        else{"Compact target plus actual-q sharded block. Optional fused arm packs proposal/clears errors within boundary transport and computes normalization factors within the tile kernel; both arms already bundle target/residual IDs. Identical RNG, FP64 p/q, canonical tile arithmetic and corrected output. Invalid-input phases must reject. Component only."}}))
}
