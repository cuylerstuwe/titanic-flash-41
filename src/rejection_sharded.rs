//! Distributed actual-q block rejection. Target normalization is unchanged.
use super::*;
use crate::{roce::Roce,sharded_block::ShardedBlock};

pub(super) struct ShardedGraph {
    key:(u64,u64,[u64;5],usize,u64,usize,u64,bool,bool),
    graph:Graph,
}
impl Rejection {
    pub(crate) fn capture_root_branch_for_probe(&mut self,q:&Buffer<f64>,live:usize,
        seed:u64,position:u64,rank:usize)->Result<()> {
        if let Some(capture)=&mut self.root_branch_capture {
            capture.record(&self.cuda,&self.p,q,&self.block_weights,&self.proposals,
                self.sharded.as_ref().ok_or_else(||anyhow::anyhow!("root capture missing sharded block"))?,
                live,seed,position,rank)?;
        }
        Ok(())
    }
    pub(super) fn enqueue_sharded_readout(&self,live:usize,local_target:bool,stream:&Stream)->Result<()> {
        if local_target{
            let target=self.local_targets[live-1].as_ref().unwrap();
            self.enqueue_readout_from(target.errors(),target.output(),live,stream)
        }else{self.enqueue_readout(stream)}
    }
    pub(super) fn finish_sharded_readout(&self,live:usize,local_target:bool,stream:&Stream,enqueued:bool)
        ->Result<(Vec<i32>,Vec<u32>)>{
        let target=if local_target{
            let target=self.local_targets[live-1].as_ref().unwrap();Some((target.errors(),target.output()))
        }else{None};
        self.finish_readout_from(live,stream,enqueued,target)
    }
    pub(crate) fn sample_point_mass_sharded(&mut self,logits:&Buffer<f32>,live:usize,
        seed:u64,position:u64,comm:&Roce,stream:&Stream)->Result<Vec<u32>> {
        ensure!((2..=6).contains(&live)&&live<=self.capacity
            &&logits.len()>=live*32320&&self.cuda.owns_buffer(logits)
            &&self.sharded_comm.is_none_or(|id|id==(comm.rank(),comm.instance())),
            "point-mass compact target shape/collective identity");
        let sampler=self.local_targets[live-1].as_mut()
            .ok_or_else(||anyhow::anyhow!("point-mass compact target not prepared"))?;
        sampler.bind_rank(comm.rank())?;self.sharded_comm=Some((comm.rank(),comm.instance()));
        // This deterministic proposal needs independent target draws only.
        // Keep the original target RNG domain and absolute positions exactly.
        let local=unsafe{logits.view(0,live*32320)}?;
        sampler.sample(&local,seed,position,comm,stream)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample_block_sharded(&mut self,logits:&Buffer<f32>,q:&Buffer<f64>,
        selected:[u64;5],proposals:&[u32],live:usize,seed:u64,position:u64,
        local_target:bool,comm:&Roce,stream:&Stream)->Result<Draw> {
        ensure!(self.sharded.is_some()
            &&live>0&&live<=6&&live<=self.capacity&&proposals.len()+1==live
            &&position.checked_add(live as u64).is_some()
            &&q.len()>=(live-1)*32320&&logits.len()>=live*if local_target{32320}else{VOCAB}
            &&self.cuda.owns_buffer(q)&&self.cuda.owns_buffer(logits)&&self.cuda.owns_stream(stream)
            &&proposals.iter().all(|&x|x<VOCAB as u32)
            &&selected[..live-1].iter().all(|&p|p!=0)
            &&self.progress.is_none()&&self.decision_control.is_none()&&self.shadow_expected.is_none(),
            "sharded block admission/shape/context");
        // No proposal is consumed for a final one-token remainder. The existing
        // canonical target-only path never reads q in this case.
        ensure!(!local_target||(live>1&&self.local_targets[live-1].is_some()),"local target workspace/extent");
        if live==1{return self.sample_block(logits,q,proposals,live,seed,position,stream);}
        let identity=(comm.rank(),comm.instance());
        ensure!(self.sharded_comm.is_none_or(|previous|previous==identity),"sharded sampler collective identity");
        let mut h=self.sharded.take().ok_or_else(||anyhow::anyhow!("sharded workspace not prepared"))?;
        let result=(||{
            h.bind_rank(comm.rank())?;self.sharded_comm=Some(identity);
            h.selected.upload(&selected)?;h.uniforms(seed,position,live)?;
            let mut ids=vec![0;self.capacity];
            for(i,&id)in proposals.iter().enumerate(){ids[i]=id as i32;}
            let mut a=vec![0.;self.capacity];let mut r=a.clone();
            let mut bonus=vec![0.;live];
            for row in 0..live {
                a[row]=uniform(seed,ACCEPT_DOMAIN,position+row as u64);
                r[row]=uniform(seed,RESIDUAL_DOMAIN,position+row as u64);
                bonus[row]=uniform(seed,TARGET_DOMAIN,position+row as u64);
            }
            self.proposals.upload(&ids)?;self.accept_u.upload(&a)?;self.residual_u.upload(&r)?;
            if local_target {
                let sampler=self.local_targets[live-1].as_mut().unwrap();
                sampler.bind_rank(comm.rank())?;sampler.upload_uniforms(&bonus)?;
            }else{self.target.upload_uniforms(&bonus)?;}
            stream.join_default()?;
            if let Some(staging)=&self.staged_inputs{unsafe{staging.flush(Some(stream))?;}}
            let captured_readout=self.graph_replay&&self.graph_readout&&self.readout.is_some();
            if self.graph_replay {
                let readout=if captured_readout{self.readout.as_ref().unwrap().pointer()}else{0};
                let key=(logits.ptr(),q.ptr(),selected,live,comm.instance(),stream.raw()as usize,readout,local_target,self.bundled_ids);
                if self.sharded_graphs[live-1].as_ref().is_none_or(|g|g.key!=key) {
                    self.sharded_graphs[live-1]=None;
                    let graph=stream.capture(||{
                        self.enqueue_sharded(logits,q,&mut h,live,local_target,comm,stream)?;
                        if captured_readout{self.enqueue_sharded_readout(live,local_target,stream)?;}
                        Ok(())
                    })?;
                    self.sharded_graphs[live-1]=Some(ShardedGraph{key,graph});
                }
                unsafe{self.sharded_graphs[live-1].as_ref().unwrap().graph.launch(stream)?;}
            }else{self.enqueue_sharded(logits,q,&mut h,live,local_target,comm,stream)?;}
            let(output,state)=self.finish_sharded_readout(live,local_target,stream,captured_readout)?;
            comm.health()?;
            let prefix=state[..live-1].iter().take_while(|&&x|x==1).count();
            ensure!(state[..live-1].iter().enumerate().all(|(i,&x)|(x==1)==(i<prefix)),
                "sharded block acceptance is not a prefix");
            Ok(Draw{ids:output[..live].iter().map(|&x|x as u32).collect(),
                accepted:state[..live].iter().map(|&x|x==1).collect(),accepted_prefix:Some(prefix),
                acceptance_expected_lengths:None,first_proposal_overlap:None,local_proposal_fit:None})
        })();
        self.sharded=Some(h);result
    }
    fn enqueue_sharded(&mut self,logits:&Buffer<f32>,q:&Buffer<f64>,h:&mut ShardedBlock,
        live:usize,local_target:bool,comm:&Roce,stream:&Stream)->Result<()> {
        if local_target {
            let sampler=self.local_targets[live-1].as_mut().unwrap();
            let local=unsafe{logits.view(0,live*32320)}?;
            if self.bundled_ids{sampler.enqueue_deferred_probabilities(&local,&self.proposals,&self.p,1.,comm,stream)?;}
            else{sampler.enqueue_local_probabilities(&local,&self.proposals,&self.p,1.,comm,stream)?;}
        }else {
            self.target.enqueue(logits,0,live,1.,stream)?;
            self.target.probabilities(logits,0,live,1.,&mut self.p,0,stream)?;
        }
        let bonus=if local_target{self.local_targets[live-1].as_ref().unwrap().output()}
            else{self.target.device_output()};
        if local_target&&self.bundled_ids{
            let target=self.local_targets[live-1].as_ref().unwrap();
            h.enqueue_deferred(&self.p,q,&self.proposals,&self.accept_u,&self.residual_u,bonus,
                &self.block_weights,&self.state,&self.output,&self.errors,live,comm,stream,
                target.local_output(),target.errors())
        }else{
            h.enqueue(&self.p,q,&self.proposals,&self.accept_u,&self.residual_u,bonus,
                &self.block_weights,&self.state,&self.output,&self.errors,live,comm,stream)
        }
    }
}
