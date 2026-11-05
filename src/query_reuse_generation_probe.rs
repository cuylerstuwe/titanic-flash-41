//! Shadow-only root-fit opportunity screen for unused query predictions.
use super::*;
use crate::cuda::Buffer;

pub(crate) struct QueryReuseWorkspace {
    p:Buffer<f64>,fresh_q:Buffer<f64>,reused_q:Buffer<f64>,
    overlap:crate::root_feedback::RootFeedback,
}
impl QueryReuseWorkspace {
    pub(crate) fn new(cuda:&Cuda,core:&[u8])->Result<Self>{
        ensure!(std::env::var("CANADA_SCRATCH_QUERY_REUSE_SCREEN").as_deref()==Ok("1"),"finite query reuse workspace");
        Ok(Self{p:cuda.allocate(129280)?,fresh_q:cuda.allocate(129280)?,reused_q:cuda.allocate(129280)?,
            overlap:crate::root_feedback::RootFeedback::new(cuda,core)?})
    }
}
impl GreedySession {
    pub(crate) fn capture_query_for_reuse_probe(&mut self,anchor:u32,comm:&Roce,stream:&Stream)
        ->Result<crate::dspark::QueryOutputSnapshot>{
        ensure!(!self.poisoned&&self.block_verification&&self.target_seed.is_some()
            &&self.draft.hierarchical_active()
            &&std::env::var("CANADA_SCRATCH_QUERY_REUSE_SCREEN").as_deref()==Ok("1"),
            "finite sampled query capture");
        self.draft.run(anchor,comm,stream)?;self.draft.snapshot_query_outputs_for_probe(stream)
    }
    pub(crate) fn measure_query_reuse_for_probe(&mut self,target:&mut Target,anchor:u32,
        old:&crate::dspark::QueryOutputSnapshot,workspace:&mut QueryReuseWorkspace,
        comm:&Roce,stream:&Stream)->Result<serde_json::Value>{
        ensure!(!self.poisoned&&self.block_verification&&self.target_seed.is_some()
            &&self.draft.hierarchical_active()&&target.local_target_enabled()
            &&self.draft.position()?==target.position()
            &&std::env::var("CANADA_SCRATCH_QUERY_REUSE_SCREEN").as_deref()==Ok("1"),
            "finite sampled query reuse measurement");
        let position=target.position();let offset=position.checked_sub(old.position)
            .ok_or_else(||anyhow::anyhow!("query snapshot is from a future position"))?as usize;
        ensure!((1..5).contains(&offset),"no unused query prediction remains");
        let seed=crate::categorical::block_round_seed(self.target_seed.unwrap(),position);
        let started=Instant::now();self.draft.run(anchor,comm,stream)?;
        stream.synchronize()?;let fresh_query_seconds=started.elapsed().as_secs_f64();
        let fresh=self.draft.snapshot_query_outputs_for_probe(stream)?;
        let proposals=self.draft.stochastic_proposals(seed,comm,stream)?;
        let mut inputs=[anchor;6];inputs[1..].copy_from_slice(&proposals);
        target.begin_local_verification(&inputs,comm,stream)?;
        let result=(||{
            let(logits,live)=target.pending_local_target_logits()?;ensure!(live==6,"query reuse verification extent");
            let(q,selected)=self.draft.sharded_probabilities(5)?;
            let rejection=self.rejection.as_mut().ok_or_else(||anyhow::anyhow!("query reuse needs block sampler"))?;
            // This checked draw is discarded, and target verification is
            // rolled back. It fills exactly the production target p buffer.
            let _discarded=rejection.sample_block_sharded(logits,q,selected,&proposals,6,seed,position+1,true,comm,stream)?;
            let p=rejection.root_probabilities_after_draw_for_probe()?;
            let p_local=unsafe{p.view(comm.rank()*32320,32320)}?;
            let q_local=unsafe{q.view(0,32320)}?;
            unsafe{
                comm.gather_f64(&p_local,&mut workspace.p,32320,stream)?;
                comm.gather_f64(&q_local,&mut workspace.fresh_q,32320,stream)?;
            }
            self.draft.install_shifted_query_for_probe(old,offset,stream)?;
            let _discarded=self.draft.stochastic_proposals(seed,comm,stream)?;
            let(q,_)=self.draft.sharded_probabilities(5)?;let q_local=unsafe{q.view(0,32320)}?;
            unsafe{comm.gather_f64(&q_local,&mut workspace.reused_q,32320,stream)?;}
            let overlaps=workspace.overlap.observe_pair_for_query_probe(&workspace.p,&workspace.fresh_q,&workspace.reused_q,stream)?;
            // Restore current private outputs and their exact sampled q so
            // this observer cannot change a subsequent ordinary step.
            self.draft.install_shifted_query_for_probe(&fresh,0,stream)?;
            let restored=self.draft.stochastic_proposals(seed,comm,stream)?;
            ensure!(restored==proposals,"fresh query proposal restoration");
            let(q,_)=self.draft.sharded_probabilities(5)?;
            let local=unsafe{self.cuda.read_device(q.ptr(),32320*8)}?;
            let expected=unsafe{self.cuda.read_device(workspace.fresh_q.ptr()+(comm.rank()*32320*8)as u64,32320*8)}?;
            ensure!(local==expected,"fresh query actual-q restoration");
            Ok(serde_json::json!({"position":position,"old_position":old.position,"shift":offset,
                "available_old_predictions":5-offset,"fresh_root_overlap":overlaps[0],
                "reused_root_overlap":overlaps[1],"fresh_query_seconds":fresh_query_seconds,
                "fresh_root_q_and_proposals_restored":true,
                "scope":"First-proposal tokenwise overlap only; not block expected progress or a generation speed/quality result. Every diagnostic draw is discarded."}))
        })();
        // Even a failed observer attempts ordinary target rollback. The
        // enclosing finite worker retires on any error; it never serves HTTP.
        let rollback=target.finish_verification(0,comm,stream);
        match(result,rollback){(Ok(value),Ok(_))=>Ok(value),(Err(error),_)=>Err(error),(_,Err(error))=>Err(error)}
    }
}
