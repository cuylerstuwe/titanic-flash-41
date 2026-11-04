//! Finite whole-model pair replay. Each sequence retains its own attention,
//! compression, Engram rows, token metadata, head and commit authority.
use super::*;
use serde_json::{Value,json};
use sha2::{Digest,Sha256};

#[path = "paired_execution_probe.rs"]
mod execution;
pub(crate) use execution::PairExecutor;

impl Target {
    /// Recreate the ordinary per-owner Verify policies while recording each
    /// portion of a paired graph. The scopes end before the peer is enqueued.
    fn pair_scopes<'c>(&self,comm:&'c Roce,stream:&Stream)
        ->Result<(crate::mhc::ActiveRowsScope,crate::roce::NativeRowScope<'c>)> {
        let live=self.tokens.active_rows();
        let mhc=crate::mhc::ActiveRowsScope::new(live<=self.verification_mhc_live_rows,
            live,self.rows,stream,false)?;
        comm.select_prefill_rows(live,self.rows,stream)?;
        let native=comm.native_row_scope(self.tiny_prefix_sum,self.tiny_prefix_gather,live,self.rows,stream)?;
        Ok((mhc,native))
    }
    fn pair_defer_ffn(&mut self,enabled:bool)->Result<()> {
        for layer in self.early.iter_mut().chain(self.group8.layers_mut())
            .chain(self.group14.layers_mut()).chain(self.late.layers_mut()) {
            layer.defer_pair_ffn(enabled)?;
            if !enabled {layer.clear_pair_attention();}
        }
        Ok(())
    }
    fn pair_layer_mut(&mut self,index:usize)->&mut SwaLayer {
        match index {
            0..=7=>&mut self.early[index],
            8..=13=>&mut self.group8.layers_mut()[index-8],
            14..=19=>&mut self.group14.layers_mut()[index-14],
            _=>&mut self.late.layers_mut()[index-20],
        }
    }
    fn pair_begin_layer(&mut self,index:usize,comm:&Roce,stream:&Stream)->Result<()> {
        match index {
            0=>self.enqueue_initial(comm,stream),
            1..=7=>{
                let (previous,rest)=self.early.split_at_mut(index);
                let input=previous.last().unwrap();let layer=&mut rest[0];
                let cache=&mut self.early_caches[index];
                match index {
                    1=>layer.run_from_previous(input.output(),Some(input.pre()),Some(&mut self.rows1),
                        &self.step.positions,&self.step.locations,cache,&self.step.candidates,&self.tokens,comm,stream),
                    2=>layer.run_indexed_from_previous(input.output(),input.pre(),&self.step.compressed_positions,
                        &self.step.compressed_locations,cache,&self.step.candidates,&self.tokens,&mut self.source2,
                        &self.step.m2,comm,stream),
                    _=>layer.run_shared_from_previous(input.output(),input.pre(),&self.step.compressed_positions,
                        &self.step.compressed_locations,cache,&self.step.candidates,&self.tokens,&self.source2,comm,stream),
                }
            }
            8..=13=>self.group8.begin_pair_layer(index-8,self.early[7].output(),self.early[7].pre(),
                &self.step.compressed_positions,&self.step.compressed_locations,&self.step.candidates,
                &self.tokens,&self.step.m8,comm,stream),
            14..=19=>self.group14.begin_pair_layer(index-14,self.group8.output(),self.group8.pre(),
                &self.step.compressed_positions,&self.step.compressed_locations,&self.step.candidates,
                &self.tokens,&self.step.m14,comm,stream),
            20..=39=>self.late.begin_pair_layer(index-20,self.group14.output(),self.group14.pre(),
                &self.step.compressed_positions,&self.step.compressed_locations,&self.step.candidates,
                &self.tokens,&self.step.m20,comm,stream),
            _=>anyhow::bail!("paired target layer outside model"),
        }
    }
    fn enqueue_pair(&mut self,other:&mut Self,pooled:Option<&crate::ffn::Paired>,coalesce:bool,
        head:Option<&crate::model_io::PairedHead>,
        comm:&Roce,stream:&Stream)->Result<()> {
        self.enqueue_pair_features(other,pooled,coalesce,head,7,comm,stream)
    }
    /// Bits select shared-up, head, and QKV pooling for a same-load component
    /// attribution. The ordinary pair executor always requests all built parts.
    fn enqueue_pair_features(&mut self,other:&mut Self,pooled:Option<&crate::ffn::Paired>,coalesce:bool,
        head:Option<&crate::model_io::PairedHead>,features:u8,comm:&Roce,stream:&Stream)->Result<()> {
        ensure!(features<=7,"paired dense feature mask");
        self.pair_defer_ffn(true)?;other.pair_defer_ffn(true)?;
        let result=(||->Result<()>{
            for index in 0..40 {
                // The three entry layers keep their embedding/Engram schedule.
                // Elsewhere, prepare both private normalizations before pooling
                // only QKV. Their source/index/KV state is still advanced once.
                if let Some(qkv)=pooled.and_then(|p|p.qkv()).filter(|_|
                    features&4!=0&&index>=2&&index!=14&&matches!(self.tokens.active_rows(),3|6)
                    &&matches!(other.tokens.active_rows(),3|6)) {
                    self.pair_layer_mut(index).defer_pair_attention()?;
                    other.pair_layer_mut(index).defer_pair_attention()?;
                    {let _scope=self.pair_scopes(comm,stream)?;self.pair_begin_layer(index,comm,stream)?;}
                    {let _scope=other.pair_scopes(comm,stream)?;other.pair_begin_layer(index,comm,stream)?;}
                    let live=self.tokens.active_rows();let other_live=other.tokens.active_rows();
                    self.pair_layer_mut(index).project_pair_attention(other.pair_layer_mut(index),
                        qkv,live,other_live,stream)?;
                }
                {let _scope=self.pair_scopes(comm,stream)?;self.pair_begin_layer(index,comm,stream)?;}
                {let _scope=other.pair_scopes(comm,stream)?;other.pair_begin_layer(index,comm,stream)?;}
                let a=self.layer_owners()[index];let b=other.layer_owners()[index];
                if let Some(pool)=pooled {
                    pool.run_with_shared_pool(a.ffn(),a.ffn_boundary().input(),a.ffn_boundary().quantized_input(),&self.tokens,
                        b.ffn(),b.ffn_boundary().input(),b.ffn_boundary().quantized_input(),&other.tokens,features&1!=0,stream)?;
                }else{
                    for (target,layer,tokens)in [(&*self,a,&self.tokens),(&*other,b,&other.tokens)] {
                        let _scope=target.pair_scopes(comm,stream)?;
                        layer.ffn().run_quantized_input(&layer.narrow_workspace(),layer.ffn_boundary().input(),
                            layer.ffn_boundary().quantized_input(),tokens,stream)?;
                    }
                }
                if coalesce {
                    let reduced=pooled.context("paired reduction needs pooled storage")?
                        .reduce(a.ffn().output(),b.ffn().output(),self.tokens.active_rows(),other.tokens.active_rows(),comm,stream)?;
                    let cuda=self.cuda.clone();
                    {let _scope=self.pair_scopes(comm,stream)?;
                        self.pair_layer_mut(index).finish_pair_ffn_reduced(&reduced[0],&cuda,stream)?;}
                    {let _scope=other.pair_scopes(comm,stream)?;
                        other.pair_layer_mut(index).finish_pair_ffn_reduced(&reduced[1],&cuda,stream)?;}
                }else{
                    {let _scope=self.pair_scopes(comm,stream)?;self.pair_layer_mut(index).finish_pair_ffn(comm,stream)?;}
                    {let _scope=other.pair_scopes(comm,stream)?;other.pair_layer_mut(index).finish_pair_ffn(comm,stream)?;}
                }
            }
            let pooled_head=head.filter(|_|features&2!=0&&matches!(self.tokens.active_rows(),3|6)&&matches!(other.tokens.active_rows(),3|6));
            if let Some(head)=pooled_head {
                // Preserve each owner's live-row policy for normalization;
                // only the final canonical vocabulary multiplication is pooled.
                {let _scope=self.pair_scopes(comm,stream)?;
                    self.io.prepare_head(self.late.output(),self.late.pre(),stream)?;}
                {let _scope=other.pair_scopes(comm,stream)?;
                    other.io.prepare_head(other.late.output(),other.late.pre(),stream)?;}
                head.run_normalized(&self.io,self.tokens.active_rows(),&other.io,other.tokens.active_rows(),stream)?;
            }
            for target in [&mut *self,&mut *other] {
                let _scope=target.pair_scopes(comm,stream)?;
                if pooled_head.is_none(){target.io.run_head(target.late.output(),target.late.pre(),stream)?;}
                // The paired M6 executor retains its complete target logits.
                // Solo local-target transport has a separate graph bank and
                // must not silently alter this executor's output contract.
                target.io.gather_logits(comm,stream)?;
            }
            Ok(())
        })();
        // Graph recording changes only host enqueue policy. Ordinary solo
        // graphs and all later acceptance/rollback paths retain their policy.
        self.pair_defer_ffn(false)?;other.pair_defer_ffn(false)?;result
    }
    fn pair_model_taps(&self,full:bool)->Result<Value> {
        let (logits,live)=self.pending_local_logits_for_probe()?;
        let hash=|ptr,bytes|->Result<String>{Ok(format!("{:x}",Sha256::digest(unsafe{
            self.cuda.read_device(ptr,bytes)
        }?)))};
        let mut value=json!({"local_logits":hash(logits.ptr(),live*32320*4)?,
            "residual":hash(self.late.output().ptr(),self.late.output().bytes()/self.rows*live)?,
            "pre":hash(self.late.pre().ptr(),self.late.pre().bytes()/self.rows*live)?,
            "carry":self.carry_snapshot.as_ref().context("paired carry snapshot")?.live_digests()?});
        if full {
            value["regions"]=serde_json::to_value(self.state_image_regions(self.position()as usize+live,true)?
                .iter().map(|r|r.repeat_digest()).collect::<Result<Vec<_>>>()?)?;
        }
        Ok(value)
    }
    fn pair_restore_carry(&self,other:&Self,stream:&Stream)->Result<()> {
        for target in [self,other] {
            unsafe{target.carry_snapshot.as_ref().context("missing incoming pair carry")?.restore(stream)?;}
        }
        Ok(())
    }
    fn pair_replay_enqueue(&mut self,other:&mut Self,pooled:&crate::ffn::Paired,
        mode:usize,comm:&Roce,stream:&Stream)->Result<()> {
        // Diagnostic replays must start from the actual incoming compressor
        // rings. This small replay-only cost is identical in all timed arms.
        self.pair_restore_carry(other,stream)?;
        match mode {
            0=>{
                {let _scope=self.pair_scopes(comm,stream)?;self.enqueue(comm,stream)?;}
                let _scope=other.pair_scopes(comm,stream)?;other.enqueue(comm,stream)
            },
            1=>self.enqueue_pair(other,None,false,None,comm,stream),
            2=>self.enqueue_pair(other,Some(pooled),false,None,comm,stream),
            3=>other.enqueue_pair(self,Some(pooled),false,None,comm,stream),
            4=>self.enqueue_pair(other,Some(pooled),true,None,comm,stream),
            5=>other.enqueue_pair(self,Some(pooled),true,None,comm,stream),
            _=>anyhow::bail!("pair replay variant"),
        }
    }
    pub(crate) fn paired_model_screen(&mut self,other:&mut Self,pooled:&crate::ffn::Paired,
        comm:&Roce,stream:&Stream)->Result<Value> {
        ensure!(std::env::var("CANADA_SCRATCH_PAIRED_MODEL_SCREEN").as_deref()==Ok("1")
            && !self.lease.shares_owner(&other.lease) && self.stream==other.stream
            && self.cuda.owns_stream(stream) && self.stream==stream.raw()as usize,
            "finite paired whole-model owner/stream admission");
        for t in [&*self,&*other] {
            ensure!(t.rows==6&&t.source_phase==Phase::Verify&&t.wide.is_none()&&t.cached_decoder.is_none()
                &&t.lease.is_busy_revision()&&t.pending.as_ref().is_some_and(|p|p.live>0&&p.live<=6)
                &&t.comm_instance==Some(comm.instance())&&t.rank==comm.rank()&&!t.prefill_last_head
                &&t.layer_owners().iter().all(|l|l.sp_rank().is_none()),"paired model profile contract");
        }
        comm.select_decode_reduction(true,stream)?;
        stream.synchronize()?;
        let expected=[self.pair_model_taps(true)?,other.pair_model_taps(true)?];
        let brief=[self.pair_model_taps(false)?,other.pair_model_taps(false)?];
        let modes=if std::env::var("CANADA_SCRATCH_PAIRED_REDUCE_SCREEN").as_deref()==Ok("1") {
            [0usize,2,4,5]
        }else{[0,1,2,3]};
        let mut graphs=Vec::new();
        for mode in modes {
            self.pair_replay_enqueue(other,pooled,mode,comm,stream)?;
            stream.synchronize()?;comm.health()?;self.check_candidates()?;other.check_candidates()?;
            ensure!([self.pair_model_taps(true)?,other.pair_model_taps(true)?]==expected,
                "paired model mode{mode} changed logits or causal state");
            graphs.push(stream.capture(||self.pair_replay_enqueue(other,pooled,mode,comm,stream))?);
        }
        let mut records=Vec::new();
        for repeat in 0..4 {
            let order=if repeat%2==0 {[0,1,2,3]}else{[3,2,1,0]};
            let mut milliseconds=[0f32;4];
            for mode in order {
                unsafe{graphs[mode].launch(stream)?;}stream.synchronize()?;
                let begin=self.cuda.event()?;let end=self.cuda.event()?;begin.record(stream)?;
                for _ in 0..4 {unsafe{graphs[mode].launch(stream)?;}}
                end.record(stream)?;milliseconds[mode]=end.elapsed_since(&begin)?/4.;comm.health()?;
                ensure!([self.pair_model_taps(false)?,other.pair_model_taps(false)?]==brief,
                    "paired model graph{mode} changed replay output");
            }
            records.push(json!({"repeat":repeat,"order":order,"milliseconds":milliseconds}));
        }
        ensure!([self.pair_model_taps(true)?,other.pair_model_taps(true)?]==expected,
            "paired model repeated replay changed retained state");
        let names=["serial_ab","interleaved_separate","pooled_ab","pooled_ba","pooled_sum_ab","pooled_sum_ba"];
        Ok(json!({"all_exact":true,"modes":modes.map(|m|names[m]),
            "live_rows":[self.tokens.active_rows(),other.tokens.active_rows()],
            "original_forward":expected,"records":records,
            "scope":"Full forty-layer paired target replay at real independent frontiers. Both local vocabulary shards, residuals, compressor carry and causal storage match original per-owner forwards. Timings include identical replay-only incoming carry restores; exclude metadata upload, Engram I/O, drafting and acceptance. Not serving throughput or long-context memory qualification."}))
    }
}
