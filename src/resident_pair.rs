//! Two committed request owners share only the row-independent target work.
//! Wide history copies retain their ordinary path. No proposal is prepared
//! until both owners are eligible, and no token is published before agreement.
use super::*;
use crate::dspark_generation::Step;

struct View<'a> {
    id:u64,
    session:&'a mut GreedySession,
    verify:&'a mut Target,
    long:&'a mut Option<Target>,
    long_buffers:&'a mut Option<LongHistoryBuffers>,
    copy:&'a mut Option<Target>,
    copy_buffers:&'a mut Option<LongHistoryBuffers>,
    history:&'a mut ResidentHistory,
    emitted:&'a mut usize,
    limit:usize,
}
impl View<'_> {
    fn step_limit(&self)->Result<usize> {
        self.history.step_limit_for_rows(if self.copy.is_some(){32}else if self.long.is_some(){16}else{6})
    }
    fn try_wide(&mut self,comm:&Roce,stream:&Stream)->Result<Option<Step>> {
        let limit=self.step_limit()?;let anchor=self.history.pending().context("paired anchor missing")?;
        if let(Some(target),Some(buffers))=(self.copy.as_mut(),self.copy_buffers.as_mut()) {
            if let Some(step)=self.session.try_long_history_step(self.verify,target,buffers,anchor,limit,comm,stream)? {
                return Ok(Some(step));
            }
        }
        if let(Some(target),Some(buffers))=(self.long.as_mut(),self.long_buffers.as_mut()) {
            return self.session.try_long_history_step(self.verify,target,buffers,anchor,limit,comm,stream);
        }
        Ok(None)
    }
    fn ordinary(&mut self,comm:&Roce,stream:&Stream,control:&mut ControlChannel)->Result<Step> {
        self.session.step_with_profile_barrier(self.verify,self.history.pending().context("paired anchor missing")?,
            self.step_limit()?.min(6),comm,stream,&mut||control.agree(b"warm-verify-profile-rendezvous"))
    }
    fn commit(&mut self,step:Step,limit:usize,began:Instant)->Result<Value> {
        let output=self.history.committed_step(&step.tokens,step.next_anchor,self.verify.position()as usize)?;
        *self.emitted+=output.len();
        ensure!(*self.emitted<=self.limit,"paired output exceeded request limit");
        let mut event=json!({"type":"tokens","request_id":self.id,"output_ids":output,
            "frontier":self.history.frontier(),"pending":self.history.pending(),
            "completion_tokens_so_far":*self.emitted,"seconds":began.elapsed().as_secs_f64(),
            "phase_ms":step.phase_ms,"verified_rows":step.verified_rows,"step_limit":limit,
            "committed_model_rows":step.tokens.len(),"accepted_draft_tokens":step.matching_proposals,
            "emitted_tokens":output.len(),"active_sparse_index_layers":self.verify.active_sparse_index_layers(),
            "compact_sparse_index_layers":self.verify.compact_sparse_index_layers(),
            "proposal_source":step.proposal_source,"lookup_match_tokens":step.lookup_match_tokens,
            "history_lookup_position":self.session.history_lookup_metadata().map(|(position,_)|position)});
        if let Some(expected)=step.acceptance_expected_lengths {
            event["acceptance_expected_lengths"]=json!(expected);event["acceptance_shadow_preserved_draw"]=json!(true);
        }
        Ok(event)
    }
}

impl Worker {
    pub(crate) fn resident_peer_terminal(&self,id:u64)->bool {
        self.peer_bank.as_ref().is_some_and(|p|p.current==Some(id)&&p.pending_prefill.is_none()
            &&(p.prefill_cancelled||p.history.pending()==Some(self.cfg.eos_token_id)||p.emitted==p.limit))
    }
    pub(crate) fn resident_pair_can_step(&self,id:u64)->Result<bool> {
        if self.resident_pair.is_none()||self.current.is_none()||self.pending_prefill.is_some()
            ||self.prefill_cancelled||self.history.pending()==Some(self.cfg.eos_token_id)||self.emitted==self.limit {
            return Ok(false);
        }
        let Some(peer)=self.peer_bank.as_ref()else{return Ok(false);};
        if peer.current!=Some(id)||peer.pending_prefill.is_some()||peer.prefill_cancelled
            ||peer.history.pending()==Some(self.cfg.eos_token_id)||peer.emitted==peer.limit {
            return Ok(false);
        }
        let(Some(a),Some(b))=(self.session.as_ref(),peer.session.as_ref())else{return Ok(false);};
        Ok(a.pair_proposal_eligible(&self.verify,self.history.step_limit()?)?
            &&b.pair_proposal_eligible(&peer.verify,peer.history.step_limit()?)?)
    }

    pub(crate) fn execute_resident_pair(&mut self,id:u64)->Result<Value> {
        ensure!(self.resident_pair_can_step(id)?,"paired serving owners became ineligible");
        let began=Instant::now();let bank=self.bank_id;
        let(mut events,timing)={
            let pool=self.resident_pair.as_mut().context("paired executor missing")?;
            let peer=self.peer_bank.as_mut().context("paired owner missing")?;
            let a=View{id:self.current.unwrap(),session:self.session.as_mut().unwrap(),verify:&mut self.verify,
                long:&mut self.long_verify,long_buffers:&mut self.long_history_buffers,
                copy:&mut self.copy32_verify,copy_buffers:&mut self.copy32_buffers,
                history:&mut self.history,emitted:&mut self.emitted,limit:self.limit};
            let b=View{id:peer.current.unwrap(),session:peer.session.as_mut().unwrap(),verify:&mut peer.verify,
                long:&mut peer.long_verify,long_buffers:&mut peer.long_history_buffers,
                copy:&mut peer.copy32_verify,copy_buffers:&mut peer.copy32_buffers,
                history:&mut peer.history,emitted:&mut peer.emitted,limit:peer.limit};
            // Graph identity binds physical owner0 then owner1; HTTP turn order
            // must not exchange their storage or collective order.
            let(mut first,mut second)=if bank==0{(a,b)}else{(b,a)};
            let limits=[first.step_limit()?,second.step_limit()?];
            let a=first.try_wide(&self.comm,&self.stream)?;
            let b=second.try_wide(&self.comm,&self.stream)?;
            let(a,b,timing)=match(a,b) {
                (None,None)=>{
                    let(a,b,head_pooled)=first.session.prepare_pair_proposals(first.verify,first.history.pending().unwrap(),
                        second.session,second.verify,second.history.pending().unwrap(),true,&self.comm,&self.stream)?;
                    ensure!(first.session.paired_probabilities_disjoint(&a,second.session,&b)?,"paired q storage aliased");
                    let mut timing=pool.begin_with_capture_admission(first.verify,a.inputs(),second.verify,b.inputs(),
                        &self.comm,&self.stream,&mut||self.control.all_admit(
                            self.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+256*1024usize.pow(2)))?;
                    timing["pooled"]=json!(true);
                    timing["pooled_draft_head"]=json!(head_pooled);
                    let a=first.session.finish_pair_proposal(first.verify,a,&self.comm,&self.stream)?;
                    let b=second.session.finish_pair_proposal(second.verify,b,&self.comm,&self.stream)?;
                    (a,b,timing)
                },
                (a,b)=>{
                    // These wide steps have already committed normally. The
                    // remaining owner runs its usual M3/M6 path exactly once.
                    let a=if let Some(a)=a{a}else{first.ordinary(&self.comm,&self.stream,&mut self.control)?};
                    let b=if let Some(b)=b{b}else{second.ordinary(&self.comm,&self.stream,&mut self.control)?};
                    (a,b,json!({"pooled":false,"reason":"ordinary_wide_history"}))
                },
            };
            ([first.commit(a,limits[0],began)?,second.commit(b,limits[1],began)?],timing)
        };
        let proof=events.iter().map(|e|json!({"request_id":e["request_id"],"output_ids":e["output_ids"],
            "frontier":e["frontier"],"pending":e["pending"],"emitted":e["completion_tokens_so_far"]})).collect::<Vec<_>>();
        self.control.agree(&serde_json::to_vec(&json!({"paired_committed_v1":proof}))?)?;
        if bank==1 {events.swap(0,1);}
        let [mut active,peer]=events;
        active["paired_peer"]=peer;active["paired_target"]=timing;
        Ok(active)
    }
}
