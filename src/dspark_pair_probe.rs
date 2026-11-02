//! Finite neural-draft staging around a shared target forward. Proposal q,
//! Markov draws, rejection sampling and accepted conditioning remain private.
//! Wider history proposals and context-edge fallback execute separately.
use super::*;

pub(crate) struct PreparedProposal {
    owner:Rc<()>,
    target_owner:crate::session_lease::Lease,
    start:u64,
    seed:u64,
    round_seed:u64,
    verify:[u32;6],
    draft_tokens:[u32;5],
    confidence:Option<[f32;5]>,
    lookup:Option<crate::history_lookup::Proposal>,
    produced:usize,
    live:usize,
    began:Instant,
    drafted:Instant,
}
impl PreparedProposal {
    pub(crate) fn inputs(&self)->&[u32] {&self.verify[..self.live]}
    pub(crate) fn matches_target(&self,target:&Target)->bool {target.pair_proposal_matches(&self.target_owner)}
}

impl GreedySession {
    pub(crate) fn probe_pooled_query(&mut self,other:&mut Self,anchors:[u32;2],
        comm:&Roce,stream:&Stream)->Result<serde_json::Value> {
        ensure!(!self.poisoned&&!other.poisoned,"query screen requires committed sessions");
        self.draft.probe_pooled_query(&mut other.draft,anchors,comm,stream)
    }
    /// Inspect only accepted state and configured policy. Ineligible requests
    /// retain the ordinary path; do not consume then discard any proposal draw.
    pub(crate) fn pair_proposal_eligible(&self,target:&Target,limit:usize)->Result<bool> {
        Ok(limit>=6&&!self.poisoned&&target.profile_is_current()
            &&self.draft.position()?==target.position()
            &&self.target_seed.is_some()&&self.rejection.is_some()&&self.block_verification
            &&self.history_lookup.as_ref().is_none_or(|h|h.position()==target.position())
            &&!self.stochastic_prefix_schedule
            &&self.causal_short_rows==3&&self.causal_full_threshold.is_none()
            &&!self.profile_verify&&self.reasoning_temperature_probe.is_none()
            &&self.root_feedback.is_none()&&self.encoder_cache_trace.is_none()&&self.buffered_draw_trace.is_none()
            &&target.context_capacity().saturating_sub(target.position()as usize)>=6)
    }
    pub(crate) fn prepare_pair_proposal(&mut self,target:&Target,anchor:u32,
        comm:&Roce,stream:&Stream)->Result<PreparedProposal> {
        let proposal=self.begin_pair_proposal(target,anchor)?;
        if proposal.lookup.is_none(){self.draft.run(anchor,comm,stream)?;}
        self.finish_preparing_pair(proposal,comm,stream)
    }
    /// Inspect both accepted histories before any draft draw. Only two neural
    /// proposals can share a head; point-mass history and ordinary fallbacks
    /// keep their existing q and acceptance law.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_pair_proposals(&mut self,target:&Target,anchor:u32,
        other:&mut Self,peer:&Target,peer_anchor:u32,pool:bool,comm:&Roce,stream:&Stream)
        ->Result<(PreparedProposal,PreparedProposal,bool)> {
        if !pool||!self.draft.can_pool_query_head(&other.draft) {
            let a=self.prepare_pair_proposal(target,anchor,comm,stream)?;
            let b=other.prepare_pair_proposal(peer,peer_anchor,comm,stream)?;
            return Ok((a,b,false));
        }
        let a=self.begin_pair_proposal(target,anchor)?;
        let b=other.begin_pair_proposal(peer,peer_anchor)?;
        let pooled=a.lookup.is_none()&&b.lookup.is_none();
        if pooled {self.draft.run_paired(&mut other.draft,[anchor,peer_anchor],comm,stream)?;}
        else {
            if a.lookup.is_none(){self.draft.run(anchor,comm,stream)?;}
            if b.lookup.is_none(){other.draft.run(peer_anchor,comm,stream)?;}
        }
        let a=self.finish_preparing_pair(a,comm,stream)?;
        let b=other.finish_preparing_pair(b,comm,stream)?;
        Ok((a,b,pooled))
    }
    fn begin_pair_proposal(&mut self,target:&Target,anchor:u32)->Result<PreparedProposal> {
        ensure!((std::env::var("CANADA_SCRATCH_PAIRED_GENERATION_SCREEN").as_deref()==Ok("1")
            ||crate::config::paired_serving_requested())&&self.pair_proposal_eligible(target,6)?,
            "paired proposal policy/owner/frontier");
        self.poisoned=true;
        let began=Instant::now();let start=target.position();let seed=self.target_seed.unwrap();
        let round_seed=crate::categorical::block_round_seed(seed,start);
        let fallback_warm=self.draft_warmed||self.draft.sampled_graphs_ready();
        let lookup=if fallback_warm {self.history_lookup.as_ref()
            .map(|h|h.propose(start,anchor)).transpose()?.flatten()}else{None};
        Ok(PreparedProposal{owner:self.image_owner.clone(),target_owner:target.pair_proposal_owner()?,start,seed,round_seed,
            verify:[anchor;6],draft_tokens:[anchor;5],confidence:None,lookup,produced:0,live:0,began,drafted:began})
    }
    fn finish_preparing_pair(&mut self,mut proposal:PreparedProposal,comm:&Roce,stream:&Stream)
        ->Result<PreparedProposal> {
        ensure!(self.poisoned&&Rc::ptr_eq(&self.image_owner,&proposal.owner)
            &&self.target_seed==Some(proposal.seed)&&self.draft.position()?==proposal.start,
            "pair preparation lost draft authority");
        let (draft_tokens,confidence,live,produced)=if let Some(history)=&proposal.lookup {
            // Deterministic history proposals have point-mass q. Do not touch
            // stale neural probabilities or confidence on this path.
            (history.tokens,None,6,5)
        }else{
            let tokens=if let Some(threshold)=self.causal_three_threshold {
                self.draft.stochastic_proposals_causal_three(proposal.round_seed,threshold,comm,stream)?
            }else{self.draft.stochastic_proposals(proposal.round_seed,comm,stream)?};
            let(confidence,_)=self.draft.confidence_schedule()?;
            let live=self.causal_three_threshold.map_or(6,|t|causal_prefix_rows(&confidence,t,3,6));
            self.draft_warmed=true;
            (tokens,Some(confidence),live,self.draft.produced_proposals()?)
        };
        ensure!([3,6].contains(&live)&&live-1<=produced,"paired proposal causal extent");
        proposal.verify[1..].copy_from_slice(&draft_tokens);
        proposal.draft_tokens=draft_tokens;proposal.confidence=confidence;proposal.produced=produced;
        proposal.live=live;proposal.drafted=Instant::now();Ok(proposal)
    }
    pub(crate) fn finish_pair_proposal(&mut self,target:&mut Target,proposal:PreparedProposal,
        comm:&Roce,stream:&Stream)->Result<Step> {
        ensure!(self.poisoned&&Rc::ptr_eq(&self.image_owner,&proposal.owner)&&proposal.matches_target(target)
            &&self.target_seed==Some(proposal.seed)&&target.position()==proposal.start
            &&self.draft.position()?==proposal.start&&self.block_verification
            &&self.history_lookup.as_ref().is_none_or(|h|h.position()==proposal.start),
            "paired acceptance lost request/proposal authority");
        let result=(||->Result<Step>{
            let(logits,live)=target.pending_logits()?;
            ensure!(live==proposal.live,"paired target/proposal length mismatch");
            let(predictions,prefix,expected_lengths)=if proposal.lookup.is_some() {
                (target.verification_sample(proposal.seed,stream)?,None,None)
            }else{
                let rejection=self.rejection.as_mut().ok_or_else(||anyhow::anyhow!("missing pair rejection sampler"))?;
                let draw=rejection.sample_block(logits,self.draft.proposal_probabilities_prefix(live-1)?,
                    &proposal.draft_tokens[..live-1],live,proposal.round_seed,proposal.start+1,stream)?;
                (draw.ids,draw.accepted_prefix,draw.acceptance_expected_lengths)
            };
            ensure!(predictions.len()==live,"paired prediction extent");
            let matched=if let Some(prefix)=prefix {prefix}else{
                let mut n=0;while n+1<live&&proposal.draft_tokens[n]==predictions[n]{n+=1;}n
            };
            let accepted=stop_frontier(&predictions,(1+matched).min(6),self.stop_token);
            let verify_host_ms=target.verification_host_ms()?;
            let verified=Instant::now();
            self.features.run(target,stream)?;
            target.finish_verification(accepted,comm,stream)?;
            let committed=Instant::now();
            let(features,rows)=self.features.values()?;
            self.draft.inject_distributed(features,rows,accepted,proposal.start,comm,stream)?;
            if let Some(history)=&mut self.history_lookup {
                history.commit(proposal.start,&proposal.verify[..accepted])?;
                ensure!(history.position()==target.position(),"paired lookup/target commit disagreement");
            }
            ensure!(target.position()==proposal.start+accepted as u64&&self.draft.position()?==target.position(),
                "paired target/drafter commit disagreement");
            Ok(Step{tokens:proposal.verify[..accepted].to_vec(),next_anchor:predictions[accepted-1],
                draft_tokens:proposal.draft_tokens[..proposal.produced].to_vec(),matching_proposals:accepted-1,
                verified_rows:rows,draft_confidence:proposal.confidence,
                proposal_source:if proposal.lookup.is_some(){"history_point_mass"}else{"dspark"},
                lookup_match_tokens:proposal.lookup.as_ref().map(|p|p.matched),
                lookup_source_end:proposal.lookup.as_ref().map(|p|p.source_end),profiled_verify:false,profile_append_metrics:None,
                position:target.position(),verify_host_ms,acceptance_expected_lengths:expected_lengths,
                acceptance_temperature_probe:None,
                phase_ms:[proposal.drafted.duration_since(proposal.began).as_secs_f64()*1000.,
                    verified.duration_since(proposal.drafted).as_secs_f64()*1000.,
                    committed.duration_since(verified).as_secs_f64()*1000.,committed.elapsed().as_secs_f64()*1000.]})
        })();
        if result.is_ok(){self.poisoned=false;}result
    }
    pub(crate) fn paired_draft_digest(&self,stream:&Stream)->Result<serde_json::Value> {
        ensure!(std::env::var("CANADA_SCRATCH_PAIRED_GENERATION_SCREEN").as_deref()==Ok("1")&&!self.poisoned,
            "finite committed pair draft digest");
        let kv=self.draft.accepted_repeat_digest(stream)?;
        Ok(if let Some(history)=&self.history_lookup {
            serde_json::json!({"kv":kv,"history":history.state_digest()})
        }else{kv})
    }
    pub(crate) fn paired_probabilities_disjoint(&self,a:&PreparedProposal,other:&Self,b:&PreparedProposal)->Result<bool> {
        ensure!(self.poisoned&&other.poisoned&&Rc::ptr_eq(&self.image_owner,&a.owner)
            &&Rc::ptr_eq(&other.image_owner,&b.owner),"pair q ownership check");
        if a.lookup.is_some()||b.lookup.is_some(){return Ok(!Rc::ptr_eq(&self.image_owner,&other.image_owner));}
        let x=self.draft.proposal_probabilities_prefix(a.live-1)?;
        let y=other.draft.proposal_probabilities_prefix(b.live-1)?;
        Ok(!Rc::ptr_eq(&self.image_owner,&other.image_owner)
            &&(x.ptr()+x.bytes()as u64<=y.ptr()||y.ptr()+y.bytes()as u64<=x.ptr()))
    }
}
