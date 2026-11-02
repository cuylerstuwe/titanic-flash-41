//! Fixed accepted histories, actual stochastic q, and GPU acceptance metrics.
//! Diagnostic calls never publish a sampled proposal or change serving policy.
use super::*;
use anyhow::Context;
use serde_json::{json,Value};

impl GreedySession {
    pub(crate) fn matched_path_frontier(&mut self,target:&mut Target,anchor:u32,
        seeds:&[u64],shadow:&mut Rejection,comm:&Roce,stream:&Stream)->Result<Value> {
        ensure!(std::env::var("CANADA_SCRATCH_MATCHED_PATH_SCREEN").as_deref()==Ok("1")
            &&!self.poisoned&&self.target_seed.is_some()&&self.block_verification
            &&target.profile_is_current()&&target.physical_rows()==6
            &&self.draft.position()?==target.position()&&(4..=32).contains(&seeds.len())&&anchor<129280,
            "matched path frontier admission");
        let start=target.position();
        let carry=target.carry_digests()?;let encoder=target.encoder_tail_digest()?;
        let history=self.history_lookup.as_ref().map(HistoryLookup::state_digest);
        let accepted_draft=self.draft.accepted_repeat_digest(stream)?;
        let lookup=self.history_lookup.as_ref().map(|h|h.propose(start,anchor)).transpose()?.flatten();
        let began=Instant::now();self.draft.run(anchor,comm,stream)?;
        let query_seconds=began.elapsed().as_secs_f64();
        let economics=std::env::var("CANADA_SCRATCH_MATCHED_PRECISION_ECONOMICS").as_deref()==Ok("1");
        let mut records=Vec::new();
        // Full proposals expose both possible causal widths. Selecting three
        // uses only the first three confidences, before any excluded draw.
        for (ordinal,&seed) in seeds.iter().chain(seeds[..1].iter()).enumerate() {
            let began=Instant::now();let proposals=self.draft.stochastic_proposals(seed,comm,stream)?;
            let proposal_seconds=began.elapsed().as_secs_f64();
            let(confidence,scheduled)=self.draft.confidence_schedule()?;
            let selected=if let Some(threshold)=self.causal_full_threshold {
                causal_full_prefix_rows(&confidence,threshold,6)
            }else if let Some(threshold)=self.causal_three_threshold {
                causal_prefix_rows(&confidence,threshold,self.causal_short_rows,6)
            }else if self.stochastic_prefix_schedule {scheduled}else{6};
            let mut inputs=[anchor;6];inputs[1..].copy_from_slice(&proposals);
            let began=Instant::now();target.begin_verification(&inputs,comm,stream)?;
            let verify6_seconds=began.elapsed().as_secs_f64();
            let(logits,live)=target.pending_logits()?;ensure!(live==6,"matched proposal rows");
            let draw=shadow.sample_block(logits,self.draft.proposal_probabilities()?,
                &proposals,6,seed,start+1,stream)?;
            let expected=draw.acceptance_expected_lengths.ok_or_else(||anyhow::anyhow!("matched GPU expectations absent"))?;
            ensure!(expected.len()==6&&(1..=6).contains(&selected),"matched expected-length extent");
            let local=draw.local_proposal_fit.ok_or_else(||anyhow::anyhow!("matched overlap metrics absent"))?;
            let result=json!({"seed":seed,"proposals":proposals,"confidence":confidence,
                "selected_neural_rows":selected,"expected_committed_lengths":expected,
                "selected_expected_lengths":expected[selected-1],
                "first_proposal_overlap":draw.first_proposal_overlap,
                "local_overlaps":local.0,"sequential_rb_length":local.1,
                "overlap_product_heuristic":local.2});
            target.finish_verification(0,comm,stream)?;
            let mut actual_m3_metrics=None;
            let verify3_seconds=if economics {
                let began=Instant::now();target.begin_verification(&inputs[..3],comm,stream)?;
                let seconds=began.elapsed().as_secs_f64();
                // Score the actual short path: do not assume the first three
                // M6 logits have identical numerics to a live-three forward.
                let(logits,live)=target.pending_logits()?;ensure!(live==3,"matched short rows");
                let draw=shadow.sample_block(logits,self.draft.proposal_probabilities()?,
                    &proposals[..2],3,seed,start+1,stream)?;
                let expected=draw.acceptance_expected_lengths.context("matched M3 expectations absent")?;
                ensure!(expected.len()==3,"matched M3 expectation extent");
                actual_m3_metrics=Some(json!({"expected_committed_lengths":expected,
                    "first_proposal_overlap":draw.first_proposal_overlap,
                    "local_proposal_fit":draw.local_proposal_fit}));
                target.finish_verification(0,comm,stream)?;Some(seconds)
            }else{None};
            ensure!(target.position()==start&&self.draft.position()?==start
                &&target.carry_digests()?==carry&&target.encoder_tail_digest()?==encoder
                &&self.history_lookup.as_ref().map(HistoryLookup::state_digest)==history
                &&self.draft.accepted_repeat_digest(stream)?==accepted_draft,
                "matched proposal scoring changed committed model/drafter/history state");
            records.push(json!({"repeat_of_first":ordinal==seeds.len(),"metrics":result,"actual_m3_metrics":actual_m3_metrics,
                "timing":{"proposal_seconds":proposal_seconds,"verify6_seconds":verify6_seconds,
                    "verify3_seconds":verify3_seconds}}));
        }
        Ok(json!({"position":start,"anchor":anchor,"draft_temperature":self.draft.draft_temperature(),
            "history_match":lookup.map(|p|json!({"matched":p.matched,"source_end":p.source_end})),
            "same_seed_repeat_exact":records[0]["metrics"]==records[seeds.len()]["metrics"]
                &&records[0]["actual_m3_metrics"]==records[seeds.len()]["actual_m3_metrics"],
            "query_seconds":query_seconds,
            "records":records,"rollback_exact":true,
            "scope":"Neural proposal fit at one fixed accepted prefix. GPU integrates acceptance randomness conditional on each sampled proposal path; finite proposal draws do not integrate all q paths. History selection is reported, not substituted into neural expected lengths. Optional M3/M6 timings exclude capture/warmup samples during analysis."}))
    }

    /// Commit an explicit continuation using the ordinary target feature and
    /// distributed draft conditioning operations, without sampling a new path.
    pub(crate) fn matched_path_commit(&mut self,target:&mut Target,ids:&[u32],
        capture:bool,comm:&Roce,stream:&Stream)->Result<Option<Vec<u8>>> {
        ensure!(std::env::var("CANADA_SCRATCH_MATCHED_PATH_SCREEN").as_deref()==Ok("1")
            &&!self.poisoned&&ids.len()==6&&target.physical_rows()==6
            &&self.draft.position()?==target.position(),"matched path commit admission");
        let start=target.position();self.poisoned=true;
        target.begin_verification(ids,comm,stream)?;
        let bytes=if capture {
            let(logits,live)=target.pending_logits()?;ensure!(live==6,"matched teacher rows");
            stream.synchronize()?;Some(unsafe{self.cuda.read_device(logits.ptr(),6*129280*4)}?)
        }else{None};
        self.features.run(target,stream)?;
        target.finish_verification(6,comm,stream)?;
        let(features,live)=self.features.values()?;ensure!(live==6,"matched conditioning rows");
        self.draft.inject_distributed(features,live,6,start,comm,stream)?;
        if let Some(history)=&mut self.history_lookup {history.commit(start,ids)?;}
        ensure!(target.position()==start+6&&self.draft.position()?==target.position()
            &&self.history_lookup.as_ref().is_none_or(|h|h.position()==target.position()),
            "matched teacher commit frontiers disagree");
        self.poisoned=false;Ok(bytes)
    }
}
