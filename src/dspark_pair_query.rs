//! Independent three-stage query graphs followed by one private M10 head.
//! Markov correction and proposal sampling remain separate for each owner.
use super::*;

impl QueryNetwork {
    pub(crate) fn probe_pooled_query(&mut self,other:&mut Self,anchors:[u32;2],
        comm:&Roce,stream:&Stream)->Result<serde_json::Value> {
        use sha2::{Digest,Sha256};
        use serde_json::json;
        ensure!(std::env::var("CANADA_SCRATCH_PAIRED_GENERATION_SCREEN").as_deref()==Ok("1")
            &&self.can_pool_query_head(other),"finite paired query screen requires two admitted heads");
        let positions=[self.position()?,other.position()?];
        let mut expected=None;let mut trials=Vec::new();
        for (trial,pooled)in [false,true,true,false,false,true,true,false].into_iter().enumerate() {
            let began=std::time::Instant::now();
            if pooled {self.run_paired(other,anchors,comm,stream)?;}
            else {self.run(anchors[0],comm,stream)?;other.run(anchors[1],comm,stream)?;}
            let milliseconds=began.elapsed().as_secs_f64()*1000.;
            let digests=[&*self,&*other].into_iter().map(|n|->Result<_>{
                let normalized=n.normalized.download()?.iter().flat_map(|x|x.to_bits().to_le_bytes()).collect::<Vec<_>>();
                let logits=n.io.local_logits().download()?.iter().flat_map(|x|x.to_bits().to_le_bytes()).collect::<Vec<_>>();
                Ok(json!({"normalized":format!("{:x}",Sha256::digest(normalized)),
                    "base_logits":format!("{:x}",Sha256::digest(logits)),
                    "retained":n.accepted_repeat_digest(stream)?}))
            }).collect::<Result<Vec<_>>>()?;
            ensure!([self.position()?,other.position()?]==positions,"query screen advanced accepted state");
            if let Some(reference)=&expected {ensure!(reference==&digests,"pooled query changed inputs, FP32 base logits or retained state");}
            else{expected=Some(digests.clone());}
            trials.push(json!({"trial":trial,"pooled":pooled,"warmup":trial<2,
                "milliseconds":milliseconds,"digests":digests}));
        }
        Ok(json!({"rank":self.rank,"positions":positions,"all_exact":true,"trials":trials,
            "scope":"Whole independent query graphs plus packing/M10 head/scatter; Markov sampling excluded. Warm reversed pairs, private FP32 logits and accepted state exact."}))
    }
    pub(crate) fn can_pool_query_head(&self,other:&Self)->bool {
        self.pair_head.is_some()&&other.pair_head.is_some()
            &&self.private_head_active&&other.private_head_active
            &&!self.gather_base_logits&&!other.gather_base_logits
            &&self.private_head.as_ref().zip(other.private_head.as_ref())
                .is_some_and(|(a,b)|a.can_pool_with(b))
    }
    pub(crate) fn run_paired(&mut self,other:&mut Self,anchors:[u32;2],
        comm:&Roce,stream:&Stream)->Result<()> {
        self.guard(stream)?;other.guard(stream)?;
        ensure!(self.can_pool_query_head(other)&&anchors.iter().all(|a|*a<129280)
            &&comm.rank()==self.rank&&self.rank==other.rank
            &&self.comm_instance.is_none_or(|c|c==comm.instance())
            &&other.comm_instance.is_none_or(|c|c==comm.instance())
            &&Rc::ptr_eq(&self.weights,&other.weights)&&self.state.independent_from(&other.state)?
            &&self.normalized.ptr()!=other.normalized.ptr()
            &&self.io.local_logits().ptr()!=other.io.local_logits().ptr(),
            "paired query requires independent owners on one canonical atlas/stream");
        let result=(||->Result<()> {
            for (network,anchor) in [(&mut *self,anchors[0]),(&mut *other,anchors[1])] {
                network.ready=false;network.last_anchor=None;network.markov.invalidate_results();
                let layout=network.state.prepare_query(stream)?;
                network.positions.update(&layout.positions)?;
                network.tokens.update(&[i64::from(anchor),128799,128799,128799,128799],QUERY_ROWS)?;
                comm.select_decode_reduction(true,stream)?;
                comm.select_prefill_rows(QUERY_ROWS,QUERY_ROWS,stream)?;
                stream.join_default()?;
                if network.pair_body_graph.is_none() {
                    network.enqueue_body(comm,stream)?;stream.synchronize()?;comm.health()?;
                    network.pair_body_graph=Some(stream.capture(||network.enqueue_body(comm,stream))?);
                    network.comm_instance=Some(comm.instance());
                }
                unsafe{network.pair_body_graph.as_ref().unwrap().launch(stream)?;}
            }
            self.pair_head.as_mut().unwrap().run(
                [self.private_head.as_ref().unwrap(),other.private_head.as_ref().unwrap()],
                [&self.normalized,&other.normalized],[self.io.local_logits(),other.io.local_logits()],stream)?;
            self.state.publish_queries(stream)?;other.state.publish_queries(stream)?;comm.health()?;
            Ok(())
        })();
        for (network,anchor) in [(self,anchors[0]),(other,anchors[1])] {
            if result.is_err(){network.poisoned=true;network.state.invalidate();}
            else{network.ready=true;network.last_anchor=Some(anchor);}
        }
        result
    }
}
