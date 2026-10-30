//! Partial-prompt ownership for opt-in background requests. Ordinary foreground
//! prefill keeps its existing path. Encoder chunks keep their existing widths;
//! only an actual interruption materializes an extra bounded decoder tail.
use super::*;

#[derive(Clone)]
pub(super) struct PendingPrefill {
    plan: RequestPlan,
    greedy: bool,
    cursor: usize,
    reuse: bool,
    checkpoint: usize,
    checkpoint_done: bool,
    cache: Value,
    checkpoint_receipt: Value,
    target_seconds: f64,
    active_seconds: f64,
}
impl PendingPrefill {
    pub(super) fn owned_token_bytes(&self)->usize { self.plan.prompt.len()*4+self.plan.media.metadata_bytes() }
}

#[cfg(feature="serving")]
impl Worker {
    pub(super) fn start_background(&mut self,command:Command,
        queue:Option<&crate::http_server::Queue>,job:Option<&crate::http_server::Job>)->Result<Value> {
        let began=Instant::now();
        let Command::Begin{images,id,seed,greedy,limit,tokens}=command else {
            anyhow::bail!("background start requires a Begin command")
        };
        ensure!(self.current.is_none()&&self.pending_prefill.is_none(),"background start ownership");
        let media=self.prompt_identity(&images,&tokens)?;
        let mut plan=self.history.prepare_with_media(id,tokens,media,limit,seed)?;
        // Keep the vetted interruptible prefill path. The optional exact-end
        // logit shortcut is enabled only by the foreground Begin handler.
        let (cache,deferred_encoder)=self.reuse_cached_session(&mut plan,false,false)?;
        ensure!(deferred_encoder.is_none(),"background prefill uses complete cache images");
        if let Err(error)=self.prepare_media(&images,&plan,if plan.reuse {plan.previous}else{0}) {
            return self.refuse_media_input(id,error);
        }
        if self.session.is_some() {
            ensure!(self.verify.profile_is_current()&&self.verify.position()==plan.previous as u64,
                "background GPU owner disagrees with history");
            if !plan.reuse {
                self.target_lineage=self.target_lineage.checked_add(1)
                    .ok_or_else(||anyhow::anyhow!("background target lineage exhausted"))?;
                self.verify.reset(self.target_lineage,&self.comm,&self.stream)?;
            }
            self.large.handoff_from(&self.verify)?;
        }else{ensure!(!plan.reuse,"background reuse before initial prefill");}
        let cursor=if plan.reuse {plan.previous}else{0};
        let checkpoint=(plan.prompt.len().saturating_sub(128)/128)*128;
        let checkpoint=if cache["hit"]==true {checkpoint.max(cursor)}else{checkpoint};
        let checkpoint_done=!(self.prompt_checkpoint&&checkpoint>=8192&&checkpoint>=cursor);
        self.current=Some(id);self.emitted=0;self.limit=limit;
        self.pending_prefill=Some(PendingPrefill{reuse:plan.reuse,plan,greedy,cursor,checkpoint,
            checkpoint_done,cache,checkpoint_receipt:json!({"enabled":self.prompt_checkpoint,"saved":false}),
            target_seconds:0.,active_seconds:began.elapsed().as_secs_f64()});
        self.continue_background(queue,job)
    }

    pub(super) fn continue_background(&mut self,
        queue:Option<&crate::http_server::Queue>,job:Option<&crate::http_server::Job>)->Result<Value> {
        let began=Instant::now();
        let mut pending=self.pending_prefill.take().ok_or_else(||anyhow::anyhow!("no partial background prefill"))?;
        ensure!(self.current==Some(pending.plan.id)&&self.emitted==0,
            "partial prefill request identity or visibility");
        loop {
            if !self.large.profile_is_current() {self.large.handoff_from(&self.verify)?;}
            let end=if pending.checkpoint_done {pending.plan.prompt.len()}else{pending.checkpoint};
            if end>pending.cursor {
                let control=&mut self.control;
                let mut should_yield=|_done:usize|->Result<bool>{
                    let stop=queue.map(|q|q.foreground_waiting().map(|foreground|
                        foreground||q.router_hold()||q.stopping()||job.is_some_and(|j|j.cancelled()))).transpose()?;
                    let head=stop.map(|v|[u8::from(v)]);
                    let wire=control.command(head.as_ref().map(|v|v.as_slice()))?;
                    ensure!(wire.len()==1&&wire[0]<=1,"background encoder yield command");
                    Ok(wire[0]==1)
                };
                let pre=Instant::now();
                let done=self.large.prefill_wide_interruptible(&mut self.wides,
                    &pending.plan.prompt[pending.cursor..end],&self.comm,&self.stream,&mut should_yield)?;
                pending.target_seconds+=pre.elapsed().as_secs_f64();
                let frontier=pending.cursor+done;
                let complete=frontier==pending.plan.prompt.len();
                let anchor=if !complete||pending.greedy {self.large.greedy_token(&self.stream)?}
                    else{self.large.sample_token(pending.plan.seed,&self.stream)?};
                self.attach_prompt(pending.reuse,pending.cursor,&pending.plan.prompt[pending.cursor..frontier],
                    &pending.plan.prompt[..frontier],pending.plan.seed,pending.greedy)?;
                self.verify.handoff_from(&self.large)?;
                pending.cursor=frontier;pending.reuse=true;
                if complete {
                    let output=self.history.prefilled(frontier,anchor)?;
                    self.emitted=1;
                    self.install_media(None)?;
                    let lookup=self.session.as_ref().unwrap().history_lookup_metadata();
                    ensure!(lookup.is_some()==self.history_lookup_enabled
                        &&lookup.is_none_or(|(position,_)|position==frontier as u64),"background history lookup frontier");
                    if !self.progress_next {self.agree(&output)?;}
                    pending.active_seconds+=began.elapsed().as_secs_f64();
                    return Ok(json!({"type":"tokens","request_id":self.current,"output_ids":output,
                        "prompt_tokens":pending.plan.prompt.len(),"cached_tokens":if pending.plan.reuse {pending.plan.previous}else{0},
                        "new_prompt_tokens":pending.plan.delta().len(),"frontier":frontier,"pending":anchor,
                        "seconds":pending.active_seconds,"background_prefill":true,
                        "begin_phases_seconds":{"target_prefill":pending.target_seconds},
                        "retained_session_cache":pending.cache,"prompt_checkpoint":pending.checkpoint_receipt,
                        "target_temperature":if pending.greedy {0.}else{1.},"target_top_p":1.,
                        "draft_temperature":self.session.as_ref().unwrap().draft_temperature(),
                        "draft_head":self.session.as_ref().unwrap().draft_head_kind(),
                        "history_lookup_position":lookup.map(|(p,_)|p)}));
                }
                self.history.partial_prefilled(frontier,anchor)?;
                if frontier<end {
                    pending.active_seconds+=began.elapsed().as_secs_f64();
                    let event=json!({"type":"prefill_progress","request_id":self.current,
                        "frontier":frontier,"prompt_tokens":pending.plan.prompt.len(),"emitted_tokens":0,
                        "cached_tokens":if pending.plan.reuse {pending.plan.previous}else{0},
                        "target_prefill_seconds":pending.target_seconds,"active_seconds":pending.active_seconds});
                    self.pending_prefill=Some(pending);
                    self.agree(&[])?;
                    return Ok(event);
                }
            }else{
                self.verify.handoff_from(&self.large)?;
            }
            ensure!(!pending.checkpoint_done&&pending.cursor==pending.checkpoint,
                "background prefill checkpoint frontier");
            let anchor=self.history.pending().ok_or_else(||anyhow::anyhow!("background checkpoint anchor"))?;
            pending.checkpoint_receipt=self.retain_cached_prefix(&pending.plan.prompt[..pending.checkpoint],anchor,&pending.plan.media)?;
            pending.checkpoint_receipt["enabled"]=json!(true);
            pending.checkpoint_receipt["prefix_tokens"]=json!(pending.checkpoint);
            pending.checkpoint_receipt["replayed_suffix_tokens"]=json!(pending.plan.prompt.len()-pending.checkpoint);
            pending.checkpoint_done=true;
        }
    }
}
