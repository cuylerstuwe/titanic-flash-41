//! Partial-prompt ownership for background requests and opt-in peer yields.
//! Uncontended foreground prefill keeps its existing path and chunk widths;
//! only an actual interruption materializes an extra bounded decoder tail.
use super::*;

#[derive(Clone)]
pub(super) struct PendingPrefill {
    plan: RequestPlan,
    greedy: bool,
    cursor: usize,
    reuse: bool,
    checkpoints: std::collections::VecDeque<usize>,
    remaining_chunks: Vec<usize>,
    cache: Value,
    checkpoint_receipt: Value,
    target_seconds: f64,
    active_seconds: f64,
    background: bool,
    cached_tokens: usize,
}
impl PendingPrefill {
    pub(super) fn remaining_tokens(&self)->usize {self.plan.prompt.len()-self.cursor}
    pub(super) fn owned_token_bytes(&self)->usize {
        self.plan.prompt.len()*4+self.plan.media.metadata_bytes()
            +(self.checkpoints.capacity()+self.remaining_chunks.capacity())*std::mem::size_of::<usize>()
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn foreground(plan:RequestPlan,greedy:bool,cursor:usize,cache:Value,
        target_seconds:f64,active_seconds:f64,checkpoint_enabled:bool,
        checkpoints:Vec<usize>,remaining_chunks:Vec<usize>)->Result<Self> {
        let cached_tokens=if plan.reuse{plan.previous}else{cache["restored_tokens"].as_u64().unwrap_or(0)as usize};
        ensure!(!remaining_chunks.is_empty()&&checkpoints.iter().all(|&c|c>cursor&&c<plan.prompt.len())
            &&checkpoints.windows(2).all(|p|p[0]<p[1]),"resumable prefill schedule extent");
        Ok(Self{plan,greedy,cursor,reuse:true,checkpoints:checkpoints.into(),remaining_chunks,
            cache,checkpoint_receipt:json!({"enabled":checkpoint_enabled,"saved":false}),
            target_seconds,active_seconds,background:false,cached_tokens})
    }
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
        let checkpoints=if self.prompt_checkpoint&&checkpoint>=8192&&checkpoint>=cursor {
            [checkpoint].into_iter().collect()
        }else{std::collections::VecDeque::new()};
        self.current=Some(id);self.emitted=0;self.limit=limit;
        self.pending_prefill=Some(PendingPrefill{cached_tokens:if plan.reuse{plan.previous}else{0},
            background:true,reuse:plan.reuse,plan,greedy,cursor,checkpoints,remaining_chunks:Vec::new(),
            cache,checkpoint_receipt:json!({"enabled":self.prompt_checkpoint,"saved":false}),
            target_seconds:0.,active_seconds:began.elapsed().as_secs_f64()});
        self.continue_background(queue,job)
    }

    pub(super) fn continue_background(&mut self,
        queue:Option<&crate::http_server::Queue>,job:Option<&crate::http_server::Job>)->Result<Value> {
        let background=self.pending_prefill.as_ref().context("missing resumable prefill")?.background;
        let peers=self.peer_prefill_enabled;let slice=self.peer_prefill_slice;let began=Instant::now();
        self.continue_prefill(&mut |control,remaining,cached| {
            let stop=queue.map(|q|->Result<bool>{
                let priority=background&&(q.foreground_can_join()?||q.router_hold());
                let peer=peers&&q.prefill_peer_should_yield(background,remaining,q.prefill_slice_due(began.elapsed(),slice),
                    |peer|estimated_peer_remaining(peer,cached,None))?;
                Ok(priority||peer||q.stopping()||job.is_some_and(|j|j.cancelled()))
            }).transpose()?;
            let head=stop.map(|v|[u8::from(v)]);
            let wire=control.command(head.as_ref().map(|v|v.as_slice()))?;
            ensure!(wire.len()==1&&wire[0]<=1,"resumable encoder yield command");Ok(wire[0]==1)
        })
    }
    pub(super) fn continue_prefill(&mut self,
        should_yield:&mut impl FnMut(&mut ControlChannel,usize,&[CachedSession])->Result<bool>)->Result<Value> {
        let began=Instant::now();
        let mut pending=self.pending_prefill.take().ok_or_else(||anyhow::anyhow!("no partial background prefill"))?;
        ensure!(self.current==Some(pending.plan.id)&&self.emitted==0,
            "partial prefill request identity or visibility");
        loop {
            if !self.large.profile_is_current() {self.large.handoff_from(&self.verify)?;}
            let end=pending.checkpoints.front().copied().unwrap_or(pending.plan.prompt.len());
            if end>pending.cursor {
                let control=&mut self.control;
                let pre=Instant::now();
                let chunks=if pending.remaining_chunks.is_empty(){None}
                    else{Some(std::mem::take(&mut pending.remaining_chunks))};
                let progress=self.large.prefill_wide_interruptible(&mut self.wides,
                    &pending.plan.prompt[pending.cursor..end],&self.comm,&self.stream,chunks,&mut |done|
                        should_yield(control,pending.plan.prompt.len()-pending.cursor-done,&self.cached_sessions))?;
                pending.remaining_chunks=progress.remaining_chunks;
                pending.target_seconds+=pre.elapsed().as_secs_f64();
                let frontier=pending.cursor+progress.done;
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
                        "prompt_tokens":pending.plan.prompt.len(),"cached_tokens":pending.cached_tokens,
                        "new_prompt_tokens":pending.plan.prompt.len()-pending.cached_tokens,"frontier":frontier,"pending":anchor,
                        "seconds":pending.active_seconds,"background_prefill":pending.background,"resumed_prefill":true,
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
                        "cached_tokens":pending.cached_tokens,
                        "target_prefill_seconds":pending.target_seconds,"active_seconds":pending.active_seconds});
                    self.pending_prefill=Some(pending);
                    self.agree(&[])?;
                    return Ok(event);
                }
            }else{
                self.verify.handoff_from(&self.large)?;
            }
            ensure!(pending.checkpoints.front()==Some(&pending.cursor)&&pending.remaining_chunks.is_empty(),
                "background prefill checkpoint frontier");
            let checkpoint=pending.checkpoints.pop_front().unwrap();
            let anchor=self.history.pending().ok_or_else(||anyhow::anyhow!("background checkpoint anchor"))?;
            pending.checkpoint_receipt=if !pending.background&&crate::config::bank_encoder_checkpoint_requested() {
                self.retain_resumed_bank_encoder(&pending.plan.prompt[..checkpoint],&pending.plan.media)?
            }else{self.retain_cached_prefix(&pending.plan.prompt[..checkpoint],anchor,&pending.plan.media)?};
            pending.checkpoint_receipt["enabled"]=json!(true);
            pending.checkpoint_receipt["prefix_tokens"]=json!(checkpoint);
            pending.checkpoint_receipt["replayed_suffix_tokens"]=json!(pending.plan.prompt.len()-checkpoint);
        }
    }
}
