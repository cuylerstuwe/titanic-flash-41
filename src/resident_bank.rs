//! Two physical request owners, selected at committed boundaries. Immutable
//! model data and sequential prefill scratch are shared; causal state, sampler,
//! drafter, media bindings and request progress move together by Rust ownership.
//! Completed cache images remain in one globally budgeted, owner-tagged vector.
use super::*;

#[path = "resident_pair.rs"]
mod paired;

pub(super) struct Bank {
    pub(super) id:u8,
    session:Option<GreedySession>,
    prepared_session:Option<PreparedSession>,
    session_prepare_seconds:f64,
    target_lineage:u64,
    verify:Target,
    long_verify:Option<Target>,
    long_history_buffers:Option<LongHistoryBuffers>,
    copy32_verify:Option<Target>,
    copy32_buffers:Option<LongHistoryBuffers>,
    copy32_admission:Value,
    wides:Vec<Target>,
    large:Target,
    pub(super) history:ResidentHistory,
    pub(super) current:Option<u64>,
    pending_prefill:Option<PendingPrefill>,
    prefill_cancelled:bool,
    active_media:Option<std::rc::Rc<crate::prompt_media::Bindings>>,
    media_receipt:Value,
    emitted:usize,
    limit:usize,
}
impl Bank {
    fn encoder_cache_target(&self)->Option<&Target> {
        if self.verify.profile_is_current(){Some(&self.verify)}
        else if self.large.profile_is_current(){Some(&self.large)}else{None}
    }
    pub(super) fn encoder_import_source(&self,cached:&CachedSession)->Result<&Target> {
        ensure!(crate::config::encoder_prefix_import_requested()&&cached.owner==self.id&&self.committed(),
            "encoder import donor must be the known committed peer");
        let CachedState::Encoder{image,..}=&cached.image else{anyhow::bail!("only immutable encoder parents may cross owners");};
        let source=if self.session.is_some(){&self.verify}else{&self.large};
        ensure!(source.encoder_prefix_reusable_from(image),"encoder image no longer belongs to its recorded donor frontier");
        Ok(source)
    }
    pub(super) fn retire(&mut self){self.history.retire();}
    pub(super) fn prefill_remaining(&self)->Option<usize> {
        self.pending_prefill.as_ref().map(PendingPrefill::remaining_tokens)
    }
    fn committed(&self)->bool {
        if self.session.is_some() {self.verify.profile_is_current()
            &&self.verify.position()==self.history.frontier()as u64}
        else {self.current.is_none()&&self.history.frontier()==0&&self.large.position()==0}
    }
}

pub(super) fn enabled()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_RESIDENT_BANKS") {
        Ok(v)=>{ensure!(v=="0"||v=="1","resident banks flag");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(false),Err(e)=>Err(e.into()),
    }
}

impl Worker {
    /// Resumed foreground prefill reaches a complete target/drafter boundary.
    /// Keep its encoder tail under the same budget as uninterrupted prefill;
    /// copying the complete prefix here defeats long-conversation retention.
    pub(super) fn retain_resumed_bank_encoder(&mut self,prefix:&[u32],
        media:&crate::prompt_media::Identity)->Result<Value> {
        ensure!(crate::config::bank_encoder_checkpoint_requested()&&self.encoder_prompt_checkpoint
            &&self.verify.profile_is_current()&&self.verify.position()==prefix.len()as u64,
            "resumed encoder checkpoint frontier");
        self.prune_invalid_bank_images();
        let media=media.through(prefix.len());
        if self.cached_sessions.iter().any(|c|c.owner==self.bank_id&&c.prefix==prefix
            &&c.media.matches_prefix(&media,prefix.len())) {
            return Ok(json!({"saved":false,"already_cached":true,"bank_retained":true}));
        }
        self.large.handoff_from(&self.verify)?;
        let device=self.large.encoder_checkpoint_image_bytes(prefix.len())?;
        let host=if self.history_lookup_enabled {
            crate::dspark_generation::history_lookup_reserved_bytes(self.large.context_capacity())?
        }else{0};
        let charge=device.checked_add(host).and_then(|v|v.checked_add(prefix.len()*4+media.metadata_bytes()))
            .and_then(|v|v.checked_add(crate::target::EncoderPrefixImage::binding_bytes()))
            .and_then(|v|v.checked_add(std::mem::size_of::<crate::dspark_generation::ProposalPolicyImage>()))
            .ok_or_else(||anyhow::anyhow!("resumed encoder cache byte overflow"))?;
        let entries=self.cached_sessions.iter().map(|c|(c.prefix.len(),c.charge)).collect::<Vec<_>>();
        let optional=self.cached_sessions.iter().map(CachedSession::opportunistic_logits).collect::<Vec<_>>();
        let(budget,slots)=self.cache_capacity(0,0);
        let evictions=encoder_checkpoint_evictions(&entries,&optional,(prefix.len(),charge),None,
            budget,self.session_cache_long_first,slots);
        let eligible=device<=2*1024usize.pow(3)&&evictions.is_some();
        let reserve=self.cache_reserve()?;
        // Retain the ordinary encoder checkpoint's preparation allowance.
        let required=reserve+charge+host;
        let admission=self.admit_cache_save(eligible,evictions,prefix,&media,required,slots,&[])?;
        let at=Instant::now();let mut saved=false;
        if admission.admitted {
            let history=self.session.as_ref().unwrap().checkpoint_history_lookup(prefix.len()as u64,&[])?;
            ensure!(history.as_ref().map_or(0,crate::history_lookup::HistoryLookup::workspace_bytes)<=host,
                "resumed encoder history exceeds its reservation");
            let policy=self.session.as_ref().unwrap().proposal_policy_image()?;
            let control=&mut self.control;
            if let Some(image)=self.large.try_save_committed_bank_encoder(device,reserve,&self.comm,&self.stream,
                &mut |fits|control.all_admit(fits))? {
                ensure!(image.bytes()==device&&image.position()==prefix.len(),"resumed encoder image extent");
                self.cached_sessions.push(CachedSession{owner:self.bank_id,prefix:prefix.to_vec(),media,charge,
                    prompt_logits:None,image:CachedState::Encoder{image,history,policy}});
                saved=true;
            }
        }
        self.verify.handoff_from(&self.large)?;
        Ok(json!({"saved":saved,"bank_retained":true,"encoder_only":true,"resumed_frontier":true,
            "device_bytes":device,"image_charge_bytes":if saved{charge}else{0},
            "charged_bytes":self.cached_sessions.iter().map(|c|c.charge).sum::<usize>(),
            "saved_seconds":at.elapsed().as_secs_f64(),"late_memory_refused":admission.admitted&&!saved,
            "capacity_refused":!eligible,"memory_refused":eligible&&!admission.admitted,
            "evicted_sessions":admission.evicted,"pressure_evicted_sessions":admission.pressure_evicted,
            "forecast_refused":admission.forecast_refused,"admission_available_bytes":admission.available,
            "admission_required_bytes":required}))
    }

    fn encoder_cache_target(&self)->Option<&Target> {
        if self.verify.profile_is_current(){Some(&self.verify)}
        else if self.large.profile_is_current(){Some(&self.large)}else{None}
    }
    pub(super) fn cached_bank_image_valid(&self,cached:&CachedSession)->bool {
        let CachedState::Encoder{image,..}=&cached.image else{return true;};
        if !image.bank_retained(){return true;}
        let target=if cached.owner==self.bank_id{self.encoder_cache_target()}
            else{self.peer_bank.as_ref().filter(|p|p.id==cached.owner).and_then(Bank::encoder_cache_target)};
        target.is_some_and(|t|t.encoder_prefix_reusable_from(image))
    }
    /// A local sparse restore changes lineage but cannot alter the omitted
    /// pages of an earlier, currently valid image. Capture only those exact
    /// ancestors before restoring; a full image or foreign import can replace
    /// the prefix and must not inherit the destination bank's old bindings.
    pub(super) fn bank_ancestors_before_restore(&self,cached:&CachedSession)
        ->Result<Vec<crate::target::BankEncoderWitness>> {
        if cached.owner!=self.bank_id {return Ok(Vec::new());}
        let CachedState::Encoder{image:restored,..}=&cached.image else{return Ok(Vec::new());};
        if !restored.bank_retained(){return Ok(Vec::new());}
        let target=self.encoder_cache_target().context("sparse restore needs its committed bank")?;
        ensure!(target.encoder_prefix_reusable_from(restored),"sparse restore prefix is no longer resident");
        Ok(self.cached_sessions.iter().filter(|c|c.owner==self.bank_id).filter_map(|c|{
            let CachedState::Encoder{image,..}=&c.image else{return None;};
            if image.position()>restored.position(){return None;}
            target.bank_encoder_witness(image)
        }).collect())
    }
    /// Invalidate only tails whose omitted prefix has been reset or rewound.
    /// Complete snapshots retain their existing independent lifetime.
    pub(super) fn prune_invalid_bank_images(&mut self) {
        if !crate::config::bank_encoder_checkpoint_requested(){return;}
        let invalid=self.cached_sessions.iter().enumerate().filter_map(|(i,c)|
            (!self.cached_bank_image_valid(c)).then_some(i)).collect::<Vec<_>>();
        for i in invalid.into_iter().rev(){self.cached_sessions.remove(i);}
    }
    pub(super) fn can_import_cached_encoder(&self,cached:&CachedSession)->bool {
        crate::config::encoder_prefix_import_requested()&&cached.owner!=self.bank_id
            &&self.peer_bank.as_ref().is_some_and(|p|p.id==cached.owner
                &&p.encoder_import_source(cached).is_ok())
            &&matches!(&cached.image,CachedState::Encoder{..})
    }
    /// Build before startup warmup. Construction uses the same complete width
    /// banks and memory-saving layouts as the finite full-profile qualification.
    pub(super) fn initialize_resident_banks(&mut self,kernels:&Path)->Result<Value> {
        let enabled=enabled()?;
        let paired=match std::env::var("CANADA_SCRATCH_PAIRED_SERVING") {
            Ok(v)=>{ensure!(v=="0"||v=="1","paired serving flag");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
        };
        ensure!(!paired||enabled,"paired serving requires independent resident owners");
        let import=match std::env::var("CANADA_SCRATCH_ENCODER_PREFIX_IMPORT") {
            Ok(v)=>{ensure!(v=="0"||v=="1","encoder import flag");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
        };
        ensure!(!import||enabled||std::env::var("CANADA_SCRATCH_ENCODER_PREFIX_IMPORT_SCREEN").as_deref()==Ok("1"),
            "encoder prefix import requires physical owners or its finite screen");
        let bank_checkpoint=match std::env::var("CANADA_SCRATCH_BANK_ENCODER_CHECKPOINT") {
            Ok(v)=>{ensure!(v=="0"||v=="1","bank encoder checkpoint flag");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
        };
        ensure!(!bank_checkpoint||(enabled&&import&&self.encoder_prompt_checkpoint
            &&self.retain_restored_checkpoint
            &&std::env::var("CANADA_SCRATCH_STATE_EXCHANGE").as_deref()!=Ok("1")),
            "bank checkpoints require retained independent resident owners and explicit import support");
        let head=match std::env::var("CANADA_SCRATCH_PAIRED_DRAFT_HEAD") {
            Ok(v)=>{ensure!(v=="0"||v=="1","paired draft head flag");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
        };
        ensure!(!head||paired||std::env::var("CANADA_SCRATCH_PAIRED_GENERATION_SCREEN").as_deref()==Ok("1"),
            "paired draft head requires paired serving or finite generation");
        let shared_head=match std::env::var("CANADA_SCRATCH_SHARED_DRAFT_HEAD") {
            Ok(v)=>{ensure!(v=="0"||v=="1","shared draft head flag");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
        };
        ensure!(!shared_head||enabled||std::env::var("CANADA_SCRATCH_PAIRED_GENERATION_SCREEN").as_deref()==Ok("1"),
            "shared draft encoding requires independent query owners");
        self.control.agree(&[0x68,u8::from(enabled),u8::from(import),u8::from(head),u8::from(shared_head),u8::from(bank_checkpoint)])?;
        if !enabled{return Ok(json!({"enabled":false}));}
        ensure!(!import||(self.encoder_prompt_checkpoint&&self.retain_restored_checkpoint),
            "encoder imports require retained immutable parents");
        ensure!(self.peer_bank.is_none()&&self.bank_id==0&&self.current.is_none()
            &&self.history.frontier()==0&&self.session.is_none()
            &&self.large.context_capacity()==crate::config::MODEL_CONTEXT
            &&self.long_verify.is_some()&&self.copy32_verify.is_some()
            &&std::env::var("CANADA_SCRATCH_TWO_STREAM_ADMISSION").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_HTTP_INTERLEAVE").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_STATE_EXCHANGE").as_deref()!=Ok("1"),
            "physical owners require fresh full-capacity two-slot serving without image exchange");
        for key in ["CANADA_SCRATCH_NARROW_ARENA_LOAN_SCREEN",
            "CANADA_SCRATCH_PEER_WIDE_MOE_SHARE_SCREEN","CANADA_SCRATCH_WIDE_IO_ARENAS_SCREEN",
            "CANADA_SCRATCH_SHARED_MATRIX_ENCODING_SCREEN","CANADA_SCRATCH_ALLOCATION_FLOOR_SCREEN",
            "CANADA_SCRATCH_SHARED_BLAS_WORKSPACE_SCREEN","CANADA_SCRATCH_CORE_MODULE_CACHE_SCREEN",
            "CANADA_SCRATCH_WIDE_INDEX_KEYS_SCREEN","CANADA_SCRATCH_NARROW_MOE_CONSTRUCTION",
            "CANADA_SCRATCH_COMPACT_CANDIDATE_CONSTRUCTION_SCREEN","CANADA_SCRATCH_SHARED_WIDE_ARENAS"] {
            ensure!(std::env::var(key).as_deref()==Ok("1"),"missing qualified physical-owner layout {key}");
        }
        let began=Instant::now();let before=self.cuda.admission_memory()?;
        let module=self.cuda.module(&self.core)?;
        let donor=self.wides.iter().max_by_key(|t|t.physical_rows()).context("complete prefill donor bank")?;
        let large=self.large.independent_owner_for_probe(&module,&self.core,&self.stream,kernels,100,Some(donor))?;
        let fork=|rows|large.fork_profile_lending_prefill_for_probe(&module,&self.core,&self.stream,kernels,
            rows,Phase::Verify,donor);
        let mut verify=fork(6)?;verify.set_graph_replay(true)?;verify.apply_candidate_schedule(&self.stream)?;
        let wides=large.fork_peer_wide_bank_for_probe(&module,&self.core,&self.stream,kernels,&self.large,&self.wides)?;
        let mut prepared=self.prepared_session.as_ref().context("primary prepared draft")?
            .independent_for_probe(&large,&self.core,&self.stream,kernels)?;
        ensure!(Target::compact_history_sampling_requested()?,"physical owners retain compact history sampling");
        let mut long=fork(16)?;long.set_graph_replay(true)?;
        long.enable_compact_history_sampling(&self.core,&self.stream)?;
        let long_buffers=LongHistoryBuffers::new(&self.cuda,&self.core)?;
        let copy_before=self.cuda.admission_memory()?;
        let required=9*1024usize.pow(3)+768*1024usize.pow(2);
        ensure!(self.control.all_admit(copy_before.admitted_available>=required)?,
            "second copy-profile construction would consume physical reserve");
        let mut copy=fork(32)?;copy.set_graph_replay(true)?;
        copy.enable_compact_history_sampling(&self.core,&self.stream)?;
        let copy_buffers=LongHistoryBuffers::with_rows(&self.cuda,&self.core,32)?;
        ensure!(prepared.prepare_copy_conditioning(&copy_buffers,&self.comm,&self.stream)?,
            "second physical owner requires complete copy conditioning");
        self.stream.synchronize()?;
        let after=self.cuda.admission_memory()?;
        ensure!(self.control.all_admit(after.admitted_available>=9*1024usize.pow(3))?,
            "complete resident owners consumed construction reserve");
        let ownership=self.large.independent_state_report(&large)?;
        ensure!(ownership["independent_causal_owners"]==true,"physical owners share causal state");
        self.peer_bank=Some(Bank{id:1,session:None,prepared_session:Some(prepared),
            session_prepare_seconds:began.elapsed().as_secs_f64(),target_lineage:100,
            verify,long_verify:Some(long),long_history_buffers:Some(long_buffers),
            copy32_verify:Some(copy),copy32_buffers:Some(copy_buffers),
            copy32_admission:json!({"requested":true,"admitted":true,"compact_history_sampling":true,
                "wide_copy_conditioning_prepared":true,"construction_headroom_bytes":required,
                "memory_before":copy_before,"memory_after":after}),wides,large,
            history:ResidentHistory::new(self.large.context_capacity(),self.cfg.eos_token_id)?,
            current:None,pending_prefill:None,prefill_cancelled:false,active_media:None,
            media_receipt:Value::Null,emitted:0,limit:0});
        self.control.agree(&[0x69,u8::from(paired)])?;
        if paired {
            let pool=crate::ffn::Paired::new(&self.cuda,&self.core,kernels)?;
            self.resident_pair=Some(crate::target::PairExecutor::new(&self.verify,
                &self.peer_bank.as_ref().unwrap().verify,pool,&self.comm,&self.stream)?);
        }
        let final_memory=self.cuda.admission_memory()?;
        let shared_target_head=self.verify.shares_lossless_target_head(&self.peer_bank.as_ref().unwrap().verify);
        ensure!(std::env::var("CANADA_SCRATCH_SHARED_TARGET_HEAD").as_deref()!=Ok("1")||shared_target_head,
            "resident target head encoding was not shared");
        ensure!(self.control.all_admit(final_memory.admitted_available>=9*1024usize.pow(3))?,
            "paired workspace consumed construction reserve");
        Ok(json!({"enabled":true,"owners":2,"context_capacity_each":self.large.context_capacity(),
            "memory_before":before,"memory_after":after,"seconds":began.elapsed().as_secs_f64(),
            "ownership":ownership,"shared_optional_cache_budget":self.shared_image_budget,
            "paired_verification":paired,"paired_draft_head":head,"shared_draft_head_encoding":shared_head,
            "encoder_prefix_import":import,"shared_target_head_encoding":shared_target_head,"memory_after_pair":final_memory}))
    }

    /// No GPU state copy, allocation or RNG draw. Both owners are stable at
    /// this boundary; pending prefill is an already committed encoder chunk.
    pub(super) fn switch_resident_bank(&mut self)->Result<()> {
        self.stream.synchronize()?;
        let peer=self.peer_bank.as_mut().context("physical peer missing")?;
        ensure!(self.bank_id!=peer.id&&self.bank_id<2&&peer.id<2&&peer.committed()
            &&if self.session.is_some(){self.verify.profile_is_current()
                &&self.verify.position()==self.history.frontier()as u64}
              else{self.current.is_none()&&self.history.frontier()==0&&self.large.position()==0},
            "physical owner switch requires two committed independent banks");
        self.history.synchronize_request_ids(&mut peer.history)?;
        std::mem::swap(&mut self.bank_id,&mut peer.id);
        // Keep this list explicit: mutable request state must never be lent to
        // the other owner. Global communicator, policy and caches stay in Worker.
        macro_rules! swap_fields {($($field:ident),+)=>{$(std::mem::swap(&mut self.$field,&mut peer.$field);)+};}
        swap_fields!(session,prepared_session,session_prepare_seconds,target_lineage,
            verify,long_verify,long_history_buffers,copy32_verify,copy32_buffers,copy32_admission,
            wides,large,history,current,pending_prefill,prefill_cancelled,active_media,media_receipt,emitted,limit);
        let peer_id=peer.id;let peer_request=peer.current;
        self.loaded_request_is_suspended=false;
        let mut proof=self.agreement_state(&[]);proof.extend_from_slice(b"physical-owner-switch-v1");
        proof.extend_from_slice(&[self.bank_id,peer_id]);
        proof.extend_from_slice(&peer_request.unwrap_or(0).to_le_bytes());
        self.control.agree(&proof)?;Ok(())
    }

    #[cfg(feature = "serving")]
    pub(super) fn warm_resident_before_ready(&mut self,rank:usize)->Result<Value> {
        let first=self.warm_before_ready(rank)?;
        if self.peer_bank.is_none(){return Ok(first);}
        self.switch_resident_bank()?;
        let second=self.warm_before_ready(rank)?;
        self.switch_resident_bank()?;
        Ok(json!({"owners":[first,second],"active_bank":self.bank_id}))
    }

    /// Prefer a free physical owner with useful resident/cached state. A busy
    /// owner is never replaced to obtain a cache hit. Cross-owner import is a
    /// separate operation; ordinary foreign-image restore remains forbidden.
    pub(super) fn select_resident_bank(&mut self,command:&Command)->Result<()> {
        let Command::Begin{tokens,images,..}=command else{return Ok(());};
        self.prune_invalid_bank_images();
        let Some(peer)=self.peer_bank.as_ref()else{return Ok(());};
        ensure!(self.current.is_none(),"new physical request would replace its owner");
        if peer.current.is_some(){return Ok(());}
        let media=self.prompt_identity(images,tokens)?;
        let score=|id:u8,history:&ResidentHistory| {
            let live=history.frontier();
            let live=if live>0&&tokens.len()>live&&tokens.starts_with(history.committed())
                &&media.matches_prefix(history.media(),live){live}else{0};
            self.cached_sessions.iter().filter(|c|c.owner==id&&tokens.starts_with(&c.prefix)
                &&media.matches_prefix(&c.media,c.prefix.len()))
                .map(|c|c.prefix.len()).max().unwrap_or(0).max(live)
        };
        if score(peer.id,&peer.history)>score(self.bank_id,&self.history){self.switch_resident_bank()?;}
        Ok(())
    }
}
