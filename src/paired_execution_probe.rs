//! Finite genuinely new paired verification, distinct from replaying two
//! already-computed forwards. Ordinary solo execution remains unchanged.
use super::*;
use std::collections::BTreeMap;

/// Drop captures before their private pooled workspace. The finite caller must
/// keep both target profiles alive until this executor has been dropped.
pub(crate) struct PairExecutor {
    graphs:BTreeMap<(usize,usize,u16,u16),Graph>,
    head:Option<crate::model_io::PairedHead>,
    pooled:crate::ffn::Paired,
    owners:[crate::session_lease::Lease;2],
    storage:[Vec<u64>;2],
    stream:usize,
    comm_instance:u64,
    rank:usize,
}

struct PreparedRows {
    pending:PendingVerification,
    marks:Vec<(&'static str,Instant)>,
}

impl Target {
    pub(crate) fn pair_proposal_owner(&self)->Result<crate::session_lease::Lease> {
        ensure!(self.lease.is_current()&&self.pending.is_none(),"pair proposal needs committed target authority");
        Ok(self.lease.fork())
    }
    pub(crate) fn pair_proposal_matches(&self,owner:&crate::session_lease::Lease)->bool {
        self.lease.shares_owner(owner)
    }
    fn pair_execution_storage(&self)->Vec<u64> {
        let mut ids=self.causal_storage_ids();
        ids.extend([self.io.local_logits().ptr(),self.late.output().ptr(),self.late.pre().ptr()]);
        ids
    }
    fn pair_execution_key(&self)->u16 {
        let mut key=u16::from(self.adaptive_sparse_active);
        for (i,q)in self.index_queries().into_iter().enumerate() {
            if q.is_some_and(|q|q.key_sharded_active()){key|=1<<(i+1);}
        }
        key
    }
    fn prepare_pair_rows(&mut self,ids:&[u32],comm:&Roce,stream:&Stream)->Result<PreparedRows> {
        ensure!((std::env::var("CANADA_SCRATCH_PAIRED_ACTUAL_SCREEN").as_deref()==Ok("1")
            ||crate::config::paired_serving_requested())
            &&self.rows==6&&self.source_phase==Phase::Verify&&self.graph_replay
            &&self.wide.is_none()&&self.cached_decoder.is_none()&&!self.prefill_last_head
            &&Self::independent_probe_context_admitted(self.context)&&self.pending.is_none()&&self.lease.is_current()
            &&[3,6].contains(&ids.len())&&ids.iter().all(|&id|id<129280)
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize
            &&self.rank==comm.rank()&&self.comm_instance==Some(comm.instance()),
            "finite fresh paired verification profile/owner/stream");
        self.last_live=0;self.last_append_metrics=None;self.carry_saved_start=None;
        let mut marks=vec![("begin",Instant::now())];
        self.select_index_graph_bank(stream)?;
        let plan=Plan::append_with_layout(self.history.position,ids.len(),self.rows,self.context,self.swa_layout)?;
        let mut padded=vec![2u32;self.rows];padded[..ids.len()].copy_from_slice(ids);
        let batch=self.history.prepare_padded(&padded,ids.len(),&self.hashes)?;
        marks.push(("history_hash",Instant::now()));
        // First implementation prepares both immutable Engram row buffers
        // eagerly. Overlap can be restored after this ownership path is proven.
        self.prepare_engram_rows(&batch.rows)?;
        marks.push(("engram_rows",Instant::now()));
        self.io.update_tokens(&mut self.tokens,self.history.position as usize,
            &padded.iter().copied().map(i64::from).collect::<Vec<_>>(),ids.len())?;
        marks.push(("token_upload",Instant::now()));
        self.step.update(&plan,stream)?;
        marks.push(("causal_metadata",Instant::now()));
        comm.select_decode_reduction(true,stream)?;
        comm.select_prefill_rows(ids.len(),self.rows,stream)?;
        stream.join_default()?;
        self.prepare_gather_caches(stream)?;
        self.lease.begin()?;
        if self.carry_snapshot.is_none(){self.carry_snapshot=Some(self.new_carry_snapshot()?);}
        unsafe{self.carry_snapshot.as_mut().unwrap().save(stream)?;}
        self.carry_saved_start=Some(self.history.position);
        marks.push(("carry_snapshot",Instant::now()));
        Ok(PreparedRows{pending:PendingVerification{hash_batch:batch,start:self.history.position,live:ids.len()},marks})
    }
    fn complete_pair_rows(&mut self,mut prepared:PreparedRows,captured:bool,
        enqueue:Instant,completed:Instant)->Result<()> {
        ensure!(self.pending.is_none()&&self.lease.is_busy_revision()
            &&self.history.position==prepared.pending.start,"pair completion lost ownership");
        self.check_candidates()?;
        if let Some(cache)=&self.encoder_cache {ensure!(cache.error()?==0,"paired encoder publication fault");}
        prepared.marks.push(("forward_enqueue",enqueue));
        prepared.marks.push(("gpu_completion",completed));
        prepared.marks.push(("candidate_checks",Instant::now()));
        self.last_head_local_only=self.compact_sampling_active();
        self.last_append_metrics=Some(AppendMetrics{
            live_rows:prepared.pending.live,physical_rows:self.rows,phase:format!("{:?}",self.source_phase),
            captured_new_graph:captured,engram_split14:false,engram_split14_adaptive:false,
            encoder_graph_chunks:0,lookahead:Vec::new(),stall_counters:Vec::new(),native_fp32_rows:None,
            total_ms:prepared.marks.last().unwrap().1.duration_since(prepared.marks[0].1).as_secs_f64()*1000.,
            stages_ms:prepared.marks.windows(2).map(|p|(p[1].0,p[1].1.duration_since(p[0].1).as_secs_f64()*1000.)).collect(),
        });
        self.pending=Some(prepared.pending);Ok(())
    }
    pub(crate) fn paired_completed_taps(&self)->Result<Value> {
        ensure!(std::env::var("CANADA_SCRATCH_PAIRED_ACTUAL_SCREEN").as_deref()==Ok("1"),
            "finite paired result readout");
        self.pair_model_taps(false)
    }
    pub(crate) fn paired_completed_endpoint_taps(&self,stream:&Stream)->Result<Value> {
        ensure!(std::env::var("CANADA_SCRATCH_PAIRED_ENDPOINT_MATCHED_SCREEN").as_deref()==Ok("1"),
            "matched endpoint state readout is diagnostic only");
        stream.synchronize()?;
        let mut value=self.pair_model_taps(true)?;
        let (features,live)=self.dspark_feature_inputs(stream)?;
        value["draft_features"]=serde_json::to_value(features.into_iter().map(|f|->Result<String>{
            Ok(format!("{:x}",Sha256::digest(unsafe{self.cuda.read_device(f.ptr(),f.bytes()/self.rows*live)}?)))
        }).collect::<Result<Vec<_>>>()?)?;
        Ok(value)
    }
}

impl PairExecutor {
    /// Finite same-allocation replay at an already checked pending frontier.
    /// Carry restores are identical in every arm. End with the same causal
    /// state/logits and leave acceptance entirely to the original caller.
    pub(crate) fn dense_timing(&self,a:&mut Target,b:&mut Target,comm:&Roce,stream:&Stream)->Result<Value> {
        ensure!(std::env::var("CANADA_SCRATCH_PAIRED_DENSE_TIMING_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_PAIRED_ACTUAL_SCREEN").as_deref()==Ok("1")
            &&self.head.is_some()&&self.pooled.pooled_shared_up()&&self.pooled.qkv().is_some()
            &&a.pending.is_some()&&b.pending.is_some()
            &&self.owners[0].shares_owner(&a.lease)&&self.owners[1].shares_owner(&b.lease)
            &&self.stream==stream.raw()as usize&&self.comm_instance==comm.instance(),
            "dense attribution requires the complete admitted pair at a pending frontier");
        stream.synchronize()?;
        let expected=[a.pair_model_taps(true)?,b.pair_model_taps(true)?];
        let brief=[a.pair_model_taps(false)?,b.pair_model_taps(false)?];
        let mut graphs=Vec::new();
        for mask in [0,3,7] {
            let mut enqueue=||->Result<()> {
                a.pair_restore_carry(b,stream)?;
                a.enqueue_pair_features(b,Some(&self.pooled),false,self.head.as_ref(),mask,comm,stream)
            };
            enqueue()?;stream.synchronize()?;comm.health()?;
            graphs.push(stream.capture(enqueue)?);
            ensure!([a.pair_model_taps(false)?,b.pair_model_taps(false)?]==brief,
                "paired dense mask{mask} changed logits or compressor carry");
        }
        let mut records=Vec::new();
        for repeat in 0..4 {
            let order=if repeat%2==0{[0,1,2]}else{[2,1,0]};let mut milliseconds=[0f32;3];
            for mode in order {
                unsafe{graphs[mode].launch(stream)?;}stream.synchronize()?;
                let begin=a.cuda.event()?;let end=a.cuda.event()?;begin.record(stream)?;
                for _ in 0..2{unsafe{graphs[mode].launch(stream)?;}}
                end.record(stream)?;milliseconds[mode]=end.elapsed_since(&begin)?/2.;comm.health()?;
                ensure!([a.pair_model_taps(false)?,b.pair_model_taps(false)?]==brief,
                    "paired dense replay changed output");
            }
            records.push(json!({"repeat":repeat,"order":order,"milliseconds":milliseconds}));
        }
        ensure!([a.pair_model_taps(true)?,b.pair_model_taps(true)?]==expected,
            "paired dense attribution changed retained state");
        Ok(json!({"all_exact":true,"masks":[0,3,7],
            "modes":["routed_pool","plus_shared_up_and_head","plus_qkv"],"records":records,
            "scope":"Same weight addresses, pending prefixes, private state, incoming carry restore and warm graphs. Alternated order; excludes drafting, acceptance, Engram input reads, metadata preparation and capture. Not serving throughput."}))
    }
    pub(crate) fn new(a:&Target,b:&Target,pooled:crate::ffn::Paired,comm:&Roce,stream:&Stream)->Result<Self> {
        ensure!((std::env::var("CANADA_SCRATCH_PAIRED_ACTUAL_SCREEN").as_deref()==Ok("1")
            ||crate::config::paired_serving_requested())
            &&a.rows==6&&b.rows==6&&!a.lease.shares_owner(&b.lease)
            &&a.stream==b.stream&&a.stream==stream.raw()as usize&&a.cuda.owns_stream(stream)
            &&a.rank==comm.rank()&&b.rank==comm.rank(),"finite independent pair executor");
        let head=match std::env::var("CANADA_SCRATCH_PAIRED_TARGET_HEAD") {
            Ok(value)=>{ensure!(value=="0"||value=="1","paired target head flag");
                if value=="1"{Some(crate::model_io::PairedHead::new(&a.io,&b.io,stream)?)}else{None}},
            Err(std::env::VarError::NotPresent)=>None,Err(error)=>return Err(error.into()),
        };
        Ok(Self{graphs:BTreeMap::new(),head,pooled,owners:[a.lease.fork(),b.lease.fork()],
            storage:[a.pair_execution_storage(),b.pair_execution_storage()],
            stream:stream.raw()as usize,comm_instance:comm.instance(),rank:comm.rank()})
    }
    pub(crate) fn begin(&mut self,a:&mut Target,a_ids:&[u32],b:&mut Target,b_ids:&[u32],
        comm:&Roce,stream:&Stream)->Result<Value> {
        self.begin_with_capture_admission(a,a_ids,b,b_ids,comm,stream,&mut||Ok(true))
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_with_capture_admission(&mut self,a:&mut Target,a_ids:&[u32],b:&mut Target,b_ids:&[u32],
        comm:&Roce,stream:&Stream,admit_capture:&mut impl FnMut()->Result<bool>)->Result<Value> {
        ensure!(self.stream==stream.raw()as usize&&self.comm_instance==comm.instance()&&self.rank==comm.rank()
            &&self.owners[0].shares_owner(&a.lease)&&self.owners[1].shares_owner(&b.lease)
            &&self.storage==[a.pair_execution_storage(),b.pair_execution_storage()],
            "paired graph cannot adopt different profiles, storage or transport");
        let began=Instant::now();
        let first=a.prepare_pair_rows(a_ids,comm,stream)?;
        let second=b.prepare_pair_rows(b_ids,comm,stream)?;
        let prepared=Instant::now();
        let key=(a_ids.len(),b_ids.len(),a.pair_execution_key(),b.pair_execution_key());
        // An optional graph must never consume the request's physical reserve.
        // Declining a new capture retains the same eager target computation.
        let missing=!self.graphs.contains_key(&key);
        let captured=missing&&admit_capture()?;
        if let Some(graph)=self.graphs.get(&key) {
            unsafe{graph.launch(stream)?;}
        }else{
            // Retain this first eager result. Recording the future graph does
            // not execute it, so no extra target pass or carry restoration.
            a.enqueue_pair(b,Some(&self.pooled),false,self.head.as_ref(),comm,stream)?;
            if captured {
                stream.synchronize()?;comm.health()?;
                let graph=stream.capture(||a.enqueue_pair(b,Some(&self.pooled),false,self.head.as_ref(),comm,stream))?;
                self.graphs.insert(key,graph);
            }
        }
        let enqueued=Instant::now();
        let poll=a.verification_poll_us.max(b.verification_poll_us);
        if poll>0{stream.synchronize_polling(poll)?;}else{stream.synchronize()?;}
        comm.health()?;let completed=Instant::now();
        a.complete_pair_rows(first,captured,enqueued,completed)?;
        b.complete_pair_rows(second,captured,enqueued,completed)?;
        Ok(json!({"live_rows":[a_ids.len(),b_ids.len()],"captured":captured,
            "pooled_target_head":self.head.is_some()&&matches!(a_ids.len(),3|6)&&matches!(b_ids.len(),3|6),
            "pooled_shared_up":self.pooled.pooled_shared_up(),
            "pooled_qkv_layers":if self.pooled.qkv().is_some(){37}else{0},
            "uncached_eager":missing&&!captured,"graph_count":self.graphs.len(),
            "prepare_ms":prepared.duration_since(began).as_secs_f64()*1000.,
            "forward_ms":completed.duration_since(prepared).as_secs_f64()*1000.,
            "total_ms":began.elapsed().as_secs_f64()*1000.}))
    }
}
