//! Finite GPU ownership checks after optional save refusal or prefill cancellation.
//! Selected only by the file runner in a fault-injection build; never HTTP.
use super::*;
use cache_admission_fault::Point;

fn execute(w:&mut Worker,c:Command)->Result<Value> {
    let encoded=c.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(encoded.as_slice()))?;
    ensure!(wire==encoded,"retention probe commands disagree");
    w.execute(Command::decode(&wire)?)
}
fn begin(w:&mut Worker,id:&mut u64,prompt:Vec<u32>)->Result<Value> {
    *id+=1;
    let event=execute(w,Command::Begin{images:Vec::new(),id:*id,seed:20261029,greedy:false,limit:1,tokens:prompt})?;
    ensure!(event["target_temperature"]==1.0&&event["target_top_p"]==1.0,
        "cache retention screen must preserve target T1/P1");
    Ok(event)
}
fn finish(w:&mut Worker)->Result<()> {
    execute(w,Command::Finish(Finish::Length))?;Ok(())
}
fn state(w:&mut Worker)->Result<Value> {
    ensure!(w.verify.position()as usize==w.history.frontier(),"retention probe frontier mismatch");
    Ok(json!({"frontier":w.verify.position(),"carry":w.verify.carry_digests()?,
        "encoder":w.verify.encoder_tail_digest()?,
        "draft":w.session.as_mut().unwrap().conditioning_state_digest_for_probe(&w.comm,&w.stream)?,
        "history":w.session.as_ref().unwrap().history_lookup_digest_for_probe()?}))
}
fn following(w:&mut Worker,ids:&[u32])->Result<(Value,Vec<u8>)> {
    following_trace(w,ids,None)
}
fn following_trace(w:&mut Worker,ids:&[u32],trace:Option<&Path>)->Result<(Value,Vec<u8>)> {
    let before=state(w)?;
    w.verify.begin_verification(ids,&w.comm,&w.stream)?;
    if std::env::var("CANADA_SCRATCH_CACHE_CHUNK_TRACE").as_deref()==Ok("1") {
        if let Some(path)=trace {w.verify.dump_chunk_plan_trace(path)?;}
    }
    let(y,n)=w.verify.pending_logits()?;ensure!(n==6,"retention probe following row count");
    let raw=unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?;
    w.verify.finish_verification(0,&w.comm,&w.stream)?;
    ensure!(state(w)?==before,"read-only verification rollback changed committed state");
    Ok((json!({"state":before,"following_logits_sha256":format!("{:x}",Sha256::digest(&raw))}),raw))
}
fn parent(w:&Worker,prefix:&[u32])->Result<Value> {
    let matches=w.cached_sessions.iter().filter(|c|c.prefix==prefix).collect::<Vec<_>>();
    ensure!(matches.len()==1,"borrowed parent absent or duplicated");
    let c=matches[0];ensure!(c.media.placements().is_empty()&&c.prompt_logits.is_none(),"public encoder parent identity");
    let CachedState::Encoder{image,history,policy}=&c.image else {anyhow::bail!("expected retained encoder parent")};
    Ok(json!({"prefix_tokens":c.prefix.len(),"charge":c.charge,"image":image.retention_digest()?,
        "history":history.as_ref().map(crate::history_lookup::HistoryLookup::state_digest),
        "policy":policy.retention_metadata()}))
}
fn setup(w:&mut Worker,id:&mut u64,tokens:&[u32])->Result<()> {
    w.cached_sessions.clear();w.encoder_prompt_checkpoint=true;w.prompt_checkpoint=true;
    w.encoder_active_frontier=false;w.encoder_upgrade_reconstructed=false;w.prompt_cache_logits=false;
    if let Some(session)=&mut w.session {session.configure_greedy(Some(w.cfg.eos_token_id))?;}
    let event=begin(w,id,tokens[..8892].to_vec())?;
    ensure!(event["cached_tokens"]==0&&event["prompt_checkpoint"]["saved"]==true
        &&event["prompt_checkpoint"]["prefix_tokens"]==8704,"fresh probe parent not saved");
    finish(w)
}
fn budget(w:&Worker)->Result<()> {
    ensure!(w.cached_sessions.len()+w.suspended_image_count<=4
        &&w.cached_sessions.iter().map(|c|c.charge).sum::<usize>()+w.suspended_image_bytes<=w.shared_image_budget,
        "retention probe exceeded shared image limits");Ok(())
}
fn compare(out:&Path,label:&str,a:&(Value,Vec<u8>),b:&(Value,Vec<u8>))->Result<bool> {
    let exact=a.0==b.0;
    if !exact {
        // Keep full FP32 rows for GPU difference/margin analysis. Hashes alone
        // cannot distinguish harmless rounding from a meaningful divergence.
        std::fs::write(out.join(format!("{label}-control-logits.f32")),&a.1)?;
        std::fs::write(out.join(format!("{label}-candidate-logits.f32")),&b.1)?;
    }
    Ok(exact)
}
fn save(out:&Path,records:&[Value])->Result<()> {
    std::fs::write(out.join("cache-retention-partial.json"),serde_json::to_vec_pretty(records)?)?;Ok(())
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_CACHE_RETENTION_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_ENCODER_CACHE_SCREEN").as_deref()==Ok("1")
        &&w.encoder_prompt_checkpoint&&w.prompt_checkpoint&&w.retain_restored_checkpoint
        &&w.history_lookup_enabled&&w.current.is_none()&&w.suspended_image_count==0,
        "finite cache retention probe admission");
    let raw=include_bytes!("../tests/fixtures/target-agentic-repository-v1.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture.get("tokens").unwrap_or(&fixture["batches"][0]).clone())?;
    ensure!(tokens.len()>=27000,"retention fixture extent");
    let common=&tokens[..8704];let following_ids=&tokens[26000..26006];
    let mut id=7_750_000;let mut records=Vec::new();let mut comparisons=Vec::new();
    let cancel_only=std::env::var("CANADA_SCRATCH_CACHE_RETENTION_CANCEL_ONLY").as_deref()==Ok("1");
    w.control.agree(&[u8::from(cancel_only)])?;
    if !cancel_only {
    for(point_index,point)in [Point::EncoderBeforeAllocate,Point::EncoderAfterAllocate,Point::ReconstructedAfterCopy].into_iter().enumerate() {
        for repeat in 0..2u8 {
            // Rotate head/follower refusal while keeping the exact same wire
            // sequence on every rank. Any earlier real refusal fails coverage.
            let refusing_rank=if repeat==0{0}else{3};
            let mut control=None;let mut candidate=None;
            for inject in if repeat==0{[false,true]}else{[true,false]} {
                w.control.agree(&[point_index as u8,repeat,u8::from(inject),refusing_rank as u8])?;
                setup(w,&mut id,&tokens)?;let original=parent(w,common)?;
                let short=point==Point::ReconstructedAfterCopy;
                w.encoder_upgrade_reconstructed=short&&inject;
                if inject {w.cache_save_fault.arm(point,refusing_rank)?;}
                let prompt=[common,&tokens[15000..if short{15017}else{15349}]].concat();
                let event=begin(w,&mut id,prompt)?;
                let fault=if inject {
                    let observation=w.cache_save_fault.take_observation()?;
                    ensure!(observation.natural_admission,"real memory refusal obscured the injected case");
                    Some(observation)
                }else{None};
                if short {
                    ensure!(event["retained_session_cache"]["encoder_reconstructed"]==true
                        &&event["retained_session_cache"]["reconstructed_upgrade"]==false,"short reconstruction path absent");
                    if inject {ensure!(event["retained_session_cache"]["upgrade_late_memory_refused"]==true
                        &&event["retained_session_cache"]["upgrade_parent_retained_on_refusal"]==true,"late upgrade refusal lost its parent");}
                }else{
                    ensure!(event["encoder_checkpoint_phases"]["consumed_parent"]==false
                        &&event["retained_session_cache"]["retained_image"]==true
                        &&event["prompt_checkpoint"]["saved"]==!inject,"advanced successor took unexpected save path");
                    if inject {ensure!(event["prompt_checkpoint"]["late_memory_refused"]==true,"late encoder refusal not reported");}
                }
                ensure!(parent(w,common)?==original,"optional save changed the stored parent bytes/metadata");
                let after=following(w,following_ids)?;finish(w)?;
                // Return through the exact parent after the failed/successful
                // branch, without upgrading it during this observation.
                w.encoder_upgrade_reconstructed=false;
                let reused=begin(w,&mut id,[common,&tokens[17000..17017]].concat())?;
                ensure!(reused["cached_tokens"]==8704&&reused["retained_session_cache"]["retained_image"]==true,
                    "parent not reusable after optional save");
                ensure!(parent(w,common)?==original,"parent restore mutated its immutable image");
                let resumed=following(w,following_ids)?;finish(w)?;budget(w)?;
                let semantic=json!({"after":after.0,"resumed":resumed.0});
                let observation=(semantic.clone(),[after.1,resumed.1].concat());
                records.push(json!({"point":point,"repeat":repeat,"injected":inject,"fault":fault,
                    "begin":event,"reuse":reused,"parent_immutable":true,"semantic":semantic}));save(out,&records)?;
                if inject{candidate=Some(observation);}else{control=Some(observation);}
            }
            let label=format!("save-{point_index}-{repeat}");
            let exact=compare(out,&label,&control.unwrap(),&candidate.unwrap())?;
            comparisons.push(json!({"case":label,"semantic_exact":exact}));
        }
    }
    }
    // Cancel after the first committed wide chunk of an advanced branch.
    // The all-rank decision uses the same control channel as HTTP cancellation.
    setup(w,&mut id,&tokens)?;let original=parent(w,common)?;
    let prompt=[common,&tokens[15000..27000]].concat();id+=1;
    let command=Command::Begin{images:Vec::new(),id,seed:20261029,greedy:false,limit:1,tokens:prompt.clone()};
    let encoded=command.encode()?;
    let wire=w.control.command((rank==0).then_some(encoded.as_slice()))?;
    ensure!(wire==encoded,"cancellation command replicas disagree");
    let cancelled=w.execute_with_prefill_cancel(Command::decode(&wire)?,&mut |control|{
        let decision=control.command((rank==0).then_some(&[1u8][..]))?;
        ensure!(decision==[1],"cancellation decision mismatch");Ok(true)
    })?;
    ensure!(cancelled["cancelled"]==true&&cancelled["emitted_tokens"]==0
        &&cancelled["cached_tokens"]==8704&&cancelled["retained_session_cache"]["retained_image"]==true,
        "cancellation missed the borrowed-parent path");
    ensure!(parent(w,common)?==original,"cancelled prefill changed its stored parent");
    let frontier=w.verify.position()as usize;ensure!(frontier>8704&&frontier<prompt.len(),"cancellation frontier");
    let partial=following_trace(w,following_ids,Some(&out.join("chunk-partial")))?;
    execute(w,Command::Finish(Finish::Cancelled))?;
    let reused=begin(w,&mut id,[common,&tokens[17000..17017]].concat())?;
    ensure!(reused["cached_tokens"]==8704&&parent(w,common)?==original,"cancelled parent cannot be reused");
    finish(w)?;
    // Independent ordinary-serving control with the same encoder chunk plan:
    // save an advanced encoder checkpoint at the cancelled frontier, finish a
    // real128-token suffix, then reconstruct that checkpoint without cancelling.
    // The fresh-prefix control below uses different chunk boundaries, so its
    // differences alone cannot identify a cancellation regression.
    setup(w,&mut id,&tokens)?;
    let suffix_end=15000+(frontier-common.len())+128;
    ensure!(suffix_end<=tokens.len(),"ordinary cancellation-control fixture extent");
    let normal=begin(w,&mut id,[common,&tokens[15000..suffix_end]].concat())?;
    ensure!(normal["prompt_checkpoint"]["saved"]==true
        &&normal["prompt_checkpoint"]["prefix_tokens"]==frontier,
        "ordinary control did not save the same encoder frontier");
    finish(w)?;
    // An exact encoder-only cache hit is intentionally ineligible without
    // saved prompt logits. Request one real suffix token, and observe the
    // ordinary reconstructed owner immediately before that append.
    id+=1;
    let mut plan=w.history.prepare(id,[&prompt[..frontier],&following_ids[..1]].concat(),1,20261029)?;
    let(reconstructed,deferred)=w.reuse_cached_session(&mut plan,true,true)?;
    records.push(json!({"case":"ordinary-reconstruction-admission","begin":normal,
        "reconstructed":reconstructed}));save(out,&records)?;
    ensure!(deferred.is_none()&&plan.previous==frontier&&plan.reuse
        &&reconstructed["encoder_reconstructed"]==true,
        "ordinary control did not reconstruct the accepted prefix");
    let ordinary=following(w,following_ids)?;
    w.begin_small_append(plan,false,Instant::now(),reconstructed.clone())?;finish(w)?;
    let ordinary_exact=compare(out,"cancelled-versus-ordinary-reconstruction",&ordinary,&partial)?;
    comparisons.push(json!({"case":"cancelled-versus-ordinary-reconstruction","semantic_exact":ordinary_exact}));
    records.push(json!({"case":"ordinary-reconstruction-control","begin":normal,"reconstructed":reconstructed,
        "ordinary":ordinary.0,"partial":partial.0}));save(out,&records)?;
    w.cached_sessions.clear();w.encoder_prompt_checkpoint=false;w.prompt_checkpoint=false;
    // Fresh computation of exactly the accepted prefix is a numerical control,
    // not a demand that sampled tokens across different plans be identical.
    let fresh=begin(w,&mut id,prompt[..frontier].to_vec())?;
    ensure!(fresh["cached_tokens"]==0,"equivalent-prefix control was not fresh");
    let reference=following_trace(w,following_ids,Some(&out.join("chunk-fresh")))?;finish(w)?;
    let exact=compare(out,"cancelled-prefix",&reference,&partial)?;
    comparisons.push(json!({"case":"cancelled-prefix","semantic_exact":exact}));
    records.push(json!({"case":"cancelled-prefix","cancelled":cancelled,"reuse":reused,"fresh":fresh,
        "partial":partial.0,"reference":reference.0,"parent_immutable":true}));save(out,&records)?;
    // A normal live extension reproduces the encoder's8704+8192 chunk plan
    // without restoring any saved image and without cancellation. This
    // separates cache reconstruction from fresh/cached batch-shape changes.
    let split_begin=begin(w,&mut id,common.to_vec())?;
    ensure!(split_begin["cached_tokens"]==0,"split control was not fresh");
    finish(w)?;
    let split_extend=begin(w,&mut id,prompt[..frontier].to_vec())?;
    ensure!(split_extend["cached_tokens"]==common.len(),"split control missed live prefix");
    let split=following_trace(w,following_ids,Some(&out.join("chunk-split")))?;finish(w)?;
    let split_exact=compare(out,"cancelled-versus-live-split",&split,&partial)?;
    comparisons.push(json!({"case":"cancelled-versus-live-split","semantic_exact":split_exact}));
    records.push(json!({"case":"live-split-control","begin":split_begin,"extend":split_extend,
        "split":split.0,"partial":partial.0}));save(out,&records)?;
    budget(w)?;execute(w,Command::Shutdown)?;
    std::fs::write(out.join("cache-retention-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"comparisons":comparisons,"ownership_checks_passed":true,"cancel_only":cancel_only,
        "all_semantic_exact":comparisons.iter().all(|x|x["semantic_exact"]==true),
        "fixture_sha256":format!("{:x}",Sha256::digest(raw)),"following_ids":following_ids,
        "comparison_payload":"Save comparisons contain two consecutive6x129280FP32 logit matrices: after branch, then parent reuse. Cancellation contains one such matrix. Files are emitted only for nonexact comparisons.",
        "scope":"Finite foreground ownership screen, one public8Krepository parent, three forced late-refusal points in two orders, then cancellation and equivalent accepted-prefix recomputation. Target T1/P1; one emitted token per completed request. Immutable parent and same-transaction rollback are hard requirements. Cross-path numerical differences are retained for GPU analysis, not automatically labeled quality failures. No production or completed-task-quality claim."}))?)?;
    Ok(())
}
