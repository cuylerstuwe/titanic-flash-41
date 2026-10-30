//! Compare checkpoint storage with the ordinary unsegmented worker append.
use super::*;
fn execute(w:&mut Worker,c:Command)->Result<Value> {
    let encoded=c.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(encoded.as_slice()))?;
    ensure!(wire==encoded,"active checkpoint commands disagree");w.execute(Command::decode(&wire)?)
}
fn state(w:&mut Worker)->Result<Value> {
    Ok(json!({"frontier":w.verify.position(),"pending":w.history.pending(),
        "carry":w.verify.carry_digests()?,"encoder":w.verify.encoder_tail_digest()?,
        "draft":w.session.as_mut().unwrap().conditioning_state_digest_for_probe(&w.comm,&w.stream)?,
        "history":w.session.as_ref().unwrap().history_lookup_digest_for_probe()?}))
}
fn following(w:&mut Worker,ids:&[u32])->Result<String> {
    let before=state(w)?;
    w.verify.begin_verification(ids,&w.comm,&w.stream)?;
    let (y,n)=w.verify.pending_logits()?;ensure!(n==6,"active checkpoint following rows");
    let raw=unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?;
    w.verify.finish_verification(0,&w.comm,&w.stream)?;
    ensure!(state(w)?==before,"active checkpoint abort changed state");
    Ok(format!("{:x}",Sha256::digest(raw)))
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_ACTIVE_ENCODER_CHECKPOINT_SCREEN").as_deref()==Ok("1")
        &&w.encoder_prompt_checkpoint&&w.encoder_active_frontier&&w.prompt_checkpoint
        &&w.history_lookup_enabled&&w.current.is_none()&&w.suspended_image_count==0,
        "active checkpoint finite worker admission");
    let raw=include_bytes!("../tests/fixtures/target-agentic-repository-v1.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture.get("tokens").unwrap_or(&fixture["batches"][0]).clone())?;
    let base=8209usize;ensure!(tokens.len()>=base+518,"active checkpoint fixture extent");
    // Keep this finite range below the subsequent encoder-cache probe's range.
    // The production history deliberately retains its monotonic request ID.
    let mut id=7_730_000;let mut records=Vec::new();let mut ordinary_exact=true;
    for delta in [256usize,512] {
        for repeat in 0..2u8 {
            let mut standard=None;let mut candidate=None;
            for path in if repeat==0{[2u8,1,0]}else{[0,1,2]} {
                w.control.agree(&[(delta/256)as u8,repeat,path])?;
                w.cached_sessions.clear();w.encoder_prompt_checkpoint=true;
                w.encoder_active_frontier=false;w.prompt_checkpoint=true;w.prompt_cache_logits=false;
                if let Some(s)=&mut w.session{s.configure_greedy(Some(w.cfg.eos_token_id))?;}
                id+=1;
                execute(w,Command::Begin{images: Vec::new(),id,seed:20261028,greedy:false,limit:1,tokens:tokens[..base].to_vec()})?;
                execute(w,Command::Finish(Finish::Length))?;
                let base_state=state(w)?;
                // No retained image can affect admission or turn this active
                // append into a cache-restore request.
                w.cached_sessions.clear();
                w.encoder_prompt_checkpoint=path!=2;w.prompt_checkpoint=path!=2;
                w.encoder_active_frontier=path==1;
                id+=1;
                let begin=execute(w,Command::Begin{images: Vec::new(),id,seed:20261028+delta as u64,greedy:false,
                    limit:16,tokens:tokens[..base+delta].to_vec()})?;
                ensure!(begin["retained_session_cache"]["active_reuse"]==true,"expected true active append");
                if path==1 {
                    ensure!(begin["prompt_checkpoint"]["active_frontier"]==true
                        &&begin["prompt_checkpoint"]["prefix_tokens"]==base
                        &&begin["prompt_checkpoint"]["saved"]==true,"live checkpoint not saved at active frontier");
                }
                let initial=state(w)?;
                let follow=following(w,&tokens[base+delta..base+delta+6])?;
                let policy=w.session.as_ref().unwrap().proposal_policy_for_cache_probe()?;
                let mut output:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;
                while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
                    let step=execute(w,Command::Step)?;
                    output.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);
                }
                let after=state(w)?;
                let reason=if w.history.pending()==Some(w.cfg.eos_token_id){Finish::Stop}else{Finish::Length};
                execute(w,Command::Finish(reason))?;
                let semantic=json!({"base":base_state,"initial":initial,"following_sha256":follow,
                    "after":after,"output":output});
                if path==2{standard=Some(semantic.clone());}
                if path==1{candidate=Some(semantic.clone());}
                records.push(json!({"delta":delta,"repeat":repeat,"path":path,"begin":begin,
                    "semantic":semantic,"policy":policy}));
                std::fs::write(out.join("active-checkpoint-partial.json"),serde_json::to_vec_pretty(&records)?)?;
                ensure!(w.cached_sessions.len()<=4&&w.cached_sessions.iter().map(|c|c.charge).sum::<usize>()<=w.shared_image_budget,
                    "active checkpoint exceeded cache budget");
            }
            // Preserve observations across a numerical mismatch. Transactional
            // restore/abort assertions above remain hard failures.
            ordinary_exact&=standard==candidate;
        }
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("active-checkpoint-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"ordinary_unsplit_exact":ordinary_exact,
        "scope":"One real8K repository prefix,256/512-token active appends, three paths in reversed order. RecommendedT1/P1,16-token caps. Compare live-frontier checkpoint against ordinary unsegmented append in initial/final carry,encoder,drafter,history, six following logits and emitted tokens; report rather than discard numerical mismatches. Current split policy retained as timing control; no completed-task quality or background panel."}))?)?;Ok(())
}
