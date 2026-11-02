//! Diagnose one observed fresh/restored prompt divergence on fixed public input.
//! Capture bytes only; probability comparisons run separately on the GPU.
use super::*;

fn execute(w:&mut Worker,c:Command)->Result<Value> {
    let bytes=c.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(bytes.as_slice()))?;
    ensure!(wire==bytes,"restore diagnostic command disagreement");
    w.execute(Command::decode(&wire)?)
}
fn reset(w:&mut Worker)->Result<()> {
    ensure!(w.current.is_none()&&w.pending_prefill.is_none(),"restore diagnostic reset requires idle owner");
    w.cached_sessions.clear();
    w.target_lineage=w.target_lineage.checked_add(1).context("diagnostic lineage exhausted")?;
    w.verify.reset(w.target_lineage,&w.comm,&w.stream)?;
    w.large.handoff_from(&w.verify)?;
    w.history=ResidentHistory::new(w.large.context_capacity(),w.cfg.eos_token_id)?;
    Ok(())
}
fn state(w:&Worker)->Result<Value> {
    let session=w.session.as_ref().context("diagnostic attached drafter")?;
    let draft=session.starting_repeat_digest(&w.stream)?;
    Ok(json!({"position":w.verify.position(),"carry":w.verify.carry_digests()?,
        "encoder":w.verify.encoder_tail_digest()?,"draft_accepted":draft["draft"],
        "history":session.history_lookup_digest_for_probe()?,
        "proposal_policy":session.proposal_policy_for_cache_probe()?}))
}
fn persist(out:&Path,name:&str,bytes:&[u8])->Result<Value> {
    let file=format!("{name}.f32");
    std::fs::write(out.join(&file),bytes)?;
    Ok(json!({"file":file,"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes))}))
}
fn request(w:&mut Worker,id:u64,case:&Value,cap:usize,capture:Option<(&Path,&str)>)->Result<Value> {
    let ids:Vec<u32>=serde_json::from_value(case["input_ids"].clone())?;
    let seed=case["seed"].as_u64().context("diagnostic seed")?;
    let begin=execute(w,Command::Begin{images:Vec::new(),id,seed,greedy:false,limit:cap,tokens:ids.clone()})?;
    let initial=if let Some((out,name))=capture {
        let bytes=crate::target::take_restore_initial_logits_for_probe()?;
        ensure!(bytes.len()==129280*4,"diagnostic prompt logit extent");
        Some(json!({"logits":persist(out,name,&bytes)?,"state":state(w)?,
            "anchor":w.history.pending(),"begin":begin}))
    }else{None};
    let mut steps=Vec::new();
    while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
        let step=execute(w,Command::Step)?;
        if capture.is_some(){steps.push(step);}
    }
    let reason=if w.history.pending()==Some(w.cfg.eos_token_id){Finish::Stop}else{Finish::Length};
    let finish=execute(w,Command::Finish(reason))?;
    Ok(json!({"initial":initial,"steps":steps,"finish":finish}))
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_RESTORE_INITIAL_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_ENCODER_CACHE_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_DRAFT_REPLAY_TRACE").as_deref()==Ok("1")
        &&w.current.is_none()&&w.encoder_prompt_checkpoint&&w.prompt_cache_logits
        &&w.session_cache_budget>0,"finite initial-restore diagnostic admission");
    let raw=include_bytes!("../tests/fixtures/restore-initial19k-v1.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let cases=fixture["cases"].as_array().context("diagnostic cases")?;
    ensure!(cases.len()==4,"diagnostic fixture shape");
    let child=&cases[3];let ids:Vec<u32>=serde_json::from_value(child["input_ids"].clone())?;
    ensure!(ids.len()==18979&&cases[1]["input_ids"].as_array().unwrap().len()==18914
        &&w.prompt_checkpoint_positions(ids.len())[0]==18816,"diagnostic prompt/checkpoint geometry");
    let parent:Vec<u32>=serde_json::from_value(cases[1]["input_ids"].clone())?;
    ensure!(parent[..18688]==ids[..18688],"diagnostic parent prefix differs");
    let mut records=Vec::new();let mut id=9_340_000u64;
    request(w,id,&cases[0],32,None)?;id+=1;
    reset(w)?;
    records.push(json!({"path":"fresh","request":request(w,id,child,16,Some((out,"fresh")))?}));id+=1;
    reset(w)?;
    for case in [&cases[1],&cases[1],&cases[2]] {
        request(w,id,case,384,None)?;id+=1;
    }
    let restored=request(w,id,child,16,Some((out,"restored")))?;id+=1;
    ensure!(restored["initial"]["begin"]["cached_tokens"]==18688,
        "diagnostic did not reproduce the observed encoder-parent restore");
    records.push(json!({"path":"restored","request":restored}));
    std::fs::write(out.join("restore-initial-progress.json"),serde_json::to_vec_pretty(&records)?)?;

    // Recreate the cached path's encoder chunk boundaries, with no snapshot,
    // restore, intervening generation or omitted-state dependency.
    reset(w)?;
    w.large.prefill_encoder_prefix_for_restore_probe(&mut w.wides,&ids[..18688],&w.comm,&w.stream)?;
    let (_,details)=w.large.prefill_with_encoder_checkpoint(&mut w.wides,None,
        &ids[18688..18816],&ids[18816..],None,&w.comm,&w.stream)?;
    let values=w.large.last_prompt_logits()?;
    let bytes=values.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect::<Vec<_>>();
    let anchor=w.large.sample_token(child["seed"].as_u64().unwrap(),&w.stream)?;
    w.attach_prompt(false,0,&ids,&ids,child["seed"].as_u64().unwrap(),false)?;
    w.verify.handoff_from(&w.large)?;
    let split=json!({"logits":persist(out,"fresh-matched-chunks",&bytes)?,"state":state(w)?,
        "anchor":anchor,"details":details});
    let matched=&records[1]["request"]["initial"];
    let equal=json!({"logits":split["logits"]["sha256"]==matched["logits"]["sha256"],
        "anchor":split["anchor"]==matched["anchor"],
        "carry":split["state"]["carry"]==matched["state"]["carry"],
        "encoder":split["state"]["encoder"]==matched["state"]["encoder"],
        "draft_accepted":split["state"]["draft_accepted"]==matched["state"]["draft_accepted"],
        "history":split["state"]["history"]==matched["state"]["history"]});
    records.push(json!({"path":"fresh-matched-chunks","initial":split,"matched_restore_exact":equal}));
    reset(w)?;
    records.push(json!({"path":"fresh-repeat","request":request(w,id,child,16,Some((out,"fresh-repeat")))?}));
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("restore-initial-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),"records":records,
        "scope":"One public19K follow-up at T1/P1/high: ordinary fresh twice, actual parent+repeat+second-seed384-token requests then restore, and fresh prefill with the restored path's chunk boundaries. Raw initial logits and committed causal/draft/history bytes. Cross-plan exactness is diagnostic, not a quality gate. No serving-policy change."}))?)?;
    Ok(())
}
