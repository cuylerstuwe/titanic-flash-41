//! Finite current-worker screen, reusing a snapshot for each index policy.
use super::*;
use anyhow::Context;

fn execute(w:&mut Worker,c:Command)->Result<Value>{
    let bytes=c.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(bytes.as_slice()))?;
    ensure!(wire==bytes,"history index screen command mismatch");w.execute(c)
}
fn advance(w:&mut Worker,anchor:&mut u32,count:usize)->Result<Value>{
    let start=w.verify.position();let captured=w.verify.capture_statistics().1+w.long_verify.as_ref().unwrap().capture_statistics().1;
    let mut ids=Vec::new();let mut steps=Vec::new();
    w.control.agree(b"associativity-timing-start")?;let began=Instant::now();
    while ids.len()<count&&*anchor!=w.cfg.eos_token_id {
        let left=count-ids.len();let s=w.session.as_mut().unwrap();
        let long=s.try_long_history_step(&mut w.verify,w.long_verify.as_mut().unwrap(),
            w.long_history_buffers.as_mut().unwrap(),*anchor,left,&w.comm,&w.stream)?;
        let step=match long {Some(s)=>s,None=>s.step(&mut w.verify,*anchor,left.min(6),&w.comm,&w.stream)?};
        ensure!(step.tokens.first()==Some(anchor)&&step.tokens.len()<=left,"index screen accepted frontier");
        ids.extend_from_slice(&step.tokens);*anchor=step.next_anchor;steps.push(step);
    }
    w.stream.synchronize()?;w.comm.health()?;let seconds=began.elapsed().as_secs_f64();
    ensure!(w.verify.position()==start+ids.len()as u64&&w.session.as_ref().unwrap().history_lookup_metadata().unwrap().0==w.verify.position(),
        "index screen target/history frontiers disagree");
    Ok(json!({"seconds":seconds,"committed_inputs":ids,"pending_anchor":anchor,"steps":steps,
        "tokens_per_second":ids.len()as f64/seconds,
        "new_verifier_captures":w.verify.capture_statistics().1+w.long_verify.as_ref().unwrap().capture_statistics().1-captured,
        "frontier":w.verify.position()}))
}
fn readout(w:&mut Worker,teacher:&[u32])->Result<String>{
    w.verify.begin_verification(teacher,&w.comm,&w.stream)?;
    let(y,n)=w.verify.pending_logits()?;ensure!(n==6,"index screen following rows");
    let bytes=unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?;
    w.verify.finish_verification(0,&w.comm,&w.stream)?;
    Ok(format!("{:x}",Sha256::digest(bytes)))
}
fn warm_profiles(w:&mut Worker,teacher:&[u32])->Result<()> {
    for live in 1..=6 {w.verify.begin_verification(&teacher[..live],&w.comm,&w.stream)?;w.verify.finish_verification(0,&w.comm,&w.stream)?;}
    let long=w.long_verify.as_mut().unwrap();long.handoff_from(&w.verify)?;
    long.begin_verification(teacher,&w.comm,&w.stream)?;long.finish_verification(0,&w.comm,&w.stream)?;
    w.verify.handoff_from(long)
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_HISTORY_ASSOCIATIVITY_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none()&&w.long_verify.is_some()&&w.long_history_buffers.is_some()
        &&w.history_lookup_enabled,"finite history index screen admission");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture:Value=serde_json::from_slice(raw)?;let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let strong=std::env::var("CANADA_SCRATCH_HISTORY_STRONG_BACKUP_SCREEN").as_deref()==Ok("1");
    let transfer=std::env::var("CANADA_SCRATCH_HISTORY_BACKUP_TRANSFER").as_deref()==Ok("1");
    ensure!(!transfer||strong,"backup transfer needs the strong backup screen");
    let frontiers=include_bytes!("../tests/fixtures/history-strong-backup-frontiers-v1.json");
    let transfer_prompts=include_bytes!("../tests/fixtures/root-feedback-exploration-tokens-v1.json");
    let transfer_phases=include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json");
    let cases:Vec<(String,Vec<u32>,Vec<u32>)>=if transfer {
        // Different task families plus an unselected long-context frontier.
        // These are ordinary target-sampled continuations, never selected by
        // whether the backup index happens to recover a missing copy.
        let prompts:Value=serde_json::from_slice(transfer_prompts)?;
        let phases:Value=serde_json::from_slice(transfer_phases)?;
        let mut cases=Vec::new();
        for c in prompts["cases"].as_array().unwrap().iter().take(2)
            .chain(phases["cases"].as_array().unwrap().iter().filter(|c|c["id"]=="offline-product-ideation-answer-2")) {
            cases.push((c["id"].as_str().unwrap().to_string(),serde_json::from_value(c["input_ids"].clone())?,tokens[262500..262516].to_vec()));
        }
        cases.push(("unselected-tokio-262144".to_string(),tokens[..262144].to_vec(),tokens[262500..262516].to_vec()));
        ensure!(cases.len()==4,"history transfer task coverage");cases
    }else if strong {
        let f:Value=serde_json::from_slice(frontiers)?;
        f["cases"].as_array().unwrap().iter().map(|c|->Result<_>{
            let n=c["corpus_tokens"].as_u64().unwrap()as usize;let mut prefix=tokens[..n].to_vec();
            prefix.extend(serde_json::from_value::<Vec<u32>>(c["suffix_ids"].clone())?);
            Ok((c["name"].as_str().unwrap().to_string(),prefix,serde_json::from_value(c["teacher_ids"].clone())?))
        }).collect::<Result<_>>()?
    }else{[65520usize,262144].into_iter().map(|n|(format!("tokio-{n}"),tokens[..n].to_vec(),tokens[262500..262516].to_vec())).collect()};
    let mut records=Vec::new();
    // Cache policies are outside this algorithm screen. One source image is
    // explicitly budgeted and restored; normal inactive images stay empty.
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    for(case,(name,prompt,teacher))in cases.into_iter().enumerate(){
        let length=prompt.len();ensure!(teacher.len()==16,"history screen teacher extent");
        let begin=execute(w,Command::Begin{images:Vec::new(),id:7_810_000+case as u64,
            seed:2026103000+case as u64,greedy:false,limit:1,tokens:prompt})?;
        ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.
            &&w.session.as_ref().unwrap().draft_temperature()==0.7f32
            &&begin["block_verification"]==true,
            "index screen changed the recommended target or carried draft settings");
        let anchor=w.history.pending().unwrap();let frontier=w.verify.position();
        let(bytes,host)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
        ensure!(bytes+host<=1024usize.pow(3),"bounded history screen image");
        let fits=w.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+bytes+host;
        ensure!(w.control.all_admit(fits)?,"history screen image admission");
        let saved=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
        let original=readout(w,&teacher[..6])?;
        warm_profiles(w,&teacher)?;
        w.session.as_mut().unwrap().configure_history_policy_for_probe(None)?;
        let mut warm_anchor=anchor;let warm=advance(w,&mut warm_anchor,64)?;
        // The65,520 frontier crosses the adaptive-index boundary at65,536.
        // Warm every live width in the bank selected after that crossing too;
        // ordinary warmup may only encounter M3/M6, not the final short tail.
        warm_profiles(w,&teacher)?;
        for repeat in 0..2u64 {
            for ways in if repeat==0{[1,4]}else{[4,1]} {
                // This probe temporarily exercises the real worker's state.
                // Allocate from its counter so a later divergent Begin can
                // reset to a strictly newer lineage after the probe restores.
                w.target_lineage=w.target_lineage.checked_add(1).context("history probe lineage exhausted")?;
                let mut anchor=w.session.as_mut().unwrap().restore_state_image(&mut w.verify,&saved,
                    w.target_lineage,&w.comm,&w.stream)?;
                ensure!(w.verify.position()==frontier&&readout(w,&teacher[..6])?==original,
                    "index policy trial did not restore the same target logits");
                let s=w.session.as_mut().unwrap();s.configure_sampling(2026103100+case as u64*10+repeat,Some(w.cfg.eos_token_id))?;
                let base_bytes=crate::dspark_generation::history_lookup_reserved_bytes(w.verify.context_capacity())?;
                ensure!(s.history_lookup_metadata().unwrap().1==base_bytes,
                    "restored complete-session image lost its reserved history capacity");
                let expected_bytes=base_bytes+if strong&&ways==4{65536*4}else{0};
                let rebuilding=Instant::now();
                if strong{s.configure_history_strong_backup_for_probe(ways==4)?;}
                else{s.configure_history_associativity_for_probe(ways)?;}
                let rebuild_seconds=rebuilding.elapsed().as_secs_f64();
                ensure!(s.history_lookup_metadata().unwrap().1==expected_bytes,"associative index changed its byte budget");
                let result=advance(w,&mut anchor,256)?;
                let captures=result["new_verifier_captures"].as_u64().unwrap();
                let record=json!({"case":case,"name":name,"prefix_tokens":length,"repeat":repeat,"ways":if strong{1}else{ways},
                    "strong_backup":strong&&ways==4,"variant":if strong&&ways==4{"strong_backup"}else if ways==1{"direct"}else{"four_way"},
                    "rebuild_seconds":rebuild_seconds,"index_bytes":expected_bytes,"result":result,
                    "same_prefix_logits_before_trial":true,"restored_full_history_reservation":true});
                eprintln!("history buckets rank{rank} context{length} repeat{repeat} ways{ways}: {:.2}tok/s",record["result"]["tokens_per_second"].as_f64().unwrap());
                records.push(record);std::fs::write(out.join("history-associativity-partial.json"),serde_json::to_vec_pretty(&records)?)?;
                ensure!(captures==0,"measured index continuation captured {captures} verifier graphs");
            }
        }
        w.target_lineage=w.target_lineage.checked_add(1).context("history probe lineage exhausted")?;
        w.session.as_mut().unwrap().restore_state_image(&mut w.verify,&saved,w.target_lineage,&w.comm,&w.stream)?;
        ensure!(w.verify.position()==w.history.frontier()as u64&&readout(w,&teacher[..6])?==original,
            "index screen failed final state restoration");
        execute(w,Command::Finish(Finish::Length))?;
        records.push(json!({"case":case,"warmup":warm,"begin":begin,"restored_before_finish":true}));
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("history-associativity-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "transfer_screen":transfer,"transfer_fixture_sha256":[format!("{:x}",Sha256::digest(transfer_prompts)),format!("{:x}",Sha256::digest(transfer_phases))],
        "strong_backup_screen":strong,"frontier_fixture_sha256":if strong{Some(format!("{:x}",Sha256::digest(frontiers)))}else{None},
        "scope":if transfer{"Go concurrency, JavaScript streaming, an ideation answer frontier and unselected262Krepository continuation. Two seeds, opposite-order256-token pairs; current targetT1/P1,actual-q neuralT.7,primarydirectversus256KiB64keybackup. Prefix selection independentofbackuphits. Fullhistoryreservation and restoredlogits checked. Capped transfer/overhead screen, notcompleted taskquality."}else if strong{"Two selected public copy frontiers, two seeds and opposite-order256-token pairs. Primary direct index versus256KiB64-key collision backup. TargetT1/P1, actual-q neuralT.7 and corrected history proposals; M6/M16. Complete history reservation and restored target logits checked each trial. Mechanism screen, not workload incidence, outcomes or generalization."}else{"One public repository conversation at65K/262K, two seeds and opposite-order256-token pairs. Current neural actual-q and point-mass rejection, targetT1/P1, draftT.7, M6/M16. Equal metadata memory; restored target logits checked before every trial. Short opportunity screen; not completed outcomes or generalization proof."}}))?)?;Ok(())
}
