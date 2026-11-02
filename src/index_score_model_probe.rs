//! Finite storage-only comparison across separately pinned off/on launches.
//! Wide prefill, cached branches, graph reuse and M6/M16 transaction boundaries.
use super::*;
use super::history32_generation_probe::execute;

fn state(w:&Worker)->Result<Value> {
    Ok(json!({"position":w.verify.position(),"carry":w.verify.carry_digests()?,
        "encoder":w.verify.encoder_tail_digest()?}))
}
fn restore(w:&mut Worker,image:&crate::target::StateImage)->Result<()> {
    w.target_lineage=w.target_lineage.checked_add(1).context("score pool lineage exhausted")?;
    w.verify.restore_state_image(image,w.target_lineage,&w.comm,&w.stream)
}
fn following(w:&mut Worker,ids:&[u32])->Result<String> {
    w.verify.begin_verification(ids,&w.comm,&w.stream)?;
    let(y,n)=w.verify.pending_logits()?;ensure!(n==ids.len(),"score following extent");
    let hash=format!("{:x}",Sha256::digest(unsafe{w.cuda.read_device(y.ptr(),n*129280*4)}?));
    w.verify.finish_verification(0,&w.comm,&w.stream)?;Ok(hash)
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_INDEX_SCORE_MODEL_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none()&&w.encoder_prompt_checkpoint&&w.long_verify.is_some(),
        "index score storage screen admission");
    let shared=std::env::var("CANADA_SCRATCH_SHARED_INDEX_SCORES").as_deref()==Ok("1");
    let loaded_memory=w.cuda.admission_memory()?;
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    w.cached_sessions.clear();w.encoder_active_frontier=false;
    w.encoder_upgrade_reconstructed=false;w.prompt_cache_logits=false;
    let mut parent=0;let mut records=Vec::new();
    for(case,delta)in [0usize,8321,641].into_iter().enumerate() {
        let start=200000+case*10000;
        let prompt=if case==0 {tokens[..131328].to_vec()}
            else {[&tokens[..parent],&tokens[start..start+delta]].concat()};
        let event=execute(w,Command::Begin{images:Vec::new(),id:7_930_000+case as u64,
            seed:20261031,greedy:false,limit:1,tokens:prompt})?;
        ensure!(event["target_temperature"]==1.0&&event["target_top_p"]==1.0,
            "score pool sampler contract");
        if case==0 {
            parent=event["prompt_checkpoint"]["prefix_tokens"].as_u64().context("score parent absent")?as usize;
            ensure!(event["prompt_checkpoint"]["saved"]==true&&parent>=131072,"score parent not retained");
        }else{ensure!(event["cached_tokens"]==parent,"score pool branch missed parent");}
        let bytes=w.verify.compact_state_image_bytes()?;
        ensure!(bytes<512*1024usize.pow(2)&&w.control.all_admit(
            w.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+bytes)?,
            "score pool state snapshot reserve");
        let image=w.verify.save_state_image(bytes,&w.comm,&w.stream)?;
        let original=state(w)?;
        let draft=w.session.as_mut().unwrap().conditioning_state_digest_for_probe(&w.comm,&w.stream)?;
        let history=w.session.as_ref().unwrap().history_lookup_digest_for_probe()?;
        let mut trials=Vec::new();
        // Each row shape exercises a cold capture, replay, padding, rejection
        // and committed prefix. Never infer outcome quality from these readouts.
        for(rows,live,accepted)in [(6,6,0),(16,16,0),(6,3,1),(16,9,7),(6,6,6),(16,16,16)] {
            restore(w,&image)?;
            if rows==16 {w.long_verify.as_mut().unwrap().handoff_from(&w.verify)?;}
            let target=if rows==6 {&mut w.verify}else{w.long_verify.as_mut().unwrap()};
            let started=Instant::now();
            target.begin_verification(&tokens[start..start+live],&w.comm,&w.stream)?;
            let (y,n)=target.pending_local_logits_for_probe()?;ensure!(n==live,"score pool live rows");
            let logits=format!("{:x}",Sha256::digest(unsafe{w.cuda.read_device(y.ptr(),n*32320*4)}?));
            let elapsed=started.elapsed().as_secs_f64();
            target.finish_verification(accepted,&w.comm,&w.stream)?;
            if rows==16 {w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;}
            let retained=state(w)?;
            ensure!(accepted!=0||retained==original,"score pool rollback changed state");
            let next=following(w,&tokens[start+40..start+46])?;
            ensure!(state(w)?==retained,"score pool following rollback changed state");
            trials.push(json!({"rows":rows,"live":live,"accepted":accepted,
                "logits":logits,"state":retained,"following_logits":next,
                "forward_and_readout_seconds":elapsed}));
        }
        restore(w,&image)?;ensure!(state(w)?==original,"score base restoration");
        ensure!(w.session.as_mut().unwrap().conditioning_state_digest_for_probe(&w.comm,&w.stream)?==draft
            &&w.session.as_ref().unwrap().history_lookup_digest_for_probe()?==history,
            "score probe changed accepted drafter/history");
        let record=json!({"case":case,"begin":event,"state":original,"draft":draft,"history":history,
            "trials":trials,"memory":w.cuda.admission_memory()?,"restoration_exact":true});
        std::fs::write(out.join(format!("score-pool-c{case}.json")),serde_json::to_vec_pretty(&record)?)?;
        records.push(record);drop(image);execute(w,Command::Finish(Finish::Length))?;
        eprintln!("rank{rank} score pool case{case} complete, shared={shared}");
    }
    let memory=w.cuda.admission_memory()?;execute(w,Command::Shutdown)?;
    std::fs::write(out.join("index-score-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"shared":shared,"records":records,"loaded_memory":loaded_memory,"memory":memory,
        "fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "scope":"Separate pinned flag-off/on launches. One131Kpublic prompt plus8K/641-token cached branches; wide prefill, complete local FP32 vocabulary shards on all4ranks for M6/M16, padding, partial/full acceptance, rollback and following gathered M6. Canonical arithmetic and compact head unchanged. Compare cross-launch state/logit signatures; no task-quality or generation-speed claim. M32 capacity is a separate next screen."
    }))?)?;Ok(())
}
