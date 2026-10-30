//! Finite native compact-mask screen. Compare two separately pinned launches
//! with the flag off/on; preserve complete state and following FP32 rows.
use super::*;

fn execute(w:&mut Worker,c:Command)->Result<Value> {
    let encoded=c.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(encoded.as_slice()))?;
    ensure!(wire==encoded,"mask probe command replicas disagree");
    w.execute(Command::decode(&wire)?)
}
fn state(w:&mut Worker)->Result<Value> {
    Ok(json!({"frontier":w.verify.position(),"carry":w.verify.carry_digests()?,
        "encoder":w.verify.encoder_tail_digest()?,
        "draft":w.session.as_mut().unwrap().conditioning_state_digest_for_probe(&w.comm,&w.stream)?,
        "history":w.session.as_ref().unwrap().history_lookup_digest_for_probe()?}))
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_MASK_STORAGE_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none()&&w.suspended_image_count==0&&w.encoder_prompt_checkpoint,
        "native mask screen admission");
    let compact=std::env::var("CANADA_SCRATCH_COMPACT_RETAINED_MASK").as_deref()!=Ok("0");
    let profiles:Vec<_>=w.wides.iter().filter_map(Target::wide_report).collect();
    ensure!(profiles.len()>=3&&profiles.iter().all(|p|p["candidate_scratch"]["compact_mask"]==compact),
        "native mask layout was not applied to every wide profile");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    ensure!(tokens.len()>=200006,"native mask fixture extent");
    w.cached_sessions.clear();w.encoder_active_frontier=false;
    w.encoder_upgrade_reconstructed=false;w.prompt_cache_logits=false;
    let mut records=Vec::new();let mut parent=0usize;
    // One long source plus three partial-width branches; the final changed
    // branch replays the8K graph and reuses the original shared parent again.
    for(case,delta)in [0usize,8321,1153,641,8321].into_iter().enumerate() {
        let start=150000+case*9000;
        let prompt=if case==0 {tokens[..65792].to_vec()}
            else {[&tokens[..parent],&tokens[start..start+delta]].concat()};
        let event=execute(w,Command::Begin{images:Vec::new(),id:7_790_000+case as u64,
            seed:20261030,greedy:false,limit:1,tokens:prompt})?;
        ensure!(event["target_temperature"]==1.0&&event["target_top_p"]==1.0,
            "native mask screen target sampler changed");
        if case==0 {
            parent=event["prompt_checkpoint"]["prefix_tokens"].as_u64()
                .ok_or_else(||anyhow::anyhow!("mask probe parent absent"))?as usize;
            ensure!(event["prompt_checkpoint"]["saved"]==true&&parent>=65536,
                "mask probe long parent not saved");
        } else {ensure!(event["cached_tokens"]==parent,"mask branch missed original parent");}
        let before=state(w)?;
        w.verify.begin_verification(&tokens[200000..200006],&w.comm,&w.stream)?;
        let(logits,rows)=w.verify.pending_logits()?;ensure!(rows==6,"mask verification rows");
        let bytes=unsafe{w.cuda.read_device(logits.ptr(),6*129280*4)}?;
        w.verify.finish_verification(0,&w.comm,&w.stream)?;
        ensure!(state(w)?==before,"mask screen verification rollback changed state");
        let name=format!("mask-c{case}-following.f32");std::fs::write(out.join(&name),&bytes)?;
        let record=json!({"case":case,"begin":event,"state":before,"logits_file":name,
            "transport":w.comm.transport_memory(),
            "following_sha256":format!("{:x}",Sha256::digest(&bytes)),"rollback_exact":true});
        std::fs::write(out.join(format!("mask-c{case}.json")),serde_json::to_vec_pretty(&record)?)?;
        records.push(record);execute(w,Command::Finish(Finish::Length))?;
    }
    let memory=w.cuda.admission_memory()?;execute(w,Command::Shutdown)?;
    std::fs::write(out.join("mask-storage-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"compact_mask":compact,"profiles":profiles,"records":records,"memory":memory,
        "fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "scope":"One65Kpublic context, three retained partial-width branches and changed-input8K replay. Compare matched off/on native launches for whole-model state and6following FP32 rows. Memory/layout screen, not throughput or task-quality qualification."}))?)?;
    Ok(())
}
