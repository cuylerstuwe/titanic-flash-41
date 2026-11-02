//! Fixed committed-history copy operands for a bounded16/32-row cost screen.
use super::*;
fn execute(w:&mut Worker,c:Command)->Result<Value>{
    let bytes=c.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(bytes.as_slice()))?;
    ensure!(wire==bytes,"copy route command mismatch");w.execute(c)
}
fn readout(w:&mut Worker,teacher:&[u32])->Result<String>{
    w.verify.begin_verification(teacher,&w.comm,&w.stream)?;
    let(y,n)=w.verify.pending_logits()?;ensure!(n==6,"copy route following readout geometry");
    let bytes=unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?;
    w.verify.finish_verification(0,&w.comm,&w.stream)?;
    Ok(format!("{:x}",Sha256::digest(bytes)))
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_HISTORY_COPY_ROUTE_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none()&&w.long_verify.is_some(),"finite copy route admission");
    let fixture_raw=include_bytes!("../tests/fixtures/history-copy-width32-v1.json");
    let fixture:Value=serde_json::from_slice(fixture_raw)?;
    let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    let selected:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/history-strong-backup-frontiers-v1.json"))?;
    let tools:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-history-score-v1.json"))?;
    ensure!(fixture["schema"]==1&&fixture["cases"].as_array().unwrap().len()==3,"bounded copy fixture");
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let mut records=Vec::new();let teacher=&tokens[262500..262506];
    for(case,c)in fixture["cases"].as_array().unwrap().iter().enumerate(){
        let ordinal=c["frontier_index"].as_u64().unwrap()as usize;
        let prompt:Vec<u32>=if c["kind"]=="repository"{
            let f=&selected["cases"][ordinal];let mut p=tokens[..f["corpus_tokens"].as_u64().unwrap()as usize].to_vec();
            p.extend(serde_json::from_value::<Vec<u32>>(f["suffix_ids"].clone())?);p
        }else{ensure!(c["kind"]=="tool","copy fixture kind");serde_json::from_value(tools["cases"][ordinal]["prompt_ids"].clone())?};
        let n=prompt.len();let end=c["source_end"].as_u64().unwrap()as usize;
        let anchor=c["anchor"].as_u64().unwrap()as u32;
        ensure!(n==c["prefix_tokens"].as_u64().unwrap()as usize&&n>=64&&end>=63&&end+32<=n
            &&prompt[end]==anchor&&prompt[end-63..end]==prompt[n-63..],"copy must extend a64-token committed-history match");
        let mut candidate=vec![anchor];candidate.extend_from_slice(&prompt[end+1..end+32]);
        let begin=execute(w,Command::Begin{images:Vec::new(),id:7_830_000+case as u64,
            seed:2026110100+case as u64,greedy:false,limit:1,tokens:prompt})?;
        ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.
            &&w.verify.position()==n as u64,"copy capture prompt policy/frontier");
        let budget=w.verify.compact_state_image_bytes()?;
        ensure!(budget<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
            >=8*1024usize.pow(3)+budget)?,"bounded target-only copy snapshot");
        let saved=w.verify.save_state_image(budget,&w.comm,&w.stream)?;
        let original=readout(w,teacher)?;
        let state=(w.verify.carry_digests()?,w.verify.encoder_tail_digest()?);
        let mut captures=Vec::new();
        let long=w.long_verify.as_mut().unwrap();long.handoff_from(&w.verify)?;
        for chunk in 0..2{
            // Evaluate the known candidate as a hypothetical causal prefix.
            // It is never emitted or added to the request/drafter history.
            long.begin_verification(&candidate[chunk*16..chunk*16+16],&w.comm,&w.stream)?;
            long.finish_verification(16,&w.comm,&w.stream)?;
            let before=(long.carry_digests()?,long.encoder_tail_digest()?);
            let directory=format!("copy-routes/case{case}/chunk{chunk}");
            let snapshot=long.capture_expert_components(&out.join(&directory),&w.stream)?;
            ensure!(before==(long.carry_digests()?,long.encoder_tail_digest()?),"route readout changed retained state");
            captures.push(json!({"directory":directory,"chunk":chunk,"snapshot":snapshot}));
            w.control.agree(b"copy-route-capture-complete")?;
        }
        ensure!(long.position()==n as u64+32,"copy capture target extent");
        w.verify.handoff_from(long)?;
        w.target_lineage=w.target_lineage.checked_add(1).context("copy route lineage exhausted")?;
        w.verify.restore_state_image(&saved,w.target_lineage,&w.comm,&w.stream)?;
        ensure!(w.verify.position()==n as u64&&state==(w.verify.carry_digests()?,w.verify.encoder_tail_digest()?)
            &&readout(w,teacher)?==original,"copy capture did not restore exact target state/logits");
        ensure!(w.session.as_ref().unwrap().history_lookup_metadata().unwrap().0==n as u64
            &&w.history.frontier()==n,"copy capture touched accepted request history");
        records.push(json!({"case":case,"name":c["name"],"kind":c["kind"],"prompt_tokens":n,
            "source_end":end,"candidate_ids":candidate,"begin_pending_anchor":w.history.pending(),
            "restored_logits_sha256":original,"restoration_exact":true,"captures":captures}));
        execute(w,Command::Finish(Finish::Length))?;
        eprintln!("copy routes rank{rank} case{case}: twoM16captures, exact target/history restoration");
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("copy-route-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,
        "fixture_sha256":format!("{:x}",Sha256::digest(fixture_raw)),"records":records,"all_restorations_exact":true,
        "scope":"Three public committed-history64-match candidates. Capture two consecutive16-row teacher-forced target blocks, all40router IDs and layers0/28expert operands. Hypothetical candidates are not emitted and need not be accepted by sampling. Exact target restoration and six following logits checked. Component economics only; no native32verifier, new selection policy, outcome-quality or speed claim."}))?)?;
    Ok(())
}
