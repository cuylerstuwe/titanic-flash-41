//! Actual public conversation operands for the next pooled component screen.
use super::*;
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path,startup:Value)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_POOLED_ROUTE_CAPTURE").as_deref()==Ok("1")
        &&w.current.is_none()&&(w.history.frontier()==0
            ||std::env::var("CANADA_SCRATCH_ATTENTION16_COMMIT_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_PEER_PREFILL_SCREEN").as_deref()==Ok("1"))
        &&w.session_cache_budget==0
        &&!w.prompt_checkpoint&&!w.prompt_cache_logits,"finite pooled route capture admission");
    let short:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/root-feedback-exploration-tokens-v1.json"))?;
    ensure!(short["thinking_mode"]=="thinking"&&short["reasoning_effort"]=="high","canonical public templates");
    let saved:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json"))?;
    let long:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let c=short["cases"].as_array().unwrap().iter().find(|c|c["id"]=="go-coalescing-cancellation").unwrap();
    let go:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
    let c=saved["cases"].as_array().unwrap().iter().find(|c|c["id"]=="offline-product-ideation-answer-3").unwrap();
    let mut idea:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;idea.push(c["anchor"].as_u64().unwrap()as u32);
    let tokens:Vec<u32>=serde_json::from_value(long["tokens"].clone())?;
    let inherited=std::env::var("CANADA_SCRATCH_POOLED_INHERITED_ROUTE_CAPTURE").as_deref()==Ok("1");
    let cases:Vec<(String,String,Vec<u32>)>=if inherited {
        let f:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/inherited-route-pair-v1.json"))?;
        ensure!(f["shared_prefix_tokens"].as_u64().is_some_and(|n|n>=18000),"inherited route fixture prefix");
        f["cases"].as_array().context("inherited route cases")?.iter().map(|c|Ok((
            c["id"].as_str().context("route case id")?.to_owned(),"reasoning".to_owned(),
            serde_json::from_value(c["input_ids"].clone())?))).collect::<Result<_>>()?
    }else{vec![("go-concurrency".into(),"reasoning".into(),go),
        ("product-ideation".into(),"answer".into(),idea),
        ("repository-131k".into(),"continuation".into(),tokens[..131072].to_vec())]};
    ensure!(cases.len()==3,"bounded route capture case count");
    // The optional preceding committed-state/prefill screens use IDs below8M.
    // Preserve the ordinary monotonic identity contract in this finite chain.
    let mut records=Vec::new();let mut id=8_684_001;
    let mut progress=std::fs::OpenOptions::new().create_new(true).write(true).open(out.join("pooled-route-records.jsonl"))?;
    for (ordinal,(name,phase,prompt))in cases.into_iter().enumerate() {
        let seed=202684000+ordinal as u64;let cap=192;
        let begin=w.execute(Command::Begin{images: Vec::new(),id,seed,greedy:false,limit:cap,tokens:prompt.clone()})?;id+=1;
        ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.
            &&begin["generation_fp32_attention"]==true&&begin["block_verification"]==true,"capture target policy fixed");
        let mut ids:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;
        let mut steps=Vec::new();let mut captures=Vec::new();let mut decode_seconds=0.;
        while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
            let at=Instant::now();let step=w.execute(Command::Step)?;decode_seconds+=at.elapsed().as_secs_f64();
            ids.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);
            if captures.len()<2&&w.emitted>=[32,96][captures.len()]&&step["verified_rows"]==6&&step["proposal_source"]=="dspark" {
                let relative=format!("pooled-routes/{name}/front{}",captures.len());
                let before=(w.verify.carry_digests()?,w.verify.encoder_tail_digest()?);
                let snapshot=w.verify.capture_expert_components(&out.join(&relative),&w.stream)?;
                ensure!(before==(w.verify.carry_digests()?,w.verify.encoder_tail_digest()?),"read-only route capture changed retained state");
                captures.push(json!({"directory":relative,"emitted":w.emitted,"step":steps.len(),"snapshot":snapshot}));
                w.control.agree(b"pooled-capture-complete")?;
            }
            steps.push(step);
        }
        let stop=w.history.pending()==Some(w.cfg.eos_token_id);
        let finish=w.execute(Command::Finish(if stop{Finish::Stop}else{Finish::Length}))?;
        ensure!(w.history.frontier()==prompt.len()+ids.len()-1&&w.history.pending()==ids.last().copied()
            &&ids.len()==w.emitted&&(stop||w.emitted==cap),"captured conversation frontier");
        let record=json!({"case":name,"phase":phase,"seed":seed,"cap":cap,"prompt_tokens":prompt.len(),
            "begin":begin,"finish":finish,"output_ids":ids,"steps":steps,"decode_seconds_without_capture":decode_seconds,"captures":captures});
        writeln!(progress,"{}",record)?;progress.flush()?;records.push(record);
        eprintln!("pooled routes rank{rank} {name} complete");
    }
    w.execute(Command::Shutdown)?;
    std::fs::write(out.join("pooled-route-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,"startup":startup,
        "records":records,"all_frontiers_checked":true,"inherited_pair":inherited,
        "scope":"Three public tasks; optional inherited-prefix edit/test pair plus independent review.192token cap each, at mosttwo actualM6snapshots per case. CaptureIO excluded from step times; no timingclaim or completed-taskquality. Foregroundonly, current targetT1/P1/FP32attention and actual-q sampling unchanged."}))?)?;
    Ok(())
}
