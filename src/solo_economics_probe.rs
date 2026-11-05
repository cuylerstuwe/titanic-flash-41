//! Bounded pricing and route-locality screen, with no new serving policy.
use super::*;

fn retained(target: &Target) -> Result<Value> {
    Ok(json!({"position":target.position(),"carry":target.carry_digests()?,
        "encoder":target.encoder_tail_digest()?}))
}

fn price_rows(w: &mut Worker, prompt: &[u32]) -> Result<Value> {
    let before = retained(&w.verify)?;
    let mut proposed = [w.history.pending().context("pricing anchor")?; 6];
    proposed[1..].copy_from_slice(&prompt[prompt.len()-5..]);
    let mut records = Vec::new();
    // The measurements share an allocation, accepted prefix and model weights.
    // Warm every width before alternating their order; do not time readbacks.
    for rows in [2, 3, 6] {
        for _ in 0..3 {
            w.verify.begin_local_verification(&proposed[..rows], &w.comm, &w.stream)?;
            w.verify.finish_verification(0, &w.comm, &w.stream)?;
        }
    }
    let captures = w.verify.capture_statistics().1;
    for round in 0..6 {
        for rows in if round%2==0 {[2,3,6]} else {[6,3,2]} {
            w.control.agree(b"solo-row-pricing")?;
            let start = Instant::now();
            w.verify.begin_local_verification(&proposed[..rows], &w.comm, &w.stream)?;
            let target_ms = start.elapsed().as_secs_f64()*1000.;
            let metrics = serde_json::to_value(w.verify.verification_metrics()?)?;
            let commit = Instant::now();
            w.verify.finish_verification(0, &w.comm, &w.stream)?;
            records.push(json!({"rows":rows,"round":round,"target_ms":target_ms,
                "abort_ms":commit.elapsed().as_secs_f64()*1000.,"metrics":metrics}));
        }
    }
    ensure!(captures==w.verify.capture_statistics().1,"pricing recaptured a warm graph");
    ensure!(retained(&w.verify)?==before,"row-pricing changed retained state");
    let mut checks = Vec::new();
    for rows in [2,3,6] {
        w.verify.begin_local_verification(&proposed[..rows], &w.comm, &w.stream)?;
        let (buffer,live) = w.verify.pending_local_target_logits()?;
        ensure!(live==rows,"local pricing live extent");
        let values = buffer.download()?;
        let width = 129280/4;
        ensure!(values.len()>=live*width && values[..live*width].iter().all(|x|x.is_finite()),
            "nonfinite local pricing logits");
        let bytes: Vec<u8> = values[..live*width].iter().flat_map(|x|x.to_le_bytes()).collect();
        checks.push(json!({"rows":rows,"local_logits_sha256":format!("{:x}",Sha256::digest(bytes)),
            "all_live_logits_finite":true}));
        w.verify.finish_verification(0,&w.comm,&w.stream)?;
    }
    ensure!(retained(&w.verify)?==before,"pricing readback changed retained state");
    Ok(json!({"samples":records,"checks":checks,"rollback_exact":true,
        "new_timed_graph_captures":0,
        "scope":"Fixed-prefix target and zero-acceptance abort only. No drafter, sampler or accepted commit cost; not a complete-step or policy throughput claim."}))
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path,startup:Value)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_SOLO_ECONOMICS_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none()&&w.history.frontier()==0&&w.session_cache_budget==0
        &&!w.prompt_checkpoint&&!w.prompt_cache_logits&&w.verify.local_target_enabled(),
        "finite solo economics admission");
    let short:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/root-feedback-exploration-tokens-v1.json"))?;
    let saved:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json"))?;
    let long:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let c=short["cases"].as_array().unwrap().iter().find(|c|c["id"]=="go-coalescing-cancellation").unwrap();
    let go:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
    let c=saved["cases"].as_array().unwrap().iter().find(|c|c["id"]=="offline-product-ideation-answer-3").unwrap();
    let mut idea:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
    idea.push(c["anchor"].as_u64().unwrap()as u32);
    let tokens:Vec<u32>=serde_json::from_value(long["tokens"].clone())?;
    let mut records=Vec::new();
    for (ordinal,(name,phase,prompt)) in [
        ("go-concurrency","reasoning",go),
        ("product-ideation","answer",idea),
        ("repository-131k","continuation",tokens[..131072].to_vec()),
    ].into_iter().enumerate() {
        let cap=128;let seed=202611034300+ordinal as u64;
        let begin=w.execute(Command::Begin{images:Vec::new(),id:9_243_001+ordinal as u64,
            seed,greedy:false,limit:cap,tokens:prompt.clone()})?;
        ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.
            &&begin["generation_fp32_attention"]==true&&begin["block_verification"]==true,
            "solo economics canonical target policy");
        // Measure neural proposal locality separately from wide text copies.
        w.session.as_mut().context("solo economics session")?.configure_history_policy_for_probe(None)?;
        let prices=price_rows(w,&prompt)?;
        let mut ids:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;
        let mut steps=Vec::new();let mut routes=Vec::new();
        while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
            let at=Instant::now();let step=w.execute(Command::Step)?;
            let step_seconds=at.elapsed().as_secs_f64();
            ids.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);
            ensure!(step["proposal_source"]=="dspark","neural route screen used history copy");
            let live=step["verified_rows"].as_u64().context("verified rows")?as usize;
            if routes.len()<24 {
                let snapshot=w.verify.capture_consecutive_routes_for_probe(live,&w.stream)?;
                routes.push(json!({"step":steps.len(),"snapshot":snapshot}));
                w.control.agree(b"solo-routes-read-complete")?;
            }
            steps.push(json!({"step":step,"seconds_without_capture":step_seconds}));
        }
        let stop=w.history.pending()==Some(w.cfg.eos_token_id);
        let finish=w.execute(Command::Finish(if stop{Finish::Stop}else{Finish::Length}))?;
        ensure!(w.history.frontier()==prompt.len()+ids.len()-1&&w.history.pending()==ids.last().copied()
            &&ids.len()==w.emitted&&(stop||w.emitted==cap),"solo economics committed frontier");
        let record=json!({"case":name,"phase":phase,"seed":seed,"cap":cap,"prompt_tokens":prompt.len(),
            "begin":begin,"finish":finish,"prices":prices,"steps":steps,"routes":routes,
            "output_ids":ids,"frontier_checked":true});
        std::fs::write(out.join(format!("solo-economics-case{ordinal}.json")),serde_json::to_vec_pretty(&record)?)?;
        records.push(record);
        eprintln!("solo economics rank{rank} {name} complete");
    }
    w.execute(Command::Shutdown)?;
    std::fs::write(out.join("solo-economics-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"startup":startup,"records":records,
        "scope":"Three public phases. Within-prefix M2/M3/M6 local-target pricing and at most24 consecutive neural-route captures per task; no proposal/sampling policy change. Captures perturb caches and are excluded from step times, so these are locality/economics diagnostics, not serving performance or completed-task quality."}))?)?;
    Ok(())
}
