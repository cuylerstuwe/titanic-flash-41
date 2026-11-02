//! Finite native-generation confirmation of the preselected0.25cutoff.
//! Keep target precision, sampling, and ordinary history proposals unchanged.
use super::*;

pub(super) fn enabled()->bool {
    std::env::var("CANADA_SCRATCH_CONFIDENCE_POLICY_SCREEN").as_deref()==Ok("1")
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path,startup:Value)->Result<()> {
    ensure!(enabled()&&!confidence_economics_probe::enabled()
        &&w.current.is_none()&&w.history.frontier()==0&&w.session_cache_budget==0
        &&!w.prompt_checkpoint&&!w.prompt_cache_logits&&w.history_lookup_enabled
        &&crate::compressed_source::source20_fp32_reduction_enabled()?
        &&crate::layer_attention::generation_swa_bf16_enabled()?,
        "confidence policy isolated canonical-precision generation");
    let configured=crate::dspark_generation::serving_causal_three_threshold()?;
    ensure!(configured==0.25,"confirmation package explicitly selects0.25");
    let transfer:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/confidence-transfer-v1.json"))?;
    let economics:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/confidence-economics-v1.json"))?;
    let matched:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/matched-paths-v1.json"))?;
    let long=matched["cases"].as_array().unwrap().iter()
        .find(|c|c["input_ids"].as_array().is_some_and(|a|a.len()>120000))
        .context("long-context policy case")?;
    let cases=[&transfer["cases"][0],&economics["cases"][0],&economics["cases"][2],long];
    let mut records=Vec::new();let mut id=1u64;
    for (ordinal,case) in cases.into_iter().enumerate() {
        let prompt:Vec<u32>=serde_json::from_value(case["input_ids"].clone())?;
        ensure!((128..=131072).contains(&prompt.len()),"policy prompt extent");
        let arms:Vec<(bool,f32)>=if ordinal==0 {vec![(true,0.25),(false,0.20),(false,0.25)]}
            else if ordinal%2==1 {vec![(false,0.25),(false,0.20)]}
            else {vec![(false,0.20),(false,0.25)]};
        for (warmup,threshold) in arms {
            let cap=if warmup{48}else{512};
            let seed=202611013700+ordinal as u64+if warmup{100}else{0};
            let begin=w.execute(Command::Begin{images:Vec::new(),id,seed,greedy:false,
                limit:cap,tokens:prompt.clone()})?;id+=1;
            ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.
                &&begin["block_verification"]==true&&begin["generation_fp32_attention"]==true
                &&begin["causal_verify_threshold"]==0.25,
                "policy canonical target law and configured cutoff");
            let session=w.session.as_mut().unwrap();
            ensure!(session.draft_temperature()==0.7,"policy actual-q temperature");
            session.set_causal_three_threshold_for_probe(Some(threshold))?;
            let mut ids:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;
            let mut steps=Vec::new();w.control.agree(b"confidence-policy-start")?;
            let started=Instant::now();
            while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
                let step=w.execute(Command::Step)?;
                ids.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);
                steps.push(step);
            }
            let seconds=started.elapsed().as_secs_f64();
            w.session.as_mut().unwrap().set_causal_three_threshold_for_probe(Some(configured))?;
            let stop=w.history.pending()==Some(w.cfg.eos_token_id);
            let finish=w.execute(Command::Finish(if stop{Finish::Stop}else{Finish::Length}))?;
            ensure!(w.history.frontier()==prompt.len()+ids.len()-1
                &&w.history.pending()==ids.last().copied()&&ids.len()==w.emitted
                &&(stop||w.emitted==cap),"policy committed frontier");
            let record=json!({"case":case["id"],"family":case["family"],
                "initial_phase":case.get("phase").cloned().unwrap_or(json!("reasoning")),
                "warmup":warmup,"threshold":threshold,"cap":cap,"seed":seed,
                "prompt_tokens":prompt.len(),"begin":begin,"finish":finish,"steps":steps,
                "output_ids":ids,"seconds":seconds,"policy_restored":true,"frontier_checked":true});
            std::fs::write(out.join(format!("confidence-policy-c{ordinal}-t{threshold}-warm{warmup}.json")),
                serde_json::to_vec_pretty(&record)?)?;
            records.push(record);
            eprintln!("confidence policy rank{rank} case{ordinal} threshold{threshold} warmup{warmup}");
        }
    }
    w.execute(Command::Shutdown)?;
    std::fs::write(out.join("confidence-policy-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"startup":startup,"records":records,"configured_cutoff":configured,
        "scope":"Four public tasks, one512-token continuation per policy per task, plus excluded48-token warmup. Same model allocations and2350MHz cap, alternating policy order. Actual sampled native generation atT1/P1,draftT.7,both target precision improvements and normal history proposals retained. Finite state/policy integration check and descriptive timings, not a precise estimate of a1percent gain or completed-task quality."}))?)?;
    Ok(())
}
