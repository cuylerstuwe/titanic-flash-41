//! Finite opportunity screen: causal confidence versus measured 3/6 economics.
//! All probability-derived expectations use the existing checked GPU recurrence.
//! Fixed width choices precede proposals; no fitted serving policy is enabled.
use super::*;

pub(super) fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CANADA_SCRATCH_CONFIDENCE_ECONOMICS_SCREEN").as_deref()==Ok("1"))
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path,startup:Value)->Result<()> {
    ensure!(enabled()
        &&std::env::var("CANADA_SCRATCH_BLOCK_PROGRESS_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none()&&w.history.frontier()==0&&w.session_cache_budget==0
        &&!w.prompt_checkpoint&&!w.prompt_cache_logits&&w.history_lookup_enabled,
        "confidence economics finite foreground admission");
    let transfer=std::env::var("CANADA_SCRATCH_CONFIDENCE_TRANSFER_SCREEN").as_deref()==Ok("1");
    let raw:&[u8]=if transfer {include_bytes!("../tests/fixtures/confidence-transfer-v1.json")}
        else {include_bytes!("../tests/fixtures/confidence-economics-v1.json")};
    let fixture:Value=serde_json::from_slice(raw)?;
    ensure!(fixture["schema"]==if transfer{"confidence-transfer-v1"}else{"confidence-economics-v1"},
        "confidence economics fixture");
    let cases=fixture["cases"].as_array().unwrap();
    ensure!(cases.len()==if transfer{2}else{3},"bounded confidence economics cases");
    let mut records=Vec::new();let mut id=1u64;
    for (ordinal,case) in cases.iter().enumerate() {
        let prompt:Vec<u32>=serde_json::from_value(case["input_ids"].clone())?;
        ensure!((128..=65536).contains(&prompt.len()),"confidence economics prompt extent");
        // The first short request warms both verification widths and the
        // telemetry path. It is excluded from economics and inference claims.
        for warmup in if ordinal==0 {vec![true,false]}else{vec![false]} {
            let cap=if warmup {48}else{384};
            let seed=(if transfer{202611013600}else{2026110900})+ordinal as u64+if warmup{100}else{0};
            let begin=w.execute(Command::Begin{images:Vec::new(),id,seed,greedy:false,
                limit:cap,tokens:prompt.clone()})?;id+=1;
            ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.
                &&begin["block_verification"]==true&&begin["generation_fp32_attention"]==true,
                "confidence economics canonical target law");
            let session=w.session.as_mut().unwrap();
            ensure!(session.draft_temperature()==0.7,"confidence economics fixed actual-q temperature");
            session.configure_history_policy_for_probe(None)?;
            session.set_block_progress_for_probe(true,false,&w.stream)?;
            let mut ids:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;
            let mut steps=Vec::new();let mut observations=Vec::new();let mut round=0usize;
            w.control.agree(b"confidence-economics-start")?;
            let started=Instant::now();
            while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
                // One in eight short steps supplies real skip-draft/verify/
                // commit cost. Rotate its position across tasks. The schedule
                // is independent of current/future proposals and target draws.
                let rows=if (round+ordinal)%8==0 {3}else{6};
                w.session.as_mut().unwrap().set_block_rows_for_probe(Some(rows))?;
                let at=Instant::now();let step=w.execute(Command::Step)?;
                let seconds=at.elapsed().as_secs_f64();
                ids.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);
                let live=step["verified_rows"].as_u64().unwrap()as usize;
                ensure!(step["proposal_source"]=="dspark"&&live<=rows,"confidence economics isolated proposals");
                if [3,6].contains(&live) {
                    let (_ignored_choice,mut observation)=w.session.as_mut().unwrap()
                        .observe_block_progress_for_probe(live,seconds,&w.stream)?;
                    observation["confidence"]=step["draft_confidence"].clone();
                    observation["frontier"]=step["frontier"].clone();
                    observation["phase_ms"]=step["phase_ms"].clone();
                    observation["round"]=json!(round);
                    observations.push(observation);
                }
                steps.push(step);round+=1;
            }
            let seconds=started.elapsed().as_secs_f64();
            w.session.as_mut().unwrap().set_block_progress_for_probe(false,false,&w.stream)?;
            w.session.as_mut().unwrap().set_block_rows_for_probe(None)?;
            let stop=w.history.pending()==Some(w.cfg.eos_token_id);
            let finish=w.execute(Command::Finish(if stop{Finish::Stop}else{Finish::Length}))?;
            ensure!(w.history.frontier()==prompt.len()+ids.len()-1
                &&w.history.pending()==ids.last().copied()&&ids.len()==w.emitted
                &&(stop||w.emitted==cap),"confidence economics committed frontier");
            let record=json!({"case":case["id"],"family":case["family"],"initial_phase":case["phase"],
                "fresh_task":case["fresh_task"],"warmup":warmup,"cap":cap,"seed":seed,
                "prompt_tokens":prompt.len(),"begin":begin,"finish":finish,"steps":steps,
                "observations":observations,"output_ids":ids,"seconds":seconds,
                "policy_restored":true,"frontier_checked":true});
            std::fs::write(out.join(format!("confidence-economics-case{ordinal}-warm{warmup}.json")),
                serde_json::to_vec_pretty(&record)?)?;
            records.push(record);
            eprintln!("confidence economics rank{rank} case{ordinal} warmup{warmup}");
        }
    }
    w.execute(Command::Shutdown)?;
    std::fs::write(out.join("confidence-economics-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"startup":startup,"records":records,"transfer_screen":transfer,
        "fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "fixture_scope":fixture["scope"],
        "scope":"Finite public frontier panel with384-token caps and one48-token excluded warmup. Fixed predetermined3/6 schedule, no history copies, actual draftT.7 and targetT1/P1. Existing fused GPU conditional expected progress plus causal confidence and complete step costs. Estimates integrate verifier uniforms conditional on each sampled proposal path, not future-policy throughput or completed quality. No new serving policy or background matrix."}))?)?;
    Ok(())
}
