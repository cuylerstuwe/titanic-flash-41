//! Bounded decode-step anatomy: drafting, verification, commit and host gaps
//! at retained contexts of 32768 and 131071 public tokens.
//! Diagnostic only. No serving dispatch, planner, precision or cache change.
use super::*;
use super::history32_generation_probe::execute;

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&std::env::var("CANADA_SCRATCH_CLAUDE_DECODE_PROFILE").as_deref()==Ok("1"),
        "finite Claude decode profile");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let corpus:Value=serde_json::from_slice(raw)?;
    let ids:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    let steps_per_case:usize=std::env::var("CANADA_SCRATCH_CLAUDE_DECODE_STEPS").ok()
        .and_then(|v|v.parse().ok()).unwrap_or(160);
    ensure!((64..=1024).contains(&steps_per_case),"decode profile step bound");
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let mut cases=Vec::new();
    for (case,prefix) in [(0u8,32768usize),(1,131071)] {
        ensure!(ids.len()>prefix,"decode fixture too short");
        let cold=Instant::now();
        let begin=execute(w,Command::Begin{images:Vec::new(),id:8_081_200+case as u64,
            seed:2026110860+case as u64,greedy:false,limit:8192,tokens:ids[..prefix].to_vec()})?;
        let cold_ms=cold.elapsed().as_secs_f64()*1000.;
        let mut steps=Vec::new();
        let mut emitted=begin["output_ids"].as_array().map(|a|a.len()).unwrap_or(0);
        // Steps 48..56 form one captured range: eight consecutive warm steps,
        // including host work between steps. All other steps are unprofiled.
        for step in 0..steps_per_case {
            // An EOS or the request limit ends the case; the probe never steps past it.
            if w.history.step_limit()?==0 {break;}
            if step==48 {
                w.stream.synchronize()?;w.cuda.profiler_start()?;w.control.agree(&[0xd1,case])?;
            }
            let at=Instant::now();
            let event=execute(w,Command::Step)?;
            let wall_ms=at.elapsed().as_secs_f64()*1000.;
            if step==55 {w.stream.synchronize()?;w.cuda.profiler_stop()?;}
            let produced=event["output_ids"].as_array().map(|a|a.len()).unwrap_or(0);
            emitted+=produced;
            steps.push(json!({"step":step,"profiled":(48..56).contains(&step),"wall_ms":wall_ms,
                "phase_ms":event["phase_ms"],"verified_rows":event["verified_rows"],
                "committed_model_rows":event["committed_model_rows"],
                "accepted_draft_tokens":event["accepted_draft_tokens"],"emitted_tokens":produced,
                "proposal_source":event["proposal_source"],"lookup_match_tokens":event["lookup_match_tokens"],
                "position":w.verify.position()}));
        }
        // A bounded early stop is a cancellation; Stop/Length are reserved for EOS and the limit.
        execute(w,Command::Finish(Finish::Cancelled))?;
        let unprofiled:Vec<f64>=steps.iter().filter(|s|s["profiled"]==false&&s["step"].as_u64().unwrap()>=8)
            .map(|s|s["wall_ms"].as_f64().unwrap()).collect();
        let rows:Vec<u64>=steps.iter().filter(|s|s["step"].as_u64().unwrap()>=8)
            .map(|s|s["committed_model_rows"].as_u64().unwrap_or(0)).collect();
        let total_ms:f64=unprofiled.iter().sum();
        eprintln!("claude decode rank{rank} case{case} prefix{prefix}: {} steps, mean {:.2} ms, {:.2} committed rows/step",
            steps.len(),total_ms/unprofiled.len().max(1) as f64,rows.iter().sum::<u64>() as f64/rows.len().max(1) as f64);
        cases.push(json!({"case":case,"prefix_tokens":prefix,"cold_begin_ms":cold_ms,"emitted":emitted,"steps":steps}));
        std::fs::write(out.join("claude-decode-partial.json"),serde_json::to_vec_pretty(&cases)?)?;
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("claude-decode-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"cases":cases,
        "scope":"Sampled DSpark decode after 32768- and 131071-token public prefixes; steps 48-55 of each case form one captured range. Diagnostic only; no serving or precision change."
    }))?)?;
    Ok(())
}
