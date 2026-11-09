//! Bounded short/medium retained-append anatomy (agentic-turn-shaped sizes).
//! Diagnostic only: the serving planner chooses every chunk (all wide profiles are
//! offered, as in serving); no dispatch, planner, precision or cache change.
use super::*;
use super::history32_generation_probe::{execute,state,restore};

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&std::env::var("CANADA_SCRATCH_CLAUDE_SMALL_PROFILE").as_deref()==Ok("1"),
        "finite Claude short-append profile");
    let prefix:usize=std::env::var("CANADA_SCRATCH_CLAUDE_SMALL_PREFIX").ok().map(|v|v.parse()).transpose()?.unwrap_or(200_000);
    let sizes:Vec<usize>=std::env::var("CANADA_SCRATCH_CLAUDE_SMALL_SIZES").unwrap_or("16,27,48,75,110,160,254,385,548,970".into())
        .split(',').map(|v|v.parse::<usize>().map_err(anyhow::Error::from)).collect::<Result<_>>()?;
    let profiled:Vec<usize>=std::env::var("CANADA_SCRATCH_CLAUDE_SMALL_PROFILED").unwrap_or("75,548".into())
        .split(',').filter(|v|!v.is_empty()).map(|v|v.parse::<usize>().map_err(anyhow::Error::from)).collect::<Result<_>>()?;
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let corpus:Value=serde_json::from_slice(raw)?;
    let ids:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    ensure!(ids.len()>=prefix+sizes.iter().max().copied().unwrap_or(0)+8,"short-append fixture too short");
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let cold=Instant::now();
    let begin=execute(w,Command::Begin{images:Vec::new(),id:8_081_301,seed:2026110870,
        greedy:false,limit:1,tokens:ids[..prefix].to_vec()})?;
    let cold_ms=cold.elapsed().as_secs_f64()*1000.;
    let anchor=begin["output_ids"][0].as_u64().context("short-append anchor")?as u32;
    let original=state(w)?;
    let(bytes,cpu)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
    ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
        >=8*1024usize.pow(3)+bytes+cpu)?,"short-append snapshot reserve");
    let saved=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
    let mut records=Vec::new();
    for &live in &sizes {
        let input=&ids[prefix..prefix+live];
        for trial in 0..4usize {
            restore(w,&saved,anchor,&original)?;
            w.large.handoff_from(&w.verify)?;
            w.control.agree(&[0xc5,(live&0xff)as u8,(live>>8)as u8,trial as u8])?;
            // Trial 2 of the selected sizes is the captured range; trials 1-3 are warm timings.
            let traced=trial==2&&profiled.contains(&live);
            if traced {w.stream.synchronize()?;w.cuda.profiler_start()?;w.control.agree(&[0xc6,trial as u8])?;}
            let at=Instant::now();
            let result=w.large.prefill_wide(&mut w.wides,input,&w.comm,&w.stream);
            w.stream.synchronize()?;
            let milliseconds=at.elapsed().as_secs_f64()*1000.;
            if traced {w.cuda.profiler_stop()?;}
            result?;
            let phases=serde_json::to_value(w.large.append_metrics()?)?;
            let values=w.large.last_prompt_logits()?;
            ensure!(values.iter().all(|v|v.is_finite()),"nonfinite short-append output");
            w.session.as_mut().unwrap().resume_after_extend(&w.large,prefix as u64,input,&w.stream)?;
            w.verify.handoff_from(&w.large)?;
            records.push(json!({"live":live,"trial":trial,"traced":traced,"milliseconds":milliseconds,"phases":phases}));
            std::fs::write(out.join("claude-small-partial.json"),serde_json::to_vec_pretty(&records)?)?;
            eprintln!("claude small rank{rank} live{live} trial{trial}: {milliseconds:.2} ms phases={}",
                phases["stages_ms"]);
        }
    }
    restore(w,&saved,anchor,&original)?;drop(saved);
    execute(w,Command::Finish(Finish::Length))?;execute(w,Command::Shutdown)?;
    std::fs::write(out.join("claude-small-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"prefix_tokens":prefix,"cold_begin_ms":cold_ms,"records":records,
        "scope":"Warm retained appends of agentic-turn-shaped sizes after a public long prefix; the serving planner chooses chunks from all wide profiles; trial 2 of selected sizes is a captured range. Diagnostic only."
    }))?)?;
    Ok(())
}
