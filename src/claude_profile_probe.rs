//! Bounded prefill cost profile: warm retained appends at four physical widths.
//! Diagnostic only. No serving dispatch, planner, precision or cache change.
use super::*;
use super::history32_generation_probe::{execute,state,restore};

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&std::env::var("CANADA_SCRATCH_CLAUDE_PREFILL_PROFILE").as_deref()==Ok("1"),
        "finite Claude prefill profile");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let corpus:Value=serde_json::from_slice(raw)?;
    let ids:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    let prefix=131071;
    ensure!(ids.len()>=prefix+8192+8,"profile fixture too short");
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let cold=Instant::now();
    let begin=execute(w,Command::Begin{images:Vec::new(),id:8_081_108,seed:2026110850,
        greedy:false,limit:1,tokens:ids[..prefix].to_vec()})?;
    let cold_ms=cold.elapsed().as_secs_f64()*1000.;
    let anchor=begin["output_ids"][0].as_u64().context("profile anchor")?as u32;
    let original=state(w)?;
    let(bytes,cpu)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
    ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
        >=8*1024usize.pow(3)+bytes+cpu)?,"profile snapshot reserve");
    let saved=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
    let mut records=Vec::new();
    for (width,live) in [(512usize,385usize),(1024,1023),(2048,1535),(8192,8192)] {
        let input=&ids[prefix..prefix+live];
        for trial in 0..4usize {
            restore(w,&saved,anchor,&original)?;
            w.large.handoff_from(&w.verify)?;
            let index=w.wides.iter().position(|t|t.physical_rows()==width).context("profile width absent")?;
            w.control.agree(&[0xc1,(width/128)as u8,trial as u8])?;
            // Trials 0-2 are unprofiled timings; trial 3 is the one captured range.
            let profiled=trial==3;
            if profiled {w.stream.synchronize()?;w.cuda.profiler_start()?;}
            let at=Instant::now();
            let result=w.large.prefill_wide(&mut w.wides[index..index+1],input,&w.comm,&w.stream);
            let milliseconds=at.elapsed().as_secs_f64()*1000.;
            if profiled {w.cuda.profiler_stop()?;}
            result?;
            let phases=serde_json::to_value(w.large.append_metrics()?)?;
            let values=w.large.last_prompt_logits()?;
            ensure!(values.iter().all(|v|v.is_finite()),"nonfinite profile output");
            // Return ownership to the session/verify owner exactly as the prefill-bounds screen does,
            // so the next full-state restore sees the expected owner and frontier.
            w.session.as_mut().unwrap().resume_after_extend(&w.large,prefix as u64,input,&w.stream)?;
            w.verify.handoff_from(&w.large)?;
            ensure!(phases["physical_rows"]==width,"profile did not use selected width");
            records.push(json!({"physical_rows":width,"live":live,"trial":trial,"profiled":profiled,
                "milliseconds":milliseconds,"phases":phases}));
            std::fs::write(out.join("claude-profile-partial.json"),serde_json::to_vec_pretty(&records)?)?;
            eprintln!("claude profile rank{rank} M{width} live{live} trial{trial}: {milliseconds:.2} ms");
        }
    }
    restore(w,&saved,anchor,&original)?;drop(saved);
    execute(w,Command::Finish(Finish::Length))?;execute(w,Command::Shutdown)?;
    std::fs::write(out.join("claude-profile-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"prefix_tokens":prefix,"cold_begin_ms":cold_ms,"records":records,
        "scope":"Warm retained appends after a 131071-token public prefix at M512/M1024/M2048/M8192; trial 3 of each width is the profiled range. Diagnostic only; no serving or precision change."
    }))?)?;
    Ok(())
}
