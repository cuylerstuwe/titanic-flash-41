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
    // Route census (Nov 11, CANADA_SCRATCH_ROUTE_CENSUS=1): which experts every layer's router selected in each
    // step, read back between steps. Aggregates only: how many distinct experts a step reads (the bytes a step
    // must pull from memory), and how well the previous step's selection or a fixed popular set predicts it
    // (whether expert weights could be hinted before the router has run). Timings of a census run are not used.
    let census=std::env::var("CANADA_SCRATCH_ROUTE_CENSUS").as_deref()==Ok("1");
    // CANADA_SCRATCH_CLAUDE_DECODE_PREFIXES=a,b,... (default 32768,131071; at most six): the retained contexts.
    // A context longer than the fixture (262,656 tokens) repeats it; step times by rows are what such a case is for.
    let prefixes:Vec<usize>=match std::env::var("CANADA_SCRATCH_CLAUDE_DECODE_PREFIXES") {
        Ok(v)=>v.split(',').map(|x|x.parse::<usize>().context("decode profile prefix")).collect::<Result<_>>()?,
        Err(std::env::VarError::NotPresent)=>vec![32768,131071],
        Err(e)=>return Err(e.into()),
    };
    ensure!((1..=6).contains(&prefixes.len())&&prefixes.iter().all(|p|(1024..=1_040_000).contains(p)),"decode profile prefixes");
    // CANADA_SCRATCH_CLAUDE_DECODE_ARMS=a,b,... with each arm two digits <draft pre-launch><sampler ahead> (for
    // example 00,10,01,11): the arms take turns in blocks of eight steps within every case, replacing the two
    // environment flags, so their step times compare within one load on the same text. Every step records its arm.
    let arms:Vec<(bool,bool)>=match std::env::var("CANADA_SCRATCH_CLAUDE_DECODE_ARMS") {
        Ok(v)=>v.split(',').map(|x|{ensure!(x.len()==2&&x.bytes().all(|b|b==b'0'||b==b'1'),"decode profile arm");
            Ok((x.as_bytes()[0]==b'1',x.as_bytes()[1]==b'1'))}).collect::<Result<_>>()?,
        Err(std::env::VarError::NotPresent)=>Vec::new(),
        Err(e)=>return Err(e.into()),
    };
    ensure!(arms.len()<=4,"decode profile arms");
    let mut cases=Vec::new();
    for (case,prefix) in prefixes.iter().copied().enumerate().map(|(i,p)|(i as u8,p)) {
        let mut previous:Vec<Vec<i32>>=Vec::new();
        let mut frequency=vec![[0u32;384];40];
        // by verified rows 1..=6: steps, distinct experts, with the previous step: shared / previous size,
        // covered by the layer's 16 and 32 most frequent experts so far
        let mut tally=[[0f64;6];7];
        let mut by_layer=vec![[0f64;3];40];
        ensure!(ids.len()>1024,"decode fixture too short");
        let cold=Instant::now();
        let begin=execute(w,Command::Begin{images:Vec::new(),id:8_081_200+case as u64,
            seed:2026110860+case as u64,greedy:false,limit:8192,tokens:(0..prefix).map(|i|ids[i%ids.len()]).collect()})?;
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
            let arm=(!arms.is_empty()).then(||(step/8)%arms.len());
            if let Some(arm)=arm {crate::dspark_generation::set_step_arm_for_probe(Some(arms[arm]));}
            let at=Instant::now();
            let event=execute(w,Command::Step)?;
            let wall_ms=at.elapsed().as_secs_f64()*1000.;
            if step==55 {w.stream.synchronize()?;w.cuda.profiler_stop()?;}
            let produced=event["output_ids"].as_array().map(|a|a.len()).unwrap_or(0);
            emitted+=produced;
            // A step without a neural six-row verification (history proposals, target-only) has no routes to count.
            let live=event["verified_rows"].as_u64().unwrap_or(0)as usize;
            if census&&step>=8&&(1..=6).contains(&live) {
                let owners=w.verify.layer_owners();
                ensure!(owners.len()==40,"route census geometry");
                let mut now=Vec::with_capacity(40);
                for owner in &owners {
                    let ids=owner.ffn().router().ids().download()?;
                    ensure!(ids.len()>=live*6,"route census ids");
                    let mut set:Vec<i32>=ids[..live*6].iter().copied().filter(|id|(0..384).contains(id)).collect();
                    set.sort_unstable();set.dedup();now.push(set);
                }
                for (layer,set) in now.iter().enumerate() {
                    let mut ranked:Vec<usize>=(0..384).collect();
                    ranked.sort_by_key(|&e|std::cmp::Reverse(frequency[layer][e]));
                    let covered=|k:usize|set.iter().filter(|&&e|ranked[..k].contains(&(e as usize))).count()as f64;
                    let t=&mut tally[live];
                    t[0]+=1.;t[1]+=set.len()as f64;
                    if let Some(old)=previous.get(layer) {
                        t[2]+=set.iter().filter(|e|old.contains(e)).count()as f64;t[3]+=old.len()as f64;
                        by_layer[layer][1]+=set.iter().filter(|e|old.contains(e)).count()as f64/set.len().max(1)as f64;
                        by_layer[layer][2]+=1.;
                    }
                    t[4]+=covered(16);t[5]+=covered(32);
                    by_layer[layer][0]+=set.len()as f64;
                    for &e in set {frequency[layer][e as usize]+=1;}
                }
                previous=now;
            }
            steps.push(json!({"step":step,"profiled":(48..56).contains(&step),"wall_ms":wall_ms,"arm":arm,
                "phase_ms":event["phase_ms"],"verified_rows":event["verified_rows"],
                "committed_model_rows":event["committed_model_rows"],
                "accepted_draft_tokens":event["accepted_draft_tokens"],"emitted_tokens":produced,
                "draft_confidence":event["draft_confidence"],
                "proposal_source":event["proposal_source"],"lookup_match_tokens":event["lookup_match_tokens"],
                "position":w.verify.position()}));
        }
        crate::dspark_generation::set_step_arm_for_probe(None);
        // A bounded early stop is a cancellation; Stop/Length are reserved for EOS and the limit.
        execute(w,Command::Finish(Finish::Cancelled))?;
        let unprofiled:Vec<f64>=steps.iter().filter(|s|s["profiled"]==false&&s["step"].as_u64().unwrap()>=8)
            .map(|s|s["wall_ms"].as_f64().unwrap()).collect();
        let rows:Vec<u64>=steps.iter().filter(|s|s["step"].as_u64().unwrap()>=8)
            .map(|s|s["committed_model_rows"].as_u64().unwrap_or(0)).collect();
        let total_ms:f64=unprofiled.iter().sum();
        eprintln!("claude decode rank{rank} case{case} prefix{prefix}: {} steps, mean {:.2} ms, {:.2} committed rows/step",
            steps.len(),total_ms/unprofiled.len().max(1) as f64,rows.iter().sum::<u64>() as f64/rows.len().max(1) as f64);
        let route_census=if census {
            let rows:Vec<Value>=(1..=6).filter(|&live|tally[live][0]>0.).map(|live|{let t=tally[live];let layers=t[0];json!({
                "verified_rows":live,"layer_steps":layers,"selected_pairs":live*6,"mean_distinct_experts":t[1]/layers,
                "shared_with_previous_step_per_distinct":if t[1]>0.{t[2]/t[1]}else{0.},
                "previous_step_set_size_mean":t[3]/layers,
                "covered_by_16_most_frequent":t[4]/t[1].max(1.),"covered_by_32_most_frequent":t[5]/t[1].max(1.)})}).collect();
            let steps_counted=by_layer[0][2].max(1.);
            json!({"by_verified_rows":rows,
                "mean_distinct_by_layer":by_layer.iter().map(|l|((l[0]/(steps_counted+1.))*100.).round()/100.).collect::<Vec<_>>(),
                "mean_share_seen_in_previous_step_by_layer":by_layer.iter().map(|l|((l[1]/steps_counted)*1000.).round()/1000.).collect::<Vec<_>>(),
                "scope":"Rank-local router ids of the six-row verifier after each step from step 8. Distinct = different experts among verified rows x 6. Frequent sets are ranked on the steps seen so far in this case."})
        }else{Value::Null};
        cases.push(json!({"case":case,"prefix_tokens":prefix,"cold_begin_ms":cold_ms,"emitted":emitted,"steps":steps,
            "route_census":route_census}));
        std::fs::write(out.join("claude-decode-partial.json"),serde_json::to_vec_pretty(&cases)?)?;
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("claude-decode-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"cases":cases,
        "prefixes":prefixes,
        "scope":"Sampled DSpark decode after public prefixes of the listed lengths (default 32768 and 131071); steps 48-55 of each case form one captured range. Diagnostic only; no serving or precision change."
    }))?)?;
    Ok(())
}
