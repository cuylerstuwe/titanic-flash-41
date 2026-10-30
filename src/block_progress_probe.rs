//! Small foreground intervention; no cache/background test expansion.
use super::*;
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path,startup:Value)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_BLOCK_PROGRESS_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none()&&w.history.frontier()==0&&w.session_cache_budget==0
        &&!w.prompt_checkpoint&&!w.prompt_cache_logits,"block progress finite foreground admission");
    let raw=include_bytes!("../tests/fixtures/root-feedback-exploration-tokens-v1.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    ensure!(fixture["thinking_mode"]=="thinking"&&fixture["reasoning_effort"]=="high","block progress template");
    let mut records=Vec::new();let mut id=1u64;let mut exact_control:Option<(Vec<u32>,Value)>=None;
    let mut progress=std::fs::OpenOptions::new().create_new(true).write(true).open(out.join("block-progress-records.jsonl"))?;
    for (ordinal,name)in ["go-coalescing-cancellation","library-workshop-tradeoffs"].into_iter().enumerate() {
        let case=fixture["cases"].as_array().unwrap().iter().find(|c|c["id"]==name).unwrap();
        let prompt:Vec<u32>=serde_json::from_value(case["input_ids"].clone())?;
        // Warm once, then telemetry off/on must preserve identical token/state
        // results. The two capped policy pairs use new seeds and reversed order.
        let arms:Vec<(bool,bool,bool,usize)>=if ordinal==0 {
            vec![(false,false,true,64),(false,false,false,96),(true,false,false,96),
                 (false,false,false,256),(true,true,false,256)]
        }else{vec![(true,true,false,256),(false,false,false,256)]};
        for (telemetry,adaptive,warmup,cap)in arms {
            let seed=202683500+ordinal as u64+if cap==256{100}else{0};
            let begin=w.execute(Command::Begin{images: Vec::new(),id,seed,greedy:false,limit:cap,tokens:prompt.clone()})?;id+=1;
            ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.
                &&begin["generation_fp32_attention"]==true&&begin["block_verification"]==true,
                "block progress target law fixed");
            w.session.as_mut().unwrap().set_block_progress_for_probe(telemetry,telemetry&&!adaptive,&w.stream)?;
            w.session.as_mut().unwrap().set_block_rows_for_probe(adaptive.then_some(6))?;
            let mut ids:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;
            let mut steps=Vec::new();let mut observations=Vec::new();
            let initial=if cap==96 {Some(json!({"carry":w.verify.carry_digests()?,"encoder":w.verify.encoder_tail_digest()?,
                "policy":w.session.as_ref().unwrap().causal_verify_threshold(),"anchor":w.history.pending()}))}else{None};
            w.control.agree(b"block-progress-measure")?;let started=Instant::now();
            while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
                let at=Instant::now();let s=w.execute(Command::Step)?;let seconds=at.elapsed().as_secs_f64();
                ids.extend(serde_json::from_value::<Vec<u32>>(s["output_ids"].clone())?);
                let rows=s["verified_rows"].as_u64().unwrap()as usize;
                if telemetry&&s["proposal_source"]=="dspark"&&[3,6].contains(&rows) {
                    let (next,mut obs)=w.session.as_mut().unwrap().observe_block_progress_for_probe(rows,seconds,&w.stream)?;
                    if adaptive {
                        let byte=[next as u8];let wire=w.control.command((rank==0).then_some(byte.as_slice()))?;
                        ensure!(wire.len()==1&&[3,6].contains(&wire[0]),"block progress choice wire");
                        w.session.as_mut().unwrap().set_block_rows_for_probe(Some(wire[0]as usize))?;
                        obs["rank0_next_rows"]=json!(wire[0]);
                    }
                    obs["position"]=s["frontier"].clone();observations.push(obs);
                }
                steps.push(s);
            }
            let seconds=started.elapsed().as_secs_f64();
            w.session.as_mut().unwrap().set_block_progress_for_probe(false,false,&w.stream)?;
            w.session.as_mut().unwrap().set_block_rows_for_probe(None)?;
            let stop=w.history.pending()==Some(w.cfg.eos_token_id);
            let finish=w.execute(Command::Finish(if stop{Finish::Stop}else{Finish::Length}))?;
            ensure!(w.history.frontier()==prompt.len()+ids.len()-1&&w.history.pending()==ids.last().copied()
                &&ids.len()==w.emitted&&(stop||w.emitted==cap),"block progress committed frontier");
            let state=json!({"carry":w.verify.carry_digests()?,"encoder":w.verify.encoder_tail_digest()?,
                "outputs":w.verify.valid_output_digests()?,"position":w.verify.position()});
            let parity=if cap==96 {
                let exact=(ids.clone(),state.clone());
                if telemetry {let control=exact_control.as_ref().unwrap();Some(json!({"tokens_equal":control.0==ids,
                    "state_equal":control.1==state,"control_state":control.1}))}
                else {exact_control=Some(exact);None}
            }else{None};
            let r=json!({"case":name,"family":case["family"],"telemetry":telemetry,"adaptive":adaptive,
                "warmup":warmup,"cap":cap,"seed":seed,"prompt_tokens":prompt.len(),"begin":begin,"finish":finish,
                "output_ids":ids,"state":state,"decode_seconds":seconds,"steps":steps,"observations":observations,
                "initial":initial,"parity":parity,"sampler_decisions_mirrored":telemetry&&!adaptive});
            writeln!(progress,"{}",r)?;progress.flush()?;records.push(r);
            if let Some(p)=&parity {ensure!(p["tokens_equal"]==true&&p["state_equal"]==true,"fused progress changed tokens/state; complete diagnostic record saved");}
            eprintln!("block-progress rank{rank} {name} cap{cap} adaptive{adaptive}");
        }
    }
    w.execute(Command::Shutdown)?;
    std::fs::write(out.join("block-progress-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,"startup":startup,
        "records":records,"telemetry_pair_exact":true,"all_frontiers_checked":true,
        "scope":"Two public high-reasoning fronts; one96-token telemetry parity pair then two256-token policy pairs in reversed order. Canonical target T1/P1, FP32 generation attention. Fused conditional expected accepted progress updates NEXT-round 3/6 preference using past measured cost, EMA.25/hysteresis3percent and periodic exploration. All controller probability arithmetic GPU. Foreground only; caps are not completed-task quality or production qualification."}))?)?;
    Ok(())
}
