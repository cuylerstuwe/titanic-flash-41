//! Fixed causal expert hints overlapping real model work, on one allocation.
//! The preceding neural step selects one expert per layer. CPU readback and
//! graph preparation are outside timing; a production selector is not priced.
use super::*;

fn state(target:&Target)->Result<Value> {
    Ok(json!({"position":target.position(),"carry":target.carry_digests()?,
        "encoder":target.encoder_tail_digest()?}))
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path,startup:Value)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_EXPERT_PREFETCH_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none()&&w.history.frontier()==0&&w.session_cache_budget==0
        &&!w.prompt_checkpoint&&!w.prompt_cache_logits&&w.verify.local_target_enabled(),
        "finite expert hint admission");
    let short:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/root-feedback-exploration-tokens-v1.json"))?;
    let phase:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json"))?;
    let long:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let js=short["cases"].as_array().unwrap().iter().find(|c|c["id"]=="javascript-incremental-ndjson").unwrap();
    let idea=phase["cases"].as_array().unwrap().iter().find(|c|c["id"]=="offline-product-ideation-answer-3").unwrap();
    let mut answer:Vec<u32>=serde_json::from_value(idea["input_ids"].clone())?;
    answer.push(idea["anchor"].as_u64().unwrap()as u32);
    let long_tokens:Vec<u32>=serde_json::from_value(long["tokens"].clone())?;
    let cases=[("javascript-ndjson",serde_json::from_value(js["input_ids"].clone())?),
        ("product-ideation",answer),("repository-131k",long_tokens[..131072].to_vec())];
    let mut records=Vec::new();
    for (ordinal,(name,prompt)) in cases.into_iter().enumerate() {
        let seed=202611034500+ordinal as u64;let cap=96;
        let begin=w.execute(Command::Begin{images:Vec::new(),id:9_245_001+ordinal as u64,
            seed,greedy:false,limit:cap,tokens:prompt.clone()})?;
        ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.
            &&begin["generation_fp32_attention"]==true&&begin["block_verification"]==true,
            "expert hint canonical sampled target");
        w.session.as_mut().unwrap().configure_history_policy_for_probe(None)?;
        let mut ids:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;
        let mut steps=Vec::new();
        for _ in 0..4 {
            ensure!(w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id),"expert hint warmup terminated");
            let step=w.execute(Command::Step)?;
            ids.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);steps.push(step);
        }
        let previous=steps.last().unwrap();
        ensure!(previous["proposal_source"]=="dspark","expert hint previous neural routes");
        let routes=w.verify.capture_consecutive_routes_for_probe(previous["verified_rows"].as_u64().unwrap()as usize,&w.stream)?;
        let mut selected=Vec::new();
        for layer in routes["layers"].as_array().context("previous layer routes")? {
            let mut counts=[0usize;384];
            for token in layer["ids"].as_array().unwrap(){counts[token.as_u64().unwrap()as usize]+=1;}
            let expert=(0..384).max_by_key(|&i|(counts[i],std::cmp::Reverse(i))).unwrap();
            ensure!(counts[expert]>0,"empty route prediction");selected.push(expert);
        }
        ensure!(selected.len()==40,"expert hint layer count");
        // Freeze the prediction before generating or evaluating the next path.
        let anchor=w.history.pending().context("expert hint anchor")?;
        let proposed=w.session.as_mut().unwrap().expert_prefetch_inputs_for_probe(&w.verify,anchor,
            crate::categorical::block_round_seed(seed,w.verify.position()),&w.comm,&w.stream)?;
        let mut changed=proposed;changed[1]=(changed[1]+1)%129280;
        let original=state(&w.verify)?;let mut trials=Vec::new();
        for live in [3,6] {
            let mut expected=None;
            let arms=if ordinal%2==0 {[false,true,true,false]}else{[true,false,false,true]};
            for (trial,enabled) in arms.into_iter().enumerate() {
                w.verify.set_expert_prefetch_for_probe(enabled.then_some(selected.as_slice()),&w.core,&w.stream)?;
                for _ in 0..3 {
                    w.verify.begin_local_verification(&proposed[..live],&w.comm,&w.stream)?;
                    w.verify.finish_verification(0,&w.comm,&w.stream)?;
                }
                let captures=w.verify.capture_statistics().1;let mut samples=Vec::new();
                for _ in 0..6 {
                    w.control.agree(b"expert-hint-measurement")?;
                    let start=Instant::now();
                    w.verify.begin_local_verification(&proposed[..live],&w.comm,&w.stream)?;
                    w.verify.finish_verification(0,&w.comm,&w.stream)?;
                    samples.push(start.elapsed().as_secs_f64()*1000.);
                }
                ensure!(captures==w.verify.capture_statistics().1,"expert hint timed graph recapture");
                let mut checks=Vec::new();
                for input in [&proposed[..live],&changed[..live]] {
                    w.verify.begin_local_verification(input,&w.comm,&w.stream)?;
                    let(logits,rows)=w.verify.pending_local_target_logits()?;
                    ensure!(rows==live,"expert hint local logit extent");
                    let values=logits.download()?;let values=&values[..rows*(129280/4)];
                    ensure!(values.iter().all(|v|v.is_finite()),"expert hint nonfinite target logits");
                    let bytes:Vec<u8>=values.iter().flat_map(|v|v.to_le_bytes()).collect();
                    checks.push(format!("{:x}",Sha256::digest(bytes)));
                    w.verify.finish_verification(0,&w.comm,&w.stream)?;
                    ensure!(state(&w.verify)?==original,"expert hint changed retained state");
                }
                ensure!(checks[0]!=checks[1],"expert hint changed input did not affect logits");
                if let Some(reference)=&expected {ensure!(reference==&checks,"expert hint changed canonical local logits");}
                else{expected=Some(checks.clone());}
                eprintln!("expert hint rank{rank} {name} live{live} trial{trial} enabled{enabled}: {:.4}ms",
                    samples.iter().sum::<f64>()/samples.len()as f64);
                trials.push(json!({"live":live,"trial":trial,"enabled":enabled,"samples_ms":samples,
                    "local_logit_sha256":checks,"all_logits_and_state_exact":true,"new_timed_captures":0}));
            }
        }
        w.verify.set_expert_prefetch_for_probe(None,&w.core,&w.stream)?;
        while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
            let step=w.execute(Command::Step)?;
            ids.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);steps.push(step);
        }
        let stop=w.history.pending()==Some(w.cfg.eos_token_id);
        let finish=w.execute(Command::Finish(if stop{Finish::Stop}else{Finish::Length}))?;
        ensure!(w.history.frontier()==prompt.len()+ids.len()-1&&w.history.pending()==ids.last().copied()
            &&ids.len()==w.emitted&&(stop||w.emitted==cap),"expert hint subsequent committed frontier");
        let record=json!({"case":name,"seed":seed,"prompt_tokens":prompt.len(),"begin":begin,
            "previous_routes":routes,"selected":selected,"proposed":proposed,"trials":trials,
            "steps":steps,"output_ids":ids,"finish":finish,"subsequent_frontier_checked":true});
        std::fs::write(out.join(format!("expert-prefetch-c{ordinal}.json")),serde_json::to_vec_pretty(&record)?)?;
        records.push(record);
    }
    w.execute(Command::Shutdown)?;
    std::fs::write(out.join("expert-prefetch-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"startup":startup,"records":records,"all_logits_and_state_exact":true,
        "scope":"Three public phases, one causal next-step frontier each, liveM3/M6, reversed four-arm schedules. One FC1 expert per layer predicted from the preceding neural step; fixed GPU descriptor before capture. Actual attention collective/residual/normalization overlap. Six warm timings per arm on the same weights/allocations, changed-input exact local logits and retained state on all four ranks, ordinary subsequent generation. CPU route readback/descriptor preparation and a future GPU predictor are excluded; any win needs a real online selector before becoming a serving feature."}))?)?;
    Ok(())
}
