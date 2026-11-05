//! Small native confirmation of a fixed causal M2/M3/M6 proposal policy.
//! T1/P1, actual-q block verification, target precision and history unchanged.
use super::*;

pub(super) fn enabled()->bool {
    std::env::var("CANADA_SCRATCH_CAUSAL_TWO_THREE_SCREEN").as_deref()==Ok("1")
}

fn warm_widths(w:&mut Worker,prompt:&[u32])->Result<()> {
    let before=(w.verify.position(),w.verify.carry_digests()?,w.verify.encoder_tail_digest()?);
    let mut inputs=[w.history.pending().context("width warmup anchor")?;6];
    inputs[1..].copy_from_slice(&prompt[prompt.len()-5..]);
    for rows in [2,3,6] {
        w.verify.begin_local_verification(&inputs[..rows],&w.comm,&w.stream)?;
        w.verify.finish_verification(0,&w.comm,&w.stream)?;
    }
    ensure!((w.verify.position(),w.verify.carry_digests()?,w.verify.encoder_tail_digest()?)==before,
        "width warmup changed retained target state");
    Ok(())
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path,startup:Value)->Result<()> {
    ensure!(enabled()&&w.current.is_none()&&w.history.frontier()==0&&w.session_cache_budget==0
        &&!w.prompt_checkpoint&&!w.prompt_cache_logits&&w.history_lookup_enabled
        &&w.verify.local_target_enabled(),"isolated two/three policy admission");
    let short:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/root-feedback-exploration-tokens-v1.json"))?;
    let phase:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json"))?;
    let long:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let js=short["cases"].as_array().unwrap().iter().find(|c|c["id"]=="javascript-incremental-ndjson").unwrap();
    let plan=short["cases"].as_array().unwrap().iter().find(|c|c["id"]=="library-workshop-tradeoffs").unwrap();
    let idea=phase["cases"].as_array().unwrap().iter().find(|c|c["id"]=="offline-product-ideation-answer-3").unwrap();
    let mut answer:Vec<u32>=serde_json::from_value(idea["input_ids"].clone())?;
    answer.push(idea["anchor"].as_u64().unwrap()as u32);
    let long_tokens:Vec<u32>=serde_json::from_value(long["tokens"].clone())?;
    let cases=[("javascript-ndjson","coding-reasoning",serde_json::from_value(js["input_ids"].clone())?),
        ("library-workshop","planning-reasoning",serde_json::from_value(plan["input_ids"].clone())?),
        ("product-ideation","answer",answer),
        ("repository-131k","long-continuation",long_tokens[..131072].to_vec())];
    let mut records=Vec::new();let mut id=9_244_001;
    for (ordinal,(name,phase,prompt)) in cases.into_iter().enumerate() {
        let seed_count=if ordinal==3{1}else{2};
        for seed_offset in 0..seed_count {
            let seed=202611034400+ordinal as u64*10+seed_offset;
            let mut arms=Vec::new();
            if ordinal==0&&seed_offset==0 {arms.extend([(true,false),(true,true)]);}
            arms.extend(if (ordinal as u64+seed_offset)%2==0 {[(false,false),(false,true)]}
                else {[(false,true),(false,false)]});
            for (warmup,two_rows) in arms {
                let cap=if warmup{48}else{256};
                let actual_seed=seed+if warmup{1000}else{0};
                let begin=w.execute(Command::Begin{images:Vec::new(),id,seed:actual_seed,
                    greedy:false,limit:cap,tokens:prompt.clone()})?;id+=1;
                ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.
                    &&begin["block_verification"]==true&&begin["generation_fp32_attention"]==true
                    &&begin["causal_verify_threshold"]==0.25,"two/three canonical target law");
                warm_widths(w,&prompt)?;
                let session=w.session.as_mut().context("two/three session")?;
                ensure!(session.draft_temperature()==0.7,"actual-q draft temperature");
                let saved=session.proposal_policy_image()?;
                session.set_causal_two_three_for_probe(two_rows.then_some(0.25))?;
                let mut ids:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;
                let mut steps=Vec::new();w.control.agree(b"two-three-policy-start")?;
                let start=Instant::now();
                while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
                    let limit=(w.limit-w.emitted).min(6);
                    let step=w.execute(Command::Step)?;
                    if step["proposal_source"]=="dspark" {
                        let confidence:[f32;5]=serde_json::from_value(step["draft_confidence"].clone())?;
                        let expected=if two_rows&&confidence[0]*confidence[1]<0.25 {limit.min(2)}
                            else if confidence[..3].iter().product::<f32>()<0.25 {limit.min(3)}else{limit};
                        ensure!(step["verified_rows"].as_u64()==Some(expected as u64),"noncausal width decision");
                    }
                    ids.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);
                    steps.push(step);
                }
                let seconds=start.elapsed().as_secs_f64();
                w.session.as_mut().unwrap().restore_proposal_policy(&saved,&w.stream)?;
                let stop=w.history.pending()==Some(w.cfg.eos_token_id);
                let finish=w.execute(Command::Finish(if stop{Finish::Stop}else{Finish::Length}))?;
                ensure!(w.history.frontier()==prompt.len()+ids.len()-1&&w.history.pending()==ids.last().copied()
                    &&ids.len()==w.emitted&&(stop||w.emitted==cap),"two/three committed frontier");
                let record=json!({"case":name,"phase":phase,"seed":actual_seed,"warmup":warmup,
                    "two_rows":two_rows,"cap":cap,"prompt_tokens":prompt.len(),"begin":begin,
                    "finish":finish,"steps":steps,"output_ids":ids,"seconds":seconds,
                    "frontier_checked":true,"causal_width_checked":true,"policy_restored":true});
                std::fs::write(out.join(format!("two-three-c{ordinal}-s{seed_offset}-warm{warmup}-m2{two_rows}.json")),
                    serde_json::to_vec_pretty(&record)?)?;
                records.push(record);
                eprintln!("two/three rank{rank} {name} seed{seed_offset} warm{warmup} m2{two_rows} complete");
            }
        }
    }
    w.execute(Command::Shutdown)?;
    std::fs::write(out.join("two-three-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"startup":startup,"records":records,"two_threshold":0.25,"three_threshold":0.25,
        "scope":"Four public phases; two fresh seeds each except one131K pair. Alternating native policy order,256-token capped continuations plus two excluded48-token warmups. Two short prompt families are held out from M2 cost/surrogate selection. Unchanged targetT1/P1,actual-q rejection,draftT.7,history policy and2350MHz. No completed-task quality claim or serving/default change."}))?)?;
    Ok(())
}
