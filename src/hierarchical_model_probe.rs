//! Small within-load foreground screen. A different draw mapping is expected;
//! target law, state ownership and phase costs are checked independently.
use super::*;
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.history.frontier()==0,"hierarchical screen starts idle");
    let local_target_screen=std::env::var("CANADA_SCRATCH_LOCAL_TARGET_MODEL_SCREEN").as_deref()==Ok("1");
    let local_history_screen=std::env::var("CANADA_SCRATCH_LOCAL_HISTORY_MODEL_SCREEN").as_deref()==Ok("1");
    let root_capture=std::env::var_os("CANADA_SCRATCH_ROOT_BRANCH_CAPTURE").is_some();
    ensure!(!root_capture||local_history_screen,"root capture uses the current combined candidate");
    ensure!(!local_history_screen||local_target_screen,"local history screen requires local target workspace");
    if std::env::var("CANADA_SCRATCH_HIERARCHICAL_LAW_ONLY").as_deref()==Ok("1") {
        ensure!(sampler_law_probe::enabled(),"law-only screen requires sampler law admission");
        return sampler_law_probe::run(w,rank,out);
    }
    let fresh:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/source-prior-transfer-tokens-v1.json"))?;
    let saved:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json"))?;
    let long:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let typescript=fresh["cases"].as_array().unwrap().iter().find(|x|x["id"]=="typescript-singleflight-cache").unwrap();
    let prose=saved["cases"].as_array().unwrap().iter().find(|x|x["id"]=="offline-product-ideation-answer-3").unwrap();
    let mut prose_ids:Vec<u32>=serde_json::from_value(prose["input_ids"].clone())?;
    prose_ids.push(prose["anchor"].as_u64().unwrap()as u32);
    let long_ids:Vec<u32>=serde_json::from_value(long["tokens"].clone())?;
    let cases=vec![("typescript-reasoning",serde_json::from_value::<Vec<u32>>(typescript["input_ids"].clone())?),
        ("ideation-answer",prose_ids),("repository-131k",long_ids[..131072].to_vec())];
    // Keep this interval below the following law panel's100000 interval.
    let mut id=9_000;let mut records=Vec::new();
    let mut progress=std::fs::OpenOptions::new().write(true).create_new(true).open(out.join("hierarchical-model-records.jsonl"))?;
    for(family,(name,prompt))in cases.into_iter().enumerate() {
        let mut initial=None;
        let mut exact_continuation=None;
        for repeat in 0..if root_capture{1}else{2} {
            for active in if root_capture{vec![true]}else if repeat==0{vec![false,true]}else{vec![true,false]} {
                let begin=w.execute(Command::Begin{id,seed:202611030010+family as u64,greedy:false,
                    limit:320,tokens:prompt.clone(),images:Vec::new()})?;id+=1;
                ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.&&begin["block_verification"]==true,
                    "hierarchical target policy changed");
                let start=json!({"carry":w.verify.carry_digests()?,"encoder":w.verify.encoder_tail_digest()?,
                    "frontier":w.history.frontier(),"pending":w.history.pending()});
                if let Some(reference)=&initial{ensure!(reference==&start,"hierarchical paired initial target state changed");}
                else{initial=Some(start.clone());}
                w.session.as_mut().context("hierarchical sampled session")?
                    .set_hierarchical_for_probe(local_target_screen||active,&w.stream)?;
                if local_target_screen{w.verify.set_local_target_for_probe(local_history_screen||active,&w.stream)?;}
                if local_history_screen{w.session.as_mut().unwrap().set_local_history_for_probe(active)?;}
                let mut ids:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;
                let mut warmup=Vec::new();let mut steps=Vec::new();
                for _ in 0..4 {
                    if w.history.pending()==Some(w.cfg.eos_token_id){break;}
                    let step=w.execute(Command::Step)?;
                    ids.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);warmup.push(step);
                }
                w.control.agree(b"hierarchical-model-measure")?;
                let emitted_before=w.emitted;let at=Instant::now();
                for _ in 0..24 {
                    if w.emitted>=w.limit||w.history.pending()==Some(w.cfg.eos_token_id){break;}
                    let captures=w.verify.capture_statistics().1;
                    let mut step=w.execute(Command::Step)?;
                    step["new_verifier_captures"]=json!(w.verify.capture_statistics().1-captures);
                    ids.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);steps.push(step);
                }
                let seconds=at.elapsed().as_secs_f64();let interval_tokens=w.emitted-emitted_before;
                ensure!(w.history.frontier()==prompt.len()+ids.len()-1&&w.history.pending()==ids.last().copied()
                    &&ids.len()==w.emitted,"hierarchical accepted frontier");
                let state=json!({"carry":w.verify.carry_digests()?,"encoder":w.verify.encoder_tail_digest()?,
                    "frontier":w.history.frontier(),"pending":w.history.pending()});
                if local_target_screen {
                    let observed=json!({"ids":ids,"state":state});
                    if let Some(previous)=&exact_continuation{ensure!(previous==&observed,"local target changed emitted tokens or retained causal state");}
                    else{exact_continuation=Some(observed);}
                    w.verify.set_local_target_for_probe(false,&w.stream)?;
                    if local_history_screen{w.session.as_mut().unwrap().set_local_history_for_probe(false)?;}
                }
                let stop=w.history.pending()==Some(w.cfg.eos_token_id);
                w.session.as_mut().unwrap().set_hierarchical_for_probe(false,&w.stream)?;
                let finish=w.execute(Command::Finish(if stop{Finish::Stop}else{Finish::Cancelled}))?;
                let record=json!({"family":name,"repeat":repeat,"hierarchical":local_target_screen||active,
                    "local_target":local_target_screen&&(local_history_screen||active),"local_history":local_history_screen&&active,"prompt_tokens":prompt.len(),
                    "initial":start,"begin":begin,"warmup":warmup,"steps":steps,"ids":ids,"state":state,
                    "seconds":seconds,"interval_tokens":interval_tokens,"tokens_per_second":interval_tokens as f64/seconds,
                    "finish":finish,"completed_answer":stop});
                writeln!(progress,"{record}")?;progress.flush()?;records.push(record);
            }
        }
    }
    if local_target_screen&&!root_capture {
        // Final single-token remainder deliberately takes the ordinary full
        // target fallback while the compact graph policy is active.
        let prompt:Vec<u32>=serde_json::from_value(typescript["input_ids"].clone())?;
        let mut previous=None;
        for active in [false,true] {
            let begin=w.execute(Command::Begin{id,seed:202611030091,greedy:false,limit:2,tokens:prompt.clone(),images:Vec::new()})?;id+=1;
            w.session.as_mut().unwrap().set_hierarchical_for_probe(true,&w.stream)?;
            w.verify.set_local_target_for_probe(active,&w.stream)?;
            let step=w.execute(Command::Step)?;
            ensure!(w.emitted==2,"final remainder did not finish");
            let observed=json!({"begin":begin["output_ids"],"output":step["output_ids"],
                "carry":w.verify.carry_digests()?,"encoder":w.verify.encoder_tail_digest()?});
            if let Some(prior)=&previous{ensure!(prior==&observed,"local-target full fallback changed output/state");}
            else{previous=Some(observed);}
            w.verify.set_local_target_for_probe(false,&w.stream)?;
            w.session.as_mut().unwrap().set_hierarchical_for_probe(false,&w.stream)?;
            w.execute(Command::Finish(Finish::Length))?;
        }
    }
    std::fs::write(out.join("hierarchical-model-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"target_temperature":1.,"target_top_p":1.,"root_branch_capture":root_capture,
        "local_target_screen":local_target_screen,"local_history_screen":local_history_screen,"exact_continuations_and_fallback":local_target_screen&&!root_capture,
        "scope":if root_capture{"Three public prompt/phase families, one capped continuation each. Read-only first-root probability shards every fourth neural step, at most8per case. Includes synchronous diagnostic downloads; not a performance or completed-quality panel."}
            else{"Three public prompt/phase families, two reversed mode pairs each, four warm and24 measured ordinary steps. Same loaded weights and target state at each start; draw mapping differs. Capped diagnostics, not completed-task quality evidence. Includes causal draft length and actual history policy; separately records graph capture costs."}}))?)?;
    if sampler_law_probe::enabled(){sampler_law_probe::run(w,rank,out)?;}
    Ok(())
}
