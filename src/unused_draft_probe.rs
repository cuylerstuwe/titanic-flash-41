//! Narrow finishing-boundary screen; no target or proposal precision change.
use super::*;

fn accepted_state(w:&mut Worker)->Result<(Value,Value)> {
    let mut state=history32_generation_probe::state(w)?;
    let private_storage=state["draft"].clone();
    // A recovery image covers all384 slots, including the physically separate
    // five-row query scratch. Skipping that query intentionally leaves those
    // slots untouched. Compare the complete observable128-row accepted ring;
    // retain the full storage hash separately and still restore every byte.
    state["draft"]=w.session.as_ref().unwrap().accepted_conditioning_state_for_probe(&w.stream)?;
    Ok((state,private_storage))
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_UNUSED_DRAFT_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none()&&w.history.frontier()==0,"finite unused-draft screen");
    let fresh:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/source-prior-transfer-tokens-v1.json"))?;
    let saved:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json"))?;
    let c=fresh["cases"].as_array().unwrap().iter().find(|c|c["id"]=="rust-incremental-frames").unwrap();
    let reasoning:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
    let c=saved["cases"].as_array().unwrap().iter().find(|c|c["id"]=="offline-product-ideation-answer-2").unwrap();
    let mut answer:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
    answer.push(c["anchor"].as_u64().unwrap()as u32);
    let mut records=Vec::new();
    for(case,(name,prompt))in [("rust-reasoning",reasoning),("ideation-answer",answer)].into_iter().enumerate(){
        let begin=history32_generation_probe::execute(w,Command::Begin{images:Vec::new(),
            // Manual session steps below deliberately do not publish through
            // the worker request. Begin owns its single emitted anchor; after
            // exact restoration, Length is therefore a truthful completion.
            id:8_060_000+case as u64,seed:2026110401+case as u64,greedy:false,limit:1,tokens:prompt})?;
        ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.,"unused-draft target law");
        let anchor=w.history.pending().context("unused-draft pending anchor")?;
        let (bytes,cpu)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
        ensure!(bytes<=768*1024usize.pow(2)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
            >=8*1024usize.pow(3)+bytes+cpu)?,"unused-draft snapshot reserve");
        let image=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
        let original=history32_generation_probe::state(w)?;
        let mut expected=None;
        let mut following=None;
        let mut fixed_next=None;
        // First enabled arm runs before a neural query on the first prompt.
        // The first pair is excluded from timing; subsequent arms alternate
        // twice in each direction with fixed weights and accepted prefix.
        for(trial,enabled)in [true,false,false,true,true,false,false,true,true,false].into_iter().enumerate(){
            history32_generation_probe::restore(w,&image,anchor,&original)?;
            w.session.as_mut().unwrap().set_skip_unused_draft_for_probe(enabled)?;
            w.control.agree(&[0xd5,case as u8,trial as u8,enabled as u8])?;
            let at=Instant::now();
            let step=w.session.as_mut().unwrap().step(&mut w.verify,anchor,1,&w.comm,&w.stream)?;
            let seconds=at.elapsed().as_secs_f64();
            ensure!(step.tokens==[anchor]&&step.verified_rows==1&&step.matching_proposals==0
                &&(!enabled||(step.draft_tokens.is_empty()&&step.draft_confidence.is_none()
                    &&step.proposal_source=="one_row_target")),"unused-draft extent/publication");
            let (state,private_storage)=accepted_state(w)?;
            let outputs=w.verify.valid_output_digests()?;
            let check=json!({"state":state,"outputs":outputs});
            let exact=expected.as_ref().is_none_or(|value|value==&check);
            // Preserve a first-use mismatch before retiring the distributed
            // group; a digest alone cannot distinguish numerical drift from
            // an incorrect accepted-state transition.
            if trial<2 || !exact {
                let logits=w.verify.logits()?;
                let raw=unsafe{w.cuda.read_device(logits.ptr(),129280*4)}?;
                std::fs::write(out.join(format!("unused-draft-logits-{case}-{trial}.bin")),raw)?;
                std::fs::write(out.join(format!("unused-draft-check-{case}-{trial}.json")),
                    serde_json::to_vec_pretty(&json!({"case":name,"trial":trial,"enabled":enabled,
                        "seconds":seconds,"phase_ms":step.phase_ms,"actual":check,"expected":expected,
                        "exact":exact,"next_anchor":step.next_anchor,"private_draft_storage":private_storage}))?)?;
            }
            ensure!(w.control.all_admit(exact)?,"one-row target or accepted state changed");
            if expected.is_none(){expected=Some(check);fixed_next=Some(step.next_anchor);}
            let continuation=if trial<2 {
                // Samplers can consume different independent random draws.
                // Force the same next anchor only for this causal-state check;
                // following proposals and verification must then agree.
                let mut next=fixed_next.unwrap();let mut follow=Vec::new();
                for _ in 0..3 {
                    let s=w.session.as_mut().unwrap().step(&mut w.verify,next,6,&w.comm,&w.stream)?;
                    next=s.next_anchor;
                    follow.push(json!({"tokens":s.tokens,"next":s.next_anchor,
                        "draft":s.draft_tokens,"confidence":s.draft_confidence,"rows":s.verified_rows}));
                }
                let signature=json!({"steps":follow,"state":accepted_state(w)?.0});
                let exact=following.as_ref().is_none_or(|value|value==&signature);
                std::fs::write(out.join(format!("unused-draft-follow-{case}-{trial}.json")),
                    serde_json::to_vec_pretty(&json!({"actual":signature,"expected":following,"exact":exact}))?)?;
                ensure!(w.control.all_admit(exact)?,"following normal draft changed after skipped query");
                if following.is_none(){following=Some(signature);}
                Some(json!({"three_following_steps_exact":true}))
            }else{None};
            records.push(json!({"case":name,"trial":trial,"enabled":enabled,"warmup":trial<2,
                "seconds":seconds,"phase_ms":step.phase_ms,"next_anchor":step.next_anchor,
                "source":step.proposal_source,"accepted_state_and_logits_exact":exact,
                "continuation":continuation,"state":state,"outputs":outputs,"private_draft_storage":private_storage}));
            std::fs::write(out.join("unused-draft-partial.json"),serde_json::to_vec_pretty(&records)?)?;
        }
        history32_generation_probe::restore(w,&image,anchor,&original)?;
        w.session.as_mut().unwrap().set_skip_unused_draft_for_probe(false)?;
        history32_generation_probe::execute(w,Command::Finish(Finish::Length))?;
    }
    history32_generation_probe::execute(w,Command::Shutdown)?;
    std::fs::write(out.join("unused-draft-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,
        "records":records,"all_exact":true,"scope":"Two public reasoning/answer prefixes;T1/P1. One-row target-only versus unused five-row neural proposal. Cold first enabled arm, eight warmed alternating timings per prefix. Full target outputs and accepted target/draft/history state exact; three following normal steps checked at a common next anchor. Different independent final categorical draws permitted. No completed-task or general TPS claim."}))?)?;
    Ok(())
}
