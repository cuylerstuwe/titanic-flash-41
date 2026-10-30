//! Small paired worker screen. Public token fixtures, recommended T1/P1,
//! ordinary Begin/Step/Finish commands; no background or outcome-quality panel.
use super::*;
fn execute(w:&mut Worker,c:Command)->Result<Value> {
    let encoded=c.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(encoded.as_slice()))?;
    ensure!(wire==encoded,"encoder cache command replicas disagree");w.execute(Command::decode(&wire)?)
}
fn state(w:&mut Worker)->Result<Value> {
    Ok(json!({"frontier":w.verify.position(),"pending":w.history.pending(),
        "carry":w.verify.carry_digests()?,"encoder":w.verify.encoder_tail_digest()?,
        "draft":w.session.as_mut().unwrap().conditioning_state_digest_for_probe(&w.comm,&w.stream)?,
        "history":w.session.as_ref().unwrap().history_lookup_digest_for_probe()?}))
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_ENCODER_CACHE_SCREEN").as_deref()==Ok("1")
        &&w.encoder_prompt_checkpoint&&w.prompt_checkpoint&&w.history_lookup_enabled
        &&w.current.is_none()&&w.suspended_image_count==0,"encoder worker screen admission");
    let long=std::env::var("CANADA_SCRATCH_ENCODER_CACHE_LONG_SCREEN").as_deref()==Ok("1");
    let trace=std::env::var("CANADA_SCRATCH_ENCODER_CACHE_TRACE").as_deref()==Ok("1");
    let unexpected_only=std::env::var("CANADA_SCRATCH_ENCODER_CACHE_TRACE_UNEXPECTED_ONLY").as_deref()==Ok("1");
    ensure!(!unexpected_only||(trace&&!long),"failure-only capture uses the known short fixture sample");
    let upgrade=std::env::var("CANADA_SCRATCH_ENCODER_UPGRADE_RECONSTRUCTED").as_deref()==Ok("1");
    let poison=std::env::var("CANADA_SCRATCH_ENCODER_CACHE_POISON_SCREEN").as_deref()==Ok("1");
    ensure!(!poison||(!upgrade&&!long&&trace&&unexpected_only&&crate::target::defer_checkpoint_decoder128()?),
        "decoder poison isolates short omission with failure-only capture");
    // Isolate cache reconstruction from the independently qualified active
    // frontier policy, including when this screen follows another probe.
    w.encoder_active_frontier=false;
    let raw:&[u8]=if long {include_bytes!("../tests/fixtures/long-context-continuation-v2.json")}
        else{include_bytes!("../tests/fixtures/target-agentic-repository-v1.json")};
    let base=if long{131072}else{8192};
    let common=base+512;let source=if long{160000}else{15000};
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture.get("tokens").unwrap_or(&fixture["batches"][0]).clone())?;
    ensure!(tokens.len()>=source+3512,"encoder worker real repository fixture extent");
    let mut id=7_740_000;let mut records=Vec::new();let mut expected=std::collections::BTreeMap::<usize,Value>::new();
    for repeat in 0..2 {
        let mut variants=if poison{vec![(false,false,false),(true,false,false),(true,false,true)]}
            else if upgrade{vec![(false,false,false),(true,false,false),(true,true,false)]}
            else{vec![(false,false,false),(true,false,false)]};
        if repeat==1{variants.reverse();}
        for (enabled,upgrade_enabled,poison_enabled) in variants {
            w.control.agree(&[repeat,u8::from(enabled),u8::from(upgrade_enabled),u8::from(poison_enabled)])?;
            // The finite worker owns a single host thread; all ranks agree
            // before this diagnostic-only switch is consulted by the target.
            if poison {unsafe{std::env::set_var("CANADA_SCRATCH_ENCODER_POISON_DECODER",if poison_enabled{"1"}else{"0"});}}
            w.encoder_upgrade_reconstructed=upgrade_enabled;
            w.cached_sessions.clear();w.encoder_prompt_checkpoint=enabled;
            // Give each cold prompt the same proposal warmup metadata. Graph
            // allocations remain resident, exactly as across serving requests.
            if let Some(session)=&mut w.session {session.configure_greedy(Some(w.cfg.eos_token_id))?;}
            // Exercise ordinary aligned checkpoints, not the separate optional
            // exact-prompt-logit fast path. Its complete image format is unchanged.
            w.prompt_cache_logits=false;
            for &case in if poison{&[0usize,3][..]}else if upgrade{&[0usize,1,2,6,3,4,5][..]}else{&[0usize,1,2,3,4,5][..]} {
                let prompt=match case {
                    0=>tokens[..base+700].to_vec(),
                    1=>[&tokens[..common],&tokens[source..source+221]].concat(),
                    2=>[&tokens[..common],&tokens[source+1000..source+1017]].concat(),
                    3=>[&tokens[..common],&tokens[source+2000..source+2349]].concat(),
                    4=>[&tokens[..common],&tokens[source+2000..source+2128],&tokens[source+2500..source+2721]].concat(),
                    6=>[&tokens[..common],&tokens[source+1100..source+1117]].concat(),
                    _=>[w.history.committed(),&tokens[source+3000..source+3512]].concat(),
                };
                id+=1;
                let begin=execute(w,Command::Begin{images: Vec::new(),id,seed:20261028+case as u64,greedy:false,limit:16,tokens:prompt})?;
                if poison&&enabled&&case==3 {
                    let bytes=begin["encoder_checkpoint_phases"]["poisoned_decoder_bytes"].as_u64().unwrap();
                    ensure!((bytes>0)==poison_enabled,"decoder poison did not reach affected request");
                }
                if enabled&&case==3 {
                    ensure!(begin["prompt_checkpoint"]["saved"]==true
                        &&(upgrade_enabled||(begin["encoder_checkpoint_phases"]["consumed_parent"]==false
                            &&begin["retained_session_cache"]["retained_image"]==true)),
                        "advancing encoder checkpoint failed to retain its borrowed parent");
                }
                if upgrade_enabled&&case==2 {
                    ensure!(begin["retained_session_cache"]["reconstructed_upgrade"]==true,
                        "short branch failed to retain reconstructed complete parent");
                }
                if upgrade&&case==6 {
                    ensure!(begin["retained_session_cache"]["restored_tokens"]==common
                        &&begin["retained_session_cache"]["reconstructed_upgrade"]==false
                        &&begin["retained_session_cache"]["encoder_reconstructed"]==(enabled&&!upgrade_enabled),
                        "second short branch used unexpected reconstruction policy");
                }
                if case==4 {
                    ensure!(begin["retained_session_cache"]["restored_tokens"].as_u64()==Some((common+128)as u64),
                        "follow-up did not restore the replacement checkpoint");
                }
                let initial=state(w)?;
                let policy=w.session.as_ref().unwrap().proposal_policy_for_cache_probe()?;
                if trace {
                    let path=(case==3&&rank==0).then(||out.join(if poison {
                        format!("cache-trace-r{repeat}-p{}-poison{}",u8::from(enabled),u8::from(poison_enabled))
                    }else{format!("cache-trace-r{repeat}-p{}",u8::from(enabled))}));
                    w.session.as_mut().unwrap().set_encoder_cache_trace_for_probe(path,w.verify.position()+2,
                        unexpected_only.then_some(17535))?;
                }
                let mut steps=Vec::new();
                let mut output=serde_json::from_value::<Vec<u32>>(begin["output_ids"].clone())?;
                while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
                    let step=execute(w,Command::Step)?;
                    output.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);
                    steps.push(step);
                }
                let after=state(w)?;
                let reason=if w.history.pending()==Some(w.cfg.eos_token_id){Finish::Stop}else{Finish::Length};
                execute(w,Command::Finish(reason))?;
                let semantic=json!({"initial":initial,"after":after,"output":output});
                let equal=if repeat==0&&!enabled {expected.insert(case,semantic.clone());true}else{expected[&case]==semantic};
                records.push(json!({"repeat":repeat,"encoder":enabled,"upgrade":upgrade_enabled,"poison":poison_enabled,"case":case,"begin":begin,
                    "semantic":semantic,"exact":equal,"steps":steps,"policy":policy}));
                std::fs::write(out.join("encoder-cache-partial.json"),serde_json::to_vec_pretty(&records)?)?;
                ensure!(equal||trace,"encoder cache changed request{case} state, history index or sampled output");
                ensure!(w.cached_sessions.iter().map(|c|c.charge).sum::<usize>()<=w.shared_image_budget
                    &&w.cached_sessions.len()<=4,"encoder cache budget exceeded");
            }
        }
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("encoder-cache-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"long_context":long,"base_tokens":base,"repeats":2,"cases":if poison{2}else if upgrade{7}else{6},"paths":if poison||upgrade{3}else{2},"decoder_poison_screen":poison,"reconstructed_upgrade":upgrade,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "all_exact":records.iter().all(|x|x["exact"]==true),"sample_trace":trace,"trace_unexpected_only":unexpected_only,
        "scope":if poison{"Focused omitted128decoder diagnostic. Two reversed orders of complete checkpoint, encoder checkpoint and encoder with0xff poison in layers21..39SWA plus last decoder/replay hidden/pre. Fresh parent then affected128-prefix request; T1/P1,16-token caps. Compare checked state and output. Poison retains encoder inputs and does not test every transient allocation or stochastic race."}else if upgrade{"One real8Krepository prefix; complete checkpoint, encoder checkpoint, and encoder with lazy complete upgrade. Seven request shapes include two successive17-token branches, two reversed orders, T1/P1,16-token caps. Exact state and output, explicit reconstruction/upgrade checks. Foreground only, no completed-task quality claim."}else{"One real repository/context prefix; fresh creation, changed221-token suffix,17-token fallback, advancing128-token checkpoint, restoring its replacement, and active512-token append. Two reversed-order pairs, T1/P1,16-token caps. Compare causal carry/encoder/drafter/history state and output. Optional trace records a divergent sample without gating on identical text. Foreground worker core only; no completed-task quality or HTTP/production claim."}}))?)?;
    Ok(())
}
