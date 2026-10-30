//! Omit work that a >=128-token suffix replaces. No serving-cache changes.
use crate::{control_channel::ControlChannel,cuda::{Cuda,Stream},dspark_generation::{
    GreedySession,PreparedSession,StateImage,teacher_forced_logits},roce::Roce,
    target::{Target,EncoderPrefixImage}};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};
const BUDGET:usize=2*1024usize.pow(3);
enum Saved {Complete(StateImage),Encoder(EncoderPrefixImage)}
fn hash(x:&[u8])->String {format!("{:x}",Sha256::digest(x))}
fn logit_hash(x:&[f32])->String {
    let mut h=Sha256::new();for f in x {h.update(f.to_bits().to_le_bytes());}format!("{:x}",h.finalize())
}
fn attach(session:&mut Option<GreedySession>,prepared:&mut Option<PreparedSession>,
    target:&Target,cuda:&Cuda,core:&[u8],stream:&Stream)->Result<()> {
    if let Some(s)=session {s.reset_from_prefill(target,cuda,core,stream)?;}
    else {*session=Some(prepared.take().ok_or_else(||anyhow::anyhow!("prepared drafter missing"))?.attach(target,stream)?);}
    Ok(())
}
fn capture_state(target:&Target,session:&mut GreedySession,comm:&Roce,stream:&Stream)->Result<Value> {
    Ok(json!({"position":target.position(),"carry":target.carry_digests()?,
        "encoder":target.encoder_tail_digest()?,"draft":session.conditioning_state_digest_for_probe(comm,stream)?}))
}
fn capture(target:&Target,session:&mut GreedySession,comm:&Roce,stream:&Stream)->Result<Value> {
    let mut state=capture_state(target,session,comm,stream)?;
    state["logits"]=json!(logit_hash(&target.last_prompt_logits()?));Ok(state)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,core:&[u8],large:&mut Target,verify:&mut Target,wides:&mut[Target],
    prepared:&mut Option<PreparedSession>,session:&mut Option<GreedySession>,comm:&Roce,stream:&Stream,
    control:&mut ControlChannel,output:&Path,rank:usize)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_ENCODER_CHECKPOINT_SCREEN").as_deref()==Ok("1")
        && large.position()==0&&large.context_capacity()==1048576,"encoder checkpoint probe admission");
    let fixtures=[include_bytes!("../tests/fixtures/target-agentic-repository-v1.json").as_slice(),
        include_bytes!("../tests/fixtures/long-context-continuation-v2.json").as_slice()];
    let fallback_only=std::env::var("CANADA_SCRATCH_ENCODER_CHECKPOINT_FALLBACK_SCREEN").as_deref()==Ok("1");
    let mut records=Vec::new();let mut fallback_records=Vec::new();let mut lineage=7_670_000u64;
    let cases=if fallback_only {vec![(8192usize,512usize,128usize)]}
        else {vec![(8192,512,128),(131072,1024,221)]};
    for(case,(base_tokens,added,suffix_tokens))in cases.into_iter().enumerate() {
        let fixture:Value=serde_json::from_slice(fixtures[case])?;
        let tokens:Vec<u32>=serde_json::from_value(fixture.get("tokens").unwrap_or(&fixture["batches"][0]).clone())?;
        let end=base_tokens+added+suffix_tokens;
        ensure!(tokens.len()>=end+6,"encoder checkpoint fixture extent");
        if case>0 {lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;}
        large.prefill_wide(wides,&tokens[..base_tokens],comm,stream)?;
        let base=large.save_state_image(BUDGET,comm,stream)?;
        verify.handoff_from(large)?;
        let prefix=&tokens[base_tokens..base_tokens+added];let suffix=&tokens[base_tokens+added..end];
        let mut expected=None;let mut expected_following=None;
        let mut expected_prefix=None;let mut expected_short=None;let mut expected_short_following=None;
        for repeat in 0..if fallback_only {1u8}else{3u8} {
            for path in if repeat==2 {[1u8,0]}else{[0u8,1]} {
                control.agree(&[case as u8,repeat,path])?;
                large.handoff_from(verify)?;lineage+=1;
                large.restore_state_image(&base,lineage,comm,stream)?;
                let started=Instant::now();
                let(saved,details)=if path==0 {
                    large.prefill_wide(wides,prefix,comm,stream)?;
                    attach(session,prepared,large,cuda,core,stream)?;
                    if fallback_only {expected_prefix=Some(capture(large,session.as_mut().unwrap(),comm,stream)?);}
                    let anchor=large.greedy_token(stream)?;
                    let prefix_seconds=started.elapsed().as_secs_f64();let save_at=Instant::now();
                    let(device_bytes,_)=session.as_ref().unwrap().compact_state_image_bytes(large)?;
                    ensure!(device_bytes<=BUDGET,"complete checkpoint exceeds probe budget");
                    let saved=session.as_mut().unwrap().save_state_image(large,anchor,device_bytes,comm,stream)?;
                    let save_seconds=save_at.elapsed().as_secs_f64();
                    large.prefill_wide(wides,suffix,comm,stream)?;
                    session.as_mut().unwrap().resume_after_extend(large,(base_tokens+added)as u64,suffix,stream)?;
                    (Saved::Complete(saved),json!({"prefix_seconds":prefix_seconds,"save_seconds":save_seconds,
                        "suffix_phases":large.append_metrics()?}))
                }else{
                    let(saved,details)=large.prefill_encoder_checkpoint_for_probe(wides,prefix,suffix,BUDGET,comm,stream)?;
                    attach(session,prepared,large,cuda,core,stream)?;
                    (Saved::Encoder(saved),details)
                };
                let fresh_seconds=started.elapsed().as_secs_f64();
                let state=capture(large,session.as_mut().unwrap(),comm,stream)?;
                std::fs::write(output.join(format!("encoder-checkpoint-c{case}-r{repeat}-p{path}-first.json")),
                    serde_json::to_vec_pretty(&json!({"state":state,"expected":expected,"details":details,
                        "fresh_seconds":fresh_seconds}))?)?;
                if expected.is_none(){expected=Some(state.clone());}
                ensure!(expected.as_ref()==Some(&state),"encoder checkpoint changed final target/drafter state or logits");
                verify.handoff_from(large)?;
                let following=teacher_forced_logits(cuda,verify,&tokens[end..end+6],1,comm,stream)?
                    .iter().map(|x|logit_hash(x)).collect::<Vec<_>>();
                session.as_mut().unwrap().resume_after_small_extend(verify,end as u64,
                    &tokens[end..end+6],comm,stream)?;
                if expected_following.is_none(){expected_following=Some(following.clone());}
                ensure!(expected_following.as_ref()==Some(&following),"encoder checkpoint changed following positions");
                large.handoff_from(verify)?;lineage+=1;
                let bytes=match &saved {Saved::Complete(s)=>s.device_bytes(),Saved::Encoder(s)=>s.bytes()};
                // Invalid suffix admission must fail before any target mutation.
                if let Saved::Encoder(s)=&saved {
                    let position=large.position();
                    ensure!(large.restore_encoder_checkpoint_for_probe(wides,s,lineage,&suffix[..127],comm,stream).is_err()
                        &&large.position()==position&&large.profile_is_current(),"short suffix admission mutated target");
                }
                let restore_at=Instant::now();
                match &saved {
                    Saved::Complete(s)=>{
                        session.as_mut().unwrap().restore_state_image(large,s,lineage,comm,stream)?;
                        large.prefill_wide(wides,suffix,comm,stream)?;
                        session.as_mut().unwrap().resume_after_extend(large,(base_tokens+added)as u64,suffix,stream)?;
                    },
                    Saved::Encoder(s)=>{
                        large.restore_encoder_checkpoint_for_probe(wides,s,lineage,suffix,comm,stream)?;
                        attach(session,prepared,large,cuda,core,stream)?;
                    },
                }
                let restore_seconds=restore_at.elapsed().as_secs_f64();
                let restored=capture(large,session.as_mut().unwrap(),comm,stream)?;
                std::fs::write(output.join(format!("encoder-checkpoint-c{case}-r{repeat}-p{path}-restored.json")),
                    serde_json::to_vec_pretty(&json!({"restored":restored,"expected":state,
                        "restore_seconds":restore_seconds}))?)?;
                ensure!(restored==state,"encoder checkpoint restore/replay state or logits differ");
                verify.handoff_from(large)?;
                let replay=teacher_forced_logits(cuda,verify,&tokens[end..end+6],1,comm,stream)?
                    .iter().map(|x|logit_hash(x)).collect::<Vec<_>>();
                session.as_mut().unwrap().resume_after_small_extend(verify,end as u64,
                    &tokens[end..end+6],comm,stream)?;
                ensure!(following==replay,"encoder checkpoint restore changed subsequent positions");
                records.push(json!({"case":case,"repeat":repeat,"warmup":repeat==0,"path":path,
                    "base_tokens":base_tokens,"checkpoint_tokens":base_tokens+added,"suffix_tokens":suffix_tokens,
                    "fresh_seconds":fresh_seconds,"restore_seconds":restore_seconds,"image_bytes":bytes,
                    "details":details,"state":state,"following":following,"all_exact":true}));
                if fallback_only {
                    large.handoff_from(verify)?;lineage+=1;
                    let restore_at=Instant::now();
                    match &saved {
                        Saved::Complete(s)=>{session.as_mut().unwrap().restore_state_image(large,s,lineage,comm,stream)?;},
                        Saved::Encoder(s)=>{
                            large.restore_encoder_checkpoint_full(s,lineage,comm,stream)?;
                            attach(session,prepared,large,cuda,core,stream)?;
                        },
                    }
                    let reconstructed_seconds=restore_at.elapsed().as_secs_f64();
                    // Complete snapshots omit scratch logits. Validate their
                    // retained state without asking the restored target to
                    // publish a stale row. The encoder path reconstructs the
                    // row and must match the original full-prefix logits too.
                    let prefix_state=if path==0 {
                        let state=capture_state(large,session.as_mut().unwrap(),comm,stream)?;
                        let mut expected=expected_prefix.clone().unwrap();
                        expected.as_object_mut().unwrap().remove("logits");
                        ensure!(state==expected,"complete checkpoint restored state differs");state
                    }else{
                        let state=capture(large,session.as_mut().unwrap(),comm,stream)?;
                        ensure!(expected_prefix.as_ref()==Some(&state),"reconstructed prefix differs from full checkpoint");state
                    };
                    let short_end=base_tokens+added+17;
                    let short=&tokens[base_tokens+added..short_end];
                    large.prefill_wide(wides,short,comm,stream)?;
                    session.as_mut().unwrap().resume_after_extend(large,(base_tokens+added)as u64,short,stream)?;
                    let short_state=capture(large,session.as_mut().unwrap(),comm,stream)?;
                    if expected_short.is_none(){expected_short=Some(short_state.clone());}
                    ensure!(expected_short.as_ref()==Some(&short_state),"short-suffix fallback state differs");
                    verify.handoff_from(large)?;
                    let following=teacher_forced_logits(cuda,verify,&tokens[short_end..short_end+6],1,comm,stream)?
                        .iter().map(|x|logit_hash(x)).collect::<Vec<_>>();
                    session.as_mut().unwrap().resume_after_small_extend(verify,short_end as u64,
                        &tokens[short_end..short_end+6],comm,stream)?;
                    if expected_short_following.is_none(){expected_short_following=Some(following.clone());}
                    ensure!(expected_short_following.as_ref()==Some(&following),"short-suffix fallback following logits differ");
                    fallback_records.push(json!({"case":case,"path":path,"reconstructed_seconds":reconstructed_seconds,
                        "prefix_state":prefix_state,"suffix_tokens":17,"short_state":short_state,"following":following,"all_exact":true}));
                }
            }
        }
    }
    std::fs::write(output.join("encoder-checkpoint-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"fallback_only":fallback_only,"fallback_records":fallback_records,
        "fixture_sha256":fixtures.iter().map(|f|hash(f)).collect::<Vec<_>>(),
        "scope":if fallback_only {"One8K-prefix paired fallback check. Reconstruct complete128-row decoder/drafter prefix from encoder image, then17newtokens and6following positions; compare to the complete checkpoint. Also recheck the128-token fast restore. No performance qualification or serving/background changes."}else{
        "Finite foreground mechanism screen: two real repository prefixes,128/221token suffixes, excluded warmups and two reversed-order pairs. Complete target+draft checkpoint versus encoder-only checkpoint; final logits/carries/encoder/draft and6following positions, including restore after further writes. No serving-cache or background ownership changes."}}))?)?;
    Ok(())
}
