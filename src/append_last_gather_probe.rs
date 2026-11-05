//! Same-storage retained-append test of a pure transport pruning change.
use super::*;
use super::history32_generation_probe::{execute,state,restore};
fn hash(v:&[u8])->String {format!("{:x}",Sha256::digest(v))}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.copy32_verify.is_some()&&w.copy32_buffers.is_some()
        &&w.long_verify.is_some(),"last gather screen needs admitted append profiles");
    let head_screen=std::env::var("CANADA_SCRATCH_APPEND_HEAD_SCREEN").as_deref()==Ok("1");
    let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let ids:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    let prefix=32773;let mut changed=ids[prefix..prefix+32].to_vec();
    changed.rotate_left(7);changed[0]=(changed[0]+53)%129280;
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let begin=execute(w,Command::Begin{images:Vec::new(),id:8_041_295,seed:2026110441,
        greedy:false,limit:1,tokens:ids[..prefix].to_vec()})?;
    let anchor=begin["output_ids"][0].as_u64().context("last gather anchor")?as u32;
    let original=state(w)?;
    let(bytes,cpu)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
    ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
        >=8*1024usize.pow(3)+bytes+cpu)?,"last gather bounded snapshot reserve");
    let saved=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
    let mut records=Vec::new();
    let mut expected=std::collections::BTreeMap::<(usize,usize,usize),Value>::new();
    let mut path_expected=std::collections::BTreeMap::new();
    let mut control_logits=std::collections::BTreeMap::<(usize,usize,usize),Vec<f32>>::new();
    for rows in if head_screen{vec![16usize,32]}else{vec![6usize,16,32]} {
        for(trial,enabled)in[false,true,true,false].into_iter().enumerate() {
            for live in [rows/2+1,rows] {
                for repeat in 0..3 {
                    restore(w,&saved,anchor,&original)?;
                    let kind=usize::from(repeat==1);
                    let input=if kind==0{&ids[prefix..prefix+live]}else{&changed[..live]};
                    let target=if rows==16 {
                        let t=w.long_verify.as_mut().unwrap();t.handoff_from(&w.verify)?;t
                    }else if rows==32 {
                        let t=w.copy32_verify.as_mut().unwrap();t.handoff_from(&w.verify)?;t
                    }else{&mut w.verify};
                    target.set_append_last_gather_for_probe(head_screen||enabled,&w.stream)?;
                    if head_screen {target.set_append_encoded_head_for_probe(enabled,&w.stream)?;}
                    target.set_tiny_extend_for_probe(true,&w.stream)?;
                    w.control.agree(&[0xe3,rows as u8,live as u8,trial as u8,repeat])?;
                    let captures=target.capture_statistics().1;
                    let at=Instant::now();
                    target.begin_verification(input,&w.comm,&w.stream)?;
                    if head_screen||enabled {
                        ensure!(target.pending_logits().is_err(),"last-only append exposed all pending rows");
                        ensure!(target.finish_verification(live-1,&w.comm,&w.stream).is_err(),
                            "last-only append allowed a partial commit");
                    }
                    target.finish_verification(live,&w.comm,&w.stream)?;
                    let target_ms=at.elapsed().as_secs_f64()*1000.;
                    let recaptured=target.capture_statistics().1-captures;
                    let logits=unsafe{w.cuda.read_device(target.logits()?.ptr()+((live-1)*129280*4)as u64,129280*4)}?;
                    let sampled=target.sample_token(2026110442,&w.stream)?;
                    let normalized=if head_screen {Some(target.append_head_normalized_for_probe()?)}else{None};
                    let numeric=if head_screen {
                        let values:Vec<f32>=logits.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();
                        ensure!(values.iter().all(|v|v.is_finite()),"nonfinite final-head output");
                        let control=control_logits.entry((rows,live,kind)).or_insert_with(||values.clone());
                        let metrics=crate::dspark_generation::compare_logits(control,&values);
                        ensure!(metrics["total_variation"].as_f64().unwrap()<0.0001,"final-head distribution differs unexpectedly");
                        if rank==0&&trial<2&&repeat<2 {
                            let stem=format!("head-m{rows}-l{live}-k{kind}-variant{}",usize::from(enabled));
                            std::fs::write(out.join(format!("{stem}.logits.bin")),&logits)?;
                            std::fs::write(out.join(format!("{stem}.normalized.bin")),normalized.as_ref().unwrap())?;
                        }
                        metrics
                    }else{Value::Null};
                    let at=Instant::now();
                    let session=w.session.as_mut().unwrap();
                    if rows==16 {session.resume_after_medium_extend(target,w.long_history_buffers.as_mut().unwrap(),
                        prefix as u64,input,&w.comm,&w.stream)?;}
                    else if rows==32 {session.resume_after_medium_extend(target,w.copy32_buffers.as_mut().unwrap(),
                        prefix as u64,input,&w.comm,&w.stream)?;}
                    else {session.resume_after_small_extend(target,prefix as u64,input,&w.comm,&w.stream)?;}
                    let draft_ms=at.elapsed().as_secs_f64()*1000.;
                    target.set_tiny_extend_for_probe(false,&w.stream)?;
                    if rows==16 {w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;}
                    else if rows==32 {w.verify.handoff_from(w.copy32_verify.as_ref().unwrap())?;}
                    let accepted=state(w)?;
                    w.verify.begin_verification(&ids[prefix+live..prefix+live+6],&w.comm,&w.stream)?;
                    let(y,n)=w.verify.pending_logits()?;ensure!(n==6,"last gather following extent");
                    let following=unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?;
                    w.verify.finish_verification(0,&w.comm,&w.stream)?;
                    ensure!(state(w)?==accepted,"last gather following rollback");
                    let mut proof=json!({"logits":hash(&logits),"following":hash(&following),"state":accepted,"sampled":sampled});
                    let same_path=path_expected.entry((rows,live,kind,enabled)).or_insert_with(||proof.clone())==&proof;
                    ensure!(same_path,"same-path final-head repeat differs");
                    if head_screen {proof.as_object_mut().unwrap().remove("logits");proof.as_object_mut().unwrap().remove("sampled");
                        proof["normalized"]=json!(hash(normalized.as_ref().unwrap()));}
                    let exact=expected.entry((rows,live,kind)).or_insert_with(||proof.clone())==&proof;
                    ensure!(w.control.all_admit(exact)?,"last gather changed logits or accepted state");
                    records.push(json!({"rows":rows,"live":live,"trial":trial,"repeat":repeat,
                        "enabled":enabled,"kind":kind,"warmup":recaptured>0,"new_captures":recaptured,
                        "target_ms":target_ms,"draft_ms":draft_ms,"all_exact":!head_screen,"state_exact":true,"rollback_exact":true,"proof":proof,"distribution":numeric,"same_path_exact":same_path,"sampled":sampled}));
                }
            }
            std::fs::write(out.join("append-last-gather-progress.json"),serde_json::to_vec_pretty(&records)?)?;
        }
        eprintln!("rank{rank} append M{rows} state/following exact, head_screen={head_screen}");
    }
    restore(w,&saved,anchor,&original)?;
    execute(w,Command::Finish(Finish::Length))?;
    std::fs::write(out.join("append-last-gather-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"prefix_tokens":prefix,"all_exact":!head_screen,"all_state_exact":true,"all_same_path_exact":true,"head_screen":head_screen,"records":records,
        "scope":if head_screen {"Existing losslessM6 head used for final-only M16/M32 append. Canonical operands and FP32 output; FP32 accumulation order changes. Same normalized input, target/drafter/lookup state, following6 logits and rollback exact; per-path repeat exact; distribution delta retained. No serving change."}else{"Same-allocation ABBA last-row gather at M6/M16/M32 retained append. Two input patterns, final logits, fixed-seed sample, target/drafter/lookup state and following6 logits/rollback exact. Head arithmetic unchanged; no serving default change."}
    }))?)?;
    Ok(())
}
