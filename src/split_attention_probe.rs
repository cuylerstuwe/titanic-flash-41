//! Fixed-prefix comparison; FP8 and BF16 distributions need not be bitwise equal.
use crate::{control_channel::ControlChannel,cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};
fn hash(x:&[u8])->String {format!("{:x}",Sha256::digest(x))}
fn state(t:&Target)->Result<Value>{Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))}
fn logits(cuda:&Cuda,t:&Target)->Result<Vec<u8>>{let(y,rows)=t.pending_logits()?;unsafe{cuda.read_device(y.ptr(),rows*129280*4)}}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_append(cuda:&Cuda,large:&mut Target,verify:&mut Target,wide:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,control:&mut ControlChannel,out:&Path,rank:usize)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_APPEND_SPLIT_ATTENTION_SCREEN").as_deref()==Ok("1")
        &&large.position()==0&&verify.physical_rows()==6&&wide.physical_rows()==16,"append split finite admission");
    let short_raw=include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json");
    let short:Value=serde_json::from_slice(short_raw)?;
    let c=short["cases"].as_array().unwrap().iter().find(|c|c["id"]=="offline-product-ideation-answer-3").unwrap();
    let tokens:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
    let long_raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let long:Value=serde_json::from_slice(long_raw)?;let long_tokens:Vec<u32>=serde_json::from_value(long["tokens"].clone())?;
    let cases=[("ideation-answer",tokens.len()-22,tokens),("long-agentic-context",131072,long_tokens)];
    let mut lineage=7_980_000;let mut records=Vec::new();
    for(case,(id,prefix,tokens))in cases.iter().enumerate(){
        if case>0{lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;}
        large.prefill_wide(wides,&tokens[..*prefix],comm,stream)?;verify.handoff_from(large)?;
        let bytes=verify.compact_state_image_bytes()?;
        ensure!(bytes<=1024usize.pow(3)&&cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+bytes,
            "append split image reserve");
        let image=verify.save_state_image(bytes,comm,stream)?;let incoming=state(verify)?;
        for rows in [6usize,16] {
            let mut references:Vec<Option<Value>>=vec![None;2];
            for repeat in 0..2u8 {
                for path in if repeat==0{[0u8,1]}else{[1,0]} {
                    if rows==16 {wide.handoff_from(verify)?;wide.set_append_split_for_probe(path==1,stream)?;verify.handoff_from(wide)?;}
                    else{verify.set_append_split_for_probe(path==1,stream)?;}
                    let mut times=Vec::new();let mut first_seconds=0.;let mut signature=Value::Null;
                    for replay in 0..4u8 {
                        lineage+=1;verify.restore_state_image(&image,lineage,comm,stream)?;
                        ensure!(state(verify)?==incoming,"append split prefix restore differs");
                        control.agree(&[case as u8,rows as u8,repeat,path,replay])?;
                        let target=if rows==16{wide.handoff_from(verify)?;&mut *wide}else{&mut *verify};
                        target.set_tiny_extend_for_probe(true,stream)?;
                        let start=Instant::now();target.begin_verification(&tokens[*prefix..*prefix+rows],comm,stream)?;
                        target.finish_verification(rows,comm,stream)?;
                        let seconds=start.elapsed().as_secs_f64();
                        if replay==0{first_seconds=seconds;}else{times.push(seconds);}
                        let y=target.logits()?;
                        let append=unsafe{cuda.read_device(y.ptr()+((rows-1)*129280*4)as u64,129280*4)}?;
                        let accepted=state(target)?;
                        target.set_tiny_extend_for_probe(false,stream)?;
                        if rows==16{verify.handoff_from(wide)?;}
                        let before=state(verify)?;verify.begin_verification(&tokens[*prefix+rows..*prefix+rows+6],comm,stream)?;
                        let follow=logits(cuda,verify)?;verify.finish_verification(0,comm,stream)?;
                        ensure!(state(verify)?==before,"append split following abort differs");
                        signature=json!({"append_sha256":hash(&append),"follow_sha256":hash(&follow),"state":accepted});
                        if let Some(reference)=&references[path as usize]{
                            if *reference!=signature {
                                std::fs::write(out.join("append-split-mismatch.json"),serde_json::to_vec_pretty(&json!({
                                    "case":case,"id":id,"prefix_tokens":prefix,"rows":rows,"repeat":repeat,"path":path,
                                    "replay":replay,"lineage":lineage,"reference":reference,"observed":signature,
                                    "reference_stem":format!("append-split-c{case}-m{rows}-p{path}"),
                                    "first_seconds":first_seconds,"seconds_so_far":times,
                                    "prefix_restore_exact":true,"following_abort_exact":true}))?)?;
                                if rank==0 {
                                    std::fs::write(out.join("append-split-mismatch-append.bin"),&append)?;
                                    std::fs::write(out.join("append-split-mismatch-follow.bin"),&follow)?;
                                }
                                anyhow::bail!("append split same-path replay differs; actual tensors and timing preserved");
                            }
                        }
                        else{references[path as usize]=Some(signature.clone());}
                        if rank==0&&repeat==0&&replay==0 {
                            let stem=format!("append-split-c{case}-m{rows}-p{path}");
                            std::fs::write(out.join(format!("{stem}-append.bin")),append)?;
                            std::fs::write(out.join(format!("{stem}-follow.bin")),follow)?;
                        }
                    }
                    records.push(json!({"case":case,"id":id,"prefix_tokens":prefix,"rows":rows,"repeat":repeat,"path":path,
                        "stem":format!("append-split-c{case}-m{rows}-p{path}"),"first_seconds":first_seconds,
                        "seconds":times,"signature":signature,"replay_and_abort_exact":true}));
                    std::fs::write(out.join("append-split-partial.json"),serde_json::to_vec(&records)?)?;
                }
            }
        }
    }
    std::fs::write(out.join("append-split-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,"records":records,
        "fp32_partials":std::env::var("CANADA_SCRATCH_SPLIT_ATTENTION_FP32").as_deref()==Ok("1"),
        "scope":"Two public prefixes,6/16-token BF16 appends with ordinary/split QK kernels. Two reversed pairs,3warm replays, exact same-path state/logits and following6rollback. Normal generation kernels unchanged. CUDA distribution diagnostics, not completed outcomes."}))?)?;Ok(())
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,control:&mut ControlChannel,output:&Path,rank:usize)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_SPLIT_ATTENTION_SCREEN").as_deref()==Ok("1")
        && large.position()==0 && verify.physical_rows()==6,"split attention probe admission");
    let short_raw=include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json");
    let short:Value=serde_json::from_slice(short_raw)?;
    let short_case=short["cases"].as_array().unwrap().iter().find(|c|c["id"]=="offline-product-ideation-answer-2").unwrap();
    let tokens:Vec<u32>=serde_json::from_value(short_case["input_ids"].clone())?;
    let long_raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let long:Value=serde_json::from_slice(long_raw)?;let long_tokens:Vec<u32>=serde_json::from_value(long["tokens"].clone())?;
    let cases=[("ideation-answer",tokens.len()-6,tokens),("long-agentic-context",131072,long_tokens)];
    let mut records=Vec::new();
    for(case,(id,prefix,tokens))in cases.iter().enumerate(){
        ensure!(tokens.len()>=prefix+6,"split attention fixture extent");
        if case>0 {verify.set_split_attention_for_probe(0,stream)?;verify.reset(7_810_000+case as u64,comm,stream)?;large.handoff_from(verify)?;}
        large.prefill_wide(wides,&tokens[..*prefix],comm,stream)?;verify.handoff_from(large)?;
        let accepted=state(verify)?;
        for rows in [3usize,6] {
            let mut references:Vec<Option<Vec<u8>>>=vec![None;3];
            for repeat in 0..2u8 {
                for path in if repeat==0 {[0,1,2]}else{[2,1,0]} {
                    verify.set_split_attention_for_probe(path,stream)?;
                    control.agree(&[case as u8,rows as u8,repeat,path,0])?;
                    verify.begin_verification(&tokens[*prefix..*prefix+rows],comm,stream)?;
                    let warm=logits(cuda,verify)?;verify.finish_verification(0,comm,stream)?;
                    if let Some(reference)=&references[path as usize]{ensure!(*reference==warm,"same attention path changed logits");}
                    else{references[path as usize]=Some(warm.clone());}
                    let mut seconds=Vec::new();
                    for replay in 1..=3u8 {
                        control.agree(&[case as u8,rows as u8,repeat,path,replay])?;
                        let start=Instant::now();verify.begin_verification(&tokens[*prefix..*prefix+rows],comm,stream)?;
                        seconds.push(start.elapsed().as_secs_f64());let observed=logits(cuda,verify)?;
                        verify.finish_verification(0,comm,stream)?;ensure!(observed==warm,"same attention path changed replay logits");
                    }
                    ensure!(state(verify)?==accepted,"attention abort changed retained state");
                    let stem=format!("split-c{case}-m{rows}-p{path}");
                    if rank==0 && repeat==0 {std::fs::write(output.join(format!("{stem}.bin")),&warm)?;}
                    records.push(json!({"case":case,"id":id,"prefix_tokens":prefix,"rows":rows,"path":path,
                        "repeat":repeat,"stem":stem,"forward_seconds":seconds,"logits_sha256":hash(&warm),"aborted_state_exact":true}));
                    std::fs::write(output.join("split-attention-partial.json"),serde_json::to_vec(&records)?)?;
                }
            }
        }
    }
    std::fs::write(output.join("split-attention-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"fixture_sha256":[hash(short_raw),hash(long_raw)],"all_replays_and_abort_state_exact":true,
        "fp32_partials":std::env::var("CANADA_SCRATCH_SPLIT_ATTENTION_FP32").as_deref()==Ok("1"),
        "scope":"Two accepted public prefixes: ideation answer and131Kagentic context;3/6live verification rows, FP8/unsplitQKBF16/directsplitQKBF16, reversed pairs with3warm timings. Only compressed attention differs. Within-path replay/rollback checks and GPU distribution diagnostics; no taskquality or serving promotion."}))?)?;
    Ok(())
}
