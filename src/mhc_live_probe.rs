//! Small integration screen of an exact component change, with cold graph cost excluded.
use crate::{control_channel::ControlChannel,cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};
fn hash(x:&[u8])->String{format!("{:x}",Sha256::digest(x))}
fn state(t:&Target)->Result<Value>{Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))}
fn logits(cuda:&Cuda,t:&Target)->Result<Vec<u8>>{let(y,rows)=t.pending_logits()?;unsafe{cuda.read_device(y.ptr(),rows*129280*4)}}
fn exact(reference:&[u8],observed:&[u8],out:&Path,stem:&str)->Result<()> {
    if reference!=observed {
        std::fs::write(out.join(format!("{stem}-reference.bin")),reference)?;
        std::fs::write(out.join(format!("{stem}-observed.bin")),observed)?;
        anyhow::bail!("live-row logit mismatch saved: {stem}");
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wide:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,control:&mut ControlChannel,out:&Path,rank:usize,mut lineage:u64)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_MHC_LIVE_ROWS_SCREEN").as_deref()==Ok("1")
        &&large.position()==0&&verify.physical_rows()==6&&wide.physical_rows()==16,"live-row probe admission");
    let short_raw=include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json");
    let short:Value=serde_json::from_slice(short_raw)?;
    let c=short["cases"].as_array().unwrap().iter().find(|c|c["id"]=="offline-product-ideation-answer-2").unwrap();
    let tokens:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
    let long_raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let long:Value=serde_json::from_slice(long_raw)?;let long_tokens:Vec<u32>=serde_json::from_value(long["tokens"].clone())?;
    let cases=[("ideation-answer",tokens.len()-16,tokens),("long-agentic-context",131072,long_tokens)];
    let mut records=Vec::new();
    for(case,(id,prefix,tokens))in cases.iter().enumerate(){
        if case>0{lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;}
        large.prefill_wide(wides,&tokens[..*prefix],comm,stream)?;verify.handoff_from(large)?;
        let bytes=verify.compact_state_image_bytes()?;
        ensure!(bytes<=1024usize.pow(3)&&control.all_admit(cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+bytes)?,
            "mHC live-row screen snapshot reserve");
        let saved=verify.save_state_image(bytes,comm,stream)?;
        for &rows in if case==0{&[1usize,3,5,6][..]}else{&[3usize,6][..]} {
            if rows>6{wide.handoff_from(verify)?;}
            let target=if rows>6{&mut *wide}else{&mut *verify};
            let accepted=state(target)?;let mut reference:Option<Vec<u8>>=None;
            let mut committed_reference:Option<(Value,Vec<u8>)>=None;
            for repeat in 0..2u8 {
                for path in if repeat==0{[0u8,1]}else{[1u8,0]} {
                    target.set_mhc_live_rows_for_probe(path==1,stream)?;
                    control.agree(&[case as u8,rows as u8,repeat,path,0])?;
                    target.begin_verification(&tokens[*prefix..*prefix+rows],comm,stream)?;
                    let warm=logits(cuda,target)?;target.finish_verification(0,comm,stream)?;
                    if let Some(r)=&reference{exact(r,&warm,out,&format!("mhc-c{case}-m{rows}-p{path}-r{repeat}-warm"))?;}
                    else{reference=Some(warm.clone());}
                    let mut seconds=Vec::new();
                    for replay in 1..=3u8 {
                        control.agree(&[case as u8,rows as u8,repeat,path,replay])?;
                        let at=Instant::now();target.begin_verification(&tokens[*prefix..*prefix+rows],comm,stream)?;
                        seconds.push(at.elapsed().as_secs_f64());let observed=logits(cuda,target)?;
                        target.finish_verification(0,comm,stream)?;
                        exact(&warm,&observed,out,&format!("mhc-c{case}-m{rows}-p{path}-r{repeat}-replay{replay}"))?;
                    }
                    ensure!(state(target)?==accepted,"live-row abort changed retained state");
                    let committed_check=if rows==3 {
                        target.begin_verification(&tokens[*prefix..*prefix+rows],comm,stream)?;
                        target.finish_verification(rows,comm,stream)?;
                        let after=state(target)?;
                        target.begin_verification(&tokens[*prefix+rows..*prefix+rows+6],comm,stream)?;
                        let following=logits(cuda,target)?;target.finish_verification(0,comm,stream)?;
                        ensure!(state(target)?==after,"live-row following abort changed accepted state");
                        if let Some((prior,values))=&committed_reference {
                            ensure!(prior==&after,"live-row committed state mismatch");
                            exact(values,&following,out,&format!("mhc-live-following-c{case}-p{path}-r{repeat}"))?;
                        }else{committed_reference=Some((after.clone(),following.clone()));}
                        lineage+=1;target.restore_state_image(&saved,lineage,comm,stream)?;
                        ensure!(state(target)?==accepted,"live-row original frontier restore differs");
                        Some(json!({"accepted":rows,"state":after,"following_sha256":hash(&following)}))
                    }else{None};
                    records.push(json!({"case":case,"id":id,"prefix_tokens":prefix,"rows":rows,"path":path,
                        "repeat":repeat,"seconds":seconds,"logits_sha256":hash(&warm),"state_exact":true,"committed_check":committed_check}));
                    std::fs::write(out.join("mhc-live-partial.json"),serde_json::to_vec(&records)?)?;
                }
            }
            target.set_mhc_live_rows_for_probe(false,stream)?;
            if rows>6{verify.handoff_from(wide)?;}
        }
    }
    std::fs::write(out.join("mhc-live-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,
        "records":records,"fixtures_sha256":[hash(short_raw),hash(long_raw)],"all_logits_and_abort_state_exact":true,
        "scope":"Six fixed-prefix/live-row combinations: ideation M6 with1/3/5/6live and131K M6 with3/6live. Physical versus live statistics and residual rows, reversed order, three timed warm replays. Exact all logits and retained abort state. No sampler, precision or background policy change; no end-to-end generation claim."}))?)?;
    Ok(())
}
