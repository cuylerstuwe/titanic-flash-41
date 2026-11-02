//! First-use verification: retain eager output while recording future replay.
//! Fixed allocations, reversed controls, changed proposals and partial commits.
use crate::{control_channel::ControlChannel,cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};

fn hash(bytes:&[u8])->String { format!("{:x}",Sha256::digest(bytes)) }
fn state(t:&Target)->Result<Value> {
    Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))
}
fn pending(cuda:&Cuda,t:&Target,stream:&Stream)->Result<(Vec<u8>,Vec<String>)> {
    let (logits,live)=t.pending_logits()?;
    let bytes=unsafe{cuda.read_device(logits.ptr(),live*129280*4)}?;
    let (features,feature_live)=t.dspark_feature_inputs(stream)?;
    ensure!(live==feature_live,"capture feature frontier");
    let hashes=features.into_iter().map(|f| {
        ensure!(f.bytes()%t.physical_rows()==0,"capture feature row geometry");
        Ok(hash(&unsafe{cuda.read_device(f.ptr(),f.bytes()/t.physical_rows()*live)}?))
    }).collect::<Result<Vec<_>>>()?;
    Ok((bytes,hashes))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wide:&mut Target,
    wides:&mut[Target],comm:&Roce,stream:&Stream,control:&mut ControlChannel,
    out:&Path,rank:usize)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_VERIFY_CAPTURE_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_COMPACT_STATE_IMAGE").as_deref()==Ok("1")
        &&large.position()==0&&verify.physical_rows()==6&&wide.physical_rows()==16,
        "verification capture screen admission");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let value:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(value["tokens"].clone())?;
    ensure!(tokens.len()>65537+128&&tokens.iter().all(|&id|id<129280),"capture fixture bounds");
    let mut lineage=8_210_000u64;
    let mut records=Vec::new();
    for (case,prefix) in [513usize,65537].into_iter().enumerate() {
        if case>0 {
            lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;
        }
        large.prefill_wide(wides,&tokens[..prefix],comm,stream)?;verify.handoff_from(large)?;
        let size=verify.compact_state_image_bytes()?;
        ensure!(size<=1024usize.pow(3)&&control.all_admit(
            cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+size)?,
            "capture snapshot memory reserve");
        let saved=verify.save_state_image(size,comm,stream)?;
        let accepted_state=state(verify)?;
        let shapes=if case==0{[(6,3,1),(6,6,0)]}else{[(6,6,5),(16,16,9)]};
        for (physical,live,accepted) in shapes {
            if physical==16{wide.handoff_from(verify)?;}
            let target=if physical==16{&mut *wide}else{&mut *verify};
            let mut reference:Option<(Vec<Vec<u8>>,Vec<Value>)>=None;
            for (trial,enabled) in [false,true,true,false].into_iter().enumerate() {
                lineage+=1;target.restore_state_image(&saved,lineage,comm,stream)?;
                ensure!(state(target)?==accepted_state,"capture incoming restore differs");
                target.set_verification_capture_once_for_probe(enabled,stream)?;
                let mut outputs=Vec::new();let mut observations=Vec::new();
                let mut timings=Vec::new();let mut metrics=Vec::new();
                // First use exercises partial acceptance or abort. Two warm
                // replays then reject changed proposals and commit another
                // changed suffix, exposing stale carry or feature publication.
                for step in 0..3 {
                    let ids=&tokens[prefix+step*32..prefix+step*32+live];
                    let count=match step{0=>accepted,1=>0,_=>live};
                    control.agree(&[case as u8,physical as u8,live as u8,trial as u8,step as u8])?;
                    let before=state(target)?;
                    let at=Instant::now();target.begin_verification(ids,comm,stream)?;
                    timings.push(at.elapsed().as_secs_f64());
                    let metric=serde_json::to_value(target.verification_metrics()?)?;
                    ensure!(metric["captured_new_graph"]==(step==0),"capture/replay coverage mismatch");
                    let (bytes,features)=pending(cuda,target,stream)?;
                    target.finish_verification(count,comm,stream)?;
                    let after=state(target)?;
                    if count==0{ensure!(before==after,"capture rejection changed retained state");}
                    ensure!(target.position()==prefix as u64+if step<2{accepted}else{accepted+live}as u64,
                        "capture accepted frontier");
                    observations.push(json!({"features":features,"state":after,"accepted":count}));
                    outputs.push(bytes);metrics.push(metric);
                }
                if let Some((expected,expected_state))=&reference {
                    for (step,(a,b)) in expected.iter().zip(&outputs).enumerate() {
                        if a!=b {
                            let stem=format!("verify-capture-c{case}-m{physical}-l{live}-t{trial}-s{step}");
                            std::fs::write(out.join(format!("{stem}-reference.bin")),a)?;
                            std::fs::write(out.join(format!("{stem}-observed.bin")),b)?;
                            anyhow::bail!("verification capture changed logits; paired bytes saved");
                        }
                    }
                    ensure!(&observations==expected_state,"capture changed feature/retained state");
                }else{reference=Some((outputs.clone(),observations.clone()));}
                records.push(json!({"prefix_tokens":prefix,"physical_rows":physical,"live":live,
                    "first_accepted":accepted,"trial":trial,"enabled":enabled,"seconds":timings,
                    "metrics":metrics,"logits_sha256":outputs.iter().map(|v|hash(v)).collect::<Vec<_>>(),
                    "observations":observations,"all_exact":true}));
                std::fs::write(out.join("verify-capture-partial.json"),serde_json::to_vec(&records)?)?;
                eprintln!("verification capture rank{rank} prefix{prefix} M{physical}/{live} path{enabled} trial{trial}: exact");
            }
            lineage+=1;target.restore_state_image(&saved,lineage,comm,stream)?;
            if physical==16{verify.handoff_from(wide)?;}
        }
    }
    std::fs::write(out.join("verify-capture-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"fixture_sha256":hash(raw),"all_exact":true,
        "scope":"Four prefix/profile/live-row cases; ABBA on identical restored state and fixed allocations. First capture partial acceptance or abort, changed-proposal warm abort, changed-proposal full commit. All target FP32 logits, DSpark feature hashes and retained carry/encoder states compared. First-use costs separated from warm execution; no sampler, precision, HTTP or background-policy change."}))?)?;
    Ok(())
}
