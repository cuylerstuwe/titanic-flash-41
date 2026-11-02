//! Native M16 integration, with M6 guarding the generalized gather ABI.
use crate::{control_channel::ControlChannel,cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::BTreeMap,path::Path,time::Instant};
fn hash(x:&[u8])->String{format!("{:x}",Sha256::digest(x))}
fn state(t:&Target)->Result<Value>{Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))}
fn logits(cuda:&Cuda,t:&Target)->Result<Vec<u8>>{let(y,rows)=t.pending_logits()?;unsafe{cuda.read_device(y.ptr(),rows*129280*4)}}
fn exact(a:&[u8],b:&[u8],out:&Path,stem:&str)->Result<()> {
    if a!=b {
        std::fs::write(out.join(format!("{stem}-reference.bin")),a)?;
        std::fs::write(out.join(format!("{stem}-observed.bin")),b)?;
        anyhow::bail!("M16 key partition changed logits: {stem}");
    }Ok(())
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wide:&mut Target,
    wides:&mut[Target],comm:&Roce,stream:&Stream,control:&mut ControlChannel,
    out:&Path,rank:usize)->Result<()> {
    ensure!(large.position()==0&&verify.physical_rows()==6&&wide.physical_rows()==16
        &&std::env::var("CANADA_SCRATCH_KEY_SHARDED16_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_KEY_SHARD_AUTO_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_KEY_SHARDED_PUBLISHER_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_COMPACT_STATE_IMAGE").as_deref()==Ok("1"),
        "M16 key partition probe admission");
    let raw_a=include_bytes!("../tests/fixtures/long-history-context-v1.json");
    let raw_b=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let a:Value=serde_json::from_slice(raw_a)?;let b:Value=serde_json::from_slice(raw_b)?;
    let mut tokens:Vec<u32>=serde_json::from_value(a["cases"][0]["prompt_ids"].clone())?;
    tokens.extend(serde_json::from_value::<Vec<u32>>(b["tokens"].clone())?);
    ensure!(tokens.len()>524288+128,"M16 key partition fixture bounds");
    let mut lineage=8_220_000u64;let mut records=Vec::new();let mut transitions=Vec::new();
    for(case,prefix)in [513usize,524288,513].into_iter().enumerate(){
        if case>0 {
            if prefix<verify.position()as usize {lineage+=1;verify.reset(lineage,comm,stream)?;}
            large.handoff_from(verify)?;
        }
        large.prefill_wide(wides,&tokens[large.position()as usize..prefix],comm,stream)?;
        verify.handoff_from(large)?;
        let size=verify.compact_state_image_bytes()?;
        ensure!(size<=2*1024usize.pow(3)&&control.all_admit(
            cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+size)?,
            "M16 key partition snapshot reserve");
        let saved=verify.save_state_image(size,comm,stream)?;let accepted_state=state(verify)?;
        for physical in [6,16] {
            if physical==16{wide.handoff_from(verify)?;}
            let target=if physical==6{&mut *verify}else{&mut *wide};
            if case==0{target.set_key_sharded_index_for_probe(true,stream)?;}
            target.begin_verification(&tokens[prefix..prefix+physical],comm,stream)?;
            ensure!(target.active_key_sharded_layers()==if prefix>=524288{4}else{0},
                "M16 key partition automatic transition");
            let automatic=logits(cuda,target)?;target.finish_verification(0,comm,stream)?;
            ensure!(state(target)?==accepted_state,"M16 key partition automatic abort");
            transitions.push(json!({"case":case,"prefix_tokens":prefix,"physical_rows":physical,
                "active_layers":target.active_key_sharded_layers(),"logits_sha256":hash(&automatic)}));
            // Reset only checks automatic deactivation; it does not repeat the
            // unchanged short-context timing matrix already measured above.
            if case==2 {
                if physical==16{verify.handoff_from(wide)?;}
                continue;
            }
            let lives=if physical==6{vec![6]}else{vec![8,16]};
            let mut reference:BTreeMap<usize,(Vec<u8>,Vec<u8>,Value)>=BTreeMap::new();
            for(trial,enabled)in [false,true,true,false].into_iter().enumerate(){
                for &live in &lives {
                    lineage+=1;target.restore_state_image(&saved,lineage,comm,stream)?;
                    target.set_key_sharded_index_for_probe(enabled,stream)?;
                    // Warm both vendor capture and gathered keys. Timing uses
                    // uninterrupted normal cadence on the same allocations.
                    for _ in 0..4{target.begin_verification(&tokens[prefix..prefix+live],comm,stream)?;
                        target.finish_verification(0,comm,stream)?;}
                    control.agree(&[case as u8,physical as u8,live as u8,trial as u8])?;
                    let mut seconds=Vec::new();
                    for _ in 0..8 {
                        let at=Instant::now();target.begin_verification(&tokens[prefix..prefix+live],comm,stream)?;
                        seconds.push(at.elapsed().as_secs_f64());target.finish_verification(0,comm,stream)?;
                    }
                    target.begin_verification(&tokens[prefix..prefix+live],comm,stream)?;
                    let first=logits(cuda,target)?;
                    let accepted=if physical==16{live-7}else{3};
                    target.finish_verification(accepted,comm,stream)?;
                    target.begin_verification(&tokens[prefix+48..prefix+48+live],comm,stream)?;
                    let following=logits(cuda,target)?;target.finish_verification(0,comm,stream)?;
                    let after=state(target)?;
                    ensure!(target.position()==(prefix+accepted)as u64,"M16 partition partial commit");
                    if let Some((a,b,s))=reference.get(&live) {
                        let stem=format!("key16-c{case}-m{physical}-l{live}-t{trial}");
                        exact(a,&first,out,&stem)?;exact(b,&following,out,&(stem+"-changed"))?;
                        ensure!(s==&after,"M16 partition changed retained partial/abort state");
                    }else{reference.insert(live,(first.clone(),following.clone(),after.clone()));}
                    records.push(json!({"case":case,"prefix_tokens":prefix,"physical_rows":physical,
                        "live":live,"trial":trial,"enabled":enabled,"seconds":seconds,
                        "first_logits_sha256":hash(&first),"changed_logits_sha256":hash(&following),
                        "accepted":accepted,"state":after,"all_exact":true}));
                    std::fs::write(out.join("key16-partial.json"),serde_json::to_vec(&records)?)?;
                    eprintln!("key16 rank{rank} prefix{prefix} M{physical}/{live} path{enabled} trial{trial}: exact");
                }
            }
            lineage+=1;target.restore_state_image(&saved,lineage,comm,stream)?;
            target.set_key_sharded_index_for_probe(true,stream)?;
            target.begin_verification(&tokens[prefix..prefix+physical],comm,stream)?;
            exact(&automatic,&logits(cuda,target)?,out,&format!("key16-c{case}-m{physical}-final"))?;
            target.finish_verification(0,comm,stream)?;
            ensure!(state(target)?==accepted_state,"M16 final rollback");
            if physical==16{verify.handoff_from(wide)?;}
        }
    }
    std::fs::write(out.join("key16-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"transitions":transitions,"all_exact":true,
        "fixture_sha256":[hash(raw_a),hash(raw_b)],
        "scope":"Same-load M6/M16 source-key partition integration at513,524288,reset513. ABBA with four excluded warm steps then eight uninterrupted timing steps. Partial acceptance followed by changed-proposal abort exercises mutable gathered tail. Exact logits and retained state; no HTTP, sampler or background-policy change; not serving TPS."}))?)?;
    Ok(())
}
