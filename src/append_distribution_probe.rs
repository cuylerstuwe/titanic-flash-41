//! Three varied accepted prefixes, scoped to the carried append arithmetic.
//! Raw logits are analyzed on GPU; host work is control and byte checks only.
use crate::{control_channel::ControlChannel,cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};
fn hash(x:&[u8])->String {format!("{:x}",Sha256::digest(x))}
fn state(t:&Target)->Result<Value> {Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wide:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,control:&mut ControlChannel,output:&Path,rank:usize)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_APPEND_DISTRIBUTION_SCREEN").as_deref()==Ok("1")
        && std::env::var("CANADA_SCRATCH_MEDIUM_NATIVE_TRANSPORT").as_deref()!=Ok("1")
        && large.position()==0 && verify.physical_rows()==6 && wide.physical_rows()==16,
        "append distribution probe admission");
    let raw=include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let mut cases=Vec::new();
    for id in ["rust-config-parser-reasoning-1","offline-product-ideation-answer-2"] {
        let c=fixture["cases"].as_array().unwrap().iter().find(|c|c["id"]==id).unwrap();
        cases.push((id.to_string(),serde_json::from_value::<Vec<u32>>(c["input_ids"].clone())?));
    }
    let repo_raw=include_bytes!("../tests/fixtures/target-agentic-python-v1.json");
    let repo:Value=serde_json::from_slice(repo_raw)?;
    cases.push(("python-repository-prompt".into(),serde_json::from_value(repo["batches"][0].clone())?));
    let mut lineage=7_460_000;let mut records=Vec::new();let mut image_sizes=Vec::new();
    for (case_index,(id,tokens)) in cases.iter().enumerate() {
        ensure!(tokens.len()>128+22,"append distribution fixture extent");
        let prefix=tokens.len()-22;
        if case_index>0 {
            lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;
        }
        large.prefill_wide(wides,&tokens[..prefix],comm,stream)?;verify.handoff_from(large)?;
        let bytes=verify.compact_state_image_bytes()?;
        ensure!(bytes<=1024usize.pow(3) && cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+bytes,
            "append diagnostic image reserve");
        let image=verify.save_state_image(bytes,comm,stream)?;let incoming=state(verify)?;
        image_sizes.push(bytes);
        for delta in [1usize,6,16] {
            let paths=if delta<=6 {vec![0u8,3,6,16]}else{vec![0u8,3,16]};
            for repeat in 0..2u8 {
                let mut order=paths.clone();if repeat==1 {order.reverse();}
                for path in order {
                    lineage+=1;verify.restore_state_image(&image,lineage,comm,stream)?;
                    ensure!(state(verify)?==incoming,"restored append prefix changed");
                    control.agree(&[case_index as u8,delta as u8,repeat,path])?;
                    let began=Instant::now();
                    let target=match path {
                        0|3=>{
                            large.handoff_from(verify)?;
                            large.set_tiny_native_reduce_for_probe(path==3,stream)?;
                            large.set_tiny_prefix_transport_for_probe(path==3,path==3,stream)?;
                            large.prefill_wide(wides,&tokens[prefix..prefix+delta],comm,stream)?;
                            &mut *large
                        },
                        6=>{verify.set_tiny_extend_for_probe(true,stream)?;
                            verify.begin_verification(&tokens[prefix..prefix+delta],comm,stream)?;
                            verify.finish_verification(delta,comm,stream)?;&mut *verify},
                        16=>{wide.handoff_from(verify)?;wide.set_tiny_extend_for_probe(true,stream)?;
                            wide.begin_verification(&tokens[prefix..prefix+delta],comm,stream)?;
                            wide.finish_verification(delta,comm,stream)?;&mut *wide},
                        _=>unreachable!(),
                    };
                    let append_seconds=began.elapsed().as_secs_f64();
                    let logits=target.logits()?;
                    let append=unsafe{cuda.read_device(logits.ptr()+((delta-1)*129280*4)as u64,129280*4)}?;
                    let accepted=state(target)?;
                    if path==16 {wide.set_tiny_extend_for_probe(false,stream)?;verify.handoff_from(wide)?;}
                    else if path==6 {verify.set_tiny_extend_for_probe(false,stream)?;}
                    else {verify.handoff_from(large)?;}
                    let before=state(verify)?;verify.begin_verification(&tokens[prefix+delta..prefix+delta+6],comm,stream)?;
                    let (logits,live)=verify.pending_logits()?;ensure!(live==6,"append diagnostic follow rows");
                    let follow=unsafe{cuda.read_device(logits.ptr(),6*129280*4)}?;
                    verify.finish_verification(0,comm,stream)?;
                    ensure!(state(verify)?==before,"following verification abort changed accepted state");
                    ensure!(verify.position()==(prefix+delta)as u64,"append distribution frontier");
                    let stem=format!("append-c{case_index}-d{delta}-p{path}-r{repeat}");
                    if rank==0 {
                        std::fs::write(output.join(format!("{stem}-append.bin")),&append)?;
                        std::fs::write(output.join(format!("{stem}-follow.bin")),&follow)?;
                    }
                    records.push(json!({"case":case_index,"id":id,"prefix_tokens":prefix,"delta":delta,"path":path,
                        "repeat":repeat,"append_seconds":append_seconds,"stem":stem,"append_sha256":hash(&append),
                        "follow_sha256":hash(&follow),"accepted_state":accepted,"following_abort_exact":true}));
                }
            }
        }
    }
    std::fs::write(output.join("append-distribution-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"image_bytes":image_sizes,"phase_fixture_sha256":hash(raw),
        "repository_fixture_sha256":hash(repo_raw),"all_frontiers_and_abort_state_checked":true,
        "scope":"Three public prefix phases: Rust reasoning, ideation answer, Python repository prompt. Appends1/6/16; M128NCCL, M128nativecropped, M6BF16, M16BF16 where eligible. Two reversed-order replays and six following verification positions. GPU distribution diagnostics only; no taskquality or throughput claim."}))?)?;
    Ok(())
}
