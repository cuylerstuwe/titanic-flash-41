//! Compare the carried M16 schedule with legacy QKV and coefficient overlap.
use crate::{cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};
fn digest(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}
fn pending(cuda:&Cuda,target:&Target)->Result<Vec<u8>> {
    let(logits,live)=target.pending_logits()?;
    unsafe{cuda.read_device(logits.ptr(),live*129280*4)}
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,verify:&mut Target,wide:&mut Target,ids:&[u32],kernels:&Path,
    comm:&Roce,stream:&Stream,output:&Path,rank:usize,ticket:&[u8])->Result<Value> {
    ensure!(ids.len()==16 && wide.position()==274108 && wide.physical_rows()==16,
        "QKV schedule fixed-prefix geometry");
    let mut other=wide.prepare_qkv_alternates_for_probe(kernels)?;
    let ips=std::array::from_fn(|r|std::net::IpAddr::V4(std::net::Ipv4Addr::new(192,168,200,12+r as u8)));
    let mut control=crate::control_channel::ControlChannel::connect(rank,
        std::net::SocketAddr::new(ips[0],18089),ips,ticket,&Sha256::digest(b"qkv-schedule-v1").into(),
        std::time::Duration::from_secs(60))?;
    let mut reference=std::collections::BTreeMap::new();let mut records=Vec::new();
    for repeat in 0..2u8 {
        for path in if repeat==0 {[0,1,2]}else{[2,1,0]} {
            wide.set_qkv_schedule_for_probe(path!=0,path==2,&mut other,stream)?;
            for rows in [8usize,16] {
                control.agree(&[repeat,path,rows as u8,0])?;
                wide.begin_verification(&ids[..rows],comm,stream)?;
                let logits=pending(cuda,wide)?;wide.finish_verification(0,comm,stream)?;
                if path==0 && repeat==0 {reference.insert(rows,logits.clone());}
                ensure!(reference.get(&rows)==Some(&logits),"QKV schedule warm logits differ");
                let before=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
                let mut seconds=Vec::new();
                for replay in 1..=2 {
                    control.agree(&[repeat,path,rows as u8,replay])?;
                    let start=Instant::now();wide.begin_verification(&ids[..rows],comm,stream)?;
                    seconds.push(start.elapsed().as_secs_f64());
                    let observed=pending(cuda,wide)?;wide.finish_verification(0,comm,stream)?;
                    ensure!(observed==logits,"QKV schedule replay logits differ");
                }
                let after=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
                ensure!(before==after,"QKV schedule aborted state changed");
                records.push(json!({"repeat":repeat,"path":path,"rows":rows,"forward_seconds":seconds,
                    "legacy_qkv":path!=0,"coefficient_overlap":path==2,"query_overlap":false,
                    "logits_sha256":digest(&logits),"aborted_state_exact":true}));
            }
        }
    }
    // Change the accepted history through the ordinary M6 owner, then compare
    // all three schedules once at the new frontier. No wide background panel.
    verify.handoff_from(wide)?;verify.begin_verification(&ids[..6],comm,stream)?;
    verify.finish_verification(6,comm,stream)?;wide.handoff_from(verify)?;
    let mut following=Vec::new();let mut first=None;
    for path in 0..3u8 {
        wide.set_qkv_schedule_for_probe(path!=0,path==2,&mut other,stream)?;
        control.agree(&[3,path,8,0])?;
        let before=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
        wide.begin_verification(&ids[6..14],comm,stream)?;
        let logits=pending(cuda,wide)?;wide.finish_verification(0,comm,stream)?;
        let after=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
        if path==0 {first=Some(logits.clone());}
        ensure!(first.as_ref()==Some(&logits) && before==after,"QKV schedule changed-history handoff");
        following.push(json!({"path":path,"logits_sha256":digest(&logits),"state_exact":true}));
    }
    let result=json!({"rank":rank,"qkv_schedule_screen":true,"prompt_tokens":274108,"records":records,
        "following":following,"all_logits_and_state_exact":true,"target_precision_changed":false,
        "scope":"Current compact-QKV/serial attention versus legacy-QKV/serial and legacy-QKV/coefficient overlap. Compact sparse index held on, query overlap off. Two reversed-order pairs at8/16live rows and one changed-history handoff. A host barrier precedes each timed launch; no layer captures. No outcome-quality or production claim."});
    std::fs::write(output.join("qkv-schedule-summary.json"),serde_json::to_vec_pretty(&result)?)?;
    Ok(result)
}
