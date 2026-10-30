//! Narrow retained append with the canonical EXTEND decoder floor.
//! CPU work is metadata and raw-byte evidence only; all model math stays GPU.
use crate::{cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};
fn hash(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}
fn last_logits(cuda:&Cuda,target:&Target,live:usize)->Result<Vec<u8>> {
    let logits=target.logits()?;
    unsafe{cuda.read_device(logits.ptr()+((live-1)*129280*4)as u64,129280*4)}
}
fn state(target:&Target)->Result<Value> {
    Ok(json!({"encoder":target.encoder_tail_digest()?,"carry":target.carry_digests()?}))
}
// One valid row per boundary, after measured work. No CPU numerical oracle.
fn trace_first_row(cuda:&Cuda,target:&Target,output:&Path,path:usize)->Result<()> {
    let mut raw=Vec::new();let mut entries=Vec::new();
    for (layer,owner)in target.layer_owners().iter().enumerate() {
        for (name,buffer)in [("input",owner.attn_boundary().input()),("attention",owner.attn_reduced()),
            ("ffn",owner.ffn_reduced()),("output",owner.output())] {
            ensure!(buffer.len()%target.physical_rows()==0,"append boundary row geometry");
            let bytes=buffer.bytes()/target.physical_rows();let offset=raw.len();
            raw.extend(unsafe{cuda.read_device(buffer.ptr(),bytes)}?);
            entries.push(json!({"layer":layer,"name":name,"offset":offset,"bytes":bytes}));
        }
    }
    let name=format!("tiny-append-d1-p{path}-boundaries.bin");
    std::fs::write(output.join(&name),&raw)?;
    std::fs::write(output.join(format!("tiny-append-d1-p{path}-boundaries.json")),
        serde_json::to_vec_pretty(&json!({"path":path,"file":name,"sha256":hash(&raw),"entries":entries}))?)?;
    Ok(())
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,output:&Path,rank:usize,mut extend:Option<&mut Target>,ticket:&[u8])->Result<Value> {
    ensure!(std::env::var("CANADA_SCRATCH_TINY_APPEND_SCREEN").as_deref()==Ok("1")
        && std::env::var("CANADA_SCRATCH_COMPACT_STATE_IMAGE").as_deref()==Ok("1")
        && large.position()==0 && large.physical_rows()==128 && verify.physical_rows()==6
        && large.context_capacity()==1048576,"tiny append screen admission");
    let retained=std::env::var("CANADA_SCRATCH_TINY_APPEND_RETAIN_GRAPHS").as_deref()==Ok("1");
    let native128=std::env::var("CANADA_SCRATCH_TINY_APPEND_NATIVE128").as_deref()==Ok("1");
    let prefix_sum=std::env::var("CANADA_SCRATCH_TINY_APPEND_PREFIX_SUM").as_deref()==Ok("1");
    let prefix_gather=std::env::var("CANADA_SCRATCH_TINY_APPEND_PREFIX_GATHER").as_deref()==Ok("1");
    let medium_transport=std::env::var("CANADA_SCRATCH_MEDIUM_NATIVE_TRANSPORT_SCREEN").as_deref()==Ok("1");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let ids:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    const PREFIX:usize=65536;
    ensure!(ids.len()>PREFIX+134,"tiny append fixture extent");
    let ips=std::array::from_fn(|r|std::net::IpAddr::V4(std::net::Ipv4Addr::new(192,168,200,12+r as u8)));
    let mut control=crate::control_channel::ControlChannel::connect(rank,
        std::net::SocketAddr::new(ips[0],18089),ips,ticket,&Sha256::digest(b"tiny-append-transition-v1").into(),
        std::time::Duration::from_secs(60))?;
    large.prefill_wide(wides,&ids[..PREFIX],comm,stream)?;
    verify.handoff_from(large)?;
    let bytes=verify.compact_state_image_bytes()?;
    ensure!(bytes<=1024usize.pow(3) && cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+bytes,
        "tiny append image must preserve the operating reserve");
    let saved=verify.save_state_image(bytes,comm,stream)?;
    let incoming=state(verify)?;
    let mut paths=if extend.is_some(){vec![0,1,2]}else{vec![0,1]};
    if native128 {paths.push(3);}
    if prefix_sum {ensure!(native128 && retained && extend.is_none(),"native prefix screen controls");paths.extend([4,5]);}
    if prefix_gather {ensure!(prefix_sum,"prefix gather requires sum controls");paths.push(8);}
    if medium_transport {ensure!(prefix_gather && extend.is_none(),"medium transport controls");paths=vec![0,3,4,8];}
    let deltas=if medium_transport {vec![17usize,32,64,127]}else{vec![1usize,6]};
    let mut records=Vec::new();let mut follows=Vec::new();let mut transitions=Vec::new();let mut lineage=112000;
    for (delta_index,&delta) in deltas.iter().enumerate() {
        let mut order=paths.clone();if delta_index%2==1 {order.reverse();}
        for path in order {
            let large_path=[0,3,4,8].contains(&path);
            let verify_path=[1,5].contains(&path);
            for repeat in 0..3 {
                lineage+=1;verify.restore_state_image(&saved,lineage,comm,stream)?;
                ensure!(state(verify)?==incoming,"tiny append incoming state changed");
                control.agree(&[delta as u8,path as u8,repeat,0])?;
                let start=Instant::now();
                verify.set_tiny_extend_for_probe(verify_path,stream)?;
                if prefix_sum {verify.set_tiny_prefix_transport_for_probe(path==5,prefix_gather && path==5,stream)?;}
                if large_path {
                    large.handoff_from(verify)?;
                    if native128 {large.set_tiny_native_reduce_for_probe([3,4].contains(&path),stream)?;}
                    if prefix_sum {large.set_tiny_prefix_transport_for_probe(path==4,prefix_gather && [4,8].contains(&path),stream)?;}
                } else if path==2 {extend.as_mut().unwrap().handoff_from(verify)?;}
                let switch_seconds=start.elapsed().as_secs_f64();
                let captured_before=match path {0|3|4|8=>large.capture_statistics().1,
                    2=>extend.as_ref().unwrap().capture_statistics().1,_=>verify.capture_statistics().1};
                let start=Instant::now();
                let (seconds,phases,logits,accepted_state)=if large_path {
                    large.prefill_wide(wides,&ids[PREFIX..PREFIX+delta],comm,stream)?;
                    let seconds=start.elapsed().as_secs_f64();
                    let phases=serde_json::to_value(large.append_metrics()?)?;
                    let logits=last_logits(cuda,large,delta)?;let accepted_state=state(large)?;
                    verify.handoff_from(large)?;
                    (seconds,phases,logits,accepted_state)
                } else if verify_path {
                    verify.begin_verification(&ids[PREFIX..PREFIX+delta],comm,stream)?;
                    verify.finish_verification(delta,comm,stream)?;
                    let seconds=start.elapsed().as_secs_f64();
                    let logits=last_logits(cuda,verify,delta)?;
                    (seconds,Value::Null,logits,state(verify)?)
                } else {
                    let target=extend.as_mut().unwrap();
                    target.append(&ids[PREFIX..PREFIX+delta],comm,stream)?;
                    let seconds=start.elapsed().as_secs_f64();
                    let phases=serde_json::to_value(target.append_metrics()?)?;
                    let logits=last_logits(cuda,target,delta)?;let accepted_state=state(target)?;
                    verify.handoff_from(target)?;
                    (seconds,phases,logits,accepted_state)
                };
                ensure!(verify.position()==(PREFIX+delta)as u64,"tiny append publication frontier");
                let new_captures=match path {0|3|4|8=>large.capture_statistics().1-captured_before,
                    2=>extend.as_ref().unwrap().capture_statistics().1-captured_before,
                    _=>verify.capture_statistics().1-captured_before};
                ensure!(repeat==0 || new_captures==0,"tiny append measured graph recapture");
                let name=format!("tiny-append-d{delta}-p{path}-r{repeat}-logits.bin");
                if rank==0 {std::fs::write(output.join(&name),&logits)?;}
                records.push(json!({"delta":delta,"path":path,"repeat":repeat,"warmup":repeat==0,
                    "reduction":if [0,8].contains(&path) {"prefill NCCL BF16"}else{"native FP32"},
                    "seconds":seconds,"switch_seconds":switch_seconds,"phases":phases,"new_captures":new_captures,"logits_file":name,
                    "logits_sha256":hash(&logits),"state":accepted_state,"position":verify.position()}));
                if prefix_sum && !prefix_gather && delta==1 && repeat==2 && rank==0 && [0,1,3].contains(&path) {
                    trace_first_row(cuda,if large_path {large}else{verify},output,path)?;
                }
                if retained || repeat==2 {
                    // Real transitions alternate append and ordinary decoding.
                    // Exclude diagnostic DtoH reads/hashes from measured work.
                    let before=state(verify)?;
                    control.agree(&[delta as u8,path as u8,repeat,1])?;
                    let start=Instant::now();verify.set_tiny_extend_for_probe(false,stream)?;
                    let return_switch_seconds=start.elapsed().as_secs_f64();
                    let captured_before=verify.capture_statistics().1;
                    let start=Instant::now();verify.begin_verification(&ids[PREFIX+delta..PREFIX+delta+6],comm,stream)?;
                    let forward_seconds=start.elapsed().as_secs_f64();
                    let (values,live)=verify.pending_logits()?;ensure!(live==6,"tiny append follow extent");
                    let following_raw=unsafe{cuda.read_device(values.ptr(),live*129280*4)}?;
                    let start=Instant::now();verify.finish_verification(0,comm,stream)?;
                    let abort_seconds=start.elapsed().as_secs_f64();
                    let new_captures=verify.capture_statistics().1-captured_before;
                    ensure!(!retained || repeat==0 || new_captures==0,"ordinary verify recaptured after append");
                    ensure!(before==state(verify)? && verify.position()==(PREFIX+delta)as u64,"following verification abort changed state");
                    transitions.push(json!({"delta":delta,"path":path,"repeat":repeat,
                        "switch_seconds":return_switch_seconds,"forward_seconds":forward_seconds,"abort_seconds":abort_seconds,
                        "new_captures":new_captures,"logits_sha256":hash(&following_raw),"aborted_follow_state_exact":true}));
                    if repeat==2 {
                        let name=format!("tiny-append-d{delta}-p{path}-following.bin");
                        if rank==0 {std::fs::write(output.join(&name),&following_raw)?;}
                        follows.push(json!({"delta":delta,"path":path,"logits_file":name,"logits_sha256":hash(&following_raw),
                            "position":verify.position(),"aborted_follow_state_exact":true,"aborted_follow_position_exact":true}));
                    }
                }
            }
        }
    }
    let result=json!({"rank":rank,"prefix_tokens":PREFIX,"fixture_sha256":hash(raw),"image_bytes":bytes,
        "phase_screen":extend.is_some(),"prefix_sum_screen":prefix_sum,"prefix_gather_screen":prefix_gather,"native128_screen":native128,"retained_graph_screen":retained,"paths":paths,"deltas":deltas,"medium_transport_screen":medium_transport,
        "records":records,"following":follows,"transitions":transitions,"all_frontiers_exact":true,
        "scope":"One65K accepted prefix; lengths and paths explicitly listed. Path0=M128 EXTEND/NCCL BF16,path1=M6 Verify/EXTEND decoder floor/native FP32,optional path2=M6 EXTEND/native FP32,path3=M128 EXTEND/native FP32. Path4=M128 native with live-prefix sums,path5=M6 native with live-prefix sums. Gather screen crops gathers in paths4/5 and adds path8=M128 NCCL with cropped gathers. One warmup and two replays per path; order alternates across lengths, host barriers before measured launches. Every retained-mode append is followed by ordinary6-position verification and exact abort-state check. First-use and warm times separate; diagnostic copies/hashes excluded. No drafter, background, outcome-quality or production qualification."});
    std::fs::write(output.join("tiny-append-summary.json"),serde_json::to_vec_pretty(&result)?)?;
    Ok(result)
}
