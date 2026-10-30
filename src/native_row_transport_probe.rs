//! Paired native live-row transport screen, with bounded first-layer diagnostics.
use crate::{cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};
fn hash(x:&[u8])->String {format!("{:x}",Sha256::digest(x))}
fn state(t:&Target)->Result<Value> {Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))}
fn pending(cuda:&Cuda,t:&Target)->Result<Vec<u8>> {
    let (p,live)=t.pending_logits()?;unsafe{cuda.read_device(p.ptr(),live*129280*4)}
}
fn trace(cuda:&Cuda,t:&Target,output:&Path,delta:usize,path:usize)->Result<Value> {
    let owners=t.layer_owners();let l=owners[0];let rows=t.physical_rows();
    let mut raw=Vec::new();let mut entries=Vec::new();
    let mut take=|name:&str,ptr:u64,bytes:usize,dtype:&str|->Result<()> {
        let offset=raw.len();raw.extend(unsafe{cuda.read_device(ptr,bytes)}?);
        entries.push(json!({"name":name,"offset":offset,"bytes":bytes,"dtype":dtype}));Ok(())
    };
    take("input",l.attn_boundary().input().ptr(),5120*2,"bf16")?;
    for(name,ptr,bytes)in l.attention().local_qkv_replay_taps() {
        if name=="qkv-scales" {continue;}
        take(name,ptr,bytes/rows,if name=="qkv-quant" {"u8"}else{"bf16"})?;
    }
    for(name,ptr,bytes,_)in l.attention().taps() {
        if ["norm_scales","woa_scales","woa_partial","woa"].contains(&name) {continue;}
        take(name,ptr,bytes/rows,if ["norm_quant","woa_quant"].contains(&name) {"u8"}else{"bf16"})?;
    }
    take("attention_reduced",l.attn_reduced().ptr(),5120*2,"bf16")?;
    let name=format!("row-transport-d{delta}-p{path}-attention.bin");
    std::fs::write(output.join(&name),&raw)?;
    Ok(json!({"delta":delta,"path":path,"file":name,"sha256":hash(&raw),"entries":entries}))
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wide:&mut Target,ids:&[u32],
    comm:&Roce,stream:&Stream,output:&Path,rank:usize,ticket:&[u8])->Result<Value> {
    ensure!(std::env::var("CANADA_SCRATCH_NATIVE_ROW_TRANSPORT_SCREEN").as_deref()==Ok("1")
        && ids.len()==16 && wide.position()==274108,"native row transport screen geometry");
    wide.set_grouped16_for_probe(true,stream)?;
    ensure!(wide.query_overlap_layers()==0 && wide.attention_coefficient_overlap_layers()==0,
        "native transport screen uses serial candidate attention");
    let ips=std::array::from_fn(|r|std::net::IpAddr::V4(std::net::Ipv4Addr::new(192,168,200,12+r as u8)));
    let mut control=crate::control_channel::ControlChannel::connect(rank,
        std::net::SocketAddr::new(ips[0],18089),ips,ticket,&Sha256::digest(b"native-row-transport-v1").into(),
        std::time::Duration::from_secs(60))?;
    let mut reference=std::collections::BTreeMap::new();let mut records=Vec::new();
    let timing_only=std::env::var("CANADA_SCRATCH_NATIVE_ROW_TRANSPORT_TIMING_ONLY").as_deref()==Ok("1");
    let rounds=if timing_only {4u8}else{2};
    let replays=if timing_only {8u8}else{2};
    for capacity in if timing_only {vec![6usize]}else{vec![6usize,16]} {
        if capacity==6 {verify.handoff_from(wide)?;}else{wide.handoff_from(verify)?;}
        let target=if capacity==6 {&mut *verify}else{&mut *wide};
        for round in 0..rounds {
            for variant in if round%2==0 {[0u8,1]}else{[1,0]} {
                target.set_tiny_prefix_transport_for_probe(variant==1,variant==1,stream)?;
                for live in if timing_only {vec![3usize]}else if capacity==6 {vec![1usize,3,6]}else{vec![8,16]} {
                    let before=state(target)?;
                    control.agree(&[capacity as u8,round,variant,live as u8,0])?;
                    target.begin_verification(&ids[..live],comm,stream)?;
                    let values=pending(cuda,target)?;target.finish_verification(0,comm,stream)?;
                    if round==0 && variant==0 {reference.insert((capacity,live),values.clone());}
                    ensure!(reference.get(&(capacity,live))==Some(&values),"native transport live logits changed");
                    let mut seconds=Vec::new();
                    for replay in 1..=replays {
                        control.agree(&[capacity as u8,round,variant,live as u8,replay])?;
                        let start=Instant::now();target.begin_verification(&ids[..live],comm,stream)?;
                        seconds.push(start.elapsed().as_secs_f64());
                        ensure!(pending(cuda,target)?==values,"native transport replay changed logits");
                        target.finish_verification(0,comm,stream)?;
                    }
                    ensure!(state(target)?==before,"native transport rejected state changed");
                    records.push(json!({"capacity":capacity,"live":live,"round":round,"variant":variant,
                        "seconds":seconds,"logits_sha256":hash(&values),"aborted_state_exact":true}));
                }
            }
        }
    }
    if timing_only {
        let result=json!({"rank":rank,"native_row_transport_screen":true,"timing_only":true,
            "prefix_tokens":274108,"records":records,"all_logits_and_state_exact":true,
            "target_precision_changed":false,"scope":"Only the ambiguous M6/live3 timing decision: four alternating-order pairs, eight timed replays per variant. Exact logits and abort state; no new attention, outcome or background panel."});
        std::fs::write(output.join("native-row-transport-summary.json"),serde_json::to_vec_pretty(&result)?)?;
        return Ok(result)
    }
    // Capture only the first attention layer at two append lengths. This
    // distinguishes projection, RoPE, attention and output projection before
    // any compression or decoder-window differences. Outside measured work.
    verify.handoff_from(wide)?;
    let bytes=verify.compact_state_image_bytes()?;
    ensure!(bytes<=4*1024usize.pow(3) && cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+bytes,
        "transport trace image must preserve the operating reserve");
    let saved=verify.save_state_image(bytes,comm,stream)?;
    let incoming=state(verify)?;let mut lineage=990000;let mut traces=Vec::new();
    verify.set_tiny_prefix_transport_for_probe(false,false,stream)?;
    for delta in [1usize,6] {
        for path in [1usize,3] {
            lineage+=1;verify.restore_state_image(&saved,lineage,comm,stream)?;
            ensure!(state(verify)?==incoming,"attention trace incoming state changed");
            verify.set_tiny_extend_for_probe(path==1,stream)?;
            control.agree(&[40,delta as u8,path as u8])?;
            if path==1 {
                verify.begin_verification(&ids[..delta],comm,stream)?;verify.finish_verification(delta,comm,stream)?;
                traces.push(trace(cuda,verify,output,delta,path)?);
            }else{
                large.handoff_from(verify)?;large.set_tiny_native_reduce_for_probe(true,stream)?;
                large.append(&ids[..delta],comm,stream)?;
                traces.push(trace(cuda,large,output,delta,path)?);verify.handoff_from(large)?;
            }
        }
    }
    lineage+=1;verify.restore_state_image(&saved,lineage,comm,stream)?;
    verify.set_tiny_extend_for_probe(false,stream)?;
    verify.set_tiny_prefix_transport_for_probe(true,true,stream)?;
    verify.begin_verification(&ids[..3],comm,stream)?;verify.finish_verification(3,comm,stream)?;
    wide.handoff_from(verify)?;
    let mut first=None;let mut following=Vec::new();
    for variant in 0..=1u8 {
        wide.set_tiny_prefix_transport_for_probe(variant==1,variant==1,stream)?;
        let before=state(wide)?;control.agree(&[50,variant])?;
        wide.begin_verification(&ids[3..11],comm,stream)?;
        let values=pending(cuda,wide)?;wide.finish_verification(0,comm,stream)?;
        if variant==0 {first=Some(values.clone());}
        ensure!(first.as_ref()==Some(&values) && state(wide)?==before,"native transport committed-history handoff changed");
        following.push(json!({"variant":variant,"logits_sha256":hash(&values),"state_exact":true}));
    }
    let result=json!({"rank":rank,"native_row_transport_screen":true,"prefix_tokens":274108,
        "records":records,"following":following,"attention_traces":traces,"trace_image_bytes":bytes,
        "all_logits_and_state_exact":true,"target_precision_changed":false,
        "scope":"Current M6 and serial/compact-index legacy-QKV M16. Full versus live-row sums/gathers at1/3/6 and8/16live rows, two reversed orders with two measured replays each. Exact logits/rollback and following M16after3accepted M6rows. Separate untimed layer0 row0 traces for M6 versus M128 native append at1/6tokens. No background, full outcome or production panel."});
    std::fs::write(output.join("native-row-transport-summary.json"),serde_json::to_vec_pretty(&result)?)?;Ok(result)
}
