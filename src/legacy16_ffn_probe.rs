//! Revisit FFN overlap after carrying legacy QKV and serial compact attention.
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
fn ffn_taps(cuda:&Cuda,target:&Target)->Result<Vec<(String,Vec<u8>)>> {
    ensure!(target.physical_rows()==16,"bounded FFN replay capture");
    let mut result=Vec::new();
    for(i,layer)in target.layer_owners().iter().enumerate() {
        let ffn=layer.ffn();let(q,scales)=ffn.projection_quantized_input();
        for(name,ptr,bytes)in [
            ("attn-reduced",layer.attn_reduced().ptr(),layer.attn_reduced().bytes()),
            ("ffn-input",layer.ffn_boundary().input().ptr(),layer.ffn_boundary().input().bytes()),
            ("route-logits",ffn.router().logits().ptr(),ffn.router().logits().bytes()),
            ("route-ids",ffn.router().ids().ptr(),ffn.router().ids().bytes()),
            ("route-weights",ffn.router().routing().ptr(),ffn.router().routing().bytes()),
            ("shared-quant",q.ptr(),q.bytes()),("shared-scales",scales.ptr(),scales.bytes()),
            ("shared-up",ffn.shared_up().ptr(),ffn.shared_up().bytes()),
            ("shared-activation",ffn.activation().ptr(),ffn.activation().bytes()),
            ("shared-down",ffn.shared_down().ptr(),ffn.shared_down().bytes()),
            ("ffn-local",ffn.output().ptr(),ffn.output().bytes()),
            ("ffn-reduced",layer.ffn_reduced().ptr(),layer.ffn_reduced().bytes()),
            ("residual",layer.output().ptr(),layer.output().bytes())] {
            ensure!(bytes<=16*20480*4,"FFN tap bound");
            result.push((format!("l{i:02}-{name}"),unsafe{cuda.read_device(ptr,bytes)}?));
        }
    }
    Ok(result)
}
fn retain_changed_layer(output:&Path,label:&str,reference:&[(String,Vec<u8>)],actual:&[(String,Vec<u8>)])->Result<()> {
    ensure!(reference.len()==actual.len(),"FFN capture geometry");
    let mut differences=Vec::new();
    for((name,a),(other,b))in reference.iter().zip(actual) {
        ensure!(name==other && a.len()==b.len(),"FFN capture layout");
        if a!=b {differences.push(json!({"name":name,"changed_bytes":a.iter().zip(b).filter(|(a,b)|a!=b).count(),
            "reference_sha256":digest(a),"observed_sha256":digest(b)}));}
    }
    if let Some(first)=differences.first() {
        let prefix=first["name"].as_str().unwrap().split('-').next().unwrap();
        for((name,a),(_,b))in reference.iter().zip(actual).filter(|((name,_),_)|name.starts_with(&format!("{prefix}-"))) {
            std::fs::write(output.join(format!("failure-{name}-reference.bin")),a)?;
            std::fs::write(output.join(format!("failure-{name}-observed.bin")),b)?;
        }
    }
    std::fs::write(output.join("failure-ffn-taps.json"),serde_json::to_vec_pretty(&json!({
        "label":label,"differences":differences,"scope":"Post-replay bytes from layer-owned FFN inputs, router, shared branch and output. Routed scratch is reused between layers and deliberately omitted. First changed layer operands retained for actual-partner reproduction."}))?)?;
    Ok(())
}
fn compare(output:&Path,label:&str,expected:&[u8],actual:&[u8])->Result<()> {
    if expected!=actual {
        std::fs::write(output.join(format!("{label}-expected.bin")),expected)?;
        std::fs::write(output.join(format!("{label}-actual.bin")),actual)?;
        anyhow::bail!("legacy-QKV FFN mismatch: {label}; raw logits retained");
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn finish_and_capture(cuda:&Cuda,wide:&mut Target,ids:&[u32],comm:&Roce,stream:&Stream,
    output:&Path,label:&str,expected:&[u8],actual:&[u8],taps:bool)->Result<()> {
    // Keep successful execution unchanged. Warm-up and following-prefix checks
    // can fail too; capture their still-live operands before aborting.
    let failed=if taps && expected!=actual {Some(ffn_taps(cuda,wide)?)}else{None};
    wide.finish_verification(0,comm,stream)?;
    if let Some(failed)=failed {
        wide.set_verify16_schedule_for_probe(0,stream)?;
        wide.begin_verification(ids,comm,stream)?;
        let serial=pending(cuda,wide)?;let reference=ffn_taps(cuda,wide)?;
        wide.finish_verification(0,comm,stream)?;
        retain_changed_layer(output,label,&reference,&failed)?;
        std::fs::write(output.join("failure-serial-reference.json"),serde_json::to_vec_pretty(&json!({
            "label":label,"live_rows":ids.len(),"input_ids":ids,
            "logits_match_original_healthy_replay":serial==expected,
            "serial_sha256":digest(&serial),"healthy_sha256":digest(expected),
            "capture_only_after_failure":true}))?)?;
    }
    compare(output,label,expected,actual)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,verify:&mut Target,wide:&mut Target,ids:&[u32],
    kernels:&Path,comm:&Roce,stream:&Stream,output:&Path,rank:usize,ticket:&[u8])->Result<Value> {
    ensure!(ids.len()==16 && wide.position()==274108 && wide.physical_rows()==16,
        "legacy-QKV FFN schedule geometry");
    ensure!(std::env::var("CANADA_SCRATCH_LEGACY16_FFN_SCREEN").as_deref()==Ok("1")
        && wide.narrow_projection_layers()==0,"FFN screen requires carried legacy QKV");
    wide.set_grouped16_for_probe(true,stream)?;
    ensure!(wide.query_overlap_layers()==0 && wide.attention_coefficient_overlap_layers()==0,
        "FFN comparison holds compact attention serial");
    let legacy_shared=std::env::var("CANADA_SCRATCH_LEGACY16_FFN_W13").as_deref()==Ok("1");
    let mut alternate=if legacy_shared {Some(wide.prepare_shared_up_alternates_for_probe(kernels)?)}else{None};
    let ips=std::array::from_fn(|r|std::net::IpAddr::V4(std::net::Ipv4Addr::new(192,168,200,12+r as u8)));
    let mut control=crate::control_channel::ControlChannel::connect(rank,
        std::net::SocketAddr::new(ips[0],18089),ips,ticket,&Sha256::digest(b"legacy16-ffn-v1").into(),
        std::time::Duration::from_secs(60))?;
    let mut reference=std::collections::BTreeMap::new();let mut records=Vec::new();
    let taps=std::env::var("CANADA_SCRATCH_LEGACY16_FFN_TAPS").as_deref()==Ok("1");
    let repeats=if std::env::var("CANADA_SCRATCH_LEGACY16_FFN_STRESS").as_deref()==Ok("1") {8u8}else{2};
    ensure!(repeats==2 || taps,"bounded intermittent-failure stress requires operand capture");
    for repeat in 0..repeats {
        for path in if repeat%2==0 {[0,1]}else{[1,0]} {
            let mask=if path==0 {0}else{3};
            if let Some(other)=alternate.as_mut() {
                wide.set_legacy16_ffn_projection_for_probe(path!=0,mask,other,stream)?;
            } else {wide.set_verify16_schedule_for_probe(mask,stream)?;}
            for rows in [8usize,16] {
                control.agree(&[repeat,path,rows as u8,0])?;
                wide.begin_verification(&ids[..rows],comm,stream)?;
                let logits=pending(cuda,wide)?;
                if path==0 && repeat==0 {reference.insert(rows,logits.clone());}
                finish_and_capture(cuda,wide,&ids[..rows],comm,stream,output,
                    &format!("warm-r{repeat}-p{path}-n{rows}"),reference.get(&rows).unwrap(),&logits,taps)?;
                let before=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
                let mut seconds=Vec::new();
                for replay in 1..=2 {
                    control.agree(&[repeat,path,rows as u8,replay])?;
                    let start=Instant::now();wide.begin_verification(&ids[..rows],comm,stream)?;
                    seconds.push(start.elapsed().as_secs_f64());
                    let observed=pending(cuda,wide)?;
                    finish_and_capture(cuda,wide,&ids[..rows],comm,stream,output,
                        &format!("replay-r{repeat}-p{path}-n{rows}-i{replay}"),&logits,&observed,taps)?;
                }
                let after=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
                ensure!(before==after,"legacy-QKV FFN schedule aborted state changed");
                records.push(json!({"repeat":repeat,"path":path,"rows":rows,"forward_seconds":seconds,
                    "legacy_qkv":true,"legacy_shared_up":legacy_shared&&path!=0,"ffn_overlap":path!=0,"query_overlap":false,
                    "logits_sha256":digest(&logits),"aborted_state_exact":true}));
            }
        }
    }
    // Change the accepted history through the ordinary M6 owner, then compare
    // both schedules after16 committed rows, crossing the earlier failing wrap.
    verify.handoff_from(wide)?;
    for chunk in ids.chunks(6) {
        verify.begin_verification(chunk,comm,stream)?;
        verify.finish_verification(chunk.len(),comm,stream)?;
    }
    wide.handoff_from(verify)?;
    let mut following=Vec::new();let mut first=None;
    for path in 0..2u8 {
        let mask=if path==0 {0}else{3};
        if let Some(other)=alternate.as_mut() {
            wide.set_legacy16_ffn_projection_for_probe(path!=0,mask,other,stream)?;
        } else {wide.set_verify16_schedule_for_probe(mask,stream)?;}
        control.agree(&[255,path,8,0])?;
        let before=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
        wide.begin_verification(&ids[..8],comm,stream)?;
        let logits=pending(cuda,wide)?;
        if path==0 {first=Some(logits.clone());}
        finish_and_capture(cuda,wide,&ids[..8],comm,stream,output,
            &format!("following-p{path}"),first.as_ref().unwrap(),&logits,taps)?;
        let after=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
        ensure!(before==after,"legacy-QKV FFN changed-history abort state");
        following.push(json!({"path":path,"logits_sha256":digest(&logits),"state_exact":true}));
    }
    // No graph may outlive the alternate projection buffers owned by this probe.
    if let Some(other)=alternate.as_mut() {wide.set_legacy16_ffn_projection_for_probe(false,0,other,stream)?;}
    let result=json!({"rank":rank,"legacy16_ffn_screen":true,"legacy_shared_up_variant":legacy_shared,"prompt_tokens":274108,"repeats":repeats,"records":records,
        "following":following,"all_logits_and_state_exact":true,"target_precision_changed":false,
        "scope":"Carried legacy QKV, compact sparse index, serial attention and live-row transport. Serial compact-W13 FFN versus shared-expert plus residual-statistics overlap; legacy_shared_up_variant selects the B12x W13 implementation only on the overlapping path. Counterbalanced pairs at8/16live rows and a changed-history handoff after16 accepted rows through M6. Optional eight-pair stress specifically targets intermittent overlap failure. Exact logits and abort state; no outcome or background panel."});
    std::fs::write(output.join("legacy16-ffn-summary.json"),serde_json::to_vec_pretty(&result)?)?;
    Ok(result)
}
