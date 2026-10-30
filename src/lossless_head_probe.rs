//! Full-model comparison of two exactly equivalent target-head encodings.
use crate::{control_channel::ControlChannel,cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};
fn hash(x:&[u8])->String {format!("{:x}",Sha256::digest(x))}
fn state(t:&Target)->Result<Value> {Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))}
fn logits(cuda:&Cuda,t:&Target)->Result<Vec<u8>> {
    let(y,rows)=t.pending_logits()?;unsafe{cuda.read_device(y.ptr(),rows*129280*4)}
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,control:&mut ControlChannel,output:&Path,rank:usize)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_LOSSLESS_HEAD_SCREEN").as_deref()==Ok("1")
        && large.position()==0 && verify.physical_rows()==6,"lossless head probe admission");
    let raw=include_bytes!("../tests/fixtures/target-agentic-python-v1.json");
    let repo:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(repo["batches"][0].clone())?;
    let long_raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let long:Value=serde_json::from_slice(long_raw)?;
    let long_tokens:Vec<u32>=serde_json::from_value(long["tokens"].clone())?;
    let cases=[("python-repository",tokens.len()-6,tokens),("long-agentic-context",131072,long_tokens)];
    let mut records=Vec::new();
    for (case,(id,prefix,tokens))in cases.iter().enumerate() {
        ensure!(tokens.len()>=prefix+6,"head fixture extent");
        if case>0 {verify.reset(7_540_000+case as u64,comm,stream)?;large.handoff_from(verify)?;}
        large.prefill_wide(wides,&tokens[..*prefix],comm,stream)?;verify.handoff_from(large)?;
        let accepted=state(verify)?;
        for rows in [3usize,6] {
            let mut reference=None;
            for repeat in 0..2u8 {
                for path in if repeat==0 {[0,1]}else{[1,0]} {
                    verify.set_lossless_head_for_probe(path==1,stream)?;
                    control.agree(&[case as u8,rows as u8,repeat,path,0])?;
                    verify.begin_verification(&tokens[*prefix..*prefix+rows],comm,stream)?;
                    let warm=logits(cuda,verify)?;verify.finish_verification(0,comm,stream)?;
                    if reference.is_none(){reference=Some(warm.clone());}
                    ensure!(reference.as_ref()==Some(&warm),"lossless head changed warm logits");
                    let mut seconds=Vec::new();
                    for replay in 1..=4u8 {
                        control.agree(&[case as u8,rows as u8,repeat,path,replay])?;
                        let start=Instant::now();verify.begin_verification(&tokens[*prefix..*prefix+rows],comm,stream)?;
                        seconds.push(start.elapsed().as_secs_f64());
                        let observed=logits(cuda,verify)?;verify.finish_verification(0,comm,stream)?;
                        ensure!(observed==warm,"lossless head changed replay logits");
                    }
                    ensure!(state(verify)?==accepted,"head comparison changed retained state");
                    records.push(json!({"case":case,"id":id,"prefix_tokens":prefix,"rows":rows,
                        "repeat":repeat,"path":path,"forward_seconds":seconds,"logits_sha256":hash(&warm),
                        "all_logits_exact":true,"aborted_state_exact":true}));
                }
            }
        }
    }
    std::fs::write(output.join("lossless-head-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"fixture_sha256":[hash(raw),hash(long_raw)],
        "all_logits_and_retained_state_exact":true,"encoded_bytes_per_rank":253388800,
        "scope":"Two public accepted prefixes, 3/6 live verification rows, two reversed-order pairs with four timed replays after excluded capture. Canonical BF16 bits and FP32 head output; no sampler, saved-state or background changes."}))?)?;
    Ok(())
}
