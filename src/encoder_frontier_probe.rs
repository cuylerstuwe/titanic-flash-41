//! Change the optional checkpoint frontier, not its stored-state contract.
use crate::{control_channel::ControlChannel,cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};
const BUDGET:usize=2*1024usize.pow(3);
fn hash(x:&[u8])->String{format!("{:x}",Sha256::digest(x))}
fn state(t:&Target)->Result<Value>{Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))}
fn last_logits(t:&Target)->Result<Vec<u8>>{Ok(t.last_prompt_logits()?.iter().flat_map(|f|f.to_bits().to_le_bytes()).collect())}
fn following(cuda:&Cuda,t:&mut Target,ids:&[u32],comm:&Roce,stream:&Stream)->Result<Vec<u8>> {
    let before=state(t)?;t.begin_verification(ids,comm,stream)?;
    let (y,n)=t.pending_logits()?;ensure!(n==6,"frontier probe following rows");
    let raw=unsafe{cuda.read_device(y.ptr(),6*129280*4)}?;
    t.finish_verification(0,comm,stream)?;ensure!(state(t)?==before,"frontier probe abort changed state");Ok(raw)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,control:&mut ControlChannel,out:&Path,rank:usize)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_ENCODER_FRONTIER_SCREEN").as_deref()==Ok("1")
        &&large.position()==0,"frontier checkpoint finite admission");
    let fixtures=[include_bytes!("../tests/fixtures/target-agentic-repository-v1.json").as_slice(),
        include_bytes!("../tests/fixtures/long-context-continuation-v2.json").as_slice()];
    let mut records=Vec::new();let mut lineage=7_870_000;
    for (case,base_tokens) in [8209usize,131089].into_iter().enumerate() {
        let data:Value=serde_json::from_slice(fixtures[case])?;
        let tokens:Vec<u32>=serde_json::from_value(data.get("tokens").unwrap_or(&data["batches"][0]).clone())?;
        ensure!(tokens.len()>=base_tokens+518,"frontier fixture extent");
        if case>0{lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;}
        large.prefill_wide(wides,&tokens[..base_tokens],comm,stream)?;
        let base=large.save_state_image(BUDGET,comm,stream)?;
        let base_state=state(large)?;verify.handoff_from(large)?;
        for delta in [256usize,512] {
            let end=base_tokens+delta;
            for repeat in 0..2u8 {
                for path in if repeat==0{[0u8,1]}else{[1,0]} {
                    control.agree(&[case as u8,(delta/256)as u8,repeat,path])?;
                    large.handoff_from(verify)?;lineage+=1;
                    large.restore_state_image(&base,lineage,comm,stream)?;
                    ensure!(state(large)?==base_state,"frontier base restore differs");
                    let checkpoint=if path==0{(end-128)/128*128}else{base_tokens};
                    let prefix=&tokens[base_tokens..checkpoint];let suffix=&tokens[checkpoint..end];
                    let start=Instant::now();
                    let(saved,details)=large.prefill_with_encoder_checkpoint(wides,None,prefix,suffix,Some(BUDGET),comm,stream)?;
                    let append_seconds=start.elapsed().as_secs_f64();let saved=saved.unwrap();
                    ensure!(saved.position()==checkpoint,"frontier checkpoint position differs");
                    let accepted=state(large)?;let logits=last_logits(large)?;verify.handoff_from(large)?;
                    let follow=following(cuda,verify,&tokens[end..end+6],comm,stream)?;
                    // Restore after verification has written future candidates.
                    // Every path must recover its own exact complete owner.
                    large.handoff_from(verify)?;lineage+=1;let start=Instant::now();
                    large.restore_encoder_checkpoint_for_probe(wides,&saved,lineage,suffix,comm,stream)?;
                    let restore_seconds=start.elapsed().as_secs_f64();
                    ensure!(state(large)?==accepted&&last_logits(large)?==logits,"frontier checkpoint replay differs");
                    verify.handoff_from(large)?;
                    ensure!(following(cuda,verify,&tokens[end..end+6],comm,stream)?==follow,
                        "frontier checkpoint replay changed following logits");
                    let stem=format!("frontier-c{case}-d{delta}-p{path}");
                    if rank==0&&repeat==0{
                        std::fs::write(out.join(format!("{stem}-append.bin")),&logits)?;
                        std::fs::write(out.join(format!("{stem}-follow.bin")),&follow)?;
                    }
                    records.push(json!({"case":case,"base_tokens":base_tokens,"delta":delta,"path":path,"repeat":repeat,
                        "checkpoint_tokens":checkpoint,"stem":stem,"append_seconds":append_seconds,"restore_seconds":restore_seconds,
                        "append_sha256":hash(&logits),"follow_sha256":hash(&follow),"state":accepted,"details":details,
                        "image_bytes":saved.bytes(),"replay_and_abort_exact":true}));
                    std::fs::write(out.join("encoder-frontier-partial.json"),serde_json::to_vec(&records)?)?;
                }
            }
        }
    }
    std::fs::write(out.join("encoder-frontier-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"fixture_sha256":fixtures.iter().map(|x|hash(x)).collect::<Vec<_>>(),
        "scope":"Two real8K/131K prefixes at odd frontiers,256/512-token appends, two reversed pairs. Aligned near-end encoder checkpoint versus live-frontier checkpoint followed by one encoder plan. Exact same-path restore/replay and six following logits; cross-path CUDA distribution diagnostics. Older checkpoint may require more branch replay; no serving policy or taskquality claim."}))?)?;Ok(())
}
