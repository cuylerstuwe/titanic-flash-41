//! One long accepted prefix, exact arithmetic, bounded ABBA kernel comparison.
use crate::{control_channel::ControlChannel,cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::BTreeMap,path::Path,time::Instant};
fn hash(x:&[u8])->String{format!("{:x}",Sha256::digest(x))}
fn state(t:&Target)->Result<Value>{Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))}
fn logits(cuda:&Cuda,t:&Target)->Result<Vec<u8>>{let(y,rows)=t.pending_logits()?;unsafe{cuda.read_device(y.ptr(),rows*129280*4)}}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,control:&mut ControlChannel,out:&Path,rank:usize)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_MOE_INPUT_ADDRESS_SCREEN").as_deref()==Ok("1")
        &&large.position()==0&&verify.physical_rows()==6,"expert address probe admission");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let prefix=131072;
    ensure!(tokens.len()>=prefix+6,"long expert probe fixture");
    large.prefill_wide(wides,&tokens[..prefix],comm,stream)?;verify.handoff_from(large)?;
    let accepted=state(verify)?;let mut reference=BTreeMap::new();let mut records=Vec::new();
    for (trial,enabled) in [false,true,true,false].into_iter().enumerate() {
        verify.set_moe_input_address_for_probe(enabled,stream)?;
        for live in [3,6] {
            control.agree(&[trial as u8,enabled as u8,live as u8,0])?;
            verify.begin_verification(&tokens[prefix..prefix+live],comm,stream)?;
            let warm=logits(cuda,verify)?;verify.finish_verification(0,comm,stream)?;
            if let Some(expected)=reference.get(&live){ensure!(&warm==expected,"expert address changed target logits");}
            else{reference.insert(live,warm.clone());}
            let mut seconds=Vec::new();let mut metrics=Vec::new();
            for replay in 1..=3u8 {
                control.agree(&[trial as u8,enabled as u8,live as u8,replay])?;
                let at=Instant::now();verify.begin_verification(&tokens[prefix..prefix+live],comm,stream)?;
                stream.synchronize()?;seconds.push(at.elapsed().as_secs_f64());
                metrics.push(serde_json::to_value(verify.verification_metrics()?)?);
                let observed=logits(cuda,verify)?;verify.finish_verification(0,comm,stream)?;
                ensure!(warm==observed,"expert address replay logits changed");
            }
            ensure!(state(verify)?==accepted,"expert address abort changed retained state");
            records.push(json!({"trial":trial,"enabled":enabled,"live":live,"seconds":seconds,
                "metrics":metrics,"logits_sha256":hash(&warm),"exact":true,"state_exact":true}));
            std::fs::write(out.join("moe-input-address-partial.json"),serde_json::to_vec(&records)?)?;
            eprintln!("expert input-address rank{rank} trial{trial} enabled{enabled} M{live}: exact");
        }
    }
    verify.set_moe_input_address_for_probe(false,stream)?;
    std::fs::write(out.join("moe-input-address-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,
        "prefix_tokens":prefix,"records":records,"fixture_sha256":hash(raw),"all_exact":true,
        "scope":"One131K public repository-history prefix. Three/six-row target verification, ABBA paths with three warm replays each. Canonical precision and arithmetic unchanged; mapped sampler and candidate-error readout retained. Exact logits and rollback state. Target-only timing, not completed-task quality or generation TPS."}))?)?;
    Ok(())
}
