//! Same canonical operands and stable output buffers, alternate WO-B bindings.
use crate::{control_channel::ControlChannel,cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};
fn hash(x:&[u8])->String{format!("{:x}",Sha256::digest(x))}
fn state(t:&Target)->Result<Value>{Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))}
fn logits(cuda:&Cuda,t:&Target)->Result<Vec<u8>>{let(y,rows)=t.pending_logits()?;unsafe{cuda.read_device(y.ptr(),rows*129280*4)}}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,control:&mut ControlChannel,kernels:&Path,out:&Path,rank:usize)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_WO_B_PDL_SCREEN").as_deref()==Ok("1")
        &&large.position()==0&&verify.physical_rows()==6,"WO-B probe admission");
    let short_raw=include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json");
    let short:Value=serde_json::from_slice(short_raw)?;
    let c=short["cases"].as_array().unwrap().iter().find(|c|c["id"]=="offline-product-ideation-answer-2").unwrap();
    let tokens:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
    let prefix=tokens.len()-16;
    large.prefill_wide(wides,&tokens[..prefix],comm,stream)?;verify.handoff_from(large)?;
    let mut other=verify.prepare_wo_b_alternates_for_probe(kernels)?;
    let accepted=state(verify)?;let mut reference:Option<Vec<u8>>=None;let mut records=Vec::new();
    // Rebuild the same backend twice as well as crossing between backends.
    // This isolates PDL from backend choice and from graph reconstruction.
    for (trial,path) in [0u8,0,1,1,2,2,1,0].into_iter().enumerate() {
        verify.set_wo_b_pdl_for_probe(path,&mut other,stream)?;
        control.agree(&[trial as u8,path,0])?;
        verify.begin_verification(&tokens[prefix..prefix+6],comm,stream)?;
        let warm=logits(cuda,verify)?;verify.finish_verification(0,comm,stream)?;
        let exact_to_first=reference.as_ref().is_none_or(|r|r==&warm);
        if reference.is_none(){reference=Some(warm.clone());}
        std::fs::write(out.join(format!("wo-b-trial{trial}-warm.bin")),&warm)?;
        let mut seconds=Vec::new();let mut replays_exact=true;let mut replay_hashes=Vec::new();
        for replay in 1..=3u8 {
            control.agree(&[trial as u8,path,replay])?;
            let at=Instant::now();verify.begin_verification(&tokens[prefix..prefix+6],comm,stream)?;
            seconds.push(at.elapsed().as_secs_f64());let observed=logits(cuda,verify)?;
            verify.finish_verification(0,comm,stream)?;
            let exact=warm==observed;replays_exact&=exact;replay_hashes.push(hash(&observed));
            if !exact{std::fs::write(out.join(format!("wo-b-trial{trial}-replay{replay}.bin")),observed)?;}
        }
        ensure!(state(verify)?==accepted,"WO-B abort changed retained state");
        records.push(json!({"trial":trial,"path":path,"prefix_tokens":prefix,"seconds":seconds,
            "logits_sha256":hash(&warm),"replay_sha256":replay_hashes,"state_exact":true,
            "exact_to_first":exact_to_first,"replays_exact":replays_exact}));
        std::fs::write(out.join("wo-b-partial.json"),serde_json::to_vec(&records)?)?;
        eprintln!("WO-B rank{rank} trial{trial} path{path} exact_to_first={exact_to_first} replays_exact={replays_exact}");
    }
    verify.set_wo_b_pdl_for_probe(0,&mut other,stream)?;
    std::fs::write(out.join("wo-b-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,
        "records":records,"fixture_sha256":hash(short_raw),"all_abort_state_exact":true,
        "scope":"One fixed accepted prefix. Eight trials original/original/FIoff/FIoff/FIon/FIon/FIoff/original; graph rebuilding every trial, three replays each. Differences captured for GPU diagnosis rather than accepted as harmless. No serving change or general performance claim."}))?)?;
    Ok(())
}
