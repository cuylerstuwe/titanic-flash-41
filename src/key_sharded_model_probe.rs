//! Bounded whole-target screen after the four-device score/select component.
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
    ensure!(std::env::var("CANADA_SCRATCH_KEY_SHARDED_INDEX_SCREEN").as_deref()==Ok("1")
        &&large.position()==0&&verify.physical_rows()==6,"key-shard model probe admission");
    let raw_a=include_bytes!("../tests/fixtures/long-history-context-v1.json");
    let raw_b=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let a:Value=serde_json::from_slice(raw_a)?;let b:Value=serde_json::from_slice(raw_b)?;
    let mut tokens:Vec<u32>=serde_json::from_value(a["cases"][0]["prompt_ids"].clone())?;
    tokens.extend(serde_json::from_value::<Vec<u32>>(b["tokens"].clone())?);
    ensure!(tokens.len()>524288+18&&tokens.iter().all(|&v|v<129280),"public key-shard corpus");
    let steady=std::env::var("CANADA_SCRATCH_KEY_SHARD_STEADY").as_deref()==Ok("1");
    let automatic=std::env::var("CANADA_SCRATCH_KEY_SHARD_AUTO_SCREEN").as_deref()==Ok("1");
    let publisher_only=std::env::var("CANADA_SCRATCH_KEY_SHARDED_PUBLISHER_SCREEN").as_deref()==Ok("1");
    let midpoint=std::env::var("CANADA_SCRATCH_KEY_SHARD_MIDPOINT_SCREEN").as_deref()==Ok("1");
    ensure!(!midpoint||(steady&&automatic&&publisher_only),"midpoint uses steady automatic publisher screen");
    let prefixes=if midpoint{vec![393216,513]}else if automatic{vec![513,524288,513]}
        else if steady&&!publisher_only{vec![524288]}else{vec![513,524288]};
    let mut records=Vec::new();let mut transitions=Vec::new();
    for (case,prefix) in prefixes.into_iter().enumerate(){
        if case>0{
            if prefix<verify.position()as usize{verify.reset(8_110_000+case as u64,comm,stream)?;}
            large.handoff_from(verify)?;
        }
        let at=Instant::now();large.prefill_wide(wides,&tokens[large.position()as usize..prefix],comm,stream)?;
        let prefill_seconds=at.elapsed().as_secs_f64();verify.handoff_from(large)?;
        let accepted=state(verify)?;let mut reference=BTreeMap::new();
        if automatic {
            if case==0{verify.set_key_sharded_index_for_probe(true,stream)?;}
            // The previous case leaves automatic mode and a warm graph active.
            // Handoff/reset must select the new context's mode on its own.
            let at=Instant::now();verify.begin_verification(&tokens[prefix..prefix+6],comm,stream)?;
            let first_use_seconds=at.elapsed().as_secs_f64();
            let first_use_metrics=serde_json::to_value(verify.verification_metrics()?)?;
            let value=logits(cuda,verify)?;
            let expected=(if prefix>=524288{3}else{0})
                +usize::from(publisher_only&&prefix>=if midpoint{393216}else{524288});
            ensure!(verify.active_key_sharded_layers()==expected,"key-shard automatic threshold");
            verify.finish_verification(0,comm,stream)?;
            ensure!(state(verify)?==accepted,"key-shard automatic rollback");
            let at=Instant::now();verify.begin_verification(&tokens[prefix..prefix+6],comm,stream)?;
            let warm_seconds=at.elapsed().as_secs_f64();
            let warm_metrics=serde_json::to_value(verify.verification_metrics()?)?;
            ensure!(value==logits(cuda,verify)?,"automatic warm replay logits");
            verify.finish_verification(0,comm,stream)?;
            ensure!(state(verify)?==accepted,"automatic warm replay state");
            reference.insert(6,value);
            transitions.push(json!({"case":case,"prefix_tokens":prefix,"active_layers":verify.active_key_sharded_layers(),"state_exact":true,
                "first_use_seconds":first_use_seconds,"first_use_metrics":first_use_metrics,"warm_seconds":warm_seconds,"warm_metrics":warm_metrics}));
        }
        if steady {
            // Fill both graph/cache paths before measuring either. Each timing
            // block below then keeps the normal model execution cadence: no
            // full-vocabulary host copy or external barrier between its steps.
            for enabled in [false,true]{
                verify.set_key_sharded_index_for_probe(enabled,stream)?;
                for live in [3,6]{for _ in 0..4{
                    verify.begin_verification(&tokens[prefix..prefix+live],comm,stream)?;
                    verify.finish_verification(0,comm,stream)?;
                }}
            }
            ensure!(state(verify)?==accepted,"key-shard prewarm changed retained state");
        }
        for (trial,enabled) in [false,true,true,false].into_iter().enumerate(){
            verify.set_key_sharded_index_for_probe(enabled,stream)?;
            for live in [3,6]{
                control.agree(&[case as u8,trial as u8,live as u8,0])?;
                let at=Instant::now();verify.begin_verification(&tokens[prefix..prefix+live],comm,stream)?;
                let first_use_seconds=at.elapsed().as_secs_f64();
                let first_use_metrics=serde_json::to_value(verify.verification_metrics()?)?;
                let warm=logits(cuda,verify)?;verify.finish_verification(0,comm,stream)?;
                let exact=reference.get(&live).is_none_or(|expected|expected==&warm);
                if !exact{
                    // Preserve diagnostic bytes before rejecting the candidate;
                    // equal-score ties are investigated separately from corruption.
                    std::fs::write(out.join(format!("key-shard-c{case}-t{trial}-m{live}-control.bin")),reference.get(&live).unwrap())?;
                    std::fs::write(out.join(format!("key-shard-c{case}-t{trial}-m{live}-candidate.bin")),&warm)?;
                }
                ensure!(exact,"key-shard changed target logits; paired bytes saved");
                reference.entry(live).or_insert_with(||warm.clone());
                if steady {for _ in 0..4{
                    verify.begin_verification(&tokens[prefix..prefix+live],comm,stream)?;
                    verify.finish_verification(0,comm,stream)?;
                }}
                let repeats=if steady{12}else{3};
                let mut seconds=Vec::new();let mut metrics=Vec::new();
                for replay in 1..=repeats{
                    if !steady {control.agree(&[case as u8,trial as u8,live as u8,replay])?;}
                    let at=Instant::now();verify.begin_verification(&tokens[prefix..prefix+live],comm,stream)?;
                    stream.synchronize()?;seconds.push(at.elapsed().as_secs_f64());
                    metrics.push(serde_json::to_value(verify.verification_metrics()?)?);
                    if !steady||replay==repeats {ensure!(warm==logits(cuda,verify)?,"key-shard replay changed logits");}
                    verify.finish_verification(0,comm,stream)?;
                }
                ensure!(state(verify)?==accepted,"key-shard abort changed retained state");
                records.push(json!({"prefix_tokens":prefix,"trial":trial,"enabled":enabled,"live":live,
                    "seconds":seconds,"metrics":metrics,"prefill_seconds":prefill_seconds,"logits_sha256":hash(&warm),"exact":true,"state_exact":true,
                    "first_use_seconds":first_use_seconds,"first_use_metrics":first_use_metrics}));
                std::fs::write(out.join("key-shard-partial.json"),serde_json::to_vec(&records)?)?;
                eprintln!("key-shard rank{rank} prefix{prefix} trial{trial} enabled{enabled} M{live}: exact");
            }
        }
        verify.set_key_sharded_index_for_probe(automatic,stream)?;
        if automatic {
            verify.begin_verification(&tokens[prefix..prefix+6],comm,stream)?;
            ensure!(reference[&6]==logits(cuda,verify)?,"key-shard final automatic replay");
            verify.finish_verification(0,comm,stream)?;
            ensure!(state(verify)?==accepted,"key-shard final automatic state");
        }
    }
    std::fs::write(out.join("key-shard-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,
        "records":records,"fixture_sha256":[hash(raw_a),hash(raw_b)],"all_exact":true,"steady_cadence":steady,
        "automatic_threshold_screen":automatic,"transitions":transitions,
        "publisher_only":publisher_only,
        "midpoint_screen":midpoint,
        "scope":"Concatenated public repository/history fixtures. Three/six-row target verification,ABBA; prefix_tokens and steady_cadence identify the screen. Steady mode warms both paths, adds four excluded warm steps per block, then measures12 steps with logits checked only at the end and no per-step control barrier. Actual native ABI3 byte gather, identical canonical precision, exact logits and retained rollback state. Target-only timing, not serving TPS or completed-task quality."}))?)?;
    Ok(())
}
