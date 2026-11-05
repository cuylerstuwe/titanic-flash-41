//! Same-allocation verification of a dense-only backend with unchanged producer.
use super::*;
use super::history32_generation_probe::execute;
fn target_state(t:&Target)->Result<Value>{Ok(json!({"position":t.position(),
    "carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))}
fn logits(t:&Target,cuda:&Cuda)->Result<String>{
    // Production verification retains rank-local logits. Compare all columns
    // on all ranks without changing that graph or requiring an unused gather.
    let(y,rows)=t.pending_local_logits_for_probe()?;
    Ok(format!("{:x}",Sha256::digest(unsafe{cuda.read_device(y.ptr(),rows*32320*4)}?)))
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.long_verify.is_some()
        &&std::env::var("CANADA_SCRATCH_SHARED_DOWN_FLASH_SCREEN").as_deref()==Ok("1"),
        "isolated shared-down screen");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let prefix=131073;
    // This probe may follow prefill_head_probe (request 8_041_314).
    execute(w,Command::Begin{images:Vec::new(),id:8_041_315,seed:2026110448,
        greedy:false,limit:1,tokens:tokens[..prefix].to_vec()})?;
    let mut records=Vec::new();
    let transfer_screen=std::env::var("CANADA_SCRATCH_MOE_WIDE_TRANSFER_SCREEN").as_deref()==Ok("1");
    let mut families=if transfer_screen{vec![(16usize,"expert_transfer")]}
        else{vec![(6usize,"shared_down"),(16,"shared_down")]};
    if std::env::var("CANADA_SCRATCH_INDEX_QUERY_FLASH_SCREEN").as_deref()==Ok("1") {
        families.push((16,"index_query"));
    }
    for (physical,family) in families {
        if physical==16{w.long_verify.as_mut().unwrap().handoff_from(&w.verify)?;}
        let t=if physical==6{&mut w.verify}else{w.long_verify.as_mut().unwrap()};
        let before=target_state(t)?;
        let mut expected=std::collections::BTreeMap::new();
        for(trial,enabled)in[false,true,true,false].into_iter().enumerate() {
            if family=="expert_transfer" {t.set_moe_wide_transfer_for_probe(enabled,&w.stream)?;}
            else if family=="shared_down" {t.set_shared_down_flash_for_probe(enabled,&w.stream)?;}
            else {t.set_index_query_flash_for_probe(enabled,&w.stream)?;}
            for live in if physical==6{vec![3usize,6]}else if transfer_screen{vec![8,16]}else{vec![16]} {
                w.control.agree(&[0xd4,physical as u8,live as u8,trial as u8])?;
                for _ in 0..6 {
                    t.begin_verification(&tokens[prefix..prefix+live],&w.comm,&w.stream)?;
                    t.finish_verification(0,&w.comm,&w.stream)?;
                }
                let captures=t.capture_statistics().1;let mut seconds=Vec::new();
                for _ in 0..16 {
                    let at=Instant::now();
                    t.begin_verification(&tokens[prefix..prefix+live],&w.comm,&w.stream)?;
                    w.stream.synchronize()?;
                    seconds.push(at.elapsed().as_secs_f64());
                    t.finish_verification(0,&w.comm,&w.stream)?;
                }
                ensure!(t.capture_statistics().1==captures,"timed shared-down recapture");
                let mut checks=Vec::new();
                for offset in [prefix,prefix+32] {
                    t.begin_verification(&tokens[offset..offset+live],&w.comm,&w.stream)?;
                    checks.push(logits(t,&w.cuda)?);
                    t.finish_verification(0,&w.comm,&w.stream)?;
                }
                ensure!(checks[0]!=checks[1]&&target_state(t)?==before,"shared-down input/state invariant");
                let exact=expected.entry(live).or_insert_with(||checks.clone())==&checks;
                ensure!(w.control.all_admit(exact)?,"shared-down target logits changed");
                records.push(json!({"family":family,"physical_rows":physical,"live":live,"trial":trial,
                    "enabled":enabled,"seconds":seconds,"checks":checks,"state_exact":true}));
                std::fs::write(out.join("shared-down-flash-partial.json"),serde_json::to_vec_pretty(&records)?)?;
                eprintln!("prepared dense {family} rank{rank} M{physical} live{live} enabled{enabled}: exact");
            }
        }
        if family=="expert_transfer" {t.set_moe_wide_transfer_for_probe(crate::moe::wide_transfer_requested()?,&w.stream)?;}
        else if family=="shared_down" {t.set_shared_down_flash_for_probe(false,&w.stream)?;}
        else {t.set_index_query_flash_for_probe(false,&w.stream)?;}
        if physical==16{w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;}
    }
    execute(w,Command::Finish(Finish::Length))?;
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("shared-down-flash-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"prefix_tokens":prefix,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "records":records,"all_logits_and_retained_state_exact":true,"original_backend_restored":true,
        "comparison":if transfer_screen{"M16 exact expert kernel transfer"}else{"prepared dense consumer"},
        "scope":"Same-allocation ABBA M3/M6/M16 shared-down and optional M16 indexer-query verification at131K. Existing native quantizer/fused producer and padding, same weights and buffers. Dense consumer only changes. Two changed-input patterns, every local vocabulary column and rollback state exact on all ranks;16 warmed timings per arm including stream completion. No generation or task-quality claim."
    }))?)?;Ok(())
}
