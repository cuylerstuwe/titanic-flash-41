//! Immutable-weight L2 hints across the M16/M32 communication windows.
//! Capture selection only; mathematical results and retained state stay exact.
use super::*;
fn state(t:&Target)->Result<Value> {
    Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))
}
fn local_hash(cuda:&Cuda,t:&Target)->Result<String> {
    let(y,n)=t.pending_local_logits_for_probe()?;
    Ok(format!("{:x}",Sha256::digest(unsafe{cuda.read_device(y.ptr(),n*32320*4)}?)))
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.large.position()==0&&w.copy32_verify.is_some()
        &&crate::l2_prefetch::enabled()
        &&!crate::woa_medium::compact_policy()?
        &&std::env::var("CANADA_SCRATCH_WIDE_L2_SCREEN").as_deref()==Ok("1"),
        "finite wide hint screen requires ordinary WO-A and prepared windows");
    let raw=include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json");
    let f:Value=serde_json::from_slice(raw)?;
    let c=f["cases"].as_array().unwrap().iter().find(|c|c["id"]=="rust-config-parser-reasoning-1").unwrap();
    let ids:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
    let mut cases=vec![("rust-config-parser-reasoning-1",ids.len()-40,ids)];
    let long_raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let f:Value=serde_json::from_slice(long_raw)?;
    cases.push(("long-agentic-context",131073,serde_json::from_value(f["tokens"].clone())?));
    let mut targets=[w.long_verify.take().context("M16 owner")?,w.copy32_verify.take().context("M32 owner")?];
    let mut records=Vec::new();
    for(case,(id,prefix,tokens))in cases.iter().enumerate() {
        if case>0 {
            w.target_lineage=w.target_lineage.checked_add(1).context("wide hint lineage")?;
            w.verify.reset(w.target_lineage,&w.comm,&w.stream)?;w.large.handoff_from(&w.verify)?;
        }
        w.large.prefill_wide(&mut w.wides,&tokens[..*prefix],&w.comm,&w.stream)?;
        w.verify.handoff_from(&w.large)?;
        for t in &mut targets {
            let width=t.physical_rows();t.handoff_from(&w.verify)?;
            let incoming=state(t)?;let bytes=t.compact_state_image_bytes()?;
            ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
                >=8*1024usize.pow(3)+bytes+64*1024usize.pow(2))?,"wide hint snapshot reserve");
            let saved=t.save_state_image(bytes,&w.comm,&w.stream)?;
            let mut expected=None;let mut expected_commit=None;
            for(trial,mask)in[0u8,7,1,2,4,4,2,1,7,0].into_iter().enumerate() {
                t.set_wide_hints_for_probe(mask,&w.stream)?;
                let mut samples=Vec::new();let mut captured=None;
                for repeat in 0..6 {
                    w.control.agree(&[0xd8,case as u8,width as u8,trial as u8,mask,repeat])?;
                    let at=Instant::now();t.begin_verification(&tokens[*prefix..*prefix+width],&w.comm,&w.stream)?;
                    w.stream.synchronize()?;let ms=at.elapsed().as_secs_f64()*1000.;
                    t.finish_verification(0,&w.comm,&w.stream)?;
                    ensure!(state(t)?==incoming,"wide hint timed rollback");
                    if repeat==1{captured=Some(t.capture_statistics().1);}
                    if repeat>=2 {
                        ensure!(captured==Some(t.capture_statistics().1),"wide hint timed graph recapture");
                        samples.push(ms);
                    }
                }
                let mut changed=tokens[*prefix..*prefix+width].to_vec();
                for token in &mut changed[width/2..]{*token=2+(*token+123)%120000;}
                let live=if width==16{8}else{17};let mut checks=Vec::new();
                for input in [&tokens[*prefix..*prefix+width],changed.as_slice(),&tokens[*prefix..*prefix+live]] {
                    t.begin_verification(input,&w.comm,&w.stream)?;
                    checks.push(local_hash(&w.cuda,t)?);
                    t.finish_verification(0,&w.comm,&w.stream)?;
                    ensure!(state(t)?==incoming,"wide hint changed/partial input rollback");
                }
                if let Some(old)=&expected{ensure!(old==&checks,"wide hint changed target logits");}
                else{expected=Some(checks.clone());}
                let accept=if width==16{7}else{15};
                t.begin_verification(&tokens[*prefix..*prefix+width],&w.comm,&w.stream)?;
                t.verification_history_sample(103200,&w.comm,&w.stream)?;
                t.finish_verification(accept,&w.comm,&w.stream)?;let after=state(t)?;
                w.verify.handoff_from(t)?;
                w.verify.begin_verification(&tokens[*prefix+accept..*prefix+accept+6],&w.comm,&w.stream)?;
                let following=local_hash(&w.cuda,&w.verify)?;
                w.verify.finish_verification(0,&w.comm,&w.stream)?;
                ensure!(state(&w.verify)?==after,"wide hint following rollback");
                let proof=(after,following);
                if let Some(old)=&expected_commit{ensure!(old==&proof,"wide hint changed committed/following state");}
                else{expected_commit=Some(proof.clone());}
                w.target_lineage=w.target_lineage.checked_add(1).context("wide hint restore lineage")?;
                t.handoff_from(&w.verify)?;
                t.restore_state_image(&saved,w.target_lineage,&w.comm,&w.stream)?;
                ensure!(state(t)?==incoming,"wide hint restored snapshot");
                records.push(json!({"case":case,"id":id,"prefix":prefix,"width":width,"trial":trial,"mask":mask,
                    "samples_ms":samples,"checks":checks,"following_logits_sha256":proof.1,
                    "all_logits_state_exact":true,"new_timed_graph_captures":0}));
                std::fs::write(out.join("wide-hint-partial.json"),serde_json::to_vec_pretty(&records)?)?;
                eprintln!("wide hint rank{rank} case{case} M{width} trial{trial} mask{mask}: exact");
            }
            t.set_wide_hints_for_probe(crate::l2_prefetch::wide_mask()?,&w.stream)?;
            w.verify.handoff_from(t)?;
        }
    }
    let [long,copy]=targets;w.long_verify=Some(long);w.copy32_verify=Some(copy);
    super::history32_generation_probe::execute(w,Command::Shutdown)?;
    std::fs::write(out.join("wide-hint-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"all_logits_state_exact":true,
        "phase_fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "long_fixture_sha256":format!("{:x}",Sha256::digest(long_raw)),
        "scope":"Two public prefixes: short Rust reasoning and131Krepository. M16/M32 full-row timing in reversed orders; original, combined and separate FFN/Q-B/input hints. Same immutable6MiB plans, allocations and arithmetic. Changed and partial inputs, zero/partial commit, following M6 and restored state exact on all vocab shards. No new prefill policy."
    }))?)?;Ok(())
}
