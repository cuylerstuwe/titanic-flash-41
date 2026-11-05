//! Finite paired precision/performance screen. Cross-path rounding may differ;
//! same-path repeats, rollback and snapshot restoration must remain exact.
use super::*;
fn hash(x:&[u8])->String {format!("{:x}",Sha256::digest(x))}
fn state(t:&Target)->Result<Value> {
    Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))
}
fn local(cuda:&Cuda,t:&Target)->Result<Vec<u8>> {
    let(y,n)=t.pending_local_logits_for_probe()?;
    unsafe{cuda.read_device(y.ptr(),n*32320*4)}
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.large.position()==0&&w.copy32_verify.is_some()
        &&std::env::var("CANADA_SCRATCH_WOA_MID_SCREEN").as_deref()==Ok("1"),"finite compact WO-A screen");
    let raw=include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json");
    let f:Value=serde_json::from_slice(raw)?;let mut cases=Vec::new();
    for id in ["rust-config-parser-reasoning-1","offline-product-ideation-answer-2"] {
        let c=f["cases"].as_array().unwrap().iter().find(|c|c["id"]==id).unwrap();
        let ids:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
        ensure!(ids.len()>168,"compact WO-A fixture extent");
        cases.push((id.to_owned(),ids.len()-40,ids));
    }
    let long_raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let f:Value=serde_json::from_slice(long_raw)?;
    cases.push(("long-agentic-context".into(),131073,serde_json::from_value(f["tokens"].clone())?));
    let mut targets=[w.long_verify.take().context("M16 owner")?,w.copy32_verify.take().context("M32 owner")?];
    let mut records=Vec::new();
    let reference=std::env::var("CANADA_SCRATCH_WOA_MID_REFERENCE_SCREEN").as_deref()==Ok("1");
    for(case,(id,prefix,tokens))in cases.iter().enumerate() {
        if case>0 {
            w.target_lineage=w.target_lineage.checked_add(1).context("compact WO-A lineage")?;
            w.verify.reset(w.target_lineage,&w.comm,&w.stream)?;w.large.handoff_from(&w.verify)?;
        }
        w.large.prefill_wide(&mut w.wides,&tokens[..*prefix],&w.comm,&w.stream)?;
        w.verify.handoff_from(&w.large)?;
        for t in &mut targets {
            let width=t.physical_rows();t.handoff_from(&w.verify)?;
            let incoming=state(t)?;let bytes=t.compact_state_image_bytes()?;
            ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
                >=8*1024usize.pow(3)+bytes+64*1024usize.pow(2))?,"compact WO-A snapshot reserve");
            let saved=t.save_state_image(bytes,&w.comm,&w.stream)?;
            let mut expected=std::collections::BTreeMap::<(u8,usize,usize),Vec<u8>>::new();
            let mut committed=std::collections::BTreeMap::<u8,(Value,Vec<u8>)>::new();
            let modes=if reference{vec![0u8,1,2]}else{vec![0,1,1,0]};
            for(trial,mode)in modes.into_iter().enumerate() {
                t.set_woa_compact_mode_for_probe(mode,&w.stream)?;
                let lives=if reference{vec![width]}else if width==16{vec![8,16]}else{vec![17,32]};
                for live in lives {
                    let mut samples=Vec::new();
                    for repeat in 0..6 {
                        w.control.agree(&[0xd7,case as u8,width as u8,trial as u8,live as u8,repeat])?;
                        let at=Instant::now();t.begin_verification(&tokens[*prefix..*prefix+live],&w.comm,&w.stream)?;
                        w.stream.synchronize()?;let ms=at.elapsed().as_secs_f64()*1000.;
                        t.finish_verification(0,&w.comm,&w.stream)?;
                        ensure!(state(t)?==incoming,"compact WO-A timed rollback");
                        if repeat>=2{samples.push(ms);}
                    }
                    let mut changed=tokens[*prefix..*prefix+live].to_vec();
                    for token in &mut changed[live/2..]{*token=2+(*token+123)%120000;}
                    for(kind,input)in[&tokens[*prefix..*prefix+live],changed.as_slice()].into_iter().enumerate() {
                        t.begin_verification(input,&w.comm,&w.stream)?;let y=local(&w.cuda,t)?;
                        let key=(mode,live,kind);
                        if let Some(old)=expected.get(&key){ensure!(*old==y,"same-path compact WO-A logits changed");}
                        else {
                            let stem=format!("woa-c{case}-m{width}-n{live}-p{mode}-k{kind}");
                            std::fs::write(out.join(format!("{stem}.bin")),&y)?;expected.insert(key,y.clone());
                        }
                        t.finish_verification(0,&w.comm,&w.stream)?;
                        ensure!(state(t)?==incoming,"compact WO-A changed-input rollback");
                        if reference {
                            t.begin_verification(input,&w.comm,&w.stream)?;
                            ensure!(local(&w.cuda,t)?==y,"three-way WO-A same-path repeat");
                            t.finish_verification(0,&w.comm,&w.stream)?;
                            ensure!(state(t)?==incoming,"three-way WO-A repeat rollback");
                        }
                    }
                    records.push(json!({"case":case,"id":id,"prefix":prefix,"width":width,"live":live,
                        "trial":trial,"enabled":mode==1,"mode":mode,"samples_ms":samples,"repeat_and_rollback_exact":true}));
                    std::fs::write(out.join("woa-mid-partial.json"),serde_json::to_vec_pretty(&records)?)?;
                }
                let accept=if width==16{7}else{15};
                t.begin_verification(&tokens[*prefix..*prefix+width],&w.comm,&w.stream)?;
                t.verification_history_sample(103200,&w.comm,&w.stream)?;
                t.finish_verification(accept,&w.comm,&w.stream)?;let after=state(t)?;
                w.verify.handoff_from(t)?;
                w.verify.begin_verification(&tokens[*prefix+accept..*prefix+accept+6],&w.comm,&w.stream)?;
                let(y,n)=w.verify.pending_local_logits_for_probe()?;ensure!(n==6,"compact WO-A following extent");
                let following=unsafe{w.cuda.read_device(y.ptr(),6*32320*4)}?;
                w.verify.finish_verification(0,&w.comm,&w.stream)?;
                ensure!(state(&w.verify)?==after,"compact WO-A following rollback");
                let proof=(after,following);
                if let Some(old)=committed.get(&mode){ensure!(*old==proof,"compact WO-A committed repeat changed");}
                else {
                    std::fs::write(out.join(format!("woa-c{case}-m{width}-p{mode}-following.bin")),&proof.1)?;
                    committed.insert(mode,proof);
                }
                w.target_lineage=w.target_lineage.checked_add(1).context("compact WO-A restore lineage")?;
                // The following M6 verification, including zero acceptance,
                // advances the shared writer revision. Adopt it before restore.
                t.handoff_from(&w.verify)?;
                t.restore_state_image(&saved,w.target_lineage,&w.comm,&w.stream)?;
                ensure!(state(t)?==incoming,"compact WO-A restored snapshot");
                std::fs::write(out.join("woa-mid-partial.json"),serde_json::to_vec_pretty(&records)?)?;
                eprintln!("compact WO-A rank{rank} case{case} M{width} trial{trial} mode{mode}: repeat/state pass");
            }
            t.set_woa_compact_for_probe(crate::woa_medium::compact_policy()?,&w.stream)?;
            w.verify.handoff_from(t)?;
        }
    }
    let [long,copy]=targets;w.long_verify=Some(long);w.copy32_verify=Some(copy);
    super::history32_generation_probe::execute(w,Command::Shutdown)?;
    std::fs::write(out.join("woa-mid-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"fp64_woa_reference":reference,"all_repeat_and_state_checks_pass":true,
        "phase_fixture_sha256":hash(raw),"long_fixture_sha256":hash(long_raw),
        "scope":if reference {"Three accepted prefixes, full M16/M32: native BF16 GEMM, compact BF16 operands/FP32 partials, and FP64-accumulated WO-A with the same BF16 output boundary. Same-allocation changed-input repeat/state, partial commit, following M6 and restoration. Only WO-A is higher precision; other model operations remain unchanged. Distribution diagnostic, not an official full-model reference or task-quality claim."}
            else {"Three public accepted prefixes: Rust reasoning, ideation answer and131Krepository. M16live8/16,M32live17/32; same-allocation ABBA. Same-path changed-input logits, zero rollback, partial commit, following M6 logits and snapshot restoration exact. Cross-path floating-point sums may differ; raw logits for GPU distribution analysis. No task-quality or serving claim."}
    }))?)?;Ok(())
}
