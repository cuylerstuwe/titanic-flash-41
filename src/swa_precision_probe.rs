//! Finite first-two-layer precision screen; no sampling or serving policy.
use super::*;
fn digest(b:&[u8])->String {format!("{:x}",Sha256::digest(b))}
fn state(t:&Target)->Result<Value> {
    Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))
}
fn pending(w:&Worker)->Result<Vec<u8>> {
    let(y,rows)=w.verify.pending_logits()?;ensure!(rows==6,"SWA precision logit extent");
    unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    let source20=std::env::var("CANADA_SCRATCH_SOURCE20_PRECISION_SCREEN").as_deref()==Ok("1");
    ensure!(!source20||std::env::var("CANADA_SCRATCH_SWA_PRECISION_SCREEN").as_deref()!=Ok("1"),
        "one M6 precision comparison per load");
    let configure=|target:&mut Target,enabled:bool,stream:&Stream|->Result<()> {
        if source20 {target.set_source20_precision_for_probe(enabled,stream)}
        else {target.set_swa_precision_for_probe(enabled,stream)}
    };
    ensure!(w.current.is_none()&&w.history.frontier()==0&&w.large.profile_is_current()
        &&w.large.position()==0,"SWA screen needs the fresh pre-warmup target");
    let raw=include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json");
    let f:Value=serde_json::from_slice(raw)?;
    let mut cases=Vec::new();
    for id in ["rust-config-parser-reasoning-0","offline-product-ideation-answer-2"] {
        let c=f["cases"].as_array().unwrap().iter().find(|c|c["id"]==id).unwrap();
        let ids:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
        let prefix=ids.len()-6;cases.push((id.to_owned(),prefix,ids));
    }
    let long_raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let long:Value=serde_json::from_slice(long_raw)?;
    cases.push(("long-agentic-context".into(),131073,serde_json::from_value(long["tokens"].clone())?));
    let mut records=Vec::new();
    for(case,(id,prefix,tokens))in cases.iter().enumerate() {
        ensure!(tokens.len()>=prefix+6,"SWA fixture extent");
        if case>0 {
            configure(&mut w.verify,false,&w.stream)?;
            w.target_lineage=w.target_lineage.checked_add(1).context("SWA lineage exhausted")?;
            w.verify.reset(w.target_lineage,&w.comm,&w.stream)?;
            w.large.handoff_from(&w.verify)?;
        }
        w.large.prefill_wide(&mut w.wides,&tokens[..*prefix],&w.comm,&w.stream)?;
        w.verify.handoff_from(&w.large)?;
        let incoming=state(&w.verify)?;
        let mut expected:[Option<Vec<u8>>;2]=[None,None];
        for (trial,path) in [0usize,1,1,0].into_iter().enumerate() {
            w.control.agree(&[case as u8,trial as u8,path as u8])?;
            configure(&mut w.verify,path==1,&w.stream)?;
            let mut seconds=Vec::new();let mut original=Vec::new();
            for replay in 0..4 {
                let began=Instant::now();
                w.verify.begin_verification(&tokens[*prefix..*prefix+6],&w.comm,&w.stream)?;
                let elapsed=began.elapsed().as_secs_f64();
                let bytes=pending(w)?;
                w.verify.finish_verification(0,&w.comm,&w.stream)?;
                ensure!(state(&w.verify)?==incoming,"SWA precision abort changed retained state");
                if let Some(old)=&expected[path] {ensure!(*old==bytes,"SWA precision same-path repeat differs");}
                else{expected[path]=Some(bytes.clone());}
                if replay>0 {seconds.push(elapsed);}
                original=bytes;
            }
            let mut changed=tokens[*prefix..*prefix+6].to_vec();
            for token in &mut changed[3..] {*token=2+(*token+123)%120000;}
            w.verify.begin_verification(&changed,&w.comm,&w.stream)?;
            let suffix=pending(w)?;w.verify.finish_verification(0,&w.comm,&w.stream)?;
            ensure!(original[..3*129280*4]==suffix[..3*129280*4]
                &&state(&w.verify)?==incoming,"SWA changed rejected suffix affected causal prefix/state");
            let stem=format!("swa-c{case}-p{path}");
            if rank==3&&trial<2 {std::fs::write(out.join(format!("{stem}.bin")),&original)?;}
            records.push(json!({"case":case,"id":id,"prefix_tokens":prefix,"trial":trial,
                "path":path,"warm_forward_seconds":seconds,"logits_sha256":digest(&original),
                "stem":stem,"abort_and_suffix_exact":true}));
            std::fs::write(out.join("swa-precision-progress.json"),serde_json::to_vec_pretty(&records)?)?;
        }
        eprintln!("rank{rank} SWA precision case{case} passed same-path repeat, abort and suffix checks");
    }
    configure(&mut w.verify,false,&w.stream)?;
    std::fs::write(out.join("swa-precision-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"all_repeat_abort_and_suffix_checks_exact":true,
        "precision_scope":if source20 {"source20_projection"}else{"swa_layers_0_and_1"},
        "fixture_sha256":[digest(raw),digest(long_raw)],
        "scope":"Three public accepted prefixes; see precision_scope for the isolated operator. Fixed allocations, ABBA, three warm forwards per arm; same-path repeats, rejected-suffix isolation and rollback. Full-vocabulary CUDA comparison required. No sampling, serving selection or completed-task quality claim."
    }))?)?;
    Ok(())
}

/// M16 promotion gate: partial acceptance, changed rejected suffix and a
/// following M6 step, on two public prefixes including a carry-ring wrap.
pub(super) fn run_wide(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    let sparse=std::env::var("CANADA_SCRATCH_SPARSE_PRECISION_WIDE_SCREEN").as_deref()==Ok("1");
    ensure!(!sparse||std::env::var("CANADA_SCRATCH_SWA_PRECISION_WIDE_SCREEN").as_deref()!=Ok("1"),
        "one wide attention comparison per load");
    let configure=|target:&mut Target,enabled:bool,stream:&Stream|->Result<()> {
        if sparse {target.set_split_attention_for_probe(if enabled {2}else{0},stream)}
        else {target.set_swa_precision_for_probe(enabled,stream)}
    };
    ensure!(w.current.is_none()&&w.history.frontier()==0&&w.large.profile_is_current()
        &&w.large.position()==0&&w.long_verify.is_some(),"wide SWA screen needs fresh owners");
    let tool_raw=include_bytes!("../tests/fixtures/long-history-score-v1.json");
    let tools:Value=serde_json::from_slice(tool_raw)?;
    let tool:Vec<u32>=serde_json::from_value(tools["cases"][6]["prompt_ids"].clone())?;
    let long_raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let long:Value=serde_json::from_slice(long_raw)?;
    let cases=[("tool-history",tool.len()-22,tool),
        ("long-carry-wrap",131097,serde_json::from_value(long["tokens"].clone())?)];
    let mut records=Vec::new();
    for(case,(id,prefix,tokens))in cases.iter().enumerate() {
        ensure!(tokens.len()>=prefix+22,"wide SWA fixture extent");
        if case>0 {
            w.target_lineage=w.target_lineage.checked_add(1).context("wide SWA lineage")?;
            w.verify.reset(w.target_lineage,&w.comm,&w.stream)?;
            w.large.handoff_from(&w.verify)?;
        }
        w.large.prefill_wide(&mut w.wides,&tokens[..*prefix],&w.comm,&w.stream)?;
        w.verify.handoff_from(&w.large)?;
        let bytes=w.verify.compact_state_image_bytes()?;
        ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(
            w.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+bytes)?,
            "wide SWA bounded snapshot reserve");
        let image=w.verify.save_state_image(bytes,&w.comm,&w.stream)?;
        let incoming=state(&w.verify)?;
        let mut expected:[Option<Value>;2]=[None,None];
        for(trial,path)in[0usize,1,1,0].into_iter().enumerate() {
            let mut times=Vec::new();
            // Two warm forwards, four measurements, then one suffix mutation.
            for replay in 0..7 {
                let changed=replay==6;
                w.target_lineage=w.target_lineage.checked_add(1).context("wide SWA lineage")?;
                w.verify.restore_state_image(&image,w.target_lineage,&w.comm,&w.stream)?;
                ensure!(state(&w.verify)?==incoming,"wide SWA restore differs");
                w.control.agree(&[case as u8,trial as u8,path as u8,replay as u8])?;
                let target=w.long_verify.as_mut().unwrap();
                target.handoff_from(&w.verify)?;
                if replay==0 {configure(target,path==1,&w.stream)?;}
                let mut ids=tokens[*prefix..*prefix+16].to_vec();
                if changed {for token in &mut ids[7..] {*token=2+(*token+123)%120000;}}
                let at=Instant::now();target.begin_verification(&ids,&w.comm,&w.stream)?;
                let seconds=at.elapsed().as_secs_f64();
                if(2..6).contains(&replay){times.push(seconds);}
                let(y,n)=target.pending_logits()?;ensure!(n==16,"wide SWA logit extent");
                let logits=unsafe{w.cuda.read_device(y.ptr(),16*129280*4)}?;
                target.verification_sample(103200,&w.stream)?;
                target.finish_verification(7,&w.comm,&w.stream)?;
                w.verify.handoff_from(target)?;
                let accepted=state(&w.verify)?;
                w.verify.begin_verification(&tokens[*prefix+7..*prefix+13],&w.comm,&w.stream)?;
                let following=pending(w)?;
                w.verify.verification_sample(103201,&w.stream)?;
                w.verify.finish_verification(0,&w.comm,&w.stream)?;
                ensure!(state(&w.verify)?==accepted&&w.verify.position()==(*prefix+7)as u64,
                    "wide SWA following abort changed state");
                let signature=json!({"prefix_sha256":digest(&logits[..7*129280*4]),
                    "following_sha256":digest(&following),"accepted":accepted});
                if let Some(old)=&expected[path] {ensure!(*old==signature,
                    "wide SWA same-path repeat or rejected-suffix isolation differs");}
                else{expected[path]=Some(signature.clone());}
                if rank==3&&trial<2&&replay==0 {
                    let stem=format!("swa-wide-c{case}-p{path}");
                    std::fs::write(out.join(format!("{stem}-logits.bin")),&logits)?;
                    std::fs::write(out.join(format!("{stem}-following.bin")),&following)?;
                }
            }
            records.push(json!({"case":case,"id":id,"prefix_tokens":prefix,"trial":trial,
                "path":path,"warm_forward_seconds":times,"signature":expected[path],
                "partial7_following_and_suffix_exact":true}));
            std::fs::write(out.join("swa-wide-progress.json"),serde_json::to_vec_pretty(&records)?)?;
        }
        let target=w.long_verify.as_mut().unwrap();target.handoff_from(&w.verify)?;
        configure(target,false,&w.stream)?;w.verify.handoff_from(target)?;
        eprintln!("rank{rank} wide SWA case{case}: partial7, following M6 and suffix checks passed");
    }
    std::fs::write(out.join("swa-wide-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"all_partial_following_and_suffix_checks_exact":true,
        "precision_scope":if sparse {"sparse_layers_2_to_39"}else{"swa_layers_0_and_1"},
        "fixture_sha256":[digest(tool_raw),digest(long_raw)],
        "scope":"M16 attention comparison (see precision_scope); ABBA on two fixed public prefixes. Partial7 acceptance, following M6, changed rejected suffix and restore, including carry wrap. No serving promotion or completed-task quality claim."
    }))?)?;Ok(())
}
