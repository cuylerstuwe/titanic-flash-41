//! Paired-history diagnostic only. No generated answer or performance claim.
use super::*;

const FIXTURE:&[u8]=include_bytes!("../tests/fixtures/matched-paths-v1.json");
fn economics_enabled()->bool {
    std::env::var("CANADA_SCRATCH_MATCHED_PRECISION_ECONOMICS").as_deref()==Ok("1")
}
fn factorial_enabled()->Result<bool> {
    match std::env::var("CANADA_SCRATCH_MATCHED_PRECISION_FACTORIAL").as_deref() {
        Ok("1")=>Ok(true),Ok("0")|Err(std::env::VarError::NotPresent)=>Ok(false),
        _=>anyhow::bail!("invalid matched precision factorial switch"),
    }
}

pub(super) fn validate()->Result<()> {
    ensure!(!factorial_enabled()?||economics_enabled(),"factorial screen requires precision economics");
    if economics_enabled() {
        ensure!(std::env::var("CANADA_SCRATCH_SOURCE20_PRECISION_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_SWA_PRECISION_SCREEN").as_deref()==Ok("1")
            &&!crate::compressed_source::source20_fp32_reduction_enabled()?
            &&!crate::layer_attention::generation_swa_bf16_enabled()?
            &&!crate::config::history32_enabled(),
            "matched precision economics needs both finite switches, legacy constructors and no M32");
    }
    let f:Value=serde_json::from_slice(FIXTURE)?;
    ensure!(f["schema"]=="matched-path-v1"&&f["temperature"]==1&&f["top_p"]==1
        &&f["top_k"]==-1&&f["cases"].as_array().is_some_and(|x|x.len()==6)
        &&f["frontiers"]==json!([0,48,192,336])&&f["proposal_seeds"].as_array().is_some_and(|x|x.len()==4),
        "matched path fixture schema");
    for c in f["cases"].as_array().unwrap() {
        let input:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
        let output:Vec<u32>=serde_json::from_value(c["output_ids"].clone())?;
        ensure!((64..=131072).contains(&input.len())&&output.len()==342
            &&input.iter().chain(&output).all(|&x|x<129280),"matched token geometry");
    }
    Ok(())
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    validate()?;
    ensure!(std::env::var("CANADA_SCRATCH_MATCHED_PATH_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none()&&w.session_cache_budget==0&&!w.prompt_checkpoint
        &&!w.encoder_prompt_checkpoint&&!w.prompt_cache_logits&&w.verify.physical_rows()==6,
        "matched path needs isolated uncached file worker");
    let f:Value=serde_json::from_slice(FIXTURE)?;
    let economics=economics_enabled();
    let factorial=factorial_enabled()?;
    let mut seeds:Vec<u64>=serde_json::from_value(f["proposal_seeds"].clone())?;
    // Preserve the original four draws as a numerical bridge to the frozen
    // baseline/candidate diagnostics. Additional values are RNG seeds only.
    if economics {seeds.extend((0..if factorial{12}else{28}).map(|i|0x5eed_2026_1101_0000+i));}
    ensure!(seeds.iter().collect::<std::collections::BTreeSet<_>>().len()==seeds.len(),"unique proposal seeds");
    let frontiers:Vec<usize>=serde_json::from_value(f["frontiers"].clone())?;
    let mut shadow=crate::rejection::Rejection::new(&w.cuda,&w.core,6)?;
    shadow.enable_history_expectation_probe()?;
    let mut records=Vec::new();
    for(case,c)in f["cases"].as_array().unwrap().iter().enumerate() {
      let mut arms=if factorial {vec![(false,false),(true,true),(true,false),(false,true)]}
          else if economics {vec![(false,false),(true,true)]}else{vec![(false,false)]};
      if factorial {arms.rotate_left(case%4);}else if case%2==1 {arms.reverse();}
      for(trial,(source20,swa))in arms.into_iter().enumerate() {
        let arm=match(source20,swa){(false,false)=>"legacy",(true,true)=>"candidate",
            (true,false)=>"source20_only",(false,true)=>"swa_only"};
        let arm_id=(source20 as u8)*2+swa as u8;
        let input:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
        let output:Vec<u32>=serde_json::from_value(c["output_ids"].clone())?;
        ensure!(output.iter().all(|&x|x!=w.cfg.eos_token_id),"matched path crosses a terminal token");
        w.control.agree(&[case as u8,trial as u8,arm_id,0])?;
        if economics&&case==0&&trial==0 {w.verify.handoff_from(&w.large)?;}
        if w.verify.profile_is_current() {
            if economics {
                w.verify.set_source20_precision_for_probe(source20,&w.stream)?;
                w.verify.set_swa_precision_for_probe(swa,&w.stream)?;
            }
            w.target_lineage=w.target_lineage.checked_add(1).context("matched lineage exhausted")?;
            w.verify.reset(w.target_lineage,&w.comm,&w.stream)?;
            w.large.handoff_from(&w.verify)?;
        }else{ensure!(case==0&&w.large.profile_is_current()&&w.large.position()==0,"matched initial owner");}
        w.large.prefill_wide(&mut w.wides,&input,&w.comm,&w.stream)?;
        w.attach_prompt(false,0,&input,&input,seeds[0],false)?;
        w.verify.handoff_from(&w.large)?;
        for(offset,ids)in output.chunks_exact(6).enumerate().map(|(n,x)|(n*6,x)) {
            let selected=frontiers.contains(&offset);
            let observation=if selected {Some(w.session.as_mut().unwrap().matched_path_frontier(
                &mut w.verify,ids[0],&seeds,&mut shadow,&w.comm,&w.stream)?)}else{None};
            let logits=w.session.as_mut().unwrap().matched_path_commit(
                &mut w.verify,ids,selected,&w.comm,&w.stream)?;
            if let Some(observation)=observation {
                let bytes=logits.context("matched teacher logits absent")?;
                let file=if economics {format!("matched-c{case}-{arm}-p{offset}.f32")}
                    else{format!("matched-c{case}-p{offset}.f32")};
                let digest=format!("{:x}",Sha256::digest(&bytes));
                if rank==0{std::fs::write(out.join(&file),&bytes)?;}
                records.push(json!({"case":case,"id":c["id"],"family":c["family"],"origin":c["origin"],
                    "precision_arm":if economics {Some(arm)}else{None},"trial":trial,
                    "prompt_tokens":input.len(),"offset":offset,"observation":observation,
                    "teacher_logits":{"file":file,"rows":6,"bytes":bytes.len(),"sha256":digest},
                    "committed_frontier":w.verify.position()}));
                std::fs::write(out.join("matched-path-progress.json"),serde_json::to_vec_pretty(&records)?)?;
            }
        }
        ensure!(w.verify.position()==(input.len()+output.len())as u64,"matched final frontier");
        w.control.agree(&[case as u8,trial as u8,arm_id,1])?;
        eprintln!("matched path rank{rank} case{case} trial{trial} arm{arm}: four fixed frontiers scored,342 forced inputs committed");
      }
    }
    ensure!(records.len()==if factorial{96}else if economics{48}else{24},"matched screen completeness");
    std::fs::write(out.join("matched-path-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"fixture_sha256":format!("{:x}",Sha256::digest(FIXTURE)),"records":records,
        "all_rollbacks_exact":true,"target_temperature":1,"target_top_p":1,"top_k":-1,
        "proposal_seeds":seeds,"same_load_precision_ablation":economics,"factorial_precision_ablation":factorial,
        "scope":"Six fixed public reasoning paths, balanced between saved s986 and s1158 text, canonically retokenized. Four prefix locations; proposal_seeds lists the distinct draws, followed by one same-seed repeat. Optional paired precision arms alternate order across cases on one set of model allocations, with independent prefill and forced commits per arm. Actual neural q and target verification, GPU-only probability calculations, zero-accept rollback, ordinary feature conditioning on six-row forced commits. Diagnostic proposal acceptance/cost opportunity, not task quality, serving speed, or the complete history-copy selection policy."}))?)?;
    Ok(())
}
