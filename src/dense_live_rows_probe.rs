//! Same-prefix current-model M128 row-bound comparison; all tensor math GPU.
use super::*;
use anyhow::Context;
fn execute(w:&mut Worker,c:Command)->Result<Value> {
    let bytes=c.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(bytes.as_slice()))?;
    ensure!(wire==bytes,"dense row screen command replicas disagree");w.execute(c)
}
fn state(w:&Worker)->Result<Value> {
    Ok(json!({"position":w.verify.position(),"carry":w.verify.carry_digests()?,
        "encoder":w.verify.encoder_tail_digest()?}))
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    let edge=std::env::var("CANADA_SCRATCH_DENSE_LIVE_ROWS_EDGE").as_deref()==Ok("1");
    let partial=std::env::var("CANADA_SCRATCH_DENSE_PARTIAL_SCREEN").as_deref()==Ok("1");
    if edge&&partial {
        let edges=out.join("short-edge");std::fs::create_dir(&edges)?;
        run_cases(w,rank,&edges,true,false)?;
        let partials=out.join("partial512");std::fs::create_dir(&partials)?;
        run_cases(w,rank,&partials,false,true)
    }else{run_cases(w,rank,out,edge,partial)}
}
fn run_cases(w:&mut Worker,rank:usize,out:&Path,edge:bool,partial:bool)->Result<()> {
    let mhc=std::env::var("CANADA_SCRATCH_PREFILL_MHC_ROWS_SCREEN").as_deref()==Ok("1");
    let norm=std::env::var("CANADA_SCRATCH_PREFILL_MHC_NORM_SCREEN").as_deref()==Ok("1");
    let columns=std::env::var("CANADA_SCRATCH_PREFILL_COLUMN_GATHER_SCREEN").as_deref()==Ok("1");
    let moe512=std::env::var("CANADA_SCRATCH_MOE_PARTIAL_SCREEN").as_deref()==Ok("1");
    let partial1024=std::env::var("CANADA_SCRATCH_PARTIAL1024_SCREEN").as_deref()==Ok("1");
    ensure!(!partial1024||(!moe512&&!mhc&&!partial&&!edge&&!columns),"M1024 screen is a separate paired case set");
    ensure!(!moe512||(!mhc&&!partial&&!edge&&!columns),"M512 expert screen is a separate paired case set");
    ensure!(!columns||(!mhc&&!partial&&!edge),"column gather screen is a separate paired case set");
    ensure!(!norm||mhc,"prefill normalization comparison requires the mHC screen");
    ensure!(!mhc||(!edge&&!partial),"prefill mHC screen is a separate paired case set");
    ensure!(std::env::var("CANADA_SCRATCH_DENSE_LIVE_ROWS_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none(),"finite dense live-row screen admission");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let f:Value=serde_json::from_slice(raw)?;let tokens:Vec<u32>=serde_json::from_value(f["tokens"].clone())?;
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let mut records=Vec::new();let mut expected=std::collections::BTreeMap::new();
    for(case,prefix)in (if edge{[0usize,1]}else if partial||mhc||columns||moe512||partial1024{[0,131071]}else{[65536usize,131071]}).into_iter().enumerate() {
        let resume=if prefix==0&&w.session.is_some(){
            let bytes=w.verify.compact_state_image_bytes()?;
            ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(
                w.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+bytes)?,
                "fresh diagnostic must budget the existing target owner");
            Some((w.verify.save_state_image(bytes,&w.comm,&w.stream)?,state(w)?))
        }else{None};
        if prefix>0 {
            execute(w,Command::Begin{images:Vec::new(),id:7_840_000+(if partial{100}else if mhc{200}else if columns{300}else if moe512{400}else if partial1024{500}else{0})+case as u64,
                seed:20261030,greedy:false,limit:1,tokens:tokens[..prefix].to_vec()})?;
        }else{
            w.target_lineage=w.target_lineage.checked_add(1).context("dense probe lineage exhausted")?;
            w.verify.reset(w.target_lineage,&w.comm,&w.stream)?;
        }
        // This probe advances only the target, never the drafter or its
        // accepted-history lookup. Keep their original state untouched and
        // restore the matching target-only image before each comparison.
        let bytes=if prefix>0{w.verify.compact_state_image_bytes()?}else{0};
        ensure!(bytes<=1024usize.pow(3),"dense row screen image extent");
        ensure!(w.control.all_admit(w.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+bytes)?,
            "dense row screen image reserve");
        let saved=if prefix>0{Some(w.verify.save_state_image(bytes,&w.comm,&w.stream)?)}else{None};
        let original=state(w)?;
        let variants=if edge||partial||mhc||columns||moe512{if case==0{vec![0usize,2]}else{vec![2,0]}}
            else if case==0{vec![0usize,1,2]}else{vec![2,1,0]};
        for variant in variants {
            let enabled=variant>0;let experts=variant==2;
            let samples=if partial1024{vec![(513usize,200000usize),(769,200800),(513,220000)]}else if norm{vec![(17usize,200000usize),(129,200200),(385,201000),(129,220200)]}
                else if mhc{vec![(17usize,200000usize),(64,200064),(129,200200),(257,200400),(385,201000),(17,220000),(129,220200)]}
                else if partial||columns||moe512{vec![(129usize,200000usize),(257,200200),(385,201000),(129,220000)]}
                else{vec![(17usize,200000usize),(64,200064),(17,220000)]};
            for(sample,(delta,offset))in samples.into_iter().enumerate() {
                for repeat in 0..2 {
                    w.target_lineage=w.target_lineage.checked_add(1).context("dense probe lineage exhausted")?;
                    if let Some(saved)=&saved{w.verify.restore_state_image(saved,w.target_lineage,&w.comm,&w.stream)?;}
                    else{w.verify.reset(w.target_lineage,&w.comm,&w.stream)?;}
                    ensure!(state(w)?==original,"dense screen incoming state differs");
                    w.large.handoff_from(&w.verify)?;
                    if partial1024 {
                        let wide=w.wides.iter_mut().find(|p|p.physical_rows()==1024).context("M1024 profile missing")?;
                        wide.set_partial1024_rows_for_probe(variant as u8,&w.stream)?;
                    }else if moe512 {
                        let wide=w.wides.iter_mut().find(|p|p.physical_rows()==512).context("M512 profile missing")?;
                        wide.set_partial_moe_rows_for_probe(enabled,&w.stream)?;
                    }else if columns {
                        w.comm.set_column_gather_rows_for_probe(enabled,&w.stream)?;
                    }else if mhc {
                        w.large.set_prefill_mhc_rows_for_probe(enabled||norm,enabled&&norm,&w.stream)?;
                        let wide=w.wides.iter_mut().find(|p|p.physical_rows()==512).context("M512 profile missing")?;
                        wide.set_prefill_mhc_rows_for_probe(enabled||norm,enabled&&norm,&w.stream)?;
                    }else if partial {
                        let wide=w.wides.iter_mut().find(|p|p.physical_rows()==512).context("M512 profile missing")?;
                        wide.set_partial_dense_rows_for_probe(enabled,&w.stream)?;
                    }else{w.large.set_dense_live_rows_for_probe(enabled,experts,&w.stream)?;}
                    let captures=w.large.capture_statistics().1;
                    w.control.agree(&[case as u8,variant as u8,sample as u8,repeat])?;
                    let began=Instant::now();
                    let experts_before=crate::moe::prefill_wide_bound_calls();
                    let gather_before=w.comm.column_gather_prefix_calls();
                    w.large.prefill_wide(&mut w.wides,&tokens[offset..offset+delta],&w.comm,&w.stream)?;
                    let seconds=began.elapsed().as_secs_f64();
                    let live_gather_calls=w.comm.column_gather_prefix_calls()-gather_before;
                    if columns {ensure!((live_gather_calls>0)==enabled,"column gather comparison did not exercise intended path");}
                    let live_expert_calls=crate::moe::prefill_wide_bound_calls()-experts_before;
                    if moe512 {ensure!((live_expert_calls>0)==enabled,"M512expertcomparison did not exercise intended path");}
                    if partial1024 {ensure!((live_expert_calls>0)==(variant==2),"M1024expertcomparison did not exercise intended path");}
                    let new_captures=w.large.capture_statistics().1-captures;
                    ensure!(repeat==0||new_captures==0,"warm dense row screen recaptured");
                    let output=w.large.valid_output_digests()?;
                    let phases=serde_json::to_value(w.large.append_metrics()?)?;
                    w.verify.handoff_from(&w.large)?;let after=state(w)?;
                    w.verify.begin_verification(&tokens[offset+delta..offset+delta+6],&w.comm,&w.stream)?;
                    let(logits,n)=w.verify.pending_logits()?;ensure!(n==6,"dense screen following extent");
                    let following=unsafe{w.cuda.read_device(logits.ptr(),6*129280*4)}?;
                    w.verify.finish_verification(0,&w.comm,&w.stream)?;
                    ensure!(state(w)?==after,"dense screen following rollback differs");
                    let evidence=json!({"output":output,"state":after,"following_sha256":format!("{:x}",Sha256::digest(&following))});
                    if let Some(prior)=expected.get(&(case,sample)) {
                        ensure!(prior==&evidence,"dense row bounds changed valid output/state/following logits at case{case} sample{sample}");
                    }else{expected.insert((case,sample),evidence.clone());}
                    records.push(json!({"case":case,"prefix":prefix,"sample":sample,"delta":delta,"changed_input":if norm{sample==3}else if mhc{sample>=5}else{sample==if partial||columns||moe512{3}else{2}},
                        "enabled":enabled,"experts":experts,"variant":(["baseline","dense","combined"][variant]),
                        "repeat":repeat,"seconds":seconds,"new_captures":new_captures,"phases":phases,"live_gather_calls":live_gather_calls,"live_expert_calls":live_expert_calls,
                        "evidence":evidence,"following_rollback_exact":true}));
                    std::fs::write(out.join("dense-live-rows-partial.json"),serde_json::to_vec_pretty(&records)?)?;
                }
            }
        }
        w.target_lineage=w.target_lineage.checked_add(1).context("dense probe lineage exhausted")?;
        if let Some(saved)=&saved{w.verify.restore_state_image(saved,w.target_lineage,&w.comm,&w.stream)?;}
        else{w.verify.reset(w.target_lineage,&w.comm,&w.stream)?;}
        ensure!(state(w)?==original,"dense screen final state restore differs");
        if prefix>0{execute(w,Command::Finish(Finish::Length))?;}
        else if let Some((saved,prior))=resume {
            w.target_lineage=w.target_lineage.checked_add(1).context("dense probe lineage exhausted")?;
            w.verify.restore_state_image(&saved,w.target_lineage,&w.comm,&w.stream)?;
            ensure!(state(w)?==prior,"fresh screen did not return the real worker frontier");
        }else if w.session.is_none(){
            // Before its first real Begin, Worker assumes the initial large
            // profile owns the zero frontier. Our target-only fresh screen
            // must return that lease, as well as clearing its temporary state.
            w.large.handoff_from(&w.verify)?;
        }
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("dense-live-rows-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"all_outputs_state_and_following_exact":true,"fresh_and_odd_prefix":edge,"partial512":partial,"prefill_mhc":mhc,"prefill_mhc_norm":norm,"column_gather":columns,"partial_moe":moe512,"partial1024":partial1024,
        "fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "scope":if partial1024{"Fresh/131K513/769-rowprefills and changed513input, sixfollowinglogits. Baseline versus M1024 dense live rows versus dense+expert live rows, existing pinned kernels and allocations. All earlier row optimizations fixed in each variant. Enqueuecounter proves expertpath."}else if moe512{"Fresh/131K129/257/385-rowprefills and changed129input, sixfollowinglogits. Baseline versus pinned M512 MoE with live num_tokens/active_m; unchanged capacities, canonical weights and arithmetic. Enqueuecounter proves optimizedpath. Carried dense/mHC/gather policies fixed in both variants."}else if columns{"Fresh/131K129/257/385-rowprefills and changed129input, sixfollowinglogits. Baseline full M512 Engram column all-gather versus active-row NCCL byte gather and transpose withzeroedunusedsuffix. Enqueuecounter proves the optimizedpathwasexercised. No targetprecisionchange."}else if norm{"Fresh/131K17/129/385-rowprefills, changed129input, sixfollowinglogits. Baseline already trims mHC statistics and residuals; combined additionally trims existing fused combine/RMS rows. Active precision and allocations unchanged."}else if mhc{"Fresh and131K prefixes, M12817/64 andM512129/257/385 live rows, changed inputs, capture/replay and six following logits. Existing mHC statistics and residual arithmetic, only inactive rows skipped. Baseline and combined labels refer solely to this mHC switch; carried dense/expert bounds remain unchanged."}else if partial{"Fresh and131Kodd retained prefixes;129/257/385-rowappends and changed129-rowinput. Baseline versus existing M512densekernels withliveM. Exact fullvalidoutputs,retainedstate and6followinglogits. Scoped mechanismcheck,notwhole-taskquality."}else if edge{"Fresh and one-token odd-prefix17/64-row requests and changed17-row inputs. Baseline versus combined dense+expert bounds; fresh-to-retained graph reuse. Full validoutputs,carry,encoder and6following targetrows exact; rollback exact. Scoped qualification,notwhole-taskquality."}else{"Two65K/131Kpublic repository prefixes;17/64-rowappends and changed17-rowinput with capture reuse. Baseline,denseliveandcombineddense+expertbounds; same existing M128 kernels and arithmetic. Warm and first-use times separate. Full validoutputs,carry,encoder and6following targetrows exact; rollback exact. No whole-task or production qualification claim."}}))?)?;
    Ok(())
}
