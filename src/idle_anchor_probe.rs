//! Anticipatory checkpoint mechanism screen; never scheduled by HTTP.
use super::*;
use super::cache_slots_probe::{execute,following};

fn entries(w:&Worker)->Vec<(u8,Vec<u32>,usize)> {
    w.cached_sessions.iter().map(|c|(c.owner,c.prefix.clone(),c.charge)).collect()
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.encoder_prompt_checkpoint&&w.encoder_active_frontier&&w.history_lookup_enabled
        &&crate::config::bank_encoder_checkpoint_requested()&&w.current.is_none()
        &&w.suspended_image_count==0&&w.session_cache_budget==2usize<<30,
        "idle anchor finite screen requires ordinary encoder cache admission");
    let raw=include_bytes!("../tests/fixtures/long-context-verification-corpus-v1.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let duplicate_screen=std::env::var("CANADA_SCRATCH_IDLE_DUPLICATE_SCREEN").as_deref()==Ok("1");
    let copy_screen=std::env::var("CANADA_SCRATCH_IDLE_COPY_SCREEN").as_deref()==Ok("1");
    ensure!(!copy_screen||!duplicate_screen,"idle copy and duplicate screens must be separate");
    w.control.agree(&[u8::from(duplicate_screen),u8::from(copy_screen)])?;
    let frontier=16385usize;let follow=&tokens[131200..131206];
    // Leave room for the preceding encoder-recycle probe when sharing a load.
    let mut records=Vec::new();let mut id=if copy_screen{9_104_000u64}else{9_102_000};
    let mut references=std::collections::BTreeMap::<usize,(String,Value,Vec<u8>)>::new();
    let arms=if copy_screen{[(true,true);4]}
        else if duplicate_screen{[(true,false),(true,true),(true,true),(true,false)]}
        else{[(false,true),(true,true),(true,true),(false,true)]};
    for(arm,(proactive,duplicate_skip))in arms.into_iter().enumerate() {
        w.encoder_duplicate_skip=duplicate_skip;
        if copy_screen {crate::state_image::copy::set_for_probe([false,true,true,false][arm])?;}
        for case in 0..if copy_screen{1usize}else{2} {
            w.control.agree(&[arm as u8,case as u8,proactive as u8,duplicate_skip as u8])?;
            w.stream.synchronize()?;w.cached_sessions.clear();
            let budget=w.session_cache_budget;w.session_cache_budget=0;id+=1;
            execute(w,Command::Begin{images:Vec::new(),id,seed:20261103,greedy:false,limit:1,
                tokens:tokens[..127].to_vec()})?;
            execute(w,Command::Finish(Finish::Length))?;
            w.cached_sessions.clear();w.session_cache_budget=budget;id+=1;
            let base=execute(w,Command::Begin{images:Vec::new(),id,seed:20261103,greedy:false,limit:9,
                tokens:tokens[..frontier].to_vec()})?;
            let mut generated:Vec<u32>=serde_json::from_value(base["output_ids"].clone())?;
            while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
                generated.extend(serde_json::from_value::<Vec<u32>>(execute(w,Command::Step)?["output_ids"].clone())?);
            }
            execute(w,Command::Finish(if w.history.pending()==Some(w.cfg.eos_token_id){Finish::Stop}else{Finish::Length}))?;
            let prior=entries(w);let prior_semantic=following(w,follow)?.0;
            let committed=w.history.frontier();
            // A no-capacity prediction must preserve every existing image.
            // This is a logical refusal, without inducing physical pressure.
            let refusal=if proactive {
                w.session_cache_budget=prior.iter().map(|c|c.2).sum();
                let r=w.retain_idle_bank_encoder()?;w.session_cache_budget=budget;
                ensure!(r["saved"]==false&&entries(w)==prior,"idle refusal altered retained cache");Some(r)
            }else{None};
            let idle_at=Instant::now();
            let idle=if proactive{w.retain_idle_bank_encoder()?}else{json!({"saved":false,"disabled":true})};
            let idle_seconds=idle_at.elapsed().as_secs_f64();
            let after_idle=entries(w);
            ensure!(after_idle.starts_with(&prior),"idle prediction evicted or replaced an existing checkpoint");
            ensure!(following(w,follow)?.0==prior_semantic,"idle preparation changed live state or following logits");
            if proactive {
                ensure!(idle["saved"]==true&&idle["evicted_sessions"]==0&&idle["pressure_evicted_sessions"]==0,
                    "idle preparation did not exercise a new checkpoint: {idle}");
                if copy_screen {ensure!(idle["fused_copy"]==[false,true,true,false][arm],
                    "idle copy screen did not exercise the selected byte-copy path");}
                let duplicate=w.retain_idle_bank_encoder()?;
                ensure!(duplicate["prediction_pending"]==true&&entries(w)==after_idle,"duplicate idle work changed storage");
            }
            let saved_prefix=w.history.committed().to_vec();
            let next=if case==0 {
                let mut next=w.history.committed().to_vec();
                next.push(w.history.pending().context("idle forward pending token")?);
                next.extend_from_slice(&tokens[frontier+1000..frontier+1255]);next
            }else if duplicate_screen {
                // A short append does not need the predicted checkpoint. Its
                // full preparation cost and retained bytes are wasted work.
                let mut next=w.history.committed().to_vec();
                next.push(w.history.pending().context("idle short pending token")?);
                next.extend_from_slice(&tokens[frontier+2000..frontier+2063]);next
            }else{
                // Rewind before the generated answer and choose a new suffix.
                let mut next=tokens[..frontier].to_vec();
                next.extend_from_slice(&tokens[frontier+2000..frontier+2256]);next
            };
            let input_hash=format!("{:x}",Sha256::digest(serde_json::to_vec(&next)?));
            id+=1;let at=Instant::now();
            let begin=execute(w,Command::Begin{images:Vec::new(),id,seed:20261113+case as u64,
                greedy:false,limit:1,tokens:next})?;
            let begin_seconds=at.elapsed().as_secs_f64();
            ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.,"idle screen sampling profile");
            if case==0 {
                ensure!(begin["cached_tokens"]==committed&&begin["retained_session_cache"]["active_reuse"]==true,
                    "idle checkpoint changed active continuation selection");
                ensure!(begin["prompt_checkpoint"]["already_cached"]==proactive
                    &&begin["prompt_checkpoint"]["saved"]==!proactive,
                    "forward request did not reuse proactive checkpoint: {begin}");
                ensure!(begin["prompt_checkpoint"]["admission_skipped_duplicate"]==(proactive&&duplicate_skip),
                    "encoder duplicate path was not exercised as configured");
                if proactive&&duplicate_skip {
                    ensure!(begin["prompt_checkpoint"]["forecast_refused"]==false
                        &&begin["prompt_checkpoint"]["admission_available_bytes"].is_null(),
                        "duplicate reported a memory refusal or unmeasured memory");
                }
            }else if duplicate_screen {
                ensure!(begin["cached_tokens"]==committed&&begin["uncached_tokens"]==64
                    &&begin["retained_session_cache"]["active_reuse"]==true,
                    "short wrong prediction changed live reuse");
            }
            let semantic=following(w,follow)?;
            let(matched,exact)=if let Some((hash,value,rows))=references.get(&case) {
                let matched=hash==&input_hash;let exact=value==&semantic.0;
                if matched&&!exact {
                    std::fs::write(out.join(format!("idle-anchor-a{arm}-c{case}-reference.f32")),rows)?;
                    std::fs::write(out.join(format!("idle-anchor-a{arm}-c{case}-candidate.f32")),&semantic.1)?;
                }
                (matched,exact)
            }else{references.insert(case,(input_hash.clone(),semantic.0.clone(),semantic.1));(true,true)};
            execute(w,Command::Finish(Finish::Length))?;
            let restored=if case==0&&!duplicate_screen {
                // The first request used the live state. Now actually restore
                // the saved prefix after advancing beyond it, proving the
                // prepared image remains a valid rewind anchor too.
                let mut branch=saved_prefix.clone();
                branch.extend_from_slice(&tokens[frontier+3000..frontier+3128]);
                id+=1;let at=Instant::now();
                let restored=execute(w,Command::Begin{images:Vec::new(),id,seed:20261121,
                    greedy:false,limit:1,tokens:branch})?;
                let seconds=at.elapsed().as_secs_f64();
                ensure!(restored["cached_tokens"]==committed&&restored["retained_session_cache"]["encoder_restored"]==true,
                    "prepared anchor was not actually restored: {restored}");
                let check=following(w,follow)?;
                let exact=if let Some((_,value,rows))=references.get(&2) {
                    let exact=value==&check.0;
                    if !exact {
                        std::fs::write(out.join(format!("idle-anchor-a{arm}-restore-reference.f32")),rows)?;
                        std::fs::write(out.join(format!("idle-anchor-a{arm}-restore-candidate.f32")),&check.1)?;
                    }
                    exact
                }else{references.insert(2,(String::new(),check.0.clone(),check.1));true};
                execute(w,Command::Finish(Finish::Length))?;
                Some(json!({"begin":restored,"seconds":seconds,"semantic":check.0,"exact":exact}))
            }else{None};
            let charge=w.cached_sessions.iter().map(|c|c.charge).sum::<usize>();
            ensure!(charge<=budget&&w.cached_sessions.len()<=w.session_cache_slots,"idle cache exceeded ordinary bounds");
            records.push(json!({"arm":arm,"proactive":proactive,"duplicate_skip":duplicate_skip,
                "copy_screen":copy_screen,"copy_kernel_enabled":crate::state_image::copy::enabled(),
                "short_append_screen":duplicate_screen,"case":case,"frontier":frontier,
                "committed":committed,"generated":generated,"idle":idle,"refusal":refusal,
                "idle_seconds":idle_seconds,"begin":begin,"begin_seconds":begin_seconds,"restore":restored,
                "input_sha256":input_hash,"input_matches_first_arm":matched,"semantic":semantic.0,
                "semantic_matches_first_arm":exact,"entries_before":prior.len(),"entries_after_idle":after_idle.len(),
                "bytes_before":prior.iter().map(|c|c.2).sum::<usize>(),"bytes_after_idle":after_idle.iter().map(|c|c.2).sum::<usize>(),
                "bytes_after_request":charge,"memory":w.cuda.admission_memory()?}));
            std::fs::write(out.join("idle-anchor-partial.json"),serde_json::to_vec_pretty(&records)?)?;
            eprintln!("idle anchor rank{rank} arm{arm} case{case} proactive{proactive}: idle{:.3}ms begin{:.3}ms",
                idle_seconds*1000.,begin_seconds*1000.);
        }
    }
    w.stream.synchronize()?;w.cached_sessions.clear();execute(w,Command::Shutdown)?;
    std::fs::write(out.join("idle-anchor-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "scope":if copy_screen {
            "Same-load ABBA bank-encoder byte-copy path off/on. Four16K public-prefix cases with proactive save, forward256-token continuation, then actual restoration after advancement. Live state and following six FP32 logit rows compared exactly. Ordinary budgets, admission and checkpoint policy; no HTTP latency or production claim."
        }else if duplicate_screen {
            "Same-load ABBA duplicate admission on/off, with proactive checkpoints in every arm. Real16K public prefix, nine generated tokens at T1/P1, forward256-token and short64-token continuation. The short append does not require the preparation. Live state and six following logit rows compared exactly. No real arrival-latency bound or production claim."
        }else{
            "Same-load ABBA anticipatory checkpoint screen. Real16K public prefix, nine generated tokens at T1/P1, forward256-token continuation and wrong-prediction rewind. Free optional capacity only; refusal and duplicate preserve existing images. Actual saved-prefix restoration after advancement. Live state and six following logit rows compared exactly. No real arrival-latency bound or production claim."
        }}))?)?;
    Ok(())
}
