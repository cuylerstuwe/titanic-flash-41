//! Same-load causal comparison of disabled and rolling proactive checkpoints.
//! Public token fixture; no HTTP timing or production-policy change.
use super::*;
use super::cache_slots_probe::{execute,following};
use super::interleave_http::idle_anchor_policy::{Mode,Policy};

fn entries(w:&Worker)->Vec<Value> {
    w.cached_sessions.iter().map(|c|json!({"owner":c.owner,"tokens":c.prefix.len(),
        "prefix_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&c.prefix).unwrap())),
        "bytes":c.charge,"speculative":c.speculative})).collect()
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.encoder_prompt_checkpoint&&w.encoder_active_frontier&&w.history_lookup_enabled
        &&crate::config::bank_encoder_checkpoint_requested()&&w.current.is_none()
        &&w.suspended_image_count==0&&w.session_cache_budget==2usize<<30
        &&w.session_cache_slots==8&&w.encoder_duplicate_skip,
        "idle policy probe requires accumulated ordinary cache settings");
    let raw=include_bytes!("../tests/fixtures/long-context-verification-corpus-v1.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let source:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let frontier=16385;let follow=&source[131200..131206];
    // Start, long append, short append, and subsequent forward turns exercise
    // both useful predictions and stale unused ones. Enough ordinary appends
    // follow to reach the existing eight-entry bound without shrinking it.
    let deltas=[0usize,256,64,256,256,16,256,128,256,64,256,256];
    let modes=[(Mode::Off,false),(Mode::Off,false),(Mode::EveryEligibleGap,true),
        (Mode::EveryEligibleGap,true),(Mode::Off,false)];
    let mut records=Vec::new();let mut references=std::collections::BTreeMap::<usize,(String,Value,Vec<u8>)>::new();
    let mut id=9_130_000;
    for (arm,(mode,refresh)) in modes.into_iter().enumerate() {
        w.control.agree(&[0xd1,arm as u8])?;w.stream.synchronize()?;
        w.cached_sessions.clear();let budget=w.session_cache_budget;w.session_cache_budget=0;id+=1;
        execute(w,Command::Begin{images:Vec::new(),id,seed:20261104,greedy:false,limit:1,tokens:source[..127].to_vec()})?;
        execute(w,Command::Finish(Finish::Length))?;
        w.cached_sessions.clear();w.session_cache_budget=budget;
        let mut policy=Policy::with_mode(mode);
        for (turn,&delta) in deltas.iter().enumerate() {
            let tokens=if turn==0{source[..frontier].to_vec()}else{
                let mut tokens=w.history.committed().to_vec();
                tokens.push(w.history.pending().context("idle policy continuation token")?);
                let offset=frontier+1000+turn*512;
                tokens.extend_from_slice(&source[offset..offset+delta-1]);tokens
            };
            let input_sha256=format!("{:x}",Sha256::digest(serde_json::to_vec(&tokens)?));
            let before=entries(w);id+=1;let at=Instant::now();
            let begin=execute(w,Command::Begin{images:Vec::new(),id,seed:20261104+turn as u64,
                greedy:false,limit:9,tokens})?;
            let begin_seconds=at.elapsed().as_secs_f64();
            ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.,"idle policy sampling");
            if turn>0 {ensure!(begin["retained_session_cache"]["active_reuse"]==true
                &&begin["uncached_tokens"]==delta,"idle policy lost live forward reuse");}
            policy.began(begin["retained_session_cache"]["active_reuse"]==true,
                begin["cached_tokens"].as_u64().context("cached token count")? as usize,
                begin["uncached_tokens"].as_u64().context("uncached token count")? as usize);
            let after_begin=entries(w);
            let semantic=following(w,follow)?;
            let (input_matched,exact)=if let Some((input,value,rows))=references.get(&turn) {
                let matched=input==&input_sha256;let exact=value==&semantic.0;
                if matched&&!exact {
                    std::fs::write(out.join(format!("idle-policy-a{arm}-t{turn}-reference.f32")),rows)?;
                    std::fs::write(out.join(format!("idle-policy-a{arm}-t{turn}-candidate.f32")),&semantic.1)?;
                }
                (matched,exact)
            }else{references.insert(turn,(input_sha256.clone(),semantic.0.clone(),semantic.1));(true,true)};
            let mut generated:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;
            let decode_at=Instant::now();
            while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
                generated.extend(serde_json::from_value::<Vec<u32>>(execute(w,Command::Step)?["output_ids"].clone())?);
            }
            let decode_seconds=decode_at.elapsed().as_secs_f64();
            execute(w,Command::Finish(if w.history.pending()==Some(w.cfg.eos_token_id){Finish::Stop}else{Finish::Length}))?;
            policy.finished(true,false);
            let prediction=policy.take();
            let free=w.idle_anchor_has_free_slot()||(refresh&&w.cached_sessions.iter().any(|c|c.speculative));
            let before_idle=entries(w);let semantic_before=following(w,follow)?.0;
            let at=Instant::now();
            let idle=if prediction&&free {
                // Include the command-agreement cost. HTTP ingress veto and
                // actual request overlap were checked separately; this screen
                // does not pretend to simulate an unknown inter-request gap.
                w.control.agree(&[0xd2,arm as u8,turn as u8])?;
                if refresh{w.refresh_idle_bank_encoder_for_probe()?}else{w.retain_idle_bank_encoder()?}
            }else{json!({"saved":false,"policy_veto":!prediction,"no_free_slot":!free})};
            let idle_seconds=at.elapsed().as_secs_f64();
            let after_idle=entries(w);
            ensure!(before_idle.iter().filter(|e|e["speculative"]!=true)
                .all(|e|after_idle.contains(e)),"prediction displaced an ordinary cache entry");
            if !refresh {ensure!(after_idle.starts_with(&before_idle),"nonrefresh prediction displaced prior cache");}
            ensure!(following(w,follow)?.0==semantic_before,"prediction changed live state or logits");
            ensure!(w.cached_sessions.len()<=8&&w.cached_sessions.iter().map(|c|c.charge).sum::<usize>()<=budget
                &&w.cached_sessions.iter().filter(|c|c.speculative).count()<=1,"idle policy bounds");
            records.push(json!({"arm":arm,"warmup":arm==0,"mode":format!("{mode:?}"),"refresh":refresh,"turn":turn,"delta":delta,
                "begin":begin,"begin_seconds":begin_seconds,"decode_seconds":decode_seconds,"generated":generated,
                "input_sha256":input_sha256,"input_matches_first_arm":input_matched,
                "semantic_matches_first_arm":exact,"semantic":semantic.0,"entries_before":before,
                "entries_after_begin":after_begin,"entries_before_idle":before_idle,"entries_after_idle":after_idle,
                "idle":idle,"idle_seconds":idle_seconds,"prediction":prediction,"free_slot":free}));
            std::fs::write(out.join("idle-policy-partial.json"),serde_json::to_vec_pretty(&records)?)?;
            eprintln!("idle policy rank{rank} arm{arm} {mode:?} turn{turn}: begin{:.3}ms idle{:.3}ms saved{}",
                begin_seconds*1000.,idle_seconds*1000.,records.last().unwrap()["idle"]["saved"]);
        }
    }
    w.stream.synchronize()?;w.cached_sessions.clear();execute(w,Command::Shutdown)?;
    std::fs::write(out.join("idle-policy-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "scope":"Same allocation, one unmeasured warmup then disabled/rolling/rolling/disabled. Twelve public16K forward turns per arm; T1/P1, short diagnostic generations. Existing eight entries/twoGiB budget, protects all ordinary images, at most one unconsumed prediction. Replaces only a same-bank forward ancestor with compatible byte storage. Server-native begin timing only; idle cost is not assumed hidden, and real arrival/HTTP/quality benefits require later evidence."}))?)?;Ok(())
}
