//! Same-load eviction/reuse screen; normal cache policy and arithmetic remain.
use super::*;
use super::cache_slots_probe::{execute,following};

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.encoder_prompt_checkpoint&&w.encoder_active_frontier&&w.cache_pressure_reclaim
        &&crate::config::bank_encoder_checkpoint_requested()&&w.session_cache_slots==8
        &&w.session_cache_budget==2usize<<30&&w.current.is_none()&&w.suspended_image_count==0,
        "encoder reuse screen requires ordinary eight-slot bank cache");
    let raw=include_bytes!("../tests/fixtures/long-context-verification-corpus-v1.json");
    let data:Value=serde_json::from_slice(raw)?;let tokens:Vec<u32>=serde_json::from_value(data["tokens"].clone())?;
    let mut id=9_103_000;let mut records=Vec::new();
    let mut references=std::collections::BTreeMap::<usize,(String,Value,Vec<u8>)>::new();
    for(arm,recycle)in [false,true,true,false].into_iter().enumerate() {
        w.control.agree(&[arm as u8,recycle as u8])?;w.encoder_image_recycle=recycle;
        w.stream.synchronize()?;w.cached_sessions.clear();
        let budget=w.session_cache_budget;w.session_cache_budget=0;id+=1;
        execute(w,Command::Begin{images:Vec::new(),id,seed:20261103,greedy:false,limit:1,tokens:tokens[..127].to_vec()})?;
        execute(w,Command::Finish(Finish::Length))?;w.cached_sessions.clear();w.session_cache_budget=budget;id+=1;
        execute(w,Command::Begin{images:Vec::new(),id,seed:20261103,greedy:false,limit:1,tokens:tokens[..16385].to_vec()})?;
        execute(w,Command::Finish(Finish::Length))?;
        let mut branch_prefix=None;
        // Reach actual ordinary eviction, then measure three forward turns.
        // No test-only deletion or fabricated snapshot makes reuse possible.
        for step in 0..12usize {
            let mut next=w.history.committed().to_vec();let prefix=next.clone();
            next.push(w.history.pending().context("encoder recycle pending anchor")?);
            let at=33000+step*300;next.extend_from_slice(&tokens[at..at+255]);
            let input_hash=format!("{:x}",Sha256::digest(serde_json::to_vec(&next)?));
            id+=1;let at=Instant::now();
            let begin=execute(w,Command::Begin{images:Vec::new(),id,seed:20261113+step as u64,
                greedy:false,limit:1,tokens:next})?;
            let seconds=at.elapsed().as_secs_f64();
            ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.,"encoder reuse sampling");
            if step>=9 {
                ensure!(begin["prompt_checkpoint"]["saved"]==true
                    &&begin["prompt_checkpoint"]["evicted_sessions"].as_u64().unwrap_or(0)>0,
                    "encoder reuse did not exercise ordinary eviction: {begin}");
                let reused=begin["prompt_checkpoint"]["recycled_device_bytes"].as_u64().unwrap_or(0);
                ensure!((reused>0)==recycle,"selected compatible eviction did not exercise reuse");
                let semantic=following(w,&tokens[131200..131206])?;
                let exact=if let Some((hash,value,raw))=references.get(&step) {
                    let exact=hash==&input_hash&&value==&semantic.0;
                    if !exact {
                        std::fs::write(out.join(format!("encoder-recycle-a{arm}-s{step}-reference.f32")),raw)?;
                        std::fs::write(out.join(format!("encoder-recycle-a{arm}-s{step}-candidate.f32")),&semantic.1)?;
                    }exact
                }else{references.insert(step,(input_hash.clone(),semantic.0.clone(),semantic.1));true};
                ensure!(exact,"encoder reuse changed input, state or following logits");
                if step==10 {branch_prefix=Some(prefix);}
                records.push(json!({"arm":arm,"recycled":recycle,"step":step,"seconds":seconds,
                    "begin":begin,"input_sha256":input_hash,"state_and_logits_exact":exact,
                    "physical_image_bytes":crate::memory::retained_image_bytes(),"memory":w.cuda.admission_memory()?}));
                std::fs::write(out.join("encoder-recycle-partial.json"),serde_json::to_vec_pretty(&records)?)?;
            }
            execute(w,Command::Finish(Finish::Length))?;
        }
        // Restore a checkpoint captured into the reused allocation, after a
        // later forward turn has changed the live bank and its normal cache.
        let prefix=branch_prefix.context("missing recycled branch prefix")?;
        let mut branch=prefix.clone();branch.extend_from_slice(&tokens[65000..65128]);id+=1;
        let begin=execute(w,Command::Begin{images:Vec::new(),id,seed:20261129,greedy:false,limit:1,tokens:branch})?;
        ensure!(begin["cached_tokens"]==prefix.len()&&begin["retained_session_cache"]["encoder_restored"]==true,
            "saved recycled checkpoint was not actually restored");
        let semantic=following(w,&tokens[131200..131206])?;
        let exact=if let Some((_,value,raw))=references.get(&100) {
            let exact=value==&semantic.0;
            if !exact {
                std::fs::write(out.join(format!("encoder-recycle-a{arm}-restore-reference.f32")),raw)?;
                std::fs::write(out.join(format!("encoder-recycle-a{arm}-restore-candidate.f32")),&semantic.1)?;
            }exact
        }else{references.insert(100,(String::new(),semantic.0,semantic.1));true};
        ensure!(exact,"restoring recycled storage changed state or following logits");
        records.push(json!({"arm":arm,"recycled":recycle,"restoration":true,"begin":begin,"state_and_logits_exact":exact}));
        execute(w,Command::Finish(Finish::Length))?;
    }
    w.stream.synchronize()?;w.cached_sessions.clear();
    std::fs::write(out.join("encoder-recycle-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "scope":"Same-load ABBA exact-size encoder storage reuse after actual eight-slot eviction. Three measured forward turns per arm plus actual restoration after advancement. Following six FP32 logit rows, target/draft/encoder/history state and input identity checked.16K public prefix and recommended T1/P1; no completed-outcome, long-context throughput or production claim."}))?)?;
    if std::env::var("CANADA_SCRATCH_IDLE_COPY_SCREEN").as_deref()==Ok("1") {
        ensure!(std::env::var("CANADA_SCRATCH_IDLE_ANCHOR_SCREEN").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_STATE_COPY_SCREEN").as_deref()==Ok("1"),
            "combined encoder screen needs both finite copy/anchor permissions");
        super::idle_anchor_probe::run(w,rank,out)
    }else{execute(w,Command::Shutdown)?;Ok(())}
}
