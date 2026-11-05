//! Finite same-load cache-entry screen. This is never reachable through HTTP.
//! The normal byte budget, all-rank admission and model arithmetic are unchanged.
use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Suite {tokenizer:PathBuf,bodies:Vec<Value>}

pub(super) fn execute(w:&mut Worker,c:Command)->Result<Value> {
    let encoded=c.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(encoded.as_slice()))?;
    ensure!(wire==encoded,"cache slot commands disagree");
    w.execute(Command::decode(&wire)?)
}
fn state(w:&mut Worker)->Result<Value> {
    ensure!(w.verify.position()as usize==w.history.frontier(),"cache slot frontier mismatch");
    Ok(json!({"frontier":w.verify.position(),"carry":w.verify.carry_digests()?,
        "encoder":w.verify.encoder_tail_digest()?,
        "draft":w.session.as_mut().unwrap().conditioning_state_digest_for_probe(&w.comm,&w.stream)?,
        "history":w.session.as_ref().unwrap().history_lookup_digest_for_probe()?}))
}
pub(super) fn following(w:&mut Worker,ids:&[u32])->Result<(Value,Vec<u8>)> {
    let before=state(w)?;
    w.verify.begin_verification(ids,&w.comm,&w.stream)?;
    let(y,n)=w.verify.pending_logits()?;ensure!(n==6,"cache slot following row count");
    let raw=unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?;
    w.verify.finish_verification(0,&w.comm,&w.stream)?;
    ensure!(state(w)?==before,"cache slot verification abort changed committed state");
    Ok((json!({"state":before,"following_logits_sha256":format!("{:x}",Sha256::digest(&raw))}),raw))
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path,path:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.suspended_image_count==0&&w.media.is_some()
        &&w.session_cache_budget==2usize<<30&&w.shared_image_budget==2usize<<30
        &&w.prompt_cache_logits&&w.retain_restored_checkpoint,
        "cache slot screen needs the ordinary bounded image cache");
    let raw=std::fs::read(path)?;
    ensure!(raw.len()<=4<<20,"cache slot fixture size bound");
    w.control.agree(&Sha256::digest(&raw))?;
    let suite:Suite=serde_json::from_slice(&raw)?;
    ensure!(suite.bodies.len()==3,"cache slot screen needs three image revisions");
    // Only ingress prepares images, just as in serving. Other ranks receive
    // canonical tokens and descriptors through the ordinary command channel.
    let frontend=if rank==0 {Some(crate::protocol::Frontend::load_with_images(&suite.tokenizer,true)?)}else{None};
    let prepared=frontend.as_ref().map(|f|suite.bodies.iter().map(|b|
        f.prepare_chat(&serde_json::to_vec(b)?,w.large.context_capacity())).collect::<Result<Vec<_>>>()).transpose()?;
    if let Some(prepared)=&prepared {for p in prepared {
        ensure!(!p.images.is_empty()&&p.tokens.len()<8192&&p.max_new_tokens==24
            &&p.converted.inference_options.temperature==Some(1.)
            &&p.converted.inference_options.top_p==Some(1.),"cache slot input bounds/settings");
    }}
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/target-agentic-python-errors-v2.json"))?;
    let text:Vec<u32>=serde_json::from_value(fixture["batches"][0].clone())?;
    let following_ids=&text[100..106];
    let repository:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/target-agentic-repository-v1.json"))?;
    let repository:Vec<u32>=serde_json::from_value(repository.get("tokens").unwrap_or(&repository["batches"][0]).clone())?;
    ensure!(repository.len()>=8192,"forward append fixture extent");
    let mut id=8_001_000u64;
    let original_slots=w.session_cache_slots;
    let mut records=Vec::new();let mut forward=Vec::new();
    let mut references=std::collections::BTreeMap::<usize,(Value,Vec<u8>)>::new();
    for (arm,slots) in [4usize,8,8,4].into_iter().enumerate() {
        w.control.agree(&[arm as u8,slots as u8])?;
        ensure!(w.current.is_none()&&w.suspended_image_count==0,"cache slot arm still owns a request");
        w.stream.synchronize()?;w.cached_sessions.clear();w.session_cache_slots=slots;
        // A real unrelated request gives every arm the same prior owner.
        // No probe edits or fabricates a GPU frontier or history state.
        let budget=w.session_cache_budget;w.session_cache_budget=0;id+=1;
        execute(w,Command::Begin{images:Vec::new(),id,seed:20261103,greedy:false,limit:1,tokens:text[..127].to_vec()})?;
        execute(w,Command::Finish(Finish::Length))?;
        w.cached_sessions.clear();w.session_cache_budget=budget;
        for (index,revision) in [0usize,1,2,0,1,2,0].into_iter().enumerate() {
            id+=1;
            let encoded=if let Some(prepared)=&prepared {
                let p=&prepared[revision];w.media.as_mut().unwrap().set_inputs(&p.image_inputs)?;
                Some(Command::Begin{images:p.images.clone(),id,seed:20261103+index as u64,
                    greedy:false,limit:24,tokens:p.tokens.clone()}.encode()?)
            }else{None};
            let wire=w.control.command(encoded.as_deref())?;
            let began=Instant::now();let begin=w.execute(Command::decode(&wire)?)?;
            let begin_seconds=began.elapsed().as_secs_f64();
            ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.,"cache slot target settings");
            let semantic=following(w,following_ids)?;
            let exact=if let Some(reference)=references.get(&index) {
                let exact=reference.0==semantic.0;
                if !exact {
                    // Preserve actual FP32 rows for a numerical follow-up;
                    // a hash mismatch alone is not evidence of lost quality.
                    let tag=format!("cache-slots-a{arm}-q{index}");
                    std::fs::write(out.join(format!("{tag}-reference.f32")),&reference.1)?;
                    std::fs::write(out.join(format!("{tag}-candidate.f32")),&semantic.1)?;
                }
                exact
            }else{references.insert(index,(semantic.0.clone(),semantic.1));true};
            let mut emitted:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;
            let cancel=index==5;
            let decode_at=Instant::now();
            while !cancel&&w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
                let step=execute(w,Command::Step)?;
                emitted.extend(serde_json::from_value::<Vec<u32>>(step["output_ids"].clone())?);
            }
            let decode_seconds=decode_at.elapsed().as_secs_f64();
            let reason=if cancel{Finish::Cancelled}else if w.history.pending()==Some(w.cfg.eos_token_id){Finish::Stop}else{Finish::Length};
            let finish=execute(w,Command::Finish(reason))?;
            let retained=w.cached_sessions.iter().map(|c|json!({"tokens":c.prefix.len(),
                "bytes":c.charge,"prompt_logits":c.prompt_logits.is_some(),"images":c.media.placements().len()})).collect::<Vec<_>>();
            let charged=w.cached_sessions.iter().map(|c|c.charge).sum::<usize>();
            ensure!(w.cached_sessions.len()<=slots&&charged<=w.session_cache_budget,"cache slot budget exceeded");
            records.push(json!({"arm":arm,"slots":slots,"index":index,"revision":revision,
                "begin":begin,"begin_seconds":begin_seconds,"decode_seconds":decode_seconds,
                "output_ids":emitted,"finish":finish,"cancelled":cancel,"semantic":semantic.0,
                "semantic_exact_to_first_arm":exact,"retained":retained,"charged_bytes":charged,
                "memory":w.cuda.admission_memory()?}));
            std::fs::write(out.join("cache-slots-partial.json"),serde_json::to_vec_pretty(&records)?)?;
            eprintln!("cache slots rank{rank} arm{arm} slots{slots} request{index}: {:.2}ms, cached{}",begin_seconds*1000.,records.last().unwrap()["begin"]["cached_tokens"]);
        }
        // The user's forward stream is primary. Exercise actual accepted
        // continuations while the optional branch cache remains occupied.
        // The wide append crosses the encoder-checkpoint threshold; the next
        // append exercises its live-frontier checkpoint maintenance.
        for (index,delta) in [8usize,256,8192,128].into_iter().enumerate() {
            id+=1;let previous=w.history.frontier();
            let mut tokens=w.history.committed().to_vec();
            tokens.push(w.history.pending().context("forward probe pending anchor")?);
            tokens.extend_from_slice(&repository[..delta-1]);
            let charged_before=w.cached_sessions.iter().map(|c|c.charge).sum::<usize>();
            let entries_before=w.cached_sessions.len();
            let wire=if let Some(prepared)=&prepared {
                let p=&prepared[0];w.media.as_mut().unwrap().set_inputs(&p.image_inputs)?;
                Some(Command::Begin{images:p.images.clone(),id,seed:20261200+index as u64,
                    greedy:false,limit:1,tokens:tokens.clone()}.encode()?)
            }else{None};
            let wire=w.control.command(wire.as_deref())?;
            let began=Instant::now();let begin=w.execute(Command::decode(&wire)?)?;
            let begin_seconds=began.elapsed().as_secs_f64();
            ensure!(begin["cached_tokens"]==previous&&begin["retained_session_cache"]["active_reuse"]==true,
                "forward cost screen unexpectedly restored or recomputed a prefix");
            let semantic=following(w,following_ids)?;
            execute(w,Command::Finish(Finish::Length))?;
            let charged=w.cached_sessions.iter().map(|c|c.charge).sum::<usize>();
            ensure!(w.cached_sessions.len()<=slots&&charged<=w.session_cache_budget,"forward cache budget exceeded");
            forward.push(json!({"arm":arm,"slots":slots,"index":index,"delta":delta,
                "prefix_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&tokens)?)),
                "begin":begin,"begin_seconds":begin_seconds,"semantic":semantic.0,
                "charged_before":charged_before,"charged_after":charged,
                "entries_before":entries_before,"entries_after":w.cached_sessions.len(),
                "memory":w.cuda.admission_memory()?}));
            std::fs::write(out.join("cache-slots-forward-partial.json"),serde_json::to_vec_pretty(&forward)?)?;
        }
    }
    w.stream.synchronize()?;w.cached_sessions.clear();w.session_cache_slots=original_slots;
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("cache-slots-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"forward":forward,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "all_semantic_exact":records.iter().all(|r|r["semantic_exact_to_first_arm"]==true),
        "byte_budget":2usize<<30,"following_ids":following_ids,
        "scope":"Same-weight-address ABBA screen of four versus eight optional entries. Three short synthetic image revisions at canonical T1/P1/high; 24-token caps and one cancel/rewind per arm. Then four actual forward appends through the occupied cache, including encoder checkpoint maintenance; report input hashes and distinguish matching prefixes. Normal slot/byte/physical admission and exact media identity. Read-only following verification must roll back exactly. Numeric differences across fresh/restored paths are recorded, not treated as quality failures from a digest alone. No HTTP, full-context pressure, completed-task quality or production claim."}))?)?;
    Ok(())
}
