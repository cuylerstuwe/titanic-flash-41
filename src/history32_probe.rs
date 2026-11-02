//! Finite M32 target-only feasibility screen; no proposal or serving policy.
use super::*;

fn execute(w:&mut Worker,c:Command)->Result<Value> {
    let bytes=c.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(bytes.as_slice()))?;
    ensure!(wire==bytes,"history32 command mismatch");w.execute(c)
}
fn readout(w:&mut Worker,teacher:&[u32])->Result<Vec<u8>> {
    w.verify.begin_verification(teacher,&w.comm,&w.stream)?;
    let(y,n)=w.verify.pending_logits()?;ensure!(n==6,"history32 following rows");
    let bytes=unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?;
    w.verify.verification_sample(102400,&w.stream)?;
    w.verify.finish_verification(0,&w.comm,&w.stream)?;
    Ok(bytes)
}
fn restore(w:&mut Worker,image:&crate::target::StateImage,teacher:&[u32],expected:&[u8])->Result<()> {
    w.target_lineage=w.target_lineage.checked_add(1).context("history32 lineage exhausted")?;
    w.verify.restore_state_image(image,w.target_lineage,&w.comm,&w.stream)?;
    ensure!(readout(w,teacher)?==expected,"history32 target restoration changed following logits");
    Ok(())
}
fn digest(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}

pub(super) fn run(w:&mut Worker,rank:usize,kernels:&Path,out:&Path)->Result<()> {
    ensure!(crate::config::history32_probe_enabled()&&crate::config::verification_carry_rows()==64
        &&w.current.is_none()&&w.long_verify.is_some(),"finite M32 admission");
    let before=w.cuda.admission_memory()?;
    let module=w.cuda.module(&w.core)?;
    let mut wider=w.large.fork_profile(&module,&w.core,&w.stream,kernels,32,Phase::Verify)?;
    wider.set_graph_replay(true)?;
    ensure!(w.large.shared_state_report(&wider)["all_weights_and_causal_storage_shared"]==true,
        "M32 duplicated canonical model state");
    let after=w.cuda.admission_memory()?;
    ensure!(w.control.all_admit(after.admitted_available>=9*1024usize.pow(3))?,"M32 snapshot reserve");
    eprintln!("rank{rank} M32 admitted: before={before:?}, after={after:?}");
    let fixtures:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/history-copy-width32-v1.json"))?;
    let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    let selected:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/history-strong-backup-frontiers-v1.json"))?;
    let tools:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-history-score-v1.json"))?;
    let teacher=&tokens[262500..262506];let mut records=Vec::new();
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    // Opposite parity at the64-slot carry wrap; public repository and tool text.
    for case in [1usize,2] {
        let c=&fixtures["cases"][case];let ordinal=c["frontier_index"].as_u64().unwrap()as usize;
        let prompt:Vec<u32>=if c["kind"]=="repository" {
            let f=&selected["cases"][ordinal];let mut p=tokens[..f["corpus_tokens"].as_u64().unwrap()as usize].to_vec();
            p.extend(serde_json::from_value::<Vec<u32>>(f["suffix_ids"].clone())?);p
        }else{ensure!(c["kind"]=="tool","M32 fixture kind");serde_json::from_value(tools["cases"][ordinal]["prompt_ids"].clone())?};
        let n=prompt.len();let end=c["source_end"].as_u64().unwrap()as usize;
        let anchor=c["anchor"].as_u64().unwrap()as u32;
        ensure!(n==c["prefix_tokens"].as_u64().unwrap()as usize&&end>=63&&end+32<=n
            &&prompt[end]==anchor&&prompt[end-63..end]==prompt[n-63..],"M32 committed-history source");
        let mut candidate=vec![anchor];candidate.extend_from_slice(&prompt[end+1..end+32]);
        execute(w,Command::Begin{images:Vec::new(),id:7_840_000+case as u64,seed:2026110200+case as u64,
            greedy:false,limit:1,tokens:prompt})?;
        let bytes=w.verify.compact_state_image_bytes()?;
        ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
            >=8*1024usize.pow(3)+bytes)?,"M32 bounded snapshot");
        let image=w.verify.save_state_image(bytes,&w.comm,&w.stream)?;
        let original=readout(w,teacher)?;
        let state=(w.verify.carry_digests()?,w.verify.encoder_tail_digest()?);
        let mut trials=Vec::new();
        for trial in 0..6 {
            let rows=[16,32,16,32,32,16][trial];
            restore(w,&image,teacher,&original)?;
            let target=if rows==16 {w.long_verify.as_mut().unwrap()}else{&mut wider};
            target.handoff_from(&w.verify)?;
            let mut logits=Vec::new();let mut forward=0.;let mut commit=0.;
            for (chunk,ids) in candidate.chunks(rows).enumerate() {
                let began=Instant::now();target.begin_verification(ids,&w.comm,&w.stream)?;
                forward+=began.elapsed().as_secs_f64();
                let(y,live)=target.pending_logits()?;ensure!(live==ids.len(),"M32 live logits");
                logits.extend(unsafe{w.cuda.read_device(y.ptr(),live*129280*4)}?);
                if case==1 && trial<2 && chunk==0
                    && std::env::var("CANADA_SCRATCH_HISTORY32_TRACE").as_deref()==Ok("1") {
                    target.dump_history32_trace(&out.join(format!("case{case}-m{rows}-layers")))?;
                }
                target.verification_sample(102401,&w.stream)?; // finite normalized target distribution
                let began=Instant::now();target.finish_verification(ids.len(),&w.comm,&w.stream)?;
                commit+=began.elapsed().as_secs_f64();
            }
            w.verify.handoff_from(target)?;
            let carry=w.verify.carry_digests()?;let encoder=w.verify.encoder_tail_digest()?;
            let following=readout(w,teacher)?;
            if rank==3&&trial<2 {
                std::fs::write(out.join(format!("case{case}-m{rows}-logits.bin")),&logits)?;
                std::fs::write(out.join(format!("case{case}-m{rows}-following.bin")),&following)?;
            }
            trials.push(json!({"trial":trial,"rows":rows,"warmup":trial<2,"forward_seconds":forward,
                "commit_seconds":commit,"logits_sha256":digest(&logits),"following_sha256":digest(&following),
                "carry":carry,"encoder":encoder,"position":w.verify.position()}));
        }
        // Same M32 graph, accepted prefix unchanged, every rejected token changed.
        // This directly checks causal isolation and rollback across the ring wrap.
        let extra_partial=match std::env::var("CANADA_SCRATCH_HISTORY32_PARTIAL16").as_deref() {
            Ok("1")=>true,Ok("0")|Err(_)=>false,
            _=>anyhow::bail!("history32 partial16 must be 0 or 1"),
        };
        let mut suffix_checks=Vec::new();
        let accepted_counts:&[usize]=if extra_partial {&[16,17]} else {&[17]};
        for &accepted in accepted_counts {
          let mut partial=Vec::new();
          for changed in [false,true] {
            restore(w,&image,teacher,&original)?;wider.handoff_from(&w.verify)?;
            let mut inputs=candidate.clone();
            if changed {for id in &mut inputs[accepted..] {*id=2+(*id+123)%120000;}}
            wider.begin_verification(&inputs,&w.comm,&w.stream)?;
            let(y,live)=wider.pending_logits()?;ensure!(live==32,"M32 partial live rows");
            let prefix=unsafe{w.cuda.read_device(y.ptr(),accepted*129280*4)}?;
            wider.verification_sample(102402,&w.stream)?;
            wider.finish_verification(accepted,&w.comm,&w.stream)?;w.verify.handoff_from(&wider)?;
            let following=readout(w,teacher)?;
            partial.push(json!({"prefix_sha256":digest(&prefix),"following_sha256":digest(&following),
                "carry":w.verify.carry_digests()?,"encoder":w.verify.encoder_tail_digest()?,"position":w.verify.position()}));
          }
          ensure!(partial[0]==partial[1],"M32 rejected suffix changed accepted{accepted} prefix/state");
          suffix_checks.push(json!({"accepted":accepted,"variants":partial}));
        }
        restore(w,&image,teacher,&original)?;
        ensure!(w.verify.position()==n as u64&&state==(w.verify.carry_digests()?,w.verify.encoder_tail_digest()?)
            &&w.history.frontier()==n&&w.session.as_ref().unwrap().history_lookup_metadata().unwrap().0==n as u64,
            "M32 diagnostic changed accepted worker history");
        records.push(json!({"case":case,"name":c["name"],"prompt_tokens":n,"candidate_ids":candidate,
            "trials":trials,"suffix_checks":suffix_checks,
            "rejected_suffix_isolated":true,"restoration_exact":true}));
        execute(w,Command::Finish(Finish::Length))?;
        eprintln!("rank{rank} M32 case{case}: restored; rejected-suffix isolation passed");
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("history32-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,
        "match_attention_grouping":std::env::var("CANADA_SCRATCH_HISTORY32_MATCH_GROUPING").as_deref()==Ok("1"),
        "split_native_router":std::env::var("CANADA_SCRATCH_HISTORY32_SPLIT_ROUTER").as_deref()==Ok("1"),
        "before_m32":before,"after_m32":after,"records":records,"all_restorations_exact":true,
        "scope":"Finite target-only M32 prototype against twoM16calls, same canonical weights/allocations. Two public committed-history candidates, full32 and partial17, optionally partial16 to isolate the first16 causal rows. ABBA warm timing excludes readback and sampler diagnostics. Cross-width logits may differ numerically; raw FP32 comparisons required. No draft-selection, serving-policy or outcome-speed claim."}))?)?;
    Ok(())
}
