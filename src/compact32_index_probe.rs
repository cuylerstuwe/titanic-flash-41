//! Same-load M32 indexing comparison. Canonical numerical kernels are unchanged.
use super::*;
use super::history32_generation_probe::execute;

fn digest(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}
fn following(w:&mut Worker,teacher:&[u32])->Result<String> {
    w.verify.begin_verification(teacher,&w.comm,&w.stream)?;
    let(y,n)=w.verify.pending_logits()?;ensure!(n==6,"compact32 following rows");
    let hash=digest(&unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?);
    w.verify.finish_verification(0,&w.comm,&w.stream)?;Ok(hash)
}
fn restore(w:&mut Worker,image:&crate::target::StateImage,teacher:&[u32],expected:&str)->Result<()> {
    w.target_lineage=w.target_lineage.checked_add(1).context("compact32 lineage exhausted")?;
    w.verify.restore_state_image(image,w.target_lineage,&w.comm,&w.stream)?;
    let same=following(w,teacher)?==expected;
    ensure!(w.control.all_admit(same)?,"compact32 restored following logits changed");Ok(())
}
fn state(w:&mut Worker,teacher:&[u32])->Result<Value> {
    let carry=w.verify.carry_digests()?;let encoder=w.verify.encoder_tail_digest()?;
    Ok(json!({"position":w.verify.position(),"carry":carry,"encoder":encoder,
        "following_logits":following(w,teacher)?}))
}
pub(super) fn run(w:&mut Worker,rank:usize,kernels:&Path,out:&Path)->Result<()> {
    ensure!(crate::config::history32_probe_enabled()
        &&std::env::var("CANADA_SCRATCH_COMPACT32_INDEX_SCREEN").as_deref()==Ok("1")
        &&w.current.is_none(),"finite compact32 admission");
    let before=w.cuda.admission_memory()?;let module=w.cuda.module(&w.core)?;
    let mut wider=w.large.fork_profile(&module,&w.core,&w.stream,kernels,32,Phase::Verify)?;
    wider.set_graph_replay(true)?;
    ensure!(w.large.shared_state_report(&wider)["all_weights_and_causal_storage_shared"]==true,
        "compact32 duplicated canonical state");
    let after=w.cuda.admission_memory()?;
    ensure!(w.control.all_admit(after.admitted_available>=9*1024usize.pow(3))?,"compact32 snapshot reserve");
    let fixtures:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/history-copy-width32-v1.json"))?;
    let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    let selected:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/history-strong-backup-frontiers-v1.json"))?;
    let tools:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-history-score-v1.json"))?;
    let teacher=&tokens[262500..262506];let mut records=Vec::new();
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    for case in [1usize,2] {
        let c=&fixtures["cases"][case];let ordinal=c["frontier_index"].as_u64().unwrap()as usize;
        let prompt:Vec<u32>=if c["kind"]=="repository" {
            let f=&selected["cases"][ordinal];let mut p=tokens[..f["corpus_tokens"].as_u64().unwrap()as usize].to_vec();
            p.extend(serde_json::from_value::<Vec<u32>>(f["suffix_ids"].clone())?);p
        }else{ensure!(c["kind"]=="tool","compact32 fixture kind");serde_json::from_value(tools["cases"][ordinal]["prompt_ids"].clone())?};
        let n=prompt.len();let end=c["source_end"].as_u64().unwrap()as usize;
        let anchor=c["anchor"].as_u64().unwrap()as u32;
        ensure!(n==c["prefix_tokens"].as_u64().unwrap()as usize&&end>=63&&end+32<=n
            &&prompt[end]==anchor&&prompt[end-63..end]==prompt[n-63..],"compact32 committed source");
        let mut inputs=vec![anchor];inputs.extend_from_slice(&prompt[end+1..end+32]);
        execute(w,Command::Begin{images:Vec::new(),id:7_880_000+case as u64,seed:2026110900+case as u64,
            greedy:false,limit:1,tokens:prompt})?;
        let bytes=w.verify.compact_state_image_bytes()?;
        ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
            >=8*1024usize.pow(3)+bytes)?,"compact32 bounded snapshot");
        let image=w.verify.save_state_image(bytes,&w.comm,&w.stream)?;
        let original=following(w,teacher)?;let accepted_state=state(w,teacher)?;
        let mut expected=None;let mut trials=Vec::new();
        // Warm each graph bank once, then ABBA over the same physical weights.
        for (trial,enabled) in [false,true,true,false,false,true].into_iter().enumerate() {
            restore(w,&image,teacher,&original)?;wider.handoff_from(&w.verify)?;
            wider.set_compact32_for_probe(enabled,&w.stream)?;
            let captures=wider.capture_statistics().1;
            let began=Instant::now();wider.begin_verification(&inputs,&w.comm,&w.stream)?;
            let forward=began.elapsed().as_secs_f64();
            let(y,live)=wider.pending_logits()?;ensure!(live==32,"compact32 logits extent");
            let logits=digest(&unsafe{w.cuda.read_device(y.ptr(),live*129280*4)}?);
            wider.verification_sample(104400,&w.stream)?;
            let began=Instant::now();wider.finish_verification(32,&w.comm,&w.stream)?;
            let commit=began.elapsed().as_secs_f64();w.verify.handoff_from(&wider)?;
            let outcome=json!({"logits":logits,"state":state(w,teacher)?});
            let same=expected.as_ref().is_none_or(|e|e==&outcome);
            if expected.is_none(){expected=Some(outcome.clone());}
            trials.push(json!({"trial":trial,"compact":enabled,"warmup":trial<2,
                "forward_seconds":forward,"commit_seconds":commit,"outcome":outcome,
                "new_captures":wider.capture_statistics().1-captures}));
            if !same {std::fs::write(out.join(format!("compact32-mismatch-case{case}.json")),serde_json::to_vec_pretty(&trials)?)?;}
            ensure!(w.control.all_admit(same)?,"compact32 changed full-model logits or accepted state");
        }
        // A padded19-row replay and changed rejected suffix exercise the new
        // union extent without expanding this into another serving panel.
        let mut checks=Vec::new();let mut partial_reference=None;
        for (kind,live,accepted,changed) in [("partial",32,19,false),("padded",19,7,false),("suffix",32,19,true)] {
            let mut pair=Vec::new();
            for enabled in [false,true] {
                restore(w,&image,teacher,&original)?;wider.handoff_from(&w.verify)?;
                wider.set_compact32_for_probe(enabled,&w.stream)?;
                let mut ids=inputs[..live].to_vec();
                if changed {for id in &mut ids[accepted..] {*id=2+(*id+123)%120000;}}
                wider.begin_verification(&ids,&w.comm,&w.stream)?;
                let(y,rows)=wider.pending_logits()?;ensure!(rows==live,"compact32 padded rows");
                let prefix=digest(&unsafe{w.cuda.read_device(y.ptr(),accepted*129280*4)}?);
                wider.finish_verification(accepted,&w.comm,&w.stream)?;w.verify.handoff_from(&wider)?;
                pair.push(json!({"prefix_logits":prefix,"state":state(w,teacher)?}));
            }
            let same=pair[0]==pair[1] && (!changed||partial_reference.as_ref()==Some(&pair[0]));
            if kind=="partial" {partial_reference=Some(pair[0].clone());}
            ensure!(w.control.all_admit(same)?,"compact32 partial, padded or future-suffix mismatch");
            checks.push(json!({"kind":kind,"live":live,"accepted":accepted,"exact":true,"outcome":pair[0]}));
        }
        restore(w,&image,teacher,&original)?;
        let same=state(w,teacher)?==accepted_state&&w.history.frontier()==n
            &&w.session.as_ref().unwrap().history_lookup_metadata().unwrap().0==n as u64;
        ensure!(w.control.all_admit(same)?,"compact32 changed accepted request history");
        records.push(json!({"case":case,"name":c["name"],"prompt_tokens":n,"trials":trials,
            "checks":checks,"restoration_exact":true}));
        execute(w,Command::Finish(Finish::Length))?;
        eprintln!("rank{rank} compact32 case{case}: exact logits, accepted state and restoration");
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("compact32-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"before_m32":before,"after_m32":after,"records":records,"all_exact":true,
        "scope":"Two fixed public frontiers,131K repository and11K tool material. Same M32 numerical path/physical weights, dense versus compact late consumers plus packed early gather. One warm pair and ABBA timings, padded19, partial acceptance, changed rejected suffix, following M6 and exact restore. No generation-TPS or task-outcome claim."
    }))?)?;Ok(())
}
