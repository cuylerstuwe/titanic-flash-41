//! Small public-prefix opportunity screen; no reused draft is ever emitted.
use super::*;

fn accepted_state(w:&Worker)->Result<Value>{
    Ok(json!({"position":w.verify.position(),"carry":w.verify.carry_digests()?,
        "encoder":w.verify.encoder_tail_digest()?,
        "draft":w.session.as_ref().unwrap().accepted_conditioning_state_for_probe(&w.stream)?}))
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.history.frontier()==0&&!w.progress_next
        &&std::env::var("CANADA_SCRATCH_QUERY_REUSE_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_HIERARCHICAL_DRAFT").as_deref()==Ok("1"),
        "finite current query reuse opportunity screen");
    ensure!(w.control.all_admit(w.cuda.admission_memory()?.admitted_available>=
        8*1024usize.pow(3)+512*1024usize.pow(2)+32*1024usize.pow(2))?,"query diagnostic workspace reserve");
    let mut workspace=crate::dspark_generation::QueryReuseWorkspace::new(&w.cuda,&w.core)?;
    let common:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/confidence-economics-v1.json"))?;
    let transfer:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/confidence-transfer-v1.json"))?;
    let selected=[&common["cases"][0],&common["cases"][2],&transfer["cases"][0]];
    let mut records=Vec::new();
    for(ordinal,case)in selected.into_iter().enumerate(){
        w.cached_sessions.clear();
        let tokens:Vec<u32>=serde_json::from_value(case["input_ids"].clone())?;
        let begin=history32_generation_probe::execute(w,Command::Begin{images:Vec::new(),
            id:8_100_000+ordinal as u64,seed:2026110403+ordinal as u64,greedy:false,limit:160,tokens})?;
        ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.,"query screen target law");
        let mut cases=Vec::new();let mut measured=0;
        // The first two ordinary rounds warm the current graph paths. At
        // most12 further rounds seek six usable old suffixes per task.
        for trial in 0..14 {
            if w.emitted>=w.limit||w.history.pending()==Some(w.cfg.eos_token_id)||measured==6{break;}
            let anchor=w.history.pending().context("query probe anchor")?;
            let snapshot=if trial>=2 {Some(w.session.as_mut().unwrap()
                .capture_query_for_reuse_probe(anchor,&w.comm,&w.stream)?)}else{None};
            let step=history32_generation_probe::execute(w,Command::Step)?;
            if let Some(old)=snapshot {
                let shift=w.verify.position().checked_sub(old.position).context("query probe frontier")?;
                let sample=if shift>0&&shift<5&&w.history.pending()!=Some(w.cfg.eos_token_id){
                    let before=accepted_state(w)?;
                    let current=w.history.pending().context("query probe next anchor")?;
                    let metric=w.session.as_mut().unwrap().measure_query_reuse_for_probe(
                        &mut w.verify,current,&old,&mut workspace,&w.comm,&w.stream)?;
                    let after=accepted_state(w)?;let exact=before==after;
                    std::fs::write(out.join(format!("query-reuse-state-{ordinal}-{trial}.json")),
                        serde_json::to_vec_pretty(&json!({"before":before,"after":after,"exact":exact}))?)?;
                    ensure!(w.control.all_admit(exact)?,"query observer changed accepted state");
                    measured+=1;Some(metric)
                }else{None};
                cases.push(json!({"trial":trial,"ordinary_step":step,"shift":shift,"sample":sample}));
                std::fs::write(out.join("query-reuse-partial.json"),serde_json::to_vec_pretty(&json!({
                    "completed":records,"current_case":case["id"],"current":cases}))?)?;
            }
        }
        let finish=history32_generation_probe::execute(w,Command::Finish(
            if w.history.pending()==Some(w.cfg.eos_token_id){Finish::Stop}else{Finish::Cancelled}))?;
        records.push(json!({"case":case["id"],"begin":begin,"records":cases,"measured_roots":measured,"finish":finish}));
    }
    history32_generation_probe::execute(w,Command::Shutdown)?;
    std::fs::write(out.join("query-reuse-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"accepted_state_preserved":true,"serving_policy_changed":false,
        "scope":"Three public task frontiers, at most12 observed steps after two ordinary warmups; up to six eligible unused suffixes per task. Fresh and shifted-old query root q are integrated against the same actual target p by existing GPU kernels. New anchor correction, current draft temperature and independent normal block RNG remain. All diagnostic draws discarded; target rollback and accepted draft state checked. Root tokenwise overlap is a pruning statistic, NOT block expected length, adaptive-policy throughput or completed-task quality. No reused proposal enters ordinary generation."}))?)?;Ok(())
}
