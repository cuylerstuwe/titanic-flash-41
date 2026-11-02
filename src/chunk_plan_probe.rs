//! Finite target-only chunk-plan diagnostic. No HTTP or sampling policy changes.
//! Store raw logits; distribution calculations are performed separately on GPU.
use super::*;

fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn state(t: &Target) -> Result<Value> {
    Ok(json!({"position":t.position(),"carry":t.carry_digests()?,
        "encoder":t.encoder_tail_digest()?}))
}
fn following(w: &mut Worker, ids: &[u32]) -> Result<Vec<u8>> {
    let before=state(&w.verify)?;
    w.verify.begin_verification(ids,&w.comm,&w.stream)?;
    let (logits,rows)=w.verify.pending_logits()?;
    ensure!(rows==6,"chunk-plan teacher forcing needs six rows");
    let bytes=unsafe {w.cuda.read_device(logits.ptr(),rows*129280*4)}?;
    w.verify.finish_verification(0,&w.comm,&w.stream)?;
    ensure!(state(&w.verify)?==before,"chunk-plan diagnostic rollback changed state");
    Ok(bytes)
}

pub(super) fn run(w: &mut Worker,rank: usize,out: &Path) -> Result<()> {
    ensure!(w.current.is_none() && w.history.frontier()==0 && w.verify.physical_rows()==6,
        "chunk-plan screen requires fresh isolated target");
    let raw=include_bytes!("../tests/fixtures/chunk-plan-quality-v1.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let cases=fixture["cases"].as_array().context("chunk-plan cases")?;
    let mut records=Vec::new();
    for (case,c) in cases.iter().enumerate() {
        let ids:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
        let prefix=c["prefix"].as_u64().context("prefix length")?as usize;
        let split=c["split"].as_u64().context("split position")?as usize;
        ensure!(prefix+6==ids.len() && split>=128 && split+128<=prefix
            && ids.iter().all(|&v|v<129280),"invalid chunk-plan fixture");
        // Reverse the plan order in the second pair to expose history/order effects.
        for (trial,plan) in [0usize,1,1,0].into_iter().enumerate() {
            w.control.agree(&[case as u8,trial as u8,plan as u8])?;
            w.target_lineage=w.target_lineage.checked_add(1).context("probe lineage exhausted")?;
            if w.verify.profile_is_current() {
                w.verify.reset(w.target_lineage,&w.comm,&w.stream)?;
                w.large.handoff_from(&w.verify)?;
            } else {
                ensure!(w.large.profile_is_current(),"chunk-plan target owner absent");
                w.large.reset(w.target_lineage,&w.comm,&w.stream)?;
            }
            w.stream.synchronize()?;
            let began=Instant::now();
            if plan==0 {
                w.large.prefill_wide(&mut w.wides,&ids[..prefix],&w.comm,&w.stream)?;
            } else {
                w.large.prefill_wide(&mut w.wides,&ids[..split],&w.comm,&w.stream)?;
                w.large.prefill_wide(&mut w.wides,&ids[split..prefix],&w.comm,&w.stream)?;
            }
            w.stream.synchronize()?;
            let seconds=began.elapsed().as_secs_f64();
            let prompt=w.large.snapshot_prompt_logits(&w.stream)?;
            let mut bytes=unsafe {w.cuda.read_device(prompt.ptr(),129280*4)}?;
            w.verify.handoff_from(&w.large)?;
            let original=state(&w.verify)?;
            let after=following(w,&ids[prefix..])?;
            bytes.extend_from_slice(&after);
            // Check the exact retained-state path separately from changing chunks.
            let restored=if trial==1 {
                let size=w.verify.compact_state_image_bytes()?;
                ensure!(size<512*1024*1024 && w.control.all_admit(
                    w.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+size)?,
                    "chunk-plan snapshot reserve");
                let saved=w.verify.save_state_image(size,&w.comm,&w.stream)?;
                w.target_lineage=w.target_lineage.checked_add(1).context("probe lineage exhausted")?;
                w.verify.reset(w.target_lineage,&w.comm,&w.stream)?;
                w.target_lineage=w.target_lineage.checked_add(1).context("probe lineage exhausted")?;
                w.verify.restore_state_image(&saved,w.target_lineage,&w.comm,&w.stream)?;
                ensure!(state(&w.verify)?==original && following(w,&ids[prefix..])?==after,
                    "chunk-plan saved-state reconstruction changed bytes");
                true
            } else {false};
            let filename=format!("chunk-c{case}-t{trial}-p{plan}.f32");
            if rank==0 {std::fs::write(out.join(&filename),&bytes)?;}
            records.push(json!({"case":case,"id":c["id"],"trial":trial,"plan":plan,
                "prefill_tokens":prefix,"split":split,"prefill_seconds":seconds,"rows":7,
                "file":filename,"sha256":digest(&bytes),"state":original,
                "rollback_exact":true,"restore_checked":restored,"restore_exact":restored}));
            std::fs::write(out.join("chunk-plan-progress.json"),serde_json::to_vec_pretty(&records)?)?;
            eprintln!("chunk-plan rank{rank} case{case} trial{trial} plan{plan} captured");
        }
    }
    std::fs::write(out.join("chunk-plan-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"fixture_sha256":digest(raw),"records":records,
        "scope":"Small target-only fresh/split ABBA screen. Row0 is prompt logits; rows1..6 follow fixed teacher-forced tokens. Per-transaction rollback and snapshot restore require exactness; cross-plan distributions are GPU diagnostics, not an exactness or outcome-quality gate."}))?)?;
    Ok(())
}
