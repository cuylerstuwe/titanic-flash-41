//! Finite same-allocation check of short cached-extension chunk choices.
//! Alternative plans use the ordinary target operators and preserve precision.
use super::*;
use super::history32_generation_probe::execute;
fn state(w:&Worker)->Result<Value> {
    Ok(json!({"position":w.verify.position(),"carry":w.verify.carry_digests()?,
        "encoder":w.verify.encoder_tail_digest()?}))
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()
        &&std::env::var("CANADA_SCRATCH_CHUNK_ECONOMICS_SCREEN").as_deref()==Ok("1"),
        "isolated chunk economics screen");
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let ids:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let prefix=65541usize;
    execute(w,Command::Begin{images:Vec::new(),id:8_020_200,seed:20261102,
        greedy:false,limit:1,tokens:ids[..prefix].to_vec()})?;
    let bytes=w.verify.compact_state_image_bytes()?;
    ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(
        w.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+bytes)?,
        "chunk economics snapshot reserve");
    let saved=w.verify.save_state_image(bytes,&w.comm,&w.stream)?;
    let original=state(w)?;
    let widths:Vec<usize>=std::iter::once(128).chain(w.wides.iter().map(Target::physical_rows)).collect();
    let cases=[(129,vec![128,128]),(193,vec![128,128]),(257,vec![128,128,128]),
        (513,vec![512,128]),(769,vec![512,512]),(1025,vec![2048]),(2049,vec![4096])];
    let mut records=Vec::new();
    for (case,(delta,alternative)) in cases.into_iter().enumerate() {
        let control=crate::config::plan_encoder_chunks(delta,&widths);
        ensure!(control!=alternative,"chunk economics duplicate plans");
        let offset=200000+case*4096;
        let plans=[control,alternative];
        for (trial,mode) in [0usize,1,1,0,0,1,1,0].into_iter().enumerate() {
            w.target_lineage=w.target_lineage.checked_add(1).context("chunk economics lineage exhausted")?;
            w.verify.restore_state_image(&saved,w.target_lineage,&w.comm,&w.stream)?;
            ensure!(state(w)?==original,"chunk economics incoming snapshot differs");
            w.large.handoff_from(&w.verify)?;
            w.control.agree(&[0xaf,case as u8,trial as u8,mode as u8])?;
            let began=Instant::now();
            let progress=w.large.prefill_wide_interruptible(&mut w.wides,
                &ids[offset..offset+delta],&w.comm,&w.stream,Some(plans[mode].clone()),&mut |_|Ok(false))?;
            w.stream.synchronize()?;
            let seconds=began.elapsed().as_secs_f64();
            ensure!(progress.done==delta&&progress.remaining_chunks.is_empty(),"chunk economics extent");
            let phases=serde_json::to_value(w.large.append_metrics()?)?;
            ensure!(trial<4||phases["captured_new_graph"]==false,"chunk economics measured capture");
            let mut logits=if trial>=6 {
                let prompt=w.large.snapshot_prompt_logits(&w.stream)?;
                unsafe{w.cuda.read_device(prompt.ptr(),129280*4)}?
            }else{Vec::new()};
            w.verify.handoff_from(&w.large)?;
            let after=state(w)?;
            if trial>=6 {
                w.verify.begin_verification(&ids[offset+delta..offset+delta+6],&w.comm,&w.stream)?;
                let (y,n)=w.verify.pending_logits()?;ensure!(n==6,"chunk economics following extent");
                logits.extend(unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?);
                w.verify.finish_verification(0,&w.comm,&w.stream)?;
                ensure!(state(w)?==after,"chunk economics following rollback differs");
                if rank==0 {std::fs::write(out.join(format!("chunk-economics-c{case}-m{mode}.bin")),&logits)?;}
            }
            records.push(json!({"case":case,"prefix":prefix,"delta":delta,"mode":mode,"plan":plans[mode],
                "trial":trial,"warmup":trial<4,"seconds":seconds,"phases":phases,"after":after,
                "incoming_restore_exact":true,"following_rollback_exact":trial>=6,
                "logit_sha256":if logits.is_empty(){None}else{Some(format!("{:x}",Sha256::digest(&logits)))}}));
            std::fs::write(out.join("chunk-economics-partial.json"),serde_json::to_vec_pretty(&records)?)?;
        }
        eprintln!("chunk economics rank{rank} case{case} delta{delta} complete");
    }
    w.target_lineage=w.target_lineage.checked_add(1).context("chunk economics lineage exhausted")?;
    w.verify.restore_state_image(&saved,w.target_lineage,&w.comm,&w.stream)?;
    ensure!(state(w)?==original,"chunk economics final restore differs");
    execute(w,Command::Finish(Finish::Length))?;execute(w,Command::Shutdown)?;
    std::fs::write(out.join("chunk-economics-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "scope":"Seven public repository extensions at one65K retained frontier. Ordinary versus alternative chunk sequences, two warm and two measured executions per plan in reversed order. Snapshot/rollback exact. Canonical target precision unchanged; different width/collective orders may change rounding. Saved prompt and six following distributions require GPU analysis before any plan-policy change. This is an opportunity screen, not an outcome or serving-default claim."
    }))?)?;Ok(())
}
