//! Revisit the isolated M16 FFN schedule after the dense pipeline fence repair.
use super::*;
use super::history32_generation_probe::execute;
fn state(t:&Target)->Result<Value>{
    Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))
}
fn logits(cuda:&Cuda,t:&Target)->Result<Vec<u8>>{
    let (y,n)=t.pending_local_logits_for_probe()?;
    unsafe{cuda.read_device(y.ptr(),n*32320*4)}
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.long_verify.is_some()
        &&std::env::var("CANADA_SCRATCH_FFN16_REFRESH_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_VERIFY16_OVERLAP_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_VERIFY16_ATTN_ONLY").as_deref()!=Ok("1")
        &&std::env::var("CANADA_SCRATCH_DENSE_PIPELINE_FENCE").as_deref()==Ok("1"),
        "isolated fenced FFN schedule screen");
    let short:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let ids:Vec<u32>=serde_json::from_value(short["tokens"].clone())?;
    let long:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-history-context-v1.json"))?;
    let case=&long["cases"][0];
    let long_prompt:Vec<u32>=serde_json::from_value(case["prompt_ids"].clone())?;
    ensure!(long_prompt.len()==274108,"old FFN failure frontier");
    let mut teacher=vec![case["anchor"].as_u64().context("old FFN anchor")? as u32];
    teacher.extend(serde_json::from_value::<Vec<u32>>(case["proposal_ids"].clone())?);
    ensure!(teacher.len()==16,"FFN proposal extent");
    let mut records=Vec::new();
    for (ordinal,(prompt,input,following)) in [
        (ids[..65541].to_vec(),ids[65541..65557].to_vec(),ids[65557..65563].to_vec()),
        (long_prompt,teacher,ids[7911..7917].to_vec())].into_iter().enumerate() {
        let prefix=prompt.len();
        execute(w,Command::Begin{images:Vec::new(),id:8_020_400+ordinal as u64,
            seed:20261102,greedy:false,limit:1,tokens:prompt})?;
        w.long_verify.as_mut().unwrap().handoff_from(&w.verify)?;
        let t=w.long_verify.as_mut().unwrap();
        let accepted=state(t)?;let bytes=t.compact_state_image_bytes()?;
        ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(
            w.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+bytes)?,
            "FFN schedule snapshot reserve");
        let saved=t.save_state_image(bytes,&w.comm,&w.stream)?;
        let mut changed=input.clone();changed.rotate_left(8);changed[0]=(changed[0]+37)%129280;
        let mut expected=std::collections::BTreeMap::<(usize,usize),Vec<u8>>::new();
        let mut continuation:Option<(Value,Vec<u8>)>=None;
        for (trial,mask) in [0u8,1,3,2,2,3,1,0].into_iter().enumerate(){
            t.set_verify16_schedule_for_probe(mask,&w.stream)?;
            for live in [9usize,16] {
                for _ in 0..3 {t.begin_verification(&input[..live],&w.comm,&w.stream)?;
                    t.finish_verification(0,&w.comm,&w.stream)?;}
                let captures=t.capture_statistics().1;let mut samples=Vec::new();
                for repeat in 0..4u8 {
                    w.control.agree(&[0xb1,ordinal as u8,trial as u8,live as u8,repeat])?;
                    let at=Instant::now();t.begin_verification(&input[..live],&w.comm,&w.stream)?;
                    w.stream.synchronize()?;samples.push(at.elapsed().as_secs_f64()*1000.);
                    t.finish_verification(0,&w.comm,&w.stream)?;
                }
                ensure!(t.capture_statistics().1==captures,"FFN timed graph capture");
                for (kind,proposal) in [&input,&changed].into_iter().enumerate(){
                    t.begin_verification(&proposal[..live],&w.comm,&w.stream)?;
                    let y=logits(&w.cuda,t)?;
                    let exact=expected.entry((live,kind)).or_insert_with(||y.clone())==&y;
                    t.finish_verification(0,&w.comm,&w.stream)?;
                    ensure!(w.control.all_admit(exact&&state(t)?==accepted)?,
                        "FFN schedule changed logits or rollback: prefix{prefix} trial{trial} mask{mask} live{live} kind{kind}");
                }
                records.push(json!({"prefix":prefix,"trial":trial,"mask":mask,"live":live,
                    "samples_ms":samples,"changed_input_logits_exact":true,"rollback_exact":true}));
                std::fs::write(out.join("ffn16-refresh-partial.json"),serde_json::to_vec_pretty(&records)?)?;
            }
            // Partial acceptance crosses the old failing page boundary. The
            // following proposal is unrelated, so a stale auxiliary result
            // cannot pass merely by replaying the same copy candidate.
            t.begin_verification(&input,&w.comm,&w.stream)?;
            t.finish_verification(7,&w.comm,&w.stream)?;
            let after=state(t)?;
            t.begin_verification(&following,&w.comm,&w.stream)?;
            let y=logits(&w.cuda,t)?;t.finish_verification(0,&w.comm,&w.stream)?;
            let proof=(after.clone(),y);
            let exact=continuation.as_ref().is_none_or(|first|first==&proof);
            if continuation.is_none(){continuation=Some(proof);}
            ensure!(w.control.all_admit(exact&&state(t)?==after)?,"FFN commit/following state differs");
            w.target_lineage=w.target_lineage.checked_add(1).context("FFN lineage exhausted")?;
            t.restore_state_image(&saved,w.target_lineage,&w.comm,&w.stream)?;
            ensure!(state(t)?==accepted,"FFN snapshot restore differs");
            eprintln!("FFN schedule rank{rank} prefix{prefix} trial{trial} mask{mask} complete");
        }
        t.set_verify16_schedule_for_probe(0,&w.stream)?;
        w.verify.handoff_from(t)?;
        execute(w,Command::Finish(Finish::Length))?;
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("ffn16-refresh-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"commit_and_following_exact":true,
        "scope":"65K public continuation and old274108-token failing frontier; serial/shared-expert/coefficient/both FFN schedules, reversed order, current fenced dense kernels and target precision. Same-load timing, changed proposals, partial7 commits, following logits and restored state. No serving-policy change."}))?)?;
    Ok(())
}
