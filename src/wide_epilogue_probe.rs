//! Exact wider query-store/residual-update integration in existing model storage.
use super::*;
use super::history32_generation_probe::execute;
fn state(t:&Target)->Result<Value>{
    Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))
}
fn logits(cuda:&Cuda,t:&Target)->Result<Vec<u8>>{
    let (y,n)=t.pending_local_logits_for_probe()?;
    unsafe{cuda.read_device(y.ptr(),n*32320*4)}
}
pub(super) fn run(w:&mut Worker,rank:usize,kernels:&Path,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.copy32_verify.is_some()&&w.long_verify.is_some(),"wide epilogue profiles");
    let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let ids:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    let prefix=65541;let input=&ids[prefix..prefix+32];let following=&ids[prefix+32..prefix+38];
    execute(w,Command::Begin{images:Vec::new(),id:8_021_100,seed:2026110211,
        greedy:false,limit:1,tokens:ids[..prefix].to_vec()})?;
    let mut records=Vec::new();
    for rows in [16usize,32] {
        let t=if rows==16{w.long_verify.as_mut().unwrap()}else{w.copy32_verify.as_mut().unwrap()};
        t.handoff_from(&w.verify)?;
        let accepted=state(t)?;let bytes=t.compact_state_image_bytes()?;
        ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(
            w.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+bytes)?,"epilogue snapshot reserve");
        let saved=t.save_state_image(bytes,&w.comm,&w.stream)?;
        let mut changed=input.to_vec();changed.rotate_left(8);changed[0]=(changed[0]+37)%129280;
        let mut expected=std::collections::BTreeMap::<(usize,usize,bool),Vec<u8>>::new();
        let mut continuation:Option<(Value,Vec<u8>)>=None;
        for (trial,enabled) in [false,true,true,false].into_iter().enumerate() {
            t.set_wide_epilogues_for_probe(enabled,kernels,&w.stream)?;
            for live in [rows/2+1,rows] {
                for _ in 0..3 {t.begin_verification(&input[..live],&w.comm,&w.stream)?;
                    t.finish_verification(0,&w.comm,&w.stream)?;}
                let captures=t.capture_statistics().1;let mut samples=Vec::new();let mut metrics=Vec::new();
                for repeat in 0..4u8 {
                    w.control.agree(&[0xb2,rows as u8,trial as u8,live as u8,repeat])?;
                    let at=Instant::now();t.begin_verification(&input[..live],&w.comm,&w.stream)?;
                    w.stream.synchronize()?;samples.push(at.elapsed().as_secs_f64()*1000.);
                    metrics.push(serde_json::to_value(t.verification_metrics()?)?);
                    t.finish_verification(0,&w.comm,&w.stream)?;
                }
                ensure!(t.capture_statistics().1==captures,"epilogue timed graph capture");
                for (kind,proposal) in [input,changed.as_slice()].into_iter().enumerate() {
                    t.begin_verification(&proposal[..live],&w.comm,&w.stream)?;
                    let y=logits(&w.cuda,t)?;
                    let exact=expected.entry((live,kind,false)).or_insert_with(||y.clone())==&y;
                    t.finish_verification(0,&w.comm,&w.stream)?;
                    ensure!(w.control.all_admit(exact&&state(t)?==accepted)?,"epilogue logits/rollback changed");
                }
                records.push(json!({"rows":rows,"prefix":prefix,"trial":trial,"enabled":enabled,"live":live,
                    "samples_ms":samples,"metrics":metrics,"changed_input_logits_exact":true,"rollback_exact":true}));
            }
            if rows==16 {
                t.set_tiny_extend_for_probe(true,&w.stream)?;
                for live in [7usize,16] {
                    for _ in 0..2 {
                        t.begin_verification(&input[..live],&w.comm,&w.stream)?;
                        let y=logits(&w.cuda,t)?;
                        let exact=expected.entry((live,0,true)).or_insert_with(||y.clone())==&y;
                        t.finish_verification(0,&w.comm,&w.stream)?;
                        ensure!(w.control.all_admit(exact&&state(t)?==accepted)?,"epilogue append replay changed");
                    }
                }
                t.set_tiny_extend_for_probe(false,&w.stream)?;
            }
            t.begin_verification(&input[..rows],&w.comm,&w.stream)?;
            t.finish_verification(rows/2-1,&w.comm,&w.stream)?;
            let after=state(t)?;
            t.begin_verification(following,&w.comm,&w.stream)?;
            let y=logits(&w.cuda,t)?;t.finish_verification(0,&w.comm,&w.stream)?;
            let proof=(after.clone(),y);let exact=continuation.as_ref().is_none_or(|first|first==&proof);
            if continuation.is_none(){continuation=Some(proof);}
            ensure!(w.control.all_admit(exact&&state(t)?==after)?,"epilogue commit/following state differs");
            w.target_lineage=w.target_lineage.checked_add(1).context("epilogue lineage exhausted")?;
            t.restore_state_image(&saved,w.target_lineage,&w.comm,&w.stream)?;
            ensure!(state(t)?==accepted,"epilogue snapshot restore differs");
            std::fs::write(out.join("wide-epilogue-partial.json"),serde_json::to_vec_pretty(&records)?)?;
            eprintln!("Wide epilogue rank{rank} rows{rows} trial{trial} complete");
        }
        t.set_wide_epilogues_for_probe(true,kernels,&w.stream)?;
        w.verify.handoff_from(t)?;
    }
    execute(w,Command::Finish(Finish::Length))?;
    if std::env::var("CANADA_SCRATCH_SOLO_MICRO_COMBINED_SCREEN").as_deref()!=Ok("1") {
        execute(w,Command::Shutdown)?;
    }
    std::fs::write(out.join("wide-epilogue-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,
        "records":records,"all_exact":true,"commit_and_following_exact":true,"append_exact":true,
        "scope":"One65K public coding frontier. M16/M32 same-storage reversed pairs with existing FFN overlap. Changed proposals, partial commits, next logits, exact rollback and snapshot restore; M16 tiny-append bank replay. No changed target arithmetic."}))?)?;
    Ok(())
}
