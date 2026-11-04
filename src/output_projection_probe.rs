//! Same-allocation output-projection comparison; current target precision.
use super::*;
use super::history32_generation_probe::execute;
fn state(t:&Target)->Result<Value> {
    Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))
}
fn logits(cuda:&Cuda,t:&Target)->Result<Vec<u8>> {
    let (y,n)=t.pending_local_logits_for_probe()?;
    unsafe{cuda.read_device(y.ptr(),n*32320*4)}
}
pub(super) fn run(w:&mut Worker,rank:usize,kernels:&Path,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.long_verify.is_some()
        &&std::env::var("CANADA_SCRATCH_OUTPUT_PROJECTION_SCREEN").as_deref()==Ok("1"),
        "isolated output projection screen");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let ids:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let prefix=65541usize;
    execute(w,Command::Begin{images:Vec::new(),id:8_020_300,seed:20261102,
        greedy:false,limit:1,tokens:ids[..prefix].to_vec()})?;
    let mut records=Vec::new();
    for physical in [6usize,16] {
        if physical==16 {w.long_verify.as_mut().unwrap().handoff_from(&w.verify)?;}
        let t=if physical==6{&mut w.verify}else{w.long_verify.as_mut().unwrap()};
        let accepted=state(t)?;let bytes=t.compact_state_image_bytes()?;
        ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(
            w.cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+bytes+64*1024usize.pow(2))?,
            "output projection state reserve");
        let saved=t.save_state_image(bytes,&w.comm,&w.stream)?;
        let mut other=t.prepare_wo_b_alternates_for_probe(kernels)?;
        let mut expected=std::collections::BTreeMap::<(usize,usize),Vec<u8>>::new();
        let mut commit_reference:Option<(Value,Vec<u8>)>=None;
        for (trial,mode) in [0u8,2,1,1,2,0].into_iter().enumerate() {
            t.set_wo_b_pdl_for_probe(mode,&mut other,&w.stream)?;
            for live in if physical==6{[3,6]}else{[9,16]} {
                let input=&ids[prefix..prefix+live];
                for _ in 0..3 {t.begin_verification(input,&w.comm,&w.stream)?;
                    t.finish_verification(0,&w.comm,&w.stream)?;}
                let captures=t.capture_statistics().1;let mut samples=Vec::new();
                for repeat in 0..6u8 {
                    w.control.agree(&[0xb0,physical as u8,trial as u8,live as u8,repeat])?;
                    let at=Instant::now();t.begin_verification(input,&w.comm,&w.stream)?;
                    w.stream.synchronize()?;samples.push(at.elapsed().as_secs_f64()*1000.);
                    t.finish_verification(0,&w.comm,&w.stream)?;
                }
                ensure!(t.capture_statistics().1==captures,"output projection timed capture");
                for (changed,input) in [input,&ids[prefix+32..prefix+32+live]].into_iter().enumerate() {
                    t.begin_verification(input,&w.comm,&w.stream)?;
                    let y=logits(&w.cuda,t)?;
                    let first=expected.entry((live,changed)).or_insert_with(||y.clone());
                    let exact=first==&y;t.finish_verification(0,&w.comm,&w.stream)?;
                    ensure!(w.control.all_admit(exact&&state(t)?==accepted)?,
                        "output projection logits or rollback differ");
                }
                records.push(json!({"physical":physical,"live":live,"trial":trial,"mode":mode,
                    "samples_ms":samples,"changed_input_logits_exact":true,"rollback_exact":true}));
                std::fs::write(out.join("output-projection-partial.json"),serde_json::to_vec_pretty(&records)?)?;
            }
            let commit=if physical==6{3}else{7};
            t.begin_verification(&ids[prefix..prefix+physical],&w.comm,&w.stream)?;
            t.finish_verification(commit,&w.comm,&w.stream)?;
            let after=state(t)?;
            t.begin_verification(&ids[prefix+commit..prefix+commit+6],&w.comm,&w.stream)?;
            let following=logits(&w.cuda,t)?;t.finish_verification(0,&w.comm,&w.stream)?;
            let proof=(after.clone(),following);
            let exact=commit_reference.as_ref().is_none_or(|r|r==&proof);
            if commit_reference.is_none(){commit_reference=Some(proof);}
            ensure!(w.control.all_admit(exact&&state(t)?==after)?,"output projection commit/following differs");
            w.target_lineage=w.target_lineage.checked_add(1).context("output projection lineage exhausted")?;
            t.restore_state_image(&saved,w.target_lineage,&w.comm,&w.stream)?;
            ensure!(state(t)?==accepted,"output projection snapshot restore differs");
            eprintln!("output projection rank{rank} M{physical} trial{trial} mode{mode} complete");
        }
        t.set_wo_b_pdl_for_probe(0,&mut other,&w.stream)?;
        if physical==16 {w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;}
    }
    execute(w,Command::Finish(Finish::Length))?;execute(w,Command::Shutdown)?;
    std::fs::write(out.join("output-projection-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"commit_and_following_exact":true,
        "fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "scope":"65K public prefix; live3/6/9/16; original, fenced FI without/with PDL, reverse order; three warm and six measured replays. Full local-vocabulary, changed input, retained/partial commit/following/restore checks on four ranks. Same operand allocations and precision. No serving change or outcome-speed claim."
    }))?)?;Ok(())
}
