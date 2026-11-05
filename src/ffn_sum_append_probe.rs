//! One-load fixed-storage FFN sum fusion and small retained-append screen.
use super::*;
use super::history32_generation_probe::{execute,state,restore};
fn target_state(t:&Target)->Result<Value> {
    Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))
}
fn hash(v:&[u8])->String {format!("{:x}",Sha256::digest(v))}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.copy32_verify.is_some()&&w.copy32_buffers.is_some()
        &&w.long_verify.is_some(),"sum/append probe needs existing M6/M16/M32 owners");
    let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let ids:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    let prefix=65541;let input=&ids[prefix..prefix+32];let mut changed=input.to_vec();
    changed.rotate_left(8);changed[0]=(changed[0]+37)%129280;
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let begin=execute(w,Command::Begin{images:Vec::new(),id:8_041_289,seed:2026110417,
        greedy:false,limit:1,tokens:ids[..prefix].to_vec()})?;
    let anchor=begin["output_ids"][0].as_u64().context("sum/append anchor")? as u32;
    let original=state(w)?;
    if std::env::var("CANADA_SCRATCH_GATHER_ROWS_SCREEN").as_deref()==Ok("1") {
        super::gather_rows_probe::check(w,rank,out)?;
    }
    let (bytes,cpu)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
    ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
        >=8*1024usize.pow(3)+bytes+cpu)?,"sum/append snapshot reserve");
    let saved=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
    let mut fusion=Vec::new();
    for rows in [6usize,16,32] {
        let mut expected=std::collections::BTreeMap::<(usize,usize),String>::new();
        for (trial,enabled) in [false,true,true,false].into_iter().enumerate() {
            let t=match rows {6=>&mut w.verify,16=>w.long_verify.as_mut().unwrap(),_=>w.copy32_verify.as_mut().unwrap()};
            // All profiles share the accepted storage; handoff follows below.
            if !t.profile_is_current() {anyhow::bail!("sum probe profile handoff absent");}
            t.set_ffn_sum_for_probe(enabled,&w.stream)?;
            let incoming=target_state(t)?;
            for live in [if rows==6{3}else{rows/2+1},rows] {
                for _ in 0..3 {t.begin_verification(&input[..live],&w.comm,&w.stream)?;
                    t.finish_verification(0,&w.comm,&w.stream)?;}
                let captures=t.capture_statistics().1;let mut samples=Vec::new();
                for repeat in 0..6 {
                    w.control.agree(&[0xe1,rows as u8,live as u8,trial as u8,repeat])?;
                    let at=Instant::now();t.begin_verification(&input[..live],&w.comm,&w.stream)?;
                    t.finish_verification(0,&w.comm,&w.stream)?;samples.push(at.elapsed().as_secs_f64()*1000.);
                }
                ensure!(captures==t.capture_statistics().1,"fusion timed recapture");
                for(kind,proposal)in[input,changed.as_slice()].into_iter().enumerate() {
                    t.begin_verification(&proposal[..live],&w.comm,&w.stream)?;
                    let(y,n)=t.pending_local_logits_for_probe()?;ensure!(n==live,"fusion local-logit extent");
                    let digest=hash(&unsafe{w.cuda.read_device(y.ptr(),live*32320*4)}?);
                    let exact=expected.entry((live,kind)).or_insert_with(||digest.clone())==&digest;
                    t.finish_verification(0,&w.comm,&w.stream)?;
                    ensure!(w.control.all_admit(exact&&target_state(t)?==incoming)?,"fusion logits or rollback differ");
                }
                fusion.push(json!({"rows":rows,"live":live,"trial":trial,"enabled":enabled,
                    "samples_ms":samples,"all_exact":true,"new_timed_captures":0}));
            }
        }
        // Enable the accumulated candidate, then hand the idle state to the
        // next profile. No target/drafter history changed during these trials.
        match rows {
            6=>{w.verify.set_ffn_sum_for_probe(true,&w.stream)?;
                w.long_verify.as_mut().unwrap().handoff_from(&w.verify)?;},
            16=>{let t=w.long_verify.as_mut().unwrap();t.set_ffn_sum_for_probe(true,&w.stream)?;
                w.copy32_verify.as_mut().unwrap().handoff_from(t)?;},
            _=>{let t=w.copy32_verify.as_mut().unwrap();t.set_ffn_sum_for_probe(true,&w.stream)?;
                w.verify.handoff_from(t)?;}
        }
        std::fs::write(out.join("sum-append-progress.json"),serde_json::to_vec_pretty(&fusion)?)?;
        eprintln!("rank{rank} FFN sum rows{rows} exact");
    }
    ensure!(state(w)?==original,"fusion altered accepted session");
    let mut appends=Vec::new();
    if std::env::var("CANADA_SCRATCH_APPEND32_SCREEN").as_deref()==Ok("1") {
        let mut expected=std::collections::BTreeMap::<(usize,usize),Value>::new();
        for live in [17usize,24,32] {
            for (trial,path) in [128usize,32,32,128].into_iter().enumerate() {
                for repeat in 0..3 {
                    restore(w,&saved,anchor,&original)?;
                    let input=&ids[prefix..prefix+live];
                    if path==128 {w.large.handoff_from(&w.verify)?;}
                    else {let t=w.copy32_verify.as_mut().unwrap();t.handoff_from(&w.verify)?;
                        t.set_tiny_extend_for_probe(true,&w.stream)?;}
                    w.control.agree(&[0xe2,live as u8,path as u8,trial as u8,repeat])?;
                    let at=Instant::now();
                    if path==128 {w.large.prefill_wide(&mut w.wides,input,&w.comm,&w.stream)?;}
                    else {let t=w.copy32_verify.as_mut().unwrap();t.begin_verification(input,&w.comm,&w.stream)?;
                        t.finish_verification(live,&w.comm,&w.stream)?;}
                    let target_ms=at.elapsed().as_secs_f64()*1000.;
                    let t=if path==128{&mut w.large}else{w.copy32_verify.as_mut().unwrap()};
                    let logits=unsafe{w.cuda.read_device(t.logits()?.ptr()+((live-1)*129280*4)as u64,129280*4)}?;
                    let sampled=t.sample_token(2026110433,&w.stream)?;
                    let at=Instant::now();
                    if path==128 {w.session.as_mut().unwrap().resume_after_extend(t,prefix as u64,input,&w.stream)?;}
                    else {w.session.as_mut().unwrap().resume_after_medium_extend(t,w.copy32_buffers.as_mut().unwrap(),
                        prefix as u64,input,&w.comm,&w.stream)?;t.set_tiny_extend_for_probe(false,&w.stream)?;}
                    let draft_ms=at.elapsed().as_secs_f64()*1000.;w.verify.handoff_from(t)?;
                    let accepted=state(w)?;
                    w.verify.begin_verification(&ids[prefix+live..prefix+live+6],&w.comm,&w.stream)?;
                    let(y,n)=w.verify.pending_logits()?;ensure!(n==6,"append following extent");
                    let following=unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?;
                    w.verify.finish_verification(0,&w.comm,&w.stream)?;
                    ensure!(state(w)?==accepted,"append following rollback changed accepted state");
                    let proof=json!({"logits":hash(&logits),"following":hash(&following),"state":accepted,"sampled":sampled});
                    let exact=expected.entry((live,path)).or_insert_with(||proof.clone())==&proof;
                    ensure!(w.control.all_admit(exact)?,"same-path retained append drift");
                    if rank==0&&repeat==0&&trial<2 {
                        std::fs::write(out.join(format!("append{live}-m{path}-last.bin")),&logits)?;
                        std::fs::write(out.join(format!("append{live}-m{path}-following.bin")),&following)?;
                    }
                    appends.push(json!({"live":live,"path":path,"trial":trial,"repeat":repeat,"warmup":repeat==0,
                        "target_ms":target_ms,"draft_ms":draft_ms,"proof":proof,"same_path_exact":true,"rollback_exact":true}));
                }
            }
            std::fs::write(out.join("append32-progress.json"),serde_json::to_vec_pretty(&appends)?)?;
            eprintln!("rank{rank} append32 live{live} passed same-path state checks");
        }
    }
    restore(w,&saved,anchor,&original)?;
    execute(w,Command::Finish(Finish::Length))?;
    std::fs::write(out.join("sum-append-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"prefix_tokens":prefix,"fusion":fusion,"appends":appends,"fusion_exact":true,
        "append_same_path_exact":true,"scope":"Same-allocation ABBA FFN fusion at M3/M6/M16/M32; short retained append timing at17/24/32 with original128 versus32-row request-window path. Target/drafter restoration and following rollback checked; cross-width numerical comparison remains separate. No serving-policy promotion."
    }))?)?;
    Ok(())
}
