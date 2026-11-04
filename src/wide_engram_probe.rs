//! Focused fixed-frontier check for extending layer0/row-I/O overlap to M16/32.
use super::*;
use super::history32_generation_probe::execute;
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.long_verify.is_some()&&w.copy32_verify.is_some()
        &&std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref()==Ok("1"),
        "wide Engram screen requires idle graph-enabled wide owners and io_uring");
    let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    let prefix=131072usize;ensure!(tokens.len()>=prefix+32,"wide Engram fixture extent");
    execute(w,Command::Begin{id:9_100_101,seed:2026110321,greedy:false,limit:1,
        tokens:tokens[..prefix].to_vec(),images:Vec::new()})?;
    let initial=json!({"position":w.verify.position(),"carry":w.verify.carry_digests()?,
        "encoder":w.verify.encoder_tail_digest()?});
    let mut records=Vec::new();
    let worker=std::env::var("CANADA_SCRATCH_ENGRAM32_WORKER_SCREEN").as_deref()==Ok("1");
    for rows in if worker{vec![32usize]}else{vec![16usize,32]} {for cold in [false,true] {
        let variants=if worker{vec![5u8,7]}else if rows==16{vec![0u8,1]}else{vec![0u8,1,2,3,4,5]};
        let reference=variants[0]as usize;
        let mut logits:Vec<Vec<String>>=vec![Vec::new();8];let mut trials=Vec::new();
        for pair in 0..8 {
            let mut order=variants.clone();order.rotate_left((pair/2)%variants.len());
            if pair%2==1{order.reverse();}
            for variant in order {
                let overlap=variant&1!=0;let paired32=variant>=2;let adaptive=variant>=4;
                let target=if rows==16{w.long_verify.as_mut().unwrap()}else{w.copy32_verify.as_mut().unwrap()};
                target.handoff_from(&w.verify)?;target.set_engram_overlap_for_probe(overlap)?;
                if rows==32{
                    if adaptive{target.set_adaptive_engram32_for_probe(true)?;}
                    else{target.set_paired_engram32_for_probe(paired32)?;}
                    if worker{target.set_engram32_worker_for_probe(variant==7)?;}
                }
                if cold{target.clear_engram_cache_for_probe(&w.stream)?;}
                w.control.agree(b"wide-engram-trial-ready")?;
                let before=target.capture_statistics().1;let began=Instant::now();
                target.begin_verification(&tokens[prefix..prefix+rows],&w.comm,&w.stream)?;
                let milliseconds=began.elapsed().as_secs_f64()*1000.;
                let host=target.verification_host_ms()?;
                let metrics=serde_json::to_value(target.verification_metrics()?)?;
                if pair>=2{
                    ensure!(target.capture_statistics().1==before,"timed wide Engram graph capture");
                    if overlap{ensure!(metrics["stages_ms"].as_array().unwrap().iter()
                        .any(|x|x[0]=="engram_upload_order"),"wide Engram overlap not exercised");}
                    trials.push(json!({"pair":pair-2,"variant":variant,"overlap":overlap,
                        "paired32":paired32,"adaptive":adaptive,"milliseconds":milliseconds,"host_ms":host}));
                }
                if pair==1||pair==7 {
                    let (values,live)=target.pending_local_logits_for_probe()?;ensure!(live==rows,"wide Engram logits extent");
                    logits[variant as usize].push(format!("{:x}",Sha256::digest(
                        unsafe{w.cuda.read_device(values.ptr(),rows*32320*4)}?)));
                }
                target.finish_verification(0,&w.comm,&w.stream)?;
                w.verify.handoff_from(target)?;
            }
        }
        let after=json!({"position":w.verify.position(),"carry":w.verify.carry_digests()?,
            "encoder":w.verify.encoder_tail_digest()?});
        ensure!(initial==after&&variants.iter().all(|&v|logits[v as usize]==logits[reference])&&logits[reference].len()==2,
            "wide Engram overlap changed logits or retained state");
        records.push(json!({"rows":rows,"application_cold":cold,"trials":trials,
            "variants":variants,"logits_sha256":logits[reference],"retained_state_exact":true}));
        std::fs::write(out.join("wide-engram-partial.json"),serde_json::to_vec(&records)?)?;
    }}
    std::fs::write(out.join("wide-engram-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"all_exact":true,"records":records,"prompt_tokens":prefix,"logit_columns_per_rank":32320,"persistent_helper_screen":worker,
        "scope":"One public131K repository frontier, fixed teacher-forced16/32 rows; existing arithmetic and memory. Six alternating pairs after two warmups, warm and application-cache-cold states. Exact FP32 logits and retained carry/encoder bytes. Includes host row lookup and GPU completion, excludes measured graph capture and cross-rank diagnostic downloads. No end-to-end or general task-speed claim."}))?)?;
    execute(w,Command::Finish(Finish::Length))?;execute(w,Command::Shutdown)?;Ok(())
}
