//! Same-allocation final-head comparison on real retained model continuations.
use super::*;
use super::history32_generation_probe::{execute,state,restore};
fn hash(v:&[u8])->String {format!("{:x}",Sha256::digest(v))}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&std::env::var("CANADA_SCRATCH_PREFILL_HEAD_SCREEN").as_deref()==Ok("1"),
        "finite final-prefill head screen");
    let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let ids:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    let prefix=8193;let mut changed=ids[prefix..prefix+128].to_vec();
    changed.rotate_left(17);changed[0]=(changed[0]+53)%129280;
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let original_head=w.large.prefill_encoded_head_enabled();
    let row_screen=std::env::var("CANADA_SCRATCH_PREFILL_ROW_HEAD_SCREEN").as_deref()==Ok("1");
    let original_row_head=w.large.prefill_row_head_enabled();
    let publish_screen=std::env::var("CANADA_SCRATCH_PREFILL_PUBLISH_SCREEN").as_deref()==Ok("1");
    let transfer_screen=std::env::var("CANADA_SCRATCH_MOE_WIDE_TRANSFER_SCREEN").as_deref()==Ok("1");
    let original_transfer=crate::moe::wide_transfer_requested()?;
    let original_publish=w.large.prefill_sparse_publish_enabled();
    // Also exercise the fresh wide-prefill decoder-tail call site.
    w.large.set_prefill_encoded_head(true,&w.stream)?;
    if row_screen {w.large.set_prefill_row_head(true,&w.stream)?;}
    if publish_screen {w.large.set_prefill_sparse_publish(true,&w.stream)?;}
    if transfer_screen {w.large.set_moe_wide_transfer_for_probe(true,&w.stream)?;}
    let begin=execute(w,Command::Begin{images:Vec::new(),id:8_041_314,seed:2026110449,
        greedy:false,limit:1,tokens:ids[..prefix].to_vec()})?;
    let anchor=begin["output_ids"][0].as_u64().context("prefill head anchor")?as u32;
    let original=state(w)?;
    let(bytes,cpu)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
    ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
        >=8*1024usize.pow(3)+bytes+cpu)?,"prefill head bounded snapshot reserve");
    let saved=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
    let mut expected=std::collections::BTreeMap::<(usize,usize),Value>::new();
    let mut path_expected=std::collections::BTreeMap::new();
    let mut control_logits=std::collections::BTreeMap::<(usize,usize),Vec<f32>>::new();
    let mut records=Vec::new();
    for(trial,enabled)in[false,true,true,false].into_iter().enumerate() {
        for live in [33usize,128] {
            for repeat in 0..6 {
                restore(w,&saved,anchor,&original)?;
                let kind=repeat%2;
                let input=if kind==0{&ids[prefix..prefix+live]}else{&changed[..live]};
                w.large.handoff_from(&w.verify)?;
                if transfer_screen {w.large.set_moe_wide_transfer_for_probe(enabled,&w.stream)?;}
                else if publish_screen {w.large.set_prefill_sparse_publish(enabled,&w.stream)?;}
                else if row_screen {w.large.set_prefill_row_head(enabled,&w.stream)?;}
                else {w.large.set_prefill_encoded_head(enabled,&w.stream)?;}
                w.control.agree(&[0xd5,live as u8,trial as u8,repeat as u8])?;
                let captures=w.large.capture_statistics().1;
                let at=Instant::now();w.large.append(input,&w.comm,&w.stream)?;
                let milliseconds=at.elapsed().as_secs_f64()*1000.;
                let recaptured=w.large.capture_statistics().1-captures;
                let values=w.large.last_prompt_logits()?;
                ensure!(values.iter().all(|v|v.is_finite()),"nonfinite prefill head output");
                let control=control_logits.entry((live,kind)).or_insert_with(||values.clone());
                let numeric=crate::dspark_generation::compare_logits(control,&values);
                ensure!(numeric["total_variation"].as_f64().unwrap()<0.0001,
                    "prefill head distribution differs unexpectedly");
                if publish_screen||transfer_screen {ensure!(numeric["bitwise_equal"]==true,"exact prefill candidate changed logits");}
                let logits:Vec<u8>=values.iter().flat_map(|v|v.to_le_bytes()).collect();
                let normalized=w.large.prefill_head_input_for_probe()?;
                if repeat==0||publish_screen {
                    let all=w.large.logits()?;
                    let raw=unsafe{w.cuda.read_device(all.ptr(),all.bytes())}?;
                    let start=(live-1)*129280*4;let end=live*129280*4;
                    ensure!(raw[..start].iter().chain(&raw[end..]).all(|&x|x==0),
                        "prefill final-only output leaked an unpublished row");
                }
                // Match the ordinary retained EXTEND boundary: the target has
                // committed the suffix, so advance conditioning and history
                // before comparing or restoring a complete session image.
                w.session.as_mut().unwrap().resume_after_extend(
                    &w.large,prefix as u64,input,&w.stream)?;
                w.verify.handoff_from(&w.large)?;
                let accepted=state(w)?;
                w.verify.begin_verification(&ids[prefix+live..prefix+live+6],&w.comm,&w.stream)?;
                let(y,n)=w.verify.pending_logits()?;ensure!(n==6,"prefill head following extent");
                let following=unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?;
                w.verify.finish_verification(0,&w.comm,&w.stream)?;
                ensure!(state(w)?==accepted,"prefill head following rollback");
                let proof=json!({"normalized":hash(&normalized),"following":hash(&following),"state":accepted});
                let exact=expected.entry((live,kind)).or_insert_with(||proof.clone())==&proof;
                let path=json!({"logits":hash(&logits),"proof":proof});
                let same_path=path_expected.entry((live,kind,enabled)).or_insert_with(||path.clone())==&path;
                ensure!(w.control.all_admit(exact&&same_path)?,"prefill head state/following or same-path mismatch");
                records.push(json!({"live":live,"trial":trial,"repeat":repeat,"enabled":enabled,
                    "kind":kind,"warmup":recaptured>0||repeat<2,"new_captures":recaptured,
                    "milliseconds":milliseconds,"state_exact":exact,"same_path_exact":same_path,
                    "distribution":numeric,"proof":path}));
            }
        }
        std::fs::write(out.join("prefill-head-partial.json"),serde_json::to_vec_pretty(&records)?)?;
        eprintln!("prefill head rank{rank} trial{trial} enabled{enabled}: state/following exact");
    }
    restore(w,&saved,anchor,&original)?;drop(saved);
    w.large.set_prefill_encoded_head(original_head,&w.stream)?;
    w.large.set_prefill_row_head(original_row_head,&w.stream)?;
    w.large.set_prefill_sparse_publish(original_publish,&w.stream)?;
    if transfer_screen {w.large.set_moe_wide_transfer_for_probe(original_transfer,&w.stream)?;}
    execute(w,Command::Finish(Finish::Length))?;
    std::fs::write(out.join("prefill-head-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"prefix_tokens":prefix,"fresh_encoded_prefill_completed":true,
        "comparison":if transfer_screen{"M128 exact expert kernel transfer"}else if publish_screen{"sparse final-logit publication with all unpublished rows zero"}else if row_screen{"last-row HC/RMS preparation with the same encoded head"}else{"encoded head with full-row normalization"},
        "records":records,"all_state_and_following_exact":true,"original_backend_restored":true,
        "scope":"One fresh8193-token encoded-head prefill, then ABBA retained33/128-row appends, two actual model-input patterns, four warmed timings each. Canonical BF16 values and FP32 logits; only FP32 reduction order differs. Same normalized inputs, retained target/draft state, following6 logits and rollback; unpublished rows remain zero. No full outcome-quality claim."
    }))?)?;Ok(())
}
