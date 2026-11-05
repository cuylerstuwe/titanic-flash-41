//! Grouped, same-allocation prefill mechanism screen. No serving dispatch.
use super::*;
use super::history32_generation_probe::{execute,state,restore};
fn hash(v:&[u8])->String {format!("{:x}",Sha256::digest(v))}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&std::env::var("CANADA_SCRATCH_PREFILL_BOUNDS_SCREEN").as_deref()==Ok("1"),
        "finite prefill bounds screen");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let corpus:Value=serde_json::from_slice(raw)?;
    let ids:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    let prefix=8193;let mut changed=ids[prefix..prefix+1024].to_vec();
    changed.rotate_left(17);changed[0]=(changed[0]+53)%129280;
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    let begin=execute(w,Command::Begin{images:Vec::new(),id:8_041_320,seed:2026110450,
        greedy:false,limit:1,tokens:ids[..prefix].to_vec()})?;
    let anchor=begin["output_ids"][0].as_u64().context("prefill bounds anchor")?as u32;
    let original=state(w)?;
    let(bytes,cpu)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
    ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
        >=8*1024usize.pow(3)+bytes+cpu)?,"prefill bounds snapshot reserve");
    let saved=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
    let mut records=Vec::new();
    for (family,width,lives) in [("upper_rows",128,vec![65usize,96,127]),
        ("expert_transfer",512,vec![129,512]),("expert_transfer",1024,vec![513,1024])] {
        let mut expected=std::collections::BTreeMap::<(usize,usize),Value>::new();
        for(trial,enabled)in[false,true,true,false].into_iter().enumerate() {
            for &live in &lives {
                for repeat in 0..4 {
                    restore(w,&saved,anchor,&original)?;
                    let kind=repeat%2;
                    let input=if kind==0{&ids[prefix..prefix+live]}else{&changed[..live]};
                    w.large.handoff_from(&w.verify)?;
                    let wide_index=if width==128 {
                        w.large.set_prefill_upper_rows_for_probe(enabled,&w.stream)?;None
                    }else {
                        let index=w.wides.iter().position(|t|t.physical_rows()==width).context("prefill transfer width absent")?;
                        w.wides[index].set_moe_wide_transfer_for_probe(enabled,&w.stream)?;Some(index)
                    };
                    w.control.agree(&[0xd6,(width/128)as u8,live as u8,trial as u8,repeat as u8])?;
                    let at=Instant::now();
                    if let Some(index)=wide_index {
                        w.large.prefill_wide(&mut w.wides[index..index+1],input,&w.comm,&w.stream)?;
                    }else{w.large.append(input,&w.comm,&w.stream)?;}
                    let milliseconds=at.elapsed().as_secs_f64()*1000.;
                    let phases=serde_json::to_value(w.large.append_metrics()?)?;
                    ensure!(phases["physical_rows"]==width,"prefill probe did not use selected width");
                    let values=w.large.last_prompt_logits()?;
                    ensure!(values.iter().all(|v|v.is_finite()),"nonfinite prefill bounds output");
                    let logits:Vec<u8>=values.iter().flat_map(|v|v.to_le_bytes()).collect();
                    let normalized=w.large.prefill_head_input_for_probe()?;
                    w.session.as_mut().unwrap().resume_after_extend(&w.large,prefix as u64,input,&w.stream)?;
                    w.verify.handoff_from(&w.large)?;let accepted=state(w)?;
                    w.verify.begin_verification(&ids[prefix+live..prefix+live+6],&w.comm,&w.stream)?;
                    let(y,n)=w.verify.pending_logits()?;ensure!(n==6,"prefill bounds following extent");
                    let following=unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?;
                    w.verify.finish_verification(0,&w.comm,&w.stream)?;
                    ensure!(state(w)?==accepted,"prefill bounds following rollback");
                    let proof=json!({"logits":hash(&logits),"normalized":hash(&normalized),
                        "following":hash(&following),"state":accepted});
                    let exact=expected.entry((live,kind)).or_insert_with(||proof.clone())==&proof;
                    ensure!(w.control.all_admit(exact)?,"prefill bounds changed output/state/following");
                    records.push(json!({"family":family,"physical_rows":width,"live":live,"trial":trial,
                        "repeat":repeat,"enabled":enabled,"kind":kind,"warmup":repeat<2,
                        "milliseconds":milliseconds,"all_exact":exact,"phases":phases,"proof":proof}));
                }
            }
            std::fs::write(out.join("prefill-bounds-partial.json"),serde_json::to_vec_pretty(&records)?)?;
            eprintln!("prefill bounds rank{rank} {family} M{width} trial{trial} enabled{enabled}: exact");
        }
        restore(w,&saved,anchor,&original)?;
        if width==128 {w.large.set_prefill_upper_rows_for_probe(crate::target::prefill_upper_rows_requested()?,&w.stream)?;}
        else {
            w.wides.iter_mut().find(|t|t.physical_rows()==width).unwrap()
                .set_moe_wide_transfer_for_probe(crate::moe::prefill_transfer_requested()?,&w.stream)?;
        }
    }
    restore(w,&saved,anchor,&original)?;drop(saved);
    execute(w,Command::Finish(Finish::Length))?;execute(w,Command::Shutdown)?;
    std::fs::write(out.join("prefill-bounds-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"prefix_tokens":prefix,"fixture_sha256":hash(raw),"records":records,
        "all_logits_state_and_following_exact":true,"original_backend_restored":true,
        "scope":"Same-allocation ABBA at8193 retained tokens. M12865/96/127 dense/expert/mHC row bounds and M512129/512 plus M1024513/1024 expert code transfer. Two input patterns, first two of four samples per arm excluded for warmup. Exact final logits, normalized input, complete accepted session, six following logits and rollback. Existing canonical precision, weights, capacities and scratch. No serving/task-speed claim."
    }))?)?;Ok(())
}
