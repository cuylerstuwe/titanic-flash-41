//! Small sampled generation comparison on a saved public tool-call frontier.
use crate::{checkpoint::Checkpoint,cuda::{Cuda,Stream},dspark_generation::{GreedySession,LongHistoryBuffers},
    layout::RankPlan,roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde::Deserialize;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};
#[derive(Deserialize)]
struct Case {name:String,prompt_ids:Vec<u32>,anchor:u32,proposal_ids:Vec<u32>,prefix_match:usize}
#[derive(Deserialize)]
struct Fixture {cases:Vec<Case>}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,core:&[u8],stream:&Stream,cp:&Checkpoint,plan:&RankPlan,
    large:&mut Target,verify:&mut Target,longer:&mut Target,wides:&mut[Target],comm:&Roce,
    staged:&Path,kernels:&Path)->Result<Value> {
    ensure!(std::env::var("CANADA_SCRATCH_LONG_HISTORY_GENERATION").as_deref()==Ok("1")
        && std::env::var("CANADA_SCRATCH_STOCHASTIC_DRAFT").as_deref()==Ok("1")
        && std::env::var("CANADA_SCRATCH_HISTORY_MAX_MATCH").as_deref()==Ok("64")
        && large.position()==0 && large.context_capacity()==1048576,"isolated sampled long history screen");
    let long_context=std::env::var("CANADA_SCRATCH_LONG_HISTORY_CONTEXT_SCREEN").as_deref()==Ok("1");
    let raw:&[u8]=if long_context {include_bytes!("../tests/fixtures/long-history-context-v1.json")}
        else {include_bytes!("../tests/fixtures/long-history-score-v1.json")};
    let fixture:Fixture=serde_json::from_slice(raw)?;
    let case=&fixture.cases[if long_context {0}else{8}];
    ensure!(case.prefix_match==64 && case.proposal_ids.len()==15
        && (!long_context || case.prompt_ids.len()==274108),"strong causal source");
    let mut teacher=vec![case.anchor];teacher.extend_from_slice(&case.proposal_ids);
    let mut buffers=LongHistoryBuffers::new(cuda,core)?;
    let mut session:Option<GreedySession>=None;
    let mut records=Vec::new();let mut saved=None;let mut image_bytes=0;
    large.enable_cached_extend_floor()?;
    for repeat in 0..3 {
        let seed=20261058+repeat as u64;
        let maximum=if repeat==0 {128}else{256};
        for slot in 0..2 {
            let enabled=(repeat+slot)%2==1;
            let variant=if enabled {"long"}else{"six"};
            if !long_context || saved.is_none() {
                large.prefill_wide(wides,&case.prompt_ids,comm,stream)?;
                if let Some(s)=session.as_mut(){s.reset_from_prefill(large,cuda,core,stream)?;}
                else{session=Some(GreedySession::from_prefill(large,cuda,core,stream,cp,plan,staged,kernels)?);}
                let s=session.as_mut().unwrap();s.configure_sampling(seed,Some(1))?;
                s.configure_history_lookup(large,&case.prompt_ids)?;
                verify.handoff_from(large)?;
            } else {
                let anchor=session.as_mut().unwrap().restore_state_image(verify,saved.as_ref().unwrap(),
                    857600+(repeat*2+slot)as u64,comm,stream)?;
                ensure!(anchor==case.anchor && verify.position()==case.prompt_ids.len()as u64,
                    "long-context exact saved frontier");
            }
            let s=session.as_mut().unwrap();s.configure_sampling(seed,Some(1))?;
            s.configure_algorithm_for_probe(true,0.7,stream)?;s.set_causal_three_threshold_for_probe(Some(0.20))?;
            if repeat==0 && slot==0 {
                for live in 1..=6 {
                    verify.begin_verification(&teacher[..live],comm,stream)?;
                    verify.verification_sample(seed,stream)?;
                    if long_context && live==6 && std::env::var("CANADA_SCRATCH_NARROW_SHARED_MOE").as_deref()==Ok("1") {
                        let (logits,count)=verify.pending_logits()?;stream.synchronize()?;
                        let bytes=unsafe{cuda.read_device(logits.ptr(),count*129280*4)?};
                        let hash=format!("{:x}",Sha256::digest(&bytes));
                        ensure!(count==6 && hash=="c4481a9ebc3020c6d8ff82cd847dcd8d717783291c67a7a4aa24f7ce4e9c0a77",
                            "shared M6 fixed-frontier logits differ from independent s574: {hash}");
                        eprintln!("rank{} shared M6 logits match s574 exactly",plan.rank);
                    }
                    verify.finish_verification(0,comm,stream)?;
                }
                longer.handoff_from(verify)?;longer.begin_verification(&teacher,comm,stream)?;
                longer.verification_sample(seed,stream)?;
                if long_context && std::env::var("CANADA_SCRATCH_VERIFY16_SHARED_MOE").as_deref()==Ok("1") {
                    let (logits,live)=longer.pending_logits()?;stream.synchronize()?;
                    let bytes=unsafe{cuda.read_device(logits.ptr(),live*129280*4)?};
                    let hash=format!("{:x}",Sha256::digest(&bytes));
                    ensure!(live==16 && hash=="9cda10a267972729ba215eb3632a973007a5bc466e798fed5e8d2f8b98c63d2c",
                        "shared-workspace fixed-frontier logits differ from independent s574: {hash}");
                    eprintln!("rank{} shared M16 logits match s574 exactly",plan.rank);
                }
                longer.finish_verification(0,comm,stream)?;
                verify.handoff_from(longer)?;
            }
            if long_context && saved.is_none() {
                image_bytes=s.compact_state_image_bytes(verify)?.0;
                ensure!(image_bytes<=768*1024usize.pow(2),"long-context snapshot budget");
                let image=s.save_state_image(verify,case.anchor,image_bytes,comm,stream)?;
                ensure!(image.manifest()["target"]["compact"]==true,"compact comparison image");saved=Some(image);
            }
            let captures_before=verify.capture_statistics().1+longer.capture_statistics().1;
            let mut emitted=vec![case.anchor];let mut anchor=case.anchor;
            let mut steps=Vec::new();let mut forced_neural=false;
            let begin=Instant::now();
            while emitted.len()<maximum && anchor!=1 {
                let remaining=maximum-emitted.len();
                let force=!forced_neural && emitted.len()>=64;
                if force{s.configure_history_policy_for_probe(None)?;}
                let long=if enabled && !force {s.try_long_history_step(verify,longer,&mut buffers,anchor,remaining,comm,stream)?}else{None};
                let step=match long {Some(step)=>step,None=>s.step(verify,anchor,remaining.min(6),comm,stream)?};
                if force{ensure!(step.proposal_source=="dspark","forced neural fallback");forced_neural=true;s.configure_history_policy_for_probe(Some(8))?;}
                ensure!(step.tokens[0]==anchor && step.tokens.len()<=remaining,"visible output limit/anchor");
                emitted.extend_from_slice(&step.tokens[1..]);emitted.push(step.next_anchor);anchor=step.next_anchor;steps.push(step);
            }
            let seconds=begin.elapsed().as_secs_f64();
            ensure!(emitted.len()<=maximum && verify.position()==(case.prompt_ids.len()+emitted.len()-1)as u64,
                "sampled output/target frontier");
            let long_steps=steps.iter().filter(|s|s.proposal_source=="long_history_point_mass").count();
            let long_committed:usize=steps.iter().filter(|s|s.proposal_source=="long_history_point_mass").map(|s|s.tokens.len()).sum();
            let raw=emitted.iter().flat_map(|id|id.to_le_bytes()).collect::<Vec<_>>();
            let record=json!({"repeat":repeat,"warmup":repeat==0,"seed":seed,"variant":variant,"maximum_output_tokens":maximum,
                "output_tokens":emitted.len(),"decode_seconds":seconds,"decode_tps":(emitted.len()-1)as f64/seconds,
                "long_steps":long_steps,"long_committed_inputs":long_committed,"forced_neural_fallback":forced_neural,
                "new_target_graph_captures":verify.capture_statistics().1+longer.capture_statistics().1-captures_before,
                "output_sha256":format!("{:x}",Sha256::digest(raw)),"output_ids":emitted,"steps":steps,
                "state":{"target_position":verify.position(),"carry":verify.carry_digests()?,"encoder":verify.encoder_tail_digest()?,
                    "draft":s.conditioning_state_digest_for_probe(comm,stream)?}});
            eprintln!("long history generation rank{} repeat{repeat} {variant}: {:.2}tps {long_steps}longsteps",plan.rank,(record["output_tokens"].as_u64().unwrap()-1)as f64/seconds);
            records.push(record);
            if !long_context {
                verify.reset(856900+(repeat*2+slot)as u64,comm,stream)?;large.handoff_from(verify)?;
            }
        }
    }
    Ok(json!({"rank":plan.rank,"long_history_generation":true,"long_context":long_context,
        "prefill_count":if long_context {1}else{6},"reused_state_image_bytes":image_bytes,"case":case.name,"prompt_tokens":case.prompt_ids.len(),
        "target_temperature":1.,"target_top_p":1.,"draft_temperature":0.7,"records":records,
        "full_engine_acceptance":false,"scope":if long_context {"One controlled274108token public code-rewrite frontier; one prefill and a compact exact state image reused across one128token warmup pair and two256token sampled pairs. T1/topP1, high-reasoning canonical prefix, unchanged target precision, actual-q neural/point-mass rejection. Forced neural fallback in each trial. Short opportunity screen, not completed-outcome quality."}else{"One saved public DSML code-rewrite frontier, with the same emitted anchor fixed as conditional context. T1/top_p1 and a high-reasoning canonical prefix; actual-q neural and point-mass history acceptance. One 128-token warmup and two 256-token measured pairs; one forced neural fallback per trial. Narrow opportunity screen, not completed-outcome or broad workload quality proof."}}))
}
