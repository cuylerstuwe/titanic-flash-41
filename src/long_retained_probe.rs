//! Short retained-increment and candidate-mask screen, reusing each long prefix.
//! Model math and mask reductions stay on GPU; CPU hashes bytes and records times.
use crate::{cuda::{Cuda,Module,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,core:&Module,large:&mut Target,verify:&mut Target,
    mut key_candidate:Option<&mut Target>,wides:&mut[Target],comm:&Roce,stream:&Stream,output:&Path,rank:usize)->Result<Value>{
    ensure!(std::env::var("CANADA_SCRATCH_LONG_RETAINED_PROBE").as_deref()==Ok("1")
        && large.context_capacity()==1048576 && verify.adaptive_sparse_index_layers()==4
        && verify.query_overlap_layers()==40,"long retained probe profiles");
    let raw=include_bytes!("../tests/fixtures/long-retained-prefill-v1.json");
    let fixture_sha=format!("{:x}",Sha256::digest(raw));
    ensure!(fixture_sha=="d9a0ee01b370495bfe57654d7caef539024bba860594ef7adbf4852af3c74c77","long retained fixture hash");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    ensure!(tokens.len()==284000 && tokens.iter().all(|&t|t<129280),"long retained fixture geometry");
    let mut barrier_in=cuda.allocate::<f32>(4)?;barrier_in.upload(&[0.;4])?;
    let mut barrier_out=cuda.allocate::<f32>(16)?;
    large.enable_cached_extend_floor()?;
    let mut records=Vec::new();let mut prefills=Vec::new();let mut lineage=170000;
    let mut key_split_trials=Vec::new();
    let quick=std::env::var("CANADA_SCRATCH_GROUPED_SPARSE_QUICK").as_deref()==Ok("1");
    let single=std::env::var("CANADA_SCRATCH_LONG_RETAINED_SINGLE").as_deref()==Ok("1");
    ensure!(!single || (!quick && verify.paired_engram_io_enabled() && verify.engram_input_kind()=="mapped"),"single retained chain candidate flags");
    let encoder_profiles=wides.iter().filter_map(|t|t.wide_report()).collect::<Vec<_>>();
    let prefill_key_splits=std::env::var("CANADA_SCRATCH_KEY_SPLIT_PREFILL").as_deref()==Ok("1");
    ensure!(!prefill_key_splits || ((encoder_profiles.len()==4 || (single && encoder_profiles.len()==5)) && encoder_profiles.iter().all(|p|
        p["index_key_splits"]==json!([6,6,6,12,0,0,0,0]))),"prefill key partition schedule did not engage");
    let publisher_sharding=encoder_profiles.iter().any(|p|p["index_row_sharded_sources"]==json!([2,8,14,20]));
    let row_sharding=publisher_sharding || encoder_profiles.iter().any(|p|p["index_row_sharded_sources"]==json!([2,8,14]));
    if row_sharding {
        ensure!(!encoder_profiles.is_empty() && encoder_profiles.iter().all(|p|
            p["index_row_sharded_sources"]==if publisher_sharding {json!([2,8,14,20])}else{json!([2,8,14])}), "index row-shard experiment did not engage");
    }
    ensure!(std::env::var("CANADA_SCRATCH_ROW_SHARDED_PUBLISHER").as_deref()!=Ok("1")||publisher_sharding,
        "publisher row-shard experiment did not engage");
    let bases:&[usize]=if single {&[262144]} else if quick {&[65536,262144]} else {&[65536,131072,262144]};
    let deltas:&[usize]=if quick && row_sharding {&[32,256,1024]} else if quick {&[32,256]} else {&[32,256,1024,4096,16384]};
    for &base in bases {
        let mut prior=Vec::<Value>::new();
        for repeat in 0..if quick {1} else {2} {
            let begin=Instant::now();large.prefill_wide(wides,&tokens[..base],comm,stream)?;
            let anchor=large.greedy_token(stream)?;
            let seconds=begin.elapsed().as_secs_f64();
            prefills.push(json!({"base_context":base,"repeat":repeat,"warmup":repeat==0,
                "seconds":seconds,"anchor":anchor,"phases":large.append_metrics()?,
                "last_logit_sha256":large.last_prompt_logit_digest()?}));
            for(index,&delta)in deltas.iter().enumerate(){
                let start=large.position()as usize;let end=start+delta;
                stream.join_default()?;
                unsafe{comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)?;}
                stream.synchronize()?;comm.health()?;
                let begin=Instant::now();large.prefill_wide(wides,&tokens[start..end],comm,stream)?;
                let anchor=large.greedy_token(stream)?;
                let seconds=begin.elapsed().as_secs_f64();
                ensure!(large.position()as usize==end,"retained increment frontier");
                let phases=serde_json::to_value(large.append_metrics()?)?;
                let last_logit=large.last_prompt_logit_digest()?;
                let mask=large.tail_mask_statistics_for_probe(core,stream)?;
                let carry=large.carry_digests()?;let encoder=large.encoder_tail_digest()?;
                verify.handoff_from(large)?;
                verify.begin_verification(&tokens[end..end+6],comm,stream)?;
                let predictions=verify.verification_greedy(stream)?;
                let (logits,live)=verify.pending_logits()?;ensure!(live==6,"retained continuation geometry");
                let bytes=unsafe{cuda.read_device(logits.ptr(),6*129280*4)}?;
                let following=format!("{:x}",Sha256::digest(bytes));
                verify.finish_verification(0,comm,stream)?;
                let rollback_exact=verify.position()as usize==end
                    && verify.carry_digests()?==carry && verify.encoder_tail_digest()?==encoder;
                ensure!(rollback_exact,"retained continuation changed accepted state");
                let signature=json!({"anchor":anchor,"last_logit_sha256":last_logit,
                    "following_logits_sha256":following,"predictions":predictions,
                    "carry":carry,"encoder":encoder,"mask":mask});
                let repeat_exact=repeat==0 || signature==prior[index];
                if repeat==0{prior.push(signature.clone());}
                let record=json!({"base_context":base,"repeat":repeat,"warmup":repeat==0,"increment_index":index,
                    "incoming_tokens":start,"uncached_tokens":delta,"frontier":end,"seconds":seconds,
                    "phases":phases,"signature":signature,"repeat_exact":repeat_exact,"rollback_exact":rollback_exact});
                std::fs::write(output.join(format!("retained-b{base}-r{repeat}-d{delta}.json")),serde_json::to_vec_pretty(&record)?)?;
                eprintln!("retained rank{rank} base{base} repeat{repeat} incoming{start} delta{delta}: {seconds:.4}s repeat_exact={repeat_exact}");
                if delta==1024 && let Some(other)=key_candidate.as_deref_mut() {
                    let mut trials=Vec::new();
                    for iteration in 0..4 {
                        for slot in 0..2 {
                            let candidate=(iteration+slot)%2==1;
                            if candidate {other.handoff_from(verify)?;}
                            let t=if candidate {&mut *other}else{&mut *verify};
                            stream.join_default()?;
                            unsafe{comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)?;}
                            stream.synchronize()?;comm.health()?;
                            let begin=Instant::now();
                            t.begin_verification(&tokens[end..end+6],comm,stream)?;
                            let predictions=t.verification_greedy(stream)?;
                            let seconds=begin.elapsed().as_secs_f64();
                            let (logits,live)=t.pending_logits()?;ensure!(live==6,"key split trial rows");
                            let bytes=unsafe{cuda.read_device(logits.ptr(),6*129280*4)}?;
                            let digest=format!("{:x}",Sha256::digest(bytes));
                            ensure!(digest==following && serde_json::to_value(&predictions)?==signature["predictions"],
                                "key split changed target output");
                            t.finish_verification(0,comm,stream)?;
                            ensure!(t.position()as usize==end && t.carry_digests()?==carry
                                && t.encoder_tail_digest()?==encoder,"key split changed retained state");
                            trials.push(json!({"iteration":iteration,"warmup":iteration==0,
                                "candidate":candidate,"seconds":seconds,"all_logits_and_state_exact":true}));
                            if candidate {verify.handoff_from(other)?;}
                        }
                    }
                    let comparison=json!({"base_context":base,"frontier":end,"rows":6,
                        "control_key_splits":verify.index_key_splits(),"candidate_key_splits":other.index_key_splits(),
                        "trials":trials,"full_engine_acceptance":false});
                    std::fs::write(output.join(format!("keysplit-b{base}.json")),serde_json::to_vec_pretty(&comparison)?)?;
                    key_split_trials.push(comparison);
                }
                records.push(record);large.handoff_from(verify)?;
            }
            lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;
        }
    }
    Ok(json!({"rank":rank,"quick_component_integration_screen":quick,"single_long_retained_chain":single,"paired_engram_io":verify.paired_engram_io_enabled(),"engram_staging":verify.engram_input_kind(),"index_row_sharding_enabled":row_sharding,"source20_row_sharding_enabled":publisher_sharding,"context_capacity":large.context_capacity(),"fixture_sha256":fixture_sha,
        "prefills":prefills,"records":records,"encoder_profiles":encoder_profiles,"memory":cuda.admission_memory()?,
        "key_split_trials":key_split_trials,
        "prefill_key_splits_enabled":prefill_key_splits,
        "all_repeat_exact":records.iter().all(|r|r["repeat_exact"]==true),"full_engine_acceptance":false,
        "scope":if single {json!("One representative262K retained chain, repeated once warm, with32/256/1024/4096/16384-token increments. Current full width bank, mapped staging and paired narrow I/O. No tail-latency or quality-parity claim.")} else if quick {json!("One short retained chain per base, with no second warm repetition. Same frozen public repository conversation. Saved-control causal signatures checked separately; timings are exploratory, not latency quantiles.")} else {fixture["scope"].clone()},"timing_boundary":"Internal prefill through last-row greedy readback; byte hashes, GPU mask statistics and six-position verifier/rollback checks outside timing. Phase vectors retain per-chunk host and completion intervals. No HTTP, outcome or matched SGLang claim."}))
}
