//! Current adaptive candidate at the deployed one-million-token capacity.
use crate::{checkpoint::Checkpoint,cuda::{Cuda,Stream},dspark_generation::GreedySession,
    layout::RankPlan,roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};

fn warm(target:&mut Target,teacher:&[u32],comm:&Roce,stream:&Stream)->Result<()> {
    for live in 1..=6 {
        target.begin_verification(&teacher[..live],comm,stream)?;
        target.verification_greedy(stream)?;
        target.finish_verification(0,comm,stream)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,core:&[u8],stream:&Stream,cp:&Checkpoint,plan:&RankPlan,
    large:&mut Target,verify:&mut Target,wides:&mut[Target],comm:&Roce,
    staged:&Path,kernels:&Path,output:&Path)->Result<Value> {
    ensure!(large.context_capacity()==1048576 && verify.adaptive_sparse_index_layers()==4
        && verify.query_overlap_layers()==40
        && std::env::var("CANADA_SCRATCH_LONG_CONTINUATION").as_deref()==Ok("1")
        && std::env::var("CANADA_SCRATCH_STOCHASTIC_DRAFT").as_deref()==Ok("1")
        && std::env::var("CANADA_SCRATCH_DRAFT_MAIN_SPLIT").as_deref()==Ok("1"),
        "long continuation requires current one-million-capacity target/split drafter");
    let full_cap=std::env::var("CANADA_SCRATCH_LONG_FULL_CAP_SCREEN").as_deref()==Ok("1");
    let history_screen=std::env::var("CANADA_SCRATCH_HISTORY_OPPORTUNITY").as_deref()==Ok("1");
    let current=std::env::var("CANADA_SCRATCH_CURRENT_CONTINUATION").as_deref()==Ok("1");
    let current_draft_temperature=if current {
        match std::env::var("CANADA_SCRATCH_DRAFT_TEMPERATURE") {
            Ok(v)=>match v.as_str(){"1"=>1.,"0.7"=>0.7,"1.2"=>1.2,
                _=>anyhow::bail!("unsupported current-continuation draft temperature")},
            Err(std::env::VarError::NotPresent)=>1.2,
            Err(e)=>return Err(e.into()),
        }
    }else{1.};
    let current_name=format!("current_block_t{current_draft_temperature}");
    let wide_budget_screen=std::env::var("CANADA_SCRATCH_WIDE_BUDGET_SCREEN").as_deref()==Ok("1");
    let index_key_splits=verify.index_key_splits();
    let encoder_profiles=wides.iter().filter_map(|t|t.wide_report()).collect::<Vec<_>>();
    if std::env::var("CANADA_SCRATCH_KEY_SPLIT_MQA").as_deref()==Ok("24") {
        ensure!(index_key_splits==[24;8] && (encoder_profiles.len()==4 || ((full_cap || history_screen || current) && encoder_profiles.len()==5) || (wide_budget_screen && encoder_profiles.len()==1))
            && encoder_profiles.iter().all(|p|p["index_key_splits"]==json!([6,6,6,12,0,0,0,0])),
            "long continuation key scheduling profiles: {:?}, {:?}",index_key_splits,encoder_profiles);
    }
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture_sha=format!("{:x}",Sha256::digest(raw));
    ensure!(fixture_sha=="107ceccbefc53247fc827cdee7eb3908f7b89a907d8c7a2d3de3ef6b95057fbb","long fixture hash");
    let fixture:Value=serde_json::from_slice(raw)?;
    let corpus:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    ensure!(corpus.len()==262656,"long fixture geometry");
    let teacher:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/fixed-verify-python-v1.json"))?;
    let teacher:Vec<u32>=serde_json::from_value(teacher["teacher_forced_tokens"].clone())?;
    let seed=std::env::var("CANADA_SCRATCH_TARGET_SEED")?.parse::<u64>()?;
    let policy_screen=std::env::var("CANADA_SCRATCH_LONG_POLICY_SCREEN").as_deref()==Ok("1");
    let policy_262k=std::env::var("CANADA_SCRATCH_LONG_POLICY_262K").as_deref()==Ok("1");
    ensure!(!full_cap || (policy_screen && policy_262k), "full cap screen requires262K policy mode");
    let head_screen=std::env::var("CANADA_SCRATCH_LONG_HEAD_SCREEN").as_deref()==Ok("1");
    ensure!(!current || (!history_screen && !head_screen && !full_cap && !wide_budget_screen && !policy_screen
        && verify.head_gemv_enabled() && verify.engram_input_kind()=="mapped" && verify.paired_engram_io_enabled()
        && encoder_profiles.iter().map(|p|p["rows"].as_u64().unwrap_or(0)).collect::<Vec<_>>()
            ==vec![8192,4096,2048,1024,512]
        && std::env::var("CANADA_SCRATCH_HISTORY_TABLE_SLOTS").as_deref()==Ok("262144")
        && std::env::var("CANADA_SCRATCH_ENGRAM_PREFETCH").as_deref()==Ok("1")
        && std::env::var("CANADA_SCRATCH_ENGRAM_PREFETCH_BATCH_ROWS").as_deref()==Ok("4096")),
        "current continuation requires the complete carried candidate");
    ensure!(!history_screen || (!head_screen && !full_cap && !wide_budget_screen && !policy_screen
        && verify.engram_input_kind()=="mapped" && verify.paired_engram_io_enabled()
        && encoder_profiles.len()==5), "history opportunity requires the current complete candidate");
    ensure!(!head_screen || (!policy_screen && verify.head_gemv_enabled()),
        "head screen requires selected native GEMV in the verifier");
    ensure!(!policy_262k || (policy_screen && verify.head_gemv_enabled()),
        "262K policy screen requires the accumulated native head candidate");
    // A small policy experiment, not another complete comparison panel. Keep
    // the original fixture index/seeds for128K and rotate the three variants.
    ensure!(!wide_budget_screen || (!head_screen && !policy_screen && verify.head_gemv_enabled()
        && encoder_profiles.len()==1 && [json!(4096),json!(8192)].contains(&encoder_profiles[0]["rows"])),
        "wide budget screen needs exactly one explicit encoder width");
    let contexts:&[(usize,usize)]=if head_screen || policy_262k || wide_budget_screen || history_screen {&[(2,262144)]}
        else if policy_screen {&[(1,131072)]}
        else {&[(0,65520),(1,131072),(2,262144)]};
    let outputs=if head_screen || wide_budget_screen {128}else if full_cap {192}else if policy_screen {256}else{512};
    let variants=if full_cap {2}else if policy_screen {3}else{1};
    let repeats=if full_cap || history_screen {2}else{3};
    let names=if current {vec![current_name.as_str()]}
        else if history_screen {vec!["block_t1_history"]}
        else if full_cap {vec!["token_t1_causal3", "token_t1_full_causal_0.1"]}
        else {vec!["token_t1","block_t1",if policy_262k {"token_t0.4"}else{"block_t0.7"}]};
    let mut barrier_in=cuda.allocate::<f32>(4)?;barrier_in.upload(&[0.;4])?;
    let mut barrier_out=cuda.allocate::<f32>(16)?;
    large.enable_cached_extend_floor()?;
    let mut session:Option<GreedySession>=None;let mut lineage=150000;let mut records=Vec::new();
    let mut compact_state_image_screen=None;
    for &(prompt_index,length) in contexts {
        let prompt=&corpus[..length];
        for repeat in 0..repeats {
          for slot in 0..variants {
            let variant=(repeat+slot)%variants;
            let block=current || history_screen || (policy_screen && !full_cap && (variant==1 || (!policy_262k && variant==2)));
            let draft_temperature=if current {current_draft_temperature}else if policy_screen && variant==2 {
                if policy_262k {0.4}else{0.7}
            }else{1.};
            let request_seed=seed.wrapping_add((prompt_index*100+repeat)as u64);
            let begin=Instant::now();large.prefill_wide(wides,prompt,comm,stream)?;
            let mut anchor=large.sample_token(request_seed,stream)?;
            let prefill_seconds=begin.elapsed().as_secs_f64();
            let prefill_state=if wide_budget_screen {
                let logits=large.last_prompt_logits()?;
                ensure!(logits.iter().all(|x|x.is_finite()), "wide prefill finite logits");
                let bytes:Vec<u8>=logits.iter().flat_map(|x|x.to_le_bytes()).collect();
                let filename=format!("prefill-p{prompt_index}-r{repeat}-logits.bin");
                std::fs::write(output.join(&filename),&bytes)?;
                Some(json!({"logits_file":filename,"logits_sha256":format!("{:x}",Sha256::digest(&bytes)),
                    "logit_values":logits.len(),"position":large.position(),
                    "encoder":large.encoder_tail_digest()?,"carry":large.carry_digests()?,
                    "phases":large.append_metrics()?}))
            }else{None};
            let setup=Instant::now();
            if let Some(s)=session.as_mut(){s.reset_from_prefill(large,cuda,core,stream)?;}
            else{session=Some(GreedySession::from_prefill(large,cuda,core,stream,cp,plan,staged,kernels)?);}
            let s=session.as_mut().unwrap();s.configure_sampling(request_seed,None)?;
            if policy_screen || history_screen || current{s.configure_algorithm_for_probe(block,draft_temperature,stream)?;}
            s.set_causal_three_threshold_for_probe(Some(0.20))?;
            if full_cap && variant == 1 { s.set_full_causal_prefix_for_probe(0.1)?; }
            s.configure_history_lookup(large,prompt)?;
            ensure!(s.conditioning_split()&&s.block_verification()==block
                &&s.draft_temperature()==draft_temperature,"requested draft policy");
            verify.handoff_from(large)?;
            let setup_seconds=setup.elapsed().as_secs_f64();
            if repeat==0{warm(verify,&teacher,comm,stream)?;}
            stream.join_default()?;
            unsafe{comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)?;}
            stream.synchronize()?;comm.health()?;
            let captures=verify.capture_statistics().1;
            let begin=Instant::now();let mut emitted=vec![anchor];let mut steps=Vec::new();
            let mut history_comparisons=Vec::new();
            let mut crossed_warm=length>=65536;
            while emitted.len()<outputs {
                if repeat==0&&!crossed_warm&&verify.position()>=65536 {
                    warm(verify,&teacher,comm,stream)?;crossed_warm=true;
                }
                let position=verify.position();
                if history_screen && repeat==0 && history_comparisons.len()<12 && outputs-emitted.len()>=6 {
                    if let Some(probe)=s.probe_history_candidates(verify,anchor,history_comparisons.len(),core,comm,stream)? {
                        history_comparisons.push(probe);
                    }
                }
                let step=s.step(verify,anchor,(outputs-emitted.len()).min(6),comm,stream)?;
                ensure!(step.tokens[0]==anchor,"pending anchor mismatch");
                ensure!(verify.active_sparse_index_layers()==if position>=65536{4}else{0},"wrong adaptive mode");
                emitted.extend_from_slice(&step.tokens[1..]);emitted.push(step.next_anchor);
                anchor=step.next_anchor;steps.push(step);
            }
            let decode_seconds=begin.elapsed().as_secs_f64();
            let new_captures=verify.capture_statistics().1-captures;
            ensure!(repeat==0||new_captures==0,"measured continuation recaptured");
            ensure!(emitted.len()==outputs&&verify.position()==(length+outputs-1)as u64,"continuation frontier");
            let record=json!({"rank":plan.rank,"prompt_index":prompt_index,"prompt_tokens":length,
                "repeat":repeat,"warmup":repeat==0,"seed":request_seed,"output_ids":emitted,"steps":steps,
                "policy_screen":policy_screen,"policy_262k":policy_262k,"head_screen":head_screen,
                "full_causal_cap_screen":full_cap,
                "history_opportunity_screen":history_screen,"history_comparisons":history_comparisons,
                "current_continuation":current,"history_lookup":s.history_lookup_metadata(),
                "wide_budget_screen":wide_budget_screen,"prefill_state":prefill_state,
                "head_gemv":verify.head_gemv_enabled(),"variant":variant,"name":names[variant],"slot":slot,
                "block_verification":s.block_verification(),"draft_temperature":s.draft_temperature(),
                "prefill_seconds":prefill_seconds,"setup_seconds":setup_seconds,"decode_seconds":decode_seconds,
                "interval_tokens":outputs-1,"decode_tokens_per_second":(outputs-1)as f64/decode_seconds,"new_verifier_captures":new_captures,
                "draft_head":s.draft_head_kind(),"conditioning_split":s.conditioning_split(),
                "final_state":if policy_screen || head_screen || wide_budget_screen || history_screen {Some(json!({"carry":verify.carry_digests()?,
                    "encoder":verify.encoder_tail_digest()?,"pending_anchor":anchor}))}else{None},
                "target_head":"canonical_bf16_weights_fp32_output","frontier":verify.position()});
            let filename=if policy_screen {format!("policy-p{prompt_index}-r{repeat}-v{variant}.json")}
                else {format!("continuation-p{prompt_index}-r{repeat}.json")};
            std::fs::write(output.join(filename),serde_json::to_vec_pretty(&record)?)?;
            eprintln!("long continuation rank{} prompt{length} repeat{repeat} {}: {:.2}tok/s",plan.rank,names[variant],(outputs-1)as f64/decode_seconds);
            records.push(record);
            if history_screen && repeat==repeats-1
                && std::env::var("CANADA_SCRATCH_LONG_STATE_IMAGE_SCREEN").as_deref()==Ok("1") {
                compact_state_image_screen=Some(crate::state_image_probe::probe_long_boundary(cuda,core,stream,
                    large,verify,wides,comm,s,anchor,request_seed,&mut lineage,&corpus,output)?);
            }
            lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;
          }
        }
    }
    Ok(json!({"rank":plan.rank,"records":records,"fixture_sha256":fixture_sha,
        "policy_screen":policy_screen,"policy_262k":policy_262k,"head_screen":head_screen,"head_gemv":verify.head_gemv_enabled(),
        "full_causal_cap_screen":full_cap,
        "history_opportunity_screen":history_screen,
        "current_continuation":current,"configured_current_draft_temperature":current_draft_temperature,
        "staged_metadata_bytes":verify.staged_metadata_bytes(),"fused_compact_publication_layers":verify.fused_compact_publication_layers(),
        "compact_state_image_screen":compact_state_image_screen,
        "names":if policy_screen {names.to_vec()}else{vec![names[0]]},
        "wide_budget_screen":wide_budget_screen,
        "context_capacity":large.context_capacity(),"temperature":1.,"top_p":1.,"outputs_per_trial":outputs,
        "index_key_splits":index_key_splits,"encoder_profiles":encoder_profiles,
        "warmup_repeats":1,"measured_repeats":repeats-1,"memory":cuda.admission_memory()?,"full_engine_acceptance":false,
        "scope":if current {"Accumulated candidate at1Mcapacity: the frozen65520/131072/262144-token prefixes,512outputs,one warmup/two measured seeds. Block verification, configured actual-q draft temperature, targetT1/topP1, canonical BF16 target weights and FP32 logits,256K history tables, full8K encoder bank, mapped paired Engram I/O and4096-ID prefill lookahead. Internal fresh-prefill/setup/decode timings; compare SGLang streaming intervals with the boundary difference explicit. Capped continuation, not completed outcomes or quality parity."}
        else if history_screen {"One262K prompt,512outputs each: warmup with at most12 matched history-versus-DSpark GPU expectation probes, then one clean timed continuation. Current block/T1, canonical target precision, full8Kbank, mapped Engram and paired narrow I/O. Counterfactuals roll back; no proposal policy change, outcome quality or production claim."}
        else if full_cap {"Short262K generation experiment: current3-or6 causal rule versus all lengths2..6 under the deployed confidence threshold0.1. Same token rejection, target/drafter T1/topP1 and canonical precision;192outputs,one warmup/one reversed-order measured pair. The prefix threshold never uses an excluded proposal to choose length. Different valid continuations; no outcome or production claim."}
        else if wide_budget_screen {"Focused262K whole-model prefill budget screen at1Mallocated capacity. One explicit4096or8192encoder profile,128outputs,one warmup/two measured seeds. Prefill logits and retained carry/encoder hashes outside timing. Canonical precision andT1/topP1; not outcome or production qualification."}
        else if head_screen {"Focused262K native head integration screen at1Mallocated capacity,128outputs,one warmup/two measured seeds. Canonical BF16 operands and FP32 logits, target T1/topP1, actual-q rejection and current accumulated candidate. Functionality and timing observation; not a paired whole-model speedup or outcome-quality claim."}
        else if policy_262k {"Small262K acceptance screen with the accumulated head candidate: token/T1 control, block/T1 and token/T0.4 drafting,256outputs,one warmup/two order-rotated measured seeds. Only actual-q proposal/verification policy varies; target T1/topP1 and canonical precision unchanged. Different valid trajectories; no outcome-quality or production claim."}
        else if policy_screen {"Small128K policy screen at1Mallocated capacity: current token/T1, block/T1 and block/T0.7 drafting,256outputs,one warmup/two order-rotated measured seeds. Target T1/topP1 and canonical precision unchanged. Different valid trajectories; no outcome-quality or production claim."}
        else {"Current adaptive sparse candidate at1Mcapacity, actual65520/131072/262144-token prompts and512outputs. Internal fresh prefill/setup/decode timings; one warmup/two measured seeds. T1/topP1,FP8split drafter,actual-q rejection,history lookup andcausal3. Truncated continuation, not completed outcomes or quality parity. Context points do not imply a usage distribution or ceiling."}}))
}
