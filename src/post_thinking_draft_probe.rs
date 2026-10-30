//! Actual-q temperature opportunity at four saved public tool-call frontiers.
use crate::{checkpoint::Checkpoint,cuda::{Cuda,Stream},dspark_generation::GreedySession,
    layout::RankPlan,roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde::Deserialize;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::path::Path;
#[derive(Deserialize)]
struct Case {name:String,prompt_ids:Vec<u32>,anchor:u32,proposal_ids:Vec<u32>}
#[derive(Deserialize)]
struct Fixture {schema:String,cases:Vec<Case>}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,core:&[u8],stream:&Stream,cp:&Checkpoint,plan:&RankPlan,
    large:&mut Target,verify:&mut Target,wides:&mut[Target],comm:&Roce,
    staged:&Path,kernels:&Path)->Result<Value> {
    ensure!(std::env::var("CANADA_SCRATCH_POST_THINKING_DRAFT_SCREEN").as_deref()==Ok("1")
        && std::env::var("CANADA_SCRATCH_STOCHASTIC_DRAFT").as_deref()==Ok("1")
        && large.position()==0 && large.context_capacity()==1048576 && verify.physical_rows()==6,
        "isolated post-thinking draft screen");
    if std::env::var("CANADA_SCRATCH_NATIVE_PREFIX_STOP_SCREEN").as_deref()==Ok("1")
        || std::env::var("CANADA_SCRATCH_CONDITIONING_GRAPH_NATIVE_SCREEN").as_deref()==Ok("1") {
        let mut result=native_prefix_stop(cuda,core,stream,cp,plan,large,verify,wides,comm,staged,kernels)?;
        if std::env::var("CANADA_SCRATCH_CONDITIONING_CALIBRATION_BUNDLE").as_deref()==Ok("1") {
            ensure!(std::env::var("CANADA_SCRATCH_CONDITIONING_GRAPH_NATIVE_SCREEN").as_deref()==Ok("1")
                &&large.position()==0,"calibration bundle needs a completed conditioning comparison");
            // Publish the completed independent comparison before the next
            // experiment, so a later failure cannot discard its observations.
            use std::io::Write;
            let path=std::path::PathBuf::from(std::env::var("CANADA_SCRATCH_CONDITIONING_NATIVE_REPORT")?);
            ensure!(!path.exists(),"immutable conditioning checkpoint already exists");
            let temporary=path.with_extension("tmp");
            let mut file=std::fs::OpenOptions::new().write(true).create_new(true).open(&temporary)?;
            file.write_all(&serde_json::to_vec_pretty(&result)?)?;file.sync_all()?;drop(file);
            std::fs::rename(temporary,path)?;
            result["calibration"]=repeated_frontiers(cuda,core,stream,cp,plan,large,verify,wides,comm,staged,kernels)?;
        }
        return Ok(result);
    }
    if std::env::var("CANADA_SCRATCH_PHASE_REPEAT_DRAFT_SCREEN").as_deref()==Ok("1") {
        return repeated_frontiers(cuda,core,stream,cp,plan,large,verify,wides,comm,staged,kernels);
    }
    if std::env::var("CANADA_SCRATCH_FRESH_PHASE_DRAFT_SCREEN").as_deref()==Ok("1") {
        return fresh_tasks(cuda,core,stream,cp,plan,large,verify,wides,comm,staged,kernels);
    }
    let raw=include_bytes!("../tests/fixtures/long-history-score-v1.json");
    let fixture:Fixture=serde_json::from_slice(raw)?;
    ensure!(fixture.schema=="long-history-score-v1" && fixture.cases.len()==15,"post-thinking fixture identity");
    let mut session:Option<GreedySession>=None;let mut records=Vec::new();
    large.enable_cached_extend_floor()?;
    for (ordinal,index) in [0,6,10,11].into_iter().enumerate() {
        let case=&fixture.cases[index];
        // IDs verified against the canonical1017 tokenizer. Require the latest
        // reasoning span to have ended, rather than guessing from plain text.
        let think=case.prompt_ids.iter().rposition(|&id|id==128821);
        let end=case.prompt_ids.iter().rposition(|&id|id==128822);
        ensure!(think.is_some() && end>think && (512..=16384).contains(&case.prompt_ids.len())
            && case.proposal_ids.len()==15 && case.anchor<129280,"post-thinking phase and extent");
        large.prefill_wide(wides,&case.prompt_ids,comm,stream)?;
        if let Some(s)=session.as_mut(){s.reset_from_prefill(large,cuda,core,stream)?;}
        else{session=Some(GreedySession::from_prefill(large,cuda,core,stream,cp,plan,staged,kernels)?);}
        let s=session.as_mut().unwrap();s.configure_sampling(20261080,Some(1))?;
        s.configure_algorithm_for_probe(true,0.7,stream)?;
        verify.handoff_from(large)?;
        let mut warm=vec![case.anchor];warm.extend_from_slice(&case.proposal_ids[..5]);
        verify.begin_verification(&warm,comm,stream)?;verify.finish_verification(0,comm,stream)?;
        for repeat in 0..2 {
            let seed=20261080+(ordinal*2+repeat)as u64;
            let probe=s.probe_post_thinking_temperatures(verify,case.anchor,seed,comm,stream)?;
            records.push(json!({"case":index,"name":case.name,"repeat":repeat,"prompt_tokens":case.prompt_ids.len(),
                "last_think":think,"last_end_think":end,"anchor":case.anchor,"probe":probe}));
        }
        eprintln!("post-thinking draft rank{} case{index}:two seeds, three temperatures",plan.rank);
        verify.reset(858300+ordinal as u64,comm,stream)?;large.handoff_from(verify)?;
    }
    Ok(json!({"rank":plan.rank,"post_thinking_draft_probe":true,"records":records,
        "fixture_sha256":format!("{:x}",Sha256::digest(raw)),"target_temperature":1.,"target_top_p":1.,
        "full_engine_acceptance":false,"serving_policy_changed":false,
        "scope":"Four saved public post-thinking tool-call frontiers, two actual-q draws per temperature, GPU block-acceptance expectations and exact rollback. Fixtures were selected originally for history matches and are not representative of all answers. Opportunity screen, not measured generation or outcome quality."}))
}

#[derive(Deserialize)]
struct PhaseFrontier {id:String,family:String,phase:String,input_ids:Vec<u32>,anchor:u32,warm_ids:Vec<u32>}
#[derive(Deserialize)]
struct PhaseFixture {schema:String,cases:Vec<PhaseFrontier>}
/// Three small saved frontiers, two matched12-round pairs each. No new draft
/// policy is chosen: only work for already-excluded proposals is removed.
#[allow(clippy::too_many_arguments)]
fn native_prefix_stop(cuda:&Cuda,core:&[u8],stream:&Stream,cp:&Checkpoint,plan:&RankPlan,
    large:&mut Target,verify:&mut Target,wides:&mut[Target],comm:&Roce,
    staged:&Path,kernels:&Path)->Result<Value>{
    use std::time::Instant;
    let conditioning_graph=std::env::var("CANADA_SCRATCH_CONDITIONING_GRAPH_NATIVE_SCREEN").as_deref()==Ok("1");
    let conditioning_kv_reuse=std::env::var("CANADA_SCRATCH_CONDITIONING_KV_REUSE_NATIVE_SCREEN").as_deref()==Ok("1");
    let markov_cache=std::env::var("CANADA_SCRATCH_MARKOV_CACHE_NATIVE_SCREEN").as_deref()==Ok("1");
    ensure!(!markov_cache||(conditioning_graph&&!conditioning_kv_reuse),"Markov cache needs its own paired harness");
    ensure!(!conditioning_kv_reuse||conditioning_graph,"KV reuse needs the paired conditioning harness");
    ensure!(!conditioning_graph || std::env::var("CANADA_SCRATCH_NATIVE_PREFIX_STOP_SCREEN").as_deref()!=Ok("1"),
        "native conditioning graph and prefix stop are separate comparisons");
    let configure=|s:&mut GreedySession,enabled|->Result<()>{
        if markov_cache{s.set_markov_cache_for_probe(enabled,stream)}
        else if conditioning_kv_reuse{s.set_conditioning_kv_reuse_for_probe(enabled,stream)}
        else if conditioning_graph{s.set_conditioning_graph_for_probe(enabled,stream)}
        else{s.configure_prefix_stop_for_probe(enabled,stream)}
    };
    let raw=include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json");
    ensure!(format!("{:x}",Sha256::digest(raw))=="68eb5ddddbdee3b59b2003fc4ebfea463ddbbbad691fa44a018edfe947f41096",
        "native prefix frontier identity");
    let fixture:PhaseFixture=serde_json::from_slice(raw)?;
    ensure!(fixture.schema=="phase-frontiers-s591-v1"&&fixture.cases.len()==6,"native prefix fixtures");
    let mut session:Option<GreedySession>=None;let mut cases=Vec::new();large.enable_cached_extend_floor()?;
    // Restore identities must increase in execution order, independently of
    // reversed A/B order and case index. Never reuse a previous owner lineage.
    let mut lineage=860700u64;
    let mut barrier_in=cuda.allocate::<f32>(4)?;let mut barrier_out=cuda.allocate::<f32>(16)?;barrier_in.upload(&[0.;4])?;
    for (ordinal,index)in [0,2,5].into_iter().enumerate(){
        let case=&fixture.cases[index];
        large.prefill_wide(wides,&case.input_ids,comm,stream)?;
        if let Some(s)=session.as_mut(){s.reset_from_prefill(large,cuda,core,stream)?;}
        else{session=Some(GreedySession::from_prefill(large,cuda,core,stream,cp,plan,staged,kernels)?);}
        let s=session.as_mut().unwrap();s.configure_sampling(20261200,None)?;
        s.configure_algorithm_for_probe(true,0.7,stream)?;s.set_causal_three_threshold_for_probe(Some(0.2))?;
        verify.handoff_from(large)?;
        for rows in [3,6]{verify.begin_verification(&case.warm_ids[..rows],comm,stream)?;
            verify.finish_verification(0,comm,stream)?;}
        let bytes=s.compact_state_image_bytes(verify)?.0;
        ensure!(bytes<=768*1024usize.pow(2),"prefix snapshot extent");
        let saved=s.save_state_image(verify,case.anchor,bytes,comm,stream)?;
        ensure!(saved.manifest()["target"]["compact"]==true,"prefix compact image");
        let mut warmups=Vec::new();
        for enabled in [false,true]{
            lineage+=1;
            let anchor=s.restore_state_image(verify,&saved,lineage,comm,stream)?;
            configure(s,enabled)?;
            let began=Instant::now();let _=s.step(verify,anchor,6,comm,stream)?;
            warmups.push(json!({"enabled":enabled,"first_step_seconds":began.elapsed().as_secs_f64()}));
        }
        let mut records=Vec::new();
        for repeat in 0..2{
            let mut reference:Option<Value>=None;
            for enabled in if repeat==0{[false,true]}else{[true,false]}{
                lineage+=1;
                let mut anchor=s.restore_state_image(verify,&saved,lineage,comm,stream)?;
                let seed=20261200+(ordinal*2+repeat)as u64;s.configure_sampling(seed,None)?;
                configure(s,enabled)?;
                if markov_cache{s.reset_markov_cache_counters_for_probe(stream)?;}
                stream.join_default()?;unsafe{comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)}?;stream.synchronize()?;
                let began=Instant::now();let mut emitted=vec![anchor];let mut steps=Vec::new();
                for _ in 0..12{
                    let step=s.step(verify,anchor,6,comm,stream)?;
                    ensure!(step.tokens[0]==anchor&&step.proposal_source=="dspark"
                        &&[3,6].contains(&step.verified_rows)&&[2,5].contains(&step.draft_tokens.len()),"native prefix step shape");
                    if step.draft_tokens.len()==2{ensure!(!conditioning_graph&&enabled&&step.verified_rows==3,"partial proposals used by wide verify");}
                    emitted.extend_from_slice(&step.tokens[1..]);emitted.push(step.next_anchor);anchor=step.next_anchor;steps.push(step);
                }
                let seconds=began.elapsed().as_secs_f64();
                let cache_counters=if markov_cache{Some(s.markov_cache_counters_for_probe(stream)?)}else{None};
                let end_bytes=s.compact_state_image_bytes(verify)?.0;
                let state=s.save_state_image(verify,anchor,end_bytes,comm,stream)?.digests()?;
                let semantic_steps:Vec<_>=steps.iter().map(|s|json!({"tokens":s.tokens,"next_anchor":s.next_anchor,
                    "matching":s.matching_proposals,"verified_rows":s.verified_rows})).collect();
                let signature=json!({"emitted":emitted,"state":state,"steps":semantic_steps});
                let exact=reference.as_ref().is_none_or(|control|control==&signature);
                let record=json!({"repeat":repeat,"enabled":enabled,"seed":seed,"rounds":12,
                    "cache_counters":cache_counters,
                    "seconds":seconds,"draft_ms":steps.iter().map(|s|s.phase_ms[0]).sum::<f64>(),
                    "phase_ms":(0..4).map(|i|steps.iter().map(|s|s.phase_ms[i]).sum::<f64>()).collect::<Vec<_>>(),
                    "short_steps":steps.iter().filter(|s|s.verified_rows==3).count(),
                    "produced_two_steps":steps.iter().filter(|s|s.draft_tokens.len()==2).count(),
                    "output_tokens":emitted.len(),"output_ids":emitted,"state":state,"all_exact":exact});
                if conditioning_graph {
                    use std::io::Write;
                    let path=std::path::PathBuf::from(std::env::var("CANADA_SCRATCH_CONDITIONING_NATIVE_REPORT")?)
                        .with_extension("jsonl");
                    let mut file=std::fs::OpenOptions::new().create(true).append(true).open(path)?;
                    writeln!(file,"{}",json!({"rank":plan.rank,"id":case.id,"record":record,
                        "conditioning_split":s.conditioning_split(),"signature":signature,"control":reference}))?;
                    file.sync_all()?;
                }
                if let Some(control)=&reference {
                    ensure!(exact,"native prefix changed accepted output/state: case={} repeat={repeat} enabled={enabled} emitted_equal={} steps_equal={} state_equal={} control_state={} candidate_state={}",
                        case.id,control["emitted"]==signature["emitted"],control["steps"]==signature["steps"],
                        control["state"]==signature["state"],control["state"],signature["state"]);
                }else{reference=Some(signature);}
                records.push(record);
            }
        }
        cases.push(json!({"id":case.id,"family":case.family,"phase":case.phase,"prompt_tokens":case.input_ids.len(),
            "image_bytes":bytes,"warmups":warmups,"records":records}));
        eprintln!("native prefix rank{} {}:two12-round pairs exact",plan.rank,case.id);
        lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;
    }
    Ok(json!({"rank":plan.rank,"native_prefix_stop_probe":!conditioning_graph,
        "conditioning_variant":if markov_cache{"markov_correction_cache"}else if conditioning_kv_reuse{"kv_input_reuse"}else{"cached_graph"},
        "conditioning_graph_native_probe":conditioning_graph,"cases":cases,"all_exact":true,
        "gpu_prefix_stop":std::env::var("CANADA_SCRATCH_PREFIX_DRAFT_STOP_GPU").as_deref()==Ok("1"),
        "target_temperature":1.,"target_top_p":1.,"draft_temperature":0.7,"causal_three_threshold":0.2,
        "history_lookup_enabled":false,"full_engine_acceptance":false,
        "scope":"Three saved public reasoning/answer frontiers, two paired12-round continuations each. Exact accepted tokens and complete compact target/draft state; existing causal-three/block-rejection policy. Short native integration screen, not completed outcomes or a general speed claim."}))
}
/// A bounded headroom screen: repeated draws distinguish source/phase effects
/// from one lucky proposal. No adaptive policy is selected during generation.
#[allow(clippy::too_many_arguments)]
fn repeated_frontiers(cuda:&Cuda,core:&[u8],stream:&Stream,cp:&Checkpoint,plan:&RankPlan,
    large:&mut Target,verify:&mut Target,wides:&mut[Target],comm:&Roce,
    staged:&Path,kernels:&Path)->Result<Value> {
    use std::time::Instant;
    let calibration=std::env::var("CANADA_SCRATCH_PHASE_CALIBRATION_SCREEN").as_deref()==Ok("1");
    let raw=include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json");
    ensure!(format!("{:x}",Sha256::digest(raw))=="68eb5ddddbdee3b59b2003fc4ebfea463ddbbbad691fa44a018edfe947f41096", "saved frontier byte identity");
    let fixture:PhaseFixture=serde_json::from_slice(raw)?;
    ensure!(fixture.schema=="phase-frontiers-s591-v1" && fixture.cases.len()==6,
        "saved phase-frontier fixture");
    let mut session:Option<GreedySession>=None;let mut cases=Vec::new();
    let bundled=std::env::var("CANADA_SCRATCH_CONDITIONING_CALIBRATION_BUNDLE").as_deref()==Ok("1");
    if !bundled {
        large.enable_cached_extend_floor()?;
    }
    for (ordinal,case) in fixture.cases.iter().enumerate() {
        let think=case.input_ids.iter().rposition(|&id|id==128821);
        let end=case.input_ids.iter().rposition(|&id|id==128822);
        ensure!((512..8192).contains(&case.input_ids.len()) && think.is_some()
            && (case.phase=="reasoning" && end<think || case.phase=="answer" && end>think)
            && case.warm_ids.len()==6 && case.warm_ids[0]==case.anchor,
            "saved phase identity/extent");
        let begin=Instant::now();large.prefill_wide(wides,&case.input_ids,comm,stream)?;
        let prefill_seconds=begin.elapsed().as_secs_f64();
        if let Some(s)=session.as_mut(){s.reset_from_prefill(large,cuda,core,stream)?;}
        else{session=Some(GreedySession::from_prefill(large,cuda,core,stream,cp,plan,staged,kernels)?);}
        let s=session.as_mut().unwrap();s.configure_sampling(20261110,Some(1))?;
        s.configure_algorithm_for_probe(true,0.7,stream)?;
        verify.handoff_from(large)?;
        let carry=verify.carry_digests()?;let encoder=verify.encoder_tail_digest()?;
        let mut costs=Vec::new();
        for repeat in 0..if calibration{1}else{3} {
            for rows in if repeat%2==0 {[3,6]}else{[6,3]} {
                let begin=Instant::now();verify.begin_verification(&case.warm_ids[..rows],comm,stream)?;
                stream.synchronize()?;let seconds=begin.elapsed().as_secs_f64();
                verify.finish_verification(0,comm,stream)?;
                ensure!(verify.position()==case.input_ids.len()as u64 && verify.carry_digests()?==carry
                    && verify.encoder_tail_digest()?==encoder,"phase cost rollback");
                if repeat>0 {costs.push(json!({"repeat":repeat,"rows":rows,"seconds":seconds}));}
            }
        }
        let mut probes=Vec::new();
        for repeat in 0..if calibration{1}else{4} {
            let seed=20261110+(ordinal*4+repeat)as u64;
            let probe=s.probe_post_thinking_temperatures(verify,case.anchor,seed,comm,stream)?;
            probes.push(json!({"repeat":repeat,"probe":probe}));
        }
        cases.push(json!({"id":case.id,"family":case.family,"phase":case.phase,
            "prompt_tokens":case.input_ids.len(),"anchor":case.anchor,"prefill_seconds":prefill_seconds,
            "verify_costs":costs,"probes":probes}));
        eprintln!("phase rank{} {}:{} draws, three temperatures",plan.rank,case.id,if calibration{1}else{4});
        // Bundled execution follows the native comparison's860700 lineage.
        verify.reset((if bundled{861700}else{859700})+ordinal as u64,comm,stream)?;
        large.handoff_from(verify)?;
    }
    Ok(json!({"rank":plan.rank,"phase_repeat_draft_probe":true,"phase_calibration_probe":calibration,"cases":cases,
        "fixture_sha256":format!("{:x}",Sha256::digest(raw)),"target_temperature":1.,"target_top_p":1.,
        "history_lookup_enabled":false,"serving_policy_changed":false,"full_engine_acceptance":false,
        "scope":if calibration{"Six saved frontiers, one draw at three temperatures. GPU first-proposal overlap integrates the full distribution. Reuse previous independent draws to assess whether marginal calibration predicts block acceptance; no policy or throughput claim."}else{"Six saved public s591 coding/ideation reasoning/answer frontiers, four independent actual-q proposal draws at each temperature0.7/1/1.2. GPU conditional acceptance expectations, warm3/6row cost samples and exact rollback. Small configuration-headroom screen, not a deployed classifier, representative workload, or outcome-quality/throughput result. Prefill reconstruction may differ numerically from original decode states."}}))
}

#[derive(Deserialize)]
struct FreshCase {id:String,family:String,split:String,input_ids:Vec<u32>}
#[derive(Deserialize)]
struct FreshFixture {thinking_mode:String,reasoning_effort:String,cases:Vec<FreshCase>}
/// Generate fresh prefixes instead of choosing only strong copy matches.
/// Two task families, at most two reasoning and two answer frontiers each.
#[allow(clippy::too_many_arguments)]
fn fresh_tasks(cuda:&Cuda,core:&[u8],stream:&Stream,cp:&Checkpoint,plan:&RankPlan,
    large:&mut Target,verify:&mut Target,wides:&mut[Target],comm:&Roce,
    staged:&Path,kernels:&Path)->Result<Value> {
    use std::time::Instant;
    let strength=std::env::var("CANADA_SCRATCH_MARKOV_STRENGTH_SCREEN").as_deref()==Ok("1");
    let limit=if strength{1024}else{6144};
    let raw=include_bytes!("../tests/fixtures/agentic-exploration-tokens-v1.json");
    let hash=format!("{:x}",Sha256::digest(raw));
    ensure!(hash=="2a8091bfbb57bb8d83d4542c71fb797daad4cf5f8d816fec9b10214a92a81638",
        "fresh-task canonical encoding identity");
    let fixture:FreshFixture=serde_json::from_slice(raw)?;
    ensure!(fixture.thinking_mode=="thinking" && fixture.reasoning_effort=="high",
        "fresh-task recommended reasoning template");
    let mut session:Option<GreedySession>=None;let mut cases=Vec::new();
    large.enable_cached_extend_floor()?;
    let ids:&[&str]=if strength{&["python-pagination-streaming","sqlite-migration-plan","offline-product-ideation"]}
        else{&["rust-config-parser","offline-product-ideation"]};
    for (ordinal,id) in ids.iter().copied().enumerate() {
        let case=fixture.cases.iter().find(|c|c.id==id).ok_or_else(||anyhow::anyhow!("fresh task missing"))?;
        ensure!(case.split=="exploration" && case.input_ids.contains(&128821)
            && case.input_ids.len()<512,"fresh short high-reasoning prompt");
        let seed=if strength{20261700}else{20261090}+ordinal as u64;
        let began=Instant::now();large.prefill_wide(wides,&case.input_ids,comm,stream)?;
        let mut anchor=large.sample_token(seed,stream)?;let prefill_seconds=began.elapsed().as_secs_f64();
        if let Some(s)=session.as_mut(){s.reset_from_prefill(large,cuda,core,stream)?;}
        else{session=Some(GreedySession::from_prefill(large,cuda,core,stream,cp,plan,staged,kernels)?);}
        let s=session.as_mut().unwrap();s.configure_sampling(seed,Some(1))?;
        s.configure_algorithm_for_probe(true,0.7,stream)?;s.set_causal_three_threshold_for_probe(Some(0.2))?;
        verify.handoff_from(large)?;
        let mut tokens=vec![anchor];let mut probes=Vec::new();
        let(mut reasoning_count,mut answer_count)=(0usize,0usize);
        let mut step_seconds=0.;let mut steps=0usize;let began=Instant::now();
        while tokens.len()<limit && anchor!=1 {
            let ended=tokens.iter().position(|&t|t==128822);
            if ended.is_some_and(|i|tokens.len()>i+512){break;}
            if strength&&probes.len()>=4{break;}
            let selected=if strength{verify.position()>=512+256*(probes.len()/2)as u64}else if let Some(i)=ended {
                answer_count<2 && tokens.len()-i-1>=[64,256][answer_count]
            }else{reasoning_count<2 && tokens.len()>=768*(reasoning_count+1)};
            if selected && verify.position()>=512 {
                let phase=if ended.is_some(){"answer"}else{"reasoning"};
                for _ in 0..if strength{2}else{1}{
                let probe=s.probe_post_thinking_temperatures(verify,anchor,seed+100+probes.len()as u64,comm,stream)?;
                probes.push(json!({"phase":phase,"generated_tokens":tokens.len(),"anchor":anchor,"probe":probe}));
                if ended.is_some(){answer_count+=1;}else{reasoning_count+=1;}
                eprintln!("fresh phase rank{} {id} {phase} frontier{}",plan.rank,tokens.len());
                }
            }
            let t=Instant::now();let step=s.step(verify,anchor,(limit-tokens.len()).min(6),comm,stream)?;
            step_seconds+=t.elapsed().as_secs_f64();steps+=1;
            ensure!(step.tokens[0]==anchor && step.tokens.len()<=limit-tokens.len(),"fresh output boundary");
            tokens.extend_from_slice(&step.tokens[1..]);tokens.push(step.next_anchor);anchor=step.next_anchor;
            ensure!(tokens[..tokens.len()-1].iter().all(|&id|id!=1),"fresh output after EOS");
        }
        ensure!(verify.position()==(case.input_ids.len()+tokens.len()-1)as u64,"fresh target frontier");
        let ended=tokens.iter().position(|&t|t==128822);
        let output_hash=format!("{:x}",Sha256::digest(tokens.iter().flat_map(|id|id.to_le_bytes()).collect::<Vec<_>>()));
        cases.push(json!({"id":id,"family":case.family,"split":case.split,"seed":seed,
            "prompt_tokens":case.input_ids.len(),"output_ids":tokens,"output_sha256":output_hash,
            "last_end_think":ended,"probes":probes,"reasoning_probes":reasoning_count,"answer_probes":answer_count,
            "finish_reason":if anchor==1{"stop"}else if tokens.len()>=limit{"length"}else if strength{"diagnostic_frontier_limit"}else{"answer_sample_limit"},
            "max_new_tokens":limit,"prefill_seconds":prefill_seconds,"steps":steps,
            "step_seconds":step_seconds,"generation_with_diagnostics_seconds":began.elapsed().as_secs_f64(),
            "committed_position":verify.position(),"state":{"carry":verify.carry_digests()?,"encoder":verify.encoder_tail_digest()?}}));
        eprintln!("fresh phase rank{} {id}:{} outputs, {reasoning_count}reasoning/{answer_count}answer probes",plan.rank,tokens.len());
        verify.reset(859100+ordinal as u64,comm,stream)?;large.handoff_from(verify)?;
    }
    Ok(json!({"rank":plan.rank,"fresh_phase_draft_probe":true,"markov_strength_probe":strength,"fixture_sha256":hash,"cases":cases,
        "target_temperature":1.,"target_top_p":1.,"generation_draft_temperature":0.7,"history_lookup_enabled":false,
        "serving_policy_changed":false,"full_engine_acceptance":false,
        "scope":if strength{"Three fresh public Python-debugging/SQL-migration/ideation tasks; early accepted positions512/768 with two draws each and draft correction strengths1/.75 atT.7. TargetT1/P1/high, actual-q GPU expectations and exact rollback. Normal continuation uses strength1. Capped diagnostics, not completed tasks, policy selection or throughput."}else{"Two fresh public tasks (Rust parser and product ideation), canonical high-reasoning template, T1/P1. Generate up to6144 tokens or512 answer tokens; caps are not completed outcomes. At most two reasoning and two answer frontiers per task, actual-q temperatures0.7/1/1.2 with exact rollback. Isolated neural drafting: history proposals disabled. Diagnostic wall time is not a throughput benchmark. Exploration families only; transfer tasks remain unused."}}))
}
