//! Small integer-transport integration check, followed by an optional target
//! profile. Sampling probabilities, random draws and target arithmetic are fixed.
use super::*;
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path,startup:Value)->Result<()> {
    ensure!(w.current.is_none()&&w.history.frontier()==0&&w.session_cache_budget==0
        &&!w.prompt_checkpoint&&!w.prompt_cache_logits,"readout screen starts without request caches");
    let fresh:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/source-prior-transfer-tokens-v1.json"))?;
    let saved:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/phase-frontiers-s591-v1.json"))?;
    let c=fresh["cases"].as_array().unwrap().iter().find(|c|c["id"]=="rust-incremental-frames").unwrap();
    let rust:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
    let c=saved["cases"].as_array().unwrap().iter().find(|c|c["id"]=="offline-product-ideation-answer-3").unwrap();
    let mut prose:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;prose.push(c["anchor"].as_u64().unwrap()as u32);
    let ideation_trace=std::env::var("CANADA_SCRATCH_READOUT_IDEATION_TRACE").as_deref()==Ok("1");
    let trace_frontier=std::env::var("CANADA_SCRATCH_READOUT_TRACE_FRONTIER")
        .ok().map(|x|x.parse::<u64>()).transpose()?.unwrap_or(5957);
    ensure!((5772..5964).contains(&trace_frontier),"ideation trace frontier outside finite fixture");
    ensure!(!ideation_trace||std::env::var("CANADA_SCRATCH_ENCODER_CACHE_TRACE").as_deref()==Ok("1"),
        "ideation draw capture requires finite trace admission");
    let buffered=std::env::var("CANADA_SCRATCH_READOUT_BUFFERED_TRACE").as_deref()==Ok("1");
    let layer_trace=std::env::var("CANADA_SCRATCH_READOUT_LAYER_TRACE").as_deref()==Ok("1");
    ensure!(!layer_trace||buffered,"layer trace needs buffered diagnosis");
    ensure!(!buffered||!ideation_trace,"buffered diagnosis preserves the original request sequence");
    // Every rank must advertise the same diagnostic policy in its control
    // identity. Select the recording rank here, not through differing envs.
    let root_only=std::env::var("CANADA_SCRATCH_READOUT_TRACE_ROOT_ONLY").as_deref()==Ok("1");
    ensure!(!root_only||buffered,"root-only recording requires buffered diagnosis");
    let peer_reductions=std::env::var("CANADA_SCRATCH_READOUT_TRACE_PEER_REDUCTIONS").as_deref()==Ok("1");
    ensure!(!peer_reductions||(!root_only&&buffered
        &&std::env::var("CANADA_SCRATCH_READOUT_DEEP_TRACE").as_deref()==Ok("1")),
        "peer reduction recording requires shared deep trace policy");
    let all_detail=std::env::var("CANADA_SCRATCH_READOUT_TRACE_ALL_DETAIL").as_deref()==Ok("1");
    ensure!(!all_detail||(peer_reductions
        &&std::env::var("CANADA_SCRATCH_READOUT_TRACE_PEER_DETAIL").as_deref()==Ok("1")),
        "all-device operator recording requires detailed peer capture");
    let recorder=if buffered&&(rank==0
        ||(!root_only&&std::env::var("CANADA_SCRATCH_READOUT_DEEP_TRACE").as_deref()==Ok("1"))) {
        Some(std::rc::Rc::new(std::cell::RefCell::new(crate::draw_trace::DrawTrace::new(&w.cuda,&w.core,
            (rank!=0||all_detail)&&peer_reductions)?)))
    }else{None};
    let mut records=Vec::new();let mut id=1;
    let mut progress=std::fs::OpenOptions::new().create_new(true).write(true).open(out.join("sampler-readout-records.jsonl"))?;
    for (ordinal,(name,prompt))in [("rust-reasoning",rust),("ideation-answer",prose)].into_iter().enumerate() {
        if ideation_trace&&ordinal==0 {continue;}
        for repeat in if ideation_trace{vec![0]}else if ordinal==0{vec![-1,0,1]}else{vec![0,1]} {
            let warmup=repeat<0;let cap=if warmup{64}else{192};let seed=202682100+ordinal as u64;
            let mut reference=None;
            for mode in if repeat==1{[2,0]}else{[0,2]} {
                let first_arm=reference.is_none();
                let begin=w.execute(Command::Begin{images: Vec::new(),id,seed,greedy:false,limit:cap,tokens:prompt.clone()})?;id+=1;
                let starting_state=if layer_trace {Some(w.verify.repeat_state_digests(&w.stream)?)}else{None};
                let starting_draft=if layer_trace&&std::env::var("CANADA_SCRATCH_DRAFT_REPLAY_TRACE").as_deref()==Ok("1") {
                    Some(w.session.as_ref().unwrap().starting_repeat_digest(&w.stream)?)
                }else{None};
                ensure!(begin["target_temperature"]==1.&&begin["target_top_p"]==1.
                    &&begin["generation_fp32_attention"]==true&&begin["block_verification"]==true,"readout target policy fixed");
                w.session.as_mut().unwrap().set_sampler_readout_for_probe(mode,&w.stream)?;
                let trace=!buffered&&std::env::var("CANADA_SCRATCH_ENCODER_CACHE_TRACE").as_deref()==Ok("1");
                if let Some(recorder)=&recorder {
                    recorder.borrow_mut().begin(first_arm,&w.stream)?;
                    w.session.as_mut().unwrap().set_buffered_draw_trace_for_probe(recorder.clone())?;
                }
                if trace {
                    // s844 first differed at this committed frontier. Record
                    // both arms only after that draw has completed; there is
                    // no tensor read or new synchronization before the draw.
                    let path=(ordinal==1).then(||out.join(format!("readout-trace-{repeat}-{mode}")));
                    w.session.as_mut().unwrap().set_encoder_cache_trace_for_probe(path,trace_frontier,None)?;
                }
                let mut ids:Vec<u32>=serde_json::from_value(begin["output_ids"].clone())?;let mut steps=Vec::new();
                w.control.agree(b"sampler-readout-measure")?;let at=Instant::now();
                while w.emitted<w.limit&&w.history.pending()!=Some(w.cfg.eos_token_id) {
                    if trace && repeat==0 && rank==0 && (ordinal==0||ideation_trace) {
                        // s869 differed at Rust step22; s871 at ideation step11.
                        // Capture only after completed draws, on rank zero:
                        // first32 Rust or first64 ideation steps (under2GiB).
                        // These diagnostics do not change the serving path.
                        let count=if ideation_trace{64}else{32};
                        let path=(steps.len()<count).then(||out.join(format!(
                            "readout-early-{repeat}-{mode}-{:03}",steps.len())));
                        let (path,position)=if ideation_trace&&path.is_none(){
                            (Some(out.join(format!("readout-trace-{repeat}-{mode}"))),trace_frontier)
                        }else{(path,w.history.frontier()as u64)};
                        w.session.as_mut().unwrap().set_encoder_cache_trace_for_probe(path,position,None)?;
                    }
                    let s=w.execute(Command::Step)?;ids.extend(serde_json::from_value::<Vec<u32>>(s["output_ids"].clone())?);steps.push(s);
                }
                let seconds=at.elapsed().as_secs_f64();let stop=w.history.pending()==Some(w.cfg.eos_token_id);
                let finish=w.execute(Command::Finish(if stop{Finish::Stop}else{Finish::Length}))?;
                ensure!(w.history.frontier()==prompt.len()+ids.len()-1&&w.history.pending()==ids.last().copied()
                    &&ids.len()==w.emitted&&(stop||w.emitted==cap),"readout committed frontier");
                let state=json!({"carry":w.verify.carry_digests()?,"encoder":w.verify.encoder_tail_digest()?,
                    "outputs":w.verify.valid_output_digests()?,"position":w.verify.position()});
                let exact=(ids.clone(),state.clone());
                let pair_exact=reference.as_ref().is_none_or(|expected| &exact==expected);
                let captured=if !first_arm {
                    recorder.as_ref().map(|r|r.borrow().finish_pair(
                        &out.join(format!("buffered-draw-{ordinal}-{repeat}")),
                        if repeat==1{[2,0]}else{[0,2]},&w.stream)).transpose()?
                }else{None};
                if reference.is_none(){reference=Some(exact);}
                let r=json!({"case":name,"repeat":repeat,"warmup":warmup,"mode":mode,"seed":seed,"cap":cap,
                    "prompt_tokens":prompt.len(),"begin":begin,"finish":finish,"output_ids":ids,"state":state,
                    "starting_state":starting_state,"starting_draft":starting_draft,
                    "decode_seconds":seconds,"steps":steps,"pair_exact":pair_exact,
                    "trace_frontier":if trace{Some(trace_frontier)}else{None},
                    "buffered_trace":buffered,"buffered_capture":captured,
                    "early_draw_trace_steps":if trace&&repeat==0&&rank==0&&(ordinal==0||ideation_trace){
                        if ideation_trace{64}else{32}
                    }else{0},
                    "block_decision_mirror":std::env::var("CANADA_SCRATCH_BLOCK_DECIDE_ROWS_VALIDATE").as_deref()==Ok("1")});
                writeln!(progress,"{}",r)?;progress.flush()?;records.push(r);
                // Let rank zero persist the selected GPU pair before another
                // participant exits on the known mismatch and cleanup begins.
                if buffered {w.control.agree(b"buffered-readout-pair-persisted")?;}
                ensure!(pair_exact,"readout changed output or retained state; both records saved");
            }
        }
    }
    let profile=if std::env::var("CANADA_SCRATCH_POST_READOUT_VERIFY_PROFILE").as_deref()==Ok("1") {
        let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
        let tokens:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;let context=131072;
        let begin=w.execute(Command::Begin{images: Vec::new(),id,seed:202682199,greedy:false,limit:1,tokens:tokens[..context].to_vec()})?;
        let mut input=tokens[context..context+6].to_vec();input[0]=w.history.pending().unwrap();
        let before=(w.verify.carry_digests()?,w.verify.encoder_tail_digest()?);
        w.verify.begin_verification(&input,&w.comm,&w.stream)?;w.verify.verification_greedy(&w.stream)?;
        w.verify.finish_verification(0,&w.comm,&w.stream)?;
        ensure!(w.verify.verification_graph_is_warm(6),"profile graph must be warm");
        w.stream.synchronize()?;w.control.agree(b"readout-profile-ready")?;w.cuda.profiler_start()?;
        w.control.agree(b"readout-profile-active")?;
        let at=Instant::now();w.verify.begin_verification(&input,&w.comm,&w.stream)?;w.stream.synchronize()?;
        let seconds=at.elapsed().as_secs_f64();w.cuda.profiler_stop()?;
        let metrics=serde_json::to_value(w.verify.verification_metrics()?)?;
        w.verify.finish_verification(0,&w.comm,&w.stream)?;
        ensure!(before==(w.verify.carry_digests()?,w.verify.encoder_tail_digest()?),"profile rollback changed state");
        w.execute(Command::Finish(Finish::Length))?;
        Some(json!({"prompt_tokens":context,"rows":6,"begin":begin,"seconds":seconds,"metrics":metrics,
            "scope":"One warm target-only M6 verification at131K real public corpus; zero-accept rollback. Profiling timings excluded from readout A/B."}))
    }else{None};
    w.execute(Command::Shutdown)?;
    std::fs::write(out.join("sampler-readout-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,
        "records":records,"all_pairs_exact":true,"all_frontiers_checked":true,"startup":startup,"profile":profile,
        "scope":"Two public fronts, two reversed pairs,192-token caps;one excluded64-token warmup pair. Integer readout only, fixed targetT1/P1/high and FP32 generation attention. Exact tokens and retained state; no completed-task quality claim."}))?)?;
    Ok(())
}
