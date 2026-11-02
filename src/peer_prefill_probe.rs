//! Fixed-input proof for resumable foreground prefill. HTTP fairness is separate.
use super::*;
use super::history32_generation_probe::execute;

fn start(w:&mut Worker,c:Command,yielding:bool)->Result<Value> {
    let encoded=c.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(encoded.as_slice()))?;
    ensure!(wire==encoded,"peer prefill probe command agreement");
    let root=w.comm.rank()==0;
    w.execute_with_prefill_cancel(Command::decode(&wire)?,&mut |control,_,_,_|{
        let code=[if yielding{2u8}else{0}];
        let wire=control.command(root.then_some(code.as_slice()))?;
        ensure!(wire==code,"peer prefill initial yield agreement");
        Ok(if yielding{PrefillDirective::Yield}else{PrefillDirective::Continue})
    })
}
fn signature(w:&mut Worker,teacher:&[u32],event:&Value,raw:Option<&Path>)->Result<Value> {
    let carry=w.verify.carry_digests()?;let encoder=w.verify.encoder_tail_digest()?;
    let draft=w.session.as_mut().unwrap().conditioning_state_digest_for_probe(&w.comm,&w.stream)?;
    let history=w.session.as_ref().unwrap().history_lookup_digest_for_probe()?;
    w.verify.begin_verification(teacher,&w.comm,&w.stream)?;
    let(y,n)=w.verify.pending_logits()?;ensure!(n==6,"peer prefill following extent");
    let bytes=unsafe{w.cuda.read_device(y.ptr(),n*129280*4)}?;
    let logits=format!("{:x}",Sha256::digest(&bytes));
    if let Some(path)=raw {std::fs::write(path,&bytes)?;}
    w.verify.finish_verification(0,&w.comm,&w.stream)?;
    ensure!(w.verify.carry_digests()?==carry&&w.verify.encoder_tail_digest()?==encoder,
        "peer prefill readout changed retained state");
    Ok(json!({"anchor":event["output_ids"],"frontier":w.history.frontier(),"carry":carry,
        "encoder":encoder,"draft":draft,"history":history,"following_logits":logits}))
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_PEER_PREFILL_SCREEN").as_deref()==Ok("1")
        &&w.peer_prefill_enabled&&w.current.is_none()&&w.encoder_prompt_checkpoint,
        "finite peer prefill screen admission");
    if std::env::var("CANADA_SCRATCH_PEER_PREFILL_EDGE_SCREEN").as_deref()==Ok("1") {
        return run_edges(w,rank,out);
    }
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let teacher=&tokens[250000..250006];let mut records=Vec::new();let mut id=7_940_000;
    let numerical_diagnostic=std::env::var("CANADA_SCRATCH_PEER_PREFILL_NUMERICS").as_deref()==Ok("1");
    w.encoder_active_frontier=false;w.encoder_upgrade_reconstructed=false;w.prompt_cache_logits=false;
    // Preserve the normal foreground cache path for both controls. Only an
    // actual encoder interruption creates the resumable owner.
    for cached in [false,true] {
        w.cached_sessions.clear();
        let parent=if cached {
            id+=1;let event=execute(w,Command::Begin{images:Vec::new(),id,seed:20261031,
                greedy:false,limit:1,tokens:tokens[..65792].to_vec()})?;
            let parent=event["prompt_checkpoint"]["prefix_tokens"].as_u64().context("peer prefill parent")?as usize;
            ensure!(event["prompt_checkpoint"]["saved"]==true,"peer prefill parent not saved");
            execute(w,Command::Finish(Finish::Length))?;parent
        }else{0};
        let prompt=if cached{[&tokens[..parent],&tokens[200000..216641]].concat()}
            else{tokens[..33024].to_vec()};
        let mut expected=None;
        let variants=if numerical_diagnostic {vec![("encoder",false,false),("complete",false,true),("yield",true,false)]}
            else{vec![("encoder",false,false),("yield",true,false)]};
        for (variant,yielding,complete_checkpoint) in variants {
            // An encoder-only parent requires the compound restore path. Its
            // unsupported complete-checkpoint control is not a serving arm.
            if cached&&complete_checkpoint {continue;}
            w.cached_sessions.retain(|c|cached&&c.prefix==tokens[..parent]);
            // Control for the existing complete-checkpoint numerical path.
            // This is a diagnostic arm, never a serving policy change.
            w.encoder_prompt_checkpoint=!complete_checkpoint;
            let traced=!cached&&numerical_diagnostic
                &&std::env::var("CANADA_SCRATCH_PEER_PREFILL_TRACE").as_deref()==Ok("1");
            if traced{crate::target::begin_prefill_boundary_trace_for_probe()?;}
            id+=1;let mut event=start(w,Command::Begin{images:Vec::new(),id,seed:20261031,
                greedy:false,limit:1,tokens:prompt.clone()},yielding)?;
            let first=event.clone();let mut frontiers=Vec::new();
            if yielding {
                ensure!(event["type"]=="prefill_progress"&&event["cancelled"]==false
                    &&w.pending_prefill.is_some()&&!w.prefill_cancelled,"foreground yield lost ownership");
            }
            while w.pending_prefill.is_some() {
                let previous=w.history.frontier();frontiers.push(previous);
                let root=rank==0;
                event=w.continue_prefill(&mut |control,_,_|{
                    let wire=control.command(root.then_some(&[1u8][..]))?;
                    ensure!(wire==[1],"peer prefill resume agreement");Ok(true)
                })?;
                ensure!(w.history.frontier()>previous&&frontiers.len()<16,"resumable prefill made no progress");
            }
            ensure!(event["type"]=="tokens"&&event["cached_tokens"]==parent
                &&w.history.frontier()==prompt.len(),"peer prefill final extent or cached count");
            let raw=out.join(format!("peer-prefill-c{}-{variant}-logits.bin",u8::from(cached)));
            let actual=signature(w,teacher,&event,(numerical_diagnostic&&rank==0).then_some(raw.as_path()))?;
            let trace=if traced{crate::target::take_prefill_boundary_trace_for_probe()?}else{Vec::new()};
            let exact=expected.as_ref().is_none_or(|v|v==&actual);
            if expected.is_none(){expected=Some(actual.clone());}
            let record=json!({"cached":cached,"variant":variant,"yielding":yielding,"first":first,"completed":event,
                "frontiers":frontiers,"signature":actual,"trace":trace,"exact":exact});
            std::fs::write(out.join(format!("peer-prefill-c{}-{variant}.json",u8::from(cached))),
                serde_json::to_vec_pretty(&record)?)?;
            let all_exact=w.control.all_admit(exact)?;
            ensure!(numerical_diagnostic||all_exact,"foreground yield changed fixed-prefix logits or retained state");
            w.encoder_prompt_checkpoint=true;
            records.push(record);execute(w,Command::Finish(Finish::Length))?;
        }
    }
    // Cancellation after a committed partial prompt must release the owner;
    // the next fresh request must not inherit a continuation or sampled token.
    w.cached_sessions.clear();id+=1;
    let interrupted=start(w,Command::Begin{images:Vec::new(),id,seed:17,greedy:false,
        limit:1,tokens:tokens[131072..164096].to_vec()},true)?;
    ensure!(interrupted["type"]=="prefill_progress","cancel probe did not yield");
    execute(w,Command::Finish(Finish::Cancelled))?;
    ensure!(w.current.is_none()&&w.pending_prefill.is_none()&&!w.prefill_cancelled,"partial cancel retained owner");
    id+=1;let recovery=execute(w,Command::Begin{images:Vec::new(),id,seed:18,greedy:false,
        limit:1,tokens:tokens[200000..200141].to_vec()})?;
    ensure!(recovery["type"]=="tokens","partial cancellation recovery");
    execute(w,Command::Finish(Finish::Length))?;execute(w,Command::Shutdown)?;
    std::fs::write(out.join("peer-prefill-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"all_exact":records.iter().all(|r|r["exact"]==true),
        "numerical_diagnostic":numerical_diagnostic,"records":records,"cancel_recovery":true,
        "memory":w.cuda.admission_memory()?,"scope":"Same-load fresh33K and cached16K append. Ordinary foreground versus forced encoder-boundary yields; complete target/draft/history, full following M6 logits and sampled anchor. Partial cancellation and fresh recovery. Scheduler fairness, image-bearing prompts and2x1M remain separate."}))?)?;Ok(())
}

/// Different continuation segments than the initial aligned-prefix screen:
/// a ragged final wide chunk, active-frontier suffix, and two cache checkpoints.
fn run_edges(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_PEER_PREFILL_NUMERICS").as_deref()!=Ok("1"),
        "prefill edge screen requires strict agreement");
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let teacher=&tokens[250000..250006];let mut id=7_950_100;let mut records=Vec::new();
    w.encoder_upgrade_reconstructed=false;w.prompt_cache_logits=false;
    for case in ["ragged-final","active-frontier","two-checkpoints"] {
        let active=case=="active-frontier";
        let prompt=match case {"ragged-final"=>tokens[..8257].to_vec(),
            "active-frontier"=>[&tokens[..16384],&tokens[200000..208257]].concat(),
            _=>tokens[..33024].to_vec()};
        let mut expected=None;
        for yielding in [false,true] {
            w.cached_sessions.clear();w.prompt_checkpoint=true;
            w.encoder_prompt_checkpoint=true;w.encoder_active_frontier=active;
            w.prompt_checkpoint_tail=128;
            // Both arms must build the parent from the same fresh frontier;
            // otherwise the previous case can supply an unintended active hit.
            id+=1;execute(w,Command::Begin{images:Vec::new(),id,seed:19,
                greedy:false,limit:1,tokens:vec![(tokens[0]+1)%129280]})?;
            execute(w,Command::Finish(Finish::Length))?;w.cached_sessions.clear();
            if active {
                id+=1;execute(w,Command::Begin{images:Vec::new(),id,seed:19,
                    greedy:false,limit:1,tokens:tokens[..16384].to_vec()})?;
                execute(w,Command::Finish(Finish::Length))?;
                w.cached_sessions.clear();
            }
            if case=="two-checkpoints" {
                w.encoder_prompt_checkpoint=false;w.prompt_checkpoint_tail=1;
            }
            id+=1;let mut event=start(w,Command::Begin{images:Vec::new(),id,seed:20261031,
                greedy:false,limit:1,tokens:prompt.clone()},yielding)?;
            let first=event.clone();let mut frontiers=Vec::new();
            if active {ensure!(first["retained_session_cache"]["active_reuse"]==true,
                "active-frontier edge failed to use live state");}
            if yielding {ensure!(w.pending_prefill.is_some(),"edge prefill did not yield");}
            while w.pending_prefill.is_some() {
                let previous=w.history.frontier();frontiers.push(previous);
                event=w.continue_prefill(&mut |control,_,_|{
                    let wire=control.command((rank==0).then_some(&[1u8][..]))?;
                    ensure!(wire==[1],"edge resume agreement");Ok(true)
                })?;
                ensure!(w.history.frontier()>previous&&frontiers.len()<16,"edge resume made no progress");
            }
            ensure!(event["type"]=="tokens"&&w.history.frontier()==prompt.len(),"edge prefill extent");
            if yielding&&case!="two-checkpoints" {
                ensure!(frontiers.last()==Some(&(prompt.len()-65)),"ragged final chunk was not resumed");
            }
            let actual=signature(w,teacher,&event,None)?;
            let exact=expected.as_ref().is_none_or(|v|v==&actual);
            if expected.is_none(){expected=Some(actual.clone());}
            let record=json!({"case":case,"yielding":yielding,"first":first,"completed":event,
                "frontiers":frontiers,"signature":actual,"exact":exact});
            std::fs::write(out.join(format!("prefill-edge-{case}-{yielding}.json")),serde_json::to_vec_pretty(&record)?)?;
            ensure!(w.control.all_admit(exact)?,"edge prefill changed logits or retained state");
            records.push(record);execute(w,Command::Finish(Finish::Length))?;
        }
    }
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("peer-prefill-edges.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"all_exact":true,"scope":"Three directly affected schedule edges: fresh8257, active16Kplus8257, and fresh33Kwith two checkpoint boundaries. Forced every-boundary pause versus uninterrupted. Full followingM6logits and target/drafter/history exact."}))?)?;
    Ok(())
}
