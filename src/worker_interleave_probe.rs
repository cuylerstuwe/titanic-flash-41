//! Bounded worker-level feasibility probe, not an HTTP scheduler. The saved
//! state includes unfinished request budgets/visibility as well as the GPU
//! model, drafter, actual proposal policy and counter-based sampling seed.
use super::*;

const BUDGET: usize = 2 * 1024usize.pow(3);
use super::request_checkpoint::{RequestCheckpoint,try_checkpoint,suspend,restore};

fn checkpoint(w: &mut Worker, other_charge: usize) -> Result<(RequestCheckpoint, Value)> {
    try_checkpoint(w,other_charge,usize::from(other_charge>0))?
        .ok_or_else(||anyhow::anyhow!("probe checkpoint admission refused"))
}

fn execute(w: &mut Worker, command: Command) -> Result<Value> {
    // The fixture is public and replicated, but commands still pass through
    // the same ordered rank-zero protocol as the ordinary worker.
    let encoded=command.encode()?;
    let wire=w.control.command((w.comm.rank()==0).then_some(encoded.as_slice()))?;
    ensure!(wire==encoded,"probe command replicas disagree");
    w.execute(Command::decode(&wire)?)
}

fn advance(w: &mut Worker, until: usize) -> Result<(Vec<Value>, f64)> {
    let began=Instant::now();let mut decisions=Vec::new();
    while w.current.is_some() && w.emitted<until {
        let command=w.active_command(false)?;
        if matches!(command,Command::Finish(_)) {break;}
        let event=execute(w,command)?;
        let keys=["request_id","output_ids","frontier","pending","verified_rows","step_limit",
            "committed_model_rows","accepted_draft_tokens","emitted_tokens","proposal_source",
            "lookup_match_tokens","history_lookup_position"];
        let mut decision=serde_json::Map::new();
        for key in keys {decision.insert(key.to_owned(),event[key].clone());}
        decisions.push(Value::Object(decision));
    }
    Ok((decisions,began.elapsed().as_secs_f64()))
}

fn finish(w: &mut Worker) -> Result<Value> {
    let command=w.active_command(false)?;
    ensure!(matches!(command,Command::Finish(_)),"probe must finish naturally or at original budget");
    let event=execute(w,command)?;
    Ok(json!({"event":event,"history_sha256":format!("{:x}",Sha256::digest(
        w.history.committed().iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())),
        "carry":w.verify.carry_digests()?,"encoder":w.verify.encoder_tail_digest()?}))
}

pub(super) fn run(w: &mut Worker,p: &WorkerConfig) -> Result<Value> {
    ensure!(p.context==1048576 && w.current.is_none() && w.session.is_none()
        && w.session_cache_budget==0 && !w.prompt_checkpoint && !w.progress_next
        && std::env::var("CANADA_SCRATCH_COMPACT_STATE_IMAGE").as_deref()==Ok("1")
        && std::env::var("CANADA_SCRATCH_DRAFT_TEMPERATURE").as_deref()==Ok("0.7"),
        "worker interleave probe requires isolated current candidate");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let corpus:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    ensure!(corpus.len()==262656,"worker probe long fixture geometry");
    let child:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/target-agentic-rust-v1.json"))?;
    let child:Vec<u32>=serde_json::from_value(child["batches"][0].clone())?;
    let mut swaps=Vec::new();
    let parent_begin=execute(w,Command::Begin{images: Vec::new(),id:1,seed:20261051,greedy:false,limit:256,
        tokens:corpus[..131072].to_vec()})?;
    let (parent_prefix,parent_prefix_seconds)=advance(w,64)?;
    ensure!(w.emitted>=64 && w.history.step_limit()?>0,"parent ended before the pause boundary");
    let (parent,receipt)=checkpoint(w,0)?;swaps.push(receipt);
    let (parent_control,parent_control_seconds)=advance(w,usize::MAX)?;
    let parent_final=finish(w)?;
    swaps.push(restore(w,&parent)?);swaps.push(suspend(w)?);
    let child_begin=execute(w,Command::Begin{images: Vec::new(),id:2,seed:20261052,greedy:false,limit:96,tokens:child})?;
    let (child_prefix,child_prefix_seconds)=advance(w,32)?;
    ensure!(w.emitted>=32 && w.history.step_limit()?>0,"child ended before the pause boundary");
    let (child,receipt)=checkpoint(w,parent.charge)?;swaps.push(receipt);
    let (child_control,child_control_seconds)=advance(w,usize::MAX)?;
    let child_final=finish(w)?;
    swaps.push(restore(w,&parent)?);
    let boundary=w.emitted+32;
    let (mut parent_resumed,parent_first_seconds)=advance(w,boundary)?;
    ensure!(w.emitted>=boundary && w.history.step_limit()?>0,"resumed parent ended before second pause");
    drop(parent);
    let (parent,receipt)=checkpoint(w,child.charge)?;swaps.push(receipt);
    swaps.push(suspend(w)?);
    swaps.push(restore(w,&child)?);
    let (child_resumed,child_resumed_seconds)=advance(w,usize::MAX)?;
    let child_resumed_final=finish(w)?;
    ensure!(child_control==child_resumed && child_final==child_resumed_final,
        "child resume changed decisions, output, budget or final model state");
    drop(child);
    swaps.push(restore(w,&parent)?);
    let (remaining,parent_remaining_seconds)=advance(w,usize::MAX)?;
    parent_resumed.extend(remaining);
    let parent_resumed_final=finish(w)?;
    ensure!(parent_control==parent_resumed && parent_final==parent_resumed_final,
        "parent resume changed decisions, output, budget or final model state");
    drop(parent);
    let shutdown=execute(w,Command::Shutdown)?;
    let result=json!({"type":"worker_interleave_probe","rank":p.rank,"all_exact":true,
        "context_capacity":p.context,"parent_prompt_tokens":131072,
        "target_temperature":1.,"target_top_p":1.,"draft_temperature":0.7,
        "checkpoint_budget_bytes":BUDGET,"swaps":swaps,
        "parent":{"begin":parent_begin,"prefix":parent_prefix,"control":parent_control,
            "resumed":parent_resumed,"final":parent_final,"resumed_final":parent_resumed_final,
            "prefix_seconds":parent_prefix_seconds,"control_seconds":parent_control_seconds,
            "resumed_seconds":parent_first_seconds+parent_remaining_seconds},
        "child":{"begin":child_begin,"prefix":child_prefix,"control":child_control,
            "resumed":child_resumed,"final":child_final,"resumed_final":child_resumed_final,
            "prefix_seconds":child_prefix_seconds,"control_seconds":child_control_seconds,
            "resumed_seconds":child_resumed_seconds},"shutdown":shutdown,
        "production_qualified":false,"http_interleaving_enabled":false,
        "scope":"Two unfinished sampled worker requests,131K parent and short unrelated child. Original256/96-token budgets, no client finish on suspension, at most two compact images sharing2GiB plus8GiB UMA reserve. Alternate twice and compare every continuation token/step decision, final budget/history/carry/encoder to uninterrupted controls. Existing graph and model math unchanged. Small feasibility screen; HTTP scheduling, cancellation, fairness and concurrency remain unqualified."});
    let output=p.events.parent().unwrap().join("worker-interleave.json");
    let mut file=std::fs::OpenOptions::new().write(true).create_new(true).open(output)?;
    serde_json::to_writer_pretty(&mut file,&result)?;file.write_all(b"\n")?;
    Ok(json!({"type":"worker_interleave_probe","rank":p.rank,"all_exact":true,
        "http_interleaving_enabled":false,"production_qualified":false}))
}
