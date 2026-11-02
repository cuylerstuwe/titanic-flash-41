//! Same-prefix comparison at the real context limit. A proposal is drawn once;
//! ordinary target execution is rolled back, then paired execution verifies the
//! exact same proposal. This does not require two independent prefills to pick
//! identical sparse-index tie identities or generate identical continuations.
use super::*;
use super::pair_endpoints::{endpoint,load_prompts};
use crate::dspark_generation::GreedySession;

fn committed(target:&Target,stream:&Stream,full:bool)->Result<Value> {
    Ok(json!({"position":target.position(),"carry":target.carry_digests()?,
        "encoder":target.encoder_tail_digest()?,
        "published":if full {target.published_state_digests_for_probe(stream)?}else{Value::Null}}))
}
fn local_logits(target:&Target,cuda:&Cuda)->Result<Vec<u8>> {
    let (values,live)=target.pending_local_logits_for_probe()?;
    unsafe{cuda.read_device(values.ptr(),live*32320*4)}
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(w:&mut Worker,other:&mut Target,verify:&mut Target,wides:&mut [Target],
    pooled:crate::ffn::Paired,first:PreparedSession,second:PreparedSession,
    profiles:&mut pair_serving::Profiles,rank:usize,out:&Path,ownership:Value)->Result<()> {
    let prompts=load_prompts(out)?;
    let progress=|phase:&str,frontiers:[u64;2]|->Result<()> {
        std::fs::write(out.join("paired-endpoint-progress.json"),serde_json::to_vec_pretty(
            &json!({"rank":rank,"arm":0,"comparison":"same_prefix","phase":phase,"frontiers":frontiers}))?)?;
        Ok(())
    };
    w.large.reset(1400,&w.comm,&w.stream)?;other.reset(1401,&w.comm,&w.stream)?;
    let mut prefill_seconds=Vec::new();
    progress("prefill_first",[0,0])?;
    let began=std::time::Instant::now();
    w.large.prefill_wide(&mut w.wides,&prompts[0],&w.comm,&w.stream)?;
    prefill_seconds.push(began.elapsed().as_secs_f64());
    let untouched=w.large.repeat_state_digests(&w.stream)?;
    progress("prefill_second",[w.large.position(),0])?;
    let began=std::time::Instant::now();
    other.prefill_wide(wides,&prompts[1],&w.comm,&w.stream)?;
    prefill_seconds.push(began.elapsed().as_secs_f64());
    ensure!(w.control.all_admit(w.large.repeat_state_digests(&w.stream)?==untouched)?,
        "matched endpoint peer prefill changed first causal owner");
    let mut a=Box::new(first.attach(&w.large,&w.stream)?);
    let mut b=Box::new(second.attach(other,&w.stream)?);
    let seeds=[9371u64,9372];
    for (i,session,target) in [(0,&mut a,&w.large),(1,&mut b,&*other)] {
        session.configure_sampling(seeds[i],None)?;
        session.configure_history_lookup(target,&prompts[i])?;
    }
    let mut anchors=[w.large.sample_token(seeds[0],&w.stream)?,other.sample_token(seeds[1],&w.stream)?];
    w.verify.handoff_from(&w.large)?;verify.handoff_from(other)?;
    let copy=profiles.check(w,verify,&[2u32;6])?;
    let mut executor=crate::target::PairExecutor::new(&w.verify,verify,pooled,&w.comm,&w.stream)?;
    let (mut paired_steps,mut separate_steps,mut single_survivor_steps)=(0,0,0);
    let mut records=Vec::new();
    while w.verify.position()<1048576||verify.position()<1048576 {
        ensure!(records.len()<128,"bounded matched endpoint steps");
        let remaining=[1048576-w.verify.position()as usize,1048576-verify.position()as usize];
        if remaining.iter().all(|&n|n>=6) {
            let (first,second,head_pooled)=a.prepare_pair_proposals(&w.verify,anchors[0],&mut b,verify,
                anchors[1],crate::config::paired_draft_head_requested(),&w.comm,&w.stream)?;
            ensure!(a.paired_probabilities_disjoint(&first,&b,&second)?,"matched endpoint proposal q aliased");
            let mut expected=Vec::new();let mut logits=Vec::new();
            for (target,ids) in [(&mut w.verify,first.inputs()),(&mut *verify,second.inputs())] {
                let before=committed(target,&w.stream,paired_steps==0)?;
                target.begin_verification(ids,&w.comm,&w.stream)?;
                expected.push(target.paired_completed_endpoint_taps(&w.stream)?);
                logits.push(local_logits(target,&w.cuda)?);
                target.finish_verification(0,&w.comm,&w.stream)?;
                ensure!(w.control.all_admit(committed(target,&w.stream,paired_steps==0)?==before)?,
                    "ordinary endpoint diagnostic rollback changed accepted state");
            }
            let paired=executor.begin(&mut w.verify,first.inputs(),verify,second.inputs(),&w.comm,&w.stream)?;
            let actual=vec![w.verify.paired_completed_endpoint_taps(&w.stream)?,
                verify.paired_completed_endpoint_taps(&w.stream)?];
            let same=actual==expected;
            if !same {
                let stem=format!("endpoint-same-prefix-mismatch-{paired_steps}");
                std::fs::write(out.join(format!("{stem}.json")),serde_json::to_vec_pretty(&json!({
                    "rank":rank,"frontiers":[w.verify.position(),verify.position()],
                    "inputs":[first.inputs(),second.inputs()],"expected":expected,"actual":actual}))?)?;
                for (i,target) in [&w.verify,&*verify].into_iter().enumerate() {
                    std::fs::write(out.join(format!("{stem}-owner{i}-ordinary.f32")),&logits[i])?;
                    std::fs::write(out.join(format!("{stem}-owner{i}-paired.f32")),local_logits(target,&w.cuda)?)?;
                }
            }
            ensure!(w.control.all_admit(same)?,"same-prefix endpoint target logits/state/features changed");
            let first=a.finish_pair_proposal(&mut w.verify,first,&w.comm,&w.stream)?;
            let second=b.finish_pair_proposal(verify,second,&w.comm,&w.stream)?;
            anchors=[first.next_anchor,second.next_anchor];
            records.push(json!({"kind":"paired","exact":true,"draft_head_pooled":head_pooled,
                "target":paired,"ordinary_pending":expected,
                "steps":[pair_generation::semantic_step(&first)?,pair_generation::semantic_step(&second)?]}));
            paired_steps+=1;
        }else{
            let mut steps=Vec::new();
            for (i,session,target) in [(0,&mut a,&mut w.verify),(1,&mut b,&mut *verify)] {
                if remaining[i]==0{continue;}
                let step=session.step(target,anchors[i],remaining[i].min(6),&w.comm,&w.stream)?;
                anchors[i]=step.next_anchor;separate_steps+=1;
                single_survivor_steps+=usize::from(remaining[1-i]==0);
                steps.push(json!({"owner":i,"step":pair_generation::semantic_step(&step)?}));
            }
            records.push(json!({"kind":"terminal_fallback","steps":steps}));
        }
        progress("matched_generation",[w.verify.position(),verify.position()])?;
    }
    ensure!(paired_steps>0&&separate_steps>0&&single_survivor_steps>0,"matched endpoint coverage incomplete");
    let endpoints=vec![endpoint(&w.verify,&a,&w.stream)?,endpoint(verify,&b,&w.stream)?];
    let memory=w.cuda.admission_memory()?;
    ensure!(w.control.all_admit(memory.admitted_available>=8*1024usize.pow(3))?,"matched endpoint reserve");
    drop(executor);
    std::fs::write(out.join("paired-endpoint-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"paired_execution_exact":true,"comparison":"same_prefix_same_proposals",
        "ownership":ownership,"memory":memory,"full_serving_profiles":profiles.memory,
        "prompt_tokens":prompts.iter().map(Vec::len).collect::<Vec<_>>(),"prefill_seconds":prefill_seconds,
        "endpoints":endpoints,"copy_profiles":copy,"paired_steps":paired_steps,
        "separate_steps":separate_steps,"single_survivor_steps":single_survivor_steps,"records":records,
        "scope":"Two actual1M endpoints. Every paired target step matches ordinary execution on identical accepted prefixes/proposals, including FP32 logits, residual, carry, retained storage and drafter-conditioning taps. Zero-accept diagnostic rollback checked; proposals drawn once, independent original acceptance law retained. Short-tail and survivor paths are ordinary execution. EOS ignored only for finite capacity. No repeated-prefill reproducibility, HTTP lifecycle, task-quality or throughput claim."
    }))?)?;Ok(())
}
