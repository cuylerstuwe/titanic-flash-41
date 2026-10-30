//! A small screen of two independent draft branches at matched frontiers.
//! The target evaluates branches serially; no fused-tree speed is claimed.
use crate::{checkpoint::Checkpoint,cuda::{Cuda,Stream},dspark_generation::GreedySession,
    layout::RankPlan,roce::Roce,target::Target};
use anyhow::{Result,ensure};
use serde_json::{Value,json};
use std::path::Path;

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,core:&[u8],stream:&Stream,cp:&Checkpoint,plan:&RankPlan,
    large:&mut Target,verify:&mut Target,wides:&mut [Target],comm:&Roce,prefix:&[u32],
    staged:&Path,kernels:&Path,output:&Path,session:&mut Option<GreedySession>) -> Result<Value> {
    ensure!(large.position()==0 && large.context_capacity()==32768 && prefix.len()==511
        && std::env::var("CANADA_SCRATCH_ALGORITHM_SCREEN").as_deref()==Ok("1"),
        "tree opportunity screen requires the isolated32K algorithm fixture");
    let seed=std::env::var("CANADA_SCRATCH_TARGET_SEED")?.parse::<u64>()?;
    let long:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/target-agentic-long15800-v1.json"))?;
    let mut long:Vec<u32>=serde_json::from_value(long["batches"][0].clone())?;
    let teacher:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/fixed-verify-python-v1.json"))?;
    let teacher:Vec<u32>=serde_json::from_value(teacher["teacher_forced_tokens"].clone())?;
    ensure!(long.len()==15800 && teacher.len()==512,"tree screen fixture shape");
    long.extend_from_slice(&teacher);long.extend_from_slice(&teacher);
    large.enable_cached_extend_floor()?;
    let mut records=Vec::new();let mut continuations=Vec::new();
    for (prompt_index,prompt) in [prefix,long.as_slice()].into_iter().enumerate() {
        large.prefill_wide(wides,prompt,comm,stream)?;
        let request_seed=seed.wrapping_add(prompt_index as u64);
        let mut anchor=large.sample_token(request_seed,stream)?;
        if let Some(s)=session.as_mut() {s.reset_from_prefill(large,cuda,core,stream)?;}
        else {*session=Some(GreedySession::from_prefill(large,cuda,core,stream,cp,plan,staged,kernels)?);}
        let s=session.as_mut().unwrap();s.configure_sampling(request_seed,None)?;
        s.set_causal_three_threshold_for_probe(Some(0.20))?;
        verify.handoff_from(large)?;
        for live in [3,5,6] {
            verify.begin_verification(&teacher[..live],comm,stream)?;
            verify.verification_greedy(stream)?;
            verify.finish_verification(0,comm,stream)?;
        }
        for sample in 0..4 {
            // Order rotation limits systematic cold-order differences. The
            // first frontier also warms both drafter-temperature graph slots.
            for slot in 0..2 {
                let variant=(sample+slot)%2;
                let temperature=if variant==0 {1.} else {0.7};
                s.configure_algorithm_for_probe(false,temperature,stream)?;
                let case=format!("tree-p{prompt_index}-s{sample}-t{variant}");
                let mut record=s.probe_tree_branches(verify,anchor,
                    request_seed.wrapping_add(1000+sample as u64),plan.rank,
                    &output.join(&case),comm,stream)?;
                record["case"]=json!(case);record["prompt_index"]=json!(prompt_index);
                record["sample"]=json!(sample);record["temperature"]=json!(temperature);
                record["warmup"]=json!(sample==0);record["rank"]=json!(plan.rank);
                record["prompt_tokens"]=json!(prompt.len());records.push(record);
            }
            // A separate fixed-policy trajectory selects frontiers. Neither
            // branch acceptance score nor a future confidence selects them.
            s.configure_algorithm_for_probe(false,1.,stream)?;
            for _ in 0..8 {
                let step=s.step(verify,anchor,6,comm,stream)?;
                anchor=step.next_anchor;continuations.push(serde_json::to_value(step)?);
            }
        }
        verify.reset(140000+prompt_index as u64,comm,stream)?;
        large.handoff_from(verify)?;
    }
    Ok(json!({"rank":plan.rank,"tree_opportunity_screen":true,"records":records,
        "continuations":continuations,"context_capacity":32768,"target_temperature":1.,
        "target_top_p":1.,"target_head":"canonical_bf16_weights_fp32_output",
        "scope":"16 matched-frontier fixtures: two prompts, four frontiers, two drafter temperatures. Linear four drafts versus two independent branches of two drafts, each five logical target nodes including shared anchor. Branches are evaluated serially and rolled back; this is acceptance opportunity and serial cost evidence, not a working tree decoder or a fused-tree speed measurement."}))
}
