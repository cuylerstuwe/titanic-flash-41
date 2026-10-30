//! Isolate small encoder graphs from the already carried decoder-tail graph.
use crate::{cuda::{Cuda,Stream}, dspark_generation::{compare_logits,compare_positions,teacher_forced_logits}, roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use std::{path::Path,time::Instant};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,output:&Path,rank:usize)->Result<Value> {
    let only128=std::env::var("CANADA_SCRATCH_TINY_ENCODER_128_SCREEN").as_deref()==Ok("1");
    ensure!(large.context_capacity()==1048576
        && (only128 || std::env::var("CANADA_SCRATCH_TINY_ENCODER_SCREEN").as_deref()==Ok("1")),
        "tiny encoder screen requires isolated accumulated1M profiles");
    let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/target-agentic-repository-v1.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(corpus["batches"][0].clone())?;
    let continuation:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/fixed-verify-python-v1.json"))?;
    let continuation:Vec<u32>=serde_json::from_value(continuation["teacher_forced_tokens"].clone())?;
    ensure!(tokens.len()>=458+4*256 && continuation.len()>=6,"tiny encoder fixture lengths");
    let mut records=Vec::new();let mut lineage=91000u64;
    for &delta in if only128{&[128usize][..]}else{&[128usize,256][..]} {
        let variants:&[usize]=if delta==128 {&[0,2]}else{&[0,1,2]};
        for request in 0..4 {
            let mut outputs=vec![Value::Null;3];
            let mut logits:Vec<Vec<f32>>=vec![Vec::new();3];
            let mut forced:Vec<Vec<Vec<f32>>>=vec![Vec::new();3];
            for offset in 0..variants.len() {
                let path=variants[if request%2==0 {offset}else{variants.len()-1-offset}];
                lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;
                large.prefill_wide_with_graph_policy(wides,&tokens[..458],comm,stream,true,true,false)?;
                let incoming=json!({"position":large.position(),"encoder":large.encoder_tail_digest()?,"carry":large.carry_digests()?});
                let new_tokens=&tokens[458+request*delta..458+(request+1)*delta];
                let start=Instant::now();
                // The current cost planner selects a partially live512-row
                // profile for256 tokens, which cannot replay an encoder graph.
                // Path1 isolates the choice of two128-row eager chunks; path2
                // adds encoder graphs. Every path keeps decoder-tail graphs.
                if path==0 {
                    large.prefill_wide_with_graph_policy(wides,new_tokens,comm,stream,true,true,false)?;
                }else{
                    large.prefill_wide_with_graph_policy(&mut[],new_tokens,comm,stream,true,true,path==2)?;
                }
                let anchor=large.greedy_token(stream)?;
                let seconds=start.elapsed().as_secs_f64();
                ensure!(large.position()as usize==458+delta,"tiny encoder frontier");
                let phases=serde_json::to_value(large.append_metrics()?)?;
                ensure!(request<2 || phases["captured_new_graph"]==false,"tiny encoder measured recapture");
                if path==2 && request>=2 {
                    ensure!(phases["encoder_graph_chunks"].as_u64()==Some((delta/128)as u64),"tiny encoder candidate did not replay all chunks");
                }else if path!=2 {ensure!(phases["encoder_graph_chunks"]==0,"eager encoder control replayed");}
                logits[path]=large.last_prompt_logits()?;
                let outgoing=json!({"position":large.position(),"anchor":anchor,"encoder":large.encoder_tail_digest()?,"carry":large.carry_digests()?});
                verify.handoff_from(large)?;
                forced[path]=teacher_forced_logits(cuda,verify,&continuation[..6],1,comm,stream)?;
                ensure!(logits[path].iter().chain(forced[path].iter().flatten()).all(|v|v.is_finite()),"tiny encoder nonfinite outputs");
                outputs[path]=json!({"seconds":seconds,"phases":phases,"incoming":incoming,"outgoing":outgoing});
            }
            let comparisons:Vec<Value>=variants.iter().copied().filter(|&p|p!=0).map(|path|json!({
                "path":path,"incoming_state_equal":outputs[0]["incoming"]==outputs[path]["incoming"],
                "outgoing_state_equal":outputs[0]["outgoing"]==outputs[path]["outgoing"],
                "last_row_logits":compare_logits(&logits[0],&logits[path]),
                "teacher_forced":compare_positions(&forced[0],&forced[path]),
                "teacher_forced_bitwise_equal":forced[0].iter().zip(&forced[path]).all(|(a,b)|a.iter().zip(b).all(|(x,y)|x.to_bits()==y.to_bits()))})).collect();
            ensure!(comparisons.iter().all(|r|r["incoming_state_equal"]==true),"tiny encoder unmatched incoming states");
            let record=json!({"prefix_tokens":458,"new_tokens":delta,"request":request,"warmup":request<2,
                "variants":variants,"outputs":outputs,"comparisons":comparisons});
            std::fs::write(output.join(format!("tiny-encoder-d{delta}-r{request}.json")),serde_json::to_vec_pretty(&record)?)?;
            records.push(record);
        }
    }
    Ok(json!({"rank":rank,"records":records,"only128":only128,"cached_grouped_union_layers":large.cached_grouped_union_layers(),"scope":"Small retained increments (128only when indicated; otherwise128/256) with changed input tokens;2warm+2measured order-balanced rounds. Current planner/eager encoder versus128-only eager versus128-only graphs; decoder-tail graphs and lookahead kept on throughout. Prompt logits, state digests and6following positions. Same model load as a narrow normalization timing follow-up. Foreground only; no outcome or production claim."}))
}

/// A repeated eager control separates graph differences from ordinary run-to-run
/// variation. One input, one frontier, no broad numerical or performance matrix.
#[allow(clippy::too_many_arguments)]
pub(crate) fn diagnose(cuda:&Cuda,large:&mut Target,verify:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,output:&Path,rank:usize)->Result<Value> {
    ensure!(large.context_capacity()==1048576
        && std::env::var("CANADA_SCRATCH_TINY_ENCODER_DIAGNOSTIC").as_deref()==Ok("1"),
        "tiny encoder diagnostic admission");
    let corpus:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/target-agentic-repository-v1.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(corpus["batches"][0].clone())?;
    let mut records=Vec::new();let mut reference:Option<Vec<f32>>=None;
    for (case,graph)in [false,false,true,true,false].into_iter().enumerate() {
        verify.reset(92000+case as u64,comm,stream)?;large.handoff_from(verify)?;
        large.prefill_wide_with_graph_policy(wides,&tokens[..458],comm,stream,true,true,false)?;
        let incoming=json!({"encoder":large.encoder_tail_digest()?,"carry":large.carry_digests()?});
        large.prefill_wide_with_graph_policy(wides,&tokens[458..586],comm,stream,true,true,graph)?;
        stream.synchronize()?;comm.health()?;
        let logits=large.last_prompt_logits()?;ensure!(logits.iter().all(|x|x.is_finite()),"tiny diagnostic nonfinite");
        let trace=large.tiny_encoder_trace()?;
        let comparison=reference.as_ref().map(|r|compare_logits(r,&logits));
        if reference.is_none(){reference=Some(logits);}
        let record=json!({"case":case,"encoder_graph":graph,"incoming":incoming,
            "outgoing":json!({"encoder":large.encoder_tail_digest()?,"carry":large.carry_digests()?}),
            "phases":large.append_metrics()?,"trace":trace,"logits_vs_first_eager":comparison});
        std::fs::write(output.join(format!("tiny-encoder-diagnostic-{case}.json")),serde_json::to_vec_pretty(&record)?)?;
        verify.handoff_from(large)?;records.push(record);
    }
    let _=cuda;
    Ok(json!({"rank":rank,"tiny_encoder_diagnostic":true,"records":records,
        "scope":"One458-token prefix plus128new tokens, eager/eager/capture/replay/eager. Retained outputs, source20 pages, masks, metadata and last-row layer hashes. No timed qualification or background checks."}))
}
