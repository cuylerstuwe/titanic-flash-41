//! Small foreground prefill screen: one16K encoder chunk versus two8K chunks.
use crate::{cuda::{Cuda,Stream},dspark_generation::teacher_forced_logits,roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,output:&Path,rank:usize)->Result<Value> {
    ensure!(std::env::var("CANADA_SCRATCH_WIDE16K_SCREEN").as_deref()==Ok("1")
        && large.context_capacity()==32768 && wides.len()==2,"isolated16K prefill screen");
    let widths:Vec<_>=wides.iter().map(Target::physical_rows).collect();
    ensure!(widths==[16384,8192] && wides.iter().all(|w|w.moe_gather_layers()==18 && w.quant_gather_layers()==15),
        "both width profiles retain current packed MoE and attention transport");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    ensure!(format!("{:x}",Sha256::digest(raw))=="107ceccbefc53247fc827cdee7eb3908f7b89a907d8c7a2d3de3ef6b95057fbb","width fixture identity");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let mut initial=true;let mut lineage=190000;let mut records=Vec::new();
    for prefix in [0,8191] {
        for iteration in 0..3 {
            let mut paths=[Value::Null,Value::Null];
            for path in [iteration%2,1-iteration%2] {
                if !initial {lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;}
                initial=false;
                if prefix>0 {large.prefill_wide_with_policy(&mut wides[1..2],&tokens[..prefix],comm,stream,false,false)?;}
                let incoming=json!({"position":large.position(),"encoder":large.encoder_tail_digest()?,"carry":large.carry_digests()?});
                let index=1-path;
                let start=Instant::now();
                large.prefill_wide_with_policy(&mut wides[index..index+1],&tokens[prefix..prefix+16384],comm,stream,false,false)?;
                let anchor=large.greedy_token(stream)?;
                let seconds=start.elapsed().as_secs_f64();
                ensure!(large.position()as usize==prefix+16384,"width screen final frontier");
                let phases=serde_json::to_value(large.append_metrics()?)?;
                ensure!(phases["encoder_graph_chunks"]==0 && phases["captured_new_graph"]==false,"eager width comparison");
                let outgoing=json!({"position":large.position(),"encoder":large.encoder_tail_digest()?,"carry":large.carry_digests()?});
                let mut logits=vec![large.last_prompt_logits()?];verify.handoff_from(large)?;
                logits.extend(teacher_forced_logits(cuda,verify,&tokens[prefix+16384..prefix+16390],1,comm,stream)?);
                ensure!(logits.len()==7 && logits.iter().all(|row|row.len()==129280 && row.iter().all(|x|x.is_finite())),"finite width-screen logits");
                let bytes:Vec<u8>=logits.iter().flatten().flat_map(|v|v.to_le_bytes()).collect();
                let file=format!("wide16k-p{prefix}-i{iteration}-v{path}.f32");
                std::fs::write(output.join(&file),&bytes)?;
                paths[path]=json!({"width":widths[index],"seconds":seconds,"anchor":anchor,"phases":phases,
                    "incoming":incoming,"outgoing":outgoing,"logit_rows":logits.len(),"logits_file":file,"logits_sha256":format!("{:x}",Sha256::digest(&bytes))});
            }
            ensure!(paths[0]["incoming"]==paths[1]["incoming"],"width pair incoming state");
            let record=json!({"prefix_tokens":prefix,"new_tokens":16384,"iteration":iteration,"warmup":iteration==0,
                "outgoing_state_exact":paths[0]["outgoing"]==paths[1]["outgoing"],
                "logits_exact":paths[0]["logits_sha256"]==paths[1]["logits_sha256"],"paths":paths});
            std::fs::write(output.join(format!("wide16k-p{prefix}-i{iteration}.json")),serde_json::to_vec_pretty(&record)?)?;
            records.push(record);
        }
    }
    Ok(json!({"rank":rank,"wide16k_screen":true,"records":records,"profiles":wides.iter().filter_map(Target::wide_report).collect::<Vec<_>>(),
        "memory":cuda.admission_memory()?,"full_engine_acceptance":false,
        "scope":"Six foreground paired eager prefills at32K capacity, one warmup/two order-balanced pairs each for fresh16K and8191+16K. Both widths use accumulated packed transport. Full FP32 prompt plus six teacher-forced predictions, state digests and memory. No sampled-outcome or production claim; target profile remains T1/P1/high reasoning."}))
}
