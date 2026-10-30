//! Three independent small scheduling/publication changes under one model load.
use crate::{cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};
const NAMES:[&str;4]=["control","compact_remap_sort","task_order_m3","engram_split14_m3"];

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,candidates:&mut[Target],
    wides:&mut[Target],comm:&Roce,stream:&Stream,output:&Path,rank:usize)->Result<Value> {
    ensure!(std::env::var("CANADA_SCRATCH_VERIFICATION_FUSION_SCREEN").as_deref()==Ok("1")
        && large.context_capacity()==1048576 && candidates.len()==3
        && verify.fused_compact_publication_layers()==0 && verify.task_order_m3_layers()==0
        && candidates[0].fused_compact_publication_layers()==4 && candidates[0].task_order_m3_layers()==0
        && candidates[1].fused_compact_publication_layers()==0 && candidates[1].task_order_m3_layers()==40
        && candidates[2].fused_compact_publication_layers()==0 && candidates[2].task_order_m3_layers()==0
        && candidates.iter().all(|t|t.schedule_counts_for_probe()==verify.schedule_counts_for_probe()
            && t.compact_sparse_index_layers()==4 && t.narrow_projection_layers()==0 && t.task_order_moe_layers()==0),
        "independent verification variants must share every other policy");
    let raw=include_bytes!("../tests/fixtures/long-retained-prefill-v1.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let mut barrier_in=cuda.allocate::<f32>(4)?;barrier_in.upload(&[0.;4])?;
    let mut barrier_out=cuda.allocate::<f32>(16)?;
    let mut records=Vec::new();let mut prefills=Vec::new();
    for (index,frontier) in [131072,16384].into_iter().enumerate() {
        if index>0 {verify.reset(640000+index as u64,comm,stream)?;large.handoff_from(verify)?;}
        let start=Instant::now();large.prefill_wide_with_prefetch(wides,&tokens[..frontier],comm,stream,true)?;
        prefills.push(json!({"tokens":frontier,"seconds":start.elapsed().as_secs_f64()}));
        verify.handoff_from(large)?;
        let carry=verify.carry_digests()?;let encoder=verify.encoder_tail_digest()?;
        for live in [3,6] {
            let mut reference=None;
            let measured=if frontier==131072 {4}else{1};
            for iteration in 0..=measured {
                for slot in 0..4 {
                    let path=(slot+iteration)%4;
                    if path>0 {candidates[path-1].handoff_from(verify)?;}
                    let target=if path==0 {&mut *verify}else{&mut candidates[path-1]};
                    stream.join_default()?;
                    unsafe{comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)?;}
                    stream.synchronize()?;comm.health()?;
                    let start=Instant::now();
                    target.begin_verification(&tokens[frontier..frontier+live],comm,stream)?;
                    let predictions=target.verification_greedy(stream)?;
                    let ms=start.elapsed().as_secs_f64()*1000.;
                    let phases=serde_json::to_value(target.verification_metrics()?)?;
                    let (logits,rows)=target.pending_logits()?;ensure!(rows==live,"variant live logits");
                    let hash=format!("{:x}",Sha256::digest(unsafe{cuda.read_device(logits.ptr(),live*129280*4)}?));
                    let signature=json!({"logits":hash,"predictions":predictions});
                    let exact=reference.as_ref().is_none_or(|r|r==&signature);
                    if reference.is_none(){reference=Some(signature.clone());}
                    let active=target.active_sparse_index_layers();
                    target.finish_verification(0,comm,stream)?;
                    let rollback=target.position()as usize==frontier && target.carry_digests()?==carry
                        && target.encoder_tail_digest()?==encoder;
                    let record=json!({"frontier":frontier,"live_rows":live,"iteration":iteration,"warmup":iteration==0,
                        "variant":NAMES[path],"milliseconds":ms,"phases":phases,"signature":signature,
                        "same_signature":exact,"rollback_exact":rollback,"active_sparse_layers":active,
                        "capture_statistics":target.capture_statistics()});
                    std::fs::write(output.join(format!("verification-fusion-f{frontier}-m{live}-i{iteration}-v{path}.json")),
                        serde_json::to_vec_pretty(&record)?)?;
                    ensure!(exact&&rollback&&active==if frontier==131072 {4}else{0},
                        "verification variant output/state or adaptive fallback changed; saved evidence");
                    if path>0 {verify.handoff_from(&candidates[path-1])?;}
                    records.push(record);
                }
            }
            eprintln!("verification variants rank{rank} frontier{frontier} M{live}: complete");
        }
    }
    Ok(json!({"rank":rank,"verification_fusion_screen":true,"all_exact":true,"names":NAMES,"records":records,
        "prefills":prefills,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),"memory":cuda.admission_memory()?,
        "full_engine_acceptance":false,"scope":"Three independent changes against current control under one1Mmodel load: compact integer remap/sort/output fusion, task-order only atM3, Engram split only atM3. At128K, warmM3/M6 with one warmup and four balanced iterations. Reset to16K checks dense fallback with one warmup/one measured iteration per shape. Full live logits, all-rank agreement, graph reuse and rollback; no broad throughput/quality claim."}))
}
