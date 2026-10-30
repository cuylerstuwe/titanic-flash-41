//! Score causally chosen long history proposals with the unchanged target.
//! Export FP32 logits only; probability calculations run on a GPU afterwards.
use crate::{cuda::{Cuda,Stream},history_lookup::HistoryLookup,roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde::Deserialize;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::path::Path;

#[derive(Deserialize)]
struct Case {
    name:String, prompt_ids:Vec<u32>, anchor:u32, proposal_ids:Vec<u32>,
    prefix_match:usize, source_end:usize,
}
#[derive(Deserialize)]
struct Fixture { schema:String, cases:Vec<Case> }

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,output:&Path,rank:usize)->Result<Value> {
    ensure!(std::env::var("CANADA_SCRATCH_LONG_HISTORY_SCORE").as_deref()==Ok("1")
        && large.position()==0 && large.context_capacity()==1048576
        && verify.physical_rows()==6 && verify.head_gemv_enabled()
        && verify.query_overlap_layers()==40 && wides.len()==5,
        "long history score requires the current isolated target profiles");
    let raw=include_bytes!("../tests/fixtures/long-history-score-v1.json");
    let fixture:Fixture=serde_json::from_slice(raw)?;
    ensure!(fixture.schema=="long-history-score-v1" && (1..=20).contains(&fixture.cases.len()),
        "bounded history score fixture");
    let directory=output.file_name().and_then(|s|s.to_str()).unwrap_or("");
    let prefix=format!("target-native-r{rank}-");
    let version=directory.strip_prefix(&prefix).ok_or_else(||anyhow::anyhow!("score output identity"))?;
    ensure!(version.starts_with('s') && version[1..].bytes().all(|b|b.is_ascii_digit()),"score version");
    let raw_root=output.parent().unwrap().join(format!("long-history-score-{version}"));
    if rank==3 {std::fs::create_dir(&raw_root)?;}
    large.enable_cached_extend_floor()?;
    let mut records=Vec::new();
    for (index,case) in fixture.cases.iter().enumerate() {
        ensure!((512..=16384).contains(&case.prompt_ids.len()) && case.proposal_ids.len()==15
            && case.anchor<129280 && case.prompt_ids.iter().chain(&case.proposal_ids).all(|&id|id<129280),
            "history score token geometry");
        let lookup=HistoryLookup::with_maximum(1048576,&case.prompt_ids,262144,64)?;
        let proposal=lookup.propose(lookup.position(),case.anchor)?
            .ok_or_else(||anyhow::anyhow!("causal history proposal missing"))?;
        ensure!(proposal.matched==case.prefix_match && proposal.matched>=16
            && proposal.source_end==case.source_end
            && case.source_end+16<=case.prompt_ids.len()
            && case.prompt_ids[case.source_end+1..case.source_end+16]==case.proposal_ids,
            "fixture proposal was not selected from committed history");
        large.prefill_wide(wides,&case.prompt_ids,comm,stream)?;
        verify.handoff_from(large)?;
        let mut inputs=vec![case.anchor];inputs.extend_from_slice(&case.proposal_ids);
        let mut bytes=Vec::with_capacity(16*129280*4);
        for offset in (0..16).step_by(6) {
            let live=(16-offset).min(6);
            ensure!(verify.position()as usize==case.prompt_ids.len()+offset,"history score causal frontier");
            verify.begin_verification(&inputs[offset..offset+live],comm,stream)?;
            let (logits,rows)=verify.pending_logits()?;ensure!(rows==live,"score live logits");
            stream.synchronize()?;
            bytes.extend_from_slice(&unsafe{cuda.read_device(logits.ptr(),live*129280*4)}?);
            verify.finish_verification(live,comm,stream)?;
        }
        ensure!(bytes.len()==16*129280*4 && verify.position()as usize==case.prompt_ids.len()+16,
            "score complete conditional history");
        let file=format!("case-{index}.bin");
        if rank==3 {std::fs::write(raw_root.join(&file),&bytes)?;}
        records.push(json!({"case":index,"name":case.name,"prompt_tokens":case.prompt_ids.len(),
            "anchor":case.anchor,"proposal_ids":case.proposal_ids,"prefix_match":case.prefix_match,
            "source_end":case.source_end,"rows":16,"bytes":bytes.len(),"file":file,
            "sha256":format!("{:x}",Sha256::digest(&bytes)),"carry":verify.carry_digests()?,
            "encoder":verify.encoder_tail_digest()?}));
        eprintln!("long history score rank{rank} case{index}:16 conditional rows");
        verify.reset(850000+index as u64,comm,stream)?;large.handoff_from(verify)?;
    }
    let result=json!({"rank":rank,"long_history_score":true,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "records":records,"raw_directory":raw_root,"raw_rank":3,"target_temperature":1.,"target_top_p":1.,
        "target_head":"canonical_bf16_weights_fp32_output","performance_result":false,"full_engine_acceptance":false,
        "scope":"At most20 causal history-copy frontiers from saved public tool-call outputs. Score15 known-history proposals with the current target via6/6/4-row teacher forcing. GPU probability analysis follows. This is acceptance opportunity, not a wider verifier, decoding speed, or task-quality result."});
    if rank==3 {std::fs::write(raw_root.join("index.json"),serde_json::to_vec_pretty(&result)?)?;}
    Ok(result)
}
