//! Discrete CPU token-index experiment, not a CPU model/probability oracle.
#[path="../src/history_lookup.rs"]
mod history_lookup;
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::PathBuf,time::Instant};

fn main()->Result<()> {
    let args:Vec<_>=std::env::args_os().collect();ensure!(args.len()==3,"input and output paths");
    let input:Value=serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let output=PathBuf::from(&args[2]);ensure!(!output.exists(),"immutable screen output");
    let raw=std::fs::read(input["fixture_path"].as_str().unwrap())?;
    ensure!(format!("{:x}",Sha256::digest(&raw))==input["fixture_sha256"].as_str().unwrap(),"fixture hash");
    let fixture:Value=serde_json::from_slice(&raw)?;
    let corpus:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let mut cases=Vec::new();
    for case in input["cases"].as_array().unwrap() {
        let prefix_len=case["corpus_tokens"].as_u64().unwrap()as usize;
        let mut prefix=corpus[..prefix_len].to_vec();
        prefix.extend(serde_json::from_value::<Vec<u32>>(case["warmup"].clone())?);
        let emitted:Vec<u32>=serde_json::from_value(case["tokens"].clone())?;
        let mut variants=Vec::new();let mut baseline=None;
        for maximum in [16,32,64] {
            let began=Instant::now();
            let mut lookup=history_lookup::HistoryLookup::with_slots(1048576,&prefix,262144)?;
            if maximum>16 {lookup.resize_maximum_for_probe(maximum)?;}
            let build_seconds=began.elapsed().as_secs_f64();let mut records=Vec::new();
            let began=Instant::now();
            for (offset,&anchor) in emitted.iter().enumerate() {
                let position=lookup.position();
                if offset+5<emitted.len() {
                    let proposal=lookup.propose(position,anchor)?;
                    let matched=proposal.as_ref().map_or(0,|p|p.tokens.iter().zip(&emitted[offset+1..offset+6]).take_while(|(a,b)|a==b).count());
                    records.push(json!({"offset":offset,"matched_history":proposal.as_ref().map(|p|p.matched),
                        "proposal":proposal.as_ref().map(|p|p.tokens),"source_end":proposal.as_ref().map(|p|p.source_end),
                        "observed_continuation_prefix_matches":matched}));
                }
                lookup.commit(position,&[anchor])?;
            }
            let replay_seconds=began.elapsed().as_secs_f64();
            if baseline.is_none(){baseline=Some(records.clone());}
            let b=baseline.as_ref().unwrap();
            let changed:Vec<_>=records.iter().zip(b).filter(|(a,b)|a["proposal"]!=b["proposal"]).map(|(a,b)|json!({"baseline":b,"candidate":a})).collect();
            variants.push(json!({"maximum_match":maximum,"positions":records.len(),
                "proposals":records.iter().filter(|r|!r["proposal"].is_null()).count(),
                "long_matches":records.iter().filter(|r|r["matched_history"].as_u64().is_some_and(|n|n>=32)).count(),
                "observed_prefix_matches":records.iter().map(|r|r["observed_continuation_prefix_matches"].as_u64().unwrap()).sum::<u64>(),
                "changed":changed,"build_seconds":build_seconds,"replay_seconds":replay_seconds,
                "workspace_bytes":lookup.workspace_bytes()}));
        }
        cases.push(json!({"name":case["name"],"corpus_tokens":prefix_len,"variants":variants}));
    }
    std::fs::write(&output,serde_json::to_vec_pretty(&json!({"cases":cases,"scope":
        "Discrete causal history lookup on saved target-output streams. Overlapping prefix-match observations are correlated opportunity proxies, not measured target probabilities, emitted TPS or outcomes. All model arithmetic remains on GPU."}))?)?;
    Ok(())
}
