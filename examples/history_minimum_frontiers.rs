//! Select first eligible public prefixes with the real discrete token index.
//! No future continuation, model arithmetic or probability enters selection.
#[allow(dead_code)]
#[path = "../src/history_lookup.rs"]
mod history_lookup;
use anyhow::{ensure,Context,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
fn main()->Result<()> {
    let args:Vec<_>=std::env::args_os().collect();ensure!(args.len()==4,"fixture, recorded public generation, output");
    let input=std::fs::read(&args[1])?;let generation=std::fs::read(&args[2])?;
    let fixtures:Value=serde_json::from_slice(&input)?;let records:Value=serde_json::from_slice(&generation)?;
    let out=std::path::Path::new(&args[3]);ensure!(!out.exists(),"immutable output already exists");
    let mut selected=Vec::new();
    for case in [0usize,1] {
        let c=&fixtures["cases"][case];let r=&records["records"][case];ensure!(c["name"]==r["name"],"case identity");
        let trial=&r["trials"][0];let emitted:Vec<u32>=serde_json::from_value(trial["output_ids"].clone())?;
        let mut prefix:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
        let mut lookup=history_lookup::HistoryLookup::with_strong_backup(1048576,&prefix,262144)?;
        let mut answer=false;let mut found=false;
        for (offset,&anchor) in emitted.iter().enumerate() {
            if anchor==128822 {answer=true;}
            let phase_ok=if case==0 {!answer}else{answer};
            if offset>=32&&phase_ok {
                if let Some((p,ids))=lookup.propose_block_minimum::<32>(lookup.position(),anchor,32)? {
                    if p.matched==32 {
                        ensure!(lookup.propose_block::<32>(lookup.position(),anchor)?.is_none(),"not a new opportunity");
                        selected.push(json!({"name":c["name"],"phase":if answer{"answer"}else{"reasoning"},
                            "input_ids":prefix,"anchor":anchor,"source_end":p.source_end,"prefix_match":p.matched,
                            "teacher_ids":ids,"recorded_output_offset":offset,"recorded_width":trial["width"],
                            "seed":2026110600+case as u64,"max_new_tokens":256}));found=true;break;
                    }
                }
            }
            lookup.commit(lookup.position(),&[anchor])?;prefix.push(anchor);
        }
        ensure!(found,"no32-only committed frontier in {}",c["name"].as_str().context("name")?);
    }
    std::fs::write(out,serde_json::to_vec_pretty(&json!({"cases":selected,
        "input_sha256":format!("{:x}",Sha256::digest(input)),"recorded_generation_sha256":format!("{:x}",Sha256::digest(generation)),
        "scope":"First32-only full-source opportunity per phase, using real262144-slot primary and65536-slot strong backup. Public recorded prefixes only. No future acceptance used for selection; conditional screens, not fresh task outcomes."}))?)?;
    for c in &selected {println!("{} {} prefix={} offset={}",c["name"],c["phase"],c["input_ids"].as_array().unwrap().len(),c["recorded_output_offset"]);}
    Ok(())
}
