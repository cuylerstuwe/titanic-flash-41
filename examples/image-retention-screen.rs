//! Finite CPU image-history screen. No CUDA context, model evaluation or network.
use anyhow::{ensure, Result};
use base64::Engine as _;
use canada_engine::image_input_cache::{Expanded, ImageInputCache};
use deepseek_recipe_core::multimodal::{ImageDetail, ImageSource};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

fn signature(prepared: &Expanded) -> String {
    let mut h = Sha256::new();
    for token in &prepared.tokens { h.update(token.to_le_bytes()); }
    for input in &prepared.inputs {
        h.update(input.identity);
        h.update(serde_json::to_vec(&input.geometry).unwrap());
        for patch in &input.patches { h.update(patch.to_le_bytes()); }
    }
    format!("{:x}", h.finalize())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 3, "private image directory, new result path");
    let root = Path::new(&args[1]); let output = Path::new(&args[2]);
    ensure!(!output.exists(), "immutable result already exists");
    let manifest: Vec<Value> = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
    ensure!(manifest.len() == 96, "expected96image screen");
    let images = manifest.iter().map(|m| -> Result<ImageSource> {
        let raw = std::fs::read(root.join(m["name"].as_str().unwrap()))?;
        ensure!(format!("{:x}",Sha256::digest(&raw))==m["sha256"].as_str().unwrap(),"source changed");
        Ok(ImageSource::DataUrl { data_url:format!("data:{};base64,{}",
            m["media_type"].as_str().unwrap(),base64::engine::general_purpose::STANDARD.encode(raw)),detail:ImageDetail::Auto,path:String::new() })
    }).collect::<Result<Vec<_>>>()?;
    let mut records=Vec::new();
    for count in [16,50,96] {
        let cache=ImageInputCache::new();
        let mut expected=None;let mut runs=Vec::new();
        for trial in 0..7 {
            let start=Instant::now();let prepared=cache.expand(vec![129264;count],&images[..count],1048576)?;
            let milliseconds=start.elapsed().as_secs_f64()*1000.;
            let hash=signature(&prepared);
            if let Some(ref expected)=expected {ensure!(*expected==hash,"prepared bytes changed");}
            else {expected=Some(hash.clone());}
            // Do not hold a previous request alive: that would hide LRU churn.
            drop(prepared);
            runs.push(json!({"trial":trial,"milliseconds":milliseconds,"sha256":hash,
                "cache":cache.metadata()?}));
        }
        let row=json!({"images":count,"runs":runs});
        println!("{}",row);records.push(row);
    }
    std::fs::write(output,serde_json::to_vec_pretty(&json!({"records":records,
        "scope":"CPU canonical image-input preprocessing with immutable private screenshots. Requests drop ownership between turns. Hash covers token IDs, image identity, geometry and every BF16 patch. Timing excludes hashing; no GPU/serving speed claim."}))?)?;
    Ok(())
}
