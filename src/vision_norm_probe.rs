//! Finite full-tower comparison sharing canonical weights and scratch arena.
//! No language-model load, new image policy, or persistent serving change.
use crate::{checkpoint::Checkpoint, config::ModelConfig, cuda::Cuda,
    layout::RankPlan, vision::Vision, vision_input::{self,Detail}};
use anyhow::{Result,ensure};
use half::{bf16,slice::HalfBitsSliceExt};
use serde::Deserialize;
use serde_json::{Value,json};
use sha2::{Digest,Sha256};
use std::{path::{Path,PathBuf},time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    config:PathBuf, checkpoint:PathBuf, stage:PathBuf, kernel:PathBuf,
    attention:PathBuf, images:Vec<PathBuf>, weights_sha256:String,
}
pub fn run(path:&Path)->Result<Value> {
    let p:Config=serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!((1..=4).contains(&p.images.len()),"bounded vision norm probe");
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let config=ModelConfig::parse(&std::fs::read(&p.config)?)?;
    let cp=Checkpoint::from_manifest(&p.checkpoint)?;
    let plan=RankPlan::build(&cp,&config,0,4)?;
    let mut vision=Vision::new_staged(&cuda,&std::fs::read(&p.kernel)?,&p.attention,&cp,&plan,&p.stage)?;
    ensure!(vision.source_sha256()==p.weights_sha256,"vision weight identity");
    let arena=cuda.row_arena(503316480)?;
    let mut records=Vec::new();
    for (index,image) in p.images.iter().enumerate() {
        let raw=std::fs::read(image)?;
        let input=vision_input::prepare(&raw,Detail::Auto)?;
        let g=crate::vision::Geometry{height:input.plan.vit_height,width:input.plan.vit_width}.validate()?;
        let mut patches=cuda.allocate::<bf16>(input.patches.len())?;
        patches.upload(input.patches.reinterpret_cast())?;
        stream.join_default()?;
        let mut expected=None;
        let mut timings=[Vec::new(),Vec::new(),Vec::new()];
        let mut hashes=Vec::new();
        // One warm cycle, then rotating order to limit drift/order bias.
        for cycle in 0..7usize {
            for position in 0..3usize {
                let mode=(position+cycle)%3;
                vision.set_probe_norm_mode(mode as u8)?;
                let begin=cuda.event()?;let end=cuda.event()?;
                let wall=Instant::now();begin.record(&stream)?;
                let span=vision.encode_with_arena(&patches,g,&stream,true,Some(&arena))?;
                end.record(&stream)?;stream.synchronize()?;
                let wall_ms=wall.elapsed().as_secs_f64()*1000.;
                let event_ms=end.elapsed_since(&begin)?;
                let bits=span.data.download()?.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
                if let Some(ref reference)=expected {ensure!(&bits==reference,"vision final span differs: image{index} mode{mode} cycle{cycle}");}
                else {ensure!(mode==0,"baseline first");expected=Some(bits.clone());}
                if cycle==0 {
                    let mut h=Sha256::new();for b in bits {h.update(b.to_le_bytes());}
                    hashes.push(json!({"mode":mode,"span_sha256":format!("{:x}",h.finalize())}));
                }else{timings[mode].push(json!({"cycle":cycle,"wall_ms":wall_ms,"event_ms":event_ms}));}
            }
        }
        let record=json!({"case":index,"geometry":g,"image_tokens":g.tokens(),
            "encoded_sha256":format!("{:x}",Sha256::digest(&raw)),
            "all_spans_exact":true,"hashes":hashes,"timings":timings});
        println!("{}",record);records.push(record);
    }
    Ok(json!({"cases":records,"weights_sha256":p.weights_sha256,
        "scratch_bytes":arena.bytes(),"all_spans_exact":true,
        "scope":"Full canonical vision tower and aligner on identical private screenshot inputs, same resident weights and borrowed scratch. No full language-model or serving performance claim."}))
}
