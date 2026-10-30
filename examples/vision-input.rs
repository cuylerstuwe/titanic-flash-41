//! CPU-only image input qualification. Never opens a GPU or loads model weights.
use anyhow::{Result, ensure};
use canada_engine::vision_input::{Detail, decode_rgb, identity, prepare_rgb};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 2,
        "usage: vision-input cases.json output-directory"
    );
    let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let root = Path::new(&args[0]).parent().unwrap();
    let out = Path::new(&args[1]);
    std::fs::create_dir_all(out)?;
    let mut records = Vec::new();
    for c in cases["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        ensure!(
            name.bytes()
                .all(|x| x.is_ascii_alphanumeric() || x == b'-' || x == b'_'),
            "case name"
        );
        let at = Instant::now();
        let encoded = std::fs::read(root.join(c["file"].as_str().unwrap()))?;
        let detail: Detail = serde_json::from_value(
            c.get("detail")
                .cloned()
                .unwrap_or(serde_json::json!("auto")),
        )?;
        let (w, h, rgb) = if c["raw"] == true {
            (
                c["width"].as_u64().unwrap() as usize,
                c["height"].as_u64().unwrap() as usize,
                encoded.clone(),
            )
        } else {
            decode_rgb(&encoded)?
        };
        let decode = at.elapsed().as_secs_f64();
        let at = Instant::now();
        let (plan, patches) = prepare_rgb(&rgb, w, h)?;
        let prep = at.elapsed().as_secs_f64();
        let bytes: Vec<u8> = patches.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out.join(format!("{name}.bf16")), &bytes)?;
        records.push(serde_json::json!({"name":name,"plan":plan,"source_dimensions":[w,h],
            "rgb_sha256":format!("{:x}",Sha256::digest(&rgb)),"patch_sha256":format!("{:x}",Sha256::digest(&bytes)),
            "identity":identity(&encoded,detail),"decode_seconds":decode,"prepare_seconds":prep}));
    }
    std::fs::write(
        out.join("results.json"),
        serde_json::to_vec_pretty(&records)?,
    )?;
    Ok(())
}
