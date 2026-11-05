//! Finite CPU input screen; never opens a CUDA context or inference endpoint.
use anyhow::{Result, ensure};
use base64::Engine as _;
use canada_engine::protocol::{Frontend, PreparedChat};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{io::Cursor, path::Path, time::Instant};
const MODEL_CONTEXT: usize = 1048576;

fn image_url(width: u32, height: u32) -> String {
    let mut seed = 81213u32;
    let image = image::RgbImage::from_fn(width, height, |_, _| {
        image::Rgb(std::array::from_fn(|_| {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            seed as u8
        }))
    });
    let mut encoded = Cursor::new(Vec::new());
    image.write_to(&mut encoded, image::ImageFormat::Png).unwrap();
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(encoded.into_inner()))
}

fn identical(a: &PreparedChat, b: &PreparedChat) -> Result<()> {
    ensure!(a.tokens == b.tokens && a.rendered_sha256 == b.rendered_sha256
        && a.max_new_tokens == b.max_new_tokens && a.image_inputs.len() == b.image_inputs.len(),
        "changed canonical prompt");
    for (x, y) in a.image_inputs.iter().zip(&b.image_inputs) {
        ensure!(x.identity == y.identity && x.geometry == y.geometry && x.patches == y.patches,
            "changed prepared image");
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 3, "tokenizer path and new output path required");
    let output = Path::new(&args[2]); ensure!(!output.exists(), "output already exists");
    let frontend = Frontend::load_with_images(Path::new(&args[1]), true)?;
    let small = image_url(16, 16); let large = image_url(1024, 768);
    let mut records = Vec::new();
    for (name, url, count) in [("small", &small, 1), ("large", &large, 1), ("four_repeated_large", &large, 4)] {
        let mut content = vec![json!({"type":"text","text":"Compare the provided images and describe their colors."})];
        for _ in 0..count { content.push(json!({"type":"image_url","image_url":{"url":url,"detail":"auto"}})); }
        let body = serde_json::to_vec(&json!({"messages":[{"role":"user","content":content}],
            "temperature":1,"top_p":1,"top_k":-1,"reasoning_effort":"high","max_tokens":64}))?;
        let start=Instant::now(); let reference=frontend.prepare_chat(&body, MODEL_CONTEXT)?;
        let first_ms=start.elapsed().as_secs_f64()*1000.;
        let mut warm_ms=Vec::new();
        for _ in 0..6 {
            let mut elapsed=0.;
            for _ in 0..3 {
                let start=Instant::now();let prepared=frontend.prepare_chat(&body, MODEL_CONTEXT)?;
                elapsed+=start.elapsed().as_secs_f64()*1000.;
                identical(&reference, &prepared)?;
                std::hint::black_box(prepared);
            }
            warm_ms.push(elapsed/3.);
        }
        let ids: Vec<u8> = reference.tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        let patches: Vec<u8> = reference.image_inputs[0].patches.iter().flat_map(|p| p.to_le_bytes()).collect();
        let record=json!({"name":name,"request_bytes":body.len(),"images":count,
            "prompt_tokens":reference.tokens.len(),"first_ms":first_ms,"warm_ms":warm_ms,
            "token_sha256":format!("{:x}",Sha256::digest(&ids)),
            "image_identity":reference.image_inputs[0].identity,"patch_sha256":format!("{:x}",Sha256::digest(&patches)),
            "all_tokens_and_patches_exact":true});
        println!("{record}");records.push(record);
    }
    std::fs::write(output, serde_json::to_vec_pretty(&json!({"complete":true,"records":records,
        "scope":"Mac/ARM CPU frontend screen with real PNG input and canonical tokenizer. Includes JSON/conversion/tokenization/preprocessing; no GPU, network, inference or end-to-end TTFT."}))?)?;
    Ok(())
}
