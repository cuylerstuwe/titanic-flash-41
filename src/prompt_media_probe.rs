//! Lossless image injection qualification against the frozen text expansion
//! kernel plus GPU byte copies. There is no CPU model/reference arithmetic.
use crate::{cuda::{Cuda, Module, Stream}, ffn::TokenRows,
    prompt_media::{Bindings, Identity, ImageKey, Placement, IMAGE_TOKEN},
    vision::{Geometry, ImageSpan}};
use anyhow::{Result, ensure};
use half::bf16;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::{Path, PathBuf}, rc::Rc};

#[derive(Deserialize)]
struct Config { kernel: PathBuf, reference_kernel: PathBuf, reference: PathBuf, output: PathBuf }

fn expand(module: &Module, input: u64, output: u64, rows: usize, images: Option<u64>, stream: &Stream) -> Result<()> {
    let (mut input, mut output, mut rows, mut images_ptr) = (input, output, rows as i32, images.unwrap_or(0));
    let mut args = vec![(&mut input as *mut u64).cast(), (&mut output as *mut u64).cast(),
        (&mut rows as *mut i32).cast()];
    if images.is_some() { args.push((&mut images_ptr as *mut u64).cast()); }
    unsafe { module.function("ca_expand_hc_embedding")?.launch(
        [((rows as usize * 20480).div_ceil(256)) as u32, 1, 1], [256,1,1], 0, stream, &mut args) }
}

pub fn run(config: &Path) -> Result<Value> {
    let p: Config = serde_json::from_slice(&std::fs::read(config)?)?;
    ensure!(!p.output.exists(), "media probe output already exists");
    let index: Value = serde_json::from_slice(&std::fs::read(p.reference.join("index.json"))?)?;
    let cases = index["cases"].as_array().ok_or_else(|| anyhow::anyhow!("vision fixture cases"))?;
    ensure!(cases.len() == 2, "media probe requires two image geometries");
    let cuda = Cuda::open(0)?; cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let module = cuda.module(&std::fs::read(p.kernel)?)?;
    let old = cuda.module(&std::fs::read(p.reference_kernel)?)?;
    let mut spans = Vec::new(); let mut keys = Vec::new();
    for case in cases {
        let name = case["name"].as_str().ok_or_else(|| anyhow::anyhow!("vision fixture name"))?;
        ensure!(name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-'), "fixture basename");
        let geometry = serde_json::from_value::<Geometry>(case["geometry"].clone())?.validate()?;
        let bytes = std::fs::read(p.reference.join(name).join("span.bin"))?;
        ensure!(bytes.len() == geometry.tokens() * 5120 * 2, "fixture span size");
        let digest = Sha256::digest(&bytes);
        let tap = case["taps"].as_array().unwrap().iter().find(|t| t["name"] == "span").unwrap();
        ensure!(format!("{digest:x}") == tap["sha256"].as_str().unwrap(), "fixture span hash");
        let mut data = cuda.allocate::<bf16>(geometry.tokens() * 5120)?;
        data.upload(&bytes.chunks_exact(2).map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap()))).collect::<Vec<_>>())?;
        spans.push(Rc::new(ImageSpan { data, geometry }));
        keys.push(ImageKey { input: digest.into(), encoder: [1;32] });
    }
    let mut records = Vec::new();
    for rows in [6usize,16,128,8192] {
        let text = cuda.allocate::<bf16>(rows * 5120)?;
        let output = cuda.allocate::<bf16>(rows * 20480)?;
        let expected = cuda.allocate::<bf16>(rows * 20480)?;
        // Construct a deterministic input on GPU by repeating immutable rows.
        for start in (0..rows).step_by(spans[0].geometry.tokens()) {
            let count = (rows-start).min(spans[0].geometry.tokens());
            unsafe { cuda.copy_device(text.ptr() + (start*5120*2) as u64, spans[0].data.ptr(), count*5120*2, &stream)?; }
        }
        stream.synchronize()?;
        let mut token_rows = TokenRows::new(&cuda, &vec![2;rows], rows)?;
        stream.join_default()?;
        // Capture while all rows are text, then change both contents and image
        // addresses without recapturing. Later text must clear stale pointers.
        let graph = stream.capture(|| expand(&module, text.ptr(), output.ptr(), rows,
            Some(token_rows.image_pointers().ptr()), &stream))?;
        for (mode, start, live) in [(0,0,rows), (1,0,rows), (1,spans[0].geometry.tokens()+1,rows-1),
            (2,0,rows), (0,0,rows)] {
            let mut ids = vec![2i64;rows];
            let mut refs = Vec::new();
            let mut placements = Vec::new();
            if mode != 0 {
                let order = if mode == 1 { [0,1] } else { [1,0] };
                let mut offset = 3;
                for i in order {
                    placements.push(Placement { start:offset, tokens:spans[i].geometry.tokens(), key:keys[i] });
                    refs.push(spans[i].clone()); offset += spans[i].geometry.tokens()+7;
                }
            }
            let mut full = vec![2u32; start+rows+4096];
            for placement in &placements { full[placement.start..placement.start+placement.tokens].fill(IMAGE_TOKEN); }
            for row in 0..live { ids[row] = full[start+row] as i64; }
            let bindings = Rc::new(Bindings::new(&cuda, Identity::new(placements.clone(), &full)?, refs.clone())?);
            token_rows.update(&ids,live)?;
            token_rows.bind_images(bindings,start,&ids,live)?;
            stream.join_default()?;
            expand(&old,text.ptr(),expected.ptr(),rows,None,&stream)?;
            for (placement,span) in placements.iter().zip(&refs) {
                for position in start.max(placement.start)..(start+live).min(placement.start+placement.tokens) {
                    for hc in 0..4 {
                        unsafe { cuda.copy_device(expected.ptr()+(((position-start)*20480+hc*5120)*2)as u64,
                            span.data.ptr()+((position-placement.start)*5120*2)as u64,5120*2,&stream)?; }
                    }
                }
            }
            unsafe { graph.launch(&stream)?; } stream.synchronize()?;
            let actual=output.download()?; let reference=expected.download()?;
            let exact=actual.iter().zip(&reference).all(|(a,b)|a.to_bits()==b.to_bits());
            let record=json!({"rows":rows,"mode":mode,"start":start,"live":live,"exact":exact,
                "same_graph":true,"compared_bf16_words":actual.len()});
            println!("{record}"); records.push(record);
            ensure!(exact,"image embedding injection differs from GPU byte-copy reference");
        }
    }
    let report=json!({"records":records,"all_exact":true,
        "scope":"Image/text selection and four HC copies; graph metadata refresh, swapped image order, partial spans and return to text. No CPU model/reference arithmetic; full target and serving qualification pending."});
    std::fs::write(p.output,serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}
