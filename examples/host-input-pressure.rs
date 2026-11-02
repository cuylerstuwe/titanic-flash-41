//! CPU-only screen of allocator retention after two canonical long inputs.
use anyhow::{Result, ensure};
use canada_engine::protocol::{Frontend, PreparedChat};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

fn resident() -> Result<Value> {
    let text=std::fs::read_to_string("/proc/self/status")?;
    Ok(json!(text.lines().filter_map(|line| {
        let (name,rest)=line.split_once(':')?;
        matches!(name,"VmRSS"|"RssAnon"|"RssFile"|"VmSwap").then(||
            (name.to_string(),rest.split_whitespace().next().unwrap().parse::<u64>().unwrap()*1024))
    }).collect::<std::collections::BTreeMap<_,_>>()))
}
fn identity(value:&PreparedChat)->Value {
    let mut hash=Sha256::new();for token in &value.tokens {hash.update(token.to_le_bytes());}
    json!({"tokens":value.tokens.len(),"max_new_tokens":value.max_new_tokens,
        "token_sha256":format!("{:x}",hash.finalize()),"rendered_sha256":value.rendered_sha256,
        "images":value.images.len()})
}
fn trim()->Result<Value> {
    let before=resident()?;let began=Instant::now();
    #[cfg(all(target_os="linux",target_env="gnu"))]
    let returned_pages=unsafe{libc::malloc_trim(0)!=0};
    #[cfg(not(all(target_os="linux",target_env="gnu")))]
    let returned_pages=false;
    Ok(json!({"before":before,"after":resident()?,"seconds":began.elapsed().as_secs_f64(),
        "returned_pages":returned_pages}))
}
fn main()->Result<()> {
    let args=std::env::args().collect::<Vec<_>>();ensure!(args.len()==4,"tokenizer input_a input_b");
    let frontend=Frontend::load(Path::new(&args[1]))?;
    let bodies=[std::fs::read(&args[2])?,std::fs::read(&args[3])?];
    let mut records=Vec::new();
    for round in 0..2 {
        let before=resident()?;let began=Instant::now();
        let prepared=std::thread::scope(|scope| {
            let a=scope.spawn(||frontend.prepare_chat(&bodies[0],1048576));
            let b=scope.spawn(||frontend.prepare_chat(&bodies[1],1048576));
            Ok::<_,anyhow::Error>([a.join().expect("input thread")?,b.join().expect("input thread")?])
        })?;
        let preparation_seconds=began.elapsed().as_secs_f64();
        let identities=prepared.iter().map(identity).collect::<Vec<_>>();
        let live_trim=trim()?;
        ensure!(identities==prepared.iter().map(identity).collect::<Vec<_>>(),"live input changed");
        drop(prepared);let retired_trim=trim()?;
        records.push(json!({"round":round,"before":before,"preparation_seconds":preparation_seconds,
            "identities":identities,"live_trim":live_trim,"retired_trim":retired_trim}));
    }
    println!("{}",serde_json::to_string_pretty(&json!({"passed":true,"records":records}))?);
    Ok(())
}
