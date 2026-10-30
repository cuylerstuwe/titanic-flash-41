//! Four-rank, one-layer reduction-overlap experiment. Never a serving policy.
use crate::{cuda::{Buffer,Cuda,Stream},ffn::{LayerFfn,TokenRows},moe::Workspace,
    roce::Roce,ffn_chunk_bridge::Bridge};
use anyhow::{Result,ensure};
use half::bf16;
use serde::Deserialize;
use serde_json::{Value,json};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};

fn bits(b:&Buffer<bf16>)->Result<Vec<u16>>{Ok(b.download()?.into_iter().map(bf16::to_bits).collect())}
fn hash(b:&[u16])->String{format!("{:x}",Sha256::digest(b.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()))}
#[derive(Deserialize)]struct Case{name:String,input_sha256:String,tokens_sha256:String,live:usize}
#[derive(Deserialize)]struct Cases{rows:usize,world:usize,cases:Vec<Case>}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,stream:&Stream,ffn:&LayerFfn,control:&Workspace,
    input:&mut Buffer<bf16>,tokens:&mut TokenRows,output:&mut Buffer<bf16>,comm:&Roce,
    fixture:&Path,global_fixture:&Path)->Result<Value>{
    ensure!(std::env::var("CANADA_SCRATCH_FFN_CHUNK_SCREEN").as_deref()==Ok("1")
        &&std::env::var("CANADA_SCRATCH_FFN_OVERLAP").as_deref()==Ok("1"),"isolated chunk overlap admission");
    let root=std::env::var("CANADA_SCRATCH_FFN_CHUNK_BUNDLE")?;let root=Path::new(&root);
    let contract:Value=serde_json::from_slice(&std::fs::read(root.join("runtime-contract.json"))?)?;
    let patch=&contract["fc2_output_chunk_overlay"];
    let qualification:Value=serde_json::from_slice(&std::fs::read(root.join("qualification.json"))?)?;
    ensure!(contract["kernel_library_sha256"]=="0558b822898c6c305ccc276bd8ebbd03d1f5c2c75276deaf9a91d46839df5c0a"
        &&patch["base_commit"]=="a7d7d29b2ef8869086e0ceaa787321f17544e3c9"
        &&patch["fc2_output_chunks"]==2&&patch["output_major_tasks"]==false
        &&qualification["all_live_exact"]==true&&qualification["cases"].as_array().is_some_and(|v|v.len()==8),
        "s611 exact chunk bundle required");
    let candidate=Workspace::new(cuda,root)?;
    let library=std::env::var("CANADA_SCRATCH_FFN_CHUNK_LIBRARY")?;
    let bridge=Bridge::open(cuda,Path::new(&library))?;
    let low=cuda.allocate::<bf16>(6*2560)?;let high=cuda.allocate::<bf16>(6*2560)?;
    let mut low_sum=cuda.allocate::<bf16>(6*2560)?;let mut high_sum=cuda.allocate::<bf16>(6*2560)?;
    let joined=cuda.allocate::<bf16>(6*5120)?;let local_joined=cuda.allocate::<bf16>(6*5120)?;
    // Graphs are declared after their captured buffer/workspace owners and are
    // released first. Every timed replay is drained before comparison/return.
    let mut graphs=None;
    let cases:Cases=serde_json::from_slice(&std::fs::read(global_fixture.join("index.json"))?)?;
    ensure!(cases.rows==6&&cases.world==4&&cases.cases.len()==4,"FFN chunk fixture shape");
    let mut reports=Vec::new();
    for case in cases.cases {
        ensure!(["initial","changed","zero","padded"].contains(&case.name.as_str())&&case.live<=6,"FFN chunk case");
        let dir=fixture.join(&case.name);
        let bytes=std::fs::read(dir.join("input.bin"))?;let ids=std::fs::read(dir.join("tokens.bin"))?;
        ensure!(bytes.len()==input.bytes()&&ids.len()==48
            &&format!("{:x}",Sha256::digest(&bytes))==case.input_sha256
            &&format!("{:x}",Sha256::digest(&ids))==case.tokens_sha256,"FFN chunk input identity");
        input.upload(&bytes.chunks_exact(2).map(|b|bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap()))).collect::<Vec<_>>())?;
        tokens.update(&ids.chunks_exact(8).map(|b|i64::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>(),case.live)?;
        stream.join_default()?;
        if graphs.is_none(){
            ffn.run(control,input,tokens,stream)?;unsafe{comm.sum_bf16(ffn.output(),output,stream)}?;
            stream.synchronize()?;comm.health()?;
            let original=stream.capture(||{ffn.run(control,input,tokens,stream)?;unsafe{comm.sum_bf16(ffn.output(),output,stream)}})?;
            ffn.run(&candidate,input,tokens,stream)?;stream.synchronize()?;
            let source=bridge.capture(stream,||ffn.run(&candidate,input,tokens,stream))?;
            // Warm the packing and both distinct collective shapes before capture.
            unsafe{bridge.pack(&candidate,ffn.shared_down(),&low,0,stream)?;comm.sum_bf16(&low,&mut low_sum,stream)?;
                bridge.pack(&candidate,ffn.shared_down(),&high,1,stream)?;comm.sum_bf16(&high,&mut high_sum,stream)?;}
            bridge.join(&low_sum,&high_sum,&joined,stream)?;stream.synchronize()?;comm.health()?;
            let first=bridge.capture(stream,||unsafe{bridge.pack(&candidate,ffn.shared_down(),&low,0,stream)?;
                comm.sum_bf16(&low,&mut low_sum,stream)})?;
            let second=bridge.capture(stream,||unsafe{bridge.pack(&candidate,ffn.shared_down(),&high,1,stream)?;
                comm.sum_bf16(&high,&mut high_sum,stream)})?;
            let finish=bridge.capture(stream,||bridge.join(&low_sum,&high_sum,&joined,stream))?;
            let serial=unsafe{bridge.graph(&source,&first,&second,&finish,0,stream)}?;
            let overlap=unsafe{bridge.graph(&source,&first,&second,&finish,1,stream)}?;
            let priority=unsafe{bridge.graph(&source,&first,&second,&finish,2,stream)}?;
            graphs=Some((original,serial,overlap,priority));
        }
        let (original,serial,overlap,priority)=graphs.as_ref().unwrap();
        unsafe{original.launch(stream)}?;stream.synchronize()?;comm.health()?;
        let expected=bits(output)?;let local=bits(ffn.output())?;
        let fixed=std::fs::read(global_fixture.join(&case.name).join("output.bin"))?;
        ensure!(expected.iter().flat_map(|v|v.to_le_bytes()).eq(fixed.iter().copied()),"control differs from canonical FFN reference");
        let mut records=Vec::new();
        for repeat in 0..2 {
            for mode in if repeat==0{[0,1,2,3]}else{[3,2,1,0]} {
                let launch=||unsafe{match mode{0=>original.launch(stream),1=>serial.launch(stream),2=>overlap.launch(stream),_=>priority.launch(stream)}};
                launch()?;stream.synchronize()?;comm.health()?;
                let begin=cuda.event()?;let end=cuda.event()?;let start=Instant::now();
                begin.record(stream)?;for _ in 0..16{launch()?;}end.record(stream)?;
                let gpu_us=end.elapsed_since(&begin)? as f64*1000./16.;
                let wall_us=start.elapsed().as_secs_f64()*1e6/16.;comm.health()?;
                if mode==0 {ensure!(bits(output)?==expected&&bits(ffn.output())?==local,"control repeat changed");}
                else {
                    ensure!(bits(&joined)?==expected,"chunk global reduction mismatch");
                    bridge.join(&low,&high,&local_joined,stream)?;stream.synchronize()?;
                    ensure!(bits(&local_joined)?==local,"chunk top-k/shared BF16 boundary mismatch");
                }
                records.push(json!({"repeat":repeat,"mode":(["original","chunks_serial","chunks_overlap","communication_priority"][mode]),
                    "gpu_us":gpu_us,"wall_us":wall_us,"replays":16,"local_exact":true,"global_exact":true}));
            }
        }
        reports.push(json!({"case":case.name,"live_rows":case.live,"global_sha256":hash(&expected),
            "local_sha256":hash(&local),"source_graph_counts":overlap.counts,"records":records}));
        eprintln!("FFN chunk rank{} {} exact",comm.rank(),case.name);
    }
    stream.synchronize()?;comm.health()?;
    Ok(json!({"rank":comm.rank(),"ffn_chunk_probe":true,"all_bitwise_equal":true,"cases":reports,
        "scope":"One canonical TP4 FFN layer, four existing input/routing fixtures, two opposite-order 16-replay measurements. Includes split compute, BF16-preserving top-k/shared packing, two native RoCE reductions and unpack. Compare original full reduction, serialized chunks, overlapping chunks and prioritized communication. No whole-model throughput claim."}))
}
