//! Small current-bank cache-lookahead screen. Warm graph/model state is kept
//! separate from application row-cache and advisory file-page-cache state.
use crate::{cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Context,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};

fn physical_read_bytes()->Result<u64> {
    let text=std::fs::read_to_string("/proc/self/io")?;
    text.lines().find_map(|line|line.strip_prefix("read_bytes:"))
        .context("missing physical I/O counter")?.trim().parse().map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,output:&Path,rank:usize)->Result<Value> {
    ensure!(std::env::var("CANADA_SCRATCH_ENGRAM_COLD_SCREEN").as_deref()==Ok("1")
        && std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref()==Ok("1")
        && std::env::var_os("CANADA_SCRATCH_ENGRAM_PREFETCH").is_none()
        && large.context_capacity()==1048576 && wides.iter().any(|t|t.physical_rows()==8192),
        "cold Engram screen needs current1M/8Kbank and explicit lookahead policy");
    let raw=include_bytes!("../tests/fixtures/long-retained-prefill-v1.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let inputs=&tokens[..65536];
    let mut barrier_in=cuda.allocate::<f32>(4)?;barrier_in.upload(&[0.;4])?;
    let mut barrier_out=cuda.allocate::<f32>(16)?;
    let mut initial=true;let mut lineage=510000;let mut records=Vec::new();
    let mut reference=None;
    for (regime,pairs) in [("warmup_uncontrolled",1),("application_and_file_cold",2),("application_warm",2)] {
        for pair in 0..pairs {
            for path in [pair%2,1-pair%2] {
                if !initial {lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;}
                initial=false;
                ensure!(large.position()==0,"cache screen fresh frontier");
                if regime=="application_and_file_cold" {
                    #[cfg(target_os="linux")]
                    large.evict_engram_file_pages_for_probe(stream)?;
                    #[cfg(not(target_os="linux"))]
                    anyhow::bail!("cold Engram screen requires Linux");
                }
                let before=large.engram_cache_stats();let reads=physical_read_bytes()?;
                stream.join_default()?;
                unsafe{comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)?;}
                stream.synchronize()?;comm.health()?;
                let start=Instant::now();
                large.prefill_wide_with_prefetch(wides,inputs,comm,stream,path==1)?;
                let anchor=large.greedy_token(stream)?;let seconds=start.elapsed().as_secs_f64();
                let read_bytes=physical_read_bytes()?.checked_sub(reads).context("I/O counter regressed")?;
                let phases=serde_json::to_value(large.append_metrics()?)?;
                let after=large.engram_cache_stats();
                let calls=after.as_array().unwrap().iter().map(|r|r["prefetch_calls"].as_u64().unwrap()).sum::<u64>()
                    -before.as_array().unwrap().iter().map(|r|r["prefetch_calls"].as_u64().unwrap()).sum::<u64>();
                ensure!((path==0 && calls==0)||(path==1 && calls==14),"lookahead chunk/layer call count");
                let carry=large.carry_digests()?;let encoder=large.encoder_tail_digest()?;
                let logits=large.last_prompt_logit_digest()?;
                verify.handoff_from(large)?;
                verify.begin_verification(&tokens[65536..65542],comm,stream)?;
                let (pending,live)=verify.pending_logits()?;ensure!(live==6,"following verify rows");
                let next=format!("{:x}",Sha256::digest(unsafe{cuda.read_device(pending.ptr(),6*129280*4)}?));
                verify.finish_verification(0,comm,stream)?;
                let rollback_exact=verify.position()==65536 && verify.carry_digests()?==carry
                    && verify.encoder_tail_digest()?==encoder;
                let signature=json!({"anchor":anchor,"logits":logits,"following_logits":next,"carry":carry,"encoder":encoder});
                let exact=reference.as_ref().is_none_or(|r|r==&signature);
                if reference.is_none(){reference=Some(signature.clone());}
                let record=json!({"regime":regime,"pair":pair,"lookahead":path==1,"warmup":regime=="warmup_uncontrolled",
                    "prompt_tokens":65536,"seconds":seconds,"physical_read_bytes":read_bytes,
                    "prefetch_batch_rows":std::env::var("CANADA_SCRATCH_ENGRAM_PREFETCH_BATCH_ROWS").unwrap_or_else(|_|"256".into()),
                    "phases":phases,"row_cache_before":before,"row_cache_after":after,"prefetch_calls":calls,
                    "signature":signature,"same_signature":exact,"rollback_exact":rollback_exact});
                std::fs::write(output.join(format!("engram-cold-{regime}-p{pair}-v{path}.json")),serde_json::to_vec_pretty(&record)?)?;
                ensure!(exact&&rollback_exact,"cache-only lookahead changed checked model state/output; saved evidence");
                eprintln!("cold Engram rank{rank} {regime} pair{pair} lookahead{}: {seconds:.4}s read_bytes{read_bytes}",path==1);
                records.push(record);
            }
        }
    }
    Ok(json!({"rank":rank,"all_exact":true,"cold_storage_screen":true,"records":records,
        "fixture_sha256":format!("{:x}",Sha256::digest(raw)),"memory":cuda.admission_memory()?,
        "full_engine_acceptance":false,"scope":"One64Kpublicprefix/current1Mcapacity/full8Kbank. Two warmup paths then two order-balanced pairs per application+file-cold and application-warm regime. DONTNEED applies only to immutable Engram shard data; physicalreadbytes reveal actual disk activity. Sameprecision/fullfinal logits/carry/encoder and six following verificationrows checked. No serving or outcome qualification."}))
}
