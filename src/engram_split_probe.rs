//! Paired verification-only screen for a later Engram I/O deadline.
use crate::{cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Context,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};

fn physical_read_bytes()->Result<u64> {
    std::fs::read_to_string("/proc/self/io")?.lines()
        .find_map(|line|line.strip_prefix("read_bytes:"))
        .context("missing physical I/O counter")?.trim().parse().map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,candidate:&mut Target,
    wides:&mut[Target],comm:&Roce,stream:&Stream,output:&Path,rank:usize)->Result<Value> {
    ensure!(std::env::var("CANADA_SCRATCH_ENGRAM_SPLIT14_SCREEN").as_deref()==Ok("1")
        && std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref()==Ok("1")
        && large.context_capacity()==1048576 && wides.iter().any(|t|t.physical_rows()==8192)
        && verify.engram_input_kind()=="mapped" && candidate.engram_input_kind()=="mapped"
        && verify.paired_engram_io_enabled() && candidate.paired_engram_io_enabled()
        && verify.schedule_counts_for_probe()==candidate.schedule_counts_for_probe(),
        "split14 screen requires current mapped/paired full-capacity profiles");
    let raw=include_bytes!("../tests/fixtures/long-retained-prefill-v1.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let frontier=131072;
    let adaptive=std::env::var("CANADA_SCRATCH_ENGRAM_ADAPTIVE_SPLIT14_SCREEN").as_deref()==Ok("1");
    let regimes=if adaptive {vec!["application_and_file_cold","application_warm","rank3_application_cold"]}
        else {vec!["application_and_file_cold","application_warm"]};
    let iterations=if adaptive {3}else{5};
    let mut barrier_in=cuda.allocate::<f32>(4)?;barrier_in.upload(&[0.;4])?;
    let mut barrier_out=cuda.allocate::<f32>(16)?;
    let prefill=Instant::now();
    large.prefill_wide_with_prefetch(wides,&tokens[..frontier],comm,stream,true)?;
    let prefill_seconds=prefill.elapsed().as_secs_f64();
    verify.handoff_from(large)?;
    let carry=verify.carry_digests()?;let encoder=verify.encoder_tail_digest()?;
    let mut references=[None,None];let mut records=Vec::new();
    for regime in regimes {
        for (depth_index,live) in [3,6].into_iter().enumerate() {
            for iteration in 0..iterations {
                for path in [iteration%2,1-iteration%2] {
                    if path==1 {candidate.handoff_from(verify)?;}
                    let target=if path==1 {&mut *candidate}else{&mut *verify};
                    if regime=="application_and_file_cold" {
                        #[cfg(target_os="linux")]
                        target.evict_engram_file_pages_for_probe(stream)?;
                        #[cfg(not(target_os="linux"))]
                        anyhow::bail!("split14 cold screen requires Linux");
                    }else if regime=="rank3_application_cold" && rank==3 {
                        target.clear_engram_cache_for_probe(stream)?;
                    }
                    let before=target.engram_cache_stats();
                    let reads=physical_read_bytes()?;
                    stream.join_default()?;
                    unsafe{comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)?;}
                    stream.synchronize()?;comm.health()?;
                    let start=Instant::now();
                    target.begin_verification(&tokens[frontier..frontier+live],comm,stream)?;
                    let predictions=target.verification_greedy(stream)?;
                    let ms=start.elapsed().as_secs_f64()*1000.;
                    let read_bytes=physical_read_bytes()?.checked_sub(reads).context("I/O counter regressed")?;
                    let phases=serde_json::to_value(target.verification_metrics()?)?;
                    let (logits,rows)=target.pending_logits()?;
                    ensure!(rows==live,"split14 live logits geometry");
                    let digest=format!("{:x}",Sha256::digest(unsafe{cuda.read_device(logits.ptr(),live*129280*4)}?));
                    let signature=json!({"logits":digest,"predictions":predictions});
                    let exact=references[depth_index].as_ref().is_none_or(|r|r==&signature);
                    if references[depth_index].is_none(){references[depth_index]=Some(signature.clone());}
                    target.finish_verification(0,comm,stream)?;
                    let rollback_exact=target.position() as usize==frontier && target.carry_digests()?==carry
                        && target.encoder_tail_digest()?==encoder;
                    let record=json!({"regime":regime,"live_rows":live,"iteration":iteration,"warmup":iteration==0,
                        "split14":path==1,"milliseconds":ms,"physical_read_bytes":read_bytes,"phases":phases,
                        "row_cache_before":before,"row_cache_after":target.engram_cache_stats(),
                        "capture_statistics":target.capture_statistics(),"signature":signature,
                        "same_signature":exact,"rollback_exact":rollback_exact});
                    std::fs::write(output.join(format!("engram-split14-{regime}-m{live}-i{iteration}-v{path}.json")),
                        serde_json::to_vec_pretty(&record)?)?;
                    ensure!(exact&&rollback_exact,"split14 changed checked model outputs/state; saved evidence");
                    if path==1 {verify.handoff_from(candidate)?;}
                    records.push(record);
                }
            }
            eprintln!("split14 rank{rank} {regime} M{live}: {iterations} pairs complete");
        }
    }
    Ok(json!({"rank":rank,"split14_screen":true,"adaptive_split14":adaptive,"all_exact":true,"frontier":frontier,"records":records,
        "prefill_seconds":prefill_seconds,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "memory":cuda.admission_memory()?,"full_engine_acceptance":false,
        "scope":if adaptive {
            "One128K public frontier/current1Mcapacity. Current two-graph paired reads versus rank-local cache-miss-dependent split. M3/M6, one excluded warmup and two order-balanced measured pairs per regime. All-rank file/application cold, all warm, and rank3-only application cold exercise uniform and asymmetric graph boundaries. Exact logits, predictions, rollback and collective order; no generation or production claim."
        }else{
            "One128Kpublicfrontier/current1Mcapacity. Current two-graph paired reads versus three graphs with layer14 read overlapping layers1..13. M3/M6, one excluded warmup and four order-balanced pairs per cache regime. Cold clears application tags and advises file-page eviction; actual physical reads retained. Full live logits, predictions and rollback state checked. Same precision and all other schedules; no generation or production claim."
        }}))
}
