//! Small cost/rollback screen for a16-row history verifier. All model arithmetic
//! stays on GPU; raw FP32 logits go to a separate bounded GPU comparison.
use crate::{cuda::{Cuda,Stream},history_lookup::HistoryLookup,roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde::Deserialize;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};

#[derive(Deserialize)]
struct Case {name:String,prompt_ids:Vec<u32>,anchor:u32,proposal_ids:Vec<u32>,prefix_match:usize,source_end:usize}
#[derive(Deserialize)]
struct Fixture {schema:String,cases:Vec<Case>}
fn digest(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}
fn pending(cuda:&Cuda,target:&Target,stream:&Stream)->Result<Vec<u8>> {
    let (logits,live)=target.pending_logits()?;stream.synchronize()?;
    unsafe{cuda.read_device(logits.ptr(),live*129280*4)}
}
fn follow(cuda:&Cuda,verify:&mut Target,wide:&mut Target,ids:&[u32],comm:&Roce,stream:&Stream)->Result<(Vec<u8>,Value)> {
    verify.handoff_from(wide)?;verify.begin_verification(ids,comm,stream)?;
    let bytes=pending(cuda,verify,stream)?;verify.finish_verification(ids.len(),comm,stream)?;
    Ok((bytes,json!({"carry":verify.carry_digests()?,"encoder":verify.encoder_tail_digest()?,"position":verify.position()})))
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wide:&mut Target,wides:&mut[Target],
    comm:&Roce,stream:&Stream,output:&Path,rank:usize,ticket:&[u8],kernels:&Path)->Result<Value> {
    ensure!(std::env::var("CANADA_SCRATCH_LONG_HISTORY_VERIFY").as_deref()==Ok("1")
        && large.position()==0 && large.context_capacity()==1048576
        && verify.physical_rows()==6 && wide.physical_rows()==16 && wides.len()==5,
        "isolated16-row verifier screen geometry");
    let long_context=std::env::var("CANADA_SCRATCH_LONG_HISTORY_CONTEXT_SCREEN").as_deref()==Ok("1");
    let live_widths=std::env::var("CANADA_SCRATCH_VERIFY16_LIVE_MOE").as_deref()==Ok("1");
    let compact_dense=std::env::var("CANADA_SCRATCH_VERIFY16_COMPACT_DENSE").as_deref()==Ok("1");
    let small_kernels=std::env::var("CANADA_SCRATCH_VERIFY16_SMALL_KERNELS").as_deref()==Ok("1");
    let profile_widths=std::env::var("CANADA_SCRATCH_PROFILE_VERIFY_WIDTHS").as_deref()==Ok("1");
    let index16_screen=std::env::var("CANADA_SCRATCH_CACHED_INDEX16_SCREEN").as_deref()==Ok("1");
    let index16_only=std::env::var("CANADA_SCRATCH_INDEX16_ONLY").as_deref()==Ok("1");
    let handoff_schedules=std::env::var("CANADA_SCRATCH_HANDOFF_SCHEDULES").as_deref()==Ok("1");
    ensure!(!index16_only || index16_screen,"index-only screen requires cache preparation");
    ensure!(!handoff_schedules || (index16_screen&&!index16_only),"handoff schedule isolation requires the cache/overlap screen");
    let overlap_screen=std::env::var("CANADA_SCRATCH_VERIFY16_OVERLAP_SCREEN").as_deref()==Ok("1");
    let attention_screen=std::env::var("CANADA_SCRATCH_VERIFY16_ATTN_ONLY").as_deref()==Ok("1");
    let grouped16_screen=std::env::var("CANADA_SCRATCH_GROUPED_SPARSE16_SCREEN").as_deref()==Ok("1");
    let legacy_ffn=std::env::var("CANADA_SCRATCH_LEGACY16_FFN_SCREEN").as_deref()==Ok("1");
    let shared_quant16=std::env::var("CANADA_SCRATCH_FUSED_SHARED_QUANT16").as_deref()==Ok("1");
    ensure!(!shared_quant16 || wide.fused_shared_quantization_layers()==40,"M16 shared fusion must cover all layers");
    // Large host snapshots perturb inter-rank pacing. Opt in only to localize
    // a failure; ordinary overlap screens preserve the original replay pace.
    let layer_taps=(overlap_screen||grouped16_screen) && std::env::var("CANADA_SCRATCH_REPLAY_LAYER_TAPS").as_deref()==Ok("1");
    let woa_cache_screen=std::env::var("CANADA_SCRATCH_WOA_CACHE_SCREEN").as_deref()==Ok("1");
    ensure!(!woa_cache_screen || (long_context&&live_widths&&compact_dense&&small_kernels&&!profile_widths),
        "WO-A cache screen requires the isolated long-context verifier");
    ensure!(!overlap_screen || (long_context&&live_widths&&compact_dense&&small_kernels&&!profile_widths&&!woa_cache_screen),
        "M16 overlap screen requires the isolated long-context verifier");
    ensure!(!index16_screen || (overlap_screen && wide.cached_index_gather_layers()==8),"M16 index-cache screen needs all eight prepared dense gather caches");
    ensure!(!grouped16_screen || (long_context&&live_widths&&compact_dense&&small_kernels
        && !profile_widths&&!woa_cache_screen&&(!overlap_screen||legacy_ffn)&&!index16_screen&&!handoff_schedules),
        "M16 compact-index screen requires otherwise unchanged native schedules");
    let paired_screen=woa_cache_screen||overlap_screen||grouped16_screen;
    let cached_woa=wide.cached_woa_allocations();
    if woa_cache_screen {
        ensure!(cached_woa.len()==40 && large.cached_woa_allocations()==cached_woa
            && wides.iter().all(|p|p.cached_woa_allocations()==cached_woa[..21]),
            "medium profiles must share forty WO-A allocations; wide encoder profiles share the first21");
    }
    ensure!(!profile_widths || (long_context&&live_widths&&compact_dense&&small_kernels),
        "width profiling requires the accumulated isolated long-context verifier");
    ensure!(!small_kernels || (compact_dense && wide.query_norm_quant_layers()==40),
        "M16 query fusion must cover all forty layers");
    let legacy_qkv=std::env::var("CANADA_SCRATCH_VERIFY16_LEGACY_QKV").as_deref()==Ok("1");
    ensure!(!legacy_qkv || grouped16_screen,"QKV fallback is a diagnostic-only candidate");
    ensure!(!compact_dense || (live_widths && wide.narrow_projection_layers()==if legacy_qkv{0}else{40}),
        "M16 compact dense must cover all forty layers");
    ensure!(!live_widths || long_context,"live widths require the isolated long-context screen");
    let raw:&[u8]=if long_context {include_bytes!("../tests/fixtures/long-history-context-v1.json")}
        else {include_bytes!("../tests/fixtures/long-history-score-v1.json")};
    let fixture:Fixture=serde_json::from_slice(raw)?;
    ensure!(if long_context {fixture.schema=="long-history-context-v1" && fixture.cases.len()==1}
        else {fixture.schema=="long-history-score-v1" && fixture.cases.len()==15},"frozen causal fixture");
    let case=&fixture.cases[if long_context {0}else{8}];
    ensure!(case.prefix_match==64 && case.proposal_ids.len()==15
        && if long_context {case.prompt_ids.len()==274108}else{case.prompt_ids.len()%32==22},
        "strong known-prefix context/carry geometry");
    let lookup=HistoryLookup::with_maximum(1048576,&case.prompt_ids,262144,64)?;
    let proposal=lookup.propose(lookup.position(),case.anchor)?.ok_or_else(||anyhow::anyhow!("history proposal missing"))?;
    ensure!(proposal.matched==case.prefix_match && proposal.source_end==case.source_end
        && case.prompt_ids[case.source_end+1..case.source_end+16]==case.proposal_ids,
        "proposal must come only from committed history");
    let mut inputs=vec![case.anchor];inputs.extend_from_slice(&case.proposal_ids);
    let directory=output.file_name().and_then(|s|s.to_str()).unwrap_or("");
    let prefix=format!("target-native-r{rank}-");
    let version=directory.strip_prefix(&prefix).ok_or_else(||anyhow::anyhow!("verify output identity"))?;
    ensure!(version.starts_with('s') && version[1..].bytes().all(|b|b.is_ascii_digit()),"verify version");
    let raw_name=if rank==3 {format!("long-history-verify-{version}")}
        else {format!("long-history-verify-r{rank}-{version}")};
    let raw_root=output.parent().unwrap().join(raw_name);
    if rank==3 || layer_taps {std::fs::create_dir(&raw_root)?;}
    eprintln!("long history verify rank{rank}: profiles ready, prefill{}tokens",case.prompt_ids.len());
    large.enable_cached_extend_floor()?;
    large.prefill_wide(wides,&case.prompt_ids,comm,stream)?;
    verify.handoff_from(large)?;
    verify.begin_verification(&inputs[..6],comm,stream)?;verify.finish_verification(0,comm,stream)?;
    wide.handoff_from(verify)?;
    if legacy_ffn {
        return crate::legacy16_ffn_probe::run(cuda,verify,wide,&inputs,kernels,comm,stream,output,rank,ticket);
    }
    if std::env::var("CANADA_SCRATCH_NATIVE_ROW_TRANSPORT_SCREEN").as_deref()==Ok("1") {
        return crate::native_row_transport_probe::run(cuda,large,verify,wide,&inputs,comm,stream,output,rank,ticket);
    }
    if std::env::var("CANADA_SCRATCH_QKV_SCHEDULE_SCREEN").as_deref()==Ok("1") {
        return crate::qkv_schedule_probe::run(cuda,verify,wide,&inputs,kernels,comm,stream,output,rank,ticket);
    }
    wide.begin_verification(&inputs,comm,stream)?;
    let warm_logits=if live_widths {Some(pending(cuda,wide,stream)?)}else{None};
    wide.finish_verification(0,comm,stream)?;
    let mut width_checks=Vec::new();
    if let Some(reference)=warm_logits {
        for rows in if profile_widths||paired_screen{vec![8]}else{vec![8,10,12]} {
            let before=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
            wide.begin_verification(&inputs[..rows],comm,stream)?;
            let logits=pending(cuda,wide,stream)?;
            wide.finish_verification(0,comm,stream)?;
            let after=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
            ensure!(logits==reference[..rows*129280*4] && before==after,
                "intermediate width changed live logits or aborted state");
            width_checks.push(json!({"rows":rows,"logits_exact":true,"aborted_state_exact":true,"logits_sha256":digest(&logits)}));
        }
    }
    eprintln!("long history verify rank{rank}: M6 and M16graphs warm");
    let mut current_wide=true;
    let mut timings=Vec::new();
    let mut abort_checks=Vec::new();
    let mut profile_ranges=Vec::new();
    let mut woa_timings=Vec::new();
    if paired_screen && !handoff_schedules {
        for repeat in 0..4 {
            let mut order=if index16_only {vec![0,2]}else if index16_screen {vec![0,1,2,3]}else{vec![0,1]};
            if index16_screen {order.rotate_left(repeat/2);}
            if repeat%2==1 {order.reverse();}
            for mode in order {
                let enabled=mode&1!=0;
                if index16_screen {wide.set_cached_index16_for_probe(mode&2!=0,stream)?;}
                if grouped16_screen {wide.set_grouped16_for_probe(enabled,stream)?;}
                else if overlap_screen { wide.set_verify16_overlap_for_probe(enabled,stream)?; }
                else { wide.set_woa_cache_for_probe(enabled,stream)?; }
                for rows in [8,16] {
                    // Capture/warm each binding before measuring replay.
                    wide.begin_verification(&inputs[..rows],comm,stream)?;
                    let reference=pending(cuda,wide,stream)?;
                    let reference_taps=if layer_taps {Some(replay_taps(cuda,wide)?)}else{None};
                    wide.finish_verification(0,comm,stream)?;
                    let before=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
                    let mut seconds=Vec::new();
                    for _ in 0..2 {
                        let start=Instant::now();wide.begin_verification(&inputs[..rows],comm,stream)?;
                        seconds.push(start.elapsed().as_secs_f64());
                        let logits=pending(cuda,wide,stream)?;
                        wide.finish_verification(0,comm,stream)?;
                        if logits!=reference {
                            let first=logits.chunks_exact(4).zip(reference.chunks_exact(4)).position(|(a,b)|a!=b);
                            let changed=logits.chunks_exact(4).zip(reference.chunks_exact(4)).filter(|(a,b)|a!=b).count();
                            let failure=json!({"rank":rank,"repeat":repeat,"mode":mode,"rows":rows,
                                "changed_fp32_values":changed,"first_changed_value":first,
                                "reference_sha256":digest(&reference),"observed_sha256":digest(&logits),
                                "completed_pairs":woa_timings});
                            eprintln!("paired verifier mismatch repeat{repeat} mode{mode} rows{rows}: {changed} FP32 values, first {first:?}");
                            let observed_taps=if layer_taps {Some(replay_taps(cuda,wide)?)}else{None};
                            if rank==3 || layer_taps {
                                std::fs::write(raw_root.join("failure-reference.bin"),&reference)?;
                                std::fs::write(raw_root.join("failure-observed.bin"),&logits)?;
                                std::fs::write(raw_root.join("failure.json"),serde_json::to_vec_pretty(&failure)?)?;
                                if let(Some(reference_taps),Some(observed_taps))=(&reference_taps,&observed_taps) {
                                    let mut differences=Vec::new();
                                    for((name,a),(other,b))in reference_taps.iter().zip(observed_taps) {
                                        ensure!(name==other && a.len()==b.len(),"replay diagnostic layout");
                                        if a!=b {differences.push(json!({"name":name,"changed_bytes":a.iter().zip(b).filter(|(a,b)|a!=b).count(),
                                            "reference_sha256":digest(a),"observed_sha256":digest(b)}));}
                                    }
                                    if let Some(first)=differences.first() {
                                        let name=first["name"].as_str().unwrap();let layer=name.split('-').next().unwrap();
                                        let layer_number=layer.strip_prefix('l').unwrap().parse::<usize>()?;
                                        let producer=[24,28,32,36].into_iter().filter(|i|*i<layer_number).next_back();
                                        let index_prefix=producer.map(|i|format!("l{i:02}-index-"));
                                        for((name,a),(_,b))in reference_taps.iter().zip(observed_taps).filter(|((n,_),_)|
                                            n.starts_with(&format!("{layer}-"))||index_prefix.as_ref().is_some_and(|p|n.starts_with(p))) {
                                            std::fs::write(raw_root.join(format!("failure-{name}-reference.bin")),a)?;
                                            std::fs::write(raw_root.join(format!("failure-{name}-observed.bin")),b)?;
                                        }
                                    }
                                    std::fs::write(raw_root.join("failure-layer-taps.json"),serde_json::to_vec_pretty(&differences)?)?;
                                }

                            }
                            anyhow::bail!("Paired verifier replay changed logits: repeat{repeat} mode{mode} rows{rows} changed{changed}");
                        }

                    }
                    let after=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
                    ensure!(before==after,"Paired verifier replay changed aborted state");
                    eprintln!("paired verifier complete: repeat{repeat} mode{mode} rows{rows}");
                    woa_timings.push(json!({"repeat":repeat,"mode":mode,"cached":enabled,"enabled":enabled,"rows":rows,
                        "forward_seconds":seconds,"query_overlap_layers":wide.query_overlap_layers(),
                        "attention_coefficient_overlap_layers":wide.attention_coefficient_overlap_layers(),
                        "logits_sha256":digest(&reference),"aborted_state_exact":true}));
                }
            }
        }
        for rows in [8,16] {
            let hashes=woa_timings.iter().filter(|r|r["rows"]==rows)
                .map(|r|r["logits_sha256"].as_str().unwrap()).collect::<std::collections::BTreeSet<_>>();
            ensure!(hashes.len()==1,"Paired verifier full-model logits differ");
        }
        if index16_screen {wide.set_cached_index16_for_probe(true,stream)?;}
        if grouped16_screen {wide.set_grouped16_for_probe(true,stream)?;}
        else if overlap_screen { wide.set_verify16_overlap_for_probe(!index16_only,stream)?; }
        else { wide.set_woa_cache_for_probe(true,stream)?; }
    }
    let mut control=if profile_widths {
        let ips=std::array::from_fn(|r|std::net::IpAddr::V4(std::net::Ipv4Addr::new(192,168,200,12+r as u8)));
        Some(crate::control_channel::ControlChannel::connect(rank,
            std::net::SocketAddr::new(ips[0],18089),ips,ticket,&Sha256::digest(raw).into(),
            std::time::Duration::from_secs(60))?)
    }else{None};
    for repeat in 0..if paired_screen{0}else if profile_widths{2}else if long_context {3}else{6} {
        let widths=if profile_widths{vec![6,8]}else if live_widths {vec![6,8,10,12,16]}else{vec![6,16]};
        let order:Vec<_>=(0..widths.len()).map(|i|widths[(i+repeat)%widths.len()]).collect();
        for rows in order {
            let is_wide=rows>6;
            if is_wide && !current_wide {wide.handoff_from(verify)?;}
            if !is_wide && current_wide {verify.handoff_from(wide)?;}
            let target=if is_wide {&mut *wide} else {&mut *verify};
            let before=json!({"carry":target.carry_digests()?,"encoder":target.encoder_tail_digest()?,"position":target.position()});
            let profile=profile_widths&&repeat==1;
            if profile {
                stream.synchronize()?;control.as_mut().unwrap().agree(&[rows as u8,0])?;
                cuda.profiler_start()?;
                if let Err(error)=control.as_mut().unwrap().agree(&[rows as u8,1]) {
                    cuda.profiler_stop()?;return Err(error);
                }
            }
            let start=Instant::now();let result=target.begin_verification(&inputs[..rows],comm,stream);
            let forward=start.elapsed().as_secs_f64();
            if profile {
                stream.synchronize()?;cuda.profiler_stop()?;
                control.as_mut().unwrap().agree(&[rows as u8,2])?;
                profile_ranges.push(rows);
            }
            result?;
            target.finish_verification(0,comm,stream)?;
            timings.push(json!({"repeat":repeat,"rows":rows,"profiled":profile,"forward_seconds":forward,"including_abort_seconds":start.elapsed().as_secs_f64()}));
            let after=json!({"carry":target.carry_digests()?,"encoder":target.encoder_tail_digest()?,"position":target.position()});
            ensure!(before==after,"verification abort changed retained state");
            abort_checks.push(json!({"repeat":repeat,"rows":rows,"exact":true}));
            current_wide=is_wide;
        }
    }
    eprintln!("long history verify rank{rank}: warm timings{timings:?}");
    if !current_wide {wide.handoff_from(verify)?;}
    wide.begin_verification(&inputs,comm,stream)?;
    let logits16=pending(cuda,wide,stream)?;
    let rollback=if long_context {
        wide.finish_verification(0,comm,stream)?;
        verify.handoff_from(wide)?;
        json!({"all_timed_aborts_exact":true,"checks":abort_checks,"partial_commit_repeated":false})
    } else {
    let restored=wide.finish_verification(7,comm,stream)?;
    let partial_state=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
    let (follow_full,follow_state)=follow(cuda,verify,wide,&inputs[7..13],comm,stream)?;
    verify.reset(856700,comm,stream)?;large.handoff_from(verify)?;

    large.prefill_wide(wides,&case.prompt_ids,comm,stream)?;wide.handoff_from(large)?;
    wide.begin_verification(&inputs[..7],comm,stream)?;
    let prefix7=pending(cuda,wide,stream)?;wide.finish_verification(7,comm,stream)?;
    let truncated_state=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
    let (follow_short,follow_short_state)=follow(cuda,verify,wide,&inputs[7..13],comm,stream)?;
    eprintln!("long history verify rank{rank}: partial and truncated commits complete");
    let rollback=json!({"accepted":7,"proposed_inputs":16,"restored_bytes":restored,
        "accepted_logits_exact":prefix7==logits16[..7*129280*4],"accepted_state_exact":partial_state==truncated_state,
        "following_six_logits_exact":follow_full==follow_short,"following_state_exact":follow_state==follow_short_state,
        "accepted_full":partial_state,"accepted_truncated":truncated_state,
        "following_full":follow_state,"following_truncated":follow_short_state});
    verify.reset(856701,comm,stream)?;large.handoff_from(verify)?;
    large.prefill_wide(wides,&case.prompt_ids,comm,stream)?;verify.handoff_from(large)?;
    rollback
    };
    let mut logits6=Vec::new();
    for offset in (0..16).step_by(6) {
        let end=(offset+6).min(16);verify.begin_verification(&inputs[offset..end],comm,stream)?;
        logits6.extend_from_slice(&pending(cuda,verify,stream)?);
        verify.finish_verification(end-offset,comm,stream)?;
    }
    ensure!(logits6.len()==logits16.len() && logits6.len()==16*129280*4,"conditional logit extent");
    if live_widths {
        ensure!(digest(&logits16)=="9cda10a267972729ba215eb3632a973007a5bc466e798fed5e8d2f8b98c63d2c"
            && digest(&logits6)=="6f2621cfe6be7683184ca3ca3e1684ae449adcbbdaf1fa65b405dbeba4bf6b49",
            "M6/M16 fixed-prefix logits must match the independent s574 run");
    }
    let index16_handoff=if handoff_schedules {
        wide.handoff_from(verify)?;
        let mut checks=Vec::new();
        for repeat in 0..2 {
            let modes=if repeat==0 {vec![0u8,1,2,3]}else{vec![3u8,2,1,0]};
            for mask in modes {
                wide.set_cached_index16_for_probe(true,stream)?;
                wide.set_verify16_schedule_for_probe(mask,stream)?;
                // Capture at sixteen rows, then change the accepted history
                // through M6 and replay at eight rows without recapturing.
                wide.begin_verification(&inputs,comm,stream)?;
                wide.finish_verification(0,comm,stream)?;
                verify.handoff_from(wide)?;
                // Sixteen commits cover both sides of the 32-row carry wrap
                // across the reversed cases, matching the earlier failing gap.
                for offset in (0..16).step_by(6) {
                    let end=(offset+6).min(16);
                    verify.begin_verification(&inputs[offset..end],comm,stream)?;
                    verify.finish_verification(end-offset,comm,stream)?;
                }
                wide.handoff_from(verify)?;
                let before=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
                wide.begin_verification(&inputs[..8],comm,stream)?;
                let cached=pending(cuda,wide,stream)?;wide.finish_verification(0,comm,stream)?;
                wide.set_cached_index16_for_probe(false,stream)?;
                wide.begin_verification(&inputs[..8],comm,stream)?;
                let control=pending(cuda,wide,stream)?;wide.finish_verification(0,comm,stream)?;
                let after=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
                let logits_exact=cached==control;let state_exact=before==after;
                let changed=cached.chunks_exact(4).zip(control.chunks_exact(4)).filter(|(a,b)|a!=b).count();
                let record=json!({"repeat":repeat,"mask":mask,"shared_expert":mask&1!=0,"ffn_stats":mask&2!=0,
                    "logits_exact":logits_exact,"state_exact":state_exact,"changed_fp32_values":changed,"advanced_tokens":16,
                    "cached_sha256":digest(&cached),"control_sha256":digest(&control),"before":before,"after":after});
                eprintln!("handoff schedule repeat{repeat} mask{mask}: logits={logits_exact} state={state_exact} changed={changed}");
                checks.push(record);
                if rank==3 {
                    std::fs::write(raw_root.join("handoff-schedules.json"),serde_json::to_vec_pretty(&checks)?)?;
                    if !logits_exact {
                        std::fs::write(raw_root.join(format!("handoff-r{repeat}-m{mask}-cached.bin")),&cached)?;
                        std::fs::write(raw_root.join(format!("handoff-r{repeat}-m{mask}-control.bin")),&control)?;
                    }
                }
                ensure!(state_exact,"handoff schedule changed retained state");
            }
        }
        let exact=checks.iter().all(|c|c["logits_exact"]==true);
        ensure!(exact,"handoff schedule comparison changed logits; independent component masks saved");
        json!({"new_history_logits_exact":true,"aborted_state_exact":true,"schedule_checks":checks})
    }else if index16_screen || attention_screen || grouped16_screen {
        if rank==3 {
            std::fs::write(raw_root.join("paired-timings.json"),serde_json::to_vec_pretty(&woa_timings)?)?;
        }
        // M6 just advanced the accepted history. Exercise real profile-handoff
        // invalidation before any diagnostic setter can invalidate the cache.
        wide.handoff_from(verify)?;
        let before=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
        wide.begin_verification(&inputs[..8],comm,stream)?;
        let cached=pending(cuda,wide,stream)?;wide.finish_verification(0,comm,stream)?;
        if grouped16_screen {wide.set_grouped16_for_probe(false,stream)?;}
        else if attention_screen {wide.set_verify16_overlap_for_probe(false,stream)?;}
        else {wide.set_cached_index16_for_probe(false,stream)?;}
        wide.begin_verification(&inputs[..8],comm,stream)?;
        let control=pending(cuda,wide,stream)?;wide.finish_verification(0,comm,stream)?;
        let after=json!({"carry":wide.carry_digests()?,"encoder":wide.encoder_tail_digest()?,"position":wide.position()});
        let logits_exact=cached==control;let state_exact=before==after;
        let changed=cached.chunks_exact(4).zip(control.chunks_exact(4)).filter(|(a,b)|a!=b).count();
        let check=json!({"rank":rank,"logits_exact":logits_exact,"state_exact":state_exact,"changed_fp32_values":changed,
            "cached_sha256":digest(&cached),"control_sha256":digest(&control),"before":before,"after":after,
            "index_only":index16_only,"attention_schedule":attention_screen,"grouped16":grouped16_screen});
        if rank==3 {
            std::fs::write(raw_root.join("handoff-check.json"),serde_json::to_vec_pretty(&check)?)?;
            if !logits_exact {
                std::fs::write(raw_root.join("handoff-cached.bin"),&cached)?;
                std::fs::write(raw_root.join("handoff-control.bin"),&control)?;
            }
        }
        ensure!(logits_exact && state_exact,"M16 schedule/cache handoff: logits_exact={logits_exact},state_exact={state_exact},changed={changed}");
        json!({"new_history_logits_exact":true,"aborted_state_exact":true,"logits_sha256":digest(&cached)})
    }else{json!(null)};
    if rank==3 {
        std::fs::write(raw_root.join("m16.bin"),&logits16)?;
        std::fs::write(raw_root.join("m6.bin"),&logits6)?;
    }
    let result=json!({"rank":rank,"long_history_verify":true,"long_context":long_context,"case":case.name,"prompt_tokens":case.prompt_ids.len(),
        "fixture_sha256":digest(raw),"prefix_match":case.prefix_match,"proposal_ids":case.proposal_ids,
        "carry_ring_rows":32,"target_head":"canonical_bf16_weights_fp32_output",
        "logits":{"m16":digest(&logits16),"m6":digest(&logits6),"bytes_each":logits16.len()},
        "rollback":rollback,"timings":timings,"raw_directory":raw_root,"raw_rank":3,
        "live_moe_widths":live_widths,"width_checks":width_checks,
        "compact_dense":compact_dense,"compact_dense_layers":wide.narrow_projection_layers(),
        "small_kernels":small_kernels,"query_norm_quant_layers":wide.query_norm_quant_layers(),
        "profile_widths":profile_widths,"profile_ranges":profile_ranges,
        "woa_cache_screen":woa_cache_screen,"woa_timings":if woa_cache_screen {json!(&woa_timings)}else{json!([])},
        "cached_index16_screen":index16_screen,"index16_handoff":index16_handoff,
        "attention16_native_screen":attention_screen,"fused_shared_quant16":shared_quant16,
        "grouped16_screen":grouped16_screen,"grouped16_timings":if grouped16_screen{json!(&woa_timings)}else{json!([])},
        "verify16_overlap_screen":overlap_screen,"index16_only":index16_only,"handoff_schedules":handoff_schedules,"overlap_timings":if overlap_screen{json!(&woa_timings)}else{json!([])},
        "shared_cached_woa_bytes":cached_woa.iter().map(|v|v.1).sum::<usize>(),
        "full_engine_acceptance":false,"scope":if profile_widths{"One controlled274108-token prefix, one prefill. Warm M6/live8 shapes then one instrumented step of each with CPU-only peer barriers before/after profiler start. Exact logits and aborted state; no sampled outcome or throughput qualification. Wider timing matrix omitted."}
            else if long_context {"One controlled274108token context and causal64-token copy match; one prefill, three warm alternating width pairs with exact abort state, and sequential6/6/4GPUlogits. No repeat of short-context partial7qualification, no outcome-quality claim."}else{"One causally selected64-token history match; six alternating warm timings per width, partial7of16commit crossing ring wrap versus padded7row execution, next6logits/state, and sequential6/6/4reference. GPU probability comparison follows. No serving-policy change or broad quality claim."}});
    if rank==3 {std::fs::write(raw_root.join("index.json"),serde_json::to_vec_pretty(&result)?)?;}
    ensure!(long_context || ["accepted_logits_exact","accepted_state_exact","following_six_logits_exact","following_state_exact"]
        .iter().all(|key|result["rollback"][key]==true),"16-row partial-commit comparison failed: {rollback}");
    Ok(result)
}

// Read-only diagnostic snapshots after a completed forward. These never feed
// model math; a first divergent layer can seed a much smaller reproduction.
fn replay_taps(cuda:&crate::cuda::Cuda,target:&crate::target::Target)->Result<Vec<(String,Vec<u8>)>> {
    let mut result=Vec::new();
    for(i,layer)in target.layer_owners().iter().enumerate() {
        // A post-reduction difference can originate on any rank. Retain each
        // compact consumer's private projection/attention boundaries as well
        // as the reduced output; rank3 alone cannot identify that origin.
        if i>=24 {
            let input=layer.attn_boundary().input();
            result.push((format!("l{i:02}-local-input"),unsafe{cuda.read_device(input.ptr(),input.bytes())}?));
            for(name,ptr,bytes)in layer.attention().local_qkv_replay_taps() {
                result.push((format!("l{i:02}-local-{name}"),unsafe{cuda.read_device(ptr,bytes)}?));
            }
            for(name,ptr,bytes,stride)in layer.attention().taps() {
                let values=if stride==0 {unsafe{cuda.read_device(ptr,bytes)}?}else{
                    let row_bytes=bytes/target.physical_rows();
                    let mut values=Vec::with_capacity(bytes);
                    for row in 0..target.physical_rows() {
                        values.extend(unsafe{cuda.read_device(ptr+(row*stride)as u64,row_bytes)}?);
                    }
                    values
                };
                result.push((format!("l{i:02}-local-{name}"),values));
            }
        }
        for(name,buffer)in [("attn",layer.attn_reduced()),("ffn-input",layer.ffn_boundary().input()),
            ("ffn",layer.ffn_reduced()),("output",layer.output())] {
            result.push((format!("l{i:02}-{name}"),unsafe{cuda.read_device(buffer.ptr(),buffer.bytes())}?));
        }
    }
    // The first divergence moved from33 to37 in s710. Snapshot all four
    // compact producers; persist only the one preceding the first changed layer.
    for (layer,query) in [24,28,32,36].into_iter().zip(target.index_queries()[4..].iter()) {
        let Some(query)=query else{continue;};
        for(name,ptr,bytes)in query.grouped16_replay_taps() {
            result.push((format!("l{layer:02}-index-{name}"),unsafe{cuda.read_device(ptr,bytes)}?));
        }
    }
    if target.index_queries()[4].is_some_and(|q|q.grouped16_enabled()) {
        if let Some(mask)=target.index_queries()[3].and_then(|q|q.candidate_mask()) {
            result.push(("l20-index-source-mask".to_owned(),unsafe{cuda.read_device(mask.ptr(),mask.bytes())}?));
        }
    }
    Ok(result)
}
