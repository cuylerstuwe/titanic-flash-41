//! Same-allocation, graph-replayed exact M3/M6 FFN cache-hint screen.
use super::*;
use super::history32_generation_probe::execute;

fn state(target: &Target) -> Result<Value> {
    Ok(json!({"position":target.position(),"carry":target.carry_digests()?,
        "encoder":target.encoder_tail_digest()?}))
}
pub(super) fn run(w: &mut Worker, rank: usize, out: &Path) -> Result<()> {
    // Reuse the established fixed-prefix/state fixture for a separate finite
    // transport screen. This branch is never entered by HTTP serving.
    let posting=std::env::var("CANADA_SCRATCH_POSTING_API_SCREEN").as_deref()==Ok("1");
    ensure!(w.current.is_none(),"isolated FFN hint screen requires no active request");
    ensure!(w.control.all_admit(w.cuda.admission_memory()?.admitted_available >= 9*1024usize.pow(3))?,
        "FFN hint screen memory reserve");
    let raw = include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture: Value = serde_json::from_slice(raw)?;
    let tokens: Vec<u32> = serde_json::from_value(fixture["tokens"].clone())?;
    let prefix = 65793;
    ensure!(tokens.len() > prefix + 12, "FFN hint public fixture extent");
    execute(w,Command::Begin{images:Vec::new(),id:8_041_304,seed:2026110434,
        greedy:false,limit:1,tokens:tokens[..prefix].to_vec()})?;
    let original = state(&w.verify)?;
    let session_original=super::history32_generation_probe::state(w)?;
    let mut records = Vec::new();
    for live in [3, 6] {
        let proposed = &tokens[prefix..prefix+live];
        let changed = &tokens[prefix+6..prefix+6+live];
        ensure!(proposed != changed, "FFN hint changed-input probe must differ");
        let mut expected = None;
        for (trial, enabled) in [false,true,true,false].into_iter().enumerate() {
            let chunk=if enabled{2048}else{16384};
            if posting {
                w.control.agree(&[0x67,live as u8,trial as u8,u8::from(enabled)])?;
                w.comm.set_posting_mode_for_probe(u32::from(enabled),&w.stream)?;
            } else {w.verify.set_ffn_hint_chunk_for_probe(chunk, &w.stream)?;}
            for _ in 0..4 {
                w.verify.begin_verification(proposed, &w.comm, &w.stream)?;
                w.verify.finish_verification(0, &w.comm, &w.stream)?;
            }
            let captures = w.verify.capture_statistics().1;
            let mut samples = Vec::new();let mut metrics=Vec::new();
            for _ in 0..6 {
                let start = Instant::now();
                w.verify.begin_verification(proposed, &w.comm, &w.stream)?;
                metrics.push(serde_json::to_value(w.verify.verification_metrics()?)?);
                w.verify.finish_verification(0, &w.comm, &w.stream)?;
                samples.push(start.elapsed().as_secs_f64()*1000.);
            }
            ensure!(w.verify.capture_statistics().1 == captures, "timed FFN hint graph recapture");
            let mut checks = Vec::new();
            for input in [proposed, changed] {
                w.verify.begin_verification(input, &w.comm, &w.stream)?;
                let (logits, count) = w.verify.pending_local_logits_for_probe()?;
                ensure!(count == live, "FFN hint live logits extent");
                let bytes = unsafe { w.cuda.read_device(logits.ptr(), live*32320*4) }?;
                let hash = format!("{:x}", Sha256::digest(bytes));
                w.verify.finish_verification(0, &w.comm, &w.stream)?;
                ensure!(state(&w.verify)? == original, "FFN hint retained state/rollback mismatch");
                checks.push(json!({"local_logits_sha256":hash}));
            }
            ensure!(checks[0]["local_logits_sha256"] != checks[1]["local_logits_sha256"],
                "changed FFN hint inputs did not change logits");
            if let Some(reference) = &expected {
                ensure!(reference == &checks, "FFN hint changed local logits");
            } else { expected = Some(checks.clone()); }
            eprintln!("rank{rank} FFN hint live{live} enabled{enabled} trial{trial}: {:.4}ms",
                samples.iter().sum::<f64>() / samples.len() as f64);
            records.push(json!({"live":live,"trial":trial,"enabled":enabled,"hint_chunk":if posting{Value::Null}else{json!(chunk)},"samples_ms":samples,"metrics":metrics,
                "checks":checks,"state_and_rollback_exact":true,"new_timed_graph_captures":0,
                "posting_inline_screen":posting}));
        }
    }
    if posting {w.comm.set_posting_mode_for_probe(0,&w.stream)?;}
    else {w.verify.set_ffn_hint_chunk_for_probe(16384, &w.stream)?;}
    ensure!(super::history32_generation_probe::state(w)?==session_original,"FFN hint changed accepted session");
    w.control.agree(&[0x76, 0x65])?;
    std::fs::write(out.join(if posting{"posting-model-summary.json"}else{"ffn-hint-model-summary.json"}), serde_json::to_vec_pretty(&json!({
        "rank":rank,"prefix_tokens":prefix,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "all_logits_state_exact":true,"records":records,
        "scope":if posting{"One65K public frontier, M3/M6, generic verbs versus identical verbs with <=16B payload stripes inline. ABBA,4warmups,6timings, same allocation/captured graphs/QPs/packets/precision. All4local vocabulary shards and retained rollback exact. Hint settings remain fixed. No serving default change."}else{"One 65K public frontier, M3/M6, 16KiB versus 2KiB FFN hints, ABBA with 4 warmups and 6 timings. Same 6MiB coverage and allocations. All four local vocabulary shards and retained rollback exact; no serving default change."}
    }))?)?;
    execute(w,Command::Finish(Finish::Length))?;

    Ok(())
}
