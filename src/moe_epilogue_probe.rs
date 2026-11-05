//! Same-allocation, graph-replayed exact M3/M6 expert epilogue screen.
use super::*;
use super::history32_generation_probe::execute;

fn state(target: &Target) -> Result<Value> {
    Ok(json!({"position":target.position(),"carry":target.carry_digests()?,
        "encoder":target.encoder_tail_digest()?}))
}
pub(super) fn run(w: &mut Worker, rank: usize, out: &Path) -> Result<()> {
    ensure!(w.current.is_none(),"isolated MoE cooperative epilogue screen requires no active request");
    ensure!(w.control.all_admit(w.cuda.admission_memory()?.admitted_available >= 9*1024usize.pow(3))?,
        "MoE cooperative epilogue screen memory reserve");
    let raw = include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture: Value = serde_json::from_slice(raw)?;
    let tokens: Vec<u32> = serde_json::from_value(fixture["tokens"].clone())?;
    let prefix = 65793;
    ensure!(tokens.len() > prefix + 12, "cooperative-epilogue public fixture extent");
    execute(w,Command::Begin{images:Vec::new(),id:8_041_285,seed:2026110212,
        greedy:false,limit:1,tokens:tokens[..prefix].to_vec()})?;
    let original = state(&w.verify)?;
    let mut records = Vec::new();
    for live in [3, 6] {
        let proposed = &tokens[prefix..prefix+live];
        let changed = &tokens[prefix+6..prefix+6+live];
        ensure!(proposed != changed, "cooperative-epilogue changed-input probe must differ");
        let mut expected = None;
        for (trial, enabled) in [false,true,true,false].into_iter().enumerate() {
            w.verify.set_cooperative_epilogues_for_probe(enabled, &w.stream)?;
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
            ensure!(w.verify.capture_statistics().1 == captures, "timed cooperative-epilogue graph recapture");
            let mut checks = Vec::new();
            for input in [proposed, changed] {
                w.verify.begin_verification(input, &w.comm, &w.stream)?;
                let (logits, count) = w.verify.pending_logits()?;
                ensure!(count == live, "cooperative-epilogue live logits extent");
                let bytes = unsafe { w.cuda.read_device(logits.ptr(), live*129280*4) }?;
                let hash = format!("{:x}", Sha256::digest(bytes));
                let ids = w.verify.verification_sample(2026111700, &w.stream)?;
                w.verify.finish_verification(0, &w.comm, &w.stream)?;
                ensure!(state(&w.verify)? == original, "cooperative-epilogue retained state/rollback mismatch");
                checks.push(json!({"logits_sha256":hash,"sampled_ids":ids}));
            }
            ensure!(checks[0]["logits_sha256"] != checks[1]["logits_sha256"],
                "changed cooperative-epilogue inputs did not change logits");
            if let Some(reference) = &expected {
                ensure!(reference == &checks, "cooperative-epilogue changed full logits or target T1 samples");
            } else { expected = Some(checks.clone()); }
            eprintln!("rank{rank} MoE cooperative epilogue live{live} enabled{enabled} trial{trial}: {:.4}ms",
                samples.iter().sum::<f64>() / samples.len() as f64);
            records.push(json!({"live":live,"trial":trial,"enabled":enabled,"samples_ms":samples,"metrics":metrics,
                "checks":checks,"state_and_rollback_exact":true,"new_timed_graph_captures":0}));
        }
    }
    w.verify.set_cooperative_epilogues_for_probe(false, &w.stream)?;
    w.control.agree(&[0x76, 0x65])?;
    std::fs::write(out.join("moe-epilogue-summary.json"), serde_json::to_vec_pretty(&json!({
        "rank":rank,"prefix_tokens":prefix,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "all_logits_tokens_state_exact":true,"records":records,
        "scope":"One65K public frontier, M3/M6 fixed verification, four warmups and six timings, reversed arms. M3 retains early PDL and cg caching with direct register conversion; M6 retains ca caching with four-lane cooperative shared-tile conversion. BF16 rounding and MXFP8 arithmetic are unchanged. Unlike the initial standalone kernel screen, this includes shared-expert and coefficient-stream contention. Same compressed weights, exact full logits/T1 samples and rollback. No serving-policy change."
    }))?)?;
    execute(w,Command::Finish(Finish::Length))?;

    Ok(())
}
