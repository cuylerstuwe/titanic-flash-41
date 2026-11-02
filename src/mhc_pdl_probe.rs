//! Same-allocation, graph-replayed M6 residual/norm scheduling screen.
use super::*;

fn state(target: &Target) -> Result<Value> {
    Ok(json!({"position":target.position(),"carry":target.carry_digests()?,
        "encoder":target.encoder_tail_digest()?}))
}
pub(super) fn run(w: &mut Worker, rank: usize, out: &Path) -> Result<()> {
    ensure!(w.current.is_none() && w.history.frontier() == 0 && w.large.position() == 0,
        "isolated MHC PDL screen requires fresh model state");
    ensure!(w.control.all_admit(w.cuda.admission_memory()?.admitted_available >= 9*1024usize.pow(3))?,
        "MHC PDL screen memory reserve");
    let raw = include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture: Value = serde_json::from_slice(raw)?;
    let tokens: Vec<u32> = serde_json::from_value(fixture["tokens"].clone())?;
    let prefix = 65793;
    ensure!(tokens.len() > prefix + 12, "PDL public fixture extent");
    w.large.prefill_wide(&mut w.wides, &tokens[..prefix], &w.comm, &w.stream)?;
    w.verify.handoff_from(&w.large)?;
    let original = state(&w.verify)?;
    let mut records = Vec::new();
    for live in [3, 6] {
        let proposed = &tokens[prefix..prefix+live];
        let changed = &tokens[prefix+6..prefix+6+live];
        ensure!(proposed != changed, "PDL changed-input probe must differ");
        let mut expected = None;
        for (trial, mode) in [0, 1, 3, 2, 2, 3, 1, 0].into_iter().enumerate() {
            w.verify.set_mhc_pdl_pair_for_probe(mode, &w.stream)?;
            for _ in 0..8 {
                w.verify.begin_verification(proposed, &w.comm, &w.stream)?;
                w.verify.finish_verification(0, &w.comm, &w.stream)?;
            }
            let captures = w.verify.capture_statistics().1;
            let mut samples = Vec::new();
            for _ in 0..12 {
                let start = Instant::now();
                w.verify.begin_verification(proposed, &w.comm, &w.stream)?;
                w.verify.finish_verification(0, &w.comm, &w.stream)?;
                samples.push(start.elapsed().as_secs_f64()*1000.);
            }
            ensure!(w.verify.capture_statistics().1 == captures, "timed PDL graph recapture");
            let mut checks = Vec::new();
            for input in [proposed, changed] {
                w.verify.begin_verification(input, &w.comm, &w.stream)?;
                let (logits, count) = w.verify.pending_logits()?;
                ensure!(count == live, "PDL live logits extent");
                let bytes = unsafe { w.cuda.read_device(logits.ptr(), live*129280*4) }?;
                let hash = format!("{:x}", Sha256::digest(bytes));
                let ids = w.verify.verification_sample(2026111700, &w.stream)?;
                w.verify.finish_verification(0, &w.comm, &w.stream)?;
                ensure!(state(&w.verify)? == original, "PDL retained state/rollback mismatch");
                checks.push(json!({"logits_sha256":hash,"sampled_ids":ids}));
            }
            ensure!(checks[0]["logits_sha256"] != checks[1]["logits_sha256"],
                "changed PDL inputs did not change logits");
            if let Some(reference) = &expected {
                ensure!(reference == &checks, "PDL changed full logits or target T1 samples");
            } else { expected = Some(checks.clone()); }
            eprintln!("rank{rank} MHC PDL live{live} mode{mode} trial{trial}: {:.4}ms",
                samples.iter().sum::<f64>() / samples.len() as f64);
            records.push(json!({"live":live,"trial":trial,"mode":mode,"samples_ms":samples,
                "checks":checks,"state_and_rollback_exact":true,"new_timed_graph_captures":0}));
        }
    }
    w.verify.set_mhc_pdl_pair_for_probe(0, &w.stream)?;
    w.control.agree(&[0x76, 0x65])?;
    std::fs::write(out.join("mhc-pdl-summary.json"), serde_json::to_vec_pretty(&json!({
        "rank":rank,"prefix_tokens":prefix,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "all_logits_tokens_state_exact":true,"records":records,
        "scope":"M6 live3/6, one public 65K prefix, same weights and allocations. Original kernels, PDL residual/norm pair, PDL coefficient stats/reduce, and combined. Forward/reverse order with eight untimed warmups and twelve measured verification/rollback steps per arm. Normal and changed proposals check full FP32 logits, T1 samples, retained state and rollback. Existing 25us completion polling and compact-copy option held constant; this screen exercises M6. No task-outcome or serving promotion claim."
    }))?)?;
    Ok(())
}
