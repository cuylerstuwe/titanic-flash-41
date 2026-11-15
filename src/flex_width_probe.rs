//! Finite flexible-encoder-width screen (CANADA_SCRATCH_FLEX_WIDTH_SCREEN=1 with CANADA_SCRATCH_FLEX_WIDTHS).
//! Fresh prompts of fixed lengths from the public fixture are prefilled with the ordinary chunk plan and with
//! one flex-width chunk, in ABBA order: two warm-up pairs, then two measured pairs. The last pair saves, per
//! plan, the prompt's logits (row 0) and six teacher-forced following rows. Case 0 compares two ordinary plans
//! for the same rows, the engine's own chunk-boundary noise, as the yardstick for the flex cases. Distributions
//! are compared offline. Target precision and every serving default are unchanged.
use super::*;

fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn state(t: &Target) -> Result<Value> {
    Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))
}
fn following(w: &mut Worker, ids: &[u32]) -> Result<Vec<u8>> {
    let before = state(&w.verify)?;
    w.verify.begin_verification(ids, &w.comm, &w.stream)?;
    let (logits, rows) = w.verify.pending_logits()?;
    ensure!(rows == 6, "flex width teacher forcing needs six rows");
    let bytes = unsafe { w.cuda.read_device(logits.ptr(), rows * 129280 * 4) }?;
    w.verify.finish_verification(0, &w.comm, &w.stream)?;
    ensure!(state(&w.verify)? == before, "flex width rollback changed state");
    Ok(bytes)
}

pub(super) fn run(w: &mut Worker, rank: usize, out: &Path) -> Result<()> {
    ensure!(w.current.is_none() && w.history.frontier() == 0 && w.verify.physical_rows() == 6
        && std::env::var("CANADA_SCRATCH_FLEX_WIDTH_SCREEN").as_deref() == Ok("1")
        && (!crate::config::flex_widths().is_empty() || std::env::var("CANADA_SCRATCH_FLEX_WIDTH_CASES").is_ok()),
        "plan screen requires a fresh isolated target and flex widths or explicit cases");
    w.cached_sessions.clear(); w.session_cache_budget = 0; w.prompt_checkpoint = false;
    w.encoder_prompt_checkpoint = false; w.prompt_cache_logits = false;
    let raw = include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture: Value = serde_json::from_slice(raw)?;
    let tokens: Vec<u32> = serde_json::from_value(fixture["tokens"].clone())?;
    let all: Vec<usize> = std::iter::once(128).chain(w.wides.iter().map(Target::physical_rows)).collect();
    let native: Vec<usize> = all.iter().copied().filter(|&r| !crate::config::flex_width(r)).collect();
    let flex = crate::config::flex_widths();
    // CANADA_SCRATCH_FLEX_WIDTH_CASES="DELTA:A+B/C+D,..." replaces the built-in cases with explicit plan pairs
    // (any configured widths), e.g. "8192:8192/4096+4096" to separate chunk-boundary state from partial rows.
    // A plan ending in "p" (e.g. "8192p") runs with partial wide chunks bounded to their live
    // rows (Target::set_partial_wide_rows_for_probe, mask CANADA_SCRATCH_PARTIAL_WIDE_SCREEN_MASK,
    // default 3 = dense and routed experts); other plans run with the bound off.
    let mut cases: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();
    let mut bounded: Vec<[bool; 2]> = Vec::new();
    let mask: u8 = match std::env::var("CANADA_SCRATCH_PARTIAL_WIDE_SCREEN_MASK") {
        Ok(v) => v.parse()?, Err(_) => 3 };
    if let Ok(spec) = std::env::var("CANADA_SCRATCH_FLEX_WIDTH_CASES") {
        let plan = |p: &str| p.trim_end_matches('p').split('+').map(|w| w.parse::<usize>()).collect::<Result<Vec<_>, _>>();
        for case in spec.split(',').filter(|c| !c.is_empty()) {
            let (delta, plans) = case.split_once(':').context("flex width case needs DELTA:PLAN/PLAN")?;
            let (a, b) = plans.split_once('/').context("flex width case needs two plans")?;
            cases.push((delta.parse()?, plan(a)?, plan(b)?));
            bounded.push([a.ends_with('p'), b.ends_with('p')]);
        }
    } else {
        cases.push((4350, vec![4096, 512], vec![4096, 128, 128]));
        for delta in [4350usize, 4770, 6187, 7295] {
            let control = crate::config::plan_encoder_chunks(delta, &native);
            if let Some(&width) = flex.iter().find(|&&f| f >= delta) {
                if control != vec![width] { cases.push((delta, control, vec![width])); }
            }
        }
    }
    bounded.resize(cases.len(), [false, false]);
    ensure!(!cases.is_empty() && cases.len() <= 16
        && cases.iter().all(|(delta, _, _)| (128..=16384).contains(delta) && tokens.len() > delta + 6),
        "flex width screen cases");
    let mut records = Vec::new();
    for (case, (delta, control, alternative)) in cases.iter().enumerate() {
        let plans = [control.clone(), alternative.clone()];
        for plan in &plans { crate::config::validate_encoder_chunks(*delta, plan, &all)?; }
        for (trial, mode) in [0usize, 1, 1, 0, 0, 1, 1, 0].into_iter().enumerate() {
            w.control.agree(&[0xf1, case as u8, trial as u8, mode as u8])?;
            w.target_lineage = w.target_lineage.checked_add(1).context("flex width lineage exhausted")?;
            if w.verify.profile_is_current() {
                w.verify.reset(w.target_lineage, &w.comm, &w.stream)?;
                w.large.handoff_from(&w.verify)?;
            } else {
                ensure!(w.large.profile_is_current(), "flex width target owner absent");
                w.large.reset(w.target_lineage, &w.comm, &w.stream)?;
            }
            for wide in w.wides.iter_mut() {
                wide.set_partial_wide_rows_for_probe(if bounded[case][mode] { mask } else { 0 }, &w.stream)?;
            }
            w.stream.synchronize()?;
            let began = Instant::now();
            let progress = w.large.prefill_wide_interruptible(&mut w.wides, &tokens[..*delta], &w.comm, &w.stream,
                Some(plans[mode].clone()), &mut |_| Ok(false))?;
            w.stream.synchronize()?;
            let seconds = began.elapsed().as_secs_f64();
            ensure!(progress.done == *delta && progress.remaining_chunks.is_empty(), "flex width screen extent");
            let phases = serde_json::to_value(w.large.append_metrics()?)?;
            let (mut file, mut sha) = (Value::Null, Value::Null);
            if trial >= 4 {
                // Both measured repeats of each plan are saved: equal bytes show the plan is deterministic.
                let prompt = w.large.snapshot_prompt_logits(&w.stream)?;
                let mut bytes = unsafe { w.cuda.read_device(prompt.ptr(), 129280 * 4) }?;
                w.verify.handoff_from(&w.large)?;
                let after = following(w, &tokens[*delta..*delta + 6])?;
                bytes.extend_from_slice(&after);
                let name = if trial >= 6 { format!("flex-c{case}-m{mode}.f32") } else { format!("flex-c{case}-m{mode}-t{trial}.f32") };
                if rank == 0 { std::fs::write(out.join(&name), &bytes)?; }
                file = json!(name); sha = json!(digest(&bytes));
            }
            records.push(json!({"case":case,"delta":delta,"mode":mode,"plan":plans[mode],"trial":trial,
                "partial_wide_mask":if bounded[case][mode] { mask } else { 0 },
                "warmup":trial < 4,"seconds":seconds,"phases":phases,"rows":7,"file":file,"sha256":sha}));
            std::fs::write(out.join("flex-width-progress.json"), serde_json::to_vec_pretty(&records)?)?;
        }
        eprintln!("flex width rank{rank} case{case} delta{delta} {:?} vs {:?} complete", plans[0], plans[1]);
    }
    std::fs::write(out.join("flex-width-summary.json"), serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"flex_widths":flex,"native_widths":native,
        "fixture_sha256":digest(raw),
        "scope":"Fresh prompts from the public fixture, each prefilled with two chunk plans (built-in: the ordinary plan against one flex-width chunk at 4,350/4,770/6,187/7,295 tokens, case 0 two ordinary plans; or CANADA_SCRATCH_FLEX_WIDTH_CASES). ABBA, two warm-up and two measured pairs; both measured repeats saved (flex-cC-mM-t4|5.f32 and flex-cC-mM.f32). Row 0 is the prompt's logits, rows 1-6 follow fixed teacher-forced tokens. Distributions are compared offline; no serving default changes."
    }))?)?;
    Ok(())
}
