//! Finite inline-encoder-checkpoint screen (CANADA_SCRATCH_INLINE_CHECKPOINT_SCREEN=1 with
//! CANADA_SCRATCH_ENCODER_CHECKPOINT_SCREEN=1). After a base prefill, the same prefix and suffix run three ways
//! with one forced chunk width: "plain" (one prefill, no checkpoint), "old" (prefix, checkpoint, suffix) and
//! "inline" (one prefill whose last chunk captures the checkpoint as of the prefix end). Checks, all byte-exact:
//!   1. inline leaves the same state, prompt logits and following rows as plain (the capture changes nothing);
//!   2. each image, restored and followed by the same suffix, gives the same state, logits and following rows
//!      whether it came from old or inline;
//!   3. the same for the short-suffix path (reconstruct the 128-row tail from the image, then 17 tokens);
//!   4. the images' carry-ring, encoder-tail and sliding-window regions are byte-identical (the compressed
//!      pages are compared through 2 and 3: their slots past the frontier legitimately differ).
//! Target precision and every serving default are unchanged.
use super::*;

const BUDGET: usize = 2 * 1024usize.pow(3);
fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn state(t: &Target) -> Result<Value> {
    Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))
}
fn following(w: &mut Worker, ids: &[u32]) -> Result<Vec<u8>> {
    let before = state(&w.verify)?;
    w.verify.begin_verification(ids, &w.comm, &w.stream)?;
    let (logits, rows) = w.verify.pending_logits()?;
    ensure!(rows == 6, "inline checkpoint teacher forcing needs six rows");
    let bytes = unsafe { w.cuda.read_device(logits.ptr(), rows * 129280 * 4) }?;
    w.verify.finish_verification(0, &w.comm, &w.stream)?;
    ensure!(state(&w.verify)? == before, "inline checkpoint rollback changed state");
    Ok(bytes)
}
/// After a prefill on the large owner: its state, and one digest over the prompt's logits
/// and six teacher-forced following rows. Leaves the verify owner current.
fn observe(w: &mut Worker, next: &[u32]) -> Result<Value> {
    let at = state(&w.large)?;
    let prompt = w.large.snapshot_prompt_logits(&w.stream)?;
    let mut bytes = unsafe { w.cuda.read_device(prompt.ptr(), 129280 * 4) }?;
    w.verify.handoff_from(&w.large)?;
    bytes.extend_from_slice(&following(w, next)?);
    Ok(json!({"state":at,"logits":digest(&bytes)}))
}

pub(super) fn run(w: &mut Worker, rank: usize, out: &Path) -> Result<()> {
    ensure!(w.current.is_none() && w.history.frontier() == 0 && w.verify.physical_rows() == 6
        && std::env::var("CANADA_SCRATCH_ENCODER_CHECKPOINT_SCREEN").as_deref() == Ok("1"),
        "inline checkpoint screen requires a fresh isolated target and the encoder checkpoint screen");
    w.cached_sessions.clear(); w.session_cache_budget = 0; w.prompt_checkpoint = false;
    w.encoder_prompt_checkpoint = false; w.prompt_cache_logits = false;
    let raw = include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture: Value = serde_json::from_slice(raw)?;
    let tokens: Vec<u32> = serde_json::from_value(fixture["tokens"].clone())?;
    // CANADA_SCRATCH_INLINE_CHECKPOINT_CASES="BASE:PREFIX:SUFFIX:WIDTH,..." (the checkpoint is at BASE+PREFIX;
    // WIDTH is the one chunk that holds PREFIX, and PREFIX+SUFFIX, rows).
    let spec = std::env::var("CANADA_SCRATCH_INLINE_CHECKPOINT_CASES")
        .unwrap_or_else(|_| "8192:768:173:1024,24576:4096:221:8192".into());
    let mut cases = Vec::new();
    for case in spec.split(',').filter(|c| !c.is_empty()) {
        let v = case.split(':').map(|x| x.parse::<usize>()).collect::<Result<Vec<_>, _>>()?;
        ensure!(v.len() == 4 && v[0] >= 8192 && v[1] >= 256 && (128..=255).contains(&v[2]) && v[1] + v[2] <= v[3]
            && tokens.len() >= v[0] + v[1] + v[2] + 6, "inline checkpoint case BASE:PREFIX:SUFFIX:WIDTH");
        cases.push((v[0], v[1], v[2], v[3]));
    }
    ensure!(!cases.is_empty() && cases.len() <= 8, "inline checkpoint cases");
    let mut records = Vec::new();
    let mut exact = true;
    for (case, &(base_tokens, added, suffix_tokens, width)) in cases.iter().enumerate() {
        let checkpoint = base_tokens + added;
        let end = checkpoint + suffix_tokens;
        w.control.agree(&[0xf2, case as u8])?;
        w.target_lineage = w.target_lineage.checked_add(1).context("inline checkpoint lineage exhausted")?;
        if w.verify.profile_is_current() {
            w.verify.reset(w.target_lineage, &w.comm, &w.stream)?;
            w.large.handoff_from(&w.verify)?;
        } else {
            ensure!(w.large.profile_is_current(), "inline checkpoint target owner absent");
            w.large.reset(w.target_lineage, &w.comm, &w.stream)?;
        }
        w.large.prefill_wide(&mut w.wides, &tokens[..base_tokens], &w.comm, &w.stream)?;
        let base = w.large.save_state_image(BUDGET, &w.comm, &w.stream)?;
        w.verify.handoff_from(&w.large)?;
        let (prefix, suffix) = (&tokens[base_tokens..checkpoint], &tokens[checkpoint..end]);
        let all = &tokens[base_tokens..end];
        let next = &tokens[end..end + 6];
        for round in 0..2usize {
            let mut seen: Vec<Value> = Vec::new();
            for (flow, inline) in [("plain", None), ("old", Some(false)), ("inline", Some(true))] {
                w.control.agree(&[0xf3, case as u8, round as u8, inline.map_or(2, u8::from)])?;
                w.large.handoff_from(&w.verify)?;
                w.target_lineage += 1;
                w.large.restore_state_image(&base, w.target_lineage, &w.comm, &w.stream)?;
                w.stream.synchronize()?;
                let began = Instant::now();
                let saved = match inline {
                    None => {
                        let progress = w.large.prefill_wide_interruptible(&mut w.wides, all, &w.comm, &w.stream,
                            Some(vec![width]), &mut |_| Ok(false))?;
                        ensure!(progress.done == all.len(), "inline checkpoint plain extent");
                        None
                    }
                    Some(inline) => {
                        w.large.set_inline_checkpoint_for_probe(Some(inline), Some(vec![width]))?;
                        let (saved, details) = w.large.prefill_encoder_checkpoint_for_probe(&mut w.wides, prefix, suffix,
                            BUDGET, &w.comm, &w.stream)?;
                        w.large.set_inline_checkpoint_for_probe(None, None)?;
                        ensure!(details["inline_checkpoint"].as_bool().unwrap_or(false) == inline
                            && saved.position() == checkpoint, "inline checkpoint path or frontier");
                        Some((saved, details))
                    }
                };
                w.stream.synchronize()?;
                let seconds = began.elapsed().as_secs_f64();
                let phases = serde_json::to_value(w.large.append_metrics()?)?;
                let first = observe(w, next)?;
                let mut record = json!({"case":case,"round":round,"flow":flow,"base":base_tokens,"checkpoint":checkpoint,
                    "end":end,"width":width,"seconds":seconds,"phases":phases,"first":first});
                if let Some((saved, details)) = saved {
                    record["details"] = details;
                    record["image_bytes"] = json!(saved.bytes());
                    record["regions"] = json!(saved.region_digests_for_probe()?);
                    // The image restored, then the same suffix.
                    w.large.handoff_from(&w.verify)?;
                    w.target_lineage += 1;
                    w.large.restore_encoder_checkpoint_for_probe(&mut w.wides, &saved, w.target_lineage, suffix,
                        &w.comm, &w.stream)?;
                    record["restored"] = observe(w, next)?;
                    // The short-suffix path: the 128-row tail rebuilt from the image, then 17 tokens.
                    w.large.handoff_from(&w.verify)?;
                    w.target_lineage += 1;
                    w.large.restore_encoder_checkpoint_full(&saved, w.target_lineage, &w.comm, &w.stream)?;
                    let logits = w.large.last_prompt_logits()?;
                    let mut hash = Sha256::new();
                    for f in &logits { hash.update(f.to_bits().to_le_bytes()); }
                    record["rebuilt"] = json!({"state":state(&w.large)?,"logits":format!("{:x}", hash.finalize())});
                    w.large.prefill_wide(&mut w.wides, &tokens[checkpoint..checkpoint + 17], &w.comm, &w.stream)?;
                    record["short"] = observe(w, &tokens[checkpoint + 17..checkpoint + 23])?;
                }
                seen.push(record);
            }
            let (plain, old, inline) = (&seen[0], &seen[1], &seen[2]);
            let same_regions = |prefix: &str| {
                let pick = |r: &Value| r["regions"].as_array().unwrap().iter()
                    .filter(|x| x[0].as_str().unwrap().starts_with(prefix) || x[0].as_str().unwrap().ends_with(prefix))
                    .cloned().collect::<Vec<_>>();
                let (a, b) = (pick(old), pick(inline));
                !a.is_empty() && a == b
            };
            let checks = json!({
                "inline_first_equals_plain":inline["first"] == plain["first"],
                "restored_equal":inline["restored"] == old["restored"],
                "rebuilt_equal":inline["rebuilt"] == old["rebuilt"],
                "short_equal":inline["short"] == old["short"],
                "image_bytes_equal":inline["image_bytes"] == old["image_bytes"],
                "carry_regions_equal":same_regions(".carry"),
                "encoder_regions_equal":same_regions("encoder."),
                "swa_regions_equal":same_regions("swa."),
                "seconds":{"plain":plain["seconds"],"old":old["seconds"],"inline":inline["seconds"]}});
            exact &= ["inline_first_equals_plain", "restored_equal", "rebuilt_equal", "short_equal", "image_bytes_equal",
                "carry_regions_equal", "encoder_regions_equal", "swa_regions_equal"].iter().all(|k| checks[*k] == true);
            eprintln!("inline checkpoint rank{rank} case{case} round{round}: {checks}");
            records.push(json!({"case":case,"round":round,"checks":checks,"flows":seen}));
            std::fs::write(out.join("inline-checkpoint-progress.json"), serde_json::to_vec_pretty(&records)?)?;
        }
    }
    std::fs::write(out.join("inline-checkpoint-summary.json"), serde_json::to_vec_pretty(&json!({
        "rank":rank,"all_exact":exact,"records":records,"fixture_sha256":digest(raw),
        "scope":"Finite screen of the inline encoder checkpoint: one prefill whose last chunk captures the checkpoint as of the prefix end, against the ordinary prefix/checkpoint/suffix transaction and a plain prefill, with one forced chunk width. Byte comparisons only; no serving default or precision changes.",
    }))?)?;
    eprintln!("inline checkpoint rank{rank} screen: all exact {exact}");
    Ok(())
}
