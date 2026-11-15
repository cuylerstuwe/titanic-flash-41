//! Finite batched-decode screen, stage 1: two sessions in one six-row verify owner.
//! Each session occupies its own region of every cache (SessionBase) and is built by
//! six-row appends from its own history. The check: one verification step over both
//! sessions' rows gives every row exactly the logits of that session's rows verified
//! alone, and the base-zero override equals the ordinary single-session path.
use super::*;
use crate::target::SessionBase;

fn row_hashes(w: &mut Worker, rows: usize) -> Result<Vec<String>> {
    // A batched verification exposes its logits through one staged session;
    // staging all rows at offset zero reads the whole step in place.
    if w.verify.batched_pending_for_probe() { w.verify.stage_batched_session(0, rows, &w.stream)?; }
    let (logits, count) = w.verify.pending_logits()?;
    ensure!(count == rows, "batched screen logits extent");
    let bytes = unsafe { w.cuda.read_device(logits.ptr(), rows * 129280 * 4) }?;
    Ok(bytes.chunks(129280 * 4).map(|r| format!("{:x}", Sha256::digest(r))).collect())
}

/// Verify `next` rows of each listed session together (one part per session) and
/// return every row's full-logit hash, aborting (full carry restore) afterwards.
fn verify_abort(w: &mut Worker, parts: &[(SessionBase, &crate::hash_history::History, &[u32])]) -> Result<Vec<String>> {
    let ids: Vec<u32> = parts.iter().flat_map(|p| p.2.iter().copied()).collect();
    w.verify.set_batched_rows_for_probe(parts.iter().map(|p| (p.0, p.1.clone(), p.2.len())).collect())?;
    w.verify.begin_verification(&ids, &w.comm, &w.stream)?;
    let rows = row_hashes(w, ids.len())?;
    w.verify.abort_batched_verification_for_probe(&w.stream)?;
    Ok(rows)
}

/// Diagnostic: verify each part's rows (one block per part), hash every part's staged rows
/// (this rank's vocabulary shard when `local`, else the gathered logits), then abort.
fn staged_hashes(w: &mut Worker, parts: &[(SessionBase, &crate::hash_history::History, &[u32])], local: bool) -> Result<Vec<String>> {
    let ids: Vec<u32> = parts.iter().flat_map(|p| p.2.iter().copied()).collect();
    w.verify.set_batched_rows_for_probe(parts.iter().map(|p| (p.0, p.1.clone(), p.2.len())).collect())?;
    if local { w.verify.begin_local_verification(&ids, &w.comm, &w.stream)?; }
    else { w.verify.begin_verification(&ids, &w.comm, &w.stream)?; }
    let mut out = Vec::new();
    for (k, part) in parts.iter().enumerate() {
        let offset = w.verify.batched_offset(k)?;
        w.verify.stage_batched_session(offset, part.2.len(), &w.stream)?;
        w.stream.synchronize()?;
        let (logits, rows) = if local { w.verify.pending_local_target_logits()? } else { w.verify.pending_logits()? };
        ensure!(rows == part.2.len(), "staged diagnostic rows");
        let bytes = unsafe { w.cuda.read_device(logits.ptr(), rows * if local { 32320 } else { 129280 } * 4) }?;
        out.push(format!("{:x}", Sha256::digest(&bytes)));
    }
    w.verify.abort_batched_verification_for_probe(&w.stream)?;
    Ok(out)
}

/// Stage 2a: four regions hold A, B and their twins A2, B2 (same texts). Region
/// invariance: each twin verifies exactly like its original. Commit equivalence:
/// A+B verified together and committed with (2, 1) accepted rows equal A2 and B2
/// verified alone and committed with the same counts, judged by the next rows.
fn run_commit(w: &mut Worker, rank: usize, out: &Path) -> Result<()> {
    ensure!(w.current.is_none() && crate::config::batch_screen_sessions()? == 4,
        "batched commit screen needs an idle worker sized for four sessions");
    let raw = include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture: Value = serde_json::from_slice(raw)?;
    let tokens: Vec<u32> = serde_json::from_value(fixture["tokens"].clone())?;
    let len = 1536;
    let texts = [&tokens[..len + 24], &tokens[60000..60000 + len + 24]];
    let capacity = w.verify.context_capacity() / 4;
    let window = crate::swa_layout::OFFSET + w.verify.swa_layout_for_probe().ring().context("ring window layout")?;
    let bases: Vec<SessionBase> = (0..4).map(|i| SessionBase::region(i, capacity, window)).collect::<Result<_>>()?;
    let fresh = w.verify.history_for_probe();
    ensure!(fresh.position == 0, "batched commit screen needs a fresh verify owner");
    // Region 0 = A (ordinary path), 1 = B, 2 = A2 (text A), 3 = B2 (text B).
    let mut h = vec![fresh.clone(); 4];
    let own = w.verify.swap_history_for_probe(fresh.clone())?;
    for chunk in texts[0][..len].chunks(6) { w.verify.append(chunk, &w.comm, &w.stream)?; }
    h[0] = w.verify.swap_history_for_probe(own)?;
    for (region, text) in [(1usize, 1usize), (2, 0), (3, 1)] {
        for chunk in texts[text][..len].chunks(6) {
            w.verify.set_batched_rows_for_probe(vec![(bases[region], h[region].clone(), chunk.len())])?;
            w.verify.append(chunk, &w.comm, &w.stream)?;
            h[region] = w.verify.take_batched_histories_for_probe()?.pop().context("advanced history")?;
        }
    }
    let next = |text: usize, at: usize| &texts[text][at..at + 3];
    // Region invariance at the shared frontier.
    let a = verify_abort(w, &[(bases[0], &h[0], next(0, len))])?;
    let a2 = verify_abort(w, &[(bases[2], &h[2], next(0, len))])?;
    let b = verify_abort(w, &[(bases[1], &h[1], next(1, len))])?;
    let b2 = verify_abort(w, &[(bases[3], &h[3], next(1, len))])?;
    let invariant = a == a2 && b == b2;
    // Commit equivalence: A+B together (2, 1) against A2 and B2 alone with the same counts.
    let ids: Vec<u32> = next(0, len).iter().chain(next(1, len)).copied().collect();
    w.verify.set_batched_rows_for_probe(vec![(bases[0], h[0].clone(), 3), (bases[1], h[1].clone(), 3)])?;
    w.verify.begin_verification(&ids, &w.comm, &w.stream)?;
    let together = w.verify.finish_batched_verification(&[2, 1], &w.stream)?;
    h[0] = together[0].clone(); h[1] = together[1].clone();
    for (region, text, accept) in [(2usize, 0usize, 2usize), (3, 1, 1)] {
        w.verify.set_batched_rows_for_probe(vec![(bases[region], h[region].clone(), 3)])?;
        w.verify.begin_verification(next(text, len), &w.comm, &w.stream)?;
        h[region] = w.verify.finish_batched_verification(&[accept], &w.stream)?.pop().context("committed")?;
    }
    ensure!(h[0].position == h[2].position && h[1].position == h[3].position, "committed frontiers");
    let pa = h[0].position as usize; let pb = h[1].position as usize;
    let after_a = verify_abort(w, &[(bases[0], &h[0], next(0, pa))])?;
    let after_a2 = verify_abort(w, &[(bases[2], &h[2], next(0, pa))])?;
    let after_b = verify_abort(w, &[(bases[1], &h[1], next(1, pb))])?;
    let after_b2 = verify_abort(w, &[(bases[3], &h[3], next(1, pb))])?;
    let equivalent = after_a == after_a2 && after_b == after_b2;
    // And the four sessions' next rows in one 12-row-equivalent pass of two steps (two per step).
    let both = verify_abort(w, &[(bases[0], &h[0], next(0, pa)), (bases[1], &h[1], next(1, pb))])?;
    let batched_after_exact = both[..3] == after_a[..] && both[3..] == after_b[..];
    eprintln!("rank{rank} batched commit screen: region invariance {invariant}, commit equivalence {equivalent}, batched after commit {batched_after_exact}");
    w.control.agree(&[0x62, 0x63])?;
    std::fs::write(out.join("batch-screen-summary.json"), serde_json::to_vec_pretty(&json!({
        "rank":rank,"mode":"commit","session_tokens":len,"session_capacity":capacity,"window_region_slots":window,
        "records":[{"region_invariance":invariant,"commit_equivalence":equivalent,"batched_after_commit_exact":batched_after_exact,
            "frontiers":[h[0].position,h[1].position,h[2].position,h[3].position]}],
        "scope":"Four regions: A (ordinary path), B, and twins A2/B2 of the same texts built by batched appends. Twins verify identically (region invariance); A+B verified together and committed (2,1) equal A2/B2 verified alone and committed (2,1), judged by the next rows' full logits; then A+B together again. Diagnostic only."
    }))?)?;
    Ok(())
}

/// Stage 3: generation. Four regions with their own drafters: A and B decode in batched
/// steps (one verification for both), their twins A2 and B2 (same texts, same seeds)
/// decode alone through one-session steps. Every session's tokens must match its twin's.
fn run_generate(w: &mut Worker, rank: usize, out: &Path, kernels: &Path) -> Result<()> {
    use crate::dspark_generation::{BatchedSession, GreedySession};
    ensure!(w.current.is_none() && crate::config::batch_screen_sessions()? == 4,
        "batched generation screen needs an idle worker sized for four sessions");
    let raw = include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture: Value = serde_json::from_slice(raw)?;
    let tokens: Vec<u32> = serde_json::from_value(fixture["tokens"].clone())?;
    let len = 1536;
    let texts = [&tokens[..len + 3], &tokens[60000..60000 + len + 3]];
    let capacity = w.verify.context_capacity() / 4;
    let window = crate::swa_layout::OFFSET + w.verify.swa_layout_for_probe().ring().context("ring window layout")?;
    let bases: Vec<SessionBase> = (0..4).map(|i| SessionBase::region(i, capacity, window)).collect::<Result<_>>()?;
    let fresh = w.verify.history_for_probe();
    ensure!(fresh.position == 0, "batched generation screen needs a fresh verify owner");
    let prepared = w.prepared_session.as_ref().context("prepared drafter")?;
    let eos = w.cfg.eos_token_id;
    let mut sessions: Vec<GreedySession> = Vec::new();
    for region in 0..4 {
        let mut s = prepared.independent_for_probe(&w.verify, &w.core, &w.stream, kernels)?.detached_for_probe()?;
        s.configure_sampling(2026111500 + (region % 2) as u64, Some(eos))?;
        s.configure_history_lookup(&w.verify, &[])?;
        sessions.push(s);
    }
    // Regions 0/2 hold text A, 1/3 text B; every prompt is absorbed by six-row appends.
    let mut histories = vec![fresh.clone(); 4];
    let built = Instant::now();
    for region in 0..4 {
        let text = texts[region % 2];
        for (i, chunk) in text[..len].chunks(6).enumerate() {
            w.verify.set_batched_rows_for_probe(vec![(bases[region], histories[region].clone(), chunk.len())])?;
            w.verify.append(chunk, &w.comm, &w.stream)?;
            histories[region] = w.verify.take_batched_histories_for_probe()?.pop().context("advanced history")?;
            sessions[region].absorb_batched_append_for_probe(&w.verify, (i * 6) as u64, chunk, &w.comm, &w.stream)?;
        }
    }
    let build_seconds = built.elapsed().as_secs_f64();
    // Diagnostic (with the trace): at the build frontier, each region's next three rows alone,
    // then in pairs (A+B, the twins, and B+A swapped), with the local and the gathered head.
    // Separates a region whose state differs from its twin's from a block-position effect.
    let mut frontier = Value::Null;
    if std::env::var("CANADA_SCRATCH_BATCH_SCREEN_TRACE").as_deref() == Ok("1") {
        let next = |r: usize| &texts[r % 2][len..len + 3];
        let mut checks = serde_json::Map::new();
        for local in [true, false] {
            let mut solo = Vec::new();
            for r in 0..4 { solo.push(staged_hashes(w, &[(bases[r], &histories[r], next(r))], local)?.remove(0)); }
            let p01 = staged_hashes(w, &[(bases[0], &histories[0], next(0)), (bases[1], &histories[1], next(1))], local)?;
            let p23 = staged_hashes(w, &[(bases[2], &histories[2], next(2)), (bases[3], &histories[3], next(3))], local)?;
            let p10 = staged_hashes(w, &[(bases[1], &histories[1], next(1)), (bases[0], &histories[0], next(0))], local)?;
            let p21 = staged_hashes(w, &[(bases[2], &histories[2], next(2)), (bases[1], &histories[1], next(1))], local)?;
            let again = staged_hashes(w, &[(bases[0], &histories[0], next(0))], local)?.remove(0);
            checks.insert(if local { "local" } else { "gathered" }.into(), json!({
                "twin_regions_equal":[solo[0] == solo[2], solo[1] == solo[3]],
                "solo_repeat_equal":again == solo[0],
                "pair_ab_equals_solo":[p01[0] == solo[0], p01[1] == solo[1]],
                "pair_twins_equals_solo":[p23[0] == solo[2], p23[1] == solo[3]],
                "pair_ba_equals_solo":[p10[0] == solo[1], p10[1] == solo[0]],
                "pair_a2b_equals_solo":[p21[0] == solo[2], p21[1] == solo[1]],
                "solo":solo,"pair_ab":p01,"pair_twins":p23,"pair_ba":p10}));
        }
        eprintln!("rank{rank} batched generation frontier checks: {}", serde_json::to_string(&json!({
            "local":{"twins":checks["local"]["twin_regions_equal"],"ab":checks["local"]["pair_ab_equals_solo"],
                "twins_pair":checks["local"]["pair_twins_equals_solo"],"ba":checks["local"]["pair_ba_equals_solo"],
                "a2b":checks["local"]["pair_a2b_equals_solo"],"repeat":checks["local"]["solo_repeat_equal"]},
            "gathered":{"twins":checks["gathered"]["twin_regions_equal"],"ab":checks["gathered"]["pair_ab_equals_solo"],
                "twins_pair":checks["gathered"]["pair_twins_equals_solo"],"ba":checks["gathered"]["pair_ba_equals_solo"],
                "a2b":checks["gathered"]["pair_a2b_equals_solo"],"repeat":checks["gathered"]["solo_repeat_equal"]}}))?);
        frontier = Value::Object(checks);
    }
    // CANADA_SCRATCH_BATCH_SCREEN_STEPS (default 48) sets the length.
    // CANADA_SCRATCH_BATCH_SCREEN_ROWS (6 or 16) selects the verify owner of every step: the six-row owner, or the
    // 16-row one (Worker::long_verify) that serving uses for long history blocks. CANADA_SCRATCH_BATCH_SCREEN_PARTS
    // (2 on six rows; 2, 3 or 4 on sixteen) is the number of conversations in a batched step: regions 0..parts. The
    // other regions decode alone through the same owner. A batched conversation gets rows/parts rows a step, at most
    // six: 3 on six rows; 6, 5 or 4 on sixteen. CANADA_SCRATCH_BATCH_SCREEN_SOLO_LIMIT (default: the batched limit)
    // is the solo twins' rows a step; with the default the twins must match their batched partners token for token.
    let steps: usize = match std::env::var("CANADA_SCRATCH_BATCH_SCREEN_STEPS") { Ok(v) => v.parse()?, Err(_) => 48 };
    let rows: usize = match std::env::var("CANADA_SCRATCH_BATCH_SCREEN_ROWS") { Ok(v) => v.parse()?, Err(_) => 6 };
    let parts_n: usize = match std::env::var("CANADA_SCRATCH_BATCH_SCREEN_PARTS") { Ok(v) => v.parse()?, Err(_) => 2 };
    ensure!(matches!((rows, parts_n), (6, 2) | (16, 2..=4)), "batched generation screen rows/parts");
    let limit = if rows == 6 { 3 } else { (16 / parts_n).min(6) };
    let solo_limit: usize = match std::env::var("CANADA_SCRATCH_BATCH_SCREEN_SOLO_LIMIT") { Ok(v) => v.parse()?, Err(_) => limit };
    ensure!((8..=512).contains(&steps) && (1..=6).contains(&solo_limit), "batched generation screen steps/solo limit");
    if rows == 16 {
        let longer = w.long_verify.as_mut().context("batched screen needs the 16-row verify owner")?;
        longer.handoff_from(&w.verify)?;
        longer.set_engram_overlap_for_probe(false)?;
        if !crate::config::batched_graphs_requested() { longer.set_graph_replay(false)?; }
        w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;
    }
    let (mut pair_tokens, mut solo_tokens) = (Vec::new(), Vec::new());
    let mut anchors: Vec<u32> = (0..4).map(|r| texts[r % 2][len]).collect();
    let mut emitted: Vec<Vec<u32>> = vec![Vec::new(); 4];
    let mut sources: Vec<Vec<String>> = vec![Vec::new(); 4];
    let (mut pair_ms, mut solo_ms) = (Vec::new(), Vec::new());
    let tracing = std::env::var("CANADA_SCRATCH_BATCH_SCREEN_TRACE").as_deref() == Ok("1");
    let mut traces: Vec<Vec<Value>> = vec![Vec::new(); 4];
    let mut sessions = sessions;
    for _ in 0..steps {
        let target: &mut Target = if rows == 16 {
            let longer = w.long_verify.as_mut().unwrap();
            longer.handoff_from(&w.verify)?;
            longer
        } else { &mut w.verify };
        let (batched, alone) = sessions.split_at_mut(parts_n);
        // Regions 0..parts in one batched step.
        let t = Instant::now();
        let mut group: Vec<BatchedSession> = batched.iter_mut().enumerate().map(|(region, session)| BatchedSession {
            session, base: bases[region], history: histories[region].clone(), anchor: anchors[region], limit,
            trace: tracing.then(Vec::new) }).collect();
        let out_group = GreedySession::batched_step(&mut group, target, &w.comm, &w.stream)?;
        pair_ms.push(t.elapsed().as_secs_f64() * 1000.);
        for (region, part) in group.iter_mut().enumerate() {
            histories[region] = part.history.clone();
            traces[region].extend(part.trace.take().unwrap_or_default());
        }
        drop(group);
        // The remaining regions each alone (one-session steps through the same owner).
        let t = Instant::now();
        let mut out_solo = Vec::new();
        for (k, session) in alone.iter_mut().enumerate() {
            let region = parts_n + k;
            let mut one = [BatchedSession { session, base: bases[region], history: histories[region].clone(),
                anchor: anchors[region], limit: solo_limit, trace: tracing.then(Vec::new) }];
            out_solo.push(GreedySession::batched_step(&mut one, target, &w.comm, &w.stream)?.pop().unwrap());
            histories[region] = one[0].history.clone();
            traces[region].extend(one[0].trace.take().unwrap_or_default());
        }
        solo_ms.push(t.elapsed().as_secs_f64() * 1000.);
        if rows == 16 { w.verify.handoff_from(w.long_verify.as_ref().unwrap())?; }
        pair_tokens.push(out_group.iter().map(|s| s.tokens.len()).sum::<usize>());
        solo_tokens.push(out_solo.iter().map(|s| s.tokens.len()).sum::<usize>());
        for (region, step) in out_group.iter().chain(out_solo.iter()).enumerate() {
            emitted[region].extend_from_slice(&step.tokens);
            anchors[region] = step.next_anchor;
            sources[region].push(format!("{}:{}", step.proposal_source, step.verified_rows));
        }
    }
    // Regions 0 and 2 hold one text and seed, regions 1 and 3 the other.
    let exact_a = emitted[0] == emitted[2] && anchors[0] == anchors[2];
    let exact_b = emitted[1] == emitted[3] && anchors[1] == anchors[3];
    let median = |v: &Vec<f64>| { let mut s = v[2..].to_vec(); s.sort_by(|a, b| a.partial_cmp(b).unwrap()); s[s.len() / 2] };
    // Aggregate rates after two warm-up steps: the batched conversations in one step, against the others taking turns.
    let rate = |tokens: &Vec<usize>, ms: &Vec<f64>| tokens[2..].iter().sum::<usize>() as f64 / (ms[2..].iter().sum::<f64>() / 1000.);
    let (pair_rate, solo_rate) = (rate(&pair_tokens, &pair_ms), rate(&solo_tokens, &solo_ms));
    let lengths = emitted.iter().map(|e| e.len()).collect::<Vec<_>>();
    eprintln!("rank{rank} batched generation screen: A==A2 {exact_a}, B==B2 {exact_b}; tokens {} / {}; pair step {:.2} ms, two solo steps {:.2} ms",
        emitted[0].len(), emitted[1].len(), median(&pair_ms), median(&solo_ms));
    eprintln!("rank{rank} batched generation rates: {steps} steps, {rows}-row owner, {parts_n} conversations a step at {limit} rows, {} alone at {solo_limit} rows; batched {pair_rate:.1} tok/s, alone taking turns {solo_rate:.1} tok/s; tokens by region {lengths:?}; batched step {:.2} ms, alone steps {:.2} ms",
        4 - parts_n, median(&pair_ms), median(&solo_ms));
    w.control.agree(&[0x62, 0x67])?;
    std::fs::write(out.join("batch-screen-summary.json"), serde_json::to_vec_pretty(&json!({
        "rank":rank,"mode":"generate","session_tokens":len,"steps":steps,"build_seconds":build_seconds,
        "records":[{"exact_session_a":exact_a,"exact_session_b":exact_b,
            "emitted_lengths":emitted.iter().map(|e| e.len()).collect::<Vec<_>>(),
            "pair_step_ms":pair_ms,"two_solo_steps_ms":solo_ms,
            "pair_step_median_ms":median(&pair_ms),"two_solo_steps_median_ms":median(&solo_ms),
            "solo_limit":solo_limit,"pair_tokens_per_second":pair_rate,"solo_turns_tokens_per_second":solo_rate,
            "owner_rows":rows,"batched_conversations":parts_n,"batched_limit":limit,
            "sources":sources,"trace":traces,"frontier_checks":frontier}],
        "scope":"Four regions with independent drafters; prompts absorbed by six-row appends and feature injection. A+B decode in batched steps (one verification), twins A2/B2 alone with the same seeds; tokens must match per session. T1 sampling with the production drafter, causal-three and block verification; three rows per session (one six-row step holds two sessions' blocks); every step eager. Diagnostic only."
    }))?)?;
    Ok(())
}

/// One conversation of the relocation screen: its drafter, region, accepted history and anchor.
struct Conversation {
    session: crate::dspark_generation::GreedySession,
    base: SessionBase,
    history: crate::hash_history::History,
    anchor: u32,
}

/// Start a conversation in region 0 the way serving does: a fresh lineage (with serving regions
/// the reset clears region 0's state only), the ordinary wide prefill, the sampled anchor and a
/// drafter of its own. Returns the conversation and its prefill seconds.
fn prefill_conversation(w: &mut Worker, text: &[u32], seed: u64,
    fork: crate::dspark_generation::PreparedSession) -> Result<(Conversation, f64)> {
    w.target_lineage = w.target_lineage.checked_add(1).context("relocation screen lineage")?;
    w.verify.reset(w.target_lineage, &w.comm, &w.stream)?;
    w.large.handoff_from(&w.verify)?;
    let began = Instant::now();
    w.large.prefill_wide(&mut w.wides, text, &w.comm, &w.stream)?;
    let seconds = began.elapsed().as_secs_f64();
    let anchor = w.large.sample_token(seed, &w.stream)?;
    let mut session = fork.attach(&w.large, &w.stream)?;
    session.configure_sampling(seed, Some(w.cfg.eos_token_id))?;
    session.configure_history_lookup(&w.large, text)?;
    w.verify.handoff_from(&w.large)?;
    let history = w.verify.history_for_probe();
    ensure!(history.position == text.len() as u64, "relocation screen prefill frontier");
    Ok((Conversation { session, base: SessionBase::default(), history, anchor }, seconds))
}

/// `steps` decode steps, each one batched step over every listed conversation (one part each,
/// `limit` rows each) on the 6- or 16-row owner. Returns each conversation's emitted tokens and
/// the step times in ms.
fn decode_conversations(w: &mut Worker, convs: &mut [Conversation], steps: usize, rows: usize,
    limit: usize) -> Result<(Vec<Vec<u32>>, Vec<f64>)> {
    use crate::dspark_generation::{BatchedSession, GreedySession};
    let mut emitted: Vec<Vec<u32>> = vec![Vec::new(); convs.len()];
    let mut ms = Vec::with_capacity(steps);
    for _ in 0..steps {
        let target: &mut Target = if rows == 16 {
            let longer = w.long_verify.as_mut().context("relocation screen 16-row owner")?;
            longer.handoff_from(&w.verify)?;
            longer
        } else { &mut w.verify };
        let t = Instant::now();
        // A region-0 conversation in part 0 is the owner's own: its commit advances the owner's
        // history and encoder tail, as serving does.
        target.set_batched_owner_part(convs[0].base == SessionBase::default())?;
        let mut group: Vec<BatchedSession> = convs.iter_mut().map(|c| BatchedSession { session: &mut c.session,
            base: c.base, history: c.history.clone(), anchor: c.anchor, limit, trace: None }).collect();
        let outs = GreedySession::batched_step(&mut group, target, &w.comm, &w.stream)?;
        ms.push(t.elapsed().as_secs_f64() * 1000.);
        let histories: Vec<_> = group.iter().map(|p| p.history.clone()).collect();
        drop(group);
        if rows == 16 { w.verify.handoff_from(w.long_verify.as_ref().unwrap())?; }
        for (((c, h), step), e) in convs.iter_mut().zip(histories).zip(&outs).zip(emitted.iter_mut()) {
            c.history = h;
            c.anchor = step.next_anchor;
            e.extend_from_slice(&step.tokens);
        }
    }
    Ok((emitted, ms))
}

/// The next turn of a region-0 conversation: `anchor` and `text` appended by the ordinary prefill
/// path (which replays the encoder tail), returning the greedy and the seeded next token.
fn continuation(w: &mut Worker, anchor: u32, text: &[u32], seed: u64) -> Result<(u32, u32)> {
    let mut ids = vec![anchor];
    ids.extend_from_slice(text);
    w.large.handoff_from(&w.verify)?;
    w.large.prefill_wide(&mut w.wides, &ids, &w.comm, &w.stream)?;
    let greedy = w.large.greedy_token(&w.stream)?;
    let sampled = w.large.sample_token(seed, &w.stream)?;
    w.verify.handoff_from(&w.large)?;
    Ok((greedy, sampled))
}

/// Serving step 1 (CANADA_SCRATCH_BATCH_SCREEN_MODE=relocate, CANADA_SCRATCH_BATCHED_REGIONS=R >= 2,
/// CANADA_SCRATCH_RESET_KEEP_CACHES=1): a conversation prefilled by the ordinary wide path in region 0
/// and then moved into region R-1 (its small window ring, request slot and compressed share) decodes
/// exactly as it does in region 0.
/// - References: text A prefilled and decoded alone in region 0; with PARTS=2 text Q as well.
/// - Moved run: A is prefilled again and moved to region R-1. Region 0 is then reset (region 0 only)
///   and prefilled with Q, overwriting what A left there. A decodes in region R-1, alone (PARTS=1) or
///   in one batched step with Q (PARTS=2).
/// Every decode uses `rows / parts` rows per conversation (at most 6) and the same seeds. Checks: the
/// first verification's full-vocabulary logits (3 rows) bit for bit, then the emitted tokens and final
/// anchors over STEPS steps.
fn run_relocate(w: &mut Worker, rank: usize, out: &Path, kernels: &Path) -> Result<()> {
    let regions = crate::config::batched_regions()?;
    ensure!(w.current.is_none() && regions >= 2 && crate::config::batch_screen_sessions()? == 1
        && crate::config::reset_keeps_caches()?,
        "relocation screen needs an idle worker, serving regions and the kept-cache reset");
    let raw = include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture: Value = serde_json::from_slice(raw)?;
    let tokens: Vec<u32> = serde_json::from_value(fixture["tokens"].clone())?;
    let setting = |name: &str, default: usize| -> Result<usize> {
        match std::env::var(name) { Ok(v) => Ok(v.parse()?), Err(_) => Ok(default) } };
    let len_a = setting("CANADA_SCRATCH_RELOCATE_TOKENS", 30001)?;
    let len_q = setting("CANADA_SCRATCH_RELOCATE_SECOND_TOKENS", 20000)?;
    let steps = setting("CANADA_SCRATCH_BATCH_SCREEN_STEPS", 48)?;
    let rows = setting("CANADA_SCRATCH_BATCH_SCREEN_ROWS", 6)?;
    let parts_n = setting("CANADA_SCRATCH_BATCH_SCREEN_PARTS", 1)?;
    ensure!(matches!(rows, 6 | 16) && (1..=2).contains(&parts_n) && (8..=512).contains(&steps),
        "relocation screen rows/parts/steps");
    let limit = (rows / parts_n).min(6);
    let context = w.verify.context_capacity();
    let region_tokens = SessionBase::region_capacity(context, regions);
    let q_start = 100_000;
    ensure!(len_a >= 256 && len_q >= 256 && len_a + 2 <= q_start && q_start + len_q + 208 <= tokens.len()
        && len_a.max(len_q) + 8 * steps < region_tokens, "relocation screen prompt lengths");
    let (text_a, text_q) = (&tokens[..len_a], &tokens[q_start..q_start + len_q]);
    let (seed_a, seed_q) = (2026111100u64, 2026111101u64);
    if rows == 16 {
        let longer = w.long_verify.as_mut().context("relocation screen needs the 16-row verify owner")?;
        longer.handoff_from(&w.verify)?;
        longer.set_engram_overlap_for_probe(false)?;
        if !crate::config::batched_graphs_requested() { longer.set_graph_replay(false)?; }
        w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;
    }
    // One drafter per conversation run, forked before any prefill.
    let prepared = w.prepared_session.as_ref().context("prepared drafter")?;
    let mut forks = Vec::new();
    for _ in 0..if parts_n == 2 { 4 } else { 3 } {
        forks.push(prepared.independent_for_probe(&w.verify, &w.core, &w.stream, kernels)?);
    }
    let mut forks = forks.into_iter();
    let median = |v: &[f64]| { let mut s = v[2..].to_vec(); s.sort_by(|a, b| a.partial_cmp(b).unwrap()); s[s.len() / 2] };
    let rate = |emitted: &[Vec<u32>], ms: &[f64]| emitted.iter().map(|e| e.len()).sum::<usize>() as f64
        / (ms.iter().sum::<f64>() / 1000.);
    // References: each conversation alone in region 0.
    let (mut a_ref, a_ref_prefill) = prefill_conversation(w, text_a, seed_a, forks.next().unwrap())?;
    let ids_a = [a_ref.anchor, tokens[len_a], tokens[len_a + 1]];
    let logits_a = verify_abort(w, &[(a_ref.base, &a_ref.history, &ids_a[..])])?;
    let (mut ref_a, ref_a_ms) = decode_conversations(w, std::slice::from_mut(&mut a_ref), steps, rows, limit)?;
    let ref_a = ref_a.remove(0);
    let ref_a_anchor = a_ref.anchor;
    // With region tails (CANADA_SCRATCH_BATCHED_DECODE=1): the reference's encoder tail and next turn.
    let tails = super::batched_serving::enabled()?;
    let next_turn = &tokens[q_start + len_q + 8..q_start + len_q + 8 + 199];
    let back_ref = if tails {
        let digest = w.verify.encoder_tail_digest()?;
        Some((digest, continuation(w, ref_a_anchor, next_turn, seed_a)?))
    } else { None };
    drop(a_ref);
    let mut q_reference = None;
    if parts_n == 2 {
        let (mut q_ref, _) = prefill_conversation(w, text_q, seed_q, forks.next().unwrap())?;
        let ids_q = [q_ref.anchor, tokens[q_start + len_q], tokens[q_start + len_q + 1]];
        let logits_q = verify_abort(w, &[(q_ref.base, &q_ref.history, &ids_q[..])])?;
        let (mut ref_q, ref_q_ms) = decode_conversations(w, std::slice::from_mut(&mut q_ref), steps, rows, limit)?;
        let ref_q = ref_q.remove(0);
        q_reference = Some((ids_q, logits_q, ref_q, ref_q_ms, q_ref.anchor));
    }
    // Moved run.
    let (mut a_moved, a_prefill) = prefill_conversation(w, text_a, seed_a, forks.next().unwrap())?;
    let anchor_repeat = a_moved.anchor == ids_a[0];
    let region = SessionBase::serving_region(regions - 1, regions, context, w.verify.swa_layout_for_probe())?;
    let moved_at = Instant::now();
    let moved = w.verify.relocate_session(a_moved.base, region, len_a, &w.stream)?;
    let relocate_ms = moved_at.elapsed().as_secs_f64() * 1000.;
    a_moved.base = region;
    // Region 0 takes the next conversation (its reset clears region 0 only), overwriting A's old state there.
    let (q_conv, q_prefill) = prefill_conversation(w, text_q, seed_q, forks.next().unwrap())?;
    let logits_moved = verify_abort(w, &[(a_moved.base, &a_moved.history, &ids_a[..])])?;
    let mut checks = serde_json::Map::new();
    checks.insert("anchor_repeat_equal".into(), json!(anchor_repeat));
    checks.insert("logits_moved_equal".into(), json!(logits_moved == logits_a));
    let (convs, moved_tokens, moved_ms) = if let Some((ids_q, logits_q, _, _, ref_q_anchor)) = &q_reference {
        let pair = verify_abort(w, &[(q_conv.base, &q_conv.history, &ids_q[..]), (a_moved.base, &a_moved.history, &ids_a[..])])?;
        checks.insert("q_anchor_equal".into(), json!(q_conv.anchor == ids_q[0]));
        checks.insert("logits_pair_equal".into(), json!([pair[..3] == logits_q[..], pair[3..] == logits_a[..]]));
        let _ = ref_q_anchor;
        // Q (region 0) is part 0, the owner's own; A's rows start at the second block.
        let mut convs = vec![q_conv, a_moved];
        let (mut t, ms) = decode_conversations(w, &mut convs, steps, rows, limit)?;
        convs.swap(0, 1);
        t.swap(0, 1);
        (convs, t, ms)
    } else {
        drop(q_conv);
        let mut convs = vec![a_moved];
        let (t, ms) = decode_conversations(w, &mut convs, steps, rows, limit)?;
        (convs, t, ms)
    };
    let exact_a = moved_tokens[0] == ref_a && convs[0].anchor == ref_a_anchor;
    checks.insert("tokens_a_equal".into(), json!(exact_a));
    let mut exact = exact_a && anchor_repeat && logits_moved == logits_a;
    // Move A back to region 0 (region 0's conversation is discarded) and compare its encoder tail and
    // next turn with the reference that never left region 0.
    let mut back = Value::Null;
    if let Some((tail_ref, (greedy_ref, sampled_ref))) = &back_ref {
        let a = &convs[0];
        w.target_lineage = w.target_lineage.checked_add(1).context("relocation screen lineage")?;
        w.verify.reset(w.target_lineage, &w.comm, &w.stream)?;
        let moved_back = w.verify.relocate_session(a.base, SessionBase::default(), a.history.position as usize, &w.stream)?;
        w.verify.adopt_moved_history(a.history.clone())?;
        let tail = w.verify.encoder_tail_digest()?;
        let (greedy, sampled) = continuation(w, a.anchor, next_turn, seed_a)?;
        let tail_equal = tail == *tail_ref;
        let next_equal = greedy == *greedy_ref && sampled == *sampled_ref;
        checks.insert("moved_back_tail_equal".into(), json!(tail_equal));
        checks.insert("moved_back_next_turn_equal".into(), json!(next_equal));
        exact = exact && tail_equal && next_equal;
        back = json!({"relocation":moved_back,"tail":tail,"reference_tail":tail_ref,
            "next_turn":[greedy, sampled],"reference_next_turn":[greedy_ref, sampled_ref]});
    }
    let mut q_record = Value::Null;
    if let Some((ids_q, _, ref_q, ref_q_ms, ref_q_anchor)) = &q_reference {
        let exact_q = moved_tokens[1] == *ref_q && convs[1].anchor == *ref_q_anchor;
        checks.insert("tokens_q_equal".into(), json!(exact_q));
        exact = exact && exact_q && checks["q_anchor_equal"] == true
            && checks["logits_pair_equal"] == json!([true, true]);
        q_record = json!({"ids":ids_q,"reference_tokens":ref_q.len(),"reference_step_median_ms":median(ref_q_ms),
            "reference_tokens_per_second":rate(std::slice::from_ref(ref_q), ref_q_ms)});
    }
    drop(convs);
    let lengths: Vec<usize> = moved_tokens.iter().map(|e| e.len()).collect();
    eprintln!("rank{rank} relocation screen: exact {exact}; {regions} regions, moved to region {} ({} B in {} copies, {relocate_ms:.2} ms); checks {}; tokens {lengths:?} vs reference {}; step {:.2} ms (reference {:.2} ms); {:.1} tok/s (reference A alone {:.1})",
        regions - 1, moved["bytes"], moved["copies"], serde_json::to_string(&checks)?, ref_a.len(),
        median(&moved_ms), median(&ref_a_ms), rate(&moved_tokens, &moved_ms), rate(std::slice::from_ref(&ref_a), &ref_a_ms));
    w.control.agree(&[0x62, 0x72])?;
    std::fs::write(out.join("batch-screen-summary.json"), serde_json::to_vec_pretty(&json!({
        "rank":rank,"mode":"relocate","regions":regions,"context":context,"region_tokens":region_tokens,
        "records":[{"exact":exact,"checks":checks,"relocation":moved,"relocate_ms":relocate_ms,
            "region":format!("{region:?}"),"tokens_a":len_a,"tokens_q":len_q,"steps":steps,"owner_rows":rows,
            "conversations":parts_n,"rows_per_conversation":limit,
            "prefill_seconds":{"a_reference":a_ref_prefill,"a_moved":a_prefill,"q":q_prefill},
            "logits_a":logits_a,"logits_moved":logits_moved,
            "reference_a_tokens":ref_a.len(),"moved_tokens":lengths,
            "reference_step_median_ms":median(&ref_a_ms),"moved_step_median_ms":median(&moved_ms),
            "reference_a_tokens_per_second":rate(std::slice::from_ref(&ref_a), &ref_a_ms),
            "moved_tokens_per_second":rate(&moved_tokens, &moved_ms),
            "q":q_record,"move_back":back,"moved_step_ms":moved_ms,"reference_step_ms":ref_a_ms}],
        "scope":"Serving step 1: per-region small window rings, region-0-only reset, and moving a prefilled conversation into another region by page copies. Public fixture texts; same seeds; T1 sampling with the production drafter. Diagnostic only; no serving change."
    }))?)?;
    Ok(())
}

/// Timing only (CANADA_SCRATCH_BATCH_SCREEN_MODE=timing, CANADA_SCRATCH_BATCHED_REGIONS=4,
/// CANADA_SCRATCH_RESET_KEEP_CACHES=1, CANADA_SCRATCH_BATCHED_DECODE=1): for each length in
/// CANADA_SCRATCH_TIMING_TOKENS (default "1536,14000"), four conversations of different texts
/// prefilled by the ordinary wide path, three moved into regions 1..3 and one left in region 0, as
/// serving places them. Then:
/// - padding routes leave live rows unchanged: one batched verification with padding rows inside
///   blocks (lives 4, 2, 3, 2 on the 16-row owner), every part's logits hashed, routes off/on/off;
/// - for every padding-route setting in CANADA_SCRATCH_TIMING_PADDING (default "0,1") and batched
///   gate in CANADA_SCRATCH_TIMING_GATES (default "off,0.3,0.45,0.6"): four on 16 rows (4 each),
///   three on 16 rows (5 each), two on 6 rows (3 each) and two on 16 rows (6 each), each phase four
///   warm-up steps and STEPS measured steps (default 32);
/// - region 0's and region 3's conversation alone (6 rows), the single-conversation reference.
/// Reports step medians and means, tokens a step and a second, rows verified and executed a step,
/// and graph captures. CANADA_SCRATCH_BATCHED_PHASE_SYNC=1 adds the phase breakdown (synced steps).
fn run_timing(w: &mut Worker, rank: usize, out: &Path, kernels: &Path) -> Result<()> {
    let regions = crate::config::batched_regions()?;
    ensure!(w.current.is_none() && regions == 4 && crate::config::batch_screen_sessions()? == 1
        && crate::config::reset_keeps_caches()?,
        "timing screen needs an idle worker, four serving regions and the kept-cache reset");
    // CANADA_SCRATCH_TIMING_PROFILE=GATE (a threshold or "off"; runs under nsys with
    // --capture-range=cudaProfilerApi): after each length's sweep, four steps of four conversations on
    // 16 rows with padding routes and that gate, then four steps of region 0's conversation alone,
    // each one profiler range.
    let profile_gate: Option<Option<f32>> = match std::env::var("CANADA_SCRATCH_TIMING_PROFILE") {
        Ok(v) if v == "off" => Some(None),
        Ok(v) => Some(Some(v.parse::<f32>()?)),
        Err(_) => None,
    };
    let raw = include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture: Value = serde_json::from_slice(raw)?;
    let tokens: Vec<u32> = serde_json::from_value(fixture["tokens"].clone())?;
    let list = |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.into())
        .split(',').map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()).collect::<Vec<_>>();
    let lengths = list("CANADA_SCRATCH_TIMING_TOKENS", "1536,14000").iter().map(|s| s.parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let gates = list("CANADA_SCRATCH_TIMING_GATES", "off,0.3,0.45,0.6").iter()
        .map(|s| if s == "off" { Ok(None) } else { s.parse::<f32>().map(Some) })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let paddings = list("CANADA_SCRATCH_TIMING_PADDING", "0,1").iter()
        .map(|s| match s.as_str() { "0" => Ok(false), "1" => Ok(true), _ => Err(anyhow::anyhow!("timing padding setting")) })
        .collect::<Result<Vec<_>>>()?;
    let steps: usize = match std::env::var("CANADA_SCRATCH_BATCH_SCREEN_STEPS") { Ok(v) => v.parse()?, Err(_) => 32 };
    let context = w.verify.context_capacity();
    let region_tokens = SessionBase::region_capacity(context, regions);
    let multi_phases = paddings.len() * gates.len() * 4 + 2;
    ensure!((8..=512).contains(&steps) && !lengths.is_empty() && !gates.is_empty() && !paddings.is_empty()
        && gates.iter().all(|g| g.is_none_or(|g| (0.0..=1.0).contains(&g)))
        && lengths.iter().all(|&l| l >= 256 && l + 8 <= tokens.len()
            && l + 6 * multi_phases * (steps + 4) + 64 < region_tokens),
        "timing screen lengths/steps/gates");
    {
        // CANADA_SCRATCH_BATCHED_ENGRAM_OVERLAP=1: both verify owners overlap Engram lookups, as in
        // serving (the screens otherwise run without it).
        let overlap = crate::config::batched_engram_overlap() && crate::config::batched_graphs_requested();
        let longer = w.long_verify.as_mut().context("timing screen needs the 16-row verify owner")?;
        longer.handoff_from(&w.verify)?;
        longer.set_engram_overlap_for_probe(overlap)?;
        if !crate::config::batched_graphs_requested() { longer.set_graph_replay(false)?; }
        w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;
        if overlap { w.verify.set_engram_overlap_for_probe(true)?; }
    }
    let mut records = Vec::new();
    let mut checks = Vec::new();
    for &len in &lengths {
        let mut forks = Vec::new();
        {
            let prepared = w.prepared_session.as_ref().context("prepared drafter")?;
            for _ in 0..4 { forks.push(prepared.independent_for_probe(&w.verify, &w.core, &w.stream, kernels)?); }
        }
        // Conversations 3, 2 and 1 move into their regions; conversation 0 stays in region 0. Texts
        // start 30,000 tokens apart, closer (overlapping) for long conversations.
        let stride = 30_000.min((tokens.len() - len - 8) / 3);
        let mut convs: Vec<Conversation> = Vec::with_capacity(4);
        for (k, fork) in forks.into_iter().enumerate().rev() {
            let text = &tokens[stride * k..stride * k + len];
            let (mut c, _) = prefill_conversation(w, text, 2026111200 + k as u64, fork)?;
            if k > 0 {
                let region = SessionBase::serving_region(k, regions, context, w.verify.swa_layout_for_probe())?;
                w.verify.relocate_session(c.base, region, len, &w.stream)?;
                c.base = region;
            }
            convs.push(c);
        }
        convs.reverse();
        let check = padding_exactness(w, &convs, &tokens, len, stride)?;
        eprintln!("rank{rank} timing screen: {len} tokens, padding routes exact {} (repeatable {})",
            check["equal"], check["repeatable"]);
        checks.push(json!({"tokens":len,"padding_routes":check}));
        if crate::config::batched_draft_body() {
            // The batched drafter body against separate passes, before any step: two owners, then three.
            for n in [2usize, 3] {
                let anchors: Vec<u32> = convs[..n].iter().map(|c| c.anchor).collect();
                let mut sessions: Vec<&mut crate::dspark_generation::GreedySession> =
                    convs[..n].iter_mut().map(|c| &mut c.session).collect();
                let proof = crate::dspark_generation::GreedySession::probe_batched_draft_body(&mut sessions, &anchors,
                    &w.comm, &w.stream)?;
                eprintln!("rank{rank} timing screen: {len} tokens, batched drafter body of {n}: exact {}, separate {:.3} ms, batched {:.3} ms (best {:.3} and {:.3}); first difference {}",
                    proof["all_exact"], proof["separate_ms"].as_f64().unwrap_or(0.), proof["batched_ms"].as_f64().unwrap_or(0.),
                    proof["separate_best_ms"].as_f64().unwrap_or(0.), proof["batched_best_ms"].as_f64().unwrap_or(0.),
                    proof["first_difference"]);
                checks.push(json!({"tokens":len,"batched_draft_body":n,"proof":proof}));
            }
            // The same two owners in the other order: whether a difference follows the owner or its place.
            {
                let anchors = [convs[1].anchor, convs[0].anchor];
                let (a, b) = convs.split_at_mut(1);
                let mut sessions: Vec<&mut crate::dspark_generation::GreedySession> = vec![&mut b[0].session, &mut a[0].session];
                let proof = crate::dspark_generation::GreedySession::probe_batched_draft_body(&mut sessions, &anchors,
                    &w.comm, &w.stream)?;
                eprintln!("rank{rank} timing screen: {len} tokens, batched drafter body of owners 1 and 0: exact {}, first difference {}",
                    proof["all_exact"], proof["first_difference"]);
                checks.push(json!({"tokens":len,"batched_draft_body":"1,0","proof":proof}));
            }
        }
        let mut runs: Vec<(String, std::ops::Range<usize>, usize, usize, bool, Option<f32>)> = Vec::new();
        // CANADA_SCRATCH_TIMING_INTERLEAVE=1: per phase, every padding x gate setting takes turns step by
        // step over the same conversations, so each setting sees the same text (STEPS rounds).
        if std::env::var("CANADA_SCRATCH_TIMING_INTERLEAVE").as_deref() == Ok("1") {
            let configs: Vec<(bool, Option<f32>)> = paddings.iter().flat_map(|&p| gates.iter().map(move |&g| (p, g))).collect();
            // CANADA_SCRATCH_TIMING_TRACE=1: each setting's record carries its batched-step trace.
            let tracing = std::env::var("CANADA_SCRATCH_TIMING_TRACE").as_deref() == Ok("1");
            crate::dspark_generation::set_batched_trace_for_probe(tracing);
            for (name, range, rows, limit) in [("four on 16 rows", 0..4, 16, crate::target::batched_block(16, 4).min(6)), ("three on 16 rows", 0..3, 16, crate::target::batched_block(16, 3).min(6)),
                ("two on 6 rows", 0..2, 6, 3), ("two on 16 rows", 0..2, 16, crate::target::batched_block(16, 2).min(6))] {
                let mut stats: Vec<(Vec<f64>, usize, f64, f64, f64)> = vec![(Vec::new(), 0, 0., 0., 0.); configs.len()];
                let mut traces: Vec<Vec<Value>> = vec![Vec::new(); configs.len()];
                for round in 0..steps + 2 {
                    for (c, &(padded, gate)) in configs.iter().enumerate() {
                        w.verify.set_padding_routes_for_probe(padded)?;
                        w.long_verify.as_mut().unwrap().set_padding_routes_for_probe(padded)?;
                        crate::dspark_generation::set_batched_gate_for_probe(gate);
                        let _ = crate::dspark_generation::take_batched_phases();
                        let step = decode_conversations(w, &mut convs[range.clone()], 1, rows, limit);
                        crate::dspark_generation::set_batched_gate_for_probe(None);
                        let (emitted, ms) = step?;
                        let acc = crate::dspark_generation::take_batched_phases();
                        let trace = crate::dspark_generation::take_batched_trace();
                        if round < 2 { continue; }
                        traces[c].extend(trace);
                        let s = &mut stats[c];
                        s.0.push(ms[0]);
                        s.1 += emitted.iter().map(|e| e.len()).sum::<usize>();
                        s.2 += acc[0]["live_rows_per_step"].as_f64().unwrap_or(0.);
                        s.3 += acc[0]["executed_rows_per_step"].as_f64().unwrap_or(0.);
                        s.4 += acc[0]["lookup_drafts_per_step"].as_f64().unwrap_or(0.);
                    }
                }
                for (((padded, gate), (ms, produced, live, executed, lookups)), trace) in configs.iter().zip(stats).zip(traces) {
                    let mut sorted = ms.clone();
                    sorted.sort_by(|a, b| a.total_cmp(b));
                    let total: f64 = ms.iter().sum();
                    let n = ms.len() as f64;
                    let r = json!({"tokens":len,"phase":name,"interleaved":true,"padding_routes":padded,"gate":gate,
                        "owner_rows":rows,"rows_per_conversation":limit,"steps":ms.len(),
                        "step_median_ms":sorted[sorted.len() / 2],"step_mean_ms":total / n,
                        "tokens_per_step":produced as f64 / n,"tokens_per_second":produced as f64 / (total / 1000.),
                        "live_rows_per_step":live / n,"executed_rows_per_step":executed / n,"lookup_drafts_per_step":lookups / n,
                        "step_ms":ms,"trace":trace});
                    eprintln!("rank{rank} timing screen: {len} tokens, {name}, interleaved, padding routes {padded}, gate {gate:?}: median step {:.2} ms, mean {:.2}; {:.2} tokens/step, {:.1} tok/s; live {:.2}, executed {:.2}, lookups {:.2}",
                        sorted[sorted.len() / 2], total / n, produced as f64 / n, produced as f64 / (total / 1000.),
                        live / n, executed / n, lookups / n);
                    records.push(r);
                }
            }
            w.verify.set_padding_routes_for_probe(false)?;
            w.long_verify.as_mut().unwrap().set_padding_routes_for_probe(false)?;
            crate::dspark_generation::set_batched_trace_for_probe(false);
        } else {
        for &padded in &paddings {
            for &gate in &gates {
                for (name, range, rows, limit) in [("four on 16 rows", 0..4, 16, crate::target::batched_block(16, 4).min(6)), ("three on 16 rows", 0..3, 16, crate::target::batched_block(16, 3).min(6)),
                    ("two on 6 rows", 0..2, 6, 3), ("two on 16 rows", 0..2, 16, crate::target::batched_block(16, 2).min(6))] {
                    runs.push((name.to_owned(), range, rows, limit, padded, gate));
                }
            }
        }
        }
        runs.push(("region 0 alone".into(), 0..1, 6, 6, false, None));
        runs.push(("region 3 alone".into(), 3..4, 6, 6, false, None));
        let mut profiled = Vec::new();
        for (name, range, rows, limit, padded, gate) in runs {
            w.verify.set_padding_routes_for_probe(padded)?;
            w.long_verify.as_mut().unwrap().set_padding_routes_for_probe(padded)?;
            crate::dspark_generation::set_batched_gate_for_probe(gate);
            let r = timing_phase(w, &mut convs[range.clone()], steps, rows, limit);
            crate::dspark_generation::set_batched_gate_for_probe(None);
            let mut r = r?;
            let positions: Vec<u64> = convs[range.clone()].iter().map(|c| c.history.position).collect();
            eprintln!("rank{rank} timing screen: {len} tokens, {name}, padding routes {padded}, gate {gate:?}: median step {:.2} ms, mean {:.2}; {:.2} tokens/step, {:.1} tok/s; rows {}; captures {} + {}",
                r["step_median_ms"].as_f64().unwrap_or(0.), r["step_mean_ms"].as_f64().unwrap_or(0.),
                r["tokens_per_step"].as_f64().unwrap_or(0.), r["tokens_per_second"].as_f64().unwrap_or(0.),
                r["accounting"], r["warmup_captures"], r["measured_captures"]);
            r["tokens"] = json!(len); r["phase"] = json!(name); r["padding_routes"] = json!(padded);
            r["gate"] = json!(gate); r["owner_rows"] = json!(rows); r["rows_per_conversation"] = json!(limit);
            r["positions"] = json!(positions);
            records.push(r);
        }
        if let Some(gate) = profile_gate {
            // CANADA_SCRATCH_TIMING_PROFILE_CONVERSATIONS (default 4): the profiled batched phase's
            // conversation count on the 16-row owner (2: six rows each, 3: five, 4: four).
            let n: usize = match std::env::var("CANADA_SCRATCH_TIMING_PROFILE_CONVERSATIONS") { Ok(v) => v.parse()?, Err(_) => 4 };
            ensure!((2..=4).contains(&n), "profiled conversations");
            // CANADA_SCRATCH_TIMING_PROFILE_OWNER_ROWS=6 (two conversations only): the batched phase runs on the
            // 6-row owner, three rows each and no padding, as the timing phase "two on 6 rows" does.
            let owner: usize = match std::env::var("CANADA_SCRATCH_TIMING_PROFILE_OWNER_ROWS") { Ok(v) => v.parse()?, Err(_) => 16 };
            ensure!(owner == 16 || (owner == 6 && n == 2), "profiled owner rows");
            for (index, (range, rows, limit, padded)) in [(0..n, owner, (owner / n).min(6), owner == 16), (0..1, 6, 6, false)].into_iter().enumerate() {
                w.verify.set_padding_routes_for_probe(padded)?;
                w.long_verify.as_mut().unwrap().set_padding_routes_for_probe(padded)?;
                crate::dspark_generation::set_batched_gate_for_probe(gate);
                decode_conversations(w, &mut convs[range.clone()], 4, rows, limit)?;
                let _ = crate::dspark_generation::take_batched_phases();
                w.stream.synchronize()?;
                w.cuda.profiler_start()?;
                w.control.agree(&[0xd8, index as u8])?;
                let run = decode_conversations(w, &mut convs[range.clone()], 4, rows, limit);
                w.stream.synchronize()?;
                w.cuda.profiler_stop()?;
                crate::dspark_generation::set_batched_gate_for_probe(None);
                let (_, ms) = run?;
                profiled.push(json!({"conversations":range.len(),"owner_rows":rows,"padding_routes":padded,"gate":gate,
                    "step_ms":ms,"accounting":crate::dspark_generation::take_batched_phases()}));
            }
            checks.push(json!({"tokens":len,"profiled":profiled}));
        }
        w.verify.set_padding_routes_for_probe(false)?;
        w.long_verify.as_mut().unwrap().set_padding_routes_for_probe(false)?;
        drop(convs);
    }
    eprintln!("rank{rank} timing screen: region gather cache {}, checks passed {}", crate::config::region_gather_cache(),
        crate::config::REGION_GATHER_CHECKS.load(std::sync::atomic::Ordering::Relaxed));
    w.control.agree(&[0x62, 0x74])?;
    std::fs::write(out.join("batch-screen-summary.json"), serde_json::to_vec_pretty(&json!({
        "rank":rank,"mode":"timing","regions":regions,"context":context,"region_tokens":region_tokens,"steps":steps,
        "records":records,"checks":checks,
        "region_gather_cache":crate::config::region_gather_cache(),
        "region_gather_checks":crate::config::REGION_GATHER_CHECKS.load(std::sync::atomic::Ordering::Relaxed),
        "scope":"Batched decode step timing as serving places conversations (region 0 plus moved regions 1..3), at several prompt lengths, with and without padding routes and across batched draft gates. Public fixture texts; T1 sampling with the production drafter. Diagnostic only."
    }))?)?;
    Ok(())
}

/// One timing phase: four warm-up steps, then `steps` measured batched steps of `convs`.
fn timing_phase(w: &mut Worker, convs: &mut [Conversation], steps: usize, rows: usize, limit: usize) -> Result<Value> {
    fn captures(w: &Worker) -> usize {
        w.verify.graph_capture_count() + w.long_verify.as_ref().map_or(0, |t| t.graph_capture_count())
    }
    let before = captures(w);
    decode_conversations(w, convs, 4, rows, limit)?;
    let warm = captures(w);
    let _ = crate::dspark_generation::take_batched_phases();
    let (emitted, ms) = decode_conversations(w, convs, steps, rows, limit)?;
    let measured = captures(w) - warm;
    let accounting = crate::dspark_generation::take_batched_phases();
    let mut sorted = ms.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let produced: usize = emitted.iter().map(|e| e.len()).sum();
    let total_ms: f64 = ms.iter().sum();
    let emitted_sha256: Vec<String> = emitted.iter().map(|e| {
        format!("{:x}", Sha256::digest(e.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>()))[..16].to_owned()
    }).collect();
    Ok(json!({"emitted_sha256":emitted_sha256,"step_median_ms":sorted[sorted.len() / 2],"step_mean_ms":total_ms / ms.len() as f64,
        "step_max_ms":sorted[sorted.len() - 1],"tokens_per_step":produced as f64 / steps as f64,
        "tokens_per_second":produced as f64 / (total_ms / 1000.),"warmup_captures":warm - before,
        "measured_captures":measured,"accounting":accounting,"step_ms":ms}))
}

/// Padding routes leave every live row's logits unchanged: one batched verification on the 16-row
/// owner with padding rows inside blocks (lives 4, 2, 3, 2: rows 6-7 and 11 are padding), every
/// part's staged logits hashed, with padding routes off, on and off again (aborted each time).
fn padding_exactness(w: &mut Worker, convs: &[Conversation], tokens: &[u32], len: usize, stride: usize) -> Result<Value> {
    let lives = [4usize, 2, 3, 2];
    let ids: Vec<Vec<u32>> = convs.iter().zip(lives).enumerate().map(|(k, (c, live))| {
        let mut v = vec![c.anchor];
        v.extend_from_slice(&tokens[stride * k + len..stride * k + len + live - 1]);
        v
    }).collect();
    let all: Vec<u32> = ids.iter().flatten().copied().collect();
    let mut hashes: Vec<Vec<String>> = Vec::new();
    for padded in [false, true, false] {
        let parts = {
            let longer = w.long_verify.as_mut().context("16-row owner")?;
            longer.handoff_from(&w.verify)?;
            longer.set_padding_routes_for_probe(padded)?;
            longer.set_batched_rows_for_probe(convs.iter().zip(&ids).map(|(c, v)| (c.base, c.history.clone(), v.len())).collect())?;
            longer.begin_verification(&all, &w.comm, &w.stream)?;
            let mut parts = Vec::new();
            for (k, v) in ids.iter().enumerate() {
                let offset = longer.batched_offset(k)?;
                longer.stage_batched_session(offset, v.len(), &w.stream)?;
                w.stream.synchronize()?;
                let (logits, rows) = longer.pending_logits()?;
                ensure!(rows == v.len(), "padding exactness staged rows");
                let bytes = unsafe { w.cuda.read_device(logits.ptr(), rows * 129280 * 4) }?;
                parts.push(format!("{:x}", Sha256::digest(&bytes)));
            }
            longer.abort_batched_verification_for_probe(&w.stream)?;
            longer.set_padding_routes_for_probe(false)?;
            parts
        };
        w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;
        hashes.push(parts);
    }
    Ok(json!({"lives":lives,"hashes":hashes,"equal":hashes[0] == hashes[1],"repeatable":hashes[0] == hashes[2]}))
}

pub(super) fn run(w: &mut Worker, rank: usize, out: &Path, kernels: &Path) -> Result<()> {
    // The screens build sessions by six-row appends on the verify owner, which the
    // production Engram upload overlap (graph-replayed verification only) refuses.
    // The overlap only schedules the same uploads earlier, so it stays off here.
    w.verify.set_engram_overlap_for_probe(false)?;
    // Without CANADA_SCRATCH_BATCHED_GRAPHS a batched step of several sessions runs
    // eagerly, so every step of the screens then runs eagerly (the comparison isolates
    // batching; timings are eager). With it, every step replays graphs.
    if !crate::config::batched_graphs_requested() { w.verify.set_graph_replay(false)?; }
    if std::env::var("CANADA_SCRATCH_BATCH_SCREEN_MODE").as_deref() == Ok("relocate") {
        return run_relocate(w, rank, out, kernels);
    }
    if std::env::var("CANADA_SCRATCH_BATCH_SCREEN_MODE").as_deref() == Ok("timing") {
        return run_timing(w, rank, out, kernels);
    }
    if crate::config::batch_screen_sessions()? == 4 {
        if std::env::var("CANADA_SCRATCH_BATCH_SCREEN_MODE").as_deref() == Ok("generate") {
            return run_generate(w, rank, out, kernels);
        }
        return run_commit(w, rank, out);
    }
    ensure!(w.current.is_none() && crate::config::batch_screen_sessions()? == 2,
        "batched screen needs an idle worker sized for two sessions");
    let raw = include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture: Value = serde_json::from_slice(raw)?;
    let tokens: Vec<u32> = serde_json::from_value(fixture["tokens"].clone())?;
    let len = 1536;
    ensure!(tokens.len() > 60000 + len + 12, "batched screen fixture extent");
    let texts = [&tokens[..len + 12], &tokens[60000..60000 + len + 12]];
    let capacity = w.verify.context_capacity() / 2;
    let window = crate::swa_layout::OFFSET + w.verify.swa_layout_for_probe().ring().context("ring window layout")?;
    let bases = [SessionBase::region(0, capacity, window)?, SessionBase::region(1, capacity, window)?];
    let fresh = w.verify.history_for_probe();
    ensure!(fresh.position == 0, "batched screen needs a fresh verify owner");
    let mut histories = [fresh.clone(), fresh.clone()];
    let began = Instant::now();
    // Session A is built by the ordinary path (it owns the shared encoder-tail cache,
    // exactly as in serving); session B by batched appends in its own region.
    let own = w.verify.swap_history_for_probe(fresh.clone())?;
    for chunk in texts[0][..len].chunks(6) { w.verify.append(chunk, &w.comm, &w.stream)?; }
    histories[0] = w.verify.swap_history_for_probe(own)?;
    for chunk in texts[1][..len].chunks(6) {
        w.verify.set_batched_rows_for_probe(vec![(bases[1], histories[1].clone(), chunk.len())])?;
        w.verify.append(chunk, &w.comm, &w.stream)?;
        histories[1] = w.verify.take_batched_histories_for_probe()?.pop().context("advanced history")?;
    }
    ensure!(histories.iter().all(|h| h.position == len as u64), "session frontiers after build");
    let build_seconds = began.elapsed().as_secs_f64();
    let mut records = Vec::new();
    for (trial, offset) in [len, len + 3, len, len + 3].into_iter().enumerate() {
        let next = [&texts[0][offset..offset + 3], &texts[1][offset..offset + 3]];
        // Session rows verified alone through the override, then the ordinary path for session A.
        let mut alone = Vec::new(); let mut alone_ms = Vec::new();
        for s in 0..2 {
            let t = Instant::now();
            w.verify.set_batched_rows_for_probe(vec![(bases[s], histories[s].clone(), 3)])?;
            w.verify.begin_verification(next[s], &w.comm, &w.stream)?;
            alone_ms.push(t.elapsed().as_secs_f64() * 1000.);
            alone.push(row_hashes(w, 3)?);
            w.verify.abort_batched_verification_for_probe(&w.stream)?;
        }
        let own = w.verify.swap_history_for_probe(histories[0].clone())?;
        w.verify.begin_verification(next[0], &w.comm, &w.stream)?;
        let ordinary = row_hashes(w, 3)?;
        w.verify.finish_verification(0, &w.comm, &w.stream)?;
        let back = w.verify.swap_history_for_probe(own)?;
        ensure!(back == histories[0], "ordinary rollback advanced session A");
        // Both sessions' rows in one step.
        let ids: Vec<u32> = next[0].iter().chain(next[1]).copied().collect();
        let t = Instant::now();
        w.verify.set_batched_rows_for_probe(vec![(bases[0], histories[0].clone(), 3), (bases[1], histories[1].clone(), 3)])?;
        w.verify.begin_verification(&ids, &w.comm, &w.stream)?;
        let batched_ms = t.elapsed().as_secs_f64() * 1000.;
        let batched = row_hashes(w, 6)?;
        w.verify.abort_batched_verification_for_probe(&w.stream)?;
        let exact_a = batched[..3] == alone[0][..];
        let exact_b = batched[3..] == alone[1][..];
        let override_equals_ordinary = ordinary == alone[0];
        eprintln!("rank{rank} batched screen trial{trial}: exact A {exact_a} B {exact_b} override==ordinary {override_equals_ordinary}; alone {:.2}+{:.2} ms, batched {:.2} ms",
            alone_ms[0], alone_ms[1], batched_ms);
        records.push(json!({"trial":trial,"offset":offset,"exact_session_a":exact_a,"exact_session_b":exact_b,
            "override_equals_ordinary":override_equals_ordinary,"alone_ms":alone_ms,"batched_ms":batched_ms,
            "alone_rows":alone,"batched_rows":batched,"ordinary_rows":ordinary}));
    }
    w.control.agree(&[0x62, 0x73])?;
    std::fs::write(out.join("batch-screen-summary.json"), serde_json::to_vec_pretty(&json!({
        "rank":rank,"session_tokens":len,"session_capacity":capacity,"window_region_slots":window,
        "bases":[format!("{:?}",bases[0]),format!("{:?}",bases[1])],"build_seconds":build_seconds,
        "fixture_sha256":format!("{:x}",Sha256::digest(raw)),"records":records,
        "scope":"Two public texts as two sessions in one six-row verify owner (regions per session, two compressor request rings); A built by ordinary six-row appends (it owns the shared encoder-tail cache), B by batched appends in its own region. Each trial verifies 3 rows of each session alone and both in one 6-row step; full-vocabulary logits compared per row by SHA-256. Every verification is aborted with a full carry restore. Diagnostic only; no serving change."
    }))?)?;
    Ok(())
}
