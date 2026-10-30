//! Small paired screen of shared temporary arenas across prefill widths.
use crate::{checkpoint::Checkpoint, cuda::{Cuda, Stream},
    dspark_generation::{GreedySession, teacher_forced_logits},
    layout::RankPlan, roce::Roce, target::Target};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

fn save_logits(output: &Path, name: &str, rows: &[Vec<f32>]) -> Result<Vec<String>> {
    let mut bytes = Vec::new();
    let mut hashes = Vec::new();
    for row in rows {
        ensure!(row.len() == 129280 && row.iter().all(|v| v.is_finite()),
            "wide bank finite full-vocabulary logits");
        let first = bytes.len();
        bytes.extend(row.iter().flat_map(|v| v.to_le_bytes()));
        hashes.push(format!("{:x}", Sha256::digest(&bytes[first..])));
    }
    std::fs::write(output.join(name), bytes)?;
    Ok(hashes)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda: &Cuda, core: &[u8], stream: &Stream, cp: &Checkpoint, plan: &RankPlan,
    large: &mut Target, verify: &mut Target, wides: &mut [Target], comm: &Roce,
    staged: &Path, kernels: &Path, output: &Path,
) -> Result<Value> {
    let rank = plan.rank;
    let shared_candidates = std::env::var("CANADA_SCRATCH_SHARED_CANDIDATE_SCRATCH").as_deref() == Ok("1");
    let local_scores = std::env::var("CANADA_SCRATCH_LOCAL_CANDIDATE_SCORES").as_deref() == Ok("1");
    ensure!(!local_scores || shared_candidates, "local score screen requires the shared bank");
    let profiles: Vec<_> = wides.iter().filter_map(Target::wide_report).collect();
    let retained_publication = local_scores
        && std::env::var("CANADA_SCRATCH_RETAINED_CANDIDATE_PUBLICATION").as_deref() != Ok("0");
    let compact_candidate_gather = retained_publication
        && std::env::var("CANADA_SCRATCH_COMPACT_CANDIDATE_GATHER").as_deref() != Ok("0");
    let compact_mask = retained_publication
        && std::env::var("CANADA_SCRATCH_COMPACT_RETAINED_MASK").as_deref() != Ok("0");
    let compact_scores=retained_publication
        &&std::env::var("CANADA_SCRATCH_COMPACT_RETAINED_SCORES").as_deref()!=Ok("0");
    ensure!(profiles.iter().all(|p| p["candidate_scratch"]["retained_only"] == retained_publication),
        "wide bank retained publication policy was not applied");
    ensure!(profiles.iter().all(|p| {
        let enabled = (compact_candidate_gather||compact_scores) && p["rows"].as_u64().unwrap_or(0) > 512;
        p["candidate_scratch"]["compact_gather"] == enabled
            && p["candidate_scratch"]["compact_gather_bytes"] == if enabled {4194304} else {0}
    }), "wide bank compact candidate gather policy was not applied");
    let expected = if shared_candidates { vec![8192, 4096, 2048, 1024, 512] }
        else { vec![8192, 4096, 1024, 512] };
    ensure!(large.context_capacity() == 1048576 && verify.head_gemv_enabled()
        && std::env::var("CANADA_SCRATCH_SHARED_WIDE_ARENAS").as_deref() == Ok("1")
        && profiles.iter().map(|p| p["rows"].as_u64().unwrap_or(0)).collect::<Vec<_>>()
            == expected
        && profiles[0]["shared_row_arenas"] == false
        && profiles[0]["owned_arena_bytes"] == json!(6 * 1024usize.pow(3))
        && profiles[1..].iter().all(|p| p["shared_row_arenas"] == true
            && p["owned_arena_bytes"] == 0), "isolated shared wide bank geometry");
    if shared_candidates {
        let expected_bytes = (if compact_scores {64*1024usize.pow(2)}
            else {(if local_scores {1} else {4}) * 1024usize.pow(3)})
            + if compact_mask {16 * 1024usize.pow(2)} else {1024usize.pow(3)};
        ensure!(profiles[0]["candidate_scratch"]["owned_bytes"] == json!(expected_bytes)
            && profiles[0]["candidate_scratch"]["shared"] == false
            && profiles[1..].iter().all(|p| p["candidate_scratch"]["shared"] == true
                && p["candidate_scratch"]["owned_bytes"] == 0
                && p["candidate_scratch"]["backing_rows"] == if compact_scores {512}else{8192})
            && profiles.iter().all(|p| p["candidate_scratch"]["score_shards"] == if local_scores {4} else {1}),
            "wide bank candidate scratch ownership");
        if std::env::var("CANADA_SCRATCH_WIDTH_CALIBRATION").as_deref() == Ok("1") {
            ensure!(local_scores && verify.paired_engram_io_enabled()
                && verify.engram_input_kind() == "mapped", "width calibration candidate policy");
            return calibrate_widths(cuda, stream, large, verify, wides, comm, output, rank, profiles);
        }
        // Preserve s391's exact profile choices for its first two cases, so
        // their raw logits/state can be compared across the memory-only change.
        // Restore the complete bank for the new 2K case and generation below.
        wides[2..].rotate_left(1);
    }
    std::fs::write(output.join("bank-memory-after-profiles.json"),
        serde_json::to_vec_pretty(&cuda.admission_memory()?)?)?;
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/long-context-continuation-v2.json"))?;
    let corpus: Vec<u32> = serde_json::from_value(fixture["tokens"].clone())?;
    let teacher: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/fixed-verify-python-v1.json"))?;
    let teacher: Vec<u32> = serde_json::from_value(teacher["teacher_forced_tokens"].clone())?;
    ensure!(corpus.len() == 262656 && teacher.len() >= 12, "wide bank fixture geometry");
    large.enable_cached_extend_floor()?;
    let mut lineage = 170000;
    let mut initial = true;
    let mut records = Vec::new();
    // A short fresh request and one retained long-context update. Both switch
    // widths, use a partial final chunk, and repeat with changed new tokens.
    let mut cases = vec![(0usize, 8577usize), (131072, 9473)];
    if shared_candidates { cases.push((0, 2305)); }
    for (case, (prefix, delta)) in cases.into_iter().enumerate() {
        if shared_candidates && case == 2 { wides[2..].rotate_right(1); }
        let bank_end = if case < 2 {4} else {wides.len()};
        // The local-score retry changes storage only. One pair per case is
        // enough to compare bytes and exercise memory; no new warm timing claim.
        for repeat in 0..if local_scores {1} else {2} {
            let mut paths = [Value::Null, Value::Null];
            for path in [repeat % 2, 1 - repeat % 2] {
                if !initial {
                    lineage += 1;
                    verify.reset(lineage, comm, stream)?;
                    large.handoff_from(verify)?;
                }
                initial = false;
                if prefix != 0 {
                    // The same 4K preparation gives both alternatives the
                    // identical incoming prefix independently of the candidate.
                    large.prefill_wide(&mut wides[1..bank_end], &corpus[..prefix], comm, stream)?;
                }
                let incoming = json!({"position":large.position(),
                    "encoder":large.encoder_tail_digest()?,"carry":large.carry_digests()?});
                let begin = Instant::now();
                let bank = if path == 0 { &mut wides[1..bank_end] } else { &mut wides[..bank_end] };
                let first = prefix + repeat * 17;
                large.prefill_wide(bank, &corpus[first..first + delta], comm, stream)?;
                let anchor = large.greedy_token(stream)?;
                let seconds = begin.elapsed().as_secs_f64();
                ensure!(large.position() as usize == prefix + delta, "wide bank frontier");
                let phases = serde_json::to_value(large.append_metrics()?)?;
                ensure!(repeat == 0 || phases["captured_new_graph"] == false,
                    "measured wide bank recaptured decoder tail");
                let outgoing = json!({"position":large.position(),"anchor":anchor,
                    "encoder":large.encoder_tail_digest()?,"carry":large.carry_digests()?});
                let mut logits = vec![large.last_prompt_logits()?];
                verify.handoff_from(large)?;
                logits.extend(teacher_forced_logits(cuda, verify, &teacher[..12], 2, comm, stream)?);
                let name = format!("bank-c{case}-r{repeat}-p{path}-logits.bin");
                let hashes = save_logits(output, &name, &logits)?;
                paths[path] = json!({"path":path,"seconds":seconds,"phases":phases,
                    "incoming":incoming,"outgoing":outgoing,"logits_file":name,
                    "logit_rows_sha256":hashes,"final_frontier":verify.position()});
            }
            let exact = paths[0]["incoming"] == paths[1]["incoming"]
                && paths[0]["outgoing"] == paths[1]["outgoing"]
                && paths[0]["logit_rows_sha256"] == paths[1]["logit_rows_sha256"];
            let record = json!({"case":case,"prefix_tokens":prefix,"new_tokens":delta,
                "repeat":repeat,"warmup":repeat==0,"paths":paths,"all_exact":exact});
            std::fs::write(output.join(format!("bank-c{case}-r{repeat}.json")),
                serde_json::to_vec_pretty(&record)?)?;
            eprintln!("wide bank rank{rank} prefix{prefix} delta{delta} repeat{repeat}: exact={exact}");
            records.push(record);
        }
    }
    let mut generation = Value::Null;
    if shared_candidates {
        // Keep the full profile bank resident while loading and exercising the
        // drafter. This is a memory/functionality screen, not a TPS benchmark.
        lineage += 1;
        verify.reset(lineage, comm, stream)?;
        large.handoff_from(verify)?;
        let prompt = &corpus[..8577];
        large.prefill_wide(wides, prompt, comm, stream)?;
        let seed = 1017;
        let mut anchor = large.sample_token(seed, stream)?;
        let memory_before_draft = cuda.admission_memory()?;
        std::fs::write(output.join("bank-memory-before-draft.json"),
            serde_json::to_vec_pretty(&memory_before_draft)?)?;
        let mut session = GreedySession::from_prefill(large, cuda, core, stream, cp, plan, staged, kernels)?;
        session.configure_sampling(seed, None)?;
        session.set_causal_three_threshold_for_probe(Some(0.20))?;
        std::fs::write(output.join("bank-memory-after-draft-load.json"),
            serde_json::to_vec_pretty(&cuda.admission_memory()?)?)?;
        session.configure_history_lookup(large, prompt)?;
        ensure!(session.conditioning_split() && !session.block_verification()
            && session.draft_temperature() == 1., "wide bank drafter policy");
        verify.handoff_from(large)?;
        let mut emitted = vec![anchor];
        let mut steps = Vec::new();
        while emitted.len() < 64 {
            let step = session.step(verify, anchor, (64-emitted.len()).min(6), comm, stream)?;
            ensure!(step.tokens[0] == anchor, "wide bank pending anchor mismatch");
            emitted.extend_from_slice(&step.tokens[1..]);
            emitted.push(step.next_anchor);
            anchor = step.next_anchor;
            steps.push(step);
        }
        stream.synchronize()?;
        comm.health()?;
        ensure!(emitted.len() == 64 && verify.position() == (prompt.len()+63) as u64,
            "wide bank generation frontier");
        let memory_with_draft = cuda.admission_memory()?;
        ensure!(memory_with_draft.admitted_available >= 8 * 1024usize.pow(3),
            "wide bank plus drafter consumed reserved headroom");
        generation = json!({"prompt_tokens":prompt.len(),"seed":seed,"temperature":1.,"top_p":1.,
            "output_ids":emitted,"steps":steps,"frontier":verify.position(),
            "draft_head":session.draft_head_kind(),"conditioning_split":session.conditioning_split(),
            "target_head":"canonical_bf16_weights_fp32_output", "memory_before_draft":memory_before_draft,
            "memory_with_draft":memory_with_draft,"scope":"64-token functionality and memory check; no timing claim"});
        std::fs::write(output.join("bank-generation.json"), serde_json::to_vec_pretty(&generation)?)?;
    }
    Ok(json!({"rank":rank,"wide_bank_screen":true,"context_capacity":1048576,
        "shared_candidates":shared_candidates,"local_scores":local_scores,
        "retained_publication":retained_publication,"compact_candidate_gather":compact_candidate_gather,
        "generation":generation,
        "encoder_profiles":profiles,"all_exact":records.iter().all(|r|r["all_exact"]==true),
        "records":records,"memory":cuda.admission_memory()?,"full_engine_acceptance":false,
        "scope":if local_scores {"Local quarter of candidate score rows plus shared scratch across8192/4096/2048/1024/512. One pair each for fresh8577, retained131072+9473 and fresh2305; same input/chunk choices as s391/s394. Then64sampled outputs with drafter/history resident and unchanged8GiB reserve. Memory and byte/state screen only; no speed, outcome or production claim."}
        else if shared_candidates {"Shared candidate score/mask scratch across8192/4096/2048/1024/512. Preserve the two s391 cases and chunk choices; add fresh2305 to exercise2K and64sampled outputs with all profiles and drafter resident. Full-vocabulary byte/state checks and unchanged8GiB reserve. No outcome or production claim."}
        else {"Two paired cases, fresh8577 and retained131072+9473; one warmup/one measured pair with changed new tokens and alternating order. Shared transient arenas across8192/4096/1024/512. Same incoming4K prefix, final prompt and12following full-vocabulary logit hashes plus retained carry/encoder/frontier checks. No sampled generation, outcome or production claim."}}))
}

/// Compare width changes with the same shared memory and token positions.
/// A repeated 4K path after both alternatives checks stale arena/state writes.
#[allow(clippy::too_many_arguments)]
fn calibrate_widths(cuda: &Cuda, stream: &Stream, large: &mut Target, verify: &mut Target,
    wides: &mut [Target], comm: &Roce, output: &Path, rank: usize,
    profiles: Vec<Value>,
) -> Result<Value> {
    let raw = include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture: Value = serde_json::from_slice(raw)?;
    let corpus: Vec<u32> = serde_json::from_value(fixture["tokens"].clone())?;
    let teacher: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/fixed-verify-python-v1.json"))?;
    let teacher: Vec<u32> = serde_json::from_value(teacher["teacher_forced_tokens"].clone())?;
    ensure!(corpus.len() == 262656 && teacher.len() >= 12, "width calibration fixture");
    large.enable_cached_extend_floor()?;
    let mut records = Vec::<Value>::new();
    for (path, start) in [1usize, 2, 0, 1].into_iter().enumerate() {
        if path != 0 {
            verify.reset(190000 + path as u64, comm, stream)?;
            large.handoff_from(verify)?;
        }
        let max_rows = [4096, 2048, 8192, 4096][path];
        ensure!(wides[start].wide_report().unwrap()["rows"] == max_rows,
            "width calibration bank order");
        let incoming = json!({"position":large.position(),
            "encoder":large.encoder_tail_digest()?, "carry":large.carry_digests()?});
        ensure!(large.position() == 0, "width calibration empty initial state");
        let begin = Instant::now();
        large.prefill_wide(&mut wides[start..], &corpus[..8577], comm, stream)?;
        let anchor = large.greedy_token(stream)?;
        let seconds = begin.elapsed().as_secs_f64();
        let phases = serde_json::to_value(large.append_metrics()?)?;
        ensure!(large.position() == 8577, "width calibration prompt frontier");
        let outgoing = json!({"position":large.position(), "anchor":anchor,
            "encoder":large.encoder_tail_digest()?, "carry":large.carry_digests()?});
        let mut logits = vec![large.last_prompt_logits()?];
        verify.handoff_from(large)?;
        logits.extend(teacher_forced_logits(cuda, verify, &teacher[..12], 2, comm, stream)?);
        let name = format!("width-calibration-p{path}-logits.bin");
        let hashes = save_logits(output, &name, &logits)?;
        ensure!(verify.position() == 8589, "width calibration continuation frontier");
        let record = json!({"path":path, "maximum_width":max_rows, "seconds":seconds,
            "phases":phases, "incoming":incoming, "outgoing":outgoing,
            "logits_file":name, "logit_rows_sha256":hashes, "final_frontier":verify.position()});
        if path > 0 {
            ensure!(record["incoming"] == records[0]["incoming"], "width calibration initial state differs");
        }
        if path == 3 {
            for key in ["outgoing", "logit_rows_sha256", "final_frontier"] {
                ensure!(record[key] == records[0][key], "repeated 4K width changed {key}");
            }
        }
        std::fs::write(output.join(format!("width-calibration-p{path}.json")),
            serde_json::to_vec_pretty(&record)?)?;
        eprintln!("width calibration rank{rank} path{path} maximum{max_rows}: {seconds:.4}s");
        records.push(record);
    }
    Ok(json!({"rank":rank, "width_calibration":true, "context_capacity":1048576,
        "fixture_sha256":format!("{:x}", Sha256::digest(raw)),
        "prompt_tokens":8577, "teacher_forced_tokens":&teacher[..12],
        "encoder_profiles":profiles, "records":records, "repeated_control_exact":true,
        "memory":cuda.admission_memory()?, "full_engine_acceptance":false,
        "scope":"One fresh8577-token fixture, 4K/2K/8K/4K maximum widths in the same shared bank. Final prompt plus12 teacher-forced rows. Repeated4K state/logits must be exact. GPU distribution comparison is descriptive, not a quality or performance qualification."}))
}
