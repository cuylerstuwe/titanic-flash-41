//! Discrete token-index opportunity screen; no CPU model/probability arithmetic.
#[path = "../src/history_lookup.rs"]
mod history_lookup;

#[cfg(not(feature = "protocol"))]
fn main() { panic!("use --features protocol"); }

#[cfg(feature = "protocol")]
fn main() -> anyhow::Result<()> {
    use anyhow::ensure;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::{path::Path, time::Instant};
    let args: Vec<_> = std::env::args_os().collect();
    ensure!(args.len() == 3, "input and immutable output paths");
    let raw = std::fs::read(&args[1])?;
    let input: Value = serde_json::from_slice(&raw)?;
    let output = Path::new(&args[2]); ensure!(!output.exists(), "output exists");
    let tokenizer_path = Path::new(input["tokenizer"].as_str().unwrap());
    let frontend = canada_engine::protocol::Frontend::load(tokenizer_path)?;
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut results = Vec::new();
    for case in input["cases"].as_array().unwrap() {
        let mut prefix: Vec<u32> = if let Some(request) = case["request"].as_str() {
            frontend.prepare_chat(&std::fs::read(request)?, 1048576)?.tokens
        } else {
            let raw = std::fs::read(case["fixture"].as_str().unwrap())?;
            ensure!(format!("{:x}", Sha256::digest(&raw)) == case["fixture_sha256"].as_str().unwrap(), "fixture hash");
            let fixture: Value = serde_json::from_slice(&raw)?;
            serde_json::from_value::<Vec<u32>>(fixture["tokens"].clone())?[..case["prefix_length"].as_u64().unwrap() as usize].to_vec()
        };
        if let Some(warmup) = case.get("warmup") { prefix.extend(serde_json::from_value::<Vec<u32>>(warmup.clone())?); }
        let (emitted, provenance): (Vec<u32>, &str) = if let Some(next) = case["next_request"].as_str() {
            let full = frontend.prepare_chat(&std::fs::read(next)?, 1048576)?.tokens;
            ensure!(full.starts_with(&prefix), "next request changed prompt prefix");
            let end = prefix.len() + case["completion_tokens"].as_u64().unwrap() as usize;
            ensure!(end <= full.len(), "completion extends past next prompt");
            if let Some(proof) = case.get("cache_reuse_proof") {
                let raw = std::fs::read(proof["events"].as_str().unwrap())?;
                ensure!(format!("{:x}", Sha256::digest(&raw)) == proof["sha256"].as_str().unwrap(), "cache evidence hash");
                let records: Vec<Value> = std::str::from_utf8(&raw)?.lines()
                    .map(serde_json::from_str).collect::<std::result::Result<_,_>>()?;
                let turn = proof["turn"].as_u64().unwrap();
                let response = |n| records.iter().find(|v| v["kind"] == "model_response" && v["turn"] == n)
                    .map(|v| &v["response"]["usage"]);
                let previous = response(turn).ok_or_else(|| anyhow::anyhow!("missing previous response"))?;
                let following = response(turn + 1).ok_or_else(|| anyhow::anyhow!("missing following response"))?;
                ensure!(previous["prompt_tokens"].as_u64() == Some(prefix.len() as u64)
                    && previous["completion_tokens"] == case["completion_tokens"], "previous usage mismatch");
                ensure!(following["prompt_tokens"].as_u64() == Some(full.len() as u64)
                    && following["prompt_cache_hit_tokens"].as_u64().is_some_and(|n| n >= end as u64),
                    "following request did not prove complete generated-prefix reuse");
            }
            (full[prefix.len()..end].to_vec(), if case.get("cache_reuse_proof").is_some() {
                "canonical next-request prefix, with saved usage proving exact reuse through the full emitted sequence"
            } else { "canonical next-request prefix; reuse proof supplied by the original case's external evidence" })
        } else if let Some(text) = case["output_text"].as_str() {
            (tokenizer.encode(text, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec(), "retokenized public reasoning text; token count checked, raw generation IDs unavailable")
        } else { (serde_json::from_value(case["tokens"].clone())?, "saved raw target token IDs") };
        if let Some(expected) = case["prompt_tokens"].as_u64() { ensure!(prefix.len() == expected as usize, "prompt token count for {}: {} vs {}",case["name"],prefix.len(),expected); }
        let retokenization_delta = case["completion_tokens"].as_i64().map(|n| emitted.len() as i64 - n);
        if provenance.starts_with("retokenized") { ensure!(retokenization_delta.is_some_and(|d| d.abs() <= 2), "unexpected retokenization length drift"); }
        let prefix_length = prefix.len();
        let mut lookup = history_lookup::HistoryLookup::with_maximum(1048576, &prefix, 262144, 64)?;
        let mut lengths = Vec::new(); let mut frontiers = Vec::new(); let mut last_sample = None;
        let sample_limit = case["sample_frontiers"].as_u64().unwrap_or(0) as usize;
        ensure!(sample_limit <= 4, "bounded opportunity samples");
        let tool_only = case["sample_inside_tool_calls"].as_bool().unwrap_or(false);
        let mut in_tool_call = false;
        let began = Instant::now();
        for (offset, &anchor) in emitted.iter().enumerate() {
            if tool_only {
                // Decode a bounded *past* suffix, so BPE punctuation/newline
                // merges around the canonical spaced DSML marker are harmless.
                let tail=tokenizer.decode(&emitted[offset.saturating_sub(15)..=offset],false)
                    .map_err(|e|anyhow::anyhow!("{e}"))?;
                if tail.contains("<｜DSML｜ calls>") || tail.contains("<｜DSML｜tool_calls>") {in_tool_call=true;}
                if tail.contains("</｜DSML｜ calls>") || tail.contains("</｜DSML｜tool_calls>") {in_tool_call=false;}
            }
            let proposal = lookup.propose(lookup.position(), anchor)?;
            let (available, matched) = if let Some(p) = proposal {
                let available = (prefix.len() - p.source_end - 1).min(64).min(emitted.len() - offset - 1);
                let matched = prefix[p.source_end+1..p.source_end+1+available].iter()
                    .zip(&emitted[offset+1..offset+1+available]).take_while(|(a,b)| a == b).count();
                // Selection sees only the committed prefix, pending anchor,
                // match length, and known source availability. It never uses
                // the observed future match computed above. Fixed spacing
                // avoids spending both samples on adjacent copying positions.
                if p.matched >= 16 && prefix.len() - p.source_end - 1 >= 15
                    && (!tool_only || in_tool_call)
                    && frontiers.len() < sample_limit
                    && last_sample.is_none_or(|n| offset >= n + 512) {
                    frontiers.push(json!({"offset":offset,"prefix_match":p.matched,
                        "source_end":p.source_end,"prompt_ids":prefix,"anchor":anchor,
                        "proposal_ids":prefix[p.source_end+1..p.source_end+16],
                        "observed_future_match_diagnostic_only":matched}));
                    last_sample = Some(offset);
                }
                (available, matched)
            } else { (0, 0) };
            lengths.push((available, matched));
            lookup.commit(lookup.position(), &[anchor])?; prefix.push(anchor);
        }
        let coverage: Vec<_> = [8,16,32,64].into_iter().map(|n| {
            let mut covered = vec![false; emitted.len()]; let mut starts = 0;
            for (offset, &(_, matched)) in lengths.iter().enumerate() {
                if matched >= n { starts += 1; covered[offset+1..offset+1+matched].fill(true); }
            }
            json!({"minimum_matched_continuation":n,"eligible_starts":lengths.iter().filter(|x|x.0>=n).count(),
                "matching_starts":starts,"observed_covered_tokens":covered.iter().filter(|&&b|b).count(),
                "observed_covered_fraction":covered.iter().filter(|&&b|b).count() as f64 / emitted.len() as f64})
        }).collect();
        results.push(json!({"name":case["name"],"group":case["group"],"prompt_tokens":prefix_length,
            "output_tokens":emitted.len(),"output_provenance":provenance,"retokenization_token_delta":retokenization_delta,
            "maximum_observed_match":lengths.iter().map(|x|x.1).max(),"coverage":coverage,
            "frontier_samples":frontiers,"replay_seconds":began.elapsed().as_secs_f64()}));
    }
    std::fs::write(output, serde_json::to_vec_pretty(&json!({"input_sha256":format!("{:x}",Sha256::digest(&raw)),
        "cases":results,"scope":"Current causal history lookup source selection, extended through at most64 already committed source tokens. Observed future output used only to measure opportunity. Overlapping coverage is an optimistic proxy on these saved trajectories, not target probability, a causal length-selection policy, TPS or a quality result. No inference or serving change."}))?)?;
    Ok(())
}
