//! Finite three-token distribution screen through ordinary Begin/Step/Finish.
//! Short prompts are admitted to the existing exact-state cache only to avoid
//! repeated prefill. Sampling, proposals, verification and rollback are intact.
use super::*;

pub(super) fn enabled() -> bool {
    std::env::var("CANADA_SCRATCH_SAMPLER_LAW_SCREEN").as_deref() == Ok("1")
}

fn top4(w: &mut Worker, module: &crate::cuda::Module, pointer: u64) -> Result<Vec<u32>> {
    let output = w.cuda.allocate::<i32>(4)?;
    let (mut source, mut destination, mut eos) = (pointer, output.ptr(), w.cfg.eos_token_id as i32);
    unsafe {
        module.function("ca_sampler_law_top4")?.launch(
            [1, 1, 1],
            [256, 1, 1],
            0,
            &w.stream,
            &mut [
                (&mut source as *mut u64).cast(),
                (&mut destination as *mut u64).cast(),
                (&mut eos as *mut i32).cast(),
            ],
        )?;
    }
    w.stream.synchronize()?;
    let ids = output.download()?;
    ensure!(
        ids.iter().all(|&t| (0..129280).contains(&t) && t != eos),
        "law top-four IDs"
    );
    w.control
        .agree(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    Ok(ids.into_iter().map(|v| v as u32).collect())
}

fn oracle_row(
    w: &mut Worker,
    module: &crate::cuda::Module,
    path: &[u32],
    root: u64,
    rank: usize,
    out: &Path,
    number: usize,
) -> Result<Value> {
    let pointer = if path.is_empty() {
        root
    } else {
        let mut tokens = [0u32; 6];
        tokens[..path.len()].copy_from_slice(path);
        w.verify.begin_verification(&tokens, &w.comm, &w.stream)?;
        let (logits, rows) = w.verify.pending_logits()?;
        ensure!(rows == 6, "law oracle width");
        logits.ptr() + ((path.len() - 1) * 129280 * 4) as u64
    };
    let ids = top4(w, module, pointer)?;
    let file = format!("node-{number}.f32");
    let hash = if rank == 0 {
        let bytes = unsafe { w.cuda.read_device(pointer, 129280 * 4) }?;
        let h = format!("{:x}", Sha256::digest(&bytes));
        std::fs::write(out.join(&file), bytes)?;
        Some(h)
    } else {
        None
    };
    if !path.is_empty() {
        w.verify.finish_verification(0, &w.comm, &w.stream)?;
    }
    Ok(json!({"path":path,"top_ids":ids,"file":file,"sha256":hash}))
}

pub(super) fn run(w: &mut Worker, rank: usize, out: &Path) -> Result<()> {
    ensure!(
        enabled()
            && w.current.is_none()
            && w.prompt_cache_logits
            && w.retain_restored_checkpoint
            && w.session_cache_budget >= 512 * 1024 * 1024,
        "sampler law requires an isolated exact-cache worker"
    );
    let samples = std::env::var("CANADA_SCRATCH_SAMPLER_LAW_SAMPLES")
        .unwrap_or_else(|_| "256".into())
        .parse::<usize>()?;
    ensure!([256, 2048].contains(&samples), "finite law sample count");
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/chunk-plan-quality-v1.json"
    ))?;
    let history = std::env::var("CANADA_SCRATCH_SAMPLER_LAW_HISTORY").as_deref() == Ok("1");
    let cases: Vec<(usize, Value)> = if history {
        let saved: Value = serde_json::from_slice(include_bytes!(
            "../tests/fixtures/long-history-score-v1.json"
        ))?;
        // The longer fixture extends the shorter one. Visit it first so each
        // independent case is a fresh prompt, not an active append (whose
        // ordinary policy deliberately skips an exact prompt-logit snapshot).
        [8usize, 0]
            .into_iter()
            .map(|case| {
                let item = &saved["cases"][case];
                (
                    case,
                    json!({"id":item["name"], "input_ids":item["prompt_ids"],
                "prefix":item["prompt_ids"].as_array().unwrap().len()}),
                )
            })
            .collect()
    } else {
        [0usize, 2, 3]
            .into_iter()
            .map(|case| (case, fixture["cases"][case].clone()))
            .collect()
    };
    let module = w.cuda.module(&w.core)?;
    let mut request_id = 100_000u64;
    let mut reports = Vec::new();
    for (case, item) in cases {
        let all: Vec<u32> = serde_json::from_value(item["input_ids"].clone())?;
        let prompt = all[..item["prefix"].as_u64().unwrap() as usize].to_vec();
        let case_out = out.join(format!("law-case{case}"));
        std::fs::create_dir(&case_out)?;
        w.cached_sessions.clear();
        request_id += 1;
        let begin = w.execute(Command::Begin {
            id: request_id,
            seed: 202610300001,
            greedy: false,
            limit: 32,
            tokens: prompt.clone(),
            images: Vec::new(),
        })?;
        ensure!(
            begin["prompt_logit_cache"]["saved"] == true,
            "law initial exact image was not saved: {}",
            begin["prompt_logit_cache"]
        );
        let root = w
            .cached_sessions
            .iter()
            .find(|c| c.prefix == prompt)
            .and_then(|c| c.prompt_logits.as_ref())
            .context("law root logits")?
            .ptr();
        let before = w.verify.carry_digests()?;
        let mut nodes = vec![oracle_row(w, &module, &[], root, rank, &case_out, 0)?];
        let first: Vec<u32> = serde_json::from_value(nodes[0]["top_ids"].clone())?;
        for a in first {
            let node = oracle_row(w, &module, &[a], root, rank, &case_out, nodes.len())?;
            let second: Vec<u32> = serde_json::from_value(node["top_ids"].clone())?;
            nodes.push(node);
            for b in second {
                nodes.push(oracle_row(
                    w,
                    &module,
                    &[a, b],
                    root,
                    rank,
                    &case_out,
                    nodes.len(),
                )?);
            }
        }
        ensure!(
            w.verify.carry_digests()? == before,
            "law oracle rollback changed carry state"
        );
        w.execute(Command::Finish(Finish::Cancelled))?;
        let mut draws = Vec::with_capacity(samples);
        let began = Instant::now();
        for sample in 0..samples {
            w.cached_sessions
                .retain(|c| c.prefix == prompt && c.prompt_logits.is_some());
            ensure!(w.cached_sessions.len() == 1, "law exact image was lost");
            request_id += 1;
            let seed = 0x4c41570000000000u64 + (case as u64) * 1_000_000 + sample as u64;
            let event = w.execute(Command::Begin {
                id: request_id,
                seed,
                greedy: false,
                limit: 32,
                tokens: prompt.clone(),
                images: Vec::new(),
            })?;
            ensure!(
                event["cached_tokens"].as_u64() == Some(prompt.len() as u64),
                "law did not restore the exact prompt"
            );
            let mut ids: Vec<u32> = serde_json::from_value(event["output_ids"].clone())?;
            let mut steps = Vec::new();
            while ids.len() < 3 && !ids.contains(&w.cfg.eos_token_id) {
                let step = w.execute(Command::Step)?;
                ids.extend(serde_json::from_value::<Vec<u32>>(
                    step["output_ids"].clone(),
                )?);
                steps.push(json!({"source":step["proposal_source"],"rows":step["verified_rows"],
                    "accepted":step["accepted_draft_tokens"],"committed":step["committed_model_rows"]}));
                ensure!(steps.len() <= 2, "three-token law sample made no progress");
            }
            ids.truncate(3);
            w.execute(Command::Finish(Finish::Cancelled))?;
            draws.push(json!({"seed":seed,"ids":ids,"steps":steps}));
            if sample % 64 == 63 {
                eprintln!("sampler-law rank{rank} case{case} samples{}", sample + 1);
            }
        }
        let report = json!({"case":case,"id":item["id"],"prefix_tokens":prompt.len(),"samples":samples,
            "seconds":began.elapsed().as_secs_f64(),"nodes":nodes,"draws":draws,
            "initial":begin,"oracle_carry_rollback_exact":true});
        std::fs::write(
            case_out.join("index.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        reports.push(json!({"case":case,"id":item["id"],"samples":samples,"directory":format!("law-case{case}")}));
    }
    std::fs::write(
        out.join("sampler-law-summary.json"),
        serde_json::to_vec_pretty(&json!({
        "rank":rank,"cases":reports,"temperature":1.,"top_p":1.,"top_k":-1,
        "scope":"Finite three-token sampling through ordinary worker Begin/Step/Finish, actual drafter and history proposals. Short exact-prompt cache admission enabled only for this diagnostic; all budget/reserve guards remain. Teacher-forced target tree supplies reference first/second/third-token probabilities. No HTTP parsing or completed-task quality claim."}))?,
    )?;
    Ok(())
}
