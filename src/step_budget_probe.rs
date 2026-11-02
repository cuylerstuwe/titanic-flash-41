//! Finite whole-step profile; CUDA traces remain private and are not speed claims.
use super::*;

pub(super) fn run(w: &mut Worker, rank: usize, out: &Path) -> Result<()> {
    ensure!(
        w.current.is_none() && w.history.frontier() == 0,
        "step profile starts idle"
    );
    let corpus: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/long-context-continuation-v2.json"
    ))?;
    let tokens: Vec<u32> = serde_json::from_value(corpus["tokens"].clone())?;
    let context = 131072;
    ensure!(tokens.len() >= context + 16, "step profile corpus extent");
    let begin = w.execute(Command::Begin {
        id: 810_001,
        seed: 2026103008,
        greedy: false,
        limit: 256,
        tokens: tokens[..context].to_vec(),
        images: Vec::new(),
    })?;
    let mut warmup = Vec::new();
    for _ in 0..4 {
        ensure!(
            w.history.pending() != Some(w.cfg.eos_token_id),
            "profile warmup ended early"
        );
        warmup.push(w.execute(Command::Step)?);
    }
    w.stream.synchronize()?;
    w.control.agree(b"round25-step-profile-ready")?;
    w.cuda.profiler_start()?;
    w.control.agree(b"round25-step-profile-active")?;
    let at = Instant::now();
    let mut steps = Vec::new();
    for _ in 0..8 {
        if w.history.pending() == Some(w.cfg.eos_token_id) || w.emitted >= w.limit {
            break;
        }
        steps.push(w.execute(Command::Step)?);
    }
    w.stream.synchronize()?;
    let seconds = at.elapsed().as_secs_f64();
    w.cuda.profiler_stop()?;
    ensure!(!steps.is_empty(), "empty step capture");
    w.control.agree(b"round25-step-profile-finished")?;
    let frontier = w.history.frontier();
    let pending = w.history.pending();
    ensure!(
        frontier == context + w.emitted - 1 && pending.is_some(),
        "profile accepted frontier"
    );
    let finish = w.execute(Command::Finish(Finish::Cancelled))?;
    std::fs::write(
        out.join("step-budget.json"),
        serde_json::to_vec_pretty(&json!({
            "rank": rank, "context": context, "begin": begin, "warmup": warmup,
            "steps": steps, "profile_seconds": seconds, "frontier": frontier,
            "pending": pending, "finish": finish, "memory": w.cuda.admission_memory()?,
            "scope": "Eight ordinary worker steps after four warm steps on a public 131K prefix. Actual adaptive drafting, verification, sampling and commit included. All-rank host rendezvous surrounds profiler activation. Instrumented cost attribution only, not uninstrumented throughput or quality evidence."
        }))?,
    )?;
    Ok(())
}
