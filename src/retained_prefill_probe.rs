//! Paired retained EXTEND diagnostic. Same real accepted prefix, new encoder
//! rows and qualified decoder floor; no draft or CPU model approximation.
use crate::{
    cuda::{Cuda, Stream},
    dspark_generation::{compare_logits, compare_positions, teacher_forced_logits},
    roce::Roce,
    target::Target,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    cuda: &Cuda,
    large: &mut Target,
    verify: &mut Target,
    wides: &mut [Target],
    comm: &Roce,
    stream: &Stream,
    prefix: &[u32],
    rank: usize,
) -> Result<Value> {
    let corpus: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/target-agentic-repository-v1.json"
    ))?;
    let tokens: Vec<u32> = serde_json::from_value(corpus["batches"][0].clone())?;
    let continuation: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/fixed-verify-python-v1.json"
    ))?;
    let continuation: Vec<u32> =
        serde_json::from_value(continuation["teacher_forced_tokens"].clone())?;
    ensure!(
        prefix.len() + 4096 + 48 <= large.context_capacity() && tokens.len() >= 4096,
        "retained probe context/corpus bounds"
    );
    let mut lineage = 1000;
    let mut records = Vec::new();
    large.enable_cached_extend_floor()?;
    for delta in [1, 6, 32, 127, 128, 256, 1024, 4096] {
        let mut previous: [Option<Vec<Vec<f32>>>; 2] = [None, None];
        for request in 0..4 {
            let mut times = [0.; 2];
            let mut prompt_logits: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
            let mut forced: [Vec<Vec<f32>>; 2] = [Vec::new(), Vec::new()];
            let mut phases = [Value::Null, Value::Null];
            let mut incoming = [Value::Null, Value::Null];
            // Alternate order to avoid consistently favoring one path's cache.
            for path in [request % 2, 1 - request % 2] {
                lineage += 1;
                if !records.is_empty() || request != 0 || path != 0 {
                    verify.reset(lineage, comm, stream)?;
                    large.handoff_from(verify)?;
                }
                large.prefill_wide(wides, prefix, comm, stream)?;
                incoming[path] = large.encoder_tail_digest()?.unwrap();
                let start = large.position();
                let begun = Instant::now();
                if path == 0 {
                    large.prefill_wide(&mut [], &tokens[..delta], comm, stream)?;
                } else {
                    large.prefill_wide(wides, &tokens[..delta], comm, stream)?;
                }
                let _anchor = large.greedy_token(stream)?;
                times[path] = begun.elapsed().as_secs_f64();
                ensure!(
                    large.position() == start + delta as u64,
                    "retained history frontier"
                );
                phases[path] = serde_json::to_value(large.append_metrics()?)?;
                prompt_logits[path] = large.last_prompt_logits()?;
                verify.handoff_from(large)?;
                forced[path] = teacher_forced_logits(cuda, verify, &continuation, 8, comm, stream)?;
            }
            let repeat: Vec<_> = (0..2)
                .map(|p| {
                    previous[p]
                        .as_ref()
                        .map(|old| compare_positions(old, &forced[p]))
                })
                .collect();
            let record = json!({"prefix_tokens":prefix.len(),"new_tokens":delta,"request":request,"warmup":request==0,
                "seconds128":times[0],"seconds_wide":times[1],"phases":phases,
                "incoming_encoder_digest_equal":incoming[0]==incoming[1],
                "last_row_logits":compare_logits(&prompt_logits[0],&prompt_logits[1]),
                "teacher_forced":compare_positions(&forced[0],&forced[1]),"repeat_teacher_forced":repeat});
            previous = forced.map(Some);
            eprintln!(
                "retained rank{rank} prefix{} delta{delta} request{request}:128 {:.3}s wide {:.3}s",
                prefix.len(),
                times[0],
                times[1]
            );
            records.push(record);
        }
    }
    Ok(
        json!({"rank":rank,"records":records,"memory":cuda.admission_memory()?,"full_engine_acceptance":false,
        "scope":"Paired retained encoder prefill128/wide, new increments1/6/32/127/128/256/1024/4096, 1warmup+3measured, alternating order, identical fresh-prefix procedure. Both paths use the same cached-floor implementation for increments below128. Full-vocabulary comparison over48 teacher-forced positions and repeat variability. No HTTP or SGLang claim."}),
    )
}
