//! Short paired screen of independently captured decoder tails after eager wide
//! encoder prefill. Accepted history, precision and numerical operators match.
use crate::{
    cuda::{Cuda, Stream},
    dspark_generation::{compare_logits, compare_positions, teacher_forced_logits},
    roce::Roce,
    target::Target,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    cuda: &Cuda,
    large: &mut Target,
    verify: &mut Target,
    wides: &mut [Target],
    comm: &Roce,
    stream: &Stream,
    prefix: &[u32],
    output: &Path,
    rank: usize,
) -> Result<Value> {
    ensure!(
        large.context_capacity() == 32768
            && prefix.len() == 458
            && std::env::var_os("CANADA_SCRATCH_PREFILL_TAIL_GRAPH").is_none(),
        "tail graph screen requires explicit per-call policy and32K profile"
    );
    let corpus: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/target-agentic-repository-v1.json"
    ))?;
    let tokens: Vec<u32> = serde_json::from_value(corpus["batches"][0].clone())?;
    let long: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/target-agentic-long15800-v1.json"
    ))?;
    let long: Vec<u32> = serde_json::from_value(long["batches"][0].clone())?;
    let continuation: Value = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/fixed-verify-python-v1.json"
    ))?;
    let continuation: Vec<u32> =
        serde_json::from_value(continuation["teacher_forced_tokens"].clone())?;
    ensure!(
        tokens.len() >= 4096 && long.len() == 15800 && continuation.len() >= 12,
        "tail graph fixture bounds"
    );
    large.enable_cached_extend_floor()?;
    let mut records = Vec::new();
    let mut lineage = 80000;
    let mut initial = true;
    let encoder_graph =
        std::env::var("CANADA_SCRATCH_PREFILL_ENCODER_GRAPH_AB").as_deref() == Ok("1");
    let mut cases: Vec<(&str, &[u32], usize)> = vec![
        ("fresh", &[], 129),
        ("fresh", &[], 4096),
        ("retained", prefix, 128),
        ("retained", prefix, 256),
        ("retained", prefix, 4096),
        ("retained_long", &long, 256),
    ];
    if encoder_graph {
        ensure!(tokens.len() >= 8192, "multi-chunk encoder graph corpus");
        cases.extend([("fresh", &[][..], 1024), ("retained_long", &long[..], 8192)]);
    }
    let case_count = cases.len();
    for (history, base, delta) in cases {
        for request in 0..6 {
            let mut seconds = [0.; 2];
            let mut logits: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
            let mut forced: [Vec<Vec<f32>>; 2] = [Vec::new(), Vec::new()];
            let mut incoming = [Value::Null, Value::Null];
            let mut outgoing = [Value::Null, Value::Null];
            let mut phases = [Value::Null, Value::Null];
            for path in [request % 2, 1 - request % 2] {
                if !initial {
                    lineage += 1;
                    verify.reset(lineage, comm, stream)?;
                    large.handoff_from(verify)?;
                }
                initial = false;
                if !base.is_empty() {
                    large.prefill_wide_with_policy(wides, base, comm, stream, false, false)?;
                }
                incoming[path] = json!({"position":large.position(),
                    "encoder":large.encoder_tail_digest()?,"carry":large.carry_digests()?});
                let start = Instant::now();
                large.prefill_wide_with_graph_policy(
                    wides,
                    &tokens[..delta],
                    comm,
                    stream,
                    false,
                    path == 1,
                    path == 1 && encoder_graph,
                )?;
                let anchor = large.greedy_token(stream)?;
                seconds[path] = start.elapsed().as_secs_f64();
                ensure!(
                    large.position() as usize == base.len() + delta,
                    "tail graph changed history frontier"
                );
                phases[path] = serde_json::to_value(large.append_metrics()?)?;
                ensure!(
                    request < 2 || phases[path]["captured_new_graph"] == false,
                    "tail graph unexpectedly recaptured in measured trial"
                );
                if path == 0 || !encoder_graph {
                    ensure!(
                        phases[path]["encoder_graph_chunks"] == 0,
                        "control replayed encoder graph"
                    );
                } else if request >= 2 && [128, 1024, 4096, 8192].contains(&delta) {
                    ensure!(
                        phases[path]["encoder_graph_chunks"].as_u64().unwrap_or(0) > 0,
                        "measured encoder candidate did not replay a graph"
                    );
                }
                logits[path] = large.last_prompt_logits()?;
                outgoing[path] = json!({"position":large.position(),"anchor":anchor,
                    "encoder":large.encoder_tail_digest()?,"carry":large.carry_digests()?});
                verify.handoff_from(large)?;
                // Two Verify6 blocks exercise subsequent reads of the replayed
                // tail, compression carry and changed token/position metadata.
                forced[path] =
                    teacher_forced_logits(cuda, verify, &continuation[..12], 2, comm, stream)?;
                ensure!(
                    logits[path]
                        .iter()
                        .chain(forced[path].iter().flatten())
                        .all(|x| x.is_finite()),
                    "tail graph nonfinite output"
                );
            }
            let record = json!({"history_mode":history,"prefix_tokens":base.len(),"new_tokens":delta,
                "request":request,"warmup":request<2,"seconds_control":seconds[0],"seconds_graph":seconds[1],
                "phases":phases,"incoming_state_equal":incoming[0]==incoming[1],
                "outgoing_state_equal":outgoing[0]==outgoing[1],
                "last_row_logits":compare_logits(&logits[0],&logits[1]),
                "teacher_forced":compare_positions(&forced[0],&forced[1]),
                "teacher_forced_bitwise_equal":forced[0].iter().zip(&forced[1]).all(|(a,b)|
                    a.iter().zip(b).all(|(x,y)|x.to_bits()==y.to_bits()))});
            std::fs::write(
                output.join(format!(
                    "prefill-tail-graph-{history}-d{delta}-r{request}.json"
                )),
                serde_json::to_vec_pretty(&record)?,
            )?;
            eprintln!(
                "tail graph rank{rank} {history} delta{delta} request{request}: eager {:.4}s graph {:.4}s",
                seconds[0], seconds[1]
            );
            records.push(record);
        }
    }
    let all_exact = records.iter().all(|r| {
        r["incoming_state_equal"] == true
            && r["outgoing_state_equal"] == true
            && r["last_row_logits"]["bitwise_equal"] == true
            && r["teacher_forced_bitwise_equal"] == true
    });
    Ok(
        json!({"rank":rank,"case_count":case_count,"encoder_graph":encoder_graph,"all_exact":all_exact,"records":records,"memory":cuda.admission_memory()?,
        "full_engine_acceptance":false,"scope":"One32K model load; fresh/retained cases, 2warm+4measured alternating pairs, independently captured128-row decoder, optionally full-width encoder graphs (partial chunks eager). Full prompt-row logits and12 subsequent teacher-forced distributions; encoder/carry digests and frontier checks. Initial component/integrated prefill screen, no HTTP, outcome or p95/p99 qualification."}),
    )
}
