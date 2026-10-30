//! Teacher-forced, same-prefix GPU differential for the real-prompt failure.
//! This is diagnostic evidence, never a speed or engine-acceptance result.
use crate::{
    cuda::{Cuda, Stream},
    roce::Roce,
    target::Target,
};
use anyhow::{Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::Path;
pub(crate) fn run(
    cuda: &Cuda,
    large: &mut Target,
    verify: &mut Target,
    comm: &Roce,
    stream: &Stream,
    prefix: &[u32],
    output: &Path,
    rank: usize,
) -> Result<serde_json::Value> {
    ensure!(prefix.len() == 458, "exact real-prompt diagnostic prefix");
    large.prefill(prefix, comm, stream)?;
    let first = large.greedy_token(stream)?;
    verify.handoff_from(large)?;
    let mut tokens = vec![first];
    let mut expected = None;
    while tokens.len() < 120 {
        verify.append(&[*tokens.last().unwrap()], comm, stream)?;
        let next = verify.greedy_token(stream)?;
        if tokens.len() == 106 {
            let raw = unsafe { cuda.read_device(verify.logits()?.ptr(), 129280 * 4) }?;
            if rank == 0 {
                std::fs::write(output.join("control-logits.bin"), &raw)?;
                verify.dump_layer_trace(&output.join("control106"))?;
            }
            expected = Some(raw);
        }
        tokens.push(next);
    }
    std::fs::write(
        output.join("control-tokens.json"),
        serde_json::to_vec_pretty(&tokens)?,
    )?;
    let expected = expected.unwrap();
    let mut records = Vec::new();
    for row in 0..6 {
        verify.reset(2000 + row as u64, comm, stream)?;
        large.handoff_from(verify)?;
        large.prefill(prefix, comm, stream)?;
        ensure!(
            large.greedy_token(stream)? == first,
            "fresh prefill anchor drift"
        );
        verify.handoff_from(large)?;
        let start = 105 - row;
        for &input in &tokens[..start] {
            verify.append(&[input], comm, stream)?;
        }
        ensure!(
            verify.position() == 458 + start as u64,
            "teacher-forced block frontier"
        );
        verify.begin_verification(&tokens[start..start + 6], comm, stream)?;
        let predictions = verify.verification_greedy(stream)?;
        let logits = verify.pending_logits()?.0;
        let actual =
            unsafe { cuda.read_device(logits.ptr() + (row * 129280 * 4) as u64, 129280 * 4) }?;
        verify.finish_verification(row + 1, comm, stream)?;
        ensure!(verify.position() == 564, "teacher-forced accepted frontier");
        if rank == 0 {
            std::fs::write(output.join(format!("batch-row{row}-logits.bin")), &actual)?;
            verify.dump_layer_trace(&output.join(format!("batch-row{row}")))?;
        }
        eprintln!(
            "teacherforced row{row} frontier{} token{} expected{} logits_exact{}",
            verify.position(),
            predictions[row],
            tokens[106],
            actual == expected
        );
        records.push(json!({"row":row,"start_position":458+start,"position":verify.position(),"expected_token":tokens[106],"actual_token":predictions[row],"greedy_equal":predictions[row]==tokens[106],"logits_exact":actual==expected,"logits_sha256":format!("{:x}",Sha256::digest(&actual))}));
    }
    Ok(
        json!({"rank":rank,"prompt_tokens":458,"diagnostic_output_index":106,"accepted_frontier":564,"control_tokens":tokens,"control_logits_sha256":format!("{:x}",Sha256::digest(&expected)),"records":records,"all_rows_exact":records.iter().all(|r|r["logits_exact"]==true),"full_engine_acceptance":false,"performance_result":false,"scope":"Same actual accepted prefix at position564: compare target-only row0 against each offset in teacher-forced Verify6. Full40-layer rank0 traces retained; no drafter, no performance claim, no CPU numeric oracle."}),
    )
}
