//! Replay the saved, target-only SGLang trajectory with identical input history.
//! Reference logits are diagnostic outputs, never inputs to model execution.
use crate::{cuda::{Cuda, Stream}, dspark_generation::compare_logits, roce::Roce, target::Target};
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;

const VOCAB: usize = 129280;
#[derive(Deserialize)]
struct Row {
    index: usize,
    positions: Vec<usize>,
    input_ids: Vec<u32>,
    rows: usize,
    seq_lens: Vec<usize>,
    file: String,
    bytes: usize,
    sha256: String,
}
#[derive(Deserialize)]
struct Fixture {
    prompt_ids: Vec<u32>,
    output_ids: Vec<u32>,
    rows: Vec<Row>,
    reference_root: String,
    source_index_sha256: String,
}
fn fixture() -> Result<Fixture> {
    let f: Fixture = serde_json::from_slice(include_bytes!("../tests/fixtures/sglang-v31-teacher-forcing.json"))?;
    ensure!(f.prompt_ids.len() == 458 && f.output_ids.len() == 512 && f.rows.len() == 511,
        "oracle trajectory length");
    ensure!(f.prompt_ids.iter().chain(&f.output_ids).all(|&t| (t as usize) < VOCAB), "oracle token domain");
    for (i, r) in f.rows.iter().enumerate() {
        ensure!(r.index == i && r.positions == [458+i] && r.input_ids == [f.output_ids[i]]
            && r.seq_lens == [459+i] && r.rows == 1 && r.bytes == VOCAB*4
            && r.file == format!("{i:04}.bin") && r.sha256.len() == 64,
            "oracle row {i} chronology, shape or source differs");
    }
    Ok(f)
}
fn floats(raw: &[u8]) -> Result<Vec<f32>> {
    ensure!(raw.len() == VOCAB*4, "oracle logit byte shape");
    let values: Vec<_> = raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    ensure!(values.iter().all(|x| x.is_finite()), "non-finite oracle/native logits");
    Ok(values)
}
// Host analysis of captured GPU outputs only; never used by inference/sampling.
fn token_nll(values: &[f32], token: u32) -> f64 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    max + values.iter().map(|&x| (x as f64-max).exp()).sum::<f64>().ln() - values[token as usize] as f64
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda: &Cuda, large: &mut Target, verify: &mut Target, wides: &mut [Target],
    comm: &Roce, stream: &Stream, prefix: &[u32], output: &Path, rank: usize) -> Result<Value> {
    let f = fixture()?;
    ensure!(prefix == f.prompt_ids && large.context_capacity() >= 970, "oracle prefix/context mismatch");
    let reference = Path::new(&f.reference_root);
    let mut records = Vec::new();
    let mut first_hashes = Vec::new();
    let mut first_prefix = String::new();
    for repeat in 0..2 {
        if repeat > 0 { verify.reset(4000+repeat, comm, stream)?; large.handoff_from(verify)?; }
        large.prefill_wide(wides, prefix, comm, stream)?;
        let prompt = large.last_prompt_logits()?;
        let prompt_raw: Vec<_> = prompt.iter().flat_map(|x| x.to_le_bytes()).collect();
        let prefix_hash = format!("{:x}", Sha256::digest(&prompt_raw));
        if repeat == 0 { first_prefix = prefix_hash.clone(); }
        verify.handoff_from(large)?;
        for (i, row) in f.rows.iter().enumerate() {
            ensure!(verify.position() == row.positions[0] as u64, "oracle frontier mismatch");
            verify.begin_verification(&row.input_ids, comm, stream)?;
            let (logits, live) = verify.pending_logits()?;
            ensure!(live == 1, "oracle verification live rows");
            stream.synchronize()?;
            let raw = unsafe { cuda.read_device(logits.ptr(), VOCAB*4) }?;
            let hash = format!("{:x}", Sha256::digest(&raw));
            if repeat == 0 { first_hashes.push(hash.clone()); }
            let mut record = json!({"repeat":repeat,"position":row.positions[0],"native_sha256":hash,
                "repeat_bitwise_equal":hash==first_hashes[i],"prefill_sha256":prefix_hash,
                "prefill_repeat_bitwise_equal":prefix_hash==first_prefix});
            if rank == 0 {
                let oracle = std::fs::read(reference.join(&row.file))?;
                ensure!(format!("{:x}",Sha256::digest(&oracle)) == row.sha256, "reference hash mismatch row{i}");
                let (native, oracle) = (floats(&raw)?, floats(&oracle)?);
                record["sglang_to_native"] = compare_logits(&oracle, &native);
                record["sglang_next_token_nll"] = json!(token_nll(&oracle, f.output_ids[i+1]));
                record["native_next_token_nll"] = json!(token_nll(&native, f.output_ids[i+1]));
                if repeat == 0 { std::fs::write(output.join(&row.file), &raw)?; }
                else {
                    let first = floats(&std::fs::read(output.join(&row.file))?)?;
                    record["native_repeat"] = compare_logits(&first, &native);
                }
            }
            records.push(record);
            verify.finish_verification(1, comm, stream)?;
            if i % 128 == 0 { eprintln!("oracle rank{rank} repeat{repeat} position{}",row.positions[0]); }
        }
    }
    Ok(json!({"rank":rank,"oracle":"sglang-observer-v31/target-only","reference_index_sha256":f.source_index_sha256,
        "records":records,"memory":cuda.admission_memory()?,"full_engine_acceptance":false,"performance_result":false,
        "scope":"Two native replays of all511 saved SGLang next-token distributions with identical458-token prompt and accepted histories. Native canonical BF16 target-head operands with FP32 output; inherited qualified SGLang target output is BF16. The qualified FP8 adapter applies only to the drafter head, not target distributions. Host comparisons only analyze GPU-produced logits. This single SGLang-generated trajectory does not establish held-out task quality or serving parity."}))
}

#[cfg(test)]
mod tests {
    #[test]
    fn oracle_fixture_is_one_exact_contiguous_history() { super::fixture().unwrap(); }
}
