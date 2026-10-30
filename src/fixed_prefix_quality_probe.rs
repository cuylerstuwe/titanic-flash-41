//! Small target-only teacher-forcing capture. All probability math stays on GPU.
//! This does not change the serving path or choose a preferred sampled answer.
use super::*;
use anyhow::Context;

fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn state(target: &Target) -> Result<Value> {
    Ok(json!({"position":target.position(),"carry":target.carry_digests()?,
        "encoder":target.encoder_tail_digest()?}))
}

pub(super) fn run(w: &mut Worker, rank: usize, out: &Path) -> Result<()> {
    ensure!(w.current.is_none() && w.history.frontier()==0 && w.session_cache_budget==0
        && !w.prompt_checkpoint && !w.prompt_cache_logits && w.verify.physical_rows()==6,
        "fixed-prefix quality probe needs a fresh isolated worker");
    for flag in ["CANADA_SCRATCH_GENERATION_FP32_ATTENTION", "CANADA_SCRATCH_LOSSLESS_HEAD",
        "CANADA_SCRATCH_MEDIUM_NATIVE_TRANSPORT", "CANADA_SCRATCH_DENSE_PIPELINE_FENCE"] {
        ensure!(std::env::var(flag).as_deref()==Ok("1"), "fixed-prefix candidate requires {flag}");
    }
    let raw=include_bytes!("../tests/fixtures/fixed-prefix-quality-v1.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let cases=fixture["cases"].as_array().context("fixed-prefix cases")?;
    ensure!(cases.len()==8 && fixture["temperature"]==1 && fixture["top_p"]==1
        && fixture["top_k"]==-1 && fixture["end_thinking_token"]==128822,
        "fixed-prefix fixture contract");
    let mut records=Vec::new();
    let prefill_only=std::env::var("CANADA_SCRATCH_FIXED_PREFIX_PREFILL_SCREEN").as_deref()==Ok("1");
    for (case,c) in cases.iter().enumerate() {
        let tokens:Vec<u32>=serde_json::from_value(c["input_ids"].clone())?;
        ensure!((129..16384).contains(&tokens.len()) && tokens.iter().all(|&x|x<129280),
            "fixed-prefix tokens");
        let prefix=if prefill_only {tokens.len()} else {tokens.len()-6};
        for repeat in 0..2usize {
            // Startup warmup leaves the verifier as the current owner even
            // after clearing history. Adopt that owner just as Begin does.
            if w.verify.profile_is_current() {
                w.verify.reset(9_050_000+(case*2+repeat)as u64,&w.comm,&w.stream)?;
                w.large.handoff_from(&w.verify)?;
            } else {
                ensure!(case==0 && repeat==0 && w.large.profile_is_current()
                    && w.large.position()==0,"fixed-prefix initial owner");
            }
            w.control.agree(&[case as u8,repeat as u8,0])?;
            w.large.prefill_wide(&mut w.wides,&tokens[..prefix],&w.comm,&w.stream)?;
            // Host transfers and hashes are raw evidence only. The comparison
            // tool applies log-softmax and measures distributions on CUDA.
            let (bytes,rows)=if prefill_only {
                let before=state(&w.large)?;
                let logits=w.large.snapshot_prompt_logits(&w.stream)?;
                let bytes=unsafe{w.cuda.read_device(logits.ptr(),129280*4)}?;
                ensure!(before==state(&w.large)?,"prompt logit snapshot changed retained state");
                w.verify.handoff_from(&w.large)?;
                (bytes,1)
            } else {
                w.verify.handoff_from(&w.large)?;
                let before=state(&w.verify)?;
                w.verify.begin_verification(&tokens[prefix..],&w.comm,&w.stream)?;
                let (logits,rows)=w.verify.pending_logits()?;
                ensure!(rows==6,"fixed-prefix verification width");
                let bytes=unsafe{w.cuda.read_device(logits.ptr(),rows*129280*4)}?;
                w.verify.finish_verification(0,&w.comm,&w.stream)?;
                ensure!(before==state(&w.verify)?,"teacher-forcing abort changed retained state");
                (bytes,rows)
            };
            let filename=format!("fixed-prefix-c{case}-r{repeat}.f32");
            if rank==0 {std::fs::write(out.join(&filename),&bytes)?;}
            records.push(json!({"case":case,"id":c["id"],"repeat":repeat,
                "prefill_tokens":prefix,"input_tokens":tokens.len(),"rows":rows,
                "file":filename,"sha256":digest(&bytes),"last_row_sha256":digest(&bytes[(rows-1)*129280*4..]),
                "retained_state_unchanged":true,
                "state_check":if prefill_only {"read_only_prompt_logit_snapshot"}else{"zero_accept_verification_rollback"}}));
            std::fs::write(out.join("fixed-prefix-progress.json"),serde_json::to_vec(&records)?)?;
            eprintln!("fixed-prefix rank{rank} case{case} repeat{repeat} captured");
            w.control.agree(&[case as u8,repeat as u8,1])?;
        }
    }
    std::fs::write(out.join("fixed-prefix-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"fixture_sha256":digest(raw),"records":records,"temperature":1,"top_p":1,
        "top_k":-1,"generation_attention":"fp32","target_head":"canonical_bf16_weights_fp32_output",
        "target_path":if prefill_only {"fresh_prefill_final_logit"}else{"fresh_prefill_then_six_teacher_forced_rows"},
        "scope":"Eight fixed prefixes, two fresh replays each. Explicit target_path identifies full prefill or final six teacher-forced verification rows. GPU distribution diagnosis; no completed-task, speed or general quality claim."}))?)?;
    Ok(())
}
