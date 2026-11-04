//! Regression for full-logit tail captures after different live row counts.
use super::*;
fn state(t:&Target)->Result<Value> {
    Ok(json!({"position":t.position(),"carry":t.carry_digests()?,"encoder":t.encoder_tail_digest()?}))
}
pub(super) fn check(w:&mut Worker,rank:usize,out:&Path)->Result<Value> {
    ensure!(w.verify.local_target_enabled()&&w.verify.profile_is_current(),
        "gather row screen requires the local-target fallback");
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/fixed-verify-python-v1.json"))?;
    let ids:Vec<u32>=serde_json::from_value(fixture["teacher_forced_tokens"].clone())?;
    let before=state(&w.verify)?;let mut records=Vec::new();
    // First capture one row, matching ordinary startup. Then grow, shrink and
    // revisit counts with changed tokens so stale buffers cannot pass.
    for (round,widths) in [[1usize,3,6,2,5,4],[6,4,2,5,3,1]].into_iter().enumerate() {
        for live in widths {
            let offset=round*32+live*6;
            w.verify.begin_verification(&ids[offset..offset+live],&w.comm,&w.stream)?;
            let proof=w.verify.gathered_rows_proof_for_probe()?;
            ensure!(w.control.all_admit(proof["exact"]==true)?,"gathered vocabulary rows differ from local shards");
            w.verify.finish_verification(0,&w.comm,&w.stream)?;
            ensure!(state(&w.verify)?==before,"gather row screen changed accepted state");
            records.push(json!({"round":round,"live":live,"proof":proof,"rollback_exact":true}));
        }
    }
    let report=json!({"rank":rank,"passed":true,"records":records,
        "scope":"Each rank compares every gathered live row against its own FP32 vocabulary shard, byte for byte. Widths grow/shrink/revisit with changed inputs; speculative state is fully rolled back. CPU only hashes and compares raw bytes."});
    std::fs::write(out.join("gather-rows-summary.json"),serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}
pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none(),"finite gather regression begins idle");
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/target-agentic-python-errors-v2.json"))?;
    let ids:Vec<u32>=serde_json::from_value(fixture["batches"][0].clone())?;
    super::history32_generation_probe::execute(w,Command::Begin{images:Vec::new(),id:8_041_291,
        seed:20261104,greedy:false,limit:1,tokens:ids})?;
    check(w,rank,out)?;
    super::history32_generation_probe::execute(w,Command::Finish(Finish::Length))?;Ok(())
}
