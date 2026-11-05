//! Same-load snapshot-copy screen; no HTTP or cache policy changes.
use super::*;
use super::cache_slots_probe::{execute,following};

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.suspended_image_count==0
        &&std::env::var("CANADA_SCRATCH_COMPACT_STATE_IMAGE").as_deref()==Ok("1"),
        "state copy screen needs a committed ordinary owner");
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-verification-corpus-v1.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let follow=&tokens[131200..131206];
    let mut records=Vec::new();let mut id=9_101_000;
    // No optional cache policy change: direct snapshots isolate copy costs.
    w.session_cache_budget=0;w.prompt_checkpoint=false;w.encoder_prompt_checkpoint=false;
    w.prompt_cache_logits=false;w.cached_sessions.clear();
    for frontier in [16385usize,131105] {
        id+=1;
        execute(w,Command::Begin{images:Vec::new(),id,seed:20261103,greedy:false,limit:1,
            tokens:tokens[..frontier].to_vec()})?;
        execute(w,Command::Finish(Finish::Length))?;
        let anchor=w.history.pending().context("state copy pending anchor")?;
        let expected=following(w,follow)?.0;
        let (budget,_)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
        let mut reference=None;
        for (arm,fused) in [false,true,true,false].into_iter().enumerate() {
            w.control.agree(&[arm as u8,fused as u8])?;
            crate::state_image::copy::set_for_probe(fused)?;
            let began=Instant::now();
            let saved=w.try_save_session_image(anchor,budget)?.context("snapshot capacity refused")?;
            let save_seconds=began.elapsed().as_secs_f64();
            let manifest=saved.manifest();
            ensure!(manifest["target"]["fused_copy"]==fused,
                "copy screen did not exercise its selected target path: {manifest}");
            let digest=saved.digests()?;
            let reference=reference.get_or_insert_with(||digest.clone());
            ensure!(digest==*reference,"state copy changed captured bytes or sampler state");
            // Advance every state owner before restoration. Worker history
            // remains the saved frontier; no altered owner escapes the probe.
            w.session.as_mut().unwrap().step(&mut w.verify,anchor,6,&w.comm,&w.stream)?;
            w.target_lineage+=1;
            let began=Instant::now();
            let restored=w.session.as_mut().unwrap().restore_state_image(&mut w.verify,&saved,
                w.target_lineage,&w.comm,&w.stream)?;
            let restore_seconds=began.elapsed().as_secs_f64();
            ensure!(restored==anchor&&following(w,follow)?.0==expected,
                "restored snapshot changed state or following logits");
            crate::state_image::copy::set_for_probe(false)?;
            let roundtrip=w.try_save_session_image(anchor,budget)?.context("roundtrip capacity refused")?;
            ensure!(roundtrip.digests()?==*reference,"restored live bytes differ from loop control");
            drop(roundtrip);let began=Instant::now();drop(saved);
            let release_seconds=began.elapsed().as_secs_f64();
            records.push(json!({"frontier":frontier,"arm":arm,"fused":fused,
                "save_seconds":save_seconds,"restore_seconds":restore_seconds,"release_seconds":release_seconds,
                "manifest":manifest,"saved_and_restored_bytes_exact":true,"following_logits_and_state_exact":true,
                "memory":w.cuda.admission_memory()?}));
            std::fs::write(out.join("state-copy-partial.json"),serde_json::to_vec_pretty(&records)?)?;
            eprintln!("state copy rank{rank} frontier{frontier} arm{arm} fused{fused}: save{:.3}ms restore{:.3}ms",
                save_seconds*1000.,restore_seconds*1000.);
        }
    }
    crate::state_image::copy::set_for_probe(false)?;
    execute(w,Command::Shutdown)?;
    std::fs::write(out.join("state-copy-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,
        "scope":"Same-load ABBA raw-byte copy screen at16K and128K with off-boundary frontiers. Advance target/draft/history, then restore. Compare complete saved and restored bytes/sampler state, carry/encoder/draft and following six logit rows. Snapshot save includes ordinary allocation and all-rank admission. No changed precision, cache budget, HTTP or end-to-end serving claim."}))?)?;
    Ok(())
}
