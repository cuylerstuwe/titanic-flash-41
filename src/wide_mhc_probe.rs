//! Short same-allocation M16 input-fusion and coefficient-scheduling comparison.
use super::*;
use super::history32_generation_probe::execute;

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none()&&w.long_verify.is_some()
        &&std::env::var("CANADA_SCRATCH_WIDE_MHC_SCREEN").as_deref()==Ok("1"),
        "isolated wider mHC screen");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let ids:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let mut records=Vec::new();
    for (case,prefix) in [65541usize,131109].into_iter().enumerate() {
        execute(w,Command::Begin{images:Vec::new(),id:8_020_100+case as u64,
            seed:20261102,greedy:false,limit:1,tokens:ids[..prefix].to_vec()})?;
        let original=(w.verify.carry_digests()?,w.verify.encoder_tail_digest()?);
        let target=w.long_verify.as_mut().unwrap();target.handoff_from(&w.verify)?;
        let mut expected=std::collections::BTreeMap::<(bool,usize,usize),Vec<u8>>::new();
        for (repeat,order) in [[0u8,1,3,2],[2,3,1,0]].into_iter().enumerate() {
            for mode in order {
                target.set_wide_mhc_for_probe(mode,&w.stream)?;
                for live in [9usize,16] {
                    let proposals=&ids[prefix..prefix+live];
                    for _ in 0..3 {
                        target.begin_verification(proposals,&w.comm,&w.stream)?;
                        target.finish_verification(0,&w.comm,&w.stream)?;
                    }
                    let captures=target.capture_statistics().1;
                    let mut samples=Vec::new();
                    for trial in 0..6u8 {
                        w.control.agree(&[0xae,case as u8,repeat as u8,mode,live as u8,trial])?;
                        let at=Instant::now();target.begin_verification(proposals,&w.comm,&w.stream)?;
                        w.stream.synchronize()?;samples.push(at.elapsed().as_secs_f64()*1000.);
                        target.finish_verification(0,&w.comm,&w.stream)?;
                    }
                    ensure!(captures==target.capture_statistics().1,"timed wider mHC graph capture");
                    let mut hashes=Vec::new();
                    for (changed,input) in [proposals,&ids[prefix+32..prefix+32+live]].into_iter().enumerate() {
                        target.begin_verification(input,&w.comm,&w.stream)?;
                        let (y,n)=target.pending_local_logits_for_probe()?;
                        ensure!(n==live,"wider mHC vocabulary extent");
                        let logits=unsafe{w.cuda.read_device(y.ptr(),live*32320*4)}?;
                        hashes.push(format!("{:x}",Sha256::digest(&logits)));
                        let key=(mode&2!=0,live,changed);
                        let reference=expected.entry(key).or_insert_with(||logits.clone());
                        let exact=reference==&logits;
                        if repeat==0 && (mode==0||mode==2) {
                            std::fs::write(out.join(format!("wide-mhc-c{case}-m{mode}-n{live}-x{changed}.bin")),&logits)?;
                        }
                        target.finish_verification(0,&w.comm,&w.stream)?;
                        let state=(target.carry_digests()?,target.encoder_tail_digest()?)==original;
                        ensure!(w.control.all_admit(exact&&state)?,"wider mHC same-path repeat/PDL or rollback changed");
                    }
                    ensure!(hashes[0]!=hashes[1],"changed wider mHC inputs did not change logits");
                    records.push(json!({"case":case,"prefix":prefix,"repeat":repeat,"mode":mode,"live":live,
                        "samples_ms":samples,"logit_hashes":hashes,"pdl_and_repeat_exact":true,"rollback_exact":true}));
                    std::fs::write(out.join("wide-mhc-partial.json"),serde_json::to_vec_pretty(&records)?)?;
                }
            }
        }
        target.set_wide_mhc_for_probe(0,&w.stream)?;
        w.verify.handoff_from(target)?;execute(w,Command::Finish(Finish::Length))?;
    }
    std::fs::write(out.join("wide-mhc-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"fixture_sha256":format!("{:x}",Sha256::digest(raw)),
        "pdl_and_repeat_exact":true,"rollback_exact":true,
        "scope":"Two public 65K/131K prefixes, M16 live9/16, four forward/reverse variants. Three warm plus six measured steps. Changed proposals, all local vocabulary shards, exact PDL/same-path repeat and rollback. Fusion changes FP32 reduction order while retaining BF16 boundaries; saved logits need GPU distribution analysis. No outcome or serving-default claim."
    }))?)?;
    if std::env::var("CANADA_SCRATCH_CHUNK_ECONOMICS_SCREEN").as_deref()==Ok("1") {
        super::chunk_economics_probe::run(w,rank,out)
    }else{execute(w,Command::Shutdown)?;Ok(())}
}
