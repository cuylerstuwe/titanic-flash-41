//! Finite same-allocation clock-cap comparison. The host owns clock changes
//! and independent reset timers; model arithmetic and accepted state are fixed.
use super::*;

fn state(t:&Target)->Result<Value> {
    Ok(json!({"position":t.position(),"carry":t.carry_digests()?,
        "encoder":t.encoder_tail_digest()?}))
}
fn digest(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    ensure!(w.current.is_none() && w.history.frontier()==0 && w.long_verify.is_some(),
        "clock screen requires isolated M6/M16 targets");
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let fixture:Value=serde_json::from_slice(raw)?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let prefix=131073;
    ensure!(tokens.len()>=prefix+16,"clock screen fixture extent");
    w.large.prefill_wide(&mut w.wides,&tokens[..prefix],&w.comm,&w.stream)?;
    w.verify.handoff_from(&w.large)?;
    let mut expected:[Option<(String,Value)>;2]=[None,None];
    let mut records=Vec::new();
    for (phase,cap) in [0u32,2350,2350,0].into_iter().enumerate() {
        w.stream.synchronize()?;
        w.control.agree(&[0xc1,phase as u8])?;
        let requested=json!({"phase":phase,"maximum_clock_MHz":cap,"rank":rank});
        let ready=out.join(format!("clock-ready-{phase}.json"));
        let temporary=ready.with_extension("tmp");
        std::fs::write(&temporary,serde_json::to_vec(&requested)?)?;
        std::fs::rename(temporary,ready)?;
        let release=out.join(format!("clock-go-{phase}.json"));let at=Instant::now();
        while !release.exists() {
            ensure!(at.elapsed()<Duration::from_secs(90),"clock screen host release deadline");
            std::thread::sleep(Duration::from_millis(10));
        }
        let admitted:Value=serde_json::from_slice(&std::fs::read(release)?)?;
        ensure!(admitted==requested,"clock screen release identity differs");
        w.control.agree(&[0xc2,phase as u8])?;
        for (slot,rows) in [6usize,16].into_iter().enumerate() {
            if rows==16 {w.long_verify.as_mut().unwrap().handoff_from(&w.verify)?;}
            let target=if rows==16 {w.long_verify.as_mut().unwrap()}else{&mut w.verify};
            let before=state(target)?;
            let mut times=Vec::new();let mut logits_hash=String::new();
            // Two warmups exclude graph creation/clock settling; eight timed
            // replays are enough for this small screening decision.
            for trial in 0..10 {
                w.control.agree(&[0xc3,phase as u8,rows as u8,trial])?;
                let began=Instant::now();
                target.begin_verification(&tokens[prefix..prefix+rows],&w.comm,&w.stream)?;
                let seconds=began.elapsed().as_secs_f64();
                if trial==2 || trial==9 {
                    let(y,live)=target.pending_logits()?;ensure!(live==rows,"clock screen row extent");
                    let hash=digest(&unsafe {w.cuda.read_device(y.ptr(),rows*129280*4)}?);
                    if logits_hash.is_empty(){logits_hash=hash;}else{ensure!(hash==logits_hash,"clock screen repeat logits changed");}
                }
                target.finish_verification(0,&w.comm,&w.stream)?;
                if trial>=2 {times.push(seconds);}
            }
            ensure!(state(target)?==before,"clock screen abort changed accepted state");
            if let Some((hash,saved))=&expected[slot] {
                ensure!(hash==&logits_hash && saved==&before,"clock cap changed logits or accepted state");
            }else{expected[slot]=Some((logits_hash.clone(),before.clone()));}
            records.push(json!({"phase":phase,"cap_MHz":cap,"rows":rows,
                "seconds":times,"logits_sha256":logits_hash,"state":before,"all_exact":true}));
            if rows==16 {w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;}
        }
        std::fs::write(out.join("clock-verify-progress.json"),serde_json::to_vec_pretty(&records)?)?;
    }
    w.control.agree(&[0xc4])?;
    std::fs::write(out.join("clock-verify-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"records":records,"all_logits_and_state_exact":true,
        "fixture_sha256":digest(raw),"prefix_tokens":prefix,
        "scope":"Fixed model allocations and accepted prefix; stock/2350/2350/stock. M6/M16 each have two warmups and eight timed replays per phase. No sampler, background policy or persistent clock change."
    }))?)?;
    Ok(())
}
