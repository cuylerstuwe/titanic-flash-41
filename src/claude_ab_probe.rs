//! Same-allocation A/B screen for retained-append prefill mechanisms.
//! Diagnostic only: no serving dispatch or default change. Control (mode 0) and
//! candidate (mode 1) alternate ABBA with full incoming-state restoration. Exactness
//! is required only within a mode; cross-mode numerical differences are recorded,
//! not rejected, because legal reduction-order changes are admissible.
use super::*;
use super::history32_generation_probe::{execute,state,restore};

fn hash(v:&[u8])->String {format!("{:x}",Sha256::digest(v))}

/// Total variation and argmax agreement between two logit rows at temperature 1.
fn compare(a:&[f32],b:&[f32])->(f64,f64,bool) {
    let soft=|x:&[f32]|{let m=x.iter().cloned().fold(f32::NEG_INFINITY,f32::max) as f64;
        let e:Vec<f64>=x.iter().map(|v|((*v as f64)-m).exp()).collect();let s:f64=e.iter().sum();
        e.into_iter().map(|v|v/s).collect::<Vec<_>>()};
    let(p,q)=(soft(a),soft(b));
    let tv=0.5*p.iter().zip(&q).map(|(x,y)|(x-y).abs()).sum::<f64>();
    let max=a.iter().zip(b).map(|(x,y)|((*x-*y) as f64).abs()).fold(0.,f64::max);
    let arg=|x:&[f32]|x.iter().enumerate().max_by(|i,j|i.1.total_cmp(j.1)).unwrap().0;
    (tv,max,arg(a)==arg(b))
}

/// Switch the selected wide profile between control (0) and candidate (1).
fn set_mode(w:&mut Worker,index:usize,name:&str,mode:u8)->Result<()> {
    match name {
        "noop"=>Ok(()),
        other=>w.wides[index].set_claude_experiment_for_probe(other,mode,&w.stream),
    }
}

pub(super) fn run(w:&mut Worker,rank:usize,out:&Path)->Result<()> {
    let name=std::env::var("CANADA_SCRATCH_CLAUDE_AB").context("A/B experiment name")?;
    ensure!(w.current.is_none(),"finite Claude A/B screen");
    let prefix:usize=std::env::var("CANADA_SCRATCH_CLAUDE_AB_PREFIX").ok().map(|v|v.parse()).transpose()?.unwrap_or(131071);
    let cases:Vec<(usize,usize)>=std::env::var("CANADA_SCRATCH_CLAUDE_AB_CASES").unwrap_or("512:385,1024:1023,2048:1535".into())
        .split(',').map(|c|{let(a,b)=c.split_once(':').context("case width:live")?;Ok((a.parse()?,b.parse()?))})
        .collect::<Result<_>>()?;
    let repeats:usize=std::env::var("CANADA_SCRATCH_CLAUDE_AB_REPEATS").ok().map(|v|v.parse()).transpose()?.unwrap_or(3);
    let raw=include_bytes!("../tests/fixtures/long-context-continuation-v2.json");
    let corpus:Value=serde_json::from_slice(raw)?;
    let ids:Vec<u32>=serde_json::from_value(corpus["tokens"].clone())?;
    ensure!(ids.len()>=prefix+cases.iter().map(|c|c.1).max().unwrap_or(0)+8,"A/B fixture too short");
    w.cached_sessions.clear();w.session_cache_budget=0;w.prompt_checkpoint=false;
    w.encoder_prompt_checkpoint=false;w.prompt_cache_logits=false;
    // Cold encoder prefill of the public prefix under this load's construction defaults.
    w.stream.synchronize()?;
    // Optional captured range over the whole cold prefill (diagnostic chunk anatomy).
    let profile_begin=std::env::var("CANADA_SCRATCH_CLAUDE_AB_PROFILE_BEGIN").as_deref()==Ok("1");
    if profile_begin {w.cuda.profiler_start()?;w.control.agree(&[0xb0])?;}
    let cold=Instant::now();
    let begin=execute(w,Command::Begin{images:Vec::new(),id:8_081_208,seed:2026110851,
        greedy:false,limit:1,tokens:ids[..prefix].to_vec()})?;
    let cold_begin_ms=cold.elapsed().as_secs_f64()*1000.;
    if profile_begin {w.stream.synchronize()?;w.cuda.profiler_stop()?;}
    eprintln!("claude ab rank{rank} cold begin {prefix} tokens: {cold_begin_ms:.1} ms");
    let anchor=begin["output_ids"][0].as_u64().context("A/B anchor")?as u32;
    let original=state(w)?;
    let(bytes,cpu)=w.session.as_ref().unwrap().compact_state_image_bytes(&w.verify)?;
    ensure!(bytes<=1024usize.pow(3)&&w.control.all_admit(w.cuda.admission_memory()?.admitted_available
        >=8*1024usize.pow(3)+bytes+cpu)?,"A/B snapshot reserve");
    let saved=w.session.as_mut().unwrap().save_state_image(&w.verify,anchor,bytes,&w.comm,&w.stream)?;
    // Optional teacher-forced continuation scoring after the cold prefix: six ground-truth
    // rows per verification, all committed; per-token NLL/argmax, full logits dumped on rank 0.
    let mut score=Value::Null;
    if let Some(steps)=std::env::var("CANADA_SCRATCH_CLAUDE_AB_SCORE").ok().map(|v|v.parse::<usize>()).transpose()? {
        ensure!(steps>0&&steps<=256&&ids.len()>prefix+6*steps+1,"continuation score extent");
        restore(w,&saved,anchor,&original)?;
        let mut nll=Vec::with_capacity(6*steps);let mut hits=0usize;
        let mut dump=std::io::BufWriter::new(std::fs::File::create(out.join(format!("claude-ab-score-r{rank}.bin")))?);
        for step in 0..steps {
            let p=prefix+6*step;
            w.verify.begin_verification(&ids[p..p+6],&w.comm,&w.stream)?;
            let(y,n)=w.verify.pending_logits()?;ensure!(n==6,"score verification extent");
            let raw=unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?;
            w.verify.finish_verification(6,&w.comm,&w.stream)?;
            for r in 0..6 {
                let row:Vec<f32>=raw[r*129280*4..(r+1)*129280*4].chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
                ensure!(row.iter().all(|v|v.is_finite()),"nonfinite score logits");
                let m=row.iter().cloned().fold(f32::NEG_INFINITY,f32::max) as f64;
                let lse=m+row.iter().map(|v|((*v as f64)-m).exp()).sum::<f64>().ln();
                let truth=ids[p+r+1] as usize;
                nll.push(lse-row[truth] as f64);
                let arg=row.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).unwrap().0;
                if arg==truth {hits+=1;}
            }
            if rank==0 {use std::io::Write;dump.write_all(&raw)?;}
        }
        drop(dump);
        if rank!=0 {std::fs::remove_file(out.join(format!("claude-ab-score-r{rank}.bin")))?;}
        let mean=nll.iter().sum::<f64>()/nll.len() as f64;
        eprintln!("claude ab rank{rank} score {} tokens: mean nll {mean:.5} argmax hits {hits}",nll.len());
        score=json!({"steps":steps,"tokens":nll.len(),"mean_nll":mean,"argmax_hits":hits,"nll":nll});
        restore(w,&saved,anchor,&original)?;
    }
    let mut records=Vec::new();
    for &(width,live) in &cases {
        let input=&ids[prefix..prefix+live];
        let index=w.wides.iter().position(|t|t.physical_rows()==width).context("A/B width absent")?;
        let mut proofs=std::collections::BTreeMap::<u8,Value>::new();
        let mut control:Option<(Vec<f32>,Vec<f32>)>=None;
        for(trial,mode)in[0u8,1,1,0].into_iter().enumerate() {
            for repeat in 0..repeats {
                restore(w,&saved,anchor,&original)?;
                w.large.handoff_from(&w.verify)?;
                set_mode(w,index,&name,mode)?;
                w.control.agree(&[0xab,(width/128)as u8,trial as u8,repeat as u8])?;
                let at=Instant::now();
                w.large.prefill_wide(&mut w.wides[index..index+1],input,&w.comm,&w.stream)?;
                let milliseconds=at.elapsed().as_secs_f64()*1000.;
                let phases=serde_json::to_value(w.large.append_metrics()?)?;
                ensure!(phases["physical_rows"]==width,"A/B did not use selected width");
                let logits=w.large.last_prompt_logits()?;
                ensure!(logits.iter().all(|v|v.is_finite()),"nonfinite A/B output");
                w.session.as_mut().unwrap().resume_after_extend(&w.large,prefix as u64,input,&w.stream)?;
                w.verify.handoff_from(&w.large)?;let accepted=state(w)?;
                w.verify.begin_verification(&ids[prefix+live..prefix+live+6],&w.comm,&w.stream)?;
                let(y,n)=w.verify.pending_logits()?;ensure!(n==6,"A/B following extent");
                let raw_following=unsafe{w.cuda.read_device(y.ptr(),6*129280*4)}?;
                w.verify.finish_verification(0,&w.comm,&w.stream)?;
                ensure!(state(w)?==accepted,"A/B following rollback");
                let following:Vec<f32>=raw_following.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
                let logit_bytes:Vec<u8>=logits.iter().flat_map(|v|v.to_le_bytes()).collect();
                let proof=json!({"logits":hash(&logit_bytes),"following":hash(&raw_following),"state":accepted});
                let same_mode_exact=proofs.entry(mode).or_insert_with(||proof.clone())==&proof;
                let mut numerics=Value::Null;
                if mode==0&&control.is_none() {
                    // Cross-load comparisons (construction-time candidates) read these rows offline.
                    if rank==0&&std::env::var("CANADA_SCRATCH_CLAUDE_AB_DUMP").as_deref()==Ok("1") {
                        let mut bytes=logit_bytes.clone();bytes.extend_from_slice(&raw_following);
                        std::fs::write(out.join(format!("claude-ab-dump-m{width}.bin")),&bytes)?;
                    }
                    control=Some((logits.clone(),following.clone()));
                }
                if let Some((c_logits,c_following))=&control {
                    let(tv,max,same)=compare(c_logits,&logits);
                    let rows:Vec<Value>=(0..6).map(|r|{let(t,m,s)=compare(&c_following[r*129280..(r+1)*129280],
                        &following[r*129280..(r+1)*129280]);json!({"tv":t,"max_abs":m,"same_argmax":s})}).collect();
                    numerics=json!({"final_tv":tv,"final_max_abs":max,"final_same_argmax":same,"following":rows});
                }
                records.push(json!({"experiment":name,"physical_rows":width,"live":live,"trial":trial,"repeat":repeat,
                    "mode":mode,"warmup":repeat==0,"milliseconds":milliseconds,"same_mode_exact":same_mode_exact,
                    "numerics_vs_first_control":numerics,"phases":phases}));
                std::fs::write(out.join("claude-ab-partial.json"),serde_json::to_vec_pretty(&records)?)?;
                eprintln!("claude ab rank{rank} {name} M{width} live{live} trial{trial} mode{mode} repeat{repeat}: {milliseconds:.2} ms exact_in_mode={same_mode_exact}");
            }
        }
        set_mode(w,index,&name,0)?;
    }
    restore(w,&saved,anchor,&original)?;drop(saved);
    execute(w,Command::Finish(Finish::Length))?;execute(w,Command::Shutdown)?;
    std::fs::write(out.join("claude-ab-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"experiment":name,"prefix_tokens":prefix,"cold_begin_ms":cold_begin_ms,"score":score,"records":records,
        "scope":"Same-allocation ABBA retained-append prefill screen after a public long-context prefix; exactness within mode, cross-mode TV/max-abs/argmax on final and six teacher-forced following positions. No serving or default change."
    }))?)?;
    Ok(())
}
