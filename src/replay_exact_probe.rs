//! Diagnostic only (CANADA_SCRATCH_REPLAY_EXACT_SCREEN=1): what the bounded decoder replay costs against exact decoder
//! windows, on the same prompts (the screen proposed for "exact encoder-decoder prefill" in the independent leads
//! document of Nov 13). Decoding runs every layer with full windows, so a prompt whose last K tokens are fed through
//! the verifier as forced tokens ends with decoder windows that no longer depend on the replay once K >= 19 x 127 +
//! 128 (each decoder layer's window is built from its inputs, and the replay's truncation moves 127 positions
//! further out per layer). Each case reaches the first N tokens of the long-context fixture four ways on a reset
//! owner:
//! - bounded: wide prefill of all N tokens (serving: decoder layers on the last 128 tokens, floored windows);
//! - exact: wide prefill of N - 2,562 tokens, then 2,562 forced tokens;
//! - exact2: wide prefill of N - 3,072 tokens, then 3,072 forced tokens (also exact; against `exact` it is the
//!   yardstick for everything that is not the replay: where the wide path hands over to the six-row path);
//! - near: wide prefill of N - 258 tokens, then 258 forced tokens (what a turn's first 258 tokens repair).
//! After each: teacher-forced logits for the fixture's next 258 tokens on the verifier. Rank 0 reports KL / total
//! variation / top-1 flips of bounded against exact beside exact2 against exact, by distance from the prompt end.
use crate::{control_channel::ControlChannel,cuda::{Cuda,Stream},roce::Roce,target::Target};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use std::path::Path;

const BLOCKS:usize=43;
fn softmax(x:&[f32])->Vec<f64> {
    let m=x.iter().fold(f32::NEG_INFINITY,|a,&b|a.max(b)) as f64;
    let e:Vec<f64>=x.iter().map(|&v|(f64::from(v)-m).exp()).collect();
    let s:f64=e.iter().sum();
    e.into_iter().map(|v|v/s).collect()
}
/// (KL(p||q), total variation, same top-1, entropy of q in nats, q's largest probability) of two logit rows.
fn pair(p:&[f32],q:&[f32])->(f64,f64,bool,f64,f64) {
    let (a,b)=(softmax(p),softmax(q));
    let (mut kl,mut tv,mut h,mut top_q)=(0f64,0f64,0f64,0f64);
    for (x,y) in a.iter().zip(&b) {
        if *x>0. {kl+=x*(x.ln()-y.max(1e-300).ln());}
        if *y>0. {h-=y*y.ln();}
        top_q=top_q.max(*y);
        tv+=(x-y).abs();
    }
    let top=|v:&[f32]|v.iter().enumerate().max_by(|l,r|l.1.total_cmp(r.1)).map(|(i,_)|i).unwrap_or(0);
    (kl,tv/2.,top(p)==top(q),h,top_q)
}
fn forced(a:&[Vec<f32>],b:&[Vec<f32>])->Value {
    let rows:Vec<(f64,f64,bool,f64,f64)>=a.iter().zip(b).map(|(x,y)|pair(x,y)).collect();
    let part=|lo:usize,hi:usize|->Value {
        let r=&rows[lo.min(rows.len())..hi.min(rows.len())];
        if r.is_empty() {return Value::Null;}
        let n=r.len() as f64;let mut kl:Vec<f64>=r.iter().map(|x|x.0).collect();kl.sort_by(f64::total_cmp);
        json!({"positions":r.len(),"mean_kl":r.iter().map(|x|x.0).sum::<f64>()/n,"median_kl":kl[kl.len()/2],"max_kl":kl[kl.len()-1],
            "mean_tv":r.iter().map(|x|x.1).sum::<f64>()/n,"top1_flips":r.iter().filter(|x|!x.2).count(),
            // The second arm's own distributions: a flat region (high entropy, low top-1 probability) makes any two
            // arms differ, whatever is compared.
            "second_entropy":r.iter().map(|x|x.3).sum::<f64>()/n,"second_top1_probability":r.iter().map(|x|x.4).sum::<f64>()/n})
    };
    json!({"all":part(0,rows.len()),"first_32":part(0,32),"32_to_128":part(32,128),"after_128":part(128,rows.len())})
}
/// Feed `tokens` (a multiple of six) through the verifier as forced blocks, committing each.
fn force(verify:&mut Target,tokens:&[u32],comm:&Roce,stream:&Stream)->Result<()> {
    ensure!(tokens.len()%6==0,"forced tokens come in six-row blocks");
    for block in tokens.chunks_exact(6) {
        verify.begin_verification(block,comm,stream)?;
        verify.finish_verification(6,comm,stream)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(cuda:&Cuda,large:&mut Target,verify:&mut Target,wides:&mut[Target],comm:&Roce,stream:&Stream,
    control:&mut ControlChannel,output:&Path,rank:usize)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_REPLAY_EXACT_SCREEN").as_deref()==Ok("1")
        &&large.position()==0&&large.context_capacity()==1048576&&!wides.is_empty(),"replay exactness probe admission");
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let cases:Vec<usize>=match std::env::var("CANADA_SCRATCH_REPLAY_EXACT_CASES") {
        Ok(v)=>v.split(',').map(|x|x.parse::<usize>().map_err(Into::into)).collect::<Result<_>>()?,
        Err(_)=>vec![8000,16260,32900,65000],
    };
    ensure!(!cases.is_empty()&&cases.len()<=8&&cases.iter().all(|&n|n>=3072+1024&&tokens.len()>=n+BLOCKS*6),"replay exactness cases");
    let mut lineage=7_810_000u64;
    let mut records=Vec::new();
    let mut verifier_owns=false;
    // (name, forced tokens before the prompt end); 0 = the whole prompt through the wide path.
    const ARMS:[(&str,usize);4]=[("bounded",0),("exact",2562),("exact2",3072),("near",258)];
    for (case,&n) in cases.iter().enumerate() {
        control.agree(&[0xd8,case as u8])?;
        let continuation=&tokens[n..n+BLOCKS*6];
        let mut arms=Vec::new();
        for &(_,forced_tokens) in ARMS.iter() {
            if !verifier_owns {verify.handoff_from(large)?;}
            lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;
            let started=std::time::Instant::now();
            large.prefill_wide(wides,&tokens[..n-forced_tokens],comm,stream)?;
            let wide_seconds=started.elapsed().as_secs_f64();
            verify.handoff_from(large)?;verifier_owns=true;
            force(verify,&tokens[n-forced_tokens..n],comm,stream)?;
            let seconds=started.elapsed().as_secs_f64();
            let rows=crate::dspark_generation::teacher_forced_logits(cuda,verify,continuation,BLOCKS,comm,stream)?;
            arms.push((rows,wide_seconds,seconds));
        }
        let record=json!({"case":case,"prompt_tokens":n,"forced_positions":BLOCKS*6,
            "seconds":ARMS.iter().zip(&arms).map(|(a,r)|json!({"arm":a.0,"forced_before_end":a.1,"wide":r.1,"total":r.2})).collect::<Vec<_>>(),
            "forced":{"bounded_vs_exact":forced(&arms[1].0,&arms[0].0),"exact2_vs_exact":forced(&arms[1].0,&arms[2].0),
                "near_vs_exact":forced(&arms[1].0,&arms[3].0),"bounded_vs_exact2":forced(&arms[2].0,&arms[0].0)}});
        if rank==0 {eprintln!("replay exactness case {case} ({n} tokens): {}",json!({
            "bounded_vs_exact":record["forced"]["bounded_vs_exact"],"exact2_vs_exact":record["forced"]["exact2_vs_exact"]["all"],
            "near_vs_exact":record["forced"]["near_vs_exact"]["all"]}));}
        records.push(record);
        std::fs::write(output.join("replay-exact-partial.json"),serde_json::to_vec_pretty(&records)?)?;
    }
    lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;
    let mean=|pair:&str,part:&str,key:&str|->f64 {
        let v:Vec<f64>=records.iter().filter_map(|r|r["forced"][pair][part][key].as_f64()).collect();
        if v.is_empty(){0.}else{v.iter().sum::<f64>()/v.len() as f64}
    };
    let table=|pair:&str|json!({"first_32_mean_kl":mean(pair,"first_32","mean_kl"),"32_to_128_mean_kl":mean(pair,"32_to_128","mean_kl"),
        "after_128_mean_kl":mean(pair,"after_128","mean_kl"),"all_mean_kl":mean(pair,"all","mean_kl"),"all_mean_tv":mean(pair,"all","mean_tv"),
        "top1_flips":records.iter().filter_map(|r|r["forced"][pair]["all"]["top1_flips"].as_u64()).sum::<u64>()});
    let summary=json!({"cases":records.len(),"bounded_vs_exact":table("bounded_vs_exact"),"exact2_vs_exact":table("exact2_vs_exact"),
        "near_vs_exact":table("near_vs_exact"),"bounded_vs_exact2":table("bounded_vs_exact2")});
    if rank==0 {eprintln!("replay exactness summary: {summary}");}
    std::fs::write(output.join("replay-exact-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"summary":summary,"records":records,
        "scope":"Bounded decoder replay (128-row tail, floored windows) against exact decoder windows (the last 2,562 prompt tokens decoded as forced tokens), on the same prompts: teacher-forced next-token distributions for 258 positions. exact2_vs_exact (3,072 against 2,562 forced tokens, both exact) is the yardstick for everything else. Diagnostic only."
    }))?)?;
    Ok(())
}
