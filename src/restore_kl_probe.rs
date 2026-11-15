//! Diagnostic only (CANADA_SCRATCH_RESTORE_KL_SCREEN=1): the first-token distribution of a continuing turn,
//! computed four ways on the same prompt B, which shares |A|-1 tokens with the previous prompt A as consecutive
//! chat turns do:
//! - cold: one prefill of B;
//! - split: B prefilled in two calls at A's aligned checkpoint position (a chunking-only yardstick);
//! - base: B restores the aligned encoder image A saved (today's turns);
//! - near: A continues that image and saves a near-end image at |A|-1 (CANADA_SCRATCH_ENCODER_NEAR_END=1); B
//!   restores it, with the ring replay of CANADA_SCRATCH_ENCODER_SHORT_SUFFIX=1 when fewer than 128 rows follow.
//! Rank 0 reports KL(cold||x), total variation and top-1 agreement per case. Images are not bank-retained
//! (CANADA_SCRATCH_BANK_ENCODER_CHECKPOINT=0) because each arm starts from a reset owner.
use crate::{control_channel::ControlChannel,cuda::Stream,roce::Roce,target::{Target,EncoderPrefixRestore}};
use anyhow::{ensure,Result};
use serde_json::{json,Value};
use std::path::Path;

fn softmax(x:&[f32])->Vec<f64> {
    let m=x.iter().fold(f32::NEG_INFINITY,|a,&b|a.max(b)) as f64;
    let e:Vec<f64>=x.iter().map(|&v|(f64::from(v)-m).exp()).collect();
    let s:f64=e.iter().sum();
    e.into_iter().map(|v|v/s).collect()
}
fn compare(p:&[f32],q:&[f32])->Value {
    let (a,b)=(softmax(p),softmax(q));
    let (mut kl,mut tv)=(0f64,0f64);
    for (x,y) in a.iter().zip(&b) {
        if *x>0. {kl+=x*(x.ln()-y.max(1e-300).ln());}
        tv+=(x-y).abs();
    }
    let top=|v:&[f32]|v.iter().enumerate().max_by(|l,r|l.1.total_cmp(r.1)).map(|(i,_)|i).unwrap_or(0);
    json!({"kl":kl,"tv":tv/2.,"top1_equal":top(p)==top(q)})
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(large:&mut Target,verify:&mut Target,wides:&mut[Target],comm:&Roce,stream:&Stream,
    control:&mut ControlChannel,output:&Path,rank:usize)->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_RESTORE_KL_SCREEN").as_deref()==Ok("1")
        &&crate::config::encoder_near_end_checkpoint()?==1&&crate::config::encoder_short_suffix()?
        &&!crate::config::bank_encoder_checkpoint_requested()
        &&large.position()==0&&large.context_capacity()==1048576,"restore KL probe admission");
    if std::env::var("CANADA_SCRATCH_RESTORE_LAYER_DIFF").as_deref()==Ok("1") {
        return run_layers(large,verify,wides,comm,stream,control,output,rank);
    }
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let mut lineage=7_710_000u64;
    let mut records=Vec::new();
    // (|A|, rows of B after the |A|-1 shared tokens): CANADA_SCRATCH_RESTORE_KL_CASES=24 spreads 24 cases over
    // ~9K-118K previous prompts and 30-600 added rows (|A| 130-229 rows past its aligned position); default six.
    let wide=std::env::var("CANADA_SCRATCH_RESTORE_KL_CASES").as_deref()==Ok("24");
    let cases:Vec<(usize,usize)>=if wide {
        (0..24usize).map(|k|((70+k*37)*128+130+(k*17)%100,[30usize,60,100,150,300,600][k%6])).collect()
    }else{vec![(16260,40),(16260,150),(20400,90),(20400,400),(32900,60),(32900,200)]};
    for (case,&(a_len,added)) in cases.iter().enumerate() {
        ensure!(tokens.len()>=a_len+8+added,"restore KL fixture extent");
        let a=&tokens[..a_len];
        let b:Vec<u32>=tokens[..a_len-1].iter().chain(&tokens[a_len+7..a_len+7+added]).copied().collect();
        let aligned=((a_len-128)/128)*128;
        ensure!(a_len-1-aligned>=128,"near-end case needs an inline-admissible prefix");
        control.agree(&[0xd7,case as u8])?;
        // The owner that ran last (large) hands the writer back before the reset.
        let mut fresh=|large:&mut Target,verify:&mut Target,lineage:&mut u64|->Result<()> {
            verify.handoff_from(large)?;*lineage+=1;verify.reset(*lineage,comm,stream)?;large.handoff_from(verify)
        };
        fresh(large,verify,&mut lineage)?;
        large.prefill_wide(wides,&b,comm,stream)?;
        let cold=large.last_prompt_logits()?;
        let cold_phases=serde_json::to_value(large.append_metrics()?)?;
        fresh(large,verify,&mut lineage)?;
        large.prefill_wide(wides,&b[..aligned],comm,stream)?;
        large.prefill_wide(wides,&b[aligned..],comm,stream)?;
        let split=large.last_prompt_logits()?;
        fresh(large,verify,&mut lineage)?;
        let budget=large.encoder_checkpoint_image_bytes(aligned)?;
        let (image_a,_)=large.prefill_with_encoder_checkpoint(wides,None,&a[..aligned],&a[aligned..],Some(budget),comm,stream)?;
        let image_a=image_a.ok_or_else(||anyhow::anyhow!("aligned image not saved"))?;
        fresh(large,verify,&mut lineage)?;lineage+=1;
        let (_,base_details)=large.prefill_with_encoder_checkpoint(wides,
            Some(EncoderPrefixRestore::Borrowed(&image_a,lineage)),&[],&b[aligned..],None,comm,stream)?;
        let base=large.last_prompt_logits()?;
        fresh(large,verify,&mut lineage)?;lineage+=1;
        let budget=large.encoder_checkpoint_image_bytes(a_len-1)?;
        let (image_n,_)=large.prefill_with_encoder_checkpoint(wides,
            Some(EncoderPrefixRestore::Borrowed(&image_a,lineage)),&a[aligned..a_len-1],&a[a_len-1..],Some(budget),comm,stream)?;
        let image_n=image_n.ok_or_else(||anyhow::anyhow!("near-end image not saved"))?;
        ensure!(image_n.position()==a_len-1,"near-end image position");
        fresh(large,verify,&mut lineage)?;lineage+=1;
        let (_,near_details)=large.prefill_with_encoder_checkpoint(wides,
            Some(EncoderPrefixRestore::Borrowed(&image_n,lineage)),&[],&b[a_len-1..],None,comm,stream)?;
        let near=large.last_prompt_logits()?;
        // Level 2 (CANADA_SCRATCH_ENCODER_SHORT_SUFFIX=2): fewer than 128 rows after the near-end image run as the
        // qualified EXTEND (all layers over the new rows, decoder floor at the restore point).
        let extend=if added<128&&crate::config::encoder_short_suffix_level()?==2 {
            fresh(large,verify,&mut lineage)?;lineage+=1;
            large.set_short_restore_extend(true)?;
            let result=large.prefill_with_encoder_checkpoint(wides,
                Some(EncoderPrefixRestore::Borrowed(&image_n,lineage)),&[],&b[a_len-1..],None,comm,stream);
            large.set_short_restore_extend(false)?;
            let (_,details)=result?;
            ensure!(details["short_restore_extend"]==true,"EXTEND arm did not take the EXTEND path");
            Some(large.last_prompt_logits()?)
        }else{None};
        let record=json!({"case":case,"previous_tokens":a_len,"prompt_tokens":b.len(),"aligned":aligned,
            "base_rows":b.len()-aligned,"near_rows":b.len()-(a_len-1),
            "split_vs_cold":compare(&cold,&split),"base_vs_cold":compare(&cold,&base),
            "near_vs_cold":compare(&cold,&near),"near_vs_base":compare(&base,&near),
            "extend_vs_cold":extend.as_ref().map(|x|compare(&cold,x)),
            "cold_phases":cold_phases,"base_suffix":base_details["suffix_phases"],"near_suffix":near_details["suffix_phases"]});
        if rank==0 {eprintln!("restore KL case {case}: {}",json!({"split":record["split_vs_cold"],
            "base":record["base_vs_cold"],"near":record["near_vs_cold"],"extend":record["extend_vs_cold"]}));}
        records.push(record);
        std::fs::write(output.join("restore-kl-partial.json"),serde_json::to_vec_pretty(&records)?)?;
    }
    fresh_end(large,verify,&mut lineage,comm,stream)?;
    let mean=|key:&str|->Value {
        let v:Vec<f64>=records.iter().filter_map(|r|r[key]["kl"].as_f64()).collect();
        let flips=records.iter().filter(|r|r[key]["top1_equal"]==false).count();
        json!({"cases":v.len(),"mean_kl":if v.is_empty(){0.}else{v.iter().sum::<f64>()/v.len() as f64},
            "max_kl":v.iter().cloned().fold(0.,f64::max),"top1_flips":flips})
    };
    let summary=json!({"split_vs_cold":mean("split_vs_cold"),"base_vs_cold":mean("base_vs_cold"),
        "near_vs_cold":mean("near_vs_cold"),"near_vs_base":mean("near_vs_base"),"extend_vs_cold":mean("extend_vs_cold")});
    if rank==0 {eprintln!("restore KL summary: {summary}");}
    std::fs::write(output.join("restore-kl-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"summary":summary,"records":records,
        "scope":"First-token distributions of a continuing turn (B shares |A|-1 tokens with A): cold, split-at-aligned (chunking yardstick), aligned-image restore (today), near-end image restore (s1447 switches). Diagnostic only."
    }))?)?;
    Ok(())
}
fn fresh_end(large:&mut Target,verify:&mut Target,lineage:&mut u64,comm:&Roce,stream:&Stream)->Result<()> {
    verify.handoff_from(large)?;*lineage+=1;verify.reset(*lineage,comm,stream)?;large.handoff_from(verify)
}

/// Layers whose output is compared (0-20 run on every prompt row, 21-39 only on the last 128).
const LAYERS:[usize;14]=[0,1,2,3,5,8,11,13,14,17,20,21,25,39];
const DIFF_BLOCKS:usize=43;
/// Teacher-force `continuation` through the verifier in six-row blocks from the state `large` holds; per block the
/// residual after each of LAYERS (six rows x 5,120 BF16; rank 0 only). Leaves the verifier owning the writer.
fn capture(large:&mut Target,verify:&mut Target,continuation:&[u32],comm:&Roce,stream:&Stream,rank:usize)->Result<Vec<Vec<Vec<u8>>>> {
    verify.handoff_from(large)?;
    let mut out=Vec::new();
    for block in continuation.chunks_exact(6) {
        verify.begin_verification(block,comm,stream)?;
        verify.finish_verification(6,comm,stream)?;
        stream.synchronize()?;
        // Every layer keeps its output until the next forward; the owner's lease is current again after the commit
        // (the serving mid-pass dump reads at the same point).
        out.push(if rank==0 {verify.layer_outputs_for_probe(&LAYERS)?} else {Vec::new()});
    }
    Ok(out)
}
fn relative(a:&[u8],b:&[u8],row:usize)->f64 {
    let f=|x:&[u8]|->Vec<f64>{x[row*5120*2..(row+1)*5120*2].chunks_exact(2)
        .map(|c|f64::from(half::bf16::from_bits(u16::from_le_bytes([c[0],c[1]])).to_f32())).collect()};
    let (x,y)=(f(a),f(b));
    let num:f64=x.iter().zip(&y).map(|(p,q)|(p-q)*(p-q)).sum();
    let den:f64=x.iter().map(|p|p*p).sum();
    if den>0. {(num/den).sqrt()} else {0.}
}
/// CANADA_SCRATCH_RESTORE_LAYER_DIFF=1 (with the restore KL screen's switches): where does a restored prompt's
/// state first differ from a cold prefill's? (Nov 13: greedy serving requests showed the residual after layer 13
/// of newly generated rows ~10 % apart after a near-end restore at 124K, bit-equal after a complete-image hit.)
/// For each case the same prompt B is reached four ways as in the KL screen (cold; split at the aligned position,
/// the chunking yardstick; aligned image restore; near-end image restore), then the same 258 continuation tokens
/// are teacher-forced and the residual after LAYERS is compared with the cold arm's, by layer and by distance
/// from the prompt end. Cases: CANADA_SCRATCH_RESTORE_LAYER_CASES=len:added,... (default 16260:1,117890:1).
#[allow(clippy::too_many_arguments)]
fn run_layers(large:&mut Target,verify:&mut Target,wides:&mut[Target],comm:&Roce,stream:&Stream,
    control:&mut ControlChannel,output:&Path,rank:usize)->Result<()> {
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let cases:Vec<(usize,usize)>=match std::env::var("CANADA_SCRATCH_RESTORE_LAYER_CASES") {
        Ok(v)=>v.split(',').map(|x|{let (a,b)=x.split_once(':').ok_or_else(||anyhow::anyhow!("case is len:added"))?;
            Ok((a.parse::<usize>()?,b.parse::<usize>()?))}).collect::<Result<_>>()?,
        Err(_)=>vec![(16260,1),(117890,1)],
    };
    ensure!(!cases.is_empty()&&cases.len()<=6,"restore layer diff cases");
    let mut lineage=7_730_000u64;
    let mut owns=false;   // the verifier owns the writer (after a capture)
    let mut records=Vec::new();
    const BUCKETS:[(usize,usize,&str);5]=[(0,6,"0-5"),(6,32,"6-31"),(32,128,"32-127"),(128,192,"128-191"),(192,258,"192-257")];
    for (case,&(a_len,added)) in cases.iter().enumerate() {
        ensure!(added>=1&&tokens.len()>=a_len+7+added+DIFF_BLOCKS*6,"restore layer diff fixture extent");
        let a=&tokens[..a_len];
        let b:Vec<u32>=tokens[..a_len-1].iter().chain(&tokens[a_len+7..a_len+7+added]).copied().collect();
        let continuation=&tokens[a_len+7+added..a_len+7+added+DIFF_BLOCKS*6];
        let aligned=((a_len-128)/128)*128;
        ensure!(a_len-1-aligned>=128,"near-end case needs an inline-admissible prefix");
        control.agree(&[0xd9,case as u8])?;
        let mut fresh=|large:&mut Target,verify:&mut Target,lineage:&mut u64,owns:&mut bool|->Result<()> {
            if !*owns {verify.handoff_from(large)?;}
            *lineage+=1;verify.reset(*lineage,comm,stream)?;large.handoff_from(verify)?;*owns=false;Ok(())
        };
        fresh(large,verify,&mut lineage,&mut owns)?;
        large.prefill_wide(wides,&b,comm,stream)?;
        let cold=capture(large,verify,continuation,comm,stream,rank)?;owns=true;
        fresh(large,verify,&mut lineage,&mut owns)?;
        large.prefill_wide(wides,&b[..aligned],comm,stream)?;
        large.prefill_wide(wides,&b[aligned..],comm,stream)?;
        let split=capture(large,verify,continuation,comm,stream,rank)?;owns=true;
        fresh(large,verify,&mut lineage,&mut owns)?;
        let budget=large.encoder_checkpoint_image_bytes(aligned)?;
        let (image_a,_)=large.prefill_with_encoder_checkpoint(wides,None,&a[..aligned],&a[aligned..],Some(budget),comm,stream)?;
        let image_a=image_a.ok_or_else(||anyhow::anyhow!("aligned image not saved"))?;
        fresh(large,verify,&mut lineage,&mut owns)?;lineage+=1;
        large.prefill_with_encoder_checkpoint(wides,
            Some(EncoderPrefixRestore::Borrowed(&image_a,lineage)),&[],&b[aligned..],None,comm,stream)?;
        let base=capture(large,verify,continuation,comm,stream,rank)?;owns=true;
        fresh(large,verify,&mut lineage,&mut owns)?;lineage+=1;
        let budget=large.encoder_checkpoint_image_bytes(a_len-1)?;
        let (image_n,_)=large.prefill_with_encoder_checkpoint(wides,
            Some(EncoderPrefixRestore::Borrowed(&image_a,lineage)),&a[aligned..a_len-1],&a[a_len-1..],Some(budget),comm,stream)?;
        let image_n=image_n.ok_or_else(||anyhow::anyhow!("near-end image not saved"))?;
        ensure!(image_n.position()==a_len-1,"near-end image position");
        fresh(large,verify,&mut lineage,&mut owns)?;lineage+=1;
        let (_,near_details)=large.prefill_with_encoder_checkpoint(wides,
            Some(EncoderPrefixRestore::Borrowed(&image_n,lineage)),&[],&b[a_len-1..],None,comm,stream)?;
        let near=capture(large,verify,continuation,comm,stream,rank)?;owns=true;
        if rank==0 {
            let mut arms=serde_json::Map::new();
            for (name,arm) in [("split",&split),("base",&base),("near",&near)] {
                let mut table=serde_json::Map::new();
                for (li,&layer) in LAYERS.iter().enumerate() {
                    let mut row=serde_json::Map::new();
                    for &(lo,hi,label) in BUCKETS.iter() {
                        let (mut sum,mut max,mut n)=(0f64,0f64,0usize);
                        for position in lo..hi.min(DIFF_BLOCKS*6) {
                            let d=relative(&cold[position/6][li],&arm[position/6][li],position%6);
                            sum+=d;max=max.max(d);n+=1;
                        }
                        row.insert(label.into(),json!({"mean":sum/n.max(1) as f64,"max":max}));
                    }
                    table.insert(format!("L{layer}"),Value::Object(row));
                }
                eprintln!("restore layer diff case {case} ({} prompt tokens, {} fresh rows) {name} against cold, mean relative difference x1e3 by layer; positions 0-5 | 6-31 | 32-127 | 128-191 | 192-257: {}",
                    b.len(),b.len()-(a_len-1),LAYERS.iter().map(|l|format!("L{l} {}",BUCKETS.iter()
                        .map(|x|format!("{:.1}",table[&format!("L{l}")][x.2]["mean"].as_f64().unwrap()*1e3)).collect::<Vec<_>>().join("|")))
                        .collect::<Vec<_>>().join("  "));
                arms.insert(name.into(),Value::Object(table));
            }
            records.push(json!({"case":case,"previous_tokens":a_len,"prompt_tokens":b.len(),"aligned":aligned,
                "fresh_rows_after_near_end":b.len()-(a_len-1),"near_suffix":near_details["suffix_phases"],"against_cold":arms}));
            std::fs::write(output.join("restore-layer-diff-partial.json"),serde_json::to_vec_pretty(&records)?)?;
        }
    }
    if !owns {verify.handoff_from(large)?;}
    lineage+=1;verify.reset(lineage,comm,stream)?;large.handoff_from(verify)?;
    std::fs::write(output.join("restore-layer-diff-summary.json"),serde_json::to_vec_pretty(&json!({"rank":rank,
        "layers":LAYERS,"records":records,
        "scope":"Residual after selected layers for 258 teacher-forced continuation tokens after the same prompt reached cold, split at the aligned position, by aligned image restore and by near-end image restore; relative L2 difference from the cold arm by layer and distance from the prompt end. Diagnostic only."}))?)?;
    Ok(())
}
