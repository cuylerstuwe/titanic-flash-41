//! Fixed-prefix proof of explicit immutable encoder import in both directions.
//! No HTTP, sampler-law or throughput claim is made by this small screen.
use super::*;

fn capture(target:&mut Target,teacher:&[u32],cuda:&Cuda,comm:&Roce,stream:&Stream)->Result<Value> {
    let following=logits(target,teacher,0,cuda,comm,stream)?;
    Ok(json!({"position":target.position(),"following_logits":following,
        "carry":target.carry_digests()?,"encoder":target.encoder_tail_digest()?,
        "regions":target.published_state_digests_for_probe(stream)?}))
}

#[allow(clippy::too_many_arguments)]
fn direction(source:&mut Target,source_verify:&mut Target,destination:&mut Target,destination_verify:&mut Target,
    tokens:&[u32],teacher:&[u32],lineage:u64,cuda:&Cuda,comm:&Roce,stream:&Stream,
    control:&mut ControlChannel)->Result<Value> {
    source.reset(lineage,comm,stream)?;
    let prefix=&tokens[..8192];let tail=&tokens[8192..8448];
    let(image,_)=source.prefill_encoder_checkpoint_for_probe(&mut[],prefix,tail,256<<20,comm,stream)?;
    let original_image=image.digest_for_import_probe()?;
    source_verify.handoff_from(source)?;
    destination.reset(lineage+1,comm,stream)?;
    destination.prefill_wide(&mut[],&tokens[9000..9257],comm,stream)?;
    destination_verify.handoff_from(destination)?;
    let untouched=capture(destination_verify,teacher,cuda,comm,stream)?;
    destination.handoff_from(destination_verify)?;
    let refused=destination.restore_encoder_checkpoint_full(&image,lineage+2,comm,stream).is_err();
    destination_verify.handoff_from(destination)?;
    ensure!(control.all_admit(refused&&untouched==capture(destination_verify,teacher,cuda,comm,stream)?)?,
        "ordinary restore accepted a foreign encoder image or changed state");
    let mut records=Vec::new();
    for (case,suffix_len)in [256usize,17].into_iter().enumerate() {
        let suffix=&tokens[8192..8192+suffix_len];let serial=lineage+10+case as u64*10;
        source.handoff_from(source_verify)?;
        if suffix_len>=128 {
            source.prefill_with_encoder_checkpoint(&mut[],
                Some(crate::target::EncoderPrefixRestore::Borrowed(&image,serial)),&[],suffix,None,comm,stream)?;
        }else{
            source.restore_encoder_checkpoint_full(&image,serial,comm,stream)?;
            source.prefill_wide(&mut[],suffix,comm,stream)?;
        }
        source_verify.handoff_from(source)?;
        let expected=capture(source_verify,teacher,cuda,comm,stream)?;
        destination.handoff_from(destination_verify)?;
        let began=Instant::now();
        if suffix_len>=128 {
            destination.prefill_with_encoder_checkpoint(&mut[],
                Some(crate::target::EncoderPrefixRestore::Imported(&image,source_verify,serial+1)),
                &[],suffix,None,comm,stream)?;
        }else{
            destination.import_encoder_checkpoint_full(&image,source_verify,serial+1,comm,stream)?;
            destination.prefill_wide(&mut[],suffix,comm,stream)?;
        }
        let seconds=began.elapsed().as_secs_f64();
        destination_verify.handoff_from(destination)?;
        let actual=capture(destination_verify,teacher,cuda,comm,stream)?;
        let donor=capture(source_verify,teacher,cuda,comm,stream)?;
        let immutable=image.digest_for_import_probe()?==original_image;
        ensure!(control.all_admit(expected==actual&&donor==expected&&immutable)?,
            "encoder import changed target logits/published state, donor or immutable image");
        records.push(json!({"suffix_tokens":suffix_len,"all_exact":true,"donor_unchanged":true,
            "image_unchanged":immutable,"seconds":seconds,"state":actual}));
    }
    source.handoff_from(source_verify)?;destination.handoff_from(destination_verify)?;
    Ok(json!({"prefix_tokens":prefix.len(),"image_bytes":image.bytes(),"foreign_restore_refused":refused,
        "image_sha256":original_image,"records":records}))
}

pub(super) fn run(w:&mut Worker,other:&mut Target,verify:&mut Target,rank:usize,out:&Path,ownership:Value)->Result<()> {
    ensure!(crate::config::encoder_prefix_import_requested()&&w.large.context_capacity()==32768,
        "bounded encoder import screen admission");
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let teacher=&tokens[250000..250006];
    let forward=direction(&mut w.large,&mut w.verify,other,verify,&tokens,teacher,1000,
        &w.cuda,&w.comm,&w.stream,&mut w.control)?;
    let reverse=direction(other,verify,&mut w.large,&mut w.verify,&tokens[20000..],teacher,2000,
        &w.cuda,&w.comm,&w.stream,&mut w.control)?;
    std::fs::write(out.join("encoder-import-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"all_exact":true,"ownership":ownership,"directions":[forward,reverse],
        "scope":"Two independent owners, immutable8192-token encoder parents imported in both directions.256-token replacement suffix and17-token reconstructed suffix; followingM6 logits, all published causal regions, carry and encoder state compared with ordinary same-owner restore. Donor and immutable image unchanged, ordinary foreign restore rejected. Not HTTP, full1M, drafter/sampler, or performance qualification."
    }))?)?;
    Ok(())
}
