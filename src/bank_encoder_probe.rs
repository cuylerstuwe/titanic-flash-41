//! Sparse bank-tail snapshots versus complete encoder images on real GPU state.
use super::*;

fn capture(large:&Target,verify:&mut Target,teacher:&[u32],cuda:&Cuda,comm:&Roce,stream:&Stream)->Result<Value> {
    let prompt=large.last_prompt_logits()?;
    let mut hash=Sha256::new();for value in prompt{hash.update(value.to_bits().to_le_bytes());}
    verify.handoff_from(large)?;
    Ok(json!({"prompt_logits":format!("{:x}",hash.finalize()),
        "following_logits":logits(verify,teacher,0,cuda,comm,stream)?,
        "published":verify.published_state_digests_for_probe(stream)?,
        "causal_published":verify.causal_state_digests_for_bank_probe(stream)?,
        "carry":verify.carry_digests()?,"encoder":verify.encoder_tail_digest()?}))
}

pub(super) fn run(w:&mut Worker,other:&mut Target,other_verify:&mut Target,rank:usize,out:&Path,
    ownership:Value)->Result<()> {
    ensure!(w.large.context_capacity()==1048576&&w.long_verify.is_some()&&w.copy32_verify.is_some(),
        "bank encoder screen requires full context and all verification widths");
    ensure!(crate::config::encoder_prefix_import_requested(),"bank screen requires explicit donor import");
    let fixture:Value=serde_json::from_slice(include_bytes!("../tests/fixtures/long-context-continuation-v2.json"))?;
    let tokens:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let teacher=&tokens[250000..250006];
    other.reset(1001,&w.comm,&w.stream)?;
    other.prefill_wide(&mut[],&tokens[20000..20257],&w.comm,&w.stream)?;
    let mut records=Vec::new();
    for(case,prefix)in [8193usize,131073].into_iter().enumerate() {
        let serial=2000+case as u64*100;
        if case>0{w.large.handoff_from(&w.verify)?;}
        w.large.reset(serial,&w.comm,&w.stream)?;
        w.large.prefill_wide(&mut w.wides,&tokens[..prefix],&w.comm,&w.stream)?;
        let full=w.large.save_bank_encoder_image_for_probe(false,2<<30,&w.comm,&w.stream)?;
        let sparse=w.large.save_bank_encoder_image_for_probe(true,2<<30,&w.comm,&w.stream)?;
        let omitted=w.large.bank_encoder_immutable_digests_for_probe(prefix,&w.stream)?;
        let full_bytes=full.bytes();let sparse_bytes=sparse.bytes();
        let projected=json!({"position":1047552,
            "full_device_bytes":w.large.bank_encoder_image_bytes_for_probe(1047552,false)?,
            "sparse_device_bytes":w.large.bank_encoder_image_bytes_for_probe(1047552,true)?});
        other_verify.handoff_from(other)?;
        let foreign_before=state(other_verify,teacher,&w.cuda,&w.comm,&w.stream)?;
        other.handoff_from(other_verify)?;
        let foreign_refused=other.restore_encoder_checkpoint_full(&sparse,serial+1,&w.comm,&w.stream).is_err();
        other_verify.handoff_from(other)?;
        ensure!(w.control.all_admit(foreign_refused&&foreign_before==state(other_verify,teacher,&w.cuda,&w.comm,&w.stream)?)?,
            "sparse encoder image crossed owner identity");
        other.handoff_from(other_verify)?;
        // Overwrite rings and the final compressed pages, including rejected
        // M16/M32 future writes and an odd partial C2 commit.
        w.large.prefill_wide(&mut w.wides,&tokens[prefix..prefix+259],&w.comm,&w.stream)?;
        w.verify.handoff_from(&w.large)?;
        for(rows,accepted)in [(6usize,3usize),(16,0),(32,5),(6,6)] {
            let target=match rows{6=>&mut w.verify,16=>{
                w.long_verify.as_mut().unwrap().handoff_from(&w.verify)?;w.long_verify.as_mut().unwrap()
            },32=>{
                w.copy32_verify.as_mut().unwrap().handoff_from(&w.verify)?;w.copy32_verify.as_mut().unwrap()
            },_=>unreachable!()};
            target.begin_verification(&tokens[22000..22000+rows],&w.comm,&w.stream)?;
            target.finish_verification(accepted,&w.comm,&w.stream)?;
            if rows==16{w.verify.handoff_from(w.long_verify.as_ref().unwrap())?;}
            if rows==32{w.verify.handoff_from(w.copy32_verify.as_ref().unwrap())?;}
        }
        ensure!(w.control.all_admit(omitted==w.verify.bank_encoder_immutable_digests_for_probe(prefix,&w.stream)?)?,
            "generation or rejected verification changed omitted compressed pages");
        w.large.handoff_from(&w.verify)?;
        // Rewinding a newer sparse image must leave a witnessed older parent
        // usable. The later full-image comparison below checks the old parent's
        // prompt logits and following verification, not just the cache marker.
        let ancestor=w.large.bank_encoder_witness(&sparse)
            .context("older sparse parent lost before newer-prefix restore")?;
        let newer=w.large.save_bank_encoder_image_for_probe(true,2<<30,&w.comm,&w.stream)?;
        w.large.restore_encoder_checkpoint_full(&newer,serial+8,&w.comm,&w.stream)?;
        ensure!(!w.large.encoder_prefix_reusable_from(&sparse),"restore failed to advance lineage");
        w.large.rebind_encoder_ancestors(&[ancestor])?;
        ensure!(w.large.encoder_prefix_reusable_from(&sparse),"older sparse ancestor was not rebound");
        drop(newer);
        let suffix=&tokens[150000..150221];
        let at=Instant::now();
        // Match the import destination's128-row schedule. A wide512 suffix
        // versus two128 chunks is a numerical-path comparison, not an import
        // correctness oracle; both are valid ordinary prefill choices.
        let(_,sparse_details)=w.large.prefill_with_encoder_checkpoint(&mut [],
            Some(crate::target::EncoderPrefixRestore::Borrowed(&sparse,serial+10)),&[],suffix,None,&w.comm,&w.stream)?;
        let sparse_restore_seconds=at.elapsed().as_secs_f64();
        let actual=capture(&w.large,&mut w.verify,teacher,&w.cuda,&w.comm,&w.stream)?;
        w.large.handoff_from(&w.verify)?;
        w.large.prefill_with_encoder_checkpoint(&mut [],
            Some(crate::target::EncoderPrefixRestore::Borrowed(&sparse,serial+11)),&[],suffix,None,&w.comm,&w.stream)?;
        let repeated=capture(&w.large,&mut w.verify,teacher,&w.cuda,&w.comm,&w.stream)?;
        ensure!(w.control.all_admit(repeated==actual)?,"restored sparse image could not be reused exactly");
        w.large.handoff_from(&w.verify)?;
        let donor_before=w.large.published_state_digests_for_probe(&w.stream)?;
        let(_,import_details)=other.prefill_with_encoder_checkpoint(&mut[],
            Some(crate::target::EncoderPrefixRestore::Imported(&sparse,&w.large,serial+12)),&[],suffix,None,&w.comm,&w.stream)?;
        let imported=capture(other,other_verify,teacher,&w.cuda,&w.comm,&w.stream)?;
        let donor_after=w.large.published_state_digests_for_probe(&w.stream)?;
        // The replacement rebuilds every live decoder row. The wider256-row
        // debug digest also covers obsolete decoder storage that the bounded
        // decoder-tail replay never needed to import. Retain both views.
        let import_exact=["prompt_logits","following_logits","causal_published","carry","encoder"]
            .into_iter().all(|key|imported[key]==actual[key]);
        std::fs::write(out.join(format!("bank-encoder-c{case}-import.json")),
            serde_json::to_vec_pretty(&json!({"expected":actual,"imported":imported,
                "observable_exact":import_exact,"donor_before":donor_before,"donor_after":donor_after,
                "source_plan":sparse_details,"import_plan":import_details}))?)?;
        ensure!(w.control.all_admit(import_exact&&donor_before==donor_after)?,
            "sparse prefix import differs or mutated its donor");
        other.handoff_from(other_verify)?;
        let at=Instant::now();
        w.large.prefill_with_encoder_checkpoint(&mut [],
            Some(crate::target::EncoderPrefixRestore::Borrowed(&full,serial+20)),&[],suffix,None,&w.comm,&w.stream)?;
        let full_restore_seconds=at.elapsed().as_secs_f64();
        let expected=capture(&w.large,&mut w.verify,teacher,&w.cuda,&w.comm,&w.stream)?;
        std::fs::write(out.join(format!("bank-encoder-c{case}-compare.json")),
            serde_json::to_vec_pretty(&json!({"actual":actual,"expected":expected}))?)?;
        ensure!(w.control.all_admit(actual==expected)?,"bank-tail restore differs from full encoder image");
        w.large.handoff_from(&w.verify)?;
        let before=w.large.published_state_digests_for_probe(&w.stream)?;
        let stale_refused=w.large.restore_encoder_checkpoint_full(&sparse,serial+21,&w.comm,&w.stream).is_err();
        ensure!(w.control.all_admit(stale_refused&&before==w.large.published_state_digests_for_probe(&w.stream)?)?,
            "stale bank image was accepted after a lineage change");
        let new_sparse=w.large.save_bank_encoder_image_for_probe(true,2<<30,&w.comm,&w.stream)?;
        w.large.reset(serial+30,&w.comm,&w.stream)?;
        w.large.prefill_wide(&mut [],&tokens[21000..21257],&w.comm,&w.stream)?;
        let before=w.large.published_state_digests_for_probe(&w.stream)?;
        let reset_refused=w.large.restore_encoder_checkpoint_full(&new_sparse,serial+31,&w.comm,&w.stream).is_err();
        ensure!(w.control.all_admit(reset_refused&&before==w.large.published_state_digests_for_probe(&w.stream)?)?,
            "bank reset did not invalidate sparse image before mutation");
        let peer_before=other.published_state_digests_for_probe(&w.stream)?;
        let stale_import_refused=other.import_encoder_checkpoint_full(&new_sparse,&w.large,serial+32,&w.comm,&w.stream).is_err();
        ensure!(w.control.all_admit(stale_import_refused&&peer_before==other.published_state_digests_for_probe(&w.stream)?)?,
            "stale donor prefix was imported or mutated destination");
        w.verify.handoff_from(&w.large)?;
        records.push(json!({"prefix_tokens":prefix,"suffix_tokens":suffix.len(),"full_device_bytes":full_bytes,
            "sparse_device_bytes":sparse_bytes,"projected_near1M":projected,
            "full_restore_seconds":full_restore_seconds,"sparse_restore_seconds":sparse_restore_seconds,
            "all_exact":true,"omitted_pages_unchanged":true,"foreign_refused":foreign_refused,
            "lineage_change_refused":stale_refused,"reset_refused":reset_refused,
            "repeat_restore_exact":true,"older_ancestor_restore_exact":true,
            "import_exact":true,"import_donor_unchanged":true,
            "stale_import_refused":stale_import_refused}));
    }
    std::fs::write(out.join("bank-encoder-summary.json"),serde_json::to_vec_pretty(&json!({
        "rank":rank,"all_exact":true,"ownership":ownership,"records":records,
        "scope":"Two odd8193/131073 boundaries. Sparse encoder tail versus full image after259-token append and partial/rejected M6/M16/M32 writes. Full prompt logits, followingM6, published state, carry and encoder tail compared. Reuse after restore and explicit cross-bank import compared, donor unchanged. Omitted pages unchanged; stale-lineage/reset/foreign restore and stale import refused. No serving selection,1M return, timing qualification or task-quality claim."}))?)?;
    Ok(())
}
