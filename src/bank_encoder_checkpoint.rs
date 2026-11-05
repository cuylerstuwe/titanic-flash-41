//! Bank-retained encoder checkpoints and finite byte-identity diagnostics.
use super::*;

fn admitted()->Result<()> {
    ensure!(std::env::var("CANADA_SCRATCH_BANK_ENCODER_SCREEN").as_deref()==Ok("1"),
        "bank-retained encoder operations require the finite mechanism screen");
    Ok(())
}

impl Target {
    /// Capture identity provenance only while the ancestor's omitted pages
    /// are still present in the exact bank state being snapshotted.
    pub(crate) fn bank_encoder_witness(&self,saved:&EncoderPrefixImage)->Option<BankEncoderWitness> {
        let binding=saved.1.as_ref()?;
        self.encoder_prefix_reusable_from(saved).then(||
            BankEncoderWitness::capture(binding,&self.lease,saved.position()as u64))
    }

    /// Caller has successfully restored either the complete immutable snapshot
    /// that captured these witnesses, or a same-bank sparse prefix at/after
    /// every witnessed ancestor. In the latter case they must have been valid
    /// immediately before restore: its omitted pages remain untouched. Never
    /// transfer these bindings across an import or unrelated reset.
    pub(crate) fn rebind_encoder_ancestors(&self,saved:&[BankEncoderWitness])->Result<()> {
        ensure!(self.pending.is_none(),"ancestor rebind needs a committed restore");
        for ancestor in saved {ancestor.validate_restore(&self.lease,self.history.position)?;}
        for ancestor in saved {ancestor.rebind(self.history.lineage);}
        Ok(())
    }

    /// Capture a complete committed prefill frontier after a scheduler yield.
    /// Optional storage is voted on before allocation and before any snapshot
    /// copy. No model arithmetic or target frontier is changed.
    pub(crate) fn try_save_committed_bank_encoder(&self,budget:usize,reserve:usize,
        recycled:Option<crate::state_image::Image>,
        comm:&Roce,stream:&Stream,admit:&mut impl FnMut(bool)->Result<bool>)
        ->Result<Option<EncoderPrefixImage>> {
        ensure!(crate::config::bank_encoder_checkpoint_requested()&&encoder_checkpoint_enabled()
            &&self.rows==128&&self.history.position>=256,"committed bank encoder snapshot admission");
        self.state_image_guard(comm,stream)?;
        let reused=recycled.as_ref().map_or(0,crate::state_image::Image::bytes);
        ensure!(recycled.is_none()||reused==budget,"bank snapshot recycled storage extent");
        let required=reserve.checked_add(budget-reused).ok_or_else(||anyhow::anyhow!("bank snapshot byte overflow"))?;
        if !admit(self.cuda.admission_memory()?.admitted_available>=required)?{return Ok(None);}
        let regions=self.state_image_regions_for_layers_mode(self.history.position as usize,true,true,true)?;
        let mut allocation=match recycled {
            Some(retired)=>retired.try_recycle(&regions,budget,stream)?,
            None=>crate::state_image::Image::try_allocate(&self.cuda,&regions,budget,stream)?,
        };
        if !admit(allocation.is_some()&&self.cuda.admission_memory()?.admitted_available>=reserve)? {
            return Ok(None);
        }
        let mut image=allocation.take().ok_or_else(||anyhow::anyhow!("all-rank bank snapshot admission differs"))?;
        image.configure_copy_kernel(self.io.core_module())?;
        stream.synchronize()?;comm.health()?;self.lease.begin()?;
        ensure!(self.encoder_cache.as_ref().unwrap().error()?==0,"committed bank encoder snapshot fault");
        image.save(&regions,stream)?;comm.health()?;self.lease.finish_read()?;
        Ok(Some(EncoderPrefixImage(StateImage{image:image.into(),owner:self.lease.fork(),history:self.history.clone(),
            context:self.context,rank:self.rank,stream:self.stream,compact:true},
            Some(bank_encoder_lineage::Binding::new(self.history.lineage)))))
    }

    pub(crate) fn bank_encoder_image_bytes_for_probe(&self,position:usize,retained:bool)->Result<usize> {
        admitted()?;
        Ok(self.state_image_regions_for_layers_mode(position,true,true,retained)?.iter().map(|r|r.bytes()).sum())
    }

    pub(crate) fn save_bank_encoder_image_for_probe(&self,retained:bool,budget:usize,
        comm:&Roce,stream:&Stream)->Result<EncoderPrefixImage> {
        admitted()?;self.state_image_guard(comm,stream)?;
        ensure!(encoder_checkpoint_enabled()&&self.rows==128&&self.history.position>=256,
            "bank encoder snapshot requires complete committed prefill");
        let regions=self.state_image_regions_for_layers_mode(self.history.position as usize,true,true,retained)?;
        let mut image=crate::state_image::Image::allocate(&self.cuda,&regions,budget,stream)?;
        image.configure_copy_kernel(self.io.core_module())?;
        stream.synchronize()?;comm.health()?;self.lease.begin()?;
        ensure!(self.encoder_cache.as_ref().unwrap().error()?==0,"bank encoder snapshot fault");
        image.save(&regions,stream)?;comm.health()?;self.lease.finish_read()?;
        Ok(EncoderPrefixImage(StateImage{image:image.into(),owner:self.lease.fork(),history:self.history.clone(),
            context:self.context,rank:self.rank,stream:self.stream,compact:true},
            retained.then(||bank_encoder_lineage::Binding::new(self.history.lineage))))
    }

    /// Preflight omitted-prefix copies before acquiring destination authority.
    /// Caller verifies lineage and disjointness of the complete target owners.
    pub(super) fn bank_encoder_prefix_copy_plan(&self,source:&Self,position:usize)
        ->Result<Vec<(u64,u64,usize)>> {
        let mut copies=Vec::new();
        for(layer,(dst,src))in [2,8,14,20].into_iter().zip([
            &self.source2,self.group8.source(),self.group14.source(),self.late.source()].into_iter().zip([
            &source.source2,source.group8.source(),source.group14.source(),source.late.source()])) {
            let ratio=if layer==20{1}else{2};
            let pages=(COMPRESSED_OFFSET+position.div_ceil(ratio)).div_ceil(64).saturating_sub(1);
            ensure!(dst.cache().page_tokens()==64&&src.cache().page_tokens()==64
                &&dst.cache().page_bytes()==src.cache().page_bytes(),"bank prefix import geometry");
            if pages==0{continue;}
            for(d,s,bytes)in [(dst.cache().data(),src.cache().data(),pages*dst.cache().page_bytes()),
                (dst.keys().data(),src.keys().data(),pages*64*68)] {
                ensure!(bytes<=d.bytes()&&bytes<=s.bytes()
                    &&(d.ptr()+bytes as u64<=s.ptr()||s.ptr()+bytes as u64<=d.ptr()),
                    "bank prefix import bounds or alias");
                copies.push((d.ptr(),s.ptr(),bytes));
            }
        }
        Ok(copies)
    }

    /// Hash only the pages omitted by the sparse image. This costly readback
    /// is a GPU-produced byte check, never a serving operation or CPU model.
    pub(crate) fn bank_encoder_immutable_digests_for_probe(&self,position:usize,stream:&Stream)
        ->Result<serde_json::Value> {
        admitted()?;
        ensure!(self.lease.is_current()&&self.pending.is_none()&&position>0
            &&position<=self.history.position as usize&&self.cuda.owns_stream(stream),
            "immutable bank prefix needs committed ownership");
        stream.synchronize()?;
        let mut records=Vec::new();
        for(layer,source)in [2,8,14,20].into_iter().zip([
            &self.source2,self.group8.source(),self.group14.source(),self.late.source()]) {
            let ratio=if layer==20{1}else{2};
            let end=(COMPRESSED_OFFSET+position.div_ceil(ratio)).div_ceil(64).saturating_sub(1);
            if end==0{continue;}
            for(name,ptr,bytes)in [
                ("kv",source.cache().data().ptr(),end*source.cache().page_bytes()),
                ("keys",source.keys().data().ptr(),end*64*68)] {
                use sha2::{Digest,Sha256};
                let mut hash=Sha256::new();
                for offset in (0..bytes).step_by(16<<20) {
                    hash.update(unsafe{self.cuda.read_device(ptr+offset as u64,(bytes-offset).min(16<<20))}?);
                }
                records.push(serde_json::json!({"layer":layer,"region":name,"bytes":bytes,
                    "sha256":format!("{:x}",hash.finalize())}));
            }
        }
        Ok(serde_json::json!(records))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn speculative_and_prefill_locations_never_reenter_omitted_compressed_pages() {
        // Odd C2 boundaries, page edges, large wide chunks and ragged padding.
        // This validates address planning only; GPU packing/publication still
        // needs the separate immutable-page digest and full-restore comparison.
        for position in [255usize,256,257,8191,8192,8193,131071,131072,131073,1047552] {
            for (rows,live)in [(6usize,3usize),(6,6),(16,12),(128,32),(128,17),(8192,8192)] {
                if position+live>1048576{continue;}
                let p=Plan::append_with_layout(position as u64,live,rows,1048576,
                    crate::swa_layout::Layout::WideRing8192).unwrap();
                for ratio in [1usize,2] {
                    let omitted_end=((COMPRESSED_OFFSET+position.div_ceil(ratio)).div_ceil(64)-1)*64;
                    let raw=if ratio==1{&p.raw_single}else{&p.raw_pair};
                    for(&pos,&location)in p.positions.iter().zip(raw) {
                        if location!=0&&(pos as usize+1)%ratio==0 {
                            assert!(location as usize/ratio>=omitted_end);
                        }
                    }
                    assert!(raw[live..].iter().all(|&v|v==0));
                }
            }
        }
    }
}
