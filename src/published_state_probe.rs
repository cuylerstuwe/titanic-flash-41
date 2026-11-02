//! Read-only byte hashes of published rows. Whole-page checkpoint images also
//! retain unpublished speculative cells, which need not be erased on rejection.
//! No state restoration, model arithmetic or serving path uses this diagnostic.
use super::*;
use sha2::{Digest,Sha256};
use serde_json::{Value,json};

#[allow(clippy::too_many_arguments)]
fn trace_row(cuda:&Cuda,root:&std::path::Path,name:&str,dtype:&str,ptr:u64,bytes:usize,
    rows:usize,row:usize,stride:usize,all_hashes:bool)->Result<Value> {
    ensure!(rows==32&&row<rows&&bytes%rows==0,"copy trace row extent");
    let row_bytes=bytes/rows;
    let bound=if name=="source20_mask"&&dtype=="u8" {1048576/8}
        else if name=="source20_block_scores"&&dtype=="f32" {1048576/8*4}else{20480*4};
    ensure!(row_bytes>0&&row_bytes<=bound,"bounded copy trace row");
    let (selected,hashes)=if all_hashes {
        ensure!(stride==0,"contiguous boundary trace");
        let raw=unsafe{cuda.read_device(ptr,bytes)}?;
        let hashes=raw.chunks_exact(row_bytes).map(|b|format!("{:x}",Sha256::digest(b))).collect::<Vec<_>>();
        (raw[row*row_bytes..(row+1)*row_bytes].to_vec(),Some(hashes))
    }else{
        (unsafe{cuda.read_device(ptr+(row*if stride==0{row_bytes}else{stride})as u64,row_bytes)}?,None)
    };
    let file=format!("{name}-{dtype}-row{row}.bin");
    std::fs::write(root.join(&file),&selected)?;
    Ok(json!({"name":name,"dtype":dtype,"row":row,"file":file,"row_bytes":row_bytes,"ptr":ptr,"stride":stride,
        "sha256":format!("{:x}",Sha256::digest(&selected)),"row_hashes":hashes}))
}

fn trace_kv_row(cuda:&Cuda,root:&std::path::Path,name:&str,kv:&PagedKv,ids:&Candidates)->Result<Value> {
    let rows=32;let row=20;let width=ids.ids().len()/rows;
    ensure!(ids.ids().len()%rows==0&&[128,512,1024].contains(&width)&&ids.lengths().len()==rows
        &&kv.page_tokens()==64,"bounded copy trace KV geometry");
    let raw=unsafe{cuda.read_device(ids.ids().ptr()+(row*width*4)as u64,width*4)}?;
    let slots=raw.chunks_exact(4).map(|b|i32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    let n=unsafe{cuda.read_device(ids.lengths().ptr()+(row*4)as u64,4)}?;
    let count=i32::from_le_bytes(n.try_into().unwrap());ensure!(count>=0&&count as usize<=width,"KV trace count");
    let mut last_page:Option<(usize,Vec<u8>)>=None;let mut payload=Vec::new();
    for &slot in slots.iter().take(count as usize) {
        if slot<0 {payload.extend_from_slice(&[0u8;583]);continue;}
        let slot=slot as usize;ensure!(slot<kv.capacity(),"KV trace slot bound");
        let page=slot/64;
        if last_page.as_ref().is_none_or(|(p,_)|*p!=page) {
            last_page=Some((page,unsafe{cuda.read_device(kv.data().ptr()+(page*kv.page_bytes())as u64,kv.page_bytes())}?));
        }
        let data=&last_page.as_ref().unwrap().1;let local=slot%64;
        payload.extend_from_slice(&data[local*576..(local+1)*576]);
        payload.extend_from_slice(&data[64*576+local*8..64*576+local*8+7]);
    }
    let file=format!("{name}-row20-kv.bin");std::fs::write(root.join(&file),&payload)?;
    Ok(json!({"name":name,"row":row,"count":count,"slots":slots,"file":file,
        "row_bytes":583,"sha256":format!("{:x}",Sha256::digest(&payload))}))
}

fn hash_rows(hash:&mut Sha256,page:&[u8],rows:impl IntoIterator<Item=usize>,
    payload:usize,scale_stride:usize,scale_bytes:usize) {
    for row in rows {
        hash.update(&page[row*payload..(row+1)*payload]);
        let offset=64*payload+row*scale_stride;
        hash.update(&page[offset..offset+scale_bytes]);
    }
}

fn prefix(cuda:&Cuda,data:&Buffer<u8>,slots:usize,stride:usize,payload:usize,
    scale_stride:usize,scale_bytes:usize)->Result<Value> {
    let pages=slots.div_ceil(64);
    ensure!(pages*stride<=data.bytes()&&64*(payload+scale_stride)<=stride,
        "published prefix byte geometry");
    let mut hash=Sha256::new();
    // Keep host staging below 10 MiB, including at a full million positions.
    for first in (0..pages).step_by(256) {
        let count=(pages-first).min(256);
        let bytes=unsafe{cuda.read_device(data.ptr()+(first*stride)as u64,count*stride)}?;
        for i in 0..count {
            let rows=(slots-(first+i)*64).min(64);
            hash_rows(&mut hash,&bytes[i*stride..(i+1)*stride],0..rows,payload,scale_stride,scale_bytes);
        }
    }
    Ok(json!({"published_slots":slots,"observable_bytes":slots*(payload+scale_bytes),
        "sha256":format!("{:x}",hash.finalize())}))
}

impl Target {
    /// Selected row values and per-row boundary hashes from one already
    /// completed M32 forward. Shared routed-expert scratch is deliberately
    /// excluded: its post-forward contents belong only to the final layer.
    pub(crate) fn copy_repeat_trace(&self,root:&std::path::Path,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_COPY_PROFILE_TRACE").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_PAIRED_SERVING_PROFILE_SCREEN").as_deref()==Ok("1")
            &&self.rows==32&&self.pending.is_some()&&self.lease.is_busy_revision()
            &&self.source_phase==Phase::Verify&&self.history.position>100000&&!root.exists()
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,"copy repeat trace admission");
        stream.synchronize()?;std::fs::create_dir_all(root)?;
        let mut layers=Vec::new();
        for(id,layer)in self.layer_owners().into_iter().enumerate() {
            let dir=root.join(format!("layer{id}"));std::fs::create_dir(&dir)?;
            let mut taps=Vec::new();
            for(name,b)in [("attn_input",layer.attn_boundary().input()),
                ("attn_reduced",layer.attn_reduced()),("ffn_input",layer.ffn_boundary().input()),
                ("ffn_reduced",layer.ffn_reduced()),("residual",layer.output())] {
                taps.push(trace_row(&self.cuda,&dir,name,"bf16",b.ptr(),b.bytes(),32,20,0,true)?);
            }
            for(name,ptr,bytes,stride)in layer.attention().taps() {
                if ["qa","kv","normalized","qb","q_rope","inverse_attention","woa","output"].contains(&name) {
                    taps.push(trace_row(&self.cuda,&dir,&format!("attn_{name}"),"bf16",ptr,bytes,32,20,stride,false)?);
                }
            }
            let f=layer.ffn();
            for(name,b)in [("shared_up",f.shared_up()),("activation",f.activation()),
                ("shared_down",f.shared_down()),("ffn_output",f.output())] {
                taps.push(trace_row(&self.cuda,&dir,name,"bf16",b.ptr(),b.bytes(),32,20,0,false)?);
            }
            for(name,b)in [("router_logits",f.router().logits()),("router_weights",f.router().routing())] {
                taps.push(trace_row(&self.cuda,&dir,name,"f32",b.ptr(),b.bytes(),32,20,0,false)?);
            }
            let ids=f.router().ids();
            taps.push(trace_row(&self.cuda,&dir,"router_ids","i32",ids.ptr(),ids.bytes(),32,20,0,false)?);
            layers.push(json!({"layer":id,"taps":taps}));
        }
        ensure!(layers.len()==40,"complete copy trace layers");
        let mut index_taps=Vec::new();let dir=root.join("index28");std::fs::create_dir(&dir)?;
        let index=self.late.index(28).context("copy trace index28")?;let q=index.query();
        for(name,b)in [("query_projection",q.query_projection()),("head_projection",q.head_projection())] {
            index_taps.push(trace_row(&self.cuda,&dir,name,"bf16",b.ptr(),b.bytes(),32,20,0,true)?);
        }
        let packed=q.packed_query();
        for(name,dtype,ptr,bytes)in [
            ("packed_query","u8",packed.payload().ptr(),packed.payload().bytes()),
            ("query_scales","i32",packed.scales().ptr(),packed.scales().bytes()),
            ("query_weights","f32",packed.weights().ptr(),packed.weights().bytes()),
            ("selected","i32",q.selected().ptr(),q.selected().bytes()),
            ("physical_ids","i32",index.candidates().ids().ptr(),index.candidates().ids().bytes())] {
            index_taps.push(trace_row(&self.cuda,&dir,name,dtype,ptr,bytes,32,20,0,true)?);
        }
        let source=self.late.source();let mask=source.query().candidate_mask().context("source20 candidate mask")?;
        ensure!(!source.query().candidate_mask_is_compact(),"M32 complete candidate mask");
        index_taps.push(trace_row(&self.cuda,&dir,"source20_mask","u8",mask.ptr(),mask.bytes(),32,20,0,true)?);
        let kv=vec![trace_kv_row(&self.cuda,&dir,"swa28",&self.late.caches()[8],&self.step.candidates)?,
            trace_kv_row(&self.cuda,&dir,"source20",source.cache(),index.candidates())?];
        ensure!(!source.query().key_sharded_active(),"block score trace requires the active publisher scores");
        let mut blocks=Vec::new();
        for(name,ptr,bytes)in source.query().published_block_taps() {
            if name=="block_mask" {continue;}
            blocks.push(trace_row(&self.cuda,&dir,&format!("source20_{name}"),
                if name=="block_scores" {"f32"}else{"i32"},ptr,bytes,32,20,0,true)?);
        }
        ensure!(blocks.len()==3,"complete source20 block score trace");
        std::fs::write(root.join("index.json"),serde_json::to_vec_pretty(&json!({
            "position":self.history.position,"rows":32,"sampled_row":20,"layers":layers,"index28":index_taps,"kv28":kv,"source20_blocks":blocks,
            "scope":"One completed forward, all-row layer-boundary hashes and row20 private activation/router values. No altered computation or reference oracle."}))?)?;
        Ok(())
    }
    pub(crate) fn published_state_digests_for_probe(&self,stream:&Stream)->Result<Value> {
        self.published_state_digests_with_swa_window(stream,
            crate::swa_layout::WINDOW+crate::swa_layout::MAX_ROWS)
    }
    /// Cross-owner encoder import rebuilds the decoder's complete causal
    /// window, not historical decoder cells older than that window. Keep the
    /// wider recovery-window hashes above as separate diagnostic evidence.
    pub(crate) fn causal_state_digests_for_bank_probe(&self,stream:&Stream)->Result<Value> {
        ensure!(std::env::var("CANADA_SCRATCH_BANK_ENCODER_SCREEN").as_deref()==Ok("1"),
            "causal bank digest requires the finite bank checkpoint screen");
        self.published_state_digests_with_swa_window(stream,crate::swa_layout::WINDOW)
    }
    fn published_state_digests_with_swa_window(&self,stream:&Stream,retained:usize)->Result<Value> {
        ensure!((std::env::var("CANADA_SCRATCH_PAIRED_SERVING_PROFILE_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_BANK_ENCODER_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_ENCODER_PREFIX_IMPORT_SCREEN").as_deref()==Ok("1"))
            &&self.lease.is_current()&&self.pending.is_none()&&self.wide.is_none()
            &&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize
            &&self.swa_layout.max_rows()>=8192,"published state diagnostic ownership");
        stream.synchronize()?;
        let position=self.history.position as usize;
        let mut slots=(0..crate::swa_layout::OFFSET).collect::<std::collections::BTreeSet<_>>();
        slots.extend((position.saturating_sub(retained)..position)
            .map(|p|self.swa_layout.slot(p)));
        let mut pages=std::collections::BTreeMap::<usize,Vec<usize>>::new();
        for &slot in &slots {pages.entry(slot/64).or_default().push(slot%64);}
        let mut swa=Vec::new();
        for cache in self.early_caches.iter().chain(self.group8.caches())
            .chain(self.group14.caches()).chain(self.late.caches()) {
            ensure!(cache.page_tokens()==64,"published SWA page geometry");
            let mut hash=Sha256::new();
            for (&page,rows)in &pages {
                ensure!(page<cache.pages(),"published SWA page bounds");
                let bytes=unsafe{self.cuda.read_device(cache.data().ptr()+(page*cache.page_bytes())as u64,
                    cache.page_bytes())}?;
                hash_rows(&mut hash,&bytes,rows.iter().copied(),576,8,7);
            }
            swa.push(format!("{:x}",hash.finalize()));
        }
        let mut compressed=Vec::new();
        for (layer,source)in [2,8,14,20].into_iter().zip([
            &self.source2,self.group8.source(),self.group14.source(),self.late.source()]) {
            // C2 publishes only completed pairs. An odd pending token is held
            // by the separately checked carry ring, not an extra visible KV.
            let count=COMPRESSED_OFFSET+position/(if layer==20{1}else{2});
            ensure!(source.cache().page_tokens()==64,"published compressed page geometry");
            compressed.push(json!({"layer":layer,
                "kv":prefix(&self.cuda,source.cache().data(),count,source.cache().page_bytes(),576,8,7)?,
                "keys":prefix(&self.cuda,source.keys().data(),count,64*68,64,4,4)?}));
        }
        Ok(json!({"position":position,"swa_retained_slots":slots.len(),"swa":swa,"compressed":compressed}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn published_hash_detects_past_changes_but_ignores_future_and_padding() {
        for (payload,stride,observed)in [(576,8,7),(64,4,4)] {
            for rows in [1,31,63,64] {
                let mut page=vec![0x5a;64*(payload+stride)+32];
                let digest=|page:&[u8]| {let mut h=Sha256::new();hash_rows(&mut h,page,0..rows,payload,stride,observed);h.finalize()};
                let expected=digest(&page);
                page[rows*payload..64*payload].fill(0x17);
                page[64*payload+rows*stride..].fill(0x29);
                for row in 0..rows {page[64*payload+row*stride+observed..64*payload+(row+1)*stride].fill(0x33);}
                assert_eq!(digest(&page),expected);
                page[(rows-1)*payload]^=1;assert_ne!(digest(&page),expected);
                page[(rows-1)*payload]^=1;
                page[64*payload+(rows-1)*stride]^=1;assert_ne!(digest(&page),expected);
            }
        }
    }
}
