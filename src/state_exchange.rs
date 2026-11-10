//! One inactive owner, with independently growing lossless prefix chunks.
//! Small/rotating regions use two bounded images during exchange. All old
//! small regions are retained until every outgoing small region is copied,
//! so wrapped SWA layouts cannot overwrite another region's outgoing bytes.
use super::{Descriptor,Region,Image as SmallImage,ImageBuffer,layout,MAX_IMAGE};
use crate::cuda::{Buffer,Cuda,Module,Stream};
use anyhow::{Result,ensure};
use sha2::{Digest,Sha256};
use std::ffi::c_void;

const CHUNK:usize=4<<20;
struct Prefix {descriptor:Descriptor,chunks:Vec<ImageBuffer>}
pub(crate) struct Image {
    cuda:Cuda,
    descriptors:Vec<Descriptor>,
    prefixes:Vec<Prefix>,
    small:SmallImage,
    saved:bool,
}
pub(crate) struct Prepared {
    previous:Vec<Descriptor>,
    next:Vec<Descriptor>,
    extra:Vec<Vec<ImageBuffer>>,
    small:SmallImage,
    table:Buffer<u64>,
    capacity:usize,
}
fn descriptors(regions:&[Region<'_>])->Vec<Descriptor>{regions.iter().map(|r|r.descriptor.clone()).collect()}
fn small_regions<'a>(regions:&[Region<'a>])->Vec<Region<'a>>{regions.iter().filter(|r|!r.prefix).cloned().collect()}
fn prefix_regions<'a>(regions:&'a[Region<'_>])->Vec<&'a Descriptor>{regions.iter().filter(|r|r.prefix).map(|r|&r.descriptor).collect()}
fn prefix_capacity(bytes:usize)->usize{bytes.div_ceil(CHUNK)*CHUNK}
fn shape(regions:&[Region<'_>],budget:usize)->Result<usize>{
    layout(&descriptors(regions),budget)?;
    let prefix=prefix_regions(regions);
    ensure!(!prefix.is_empty() && prefix.len()<=32 && regions.len()>prefix.len(),"exchange prefix/small partition");
    ensure!(prefix.iter().all(|p|p.pointer%16==0 && p.bytes%16==0),"exchange prefix alignment");
    let capacity=prefix.iter().map(|p|prefix_capacity(p.bytes)).sum::<usize>()
        +regions.iter().filter(|r|!r.prefix).map(|r|r.bytes()).sum::<usize>();
    ensure!(capacity<=budget && budget<=MAX_IMAGE,"exchange physical capacity exceeds budget");
    Ok(capacity)
}
fn overlaps(a:&Descriptor,b:&Descriptor)->bool{a.pointer<b.pointer+b.bytes as u64 && b.pointer<a.pointer+a.bytes as u64}
impl Image {
    pub(crate) fn estimate(regions:&[Region<'_>],budget:usize)->Result<usize>{shape(regions,budget)}
    pub(crate) fn try_allocate(cuda:&Cuda,regions:&[Region<'_>],budget:usize,stream:&Stream)->Result<Option<Self>>{
        ensure!(cuda.owns_stream(stream)&&regions.iter().all(|r|r.cuda.owns_stream(stream)),"exchange context");
        let capacity=shape(regions,budget)?;
        if cuda.admission_memory()?.admitted_available<crate::memory::floor_bytes()+capacity{return Ok(None);}
        let mut prefixes=Vec::new();
        for descriptor in prefix_regions(regions){
            let chunks=(0..descriptor.bytes.div_ceil(CHUNK)).map(|_|ImageBuffer::allocate(cuda,CHUNK)).collect::<Result<Vec<_>>>()?;
            prefixes.push(Prefix{descriptor:descriptor.clone(),chunks});
        }
        let Some(small)=SmallImage::try_allocate(cuda,&small_regions(regions),budget,stream)?else{return Ok(None);};
        if cuda.admission_memory()?.admitted_available<crate::memory::floor_bytes(){return Ok(None);}
        Ok(Some(Self{cuda:cuda.clone(),descriptors:descriptors(regions),prefixes,small,saved:false}))
    }
    pub(crate) fn bytes(&self)->usize{self.small.bytes()+self.prefixes.iter().map(|p|p.chunks.len()*CHUNK).sum::<usize>()}
    pub(crate) fn validate(&self,regions:&[Region<'_>],stream:&Stream)->Result<()>{
        ensure!(self.cuda.owns_stream(stream)&&regions.iter().all(|r|r.cuda.owns_stream(stream))
            && self.descriptors==descriptors(regions)
            && self.prefixes.iter().map(|p|&p.descriptor).eq(prefix_regions(regions)),"exchange storage/stream identity");
        self.small.validate(&small_regions(regions),stream)
    }
    fn copy_prefixes(&self,save:bool,stream:&Stream)->Result<()>{
        for p in &self.prefixes{for(i,chunk)in p.chunks.iter().enumerate(){
            let offset=i*CHUNK;let bytes=p.descriptor.bytes.saturating_sub(offset).min(CHUNK);if bytes==0{break;}
            let live=p.descriptor.pointer+offset as u64;
            unsafe{self.cuda.copy_device(if save{chunk.ptr()}else{live},if save{live}else{chunk.ptr()},bytes,stream)}?;
        }}Ok(())
    }
    pub(crate) fn save(&mut self,regions:&[Region<'_>],stream:&Stream)->Result<()>{
        self.validate(regions,stream)?;ensure!(!self.saved,"exchange image already captured");
        let result=self.copy_prefixes(true,stream).and_then(|_|self.small.save(&small_regions(regions),stream));
        result.and(stream.synchronize())?;self.saved=true;Ok(())
    }
    pub(crate) unsafe fn restore(&self,regions:&[Region<'_>],stream:&Stream)->Result<()>{
        self.validate(regions,stream)?;ensure!(self.saved,"uncaptured exchange image");
        let result=self.copy_prefixes(false,stream).and_then(|_|unsafe{self.small.restore(&small_regions(regions),stream)});
        result.and(stream.synchronize())
    }
    fn compatible(&self,next:&[Region<'_>])->Result<()>{
        layout(&descriptors(next),MAX_IMAGE)?;let prefix=prefix_regions(next);
        ensure!(prefix.len()==self.prefixes.len(),"exchange prefix set changed");
        for(old,new)in self.prefixes.iter().zip(&prefix){
            ensure!(old.descriptor.name==new.name&&old.descriptor.pointer==new.pointer
                &&new.bytes%16==0,"exchange prefix base/name changed");
            let union=Descriptor{bytes:old.descriptor.bytes.max(new.bytes),..old.descriptor.clone()};
            ensure!(self.small.descriptors.iter().all(|s|!overlaps(s,&union))
                &&next.iter().filter(|r|!r.prefix).all(|r|!overlaps(&r.descriptor,&union)),"exchange cross-layout alias");
        }Ok(())
    }
    /// Extra physical bytes needed while both old and new small images exist.
    /// Current retained prefix chunks are reused even when the new frontier is shorter.
    pub(crate) fn growth(&self,next:&[Region<'_>],budget:usize)->Result<(usize,usize)>{
        ensure!(self.saved,"uncaptured exchange growth");self.compatible(next)?;
        let prefix=prefix_regions(next);let mut extra=0;let mut prefix_bytes=0;let mut rows=0;
        for(old,new)in self.prefixes.iter().zip(prefix){
            let count=old.chunks.len().max(new.bytes.div_ceil(CHUNK));
            extra+=(count-old.chunks.len())*CHUNK;prefix_bytes+=count*CHUNK;rows+=count;
        }
        let small=next.iter().filter(|r|!r.prefix).map(|r|r.bytes()).sum::<usize>();
        ensure!(prefix_bytes+small<=budget&&budget<=MAX_IMAGE,"exchange growth exceeds budget");
        Ok((extra+small+rows*32,prefix_bytes+small))
    }
    /// Allocations only: the old image and the live owner remain unchanged on
    /// refusal or when another rank rejects its own preparation.
    pub(crate) fn try_prepare(&self,next:&[Region<'_>],budget:usize,stream:&Stream)->Result<Option<Prepared>>{
        ensure!(self.cuda.owns_stream(stream)&&next.iter().all(|r|r.cuda.owns_stream(stream)),"exchange preparation context");
        let(extra_bytes,capacity)=self.growth(next,budget)?;
        if self.cuda.admission_memory()?.admitted_available<crate::memory::floor_bytes()+extra_bytes{return Ok(None);}
        let mut extra=Vec::new();let mut table=Vec::new();
        for(old,new)in self.prefixes.iter().zip(prefix_regions(next)){
            let count=old.chunks.len().max(new.bytes.div_ceil(CHUNK));
            let added=(old.chunks.len()..count).map(|_|ImageBuffer::allocate(&self.cuda,CHUNK)).collect::<Result<Vec<_>>>()?;
            for(i,chunk)in old.chunks.iter().chain(&added).enumerate(){
                let offset=i*CHUNK;
                let save=new.bytes.saturating_sub(offset).min(CHUNK);
                let restore=old.descriptor.bytes.saturating_sub(offset).min(CHUNK);
                if save+restore>0{table.extend([new.pointer+offset as u64,chunk.ptr(),(save/16)as u64,(restore/16)as u64]);}
            }extra.push(added);
        }
        let Some(small)=SmallImage::try_allocate(&self.cuda,&small_regions(next),budget,stream)?else{return Ok(None);};
        let mut gpu_table=self.cuda.allocate(table.len())?;gpu_table.upload(&table)?;
        if self.cuda.admission_memory()?.admitted_available<crate::memory::floor_bytes(){return Ok(None);}
        Ok(Some(Prepared{previous:self.descriptors.clone(),next:descriptors(next),extra,small,table:gpu_table,capacity}))
    }
    /// Safety: exclusive, drained model ownership; poison that owner if any
    /// device operation fails. All ranks must have accepted preparation first.
    pub(crate) unsafe fn exchange(&mut self,mut prepared:Prepared,current:&[Region<'_>],previous:&[Region<'_>],
        module:&Module,stream:&Stream)->Result<()>{
        self.validate(previous,stream)?;self.compatible(current)?;
        ensure!(self.saved&&prepared.previous==self.descriptors&&prepared.next==descriptors(current),"stale exchange preparation");
        let function=module.function("ca_state_exchange_prefix")?;
        stream.synchronize()?;self.saved=false;
        // Save every rotating/small source before any restored small write.
        prepared.small.save(&small_regions(current),stream)?;
        let mut pointer=prepared.table.ptr();
        // Same disjoint uint4 ownership; a wider grid modestly improves the
        // measured long-prefix copy bandwidth on GB10 without extra storage.
        unsafe{function.launch([64,(prepared.table.len()/4).try_into()?,1],[256,1,1],0,stream,
            &mut[&mut pointer as *mut _ as *mut c_void])}?;
        let copied=unsafe{self.small.restore(&small_regions(previous),stream)};
        copied.and(stream.synchronize())?;
        for((old,extra),new)in self.prefixes.iter_mut().zip(prepared.extra).zip(prefix_regions(current)){
            old.chunks.extend(extra);old.descriptor=new.clone();
        }
        self.small=prepared.small;self.descriptors=prepared.next;self.saved=true;
        ensure!(self.bytes()==prepared.capacity,"exchange physical charge changed");Ok(())
    }
    pub(crate) fn manifest(&self)->serde_json::Value{
        serde_json::json!({"bytes":self.bytes(),"exchange":true,"chunk_bytes":CHUNK,
            "regions":self.descriptors.iter().map(|r|serde_json::json!({"name":r.name,"bytes":r.bytes})).collect::<Vec<_>>()})
    }
    pub(crate) fn digest(&self)->Result<String>{
        ensure!(self.saved,"uncaptured exchange digest");let mut hash=Sha256::new();
        // Same logical descriptor order as a conventional compact snapshot;
        // unused physical chunk tails never become semantic model state.
        for d in &self.descriptors{
            if let Some(p)=self.prefixes.iter().find(|p|p.descriptor==*d){
                for(i,chunk)in p.chunks.iter().enumerate(){let n=d.bytes.saturating_sub(i*CHUNK).min(CHUNK);if n==0{break;}hash.update(chunk.download_part(0,n)?);}
            }else{
                let index=self.small.descriptors.iter().position(|r|r==d).ok_or_else(||anyhow::anyhow!("exchange digest region absent"))?;
                let offset=self.small.offsets[index];
                for pos in (0..d.bytes).step_by(16<<20){hash.update(self.small.storage.download_part(offset+pos,(d.bytes-pos).min(16<<20))?);}
            }
        }Ok(format!("{:x}",hash.finalize()))
    }
}
