//! Opt-in TP4 key partition for the early source indexers. Score math
//! and precision are unchanged. Local top512 contains every global winner;
//! the final top512 is selected from their exact score/ID byte payloads.
use crate::{cuda::{Buffer,Cuda,Module,Stream}, index::{KeyCache,KeySlots,QueryRows},
    mqa::{Dense,KeyMatrix,Native,Ranges}, roce::Roce, topk::{self,Selection,Windows}};
use anyhow::{ensure,Result};
use std::{cell::Cell,ffi::c_void,rc::Rc};
macro_rules! args {($($v:ident),* $(,)?)=>{[$(std::ptr::addr_of_mut!($v).cast::<c_void>()),*]};}
pub(crate) struct KeySharded {
    cuda:Cuda,module:Module,columns:usize,rows:usize,keys:Rc<KeyMatrix>,scorer:Dense,
    ranges:Ranges,windows:Windows,selection:Selection,
    merge_windows:Windows,merge:Selection,merge_scores:Buffer<f32>,ids:Buffer<i32>,
    send:Buffer<f32>,receive:Buffer<f32>,valid:Buffer<i32>,dirty:Cell<bool>,
    binding:Cell<(u64,u64,usize)>,
    publisher:Option<crate::key_sharded_publisher::Publisher>,
}
impl KeySharded {
    pub(crate) fn new(cuda:&Cuda,core:&[u8],native:Rc<Native>,topk:Rc<topk::Native>,
        query:Rc<QueryRows>,columns:usize)->Result<Self>{
        let rows=query.rows();
        ensure!([6,16].contains(&rows)&&[524288,1048576].contains(&columns),"key-shard verification geometry");
        let local=columns/4;let zero=vec![0;rows];
        let keys=Rc::new(KeyMatrix::new(cuda,local)?);
        let scorer=Dense::new(cuda,native,query,keys.clone(),local,rows.div_ceil(4)*4*local*4)?;
        let selection=Selection::new(cuda,topk.clone(),rows,local,scorer.stride(),512)?;
        let publisher=(columns==1048576).then(||crate::key_sharded_publisher::Publisher::new(cuda,core,topk.clone(),local,rows)).transpose()?;
        let words=if publisher.is_some(){crate::key_sharded_publisher::PACKED_WORDS_PER_ROW}else{1024};
        let mut valid=cuda.allocate(1)?;valid.zero()?;
        Ok(Self{cuda:cuda.clone(),module:cuda.module(core)?,columns,rows,keys,scorer,
            ranges:Ranges::new(cuda,local,local,&zero,&zero)?,
            windows:Windows::new(cuda,local,&zero,&zero,&zero)?,selection,
            merge_windows:Windows::new(cuda,2048,&zero,&vec![2048;rows],&zero)?,
            merge:Selection::new(cuda,topk,rows,2048,2048,512)?,
            merge_scores:cuda.allocate(rows*2048)?,ids:cuda.allocate(rows*2048)?,
            send:cuda.allocate(rows*words)?,receive:cuda.allocate(4*rows*words)?,valid,
            dirty:Cell::new(true),binding:Cell::new((0,0,usize::MAX)),publisher})
    }
    pub(crate) fn invalidate(&self){self.dirty.set(true);}
    pub(crate) fn prepare(&self,stream:&Stream)->Result<()>{
        ensure!(self.cuda.owns_stream(stream),"key-shard prepare context");
        if self.dirty.get(){
            unsafe{self.cuda.zero_device(self.valid.ptr(),self.valid.bytes(),stream)?;}
            self.dirty.set(false);
        }Ok(())
    }
    // The merge selector always sees2048 candidates, even for a short causal
    // prefix. Its remapped IDs must therefore be sorted on the short path too.
    pub(crate) fn selection_lengths(&self)->&Buffer<i32>{self.merge_windows.lengths()}
    pub(crate) fn block_ids(&self)->Option<&Buffer<i32>>{self.publisher.as_ref().map(|p|p.selected())}
    fn launch(&self,name:&str,grid:usize,threads:usize,arguments:&mut[*mut c_void],stream:&Stream)->Result<()>{
        unsafe{self.module.function(name)?.launch([grid as u32,1,1],[threads as u32,1,1],0,stream,arguments)}
    }
    pub(crate) fn run(&mut self,cache:&KeyCache,slots:&KeySlots,ends:&Buffer<i32>,
        windows:&Windows,output:&Selection,comm:&Roce,stream:&Stream)->Result<()>{
        ensure!(!self.dirty.get()&&comm.rank()<4&&ends.len()==self.rows&&windows.rows()==self.rows
            &&slots.values().len()==self.columns&&slots.capacity()==cache.capacity()&&slots.contiguous()
            &&self.cuda.owns_buffer(cache.data())&&self.cuda.owns_buffer(slots.values())
            &&self.cuda.owns_buffer(ends)&&self.cuda.owns_stream(stream),"key-shard cache/metadata ownership");
        let binding=(cache.data().ptr(),slots.values().ptr(),comm.rank());
        ensure!(self.binding.get()==(0,0,usize::MAX)||self.binding.get()==binding,"key-shard immutable storage/rank changed");
        self.binding.set(binding);cache.bind(stream)?;self.keys.bind(stream)?;
        let(mut end,mut local_end,mut length,mut rank,mut rows)=(ends.ptr(),self.ranges.key_ends_mut().ptr(),
            self.windows.lengths_mut().ptr(),comm.rank()as i32,self.rows as i32);
        self.launch("ca_key_shard_ranges",1,32,&mut args![end,local_end,length,rank,rows],stream)?;
        let(mut source,mut slot,mut payload,mut scales,mut valid,mut columns)=(cache.data().ptr(),slots.values().ptr(),
            self.keys.payload.ptr(),self.keys.scales.ptr(),self.valid.ptr(),self.columns as i32);
        self.launch("ca_key_shard_gather",192,128,&mut args![source,slot,payload,scales,end,valid,rank,columns,rows],stream)?;
        self.launch("ca_key_shard_advance",1,32,&mut args![end,valid,columns,rows],stream)?;
        self.keys.ready.set(true);self.scorer.run(&self.ranges,stream)?;
        if let Some(p)=&mut self.publisher {
            p.score(self.scorer.output(),self.ranges.key_ends(),ends,comm.rank(),stream)?;
        }
        self.selection.run(self.scorer.output_mut(),&self.windows,stream)?;
        let(mut scores,mut selected,mut packed,mut stride)=(self.scorer.output().ptr(),self.selection.indices().ptr(),
            self.send.ptr(),self.scorer.stride()as i32);
        if let Some(p)=&self.publisher {
            p.pack(self.scorer.output(),self.selection.indices(),self.ranges.key_ends(),&self.send,comm.rank(),stream)?;
        }else{self.launch("ca_key_shard_pack",self.rows,512,&mut args![scores,selected,local_end,packed,rank,stride],stream)?;}
        // One flat byte-preserving row gives rank-major ownership. It must not
        // be cropped to the target's live query rows by the transport layer.
        unsafe{comm.gather_f32(&self.send,&mut self.receive,self.send.len(),stream)?;}
        let(mut received,mut merged,mut ids)=(self.receive.ptr(),self.merge_scores.ptr(),self.ids.ptr());
        if let Some(p)=&mut self.publisher {
            p.unpack_and_select(&self.receive,&self.merge_scores,&self.ids,stream)?;
        }else{self.launch("ca_key_shard_unpack",self.rows*8,256,&mut args![received,merged,ids,rows],stream)?;}
        self.merge.run(&mut self.merge_scores,&self.merge_windows,stream)?;
        selected=self.merge.indices().ptr();
        output.write_remapped_from(&self.merge,windows,stream,|mut destination|{
            self.launch("ca_key_shard_remap",self.rows*2,256,&mut args![selected,ids,destination,rows],stream)
        })
    }
}
