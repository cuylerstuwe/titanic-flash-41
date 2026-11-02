//! Optional source20 companion to TP4 key partition. Local canonical block
//! maxima/top2048 travel alongside top512 keys in one exact byte gather.
use crate::{cuda::{Buffer,Cuda,Module,Stream},topk::{self,Selection,Windows}};
use anyhow::{ensure,Result};
use std::{ffi::c_void,rc::Rc};
macro_rules! args {($($v:ident),* $(,)?)=>{[$(std::ptr::addr_of_mut!($v).cast::<c_void>()),*]};}
pub(crate) const PACKED_WORDS_PER_ROW:usize=2560*2;
pub(crate) struct Publisher {
    module:Module,columns:usize,rows:usize,scores:Buffer<f32>,windows:Windows,selection:Selection,
    merged:Buffer<f32>,candidates:Buffer<i32>,merge_windows:Windows,merge:Selection,
    output:Buffer<i32>,
}
impl Publisher {
    pub(crate) fn new(cuda:&Cuda,core:&[u8],native:Rc<topk::Native>,local_columns:usize,rows:usize)->Result<Self>{
        ensure!(local_columns==262144&&[6,16].contains(&rows),"key publisher local geometry");
        let blocks=local_columns/8;let zero=vec![0;rows];
        Ok(Self{module:cuda.module(core)?,columns:local_columns,rows,
            scores:cuda.allocate(rows*blocks)?,
            windows:Windows::new(cuda,blocks,&zero,&zero,&zero)?,
            selection:Selection::new(cuda,native.clone(),rows,blocks,blocks,2048)?,
            merged:cuda.allocate(rows*8192)?,candidates:cuda.allocate(rows*8192)?,
            merge_windows:Windows::new(cuda,8192,&zero,&vec![8192;rows],&zero)?,
            merge:Selection::new(cuda,native,rows,8192,8192,2048)?,output:cuda.allocate(rows*2048)?})
    }
    fn launch(&self,name:&str,grid:[u32;3],arguments:&mut[*mut c_void],stream:&Stream)->Result<()>{
        unsafe{self.module.function(name)?.launch(grid,[256,1,1],0,stream,arguments)}
    }
    pub(crate) fn score(&mut self,scores:&Buffer<f32>,ends:&Buffer<i32>,global_ends:&Buffer<i32>,rank:usize,stream:&Stream)->Result<()>{
        ensure!(rank<4&&scores.len()>=self.rows*self.columns&&ends.len()==self.rows&&global_ends.len()==self.rows,"key publisher score geometry");
        let(mut rank,mut stride,mut scores,mut ends,mut global,mut blocks,mut lengths)=
            (rank as i32,self.columns as i32,scores.ptr(),ends.ptr(),global_ends.ptr(),self.scores.ptr(),self.windows.lengths_mut().ptr());
        self.launch("ca_key_publisher_block_scores",[16,self.rows as u32,1],&mut args![rank,stride,scores,ends,global,blocks,lengths],stream)?;
        self.selection.run(&mut self.scores,&self.windows,stream)
    }
    pub(crate) fn pack(&self,scores:&Buffer<f32>,ids:&Buffer<i32>,ends:&Buffer<i32>,payload:&Buffer<f32>,rank:usize,stream:&Stream)->Result<()>{
        ensure!(rank<4&&ids.len()==self.rows*512&&payload.len()==self.rows*PACKED_WORDS_PER_ROW,"key publisher packet geometry");
        let(mut rank,mut stride,mut scores,mut ids,mut ends,mut blocks,mut selected,mut block_ends,mut payload)=
            (rank as i32,self.columns as i32,scores.ptr(),ids.ptr(),ends.ptr(),self.scores.ptr(),
             self.selection.indices().ptr(),self.windows.lengths().ptr(),payload.ptr());
        self.launch("ca_key_publisher_pack_selected",[self.rows as u32,1,1],
            &mut args![rank,stride,scores,ids,ends,blocks,selected,block_ends,payload],stream)
    }
    pub(crate) fn unpack_and_select(&mut self,received:&Buffer<f32>,key_scores:&Buffer<f32>,key_ids:&Buffer<i32>,stream:&Stream)->Result<()>{
        ensure!(received.len()==4*self.rows*PACKED_WORDS_PER_ROW&&key_scores.len()==self.rows*2048&&key_ids.len()==self.rows*2048,"key publisher merge geometry");
        let(mut received,mut scores,mut ids,mut blocks,mut block_ids,mut rows)=
            (received.ptr(),key_scores.ptr(),key_ids.ptr(),self.merged.ptr(),self.candidates.ptr(),self.rows as i32);
        self.launch("ca_key_publisher_unpack_candidates",[(self.rows*10240/256)as u32,1,1],
            &mut args![received,scores,ids,blocks,block_ids,rows],stream)?;
        self.merge.run(&mut self.merged,&self.merge_windows,stream)?;
        let(mut chosen,mut candidates,mut output,mut topk)=
            (self.merge.indices().ptr(),self.candidates.ptr(),self.output.ptr(),2048i32);
        self.launch("ca_key_publisher_remap_selected",[(self.rows*2048/256)as u32,1,1],
            &mut args![chosen,candidates,output,rows,topk],stream)
    }
    pub(crate) fn selected(&self)->&Buffer<i32>{&self.output}
}
