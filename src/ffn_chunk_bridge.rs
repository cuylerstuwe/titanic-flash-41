//! Isolated graph partitioning and canonical BF16 packing for an FFN component.
use crate::cuda::{Buffer,Cuda,Stream};
use anyhow::{Result,ensure};
use half::bf16;
use libloading::Library;
use sha2::{Digest,Sha256};
use std::{ffi::c_void,path::Path,ptr,rc::Rc};
type Handle=*mut c_void;
pub(crate) struct Bridge {
    _lib:Library,cuda:Cuda,
    begin:unsafe extern "C" fn(Handle)->i32,
    end:unsafe extern "C" fn(Handle,*mut Handle)->i32,
    source_close:unsafe extern "C" fn(Handle),
    pack:unsafe extern "C" fn(Handle,Handle,Handle,i32,Handle)->i32,
    join:unsafe extern "C" fn(Handle,Handle,Handle,Handle)->i32,
    create:unsafe extern "C" fn(Handle,Handle,Handle,Handle,i32,*mut Handle,*mut u32)->i32,
    launch:unsafe extern "C" fn(Handle,Handle)->i32,
    close:unsafe extern "C" fn(Handle),
}
impl Bridge {
    pub(crate) fn open(cuda:&Cuda,path:&Path)->Result<Rc<Self>>{
        ensure!(std::env::var("CANADA_SCRATCH_FFN_CHUNK_SCREEN").as_deref()==Ok("1"),"isolated FFN chunk admission");
        let proof:serde_json::Value=serde_json::from_slice(&std::fs::read(path.with_extension("json"))?)?;
        ensure!(proof["architecture"]==121
            &&proof["artifact_sha256"]==format!("{:x}",Sha256::digest(std::fs::read(path)?))
            &&proof["source_sha256"]==format!("{:x}",Sha256::digest(include_bytes!("../kernels/ffn_chunk_probe.cu"))),"FFN chunk bridge provenance");
        let lib=unsafe{Library::new(path)}?;
        macro_rules! sym{($name:literal)=>{*unsafe{lib.get(concat!($name,"\0").as_bytes())}?};}
        Ok(Rc::new(Self{begin:sym!("ca_ffn_chunk_capture_begin"),end:sym!("ca_ffn_chunk_capture_end"),
            source_close:sym!("ca_ffn_chunk_capture_close"),
            pack:sym!("ca_ffn_chunk_pack"),join:sym!("ca_ffn_chunk_join"),create:sym!("ca_ffn_chunk_create"),
            launch:sym!("ca_ffn_chunk_launch"),close:sym!("ca_ffn_chunk_close"),_lib:lib,cuda:cuda.clone()}))
    }
    pub(crate) fn capture(self:&Rc<Self>,stream:&Stream,f:impl FnOnce()->Result<()>)->Result<Source>{
        ensure!(self.cuda.owns_stream(stream),"FFN source stream context");
        ensure!(unsafe{(self.begin)(stream.raw())}==0,"FFN source capture begin");
        let body=f();let mut raw=ptr::null_mut();let ended=unsafe{(self.end)(stream.raw(),&mut raw)};
        if body.is_err()||ended!=0||raw.is_null(){
            if !raw.is_null(){unsafe{(self.source_close)(raw);}}
            body?;anyhow::bail!("FFN source capture end: {ended}");
        }
        Ok(Source{bridge:self.clone(),raw})
    }
    /// Caller keeps the captured workspace/module owner alive and exclusively
    /// accessed. The workspace's checked contract supplies its route pointer.
    pub(crate) unsafe fn pack(&self,workspace:&crate::moe::Workspace,shared:&Buffer<bf16>,out:&Buffer<bf16>,half:i32,stream:&Stream)->Result<()> {
        ensure!(self.cuda.owns_stream(stream)
            &&self.cuda.owns_buffer(shared)&&self.cuda.owns_buffer(out)
            &&shared.len()==6*5120&&out.len()==6*2560&&(0..=1).contains(&half),"FFN pack context/geometry");
        let routes=workspace.chunk_route_region(stream)?;
        let code=unsafe{(self.pack)(routes as Handle,shared.ptr()as Handle,out.ptr()as Handle,half,stream.raw())};
        ensure!(code==0,"FFN half pack: {code}");Ok(())
    }
    pub(crate) fn join(&self,low:&Buffer<bf16>,high:&Buffer<bf16>,out:&Buffer<bf16>,stream:&Stream)->Result<()> {
        ensure!(self.cuda.owns_stream(stream)&&[low,high,out].into_iter().all(|b|self.cuda.owns_buffer(b))
            &&low.len()==6*2560&&high.len()==low.len()&&out.len()==6*5120,"FFN join context/geometry");
        let code=unsafe{(self.join)(low.ptr()as Handle,high.ptr()as Handle,out.ptr()as Handle,stream.raw())};
        ensure!(code==0,"FFN half join: {code}");Ok(())
    }
    /// All source captures' storage and loaded kernel modules outlive replays.
    pub(crate) unsafe fn graph(self:&Rc<Self>,source:&Source,low:&Source,high:&Source,finish:&Source,
        mode:i32,stream:&Stream)->Result<ChunkGraph>{
        ensure!([source,low,high,finish].into_iter().all(|s|Rc::ptr_eq(self,&s.bridge))
            &&self.cuda.owns_stream(stream)&&(0..=2).contains(&mode),"FFN graph source context");
        let mut raw=ptr::null_mut();let mut counts=[0u32;4];
        let code=unsafe{(self.create)(source.raw,low.raw,high.raw,finish.raw,mode,&mut raw,counts.as_mut_ptr())};
        ensure!(code==0&&!raw.is_null()&&counts[1..]==[2,1,1],"FFN partition: {code}, {counts:?}");
        Ok(ChunkGraph{bridge:self.clone(),raw,stream:stream.raw()as usize,counts})
    }
}
pub(crate) struct Source{bridge:Rc<Bridge>,raw:Handle}
impl Drop for Source{fn drop(&mut self){unsafe{(self.bridge.source_close)(self.raw);}}}
pub(crate) struct ChunkGraph{bridge:Rc<Bridge>,raw:Handle,stream:usize,pub(crate) counts:[u32;4]}
impl ChunkGraph{
    pub(crate) unsafe fn launch(&self,stream:&Stream)->Result<()>{
        ensure!(self.bridge.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,"FFN graph replay stream");
        let code=unsafe{(self.bridge.launch)(self.raw,stream.raw())};ensure!(code==0,"FFN graph replay: {code}");Ok(())
    }
}
impl Drop for ChunkGraph{fn drop(&mut self){unsafe{(self.bridge.close)(self.raw);}}}
