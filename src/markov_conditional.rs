//! Isolated CUDA conditional-graph bridge. Rust owns capture order and storage.
use crate::cuda::{Buffer,Cuda,Stream};
use anyhow::{Result,ensure};
use libloading::Library;
use sha2::{Digest,Sha256};
use std::{ffi::c_void,path::Path,ptr,rc::Rc};
type Handle=*mut c_void;
pub(crate) struct Bridge {
    _lib:Library,
    cuda:Cuda,
    begin:unsafe extern "C" fn(Handle)->i32,
    end:unsafe extern "C" fn(Handle,*mut Handle)->i32,
    source_close:unsafe extern "C" fn(Handle),
    create:unsafe extern "C" fn(Handle,Handle,*const f32,*mut u32,i32,*mut Handle)->i32,
    create_readout:unsafe extern "C" fn(Handle,Handle,*const f32,*mut u32,f32,*const u32,*mut u32,*mut Handle)->i32,
    launch:unsafe extern "C" fn(Handle,Handle)->i32,
    close:unsafe extern "C" fn(Handle),
}
impl Bridge {
    pub(crate) fn open(cuda:&Cuda,path:&Path)->Result<Rc<Self>>{
        ensure!(std::env::var("CANADA_SCRATCH_MARKOV_CONDITIONAL_SCREEN").as_deref()==Ok("1")
            ||std::env::var("CANADA_SCRATCH_PREFIX_DRAFT_STOP_GPU").as_deref()==Ok("1"),
            "conditional graph requires isolated admission");
        let proof:serde_json::Value=serde_json::from_slice(&std::fs::read(path.with_extension("json"))?)?;
        ensure!(proof["architecture"]==121
            &&proof["artifact_sha256"]==format!("{:x}",Sha256::digest(std::fs::read(path)?))
            &&proof["source_sha256"]==format!("{:x}",Sha256::digest(include_bytes!("../kernels/markov_conditional_probe.cu"))),
            "conditional bridge provenance");
        let lib=unsafe{Library::new(path)}?;
        macro_rules! sym{($name:literal)=>{*unsafe{lib.get(concat!($name,"\0").as_bytes())}?};}
        Ok(Rc::new(Self{begin:sym!("ca_markov_capture_begin"),end:sym!("ca_markov_capture_end"),
            source_close:sym!("ca_markov_capture_close"),create:sym!("ca_markov_cond_create"),
            create_readout:sym!("ca_markov_cond_create_readout"),
            launch:sym!("ca_markov_cond_launch"),close:sym!("ca_markov_cond_close"),_lib:lib,cuda:cuda.clone()}))
    }
    pub(crate) fn capture(self:&Rc<Self>,stream:&Stream,f:impl FnOnce()->Result<()>)->Result<Source>{
        ensure!(self.cuda.owns_stream(stream),"conditional source stream context");
        ensure!(unsafe{(self.begin)(stream.raw())}==0,"conditional source capture begin");
        let body=f();let mut raw=ptr::null_mut();
        let ended=unsafe{(self.end)(stream.raw(),&mut raw)};
        if body.is_err()||ended!=0||raw.is_null(){
            if !raw.is_null(){unsafe{(self.source_close)(raw);}}
            body?;anyhow::bail!("conditional source capture end: {ended}");
        }
        Ok(Source{bridge:self.clone(),raw})
    }
    /// Captured pointers and consensus/result storage must outlive every replay.
    pub(crate) unsafe fn graph(self:&Rc<Self>,prefix:&Source,tail:&Source,
        consensus:&Buffer<f32>,result:&mut Buffer<u32>,force:i32,stream:&Stream)->Result<Conditional>{
        ensure!(Rc::ptr_eq(self,&prefix.bridge)&&Rc::ptr_eq(self,&tail.bridge)
            &&self.cuda.owns_stream(stream)&&self.cuda.owns_buffer(consensus)&&self.cuda.owns_buffer(result)
            &&consensus.len()>=3&&result.len()==4&&(0..=2).contains(&force),"conditional graph storage/context");
        let mut raw=ptr::null_mut();
        let code=unsafe{(self.create)(prefix.raw,tail.raw,consensus.ptr()as *const f32,
            result.ptr()as *mut u32,force,&mut raw)};
        ensure!(code==0&&!raw.is_null(),"conditional graph create: {code}");
        Ok(Conditional{bridge:self.clone(),raw,stream:stream.raw()as usize})
    }
    /// One final device readout contains both produced IDs and the GPU's
    /// stopping decision. All source and readout storage outlives replay.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn graph_readout(self:&Rc<Self>,prefix:&Source,tail:&Source,
        consensus:&Buffer<f32>,result:&mut Buffer<u32>,threshold:f32,
        readout:&Buffer<u32>,combined:&mut Buffer<u32>,stream:&Stream)->Result<Conditional>{
        ensure!(Rc::ptr_eq(self,&prefix.bridge)&&Rc::ptr_eq(self,&tail.bridge)
            &&self.cuda.owns_stream(stream)&&self.cuda.owns_buffer(consensus)
            &&[&*result,readout,&*combined].into_iter().all(|b|self.cuda.owns_buffer(b))
            &&consensus.len()>=3&&result.len()==4&&readout.len()==32&&combined.len()==36
            &&threshold.is_finite()&&(0. ..=1.).contains(&threshold),"conditional readout context/storage");
        let mut raw=ptr::null_mut();
        let code=unsafe{(self.create_readout)(prefix.raw,tail.raw,consensus.ptr()as *const f32,
            result.ptr()as *mut u32,threshold,readout.ptr()as *const u32,combined.ptr()as *mut u32,&mut raw)};
        ensure!(code==0&&!raw.is_null(),"conditional readout graph create: {code}");
        Ok(Conditional{bridge:self.clone(),raw,stream:stream.raw()as usize})
    }
}
pub(crate) struct Source{bridge:Rc<Bridge>,raw:Handle}
impl Drop for Source{fn drop(&mut self){unsafe{(self.bridge.source_close)(self.raw);}}}
pub(crate) struct Conditional{bridge:Rc<Bridge>,raw:Handle,stream:usize}
impl Conditional{
    /// All captured storage remains alive and exclusively accessed by this graph.
    pub(crate) unsafe fn launch(&self,stream:&Stream)->Result<()>{
        ensure!(self.bridge.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,"conditional replay context");
        let code=unsafe{(self.bridge.launch)(self.raw,stream.raw())};
        ensure!(code==0,"conditional replay: {code}");Ok(())
    }
}
impl Drop for Conditional{fn drop(&mut self){unsafe{(self.bridge.close)(self.raw);}}}
