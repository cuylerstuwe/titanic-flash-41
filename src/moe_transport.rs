//! Rust-owned, one-stream packed MoE transport. The pinned native dependency
//! performs codec kernels, verbs and CUDA graph control; it owns no model state.
use crate::cuda::{Buffer,Cuda,Stream};
use anyhow::{ensure,Result};
use half::bf16;
use libloading::Library;
use sha2::{Digest,Sha256};
use std::{cell::Cell,ffi::{c_char,c_void,CString},path::Path};
type Handle=*mut c_void;
pub(crate) struct Pipeline {
    cuda:Cuda,
    _library:Library,
    transport:Handle,
    codec:Handle,
    blob_bytes:usize,
    blob:unsafe extern "C" fn(Handle,Handle)->i32,
    connect:unsafe extern "C" fn(Handle,*const c_void)->i32,
    begin:unsafe extern "C" fn(Handle,*const c_void,Handle)->i32,
    finish:unsafe extern "C" fn(Handle,Handle,Handle,Handle,Handle,Handle)->i32,
    health:unsafe extern "C" fn(Handle)->i32,
    destroy_transport:unsafe extern "C" fn(Handle),
    destroy_codec:unsafe extern "C" fn(Handle),
    connected:Cell<bool>,
    stream:Cell<Option<usize>>,
    poisoned:Cell<bool>,
}
impl Pipeline {
    pub(crate) fn new(cuda:&Cuda,root:&Path,rank:usize)->Result<Self> {
        ensure!(rank<4,"pipeline rank");
        const REGISTERED_BYTES:usize=128*1024*1024; // includes allocation rounding and controls
        ensure!(cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+REGISTERED_BYTES,
            "MoE pipeline must preserve the 8GiB UMA reserve");
        for(name,expected)in [
            ("bridge.so","10c49a2b5b55849e6a97370d3ad6a8df4100ca5fad3732642c74f311ae463bdc"),
            ("proxy.so","d7a2206e70cd4847b921abd4144c78fbc7b530f9a675bf73049394de7a50733b"),
            ("registered-codec.so","fd7f93fabbf7f8b8ae5e4b0896a12279c7db6fe94416e0e7ec13df9c0d1f99c6")
        ] {
            ensure!(format!("{:x}",Sha256::digest(std::fs::read(root.join(name))?))==expected,
                "pinned MoE pipeline dependency hash: {name}");
        }
        let library=unsafe{Library::new(root.join("bridge.so"))}?;
        macro_rules! sym {($name:literal)=>{*unsafe{library.get(concat!($name,"\0").as_bytes())}?};}
        let create:unsafe extern "C" fn(i32,usize,*const c_char)->Handle=sym!("pg_create");
        let codec_create:unsafe extern "C" fn(Handle,*const c_char)->Handle=sym!("pp_create");
        let bytes:unsafe extern "C" fn(Handle)->usize=sym!("pg_blob_bytes");
        let destroy_transport:unsafe extern "C" fn(Handle)=sym!("pg_destroy");
        let proxy=CString::new(root.join("proxy.so").to_str().ok_or_else(||anyhow::anyhow!("pipeline path encoding"))?)?;
        let codec_path=CString::new(root.join("registered-codec.so").to_str().unwrap())?;
        let transport=unsafe{create(rank as i32,2048*5920,proxy.as_ptr())};
        ensure!(!transport.is_null(),"create MoE registered transport");
        let codec=unsafe{codec_create(transport,codec_path.as_ptr())};
        if codec.is_null(){unsafe{destroy_transport(transport)};anyhow::bail!("create MoE pipeline codec");}
        let blob_bytes=unsafe{bytes(transport)};
        ensure!((1..=65536).contains(&blob_bytes),"pipeline endpoint size");
        Ok(Self{cuda:cuda.clone(),transport,codec,blob_bytes,
            blob:sym!("pg_blob"),connect:sym!("pg_connect"),begin:sym!("pp_begin"),finish:sym!("pp_finish"),health:sym!("pg_health"),
            destroy_transport,destroy_codec:sym!("pp_destroy"),_library:library,
            connected:Cell::new(false),stream:Cell::new(None),poisoned:Cell::new(false)})
    }
    fn checked(&self,status:i32)->Result<()> {
        if status!=0{self.poisoned.set(true);}
        ensure!(!self.poisoned.get(),"MoE pipeline failed/poisoned: {status}");Ok(())
    }
    pub(crate) fn blob(&self)->Result<Vec<u8>> {
        let mut bytes=vec![0;self.blob_bytes];
        self.checked(unsafe{(self.blob)(self.transport,bytes.as_mut_ptr().cast())})?;Ok(bytes)
    }
    pub(crate) fn connect(&self,all:&[u8])->Result<()> {
        ensure!(!self.connected.get()&&all.len()==4*self.blob_bytes,"pipeline peer exchange extent/state");
        self.checked(unsafe{(self.connect)(self.transport,all.as_ptr().cast())})?;
        self.connected.set(true);Ok(())
    }
    pub(crate) fn health(&self)->Result<()> {self.checked(unsafe{(self.health)(self.transport)})}
    pub(crate) fn enqueue(&self,input:&Buffer<bf16>,packed:&Buffer<u8>,q:&Buffer<u8>,
        sr:&Buffer<u8>,sm:&Buffer<u8>,stream:&Stream,metadata:impl FnOnce()->Result<()>)->Result<()> {
        self.health()?;
        ensure!(self.connected.get()&&self.cuda.owns_stream(stream)&&self.cuda.owns_buffer(input)
            &&[packed,q,sr,sm].into_iter().all(|b|self.cuda.owns_buffer(b))
            &&input.len()==2048*5120&&packed.len()==8192*5920&&q.len()==8192*5120
            &&sr.len()==8192*160&&sm.len()==8192*160
            &&self.stream.get().is_none_or(|s|s==stream.raw()as usize),"MoE pipeline operand/stream contract");
        self.stream.set(Some(stream.raw()as usize));
        self.checked(unsafe{(self.begin)(self.codec,input.ptr()as *const c_void,stream.raw())})?;
        if let Err(error)=metadata(){self.poisoned.set(true);return Err(error);}
        self.checked(unsafe{(self.finish)(self.codec,packed.ptr()as Handle,q.ptr()as Handle,
            sr.ptr()as Handle,sm.ptr()as Handle,stream.raw())})
    }
}
impl Drop for Pipeline {
    fn drop(&mut self) {
        // Roce's owner must drain/destroy model graphs before communicator
        // teardown, the same lifetime contract as its existing transport.
        unsafe{(self.destroy_codec)(self.codec);(self.destroy_transport)(self.transport);}
    }
}
