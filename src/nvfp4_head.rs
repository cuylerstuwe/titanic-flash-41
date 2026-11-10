//! Finite, private-only Marlin NVFP4 DSpark head. The canonical target is untouched.
//! CUDA driver calls own all buffers and stream dependencies; no foreign engine.
use crate::cuda::{Buffer,Cuda,Module,Stream};
use anyhow::{ensure,Result};
use half::bf16;
use serde::Deserialize;
use sha2::{Digest,Sha256};
use std::{collections::BTreeMap,ffi::c_void,path::Path};

const N:usize=32320;
const K:usize=5120;
const M:usize=5;
const SM:usize=48;
const SHARED:u32=101376;
#[derive(Deserialize)]
struct Entry {name:String,symbol:String,parameters:Vec<String>}
#[derive(Deserialize)]
struct Manifest {artifact:String,artifact_sha256:String,functions:Vec<Entry>}
pub(crate) struct Head {
    cuda:Cuda,
    module:Module,
    entries:BTreeMap<String,Entry>,
    weights:Buffer<u32>,
    scales:Buffer<u8>,
    global:Buffer<f32>,
    temporary:Buffer<f32>,
    output:Buffer<bf16>,
    locks:Buffer<u32>,
    stream:usize,
}
impl Head {
    pub(crate) fn new(cuda:&Cuda,canonical:u64,root:&Path,stream:&Stream)->Result<Self> {
        ensure!(std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_NVFP4_SCREEN").as_deref()==Ok("1")
            &&cuda.owns_stream(stream)&&canonical!=0&&canonical%256==0,
            "private NVFP4 head requires isolated sampled screen");
        cuda.qualify_gb10()?;
        // Includes the one-shot transpose/repack buffer and every persistent
        // projection buffer. Existing driver allocation floors remain active.
        let bytes=N*K+N*K/16+SM*16*256*4+M*N*2+SM*4+16;
        ensure!(cuda.admission_memory()?.admitted_available>=(crate::memory::floor_bytes()+512*1024usize.pow(2))+bytes,
            "private NVFP4 scratch would consume the physical reserve");
        let raw=std::fs::read(root.join("manifest.json"))?;
        ensure!(format!("{:x}",Sha256::digest(&raw))==
            "dce4e236fd276accae1f4225f71bc0199c449474ed9770719dca273fe263ed63",
            "qualified private NVFP4 manifest changed");
        let manifest:Manifest=serde_json::from_slice(&raw)?;
        ensure!(manifest.artifact=="head.cubin","private NVFP4 artifact path");
        let artifact=std::fs::read(root.join(&manifest.artifact))?;
        ensure!(format!("{:x}",Sha256::digest(&artifact))==manifest.artifact_sha256,
            "private NVFP4 CUDA artifact changed");
        let entries:BTreeMap<_,_>=manifest.functions.into_iter().map(|e|(e.name.clone(),e)).collect();
        ensure!(entries.len()==5,"private NVFP4 operator count");
        for(name,types)in [
            ("amax",vec!["u64","u64"]),
            ("quantize",vec!["u64";6]),
            ("publish",vec!["u64","u64"]),
            ("repack",vec!["u64","u64","u32","u32"]),
            ("projection",vec!["u64","u64","u64","u64","u64","u64","u64","u64","u64",
                "u32","u32","u32","u32","u64","u8","u8","u8","u32"])
        ] {ensure!(entries.get(name).is_some_and(|e|e.parameters==types),"private NVFP4 driver ABI {name}");}
        let module=cuda.module(&artifact)?;
        for name in ["projection","repack"] {
            module.function(&entries[name].symbol)?.opt_in_shared_memory(SHARED)?;
        }
        let weights=cuda.allocate::<u32>(N*K/8)?;
        let scales=cuda.allocate::<u8>(N*K/16)?;
        let global=cuda.allocate::<f32>(1)?;
        let temporary=cuda.allocate::<f32>(SM*16*256)?;
        let output=cuda.allocate::<bf16>(M*N)?;
        let mut locks=cuda.allocate::<u32>(SM)?;locks.zero()?;
        let mut maximum=cuda.allocate::<f32>(1)?;maximum.zero()?;
        let mut error=cuda.allocate::<u32>(1)?;error.zero()?;
        let packed=cuda.allocate::<u32>(N*K/8)?;
        let result=Self{cuda:cuda.clone(),module,entries,weights,scales,global,temporary,output,locks,
            stream:stream.raw()as usize};
        stream.join_default()?;
        unsafe {
            result.launch("amax",&[canonical,maximum.ptr()],192,256,0,stream)?;
            result.launch("quantize",&[canonical,maximum.ptr(),packed.ptr(),result.scales.ptr(),
                result.global.ptr(),error.ptr()],(N*K/256)as u32,256,0,stream)?;
            result.launch("repack",&[packed.ptr(),result.weights.ptr(),K as u64,N as u64],
                SM as u32,256,SHARED,stream)?;
        }
        stream.synchronize()?;
        ensure!(error.download()?==[0],"NVFP4 positive-scale encoding range");
        eprintln!("DSpark private NVFP4 Marlin: {} encoded bytes, target unchanged",N*K/2+N*K/16+4);
        Ok(result)
    }
    /// Values obey the pinned little-endian SM121 pointer/scalar ABI; the
    /// driver reads each parameter's declared byte width from its address.
    unsafe fn launch(&self,name:&str,values:&[u64],grid:u32,threads:u32,shared:u32,stream:&Stream)->Result<()> {
        let e=&self.entries[name];ensure!(e.parameters.len()==values.len(),"NVFP4 launch arity");
        let mut words=values.to_vec();
        let mut args:Vec<*mut c_void>=words.iter_mut().map(|x|(x as *mut u64).cast()).collect();
        unsafe{self.module.function(&e.symbol)?.launch([grid,1,1],[threads,1,1],shared,stream,&mut args)}
    }
    pub(crate) fn run(&self,input:&Buffer<bf16>,output:&Buffer<f32>,stream:&Stream)->Result<()> {
        ensure!(self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize
            &&self.cuda.owns_buffer(input)&&self.cuda.owns_buffer(output)
            &&input.len()==M*K&&output.len()==M*N,"NVFP4 private M5 context/stream/extents");
        unsafe {
            self.launch("projection",&[input.ptr(),self.weights.ptr(),self.output.ptr(),
                self.temporary.ptr(),0,0,self.scales.ptr(),self.global.ptr(),0,
                M as u64,N as u64,K as u64,K as u64,self.locks.ptr(),0,0,1,SHARED as u64],
                SM as u32,128,SHARED,stream)?;
            self.launch("publish",&[self.output.ptr(),output.ptr()],(M*N).div_ceil(256)as u32,256,0,stream)
        }
    }
}

