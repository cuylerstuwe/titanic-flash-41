//! Memoize the immutable canonical Markov projection by preceding token.
//! This is derived weight data, not conversation state. One graph-bank owner
//! and one ordered stream own it for their entire lifetime.
use crate::{cuda::{Buffer,Cuda,Stream},triton_ops::Operator};
use anyhow::{ensure,Result};
use std::path::Path;
const N:usize=32320;
const SLOTS:usize=256;
const BLOCKS:usize=N/64;
pub(crate) struct CorrectionCache {
    cuda:Cuda,
    stream:usize,
    op:Operator,
    values:Buffer<f32>,
    tags:Buffer<i32>,
    hit:Buffer<i32>,
    counters:Buffer<u64>,
}
impl CorrectionCache {
    pub(crate) fn new(cuda:&Cuda,kernels:&Path,stream:&Stream)->Result<Self>{
        ensure!(cuda.owns_stream(stream),"Markov cache stream context");
        let root=kernels.join("markov-correction-cache-v1");
        let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(manifest["entries"][0]["constexprs"]==serde_json::json!({"N":N,"SLOTS":SLOTS,"BN":64}),
            "Markov cache fixed geometry");
        let op=Operator::load(cuda,&root,"markov_cache_m1")?;
        ensure!(op.entry.arguments.iter().map(|a|(a.name.as_str(),a.kind.as_str())).collect::<Vec<_>>()
            ==[("X","*bf16"),("W","*bf16"),("Base","*fp32"),("Token","*i32"),("Cache","*fp32"),
                ("Tags","*i32"),("Y","*fp32"),("Hit","*i32"),("Counters","*u64")]
            &&op.entry.grid.rows_div==64&&op.entry.grid.x_mul==1&&op.entry.grid.x_add==0
            &&op.entry.grid.y==1&&op.entry.grid.z==1&&op.entry.metadata.num_warps==4,
            "Markov cache ABI/grid");
        let bytes=SLOTS*(N+BLOCKS)*4+20;
        ensure!(cuda.admission_memory()?.admitted_available>bytes+crate::memory::floor_bytes(),"Markov cache reserve");
        let values=cuda.allocate(SLOTS*N)?;let mut tags=cuda.allocate(SLOTS*BLOCKS)?;
        tags.upload(&vec![-1;SLOTS*BLOCKS])?;
        let hit=cuda.allocate(1)?;let mut counters=cuda.allocate(2)?;counters.upload(&[0,0])?;
        stream.join_default()?;
        Ok(Self{cuda:cuda.clone(),stream:stream.raw()as usize,op,values,tags,hit,counters})
    }
    /// Pointers are the graph-bank's checked canonical latent/weight, one
    /// row of its base logits, its preceding-token scalar and local output.
    pub(crate) unsafe fn enqueue(&self,x:u64,weight:u64,base:u64,token:u64,output:u64,stream:&Stream)->Result<()>{
        ensure!(self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,"Markov cache stream changed");
        unsafe{self.op.launch(N as u32,&[x,weight,base,token,self.values.ptr(),self.tags.ptr(),output,
            self.hit.ptr(),self.counters.ptr()],stream)}
    }
    pub(crate) fn reset_counters(&self,stream:&Stream)->Result<()>{
        ensure!(self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,"Markov cache counter stream");
        unsafe{self.cuda.zero_device(self.counters.ptr(),self.counters.bytes(),stream)}
    }
    pub(crate) fn counters(&self,stream:&Stream)->Result<[u64;2]>{
        ensure!(self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,"Markov cache counter stream");
        stream.synchronize()?;let values=self.counters.download()?;
        ensure!(values[0]<=values[1],"Markov cache hit counter invalid");
        Ok(values.try_into().unwrap())
    }
}
