//! Compact categorical transport plus its isolated GPU component screen.
//! Retain the native sampler's global tile boundaries and exact GPU arithmetic.
use crate::{categorical::{Categorical,TARGET_DOMAIN,uniform},cuda::{Buffer,Cuda,Graph,Module,Stream},roce::Roce};
use anyhow::{Result,ensure};
use serde_json::{Value,json};
use std::{net::SocketAddr,path::Path};
const COLS:usize=32320;
const WORDS:usize=2176;
const VOCAB:usize=COLS*4;
macro_rules! launch {
    ($module:expr,$name:expr,$grid:expr,$stream:expr; $($arg:ident),+ $(,)?)=>{{
        let mut args=[$(std::ptr::addr_of_mut!($arg).cast::<std::ffi::c_void>()),+];
        unsafe{$module.function($name)?.launch($grid,[256,1,1],0,$stream,&mut args)}
    }};
}
pub(crate) struct CompactSampler {
    // Executable pointers must die before their captured owned storage.
    graph:Option<Graph>,graph_input:Option<u64>,graph_comm:Option<u64>,captures:usize,
    cuda:Cuda,module:Module,rows:usize,rank:usize,
    packed:Buffer<f32>,gathered:Buffer<f32>,maxima:Buffer<f32>,sums:Buffer<f64>,
    uniforms:Buffer<f64>,tiles:Buffer<i32>,fractions:Buffer<f64>,normalization:Buffer<f64>,
    factors:Buffer<f64>,errors:Buffer<u32>,owned:Buffer<i32>,scratch:Buffer<f32>,
    local_ids:Buffer<i32>,packed_ids:Buffer<f32>,gathered_ids:Buffer<f32>,output:Buffer<i32>,
}
impl CompactSampler {
    pub(crate) fn new(cuda:&Cuda,core:&[u8],rows:usize,rank:usize)->Result<Self>{
        ensure!([1,6,16,32].contains(&rows)&&rank<4,"finite compact sampler shape");
        Ok(Self{graph:None,graph_input:None,graph_comm:None,captures:0,
            cuda:cuda.clone(),module:cuda.module(core)?,rows,rank,
            packed:cuda.allocate(rows*WORDS)?,gathered:cuda.allocate(rows*WORDS*4)?,
            maxima:cuda.allocate(rows*127)?,sums:cuda.allocate(rows*127)?,uniforms:cuda.allocate(rows)?,
            tiles:cuda.allocate(rows)?,fractions:cuda.allocate(rows)?,normalization:cuda.allocate(rows*2)?,
            factors:cuda.allocate(rows*127)?,errors:cuda.allocate(rows)?,owned:cuda.allocate(rows)?,
            scratch:cuda.allocate(rows*VOCAB)?,local_ids:cuda.allocate(rows)?,
            packed_ids:cuda.allocate(rows*4)?,gathered_ids:cuda.allocate(rows*16)?,output:cuda.allocate(rows)?})
    }
    pub(crate) fn captures(&self)->usize {self.captures}
    pub(crate) fn sample(&mut self,local:&Buffer<f32>,seed:u64,position:u64,comm:&Roce,stream:&Stream)->Result<Vec<u32>> {
        ensure!([16,32].contains(&self.rows)&&position.checked_add(self.rows as u64).is_some()
            &&self.cuda.owns_buffer(local)&&self.cuda.owns_stream(stream)&&comm.rank()==self.rank
            &&self.graph_input.is_none_or(|p|p==local.ptr())
            &&self.graph_comm.is_none_or(|id|id==comm.instance()),"compact sample ownership");
        let uniforms:Vec<_>=(0..self.rows).map(|row|uniform(seed,TARGET_DOMAIN,position+row as u64)).collect();
        self.uniforms.upload(&uniforms)?;stream.join_default()?;
        if self.graph.is_none() {
            self.enqueue(local,1.,comm,stream)?;stream.synchronize()?;comm.health()?;
            self.graph=Some(stream.capture(||self.enqueue(local,1.,comm,stream))?);
            self.graph_input=Some(local.ptr());self.graph_comm=Some(comm.instance());self.captures+=1;
        }
        unsafe{self.graph.as_ref().unwrap().launch(stream)?;}
        stream.synchronize()?;comm.health()?;
        ensure!(self.errors.download()?.iter().all(|&e|e==0),"invalid compact categorical logits or probability mass");
        let ids=self.output.download()?;ensure!(ids.iter().all(|&id|(0..VOCAB as i32).contains(&id)),"compact categorical output range");
        Ok(ids.into_iter().map(|id|id as u32).collect())
    }
    fn enqueue(&mut self,local:&Buffer<f32>,temperature:f32,comm:&Roce,stream:&Stream)->Result<()> {
        ensure!(local.len()==self.rows*COLS&&temperature.is_finite()&&temperature>0.,"compact local logits");
        let (mut input,mut packed,mut rank,mut temp)=(local.ptr(),self.packed.ptr(),self.rank as i32,temperature);
        launch!(self.module,"ca_sharded_categorical_pack",[34,self.rows as u32,1],stream;input,packed,rank,temp)?;
        unsafe{comm.gather_f32(&self.packed,&mut self.gathered,WORDS,stream)?;
            self.cuda.zero_device(self.errors.ptr(),self.errors.bytes(),stream)?;}
        let (mut gathered,mut maxima,mut sums,mut errors)=(self.gathered.ptr(),self.maxima.ptr(),self.sums.ptr(),self.errors.ptr());
        launch!(self.module,"ca_sharded_categorical_unpack",[127,self.rows as u32,1],stream;gathered,maxima,sums,errors,temp)?;
        let (mut uniforms,mut tiles,mut fractions,mut normalization,mut factors)=
            (self.uniforms.ptr(),self.tiles.ptr(),self.fractions.ptr(),self.normalization.ptr(),self.factors.ptr());
        launch!(self.module,"ca_categorical_tile_cached",[self.rows as u32,1,1],stream;
            maxima,sums,uniforms,tiles,fractions,normalization,factors,errors,temp)?;
        let (mut owned,mut scratch)=(self.owned.ptr(),self.scratch.ptr());
        launch!(self.module,"ca_sharded_categorical_selected",[self.rows as u32,1,1],stream;
            input,gathered,tiles,errors,owned,scratch,rank)?;
        let mut local_ids=self.local_ids.ptr();
        launch!(self.module,"ca_categorical_finish_raking",[self.rows as u32,1,1],stream;
            scratch,maxima,owned,fractions,errors,local_ids,temp)?;
        let (mut packed_ids,mut rows)=(self.packed_ids.ptr(),self.rows as i32);
        launch!(self.module,"ca_sharded_categorical_id_pack",[1,1,1],stream;local_ids,errors,packed_ids,rows)?;
        unsafe{comm.gather_f32(&self.packed_ids,&mut self.gathered_ids,4,stream)?;}
        let (mut gathered_ids,mut output)=(self.gathered_ids.ptr(),self.output.ptr());
        launch!(self.module,"ca_sharded_categorical_id_finish",[1,1,1],stream;gathered_ids,output,errors,rows)
    }
}

pub fn run(rank:usize,endpoint:&str,nonce:&[u8],root:&Path,core_path:&Path)->Result<Value>{
    ensure!(std::env::var("CANADA_SCRATCH_SHARDED_CATEGORICAL_SCREEN").as_deref()==Ok("1"),
        "isolated compact sampler admission");
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let core=std::fs::read(core_path)?;let module=cuda.module(&core)?;
    let mut comm=Roce::local(&cuda,&root.join("roce-v0"),rank)?;comm.attach_gather(&root.join("roce-gather-v1"))?;
    let mut address:SocketAddr=endpoint.parse()?;
    let blobs=crate::rendezvous::exchange_blob(rank,address,nonce,&comm.blob()?)?;comm.connect(&blobs)?;
    address.set_port(address.port().checked_add(1).ok_or_else(||anyhow::anyhow!("compact sampler ready port"))?);
    ensure!(crate::rendezvous::exchange_blob(rank,address,nonce,b"R")?==b"RRRR","all compact sampler peers ready");
    let mut sum_input=cuda.allocate::<u16>(8)?;sum_input.upload(&vec![half::bf16::from_f32((rank+1)as f32).to_bits();8])?;
    let mut sum_output=cuda.allocate::<u16>(8)?;
    let mut results=Vec::new();
    for rows in [1usize,6,16,32] {
        let mut local=cuda.allocate::<f32>(rows*COLS)?;let mut full=cuda.allocate::<f32>(rows*VOCAB)?;
        let mut reference=Categorical::new(&cuda,&core,rows)?;
        ensure!(reference.cached_probability_factors()&&reference.parallel_finish_enabled(),"native reference sampler settings");
        let mut compact=CompactSampler::new(&cuda,&core,rows,rank)?;
        local.zero()?;stream.join_default()?;
        let mut cases=Vec::new();
        for phase in 0..8 {
            let (mut p,mut r,mut n,mut f)=(local.ptr(),rank as i32,rows as i32,phase);
            launch!(module,"ca_sharded_categorical_fixture",[(local.len().div_ceil(256))as u32,1,1],&stream;p,r,n,f)?;
            for draw in 0..3 {
                let uniforms:Vec<_>=(0..rows).map(|row|match draw {
                    0=>0.,1=>1.-1./9007199254740992.,_=>uniform(2026111100,TARGET_DOMAIN,(phase as u64*32+row as u64)*3+draw),
                }).collect();
                reference.upload_uniforms(&uniforms)?;compact.uniforms.upload(&uniforms)?;stream.join_default()?;
                let temperature=if phase==3{0.7}else{1.};
                unsafe{comm.gather_f32(&local,&mut full,COLS,&stream)?;}
                reference.enqueue(&full,0,rows,temperature,&stream)?;
                compact.enqueue(&local,temperature,&comm,&stream)?;
                stream.synchronize()?;comm.health()?;
                reference.compare_cdf_for_sharded_probe(&compact.maxima,&compact.sums,&compact.tiles,&compact.fractions)?;
                let ids=reference.device_output().download()?;let errors=reference.device_errors().download()?;
                ensure!(ids==compact.output.download()?,"compact sample IDs M{rows}/phase{phase}/draw{draw}");
                ensure!(errors==compact.errors.download()?,"compact sample errors M{rows}/phase{phase}/draw{draw}");
                if phase==2 {
                    const EDGES:[i32;22]=[0,31743,31744,32319,32320,32767,32768,64511,64512,64639,64640,
                        65535,65536,96255,96256,96959,96960,97279,97280,129023,129024,129279];
                    ensure!(ids.iter().enumerate().all(|(row,&id)|id==EDGES[row%22]),"single-support boundary ownership");
                }
                cases.push(json!({"phase":phase,"draw":draw,"temperature":temperature,"exact_partials_and_cdf":true,
                    "ids":ids,"errors":errors}));
            }
        }
        // Same arbitrary dense GPU input for both timed paths. All RNG upload
        // and CPU readout stays outside the GPU interval, just as graph replay.
        let (mut p,mut r,mut n,mut phase)=(local.ptr(),rank as i32,rows as i32,0i32);
        launch!(module,"ca_sharded_categorical_fixture",[(local.len().div_ceil(256))as u32,1,1],&stream;p,r,n,phase)?;
        let uniforms:Vec<_>=(0..rows).map(|i|uniform(2026111101,TARGET_DOMAIN,i as u64)).collect();
        reference.upload_uniforms(&uniforms)?;compact.uniforms.upload(&uniforms)?;stream.join_default()?;
        unsafe{comm.gather_f32(&local,&mut full,COLS,&stream)?;}
        reference.enqueue(&full,0,rows,1.,&stream)?;compact.enqueue(&local,1.,&comm,&stream)?;stream.synchronize()?;
        let control=stream.capture(||{for _ in 0..8 {
            unsafe{comm.gather_f32(&local,&mut full,COLS,&stream)?;}
            reference.enqueue(&full,0,rows,1.,&stream)?;
        }Ok(())})?;
        let candidate=stream.capture(||{for _ in 0..8 {compact.enqueue(&local,1.,&comm,&stream)?;}Ok(())})?;
        let mut pairs=Vec::new();
        for repeat in 0..4 {
            let order=if repeat%2==0{[0,1]}else{[1,0]};let mut milliseconds=[0f32;2];
            for variant in order {
                unsafe{comm.sum_bf16(&sum_input,&mut sum_output,&stream)?;}
                let graph=if variant==0{&control}else{&candidate};
                for _ in 0..2{unsafe{graph.launch(&stream)?;}}stream.synchronize()?;
                let start=cuda.event()?;let end=cuda.event()?;start.record(&stream)?;
                for _ in 0..8{unsafe{graph.launch(&stream)?;}}
                end.record(&stream)?;milliseconds[variant]=end.elapsed_since(&start)?/64.;comm.health()?;
                ensure!(sum_output.download()?.iter().all(|&v|v==half::bf16::from_f32(10.).to_bits()),
                    "compact sampler mixed collective ordering");
            }
            ensure!(reference.device_output().download()?==compact.output.download()?,"replayed compact sampled IDs");
            ensure!(reference.device_errors().download()?==compact.errors.download()?,"replayed compact errors");
            reference.compare_cdf_for_sharded_probe(&compact.maxima,&compact.sums,&compact.tiles,&compact.fractions)?;
            pairs.push(json!({"repeat":repeat,"order":order,"milliseconds":milliseconds}));
        }
        results.push(json!({"rows":rows,"full_local_bytes":rows*COLS*4,"compact_local_bytes":rows*(WORDS+4)*4,
            "cases":cases,"pairs":pairs,"mixed_collective_ordering_exact":true}));
        eprintln!("rank{rank}: compact categorical M{rows} exact");
    }
    stream.synchronize()?;comm.health()?;
    Ok(json!({"rank":rank,"all_exact":true,"results":results,"all_peers_ready_before_gpu":true,
        "scope":"GPU-only component. Exact native127 global tile maxima, FP64 partial sums, tile selections, conditional fractions, error masks and sampled IDs. Eight synthetic phases, three uniforms, four row shapes,4reversed timing pairs and mixed BF16 reductions. Canonical expf/FP64 math and CUB finish reused. No serving integration, no model or quality claim."}))
}
