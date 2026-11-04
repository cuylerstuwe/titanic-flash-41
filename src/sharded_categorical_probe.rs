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
    local_probabilities:Option<LocalProbabilities>,
    fused_controls:bool,
    fused_finish:bool,
    staged_inputs:Option<std::rc::Rc<crate::cuda::MetadataStaging>>,
}
struct LocalProbabilities { normalized:Buffer<f64>, selected:Buffer<f64> }
impl CompactSampler {
    pub(crate) fn bind_rank(&mut self,rank:usize)->Result<()> {
        ensure!(rank<4&&(self.rank==rank||self.graph.is_none()),"compact target rank binding");self.rank=rank;Ok(())
    }
    pub(crate) fn new(cuda:&Cuda,core:&[u8],rows:usize,rank:usize)->Result<Self>{
        Self::with_uniform_allocator(cuda,core,rows,rank,cuda)
    }
    pub(crate) fn with_uniform_allocator(cuda:&Cuda,core:&[u8],rows:usize,rank:usize,controls:&Cuda)->Result<Self>{
        ensure!(((1..=6).contains(&rows)||[16,32].contains(&rows))&&rank<4,"finite compact sampler shape");
        let uniforms=controls.allocate(rows)?;ensure!(cuda.owns_buffer(&uniforms),"compact target control context");
        Ok(Self{graph:None,graph_input:None,graph_comm:None,captures:0,
            cuda:cuda.clone(),module:cuda.module(core)?,rows,rank,
            packed:cuda.allocate(rows*WORDS)?,gathered:cuda.allocate(rows*WORDS*4)?,
            maxima:cuda.allocate(rows*127)?,sums:cuda.allocate(rows*127)?,uniforms,
            tiles:cuda.allocate(rows)?,fractions:cuda.allocate(rows)?,normalization:cuda.allocate(rows*2)?,
            factors:cuda.allocate(rows*127)?,errors:cuda.allocate(rows)?,owned:cuda.allocate(rows)?,
            scratch:cuda.allocate(rows*VOCAB)?,local_ids:cuda.allocate(rows)?,
            packed_ids:cuda.allocate(rows*4)?,gathered_ids:cuda.allocate(rows*16)?,output:cuda.allocate(rows)?,
            local_probabilities:None,fused_controls:crate::sampler_transport::Policy::from_env()?.fused_target_controls,
            fused_finish:crate::sampler_transport::Policy::from_env()?.fused_target_finish,
            staged_inputs:controls.metadata_staging()})
    }
    pub(crate) fn set_fused_controls_for_probe(&mut self,enabled:bool)->Result<()> {
        ensure!(self.graph.is_none()&&self.captures==0&&self.local_probabilities.is_some(),
            "fused controls must be selected before any graph capture");
        self.fused_controls=enabled;Ok(())
    }
    pub(crate) fn set_fused_finish_for_probe(&mut self,enabled:bool)->Result<()> {
        ensure!(self.graph.is_none()&&self.captures==0&&self.local_probabilities.is_some(),
            "fused finish must be selected before any graph capture");
        self.fused_finish=enabled;Ok(())
    }
    pub(crate) fn prepare_local_probabilities(&mut self)->Result<()> {
        ensure!(self.local_probabilities.is_none(),"compact probabilities already prepared");
        self.local_probabilities=Some(LocalProbabilities{normalized:self.cuda.allocate(self.rows*127)?,
            selected:self.cuda.allocate(self.rows)?});Ok(())
    }
    pub(crate) fn upload_uniforms(&mut self,values:&[f64])->Result<()> {
        ensure!(values.len()==self.rows,"compact target uniform extent");self.uniforms.upload(values)
    }
    pub(crate) fn output(&self)->&Buffer<i32>{&self.output}
    pub(crate) fn local_output(&self)->&Buffer<i32>{&self.local_ids}
    pub(crate) fn errors(&self)->&Buffer<u32>{&self.errors}
    pub(crate) fn enqueue_local_probabilities(&mut self,local:&Buffer<f32>,proposals:&Buffer<i32>,
        probabilities:&Buffer<f64>,temperature:f32,comm:&Roce,stream:&Stream)->Result<()> {
        ensure!(proposals.len()>=self.rows&&probabilities.len()>=self.rows*VOCAB
            &&self.cuda.owns_buffer(proposals)&&self.cuda.owns_buffer(probabilities)
            &&self.local_probabilities.is_some(),"compact local probability storage");
        self.enqueue_inner(local,temperature,comm,stream,Some((proposals.ptr(),probabilities.ptr())),false)
    }
    pub(crate) fn enqueue_deferred_probabilities(&mut self,local:&Buffer<f32>,proposals:&Buffer<i32>,
        probabilities:&Buffer<f64>,temperature:f32,comm:&Roce,stream:&Stream)->Result<()> {
        ensure!(proposals.len()>=self.rows&&probabilities.len()>=self.rows*VOCAB
            &&self.cuda.owns_buffer(proposals)&&self.cuda.owns_buffer(probabilities)
            &&self.local_probabilities.is_some(),"deferred compact probability storage");
        self.enqueue_inner(local,temperature,comm,stream,Some((proposals.ptr(),probabilities.ptr())),true)
    }
    pub(crate) fn captures(&self)->usize {self.captures}
    pub(crate) fn sample(&mut self,local:&Buffer<f32>,seed:u64,position:u64,comm:&Roce,stream:&Stream)->Result<Vec<u32>> {
        ensure!(((1..=6).contains(&self.rows)||[16,32].contains(&self.rows))&&position.checked_add(self.rows as u64).is_some()
            &&self.cuda.owns_buffer(local)&&self.cuda.owns_stream(stream)&&comm.rank()==self.rank
            &&self.graph_input.is_none_or(|p|p==local.ptr())
            &&self.graph_comm.is_none_or(|id|id==comm.instance()),"compact sample ownership");
        let uniforms:Vec<_>=(0..self.rows).map(|row|uniform(seed,TARGET_DOMAIN,position+row as u64)).collect();
        self.uniforms.upload(&uniforms)?;stream.join_default()?;
        if let Some(staging)=&self.staged_inputs{unsafe{staging.flush(Some(stream))?;}}
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
        self.enqueue_inner(local,temperature,comm,stream,None,false)
    }
    fn enqueue_inner(&mut self,local:&Buffer<f32>,temperature:f32,comm:&Roce,stream:&Stream,
        probabilities:Option<(u64,u64)>,defer_ids:bool)->Result<()> {
        ensure!(local.len()==self.rows*COLS&&temperature.is_finite()&&temperature>0.
            &&self.cuda.owns_buffer(local)&&self.cuda.owns_stream(stream)&&comm.rank()==self.rank,"compact local logits");
        let (mut input,mut packed,mut rank,mut temp)=(local.ptr(),self.packed.ptr(),self.rank as i32,temperature);
        let fused=self.fused_controls&&probabilities.is_some();
        if fused {
            let mut proposals=probabilities.unwrap().0;let mut errors=self.errors.ptr();
            launch!(self.module,"ca_target_pack_fused",[34,self.rows as u32,1],stream;
                input,packed,rank,temp,proposals,errors)?;
        }else{
            launch!(self.module,"ca_sharded_categorical_pack",[34,self.rows as u32,1],stream;input,packed,rank,temp)?;
            if let Some((mut proposals,_))=probabilities {
                let mut rows=self.rows as i32;
                launch!(self.module,"ca_target_proposal_pack",[1,1,1],stream;input,proposals,packed,rank,rows)?;
            }
        }
        unsafe{comm.gather_f32(&self.packed,&mut self.gathered,WORDS,stream)?;
            if !fused{self.cuda.zero_device(self.errors.ptr(),self.errors.bytes(),stream)?;}}
        let (mut gathered,mut maxima,mut sums,mut errors)=(self.gathered.ptr(),self.maxima.ptr(),self.sums.ptr(),self.errors.ptr());
        launch!(self.module,"ca_sharded_categorical_unpack",[127,self.rows as u32,1],stream;gathered,maxima,sums,errors,temp)?;
        let (mut uniforms,mut tiles,mut fractions,mut normalization,mut factors)=
            (self.uniforms.ptr(),self.tiles.ptr(),self.fractions.ptr(),self.normalization.ptr(),self.factors.ptr());
        if fused {
            let(mut proposals,mut p)=probabilities.unwrap();let extra=self.local_probabilities.as_ref().unwrap();
            let(mut normalized,mut selected)=(extra.normalized.ptr(),extra.selected.ptr());
            launch!(self.module,"ca_target_tile_factors_fused",[self.rows as u32,1,1],stream;
                maxima,sums,uniforms,tiles,fractions,normalization,factors,errors,temp,gathered,proposals,normalized,p,selected)?;
        }else{
            launch!(self.module,"ca_categorical_tile_cached",[self.rows as u32,1,1],stream;
                maxima,sums,uniforms,tiles,fractions,normalization,factors,errors,temp)?;
        }
        let mut local_ids=self.local_ids.ptr();
        if self.fused_finish&&probabilities.is_some() {
            launch!(self.module,"ca_target_selected_finish",[self.rows as u32,1,1],stream;
                input,gathered,maxima,tiles,fractions,errors,local_ids,rank,temp)?;
        } else {
            let (mut owned,mut scratch)=(self.owned.ptr(),self.scratch.ptr());
            launch!(self.module,"ca_sharded_categorical_selected",[self.rows as u32,1,1],stream;
                input,gathered,tiles,errors,owned,scratch,rank)?;
            launch!(self.module,"ca_categorical_finish_raking",[self.rows as u32,1,1],stream;
                scratch,maxima,owned,fractions,errors,local_ids,temp)?;
        }
        if let Some((mut proposals,mut p))=probabilities {
            let extra=self.local_probabilities.as_ref().unwrap();
            let(mut normalized,mut selected,mut rows)=(extra.normalized.ptr(),extra.selected.ptr(),self.rows as i32);
            if !fused{launch!(self.module,"ca_target_local_factors",[self.rows as u32,1,1],stream;
                gathered,maxima,normalization,factors,proposals,errors,normalized,p,selected,temp)?;}
            launch!(self.module,"ca_target_local_probabilities",[127,self.rows as u32,1],stream;
                input,maxima,normalized,errors,p,rank,rows,temp)?;
        }
        // The sharded block can carry these IDs/errors in its residual wire,
        // then resolve the original output buffers before final readout.
        if defer_ids{return Ok(());}
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
    let local_probabilities=std::env::var("CANADA_SCRATCH_LOCAL_TARGET_SCREEN").as_deref()==Ok("1");
    let narrow=std::env::var("CANADA_SCRATCH_LOCAL_HISTORY_SCREEN").as_deref()==Ok("1");
    let shapes=if local_probabilities||narrow{vec![1usize,2,3,4,5,6]}else{vec![1usize,6,16,32]};
    for rows in shapes {
        let mut local=cuda.allocate::<f32>(rows*COLS)?;let mut full=cuda.allocate::<f32>(rows*VOCAB)?;
        let mut reference=Categorical::new(&cuda,&core,rows)?;
        ensure!(reference.cached_probability_factors()&&reference.parallel_finish_enabled(),"native reference sampler settings");
        let mut compact=CompactSampler::new(&cuda,&core,rows,rank)?;
        let mut expected_p=cuda.allocate::<f64>(rows*VOCAB)?;
        let actual_p=cuda.allocate::<f64>(rows*VOCAB)?;
        let mut proposals=cuda.allocate::<i32>(rows)?;
        if local_probabilities{compact.prepare_local_probabilities()?;}
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
                let ids:Vec<_>=(0..rows).map(|row|[0,32319,32320,64639,64640,96959,96960,129279]
                    [(row+draw as usize+phase as usize)%8]).collect();
                proposals.upload(&ids)?;stream.join_default()?;
                let temperature=if phase==3{0.7}else{1.};
                unsafe{comm.gather_f32(&local,&mut full,COLS,&stream)?;}
                reference.enqueue(&full,0,rows,temperature,&stream)?;
                if local_probabilities {
                    reference.probabilities(&full,0,rows,temperature,&mut expected_p,0,&stream)?;
                    compact.enqueue_local_probabilities(&local,&proposals,&actual_p,temperature,&comm,&stream)?;
                }else{compact.enqueue(&local,temperature,&comm,&stream)?;}
                stream.synchronize()?;comm.health()?;
                reference.compare_cdf_for_sharded_probe(&compact.maxima,&compact.sums,&compact.tiles,&compact.fractions)?;
                let ids=reference.device_output().download()?;let errors=reference.device_errors().download()?;
                ensure!(ids==compact.output.download()?,"compact sample IDs M{rows}/phase{phase}/draw{draw}");
                ensure!(errors==compact.errors.download()?,"compact sample errors M{rows}/phase{phase}/draw{draw}");
                if local_probabilities {
                    let a=actual_p.download()?;let b=expected_p.download()?;let proposed=proposals.download()?;
                    let same=|i:usize|a[i].to_bits()==b[i].to_bits()||(a[i].is_nan()&&b[i].is_nan());
                    ensure!((0..rows).all(|row|(rank*COLS..(rank+1)*COLS).all(|id|same(row*VOCAB+id))
                        &&same(row*VOCAB+proposed[row]as usize)),"compact target probability changed M{rows}/phase{phase}/draw{draw}");
                }
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
            if local_probabilities{reference.probabilities(&full,0,rows,1.,&mut expected_p,0,&stream)?;}
        }Ok(())})?;
        let candidate=stream.capture(||{for _ in 0..8 {
            if local_probabilities{compact.enqueue_local_probabilities(&local,&proposals,&actual_p,1.,&comm,&stream)?;}
            else{compact.enqueue(&local,1.,&comm,&stream)?;}
        }Ok(())})?;
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
        "local_target_probabilities":local_probabilities,
        "scope":"GPU-only component. Exact native127 global tile maxima, FP64 partial sums, tile selections, conditional fractions, error masks and sampled IDs; when enabled, owned target probabilities and proposal scalar values also exact. Eight synthetic phases, three uniforms, declared row shapes,4reversed timing pairs and mixed BF16 reductions. Actual RoCE transport included. No serving integration, no model or quality claim."}))
}
