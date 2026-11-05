//! Finite draft-only hierarchical categorical screen. One compact exchange
//! selects among independent conditional-shard draws. All actual q is FP64.
use crate::{categorical::{Categorical,uniform,DRAFT_DOMAIN},cuda::{Buffer,Cuda,Module,Stream},roce::Roce};
use anyhow::{Result,ensure};
use serde_json::{Value,json};
use std::{net::SocketAddr,path::Path};
const COLS:usize=32320;const VOCAB:usize=4*COLS;
const LOCAL_DOMAIN:u64=0x484945525f4c4f43;
macro_rules! launch {
 ($module:expr,$name:expr,$grid:expr,$stream:expr;$($arg:ident),+$(,)?)=>{{
  let mut args=[$(std::ptr::addr_of_mut!($arg).cast::<std::ffi::c_void>()),+];
  unsafe{$module.function($name)?.launch($grid,[256,1,1],0,$stream,&mut args)}
 }};
}
pub(crate) struct Hierarchical {
 cuda:Cuda,module:Module,rank:usize,uniforms:Buffer<f64>,maxima:Buffer<f32>,sums:Buffer<f64>,
 normalization:Buffer<f64>,factors:Buffer<f64>,tile:Buffer<i32>,fraction:Buffer<f64>,
 local_error:Buffer<u32>,local_id:Buffer<i32>,packed:Buffer<f32>,gathered:Buffer<f32>,
 rank_fractions:Buffer<f64>,output:Buffer<i32>,errors:Buffer<u32>,selected_q:Buffer<f64>,
 fused_local:bool,
 warp_q:bool,
}
impl Hierarchical {
 pub(crate) fn new(cuda:&Cuda,core:&[u8],rank:usize)->Result<Self>{
  Self::with_uniform_allocator(cuda,core,rank,cuda)
 }
 pub(crate) fn with_uniform_allocator(cuda:&Cuda,core:&[u8],rank:usize,controls:&Cuda)->Result<Self>{
  ensure!(rank<4,"hierarchical sampler rank");
  let policy=crate::sampler_transport::Policy::from_env()?;
  let uniforms=controls.allocate(2)?;ensure!(cuda.owns_buffer(&uniforms),"hierarchical control context");
  Ok(Self{cuda:cuda.clone(),module:cuda.module(core)?,rank,uniforms,
   maxima:cuda.allocate(32)?,sums:cuda.allocate(32)?,normalization:cuda.allocate(2)?,
   factors:cuda.allocate(32)?,tile:cuda.allocate(1)?,fraction:cuda.allocate(1)?,
   local_error:cuda.allocate(1)?,local_id:cuda.allocate(1)?,packed:cuda.allocate(8)?,
   gathered:cuda.allocate(32)?,rank_fractions:cuda.allocate(4)?,output:cuda.allocate(1)?,errors:cuda.allocate(1)?,selected_q:cuda.allocate(1)?,
   fused_local:policy.fused_draft_local,warp_q:policy.warp_draft_q})
 }
 pub(crate) fn upload_uniforms(&mut self,seed:u64,position:u64)->Result<()> {
  // The caller supplies a fresh block-round seed. Local draws have distinct
  // domains for all four ranks, independent of the common shard-choice draw.
  self.uniforms.upload(&[uniform(seed,LOCAL_DOMAIN^(self.rank as u64),position),
      uniform(seed,DRAFT_DOMAIN,position)])
 }
 pub(crate) fn output(&self)->&Buffer<i32>{&self.output}
 pub(crate) fn errors(&self)->&Buffer<u32>{&self.errors}
 pub(crate) fn selected_q(&self)->&Buffer<f64>{&self.selected_q}
 pub(crate) fn enqueue(&mut self,local:&Buffer<f32>,first:usize,q:&Buffer<f64>,qrow:usize,
     temperature:f32,comm:&Roce,stream:&Stream)->Result<()> {
  ensure!(first.checked_add(1).and_then(|n|n.checked_mul(COLS)).is_some_and(|n|n<=local.len())
   &&qrow.checked_add(1).and_then(|n|n.checked_mul(COLS)).is_some_and(|n|n<=q.len())
   &&temperature.is_finite()&&temperature>0.&&comm.rank()==self.rank
   &&self.cuda.owns_buffer(local)&&self.cuda.owns_buffer(q)&&self.cuda.owns_stream(stream),
   "hierarchical sampler geometry/owner");
  unsafe{self.cuda.zero_device(self.local_error.ptr(),4,stream)?;}
  let (mut x,mut maxima,mut sums,mut local_errors,mut temp)=
   (local.ptr()+(first*COLS*4)as u64,self.maxima.ptr(),self.sums.ptr(),self.local_error.ptr(),temperature);
  launch!(self.module,"ca_hierarchy_partials",[32,1,1],stream;x,maxima,sums,local_errors,temp)?;
  let(mut u,mut tile,mut fraction,mut norm,mut factors)=
   (self.uniforms.ptr(),self.tile.ptr(),self.fraction.ptr(),self.normalization.ptr(),self.factors.ptr());
  let mut local_id=self.local_id.ptr();
  let mut packed=self.packed.ptr();
  if self.fused_local {
   launch!(self.module,"ca_hierarchy_local_fused",[1,1,1],stream;
    x,maxima,sums,u,tile,fraction,norm,factors,local_errors,local_id,packed,temp)?;
  }else{
   launch!(self.module,"ca_hierarchy_tile",[1,1,1],stream;maxima,sums,u,tile,fraction,norm,factors,local_errors,temp)?;
   launch!(self.module,"ca_hierarchy_local_finish",[1,1,1],stream;x,maxima,tile,fraction,local_errors,local_id,temp)?;
   launch!(self.module,"ca_hierarchy_pack",[1,1,1],stream;norm,local_id,local_errors,packed,x,maxima,factors,temp)?;
  }
  unsafe{comm.gather_f32(&self.packed,&mut self.gathered,8,stream)?;}
  let(mut gathered,mut global_u,mut rank_fractions,mut output,mut errors)=
   (self.gathered.ptr(),self.uniforms.ptr()+8,self.rank_fractions.ptr(),self.output.ptr(),self.errors.ptr());
  let mut selected_q=self.selected_q.ptr();
  launch!(self.module,"ca_hierarchy_finish",[1,1,1],stream;gathered,global_u,rank_fractions,output,errors,selected_q,temp)?;
  let(mut qptr,mut rank)=(q.ptr()+(qrow*COLS*8)as u64,self.rank as i32);
  launch!(self.module,if self.warp_q{"ca_hierarchy_q_warp"}else{"ca_hierarchy_q"},[127,1,1],stream;x,maxima,norm,factors,rank_fractions,qptr,errors,rank,temp)
 }
}

pub fn run(rank:usize,endpoint:&str,nonce:&[u8],root:&Path,core_path:&Path)->Result<Value>{
 ensure!(std::env::var("CANADA_SCRATCH_HIERARCHICAL_SCREEN").as_deref()==Ok("1"),"finite hierarchical sampler admission");
 let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
 let core=std::fs::read(core_path)?;let module=cuda.module(&core)?;
 let mut comm=Roce::local(&cuda,&root.join("roce-v0"),rank)?;comm.attach_gather(&root.join("roce-gather-v1"))?;
 let mut address:SocketAddr=endpoint.parse()?;
 let peers=crate::rendezvous::exchange_blob(rank,address,nonce,&comm.blob()?)?;comm.connect(&peers)?;
 address.set_port(address.port().checked_add(1).ok_or_else(||anyhow::anyhow!("hierarchy ready port"))?);
 ensure!(crate::rendezvous::exchange_blob(rank,address,nonce,b"R")?==b"RRRR","all hierarchy peers ready");
 let rows=5usize;let local=cuda.allocate::<f32>(rows*COLS)?;let mut full=cuda.allocate::<f32>(rows*VOCAB)?;
 let localq=cuda.allocate::<f64>(rows*COLS)?;let mut candidateq=cuda.allocate::<f64>(rows*VOCAB)?;
 let mut referenceq=cuda.allocate::<f64>(rows*VOCAB)?;
 let mut reference=(0..rows).map(|_|Categorical::new(&cuda,&core,1)).collect::<Result<Vec<_>>>()?;
 let mut candidate=(0..rows).map(|_|Hierarchical::new(&cuda,&core,rank)).collect::<Result<Vec<_>>>()?;
 if std::env::var("CANADA_SCRATCH_FUSED_PROBABILITY_RESIDUAL_SCREEN").as_deref()==Ok("1") {
  // Isolate the new composition instead of repeating the already-qualified
  // hierarchical algorithm and transport panels on every local fusion.
  let(mut ptr,mut r,mut n,mut phase)=(local.ptr(),rank as i32,rows as i32,0i32);
  launch!(module,"ca_sharded_categorical_fixture",[local.len().div_ceil(256)as u32,1,1],&stream;ptr,r,n,phase)?;
  for step in 0..rows {
   candidate[step].upload_uniforms(2026110401,step as u64)?;
   stream.join_default()?;
   candidate[step].enqueue(&local,step,&localq,step,1.,&comm,&stream)?;
  }
  stream.synchronize()?;comm.health()?;
  let bundled=crate::bundled_sampler_probe::run(&cuda,&core,&localq,&candidate,&comm,&stream)?;
  stream.synchronize()?;comm.health()?;
  return Ok(json!({"rank":rank,"passed":true,"bundled":bundled,"all_peers_ready_before_gpu":true,
   "scope":"Focused probability/residual fusion with the current hierarchical actual-q proposal law; no model weights loaded."}));
 }
 let mut sum_input=cuda.allocate::<u16>(8)?;let mut sum_output=cuda.allocate::<u16>(8)?;
 sum_input.upload(&vec![half::bf16::from_f32((rank+1)as f32).to_bits();8])?;
 let mut cases=Vec::new();
 for phase in 0..8i32 {
  let(mut ptr,mut r,mut n,mut ph)=(local.ptr(),rank as i32,rows as i32,phase);
  launch!(module,"ca_sharded_categorical_fixture",[local.len().div_ceil(256)as u32,1,1],&stream;ptr,r,n,ph)?;
  for draw in 0..2u64 {
   let temp=if phase==3{0.7}else{1.};
   for step in 0..rows {
    let position=phase as u64*100+draw*5+step as u64;
    reference[step].upload_uniforms(&[uniform(2026110303,DRAFT_DOMAIN,position)])?;
    candidate[step].upload_uniforms(2026110303,position)?;
   }
   stream.join_default()?;
   unsafe{comm.gather_f32(&local,&mut full,COLS,&stream)?;}
   for step in 0..rows {
    reference[step].enqueue(&full,step,1,temp,&stream)?;
    reference[step].probabilities(&full,step,1,temp,&mut referenceq,step,&stream)?;
    candidate[step].enqueue(&local,step,&localq,step,temp,&comm,&stream)?;
   }
   unsafe{comm.gather_f64(&localq,&mut candidateq,COLS,&stream)?;}
   stream.synchronize()?;comm.health()?;
   let expected=referenceq.download()?;let actual=candidateq.download()?;
   let mut checks=Vec::new();
   for step in 0..rows {
    let error=candidate[step].errors.download()?[0];
    let baseline_error=reference[step].device_errors().download()?[0];
    let id=candidate[step].output.download()?[0];
    ensure!((error==0)==(baseline_error==0),"hierarchical error validity phase{phase} row{step}");
    let a=&actual[step*VOCAB..(step+1)*VOCAB];let b=&expected[step*VOCAB..(step+1)*VOCAB];
    let(l1,mass)=if error==0 {
     ensure!((0..VOCAB as i32).contains(&id)&&a[id as usize]>0.,"hierarchical positive support");
     ensure!(candidate[step].selected_q.download()?[0].to_bits()==a[id as usize].to_bits(),
       "compact selected-q differs from the actual full proposal law");
     let l1=a.iter().zip(b).map(|(a,b)|(a-b).abs()).sum::<f64>();let mass=a.iter().sum::<f64>();
     ensure!(l1<2e-7&&(mass-1.).abs()<2e-12,"hierarchical actual-q law: {l1}/{mass}");
     (Some(l1),Some(mass))
    }else{ensure!(id==-1&&a.iter().all(|v|v.is_nan()),"invalid hierarchy propagation");(None,None)};
    checks.push(json!({"row":step,"id":id,"error":error,"baseline_error":baseline_error,"q_l1":l1,"q_mass":mass}));
   }
   use sha2::{Digest,Sha256};
   let raw=unsafe{cuda.read_device(candidateq.ptr(),candidateq.bytes())}?;
   cases.push(json!({"phase":phase,"draw":draw,"temperature":temp,"rows":checks,
      "all_rank_q_sha256":format!("{:x}",Sha256::digest(raw))}));
  }
 }
 let(mut ptr,mut r,mut n,mut ph)=(local.ptr(),rank as i32,rows as i32,0i32);
 launch!(module,"ca_sharded_categorical_fixture",[local.len().div_ceil(256)as u32,1,1],&stream;ptr,r,n,ph)?;
 for step in 0..rows {candidate[step].upload_uniforms(2026110304,step as u64)?;
  reference[step].upload_uniforms(&[uniform(2026110304,DRAFT_DOMAIN,step as u64)])?;}
 stream.join_default()?;
 let mut graphs=Vec::new();
 for variant in 0..3 {
  let mut enqueue=||->Result<()> {
   if variant==0 {
    for step in 0..rows {
     let source=unsafe{local.view(step*COLS,COLS)}?;
     unsafe{comm.gather_f32_at(&source,&mut full,step*VOCAB,COLS,&stream)?;}
     reference[step].enqueue(&full,step,1,1.,&stream)?;
     reference[step].probabilities(&full,step,1,1.,&mut referenceq,step,&stream)?;
    }
   }else{
    for step in 0..rows {candidate[step].enqueue(&local,step,&localq,step,1.,&comm,&stream)?;}
    if variant==1{unsafe{comm.gather_f64(&localq,&mut candidateq,COLS,&stream)?;}}
   }Ok(())
  };
  enqueue()?;stream.synchronize()?;
  graphs.push(stream.capture(||{for _ in 0..8{enqueue()?;}Ok(())})?);
 }
 let mut pairs=Vec::new();
 for repeat in 0..6 {
  let order=if repeat%2==0{[0,1,2]}else{[2,1,0]};let mut times=[0f32;3];
  for variant in order {
   unsafe{comm.sum_bf16(&sum_input,&mut sum_output,&stream)?;}
   for _ in 0..2{unsafe{graphs[variant].launch(&stream)?;}}stream.synchronize()?;
   let start=cuda.event()?;let end=cuda.event()?;start.record(&stream)?;
   for _ in 0..8{unsafe{graphs[variant].launch(&stream)?;}}
   end.record(&stream)?;times[variant]=end.elapsed_since(&start)?/64.;comm.health()?;
   ensure!(sum_output.download()?.iter().all(|&v|v==half::bf16::from_f32(10.).to_bits()),"mixed reduction ordering");
  }
  pairs.push(json!({"repeat":repeat,"order":order,"five_proposals_ms":times}));
 }
 let block=crate::sharded_block::probe(&cuda,&core,&localq,&candidateq,&candidate,&comm,&stream)?;
 let bundled=if std::env::var("CANADA_SCRATCH_BUNDLED_SAMPLER_SCREEN").as_deref()==Ok("1"){
  Some(crate::bundled_sampler_probe::run(&cuda,&core,&localq,&candidate,&comm,&stream)?)
 }else{None};
 let staged=if std::env::var("CANADA_SCRATCH_HIERARCHICAL_STAGING_SCREEN").as_deref()==Ok("1") {
  Some(probe_staging(&cuda,&core,&local,&mut candidate,&localq,&comm,&stream)?)
 }else{None};
 let fused=if std::env::var("CANADA_SCRATCH_FUSED_DRAFT_LOCAL_SCREEN").as_deref()==Ok("1") {
  Some(probe_fusion(&cuda,&core,&local,&mut candidate,&localq,&comm,&stream)?)
 }else{None};
 stream.synchronize()?;comm.health()?;
 Ok(json!({"rank":rank,"passed":true,"cases":cases,"pairs":pairs,"sharded_block":block,"bundled":bundled,"staged_controls":staged,"fused_local":fused,"all_peers_ready_before_gpu":true,
  "variants":["five_full_logit_gathers","five_compact_draws_plus_one_full_fp64_q_gather","five_compact_draws_q_remains_sharded"],
  "scope":"Component only: no Markov or target. Proposal law changes only by admissible tile partition rounding; actual FP64 q matches sampling. Independent per-shard and rank-choice draws. Full-q variant can retain existing target rejection unchanged."}))
}

/// Fusion screen uses the same five conditional draws and actual-q buffers.
/// All input fixtures and probability calculations execute on the GPU.
fn probe_fusion(cuda:&Cuda,core:&[u8],local:&Buffer<f32>,ordinary:&mut [Hierarchical],
 q:&Buffer<f64>,comm:&Roce,stream:&Stream)->Result<Value>{
 let module=cuda.module(core)?;let rank=comm.rank();
 let warp_q=std::env::var("CANADA_SCRATCH_WARP_DRAFT_Q_SCREEN").as_deref()==Ok("1");
 let mut candidate=(0..5).map(|_|Hierarchical::new(cuda,core,rank)).collect::<Result<Vec<_>>>()?;
 for draw in &mut *ordinary{draw.fused_local=warp_q;draw.warp_q=false;}
 for draw in &mut candidate{draw.fused_local=true;draw.warp_q=warp_q;}
 let localq=cuda.allocate::<f64>(5*COLS)?;
 let fill=|phase:i32|->Result<()>{
  let(mut p,mut r,mut rows,mut ph)=(local.ptr(),rank as i32,5i32,phase);
  launch!(module,"ca_sharded_categorical_fixture",[local.len().div_ceil(256)as u32,1,1],stream;p,r,rows,ph)
 };
 let mut results=Vec::new();
 for temperature in [1f32,0.7]{
  fill(0)?;stream.synchronize()?;
  let control=stream.capture(||{for i in 0..5{ordinary[i].enqueue(local,i,q,i,temperature,comm,stream)?;}Ok(())})?;
  let fused=stream.capture(||{for i in 0..5{candidate[i].enqueue(local,i,&localq,i,temperature,comm,stream)?;}Ok(())})?;
  let mut cases=Vec::new();
  for phase in 0..8{for trial in 0..3{
   fill(phase)?;
   for i in 0..5{
    let values=match trial{
     0=>[0.,0.],1=>[1.-1./9007199254740992.;2],
     _=>[uniform(2026110320+rank as u64,LOCAL_DOMAIN,phase as u64*5+i as u64),
          uniform(2026110320,DRAFT_DOMAIN,phase as u64*5+i as u64)]};
    ordinary[i].uniforms.upload(&values)?;candidate[i].uniforms.upload(&values)?;
   }
   stream.join_default()?;
   unsafe{control.launch(stream)?;fused.launch(stream)?;}
   stream.synchronize()?;comm.health()?;
   let a=q.download()?;let b=localq.download()?;
   ensure!(a.iter().zip(&b).all(|(a,b)|a.to_bits()==b.to_bits()),"fused local draft actual-q bits");
   let mut ids=Vec::new();let mut errors=Vec::new();
   for i in 0..5{
    let id=ordinary[i].output.download()?;let error=ordinary[i].errors.download()?;
    ensure!(id==candidate[i].output.download()?&&error==candidate[i].errors.download()?,
      "fused local draft IDs/errors");
    ensure!(ordinary[i].selected_q.download()?[0].to_bits()==candidate[i].selected_q.download()?[0].to_bits(),
      "fused local draft selected actual q");
    ids.push(id[0]);errors.push(error[0]);
   }
   cases.push(json!({"phase":phase,"trial":trial,"ids":ids,"errors":errors,"exact":true}));
  }}
  fill(0)?;stream.synchronize()?;
  let mut pairs=Vec::new();
  for repeat in 0..6{
   let order=if repeat%2==0{[0,1]}else{[1,0]};let mut ms=[0.;2];
   for variant in order{
    let graph=if variant==0{&control}else{&fused};
    for _ in 0..8{unsafe{graph.launch(stream)?;}}stream.synchronize()?;
    let start=cuda.event()?;let end=cuda.event()?;start.record(stream)?;
    for _ in 0..64{unsafe{graph.launch(stream)?;}}
    end.record(stream)?;ms[variant]=end.elapsed_since(&start)?/64.;comm.health()?;
   }
   ensure!(q.download()?.iter().zip(localq.download()?).all(|(a,b)|a.to_bits()==b.to_bits()),
     "fused local draft timed replay actual q");
   pairs.push(json!({"repeat":repeat,"order":order,"five_proposals_ms":ms}));
  }
  results.push(json!({"temperature":temperature,"cases":cases,"pairs":pairs}));
 }
 Ok(json!({"all_exact":true,"results":results,"changed_draws_per_rank":240,"warp_q":warp_q,
  "scope":if warp_q{"One FP64 tile quotient per warp instead of per token, keeping both multiplication boundaries. Fused-local sampling in both arms. Exact actual-q, selected-q, IDs and errors across changed-input captured replays, empty/invalid shards, endpoints and two temperatures. Actual four-rank transport; no model or outcome-speed claim."}else{"Three local kernels fused into one CTA; unchanged conditional tile law, FP64 CUB CDF and wire packing. Exact actual-q, selected-q, IDs and errors across changed-input captured replays, empty/invalid shards, endpoints and two temperatures. Actual four-rank transport. No Markov/target model or outcome-speed claim."}}))
}

/// Price changed CPU uniforms and a representative anchor update, as in the
/// actual Markov caller. GPU-written model/probability values remain ordinary.
fn probe_staging(cuda:&Cuda,core:&[u8],local:&Buffer<f32>,ordinary:&mut [Hierarchical],
 q:&Buffer<f64>,comm:&Roce,stream:&Stream)->Result<Value>{
 let controls=unsafe{cuda.staged_metadata_allocator()?};
 let staging=controls.metadata_staging().unwrap();
 let mut candidate=(0..5).map(|_|Hierarchical::with_uniform_allocator(cuda,core,comm.rank(),&controls))
  .collect::<Result<Vec<_>>>()?;
 let localq=cuda.allocate::<f64>(5*COLS)?;
 let mut anchors=[cuda.allocate::<i32>(1)?,controls.allocate::<i32>(1)?];
 anchors[0].upload(&[7])?;anchors[1].upload(&[7])?;
 for i in 0..5 {ordinary[i].upload_uniforms(2026110311,i as u64)?;candidate[i].upload_uniforms(2026110311,i as u64)?;}
 stream.join_default()?;unsafe{staging.flush(Some(stream))?;}
 for i in 0..5 {ordinary[i].enqueue(local,i,q,i,0.7,comm,stream)?;candidate[i].enqueue(local,i,&localq,i,0.7,comm,stream)?;}
 stream.synchronize()?;comm.health()?;
 let graph0=stream.capture(||{for i in 0..5{ordinary[i].enqueue(local,i,q,i,0.7,comm,stream)?;}Ok(())})?;
 let graph1=stream.capture(||{for i in 0..5{candidate[i].enqueue(local,i,&localq,i,0.7,comm,stream)?;}Ok(())})?;
 let mut barrier_in=cuda.allocate::<u16>(8)?;let mut barrier_out=cuda.allocate::<u16>(8)?;
 barrier_in.upload(&vec![half::bf16::from_f32((comm.rank()+1)as f32).to_bits();8])?;
 let mut pairs=Vec::new();
 for repeat in 0..12 {
  let order=if repeat%2==0{[0,1]}else{[1,0]};let mut ms=[0.;2];let mut signatures=Vec::new();
  for mode in order {
   unsafe{comm.sum_bf16(&barrier_in,&mut barrier_out,stream)?;}stream.synchronize()?;comm.health()?;
   let began=std::time::Instant::now();
   for iteration in 0..32 {
    anchors[mode].upload(&[(repeat*32+iteration)as i32])?;
    let draws=if mode==0{&mut *ordinary}else{&mut candidate};
    for(i,draw)in draws.iter_mut().enumerate(){draw.upload_uniforms(2026110312+repeat as u64,(iteration*5+i)as u64)?;}
    stream.join_default()?;
    if mode==1{unsafe{staging.flush(Some(stream))?;}}
    unsafe{if mode==0{graph0.launch(stream)?;}else{graph1.launch(stream)?;}}
    stream.synchronize()?;comm.health()?;
   }
   ms[mode]=began.elapsed().as_secs_f64()*1000./32.;
   let draws=if mode==0{&mut *ordinary}else{&mut candidate};
   let ids=draws.iter().map(|d|d.output.download()).collect::<Result<Vec<_>>>()?;
   let errors=draws.iter().map(|d|d.errors.download()).collect::<Result<Vec<_>>>()?;
   let selected=draws.iter().map(|d|d.selected_q.download()).collect::<Result<Vec<_>>>()?;
   let probabilities=if mode==0{q.download()?}else{localq.download()?};
   signatures.push((ids,errors,selected,probabilities,anchors[mode].download()?));
  }
  ensure!(signatures[0]==signatures[1],"staging changed uniforms, sampled IDs, actual q, errors or anchor");
  pairs.push(json!({"repeat":repeat,"order":order,"host_milliseconds":ms}));
 }
 Ok(json!({"all_exact":true,"pairs":pairs,"changed_blocks_per_arm":384,
  "scope":"Native caller includes six CPU uploads, default-stream ordering, optional one staged copy, graph replay and synchronization. Five actual compact draws per block. Exact final IDs/q/errors/anchor each pair. Component only; not full Markov or serving."}))
}
