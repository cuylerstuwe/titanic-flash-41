//! Exact-law block rejection with rank-local draft q and compact residual draws.
use crate::{categorical::{uniform,RESIDUAL_DOMAIN},cuda::{Buffer,Cuda,Module,Stream},roce::Roce};
use anyhow::{Result,ensure};
use serde_json::{json,Value};
const COLS:usize=32320;const VOCAB:usize=4*COLS;
const LOCAL_RESIDUAL_DOMAIN:u64=0x485245535f4c4f43;
macro_rules! launch {
 ($m:expr,$name:expr,$grid:expr,$threads:expr,$stream:expr;$($a:ident),+$(,)?)=>{{
  let mut args=[$(std::ptr::addr_of_mut!($a).cast::<std::ffi::c_void>()),+];
  unsafe{$m.function($name)?.launch($grid,[$threads,1,1],0,$stream,&mut args)}
 }};
}
pub(crate) struct ShardedBlock {
 cuda:Cuda,module:Module,rank:usize,capacity:usize,
 pub(crate) selected:Buffer<u64>,pub(crate) local_uniforms:Buffer<f64>,
 sums:Buffer<f64>,tile:Buffer<i32>,fraction:Buffer<f64>,total:Buffer<f64>,
 packed:Buffer<f32>,gathered:Buffer<f32>,pub(crate) global_mass:Buffer<f64>,
 bundled:Buffer<f32>,bundled_gathered:Buffer<f32>,
 pub(crate) fused_local:bool,
 pub(crate) fused_probabilities:bool,
}
impl ShardedBlock {
 pub(crate) fn bind_rank(&mut self,rank:usize)->Result<()> {
  ensure!(rank<4,"sharded block rank");self.rank=rank;Ok(())
 }
 pub(crate) fn new(cuda:&Cuda,core:&[u8],rank:usize,capacity:usize)->Result<Self>{
  Self::with_controls(cuda,core,rank,capacity,cuda)
 }
 pub(crate) fn with_controls(cuda:&Cuda,core:&[u8],rank:usize,capacity:usize,controls:&Cuda)->Result<Self>{
  ensure!(rank<4&&(2..=6).contains(&capacity),"sharded rejection capacity/rank");
  let selected=controls.allocate(5)?;let local_uniforms=controls.allocate(capacity)?;
  ensure!(cuda.owns_buffer(&selected)&&cuda.owns_buffer(&local_uniforms),"sharded block control context");
  Ok(Self{cuda:cuda.clone(),module:cuda.module(core)?,rank,capacity,
   selected,local_uniforms,sums:cuda.allocate(capacity*32)?,
   tile:cuda.allocate(capacity)?,fraction:cuda.allocate(capacity)?,total:cuda.allocate(capacity)?,
   packed:cuda.allocate(capacity*4)?,gathered:cuda.allocate(capacity*16)?,global_mass:cuda.allocate(capacity)?,
   bundled:cuda.allocate(capacity*8)?,bundled_gathered:cuda.allocate(capacity*32)?,
   fused_local:crate::sampler_transport::Policy::from_env()?.fused_residual_local,
   fused_probabilities:crate::sampler_transport::Policy::from_env()?.fused_probability_residual})
 }
 pub(crate) fn uniforms(&mut self,seed:u64,position:u64,live:usize)->Result<()> {
  ensure!(live<=self.capacity&&position.checked_add(live as u64).is_some(),"sharded residual RNG extent");
  let mut values=vec![0.;self.capacity];
  for(i,v)in values[..live].iter_mut().enumerate(){*v=uniform(seed,LOCAL_RESIDUAL_DOMAIN^(self.rank as u64),position+i as u64);}
  self.local_uniforms.upload(&values)
 }
 #[allow(clippy::too_many_arguments)]
 pub(crate) fn enqueue(&mut self,p:&Buffer<f64>,q:&Buffer<f64>,proposals:&Buffer<i32>,
  accept:&Buffer<f64>,residual:&Buffer<f64>,bonus:&Buffer<i32>,weights:&Buffer<f64>,
  state:&Buffer<u32>,output:&Buffer<i32>,errors:&Buffer<u32>,live:usize,comm:&Roce,stream:&Stream)->Result<()> {
  self.enqueue_inner(p,q,proposals,accept,residual,bonus,weights,state,output,errors,live,comm,stream,None,None)
 }
 #[allow(clippy::too_many_arguments)]
 pub(crate) fn enqueue_deferred(&mut self,p:&Buffer<f64>,q:&Buffer<f64>,proposals:&Buffer<i32>,
  accept:&Buffer<f64>,residual:&Buffer<f64>,bonus:&Buffer<i32>,weights:&Buffer<f64>,
  state:&Buffer<u32>,output:&Buffer<i32>,errors:&Buffer<u32>,live:usize,comm:&Roce,stream:&Stream,
  local_bonus:&Buffer<i32>,target_errors:&Buffer<u32>,
  probabilities:Option<crate::sharded_categorical_probe::DeferredProbabilities<'_>>)->Result<()> {
  ensure!(local_bonus.len()>=live&&target_errors.len()>=live&&bonus.len()>=live
    &&self.cuda.owns_buffer(local_bonus)&&self.cuda.owns_buffer(target_errors)&&self.cuda.owns_buffer(bonus),
    "deferred target readout identity/extent");
  self.enqueue_inner(p,q,proposals,accept,residual,bonus,weights,state,output,errors,live,comm,stream,
    Some((local_bonus,target_errors)),probabilities)
 }
 #[allow(clippy::too_many_arguments)]
 fn enqueue_inner(&mut self,p:&Buffer<f64>,q:&Buffer<f64>,proposals:&Buffer<i32>,
  accept:&Buffer<f64>,residual:&Buffer<f64>,bonus:&Buffer<i32>,weights:&Buffer<f64>,
  state:&Buffer<u32>,output:&Buffer<i32>,errors:&Buffer<u32>,live:usize,comm:&Roce,stream:&Stream,
  deferred:Option<(&Buffer<i32>,&Buffer<u32>)>,
  probabilities:Option<crate::sharded_categorical_probe::DeferredProbabilities<'_>>)->Result<()> {
  ensure!(live>=2&&live<=self.capacity&&p.len()>=live*VOCAB&&q.len()>=(live-1)*COLS
   &&comm.rank()==self.rank&&self.cuda.owns_stream(stream)&&self.cuda.owns_buffer(p)&&self.cuda.owns_buffer(q),
   "sharded block owner/geometry");
  if let Some(input)=&probabilities {
   ensure!(deferred.is_some_and(|(_,e)|e.ptr()==input.errors.ptr())
     &&input.logits.len()==live*COLS&&input.maxima.len()>=live*127&&input.normalized.len()>=live*127
     &&input.errors.len()>=live&&self.cuda.owns_buffer(input.logits)&&self.cuda.owns_buffer(input.maxima)
     &&self.cuda.owns_buffer(input.normalized)&&self.cuda.owns_buffer(input.errors),
     "fused target probability ownership/geometry");
  }
  let(mut pp,mut qp,mut sp,mut props,mut wp,mut ep,mut drafts,mut rank)=
   (p.ptr(),q.ptr(),self.selected.ptr(),proposals.ptr(),weights.ptr(),errors.ptr(),(live-1)as i32,self.rank as i32);
  unsafe{self.cuda.zero_device(errors.ptr(),errors.bytes(),stream)?;}
  launch!(self.module,"ca_sharded_block_weights",[1,1,1],32,stream;pp,sp,props,wp,ep,drafts)?;
  let(mut sums,mut tiles,mut fractions,mut totals,mut u)=
   (self.sums.ptr(),self.tile.ptr(),self.fraction.ptr(),self.total.ptr(),self.local_uniforms.ptr());
  if let Some(input)=probabilities {
   let(mut x,mut maxima,mut normalized,mut te)=(input.logits.ptr(),input.maxima.ptr(),input.normalized.ptr(),input.errors.ptr());
   launch!(self.module,"ca_target_probability_residual",[32,live as u32,1],256,stream;
     x,maxima,normalized,te,pp,qp,wp,sums,ep,drafts,rank)?;
  }else{
   launch!(self.module,"ca_sharded_block_partials",[32,live as u32,1],256,stream;pp,qp,wp,sums,ep,drafts,rank)?;
  }
  let mut packed=self.packed.ptr();
  let fused=self.fused_local&&deferred.is_some();
  if fused {
   let(ids,target_errors)=deferred.unwrap();
   let(mut local_ids,mut te,mut wire)=(ids.ptr(),target_errors.ptr(),self.bundled.ptr());
   launch!(self.module,"ca_sharded_block_tile_finish_bundle",[live as u32,1,1],256,stream;
     sums,u,tiles,fractions,totals,ep,pp,qp,wp,packed,local_ids,te,wire,drafts,rank)?;
  }else{
   launch!(self.module,"ca_sharded_block_tile",[live as u32,1,1],32,stream;sums,u,tiles,fractions,totals,ep)?;
   launch!(self.module,"ca_sharded_block_local_finish",[live as u32,1,1],256,stream;
     pp,qp,wp,tiles,fractions,totals,ep,packed,drafts,rank)?;
  }
  if let Some((ids,target_errors))=deferred{
   let(mut local_ids,mut te,mut wire,mut rows)=(ids.ptr(),target_errors.ptr(),self.bundled.ptr(),live as i32);
   if !fused{launch!(self.module,"ca_sharded_block_bundle",[1,1,1],32,stream;packed,local_ids,te,wire,rows)?;}
   let source=unsafe{self.bundled.view(0,live*8)}?;
   let mut destination=unsafe{self.bundled_gathered.view(0,live*32)}?;
   unsafe{comm.gather_f32(&source,&mut destination,8,stream)?;}
   let(mut gathered,mut residuals,mut out)=(self.bundled_gathered.ptr(),self.gathered.ptr(),bonus.ptr());
   launch!(self.module,"ca_sharded_block_unbundle",[1,1,1],32,stream;gathered,residuals,out,te,rows)?;
  }else{
   let source=unsafe{self.packed.view(0,live*4)}?;
   let mut destination=unsafe{self.gathered.view(0,live*16)}?;
   unsafe{comm.gather_f32(&source,&mut destination,4,stream)?;}
  }
  let(mut gathered,mut ua,mut ur,mut b,mut st,mut out,mut mass)=
   (self.gathered.ptr(),accept.ptr(),residual.ptr(),bonus.ptr(),state.ptr(),output.ptr(),self.global_mass.ptr());
  launch!(self.module,"ca_sharded_block_decide",[1,1,1],32,stream;
      gathered,wp,props,ua,b,ur,st,out,ep,drafts,mass)
 }
}

pub(crate) fn probe(cuda:&Cuda,core:&[u8],q_local:&Buffer<f64>,q_full:&Buffer<f64>,
 draws:&[crate::hierarchical_categorical::Hierarchical],comm:&Roce,stream:&Stream)->Result<Value>{
 let rank=comm.rank();let module=cuda.module(core)?;let mut results=Vec::new();
 for live in [3usize,6] {
  let mut h=ShardedBlock::new(cuda,core,rank,6)?;
  let mut qtable=[0u64;5];let mut ids=vec![0i32;6];
  for i in 0..5{qtable[i]=draws[i].selected_q().ptr();ids[i]=draws[i].output().download()?[0];}
  h.selected.upload(&qtable)?;
  let mut proposals=cuda.allocate::<i32>(6)?;proposals.upload(&ids)?;
  let mut accept=cuda.allocate::<f64>(6)?;let mut residual=cuda.allocate::<f64>(6)?;
  let mut bonus=cuda.allocate::<i32>(6)?;bonus.upload(&vec![0;6])?;
  let mut local=cuda.allocate::<f32>(6*COLS)?;let mut full=cuda.allocate::<f32>(6*VOCAB)?;
  let mut target=crate::categorical::Categorical::new(cuda,core,6)?;let mut p=cuda.allocate::<f64>(6*VOCAB)?;
  let weights=cuda.allocate::<f64>(6)?;let state=cuda.allocate::<u32>(6)?;
  let output=cuda.allocate::<i32>(6)?;let errors=cuda.allocate::<u32>(6)?;
  let base_w=cuda.allocate::<f64>(6)?;let base_state=cuda.allocate::<u32>(6)?;
  let base_out=cuda.allocate::<i32>(6)?;let base_errors=cuda.allocate::<u32>(6)?;
  let base_sums=cuda.allocate::<f64>(6*127)?;let base_tiles=cuda.allocate::<i32>(6)?;
  let base_frac=cuda.allocate::<f64>(6)?;
  let baseline=|p:&Buffer<f64>,accept:&Buffer<f64>,residual:&Buffer<f64>,bonus:&Buffer<i32>|->Result<()> {
   unsafe{cuda.zero_device(base_errors.ptr(),base_errors.bytes(),stream)?;}
   let(mut pp,mut qp,mut props,mut w,mut err,mut d)=
    (p.ptr(),q_full.ptr(),proposals.ptr(),base_w.ptr(),base_errors.ptr(),(live-1)as i32);
   launch!(module,"ca_block_weights",[1,1,1],32,stream;pp,qp,props,w,err,d)?;
   let mut sums=base_sums.ptr();
   launch!(module,"ca_block_partials",[127,live as u32,1],256,stream;pp,qp,w,sums,err,d)?;
   let(mut ua,mut ur,mut b,mut st,mut out,mut tile,mut frac)=
    (accept.ptr(),residual.ptr(),bonus.ptr(),base_state.ptr(),base_out.ptr(),base_tiles.ptr(),base_frac.ptr());
   launch!(module,"ca_block_decide_finish_fused",[1,1,1],256,stream;
    pp,qp,w,sums,props,ua,b,ur,st,out,err,tile,frac,d)
  };
  let mut cases=Vec::new();
  for phase in 0..4i32 {
   let(mut lp,mut r,mut rows,mut ph)=(local.ptr(),rank as i32,6i32,phase);
   launch!(module,"ca_sharded_categorical_fixture",[local.len().div_ceil(256)as u32,1,1],256,stream;lp,r,rows,ph)?;
   unsafe{comm.gather_f32(&local,&mut full,COLS,stream)?;}
   let uniforms:Vec<_>=(0..live).map(|i|uniform(2026110306,crate::categorical::TARGET_DOMAIN,i as u64)).collect();
   target.upload_uniforms(&uniforms)?;stream.join_default()?;
   target.enqueue(&full,0,live,1.,stream)?;target.probabilities(&full,0,live,1.,&mut p,0,stream)?;
   stream.synchronize()?;bonus.upload(&target.device_output().download()?)?;
   for trial in 0..3 {
    let a:Vec<_>=(0..6).map(|i|if trial==0{0.}else if trial==1{1.-1./9007199254740992.}else{uniform(2026110306,crate::categorical::ACCEPT_DOMAIN,i)}).collect();
    let u:Vec<_>=(0..6).map(|i|uniform(2026110307+trial,RESIDUAL_DOMAIN,i)).collect();
    accept.upload(&a)?;residual.upload(&u)?;h.uniforms(2026110307+trial,0,live)?;stream.join_default()?;
    baseline(&p,&accept,&residual,&bonus)?;h.enqueue(&p,q_local,&proposals,&accept,&residual,&bonus,&weights,&state,&output,&errors,live,comm,stream)?;
    stream.synchronize()?;comm.health()?;
    ensure!(base_errors.download()?[..live].iter().all(|&x|x==0)&&errors.download()?[..live].iter().all(|&x|x==0),"sharded block error");
    ensure!(base_w.download()?[..live]==weights.download()?[..live]&&base_state.download()?[..live]==state.download()?[..live],"sharded block weight/acceptance");
    let masses=h.global_mass.download()?;let reference=base_sums.download()?;
    let max_mass_error=(0..live).map(|i|(masses[i]-reference[i*127..(i+1)*127].iter().sum::<f64>()).abs()).fold(0.,f64::max);
    ensure!(max_mass_error<2e-13,"sharded block residual mass");
    let ids=output.download()?;let states=state.download()?;
    ensure!(ids[..live].iter().all(|&id|(0..VOCAB as i32).contains(&id)),"sharded block output range");
    cases.push(json!({"phase":phase,"trial":trial,"ids":&ids[..live],"states":&states[..live],"max_mass_error":max_mass_error}));
   }
  }
  let control=stream.capture(||{for _ in 0..8{baseline(&p,&accept,&residual,&bonus)?;}Ok(())})?;
  let candidate=stream.capture(||{for _ in 0..8{h.enqueue(&p,q_local,&proposals,&accept,&residual,&bonus,&weights,&state,&output,&errors,live,comm,stream)?;}Ok(())})?;
  let mut pairs=Vec::new();
  for repeat in 0..6 {
   let mut ms=[0f32;2];let order=if repeat%2==0{[0,1]}else{[1,0]};
   for variant in order {
    let graph=if variant==0{&control}else{&candidate};
    for _ in 0..2{unsafe{graph.launch(stream)?;}}stream.synchronize()?;
    let start=cuda.event()?;let end=cuda.event()?;start.record(stream)?;
    for _ in 0..8{unsafe{graph.launch(stream)?;}}
    end.record(stream)?;ms[variant]=end.elapsed_since(&start)?/64.;comm.health()?;
   }
   pairs.push(json!({"repeat":repeat,"order":order,"milliseconds":ms}));
  }
  results.push(json!({"live":live,"cases":cases,"pairs":pairs}));
 }
 Ok(json!({"passed":true,"results":results,"scope":"Target p and proposal q held fixed; same block recurrence/acceptance with rank-local residual draws. Full p preparation excluded from both. Actual compact RoCE exchange included; target arithmetic unchanged."}))
}
