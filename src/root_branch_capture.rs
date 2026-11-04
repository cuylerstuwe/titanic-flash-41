//! Bounded probability-only evidence for the offline branch economics gate.
//! Enabled only inside the public-fixture model probe; ordinary serving never
//! allocates this collector or writes these diagnostic tensors.
use crate::{cuda::{Cuda,Buffer},sharded_block::ShardedBlock};
use anyhow::{Result,ensure};
use std::{path::PathBuf,io::Write,collections::BTreeMap};
pub(crate) struct Capture { root:PathBuf, calls:BTreeMap<u64,usize> }
impl Capture {
 pub(crate) fn from_env()->Result<Option<Self>> {
  let Some(root)=std::env::var_os("CANADA_SCRATCH_ROOT_BRANCH_CAPTURE")else{return Ok(None)};
  ensure!(std::env::var("CANADA_SCRATCH_HIERARCHICAL_MODEL_SCREEN").as_deref()==Ok("1")
   &&crate::sampler_transport::Policy::from_env()?.local_target,
   "root branch capture requires the finite local-target public model probe");
  let root=PathBuf::from(root);ensure!(root.is_absolute(),"root evidence path must be absolute");
  Ok(Some(Self{root,calls:BTreeMap::new()}))
 }
 #[allow(clippy::too_many_arguments)]
 pub(crate) fn record(&mut self,cuda:&Cuda,p:&Buffer<f64>,q:&Buffer<f64>,
  weights:&Buffer<f64>,proposals:&Buffer<i32>,block:&ShardedBlock,
  live:usize,seed:u64,position:u64,rank:usize)->Result<()> {
  // The finite probe uses one warmup seed and three task-family seeds. Keep
  // each family's bounded sample; a global counter only records the first.
  if !self.calls.contains_key(&seed)&&self.calls.len()>=4{return Ok(())}
  let counter=self.calls.entry(seed).or_insert(0);
  let call=*counter;*counter=counter.saturating_add(1);
  if call>=32||call%4!=0{return Ok(())}
  ensure!(rank<4&&(2..=6).contains(&live)&&p.len()>=live*129280&&q.len()>=(live-1)*32320,
   "root branch capture extent");
  let directory=self.root.join(format!("rank{rank}")).join("root-branch");
  std::fs::create_dir_all(&directory)?;
  let stem=format!("seed{seed}-position{position}");let metadata=directory.join(format!("{stem}.json"));
  ensure!(!metadata.exists(),"root branch capture identity repeated");
  let pp=unsafe{cuda.read_device(p.ptr()+(rank*32320*8)as u64,32320*8)}?;
  let qp=unsafe{cuda.read_device(q.ptr(),32320*8)}?;
  // create_new prevents overwriting a prior or interrupted evidence record.
  for (name,bytes)in [("p",pp),("q",qp)] {
   let mut f=std::fs::OpenOptions::new().create_new(true).write(true)
    .open(directory.join(format!("{stem}-{name}.bin")))?;f.write_all(&bytes)?;
  }
  let mut f=std::fs::OpenOptions::new().create_new(true).write(true).open(metadata)?;
  serde_json::to_writer_pretty(&mut f,&serde_json::json!({"rank":rank,"seed":seed,"position":position,
   "rows":live,"call":call,"root_shard_elements":32320,"dtype":"little-endian-f64",
   "weights":&weights.download()?[..live],"mass":&block.global_mass.download()?[..live],
   "proposals":&proposals.download()?[..live-1],
   "scope":"Read-only actual normalized target p and hierarchical actual-q shards, after successful GPU rejection draw. Public fixture probe only. Timing is diagnostic, not performance evidence."}))?;
  Ok(())
 }
}
