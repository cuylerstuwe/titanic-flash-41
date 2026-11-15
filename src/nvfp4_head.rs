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
/// A kernel of the engine's own module with the same by-address parameter convention as `Head::launch`
/// (little-endian words; the driver reads each parameter's declared width).
unsafe fn launch_core(module:&Module,name:&str,values:&[u64],grid:u32,threads:u32,stream:&Stream)->Result<()> {
    let mut words=values.to_vec();
    let mut args:Vec<*mut c_void>=words.iter_mut().map(|x|(x as *mut u64).cast()).collect();
    unsafe{module.function(name)?.launch([grid,1,1],[threads,1,1],0,stream,&mut args)}
}
/// Timing probe only (CANADA_SCRATCH_DRAFT_HEAD_ROWS_PROBE=R, a multiple of 64 below 32,320): the projection runs
/// over the first R vocabulary rows of each rank's slice and the other logits stay zero. Drafts then come from a
/// wrong proposal law (sampling stays exact, acceptance falls): it measures what a draft pass costs with a head of
/// R rows a rank, which is what a script-subset head would run. Not a serving setting.
fn rows_probe()->Result<Option<usize>> {
    match std::env::var("CANADA_SCRATCH_DRAFT_HEAD_ROWS_PROBE") {
        Ok(v)=>{let rows:usize=v.parse()?;ensure!(rows>=64&&rows<N&&rows%64==0,"draft head rows probe: a multiple of 64 below {N}");Ok(Some(rows))}
        Err(std::env::VarError::NotPresent)=>Ok(None),
        Err(e)=>Err(e.into()),
    }
}
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
    subset:Option<Subset>,
    /// Which form the next `run` launches (and a capture records): the subset copy or the whole head.
    subset_mode:std::cell::Cell<bool>,
    pool:Option<Pool>,
    /// Diagnostic export only (`export_logits`): FP32 logits of the five rows, allocated on first use.
    export_out:std::cell::RefCell<Option<Buffer<f32>>>,
}
/// One pass for the five rows of two or three conversations (config::draft_head_pool): upstream's 16-row instance
/// of the same Marlin kernel (bundle draft-head-nvfp4-marlin-m16-v1: the template argument m_block_size_8 false)
/// over this head's weights, scales and global scale, with its own staging, output and reduction buffers.
struct Pool {
    module:Module,
    symbol:String,
    input:Buffer<bf16>,
    output:Buffer<bf16>,
    temporary:Buffer<f32>,
    locks:Buffer<u32>,
}
const POOL_ROWS:usize=16;
impl Pool {
    fn new(cuda:&Cuda,root:&Path,parameters:&[String])->Result<Self> {
        let raw=std::fs::read(root.join("manifest.json"))?;
        ensure!(format!("{:x}",Sha256::digest(&raw))==
            "8ec25f11ad5a613235869f21843bce940b3f724d7af06bde8edb48984236cc85",
            "pooled NVFP4 head manifest changed");
        let manifest:Manifest=serde_json::from_slice(&raw)?;
        ensure!(manifest.artifact=="head16.cubin","pooled NVFP4 artifact path");
        let artifact=std::fs::read(root.join(&manifest.artifact))?;
        ensure!(format!("{:x}",Sha256::digest(&artifact))==manifest.artifact_sha256,"pooled NVFP4 CUDA artifact changed");
        ensure!(manifest.functions.len()==1&&manifest.functions[0].name=="projection"
            &&manifest.functions[0].parameters==parameters,"pooled NVFP4 driver ABI");
        let module=cuda.module(&artifact)?;
        let symbol=manifest.functions[0].symbol.clone();
        module.function(&symbol)?.opt_in_shared_memory(SHARED)?;
        let mut locks=cuda.allocate::<u32>(SM)?;locks.zero()?;
        Ok(Self{module,symbol,input:cuda.allocate(POOL_ROWS*K)?,output:cuda.allocate(POOL_ROWS*N)?,
            temporary:cuda.allocate(SM*16*256)?,locks})
    }
}
/// The script-subset form of this rank's head (CANADA_SCRATCH_DRAFT_HEAD_SUBSET_SCREEN): the vocabulary rows of the
/// rank's slice in the order `order` (subset first), and a second packed copy of the first `run` of them.
struct Subset {
    core:Module,
    order:Buffer<u32>,
    run:usize,
    inside:usize,
    weights:Buffer<u32>,
    scales:Buffer<u8>,
}
/// Logit of a vocabulary row the subset head does not project: far below any real logit, so its probability is an
/// exact zero in every later exponential, and finite in FP16, BF16 and FP32.
const ABSENT:f32=-8192.0;
/// Bit i = vocabulary id i is in the script subset (tools/latin_subset_component.py over the model's tokenizer:
/// ASCII, Latin letters, punctuation, digits, symbols, special tokens, partial UTF-8 sequences).
static SUBSET_BITS:&[u8]=include_bytes!("latin_subset_v1.bin");
const SUBSET_SHA256:&str="dc990eae804e35ffac4bbac460de0af80e0696b512b3261ac876c3e5fe6bb314";
/// Whether a vocabulary id is in the script subset (ids outside the vocabulary are not).
pub(crate) fn in_script_subset(id:u32)->bool {
    let id=id as usize;id<4*N&&SUBSET_BITS[id>>3]>>(id&7)&1==1
}
/// This rank's vocabulary rows, subset first: (order, rows inside the subset).
fn subset_order(rank:usize)->Result<(Vec<u32>,usize)> {
    ensure!(rank<4&&SUBSET_BITS.len()==4*N/8&&format!("{:x}",Sha256::digest(SUBSET_BITS))==SUBSET_SHA256,
        "draft head subset bitmap");
    let inside=|j:usize|{let id=rank*N+j;SUBSET_BITS[id>>3]>>(id&7)&1==1};
    let mut order:Vec<u32>=(0..N).filter(|&j|inside(j)).map(|j|j as u32).collect();
    let count=order.len();
    order.extend((0..N).filter(|&j|!inside(j)).map(|j|j as u32));
    Ok((order,count))
}
impl Head {
    pub(crate) fn new(cuda:&Cuda,canonical:u64,root:&Path,stream:&Stream,core:&[u8],rank:usize)->Result<Self> {
        ensure!((std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_NVFP4_SCREEN").as_deref()==Ok("1")||crate::config::draft_head_nvfp4())
            &&cuda.owns_stream(stream)&&canonical!=0&&canonical%256==0,
            "private NVFP4 head requires isolated sampled screen");
        cuda.qualify_gb10()?;
        // Includes the one-shot transpose/repack buffer and every persistent
        // projection buffer. Existing driver allocation floors remain active.
        let subset_setting=crate::config::draft_head_subset()?;
        let subset_requested=subset_setting!=0;
        // The subset adds a transient row-ordered copy of the canonical slice and a second packed copy.
        let pool_requested=crate::config::draft_head_pool()?;
        ensure!(!pool_requested||rows_probe()?.is_none(),"the pooled draft head does not combine with the rows probe");
        let bytes=N*K+N*K/16+SM*16*256*4+M*N*2+SM*4+16+if subset_requested {N*K*2+N*K/2+N*K/16+N*4}else{0}
            +if pool_requested {POOL_ROWS*(K+N)*2+SM*16*256*4+SM*4}else{0};
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
        let pool=if pool_requested {
            let sibling=root.parent().ok_or_else(||anyhow::anyhow!("NVFP4 head bundle root"))?.join("draft-head-nvfp4-marlin-m16-v1");
            Some(Pool::new(cuda,&sibling,&entries["projection"].parameters)?)
        }else{None};
        let weights=cuda.allocate::<u32>(N*K/8)?;
        let scales=cuda.allocate::<u8>(N*K/16)?;
        let global=cuda.allocate::<f32>(1)?;
        let temporary=cuda.allocate::<f32>(SM*16*256)?;
        let mut output=cuda.allocate::<bf16>(M*N)?;
        if rows_probe()?.is_some() {output.zero()?;}
        let mut locks=cuda.allocate::<u32>(SM)?;locks.zero()?;
        let mut maximum=cuda.allocate::<f32>(1)?;maximum.zero()?;
        let mut error=cuda.allocate::<u32>(1)?;error.zero()?;
        let packed=cuda.allocate::<u32>(N*K/8)?;
        let mut result=Self{cuda:cuda.clone(),module,entries,weights,scales,global,temporary,output,locks,
            stream:stream.raw()as usize,subset:None,subset_mode:std::cell::Cell::new(subset_setting==2),pool,
            export_out:std::cell::RefCell::new(None)};
        stream.join_default()?;
        // With the subset the whole head is quantized from a row-ordered copy of the slice (subset rows first):
        // the same per-row values and scales as without it (the scale groups run along a row, the global maximum
        // is over the same elements), only the row order of the packed copy changes.
        let mut ordered=None;
        if subset_requested {
            let (order_host,inside)=subset_order(rank)?;
            let run=inside.div_ceil(64)*64;
            ensure!(run>=64&&run<=N,"draft head subset extent");
            let core=cuda.module(core)?;
            for name in ["ca_subset_gather_rows","ca_subset_take_columns","ca_subset_publish"] {core.function(name)?;}
            let mut order=cuda.allocate::<u32>(N)?;order.upload(&order_host)?;
            let copy=cuda.allocate::<u16>(N*K)?;
            unsafe{launch_core(&core,"ca_subset_gather_rows",&[canonical,copy.ptr(),order.ptr(),N as u64,K as u64],N as u32,256,stream)?;}
            let weights=cuda.allocate::<u32>(run*K/8)?;
            let scales=cuda.allocate::<u8>(run*K/16)?;
            result.subset=Some(Subset{core,order,run,inside,weights,scales});
            ordered=Some(copy);
        }
        let source=ordered.as_ref().map_or(canonical,|copy|copy.ptr());
        unsafe {
            result.launch("amax",&[source,maximum.ptr()],192,256,0,stream)?;
            result.launch("quantize",&[source,maximum.ptr(),packed.ptr(),result.scales.ptr(),
                result.global.ptr(),error.ptr()],(N*K/256)as u32,256,0,stream)?;
            result.launch("repack",&[packed.ptr(),result.weights.ptr(),K as u64,N as u64],
                SM as u32,256,SHARED,stream)?;
        }
        if let Some(subset)=&result.subset {
            // The quantizer writes packed words [K/8][N] and scales [K/16][N] (N columns a row; the scale kernel
            // permutes columns only inside tiles of 64): the subset copy is the first `run` columns of each.
            let packed_subset=cuda.allocate::<u32>(subset.run*K/8)?;
            unsafe {
                launch_core(&subset.core,"ca_subset_take_columns",&[packed.ptr(),packed_subset.ptr(),(K/8)as u64,N as u64,
                    subset.run as u64,4],(K/8)as u32,256,stream)?;
                launch_core(&subset.core,"ca_subset_take_columns",&[result.scales.ptr(),subset.scales.ptr(),(K/16)as u64,N as u64,
                    subset.run as u64,1],(K/16)as u32,256,stream)?;
                result.launch("repack",&[packed_subset.ptr(),subset.weights.ptr(),K as u64,subset.run as u64],
                    SM as u32,256,SHARED,stream)?;
            }
            stream.synchronize()?;
        }
        stream.synchronize()?;
        ensure!(error.download()?==[0],"NVFP4 positive-scale encoding range");
        eprintln!("DSpark private NVFP4 Marlin: {} encoded bytes, target unchanged",N*K/2+N*K/16+4);
        if result.subset.is_some() {result.check_subset(rank,stream)?;}
        if result.pool.is_some() {result.check_pool(rank,stream)?;}
        Ok(result)
    }
    pub(crate) fn pool_available(&self)->bool {self.pool.is_some()}
    /// One pass for several conversations' five rows each (two or three), in the form `set_subset_mode` selected:
    /// their inputs are staged in order, the 16-row instance projects them over one read of the head, and each
    /// conversation's rows are published to its own FP32 logits exactly as its own pass publishes them.
    pub(crate) fn run_pooled(&self,parts:&[(&Buffer<bf16>,&Buffer<f32>)],stream:&Stream)->Result<()> {
        let pool=self.pool.as_ref().ok_or_else(||anyhow::anyhow!("pooled NVFP4 head was not loaded"))?;
        ensure!(self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize&&(2..=POOL_ROWS/M).contains(&parts.len())
            &&parts.iter().all(|(i,o)|self.cuda.owns_buffer(i)&&self.cuda.owns_buffer(o)&&i.len()==M*K&&o.len()==M*N),
            "pooled NVFP4 context/stream/extents");
        let (rows,weights,scales)=match &self.subset {
            Some(subset) if self.subset_mode.get()=>(subset.run,subset.weights.ptr(),subset.scales.ptr()),
            _=>(N,self.weights.ptr(),self.scales.ptr()),
        };
        let m=M*parts.len();
        unsafe {
            for (k,(input,_)) in parts.iter().enumerate() {
                self.cuda.copy_device(pool.input.ptr()+(k*M*K*2)as u64,input.ptr(),M*K*2,stream)?;
            }
            let mut words=[pool.input.ptr(),weights,pool.output.ptr(),pool.temporary.ptr(),0,0,scales,self.global.ptr(),0,
                m as u64,rows as u64,K as u64,K as u64,pool.locks.ptr(),0,0,1,SHARED as u64];
            let mut args:Vec<*mut c_void>=words.iter_mut().map(|x|(x as *mut u64).cast()).collect();
            pool.module.function(&pool.symbol)?.launch([SM as u32,1,1],[128,1,1],SHARED,stream,&mut args)?;
            for (k,(_,output)) in parts.iter().enumerate() {
                let source=pool.output.ptr()+(k*M*rows*2)as u64;
                match &self.subset {
                    Some(subset)=>launch_core(&subset.core,"ca_subset_publish",&[source,output.ptr(),subset.order.ptr(),M as u64,
                        N as u64,rows as u64,ABSENT.to_bits()as u64],(M*N).div_ceil(256)as u32,256,stream)?,
                    None=>self.launch("publish",&[source,output.ptr()],(M*N).div_ceil(256)as u32,256,0,stream)?,
                }
            }
        }
        Ok(())
    }
    /// The whole head on the first `rows` rows at `input` ([rows][5120] BF16, e.g. the target's head input) through
    /// the 16-row instance: BF16 logits [rows][N] in this head's row order. Returns (logits, the order's pointer or
    /// 0 when the rows are in vocabulary order). For crate::head4_rescore.
    pub(crate) fn project_rows(&self,input:u64,rows:usize,stream:&Stream)->Result<(u64,u64)> {
        let pool=self.pool.as_ref().ok_or_else(||anyhow::anyhow!("the 16-row NVFP4 instance is not loaded (CANADA_SCRATCH_DRAFT_HEAD_POOL=1)"))?;
        ensure!(self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize&&(1..=POOL_ROWS).contains(&rows)&&input!=0,
            "NVFP4 row projection context/stream/extents");
        unsafe {
            // Staged: the instance's input is a 16-row buffer of its own.
            self.cuda.copy_device(pool.input.ptr(),input,rows*K*2,stream)?;
            let mut words=[pool.input.ptr(),self.weights.ptr(),pool.output.ptr(),pool.temporary.ptr(),0,0,self.scales.ptr(),self.global.ptr(),0,
                rows as u64,N as u64,K as u64,K as u64,pool.locks.ptr(),0,0,1,SHARED as u64];
            let mut args:Vec<*mut c_void>=words.iter_mut().map(|x|(x as *mut u64).cast()).collect();
            pool.module.function(&pool.symbol)?.launch([SM as u32,1,1],[128,1,1],SHARED,stream,&mut args)?;
        }
        Ok((pool.output.ptr(),self.subset.as_ref().map_or(0,|s|s.order.ptr())))
    }
    /// Synthetic inputs for three conversations through their own passes and through pooled passes of three and of
    /// two, in every form this head has: reports the largest difference of any published logit.
    fn check_pool(&self,rank:usize,stream:&Stream)->Result<()> {
        let initial=self.subset_mode.get();
        let mut state=0xd1b54a32d192ed03u64^rank as u64;
        let mut inputs=Vec::new();
        for _ in 0..3 {
            let mut input=self.cuda.allocate::<bf16>(M*K)?;
            input.upload(&(0..M*K).map(|_|{state^=state<<13;state^=state>>7;state^=state<<17;
                bf16::from_f32(((state>>40)as f32/(1u64<<24)as f32-0.5)*0.25)}).collect::<Vec<_>>())?;
            inputs.push(input);
        }
        let own:Vec<Buffer<f32>>=(0..3).map(|_|self.cuda.allocate::<f32>(M*N)).collect::<Result<_>>()?;
        let pooled:Vec<Buffer<f32>>=(0..3).map(|_|self.cuda.allocate::<f32>(M*N)).collect::<Result<_>>()?;
        let forms:&[bool]=if self.subset.is_some(){&[false,true]}else{&[false]};
        let mut report=Vec::new();
        for &form in forms {
            if self.subset.is_some() {self.subset_mode.set(form);}
            for (input,output) in inputs.iter().zip(&own) {self.run(input,output,stream)?;}
            stream.synchronize()?;
            let reference:Vec<Vec<f32>>=own.iter().map(|b|b.download()).collect::<Result<_>>()?;
            let mut worst=0f32;
            for count in [3usize,2] {
                let parts:Vec<(&Buffer<bf16>,&Buffer<f32>)>=inputs.iter().zip(&pooled).take(count).collect();
                self.run_pooled(&parts,stream)?;stream.synchronize()?;
                for (k,buffer) in pooled.iter().take(count).enumerate() {
                    let host=buffer.download()?;
                    for (a,b) in reference[k].iter().zip(&host) {
                        ensure!(a.is_finite()&&b.is_finite(),"pooled draft head: non-finite logit");
                        ensure!((*a==ABSENT)==(*b==ABSENT),"pooled draft head: absent rows differ from the own pass");
                        worst=worst.max((a-b).abs());
                    }
                }
            }
            let scale=reference.iter().flatten().filter(|v|**v!=ABSENT).fold(0f32,|m,v|m.max(v.abs()));
            ensure!(worst<=1e-3*scale.max(1.0),"pooled draft head differs from the own passes: {worst} of {scale}");
            report.push(format!("{} form {worst:.3e} (logit scale {scale:.2})",if form{"subset"}else{"whole"}));
        }
        self.subset_mode.set(initial);
        eprintln!("rank{rank} DSpark draft head pool: one pass for two or three conversations; largest difference from their own passes: {}",
            report.join(", "));
        Ok(())
    }
    pub(crate) fn subset_available(&self)->bool {self.subset.is_some()}
    /// Selects the form of the next pass (its caller captures or launches right after).
    pub(crate) fn set_subset_mode(&self,subset:bool)->Result<()> {
        ensure!(!subset||self.subset.is_some(),"draft head subset was not built");
        self.subset_mode.set(subset);Ok(())
    }
    /// One synthetic pass through the whole head (row order `order`, all N rows) and through the subset copy:
    /// every projected row must agree to FP32 rounding of the reduction, every other row must be ABSENT.
    fn check_subset(&self,rank:usize,stream:&Stream)->Result<()> {
        let subset=self.subset.as_ref().expect("subset built");
        let mut input=self.cuda.allocate::<bf16>(M*K)?;
        let mut state=0x9e3779b97f4a7c15u64^rank as u64;
        input.upload(&(0..M*K).map(|_|{state^=state<<13;state^=state>>7;state^=state<<17;
            bf16::from_f32(((state>>40)as f32/(1u64<<24)as f32-0.5)*0.25)}).collect::<Vec<_>>())?;
        let (whole,part)=(self.cuda.allocate::<f32>(M*N)?,self.cuda.allocate::<f32>(M*N)?);
        self.project(&input,&whole,N,stream)?;stream.synchronize()?;
        let whole_host=whole.download()?;
        self.project(&input,&part,subset.run,stream)?;stream.synchronize()?;
        let part_host=part.download()?;
        let order=subset.order.download()?;
        let (mut worst,mut absent)=(0f32,0usize);
        for row in 0..M {for (j,&id) in order.iter().enumerate() {
            let (a,b)=(whole_host[row*N+id as usize],part_host[row*N+id as usize]);
            if j<subset.run {ensure!(a.is_finite()&&b.is_finite(),"draft head subset: non-finite logit");worst=worst.max((a-b).abs());}
            else {ensure!(b==ABSENT,"draft head subset: an unprojected row was written");absent+=1;}
        }}
        let scale=whole_host.iter().fold(0f32,|m,v|m.max(v.abs()));
        ensure!(worst<=1e-3*scale.max(1.0),"draft head subset differs from the whole head: {worst} of {scale}");
        eprintln!("rank{rank} DSpark draft head subset: {} of {N} vocabulary rows in the subset, {} projected a pass, largest difference from the whole head {worst:.3e} (logit scale {scale:.2}), {} rows absent a pass",
            subset.inside,subset.run,absent/M);
        Ok(())
    }
    /// The head over the first `rows` rows of the subset order (all N = the whole head in that order), published
    /// in vocabulary order with ABSENT for the rest.
    fn project(&self,input:&Buffer<bf16>,output:&Buffer<f32>,rows:usize,stream:&Stream)->Result<()> {
        let subset=self.subset.as_ref().expect("subset built");
        ensure!(rows==N||rows==subset.run,"draft head subset rows");
        let (weights,scales)=if rows==N {(self.weights.ptr(),self.scales.ptr())}else{(subset.weights.ptr(),subset.scales.ptr())};
        unsafe {
            self.launch("projection",&[input.ptr(),weights,self.output.ptr(),
                self.temporary.ptr(),0,0,scales,self.global.ptr(),0,
                M as u64,rows as u64,K as u64,K as u64,self.locks.ptr(),0,0,1,SHARED as u64],
                SM as u32,128,SHARED,stream)?;
            launch_core(&subset.core,"ca_subset_publish",&[self.output.ptr(),output.ptr(),subset.order.ptr(),M as u64,N as u64,
                rows as u64,ABSENT.to_bits()as u64],(M*N).div_ceil(256)as u32,256,stream)
        }
    }
    /// Values obey the pinned little-endian SM121 pointer/scalar ABI; the
    /// driver reads each parameter's declared byte width from its address.
    unsafe fn launch(&self,name:&str,values:&[u64],grid:u32,threads:u32,shared:u32,stream:&Stream)->Result<()> {
        let e=&self.entries[name];ensure!(e.parameters.len()==values.len(),"NVFP4 launch arity");
        let mut words=values.to_vec();
        let mut args:Vec<*mut c_void>=words.iter_mut().map(|x|(x as *mut u64).cast()).collect();
        unsafe{self.module.function(&e.symbol)?.launch([grid,1,1],[threads,1,1],shared,stream,&mut args)}
    }
    /// Diagnostic only (CANADA_SCRATCH_CLAUDE_HEAD4_EXPORT): this rank's whole-vocabulary 4-bit logits of the first
    /// five rows of `normalized` (any [rows][5120] BF16 head input, e.g. the target's), as FP32 bytes. Uses the
    /// head's own scratch between two of the drafter's passes and a buffer of its own for the result.
    pub(crate) fn export_logits(&self,normalized:&Buffer<bf16>,stream:&Stream)->Result<Vec<u8>> {
        ensure!(normalized.len()>=M*K&&self.cuda.owns_buffer(normalized),"NVFP4 export input rows");
        let input=unsafe{normalized.view(0,M*K)}?;
        let mut slot=self.export_out.borrow_mut();
        if slot.is_none() {*slot=Some(self.cuda.allocate::<f32>(M*N)?);}
        let out=slot.as_ref().unwrap();
        let previous=self.subset_mode.get();
        if self.subset.is_some() {self.subset_mode.set(false);}
        let result=self.run(&input,out,stream);
        self.subset_mode.set(previous);
        result?;
        stream.synchronize()?;
        unsafe{self.cuda.read_device(out.ptr(),M*N*4)}
    }
    pub(crate) fn run(&self,input:&Buffer<bf16>,output:&Buffer<f32>,stream:&Stream)->Result<()> {
        ensure!(self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize
            &&self.cuda.owns_buffer(input)&&self.cuda.owns_buffer(output)
            &&input.len()==M*K&&output.len()==M*N,"NVFP4 private M5 context/stream/extents");
        if let Some(subset)=&self.subset {return self.project(input,output,if self.subset_mode.get(){subset.run}else{N},stream);}
        let rows=rows_probe()?.unwrap_or(N);
        unsafe {
            self.launch("projection",&[input.ptr(),self.weights.ptr(),self.output.ptr(),
                self.temporary.ptr(),0,0,self.scales.ptr(),self.global.ptr(),0,
                M as u64,rows as u64,K as u64,K as u64,self.locks.ptr(),0,0,1,SHARED as u64],
                SM as u32,128,SHARED,stream)?;
            self.launch("publish",&[self.output.ptr(),output.ptr()],(M*N).div_ceil(256)as u32,256,0,stream)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn subset_order_is_a_permutation_with_the_subset_first() {
        let mut total=0;
        for rank in 0..4 {
            let (order,inside)=subset_order(rank).unwrap();
            assert_eq!(order.len(),N);
            let mut seen=vec![false;N];
            for &j in &order {assert!(!seen[j as usize]);seen[j as usize]=true;}
            let bit=|j:u32|{let id=rank*N+j as usize;SUBSET_BITS[id>>3]>>(id&7)&1==1};
            assert!(order[..inside].iter().all(|&j|bit(j))&&order[inside..].iter().all(|&j|!bit(j)));
            assert!(order[..inside].windows(2).all(|w|w[0]<w[1])&&order[inside..].windows(2).all(|w|w[0]<w[1]));
            assert!(inside.div_ceil(64)*64<=N);
            total+=inside;
        }
        // the component's own counts (vocab/latin-subset-v1/result.json)
        assert_eq!(total,79213);
        assert_eq!((0..4).map(|rank|subset_order(rank).unwrap().1).collect::<Vec<_>>(),[21979,19925,18711,18598]);
        assert!(subset_order(4).is_err());
        // the guard: 128 tokens of context, all inside
        let inside:Vec<u32>=(0..4*N as u32).filter(|&id|in_script_subset(id)).take(300).collect();
        let outside=(0..4*N as u32).find(|&id|!in_script_subset(id)).unwrap();
        assert!(!in_script_subset(4*N as u32)&&!in_script_subset(u32::MAX));
        assert!(crate::config::draft_head_subset_guard(&inside,inside[0]));
        assert!(!crate::config::draft_head_subset_guard(&inside,outside));
        let mut context=inside.clone();let n=context.len();
        context[n-127]=outside;assert!(!crate::config::draft_head_subset_guard(&context,inside[0]));
        context[n-127]=inside[1];context[n-128]=outside;assert!(crate::config::draft_head_subset_guard(&context,inside[0]));
        assert!(crate::config::draft_head_subset_guard(&[],inside[0]));
    }
}
