//! Rust-owned target-feature projection for the three DSpark stages.
//! Canonical main_proj -> main_norm -> each stage's replicated WKV. This
//! component does not yet write draft KV or execute the draft query network.
use crate::{
    blockscaled::Matrix,
    checkpoint::{Checkpoint, Dtype},
    config::ModelConfig,
    cuda::{Buffer, Cuda, Graph, Stream},
    fp8_linear::Linear,
    layout::RankPlan,
    prelude::Native,
    roce::Roce,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    rc::Rc,
};

struct ConditioningReplay {
    // Destroy the executable before its owned input or the other captured
    // projection workspaces. External feature allocations are never captured.
    graph: Graph,
    input: Buffer<bf16>,
    comm_instance: Option<u64>,
}
pub struct Conditioning {
    replay: Option<ConditioningReplay>,
    cuda: Cuda,
    rows: usize,
    stream: usize,
    main: Linear,
    gathered: Option<Buffer<bf16>>,
    rank: Option<usize>,
    norm: Native,
    norm_weight: Buffer<bf16>,
    normalized: Buffer<bf16>,
    kv: [Linear; 3],
    reuse_kv_input: bool,
    live: usize,
}
impl Conditioning {
    pub fn new(
        cuda: &Cuda,
        core: &[u8],
        stream: &Stream,
        weights: Rc<Weights>,
        kernels: &Path,
        rows: usize,
    ) -> Result<Self> {
        Self::new_inner(cuda,core,stream,weights,kernels,rows,None)
    }
    pub(crate) fn new_split(
        cuda:&Cuda,core:&[u8],stream:&Stream,weights:Rc<Weights>,kernels:&Path,rank:usize,
    )->Result<Self>{
        Self::new_inner(cuda,core,stream,weights,kernels,6,Some(rank))
    }
    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        cuda:&Cuda,core:&[u8],stream:&Stream,weights:Rc<Weights>,kernels:&Path,rows:usize,rank:Option<usize>,
    )->Result<Self>{
        ensure!(
            [6, 128].contains(&rows) && cuda.owns_stream(stream) && weights.in_context(cuda),
            "DSpark conditioning geometry/context"
        );
        stream.join_default()?;
        let module = cuda.module(core)?;
        ensure!(rank.is_none_or(|r|r<4&&rows==6),"split conditioning geometry");
        let main = if let Some(rank)=rank { Linear::new(cuda,
            &kernels.join("fp8-dspark-main-split-m6-v1"),
            Matrix::prepare_draft_main_split(cuda,&module,stream,&weights,rank)?)?
        }else{Linear::new(
            cuda,
            &kernels.join(format!("fp8-dspark-main-m{rows}-v1")),
            Matrix::prepare(cuda, &module, stream, &weights, "mtp.0.main_proj")?,
        )?};
        let weight = weights.view("mtp.0.main_norm.weight")?;
        ensure!(
            weight.dtype == Dtype::Bf16 && weight.shape == [5120],
            "DSpark main norm geometry"
        );
        let norm_weight = cuda.allocate::<bf16>(5120)?;
        unsafe {
            cuda.copy_device(
                norm_weight.ptr(),
                weights.ptr("mtp.0.main_norm.weight")?,
                norm_weight.bytes(),
                stream,
            )
        }?;
        let mut kv = Vec::new();
        for stage in 0..3 {
            kv.push(Linear::new(
                cuda,
                &kernels.join(format!("fp8-dspark-kv-m{rows}-v1")),
                Matrix::prepare(
                    cuda,
                    &module,
                    stream,
                    &weights,
                    &format!("mtp.{stage}.attn.wkv"),
                )?,
            )?);
        }
        Ok(Self {
            replay: None,
            cuda: cuda.clone(),
            rows,
            stream: stream.raw() as usize,
            main,
            gathered:if rank.is_some(){Some(cuda.allocate(rows*5120)?)}else{None},
            rank,
            norm: Native::open(cuda, &kernels.join("prelude-v1"))?,
            norm_weight,
            normalized: cuda.allocate(rows * 5120)?,
            kv: kv.try_into().ok().unwrap(),
            reuse_kv_input: match std::env::var("CANADA_SCRATCH_CONDITIONING_KV_REUSE").as_deref() {
                Err(std::env::VarError::NotPresent) | Ok("1") => true,
                Ok("0") => false,
                _ => anyhow::bail!("CANADA_SCRATCH_CONDITIONING_KV_REUSE must be 0 or 1"),
            },
            live: 0,
        })
    }
    pub fn run(&mut self, features: &Buffer<bf16>, live: usize, stream: &Stream) -> Result<()> {
        self.run_distributed(features,live,None,stream)
    }
    pub(crate) fn set_kv_input_reuse_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_CONDITIONING_KV_REUSE_NATIVE_SCREEN").as_deref()==Ok("1")
            && self.cuda.owns_stream(stream) && self.stream==stream.raw()as usize,
            "KV input reuse switch requires the isolated native screen");
        stream.synchronize()?;
        if self.reuse_kv_input!=enabled{self.replay=None;self.reuse_kv_input=enabled;}
        Ok(())
    }
    pub(crate) fn run_distributed(
        &mut self,features:&Buffer<bf16>,live:usize,comm:Option<&Roce>,stream:&Stream,
    )->Result<()>{
        self.live = 0;
        self.validate_input(features,live,comm,stream)?;
        self.main.run(features, stream)?;
        if let Some(gathered)=&mut self.gathered {
            unsafe {comm.unwrap().gather_bf16(self.main.output(),gathered,1280,stream)?;}
        }
        self.norm.rms(
            self.gathered.as_ref().unwrap_or_else(||self.main.output()),
            &self.norm_weight,
            &mut self.normalized,
            self.rows,
            5120,
            stream,
        )?;
        self.kv[0].run(&self.normalized,stream)?;
        for kv in &self.kv[1..] {
            if self.reuse_kv_input {
                // All three stages use the same activation and the same
                // pinned fp8-dspark-kv-m{rows} quantizer/layout. Only weights
                // differ. Reuse exact produced bytes, with no new rounding.
                let (values,scales)=self.kv[0].quantized_input();
                kv.run_quantized(values,scales,stream)?;
            }else{kv.run(&self.normalized,stream)?;}
        }
        self.live = live;
        Ok(())
    }
    fn validate_input(
        &self,features:&Buffer<bf16>,live:usize,comm:Option<&Roce>,stream:&Stream,
    )->Result<()>{
        ensure!(
            live > 0
                && live <= self.rows
                && features.len() == self.rows * 15360
                && self.cuda.owns_buffer(features)
                && self.cuda.owns_stream(stream)
                && self.stream == stream.raw() as usize,
            "DSpark conditioning input/live/stream"
        );
        ensure!(self.rank.is_none_or(|r|comm.is_some_and(|c|c.rank()==r)),
            "split conditioning requires its TP4 communicator");
        Ok(())
    }
    pub(crate) fn run_cached_distributed(
        &mut self,features:&Buffer<bf16>,live:usize,comm:Option<&Roce>,stream:&Stream,
    )->Result<()>{
        self.live=0;
        self.validate_input(features,live,comm,stream)?;
        ensure!(self.rows==6 || (self.rows==128
            && (std::env::var("CANADA_SCRATCH_COPY_CONDITIONING_SCREEN").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_COPY_WIDE_CONDITIONING").as_deref()==Ok("1"))),
            "wide conditioning replay requires explicit copy admission");
        let comm_instance=if self.rank.is_some(){comm.map(Roce::instance)}else{None};
        if let Some(replay)=&self.replay {
            ensure!(replay.comm_instance==comm_instance,"conditioning replay communicator changed");
            // The input copy is deliberately outside capture, so arbitrary
            // request-owned feature buffers can feed the same executable.
            unsafe {
                self.cuda.copy_device(replay.input.ptr(),features.ptr(),features.bytes(),stream)?;
                replay.graph.launch(stream)?;
            }
        }else{
            let input=self.cuda.allocate::<bf16>(features.len())?;
            unsafe{self.cuda.copy_device(input.ptr(),features.ptr(),features.bytes(),stream)?;}
            self.run_distributed(&input,live,comm,stream)?;
            stream.synchronize()?;
            if let Some(c)=comm {c.health()?;}
            let graph=stream.capture(||self.run_distributed(&input,live,comm,stream))?;
            self.replay=Some(ConditioningReplay{graph,input,comm_instance});
            // Eager warmup already produced this call's output. Capture does
            // not execute or advance the device-side collective counters.
        }
        self.live = live;
        Ok(())
    }
    fn projected(&self)->&Buffer<bf16>{self.gathered.as_ref().unwrap_or_else(||self.main.output())}
    pub(crate) fn release_replay(&mut self,stream:&Stream)->Result<()> {
        ensure!(self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize,
            "conditioning replay release stream");
        stream.synchronize()?;self.replay=None;self.live=0;Ok(())
    }
    pub fn outputs(&self) -> Result<(&Buffer<bf16>, [&Buffer<bf16>; 3], usize)> {
        ensure!(self.live > 0, "DSpark conditioning output not ready");
        Ok((
            &self.normalized,
            self.kv.each_ref().map(|x| x.output()),
            self.live,
        ))
    }
}

/// Small TP4 component screen: existing split M6 chunks versus the existing
/// replicated M128 projection. Only accepted feature rows are copied; all other
/// rows are zero. This does not mutate the live drafter or target state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn probe_copy_width(
    cuda:&Cuda,core:&[u8],stream:&Stream,weights:Rc<Weights>,kernels:&Path,
    rank:usize,features:&Buffer<bf16>,comm:&Roce,
)->Result<serde_json::Value>{
    use serde_json::json;
    ensure!(std::env::var("CANADA_SCRATCH_COPY_CONDITIONING_SCREEN").as_deref()==Ok("1")
        && features.len()==128*15360 && comm.rank()==rank,"copy conditioning screen admission");
    let mut narrow=Conditioning::new_split(cuda,core,stream,weights.clone(),kernels,rank)?;
    let mut wide=Conditioning::new(cuda,core,stream,weights,kernels,128)?;
    let chunk=cuda.allocate::<bf16>(6*15360)?;
    let padded=cuda.allocate::<bf16>(128*15360)?;
    let output=cuda.allocate::<bf16>(32*512*3)?;
    let eviction=cuda.allocate::<u8>(128*1024*1024)?;
    let mut barrier_in=cuda.allocate::<f32>(4)?;barrier_in.upload(&[0.;4])?;
    let mut barrier_out=cuda.allocate::<f32>(16)?;stream.join_default()?;
    let mut execute=|mode:usize,live:usize,offset:usize|->Result<()> {
        ensure!(live>0&&live<=32&&offset+live<=128,"copy conditioning extent");
        unsafe{cuda.zero_device(output.ptr(),output.bytes(),stream)?;}
        if mode==0 {
            for begin in (0..live).step_by(6) {
                let n=(live-begin).min(6);
                unsafe{cuda.zero_device(chunk.ptr(),chunk.bytes(),stream)?;
                    cuda.copy_device(chunk.ptr(),features.ptr()+((offset+begin)*15360*2)as u64,n*15360*2,stream)?;}
                narrow.run_cached_distributed(&chunk,n,Some(comm),stream)?;
                let(_,kv,_)=narrow.outputs()?;
                for (stage,raw)in kv.into_iter().enumerate(){unsafe{
                    cuda.copy_device(output.ptr()+((stage*32+begin)*512*2)as u64,raw.ptr(),n*512*2,stream)?;}}
            }
        }else{
            unsafe{cuda.zero_device(padded.ptr(),padded.bytes(),stream)?;
                cuda.copy_device(padded.ptr(),features.ptr()+(offset*15360*2)as u64,live*15360*2,stream)?;}
            wide.run_cached_distributed(&padded,live,Some(comm),stream)?;
            let(_,kv,_)=wide.outputs()?;
            for(stage,raw)in kv.into_iter().enumerate(){unsafe{
                cuda.copy_device(output.ptr()+(stage*32*512*2)as u64,raw.ptr(),live*512*2,stream)?;}}
        }Ok(())
    };
    // First-use capture is excluded from the paired timings.
    for mode in 0..2 {execute(mode,32,0)?;stream.synchronize()?;comm.health()?;}
    let read=||->Result<Vec<u8>>{unsafe{cuda.read_device(output.ptr(),output.bytes())}};
    let mut records=Vec::new();
    for (live,offset)in[(7,0),(12,0),(16,0),(24,0),(32,0),(32,32)] {
        let mut expected=Vec::new();
        for mode in 0..2 {
            execute(mode,live,offset)?;stream.synchronize()?;comm.health()?;
            let raw=read()?;
            ensure!(raw.chunks_exact(2).all(|x|u16::from_le_bytes([x[0],x[1]])&0x7f80!=0x7f80),
                "copy conditioning produced nonfinite KV");
            expected.push(raw);
        }
        let mut pairs=Vec::new();
        for cold in [false,true] {for repeat in 0..2 {
            let mut ms=[0.;2];let mut wall=[0.;2];
            for mode in if repeat==0{[0,1]}else{[1,0]} {
                let replays=if cold{1}else{8};
                if cold{unsafe{cuda.zero_device(eviction.ptr(),eviction.bytes(),stream)?;}}
                else{execute(mode,live,offset)?;}
                unsafe{comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)?;}
                stream.synchronize()?;comm.health()?;
                let begin=cuda.event()?;let end=cuda.event()?;let start=std::time::Instant::now();
                begin.record(stream)?;for _ in 0..replays{execute(mode,live,offset)?;}end.record(stream)?;
                ms[mode]=f64::from(end.elapsed_since(&begin)?)/replays as f64;
                wall[mode]=start.elapsed().as_secs_f64()*1000./replays as f64;comm.health()?;
                ensure!(read()?==expected[mode],"copy conditioning replay changed output");
            }
            pairs.push(json!({"cold":cold,"repeat":repeat,"timeline_ms":ms,"wall_ms":wall}));
        }}
        records.push(json!({"live":live,"feature_offset":offset,"pairs":pairs,
            "kv_sha256":expected.iter().map(|x|format!("{:x}",Sha256::digest(x))).collect::<Vec<_>>(),
            "different_bf16_values":expected[0].chunks_exact(2).zip(expected[1].chunks_exact(2)).filter(|(a,b)|a!=b).count(),
            "replay_exact":true,"finite":true}));
    }
    Ok(json!({"rank":rank,"records":records,"all_replays_exact":true,
        "scope":"Accepted-only real features, split cached M6 chunks versus replicated cached M128 projection. Includes padding/input copies and KV output copies, excludes actual draft KV commit. Six small row/offset cases, hot8-replay and128MiB displaced single calls in opposite orders. Different width arithmetic is reported, not assumed equivalent. No serving change or generation-speed claim."}))
}

/// Eager versus cached split conditioning, including stable-input copy cost.
#[allow(clippy::too_many_arguments)]
pub(crate) fn probe_graph(
    cuda:&Cuda,core:&[u8],stream:&Stream,weights:Rc<Weights>,kernels:&Path,rank:usize,
    features:&Buffer<bf16>,comm:&Roce,
)->Result<serde_json::Value>{
    ensure!(std::env::var("CANADA_SCRATCH_CONDITIONING_GRAPH_SCREEN").as_deref()==Ok("1")
        && features.len()==128*15360 && comm.rank()==rank,"conditioning graph isolated screen");
    if std::env::var("CANADA_SCRATCH_CONDITIONING_KV_REUSE_SCREEN").as_deref()==Ok("1") {
        return probe_kv_reuse(cuda,core,stream,weights,kernels,rank,features,comm);
    }
    let replicated=std::env::var("CANADA_SCRATCH_CONDITIONING_GRAPH_REPLICATED").as_deref()==Ok("1");
    let build=|weights|if replicated {Conditioning::new(cuda,core,stream,weights,kernels,6)}
        else{Conditioning::new_split(cuda,core,stream,weights,kernels,rank)};
    let mut control=build(weights.clone())?;
    let mut candidate=build(weights)?;
    let inputs=[cuda.allocate::<bf16>(6*15360)?,cuda.allocate::<bf16>(6*15360)?];
    let mut barrier_in=cuda.allocate::<f32>(4)?;barrier_in.upload(&[0.;4])?;
    let mut barrier_out=cuda.allocate::<f32>(16)?;
    stream.join_default()?;
    let read=|c:&Conditioning|->Result<Vec<u8>>{
        let (norm,kv,_)=c.outputs()?;
        let mut raw=Vec::new();
        for b in std::iter::once(c.projected()).chain(std::iter::once(norm)).chain(kv){
            raw.extend(unsafe{cuda.read_device(b.ptr(),b.bytes())}?);
        }
        Ok(raw)
    };
    unsafe{cuda.copy_device(inputs[0].ptr(),features.ptr(),inputs[0].bytes(),stream)?;}
    control.run_distributed(&inputs[0],6,Some(comm),stream)?;
    stream.synchronize()?;comm.health()?;
    let first=std::time::Instant::now();
    candidate.run_cached_distributed(&inputs[0],6,Some(comm),stream)?;
    stream.synchronize()?;comm.health()?;
    let first_use_ms=first.elapsed().as_secs_f64()*1000.;
    ensure!(read(&control)?==read(&candidate)?,"conditioning first-use output changed");
    let mut records=Vec::new();
    for (index,(case,offset,live,zero))in[("real",0,6,false),("changed",6,6,false),
        ("partial",0,1,false),("zero",0,6,true)].into_iter().enumerate(){
        // Alternate external input allocations as well as contents/live rows.
        let input=&inputs[index%2];
        unsafe{cuda.zero_device(input.ptr(),input.bytes(),stream)?;}
        if !zero {unsafe{cuda.copy_device(input.ptr(),features.ptr()+(offset*15360*2)as u64,live*15360*2,stream)?;}}
        control.run_distributed(input,live,Some(comm),stream)?;
        candidate.run_cached_distributed(input,live,Some(comm),stream)?;
        stream.synchronize()?;comm.health()?;
        let expected=read(&control)?;
        ensure!(expected==read(&candidate)?&&candidate.outputs()?.2==live,"conditioning graph changed {case}");
        let digest=format!("{:x}",Sha256::digest(&expected));
        let mut pairs=Vec::new();
        for repeat in 0..2 {
            let mut times=[0.;2];let mut wall=[0.;2];
            for slot in 0..2 {
                let mode=(slot+repeat)%2;
                unsafe{comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)?;}
                stream.synchronize()?;comm.health()?;
                let begin=cuda.event()?;let end=cuda.event()?;
                let start=std::time::Instant::now();begin.record(stream)?;
                for _ in 0..16 {
                    if mode==0 {control.run_distributed(input,live,Some(comm),stream)?;}
                    else{candidate.run_cached_distributed(input,live,Some(comm),stream)?;}
                }
                end.record(stream)?;
                times[mode]=f64::from(end.elapsed_since(&begin)?)/16.;
                wall[mode]=start.elapsed().as_secs_f64()*1000./16.;comm.health()?;
            }
            ensure!(read(&control)?==expected&&read(&candidate)?==expected,"conditioning repeated output changed {case}");
            pairs.push(serde_json::json!({"repeat":repeat,"replays":16,"timeline_ms":times,"wall_ms":wall}));
        }
        records.push(serde_json::json!({"case":case,"live":live,"sha256":digest,"all_exact":true,"pairs":pairs}));
    }
    Ok(serde_json::json!({"rank":rank,"conditioning_graph_screen":true,"all_exact":true,
        "replicated":replicated,
        "first_use_ms":first_use_ms,"records":records,
        "scope":"Eager projection/norm/three KV projections versus cached CUDA graph, with a gather only for the split variant. Includes owned-input copy. Four inputs, two opposite-order pairs of sixteen calls. First capture separate; no generation claim."}))
}

#[allow(clippy::too_many_arguments)]
fn probe_kv_reuse(cuda:&Cuda,core:&[u8],stream:&Stream,weights:Rc<Weights>,kernels:&Path,
    rank:usize,features:&Buffer<bf16>,comm:&Roce)->Result<serde_json::Value>{
    let read=|c:&Conditioning|->Result<Vec<u8>>{
        let(norm,kv,_)=c.outputs()?;let mut raw=Vec::new();
        for b in std::iter::once(c.projected()).chain(std::iter::once(norm)).chain(kv){
            raw.extend(unsafe{cuda.read_device(b.ptr(),b.bytes())}?);
        }Ok(raw)
    };
    let mut barrier_in=cuda.allocate::<f32>(4)?;barrier_in.upload(&[0.;4])?;
    let mut barrier_out=cuda.allocate::<f32>(16)?;let mut records=Vec::new();
    for rows in [6,128] {
        let build=|weights|if rows==6{Conditioning::new_split(cuda,core,stream,weights,kernels,rank)}
            else{Conditioning::new(cuda,core,stream,weights,kernels,rows)};
        let mut control=build(weights.clone())?;let mut candidate=build(weights.clone())?;
        control.reuse_kv_input=false;
        candidate.reuse_kv_input=true;
        let input=cuda.allocate::<bf16>(rows*15360)?;
        unsafe{cuda.copy_device(input.ptr(),features.ptr(),input.bytes(),stream)?;}
        control.run_distributed(&input,rows,Some(comm),stream)?;
        candidate.run_distributed(&input,rows,Some(comm),stream)?;
        stream.synchronize()?;comm.health()?;
        let graphs=[stream.capture(||control.run_distributed(&input,rows,Some(comm),stream))?,
            stream.capture(||candidate.run_distributed(&input,rows,Some(comm),stream))?];
        for (case,partial)in[("real",false),("partial",true)] {
            let live=if partial{rows/2+1}else{rows};
            unsafe{cuda.zero_device(input.ptr(),input.bytes(),stream)?;
                cuda.copy_device(input.ptr(),features.ptr(),live*15360*2,stream)?;}
            control.run_distributed(&input,live,Some(comm),stream)?;
            candidate.run_distributed(&input,live,Some(comm),stream)?;
            stream.synchronize()?;comm.health()?;
            let expected=read(&control)?;ensure!(read(&candidate)?==expected,"KV reuse changed {rows}/{case}");
            // Check the bytes being shared against each independent quantizer,
            // including the MMA-layout scale padding that the GEMMs consume.
            let(q,s)=candidate.kv[0].quantized_input();
            let qbytes=unsafe{cuda.read_device(q.ptr(),q.bytes())}?;
            let sbytes=unsafe{cuda.read_device(s.ptr(),s.bytes())}?;
            for kv in &control.kv {
                let(cq,cs)=kv.quantized_input();
                ensure!(unsafe{cuda.read_device(cq.ptr(),cq.bytes())}?==qbytes
                    &&unsafe{cuda.read_device(cs.ptr(),cs.bytes())}?==sbytes,"conditioning quantizer bytes differ");
            }
            let mut pairs=Vec::new();
            for repeat in 0..2 {
                let mut times=[0.;2];
                for mode in if repeat==0{[0,1]}else{[1,0]} {
                    unsafe{graphs[mode].launch(stream)?;
                        comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)?;}
                    stream.synchronize()?;
                    let start=cuda.event()?;let end=cuda.event()?;start.record(stream)?;
                    for _ in 0..16{unsafe{graphs[mode].launch(stream)?;}}
                    end.record(stream)?;times[mode]=f64::from(end.elapsed_since(&start)?)/16.;comm.health()?;
                    ensure!(read(if mode==0{&control}else{&candidate})?==expected,"KV reuse graph changed {rows}/{case}");
                }
                pairs.push(serde_json::json!({"repeat":repeat,"replays":16,"timeline_ms":times}));
            }
            records.push(serde_json::json!({"rows":rows,"case":case,"live":live,"all_exact":true,
                "sha256":format!("{:x}",Sha256::digest(expected)),"quantizer_bytes_exact":true,"pairs":pairs}));
        }
        drop(graphs);
    }
    stream.synchronize()?;comm.health()?;
    Ok(serde_json::json!({"rank":rank,"conditioning_kv_reuse_screen":true,"all_exact":true,"records":records,
        "scope":"Reuse the first KV projection's exact MXFP8 activation and MMA scale bytes for the other two. Six-row split and128-row replicated conditioning; real/partial inputs; two reversed-order16-replay graph pairs. No extra input copy or target precision change."}))
}

/// Paired projection+gather+norm+KV graphs. All math remains native GPU work;
/// host comparisons are exact byte checks, with no scalar model oracle.
#[allow(clippy::too_many_arguments)]
pub(crate) fn probe_split(
    cuda:&Cuda,core:&[u8],stream:&Stream,weights:Rc<Weights>,kernels:&Path,rank:usize,
    features:&Buffer<bf16>,comm:&Roce,
)->Result<serde_json::Value>{
    ensure!(std::env::var("CANADA_SCRATCH_CONDITIONING_SPLIT_SCREEN").as_deref()==Ok("1")
        && features.len()==128*15360 && comm.rank()==rank,"conditioning split isolated screen");
    let mut control=Conditioning::new(cuda,core,stream,weights.clone(),kernels,6)?;
    let mut candidate=Conditioning::new_split(cuda,core,stream,weights,kernels,rank)?;
    let mut input=cuda.allocate::<bf16>(6*15360)?;
    let eviction=cuda.allocate::<u8>(128*1024*1024)?;
    let mut barrier_in=cuda.allocate::<f32>(4)?;barrier_in.upload(&[0.;4])?;
    let mut barrier_out=cuda.allocate::<f32>(16)?;
    let read=|c:&Conditioning|->Result<Vec<u8>>{
        let (norm,kv,_)=c.outputs()?;
        let mut raw=Vec::new();
        for b in std::iter::once(c.projected()).chain(std::iter::once(norm)).chain(kv){
            raw.extend(unsafe{cuda.read_device(b.ptr(),b.bytes())}?);
        }
        Ok(raw)
    };
    // Compile/capture once, then change inputs and live-row counts underneath
    // the same owned pointers. Projection math always consumes physical M6.
    input.zero()?;stream.join_default()?;
    control.run(&input,6,stream)?;candidate.run_distributed(&input,6,Some(comm),stream)?;
    stream.synchronize()?;comm.health()?;
    let graphs=[stream.capture(||control.run(&input,6,stream))?,
        stream.capture(||candidate.run_distributed(&input,6,Some(comm),stream))?];
    let mut records=Vec::new();
    for (case,offset,live)in[("real",0,6),("changed",6,6),("partial",0,1),("zero",0,0)]{
        unsafe{cuda.zero_device(input.ptr(),input.bytes(),stream)?;}
        if live>0{unsafe{cuda.copy_device(input.ptr(),features.ptr()+(offset*15360*2)as u64,live*15360*2,stream)?;}}
        for graph in &graphs{unsafe{graph.launch(stream)?;}}
        stream.synchronize()?;comm.health()?;
        let expected=read(&control)?;let actual=read(&candidate)?;
        ensure!(expected==actual,"conditioning split changed {case} projected/norm/KV values");
        let digest=format!("{:x}",Sha256::digest(&expected));
        for cold in [false,true]{
            let mut pairs=Vec::new();
            for repeat in 0..5{
                let mut times=[0.;2];
                for slot in 0..2{
                    let mode=(slot+repeat)%2;
                    if cold{unsafe{cuda.zero_device(eviction.ptr(),eviction.bytes(),stream)?;}}
                    else{for _ in 0..3{unsafe{graphs[mode].launch(stream)?;}}}
                    unsafe{comm.gather_f32(&barrier_in,&mut barrier_out,4,stream)?;}
                    stream.synchronize()?;comm.health()?;
                    let begin=cuda.event()?;let end=cuda.event()?;
                    begin.record(stream)?;unsafe{graphs[mode].launch(stream)?;}end.record(stream)?;
                    times[mode]=f64::from(end.elapsed_since(&begin)?);comm.health()?;
                }
                ensure!(read(&control)?==expected&&read(&candidate)?==expected,"conditioning replay changed {case}");
                pairs.push(serde_json::json!({"repeat":repeat,"warmup":repeat==0,"graph_ms":times}));
            }
            records.push(serde_json::json!({"case":case,"cache":if cold{"evicted128MiB"}else{"hot"},
                "sha256":digest,"all_exact":true,"pairs":pairs}));
        }
    }
    Ok(serde_json::json!({"rank":rank,"conditioning_split_screen":true,"all_exact":true,"records":records,
        "scope":"TP4 native conditioning graph including main projection, bit-preserving RoCE gather, norm and three KV projections. Four saved-feature cases, hot/cache-evicted, one warmup/four balanced pairs. No target generation or outcome-speed qualification."}))
}

/// Byte I/O and comparison only; all expected numeric values are GPU-produced.
pub fn fixture(path: &Path) -> Result<serde_json::Value> {
    #[derive(Deserialize)]
    struct Case {
        input: PathBuf,
        live: usize,
        projected: PathBuf,
        normalized: PathBuf,
        kv: [PathBuf; 3],
    }
    #[derive(Deserialize)]
    struct Input {
        config: PathBuf,
        checkpoint: PathBuf,
        model: PathBuf,
        core: PathBuf,
        kernels: PathBuf,
        rows: usize,
        cases: Vec<Case>,
    }
    let p: Input = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(
        !p.cases.is_empty() && p.cases.len() <= 8,
        "DSpark conditioning cases"
    );
    let cfg = ModelConfig::parse(&std::fs::read(p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let plan = RankPlan::build(&cp, &cfg, 0, 4)?;
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let stream = cuda.stream()?;
    let weights = Rc::new(Weights::load_prefixes(
        &cuda,
        &p.model,
        &cp,
        &plan,
        &[
            "mtp.0.main_proj.",
            "mtp.0.main_norm.",
            "mtp.0.attn.wkv.",
            "mtp.1.attn.wkv.",
            "mtp.2.attn.wkv.",
        ],
    )?);
    let weight_report = weights.qualify(&cp, &p.model, &plan)?;
    let mut state = Conditioning::new(
        &cuda,
        &std::fs::read(&p.core)?,
        &stream,
        weights,
        &p.kernels,
        p.rows,
    )?;
    let mut input = cuda.allocate::<bf16>(p.rows * 15360)?;
    let mut records = Vec::new();
    for case in p.cases {
        let bytes = std::fs::read(&case.input)?;
        ensure!(
            bytes.len() == input.bytes(),
            "DSpark conditioning feature bytes"
        );
        let words: Vec<_> = bytes
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
            .collect();
        input.upload(&words)?;
        stream.join_default()?;
        state.run(&input, case.live, &stream)?;
        stream.synchronize()?;
        let expected: Vec<Vec<u8>> = std::iter::once(case.projected)
            .chain(std::iter::once(case.normalized))
            .chain(case.kv)
            .map(std::fs::read)
            .collect::<std::io::Result<_>>()?;
        let read = |state: &Conditioning| -> Result<Vec<Vec<u8>>> {
            let (norm, kvs, live) = state.outputs()?;
            std::iter::once((state.main.output(), 5120))
                .chain(std::iter::once((norm, 5120)))
                .chain(kvs.into_iter().map(|b| (b, 512)))
                .map(|(b, width)| unsafe { cuda.read_device(b.ptr(), live * width * 2) })
                .collect()
        };
        let actual = read(&state)?;
        ensure!(
            expected
                .iter()
                .zip([5120, 5120, 512, 512, 512])
                .all(|(b, d)| b.len() == case.live * d * 2),
            "DSpark conditioning expected size"
        );
        let graph = stream.capture(|| state.run(&input, case.live, &stream))?;
        let begin = cuda.event()?;
        let end = cuda.event()?;
        begin.record(&stream)?;
        for _ in 0..50 {
            unsafe { graph.launch(&stream) }?;
        }
        end.record(&stream)?;
        let graph_ms = end.elapsed_since(&begin)? / 50.;
        let replay = read(&state)?;
        let comparisons:Vec<_>=expected.iter().zip(actual.iter().zip(&replay)).enumerate().map(|(i,(e,(a,r)))|
            serde_json::json!({"stage":i,"bytes":e.len(),"exact":e==a,"replay_exact":e==r,
                "actual_sha256":format!("{:x}",Sha256::digest(a)),"expected_sha256":format!("{:x}",Sha256::digest(e)),
                "different_bf16_values":e.chunks_exact(2).zip(a.chunks_exact(2)).filter(|(e,a)|e!=a).count()})).collect();
        records.push(serde_json::json!({"rows":p.rows,"live":case.live,"input_sha256":format!("{:x}",Sha256::digest(bytes)),
            "graph_ms":graph_ms,"replays":50,"comparisons":comparisons,
            "all_exact":comparisons.iter().all(|v|v["exact"]==true&&v["replay_exact"]==true)}));
    }
    Ok(
        serde_json::json!({"all_exact":records.iter().all(|v|v["all_exact"]==true),"records":records,"weights":weight_report,
        "full_engine_acceptance":false,"draft_model_implemented":false,
        "scope":"Replicated native-quant target feature projection/main RMSNorm/three raw WKV projections only. Expected numeric outputs computed on GPU. Draft KV store, query stages, sampling and integrated performance remain open."}),
    )
}
