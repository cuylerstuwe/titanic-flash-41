//! Pool only the two owners' canonical TP4 QKV input multiplications. Their
//! normalizers, Q-B projections, attention state and output buffers stay private.
use super::*;
use crate::cuda::Module;

const MANIFEST_PINS:[(&str,&str);2]=[
    ("fp8-split-qkv-m6-v1-pipeline-fence-v1","2f4ca71fa450a10133ea591787deff3daef8e12e687cc4e1e7d8e079b7d4db97"),
    ("fp8-split-qkv-m16-v1-pipeline-fence-v1","12b4142615a9f53bdb7da0689e9205ba3530ff7dfacdd41300e7722a7a0dbd44")
];

pub(crate) struct PairedQkv {
    cuda:Cuda,
    module:Module,
    dense:[(Primitives,usize);2],
    values:Buffer<u8>,
    scales:Buffer<u8>,
    output:Buffer<bf16>,
    alpha:Buffer<f32>,
    stream:Cell<usize>,
}
impl PairedQkv {
    pub(crate) fn new(cuda:&Cuda,core:&[u8],kernels:&Path)->Result<Self> {
        let mut dense=Vec::new();
        for(name,pin)in MANIFEST_PINS {
            use sha2::{Digest,Sha256};
            let root=kernels.join(name);let bytes=std::fs::read(root.join("manifest.json"))?;
            ensure!(format!("{:x}",Sha256::digest(bytes))==pin,"pooled QKV manifest changed");
            let lib=Primitives::open(cuda,&root.join("libcanada_primitives.so"),&root.join("bindings.json"))?;
            let index=find(&lib.bindings.entries,"gemm.dense")?;dense.push((lib,index));
        }
        let module=cuda.module(core)?;
        module.function("ca_pair_shared_quant")?;module.function("ca_pair_qkv_scatter")?;
        let mut values=cuda.allocate(16*5120)?;values.zero()?;
        let mut scales=cuda.allocate(scale_bytes(16,5120)?)?;scales.zero()?;
        let mut alpha=cuda.allocate(1)?;alpha.upload(&[1f32])?;
        Ok(Self{cuda:cuda.clone(),module,dense:dense.try_into().ok().unwrap(),values,scales,
            output:cuda.allocate(16*448)?,alpha,stream:Cell::new(0)})
    }
    fn operands<'a>(&self,linear:&'a Linear,input:&Buffer<bf16>,
        quantized:Option<(&'a Buffer<u8>,&'a Buffer<u8>)>,stream:&Stream)
        ->Result<(&'a Buffer<u8>,&'a Buffer<u8>)> {
        ensure!(self.cuda.owns_buffer(input)&&input.len()==6*5120,"paired QKV input shape/context");
        linear.guard_stream(stream)?;
        if let Some((q,s))=quantized {
            ensure!(self.cuda.owns_buffer(q)&&self.cuda.owns_buffer(s)
                &&q.len()==6*5120&&s.len()==scale_bytes(6,5120)?,"paired QKV prequantized input");
            Ok((q,s))
        }else{
            linear.quantize_legacy(input.ptr(),6,linear.values.ptr(),
                linear.scale_rows.as_ref().context("paired QKV legacy scales")?.ptr(),
                linear.scale_mma.ptr(),stream)?;
            Ok(linear.quantized_input())
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(&self,a:&Linear,ax:&Buffer<bf16>,aq:Option<(&Buffer<u8>,&Buffer<u8>)>,al:usize,
        b:&Linear,bx:&Buffer<bf16>,bq:Option<(&Buffer<u8>,&Buffer<u8>)>,bl:usize,stream:&Stream)->Result<()> {
        ensure!(matches!(al,3|6)&&matches!(bl,3|6)&&a.geometry.rows==6&&b.geometry.rows==6
            &&matches!(a.backend,Backend::B12x)&&matches!(b.backend,Backend::B12x)
            &&(a.matrix.n,a.matrix.k,a.matrix.padded_k)==(448,5120,5120)
            &&(b.matrix.n,b.matrix.k,b.matrix.padded_k)==(448,5120,5120)
            &&a.matrix.same_canonical_source(&b.matrix)
            &&a.matrix.in_context(&self.cuda)&&b.matrix.in_context(&self.cuda)
            &&self.cuda.owns_buffer(a.output())&&self.cuda.owns_buffer(b.output())
            &&a.output().ptr()!=b.output().ptr()&&self.cuda.owns_stream(stream),
            "paired QKV canonical source, private outputs, shape or context mismatch");
        let raw=stream.raw()as usize;
        ensure!(self.stream.get()==0||self.stream.get()==raw,"paired QKV has one ordered stream");
        self.stream.set(raw);
        let(aq,asc)=self.operands(a,ax,aq,stream)?;let(bq,bsc)=self.operands(b,bx,bq,stream)?;
        let rows=if al+bl==6 {6usize}else{16};
        let(mut ap,mut bp,mut asp,mut bsp)=(aq.ptr(),bq.ptr(),asc.ptr(),bsc.ptr());
        let(mut q,mut s)=(self.values.ptr(),self.scales.ptr());
        let(mut ai,mut bi,mut mi)=(al as i32,bl as i32,rows as i32);
        unsafe{self.module.function("ca_pair_shared_quant")?.launch([(rows*5120).div_ceil(256)as u32,1,1],
            [256,1,1],0,stream,&mut[(&mut ap as *mut u64).cast(),(&mut bp as *mut u64).cast(),
            (&mut asp as *mut u64).cast(),(&mut bsp as *mut u64).cast(),(&mut q as *mut u64).cast(),
            (&mut s as *mut u64).cast(),(&mut ai as *mut i32).cast(),(&mut bi as *mut i32).cast(),
            (&mut mi as *mut i32).cast()])?;}
        let values=BTreeMap::from([
            ("a_ptr",(q,"pointer")),("b_ptr",(a.matrix.values(),"pointer")),
            ("sfa_ptr",(s,"pointer")),("sfb_ptr",(a.matrix.scales(),"pointer")),
            ("c_ptr",(self.output.ptr(),"pointer")),("alpha_ptr",(self.alpha.ptr(),"pointer")),
            ("quant_c_values_ptr",(16,"pointer")),("quant_c_scale_rows_ptr",(16,"pointer")),
            ("quant_c_scale_mma_ptr",(16,"pointer")),("m",(rows as u64,"i32"))]);
        let(lib,index)=&self.dense[usize::from(rows==16)];
        let args=bind(&lib.bindings.entries[*index],&values)?;
        unsafe{lib.launch_words(*index as i32,&args,stream)?;}
        let(mut output,mut ay,mut by)=(self.output.ptr(),a.output().ptr(),b.output().ptr());
        unsafe{self.module.function("ca_pair_qkv_scatter")?.launch([((al+bl)*448).div_ceil(256)as u32,1,1],
            [256,1,1],0,stream,&mut[(&mut output as *mut u64).cast(),(&mut ay as *mut u64).cast(),
            (&mut by as *mut u64).cast(),(&mut ai as *mut i32).cast(),(&mut bi as *mut i32).cast()])?;}
        Ok(())
    }
}

#[test]
fn packaged_manifest_pins() {
    use sha2::{Digest,Sha256};
    let root=Path::new(env!("CARGO_MANIFEST_DIR")).join("runtime/kernels");
    for(name,pin)in MANIFEST_PINS {
        let raw=std::fs::read(root.join(name).join("manifest.json")).unwrap();
        assert_eq!(format!("{:x}",Sha256::digest(raw)),pin,"{name}");
    }
}

/// One canonical QKV matrix and two independent native projection objects.
/// GPU fixtures, both prequantized and BF16 paths, changing captured inputs.
pub fn native_probe(config:&Path,manifest:&Path,root:&Path,core:&Path,kernels:&Path)
    ->Result<serde_json::Value> {
    use crate::{checkpoint::Checkpoint,config::ModelConfig,layout::RankPlan,weights::Weights};
    use serde_json::json;
    let cfg=ModelConfig::parse(&std::fs::read(config)?)?;
    let cp=Checkpoint::from_manifest(manifest)?;let plan=RankPlan::build(&cp,&cfg,0,4)?;
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let bytes=std::fs::read(core)?;let module=cuda.module(&bytes)?;
    let prefix="layers.0.attn.w";
    let weights=Rc::new(Weights::load(&cuda,root,&cp,&plan,prefix)?);
    let make=|w:&Rc<Weights>|Linear::new(&cuda,&kernels.join("fp8-split-qkv-m6-v1-pipeline-fence-v1"),
        Matrix::prepare_qkv_split(&cuda,&module,&stream,w,0,0)?);
    let a=make(&weights)?;let b=make(&weights)?;
    ensure!(a.matrix.same_canonical_source(&b.matrix)&&a.matrix.scales()!=b.matrix.scales(),
        "QKV component needs same canonical bytes and independent encoded scales");
    let wrong_weights=Rc::new(Weights::load(&cuda,root,&cp,&plan,prefix)?);
    let wrong=make(&wrong_weights)?;let pool=PairedQkv::new(&cuda,&bytes,kernels)?;
    let xa=cuda.allocate::<bf16>(6*5120)?;let xb=cuda.allocate::<bf16>(6*5120)?;
    let mut epoch=cuda.allocate::<u32>(1)?;let epoch_ptr=epoch.ptr();
    let fill=|x:&Buffer<bf16>,owner:i32,pattern:i32|->Result<()> {
        let(mut dst,mut ep,mut n,mut owner,mut pattern,mut jitter)=
            (x.ptr(),epoch_ptr,x.len()as i32,owner,pattern,0i32);
        unsafe{module.function("ca_sequence_fill")?.launch([120,1,1],[256,1,1],0,&stream,
            &mut[(&mut dst as *mut u64).cast(),(&mut ep as *mut u64).cast(),
            (&mut n as *mut i32).cast(),(&mut owner as *mut i32).cast(),
            (&mut pattern as *mut i32).cast(),(&mut jitter as *mut i32).cast()])?;}Ok(())
    };
    ensure!(pool.run(&a,&xa,None,3,&wrong,&xb,None,3,&stream).is_err(),
        "QKV pool accepted a foreign canonical owner");
    let mut records=Vec::new();
    for(al,bl)in[(3,3),(3,6),(6,3),(6,6)] {
        for prequantized in [true,false] {
            let q_a=prequantized.then(||a.quantized_input());let q_b=prequantized.then(||b.quantized_input());
            let joined=||pool.run(&a,&xa,q_a,al,&b,&xb,q_b,bl,&stream);
            let separate=||->Result<()> {
                if prequantized {let(q,s)=a.quantized_input();a.run_quantized(q,s,&stream)?;
                    let(q,s)=b.quantized_input();b.run_quantized(q,s,&stream)}
                else{a.run(&xa,&stream)?;b.run(&xb,&stream)}
            };
            let mut graph=None;
            for phase in 0..3u32 {
                epoch.upload(&[phase])?;stream.join_default()?;
                fill(&xa,0,(phase%2)as i32)?;fill(&xb,1,(phase%2)as i32)?;
                a.run(&xa,&stream)?;b.run(&xb,&stream)?;stream.synchronize()?;
                let expected=[a.output().download()?,b.output().download()?]
                    .map(|v|v.into_iter().map(bf16::to_bits).collect::<Vec<_>>());
                if graph.is_none(){joined()?;stream.synchronize()?;graph=Some(stream.capture(joined)?);}
                unsafe{graph.as_ref().unwrap().launch(&stream)?;}stream.synchronize()?;
                let actual=[a.output().download()?,b.output().download()?]
                    .map(|v|v.into_iter().map(bf16::to_bits).collect::<Vec<_>>());
                ensure!(actual==expected,"native pooled QKV differs at {al}+{bl}, phase{phase}");
                records.push(json!({"live_rows":[al,bl],"prequantized":prequantized,"phase":phase,
                    "all_output_bits_exact":true,"inactive_rows_unchanged":true}));
            }
            let graphs=[stream.capture(||{for _ in 0..64{separate()?;}Ok(())})?,
                stream.capture(||{for _ in 0..64{joined()?;}Ok(())})?];
            let mut times=[Vec::new(),Vec::new()];
            for trial in 0..6 {
                for mode in if trial%2==0{[0,1]}else{[1,0]} {
                    unsafe{graphs[mode].launch(&stream)?;}stream.synchronize()?;
                    let start=cuda.event()?;let end=cuda.event()?;start.record(&stream)?;
                    unsafe{graphs[mode].launch(&stream)?;}end.record(&stream)?;
                    times[mode].push(end.elapsed_since(&start)?*1000./64.);
                }
            }
            records.last_mut().unwrap()["hot_graph_us"]=json!(times);
        }
    }
    Ok(json!({"complete":true,"rank_shard":0,"layer":0,"canonical_bytes":weights.bytes(),
        "separate_scale_encodings":true,"foreign_canonical_owner_rejected":true,"records":records,
        "scope":"Native QKV objects, GPU-only fixtures, four row pairs, two input paths and three changed captured inputs. Timings are hot component projections including row transport, not whole-model or serving throughput."}))
}
