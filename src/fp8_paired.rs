//! Private two-owner row transport and shared-up projection. Canonical matrix
//! storage and every per-owner nonlinear/down projection remain unchanged.
use super::*;
use crate::cuda::Module;

const MANIFEST_PINS:[(&str,&str,&str);2]=[
    ("fp8-shared-w13-r0-m6-fence-v1","gemm.dense",
     "72b10ef2d5defc5381847544e04ae1680bb8576745b0729881fdfa1b56d4e7f4"),
    // The portable manifest omits the original private build path. Kernel
    // artifacts, geometry, ABI and numerical contract are unchanged.
    ("fp8-flashinfer-compact-w13-m16-v1-pipeline-fence-v1","flashinfer.mxfp8.dense.pdl1",
     "042c81a731bcc12c3856e6ee12c89df497ebe4a46b09464a69b54904402a4280")
];

pub(crate) struct PairedSharedUp {
    cuda:Cuda,
    module:Module,
    dense:[(Primitives,usize);2],
    values:Buffer<u8>,
    scales:Buffer<u8>,
    output:Buffer<bf16>,
    alpha:Buffer<f32>,
    stream:Cell<usize>,
}
impl PairedSharedUp {
    pub(crate) fn new(cuda:&Cuda,core:&[u8],kernels:&Path)->Result<Self> {
        let mut dense=Vec::new();
        for (name,id,manifest_pin) in MANIFEST_PINS {
            use sha2::{Digest,Sha256};
            let root=kernels.join(name);let bytes=std::fs::read(root.join("manifest.json"))?;
            ensure!(format!("{:x}",Sha256::digest(bytes))==manifest_pin,"pooled shared-up manifest changed");
            let library=Primitives::open(cuda,&root.join("libcanada_primitives.so"),&root.join("bindings.json"))?;
            let index=find(&library.bindings.entries,id)?;dense.push((library,index));
        }
        let module=cuda.module(core)?;
        module.function("ca_pair_shared_quant")?;module.function("ca_pair_shared_scatter")?;
        let mut values=cuda.allocate(16*5120)?;values.zero()?;
        let mut scales=cuda.allocate(scale_bytes(16,5120)?)?;scales.zero()?;
        let mut alpha=cuda.allocate(1)?;alpha.upload(&[1f32])?;
        Ok(Self{cuda:cuda.clone(),module,dense:dense.try_into().ok().unwrap(),values,scales,
            output:cuda.allocate(16*1152)?,alpha,stream:Cell::new(0)})
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(&self,a:&Linear,aq:&Buffer<u8>,asc:&Buffer<u8>,al:usize,
        b:&Linear,bq:&Buffer<u8>,bsc:&Buffer<u8>,bl:usize,stream:&Stream)->Result<()> {
        ensure!(matches!(al,3|6)&&matches!(bl,3|6)&&a.geometry.rows==6&&b.geometry.rows==6
            &&matches!(a.backend,Backend::B12x)&&matches!(b.backend,Backend::B12x)
            &&(a.matrix.n,a.matrix.k,a.matrix.padded_k)==(1152,5120,5120)
            &&(b.matrix.n,b.matrix.k,b.matrix.padded_k)==(1152,5120,5120)
            &&a.matrix.same_canonical_source(&b.matrix)
            &&a.matrix.in_context(&self.cuda)&&b.matrix.in_context(&self.cuda)
            &&aq.len()==6*5120&&bq.len()==6*5120
            &&asc.len()==scale_bytes(6,5120)?&&bsc.len()==asc.len()
            &&self.cuda.owns_buffer(aq)&&self.cuda.owns_buffer(bq)
            &&self.cuda.owns_buffer(asc)&&self.cuda.owns_buffer(bsc)
            &&self.cuda.owns_buffer(a.output())&&self.cuda.owns_buffer(b.output())
            &&a.output().ptr()!=b.output().ptr()&&self.cuda.owns_stream(stream),
            "paired shared-up contract: rows={:?} legacy={:?} geometry={:?} canonical_same={} outputs_distinct={} operand_lengths={:?}",
            [a.geometry.rows,b.geometry.rows],
            [matches!(a.backend,Backend::B12x),matches!(b.backend,Backend::B12x)],
            [(a.matrix.n,a.matrix.k,a.matrix.padded_k),(b.matrix.n,b.matrix.k,b.matrix.padded_k)],
            a.matrix.same_canonical_source(&b.matrix),a.output().ptr()!=b.output().ptr(),
            [aq.len(),bq.len(),asc.len(),bsc.len()]);
        let raw=stream.raw()as usize;
        ensure!(self.stream.get()==0||self.stream.get()==raw,"paired shared-up has one ordered stream");
        self.stream.set(raw);
        let rows=if al+bl==6 {6usize}else{16};
        let(mut ap,mut bp,mut asp,mut bsp)=(aq.ptr(),bq.ptr(),asc.ptr(),bsc.ptr());
        let(mut q,mut s)=(self.values.ptr(),self.scales.ptr());
        let(mut ai,mut bi,mut mi)=(al as i32,bl as i32,rows as i32);
        unsafe{self.module.function("ca_pair_shared_quant")?.launch([(rows*5120).div_ceil(256)as u32,1,1],
            [256,1,1],0,stream,&mut [(&mut ap as *mut u64).cast(),(&mut bp as *mut u64).cast(),
            (&mut asp as *mut u64).cast(),(&mut bsp as *mut u64).cast(),(&mut q as *mut u64).cast(),
            (&mut s as *mut u64).cast(),(&mut ai as *mut i32).cast(),(&mut bi as *mut i32).cast(),
            (&mut mi as *mut i32).cast()])?;}
        let values=if rows==6 {BTreeMap::from([
            ("a_ptr",(q,"pointer")),("b_ptr",(a.matrix.values(),"pointer")),
            ("sfa_ptr",(s,"pointer")),("sfb_ptr",(a.matrix.scales(),"pointer")),
            ("c_ptr",(self.output.ptr(),"pointer")),("alpha_ptr",(self.alpha.ptr(),"pointer")),
            ("quant_c_values_ptr",(16,"pointer")),("quant_c_scale_rows_ptr",(16,"pointer")),
            ("quant_c_scale_mma_ptr",(16,"pointer")),("m",(6,"i32"))])
        }else{BTreeMap::from([
            ("x",(q,"pointer")),("w",(a.matrix.values(),"pointer")),("xs",(s,"pointer")),
            ("ws",(a.matrix.scales(),"pointer")),("y",(self.output.ptr(),"pointer")),
            ("alpha",(self.alpha.ptr(),"pointer"))])};
        let(lib,index)=&self.dense[usize::from(rows==16)];
        let args=bind(&lib.bindings.entries[*index],&values)?;
        unsafe{lib.launch_words(*index as i32,&args,stream)?;}
        let(mut output,mut ay,mut by)=(self.output.ptr(),a.output().ptr(),b.output().ptr());
        unsafe{self.module.function("ca_pair_shared_scatter")?.launch([((al+bl)*1152).div_ceil(256)as u32,1,1],
            [256,1,1],0,stream,&mut [(&mut output as *mut u64).cast(),(&mut ay as *mut u64).cast(),
            (&mut by as *mut u64).cast(),(&mut ai as *mut i32).cast(),(&mut bi as *mut i32).cast()])?;}
        Ok(())
    }
}

#[test]
fn packaged_manifest_pins() {
    use sha2::{Digest,Sha256};
    let root=Path::new(env!("CARGO_MANIFEST_DIR")).join("runtime/kernels");
    for(name,_,pin)in MANIFEST_PINS {
        let raw=std::fs::read(root.join(name).join("manifest.json")).unwrap();
        assert_eq!(format!("{:x}",Sha256::digest(raw)),pin,"{name}");
    }
}

/// Small native-object integration screen: one canonical shared expert, no
/// attention/KV, engine, TP communicator, or full model allocation.
pub fn native_probe(config:&Path,manifest:&Path,root:&Path,core:&Path,kernels:&Path)
    ->Result<serde_json::Value> {
    use crate::{checkpoint::Checkpoint,config::ModelConfig,layout::RankPlan,weights::Weights};
    use serde_json::json;
    let cfg=ModelConfig::parse(&std::fs::read(config)?)?;
    let cp=Checkpoint::from_manifest(manifest)?;
    let plan=RankPlan::build(&cp,&cfg,0,4)?;
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;let stream=cuda.stream()?;
    let bytes=std::fs::read(core)?;let module=cuda.module(&bytes)?;
    let prefix="layers.0.ffn.shared_experts.";
    let weights=Rc::new(Weights::load(&cuda,root,&cp,&plan,prefix)?);
    let make=|w:&Rc<Weights>|Linear::new(&cuda,&kernels.join("fp8-shared-w13-r0-m6-fence-v1"),
        Matrix::prepare_shared(&cuda,&module,&stream,w,0)?);
    let a=make(&weights)?;let b=make(&weights)?;
    ensure!(a.matrix.same_canonical_source(&b.matrix)&&a.matrix.scales()!=b.matrix.scales(),
        "native component requires shared canonical weights with separate scale encodings");
    let wrong_weights=Rc::new(Weights::load(&cuda,root,&cp,&plan,prefix)?);
    let wrong=make(&wrong_weights)?;
    let pool=PairedSharedUp::new(&cuda,&bytes,kernels)?;
    let xa=cuda.allocate::<bf16>(6*5120)?;let xb=cuda.allocate::<bf16>(6*5120)?;
    let mut epoch=cuda.allocate::<u32>(1)?;
    let epoch_ptr=epoch.ptr();
    let fill=|x:&Buffer<bf16>,owner:i32,pattern:i32|->Result<()> {
        let(mut dst,mut ep,mut n,mut owner,mut pattern,mut jitter)=
            (x.ptr(),epoch_ptr,x.len()as i32,owner,pattern,0i32);
        unsafe{module.function("ca_sequence_fill")?.launch([120,1,1],[256,1,1],0,&stream,
            &mut[(&mut dst as *mut u64).cast(),(&mut ep as *mut u64).cast(),
            (&mut n as *mut i32).cast(),(&mut owner as *mut i32).cast(),
            (&mut pattern as *mut i32).cast(),(&mut jitter as *mut i32).cast()])?;}Ok(())
    };
    let(aq,asc)=a.quantized_input();let(bq,bsc)=b.quantized_input();
    let(wq,wsc)=wrong.quantized_input();
    ensure!(pool.run(&a,aq,asc,3,&wrong,wq,wsc,3,&stream).is_err(),
        "pooled projection accepted a different canonical owner");
    let mut records=Vec::new();
    for(al,bl)in[(3,3),(3,6),(6,3),(6,6)] {
        let mut graph=None;
        for phase in 0..3u32 {
            epoch.upload(&[phase])?;stream.join_default()?;
            fill(&xa,0,(phase%2)as i32)?;fill(&xb,1,(phase%2)as i32)?;
            a.run(&xa,&stream)?;b.run(&xb,&stream)?;stream.synchronize()?;
            let expected=[a.output().download()?,b.output().download()?]
                .map(|row|row.into_iter().map(bf16::to_bits).collect::<Vec<_>>());
            if graph.is_none() {
                pool.run(&a,aq,asc,al,&b,bq,bsc,bl,&stream)?;stream.synchronize()?;
                graph=Some(stream.capture(||pool.run(&a,aq,asc,al,&b,bq,bsc,bl,&stream))?);
            }
            unsafe{graph.as_ref().unwrap().launch(&stream)?;}stream.synchronize()?;
            let actual=[a.output().download()?,b.output().download()?]
                .map(|row|row.into_iter().map(bf16::to_bits).collect::<Vec<_>>());
            ensure!(actual==expected,"native pooled shared-up differs at {al}+{bl}, phase{phase}");
            records.push(json!({"live_rows":[al,bl],"phase":phase,
                "all_output_bits_exact":true,"inactive_rows_unchanged":true}));
        }
    }
    Ok(json!({"complete":true,"rank_shard":0,"layer":0,"canonical_bytes":weights.bytes(),
        "separate_scale_encodings":true,"foreign_canonical_owner_rejected":true,"records":records,
        "scope":"Native Rust canonical loader, GPU scale conversion, separate M6 Linear objects and actual PairedSharedUp. Four widths, three changed GPU fixtures under captured replay. No model loaded; no CPU tensor arithmetic; not a serving speed or task-quality result."}))
}
