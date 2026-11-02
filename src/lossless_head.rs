//! Exact BF16-bit encoding for the bandwidth-bound target vocabulary head.
//! Original weights remain the fallback for rare wide-exponent groups.
use crate::{cuda::{Buffer,Cuda,Stream},triton_ops::Operator};
use anyhow::{ensure,Result};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};

pub(crate) struct Head {
    cuda:Cuda,
    canonical:u64,
    sign_mantissa:Buffer<u8>,
    codes:Buffer<u32>,
    bases:Buffer<u8>,
    head:Operator,
    paired:Option<[Operator;2]>,
}
impl Head {
    pub(crate) fn matches_canonical(&self,cuda:&Cuda,canonical:u64,stream:&Stream)->bool {
        self.canonical==canonical&&cuda.owns_stream(stream)
            &&cuda.owns_buffer(&self.sign_mantissa)&&cuda.owns_buffer(&self.codes)&&cuda.owns_buffer(&self.bases)
    }
    pub(crate) fn new(cuda:&Cuda,canonical:u64,root:&Path,stream:&Stream)->Result<Self> {
        let mut head=Self::new_profile(cuda,canonical,root,stream,6,"202cb25339a2d37fc231d7f5c46adb36d62101e3792d2a6e9489722b477ae6d9")?;
        if std::env::var("CANADA_SCRATCH_PAIRED_TARGET_HEAD").as_deref()==Ok("1") {
            let extra=root.with_file_name("target-head-lossless-paired-v1");
            let raw=std::fs::read(extra.join("manifest.json"))?;
            ensure!(format!("{:x}",Sha256::digest(&raw))=="f70610b580a96a550b9120a8780867502b2c830c3147528a3fe9a52a9e3b6b97",
                "paired lossless head manifest changed");
            let mut operators=Vec::new();
            for rows in [9,12] {
                let op=Operator::load(cuda,&extra,&format!("lossless_head_m{rows}"))?;
                ensure!(op.entry.arguments.iter().map(|a|(a.name.as_str(),a.kind.as_str())).collect::<Vec<_>>()
                    ==vec![("X","*bf16"),("W","*bf16"),("SM","*u8"),("Codes","*u32"),("Base","*u8"),("Y","*fp32")]
                    &&op.entry.grid.rows_div==1&&op.entry.grid.x_mul==1&&op.entry.grid.x_add==0
                    &&op.entry.grid.y==1&&op.entry.grid.z==1&&op.entry.metadata.num_warps==4,
                    "paired lossless head AOT ABI");
                operators.push(op);
            }
            head.paired=Some(operators.try_into().ok().unwrap());
        }
        Ok(head)
    }
    pub(crate) fn new_draft(cuda:&Cuda,canonical:u64,root:&Path,stream:&Stream)->Result<Self> {
        ensure!(std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_LOSSLESS_SCREEN").as_deref()==Ok("1"),
            "M5 lossless head requires isolated actual-q screen");
        Self::new_profile(cuda,canonical,root,stream,5,"11318ad0d9ac049fb416f4a738a034f3a24b56b1b3bfb379af55fcc187c553d0")
    }
    fn new_profile(cuda:&Cuda,canonical:u64,root:&Path,stream:&Stream,rows:usize,pin:&str)->Result<Self> {
        const COUNT:usize=32320*5120;
        const BYTES:usize=COUNT+COUNT/2+COUNT/32;
        ensure!(cuda.owns_stream(stream) && canonical!=0 && canonical%256==0
            && cuda.admission_memory()?.admitted_available>=8*1024usize.pow(3)+BYTES+COUNT/256,
            "lossless target-head geometry/reserve");
        let raw=std::fs::read(root.join("manifest.json"))?;
        ensure!(format!("{:x}",Sha256::digest(&raw))==pin,
            "qualified lossless head manifest changed");
        let manifest:serde_json::Value=serde_json::from_slice(&raw)?;
        ensure!(manifest["format"]=="bf16-signmant8-expdelta4-escape255-v1"
            && manifest["n"]==32320 && manifest["k"]==5120 && manifest["group"]==32,
            "lossless BF16 encoding contract");
        let pack=Operator::load(cuda,root,"lossless_pack")?;
        let check=Operator::load(cuda,root,"lossless_check")?;
        let head=Operator::load(cuda,root,&format!("lossless_head_m{rows}"))?;
        for (op,args,mul)in [
            (&pack,vec![("W","*bf16"),("SM","*u8"),("Codes","*u32"),("Base","*u8")],5),
            (&check,vec![("W","*bf16"),("SM","*u8"),("Codes","*u32"),("Base","*u8"),("Errors","*u32")],5),
            (&head,vec![("X","*bf16"),("W","*bf16"),("SM","*u8"),("Codes","*u32"),("Base","*u8"),("Y","*fp32")],1)] {
            ensure!(op.entry.arguments.iter().map(|a|(a.name.as_str(),a.kind.as_str())).collect::<Vec<_>>()==args
                && op.entry.grid.rows_div==1 && op.entry.grid.x_mul==mul && op.entry.grid.x_add==0
                && op.entry.grid.y==1 && op.entry.grid.z==1 && op.entry.metadata.num_warps==4,
                "lossless head AOT ABI");
        }
        let began=Instant::now();
        let sign_mantissa=cuda.allocate::<u8>(COUNT)?;
        let codes=cuda.allocate::<u32>(COUNT/8)?;
        let bases=cuda.allocate::<u8>(COUNT/32)?;
        let errors=cuda.allocate::<u32>(COUNT/1024)?;
        stream.join_default()?;
        unsafe {
            pack.launch(32320,&[canonical,sign_mantissa.ptr(),codes.ptr(),bases.ptr()],stream)?;
            check.launch(32320,&[canonical,sign_mantissa.ptr(),codes.ptr(),bases.ptr(),errors.ptr()],stream)?;
        }
        stream.synchronize()?;
        ensure!(errors.download()?.iter().all(|&count|count==0),"lossless head changed canonical weight bits");
        eprintln!("lossless M{rows} head: {BYTES} encoded bytes, every canonical BF16 bit verified on GPU, preparation {:.3}ms",began.elapsed().as_secs_f64()*1000.);
        Ok(Self{cuda:cuda.clone(),canonical,sign_mantissa,codes,bases,head,paired:None})
    }
    pub(crate) fn run(&self,input:u64,output:u64,stream:&Stream)->Result<()> {
        ensure!(self.cuda.owns_stream(stream),"lossless head stream context");
        unsafe{self.head.launch(32320,&[input,self.canonical,self.sign_mantissa.ptr(),self.codes.ptr(),self.bases.ptr(),output],stream)}
    }
    pub(crate) fn run_pooled(&self,input:u64,output:u64,rows:usize,stream:&Stream)->Result<()> {
        ensure!(self.cuda.owns_stream(stream)&&matches!(rows,6|9|12),"pooled lossless head rows/context");
        if rows==6{return self.run(input,output,stream);}
        let ops=self.paired.as_ref().ok_or_else(||anyhow::anyhow!("paired lossless operators absent"))?;
        let op=&ops[usize::from(rows==12)];
        unsafe{op.launch(32320,&[input,self.canonical,self.sign_mantissa.ptr(),self.codes.ptr(),self.bases.ptr(),output],stream)}
    }
}
