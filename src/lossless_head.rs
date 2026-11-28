//! Exact BF16-bit encoding for the bandwidth-bound target vocabulary head.
//! Original weights remain the fallback for rare wide-exponent groups.
use crate::{cuda::{Buffer,Cuda,Stream},triton_ops::Operator};
use anyhow::{ensure,Result};
use sha2::{Digest,Sha256};
use std::{path::Path,time::Instant};

/// The M6 operator run() launches. Serving is CANONICAL (the stock operator on the canonical backing) or, from the
/// compact installation on, COMPACT; PLANES exists only in the bit-plane screen. All three give the same logits.
pub(crate) const CANONICAL:u8=0;
pub(crate) const COMPACT:u8=1;
pub(crate) const PLANES:u8=2;
/// Launches of the bit-plane operator since the last call: the plan screen's receipt that the plane arm's captures
/// ran it and the compact arm's did not.
pub(crate) fn planes_launched()->usize { PLANES_LAUNCHED.with(|n|n.replace(0)) }
thread_local! { static PLANES_LAUNCHED:std::cell::Cell<usize>=const{std::cell::Cell::new(0)}; }
pub(crate) struct Head {
    cuda:Cuda,
    canonical:u64,
    // CANADA_SCRATCH_HEAD_COMPACT_EXCEPTIONS=1: the M6 operator that reads escaped groups from the compact table,
    // and that table. The paired M9/M12 heads always read canonical.
    compact:Option<(Operator,Buffer<u16>)>,
    // CANADA_SCRATCH_HEAD_PLANES_SCREEN=1 (plan screen only): the M6 operator on the bit-plane layout and that
    // layout's high plane (a byte a weight) and low plane (a u32 an eight-weight slice, held as bytes). Its escaped
    // slices read the same compact table.
    planes:Option<(Operator,Buffer<u8>,Buffer<u8>)>,
    // COMPACT from the compact installation on, else CANONICAL. Only the plan screens
    // (CANADA_SCRATCH_HEAD_COMPACT_SCREEN=1, CANADA_SCRATCH_HEAD_PLANES_SCREEN=1) switch it, between graph captures.
    variant:std::cell::Cell<u8>,
    sign_mantissa:Buffer<u8>,
    codes:Buffer<u32>,
    bases:Buffer<u8>,
    head:Operator,
    paired:Option<[Operator;2]>,
    paired_compact:Option<[Operator;2]>,
}
impl Head {
    pub(crate) fn matches_canonical(&self,cuda:&Cuda,canonical:u64,stream:&Stream)->bool {
        self.canonical==canonical&&cuda.owns_stream(stream)
            &&cuda.owns_buffer(&self.sign_mantissa)&&cuda.owns_buffer(&self.codes)&&cuda.owns_buffer(&self.bases)
    }
    pub(crate) fn new(cuda:&Cuda,canonical:u64,root:&Path,stream:&Stream)->Result<Self> {
        // Resolve the finite fallback once, before graph capture. run() keeps
        // one selected operator and has no cache-policy dispatch or env read.
        let weight_cg=match std::env::var("CANADA_SCRATCH_HEAD_WEIGHT_CG") {
            Err(std::env::VarError::NotPresent)=>false,
            Ok(value)if value=="0"=>false,
            Ok(value)if value=="1"=>true,
            _=>anyhow::bail!("invalid target-head weight cache policy"),
        };
        let selected=if weight_cg{root.with_file_name("target-head-lossless-weight-cg-v1")}else{root.to_path_buf()};
        let pin=if weight_cg{"d0ff14ca7341e5489a1a252dd8f9e24205fe583118ead1810e501909ba244274"}
            else{"202cb25339a2d37fc231d7f5c46adb36d62101e3792d2a6e9489722b477ae6d9"};
        let mut head=Self::new_profile(cuda,canonical,&selected,stream,6,pin)?;
        if weight_cg{eprintln!("lossless M6 target-head weight stream: global cache; metadata and all weight bits unchanged");}
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
            if crate::config::compact_paired_head() {
                let extra_compact=root.with_file_name("target-head-lossless-paired-compact-v1");
                let raw_c=std::fs::read(extra_compact.join("manifest.json"))?;
                ensure!(format!("{:x}",Sha256::digest(&raw_c))=="7f7fbc3280b388d09465f26286fee67e8eacba7b4859a954364467f05e678671",
                    "compact paired lossless head manifest changed");
                let mut ops_c=Vec::new();
                for rows in [9,12] {
                    let op=Operator::load(cuda,&extra_compact,&format!("lossless_head_m{rows}"))?;
                    ensure!(op.entry.arguments.iter().map(|a|(a.name.as_str(),a.kind.as_str())).collect::<Vec<_>>()
                        ==vec![("X","*bf16"),("W","*bf16"),("SM","*u8"),("Codes","*u32"),("Base","*u8"),("Y","*fp32")]
                        &&op.entry.grid.rows_div==1&&op.entry.grid.x_mul==1&&op.entry.grid.x_add==0
                        &&op.entry.grid.y==1&&op.entry.grid.z==1&&op.entry.metadata.num_warps==4,
                        "compact paired lossless head AOT ABI");
                    ops_c.push(op);
                }
                head.paired_compact=Some(ops_c.try_into().ok().unwrap());
            }
        }
        let compact=match std::env::var("CANADA_SCRATCH_HEAD_COMPACT_EXCEPTIONS") {
            Err(std::env::VarError::NotPresent)=>false,
            Ok(value)if value=="0"=>false,
            Ok(value)if value=="1"=>true,
            _=>anyhow::bail!("invalid compact target-head exception policy"),
        };
        if compact {
            ensure!(weight_cg,"compact target-head exceptions require the weight-cache M6 operator");
            head.install_compact(&root.with_file_name("target-head-lossless-compact-v1"),stream)?;
        }
        let planes=match std::env::var("CANADA_SCRATCH_HEAD_PLANES_SCREEN") {
            Err(std::env::VarError::NotPresent)=>false,
            Ok(value)if value=="0"=>false,
            Ok(value)if value=="1"=>true,
            _=>anyhow::bail!("invalid bit-plane target-head screen policy"),
        };
        if planes {
            ensure!(compact,"the bit-plane target-head screen requires compact target-head exceptions");
            head.install_planes(&root.with_file_name("target-head-lossless-compact-planes-v1"),stream)?;
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
            && cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+BYTES+COUNT/256,
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
        Ok(Self{cuda:cuda.clone(),canonical,compact:None,planes:None,variant:std::cell::Cell::new(CANONICAL),
            sign_mantissa,codes,bases,head,paired:None,paired_compact:None})
    }
    pub(crate) fn run(&self,input:u64,output:u64,stream:&Stream)->Result<()> {
        ensure!(self.cuda.owns_stream(stream),"lossless head stream context");
        let (operator,weights,payload,codes)=match (self.variant.get(),&self.compact,&self.planes) {
            (PLANES,Some((_,table)),Some((operator,high,low)))=>{
                PLANES_LAUNCHED.with(|n|n.set(n.get()+1));
                (operator,table.ptr(),high.ptr(),low.ptr())
            }
            (COMPACT,Some((operator,table)),_)=>(operator,table.ptr(),self.sign_mantissa.ptr(),self.codes.ptr()),
            (CANONICAL,..)=>(&self.head,self.canonical,self.sign_mantissa.ptr(),self.codes.ptr()),
            _=>anyhow::bail!("lossless head variant without its installation"),
        };
        unsafe{operator.launch(32320,&[input,weights,payload,codes,self.bases.ptr(),output],stream)}
    }
    /// Plan screens: the M6 operator for graphs captured from now on (the caller drops its graphs). The compact
    /// screen (CANADA_SCRATCH_HEAD_COMPACT_SCREEN=1) switches between CANONICAL and COMPACT, the bit-plane screen
    /// (CANADA_SCRATCH_HEAD_PLANES_SCREEN=1) between COMPACT and PLANES. All three are the same logits.
    pub(crate) fn set_variant_for_probe(&self,variant:u8)->Result<()> {
        let screen=|name:&str|std::env::var(name).as_deref()==Ok("1");
        let (compact,planes)=(screen("CANADA_SCRATCH_HEAD_COMPACT_SCREEN"),screen("CANADA_SCRATCH_HEAD_PLANES_SCREEN"));
        ensure!(self.compact.is_some()&&match variant {
            CANONICAL=>compact,
            COMPACT=>compact||planes,
            PLANES=>planes&&self.planes.is_some(),
            _=>false,
        },"head variant screen admission");
        self.variant.set(variant);Ok(())
    }
    /// Point the M6 head at a compact copy of its escaped groups instead of the 315 MiB canonical backing.
    /// Escaped groups (base 255, ~6.7K of 5.17M per rank) are the only ones whose original BF16 words the
    /// head reads; their sign/mantissa payload is otherwise unused, so it holds each eight-weight slice's
    /// byte offset into the compact table (kernels/compact_head_escapes.ptx). Built by
    /// tools/build_compact_head.py; every width this head serves is checked bit for bit before and after.
    fn install_compact(&mut self,root:&Path,stream:&Stream)->Result<()> {
        use half::bf16;
        let manifest=std::fs::read(root.join("manifest.json"))?;
        ensure!(format!("{:x}",Sha256::digest(&manifest))=="570bbb8c09e94a84843c325304ce504d977d8f6063348679e9762c11094d2304",
            "compact lossless head manifest changed");
        let alternate=Operator::load(&self.cuda,root,"lossless_head_m6")?;
        ensure!(alternate.entry.arguments.iter().map(|a|(a.name.as_str(),a.kind.as_str())).collect::<Vec<_>>()
            ==vec![("X","*bf16"),("W","*bf16"),("SM","*u8"),("Codes","*u32"),("Base","*u8"),("Y","*fp32")]
            &&alternate.entry.grid.rows_div==1&&alternate.entry.grid.x_mul==1&&alternate.entry.grid.x_add==0
            &&alternate.entry.grid.y==1&&alternate.entry.grid.z==1&&alternate.entry.metadata.num_warps==4,
            "compact lossless head AOT ABI");
        let began=Instant::now();
        let groups=self.bases.download()?.iter().enumerate()
            .filter_map(|(i,&base)|(base==255).then_some(i as u32)).collect::<Vec<_>>();
        // Offsets are (group*4+slice)*16 bytes in a u32 register: well inside 2^32 for < 2^16 groups.
        ensure!(!groups.is_empty() && groups.len()<1<<16,"unexpected compact exception extent");
        let count=groups.len();
        let spill=self.cuda.allocate::<u16>(count*32)?;
        let mut index=self.cuda.allocate::<u32>(count)?;
        index.upload(&groups)?;
        let mut input=self.cuda.allocate::<bf16>(12*5120)?;
        let mut seed=80719u32;
        let values=(0..input.len()).map(|_| {
            seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;
            bf16::from_f32((seed as i32) as f32/2147483648.)
        }).collect::<Vec<_>>();
        input.upload(&values)?;
        let output=self.cuda.allocate::<f32>(12*32320)?;
        stream.join_default()?;
        let widths=if self.paired.is_some(){&[6,9,12][..]}else{&[6][..]};
        let project=|owner:&Head,n:usize|->Result<Vec<u8>> {
            owner.run_pooled(input.ptr(),output.ptr(),n,stream)?;
            stream.synchronize()?;
            unsafe{owner.cuda.read_device(output.ptr(),n*32320*4)}
        };
        let reference=widths.iter().map(|&n|project(self,n)).collect::<Result<Vec<_>>>()?;
        let gather=self.cuda.module(include_bytes!("../kernels/compact_head_escapes.ptx"))?;
        let mut args=[self.canonical,self.sign_mantissa.ptr(),index.ptr(),spill.ptr()];
        let mut words=args.each_mut().map(|v|(v as *mut u64).cast());
        unsafe{gather.function("gather_escapes")?.launch([count as u32,1,1],[32,1,1],0,stream,&mut words)?;}
        stream.synchronize()?;
        self.compact=Some((alternate,spill));self.variant.set(COMPACT);
        for (&n,expected) in widths.iter().zip(&reference) {
            ensure!(&project(self,n)?==expected,"compact head changed M{n} logits");
        }
        eprintln!("compact_head_attachment {}",serde_json::json!({"widths":widths,"exceptions":count,
            "bytes":count*64,"checked_logits":widths.iter().sum::<usize>()*32320,"all_exact":true,
            "ms":began.elapsed().as_secs_f64()*1000.}));
        Ok(())
    }
    /// Screen only (CANADA_SCRATCH_HEAD_PLANES_SCREEN=1; checklist step 5): the M6 encoding once more in bit planes,
    /// beside the stock one, so one load can time the compact head against compact + planes.
    /// The same twelve bits a weight in another arrangement (pack_planes): the kernel gets the sign, the exponent
    /// delta and the top of the mantissa from one byte with one shift and add. Escaped slices of the high plane hold
    /// the same compact-table offsets as the stock payload (kernels/compact_head_escapes.ptx). Built by
    /// tools/build_plane_head.py. Every non-escaped weight is checked on the CPU against the canonical words, and
    /// the M6 logits bit for bit against the compact head's; the paired M9/M12 heads are untouched.
    /// This costs COUNT + COUNT/2 bytes a rank (237 MiB) on top of the stock encoding, which is why it is a screen:
    /// a serving version would have to move the paired M9/M12 kernels to planes too, or free the stock encoding.
    fn install_planes(&mut self,root:&Path,stream:&Stream)->Result<()> {
        use half::bf16;
        const COUNT:usize=32320*5120;
        // Canonical words read a piece: whole groups, 16 MiB, inside read_device's bound.
        const PIECE:usize=1<<23;
        let (table,escaped)=match &self.compact {
            Some((_,table)) if self.variant.get()==COMPACT=>(table.ptr(),table.len()/32),
            _=>anyhow::bail!("the bit-plane head needs the installed compact table"),
        };
        ensure!(self.cuda.owns_stream(stream)
            &&self.cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+COUNT+COUNT/2+4*PIECE,
            "bit-plane head screen reserve");
        let manifest=std::fs::read(root.join("manifest.json"))?;
        ensure!(format!("{:x}",Sha256::digest(&manifest))=="aa92f6b023d9c021f2624d0c8872b22f8a4e964564505d5478b76b7dca135359",
            "bit-plane lossless head manifest changed");
        let operator=Operator::load(&self.cuda,root,"lossless_head_m6")?;
        ensure!(operator.entry.arguments.iter().map(|a|(a.name.as_str(),a.kind.as_str())).collect::<Vec<_>>()
            ==vec![("X","*bf16"),("W","*bf16"),("SM","*u8"),("Codes","*u32"),("Base","*u8"),("Y","*fp32")]
            &&operator.entry.grid.rows_div==1&&operator.entry.grid.x_mul==1&&operator.entry.grid.x_add==0
            &&operator.entry.grid.y==1&&operator.entry.grid.z==1&&operator.entry.metadata.num_warps==4,
            "bit-plane lossless head AOT ABI");
        let began=Instant::now();
        let bases=self.bases.download()?;
        let groups=bases.iter().enumerate()
            .filter_map(|(i,&base)|(base==255).then_some(i as u32)).collect::<Vec<_>>();
        // The compact table's rows are the escaped groups in this order (install_compact).
        ensure!(bases.len()==COUNT/32&&!groups.is_empty()&&groups.len()==escaped,"bit-plane head exception extent");
        let mut high=self.cuda.allocate::<u8>(COUNT)?;
        let mut low=self.cuda.allocate::<u8>(COUNT/2)?;
        for start in (0..COUNT).step_by(PIECE) {
            let count=PIECE.min(COUNT-start);
            let words=unsafe{self.cuda.read_device(self.canonical+(start*2)as u64,count*2)}?;
            let (high_piece,low_piece)=pack_planes(&words,&bases[start/32..(start+count)/32])?;
            high.upload_part(start,&high_piece)?;
            low.upload_part(start/2,&low_piece)?;
        }
        let mut index=self.cuda.allocate::<u32>(escaped)?;
        index.upload(&groups)?;
        let mut input=self.cuda.allocate::<bf16>(6*5120)?;
        let mut seed=51123u32;
        let values=(0..input.len()).map(|_| {
            seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;
            bf16::from_f32((seed as i32) as f32/2147483648.)
        }).collect::<Vec<_>>();
        input.upload(&values)?;
        let output=self.cuda.allocate::<f32>(6*32320)?;
        stream.join_default()?;
        // The output is cleared first, so a launch that wrote nothing cannot pass on the other operator's logits.
        let project=|owner:&Head|->Result<Vec<u8>> {
            unsafe{owner.cuda.zero_device(output.ptr(),6*32320*4,stream)?;}
            owner.run(input.ptr(),output.ptr(),stream)?;
            stream.synchronize()?;
            unsafe{owner.cuda.read_device(output.ptr(),6*32320*4)}
        };
        let reference=project(self)?;
        ensure!(reference.iter().any(|&byte|byte!=0),"compact head wrote no logits");
        // The high plane's escaped slices get the offsets the stock payload has; the table gets the same words again.
        let gather=self.cuda.module(include_bytes!("../kernels/compact_head_escapes.ptx"))?;
        let mut args=[self.canonical,high.ptr(),index.ptr(),table];
        let mut words=args.each_mut().map(|v|(v as *mut u64).cast());
        unsafe{gather.function("gather_escapes")?.launch([escaped as u32,1,1],[32,1,1],0,stream,&mut words)?;}
        stream.synchronize()?;
        self.planes=Some((operator,high,low));
        self.variant.set(PLANES);
        ensure!(project(self)?==reference,"bit-plane head changed M6 logits");
        // A receipt that both operators run, not the measurement (that is the plan screen): sixteen samples of each,
        // a sample being sixteen launches back to back on this input, the two operators alternating.
        let mut times=[Vec::new(),Vec::new()];
        for round in 0..16 {
            let order=if round%2==0{[COMPACT,PLANES]}else{[PLANES,COMPACT]};
            for which in order {
                self.variant.set(which);
                let start=Instant::now();
                for _ in 0..16 {self.run(input.ptr(),output.ptr(),stream)?;}
                stream.synchronize()?;
                times[usize::from(which==PLANES)].push(start.elapsed().as_secs_f64()*1000./16.);
            }
        }
        // Serving stays on the compact head; only the plan screen selects the planes.
        self.variant.set(COMPACT);
        let [compact,planes]=times.map(|mut samples| {
            samples.sort_by(f64::total_cmp);
            serde_json::json!({"min":samples[0],"median":(samples[7]+samples[8])/2.})
        });
        eprintln!("plane_head_attachment {}",serde_json::json!({"exceptions":escaped,"bytes":COUNT+COUNT/2,
            "checked_weights":COUNT-escaped*32,"checked_logits":6*32320,"all_exact":true,
            "back_to_back_ms":{"compact":compact,"planes":planes},"ms":began.elapsed().as_secs_f64()*1000.}));
        Ok(())
    }
    pub(crate) fn is_compact_multirow_ready(&self) -> bool {
        self.variant.get() == COMPACT && self.compact.is_some() && self.paired_compact.is_some()
    }
    pub(crate) fn run_pooled(&self,input:u64,output:u64,rows:usize,stream:&Stream)->Result<()> {
        ensure!(self.cuda.owns_stream(stream)&&matches!(rows,6|9|12),"pooled lossless head rows/context");
        if rows==6{return self.run(input,output,stream);}
        if self.variant.get()==COMPACT {
            if let (Some((_,table)),Some(ops))=(&self.compact,&self.paired_compact) {
                let op=&ops[usize::from(rows==12)];
                return unsafe{op.launch(32320,&[input,table.ptr(),self.sign_mantissa.ptr(),self.codes.ptr(),self.bases.ptr(),output],stream)};
            }
        }
        let ops=self.paired.as_ref().ok_or_else(||anyhow::anyhow!("paired lossless operators absent"))?;
        let op=&ops[usize::from(rows==12)];
        unsafe{op.launch(32320,&[input,self.canonical,self.sign_mantissa.ptr(),self.codes.ptr(),self.bases.ptr(),output],stream)}
    }
    pub(crate) fn compact_row_tiles(live: usize, capacity: usize) -> Result<Vec<(usize, usize)>> {
        ensure!(live >= 1 && live <= capacity && capacity >= 6, "compact multirow lossless head contract");
        if live <= 6 {
            return Ok(vec![(0, 6)]);
        }
        if live <= 9 && capacity >= 9 {
            return Ok(vec![(0, 9)]);
        }
        if live <= 12 && capacity >= 12 {
            return Ok(vec![(0, 12)]);
        }
        let mut tiles = Vec::new();
        let mut done = 0usize;
        while done + 12 <= live {
            tiles.push((done, 12));
            done += 12;
        }
        let rem = live - done;
        if rem > 0 {
            let width = if rem <= 6 { 6 } else if rem <= 9 { 9 } else { 12 };
            let start = live - width;
            ensure!(start + width <= capacity, "compact head tile exceeds buffer capacity");
            tiles.push((start, width));
        }
        Ok(tiles)
    }
    /// Exact compact lossless target head for any `1 <= live <= capacity` rows (`capacity >= 6`)
    /// using compact M6/M9/M12 passes. Uses a single pass for `live <= 12` (cutting 16-row
    /// batched head reads by 25% vs BF16 M16) and overlapping tail tiles for `live > 12`
    /// without extra buffers or canonical `head.weight` access.
    pub(crate) fn run_compact_rows(&self,input:u64,output:u64,live:usize,capacity:usize,stream:&Stream)->Result<()> {
        ensure!(self.cuda.owns_stream(stream)&&self.is_compact_multirow_ready(),"compact multirow lossless head contract");
        const IN_ROW_BYTES: u64 = 5120 * 2;
        const OUT_ROW_BYTES: u64 = 32320 * 4;
        for (start, width) in Self::compact_row_tiles(live, capacity)? {
            let inp = input + (start as u64) * IN_ROW_BYTES;
            let out = output + (start as u64) * OUT_ROW_BYTES;
            self.run_pooled(inp, out, width, stream)?;
        }
        Ok(())
    }
}
/// The bit-plane arrangement of whole 32-weight groups (tools/build_plane_head.py; the reference pack_planes).
/// `words` are canonical BF16 words, little-endian; `bases` their groups' bases (255: an escaped group, whose
/// planes the kernel does not read). Returns the high plane, a byte a weight:
///     sign << 7 | ((exponent - base) & 15) << 3 | mantissa >> 4
/// and the low plane, the low four mantissa bits of an eight-weight slice in a little-endian u32 (weight i in bits
/// 4i..4i+3). Every word of a group that is not escaped is checked against plane_word, the kernel's unpack.
pub(crate) fn pack_planes(words:&[u8],bases:&[u8])->Result<(Vec<u8>,Vec<u8>)> {
    ensure!(words.len()==bases.len()*64,"bit-plane packing extent");
    let mut high=vec![0u8;bases.len()*32];
    let mut low=vec![0u8;bases.len()*16];
    for (group,&base) in bases.iter().enumerate() {
        for slice in 0..4 {
            let first=group*32+slice*8;
            let word=|i:usize|u16::from_le_bytes([words[(first+i)*2],words[(first+i)*2+1]]);
            let mut nibbles=0u32;
            for i in 0..8 {
                let word=word(i);
                let exponent=(word>>7) as u8;
                high[first+i]=((word>>8) as u8&128)|((exponent.wrapping_sub(base)&15)<<3)|((word>>4) as u8&7);
                nibbles|=u32::from(word&15)<<(4*i);
            }
            if base!=255 {
                for i in 0..8 {
                    ensure!(plane_word(high[first+i],nibbles,i,base)==word(i),"bit planes changed canonical weight bits");
                }
            }
            low[first/2..first/2+4].copy_from_slice(&nibbles.to_le_bytes());
        }
    }
    Ok((high,low))
}
/// The plane kernel's unpack of weight `position` of an eight-weight slice (exact_plane_ptx): its BF16 word.
pub(crate) fn plane_word(high:u8,low:u32,position:usize,base:u8)->u16 {
    let high=u32::from(high);
    ((((high&127)<<4)+(u32::from(base)<<7))|((low>>(4*position))&15)|((high&128)<<8)) as u16
}
#[cfg(test)]
mod tests {
    use super::{pack_planes,plane_word};
    use sha2::Digest;
    /// lossless_pack's group base (target-head-lossless-weight-cg-v1/lossless_pack-source.py).
    fn base(words:&[u16])->u8 {
        let exponents=words.iter().map(|w|(w>>7) as u8);
        let (low,high)=(exponents.clone().min().unwrap(),exponents.max().unwrap());
        if high-low>15||high==255 {255} else {low}
    }
    fn bytes(words:&[u16])->Vec<u8> { words.iter().flat_map(|w|w.to_le_bytes()).collect() }
    #[test]
    fn plane_packing_round_trip() {
        let mut seed=20261122u32;
        let mut next=||{seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;seed};
        // Groups of every kind: exponent spans 0..=15 at low, middle and high bases, zeros and denormals, an escaped
        // span of 16, a wide random group, and one holding an infinity.
        let mut words=Vec::new();
        for (low,span) in [(0u32,0u32),(0,15),(1,7),(100,15),(118,3),(127,0),(239,15),(240,14),(60,16),(0,255),(250,5)] {
            for lane in 0..32 {
                let r=next();
                let exponent=match lane {0=>low,1=>low+span,_=>low+r%(span+1)};
                words.push(((r>>16&1)<<15|exponent<<7|(r>>8&127)) as u16);
            }
        }
        let bases=words.chunks(32).map(base).collect::<Vec<_>>();
        assert_eq!(bases,[0,0,1,100,118,127,239,240,255,255,255]);
        let (high,low)=pack_planes(&bytes(&words),&bases).unwrap();
        assert_eq!((high.len(),low.len()),(words.len(),words.len()/2));
        // The reference pack_planes (torch, whose planes the kernel was verified
        // on) gives these planes for these words and bases.
        let digest=|bytes:&[u8]|format!("{:x}",sha2::Sha256::digest(bytes));
        assert_eq!(digest(&bytes(&words)),"ce650b3c775aa4a6569e9911717a4ab040b47baeade5fd6ee8dd5bb7deba3136");
        assert_eq!(digest(&high),"062bbbe639303ec9c2bcee1e98f79c72e9c96091c6a0ed0529f714ffcd558c3d");
        assert_eq!(digest(&low),"2aebf822acd787b528467cdcacc9720eaf2375b06b2ee527d8a529747ef94b4f");
        for (i,&word) in words.iter().enumerate() {
            let base=bases[i/32];
            let nibbles=u32::from_le_bytes(low[i/8*4..i/8*4+4].try_into().unwrap());
            // The stock encoding of the same word (lossless_pack): the planes regroup its bits.
            let sign_mantissa=(word&127|word>>8&128) as u8;
            let code=((word>>7) as u8).wrapping_sub(base)&15;
            assert_eq!(high[i],sign_mantissa&128|code<<3|sign_mantissa>>4&7);
            assert_eq!(nibbles>>(4*(i%8))&15,u32::from(sign_mantissa&15));
            if base!=255 {
                assert_eq!(plane_word(high[i],nibbles,i%8,base),word);
                assert_eq!(plane_word(high[i],nibbles,i%8,base),
                    u16::from(sign_mantissa&127)|u16::from(sign_mantissa&128)<<8|(u16::from(base)+u16::from(code))<<7);
            }
        }
    }
    #[test]
    fn plane_packing_known_words() {
        // -1.0 (sign 1, exponent 127, mantissa 0) and exponent 130 with mantissa 0b1011010, in a group of base 120.
        let mut words=[120u16<<7;32];
        words[3]=0xbf80;
        words[13]=130<<7|0b1011010;
        let (high,low)=pack_planes(&bytes(&words),&[120]).unwrap();
        assert_eq!(high[3],0x80|7<<3);
        assert_eq!(high[13],10<<3|0b101);
        assert_eq!(high[0],0);
        assert_eq!(low,[0,0,0,0, 0,0,0xa0,0, 0,0,0,0, 0,0,0,0]);
        assert_eq!(plane_word(high[13],u32::from_le_bytes(low[4..8].try_into().unwrap()),5,120),130<<7|0b1011010);
    }
    #[test]
    fn plane_packing_rejects_a_base_that_cannot_hold_the_group() {
        // A span of 16 under a base that claims to hold it, a base above the lowest exponent, a short extent.
        let mut words=[100u16<<7;32];
        words[31]=116<<7;
        assert!(pack_planes(&bytes(&words),&[100]).is_err());
        assert!(pack_planes(&bytes(&[100u16<<7;32]),&[101]).is_err());
        assert!(pack_planes(&bytes(&[100u16<<7;32]),&[100,100]).is_err());
        assert!(pack_planes(&bytes(&words),&[255]).is_ok());
    }
    #[test]
    fn compact_row_tiles_cover_every_live_extent_within_capacity() {
        for capacity in [6usize, 16, 32, 64] {
            for live in 1..=capacity {
                let tiles = super::Head::compact_row_tiles(live, capacity).unwrap();
                if live <= 12 && capacity >= 12 {
                    assert_eq!(tiles.len(), 1, "expected single pass for live={live}, cap={capacity}");
                }
                let mut covered = vec![false; capacity];
                for (start, width) in tiles {
                    assert!(matches!(width, 6 | 9 | 12));
                    assert!(start + width <= capacity);
                    for i in start..start + width {
                        covered[i] = true;
                    }
                }
                assert!(covered[..live].iter().all(|&c| c), "uncovered row for live={live}, cap={capacity}");
            }
        }
    }
}
