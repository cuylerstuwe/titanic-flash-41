//! Optional private E4M3 DSpark head. Target weights and target logits retain
//! their own numerical contract; rejection uses the resulting actual draft q.
use crate::{
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, Stream},
    triton_ops::Operator,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{cell::Cell,path::Path, rc::Rc};

const N: usize = 32320;
const K: usize = 5120;
const M: usize = 5;

#[path = "draft_head_pair.rs"]
mod pair;
pub(crate) use pair::Pair;

pub(crate) struct DraftHead {
    cuda: Cuda,
    canonical: Rc<Weights>,
    projection: Operator,
    gemv: Option<Operator>,
    gemv_active: Cell<bool>,
    integer: Option<IntegerHead>,
    integer_active: Cell<bool>,
    lossless: Option<crate::lossless_head::Head>,
    lossless_active: Cell<bool>,
    // Immutable after construction. Request owners may share these bytes,
    // while precision switches, operators and every activation stay private.
    weights: Rc<Buffer<u8>>,
    scales: Rc<Buffer<u8>>,
    quantizer_sha256: String,
    stream: usize,
}
struct IntegerHead { projection:Operator, weights:Buffer<u8>, scales:Buffer<f32> }

/// GPU qualification only: canonical staged atlas, independent GPU fixtures,
/// raw-byte hashes and changed-input captured replay. No CPU model arithmetic.
pub fn qualify(path: &Path) -> Result<serde_json::Value> {
    use crate::{
        checkpoint::Checkpoint,
        config::ModelConfig,
        layer_probe::{Tap, fixture},
        layout::RankPlan,
    };
    use serde::Deserialize;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::path::PathBuf;
    #[derive(Deserialize)]
    struct Input {
        config: PathBuf,
        checkpoint: PathBuf,
        staged: PathBuf,
        kernels: PathBuf,
        fixture: PathBuf,
        rank: usize,
    }
    #[derive(Deserialize)]
    struct Case {
        name: String,
        input: Tap,
        output: Tap,
    }
    #[derive(Deserialize)]
    struct Expected {
        rank: usize,
        quantized_sha256: String,
        scales_sha256: String,
        canonical_sha256: String,
        cases: Vec<Case>,
    }
    let p: Input = serde_json::from_slice(&std::fs::read(path)?)?;
    let expected: Expected = serde_json::from_slice(&std::fs::read(p.fixture.join("index.json"))?)?;
    ensure!(
        p.rank < 4 && expected.rank == p.rank && expected.cases.len() == 4,
        "draft-head fixture rank/cases"
    );
    let config = ModelConfig::parse(&std::fs::read(p.config)?)?;
    let cp = Checkpoint::from_manifest(&p.checkpoint)?;
    let plan = RankPlan::build(&cp, &config, p.rank, 4)?;
    let cuda = Cuda::open(0)?;
    let stream = cuda.stream()?;
    let weights = Rc::new(Weights::load_staged(
        &cuda,
        &cp,
        &plan,
        &["embed.", "head.", "norm."],
        &p.staged,
    )?);
    let head = DraftHead::new(&cuda, weights.clone(), &p.kernels, &stream)?;
    let mut quantized_digest = Sha256::new();
    // Diagnostic hashing respects the same bounded host staging contract as
    // atlas loading. This private shard is larger than one permitted transfer.
    for offset in (0..head.weights.bytes()).step_by(16 * 1024 * 1024) {
        let bytes = (head.weights.bytes() - offset).min(16 * 1024 * 1024);
        // Construction synchronized the quantizer and this is a live owned
        // byte buffer; offset+bytes never exceeds its allocation.
        quantized_digest
            .update(unsafe { cuda.read_device(head.weights.ptr() + offset as u64, bytes) }?);
    }
    let quantized_sha256 = format!("{:x}", quantized_digest.finalize());
    let scales_sha256 = format!("{:x}", Sha256::digest(head.scales.download()?));
    let mut all_exact =
        quantized_sha256 == expected.quantized_sha256 && scales_sha256 == expected.scales_sha256;
    let mut input = cuda.allocate::<bf16>(M * K)?;
    let output = cuda.allocate::<f32>(M * N)?;
    input.zero()?;
    head.run(&input, &output, &stream)?;
    stream.synchronize()?;
    let graph = stream.capture(|| head.run(&input, &output, &stream))?;
    let mut cases = Vec::new();
    for case in expected.cases {
        ensure!(
            ["normal", "changed", "zero", "padded"].contains(&case.name.as_str()),
            "draft-head case name"
        );
        let root = p.fixture.join(&case.name);
        let raw = fixture(&root, &case.input)?;
        ensure!(
            raw.len() == input.bytes() && case.input.element_bytes == 2,
            "draft-head input fixture shape"
        );
        input.upload(
            &raw.chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
                .collect::<Vec<_>>(),
        )?;
        unsafe {
            graph.launch(&stream)?;
        }
        stream.synchronize()?;
        let raw: Vec<u8> = output
            .download()?
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let reference = fixture(&root, &case.output)?;
        ensure!(
            reference.len() == raw.len() && case.output.element_bytes == 4,
            "draft-head output fixture shape"
        );
        let exact = raw == reference;
        all_exact &= exact;
        cases.push(json!({"case":case.name,"graph_replay":true,"bitwise_equal":exact,"actual_sha256":format!("{:x}",Sha256::digest(&raw))}));
    }
    let source = weights.view("head.weight")?;
    let mut canonical_digest = Sha256::new();
    for offset in (0..source.bytes).step_by(16 * 1024 * 1024) {
        canonical_digest.update(weights.device_range(
            source.offset + offset,
            (source.bytes - offset).min(16 * 1024 * 1024),
        )?);
    }
    let canonical_sha256 = format!("{:x}", canonical_digest.finalize());
    let canonical_unchanged = canonical_sha256 == expected.canonical_sha256;
    all_exact &= canonical_unchanged;
    Ok(
        json!({"rank":p.rank,"all_exact":all_exact,"canonical_unchanged":canonical_unchanged,
        "canonical_sha256":canonical_sha256,"quantized_sha256":quantized_sha256,"scales_sha256":scales_sha256,
        "cases":cases,"scope":"Private FP8 proposal-head preparation and FP32 output, exact native driver ABI and changed-input graph replay against the pinned GPU operator. Target head bytes unchanged; no full-model speed or quality claim."}),
    )
}
impl DraftHead {
    pub(crate) fn share_encoding_from(&mut self,source:&Self,stream:&Stream)->Result<usize> {
        ensure!(std::env::var("CANADA_SCRATCH_SHARED_DRAFT_HEAD").as_deref()==Ok("1")
            &&Rc::ptr_eq(&self.canonical,&source.canonical)&&self.cuda.owns_stream(stream)
            &&self.stream==stream.raw()as usize&&self.stream==source.stream
            &&self.projection.entry.artifact_sha256==source.projection.entry.artifact_sha256
            &&self.quantizer_sha256==source.quantizer_sha256
            &&self.weights.len()==source.weights.len()&&self.scales.len()==source.scales.len()
            &&self.cuda.owns_buffer(&source.weights)&&self.cuda.owns_buffer(&source.scales),
            "immutable draft-head sharing requires the same canonical source/encoding/stream");
        // Only a fresh QueryNetwork may invoke this: no captured peer head
        // can still reference the old buffers when they are released.
        stream.synchronize()?;
        if std::env::var("CANADA_SCRATCH_PAIRED_GENERATION_SCREEN").as_deref()==Ok("1") {
            use sha2::{Digest,Sha256};
            // Finite proof only: hash GPU-produced bytes before dropping the
            // redundant copy. Serving pays no download or hashing cost.
            let digest=|head:&Self|->Result<String>{
                let mut hash=Sha256::new();
                hash.update(head.weights.download()?);hash.update(head.scales.download()?);
                Ok(format!("{:x}",hash.finalize()))
            };
            let own=digest(self)?;let shared=digest(source)?;
            ensure!(own==shared,"independently encoded draft-head bytes differ");
            eprintln!("draft_head_shared_encoding_proof {}",serde_json::json!({
                "exact":true,"sha256":shared,"bytes":self.weights.bytes()+self.scales.bytes()}));
        }
        let reclaimed=if Rc::ptr_eq(&self.weights,&source.weights){0}else{self.weights.bytes()+self.scales.bytes()};
        self.weights=source.weights.clone();self.scales=source.scales.clone();
        Ok(reclaimed)
    }
    pub(crate) fn with_lossless(mut self,root:&Path,stream:&Stream)->Result<Self> {
        ensure!(self.stream==stream.raw()as usize&&self.lossless.is_none(),"private lossless stream/profile");
        self.lossless=Some(crate::lossless_head::Head::new_draft(&self.cuda,self.canonical.ptr("head.weight")?,root,stream)?);Ok(self)
    }
    pub(crate) fn lossless_enabled(&self)->bool {self.lossless_active.get()}
    pub(crate) fn precision_kind(&self)->u8 {if self.lossless_active.get(){2}else{u8::from(self.integer_active.get())}}
    pub(crate) fn set_precision_for_probe(&self,kind:u8)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_INT8_SCREEN").as_deref()==Ok("1")
            &&kind<=2&&!self.gemv_active.get()&&(kind!=1||self.integer.is_some())
            &&(kind!=2||self.lossless.is_some()),"private head precision isolated switch");
        self.integer_active.set(kind==1);self.lossless_active.set(kind==2);Ok(())
    }
    pub(crate) fn with_int8(mut self,root:&Path,stream:&Stream)->Result<Self> {
        ensure!(std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_INT8_SCREEN").as_deref()==Ok("1")
            &&self.stream==stream.raw()as usize&&self.cuda.owns_stream(stream)
            &&self.integer.is_none()&&!self.gemv_active.get(),"private INT8 finite admission");
        let bytes=N*K+N*(K/128)*4;
        ensure!(self.cuda.admission_memory()?.admitted_available>bytes+8*1024usize.pow(3),
            "private INT8 copy would consume reserved UMA headroom");
        let quantizer=Operator::load(&self.cuda,root,"private_int8_head_quantize")?;
        let projection=Operator::load(&self.cuda,root,"private_int8_head_m5")?;
        ensure!(quantizer.entry.arguments.iter().map(|a|(a.name.as_str(),a.kind.as_str())).collect::<Vec<_>>()
                ==[("W","*bf16"),("Q","*u8"),("S","*fp32")]
            &&projection.entry.arguments.iter().map(|a|(a.name.as_str(),a.kind.as_str())).collect::<Vec<_>>()
                ==[("X","*bf16"),("Q","*u8"),("S","*fp32"),("Y","*fp32")]
            &&[&quantizer,&projection].iter().all(|op|op.entry.grid.rows_div==1
                &&op.entry.grid.x_mul==1&&op.entry.grid.x_add==0&&op.entry.grid.z==1
                &&op.entry.metadata.num_warps==4)
            &&quantizer.entry.grid.y==40&&projection.entry.grid.y==1,"private INT8 geometry/ABI");
        let weights=self.cuda.allocate::<u8>(N*K)?;let scales=self.cuda.allocate::<f32>(N*(K/128))?;
        unsafe{quantizer.launch(N as u32,&[self.canonical.ptr("head.weight")?,weights.ptr(),scales.ptr()],stream)?;}
        stream.synchronize()?;
        self.integer=Some(IntegerHead{projection,weights,scales});
        eprintln!("DSpark private INT8 alternative: {bytes} bytes; canonical target unchanged");Ok(self)
    }
    pub(crate) fn int8_enabled(&self)->bool {self.integer_active.get()}
    pub(crate) fn with_gemv(mut self,root:&Path)->Result<Self> {
        let op=Operator::load(&self.cuda,root,"private_head_m5")?;
        ensure!(op.entry.arguments.iter().map(|a|(a.name.as_str(),a.kind.as_str())).collect::<Vec<_>>()
            ==[("X","*bf16"),("W8","*fp8e4nv"),("S","*u8"),("Y","*fp32")]
            && op.entry.grid.rows_div==1 && op.entry.grid.x_mul==1 && op.entry.grid.x_add==0
            && op.entry.grid.y==1 && op.entry.grid.z==1 && op.entry.metadata.num_warps==4,
            "private GEMV head geometry/FP32 output contract");
        self.gemv=Some(op);self.gemv_active.set(true);Ok(self)
    }
    pub(crate) fn gemv_enabled(&self)->bool {self.gemv_active.get()}
    pub(crate) fn set_gemv_for_probe(&self,active:bool)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_PRIVATE_HEAD_GEMV_SCREEN").as_deref()==Ok("1")
            && self.gemv.is_some(),"private head switch requires an isolated screen");
        self.gemv_active.set(active);Ok(())
    }
    pub(crate) fn new(
        cuda: &Cuda,
        canonical: Rc<Weights>,
        root: &Path,
        stream: &Stream,
    ) -> Result<Self> {
        let source = canonical.view("head.weight")?;
        ensure!(
            canonical.in_context(cuda)
                && cuda.owns_stream(stream)
                && source.dtype == Dtype::Bf16
                && source.shape == [N as u64, K as u64],
            "private draft-head canonical source"
        );
        let bytes = N * K + (N / 32) * (K / 32);
        ensure!(
            cuda.admission_memory()?.admitted_available > bytes + 8 * 1024usize.pow(3),
            "private draft-head copy would consume reserved host/device headroom"
        );
        let quantizer = Operator::load(cuda, root, "draft_head_quantize")?;
        let projection = Operator::load(cuda, root, "draft_head_m5")?;
        ensure!(
            quantizer
                .entry
                .arguments
                .iter()
                .map(|a| (a.name.as_str(), a.kind.as_str()))
                .collect::<Vec<_>>()
                == [("W", "*bf16"), ("Q", "*fp8e4nv"), ("S", "*u8")]
                && projection
                    .entry
                    .arguments
                    .iter()
                    .map(|a| (a.name.as_str(), a.kind.as_str()))
                    .collect::<Vec<_>>()
                    == [
                        ("X", "*bf16"),
                        ("W8", "*fp8e4nv"),
                        ("S", "*u8"),
                        ("Y", "*fp32")
                    ],
            "private draft-head operator ABI"
        );
        let weights = Rc::new(cuda.allocate::<u8>(N * K)?);
        let scales = Rc::new(cuda.allocate::<u8>((N / 32) * (K / 32))?);
        unsafe {
            // These operators use vocabulary rows for their launch grid. M5
            // and K5120 are immutable compile-time geometry, not runtime rows.
            quantizer.launch(
                N as u32,
                &[canonical.ptr("head.weight")?, weights.ptr(), scales.ptr()],
                stream,
            )?;
        }
        stream.synchronize()?;
        eprintln!("DSpark private FP8 head: {bytes} bytes; canonical target unchanged");
        Ok(Self {
            cuda: cuda.clone(),
            canonical,
            projection,
            gemv:None,
            gemv_active:Cell::new(false),
            integer:None,
            integer_active:Cell::new(false),
            lossless:None,
            lossless_active:Cell::new(false),
            weights,
            scales,
            quantizer_sha256: quantizer.entry.artifact_sha256.clone(),
            stream: stream.raw() as usize,
        })
    }
    pub(crate) fn shares_source(&self, canonical: &Rc<Weights>) -> bool {
        Rc::ptr_eq(&self.canonical, canonical)
    }
    pub(crate) fn run(
        &self,
        input: &Buffer<bf16>,
        output: &Buffer<f32>,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            self.cuda.owns_stream(stream)
                && self.stream == stream.raw() as usize
                && self.cuda.owns_buffer(input)
                && self.cuda.owns_buffer(output)
                && input.len() == M * K
                && output.len() == M * N,
            "private draft-head input/output context or geometry"
        );
        if self.lossless_active.get(){return self.lossless.as_ref().expect("active lossless draft loaded").run(input.ptr(),output.ptr(),stream);}
        unsafe {
            if self.integer_active.get() {
                let head=self.integer.as_ref().expect("active private INT8 loaded");
                return head.projection.launch(N as u32,
                    &[input.ptr(),head.weights.ptr(),head.scales.ptr(),output.ptr()],stream);
            }
            let projection=if self.gemv_active.get(){self.gemv.as_ref().expect("active private GEMV loaded")}else{&self.projection};
            projection.launch(
                N as u32,
                &[
                    input.ptr(),
                    self.weights.ptr(),
                    self.scales.ptr(),
                    output.ptr(),
                ],
                stream,
            )
        }
    }
}
