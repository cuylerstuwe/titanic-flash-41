//! Bounded FP4 index scoring. Rust owns packed inputs, TMA descriptors, range
//! schedules and output memory. The MIT device kernel is a selected primitive.
use crate::{
    cuda::{Buffer, Cuda, Stream},
    index::QueryRows,
};
use anyhow::{Result, ensure};
use libloading::Library;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::Cell, ffi::c_void, path::Path, rc::Rc};
type Encode = unsafe extern "C" fn(
    i32,
    i32,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *mut c_void,
) -> i32;
type Launch = unsafe extern "C" fn(
    u32,
    u32,
    u32,
    u32,
    *const i32,
    *const i32,
    *mut f32,
    *const c_void,
    *mut c_void,
) -> i32;
type SplitLaunch = unsafe extern "C" fn(u32, u32, u32, u32, u32,
    *const i32, *const i32, *mut f32, *const c_void, *mut c_void) -> i32;
struct KeySplit {
    _lib: Library,
    encode: Encode,
    launch: SplitLaunch,
    partitions: u32,
    prefill: bool,
}
impl KeySplit {
    fn open(root: &Path) -> Result<Option<Self>> {
        let partitions = match std::env::var("CANADA_SCRATCH_KEY_SPLIT_MQA") {
            Ok(value)=>value.parse::<u32>()?,
            Err(std::env::VarError::NotPresent)=>0,
            Err(error)=>return Err(error.into()),
        };
        ensure!([0,4,8,16,24].contains(&partitions), "invalid MQA key partition count");
        let prefill=match std::env::var("CANADA_SCRATCH_KEY_SPLIT_PREFILL") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid prefill key split policy");value=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(error)=>return Err(error.into()),
        };
        if partitions == 0 && !prefill {return Ok(None);}
        let root=root.parent().ok_or_else(||anyhow::anyhow!("MQA bundle parent"))?.join("mqa-keysplit-s366");
        let contract:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(contract["source_commit"]=="14073b4e1e706506e193231209738c848d092a1f"
            && contract["original_sha256"]=="796ee2737d26864cbdf9ba0f9358a7e094e856d8912f8efbf97b6092b4eb5184"
            && contract["adapted_source_sha256"]=="ea77d550ad839ea58a16624ca1db78d844fd5a2640bf0e079e8bd7284ee0c32f"
            && contract["adapter_sha256"]=="09e911e7f4eac60e242e7a36314c6607e919d136f500848aad97721dac063d32"
            && contract["variants"]==serde_json::json!([1,4,6,8,12,16,24]), "MQA split scheduler contract");
        for (name,field) in [("mqa_keysplit.cuh","adapted_source_sha256"),
            ("mqa_keysplit.cu","adapter_sha256"),("libcanada_mqa_keysplit.so","library_sha256")] {
            ensure!(contract[field]==format!("{:x}",Sha256::digest(std::fs::read(root.join(name))?)),
                "MQA split bundle hash: {name}");
        }
        let lib=unsafe {Library::new(root.join("libcanada_mqa_keysplit.so"))}?;
        Ok(Some(Self {encode:*unsafe {lib.get(b"ca_mqa_encode\0") }?,
            launch:*unsafe {lib.get(b"ca_mqa_launch_split\0") }?, _lib:lib,partitions,prefill}))
    }
}
#[derive(Deserialize)]
struct Contract {
    abi: u32,
    architecture: u32,
    compiler_target: String,
    physical_sms: usize,
    heads: usize,
    head_dim: usize,
    block_q: usize,
    block_kv: usize,
    q_stages: usize,
    kv_stages: usize,
    threads: usize,
    shared_bytes: usize,
    compressed_logits: bool,
    output_type: String,
    descriptor_bytes: usize,
    fast_math: bool,
    library_sha256: String,
    device_source_sha256: String,
}
pub struct Native {
    _lib: Library,
    encode: Encode,
    launch: Launch,
    key_split: Option<KeySplit>,
}
impl Native {
    pub fn open(cuda: &Cuda, root: &Path) -> Result<Rc<Self>> {
        cuda.qualify_gb10()?;
        let c: Contract = serde_json::from_slice(&std::fs::read(root.join("native.json"))?)?;
        ensure!(
            (c.abi, c.architecture) == (1, 121)
                && c.compiler_target == "sm_120f"
                && (
                    c.physical_sms,
                    c.heads,
                    c.head_dim,
                    c.block_q,
                    c.block_kv,
                    c.q_stages,
                    c.kv_stages,
                    c.threads,
                    c.shared_bytes,
                    c.descriptor_bytes
                ) == (48, 32, 128, 4, 128, 2, 5, 384, 62064, 640)
                && c.compressed_logits
                && c.output_type == "f32"
                && !c.fast_math
                && c.device_source_sha256
                    == "796ee2737d26864cbdf9ba0f9358a7e094e856d8912f8efbf97b6092b4eb5184",
            "MQA numeric/device contract"
        );
        let p = root.join("libcanada_mqa.so");
        ensure!(
            format!("{:x}", Sha256::digest(std::fs::read(&p)?)) == c.library_sha256,
            "MQA library hash"
        );
        let lib = unsafe { Library::new(p) }?;
        let abi: unsafe extern "C" fn() -> i32 = *unsafe { lib.get(b"ca_mqa_abi\0") }?;
        ensure!(unsafe { abi() } == 1, "MQA ABI");
        Ok(Rc::new(Self {
            encode: *unsafe { lib.get(b"ca_mqa_encode\0") }?,
            launch: *unsafe { lib.get(b"ca_mqa_launch\0") }?,
            _lib: lib,
            key_split: KeySplit::open(root)?,
        }))
    }
    pub(crate) fn key_splits(&self, rows:usize)->u32 {
        let Some(split)=&self.key_split else{return 0;};
        if rows==6 {split.partitions}
        else if split.prefill {match rows {16=>12,32=>6,_=>0}}
        else {0}
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_maps(&self,rows:i32,keys:i32,q:*const c_void,qs:*const c_void,
        k:*const c_void,ks:*const c_void,weights:*const c_void,maps:*mut c_void)->i32 {
        let status=unsafe {(self.encode)(rows,keys,q,qs,k,ks,weights,maps)};
        if status!=0 {return status;}
        // Both launch families require their own dynamic shared-memory limit.
        if let Some(split)=&self.key_split {
            unsafe {(split.encode)(rows,keys,q,qs,k,ks,weights,maps)}
        }else{0}
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_scores(&self,rows:u32,keys:u32,columns:u32,stride:u32,
        starts:*const i32,ends:*const i32,output:*mut f32,maps:*const c_void,stream:*mut c_void)->i32 {
        let partitions=self.key_splits(rows as usize);
        if partitions>0 && let Some(split)=&self.key_split {
            unsafe {(split.launch)(partitions,rows,keys,columns,stride,starts,ends,output,maps,stream)}
        }else{unsafe {(self.launch)(rows,keys,columns,stride,starts,ends,output,maps,stream)}}
    }
}
pub struct KeyMatrix {
    rows: usize,
    pub(crate) payload: Buffer<u8>,
    pub(crate) scales: Buffer<i32>,
    stream: Cell<usize>,
    pub(crate) ready: Cell<bool>,
}
impl KeyMatrix {
    pub fn new(cuda: &Cuda, rows: usize) -> Result<Self> {
        ensure!((1..=1048576).contains(&rows), "MQA key rows");
        let mut scales = cuda.allocate::<i32>(rows.div_ceil(4) * 4)?;
        scales.zero()?;
        Ok(Self {
            rows,
            payload: cuda.allocate(rows * 64)?,
            scales,
            stream: Cell::new(0),
            ready: Cell::new(false),
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub(crate) fn bind(&self, stream: &Stream) -> Result<()> {
        let p = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == p,
            "MQA key matrix requires one ordered stream"
        );
        self.stream.set(p);
        Ok(())
    }
}
pub struct Ranges {
    starts: Buffer<i32>,
    ends: Buffer<i32>,
    rows: usize,
    keys: usize,
    columns: usize,
}
impl Ranges {
    pub(crate) fn key_ends(&self) -> &Buffer<i32> { &self.ends }
    pub(crate) fn key_ends_mut(&mut self) -> &mut Buffer<i32> { &mut self.ends }
    pub fn update(&mut self, starts: &[i32], lengths: &[i32]) -> Result<()> {
        ensure!(
            starts.len() == self.rows
                && lengths.len() == self.rows
                && starts.iter().zip(lengths).all(|(&s, &l)| s >= 0
                    && l >= 0
                    && (l as usize) <= self.columns
                    && (s as usize) + (l as usize) <= self.keys),
            "MQA range update outside storage"
        );
        let ends = starts
            .iter()
            .zip(lengths)
            .map(|(s, l)| s + l)
            .collect::<Vec<_>>();
        self.starts.upload(starts)?;
        self.ends.upload(&ends)
    }
    pub fn new(
        cuda: &Cuda,
        keys: usize,
        columns: usize,
        starts: &[i32],
        lengths: &[i32],
    ) -> Result<Self> {
        ensure!(
            (1..=crate::config::MAX_WIDE_ROWS).contains(&starts.len())
                && lengths.len() == starts.len()
                && (1..=1048576).contains(&keys)
                && columns > 0
                && columns <= keys,
            "MQA range dimensions"
        );
        ensure!(
            starts.iter().zip(lengths).all(|(&s, &l)| s >= 0
                && l >= 0
                && (l as usize) <= columns
                && (s as usize) + (l as usize) <= keys),
            "MQA key range bounds"
        );
        let ends = starts
            .iter()
            .zip(lengths)
            .map(|(s, l)| s + l)
            .collect::<Vec<_>>();
        let mut ds = cuda.allocate(starts.len())?;
        let mut de = cuda.allocate(starts.len())?;
        ds.upload(starts)?;
        de.upload(&ends)?;
        Ok(Self {
            starts: ds,
            ends: de,
            rows: starts.len(),
            keys,
            columns,
        })
    }
}
#[repr(C, align(64))]
struct Descriptors([u8; 640]);
pub struct Dense {
    cuda: Cuda,
    native: Rc<Native>,
    queries: Rc<QueryRows>,
    keys: Rc<KeyMatrix>,
    maps: Box<Descriptors>,
    rows: usize,
    columns: usize,
    stride: usize,
    output: Buffer<f32>,
    // A view must be dropped before the optional same-owner scratch backing.
    score_owner: Option<Rc<Buffer<f32>>>,
    stream: Cell<usize>,
}
impl Dense {
    pub fn new(
        cuda: &Cuda,
        native: Rc<Native>,
        queries: Rc<QueryRows>,
        keys: Rc<KeyMatrix>,
        columns: usize,
        budget: usize,
    ) -> Result<Self> {
        let rows = queries.rows();
        ensure!(
            queries.in_context(cuda)
                && cuda.owns_buffer(&keys.payload)
                && cuda.owns_buffer(&keys.scales)
                && columns > 0
                && columns <= keys.rows
                && (1..=2 * 1024 * 1024 * 1024usize).contains(&budget),
            "MQA inputs/context/budget"
        );
        let stride = columns.div_ceil(128) * 128;
        let n = rows.div_ceil(4) * 4 * stride;
        ensure!(
            n * 4 <= budget,
            "MQA logits exceed admitted byte budget; schedule fewer query rows"
        );
        let output = cuda.allocate(n)?;
        let mut maps = Box::new(Descriptors([0; 640]));
        let status = unsafe {
            native.encode_maps(
                rows as i32,
                keys.rows as i32,
                queries.payload.ptr() as _,
                queries.scales.ptr() as _,
                keys.payload.ptr() as _,
                keys.scales.ptr() as _,
                queries.weights.ptr() as _,
                maps.0.as_mut_ptr() as _,
            )
        };
        ensure!(
            status == 0,
            "MQA TMA/attribute preparation failed: {status}"
        );
        Ok(Self {
            cuda: cuda.clone(),
            native,
            queries,
            keys,
            maps,
            rows,
            columns,
            stride,
            output,
            score_owner: None,
            stream: Cell::new(0),
        })
    }
    pub fn run(&self, ranges: &Ranges, stream: &Stream) -> Result<()> {
        ensure!(
            self.queries.ready.get()
                && self.keys.ready.get()
                && ranges.rows == self.rows
                && ranges.keys == self.keys.rows
                && ranges.columns == self.columns
                && self.cuda.owns_buffer(&ranges.starts)
                && self.cuda.owns_buffer(&ranges.ends)
                && self.cuda.owns_stream(stream),
            "MQA input readiness/ranges/context"
        );
        self.queries.bind(stream)?;
        let p = stream.raw() as usize;
        ensure!(
            [self.stream.get(), self.keys.stream.get()]
                .iter()
                .all(|v| *v == 0 || *v == p),
            "MQA state requires one ordered stream"
        );
        self.stream.set(p);
        self.keys.stream.set(p);
        let status = unsafe {
            self.native.launch_scores(
                self.rows as u32,
                self.keys.rows as u32,
                self.columns as u32,
                self.stride as u32,
                ranges.starts.ptr() as _,
                ranges.ends.ptr() as _,
                self.output.ptr() as _,
                self.maps.0.as_ptr() as _,
                stream.raw(),
            )
        };
        ensure!(status == 0, "MQA launch: {status}");
        Ok(())
    }
    /// Construction-only scratch reuse. Q/K descriptors do not contain the
    /// score pointer; launch binds the view after its producer writes it.
    pub(crate) fn share_score_scratch(&mut self, storage: Rc<Buffer<f32>>) -> Result<usize> {
        ensure!(self.stream.get()==0 && self.score_owner.is_none()
            && self.cuda.owns_buffer(&storage) && storage.len()>=self.output.len(),
            "score scratch requires unexecuted same-context sufficient storage");
        let released=self.output.bytes();
        self.output=unsafe{storage.view(0,self.output.len())}?;
        self.score_owner=Some(storage);
        Ok(released)
    }
    pub fn output(&self) -> &Buffer<f32> {
        &self.output
    }
    pub fn output_mut(&mut self) -> &mut Buffer<f32> {
        &mut self.output
    }
    pub fn stride(&self) -> usize {
        self.stride
    }
    pub(crate) fn inputs_mut(&mut self) -> Result<(&mut QueryRows, &mut KeyMatrix)> {
        let q = Rc::get_mut(&mut self.queries).ok_or_else(|| {
            anyhow::anyhow!("shared queries must be written by their GPU producer")
        })?;
        let k = Rc::get_mut(&mut self.keys)
            .ok_or_else(|| anyhow::anyhow!("shared keys must be written by their GPU producer"))?;
        Ok((q, k))
    }
}

struct RowChunk {
    start: usize,
    rows: usize,
    maps: Box<Descriptors>,
}
/// A pre-admitted query-row schedule sharing one bounded score arena. Inputs
/// stay in their producer's buffers; descriptors address each row slice without
/// copying queries. The caller must consume a chunk before launching the next.
pub struct Chunked {
    cuda: Cuda,
    native: Rc<Native>,
    queries: Rc<QueryRows>,
    keys: Rc<KeyMatrix>,
    chunks: Vec<RowChunk>,
    columns: usize,
    stride: usize,
    output: Buffer<f32>,
    // A view must be dropped before the optional same-owner scratch backing.
    score_owner: Option<Rc<Buffer<f32>>>,
    stream: Cell<usize>,
}
impl Chunked {
    pub fn new(
        cuda: &Cuda,
        native: Rc<Native>,
        queries: Rc<QueryRows>,
        keys: Rc<KeyMatrix>,
        columns: usize,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            queries.in_context(cuda)
                && cuda.owns_buffer(&keys.payload)
                && cuda.owns_buffer(&keys.scales)
                && columns > 0
                && columns <= keys.rows
                && budget <= 2 * 1024 * 1024 * 1024usize,
            "chunked MQA context/columns/budget"
        );
        let schedule = crate::prefill_plan::ScoreSchedule::new(queries.rows(), columns, budget)?;
        let stride = schedule.stride;
        let output = cuda.allocate(schedule.arena_bytes / 4)?;
        let mut chunks = Vec::new();
        for chunk in schedule.chunks {
            let start = chunk.start;
            let rows = chunk.len();
            let mut maps = Box::new(Descriptors([0; 640]));
            let status = unsafe {
                native.encode_maps(
                    rows as i32,
                    keys.rows as i32,
                    (queries.payload.ptr() + (start * 32 * 64) as u64) as _,
                    (queries.scales.ptr() + (start * 32 * 4) as u64) as _,
                    keys.payload.ptr() as _,
                    keys.scales.ptr() as _,
                    (queries.weights.ptr() + (start * 32 * 4) as u64) as _,
                    maps.0.as_mut_ptr() as _,
                )
            };
            ensure!(
                status == 0,
                "chunked MQA descriptor at row{start}: {status}"
            );
            chunks.push(RowChunk { start, rows, maps });
        }
        Ok(Self {
            cuda: cuda.clone(),
            native,
            queries,
            keys,
            chunks,
            columns,
            stride,
            output,
            score_owner: None,
            stream: Cell::new(0),
        })
    }
    pub fn len(&self) -> usize {
        self.chunks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
    pub fn rows(&self, index: usize) -> Result<std::ops::Range<usize>> {
        let c = self
            .chunks
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("MQA chunk index"))?;
        Ok(c.start..c.start + c.rows)
    }
    pub fn run_chunk(&mut self, index: usize, ranges: &Ranges, stream: &Stream) -> Result<()> {
        let c = self
            .chunks
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("MQA chunk index"))?;
        ensure!(
            self.queries.ready.get()
                && self.keys.ready.get()
                && ranges.rows == self.queries.rows()
                && ranges.keys == self.keys.rows
                && ranges.columns == self.columns
                && self.cuda.owns_buffer(&ranges.starts)
                && self.cuda.owns_buffer(&ranges.ends)
                && self.cuda.owns_stream(stream),
            "chunked MQA readiness/ranges/context"
        );
        self.queries.bind(stream)?;
        let p = stream.raw() as usize;
        ensure!(
            [self.stream.get(), self.keys.stream.get()]
                .iter()
                .all(|v| *v == 0 || *v == p),
            "chunked MQA requires one ordered stream"
        );
        self.stream.set(p);
        self.keys.stream.set(p);
        let status = unsafe {
            self.native.launch_scores(
                c.rows as u32,
                self.keys.rows as u32,
                self.columns as u32,
                self.stride as u32,
                (ranges.starts.ptr() + (c.start * 4) as u64) as _,
                (ranges.ends.ptr() + (c.start * 4) as u64) as _,
                self.output.ptr() as _,
                c.maps.0.as_ptr() as _,
                stream.raw(),
            )
        };
        ensure!(
            status == 0,
            "chunked MQA launch at row{}: {status}",
            c.start
        );
        Ok(())
    }
    /// Construction-only scratch reuse. Q/K descriptors do not contain the
    /// score pointer; launch binds the view after its producer writes it.
    pub(crate) fn share_score_scratch(&mut self, storage: Rc<Buffer<f32>>) -> Result<usize> {
        ensure!(self.stream.get()==0 && self.score_owner.is_none()
            && self.cuda.owns_buffer(&storage) && storage.len()>=self.output.len(),
            "score scratch requires unexecuted same-context sufficient storage");
        let released=self.output.bytes();
        self.output=unsafe{storage.view(0,self.output.len())}?;
        self.score_owner=Some(storage);
        Ok(released)
    }
    pub fn output(&self) -> &Buffer<f32> {
        &self.output
    }
    pub fn output_mut(&mut self) -> &mut Buffer<f32> {
        &mut self.output
    }
    pub fn stride(&self) -> usize {
        self.stride
    }
    pub(crate) fn inputs_mut(&mut self) -> Result<(&mut QueryRows, &mut KeyMatrix)> {
        let q = Rc::get_mut(&mut self.queries)
            .ok_or_else(|| anyhow::anyhow!("shared queries require their GPU producer"))?;
        let k = Rc::get_mut(&mut self.keys)
            .ok_or_else(|| anyhow::anyhow!("shared keys require their GPU producer"))?;
        Ok((q, k))
    }
}
