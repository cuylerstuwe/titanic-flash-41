//! Rust-owned low-ratio index projections, schedules, packed queries and keys.
//! Only selected GEMM/quantization device primitives are dependencies.
use crate::{
    bf16_linear::Projection,
    blas::Blas,
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, Stream},
    prelude::FrequencyTable,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use libloading::Library;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::Cell, ffi::c_void, path::Path, rc::Rc};
type Small = unsafe extern "C" fn(
    i32,
    i32,
    i64,
    *const c_void,
    *const c_void,
    *mut c_void,
    *mut c_void,
) -> i32;
type Key = unsafe extern "C" fn(
    i32,
    i32,
    *const c_void,
    *const c_void,
    *const f32,
    *const i32,
    *const i64,
    *mut u8,
    *mut c_void,
) -> i32;
type Query = unsafe extern "C" fn(
    i32,
    *const c_void,
    *const f32,
    *const i32,
    *const c_void,
    *mut u8,
    *mut i32,
    *mut f32,
    *mut c_void,
) -> i32;
#[derive(Deserialize)]
struct Contract {
    abi: u32,
    architecture: u32,
    physical_sms: u32,
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    page_tokens: usize,
    ratios: Vec<usize>,
    slot_bytes: usize,
    weight_scale: f32,
    fast_math: bool,
    pdl: bool,
    small_max_rows: usize,
    library_sha256: String,
}
pub struct Native {
    _lib: Library,
    small: Small,
    key: Key,
    query: Query,
}
impl Native {
    pub fn open(cuda: &Cuda, root: &Path) -> Result<Rc<Self>> {
        cuda.qualify_gb10()?;
        let c: Contract = serde_json::from_slice(&std::fs::read(root.join("native.json"))?)?;
        ensure!(
            (c.abi, c.architecture, c.physical_sms) == (1, 121, 48)
                && (
                    c.heads,
                    c.head_dim,
                    c.rope_dim,
                    c.page_tokens,
                    c.slot_bytes,
                    c.small_max_rows
                ) == (32, 128, 64, 64, 68, 32)
                && c.ratios == [1, 2]
                && c.weight_scale.to_bits() == 0.015625f32.to_bits()
                && !c.fast_math
                && !c.pdl,
            "index primitive contract"
        );
        let p = root.join("libcanada_index.so");
        ensure!(
            format!("{:x}", Sha256::digest(std::fs::read(&p)?)) == c.library_sha256,
            "index primitive hash"
        );
        let lib = unsafe { Library::new(p) }?;
        let abi: unsafe extern "C" fn() -> i32 = *unsafe { lib.get(b"ca_index_abi\0") }?;
        ensure!(unsafe { abi() } == 1, "index primitive ABI");
        Ok(Rc::new(Self {
            small: *unsafe { lib.get(b"ca_index_small\0") }?,
            key: *unsafe { lib.get(b"ca_index_k\0") }?,
            query: *unsafe { lib.get(b"ca_index_q\0") }?,
            _lib: lib,
        }))
    }
}
enum ProjectionKind {
    Small {
        native: Rc<Native>,
        weight: u64,
        output: Buffer<bf16>,
    },
    Vendor(Projection<bf16>),
}
/// The effective recipe uses batch-invariant small GEMM through M32 and
/// GEMMEx above it; substituting either across that boundary changes BF16 ties.
pub struct IndexProjection {
    cuda: Cuda,
    _weights: Rc<Weights>,
    kind: ProjectionKind,
    rows: usize,
    n: usize,
    k: usize,
    stream: Cell<usize>,
}
impl IndexProjection {
    pub fn new(
        cuda: &Cuda,
        weights: Rc<Weights>,
        blas: Rc<Blas>,
        native: Rc<Native>,
        prefix: &str,
        rows: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=crate::config::MAX_WIDE_ROWS).contains(&rows) && weights.in_context(cuda),
            "index projection rows/context"
        );
        let name = format!("{prefix}.weight");
        let v = weights.view(&name)?;
        ensure!(
            v.dtype == Dtype::Bf16 && (v.shape == [128, 512] || v.shape == [32, 5120]),
            "index projection canonical matrix"
        );
        let (n, k) = (v.shape[0] as usize, v.shape[1] as usize);
        let kind = if rows <= 32 {
            ProjectionKind::Small {
                native,
                weight: weights.ptr(&name)?,
                output: cuda.allocate(rows * n)?,
            }
        } else {
            ProjectionKind::Vendor(Projection::new(cuda, weights.clone(), blas, prefix, rows)?)
        };
        Ok(Self {
            cuda: cuda.clone(),
            _weights: weights,
            kind,
            rows,
            n,
            k,
            stream: Cell::new(0),
        })
    }
    pub fn output(&self) -> &Buffer<bf16> {
        match &self.kind {
            ProjectionKind::Small { output, .. } => output,
            ProjectionKind::Vendor(p) => p.output(),
        }
    }
    pub fn run(&self, input: &Buffer<bf16>, stream: &Stream) -> Result<()> {
        ensure!(
            input.len() == self.rows * self.k
                && self.cuda.owns_buffer(input)
                && self.cuda.owns_stream(stream),
            "index projection input/context"
        );
        bind(&self.stream, stream)?;
        match &self.kind {
            ProjectionKind::Small {
                native,
                weight,
                output,
            } => {
                let status = unsafe {
                    (native.small)(
                        self.n as i32,
                        self.rows as i32,
                        self.k as i64,
                        input.ptr() as _,
                        *weight as _,
                        output.ptr() as _,
                        stream.raw(),
                    )
                };
                ensure!(status == 0, "index small GEMM launch: {status}");
                Ok(())
            }
            ProjectionKind::Vendor(p) => p.run(input, stream),
        }
    }
}
fn bind(cell: &Cell<usize>, stream: &Stream) -> Result<()> {
    let p = stream.raw() as usize;
    ensure!(
        cell.get() == 0 || cell.get() == p,
        "index state requires one ordered stream"
    );
    cell.set(p);
    Ok(())
}
pub struct Schedule {
    positions: Buffer<i32>,
    locations: Buffer<i64>,
    rows: usize,
    capacity: usize,
    frequency_rows: usize,
    ready: bool,
}
impl Schedule {
    pub fn new(
        cuda: &Cuda,
        capacity: usize,
        frequency_rows: usize,
        positions: &[i32],
        locations: &[i64],
    ) -> Result<Self> {
        ensure!(
            (1..=crate::config::MAX_WIDE_ROWS).contains(&positions.len())
                && (1..=1052672).contains(&capacity)
                && (1..=1048576).contains(&frequency_rows),
            "index schedule dimensions"
        );
        let mut s = Self {
            positions: cuda.allocate(positions.len())?,
            locations: cuda.allocate(positions.len())?,
            rows: positions.len(),
            capacity,
            frequency_rows,
            ready: false,
        };
        s.update(positions, locations)?;
        Ok(s)
    }
    pub fn update(&mut self, positions: &[i32], locations: &[i64]) -> Result<()> {
        self.ready = false;
        ensure!(
            positions.len() == self.rows && locations.len() == self.rows,
            "index schedule row count"
        );
        ensure!(
            positions
                .iter()
                .all(|&p| p >= 0 && (p as usize) < self.frequency_rows),
            "index rotary position bounds"
        );
        let mut unique = std::collections::BTreeSet::new();
        ensure!(
            locations
                .iter()
                .all(|&l| l == 0 || (l > 0 && (l as usize) < self.capacity && unique.insert(l))),
            "index key write bounds/alias"
        );
        self.positions.upload(positions)?;
        self.locations.upload(locations)?;
        self.ready = true;
        Ok(())
    }
}
pub struct KeyCache {
    cuda: Cuda,
    native: Rc<Native>,
    _weights: Rc<Weights>,
    frequency: Rc<FrequencyTable>,
    norm: u64,
    ratio: usize,
    rows: usize,
    capacity: usize,
    data: Rc<Buffer<u8>>,
    stream: Rc<Cell<usize>>,
}
impl KeyCache {
    pub fn new(
        cuda: &Cuda,
        native: Rc<Native>,
        weights: Rc<Weights>,
        frequency: Rc<FrequencyTable>,
        layer: usize,
        rows: usize,
        pages: usize,
    ) -> Result<Self> {
        ensure!(
            [2, 8, 14, 20].contains(&layer)
                && (1..=crate::config::MAX_WIDE_ROWS).contains(&rows)
                && pages > 0
                && pages <= 16448
                && weights.in_context(cuda)
                && frequency.compressed()
                && cuda.owns_buffer(frequency.values()),
            "index key owner geometry/context"
        );
        let name = format!("layers.{layer}.attn.indexer.k_norm.weight");
        let v = weights.view(&name)?;
        ensure!(
            v.dtype == Dtype::Bf16 && v.shape == [128],
            "index key norm weight"
        );
        let norm = weights.ptr(&name)?;
        let mut data = cuda.allocate::<u8>(pages * 64 * 68)?;
        data.zero()?;
        Ok(Self {
            cuda: cuda.clone(),
            native,
            _weights: weights,
            frequency,
            norm,
            ratio: if layer == 20 { 1 } else { 2 },
            rows,
            capacity: pages * 64,
            data: Rc::new(data),
            stream: Rc::new(Cell::new(0)),
        })
    }
    pub(crate) fn share_rows(&self, rows: usize) -> Result<Self> {
        ensure!((1..=crate::config::MAX_WIDE_ROWS).contains(&rows), "shared index row profile");
        Ok(Self {
            cuda: self.cuda.clone(),
            native: self.native.clone(),
            _weights: self._weights.clone(),
            frequency: self.frequency.clone(),
            norm: self.norm,
            ratio: self.ratio,
            rows,
            capacity: self.capacity,
            data: self.data.clone(),
            stream: self.stream.clone(),
        })
    }
    pub fn run(&self, input: &Buffer<bf16>, schedule: &Schedule, stream: &Stream) -> Result<()> {
        ensure!(
            schedule.ready
                && schedule.rows == self.rows
                && schedule.capacity == self.capacity
                && schedule.frequency_rows == self.frequency.rows()
                && input.len() == self.rows * 128
                && self.cuda.owns_buffer(input)
                && self.cuda.owns_buffer(&schedule.positions)
                && self.cuda.owns_buffer(&schedule.locations)
                && self.cuda.owns_stream(stream),
            "index key schedule/input/context"
        );
        bind(&self.stream, stream)?;
        // The native packing ABI caps a launch at4096 rows. Cache locations
        // and rotary positions remain absolute; only input metadata is sliced.
        for start in (0..self.rows).step_by(4096) {
            let rows = (self.rows - start).min(4096);
            let status = unsafe {
                (self.native.key)(
                    rows as i32,
                    self.ratio as i32,
                    (input.ptr() + (start * 128 * 2) as u64) as _,
                    self.norm as _,
                    self.frequency.values().ptr() as _,
                    (schedule.positions.ptr() + (start * 4) as u64) as _,
                    (schedule.locations.ptr() + (start * 8) as u64) as _,
                    self.data.ptr() as _,
                    stream.raw(),
                )
            };
            ensure!(status == 0, "index K launch at row{start}: {status}");
        }
        Ok(())
    }
    pub fn data(&self) -> &Buffer<u8> {
        &self.data
    }
    pub(crate) fn clear_state(&mut self, stream: &Stream) -> Result<()> {
        self.bind(stream)?;
        unsafe {
            self.cuda
                .zero_device(self.data.ptr(), self.data.bytes(), stream)
        }
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub(crate) fn bind(&self, stream: &Stream) -> Result<()> {
        ensure!(self.cuda.owns_stream(stream), "index cache reader context");
        bind(&self.stream, stream)
    }
    pub(crate) fn upload(&mut self, data: &[u8]) -> Result<()> {
        Rc::get_mut(&mut self.data)
            .ok_or_else(|| anyhow::anyhow!("shared index seeding needs an exclusive owner"))?
            .upload(data)
    }
}

/// Admitted logical-to-physical slot map. Aliased reads are legal for shared
/// retained prefixes; every destination key row is distinct.
pub struct KeySlots {
    values: Buffer<i32>,
    capacity: usize,
    rows: usize,
    contiguous: bool,
}
impl KeySlots {
    pub(crate) fn values(&self) -> &Buffer<i32> { &self.values }
    pub(crate) fn capacity(&self) -> usize { self.capacity }
    pub fn new(cuda: &Cuda, capacity: usize, values: &[i32]) -> Result<Self> {
        ensure!(
            (1..=1048576).contains(&values.len())
                && capacity > 0
                && capacity <= 1052672
                && values.iter().all(|&v| v >= 0 && (v as usize) < capacity),
            "index gather slot bounds"
        );
        let mut b = cuda.allocate(values.len())?;
        b.upload(values)?;
        Ok(Self {
            values: b,
            capacity,
            rows: values.len(),
            contiguous: values.windows(2).all(|w| w[1] == w[0] + 1),
        })
    }
}
pub struct KeyGather {
    cuda: Cuda,
    module: crate::cuda::Module,
    packed: Option<crate::cuda::Module>,
    cached: Option<CachedGather>,
}
struct CachedGather {
    module: crate::cuda::Module,
    queries: usize,
    enabled: Cell<bool>,
    valid: Buffer<i32>,
    dirty: Cell<bool>,
    binding: Cell<(u64,u64,u64,u64)>,
}
impl KeyGather {
    pub fn new(cuda: &Cuda, core: &[u8], kernels:&Path, query_rows:usize) -> Result<Self> {
        let requested=match std::env::var("CANADA_SCRATCH_PACKED_INDEX_GATHER") {
            Ok(value)=>{ensure!(value=="0"||value=="4","unsupported packed index gather");value=="4"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(error)=>return Err(error.into()),
        };
        // The artifact's original receipt covered six queries. Its device-side
        // bound takes the query count at runtime; s575 also checked the exact
        // pinned binary at16, including a maximum in the final query.
        let packed=if requested && [6,16].contains(&query_rows) {
            let root=kernels.join("index-gather-packed-v1");
            let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
            let raw=std::fs::read(root.join("gather.cubin"))?;
            ensure!(manifest["architecture"]==121 && manifest["threads"]==128
                && manifest["grid"]==192 && manifest["query_rows"]==6 && manifest["lanes_per_key"]==4
                && manifest["source_sha256"]=="7b103a6522611d97f21795004aa9267e137a643d102ea9a9763d05bf6b57ba89"
                && manifest["artifact_sha256"]=="6a846381dd20b1911242c8786e282f156dd311b882f1b3fde52534187f39a9e0"
                && format!("{:x}",Sha256::digest(&raw))==manifest["artifact_sha256"].as_str().unwrap_or(""),
                "packed index gather artifact/geometry contract");
            Some(cuda.module(&raw)?)
        }else{None};
        let requested=match std::env::var("CANADA_SCRATCH_CACHED_INDEX_GATHER") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid cached index gather");value=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(error)=>return Err(error.into()),
        };
        let requested16=match std::env::var("CANADA_SCRATCH_CACHED_INDEX16") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid M16 index cache policy");value=="1"},
            // s661 exact component lifecycle; s670 exact full-model replay,
            // abort and changed-history handoff, with lower live8/16 latency.
            Err(std::env::VarError::NotPresent)=>true,
            Err(error)=>return Err(error.into()),
        };
        let cached=if requested && (query_rows==6 || (query_rows==16 && requested16)) {
            let root=kernels.join(if query_rows==16 {"index-gather-cached16-v1"}else{"index-gather-cached-v1"});
            let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
            let raw=std::fs::read(root.join("gather.cubin"))?;
            ensure!(manifest["architecture"]==121 && manifest["threads"]==128
                && manifest["grid"]==192 && manifest["query_rows"]==query_rows && manifest["dirty_suffix_rows"]==query_rows
                && manifest["source_sha256"]==if query_rows==16 {"acbf9adba2c69a1fc2443afcdd55a322be27d984e744ff098283bec81dc1af7c"}else{"f8cc1492ada68c5cb2daab17090ef9d764a78b275bf73038988e5a1738852ba5"}
                && manifest["artifact_sha256"]==if query_rows==16 {"dbbe4306d415a0f3a341bc635c016bacf3581545325a45aa2250cfd1cc9621af"}else{"4597cfeb6114e2a456ff0599506b8f4754252dbaa397ee403fe1214bae262608"}
                && format!("{:x}",Sha256::digest(&raw))==manifest["artifact_sha256"].as_str().unwrap_or(""),
                "cached index gather artifact contract");
            let mut valid=cuda.allocate::<i32>(1)?;valid.zero()?;
            Some(CachedGather {module:cuda.module(&raw)?,queries:query_rows,enabled:Cell::new(true),valid,dirty:Cell::new(true),binding:Cell::new((0,0,0,0))})
        }else{None};
        Ok(Self {
            cuda: cuda.clone(),
            module: cuda.module(core)?,
            packed,
            cached,
        })
    }
    pub(crate) fn packed_enabled(&self)->bool {self.packed.is_some()}
    pub(crate) fn cached_enabled(&self)->bool {self.cached.as_ref().is_some_and(|c|c.enabled.get())}
    pub(crate) fn set_cached16_for_probe(&self,enabled:bool)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_CACHED_INDEX16_SCREEN").as_deref()==Ok("1"),"M16 cached gather screen");
        let c=self.cached.as_ref().ok_or_else(||anyhow::anyhow!("M16 cache was not prepared"))?;
        ensure!(c.queries==16,"M16 cache shape");
        c.enabled.set(enabled);c.dirty.set(true);
        Ok(())
    }
    pub(crate) fn invalidate_cache(&self) {
        if let Some(c)=&self.cached {c.dirty.set(true);}
    }
    /// Called by the target before graph capture/replay, never from the graph.
    /// Capturing this memset would silently turn every replay into a cold copy.
    pub(crate) fn prepare_cache(&self,stream:&Stream)->Result<()> {
        if let Some(c)=&self.cached {
            ensure!(self.cuda.owns_stream(stream),"cached index prepare stream");
            if c.dirty.get() {
                unsafe{self.cuda.zero_device(c.valid.ptr(),c.valid.bytes(),stream)?;}
                c.dirty.set(false);
            }
        }
        Ok(())
    }
    pub fn run(
        &self,
        cache: &KeyCache,
        slots: &KeySlots,
        output: &crate::mqa::KeyMatrix,
        stream: &Stream,
    ) -> Result<()> {
        self.run_with_bound(cache, slots, output, None, stream)
    }
    pub(crate) fn run_with_bound(
        &self,
        cache: &KeyCache,
        slots: &KeySlots,
        output: &crate::mqa::KeyMatrix,
        ends: Option<&Buffer<i32>>,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(ends.is_none_or(|e| self.cuda.owns_buffer(e) && (1..=crate::config::MAX_WIDE_ROWS).contains(&e.len())),
            "bounded index gather query metadata");
        ensure!(
            slots.capacity == cache.capacity
                && slots.rows == output.rows()
                && self.cuda.owns_buffer(cache.data())
                && self.cuda.owns_buffer(&slots.values)
                && self.cuda.owns_buffer(&output.payload)
                && self.cuda.owns_buffer(&output.scales)
                && self.cuda.owns_stream(stream),
            "index gather cache/output/context"
        );
        cache.bind(stream)?;
        output.bind(stream)?;
        let mut source = cache.data.ptr();
        let mut ids = slots.values.ptr();
        let mut payload = output.payload.ptr();
        let mut scales = output.scales.ptr();
        let mut rows = slots.rows as i32;
        let mut args: Vec<*mut c_void> = vec![
            (&mut source as *mut u64).cast(),
            (&mut ids as *mut u64).cast(),
            (&mut payload as *mut u64).cast(),
            (&mut scales as *mut u64).cast(),
            (&mut rows as *mut i32).cast(),
        ];
        let mut end_ptr=ends.map_or(0, |b| b.ptr());
        let mut queries=ends.map_or(0, |b| b.len()) as i32;
        if ends.is_some() {
            args.push((&mut end_ptr as *mut u64).cast());
            args.push((&mut queries as *mut i32).cast());
        }
        if let Some(c)=self.cached.as_ref().filter(|c|c.enabled.get()) {
            ensure!(queries==c.queries as i32 && slots.contiguous && !c.dirty.get(),
                "cached gather requires its prepared query count and immutable contiguous slots");
            let binding=(source,ids,payload,scales);
            ensure!(c.binding.get()==(0,0,0,0) || c.binding.get()==binding,
                "cached gather storage or immutable slot owner changed");
            c.binding.set(binding);
            let mut valid=c.valid.ptr();args.push((&mut valid as *mut u64).cast());
            unsafe {
                c.module.function(if queries==16 {"ca_gather_index_k_cached16"}else{"ca_gather_index_k_cached4"})?.launch([192,1,1],[128,1,1],0,stream,&mut args)?;
                c.module.function(if queries==16 {"ca_index_gather_cache_advance16"}else{"ca_index_gather_cache_advance"})?.launch([1,1,1],[32,1,1],0,stream,
                    &mut [(&mut end_ptr as *mut u64).cast(),(&mut queries as *mut i32).cast(),
                        (&mut rows as *mut i32).cast(),(&mut valid as *mut u64).cast()])?;
            }
            output.ready.set(true);
            return Ok(());
        }
        let packed=self.packed.as_ref().filter(|_|[6,16].contains(&queries));
        let module=packed.unwrap_or(&self.module);
        let name=if packed.is_some(){"ca_gather_index_k_packed4"}
            else if ends.is_some(){"ca_gather_index_k_bounded"}else{"ca_gather_index_k"};
        unsafe {
            module.function(name)?.launch(
                [if ends.is_some() {192} else {slots.rows.div_ceil(4) as u32}, 1, 1],
                [128, 1, 1],
                0,
                stream,
                &mut args,
            )
        }?;
        output.ready.set(true);
        Ok(())
    }
}
/// Stable query outputs shared by the producer and scoring plans. Every
/// consumer uses the same ordered stream; retaining these rows needs no copy.
pub struct QueryRows {
    cuda: Cuda,
    rows: usize,
    pub(crate) payload: Buffer<u8>,
    pub(crate) scales: Buffer<i32>,
    pub(crate) weights: Buffer<f32>,
    stream: Cell<usize>,
    pub(crate) ready: Cell<bool>,
}
impl QueryRows {
    pub fn new(cuda: &Cuda, rows: usize) -> Result<Self> {
        ensure!((1..=crate::config::MAX_WIDE_ROWS).contains(&rows), "packed query row count");
        Ok(Self {
            cuda: cuda.clone(),
            rows,
            payload: cuda.allocate(rows * 32 * 64)?,
            scales: cuda.allocate(rows * 32)?,
            weights: cuda.allocate(rows * 32)?,
            stream: Cell::new(0),
            ready: Cell::new(false),
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub(crate) fn bind(&self, stream: &Stream) -> Result<()> {
        ensure!(self.cuda.owns_stream(stream), "query stream/context");
        bind(&self.stream, stream)
    }
    pub fn in_context(&self, cuda: &Cuda) -> bool {
        cuda.owns_buffer(&self.payload)
            && cuda.owns_buffer(&self.scales)
            && cuda.owns_buffer(&self.weights)
    }
}
pub struct QueryPack {
    cuda: Cuda,
    native: Rc<Native>,
    frequency: Rc<FrequencyTable>,
    rows: usize,
    output: Rc<QueryRows>,
}
impl QueryPack {
    pub fn new(
        cuda: &Cuda,
        native: Rc<Native>,
        frequency: Rc<FrequencyTable>,
        rows: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=crate::config::MAX_WIDE_ROWS).contains(&rows)
                && frequency.compressed()
                && cuda.owns_buffer(frequency.values()),
            "index query geometry/context"
        );
        Ok(Self {
            cuda: cuda.clone(),
            native,
            frequency,
            rows,
            output: Rc::new(QueryRows::new(cuda, rows)?),
        })
    }
    pub fn run(
        &self,
        input: &Buffer<bf16>,
        heads: &Buffer<bf16>,
        schedule: &Schedule,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            input.len() == self.rows * 32 * 128
                && heads.len() == self.rows * 32
                && schedule.ready
                && schedule.rows == self.rows
                && schedule.frequency_rows == self.frequency.rows()
                && self.cuda.owns_buffer(input)
                && self.cuda.owns_buffer(heads)
                && self.cuda.owns_buffer(&schedule.positions)
                && self.cuda.owns_stream(stream),
            "index query input/schedule/context"
        );
        self.output.bind(stream)?;
        for start in (0..self.rows).step_by(4096) {
            let rows = (self.rows - start).min(4096);
            let status = unsafe {
                (self.native.query)(
                    rows as i32,
                    (input.ptr() + (start * 32 * 128 * 2) as u64) as _,
                    self.frequency.values().ptr() as _,
                    (schedule.positions.ptr() + (start * 4) as u64) as _,
                    (heads.ptr() + (start * 32 * 2) as u64) as _,
                    (self.output.payload.ptr() + (start * 32 * 64) as u64) as _,
                    (self.output.scales.ptr() + (start * 32 * 4) as u64) as _,
                    (self.output.weights.ptr() + (start * 32 * 4) as u64) as _,
                    stream.raw(),
                )
            };
            ensure!(status == 0, "index Q launch at row{start}: {status}");
        }
        self.output.ready.set(true);
        Ok(())
    }
    pub fn payload(&self) -> &Buffer<u8> {
        &self.output.payload
    }
    pub fn scales(&self) -> &Buffer<i32> {
        &self.output.scales
    }
    pub fn weights(&self) -> &Buffer<f32> {
        &self.output.weights
    }
    pub fn output(&self) -> Rc<QueryRows> {
        self.output.clone()
    }
}
