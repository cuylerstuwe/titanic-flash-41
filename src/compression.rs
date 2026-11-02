//! Rust-owned low-ratio compressor state. The selected CUDA primitive only
//! performs pooling/norm/RoPE/store; admission, rings, pages and ordering live here.
use crate::{
    attention::PagedKv,
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, Module, Stream},
    prelude::FrequencyTable,
    triton_ops::Operator,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use libloading::Library;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::Cell, ffi::c_void, path::Path, rc::Rc};
type C1 = unsafe extern "C" fn(
    i32,
    i32,
    *const c_void,
    *const c_void,
    *const i32,
    *const i64,
    *const f32,
    *mut u8,
    *mut c_void,
    *mut c_void,
) -> i32;
type C2 = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    *const f32,
    *mut f32,
    *const c_void,
    *const i32,
    *const i64,
    *const i64,
    *const f32,
    *mut u8,
    *mut c_void,
    *mut c_void,
) -> i32;
#[derive(Deserialize)]
struct Contract {
    abi: u32,
    architecture: u32,
    physical_sms: u32,
    head_dim: usize,
    rope_dim: usize,
    position_dtype: String,
    location_dtype: String,
    request_dtype: String,
    page_tokens: Vec<usize>,
    layout: String,
    pdl: bool,
    fast_math: bool,
    eps: f32,
    library_sha256: String,
}
struct Native {
    _lib: Library,
    c1: C1,
    c2: C2,
}
impl Native {
    fn open(root: &Path) -> Result<Self> {
        let c: Contract = serde_json::from_slice(&std::fs::read(root.join("native.json"))?)?;
        ensure!(
            (
                c.abi,
                c.architecture,
                c.physical_sms,
                c.head_dim,
                c.rope_dim
            ) == (1, 121, 48, 512, 64)
                && c.position_dtype == "i32"
                && c.location_dtype == "i64"
                && c.request_dtype == "i64"
                && c.page_tokens == [32, 64, 128, 256]
                && !c.pdl
                && !c.fast_math
                && c.eps.to_bits() == 1e-20f32.to_bits()
                && c.layout == "dsv4_fp8_448_bf16_rope64_e8m0_footer8",
            "compressor primitive contract mismatch"
        );
        let p = root.join("libcanada_compression.so");
        ensure!(
            format!("{:x}", Sha256::digest(std::fs::read(&p)?)) == c.library_sha256,
            "compressor library hash"
        );
        let lib = unsafe { Library::new(p) }?;
        let abi: unsafe extern "C" fn() -> i32 = *unsafe { lib.get(b"ca_compression_abi\0") }?;
        ensure!(unsafe { abi() } == 1, "compressor ABI mismatch");
        Ok(Self {
            c1: *unsafe { lib.get(b"ca_compress1\0") }?,
            c2: *unsafe { lib.get(b"ca_compress2\0") }?,
            _lib: lib,
        })
    }
}
pub struct Metadata {
    positions: Buffer<i32>,
    requests: Buffer<i64>,
    locations: Buffer<i64>,
    rows: usize,
    ratio: usize,
    capacity: usize,
    request_slots: usize,
    ring: usize,
    draft: usize,
    frequency_rows: usize,
    ready: bool,
    single_live_first: bool,
}
impl Metadata {
    pub fn new(
        cuda: &Cuda,
        ratio: usize,
        capacity: usize,
        request_slots: usize,
        ring: usize,
        draft: usize,
        frequency_rows: usize,
        positions: &[i32],
        requests: &[i64],
        locations: &[i64],
    ) -> Result<Self> {
        ensure!(
            [1, 2].contains(&ratio)
                && (1..=crate::config::MAX_WIDE_ROWS).contains(&positions.len())
                && capacity > 0
                && capacity <= i32::MAX as usize
                && (1..=128).contains(&request_slots)
                && frequency_rows > 0
                && frequency_rows <= 1048576
                && ((ratio == 1 && draft == 1)
                    || (ratio == 2
                        && (2..=64).contains(&ring)
                        && (1..=32).contains(&draft)
                        && positions.len() % draft == 0
                        && (draft == 1 || ring > draft))),
            "compressor metadata geometry"
        );
        let mut m = Self {
            positions: cuda.allocate(positions.len())?,
            requests: cuda.allocate(positions.len())?,
            locations: cuda.allocate(positions.len())?,
            rows: positions.len(),
            ratio,
            capacity,
            request_slots,
            ring,
            draft,
            frequency_rows,
            ready: false,
            single_live_first: false,
        };
        m.update(positions, requests, locations)?;
        Ok(m)
    }
    pub fn update(&mut self, positions: &[i32], requests: &[i64], locations: &[i64]) -> Result<()> {
        self.ready = false;
        ensure!(
            positions.len() == self.rows
                && requests.len() == self.rows
                && locations.len() == self.rows,
            "compressor metadata row count"
        );
        ensure!(
            positions
                .iter()
                .all(|p| *p >= 0 && (*p as usize) < self.frequency_rows)
                && requests
                    .iter()
                    .all(|r| *r >= 0 && (*r as usize) < self.request_slots),
            "compressor position/request bounds"
        );
        let mut unique = std::collections::BTreeSet::new();
        for (i, &loc) in locations.iter().enumerate() {
            ensure!(
                loc == 0
                    || (loc >= self.ratio as i64
                        && loc <= i32::MAX as i64
                        && (loc as usize) < self.capacity * self.ratio
                        && unique.insert(loc)),
                "compressor cache location/alias"
            );
            if self.ratio == 2 && loc != 0 {
                ensure!(
                    loc % 2 == positions[i] as i64 % 2,
                    "pair position/location parity"
                );
            }
        }
        if self.ratio == 2 {
            let mut active_requests = std::collections::BTreeSet::new();
            for start in (0..self.rows).step_by(self.draft) {
                let active = (0..self.draft)
                    .take_while(|i| locations[start + i] != 0)
                    .count();
                ensure!(
                    locations[start + active..start + self.draft]
                        .iter()
                        .all(|loc| *loc == 0),
                    "verify padding must be a suffix"
                );
                if active == 0 {
                    continue;
                }
                ensure!(
                    active_requests.insert(requests[start]),
                    "duplicate request would race the pair-state ring"
                );
                for i in 0..active {
                    ensure!(
                        requests[start + i] == requests[start]
                            && positions[start + i] == positions[start] + i as i32,
                        "verify request/position block is not consecutive"
                    );
                    if i > 0 && positions[start + i] % 2 == 1 {
                        ensure!(
                            locations[start + i] == locations[start + i - 1] + 1,
                            "completed pair must share adjacent full slots"
                        );
                    }
                }
            }
        }
        self.positions.upload(positions)?;
        self.requests.upload(requests)?;
        self.locations.upload(locations)?;
        self.single_live_first = locations[0] != 0 && locations[1..].iter().all(|&v| v == 0);
        self.ready = true;
        Ok(())
    }
}
pub enum Input<'a> {
    Ratio1(&'a Buffer<bf16>),
    Ratio2(&'a Buffer<f32>),
}
pub struct Source {
    cuda: Cuda,
    _weights: Rc<Weights>,
    native: Native,
    frequencies: Rc<FrequencyTable>,
    weight: u64,
    ratio: usize,
    rows: usize,
    request_slots: usize,
    ring_size: usize,
    ring: Option<Rc<crate::pair_prefill::Ring>>,
    cache: PagedKv,
    latent: Buffer<bf16>,
    stream: Cell<usize>,
}
/// A typed wider-prefill schedule tied to one source's pointer owners. Pooling
/// reads all carried values before a second launch publishes the ring tail.
/// Normalization and cache publication consume only completing odd positions.
pub struct PairPrefill {
    pool: Option<crate::pair_prefill::Pool>,
    schedule: Option<crate::pair_prefill::Schedule>,
    positions: Buffer<i32>,
    locations: Buffer<i64>,
    module: Module,
    norm: Operator,
    store: PrefillStore,
    owner_latent: u64,
    owner_cache: u64,
    owner_ring: u64,
    rows: usize,
    ratio: usize,
    capacity: usize,
    frequency_rows: usize,
    request_slots: usize,
}
#[derive(Deserialize)]
struct StoreContract {
    abi: u32,
    architecture: u32,
    physical_sms: u32,
    width: usize,
    page_tokens: Vec<usize>,
    norm: bool,
    input_dtype: String,
    layout: String,
    library_sha256: String,
}
type StoreFn = unsafe extern "C" fn(
    i32,
    i32,
    *const c_void,
    *const i32,
    *const i64,
    *const f32,
    *mut u8,
    *mut c_void,
) -> i32;
struct PrefillStore {
    _lib: Library,
    launch: StoreFn,
}
impl PrefillStore {
    fn open(root: &Path) -> Result<Self> {
        let c: StoreContract = serde_json::from_slice(&std::fs::read(root.join("store.json"))?)?;
        ensure!(
            (c.abi, c.architecture, c.physical_sms, c.width) == (1, 121, 48, 512)
                && c.page_tokens == [32, 64, 128, 256]
                && !c.norm
                && c.input_dtype == "BF16"
                && c.layout == "dsv4_fp8_448_bf16_rope64_e8m0_footer8",
            "prefill store contract"
        );
        let path = root.join("libcanada_prefill_store.so");
        ensure!(
            format!("{:x}", Sha256::digest(std::fs::read(&path)?)) == c.library_sha256,
            "prefill store digest"
        );
        let lib = unsafe { Library::new(path) }?;
        let abi: unsafe extern "C" fn() -> i32 = *unsafe { lib.get(b"ca_prefill_store_abi\0") }?;
        ensure!(unsafe { abi() } == 1, "prefill store ABI");
        Ok(Self {
            launch: *unsafe { lib.get(b"ca_prefill_store\0") }?,
            _lib: lib,
        })
    }
}
impl PairPrefill {
    pub fn update(
        &mut self,
        positions: &[i32],
        requests: &[i64],
        raw_locations: &[i64],
    ) -> Result<()> {
        if let Some(schedule) = self.schedule.as_mut() {
            schedule.update(positions, requests, raw_locations)?;
        } else {
            Self::validate_single(
                self.rows,
                self.capacity,
                self.frequency_rows,
                self.request_slots,
                positions,
                requests,
                raw_locations,
            )?;
        }
        self.positions.upload(
            &positions
                .iter()
                .zip(raw_locations)
                .map(|(&p, &l)| {
                    if self.ratio == 1 {
                        p
                    } else if l != 0 && p % 2 == 1 {
                        p - 1
                    } else {
                        0
                    }
                })
                .collect::<Vec<_>>(),
        )?;
        self.locations.upload(
            &positions
                .iter()
                .zip(raw_locations)
                .map(|(&p, &l)| {
                    if self.ratio == 1 {
                        l
                    } else if l != 0 && p % 2 == 1 {
                        l / 2
                    } else {
                        0
                    }
                })
                .collect::<Vec<_>>(),
        )
    }
    fn validate_single(
        rows: usize,
        capacity: usize,
        frequency: usize,
        requests: usize,
        pos: &[i32],
        req: &[i64],
        loc: &[i64],
    ) -> Result<()> {
        let mut unique = std::collections::BTreeSet::new();
        ensure!(
            pos.len() == rows
                && req.len() == rows
                && loc.len() == rows
                && pos.iter().all(|&p| p >= 0 && (p as usize) < frequency)
                && req.iter().all(|&r| r >= 0 && (r as usize) < requests)
                && loc.iter().all(|&l| l == 0
                    || (l > 0
                        && (l as usize) < capacity
                        && l <= i32::MAX as i64
                        && unique.insert(l))),
            "single prefill source metadata domain/alias"
        );
        Ok(())
    }
    pub fn new(
        cuda: &Cuda,
        core: &[u8],
        kernels: &Path,
        source: &Source,
        positions: &[i32],
        requests: &[i64],
        raw_locations: &[i64],
    ) -> Result<Self> {
        ensure!(
            [1, 2].contains(&source.ratio)
                && positions.len() == source.rows
                && cuda.owns_buffer(&source.latent),
            "prefill compressor source/shape"
        );
        let schedule = if source.ratio == 2 {
            Some(crate::pair_prefill::Schedule::new(
                cuda,
                positions,
                requests,
                raw_locations,
                source.request_slots,
                source.ring_size,
                source
                    .cache
                    .capacity()
                    .checked_mul(2)
                    .ok_or_else(|| anyhow::anyhow!("prefill raw capacity overflow"))?,
                source.frequencies.rows(),
            )?)
        } else {
            Self::validate_single(
                source.rows,
                source.cache.capacity(),
                source.frequencies.rows(),
                source.request_slots,
                positions,
                requests,
                raw_locations,
            )?;
            None
        };
        let mut p = cuda.allocate(positions.len())?;
        let mut l = cuda.allocate(positions.len())?;
        p.upload(
            &positions
                .iter()
                .zip(raw_locations)
                .map(|(&p, &l)| {
                    if source.ratio == 1 {
                        p
                    } else if l != 0 && p % 2 == 1 {
                        p - 1
                    } else {
                        0
                    }
                })
                .collect::<Vec<_>>(),
        )?;
        l.upload(
            &positions
                .iter()
                .zip(raw_locations)
                .map(|(&p, &l)| {
                    if source.ratio == 1 {
                        l
                    } else if l != 0 && p % 2 == 1 {
                        l / 2
                    } else {
                        0
                    }
                })
                .collect::<Vec<_>>(),
        )?;
        let primitive = kernels.join("compression-prefill-v1");
        let norm = Operator::load(cuda, &primitive, "prefill_rms512")?;
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(primitive.join("manifest.json"))?)?;
        let contract = &manifest["entries"][0];
        ensure!(
            contract["constexprs"] == serde_json::json!({"D":512,"EPS":1e-20,"B":512})
                && contract["fp_fusion"] == false
                && norm.entry.metadata.num_warps == 4
                && norm
                    .entry
                    .arguments
                    .iter()
                    .map(|a| (a.name.as_str(), a.kind.as_str()))
                    .eq([("X", "*bf16"), ("W", "*bf16"), ("Y", "*bf16")]),
            "actual prefill RMS precision/ABI contract"
        );
        Ok(Self {
            pool: if source.ratio == 2 {
                Some(crate::pair_prefill::Pool::new(
                    cuda,
                    &kernels.join("prefill-pair-v2"),
                    source.rows,
                    source.ring_size,
                )?)
            } else {
                None
            },
            schedule,
            positions: p,
            locations: l,
            module: cuda.module(core)?,
            norm,
            store: PrefillStore::open(&primitive)?,
            owner_latent: source.latent.ptr(),
            owner_cache: source.cache.data().ptr(),
            owner_ring: source.ring.as_ref().map_or(0, |r| r.values().ptr()),
            rows: source.rows,
            ratio: source.ratio,
            capacity: source.cache.capacity(),
            frequency_rows: source.frequencies.rows(),
            request_slots: source.request_slots,
        })
    }
}
impl Source {
    pub(crate) fn matches_static(
        &self,
        weights: &Rc<Weights>,
        frequency: &Rc<FrequencyTable>,
    ) -> bool {
        Rc::ptr_eq(&self._weights, weights) && Rc::ptr_eq(&self.frequencies, frequency)
    }
    /// New row workspace over the same ordered causal cache and carried ring.
    /// This shares state; it does not create an independently writable branch.
    pub(crate) fn share_rows(&self, cuda: &Cuda, rows: usize, root: &Path) -> Result<Self> {
        ensure!(
            (1..=crate::config::MAX_WIDE_ROWS).contains(&rows) && self._weights.in_context(cuda),
            "shared compressor context/rows"
        );
        let mut latent = cuda.allocate::<bf16>(rows * 512)?;
        latent.zero()?;
        Ok(Self {
            cuda: cuda.clone(),
            _weights: self._weights.clone(),
            native: Native::open(root)?,
            frequencies: self.frequencies.clone(),
            weight: self.weight,
            ratio: self.ratio,
            rows,
            request_slots: self.request_slots,
            ring_size: self.ring_size,
            ring: self.ring.clone(),
            cache: self.cache.clone(),
            latent,
            stream: Cell::new(0),
        })
    }

    pub fn new(
        cuda: &Cuda,
        weights: Rc<Weights>,
        layer: usize,
        rows: usize,
        request_slots: usize,
        ring_size: usize,
        pages: usize,
        page_tokens: usize,
        frequencies: Rc<FrequencyTable>,
        root: &Path,
    ) -> Result<Self> {
        ensure!(
            [2, 8, 14, 20].contains(&layer)
                && (1..=crate::config::MAX_WIDE_ROWS).contains(&rows)
                && (1..=128).contains(&request_slots)
                && [32, 64, 128, 256].contains(&page_tokens)
                && frequencies.compressed()
                && cuda.owns_buffer(frequencies.values())
                && weights.in_context(cuda),
            "compressor source geometry/context"
        );
        let ratio = if layer == 20 { 1 } else { 2 };
        ensure!(
            ratio == 1 || (2..=64).contains(&ring_size),
            "pair ring capacity"
        );
        let name = format!("layers.{layer}.attn.compressor.norm.weight");
        let view = weights.view(&name)?;
        ensure!(
            view.dtype == Dtype::Bf16 && view.shape == [512],
            "compressor norm weight shape/dtype"
        );
        let weight = weights.ptr(&name)?;
        let mut latent = cuda.allocate::<bf16>(rows * 512)?;
        latent.zero()?;
        let ring = if ratio == 2 {
            Some(Rc::new(crate::pair_prefill::Ring::new(
                cuda,
                request_slots,
                ring_size,
            )?))
        } else {
            None
        };
        Ok(Self {
            cuda: cuda.clone(),
            _weights: weights,
            native: Native::open(root)?,
            frequencies,
            weight,
            ratio,
            rows,
            request_slots,
            ring_size,
            ring,
            cache: PagedKv::new(cuda, pages, page_tokens)?,
            latent,
            stream: Cell::new(0),
        })
    }
    pub fn run(&mut self, input: Input<'_>, meta: &Metadata, stream: &Stream) -> Result<()> {
        let rows = self.rows;
        self.run_rows(input, meta, rows, stream)
    }
    pub(crate) fn run_single_decode(
        &mut self,
        input: Input<'_>,
        meta: &Metadata,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            meta.draft == 1 && meta.single_live_first,
            "decode compressor needs one live first row"
        );
        self.run_rows(input, meta, 1, stream)
    }
    fn run_rows(
        &mut self,
        input: Input<'_>,
        meta: &Metadata,
        execution_rows: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            meta.ready
                && meta.ratio == self.ratio
                && meta.rows == self.rows
                && meta.capacity == self.cache.capacity()
                && meta.request_slots == self.request_slots
                && meta.frequency_rows == self.frequencies.rows()
                && (self.ratio == 1 || meta.ring == self.ring_size)
                && self.cuda.owns_buffer(&meta.positions)
                && self.cuda.owns_buffer(&meta.locations)
                && self.cuda.owns_buffer(&meta.requests)
                && self.cuda.owns_stream(stream),
            "compressor metadata/context mismatch"
        );
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == raw,
            "compressor state requires one ordered stream"
        );
        self.stream.set(raw);
        self.cache.bind_stream(stream)?;
        let status = match input {
            Input::Ratio1(b) => {
                ensure!(
                    self.ratio == 1 && b.len() == self.rows * 512 && self.cuda.owns_buffer(b),
                    "ratio1 projection input"
                );
                unsafe {
                    (self.native.c1)(
                        execution_rows as i32,
                        self.cache.page_tokens() as i32,
                        b.ptr() as _,
                        self.weight as _,
                        meta.positions.ptr() as _,
                        meta.locations.ptr() as _,
                        self.frequencies.values().ptr() as _,
                        self.cache.data().ptr() as _,
                        self.latent.ptr() as _,
                        stream.raw(),
                    )
                }
            }
            Input::Ratio2(b) => {
                ensure!(
                    self.ratio == 2 && b.len() == self.rows * 1024 && self.cuda.owns_buffer(b),
                    "ratio2 projection input"
                );
                self.ring.as_ref().unwrap().bind(stream)?;
                unsafe {
                    (self.native.c2)(
                        execution_rows as i32,
                        self.cache.page_tokens() as i32,
                        self.ring_size as i32,
                        meta.draft as i32,
                        b.ptr() as _,
                        self.ring.as_ref().unwrap().values().ptr() as _,
                        self.weight as _,
                        meta.positions.ptr() as _,
                        meta.requests.ptr() as _,
                        meta.locations.ptr() as _,
                        self.frequencies.values().ptr() as _,
                        self.cache.data().ptr() as _,
                        self.latent.ptr() as _,
                        stream.raw(),
                    )
                }
            }
        };
        ensure!(status == 0, "compressor CUDA launch failed: {status}");
        Ok(())
    }
    pub(crate) fn ring_size(&self) -> usize { self.ring_size }
    pub fn latent(&self) -> &Buffer<bf16> {
        &self.latent
    }
    /// Wider prefill has a different pooling precision boundary from the
    /// fused verify kernel. This bridge is an implementation candidate until
    /// checked against actual qualified prefill, including carried ring state.
    pub fn run_pair_prefill(
        &mut self,
        input: &Buffer<f32>,
        plan: &PairPrefill,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            self.ratio == 2
                && plan.ratio == 2
                && self.rows == plan.rows
                && input.len() == self.rows * 1024
                && plan.owner_latent == self.latent.ptr()
                && plan.owner_cache == self.cache.data().ptr()
                && plan.owner_ring == self.ring.as_ref().unwrap().values().ptr()
                && self.cuda.owns_buffer(input)
                && self.cuda.owns_stream(stream),
            "prefill compressor pointer/shape/context owner changed"
        );
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == raw,
            "prefill compressor needs one ordered stream"
        );
        self.stream.set(raw);
        self.cache.bind_stream(stream)?;
        plan.pool.as_ref().unwrap().run(
            input,
            self.ring.as_ref().unwrap(),
            plan.schedule.as_ref().unwrap(),
            stream,
        )?;
        self.finish_prefill(plan.pool.as_ref().unwrap().output(), plan, stream)
    }
    pub fn run_single_prefill(
        &mut self,
        input: &Buffer<bf16>,
        plan: &PairPrefill,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            self.ratio == 1 && plan.ratio == 1,
            "single prefill source phase/ratio"
        );
        self.finish_prefill(input, plan, stream)
    }
    fn finish_prefill(
        &mut self,
        input: &Buffer<bf16>,
        plan: &PairPrefill,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            input.len() == self.rows * 512
                && self.rows == plan.rows
                && self.ratio == plan.ratio
                && plan.owner_latent == self.latent.ptr()
                && plan.owner_cache == self.cache.data().ptr()
                && self.cuda.owns_buffer(input)
                && self.cuda.owns_stream(stream),
            "prefill finish source/context owner"
        );
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == raw,
            "prefill finish needs one ordered stream"
        );
        self.stream.set(raw);
        self.cache.bind_stream(stream)?;
        // Prefill uses the actual four-warp Triton RMS tree. The fused c1/c2
        // verify RMS tree can differ at BF16 ties on identical operands.
        unsafe {
            plan.norm.launch(
                self.rows as u32,
                &[input.ptr(), self.weight, self.latent.ptr()],
                stream,
            )
        }?;
        let mut latent = self.latent.ptr();
        let mut locations = plan.locations.ptr();
        let mut rows = self.rows as i32;
        let mut args = [
            (&mut latent as *mut u64).cast(),
            (&mut locations as *mut u64).cast(),
            (&mut rows as *mut i32).cast(),
        ];
        unsafe {
            plan.module.function("ca_mask_compressed_latent")?.launch(
                [((self.rows * 512).div_ceil(256)) as u32, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )
        }?;
        // Odd positions represent pos-1 and publish raw/2. This selected
        // primitive performs only RoPE, FP4 round trip and FP8 cache packing.
        // The pinned store adapter admits at most4096 independent rows per
        // call. Keep its kernel and absolute cache addressing unchanged while
        // advancing only the row-local inputs for a larger encoder profile.
        for start in (0..self.rows).step_by(4096) {
            let rows = (self.rows - start).min(4096);
            let status = unsafe {
                (plan.store.launch)(
                    rows as i32,
                    self.cache.page_tokens() as i32,
                    (self.latent.ptr() + (start * 512 * 2) as u64) as _,
                    (plan.positions.ptr() + (start * 4) as u64) as _,
                    (plan.locations.ptr() + (start * 8) as u64) as _,
                    self.frequencies.values().ptr() as _,
                    self.cache.data().ptr() as _,
                    stream.raw(),
                )
            };
            ensure!(status == 0, "prefill compressed cache store at row{start} failed: {status}");
        }
        Ok(())
    }
    pub(crate) fn clear_state(&mut self, stream: &Stream) -> Result<()> {
        ensure!(
            self.cuda.owns_stream(stream)
                && (self.stream.get() == 0 || self.stream.get() == stream.raw() as usize),
            "compressor reset stream"
        );
        self.cache.clear(&self.cuda, stream)?;
        unsafe {
            self.cuda
                .zero_device(self.latent.ptr(), self.latent.bytes(), stream)?;
            if let Some(ring) = &self.ring {
                ring.bind(stream)?;
                self.cuda
                    .zero_device(ring.values().ptr(), ring.values().bytes(), stream)?;
            }
        }
        Ok(())
    }
    pub(crate) fn latent_mut(&mut self) -> &mut Buffer<bf16> {
        &mut self.latent
    }
    pub fn cache(&self) -> &PagedKv {
        &self.cache
    }
    pub(crate) fn cache_mut(&mut self) -> &mut PagedKv {
        &mut self.cache
    }
    pub fn ring(&self) -> Option<&Buffer<f32>> {
        self.ring.as_ref().map(|r| r.values())
    }
    pub(crate) fn ring_mut(&mut self) -> Option<&mut Buffer<f32>> {
        self.ring
            .as_mut()
            .and_then(Rc::get_mut)
            .map(|r| r.values_mut())
    }
    pub fn pair_state_mut(&mut self) -> Option<&mut crate::pair_prefill::Ring> {
        self.ring.as_mut().and_then(Rc::get_mut)
    }
}
