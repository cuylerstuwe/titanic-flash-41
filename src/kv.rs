//! CUDA normalization/RoPE/store into Rust-owned V4 paged storage. The pinned
//! device kernel quantizes NOPE from FP32 registers, preserving SGLang's boundary.
use crate::{
    attention::PagedKv,
    cuda::{Buffer, Cuda, Module, Stream},
};
use anyhow::{Result, ensure};
use half::bf16;
use libloading::Library;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{ffi::c_void, path::Path};

pub struct RowLocations {
    positions: Buffer<i32>,
    locations: Buffer<i32>,
    rows: usize,
    capacity: usize,
    frequency_rows: usize,
}
impl RowLocations {
    /// Publish metadata in the same stream as its consumers, so a following
    /// chunk cannot overwrite an earlier store's positions or ring locations.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_ring(&mut self,cuda:&Cuda,module:&Module,start:usize,live:usize,
        base:usize,ring:usize,stream:&Stream)->Result<()> {
        ensure!(self.rows<=32&&live>0&&live<=self.rows&&ring>=self.rows
            &&base.checked_add(ring).is_some_and(|n|n<=self.capacity)
            &&start.checked_add(live).is_some_and(|n|n<=self.frequency_rows)
            &&cuda.owns_stream(stream)&&cuda.owns_buffer(&self.positions)&&cuda.owns_buffer(&self.locations),
            "queued KV ring metadata bounds/context");
        let mut positions=self.positions.ptr();let mut locations=self.locations.ptr();
        let mut rows=self.rows as u32;let mut live=live as u32;let mut start=start as u32;
        let mut base=base as u32;let mut ring=ring as u32;
        unsafe{module.function("ca_kv_ring_metadata")?.launch([1,1,1],[32,1,1],0,stream,&mut[
            (&mut positions as *mut u64).cast(),(&mut locations as *mut u64).cast(),
            (&mut rows as *mut u32).cast(),(&mut live as *mut u32).cast(),
            (&mut start as *mut u32).cast(),(&mut base as *mut u32).cast(),(&mut ring as *mut u32).cast()])}
    }
    pub fn update(&mut self, positions: &[i32], locations: &[i32]) -> Result<()> {
        ensure!(
            positions.len() == self.rows
                && locations.len() == self.rows
                && positions
                    .iter()
                    .all(|&p| p >= 0 && (p as usize) < self.frequency_rows),
            "KV metadata update shape/position"
        );
        let mut unique = std::collections::BTreeSet::new();
        ensure!(
            locations
                .iter()
                .all(|&l| l == -1 || (l >= 0 && (l as usize) < self.capacity && unique.insert(l))),
            "KV metadata update aliases/outside storage"
        );
        self.positions.upload(positions)?;
        self.locations.upload(locations)
    }
    pub fn upload(
        cuda: &Cuda,
        positions: &[i32],
        locations: &[i32],
        capacity: usize,
        frequency_rows: usize,
    ) -> Result<Self> {
        let rows = positions.len();
        ensure!(
            rows > 0
                && rows <= 65536
                && rows == locations.len()
                && capacity <= i32::MAX as usize
                && frequency_rows <= i32::MAX as usize,
            "KV row metadata dimensions"
        );
        ensure!(
            positions
                .iter()
                .all(|&x| x >= 0 && (x as usize) < frequency_rows),
            "RoPE position outside table"
        );
        let mut unique = std::collections::BTreeSet::new();
        for &loc in locations {
            ensure!(
                loc == -1 || (loc >= 0 && (loc as usize) < capacity && unique.insert(loc)),
                "KV location out of range or duplicate write target"
            );
        }
        let mut p = cuda.allocate::<i32>(rows)?;
        p.upload(positions)?;
        let mut l = cuda.allocate::<i32>(rows)?;
        l.upload(locations)?;
        Ok(Self {
            positions: p,
            locations: l,
            rows,
            capacity,
            frequency_rows,
        })
    }
}
#[derive(Deserialize)]
struct Manifest {
    abi: u32,
    architecture: u32,
    physical_sms: u32,
    layout: String,
    page_tokens: Vec<usize>,
    kv_quant_source: String,
    epsilon: f32,
    library_sha256: String,
}
type Launch = unsafe extern "C" fn(
    i32,
    i32,
    i64,
    *const c_void,
    *const c_void,
    *const f32,
    *const i32,
    *const i32,
    *mut u8,
    f32,
    *mut c_void,
) -> i32;
pub struct Store {
    _library: Library,
    cuda: Cuda,
    launch: Launch,
    /// V4 (kv-v2) or V4.1 (kv-v41-v1): the window-cache layout this store writes.
    layout: crate::attention::KvLayout,
}
impl Store {
    pub fn open(cuda: &Cuda, root: &Path) -> Result<Self> {
        cuda.qualify_gb10()?;
        let m: Manifest = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        // abi 1: the V4 store (fp32 registers quantized after the weighted RMSNorm). abi 2: the V4.1 store, which
        // rounds the normed values to bf16, applies RoPE in fp32 and rounds again, as model.py does, then writes
        // e4m3 with one ue8m0 scale per 32 values over all 512 dims.
        let layout = match (m.abi, m.layout.as_str(), m.kv_quant_source.as_str()) {
            (1, "dsv4_fp8_448_bf16_rope64_e8m0_footer8", "fp32_registers_after_weighted_rmsnorm") =>
                crate::attention::KvLayout::V4,
            (2, crate::attention::DSV41_MAIN_LAYOUT, "bf16_norm_bf16_rope") => crate::attention::KvLayout::V41,
            _ => anyhow::bail!("KV primitive contract changed"),
        };
        ensure!(
            (m.architecture, m.physical_sms) == (121, 48)
                && m.page_tokens == [64, 256]
                && m.epsilon.to_bits() == 1e-20f32.to_bits(),
            "KV primitive contract changed"
        );
        let p = root.join("libcanada_kv.so");
        ensure!(
            format!("{:x}", Sha256::digest(std::fs::read(&p)?)) == m.library_sha256,
            "KV library hash mismatch"
        );
        let lib = unsafe { Library::new(p) }?;
        let abi: unsafe extern "C" fn() -> i32 = *unsafe { lib.get(b"ca_kv_abi\0") }?;
        ensure!(unsafe { abi() } == m.abi as i32, "KV library ABI mismatch");
        let launch = *unsafe { lib.get(b"ca_kv_store\0") }?;
        Ok(Self {
            _library: lib,
            cuda: cuda.clone(),
            launch,
            layout,
        })
    }
    pub fn run(
        &self,
        input: &Buffer<bf16>,
        stride: usize,
        weight: &Buffer<bf16>,
        frequencies: &Buffer<f32>,
        rows: &RowLocations,
        cache: &mut PagedKv,
        stream: &Stream,
    ) -> Result<()> {
        self.run_at(input, 0, stride, weight, frequencies, rows, cache, stream)
    }
    pub fn run_at(
        &self,
        input: &Buffer<bf16>,
        offset: usize,
        stride: usize,
        weight: &Buffer<bf16>,
        frequencies: &Buffer<f32>,
        rows: &RowLocations,
        cache: &mut PagedKv,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            stride >= 512
                && stride <= i64::MAX as usize
                && input.len()
                    >= (rows.rows - 1)
                        .checked_mul(stride)
                        .and_then(|x| x.checked_add(512))
                        .and_then(|x| x.checked_add(offset))
                        .ok_or_else(|| anyhow::anyhow!("KV input span overflow"))?
                && weight.len() == 512,
            "KV input/weight geometry"
        );
        ensure!(
            frequencies.len() == rows.frequency_rows * 64
                && rows.capacity == cache.capacity()
                && [64, 256].contains(&cache.page_tokens()),
            "KV location/table contract mismatch"
        );
        ensure!(cache.layout() == self.layout, "KV store writes {:?} into a {:?} cache", self.layout, cache.layout());
        ensure!(
            self.cuda.owns_buffer(input)
                && self.cuda.owns_buffer(weight)
                && self.cuda.owns_buffer(frequencies)
                && self.cuda.owns_buffer(&rows.positions)
                && self.cuda.owns_buffer(&rows.locations)
                && self.cuda.owns_buffer(cache.data())
                && self.cuda.owns_stream(stream),
            "KV buffers/stream belong to another context"
        );
        cache.bind_stream(stream)?;
        let status = unsafe {
            (self.launch)(
                rows.rows as i32,
                cache.page_tokens() as i32,
                stride as i64,
                (input.ptr() + offset as u64 * 2) as _,
                weight.ptr() as _,
                frequencies.ptr() as _,
                rows.positions.ptr() as _,
                rows.locations.ptr() as _,
                cache.data().ptr() as _,
                1e-20,
                stream.raw(),
            )
        };
        ensure!(status == 0, "KV CUDA store launch failed: {status}");
        Ok(())
    }
}
