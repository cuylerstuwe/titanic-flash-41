//! Lossless, bounded device copies of explicitly enumerated causal storage.
//! Model owners supply identity, committed-frontier and single-writer checks.
//! This is in-memory checkpointing, not host/SSD parking or crash recovery.
use crate::cuda::{Buffer, Cuda, DeviceElement, Stream};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, marker::PhantomData};

const RESERVE: usize = 8 * 1024usize.pow(3);
const MAX_IMAGE: usize = 2 * 1024usize.pow(3);

/// Release the charge before the actual buffer: until CUDA frees it, this
/// temporarily over-reserves future growth, never the other way around.
struct ImageBuffer {
    _charge: crate::memory::ImageStorageCharge<'static>,
    buffer: Buffer<u8>,
}
impl ImageBuffer {
    fn allocate(cuda: &Cuda, bytes: usize) -> Result<Self> {
        let (_charge, buffer) = crate::memory::allocate_image_storage(bytes, || cuda.allocate(bytes))?;
        Ok(Self { _charge, buffer })
    }
}
impl std::ops::Deref for ImageBuffer {
    type Target = Buffer<u8>;
    fn deref(&self) -> &Self::Target { &self.buffer }
}

#[path = "state_exchange.rs"]
pub(crate) mod exchange;

/// Only returned before snapshot copying or lease mutation. Callers may drop
/// this optional snapshot and continue using the unchanged live model owner.
#[derive(Debug)]
pub(crate) struct CapacityRefused;
impl std::fmt::Display for CapacityRefused {
    fn fmt(&self, f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        f.write_str("state image cannot retain reserved UMA headroom")
    }
}
impl std::error::Error for CapacityRefused {}

#[derive(Clone)]
pub(crate) struct Region<'a> {
    cuda: Cuda,
    descriptor: Descriptor,
    prefix: bool,
    _borrow: PhantomData<&'a ()>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Descriptor {
    name: String,
    pointer: u64,
    bytes: usize,
}
impl<'a> Region<'a> {
    pub(crate) fn bytes(&self)->usize {self.descriptor.bytes}
    pub(crate) fn overlaps(&self,other:&Self)->bool {
        let a=&self.descriptor;let b=&other.descriptor;
        a.pointer < b.pointer+b.bytes as u64 && b.pointer < a.pointer+a.bytes as u64
    }
    /// A monotone prefix with a fixed base address. Unlike rotating rings,
    /// its bytes can be exchanged independently in bounded physical chunks.
    pub(crate) fn exchange_prefix(mut self)->Self {self.prefix=true;self}
    /// Byte-only readback for a finite repeat investigation. This is never
    /// called by serving admission, save or restore.
    pub(crate) fn repeat_digest(&self)->Result<serde_json::Value> {
        ensure!((std::env::var("CANADA_SCRATCH_READOUT_LAYER_TRACE").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_SAMPLER_READOUT_SCREEN").as_deref()==Ok("1"))
            || std::env::var("CANADA_SCRATCH_ATTENTION16_COMMIT_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_INDEPENDENT_OWNER_SCREEN").as_deref()==Ok("1"),
            "repeat region digest requires finite diagnostic admission");
        let mut hash=Sha256::new();
        for offset in (0..self.descriptor.bytes).step_by(16<<20) {
            let bytes=(self.descriptor.bytes-offset).min(16<<20);
            hash.update(unsafe{self.cuda.read_device(self.descriptor.pointer+offset as u64,bytes)}?);
        }
        Ok(serde_json::json!({"name":self.descriptor.name,"bytes":self.descriptor.bytes,
            "sha256":format!("{:x}",hash.finalize())}))
    }
    pub(crate) fn new<T: DeviceElement>(
        cuda: &Cuda,
        name: impl Into<String>,
        buffer: &'a Buffer<T>,
    ) -> Result<Self> {
        Self::byte_range(cuda,name,buffer,0..buffer.bytes())
    }
    /// A live allocation slice. Owners must include every byte a future
    /// operation can read at the saved frontier; this is not tensor arithmetic.
    pub(crate) fn byte_range<T: DeviceElement>(
        cuda: &Cuda, name: impl Into<String>, buffer: &'a Buffer<T>,
        range: std::ops::Range<usize>,
    ) -> Result<Self> {
        ensure!(cuda.owns_buffer(buffer), "state image region context");
        ensure!(range.start<range.end && range.end<=buffer.bytes(),"state image region bounds");
        let pointer=buffer.ptr().checked_add(range.start as u64)
            .context("state image region address overflow")?;
        Ok(Self {
            cuda: cuda.clone(),
            prefix: false,
            descriptor: Descriptor {
                name: name.into(),
                pointer,
                bytes: range.end-range.start,
            },
            _borrow: PhantomData,
        })
    }
}

/// Ordinary immutable cache images keep their existing contiguous copy path.
/// Only unfinished two-owner requests opt into mutable exchange storage.
pub(crate) enum Storage { Copy(Image), Exchange(exchange::Image) }
impl From<Image> for Storage {fn from(value:Image)->Self {Self::Copy(value)}}
impl Storage {
    pub(crate) fn bytes(&self)->usize {match self {Self::Copy(x)=>x.bytes(),Self::Exchange(x)=>x.bytes()}}
    pub(crate) fn manifest(&self)->serde_json::Value {match self {Self::Copy(x)=>x.manifest(),Self::Exchange(x)=>x.manifest()}}
    pub(crate) fn digest(&self)->Result<String> {match self {Self::Copy(x)=>x.digest(),Self::Exchange(x)=>x.digest()}}
    pub(crate) fn validate(&self,regions:&[Region<'_>],stream:&Stream)->Result<()> {
        match self {Self::Copy(x)=>x.validate(regions,stream),Self::Exchange(x)=>x.validate(regions,stream)}
    }
    pub(crate) unsafe fn restore(&self,regions:&[Region<'_>],stream:&Stream)->Result<()> {
        match self {Self::Copy(x)=>unsafe{x.restore(regions,stream)},Self::Exchange(x)=>unsafe{x.restore(regions,stream)}}
    }
    pub(crate) fn validate_import(&self,source:&[Region<'_>],destination:&[Region<'_>],stream:&Stream)->Result<()> {
        match self {Self::Copy(x)=>x.validate_import(source,destination,stream),
            Self::Exchange(_)=>anyhow::bail!("mutable request exchange images cannot be imported")}
    }
    /// Safety: the caller validates the model family and holds exclusive write
    /// authority for the destination, poisoning it on any incomplete copy.
    pub(crate) unsafe fn import(&self,source:&[Region<'_>],destination:&[Region<'_>],stream:&Stream)->Result<()> {
        match self {Self::Copy(x)=>unsafe{x.import(source,destination,stream)},
            Self::Exchange(_)=>anyhow::bail!("mutable request exchange images cannot be imported")}
    }
}

/// An import is a separately admitted copy, never a relaxation of restore's
/// exact-address contract. Both layouts must be complete, equally ordered,
/// and disjoint even across differently named regions.
fn import_layout(source:&[Descriptor],destination:&[Descriptor])->Result<()> {
    layout(source,MAX_IMAGE)?;layout(destination,MAX_IMAGE)?;
    ensure!(source.len()==destination.len()&&source.iter().zip(destination)
        .all(|(a,b)|a.name==b.name&&a.bytes==b.bytes),"import region layout differs");
    ensure!(source.iter().all(|a|destination.iter().all(|b|
        a.pointer+a.bytes as u64<=b.pointer||b.pointer+b.bytes as u64<=a.pointer)),
        "import destination aliases original causal storage");
    Ok(())
}

/// Reject all aliases, including exact ones: each persistent owner must be
/// enumerated once. Silent deduplication could conceal a missing state owner.
fn layout(regions: &[Descriptor], budget: usize) -> Result<(Vec<usize>, usize)> {
    ensure!(
        !regions.is_empty() && budget <= MAX_IMAGE,
        "state image budget"
    );
    let mut names = HashSet::new();
    let mut spans = Vec::new();
    let mut offsets = Vec::new();
    let mut total = 0usize;
    for r in regions {
        ensure!(
            !r.name.is_empty() && names.insert(&r.name),
            "duplicate state image name"
        );
        ensure!(r.pointer != 0 && r.bytes > 0, "empty state image region");
        let end = r
            .pointer
            .checked_add(u64::try_from(r.bytes)?)
            .context("state image address overflow")?;
        offsets.push(total);
        total = total
            .checked_add(r.bytes)
            .context("state image size overflow")?;
        ensure!(total <= budget, "state image exceeds explicit byte budget");
        spans.push((r.pointer, end));
    }
    spans.sort_unstable();
    ensure!(
        spans.windows(2).all(|p| p[0].1 <= p[1].0),
        "aliased state image regions"
    );
    Ok((offsets, total))
}

pub(crate) struct Image {
    cuda: Cuda,
    descriptors: Vec<Descriptor>,
    offsets: Vec<usize>,
    storage: ImageBuffer,
    saved: bool,
}
impl Image {
    /// Admission/allocation are separate from mutation so a budget refusal
    /// need not poison the caller's still-healthy model state.
    pub(crate) fn allocate(
        cuda: &Cuda,
        regions: &[Region<'_>],
        budget: usize,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self::try_allocate(cuda,regions,budget,stream)?.ok_or(CapacityRefused)?)
    }
    /// A refusal never changes the source regions or their writer lease. A
    /// serving caller must agree across ranks before capturing/retaining it.
    pub(crate) fn try_allocate(
        cuda: &Cuda,
        regions: &[Region<'_>],
        budget: usize,
        stream: &Stream,
    ) -> Result<Option<Self>> {
        ensure!(
            cuda.owns_stream(stream) && regions.iter().all(|r| r.cuda.owns_stream(stream)),
            "state image context/stream"
        );
        let descriptors: Vec<_> = regions.iter().map(|r| r.descriptor.clone()).collect();
        let (offsets, total) = layout(&descriptors, budget)?;
        if cuda.admission_memory()?.admitted_available < RESERVE + total { return Ok(None); }
        let storage = ImageBuffer::allocate(cuda, total)?;
        if cuda.admission_memory()?.admitted_available < RESERVE {
            drop(storage);
            return Ok(None);
        }
        Ok(Some(Self {
            cuda: cuda.clone(),
            descriptors,
            offsets,
            storage,
            saved: false,
        }))
    }
    pub(crate) fn validate(&self, regions: &[Region<'_>], stream: &Stream) -> Result<()> {
        ensure!(
            self.cuda.owns_stream(stream)
                && regions.iter().all(|r| r.cuda.owns_stream(stream))
                && self
                    .descriptors
                    .iter()
                    .eq(regions.iter().map(|r| &r.descriptor)),
            "state image storage/stream identity changed"
        );
        Ok(())
    }
    fn validate_import(&self,source:&[Region<'_>],destination:&[Region<'_>],stream:&Stream)->Result<()> {
        self.validate(source,stream)?;
        ensure!(self.saved&&destination.iter().all(|r|r.cuda.owns_stream(stream)),
            "import requires a captured immutable image in the original context");
        let descriptors=destination.iter().map(|r|r.descriptor.clone()).collect::<Vec<_>>();
        import_layout(&self.descriptors,&descriptors)?;
        ensure!(descriptors.iter().all(|r|r.pointer+r.bytes as u64<=self.storage.ptr()
            ||self.storage.ptr()+self.storage.bytes()as u64<=r.pointer),"import would overwrite its immutable image");
        Ok(())
    }
    unsafe fn import(&self,source:&[Region<'_>],destination:&[Region<'_>],stream:&Stream)->Result<()> {
        self.validate_import(source,destination,stream)?;
        let copies=(||{
            for (r,offset)in destination.iter().zip(&self.offsets) {
                unsafe{self.cuda.copy_device(r.descriptor.pointer,self.storage.ptr()+*offset as u64,
                    r.descriptor.bytes,stream)}?;
            }
            Ok(())
        })();
        let completed=stream.synchronize();copies.and(completed)
    }
    /// The owning model must exclude writers on every participating stream.
    pub(crate) fn save(&mut self, regions: &[Region<'_>], stream: &Stream) -> Result<()> {
        self.validate(regions, stream)?;
        ensure!(!self.saved, "state images are immutable after capture");
        let copies = (|| {
            for (r, offset) in self.descriptors.iter().zip(&self.offsets) {
                unsafe {
                    self.cuda.copy_device(
                        self.storage.ptr() + *offset as u64,
                        r.pointer,
                        r.bytes,
                        stream,
                    )
                }?;
            }
            Ok(())
        })();
        let completed = stream.synchronize();
        copies.and(completed)?;
        self.saved = true;
        Ok(())
    }
    /// Safety: the caller owns exclusive write authority over every supplied
    /// destination and has joined all producer/consumer streams. It must
    /// poison that authority on any failure and restore matching host state.
    pub(crate) unsafe fn restore(&self, regions: &[Region<'_>], stream: &Stream) -> Result<()> {
        self.validate(regions, stream)?;
        ensure!(self.saved, "uncaptured state image");
        let copies = (|| {
            for (r, offset) in self.descriptors.iter().zip(&self.offsets) {
                unsafe {
                    self.cuda.copy_device(
                        r.pointer,
                        self.storage.ptr() + *offset as u64,
                        r.bytes,
                        stream,
                    )
                }?;
            }
            Ok(())
        })();
        let completed = stream.synchronize();
        copies.and(completed)
    }
    pub(crate) fn bytes(&self) -> usize {
        self.storage.bytes()
    }
    pub(crate) fn manifest(&self) -> serde_json::Value {
        serde_json::json!({"bytes":self.bytes(),"regions":self.descriptors.iter()
            .map(|r| serde_json::json!({"name":r.name,"bytes":r.bytes})).collect::<Vec<_>>()})
    }
    /// Bounded diagnostic reads only; no CPU execution or numeric conversion.
    pub(crate) fn digest(&self) -> Result<String> {
        ensure!(self.saved, "uncaptured state image digest");
        let mut hash = Sha256::new();
        for offset in (0..self.bytes()).step_by(16 * 1024 * 1024) {
            hash.update(
                self.storage
                    .download_part(offset, (self.bytes() - offset).min(16 * 1024 * 1024))?,
            );
        }
        Ok(format!("{:x}", hash.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn r(name: &str, pointer: u64, bytes: usize) -> Descriptor {
        Descriptor {
            name: name.into(),
            pointer,
            bytes,
        }
    }
    #[test]
    fn rejects_aliases_overflows_duplicate_names_and_excess_budget() {
        assert_eq!(
            layout(&[r("a", 256, 32), r("b", 288, 16)], 48).unwrap(),
            (vec![0, 32], 48)
        );
        for bad in [
            vec![r("a", 256, 32), r("b", 256, 32)],
            vec![r("a", 256, 32), r("b", 280, 16)],
            vec![r("a", 256, 32), r("a", 288, 16)],
            vec![r("a", u64::MAX, 1)],
            vec![r("a", 0, 1)],
            vec![r("a", 256, 0)],
        ] {
            assert!(layout(&bad, 128).is_err());
        }
        assert!(layout(&[r("a", 256, 32)], 31).is_err());
        assert!(layout(&[], 128).is_err());
        assert!(layout(&[r("a", 256, 32)], MAX_IMAGE + 1).is_err());
    }
    #[test]
    fn import_requires_matching_regions_and_independent_destination() {
        let source=[r("a",256,32),r("b",512,16)];
        assert!(import_layout(&source,&[r("a",1024,32),r("b",2048,16)]).is_ok());
        for destination in [vec![r("a",1024,32)],vec![r("b",1024,32),r("a",2048,16)],
            vec![r("a",1024,31),r("b",2048,16)],vec![r("a",512,32),r("b",2048,16)],
            vec![r("a",1024,32),r("b",1040,16)],source.to_vec()] {
            assert!(import_layout(&source,&destination).is_err());
        }
    }
}
