//! Lossless, bounded device copies of explicitly enumerated causal storage.
//! Model owners supply identity, committed-frontier and single-writer checks.
//! This is in-memory checkpointing, not host/SSD parking or crash recovery.
use crate::cuda::{Buffer, Cuda, DeviceElement, Module, Stream};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, marker::PhantomData};

const MAX_IMAGE: usize = 2 * 1024usize.pow(3);

/// Release the charge before the actual buffer: until CUDA frees it, this
/// temporarily over-reserves future growth, never the other way around.
struct ImageBuffer {
    _charge: crate::memory::ImageStorageCharge<'static>,
    buffer: Buffer<u8>,
}
impl ImageBuffer {
    fn allocate(cuda: &Cuda, bytes: usize) -> Result<Self> {
        let (_charge, buffer) = crate::memory::allocate_image_storage(bytes, || match cuda.allocate_pooled(bytes)? {
            Some(pooled) => Ok(pooled),
            None => cuda.allocate(bytes),
        })?;
        Ok(Self { _charge, buffer })
    }
}
impl std::ops::Deref for ImageBuffer {
    type Target = Buffer<u8>;
    fn deref(&self) -> &Self::Target { &self.buffer }
}
impl ImageBuffer {
    fn buffer_mut(&mut self) -> &mut Buffer<u8> { &mut self.buffer }
}
/// The allocator takes the whole buffer from one source. Even a fully pooled
/// image must leave the ordinary host reserve intact.
pub(crate) fn image_storage_fits(available: usize, reserve: usize, bytes: usize, pooled: bool) -> bool {
    available.checked_sub(reserve).is_some_and(|spare| pooled || bytes <= spare)
}

/// Image storage under the allocator's admission line (see Image::try_allocate); None: no room now.
fn allocate_storage(cuda: &Cuda, total: usize) -> Result<Option<ImageBuffer>> {
    let reserve = crate::memory::floor_bytes().max(cuda.allocation_reserve_bytes());
    let available = cuda.admission_memory()?.admitted_available;
    if !image_storage_fits(available, reserve, total, cuda.image_pool_fits(total)) { return Ok(None); }
    let storage = match ImageBuffer::allocate(cuda, total) {
        Ok(storage) => storage,
        Err(error) if crate::cuda::allocation_floor_refused(&error) => return Ok(None),
        Err(error) => return Err(error),
    };
    if cuda.admission_memory()?.admitted_available < crate::memory::floor_bytes() {
        drop(storage);
        return Ok(None);
    }
    Ok(Some(storage))
}

#[path = "state_exchange.rs"]
pub(crate) mod exchange;
#[path = "state_image_copy.rs"]
pub(crate) mod copy;

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
    /// The copied image (disk tier: parking applies to copied images only, never to exchange storage).
    pub(crate) fn image_mut(&mut self)->Option<&mut Image> {match self {Self::Copy(x)=>Some(x),Self::Exchange(_)=>None}}
    pub(crate) fn image(&self)->Option<&Image> {match self {Self::Copy(x)=>Some(x),Self::Exchange(_)=>None}}
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
    /// The device copy; None while the image is parked on disk (disk tier).
    device: Option<ImageBuffer>,
    /// Logical size (the device copy's bytes when present).
    total: usize,
    /// The disk tier's copy (crate::disk_tier), when one was written.
    disk: Option<crate::disk_tier::DiskFile>,
    saved: bool,
    copy_module: Option<Module>,
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
        // Admit against the line the allocator itself enforces (the floor plus its allowance for untracked host
        // activity, Cuda::allocation_reserve_bytes), not the bare floor. With the bare floor an image inside the
        // 512 MiB between the two was admitted here and then refused inside the allocation, which is not a
        // CapacityRefused and retired the worker group: the release stress panel of Nov 14 2026, first pause of
        // a 974K-token conversation with 8.99 GiB available on the vision rank (0.84 GiB image). A refusal by
        // the guard itself (memory moved between this reading and the allocation) is the same refusal.
        // A snapshot the startup pool will hold (CANADA_SCRATCH_IMAGE_POOL_MIB) takes no host memory: host memory must
        // still keep its reserve, but not the snapshot's own bytes (allocate_storage).
        let Some(storage) = allocate_storage(cuda, total)? else { return Ok(None) };
        Ok(Some(Self {
            cuda: cuda.clone(),
            descriptors,
            offsets,
            device: Some(storage),
            total,
            disk: None,
            saved: false,
            copy_module: None,
        }))
    }
    /// Displaced-checkpoint materialization (CANADA_SCRATCH_DISPLACED_CHECKPOINT): a complete
    /// image with exactly the `full` descriptors, assembled from a captured bank-retained image
    /// of the same frontier. Every region the retained image holds is copied from it. Each
    /// monotone prefix region it holds only the final part of (the bank-retained compressed
    /// pages) gets its earlier bytes from the live allocation, which the caller guarantees
    /// still belongs to the unchanged bank lineage (bytes below the final page are never
    /// rewritten after capture). The result restores through the ordinary validated path.
    pub(crate) fn try_materialize(cuda: &Cuda, full: &[Region<'_>], retained: &Image, budget: usize,
        module: Option<&Module>, stream: &Stream) -> Result<Option<Self>> {
        ensure!(retained.saved && retained.resident(), "materialization needs a captured resident retained image");
        let Some(mut image) = Self::try_allocate(cuda, full, budget, stream)? else { return Ok(None) };
        if let Some(module) = module { image.configure_copy_kernel(module)?; }
        let mut used = vec![false; retained.descriptors.len()];
        let copies = (|| -> Result<()> {
            for (region, offset) in full.iter().zip(&image.offsets) {
                let f = &region.descriptor;
                let (i, r) = retained.descriptors.iter().enumerate().find(|(_, r)| r.name == f.name)
                    .with_context(|| format!("retained image lacks region {}", f.name))?;
                ensure!(!used[i], "retained region used twice");
                used[i] = true;
                let destination = image.storage().ptr() + *offset as u64;
                let source = retained.storage().ptr() + retained.offsets[i] as u64;
                if r.pointer == f.pointer && r.bytes == f.bytes {
                    unsafe { cuda.copy_device(destination, source, r.bytes, stream) }?;
                } else {
                    let head = f.bytes.checked_sub(r.bytes).filter(|&h| h > 0)
                        .context("retained region is not a strict tail")?;
                    ensure!(region.prefix && r.pointer == f.pointer + head as u64,
                        "only a monotone prefix may omit its earlier bytes");
                    unsafe { cuda.copy_device(destination, f.pointer, head, stream) }?;
                    unsafe { cuda.copy_device(destination + head as u64, source, r.bytes, stream) }?;
                }
            }
            ensure!(used.iter().all(|&u| u), "retained image has regions outside the complete image");
            Ok(())
        })();
        let completed = stream.synchronize();
        copies.and(completed)?;
        image.saved = true;
        Ok(Some(image))
    }
    pub(crate) fn configure_copy_kernel(&mut self,module:&Module)->Result<()> {
        ensure!(!self.saved,"copy path must be selected before capture");
        if copy::enabled() {
            let ranges=self.descriptors.iter().zip(&self.offsets).map(|(r,offset)|
                (r.pointer,self.storage().ptr()+*offset as u64,r.bytes)).collect::<Vec<_>>();
            if copy::eligible(&ranges) {self.copy_module=Some(module.clone());}
        }
        Ok(())
    }
    /// Consume an already-retired image, retaining exactly its physical storage
    /// and accounting charge. The caller must own it exclusively and finish
    /// every prior reader before retirement. This never reuses a live cache
    /// entry. Exact size only: no hidden slack, growth or persistent pool.
    pub(crate) fn try_recycle(mut self,regions:&[Region<'_>],budget:usize,stream:&Stream)
        ->Result<Option<Self>> {
        ensure!(self.cuda.owns_stream(stream)
            &&regions.iter().all(|r|r.cuda.owns_stream(stream)),"recycled image context/stream");
        // A parked image has no device storage to recycle.
        if self.device.is_none() {return Ok(None);}
        let descriptors=regions.iter().map(|r|r.descriptor.clone()).collect::<Vec<_>>();
        let(offsets,total)=layout(&descriptors,budget)?;
        if total!=self.storage().bytes() {return Ok(None);}
        ensure!(descriptors.iter().all(|r|r.pointer+r.bytes as u64<=self.storage().ptr()
            ||self.storage().ptr()+total as u64<=r.pointer),"recycled image aliases live storage");
        if self.cuda.admission_memory()?.admitted_available<crate::memory::floor_bytes().max(self.cuda.allocation_reserve_bytes()) {
            return Ok(None);
        }
        self.descriptors=descriptors;self.offsets=offsets;self.saved=false;self.copy_module=None;self.disk=None;
        Ok(Some(self))
    }
    fn copy_regions<'a>(&self,regions:impl Iterator<Item=&'a Descriptor>+Clone,save:bool,stream:&Stream)->Result<()> {
        if let Some(module)=&self.copy_module {
            let ranges=regions.clone().zip(&self.offsets).map(|(r,offset)| {
                let image=self.storage().ptr()+*offset as u64;
                if save {(r.pointer,image,r.bytes)}else{(image,r.pointer,r.bytes)}
            }).collect::<Vec<_>>();
            // layout/validate_import and the owning writer lease establish
            // disjoint ranges and lifetime. Always join below, even on error.
            if unsafe{copy::launch(&self.cuda,module,&ranges,stream)}? {return Ok(());}
        }
        for(r,offset)in regions.zip(&self.offsets) {
            let image=self.storage().ptr()+*offset as u64;
            let (source,destination)=if save{(r.pointer,image)}else{(image,r.pointer)};
            unsafe{self.cuda.copy_device(destination,source,r.bytes,stream)}?;
        }
        Ok(())
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
        ensure!(self.device.is_some(),"a parked state image must be reloaded before import");
        ensure!(self.saved&&destination.iter().all(|r|r.cuda.owns_stream(stream)),
            "import requires a captured immutable image in the original context");
        let descriptors=destination.iter().map(|r|r.descriptor.clone()).collect::<Vec<_>>();
        import_layout(&self.descriptors,&descriptors)?;
        ensure!(descriptors.iter().all(|r|r.pointer+r.bytes as u64<=self.storage().ptr()
            ||self.storage().ptr()+self.storage().bytes()as u64<=r.pointer),"import would overwrite its immutable image");
        Ok(())
    }
    unsafe fn import(&self,source:&[Region<'_>],destination:&[Region<'_>],stream:&Stream)->Result<()> {
        self.validate_import(source,destination,stream)?;
        let copies=self.copy_regions(destination.iter().map(|r|&r.descriptor),false,stream);
        let completed=stream.synchronize();copies.and(completed)
    }
    /// Restore into another owner's regions of the same layout: the same names and extents in the same
    /// order, at addresses that overlap neither the regions the image was saved from nor the image. It
    /// is for state whose bytes mean the same in any owner of one shape (the drafter's KV caches: rows
    /// addressed by position only), where the caller has established that both owners are of one family;
    /// `restore` keeps its exact-address contract and `import` its proof of the source regions.
    /// Safety: as `restore`, for the destination regions.
    pub(crate) unsafe fn restore_relocated(&self,destination:&[Region<'_>],stream:&Stream)->Result<()> {
        ensure!(self.saved&&self.device.is_some()&&self.cuda.owns_stream(stream)&&destination.iter().all(|r|r.cuda.owns_stream(stream)),
            "relocated restore needs a captured resident image in its original context");
        let descriptors=destination.iter().map(|r|r.descriptor.clone()).collect::<Vec<_>>();
        import_layout(&self.descriptors,&descriptors)?;
        ensure!(descriptors.iter().all(|r|r.pointer+r.bytes as u64<=self.storage().ptr()
            ||self.storage().ptr()+self.storage().bytes()as u64<=r.pointer),"relocated restore would overwrite its image");
        let copies=self.copy_regions(descriptors.iter(),false,stream);
        let completed=stream.synchronize();copies.and(completed)
    }
    /// The owning model must exclude writers on every participating stream.
    pub(crate) fn save(&mut self, regions: &[Region<'_>], stream: &Stream) -> Result<()> {
        self.validate(regions, stream)?;
        ensure!(!self.saved, "state images are immutable after capture");
        let copies=self.copy_regions(self.descriptors.iter(),true,stream);
        let completed = stream.synchronize();
        copies.and(completed)?;
        self.saved = true;
        Ok(())
    }
    /// As `save`, reading each region named in `sources` from that source's buffer (a
    /// state captured beside the live one, of the same extent) instead of the live region.
    /// The image keeps the live descriptors, so it restores exactly as a saved one does.
    pub(crate) fn save_with_sources(&mut self, regions: &[Region<'_>], sources: &[Region<'_>],
        stream: &Stream) -> Result<()> {
        self.validate(regions, stream)?;
        ensure!(!self.saved, "state images are immutable after capture");
        let mut from = self.descriptors.clone();
        for source in sources {
            ensure!(source.cuda.owns_stream(stream), "state image source stream");
            let mut live = from.iter_mut().filter(|d| d.name == source.descriptor.name);
            let (Some(d), None) = (live.next(), live.next()) else {
                anyhow::bail!("state image source {} names no single region", source.descriptor.name);
            };
            ensure!(d.bytes == source.descriptor.bytes, "state image source {} extent", d.name);
            d.pointer = source.descriptor.pointer;
        }
        let copies=self.copy_regions(from.iter(),true,stream);
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
        ensure!(self.device.is_some(), "a parked state image must be reloaded before restore");
        let copies=self.copy_regions(self.descriptors.iter(),false,stream);
        let completed = stream.synchronize();
        copies.and(completed)
    }
    /// Finite inline-checkpoint screen: a digest of each saved region's bytes, by name.
    pub(crate) fn region_digests_for_probe(&self) -> Result<Vec<(String, usize, String)>> {
        ensure!(std::env::var("CANADA_SCRATCH_INLINE_CHECKPOINT_SCREEN").as_deref()==Ok("1") && self.saved && self.device.is_some(),
            "region digests require the finite inline checkpoint screen");
        self.descriptors.iter().zip(&self.offsets).map(|(d, offset)| {
            let mut hash = Sha256::new();
            for at in (0..d.bytes).step_by(16 << 20) {
                let bytes = (d.bytes - at).min(16 << 20);
                hash.update(unsafe { self.cuda.read_device(self.storage().ptr() + (*offset + at) as u64, bytes) }?);
            }
            Ok((d.name.clone(), d.bytes, format!("{:x}", hash.finalize())))
        }).collect()
    }
    pub(crate) fn bytes(&self) -> usize {
        self.total
    }
    fn storage(&self) -> &ImageBuffer {
        self.device.as_ref().expect("a parked state image must be reloaded before device use")
    }
    /// Disk tier (crate::disk_tier): whether the device copy is present (false: parked on disk).
    pub(crate) fn resident(&self) -> bool { self.device.is_some() }
    pub(crate) fn has_disk_copy(&self) -> bool { self.disk.is_some() }
    /// Bytes the device copy holds now (0 while parked).
    pub(crate) fn device_bytes(&self) -> usize { if self.device.is_some() { self.total } else { 0 } }
    /// Bytes the disk copy takes (0 without one).
    pub(crate) fn disk_bytes(&self) -> usize { self.disk.as_ref().map_or(0, |d| d.disk_bytes()) }
    /// Begin a disk copy of a captured image, written a slice at a time (crate::disk_tier::PendingWrite); the device copy
    /// stays.
    pub(crate) fn disk_write_begin(&self, path: &std::path::Path) -> Result<crate::disk_tier::PendingWrite> {
        ensure!(self.saved && self.device.is_some() && self.disk.is_none(), "disk copy of a captured resident image");
        crate::disk_tier::PendingWrite::begin(self.total, path)
    }
    pub(crate) fn disk_write_step(&self, write: &mut crate::disk_tier::PendingWrite) -> Result<()> {
        ensure!(self.saved && self.device.is_some() && self.disk.is_none() && write.bytes() == self.total,
            "disk copy step of a captured resident image");
        write.step(self.storage())
    }
    pub(crate) fn disk_write_finish(&mut self, write: crate::disk_tier::PendingWrite) -> Result<()> {
        ensure!(self.device.is_some() && self.disk.is_none() && write.bytes() == self.total, "disk copy finish");
        self.disk = Some(write.finish()?);
        Ok(())
    }
    /// Free the device copy (its pool chunks return); the disk copy must exist.
    pub(crate) fn drop_device(&mut self) -> Result<()> {
        ensure!(self.saved && self.disk.is_some() && self.device.is_some(), "parking needs a disk copy");
        self.device = None;
        Ok(())
    }
    /// Reload a parked image from its disk copy (every chunk's digest checked). Ok(false): no memory for it now;
    /// it stays parked. The copy kernel's ranges are computed from the storage address on every copy.
    /// Returns the seconds spent allocating and the read's phases, or None when there is no memory for it now.
    pub(crate) fn reload(&mut self) -> Result<Option<(f64, crate::disk_tier::ReadTiming)>> {
        if self.device.is_some() { return Ok(Some((0.0, Default::default()))); }
        let disk = self.disk.as_ref().context("parked image without a disk copy")?;
        let t = std::time::Instant::now();
        let Some(mut storage) = allocate_storage(&self.cuda, self.total)? else { return Ok(None) };
        let alloc = t.elapsed().as_secs_f64();
        let timing = crate::disk_tier::read(disk, storage.buffer_mut())?;
        self.device = Some(storage);
        Ok(Some((alloc, timing)))
    }
    /// Drop the disk copy of a resident image (budget, or the image is about to change).
    pub(crate) fn drop_disk_copy(&mut self) -> Result<()> {
        ensure!(self.device.is_some(), "the disk copy is a parked image's only copy");
        self.disk = None;
        Ok(())
    }
    pub(crate) fn fused_copy(&self)->bool {self.copy_module.is_some()}
    pub(crate) fn manifest(&self) -> serde_json::Value {
        serde_json::json!({"bytes":self.bytes(),"fused_copy":self.copy_module.is_some(),"regions":self.descriptors.iter()
            .map(|r| serde_json::json!({"name":r.name,"bytes":r.bytes})).collect::<Vec<_>>()})
    }
    /// Bounded diagnostic reads only; no CPU execution or numeric conversion.
    pub(crate) fn digest(&self) -> Result<String> {
        ensure!(self.saved && self.device.is_some(), "uncaptured or parked state image digest");
        let mut hash = Sha256::new();
        for offset in (0..self.bytes()).step_by(16 * 1024 * 1024) {
            hash.update(
                self.storage()
                    .download_part(offset, (self.bytes() - offset).min(16 * 1024 * 1024))?,
            );
        }
        Ok(format!("{:x}", hash.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn displaced_checkpoint_respects_the_host_floor_and_whole_buffer_pool_fit() {
        let bytes = 471_345_156;
        let reserve = 4usize << 30;
        assert!(image_storage_fits(reserve, reserve, bytes, true));
        assert!(!image_storage_fits(reserve - 1, reserve, bytes, true));
        assert!(!image_storage_fits(reserve + bytes - 1, reserve, bytes, false));
        assert!(image_storage_fits(reserve + bytes, reserve, bytes, false));
    }

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
