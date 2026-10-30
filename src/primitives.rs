//! Checked AOT operator ABI. This library contains GPU primitives, never a model
//! engine. The independent Rust graph supplies all weights, scratch and state.
use crate::cuda::{Cuda, Stream};
use anyhow::{Context, Result, ensure};
use libloading::Library;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, collections::BTreeMap, ffi::c_void, path::{Path,PathBuf}, rc::{Rc,Weak}};

#[derive(Clone, Debug, Deserialize)]
pub struct Entry {
    pub index: i32,
    pub name: String,
    pub kernel_id: String,
    #[serde(default)]
    pub pointers: Vec<String>,
    #[serde(default)]
    pub i32_scalars: Vec<String>,
    #[serde(default)]
    pub arguments: Vec<Argument>,
}
#[derive(Clone, Debug, Deserialize)]
pub struct Argument {
    pub name: String,
    pub kind: String,
}
#[derive(Clone, Debug, Deserialize)]
pub struct Bindings {
    pub abi: i32,
    pub architecture: i32,
    pub physical_sms: i32,
    pub source: String,
    pub source_commit: String,
    pub library_sha256: String,
    pub entries: Vec<Entry>,
}
pub struct Primitives {
    owner: Rc<Owner>,
    pub bindings: Bindings,
}
// CuTe's generated wrapper caches kernel handles inside each loaded library.
// Independently opening/closing the same bundle invalidated a still-live
// sibling's eager launches (s342). Share one native handle for that bundle and
// CUDA context; graphs/workspaces continue to belong to their own operators.
struct Owner {
    _lib: Library,
    _context: Cuda,
    handle: *mut c_void,
    close: unsafe extern "C" fn(*mut c_void),
    launch: unsafe extern "C" fn(
        *mut c_void,
        i32,
        *const u64,
        usize,
        *const i32,
        usize,
        *mut c_void,
    ) -> i32,
    launch2: Option<unsafe extern "C" fn(*mut c_void, i32, *const u64, usize, *mut c_void) -> i32>,
}
type CacheKey = (usize,PathBuf,String);
thread_local! {
    static OWNERS: RefCell<BTreeMap<CacheKey,Weak<Owner>>> = RefCell::new(BTreeMap::new());
}
impl Primitives {
    pub fn open(context: &Cuda, library: &Path, bindings: &Path) -> Result<Self> {
        context.qualify_gb10()?;
        let manifest: Bindings = serde_json::from_slice(&std::fs::read(bindings)?)?;
        ensure!(
            (manifest.abi == 1 || manifest.abi == 2)
                && manifest.architecture == 121
                && manifest.physical_sms == 48,
            "unsupported primitive ABI/device"
        );
        let bytes = std::fs::read(library)?;
        ensure!(
            format!("{:x}", Sha256::digest(bytes)) == manifest.library_sha256,
            "primitive library hash mismatch"
        );
        for (i, e) in manifest.entries.iter().enumerate() {
            ensure!(e.index == i as i32, "non-contiguous primitive index");
        }
        let key=(context.context_key(),library.canonicalize()?,manifest.library_sha256.clone());
        let cached=OWNERS.with(|cache| {
            let mut cache=cache.borrow_mut();
            cache.retain(|_,owner|owner.strong_count()>0);
            cache.get(&key).and_then(Weak::upgrade)
        });
        if let Some(owner)=cached {return Ok(Self{owner,bindings:manifest});}
        let lib = unsafe { Library::new(library) }.context("load primitive library")?;
        let abi: unsafe extern "C" fn() -> i32 = *unsafe { lib.get(b"ca_primitives_abi\0") }?;
        ensure!(
            unsafe { abi() } == manifest.abi,
            "primitive entrypoint ABI mismatch"
        );
        let open: unsafe extern "C" fn(*mut i32) -> *mut c_void =
            *unsafe { lib.get(b"ca_primitives_open\0") }?;
        let close = *unsafe { lib.get(b"ca_primitives_close\0") }?;
        let launch = *unsafe { lib.get(b"ca_primitives_launch\0") }?;
        let launch2 = unsafe { lib.get(b"ca_primitives_launch2\0") }
            .ok()
            .map(|f| *f);
        let mut status = 0;
        let handle = unsafe { open(&mut status) };
        ensure!(
            status == 0 && !handle.is_null(),
            "initialize primitive library: {status}"
        );
        let owner=Rc::new(Owner {
            _lib: lib,
            _context: context.clone(),
            handle,
            close,
            launch,
            launch2,
        });
        OWNERS.with(|cache|cache.borrow_mut().insert(key,Rc::downgrade(&owner)));
        Ok(Self{owner,bindings:manifest})
    }
    /// Safety: pointers must conform to each named entry's shape/dtype/alignment
    /// contract and remain valid through stream completion. Scalar capacities
    /// cannot exceed the compiled artifact. Every captured library stays alive.
    pub unsafe fn launch(
        &self,
        index: i32,
        pointers: &[u64],
        scalars: &[i32],
        stream: &Stream,
    ) -> Result<()> {
        let e = self
            .bindings
            .entries
            .get(index as usize)
            .context("unknown primitive index")?;
        ensure!(
            e.pointers.len() == pointers.len() && e.i32_scalars.len() == scalars.len(),
            "primitive parameter count mismatch"
        );
        let status = unsafe {
            (self.owner.launch)(
                self.owner.handle,
                index,
                pointers.as_ptr(),
                pointers.len(),
                scalars.as_ptr(),
                scalars.len(),
                stream.raw(),
            )
        };
        ensure!(status == 0, "primitive {} launch failed: {status}", e.name);
        Ok(())
    }
    /// Safety: each word is an exact pointer or scalar bit representation for
    /// the entry's argument descriptor. All pointer lifetimes follow launch().
    pub unsafe fn launch_words(&self, index: i32, values: &[u64], stream: &Stream) -> Result<()> {
        ensure!(self.bindings.abi == 2, "typed-word launch needs ABI 2");
        let entry = self
            .bindings
            .entries
            .get(index as usize)
            .context("unknown primitive index")?;
        ensure!(
            entry.arguments.len() == values.len(),
            "typed primitive argument count mismatch"
        );
        let launch = self.owner.launch2.context("missing typed-word entrypoint")?;
        let status = unsafe {
            launch(
                self.owner.handle,
                index,
                values.as_ptr(),
                values.len(),
                stream.raw(),
            )
        };
        ensure!(
            status == 0,
            "primitive {} launch failed: {status}",
            entry.name
        );
        Ok(())
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        unsafe {
            (self.close)(self.handle);
        }
    }
}

/// Standalone GPU lifetime check using the actual Rust ownership wrapper.
/// Capture, sibling teardown, eager launch, last-owner close and reopen are
/// exercised without loading model weights or introducing CPU numeric math.
pub fn probe_lifetime(root:&Path)->Result<serde_json::Value>{
    let cuda=Cuda::open(0)?;cuda.qualify_gb10()?;
    let stream=cuda.stream()?;
    let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
    let m=manifest["geometry"]["rows"].as_u64().context("missing rows")? as usize;
    let k=manifest["geometry"]["in_features"].as_u64().context("missing K")? as usize;
    ensure!(m==6 && [1280,5120].contains(&k),"lifetime probe geometry");
    let mut input=cuda.allocate::<half::bf16>(m*k)?;
    // Public constant fixture, not checkpoint packing or CPU model math.
    input.upload(&vec![half::bf16::from_bits(0x3f80);m*k])?;
    let mut values=cuda.allocate::<u8>(m*k)?;
    let mut scale_rows=cuda.allocate::<u8>(m*k/32)?;
    let mut scales=cuda.allocate::<u8>(128*k/32)?;
    values.zero()?;scale_rows.zero()?;scales.zero()?;
    stream.join_default()?;
    let library=root.join("libcanada_primitives.so");let bindings=root.join("bindings.json");
    let mut records=Vec::new();
    for cycle in 0..3 {
        let mut siblings=(0..40).map(|_|Primitives::open(&cuda,&library,&bindings)).collect::<Result<Vec<_>>>()?;
        ensure!(siblings.iter().all(|p|Rc::ptr_eq(&p.owner,&siblings[0].owner)),"native owners are not shared");
        let entry=siblings[0].bindings.entries.iter().find(|e|["gemm.mxfp8_quant_cute","flashinfer.mxfp8.quant.pdl1"].contains(&e.kernel_id.as_str())).context("missing probe quantizer")?;
        let index=entry.index;
        let metadata=BTreeMap::from([
            ("x",input.ptr()),("y",values.ptr()),("scales",scales.ptr()),
            ("source_ptr",input.ptr()),("values_ptr",values.ptr()),
            ("scale_rows_ptr",scale_rows.ptr()),("scale_mma_ptr",scales.ptr()),
            ("m",m as u64),("source_k",k as u64),("grid_x",(m*(k/32).div_ceil(4)).div_ceil(4).min(192) as u64),
        ]);
        let words=entry.arguments.iter().map(|v|metadata.get(v.name.as_str()).copied().context("unknown probe argument")).collect::<Result<Vec<_>>>()?;
        unsafe{siblings[0].launch_words(index,&words,&stream)?;}
        stream.synchronize()?;
        let expected=(values.download()?,scales.download()?);
        ensure!(expected.0.iter().any(|&byte|byte!=0),"lifetime witness is empty");
        let graph=stream.capture(||unsafe{siblings[0].launch_words(index,&words,&stream)})?;
        unsafe{graph.launch(&stream)?;}stream.synchronize()?;
        siblings.truncate(1);
        // The quantizer only writes live scale entries; preserve zero padding.
        values.zero()?;scales.zero()?;
        stream.join_default()?;
        unsafe{graph.launch(&stream)?;}stream.synchronize()?;
        ensure!((values.download()?,scales.download()?)==expected,"surviving graph output changed");
        values.zero()?;scales.zero()?;
        stream.join_default()?;
        unsafe{siblings[0].launch_words(index,&words,&stream)?;}stream.synchronize()?;
        ensure!((values.download()?,scales.download()?)==expected,"surviving eager output changed");
        records.push(serde_json::json!({"cycle":cycle,"shared_native_owner":true,"surviving_graph_exact":true,"surviving_eager_exact":true}));
        drop(graph);drop(siblings);
    }
    Ok(serde_json::json!({"all_correct":true,"cycles":records,"scope":"Rust primitive owner lifetime across40siblings, graph/eager replay after39drops, complete close/reopen three times. GPU quantizer only."}))
}
