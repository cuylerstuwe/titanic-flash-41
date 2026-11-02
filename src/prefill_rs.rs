//! Optional Rust-owned prefill quarter exchange. No target weights or state live
//! in the native dependency; Rust owns its capacity, ordering and tensor views.
use crate::cuda::{Buffer, Cuda, DeviceElement, Stream};
use anyhow::{Result, ensure};
use libloading::Library;
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    ffi::{CString, c_char, c_void},
    path::Path,
};
type Handle = *mut c_void;
pub(crate) struct ReduceScatter {
    cuda: Cuda,
    _library: Library,
    handle: Handle,
    blob_bytes: usize,
    blob: unsafe extern "C" fn(Handle, Handle) -> i32,
    connect: unsafe extern "C" fn(Handle, *const c_void) -> i32,
    enqueue: unsafe extern "C" fn(Handle, *const c_void, Handle, usize, Handle) -> i32,
    health: unsafe extern "C" fn(Handle) -> i32,
    destroy: unsafe extern "C" fn(Handle),
    connected: Cell<bool>,
    poisoned: Cell<bool>,
    stream: Cell<Option<usize>>,
    calls: Cell<usize>,
}
impl ReduceScatter {
    pub(crate) fn new(cuda: &Cuda, root: &Path, rank: usize) -> Result<Self> {
        ensure!(rank < 4, "prefill RS rank");
        // Ten 20MiB slots plus controls and CUDA graph metadata.
        let reserve = 8 * 1024usize.pow(3) + 208 * 1024usize.pow(2);
        ensure!(
            cuda.admission_memory()?.admitted_available >= reserve,
            "prefill RS must preserve the 8GiB UMA reserve"
        );
        for (name, expected) in [
            (
                "bridge.so",
                "d8870e5197e363c0aeb2fe61c1c38402d2b261bbd2d77f2c829cdb58c5875a92",
            ),
            (
                "proxy.so",
                "18a43539e8c964c27078da2a452a91a9c77eca2f46cfd2f6d1c3749a14f323b3",
            ),
        ] {
            ensure!(
                format!("{:x}", Sha256::digest(std::fs::read(root.join(name))?)) == expected,
                "prefill RS dependency hash: {name}"
            );
        }
        let library = unsafe { Library::new(root.join("bridge.so")) }?;
        macro_rules! sym {
            ($name:literal) => {
                *unsafe { library.get(concat!($name, "\0").as_bytes()) }?
            };
        }
        let create: unsafe extern "C" fn(i32, usize, *const c_char) -> Handle = sym!("pg_create");
        let bytes: unsafe extern "C" fn(Handle) -> usize = sym!("pg_blob_bytes");
        let destroy: unsafe extern "C" fn(Handle) = sym!("pg_destroy");
        let path = CString::new(
            root.join("proxy.so")
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("prefill RS path encoding"))?,
        )?;
        let handle = unsafe { create(rank as i32, 2048 * 5120 * 2, path.as_ptr()) };
        ensure!(!handle.is_null(), "create prefill RS transport");
        let blob_bytes = unsafe { bytes(handle) };
        if !(1..=65536).contains(&blob_bytes) {
            unsafe {
                destroy(handle);
            }
            anyhow::bail!("prefill RS endpoint size");
        }
        Ok(Self {
            cuda: cuda.clone(),
            handle,
            blob_bytes,
            destroy,
            blob: sym!("pg_blob"),
            connect: sym!("pg_connect"),
            enqueue: sym!("rs_enqueue"),
            health: sym!("rs_health"),
            _library: library,
            connected: Cell::new(false),
            poisoned: Cell::new(false),
            stream: Cell::new(None),
            calls: Cell::new(0),
        })
    }
    fn checked(&self, status: i32) -> Result<()> {
        if status != 0 {
            self.poisoned.set(true);
        }
        ensure!(!self.poisoned.get(), "prefill RS failed/poisoned: {status}");
        Ok(())
    }
    pub(crate) fn blob(&self) -> Result<Vec<u8>> {
        let mut bytes = vec![0; self.blob_bytes];
        self.checked(unsafe { (self.blob)(self.handle, bytes.as_mut_ptr().cast()) })?;
        Ok(bytes)
    }
    pub(crate) fn connect(&self, all: &[u8]) -> Result<()> {
        ensure!(
            !self.connected.get() && all.len() == 4 * self.blob_bytes,
            "prefill RS endpoint exchange"
        );
        self.checked(unsafe { (self.connect)(self.handle, all.as_ptr().cast()) })?;
        self.connected.set(true);
        Ok(())
    }
    pub(crate) fn health(&self) -> Result<()> {
        self.checked(unsafe { (self.health)(self.handle) })
    }
    pub(crate) fn handles(count: usize) -> bool {
        // The equal-graph component screen benefited these sizes. 2048 rows
        // were flat to slightly slower; retain the existing path there.
        [512, 1024].into_iter().any(|rows| count == rows * 5120)
    }
    pub(crate) fn calls(&self) -> usize {
        self.calls.get()
    }
    pub(crate) unsafe fn run<T: DeviceElement>(
        &self,
        input: &Buffer<T>,
        output: &mut Buffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        self.health()?;
        ensure!(
            self.connected.get()
                && Self::handles(input.len())
                && input.len() == output.len() * 4
                && std::mem::size_of::<T>() == 2
                && self.cuda.owns_stream(stream)
                && self.cuda.owns_buffer(input)
                && self.cuda.owns_buffer(output)
                && self.stream.get().is_none_or(|s| s == stream.raw() as usize),
            "prefill RS tensor/stream contract"
        );
        self.stream.set(Some(stream.raw() as usize));
        self.checked(unsafe {
            (self.enqueue)(
                self.handle,
                input.ptr() as *const c_void,
                output.ptr() as Handle,
                input.len(),
                stream.raw(),
            )
        })?;
        self.calls.set(self.calls.get() + 1);
        Ok(())
    }
}
impl Drop for ReduceScatter {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.handle);
        }
    }
}
