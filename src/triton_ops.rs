//! Direct driver launch of hashed AOT GPU primitives. No resident Python/Triton.
use crate::cuda::{Cuda, Module, Stream};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{ffi::c_void, path::Path};

#[derive(Debug, Deserialize)]
pub struct Argument {
    pub name: String,
    pub kind: String,
}
fn one() -> u32 {
    1
}
#[derive(Debug, Deserialize)]
pub struct Grid {
    pub rows_div: u32,
    #[serde(default)]
    pub x_add: u32,
    #[serde(default = "one")]
    pub x_mul: u32,
    pub y: u32,
    pub z: u32,
}
#[derive(Debug, Deserialize)]
pub struct Metadata {
    pub num_warps: u32,
    pub num_ctas: u32,
    pub shared: u32,
    #[serde(default)]
    pub global_scratch_size: u64,
    #[serde(default)]
    pub profile_scratch_size: u64,
}
#[derive(Debug, Deserialize)]
pub struct Entry {
    pub name: String,
    pub symbol: String,
    pub architecture: u32,
    pub physical_sms: u32,
    pub arguments: Vec<Argument>,
    pub metadata: Metadata,
    pub grid: Grid,
    pub ptx_parameters: Vec<(String, String)>,
    pub artifact_sha256: String,
    #[serde(default)]
    pub programmatic_stream_serialization: bool,
}
#[derive(Deserialize)]
struct Manifest {
    entries: Vec<Entry>,
}
pub struct Operator {
    pub entry: Entry,
    module: Module,
}
impl Operator {
    pub fn load(cuda: &Cuda, root: &Path, name: &str) -> Result<Self> {
        cuda.qualify_gb10()?;
        let manifest: Manifest =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let entry = manifest
            .entries
            .into_iter()
            .find(|e| e.name == name)
            .context("missing AOT operator")?;
        ensure!(
            entry
                .name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_'),
            "invalid AOT basename"
        );
        ensure!(
            entry.architecture == 121 && entry.physical_sms == 48 && entry.metadata.num_ctas == 1,
            "AOT device contract mismatch"
        );
        ensure!(
            entry.metadata.global_scratch_size == 0 && entry.metadata.profile_scratch_size == 0,
            "unbudgeted hidden Triton scratch"
        );
        ensure!(
            entry.ptx_parameters.len() == entry.arguments.len() + 2
                && entry
                    .ptx_parameters
                    .iter()
                    .rev()
                    .take(2)
                    .all(|p| p.0 == "u64"),
            "unsupported hidden Triton ABI"
        );
        ensure!(
            entry.grid.rows_div > 0
                && entry.grid.y > 0
                && entry.grid.z > 0
                && (entry.grid.x_mul > 0 || entry.grid.x_add > 0)
                // The finite FP64 WO-A diagnostic uses 256 output tiles per
                // row. At the launch row cap this is 2^24 blocks, safely
                // within both u32 arithmetic and CUDA's x-grid limit.
                && entry.grid.x_mul <= 256
                && entry.grid.x_add <= 256,
            "invalid launch grid for {}: {:?}", entry.name, entry.grid
        );
        let artifact = std::fs::read(root.join(format!("{}.cubin", entry.name)))?;
        ensure!(
            format!("{:x}", Sha256::digest(&artifact)) == entry.artifact_sha256,
            "AOT operator hash mismatch"
        );
        let module = cuda.module(&artifact)?;
        let function = module.function(&entry.symbol)?;
        if entry.metadata.shared > 48 * 1024 {
            function.opt_in_shared_memory(entry.metadata.shared)?;
        }
        Ok(Self { entry, module })
    }
    /// Safety: word values match each recorded pointer/scalar argument. Buffer
    /// sizes, dtypes, alignment and lifetimes obey the source operator contract.
    /// Module/buffers remain alive until every queued/captured launch completes.
    pub unsafe fn launch(&self, rows: u32, values: &[u64], stream: &Stream) -> Result<()> {
        ensure!(
            rows > 0 && rows <= 65536 && values.len() == self.entry.arguments.len(),
            "AOT row/argument contract"
        );
        for (a, v) in self.entry.arguments.iter().zip(values) {
            if ["n", "m", "M"].contains(&a.name.as_str()) {
                ensure!(*v == rows as u64, "row scalar disagrees with launch grid");
            }
            if a.kind.starts_with('*') {
                ensure!(*v != 0 && *v % 2 == 0, "invalid AOT pointer");
            }
        }
        let mut words = values.to_vec();
        words.extend([0, 0]);
        let mut pointers: Vec<*mut c_void> =
            words.iter_mut().map(|w| (w as *mut u64).cast()).collect();
        let grid = [
            (rows * self.entry.grid.x_mul).div_ceil(self.entry.grid.rows_div)
                + self.entry.grid.x_add,
            self.entry.grid.y,
            self.entry.grid.z,
        ];
        let function = self.module.function(&self.entry.symbol)?;
        if self.entry.programmatic_stream_serialization {
            return unsafe { function.launch_pdl(grid, [32 * self.entry.metadata.num_warps, 1, 1],
                self.entry.metadata.shared, stream, &mut pointers) };
        }
        unsafe {
            function.launch(
                grid,
                [32 * self.entry.metadata.num_warps, 1, 1],
                self.entry.metadata.shared,
                stream,
                &mut pointers,
            )
        }
    }
}
