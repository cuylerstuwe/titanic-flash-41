//! GPU-only normalization/RoPE. Rust owns buffers, geometry and execution;
//! the pinned low-level dependencies expose no model or Python runtime.
use crate::{
    blockscaled::scale_bytes,
    cuda::{Buffer, Cuda, Stream},
    primitives::Primitives,
    triton_ops::Operator,
};
use anyhow::{Result, ensure};
use half::bf16;
use libloading::Library;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::Cell, ffi::c_void, path::Path};
#[derive(Deserialize)]
struct Manifest {
    abi: u32,
    architecture: u32,
    physical_sms: u32,
    rope_heads: usize,
    rope_head_width: usize,
    rope_width: usize,
    norm_widths: Vec<usize>,
    epsilon: f32,
    library_sha256: String,
}
type Rope =
    unsafe extern "C" fn(i32, i32, i32, *mut c_void, *const f32, *const i32, *mut c_void) -> i32;
pub struct Native {
    _library: Library,
    cuda: Cuda,
    rms: Primitives,
    rope: Rope,
}
impl Native {
    pub fn open(cuda: &Cuda, root: &Path) -> Result<Self> {
        Self::open_with_rms(cuda, root, &root.join("rms"))
    }
    pub fn open_with_rms(cuda: &Cuda, root: &Path, rms_root: &Path) -> Result<Self> {
        cuda.qualify_gb10()?;
        let m: Manifest = serde_json::from_slice(&std::fs::read(root.join("native.json"))?)?;
        ensure!(
            (
                m.abi,
                m.architecture,
                m.physical_sms,
                m.rope_heads,
                m.rope_head_width,
                m.rope_width
            ) == (1, 121, 48, 16, 512, 64)
                && m.norm_widths == [512, 1280, 5120]
                && m.epsilon.to_bits() == 1e-20f32.to_bits(),
            "prelude contract changed"
        );
        let p = root.join("libcanada_prelude.so");
        ensure!(
            format!("{:x}", Sha256::digest(std::fs::read(&p)?)) == m.library_sha256,
            "prelude library hash mismatch"
        );
        let lib = unsafe { Library::new(p) }?;
        let abi: unsafe extern "C" fn() -> i32 = *unsafe { lib.get(b"ca_prelude_abi\0") }?;
        ensure!(unsafe { abi() } == 1, "prelude ABI mismatch");
        let rms = Primitives::open(
            cuda,
            &rms_root.join("libcanada_primitives.so"),
            &rms_root.join("bindings.json"),
        )?;
        let rope = *unsafe { lib.get(b"ca_rope\0") }?;
        Ok(Self {
            _library: lib,
            cuda: cuda.clone(),
            rms,
            rope,
        })
    }
    pub fn rms(
        &self,
        input: &Buffer<bf16>,
        weight: &Buffer<bf16>,
        output: &mut Buffer<bf16>,
        rows: usize,
        stride: usize,
        stream: &Stream,
    ) -> Result<()> {
        let width = weight.len();
        ensure!(
            rows > 0
                && rows <= 65536
                && [512, 1280, 5120].contains(&width)
                && stride >= width
                && stride <= i32::MAX as usize,
            "RMS dimensions"
        );
        ensure!(
            input.len() >= (rows - 1) * stride + width && output.len() == rows * width,
            "RMS buffer spans"
        );
        ensure!(
            self.cuda.owns_buffer(input)
                && self.cuda.owns_buffer(weight)
                && self.cuda.owns_buffer(output)
                && self.cuda.owns_stream(stream),
            "RMS context mismatch"
        );
        let id = format!("norm.rms.h{width}.s{stride}");
        let matches: Vec<_> = self
            .rms
            .bindings
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.kernel_id == id)
            .collect();
        ensure!(matches.len() == 1, "RMS geometry is not AOT qualified");
        let (index, entry) = matches[0];
        ensure!(
            entry.arguments.len() == 5
                && entry
                    .arguments
                    .iter()
                    .map(|a| a.kind.as_str())
                    .collect::<Vec<_>>()
                    == ["pointer", "pointer", "pointer", "i64", "f32"],
            "RMS AOT ABI changed"
        );
        unsafe {
            self.rms.launch_words(
                index as i32,
                &[
                    input.ptr(),
                    weight.ptr(),
                    output.ptr(),
                    rows as u64,
                    1e-20f32.to_bits() as u64,
                ],
                stream,
            )
        }
    }
    pub fn rope(
        &self,
        input: &mut Buffer<bf16>,
        rows: usize,
        frequencies: &Buffer<f32>,
        positions: &Positions,
        inverse: bool,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            rows > 0
                && rows <= 65536
                && positions.rows == rows
                && input.len() == rows * 8192
                && frequencies.len() == positions.table_rows * 64,
            "RoPE dimensions"
        );
        ensure!(
            self.cuda.owns_buffer(input)
                && self.cuda.owns_buffer(frequencies)
                && self.cuda.owns_buffer(&positions.values)
                && self.cuda.owns_stream(stream),
            "RoPE context mismatch"
        );
        let status = unsafe {
            (self.rope)(
                rows as i32,
                16,
                i32::from(inverse),
                input.ptr() as _,
                frequencies.ptr() as _,
                positions.values.ptr() as _,
                stream.raw(),
            )
        };
        ensure!(status == 0, "RoPE GPU launch failed: {status}");
        Ok(())
    }
}
pub struct Positions {
    values: Buffer<i32>,
    rows: usize,
    table_rows: usize,
}
impl Positions {
    pub(crate) fn values(&self) -> &Buffer<i32> {
        &self.values
    }
    pub fn update(&mut self, values: &[i32]) -> Result<()> {
        ensure!(
            values.len() == self.rows
                && values
                    .iter()
                    .all(|&x| x >= 0 && (x as usize) < self.table_rows),
            "RoPE position update outside table/shape"
        );
        self.values.upload(values)
    }
    pub fn upload(cuda: &Cuda, values: &[i32], table_rows: usize) -> Result<Self> {
        ensure!(
            !values.is_empty()
                && values.len() <= 65536
                && table_rows <= i32::MAX as usize
                && values.iter().all(|&x| x >= 0 && (x as usize) < table_rows),
            "RoPE position outside table"
        );
        let mut v = cuda.allocate::<i32>(values.len())?;
        v.upload(values)?;
        Ok(Self {
            values: v,
            rows: values.len(),
            table_rows,
        })
    }
}
pub struct NormMxfp8 {
    cuda: Cuda,
    operator: Operator,
    rows: usize,
    stride: usize,
    stream: Cell<usize>,
    output: Buffer<bf16>,
    values: Buffer<u8>,
    scales: Buffer<u8>,
}
impl NormMxfp8 {
    pub fn new(cuda: &Cuda, root: &Path, rows: usize) -> Result<Self> {
        Self::with_stride(cuda, root, rows, 1280)
    }
    pub fn with_stride(cuda: &Cuda, root: &Path, rows: usize, stride: usize) -> Result<Self> {
        ensure!(
            (1..=8).contains(&rows) && [1280, 1792].contains(&stride),
            "fused Q norm geometry"
        );
        let suffix = if stride == 1280 { "" } else { "_s1792" };
        let op = Operator::load(cuda, root, &format!("q_norm_mxfp8_m{rows}{suffix}"))?;
        ensure!(
            op.entry
                .arguments
                .iter()
                .map(|x| (x.name.as_str(), x.kind.as_str()))
                .collect::<Vec<_>>()
                == [
                    ("X", "*bf16"),
                    ("W", "*bf16"),
                    ("Y", "*bf16"),
                    ("Q", "*fp8e4nv"),
                    ("S", "*u8")
                ],
            "fused Q norm ABI changed"
        );
        ensure!(
            (
                op.entry.grid.x_add,
                op.entry.grid.x_mul,
                op.entry.grid.rows_div,
                op.entry.grid.y,
                op.entry.grid.z
            ) == (10, 1, 1, 1, 1),
            "fused Q norm grid changed"
        );
        let mut output = cuda.allocate::<bf16>(rows * 1280)?;
        output.zero()?;
        let mut values = cuda.allocate::<u8>(rows * 1280)?;
        values.zero()?;
        let mut scales = cuda.allocate::<u8>(scale_bytes(rows, 1280)?)?;
        scales.zero()?;
        Ok(Self {
            cuda: cuda.clone(),
            operator: op,
            rows,
            stride,
            stream: Cell::new(0),
            output,
            values,
            scales,
        })
    }
    pub fn output(&self) -> &Buffer<bf16> {
        &self.output
    }
    pub fn values(&self) -> &Buffer<u8> {
        &self.values
    }
    pub fn scales(&self) -> &Buffer<u8> {
        &self.scales
    }
    pub fn run(&self, input: &Buffer<bf16>, weight: &Buffer<bf16>, stream: &Stream) -> Result<()> {
        ensure!(
            input.len() >= (self.rows - 1) * self.stride + 1280
                && weight.len() == 1280
                && self.cuda.owns_buffer(input)
                && self.cuda.owns_buffer(weight)
                && self.cuda.owns_stream(stream),
            "fused Q norm shape/context"
        );
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == raw,
            "fused Q norm workspace needs one ordered stream"
        );
        self.stream.set(raw);
        unsafe {
            self.operator.launch(
                self.rows as u32,
                &[
                    input.ptr(),
                    weight.ptr(),
                    self.output.ptr(),
                    self.values.ptr(),
                    self.scales.ptr(),
                ],
                stream,
            )
        }
    }
}

/// The two canonical rotary tables are constructed once on GPU and shared by
/// layers. Runtime tables do not depend on a Python-generated numerical asset.
pub struct FrequencyTable {
    compressed: bool,
    rows: usize,
    base: Buffer<f32>,
    values: Buffer<f32>,
}
impl FrequencyTable {
    pub fn new(
        cuda: &Cuda,
        module: &crate::cuda::Module,
        stream: &Stream,
        rows: usize,
        compressed: bool,
    ) -> Result<Self> {
        ensure!(
            rows > 0 && rows <= 1048576 && cuda.owns_stream(stream),
            "bounded canonical frequency table required"
        );
        let base_values = cuda.allocate::<f32>(32)?;
        let values = cuda.allocate::<f32>(rows * 64)?;
        let mut out = base_values.ptr();
        let mut theta = if compressed { 160000f32 } else { 10000f32 };
        let mut original = if compressed { 65536i32 } else { 0i32 };
        let mut factor = 16f32;
        let mut fast = 32f32;
        let mut slow = 1f32;
        let mut args: Vec<*mut c_void> = vec![
            (&mut out as *mut u64).cast(),
            (&mut theta as *mut f32).cast(),
            (&mut original as *mut i32).cast(),
            (&mut factor as *mut f32).cast(),
            (&mut fast as *mut f32).cast(),
            (&mut slow as *mut f32).cast(),
        ];
        unsafe {
            module.function("ca_rope_frequency_base")?.launch(
                [1, 1, 1],
                [32, 1, 1],
                0,
                stream,
                &mut args,
            )
        }?;
        let mut source = base_values.ptr();
        let mut destination = values.ptr();
        let mut n = rows as i32;
        let mut args: Vec<*mut c_void> = vec![
            (&mut source as *mut u64).cast(),
            (&mut destination as *mut u64).cast(),
            (&mut n as *mut i32).cast(),
        ];
        unsafe {
            module.function("ca_rope_frequency_table")?.launch(
                [(rows * 32).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )
        }?;
        stream.synchronize()?;
        Ok(Self {
            compressed,
            rows,
            base: base_values,
            values,
        })
    }
    pub fn values(&self) -> &Buffer<f32> {
        &self.values
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn compressed(&self) -> bool {
        self.compressed
    }
    pub fn base(&self) -> &Buffer<f32> {
        &self.base
    }
}

pub struct QStore {
    cuda: Cuda,
    rows: usize,
    operator: Operator,
    output: Buffer<bf16>,
    stream: Cell<usize>,
}
impl QStore {
    pub fn new(cuda: &Cuda, root: &Path, rows: usize) -> Result<Self> {
        ensure!(
            rows > 0 && (rows <= 8 || (4096..=65536).contains(&rows)),
            "Q store shape not qualified"
        );
        let name = if rows <= 8 {
            "q_rope_small"
        } else {
            "q_rope_prefill"
        };
        let operator = Operator::load(cuda, root, name)?;
        let mut output = cuda.allocate::<bf16>(rows * 8192)?;
        output.zero()?;
        Ok(Self {
            cuda: cuda.clone(),
            rows,
            operator,
            output,
            stream: Cell::new(0),
        })
    }
    pub fn output(&self) -> &Buffer<bf16> {
        &self.output
    }
    pub fn run(
        &self,
        input: &Buffer<bf16>,
        frequencies: &Buffer<f32>,
        positions: &Positions,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            input.len() == self.rows * 8192
                && positions.rows == self.rows
                && frequencies.len() == positions.table_rows * 64,
            "Q store dimensions"
        );
        ensure!(
            self.cuda.owns_buffer(input)
                && self.cuda.owns_buffer(frequencies)
                && self.cuda.owns_buffer(&positions.values)
                && self.cuda.owns_stream(stream),
            "Q store context"
        );
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == raw,
            "Q store workspace needs one ordered stream"
        );
        self.stream.set(raw);
        let mut args = vec![
            input.ptr(),
            self.output.ptr(),
            frequencies.ptr(),
            positions.values.ptr(),
        ];
        if self.rows >= 4096 {
            args.push(self.rows as u64);
        }
        unsafe { self.operator.launch(self.rows as u32, &args, stream) }
    }
}
