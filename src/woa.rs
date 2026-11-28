//! Lossless canonical W8 grouped WO-A and fused BF16-to-MXFP8 epilogue.
//! Raw FP8/E8M0 checkpoint bytes reconstruct the exact reference BF16 MMA
//! operands in registers. Rust owns weight lifetime, scratch and stream order.
use crate::{
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, Stream},
    triton_ops::Operator,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{cell::Cell, path::Path, rc::Rc};
pub struct Grouped {
    cuda: Cuda,
    _weights: Rc<Weights>,
    weight: u64,
    scales: u64,
    rows: usize,
    partial_op: Operator,
    quant_op: Operator,
    partial: Buffer<f32>,
    values: Buffer<u8>,
    output_scales: Buffer<u8>,
    stream: Cell<usize>,
    /// ca_woa_partial_w8 in place of the exported partial kernel (config::woa_native), and whether it is on.
    native: Option<crate::cuda::Module>,
    native_on: Cell<bool>,
    /// Single-kernel fused WO-A + split-K + MXFP8 quant (ca_woa_fused_quant_m6; config::woa_fused_direct).
    fused_direct: Option<crate::cuda::Module>,
    fused_direct_on: Cell<bool>,
}
impl Grouped {
    /// The immutable canonical WO-A bytes (weights, scales) and their owner, for L2 hints.
    pub(crate) fn hint_regions(&self) -> (Rc<Weights>, [(u64, usize); 2]) {
        (self._weights.clone(), [(self.weight, 2048 * 4096), (self.scales, 8192)])
    }
    pub fn new(
        cuda: &Cuda,
        kernels: &Path,
        weights: Rc<Weights>,
        layer: usize,
        rows: usize,
    ) -> Result<Self> {
        Self::new_named(
            cuda,
            kernels,
            weights,
            crate::layer_id::LayerId::target(layer)?,
            rows,
        )
    }
    pub fn new_named(
        cuda: &Cuda,
        kernels: &Path,
        weights: Rc<Weights>,
        id: crate::layer_id::LayerId,
        rows: usize,
    ) -> Result<Self> {
        ensure!((2..=8).contains(&rows), "grouped WO-A rows not qualified");
        let p = format!("{}.attn.wo_a", id.prefix());
        let wn = format!("{p}.weight");
        let sn = format!("{p}.scale");
        let w = weights.view(&wn)?;
        let s = weights.view(&sn)?;
        ensure!(
            w.dtype == Dtype::Fp8
                && w.shape == [2048, 4096]
                && w.bytes == 2048 * 4096
                && s.dtype == Dtype::E8m0
                && s.shape == [64, 128]
                && s.bytes == 8192,
            "WO-A canonical TP4 geometry"
        );
        let weight = weights.ptr(&wn)?;
        let scales = weights.ptr(&sn)?;
        let partial_op = Operator::load(cuda, kernels, &format!("woa_w8_partial_m{rows}"))?;
        let quant_op = Operator::load(cuda, kernels, &format!("woa_reduce_quant_m{rows}"))?;
        ensure!(
            partial_op
                .entry
                .arguments
                .iter()
                .map(|a| a.kind.as_str())
                .collect::<Vec<_>>()
                == ["*bf16", "*fp8e4nv", "*u8", "*fp32"]
                && quant_op
                    .entry
                    .arguments
                    .iter()
                    .map(|a| a.kind.as_str())
                    .collect::<Vec<_>>()
                    == ["*fp32", "*fp8e4nv", "*u8"],
            "WO-A ABI changed"
        );
        ensure!(
            (
                partial_op.entry.grid.x_mul,
                partial_op.entry.grid.x_add,
                partial_op.entry.grid.y,
                partial_op.entry.grid.z
            ) == (0, 16, 2, 8),
            "WO-A partial grid changed"
        );
        let mut partial = cuda.allocate::<f32>(8 * rows * 2048)?;
        partial.zero()?;
        let mut values = cuda.allocate::<u8>(rows * 2048)?;
        values.zero()?;
        let mut output_scales = cuda.allocate::<u8>(8192)?;
        output_scales.zero()?;
        Ok(Self {
            cuda: cuda.clone(),
            _weights: weights,
            weight,
            scales,
            rows,
            partial_op,
            quant_op,
            partial,
            values,
            output_scales,
            stream: Cell::new(0),
            native: None,
            native_on: Cell::new(false),
            fused_direct: None,
            fused_direct_on: Cell::new(false),
        })
    }
    /// Install single-kernel fused WO-A + MXFP8 quant (ca_woa_fused_quant_m6 from DIRECT_GEMV_CUBIN).
    pub(crate) fn install_fused_direct(&mut self, on: bool) -> Result<()> {
        ensure!(self.fused_direct.is_none() && self.stream.get() == 0 && self.rows <= 8 && self.weight % 16 == 0 && self.scales != 0,
            "fused direct WO-A requires idle small owner over aligned canonical bytes");
        let module = self.cuda.module(crate::fp8_linear::DIRECT_GEMV_CUBIN)?;
        module.function("ca_woa_fused_quant_m6")?;
        self.fused_direct = Some(module);
        self.fused_direct_on.set(on);
        Ok(())
    }
    pub(crate) fn fused_direct_installed(&self) -> bool { self.fused_direct.is_some() }
    pub(crate) fn set_fused_direct_for_probe(&self, on: bool) -> Result<()> {
        ensure!(self.fused_direct.is_some(), "fused direct WO-A not installed");
        self.fused_direct_on.set(on);
        Ok(())
    }
    /// Install ca_woa_partial_w8 (same arithmetic as the exported partial kernel; see kernels/core.cu), on for
    /// serving (CANADA_SCRATCH_WOA_NATIVE=1) or off until the plan screen switches it.
    pub(crate) fn install_native(&mut self, module: &crate::cuda::Module, on: bool) -> Result<()> {
        ensure!(self.native.is_none() && self.stream.get() == 0 && self.weight % 16 == 0 && self.scales != 0,
            "native WO-A is installed once, before use, over aligned canonical bytes");
        module.function("ca_woa_partial_w8")?;
        self.native = Some(module.clone());
        self.native_on.set(on);
        Ok(())
    }
    pub(crate) fn set_native_for_probe(&self, on: bool) -> Result<()> {
        ensure!(crate::config::woa_native_screen() && self.native.is_some(), "native WO-A screen");
        self.native_on.set(on);
        Ok(())
    }
    pub fn values(&self) -> &Buffer<u8> {
        &self.values
    }
    pub fn scales(&self) -> &Buffer<u8> {
        &self.output_scales
    }
    pub fn partial(&self) -> &Buffer<f32> {
        &self.partial
    }
    pub fn run(&self, input: &Buffer<bf16>, stream: &Stream) -> Result<()> {
        ensure!(
            input.len() == self.rows * 8192
                && self.cuda.owns_buffer(input)
                && self.cuda.owns_stream(stream),
            "WO-A input/context"
        );
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == raw,
            "WO-A workspace needs one ordered stream"
        );
        self.stream.set(raw);
        if let Some(module) = self.fused_direct.as_ref().filter(|_| self.fused_direct_on.get()) {
            let (mut x, mut w, mut s, mut vals, mut sc_mma, mut m) = (
                input.ptr(), self.weight, self.scales, self.values.ptr(), self.output_scales.ptr(), self.rows as i32
            );
            let mut args: [*mut std::ffi::c_void; 6] = [
                (&mut x as *mut u64).cast(), (&mut w as *mut u64).cast(), (&mut s as *mut u64).cast(),
                (&mut vals as *mut u64).cast(), (&mut sc_mma as *mut u64).cast(), (&mut m as *mut i32).cast(),
            ];
            return unsafe {
                module.function("ca_woa_fused_quant_m6")?
                    .launch_pdl([64, 1, 1], [256, 1, 1], 0, stream, &mut args)
            };
        }
        if let Some(module) = self.native.as_ref().filter(|_| self.native_on.get()) {
            let (mut x, mut w, mut s, mut p, mut m) =
                (input.ptr(), self.weight, self.scales, self.partial.ptr(), self.rows as i32);
            let mut args: [*mut std::ffi::c_void; 5] = [(&mut x as *mut u64).cast(), (&mut w as *mut u64).cast(),
                (&mut s as *mut u64).cast(), (&mut p as *mut u64).cast(), (&mut m as *mut i32).cast()];
            // The exported kernel's grid: 16 column tiles of 64, 2 groups, 8 K splits; four warps.
            unsafe { module.function("ca_woa_partial_w8")?.launch([16, 2, 8], [128, 1, 1], 0, stream, &mut args) }?;
        } else {
        unsafe {
            self.partial_op.launch(
                self.rows as u32,
                &[input.ptr(), self.weight, self.scales, self.partial.ptr()],
                stream,
            )
        }?;
        }
        unsafe {
            self.quant_op.launch(
                self.rows as u32,
                &[
                    self.partial.ptr(),
                    self.values.ptr(),
                    self.output_scales.ptr(),
                ],
                stream,
            )
        }
    }
}
