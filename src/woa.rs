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
}
impl Grouped {
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
        })
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
        unsafe {
            self.partial_op.launch(
                self.rows as u32,
                &[input.ptr(), self.weight, self.scales, self.partial.ptr()],
                stream,
            )
        }?;
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
