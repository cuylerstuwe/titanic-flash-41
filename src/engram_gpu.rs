//! Independent Engram execution: bounded exact row I/O, GPU dequantization,
//! ownership reduction, split FP8 projection, raw gather and canonical gate.
//! Preparation is separate from the graph so row I/O can run ahead of layers.
use crate::{
    blockscaled::Matrix,
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, MappedHost, Module, Stream},
    engram::{ROW_BYTES, RowStore},
    ffn::TokenRows,
    fp8_linear::Linear,
    roce::Roce,
    triton_ops::Operator,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{cell::Cell, path::Path, rc::Rc, sync::atomic::{fence, Ordering}};

enum RowInputs {
    Copied { packed: Buffer<u8>, ids: Buffer<i64>, staging: Vec<u8> },
    Mapped { packed: MappedHost, ids: MappedHost },
}
impl RowInputs {
    fn pointers(&self) -> (u64, u64) {
        match self {
            Self::Copied { packed, ids, .. } => (packed.ptr(), ids.ptr()),
            Self::Mapped { packed, ids } => (packed.device_ptr(), ids.device_ptr()),
        }
    }
    fn owned_by(&self, cuda: &Cuda) -> bool {
        match self {
            Self::Copied { packed, ids, .. } => cuda.owns_buffer(packed) && cuda.owns_buffer(ids),
            Self::Mapped { packed, ids } => cuda.owns_mapped_host(packed) && cuda.owns_mapped_host(ids),
        }
    }
}

pub struct Rows {
    cuda: Cuda,
    inputs: RowInputs,
    local: Buffer<bf16>,
    rows: usize,
    interval: (u64, u64, u64),
    prepared: bool,
}
impl Rows {
    pub fn new(cuda: &Cuda, rows: usize, store: &RowStore) -> Result<Self> {
        ensure!((1..=crate::config::MAX_WIDE_ROWS).contains(&rows), "Engram token-row capacity");
        let mapped=match std::env::var("CANADA_SCRATCH_ENGRAM_MAPPED") {
            Ok(value)=>{ensure!(value=="0"||value=="1","invalid Engram staging policy");value=="1"},
            Err(std::env::VarError::NotPresent)=>true,
            Err(error)=>return Err(error.into()),
        };
        let inputs=if mapped {
            // GB10's coherent UMA permits the existing GPU kernel to consume
            // the completed host lookup directly. Stable allocations survive
            // every captured graph; no quantization or row layout changes.
            RowInputs::Mapped {packed:cuda.mapped_host(rows*24*ROW_BYTES)?,ids:cuda.mapped_host(rows*24*8)?}
        }else{
            RowInputs::Copied {packed:cuda.allocate(rows*24*ROW_BYTES)?,ids:cuda.allocate(rows*24)?,staging:vec![0;rows*24*ROW_BYTES]}
        };
        Ok(Self {
            cuda: cuda.clone(),
            inputs,
            local: cuda.allocate(rows * 6144)?,
            rows,
            interval: store.interval(),
            prepared: false,
        })
    }
    pub fn prepare(&mut self, ids: &[[u64; 24]], store: &RowStore, threads: usize) -> Result<()> {
        let (flat, staging) = self.stage(ids, store)?;
        store.lookup_into(&flat, staging, threads)?;
        self.upload_staged(&flat)
    }
    /// Host half of `prepare`: validated flat row ids and the staging bytes a
    /// lookup fills. No CUDA call, so the lookup may run on another thread.
    pub fn stage(&mut self, ids: &[[u64; 24]], store: &RowStore) -> Result<(Vec<u64>, &mut [u8])> {
        ensure!(
            ids.len() == self.rows && store.interval() == self.interval,
            "Engram hash-row/store geometry changed"
        );
        self.prepared = false;
        let flat: Vec<u64> = ids.iter().flat_map(|r| r.iter().copied()).collect();
        ensure!(
            flat.iter().all(|id| *id < self.interval.2),
            "Engram id outside table"
        );
        let staging=match &mut self.inputs {
            RowInputs::Copied {staging,..}=>staging.as_mut_slice(),
            RowInputs::Mapped {packed,..}=>{
                // The caller prepares only after the prior consumer finished.
                // Scoped I/O workers borrow this slice and join before the
                // following publication; neither CUDA owner escapes.
                unsafe {std::slice::from_raw_parts_mut(packed.host_ptr().cast::<u8>(),packed.len())}
            }
        };
        Ok((flat, staging))
    }
    /// Device half of `prepare`, after `stage` and a completed lookup.
    pub fn upload_staged(&mut self, flat: &[u64]) -> Result<()> {
        ensure!(flat.len() == self.rows * 24, "Engram staged id count");
        match &mut self.inputs {
            RowInputs::Copied {packed,ids,staging}=>{
                packed.upload(staging)?;
                ids.upload(&flat.iter().map(|id| *id as i64).collect::<Vec<_>>())?;
            }
            RowInputs::Mapped {ids,..}=>{
                ensure!(ids.len()==flat.len()*8,"mapped Engram id extent");
                let destination=unsafe {std::slice::from_raw_parts_mut(ids.host_ptr().cast::<i64>(),flat.len())};
                for (out,&id) in destination.iter_mut().zip(flat){*out=id as i64;}
                // Order joined bytes and IDs before CUDA launch. The overlap
                // path's initial graph uses no Engram inputs; the remaining
                // graph starts after this publication.
                fence(Ordering::Release);
            }
        }
        self.prepared = true;
        Ok(())
    }
    pub fn local(&self) -> &Buffer<bf16> {
        &self.local
    }
    pub fn staging_bytes(&self) -> usize {
        self.rows*24*ROW_BYTES
    }
    pub(crate) fn input_kind(&self) -> &'static str {
        match self.inputs {RowInputs::Mapped {..}=>"mapped",RowInputs::Copied {..}=>"copied"}
    }
}
pub struct Layer {
    cuda: Cuda,
    weights: Rc<Weights>,
    rows: usize,
    layer: usize,
    dequant: Operator,
    gate: Operator,
    image: Operator,
    projection: Linear,
    embedding: Buffer<bf16>,
    kv: Buffer<bf16>,
    output: Buffer<bf16>,
    stream: Cell<usize>,
}
impl Layer {
    pub fn new(
        cuda: &Cuda,
        module: &Module,
        stream: &Stream,
        weights: Rc<Weights>,
        layer: usize,
        rows: usize,
        kernels: &Path,
    ) -> Result<Self> {
        ensure!(
            [1, 14].contains(&layer) && (1..=crate::config::MAX_WIDE_ROWS).contains(&rows),
            "canonical Engram layer/capacity"
        );
        let p = format!("layers.{layer}.engram");
        for name in ["q_weight", "k_weight"] {
            let v = weights.view(&format!("{p}.{name}"))?;
            ensure!(
                v.dtype == Dtype::Bf16 && v.shape == [4, 5120],
                "Engram gate weight shape/dtype"
            );
        }
        let matrix = Matrix::prepare(cuda, module, stream, &weights, &format!("{p}.wkv"))?;
        ensure!(
            (matrix.n, matrix.k) == (6400, 6144),
            "Engram WKV must use the qualified TP4 row split"
        );
        let projection = Linear::new(
            cuda,
            &kernels.join(format!("fp8-engram-wkv-m{rows}-v2")),
            matrix,
        )?;
        let dequant = Operator::load(cuda, &kernels.join("engram-v3"), "engram_packed")?;
        let gate = Operator::load(cuda, &kernels.join("engram-v3"), "engram_gate")?;
        let image = Operator::load(
            cuda,
            &kernels.join("engram-image-v1"),
            "engram_select_image",
        )?;
        ensure!(
            dequant
                .entry
                .arguments
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>()
                == ["packed_ptr", "ids_ptr", "out_ptr", "row_lo", "row_hi"],
            "Engram dequant ABI changed"
        );
        ensure!(
            gate.entry
                .arguments
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>()
                == ["X", "KV", "QW", "KW", "O"],
            "Engram gate ABI changed"
        );
        Ok(Self {
            cuda: cuda.clone(),
            weights,
            rows,
            layer,
            dequant,
            gate,
            image,
            projection,
            embedding: cuda.allocate_rows(rows * 6144)?,
            kv: cuda.allocate_rows(rows * 25600)?,
            output: cuda.allocate_rows(rows * 20480)?,
            stream: Cell::new(0),
        })
    }
    pub fn run(
        &mut self,
        residual: &Buffer<bf16>,
        rows: &mut Rows,
        tokens: &TokenRows,
        comm: &Roce,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            self.cuda.owns_buffer(residual)
                && rows.inputs.owned_by(&self.cuda)
                && rows.cuda.owns_stream(stream)
                && self.cuda.owns_stream(stream)
                && self.cuda.owns_buffer(tokens.ids())
                && tokens.ids().len() == self.rows
                && residual.len() == self.rows * 20480
                && rows.rows == self.rows
                && rows.prepared,
            "Engram prepared row/input context/shape mismatch"
        );
        let table = if self.layer == 1 {
            384006168
        } else {
            384016682
        };
        ensure!(
            rows.interval.2 == table,
            "Engram row store belongs to wrong layer table"
        );
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == raw,
            "Engram needs one ordered stream"
        );
        self.stream.set(raw);
        let p = format!("layers.{}.engram", self.layer);
        unsafe {
            self.dequant.launch(
                self.rows as u32,
                &[
                    rows.inputs.pointers().0,
                    rows.inputs.pointers().1,
                    rows.local.ptr(),
                    rows.interval.0,
                    rows.interval.1,
                ],
                stream,
            )?;
            comm.sum_bf16(&rows.local, &mut self.embedding, stream)?;
        }
        self.projection.run(&self.embedding, stream)?;
        unsafe {
            comm.gather_bf16(self.projection.output(), &mut self.kv, 6400, stream)?;
            self.gate.launch(
                self.rows as u32,
                &[
                    residual.ptr(),
                    self.kv.ptr(),
                    self.weights.ptr(&format!("{p}.q_weight"))?,
                    self.weights.ptr(&format!("{p}.k_weight"))?,
                    self.output.ptr(),
                ],
                stream,
            )?;
            self.image.launch(
                self.rows as u32,
                &[residual.ptr(), self.output.ptr(), tokens.ids().ptr()],
                stream,
            )
        }
    }
    pub fn embedding(&self) -> &Buffer<bf16> {
        &self.embedding
    }
    pub fn projected_local(&self) -> &Buffer<bf16> {
        self.projection.output()
    }
    pub fn projected(&self) -> &Buffer<bf16> {
        &self.kv
    }
    pub fn output(&self) -> &Buffer<bf16> {
        &self.output
    }
}
