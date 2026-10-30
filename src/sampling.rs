//! GPU-only greedy target sampler with explicit failure and tie contracts.
//! DSpark greedy integration lives in dspark_generation. Stochastic/top-p/top-k
//! and target-correct speculative rejection/residual sampling remain open.
use crate::cuda::{Buffer, Cuda, Module, Stream};
use anyhow::{Result, ensure};

pub const VOCAB: usize = 129280;
const TILES: usize = VOCAB.div_ceil(1024);

pub struct Greedy {
    cuda: Cuda,
    module: Module,
    rows: usize,
    values: Buffer<f32>,
    ids: Buffer<i32>,
    errors: Buffer<u32>,
    output: Buffer<i32>,
    stream: Option<usize>,
    queued: bool,
}
impl Greedy {
    /// A graph owner may wire this stable device result to the next GPU stage.
    /// It must validate all sampler errors after completing its replay.
    pub(crate) fn device_output(&self) -> Result<&Buffer<i32>> {
        ensure!(self.queued, "greedy device result not enqueued");
        Ok(&self.output)
    }
    /// Stable error storage for a retaining graph owner. The owner must
    /// validate these flags together with the output after stream completion.
    pub(crate) fn device_errors(&self) -> Result<&Buffer<u32>> {
        ensure!(self.queued, "greedy device status not enqueued");
        Ok(&self.errors)
    }
    pub fn new(cuda: &Cuda, core: &[u8], rows: usize) -> Result<Self> {
        ensure!((1..=crate::config::MAX_WIDE_ROWS).contains(&rows), "greedy row capacity");
        cuda.qualify_gb10()?;
        Ok(Self {
            cuda: cuda.clone(),
            module: cuda.module(core)?,
            rows,
            values: cuda.allocate(rows * TILES)?,
            ids: cuda.allocate(rows * TILES)?,
            errors: cuda.allocate(rows)?,
            output: cuda.allocate(rows)?,
            stream: None,
            queued: false,
        })
    }
    /// All logits and scratch remain device-resident. Enqueue is capturable;
    /// callers must finish and validate before committing sampled model state.
    pub fn run(&mut self, logits: &Buffer<f32>, stream: &Stream) -> Result<()> {
        ensure!(
            self.cuda.owns_buffer(logits)
                && logits.len() == self.rows * VOCAB
                && self.cuda.owns_stream(stream),
            "greedy input shape/context"
        );
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.is_none() || self.stream == Some(raw),
            "greedy requires one ordered stream"
        );
        self.stream = Some(raw);
        self.queued = false;
        unsafe {
            self.cuda
                .zero_device(self.errors.ptr(), self.errors.bytes(), stream)?;
        }
        let mut x = logits.ptr();
        let mut v = self.values.ptr();
        let mut ids = self.ids.ptr();
        let mut errors = self.errors.ptr();
        let mut out = self.output.ptr();
        let mut args = [
            (&mut x as *mut u64).cast(),
            (&mut v as *mut u64).cast(),
            (&mut ids as *mut u64).cast(),
            (&mut errors as *mut u64).cast(),
        ];
        unsafe {
            self.module.function("ca_greedy_partials")?.launch(
                [TILES as u32, self.rows as u32, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )?;
        }
        let mut args = [
            (&mut v as *mut u64).cast(),
            (&mut ids as *mut u64).cast(),
            (&mut errors as *mut u64).cast(),
            (&mut out as *mut u64).cast(),
        ];
        unsafe {
            self.module.function("ca_greedy_finish")?.launch(
                [self.rows as u32, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )?;
        }
        self.queued = true;
        Ok(())
    }
    pub fn finish(&mut self, stream: &Stream) -> Result<Vec<i32>> {
        ensure!(
            self.queued
                && self.stream == Some(stream.raw() as usize)
                && self.cuda.owns_stream(stream),
            "greedy result is not queued on this stream"
        );
        self.queued = false;
        stream.synchronize()?;
        ensure!(
            self.errors.download()?.iter().all(|&e| e == 0),
            "greedy logits contain NaN, positive infinity or an entirely masked row"
        );
        let ids = self.output.download()?;
        ensure!(
            ids.iter().all(|&id| (0..VOCAB as i32).contains(&id)),
            "greedy output outside vocabulary"
        );
        Ok(ids)
    }
}
