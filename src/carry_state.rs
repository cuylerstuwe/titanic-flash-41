//! Bounded GPU copies of the compressor carry rings for graph warmup/replay.
//! This is not a complete session checkpoint or speculative rollback state.
use crate::cuda::{Buffer, Cuda, Stream};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};

struct Ring {
    live: u64,
    saved: Buffer<f32>,
}
pub(crate) struct Snapshot {
    cuda: Cuda,
    rings: Vec<Ring>,
}
impl Snapshot {
    pub(crate) fn new(cuda: &Cuda, rings: Vec<&Buffer<f32>>) -> Result<Self> {
        ensure!(rings.len() <= 4, "too many canonical carry owners");
        let mut saved = Vec::new();
        for ring in rings {
            ensure!(
                cuda.owns_buffer(ring)
                    && ring.bytes() <= 1024 * 1024
                    && ring.ptr() % 16 == 0
                    && !saved.iter().any(|r: &Ring| r.live == ring.ptr()),
                "carry ring context/size/identity changed"
            );
            saved.push(Ring {
                live: ring.ptr(),
                saved: cuda.allocate(ring.len())?,
            });
        }
        Ok(Self {
            cuda: cuda.clone(),
            rings: saved,
        })
    }
    pub(crate) fn bytes(&self) -> usize {
        self.rings.iter().map(|r| r.saved.bytes()).sum()
    }
    /// Safety: the Target holds the exclusive model-state lease on its ordered
    /// stream, and every live pointer still belongs to its original source.
    pub(crate) unsafe fn save(&mut self, stream: &Stream) -> Result<()> {
        for ring in &mut self.rings {
            unsafe {
                self.cuda
                    .copy_device(ring.saved.ptr(), ring.live, ring.saved.bytes(), stream)?;
            }
        }
        Ok(())
    }
    /// Safety: same ownership/lifetime contract as save; all prior consumers
    /// are ordered on this stream. Only carry state is restored, never outputs.
    pub(crate) unsafe fn restore(&self, stream: &Stream) -> Result<()> {
        for ring in &self.rings {
            unsafe {
                self.cuda
                    .copy_device(ring.live, ring.saved.ptr(), ring.saved.bytes(), stream)?;
            }
        }
        Ok(())
    }
    /// Restore only writes from rejected positions in a bounded verification.
    /// C2 writes an even position's1024 FP32 values into pos%ring_rows;
    /// completing odd positions do not change carry (qualified_c2.cuh).
    /// Safety: same exclusive ordered-stream contract as save/restore. The
    /// caller has not run another forward since this snapshot and verification.
    pub(crate) unsafe fn restore_rejected(
        &self,
        start: u64,
        live: usize,
        accepted: usize,
        stream: &Stream,
    ) -> Result<usize> {
        ensure!(
            live > 0
                && live <= 32
                && accepted <= live
                && self.rings.len() == 3
                && self.rings.iter().all(|r| [8 * 1024, 32 * 1024, 64 * 1024].contains(&r.saved.len())
                    && live < r.saved.len() / 1024)
                && self.rings.iter().all(|r| r.saved.len() == self.rings[0].saved.len()),
            "canonical verification carry geometry"
        );
        let mut restored = 0;
        for offset in accepted..live {
            let position = start
                .checked_add(offset as u64)
                .ok_or_else(|| anyhow::anyhow!("verification position overflow"))?;
            if position % 2 != 0 {
                continue;
            }
            let row = position as usize % (self.rings[0].saved.len() / 1024);
            for ring in &self.rings {
                unsafe {
                    self.cuda.copy_device(
                        ring.live + (row * 4096) as u64,
                        ring.saved.ptr() + (row * 4096) as u64,
                        4096,
                        stream,
                    )?;
                }
                restored += 4096;
            }
        }
        Ok(restored)
    }
    /// Byte-only diagnostic after the producer stream has completed.
    pub(crate) fn live_digests(&self) -> Result<Vec<String>> {
        self.rings
            .iter()
            .map(|r| {
                let bytes = unsafe { self.cuda.read_device(r.live, r.saved.bytes()) }?;
                Ok(format!("{:x}", Sha256::digest(bytes)))
            })
            .collect()
    }
}
