//! Bounded GPU prefill pooling. Read carried state, then publish each request's
//! final ring window in a separate launch, preventing cross-chunk wrap races.
use crate::{
    cuda::{Buffer, Cuda, Stream},
    triton_ops::Operator,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{cell::Cell, path::Path};
pub struct Ring {
    cuda: Cuda,
    values: Buffer<f32>,
    requests: usize,
    width: usize,
    stream: Cell<usize>,
}
impl Ring {
    pub fn new(cuda: &Cuda, requests: usize, width: usize) -> Result<Self> {
        ensure!(
            (1..=128).contains(&requests) && (2..=64).contains(&width),
            "pair ring shape"
        );
        let mut values = cuda.allocate(requests * width * 1024)?;
        values.zero()?;
        Ok(Self {
            cuda: cuda.clone(),
            values,
            requests,
            width,
            stream: Cell::new(0),
        })
    }
    pub fn values(&self) -> &Buffer<f32> {
        &self.values
    }
    pub(crate) fn values_mut(&mut self) -> &mut Buffer<f32> {
        &mut self.values
    }
    pub(crate) fn bind(&self, stream: &Stream) -> Result<()> {
        ensure!(self.cuda.owns_stream(stream), "pair ring stream/context");
        let p = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == p,
            "pair ring requires one ordered stream"
        );
        self.stream.set(p);
        Ok(())
    }
}
struct Plan {
    previous: Vec<i32>,
    keep: Vec<i32>,
}
fn plan(
    pos: &[i32],
    req: &[i64],
    loc: &[i64],
    requests: usize,
    ring: usize,
    capacity: usize,
    freq: usize,
) -> Result<Plan> {
    let n = pos.len();
    ensure!(
        (1..=crate::config::MAX_WIDE_ROWS).contains(&n)
            && req.len() == n
            && loc.len() == n
            && (1..=128).contains(&requests)
            && [8, 16, 32, 64].contains(&ring)
            && capacity > 0
            && capacity <= i32::MAX as usize
            && freq > 0
            && freq <= 1048576,
        "prefill pair metadata shape"
    );
    let mut seen = std::collections::BTreeSet::new();
    let mut slots = std::collections::BTreeSet::new();
    let mut previous = vec![-2; n];
    let mut keep = vec![0; n];
    let mut blocks = Vec::new();
    for i in 0..n {
        ensure!(
            pos[i] >= 0 && (pos[i] as usize) < freq && req[i] >= 0 && (req[i] as usize) < requests,
            "prefill pair position/request bounds"
        );
        if loc[i] == 0 {
            continue;
        }
        ensure!(
            loc[i] >= 2
                && loc[i] <= i32::MAX as i64
                && (loc[i] as usize) < capacity
                && loc[i] % 2 == pos[i] as i64 % 2
                && slots.insert(loc[i]),
            "prefill pair location/alias"
        );
        if i > 0 && loc[i - 1] != 0 && req[i] == req[i - 1] {
            ensure!(
                pos[i] == pos[i - 1] + 1,
                "prefill request positions must be consecutive"
            );
            previous[i] = (i - 1) as i32;
            blocks.last_mut().map(|b: &mut (usize, usize)| b.1 = i + 1);
        } else {
            ensure!(
                seen.insert(req[i]),
                "prefill request must occupy one contiguous active block"
            );
            previous[i] = if pos[i] > 0 { -1 } else { -2 };
            blocks.push((i, i + 1));
        }
    }
    for (start, end) in blocks {
        for v in &mut keep[start.max(end.saturating_sub(ring))..end] {
            *v = 1;
        }
    }
    Ok(Plan { previous, keep })
}
pub struct Schedule {
    positions: Buffer<i32>,
    requests: Buffer<i64>,
    previous: Buffer<i32>,
    keep: Buffer<i32>,
    rows: usize,
    request_slots: usize,
    ring: usize,
    capacity: usize,
    frequency_rows: usize,
}
impl Schedule {
    pub fn new(
        cuda: &Cuda,
        pos: &[i32],
        req: &[i64],
        loc: &[i64],
        requests: usize,
        ring: usize,
        capacity: usize,
        freq: usize,
    ) -> Result<Self> {
        let p = plan(pos, req, loc, requests, ring, capacity, freq)?;
        let mut positions = cuda.allocate(pos.len())?;
        let mut requests_gpu = cuda.allocate(pos.len())?;
        let mut previous = cuda.allocate(pos.len())?;
        let mut keep = cuda.allocate(pos.len())?;
        positions.upload(pos)?;
        requests_gpu.upload(req)?;
        previous.upload(&p.previous)?;
        keep.upload(&p.keep)?;
        Ok(Self {
            positions,
            requests: requests_gpu,
            previous,
            keep,
            rows: pos.len(),
            request_slots: requests,
            ring,
            capacity,
            frequency_rows: freq,
        })
    }
    pub fn update(&mut self, pos: &[i32], req: &[i64], loc: &[i64]) -> Result<()> {
        ensure!(
            pos.len() == self.rows,
            "prefill pair schedule row capacity changed"
        );
        let p = plan(
            pos,
            req,
            loc,
            self.request_slots,
            self.ring,
            self.capacity,
            self.frequency_rows,
        )?;
        self.positions.upload(pos)?;
        self.requests.upload(req)?;
        self.previous.upload(&p.previous)?;
        self.keep.upload(&p.keep)
    }
}
pub struct Pool {
    cuda: Cuda,
    pool: Operator,
    publish: Operator,
    rows: usize,
    ring: usize,
    output: Buffer<bf16>,
}
impl Pool {
    pub fn new(cuda: &Cuda, root: &Path, rows: usize, ring: usize) -> Result<Self> {
        ensure!(
            (1..=crate::config::MAX_WIDE_ROWS).contains(&rows) && [8, 16, 32, 64].contains(&ring),
            "prefill pool dimensions"
        );
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        for name in [format!("pool_r{ring}"), format!("publish_r{ring}")] {
            let entry = manifest["entries"]
                .as_array()
                .and_then(|a| a.iter().find(|e| e["name"] == name))
                .ok_or_else(|| anyhow::anyhow!("prefill pool contract absent"))?;
            ensure!(
                entry["reflect_ftz"] == false
                    && entry["fp_fusion"] == false
                    && entry["ring"] == ring,
                "prefill pool precision contract"
            );
        }
        let pool = Operator::load(cuda, root, &format!("pool_r{ring}"))?;
        let publish = Operator::load(cuda, root, &format!("publish_r{ring}"))?;
        Ok(Self {
            cuda: cuda.clone(),
            pool,
            publish,
            rows,
            ring,
            output: cuda.allocate(rows * 512)?,
        })
    }
    pub fn run(
        &self,
        input: &Buffer<f32>,
        state: &Ring,
        meta: &Schedule,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            input.len() == self.rows * 1024
                && meta.rows == self.rows
                && meta.ring == self.ring
                && state.width == self.ring
                && state.requests == meta.request_slots
                && self.cuda.owns_buffer(input)
                && self.cuda.owns_buffer(state.values())
                && self.cuda.owns_buffer(&meta.positions)
                && self.cuda.owns_buffer(&meta.requests)
                && self.cuda.owns_buffer(&meta.previous)
                && self.cuda.owns_buffer(&meta.keep)
                && self.cuda.owns_stream(stream),
            "prefill pool state/metadata/context"
        );
        state.bind(stream)?;
        unsafe {
            self.pool.launch(
                self.rows as u32,
                &[
                    input.ptr(),
                    state.values.ptr(),
                    meta.positions.ptr(),
                    meta.requests.ptr(),
                    meta.previous.ptr(),
                    self.output.ptr(),
                ],
                stream,
            )?;
            self.publish.launch(
                self.rows as u32,
                &[
                    input.ptr(),
                    state.values.ptr(),
                    meta.positions.ptr(),
                    meta.requests.ptr(),
                    meta.keep.ptr(),
                ],
                stream,
            )
        }
    }
    pub fn output(&self) -> &Buffer<bf16> {
        &self.output
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn long_chunk_only_publishes_tail() {
        let p = plan(
            &(0..20).collect::<Vec<_>>(),
            &vec![0; 20],
            &(2..22).collect::<Vec<_>>(),
            1,
            8,
            128,
            32,
        )
        .unwrap();
        assert_eq!(p.previous[0], -2);
        assert_eq!(p.previous[19], 18);
        assert_eq!(p.keep.iter().sum::<i32>(), 8);
        assert!(p.keep[..12].iter().all(|v| *v == 0));
    }
    #[test]
    fn carried_and_separate_requests() {
        let p = plan(
            &[31, 32, 33, 100, 101, 0],
            &[2, 2, 2, 1, 1, 0],
            &[33, 34, 35, 100, 101, 0],
            3,
            8,
            256,
            256,
        )
        .unwrap();
        assert_eq!(p.previous, [-1, 0, 1, -1, 3, -2]);
        assert_eq!(p.keep, [1, 1, 1, 1, 1, 0]);
    }
    #[test]
    fn reject_discontinuous_or_aliasing_schedule() {
        assert!(plan(&[0, 2], &[0, 0], &[2, 4], 1, 8, 64, 64).is_err());
        assert!(plan(&[0, 1], &[0, 0], &[2, 2], 1, 8, 64, 64).is_err());
        assert!(plan(&[0, 1, 2], &[0, 1, 0], &[2, 3, 4], 2, 8, 64, 64).is_err());
    }
}
