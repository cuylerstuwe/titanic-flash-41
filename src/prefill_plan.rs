//! Integer-only prefill admission and row scheduling. No numerical inference.
use anyhow::{Result, ensure};
use serde::Serialize;
use std::ops::Range;

#[derive(Debug, Serialize)]
pub struct Tail {
    pub token_indices: Vec<i32>,
    pub request_indices: Vec<i32>,
    pub positions: Vec<i32>,
    pub window_floors: Vec<i32>,
    pub lengths: Vec<usize>,
    pub contiguous_start: Option<usize>,
}
impl Tail {
    /// Effective qualified decoder-bounded-replay contract: after source20,
    /// each request contributes its last min(128, extend) rows, and the SWA
    /// window cannot read before that request's tail start at these layers.
    pub fn new(extend: &[usize], sequence: &[usize]) -> Result<Self> {
        ensure!(
            (1..=16).contains(&extend.len()) && extend.len() == sequence.len(),
            "prefill request count"
        );
        ensure!(
            extend
                .iter()
                .zip(sequence)
                .all(|(&e, &s)| e <= s && s <= 1048576)
                && extend.iter().sum::<usize>() <= crate::config::MAX_WIDE_ROWS,
            "prefill sequence/extend bounds"
        );
        let lengths: Vec<_> = extend.iter().map(|&n| n.min(128)).collect();
        let mut result = Self {
            token_indices: Vec::new(),
            request_indices: Vec::new(),
            positions: Vec::new(),
            window_floors: Vec::new(),
            contiguous_start: if extend.len() == 1 {
                Some(extend[0] - lengths[0])
            } else {
                None
            },
            lengths,
        };
        let mut offset = 0;
        for (req, ((&e, &s), &t)) in extend.iter().zip(sequence).zip(&result.lengths).enumerate() {
            let floor = (s - t) as i32;
            for i in 0..t {
                result.token_indices.push((offset + e - t + i) as i32);
                result.request_indices.push(req as i32);
                result.positions.push(floor + i as i32);
                result.window_floors.push(floor);
            }
            offset += e;
        }
        Ok(result)
    }
}

#[derive(Debug, Serialize)]
pub struct ScoreSchedule {
    pub stride: usize,
    pub rows_per_chunk: usize,
    pub arena_bytes: usize,
    pub chunks: Vec<Range<usize>>,
}
impl ScoreSchedule {
    pub fn new(rows: usize, columns: usize, budget: usize) -> Result<Self> {
        ensure!(
            (1..=crate::config::MAX_WIDE_ROWS).contains(&rows)
                && (1..=1048576).contains(&columns)
                && budget <= 2 * 1024 * 1024 * 1024usize,
            "prefill score schedule dimensions"
        );
        let stride = columns.div_ceil(128) * 128;
        let tile_bytes = 4 * stride * 4;
        ensure!(
            budget >= tile_bytes,
            "score budget cannot hold one physical four-query tile"
        );
        let rows_per_chunk = ((budget / tile_bytes) * 4).min(crate::config::MAX_WIDE_ROWS);
        let arena_bytes = rows.min(rows_per_chunk).div_ceil(4) * tile_bytes;
        let chunks = (0..rows)
            .step_by(rows_per_chunk)
            .map(|start| start..(start + rows_per_chunk).min(rows))
            .collect();
        Ok(Self {
            stride,
            rows_per_chunk,
            arena_bytes,
            chunks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tail_preserves_multi_request_offsets_and_absolute_window_floors() {
        let t = Tail::new(&[4, 140, 0, 1], &[100, 1000, 500, 1048576]).unwrap();
        assert_eq!(t.lengths, [4, 128, 0, 1]);
        assert_eq!(&t.token_indices[..6], &[0, 1, 2, 3, 16, 17]);
        assert_eq!(t.token_indices.last(), Some(&144));
        assert_eq!(&t.positions[..6], &[96, 97, 98, 99, 872, 873]);
        assert_eq!(&t.window_floors[..6], &[96, 96, 96, 96, 872, 872]);
        assert_eq!(t.positions.last(), Some(&1048575));
        assert!(t.contiguous_start.is_none());
        let one = Tail::new(&[4096], &[800000]).unwrap();
        assert_eq!(one.contiguous_start, Some(3968));
        assert_eq!(one.token_indices.len(), 128);
        assert_eq!(one.window_floors[127], 799872);
        let wide = Tail::new(&[8192], &[800000]).unwrap();
        assert_eq!(wide.contiguous_start, Some(8064));
        assert_eq!(wide.token_indices[127], 8191);
        assert_eq!(wide.window_floors[127], 799872);
    }
    #[test]
    fn tail_rejects_impossible_retained_lengths_and_chunk_overflow() {
        assert!(Tail::new(&[5], &[4]).is_err());
        let max = crate::config::MAX_WIDE_ROWS;
        let tail = Tail::new(&[max], &[max]).unwrap();
        assert_eq!(tail.token_indices.last(), Some(&((max - 1) as i32)));
        assert!(Tail::new(&[max + 1], &[max + 1]).is_err());
        assert!(Tail::new(&[max / 2, max / 2 + 1], &[max, max]).is_err());
        assert!(Tail::new(&[1], &[1048577]).is_err());
        assert!(Tail::new(&[], &[]).is_err());
        assert!(Tail::new(&[0], &[123]).unwrap().token_indices.is_empty());
    }
    #[test]
    fn score_budget_counts_tensor_tiles_and_partial_query_tiles() {
        // 130 live columns occupy a 256-column physical row, and even one
        // query writes a four-row tile. The naive live-score budget is unsafe.
        assert!(ScoreSchedule::new(1, 130, 130 * 4).is_err());
        let p = ScoreSchedule::new(6, 130, 4096).unwrap();
        assert_eq!(p.stride, 256);
        assert_eq!(p.arena_bytes, 4096);
        assert_eq!(p.chunks, [0..4, 4..6]);
        let long = ScoreSchedule::new(4096, 1048576, 64 * 1024 * 1024).unwrap();
        assert_eq!(long.rows_per_chunk, 16);
        assert_eq!(long.chunks.len(), 256);
        assert_eq!(long.arena_bytes, 64 * 1024 * 1024);
        let wide = ScoreSchedule::new(8192, 1048576, 64 * 1024 * 1024).unwrap();
        assert_eq!(wide.chunks.len(), 512);
        assert_eq!(wide.chunks.last(), Some(&(8176..8192)));
        assert_eq!(wide.arena_bytes, long.arena_bytes);
        assert!(ScoreSchedule::new(4096, 1048576, 16 * 1024 * 1024 - 1).is_err());
    }
}
