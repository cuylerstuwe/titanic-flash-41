//! Causal Engram hashing with transactional speculative history.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HashMetadata {
    pub token_map: Vec<u32>,
    pub compressed_vocab_size: u32,
    pub pad_token: u32,
    pub image_token: u32,
    pub multipliers: [[u64; 4]; 2],
    pub primes: [[u64; 24]; 2],
    pub offsets: [[u64; 24]; 2],
    pub table_rows: [u64; 2],
}
impl HashMetadata {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let m: Self = serde_json::from_slice(bytes)?;
        m.validate()?;
        Ok(m)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.token_map.len() == 129280
                && self.compressed_vocab_size == 99092
                && self.pad_token == 2
                && self.image_token == 129264,
            "unqualified Engram tokenizer metadata"
        );
        ensure!(
            self.token_map
                .iter()
                .all(|v| *v < self.compressed_vocab_size),
            "compressed token outside vocabulary"
        );
        ensure!(
            self.table_rows == [384006168, 384016682],
            "Engram table geometry drift"
        );
        for l in 0..2 {
            let mut offset = 0;
            for h in 0..24 {
                ensure!(
                    self.primes[l][h] >= 16000000 && self.offsets[l][h] == offset,
                    "Engram prime/offset layout drift"
                );
                offset += self.primes[l][h];
            }
            ensure!(
                offset == self.table_rows[l],
                "Engram prime sum disagrees with table size"
            );
            for m in self.multipliers[l] {
                ensure!(
                    m % 2 == 1
                        && m.checked_mul(self.compressed_vocab_size as u64)
                            .is_some_and(|v| v <= i64::MAX as u64),
                    "invalid Engram multiplier"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct History {
    pub lineage: u64,
    pub generation: u64,
    pub position: u64,
    recent: [u32; 3],
}
pub struct HashBatch {
    parent: History,
    tokens: Vec<u32>,
    pub rows: Vec<[[u64; 24]; 2]>,
}
impl History {
    pub fn new(lineage: u64) -> Self {
        Self {
            lineage,
            generation: 0,
            position: 0,
            recent: [2; 3],
        }
    }
    pub fn prepare(&self, tokens: &[u32], metadata: &HashMetadata) -> Result<HashBatch> {
        ensure!(self.lineage != 0, "history needs an owned sequence lineage");
        ensure!(
            self.position
                .checked_add(tokens.len() as u64)
                .is_some_and(|p| p <= 1048576),
            "Engram context limit"
        );
        ensure!(
            tokens
                .iter()
                .all(|id| (*id as usize) < metadata.token_map.len()),
            "token outside canonical vocabulary"
        );
        let pad = metadata.token_map[metadata.pad_token as usize] as u64;
        let mut rows = Vec::with_capacity(tokens.len());
        for i in 0..tokens.len() {
            let mut compressed = [pad; 4];
            let mut blocked = false;
            for back in 0..4 {
                if self.position + (i as u64) < back as u64 {
                    blocked = true;
                }
                let token = if back <= i {
                    tokens[i - back]
                } else {
                    self.recent[back - i - 1]
                };
                if token == metadata.image_token {
                    blocked = true;
                }
                if !blocked {
                    compressed[back] = metadata.token_map[token as usize] as u64;
                }
            }
            let mut hashes = [[0u64; 24]; 2];
            for l in 0..2 {
                let mut rolling = compressed[0] * metadata.multipliers[l][0];
                for back in 1..4 {
                    rolling ^= compressed[back] * metadata.multipliers[l][back];
                    for head in 0..8 {
                        let h = (back - 1) * 8 + head;
                        hashes[l][h] = rolling % metadata.primes[l][h] + metadata.offsets[l][h];
                    }
                }
            }
            rows.push(hashes);
        }
        Ok(HashBatch {
            parent: self.clone(),
            tokens: tokens.to_vec(),
            rows,
        })
    }
    /// Raw cache-read hints for one following chunk of already known prompt
    /// input. This preview neither publishes nor advances the causal history.
    /// Reuse the exact hashing path, including image barriers and left padding.
    pub(crate) fn preview_known_input_rows(
        &self,
        current: &[u32],
        following: &[u32],
        metadata: &HashMetadata,
    ) -> Result<Vec<[[u64; 24]; 2]>> {
        ensure!(
            (1..=crate::config::MAX_WIDE_ROWS).contains(&current.len())
                && (1..=crate::config::MAX_WIDE_ROWS).contains(&following.len()),
            "Engram lookahead is bounded to one encoder chunk"
        );
        let mut tokens = Vec::with_capacity(current.len() + following.len());
        tokens.extend_from_slice(current);
        tokens.extend_from_slice(following);
        let mut batch = self.prepare(&tokens, metadata)?;
        Ok(batch.rows.split_off(current.len()))
    }
    pub fn commit(&mut self, batch: HashBatch, accepted: usize) -> Result<()> {
        ensure!(
            *self == batch.parent,
            "stale or foreign Engram history commit"
        );
        ensure!(
            accepted <= batch.tokens.len(),
            "accepted prefix exceeds prepared batch"
        );
        if accepted == 0 {
            return Ok(());
        }
        let generation = self
            .generation
            .checked_add(1)
            .context("history generation exhausted")?;
        for id in &batch.tokens[..accepted] {
            self.recent = [*id, self.recent[0], self.recent[1]];
        }
        self.position += accepted as u64;
        self.generation = generation;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn meta() -> HashMetadata {
        let mut token_map: Vec<_> = (0..129280).map(|i| i % 99092).collect();
        token_map[2] = 2;
        let primes = [[16000001; 24]; 2];
        let mut offsets = [[0; 24]; 2];
        for l in 0..2 {
            for i in 1..24 {
                offsets[l][i] = offsets[l][i - 1] + primes[l][i - 1];
            }
        }
        HashMetadata {
            token_map,
            compressed_vocab_size: 99092,
            pad_token: 2,
            image_token: 129264,
            multipliers: [[11, 13, 17, 19], [23, 29, 31, 37]],
            primes,
            offsets,
            table_rows: [384000024; 2],
        }
    }
    #[test]
    fn chunking_and_image_barrier_are_invariant() {
        let m = meta();
        let mut tokens: Vec<u32> = (0..129).map(|i| i * 997 % 129280).collect();
        for i in [0, 3, 22, 127] {
            tokens[i] = 129264;
        }
        let h = History::new(7);
        let expected = h.prepare(&tokens, &m).unwrap().rows;
        let mut h = h;
        let mut actual = Vec::new();
        let mut pos = 0;
        for n in [1, 3, 8, 7, 110] {
            let b = h.prepare(&tokens[pos..pos + n], &m).unwrap();
            actual.extend(b.rows.iter().cloned());
            h.commit(b, n).unwrap();
            pos += n;
        }
        assert_eq!(actual, expected);
    }
    #[test]
    fn rejected_future_and_stale_batches_cannot_change_history() {
        let m = meta();
        let mut h = History::new(11);
        let p = h.prepare(&[4, 5, 6, 7, 8, 9], &m).unwrap();
        h.commit(p, 3).unwrap();
        let mut expected = History::new(11);
        let p = expected.prepare(&[4, 5, 6], &m).unwrap();
        expected.commit(p, 3).unwrap();
        assert_eq!(h, expected);
        let old = h.prepare(&[10], &m).unwrap();
        let new = h.prepare(&[11], &m).unwrap();
        h.commit(new, 1).unwrap();
        assert!(h.commit(old, 1).is_err());
        let before = h.clone();
        let pending = h.prepare(&[12, 13], &m).unwrap();
        h.commit(pending, 0).unwrap();
        assert_eq!(h, before);
    }
    #[test]
    fn lookahead_matches_committed_chunk_hashes_without_publishing_history() {
        let m = meta();
        for initial in [0, 1, 127, 1024] {
            let mut h = History::new(23);
            let prefix: Vec<u32> = (0..initial).map(|i| i as u32 + 3).collect();
            let b = h.prepare(&prefix, &m).unwrap();
            h.commit(b, prefix.len()).unwrap();
            let before = h.clone();
            for current in [vec![7], vec![8, 129264, 9], vec![13; crate::config::MAX_WIDE_ROWS]] {
                let following = [129264, 17, 18, 19, 2, 2];
                let hint = h.preview_known_input_rows(&current, &following, &m).unwrap();
                let mut expected = h.clone();
                let b = expected.prepare(&current, &m).unwrap();
                expected.commit(b, current.len()).unwrap();
                assert_eq!(hint, expected.prepare(&following, &m).unwrap().rows);
                assert_eq!(h, before);
            }
            assert!(h.preview_known_input_rows(&[], &[7], &m).is_err());
            assert!(h.preview_known_input_rows(&[7], &[129280], &m).is_err());
            let following=vec![8;crate::config::MAX_WIDE_ROWS];
            let mut expected=h.clone();
            let batch=expected.prepare(&[7],&m).unwrap();
            expected.commit(batch,1).unwrap();
            assert_eq!(h.preview_known_input_rows(&[7],&following,&m).unwrap(),
                expected.prepare(&following,&m).unwrap().rows);
            assert!(h.preview_known_input_rows(&[7], &vec![8; crate::config::MAX_WIDE_ROWS+1], &m).is_err());
            assert_eq!(h, before);
        }
    }
}
