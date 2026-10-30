//! Sliding-window block-manager addresses, not a CPU model implementation.
//! A whole prefill chunk is stored before its first attention read. The ring
//! must retain 127 incoming keys plus all128 new rows, including speculative
//! writes that are not yet committed. A128-slot ring would corrupt early rows.
use anyhow::{Result, ensure};

pub(crate) const OFFSET: usize = 64;
pub(crate) const WINDOW: usize = 128;
pub(crate) const MAX_ROWS: usize = 128;
const RING_SLOTS: usize = (WINDOW + MAX_ROWS - 1).next_power_of_two();
const WIDE_RING_SLOTS: usize = (WINDOW + crate::config::DEFAULT_WIDE_ROWS - 1).next_power_of_two();
const LARGE_RING_SLOTS: usize = (WINDOW + 8192 - 1).next_power_of_two();
const XL_RING_SLOTS: usize = (WINDOW + 16384 - 1).next_power_of_two();

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Layout {
    Absolute,
    Ring,
    WideRing,
    WideRing8192,
    WideRing16384,
}
impl Layout {
    pub(crate) fn max_rows(self) -> usize {
        match self {
            Self::Ring => MAX_ROWS,
            Self::WideRing => crate::config::DEFAULT_WIDE_ROWS,
            Self::WideRing8192 => 8192,
            _ => crate::config::MAX_WIDE_ROWS,
        }
    }
    fn ring_slots(self) -> Option<usize> {
        match self {
            Self::Absolute => None,
            Self::Ring => Some(RING_SLOTS),
            Self::WideRing => Some(WIDE_RING_SLOTS),
            Self::WideRing8192 => Some(LARGE_RING_SLOTS),
            Self::WideRing16384 => Some(XL_RING_SLOTS),
        }
    }
    pub(crate) fn slot(self, position: usize) -> usize {
        OFFSET
            + match self {
                Self::Absolute => position,
                Self::Ring => position % RING_SLOTS,
                Self::WideRing => position % WIDE_RING_SLOTS,
                Self::WideRing8192 => position % LARGE_RING_SLOTS,
                Self::WideRing16384 => position % XL_RING_SLOTS,
            }
    }
    pub(crate) fn pages(self, context: usize) -> Result<usize> {
        ensure!(
            (1..=1048576).contains(&context),
            "SWA logical context capacity"
        );
        let rows = match self {
            Self::Absolute => context,
            Self::Ring => context.min(RING_SLOTS),
            Self::WideRing => context.min(WIDE_RING_SLOTS),
            Self::WideRing8192 => context.min(LARGE_RING_SLOTS),
            Self::WideRing16384 => context.min(XL_RING_SLOTS),
        };
        Ok((OFFSET + rows).div_ceil(64).next_power_of_two())
    }
    pub(crate) fn retained(self, position: usize, frontier: usize) -> bool {
        position < frontier && (self == Self::Absolute || frontier - position <= WINDOW)
    }
    pub(crate) fn report(self, context: usize) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"layout":self,"logical_context":context,
            "attention_window":WINDOW,"maximum_prefill_rows":self.max_rows(),
            "ring_slots":self.ring_slots(),
            "allocated_physical_slots_per_layer":self.pages(context)? * 64,
            "scope":"SWA storage only. Compression/index/Engram and full model context have separate capacity/quality gates."}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    #[test]
    fn full_chunk_preserves_every_incoming_causal_key_across_wraps() {
        for start in [
            0usize, 1, 127, 128, 129, 255, 256, 257, 383, 511, 512, 1023, 3968,
        ] {
            for live in [1, 6, 128] {
                let first = start.saturating_sub(WINDOW - 1);
                let slots: BTreeSet<_> = (first..start + live)
                    .map(|p| Layout::Ring.slot(p))
                    .collect();
                assert_eq!(
                    slots.len(),
                    start + live - first,
                    "a future store aliases an incoming read"
                );
                assert!(
                    slots
                        .iter()
                        .all(|&s| s >= OFFSET && s < OFFSET + RING_SLOTS)
                );
                // Every query keeps newest-to-oldest logical ordering. Physical
                // slot order may wrap but must not reorder the reduction.
                for query in start..start + live {
                    let visible: Vec<_> = (query.saturating_sub(WINDOW - 1)..=query)
                        .rev()
                        .map(|p| Layout::Ring.slot(p))
                        .collect();
                    assert_eq!(
                        visible.iter().copied().collect::<BTreeSet<_>>().len(),
                        visible.len()
                    );
                }
            }
        }
        assert_eq!(Layout::Ring.pages(4096).unwrap(), 8);
        assert_eq!(Layout::Ring.pages(1048576).unwrap(), 8);
        assert_eq!(Layout::Absolute.pages(4096).unwrap(), 128);
        assert!(!Layout::Ring.retained(255, 384));
        assert!(Layout::Ring.retained(256, 384));
    }
    #[test]
    fn wide_ring_preserves_incoming_keys_and_complete_wide_store_across_wraps() {
        for start in [0usize, 1, 8065, 8191, 8192, 12287, 1044480] {
            for live in [1, 6, 128, 512, 1024, 2048, 4096] {
                let first = start.saturating_sub(WINDOW - 1);
                let slots: BTreeSet<_> = (first..start + live)
                    .map(|p| Layout::WideRing.slot(p))
                    .collect();
                assert_eq!(slots.len(), start + live - first);
                assert!(
                    slots
                        .iter()
                        .all(|&s| s >= OFFSET && s < OFFSET + WIDE_RING_SLOTS)
                );
                for query in [start, start + live / 2, start + live - 1] {
                    let visible: Vec<_> = (query.saturating_sub(WINDOW - 1)..=query)
                        .rev()
                        .map(|p| Layout::WideRing.slot(p))
                        .collect();
                    assert_eq!(
                        visible.iter().copied().collect::<BTreeSet<_>>().len(),
                        visible.len()
                    );
                }
            }
        }
        assert_eq!(Layout::WideRing.max_rows(), 4096);
        assert_eq!(Layout::WideRing.pages(1048576).unwrap(), 256);
        assert!(!Layout::WideRing.retained(8191, 8320));
        assert!(Layout::WideRing.retained(8192, 8320));
    }
    #[test]
    fn larger_ring_preserves_8192_stores_and_incoming_window_at_wrap() {
        for start in [0usize, 127, 8065, 8191, 16383, 16384, 1040384] {
            for live in [4096, 8191, 8192] {
                let first = start.saturating_sub(WINDOW - 1);
                let slots: BTreeSet<_> = (first..start + live)
                    .map(|p| Layout::WideRing8192.slot(p)).collect();
                assert_eq!(slots.len(), start + live - first);
                assert!(slots.iter().all(|&s| s >= OFFSET && s < OFFSET + LARGE_RING_SLOTS));
            }
        }
        assert_eq!(Layout::WideRing8192.max_rows(), 8192);
        assert_eq!(Layout::WideRing8192.pages(1048576).unwrap(), 512);
        // The ordinary profile retains its previous allocation and bound.
        assert_eq!(Layout::WideRing.max_rows(), 4096);
        assert_eq!(Layout::WideRing.pages(1048576).unwrap(), 256);
    }
    #[test]
    fn sixteen_k_ring_preserves_full_chunk_and_incoming_window() {
        for start in [0usize,127,32641,32767,32768,1048576-16384] {
            for live in [8192,16383,16384] {
                let first=start.saturating_sub(WINDOW-1);
                let slots:BTreeSet<_>=(first..start+live).map(|p|Layout::WideRing16384.slot(p)).collect();
                assert_eq!(slots.len(),start+live-first);
                assert!(slots.iter().all(|&s|s>=OFFSET && s<OFFSET+XL_RING_SLOTS));
            }
        }
        assert_eq!(Layout::WideRing16384.max_rows(),16384);
        assert_eq!(Layout::WideRing16384.pages(1048576).unwrap(),1024);
        // Existing8K allocations and addresses remain unchanged.
        assert_eq!(Layout::WideRing8192.pages(1048576).unwrap(),512);
    }

}
