//! One-shot admission votes for the finite cache ownership probe only.
//! The serving build does not contain this module or its Worker field.
use anyhow::{Result, ensure};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(super) enum Point {
    EncoderBeforeAllocate,
    EncoderAfterAllocate,
    ReconstructedAfterCopy,
}

#[derive(Debug, Serialize)]
pub(super) struct Observation {
    pub point: Point,
    pub refusing_rank: usize,
    pub local_rank: usize,
    pub natural_admission: bool,
    pub published_admission: bool,
}

#[derive(Default)]
pub(super) struct Fault {
    armed: Option<(Point, usize)>,
    observation: Option<Observation>,
}

impl Fault {
    pub fn arm(&mut self, point: Point, refusing_rank: usize) -> Result<()> {
        ensure!(refusing_rank < 4 && self.armed.is_none() && self.observation.is_none(),
            "cache fault must be a fresh, single-rank vote");
        self.armed = Some((point, refusing_rank));
        Ok(())
    }

    pub fn apply(&mut self, point: Point, local_rank: usize, natural: bool) -> bool {
        if let Some((wanted, refusing_rank)) = self.armed.filter(|(wanted, _)| *wanted == point) {
            self.armed = None;
            let published = natural && local_rank != refusing_rank;
            self.observation = Some(Observation { point: wanted, refusing_rank, local_rank,
                natural_admission: natural, published_admission: published });
            published
        } else {
            natural
        }
    }

    pub fn take_observation(&mut self) -> Result<Observation> {
        ensure!(self.armed.is_none(), "requested cache refusal point was not reached");
        self.observation.take().ok_or_else(|| anyhow::anyhow!("cache refusal was not armed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_the_selected_rank_refuses_and_the_vote_is_one_shot() {
        for selected in 0..4 {
            for rank in 0..4 {
                let mut fault = Fault::default();
                fault.arm(Point::ReconstructedAfterCopy, selected).unwrap();
                assert!(fault.apply(Point::EncoderBeforeAllocate, rank, true));
                assert_eq!(fault.apply(Point::ReconstructedAfterCopy, rank, true), rank != selected);
                assert!(fault.apply(Point::ReconstructedAfterCopy, rank, true));
                let observed = fault.take_observation().unwrap();
                assert!(observed.natural_admission);
                assert_eq!(observed.local_rank, rank);
                assert_eq!(observed.refusing_rank, selected);
            }
        }
    }
    #[test]
    fn a_fault_never_overrides_a_real_refusal() {
        for point in [Point::EncoderBeforeAllocate, Point::EncoderAfterAllocate, Point::ReconstructedAfterCopy] {
            for rank in 0..4 {
                let mut fault = Fault::default();
                fault.arm(point, 0).unwrap();
                assert!(!fault.apply(point, rank, false));
                assert!(!fault.take_observation().unwrap().natural_admission);
            }
        }
    }
    #[test]
    fn missing_or_overlapping_injections_cannot_claim_coverage() {
        let mut fault = Fault::default();
        assert!(fault.arm(Point::EncoderAfterAllocate, 4).is_err());
        fault.arm(Point::EncoderAfterAllocate, 1).unwrap();
        assert!(fault.arm(Point::EncoderBeforeAllocate, 2).is_err());
        assert!(fault.take_observation().is_err());
        assert!(fault.apply(Point::EncoderBeforeAllocate, 1, true));
        assert!(fault.take_observation().is_err());
        assert!(!fault.apply(Point::EncoderAfterAllocate, 1, true));
        assert!(fault.arm(Point::EncoderAfterAllocate, 1).is_err());
        fault.take_observation().unwrap();
        fault.arm(Point::EncoderBeforeAllocate, 3).unwrap();
    }
}
