//! Retention of disposable execution plans; never owns model state.
#[derive(Debug)]
pub(crate) struct Policy {
    limit: usize,
    clock: u64,
    ages: Vec<u64>,
}
impl Policy {
    pub(crate) fn new(rows: usize, limit: usize) -> Self {
        // All variants of the common six-row decoder remain resident.
        Self { limit: if rows <= 6 {0} else {limit}, clock:0, ages:vec![0;rows+1] }
    }
    pub(crate) fn limit(&self)->usize {self.limit}
    pub(crate) fn touch(&mut self,row:usize) {
        if self.limit==0 {return;}
        self.clock=self.clock.saturating_add(1); self.ages[row]=self.clock;
    }
    pub(crate) fn victim(&self,present:&[usize],incoming:usize)->Option<usize> {
        if self.limit==0 || present.len()+usize::from(!present.contains(&incoming))<=self.limit {return None;}
        present.iter().copied().filter(|&i|i!=incoming).min_by_key(|&i|(self.ages[i],i))
    }
}

/// Slot-indexed LRU retention for execution graph banks spanning single-session
/// widths and batched/region patterns. Pinned canonical slots never count toward
/// the eviction cap and are never selected as victims.
#[derive(Debug)]
pub(crate) struct SlotPolicy {
    limit: usize,
    clock: u64,
    ages: Vec<u64>,
}
impl SlotPolicy {
    pub(crate) fn new(slots: usize, limit: usize) -> Self {
        Self { limit, clock: 0, ages: vec![0; slots] }
    }
    pub(crate) fn limit(&self) -> usize { self.limit }
    pub(crate) fn touch(&mut self, slot: usize) {
        if self.limit == 0 || slot >= self.ages.len() { return; }
        self.clock = self.clock.saturating_add(1);
        self.ages[slot] = self.clock;
    }
    pub(crate) fn victim(&self, present: &[usize], incoming: usize, pinned: impl Fn(usize) -> bool) -> Option<usize> {
        if self.limit == 0 { return None; }
        let unpinned = present.iter().copied().filter(|&i| !pinned(i)).count()
            + usize::from(!present.contains(&incoming) && !pinned(incoming));
        if unpinned <= self.limit { return None; }
        present.iter().copied()
            .filter(|&i| i != incoming && !pinned(i))
            .min_by_key(|&i| (self.ages.get(i).copied().unwrap_or(0), i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounds_a_long_changing_working_set_and_preserves_the_requested_width() {
        let mut p=Policy::new(32,4);let mut resident=vec![];
        for row in [17,18,19,20,17,21,17,18,19,20,21,17].into_iter().cycle().take(1200) {
            p.touch(row);
            while let Some(v)=p.victim(&resident,row) {assert_ne!(v,row);resident.retain(|&i|i!=v);}
            if !resident.contains(&row) {resident.push(row);}
            assert!(resident.len()<=4);assert!(resident.contains(&row));
        }
        assert!(resident.contains(&17));
    }
    #[test]
    fn trims_an_oversized_bank_without_discarding_the_requested_graph() {
        let mut p=Policy::new(16,4);let mut resident=(7..=16).collect::<Vec<_>>();p.touch(7);
        while let Some(v)=p.victim(&resident,7) {resident.retain(|&i|i!=v);}
        assert_eq!(resident.len(),4);assert!(resident.contains(&7));
    }
    #[test]
    fn small_decoder_and_explicit_control_keep_every_variant() {
        for (rows,limit) in [(6,4),(32,0)] {
            let mut p=Policy::new(rows,limit);p.touch(1);
            assert_eq!(p.victim(&(1..=rows).collect::<Vec<_>>(),1),None);
        }
    }
    #[test]
    fn slot_policy_bounds_tail_and_region_slots_while_pinning_canonical_width() {
        let mut p = SlotPolicy::new(256, 4);
        let mut resident = vec![128usize];
        let pinned = |s: usize| s == 128;
        for slot in [33, 45, 71, 88, 33, 99, 102, 128, 33, 111, 200].into_iter().cycle().take(600) {
            p.touch(slot);
            while let Some(v) = p.victim(&resident, slot, pinned) {
                assert_ne!(v, slot);
                assert!(!pinned(v));
                resident.retain(|&i| i != v);
            }
            if !resident.contains(&slot) { resident.push(slot); }
            let unpinned = resident.iter().copied().filter(|&s| !pinned(s)).count();
            assert!(unpinned <= 4);
            assert!(resident.contains(&128));
            assert!(resident.contains(&slot));
        }
        assert!(resident.contains(&33));
        let disabled = SlotPolicy::new(256, 0);
        assert_eq!(disabled.victim(&(1..=128).collect::<Vec<_>>(), 5, pinned), None);
    }
}
