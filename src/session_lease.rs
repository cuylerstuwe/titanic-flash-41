//! Single-writer revision control for execution plans sharing GPU causal state.
//! Sharing is a profile handoff, never an independent branch or a state copy.
use anyhow::{Result, ensure};
use std::{cell::Cell, rc::Rc};
struct Shared {
    revision: Cell<u64>,
    poisoned: Cell<bool>,
}
pub(crate) struct Lease {
    shared: Rc<Shared>,
    observed: u64,
}
impl Lease {
    pub fn new() -> Self {
        Self {
            shared: Rc::new(Shared {
                revision: Cell::new(1),
                poisoned: Cell::new(false),
            }),
            observed: 1,
        }
    }
    pub fn fork(&self) -> Self {
        Self {
            shared: self.shared.clone(),
            observed: 0,
        }
    }
    pub fn shares_owner(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.shared, &other.shared)
    }
    pub fn is_current(&self) -> bool {
        !self.shared.poisoned.get() && self.observed == self.shared.revision.get()
    }
    pub fn shared_idle(&self) -> bool { !self.shared.poisoned.get() }
    /// Revision check only: a caller also needs its private completed pending
    /// transaction. Other adopted profiles can observe the same busy revision.
    pub fn is_busy_revision(&self) -> bool {
        self.shared.poisoned.get() && self.observed == self.shared.revision.get()
    }
    #[cfg(test)]
    pub fn poisoned(&self) -> bool {
        self.shared.poisoned.get()
    }
    pub fn adopt(&mut self, previous: &Self) -> Result<()> {
        ensure!(
            self.shares_owner(previous) && previous.is_current(),
            "profile handoff requires current healthy shared-state authority"
        );
        self.observed = previous.observed;
        Ok(())
    }
    pub fn begin(&self) -> Result<()> {
        ensure!(
            self.is_current() && self.observed < u64::MAX,
            "stale or poisoned execution profile"
        );
        self.shared.poisoned.set(true);
        Ok(())
    }
    pub fn finish_read(&self) -> Result<()> {
        ensure!(
            self.shared.poisoned.get() && self.observed == self.shared.revision.get(),
            "profile completion lost state authority"
        );
        self.shared.poisoned.set(false);
        Ok(())
    }
    pub fn finish_write(&mut self) -> Result<()> {
        ensure!(
            self.shared.poisoned.get() && self.observed == self.shared.revision.get(),
            "profile commit lost state authority"
        );
        let next = self
            .observed
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("profile revision overflow"))?;
        self.shared.revision.set(next);
        self.observed = next;
        self.shared.poisoned.set(false);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn handoff_invalidates_stale_writers_and_refuses_foreign_owners() {
        let mut prefill = Lease::new();
        let mut decode = prefill.fork();
        assert!(!decode.is_current());
        assert!(decode.begin().is_err());
        decode.adopt(&prefill).unwrap();
        decode.begin().unwrap();
        assert!(prefill.begin().is_err());
        decode.finish_write().unwrap();
        assert!(!prefill.is_current());
        assert!(prefill.begin().is_err());
        prefill.adopt(&decode).unwrap();
        assert!(prefill.is_current());
        assert!(prefill.adopt(&Lease::new()).is_err());
    }
    #[test]
    fn unfinished_execution_poison_is_shared_and_read_preserves_revision() {
        let owner = Lease::new();
        let mut next = owner.fork();
        next.adopt(&owner).unwrap();
        owner.begin().unwrap();
        assert!(next.poisoned());
        assert!(next.adopt(&owner).is_err());
        owner.finish_read().unwrap();
        assert!(next.is_current());
        next.begin().unwrap();
        assert!(!owner.is_current());
        // An error path deliberately does not finish: every profile fails closed.
        assert!(owner.begin().is_err());
        assert!(next.begin().is_err());
    }
}
