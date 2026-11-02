//! Provenance for sparse encoder images whose immutable prefix lives in a bank.
//! A complete snapshot may witness an existing ancestor at capture time. A
//! local sparse restore may witness currently valid ancestors no later than
//! its own prefix. Successful restore rebinds only those exact images, never
//! token-matched replacements or evicted images. Witnesses hold no model data.
use crate::session_lease::Lease;
use anyhow::{Result, ensure};
use std::{cell::Cell, rc::{Rc, Weak}};

pub(super) struct Binding(Rc<Cell<u64>>);
impl Binding {
    pub(super) fn new(lineage:u64)->Self {Self(Rc::new(Cell::new(lineage)))}
    pub(super) fn get(&self)->u64 {self.0.get()}
    pub(super) fn set(&self,lineage:u64) {self.0.set(lineage);}
    pub(super) const HEAP_BYTES:usize=std::mem::size_of::<Cell<u64>>()+2*std::mem::size_of::<usize>();
}

pub(crate) struct Witness {
    binding:Weak<Cell<u64>>,
    owner:Lease,
    position:u64,
}
impl Witness {
    // The target must verify that this ancestor's omitted pages are currently
    // valid before recording provenance alongside an immutable full snapshot.
    pub(super) fn capture(binding:&Binding,owner:&Lease,position:u64)->Self {
        Self{binding:Rc::downgrade(&binding.0),owner:owner.fork(),position}
    }
    pub(super) fn validate_restore(&self,owner:&Lease,position:u64)->Result<()> {
        ensure!(owner.is_current()&&owner.shares_owner(&self.owner)&&position>=self.position,
            "restored snapshot cannot rebind a foreign or later encoder ancestor");
        Ok(())
    }
    pub(super) fn rebind(&self,lineage:u64) {
        if let Some(binding)=self.binding.upgrade(){binding.set(lineage);}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_related_restore_rebinds_the_same_image_repeatedly() {
        let mut owner=Lease::new();let binding=Binding::new(1);
        let witness=Witness::capture(&binding,&owner,18688);
        for lineage in [2,3] {
            owner.begin().unwrap();owner.finish_write().unwrap();
            witness.validate_restore(&owner,18914).unwrap();witness.rebind(lineage);
            assert_eq!(binding.get(),lineage);
        }
    }
    #[test]
    fn foreign_earlier_and_busy_restores_refuse_before_rebinding() {
        let owner=Lease::new();let binding=Binding::new(7);
        let witness=Witness::capture(&binding,&owner,8192);
        assert!(witness.validate_restore(&Lease::new(),8192).is_err());
        assert!(witness.validate_restore(&owner,8191).is_err());
        owner.begin().unwrap();assert!(witness.validate_restore(&owner,8192).is_err());
        assert_eq!(binding.get(),7);
    }
    #[test]
    fn an_evicted_ancestor_cannot_rebind_a_replacement() {
        let owner=Lease::new();let binding=Binding::new(1);
        let witness=Witness::capture(&binding,&owner,8192);
        assert_eq!(Rc::strong_count(&binding.0),1);
        drop(binding);let replacement=Binding::new(9);
        witness.validate_restore(&owner,8192).unwrap();witness.rebind(10);
        assert!(witness.binding.upgrade().is_none());assert_eq!(replacement.get(),9);
    }
}
