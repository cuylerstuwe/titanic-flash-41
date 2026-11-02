//! Physical image accounting for future two-owner growth. This is not a
//! second allocation pool: caches may borrow the same existing image budget.
use anyhow::{Context, Result};
use std::sync::Mutex;

static IMAGE_BYTES: Mutex<usize> = Mutex::new(0);

pub(crate) struct ImageStorageCharge<'a> {
    ledger: &'a Mutex<usize>,
    bytes: usize,
}
impl Drop for ImageStorageCharge<'_> {
    fn drop(&mut self) {
        let mut bytes = self.ledger.lock().unwrap();
        *bytes = bytes.checked_sub(self.bytes).expect("image storage accounting underflow");
    }
}
fn allocate_tracked<T>(ledger: &Mutex<usize>, bytes: usize, allocate: impl FnOnce() -> Result<T>)
    -> Result<(ImageStorageCharge<'_>, T)> {
    let mut held = ledger.lock().unwrap();
    let next = held.checked_add(bytes).context("image storage accounting overflow")?;
    // Serialize the CUDA allocation and counter publication with an upload's
    // physical-memory sample. Otherwise it could combine pre-allocation free
    // memory with post-allocation accounting and over-admit a queued body.
    let storage = allocate()?;
    *held = next;
    Ok((ImageStorageCharge { ledger, bytes }, storage))
}
pub(crate) fn allocate_image_storage<T>(bytes: usize, allocate: impl FnOnce() -> Result<T>)
    -> Result<(ImageStorageCharge<'static>, T)> {
    allocate_tracked(&IMAGE_BYTES, bytes, allocate)
}
pub(crate) fn with_unallocated_capacity<T>(budget: usize, sample_and_reserve: impl FnOnce(usize) -> Result<T>) -> Result<T> {
    let held = IMAGE_BYTES.lock().unwrap();
    sample_and_reserve(budget.saturating_sub(*held))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_growth_cannot_be_sampled_across_an_unpublished_allocation() {
        let ledger = Mutex::new(0);
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let ledger_ref = &ledger;
            let allocation = scope.spawn(move || allocate_tracked(ledger_ref, 128, || {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(())
            }));
            entered_rx.recv().unwrap();
            assert!(ledger.try_lock().is_err(), "queued upload could sample halfway through allocation");
            release_tx.send(()).unwrap();
            let (charge, ()) = allocation.join().unwrap().unwrap();
            assert_eq!(*ledger.lock().unwrap(), 128);
            let failed: Result<(ImageStorageCharge<'_>, ())> = allocate_tracked(&ledger, 256, || anyhow::bail!("refused"));
            assert!(failed.is_err());
            assert_eq!(*ledger.lock().unwrap(), 128, "failed allocation must not lend growth capacity");
            drop(charge);
            assert_eq!(*ledger.lock().unwrap(), 0, "released storage again reserves its future growth");
        });
    }
}
