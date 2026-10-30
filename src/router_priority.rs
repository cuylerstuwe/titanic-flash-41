//! Router-owned leases, adapted from the user's GLM Rust priority contract.
//! Epoch tombstones prevent stale release/replay; only the distributed GPU
//! owner acknowledges a revision after all ranks drain their CUDA contexts.
use anyhow::{Result, ensure};
use std::{
    collections::BTreeMap,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

const OWNERS: usize = 1024;
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct Snapshot {
    pub revision: u64,
    pub hold: bool,
}
impl Snapshot {
    fn packed(self) -> u64 {
        (self.revision << 1) | u64::from(self.hold)
    }
    fn unpack(value: u64) -> Self {
        Self {
            revision: value >> 1,
            hold: value & 1 != 0,
        }
    }
}
struct Lease {
    epoch: u64,
    foreground: bool,
    expires: Duration,
}
#[derive(Default)]
struct Gate {
    leases: BTreeMap<String, Lease>,
    current: Snapshot,
    last: Duration,
}
impl Gate {
    fn bump(&mut self) -> Result<()> {
        ensure!(
            self.current.revision < u64::MAX >> 1,
            "priority revision exhausted"
        );
        self.current.revision += 1;
        Ok(())
    }
    fn held(&self, now: Duration) -> bool {
        self.leases
            .values()
            .any(|lease| lease.foreground && lease.expires > now)
    }
    fn tick(&mut self, now: Duration) -> Result<()> {
        ensure!(now >= self.last, "priority monotonic clock went backwards");
        self.last = now;
        let hold = self.held(now);
        if hold != self.current.hold {
            self.bump()?;
            self.current.hold = hold;
        }
        Ok(())
    }
    fn update(
        &mut self,
        owner: &str,
        epoch: u64,
        foreground: bool,
        seconds: f64,
        now: Duration,
    ) -> Result<u64> {
        ensure!(
            !owner.is_empty()
                && owner.len() <= 160
                && owner
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b)),
            "invalid lease owner"
        );
        ensure!(
            seconds.is_finite() && (1.0..=300.0).contains(&seconds),
            "lease TTL must be 1..300 seconds"
        );
        if let Some(old) = self.leases.get(owner) {
            ensure!(epoch >= old.epoch, "stale lease epoch");
            ensure!(
                epoch != old.epoch || foreground == old.foreground,
                "conflicting lease epoch"
            );
        } else {
            ensure!(self.leases.len() < OWNERS, "lease owner capacity exceeded");
        }
        let expires = now
            .checked_add(Duration::from_secs_f64(seconds))
            .ok_or_else(|| anyhow::anyhow!("lease clock overflow"))?;
        self.tick(now)?;
        self.bump()?;
        self.leases.insert(
            owner.into(),
            Lease {
                epoch,
                foreground,
                expires,
            },
        );
        self.current.hold = self.held(now);
        Ok(self.current.revision)
    }
}

pub(crate) enum Receipt {
    Pending,
    Applied { hold: bool },
    Expired,
    Superseded,
}
pub(crate) struct Control {
    start: Instant,
    gate: Mutex<Gate>,
    // One atomic captures both revision and decision. The normal generation
    // path never takes the lease mutex or walks the owner map.
    current: AtomicU64,
    applied: AtomicU64,
}
impl Control {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            gate: Mutex::new(Gate::default()),
            current: AtomicU64::new(0),
            applied: AtomicU64::new(0),
        }
    }
    pub fn update(&self, owner: &str, epoch: u64, foreground: bool, seconds: f64) -> Result<u64> {
        let mut gate = self.gate.lock().unwrap();
        let revision = gate.update(owner, epoch, foreground, seconds, self.start.elapsed())?;
        self.current.store(gate.current.packed(), Ordering::Release);
        Ok(revision)
    }
    pub fn tick(&self) -> Result<()> {
        if !self.hold() {
            return Ok(());
        }
        let mut gate = self.gate.lock().unwrap();
        gate.tick(self.start.elapsed())?;
        self.current.store(gate.current.packed(), Ordering::Release);
        Ok(())
    }
    pub fn hold(&self) -> bool {
        self.current.load(Ordering::Acquire) & 1 != 0
    }
    pub fn pending(&self) -> Option<Snapshot> {
        let current = self.current.load(Ordering::Acquire);
        (current != self.applied.load(Ordering::Acquire)).then(|| Snapshot::unpack(current))
    }
    pub fn acknowledge(&self, snapshot: Snapshot) -> Result<()> {
        let applied = Snapshot::unpack(self.applied.load(Ordering::Acquire));
        let current = Snapshot::unpack(self.current.load(Ordering::Acquire));
        ensure!(
            snapshot.revision >= applied.revision && snapshot.revision <= current.revision,
            "invalid drained priority revision"
        );
        self.applied.store(snapshot.packed(), Ordering::Release);
        Ok(())
    }
    pub fn receipt(&self, owner: &str, epoch: u64, foreground: bool, revision: u64) -> Receipt {
        let gate = self.gate.lock().unwrap();
        let Some(lease) = gate.leases.get(owner) else {
            return Receipt::Superseded;
        };
        if lease.epoch != epoch || lease.foreground != foreground {
            return Receipt::Superseded;
        }
        if foreground && lease.expires <= self.start.elapsed() {
            return Receipt::Expired;
        }
        let applied = Snapshot::unpack(self.applied.load(Ordering::Acquire));
        if applied.revision >= revision && (!foreground || applied.hold) {
            Receipt::Applied { hold: applied.hold }
        } else {
            Receipt::Pending
        }
    }
    pub fn metadata(&self) -> serde_json::Value {
        let current = Snapshot::unpack(self.current.load(Ordering::Acquire));
        let applied = Snapshot::unpack(self.applied.load(Ordering::Acquire));
        serde_json::json!({"hold_requested":current.hold,"revision":current.revision,"applied_revision":applied.revision,"applied_hold":applied.hold})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(milliseconds: u64) -> Duration {
        Duration::from_millis(milliseconds)
    }
    #[test]
    fn independent_owners_expiry_and_stale_releases_preserve_holds() {
        let mut gate = Gate::default();
        gate.update("a", 4, true, 1.0, t(0)).unwrap();
        gate.update("b", 7, true, 2.0, t(0)).unwrap();
        gate.tick(t(1000)).unwrap();
        assert!(gate.current.hold);
        assert!(gate.update("a", 3, false, 1.0, t(1000)).is_err());
        assert!(gate.update("a", 4, false, 1.0, t(1000)).is_err());
        gate.update("a", 5, false, 1.0, t(1000)).unwrap();
        assert!(gate.current.hold);
        let before = gate.current.revision;
        gate.tick(t(2000)).unwrap();
        assert!(!gate.current.hold && gate.current.revision > before);
        assert!(gate.update("b", 6, true, 1.0, t(2000)).is_err());
        gate.update("b", 7, true, 1.0, t(2000)).unwrap();
        assert!(gate.current.hold);
    }
    #[test]
    fn receipt_waits_for_drain_and_newer_update_cannot_be_acknowledged_early() {
        let control = Control::new();
        let first = control.update("router", 1, true, 30.0).unwrap();
        let captured = control.pending().unwrap();
        assert!(matches!(
            control.receipt("router", 1, true, first),
            Receipt::Pending
        ));
        let second = control.update("router", 2, false, 30.0).unwrap();
        control.acknowledge(captured).unwrap();
        assert!(matches!(
            control.receipt("router", 1, true, first),
            Receipt::Superseded
        ));
        assert!(matches!(
            control.receipt("router", 2, false, second),
            Receipt::Pending
        ));
        control.acknowledge(control.pending().unwrap()).unwrap();
        assert!(matches!(
            control.receipt("router", 2, false, second),
            Receipt::Applied { hold: false }
        ));
        assert!(control.pending().is_none());
        assert!(control.acknowledge(captured).is_err());
    }
    #[test]
    fn invalid_or_unbounded_controls_do_not_change_state() {
        let mut gate = Gate::default();
        for seconds in [f64::NAN, f64::INFINITY, 0.0, 0.99, 300.01] {
            assert!(gate.update("a", 1, true, seconds, t(0)).is_err());
        }
        for owner in ["", "with space", "line\nbreak", "🦀"] {
            assert!(gate.update(owner, 1, true, 1.0, t(0)).is_err());
        }
        assert_eq!(gate.current, Snapshot::default());
        for i in 0..OWNERS {
            gate.update(&i.to_string(), 1, false, 1.0, t(0)).unwrap();
        }
        assert!(gate.update("extra", 1, true, 1.0, t(0)).is_err());
        gate.update("0", 2, true, 1.0, t(1000)).unwrap();
        assert!(gate.current.hold);
        assert!(gate.tick(t(999)).is_err());
    }
}
