//! Wake input admission after a coordinated optional-cache reclamation.
//! The generation remains authoritative. Signals only shorten the old wait;
//! cancellation/failure checks retain their previous ten-millisecond bound.
use std::{sync::{Condvar,Mutex,atomic::{AtomicU64,Ordering}},time::{Duration,Instant}};
use tokio::sync::Notify;

const CANCEL_CHECK:Duration=Duration::from_millis(10);

#[derive(Default)]
pub(super) struct Ack { lock:Mutex<()>, sync:Condvar, asynchronous:Notify }
impl Ack {
    pub(super) fn publish(&self,generation:&AtomicU64) {
        // Pair the generation update with synchronous registration. An ack
        // between its predicate check and sleeping cannot be lost.
        let _lock=self.lock.lock().unwrap_or_else(|e|e.into_inner());
        generation.fetch_add(1,Ordering::Release);
        self.sync.notify_all();self.asynchronous.notify_waiters();
    }
    pub(super) async fn wait_async(&self,generation:&AtomicU64,before:u64,
        ready:impl Fn()->bool,timeout:Duration) {
        let deadline=Instant::now()+timeout;
        loop {
            let notified=self.asynchronous.notified();tokio::pin!(notified);
            notified.as_mut().enable();
            let remaining=deadline.saturating_duration_since(Instant::now());
            if generation.load(Ordering::Acquire)!=before||!ready()||remaining.is_zero(){return;}
            let _=tokio::time::timeout(remaining.min(CANCEL_CHECK),notified).await;
        }
    }
    pub(super) fn wait_sync(&self,generation:&AtomicU64,before:u64,
        ready:impl Fn()->bool,timeout:Duration) {
        let deadline=Instant::now()+timeout;
        let mut lock=self.lock.lock().unwrap_or_else(|e|e.into_inner());
        loop {
            let remaining=deadline.saturating_duration_since(Instant::now());
            if generation.load(Ordering::Acquire)!=before||!ready()||remaining.is_zero(){return;}
            lock=self.sync.wait_timeout(lock,remaining.min(CANCEL_CHECK))
                .unwrap_or_else(|e|e.into_inner()).0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc,atomic::AtomicBool};
    #[test]
    fn acknowledgements_before_and_during_async_and_sync_waits_are_not_lost() {
        let ack=Arc::new(Ack::default());let generation=Arc::new(AtomicU64::new(0));
        let rt=tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        ack.publish(&generation);
        rt.block_on(ack.wait_async(&generation,0,||true,Duration::from_secs(1)));
        ack.wait_sync(&generation,0,||true,Duration::from_secs(1));
        let sync={let ack=ack.clone();let generation=generation.clone();std::thread::spawn(move||{
            ack.wait_sync(&generation,1,||true,Duration::from_secs(1));
            assert_eq!(generation.load(Ordering::Acquire),2);
        })};
        rt.block_on(async {
            let publish=async {tokio::task::yield_now().await;ack.publish(&generation);};
            tokio::join!(ack.wait_async(&generation,1,||true,Duration::from_secs(1)),
                ack.wait_async(&generation,1,||true,Duration::from_secs(1)),publish);
        });sync.join().unwrap();assert_eq!(generation.load(Ordering::Acquire),2);
    }
    #[test]
    fn missing_ack_and_cancellation_keep_waits_bounded_without_changing_generation() {
        let ack=Ack::default();let generation=AtomicU64::new(7);
        let rt=tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(ack.wait_async(&generation,7,||true,Duration::from_millis(1)));
        ack.wait_sync(&generation,7,||true,Duration::from_millis(1));
        let ready=AtomicBool::new(true);
        rt.block_on(async {
            let cancel=async {tokio::task::yield_now().await;ready.store(false,Ordering::Release);};
            tokio::join!(ack.wait_async(&generation,7,||ready.load(Ordering::Acquire),Duration::from_secs(1)),cancel);
        });
        ack.wait_sync(&generation,7,||false,Duration::from_secs(1));
        assert_eq!(generation.load(Ordering::Acquire),7);
    }
}
