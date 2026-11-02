//! Bounded, physically backed raw-input storage. Queued requests may borrow
//! this storage without making an allocation against future model-state growth.
use super::*;

pub(super) struct Pool { available: Mutex<Vec<Vec<u8>>>, limit: usize }
impl Pool {
    pub(super) fn new(count: usize, limit: usize) -> Result<Arc<Self>> {
        let _reservation = crate::memory::reserve_host_workspace(count * limit)?;
        let mut available = Vec::with_capacity(count);
        for _ in 0..count {
            let mut buffer = crate::memory::committed_zeroed_bytes(limit)?;
            buffer.clear();
            available.push(buffer);
        }
        Ok(Arc::new(Self { available: Mutex::new(available), limit }))
    }
    pub(super) fn take(self: &Arc<Self>, declared: usize) -> Option<Input> {
        if declared > self.limit { return None; }
        self.available.lock().unwrap().pop().map(|buffer| Input { pool: self.clone(), buffer })
    }
}
pub(super) struct Input { pool: Arc<Pool>, buffer: Vec<u8> }
impl Input {
    pub(super) async fn read(mut self, body: Body) -> Result<Bytes> {
        let mut chunks = body.into_data_stream();
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk?;
            ensure!(chunk.len() <= self.pool.limit - self.buffer.len(), "raw input exceeds reserved buffer");
            self.buffer.extend_from_slice(&chunk);
        }
        // Bytes retains the pool lease through all clones, including queued
        // SSE responses; cancellation returns it when the last owner drops.
        Ok(Bytes::from_owner(self))
    }
}
impl AsRef<[u8]> for Input { fn as_ref(&self) -> &[u8] { &self.buffer } }
impl Drop for Input {
    fn drop(&mut self) {
        let mut buffer = std::mem::take(&mut self.buffer);
        buffer.clear();
        self.pool.available.lock().unwrap().push(buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn queued_input_reuses_backing_and_returns_it_only_after_last_owner() {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        // Exercise buffer ownership independently of the production8GiB UMA
        // floor: the CPU-only build container deliberately has a6GiB limit.
        let pool = Arc::new(Pool { available: Mutex::new(vec![Vec::with_capacity(64)]), limit: 64 });
        assert!(pool.take(65).is_none());
        let input = pool.take(3).unwrap();
        let pointer = input.buffer.as_ptr();
        assert!(pool.take(1).is_none());
        let bytes = rt.block_on(input.read(Body::from("abc"))).unwrap();
        assert_eq!(bytes.as_ref(), b"abc");
        assert_eq!(bytes.as_ptr(), pointer);
        let other = bytes.clone(); drop(bytes);
        assert!(pool.take(1).is_none());
        drop(other);
        let input = pool.take(1).unwrap();
        assert_eq!(input.buffer.as_ptr(), pointer);
        assert!(rt.block_on(input.read(Body::from(vec![0u8; 65]))).is_err());
        let input = pool.take(1).unwrap();
        let future = input.read(Body::empty()); drop(future);
        assert!(pool.take(1).is_some());
    }
}
