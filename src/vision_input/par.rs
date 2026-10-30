//! Bound all preprocessing fan-out to one four-thread pool across requests.
use rayon::prelude::*;
use std::sync::OnceLock;
pub(super) fn in_pool<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .thread_name(|n| format!("image-input-{n}"))
            .build()
            .expect("image preprocessing thread pool")
    })
    .install(f)
}
pub(super) fn for_chunks_mut<T: Send>(
    data: &mut [T],
    chunk: usize,
    f: impl Fn(usize, &mut [T]) + Sync + Send,
) {
    in_pool(|| {
        data.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(i, row)| f(i, row))
    });
}
