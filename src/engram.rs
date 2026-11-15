//! Exact rank-local Engram backing and bounded set-associative row cache.
//! The existing packed checkpoint shards are read-only inputs, not engine code.
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use std::{
    fs::File,
    os::unix::fs::FileExt,
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

pub const ROW_BYTES: usize = 264;
/// CANADA_SCRATCH_HOST_STAGES=1 (diagnostic): totals of the row reads since the last take: paired lookups, ns in
/// them, read batches, rows read, ns in the batches, ns of those spent in the submission call.
pub(crate) static IO_STATS: [std::sync::atomic::AtomicU64; 6] = [
    std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];
pub(crate) fn take_io_stats() -> [u64; 6] {
    std::array::from_fn(|i| IO_STATS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}
/// CANADA_SCRATCH_ENGRAM_MISS_THREADS=N (2..=64; default 0 = the bounded io_uring batches): storage reads of a
/// lookup's cache misses run on N scoped threads. First-seen text misses most of a chunk's rows (about six owned
/// rows a token and table on each rank), and those reads sit in front of layer 1.
pub(crate) fn miss_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| validate_miss_threads().unwrap_or(0))
}
/// Fills each listed row of `out` (`rows` = (id, row position), positions ascending) with `read(id, row)` on up to
/// `threads` scoped threads. Every thread owns one contiguous range of positions, so no row is shared.
fn read_rows_threaded(rows:&[(u64,usize)],out:&mut[u8],threads:usize,
    read:&(impl Fn(u64,&mut [u8])->std::io::Result<()>+Sync))->Result<()> {
    ensure!(threads>0 && rows.windows(2).all(|w|w[0].1<w[1].1)
        && rows.last().is_none_or(|last|(last.1+1)*ROW_BYTES<=out.len()),"Engram threaded read geometry");
    if rows.is_empty() {return Ok(());}
    let per=rows.len().div_ceil(threads);
    std::thread::scope(|scope|->Result<()> {
        let mut jobs=Vec::new();
        let (mut rest,mut base)=(out,0usize);
        for slice in rows.chunks(per) {
            let (low,high)=(slice[0].1,slice[slice.len()-1].1+1);
            let (_,tail)=rest.split_at_mut((low-base)*ROW_BYTES);
            let (mine,tail)=tail.split_at_mut((high-low)*ROW_BYTES);
            rest=tail;base=high;
            jobs.push(scope.spawn(move||->std::io::Result<()> {
                for &(id,position) in slice {
                    read(id,&mut mine[(position-low)*ROW_BYTES..(position-low+1)*ROW_BYTES])?;
                }
                Ok(())
            }));
        }
        for job in jobs {
            job.join().map_err(|_|anyhow::anyhow!("Engram miss reader panicked"))?.context("Engram miss read")?;
        }
        Ok(())
    })
}
/// CANADA_SCRATCH_ENGRAM_DIRECT=1 (default 0): a row that misses the row cache is read with O_DIRECT (its one or two
/// 4 KiB blocks, crate::engram_io) instead of through the page cache. Same bytes. A buffered miss allocates a
/// page-cache page; with the boxes near their memory floor that allocation waits for reclaim, and the step waits
/// with it (Nov 14: a first-seen answer stepped 2-5 ms longer with four long conversations resident). The
/// threaded reader (CANADA_SCRATCH_ENGRAM_MISS_THREADS) stays buffered.
pub(crate) fn direct_reads() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| validate_direct_reads().unwrap_or(false))
}
pub(crate) fn validate_direct_reads() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_ENGRAM_DIRECT") {
        Ok(v) => { ensure!(v == "0" || v == "1", "Engram direct reads: 0 or 1"); Ok(v == "1") }
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(e) => Err(e.into()),
    }
}
pub(crate) fn validate_miss_threads() -> Result<usize> {
    match std::env::var("CANADA_SCRATCH_ENGRAM_MISS_THREADS") {
        Ok(v) => { let n: usize = v.parse()?; ensure!(n == 0 || (2..=64).contains(&n), "Engram miss threads: 0 or 2..=64"); Ok(n) }
        Err(std::env::VarError::NotPresent) => Ok(0),
        Err(e) => Err(e.into()),
    }
}
const HEADER_BYTES: u64 = 4096;
const MAGIC: u64 = 0x31344e4531565344;
const EMPTY: u64 = u64::MAX;
const WAYS: usize = 16;

/// Optional immutable-row cache, not model state. Full physical backing of
/// the former two 2-GiB caches left insufficient image headroom on GB10.
/// Keep a bounded explicit control for measuring the cache-hit tradeoff.
pub(crate) fn target_cache_bytes() -> Result<usize> {
    let mib = match std::env::var("CANADA_SCRATCH_ENGRAM_CACHE_MIB") {
        Err(std::env::VarError::NotPresent) => 1536,
        Ok(v) if v == "1536" => 1536,
        Ok(v) if v == "2048" => 2048,
        _ => anyhow::bail!("Engram cache must be 1536 or 2048 MiB per layer"),
    };
    Ok(mib * 1024 * 1024)
}

pub struct PackedShard {
    file: File,
    /// The same file opened with O_DIRECT (crate::engram::direct_reads), for cache-miss reads through io_uring.
    direct: Option<File>,
    pub layer: u64,
    pub start: u64,
    pub end: u64,
    pub global_rows: u64,
}
impl PackedShard {
    pub fn open(
        path: &Path,
        layer: u64,
        rank: usize,
        world: usize,
        global_rows: u64,
    ) -> Result<Self> {
        ensure!(world == 4 && rank < world, "Engram requires four ranks");
        let f = File::open(path)?;
        let mut header = [0u8; 48];
        f.read_exact_at(&mut header, 0)?;
        let v: Vec<_> = header
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let start = global_rows * rank as u64 / world as u64;
        let end = global_rows * (rank + 1) as u64 / world as u64;
        ensure!(
            v == [MAGIC, layer, start, end, global_rows, ROW_BYTES as u64],
            "packed Engram header mismatch"
        );
        ensure!(
            f.metadata()?.len() == HEADER_BYTES + (end - start) * ROW_BYTES as u64,
            "packed Engram truncated or wrong size"
        );
        #[cfg(target_os = "linux")]
        let direct = if direct_reads() {
            use std::os::unix::fs::OpenOptionsExt;
            Some(std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path)
                .context("Engram direct reads need O_DIRECT on the shard's file system")?)
        } else { None };
        #[cfg(not(target_os = "linux"))]
        let direct = None;
        Ok(Self {
            file: f,
            direct,
            layer,
            start,
            end,
            global_rows,
        })
    }
    /// The handle for io_uring miss reads and whether it is the O_DIRECT one.
    #[cfg(target_os = "linux")]
    fn io(&self) -> (&File, bool) {
        match &self.direct { Some(file) => (file, true), None => (&self.file, false) }
    }
    fn read(&self, id: u64, out: &mut [u8; ROW_BYTES]) -> Result<bool> {
        ensure!(id < self.global_rows, "Engram hash outside table");
        if id < self.start || id >= self.end {
            out.fill(0);
            return Ok(false);
        }
        self.file
            .read_exact_at(out, HEADER_BYTES + (id - self.start) * ROW_BYTES as u64)?;
        Ok(true)
    }
}

struct Segment {
    tags: Vec<u64>,
    rows: Vec<u8>,
    hands: Vec<u8>,
}

#[cfg(target_os = "linux")]
struct Misses {
    rows: Vec<(u64,usize)>,
    duplicates: Vec<(usize,usize)>,
    hits: u64,
    nonowned: u64,
}
impl Segment {
    fn bucket(&self, id: u64) -> usize {
        // Full row id remains the tag; hash collisions are always misses.
        let hash = id.wrapping_mul(0x9e3779b97f4a7c15);
        (hash ^ (hash >> 32)) as usize % self.hands.len()
    }
    fn find(&self, id: u64, out: &mut [u8; ROW_BYTES]) -> bool {
        let b = self.bucket(id) * WAYS;
        if let Some(w) = (0..WAYS).find(|w| self.tags[b + w] == id) {
            let i = (b + w) * ROW_BYTES;
            out.copy_from_slice(&self.rows[i..i + ROW_BYTES]);
            true
        } else {
            false
        }
    }
    fn insert(&mut self, id: u64, row: &[u8; ROW_BYTES]) {
        let b = self.bucket(id);
        let start = b * WAYS;
        let w = (0..WAYS)
            .find(|w| self.tags[start + w] == id)
            .or_else(|| (0..WAYS).find(|w| self.tags[start + w] == EMPTY))
            .unwrap_or_else(|| {
                let w = self.hands[b] as usize;
                self.hands[b] = ((w + 1) % WAYS) as u8;
                w
            });
        let slot = start + w;
        let offset = slot * ROW_BYTES;
        self.rows[offset..offset + ROW_BYTES].copy_from_slice(row);
        // Tag publication follows the complete row write under the same mutex.
        self.tags[slot] = id;
    }
}

pub struct RowStore {
    #[cfg(target_os = "linux")]
    io_reader: Option<Mutex<crate::engram_io::Reader>>,
    io_pool: Option<rayon::ThreadPool>,
    shard: PackedShard,
    segments: Vec<Mutex<Segment>>,
    pub resident_bytes: usize,
    hits: AtomicU64,
    misses: AtomicU64,
    nonowned: AtomicU64,
    prefetch_calls: AtomicU64,
    prefetched_rows: AtomicU64,
    /// Rows of a read-ahead that was submitted on this store's reader and is not yet in the cache, in submission
    /// order (`read_ahead_submit`); empty otherwise. Every lookup publishes them before it looks.
    #[cfg(target_os = "linux")]
    read_ahead: Mutex<Vec<u64>>,
    read_ahead_rows: AtomicU64,
}
impl RowStore {
    /// config::engram_anchor_prefetch: request, WITHOUT waiting, the rows of `ids` that this shard owns and the cache
    /// does not hold. They are read while the caller does other work (the drafter's pass) and are published into
    /// the cache by the next lookup on this store, which stays authoritative: a row that failed or was evicted
    /// meanwhile is simply read again there. Immutable row bytes; a speed hint only.
    #[cfg(target_os = "linux")]
    pub(crate) fn read_ahead_submit(&self, ids: &[u64]) -> Result<()> {
        let Some(reader) = &self.io_reader else { return Ok(()); };
        ensure!(ids.len() <= crate::engram_io::ENTRIES && ids.iter().all(|&id| id < self.shard.global_rows),
            "Engram read-ahead extent/domain");
        let mut reader = reader.lock().map_err(|_| anyhow::anyhow!("Engram I/O reader poisoned"))?;
        self.publish_read_ahead(&mut reader)?;
        let mut want: Vec<u64> = Vec::with_capacity(ids.len());
        for &id in ids {
            if id < self.shard.start || id >= self.shard.end || want.contains(&id) { continue; }
            let cache = self.segments[self.segment(id)].lock().map_err(|_| anyhow::anyhow!("Engram cache poisoned"))?;
            let start = cache.bucket(id) * WAYS;
            if !cache.tags[start..start + WAYS].contains(&id) { want.push(id); }
        }
        if want.is_empty() { return Ok(()); }
        let offsets: Vec<u64> = want.iter().map(|id| HEADER_BYTES + (id - self.shard.start) * ROW_BYTES as u64).collect();
        let (file, direct) = self.shard.io();
        reader.submit(file, direct, &offsets)?;
        *self.read_ahead.lock().map_err(|_| anyhow::anyhow!("Engram read-ahead list poisoned"))? = want;
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    pub(crate) fn read_ahead_submit(&self, _ids: &[u64]) -> Result<()> { Ok(()) }
    /// Publish a submitted read-ahead (waiting for reads still in flight). The caller holds the reader.
    #[cfg(target_os = "linux")]
    fn publish_read_ahead(&self, reader: &mut crate::engram_io::Reader) -> Result<()> {
        let ids = std::mem::take(&mut *self.read_ahead.lock().map_err(|_| anyhow::anyhow!("Engram read-ahead list poisoned"))?);
        ensure!(ids.is_empty() != reader.has_submitted(), "Engram read-ahead list and reader disagree");
        if ids.is_empty() { return Ok(()); }
        if let Some(rows) = reader.finish_submitted()? {
            ensure!(rows.len() == ids.len() * ROW_BYTES, "Engram read-ahead geometry");
            for (&id, row) in ids.iter().zip(rows.chunks_exact(ROW_BYTES)) {
                self.segments[self.segment(id)].lock().map_err(|_| anyhow::anyhow!("Engram cache poisoned"))?
                    .insert(id, row.try_into().unwrap());
            }
            self.read_ahead_rows.fetch_add(ids.len() as u64, Ordering::Relaxed);
        }
        Ok(())
    }
    /// Before any lookup on this store: a pending read-ahead goes into the cache first (and frees the reader).
    #[cfg(target_os = "linux")]
    fn drain_read_ahead(&self) -> Result<()> {
        let Some(reader) = &self.io_reader else { return Ok(()); };
        if self.read_ahead.lock().map_err(|_| anyhow::anyhow!("Engram read-ahead list poisoned"))?.is_empty() { return Ok(()); }
        let mut reader = reader.lock().map_err(|_| anyhow::anyhow!("Engram I/O reader poisoned"))?;
        self.publish_read_ahead(&mut reader)
    }
    /// Scheduling hint only: inspect immutable-row cache tags without copying
    /// bytes, changing replacement state, or issuing storage reads. A racing
    /// prefetch/eviction may change this hint; the subsequent lookup remains
    /// authoritative and always retrieves the exact row bytes.
    pub(crate) fn misses_any(&self, ids: impl IntoIterator<Item = u64>) -> Result<bool> {
        for id in ids {
            ensure!(id < self.shard.global_rows, "Engram hint outside table");
            if id < self.shard.start || id >= self.shard.end { continue; }
            let cache = self.segments[self.segment(id)].lock()
                .map_err(|_| anyhow::anyhow!("Engram cache poisoned"))?;
            let start = cache.bucket(id) * WAYS;
            if !cache.tags[start..start + WAYS].contains(&id) { return Ok(true); }
        }
        Ok(false)
    }
    pub fn interval(&self) -> (u64, u64, u64) {
        (self.shard.start, self.shard.end, self.shard.global_rows)
    }
    pub fn new(shard: PackedShard, budget: usize) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            let required = budget.checked_add(crate::memory::floor_bytes()).context("Engram cache admission overflow")?;
            ensure!(crate::memory::Available::gb10(usize::MAX, usize::MAX)?.admitted_available >= required,
                "Engram row cache cannot retain the 8 GiB UMA reserve");
        }
        let pool = std::env::var("CANADA_SCRATCH_ENGRAM_POOL").as_deref() == Ok("1");
        let uring = std::env::var("CANADA_SCRATCH_ENGRAM_URING").as_deref() == Ok("1");
        ensure!(!(pool && uring), "choose one Engram I/O worker mode");
        #[cfg(not(target_os = "linux"))]
        ensure!(!uring, "Engram io_uring requires Linux");
        #[allow(unused_mut)]
        let mut store = Self::new_with_pool(shard, budget, pool)?;
        #[cfg(target_os = "linux")]
        if uring {
            store.io_reader = Some(Mutex::new(crate::engram_io::Reader::new()?));
            store.check_direct_reads()?;
            store.check_read_ahead()?;
        }
        Ok(store)
    }
    /// config::engram_anchor_prefetch: before serving, eight rows spread over this shard go through the read-ahead
    /// path (submit without waiting, publish at the drain) and must be in the cache with the bytes a plain
    /// positional read returns.
    #[cfg(target_os = "linux")]
    fn check_read_ahead(&self) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let (start, end) = (self.shard.start, self.shard.end);
        if !crate::config::engram_anchor_prefetch() || end == start { return Ok(()); }
        let span = end - start;
        let ids: Vec<u64> = (0..8u64).map(|i| start + (span - 1) * i / 7).collect();
        self.read_ahead_submit(&ids)?;
        self.drain_read_ahead()?;
        for id in ids {
            let (mut got, mut expected) = ([0u8; ROW_BYTES], [0u8; ROW_BYTES]);
            ensure!(self.segments[self.segment(id)].lock().map_err(|_| anyhow::anyhow!("Engram cache poisoned"))?.find(id, &mut got),
                "Engram read-ahead did not publish a row");
            self.shard.file.read_exact_at(&mut expected, HEADER_BYTES + (id - start) * ROW_BYTES as u64)?;
            ensure!(got == expected, "Engram read-ahead differs from the positional read");
        }
        Ok(())
    }
    /// With direct reads on: eight rows spread over the shard read both ways must be the same bytes.
    #[cfg(target_os = "linux")]
    fn check_direct_reads(&self) -> Result<()> {
        let (file, direct) = self.shard.io();
        let (start, end) = (self.shard.start, self.shard.end);
        if !direct || end == start { return Ok(()); }
        let span = end - start;
        let ids: Vec<u64> = (0..8u64).map(|i| start + (span - 1) * i / 7).collect();
        let offsets: Vec<u64> = ids.iter().map(|id| HEADER_BYTES + (id - start) * ROW_BYTES as u64).collect();
        let mut reader = self.io_reader.as_ref().context("direct Engram reads need the io_uring reader")?
            .lock().map_err(|_| anyhow::anyhow!("Engram I/O reader poisoned"))?;
        let rows = reader.read(file, true, &offsets)?;
        for (i, offset) in offsets.iter().enumerate() {
            let mut expected = [0u8; ROW_BYTES];
            self.shard.file.read_exact_at(&mut expected, *offset)?;
            ensure!(rows[i * ROW_BYTES..(i + 1) * ROW_BYTES] == expected, "Engram direct read differs from the buffered read");
        }
        Ok(())
    }
    fn new_with_pool(shard: PackedShard, budget: usize, pool: bool) -> Result<Self> {
        let stripes = 16;
        let cost = WAYS * (ROW_BYTES + size_of::<u64>()) + 1;
        let buckets = budget / (stripes * cost);
        ensure!(buckets > 0, "Engram cache budget too small");
        let mut segments = Vec::with_capacity(stripes);
        for _ in 0..stripes {
            let slots = buckets * WAYS;
            segments.push(Mutex::new(Segment {
                tags: vec![EMPTY; slots],
                rows: crate::memory::committed_zeroed_bytes(slots * ROW_BYTES)?,
                hands: crate::memory::committed_zeroed_bytes(buckets)?,
            }));
        }
        let resident_bytes = stripes * buckets * cost;
        ensure!(resident_bytes <= budget, "cache budget arithmetic");
        Ok(Self {
            #[cfg(target_os = "linux")]
            io_reader: None,
            io_pool: if pool {
                Some(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(8)
                        .stack_size(512 * 1024)
                        .thread_name(|n| format!("engram-io-{n}"))
                        .build()
                        .context("Engram I/O pool creation")?,
                )
            } else {
                None
            },
            shard,
            segments,
            resident_bytes,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            nonowned: AtomicU64::new(0),
            prefetch_calls: AtomicU64::new(0),
            prefetched_rows: AtomicU64::new(0),
            #[cfg(target_os = "linux")]
            read_ahead: Mutex::new(Vec::new()),
            read_ahead_rows: AtomicU64::new(0),
        })
    }
    /// Rows read from storage so far (diagnostic).
    pub(crate) fn misses(&self) -> u64 { self.misses.load(Ordering::Relaxed) }
    fn segment(&self, id: u64) -> usize {
        let a = id.wrapping_mul(0xd6e8feb86659fd93);
        ((a ^ (a >> 32)) as usize) % self.segments.len()
    }
    pub fn lookup(&self, id: u64, out: &mut [u8; ROW_BYTES]) -> Result<()> {
        ensure!(id < self.shard.global_rows, "Engram hash outside table");
        if id < self.shard.start || id >= self.shard.end {
            out.fill(0);
            self.nonowned.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        let segment = &self.segments[self.segment(id)];
        {
            let cache = segment
                .lock()
                .map_err(|_| anyhow::anyhow!("Engram cache poisoned"))?;
            if cache.find(id, out) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
        }
        self.shard.read(id, out)?; // No cache lock across I/O.
        segment
            .lock()
            .map_err(|_| anyhow::anyhow!("Engram cache poisoned"))?
            .insert(id, out);
        self.misses.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    pub fn lookup_into(&self, ids: &[u64], out: &mut [u8], threads: usize) -> Result<()> {
        ensure!(
            out.len()
                == ids
                    .len()
                    .checked_mul(ROW_BYTES)
                    .context("Engram batch overflow")?,
            "Engram staging capacity"
        );
        ensure!(
            threads > 0 && threads <= 32,
            "bounded Engram I/O concurrency required"
        );
        if ids.is_empty() {
            return Ok(());
        }
        ensure!(
            ids.iter().all(|id| *id < self.shard.global_rows),
            "Engram hash outside table"
        );
        #[cfg(target_os = "linux")]
        if let Some(reader) = &self.io_reader {
            return self.lookup_batched(ids, out, reader);
        }
        // Decode/verify frequently hits every row. Copy resident raw bytes on
        // the caller rather than creating and joining workers for zero I/O.
        // Publish hit counters only when the entire fast path succeeds; a
        // partial miss falls through to the existing complete lookup path.
        let mut hits = 0u64;
        let mut nonowned = 0u64;
        let mut complete = true;
        for (id, row) in ids.iter().copied().zip(out.chunks_exact_mut(ROW_BYTES)) {
            if id < self.shard.start || id >= self.shard.end {
                row.fill(0);
                nonowned += 1;
                continue;
            }
            let cache = self.segments[self.segment(id)]
                .lock()
                .map_err(|_| anyhow::anyhow!("Engram cache poisoned"))?;
            if cache.find(id, row.try_into().unwrap()) {
                hits += 1;
            } else {
                complete = false;
                break;
            }
        }
        if complete {
            self.hits.fetch_add(hits, Ordering::Relaxed);
            self.nonowned.fetch_add(nonowned, Ordering::Relaxed);
            return Ok(());
        }
        let threads = threads.min(ids.len());
        let chunk = ids.len().div_ceil(threads);
        if let Some(pool) = &self.io_pool {
            // Borrowed row slices cannot outlive this call. No CUDA context or
            // numerical model operation enters the persistent I/O workers.
            return pool.install(|| {
                ids.par_chunks(chunk)
                    .zip(out.par_chunks_mut(chunk * ROW_BYTES))
                    .try_for_each(|(ids, out)| -> Result<()> {
                        for (id, row) in ids.iter().zip(out.chunks_exact_mut(ROW_BYTES)) {
                            self.lookup(*id, row.try_into().unwrap())?;
                        }
                        Ok(())
                    })
            });
        }
        std::thread::scope(|scope| -> Result<()> {
            let mut jobs = Vec::new();
            for (ids, out) in ids.chunks(chunk).zip(out.chunks_mut(chunk * ROW_BYTES)) {
                jobs.push(scope.spawn(move || -> Result<()> {
                    for (id, row) in ids.iter().zip(out.chunks_exact_mut(ROW_BYTES)) {
                        self.lookup(*id, row.try_into().unwrap())?;
                    }
                    Ok(())
                }));
            }
            for job in jobs {
                job.join()
                    .map_err(|_| anyhow::anyhow!("Engram I/O worker panicked"))??;
            }
            Ok(())
        })
    }
    /// Warm exact immutable row bytes only. At most one maximum-width prefill chunk is
    /// admitted; scratch stays bounded to256 (control) or4096 packed rows, independent of length.
    /// Cache evictions can lose a speed opportunity but cannot alter row data.
    pub(crate) fn prefetch(&self, ids: &[u64]) -> Result<()> {
        ensure!(
            ids.len() <= crate::config::MAX_WIDE_ROWS * 24 && ids.iter().all(|id| *id < self.shard.global_rows),
            "Engram prefetch row domain/capacity"
        );
        let batch_rows=match std::env::var("CANADA_SCRATCH_ENGRAM_PREFETCH_BATCH_ROWS") {
            Ok(value)=>{let rows:usize=value.parse()?;ensure!([256,4096].contains(&rows),"unadmitted Engram prefetch batch");rows},
            Err(std::env::VarError::NotPresent)=>256,
            Err(error)=>return Err(error.into()),
        };
        let mut scratch = vec![0u8; ids.len().min(batch_rows) * ROW_BYTES];
        for batch in ids.chunks(batch_rows) {
            self.lookup_into(batch, &mut scratch[..batch.len() * ROW_BYTES], 8)?;
        }
        self.prefetch_calls.fetch_add(1, Ordering::Relaxed);
        self.prefetched_rows.fetch_add(ids.len() as u64, Ordering::Relaxed);
        Ok(())
    }
    /// Application-cache cold diagnostic; no backing-file or OS-cache change.
    pub(crate) fn clear_for_probe(&self) -> Result<()> {
        for segment in &self.segments {
            let mut cache = segment.lock().map_err(|_| anyhow::anyhow!("Engram cache poisoned"))?;
            cache.tags.fill(EMPTY);
            cache.hands.fill(0);
        }
        Ok(())
    }
    /// Diagnostic-only eviction of this immutable shard's OS page-cache data.
    /// This is advisory, not a global drop_caches or a modification of weights.
    /// Call only after all scoped reads have drained; measured physical-read
    /// counters distinguish a genuinely cold pass from an ignored hint.
    #[cfg(target_os="linux")]
    pub(crate) fn evict_file_pages_for_probe(&self) -> Result<()> {
        use std::os::fd::AsRawFd;
        ensure!(std::env::var("CANADA_SCRATCH_ENGRAM_COLD_SCREEN").as_deref()==Ok("1")
            || std::env::var("CANADA_SCRATCH_ENGRAM_SPLIT14_SCREEN").as_deref()==Ok("1"),
            "file-cache eviction requires the isolated Engram screen");
        self.clear_for_probe()?;
        let status=unsafe{libc::posix_fadvise(self.shard.file.as_raw_fd(),HEADER_BYTES as libc::off_t,0,libc::POSIX_FADV_DONTNEED)};
        ensure!(status==0,"Engram file-cache advice failed: {}",std::io::Error::from_raw_os_error(status));
        Ok(())
    }
    /// Gather cache misses once, deduplicate raw row IDs, and issue bounded
    /// concurrent storage reads. No cache lock is held across physical I/O.
    #[cfg(target_os = "linux")]
    fn lookup_batched(
        &self,
        ids: &[u64],
        out: &mut [u8],
        reader: &Mutex<crate::engram_io::Reader>,
    ) -> Result<()> {
        self.drain_read_ahead()?;
        let plan=self.find_misses(ids,out)?;
        self.read_misses(&plan,out,reader)?;
        self.complete_lookup(plan,out);
        Ok(())
    }
    #[cfg(target_os = "linux")]
    fn read_misses(&self,plan:&Misses,out:&mut[u8],reader:&Mutex<crate::engram_io::Reader>)->Result<()> {
        let threads=miss_threads();
        if threads>0 && plan.rows.len()>=4*threads {
            return self.read_misses_threaded(plan,out,threads);
        }
        if !plan.rows.is_empty() {
            let mut reader=reader.lock().map_err(|_|anyhow::anyhow!("Engram I/O reader poisoned"))?;
            for batch in plan.rows.chunks(crate::engram_io::ENTRIES) {
                let offsets=self.offsets(batch);
                let (file,direct)=self.shard.io();
                let rows=reader.read(file,direct,&offsets)?;
                self.publish_misses(batch,rows,out)?;
            }
        }
        Ok(())
    }
    /// CANADA_SCRATCH_ENGRAM_MISS_THREADS: the misses of one lookup read by scoped threads of positional reads.
    /// The same immutable row bytes as the io_uring batches; only the I/O concurrency differs.
    #[cfg(target_os = "linux")]
    fn read_misses_threaded(&self,plan:&Misses,out:&mut[u8],threads:usize)->Result<()> {
        use std::os::unix::fs::FileExt;
        let (file,start)=(&self.shard.file,self.shard.start);
        read_rows_threaded(&plan.rows,out,threads,&|id,row|file.read_exact_at(row,HEADER_BYTES+(id-start)*ROW_BYTES as u64))?;
        for &(id,position) in &plan.rows {
            let row:&[u8;ROW_BYTES]=out[position*ROW_BYTES..(position+1)*ROW_BYTES].try_into().unwrap();
            self.segments[self.segment(id)].lock()
                .map_err(|_|anyhow::anyhow!("Engram cache poisoned"))?
                .insert(id,row);
        }
        Ok(())
    }
    #[cfg(target_os = "linux")]
    fn find_misses(&self,ids:&[u64],out:&mut[u8])->Result<Misses> {
        let mut missing = Vec::new();
        let mut first_miss = std::collections::HashMap::new();
        let mut duplicates = Vec::new();
        let mut hits = 0u64;
        let mut nonowned = 0u64;
        for (position, (&id, row)) in ids.iter().zip(out.chunks_exact_mut(ROW_BYTES)).enumerate() {
            if id < self.shard.start || id >= self.shard.end {
                row.fill(0);
                nonowned += 1;
                continue;
            }
            if self.segments[self.segment(id)]
                .lock()
                .map_err(|_| anyhow::anyhow!("Engram cache poisoned"))?
                .find(id, row.try_into().unwrap())
            {
                hits += 1;
            } else if let Some(&source) = first_miss.get(&id) {
                duplicates.push((position, source));
                hits += 1;
            } else {
                first_miss.insert(id, position);
                missing.push((id, position));
            }
        }
        Ok(Misses {rows:missing,duplicates,hits,nonowned})
    }
    #[cfg(target_os = "linux")]
    fn offsets(&self,batch:&[(u64,usize)])->Vec<u64> {
        batch.iter().map(|(id,_)|HEADER_BYTES+(id-self.shard.start)*ROW_BYTES as u64).collect()
    }
    #[cfg(target_os = "linux")]
    fn publish_misses(&self,batch:&[(u64,usize)],rows:&[u8],out:&mut[u8])->Result<()> {
        ensure!(rows.len()==batch.len()*ROW_BYTES,"Engram readback geometry");
        for (&(id,position),row) in batch.iter().zip(rows.chunks_exact(ROW_BYTES)) {
            out[position*ROW_BYTES..(position+1)*ROW_BYTES].copy_from_slice(row);
            self.segments[self.segment(id)].lock()
                .map_err(|_|anyhow::anyhow!("Engram cache poisoned"))?
                .insert(id,row.try_into().unwrap());
        }
        Ok(())
    }
    #[cfg(target_os = "linux")]
    fn complete_lookup(&self,plan:Misses,out:&mut[u8]) {
        for (destination, source) in plan.duplicates {
            out.copy_within(
                source * ROW_BYTES..(source + 1) * ROW_BYTES,
                destination * ROW_BYTES,
            );
        }
        self.hits.fetch_add(plan.hits, Ordering::Relaxed);
        self.nonowned.fetch_add(plan.nonowned, Ordering::Relaxed);
        self.misses
            .fetch_add(plan.rows.len() as u64, Ordering::Relaxed);
    }
    /// Two persistent CPU workers retain the ordinary independent io_uring
    /// lookups. Neither CUDA nor mutable model state enters these closures.
    #[cfg(target_os = "linux")]
    pub(crate) fn lookup_pair_with_pool(pool:&rayon::ThreadPool,a:&Self,ids_a:&[u64],out_a:&mut[u8],b:&Self,ids_b:&[u64],out_b:&mut[u8])->Result<()> {
        ensure!(pool.current_num_threads()==2,"bounded paired Engram worker pool");
        let(first,second)=pool.install(||rayon::join(
            ||a.lookup_into(ids_a,out_a,8),||b.lookup_into(ids_b,out_b,8)));
        first?;second?;Ok(())
    }
    /// Preserve the caller's cache/CPU placement for the first file, while
    /// borrowing one persistent worker for the other. Scope completion joins
    /// the borrowed output slice before it can be uploaded or reused.
    #[cfg(target_os = "linux")]
    pub(crate) fn lookup_pair_with_worker(pool:&rayon::ThreadPool,a:&Self,ids_a:&[u64],out_a:&mut[u8],b:&Self,ids_b:&[u64],out_b:&mut[u8])->Result<()> {
        ensure!(pool.current_num_threads()==1,"one paired Engram helper");
        let mut second=None;
        let first=pool.in_place_scope(|scope| {
            scope.spawn(|_|{second=Some(b.lookup_into(ids_b,out_b,8));});
            a.lookup_into(ids_a,out_a,8)
        });
        first?;second.context("paired Engram helper did not return")??;Ok(())
    }
    /// Keep the no-thread warm path without serializing discovery of cold
    /// rows. Tags are only a scheduling hint: the chosen lookup still checks
    /// every row under its normal lock and retrieves canonical bytes on a miss.
    #[cfg(target_os = "linux")]
    pub(crate) fn lookup_pair_warm_or_parallel(a:&Self,ids_a:&[u64],out_a:&mut[u8],b:&Self,ids_b:&[u64],out_b:&mut[u8])->Result<()> {
        Self::lookup_pair_warm_or_parallel_on(None,a,ids_a,out_a,b,ids_b,out_b)
    }
    #[cfg(target_os = "linux")]
    pub(crate) fn lookup_pair_warm_or_parallel_on(pool:Option<&rayon::ThreadPool>,a:&Self,ids_a:&[u64],out_a:&mut[u8],b:&Self,ids_b:&[u64],out_b:&mut[u8])->Result<()> {
        for (store,ids,out) in [(a,ids_a,&*out_a),(b,ids_b,&*out_b)] {
            ensure!(ids.len()<=32*24 && out.len()==ids.len()*ROW_BYTES
                && ids.iter().all(|&id|id<store.shard.global_rows)
                &&store.io_reader.is_some(),"adaptive paired Engram input extent/domain/reader");
        }
        if !a.misses_any(ids_a.iter().copied())? && !b.misses_any(ids_b.iter().copied())? {
            return Self::lookup_pair_into(a,ids_a,out_a,b,ids_b,out_b);
        }
        if let Some(pool)=pool {return Self::lookup_pair_with_worker(pool,a,ids_a,out_a,b,ids_b,out_b);}
        std::thread::scope(|scope|->Result<()> {
            let other=scope.spawn(||b.lookup_into(ids_b,out_b,8));
            let first=a.lookup_into(ids_a,out_a,8);
            let second=other.join().map_err(|_|anyhow::anyhow!("adaptive Engram reader thread panicked"))?;
            first?;second?;Ok(())
        })
    }
    /// Narrow-row experiment: preserve exact independent cache semantics while
    /// submitting both files' misses together, with no transient worker thread.
    #[cfg(target_os = "linux")]
    pub(crate) fn lookup_pair_into(a:&Self,ids_a:&[u64],out_a:&mut[u8],b:&Self,ids_b:&[u64],out_b:&mut[u8])->Result<()> {
        // Diagnostic (CANADA_SCRATCH_HOST_STAGES=1): how long a paired lookup takes, cache search and reads together.
        let started = crate::config::host_stages().then(std::time::Instant::now);
        let result = Self::lookup_pair_into_inner(a,ids_a,out_a,b,ids_b,out_b);
        if let Some(t) = started {
            IO_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            IO_STATS[1].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        result
    }
    #[cfg(target_os = "linux")]
    fn lookup_pair_into_inner(a:&Self,ids_a:&[u64],out_a:&mut[u8],b:&Self,ids_b:&[u64],out_b:&mut[u8])->Result<()> {
        for (store,ids,out) in [(a,ids_a,&*out_a),(b,ids_b,&*out_b)] {
            ensure!(ids.len()<=32*24 && out.len()==ids.len()*ROW_BYTES
                && ids.iter().all(|&id|id<store.shard.global_rows),"paired Engram input extent/domain");
            ensure!(store.io_reader.is_some(),"paired Engram requires io_uring stores");
        }
        a.drain_read_ahead()?;b.drain_read_ahead()?;
        let first=a.find_misses(ids_a,out_a)?;
        let second=b.find_misses(ids_b,out_b)?;
        if first.rows.len()+second.rows.len()>crate::engram_io::ENTRIES
            && !first.rows.is_empty() && !second.rows.is_empty() {
            // A single queue saves thread overhead for the usual narrow/mixed
            // rows. Once both files need multiple batches, preserve parallel
            // reads without repeating cache discovery or double-counting hits.
            let (left,right)=std::thread::scope(|scope| {
                let other=scope.spawn(||b.read_misses(&second,out_b,b.io_reader.as_ref().unwrap()));
                let here=a.read_misses(&first,out_a,a.io_reader.as_ref().unwrap());
                (here,other.join())
            });
            left?;
            right.map_err(|_|anyhow::anyhow!("Engram paired reader thread panicked"))??;
        } else if !first.rows.is_empty() || !second.rows.is_empty() {
            let mut reader=a.io_reader.as_ref().unwrap().lock()
                .map_err(|_|anyhow::anyhow!("Engram I/O reader poisoned"))?;
            let(mut i,mut j)=(0,0);
            while i<first.rows.len() || j<second.rows.len() {
                let na=(first.rows.len()-i).min(crate::engram_io::ENTRIES);
                let nb=(second.rows.len()-j).min(crate::engram_io::ENTRIES-na);
                let ra=&first.rows[i..i+na];let rb=&second.rows[j..j+nb];
                let oa=a.offsets(ra);let ob=b.offsets(rb);
                let ((file_a,direct_a),(file_b,direct_b))=(a.shard.io(),b.shard.io());
                ensure!(direct_a==direct_b,"paired Engram stores differ in read mode");
                let rows=reader.read_pair(file_a,&oa,file_b,&ob,direct_a)?;
                a.publish_misses(ra,&rows[..na*ROW_BYTES],out_a)?;
                b.publish_misses(rb,&rows[na*ROW_BYTES..],out_b)?;
                i+=na;j+=nb;
            }
        }
        a.complete_lookup(first,out_a);b.complete_lookup(second,out_b);
        Ok(())
    }
    pub fn stats(&self) -> serde_json::Value {
        #[cfg(target_os = "linux")]
        let io_workspace = self
            .io_reader
            .as_ref()
            .map(|r| r.lock().map(|r| r.workspace_bytes()).unwrap_or(0))
            .unwrap_or(0);
        #[cfg(not(target_os = "linux"))]
        let io_workspace = 0usize;
        serde_json::json!({"layer":self.shard.layer,"uring_workspace_bytes":io_workspace,"rank_row_start":self.shard.start,"rank_row_end":self.shard.end,
            "persistent_io_workers":self.io_pool.as_ref().map_or(0,|p|p.current_num_threads()),
            "resident_bytes":self.resident_bytes,"zero_pages_committed_at_creation":true,"hits":self.hits.load(Ordering::Relaxed),
            "misses":self.misses.load(Ordering::Relaxed),"nonowned":self.nonowned.load(Ordering::Relaxed),
            "read_ahead_rows":self.read_ahead_rows.load(Ordering::Relaxed),
            "prefetch_calls":self.prefetch_calls.load(Ordering::Relaxed),
            "prefetched_rows":self.prefetched_rows.load(Ordering::Relaxed)})
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn threaded_miss_reads_fill_exactly_their_rows() {
        for (count,threads) in [(1usize,2usize),(5,2),(64,3),(257,64),(1000,64),(1000,7)] {
            // every third position is a miss; the others must stay untouched
            let rows:Vec<(u64,usize)>=(0..count).map(|i|(1_000+7*i as u64,3*i+1)).collect();
            let mut out=vec![0xEEu8;(3*count+2)*ROW_BYTES];
            super::read_rows_threaded(&rows,&mut out,threads,&|id,row|{
                for (j,b) in row.iter_mut().enumerate() {*b=(id as usize+j) as u8;}
                Ok(())
            }).unwrap();
            for (position,row) in out.chunks_exact(ROW_BYTES).enumerate() {
                match rows.iter().find(|r|r.1==position) {
                    Some(&(id,_))=>assert!(row.iter().enumerate().all(|(j,b)|*b==(id as usize+j) as u8),"row {position}"),
                    None=>assert!(row.iter().all(|b|*b==0xEE),"untouched row {position}"),
                }
            }
        }
        // positions out of order are refused; a failing read is reported
        let mut out=vec![0u8;4*ROW_BYTES];
        assert!(super::read_rows_threaded(&[(1,2),(2,1)],&mut out,2,&|_,_|Ok(())).is_err());
        assert!(super::read_rows_threaded(&[(1,0),(2,1)],&mut out,2,&|id,_|if id==2 {
            Err(std::io::Error::other("short"))} else {Ok(())}).is_err());
        assert!(super::read_rows_threaded(&[(1,4)],&mut out,2,&|_,_|Ok(())).is_err());
    }

    use super::*;
    use std::io::Write;
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires Linux io_uring; run explicitly without GPU work"]
    fn io_uring_rows_and_error_drain() {
        let path =
            std::env::temp_dir().join(format!("canada-engram-uring-test-{}", std::process::id()));
        let n = 4096u64;
        let mut f = File::create(&path).unwrap();
        let mut header = vec![0; 4096];
        for (i, v) in [MAGIC, 1, 0, n / 4, n, ROW_BYTES as u64]
            .into_iter()
            .enumerate()
        {
            header[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        f.write_all(&header).unwrap();
        for id in 0..n / 4 {
            let row: Vec<u8> = (0..ROW_BYTES)
                .map(|j| ((id * 7 + j as u64) % 251) as u8)
                .collect();
            f.write_all(&row).unwrap();
        }
        f.sync_all().unwrap();
        let shard = PackedShard::open(&path, 1, 0, 4, n).unwrap();
        let mut store = RowStore::new_with_pool(shard, 80 * 1024, false).unwrap();
        store.io_reader = Some(Mutex::new(crate::engram_io::Reader::new().unwrap()));
        let mut ids: Vec<u64> = (0..1024).chain((0..1024).rev()).collect();
        ids.extend((0..8192).map(|i| (i * 997) % n));
        store.prefetch(&ids).unwrap();
        let check = || {
            let mut out = vec![0xa5; ids.len() * ROW_BYTES];
            store.lookup_into(&ids, &mut out, 8).unwrap();
            for (&id, row) in ids.iter().zip(out.chunks_exact(ROW_BYTES)) {
                for (j, &byte) in row.iter().enumerate() {
                    assert_eq!(
                        byte,
                        if id >= n / 4 {
                            0
                        } else {
                            ((id * 7 + j as u64) % 251) as u8
                        }
                    );
                }
            }
        };
        for _ in 0..3 {
            check();
        }
        std::thread::scope(|scope| {
            let a = scope.spawn(&check);
            let b = scope.spawn(&check);
            a.join().unwrap();
            b.join().unwrap();
        });
        assert!(store.lookup_into(&[n], &mut [0; ROW_BYTES], 8).is_err());
        store.lookup_into(&[], &mut [], 8).unwrap();
        f.set_len(HEADER_BYTES + ROW_BYTES as u64).unwrap();
        {
            let mut reader = store.io_reader.as_ref().unwrap().lock().unwrap();
            assert!(
                reader
                    .read(
                        &store.shard.file,
                        false,
                        &[HEADER_BYTES, HEADER_BYTES + 1023 * ROW_BYTES as u64]
                    )
                    .is_err()
            );
            // A short read still consumes every CQE and allows safe reuse.
            let rows = reader
                .read(&store.shard.file, false, &[HEADER_BYTES, HEADER_BYTES])
                .unwrap();
            for row in rows.chunks_exact(ROW_BYTES) {
                assert!(row.iter().enumerate().all(|(j, &b)| b == (j % 251) as u8));
            }
        }
        drop(store);
        drop(f);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn collisions_parallelism_ownership_and_truncation() {
        let path = std::env::temp_dir().join(format!("canada-engram-test-{}", std::process::id()));
        let n = 4096u64;
        let mut f = File::create(&path).unwrap();
        let mut header = vec![0; 4096];
        for (i, v) in [MAGIC, 1, 0, n / 4, n, ROW_BYTES as u64]
            .into_iter()
            .enumerate()
        {
            header[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        f.write_all(&header).unwrap();
        for id in 0..n / 4 {
            let row: Vec<_> = (0..ROW_BYTES)
                .map(|j| ((id * 7 + j as u64) % 251) as u8)
                .collect();
            f.write_all(&row).unwrap();
        }
        f.sync_all().unwrap();
        for pool in [false, true] {
            let shard = PackedShard::open(&path, 1, 0, 4, n).unwrap();
            let store = RowStore::new_with_pool(shard, 80 * 1024, pool).unwrap();
            store.prefetch(&[0, 0, n / 4, n - 1]).unwrap();
            let misses = store.stats()["misses"].as_u64().unwrap();
            let mut row = [0; ROW_BYTES];
            store.lookup(0, &mut row).unwrap();
            assert_eq!(store.stats()["misses"], misses);
            assert!(row.iter().enumerate().all(|(j, &b)| b == (j % 251) as u8));
            store.clear_for_probe().unwrap();
            store.lookup(0, &mut row).unwrap();
            assert_eq!(store.stats()["misses"], misses + 1);
            assert!(row.iter().enumerate().all(|(j, &b)| b == (j % 251) as u8));
            assert!(store.prefetch(&[n]).is_err());
            assert!(store.prefetch(&vec![0; crate::config::MAX_WIDE_ROWS * 24 + 1]).is_err());
            let ids: Vec<_> = (0..16384).map(|i| (i * 997) % n).collect();
            store.prefetch(&ids).unwrap();
            let mut out = vec![0u8; ids.len() * ROW_BYTES];
            for threads in [1, 8, 16] {
                store.lookup_into(&ids, &mut out, threads).unwrap();
                for (id, row) in ids.iter().zip(out.chunks_exact(ROW_BYTES)) {
                    if *id >= n / 4 {
                        assert!(row.iter().all(|b| *b == 0));
                    } else {
                        for (j, b) in row.iter().enumerate() {
                            assert_eq!(*b, ((id * 7 + j as u64) % 251) as u8);
                        }
                    }
                }
            }
            assert!(store.lookup(n, &mut [0; ROW_BYTES]).is_err());
        }
        f.set_len(4097).unwrap();
        assert!(PackedShard::open(&path, 1, 0, 4, n).is_err());
        drop(f);
        std::fs::remove_file(path).unwrap();
    }
}
