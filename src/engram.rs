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
        Ok(Self {
            file: f,
            layer,
            start,
            end,
            global_rows,
        })
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
}
impl RowStore {
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
        }
        Ok(store)
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
        })
    }
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
        let plan=self.find_misses(ids,out)?;
        self.read_misses(&plan,out,reader)?;
        self.complete_lookup(plan,out);
        Ok(())
    }
    #[cfg(target_os = "linux")]
    fn read_misses(&self,plan:&Misses,out:&mut[u8],reader:&Mutex<crate::engram_io::Reader>)->Result<()> {
        if !plan.rows.is_empty() {
            let mut reader=reader.lock().map_err(|_|anyhow::anyhow!("Engram I/O reader poisoned"))?;
            for batch in plan.rows.chunks(crate::engram_io::ENTRIES) {
                let offsets=self.offsets(batch);
                let rows=reader.read(&self.shard.file,&offsets)?;
                self.publish_misses(batch,rows,out)?;
            }
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
        for (store,ids,out) in [(a,ids_a,&*out_a),(b,ids_b,&*out_b)] {
            ensure!(ids.len()<=32*24 && out.len()==ids.len()*ROW_BYTES
                && ids.iter().all(|&id|id<store.shard.global_rows),"paired Engram input extent/domain");
            ensure!(store.io_reader.is_some(),"paired Engram requires io_uring stores");
        }
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
                let rows=reader.read_pair(&a.shard.file,&oa,&b.shard.file,&ob)?;
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
            "prefetch_calls":self.prefetch_calls.load(Ordering::Relaxed),
            "prefetched_rows":self.prefetched_rows.load(Ordering::Relaxed)})
    }
}

#[cfg(test)]
mod tests {
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
                        &[HEADER_BYTES, HEADER_BYTES + 1023 * ROW_BYTES as u64]
                    )
                    .is_err()
            );
            // A short read still consumes every CQE and allows safe reuse.
            let rows = reader
                .read(&store.shard.file, &[HEADER_BYTES, HEADER_BYTES])
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
