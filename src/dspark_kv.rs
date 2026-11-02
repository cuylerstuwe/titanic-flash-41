//! Independent bounded DSpark attention state. Accepted target-derived KV uses
//! a256-slot ring; the five bidirectional query slots are physically separate.
//! Reference: qualified DSpark window metadata and weighted norm/RoPE writer.
use crate::{
    attention::{Candidates, PagedKv},
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, Module, Stream},
    kv::{RowLocations, Store},
    prelude::FrequencyTable,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{path::Path, rc::Rc};

pub const QUERY_ROWS: usize = 5;
pub const CANDIDATE_WIDTH: usize = 192;
const HISTORY_RING: usize = 256;
const HISTORY_BASE: usize = 64;
const QUERY_BASE: usize = HISTORY_BASE + HISTORY_RING;
const CAPACITY: usize = 384;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Layout {
    pub positions: [i32; QUERY_ROWS],
    pub locations: [i32; QUERY_ROWS],
    pub candidates: Vec<i32>,
    pub lengths: [i32; QUERY_ROWS],
}
pub fn query_layout(frontier: u64, frequency_rows: usize) -> Result<Layout> {
    ensure!(
        frontier
            .checked_add(QUERY_ROWS as u64)
            .is_some_and(|n| n <= frequency_rows as u64)
            && frequency_rows <= 1048576,
        "DSpark query exceeds rotary/context bound"
    );
    let positions = std::array::from_fn(|i| (frontier + i as u64) as i32);
    let locations = std::array::from_fn(|i| (QUERY_BASE + i) as i32);
    let mut one = vec![-1; CANDIDATE_WIDTH];
    let first = frontier.saturating_sub(128);
    let history = (frontier - first) as usize;
    for (i, p) in (first..frontier).enumerate() {
        one[i] = (HISTORY_BASE + p as usize % HISTORY_RING) as i32;
    }
    one[history..history + QUERY_ROWS].copy_from_slice(&locations);
    // All five query rows see the same complete block. This is deliberately
    // different from the target verifier's six-row causal attention mask.
    Ok(Layout {
        positions,
        locations,
        candidates: one.repeat(QUERY_ROWS),
        lengths: [(history + QUERY_ROWS) as i32; QUERY_ROWS],
    })
}

pub(crate) struct StateImage {
    image: crate::state_image::Image,
    owner: Rc<()>,
    frontier: u64,
}
impl StateImage {
    pub(crate) fn manifest(&self) -> serde_json::Value { self.image.manifest() }
    pub(crate) fn digest(&self) -> Result<String> { self.image.digest() }
    pub(crate) fn bytes(&self) -> usize { self.image.bytes() }
}

pub struct State {
    image_owner: Rc<()>,
    cuda: Cuda,
    stream: usize,
    frequencies: Rc<FrequencyTable>,
    store: Store,
    norms: [Buffer<bf16>; 3],
    caches: [PagedKv; 3],
    six: RowLocations,
    wide: RowLocations,
    query: RowLocations,
    candidates: Candidates,
    frontier: u64,
    poisoned: bool,
    query_ready: bool,
    query_prepared: bool,
    queued_copy: Option<(u64,usize,usize)>,
}
impl State {
    pub fn new(
        cuda: &Cuda,
        stream: &Stream,
        weights: &Weights,
        kernels: &Path,
        frequencies: Rc<FrequencyTable>,
    ) -> Result<Self> {
        ensure!(
            cuda.owns_stream(stream)
                && weights.in_context(cuda)
                && !frequencies.compressed()
                && cuda.owns_buffer(frequencies.values()),
            "DSpark KV context/rotary table"
        );
        stream.join_default()?;
        let mut norms = Vec::new();
        let mut caches = Vec::new();
        for stage in 0..3 {
            let name = format!("mtp.{stage}.attn.kv_norm.weight");
            let v = weights.view(&name)?;
            ensure!(
                v.dtype == Dtype::Bf16 && v.shape == [512],
                "DSpark KV norm geometry"
            );
            let norm = cuda.allocate::<bf16>(512)?;
            unsafe { cuda.copy_device(norm.ptr(), weights.ptr(&name)?, norm.bytes(), stream) }?;
            norms.push(norm);
            caches.push(PagedKv::new(cuda, CAPACITY / 64, 64)?);
        }
        let frequency_rows = frequencies.rows();
        let metadata = |rows| {
            RowLocations::upload(
                cuda,
                &vec![0; rows],
                &vec![-1; rows],
                CAPACITY,
                frequency_rows,
            )
        };
        Ok(Self {
            image_owner: Rc::new(()),
            cuda: cuda.clone(),
            stream: stream.raw() as usize,
            frequencies,
            store: Store::open(cuda, &kernels.join("kv-v2"))?,
            norms: norms.try_into().ok().unwrap(),
            caches: caches.try_into().ok().unwrap(),
            six: metadata(6)?,
            wide: metadata(128)?,
            query: metadata(5)?,
            candidates: Candidates::upload(
                cuda,
                5,
                CANDIDATE_WIDTH,
                CAPACITY,
                &vec![-1; 5 * CANDIDATE_WIDTH],
                &[0; 5],
            )?,
            frontier: 0,
            poisoned: false,
            query_ready: false,
            query_prepared: false,
            queued_copy: None,
        })
    }
    fn guard(&self, stream: &Stream) -> Result<()> {
        ensure!(
            !self.poisoned && self.cuda.owns_stream(stream) && self.stream == stream.raw() as usize,
            "DSpark KV poisoned or foreign stream"
        );
        Ok(())
    }
    pub fn position(&self) -> Result<u64> {
        ensure!(!self.poisoned, "DSpark KV poisoned");
        Ok(self.frontier)
    }
    pub(crate) fn independent_from(&self,other:&Self)->Result<bool> {
        let a=self.image_regions()?;let b=other.image_regions()?;
        Ok(!Rc::ptr_eq(&self.image_owner,&other.image_owner)
            &&a.iter().all(|x|b.iter().all(|y|!x.overlaps(y))))
    }
    pub(crate) fn accepted_replay_digests(&self, root: Option<&Path>) -> Result<serde_json::Value> {
        use sha2::{Digest,Sha256};
        ensure!(!self.poisoned && std::env::var("CANADA_SCRATCH_DRAFT_REPLAY_TRACE").as_deref()==Ok("1"),
            "accepted draft trace requires diagnostic admission");
        let mut records=Vec::new();
        for (stage,cache) in self.caches.iter().enumerate() {
            let data=cache.data().download()?;let mut live=Vec::new();
            for position in self.frontier.saturating_sub(128)..self.frontier {
                let slot=HISTORY_BASE+position as usize%HISTORY_RING;
                let base=slot/64*cache.page_bytes();let row=slot%64;
                live.extend_from_slice(&data[base+row*576..base+(row+1)*576]);
                // Match the seven observable FP8 scale bytes, excluding padding.
                live.extend_from_slice(&data[base+64*576+row*8..base+64*576+row*8+7]);
            }
            if let Some(root)=root {std::fs::write(root.join(format!("draft-history-{stage}.bin")),&live)?;}
            records.push(serde_json::json!({"stage":stage,"bytes":live.len(),"sha256":format!("{:x}",Sha256::digest(&live))}));
        }
        Ok(serde_json::json!({"frontier":self.frontier,"live_rows":self.frontier.min(128),"stages":records}))
    }
    fn image_regions(&self) -> Result<Vec<crate::state_image::Region<'_>>> {
        self.caches.iter().enumerate().map(|(i, c)|
            crate::state_image::Region::new(&self.cuda, format!("draft.{i}.kv"), c.data())).collect()
    }
    pub(crate) fn state_image_bytes(&self)->Result<usize> {
        ensure!(!self.poisoned && self.frontier>0,"draft image size needs accepted state");
        Ok(self.image_regions()?.iter().map(|r|r.bytes()).sum())
    }
    pub(crate) fn save_state_image(&mut self, budget: usize, stream: &Stream) -> Result<StateImage> {
        self.guard(stream)?;
        ensure!(self.frontier > 0, "draft state image requires committed history");
        let mut image = crate::state_image::Image::allocate(&self.cuda, &self.image_regions()?, budget, stream)?;
        stream.synchronize()?;
        self.poisoned = true;
        image.save(&self.image_regions()?, stream)?;
        self.poisoned = false;
        Ok(StateImage { image, owner: self.image_owner.clone(), frontier: self.frontier })
    }
    pub(crate) fn restore_state_image(&mut self, saved: &StateImage, stream: &Stream) -> Result<()> {
        self.guard(stream)?;
        ensure!(Rc::ptr_eq(&self.image_owner, &saved.owner), "foreign draft state image");
        saved.image.validate(&self.image_regions()?, stream)?;
        stream.synchronize()?;
        self.poisoned = true;
        unsafe { saved.image.restore(&self.image_regions()?, stream) }?;
        self.frontier = saved.frontier;
        self.query_ready = false;
        self.query_prepared = false;
        self.poisoned = false;
        Ok(())
    }
    /// Initialize a fresh draft owner from the complete recent target tail.
    /// Older target rows are outside the128-row draft window; positions still
    /// remain absolute so ordinary RoPE and ring addressing match the target.
    pub(crate) fn seed_tail(
        &mut self,
        raw: [&Buffer<bf16>; 3],
        live: usize,
        end: u64,
        stream: &Stream,
    ) -> Result<()> {
        self.guard(stream)?;
        ensure!(
            self.frontier == 0
                && !self.query_ready
                && !self.query_prepared
                && live > 0
                && live == (end.min(128) as usize)
                && end <= self.frequencies.rows() as u64,
            "DSpark initial accepted tail"
        );
        let start = end - live as u64;
        self.frontier = start;
        let result = self.commit(raw, live, live, start, stream);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    pub fn commit(
        &mut self,
        raw: [&Buffer<bf16>; 3],
        live: usize,
        accepted: usize,
        start: u64,
        stream: &Stream,
    ) -> Result<()> {
        self.guard(stream)?;
        let rows = raw[0].len() / 512;
        ensure!(
            [6, 128].contains(&rows)
                && live > 0
                && live <= rows
                && accepted <= live
                && start == self.frontier
                && start
                    .checked_add(accepted as u64)
                    .is_some_and(|n| n <= self.frequencies.rows() as u64)
                && raw
                    .iter()
                    .all(|b| b.len() == rows * 512 && self.cuda.owns_buffer(b)),
            "DSpark accepted KV prefix/shape/frontier"
        );
        if accepted == 0 {
            self.query_ready = false;
            self.query_prepared = false;
            return Ok(());
        }
        let mut positions = vec![0; rows];
        let mut locations = vec![-1; rows];
        for i in 0..accepted {
            positions[i] = (start + i as u64) as i32;
            locations[i] = (HISTORY_BASE + (start as usize + i) % HISTORY_RING) as i32;
        }
        self.poisoned = true;
        self.query_ready = false;
        self.query_prepared = false;
        let metadata = if rows == 6 {
            &mut self.six
        } else {
            &mut self.wide
        };
        metadata.update(&positions, &locations)?;
        stream.join_default()?;
        for stage in 0..3 {
            self.store.run(
                raw[stage],
                512,
                &self.norms[stage],
                self.frequencies.values(),
                metadata,
                &mut self.caches[stage],
                stream,
            )?;
        }
        // This correctness-first commit boundary is explicit. Later fused
        // device acceptance can remove this synchronization after qualification.
        stream.synchronize()?;
        self.frontier += accepted as u64;
        self.poisoned = false;
        Ok(())
    }
    /// Private transaction: position/read/query/image APIs remain poisoned
    /// until every chunk has completed. In-flight KV writes cannot be exposed
    /// through an accepted frontier, query or saved state before publication.
    pub(crate) fn begin_copy_commit(&mut self,start:u64,total:usize,stream:&Stream)->Result<()> {
        self.guard(stream)?;
        ensure!(self.queued_copy.is_none()&&(1..=32).contains(&total)&&start==self.frontier
            &&start.checked_add(total as u64).is_some_and(|n|n<=self.frequencies.rows()as u64),
            "queued copy commit frontier/extent");
        self.poisoned=true;self.query_ready=false;self.query_prepared=false;
        self.queued_copy=Some((start,total,0));Ok(())
    }
    pub(crate) fn enqueue_copy_chunk(&mut self,raw:[&Buffer<bf16>;3],live:usize,
        module:&Module,stream:&Stream)->Result<()> {
        let(start,total,written)=self.queued_copy.ok_or_else(||anyhow::anyhow!("no queued copy transaction"))?;
        ensure!(self.poisoned&&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize
            &&(1..=6).contains(&live)&&written+live<=total
            &&raw.iter().all(|b|b.len()==6*512&&self.cuda.owns_buffer(b)),"queued copy chunk bounds/context");
        self.six.enqueue_ring(&self.cuda,module,start as usize+written,live,HISTORY_BASE,HISTORY_RING,stream)?;
        for stage in 0..3 {
            self.store.run(raw[stage],512,&self.norms[stage],self.frequencies.values(),
                &self.six,&mut self.caches[stage],stream)?;
        }
        self.queued_copy=Some((start,total,written+live));Ok(())
    }
    pub(crate) fn finish_copy_commit(&mut self,stream:&Stream)->Result<()> {
        let(start,total,written)=self.queued_copy.ok_or_else(||anyhow::anyhow!("no queued copy transaction"))?;
        ensure!(self.poisoned&&self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize
            &&self.frontier==start&&written==total,"queued copy final frontier");
        stream.synchronize()?;
        self.frontier=start+total as u64;self.queued_copy=None;self.poisoned=false;Ok(())
    }
    pub fn store_queries(
        &mut self,
        raw: [&Buffer<bf16>; 3],
        start: u64,
        stream: &Stream,
    ) -> Result<()> {
        self.guard(stream)?;
        ensure!(
            start == self.frontier
                && raw
                    .iter()
                    .all(|b| b.len() == QUERY_ROWS * 512 && self.cuda.owns_buffer(b)),
            "DSpark query KV shape/frontier"
        );
        let layout = query_layout(start, self.frequencies.rows())?;
        self.poisoned = true;
        self.query_ready = false;
        self.query.update(&layout.positions, &layout.locations)?;
        self.candidates
            .update(&layout.candidates, &layout.lengths)?;
        stream.join_default()?;
        for stage in 0..3 {
            self.store.run(
                raw[stage],
                512,
                &self.norms[stage],
                self.frequencies.values(),
                &self.query,
                &mut self.caches[stage],
                stream,
            )?;
        }
        stream.synchronize()?;
        self.query_ready = true;
        self.query_prepared = false;
        self.poisoned = false;
        Ok(())
    }
    /// Host metadata preparation precedes capture/replay; stage operators then
    /// write their own five query KV rows before reading the common full block.
    pub(crate) fn prepare_query(&mut self, stream: &Stream) -> Result<Layout> {
        self.guard(stream)?;
        let layout = query_layout(self.frontier, self.frequencies.rows())?;
        self.poisoned = true;
        self.query_ready = false;
        self.query_prepared = false;
        self.query.update(&layout.positions, &layout.locations)?;
        self.candidates
            .update(&layout.candidates, &layout.lengths)?;
        stream.join_default()?;
        self.query_prepared = true;
        self.poisoned = false;
        Ok(layout)
    }
    pub(crate) fn query_parts(
        &mut self,
        stage: usize,
        stream: &Stream,
    ) -> Result<(&mut PagedKv, &Candidates, &RowLocations)> {
        self.guard(stream)?;
        ensure!(
            stage < 3 && self.query_prepared,
            "DSpark stage requires current block metadata"
        );
        Ok((&mut self.caches[stage], &self.candidates, &self.query))
    }
    pub(crate) fn publish_queries(&mut self, stream: &Stream) -> Result<()> {
        self.guard(stream)?;
        ensure!(self.query_prepared, "DSpark block was not prepared");
        stream.synchronize()?;
        self.query_ready = true;
        self.query_prepared = false;
        Ok(())
    }
    pub(crate) fn invalidate(&mut self) {
        self.poisoned = true;
        self.query_ready = false;
        self.query_prepared = false;
    }
    pub fn attention_state(
        &self,
        stage: usize,
        stream: &Stream,
    ) -> Result<(&PagedKv, &Candidates)> {
        self.guard(stream)?;
        ensure!(stage < 3 && self.query_ready, "DSpark query KV not ready");
        Ok((&self.caches[stage], &self.candidates))
    }
    pub fn reset(&mut self, stream: &Stream) -> Result<()> {
        ensure!(
            self.cuda.owns_stream(stream) && self.stream == stream.raw() as usize,
            "DSpark reset foreign stream"
        );
        self.poisoned = true;
        self.query_ready = false;
        self.query_prepared = false;
        for c in &mut self.caches {
            c.clear(&self.cuda, stream)?;
        }
        stream.synchronize()?;
        self.frontier = 0;
        self.poisoned = false;
        Ok(())
    }
    pub(crate) fn cache(&self, stage: usize) -> Result<&PagedKv> {
        ensure!(stage < 3 && !self.poisoned, "DSpark cache unavailable");
        Ok(&self.caches[stage])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn draft_mask_has_shared_full_block_and_bounded_wrapping_history() {
        let a = query_layout(0, 1048576).unwrap();
        assert_eq!(a.positions, [0, 1, 2, 3, 4]);
        assert_eq!(&a.candidates[..5], &[320, 321, 322, 323, 324]);
        assert_eq!(a.lengths, [5; 5]);
        let b = query_layout(257, 1048576).unwrap();
        assert_eq!(b.lengths, [133; 5]);
        assert_eq!(b.candidates[0], 193);
        assert_eq!(b.candidates[126], 319);
        assert_eq!(b.candidates[127], 64);
        for row in 0..5 {
            assert_eq!(
                &b.candidates[row * 192..(row + 1) * 192],
                &b.candidates[..192]
            );
            assert_eq!(
                &b.candidates[row * 192 + 128..row * 192 + 133],
                &[320, 321, 322, 323, 324]
            );
        }
        assert!(query_layout(1048571, 1048576).is_ok());
        assert!(query_layout(1048572, 1048576).is_err());
    }
}
