//! Bounded, tagged GPU encoder-tail storage for qualified decoder replay.
//! Retaining this128-row ring avoids retaining a dense encoder hidden history.
use crate::{
    attention::Candidates,
    cuda::{Buffer, Cuda, Module, Stream},
    ffn::TokenRows,
    prelude::Positions,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{
    cell::{Cell, RefCell},
    ffi::c_void,
};
pub(crate) struct Cache {
    cuda: Cuda,
    module: Module,
    context: usize,
    kv_capacity: usize,
    hidden: Buffer<bf16>,
    pre: Buffer<f32>,
    ids: Buffer<i32>,
    lengths: Buffer<i32>,
    tokens: Buffer<i64>,
    tags: Buffer<i32>,
    error: Buffer<u32>,
    accepted: RefCell<Buffer<i32>>,
    // Beyond the read-index all-candidates window: source20 block masks.
    masks: Option<Buffer<u8>>,
    bounded_copy: bool,
    committed: Cell<usize>,
    stream: Cell<usize>,
}
pub(crate) struct Replay {
    pub hidden: Buffer<bf16>,
    pub pre: Buffer<f32>,
    pub candidates: Candidates,
    pub candidate_mask: Option<Buffer<u8>>,
    pub tokens: TokenRows,
    begin: Buffer<i32>,
    end: usize,
}
/// Bounded publication copies only the newest 128 live rows (default on).
pub(crate) fn bounded_copy_enabled() -> Result<bool> {
    Ok(match std::env::var("CANADA_SCRATCH_BOUNDED_ENCODER_TAIL") {
        Err(std::env::VarError::NotPresent) => true,
        Ok(value) if value == "1" => true,
        Ok(value) if value == "0" => false,
        _ => anyhow::bail!("CANADA_SCRATCH_BOUNDED_ENCODER_TAIL must be0or1"),
    })
}
impl Cache {
    pub(crate) fn validate_capacity(context: usize, kv_capacity: usize) -> Result<()> {
        ensure!(
            crate::config::EXPERIMENTAL_CONTEXT_CAPACITIES.contains(&context)
                && context.checked_add(32).is_some_and(|n| kv_capacity >= n),
            "encoder tail capacity"
        );
        Ok(())
    }
    pub fn new(cuda: &Cuda, core: &[u8], context: usize, kv_capacity: usize) -> Result<Self> {
        Self::validate_capacity(context, kv_capacity)?;
        let bounded_copy = bounded_copy_enabled()?;
        let mut tags = cuda.allocate(128)?;
        tags.upload(&[-1i32; 128])?;
        let mut error = cuda.allocate(1)?;
        error.zero()?;
        let mut cache=Self {
            cuda: cuda.clone(),
            module: cuda.module(core)?,
            context,
            kv_capacity,
            hidden: cuda.allocate(128 * 20480)?,
            pre: cuda.allocate(128 * 4)?,
            ids: cuda.allocate(128 * 512)?,
            lengths: cuda.allocate(128)?,
            tokens: cuda.allocate(128)?,
            tags,
            error,
            accepted: RefCell::new(cuda.allocate(1)?),
            masks: crate::candidates::needed(context)
                .then(|| cuda.allocate(128 * Self::mask_stride(context)))
                .transpose()?,
            bounded_copy,
            committed: Cell::new(0),
            stream: Cell::new(0),
        };
        // CUDA allocations may reuse nonzero workspace pages. Invalid tags
        // prevent reads, but the empty-state byte contract must also match
        // clear(): never depend on freshly allocated pages happening to be zero.
        cache.hidden.zero()?;
        cache.pre.zero()?;
        cache.ids.zero()?;
        cache.lengths.zero()?;
        cache.tokens.zero()?;
        cache.accepted.borrow_mut().zero()?;
        Ok(cache)
    }
    fn mask_stride(context: usize) -> usize {
        context.div_ceil(crate::candidates::BLOCK)
    }
    fn bind(&self, stream: &Stream) -> Result<()> {
        ensure!(
            self.cuda.owns_stream(stream)
                && (self.stream.get() == 0 || self.stream.get() == stream.raw() as usize),
            "encoder tail ordered stream"
        );
        self.stream.set(stream.raw() as usize);
        Ok(())
    }
    pub fn new_replay(&self) -> Result<Replay> {
        let mut begin = self.cuda.allocate(1)?;
        begin.upload(&[0i32])?;
        Ok(Replay {
            hidden: self.cuda.allocate(128 * 20480)?,
            pre: self.cuda.allocate(128 * 4)?,
            candidates: Candidates::device_output(
                &self.cuda,
                128,
                512,
                self.kv_capacity,
                &[0; 128],
            )?,
            candidate_mask: self
                .masks
                .as_ref()
                .map(|_| self.cuda.allocate(128 * Self::mask_stride(self.context)))
                .transpose()?,
            tokens: TokenRows::new(&self.cuda, &[2; 128], 128)?,
            begin,
            end: 0,
        })
    }
    pub fn prepare(&self, replay: &mut Replay, previous: usize, end: usize) -> Result<()> {
        ensure!(
            self.committed.get() == previous
                && previous >= 128
                && end > previous
                && end <= self.context,
            "encoder tail accepted frontier"
        );
        replay.begin.upload(&[(end - 128) as i32])?;
        replay.end = end;
        Ok(())
    }
    /// Replay the newest128 rows of an already committed encoder frontier,
    /// after a wide producer published every encoder chunk.
    pub fn prepare_committed(&self, replay: &mut Replay, end: usize) -> Result<()> {
        ensure!(
            self.committed.get() == end && end >= 128 && end <= self.context,
            "encoder tail committed frontier"
        );
        replay.begin.upload(&[(end - 128) as i32])?;
        replay.end = end;
        Ok(())
    }
    pub fn commit(&self, end: usize) -> Result<()> {
        ensure!(
            end >= self.committed.get() && end <= self.context,
            "encoder tail commit frontier"
        );
        self.committed.set(end);
        Ok(())
    }
    pub fn position(&self) -> usize {
        self.committed.get()
    }
    /// Enumerate all retained encoder inputs, including long-context masks.
    /// `accepted` and Replay are scratch, regenerated before their consumers.
    pub(crate) fn image_regions(&self) -> Result<Vec<crate::state_image::Region<'_>>> {
        use crate::state_image::Region;
        let mut regions = vec![
            Region::new(&self.cuda, "encoder.hidden", &self.hidden)?,
            Region::new(&self.cuda, "encoder.pre", &self.pre)?,
            Region::new(&self.cuda, "encoder.ids", &self.ids)?,
            Region::new(&self.cuda, "encoder.lengths", &self.lengths)?,
            Region::new(&self.cuda, "encoder.tokens", &self.tokens)?,
            Region::new(&self.cuda, "encoder.tags", &self.tags)?,
            Region::new(&self.cuda, "encoder.error", &self.error)?,
        ];
        if let Some(mask) = &self.masks {
            regions.push(Region::new(&self.cuda, "encoder.masks", mask)?);
        }
        Ok(regions)
    }
    /// Target's exclusive revision lease guards both the restored bytes and
    /// this host frontier. This does not recover a poisoned owner.
    pub(crate) fn restore_image_frontier(&self, position: usize, stream: &Stream) -> Result<()> {
        self.bind(stream)?;
        ensure!(position <= self.context && self.error()? == 0, "restored encoder state invalid");
        self.committed.set(position);
        Ok(())
    }
    /// Publish only a completed target-accepted prefix. Speculative encoder
    /// rows must never overwrite an older row still needed by the128-row tail.
    pub fn publish_prefix(
        &self,
        x: &Buffer<bf16>,
        pre: &Buffer<f32>,
        selected: &Candidates,
        tokens: &TokenRows,
        positions: &Positions,
        mask: Option<&Buffer<u8>>,
        accepted: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            accepted > 0 && accepted <= tokens.ids().len(),
            "encoder tail accepted prefix"
        );
        self.accepted.borrow_mut().upload(&[accepted as i32])?;
        stream.join_default()?;
        self.publish_live(
            x,
            pre,
            selected,
            tokens,
            positions,
            mask,
            false,
            &self.accepted.borrow(),
            stream,
        )
    }
    pub fn publish(
        &self,
        x: &Buffer<bf16>,
        pre: &Buffer<f32>,
        selected: &Candidates,
        tokens: &TokenRows,
        positions: &Positions,
        mask: Option<&Buffer<u8>>,
        compact_mask: bool,
        stream: &Stream,
    ) -> Result<()> {
        self.publish_live(x, pre, selected, tokens, positions, mask, compact_mask, tokens.live(), stream)
    }
    fn publish_live(
        &self,
        x: &Buffer<bf16>,
        pre: &Buffer<f32>,
        selected: &Candidates,
        tokens: &TokenRows,
        positions: &Positions,
        mask: Option<&Buffer<u8>>,
        compact_mask: bool,
        live: &Buffer<i32>,
        stream: &Stream,
    ) -> Result<()> {
        let rows = tokens.ids().len();
        let bounded = self.bounded_copy && (1..=crate::config::MAX_WIDE_ROWS).contains(&rows);
        ensure!(
            self.masks.is_some() == mask.is_some()
                && (!compact_mask || (mask.is_some() && crate::config::wide_rows(rows)))
                && mask.is_none_or(|m| {
                    m.len() == (if compact_mask {128} else {rows}) * Self::mask_stride(self.context)
                        && self.cuda.owns_buffer(m)
                }),
            "encoder tail candidate-mask producer"
        );
        ensure!(
            x.len() == rows * 20480
                && pre.len() == rows * 4
                && selected.ids().len() == rows * 512
                && selected.lengths().len() == rows
                && positions.values().len() == rows,
            "encoder tail producer geometry"
        );
        ensure!(
            self.cuda.owns_buffer(x)
                && self.cuda.owns_buffer(pre)
                && self.cuda.owns_buffer(selected.ids())
                && self.cuda.owns_buffer(selected.lengths())
                && self.cuda.owns_buffer(tokens.ids())
                && self.cuda.owns_buffer(tokens.live())
                && self.cuda.owns_buffer(live)
                && live.len() == 1
                && self.cuda.owns_buffer(positions.values()),
            "encoder tail producer context"
        );
        self.bind(stream)?;
        let mut values = [
            x.ptr(),
            pre.ptr(),
            selected.ids().ptr(),
            selected.lengths().ptr(),
            tokens.ids().ptr(),
            positions.values().ptr(),
            live.ptr(),
            self.hidden.ptr(),
            self.pre.ptr(),
            self.ids.ptr(),
            self.lengths.ptr(),
            self.tokens.ptr(),
            self.tags.ptr(),
            self.error.ptr(),
        ];
        let mut args = values
            .iter_mut()
            .map(|v| (v as *mut u64).cast::<c_void>())
            .collect::<Vec<_>>();
        let mut r = rows as i32;
        let mut context = self.context as i32;
        args.extend([
            (&mut r as *mut i32).cast(),
            (&mut context as *mut i32).cast(),
        ]);
        unsafe {
            self.module.function(if bounded {
                "ca_store_encoder_tail_bounded"
            } else {
                "ca_store_encoder_tail"
            })?.launch(
                [((if bounded { rows.min(128) } else { rows }) * 2560).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )
        }?;
        if let (Some(ring), Some(mask)) = (&self.masks, mask) {
            let stride = Self::mask_stride(self.context);
            let bounded_mask = (bounded || compact_mask) && stride.is_multiple_of(16)
                && mask.ptr().is_multiple_of(16) && ring.ptr().is_multiple_of(16);
            ensure!(!compact_mask || bounded_mask, "compact tail mask alignment");
            let (mut m, mut s, mut p, mut l, mut out, mut r, mut c) = (
                mask.ptr(),
                stride as i32,
                positions.values().ptr(),
                live.ptr(),
                ring.ptr(),
                rows as i32,
                self.context as i32,
            );
            let mut args = [
                (&mut m as *mut u64).cast::<c_void>(),
                (&mut s as *mut i32).cast(),
                (&mut p as *mut u64).cast(),
                (&mut l as *mut u64).cast(),
                (&mut out as *mut u64).cast(),
                (&mut r as *mut i32).cast(),
                (&mut c as *mut i32).cast(),
            ];
            unsafe {
                self.module.function(if compact_mask {
                    "ca_store_compact_tail_mask"
                } else if bounded_mask {
                    "ca_store_tail_mask_bounded"
                } else {
                    "ca_store_tail_mask"
                })?.launch(
                    [((if bounded_mask { rows.min(128) * (stride / 16) } else { rows * stride })
                        .div_ceil(256)) as u32, 1, 1],
                    [256, 1, 1],
                    0,
                    stream,
                    &mut args,
                )
            }?;
        }
        Ok(())
    }
    pub fn gather(&self, replay: &Replay, stream: &Stream) -> Result<()> {
        ensure!(
            replay.end >= 128 && replay.end <= self.context,
            "encoder tail replay not prepared"
        );
        self.bind(stream)?;
        replay.candidates.bind_stream(stream)?;
        let mut values = [
            self.hidden.ptr(),
            self.pre.ptr(),
            self.ids.ptr(),
            self.lengths.ptr(),
            self.tokens.ptr(),
            self.tags.ptr(),
            replay.begin.ptr(),
            replay.hidden.ptr(),
            replay.pre.ptr(),
            replay.candidates.ids().ptr(),
            replay.candidates.lengths().ptr(),
            replay.tokens.ids().ptr(),
            self.error.ptr(),
        ];
        let mut args = values
            .iter_mut()
            .map(|v| (v as *mut u64).cast::<c_void>())
            .collect::<Vec<_>>();
        let mut context = self.context as i32;
        args.push((&mut context as *mut i32).cast());
        unsafe {
            self.module.function("ca_gather_encoder_tail")?.launch(
                [(128usize * 2560).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )
        }?;
        if let (Some(ring), Some(out)) = (&self.masks, &replay.candidate_mask) {
            let stride = Self::mask_stride(self.context);
            let (mut r, mut s, mut t, mut b, mut o, mut c) = (
                ring.ptr(),
                stride as i32,
                self.tags.ptr(),
                replay.begin.ptr(),
                out.ptr(),
                self.context as i32,
            );
            let mut args = [
                (&mut r as *mut u64).cast::<c_void>(),
                (&mut s as *mut i32).cast(),
                (&mut t as *mut u64).cast(),
                (&mut b as *mut u64).cast(),
                (&mut o as *mut u64).cast(),
                (&mut c as *mut i32).cast(),
            ];
            unsafe {
                self.module.function("ca_gather_tail_mask")?.launch(
                    [(128 * stride).div_ceil(256) as u32, 1, 1],
                    [256, 1, 1],
                    0,
                    stream,
                    &mut args,
                )
            }?;
        }
        replay.candidates.produced();
        Ok(())
    }
    pub fn clear(&self, stream: &Stream) -> Result<()> {
        self.bind(stream)?;
        for (ptr, bytes) in [
            (self.hidden.ptr(), self.hidden.bytes()),
            (self.pre.ptr(), self.pre.bytes()),
            (self.ids.ptr(), self.ids.bytes()),
            (self.lengths.ptr(), self.lengths.bytes()),
            (self.tokens.ptr(), self.tokens.bytes()),
            (self.error.ptr(), self.error.bytes()),
        ] {
            unsafe { self.cuda.zero_device(ptr, bytes, stream) }?;
        }
        let mut tags = self.tags.ptr();
        let mut args = [(&mut tags as *mut u64).cast()];
        unsafe {
            self.module.function("ca_reset_encoder_tail_tags")?.launch(
                [1, 1, 1],
                [128, 1, 1],
                0,
                stream,
                &mut args,
            )
        }?;
        self.committed.set(0);
        Ok(())
    }
    pub fn error(&self) -> Result<u32> {
        Ok(self.error.download()?[0])
    }
    pub fn digest(&self) -> Result<serde_json::Value> {
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        for (pointer, bytes) in [
            (self.hidden.ptr(), self.hidden.bytes()),
            (self.pre.ptr(), self.pre.bytes()),
            (self.ids.ptr(), self.ids.bytes()),
            (self.lengths.ptr(), self.lengths.bytes()),
            (self.tokens.ptr(), self.tokens.bytes()),
            (self.tags.ptr(), self.tags.bytes()),
        ] {
            digest.update(unsafe { self.cuda.read_device(pointer, bytes) }?);
        }
        Ok(
            serde_json::json!({"position":self.position(),"sha256":format!("{:x}",digest.finalize())}),
        )
    }
    pub fn bytes(&self) -> usize {
        self.hidden.bytes()
            + self.pre.bytes()
            + self.ids.bytes()
            + self.lengths.bytes()
            + self.tokens.bytes()
            + self.tags.bytes()
            + self.error.bytes()
            + self.accepted.borrow().bytes()
    }
}
