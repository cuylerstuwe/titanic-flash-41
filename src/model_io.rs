//! Target embedding and output boundaries for the qualified V4.1 normalization
//! contract: predecessor pre coefficients, final RMSNorm, canonical BF16 head
//! operands, FP32 accumulation/output and lossless vocabulary gather. DeepSeek's
//! minimal reference widens both operands before its FP32 head; we preserve
//! those exact operand values without materializing another weight atlas.
//! There is no intermediate BF16 logit rounding and no learned HC-head tensor.
use crate::{
    bf16_linear::Projection,
    blas::Blas,
    checkpoint::{Checkpoint, Dtype},
    cuda::{Buffer, Cuda, Module, Stream},
    ffn::TokenRows,
    layout::RankPlan,
    prelude::Native,
    roce::Roce,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{cell::{Cell, RefCell}, path::Path, rc::Rc};
pub struct ModelIo {
    cuda: Cuda,
    module: Module,
    weights: Rc<Weights>,
    // Immutable encoding follows the canonical atlas across profiles/owners.
    // Normalization, logits, sampler inputs and media remain private.
    lossless_atlas: crate::bf16_linear::LosslessHeadAtlas,
    media: Rc<RefCell<Option<Rc<crate::prompt_media::Bindings>>>>,
    rows: usize,
    // The wide encoder profile embeds tokens only; its prompt logits come from
    // the 128-row decoder tail profile, so no rows*vocab buffers are held.
    input_only: bool,
    rank: usize,
    local_embedding: Buffer<bf16>,
    embedding: Buffer<bf16>,
    expanded: Buffer<bf16>,
    combined: Buffer<bf16>,
    norm: Native,
    norm_weight: Buffer<bf16>,
    normalized: Buffer<bf16>,
    head: Projection<f32>,
    last_head: Projection<f32>,
    last_input: Buffer<bf16>,
    last_gathered: Buffer<f32>,
    logits: Buffer<f32>,
    stream: Cell<usize>,
}
/// Only immutable head weights/encoding are shared. Pack/scatter storage is
/// private to the pair executor and outlives every graph using its addresses.
pub(crate) struct PairedHead {
    cuda:Cuda,
    input:Buffer<bf16>,
    output:Buffer<f32>,
    owners:[[u64;2];2],
    stream:usize,
}
impl PairedHead {
    pub(crate) fn new(a:&ModelIo,b:&ModelIo,stream:&Stream)->Result<Self> {
        ensure!(a.rows==6&&b.rows==6&&!a.input_only&&!b.input_only&&a.rank==b.rank
            &&a.shares_lossless_head(b)&&a.cuda.owns_stream(stream),
            "paired head requires independent M6 outputs and one immutable encoding");
        ensure!(a.head.output().ptr()!=b.head.output().ptr()&&a.normalized.ptr()!=b.normalized.ptr(),
            "paired head must not alias mutable owner buffers");
        Ok(Self{cuda:a.cuda.clone(),input:a.cuda.allocate(12*5120)?,output:a.cuda.allocate(12*32320)?,
            owners:[[a.normalized.ptr(),a.head.output().ptr()],[b.normalized.ptr(),b.head.output().ptr()]],
            stream:stream.raw()as usize})
    }
    pub(crate) fn run_normalized(&self,a:&ModelIo,a_live:usize,b:&ModelIo,b_live:usize,stream:&Stream)->Result<()> {
        ensure!(stream.raw()as usize==self.stream&&self.cuda.owns_stream(stream)
            &&self.owners==[[a.normalized.ptr(),a.head.output().ptr()],[b.normalized.ptr(),b.head.output().ptr()]]
            &&a.shares_lossless_head(b)&&matches!(a_live,3|6)&&matches!(b_live,3|6),
            "paired head changed owner/storage/context/live rows");
        unsafe{
            self.cuda.copy_device(self.input.ptr(),a.normalized.ptr(),a_live*5120*2,stream)?;
            self.cuda.copy_device(self.input.ptr()+(a_live*5120*2)as u64,b.normalized.ptr(),b_live*5120*2,stream)?;
        }
        a.head.run_pooled_lossless(&self.input,&self.output,a_live+b_live,stream)?;
        unsafe{
            self.cuda.copy_device(a.head.output().ptr(),self.output.ptr(),a_live*32320*4,stream)?;
            self.cuda.copy_device(b.head.output().ptr(),self.output.ptr()+(a_live*32320*4)as u64,b_live*32320*4,stream)?;
        }
        Ok(())
    }
}
impl ModelIo {
    pub(crate) fn core_module(&self)->&Module {&self.module}
    pub(crate) fn matches_profile(&self, rank: usize, rows: usize, cuda: &Cuda) -> bool {
        self.rank == rank && self.rows == rows && self.weights.in_context(cuda)
    }
    pub fn new(
        cuda: &Cuda,
        core: &[u8],
        stream: &Stream,
        cp: &Checkpoint,
        plan: &RankPlan,
        stage: &Path,
        kernels: &Path,
        rows: usize,
    ) -> Result<Self> {
        ensure!(
            ((1..=8).contains(&rows) || crate::config::target_rows(rows))
                && plan.world == 4
                && plan.rank < 4
                && cuda.owns_stream(stream),
            "model IO target geometry"
        );
        let weights = Rc::new(Weights::load_staged(
            cuda,
            cp,
            plan,
            &["embed.", "head.", "norm."],
            stage,
        )?);
        Self::from_weights(cuda, core, stream, weights, plan.rank, kernels, rows,Rc::new(RefCell::new(None)))
    }
    pub(crate) fn fork_profile(
        &self,
        cuda: &Cuda,
        core: &[u8],
        stream: &Stream,
        kernels: &Path,
        rows: usize,
    ) -> Result<Self> {
        let mut profile = Self::from_weights(
            cuda,
            core,
            stream,
            self.weights.clone(),
            self.rank,
            kernels,
            rows,
            self.lossless_atlas.clone(),
        )?;
        profile.media = self.media.clone();
        Ok(profile)
    }
    pub(crate) fn fork_owner(&self,cuda:&Cuda,core:&[u8],stream:&Stream,
        kernels:&Path,rows:usize)->Result<Self> {
        Self::from_weights(cuda,core,stream,self.weights.clone(),self.rank,kernels,rows,self.lossless_atlas.clone())
    }
    pub(crate) fn shares_media_owner(&self,other:&Self)->bool {
        Rc::ptr_eq(&self.media,&other.media)
    }
    pub(crate) fn shares_atlas(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.weights, &other.weights)
    }
    pub(crate) fn shares_lossless_head(&self,other:&Self)->bool {
        self.shares_atlas(other)&&self.head.shares_lossless_head(&other.head)
    }
    #[allow(clippy::too_many_arguments)]
    fn from_weights(
        cuda: &Cuda,
        core: &[u8],
        stream: &Stream,
        weights: Rc<Weights>,
        rank: usize,
        kernels: &Path,
        rows: usize,
        lossless_atlas: crate::bf16_linear::LosslessHeadAtlas,
    ) -> Result<Self> {
        ensure!(
            ((1..=8).contains(&rows) || crate::config::target_rows(rows))
                && rank < 4
                && weights.in_context(cuda),
            "IO shared profile context/geometry"
        );
        let input_only = crate::config::wide_rows(rows);
        let head_rows = if input_only { 1 } else { rows };
        for name in ["embed.weight", "head.weight"] {
            let v = weights.view(name)?;
            ensure!(
                v.dtype == Dtype::Bf16 && v.shape == [32320, 5120],
                "vocabulary TP shard shape/precision"
            );
        }
        let norm = weights.view("norm.weight")?;
        ensure!(
            norm.dtype == Dtype::Bf16 && norm.shape == [5120],
            "final norm shape/precision"
        );
        // Final model normalization uses the separate reference HC combine
        // and CuTe RMSNorm. The sublayer fused Triton norm has a different
        // reduction tree; a one-ULP normalized error can perturb many logits.
        let norm = Native::open(cuda, &kernels.join("prelude-v1"))?;
        stream.join_default()?;
        let norm_weight = cuda.allocate::<bf16>(5120)?;
        unsafe {
            cuda.copy_device(
                norm_weight.ptr(),
                weights.ptr("norm.weight")?,
                norm_weight.bytes(),
                stream,
            )?;
        }
        let blas = Rc::new(Blas::new_on(cuda,stream)?);
        // Vocabulary output remains private even when a sequential wide
        // profile borrows an embedding arena. A later prefill cannot clobber
        // an earlier request's terminal logits or pending sampling input.
        let (mut head,mut last_head)=cuda.without_row_arena(||->Result<_>{Ok((
            Projection::new(cuda, weights.clone(), blas.clone(), "head", head_rows)?,
            Projection::new(cuda, weights.clone(), blas, "head", 1)?))})?;
        if std::env::var("CANADA_SCRATCH_HEAD_GEMV").as_deref() == Ok("1") {
            let root = kernels.join("target-head-gemv-v1");
            head = head.with_head_gemv(&root)?;
            last_head = last_head.with_head_gemv(&root)?;
        }
        if head_rows==16 && std::env::var("CANADA_SCRATCH_HEAD_TILED16").as_deref()==Ok("1") {
            head=head.with_head_tiled16(&kernels.join("target-head-tiled-m16-v1"))?;
        }
        if head_rows==6 && std::env::var("CANADA_SCRATCH_LOSSLESS_HEAD").as_deref()==Ok("1") {
            let root=kernels.join("target-head-lossless-v1");
            head=if std::env::var("CANADA_SCRATCH_SHARED_TARGET_HEAD").as_deref()==Ok("1") {
                head.with_shared_lossless_head(&root,stream,&lossless_atlas)?
            }else{head.with_lossless_head(&root,stream)?};
        }
        Ok(Self {
            cuda: cuda.clone(),
            module: cuda.module(core)?,
            weights,
            lossless_atlas,
            media: Rc::new(RefCell::new(None)),
            rows,
            input_only,
            rank,
            local_embedding: cuda.allocate_rows(rows * 5120)?,
            embedding: cuda.allocate_rows(rows * 5120)?,
            expanded: cuda.allocate_rows(rows * 20480)?,
            combined: cuda.allocate(head_rows * 5120)?,
            norm,
            norm_weight,
            normalized: cuda.allocate(head_rows * 5120)?,
            head,
            last_head,
            last_input: cuda.allocate(5120)?,
            last_gathered: cuda.allocate(129280)?,
            logits: cuda.allocate(head_rows * 129280)?,
            stream: Cell::new(0),
        })
    }
    pub(crate) fn head_gemv_enabled(&self) -> bool {
        self.head.uses_head_gemv() && self.last_head.uses_head_gemv()
    }
    pub(crate) fn set_lossless_head_for_probe(&self,active:bool)->Result<()> {
        self.head.set_lossless_for_probe(active)
    }
    pub(crate) fn with_draft_head_gemv(mut self, root: &Path) -> Result<Self> {
        ensure!(self.rows == 5 && !self.input_only, "M5 drafter IO profile required");
        self.head = self.head.with_draft_head_gemv(root)?;
        Ok(self)
    }
    pub(crate) fn draft_head_gemv_enabled(&self) -> bool {
        self.rows == 5 && self.head.uses_head_gemv()
    }
    pub(crate) fn set_draft_head_gemv_for_probe(&self, active: bool) -> Result<()> {
        self.head.set_draft_head_gemv_for_probe(active)
    }
    fn bind(&self, stream: &Stream) -> Result<()> {
        ensure!(self.cuda.owns_stream(stream), "model IO stream context");
        let raw = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == raw,
            "model IO needs one ordered stream"
        );
        self.stream.set(raw);
        Ok(())
    }
    pub(crate) fn set_prompt_media(&self, media: Option<Rc<crate::prompt_media::Bindings>>, stream: &Stream) -> Result<()> {
        self.bind(stream)?;
        ensure!(media.as_ref().is_none_or(|m| m.in_context(&self.cuda)), "image prompt context");
        stream.synchronize()?;
        *self.media.borrow_mut() = media;
        Ok(())
    }
    /// Host preparation precedes the existing join-default dependency and any
    /// graph replay. All profiles share the prompt, but own stable row pointers.
    pub(crate) fn update_tokens(&self, tokens: &mut TokenRows, start: usize, ids: &[i64], live: usize) -> Result<()> {
        ensure!(self.cuda.owns_buffer(tokens.ids()), "image token context");
        tokens.update(ids, live)?;
        if let Some(media) = self.media.borrow().as_ref() {
            tokens.bind_images(media.clone(), start, ids, live)?;
        }
        Ok(())
    }
    /// Opt-in post-execution byte audit. Compare the actual model input with
    /// the immutable GPU image spans; no CPU model/reference arithmetic.
    pub(crate) fn audit_input_images(&self, tokens: &TokenRows, start: usize, ids: &[u32],
        stream: &Stream) -> Result<()> {
        if std::env::var("CANADA_SCRATCH_IMAGE_INPUT_AUDIT").as_deref() != Ok("1") {return Ok(());}
        let media = self.media.borrow();
        let Some(media) = media.as_ref() else {return Ok(());};
        let ids = ids.iter().copied().map(i64::from).collect::<Vec<_>>();
        let Some(expected) = media.row_pointers(start, &ids, ids.len())? else {return Ok(());};
        stream.synchronize()?;
        let actual = tokens.image_pointers().download()?;
        ensure!(actual[..ids.len()] == expected, "image row pointer audit failed");
        let mut rows = 0;
        for placement in media.identity().placements() {
            let first = start.max(placement.start);
            let end = (start+ids.len()).min(placement.start+placement.tokens);
            if end <= first {continue;}
            let count = end-first;
            let original = unsafe {self.cuda.read_device(expected[first-start], count*5120*2)}?;
            let expanded = unsafe {self.cuda.read_device(self.expanded.ptr()+((first-start)*20480*2)as u64,
                count*20480*2)}?;
            for row in 0..count {
                for hc in 0..4 {
                    ensure!(expanded[(row*4+hc)*10240..(row*4+hc+1)*10240]
                        == original[row*10240..(row+1)*10240],
                        "image model input bytes differ at absolute row {} HC{}", first+row, hc);
                }
            }
            rows += count;
        }
        eprintln!("image_input_audit rank={} start={} live={} image_rows={} exact=true",
            self.rank,start,ids.len(),rows);
        Ok(())
    }
    pub fn run_input(&mut self, tokens: &TokenRows, comm: &Roce, stream: &Stream) -> Result<()> {
        self.bind(stream)?;
        ensure!(
            comm.rank() == self.rank,
            "embedding transport rank mismatch"
        );
        ensure!(
            self.cuda.owns_buffer(tokens.ids()) && tokens.ids().len() == self.rows,
            "embedding token context/shape"
        );
        let mut ids = tokens.ids().ptr();
        let mut weight = self.weights.ptr("embed.weight")?;
        let mut out = self.local_embedding.ptr();
        let mut rows = self.rows as i32;
        let mut lower = (self.rank * 32320) as i32;
        let mut width = 32320i32;
        let mut args = [
            (&mut ids as *mut u64).cast(),
            (&mut weight as *mut u64).cast(),
            (&mut out as *mut u64).cast(),
            (&mut rows as *mut i32).cast(),
            (&mut lower as *mut i32).cast(),
            (&mut width as *mut i32).cast(),
        ];
        unsafe {
            self.module.function("ca_vocab_embedding")?.launch(
                [((self.rows * 5120).div_ceil(256)) as u32, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )?;
            comm.sum_bf16(&self.local_embedding, &mut self.embedding, stream)?;
        }
        let mut input = self.embedding.ptr();
        let mut output = self.expanded.ptr();
        let mut image_rows = tokens.image_pointers().ptr();
        let mut args = [
            (&mut input as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
            (&mut rows as *mut i32).cast(),
            (&mut image_rows as *mut u64).cast(),
        ];
        unsafe {
            self.module.function("ca_expand_hc_embedding")?.launch(
                [((self.rows * 20480).div_ceil(256)) as u32, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )
        }
    }
    pub fn run_head(
        &mut self,
        residual: &Buffer<bf16>,
        pre: &Buffer<f32>,
        stream: &Stream,
    ) -> Result<()> {
        self.prepare_head(residual, pre, stream)?;
        self.head.run(&self.normalized, stream)
    }
    /// DSpark supplies its own final-stage normalization while reusing the
    /// canonical target vocabulary atlas. No duplicate head/embedding weights.
    pub(crate) fn run_normalized_head(
        &self,
        normalized: &Buffer<bf16>,
        stream: &Stream,
    ) -> Result<()> {
        self.bind(stream)?;
        ensure!(!self.input_only, "wide encoder IO has no vocabulary head");
        self.head.run(normalized, stream)
    }
    pub(crate) fn make_private_draft_head(
        &self,
        root: &Path,
        stream: &Stream,
    ) -> Result<crate::draft_head::DraftHead> {
        self.bind(stream)?;
        ensure!(
            self.rows == 5 && !self.input_only,
            "private head needs DSpark M5 profile"
        );
        crate::draft_head::DraftHead::new(&self.cuda, self.weights.clone(), root, stream)
    }
    pub(crate) fn run_private_draft_head(
        &self,
        draft: &crate::draft_head::DraftHead,
        normalized: &Buffer<bf16>,
        stream: &Stream,
    ) -> Result<()> {
        self.bind(stream)?;
        ensure!(
            self.rows == 5 && !self.input_only && draft.shares_source(&self.weights),
            "private head is bound to a different DSpark atlas/profile"
        );
        draft.run(normalized, self.head.output(), stream)
    }
    /// Prefill needs only the last live next-token distribution. Row pruning
    /// avoids unused vocabulary projections while retaining FP32 head output.
    pub fn run_head_last(
        &mut self,
        residual: &Buffer<bf16>,
        pre: &Buffer<f32>,
        tokens: &TokenRows,
        stream: &Stream,
    ) -> Result<()> {
        self.prepare_head(residual, pre, stream)?;
        ensure!(
            self.cuda.owns_buffer(tokens.live()) && tokens.ids().len() == self.rows,
            "prefill head live-row metadata"
        );
        let mut input = self.normalized.ptr();
        let mut live = tokens.live().ptr();
        let mut output = self.last_input.ptr();
        let mut rows = self.rows as i32;
        let mut args = [
            (&mut input as *mut u64).cast(),
            (&mut live as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
            (&mut rows as *mut i32).cast(),
        ];
        unsafe {
            self.module.function("ca_select_last_head_row")?.launch(
                [20, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )?;
        }
        self.last_head.run(&self.last_input, stream)
    }
    pub(crate) fn prepare_head(
        &mut self,
        residual: &Buffer<bf16>,
        pre: &Buffer<f32>,
        stream: &Stream,
    ) -> Result<()> {
        self.bind(stream)?;
        ensure!(!self.input_only, "wide encoder IO has no vocabulary head");
        ensure!(
            self.cuda.owns_buffer(residual)
                && self.cuda.owns_buffer(pre)
                && residual.len() == self.rows * 20480
                && pre.len() == self.rows * 4,
            "final model boundary shape/context"
        );
        let mut input = residual.ptr();
        let mut coefficients = pre.ptr();
        let mut output = self.combined.ptr();
        let mut args = [
            (&mut input as *mut u64).cast(),
            (&mut coefficients as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
        ];
        unsafe {
            self.module.function("ca_final_hc_combine")?.launch(
                [self.rows as u32, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )?;
        }
        self.norm.rms(
            &self.combined,
            &self.norm_weight,
            &mut self.normalized,
            self.rows,
            5120,
            stream,
        )?;
        Ok(())
    }
    pub fn gather_logits_last(
        &mut self,
        comm: &Roce,
        tokens: &TokenRows,
        stream: &Stream,
    ) -> Result<()> {
        self.bind(stream)?;
        ensure!(!self.input_only, "wide encoder IO has no vocabulary head");
        ensure!(
            comm.rank() == self.rank
                && self.cuda.owns_buffer(tokens.live())
                && tokens.ids().len() == self.rows,
            "prefill logit transport/live-row context"
        );
        unsafe {
            comm.gather_f32(
                self.last_head.output(),
                &mut self.last_gathered,
                32320,
                stream,
            )?;
            self.cuda
                .zero_device(self.logits.ptr(), self.logits.bytes(), stream)?;
        }
        let mut input = self.last_gathered.ptr();
        let mut live = tokens.live().ptr();
        let mut output = self.logits.ptr();
        let mut rows = self.rows as i32;
        let mut args = [
            (&mut input as *mut u64).cast(),
            (&mut live as *mut u64).cast(),
            (&mut output as *mut u64).cast(),
            (&mut rows as *mut i32).cast(),
        ];
        unsafe {
            self.module.function("ca_publish_last_head_logits")?.launch(
                [505, 1, 1],
                [256, 1, 1],
                0,
                stream,
                &mut args,
            )
        }
    }
    pub fn gather_logits(&mut self, comm: &Roce, stream: &Stream) -> Result<()> {
        self.bind(stream)?;
        ensure!(!self.input_only, "wide encoder IO has no vocabulary head");
        ensure!(comm.rank() == self.rank, "logit transport rank mismatch");
        unsafe { comm.gather_f32(self.head.output(), &mut self.logits, 32320, stream) }
    }
    pub fn input(&self) -> &Buffer<bf16> {
        &self.expanded
    }
    pub fn local_embedding(&self) -> &Buffer<bf16> {
        &self.local_embedding
    }
    pub fn embedding(&self) -> &Buffer<bf16> {
        &self.embedding
    }
    pub fn normalized(&self) -> &Buffer<bf16> {
        &self.normalized
    }
    pub fn local_logits(&self) -> &Buffer<f32> {
        self.head.output()
    }
    pub fn logits(&self) -> &Buffer<f32> {
        &self.logits
    }
}
