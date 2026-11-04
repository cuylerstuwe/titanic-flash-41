//! Canonical HC=4 residual boundary. Rust owns coefficient/scratch/output state;
//! AOT low-level kernels preserve the reference's BF16 and FMA boundaries.
use crate::{
    checkpoint::Dtype,
    cuda::{Buffer, Cuda, Event, Stream},
    prelude::Native,
    primitives::Primitives,
    triton_ops::Operator,
    weights::Weights,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{cell::Cell, path::Path, rc::Rc};

thread_local! {
    static ACTIVE_ROWS:Cell<Option<(usize,usize,usize,bool)>>=const{Cell::new(None)};
}
pub(crate) struct ActiveRowsScope(std::marker::PhantomData<Rc<()>>);
impl ActiveRowsScope {
    pub(crate) fn new(enabled:bool,live:usize,capacity:usize,stream:&Stream,normalize:bool)->Result<Self> {
        ACTIVE_ROWS.with(|slot|->Result<()>{
            ensure!(slot.get().is_none(),"nested mHC live-row scope");
            if enabled {
                ensure!(capacity==6&&(1..=6).contains(&live)
                    ||capacity==128&&(17..=64).contains(&live)
                    ||capacity==512&&(129..512).contains(&live),"mHC live-row geometry");
                ensure!(!normalize||capacity>=128,"live normalization is prefill-only");
                slot.set(Some((live,capacity,stream.raw()as usize,normalize)));
            }
            Ok(())
        })?;Ok(Self(std::marker::PhantomData))
    }
}
impl Drop for ActiveRowsScope {
    fn drop(&mut self){ACTIVE_ROWS.with(|s|s.set(None));}
}

pub struct Boundary {
    cuda: Cuda,
    weights: Rc<Weights>,
    rows: usize,
    fn_pointer: u64,
    base_pointer: u64,
    scale_pointer: u64,
    norm: Native,
    plain_norm: Option<Native>,
    combined: Option<Buffer<bf16>>,
    norm_weight: Buffer<bf16>,
    stats: Operator,
    stats_two_mma: Option<Operator>,
    use_two_mma: bool,
    reduce: Operator,
    combine: Operator,
    fused_norm: Option<Primitives>,
    use_fused_norm: bool,
    post_op: Operator,
    partial_mix: Buffer<f32>,
    partial_square: Buffer<f32>,
    pre: Buffer<f32>,
    post: Buffer<f32>,
    comb: Buffer<f32>,
    normalized: Buffer<bf16>,
    output: Buffer<bf16>,
    stream: Cell<usize>,
    prepared: Cell<u64>,
    prepared_rows: Cell<usize>,
    prepared_pre: Cell<u64>,
    deferred_stats: Option<DeferredStats>,
    stats_pending: Cell<bool>,
    full_stats: Option<FullStats>,
    use_full_stats: bool,
    norm_quant: Option<NormQuant>,
    norm_quant_active: bool,
    quant_ready: bool,
    pdl_pair: bool,
    alternate_pdl_pair: Option<(Operator, Operator)>,
    pdl_stats: bool,
    alternate_pdl_stats: Option<(Operator, Operator)>,
}
/// Keep both canonical BF16 routing inputs and the projection's exact native
/// MXFP8 representation. These are transient, profile-owned graph buffers.
struct NormQuant {
    op: Operator,
    values: Buffer<u8>,
    row_scales: Buffer<u8>,
    mma_scales: Buffer<u8>,
}
struct FullStats {
    stats: Operator,
    reduce: Operator,
    mix: Buffer<f32>,
    square: Buffer<f32>,
}
struct DeferredStats {
    side: Rc<Stream>,
    parent: usize,
    ready: Event,
    done: Event,
}
pub(crate) fn overlap_enabled() -> bool {
    std::env::var("CANADA_SCRATCH_MHC_OVERLAP").as_deref() == Ok("1")
}
pub(crate) fn pdl_pair_requested() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_MHC_PDL_PAIR") {
        Ok(value) => { ensure!(value == "0" || value == "1", "invalid mHC PDL pair policy"); Ok(value == "1") },
        Err(std::env::VarError::NotPresent) => Ok(false), Err(error) => Err(error.into()),
    }
}
pub(crate) fn pdl_stats_requested() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_MHC_PDL_STATS") {
        Ok(value) => { ensure!(value == "0" || value == "1", "invalid mHC PDL stats policy"); Ok(value == "1") },
        Err(std::env::VarError::NotPresent) => Ok(false), Err(error) => Err(error.into()),
    }
}
fn wide_flag(name:&str)->Result<bool> {
    match std::env::var(name) {
        Ok(v)=>{ensure!(v=="0"||v=="1","invalid {name}");Ok(v=="1")},
        Err(std::env::VarError::NotPresent)=>Ok(false),Err(e)=>Err(e.into()),
    }
}
pub(crate) fn wide_norm_quant_enabled()->Result<bool>{wide_flag("CANADA_SCRATCH_MHC_NORM_QUANT16")}
pub(crate) fn wide_stats_pdl_enabled()->Result<bool>{wide_flag("CANADA_SCRATCH_MHC_STATS_PDL16")}
pub(crate) fn attention16_overlap_enabled() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_ATTENTION16_OVERLAP") {
        Ok(value) => {
            ensure!(value == "0" || value == "1", "invalid M16 attention overlap policy");
            Ok(value == "1")
        }
        Err(std::env::VarError::NotPresent) => Ok(true),
        Err(error) => Err(error.into()),
    }
}
/// Explicit opt-in for compact M16 attention after the dense pipeline fix.
/// Older packages keep the original serial guard and cannot select this by
/// setting the old overlap flag alone.
pub(crate) fn compact16_overlap_enabled() -> Result<bool> {
    let enabled=match std::env::var("CANADA_SCRATCH_ATTENTION16_COMPACT_OVERLAP") {
        Ok(v)=>{ensure!(v=="0"||v=="1","invalid compact M16 overlap policy");v=="1"},
        Err(std::env::VarError::NotPresent)=>false,Err(e)=>return Err(e.into()),
    };
    if enabled {
        ensure!(std::env::var("CANADA_SCRATCH_DENSE_PIPELINE_FENCE").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_GROUPED_SPARSE16").as_deref()==Ok("1")
            &&attention16_overlap_enabled()?,
            "compact M16 overlap requires the explicit pipeline fence and compact indexing");
    }
    Ok(enabled)
}
pub(crate) fn prefill_norm_fusion_enabled() -> Result<bool> {
    match std::env::var("CANADA_SCRATCH_PREFILL_NORM_FUSION") {
        Ok(value) => {
            ensure!(value == "0" || value == "1", "invalid prefill norm fusion policy");
            Ok(value == "1")
        }
        Err(std::env::VarError::NotPresent) => Ok(true),
        Err(error) => Err(error.into()),
    }
}
fn abi(op: &Operator, spec: &[(&str, &str)]) -> Result<()> {
    ensure!(
        op.entry
            .arguments
            .iter()
            .map(|a| (a.name.as_str(), a.kind.as_str()))
            .eq(spec.iter().copied()),
        "mHC primitive ABI drift: {}",
        op.entry.name
    );
    Ok(())
}
impl Boundary {
    pub(crate) fn set_wide_epilogues_for_probe(&mut self,enabled:bool,kernels:&Path)->Result<()> {
        ensure!([16,32].contains(&self.rows)
            &&std::env::var("CANADA_SCRATCH_WIDE_EPILOGUE_SCREEN").as_deref()==Ok("1"),
            "isolated wide residual update screen");
        if self.rows==32 {
            self.post_op=if enabled {Operator::load(&self.cuda,&kernels.join("mhc-post-tiles-v1"),"hc_post_b128_w1")?}
                else{Operator::load(&self.cuda,&kernels.join("mhc-post-v1"),"mhc_post")?};
        }
        Ok(())
    }
    pub(crate) fn enable_verify16_overlap_before_capture(&mut self,stream:&Stream)->Result<()> {
        ensure!(crate::ffn::wide_overlap_enabled(self.rows)?&&self.prepared.get()==0
            &&self.deferred_stats.is_none()&&!self.stats_pending.get()&&self.cuda.owns_stream(stream),
            "uncaptured M16 FFN coefficient overlap admission");
        self.deferred_stats=Some(DeferredStats {side:self.cuda.coefficient_stream(stream)?,
            parent:stream.raw()as usize,ready:self.cuda.fence_event()?,done:self.cuda.fence_event()?});
        Ok(())
    }
    pub(crate) fn set_two_mma_for_probe(&mut self,enabled:bool,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_MHC_TWO_MMA_SCREEN").as_deref()==Ok("1")
            &&[6,16].contains(&self.rows)&&self.stats_two_mma.is_some()
            &&!self.stats_pending.get()&&self.cuda.owns_stream(stream),"idle two-MMA statistics screen");
        self.use_two_mma=enabled;Ok(())
    }
    pub(crate) fn set_verify16_overlap_for_probe(&mut self, enabled: bool, stream: &Stream) -> Result<()> {
        ensure!((crate::ffn::wide_overlap_probe(self.rows)
            || ((std::env::var("CANADA_SCRATCH_VERIFY16_OVERLAP_SCREEN").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_GROUPED16_SERIAL_ATTENTION").as_deref()==Ok("1")
                || std::env::var("CANADA_SCRATCH_QKV_SCHEDULE_SCREEN").as_deref()==Ok("1"))
                && self.rows==16)) && !self.stats_pending.get() && self.cuda.owns_stream(stream),
            "idle wide coefficient schedule screen");
        if enabled && self.deferred_stats.is_none() {
            self.deferred_stats=Some(DeferredStats { side:self.cuda.coefficient_stream(stream)?,
                parent:stream.raw()as usize,ready:self.cuda.fence_event()?,done:self.cuda.fence_event()? });
        }
        if !enabled { self.deferred_stats=None; }
        Ok(())
    }
    pub(crate) fn enable_verify16_overlap_for_probe(&mut self, stream: &Stream) -> Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_VERIFY16_OVERLAP_SCREEN").as_deref()==Ok("1")
            && self.rows==16 && self.prepared.get()==0 && self.deferred_stats.is_none()
            && self.cuda.owns_stream(stream),"isolated uncaptured M16 coefficient overlap admission");
        self.deferred_stats=Some(DeferredStats { side:self.cuda.coefficient_stream(stream)?,
            parent:stream.raw()as usize,ready:self.cuda.fence_event()?,done:self.cuda.fence_event()? });
        Ok(())
    }
    pub fn new(
        cuda: &Cuda,
        stream: &Stream,
        weights: Rc<Weights>,
        layer: usize,
        branch: &str,
        rows: usize,
        kernels: &Path,
    ) -> Result<Self> {
        Self::new_named(
            cuda,
            stream,
            weights,
            crate::layer_id::LayerId::target(layer)?,
            branch,
            rows,
            kernels,
        )
    }
    pub fn new_named(
        cuda: &Cuda,
        stream: &Stream,
        weights: Rc<Weights>,
        id: crate::layer_id::LayerId,
        branch: &str,
        rows: usize,
        kernels: &Path,
    ) -> Result<Self> {
        ensure!(
            ["attn", "ffn"].contains(&branch)
                && crate::config::mhc_rows(rows)
                && weights.in_context(cuda)
                && cuda.owns_stream(stream),
            "mHC branch/context/geometry"
        );
        stream.join_default()?;
        let prefix = format!("{}.hc_{branch}", id.prefix());
        for (suffix, shape) in [
            ("fn", vec![24, 20480]),
            ("base", vec![24]),
            ("scale", vec![3]),
        ] {
            let v = weights.view(&format!("{prefix}_{suffix}"))?;
            ensure!(
                v.dtype == Dtype::F32 && v.shape == shape,
                "canonical mHC coefficient shape/dtype"
            );
        }
        let norm_name = format!("{}.{branch}_norm.weight", id.prefix());
        let nw = weights.view(&norm_name)?;
        ensure!(
            nw.dtype == Dtype::Bf16 && nw.shape == [5120],
            "canonical mHC norm shape/dtype"
        );
        let norm_weight = cuda.allocate::<bf16>(5120)?;
        unsafe {
            cuda.copy_device(
                norm_weight.ptr(),
                weights.ptr(&norm_name)?,
                norm_weight.bytes(),
                stream,
            )
        }?;
        let root = kernels.join("triton-hc-v2");
        let mut stats = Operator::load(
            cuda,
            &root,
            if rows <= 8 {
                "hc_stats_mb8_s2"
            } else if rows < 64 {
                "hc_stats_mb16_s2"
            } else {
                "hc_stats_mb16_s1"
            },
        )?;
        let mut reduce = Operator::load(cuda, &root, "hc_reduce80")?;
        let two_mma_requested=match std::env::var("CANADA_SCRATCH_MHC_TWO_MMA") {
            Ok(v)=>{ensure!(v=="0"||v=="1","invalid two-MMA policy");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(e)=>return Err(e.into()),
        };
        let eligible=[6,16].contains(&rows)&&!id.is_draft();
        let stats_two_mma=if eligible&&(two_mma_requested
            ||std::env::var("CANADA_SCRATCH_MHC_TWO_MMA_SCREEN").as_deref()==Ok("1")) {
            let root=if std::env::var("CANADA_SCRATCH_MHC_TWO_MMA_STAGE1").as_deref()==Ok("1")
                {"mhc-small-stats-two-mma-stage1-s806"}else{"mhc-small-stats-two-mma-s806"};
            let op=Operator::load(cuda,&kernels.join(root),
                &format!("hc_stats_two_mma_m{rows}"))?;
            abi(&op,&[("X","*bf16"),("W","*fp32"),("PM","*fp32"),("PS","*fp32")])?;
            ensure!(op.entry.grid.rows_div==rows as u32&&op.entry.grid.y==80,
                "two-MMA physical row specialization");
            Some(op)
        }else{None};
        let combine = if rows <= 8 {
            Operator::load(cuda, &root, "hc_combine_norm_small")?
        } else {
            Operator::load(
                cuda,
                &kernels.join("mhc-combine-plain-v1"),
                "hc_combine_plain",
            )?
        };
        let small16 = rows==16
            && std::env::var("CANADA_SCRATCH_VERIFY16_SMALL_KERNELS").as_deref()==Ok("1");
        let post_tiles = (rows == 32 && !id.is_draft() && crate::layer_attention::wide_epilogues_enabled()?)
            || (rows == 6 || small16) && !id.is_draft()
            && match std::env::var("CANADA_SCRATCH_MHC_POST_TILES") {
                Ok(v) => { ensure!(v == "0" || v == "1", "invalid mHC post tile policy"); v == "1" },
                Err(std::env::VarError::NotPresent) => false,
                Err(e) => return Err(e.into()),
            };
        let mut post_op = if post_tiles {
            Operator::load(cuda, &kernels.join("mhc-post-tiles-v1"), "hc_post_b128_w1")?
        } else {
            Operator::load(cuda, &kernels.join("mhc-post-v1"), "mhc_post")?
        };
        let fusion_ab = std::env::var("CANADA_SCRATCH_PREFILL_NORM_FUSION_AB").as_deref() == Ok("1");
        let use_fused_norm = rows >= 128 && prefill_norm_fusion_enabled()? && !fusion_ab;
        let fused_norm = if rows >= 128 && (use_fused_norm || fusion_ab)
        {
            let root = kernels.join("mhc-combine-norm-cute-v3");
            let op = Primitives::open(cuda, &root.join("libcanada_primitives.so"), &root.join("bindings.json"))?;
            ensure!(op.bindings.abi == 2 && op.bindings.entries.len() == 1
                && op.bindings.entries[0].kernel_id == "mhc.combine_rms.h5120"
                && op.bindings.entries[0].arguments.iter().map(|a|a.kind.as_str())
                    .eq(["pointer","pointer","pointer","pointer","i64","f32"]),
                "prefill combine/RMS primitive ABI");
            Some(op)
        } else {
            None
        };
        abi(
            &stats,
            &[
                ("x_ptr", "*bf16"),
                ("w_ptr", "*fp32"),
                ("part_mix_ptr", "*fp32"),
                ("part_sq_ptr", "*fp32"),
                ("M", "i32"),
                ("K", "i32"),
                ("x_stride_m", "i32"),
                ("w_stride_n", "i32"),
            ],
        )?;
        abi(
            &reduce,
            &[
                ("part_mix_ptr", "*fp32"),
                ("part_sq_ptr", "*fp32"),
                ("scale_ptr", "*fp32"),
                ("base_ptr", "*fp32"),
                ("pre_ptr", "*fp32"),
                ("post_ptr", "*fp32"),
                ("comb_ptr", "*fp32"),
                ("m", "i32"),
                ("inv_k", "fp32"),
                ("rms_eps", "fp32"),
            ],
        )?;
        if rows <= 8 {
            abi(
                &combine,
                &[
                    ("X", "*bf16"),
                    ("P", "*fp32"),
                    ("W", "*bf16"),
                    ("Y", "*bf16"),
                ],
            )?;
        } else {
            abi(
                &combine,
                &[
                    ("x_ptr", "*bf16"),
                    ("pre_ptr", "*fp32"),
                    ("y_ptr", "*bf16"),
                    ("H", "i32"),
                    ("x_stride_m", "i32"),
                    ("pre_stride_m", "i32"),
                    ("pre_stride_k", "i32"),
                    ("y_stride_m", "i32"),
                ],
            )?;
        }
        abi(
            &post_op,
            &[
                ("X", "*bf16"),
                ("R", "*bf16"),
                ("P", "*fp32"),
                ("C", "*fp32"),
                ("Y", "*bf16"),
            ],
        )?;
        let norm = Native::open_with_rms(
            cuda,
            &kernels.join("prelude-v1"),
            &kernels.join("mhc-rms-v1"),
        )?;
        let fullk_requested = match std::env::var("CANADA_SCRATCH_HC_FULLK") {
            Ok(value) => {
                ensure!(value == "0" || value == "1", "invalid full-K HC policy");
                value == "1"
            }
            Err(std::env::VarError::NotPresent) => false,
            Err(error) => return Err(error.into()),
        };
        let fullk_ab = std::env::var("CANADA_SCRATCH_HC_FULLK_AB").as_deref() == Ok("1");
        // The component screen found a regression at <=512 rows. Keep those
        // on the original kernels, including every decode/verification profile.
        let use_full_stats = rows >= 1024 && fullk_requested && !fullk_ab;
        let full_stats = if rows >= 1024 && (fullk_requested || fullk_ab) {
            let root = kernels.join("mhc-fullk-v1");
            let stats = Operator::load(cuda, &root, if rows <= 2048 {
                "hc_fullk_mb16_s4"
            } else if rows <= 3072 {
                "hc_fullk_mb16_s3"
            } else {
                "hc_fullk_mb32_s3"
            })?;
            abi(&stats, &[("x_ptr","*bf16"),("w_ptr","*fp32"),
                ("mix_ptr","*fp32"),("sq_ptr","*fp32"),("M","i32"),
                ("x_stride_m","i32"),("w_stride_n","i32")])?;
            let reduced = Operator::load(cuda, &root, "hc_reduce1")?;
            ensure!(reduced.entry.arguments.iter().map(|a|(&a.name,&a.kind))
                .eq(reduce.entry.arguments.iter().map(|a|(&a.name,&a.kind))),
                "one-slice HC reduction ABI drift");
            Some(FullStats { stats, reduce: reduced,
                mix: cuda.allocate_rows(rows * 24)?, square: cuda.allocate_rows(rows)? })
        } else { None };
        let wide_screen = rows == 16 && !id.is_draft()
            && std::env::var("CANADA_SCRATCH_WIDE_MHC_SCREEN").as_deref() == Ok("1");
        let wide_norm = rows == 16 && !id.is_draft() && wide_norm_quant_enabled()?;
        let mut norm_quant = if (rows == 6 && !id.is_draft()
            && std::env::var("CANADA_SCRATCH_NORM_QUANT_FUSION").as_deref() == Ok("1"))
            || wide_norm || wide_screen {
            ensure!(!crate::fp8_linear::narrow_enabled(), "norm/quant fusion requires the checked legacy input quantizer");
            let layout8 = rows == 16 || std::env::var("CANADA_SCRATCH_NORM_QUANT_LAYOUT8").as_deref() == Ok("1");
            let op = Operator::load(cuda,
                &kernels.join(if layout8 {"mhc-norm-quant-layout8-v1"} else {"mhc-norm-quant-v1"}),
                if layout8 {"hc_norm_quant_p2"} else {"hc_norm_quant_p4"})?;
            abi(&op, &[("X","*bf16"),("P","*fp32"),("W","*bf16"),("Y","*bf16"),
                ("Q","*fp8e4nv"),("SR","*u8"),("SM","*u8")])?;
            // The unused MMA tile lanes keep their initialization across
            // executions; they cannot borrow another profile's row arena.
            let mut mma_scales = cuda.allocate(20480)?;
            mma_scales.zero()?;
            stream.join_default()?;
            Some(NormQuant {op, values:cuda.allocate_rows(rows*5120)?,
                row_scales:cuda.allocate_rows(rows*160)?, mma_scales})
        } else {None};
        let pdl_pair = rows == 6 && !id.is_draft() && pdl_pair_requested()?;
        let pdl_screen = rows == 6 && !id.is_draft()
            && std::env::var("CANADA_SCRATCH_MHC_PDL_PAIR_SCREEN").as_deref() == Ok("1");
        let mut alternate_pdl_pair = if pdl_pair || pdl_screen {
            ensure!(post_tiles && norm_quant.is_some()
                && std::env::var("CANADA_SCRATCH_NORM_QUANT_LAYOUT8").as_deref() == Ok("1"),
                "mHC PDL pair requires the qualified M6 tile and layout8 quantizer");
            let root = kernels.join("mhc-pdl-pair-v1");
            let pair = (Operator::load(cuda, &root, "hc_post_b128_w1")?,
                Operator::load(cuda, &root, "hc_norm_quant_p2")?);
            for (candidate, control) in [(&pair.0, &post_op), (&pair.1, &norm_quant.as_ref().unwrap().op)] {
                ensure!(candidate.entry.programmatic_stream_serialization
                    && candidate.entry.arguments.iter().map(|a| (&a.name, &a.kind))
                        .eq(control.entry.arguments.iter().map(|a| (&a.name, &a.kind))),
                    "mHC PDL operator contract mismatch");
            }
            Some(pair)
        } else { None };
        if pdl_pair {
            let pair = alternate_pdl_pair.as_mut().unwrap();
            std::mem::swap(&mut post_op, &mut pair.0);
            std::mem::swap(&mut norm_quant.as_mut().unwrap().op, &mut pair.1);
            if !pdl_screen { alternate_pdl_pair = None; }
        }
        let pdl_stats = !id.is_draft() && ((rows == 6 && pdl_stats_requested()?)
            || (rows == 16 && wide_stats_pdl_enabled()?));
        let mut alternate_pdl_stats = if pdl_stats || pdl_screen || wide_screen {
            ensure!(!two_mma_requested && !use_full_stats,
                "mHC PDL stats requires the original 80-slice statistics path");
            let root = kernels.join(if rows == 16 {"mhc-pdl-stats16-v1"} else {"mhc-pdl-stats-v1"});
            let pair = (Operator::load(cuda, &root, if rows == 16 {"hc_stats_mb16_s2"} else {"hc_stats_mb8_s2"})?,
                Operator::load(cuda, &root, "hc_reduce80")?);
            for (candidate, control) in [(&pair.0, &stats), (&pair.1, &reduce)] {
                ensure!(candidate.entry.programmatic_stream_serialization
                    && candidate.entry.arguments.iter().map(|a| (&a.name, &a.kind))
                        .eq(control.entry.arguments.iter().map(|a| (&a.name, &a.kind))),
                    "mHC PDL statistics contract mismatch");
            }
            Some(pair)
        } else { None };
        if pdl_stats {
            let pair = alternate_pdl_stats.as_mut().unwrap();
            std::mem::swap(&mut stats, &mut pair.0);
            std::mem::swap(&mut reduce, &mut pair.1);
            if !pdl_screen && !wide_screen { alternate_pdl_stats = None; }
        }
        let result = Self {
            pdl_stats,
            alternate_pdl_stats,
            pdl_pair,
            alternate_pdl_pair,
            cuda: cuda.clone(),
            fn_pointer: weights.ptr(&format!("{prefix}_fn"))?,
            base_pointer: weights.ptr(&format!("{prefix}_base"))?,
            scale_pointer: weights.ptr(&format!("{prefix}_scale"))?,
            weights,
            rows,
            norm,
            plain_norm: if rows > 8 {
                Some(Native::open(cuda, &kernels.join("prelude-v1"))?)
            } else {
                None
            },
            combined: if rows > 8 {
                Some(cuda.allocate_rows(rows * 5120)?)
            } else {
                None
            },
            norm_weight,
            stats,
            stats_two_mma,
            use_two_mma:eligible&&two_mma_requested,
            reduce,
            combine,
            fused_norm,
            use_fused_norm,
            post_op,
            partial_mix: cuda.allocate_rows(80 * rows * 24)?,
            partial_square: cuda.allocate_rows(80 * rows)?,
            pre: cuda.allocate_rows(rows * 4)?,
            post: cuda.allocate_rows(rows * 4)?,
            comb: cuda.allocate_rows(rows * 16)?,
            normalized: cuda.allocate_rows(rows * 5120)?,
            output: cuda.allocate_rows(rows * 20480)?,
            stream: Cell::new(0),
            prepared: Cell::new(0),
            prepared_rows: Cell::new(rows),
            prepared_pre: Cell::new(0),
            // s673: the attention branch passed paired logits, abort state and
            // changed-history handoff. The independent M16 FFN path stays off.
            deferred_stats: if overlap_enabled() && !id.is_draft()
                && (rows <= 8 || (rows == 16 && branch == "attn" && attention16_overlap_enabled()?)) {
                Some(DeferredStats {
                    side: cuda.coefficient_stream(stream)?,
                    parent: stream.raw() as usize,
                    ready: cuda.fence_event()?,
                    done: cuda.fence_event()?,
                })
            } else {
                None
            },
            stats_pending: Cell::new(false),
            full_stats,
            use_full_stats,
            norm_quant_active: norm_quant.is_some() && (rows != 16 || wide_norm),
            norm_quant,
            quant_ready: false,
        };
        stream.synchronize()?;
        Ok(result)
    }
    fn guard(&self, residual: &Buffer<bf16>, stream: &Stream) -> Result<()> {
        ensure!(
            residual.len() == self.rows * 20480
                && self.cuda.owns_buffer(residual)
                && self.cuda.owns_stream(stream)
                && self.weights.in_context(&self.cuda),
            "mHC residual context/shape"
        );
        let s = stream.raw() as usize;
        ensure!(
            self.stream.get() == 0 || self.stream.get() == s,
            "mHC requires one ordered stream"
        );
        self.stream.set(s);
        Ok(())
    }
    /// The Target owner admits this switch only with no captured executions.
    pub(crate) fn overlap_active(&self)->bool { self.deferred_stats.is_some() }
    pub(crate) fn set_norm_fusion(&mut self, enabled: bool) -> Result<()> {
        ensure!(self.rows >= 128 && !self.stats_pending.get()
            && (!enabled || self.fused_norm.is_some()), "unavailable or busy prefill norm fusion");
        self.use_fused_norm = enabled;
        Ok(())
    }
    pub(crate) fn fullk_active(&self) -> bool { self.use_full_stats }
    pub(crate) fn norm_quant_enabled(&self) -> bool { self.norm_quant.is_some() && self.norm_quant_active }
    pub(crate) fn set_pdl_pair_for_probe(&mut self, enabled: bool) -> Result<()> {
        ensure!(self.rows == 6 && self.alternate_pdl_pair.is_some()
            && std::env::var("CANADA_SCRATCH_MHC_PDL_PAIR_SCREEN").as_deref() == Ok("1"),
            "mHC PDL swap requires the isolated M6 screen");
        if self.pdl_pair != enabled {
            let pair = self.alternate_pdl_pair.as_mut().unwrap();
            std::mem::swap(&mut self.post_op, &mut pair.0);
            std::mem::swap(&mut self.norm_quant.as_mut().unwrap().op, &mut pair.1);
            self.pdl_pair = enabled;
        }
        Ok(())
    }
    pub(crate) fn set_pdl_stats_for_probe(&mut self, enabled: bool) -> Result<()> {
        ensure!(self.rows == 6 && !self.use_two_mma && !self.use_full_stats
            && self.alternate_pdl_stats.is_some()
            && std::env::var("CANADA_SCRATCH_MHC_PDL_PAIR_SCREEN").as_deref() == Ok("1"),
            "mHC PDL statistics swap requires the isolated M6 screen");
        if self.pdl_stats != enabled {
            let pair = self.alternate_pdl_stats.as_mut().unwrap();
            std::mem::swap(&mut self.stats, &mut pair.0);
            std::mem::swap(&mut self.reduce, &mut pair.1);
            self.pdl_stats = enabled;
        }
        Ok(())
    }
    pub(crate) fn set_wide_mhc_for_probe(&mut self, mode:u8)->Result<()> {
        ensure!(self.rows==16 && mode<=3 && !self.stats_pending.get()
            && self.norm_quant.is_some() && self.alternate_pdl_stats.is_some()
            && std::env::var("CANADA_SCRATCH_WIDE_MHC_SCREEN").as_deref()==Ok("1"),
            "wide mHC comparison requires idle prepared operators");
        let enabled=mode&1!=0;
        if self.pdl_stats!=enabled {
            let pair=self.alternate_pdl_stats.as_mut().unwrap();
            std::mem::swap(&mut self.stats,&mut pair.0);
            std::mem::swap(&mut self.reduce,&mut pair.1);self.pdl_stats=enabled;
        }
        self.norm_quant_active=mode&2!=0;self.quant_ready=false;Ok(())
    }
    pub(crate) fn quantized_input(&self) -> Option<(&Buffer<u8>, &Buffer<u8>)> {
        self.norm_quant.as_ref().filter(|_|self.quant_ready).map(|q|(&q.values,&q.mma_scales))
    }
    pub(crate) fn set_fullk(&mut self, enabled: bool) -> Result<()> {
        ensure!(!self.stats_pending.get(), "full-K policy while statistics pending");
        if self.rows < 1024 {
            ensure!(!self.use_full_stats, "small-row full-K policy");
            return Ok(());
        }
        ensure!(!enabled || self.full_stats.is_some(), "full-K statistics not prepared");
        self.use_full_stats = enabled;
        Ok(())
    }
    /// `previous_pre=None` is the model's initial copy-0 RMS input. Later
    /// branches consume the predecessor's coefficients, not their own pre-mix.
    pub fn prepare(
        &mut self,
        residual: &Buffer<bf16>,
        previous_pre: Option<&Buffer<f32>>,
        stream: &Stream,
    ) -> Result<()> {
        self.guard(residual, stream)?;
        ensure!(
            !self.stats_pending.get(),
            "mHC prepare before prior stats join"
        );
        self.quant_ready = false;
        // Only the dynamic-M stock statistics path may shorten its packed
        // partial stride. The fixed-M two-MMA and full-K paths stay unchanged.
        // A wide profile's M/4 sequence-parallel boundaries do not match its
        // capacity and deliberately keep their existing complete row shard.
        let (live,live_norm)=if [6,128,512].contains(&self.rows)&&!self.use_two_mma&&!self.use_full_stats {
            ACTIVE_ROWS.with(|s|s.get().filter(|(_,capacity,owner,_)|*capacity==self.rows&&*owner==stream.raw()as usize)
                .map_or((self.rows,false),|(rows,_,_,normalize)|(rows,normalize)))
        }else{(self.rows,false)};
        self.prepared_rows.set(live);
        self.prepared_pre.set(previous_pre.map_or(0, |p| p.ptr()));
        if let Some(p) = previous_pre {
            ensure!(
                p.len() == self.rows * 4 && self.cuda.owns_buffer(p),
                "mHC predecessor pre shape/context"
            );
            if let Some(q) = self.norm_quant.as_ref().filter(|_| self.norm_quant_active) {
                unsafe {q.op.launch(self.rows as u32,
                    &[residual.ptr(),p.ptr(),self.norm_weight.ptr(),self.normalized.ptr(),
                      q.values.ptr(),q.row_scales.ptr(),q.mma_scales.ptr()],stream)?;}
                self.quant_ready = true;
            } else if self.use_fused_norm {
                unsafe {
                    self.fused_norm.as_ref().unwrap().launch_words(0,
                        &[residual.ptr(), p.ptr(), self.norm_weight.ptr(), self.normalized.ptr(),
                          (if live_norm{live}else{self.rows})as u64, 1.0e-20f32.to_bits() as u64], stream)?;
                }
            } else if let Some(combined) = self.combined.as_ref() {
                // Qualified9..384-row path rounds the HC combine to BF16
                // before its separate CuTe RMSNorm. Any fused implementation
                // must retain that BF16 rounding boundary explicitly.
                unsafe {
                    self.combine.launch(
                        self.rows as u32,
                        &[
                            residual.ptr(),
                            p.ptr(),
                            combined.ptr(),
                            5120,
                            20480,
                            4,
                            1,
                            5120,
                        ],
                        stream,
                    )?;
                }
                self.plain_norm.as_ref().unwrap().rms(
                    combined,
                    &self.norm_weight,
                    &mut self.normalized,
                    self.rows,
                    5120,
                    stream,
                )?;
            } else {
                unsafe {
                    self.combine.launch(
                        self.rows as u32,
                        &[
                            residual.ptr(),
                            p.ptr(),
                            self.norm_weight.ptr(),
                            self.normalized.ptr(),
                        ],
                        stream,
                    )
                }?;
            }
        } else {
            self.norm.rms(
                residual,
                &self.norm_weight,
                &mut self.normalized,
                self.rows,
                20480,
                stream,
            )?;
        }
        if let Some(branch) = &self.deferred_stats {
            ensure!(
                branch.parent == stream.raw() as usize,
                "mHC stats parent changed"
            );
            // SGLang _hc_combine records the tiny-row fork after normalization.
            // Coefficients consume the residual, independently of attention/FFN.
            branch.ready.record(stream)?;
            self.stats_pending.set(true);
        } else {
            self.run_stats(residual, stream)?;
        }
        self.prepared.set(residual.ptr());
        Ok(())
    }
    fn run_stats(&self, residual: &Buffer<bf16>, stream: &Stream) -> Result<()> {
        let rows=self.prepared_rows.get();
        ensure!((1..=self.rows).contains(&rows),"mHC prepared live-row extent");
        unsafe {
            if self.use_full_stats {
                let full = self.full_stats.as_ref().unwrap();
                full.stats.launch(rows as u32,
                    &[residual.ptr(), self.fn_pointer, full.mix.ptr(), full.square.ptr(),
                      rows as u64, 20480, 20480], stream)?;
                full.reduce.launch(rows as u32,
                    &[full.mix.ptr(), full.square.ptr(), self.scale_pointer, self.base_pointer,
                      self.pre.ptr(), self.post.ptr(), self.comb.ptr(), rows as u64,
                      (1.0f32/20480.0).to_bits() as u64, 1.0e-20f32.to_bits() as u64], stream)?;
                return Ok(());
            }
            if self.use_two_mma {
                self.stats_two_mma.as_ref().unwrap().launch(rows as u32,
                    &[residual.ptr(),self.fn_pointer,self.partial_mix.ptr(),self.partial_square.ptr()],stream)?;
            } else { self.stats.launch(
                self.prepared_rows.get() as u32,
                &[
                    residual.ptr(),
                    self.fn_pointer,
                    self.partial_mix.ptr(),
                    self.partial_square.ptr(),
                    rows as u64,
                    20480,
                    20480,
                    20480,
                ],
                stream,
            )?; }
            self.reduce.launch(
                rows as u32,
                &[
                    self.partial_mix.ptr(),
                    self.partial_square.ptr(),
                    self.scale_pointer,
                    self.base_pointer,
                    self.pre.ptr(),
                    self.post.ptr(),
                    self.comb.ptr(),
                    rows as u64,
                    (1f32 / 20480.).to_bits() as u64,
                    1e-20f32.to_bits() as u64,
                ],
                stream,
            )?;
        }
        Ok(())
    }
    /// Apply the globally reduced sublayer result. The all-reduce must retain
    /// BF16 output before this FMA-ordered residual mix.
    pub fn finish(
        &self,
        sublayer: &Buffer<bf16>,
        residual: &Buffer<bf16>,
        stream: &Stream,
    ) -> Result<()> {
        self.guard(residual, stream)?;
        ensure!(
            self.prepared.get() == residual.ptr()
                && sublayer.len() == self.rows * 5120
                && self.cuda.owns_buffer(sublayer),
            "mHC finish requires matching prepared residual and reduced BF16 sublayer"
        );
        if let Some(branch) = &self.deferred_stats {
            ensure!(
                self.stats_pending.get(),
                "mHC finish without deferred stats fork"
            );
            // Match _hc_mix_stats: issue the main sublayer first, then record
            // independent coefficient work immediately before its consuming join.
            // This preserves the main CUDA-graph chain and all numeric kernels.
            branch.ready.wait(&branch.side)?;
            self.run_stats(residual, &branch.side)?;
            branch.done.record(&branch.side)?;
            branch.done.wait(stream)?;
            self.stats_pending.set(false);
        }
        unsafe {
            self.post_op.launch(
                self.prepared_rows.get() as u32,
                &[
                    sublayer.ptr(),
                    residual.ptr(),
                    self.post.ptr(),
                    self.comb.ptr(),
                    self.output.ptr(),
                ],
                stream,
            )
        }
    }
    /// Read-only diagnostics after a completed verification. Layer-owned inputs
    /// survive the forward; never inject these snapshots into inference.
    pub(crate) fn norm_quant_diagnostic_inputs(&self) -> Result<Vec<(&'static str, Vec<u8>)>> {
        ensure!(self.rows == 6 && self.prepared.get() != 0
            && std::env::var("CANADA_SCRATCH_NORM_QUANT_DIAGNOSTIC").as_deref() == Ok("1"),
            "norm/quant diagnostic admission");
        let mut result = vec![
            ("residual", unsafe {self.cuda.read_device(self.prepared.get(),self.rows*20480*2)}?),
            ("weight", unsafe {self.cuda.read_device(self.norm_weight.ptr(),self.norm_weight.bytes())}?),
        ];
        if self.prepared_pre.get()!=0 {
            result.push(("pre",unsafe{self.cuda.read_device(self.prepared_pre.get(),self.rows*4*4)}?));
        }
        Ok(result)
    }
    pub fn input(&self) -> &Buffer<bf16> {
        &self.normalized
    }
    pub fn pre(&self) -> &Buffer<f32> {
        assert!(
            !self.stats_pending.get(),
            "mHC coefficients require finish/join"
        );
        &self.pre
    }
    pub fn post(&self) -> &Buffer<f32> {
        &self.post
    }
    pub fn comb(&self) -> &Buffer<f32> {
        &self.comb
    }
    pub fn partial_mix(&self) -> &Buffer<f32> {
        &self.partial_mix
    }
    pub fn partial_square(&self) -> &Buffer<f32> {
        &self.partial_square
    }
    pub fn output(&self) -> &Buffer<bf16> {
        &self.output
    }
}
