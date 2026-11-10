//! Independent exact byte permutation for the compact N64 W4A8 kernel ABI.
//! Logical FP4 nibbles and UE8M0 scales are preserved, including K/N64 tails.
//! Format reference: b12x a7d7d29..., Apache-2.0, low-level layout contract.
use crate::cuda::{Cuda, Module};
use anyhow::{Context, Result, ensure};
use std::ffi::c_void;
#[derive(Clone, Debug, serde::Serialize)]
pub struct Prepared {
    pub layer: usize,
    pub experts: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub w13_offset: usize,
    pub w2_offset: usize,
    pub s13_offset: usize,
    pub s2_offset: usize,
    pub resident_bytes: usize,
    pub temporary_bytes: usize,
}
pub fn prepare_layer(
    cuda: &Cuda,
    module: &Module,
    weights: &mut crate::weights::Weights,
    layer: usize,
) -> Result<Prepared> {
    prepare_named(
        cuda,
        module,
        weights,
        crate::layer_id::LayerId::target(layer)?,
    )
}
pub fn prepare_draft(
    cuda: &Cuda,
    module: &Module,
    weights: &mut crate::weights::Weights,
    stage: usize,
) -> Result<Prepared> {
    prepare_named(
        cuda,
        module,
        weights,
        crate::layer_id::LayerId::draft(stage)?,
    )
}
fn prepare_named(
    cuda: &Cuda,
    module: &Module,
    weights: &mut crate::weights::Weights,
    id: crate::layer_id::LayerId,
) -> Result<Prepared> {
    use crate::checkpoint::Dtype;
    use crate::cuda::Stream;
    let layer = id.key();
    let experts = id.experts();
    let prefix = format!("{}.ffn.experts", id.prefix());
    let name = |e: usize, proj: &str, kind: &str| format!("{prefix}.{e}.{proj}.{kind}");
    let up = weights.view(&name(0, "w3", "weight"))?;
    let down = weights.view(&name(0, "w2", "weight"))?;
    ensure!(
        up.dtype == Dtype::PackedFp4 && down.dtype == Dtype::PackedFp4,
        "not canonical FP4 experts"
    );
    let n = usize::try_from(up.shape[0])?;
    let k = usize::try_from(up.shape[1] * 2)?;
    ensure!((n, k) == (576, 5120), "target expert TP geometry mismatch");
    let wb = n * k / 2;
    let sb = n * k / 32;
    let wo = up.offset;
    let doff = down.offset;
    let so = weights.view(&name(0, "w3", "scale"))?.offset;
    let ds = weights.view(&name(0, "w2", "scale"))?.offset;
    for e in 0..experts {
        for (proj, kind, expected, bytes, dtype, shape) in [
            (
                "w3",
                "weight",
                wo + e * 2 * wb,
                wb,
                Dtype::PackedFp4,
                vec![n as u64, k as u64 / 2],
            ),
            (
                "w1",
                "weight",
                wo + (e * 2 + 1) * wb,
                wb,
                Dtype::PackedFp4,
                vec![n as u64, k as u64 / 2],
            ),
            (
                "w2",
                "weight",
                doff + e * wb,
                wb,
                Dtype::PackedFp4,
                vec![k as u64, n as u64 / 2],
            ),
            (
                "w3",
                "scale",
                so + e * 2 * sb,
                sb,
                Dtype::E8m0,
                vec![n as u64, k as u64 / 32],
            ),
            (
                "w1",
                "scale",
                so + (e * 2 + 1) * sb,
                sb,
                Dtype::E8m0,
                vec![n as u64, k as u64 / 32],
            ),
            (
                "w2",
                "scale",
                ds + e * sb,
                sb,
                Dtype::E8m0,
                vec![k as u64, n as u64 / 32],
            ),
        ] {
            let v = weights.view(&name(e, proj, kind))?;
            ensure!(
                v.offset == expected && v.bytes == bytes && v.dtype == dtype && v.shape == shape,
                "expert atlas is not the exact contiguous kernel layout"
            );
        }
    }
    let stream: Stream = cuda.stream()?;
    // Direct checkpoint uploads use the legacy default stream; this separate
    // nonblocking packer must observe their DMA completion before permutation.
    stream.join_default()?;
    let scratch = cuda.allocate::<u8>(wb)?;
    weights.begin_expert_layer(layer)?;
    for e in 0..experts {
        for (proj, rows, cols) in [("w3", n, k), ("w1", n, k), ("w2", k, n)] {
            let wn = name(e, proj, "weight");
            let sn = name(e, proj, "scale");
            // The loader checked scale codes before upload; all numeric
            // preparation stays on the GPU, without synchronous scale reads.
            for (symbol, key, total) in [
                ("ca_expert_compact_words", &wn, wb / 4),
                ("ca_expert_compact_scales", &sn, sb),
            ] {
                let pointer = weights.ptr(key)?;
                let bytes = weights.view(key)?.bytes;
                let mut src = pointer;
                let mut dst = scratch.ptr();
                let mut nr = rows as i32;
                let mut kr = cols as i32;
                let mut group = nr;
                let mut args: Vec<*mut c_void> = vec![
                    (&mut src as *mut u64).cast(),
                    (&mut dst as *mut u64).cast(),
                    (&mut nr as *mut i32).cast(),
                    (&mut kr as *mut i32).cast(),
                    (&mut group as *mut i32).cast(),
                ];
                unsafe {
                    module.function(symbol)?.launch(
                        [total.div_ceil(256) as u32, 1, 1],
                        [256, 1, 1],
                        0,
                        &stream,
                        &mut args,
                    )
                }?;
                unsafe { cuda.copy_device(pointer, scratch.ptr(), bytes, &stream) }?;
            }
        }
    }
    stream.synchronize()?;
    weights.finish_expert_layer(layer)?;
    if layer<40 && std::env::var("CANADA_SCRATCH_MOE_SCALE_CODE_SCREEN").as_deref()==Ok("1") {
        let bytes=2*experts*sb;
        ensure!(cuda.admission_memory()?.admitted_available>=crate::memory::floor_bytes()+bytes,
            "scale-code experiment must preserve the physical memory reserve");
        let mut codes=cuda.allocate::<u8>(bytes)?;
        let mut error=cuda.allocate::<u32>(1)?;
        codes.zero()?;error.zero()?;stream.join_default()?;
        let mut src=weights.pointer_at(so,bytes)?;
        let mut dst=codes.ptr();let mut status=error.ptr();let mut matrices=(experts*2)as i32;
        let mut args:Vec<*mut c_void>=vec![(&mut src as *mut u64).cast(),(&mut dst as *mut u64).cast(),
            (&mut status as *mut u64).cast(),(&mut matrices as *mut i32).cast()];
        unsafe {module.function("ca_expert_scale_codes4")?.launch([72,matrices as u32,1],[256,1,1],0,&stream,&mut args)}?;
        stream.synchronize()?;
        ensure!(error.download()?[0]==0,"unsupported expert scale-code domain or failed exact GPU roundtrip");
        weights.install_expert_scale_codes(layer,codes)?;
    }
    Ok(Prepared {
        layer,
        experts,
        hidden: k,
        intermediate: n,
        w13_offset: wo,
        w2_offset: doff,
        s13_offset: so,
        s2_offset: ds,
        resident_bytes: experts * (3 * wb + 3 * sb),
        temporary_bytes: wb,
    })
}
fn validate(n: usize, k: usize, group: usize) -> Result<()> {
    ensure!(
        n > 0
            && n <= 65536
            && k > 0
            && k <= 65536
            && group > 0
            && n % 64 == 0
            && k % 64 == 0
            && group % 64 == 0
            && n % group == 0,
        "compact expert layout requires whole N64/K64 row groups"
    );
    Ok(())
}
/// GPU-only packing probe. The caller compares these results directly with
/// the pinned reference's GPU output; there is no duplicate CPU permutation.
pub fn pack_gpu(
    core: &std::path::Path,
    source: &[u8],
    scales: &[u8],
    n: usize,
    k: usize,
    group: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    validate(n, k, group)?;
    ensure!(
        source.len() == n * k / 2 && scales.len() == n * k / 32,
        "expert source byte count mismatch"
    );
    ensure!(
        scales.iter().all(|&x| x <= 247),
        "expert scale exceeds lossless reference conversion range"
    );
    let cuda = Cuda::open(0)?;
    cuda.qualify_gb10()?;
    let module = cuda.module(&std::fs::read(core)?)?;
    let stream = cuda.stream()?;
    let mut sw = cuda.allocate::<u8>(source.len())?;
    sw.upload(source)?;
    let mut ss = cuda.allocate::<u8>(scales.len())?;
    ss.upload(scales)?;
    let mut dw = cuda.allocate::<u8>(source.len())?;
    dw.zero()?;
    let mut ds = cuda.allocate::<u8>(scales.len())?;
    ds.zero()?;
    for (name, src, dst, total) in [
        (
            "ca_expert_compact_words",
            sw.ptr(),
            dw.ptr(),
            source.len() / 4,
        ),
        ("ca_expert_compact_scales", ss.ptr(), ds.ptr(), scales.len()),
    ] {
        let mut src = src;
        let mut dst = dst;
        let mut nr = i32::try_from(n)?;
        let mut kr = i32::try_from(k)?;
        let mut gr = i32::try_from(group)?;
        let mut args: Vec<*mut c_void> = vec![
            (&mut src as *mut u64).cast(),
            (&mut dst as *mut u64).cast(),
            (&mut nr as *mut i32).cast(),
            (&mut kr as *mut i32).cast(),
            (&mut gr as *mut i32).cast(),
        ];
        unsafe {
            module.function(name)?.launch(
                [
                    u32::try_from(total.div_ceil(256)).context("grid overflow")?,
                    1,
                    1,
                ],
                [256, 1, 1],
                0,
                &stream,
                &mut args,
            )
        }?;
    }
    stream.synchronize()?;
    Ok((dw.download()?, ds.download()?))
}
/// Hash GPU-prepared regions in bounded chunks for comparison to the reference
/// GPU's exact bytes. Host hashing is a diagnostic, not a numeric implementation.
pub fn prepared_hashes(
    weights: &crate::weights::Weights,
    p: &Prepared,
) -> Result<std::collections::BTreeMap<String, String>> {
    use sha2::{Digest, Sha256};
    ensure!(weights.expert_ready(p.layer), "expert layer is not ready");
    let wb = p.experts * p.hidden * p.intermediate / 2;
    let sb = p.experts * p.hidden * p.intermediate / 32;
    let mut out = std::collections::BTreeMap::new();
    for (name, offset, bytes) in [
        ("w13", p.w13_offset, 2 * wb),
        ("w2", p.w2_offset, wb),
        ("s13", p.s13_offset, 2 * sb),
        ("s2", p.s2_offset, sb),
    ] {
        let mut digest = Sha256::new();
        for position in (0..bytes).step_by(16 * 1024 * 1024) {
            digest.update(
                weights
                    .device_range(offset + position, (bytes - position).min(16 * 1024 * 1024))?,
            );
        }
        out.insert(name.into(), format!("{:x}", digest.finalize()));
    }
    Ok(out)
}
