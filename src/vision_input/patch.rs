// Adapted from SGLang (Apache-2.0), pinned in production/vision-input-provenance.json.
// Modified for native Canada ownership: bounded shared rayon pool, no Python bindings.
//! V4.1 PIL bicubic resize, centered padding and CHW patch packing.
use super::{par, resize};
use half::bf16;

pub struct ImagePlan {
    pub out_h: usize,
    pub out_w: usize,
    pub resize_h: usize,
    pub resize_w: usize,
    pub top: usize,
    pub left: usize,
    pub patch_size: usize,
}

pub fn resize_patchify(
    rgb: &[u8],
    h: usize,
    w: usize,
    plan: ImagePlan,
) -> Result<Vec<u16>, String> {
    let ImagePlan {
        out_h,
        out_w,
        resize_h,
        resize_w,
        top,
        left,
        patch_size: ps,
    } = plan;
    if ps == 0
        || h == 0
        || w == 0
        || resize_h == 0
        || resize_w == 0
        || out_h == 0
        || out_w == 0
        || !out_h.is_multiple_of(ps)
        || !out_w.is_multiple_of(ps)
        || resize_h > out_h
        || resize_w > out_w
        || top > out_h - resize_h
        || left > out_w - resize_w
        || h.checked_mul(w).and_then(|n| n.checked_mul(3)) != Some(rgb.len())
    {
        return Err("invalid V4.1 image geometry".into());
    }
    let len = out_h
        .checked_mul(out_w)
        .and_then(|n| n.checked_mul(3))
        .ok_or("V4.1 output size overflow")?;
    let resized = resize::resize_rgb(
        rgb,
        h,
        w,
        resize_h,
        resize_w,
        resize::Resample::Pil(resize::Filter::Bicubic),
    );
    let lut: [u16; 256] =
        core::array::from_fn(|i| bf16::from_f32(((i as f32 / 255.0) - 0.5) / 0.5).to_bits());
    let mut out = vec![0u16; len];
    let patch_len = 3 * ps * ps;
    let grid_w = out_w / ps;
    par::for_chunks_mut(&mut out, patch_len, |index, patch| {
        let py = index / grid_w * ps;
        let px = index % grid_w * ps;
        for c in 0..3 {
            for y in 0..ps {
                for x in 0..ps {
                    let iy = py + y;
                    let ix = px + x;
                    let value =
                        if iy >= top && iy < top + resize_h && ix >= left && ix < left + resize_w {
                            resized[((iy - top) * resize_w + ix - left) * 3 + c]
                        } else {
                            127
                        };
                    patch[(c * ps + y) * ps + x] = lut[value as usize];
                }
            }
        }
    });
    Ok(out)
}
