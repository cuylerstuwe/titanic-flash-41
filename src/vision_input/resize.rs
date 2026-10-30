// Adapted from SGLang (Apache-2.0), pinned in production/vision-input-provenance.json.
// Modified for native Canada ownership: bounded shared rayon pool, no Python bindings.
use super::par;
const PIL_PRECISION_BITS: u32 = 22;
#[derive(Clone, Copy)]
pub enum Filter {
    Bicubic,
}
#[derive(Clone, Copy)]
pub enum Resample {
    Pil(Filter),
}
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

struct Coeffs {
    bounds: Vec<(usize, usize)>,
    kk: Vec<i32>,
    ksize: usize,
    prec: u32,
}

fn precompute_coeffs(in_size: usize, out_size: usize, _resample: Resample) -> Coeffs {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = 2.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let ss = 1.0 / filterscale;

    let mut kkf = vec![0.0f64; out_size * ksize];
    let mut bounds = vec![(0usize, 0usize); out_size];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let mut xmin = (center - support + 0.5) as i32;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i32;
        if xmax > in_size as i32 {
            xmax = in_size as i32;
        }
        let count = (xmax - xmin) as usize;
        let k = &mut kkf[xx * ksize..(xx + 1) * ksize];
        let mut ww = 0.0f64;
        for (x, kv) in k[..count].iter_mut().enumerate() {
            let w = bicubic((x as f64 + xmin as f64 - center + 0.5) * ss);
            *kv = w;
            ww += w;
        }
        if ww != 0.0 {
            for kv in k[..count].iter_mut() {
                *kv /= ww;
            }
        }
        bounds[xx] = (xmin as usize, count);
    }

    let prec = PIL_PRECISION_BITS;
    let factor = (1i64 << prec) as f64;
    let kk = kkf
        .iter()
        .map(|&v| {
            if v < 0.0 {
                (-0.5 + v * factor) as i32
            } else {
                (0.5 + v * factor) as i32
            }
        })
        .collect();
    Coeffs {
        bounds,
        kk,
        ksize,
        prec,
    }
}

#[inline]
fn clip8(v: i32, prec: u32) -> u8 {
    if v >= 1 << (prec + 8) {
        255
    } else if v <= 0 {
        0
    } else {
        (v >> prec) as u8
    }
}

fn resample_horizontal(src: &[u8], h: usize, w: usize, out_w: usize, c: &Coeffs) -> Vec<u8> {
    let mut out = vec![0u8; h * out_w * 3];
    par::for_chunks_mut(&mut out, out_w * 3, |y, row| {
        let src_row = &src[y * w * 3..(y + 1) * w * 3];
        for xx in 0..out_w {
            let (xmin, count) = c.bounds[xx];
            let k = &c.kk[xx * c.ksize..xx * c.ksize + count];
            let mut s = [1i32 << (c.prec - 1); 3];
            for (x, &coef) in k.iter().enumerate() {
                let p = (xmin + x) * 3;
                s[0] += src_row[p] as i32 * coef;
                s[1] += src_row[p + 1] as i32 * coef;
                s[2] += src_row[p + 2] as i32 * coef;
            }
            let o = xx * 3;
            row[o] = clip8(s[0], c.prec);
            row[o + 1] = clip8(s[1], c.prec);
            row[o + 2] = clip8(s[2], c.prec);
        }
    });
    out
}

fn resample_vertical(src: &[u8], w: usize, out_h: usize, c: &Coeffs) -> Vec<u8> {
    let mut out = vec![0u8; out_h * w * 3];
    par::for_chunks_mut(&mut out, w * 3, |yy, row| {
        let (ymin, count) = c.bounds[yy];
        let k = &c.kk[yy * c.ksize..yy * c.ksize + count];
        for x in 0..w {
            let mut s = [1i32 << (c.prec - 1); 3];
            for (y, &coef) in k.iter().enumerate() {
                let p = ((ymin + y) * w + x) * 3;
                s[0] += src[p] as i32 * coef;
                s[1] += src[p + 1] as i32 * coef;
                s[2] += src[p + 2] as i32 * coef;
            }
            let o = x * 3;
            row[o] = clip8(s[0], c.prec);
            row[o + 1] = clip8(s[1], c.prec);
            row[o + 2] = clip8(s[2], c.prec);
        }
    });
    out
}

/// Separable resize of a flat HWC RGB buffer, bit-exact against `resample`.
///
/// Enters the fan-out pool once for both passes; the per-row `for_chunks_mut`
/// calls inside then reuse that entry rather than injecting a job per pass.
pub fn resize_rgb(
    src: &[u8],
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    _resample: Resample,
) -> Vec<u8> {
    par::in_pool(move || resize_passes(src, h, w, out_h, out_w, _resample))
}

fn resize_passes(
    src: &[u8],
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    _resample: Resample,
) -> Vec<u8> {
    // Per-axis coefficients — and, under `AtenU8`, a per-axis precision.
    let coeffs = |in_size, out_size| precompute_coeffs(in_size, out_size, _resample);
    match (out_w != w, out_h != h) {
        (true, true) => {
            let tmp = resample_horizontal(src, h, w, out_w, &coeffs(w, out_w));
            resample_vertical(&tmp, out_w, out_h, &coeffs(h, out_h))
        }
        (true, false) => resample_horizontal(src, h, w, out_w, &coeffs(w, out_w)),
        (false, true) => resample_vertical(src, w, out_h, &coeffs(h, out_h)),
        (false, false) => src.to_vec(),
    }
}
