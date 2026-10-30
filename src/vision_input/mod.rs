//! Native image input handling; no CPU model inference. Geometry follows the
//! pinned SGLang V4.1 processor. See production/vision-input-provenance.json.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Cursor;
mod par;
mod patch;
mod resize;

pub const PATCH: usize = 14;
pub const DOWNSAMPLE: usize = 3;
pub const MAX_TOKENS: usize = 1024;
pub const MIN_PIXELS: usize = 544 * 544;
pub const MAX_ENCODED_BYTES: usize = 32 << 20;
pub const MAX_INPUT_PIXELS: usize = 89_478_485;
// deepseek-recipe-core 0.2.0 canonical invalid-image text. Kept standalone so
// the CPU codec qualification does not need the conversation protocol feature.
pub const INVALID_IMAGE: &str = "You have uploaded an unsupported image. Please make sure your image is valid and has one of the following formats: webp, png, jpeg, and gif.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Detail {
    Auto,
    Low,
    High,
    Original,
}

/// Identity is tied to input bytes and preprocessing semantics. An explicit
/// detail remains part of it even though the qualified encoder ignores that
/// hint, so a future detail policy cannot accidentally reuse an old result.
pub fn identity(encoded: &[u8], detail: Detail) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"canada-v41-image-input-pil-rgb-v1\0");
    h.update([detail as u8]);
    h.update(encoded);
    h.finalize().into()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    pub width: usize,
    pub height: usize,
    pub resize_width: usize,
    pub resize_height: usize,
    pub top: usize,
    pub left: usize,
    pub vit_height: usize,
    pub vit_width: usize,
    pub llm_height: usize,
    pub llm_width: usize,
    pub tokens: usize,
}
impl Plan {
    pub fn new(original_width: usize, original_height: usize) -> Result<Self> {
        let area = original_width
            .checked_mul(original_height)
            .ok_or_else(|| anyhow::anyhow!(INVALID_IMAGE))?;
        ensure!(area > 0 && area <= MAX_INPUT_PIXELS, "{INVALID_IMAGE}");
        let (mut width, mut height) = (original_width, original_height);
        if area < MIN_PIXELS {
            let ratio = (MIN_PIXELS as f64 / area as f64).sqrt();
            width = (width as f64 * ratio) as usize;
            height = (height as f64 * ratio) as usize;
        }
        let (mut out_w, mut out_h) = (
            width.div_ceil(PATCH) * PATCH,
            height.div_ceil(PATCH) * PATCH,
        );
        let grid = |h: usize, w: usize| {
            (
                (h / PATCH).div_ceil(DOWNSAMPLE),
                (w / PATCH).div_ceil(DOWNSAMPLE),
            )
        };
        let count = |h: usize, w: usize| h * (w + 1) + 2;
        let (lh, lw) = grid(out_h, out_w);
        if count(lh, lw) > MAX_TOKENS {
            let ratio = height as f64 / width as f64;
            let max_w = ((MAX_TOKENS - 2) as f64 / ratio + 0.25).sqrt() - 0.5;
            let max_h = max_w * ratio;
            let cell = PATCH * DOWNSAMPLE;
            if max_w < 1.0 {
                out_h = (MAX_TOKENS - 2) / 2 * cell;
                out_w = cell;
            } else if max_h < 1.0 {
                out_h = cell;
                out_w = (MAX_TOKENS - 3) * cell;
            } else {
                let beta = (max_w.floor() * cell as f64 / width as f64)
                    .min(max_h.floor() * cell as f64 / height as f64);
                out_h = (height as f64 * beta / PATCH as f64).floor() as usize * PATCH;
                out_w = (width as f64 * beta / PATCH as f64).floor() as usize * PATCH;
            }
        }
        let (lh, lw) = grid(out_h, out_w);
        let (mut resize_h, mut resize_w) = (out_h, out_w);
        let original_ratio = original_width as f64 / original_height as f64;
        let target_ratio = out_w as f64 / out_h as f64;
        // Python round / ImageOps.contain uses ties-to-even, including padding.
        if original_ratio > target_ratio {
            resize_h = (original_height as f64 / original_width as f64 * out_w as f64)
                .round_ties_even() as usize;
        } else if original_ratio < target_ratio {
            resize_w = (original_width as f64 / original_height as f64 * out_h as f64)
                .round_ties_even() as usize;
        }
        ensure!(
            resize_h > 0
                && resize_w > 0
                && resize_h <= out_h
                && resize_w <= out_w
                && count(lh, lw) <= MAX_TOKENS,
            "{INVALID_IMAGE}"
        );
        Ok(Self {
            width: out_w,
            height: out_h,
            resize_width: resize_w,
            resize_height: resize_h,
            top: ((out_h - resize_h) as f64 / 2.).round_ties_even() as usize,
            left: ((out_w - resize_w) as f64 / 2.).round_ties_even() as usize,
            vit_height: out_h / PATCH,
            vit_width: out_w / PATCH,
            llm_height: lh,
            llm_width: lw,
            tokens: count(lh, lw),
        })
    }
}

pub struct Prepared {
    pub identity: [u8; 32],
    pub original_width: usize,
    pub original_height: usize,
    pub plan: Plan,
    /// [patch_y, patch_x, channel, y, x], canonical BF16 normalized pixels.
    pub patches: Vec<u16>,
}

pub fn prepare_rgb(rgb: &[u8], width: usize, height: usize) -> Result<(Plan, Vec<u16>)> {
    let plan = Plan::new(width, height)?;
    let bits = patch::resize_patchify(
        rgb,
        height,
        width,
        patch::ImagePlan {
            out_h: plan.height,
            out_w: plan.width,
            resize_h: plan.resize_height,
            resize_w: plan.resize_width,
            top: plan.top,
            left: plan.left,
            patch_size: PATCH,
        },
    )
    .map_err(anyhow::Error::msg)?;
    Ok((plan, bits))
}

pub fn decode_rgb(encoded: &[u8]) -> Result<(usize, usize, Vec<u8>)> {
    ensure!(
        !encoded.is_empty() && encoded.len() <= MAX_ENCODED_BYTES,
        "{INVALID_IMAGE}"
    );
    if encoded.starts_with(&[0xff, 0xd8]) {
        // PIL uses libjpeg-turbo. The pure-Rust decoder's IDCT rounding differs
        // even for 4:4:4, so use accurate/default DCT and fancy upsampling here.
        let mut decoder =
            turbojpeg::Decompressor::new().map_err(|_| anyhow::anyhow!(INVALID_IMAGE))?;
        let header = decoder
            .read_header(encoded)
            .map_err(|_| anyhow::anyhow!(INVALID_IMAGE))?;
        let area = header
            .width
            .checked_mul(header.height)
            .ok_or_else(|| anyhow::anyhow!(INVALID_IMAGE))?;
        ensure!(area > 0 && area <= MAX_INPUT_PIXELS, "{INVALID_IMAGE}");
        decoder
            .set_fast_upsample(false)
            .map_err(|_| anyhow::anyhow!(INVALID_IMAGE))?;
        let cmyk = matches!(
            header.colorspace,
            turbojpeg::Colorspace::CMYK | turbojpeg::Colorspace::YCCK
        );
        let channels = if cmyk { 4 } else { 3 };
        let mut pixels = vec![0u8; area * channels];
        decoder
            .decompress(
                encoded,
                turbojpeg::Image {
                    pixels: pixels.as_mut_slice(),
                    width: header.width,
                    height: header.height,
                    pitch: header.width * channels,
                    format: if cmyk {
                        turbojpeg::PixelFormat::CMYK
                    } else {
                        turbojpeg::PixelFormat::RGB
                    },
                },
            )
            .map_err(|_| anyhow::anyhow!(INVALID_IMAGE))?;
        if cmyk {
            // PIL's JPEG raw mode is CMYK;I. Its subsequent RGB conversion
            // multiplies the inverted channels by inverted K, rounded to u8.
            pixels = pixels
                .chunks_exact(4)
                .flat_map(|p| {
                    [0, 1, 2].map(|c| ((u16::from(p[c]) * u16::from(p[3]) + 127) / 255) as u8)
                })
                .collect();
        }
        return Ok((header.width, header.height, pixels));
    }
    let mut reader = image::ImageReader::new(Cursor::new(encoded))
        .with_guessed_format()
        .map_err(|_| anyhow::anyhow!(INVALID_IMAGE))?;
    ensure!(
        matches!(
            reader.format(),
            Some(
                image::ImageFormat::Png
                    | image::ImageFormat::Jpeg
                    | image::ImageFormat::WebP
                    | image::ImageFormat::Gif
            )
        ),
        "{INVALID_IMAGE}"
    );
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(512 << 20);
    limits.max_image_width = Some(MAX_INPUT_PIXELS as u32);
    limits.max_image_height = Some(MAX_INPUT_PIXELS as u32);
    reader.limits(limits);
    let decoder = reader
        .into_decoder()
        .map_err(|_| anyhow::anyhow!(INVALID_IMAGE))?;
    use image::ImageDecoder;
    let (w, h) = decoder.dimensions();
    ensure!(
        (w as u64) * (h as u64) <= MAX_INPUT_PIXELS as u64,
        "{INVALID_IMAGE}"
    );
    // Match the deployed PIL RGB conversion: first frame, discard alpha, and
    // do not transpose EXIF orientation. The processor disables smart_to_rgb.
    let decoded =
        image::DynamicImage::from_decoder(decoder).map_err(|_| anyhow::anyhow!(INVALID_IMAGE))?;
    let rgb = match decoded {
        // PIL I;16 -> RGB clips, unlike image's normalized 16-to-8 conversion.
        image::DynamicImage::ImageLuma16(gray) => gray
            .as_raw()
            .iter()
            .flat_map(|x| [(*x).min(255) as u8; 3])
            .collect(),
        other => other.into_rgb8().into_raw(),
    };
    Ok((w as usize, h as usize, rgb))
}

pub fn prepare(encoded: &[u8], detail: Detail) -> Result<Prepared> {
    let (width, height, rgb) = decode_rgb(encoded)?;
    let (plan, patches) = prepare_rgb(&rgb, width, height)?;
    Ok(Prepared {
        identity: identity(encoded, detail),
        original_width: width,
        original_height: height,
        plan,
        patches,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dimensions_and_image_identity_are_bounded() {
        for (w, h) in [(1, 1), (640, 480), (4096, 4096), (12, 18000), (18000, 12)] {
            let p = Plan::new(w, h).unwrap();
            assert!((2..=MAX_TOKENS).contains(&p.tokens));
            assert_eq!(p.width % PATCH, 0);
            assert_eq!(p.height % PATCH, 0);
        }
        assert!(Plan::new(0, 1).is_err());
        assert!(Plan::new(usize::MAX, 2).is_err());
        assert!(Plan::new(MAX_INPUT_PIXELS + 1, 1).is_err());
        assert_ne!(identity(b"a", Detail::Auto), identity(b"b", Detail::Auto));
        assert_ne!(identity(b"a", Detail::Low), identity(b"a", Detail::High));
        for bytes in [b"not an image".as_slice(), b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF89a", b""] {
            assert_eq!(decode_rgb(bytes).unwrap_err().to_string(), INVALID_IMAGE);
        }
    }
}
