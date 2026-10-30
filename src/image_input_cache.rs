//! Bounded source resolution and memoized canonical input preprocessing.
//! This module performs input handling only, never model inference.
use crate::{
    image_cache::Descriptor,
    prompt_media::IMAGE_TOKEN,
    vision::Geometry,
    vision_input::{self, Detail, INVALID_IMAGE, MAX_ENCODED_BYTES},
    worker_media::PatchInput,
};
use anyhow::{Result, ensure};
use deepseek_recipe_core::multimodal::{ImageDetail, ImageSource};
use std::{
    collections::{HashMap, VecDeque},
    io::{Cursor, Read},
    sync::{Arc, Mutex, Weak},
    time::Duration,
};

const WARM_BYTES: usize = 256 << 20;
const LIVE_BYTES: usize = 2 << 30;
fn available_host_memory() -> Result<usize> {
    #[cfg(target_os = "linux")]
    {
        let memory = crate::memory::Available::gb10(usize::MAX, usize::MAX)?;
        Ok(memory.admitted_available)
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(usize::MAX)
    }
}
// Metadata bound, not an alternate pixel implementation. Include both the
// decoded format and RGB conversion, resize passes/coefficients, and patches.
fn workspace_bytes(width: usize, height: usize, p: &vision_input::Plan) -> usize {
    let coefficients = |input: usize, output: usize| {
        output * (4 * input.div_ceil(output).max(1) + 3) * 12 + output * 16
    };
    let rgb = width * height * 3;
    let decode = (512usize << 20) + rgb;
    let resize = rgb
        + height * p.resize_width * 3
        + p.resize_width * p.resize_height * 3
        + coefficients(width, p.resize_width)
        + coefficients(height, p.resize_height)
        + p.vit_height * p.vit_width * 588 * 2;
    decode.max(resize)
}
#[derive(Default)]
struct State {
    warm: HashMap<[u8; 32], Arc<PatchInput>>,
    live: HashMap<[u8; 32], Weak<PatchInput>>,
    oldest: VecDeque<[u8; 32]>,
    retained: usize,
    hits: u64,
    misses: u64,
}
impl State {
    fn clear_warm(&mut self) {
        self.warm.clear();
        self.oldest.clear();
        self.retained = 0;
    }
    fn retain(&mut self, input: Arc<PatchInput>) {
        self.oldest.retain(|k| *k != input.identity);
        if self.warm.contains_key(&input.identity) {
            self.oldest.push_back(input.identity);
            return;
        }
        while self.retained + input.bytes() > WARM_BYTES {
            let key = self.oldest.pop_front().unwrap();
            self.retained -= self.warm.remove(&key).unwrap().bytes();
        }
        self.retained += input.bytes();
        self.oldest.push_back(input.identity);
        self.warm.insert(input.identity, input);
    }
    fn live_bytes(&self) -> usize {
        self.live
            .values()
            .filter_map(Weak::upgrade)
            .map(|p| p.bytes())
            .sum()
    }
}
pub struct ImageInputCache {
    state: Mutex<State>,
    http: ureq::Agent,
    available_memory: fn() -> Result<usize>,
    reserve_workspace: fn(usize) -> Result<Option<crate::memory::HostWorkspaceReservation>>,
}
pub struct Expanded {
    pub tokens: Vec<u32>,
    pub images: Vec<Descriptor>,
    pub inputs: Vec<Arc<PatchInput>>,
}
impl ImageInputCache {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State::default()),
            available_memory: available_host_memory,
            reserve_workspace: |bytes|crate::memory::reserve_host_workspace(bytes).map(Some),
            http: ureq::Agent::config_builder()
                .timeout_global(Some(Duration::from_secs(20)))
                .max_redirects(5)
                .build()
                .into(),
        }
    }
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        // CPU input tests run inside the bounded compiler container. Model
        // serving always uses the actual host/cgroup probe above.
        Self {
            available_memory: || Ok(16usize << 30),
            reserve_workspace: |_|Ok(None),
            ..Self::new()
        }
    }
    fn resolve(&self, source: &ImageSource) -> Result<(Vec<u8>, Detail)> {
        let (bytes, detail) = match source {
            ImageSource::DataUrl {
                data_url, detail, ..
            } => {
                let url = data_url::DataUrl::process(data_url)
                    .map_err(|_| anyhow::anyhow!(INVALID_IMAGE))?;
                let mut bytes = Vec::new();
                url.decode(|part| -> Result<(), ()> {
                    if bytes.len() + part.len() > MAX_ENCODED_BYTES {
                        return Err(());
                    }
                    bytes.extend_from_slice(part);
                    Ok(())
                })
                .map_err(|_| anyhow::anyhow!(INVALID_IMAGE))?;
                (bytes, *detail)
            }
            ImageSource::Bytes { data, detail, .. } => {
                ensure!(data.len() <= MAX_ENCODED_BYTES, "{INVALID_IMAGE}");
                (data.clone(), *detail)
            }
            ImageSource::Url { url, detail, .. } => {
                let scheme = url.split_once("://").map(|(scheme, _)| scheme);
                ensure!(
                    url.len() <= 8192
                        && scheme
                            .is_some_and(|s| s.eq_ignore_ascii_case("http")
                                || s.eq_ignore_ascii_case("https")),
                    "image URL must use HTTP or HTTPS"
                );
                let mut response = self
                    .http
                    .get(url)
                    .call()
                    .map_err(|_| anyhow::anyhow!("image URL fetch failed"))?;
                let mut bytes = Vec::new();
                response
                    .body_mut()
                    .as_reader()
                    .take((MAX_ENCODED_BYTES + 1) as u64)
                    .read_to_end(&mut bytes)
                    .map_err(|_| anyhow::anyhow!("image URL read failed"))?;
                ensure!(bytes.len() <= MAX_ENCODED_BYTES, "{INVALID_IMAGE}");
                (bytes, *detail)
            }
        };
        ensure!(
            !bytes.is_empty() && bytes.len() <= MAX_ENCODED_BYTES,
            "{INVALID_IMAGE}"
        );
        let detail = match detail {
            ImageDetail::Low => Detail::Low,
            ImageDetail::High => Detail::High,
            ImageDetail::Auto => Detail::Auto,
            ImageDetail::Original => Detail::Original,
        };
        Ok((bytes, detail))
    }
    fn prepare(&self, bytes: &[u8], detail: Detail) -> Result<Arc<PatchInput>> {
        let identity = vision_input::identity(bytes, detail);
        // Serialize cache misses as well as mutation: at most one decoder's
        // large RGB scratch exists, and identical concurrent sources coalesce.
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("image input cache poisoned"))?;
        if let Some(input) = state.live.get(&identity).and_then(Weak::upgrade) {
            state.hits += 1;
            state.retain(input.clone());
            return Ok(input);
        }
        state.live.retain(|_, p| p.strong_count() > 0);
        let reader = image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|_| anyhow::anyhow!(INVALID_IMAGE))?;
        let (width, height) = reader
            .into_dimensions()
            .map_err(|_| anyhow::anyhow!(INVALID_IMAGE))?;
        let plan = vision_input::Plan::new(width as usize, height as usize)?;
        let patch_bytes = plan.vit_height * plan.vit_width * 588 * 2;
        let workspace = workspace_bytes(width as usize, height as usize, &plan);
        ensure!(
            workspace <= 1usize << 30,
            "image preprocessing workspace capacity exceeded"
        );
        if state.live_bytes() + patch_bytes > LIVE_BYTES {
            state.clear_warm();
        }
        ensure!(
            state.live_bytes() + patch_bytes <= LIVE_BYTES,
            crate::memory::InputCapacityError("live image input capacity exceeded", 0)
        );
        {
            // Other in-flight CPU work is already subtracted. Retained patch
            // data also counts in actual host/cgroup memory. Admit the same
            // conservative decoder/resize bound we reserve below, rather than
            // charging every small image the maximum supported 1GiB workspace.
            let enough = || -> Result<bool> {
                Ok((self.available_memory)()? > (8usize << 30) + workspace + patch_bytes)
            };
            if !enough()? {
                state.clear_warm();
            }
            ensure!(
                enough()?,
                crate::memory::InputCapacityError("image preprocessing cannot retain the UMA reserve", (8usize << 30) + workspace + patch_bytes)
            );
        }
        let _workspace = (self.reserve_workspace)(workspace + patch_bytes)?;
        let prepared = vision_input::prepare(bytes, detail)?;
        ensure!(
            prepared.plan == plan && prepared.identity == identity,
            "image header and decoded geometry differ"
        );
        let input = Arc::new(PatchInput {
            identity,
            geometry: Geometry {
                height: plan.vit_height,
                width: plan.vit_width,
            },
            patches: prepared.patches,
        });
        input.validate()?;
        state.misses += 1;
        state.live.insert(identity, Arc::downgrade(&input));
        state.retain(input.clone());
        Ok(input)
    }
    pub fn expand(
        &self,
        tokens: Vec<u32>,
        sources: &[ImageSource],
        capacity: usize,
    ) -> Result<Expanded> {
        self.expand_cancellable(tokens, sources, capacity, &|| false)
    }
    pub fn expand_cancellable(
        &self,
        tokens: Vec<u32>,
        sources: &[ImageSource],
        capacity: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Expanded> {
        ensure!(!cancelled(), "request cancelled during image processing");
        ensure!(
            tokens.iter().filter(|&&t| t == IMAGE_TOKEN).count() == sources.len(),
            "image placeholder count differs"
        );
        let mut output = Vec::with_capacity(tokens.len());
        let mut images = Vec::new();
        let mut inputs = Vec::new();
        let mut sources = sources.iter();
        let mut encoded_bytes = 0usize;
        for token in tokens {
            if token != IMAGE_TOKEN {
                output.push(token);
            } else {
                ensure!(!cancelled(), "request cancelled during image processing");
                let (bytes, detail) = self.resolve(sources.next().unwrap())?;
                ensure!(!cancelled(), "request cancelled during image processing");
                encoded_bytes += bytes.len();
                ensure!(
                    encoded_bytes <= 128 << 20,
                    "request encoded image byte capacity exceeded"
                );
                let input = self.prepare(&bytes, detail)?;
                let count = input.geometry.tokens();
                ensure!(
                    output.len() + count <= capacity,
                    "image prompt exceeds context capacity"
                );
                images.push(Descriptor {
                    start: output.len(),
                    geometry: input.geometry,
                    input: input.identity,
                });
                output.extend(std::iter::repeat_n(IMAGE_TOKEN, count));
                inputs.push(input);
            }
            ensure!(
                output.len() <= capacity,
                "image prompt exceeds context capacity"
            );
        }
        ensure!(!cancelled(), "request cancelled during image processing");
        Ok(Expanded {
            tokens: output,
            images,
            inputs,
        })
    }
    pub fn metadata(&self) -> Result<serde_json::Value> {
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => {
                return Ok(serde_json::json!({"busy":true}));
            }
            Err(_) => anyhow::bail!("image input cache poisoned"),
        };
        Ok(
            serde_json::json!({"hits":state.hits,"misses":state.misses,"warm_bytes":state.retained,
            "live_bytes":state.live_bytes(),"warm_budget_bytes":WARM_BYTES,"live_budget_bytes":LIVE_BYTES}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    fn png(color: u8) -> Vec<u8> {
        let image = image::RgbImage::from_pixel(16, 16, image::Rgb([color, 17, 240]));
        let mut out = Cursor::new(Vec::new());
        image.write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }
    fn source(data: Vec<u8>) -> ImageSource {
        ImageSource::Bytes {
            data,
            detail: ImageDetail::Auto,
            path: String::new(),
        }
    }
    #[test]
    fn encoded_content_identity_order_detail_and_partial_admission() {
        let cache = ImageInputCache::for_test();
        let a = png(20);
        let b = png(180);
        let pair = cache
            .expand(
                vec![2, IMAGE_TOKEN, 3, IMAGE_TOKEN],
                &[source(a.clone()), source(b)],
                4096,
            )
            .unwrap();
        assert_eq!(pair.images[0].start, 1);
        assert_eq!(pair.images[1].start, 2 + pair.inputs[0].geometry.tokens());
        assert_eq!(pair.inputs[0].geometry, pair.inputs[1].geometry);
        assert_ne!(pair.images[0].input, pair.images[1].input);
        let data_url = format!(
            "data:image/png,{}",
            a.iter().map(|b| format!("%{b:02X}")).collect::<String>()
        );
        let repeat = cache
            .expand(
                vec![IMAGE_TOKEN],
                &[ImageSource::DataUrl {
                    data_url,
                    detail: ImageDetail::Auto,
                    path: String::new(),
                }],
                4096,
            )
            .unwrap();
        assert!(Arc::ptr_eq(&pair.inputs[0], &repeat.inputs[0]));
        let changed = cache.prepare(&a, Detail::High).unwrap();
        assert_ne!(changed.identity, pair.inputs[0].identity);
        assert_eq!(changed.patches, pair.inputs[0].patches);
        assert!(cache.expand(vec![IMAGE_TOKEN], &[], 4096).is_err());
        assert!(
            cache
                .expand(vec![IMAGE_TOKEN], &[source(a.clone())], 8)
                .is_err()
        );
        assert!(
            cache
                .expand_cancellable(vec![IMAGE_TOKEN], &[source(a)], 4096, &|| true)
                .is_err()
        );
        assert!(cache.prepare(b"not an image", Detail::Auto).is_err());
    }
    #[test]
    fn same_url_with_changed_bytes_does_not_reuse_old_image() {
        let a = png(10);
        let b = png(100);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/image", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            for bytes in [a, b] {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut header = Vec::new();
                let mut byte = [0];
                while !header.ends_with(b"\r\n\r\n") {
                    stream.read_exact(&mut byte).unwrap();
                    header.push(byte[0]);
                    assert!(header.len() < 8192);
                }
                write!(stream,"HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: image/png\r\nConnection: close\r\n\r\n",bytes.len()).unwrap();
                stream.write_all(&bytes).unwrap();
            }
        });
        let cache = ImageInputCache::for_test();
        let input = ImageSource::Url {
            url,
            detail: ImageDetail::Auto,
            path: String::new(),
        };
        let first = cache
            .expand(vec![IMAGE_TOKEN], std::slice::from_ref(&input), 4096)
            .unwrap();
        let next = cache.expand(vec![IMAGE_TOKEN], &[input], 4096).unwrap();
        assert_ne!(first.images[0].input, next.images[0].input);
        server.join().unwrap();
    }
    #[test]
    fn insufficient_memory_rejects_before_decoding_and_releases_only_warm_inputs() {
        let mut cache = ImageInputCache::for_test();
        let live = cache.prepare(&png(10), Detail::Auto).unwrap();
        cache.available_memory = || Ok((8usize << 30) + (256 << 20));
        let error = cache.prepare(&png(20), Detail::Auto).err().unwrap();
        assert!(error.to_string().contains("UMA reserve"));
        assert!(error.downcast_ref::<crate::memory::InputCapacityError>().is_some());
        let state = cache.state.lock().unwrap();
        assert!(state.warm.is_empty());
        assert_eq!(state.live_bytes(), live.bytes());
        assert_eq!(state.misses, 1);
        drop(state);
        // The real conservative bound fits here; the old unconditional 1GiB
        // allowance refused this valid input despite preserving the 8GiB floor.
        cache.available_memory = || Ok((8usize << 30) + (768 << 20));
        assert!(cache.prepare(&png(20), Detail::Auto).is_ok());
    }
    #[test]
    #[ignore = "requires CANADA_TEST_IMAGE_CASES pointing to the qualified public input panel"]
    fn cached_inputs_keep_pinned_reference_patch_bytes() {
        use sha2::{Digest, Sha256};
        let path = std::path::PathBuf::from(std::env::var("CANADA_TEST_IMAGE_CASES").unwrap());
        let root = path.parent().unwrap();
        let cases: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("reference.json")).unwrap()).unwrap();
        let cache = ImageInputCache::for_test();
        let mut checked = 0;
        for case in cases["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["raw"] != true)
        {
            let bytes = std::fs::read(root.join(case["file"].as_str().unwrap())).unwrap();
            let input = cache.prepare(&bytes, Detail::Auto).unwrap();
            let expected = reference["records"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["name"] == case["name"])
                .unwrap();
            let bits: Vec<u8> = input.patches.iter().flat_map(|p| p.to_le_bytes()).collect();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bits)),
                expected["patch_sha256"].as_str().unwrap(),
                "{}",
                case["name"]
            );
            assert!(Arc::ptr_eq(
                &input,
                &cache.prepare(&bytes, Detail::Auto).unwrap()
            ));
            checked += 1;
        }
        assert!(checked >= 16);
        println!("cached image inputs: {checked} encoded cases byte-exact");
    }
}
