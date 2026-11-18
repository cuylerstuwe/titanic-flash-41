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
/// CANADA_SCRATCH_IMAGE_SOURCE_STORE_MIB (Nov 17; 0..=4096, default 0 = off): keep the encoded bytes of prepared
/// images (a screenshot is ~0.1-0.5 MB encoded against ~5 MB of patches), so a context reference whose patches
/// were evicted still resolves: deferred, its pixels are made again only if the worker's cached prefix does not
/// cover it. Without it such a reference is a context miss (the router then resends full bodies for 25 minutes).
pub(crate) fn source_store_budget() -> Result<usize> {
    match std::env::var("CANADA_SCRATCH_IMAGE_SOURCE_STORE_MIB") {
        Ok(v) => { let mib: usize = v.parse().map_err(|_| anyhow::anyhow!("image source store must be a whole number of MiB"))?;
                   ensure!(mib <= 4096, "image source store at most 4096 MiB"); Ok(mib << 20) },
        Err(std::env::VarError::NotPresent) => Ok(0),
        Err(error) => Err(error.into()),
    }
}
/// Encoded bytes of a prepared image, by identity (see source_store_budget).
pub(crate) struct StoredSource {
    pub(crate) bytes: Vec<u8>,
    pub(crate) detail: Detail,
    pub(crate) canonical: ImageDetail,
    pub(crate) geometry: Geometry,
}
// Metadata only: no encoded strings or extra strong image owners. Mutable HTTP
// URLs are deliberately excluded; only an exact data URL and detail can hit.
const SOURCE_ENTRIES: usize = 1024;
#[derive(Clone, Copy)]
struct SourceMemo {
    identity: [u8; 32],
    geometry: Geometry,
    encoded_bytes: usize,
}
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
    let decode = vision_input::decode_alloc_bound(width, height) + rgb;
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
    source_info: HashMap<[u8;32],(ImageDetail,usize)>,
    sources: HashMap<[u8; 32], SourceMemo>,
    source_oldest: VecDeque<[u8; 32]>,
    oldest: VecDeque<[u8; 32]>,
    retained: usize,
    hits: u64,
    misses: u64,
    source_hits: u64,
    metadata_deferrals: u64,
    stored: HashMap<[u8; 32], Arc<StoredSource>>,
    stored_oldest: VecDeque<[u8; 32]>,
    stored_bytes: usize,
    stored_references: u64,
}
impl State {
    /// Keep (or refresh) an image's encoded bytes under the store's byte budget, oldest out first.
    fn store_source(&mut self, budget: usize, identity: [u8; 32], make: impl FnOnce() -> StoredSource) {
        if budget == 0 { return; }
        if self.stored.contains_key(&identity) { self.touch_stored(&identity); return; }
        let source = make();
        if source.bytes.len() > budget { return; }
        while self.stored_bytes + source.bytes.len() > budget {
            let Some(old) = self.stored_oldest.pop_front() else { break };
            if let Some(gone) = self.stored.remove(&old) { self.stored_bytes -= gone.bytes.len(); }
        }
        self.stored_bytes += source.bytes.len();
        self.stored_oldest.push_back(identity);
        self.stored.insert(identity, Arc::new(source));
    }
    fn touch_stored(&mut self, identity: &[u8; 32]) {
        if self.stored.contains_key(identity) {
            self.stored_oldest.retain(|k| k != identity);
            self.stored_oldest.push_back(*identity);
        }
    }
    fn remember_source(&mut self, key: [u8; 32], input: &PatchInput, encoded_bytes: usize) {
        if self.sources.contains_key(&key) { return; }
        while self.sources.len() >= SOURCE_ENTRIES {
            let expired = self.source_oldest.pop_front().unwrap();
            self.sources.remove(&expired);
        }
        self.sources.insert(key, SourceMemo { identity: input.identity, geometry: input.geometry, encoded_bytes });
        self.source_oldest.push_back(key);
    }
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
    /// See source_store_budget (0 = off). Startup refuses an invalid setting (protocol.rs).
    store_budget: usize,
}
pub struct Expanded {
    pub tokens: Vec<u32>,
    pub images: Vec<Descriptor>,
    pub inputs: Vec<Arc<PatchInput>>,
    pub aliases: Vec<crate::protocol::ImageAlias>,
}
/// A separate planning result: metadata alone must never masquerade as prepared
/// pixels. A consumer must establish the real reusable model frontier/cache hit
/// before omitting a deferred image, or materialize it through the same cache.
pub struct Planned {
    pub tokens: Vec<u32>,
    pub images: Vec<Descriptor>,
    pub inputs: Vec<Arc<PatchInput>>,
    pub deferred: Vec<DeferredInput>,
    pub aliases: Vec<crate::protocol::ImageAlias>,
}
pub struct DeferredInput {
    identity: [u8; 32],
    geometry: Geometry,
    source: DeferredSource,
    cache: Arc<ImageInputCache>,
}
/// Where a deferred image's pixels come from if the worker needs them: the request's own immutable data URL, or the
/// source store's copy of a referenced image's encoded bytes (both checked against the identity on use).
enum DeferredSource { Request(ImageSource), Stored(Arc<StoredSource>) }
impl DeferredInput {
    pub fn identity(&self) -> [u8; 32] { self.identity }
    pub fn geometry(&self) -> Geometry { self.geometry }
    pub fn materialize(&self, cancelled: &dyn Fn() -> bool) -> Result<Arc<PatchInput>> {
        ensure!(!cancelled(), "request cancelled during image processing");
        // No mutable URL can enter this path. The owned source is the exact
        // immutable input from which this process previously learned metadata.
        let input = match &self.source {
            DeferredSource::Request(source) => {
                ensure!(matches!(source, ImageSource::DataUrl { .. }), "deferred source is mutable");
                let (bytes, detail) = self.cache.resolve(source)?;
                ensure!(!cancelled(), "request cancelled during image processing");
                self.cache.prepare(&bytes, detail)?
            }
            DeferredSource::Stored(stored) => self.cache.prepare(&stored.bytes, stored.detail)?,
        };
        ensure!(input.identity == self.identity && input.geometry == self.geometry,
            "deferred image identity or geometry differs");
        ensure!(!cancelled(), "request cancelled during image processing");
        Ok(input)
    }
}
impl ImageInputCache {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State::default()),
            available_memory: available_host_memory,
            reserve_workspace: |bytes|crate::memory::reserve_host_workspace(bytes).map(Some),
            store_budget: source_store_budget().unwrap_or(0),
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
    fn source_hit(&self, key: &[u8; 32], remaining_bytes: usize) -> Result<Option<(Arc<PatchInput>, usize)>> {
        let mut state = self.state.lock().map_err(|_| anyhow::anyhow!("image input cache poisoned"))?;
        let Some(memo) = state.sources.get(key).copied() else { return Ok(None); };
        let Some(input) = state.live.get(&memo.identity).and_then(Weak::upgrade) else {
            // Keep bounded metadata after pixels are evicted. The eager path
            // still resolves and admits the original source on this miss.
            return Ok(None);
        };
        // A memoized image still consumes its original request byte allowance.
        ensure!(memo.encoded_bytes <= remaining_bytes, "request encoded image byte capacity exceeded");
        state.hits += 1;
        state.source_hits += 1;
        state.retain(input.clone());
        Ok(Some((input, memo.encoded_bytes)))
    }
    fn pin_source_hits(&self, keys: &[Option<[u8; 32]>]) -> Result<Vec<Arc<PatchInput>>> {
        if keys.iter().all(Option::is_none) { return Ok(Vec::new()); }
        let state = self.state.lock().map_err(|_| anyhow::anyhow!("image input cache poisoned"))?;
        // Preserve this request's existing hits before an earlier miss can
        // evict them. These are temporary owners, not a larger warm cache.
        // Do not update LRU order, admission accounting or hit statistics here.
        Ok(keys.iter().flatten().filter_map(|key| state.sources.get(key))
            .filter_map(|memo| state.live.get(&memo.identity).and_then(Weak::upgrade)).collect())
    }
    fn prepare_with_optional_pins(&self, bytes: &[u8], detail: Detail,
        pins: &mut Vec<Arc<PatchInput>>) -> Result<Arc<PatchInput>> {
        match self.prepare(bytes, detail) {
            Err(error) if !pins.is_empty()
                && error.downcast_ref::<crate::memory::InputCapacityError>().is_some() => {
                // Future hits are expendable under physical/live-input
                // pressure. The first attempt already reclaims warm owners;
                // release these optional owners before retrying the same
                // admission once. Already prepared request inputs stay live.
                pins.clear();
                self.prepare(bytes, detail)
            }
            result => result,
        }
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
            state.touch_stored(&identity);
            return Ok(input);
        }
        state.live.retain(|_, p| p.strong_count() > 0);
        { let State {live,source_info,..}=&mut *state;source_info.retain(|key,_|live.contains_key(key)); }
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
                Ok((self.available_memory)()? > crate::memory::floor_bytes() + workspace + patch_bytes)
            };
            if !enough()? {
                state.clear_warm();
            }
            ensure!(
                enough()?,
                crate::memory::InputCapacityError("image preprocessing cannot retain the UMA reserve", crate::memory::floor_bytes() + workspace + patch_bytes)
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
        let canonical_detail=match detail {Detail::Low=>ImageDetail::Low,Detail::High=>ImageDetail::High,
            Detail::Auto=>ImageDetail::Auto,Detail::Original=>ImageDetail::Original};
        state.source_info.insert(identity,(canonical_detail,bytes.len()));
        state.retain(input.clone());
        let geometry = input.geometry;
        state.store_source(self.store_budget, identity, || StoredSource {
            bytes: bytes.to_vec(), detail, canonical: canonical_detail, geometry });
        Ok(input)
    }
    /// `allow_stored`: a reference whose patches are no longer live may resolve from the source store (the
    /// caller then plans with deferred pixels, protocol.rs); otherwise it is a context miss as before.
    pub(crate) fn pin_references(&self, keys: &[[u8;32]], allow_stored: bool) -> Result<Vec<crate::protocol::ImageReference>> {
        let mut state=self.state.lock().map_err(|_|anyhow::anyhow!("image input cache poisoned"))?;
        let mut result=Vec::new();
        for key in keys {
            if result.iter().any(|r:&crate::protocol::ImageReference|r.identity==*key){continue;}
            let hex=key.iter().map(|b|format!("{b:02x}")).collect::<String>();
            let source=format!("http://titanic-context.invalid/{hex}");
            let live=state.live.get(key).and_then(Weak::upgrade);
            if let (Some(input),Some(&(detail,encoded_bytes)))=(live,state.source_info.get(key)) {
                state.retain(input.clone());
                state.touch_stored(key);
                result.push(crate::protocol::ImageReference{source,input:Some(input),stored:None,identity:*key,detail,encoded_bytes});
                continue;
            }
            let stored=state.stored.get(key).filter(|_|allow_stored).cloned().ok_or(crate::protocol::ContextImageMiss)?;
            state.touch_stored(key);
            state.stored_references+=1;
            result.push(crate::protocol::ImageReference{source,input:None,identity:*key,detail:stored.canonical,
                encoded_bytes:stored.bytes.len(),stored:Some(stored)});
        }
        Ok(result)
    }
    #[cfg(test)]
    pub(crate) fn forget_for_test(&self) {
        let mut state=self.state.lock().unwrap();state.clear_warm();
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
        self.expand_with_references(tokens, sources, capacity, cancelled, None)
    }
    pub(crate) fn expand_referenced(&self, tokens: Vec<u32>, sources: &[ImageSource],
        capacity: usize, cancelled: &dyn Fn() -> bool, references: &[crate::protocol::ImageReference]) -> Result<Expanded> {
        self.expand_with_references(tokens, sources, capacity, cancelled, Some(references))
    }
    fn expand_with_references(&self, tokens: Vec<u32>, sources: &[ImageSource],
        capacity: usize, cancelled: &dyn Fn() -> bool, references: Option<&[crate::protocol::ImageReference]>) -> Result<Expanded> {
        let plan = self.expand_plan(tokens, sources, capacity, cancelled, references, None)?;
        ensure!(plan.deferred.is_empty(), "eager expansion deferred an image");
        Ok(Expanded { tokens: plan.tokens, images: plan.images, inputs: plan.inputs, aliases: plan.aliases })
    }
    /// Opt-in frontend planning. The worker must resolve its actual prefix and
    /// embedding-cache outcome before omitting or materializing deferred input.
    pub fn plan_cancellable(self: &Arc<Self>, tokens: Vec<u32>, sources: &[ImageSource],
        capacity: usize, cancelled: &dyn Fn() -> bool) -> Result<Planned> {
        self.expand_plan(tokens, sources, capacity, cancelled, None, Some(self))
    }
    /// Deferred planning for a compact-context request (Nov 17): live references keep their patches, stored ones
    /// (patches evicted, encoded bytes in the source store) defer their pixels to the worker.
    pub(crate) fn plan_referenced(self: &Arc<Self>, tokens: Vec<u32>, sources: &[ImageSource], capacity: usize,
        cancelled: &dyn Fn() -> bool, references: &[crate::protocol::ImageReference]) -> Result<Planned> {
        self.expand_plan(tokens, sources, capacity, cancelled, Some(references), Some(self))
    }
    fn expand_plan(&self, tokens: Vec<u32>, sources: &[ImageSource], capacity: usize,
        cancelled: &dyn Fn() -> bool, references: Option<&[crate::protocol::ImageReference]>,
        deferred_cache: Option<&Arc<Self>>) -> Result<Planned> {
        ensure!(!cancelled(), "request cancelled during image processing");
        ensure!(
            tokens.iter().filter(|&&t| t == IMAGE_TOKEN).count() == sources.len(),
            "image placeholder count differs"
        );
        let source_keys = sources.iter().map(|source| -> Result<_> {
            ensure!(!cancelled(), "request cancelled during image processing");
            Ok(match source {
                ImageSource::DataUrl { data_url, detail, .. } =>
                    Some(crate::protocol::image_source_hash(data_url, *detail)),
                _ => None, // Mutable HTTP sources must still be fetched.
            })
        }).collect::<Result<Vec<_>>>()?;
        let mut pins = self.pin_source_hits(&source_keys)?;
        let mut output = Vec::with_capacity(tokens.len());
        let mut images = Vec::new();
        let mut inputs = Vec::new();
        let mut deferred = Vec::new();
        let mut aliases = Vec::new();
        let mut sources = sources.iter().zip(source_keys);
        let mut encoded_bytes = 0usize;
        for token in tokens {
            if token != IMAGE_TOKEN {
                output.push(token);
            } else {
                ensure!(!cancelled(), "request cancelled during image processing");
                let (source, source_key) = sources.next().unwrap();
                // Only a successfully prepared exact source/detail can supply
                // geometry. Metadata does not bypass encoded or token budgets.
                // Keep already-live patches on the ordinary fast path.
                if let (Some(cache), Some(key)) = (deferred_cache, source_key) {
                    let memo = {
                        let state = self.state.lock().map_err(|_| anyhow::anyhow!("image input cache poisoned"))?;
                        state.sources.get(&key).copied().filter(|memo|
                            state.live.get(&memo.identity).and_then(Weak::upgrade).is_none())
                    };
                    if let Some(memo) = memo {
                        ensure!((1..=MAX_ENCODED_BYTES).contains(&memo.encoded_bytes), "invalid deferred image byte count");
                        encoded_bytes = encoded_bytes.checked_add(memo.encoded_bytes)
                            .ok_or_else(|| anyhow::anyhow!("request encoded image byte capacity exceeded"))?;
                        ensure!(encoded_bytes <= 128 << 20, "request encoded image byte capacity exceeded");
                        let count = memo.geometry.validate()?.tokens();
                        ensure!(output.len().checked_add(count).is_some_and(|n| n <= capacity), "image prompt exceeds context capacity");
                        images.push(Descriptor { start: output.len(), geometry: memo.geometry, input: memo.identity });
                        output.extend(std::iter::repeat_n(IMAGE_TOKEN, count));
                        self.state.lock().map_err(|_| anyhow::anyhow!("image input cache poisoned"))?.touch_stored(&memo.identity);
                        deferred.push(DeferredInput { identity: memo.identity, geometry: memo.geometry,
                            source: DeferredSource::Request(source.clone()), cache: cache.clone() });
                        continue;
                    }
                }
                let internal = match source {
                    ImageSource::Url { url, .. } if references.is_some() && url.starts_with("http://titanic-context.invalid/") => Some(url),
                    _ => None,
                };
                let (input, source_bytes) = if let Some(url) = internal {
                    let reference = references.unwrap().iter().find(|r| r.source == *url)
                        .ok_or_else(|| anyhow::anyhow!("context_image_miss"))?;
                    ensure!(reference.detail == source.detail()
                        && (1..=MAX_ENCODED_BYTES).contains(&reference.encoded_bytes), "invalid cached image reference");
                    let identity = reference.identity.iter().map(|b|format!("{b:02x}")).collect::<String>();
                    ensure!(url == &format!("http://titanic-context.invalid/{identity}"), "cached image hash differs");
                    match (&reference.input, &reference.stored) {
                        (Some(input), _) => {
                            ensure!(input.identity == reference.identity, "cached image hash differs");
                            input.validate()?;
                            (input.clone(), reference.encoded_bytes)
                        }
                        // Patches evicted, encoded bytes kept: plan it like a deferred full-source image (the
                        // worker makes its pixels only if its cached prefix does not cover it) or, on the eager
                        // path, prepare it now from the stored bytes.
                        (None, Some(stored)) => {
                            ensure!(stored.bytes.len() == reference.encoded_bytes && stored.canonical == reference.detail,
                                "stored image reference differs");
                            if let Some(cache) = deferred_cache {
                                encoded_bytes = encoded_bytes.checked_add(stored.bytes.len())
                                    .ok_or_else(|| anyhow::anyhow!("request encoded image byte capacity exceeded"))?;
                                ensure!(encoded_bytes <= 128 << 20, "request encoded image byte capacity exceeded");
                                let count = stored.geometry.validate()?.tokens();
                                ensure!(output.len().checked_add(count).is_some_and(|n| n <= capacity), "image prompt exceeds context capacity");
                                images.push(Descriptor { start: output.len(), geometry: stored.geometry, input: reference.identity });
                                output.extend(std::iter::repeat_n(IMAGE_TOKEN, count));
                                deferred.push(DeferredInput { identity: reference.identity, geometry: stored.geometry,
                                    source: DeferredSource::Stored(stored.clone()), cache: cache.clone() });
                                continue;
                            }
                            let input = self.prepare_with_optional_pins(&stored.bytes, stored.detail, &mut pins)?;
                            ensure!(input.identity == reference.identity && input.geometry == stored.geometry,
                                "stored image identity or geometry differs");
                            (input, reference.encoded_bytes)
                        }
                        (None, None) => anyhow::bail!("context_image_miss"),
                    }
                } else {
                    let cached = source_key.as_ref().map(|key| self.source_hit(key, (128usize << 20) - encoded_bytes)).transpose()?.flatten();
                    if let Some(input) = cached { input } else {
                        let (bytes, detail) = self.resolve(source)?;
                        ensure!(!cancelled(), "request cancelled during image processing");
                        ensure!(encoded_bytes.saturating_add(bytes.len()) <= 128 << 20,
                            "request encoded image byte capacity exceeded");
                        let input = self.prepare_with_optional_pins(&bytes, detail, &mut pins)?;
                        if let Some(key) = source_key {
                            self.state.lock().map_err(|_| anyhow::anyhow!("image input cache poisoned"))?
                                .remember_source(key, &input, bytes.len());
                        }
                        (input, bytes.len())
                    }
                };
                ensure!(!cancelled(), "request cancelled during image processing");
                encoded_bytes = encoded_bytes.checked_add(source_bytes)
                    .ok_or_else(|| anyhow::anyhow!("request encoded image byte capacity exceeded"))?;
                ensure!(
                    encoded_bytes <= 128 << 20,
                    "request encoded image byte capacity exceeded"
                );
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
                if references.is_some() {
                    if let Some(source_hash) = source_key {
                        aliases.push(crate::protocol::ImageAlias{
                            source_hash,identity:input.identity});
                    }
                }
                inputs.push(input);
            }
            ensure!(
                output.len() <= capacity,
                "image prompt exceeds context capacity"
            );
        }
        ensure!(!cancelled(), "request cancelled during image processing");
        if !deferred.is_empty() {
            self.state.lock().map_err(|_| anyhow::anyhow!("image input cache poisoned"))?
                .metadata_deferrals += deferred.len() as u64;
        }
        Ok(Planned {
            tokens: output,
            images,
            inputs,
            deferred,
            aliases,
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
            "live_bytes":state.live_bytes(),"warm_budget_bytes":WARM_BYTES,"live_budget_bytes":LIVE_BYTES,
            "source_hits":state.source_hits,"source_entries":state.sources.len(),"source_entry_limit":SOURCE_ENTRIES,
            "metadata_deferrals":state.metadata_deferrals,"stored_sources":state.stored.len(),"stored_bytes":state.stored_bytes,
            "store_budget_bytes":self.store_budget,"stored_references":state.stored_references}),
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
    fn data_source(bytes: &[u8], detail: ImageDetail, suffix: &str) -> ImageSource {
        use base64::Engine as _;
        ImageSource::DataUrl {
            data_url: format!("data:image/png{suffix};base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes)),
            detail,
            path: String::new(),
        }
    }
    #[test]
    fn deferred_metadata_survives_patch_eviction_and_materializes_exactly() {
        let cache = Arc::new(ImageInputCache::for_test());
        let sources = [data_source(&png(61), ImageDetail::Auto, ""),
            data_source(&png(62), ImageDetail::Auto, "")];
        let tokens = vec![17, IMAGE_TOKEN, 18, IMAGE_TOKEN, 19];
        let cold = cache.expand(tokens.clone(), &sources, 4096).unwrap();
        let expected_tokens = cold.tokens.clone(); let expected_images = cold.images.clone();
        let expected_patches: Vec<_> = cold.inputs.iter().map(|i| i.patches.clone()).collect();
        let weak: Vec<_> = cold.inputs.iter().map(Arc::downgrade).collect();
        drop(cold); cache.forget_for_test();
        assert!(weak.iter().all(|p| p.upgrade().is_none()));
        let plan = cache.plan_cancellable(tokens, &sources, 4096, &|| false).unwrap();
        assert_eq!(plan.tokens, expected_tokens); assert_eq!(plan.images, expected_images);
        assert!(plan.inputs.is_empty()); assert_eq!(plan.deferred.len(), 2);
        assert_eq!(cache.metadata().unwrap()["live_bytes"], 0);
        assert_eq!(cache.metadata().unwrap()["misses"], 2);
        for (i, deferred) in plan.deferred.iter().enumerate() {
            let input = deferred.materialize(&|| false).unwrap();
            assert_eq!(input.identity, expected_images[i].input);
            assert_eq!(input.geometry, expected_images[i].geometry);
            assert_eq!(input.patches, expected_patches[i]);
        }
        assert_eq!(cache.metadata().unwrap()["misses"], 4);
    }
    #[test]
    fn deferred_materialization_preserves_physical_admission_and_cancellation() {
        let mut cache = ImageInputCache::for_test();
        let source = data_source(&png(63), ImageDetail::Auto, "");
        drop(cache.expand(vec![IMAGE_TOKEN], std::slice::from_ref(&source), 4096).unwrap());
        cache.forget_for_test(); cache.available_memory = || Ok(8usize << 30);
        let cache = Arc::new(cache);
        let plan = cache.plan_cancellable(vec![IMAGE_TOKEN], &[source], 4096, &|| false).unwrap();
        assert_eq!(plan.deferred.len(), 1); assert!(plan.inputs.is_empty());
        assert!(plan.deferred[0].materialize(&|| true).err().unwrap().to_string().contains("cancelled"));
        assert!(plan.deferred[0].materialize(&|| false).err().unwrap()
            .downcast_ref::<crate::memory::InputCapacityError>().is_some());
        assert_eq!(cache.metadata().unwrap()["live_bytes"], 0);
    }
    #[test]
    fn deferred_hits_preserve_detail_encoded_budget_and_context_limits() {
        let cache = Arc::new(ImageInputCache::for_test());
        let bytes = png(64); let original = data_source(&bytes, ImageDetail::Auto, "");
        let cold = cache.expand(vec![IMAGE_TOKEN], std::slice::from_ref(&original), 4096).unwrap();
        let identity = cold.images[0].input; drop(cold); cache.forget_for_test();
        assert!(cache.plan_cancellable(vec![IMAGE_TOKEN], std::slice::from_ref(&original), 1, &|| false).is_err());
        assert!(cache.plan_cancellable(vec![IMAGE_TOKEN], std::slice::from_ref(&original), 4096, &|| true).is_err());
        let changed = cache.plan_cancellable(vec![IMAGE_TOKEN], &[data_source(&bytes, ImageDetail::High, "")], 4096, &|| false).unwrap();
        assert!(changed.deferred.is_empty()); assert_ne!(changed.images[0].input, identity);
        let ImageSource::DataUrl {data_url, detail, ..} = &original else { unreachable!() };
        let key = crate::protocol::image_source_hash(data_url, *detail);
        cache.state.lock().unwrap().sources.get_mut(&key).unwrap().encoded_bytes = MAX_ENCODED_BYTES;
        let count = (128usize << 20) / MAX_ENCODED_BYTES + 1;
        let sources = vec![original; count];
        let error = cache.plan_cancellable(vec![IMAGE_TOKEN; count], &sources, 1048576, &|| false).err().unwrap();
        assert!(error.to_string().contains("encoded image byte capacity"));
        assert_eq!(cache.metadata().unwrap()["misses"], 2);
    }
    #[test]
    fn request_pins_preserve_later_hits_without_expanding_warm_retention() {
        let cache = ImageInputCache::for_test();
        let sources: Vec<_> = [51, 52, 53].iter()
            .map(|&color| data_source(&png(color), ImageDetail::Auto, "")).collect();
        let first = cache.expand(vec![IMAGE_TOKEN; 3], &sources, 4096).unwrap();
        let expected: Vec<_> = first.inputs.iter().map(|input| input.identity).collect();
        let weak: Vec<_> = first.inputs.iter().map(Arc::downgrade).collect();
        let keys: Vec<_> = sources.iter().map(|source| match source {
            ImageSource::DataUrl { data_url, detail, .. } =>
                Some(crate::protocol::image_source_hash(data_url, *detail)),
            _ => unreachable!(),
        }).collect();
        drop(first);
        let pins = cache.pin_source_hits(&keys).unwrap();
        cache.forget_for_test();
        assert_eq!(cache.metadata().unwrap()["warm_bytes"], 0);
        assert_eq!(cache.metadata().unwrap()["hits"], 0);
        let repeat = cache.expand(vec![IMAGE_TOKEN; 3], &sources, 4096).unwrap();
        assert_eq!(repeat.inputs.iter().map(|p| p.identity).collect::<Vec<_>>(), expected);
        assert_eq!(cache.metadata().unwrap()["misses"], 3);
        assert_eq!(cache.metadata().unwrap()["source_hits"], 3);
        drop(repeat);
        cache.forget_for_test();
        assert!(weak.iter().all(|input| input.upgrade().is_some()));
        drop(pins);
        assert!(weak.iter().all(|input| input.upgrade().is_none()));
    }
    #[test]
    fn optional_request_pins_do_not_survive_failed_physical_admission() {
        let mut cache = ImageInputCache::for_test();
        let source = data_source(&png(54), ImageDetail::Auto, "");
        let first = cache.expand(vec![IMAGE_TOKEN], std::slice::from_ref(&source), 4096).unwrap();
        let weak = Arc::downgrade(&first.inputs[0]);
        let ImageSource::DataUrl { data_url, detail, .. } = &source else { unreachable!() };
        let mut pins = cache.pin_source_hits(&[Some(crate::protocol::image_source_hash(data_url, *detail))]).unwrap();
        drop(first);
        cache.available_memory = || Ok(8usize << 30);
        let error = cache.prepare_with_optional_pins(&png(55), Detail::Auto, &mut pins).err().unwrap();
        assert!(error.downcast_ref::<crate::memory::InputCapacityError>().is_some());
        assert!(pins.is_empty());
        assert!(weak.upgrade().is_none());
        assert_eq!(cache.metadata().unwrap()["live_bytes"], 0);
        assert_eq!(cache.metadata().unwrap()["misses"], 1);
    }
    #[test]
    fn stored_reference_resolves_after_patch_eviction_and_defers_pixels() {
        let mut cache = ImageInputCache::for_test();
        cache.store_budget = 64 << 20;
        let cache = Arc::new(cache);
        let bytes = png(61);
        let first = cache.expand(vec![IMAGE_TOKEN], &[data_source(&bytes, ImageDetail::Auto, "")], 4096).unwrap();
        let (identity, geometry, patches) = (first.inputs[0].identity, first.inputs[0].geometry, first.inputs[0].patches.clone());
        drop(first); cache.forget_for_test();
        assert!(cache.state.lock().unwrap().live.get(&identity).and_then(Weak::upgrade).is_none(), "patches evicted");
        // Where pixels may not be deferred, an evicted reference is still a context miss.
        assert!(cache.pin_references(&[identity], false).err().unwrap().downcast_ref::<crate::protocol::ContextImageMiss>().is_some());
        let refs = cache.pin_references(&[identity], true).unwrap();
        assert!(refs[0].input.is_none() && refs[0].stored.is_some() && refs[0].identity == identity);
        assert_eq!(refs[0].encoded_bytes, bytes.len());
        let url = ImageSource::Url { url: refs[0].source.clone(), detail: ImageDetail::Auto, path: String::new() };
        let planned = cache.plan_referenced(vec![IMAGE_TOKEN], std::slice::from_ref(&url), 4096, &|| false, &refs).unwrap();
        assert!(planned.inputs.is_empty() && planned.deferred.len() == 1);
        assert_eq!(planned.images[0].input, identity);
        assert_eq!(planned.tokens.len(), geometry.tokens());
        assert_eq!(cache.metadata().unwrap()["misses"], 1, "planning made no pixels");
        let made = planned.deferred[0].materialize(&|| false).unwrap();
        assert_eq!((made.identity, made.geometry), (identity, geometry));
        assert_eq!(made.patches, patches, "pixels made again from the stored bytes are the same");
        assert_eq!(cache.metadata().unwrap()["stored_references"], 1);
        // The eager path prepares the stored bytes at once.
        drop(made); drop(planned); cache.forget_for_test();
        let refs = cache.pin_references(&[identity], true).unwrap();
        let eager = cache.expand_referenced(vec![IMAGE_TOKEN], std::slice::from_ref(&url), 4096, &|| false, &refs).unwrap();
        assert_eq!(eager.inputs[0].identity, identity);
        assert_eq!(eager.inputs[0].patches, patches);
        // A live reference keeps its patches (no deferral).
        let refs = cache.pin_references(&[identity], true).unwrap();
        assert!(refs[0].input.is_some() && refs[0].stored.is_none());
    }
    #[test]
    fn source_store_is_off_by_default_and_bounded_oldest_out() {
        let cache = ImageInputCache::for_test();
        let a = cache.expand(vec![IMAGE_TOKEN], &[data_source(&png(62), ImageDetail::Auto, "")], 4096).unwrap();
        let id = a.inputs[0].identity; drop(a); cache.forget_for_test();
        if cache.store_budget == 0 {
            assert_eq!(cache.metadata().unwrap()["stored_sources"], 0);
            assert!(cache.pin_references(&[id], true).is_err(), "no store: a miss as before");
        }
        let mut cache = ImageInputCache::for_test();
        let sizes = [png(63).len(), png(64).len(), png(65).len()];
        cache.store_budget = sizes[1] + sizes[2];
        let ids = (63u8..66).map(|c| cache.expand(vec![IMAGE_TOKEN], &[data_source(&png(c), ImageDetail::Auto, "")], 4096)
            .unwrap().inputs[0].identity).collect::<Vec<_>>();
        let state = cache.state.lock().unwrap();
        assert_eq!(state.stored.len(), 2);
        assert!(!state.stored.contains_key(&ids[0]) && state.stored.contains_key(&ids[1]) && state.stored.contains_key(&ids[2]));
        assert!(state.stored_bytes <= cache.store_budget && state.stored_bytes == sizes[1] + sizes[2]);
        assert_eq!(state.stored_oldest.len(), 2);
    }
    #[test]
    fn exact_source_reuse_preserves_identity_detail_aliases_and_admission() {
        let cache = ImageInputCache::for_test();
        let bytes = png(41);
        let encoded = data_source(&bytes, ImageDetail::Auto, "");
        let first = cache.expand(vec![IMAGE_TOKEN], std::slice::from_ref(&encoded), 4096).unwrap();
        let repeat = cache.expand_referenced(vec![IMAGE_TOKEN], std::slice::from_ref(&encoded), 4096, &|| false, &[]).unwrap();
        assert!(Arc::ptr_eq(&first.inputs[0], &repeat.inputs[0]));
        assert_eq!(first.tokens, repeat.tokens);
        assert_eq!(repeat.aliases[0].identity, first.inputs[0].identity);
        let ImageSource::DataUrl { data_url, detail, .. } = &encoded else { unreachable!() };
        let key = crate::protocol::image_source_hash(data_url, *detail);
        assert_eq!(repeat.aliases[0].source_hash, key);
        assert!(cache.source_hit(&key, bytes.len() - 1).err().unwrap().to_string().contains("byte capacity"));
        assert!(cache.expand(vec![IMAGE_TOKEN], std::slice::from_ref(&encoded), 1).is_err());
        assert!(cache.expand_cancellable(vec![IMAGE_TOKEN], std::slice::from_ref(&encoded), 4096, &|| true).is_err());
        let raw = cache.expand(vec![IMAGE_TOKEN], &[source(bytes.clone())], 4096).unwrap();
        assert!(Arc::ptr_eq(&first.inputs[0], &raw.inputs[0]));
        let alternate = cache.expand(vec![IMAGE_TOKEN], &[data_source(&bytes, ImageDetail::Auto, ";variant=1")], 4096).unwrap();
        assert!(Arc::ptr_eq(&first.inputs[0], &alternate.inputs[0]));
        let changed_detail = cache.expand(vec![IMAGE_TOKEN], &[data_source(&bytes, ImageDetail::High, "")], 4096).unwrap();
        assert_ne!(first.inputs[0].identity, changed_detail.inputs[0].identity);
        assert_eq!(first.inputs[0].patches, changed_detail.inputs[0].patches);
        let changed_bytes = cache.expand(vec![IMAGE_TOKEN], &[data_source(&png(42), ImageDetail::Auto, "")], 4096).unwrap();
        assert_ne!(first.inputs[0].identity, changed_bytes.inputs[0].identity);
        assert_eq!(cache.metadata().unwrap()["misses"], 3);
    }
    #[test]
    fn source_metadata_is_bounded_and_does_not_keep_reclaimed_images_alive() {
        let mut cache = ImageInputCache::for_test();
        let bytes = png(43);
        let original = data_source(&bytes, ImageDetail::Auto, "");
        let first = cache.expand(vec![IMAGE_TOKEN], std::slice::from_ref(&original), 4096).unwrap();
        let weak = Arc::downgrade(&first.inputs[0]);
        for i in 0..SOURCE_ENTRIES + 1 {
            let other = cache.expand(vec![IMAGE_TOKEN], &[data_source(&bytes, ImageDetail::Auto, &format!(";variant={i}"))], 4096).unwrap();
            assert!(Arc::ptr_eq(&first.inputs[0], &other.inputs[0]));
        }
        {
            let state = cache.state.lock().unwrap();
            assert_eq!(state.sources.len(), SOURCE_ENTRIES);
            assert_eq!(state.source_oldest.len(), SOURCE_ENTRIES);
            assert_eq!(state.live_bytes(), first.inputs[0].bytes());
        }
        drop(first); cache.forget_for_test(); assert!(weak.upgrade().is_none());
        // A source alias cannot bypass physical admission after its patches die.
        cache.available_memory = || Ok(8usize << 30);
        assert!(cache.expand(vec![IMAGE_TOKEN], &[data_source(&bytes, ImageDetail::Auto, &format!(";variant={SOURCE_ENTRIES}"))], 4096)
            .err().unwrap().downcast_ref::<crate::memory::InputCapacityError>().is_some());
        assert_eq!(cache.metadata().unwrap()["live_bytes"], 0);
        cache.available_memory = || Ok(16usize << 30);
        let fresh = cache.expand(vec![IMAGE_TOKEN], &[original], 4096).unwrap();
        assert_eq!(fresh.inputs[0].identity, vision_input::identity(&bytes, Detail::Auto));
        assert_eq!(cache.metadata().unwrap()["misses"], 2);
    }
    #[test]
    fn concurrent_exact_sources_share_the_canonical_preparation() {
        let cache = ImageInputCache::for_test();
        let encoded = data_source(&png(44), ImageDetail::Auto, "");
        std::thread::scope(|scope| {
            let threads: Vec<_> = (0..4).map(|_| scope.spawn(|| cache.expand(vec![IMAGE_TOKEN], std::slice::from_ref(&encoded), 4096).unwrap())).collect();
            let results: Vec<_> = threads.into_iter().map(|thread| thread.join().unwrap()).collect();
            for result in &results[1..] { assert!(Arc::ptr_eq(&results[0].inputs[0], &result.inputs[0])); }
        });
        assert_eq!(cache.metadata().unwrap()["misses"], 1);
        assert_eq!(cache.metadata().unwrap()["hits"], 3);
        assert_eq!(cache.metadata().unwrap()["source_entries"], 1);
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
        assert_eq!(cache.metadata().unwrap()["source_entries"], 0);
        server.join().unwrap();
    }
    #[test]
    fn insufficient_memory_rejects_before_decoding_and_releases_only_warm_inputs() {
        let mut cache = ImageInputCache::for_test();
        let live = cache.prepare(&png(10), Detail::Auto).unwrap();
        // Nov 17: the decoder allowance is sized from the header (vision_input::decode_alloc_bound, at least 32 MiB),
        // no longer a fixed 512 MiB; 16 MiB above the floor is too little for any image.
        cache.available_memory = || Ok((8usize << 30) + (16 << 20));
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
        // ...and so does 256 MiB, which the fixed 512 MiB decoder allowance used to refuse.
        cache.available_memory = || Ok((8usize << 30) + (256 << 20));
        assert!(cache.prepare(&png(21), Detail::Auto).is_ok());
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
