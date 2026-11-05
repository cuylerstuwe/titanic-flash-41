//! Optional official Rust prompt/protocol support. It performs no inference
//! and has no Python, model-runtime proxy, or implicit tokenizer download.
use anyhow::{Result, ensure};
use deepseek_recipe::{
    openai::ChatCompletionRequest,
    request::{
        ConversationRequest, ConversionOptions, ModelVersion, OutputConstraint, ProtocolRequest,
        ReasoningContentBehavior,
    },
    response::{Accumulate, ProtocolResponse},
    stream::{InferenceChunk, InferenceFinishReason, PromptUsage, StreamProcessor},
};
use deepseek_recipe_encoding::{PromptEncoding, PromptTokenizer, TokenizerEncoder, v4::dsv41::DeepseekV41Encoding};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, sync::Arc};
use tokenizers::Tokenizer;
use tokio_stream::StreamExt;

pub const TOKENIZER_SHA256: &str =
    "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b";
pub const MODEL_NAME: &str = "DeepSeek-V4.1-Flash-1017";
const DEFAULT_MAX_NEW_TOKENS: u32 = 262144;
fn canonical_special_tokens() -> deepseek_recipe_core::special_tokens::SpecialTokensMap {
    // Recipe defaults to <｜image｜>; canonical1017 stores the same image ID
    // under this spelling. Override conversion/rendering, never user text.
    deepseek_recipe_core::special_tokens::SpecialTokensMap {
        image: "<｜deepseek_image｜>".into(),
        ..Default::default()
    }
}
pub struct Frontend {
    tokenizer: Arc<Tokenizer>,
    prompt_tokenizer: PromptTokenizer,
    region_cache: crate::prompt_regions::RegionCache,
    use_region_cache: bool,
    #[cfg(feature = "vision-protocol")]
    image_cache: Option<crate::image_input_cache::ImageInputCache>,
}
/// Already processed immutable image input for the experimental compact-context
/// path. These handles are supplied by trusted Rust cache ownership, never
/// deserialized from an ordinary HTTP request.
pub struct ImageReference {
    pub source: String,
    pub input: Arc<crate::worker_media::PatchInput>,
    pub detail: deepseek_recipe_core::multimodal::ImageDetail,
    pub encoded_bytes: usize,
}

#[derive(Clone, Copy)]
pub struct ImageAlias {
    pub source_hash: [u8; 32],
    pub identity: [u8; 32],
}
pub(crate) fn image_source_hash(url: &str, detail: deepseek_recipe_core::multimodal::ImageDetail) -> [u8;32] {
    use deepseek_recipe_core::multimodal::ImageDetail;
    let mut h=Sha256::new();h.update(b"titanic-context-image-source-v1\0");
    h.update([match detail {ImageDetail::Low=>0,ImageDetail::High=>1,ImageDetail::Auto=>2,ImageDetail::Original=>3}]);
    h.update(url.as_bytes());h.finalize().into()
}
#[derive(Debug)]
pub(crate) struct ContextImageMiss;
impl std::fmt::Display for ContextImageMiss {
    fn fmt(&self, f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {f.write_str("context_image_miss")}
}
impl std::error::Error for ContextImageMiss {}

pub struct PreparedChat {
    pub converted: ConversationRequest,
    pub tokens: Vec<u32>,
    pub max_new_tokens: u32,
    pub rendered_sha256: String,
    pub images: Vec<crate::image_cache::Descriptor>,
    pub image_inputs: Vec<Arc<crate::worker_media::PatchInput>>,
    pub image_aliases: Vec<ImageAlias>,
}

#[cfg(feature = "serving")]
pub(crate) enum BackendEvent {
    Ready {
        fingerprint: String,
        cached_tokens: usize,
    },
    Tokens(Vec<u32>),
    Finish(InferenceFinishReason),
    Failed(String),
}

#[cfg(feature = "serving")]
impl Frontend {
    /// Recipe deliberately flushes a stream on bare EOF. An inference worker
    /// failure must not acquire that normal protocol finish: inspect the input
    /// terminal state before forwarding any chunks produced by its EOF flush.
    pub(crate) fn live_completion(
        &self,
        prepared: PreparedChat,
        id: String,
        mut incoming: tokio::sync::mpsc::Receiver<BackendEvent>,
    ) -> impl tokio_stream::Stream<
        Item = Result<deepseek_recipe::openai::chat_completion::response::ChatCompletionChunk>,
    > + Send
    + 'static {
        use std::sync::{Arc, Mutex};
        let terminal: Arc<Mutex<Option<std::result::Result<(), String>>>> =
            Arc::new(Mutex::new(None));
        let observed = terminal.clone();
        let generator =
            ChatCompletionRequest::chunk_generator(&prepared.converted, id, MODEL_NAME.into());
        let processor = StreamProcessor::new(generator, prepared.converted.parsing_options)
            .with_tokenizer(self.tokenizer.clone());
        let prompt_tokens = prepared.tokens.len();
        let limit = prepared.max_new_tokens as usize;
        let inference = async_stream::stream! {
            let mut ready = false;
            let mut eos = false;
            let mut emitted = 0usize;
            while let Some(event) = incoming.recv().await {
                match event {
                    BackendEvent::Ready { fingerprint, cached_tokens } if !ready && cached_tokens <= prompt_tokens => {
                        ready = true;
                        yield InferenceChunk::Ready { system_fingerprint:Some(fingerprint),
                            prompt_usage:PromptUsage {prompt_tokens,prompt_cache_hit_tokens:cached_tokens} };
                    }
                    BackendEvent::Tokens(mut ids) if ready && !eos && !ids.is_empty()
                        && ids.len()<=4096 && emitted+ids.len()<=limit
                        && ids.iter().all(|&id|id<129280)
                        && !ids[..ids.len()-1].contains(&1) => {
                        emitted += ids.len();
                        eos = ids.last()==Some(&1);
                        if eos { ids.pop(); }
                        if !ids.is_empty() { yield InferenceChunk::Tokens {token_ids:ids,logprobs:None}; }
                    }
                    BackendEvent::Finish(reason) if ready && emitted>0 && match reason {
                        InferenceFinishReason::Stop => eos,
                        InferenceFinishReason::Length => !eos && emitted==limit,
                        _ => false,
                    } => {
                        *terminal.lock().unwrap()=Some(Ok(()));
                        yield InferenceChunk::Finish {finish_reason:reason};
                        return;
                    }
                    BackendEvent::Failed(message) => {
                        *terminal.lock().unwrap()=Some(Err(message));
                        return;
                    }
                    _ => {
                        *terminal.lock().unwrap()=Some(Err("worker output violated the request frontier".into()));
                        return;
                    }
                }
            }
            *terminal.lock().unwrap()=Some(Err("worker stream ended without committed completion".into()));
        };
        async_stream::stream! {
            let stream=processor.into_protocol_stream(inference);
            let mut stream=std::pin::pin!(stream);
            while let Some(chunk)=stream.next().await {
                let failure=observed.lock().unwrap().as_ref().and_then(|result|result.as_ref().err().cloned());
                if let Some(message)=failure { yield Err(anyhow::anyhow!(message));return; }
                match chunk {
                    Ok(chunk)=>yield Ok(chunk),
                    Err(error)=>{yield Err(error.into());return;}
                }
            }
            if !matches!(*observed.lock().unwrap(),Some(Ok(()))) {
                yield Err(anyhow::anyhow!("protocol ended before a committed backend finish"));
            }
        }
    }
}

/// Recipe 0.2.0 deliberately ignores unknown OpenAI fields and leaves several
/// inference controls to its caller. Admit only controls this engine can honor;
/// successful prompt conversion must not imply constrained decoding support.
fn normalize_chat_request(body: &[u8]) -> Result<Value> {
    let mut request: Value = serde_json::from_slice(body)?;
    let object = request
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("chat request must be an object"))?;
    for key in object.keys() {
        ensure!(
            matches!(
                key.as_str(),
                "messages"
                    | "model"
                    | "frequency_penalty"
                    | "presence_penalty"
                    | "logprobs"
                    | "top_logprobs"
                    | "max_tokens"
                    | "max_completion_tokens"
                    | "n"
                    | "seed"
                    | "stop"
                    | "stream"
                    | "stream_options"
                    | "temperature"
                    | "top_p"
                    | "top_k"
                    | "response_format"
                    | "tools"
                    | "tool_choice"
                    | "reasoning_effort"
                    | "thinking"
                    | "chat_template_kwargs"
                    | "parallel_tool_calls"
            ),
            "unsupported native chat field: {key}"
        );
    }
    // Optional nulls mean unspecified, as in the Recipe request schema.
    object.retain(|key, value| key == "messages" || key == "model" || !value.is_null());
    if let Some(kwargs) = object.remove("chat_template_kwargs") {
        let kwargs = kwargs.as_object().ok_or_else(|| anyhow::anyhow!("chat_template_kwargs must be an object"))?;
        ensure!(kwargs.keys().all(|key| key == "thinking" || key == "enable_thinking" || key == "reasoning_effort"),
            "unsupported chat template option");
        // The Canada router gives an explicit template effort precedence over
        // the top-level default. Recipe still validates the effort's domain.
        if let Some(effort) = kwargs.get("reasoning_effort") {
            object.insert("reasoning_effort".into(), effort.clone());
        }
        let mut enabled = None;
        for value in ["thinking", "enable_thinking"].iter().filter_map(|key| kwargs.get(*key)) {
            let value = value.as_bool().ok_or_else(|| anyhow::anyhow!("thinking template option must be boolean"))?;
            ensure!(enabled.is_none_or(|prior| prior == value), "conflicting thinking template options");
            enabled = Some(value);
        }
        if let Some(enabled) = enabled {
            let thinking = json!({"type":if enabled {"enabled"} else {"disabled"}});
            ensure!(object.get("thinking").is_none_or(|value| *value == thinking),
                "conflicting thinking controls");
            object.insert("thinking".into(), thinking);
        }
    }
    if let Some(limit) = object.remove("max_completion_tokens") {
        ensure!(
            !object.contains_key("max_tokens"),
            "specify only one of max_tokens and max_completion_tokens"
        );
        object.insert("max_tokens".into(), limit);
    }
    object.entry("model").or_insert(json!(MODEL_NAME));
    object.entry("reasoning_effort").or_insert(json!("high"));
    // -1 is the router's explicit "no top-k truncation" setting.
    if let Some(top_k) = object.remove("top_k") {
        ensure!(top_k.as_f64() == Some(-1.), "native protocol requires top_k=-1 when specified");
    }
    ensure!(object.get("temperature").is_none_or(|v| matches!(v.as_f64(), Some(0. | 1.))),
        "native protocol currently supports temperature=0 or1");
    ensure!(object.get("top_p").is_none_or(|v| v.as_f64() == Some(1.)),
        "native protocol currently requires top_p=1");
    object.entry("temperature").or_insert(json!(1.));
    object.entry("top_p").or_insert(json!(1.));
    for key in ["frequency_penalty", "presence_penalty"] {
        ensure!(
            object.get(key).is_none_or(|v| v.as_f64() == Some(0.)),
            "native protocol does not implement {key}"
        );
    }
    ensure!(
        object
            .get("logprobs")
            .is_none_or(|v| v.as_bool() == Some(false))
            && !object.contains_key("top_logprobs"),
        "native protocol does not yet return token log probabilities"
    );
    ensure!(
        object
            .get("parallel_tool_calls")
            .is_none_or(|v| v.as_bool() == Some(true)),
        "native protocol cannot enforce parallel_tool_calls=false"
    );
    ensure!(
        object
            .get("tool_choice")
            .is_none_or(|v| matches!(v.as_str(), Some("auto" | "none"))),
        "native protocol does not yet enforce required or named tool choices"
    );
    ensure!(
        object
            .get("response_format")
            .is_none_or(|v| v["type"].as_str() == Some("text")),
        "native protocol does not yet enforce JSON, schema, or regex output constraints"
    );
    ensure!(
        object
            .get("stop")
            .is_none_or(|v| v.as_array().is_some_and(Vec::is_empty)),
        "custom stop sequences await coordinated backend termination support"
    );
    ensure!(
        object
            .get("thinking")
            .is_none_or(|v| v.get("budget_tokens").is_none_or(Value::is_null)),
        "native protocol does not implement a separate thinking token budget"
    );
    if let Some(tools) = object.get("tools").and_then(Value::as_array) {
        ensure!(
            tools.iter().all(|tool| tool["function"]["strict"] != true),
            "native protocol does not yet enforce strict tool schemas"
        );
    }
    Ok(request)
}

fn resolve_native_max_tokens(converted: &mut ConversationRequest, prompt_tokens: u32, capacity: u32) -> Result<u32> {
    if converted.inference_options.max_tokens.is_none() {
        // Validate an unspecified completion against the prompt alone before
        // choosing our existing 256K default. Near the ceiling, use the room
        // actually left instead of rejecting a valid long conversation.
        converted.resolve_max_tokens(prompt_tokens, capacity)?;
        converted.inference_options.max_tokens = Some(DEFAULT_MAX_NEW_TOKENS.min(capacity - prompt_tokens).max(1));
    }
    // The native owner needs one available position. At a completely full
    // prompt this returns the canonical context error before GPU admission.
    Ok(converted.resolve_max_tokens(prompt_tokens, capacity)?)
}

impl Frontend {
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with_images(path, false)
    }
    pub fn load_with_images(path: &Path, images: bool) -> Result<Self> {
        ensure!(!images || cfg!(feature = "vision-protocol"), "binary lacks native image input support");
        let raw = std::fs::read(path)?;
        ensure!(
            format!("{:x}", Sha256::digest(&raw)) == TOKENIZER_SHA256,
            "tokenizer does not match pinned canonical1017 checkpoint"
        );
        let tokenizer =
            Tokenizer::from_bytes(&raw).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        ensure!(
            tokenizer.get_vocab_size(true) == 129280,
            "canonical tokenizer vocabulary size"
        );
        Ok(Self {
            prompt_tokenizer: PromptTokenizer::new(&tokenizer),
            region_cache: crate::prompt_regions::RegionCache::new(32 * 1024 * 1024),
            use_region_cache: match std::env::var("CANADA_SCRATCH_PROMPT_REGION_CACHE") {
                Ok(value) => { ensure!(value == "0" || value == "1", "invalid prompt-region cache policy"); value == "1" },
                Err(std::env::VarError::NotPresent) => true,
                Err(error) => return Err(error.into()),
            },
            tokenizer: Arc::new(tokenizer),
            #[cfg(feature = "vision-protocol")]
            image_cache: images.then(crate::image_input_cache::ImageInputCache::new),
        })
    }
    pub fn prepare_chat(&self, body: &[u8], context_capacity: usize) -> Result<PreparedChat> {
        self.prepare_chat_impl(body, context_capacity, if self.use_region_cache { 2 } else { 1 })
    }
    pub(crate) fn image_cache_metadata(&self) -> Result<Option<Value>> {
        #[cfg(feature = "vision-protocol")]
        { self.image_cache.as_ref().map(|cache|cache.metadata()).transpose() }
        #[cfg(not(feature = "vision-protocol"))]
        { Ok(None) }
    }
    #[cfg(feature = "vision-protocol")]
    pub(crate) fn pin_context_images(&self, keys: &[[u8;32]]) -> Result<Vec<ImageReference>> {
        if keys.is_empty() { return Ok(Vec::new()); }
        self.image_cache.as_ref().ok_or(ContextImageMiss)?.pin_references(keys)
    }
    #[cfg(all(test, feature = "vision-protocol"))]
    pub(crate) fn enable_test_images(&mut self) {self.image_cache=Some(crate::image_input_cache::ImageInputCache::for_test());}
    #[cfg(all(test, feature = "vision-protocol"))]
    pub(crate) fn forget_context_images_for_test(&self) {
        if let Some(cache)=&self.image_cache {cache.forget_for_test();}
    }
    pub(crate) fn prepare_chat_cancellable(&self, body: &[u8], context_capacity: usize,
        cancelled: &dyn Fn() -> bool) -> Result<PreparedChat> {
        self.prepare_chat_inner(body, context_capacity, if self.use_region_cache { 2 } else { 1 }, cancelled, None)
    }
    // The legacy branch is retained solely for a bounded CPU comparison. Both
    // branches use Recipe 0.2.0's canonical region-aware tokenizer unchanged.
    fn prepare_chat_impl(&self, body: &[u8], context_capacity: usize, mode: u8) -> Result<PreparedChat> {
        self.prepare_chat_inner(body, context_capacity, mode, &|| false, None)
    }
    /// Component probe only: preserve the full canonical converter and region
    /// tokenizer while replacing old encoded images with pinned processed inputs.
    /// The HTTP adapter does not accept these internal references yet.
    #[doc(hidden)]
    #[cfg(feature = "vision-protocol")]
    pub fn prepare_chat_referenced(&self, body: &[u8], context_capacity: usize,
        references: &[ImageReference], cancelled: &dyn Fn() -> bool) -> Result<PreparedChat> {
        self.prepare_chat_inner(body, context_capacity, if self.use_region_cache { 2 } else { 1 }, cancelled, Some(references))
    }
    fn prepare_chat_inner(&self, body: &[u8], context_capacity: usize, mode: u8,
        cancelled: &dyn Fn() -> bool, _references: Option<&[ImageReference]>) -> Result<PreparedChat> {
        ensure!(!cancelled(), "request cancelled during input processing");
        ensure!(
            context_capacity > 0 && context_capacity <= crate::config::MODEL_CONTEXT,
            "protocol context capacity"
        );
        let request = normalize_chat_request(body)?;
        let options = ConversionOptions::default()
            .with_special_tokens(canonical_special_tokens())
            .with_model_alias(MODEL_NAME, ModelVersion::DeepseekV41Flash)
            .with_model_alias("canada", ModelVersion::DeepseekV41Flash)
            .with_model_alias("deepseek-v4.1-flash-1017", ModelVersion::DeepseekV41Flash)
            .with_reasoning_content_behavior(ReasoningContentBehavior::FillEmpty)
            .with_thinking_max_tokens_upper_bound(262144)
            .with_non_thinking_max_tokens_upper_bound(262144);
        let mut converted =
            serde_json::from_value::<ChatCompletionRequest>(request)?.convert(options)?;
        ensure!(
            converted.model_version == ModelVersion::DeepseekV41Flash,
            "request names a different model family"
        );
        ensure!(
            !converted.inference_options.logprobs
                && converted.inference_options.top_logprobs.is_none()
                && converted.parsing_options.stop_sequences.is_empty()
                && match &converted.inference_options.output_constraint {
                    None => true,
                    Some(OutputConstraint::ToolCalls {
                        tools,
                        required: false,
                    }) => tools.iter().all(|tool| tool.strict != Some(true)),
                    _ => false,
                },
            "converted request requires an unsupported native inference control"
        );
        let encoding = DeepseekV41Encoding::new().with_special_tokens(canonical_special_tokens());
        let rendered = encoding.render_conversation(&converted.conversation);
        let tokens = if mode == 2 {
            self.region_cache.encode(&self.prompt_tokenizer, &self.tokenizer, &rendered.prompt, &rendered.special_ranges)
                .map_err(anyhow::Error::msg)?
        } else if mode == 1 {
            self.prompt_tokenizer.encode_ids(&rendered.prompt, &rendered.special_ranges)
                .map_err(anyhow::Error::msg)?
        } else {
            encoding.with_tokenizer(PromptTokenizer::new(&self.tokenizer))
                .encode(&converted.conversation)?
        };
        ensure!(
            !tokens.is_empty() && tokens.iter().all(|&id| id < 129280),
            "encoded prompt token domain"
        );
        let (tokens, images, image_inputs, image_aliases) = if rendered.image_sources.is_empty() {
            (tokens, Vec::new(), Vec::new(), Vec::new())
        } else {
            #[cfg(feature = "vision-protocol")]
            {
                let cache = self.image_cache.as_ref().ok_or_else(|| anyhow::anyhow!("native vision is not enabled"))?;
                let expanded = match _references {
                    None => cache.expand_cancellable(tokens, &rendered.image_sources, context_capacity, cancelled)?,
                    Some(references) => cache.expand_referenced(tokens, &rendered.image_sources, context_capacity, cancelled, references)?,
                };
                (expanded.tokens, expanded.images, expanded.inputs, expanded.aliases)
            }
            #[cfg(not(feature = "vision-protocol"))]
            { anyhow::bail!("vision inputs await native vision qualification"); }
        };
        ensure!(!cancelled(), "request cancelled during input processing");
        let max_new_tokens = resolve_native_max_tokens(&mut converted, tokens.len() as u32, context_capacity as u32)?;
        ensure!(
            matches!(converted.inference_options.temperature, Some(0. | 1.))
                && converted.inference_options.top_p == Some(1.),
            "converted request changed admitted target sampling"
        );
        Ok(PreparedChat {
            converted,
            tokens,
            max_new_tokens,
            images,
            image_inputs,
            image_aliases,
            rendered_sha256: format!("{:x}", Sha256::digest(rendered.prompt.as_bytes())),
        })
    }
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, false)
            .map_err(|e| anyhow::anyhow!("token decode: {e}"))
    }
    /// Parse continuation tokens with the same official options that encoded
    /// the request. Literal marker spellings remain content, and UTF-8 may span
    /// arbitrary incoming token chunks. EOS is backend metadata, not text.
    pub async fn parse_completion(
        &self,
        prepared: &PreparedChat,
        tokens: &[u32],
        finish: &str,
        chunk_size: usize,
    ) -> Result<Value> {
        ensure!(
            (1..=4096).contains(&chunk_size) && tokens.iter().all(|&t| t < 129280),
            "response token/chunk domain"
        );
        let finish_reason = match finish {
            "stop" => InferenceFinishReason::Stop,
            "length" => InferenceFinishReason::Length,
            _ => anyhow::bail!("unsupported backend finish reason"),
        };
        let content = if tokens.last() == Some(&1) {
            ensure!(finish == "stop", "EOS contradicts length finish");
            &tokens[..tokens.len() - 1]
        } else {
            tokens
        };
        ensure!(!content.contains(&1), "response contains tokens after EOS");
        let generator = ChatCompletionRequest::chunk_generator(
            &prepared.converted,
            "native-completion".into(),
            MODEL_NAME.into(),
        );
        let processor = StreamProcessor::new(generator, prepared.converted.parsing_options.clone())
            .with_tokenizer(self.tokenizer.clone());
        let mut inference = vec![InferenceChunk::Ready {
            system_fingerprint: Some("canonical-1017".into()),
            prompt_usage: PromptUsage {
                prompt_tokens: prepared.tokens.len(),
                prompt_cache_hit_tokens: 0,
            },
        }];
        inference.extend(content.chunks(chunk_size).map(|c| InferenceChunk::Tokens {
            token_ids: c.to_vec(),
            logprobs: None,
        }));
        inference.push(InferenceChunk::Finish { finish_reason });
        type Response = <ChatCompletionRequest as ProtocolRequest>::Response;
        let mut response = Response::new(
            "native-completion".into(),
            MODEL_NAME.into(),
            0,
            prepared.tokens.len(),
            0,
        );
        let mut chunks =
            std::pin::pin!(processor.into_protocol_stream(tokio_stream::iter(inference)));
        while let Some(chunk) = chunks.next().await {
            response.accumulate(chunk?);
        }
        Ok(serde_json::to_value(response)?)
    }
}
/// CPU-only protocol screen. No CUDA context, model arithmetic or requests to
/// an inference server; compare the former frontend with the exact reuse path.
pub fn frontend_benchmark(tokenizer: &Path, requests: &Path) -> Result<Value> {
    use std::time::Instant;
    let frontend = Frontend::load(tokenizer)?;
    let fixture: Value = serde_json::from_slice(&std::fs::read(requests)?)?;
    let cases = fixture["cases"].as_array().ok_or_else(|| anyhow::anyhow!("missing frontend cases"))?;
    ensure!(!cases.is_empty() && cases.len() <= 16, "bounded frontend screen");
    let mut records = Vec::new();
    for case in cases {
        let body = serde_json::to_vec(&case["request"])?;
        let mut reference: Option<PreparedChat> = None;
        let mut pairs = Vec::new();
        for repeat in 0..4 {
            let mut times = [0.0; 3];
            for mode in if repeat % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
                let start = Instant::now();
                let prepared = frontend.prepare_chat_impl(&body, crate::config::MODEL_CONTEXT, mode as u8)?;
                times[mode] = start.elapsed().as_secs_f64();
                if let Some(prior) = &reference {
                    ensure!(prior.tokens == prepared.tokens && prior.rendered_sha256 == prepared.rendered_sha256
                        && prior.max_new_tokens == prepared.max_new_tokens, "frontend changed canonical prompt");
                } else { reference = Some(prepared); }
            }
            pairs.push(json!({"repeat":repeat,"legacy_seconds":times[0],"reused_seconds":times[1],"region_cache_seconds":times[2]}));
        }
        let prepared = reference.unwrap();
        let start = Instant::now();
        let tokenizer = PromptTokenizer::new(&frontend.tokenizer);
        let tokenizer_setup = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let rendered = DeepseekV41Encoding::new().render_conversation(&prepared.converted.conversation);
        let render = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let ids = tokenizer.encode_ids(&rendered.prompt, &rendered.special_ranges).map_err(anyhow::Error::msg)?;
        let encode = start.elapsed().as_secs_f64();
        ensure!(ids == prepared.tokens, "isolated canonical tokenization changed IDs");
        records.push(json!({"name":case["name"],"body_bytes":body.len(),"prompt_tokens":ids.len(),
            "regions":rendered.special_ranges.len()+1,"pairs":pairs,"tokenizer_setup_seconds":tokenizer_setup,
            "render_seconds":render,"encode_seconds":encode,"all_prompt_ids_exact":true,
            "region_cache":frontend.region_cache.metadata().map_err(anyhow::Error::msg)?}));
    }
    Ok(json!({"records":records,"tokenizer_sha256":TOKENIZER_SHA256,"recipe_version":"0.2.0",
        "scope":"CPU protocol microbenchmark, not end-to-end TTFT. Four reversed-order pairs; first pair includes cold caches. No model arithmetic or sampling changes."}))
}

pub fn encode_report(tokenizer: &Path, request: &Path) -> Result<Value> {
    let frontend = Frontend::load(tokenizer)?;
    let p = frontend.prepare_chat(&std::fs::read(request)?, crate::config::MODEL_CONTEXT)?;
    Ok(
        json!({"input_ids":p.tokens,"prompt_tokens":p.tokens.len(),"max_new_tokens":p.max_new_tokens,
        "temperature":p.converted.inference_options.temperature,"top_p":p.converted.inference_options.top_p,
        "thinking_mode":p.converted.conversation.thinking_mode,"rendered_sha256":p.rendered_sha256,
        "tokenizer_sha256":TOKENIZER_SHA256,"recipe_version":"0.2.0","inference_executed":false,
        "scope":"Rust protocol conversion and canonical1017 tokenizer only; no model execution or serving acceptance."}),
    )
}

/// Decode an immutable completion's raw tokens for human/task evaluation.
/// Output retains control markers; no protocol stream parsing is implied.
pub fn decode_report(tokenizer: &Path, completion: &Path) -> Result<Value> {
    let f = Frontend::load(tokenizer)?;
    let record: Value = serde_json::from_slice(&std::fs::read(completion)?)?;
    let ids: Vec<u32> =
        serde_json::from_value(record.get("output_ids").unwrap_or(&record).clone())?;
    ensure!(
        ids.iter().all(|&t| t < 129280),
        "completion token outside canonical vocabulary"
    );
    Ok(
        json!({"output_text":f.decode(&ids)?,"completion_tokens":ids.len(),
        "tokenizer_sha256":TOKENIZER_SHA256,"inference_executed":false,
        "scope":"Raw canonical token decoding, including model control markers; no response parsing or task-quality verdict."}),
    )
}

pub fn parse_report(tokenizer: &Path, request: &Path, completion: &Path) -> Result<Value> {
    let f = Frontend::load(tokenizer)?;
    let request_body = std::fs::read(request)?;
    let mut prepared = f.prepare_chat(&request_body, crate::config::MODEL_CONTEXT)?;
    prepared.converted.stream = false;
    let record: Value = serde_json::from_slice(&std::fs::read(completion)?)?;
    let ids: Vec<u32> = serde_json::from_value(record["output_ids"].clone())?;
    let finish = record["finish_reason"]
        .as_str()
        .or_else(|| record["server_meta"]["finish_reason"]["type"].as_str())
        .ok_or_else(|| anyhow::anyhow!("completion needs explicit stop/length reason"))?;
    if let Some(prompt_tokens) = record["prompt_tokens"].as_u64() {
        ensure!(
            prompt_tokens == prepared.tokens.len() as u64,
            "completion prompt count differs from supplied request"
        );
    }
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    let response = runtime.block_on(f.parse_completion(&prepared, &ids, finish, 256))?;
    let body: Value = serde_json::from_slice(&request_body)?;
    let declared = body["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .filter_map(|t| t["function"]["name"].as_str())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let calls = response["choices"][0]["message"]["tool_calls"].as_array()
        .map(|calls|calls.iter().map(|call|{
            let name = call["function"]["name"].as_str().unwrap_or("");
            let arguments = call["function"]["arguments"].as_str().unwrap_or("");
            json!({"name":name,"name_declared":declared.contains(&name),
                "arguments_valid_json_object":serde_json::from_str::<Value>(arguments).is_ok_and(|v|v.is_object())})
        }).collect::<Vec<_>>()).unwrap_or_default();
    Ok(
        json!({"response":response,"raw_completion_tokens_including_eos":ids.len(),
        "raw_output_text":f.decode(&ids)?,"tool_call_syntax":calls,
        "tokenizer_sha256":TOKENIZER_SHA256,"recipe_version":"0.2.0",
        "scope":"Official token-aware protocol parsing and reasoning usage; tool name/JSON syntax diagnostics only, not schema/semantic correctness or task success."}),
    )
}

#[cfg(test)]
#[path = "protocol_recipe_tests.rs"]
mod recipe_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_default_uses_remaining_context_and_keeps_explicit_limits() {
        let convert = |extra: Value| {
            let mut body=json!({"model":MODEL_NAME,"messages":[{"role":"user","content":"hi"}]});
            body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            let value=normalize_chat_request(&serde_json::to_vec(&body).unwrap()).unwrap();
            serde_json::from_value::<ChatCompletionRequest>(value).unwrap()
                .convert(ConversionOptions::default().with_model_alias(MODEL_NAME,ModelVersion::DeepseekV41Flash)
                    .with_thinking_max_tokens_upper_bound(DEFAULT_MAX_NEW_TOKENS)).unwrap()
        };
        assert_eq!(resolve_native_max_tokens(&mut convert(json!({})),100,1048576).unwrap(),262144);
        assert_eq!(resolve_native_max_tokens(&mut convert(json!({})),1048576-17,1048576).unwrap(),17);
        assert_eq!(resolve_native_max_tokens(&mut convert(json!({"max_completion_tokens":7})),1048576-17,1048576).unwrap(),7);
        for (prompt,extra) in [(1048576,json!({})),(1048577,json!({})),(1048576-17,json!({"max_tokens":18}))] {
            let error=resolve_native_max_tokens(&mut convert(extra),prompt,1048576).unwrap_err();
            assert!(error.to_string().contains("This model's maximum context length is 1048576 tokens."));
        }
    }
    #[test]
    fn router_template_effort_and_unrestricted_top_k_are_admitted() {
        let request=normalize_chat_request(br#"{"messages":[],"reasoning_effort":"high","top_k":-1,"chat_template_kwargs":{"thinking":true,"enable_thinking":true,"reasoning_effort":"low"}}"#).unwrap();
        assert_eq!(request["reasoning_effort"],"low");
        assert_eq!(request["thinking"],json!({"type":"enabled"}));
        assert!(request.get("top_k").is_none()&&request.get("max_tokens").is_none());
    }
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn missing_tool_reasoning_uses_the_official_empty_history_policy() {
        let f=Frontend::load(Path::new(&std::env::var("CANADA_TEST_TOKENIZER").unwrap())).unwrap();
        let mut body=json!({"model":"deepseek-v4.1-flash-1017","max_tokens":32,
            "messages":[{"role":"user","content":"Read it."},
                {"role":"assistant","content":"","tool_calls":[{"id":"call_fixture","type":"function","function":{"name":"read","arguments":"{}"}}]},
                {"role":"tool","tool_call_id":"call_fixture","content":"done"}],
            "tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{}}}}]});
        let missing=f.prepare_chat(&serde_json::to_vec(&body).unwrap(),4096).unwrap();
        body["messages"][1]["reasoning_content"]=json!("");
        let empty=f.prepare_chat(&serde_json::to_vec(&body).unwrap(),4096).unwrap();
        assert_eq!(missing.tokens,empty.tokens);
        body["messages"][1]["reasoning_content"]=json!("Preserve this supplied reasoning.");
        let present=f.prepare_chat(&serde_json::to_vec(&body).unwrap(),4096).unwrap();
        assert_ne!(present.tokens,empty.tokens);
    }
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn tool_enabled_plain_reply_preserves_reasoning_but_normalizes_answer_whitespace() {
        let f = Frontend::load(Path::new(&std::env::var("CANADA_TEST_TOKENIZER").unwrap())).unwrap();
        let mut request = json!({"max_tokens":256,
            "messages":[{"role":"user","content":"Repeat blue green red yellow."}],
            "tools":[{"type":"function","function":{"name":"read_nonce",
                "parameters":{"type":"object","properties":{}}}}]});
        let first = f.prepare_chat(&serde_json::to_vec(&request).unwrap(), 4096).unwrap();
        let mut output = f.tokenizer.encode(
            "I can copy it directly.</think>\n\nblue green red yellow\nblue green red yellow",
            false).unwrap().get_ids().to_vec();
        output.push(1);
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let response = rt.block_on(f.parse_completion(&first, &output, "stop", 1)).unwrap();
        let assistant = response["choices"][0]["message"].clone();
        assert!(assistant.get("tool_calls").is_none_or(|value| value.is_null()));
        request["messages"].as_array_mut().unwrap().extend([
            assistant, json!({"role":"user","content":"Now give one copy."})]);
        let next = f.prepare_chat(&serde_json::to_vec(&request).unwrap(), 4096).unwrap();
        let mut complete = first.tokens;
        complete.extend(output);
        // The official parser trims the answer's leading newlines. Reasoning
        // stays present with tools, but this is not an exact full-prefix hit.
        // A resident scheduler must reset/recompute rather than reuse stale KV.
        let common = next.tokens.iter().zip(&complete).take_while(|(a,b)| a == b).count();
        assert!(!next.tokens.starts_with(&complete));
        assert_eq!(f.decode(&complete[common..]).unwrap(),
            "\n\nblue green red yellow\nblue green red yellow<｜end▁of▁sentence｜>");
        assert!(f.decode(&next.tokens[common..]).unwrap().starts_with("blue green red yellow"));
        assert!(f.decode(&next.tokens[..common]).unwrap().contains("I can copy it directly.</think>"));
    }
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn canonical_tool_result_continuation_preserves_the_complete_token_prefix() {
        let f =
            Frontend::load(Path::new(&std::env::var("CANADA_TEST_TOKENIZER").unwrap())).unwrap();
        let mut request = json!({"max_tokens":256,"messages":[{"role":"user","content":"Compute it."}],
            "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}}}]});
        let first = f
            .prepare_chat(&serde_json::to_vec(&request).unwrap(), 4096)
            .unwrap();
        let text = "I will call f.</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"a\" string=\"true\">hello 😀</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        let mut output = f.tokenizer.encode(text, false).unwrap().get_ids().to_vec();
        output.push(1);
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let response = rt
            .block_on(f.parse_completion(&first, &output, "stop", 1))
            .unwrap();
        let mut assistant = response["choices"][0]["message"].clone();
        let call_id = assistant["tool_calls"][0]["id"]
            .as_str()
            .unwrap()
            .to_string();
        for call in assistant["tool_calls"].as_array_mut().unwrap() {
            call.as_object_mut().unwrap().remove("index");
        }
        request["messages"].as_array_mut().unwrap().push(assistant);
        request["messages"]
            .as_array_mut()
            .unwrap()
            .push(json!({"role":"tool","tool_call_id":call_id,"content":"42"}));
        let next = f
            .prepare_chat(&serde_json::to_vec(&request).unwrap(), 4096)
            .unwrap();
        let mut complete = first.tokens;
        complete.extend_from_slice(&output);
        let common = next
            .tokens
            .iter()
            .zip(&complete)
            .take_while(|(a, b)| a == b)
            .count();
        assert!(
            next.tokens.starts_with(&complete),
            "tool loop prefix differs at{common}/{}; previous={:?}; next={:?}; assistant={}",
            complete.len(),
            f.decode(&complete[common..]).unwrap(),
            f.decode(&next.tokens[common..]).unwrap(),
            request["messages"][1]
        );
        request["messages"][1]
            .as_object_mut()
            .unwrap()
            .remove("reasoning_content");
        let missing=f.prepare_chat(&serde_json::to_vec(&request).unwrap(),4096).unwrap();
        assert!(!missing.tokens.starts_with(&complete),
            "empty reasoning must not falsely reuse the original reasoning prefix");
        request["messages"][1]["reasoning_content"]=json!("");
        assert_eq!(missing.tokens,f.prepare_chat(&serde_json::to_vec(&request).unwrap(),4096).unwrap().tokens);
    }
    #[cfg(feature = "serving")]
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn live_worker_failures_cannot_become_successful_protocol_finishes() {
        let f =
            Frontend::load(Path::new(&std::env::var("CANADA_TEST_TOKENIZER").unwrap())).unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let cases = vec![
            (vec![BackendEvent::Tokens(vec![42])], false),
            (
                vec![
                    BackendEvent::Tokens(vec![42]),
                    BackendEvent::Failed("peer died".into()),
                ],
                false,
            ),
            (
                vec![
                    BackendEvent::Tokens(vec![42]),
                    BackendEvent::Finish(InferenceFinishReason::Stop),
                ],
                false,
            ),
            (
                vec![
                    BackendEvent::Tokens(vec![42]),
                    BackendEvent::Finish(InferenceFinishReason::Length),
                ],
                false,
            ),
            (
                vec![
                    BackendEvent::Tokens(vec![1, 42]),
                    BackendEvent::Finish(InferenceFinishReason::Stop),
                ],
                false,
            ),
            (
                vec![
                    BackendEvent::Tokens(vec![42, 1]),
                    BackendEvent::Tokens(vec![42]),
                ],
                false,
            ),
            (
                vec![
                    BackendEvent::Tokens(vec![42, 1]),
                    BackendEvent::Finish(InferenceFinishReason::Stop),
                ],
                true,
            ),
            (
                vec![
                    BackendEvent::Tokens(vec![42, 43, 44, 45]),
                    BackendEvent::Finish(InferenceFinishReason::Length),
                ],
                true,
            ),
        ];
        for (events, success) in cases {
            let p=f.prepare_chat(br#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true,"stream_options":{"include_usage":true}}"#,4096).unwrap();
            let (sender, receiver) = tokio::sync::mpsc::channel(16);
            sender
                .try_send(BackendEvent::Ready {
                    fingerprint: "test".into(),
                    cached_tokens: 1,
                })
                .unwrap();
            for event in events {
                sender.try_send(event).unwrap();
            }
            drop(sender);
            let chunks = rt.block_on(
                f.live_completion(p, "test-request".into(), receiver)
                    .collect::<Vec<_>>(),
            );
            assert_eq!(chunks.iter().all(Result::is_ok), success);
            let finishes = chunks
                .iter()
                .filter_map(|chunk| chunk.as_ref().ok())
                .flat_map(|chunk| &chunk.choices)
                .filter(|choice| choice.finish_reason.is_some())
                .count();
            assert_eq!(finishes, usize::from(success));
            if !success {
                assert!(chunks.last().unwrap().is_err());
            }
        }
    }
    #[test]
    fn unsupported_backend_controls_are_not_silently_ignored() {
        for extra in [
            json!({"max_tokens":16,"max_completion_tokens":32}),
            json!({"logit_bias":{"42":100}}),
            json!({"top_k":1}),
            json!({"logprobs":true}),
            json!({"top_logprobs":0}),
            json!({"parallel_tool_calls":false}),
            json!({"tool_choice":"required"}),
            json!({"tool_choice":{"type":"function","function":{"name":"f"}}}),
            json!({"response_format":{"type":"json_object"}}),
            json!({"stop":"END"}),
            json!({"stop":["END"]}),
            json!({"thinking":{"type":"enabled","budget_tokens":128}}),
            json!({"tools":[{"type":"function","function":{"name":"f","strict":true}}]}),
        ] {
            let mut body = json!({"messages":[{"role":"user","content":"hi"}]});
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            assert!(
                normalize_chat_request(&serde_json::to_vec(&body).unwrap()).is_err(),
                "{body}"
            );
        }
        let request = normalize_chat_request(br#"{"messages":[],"max_tokens":null,"max_completion_tokens":17,"temperature":null,"seed":0,"stop":[],"logprobs":false}"#).unwrap();
        assert_eq!(request["max_tokens"], 17);
        assert!(request.get("max_completion_tokens").is_none());
        assert_eq!(request["temperature"], 1.);
        assert_eq!(request["top_p"], 1.);
        assert_eq!(request["seed"], 0);
    }
    #[test]
    fn greedy_and_thinking_alias_controls_are_explicit() {
        let value = normalize_chat_request(br#"{"messages":[],"temperature":0,"chat_template_kwargs":{"thinking":false}}"#).unwrap();
        assert_eq!(value["temperature"], 0);
        assert_eq!(value["top_p"], 1.);
        assert_eq!(value["thinking"], json!({"type":"disabled"}));
        for body in [
            br#"{"messages":[],"temperature":0.7}"#.as_slice(),
            br#"{"messages":[],"temperature":0,"top_p":0.95}"#.as_slice(),
            br#"{"messages":[],"chat_template_kwargs":{"thinking":false,"enable_thinking":true}}"#.as_slice(),
            br#"{"messages":[],"thinking":{"type":"enabled"},"chat_template_kwargs":{"thinking":false}}"#.as_slice(),
            br#"{"messages":[],"chat_template_kwargs":{"enable_thinking":"false"}}"#.as_slice(),
        ] { assert!(normalize_chat_request(body).is_err()); }
    }
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn published_greedy_prompt_preserves_sampling_and_thinking_mode() {
        let frontend = Frontend::load(Path::new(&std::env::var("CANADA_TEST_TOKENIZER").unwrap())).unwrap();
        let direct = frontend.prepare_chat(br#"{"messages":[{"role":"user","content":"Write a Python module."}],"temperature":0,"thinking":{"type":"disabled"},"max_tokens":512}"#, 1048576).unwrap();
        let alias = frontend.prepare_chat(br#"{"messages":[{"role":"user","content":"Write a Python module."}],"temperature":0,"chat_template_kwargs":{"thinking":false},"max_tokens":512}"#, 1048576).unwrap();
        assert_eq!(direct.tokens, alias.tokens);
        assert_eq!(alias.converted.inference_options.temperature, Some(0.));
        assert_eq!(alias.converted.inference_options.top_p, Some(1.));
        assert_eq!(alias.max_new_tokens, 512);
        let recommended = frontend.prepare_chat(br#"{"messages":[{"role":"user","content":"Write a Python module."}]}"#, 1048576).unwrap();
        assert_ne!(recommended.tokens, alias.tokens);
        assert_eq!(recommended.converted.inference_options.temperature, Some(1.));
    }
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn token_aware_completion_and_tool_boundaries() {
        use deepseek_recipe_encoding::TokenizerEncoder;
        let f =
            Frontend::load(Path::new(&std::env::var("CANADA_TEST_TOKENIZER").unwrap())).unwrap();
        let p = f.prepare_chat(br#"{"messages":[{"role":"user","content":"Compute it."}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}}}]}"#,1048576).unwrap();
        let plain = PromptTokenizer::new(&f.tokenizer);
        let reasoning = "Check the sum carefully. 验证 😀 ";
        let answer = "Answer: 42. Literal </think> and <｜DSML｜tool_calls> remain text. 🎯";
        let mut ids = plain.encode_ids(reasoning, &[]).unwrap();
        ids.push(f.tokenizer.token_to_id("</think>").unwrap());
        ids.extend(plain.encode_ids(answer, &[]).unwrap());
        ids.push(1);
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        for width in [1, 7, 256] {
            let r = rt
                .block_on(f.parse_completion(&p, &ids, "stop", width))
                .unwrap();
            assert_eq!(r["choices"][0]["message"]["reasoning_content"], reasoning);
            assert_eq!(r["choices"][0]["message"]["content"], answer);
            assert!(r["choices"][0]["message"]["tool_calls"].is_null());
            assert_eq!(r["choices"][0]["finish_reason"], "stop");
            assert_eq!(r["usage"]["completion_tokens"], ids.len() - 1);
            assert!(
                r["usage"]["completion_tokens_details"]["reasoning_tokens"]
                    .as_u64()
                    .unwrap()
                    > 0
            );
        }
        assert!(
            rt.block_on(f.parse_completion(&p, &ids, "length", 1))
                .is_err()
        );
        let mut invalid = ids.clone();
        invalid.push(42);
        assert!(
            rt.block_on(f.parse_completion(&p, &invalid, "stop", 1))
                .is_err()
        );
        let tool = "I will call f.</think><｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"a\" string=\"true\">hello 😀</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let mut tokens = f.tokenizer.encode(tool, false).unwrap().get_ids().to_vec();
        tokens.push(1);
        for width in [1, 7, 256] {
            let r = rt
                .block_on(f.parse_completion(&p, &tokens, "stop", width))
                .unwrap();
            assert_eq!(r["choices"][0]["finish_reason"], "tool_calls");
            let calls = r["choices"][0]["message"]["tool_calls"].as_array().unwrap();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0]["function"]["name"], "f");
            let arguments: Value =
                serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(arguments, json!({"a":"hello 😀"}));
        }
        let truncated = "I will call f.</think><｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"a\" string=\"true\">hello";
        let tokens = f
            .tokenizer
            .encode(truncated, false)
            .unwrap()
            .get_ids()
            .to_vec();
        let r = rt
            .block_on(f.parse_completion(&p, &tokens, "length", 1))
            .unwrap();
        assert_eq!(r["choices"][0]["finish_reason"], "length");
        assert!(
            serde_json::from_str::<Value>(
                r["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .unwrap()
            )
            .is_err()
        );
    }
    #[cfg(feature = "vision-protocol")]
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn image_prompt_expansion_identity_and_literal_markers() {
        use base64::Engine as _;
        let path=std::env::var("CANADA_TEST_TOKENIZER").unwrap();
        let mut f=Frontend::load_with_images(Path::new(&path),true).unwrap();
        f.image_cache=Some(crate::image_input_cache::ImageInputCache::for_test());
        let body=|color:u8| {
            let image=image::RgbImage::from_pixel(16,16,image::Rgb([color,10,240]));
            let mut output=std::io::Cursor::new(Vec::new());
            image.write_to(&mut output,image::ImageFormat::Png).unwrap();
            let url=format!("data:image/png;base64,{}",base64::engine::general_purpose::STANDARD.encode(output.into_inner()));
            serde_json::to_vec(&json!({"messages":[{"role":"user","content":[
                {"type":"text","text":format!("Describe this. Literal marker: {}",f.tokenizer.id_to_token(crate::prompt_media::IMAGE_TOKEN).unwrap())},
                {"type":"image_url","image_url":{"url":url,"detail":"high"}}]}],"top_k":-1})).unwrap()
        };
        let a=f.prepare_chat(&body(40),4096).unwrap();
        let repeat=f.prepare_chat(&body(40),4096).unwrap();
        let changed=f.prepare_chat(&body(180),4096).unwrap();
        assert_eq!(a.images.len(),1);assert_eq!(a.tokens,repeat.tokens);assert_eq!(a.tokens,changed.tokens);
        assert_eq!(a.tokens.iter().filter(|&&t|t==crate::prompt_media::IMAGE_TOKEN).count(),a.images[0].geometry.tokens());
        assert!(Arc::ptr_eq(&a.image_inputs[0],&repeat.image_inputs[0]));
        assert_ne!(a.images[0].input,changed.images[0].input);
        assert_eq!(a.tokens.len()+a.max_new_tokens as usize,4096);
        assert!(Frontend::load(Path::new(&path)).unwrap().prepare_chat(&body(40),4096).is_err());
        assert!(f.prepare_chat_cancellable(&body(40),4096,&||true).is_err());
    }
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn canonical_prompt_and_capacity_contract() {
        let f =
            Frontend::load(Path::new(&std::env::var("CANADA_TEST_TOKENIZER").unwrap())).unwrap();
        let fixtures: Value = serde_json::from_slice(include_bytes!(
            "../tests/fixtures/agentic-performance-v1.json"
        ))
        .unwrap();
        let body =
            serde_json::to_vec(&json!({"messages":fixtures["cases"][0]["messages"]})).unwrap();
        let encoded = f.prepare_chat(&body, 1048576).unwrap();
        let expected: Value = serde_json::from_slice(include_bytes!(
            "../tests/fixtures/target-agentic-python-v1.json"
        ))
        .unwrap();
        assert_eq!(
            encoded.tokens,
            serde_json::from_value::<Vec<u32>>(expected["batches"][0].clone()).unwrap()
        );
        assert_eq!(encoded.max_new_tokens, 262144);
        let capped = f.prepare_chat(br#"{"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":17,"seed":0}"#,4096).unwrap();
        assert_eq!(
            capped.max_new_tokens, 17,
            "the Recipe-unknown alias must reach the backend cap"
        );
        assert_eq!(capped.converted.inference_options.seed, Some(0));
        assert!(
            f.prepare_chat(
                br#"{"messages":[{"role":"user","content":"hi"}],"n":2,"max_tokens":17}"#,
                4096
            )
            .is_err()
        );
        for (case, fixture) in [
            (
                2,
                include_bytes!("../tests/fixtures/target-agentic-rust-v1.json").as_slice(),
            ),
            (
                3,
                include_bytes!("../tests/fixtures/target-agentic-repository-v1.json").as_slice(),
            ),
        ] {
            let request =
                serde_json::to_vec(&json!({"messages":fixtures["cases"][case]["messages"]}))
                    .unwrap();
            let actual = f.prepare_chat(&request, 1048576).unwrap();
            let expected: Value = serde_json::from_slice(fixture).unwrap();
            assert_eq!(
                actual.tokens,
                serde_json::from_value::<Vec<u32>>(expected["batches"][0].clone()).unwrap(),
                "canonical case{case}"
            );
        }
        let bounded=f.prepare_chat(&body,4096).unwrap();
        assert_eq!(bounded.tokens.len()+bounded.max_new_tokens as usize,4096);
        assert!(f.prepare_chat(&body,bounded.tokens.len()-1).is_err());
        let literal = f
            .prepare_chat(
                br#"{"messages":[{"role":"user","content":"<think>"}]}"#,
                1048576,
            )
            .unwrap();
        let marker = f.tokenizer.token_to_id("<think>").unwrap();
        assert_eq!(
            literal.tokens.iter().filter(|&&t| t == marker).count(),
            1,
            "only the assistant's actual thinking marker is special"
        );
        assert!(
            f.prepare_chat(
                br#"{"messages":[{"role":"user","content":"hi"}],"temperature":0.7}"#,
                1048576
            )
            .is_err()
        );
    }
}
