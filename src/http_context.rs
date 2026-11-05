//! Opt-in second-hop context transport. Authentication precedes every lookup;
//! validation precedes queuing headers, materialization follows lifetime admission.
use super::*;
use crate::context_transport::{Cache, Prepared, Receipt};

#[derive(Clone)]
pub(super) struct Confirmation { receipt: Receipt, stored: Arc<AtomicBool> }
pub(super) struct Incremental {
    prepared: Prepared,
    images: Vec<crate::protocol::ImageReference>,
    confirmation: Confirmation,
}
pub(super) enum Input { Raw(Bytes), Incremental(Incremental) }
impl Input {
    pub(super) fn expanded_len(&self) -> usize {
        match self { Self::Raw(raw)=>raw.len(),Self::Incremental(p)=>p.prepared.expanded_bytes() }
    }
    pub(super) fn streaming(&self) -> Result<bool> {
        #[derive(Deserialize)]
        struct Mode { #[serde(default)] stream:Option<bool> }
        Ok(match self { Self::Raw(raw)=>serde_json::from_slice::<Mode>(raw)?.stream.unwrap_or(false),
            Self::Incremental(p)=>p.prepared.streaming() })
    }
    pub(super) fn receipt(&self) -> Option<Confirmation> {
        match self { Self::Raw(_)=>None,Self::Incremental(p)=>Some(p.confirmation.clone()) }
    }
    pub(super) fn with_bytes<G>(self, reservation:G, shared:&Shared,
        consume:impl FnOnce(&[u8],Option<&[crate::protocol::ImageReference]>)->Result<crate::protocol::PreparedChat>)
        ->Result<crate::protocol::PreparedChat> {
        match self {
            Self::Raw(raw)=>{let _reservation=reservation;consume(&raw,None)}
            Self::Incremental(p)=>p.prepared.with_materialized(|_|Ok(reservation),|body| {
                let result=consume(body.as_bytes(),Some(&p.images))?;
                let aliases=result.image_aliases.iter().map(|a|(a.source_hash,a.identity)).collect::<HashMap<_,_>>();
                let mut cache=shared.context_cache.as_ref().unwrap().lock().unwrap();
                let retained=cache.commit(&p.prepared,now_ms(),|raw|compact_message(raw,&aliases));
                p.confirmation.stored.store(retained,Ordering::Release);
                Ok(result)
            }),
        }
    }
}
fn now_ms()->u64 {SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64}
fn compact_message(raw:&str, aliases:&HashMap<[u8;32],[u8;32]>) -> Option<crate::context_transport::StoredMessage> {
    let mut message:serde_json::Value=serde_json::from_str(raw).ok()?;
    let mut images=Vec::new();
    if let Some(parts)=message.get_mut("content").and_then(serde_json::Value::as_array_mut) {
        for part in parts {
            if part["type"]!="image_url" {continue;}
            let Some(url)=part["image_url"]["url"].as_str().filter(|u|u.starts_with("data:")) else {continue;};
            let detail=part["image_url"].get("detail").map_or(Some(deepseek_recipe_core::multimodal::ImageDetail::default()),
                |v|serde_json::from_value(v.clone()).ok())?;
            let key=crate::protocol::image_source_hash(url,detail);
            let identity=*aliases.get(&key)?;
            part["image_url"]["url"]=json!(format!("http://titanic-context.invalid/{}",crate::context_transport::hex(&identity)));
            images.push(identity);
        }
    }
    // Keep ordinary text bytes exactly as supplied. Image-bearing messages
    // retain only canonical-equivalent JSON plus compact processed-image keys.
    Some(crate::context_transport::StoredMessage {json:if images.is_empty(){raw.to_owned()}else{message.to_string()},images})
}

/// Optional transport storage is discarded before asking for GPU-cache work.
/// Never block the GPU owner behind an HTTP parser holding this mutex.
pub(super) fn reclaim(shared: &Shared) -> bool {
    let Some(cache) = &shared.context_cache else {
        return false;
    };
    let Ok(mut cache) = cache.try_lock() else {
        return false;
    };
    let before = cache.stats().retained_bytes;
    cache.clear();
    let released = before.saturating_sub(cache.stats().retained_bytes);
    drop(cache);
    if released > 0 {
        crate::memory::trim_free_host_pages();
    }
    released > 0
}

pub(super) fn reserve_after_reclaim<T>(
    shared: &Shared,
    mut reserve: impl FnMut() -> Result<T>,
) -> Result<T> {
    match reserve() {
        Err(error)
            if error
                .downcast_ref::<crate::memory::InputCapacityError>()
                .is_some()
                && reclaim(shared) =>
        {
            reserve()
        }
        result => result,
    }
}

pub(super) fn capacity_error(incremental: bool, message: &str) -> Response {
    if incremental {
        // This path has not submitted inference or committed success headers.
        // The full endpoint retains its existing owner-aware reclaim behavior.
        error(StatusCode::CONFLICT, "context_full_required", message)
    } else {
        error(StatusCode::SERVICE_UNAVAILABLE, "input_capacity", message)
    }
}

pub(super) fn prepare(shared: &Shared, wire: &[u8]) -> std::result::Result<Incremental, Response> {
    // Retained bytes may outlive an execution owner. Preserve all unallocated
    // two-owner image/state capacity even when this request already owns a slot.
    // The normal full-request path remains available if optional caching cannot fit.
    let _workspace = if shared.body_limit == MAX_BODY {
        let bytes = wire.len().saturating_mul(16).saturating_add(16 << 20);
        match reserve_after_reclaim(shared, || {
            crate::memory::reserve_waiting_host_workspace(
                bytes,
                shared.waiting_image_budget.load(Ordering::Acquire),
            )
        }) {
            Ok(guard) => Some(guard),
            Err(_) => {
                return Err(capacity_error(
                    true,
                    "context validation cannot retain the memory reserve",
                ));
            }
        }
    } else {
        None
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    // Each native listener has one authenticated bearer principal. Hash its
    // existing digest into the codec scope; never trust a body-supplied identity.
    let principal = crate::context_transport::hex(&shared.authorization);
    let mut cache = shared
        .context_cache
        .as_ref()
        .expect("enabled endpoint")
        .lock()
        .unwrap();
    let prepared=Cache::prepare_unretained(&mut cache, &principal, wire, now, shared.body_limit).map_err(|reason| {
        error(
            StatusCode::from_u16(reason.status).unwrap(),
            reason.code,
            reason.code,
        )
    })?;
    drop(cache);
    let images=shared.frontend.pin_context_images(&prepared.image_keys()).map_err(|reason| {
        if reason.downcast_ref::<crate::protocol::ContextImageMiss>().is_some() {
            error(StatusCode::CONFLICT,"context_miss","processed image input was evicted; resend the full context")
        } else {error(StatusCode::INTERNAL_SERVER_ERROR,"context_failure","processed image lookup failed")}
    })?;
    let confirmation=Confirmation{receipt:prepared.receipt().clone(),stored:Arc::new(AtomicBool::new(false))};
    Ok(Incremental{prepared,images,confirmation})
}

pub(super) fn confirm(mut response: Response, confirmation: Option<&Confirmation>) -> Response {
    if let Some(confirmation) = confirmation.filter(|_| response.status().is_success()) {
        let receipt=&confirmation.receipt;
        let headers = response.headers_mut();
        headers.insert(
            "x-titanic-context-ack",
            HeaderValue::from_str(&receipt.id).unwrap(),
        );
        headers.insert(
            "x-titanic-context-key",
            HeaderValue::from_str(&receipt.key).unwrap(),
        );
        headers.insert(
            "x-titanic-context-stored",
            HeaderValue::from_static(if confirmation.stored.load(Ordering::Acquire) { "1" } else { "0" }),
        );
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_transport::{extend, hex, message_hash, root};
    fn envelope(messages: &[serde_json::Value], base: Option<(&str, usize)>) -> serde_json::Value {
        let raw = messages.iter().map(|m| m.to_string()).collect::<Vec<_>>();
        let key = raw.iter().fold(root(), |p, m| extend(&p, &message_hash(m)));
        json!({"version":1,"namespace":"0123456789abcdef0123456789abcdef",
            "id":"11111111111111111111111111111111","key":hex(&key),
            "base":base.map(|(key,keep)|json!({"key":key,"keep":keep})),
            "append":&raw[base.map_or(0, |(_,keep)|keep)..],
            "request":{"stream":true,"max_tokens":16,"seed":7,"temperature":1,"top_p":1}})
    }
    fn request(value: &serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/context/chat/completions")
            .header("authorization", "Bearer test-token")
            .header("content-type", "application/json")
            .header("content-length", value.to_string().len())
            .header("x-titanic-context", "v1")
            .body(Body::from(value.to_string()))
            .unwrap()
    }
    #[cfg(target_os="linux")]
    #[test]
    #[ignore = "isolated12GiB cgroup, CANADA_TEST_TOKENIZER and CANADA_TEST_CONTEXT_PRESSURE=1; no GPU"]
    fn real_cgroup_pressure_preserves_future_growth_and_full_fallback() {
        assert_eq!(std::env::var("CANADA_TEST_CONTEXT_PRESSURE").as_deref(),Ok("1"));
        let available=||crate::memory::Available::gb10(usize::MAX,usize::MAX).unwrap();
        let before=available();
        assert!(before.host_available > 16usize<<30);
        assert!(before.host_workspace_reserved==0 && (11usize<<30..=12usize<<30).contains(&before.cgroup_available.unwrap()));
        let (queue,shared,rt)=super::super::tests::test_http_fixture_limits(false,MAX_BODY);
        queue.enable_two_stream_admission();
        shared.waiting_image_budget.store(2usize<<30,Ordering::Release);
        let messages=vec![json!({"role":"user","content":"Preserve this synthetic conversation through capacity refusal."})];
        let full=envelope(&messages,None);
        let response=rt.block_on(context_chat(State(shared.clone()),request(&full)));
        assert_eq!(response.status(),StatusCode::OK);
        let first=queue.try_fifo().unwrap().unwrap();let tokens=first.tokens.clone();drop(first);drop(response);
        assert_eq!(shared.execution_admission.available_permits(),2);
        let delta=envelope(&messages,Some((full["key"].as_str().unwrap(),messages.len())));
        let target=9usize*1024*1024*1024+(256<<20);
        let held_bytes=available().admitted_available.checked_sub(target).unwrap();
        assert!(held_bytes<3usize<<30);
        let held=crate::memory::committed_zeroed_bytes(held_bytes).unwrap();
        let constrained=available();
        assert!((9usize<<30..=(9usize<<30)+(512<<20)).contains(&constrained.admitted_available));
        // The context path preserves unallocated owner growth before lookup,
        // even if it could claim a currently unused execution slot.
        let refused=rt.block_on(context_chat(State(shared.clone()),request(&delta)));
        assert_eq!(refused.status(),StatusCode::CONFLICT);
        assert!(!refused.headers().contains_key("x-titanic-context-ack"));
        let error:serde_json::Value=serde_json::from_slice(&rt.block_on(to_bytes(refused.into_body(),4096)).unwrap()).unwrap();
        assert_eq!(error["error"]["code"],"context_full_required");
        assert!(queue.try_fifo().unwrap().is_none());
        assert_eq!(shared.context_cache.as_ref().unwrap().lock().unwrap().stats().retained_bytes,0,"optional context released first");
        let mut body=full["request"].clone();body["messages"]=json!(messages);
        let ordinary=|| {let raw=body.to_string();Request::builder().method("POST").uri("/v1/chat/completions")
            .header("authorization","Bearer test-token").header("content-type","application/json")
            .header("content-length",raw.len()).body(Body::from(raw)).unwrap()};
        let a=rt.block_on(chat(State(shared.clone()),ordinary()));assert_eq!(a.status(),StatusCode::OK);
        let a_job=queue.try_fifo().unwrap().unwrap();assert_eq!(a_job.tokens,tokens);
        let b=rt.block_on(chat(State(shared.clone()),ordinary()));assert_eq!(b.status(),StatusCode::OK);
        let b_job=queue.try_fifo().unwrap().unwrap();assert_eq!(b_job.tokens,tokens);
        assert_eq!(shared.execution_admission.available_permits(),0);
        let waiting=rt.block_on(chat(State(shared.clone()),ordinary()));
        assert_eq!(waiting.status(),StatusCode::SERVICE_UNAVAILABLE,"waiting upload must preserve future owner growth");
        assert!(queue.try_fifo().unwrap().is_none());
        assert_eq!(shared.execution_admission.available_permits(),0,"refusal cannot release active owners");
        drop(a_job);drop(b_job);drop(a);drop(b);drop(held);crate::memory::trim_free_host_pages();
        let miss=rt.block_on(context_chat(State(shared.clone()),request(&delta)));
        assert_eq!(miss.status(),StatusCode::CONFLICT);
        let error:serde_json::Value=serde_json::from_slice(&rt.block_on(to_bytes(miss.into_body(),4096)).unwrap()).unwrap();
        assert_eq!(error["error"]["code"],"context_miss");
        let recovery=rt.block_on(context_chat(State(shared.clone()),request(&full)));
        assert_eq!(recovery.status(),StatusCode::OK);assert_eq!(recovery.headers()["x-titanic-context-stored"],"1");
        let job=queue.try_fifo().unwrap().unwrap();assert_eq!(job.tokens,tokens);drop(job);drop(recovery);
        assert_eq!(shared.execution_admission.available_permits(),2);
        assert_eq!(shared.admission.available_permits(),QUEUE_DEPTH+1);
        assert_eq!(available().host_workspace_reserved,0);
        println!("CONTEXT_PRESSURE {}",json!({"before":before,"held_bytes":held_bytes,"constrained":constrained,
            "after":available(),"context_refused_before_dispatch":true,"full_fallback_tokens_exact":true,
            "future_owner_growth_preserved":true,"cache_miss_and_full_resync":true}));
    }
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn compact_images_share_inputs_survive_queueing_and_resync_before_output() {
        use base64::Engine;
        let (queue,shared,rt)=super::super::tests::test_http_fixture_images(true);
        queue.enable_two_stream_admission();
        let mut rng=1729u32;
        let pixels=image::RgbImage::from_fn(256,256,|_,_| {
            let mut p=[0;3];for x in &mut p {rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;*x=rng as u8;}
            image::Rgb(p)
        });
        let mut encoded=std::io::Cursor::new(Vec::new());pixels.write_to(&mut encoded,image::ImageFormat::Png).unwrap();
        let url=format!("data:image/png;base64,{}",base64::engine::general_purpose::STANDARD.encode(encoded.into_inner()));
        let messages=vec![json!({"role":"user","content":[{"type":"text","text":"Inspect this synthetic screenshot."},
            {"type":"image_url","image_url":{"url":url,"detail":"high"}}]})];
        let full=envelope(&messages,None);
        let response=rt.block_on(context_chat(State(shared.clone()),request(&full)));
        assert_eq!(response.status(),StatusCode::OK);
        assert_eq!(response.headers()["x-titanic-context-stored"],"1");
        let first=queue.try_fifo().unwrap().unwrap();
        let tokens=first.tokens.clone();let images=first.images.clone();let pixels=first.image_inputs[0].patches.clone();
        let retained=shared.context_cache.as_ref().unwrap().lock().unwrap().stats().retained_bytes;
        assert!(full.to_string().len()>250000 && retained<4096,"encoded history must not be retained: {retained}");
        let delta=envelope(&messages,Some((full["key"].as_str().unwrap(),messages.len())));
        let second_response=rt.block_on(context_chat(State(shared.clone()),request(&delta)));
        assert_eq!(second_response.status(),StatusCode::OK);
        let second=queue.try_fifo().unwrap().unwrap();
        assert_eq!(second.tokens,tokens);assert_eq!(second.images,images);
        assert!(Arc::ptr_eq(&first.image_inputs[0],&second.image_inputs[0]));
        let waiting=rt.block_on(context_chat(State(shared.clone()),request(&delta)));
        assert_eq!(waiting.headers()["x-titanic-context-stored"],"0","do not promise a post-conversion commit while queued");
        assert_eq!(shared.waiting_inputs.load(Ordering::Acquire),1);
        shared.frontend.forget_context_images_for_test();
        drop(first);drop(response);
        rt.block_on(async {
            let worker=async {
                let deadline=Instant::now()+Duration::from_secs(3);
                loop {
                    if let Some(job)=queue.try_fifo().unwrap() {
                        assert_eq!(job.tokens,tokens);assert_eq!(job.images,images);
                        assert!(Arc::ptr_eq(&job.image_inputs[0],&second.image_inputs[0]));
                        job.emit(BackendEvent::Ready{fingerprint:"fixture".into(),cached_tokens:0});
                        job.emit(BackendEvent::Tokens(vec![42,1]));
                        job.emit(BackendEvent::Finish(deepseek_recipe::stream::InferenceFinishReason::Stop));break;
                    }
                    assert!(Instant::now()<deadline);tokio::time::sleep(Duration::from_millis(1)).await;
                }
            };
            let (body,())=tokio::join!(to_bytes(waiting.into_body(),1<<20),worker);
            assert!(std::str::from_utf8(&body.unwrap()).unwrap().contains("[DONE]"));
        });
        drop(second);drop(second_response);
        let miss=rt.block_on(context_chat(State(shared.clone()),request(&delta)));
        assert_eq!(miss.status(),StatusCode::CONFLICT);
        assert!(!miss.headers().contains_key("x-titanic-context-ack"));
        assert!(queue.try_fifo().unwrap().is_none(),"no model dispatch on missing processed input");
        let recovered=rt.block_on(context_chat(State(shared.clone()),request(&full)));
        assert_eq!(recovered.status(),StatusCode::OK);
        let job=queue.try_fifo().unwrap().unwrap();assert_eq!(job.tokens,tokens);assert_eq!(job.images,images);
        assert_eq!(job.image_inputs[0].patches,pixels);
        drop(job);drop(recovered);
        assert_eq!(shared.execution_admission.available_permits(),2);
        assert_eq!(shared.admission.available_permits(),QUEUE_DEPTH+1);
        // A client cannot invent an internal image reference in a new message.
        let forged=vec![json!({"role":"user","content":[{"type":"image_url","image_url":{
            "url":format!("http://titanic-context.invalid/{}",hex(&images[0].input)),"detail":"high"}}]})];
        assert_eq!(rt.block_on(context_chat(State(shared.clone()),request(&envelope(&forged,None)))).status(),StatusCode::BAD_REQUEST);
        println!("compact-image lifecycle: full_wire_bytes={} retained_bytes={} exact_tokens={} images={}",
            full.to_string().len(),retained,tokens.len(),images.len());
    }
    #[test]
    #[ignore = "requires CANADA_TEST_TOKENIZER pointing to the pinned public1017 tokenizer"]
    fn native_context_confirms_exact_input_and_pins_queued_branches() {
        let (queue, shared, rt) = super::super::tests::test_http_fixture();
        queue.enable_two_stream_admission();
        let messages = vec![json!({"role":"user","content":"Inspect this small Rust function."})];
        let full = envelope(&messages, None);
        let response = rt.block_on(context_chat(State(shared.clone()), request(&full)));
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()["x-titanic-context-key"],
            full["key"].as_str().unwrap()
        );
        assert_eq!(
            response.headers()["x-titanic-context-ack"],
            full["id"].as_str().unwrap()
        );
        let first = queue.try_fifo().unwrap().unwrap();
        let mut ordinary = full["request"].clone();
        ordinary["messages"] = json!(messages);
        let expected = shared
            .frontend
            .prepare_chat(&serde_json::to_vec(&ordinary).unwrap(), shared.capacity)
            .unwrap();
        assert_eq!(first.tokens, expected.tokens);
        assert_eq!(first.seed, 7);

        let delta = envelope(
            &messages,
            Some((full["key"].as_str().unwrap(), messages.len())),
        );
        let second_response = rt.block_on(context_chat(State(shared.clone()), request(&delta)));
        let second = queue.try_fifo().unwrap().unwrap();
        let waiting = rt.block_on(context_chat(State(shared.clone()), request(&delta)));
        assert_eq!(waiting.status(), StatusCode::OK);
        assert_eq!(
            waiting.headers()["x-titanic-context-key"],
            full["key"].as_str().unwrap()
        );
        assert_eq!(shared.waiting_inputs.load(Ordering::Acquire), 1);
        assert!(queue.try_fifo().unwrap().is_none());
        shared
            .context_cache
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .clear();
        assert!(
            shared
                .context_cache
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .stats()
                .retained_bytes
                > 0
        );
        let miss = rt.block_on(context_chat(State(shared.clone()), request(&delta)));
        assert_eq!(
            miss.status(),
            StatusCode::CONFLICT,
            "miss must precede queued success headers"
        );
        assert!(!miss.headers().contains_key("x-titanic-context-ack"));
        drop(first);
        drop(response);
        rt.block_on(async {
            let worker = async {
                let deadline = Instant::now() + Duration::from_secs(3);
                loop {
                    if let Some(job) = queue.try_fifo().unwrap() {
                        assert_eq!(job.tokens, expected.tokens);
                        job.emit(BackendEvent::Ready {
                            fingerprint: "fixture".into(),
                            cached_tokens: 0,
                        });
                        job.emit(BackendEvent::Tokens(vec![42, 1]));
                        job.emit(BackendEvent::Finish(
                            deepseek_recipe::stream::InferenceFinishReason::Stop,
                        ));
                        break;
                    }
                    assert!(Instant::now() < deadline);
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            };
            let (body, ()) = tokio::join!(to_bytes(waiting.into_body(), 1 << 20), worker);
            assert!(
                std::str::from_utf8(&body.unwrap())
                    .unwrap()
                    .contains("[DONE]")
            );
        });
        drop(second);
        drop(second_response);
        // Successful queued conversion may retain a new compact checkpoint.
        // Clearing that optional owner must release all remaining input charges.
        shared.context_cache.as_ref().unwrap().lock().unwrap().clear();
        assert_eq!(
            shared
                .context_cache
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .stats()
                .retained_bytes,
            0
        );
        assert_eq!(shared.execution_admission.available_permits(), 2);
        assert_eq!(shared.admission.available_permits(), QUEUE_DEPTH + 1);
        assert!(shared.requests.lock().unwrap().is_empty());

        let mut invalid = full.clone();
        invalid["key"] = json!("0".repeat(64));
        assert_eq!(
            rt.block_on(context_chat(State(shared.clone()), request(&invalid)))
                .status(),
            StatusCode::BAD_REQUEST
        );
        let mut unauth = request(&full);
        unauth.headers_mut().remove("authorization");
        assert_eq!(
            rt.block_on(context_chat(State(shared.clone()), unauth))
                .status(),
            StatusCode::UNAUTHORIZED
        );
        let mut missing_header = request(&full);
        missing_header.headers_mut().remove("x-titanic-context");
        assert_eq!(
            rt.block_on(context_chat(State(shared.clone()), missing_header))
                .status(),
            StatusCode::BAD_REQUEST
        );
        assert!(queue.try_fifo().unwrap().is_none());
    }
}
