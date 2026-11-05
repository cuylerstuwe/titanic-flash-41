//! Two lifetime slots before expensive CPU conversion or GPU cache reclamation.
//! Excess requests retain only their bounded raw input. Model jobs inherit the
//! permit, so parking, priority holds and output stalls cannot lend a slot.
use super::*;
use std::future::{Future, poll_fn};

pub(super) struct Registered {
    shared: Arc<Shared>,
    pub(super) id: String,
    pub(super) cancellation: Arc<AtomicBool>,
    pub(super) enqueued: Instant,
    pub(super) transferred: bool,
}
impl Registered {
    fn new(shared: &Arc<Shared>) -> Self {
        let id = format!("chatcmpl-{}-{}",shared.nonce,shared.sequence.fetch_add(1,Ordering::Relaxed));
        let cancellation = Arc::new(AtomicBool::new(false));
        shared.requests.lock().unwrap().insert(id.clone(),Arc::downgrade(&cancellation));
        Self { shared:shared.clone(),id,cancellation,enqueued:Instant::now(),transferred:false }
    }
}
impl Drop for Registered {
    fn drop(&mut self) {
        if !self.transferred {
            self.cancellation.store(true,Ordering::Release);
            self.shared.requests.lock().unwrap().remove(&self.id);
        }
    }
}

struct WaitingInput(Arc<Shared>);
impl WaitingInput {
    fn new(shared: &Arc<Shared>) -> Self {
        shared.waiting_inputs.fetch_add(1,Ordering::AcqRel);Self(shared.clone())
    }
}
impl Drop for WaitingInput {
    fn drop(&mut self) { assert!(self.0.waiting_inputs.fetch_sub(1,Ordering::AcqRel)>0); }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn dispatch(shared:Arc<Shared>,body:context::Input,permit:OwnedSemaphorePermit,
    background_permit:Option<OwnedSemaphorePermit>,
    upload_workspace:Option<crate::memory::HostWorkspaceReservation>,background:bool,
    upload_execution:Option<OwnedSemaphorePermit>) -> Response {
    let registration=Registered::new(&shared);
    if !shared.execution_activity.load(Ordering::Acquire) || upload_execution.is_some() {
        return finish_chat(shared,body,permit,background_permit,upload_workspace,background,upload_execution,registration).await;
    }
    let mut acquire=Box::pin(shared.execution_admission.clone().acquire_owned());
    // Register in FIFO order now, not when the client happens to poll its
    // response body. Cancellation drops this exact semaphore waiter.
    match poll_fn(|cx|Poll::Ready(acquire.as_mut().poll(cx))).await {
        Poll::Ready(Ok(slot)) => return finish_chat(shared,body,permit,background_permit,
            upload_workspace,background,Some(slot),registration).await,
        Poll::Ready(Err(_)) => return error(StatusCode::SERVICE_UNAVAILABLE,"worker_not_ready","native worker is stopping"),
        Poll::Pending => {},
    }
    // Inspect only the transport flag. Serde skips message/image values rather
    // than allocating a second copy or invoking the tokenizer/image decoder.
    let streaming=match body.streaming() {
        Ok(streaming)=>streaming,
        Err(_)=>return error(StatusCode::BAD_REQUEST,"invalid_request","invalid JSON request or stream flag"),
    };
    let waiting=WaitingInput::new(&shared);
    let id=registration.id.clone();
    let response=async move {
        let acquired=loop {
            if registration.cancellation.load(Ordering::Acquire) {
                return error(StatusCode::CONFLICT,"request_cancelled","request cancelled before execution admission");
            }
            if shared.shutdown.load(Ordering::Acquire) || shared.health.load(Ordering::Acquire)!=READY {
                return error(StatusCode::SERVICE_UNAVAILABLE,"worker_not_ready","native worker is stopping");
            }
            tokio::select! {
                result=&mut acquire => break result,
                // Explicit cancellation uses the same atomic flag as model
                // jobs. Only queued requests need this bounded CPU wakeup.
                _=tokio::time::sleep(Duration::from_millis(100)) => {},
            }
        };
        drop(waiting);
        let Ok(slot)=acquired else {
            return error(StatusCode::SERVICE_UNAVAILABLE,"worker_not_ready","native worker is stopping");
        };
        if registration.cancellation.load(Ordering::Acquire) {
            return error(StatusCode::CONFLICT,"request_cancelled","request cancelled before execution admission");
        }
        finish_chat(shared,body,permit,background_permit,upload_workspace,background,Some(slot),registration).await
    };
    if !streaming {
        // Preserve ordinary HTTP status codes for direct nonstream clients.
        // The coordinated router uses streaming internally for its wait budget.
        return response.await;
    }
    waiting_response(response,&id)
}

fn waiting_response(response:impl Future<Output=Response>+Send+'static,id:&str)->Response {
    let stream=async_stream::stream! {
        let mut response=Box::pin(response);
        let mut keepalive=tokio::time::interval_at(tokio::time::Instant::now()+Duration::from_secs(5),Duration::from_secs(5));
        yield Ok::<Bytes,std::io::Error>(Bytes::from_static(b": native-execution waiting\n\n"));
        let response=loop {
            tokio::select! {
                response=&mut response => break response,
                _=keepalive.tick() => yield Ok(Bytes::from_static(b": keep-alive\n\n")),
            }
        };
        if !response.status().is_success() {
            // Headers were committed when queuing began. A late validation or
            // capacity error is explicit SSE, never a fabricated completion.
            let value=match to_bytes(response.into_body(),16<<10).await {
                Ok(body)=>serde_json::from_slice::<serde_json::Value>(&body).ok(),
                Err(_)=>None,
            }.unwrap_or_else(||json!({"error":{"type":"admission_error","message":"request preparation failed"}}));
            yield Ok(Bytes::from(format!("event: error\ndata: {}\n\n",value)));
            return;
        }
        let mut body=response.into_body().into_data_stream();
        while let Some(chunk)=body.next().await {
            yield chunk.map_err(|error|std::io::Error::other(error.to_string()));
        }
    };
    let mut response=Response::new(Body::from_stream(stream));
    for (name,value) in [("content-type","text/event-stream"),("cache-control","no-store"),
        ("x-native-execution-timing","v1")] {
        response.headers_mut().insert(name,HeaderValue::from_static(value));
    }
    response.headers_mut().insert("x-request-id",HeaderValue::from_str(id).unwrap());
    response
}
