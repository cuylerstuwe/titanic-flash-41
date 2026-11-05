# Optional incremental context transport

The ordinary `/v1/chat/completions` API remains stateless on the wire. Clients
that implement this extension can send the new messages plus an authenticated
reference to earlier input. The default fleet configuration permits a bounded
64 MiB RAM cache for this purpose. This cache is separate from the model-state
cache, is opportunistic, and does not persist conversations to disk. Ordinary
requests do not enter it.

Send `POST /v1/context/chat/completions` with the usual bearer authentication,
`Content-Type: application/json`, and `X-Titanic-Context: v1`. The envelope is:

```json
{
  "version": 1,
  "namespace": "32 lowercase hexadecimal characters",
  "id": "32 lowercase hexadecimal characters",
  "base": null,
  "append": ["{\"role\":\"user\",\"content\":\"Name a fruit.\"}"],
  "key": "64 lowercase hexadecimal characters",
  "request": {
    "model": "DeepSeek-V4.1-Flash-1017",
    "temperature": 1,
    "top_p": 1,
    "top_k": -1,
    "reasoning_effort": "high",
    "max_tokens": 128,
    "stream": true
  }
}
```

Generate a fresh random namespace for a conversation and a fresh `id` for each
request. `id` confirms the applied input; it is **not** an idempotency key.
`request` contains the complete ordinary request controls, including any tools,
but must not contain `messages`. Each `append` item is the exact JSON string for
one complete message. Preserve those UTF-8 bytes between requests.

Compute the key using SHA-256 and binary digests:

```python
key = sha256(b"titanic-context-v1/root").digest()
for message in all_retained_and_appended_message_strings:
    message_key = sha256(b"titanic-context-v1/message\0" + message.encode("utf-8")).digest()
    key = sha256(b"titanic-context-v1/chain\0" + key + message_key).digest()
wire_key = key.hex()
```

To extend an input, set `base` to `{"key": previous_key, "keep": message_count}`
and send only the new messages in `append`. To rewind or branch, use a smaller
`keep` and compute the new key over that retained prefix plus the new messages.
The server stores input messages only: the client must append the assistant's
actual response and subsequent tool results itself.

Before accepting output, require response headers `X-Titanic-Context-Ack` and
`X-Titanic-Context-Key` to match this request's `id` and `key`. Also process normal
HTTP/SSE errors and terminal completion. `X-Titanic-Context-Stored: 1` confirms
retention at header time. A queued streaming request can report `0`; it makes no
promise that the input will be available for the next request.

A missing, evicted or expired reference returns HTTP 409 `context_miss` before
inference. Capacity pressure can instead return `context_full_required`. Resend
full messages through the ordinary endpoint, or retry a full envelope with
`base: null`. Do not silently omit history. Do not blindly retry after accepting
generated output. References are scoped to the bearer principal and namespace;
restarts and memory pressure can invalidate them.

Image-bearing inputs retain compact references to processed images instead of
repeated base64 bodies. If the corresponding image is evicted, the same explicit
resynchronization is required. Clients and forwarding proxies must preserve the
extension endpoint and confirmation headers; an unmodified OpenAI client does
not acquire incremental transport automatically.

The small [protocol smoke](../tools/smoke_context.py) exercises full input,
extension, rewind and a refused missing reference. Run it against an idle test
instance with `--base-url` and `--key-file`; it uses recommended sampling settings.
