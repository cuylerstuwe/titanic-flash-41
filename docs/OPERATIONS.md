# API and operations

The Rust server exposes authenticated `GET /health`, `GET /v1/models` and `POST /v1/chat/completions`. Use the bearer credential generated in `state/keys/http.key`; keep the separate priority-control credential private. Keys are generated locally and never belong in a URL, shell history, repository or published diagnostic.

## Recommended request

```json
{
  "model": "DeepSeek-V4.1-Flash-1017",
  "messages": [{"role": "user", "content": "Review this patch and explain any correctness issues."}],
  "temperature": 1,
  "top_p": 1,
  "top_k": -1,
  "reasoning_effort": "high",
  "max_tokens": 262144,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

Use a smaller output budget where appropriate, and respect the total 1,048,576-token context limit. Reasoning tokens count toward the output budget. `top_k: -1` means no top-k truncation. Supported temperatures are exactly 0 and 1; top-p must be 1. Unsupported sampling options fail explicitly rather than silently changing the target distribution.

The protocol supports ordinary OpenAI-style tool definitions, tool calls and tool-result messages, along with DeepSeek thinking controls. Images use supported `image_url` message content and the canonical image encoder/processor. The server validates URLs, image dimensions and resource limits. This release is not a complete replacement for every OpenAI API feature: arbitrary temperatures, nucleus cutoffs, penalties and constrained JSON/grammar decoding are not implemented. A requested schema must not be confused with a constrained-decoding guarantee.

Streaming uses SSE and ends with `[DONE]`. One delta may contain multiple speculative tokens. Clients should use reported usage for token counts, not count SSE chunks. Closing an active stream requests cancellation; bounded native work already in flight may finish before it is observed. Output backpressure pauses/resumes native generation, with a bounded stall timeout.

## Prompt processing and Pi timing

Streaming responses carry one top-level `timings` object as soon as the measured prefill phase is available. Nonstreaming responses include the same object. It does not require `stream_options.include_usage`.

```json
{
  "timings": {
    "prompt_n": 249,
    "cache_n": 9088,
    "prompt_ms": 50.0,
    "prompt_per_second": 4980.0,
    "scope": "server_prefill_phase"
  }
}
```

The numbers above are illustrative. `prompt_n` counts uncached prompt tokens; `cache_n` counts reused tokens. Their sum matches reported prompt usage. PP rate is `prompt_n * 1000 / prompt_ms`. This measures the server prefill phase, not network latency, queue time or an isolated GPU kernel timer. Client TTFT remains a different measurement. Cache capacity is bounded and opportunistic; repeat calls are not guaranteed a cache hit under memory pressure.

## Foreground and background

Requests default to foreground. Set `X-LocalAI-Mode: background` for background work, or explicitly use `foreground`. Background requests are peers with each other; foreground requests are peers with each other. Among admitted requests, foreground work pauses background work at safe boundaries; background work resumes with its accepted state intact. This transition can take a few seconds and is not a hard real-time preemption promise.

At most **two requests are admitted**, each retaining the full 1,048,576-token context capacity. Additional requests wait FIFO. An admitted request keeps its slot until its final response, terminal error, client cancellation/disconnection or terminal stall timeout. A temporary output stall, priority hold or pause does not release the slot. A queued third foreground request does not displace two admitted background requests. A completed tool-call response ends that request; submitting tool results creates a new request. Model state is retained in memory, without SSD parking.

The default has **one physical execution bank**, not two permanently allocated banks. Overlap uses complete, lossless state snapshots at committed boundaries. Solo forward reuse stays in the active bank and does not require a request snapshot. The shared snapshot/cache ceiling is 4 GiB per rank, allocated only as needed; within that shared ceiling, optional completed conversation checkpoints are capped at 2 GiB and yield to unfinished requests. The 8 GiB operating margin and 8.5 GiB allocation floor remain enforced on the limiting rank. Large same-priority peers may run serially if switching cannot retain both snapshots safely; their lifetime admissions and progress are preserved. A foreground newcomer that cannot safely pause an existing background request gets an explicit capacity error rather than silently canceling that owner or waiting indefinitely behind a hold. Overlap throughput and switching latency can be worse than the former permanently allocated two-bank layout.

`CANADA_SCRATCH_SINGLE_BANK=1` selects the qualified layout. Its paired execution, cross-bank prefix import and shared paired-drafter flags are disabled in the supplied profile. Primary append/decode graph variants are fully warmed and retained (`CANADA_SCRATCH_GRAPH_CACHE_ROWS=0`). A nonzero row budget is an optional bounded-retention control, with recapture latency on misses; it is not the release default. Do not independently re-enable conflicting paired flags.

The internal priority-control endpoint uses the separate control key. Ordinary client integrations should use the request header. Do not expose or log the control credential. Background qualification is separate from normal foreground performance measurement.

## Lifecycle and diagnostics

`deploy/fleet.py` starts exactly one four-rank cohort with a fresh private ticket, validates the selected nodes and runtime files, and monitors health. Containers use Docker's init process for signal forwarding and reaping. Any failed rank causes coordinated cleanup of the recorded cohort. The controller checks project/generation ownership before stopping all four ranks concurrently, allowing up to 90 seconds for each Docker stop to finish GPU teardown. Docker sends SIGTERM and may escalate to SIGKILL after its 15-second grace period; stopping the service is not a promise to finish in-flight requests or preserve RAM caches. Systemd can restart a failed controller, with a bounded start limit. An unreachable worker can prevent safe complete cleanup; repair SSH and retry instead of starting a conflicting cohort or weakening ownership checks.

Use `systemctl stop titanic-flash.service` for a systemd-managed installation. The standalone `fleet.py stop` action also handles an abandoned cohort when no supervisor owns the lifecycle lock. Inspect `fleet.py status`, Docker state and available memory before restarting after a failure. A process being alive is not equivalent to model Ready.

Worker event diagnostics default to bounded **RAM** history. The portable launcher uses Docker's `none` log driver to avoid persistent per-token worker logs. Lifecycle state writes occur at transitions; model requests and generated text are not written by this wrapper. OS, SSH and systemd journals can still write ordinary operational logs. Authenticated `/_native/diagnostics` returns bounded diagnostic state with `Cache-Control: no-store`; save only deliberate, redacted incident evidence. A failed early startup may require a deliberate temporary bounded-log diagnostic run.

Do not modify `deploy/profile.json` casually: it is the accumulated qualified model profile. Some internal environment variables retain a legacy `CANADA_SCRATCH_` prefix as an implementation compatibility detail. Diagnostic CLI commands and the `storage-fault-injection` feature are not part of the production build.

The inventory explicitly selects the temporary host policies: a 2350 MHz GPU cap or `0` to leave clocks unmanaged, and optional `/dev/nvme0` APST disablement per node. The clock helper assumes stock clocks as its restoration baseline; do not use it over an unrelated administrator's clock lock. The NVMe helper saves and restores the exact original feature value and table. Independent systemd timers restore settings if short leases stop renewing. Delayed checks from an older generation cannot overwrite a newer owner's policy. Healthy renewals remain in RAM; only activation, failure and restoration transitions write durable records.

A transient management failure appears as `host_policy_status: degraded`; it does not immediately discard a healthy model. Local timers can restore host settings while the controller retries, so performance during that degraded interval is not comparable to a fully active fixed-clock run. Check policy status alongside Ready when collecting benchmarks. Failure during initial policy arming prevents model launch and attempts restoration on every node.

The launcher leaves firmware, networking, other model services and external routers under the operator's control. Store rollback versions outside the active runtime. Keep model weights and Engram backing on local storage; moving cold build outputs or archived versions is independent of moving hot inference data.
