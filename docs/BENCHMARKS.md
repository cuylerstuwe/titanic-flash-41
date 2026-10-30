# Reproducing the published single-stream comparison

Run on the **head node**, against its direct loopback endpoint, with no other requests. Use a freshly started model process if measuring the first uncached long prompt. Do not route through a remote client or proxy.

```sh
cd /srv/titanic-flash-41/source
python3 benchmarks/run_published.py \
  --base-url http://127.0.0.1:8000 \
  --key-file /srv/titanic-flash-41/state/keys/http.key \
  --output /srv/titanic-flash-41/benchmark-$(date -u +%Y%m%dT%H%M%SZ)
```

The wrapper downloads the external benchmark client at an immutable gist revision and verifies SHA-256 `76a078842a0dd50b8345a6feb4455b890517eac51513f030701cd4a65d0a23ee`. It reuses that client's request generation and stream timing code, and runs only the single-stream code, prose and prose2 tests. The client is not redistributed here. Inspect the downloaded code before running it in a sensitive environment.

For each context, it runs three 64-token warmups followed by three 512-token trials per workload. There are 24 requests total. Settings are temperature 0, top-p 1, thinking disabled. The public Python-standard-library context in `benchmarks/contexts/slot0.txt` has SHA-256 `6b4b21a0ed3ae0eefe2b0aab0b9555eb1332334347c8bda80a83a2eaee168457`; token count varies slightly with the appended prompt. The measured prose prompt contained 131,064 tokens. No multi-stream or sampled-chat results are substituted.

The original decode formula is `(completion_tokens - 1) / (last_content_delta_time - first_content_delta_time)`. Time to first token is measured from sending the request to its first nonempty content or reasoning delta. End-to-end throughput uses completion tokens divided by time from sending to the last content delta. Server token usage counts are used, not SSE event counts. A speculative step can emit several tokens in one delta.

The README's first-long-prompt TTFT is the initial 131K prose warmup, not the median of later cached requests. The ordinary 131K trial TTFT medians were 0.248 s for code, 0.255 s for prose and 0.224 s for prose2; they benefit from prefix reuse. The 21.05 s uncached TTFT is not a pure GPU-prefill measurement.

The benchmark was run on the qualified engine build whose compiled source fingerprint is recorded in `results-2026-10-30.json`. The public package retains its inference implementation and GPU profile, with portable network configuration and project naming. The separate portable lifecycle wrapper is not included in the throughput claim. No SSD power-state override was applied during the finite benchmark.

For normal use, restore temperature 1, top-p 1, top-k -1 and high reasoning effort. Throughput on these short greedy prompts does not replace measurements of completed tasks, reasoning length, tool correctness, or response quality under those settings.
