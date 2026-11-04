# Reproducing the published single-stream comparison

The current source release is s1291 (November 4). The published comparison below
remains the historical s1173 pass; no s1291 benchmark is claimed here.

Run on the **head node**, against its direct loopback endpoint, with no other requests. Use a freshly started model process if measuring the first uncached long prompt. Do not route through a remote client or proxy.

```sh
cd /srv/titanic-flash-41/source
python3 benchmarks/run_published.py \
  --base-url http://127.0.0.1:8000 \
  --key-file /srv/titanic-flash-41/state/keys/http.key \
  --output /srv/titanic-flash-41/benchmark-$(date -u +%Y%m%dT%H%M%SZ)
```

The base URL can also use a literal local interface address when the server binds that address instead of loopback. The wrapper requires Linux to report `type=local` and `dev=lo` for that route, and rejects remote addresses. This validation happens before request timing; the external request and timing function are unchanged.

The wrapper downloads the external benchmark client at an immutable gist revision and verifies SHA-256 `76a078842a0dd50b8345a6feb4455b890517eac51513f030701cd4a65d0a23ee`. It reuses that client's request generation and stream timing code, and runs only the single-stream code, prose and prose2 tests. The client is not redistributed here. Inspect the downloaded code before running it in a sensitive environment.

For each context, it runs three 64-token warmups followed by three 512-token trials per workload. There are 24 requests total. Settings are temperature 0, top-p 1, thinking disabled. The public Python-standard-library context in `benchmarks/contexts/slot0.txt` has SHA-256 `6b4b21a0ed3ae0eefe2b0aab0b9555eb1332334347c8bda80a83a2eaee168457`; token count varies slightly with the appended prompt. The measured prose prompt contained 131,064 tokens. No multi-stream or sampled-chat results are substituted.

The original decode formula is `(completion_tokens - 1) / (last_content_delta_time - first_content_delta_time)`. Time to first token is measured from sending the request to its first nonempty content or reasoning delta. End-to-end throughput uses completion tokens divided by time from sending to the last content delta. Server token usage counts are used, not SSE event counts. A speculative step can emit several tokens in one delta.

The README's first-long-prompt TTFT is the initial 131K prose warmup, not the median of later cached requests. In the November 1 post-reboot pass, the ordinary 131K trial TTFT medians were 0.233 s for code, 0.332 s for prose and 0.303 s for prose2; they benefit from prefix reuse. The 21.19 s uncached TTFT is not a pure GPU-prefill measurement.

## Complete post-reboot pass

The current table was measured on November 1, 20:16–20:18 Pacific, using the installed native s1173 build on the first model load after all four GB10s rebooted. All 24 requests completed in one pass without a reload. Built-in startup warmup and two short operational requests preceded it, with no reusable long benchmark prompt. All four GPUs retained the 2350 MHz maximum-clock policy; the Spark's NVMe APST was temporarily disabled. The measured build hashes are recorded in [results-2026-11-01-clean-boot.json](../benchmarks/results-2026-11-01-clean-boot.json). This is a native-build observation; the portable build has a separate earlier measurement and qualification record.

An initial client call to an unbound loopback port was refused before generating tokens. The wrapper was then adjusted to accept the server's verified local interface address (`type=local`, `dev=lo`); no proxy or physical network hop was added. That setup failure yielded no score and did not reload the model. The complete pass was not repeated to select better scores.

| Context | Workload | Earlier s1173 portable | Post-reboot s1173 | Change | Historical s986 | Change vs s986 |
|---|---|---:|---:|---:|---:|---:|
| Clean | Code | 113.2 | 117.3 | +3.6% | 115.5 | +1.5% |
| Clean | Prose | 61.9 | 63.4 | +2.3% | 59.9 | +5.7% |
| Clean | Prose 2 | 57.7 | 60.4 | +4.6% | 54.8 | +10.2% |
| ~131K | Code | 101.9 | 108.7 | +6.7% | 113.9 | -4.6% |
| ~131K | Prose | 57.9 | 60.6 | +4.6% | 60.1 | +0.8% |
| ~131K | Prose 2 | 54.1 | 56.8 | +4.9% | 58.8 | -3.5% |

All throughput cells are medians in tokens/second, with three measured trials each. The README uses the entire new pass, not a per-row selection of the best trials or runs. First-long-prompt TTFT was essentially unchanged: 21.19 s versus 21.22 s. Raw samples, ranges and standard deviations remain available for both s1173 passes in separate files. The [earlier s1173 results](../benchmarks/results-2026-11-01.json) and [October 30 s986 results](../benchmarks/results-2026-10-30.json) have not been overwritten.

The earlier s1173 pass used the portable build after two failed portable GPU startups in the same boot. The native and portable artifacts have different build hashes, and generated continuations and allocation history were not held fixed. These observations therefore do not isolate how much of the increase came from reboot or memory bandwidth. Against historical s986, long-context code and prose 2 still remain lower, while long-context prose is slightly higher. Target attention precision, drafting policy, host policy and continuations differ from s986; this is not a matched numerical-policy regression test or a recommended-sampler quality conclusion.

Record boot history and the order of engine launches when comparing separate processes. On GB10, newly allocated buffers can have different bandwidth after repeated engine launches; a process restart alone does not normalize physical memory placement. Prefer alternating candidate and control within one model load, keeping weight allocations fixed, for small optimization experiments. For comparisons requiring separate loads, use reversed-order controls and report any unresolved memory-state confound. No allocation-specific bandwidth measurement was performed in this pass. The published cross-system figures do not include a matched memory-placement control.

For normal use, restore temperature 1, top-p 1, top-k -1 and high reasoning effort. Throughput on these short greedy prompts does not replace measurements of completed tasks, reasoning length, tool correctness, or response quality under those settings.
