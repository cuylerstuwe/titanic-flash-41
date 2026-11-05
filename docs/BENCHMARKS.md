# Reproducing the published single-stream comparison

The current source release and README benchmark are s1329 (November 4).
Historical native s1291, s1173 and s986 measurements remain separately labeled below.

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

The README's first-long-prompt TTFT is the initial 131K prose warmup, not the median of later cached requests. The later trials benefit from prefix reuse. Uncached TTFT includes work outside GPU prefill; report it separately from server prefill timing.

## Current s1329 portable pass — November 4

All 24 requests completed on the first model load after a four-device reboot, against the head's loopback endpoint. The 2350 MHz cap and Spark APST policy remained active. Ordinary serving, cache/timing, synthetic-image and incremental-context checks preceded the benchmark; none populated its long prompt. Existing validated weight atlases were used. Bounded diagnostic container logs were enabled for this qualification run; normal serving defaults to logging disabled.

| Context | Workload | Earlier native s1291 | Portable s1329 | Change |
|---|---|---:|---:|---:|
| Clean | Code | 118.1 | 121.0 | +2.4% |
| Clean | Prose | 63.7 | 65.5 | +2.8% |
| Clean | Prose 2 | 60.4 | 62.4 | +3.2% |
| ~131K | Code | 109.2 | 111.9 | +2.4% |
| ~131K | Prose | 61.0 | 62.8 | +2.9% |
| ~131K | Prose 2 | 57.1 | 58.6 | +2.6% |

All throughput values are median tokens/second from three 512-token trials. The first uncached ~131K TTFT was **20.99 s**, versus **21.27 s** in the earlier native s1291 pass. [Full results](../benchmarks/results-2026-11-04-s1329.json) include ranges, settings and measured-build hashes; [all 24 samples](../benchmarks/samples-2026-11-04-s1329.json) retain token counts, output hashes and timings without generated text. These historical differences mix build, continuation and boot differences; they are not an isolated estimate of optimization effects.

Against the older s986 observations, long-context code remains 1.8% lower and prose 2 0.4% lower. Target attention precision, drafting policy and continuations differ from that older release, so these residual historical gaps are not a controlled regression measurement. The original s986 results remain below.

Two earlier portable starts failed before serving because the package omitted a MoE kernel enabled by default in Rust. Two diagnostic logging-configuration attempts also failed before GPU startup. The missing unchanged dependency and a preflight check were added. All attempts were cleaned up, followed by a fresh four-device reboot before this complete pass. No benchmark request ran in the failed attempts, and no score-based retry or per-row best-of selection occurred.

After this completed pass, the wrapper timed out stopping rank 0; fallback cleanup recovered all four devices without kernel or guard errors. Container init, concurrent rank stop and a longer GPU-teardown timeout corrected the lifecycle path, which passed a separate fresh-boot startup/request/shutdown check with normal logging disabled. The inference executable, kernels and profile are unchanged; the benchmark was not repeated for this correction.

The separate native comparison at **temperature 1 / top-p 1 / high reasoning** is more directly relevant to recommended use: eight matching continuations were **3.59%** and **3.38%** faster in opposite fresh-boot launch orders, while ~131K prefill stayed within 0.12%. [Paired records](../benchmarks/release-comparison-s1329.json) preserve changed trajectories, including a copy outcome that took longer after emitting more tokens. Neither this small panel nor the greedy table establishes general task-time or quality superiority.

## Historical native s1291 pass — November 4

Measured 07:11–07:14 Pacific on the installed native s1291 release, using the first model load after all four GB10s rebooted. All 24 requests completed with no model restart, worker timeout, or concurrent inference traffic. The 2350 MHz maximum-clock policy and Spark APST policy stayed active. Startup warmup and two short operational requests preceded the benchmark.

The first client attempt stopped after four completed short-context requests when its non-lingering user service manager exited after SSH logout. A subsequent short request was cancelled. The complete replacement pass ran in a bounded system service under the same unprivileged account; the inference service did not restart. This was a client-launch problem, not a score-based retry. No long-context request had started, so the first ~131K prompt remained uncached. [Interrupted sample metadata](../benchmarks/samples-2026-11-04-interrupted.json) is retained separately.

| Context | Workload | Prior s1173 | Current s1291 | Change | Historical s986 | Change vs s986 |
|---|---|---:|---:|---:|---:|---:|
| Clean | Code | 117.3 | 118.1 | +0.7% | 115.5 | +2.3% |
| Clean | Prose | 63.4 | 63.7 | +0.6% | 59.9 | +6.4% |
| Clean | Prose 2 | 60.4 | 60.4 | +0.1% | 54.8 | +10.3% |
| ~131K | Code | 108.7 | 109.2 | +0.5% | 113.9 | -4.1% |
| ~131K | Prose | 60.6 | 61.0 | +0.8% | 60.1 | +1.6% |
| ~131K | Prose 2 | 56.8 | 57.1 | +0.6% | 58.8 | -2.9% |

All throughput values are medians in tokens/second, with three 512-token trials each. First-long-prompt TTFT was **21.27 s**, versus **21.19 s** in the prior s1173 pass. The ordinary cached ~131K TTFT medians were 0.233 s for code, 0.332 s for prose and 0.303 s for prose 2. Uncached TTFT includes all server work before first content and is not a pure GPU-prefill measurement.

The generation differences from s1173 are small (0.1–0.8%); this pass does not establish a general speed improvement. Long-context code and prose 2 remain 4.1% and 2.9% below the historical s986 observations. Those historical gaps remain unresolved; this run does not isolate their cause.

[Full results](../benchmarks/results-2026-11-04.json) record hashes, all ranges, settings and run accounting. [All 24 completed samples](../benchmarks/samples-2026-11-04.json) include token counts, timing and output hashes without generated text. The earlier s1291 table used this one complete pass; lower results are retained. Historical runs were not repeated, and differences in implementation, continuations and host state prevent treating these changes as isolated optimization effects. This benchmark exercised the installed native engine; it does not requalify the portable Arm64 build or deployment wrapper.

## Historical s1173 post-reboot pass — November 1

The earlier table was measured on November 1, 20:16–20:18 Pacific, using the installed native s1173 build on the first model load after all four GB10s rebooted. All 24 requests completed in one pass without a reload. Built-in startup warmup and two short operational requests preceded it, with no reusable long benchmark prompt. All four GPUs retained the 2350 MHz maximum-clock policy; the Spark's NVMe APST was temporarily disabled. The measured build hashes are recorded in [results-2026-11-01-clean-boot.json](../benchmarks/results-2026-11-01-clean-boot.json). This is a native-build observation; the portable build has a separate earlier measurement and qualification record.

An initial client call to an unbound loopback port was refused before generating tokens. The wrapper was then adjusted to accept the server's verified local interface address (`type=local`, `dev=lo`); no proxy or physical network hop was added. That setup failure yielded no score and did not reload the model. The complete pass was not repeated to select better scores.

| Context | Workload | Earlier s1173 portable | Post-reboot s1173 | Change | Historical s986 | Change vs s986 |
|---|---|---:|---:|---:|---:|---:|
| Clean | Code | 113.2 | 117.3 | +3.6% | 115.5 | +1.5% |
| Clean | Prose | 61.9 | 63.4 | +2.3% | 59.9 | +5.7% |
| Clean | Prose 2 | 57.7 | 60.4 | +4.6% | 54.8 | +10.2% |
| ~131K | Code | 101.9 | 108.7 | +6.7% | 113.9 | -4.6% |
| ~131K | Prose | 57.9 | 60.6 | +4.6% | 60.1 | +0.8% |
| ~131K | Prose 2 | 54.1 | 56.8 | +4.9% | 58.8 | -3.5% |

All throughput cells are medians in tokens/second, with three measured trials each. That update used the entire completed pass, not a per-row selection of the best trials or runs. First-long-prompt TTFT was essentially unchanged: 21.19 s versus 21.22 s. Raw samples, ranges and standard deviations remain available for both s1173 passes in separate files. The [earlier s1173 results](../benchmarks/results-2026-11-01.json) and [October 30 s986 results](../benchmarks/results-2026-10-30.json) have not been overwritten.

The earlier s1173 pass used the portable build after two failed portable GPU startups in the same boot. The native and portable artifacts have different build hashes, and generated continuations and allocation history were not held fixed. These observations therefore do not isolate how much of the increase came from reboot or memory bandwidth. Against historical s986, long-context code and prose 2 still remain lower, while long-context prose is slightly higher. Target attention precision, drafting policy, host policy and continuations differ from s986; this is not a matched numerical-policy regression test or a recommended-sampler quality conclusion.

Record boot history and the order of engine launches when comparing separate processes. On GB10, newly allocated buffers can have different bandwidth after repeated engine launches; a process restart alone does not normalize physical memory placement. Prefer alternating candidate and control within one model load, keeping weight allocations fixed, for small optimization experiments. For comparisons requiring separate loads, use reversed-order controls and report any unresolved memory-state confound. No allocation-specific bandwidth measurement was performed in this pass. The published cross-system figures do not include a matched memory-placement control.

For normal use, restore temperature 1, top-p 1, top-k -1 and high reasoning effort. Throughput on these short greedy prompts does not replace measurements of completed tasks, reasoning length, tool correctness, or response quality under those settings.
