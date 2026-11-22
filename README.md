<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/81e32ce1-67aa-485b-8434-e47d80770f16" />

# Titanic Flash 4.1

A native Rust inference engine and OpenAI-compatible server for **DeepSeek-V4.1-Flash-1017**, running tensor parallel inference across **four NVIDIA GB10 systems**. The intended deployment is four DGX Sparks with dual 100 Gb/s RoCE connections and local NVMe storage.

The engine includes GPU model execution, speculative decoding, SSD-backed Engram lookup, prompt reuse, streaming tool calls, image input, and native foreground/background scheduling. It admits four requests, each with the full context capacity, and queues additional requests. A single execution bank uses independently growing regions for the four conversations. Compatible requests decode together, and the scheduler can park and restore a request when another needs the primary region. Rust handles model execution and scheduling. CUDA kernels run model arithmetic. Python utilities handle installation and fleet lifecycle.

**Current release s1690 (November 21).** Screenshot conversations can share decode steps after their images have been processed. The server coordinates vision workspace allocation across devices and preserves forward checkpoints during temporary input pressure. Four independent conversations with 70 screenshots each completed nine turns at about 1.01 million tokens per conversation. [Screenshot qualification](docs/VALIDATION.md#november-21-screenshot-serving-repair-s1690)

The preceding s1682 release added source-verified BF16 attention indexing, staged reads during sparse attention verification, and additional prefill chunk sizes. It reserves memory for forward continuations and maps the other conversation regions as they grow. A temporary Linux memory policy reduces background page movement while each worker serves.

On four GB10s, a cold 129,553-token prompt took **13.65 s** in s1682 and **14.10 s** in s1621. At roughly 131K context, six paired coding samples with the same output length decoded at **90.0 vs 85.6 tok/s** pooled. All four long chats in a separate test reached about 997K tokens and finished, and a return after 17 unrelated requests reused 997,422 tokens. [Release validation](docs/VALIDATION.md#november-21-serving-release-s1682) · [Four-Spark setup](docs/SETUP.md)

The November 19 release (s1621) introduced the local NVMe snapshot tier and full-length serving regions for concurrent conversations. The November 18 release (s1599) improved short tool continuations and retained useful snapshots under cache pressure. [Earlier release measurements](docs/VALIDATION.md#november-19-disk-tier-and-regions-release-s1621)

The **November 10 cache-format and attention release (s1399)**, on the November 10 prefill and cache release (s1394), introduced the reference cache formats. With `CANADA_SCRATCH_KV_FORMAT=v41`, the attention caches use the formats of DeepSeek's reference implementation: a V4.1 sliding-window cache (528 B/token) and a V4.1-FP4 compressed cache (288 B/token), instead of 584 B/token for both. A long conversation's resident state needs about 56% of its earlier memory. Teacher-forced against the reference forward on a 5,499-token text, the engine's divergence is 0.0104 nats per token. That is below the reference's own split-K yardstick on the same text (0.0110), with no clear top-1 flips. Above 8 rows, attention runs an adapted kernel from [pico-dgx-ds4](https://github.com/Picom-code/pico-dgx-ds4) (MIT), about twice as fast as the previous kernels in a same-load screen. On fresh boots, cold source-code prompts took 4–7% less time (127K tokens: 13.95 s instead of 15.01 s); decode is about unchanged. The session cache can now keep four long conversations resident: four 543K–605K-token conversations taking turns returned in a median 0.41 s, with 99.99% of prompt tokens served from the cache. At about 920K tokens each, two of four stayed resident within the 4 GiB shared image budget, which the November 15 release raised. The options are enabled in native production and described in [operations](docs/OPERATIONS.md#reference-cache-formats-and-attention).

The **November 10 prefill and cache release (s1394)**, on the November 9 checkpoint ladder (s1389), stopped computing layer-20 output rows that the decoder replay never reads, and reads a prompt's first Engram rows behind more of the first chunk; both are bit-identical. On the 15-request serving tape, a 129,955-token cold prefill took 15.37 s versus 16.10 s for s1389, and source-code prompts whose Engram rows were not cached took 0.87 s instead of 1.05 s at 3.4K tokens. A conversation displaced by an unrelated request (for example, a short title request between two turns of an agent session) now keeps its newest checkpoint: in a synthetic screen its 62K-token follow-up took 0.35 s instead of a 7.0 s cold prefill, with bit-identical logits to an uninterrupted restore. Decode was unchanged. The options are enabled in native production and described in [operations](docs/OPERATIONS.md#prefill-without-unused-layer-20-rows).

The **November 9 checkpoint ladder (s1389)**, on the November 8 prefill and vision release (s1387), made the server restore a nearby earlier checkpoint when an agent harness rewrites older history (offloading old images or pruning old tool output) instead of re-prefilling the whole conversation; in a synthetic 235K-token screen an edit three quarters of the way in took 8.8 s to first token instead of 27.3 s. It is a cache-retention choice with no arithmetic change, enabled in native production and described in [operations](docs/OPERATIONS.md#earlier-checkpoints-for-edited-history).

The s1387 release on the s1345 reliability base cut long-prompt prefill time by about 22% (28% higher throughput): sequence-parallel early layers with row-local Engram, a row-local shared expert overlapped with communication, an MXFP8 exchange of the WO-B input, and host-side Engram staging overlapped with GPU work. Image turns get split and image-parallel vision encoding. The features are runtime options described in [prefill and vision options](docs/PREFILL.md); most are bit-identical, and two change only BF16 reduction order. Canonical target weights, precision boundaries, memory floors, the one-bank layout, image processing and recommended sampling settings are preserved. See [release validation](docs/VALIDATION.md) for evidence and tradeoffs.

At the recommended temperature 1 / top-p 1 settings on fresh boots, a 129,955-token cold prefill took **16.06 s versus 20.59 s for s1345** (8.09K versus 6.31K tokens/s), and a 93-image turn's time to first token fell from 17.16 s to 9.50 s. Decode was unchanged (92.8 versus 92.7 tokens/s on the same 2,940-token sampled completion). The added staged weights use about 3.5–4 GiB more memory per device.

## Performance

**November 21, 2026 · installed s1682 · single stream.** The benchmark client ran on the head GB10 and connected to the server over loopback. All 24 requests completed in one pass at the 2350 MHz cap. SGLang and TensorFold figures come from the original four-Spark comparison. Titanic used three ASUS GX10s and one DGX Spark with two switched 100 Gb/s fabrics. Differences in hardware, networking and prior traffic also affect cross-system comparisons.

Speeds are in **tokens per second**. Each Titanic score is the median of three 512-token decode trials.

| Starting context | Workload | SGLang stock, published | SGLang tuned, published | TensorFold, published | Titanic s1682 |
|---|---|---:|---:|---:|---:|
| Clean | Code | 71.6 | 103.1 | 110.9 | 137.9 |
| Clean | Prose | 32.9 | 46.3 | 57.7 | 76.7 |
| Clean | Prose 2 | 26.8 | 43.7 | 56.3 | 65.9 |
| ~131K tokens | Code | 57.4 | 86.0 | 94.7 | 125.2 |
| ~131K tokens | Prose | 32.0 | 45.8 | 57.7 | 73.5 |
| ~131K tokens | Prose 2 | 28.1 | 44.1 | 53.7 | 70.8 |

| First uncached long prompt | SGLang tuned, published | TensorFold, published | Titanic s1682 |
|---|---:|---:|---:|
| Time to first generated content | 32.45 s | 56.02 s | **13.97 s** |

The server reported **0 cached tokens out of 131,064** on the first long request. The previous native s1387 run took 16.31 s on that prompt. Each of the nine measured long-context decode trials then reused all but one prompt token.

All six Titanic decode medians were higher than on s1387, by 6.3–18.3%. The releases generated different continuations on the same prompts, which can change speculative decode speed. This benchmark uses temperature 0, top-p 1 and thinking disabled. “Prose 2” is the fishing-village story. Separate qualification covers the recommended sampler and completed agentic tasks. [s1682 qualification](docs/VALIDATION.md#november-21-serving-release-s1682)

[Published comparison and client](https://gist.github.com/kevintheo-ai/60eb093a5b4c54c4ac9a0b6b678408fa) · [Full results](benchmarks/results-2026-11-21-s1682.json) · [All 24 samples](benchmarks/samples-2026-11-21-s1682.json) · [Method and previous releases](docs/BENCHMARKS.md)

## Model and quality

Use the canonical [DeepSeek-V4.1-Flash-1017 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), pinned to revision `1bae0e1088993b9b82de08f59de5be5a34e42170`. Weights are downloaded separately and are not in this repository.

The default sampler uses **temperature 1, top-p 1, top-k disabled**, with high reasoning effort. The server also supports greedy diagnostics. Other temperatures and nucleus cutoffs are rejected explicitly. The context capacity is 1,048,576 tokens; the default output budget is 262,144 tokens, subject to the remaining context.

The target retains the checkpoint's BF16 LM-head weights and produces FP32 logits. The current profile also preserves BF16 sliding-window attention operands and FP32 partial attention reductions. The draft head uses a faster representation; target verification uses distribution-preserving rejection sampling. Exact seeded output identity across hardware, graph shapes and floating-point reduction orders is not promised. A faster draft is useful only when accepted tokens and overall completion time improve.

## Setup

Follow **[the complete four-Spark setup recipe](docs/SETUP.md)**. It covers driver and network checks, the pinned build environment, checkpoint download verification, lossless Engram packing, GPU weight staging, derived prefill matrices, per-rank distribution, authentication, service startup and rollback.

The release contains Rust source, CUDA source, a hash-checked Arm64/SM121 kernel bundle, deployment utilities, benchmark inputs and third-party notices. The recipe separately fetches and compiles the required AGPL-3.0 WO-A component, including its compact wider-row variants. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before deploying or redistributing a combined build.

Do not substitute a different checkpoint, quantization, GPU architecture, tensor-parallel size or kernel bundle without requalification. This is a specialized four-GB10 implementation, not a general-purpose model server.

## API and operation

See [API and operations](docs/OPERATIONS.md) for authenticated chat requests, tool and image input, Pi-compatible server timings, foreground/background priorities, health checks and lifecycle behavior. An optional [incremental context protocol](docs/CONTEXT_TRANSPORT.md) supports extensions and rewinds with explicit confirmation and full-context fallback. Clients must implement it explicitly. The default bind is loopback. Credentials and machine inventory stay in private runtime state, outside Git.

The measured engine has passed four-device model/state checks and foreground serving checks, including long prompts, reuse, branches, coding and tools. The portable installation wrapper is a separate packaging layer; its validation scope is recorded in [release validation](docs/VALIDATION.md). Do not infer a fresh-install qualification on every four-Spark configuration from these measurements.

## Licensing

**No license has been added for the project's original code.** Third-party components retain their own licenses and notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/). The checkpoint has its own terms. This repository does not grant a blanket MIT, Apache or other license over the combined stack.
