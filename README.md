<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/81e32ce1-67aa-485b-8434-e47d80770f16" />

# Titanic Flash 4.1

A native Rust inference engine and OpenAI-compatible server for **DeepSeek-V4.1-Flash-1017**, running tensor parallel inference across **four NVIDIA GB10 systems**. The intended deployment is four DGX Sparks with dual 100 Gb/s RoCE connections and local NVMe storage.

The engine includes GPU model execution, speculative decoding, SSD-backed Engram lookup, prompt reuse, streaming tool calls, image input, and native foreground/background scheduling. It admits two requests, each with the full context capacity, and queues additional requests. Rust owns model execution and request scheduling; CUDA and compiled GPU operators perform model arithmetic. Python utilities handle installation and fleet lifecycle.

This checkout carries the **November 5 scheduling repair (s1332)** on the accumulated s1329 engine: faster prompt/image preparation, optional incremental context transport, continuation-cache preparation, reduced drafting and transport overhead, and optimized projection kernels. The repair prevents a stale solo Engram scheduling choice from failing paired execution when requests overlap. It includes the earlier image-memory and full-logit correctness fixes. Canonical target weights, precision boundaries and recommended sampling settings are preserved. See [release validation](docs/VALIDATION.md) for numerical details and deployment scope.

At the recommended temperature 1 / top-p 1 settings, matched native continuations decoded **3.38–3.59% faster than s1291** across two fresh-boot launch orders. Cold prefill was effectively unchanged. These are scoped speed measurements; different sampled reasoning paths can still change total task time. The [paired comparison](benchmarks/release-comparison-s1329.json) retains the slower copy-task outcome as well.

## Performance

Single-stream measurements from **November 4, 2026 (s1329)**, using the published comparison's prompts, context file, request settings and timing function. The qualified portable build ran **on the head GB10 against loopback HTTP**, bypassing any external router or physical HTTP network hop. All 24 requests completed in one pass on the first model load after a four-device reboot. Only Titanic was measured again; competitor results below are the author's existing published numbers.

These measurements predate the s1332 scheduling repair; they have not been relabeled or remeasured for it.

Median generation speed in **tokens/second**, three repetitions with a 512-token output budget:

| Starting context | Workload | SGLang stock, published | SGLang tuned, published | TensorFold, published | Titanic s1329 |
|---|---|---:|---:|---:|---:|
| Clean | Code | 71.6 | 103.1 | 110.9 | 121.0 |
| Clean | Prose | 32.9 | 46.3 | 57.7 | 65.5 |
| Clean | Prose 2 | 26.8 | 43.7 | 56.3 | 62.4 |
| ~131K tokens | Code | 57.4 | 86.0 | 94.7 | 111.9 |
| ~131K tokens | Prose | 32.0 | 45.8 | 57.7 | 62.8 |
| ~131K tokens | Prose 2 | 28.1 | 44.1 | 53.7 | 58.6 |

| Uncached ~131K prompt, time to first generated content | SGLang tuned, published | TensorFold, published | Titanic s1329 |
|---|---:|---:|---:|
| First long-context request | 32.45 s | 56.02 s | 20.99 s |

The comparison uses **temperature 0, top-p 1, thinking disabled**. These are benchmark settings, not the server's recommended-use defaults. “Prose 2” is the published fishing-village story workload, not the separate sampled-chat workload. Uncached timing means no reusable prompt state; it does not mean an empty operating-system file cache. Generation trials follow warmups and can reuse prompt state.

These are **cross-system comparisons**: Titanic ran on three ASUS GX10s and one DGX Spark, with two switched 100 Gb/s fabrics. The published runs used four DGX Sparks and a different network arrangement. These short throughput measurements do not establish completed-task quality or which system solves arbitrary agentic tasks fastest. Multi-stream results are deliberately omitted.

Titanic used the 2350 MHz cap. Every row comes from the complete s1329 pass, including lower results, rather than historical best scores. Failed package-startup attempts preceded the fresh reboot and produced no benchmark samples. [Historical comparisons and run accounting](docs/BENCHMARKS.md) retain earlier native-build results separately.

Source: [published comparison and benchmark client](https://gist.github.com/kevintheo-ai/60eb093a5b4c54c4ac9a0b6b678408fa), pinned revision `7ae39b401e4e0887b2b7d5e4c4412cf60aa694f2`. [Full results](benchmarks/results-2026-11-04-s1329.json) include ranges, timing details, hardware and measured-build identity; [individual samples](benchmarks/samples-2026-11-04-s1329.json) omit generated text. [Reproduction instructions](docs/BENCHMARKS.md) use the original external timing function.

## Model and quality

Use the canonical [DeepSeek-V4.1-Flash-1017 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), pinned to revision `1bae0e1088993b9b82de08f59de5be5a34e42170`. Weights are downloaded separately and are not in this repository.

The default sampler uses **temperature 1, top-p 1, top-k disabled**, with high reasoning effort. The server also supports greedy diagnostics. Other temperatures and nucleus cutoffs are rejected explicitly. The context capacity is 1,048,576 tokens; the default output budget is 262,144 tokens, subject to the remaining context.

The target retains the checkpoint's BF16 LM-head weights and produces FP32 logits. The current profile also preserves BF16 sliding-window attention operands and FP32 partial attention reductions. The draft head uses a faster representation; target verification uses distribution-preserving rejection sampling. Exact seeded output identity across hardware, graph shapes and floating-point reduction orders is not promised. A faster draft is useful only when accepted tokens and overall completion time improve.

## Setup

Follow **[the complete four-Spark setup recipe](docs/SETUP.md)**. It covers driver and network checks, the pinned build environment, checkpoint download verification, lossless Engram packing, GPU weight staging, per-rank distribution, authentication, service startup and rollback.

The release contains Rust source, CUDA source, a hash-checked Arm64/SM121 kernel bundle, deployment utilities, benchmark inputs and third-party notices. The recipe separately fetches and compiles the required AGPL-3.0 WO-A component, including its compact wider-row variants. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before deploying or redistributing a combined build.

Do not substitute a different checkpoint, quantization, GPU architecture, tensor-parallel size or kernel bundle without requalification. This is a specialized four-GB10 implementation, not a general-purpose model server.

## API and operation

See [API and operations](docs/OPERATIONS.md) for authenticated chat requests, tool and image input, Pi-compatible server timings, foreground/background priorities, health checks and lifecycle behavior. An optional [incremental context protocol](docs/CONTEXT_TRANSPORT.md) supports extensions and rewinds with explicit confirmation and full-context fallback. Clients must implement it explicitly. The default bind is loopback. Credentials and machine inventory stay in private runtime state, outside Git.

The measured engine has passed four-device model/state checks and foreground serving checks, including long prompts, reuse, branches, coding and tools. The portable installation wrapper is a separate packaging layer; its validation scope is recorded in [release validation](docs/VALIDATION.md). Do not infer a fresh-install qualification on every four-Spark configuration from these measurements.

## Licensing

**No license has been added for the project's original code.** Third-party components retain their own licenses and notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/). The checkpoint has its own terms. This repository does not grant a blanket MIT, Apache or other license over the combined stack.
