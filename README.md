<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/81e32ce1-67aa-485b-8434-e47d80770f16" />

# Titanic Flash 4.1

A native Rust inference engine and OpenAI-compatible server for **DeepSeek-V4.1-Flash-1017**, running tensor parallel inference across **four NVIDIA GB10 systems**. The intended deployment is four DGX Sparks with dual 100 Gb/s RoCE connections and local NVMe storage.

The engine includes GPU model execution, speculative decoding, SSD-backed Engram lookup, prompt reuse, streaming tool calls, image input, and native foreground/background scheduling. It admits two requests, each with the full context capacity, and queues additional requests. Rust owns model execution and request scheduling; CUDA and compiled GPU operators perform model arithmetic. Python utilities handle installation and fleet lifecycle.

This checkout carries the **November 4 engine update (s1291)**: safer image-memory admission, reduced sampling/transport overhead, wider verification overlap, faster verified weight loading, and a fix for stale full-logit rows when verification width changes. Target precision and recommended sampling settings are preserved. See [release validation](docs/VALIDATION.md) for the native deployment checks and portable-package scope.

The benchmark table below remains the **November 1 s1173 measurement**. It has not been relabeled as a measurement of s1291.

## Performance

Single-stream measurements from **November 1, 2026, after a four-device reboot (s1173)**, using the published comparison's prompts, context file, request settings and timing function. The benchmark ran **on the head GB10 against its local HTTP endpoint**, verified to route through `lo`, bypassing any external router or physical network hop. All 24 requests completed in one pass on the first model load of that boot. Only Titanic was measured again; competitor results below are the author's existing published numbers.

Median generation speed in **tokens/second**, three repetitions with a 512-token output budget:

| Starting context | Workload | SGLang stock, published | SGLang tuned, published | TensorFold, published | Titanic s1173 |
|---|---|---:|---:|---:|---:|
| Clean | Code | 71.6 | 103.1 | 110.9 | 117.3 |
| Clean | Prose | 32.9 | 46.3 | 57.7 | 63.4 |
| Clean | Prose 2 | 26.8 | 43.7 | 56.3 | 60.4 |
| ~131K tokens | Code | 57.4 | 86.0 | 94.7 | 108.7 |
| ~131K tokens | Prose | 32.0 | 45.8 | 57.7 | 60.6 |
| ~131K tokens | Prose 2 | 28.1 | 44.1 | 53.7 | 56.8 |

| Uncached ~131K prompt, time to first generated content | SGLang tuned, published | TensorFold, published | Titanic s1173 |
|---|---:|---:|---:|
| First long-context request | 32.45 s | 56.02 s | 21.19 s |

The comparison uses **temperature 0, top-p 1, thinking disabled**. These are benchmark settings, not the server's recommended-use defaults. “Prose 2” is the published fishing-village story workload, not the separate sampled-chat workload. Uncached timing means no reusable prompt state; it does not mean an empty operating-system file cache. Generation trials follow warmups and can reuse prompt state.

These are **cross-system comparisons**: Titanic ran on three ASUS GX10s and one DGX Spark, with two switched 100 Gb/s fabrics. The published runs used four DGX Sparks and a different network arrangement. These short throughput measurements do not establish completed-task quality or which system solves arbitrary agentic tasks fastest. Multi-stream results are deliberately omitted.

Titanic used the approved 2350 MHz cap. All six generation medians in this complete pass were 2.3–6.7% higher than the [earlier s1173 portable-build pass](benchmarks/results-2026-11-01.json); first-long-prompt TTFT was essentially unchanged (21.19 versus 21.22 s). The earlier pass followed repeated model launches. Build hashes, continuations and allocation history differ, so the improvement cannot be attributed solely to reboot or memory bandwidth. [The previous s986 measurements](benchmarks/results-2026-10-30.json) also remain available; long-context code and prose 2 are still below those historical observations. See the [complete comparison and limits](docs/BENCHMARKS.md).

Source: [published comparison and benchmark client](https://gist.github.com/kevintheo-ai/60eb093a5b4c54c4ac9a0b6b678408fa), pinned revision `7ae39b401e4e0887b2b7d5e4c4412cf60aa694f2`. [Full results](benchmarks/results-2026-11-01-clean-boot.json) include ranges, timing details, hardware and measured-build identity; [individual samples](benchmarks/samples-2026-11-01-clean-boot.json) omit generated text. [Reproduction instructions](docs/BENCHMARKS.md) use the original external timing function.

## Model and quality

Use the canonical [DeepSeek-V4.1-Flash-1017 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), pinned to revision `1bae0e1088993b9b82de08f59de5be5a34e42170`. Weights are downloaded separately and are not in this repository.

The default sampler uses **temperature 1, top-p 1, top-k disabled**, with high reasoning effort. The server also supports greedy diagnostics. Other temperatures and nucleus cutoffs are rejected explicitly. The context capacity is 1,048,576 tokens; the default output budget is 262,144 tokens, subject to the remaining context.

The target retains the checkpoint's BF16 LM-head weights and produces FP32 logits. The current profile also preserves BF16 sliding-window attention operands and FP32 partial attention reductions. The draft head uses a faster representation; target verification uses distribution-preserving rejection sampling. Exact seeded output identity across hardware, graph shapes and floating-point reduction orders is not promised. A faster draft is useful only when accepted tokens and overall completion time improve.

## Setup

Follow **[the complete four-Spark setup recipe](docs/SETUP.md)**. It covers driver and network checks, the pinned build environment, checkpoint download verification, lossless Engram packing, GPU weight staging, per-rank distribution, authentication, service startup and rollback.

The release contains Rust source, CUDA source, a hash-checked Arm64/SM121 kernel bundle, deployment utilities, benchmark inputs and third-party notices. The recipe separately fetches and compiles one required AGPL-3.0 WO-A operator. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before deploying or redistributing a combined build.

Do not substitute a different checkpoint, quantization, GPU architecture, tensor-parallel size or kernel bundle without requalification. This is a specialized four-GB10 implementation, not a general-purpose model server.

## API and operation

See [API and operations](docs/OPERATIONS.md) for authenticated chat requests, tool and image input, Pi-compatible server timings, foreground/background priorities, health checks and lifecycle behavior. The default bind is loopback. Credentials and machine inventory stay in private runtime state, outside Git.

The measured engine has passed four-device model/state checks and foreground serving checks, including long prompts, reuse, branches, coding and tools. The portable installation wrapper is a separate packaging layer; its validation scope is recorded in [release validation](docs/VALIDATION.md). Do not infer a fresh-install qualification on every four-Spark configuration from these measurements.

## Licensing

**No license has been added for the project's original code.** Third-party components retain their own licenses and notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/). The checkpoint has its own terms. This repository does not grant a blanket MIT, Apache or other license over the combined stack.
