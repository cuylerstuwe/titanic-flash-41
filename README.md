<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/81e32ce1-67aa-485b-8434-e47d80770f16" />

# Titanic Flash 4.1

A native Rust inference engine and OpenAI-compatible server for **DeepSeek-V4.1-Flash-1017**, running tensor parallel inference across **four NVIDIA GB10 systems**. The intended deployment is four DGX Sparks with dual 100 Gb/s RoCE connections and local NVMe storage.

The engine includes GPU model execution, speculative decoding, SSD-backed Engram lookup, prompt reuse, streaming tool calls, image input, and native foreground/background scheduling. It admits two requests, each with the full context capacity, and queues additional requests. The default layout keeps one physical execution bank; overlapping requests use bounded, lossless RAM snapshots rather than a permanently allocated second bank. Rust owns model execution and request scheduling; CUDA and compiled GPU operators perform model arithmetic. Python utilities handle installation and fleet lifecycle.

This checkout carries the **November 8 prefill and vision release (s1387)** on the s1345 reliability base. Long prompts prefill in about 22% less time (28% higher throughput): sequence-parallel early layers with row-local Engram, a row-local shared expert overlapped with communication, an MXFP8 exchange of the WO-B input, and host-side Engram staging overlapped with GPU work. Image turns get split and image-parallel vision encoding. The features are runtime options described in [prefill and vision options](docs/PREFILL.md); most are bit-identical, and two change only BF16 reduction order. Canonical target weights, precision boundaries, memory floors, the one-bank layout, image processing and recommended sampling settings are preserved. See [release validation](docs/VALIDATION.md) for evidence and tradeoffs.

At the recommended temperature 1 / top-p 1 settings on fresh boots, a 129,955-token cold prefill took **16.06 s versus 20.59 s for s1345** (8.09K versus 6.31K tokens/s), and a 93-image turn's time to first token fell from 17.16 s to 9.50 s. Decode was unchanged (92.8 versus 92.7 tokens/s on the same 2,940-token sampled completion). The added staged weights use about 3.5–4 GiB more memory per device.

## Performance

Single-stream measurements from **November 8, 2026 (s1387)**, using the published comparison's prompts, context file, request settings and timing function. The installed native production service ran **on the head GB10 against its head-local endpoint**, bypassing any external router or physical HTTP network hop. All 24 requests completed in one pass on the first model load after a four-device reboot; six short installed-service checks preceded it, none with a long prompt. Only Titanic was measured again; competitor results below are the author's existing published numbers.

The measured deployment enables the [prefill and vision options](docs/PREFILL.md). The portable profile supplied in this repository does not enable them yet, so a portable installation measures like the s1345 base for long-prompt prefill.

Median generation speed in **tokens/second**, three repetitions with a 512-token output budget:

| Starting context | Workload | SGLang stock, published | SGLang tuned, published | TensorFold, published | Titanic s1387 |
|---|---|---:|---:|---:|---:|
| Clean | Code | 71.6 | 103.1 | 110.9 | 120.3 |
| Clean | Prose | 32.9 | 46.3 | 57.7 | 65.2 |
| Clean | Prose 2 | 26.8 | 43.7 | 56.3 | 62.0 |
| ~131K tokens | Code | 57.4 | 86.0 | 94.7 | 111.0 |
| ~131K tokens | Prose | 32.0 | 45.8 | 57.7 | 62.1 |
| ~131K tokens | Prose 2 | 28.1 | 44.1 | 53.7 | 61.4 |

| Uncached ~131K prompt, time to first generated content | SGLang tuned, published | TensorFold, published | Titanic s1387 |
|---|---:|---:|---:|
| First long-context request | 32.45 s | 56.02 s | 16.31 s |

The comparison uses **temperature 0, top-p 1, thinking disabled**. These are benchmark settings, not the server's recommended-use defaults. “Prose 2” is the published fishing-village story workload, not the separate sampled-chat workload. Uncached timing means no reusable prompt state; it does not mean an empty operating-system file cache. Generation trials follow warmups and can reuse prompt state.

These are **cross-system comparisons**: Titanic ran on three ASUS GX10s and one DGX Spark, with two switched 100 Gb/s fabrics. The published runs used four DGX Sparks and a different network arrangement. These short throughput measurements do not establish completed-task quality or which system solves arbitrary agentic tasks fastest. Multi-stream results are deliberately omitted.

Titanic used the 2350 MHz cap. Every row comes from the complete s1387 pass, including lower results, rather than historical best scores. Decode rows are within about 1% of the earlier s1329 pass (prose 2 at ~131K is 4.7% higher); the first long-context request is 22% faster (20.99 s for s1329). [Historical comparisons and run accounting](docs/BENCHMARKS.md) retain earlier results separately.

Source: [published comparison and benchmark client](https://gist.github.com/kevintheo-ai/60eb093a5b4c54c4ac9a0b6b678408fa), pinned revision `7ae39b401e4e0887b2b7d5e4c4412cf60aa694f2`. [Full results](benchmarks/results-2026-11-08-s1387.json) include ranges, timing details, hardware and measured-build identity; [individual samples](benchmarks/samples-2026-11-08-s1387.json) omit generated text. [Reproduction instructions](docs/BENCHMARKS.md) use the original external timing function.

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
