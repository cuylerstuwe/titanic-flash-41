# Validation status

## November 4 engine update (s1291)

This source update carries the qualified s1242 engine and its s1291 full-logit
correctness fix. The CUDA graph cache now distinguishes the number of live rows:
replaying a capture made for a smaller batch can no longer leave later gathered
rows stale. It does not change target arithmetic, checkpoint precision or sampling
settings. The earlier distributed sampler avoids gathering the complete vocabulary
on its ordinary multirow path; full-logit and greedy consumers still require the
correct gathered rows.

The accumulated release also includes image-allocation admission and reclaim
handling, distribution-preserving sharded sampling and fused transport, overlap
for wide verification and Engram reads, and pipelined weight verification/loading.
It preserves canonical target weights, FP32 logits, the 2350 MHz profile, the
8 GiB physical-memory reserve, two full-context request owners, native priority,
image reuse and Pi-compatible timing. The serving defaults match all 167 native
model environment settings. Later performance experiments are excluded.

Native qualification before activation:

- All 48 rank/trial combinations matched gathered live rows against the local
  FP32 logits, with exact rollback state.
- Six HTTP requests passed greedy and recommended-settings text/image paths,
  prompt reuse and Pi timing/cache accounting.
- The underlying s1242 release passed a matched 36-request single-stream panel,
  followed by focused cancellation/image resubmission, duplicate image reuse,
  priority, tool and completed coding checks. These are scoped checks, not a
  general quality score or a new published-client benchmark.
- Nine installation/recovery fixtures passed, including interrupted installation
  and rollback. The native activation retains s1242 as its failed-start fallback.

Activation on November 4 freshly rebooted all four GB10s and loaded s1291 once
per device. All four runtime hashes, the 2350 MHz clock policy and independent
host-policy recovery guards passed. A synthetic image request through the
normal external router completed correctly at temperature 1 / top-p 1 / high
reasoning, with Pi-compatible prefill timings and consistent token accounting.
The maintenance reservation was released after Ready; the router configuration
was unchanged.

Portable package checks for this update:

- `cargo check --locked --features serving --all-targets` passed.
- `cargo test --locked --features serving --lib`: **190 passed, 16 ignored**.
  Ignored hardware/specialized checks are not counted as passed.
- **39 Python deployment tests** passed. A new inventory check rejects a missing
  fenced projection replacement before runtime preparation or verification.
- All **2,144** bundled artifact hashes and **83** compiled projection mappings
  were checked. Seven 32-row projection bundles were added with their native
  executable bytes unchanged and required source/license notices retained.
- Public source, binaries and metadata were scanned for personal information and
  secrets. No new private prompts, decoded conversation fixtures or operational
  configuration are included.

Portable compiled-source fingerprint:
`9a7a8351482dd691c43d0b3f7f9aed3c2223cde9b5b6c30bddde403203927bf4`.
Native qualified-source fingerprint:
`8394bfba7ba369edc643b85fa2be111b2ee841b294923231c3fa7f4e059f520f`.
They differ because the portable package retains its naming, network/path
adaptations and package tests. The existing pinned build recipe compiles the
updated Rust and core CUDA sources; prebuilt projection dependencies are checked
by the runtime inventory.

This update has **not** repeated the portable Arm64 build and four-node portable
controller startup. Those were last exercised for s1173 below. The native s1291
release has separate real-hardware qualification; CPU package checks do not
establish a new portable GPU run or empty-fleet provisioning result.

The November 4 head-local published-client benchmark completed all 24 requests
on this native s1291 generation, with six three-trial single-stream workloads
and the first uncached ~131K prompt. The README now uses that complete pass.
The engine stayed Ready without a reload; this performance run does not expand
the portable-wrapper or quality qualification. See [benchmark details](BENCHMARKS.md).

## November 1 engine update (s1173)

The updated engine is active in its qualified four-GB10 deployment. This checkout
ports that engine into the portable package. The portable controller passed a
focused four-rank startup and serving check using existing validated weight
atlases; it has not replaced the deployment's existing lifecycle controller.

Completed package checks:

- All 155 model environment settings match the activated recipe. The update
  retains canonical target weights, BF16 sliding-window attention storage,
  FP32 attention partials, draft temperature 0.7 and causal cutoff 0.25. It
  reserves two admitted owners through terminal completion, each with full 1M
  context, and preserves the 8 GiB physical-memory reserve.
- The supplemental kernel imports preserve every executable byte from the
  activated engine. The complete package inventory currently contains 2,088
  hashed files. Imported metadata is separately sanitized; the supplemental
  pin list alone is not a proof of complete runtime dependencies.
- `cargo check --locked --features serving --all-targets` passed.
  `cargo test --locked --features serving --lib` passed **168 tests**, with
  **16 ignored** hardware/specialized checks. Those ignored tests are not
  counted as passed. Two subsequently added CPU tests also passed, checking
  the four paired-projection manifest pins against the packaged metadata.
- **38 Python tests** passed for deployment configuration, rank-specific vision
  staging, temporary clock/NVMe policies, partial startup cleanup, stale timer
  ownership and lease renewal failures. These use controlled fakes; they do not
  establish a real four-node startup or physical policy-restoration result.
- A separate real four-node host-policy check passed. Without controller
  renewal, the independent timers restored stock GPU clocks on all four nodes
  and the Spark's exact original NVMe APST setting/table. Three ordinary renewal
  rounds left the durable control records unchanged; renewals stayed in RAM.
- Fifteen newly required token fixtures were decoded with the canonical
  tokenizer and reviewed privately. No private session transcripts are included.

The pinned Arm64 Rust/CUDA build also passed. The completed runtime check
verified all **2,088** bundled artifact hashes, **14** external WO-A operators,
dynamic dependencies for **258** shared libraries, and NCCL **2.30.7**. Compiled
source fingerprint:
`203eb99934f2169016a734670bf5aa57c6e1dc9ce69dbf59a328acf15eaa31ad`.
The release build needs more than a 2 GiB memory limit; the documented 8 GiB
build limit was used successfully. Build jobs must not crowd an active model's
physical-memory reserve.

An initial GPU startup correctly rejected a stale paired-projection manifest
pin: privacy scrubbing had removed a build-path string from the manifest. The
pin now matches that reviewed portable metadata. The kernel's executable bytes,
geometry and numerical contract are unchanged. The two added CPU checks cover
this failure before a model load.

The corrected package reached Ready on all four ranks with its default logging
settings. Three foreground requests passed fresh streaming, cached streaming,
cached nonstreaming and timing/cache-count checks. A fourth request correctly
identified a synthetic solid-color image using the rank-3 vision encoder. These
checks used temperature 1, top-p 1, top-k disabled and high reasoning. They test
deployment integration, not broad task quality. The end-to-end empty-fleet
provisioning limitation below still applies.

The head-local published-client comparison then completed all **24 requests**
on that same portable build, using default foreground priority and no external
router. Fresh s1173 summaries and text-free sample records are under
`benchmarks/*2026-11-01.json`. Published competitor values were reused, not
remeasured. This greedy/no-thinking diagnostic is separate from the recommended
sampling profile used in the serving checks.

A later head-local pass completed all **24 requests** on the installed native
s1173 build, on the first model load after rebooting all four devices. That
historical pass is retained; summaries and text-free samples are in
`benchmarks/*2026-11-01-clean-boot.json`. This is a separate native-build
measurement and does not extend the portable qualification above. See
[benchmark conditions and comparisons](BENCHMARKS.md).

The main build uses CUDA 13.0.88. Selected bundled mHC statistics kernels were
assembled with PTXAS 13.1.80, as recorded in their manifests; do not describe the
entire operator bundle as having a single compiler version. These prebuilt
artifacts are consumed directly by the recipe and verified by hash.

## Historical October 30 release (s986)

The released inference implementation is based on the qualified four-GB10 engine measured in the README. Public packaging adds project naming, explicit per-node network configuration, installation scripts and a portable lifecycle wrapper. It does not intentionally change target-model arithmetic, canonical weights or sampling rules.

### Completed checks

- The underlying engine passed four-device model/logit and retained-state comparisons for its accumulated changes, plus long-context, prompt reuse, branching, cancellation and lifecycle checks. Qualifying numerical reduction-order differences are distinguished from stale state or incorrect memory ownership; exact seeded output identity is not the general quality criterion.
- The final serving candidate passed six foreground requests at temperature 1, top-p 1, top-k -1 and high reasoning, covering a 267,369-token prompt, regeneration, branching, a partial-prefill path, a completed coding answer and a tool call. The coding answer passed three submitted assertions and six additional checks. This is a scoped smoke, not a comprehensive quality score.
- The same model build was activated on the four-GB10 deployment. Fresh streaming, cached streaming and cached nonstreaming calls through its normal router passed with prompt-processing timings and consistent usage/cache accounting. All four ranks were Ready under the qualified deployment controller.
- The published-client single-stream comparison completed all 24 requests on the head's loopback endpoint. Competitor numbers were taken from the published source, not remeasured. Raw timing summaries and sample metadata are included under `benchmarks/`.
- The public Arm64 toolchain image built successfully from the registry-pinned base. A clean serving-only Rust/CUDA build and external operator export succeeded in that image. CUDA was 13.0.88, Rust 1.99.0, Triton 3.7.1 and NCCL 2.30.7; the NCCL runtime matched the measured deployment.
- `verify_runtime.py` verified 1,985 bundled artifact hashes, 14 separately compiled WO-A operators and dynamic dependencies for 243 shared libraries. The host CUDA driver library is supplied at runtime by NVIDIA Container Toolkit. Compiled public source fingerprint: `d3c728d2876ff9b4c88e7325833363cab9f3bd04366385f140df42e8ccd7b06b`.
- `cargo test --locked --features serving --lib`: **135 passed, 0 failed, 12 ignored**. Ignored checks require their specialized fixtures/hardware; they are not claimed as passed by the CPU run.
- `tools/test_deployment.py`: **5 passed**, covering unsafe/ambiguous fleet configuration, exact Engram byte layout, checkpoint download integrity/path boundaries and a clean operator stop during startup. Shell and Python syntax checks passed.
- Canonical checkpoint metadata and the tokenizer were fetched and checked against the pinned revision. The bundled model configuration matched the canonical configuration. Published source and binary assets were scanned for private machine paths, addresses, credentials and development transcripts; integer-token fixtures were decoded for a separate content check. Required upstream attribution was retained.

### Scope limits

The full **download → repack all weights → stage all four ranks → serve on four newly provisioned Sparks** sequence has not been qualified end to end on a second empty fleet. Existing validated weight atlases were used for the engine qualification and benchmarks. The byte-packing and setup components have the checks above; this is not a claim that every fresh OS, driver or network configuration has been exercised.

The October 30 portable-runtime startup found an omitted nested RMS dependency. The bundle was corrected, but that historical validation stopped at CPU integrity/dependency checks. The November 1 checks above subsequently exercised the updated portable wrapper on all four GPUs. Daily serving continues to use its existing qualified lifecycle controller.

The short benchmark uses greedy decoding with thinking disabled; recommended-setting task quality and outcome speed cannot be inferred from its token rate. Concurrent-stream throughput, long-duration new-wrapper fault recovery and arbitrary sampling parameters are not qualified by this release. Kernel/profile changes, a different model revision or a different GPU/network arrangement require their own focused checks before relying on them in production.

No performance number in the README is a measurement of an untested fresh-install script or an assurance of identical speed on every four-Spark system.
