# Release validation — October 30, 2026

The released inference implementation is based on the qualified four-GB10 engine measured in the README. Public packaging adds project naming, explicit per-node network configuration, installation scripts and a portable lifecycle wrapper. It does not intentionally change target-model arithmetic, canonical weights or sampling rules.

## Completed checks

- The underlying engine passed four-device model/logit and retained-state comparisons for its accumulated changes, plus long-context, prompt reuse, branching, cancellation and lifecycle checks. Qualifying numerical reduction-order differences are distinguished from stale state or incorrect memory ownership; exact seeded output identity is not the general quality criterion.
- The final serving candidate passed six foreground requests at temperature 1, top-p 1, top-k -1 and high reasoning, covering a 267,369-token prompt, regeneration, branching, a partial-prefill path, a completed coding answer and a tool call. The coding answer passed three submitted assertions and six additional checks. This is a scoped smoke, not a comprehensive quality score.
- The same model build was activated on the four-GB10 deployment. Fresh streaming, cached streaming and cached nonstreaming calls through its normal router passed with prompt-processing timings and consistent usage/cache accounting. All four ranks were Ready under the qualified deployment controller.
- The published-client single-stream comparison completed all 24 requests on the head's loopback endpoint. Competitor numbers were taken from the published source, not remeasured. Raw timing summaries and sample metadata are included under `benchmarks/`.
- The public Arm64 toolchain image built successfully from the registry-pinned base. A clean serving-only Rust/CUDA build and external operator export succeeded in that image. CUDA was 13.0.88, Rust 1.99.0, Triton 3.7.1 and NCCL 2.30.7; the NCCL runtime matched the measured deployment.
- `verify_runtime.py` verified 1,985 bundled artifact hashes, 14 separately compiled WO-A operators and dynamic dependencies for 243 shared libraries. The host CUDA driver library is supplied at runtime by NVIDIA Container Toolkit. Compiled public source fingerprint: `d3c728d2876ff9b4c88e7325833363cab9f3bd04366385f140df42e8ccd7b06b`.
- `cargo test --locked --features serving --lib`: **135 passed, 0 failed, 12 ignored**. Ignored checks require their specialized fixtures/hardware; they are not claimed as passed by the CPU run.
- `tools/test_deployment.py`: **5 passed**, covering unsafe/ambiguous fleet configuration, exact Engram byte layout, checkpoint download integrity/path boundaries and a clean operator stop during startup. Shell and Python syntax checks passed.
- Canonical checkpoint metadata and the tokenizer were fetched and checked against the pinned revision. The bundled model configuration matched the canonical configuration. Published source and binary assets were scanned for private machine paths, addresses, credentials and development transcripts; integer-token fixtures were decoded for a separate content check. Required upstream attribution was retained.

## Scope limits

The new portable fleet wrapper and the full **download → repack all weights → stage all four ranks → serve on four newly provisioned Sparks** sequence have not been qualified end to end on a second empty fleet. Existing validated weight atlases were used for the engine qualification and benchmarks. The byte-packing and setup components have the checks above; this is not a claim that every fresh OS, driver or network configuration has been exercised.

An early portable-runtime startup found an omitted nested RMS dependency. The bundle was corrected, and the complete artifact/dependency check above passes. The corrected portable wrapper has not subsequently undergone a four-rank GPU startup in this release validation. The known-good daily deployment continues to use its qualified lifecycle controller. Do not confuse CPU integrity checks with a GPU execution proof for a new deployment environment.

The short benchmark uses greedy decoding with thinking disabled; recommended-setting task quality and outcome speed cannot be inferred from its token rate. Concurrent-stream throughput, long-duration new-wrapper fault recovery and arbitrary sampling parameters are not qualified by this release. Kernel/profile changes, a different model revision or a different GPU/network arrangement require their own focused checks before relying on them in production.

No performance number in the README is a measurement of an untested fresh-install script or an assurance of identical speed on every four-Spark system.
