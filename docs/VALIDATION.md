# Validation status

## November 8 prefill and vision release (s1387)

s1387 keeps the s1345 reliability base: one physical execution bank, the 8 GiB operating and
8.5 GiB allocation floors, the 4 GiB shared snapshot ceiling, the 2350 MHz cap, canonical
checkpoint weights, target precision boundaries and the recommended sampler. It adds the
qualified image line (fused vision normalization, image-input cache hits during history
resolution, bounded image metadata deferral with pressure-releasable span pins, short-prefill
live-row bounds and a 2048-row plan for 1153–1536-token retained appends) and the runtime
options described in [prefill and vision options](PREFILL.md).

Most options are bit-identical. Sequence-parallel layers 0/1/14 and the row-local shared expert
change BF16 reduction order. A teacher-forced check scored 384 ground-truth tokens after a cold
131,071-token prefill: mean NLL 0.22508 for the s1345 configuration, 0.22450 for the candidate
and 0.22244 for an FP32-sum reference. Candidate-versus-baseline total variation averaged 0.0111
with 381/384 argmax agreement, comparable to baseline versus FP32 sums (0.0081, 382/384).
Rank-0 logits after the same cold prefill are bit-identical between that candidate configuration
and s1387, so the later options do not change outputs. These are next-token checks, not
completed-task quality scores.

Fresh-boot comparison, same 15-request head-local panel (temperature 1, top-p 1, high
reasoning), first model load after rebooting all four devices:

| Measurement | s1345 | s1387 options* |
|---|---:|---:|
| 129,955-token cold prefill | 20.59 s (6.31K tok/s) | 16.06 s (8.09K tok/s) |
| 93-image turn: prefill / time to first token | 7.94 s / 17.16 s | 6.15 s / 9.50 s |
| Plain-code decode (2,940 sampled tokens) | 92.7 tok/s | 92.8 tok/s |
| Lowest available memory, any device | 17.80 GiB | 14.18 GiB |

\*Measured on a build without the fused WO-B sum, which is bit-identical and 0.6% faster per
8K-row chunk in paired same-process timing. A probe that prefilled 262,143 tokens took 42.4 s
with the new prefill options off and 33.0 s with them on (same binary). Decode is unchanged.

The qualification fleet contains three ASUS GX10s and one DGX Spark (all GB10). The s1345 stress
panel completed 69 requests reaching 974,076 prompt tokens: a 28-image conversation, two context
resets, 16 later fresh images, foreground image preemption of a long background request and two
974K-token foreground peers holding 3,273,868,820 bytes of temporary snapshots. The lowest
available memory was 10.996 GiB (s1345: 14.644 GiB), above both floors. As in the s1345
qualification, the temporary test observer's 90°C guard stopped the panel during the final
cancellation fixture (Spark 90.0°C; other devices 87.8–89.5°C). There was no memory-guard abort
or GPU allocation/Xid message. Production has no request-error rule at that temperature; the
2350 MHz policy and hardware protections are unchanged.

The separate cancellation check passed: a 974,039-token, 28-image prefill completed in 139 s, a
forward continuation was cancelled after output began, the retry reused all 974,056 prompt tokens,
and a new image succeeded at 975,018 tokens with 973,824 cached. All four workers exited normally;
the lowest available memory was 13.19 GiB and the hottest device reached 87.8°C.

s1387 is installed through a journaled transaction with s1345 as the failed-start fallback, after
fresh boots and one production model load per device. Five head-local streaming requests passed
text, prompt reuse, continuation, image input and text after an image; a routed synthetic image
request passed with unchanged router configuration. Runtime hashes, clocks, clock/APST recovery
guards and kernel-health checks passed, and the reservation was released. See the
[sanitized qualification record](../validation/prefill-release-2026-11-08.json).

Native serving-source fingerprint:
`73ac5ceece161bb3d558326f0d080fcf57bc6b9bb53d3c7fe842a88103b2deb5`.
Portable source fingerprint: `fc05cf642937cd042a31fbf639bb40271a562fc4217036ba42624f1971a55cfa`.
Native Arm64 CPU checks passed 219 tests (25 ignored). Ten installation interruption/rollback
fixtures passed. The portable tree's compiled sources equal native s1387 except for the existing
public-tree edits (comments and configurable RoCE device names); portable Rust and deployment
checks and the portable GPU deployment wrapper were not rerun, and the portable profile is
unchanged. The native build and production profile received the checks above.

## November 6 forward-stream memory repair (s1345)

A later long-session incident still reached the physical admission floor after
all optional checkpoints had been reclaimed. The previous scratch loan was
working; the remaining fixed footprint included two full-context execution
banks and warmed graph variants for both, even during solo use.

The default profile now keeps one physical execution bank. It preserves the
active conversation's forward-prefix checkpoints and fully warms/retains the
primary append and decode graphs. Two logical lifetime admissions remain;
overlapping requests use complete RAM snapshots only when switching is needed.
Optional completed checkpoints remain capped at 2 GiB. Shared snapshot/cache
storage may use 4 GiB per rank on demand, subject to the unchanged physical floor;
optional rewind/ancestor acceleration yields to unfinished requests. There is
no SSD parking, weight/precision/sampling change or reduction of image resolution.
Same-class requests can finish serially if switching is too expensive in memory.
Incoming foreground work that cannot safely pause a background owner receives a
capacity error instead of being stranded behind a global priority hold.

The exploratory comparison used the same source, 2350 MHz and recommended sampler
on fresh boots. Both exploratory layouts used a four-variant graph-retention
budget; the final profile uses full primary warmup. 108 matched capped requests
covered cached appends, growing image
history, context resets and fresh images at the end. The vision rank ended with
17.182 GiB available in one-bank mode versus 9.125 GiB with two physical banks.
Prefill totals were 71.663 s versus 71.534 s for the growing long image conversation;
short cached-turn totals were 30.368 s versus 30.459 s. These single-order mechanism
screens do not establish a general speedup or task-quality result. Three further
one-bank requests exercised 438K background pause/resume, foreground image input
and subsequent prefix reuse. See the [sanitized comparison](../validation/memory-layout-2026-11-06.json).

A separate graph-retention component screen passed 56 paired records per device,
including exact logits, target/drafter/lookup state and following-six-position
rollback. Limiting graph variants saved memory but introduced recapture cost.
The release therefore retains full primary warmup; the optional graph-retention
control remains disabled in the supplied profile.

The native qualification fleet contains three ASUS GX10s and one DGX Spark,
all GB10. The final fully warmed build completed 69 requests reaching 974,076 prompt
tokens, with a 28-image conversation, two context resets, 16 subsequent fresh
images, foreground image preemption of a long background request, and two
974K-token foreground peers. The latter held 3,273,868,840 bytes of temporary
snapshot storage. The minimum observed available memory was 14.644 GiB on the
vision rank. The panel did not fully pass: its external thermal guard stopped
the head at 90.2°C during the final cancellation fixture, after several cold
near-million-token prefills. That interruption is retained as a limitation;
there was no memory-guard abort or new GPU allocation/Xid message. The 90°C
cutoff belongs only to the temporary test observer. Production has no request-
error rule at that temperature; its 2350 MHz policy and hardware protections remain. The cancellation/retry case is qualified
separately below, without weakening the temperature guard or changing clocks.

A focused fresh-boot follow-up completed a 974,039-token/28-image prefill,
cancelled a forward continuation after output began, and retried it using all
974,056 cached prompt tokens (zero prompt work). A new image then succeeded at
975,018 prompt tokens, reusing 973,824. All four workers exited normally, with
no memory/thermal guard abort or new GPU allocation/Xid messages. The vision
device finished with 16.812 GiB available. These are capped state and memory
checks, not completed-task quality measurements.

Native s1345 is installed after fresh boots and one production model load per
device, with s1338 as the failed-start fallback. Five installed head-local
streaming requests passed text, prompt reuse, continuation, image input and
text after image input. A synthetic image request through the external router
passed with Pi-compatible timing and unchanged router configuration. One
production generation stayed Ready. All-rank runtime hashes, 2350 MHz clocks,
clock/APST recovery guards and kernel-health checks passed; the reservation is
released. See the [sanitized qualification record](../validation/memory-repair-2026-11-06.json).

Native serving-source fingerprint:
`703d9e69d3c279c0e7a0ea846158bc44d89f9c0cd8d0e730001308ea6faf1221`.
Portable source fingerprint:
`3badeddbc48d994e61245b7754901a7c27b81a73e77e6eb46216b20d2b0cf6d4`.
Portable checks passed 215 Rust tests (22 fixture-dependent tests ignored) and 42
Python deployment checks; Linux Arm64 serving checks passed 213 (24 ignored).
Ten installation interruption/rollback fixtures passed. The core and vision
kernel operations are unchanged; no storage fault injection is compiled in.
This repair does not remeasure the published throughput table, which remains
s1329. The portable GPU deployment/provisioning wrapper was not rerun.


## November 5 image-workspace repair (s1338)

A long image conversation could receive `Image encoding cannot retain the UMA reserve`
even with only one execution owner. In the observed incident, image admission
remained 67.51 MiB short after reclaiming optional conversation-cache entries.
The worker group stayed ready; this was a recoverable capacity refusal, with
no accompanying GPU allocation or Xid error.

The image encoder now borrows the existing wide-prefill embedding arena for
temporary tensors between completed prefill chunks. It contains no KV cache,
retained model state or terminal logits. The completed image representation
remains independently owned, so subsequent image or text work cannot overwrite
it. The same stream is synchronized around its use. Configurations without a
sufficiently large donor arena retain the owned-workspace path.

At the incident's 51 × 91 patch geometry, this avoids 190 MiB of additional
workspace admission after patches are resident. Under pressure, image admission
also returns unused host allocator pages before reclaiming useful caches. The
8 GiB operating margin, 8.5 GiB allocation floor and full 512 MiB transient-driver
allowance remain. Weights, image geometry, arithmetic, precision, sampling,
cache budgets and the 2350 MHz clock policy are unchanged.

Qualification:

- Two four-device component panels compared owned and borrowed workspaces,
  image placement, cache reuse after scratch overwrite, prefix skipping and
  allocation-failure recovery. They covered canonical image fixtures, the
  incident geometry and an 84 × 105 patch geometry near the image-token limit.
  Each panel passed 12 comparison records and 8 fault/recovery records per
  device. Completed image representations matched exactly.
- A fresh-boot normal serving build passed all ten requests in a synthetic
  conversation reaching 438,139 prompt tokens with 17 images, plus one separate
  image request. Image encoding succeeded under actual host-memory pressure;
  the machine was already below the pressure target, so the pressure helper
  needed no artificial ballast. Optional older cache entries were reclaimed.
  Seven encodings completed below the former workspace-admission requirement;
  every post-allocation observation remained above the existing safety floor.
  The pressured continuation reused 436,864 tokens, the repeat reused 437,888,
  and the long prompt still reused 438,016 tokens after an overlapping image/text
  pair. Both requests in that pair completed.
- All four workers exited normally, with no new GPU allocation/Xid messages or
  memory/thermal guard aborts. Pi-compatible timing accounting passed.
- Frozen native host checks passed 209 tests; portable locked-dependency checks
  passed 212, both excluding 22 fixture-dependent tests. Linux Arm64 normal and
  diagnostic builds passed 210 and 211 tests, respectively, with 24 ignored.
  Ten journaled install/interruption/rollback fixtures passed.
- The encoder's operation sequence is unchanged. Core-kernel differences are
  confined to compiler-generated symbol names; executable/data/header bytes
  match, and the vision kernel is byte-identical.

The first full-model attempt completed five requests through the pressured
continuation, but an overly strict harness assertion rejected permitted optional
cache eviction. It is retained as partial evidence. The complete fresh-boot
repeat above instead checked preservation of the active continuation. These
are capped correctness and memory checks, not new throughput or task-quality
benchmarks.

Native serving-source fingerprint:
`c50a67973830b6779121e98d2e38143e1848f3f897a7117e80636449ca7e60ca`.
Portable source fingerprint:
`8b638de24c9397f13b48028dcc14e3ca358fc97a5335ec14cc80ab56bbdaae06`.
Fault injection is excluded from the normal serving binary. The portable
deployment wrapper and kernel bundle are unchanged; their provisioning and
GPU deployment have not been rerun for this repair. Published scores remain
identified as s1329.

Native s1338 is installed after fresh boots and one daily model load per device,
with s1334 as the failed-start fallback. Five head-local streaming requests
passed text, exact prompt reuse, continuation, image input and text after image
input. A synthetic image request through the external router also passed with
Pi-compatible timing and unchanged router configuration. One generation stayed
Ready throughout. Runtime hashes, 2350 MHz clocks, host-policy recovery guards
and kernel-health checks passed; the maintenance reservation is released.

## November 5 prefill memory-pressure repair (s1334)

An optional Engram lookahead check could terminate one worker when estimated
available memory fell below the 8 GiB operating margin. Its peers then lost the
worker group. This was a software guard failure; the incident did not establish
a GPU allocation failure or that 8 GiB is a minimum required by the hardware.

Serving now admits prefill at a committed request boundary using an all-device
decision. Under pressure it first returns unused host allocator pages and reclaims
optional caches. If the margin remains insufficient, it refuses that request
without terminating the group. Live and suspended request state is preserved.
Optional lookahead can also fall back to demand reads if memory availability
changes later. Actual allocation guards remain: the 8 GiB operating margin and
8.5 GiB allocation floor are unchanged. There are no weight, arithmetic, sampler,
clock-policy, cache-budget or GPU-kernel operation changes.

Qualification before activation:

- A finite four-device fault-injection check exercised pressure on the head and
  vision rank independently. Both refusal cases preserved committed model state
  and allowed a subsequent healthy retry. Recoverable admission pressure also
  continued successfully.
- Four alternating normal/low-headroom prefill trials produced identical model
  state and the next six verification-logit rows on all four devices. The late
  pressure was injected only on the vision rank, covering rank-local fallback.
- A normal serving build completed a 129,553-token prompt and a continuation
  under real host-memory pressure. The continuation reused 129,408 tokens while
  the constrained device began near 8.30 GiB available. Its existing cache-restore
  path handled pressure before the new admission branch. Consequently, a test
  assertion requiring the new branch's diagnostic did not pass: this is partial
  physical-pressure coverage, not a completed full screen or a physical test of
  the new refusal branch. All workers exited cleanly, without new GPU allocation,
  Xid, thermal-guard or memory-guard errors.
- Host native/advancing CPU checks passed (209 each); portable checks passed
  (212), each excluding 22 fixture-dependent tests. The Linux Arm64 serving
  build passed 210 tests with 24 ignored. Ten installation/rollback fixtures
  passed. Core-kernel differences are limited to compiler-generated symbol names;
  executable/data/header bytes are unchanged, and the vision kernel is identical.

Native serving-source fingerprint:
`fa619ddfb0ea9eeab06ac8d2e16d67be3824a54b6c18f389a31e92a8298dcb22`.
Portable source fingerprint:
`32a28b07d86efbbf15dccfcd6280ba4497af2156d9598a47209d6ae9d7715ee0`.
Fault injection is excluded from the normal serving binary. The portable
deployment wrapper and kernel bundle are unchanged; its provisioning and GPU
benchmark have not been rerun for this repair. Published scores remain s1329.

Native s1334 is installed after fresh boots and one daily model load per device,
with s1332 as the failed-start fallback. Five head-local streaming requests
passed text, exact prompt reuse, continuation, image input and text after image
input. A synthetic image request through the external router also passed with
Pi-compatible timing and unchanged router configuration. One generation stayed
Ready throughout. Runtime hashes, 2350 MHz clocks, host-policy recovery guards
and kernel-health checks passed; the maintenance reservation is released.

## November 5 paired scheduling repair (s1332)

A solo three-row verification could leave Engram's ahead-projection selection
enabled after consuming its producer. The paired executor uses an inline
producer. On a subsequent eager execution or new graph capture, inheriting the
solo selection could fail with `Engram consumer ran before its projection producer`
and terminate the worker group. Graph replay could hide the issue until a new
execution shape was needed; overlap was a trigger, not a sufficient condition
for every request to fail.

The paired executor now selects its own inline schedule for both owners before
eager execution or graph capture. Solo execution still selects its optimized
schedule on entry. The producer/consumer safety assertion remains in place.
No target arithmetic, sampling, weights, GPU kernel operations, clock policy,
cache budget or admission policy changed.

The finite four-device regression first reproduced the exact assertion in the
unpatched control. The repaired engine passed all 48 paired trials (12 per
device), including inherited and independent prefixes, both owners arriving
from different solo row widths, graph capture/replay, full/partial/zero acceptance,
full local logits, retained state and following verification. Native CPU tests
passed (210); portable CPU tests passed (212, with 22 fixture-dependent checks
ignored). Rebuilt core-kernel differences are confined to compiler-generated
symbol names; every executable/data/header byte is unchanged. The vision kernel
is byte-identical.

After fresh boots and one daily load per device, the installed native repair
passed solo-to-paired-to-solo serving, cancellation of one owner while its peer
completed, paired execution at 129,553 prompt tokens, completion of a third
queued request, exact prompt-cache reuse, image input and Pi timing accounting.
One generation remained Ready throughout, with no GPU allocation/Xid errors.
These were capped scheduling regressions, not new completed-task quality or
performance measurements.

Native source fingerprint:
`8e39ac2f43b92b8c2b2af24969bc13596c9f0152bee99e1ac44baef6c3b54687`.
Portable source fingerprint:
`e392be37e005b67a375d436c5a7676c9408783f3b4d88a61f2fbee79e77a7774`.
The existing portable deployment wrapper and kernel bundle are unchanged.
The four-device regression used the native deployment; this repair has not
repeated the portable provisioning or benchmark run. Published performance
numbers retain their s1329 identity.

## November 4 accumulated engine update (s1329)

This update carries the qualified s1291 correctness fix plus prompt/image
frontend reuse, bounded incremental context transport, idle continuation-cache
preparation, reduced draft/sampling overhead, adaptive small-transfer transport,
projection/cache hints, and measured prefill work bounds. The serving profile
matches all **196** native model environment settings. Canonical target weights,
BF16 LM-head weights, FP32 logits, the 2350 MHz cap, the 8 GiB physical reserve,
two full-context owners, image admission, native priority and Pi timing remain.

Wide WO-A verification reconstructs the checkpoint's operands exactly, retains
FP32 split partials, and rounds its output once to BF16. Its reduction order can
change sampled continuations. Native component/model checks and higher-precision
projection diagnostics distinguish this admissible rounding change from a loss
of checkpoint precision or stale state. Bitwise output identity is not claimed.

Native release qualification:

- Two 13-request arms were repeated in reverse launch order, with fresh boots
  and one model load per node for every arm. All requests completed; all four
  nodes stayed within the physical-memory/thermal guards and recorded no GPU
  allocation or Xid errors.
- At temperature 1 / top-p 1 / high reasoning, eight matching continuations
  decoded **3.59%** and **3.38%** faster in the two orders. The final capped token
  can use a different valid draw. Cold ~131K prefill was about **20.5 seconds**,
  within 0.12% in both orders. The [complete paired records](../benchmarks/release-comparison-s1329.json)
  retain changed trajectories and individual slower results.
- Two completed coding tasks passed supplied and held-out checks, with no tool
  errors. A maximum-effort tool request also passed. These are scoped quality
  checks, not a general benchmark score.
- Final release checks passed image cancellation with immediate resubmission,
  processed-image reuse, prompt reuse, one-token completion, and background
  pause/resume while a foreground request ran exclusively.
- Nine installation/rollback fixtures passed, including interruption after
  each changed file. The retained fallback is the previously qualified s1291.

The copy fixture took **20.63 s versus 18.69 s** in the reversed pair, producing
2,362 versus 2,123 tokens. Its shared prefix was faster; the trajectory changed
after wide verification. Repeating the same seed reproduces that difference but
does not establish a distribution-wide reasoning-length or outcome-speed effect.

Portable package checks:

- Native frozen source: **209 CPU tests passed, 22 ignored**. Portable source:
  **212 passed, 22 ignored**, plus the all-target compile check. Ignored checks
  require specialized fixtures/hardware and are not counted as passed.
- **42 Python deployment tests** passed, including package/loader pin agreement
  and missing enabled-kernel rejection. The last check covers a default-enabled MoE dependency even when no environment flag explicitly selects it.
- The pinned Arm64 serving build passed. Runtime verification checked **2,248**
  bundled artifact hashes, **18** separately built WO-A operators, **272** dynamic
  libraries, and NCCL **2.30.7**. Kernel executable bytes imported from the native
  qualification remain unchanged. Required notices are retained.
- The source and artifact privacy scan found no unresolved personal identifiers,
  machine addresses, credentials or private conversation fixtures. Synthetic
  address tests and required third-party legal contacts remain.

Portable compiled-source fingerprint:
`82b473a00c2297d48e60599388aac6836d52fbf6210ab96f06a12f9ab4742504`.
Native qualified-source fingerprint:
`1e25b6aed39969302a60150edc2f6518e573719532611b44d9cccf27e89ac39f`.
The portable fingerprint includes its naming, network/path adaptations and
package checks. The new WO-A variants are fetched and built by the explicit
external-component step in the setup recipe.

The portable four-node build passed three streaming/cache/timing checks, a
synthetic image request, four incremental-context checks (full history, extension,
rewind and missing-reference rejection), and the complete 24-request head-local
published-client benchmark. Startup initially exposed a missing default-enabled
MoE artifact; the package now includes it and validates its presence before load.
After the completed benchmark, Docker stop exceeded the wrapper's old timeout.
Fallback cleanup succeeded. The final wrapper uses container init, stops the four
ranks concurrently, and allows more time for GPU teardown. A separate fresh-boot
startup/request/shutdown check passed with normal logging disabled, all four GPUs
idle afterward, and no memory, thermal or kernel errors. Inference binaries,
kernels and sampling profile are unchanged by this lifecycle correction.

Native s1329 was activated on November 4 after rebooting all four devices,
with one daily model load per node. All deployed runtime hashes, the 2350 MHz
policy, independent host-policy recovery guards, and the physical-memory floor
passed. Synthetic image and greedy text requests completed through the normal
external router; Pi-compatible prefill timing and token accounting passed. The
router configuration was unchanged. s1291 remains the failed-start fallback.
The maintenance reservation was released after Ready. The native restoration wrapper initially
exceeded its startup-command deadline after lengthy repeated asset verification. Its
stationary retry recognized the same healthy generation and completed without
reloading the model; the subsequent routed and host checks passed.

The incremental protocol was tested directly on the portable endpoint; enabling
it in an external client/proxy integration is a separate task. The empty-fleet
provisioning limitation at the end of this document still applies.

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
and the first uncached ~131K prompt. That historical pass remains available separately from the current s1329 table.
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
