# API and operations

The Rust server exposes authenticated `GET /health`, `GET /v1/models` and `POST /v1/chat/completions`. Use the bearer credential generated in `state/keys/http.key`; keep the separate priority-control credential private. Keys are generated locally and never belong in a URL, shell history, repository or published diagnostic.

## Recommended request

```json
{
  "model": "DeepSeek-V4.1-Flash-1017",
  "messages": [{"role": "user", "content": "Review this patch and explain any correctness issues."}],
  "temperature": 1,
  "top_p": 1,
  "top_k": -1,
  "reasoning_effort": "high",
  "max_tokens": 262144,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

Use a smaller output budget where appropriate, and respect the total 1,048,576-token context limit. The output budget also decides whether a conversation can decode in the same steps as others: the whole `max_tokens` counts, not the answer's real length (see [Several conversations at once](#several-conversations-at-once)). Reasoning tokens count toward the output budget. `top_k: -1` means no top-k truncation. Supported temperatures are exactly 0 and 1; top-p must be 1. Unsupported sampling options fail explicitly rather than silently changing the target distribution.

The protocol supports ordinary OpenAI-style tool definitions, tool calls and tool-result messages, along with DeepSeek thinking controls. Images use supported `image_url` message content and the canonical image encoder/processor. The server validates URLs, image dimensions and resource limits. This release is not a complete replacement for every OpenAI API feature: arbitrary temperatures, nucleus cutoffs, penalties and constrained JSON/grammar decoding are not implemented. A requested schema must not be confused with a constrained-decoding guarantee.

Streaming uses SSE and ends with `[DONE]`. One delta may contain multiple speculative tokens. Clients should use reported usage for token counts, not count SSE chunks. Closing an active stream requests cancellation; bounded native work already in flight may finish before it is observed. Output backpressure pauses/resumes native generation, with a bounded stall timeout.

## Prompt processing and Pi timing

Streaming responses carry one top-level `timings` object as soon as the measured prefill phase is available. Nonstreaming responses include the same object. It does not require `stream_options.include_usage`.

```json
{
  "timings": {
    "prompt_n": 249,
    "cache_n": 9088,
    "prompt_ms": 50.0,
    "prompt_per_second": 4980.0,
    "scope": "server_prefill_phase"
  }
}
```

The numbers above are illustrative. `prompt_n` counts uncached prompt tokens; `cache_n` counts reused tokens. Their sum matches reported prompt usage. PP rate is `prompt_n * 1000 / prompt_ms`. This measures the server prefill phase, not network latency, queue time or an isolated GPU kernel timer. Client TTFT remains a different measurement. Cache capacity is bounded and opportunistic; repeat calls are not guaranteed a cache hit under memory pressure.

## Foreground and background

Requests default to foreground. Set `X-LocalAI-Mode: background` for background work, or explicitly use `foreground`. Background requests are peers with each other; foreground requests are peers with each other. Among admitted requests, foreground work pauses background work at safe boundaries; background work resumes with its accepted state intact. This transition can take a few seconds and is not a hard real-time preemption promise.

Up to **four requests are admitted** (two before the November 15 release), each retaining the full 1,048,576-token context capacity. Additional requests wait FIFO. Admitted requests decode in the same steps or take turns, as described under [Several conversations at once](#several-conversations-at-once). An admitted request keeps its slot until its final response, terminal error, client cancellation/disconnection or terminal stall timeout. A temporary output stall, priority hold or pause does not release the slot. A queued foreground request does not displace admitted background requests. A completed tool-call response ends that request; submitting tool results creates a new request. Model state is retained in memory, without SSD parking.

The default has **one physical execution bank**, not two permanently allocated banks. Overlap uses complete, lossless state snapshots at committed boundaries. Solo forward reuse stays in the active bank and does not require a request snapshot. The shared snapshot/cache ceiling is allocated only as needed, and optional completed conversation checkpoints yield to unfinished requests (sizes under [Resident conversations](#resident-conversations)). The 8 GiB operating margin and 8.5 GiB allocation floor remain enforced on the limiting rank. Large same-priority peers may run serially if switching cannot retain both snapshots safely; their lifetime admissions and progress are preserved. With the two-owner scheduler, a foreground newcomer that cannot safely pause an existing background request gets an explicit capacity error rather than silently canceling that owner or waiting indefinitely behind a hold. The batched scheduler of the November 15 release (`CANADA_SCRATCH_BATCHED_DECODE=1`, the supplied profile) does not return that error yet: the newcomer waits for the background request. Overlap throughput and switching latency can be worse than the former permanently allocated two-bank layout.

`CANADA_SCRATCH_SINGLE_BANK=1` selects the qualified layout. Its paired execution, cross-bank prefix import and shared paired-drafter flags are disabled in the supplied profile. Primary append/decode graph variants are fully warmed and retained (`CANADA_SCRATCH_GRAPH_CACHE_ROWS=0`). A nonzero row budget is an optional bounded-retention control, with recapture latency on misses; it is not the release default. Do not independently re-enable conflicting paired flags.

The internal priority-control endpoint uses the separate control key. Ordinary client integrations should use the request header. Do not expose or log the control credential. Background qualification is separate from normal foreground performance measurement.

### Execution timing for routers

A router can leave admission to the engine: it forwards every request at once, and the engine admits up to four and
queues the rest. Such a router needs to know when a request is actually running, so it can count only running time
against a request's time budget, and it needs a way to tell a backend that reports this from one that does not.

- Every successful chat response, streamed or plain, carries the header `x-native-execution-timing: v1`.
- A streamed response also carries SSE comments, `: native-execution running` and `: native-execution waiting`, one
  at each change of state (the first one when the stream opens). A conversation runs while it owns region 0 or a
  serving region. It waits while it is in line (not started yet, or paused for another conversation), while it
  is a background conversation that a router holds, and while its reader is behind on the output. Conversations that take turns report
  each turn.

They are transport metadata: clients may ignore them, and they never appear inside `data:` frames. The final
`timings` object described above is separate and is always sent.

Earlier releases sent both with the two-owner scheduler (`CANADA_SCRATCH_TWO_STREAM_ADMISSION=1`). Release s1585
replaced that scheduler with batched decode and stopped sending them, so a router that requires the header refused
every response; s1586 sends them with batched decode too. `tools/smoke_http.py` checks the header and the first
`running` comment.

## Earlier checkpoints for edited history

Each request leaves a small bank-retained encoder checkpoint at its prompt end, and an edit of history restores
the latest retained checkpoint at or before the edit and prefills the rest. The bounded cache (8 entries, 2 GiB)
used to replace the oldest entry first, so in a long conversation every retained checkpoint sat in the last few
turns, and an edit of older history (an image offload, pruning old tool output) re-prefilled from the start.

`CANADA_SCRATCH_CHECKPOINT_LADDER=1` keeps the conversation's earlier checkpoints spread over its length instead.
When the ordinary policy would evict one of them, it evicts the one whose loss adds the least re-prefill for an
edit at a uniformly random position (the smallest product of its two neighboring gaps). The two latest
checkpoints, protected longer parents, the entry count, the byte budget and every admission check are
unchanged. In a policy model of 150 checkpoints and 8 entries, the mean re-prefill for an edit at a random
position falls from 0.46 to 0.13 of the conversation. It changes no arithmetic: a restore uses the existing
checkpoint path, and a restored suffix may differ from a cold prefill only as any cached restore does, by
floating-point summation order.

## Prefill without unused layer-20 rows

The wide prefill encoder runs layers 0 to 20 over every prompt row, then replays layers 21 to 39 over the
last 128 rows. Layer 20's output is read only for those last 128 rows. Its key/value projection, sliding-window
keys and source-20 compressed entries and index keys are read for every row, by later rows, the decoder replay
and decode.

`CANADA_SCRATCH_PREFILL_LAYER20_DEFER=1` uses this. For an eager full chunk that ends before the request's
final 128 rows, layer 20 runs only its input normalization, Q/KV projection with the sliding-window store, and
the source-20 key emission. It skips candidate selection, attention, the output projection and the expert
layer, about 46 ms per 8K-row chunk. Chunks that hold any of the final rows run unchanged.

If such a chunk ends a turn early (a cancellation, or a scheduler yield to another request), the skipped
operations are issued unchanged on the retained buffers before the decoder replay at that boundary. Every
value that is read later comes from the same kernels in the same order. Prompt logits are bit-identical with
the option off and on.

`CANADA_SCRATCH_PREFILL_ENGRAM_SPLIT14=1` changes only I/O scheduling. The first chunk of a prompt has no
earlier chunk to prefetch its Engram rows behind. By default all of its rows are read while layer 0 runs;
with this option the layer-14 rows are read while layers 1 to 13 run. It shortens a prompt whose Engram rows
are not in the page cache by about 0.15 to 0.18 s, and leaves warm prompts unchanged. Results are
bit-identical.

## Keeping a displaced conversation's checkpoint

Each request keeps a bank-retained encoder checkpoint near its prompt end. That checkpoint stores only its
tail, because the compressed prefix stays in the execution bank. An unrelated request that reuses the bank,
such as a short title or summary request between two turns of an agent session, invalidates it. The
displaced conversation's complete state is still saved, but its next turn re-renders the previous assistant
message without the thinking tokens, so that state does not match. Before this option, the next turn
re-prefilled the whole conversation.

`CANADA_SCRATCH_DISPLACED_CHECKPOINT=1` copies the displaced conversation's newest bank-retained checkpoint
into a standalone image before the incoming request touches the bank. The compressed pages it omitted are
read from the bank while they still belong to that conversation; the rest comes from the checkpoint. The
ordinary cache budget, entry limit and memory reserve apply, and all ranks agree. A refusal keeps the
previous behavior.

In a synthetic screen, a 62K-token conversation's follow-up after an unrelated 36-token request was served
from 61,824 cached tokens in 0.35 s, instead of a 7.0 s cold prefill. Its prompt logits equal, bit for bit,
those of the same follow-up restored from the original bank checkpoint without the interruption.

## Reference cache formats and attention

`CANADA_SCRATCH_KV_FORMAT=v41` stores the target model's attention caches in the formats of DeepSeek's reference
implementation instead of the engine's earlier layout:

| Cache | Earlier layout (`v4`, default) | `v41` |
|---|---|---|
| Sliding window | 448 e4m3 values, 64 BF16 RoPE values, 7 ue8m0 scales: 584 B/token | 512 e4m3 values (RoPE dimensions included), one ue8m0 scale per 32: 528 B/token |
| Compressed entries | as above | 512 e2m1 values, one e4m3 scale per 16: 288 B/token |

The drafter's caches keep the earlier layout. The store kernels reproduce the reference quantizers bit for bit
on GB10. The only exception is a block whose largest magnitude is below 448 × 10^-4 (about 0.045). There the
stored scale has a floor of 2^-13, so the block's elements below 2^-19 can round differently from the reference.

A conversation's cache, and the state image that keeps it resident, need about 56% of their earlier bytes. A
974K-token conversation's image is about 0.91 GB per rank instead of 1.64 GB.

With `v41`, attention runs two kernels:
- **Above 8 rows** (prefill and wide verification): a prefill sparse-attention kernel from
  [pico-dgx-ds4](https://github.com/Picom-code/pico-dgx-ds4) (MIT), adapted to the paged caches, explicit candidate
  lists and the engine's RoPE handling. It keeps the packed keys in shared memory, widens them exactly to f16 in
  registers for the tensor-core MMAs, and rounds P to BF16 before the sum, as the reference does.
- **At 8 rows and below** (decode): FlashInfer 0.7.3's DeepSeek-V4.1 split decode, with fp32 partials.

`CANADA_SCRATCH_PICO_ATTENTION=0` keeps FlashInfer's kernels at every row count; unset or `1` uses the adapted
kernel.

Same-load screen on one GB10, in ms per layer:

| Shape | Earlier layout | FlashInfer DeepSeek-V4.1 | Adapted kernel |
|---|---|---|---|
| 8K-row prefill with compressed entries | 15.67 | 13.86 | 7.68 |
| 8K-row prefill, window only | 3.52 | 2.41 | 1.61 |
| 32-row decode | 0.070 | 0.058 | 0.030 |

Attention is a smaller share of a full prefill than in this screen. On fresh boots, cold source-code prompts took
4-7% less time, and decode is about unchanged. Every output stays within BF16 rounding of a float64 oracle.

## Resident conversations

The session cache keeps finished conversations as state images on the GPU, so a returning conversation is
restored by a device copy instead of a prefill. Two settings size it:
- `CANADA_SCRATCH_SESSION_CACHE_SLOTS`: up to 16 entries (previously 8). Each conversation can hold several
  entries, its newest checkpoint and earlier rungs.
- `CANADA_SCRATCH_SESSION_CACHE_BYTES`: up to 8 GiB (previously 2 GiB). Native production sets 16 slots and
  6 GiB (5 GiB before the November 15 release).

The session cache shares `CANADA_SCRATCH_SHARED_IMAGE_BYTES` with the images of paused requests. The one-bank
layout allows up to 8 GiB there (4 GiB before the November 15 release); native production sets 7 GiB. The cache
can use the smaller of the two budgets, after any paused images, so production's effective cache is 6 GiB.

The budget is a ceiling, not a reservation. Every save is admitted against the measured available memory on every
rank, leaving the memory floor and the next prefill's working room free. When memory is short, older entries are
evicted or the save is refused. Nothing is allocated past the floor.

Entries are evicted least recently used first. A turn that extends its conversation saves a checkpoint near its
end, so the next turn prefills only its new tokens. A client that sends many different follow-ups to one prompt
(regenerations, branches) creates an entry for each of them; once the 16 entries are full they can push out the
entries of a conversation that stayed idle meanwhile, and that conversation is prefilled again when it returns.

With the `v41` formats, four conversations of 543K-605K tokens took turns for three rounds. Their return turns had
a median time to first token of 0.41 s, with 99.99% of prompt tokens served from the cache.

Four conversations of 912K-929K tokens need more than 4 GiB. Under the 4 GiB budget of the earlier release the
two longest stayed resident and returned in about 0.4 s, and the other two were prefilled again on each return,
in about 110 s. With the November 15 release's budgets all four stay resident: over three rounds, eight of eight return
turns were served from the cache, with a median time to first token of 0.38 s, and at least 10.6 GiB stayed
available on every rank.

All memory-floor checks now use one function, with the same 8 GiB default.
`CANADA_SCRATCH_CLAUDE_MEMORY_FLOOR_MIB` overrides the floor and is for tests only.

## November 17 memory release settings

Native production (s1595) keeps every November 15 setting and changes four, all about memory. They are runtime
settings read at startup; the portable profile is unchanged.

- `CANADA_SCRATCH_IMAGE_POOL_MIB=8192`: the snapshot chunk pool. After the startup warm-up every device creates a fixed
  pool of 16 MiB physical chunks (about 0.2 s for 8 GiB). A conversation snapshot (a cached conversation or a paused
  request) maps the chunks it needs into one contiguous range and returns them when it is dropped. Snapshots were
  almost all of the large allocations made while serving (in the stress panel, about a hundred of up to 857 MiB,
  18 GiB allocated and freed in nine minutes). On memory that days of use have fragmented, those allocations made the
  driver fall back from large pages and the kernel reclaim and compact; with the pool, serving makes no large
  allocation and such memory behaves like freshly sorted memory. Snapshots under 8 MiB keep ordinary allocations.
- `CANADA_SCRATCH_MEMORY_FLOOR_MIB=4096`: the admission floor, 4 GiB of available memory instead of the 8 GiB default.
  Accepted only together with the pool (2048-8192 MiB); without the pool the 8 GiB default stands. The test override
  `CANADA_SCRATCH_CLAUDE_MEMORY_FLOOR_MIB` still takes precedence.
- `CANADA_SCRATCH_SESSION_CACHE_BYTES` 6 GiB -> 6.75 GiB and `CANADA_SCRATCH_SHARED_IMAGE_BYTES` 7 GiB -> 7.75 GiB, so
  the cache uses the pool (with the pool, both may be set up to the pool's size).

The runtime also has `CANADA_SCRATCH_RELEASE_LOAD_FILE_CACHE` (after the warm-up, ask the kernel to drop the page
cache of the kernel bundle tree, about 2 GB a device); it is off: in the comparison against production, the first
conversation after a load ran up to a third slower for about a minute in two of three runs with it, and in none
without it.

Eviction stays least recently used. The runtime also has `CANADA_SCRATCH_CACHE_COST_ORDER` (eviction by how much
prefill an entry's loss would cost); it is off, because the qualification found cases where it evicts the entry the
next request needs. The runtime fixes one accounting bug: while a restore has a cached snapshot checked out, the
snapshot's room is now reserved against the cache budget as well as the shared one (before, a save made meanwhile
could take it and the engine stopped).

## November 15 release settings

Native production (s1586; the settings are s1585's) runs the s1399 profile with five settings changed and 58 added, and twelve encoder
widths instead of five. Every one is a runtime setting read at startup; the portable profile is unchanged. They
are grouped here by what they do. Settings that change which rows a kernel reads, in what order, or when work is
launched leave results bit-identical; the ones that change a summation order or the drafter are named as such,
and the release as a whole is judged by the reference comparison in [validation](VALIDATION.md).

### Encoder widths

`--wide-widths` (the base configuration's `wide_widths`) lists the encoder profiles; the planner picks a chunk
plan from them for each prompt or appended turn. Production's five (512, 1024, 2048, 4096, 8192) become twelve:
512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096, 5120, 6656, 8192. A chunk costs about 0.1 ms per row of its
profile's width whatever its live rows, so an appended turn of 1,100 tokens that used to run as a 2,048-row chunk
now runs as a 1,280-row one.

- `CANADA_SCRATCH_FLEX_WIDTHS`: the widths beyond the native ones. Each profile under 8,192 rows holds 0.11-0.28
  GiB of host memory plus its captured graph, which is why the list is twelve wide and not longer.
- `CANADA_SCRATCH_ENCODER_CHUNK_COSTS`: the planner's cost table (seconds per chunk by width), measured Nov 12.
- `CANADA_SCRATCH_PREFILL_PARTIAL_WIDE=3`: a partly filled chunk of the 4,096-row and wider profiles bounds its
  dense projections and routed experts to the live rows.
- `CANADA_SCRATCH_SHARED_WIDE_SOURCES=2` and `CANADA_SCRATCH_WIDE_ARENA_KIB_PER_ROW=224`: the profiles share one
  scratch arena for buffers a chunk rewrites before reading, and the two layer arenas are sized to the measured
  need (213-216 KiB a row) instead of 384.
- `CANADA_SCRATCH_WIDE_STARTUP_WARMUP=1`: every width is run once before Ready, so captures and first-use loads
  do not land in a client's request, and memory after Ready already includes them.

### Checkpoints and the conversation cache

- `CANADA_SCRATCH_SHARED_IMAGE_BYTES` 7 GiB and `CANADA_SCRATCH_SESSION_CACHE_BYTES` 6 GiB (the one-bank layout
  now allows 8 GiB); 16 entries as before, and `CANADA_SCRATCH_SESSION_CACHE_SHORT_RESERVE=4` keeps four of them
  for checkpoints under 8,192 tokens so long conversations cannot starve short ones.
- `CANADA_SCRATCH_PROMPT_CHECKPOINT_MIN=512`: conversations shorter than 8,192 tokens get reusable checkpoints
  too (previously a short chat re-prefilled its whole prompt every turn).
- `CANADA_SCRATCH_SHARED_PREFIX_CHECKPOINT=2`: a checkpoint at the end of a system prompt that several
  conversations share, so a new conversation under the same system prompt restores it.
- `CANADA_SCRATCH_ENCODER_CHECKPOINT_INLINE=1`, `CANADA_SCRATCH_ENCODER_NEAR_END=1`,
  `CANADA_SCRATCH_ENCODER_EXTRA_NEAR_END=1`, `CANADA_SCRATCH_ENCODER_SHORT_SUFFIX=1`: the checkpoint is captured
  inside the last encoder chunk and close to the prompt's end, so the next turn re-prefills only its new rows.
- `CANADA_SCRATCH_DISPLACED_RECLAIM=1`: when the conversation an incoming prompt displaces cannot be copied out
  for lack of room, older inactive images give way first instead of the displaced one being dropped.
- `CANADA_SCRATCH_RESET_KEEP_CACHES=1`, `CANADA_SCRATCH_SKIP_REPLACED_TURN=1`: a reset no longer zeroes the
  position-indexed caches (12-16 ms per cold request), and a turn that replaces a completed output skips the
  copy of that output.
- `CANADA_SCRATCH_PROMPT_PREFIX_REUSE=1` (front end): a long message that differs from a cached one only after a
  long common prefix keeps the cached token ids up to a pre-token boundary and tokenizes the rest. Two guards
  check every use; a refused splice is tokenized whole.
- `CANADA_SCRATCH_EARLY_FIRST_TOKEN=1`, `CANADA_SCRATCH_EARLY_FIRST_TOKEN_APPEND=1`: the first token is sent as
  soon as every rank agrees on it, before the drafter attaches or the prompt image is saved.

A pause (a request taking over from a conversation that cannot move to a region) is admitted against the
allocator's own reserve, the memory floor plus 512 MiB, and releases completed cache against measured memory
before it gives up. Earlier builds admitted it against the bare floor: with available memory inside that band the
image allocation failed inside the allocator and stopped the engine. The scheduler's `park` record in
`/_native/diagnostics` names the stage and the measured and required bytes when a pause is refused. When a paused
request resumes, the conversation it replaces keeps its newest checkpoints as standalone images (and its complete
image if it had finished), so its next turn still starts from its prompt; the `resume` record lists what was kept.

### Decode

- `CANADA_SCRATCH_CAUSAL_LADDER=0.15,0.2,0.15`: a step verifies two to six rows, chosen from the running product
  of the drafter's confidences (previously three or six). A causal stopping rule: the sampled distribution is
  the target's.
- `CANADA_SCRATCH_L2_QUEUE`, `CANADA_SCRATCH_L2_QUEUE16`, `CANADA_SCRATCH_L2_QUEUE_INDEX_LEAD=1`,
  `CANADA_SCRATCH_DRAFT_HINTS`: paced L2 prefetch hints for the weights a layer is about to read. Hints only.
- `CANADA_SCRATCH_MHC_STATS_LATE=15`, `CANADA_SCRATCH_MHC_STATS_LATE16=05`, `CANADA_SCRATCH_DRAFT_STATS_OVERLAP=1`:
  where the mixing statistics of a boundary are released to their side stream. Same kernels and inputs.
- `CANADA_SCRATCH_MOE_NATIVE_FRONT=1`: the routed experts' front end without its four software grid barriers.
  Bit-identical tables and quantized input.
- `CANADA_SCRATCH_QKVA_DIRECT=1`: the six-row owner's QKV-A product in a direct-read kernel. Another summation
  order (covered by the reference comparison).
- `CANADA_SCRATCH_INDEX_SHORT_ROWS=1`, `CANADA_SCRATCH_BOUNDED_CANDIDATE_MASK_DECODE=1`,
  `CANADA_SCRATCH_SPARSE_UNION_SHARE=1`: index work that the selector's own rules make redundant is skipped
  (rows with at most 512 keys; mask columns past the live keys; a candidate union already built by layer 24).
- `CANADA_SCRATCH_ENGRAM_DIRECT=1`: a table row that misses the row cache is read with `O_DIRECT` instead of
  through the page cache (same bytes; no page-cache allocation near the memory floor).
- `CANADA_SCRATCH_DRAFT_HEAD_NVFP4=1`, `CANADA_SCRATCH_DRAFT_HEAD_SUBSET=1`, `CANADA_SCRATCH_DRAFT_HEAD_POOL=1`,
  `CANADA_SCRATCH_PAIRED_DRAFT_HEAD=1`, `CANADA_SCRATCH_SHARED_DRAFT_HEAD=1`: the drafter's vocabulary head in
  4 bits, over a script subset (61 % of the vocabulary), pooled across conversations. Only the proposal
  distribution changes; every draft is verified against the unchanged target, so the sampled distribution does
  not depend on the drafter.

### Several conversations at once

`CANADA_SCRATCH_BATCHED_DECODE=1` with `CANADA_SCRATCH_BATCHED_REGIONS=4`: up to four conversations decode in the
same steps (one verification over every conversation's rows), each in a serving region of 262,144 positions.

**Which conversations share steps.** One whose prompt plus its token limit (`max_tokens`; 262,144 when the request
gives none) fits a region, with 64 positions of margin, and which has no image in its history. The limit counts
in full, whatever the answer turns out to be: a client that sends a large `max_tokens` on every request (many
agent harnesses send the model's maximum) gets no shared steps. Everything else takes turns in region 0 through
paused images, 64 tokens at a time, and all of them stay resident. A conversation in its first run yields after
16 tokens to one that has not started, so a newcomer's prefill starts at once, as it did with the two-owner
scheduler. At most two conversations are paused at once:
two such conversations alternate, and when a third or a fourth arrives while two are paused, it runs to its end
before a paused one resumes (the two-owner scheduler of earlier releases ran them as FIFO pairs: two alternate,
the others wait for their first token). Total time is the same or shorter; an early conversation can finish later.
While a region is occupied, a waiting conversation that does not fit waits, first in line, until the regions are
empty: region 0 has no upper bound of its own in the address space it shares with the regions, so a longer
conversation beside them would write over their rows. To get shared steps, send a `max_tokens` that leaves the
prompt room inside a region (prompt + `max_tokens` + 64 <= 262,144).

A conversation that moves to a region takes region 0's drafter along, and a spare takes its place; a paused or
cached conversation's drafter state restores into whichever drafter region 0 holds. The related settings (`CANADA_SCRATCH_BATCHED_GRAPHS`, `..._GRAPH_WARMUP`, `..._DRAFT_BODY`,
`..._DRAFT_FIXED_GROUPS`, `..._DRAFT_INTERLEAVE`, `..._POOLED_HEAD`, `..._ENGRAM_OVERLAP`, `..._PADDING_ROUTES`,
`..._PAIR_ROWS=16`, `..._BLOCK_CAP=6`, `..._GATE`, `..._DETACH_KEEPS_IMAGES`, `CANADA_SCRATCH_REGION_GATHER_CACHE`,
`CANADA_SCRATCH_REGION_GATHER_BY_QUERY`) size and schedule those steps.

Every batched pattern is captured before Ready (`CANADA_SCRATCH_BATCHED_GRAPH_WARMUP=1`), in the driver's own
memory. `CANADA_SCRATCH_BATCHED_TIGHT_BLOCKS` stays 0 in this release: tight blocks raise the pattern count from
15 to 43, and leaving them off kept about 1.0 GiB available on every machine (measured), for about 2 % at two
conversations sharing steps.

### Option, off by default: 4-bit output head

`CANADA_SCRATCH_HEAD4_RESCORE=<cap>[,<window>]` makes the model's own output head read a 4-bit copy of its
weights and rescore the likely tokens exactly. It changes the distribution a token is sampled from, slightly and
by design (at 512,24: about one draw in 400,000-660,000 differs), for 0.4-0.9 ms a step. It is in no profile;
`src/head4_rescore.rs` documents the difference and every measurement.

### Diagnostics that stay on

`CANADA_SCRATCH_CLAUDE_DECODE_SUMMARY=1` (one line of step timing per request on standard error),
`CANADA_SCRATCH_WEIGHT_READ_CENSUS=1` (read rates of the loaded weights at startup) and
`CANADA_SCRATCH_ATLAS_CACHE_RELEASE=1` (the weight files' page cache is released as each file is verified on the
device, so later layers are not placed across both halves of physical memory). They write counts and timings only.

## Lifecycle and diagnostics

`deploy/fleet.py` starts exactly one four-rank cohort with a fresh private ticket, validates the selected nodes and runtime files, and monitors health. Containers use Docker's init process for signal forwarding and reaping. Any failed rank causes coordinated cleanup of the recorded cohort. The controller checks project/generation ownership before stopping all four ranks concurrently, allowing up to 90 seconds for each Docker stop to finish GPU teardown. Docker sends SIGTERM and may escalate to SIGKILL after its 15-second grace period; stopping the service is not a promise to finish in-flight requests or preserve RAM caches. Systemd can restart a failed controller, with a bounded start limit. An unreachable worker can prevent safe complete cleanup; repair SSH and retry instead of starting a conflicting cohort or weakening ownership checks.

Use `systemctl stop titanic-flash.service` for a systemd-managed installation. The standalone `fleet.py stop` action also handles an abandoned cohort when no supervisor owns the lifecycle lock. Inspect `fleet.py status`, Docker state and available memory before restarting after a failure. A process being alive is not equivalent to model Ready.

Worker event diagnostics default to bounded **RAM** history. The portable launcher uses Docker's `none` log driver to avoid persistent per-token worker logs. Lifecycle state writes occur at transitions; model requests and generated text are not written by this wrapper. OS, SSH and systemd journals can still write ordinary operational logs. Authenticated `/_native/diagnostics` returns bounded diagnostic state with `Cache-Control: no-store`; save only deliberate, redacted incident evidence. A failed early startup may require a deliberate temporary bounded-log diagnostic run.

A refused CUDA device, pinned-host or compressible allocation writes one `allocation_failure` JSON line to the worker's standard error: the size, the call site, CUDA's free and total memory, and the host's `MemFree`, `MemAvailable`, `Cached` and `SReclaimable`. Nothing is written otherwise. The line is kept only where the container's log driver keeps output; the portable launcher's `none` driver does not. After long uptime or many model loads, the kernel log can show `NVRM: ... NV_ERR_NO_MEMORY ... _memdescAllocInternal(pMemDesc) @ mem_desc.c:1359` with no matching `allocation_failure` line. The driver logs each failed large-page attempt at error level and then allocates with smaller pages; these lines need no action.

Do not modify `deploy/profile.json` casually: it is the accumulated qualified model profile. Some internal environment variables retain a legacy `CANADA_SCRATCH_` prefix as an implementation compatibility detail. Diagnostic CLI commands and the `storage-fault-injection` feature are not part of the production build.

The inventory explicitly selects the temporary host policies: a 2350 MHz GPU cap or `0` to leave clocks unmanaged, and optional `/dev/nvme0` APST disablement per node. The clock helper assumes stock clocks as its restoration baseline; do not use it over an unrelated administrator's clock lock. The NVMe helper saves and restores the exact original feature value and table. Independent systemd timers restore settings if short leases stop renewing. Delayed checks from an older generation cannot overwrite a newer owner's policy. Healthy renewals remain in RAM; only activation, failure and restoration transitions write durable records.

A transient management failure appears as `host_policy_status: degraded`; it does not immediately discard a healthy model. Local timers can restore host settings while the controller retries, so performance during that degraded interval is not comparable to a fully active fixed-clock run. Check policy status alongside Ready when collecting benchmarks. Failure during initial policy arming prevents model launch and attempts restoration on every node.

The launcher leaves firmware, networking, other model services and external routers under the operator's control. Store rollback versions outside the active runtime. Keep model weights and Engram backing on local storage; moving cold build outputs or archived versions is independent of moving hot inference data.
