# Prefill and vision options (s1387)

s1387 adds long-prefill and vision optimizations as runtime options. Each is selected by an
environment variable read at worker construction; all default to off in the code. The native
production deployment and the portable profile in `deploy/profile.json` enable the set below,
and the published s1387 measurements use it.

The options marked **derived matrices** load full projection matrices that
`tools/stage_full_matrices.py` builds from the staged weight atlases (see
[setup](SETUP.md#5-build-lossless-engram-backing-and-gpu-weight-atlases)); their dense FP8 GEMM
operators and the split vision attention kernel are in the checked kernel bundle. Without the
derived files, the controller refuses to start and a worker stops at load rather than changing
outputs.

| Variable (`=1`) | What it does | Numerics | Needs |
|---|---|---|---|
| `CANADA_SCRATCH_WOA_TOKEN_MAJOR` | Per-group WO-A GEMMs write token-major output on wide prefill profiles | bit-identical | — |
| `CANADA_SCRATCH_PREFILL_Q_INPLACE` | RoPE in place on the Q-B output (no full-width Q copy) | bit-identical | — |
| `CANADA_SCRATCH_PREFILL_TAIL_WINDOW` | Gather only the 128-row encoder-tail window | bit-identical | — |
| `CANADA_SCRATCH_PREFILL_ENGRAM_BEHIND_L0` | Hash and stage a chunk's Engram rows on the host while the GPU runs the embedding and layer 0 | bit-identical | — |
| `CANADA_SCRATCH_VISION_LT_BIAS` | cuBLASLt bias epilogue in the vision tower | bit-identical image spans | — |
| `CANADA_SCRATCH_PREFILL_SP_EARLY` | Sequence-parallel layers 0, 1 and 14 with row-local Engram (full WKV on local rows) | BF16 reduction order | derived matrices |
| `CANADA_SCRATCH_PREFILL_LOCAL_SHARED` (+`_OVERLAP`, `_LATE`) | Shared expert on each rank's own rows, overlapped with the routed reduce-scatter (≥2048-row profiles) | BF16 reduction order | derived matrices |
| `CANADA_SCRATCH_PREFILL_WOB8` (+`_SUM4`) | Exchange MXFP8 WO-B inputs by row quarter; each rank applies all four WO-B K shards and sums them in the reduce-scatter's order (one fused kernel with `_SUM4`) | bit-identical | derived matrices |
| `CANADA_SCRATCH_VISION_SPLIT` | Row-split and image-parallel vision encoding with a retiled vision attention kernel | bit-identical image spans | split attention kernel; vision atlas on every rank |

Both profiles also set `CANADA_SCRATCH_IMAGE_FUTURE_PINS=1`,
`CANADA_SCRATCH_IMAGE_METADATA_DEFER=1` and `CANADA_SCRATCH_VISION_NORM_MODE=fused` from the
qualified image line, and `CANADA_SCRATCH_GRAPH_MEMORY_SCREEN=0`, which keeps a graph-capture
memory diagnostic at its default (off).

**Derived matrices.** Full (unsharded) copies of the encoder layers' WO-B (layers 0–20), the
shared expert (layers 0–20) and the two Engram WKV projections (layers 1 and 14), assembled in rank
order from the qualified per-rank weight shards: FP8 E4M3 values and canonical 32×32 UE8M0 scales,
copied without conversion. Each rank verifies every file's digest and that its own part equals its
qualified shard at load. Together with their dense FP8 GEMM operators (with pipeline-fence
variants, in the bundle) they add roughly 3.5–4 GiB of resident memory per rank and about 1.9 GB of
disk per rank. They are derived from the checkpoint, so they are not distributed with this
repository.

**Upgrading an s1345 portable installation.** Stop the service, update the checkout, rebuild and
verify the runtime (`build.sh`, `prepare_runtime.py`, `verify_runtime.py`), run
`stage_full_matrices.py` on the head, then `sync_fleet.py`, and start the controller. The staged
atlases and Engram backing are unchanged.

**Memory.** With these options the s1345 stress panel's lowest available memory fell from 14.64 GiB
to 11.00 GiB on the two middle ranks; the 8 GiB operating and 8.5 GiB allocation floors and the
4 GiB shared snapshot ceiling are unchanged. See [validation](VALIDATION.md).

**Fabric note.** The qualification fleet uses two independent 100 Gb/s switched rails per node
(200 Gb/s aggregate, the per-system ceiling). Every pair of ranks is one switch hop on both rails,
which suits the WO-B row-quarter exchange.
