// Index selection with the reference's bf16 arithmetic (DeepSeek V4 model.py: the indexer's einsum, relu, mul by
// the head weights and sum over heads, each in bf16), as one score pass writing 16-bit order-preserving keys and
// one selection pass per row (exact top-k, ties at the cut by ascending position, output ascending).
//
// Adapted from pico's engine/attn/indexer.cu (repaired copy of f211791; MIT, Copyright (c) 2026 Andy Shu, see
// licenses/pico-dgx-ds4-MIT.txt): mma_fp4, key_from_bf16, load_tile / score_tile_row (score_kernel) and
// block_excl_scan / threshold_key / select_row / select_bins (plain path). Changes for this engine:
//   - queries and keys are in this engine's plain layouts: QueryRows payload [row * 32 + head][64 B] e2m1 with one
//     u32 of four ue8m0 scales and one f32 head weight per (row, head); KeyMatrix [key][64 B] e2m1 with one u32 of
//     scales per key. A lane's MMA words t, 4 + t, 8 + t, 12 + t are read with four 4-byte loads (pico stores
//     them transposed for one 16-byte load);
//   - each row reads the keys [start, start + length) of the key matrix (its key range: a serving region's rows
//     start past region 0's), scores them into its score row at (key - start) and selects as offset + (key - start),
//     the layout of this engine's topk windows (relative scores, rebased output);
//   - no block lists, candidate lists or block maxima yet: the dense source layers select here; layers with a
//     candidate publisher or consumer take the scores as f32 values and keep their f32 path from there.
//
// Built as its own cubin for sm_121a (the block-scaled FP4 MMA is architecture-specific; core.cu is built for
// sm_121). Arithmetic is explicit (__fadd_rn / __fmul_rn / bf16 rounding), so --fmad does not change it.
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int kHeads = 32;
constexpr int kScoreWarps = 4, kScoreTiles = 4, kScorePos = kScoreWarps * kScoreTiles * 16;  // 256 keys a CTA
constexpr int kScoreRows = 8;                                                                // rows a CTA
constexpr int kSelThreads = 1024, kSelPer = 16, kSelChunk = kSelThreads * kSelPer;

__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float bf2f(uint32_t b) { return __uint_as_float(b << 16); }
__device__ __forceinline__ float bfround(float f) { return bf2f(f2bf(f)); }

// D += A (16x64 e2m1, row-major) * B (64x8 e2m1) with one ue8m0 scale per 32 of K (lane 4g+t: A rows g, g+8;
// B column g; C (g|g+8, 2t|2t+1)).
__device__ __forceinline__ void mma_fp4(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
  asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

// bf16 bits -> an unsigned key with the same order (0 stays free for "no key": it sits below -inf)
__device__ __forceinline__ uint32_t key_from_bf16(uint32_t b) { return (b & 0x8000) ? (~b & 0xFFFF) : (b | 0x8000); }

// Words t, 4 + t, 8 + t, 12 + t of a plain 64-byte e2m1 vector: lane t's operands for both K steps.
__device__ __forceinline__ uint4 words_of(const uint32_t* v, int t) {
  return make_uint4(v[t], v[4 + t], v[8 + t], v[12 + t]);
}
__device__ __forceinline__ uint4 ldg_words_of(const uint32_t* v, int t) {
  return make_uint4(__ldg(v + t), __ldg(v + 4 + t), __ldg(v + 8 + t), __ldg(v + 12 + t));
}

// exclusive prefix over the 1024 threads; *total gets the sum. tmp: 64 words.
__device__ __forceinline__ uint32_t block_excl_scan(uint32_t v, uint32_t* tmp, uint32_t* total) {
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  uint32_t x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  __syncthreads();
  if (lane == 31) tmp[w] = x;
  __syncthreads();
  if (w == 0) {
    uint32_t s = tmp[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      uint32_t y = __shfl_up_sync(0xffffffffu, s, o);
      if (lane >= o) s += y;
    }
    tmp[32 + lane] = s;
  }
  __syncthreads();
  *total = tmp[63];
  return (w ? tmp[32 + w - 1] : 0) + x - v;
}

// The 16 keys of absolute columns c0 .. c0 + 16 (c0 a multiple of 16), 0 outside [lo, hi).
__device__ __forceinline__ void load16(const uint16_t* row, int c0, int lo, int hi, uint32_t (&k)[16]) {
  if (c0 >= lo && c0 + 16 <= hi) {
    const uint4* p = reinterpret_cast<const uint4*>(row + c0);
    uint4 a = p[0], b = p[1];
    const uint32_t wd[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      k[2 * j] = wd[j] & 0xffff;
      k[2 * j + 1] = wd[j] >> 16;
    }
  } else {
#pragma unroll
    for (int j = 0; j < 16; ++j) k[j] = (c0 + j >= lo && c0 + j < hi) ? (uint32_t)row[c0 + j] : 0u;
  }
}

// The exact threshold key inside bin `bin`, from its histogram h2 of key & 31 (warp 0): info[2] gets the K-th
// key, info[3] how many keys equal to it to take; `above` keys lie above the bin.
__device__ __forceinline__ void threshold_key(const uint32_t* h2, uint32_t bin, uint32_t above, uint32_t K,
                                              uint32_t* info) {
  int tid = threadIdx.x;
  if (tid < 32) {  // lane j looks at sub-bin 31-j, counting from the top
    uint32_t c = h2[31 - tid], incl = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += y;
    }
    uint32_t excl = incl - c;
    if (above + excl < K && K <= above + incl) {
      info[2] = (bin << 5) | (31 - tid);
      info[3] = K - (above + excl);
    }
  }
  __syncthreads();
}

}  // namespace

// score[r][c] = bf16(sum_h bf16(relu(bf16(q[r,h] . k[c])) * w[r,h])) for the keys c of row r's range [start,
// start + length) (absolute key rows, inside `columns`), written as a 16-bit order-preserving key (key_from_bf16) at
// keys[r * keys_stride + (c - start)], or, with fscores, as its f32 value at fscores[r * fstride + (c - start)] (for
// layers whose candidate publisher or consumer mask reads f32 scores before topk-v2: its rule, higher value first and
// equal values by lower column, is the same selection); other entries are not written. Grid (any x, ceil(rows / 8)), 128 threads: CTA x scores the 256-column tiles x, x + gridDim.x, ...
// up to the group's last window end, so one captured launch serves every context length. The fp4 dot products
// are exact in fp32 unless a vector's block scales spread over more than ~2^11; the head sum is fp32 in pico's
// order.
extern "C" __global__ void __launch_bounds__(128)
    ca_idx16_score(const uint8_t* __restrict__ q, const uint32_t* __restrict__ qsc, const float* __restrict__ w,
                   const uint8_t* __restrict__ k, const uint32_t* __restrict__ ksc, const int* __restrict__ starts,
                   const int* __restrict__ lengths, int rows, int columns, uint16_t* __restrict__ keys,
                   int keys_stride, float* __restrict__ fscores, int fstride) {
  __shared__ __align__(16) uint32_t sq[kScoreRows][kHeads][16];
  __shared__ uint32_t ssc[kScoreRows][kHeads];
  __shared__ float sw[kScoreRows][kHeads];
  __shared__ int slo[kScoreRows], shi[kScoreRows], sspan[2];
  const int r0 = blockIdx.y * kScoreRows, nr = min(kScoreRows, rows - r0);
  if (nr <= 0) return;
  if (threadIdx.x < nr) {
    const int s = starts[r0 + threadIdx.x];
    slo[threadIdx.x] = s;
    shi[threadIdx.x] = min(s + lengths[r0 + threadIdx.x], columns);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int lo = columns, hi = 0;
    for (int rr = 0; rr < nr; ++rr)
      if (shi[rr] > slo[rr]) lo = min(lo, slo[rr]), hi = max(hi, shi[rr]);
    sspan[0] = lo & ~(kScorePos - 1);
    sspan[1] = hi;
  }
  __syncthreads();
  const int first = sspan[0] + blockIdx.x * kScorePos, span_hi = sspan[1];
  if (first >= span_hi) return;  // CTA-uniform: no column of this group's windows for this CTA
  const uint4* qsrc = reinterpret_cast<const uint4*>(q + (size_t)r0 * kHeads * 64);
  for (int i = threadIdx.x; i < nr * kHeads * 4; i += blockDim.x) reinterpret_cast<uint4*>(&sq[0][0][0])[i] = qsrc[i];
  for (int i = threadIdx.x; i < nr * kHeads; i += blockDim.x) {
    (&ssc[0][0])[i] = qsc[(size_t)r0 * kHeads + i];
    (&sw[0][0])[i] = w[(size_t)r0 * kHeads + i];
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  for (int base = first; base < span_hi; base += gridDim.x * kScorePos) {
    for (int tile = 0; tile < kScoreTiles; ++tile) {
      const int i0 = base + (warp * kScoreTiles + tile) * 16;
      if (i0 >= span_hi) break;  // warp-uniform
      bool any = false;
      for (int rr = 0; rr < nr; ++rr) any |= i0 < shi[rr] && i0 + 16 > slo[rr];
      if (!any) continue;  // warp-uniform
      const int p0 = i0 + g, p1 = i0 + 8 + g;
      uint4 k0 = make_uint4(0, 0, 0, 0), k1 = k0;
      uint32_t s0 = 0x7f7f7f7fu, s1 = 0x7f7f7f7fu;
      if (p0 < span_hi) {
        k0 = ldg_words_of(reinterpret_cast<const uint32_t*>(k + (size_t)p0 * 64), t);
        s0 = __ldg(ksc + p0);
      }
      if (p1 < span_hi) {
        k1 = ldg_words_of(reinterpret_cast<const uint32_t*>(k + (size_t)p1 * 64), t);
        s1 = __ldg(ksc + p1);
      }
      const uint32_t sa = (t & 1) ? s1 : s0;
      for (int rr = 0; rr < nr; ++rr) {
        const int lo = slo[rr], hi = shi[rr];
        if (i0 >= hi || i0 + 16 <= lo) continue;  // warp-uniform
        float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int h = nt * 8 + g;
          const uint4 qv = words_of(sq[rr][h], t);
          const uint32_t qs = ssc[rr][h];
          float d[4] = {0.f, 0.f, 0.f, 0.f};
          mma_fp4(d, k0.x, k1.x, k0.y, k1.y, qv.x, qv.y, sa & 0xffff, qs & 0xffff);
          mma_fp4(d, k0.z, k1.z, k0.w, k1.w, qv.z, qv.w, sa >> 16, qs >> 16);
          const float w0 = sw[rr][nt * 8 + 2 * t], w1 = sw[rr][nt * 8 + 2 * t + 1];
          acc0 = __fadd_rn(acc0, bfround(__fmul_rn(fmaxf(bfround(d[0]), 0.f), w0)));
          acc0 = __fadd_rn(acc0, bfround(__fmul_rn(fmaxf(bfround(d[1]), 0.f), w1)));
          acc1 = __fadd_rn(acc1, bfround(__fmul_rn(fmaxf(bfround(d[2]), 0.f), w0)));
          acc1 = __fadd_rn(acc1, bfround(__fmul_rn(fmaxf(bfround(d[3]), 0.f), w1)));
        }
        acc0 = __fadd_rn(acc0, __shfl_xor_sync(0xffffffffu, acc0, 1));
        acc0 = __fadd_rn(acc0, __shfl_xor_sync(0xffffffffu, acc0, 2));
        acc1 = __fadd_rn(acc1, __shfl_xor_sync(0xffffffffu, acc1, 1));
        acc1 = __fadd_rn(acc1, __shfl_xor_sync(0xffffffffu, acc1, 2));
        if (fscores) {
          float* frow = fscores + (size_t)(r0 + rr) * fstride - lo;
          if (t == 0 && p0 >= lo && p0 < hi) frow[p0] = bfround(acc0);
          if (t == 1 && p1 >= lo && p1 < hi) frow[p1] = bfround(acc1);
        } else {
          uint16_t* krow = keys + (size_t)(r0 + rr) * keys_stride - lo;
          if (t == 0 && p0 >= lo && p0 < hi) krow[p0] = (uint16_t)key_from_bf16(f2bf(acc0));
          if (t == 1 && p1 >= lo && p1 < hi) krow[p1] = (uint16_t)key_from_bf16(f2bf(acc1));
        }
      }
    }
  }
}

// Row blockIdx.x's selection over its lengths[r] keys (relative to its range start, as ca_idx16_score writes them) by
// 1024 threads: a 2048-bin histogram of key >> 5 finds the bin holding the k-th key, a 32-bin histogram inside it the
// exact threshold key, and an ordered pass writes the keys above it plus the first `need` keys equal to it in
// ascending order, as offset + j; then -1 up to kmax. Rows of at most kmax keys select every key, in order.
// keys_stride a multiple of 16. `starts` is not read (kept for the launch's layout).
extern "C" __global__ void __launch_bounds__(kSelThreads)
    ca_idx16_select(const uint16_t* __restrict__ keys, int keys_stride, const int* __restrict__ starts,
                    const int* __restrict__ lengths, const int* __restrict__ offsets, int kmax, int* __restrict__ out,
                    int out_stride) {
  __shared__ uint32_t hist[2048];
  __shared__ uint32_t h2[32];
  __shared__ uint32_t tmp[64];
  __shared__ uint32_t info[5];  // threshold bin, keys above it, threshold key, ties to take, keys in the bin
  const int r = blockIdx.x, tid = threadIdx.x;
  const uint16_t* row = keys + (size_t)r * keys_stride;
  const int lo = 0, hi = lengths[r], off = offsets[r];  // relative positions (ca_idx16_score's layout)
  const int a0 = 0;
  (void)starts;
  hist[tid] = 0;
  hist[tid + 1024] = 0;
  if (tid < 32) h2[tid] = 0;
  __syncthreads();
  for (int c = a0; c < hi; c += kSelChunk) {
    uint32_t k[16];
    load16(row, c + tid * kSelPer, lo, hi, k);
#pragma unroll
    for (int j = 0; j < 16; ++j)
      if (k[j]) atomicAdd(&hist[k[j] >> 5], 1u);
  }
  __syncthreads();
  const uint32_t top = hist[2047 - 2 * tid], next = hist[2046 - 2 * tid];
  uint32_t total, pre = block_excl_scan(top + next, tmp, &total);
  const uint32_t K = min((uint32_t)kmax, total);
  if (K > 0) {
    if (pre < K && K <= pre + top) {
      info[0] = 2047 - 2 * tid;
      info[1] = pre;
      info[4] = top;
    } else if (pre + top < K && K <= pre + top + next) {
      info[0] = 2046 - 2 * tid;
      info[1] = pre + top;
      info[4] = next;
    }
  }
  __syncthreads();
  int* opos = out + (size_t)r * out_stride;
  if (K > 0) {
    const uint32_t bin = info[0], above = info[1];
    for (int c = a0; c < hi; c += kSelChunk) {
      uint32_t k[16];
      load16(row, c + tid * kSelPer, lo, hi, k);
#pragma unroll
      for (int j = 0; j < 16; ++j)
        if (k[j] && (k[j] >> 5) == bin) atomicAdd(&h2[k[j] & 31], 1u);
    }
    __syncthreads();
    threshold_key(h2, bin, above, K, info);
    const uint32_t theta = info[2], need = info[3];
    uint32_t sel_base = 0, tie_base = 0;
    for (int c = a0; c < hi; c += kSelChunk) {
      const int c0 = c + tid * kSelPer;
      uint32_t k[16];
      load16(row, c0, lo, hi, k);
      uint32_t gt = 0, eq = 0;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        gt += k[j] > theta;
        eq += k[j] == theta;
      }
      uint32_t tot, p = block_excl_scan(gt << 16 | eq, tmp, &tot);
      uint32_t tie = tie_base + (p & 0xffff);
      uint32_t o = sel_base + (p >> 16) + min(tie, need);
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const bool take = k[j] > theta || (k[j] == theta && tie++ < need);
        if (take) opos[o++] = off + (c0 + j - lo);
      }
      sel_base += tot >> 16;
      tie_base += tot & 0xffff;
    }
  }
  for (int i = K + tid; i < kmax; i += kSelThreads) opos[i] = -1;
}

// ---- the fused dense path (step 2, Nov 18): one launch scores every row's window and selects it --------------
// Adapted from pico's index_dense_kernel (engine/attn/indexer.cu, repaired f211791; MIT, Copyright (c) 2026 Andy
// Shu): a persistent grid of up to one CTA per SM takes the windows' union span in chunks of 512 keys (a 16-key tile
// a warp), loads a chunk's keys once and scores them for every row (up to kDenseRows), writes each row's 16-bit keys
// and counts them in a per-row histogram of key >> 5 in shared memory (packed u16, at most 65,535 keys a CTA a row),
// then adds those to global histograms. The last R CTAs to finish (a grid ticket) select one row each from its global
// histogram: one ordered pass gathers the keys in or above the threshold bin into shared memory (an ordered compaction
// by prefix scans, so columns stay ascending) when they fit, then the exact threshold and the output come from there;
// otherwise step 1's two passes over the row. Same scores and the same selection as ca_idx16_score + ca_idx16_select.
namespace {
constexpr int kDenseRows = 8, kDensePos = (kSelThreads / 32) * 16;  // 512 keys a chunk
constexpr int kDenseCand = 6144;                                    // candidates gathered in shared memory
// dynamic shared memory: queries, scales, weights and histograms of kDenseRows rows; reused for the candidates
constexpr int kDenseSmemScore = kDenseRows * kHeads * (64 + 4 + 4) + kDenseRows * 1024 * 4;
constexpr int kDenseSmemSelect = kDenseCand * (2 + 4);
constexpr int kDenseSmem = kDenseSmemScore > kDenseSmemSelect ? kDenseSmemScore : kDenseSmemSelect;

__device__ __forceinline__ void hist_add(uint32_t* h, uint32_t key) { atomicAdd(h + (key >> 6), (key >> 5 & 1) ? 0x10000u : 1u); }
__device__ __forceinline__ void hist_flush(uint32_t w, uint32_t* g) {
  if (w & 0xffff) atomicAdd(g, w & 0xffff);
  if (w >> 16) atomicAdd(g + 1, w >> 16);
}
__device__ __forceinline__ uint32_t ld_acquire_gpu(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// 16 keys of the row (L2: other CTAs of this launch wrote them; L1 is not coherent), 0 outside [lo, hi)
__device__ __forceinline__ void load16_l2(const uint16_t* row, int c0, int lo, int hi, uint32_t (&k)[16]) {
  if (c0 >= lo && c0 + 16 <= hi) {
    const uint4* p = reinterpret_cast<const uint4*>(row + c0);
    const uint4 a = __ldcg(p), b = __ldcg(p + 1);
    const uint32_t wd[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      k[2 * j] = wd[j] & 0xffff;
      k[2 * j + 1] = wd[j] >> 16;
    }
  } else {
#pragma unroll
    for (int j = 0; j < 16; ++j) k[j] = (c0 + j >= lo && c0 + j < hi) ? (uint32_t)__ldcg(row + c0 + j) : 0u;
  }
}
// Row r's selection by the CTA from its global histogram gh (left zero): see ca_idx16_dense.
__device__ void select_dense_row(const uint16_t* row, uint32_t* gh, int lo, int hi, int off, int kmax, int* opos,
                                 uint16_t* ckey, uint32_t* cidx, uint32_t* h2, uint32_t* tmp, uint32_t* info) {
  const int tid = threadIdx.x, a0 = lo & ~15;
  const uint32_t top = __ldcg(gh + 2047 - 2 * tid), next = __ldcg(gh + 2046 - 2 * tid);
  gh[2047 - 2 * tid] = 0;
  gh[2046 - 2 * tid] = 0;
  if (tid < 32) h2[tid] = 0;
  uint32_t total, pre = block_excl_scan(top + next, tmp, &total);
  const uint32_t K = min((uint32_t)kmax, total);
  if (K > 0) {
    if (pre < K && K <= pre + top) {
      info[0] = 2047 - 2 * tid;
      info[1] = pre;
      info[4] = pre + top;  // keys in or above the bin
    } else if (pre + top < K && K <= pre + top + next) {
      info[0] = 2046 - 2 * tid;
      info[1] = pre + top;
      info[4] = pre + top + next;
    }
  }
  __syncthreads();
  if (K > 0) {
    const uint32_t bin = info[0], above = info[1], cand = info[4];
    if (cand <= (uint32_t)kDenseCand) {
      // one ordered pass: every key in or above the bin, columns ascending
      uint32_t base = 0;
      for (int c = a0; c < hi; c += kSelChunk) {
        const int c0 = c + tid * kSelPer;
        uint32_t k[16];
        load16_l2(row, c0, lo, hi, k);
        uint32_t m = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) m += k[j] && (k[j] >> 5) >= bin;
        uint32_t tot, p = block_excl_scan(m, tmp, &tot);
        uint32_t o = base + p;
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (k[j] && (k[j] >> 5) >= bin) {
            ckey[o] = (uint16_t)k[j];
            cidx[o] = (uint32_t)(c0 + j);
            if ((k[j] >> 5) == bin) atomicAdd(&h2[k[j] & 31], 1u);
            ++o;
          }
        base += tot;
      }
      __syncthreads();
      threshold_key(h2, bin, above, K, info);
      const uint32_t theta = info[2], need = info[3];
      // ordered output from the candidates: 1024 threads over cand entries in order, kSelPer each
      uint32_t sel_base = 0, tie_base = 0;
      for (uint32_t c = 0; c < cand; c += kSelChunk) {
        const uint32_t i0 = c + tid * kSelPer;
        uint32_t gt = 0, eq = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const uint32_t i = i0 + j;
          const uint32_t key = i < cand ? ckey[i] : 0u;
          gt += key > theta;
          eq += key == theta;
        }
        uint32_t tot, p = block_excl_scan(gt << 16 | eq, tmp, &tot);
        uint32_t tie = tie_base + (p & 0xffff);
        uint32_t o = sel_base + (p >> 16) + min(tie, need);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const uint32_t i = i0 + j;
          if (i >= cand) break;
          const uint32_t key = ckey[i];
          const bool take = key > theta || (key == theta && tie++ < need);
          if (take) opos[o++] = off + ((int)cidx[i] - lo);
        }
        sel_base += tot >> 16;
        tie_base += tot & 0xffff;
      }
    } else {
      // step 1's passes over the row: the 32-bin histogram inside the bin, then the ordered output
      for (int c = a0; c < hi; c += kSelChunk) {
        uint32_t k[16];
        load16_l2(row, c + tid * kSelPer, lo, hi, k);
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (k[j] && (k[j] >> 5) == bin) atomicAdd(&h2[k[j] & 31], 1u);
      }
      __syncthreads();
      threshold_key(h2, bin, above, K, info);
      const uint32_t theta = info[2], need = info[3];
      uint32_t sel_base = 0, tie_base = 0;
      for (int c = a0; c < hi; c += kSelChunk) {
        const int c0 = c + tid * kSelPer;
        uint32_t k[16];
        load16_l2(row, c0, lo, hi, k);
        uint32_t gt = 0, eq = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          gt += k[j] > theta;
          eq += k[j] == theta;
        }
        uint32_t tot, p = block_excl_scan(gt << 16 | eq, tmp, &tot);
        uint32_t tie = tie_base + (p & 0xffff);
        uint32_t o = sel_base + (p >> 16) + min(tie, need);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const bool take = k[j] > theta || (k[j] == theta && tie++ < need);
          if (take) opos[o++] = off + (c0 + j - lo);
        }
        sel_base += tot >> 16;
        tie_base += tot & 0xffff;
      }
    }
  }
  for (int i = K + tid; i < kmax; i += kSelThreads) opos[i] = -1;
  __syncthreads();
}
}  // namespace

// ca_idx16_score + ca_idx16_select of up to kDenseRows rows in one launch (see the section comment above). Grid: at
// most one CTA per SM (all co-resident; the last R spin on the grid ticket), 1024 threads, kDenseSmem dynamic shared
// bytes. scratch: 2 counters, then [rows][2048] u32 histograms; zero before the first launch, left zero after each.
// keys: [rows][keys_stride] 16-bit keys (every window column written). out: [rows][out_stride] positions, -1 padded.
extern "C" __global__ void __launch_bounds__(kSelThreads, 1)
    ca_idx16_dense(const uint8_t* __restrict__ q, const uint32_t* __restrict__ qsc, const float* __restrict__ w,
                   const uint8_t* __restrict__ k, const uint32_t* __restrict__ ksc, const int* __restrict__ starts,
                   const int* __restrict__ lengths, const int* __restrict__ offsets, int rows, int columns,
                   uint16_t* __restrict__ keys, int keys_stride, int kmax, int* __restrict__ out, int out_stride,
                   uint32_t* __restrict__ scratch) {
  extern __shared__ __align__(16) uint8_t dsm[];
  __shared__ int slo[kDenseRows], shi[kDenseRows], soff[kDenseRows], sspan[2], ticket;
  __shared__ uint32_t h2[32], tmp[64], info[5];
  uint32_t (*sq)[kHeads][16] = reinterpret_cast<uint32_t (*)[kHeads][16]>(dsm);
  uint32_t* ssc = reinterpret_cast<uint32_t*>(dsm + kDenseRows * kHeads * 64);
  float* sw = reinterpret_cast<float*>(ssc + kDenseRows * kHeads);
  uint32_t* hk = reinterpret_cast<uint32_t*>(sw + kDenseRows * kHeads);  // [row][1024] packed u16 pairs
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int R = min(rows, kDenseRows), G = gridDim.x;
  if (tid < R) {
    const int s = starts[tid];
    slo[tid] = s;
    shi[tid] = min(s + lengths[tid], columns);
    soff[tid] = offsets[tid];
  }
  for (int i = tid; i < R * 1024; i += kSelThreads) hk[i] = 0;
  __syncthreads();
  if (tid == 0) {
    int lo = columns, hi = 0;
    for (int rr = 0; rr < R; ++rr)
      if (shi[rr] > slo[rr]) lo = min(lo, slo[rr]), hi = max(hi, shi[rr]);
    sspan[0] = lo & ~(kDensePos - 1);
    sspan[1] = hi;
  }
  const uint4* qsrc = reinterpret_cast<const uint4*>(q);
  for (int i = tid; i < R * kHeads * 4; i += kSelThreads) reinterpret_cast<uint4*>(&sq[0][0][0])[i] = qsrc[i];
  for (int i = tid; i < R * kHeads; i += kSelThreads) {
    ssc[i] = qsc[i];
    sw[i] = w[i];
  }
  __syncthreads();
  const int span_lo = sspan[0], span_hi = sspan[1];
  for (int base = span_lo + blockIdx.x * kDensePos; base < span_hi; base += G * kDensePos) {
    const int i0 = base + warp * 16;
    if (i0 >= span_hi) continue;  // warp-uniform
    bool any = false;
    for (int rr = 0; rr < R; ++rr) any |= i0 < shi[rr] && i0 + 16 > slo[rr];
    if (!any) continue;  // warp-uniform
    const int p0 = i0 + g, p1 = i0 + 8 + g;
    uint4 k0 = make_uint4(0, 0, 0, 0), k1 = k0;
    uint32_t s0 = 0x7f7f7f7fu, s1 = 0x7f7f7f7fu;
    if (p0 < span_hi) {
      k0 = ldg_words_of(reinterpret_cast<const uint32_t*>(k + (size_t)p0 * 64), t);
      s0 = __ldg(ksc + p0);
    }
    if (p1 < span_hi) {
      k1 = ldg_words_of(reinterpret_cast<const uint32_t*>(k + (size_t)p1 * 64), t);
      s1 = __ldg(ksc + p1);
    }
    const uint32_t sa = (t & 1) ? s1 : s0;
    for (int rr = 0; rr < R; ++rr) {
      const int lo = slo[rr], hi = shi[rr];
      if (i0 >= hi || i0 + 16 <= lo) continue;  // warp-uniform
      float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int h = nt * 8 + g;
        const uint4 qv = words_of(sq[rr][h], t);
        const uint32_t qs = ssc[rr * kHeads + h];
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        mma_fp4(d, k0.x, k1.x, k0.y, k1.y, qv.x, qv.y, sa & 0xffff, qs & 0xffff);
        mma_fp4(d, k0.z, k1.z, k0.w, k1.w, qv.z, qv.w, sa >> 16, qs >> 16);
        const float w0 = sw[rr * kHeads + nt * 8 + 2 * t], w1 = sw[rr * kHeads + nt * 8 + 2 * t + 1];
        acc0 = __fadd_rn(acc0, bfround(__fmul_rn(fmaxf(bfround(d[0]), 0.f), w0)));
        acc0 = __fadd_rn(acc0, bfround(__fmul_rn(fmaxf(bfround(d[1]), 0.f), w1)));
        acc1 = __fadd_rn(acc1, bfround(__fmul_rn(fmaxf(bfround(d[2]), 0.f), w0)));
        acc1 = __fadd_rn(acc1, bfround(__fmul_rn(fmaxf(bfround(d[3]), 0.f), w1)));
      }
      acc0 = __fadd_rn(acc0, __shfl_xor_sync(0xffffffffu, acc0, 1));
      acc0 = __fadd_rn(acc0, __shfl_xor_sync(0xffffffffu, acc0, 2));
      acc1 = __fadd_rn(acc1, __shfl_xor_sync(0xffffffffu, acc1, 1));
      acc1 = __fadd_rn(acc1, __shfl_xor_sync(0xffffffffu, acc1, 2));
      uint16_t* krow = keys + (size_t)rr * keys_stride - lo;  // relative to the row's range start
      if (t == 0 && p0 >= lo && p0 < hi) {
        const uint32_t key = key_from_bf16(f2bf(acc0));
        krow[p0] = (uint16_t)key;
        hist_add(hk + rr * 1024, key);
      }
      if (t == 1 && p1 >= lo && p1 < hi) {
        const uint32_t key = key_from_bf16(f2bf(acc1));
        krow[p1] = (uint16_t)key;
        hist_add(hk + rr * 1024, key);
      }
    }
  }
  __syncthreads();
  uint32_t* gh = scratch + 2;
  for (int rr = 0; rr < R; ++rr) hist_flush(hk[rr * 1024 + tid], gh + rr * 2048 + 2 * tid);
  __threadfence();
  __syncthreads();
  if (tid == 0) ticket = (int)atomicAdd(scratch, 1u);
  __syncthreads();
  const int S = min(G, R), first = ticket - (G - S);
  if (first < 0) return;
  if (tid == 0)
    while (ld_acquire_gpu(scratch) < (uint32_t)G) __nanosleep(32);
  __syncthreads();
  __threadfence();
  uint16_t* ckey = reinterpret_cast<uint16_t*>(dsm);
  uint32_t* cidx = reinterpret_cast<uint32_t*>(dsm + kDenseCand * 2);
  for (int r = first; r < R; r += S)
    select_dense_row(keys + (size_t)r * keys_stride, gh + r * 2048, 0, max(shi[r] - slo[r], 0), soff[r], kmax,
                     out + (size_t)r * out_stride, ckey, cidx, h2, tmp, info);
  if (tid == 0 && atomicAdd(scratch + 1, 1u) == (uint32_t)S - 1) {
    scratch[0] = 0;
    scratch[1] = 0;
  }
}
