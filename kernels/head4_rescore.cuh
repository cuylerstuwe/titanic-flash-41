// 4-bit vocabulary head with exact rescoring for the six-row verify owner (CANADA_SCRATCH_HEAD4_RESCORE): AN OPTION,
// OFF BY DEFAULT. It acts on the model's own output head (not the drafter's) and changes the distribution a token
// is sampled from, by design: tokens outside the rescored set keep a 4-bit logit (error 0.24 rms). Measured at cap
// 512, window 24 against the exact head on the same input: total variation 1.5e-6...2.5e-6 a token on average (one
// draw in ~400,000-660,000 differs), KL 4e-7...7e-7 nats a token. Nature of the difference, every measurement and
// the decision that keeps it off: the module documentation of src/head4_rescore.rs.
// The drafter's 4-bit copy of the head projects every vocabulary row;
// ca_head4_select publishes those logits as FP32 and picks, per live row and per rank, the rows within `window` of
// the row's largest 4-bit logit (at most `cap`, the largest ones to the histogram's resolution window / 128);
// ca_head4_rescore replaces the picked rows' logits by FP32 dot products with the canonical BF16 weights.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

#define CA_H4_N 32320
#define CA_H4_K 5120
#define CA_H4_BINS 128

// Grid: one block a live row, 1024 threads.
extern "C" __global__ void __launch_bounds__(1024) ca_head4_select(
    const __nv_bfloat16* __restrict__ q,     // [rows][N] 4-bit logits, column j = vocabulary row order[j] (j when order is null)
    const uint32_t* __restrict__ order,
    float* __restrict__ logits,              // [rows][N] FP32, vocabulary order
    int32_t* __restrict__ counts,            // [rows]
    int32_t* __restrict__ indices,           // [rows][cap] vocabulary rows to rescore
    float window, int cap) {
    __shared__ float smax[32];
    __shared__ unsigned int hist[CA_H4_BINS];
    __shared__ unsigned int taken;
    __shared__ int limit;
    const int row = blockIdx.x, t = threadIdx.x, T = blockDim.x;
    const __nv_bfloat16* src = q + (size_t)row * CA_H4_N;
    float* dst = logits + (size_t)row * CA_H4_N;
    float best = -INFINITY;
    for (int j = t; j < CA_H4_N; j += T) {
        const float v = __bfloat162float(src[j]);
        dst[order ? order[j] : (uint32_t)j] = v;
        best = fmaxf(best, v);
    }
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, d));
    if ((t & 31) == 0) smax[t >> 5] = best;
    if (t < CA_H4_BINS) hist[t] = 0;
    if (t == 0) taken = 0;
    __syncthreads();
    if (t < 32) {
        float m = smax[t];
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
        if (t == 0) smax[0] = m;
    }
    __syncthreads();
    const float top = smax[0], scale = (float)CA_H4_BINS / window;
    for (int j = t; j < CA_H4_N; j += T) {
        const float gap = top - __bfloat162float(src[j]);
        if (gap < window) atomicAdd(&hist[min(CA_H4_BINS - 1, (int)(gap * scale))], 1u);
    }
    __syncthreads();
    if (t == 0) {
        unsigned int total = 0;
        int b = 0;
        for (; b < CA_H4_BINS; ++b) {
            if (total + hist[b] > (unsigned int)cap) break;
            total += hist[b];
        }
        limit = b > 0 ? b : 1;   // a first bin larger than the cap is cut at the cap below
    }
    __syncthreads();
    const int lim = limit;
    for (int j = t; j < CA_H4_N; j += T) {
        const float gap = top - __bfloat162float(src[j]);
        if (gap < window && min(CA_H4_BINS - 1, (int)(gap * scale)) < lim) {
            const unsigned int slot = atomicAdd(&taken, 1u);
            if (slot < (unsigned int)cap) indices[(size_t)row * cap + slot] = order ? (int32_t)order[j] : j;
        }
    }
    __syncthreads();
    if (t == 0) counts[row] = (int32_t)min(taken, (unsigned int)cap);
}

// Grid: ceil(rows * cap / 8) blocks of 256 threads; warp (row * cap + e) rescored entry e of a row.
extern "C" __global__ void __launch_bounds__(256) ca_head4_rescore(
    const __nv_bfloat16* __restrict__ x,     // [rows][K] the head's input
    const __nv_bfloat16* __restrict__ w,     // [N][K] canonical weights of this rank's vocabulary rows
    const int32_t* __restrict__ counts, const int32_t* __restrict__ indices,
    float* __restrict__ logits, int rows, int cap) {
    const int entry = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int row = entry / cap, e = entry - row * cap;
    if (row >= rows || e >= counts[row]) return;
    const int v = indices[(size_t)row * cap + e];
    const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)row * CA_H4_K);
    const uint4* wr = reinterpret_cast<const uint4*>(w + (size_t)v * CA_H4_K);
    float acc = 0.f;
    for (int p = lane; p < CA_H4_K / 8; p += 32) {
        const uint4 a = xr[p], b = wr[p];
        const __nv_bfloat16* xa = reinterpret_cast<const __nv_bfloat16*>(&a);
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&b);
        #pragma unroll
        for (int k = 0; k < 8; ++k) acc = __fmaf_rn(__bfloat162float(xa[k]), __bfloat162float(wb[k]), acc);
    }
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, d));
    if (lane == 0) logits[(size_t)row * CA_H4_N + v] = acc;
}
