// Direct-read FP8 x FP8 block-scaled GEMV for a few activation rows (decode / verify).
//
// Adapted from pico-dgx-ds4 (engine/kernels/gemv_fp8.cu and common.cuh at f211791), MIT License,
// Copyright (c) 2026 Andy Shu; the full notice is kept in licenses/pico-dgx-ds4-MIT.txt.
// What is taken: the lane layout of the m16n8k32 block-scaled MMA with 16-byte coalesced streaming loads, the
// two-buffer software pipeline whose first weight batch is requested before the dependency wait, and the fixed
// order split-K epilogue. What differs: one weight matrix in the checkpoint's row-major layout ([N][K] E4M3,
// [N/32][K/32] UE8M0 scales) read where the engine's atlas already holds it, activations from the engine's
// quantizer buffers (E4M3 [M][ldq]; scales either [M][K/32] or the SM12x MMA byte layout), one tile of 8 weight
// rows per CTA, K split over the CTA's warps.
#include <cstdint>
#include <cuda_bf16.h>

namespace {

// D += A(16x32, e4m3) * B(32x8, e4m3), one ue8m0 scale per row of A and per column of B.
__device__ __forceinline__ void mma_e4m3_e4m3(float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
        "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}
__device__ __forceinline__ uint4 ldg_stream128(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

constexpr int kU = 4;      // 64-byte steps loaded per batch
constexpr int kRows = 8;   // activation rows an MMA's lower half holds

}  // namespace

// Grid: ceil(N / 8) CTAs; block: 32 * wk threads (wk warps split K); dynamic shared memory:
// max(wk * 8 * 8, 264) floats. M <= 8 activation rows. K a multiple of 64, N a multiple of 8.
// scale_mma != 0: activation scales in the SM12x MMA byte layout (row r < 32, k-block kb at
// (kb / 4) * 512 + r * 16 + kb % 4); otherwise row-major [M][K / 32].
extern "C" __global__ void __launch_bounds__(512) ca_direct_gemv_fp8(
    const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs, const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ S, __nv_bfloat16* __restrict__ y, int M, int N, int K, int ldq, int wk, int scale_mma) {
  extern __shared__ float red[];  // [wk][kRows][8]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wi_k = warp;
  const int n0 = blockIdx.x * 8;
  const int KB = K / 32;
  // 64-byte steps (two k-blocks): lane (g, t) loads 16 bytes of weight row n0 + g at 16t, then trades half with
  // lane t^2 so that it holds 8 bytes of each block at offset 16(t&1) + 8(t>>1).
  const int P = K / 64;
  const int p0 = (P * wi_k) / wk, p1 = (P * (wi_k + 1)) / wk;
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  const int n = n0 < N ? n0 : 0;
  const uint8_t* wrow = W + (size_t)(n + g) * K + 16 * t;
  const uint8_t* srow = S + (size_t)(n / 32) * KB;
  const int off = 16 * (t & 1) + 8 * (t >> 1);
  const bool arow_live = g < M;
  const uint8_t* arow = xq + (size_t)(arow_live ? g : 0) * ldq + off;
  const bool odd = lane & 1, lo = t < 2;
  const int arow_index = arow_live ? g : 0;

  // Software pipeline: the weights of the next batch are in flight while this one computes. The first batch is
  // requested before waiting on the previous kernel (PDL), since weights never depend on it.
  uint4 ba[kU], bb[kU];
  auto load_w = [&](uint4 (&b)[kU], int p) {
#pragma unroll
    for (int u = 0; u < kU; ++u)
      if (p + u < p1) b[u] = ldg_stream128(wrow + (size_t)(p + u) * 64);
  };
  auto comp = [&](const uint4 (&b)[kU], int p) {
    uint2 a[kU][2];
    uint32_t sfa[kU][2], sfb[kU][2];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      if (p + u >= p1) break;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int kb = 2 * (p + u) + h;
        a[u][h] = arow_live ? __ldg(reinterpret_cast<const uint2*>(arow + kb * 32)) : make_uint2(0, 0);
        const uint8_t* sp = scale_mma ? xs + (size_t)(kb >> 2) * 512 + arow_index * 16 + (kb & 3)
                                      : xs + (size_t)arow_index * KB + kb;
        sfa[u][h] = (odd || !arow_live) ? 127u : __ldg(sp);
        sfb[u][h] = __ldg(srow + kb);
      }
    }
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      if (p + u >= p1) break;
      const uint4 w = b[u];
      uint32_t s0 = lo ? w.z : w.x, s1 = lo ? w.w : w.y;
      uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 2), r1 = __shfl_xor_sync(0xffffffffu, s1, 2);
      mma_e4m3_e4m3(c, a[u][0].x, 0u, a[u][0].y, 0u, lo ? w.x : r0, lo ? w.y : r1, sfa[u][0], sfb[u][0]);
      mma_e4m3_e4m3(c, a[u][1].x, 0u, a[u][1].y, 0u, lo ? r0 : w.z, lo ? r1 : w.w, sfa[u][1], sfb[u][1]);
    }
  };
  load_w(ba, p0);
  pdl_launch_dependents();
  pdl_wait();
  for (int p = p0; p < p1; p += 2 * kU) {
    if (p + kU < p1) load_w(bb, p + kU);
    comp(ba, p);
    if (p + kU >= p1) break;
    if (p + 2 * kU < p1) load_w(ba, p + 2 * kU);
    comp(bb, p + kU);
  }
  // Partial sums per k-slice -> shared memory [wk][row][weight row], then a fixed-order sum over the slices.
  {
    const int r = 2 * t;
    red[(wi_k * kRows + g) * 8 + r] = c[0];
    red[(wi_k * kRows + g) * 8 + r + 1] = c[1];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < M * 8; idx += blockDim.x) {
    const int m = idx / 8, r = idx - m * 8;
    if (n0 + r >= N) continue;
    float v = 0.f;
    for (int j = 0; j < wk; ++j) v += red[(j * kRows + m) * 8 + r];
    y[(size_t)m * N + n0 + r] = __float2bfloat16_rn(v);
  }
}
