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

// MT=2 direct-read FP8 x FP8 block-scaled GEMV: each CTA computes 16 consecutive weight rows
// (two 8-row tiles sharing one 32-row UE8M0 scale block), halving activation/scale loads and
// pipelining 2-step batches with zero register spills. Grid: ceil(N / 16) CTAs; block: 32 * wk
// threads (wk <= 8); dynamic shared memory: max(wk * 8 * 16, 264) floats. N % 16 == 0, K % 64 == 0.
extern "C" __global__ void __launch_bounds__(256, 2) ca_direct_gemv_fp8_mt2(
    const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs, const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ S, __nv_bfloat16* __restrict__ y, int M, int N, int K, int ldq, int wk, int scale_mma) {
  extern __shared__ float red[];  // [wk][kRows][16]
  constexpr int kUMt2 = 2;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wi_k = warp;
  const int n0 = blockIdx.x * 16;
  const int KB = K / 32;
  const int P = K / 64;
  const int p0 = (P * wi_k) / wk, p1 = (P * (wi_k + 1)) / wk;
  float c0[4] = {0.f, 0.f, 0.f, 0.f};
  float c1[4] = {0.f, 0.f, 0.f, 0.f};
  const int n = n0 < N ? n0 : 0;
  const uint8_t* wrow0 = W + (size_t)(n + g) * K + 16 * t;
  const uint8_t* wrow1 = W + (size_t)(n + 8 + g) * K + 16 * t;
  // n is a multiple of 16, so rows n..n+15 share one 32-row scale block n/32.
  const uint8_t* srow = S + (size_t)(n / 32) * KB;
  const int off = 16 * (t & 1) + 8 * (t >> 1);
  const bool arow_live = g < M;
  const uint8_t* arow = xq + (size_t)(arow_live ? g : 0) * ldq + off;
  const bool odd = lane & 1, lo = t < 2;
  const int arow_index = arow_live ? g : 0;

  uint4 ba[kUMt2][2], bb[kUMt2][2];
  auto load_w = [&](uint4 (&b)[kUMt2][2], int p) {
#pragma unroll
    for (int u = 0; u < kUMt2; ++u) {
      if (p + u < p1) {
        b[u][0] = ldg_stream128(wrow0 + (size_t)(p + u) * 64);
        b[u][1] = ldg_stream128(wrow1 + (size_t)(p + u) * 64);
      }
    }
  };
  auto comp = [&](const uint4 (&b)[kUMt2][2], int p) {
#pragma unroll
    for (int u = 0; u < kUMt2; ++u) {
      if (p + u >= p1) break;
      const int kb0 = 2 * (p + u);
      const int kb1 = kb0 + 1;
      const uint2 a0 = arow_live ? __ldg(reinterpret_cast<const uint2*>(arow + kb0 * 32)) : make_uint2(0, 0);
      const uint2 a1 = arow_live ? __ldg(reinterpret_cast<const uint2*>(arow + kb1 * 32)) : make_uint2(0, 0);
      const uint8_t* sp0 = scale_mma ? xs + (size_t)(kb0 >> 2) * 512 + arow_index * 16 + (kb0 & 3)
                                     : xs + (size_t)arow_index * KB + kb0;
      const uint8_t* sp1 = scale_mma ? xs + (size_t)(kb1 >> 2) * 512 + arow_index * 16 + (kb1 & 3)
                                     : xs + (size_t)arow_index * KB + kb1;
      const uint32_t sfa0 = (odd || !arow_live) ? 127u : __ldg(sp0);
      const uint32_t sfa1 = (odd || !arow_live) ? 127u : __ldg(sp1);
      const uint32_t sfb0 = __ldg(srow + kb0);
      const uint32_t sfb1 = __ldg(srow + kb1);
      {
        const uint4 w = b[u][0];
        uint32_t s0 = lo ? w.z : w.x, s1 = lo ? w.w : w.y;
        uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 2), r1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        mma_e4m3_e4m3(c0, a0.x, 0u, a0.y, 0u, lo ? w.x : r0, lo ? w.y : r1, sfa0, sfb0);
        mma_e4m3_e4m3(c0, a1.x, 0u, a1.y, 0u, lo ? r0 : w.z, lo ? r1 : w.w, sfa1, sfb1);
      }
      {
        const uint4 w = b[u][1];
        uint32_t s0 = lo ? w.z : w.x, s1 = lo ? w.w : w.y;
        uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 2), r1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        mma_e4m3_e4m3(c1, a0.x, 0u, a0.y, 0u, lo ? w.x : r0, lo ? w.y : r1, sfa0, sfb0);
        mma_e4m3_e4m3(c1, a1.x, 0u, a1.y, 0u, lo ? r0 : w.z, lo ? r1 : w.w, sfa1, sfb1);
      }
    }
  };
  load_w(ba, p0);
  pdl_launch_dependents();
  pdl_wait();
  for (int p = p0; p < p1; p += 2 * kUMt2) {
    if (p + kUMt2 < p1) load_w(bb, p + kUMt2);
    comp(ba, p);
    if (p + kUMt2 >= p1) break;
    if (p + 2 * kUMt2 < p1) load_w(ba, p + 2 * kUMt2);
    comp(bb, p + kUMt2);
  }
  {
    const int r = 2 * t;
    red[(wi_k * kRows + g) * 16 + r] = c0[0];
    red[(wi_k * kRows + g) * 16 + r + 1] = c0[1];
    red[(wi_k * kRows + g) * 16 + 8 + r] = c1[0];
    red[(wi_k * kRows + g) * 16 + 8 + r + 1] = c1[1];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < M * 16; idx += blockDim.x) {
    const int m = idx >> 4, r = idx & 15;
    if (n0 + r >= N) continue;
    float v = 0.f;
    for (int j = 0; j < wk; ++j) v += red[(j * kRows + m) * 16 + r];
    y[(size_t)m * N + n0 + r] = __float2bfloat16_rn(v);
  }
}

namespace {

__device__ __forceinline__ float woa_scale(uint32_t exponent) {
  float scale;
  const float x = float(exponent) - 127.f;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(scale) : "f"(x));
  return scale;
}

__device__ __forceinline__ uint32_t woa_bf16x2(uint32_t pair, float scale) {
  uint32_t out;
  const unsigned short bytes = (unsigned short)(pair & 0xffffu);
  asm volatile("{ .reg .b16 lo, hi; .reg .b32 h2; .reg .f32 f0, f1;\n"
               "cvt.rn.f16x2.e4m3x2 h2, %1;\n"
               "mov.b32 {lo, hi}, h2;\n"
               "cvt.f32.f16 f0, lo;\n"
               "cvt.f32.f16 f1, hi;\n"
               "mul.f32 f0, %2, f0;\n"
               "mul.f32 f1, %2, f1;\n"
               "cvt.rn.bf16.f32 lo, f0;\n"
               "cvt.rn.bf16.f32 hi, f1;\n"
               "mov.b32 %0, {lo, hi}; }"
               : "=r"(out)
               : "h"(bytes), "f"(scale));
  return out;
}

__device__ __forceinline__ void mma_bf16_m16n8k16(float (&c)[4], uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  const uint32_t zero = 0u;
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(zero), "r"(a2), "r"(zero), "r"(b0), "r"(b1));
}

}  // namespace

// Single-kernel fused grouped WO-A ([2048, 4096], 2 groups of [1024, 4096]) + split-K reduction +
// BF16 rounding + exact MXFP8 quantization into (values [M, 2048], scale_mma SM12x layout).
// Grid: (64, 1, 1) (one CTA per 32-column MXFP8 output scale group); block: (256, 1, 1) (wn=2 x wk=4)
// so 2 CTAs/SM fit on 48 SMs (single wave, zero tail barrier) while firing PDL for wo_b.
extern "C" __global__ void __launch_bounds__(256, 2) ca_woa_fused_quant_m6(
    const __nv_bfloat16* __restrict__ X, const uint8_t* __restrict__ W8, const uint8_t* __restrict__ S,
    uint8_t* __restrict__ values, uint8_t* __restrict__ scale_mma, int M) {
  __shared__ float red[4][8][32];  // [wk=4][rows=8][cols=32]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wi_n = warp & 1;   // 0 -> cols 0..15, 1 -> cols 16..31
  const int wi_k = warp >> 1;  // 0..3 -> K split of 16 x 64 = 1024 elements
  const int out_group = blockIdx.x;  // 0..63
  const int group = out_group >> 5;  // 0 or 1
  const int n_cta = (out_group & 31) * 32;
  const int n_tile = n_cta + wi_n * 16;
  const int p0 = wi_k * 16, p1 = p0 + 16;  // 64-byte K steps (each step = 2 scale blocks of 32)

  const uint8_t* wrow0 = W8 + (size_t)(group * 1024 + n_tile + g) * 4096 + 16 * t;
  const uint8_t* wrow1 = W8 + (size_t)(group * 1024 + n_tile + 8 + g) * 4096 + 16 * t;
  const uint8_t* srow = S + (size_t)out_group * 128;
  const bool live = g < M;
  const uint32_t* x_u32 = reinterpret_cast<const uint32_t*>(X + (size_t)(live ? g : 0) * 8192 + group * 4096) + t;

  float c0[4] = {0.f, 0.f, 0.f, 0.f};
  float c1[4] = {0.f, 0.f, 0.f, 0.f};

  constexpr int kWoaU = 2;
  uint4 ba[kWoaU][2], bb[kWoaU][2];
  uint32_t sa0[kWoaU], sa1[kWoaU], sb0[kWoaU], sb1[kWoaU];
  auto load_w = [&](uint4 (&b)[kWoaU][2], uint32_t (&sc0)[kWoaU], uint32_t (&sc1)[kWoaU], int p) {
#pragma unroll
    for (int u = 0; u < kWoaU; ++u) {
      b[u][0] = ldg_stream128(wrow0 + (size_t)(p + u) * 64);
      b[u][1] = ldg_stream128(wrow1 + (size_t)(p + u) * 64);
      sc0[u] = __ldg(srow + 2 * (p + u));
      sc1[u] = __ldg(srow + 2 * (p + u) + 1);
    }
  };
  auto comp = [&](const uint4 (&b)[kWoaU][2], const uint32_t (&sc0)[kWoaU], const uint32_t (&sc1)[kWoaU], int p) {
    const int base_lane = g << 2;
    const int shift = 16 * (t & 1);
    const bool hi_half = (t & 2) != 0;
#pragma unroll
    for (int u = 0; u < kWoaU; ++u) {
      const float scale0 = woa_scale(sc0[u]);
      const float scale1 = woa_scale(sc1[u]);
      const uint4 w0 = b[u][0];
      const uint4 w1 = b[u][1];
#pragma unroll
      for (int m_idx = 0; m_idx < 4; ++m_idx) {
        const float sc = (m_idx < 2) ? scale0 : scale1;
        const int k_base = (p + u) * 64 + m_idx * 16;
        uint32_t a0 = 0u, a2 = 0u;
        if (live) {
          a0 = __ldg(x_u32 + (k_base >> 1));
          a2 = __ldg(x_u32 + (k_base >> 1) + 4);
        }
        const int src = base_lane | m_idx;
        {
          uint32_t p0w = __shfl_sync(0xffffffffu, w0.x, src);
          uint32_t p1w = __shfl_sync(0xffffffffu, w0.y, src);
          uint32_t p2w = __shfl_sync(0xffffffffu, w0.z, src);
          uint32_t p3w = __shfl_sync(0xffffffffu, w0.w, src);
          uint32_t lo16 = (hi_half ? p1w : p0w) >> shift;
          uint32_t hi16 = (hi_half ? p3w : p2w) >> shift;
          mma_bf16_m16n8k16(c0, a0, a2, woa_bf16x2(lo16, sc), woa_bf16x2(hi16, sc));
        }
        {
          uint32_t p0w = __shfl_sync(0xffffffffu, w1.x, src);
          uint32_t p1w = __shfl_sync(0xffffffffu, w1.y, src);
          uint32_t p2w = __shfl_sync(0xffffffffu, w1.z, src);
          uint32_t p3w = __shfl_sync(0xffffffffu, w1.w, src);
          uint32_t lo16 = (hi_half ? p1w : p0w) >> shift;
          uint32_t hi16 = (hi_half ? p3w : p2w) >> shift;
          mma_bf16_m16n8k16(c1, a0, a2, woa_bf16x2(lo16, sc), woa_bf16x2(hi16, sc));
        }
      }
    }
  };

  load_w(ba, sa0, sa1, p0);
  pdl_wait();
  for (int p = p0; p < p1; p += 2 * kWoaU) {
    load_w(bb, sb0, sb1, p + kWoaU);
    comp(ba, sa0, sa1, p);
    if (p + 2 * kWoaU < p1) load_w(ba, sa0, sa1, p + 2 * kWoaU);
    comp(bb, sb0, sb1, p + kWoaU);
  }

  {
    const int c_off = wi_n * 16 + 2 * t;
    red[wi_k][g][c_off] = c0[0];
    red[wi_k][g][c_off + 1] = c0[1];
    red[wi_k][g][c_off + 8] = c1[0];
    red[wi_k][g][c_off + 9] = c1[1];
  }
  __syncthreads();
  pdl_launch_dependents();

  if (warp < M) {
    const int m = warp;
    const int col = lane;
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) sum += red[j][m][col];
    const float v = __bfloat162float(__float2bfloat16_rn(sum));
    float amax = fabsf(v);
#pragma unroll
    for (int sh = 16; sh >= 1; sh >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, sh));
    float sc;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(sc) : "f"(amax), "f"(1.f / 448.f));
    const uint32_t bits = __float_as_uint(sc);
    const uint32_t exp = (bits >> 23) & 0xFFu;
    const uint32_t man = bits & 0x7FFFFFu;
    const uint32_t bump = (man != 0u && (exp != 0u || man > 0x400000u)) ? 1u : 0u;
    const uint32_t e8m0 = (sc <= 0.f) ? 0u : min(exp + bump, 254u);
    float inv = 0.f;
    if (e8m0 != 0u) {
      const float pw = float(127 - int(e8m0));
      asm("ex2.approx.f32 %0, %1;" : "=f"(inv) : "f"(pw));
    }
    float qf;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(qf) : "f"(v), "f"(inv));
    unsigned short e4m3_pair;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3_pair) : "f"(0.f), "f"(qf));
    values[(size_t)m * 2048 + (size_t)out_group * 32 + col] = uint8_t(e4m3_pair & 0xFFu);
    if (col == 0) {
      scale_mma[(size_t)(out_group >> 2) * 512 + (size_t)m * 16 + (out_group & 3)] = uint8_t(e8m0);
    }
  }
}
