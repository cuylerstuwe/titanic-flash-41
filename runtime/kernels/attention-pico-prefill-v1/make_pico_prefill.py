#!/usr/bin/env python3
"""Generate prefill.cu: pico-dgx-ds4's prefill sparse attention (engine/kernels/prefill_attn.cu, MIT, Andy Shu) adapted
to Titanic's ca_sparse_prefill ABI 2 and cache layouts. Usage: make_pico_prefill.py PICO_ENGINE_DIR OUT.cu

Kept verbatim: the QK and PV blocks (packed keys in shared memory, exact f16 widening in registers, f16 MMAs), the
online softmax with the lazy running max, P rounded to bf16, the sink in the denominator, the persistent-CTA pipeline.
Changed (every substitution asserted):
  - window keys: the row's explicit main candidate slots (128 per row, -1 or past the row's length = none) instead
    of positions pos-127..pos; both caches addressed through Titanic's 64-token pages (64 data rows, then 64 scale
    rows; page strides from the caller) instead of flat arrays;
  - compressed keys: the row's extra candidates with explicit lengths;
  - no RoPE: Titanic applies RoPE to q before attention and the inverse RoPE to the output after it;
  - the caller's softmax scale; a log2 LSE with the sink (Titanic's convention) written per row and head.
"""
import sys
from pathlib import Path

src = (Path(sys.argv[1]) / "kernels/prefill_attn.cu").read_text()
out = Path(sys.argv[2])


def sub(text, old, new, count=1):
    n = text.count(old)
    assert n == count, (n, old[:100])
    return text.replace(old, new)


a = src.index("constexpr int kThreads = 256;")
b = src.index("}  // namespace\n\nvoid sparse_attn_p(")
k = src[a:b]

# shared memory: no RoPE table
k = sub(k, "constexpr int kOffTab = kOffP + kHeads * kPS * 2;  // RoPE cos/sin of dims 448..511 at a row's pos, 2 slots\n"
           "constexpr int kSmem = kOffTab + 2 * (kRope / 2) * 8;",
        "constexpr int kSmem = kOffP + kHeads * kPS * 2;")

k = sub(k, """struct PArgs {
  const uint16_t* q;
  const int* pos;
  int rows;
  const float2* table;
  attn::Window win;
  attn::Compressed ckv;
  const int* topk;
  int kmax, nb, win_lo;
  const float* sink;
  float scale;
  uint16_t* out;
};""", """struct PArgs {
  const uint16_t* q;       // [rows][16][512] bf16, RoPE already applied
  int rows;
  const uint8_t* mkv;      // window cache: 64-token pages of [64][512] e4m3 then [64][16] ue8m0
  size_t mstride;          // its page stride in bytes
  const int* mid;          // [rows][128] window slots (-1 = none)
  const int* mlen;         // [rows] valid window candidates
  const uint8_t* ekv;      // compressed cache: 64-entry pages of [64][256] e2m1 pairs then [64][32] e4m3
  size_t estride;
  const int* topk;         // [rows][kmax] compressed slots (-1 = none)
  const int* elen;         // [rows] valid compressed candidates
  int kmax, nb;
  const float* sink;
  float scale;
  uint16_t* out;
  float* lse;              // [rows][16] log2-sum-exp with the sink (Titanic's convention)
};""")

k = sub(k, """// block b of row r into stage st: b < 2 the window (positions pos-127+64b+k from win_lo on), else topk
// entries 64(b-2)+k.
// Invalid keys are zero-filled (their scores are masked; zero codes keep P*V finite).
__device__ __forceinline__ void issue_block(const PArgs& a, uint8_t* sm, int p, int b, int st) {""",
        """// block b of row r into stage st: b < 2 the row's window candidates 64b+k, else its compressed candidates
// 64(b-2)+k. Invalid keys are zero-filled (their scores are masked; zero codes keep P*V finite).
__device__ __forceinline__ void issue_block(const PArgs& a, uint8_t* sm, int r, int b, int st) {""")
k = sub(k, """      const int k = (tid >> 5) + 8 * it, kp = p - (kWindow - 1) + kBlk * b + k;
      const bool ok = kp >= a.win_lo;
      const size_t slot = ok ? (size_t)((kp - a.win.base) & a.win.mask) : 0;
      pa::cp16(cod + k * 512 + swz(k, c) * 16, a.win.kv + slot * kHead + c * 16, ok);
      if (c == 0) {
        pa::cp16(scl + k * 32, a.win.sc + slot * 16, ok);
        val[k] = ok;
      }""", """      const int k = (tid >> 5) + 8 * it, kk = kBlk * b + k;
      const int sl = kk < __ldg(a.mlen + r) ? __ldg(a.mid + (size_t)r * kWindow + kk) : -1;
      const bool ok = sl >= 0;
      const uint8_t* page = a.mkv + (ok ? (size_t)(sl >> 6) * a.mstride : 0);
      const int row = ok ? (sl & 63) : 0;
      pa::cp16(cod + k * 512 + swz(k, c) * 16, page + row * kHead + c * 16, ok);
      if (c == 0) {
        pa::cp16(scl + k * 32, page + kBlk * kHead + row * 16, ok);
        val[k] = ok;
      }""")
k = sub(k, """    const int* tk = reinterpret_cast<const int*>(sm + kOffTk) + kBlk * (b - 2);
    const int c = tid & 15;
    int kps[4];
#pragma unroll
    for (int it = 0; it < 4; ++it) kps[it] = tk[(tid >> 4) + 16 * it];
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int k = (tid >> 4) + 16 * it, kp = kps[it];
      const bool ok = kp >= 0;
      const size_t e = ok ? (size_t)kp : 0;
      pa::cp16(cod + k * 512 + swz(k, c) * 16, a.ckv.kv + e * (kHead / 2) + c * 16, ok);
      if (c < 2) pa::cp16(scl + k * 32 + c * 16, a.ckv.sc + e * 32 + c * 16, ok);
      if (c == 0) val[k] = ok;
    }""", """    const int* tk = reinterpret_cast<const int*>(sm + kOffTk) + kBlk * (b - 2);
    const int c = tid & 15, elen = __ldg(a.elen + r);
    int kps[4];
#pragma unroll
    for (int it = 0; it < 4; ++it) kps[it] = tk[(tid >> 4) + 16 * it];
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int k = (tid >> 4) + 16 * it, kp = kps[it];
      const bool ok = kp >= 0 && kBlk * (b - 2) + k < elen;
      const uint8_t* page = a.ekv + (ok ? (size_t)(kp >> 6) * a.estride : 0);
      const int row = ok ? (kp & 63) : 0;
      pa::cp16(cod + k * 512 + swz(k, c) * 16, page + row * (kHead / 2) + c * 16, ok);
      if (c < 2) pa::cp16(scl + k * 32 + c * 16, page + kBlk * (kHead / 2) + row * 32 + c * 16, ok);
      if (c == 0) val[k] = ok;
    }""")

# q: no RoPE table
k = sub(k, """// row r's q, and its RoPE table row (pos p) into table slot `slot`
__device__ __forceinline__ void issue_q(const PArgs& a, uint8_t* sm, int r, int p, int slot) {
  const uint8_t* src = reinterpret_cast<const uint8_t*>(a.q + (size_t)r * kHeads * kHead);
  for (int i = threadIdx.x; i < kHeads * 64; i += kThreads)
    pa::cp16(sm + kOffQ + (i >> 6) * 1024 + qswz(i >> 6, i & 63) * 16, src + i * 16, true);
  if (threadIdx.x < kRope / 4)
    pa::cp16(sm + kOffTab + slot * kRope * 4 + threadIdx.x * 16,
             reinterpret_cast<const uint8_t*>(a.table + (size_t)p * 32) + threadIdx.x * 16, true);
}""", """// row r's q (RoPE already applied by the caller)
__device__ __forceinline__ void issue_q(const PArgs& a, uint8_t* sm, int r) {
  const uint8_t* src = reinterpret_cast<const uint8_t*>(a.q + (size_t)r * kHeads * kHead);
  for (int i = threadIdx.x; i < kHeads * 64; i += kThreads)
    pa::cp16(sm + kOffQ + (i >> 6) * 1024 + qswz(i >> 6, i & 63) * 16, src + i * 16, true);
}""")
k = sub(k, "__device__ __forceinline__ void load_q(const uint8_t* sm, const float2* tb, int dg, uint32_t (&qa)[16][4]) {",
        "__device__ __forceinline__ void load_q(const uint8_t* sm, int dg, uint32_t (&qa)[16][4]) {")
k = sub(k, """      if (d0 >= kHead - kRope) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 cs = tb[(d0 - (kHead - kRope)) / 2 + j];
          attn::rope_pair(x[2 * j], x[2 * j + 1], cs.x, cs.y, false);
          x[2 * j] = attn::bfround(x[2 * j]);
          x[2 * j + 1] = attn::bfround(x[2 * j + 1]);
        }
      }
""", "")
k = sub(k, """// o (bf16) of dims (d, d+1), the inverse RoPE on dims 448..511 applied to the bf16 values (as decode)
__device__ __forceinline__ uint32_t finish_pair(float x0, float x1, int d, const float2* tb) {
  x0 = attn::bfround(x0);
  x1 = attn::bfround(x1);
  if (d >= kHead - kRope) {
    const float2 cs = tb[(d - (kHead - kRope)) >> 1];
    attn::rope_pair(x0, x1, cs.x, cs.y, true);
  }
  return pack_bf16(x0, x1);
}""", """// o (bf16) of dims (d, d+1); the caller applies the inverse RoPE
__device__ __forceinline__ uint32_t finish_pair(float x0, float x1) {
  return pack_bf16(attn::bfround(x0), attn::bfround(x1));
}""")

# main loop: rows instead of positions
k = sub(k, """  // positions of this row and the next (read a row ahead: the window loads and the q / table prefetch
  // need them before their row starts)
  int p_cur = __ldg(a.pos + blockIdx.x), p_nxt = (int)blockIdx.x + G < a.rows ? __ldg(a.pos + blockIdx.x + G) : 0;
  issue_q(a, sm, blockIdx.x, p_cur, 0);
  issue_block(a, sm, p_cur, 0, 0);""", """  issue_q(a, sm, blockIdx.x);
  issue_block(a, sm, blockIdx.x, 0, 0);""")
k = sub(k, """    const float2* tb = reinterpret_cast<const float2*>(sm + kOffTab + (ri & 1) * kRope * 4);
    if (b == 0 && ri > 0) {
      p_cur = p_nxt;
      p_nxt = r + G < a.rows ? __ldg(a.pos + r + G) : 0;
    }
""", "")
k = sub(k, "      load_q(sm, tb, dg, qa);", "      load_q(sm, dg, qa);")
k = sub(k, "    if (it + 1 < nitems) issue_block(a, sm, b + 1 < a.nb ? p_cur : p_nxt, b + 1 < a.nb ? b + 1 : 0, st ^ 1);",
        "    if (it + 1 < nitems) issue_block(a, sm, b + 1 < a.nb ? r : r + G, b + 1 < a.nb ? b + 1 : 0, st ^ 1);")
k = sub(k, "    if (b == 1 && r + G < a.rows) issue_q(a, sm, r + G, p_nxt, (ri + 1) & 1);",
        "    if (b == 1 && r + G < a.rows) issue_q(a, sm, r + G);")
k = sub(k, """    if (b == a.nb - 1) {  // finish the row: sink, normalise, bf16, inverse RoPE
      const float d0 = __fadd_rn(l0, expf(__fsub_rn(__ldg(a.sink + g), m0)));
      const float d1 = __fadd_rn(l1, expf(__fsub_rn(__ldg(a.sink + g + 8), m1)));""",
        """    if (b == a.nb - 1) {  // finish the row: sink, normalise, bf16, LSE
      const float d0 = __fadd_rn(l0, expf(__fsub_rn(__ldg(a.sink + g), m0)));
      const float d1 = __fadd_rn(l1, expf(__fsub_rn(__ldg(a.sink + g + 8), m1)));
      if (w == 0 && t == 0) {  // log2 of the denominator, sink included, in score units
        a.lse[(size_t)r * kHeads + g] = (m0 + logf(d0)) * 1.4426950408889634f;
        a.lse[(size_t)r * kHeads + g + 8] = (m1 + logf(d1)) * 1.4426950408889634f;
      }""")
k = sub(k, "          wv[i] = finish_pair(__fdiv_rn(o[2 * i][2 * hh], dd), __fdiv_rn(o[2 * i + 1][2 * hh], dd), dbase + 2 * i, tb);",
        "          wv[i] = finish_pair(__fdiv_rn(o[2 * i][2 * hh], dd), __fdiv_rn(o[2 * i + 1][2 * hh], dd));")
k = sub(k, """          wv[4 + i] = finish_pair(__fdiv_rn(o[2 * i][2 * hh + 1], dd), __fdiv_rn(o[2 * i + 1][2 * hh + 1], dd),
                                  dbase + 8 + 2 * i, tb);""",
        """          wv[4 + i] = finish_pair(__fdiv_rn(o[2 * i][2 * hh + 1], dd), __fdiv_rn(o[2 * i + 1][2 * hh + 1], dd));""")
for gone in ("a.pos", "a.table", "a.win", "a.ckv", "kOffTab", "rope_pair", "tb["):
    assert gone not in k, gone

HEADER = r'''// Titanic adaptation of pico-dgx-ds4's prefill sparse attention (engine/kernels/prefill_attn.cu, MIT License,
// Copyright (c) 2026 Andy Shu; see LICENSE-pico). Generated by make_pico_prefill.py, which lists every change.
// ca_sparse_prefill ABI 2: the window cache in the V4.1 layout (e4m3, ue8m0 per 32) and the compressed cache in the
// V4.1-FP4 layout (e2m1, e4m3 per 16), both in 64-token pages; explicit per-row candidates and lengths; q with RoPE
// applied; output before the inverse RoPE; log2 LSE with the sink.
#include <algorithm>
#include <cmath>

#include "attn/attn_common.cuh"
#include "common/check.h"
#include "kernels/pa_common.cuh"

namespace dsk {
namespace {

using attn::kHead;
using attn::kHeads;
using attn::kRope;
using attn::kWindow;

'''
FOOTER = r'''
}  // namespace
}  // namespace dsk

extern "C" int ca_sparse_prefill_abi() { return 2; }

extern "C" int ca_sparse_prefill(int rows, int extra_topk, int extra_pbs, float scale, const void* q, const void* kv,
                                 size_t stride, const void* indices, void* output, void* lse, const void* lengths,
                                 const void* sink, const void* extra_kv, size_t extra_stride, const void* extra_indices,
                                 const void* extra_lengths, cudaStream_t stream) {
  using namespace dsk;
  static int nsm = 0;
  if (!nsm) {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d) != cudaSuccess)
      return -10;
    if (cudaFuncSetAttribute(attn_p_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem) != cudaSuccess) return -11;
  }
  if (rows < 1 || rows > 16384 || !(scale > 0) || extra_topk < 0 || extra_topk > kMaxK || extra_topk % kBlk || !q ||
      !kv || !indices || !output || !lse || !lengths || !sink || stride < 64 * 528)
    return -1;
  PArgs a{};
  a.q = static_cast<const uint16_t*>(q);
  a.rows = rows;
  a.mkv = static_cast<const uint8_t*>(kv);
  a.mstride = stride;
  a.mid = static_cast<const int*>(indices);
  a.mlen = static_cast<const int*>(lengths);
  if (extra_topk > 0) {
    if (!extra_kv || !extra_indices || !extra_lengths || extra_pbs != 64 || extra_stride < 64 * 288) return -3;
    a.ekv = static_cast<const uint8_t*>(extra_kv);
    a.estride = extra_stride;
    a.topk = static_cast<const int*>(extra_indices);
    a.elen = static_cast<const int*>(extra_lengths);
  } else if (extra_kv || extra_indices || extra_lengths) {
    return -2;
  }
  a.kmax = extra_topk;
  a.nb = 2 + extra_topk / kBlk;
  a.sink = static_cast<const float*>(sink);
  a.scale = scale;
  a.out = static_cast<uint16_t*>(output);
  a.lse = static_cast<float*>(lse);
  attn_p_kernel<<<std::min(rows, nsm), kThreads, kSmem, stream>>>(a);
  return (int)cudaPeekAtLastError();
}
'''
out.write_text(HEADER + k + FOOTER)
print(f"wrote {out} ({len(out.read_text().splitlines())} lines)")
