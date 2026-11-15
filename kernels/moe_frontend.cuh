#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Native routed-expert front-end (Nov 12). It leaves for the exported phase kernels exactly what the exported
// cooperative front-end leaves (b12x a7d7d29 MoEDynamicKernel: grouped routes, split-materialized W4A8, shared
// input, deterministic output), without its four grid barriers:
//   packed / scales      token-major E4M3 input, one UE8M0 scale a K32 block (quantize_block_fp8_mx: scale =
//                        pow2_ceil(amax / 448), payload = cvt.rn.satfinite(value * 2^(127 - scale)); a zero block
//                        has scale byte 0 and a zero payload)
//   expert_tile_base     [experts + 1] prefix of tiles, experts in order; tiles of an expert = ceil(routes / tile_m)
//   task_expert / task_valid_rows   one slot a (tile, group): slot = tile * groups + group
//   token_map / token_weights       physical row (tile * tile_m + row) -> route index (row * topk + slot), weight
// Routes of one expert take its rows in route order (the exported kernel's order is a race; each physical row is
// computed on its own and the per-token sum runs in fixed top-k order, so the result does not depend on it).
// A route outside [0, experts) is inactive, as there. Launch: blocks of 512 or 1,024 threads (whole warps, more
// threads than experts, rows * topk <= 128); block 0 also plans the routes; every block quantizes one K32 block
// a thread.
__device__ __forceinline__ float ca_mf_mul(float a, float b) {
    float v; asm("mul.rn.f32 %0, %1, %2;" : "=f"(v) : "f"(a), "f"(b)); return v;
}
__device__ __forceinline__ uint32_t ca_mf_pack4(float a, float b, float c, float d) {
    uint32_t v; asm("{.reg .b16 lo,hi; cvt.rn.satfinite.e4m3x2.f32 lo,%2,%1;"
                    "cvt.rn.satfinite.e4m3x2.f32 hi,%4,%3; mov.b32 %0,{lo,hi};}"
                    : "=r"(v) : "f"(a), "f"(b), "f"(c), "f"(d)); return v;
}
// ---- Routing inside the front-end (Nov 12) ---------------------------------------------------------------------
// The target's router top-k (SGLang moe_fused_gate._router_triton_kernel as exported in primitive-exports/routing-v1:
// N 384, K 6, sqrt-softplus scores, bias by token, renormalized, scale 1.5, padding rows) with the exported kernel's
// arithmetic, instruction for instruction (routing_vl.ptx): score = sqrt.approx(x > 20 ? x : log1p(exp(x))) with
// libdevice's expf / log1pf sequences as compiled there; selection on score + bias (NaN counts as +inf), highest
// first, lowest expert on ties; weight = the selected experts' unbiased scores divided (div.full) by
// ((w3 + (w2 + (w0 + w1))) + (w4 + w5)) + 1e-20, times 1.5. One warp a row; a row at or past the live count gets
// ids -1 and weights 0. Each lane holds experts 4*lane + c + 128*chunk (c < 4, chunk < 3).
#define CA_RT_F1(name, op) __device__ __forceinline__ float name(float a) { \
    float d; asm volatile(op " %0, %1;" : "=f"(d) : "f"(a)); return d; }
#define CA_RT_F2(name, op) __device__ __forceinline__ float name(float a, float b) { \
    float d; asm volatile(op " %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
#define CA_RT_F3(name, op) __device__ __forceinline__ float name(float a, float b, float c) { \
    float d; asm volatile(op " %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
CA_RT_F3(ca_rt_fma, "fma.rn.ftz.f32")
CA_RT_F3(ca_rt_fma_down, "fma.rm.ftz.f32")
CA_RT_F2(ca_rt_add, "add.f32")
CA_RT_F2(ca_rt_add_rz, "add.rz.ftz.f32")
CA_RT_F2(ca_rt_mul, "mul.f32")
CA_RT_F2(ca_rt_div, "div.full.f32")
CA_RT_F2(ca_rt_max, "max.f32")
CA_RT_F1(ca_rt_neg, "neg.f32")
CA_RT_F1(ca_rt_sat, "cvt.ftz.sat.f32.f32")
CA_RT_F1(ca_rt_ex2, "ex2.approx.ftz.f32")
CA_RT_F1(ca_rt_sqrt, "sqrt.approx.ftz.f32")
__device__ __forceinline__ float ca_rt_from_int(int32_t a) {
    float d; asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(d) : "r"(a)); return d;
}
__device__ __forceinline__ float ca_rt_bits(uint32_t bits) { return __uint_as_float(bits); }
__device__ __forceinline__ float ca_rt_exp(float x) {
    const float t = ca_rt_sat(ca_rt_fma(x, ca_rt_bits(0x3BBB989Du), ca_rt_bits(0x3F000000u)));
    const float n = ca_rt_fma_down(t, ca_rt_bits(0x437C0000u), ca_rt_bits(0x4B400001u));
    const float m = ca_rt_neg(ca_rt_add(n, ca_rt_bits(0xCB40007Fu)));
    const float r = ca_rt_fma(x, ca_rt_bits(0x32A57060u), ca_rt_fma(x, ca_rt_bits(0x3FB8AA3Bu), m));
    return ca_rt_mul(ca_rt_ex2(r), ca_rt_bits(__float_as_uint(n) << 23));
}
__device__ __forceinline__ float ca_rt_log1p(float e) {
    const int32_t bits = __float_as_int(e);
    const int32_t k = (__float_as_int(ca_rt_add_rz(e, ca_rt_bits(0x3F800000u))) + int32_t(0xC0C00000u)) & int32_t(0xFF800000u);
    const float f = ca_rt_add(ca_rt_fma(ca_rt_bits(0x3E800000u), __int_as_float(int32_t(0x40800000) - k), ca_rt_bits(0xBF800000u)),
                              __int_as_float(bits - k));
    const float scaled = ca_rt_mul(ca_rt_from_int(k), ca_rt_bits(0x34000000u));
    float p = ca_rt_fma(ca_rt_bits(0xBD39BF78u), f, ca_rt_bits(0x3DD80012u));
    p = ca_rt_fma(p, f, ca_rt_bits(0xBE0778E0u));
    p = ca_rt_fma(p, f, ca_rt_bits(0x3E146475u));
    p = ca_rt_fma(p, f, ca_rt_bits(0xBE2A68DDu));
    p = ca_rt_fma(p, f, ca_rt_bits(0x3E4CAF9Eu));
    p = ca_rt_fma(p, f, ca_rt_bits(0xBE800042u));
    p = ca_rt_fma(p, f, ca_rt_bits(0x3EAAAAE6u));
    p = ca_rt_fma(p, f, ca_rt_bits(0xBF000000u));
    float result = ca_rt_fma(scaled, ca_rt_bits(0x3F317218u), ca_rt_fma(ca_rt_mul(f, p), f, f));
    if (!(uint32_t(bits) < 0x7F800000u)) {
        if (!(bits < int32_t(0xBF800001u))) result = ca_rt_fma(e, ca_rt_bits(0x7F800000u), ca_rt_bits(0x7F800000u));
        if (e == 0.f) result = ca_rt_bits(0x80000000u);
    }
    return result;
}
// One warp routes one row: `logits` are that row's 384 gate outputs. Every lane returns the six ids and weights.
__device__ __forceinline__ void ca_rt_route_row(const float* logits, const float* bias, int lane, int32_t* out_ids, float* out_weights) {
    float score[12], biased[12];
    #pragma unroll
    for (int chunk = 0; chunk < 3; ++chunk) {
        const float4 x = reinterpret_cast<const float4*>(logits)[chunk * 32 + lane];
        const float4 b = reinterpret_cast<const float4*>(bias)[chunk * 32 + lane];
        const float xs[4] = {x.x, x.y, x.z, x.w}, bs[4] = {b.x, b.y, b.z, b.w};
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float soft = ca_rt_log1p(ca_rt_exp(xs[c]));
            const float s = ca_rt_sqrt(xs[c] > ca_rt_bits(0x41A00000u) ? xs[c] : soft);
            const float v = ca_rt_add(s, bs[c]);
            score[chunk * 4 + c] = s;
            biased[chunk * 4 + c] = v == v ? v : ca_rt_bits(0x7F800000u);
        }
    }
    float w[6];
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        float best = ca_rt_bits(0xFF800000u);
        #pragma unroll
        for (int i = 0; i < 12; ++i) best = ca_rt_max(best, biased[i]);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) best = ca_rt_max(best, __shfl_xor_sync(0xffffffffu, best, offset));
        int32_t winner = 0x7fffffff;
        #pragma unroll
        for (int i = 0; i < 12; ++i) {
            const int32_t id = (i >> 2) * 128 + lane * 4 + (i & 3);
            if (biased[i] == best) winner = min(winner, id);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) winner = min(winner, __shfl_xor_sync(0xffffffffu, winner, offset));
        float mine = 0.f;
        #pragma unroll
        for (int i = 0; i < 12; ++i) {
            const int32_t id = (i >> 2) * 128 + lane * 4 + (i & 3);
            if (id == winner) { mine = score[i]; biased[i] = ca_rt_bits(0xFF800000u); }
        }
        w[k] = __shfl_sync(0xffffffffu, mine, (winner & 127) >> 2);
        out_ids[k] = winner;
    }
    const float total = ca_rt_add(ca_rt_add(ca_rt_add(w[3], ca_rt_add(w[2], ca_rt_add(w[0], w[1]))), ca_rt_add(w[4], w[5])),
                                  ca_rt_bits(0x1E3CE508u));
    #pragma unroll
    for (int k = 0; k < 6; ++k) out_weights[k] = ca_rt_mul(ca_rt_div(w[k], total), ca_rt_bits(0x3FC00000u));
}
// The routing of every row of a block of at most 16 warps: warp r routes row r into ids / weights [rows x 6]
// (rows = the owner's rows; the first min(rows, live[0]) are live), then padding rows copy their source row's
// experts with zero weights (ca_route_padding_rows; `padding` may be null). Ends with the block synchronized.
__device__ __forceinline__ void ca_rt_route_block(const float* logits, const float* bias, const float* bias_alt,
    const int64_t* tokens, const int32_t* live, const int32_t* padding, int32_t* ids, float* weights, int rows) {
    const int lane = threadIdx.x & 31, row = threadIdx.x >> 5;
    if (row < rows) {
        const bool is_live = row < min(rows, live[0]);
        int32_t chosen[6]; float share[6];
        if (is_live) ca_rt_route_row(logits + size_t(row) * 384, tokens[row] == 129264 ? bias_alt : bias, lane, chosen, share);
        if (lane < 6) {
            ids[row * 6 + lane] = is_live ? chosen[lane] : -1;
            weights[row * 6 + lane] = is_live ? share[lane] : 0.f;
        }
    }
    __syncthreads();
    if (padding != nullptr && int(threadIdx.x) < rows * 6) {
        const int r = threadIdx.x / 6, s = padding[r];
        if (s >= 0) { ids[threadIdx.x] = ids[s * 6 + (threadIdx.x - r * 6)]; weights[threadIdx.x] = 0.f; }
    }
    __syncthreads();
}
// The router's top-k alone (component probe; grid 1, block 512).
extern "C" __global__ void ca_route_topk(const float* logits, const float* bias, const float* bias_alt,
    const int64_t* tokens, const int32_t* live, const int32_t* padding, int32_t* ids, float* weights, int32_t rows) {
    ca_rt_route_block(logits, bias, bias_alt, tokens, live, padding, ids, weights, rows);
}
__device__ __forceinline__ void ca_mf_run(
    const uint4* input, const int32_t* ids, const float* weights, uint32_t* packed, uint8_t* scales,
    int32_t* token_map, float* token_weights, int32_t* task_expert, int32_t* task_valid_rows,
    int32_t* expert_tile_base, int32_t rows, int32_t topk, int32_t experts, int32_t cols, int32_t groups,
    int32_t tile_m, int first) {
    const int t = threadIdx.x, threads = blockDim.x;
    if (blockIdx.x == 0) {
        __shared__ int32_t s_ids[128];
        __shared__ int32_t s_base[512];
        __shared__ int32_t s_warp[32];
        const int pairs = rows * topk;
        if (t < pairs) s_ids[t] = ids[t];
        __syncthreads();
        int count = 0;
        if (t < experts) for (int p = 0; p < pairs; ++p) count += s_ids[p] == t;
        const int tiles = (count + tile_m - 1) / tile_m, lane = t & 31, warp = t >> 5;
        int prefix = tiles;
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int below = __shfl_up_sync(0xffffffffu, prefix, offset);
            if (lane >= offset) prefix += below;
        }
        if (lane == 31) s_warp[warp] = prefix;
        __syncthreads();
        int base = prefix - tiles;
        for (int w = 0; w < warp; ++w) base += s_warp[w];
        if (t <= experts) expert_tile_base[t] = base;
        if (t < experts) {
            s_base[t] = base;
            for (int k = 0; k < tiles; ++k) {
                const int valid = min(tile_m, count - k * tile_m);
                for (int g = 0; g < groups; ++g) {
                    task_expert[(base + k) * groups + g] = t;
                    task_valid_rows[(base + k) * groups + g] = valid;
                }
            }
        }
        __syncthreads();
        if (t < pairs) {
            const int expert = s_ids[t];
            if (expert >= 0 && expert < experts) {
                int rank = 0;
                for (int p = 0; p < t; ++p) rank += s_ids[p] == expert;
                const int row = (s_base[expert] + rank / tile_m) * tile_m + rank % tile_m;
                token_map[row] = t;
                token_weights[row] = weights[t];
            }
        }
    }
    // Blocks from `first` on quantize (first = 1: block 0 only routes and plans, so the kernel is as long as the
    // longer of the two, not their sum).
    const int total = rows * (cols >> 5), quantizers = int(gridDim.x) - first;
    if (int(blockIdx.x) < first) return;
    for (int q = (int(blockIdx.x) - first) * threads + t; q < total; q += quantizers * threads) {
        const uint4* source = input + size_t(q) * 4;
        float v[32];
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint4 w = source[i];
            const uint32_t word[4] = {w.x, w.y, w.z, w.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float a = __uint_as_float((word[j] & 0xffffu) << 16);
                const float b = __uint_as_float(word[j] & 0xffff0000u);
                v[i * 8 + j * 2] = a; v[i * 8 + j * 2 + 1] = b;
                amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
            }
        }
        uint32_t bits = __float_as_uint(ca_mf_mul(amax, float(1.0 / 448.0)));
        if (bits & 0x7fffffu) bits = (bits + 0x800000u) & 0x7f800000u;
        const uint32_t exponent = (bits >> 23) & 255u;
        float inverse = 0.f;
        if (exponent) {
            const float power = float(127 - int(exponent));
            asm("ex2.approx.f32 %0,%1;" : "=f"(inverse) : "f"(power));
        }
        uint32_t* out = packed + size_t(q) * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            out[i] = ca_mf_pack4(ca_mf_mul(v[i * 4], inverse), ca_mf_mul(v[i * 4 + 1], inverse),
                                 ca_mf_mul(v[i * 4 + 2], inverse), ca_mf_mul(v[i * 4 + 3], inverse));
        scales[q] = uint8_t(exponent);
    }
}
extern "C" __global__ void ca_moe_frontend(
    const uint4* input, const int32_t* ids, const float* weights, uint32_t* packed, uint8_t* scales,
    int32_t* token_map, float* token_weights, int32_t* task_expert, int32_t* task_valid_rows,
    int32_t* expert_tile_base, int32_t rows, int32_t topk, int32_t experts, int32_t cols, int32_t groups,
    int32_t tile_m) {
    ca_mf_run(input, ids, weights, packed, scales, token_map, token_weights, task_expert, task_valid_rows,
              expert_tile_base, rows, topk, experts, cols, groups, tile_m, 0);
}
// The front-end with the router's top-k inside (blocks of 512 threads; 384 experts, top 6, at most 16 owner
// rows): block 0 first routes every owner row from the gate's logits (and applies padding routes), writing the
// same ids and weights the router kernel and ca_route_padding_rows leave, then plans the `rows` executed rows as
// ca_moe_frontend does; the other blocks quantize meanwhile (launch one block more than ca_moe_frontend's grid:
// block 0 does not quantize). `live` holds the live row count.
extern "C" __global__ void ca_moe_route_frontend(
    const uint4* input, int32_t* ids, float* weights, uint32_t* packed, uint8_t* scales,
    int32_t* token_map, float* token_weights, int32_t* task_expert, int32_t* task_valid_rows,
    int32_t* expert_tile_base, const float* logits, const float* bias, const float* bias_alt,
    const int64_t* tokens, const int32_t* live, const int32_t* padding,
    int32_t rows, int32_t topk, int32_t experts, int32_t cols, int32_t groups, int32_t tile_m, int32_t owner_rows) {
    if (blockIdx.x == 0) ca_rt_route_block(logits, bias, bias_alt, tokens, live, padding, ids, weights, owner_rows);
    ca_mf_run(input, ids, weights, packed, scales, token_map, token_weights, task_expert, task_valid_rows,
              expert_tile_base, rows, topk, experts, cols, groups, tile_m, 1);
}
