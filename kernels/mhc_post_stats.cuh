// HC=4 residual post-mix fused with the next boundary's mixing statistics: one pass over the residual instead of
// mhc_post (read residual, write the new one) followed by hc_stats_mb16_s1 (read the new residual again).
//
// Post-mix, bit-identical to sglang's _mhc_post_split_h_kernel as exported in mhc-post-v1 (enable_fp_fusion=False):
//   acc = fma(post[c], x, comb[0][c] * r0);  acc = fma(comb[i][c], r_i, acc) for i = 1, 2, 3;  Y_c = bf16_rn(acc)
// with comb[i][c] = C[m * 16 + i * 4 + c] and post[c] = P[m * 4 + c].
//
// Statistics of the new residual, read as the BF16 values just stored (exactly what the separate kernel reads), in
// the partial layout hc_reduce80 consumes: the flattened 4 x 5120 row split into 80 slices of 256 columns,
//   part_mix[s][m][n] = sum_k Y[k] * W[n][k]   (n < 24, k in slice s),   part_sq[s][m] = sum_k Y[k]^2.
// The separate kernel forms these with tf32x3 tensor-core dots; here they are FP32 FMAs in a fixed order (per lane
// over its 8 columns, then a fixed warp butterfly). Deterministic, but not bit-identical to the separate kernel, so
// whatever consumes them differs in the last bits of the mixes.
//
// Launch: grid (ceil(M / 16), 20), block 256 (8 warps), dynamic shared memory 24 * 256 * 4 = 24 KiB.
// Warp w handles rows 16 * blockIdx.x + 2 * w + {0, 1}; lane l handles columns 256 * blockIdx.y + 8 * l .. + 8 of
// each of the four streams. The 24 x 256 weight tile of one stream is staged in shared memory at a time.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define CA_HC_H 5120
#define CA_HC_K 20480
#define CA_HC_MIX 24
#define CA_HC_SLICE 256
#define CA_HC_TILES 20

__device__ __forceinline__ void ca_hc_unpack8(uint4 v, float out[8]) {
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&v);
    #pragma unroll
    for (int k = 0; k < 8; ++k) out[k] = __bfloat162float(p[k]);
}

extern "C" __global__ void __launch_bounds__(256) ca_mhc_post_stats(
    const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ R,
    const float* __restrict__ P, const float* __restrict__ C, const float* __restrict__ W,
    __nv_bfloat16* __restrict__ Y, float* __restrict__ part_mix, float* __restrict__ part_sq, int M) {
    extern __shared__ float wtile[];  // [CA_HC_MIX][CA_HC_SLICE] of one stream's columns
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tile = blockIdx.y;
    const int h0 = tile * CA_HC_SLICE + lane * 8;
    int rows[2];
    rows[0] = blockIdx.x * 16 + warp * 2;
    rows[1] = rows[0] + 1;
    // Post-mix for both rows: y[r][c][k] holds the BF16-rounded outputs as FP32.
    float y[2][4][8];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int m = rows[r];
        if (m >= M) {
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                #pragma unroll
                for (int k = 0; k < 8; ++k) y[r][c][k] = 0.f;
            continue;
        }
        float x[8], res[4][8];
        ca_hc_unpack8(*reinterpret_cast<const uint4*>(X + (size_t)m * CA_HC_H + h0), x);
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            ca_hc_unpack8(*reinterpret_cast<const uint4*>(R + ((size_t)m * 4 + i) * CA_HC_H + h0), res[i]);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float post = P[m * 4 + c];
            const float comb0 = C[m * 16 + 0 * 4 + c], comb1 = C[m * 16 + 1 * 4 + c];
            const float comb2 = C[m * 16 + 2 * 4 + c], comb3 = C[m * 16 + 3 * 4 + c];
            uint4 packed;
            __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(&packed);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float first = __fmul_rn(comb0, res[0][k]);
                float acc = __fmaf_rn(post, x[k], first);
                acc = __fmaf_rn(comb1, res[1][k], acc);
                acc = __fmaf_rn(comb2, res[2][k], acc);
                acc = __fmaf_rn(comb3, res[3][k], acc);
                out[k] = __float2bfloat16_rn(acc);
                y[r][c][k] = __bfloat162float(out[k]);
            }
            *reinterpret_cast<uint4*>(Y + ((size_t)m * 4 + c) * CA_HC_H + h0) = packed;
        }
    }
    // Statistics, one stream (= one 256-column slice) at a time.
    #pragma unroll 1
    for (int c = 0; c < 4; ++c) {
        __syncthreads();
        for (int i = threadIdx.x; i < CA_HC_MIX * CA_HC_SLICE / 4; i += blockDim.x) {
            const int n = i / (CA_HC_SLICE / 4), j = (i % (CA_HC_SLICE / 4)) * 4;
            *reinterpret_cast<float4*>(wtile + n * CA_HC_SLICE + j) =
                *reinterpret_cast<const float4*>(W + (size_t)n * CA_HC_K + c * CA_HC_H + tile * CA_HC_SLICE + j);
        }
        __syncthreads();
        float mix[2][CA_HC_MIX], sq[2];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            sq[r] = 0.f;
            #pragma unroll
            for (int k = 0; k < 8; ++k) sq[r] = __fmaf_rn(y[r][c][k], y[r][c][k], sq[r]);
        }
        #pragma unroll
        for (int n = 0; n < CA_HC_MIX; ++n) {
            const float4 w0 = *reinterpret_cast<const float4*>(wtile + n * CA_HC_SLICE + lane * 8);
            const float4 w1 = *reinterpret_cast<const float4*>(wtile + n * CA_HC_SLICE + lane * 8 + 4);
            const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                float a = 0.f;
                #pragma unroll
                for (int k = 0; k < 8; ++k) a = __fmaf_rn(y[r][c][k], wv[k], a);
                mix[r][n] = a;
            }
        }
        // Fixed butterfly over the 32 lanes: every lane ends with the slice totals.
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                sq[r] = __fadd_rn(sq[r], __shfl_xor_sync(0xffffffff, sq[r], off));
                #pragma unroll
                for (int n = 0; n < CA_HC_MIX; ++n)
                    mix[r][n] = __fadd_rn(mix[r][n], __shfl_xor_sync(0xffffffff, mix[r][n], off));
            }
        }
        const int slice = c * CA_HC_TILES + tile;
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int m = rows[r];
            if (m >= M) continue;
            if (lane < CA_HC_MIX) {
                float v = mix[r][0];
                #pragma unroll
                for (int n = 1; n < CA_HC_MIX; ++n) if (lane == n) v = mix[r][n];
                part_mix[((size_t)slice * M + m) * CA_HC_MIX + lane] = v;
            }
            if (lane == 0) part_sq[(size_t)slice * M + m] = sq[r];
        }
    }
}
