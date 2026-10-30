// One E4M3 row, its routed-expert scales, and one shared-rounding correction
// bit per value. Preserve both existing quantizers; do not approximate either.
// Experimental fast paths. Preserve the original row format and quantizers.
// Fused quantizer scale formulas adapted from b12x a7d7d29b2ef8869086e0ceaa787321f17544e3c9,
// b12x_next/_lib/quant/mxfp8_rows.py and _lib/intrinsics.py (Apache-2.0):
// https://github.com/local-inference-lab/b12x
// No model, runtime, sampler or canonical weight changes.
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

__device__ __forceinline__ float f8(uint8_t bits) {
    __nv_fp8_e4m3 value; value.__x = bits; return float(value);
}
__device__ __forceinline__ uint8_t base_shared(uint8_t value, uint8_t scale,
                                             float &scaled) {
    const int shared_scale = max(int(scale), 105); // ceil_pow2(1e-4 / 448)
    // This power is normal FP32: exponent difference is in [-105,0].
    scaled = f8(value) * __uint_as_float((int(scale) - shared_scale + 127) << 23);
    return __nv_fp8_e4m3(scaled).__x;
}

__global__ void encode(const uint8_t *routed, const uint8_t *routed_scales,
                       const uint8_t *shared, uint8_t *packed, int rows, int k) {
    const int lane = threadIdx.x % 32;
    const int groups = k / 32, stride = k + groups * 5;
    for (int group = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
         group < rows * groups; group += gridDim.x * blockDim.x / 32) {
        const int row = group / groups, col = group % groups;
        const uint8_t q = routed[group * 32 + lane], sf = routed_scales[group];
        unsigned correction = 0;
        if (sf < 105) {
            float scaled;
            const uint8_t base = base_shared(q, sf, scaled);
            correction = __ballot_sync(0xffffffffu, base != shared[group * 32 + lane]);
        }
        packed[row * stride + col * 32 + lane] = q;
        if (lane == 0) {
            packed[row * stride + k + col] = sf;
            reinterpret_cast<unsigned *>(packed + row * stride + k + groups)[col] = correction;
        }
    }
}

__global__ void decode_shared(const uint8_t *packed, uint8_t *shared,
                              uint8_t *row_scales, uint8_t *mma_scales,
                              int rows, int k) {
    const int lane = threadIdx.x % 32;
    const int groups = k / 32, stride = k + groups * 5;
    for (int group = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
         group < rows * groups; group += gridDim.x * blockDim.x / 32) {
        const int row = group / groups, col = group % groups;
        const uint8_t q = packed[row * stride + col * 32 + lane];
        const uint8_t sf = packed[row * stride + k + col];
        uint8_t base = q;
        if (sf < 105) {
            const unsigned correction = reinterpret_cast<const unsigned *>(
                packed + row * stride + k + groups)[col];
            float scaled;
            base = base_shared(q, sf, scaled);
            if ((correction >> lane) & 1u)
                base += fabsf(scaled) > fabsf(f8(base)) ? 1 : -1;
        }
        shared[group * 32 + lane] = base;
        if (lane == 0) {
            const uint8_t shared_sf = max(int(sf), 105);
            row_scales[group] = shared_sf;
            const int offset = (row % 32) * 16 + ((row / 32) % 4) * 4
                + (row / 128) * (((groups + 3) / 4) * 512)
                + (col % 4) + (col / 4) * 512;
            mma_scales[offset] = shared_sf;
        }
    }
}

extern "C" int ca_encode(const uint8_t *routed, const uint8_t *scales,
                          const uint8_t *shared, uint8_t *packed,
                          int rows, int k, cudaStream_t stream) {
    if (rows <= 0 || k <= 0 || k % 128) return -1;
    encode<<<192, 256, 0, stream>>>(routed, scales, shared, packed, rows, k);
    return int(cudaGetLastError());
}
extern "C" int ca_decode(const uint8_t *packed, uint8_t *shared,
                          uint8_t *row_scales, uint8_t *mma_scales,
                          int rows, int k, cudaStream_t stream) {
    if (rows <= 0 || rows % 128 || k <= 0 || k % 128) return -1;
    decode_shared<<<192, 256, 0, stream>>>(packed, shared, row_scales, mma_scales, rows, k);
    return int(cudaGetLastError());
}

__global__ void local_live(const int *global, int *local, int rows, int rank) {
    *local = min(rows, max(0, *global - rank * rows));
}
extern "C" int ca_local_live(const int *global, int *local, int rows,
                              int rank, cudaStream_t stream) {
    if (rows <= 0 || rank < 0 || rank >= 4) return -1;
    local_live<<<1,1,0,stream>>>(global, local, rows, rank);
    return int(cudaGetLastError());
}

// The routed 8-lane producer maps a zero amax to UE8M0 byte zero.
// The shared 4-lane producer applies min_amax=1e-4 before the same rule.
__device__ __forceinline__ uint8_t quant_scale(float amax) {
    unsigned bits = __float_as_uint(amax * (1.0f / 448.0f));
    if (bits & 0x007fffffu) bits = (bits + 0x00800000u) & 0x7f800000u;
    return (bits >> 23) & 255;
}
__device__ __forceinline__ float inverse_scale(uint8_t sf) {
    return sf ? __uint_as_float((254 - int(sf)) << 23) : 0.0f;
}
__global__ void fused_pack(const __nv_bfloat16 *input, uint8_t *packed, int rows, int k) {
    const int lane = threadIdx.x % 32;
    const int groups = k / 32, stride = k + groups * 5;
    for (int group = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
         group < rows * groups; group += gridDim.x * blockDim.x / 32) {
        const int row = group / groups, col = group % groups;
        const float x = __bfloat162float(input[group * 32 + lane]);
        float amax = fabsf(x);
        for (int shift = 16; shift; shift /= 2)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, shift));
        const uint8_t sf = quant_scale(amax);
        const uint8_t q = __nv_fp8_e4m3(x * inverse_scale(sf)).__x;
        unsigned correction = 0;
        if (sf < 105) {
            const uint8_t shared_sf = quant_scale(fmaxf(amax, 1e-4f));
            const uint8_t shared = __nv_fp8_e4m3(x * inverse_scale(shared_sf)).__x;
            float scaled;
            correction = __ballot_sync(0xffffffffu, base_shared(q,sf,scaled) != shared);
        }
        packed[row*stride+col*32+lane] = q;
        if (lane == 0) {
            packed[row*stride+k+col] = sf;
            reinterpret_cast<unsigned *>(packed+row*stride+k+groups)[col] = correction;
        }
    }
}
extern "C" int ca_pack_bf16(const __nv_bfloat16 *input, uint8_t *packed,
                             int rows, int k, cudaStream_t stream) {
    if (rows <= 0 || k <= 0 || k % 128) return -1;
    fused_pack<<<192,256,0,stream>>>(input,packed,rows,k);
    return int(cudaGetLastError());
}
