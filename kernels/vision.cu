// Native V4.1 vision tensor transforms. Rust owns the model and all launches.
// See production/vision-gpu-contract.json for the pinned reference/precision.
#include <cuda_bf16.h>
#include <math.h>
using B = __nv_bfloat16;

extern "C" __global__ void vision_bias(const float* x, const B* bias, B* y,
                                        unsigned count, unsigned width) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) y[i] = __float2bfloat16_rn(x[i] + __bfloat162float(bias[i % width]));
}

extern "C" __global__ void vision_norm_256(const B* x, const B* weight, B* y) {
    __shared__ float partial[8];
    unsigned row = blockIdx.x, t = threadIdx.x;
    float sum = 0.f;
    for (unsigned c = t; c < 1024; c += 256) {
        float v = __bfloat162float(x[row * 1024 + c]);
        sum += v * v;
    }
    for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffff, sum, d);
    if ((t & 31) == 0) partial[t / 32] = sum;
    __syncthreads();
    if (t < 32) {
        sum = t < 8 ? partial[t] : 0.f;
        for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffff, sum, d);
        if (t == 0) partial[0] = rsqrtf(sum * (1.f / 1024.f) + 1.e-6f);
    }
    __syncthreads();
    for (unsigned c = t; c < 1024; c += 256) {
        float v = __bfloat162float(x[row * 1024 + c]) * partial[0];
        y[row * 1024 + c] = __float2bfloat16_rn(v * __bfloat162float(weight[c]));
    }
}

// ATen Reduce.cuh's contiguous float mean at width1024 and >=16 rows:
// four independent contiguous-lane accumulators, followed by the descending
// warp reduction. Matching this order avoids tiny initial differences being
// magnified through the32-block tower. No precision reduction is introduced.
// Adapted from the pinned PyTorch header; see the BSD license/provenance.
extern "C" __global__ void vision_norm(const B* x, const B* weight, B* y) {
    unsigned row = blockIdx.x, lane = threadIdx.x;
    // Retain the existing launch ABI; the unused warps return together.
    if (lane >= 32) return;
    float accum[4] = {0.f, 0.f, 0.f, 0.f};
    for (unsigned c = lane * 4; c < 1024; c += 128) {
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            float v = __bfloat162float(x[row * 1024 + c + j]);
            accum[j] += v * v;
        }
    }
    float sum = ((accum[0] + accum[1]) + accum[2]) + accum[3];
    for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffff, sum, d);
    float inv = rsqrtf(__shfl_sync(0xffffffff, sum, 0) * (1.f / 1024.f) + 1.e-6f);
    for (unsigned c = lane; c < 1024; c += 32) {
        float v = __bfloat162float(x[row * 1024 + c]) * inv;
        y[row * 1024 + c] = __float2bfloat16_rn(v * __bfloat162float(weight[c]));
    }
}

extern "C" __global__ void vision_rope_table(float* co, float* si,
                                               unsigned n, unsigned width) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 32) return;
    unsigned row = i / 32, c = i % 32;
    float frequency = 1.f / powf(10000.f, float(c % 16) / 16.f);
    float position = float(c < 16 ? row / width : row % width);
    float angle = position * frequency;
    co[i] = cosf(angle);
    si[i] = sinf(angle);
}

extern "C" __global__ void vision_split_rope(const B* qkv, const float* co,
                                               const float* si, B* q, B* k, B* v,
                                               unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 1024) return;
    unsigned row = i / 1024, c = i % 1024, half = c % 64;
    unsigned first = c - half + half % 32, pos = row * 32 + half % 32;
    float cosine = co[pos], sine = si[pos];
    float q1 = __bfloat162float(qkv[row * 3072 + first]);
    float q2 = __bfloat162float(qkv[row * 3072 + first + 32]);
    float k1 = __bfloat162float(qkv[row * 3072 + 1024 + first]);
    float k2 = __bfloat162float(qkv[row * 3072 + 1024 + first + 32]);
    q[i] = __float2bfloat16_rn(half < 32 ? q1 * cosine - q2 * sine : q2 * cosine + q1 * sine);
    k[i] = __float2bfloat16_rn(half < 32 ? k1 * cosine - k2 * sine : k2 * cosine + k1 * sine);
    v[i] = qkv[row * 3072 + 2048 + c];
}

extern "C" __global__ void vision_residual(B* x, const B* delta, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) + __bfloat162float(delta[i]));
}

extern "C" __global__ void vision_swiglu(const B* gate_up, B* y, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned row = i / 2816, c = i % 2816;
    float gate = __bfloat162float(gate_up[row * 5632 + c]);
    // F.silu on BF16 returns BF16; its result is rounded before multiplying up.
    B act = __float2bfloat16_rn(gate / (1.f + expf(-gate)));
    y[i] = __float2bfloat16_rn(__bfloat162float(act) * __bfloat162float(gate_up[row * 5632 + 2816 + c]));
}

extern "C" __global__ void vision_unfold(const B* x, B* y, unsigned h,
                                           unsigned w, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned row = i / 9216, col = i % 9216, out_w = (w + 2) / 3;
    unsigned sy = (row / out_w) * 3 + (col % 9) / 3;
    unsigned sx = (row % out_w) * 3 + col % 3;
    // F.unfold layout is channel, dy, dx, with bottom/right zero padding.
    y[i] = sy < h && sx < w ? x[(sy * w + sx) * 1024 + col / 9] : __float2bfloat16_rn(0.f);
}

extern "C" __global__ void vision_gelu(const B* x, B* y, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float v = __bfloat162float(x[i]);
        y[i] = __float2bfloat16_rn((v * 0.5f) * (1.f + erff(v * 0.70710678118654752440f)));
    }
}

extern "C" __global__ void vision_span(const B* features, const B* start,
                                         const B* newline, const B* end, B* out,
                                         unsigned width, unsigned tokens) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * 5120) return;
    unsigned row = i / 5120, col = i % 5120;
    if (row == 0) out[i] = start[col];
    else if (row + 1 == tokens) out[i] = end[col];
    else {
        unsigned r = row - 1, x = r % (width + 1), y = r / (width + 1);
        out[i] = x == width ? newline[col] : features[(y * width + x) * 5120 + col];
    }
}
