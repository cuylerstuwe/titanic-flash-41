// Independent fusion of the canonical six-route FP32 sum, BF16 boundary,
// and existing shared-expert BF16/FP32/BF16 addition. Qualified arithmetic component; guarded model integration.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __global__ void ca_ffn_sum_shared(
    const __nv_bfloat16* routes, const __nv_bfloat16* shared,
    __nv_bfloat16* routed_out, __nv_bfloat16* output, int n,int live) {
    int i=blockIdx.x*128+threadIdx.x;
    if(i>=n)return;
    int row=i/5120,col=i%5120;
    __nv_bfloat16 routed=routed_out[i];
    if(row<live) {
    float sum=0.f;
    #pragma unroll
    for(int route=0;route<6;++route)
        sum=__fadd_rn(sum,__bfloat162float(routes[(row*6+route)*5120+col]));
    routed=__float2bfloat16_rn(sum);
    // Retain the observable intermediate buffer for diagnostics/other callers.
    routed_out[i]=routed;
    }
    float a=__bfloat162float(routed),b=__bfloat162float(shared[i]),added;
    // The installed ca_ffn_add is compiled with --use_fast_math (FTZ).
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(added) : "f"(a), "f"(b));
    output[i]=__float2bfloat16_rn(added);
}

// Item 4 (Nov 18 2026; CANADA_SCRATCH_PREFILL_GROUPED_SUM): the canonical six-route sum above (FP32 round-to-nearest
// adds in route order, one BF16 rounding) for one row group of a sequence-parallel prefill chunk, laid out for that
// group's own reduce-scatter. Group `group` of the chunk holds, for every destination rank c, the rows
// c * quarter + group * per .. + per as [c][per][5120], so its reduce-scatter gives rank r its rows of the group
// contiguously, with the same row ownership (and so the same per-element reduction order) as one reduce-scatter of
// the whole chunk. Rows at or past `live` are not written: the topk_sum node leaves padding rows as they are too.
extern "C" __global__ void ca_moe_group_sum_rows(const __nv_bfloat16* routes, __nv_bfloat16* out,
    int quarter, int per, int group, int live) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;  // one thread per 8 columns of one row
    long n = 4L * per * (5120 / 8);
    if (i >= n) return;
    int col = (int)(i % (5120 / 8)) * 8;
    long t = i / (5120 / 8);
    int j = (int)(t % per), c = (int)(t / per);
    int row = c * quarter + group * per + j;
    if (row >= live) return;
    float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int route = 0; route < 6; ++route) {
        uint4 v = *reinterpret_cast<const uint4*>(routes + ((long)row * 6 + route) * 5120 + col);
        const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&v);
        #pragma unroll
        for (int k = 0; k < 8; ++k) sum[k] = __fadd_rn(sum[k], __bfloat162float(h[k]));
    }
    uint4 o;
    __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&o);
    #pragma unroll
    for (int k = 0; k < 8; ++k) oh[k] = __float2bfloat16_rn(sum[k]);
    *reinterpret_cast<uint4*>(out + ((long)c * per + j) * 5120 + col) = o;
}
