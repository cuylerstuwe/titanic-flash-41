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
