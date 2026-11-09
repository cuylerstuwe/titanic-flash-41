// Ordered four-part BF16 row sum for the MXFP8 WO-B exchange: out = ((a + b) + c) + d, where
// every step uses the installed ca_ffn_add arithmetic (FP32 widening, add.rn.ftz.f32, then
// round-to-nearest BF16), exactly as three sequential ca_ffn_add launches. 8 elements/thread.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ __nv_bfloat16 ca_wob8_add_ftz(__nv_bfloat16 x, __nv_bfloat16 y) {
    float a = __bfloat162float(x), b = __bfloat162float(y), s;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(s) : "f"(a), "f"(b));
    return __float2bfloat16_rn(s);
}

extern "C" __global__ void ca_wob8_sum4(
    const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* c,
    const __nv_bfloat16* d, __nv_bfloat16* out, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= n) return;
    uint4 va = *reinterpret_cast<const uint4*>(a + i), vb = *reinterpret_cast<const uint4*>(b + i);
    uint4 vc = *reinterpret_cast<const uint4*>(c + i), vd = *reinterpret_cast<const uint4*>(d + i);
    const __nv_bfloat16* pa = reinterpret_cast<const __nv_bfloat16*>(&va);
    const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&vb);
    const __nv_bfloat16* pc = reinterpret_cast<const __nv_bfloat16*>(&vc);
    const __nv_bfloat16* pd = reinterpret_cast<const __nv_bfloat16*>(&vd);
    uint4 vo;
    __nv_bfloat16* po = reinterpret_cast<__nv_bfloat16*>(&vo);
    #pragma unroll
    for (int k = 0; k < 8; ++k)
        po[k] = ca_wob8_add_ftz(ca_wob8_add_ftz(ca_wob8_add_ftz(pa[k], pb[k]), pc[k]), pd[k]);
    *reinterpret_cast<uint4*>(out + i) = vo;
}
