#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Adapted numerical contracts, Apache-2.0:
// SGLang f80c91a4b97e52e17fbcff088aa6b47b6270f6b9 silu_and_mul / clamp;
// b12x a7d7d29b2ef8869086e0ceaa787321f17544e3c9 MXFP8RowsQuantLaunch.
// Keep BF16 clamping AND the BF16 activation rounding before quantization.
// Layout: [M,1152] gate/up -> [M,576] BF16 -> [M,640] E4M3, K32 UE8M0.
__device__ __forceinline__ __nv_bfloat162 ca_shared_activation(
    __nv_bfloat162 gate,__nv_bfloat162 up) {
    const __nv_bfloat162 hi=__floats2bfloat162_rn(10.f,10.f);
    const __nv_bfloat162 lo=__floats2bfloat162_rn(-10.f,-10.f);
    gate=__hmin2(gate,hi);up=__hmin2(__hmax2(up,lo),hi);
    float2 g=__bfloat1622float2(gate),u=__bfloat1622float2(up);
    return __floats2bfloat162_rn((g.x/(1.f+__expf(-g.x)))*u.x,
                                (g.y/(1.f+__expf(-g.y)))*u.y);
}
__device__ __forceinline__ float ca_shared_quant_mul(float a,float b) {
    // Quantizer arithmetic is not FTZ even though the pinned activation uses
    // fast math. Keep its tiny-scale behavior separate from activation math.
    float v;asm("mul.rn.f32 %0, %1, %2;":"=f"(v):"f"(a),"f"(b));return v;
}
__device__ __forceinline__ uint32_t ca_shared_pack4(float a,float b,float c,float d) {
    uint32_t v;asm("{.reg .b16 lo,hi; cvt.rn.satfinite.e4m3x2.f32 lo,%2,%1;"
                  "cvt.rn.satfinite.e4m3x2.f32 hi,%4,%3; mov.b32 %0,{lo,hi};}"
                  :"=r"(v):"f"(a),"f"(b),"f"(c),"f"(d));return v;
}
template<int Threads>
__device__ __forceinline__ void ca_shared_swiglu_quant_body(
 const __nv_bfloat16* input,__nv_bfloat16* activation,uint32_t* values,
 uint8_t* scale_rows,uint8_t* scale_mma,int rows) {
    int warp=threadIdx.x/32,lane=threadIdx.x%32,subgroup=lane/8,part=lane%8;
    int task=blockIdx.x*(Threads/32)+warp;
    for(;task<rows*5;task+=gridDim.x*(Threads/32)) {
        int row=task/5,group=(task%5)*4+subgroup,k=group*32+part*4;
        float v[4]={0.f,0.f,0.f,0.f};
        if(k<576) {
            const __nv_bfloat162* gate=(const __nv_bfloat162*)(input+row*1152+k);
            const __nv_bfloat162* up=(const __nv_bfloat162*)(input+row*1152+576+k);
            __nv_bfloat162 a=ca_shared_activation(gate[0],up[0]);
            __nv_bfloat162 b=ca_shared_activation(gate[1],up[1]);
            ((__nv_bfloat162*)(activation+row*576+k))[0]=a;
            ((__nv_bfloat162*)(activation+row*576+k))[1]=b;
            float2 x=__bfloat1622float2(a),y=__bfloat1622float2(b);
            v[0]=x.x;v[1]=x.y;v[2]=y.x;v[3]=y.y;
        }
        float amax=fmaxf(fmaxf(fabsf(v[0]),fabsf(v[1])),fmaxf(fabsf(v[2]),fabsf(v[3])));
        unsigned active=__activemask();
        #pragma unroll
        for(int shift=1;shift<=4;shift*=2)amax=fmaxf(amax,__shfl_xor_sync(active,amax,shift,8));
        // V4.1's native dense activation quantizer floors real K32 groups at
        // 1e-4. The M16 native quantizer uses unit scales for physical padding.
        // The padded FP8 values remain zero; preserve those scale bytes too.
        amax=k<576?fmaxf(amax,1.e-4f):0.f;
        float scale=ca_shared_quant_mul(amax,1.f/448.f);
        uint32_t bits=__float_as_uint(scale);
        if(bits&0x7fffffu)bits=(bits+0x800000u)&0x7f800000u;
        uint32_t exponent=k<576?((bits>>23)&255):127;
        float inverse=0.f;
        if(exponent) {
            float power=float(127-int(exponent));
            asm("ex2.approx.f32 %0,%1;":"=f"(inverse):"f"(power));
        }
        values[row*160+group*8+part]=ca_shared_pack4(
            ca_shared_quant_mul(v[0],inverse),ca_shared_quant_mul(v[1],inverse),
            ca_shared_quant_mul(v[2],inverse),ca_shared_quant_mul(v[3],inverse));
        if(part==0) {
            scale_rows[row*20+group]=uint8_t(exponent);
            scale_mma[row*16+(group%4)+(group/4)*512]=uint8_t(exponent);
        }
    }
}
extern "C" __global__ void ca_shared_swiglu_quant16_128(const __nv_bfloat16* x,
 __nv_bfloat16* a,uint32_t* q,uint8_t* sr,uint8_t* sm,int rows) {
    ca_shared_swiglu_quant_body<128>(x,a,q,sr,sm,rows);
}
extern "C" __global__ void ca_shared_swiglu_quant16_64(const __nv_bfloat16* x,
 __nv_bfloat16* a,uint32_t* q,uint8_t* sr,uint8_t* sm,int rows) {
    ca_shared_swiglu_quant_body<64>(x,a,q,sr,sm,rows);
}
