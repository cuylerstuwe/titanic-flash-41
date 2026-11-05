// Original GPU-only NVFP4 proposal encoder and FP32 publication.
// Target checkpoint bytes are never modified.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
constexpr int NV_N=32320,NV_K=5120,NV_M=5;
extern "C" __global__ void nvfp4_head_amax(const __nv_bfloat16 *w,float *maximum) {
  float value=0;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<NV_N*NV_K;i+=gridDim.x*blockDim.x)
    value=fmaxf(value,fabsf(__bfloat162float(w[i])));
  for(int k=16;k;k/=2)value=fmaxf(value,__shfl_xor_sync(0xffffffff,value,k));
  __shared__ float parts[8];
  if((threadIdx.x&31)==0)parts[threadIdx.x/32]=value;
  __syncthreads();
  if(threadIdx.x<32){
    value=threadIdx.x<8?parts[threadIdx.x]:0;
    for(int k=16;k;k/=2)value=fmaxf(value,__shfl_xor_sync(0xffffffff,value,k));
    if(threadIdx.x==0)atomicMax((unsigned*)maximum,__float_as_uint(value));
  }
}
extern "C" __global__ void nvfp4_head_quantize(const __nv_bfloat16 *w,const float *maximum,
    uint32_t *q,uint8_t *scales,float *global,int *error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  // The canonical extent is divisible by 256; every subgroup remains full.
  if(i>=NV_N*NV_K)return;
  float x=__bfloat162float(w[i]),local=fabsf(x);
  for(int k=8;k;k/=2)local=fmaxf(local,__shfl_xor_sync(0xffffffff,local,k,16));
  float g=__fdiv_rn(maximum[0],2688.f);
  if(!(g>0)&&threadIdx.x==0)atomicExch(error,1);
  g=g>0?g:1;
  float scale=(float)__nv_fp8_e4m3(__fdiv_rn(local,__fmul_rn(6.f,g)));
  float norm=scale>0?__fdiv_rn(x,__fmul_rn(scale,g)):0;
  float v=fabsf(norm);
  unsigned code=(v>.25f)+(v>=.75f)+(v>1.25f)+(v>=1.75f)+(v>2.5f)+(v>=3.5f)+(v>5.f);
  code|=signbit(norm)?8:0;
  unsigned word=code<<((i%8)*4);
  for(int k=4;k;k/=2)word|=__shfl_xor_sync(0xffffffff,word,k,8);
  int row=i/NV_K,column=i%NV_K;
  if(i%8==0)q[(column/8)*NV_N+row]=word;
  if(i%16==0){
    int j=(column/16)*NV_N+row;
    int r=(j/64)*64+(j%8)*8+(j%64)/8;
    int t=r%4;t=t==1?2:(t==2?1:t);
    int dest=(r/4)*4+t;
    float scaled=__fmul_rn(scale,128.f);
    if(scale>0&&scaled<2.f)atomicExch(error,2);
    uint16_t bits=__half_as_ushort(__float2half_rn(scaled>=2.f?scaled:0));
    scales[dest]=(uint8_t)((uint32_t(bits)<<1)>>8);
  }
  if(i==0)global[0]=__fmul_rn(g,0x1p119f);
}
extern "C" __global__ void nvfp4_head_publish(const __nv_bfloat16 *x,float *y) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<NV_M*NV_N)y[i]=__bfloat162float(x[i]);
}

