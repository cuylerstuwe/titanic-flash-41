// Original Canada scheduling kernels. Canonical BF16 inputs, FP32 weights,
// accumulation and sigmoid; no target logits or proposal q are modified.
__device__ __forceinline__ float ca_confidence_reduce(float value) {
  __shared__ float partial[8];
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  for(int d=16;d;d>>=1)value+=__shfl_down_sync(0xffffffff,value,d);
  if(lane==0)partial[warp]=value;
  __syncthreads();
  value=warp==0?partial[lane&7]:0.f;
  if(warp==0){if(lane>=8)value=0.f;
    for(int d=16;d;d>>=1)value+=__shfl_down_sync(0xffffffff,value,d);}
  return value;
}
extern "C" __global__ void ca_confidence_fused_full(
 const int32_t* previous,const __nv_bfloat16* embedding,const __nv_bfloat16* hidden,
 const float* weight,__nv_bfloat16* latent,float* features,float* raw,float* probability,
 uint32_t* error,int step) {
  int t=threadIdx.x,id=previous[0];bool valid=id>=0&&id<129280;
  __nv_bfloat16 value=valid?embedding[(size_t)id*256+t]:__ushort_as_bfloat16(0);
  latent[t]=value;if(t==0)*error=valid?0u:1u;
  float sum=0.f;
  for(int i=t;i<5376;i+=256){
    float x=__bfloat162float(i<5120?hidden[step*5120+i]:value);
    features[step*5376+i]=x;sum+=x*weight[i];
  }
  sum=ca_confidence_reduce(sum);
  if(t==0){raw[step]=sum;probability[step]=1.f/(1.f+expf(-sum));}
}
extern "C" __global__ void ca_confidence_hidden_terms(
 const __nv_bfloat16* hidden,const float* weight,float* features,float* terms) {
  int step=blockIdx.x,t=threadIdx.x;float sum=0.f;
  for(int i=t;i<5120;i+=256){float x=__bfloat162float(hidden[step*5120+i]);
    features[step*5376+i]=x;sum+=x*weight[i];}
  sum=ca_confidence_reduce(sum);if(t==0)terms[step]=sum;
}
extern "C" __global__ void ca_confidence_fused_latent(
 const int32_t* previous,const __nv_bfloat16* embedding,const float* weight,
 const float* terms,__nv_bfloat16* latent,float* features,float* raw,float* probability,
 uint32_t* error,int step) {
  int t=threadIdx.x,id=previous[0];bool valid=id>=0&&id<129280;
  __nv_bfloat16 value=valid?embedding[(size_t)id*256+t]:__ushort_as_bfloat16(0);
  latent[t]=value;if(t==0)*error=valid?0u:1u;
  float x=__bfloat162float(value);features[step*5376+5120+t]=x;
  float sum=ca_confidence_reduce(x*weight[5120+t]);
  if(t==0){sum=terms[step]+sum;raw[step]=sum;probability[step]=1.f/(1.f+expf(-sum));}
}
