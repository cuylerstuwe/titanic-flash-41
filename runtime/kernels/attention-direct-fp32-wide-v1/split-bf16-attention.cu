// GPU-only packing/merging for a small BF16 split-key experiment.
// Uses the separately pinned BSD-3-Clause FlashInfer raw attention primitive;
// no FlashInfer engine code, canonical KV bytes or model weights are changed.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
__global__ void pack_parts(const __nv_bfloat16* q,const int* ids,const int* lens,
 const int* extra_ids,const int* extra_lens,const float* sink,__nv_bfloat16* pq,
 int* pi,int* pl,int* pei,int* pel,float* ps,int parts,int extra,int piece) {
 const int r=blockIdx.x/parts,p=blockIdx.x%parts,t=threadIdx.x,pr=blockIdx.x;
 for(int i=t;i<8192;i+=blockDim.x)pq[pr*8192+i]=q[r*8192+i];
 const int main_piece=128/parts;
 for(int i=t;i<128;i+=blockDim.x)pi[pr*128+i]=i<main_piece?ids[r*128+p*main_piece+i]:-1;
 if(t==0)pl[pr]=max(0,min(main_piece,lens[r]-p*main_piece));
 if(extra){
  for(int i=t;i<piece;i+=blockDim.x){int j=p*piece+i;pei[pr*piece+i]=j<extra?extra_ids[r*extra+j]:-1;}
  if(t==0)pel[pr]=max(0,min(piece,extra_lens[r]-p*piece));
 }
 // Each partition receives 1/parts of the original zero-value sink mass.
 // LSE returned by FlashInfer includes this normalizer in log2 units.
 if(pr==0&&t<16)ps[t]=sink[t]-logf((float)parts);
}
#if defined(CANADA_FP32_ATTENTION_PARTIALS)
using partial_t = float;
__device__ __forceinline__ float read_partial(float x) { return x; }
#else
using partial_t = __nv_bfloat16;
__device__ __forceinline__ float read_partial(__nv_bfloat16 x) { return __bfloat162float(x); }
#endif
__global__ void merge_parts(const partial_t* x,const float* lse,__nv_bfloat16* out,
 float* out_lse,int parts){
 const int r=blockIdx.x/16,h=blockIdx.x%16;
 __shared__ float weights[8];
 if(threadIdx.x==0){
  float m=-INFINITY;for(int p=0;p<parts;p++)m=fmaxf(m,lse[(r*parts+p)*16+h]);
  float total=0;for(int p=0;p<parts;p++){float v=exp2f(lse[(r*parts+p)*16+h]-m);weights[p]=v;total+=v;}
  for(int p=0;p<parts;p++)weights[p]/=total;
  out_lse[r*16+h]=m+log2f(total);
 }
 __syncthreads();
 for(int d=threadIdx.x;d<512;d+=blockDim.x){
  float y=0;for(int p=0;p<parts;p++)y=fmaf(weights[p],read_partial(x[((r*parts+p)*16+h)*512+d]),y);
  out[(r*16+h)*512+d]=__float2bfloat16_rn(y);
 }
}
extern "C" int ca_pack_attention_parts(const void* q,const void* ids,const void* lens,
 const void* extra_ids,const void* extra_lens,const void* sink,void* pq,void* pi,void* pl,
 void* pei,void* pel,void* ps,int rows,int parts,int extra,int piece,cudaStream_t stream){
 if(rows<1||rows>16||(parts!=1&&parts!=2&&parts!=4&&parts!=8)||extra<0||extra>1024
  ||(extra&&(piece<64||piece%64||piece*parts<extra)))return -1;
 pack_parts<<<rows*parts,256,0,stream>>>((const __nv_bfloat16*)q,(const int*)ids,(const int*)lens,
  (const int*)extra_ids,(const int*)extra_lens,(const float*)sink,(__nv_bfloat16*)pq,(int*)pi,(int*)pl,
  (int*)pei,(int*)pel,(float*)ps,parts,extra,piece);
 return (int)cudaPeekAtLastError();
}
extern "C" int ca_merge_attention_parts(const void* x,const void* lse,void* out,void* out_lse,
 int rows,int parts,cudaStream_t stream){
 if(rows<1||rows>32||(parts!=1&&parts!=2&&parts!=3&&parts!=4&&parts!=8))return -1;
 merge_parts<<<rows*16,128,0,stream>>>((const partial_t*)x,(const float*)lse,(__nv_bfloat16*)out,(float*)out_lse,parts);
 return (int)cudaPeekAtLastError();
}
