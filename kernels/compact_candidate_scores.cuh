// Experimental source20 publisher scratch. Global row metadata is unchanged;
// only the destination address of retained block maxima becomes compact.
#include <cuda_runtime.h>
#include <stdint.h>
extern "C" __global__ void ca_compact_candidate_score_windows(
 const int32_t* lengths,const int32_t* live,int32_t* compact,int32_t* first,
 int rows,int rank){
 int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=128)return;
 int count=live[0],begin=max(0,count-128),g=begin+slot,owned=rows/4;
 if(slot==0)first[0]=begin;
 bool valid=g<count&&g<rows&&g>=rank*owned&&g<(rank+1)*owned;
 compact[slot]=valid?(lengths[g]+7)/8:0;
}
extern "C" __global__ void ca_candidate_block_scores_compact(
 const float* scores,int stride,const int32_t* starts,const int32_t* lengths,
 float* blocks,int block_stride,int row0,int rows,int nblocks,const int32_t* first){
 int r=blockIdx.y;if(r>=rows)return;
 int g=row0+r,slot=g-first[0],len=lengths[g];
 if(slot<0||slot>=128||len<=0)return;
 int live=min(nblocks,(len+7)/8),last=(len-1)/8;
 for(int b=blockIdx.x*blockDim.x+threadIdx.x;b<live;b+=gridDim.x*blockDim.x){
  float best=-INFINITY;
  if(b==last)best=INFINITY;
  else{
   const float* row=scores+(size_t)r*stride+starts[g]+b*8;
   if(((uintptr_t)row&15)==0){
    float4 a=*(const float4*)row,c=*(const float4*)(row+4);
    float v[8]={a.x,a.y,a.z,a.w,c.x,c.y,c.z,c.w};
    #pragma unroll
    for(int j=0;j<8;j++)best=(v[j]>best||v[j]!=v[j])?v[j]:best;
   }else{
    #pragma unroll
    for(int j=0;j<8;j++){float v=row[j];best=(v>best||v!=v)?v:best;}
   }
  }
  blocks[(size_t)slot*block_stride+b]=best;
 }
}
