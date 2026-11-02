#pragma once
#include <stdint.h>

// Original TP4 index-key partition. Each rank owns every fourth 128-key tile;
// this keeps causal prefixes balanced and retains the qualified MQA tile math.
__device__ __forceinline__ int ca_key_shard_end(int end,int rank){
 return (end/512)*128+min(128,max(0,end%512-rank*128));
}
extern "C" __global__ void ca_key_shard_ranges(const int32_t* ends,
 int32_t* local_ends,int32_t* lengths,int rank,int rows){
 int i=threadIdx.x;if(i<rows){int n=ca_key_shard_end(ends[i],rank);local_ends[i]=n;lengths[i]=n;}
}
// Same validity contract as the qualified cached gather: reset/handoff/restore
// invalidates the output, and the last rows GLOBAL keys remain mutable. Validity
// is published by a separate ordered kernel, never by one of the copy CTAs.
extern "C" __global__ void ca_key_shard_gather(const uint8_t* cache,
 const int32_t* slots,uint32_t* payload,uint32_t* scales,const int32_t* ends,
 const int32_t* valid,int rank,int columns,int rows){
 __shared__ int bounds[2];
 if(threadIdx.x==0){
  int end=0;for(int q=0;q<rows;q++)end=max(end,ends[q]);end=min(end,columns);
  bounds[0]=ca_key_shard_end(max(0,min(*valid,end-rows)),rank);
  bounds[1]=ca_key_shard_end(min(columns,((end+127)/128)*128),rank);
 }
 __syncthreads();
 int lane=threadIdx.x%4;
 for(int row=bounds[0]+blockIdx.x*32+threadIdx.x/4;row<bounds[1];row+=gridDim.x*32){
  int global=((row/128)*4+rank)*128+row%128;
  int slot=slots[global];const uint8_t* page=cache+(int64_t)(slot/64)*4352;
  ((uint4*)(payload+(int64_t)row*16))[lane]=((const uint4*)(page+(slot%64)*64))[lane];
  if(lane==0)scales[row]=((const uint32_t*)(page+4096))[slot%64];
 }
}
extern "C" __global__ void ca_key_shard_advance(const int32_t* ends,int32_t* valid,int columns,int rows){
 if(threadIdx.x==0){int end=0;for(int q=0;q<rows;q++)end=max(end,ends[q]);*valid=max(0,min(columns,end)-rows);}
}
extern "C" __global__ void ca_key_shard_pack(const float* scores,const int32_t* ids,
 const int32_t* ends,uint64_t* payload,int rank,int stride){
 int row=blockIdx.x,i=threadIdx.x,id=ids[row*512+i];float score=-INFINITY;int global=-1;
 if(id>=0&&id<ends[row]){score=scores[(int64_t)row*stride+id];global=((id/128)*4+rank)*128+id%128;}
 payload[row*512+i]=(uint64_t)(uint32_t)global<<32|__float_as_uint(score);
}
extern "C" __global__ void ca_key_shard_unpack(const uint64_t* received,
 float* scores,int32_t* ids,int rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*2048)return;
 int row=i/2048,k=i%2048,rank=k/512,j=k%512;
 uint64_t p=received[(rank*rows+row)*512+j];scores[i]=__uint_as_float((uint32_t)p);ids[i]=(int32_t)(p>>32);
}
extern "C" __global__ void ca_key_shard_remap(const int32_t* selected,
 const int32_t* candidates,int32_t* output,int rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*512)return;
 int id=selected[i];output[i]=id<0?-1:candidates[(i/512)*2048+id];
}
