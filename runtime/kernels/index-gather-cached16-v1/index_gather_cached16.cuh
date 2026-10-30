#pragma once
#include <stdint.h>

// Exact byte-copy continuation of index_gather_packed.cuh. The caller owns an
// ordered stream, an immutable injective slot map and this output's validity.
// Reset/handoff/restore or another writer must invalidate *valid before replay.
// At most sixteen index rows can change in one sixteen-token verification. Keep that
// suffix dirty even after accepting all proposals; rounded tile padding is
// never declared immutable. This optimization changes no model arithmetic.
extern "C" __global__ void ca_gather_index_k_cached16(
 const uint8_t* cache,const int32_t* slots,uint32_t* payload,uint32_t* scales,
 int rows,const int32_t* ends,int queries,const int32_t* valid){
 __shared__ int warp_ends[4];
 int lane=threadIdx.x%32,warp=threadIdx.x/32,last=0;
 for(int q=threadIdx.x;q<queries;q+=blockDim.x)last=max(last,ends[q]);
 for(int d=16;d;d>>=1)last=max(last,__shfl_down_sync(0xffffffff,last,d));
 if(lane==0)warp_ends[warp]=last;
 __syncthreads();
 int end=min(rows,max(max(warp_ends[0],warp_ends[1]),max(warp_ends[2],warp_ends[3])));
 int first=max(0,min(*valid,max(0,end-16)));
 last=min(rows,((end+127)/128)*128);
 int sublane=threadIdx.x%4;
 for(int row=first+blockIdx.x*32+threadIdx.x/4;row<last;row+=gridDim.x*32){
  int slot=slots[row];const uint8_t* page=cache+(int64_t)(slot/64)*4352;
  const uint8_t* source=page+(slot%64)*64;
  uint8_t* destination=(uint8_t*)payload+(int64_t)row*64;
  ((uint4*)destination)[sublane]=((const uint4*)source)[sublane];
  if(sublane==0)scales[row]=((const uint32_t*)(page+4096))[slot%64];
 }
}
// Separate stream-ordered publication avoids a cross-CTA update/read race.
extern "C" __global__ void ca_index_gather_cache_advance16(
 const int32_t* ends,int queries,int rows,int32_t* valid){
 if(threadIdx.x==0&&blockIdx.x==0){
  int end=0;for(int q=0;q<queries;q++)end=max(end,ends[q]);
  *valid=max(0,min(rows,end)-16);
 }
}
