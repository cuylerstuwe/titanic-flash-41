#pragma once
#include <stdint.h>

// Same paged bytes and device-side query bound as ca_gather_index_k_bounded.
// Packing several rows into one warp uses all payload lanes. No floating-point
// operation, cache lifetime or key ordering changes.
template<int Lanes>
__device__ __forceinline__ void ca_index_gather_packed_body(
 const uint8_t* cache,const int32_t* slots,uint32_t* payload,uint32_t* scales,
 int rows,const int32_t* ends,int queries){
 static_assert(Lanes==16 || Lanes==8 || Lanes==4);
 __shared__ int warp_ends[4];
 int lane=threadIdx.x%32,warp=threadIdx.x/32,last=0;
 for(int q=threadIdx.x;q<queries;q+=blockDim.x)last=max(last,ends[q]);
 for(int d=16;d;d>>=1)last=max(last,__shfl_down_sync(0xffffffff,last,d));
 if(lane==0)warp_ends[warp]=last;
 __syncthreads();
 last=min(rows,((max(max(warp_ends[0],warp_ends[1]),max(warp_ends[2],warp_ends[3]))+127)/128)*128);
 constexpr int groups=128/Lanes;
 int sublane=threadIdx.x%Lanes;
 for(int row=blockIdx.x*groups+threadIdx.x/Lanes;row<last;row+=gridDim.x*groups){
  int slot=slots[row];
  const uint8_t* page=cache+(int64_t)(slot/64)*4352;
  const uint8_t* source=page+(slot%64)*64;
  uint8_t* destination=(uint8_t*)payload+(int64_t)row*64;
  if constexpr(Lanes==16)((uint32_t*)destination)[sublane]=((const uint32_t*)source)[sublane];
  if constexpr(Lanes==8)((uint2*)destination)[sublane]=((const uint2*)source)[sublane];
  if constexpr(Lanes==4)((uint4*)destination)[sublane]=((const uint4*)source)[sublane];
  if(sublane==0)scales[row]=((const uint32_t*)(page+4096))[slot%64];
 }
}
#define CA_INDEX_PACKED_ENTRY(N) \
extern "C" __global__ void ca_gather_index_k_packed##N( \
 const uint8_t* cache,const int32_t* slots,uint32_t* payload,uint32_t* scales, \
 int rows,const int32_t* ends,int queries){ \
 ca_index_gather_packed_body<N>(cache,slots,payload,scales,rows,ends,queries); \
}
CA_INDEX_PACKED_ENTRY(16)
CA_INDEX_PACKED_ENTRY(8)
CA_INDEX_PACKED_ENTRY(4)
#undef CA_INDEX_PACKED_ENTRY
