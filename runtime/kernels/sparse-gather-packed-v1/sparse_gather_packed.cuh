#pragma once
#include <stdint.h>

// Four lanes copy one64-byte key from the stable ascending block union.
// Preserve ca_sparse6_gather_keys' raw layout, padded zeros and error bit.
extern "C" __global__ void ca_sparse6_gather_keys_packed4(const uint8_t* cache,
 const int32_t* slots,int cache_capacity,const int32_t* ids,const int32_t* ends,
 uint32_t* payload,uint32_t* scales,int columns,int key_capacity,uint32_t* error){
 int lane=threadIdx.x&3,group=threadIdx.x/4,n=ends[0];
 int bound=min(key_capacity,((n+127)/128)*128);
 for(int row=blockIdx.x*32+group;row<bound;row+=gridDim.x*32){
  int logical=row<n?ids[row/8]*8+row%8:-1;
  int slot=(logical>=0&&logical<columns)?slots[logical]:-1;
  bool valid=row<n&&logical>=0&&logical<columns&&slot>=0&&slot<cache_capacity;
  if(row<n&&!valid&&lane==0)atomicOr(error,4u);
  const uint8_t* page=valid?cache+(int64_t)(slot/64)*4352:cache;
  uint4 value=make_uint4(0u,0u,0u,0u);
  if(valid)value=((const uint4*)(page+(slot%64)*64))[lane];
  ((uint4*)payload)[(size_t)row*4+lane]=value;
  if(lane==0)scales[row]=valid?((const uint32_t*)(page+4096))[slot%64]:0u;
 }
}
