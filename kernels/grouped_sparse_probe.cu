// Isolated schedule experiment; not part of the serving binary.
// Adapted from our candidate-only gather in sparse_index.cuh. The unchanged
// qualified SM121 MQA library owns FP4 -> FP32 score arithmetic.
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "sparse_index.cuh"

// Exact current consumer control, copied from core.cu at22eee57. Keep its
// original one-thread-per-score launch geometry in the paired measurement.
__global__ void mask_control(float* scores,int stride,const int32_t* starts,
 const int32_t* lengths,const uint8_t* mask,int mask_stride,int mask_rows,int rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*stride)return;
 int r=i/stride,p=i%stride,m=mask_rows+r;if(p>=lengths[m])return;
 if(!mask[(size_t)m*mask_stride+p/8])scores[(size_t)r*stride+starts[m]+p]=-INFINITY;
}
// Visit only live blocks. Each byte covers eight adjacent scores, allowing
// two aligned vector stores; unaligned windows and the final block stay exact.
#include "bounded_candidate_mask.cuh"

extern "C" int ca_mask_probe(int tiles,int rows,int stride,int mask_stride,int row0,
 float* scores,const int32_t* starts,const int32_t* lengths,const uint8_t* mask,cudaStream_t stream){
 if(tiles==0)mask_control<<<(rows*stride+255)/256,256,0,stream>>>(scores,stride,starts,lengths,mask,mask_stride,row0,rows);
 else ca_apply_candidate_mask_bounded<<<dim3(tiles,rows),128,0,stream>>>(scores,stride,starts,lengths,mask,mask_stride,row0);
 return (int)cudaGetLastError();
}

#include "grouped_sparse_index.cuh"

extern "C" int ca_group_prepare(int rows,int columns,int capacity,int cache_capacity,
 const uint8_t* mask,const int32_t* lengths,int32_t* counts,int32_t* ids,
 uint32_t* error,const uint8_t* cache,const int32_t* slots,uint32_t* payload,
 uint32_t* scales,int32_t* ends,cudaStream_t stream){
 int blocks=columns/8,tiles=(blocks+255)/256;
 ca_group_counts<<<tiles,256,0,stream>>>(mask,lengths,counts,blocks,columns,rows,error);
 ca_group_offsets<<<1,32,0,stream>>>(counts,tiles,capacity/8,ends,rows,error);
 ca_group_ids<<<tiles,256,0,stream>>>(mask,lengths,counts,ids,blocks,columns,capacity/8,rows,error);
 ca_sparse6_gather_keys<<<192,128,0,stream>>>(cache,slots,cache_capacity,ids,ends,payload,scales,columns,capacity,error);
 return (int)cudaGetLastError();
}
extern "C" int ca_group_dense_gather(int columns,int capacity,int cache_capacity,
 const int32_t* ids,const int32_t* ends,uint32_t* error,const uint8_t* cache,
 const int32_t* slots,uint32_t* payload,uint32_t* scales,cudaStream_t stream){
 ca_sparse6_gather_keys<<<192,128,0,stream>>>(cache,slots,cache_capacity,ids,ends,payload,scales,columns,capacity,error);
 return (int)cudaGetLastError();
}
extern "C" int ca_group_mask_probe(int rows,int stride,int blocks,int compact,float* scores,
 const int32_t* ids,const int32_t* ends,const int32_t* lengths,const uint8_t* mask,
 int32_t* selection_lengths,cudaStream_t stream){
 ca_group_mask<<<192,256,0,stream>>>(scores,stride,ids,ends,lengths,mask,blocks,rows,compact,selection_lengths);
 return (int)cudaGetLastError();
}
extern "C" int ca_group_remap_probe(int rows,int stride,int32_t* indices,const float* scores,
 const int32_t* ids,const int32_t* ends,const int32_t* lengths,uint32_t* error,cudaStream_t stream){
 ca_group_remap<<<rows,512,0,stream>>>(indices,scores,stride,ids,ends,lengths,error);
 return (int)cudaGetLastError();
}

// Exact current bounded-gather control from core.cu.
extern "C" __global__ void ca_gather_index_k_bounded(const uint8_t* cache,const int32_t* slots,
 uint32_t* payload,uint32_t* scales,int rows,const int32_t* ends,int queries){
 __shared__ int warp_ends[4];
 int lane=threadIdx.x%32,warp=threadIdx.x/32,last=0;
 for(int q=threadIdx.x;q<queries;q+=blockDim.x)last=max(last,ends[q]);
 for(int d=16;d;d>>=1)last=max(last,__shfl_down_sync(0xffffffff,last,d));
 if(lane==0)warp_ends[warp]=last;
 __syncthreads();
 last=min(rows,((max(max(warp_ends[0],warp_ends[1]),max(warp_ends[2],warp_ends[3]))+127)/128)*128);
 for(int row=blockIdx.x*4+warp;row<last;row+=gridDim.x*4){
  int slot=slots[row];const uint8_t* page=cache+(int64_t)(slot/64)*4352;
  if(lane<16)payload[(int64_t)row*16+lane]=((const uint32_t*)(page+(slot%64)*64))[lane];
  if(lane==0)scales[row]=((const uint32_t*)(page+4096))[slot%64];
 }
}
extern "C" int ca_group_native_gather(int columns,int queries,const uint8_t* cache,
 const int32_t* slots,uint32_t* payload,uint32_t* scales,const int32_t* ends,cudaStream_t stream){
 ca_gather_index_k_bounded<<<192,128,0,stream>>>(cache,slots,payload,scales,columns,ends,queries);
 return (int)cudaGetLastError();
}
