#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
// Finite component experiment: exact key scores plus canonical eight-key block
// maxima. Every fourth 128-key tile belongs to one rank; no block is split.
extern "C" __global__ void ca_key_publisher_block_scores(int rank,int stride,const float* scores,
 const int32_t* local_ends,const int32_t* global_ends,float* blocks,int32_t* lengths){
 int r=blockIdx.y,live=(local_ends[r]+7)/8,last=(global_ends[r]-1)/8;
 if(blockIdx.x==0&&threadIdx.x==0)lengths[r]=live;
 for(int b=blockIdx.x*256+threadIdx.x;b<live;b+=gridDim.x*256){
  int global=rank<0?b:((b/16)*4+rank)*16+b%16;float best=-INFINITY;
  if(global==last)best=INFINITY;
  else{const float* row=scores+(int64_t)r*stride+b*8;
   #pragma unroll
   for(int j=0;j<8;j++){float v=row[j];best=(v>best||v!=v)?v:best;}}
  blocks[(int64_t)r*(stride/8)+b]=best;
 }
}
extern "C" __global__ void ca_key_publisher_pack_selected(int rank,int stride,const float* scores,const int32_t* ids,
 const int32_t* ends,const float* blocks,const int32_t* selected_blocks,
 const int32_t* block_ends,uint64_t* payload){
 int r=blockIdx.x;
 for(int i=threadIdx.x;i<2560;i+=blockDim.x){
  bool block=i>=512;int j=block?i-512:i;
  int id=block?selected_blocks[r*2048+j]:ids[r*512+j];
  int end=block?block_ends[r]:ends[r];float score=-INFINITY;int global=-1;
  if(id>=0&&id<end){
   int tile=block?16:128;global=((id/tile)*4+rank)*tile+id%tile;
   score=block?blocks[(int64_t)r*(stride/8)+id]:scores[(int64_t)r*stride+id];
  }
  payload[r*2560+i]=(uint64_t)(uint32_t)global<<32|__float_as_uint(score);
 }
}
extern "C" __global__ void ca_key_publisher_unpack_candidates(const uint64_t* received,float* keys,int32_t* key_ids,
 float* blocks,int32_t* block_ids,int rows){
 int i=blockIdx.x*256+threadIdx.x;if(i>=rows*10240)return;
 int r=i/10240,k=i%10240,rank=k/2560,j=k%2560;
 uint64_t p=received[(rank*rows+r)*2560+j];
 if(j<512){int at=r*2048+rank*512+j;keys[at]=__uint_as_float((uint32_t)p);key_ids[at]=(int32_t)(p>>32);}
 else{int at=r*8192+rank*2048+j-512;blocks[at]=__uint_as_float((uint32_t)p);block_ids[at]=(int32_t)(p>>32);}
}
extern "C" __global__ void ca_key_publisher_remap_selected(const int32_t* chosen,const int32_t* candidates,
 int32_t* output,int rows,int topk){
 int i=blockIdx.x*256+threadIdx.x;if(i>=rows*topk)return;
 int id=chosen[i];output[i]=id<0?-1:candidates[(i/topk)*(4*topk)+id];
}
// The merge sees 8192 candidates even for short rows. Its invalid padding
// need not be a contiguous suffix, so inspect all2048 output slots. Pack has
// already replaced every noncausal local candidate with global ID-1.
extern "C" __global__ void ca_key_publisher_make_mask(const int32_t* chosen,
 uint8_t* mask,int rows,int stride){
 int i=blockIdx.x*256+threadIdx.x;if(i>=rows*2048)return;
 int id=chosen[i];if(id>=0&&id<stride)mask[(i/2048)*stride+id]=1;
}
