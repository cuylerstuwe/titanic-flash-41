// Stable candidate unions for 32-row consumer chunks. Source: local s356
// component prototype (0b1f584); canonical MQA/top-k arithmetic is unchanged.
// Integer address work and -infinity masking only. No weight/score quantization.
#pragma once
__device__ bool group_keep(const uint8_t* mask,const int32_t* lengths,
 int b,int blocks,int columns,int rows,uint32_t* error){
 if(b>=blocks)return false;
 bool keep=false;
 for(int r=0;r<rows;r++){
  int n=lengths[r];unsigned v=mask[(size_t)r*blocks+b];
  if(n<0||n>columns||v>1)atomicOr(error,1u);
  keep|=n>0&&8*b<n&&v!=0;
 }
 return keep;
}
extern "C" __global__ void ca_group_counts(const uint8_t* mask,const int32_t* lengths,
 int32_t* counts,int blocks,int columns,int rows,uint32_t* error){
 __shared__ int warp_counts[8];int t=threadIdx.x,b=256*blockIdx.x+t;
 unsigned vote=__ballot_sync(0xffffffff,group_keep(mask,lengths,b,blocks,columns,rows,error));
 if((t&31)==0)warp_counts[t/32]=__popc(vote);
 __syncthreads();
 if(t==0){int n=0;for(int w=0;w<8;w++)n+=warp_counts[w];counts[blockIdx.x]=n;}
}
extern "C" __global__ void ca_group_offsets(int32_t* counts,int tiles,int max_blocks,
 int32_t* ends,int rows,uint32_t* error){
 if(threadIdx.x)return;
 int n=0;for(int i=0;i<tiles;i++){int x=counts[i];counts[i]=n;n+=x;}
 if(n<0||n>max_blocks){atomicOr(error,2u);n=0;}
 counts[tiles]=n;for(int r=0;r<rows;r++)ends[r]=8*n;
}
extern "C" __global__ void ca_group_ids(const uint8_t* mask,const int32_t* lengths,
 const int32_t* counts,int32_t* ids,int blocks,int columns,int max_blocks,
 int rows,uint32_t* error){
 __shared__ int warp_counts[8];int t=threadIdx.x,b=256*blockIdx.x+t,lane=t&31,warp=t/32;
 bool keep=group_keep(mask,lengths,b,blocks,columns,rows,error);
 unsigned vote=__ballot_sync(0xffffffff,keep);
 if(!lane)warp_counts[warp]=__popc(vote);
 __syncthreads();
 if(keep){
  int k=counts[blockIdx.x]+__popc(vote&((1u<<lane)-1u));
  for(int w=0;w<warp;w++)k+=warp_counts[w];
  if(k<max_blocks)ids[k]=b;else atomicOr(error,2u);
 }
}
extern "C" __global__ void ca_group_mask(float* scores,int stride,const int32_t* ids,
 const int32_t* ends,const int32_t* lengths,const uint8_t* mask,
 int blocks,int rows,int compact,int32_t* selection_lengths){
 int tid=blockIdx.x*blockDim.x+threadIdx.x;
 if(tid<rows)selection_lengths[tid]=lengths[tid]>0?(compact?ends[0]:lengths[tid]):0;
 int n=compact?ends[0]:stride;
 for(int i=tid;i<rows*n;i+=gridDim.x*blockDim.x){
  int r=i/n,k=i%n,b=compact?ids[k/8]:k/8,logical=b*8+k%8;
  if(logical>=lengths[r]||!mask[(size_t)r*blocks+b])scores[(size_t)r*stride+k]=-INFINITY;
 }
}
extern "C" __global__ void ca_group_remap(int32_t* indices,const float* scores,int stride,
 const int32_t* ids,const int32_t* ends,const int32_t* lengths,uint32_t* error){
 __shared__ int valid_count;
 int r=blockIdx.x,t=threadIdx.x,k=indices[r*512+t],mapped=-1;
 if(!t)valid_count=0;
 __syncthreads();
 if(k>=0&&k<ends[0]){
  float value=scores[(size_t)r*stride+k];
  if(isnan(value)||value==INFINITY)atomicOr(error,16u);
  else if(value!=-INFINITY){mapped=ids[k/8]*8+k%8;if(mapped>=lengths[r]){atomicOr(error,8u);mapped=-1;}}
 }else if(k!=-1)atomicOr(error,8u);
 unsigned vote=__ballot_sync(0xffffffff,mapped>=0);
 if((t&31)==0)atomicAdd(&valid_count,__popc(vote));
 indices[r*512+t]=mapped;__syncthreads();
 if(!t&&valid_count!=min(lengths[r],512))atomicOr(error,32u);
}


// s549: retain integer union ballots; all pre-existing kernels above are unchanged.
// Every warp independently finds the largest causal length. No monotonic-row
// assumption; padded zero rows may follow live rows in the same group.
__device__ int ca_group_union_max_length(const int32_t* lengths,int rows,int columns,uint32_t* error){
 int lane=threadIdx.x&31,n=lane<rows?lengths[lane]:0;
 if(n<0||n>columns)atomicOr(error,1u);
 for(int d=16;d;d/=2)n=max(n,__shfl_xor_sync(0xffffffff,n,d));
 return n;
}
extern "C" __global__ void ca_group_counts_cached(const uint8_t* mask,const int32_t* lengths,
 int32_t* counts,uint32_t* bits,int blocks,int columns,int rows,uint32_t* error){
 __shared__ int warp_counts[8];int t=threadIdx.x,b=256*blockIdx.x+t;
 int bound=ca_group_union_max_length(lengths,rows,columns,error);
 bool keep=8*b<bound&&group_keep(mask,lengths,b,blocks,columns,rows,error);
 unsigned vote=__ballot_sync(0xffffffff,keep);
 if((t&31)==0){warp_counts[t/32]=__popc(vote);bits[8*blockIdx.x+t/32]=vote;}
 __syncthreads();
 if(t==0){int n=0;for(int w=0;w<8;w++)n+=warp_counts[w];counts[blockIdx.x]=n;}
}
extern "C" __global__ void ca_group_ids_cached(const int32_t* counts,const uint32_t* bits,
 int32_t* ids,int max_blocks,uint32_t* error){
 if(counts[blockIdx.x]==counts[blockIdx.x+1])return;
 __shared__ int warp_counts[8];int t=threadIdx.x,lane=t&31,warp=t/32,b=256*blockIdx.x+t;
 unsigned vote=bits[8*blockIdx.x+warp];
 if(!lane)warp_counts[warp]=__popc(vote);
 __syncthreads();
 if((vote>>lane)&1u){
  int k=counts[blockIdx.x]+__popc(vote&((1u<<lane)-1u));
  for(int w=0;w<warp;w++)k+=warp_counts[w];
  if(k<max_blocks)ids[k]=b;else atomicOr(error,2u);
 }
}
