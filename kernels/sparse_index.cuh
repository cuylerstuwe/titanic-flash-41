// Candidate-only index scoring for the six-row decode profile.
// Integer gather/scatter; the SM121 FP4-to-FP32 MQA kernel performs the scoring.
// Stable ascending union order preserves absolute-index tie handling.
__device__ __forceinline__ bool ca_sparse6_keep(const uint8_t* mask,
 const int32_t* lengths,int b,int blocks,int columns,uint32_t* error){
 bool keep=false;
 if(b>=blocks)return false;
 for(int r=0;r<6;r++){
  int n=lengths[r];uint8_t v=mask[(size_t)r*blocks+b];
  if(n<0||n>columns||v>1)atomicOr(error,1u);
  keep|=(n>0&&b*8<n&&v!=0);
 }
 return keep;
}
extern "C" __global__ void ca_sparse6_union_counts(const uint8_t* mask,
 const int32_t* lengths,int32_t* counts,int blocks,int columns,uint32_t* error){
 __shared__ int32_t warp_counts[8];
 int t=threadIdx.x,b=blockIdx.x*256+t;
 unsigned ballot=__ballot_sync(0xffffffff,ca_sparse6_keep(mask,lengths,b,blocks,columns,error));
 if((t&31)==0)warp_counts[t/32]=__popc(ballot);
 __syncthreads();
 if(t==0){int n=0;for(int w=0;w<8;w++)n+=warp_counts[w];counts[blockIdx.x]=n;}
}
extern "C" __global__ void ca_sparse6_union_offsets(int32_t* counts,int tiles,
 int max_blocks,int32_t* ends,uint32_t* error){
 if(threadIdx.x!=0)return;
 int n=0;
 for(int i=0;i<tiles;i++){int x=counts[i];counts[i]=n;n+=x;}
 if(n<0||n>max_blocks){atomicOr(error,2u);n=0;}
 counts[tiles]=n;
 for(int r=0;r<6;r++)ends[r]=n*8;
}
extern "C" __global__ void ca_sparse6_union_ids(const uint8_t* mask,
 const int32_t* lengths,const int32_t* counts,int32_t* ids,int blocks,
 int columns,int max_blocks,uint32_t* error){
 __shared__ int32_t warp_counts[8];
 int t=threadIdx.x,b=blockIdx.x*256+t,lane=t&31,warp=t/32;
 bool keep=ca_sparse6_keep(mask,lengths,b,blocks,columns,error);
 unsigned ballot=__ballot_sync(0xffffffff,keep);
 if(lane==0)warp_counts[warp]=__popc(ballot);
 __syncthreads();
 if(keep){
  int index=counts[blockIdx.x]+__popc(ballot&((1u<<lane)-1u));
  for(int w=0;w<warp;w++)index+=warp_counts[w];
  if(index<max_blocks)ids[index]=b;else atomicOr(error,2u);
 }
}
extern "C" __global__ void ca_sparse6_gather_keys(const uint8_t* cache,
 const int32_t* slots,int cache_capacity,const int32_t* ids,const int32_t* ends,
 uint32_t* payload,uint32_t* scales,int columns,int key_capacity,uint32_t* error){
 int lane=threadIdx.x&31,warp=threadIdx.x/32,n=ends[0];
 int bound=min(key_capacity,((n+127)/128)*128);
 for(int row=blockIdx.x*4+warp;row<bound;row+=gridDim.x*4){
  int logical=row<n?ids[row/8]*8+row%8:-1;
  int slot=(logical>=0&&logical<columns)?slots[logical]:-1;
  bool valid=row<n&&logical>=0&&logical<columns&&slot>=0&&slot<cache_capacity;
  if(row<n&&!valid&&lane==0)atomicOr(error,4u);
  const uint8_t* page=valid?cache+(int64_t)(slot/64)*4352:cache;
  if(lane<16)payload[(size_t)row*16+lane]=valid?((const uint32_t*)(page+(slot%64)*64))[lane]:0u;
  if(lane==0)scales[row]=valid?((const uint32_t*)(page+4096))[slot%64]:0u;
 }
}
extern "C" __global__ void ca_sparse6_fill_scores(float* scores,int count){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)scores[i]=-INFINITY;
}
extern "C" __global__ void ca_sparse6_scatter_scores(const float* compact,
 int compact_stride,float* scores,int stride,const int32_t* ids,
 const int32_t* ends,const int32_t* lengths,const uint8_t* mask,int blocks,int columns){
 int n=ends[0];
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<6*n;i+=gridDim.x*blockDim.x){
  int r=i/n,k=i%n,b=ids[k/8],logical=b*8+k%8;
  if(logical<columns&&logical<lengths[r]&&mask[(size_t)r*blocks+b])
   scores[(size_t)r*stride+logical]=compact[(size_t)r*compact_stride+k];
 }
}
// Reuse the downstream candidate mapper's existing error/publication barrier.
// -2 is deliberately invalid; -1 would merely hide an internal failure.
extern "C" __global__ void ca_sparse6_poison_selection(const uint32_t* error,int32_t* indices){
 if(threadIdx.x==0&&*error)indices[0]=-2;
}
// Experimental component path: retain the existing FP32 scores in compact
// union order. Mask row-ineligible keys before the unchanged exact selector.
extern "C" __global__ void ca_sparse6_mask_compact(float* scores,int stride,
 const int32_t* ids,const int32_t* ends,const int32_t* lengths,
 const uint8_t* mask,int blocks,int columns,int32_t* compact_lengths){
 int tid=blockIdx.x*blockDim.x+threadIdx.x,n=ends[0];
 if(tid<6)compact_lengths[tid]=lengths[tid]>0?n:0;
 for(int i=tid;i<6*n;i+=gridDim.x*blockDim.x){
  int r=i/n,k=i%n,b=ids[k/8],logical=b*8+k%8;
  if(logical>=columns||logical>=lengths[r]||!mask[(size_t)r*blocks+b])
   scores[(size_t)r*stride+k]=-INFINITY;
 }
}
// Map before absolute-index sorting. Canonical source20 must leave at least
// min(row_length,512) eligible keys; fail closed if that contract is violated.
extern "C" __global__ void ca_sparse6_remap_selection(int32_t* indices,
 const float* scores,int stride,const int32_t* ids,const int32_t* ends,
 const int32_t* lengths,uint32_t* error){
 __shared__ int valid_count;
 int r=blockIdx.x,t=threadIdx.x,n=ends[0],k=indices[r*512+t],mapped=-1;
 if(t==0)valid_count=0;
 __syncthreads();
 if(k>=0){
  if(k>=n){atomicOr(error,8u);}
  else {
   float value=scores[(size_t)r*stride+k];
   if(isnan(value)||value==INFINITY)atomicOr(error,16u);
   else if(value!=-INFINITY){
    mapped=ids[k/8]*8+k%8;
    if(mapped<0||mapped>=lengths[r]){atomicOr(error,8u);mapped=-1;}
   }
  }
 }else if(k!=-1)atomicOr(error,8u);
 unsigned vote=__ballot_sync(0xffffffff,mapped>=0);
 if((t&31)==0)atomicAdd(&valid_count,__popc(vote));
 indices[r*512+t]=mapped;
 __syncthreads();
 if(t==0&&valid_count!=min(lengths[r],512))atomicOr(error,32u);
}

// Untimed diagnostic of the existing source20 masks. Model buffers are read
// only; all mask reductions execute on GPU. Output: union, pair count,
// non-binary cells, out-of-window selections, then one count per live row.
extern "C" __global__ void ca_sparse_mask_union_stats(const uint8_t* mask,
 int blocks,int live,int start,uint32_t* stats){
 int b=blockIdx.x*blockDim.x+threadIdx.x;
 bool any=false;unsigned pairs=0,bad=0,outside=0;
 if(b<blocks){
  for(int r=0;r<live;r++){
   unsigned v=mask[(size_t)r*blocks+b];
   bool eligible=b*8<start+r+1;
   bad+=v>1;outside+=(v!=0&&!eligible);
   bool keep=v!=0&&eligible;any|=keep;pairs+=keep;
  }
 }
 unsigned total=any;
 for(int offset=16;offset;offset/=2){
  total+=__shfl_down_sync(0xffffffff,total,offset);
  pairs+=__shfl_down_sync(0xffffffff,pairs,offset);
  bad+=__shfl_down_sync(0xffffffff,bad,offset);
  outside+=__shfl_down_sync(0xffffffff,outside,offset);
 }
 if((threadIdx.x&31)==0){
  atomicAdd(stats,total);atomicAdd(stats+1,pairs);
  atomicAdd(stats+2,bad);atomicAdd(stats+3,outside);
 }
}
extern "C" __global__ void ca_sparse_mask_row_stats(const uint8_t* mask,
 int blocks,int live,int start,uint32_t* stats){
 int r=blockIdx.x;unsigned n=0;
 if(r>=live)return;
 for(int b=threadIdx.x;b<blocks;b+=blockDim.x)
  n+=mask[(size_t)r*blocks+b]!=0&&b*8<start+r+1;
 for(int offset=16;offset;offset/=2)n+=__shfl_down_sync(0xffffffff,n,offset);
 if((threadIdx.x&31)==0)atomicAdd(stats+4+r,n);
}
extern "C" __global__ void ca_sparse_mask_group_unions(const uint8_t* mask,
 int blocks,int live,int start,int group_rows,uint32_t* unions){
 int b=blockIdx.x*blockDim.x+threadIdx.x,first=blockIdx.y*group_rows;
 bool any=false;
 if(b<blocks)for(int r=first;r<min(live,first+group_rows);r++)
  any|=mask[(size_t)r*blocks+b]!=0&&b*8<start+r+1;
 unsigned n=any;
 for(int offset=16;offset;offset/=2)n+=__shfl_down_sync(0xffffffff,n,offset);
 if((threadIdx.x&31)==0)atomicAdd(unions+blockIdx.y,n);
}

// Fused publication: component proof compact-remap-sort-v1.
#include <climits>
extern "C" __global__ void ca_sparse6_remap_sort_selection(const int32_t* indices,int32_t* output,
 const float* scores,int stride,const int32_t* ids,const int32_t* ends,
 const int32_t* lengths,uint32_t* error){
 __shared__ int valid_count;
 int r=blockIdx.x,t=threadIdx.x,n=ends[0],k=indices[r*512+t],mapped=-1;
 if(t==0)valid_count=0;
 __syncthreads();
 if(k>=0){
  if(k>=n){atomicOr(error,8u);}
  else {
   float value=scores[(size_t)r*stride+k];
   if(isnan(value)||value==INFINITY)atomicOr(error,16u);
   else if(value!=-INFINITY){
    mapped=ids[k/8]*8+k%8;
    if(mapped<0||mapped>=lengths[r]){atomicOr(error,8u);mapped=-1;}
   }
  }
 }else if(k!=-1)atomicOr(error,8u);
 unsigned vote=__ballot_sync(0xffffffff,mapped>=0);
 if((t&31)==0)atomicAdd(&valid_count,__popc(vote));
 __syncthreads();
 if(t==0&&valid_count!=min(lengths[r],512))atomicOr(error,32u);
 // Preserve the selected compact IDs and exact mapping, then publish sorted
 // absolute IDs directly to their final owner. No intermediate memcpy.
 __shared__ int32_t row[512];
 int32_t x=mapped<0?INT_MAX:mapped;
 for(int width=2;width<=512;width<<=1)for(int distance=width>>1;distance>0;distance>>=1){
  int32_t other;
  if(distance>=32){row[t]=x;__syncthreads();other=row[t^distance];__syncthreads();}
  else other=__shfl_xor_sync(0xffffffff,x,distance);
  bool ascending=(t&width)==0;
  x=((t&distance)==0)==ascending?min(x,other):max(x,other);
 }
 output[r*512+t]=x==INT_MAX?-1:x;
}
