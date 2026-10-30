// Integer-only packing of the source20 publisher's newest128 rows. Each
// rank contributes a fixed128-row packet; row ownership still follows TP4.
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void ca_pack_retained_candidate_ids(
 const int32_t* ids,const int32_t* live,int32_t* packed,int rows,int rank){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=128*512)return;
 int slot=i/512,k=i%512,count=live[0],g=max(0,count-128)+slot;
 int owned=rows/4;
 bool valid=g<count&&g<rows&&g>=rank*owned&&g<(rank+1)*owned;
 int4 value=valid?reinterpret_cast<const int4*>(ids)[(size_t)g*512+k]
                  :make_int4(-1,-1,-1,-1);
 reinterpret_cast<int4*>(packed)[(size_t)rank*128*512+i]=value;
}

extern "C" __global__ void ca_build_compact_retained_candidate_mask(
 const int32_t* packed,const int32_t* top,const int32_t* live,
 uint8_t* mask,int stride,int rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=128*2048)return;
 int slot=i/2048,k=i%2048,g=max(0,live[0]-128)+slot;
 if(g>=live[0]||g>=rows||k>=top[g])return;
 int owner=g/(rows/4),b=packed[((size_t)owner*128+slot)*2048+k];
 if(b>=0&&b<stride)mask[(size_t)g*stride+b]=1;
}
