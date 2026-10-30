// Source20's wide encoder publishes only the newest128 live rows to the
// decoder ring. These are publisher-only windows; main top512 is unchanged.
// The GPU component screen s400 checks retained scores, IDs and mask bytes.
#include <cuda_runtime.h>

extern "C" __global__ void ca_retained_candidate_windows(
 const int32_t* lengths,int32_t* raw,int32_t* blocks,int32_t* top,
 const int32_t* live,int rows){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;
 int count=live[0],len=lengths[r];
 bool retained=r>=max(0,count-128)&&r<count;
 raw[r]=retained?len:0;
 int b=retained&&len>0?(len+7)/8:0;
 blocks[r]=b;top[r]=min(b,2048);
}
extern "C" __global__ void ca_zero_retained_candidate_mask(
 uint8_t* mask,int stride,int rows,const int32_t* live){
 int width=stride/16,i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=128*width)return;
 int g=max(0,live[0]-128)+i/width;
 if(g>=live[0]||g>=rows)return;
 reinterpret_cast<uint4*>(mask)[(size_t)g*width+i%width]=make_uint4(0,0,0,0);
}
extern "C" __global__ void ca_build_retained_candidate_mask(
 const int32_t* ids,const int32_t* top,const int32_t* live,
 uint8_t* mask,int stride,int rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=128*2048)return;
 int g=max(0,live[0]-128)+i/2048,k=i%2048;
 if(g>=live[0]||g>=rows||k>=top[g])return;
 int b=ids[(size_t)g*2048+k];
 if(b>=0&&b<stride)mask[(size_t)g*stride+b]=1;
}
