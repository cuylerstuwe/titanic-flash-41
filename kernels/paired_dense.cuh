// Row transport only: preserve existing MXFP8 bytes and BF16 output bits.
#include <stdint.h>
extern "C" __global__ void ca_pair_shared_quant(const uint8_t* a,const uint8_t* b,
 const uint8_t* as,const uint8_t* bs,uint8_t* q,uint8_t* s,int al,int bl,int rows) {
 const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
 if(i<rows*5120) q[i]=i<al*5120?a[i]:(i<(al+bl)*5120?b[i-al*5120]:0);
 if(i<rows*160) {
  const int r=i/160,c=i%160,src=r<al?r:r-al;
  const int oi=(c/4*32+r)*16+c%4;
  const int si=(c/4*32+src)*16+c%4;
  s[oi]=r<al?as[si]:(r<al+bl?bs[si]:127);
 }
}
extern "C" __global__ void ca_pair_shared_scatter(const uint16_t* p,
 uint16_t* a,uint16_t* b,int al,int bl) {
 const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
 if(i<al*1152)a[i]=p[i];
 else if(i<(al+bl)*1152)b[i-al*1152]=p[i];
}
extern "C" __global__ void ca_pair_qkv_scatter(const uint16_t* p,
 uint16_t* a,uint16_t* b,int al,int bl) {
 const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
 if(i<al*448)a[i]=p[i];
 else if(i<(al+bl)*448)b[i-al*448]=p[i];
}
