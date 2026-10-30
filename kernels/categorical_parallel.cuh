// Independent candidate for the final, selected-tile categorical CDF.
// CUB BlockScan is supplied by the pinned CUDA toolkit (NVIDIA CCCL, BSD-3-Clause).
// Retain expf and FP64 accumulation; only the addition order changes. No top-k,
// probability truncation, reduced precision or CPU probability work.
#include <cub/block/block_scan.cuh>

template<cub::BlockScanAlgorithm Algorithm, bool Diagnostic>
__device__ __forceinline__ void ca_parallel_cdf_body(
 const float* logits,const float* maxima,const int32_t* selected,
 const double* fractions,uint32_t* errors,int32_t* output,float temperature,
 double* diagnostic) {
  const int row=blockIdx.x,t=threadIdx.x,tile=selected[row];
  if(errors[row]||tile<0||tile>=127) {if(t==0)output[row]=-1;return;}
  using Scan=cub::BlockScan<double,256,Algorithm>;
  __shared__ typename Scan::TempStorage scratch;
  __shared__ double total;
  __shared__ int chosen;
  __shared__ int last_positive;
  double cdf[4];
  bool positive[4];
  const float m=maxima[row*127+tile];
  #pragma unroll
  for(int i=0;i<4;++i) {
    const int id=tile*1024+t*4+i;
    const float x=id<129280?logits[(size_t)row*129280+id]:-INFINITY;
    cdf[i]=isfinite(x)?double(expf((x-m)/temperature)):0.;
    positive[i]=cdf[i]>0.;
  }
  Scan(scratch).InclusiveSum(cdf,cdf);
  if(t==255)total=cdf[3];
  if(t==0) {chosen=129280;last_positive=-1;}
  __syncthreads();
  if(!isfinite(total)||total<=0.) {
    if(t==0) {errors[row]|=16u;output[row]=-1;}
    return;
  }
  const double target=fmin(fractions[row]*total,nextafter(total,0.));
  int candidate=129280;
  int last=-1;
  #pragma unroll
  for(int i=0;i<4;++i) {
    if(Diagnostic)diagnostic[(size_t)row*1024+t*4+i]=positive[i]?cdf[i]/total:0.;
    const int id=tile*1024+t*4+i;
    if(positive[i]&&id<129280) {
      last=id;
      if(target<cdf[i])candidate=min(candidate,id);
    }
  }
  for(int offset=16;offset;offset>>=1) {
    candidate=min(candidate,__shfl_down_sync(0xffffffff,candidate,offset));
    last=max(last,__shfl_down_sync(0xffffffff,last,offset));
  }
  if((t&31)==0) {atomicMin(&chosen,candidate);atomicMax(&last_positive,last);}
  __syncthreads();
  if(t==0) {
    // Positive support only. A final rounding sliver belongs to the last
    // positive mass, just as clamping the original CDF's endpoint to one.
    output[row]=chosen<129280?chosen:last_positive;
    if(last_positive<0)errors[row]|=32u;
    if(Diagnostic&&last_positive>=0)
      diagnostic[(size_t)row*1024+last_positive-tile*1024]=1.;
  }
}

extern "C" __global__ void ca_categorical_finish_raking(
 const float* x,const float* m,const int32_t* tiles,const double* u,
 uint32_t* errors,int32_t* out,float temperature) {
  ca_parallel_cdf_body<cub::BLOCK_SCAN_RAKING,false>(x,m,tiles,u,errors,out,temperature,nullptr);
}
extern "C" __global__ void ca_categorical_finish_warps(
 const float* x,const float* m,const int32_t* tiles,const double* u,
 uint32_t* errors,int32_t* out,float temperature) {
  ca_parallel_cdf_body<cub::BLOCK_SCAN_WARP_SCANS,false>(x,m,tiles,u,errors,out,temperature,nullptr);
}
