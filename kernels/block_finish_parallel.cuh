// Original block residual arithmetic with FP64 parallel CDF. CUB from the pinned CUDA toolkit (CCCL BSD-3-Clause).
#include <cub/block/block_scan.cuh>
template<cub::BlockScanAlgorithm Algorithm, bool Diagnostic>
__device__ __forceinline__ void ca_block_parallel_cdf_body_row(int row,
 const double* p,const double* q,const double* weights,const uint32_t* state,
 const int32_t* selected,const double* fractions,uint32_t* errors,int32_t* output,
 double* diagnostic) {
  const int t=threadIdx.x,tile=selected[row];
  if(state[row]!=0||errors[row])return;
  if(tile<0||tile>=127){if(t==0)errors[row]|=32u;return;}
  using Scan=cub::BlockScan<double,256,Algorithm>;
  __shared__ typename Scan::TempStorage scratch;
  __shared__ double total;
  __shared__ int chosen;
  __shared__ int last_positive;
  double cdf[4];
  bool positive[4];
  #pragma unroll
  for(int i=0;i<4;++i) {
    const int id=tile*1024+t*4+i;
    cdf[i]=id<129280?fmax(weights[row]*p[(size_t)row*129280+id]-q[(size_t)row*129280+id],0.):0.;
    positive[i]=cdf[i]>0.;
  }
  Scan(scratch).InclusiveSum(cdf,cdf);
  if(t==255)total=cdf[3];
  if(t==0) {chosen=129280;last_positive=-1;}
  __syncthreads();
  if(!isfinite(total)||total<=0.) {
    if(t==0) {errors[row]|=64u;}
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
    if(last_positive<0)errors[row]|=128u;
    if(Diagnostic&&last_positive>=0)
      diagnostic[(size_t)row*1024+last_positive-tile*1024]=1.;
  }
}


extern "C" __global__ void ca_block_finish_raking(const double* p,const double* q,const double* w,
 const uint32_t* state,const int32_t* tiles,const double* fractions,int32_t* out,uint32_t* errors) {
 ca_block_parallel_cdf_body_row<cub::BLOCK_SCAN_RAKING,false>(blockIdx.x,p,q,w,state,tiles,fractions,errors,out,nullptr);
}
