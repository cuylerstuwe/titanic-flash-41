// Read the chosen global tile directly from its existing owner. This removes
// the staging copy before the identical CUB FP64 raking CDF. Keep the original
// global tile boundaries and exact boundary values from the transport wire.
extern "C" __global__ void ca_target_selected_finish(
 const float* local,const float* gathered,const float* maxima,
 const int32_t* selected,const double* fractions,uint32_t* errors,
 int32_t* output,int rank,float temperature) {
  const int row=blockIdx.x,t=threadIdx.x,tile=selected[row];
  if(errors[row]||tile<0||tile>=127||tile*1024/CA_SHARD_COLS!=rank) {
    if(t==0)output[row]=-1;return;
  }
  using Scan=cub::BlockScan<double,256,cub::BLOCK_SCAN_RAKING>;
  __shared__ typename Scan::TempStorage scratch;
  __shared__ double total;
  __shared__ int chosen;
  __shared__ int last_positive;
  double cdf[4];
  bool positive[4];
  const float m=maxima[row*127+tile];
  #pragma unroll
  for(int i=0;i<4;++i) {
    const int off=t*4+i,id=tile*1024+off;
    const float x=id>=129280?-INFINITY:
      ca_shard_boundary(tile)?ca_shard_boundary_value(gathered,row,tile,off):
      local[(size_t)row*CA_SHARD_COLS+id-rank*CA_SHARD_COLS];
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
    output[row]=chosen<129280?chosen:last_positive;
    if(last_positive<0)errors[row]|=32u;
  }
}
