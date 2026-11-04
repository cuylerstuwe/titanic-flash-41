// Exact copies of the original tile selection and local CDF bodies.
// Keep standalone control entrypoints unchanged for the component comparison.
__device__ __forceinline__ void ca_sharded_block_tile_fused_body(const double* sums,const double* u,int* selected,
 double* fraction,double* total,uint32_t* errors) {
 int row=blockIdx.x;if(threadIdx.x)return;
 double mass=0.;for(int i=0;i<32;i++)mass+=sums[row*32+i];total[row]=mass;
 selected[row]=-1;fraction[row]=0.;double draw=u[row];
 if(!isfinite(mass)||mass<0||!isfinite(draw)||draw<0||draw>=1)errors[row]|=8u;
 if(errors[row]||mass==0)return; // Empty individual shards are valid.
 double target=fmin(draw*mass,nextafter(mass,0.)),before=0.;
 for(int i=0;i<32;i++) {
  double after=before+sums[row*32+i];
  if(target<after){selected[row]=i;fraction[row]=fmin(fmax((target-before)/sums[row*32+i],0.),nextafter(1.,0.));return;}
  before=after;
 }
 errors[row]|=16u;
}
__device__ __forceinline__ void ca_sharded_block_local_fused_body(const double* p,const double* q,const double* w,
 const int* tile,const double* fraction,const double* total,const uint32_t* errors,
 CaShardedResidual* wire,int drafts,int rank) {
 int row=blockIdx.x,t=threadIdx.x,k=tile[row];
 if(errors[row]||total[row]==0.) {
  if(t==0)wire[row]={total[row],-1,errors[row]};return;
 }
 if(k<0||k>=32){if(t==0)wire[row]={total[row],-1,32u};return;}
 using Scan=cub::BlockScan<double,256,cub::BLOCK_SCAN_RAKING>;
 __shared__ typename Scan::TempStorage scratch;
 __shared__ double norm;__shared__ int chosen,last_positive;
 double cdf[4];bool positive[4];
 #pragma unroll
 for(int i=0;i<4;i++) {
  int id=k*1024+t*4+i;
  double a=id<32320?p[(size_t)row*129280+rank*32320+id]:0.;
  double b=id<32320&&row<drafts?q[(size_t)row*32320+id]:0.;
  cdf[i]=id<32320?fmax(w[row]*a-b,0.):0.;positive[i]=cdf[i]>0.;
 }
 Scan(scratch).InclusiveSum(cdf,cdf);
 if(t==255)norm=cdf[3];if(t==0){chosen=32320;last_positive=-1;}__syncthreads();
 if(!isfinite(norm)||norm<=0){if(t==0)wire[row]={total[row],-1,64u};return;}
 double target=fmin(fraction[row]*norm,nextafter(norm,0.));int candidate=32320,last=-1;
 #pragma unroll
 for(int i=0;i<4;i++) {
  int id=k*1024+t*4+i;
  if(id<32320&&positive[i]){last=id;if(target<cdf[i])candidate=min(candidate,id);}
 }
 for(int d=16;d;d>>=1){candidate=min(candidate,__shfl_down_sync(0xffffffff,candidate,d));last=max(last,__shfl_down_sync(0xffffffff,last,d));}
 if((t&31)==0){atomicMin(&chosen,candidate);atomicMax(&last_positive,last);}__syncthreads();
 if(t==0) {
  int id=chosen<32320?chosen:last_positive;
  wire[row]={total[row],id<0?-1:rank*32320+id,id<0?128u:0u};
 }
}

extern "C" __global__ void ca_sharded_block_tile_finish_bundle(
 const double* sums,const double* u,int* tile,double* fraction,double* total,uint32_t* errors,
 const double* p,const double* q,const double* w,CaShardedResidual* packed,
 const int32_t* target_ids,const uint32_t* target_errors,CaBundledResidual* wire,int drafts,int rank) {
 ca_sharded_block_tile_fused_body(sums,u,tile,fraction,total,errors);
 __syncthreads();
 ca_sharded_block_local_fused_body(p,q,w,tile,fraction,total,errors,packed,drafts,rank);
 __syncthreads();
 if(threadIdx.x==0) {
  int row=blockIdx.x;
  wire[row]={packed[row],target_ids[row],target_errors[row],0};
 }
}
