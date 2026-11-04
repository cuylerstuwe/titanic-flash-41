// Component: target p stays canonical/full; draft q remains vocabulary-sharded.
// One compact exchange transports residual masses and independent local draws.
struct CaShardedResidual { double mass; int32_t id; uint32_t error; };
static_assert(sizeof(CaShardedResidual)==16,"residual wire ABI");
struct CaBundledResidual { CaShardedResidual residual; int32_t target_id; uint32_t target_error; uint64_t padding; };
static_assert(sizeof(CaBundledResidual)==32,"bundled residual wire ABI");
extern "C" __global__ void ca_sharded_block_bundle(const CaShardedResidual* residual,
 const int32_t* ids,const uint32_t* errors,CaBundledResidual* wire,int rows){
 int row=threadIdx.x;if(row<rows)wire[row]={residual[row],ids[row],errors[row],0};
}
extern "C" __global__ void ca_sharded_block_unbundle(const CaBundledResidual* wire,
 CaShardedResidual* residual,int32_t* ids,uint32_t* errors,int rows){
 int row=threadIdx.x;if(row>=rows)return;
 int count=0,id=-1;uint32_t error=0;
 for(int rank=0;rank<4;rank++){
  const auto value=wire[row*4+rank];residual[row*4+rank]=value.residual;
  error|=value.target_error;
  if(value.target_id>=0&&value.target_id<129280){count++;id=value.target_id;}
 }
 if(!error&&count!=1)error|=64u;
 errors[row]=error;ids[row]=error?-1:id;
}
extern "C" __global__ void ca_sharded_block_selected_q(const double* q,const int32_t* proposals,double* values,int drafts) {
 int row=threadIdx.x;if(row<drafts)values[row]=q[(size_t)row*129280+proposals[row]];
}
extern "C" __global__ void ca_sharded_block_weights(const double* p,const uint64_t* selected_q,
 const int32_t* proposals,double* w,uint32_t* errors,int drafts) {
 if(threadIdx.x)return;
 for(int i=0;i<=drafts;i++)w[i]=0.;w[0]=1.;
 for(int i=0;i<drafts;i++) {
  int id=proposals[i];if(id<0||id>=129280){errors[0]|=1u;return;}
  double a=p[(size_t)i*129280+id],b=*reinterpret_cast<const double*>(selected_q[i]);
  if(!isfinite(a)||!isfinite(b)||a<0||b<=0){errors[0]|=2u;return;}
  double numerator=w[i]*a;w[i+1]=numerator>=b?1.:numerator/b;
 }
}
extern "C" __global__ void ca_sharded_block_partials(const double* p,const double* q,const double* w,
 double* sums,uint32_t* errors,int drafts,int rank) {
 int row=blockIdx.y,tile=blockIdx.x,t=threadIdx.x;double sum=0.;
 for(int off=t;off<1024;off+=256) {
  int id=tile*1024+off;if(id>=32320)continue;
  double a=p[(size_t)row*129280+rank*32320+id],b=row<drafts?q[(size_t)row*32320+id]:0.;
  if(!isfinite(a)||!isfinite(b)||a<0||b<0)atomicOr(errors+row,4u);
  else sum+=fmax(w[row]*a-b,0.);
 }
 sum=ca_categorical_sum(sum);if(t==0)sums[row*32+tile]=sum;
}
extern "C" __global__ void ca_sharded_block_tile(const double* sums,const double* u,int* selected,
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
extern "C" __global__ void ca_sharded_block_local_finish(const double* p,const double* q,const double* w,
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
extern "C" __global__ void ca_sharded_block_decide(const CaShardedResidual* wire,const double* w,
 const int32_t* proposals,const double* ua,const int32_t* bonus,const double* ur,
 uint32_t* state,int32_t* out,uint32_t* errors,int drafts,double* global_mass) {
 int row=threadIdx.x;bool accepted=false;
 if(row<=drafts) {
  double mass=0.;for(int r=0;r<4;r++){mass+=wire[row*4+r].mass;errors[row]|=wire[row*4+r].error;}
  global_mass[row]=mass;
  if(row>0) {
   double denominator=mass+(1.-w[row]);double h=row==drafts?w[row]:(denominator>0?mass/denominator:0.);
   double u=ua[row-1];if(!isfinite(h)||h<0||h>1||!isfinite(u)||u<0||u>=1)errors[row]|=8u;
   accepted=u<h;
  }
 }
 unsigned mask=__ballot_sync(0xffffffff,accepted);int tau=mask?31-__clz(mask):0;
 if(row<=drafts) {state[row]=row<tau?1u:(row==tau&&tau<drafts?0u:2u);out[row]=row<tau?proposals[row]:bonus[row];}
 if(row==tau&&tau<drafts&&!errors[row]) {
  double sum=global_mass[row],u=ur[row];
  if(!isfinite(sum)||sum<=0||!isfinite(u)||u<0||u>=1){errors[row]|=16u;return;}
  double target=fmin(u*sum,nextafter(sum,0.)),before=0.;int owner=-1;
  for(int r=0;r<4;r++){double after=before+wire[row*4+r].mass;if(target<after){owner=r;break;}before=after;}
  if(owner<0){errors[row]|=32u;return;}
  int id=wire[row*4+owner].id;
  if(id<owner*32320||id>=(owner+1)*32320){errors[row]|=64u;return;}
  out[row]=id;
 }
}
