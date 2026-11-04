// Combine the existing row-parallel decision, serial tile choice, and
// parallel FP64 residual CDF. The first warp retains original row arithmetic.
extern "C" __global__ void ca_block_decide_finish_fused(
 const double* p,const double* q,const double* w,const double* sums,
 const int32_t* proposals,const double* ua,const int32_t* bonus,const double* ur,
 uint32_t* state,int32_t* out,uint32_t* errors,int32_t* tiles,double* fractions,int drafts) {
 const int t=threadIdx.x;
 if(blockDim.x!=256||drafts<0||drafts>=32){if(t==0)errors[0]|=256u;return;}
 __shared__ int selected_row;
 if(t<32) {
  bool accepted=false;
  if(t>0&&t<=drafts) {
   double h;
   if(t==drafts)h=w[t];
   else {double mass=0.;for(int i=0;i<127;++i)mass+=sums[t*127+i];
    double denominator=mass+(1.-w[t]);h=denominator>0.?mass/denominator:0.;}
   double u=ua[t-1];
   if(!isfinite(h)||h<0.||h>1.||!isfinite(u)||u<0.||u>=1.)errors[t]|=8u;
   accepted=u<h;
  }
  unsigned mask=__ballot_sync(0xffffffffu,accepted);
  int tau=mask?31-__clz(mask):0;
  if(t==0)selected_row=tau;
  if(t<=drafts){state[t]=t<tau?1u:(t==tau&&tau<drafts?0u:2u);out[t]=t<tau?proposals[t]:bonus[t];
   tiles[t]=-1;fractions[t]=0.;}
 }
 __syncthreads();
 const int row=selected_row;
 if(row==drafts)return;
 if(t==0&&errors[row]==0) {
  double sum=0.;for(int i=0;i<127;++i)sum+=sums[row*127+i];
  double u=ur[row];
  if(!isfinite(sum)||sum<=0.||!isfinite(u)||u<0.||u>=1.)errors[row]|=8u;
  else {
   double target=fmin(u*sum,nextafter(sum,0.)),before=0.;
   for(int i=0;i<127;++i){double mass=sums[row*127+i],after=before+mass;
    if(target<after){tiles[row]=i;fractions[row]=fmin(fmax((target-before)/mass,0.),nextafter(1.,0.));break;}
    before=after;}
   if(tiles[row]<0)errors[row]|=16u;
  }
 }
 __syncthreads();
 ca_block_parallel_cdf_body_row<cub::BLOCK_SCAN_RAKING,false>(row,p,q,w,state,tiles,fractions,errors,out,nullptr);
}
