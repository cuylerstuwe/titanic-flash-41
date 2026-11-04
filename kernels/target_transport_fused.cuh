// Fuse control work into target transport without changing tile arithmetic.
extern "C" __global__ void ca_target_pack_fused(
 const float* local,float* packed,int rank,float temperature,const int32_t* proposals,uint32_t* errors) {
  int row=blockIdx.y,slot=blockIdx.x,t=threadIdx.x;
  float* output=packed+(size_t)row*CA_SHARD_WORDS;
  int lower=rank*CA_SHARD_COLS,upper=lower+CA_SHARD_COLS;
  if(slot>=32) {
    int side=slot-32;
    int tile=side==0?lower/1024:upper/1024;
    bool present=side==0?rank>0:rank<3;
    for(int off=t;off<1024;off+=256) {
      int id=tile*1024+off;
      if(side==0&&off==0)continue;
      output[128+side*1024+off]=(present&&id>=lower&&id<upper)?
          local[(size_t)row*CA_SHARD_COLS+id-lower]:-INFINITY;
    }
    if(side==0&&t==0) {
      // This boundary word is unused by the original global tiles. One CTA
      // owns it, so proposal publication cannot race the boundary fill.
      int id=proposals[row];errors[row]=0;
      output[128]=id>=lower&&id<upper?local[(size_t)row*CA_SHARD_COLS+id-lower]:-INFINITY;
    }
    return;
  }
  int tile=(lower+1023)/1024+slot;
  int end=rank==3?127:upper/1024;
  auto* stats=reinterpret_cast<CaShardStat*>(output);
  if(tile>=end) {if(t==0)stats[slot]={0.,-INFINITY,0};return;}
  __shared__ float maximum;
  __shared__ uint32_t error;
  if(t==0)error=0;
  __syncthreads();
  CaGreedyBest best={-INFINITY,2147483647};
  for(int off=t;off<1024;off+=256) {
    int id=tile*1024+off;if(id>=129280)continue;
    float x=local[(size_t)row*CA_SHARD_COLS+id-lower];
    if(isnan(x)||x==INFINITY){atomicOr(&error,1u);continue;}
    best=ca_greedy_better(best,CaGreedyBest{x,id});
  }
  best=ca_greedy_block(best);
  if(t==0)maximum=best.value;
  __syncthreads();
  double sum=0.;
  if(isfinite(maximum))for(int off=t;off<1024;off+=256) {
    int id=tile*1024+off;if(id>=129280)continue;
    float x=local[(size_t)row*CA_SHARD_COLS+id-lower];
    if(isfinite(x))sum+=(double)expf((x-maximum)/temperature);
  }
  sum=ca_categorical_sum(sum);
  if(t==0)stats[slot]={sum,maximum,error};
}
// Original tile/CDF arithmetic followed by the original local factors. All
// threads reach the barrier even when the row is invalid or a CDF tile is found.
extern "C" __global__ void ca_target_tile_factors_fused(
 const float* maxima,const double* sums,const double* uniforms,
 int32_t* selected,double* fractions,double* normalization,double* factors,
 uint32_t* errors,float temperature,const float* gathered,const int32_t* proposals,
 double* normalized,double* p,double* selected_probability) {
 int row=blockIdx.x,t=threadIdx.x;
 CaGreedyBest best=t<127?CaGreedyBest{maxima[row*127+t],t}:CaGreedyBest{-INFINITY,2147483647};
 best=ca_greedy_block(best);
 __shared__ float m;__shared__ double mass[127];
 if(t==0)m=best.value;__syncthreads();
 if(t<127) {
  double factor=(isfinite(m)&&isfinite(maxima[row*127+t]))?
    exp((double(maxima[row*127+t])-double(m))/double(temperature)):0.;
  factors[row*127+t]=factor;mass[t]=sums[row*127+t]*factor;
 }
 __syncthreads();
 if(t==0) {
  selected[row]=-1;fractions[row]=0.;
  double sum=0.;for(int i=0;i<127;++i)sum+=mass[i];
  normalization[row*2]=double(m);normalization[row*2+1]=sum;
  double u=uniforms[row];
  if(!isfinite(sum)||sum<=0.)errors[row]|=2u;
  if(!isfinite(u)||u<0.||u>=1.)errors[row]|=4u;
  if(!errors[row]) {
   double target=fmin(u*sum,nextafter(sum,0.)),before=0.;
   for(int i=0;i<127;++i) {
    double after=before+mass[i];
    if(target<after) {
     selected[row]=i;
     fractions[row]=fmin(fmax((target-before)/mass[i],0.),nextafter(1.,0.));
     break;
    }
    before=after;
   }
   if(selected[row]<0)errors[row]|=8u;
  }
 }
 __syncthreads();
 if(t<127)normalized[row*127+t]=factors[row*127+t]/normalization[row*2+1];
 __syncthreads();
 if(t==0) {
  int id=proposals[row];double value=NAN;
  if(!errors[row]&&id>=0&&id<129280) {
   float x=gathered[((size_t)row*4+id/32320)*CA_SHARD_WORDS+128],maximum=maxima[row*127+id/1024];
   value=isfinite(x)&&isfinite(maximum)?__dmul_rn(double(expf((x-maximum)/temperature)),normalized[row*127+id/1024]):0.;
  }
  selected_probability[row]=value;if(id>=0&&id<129280)p[(size_t)row*129280+id]=value;
 }
}
