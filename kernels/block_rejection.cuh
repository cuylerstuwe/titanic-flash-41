// Block verification (Sun et al., ICLR2025, arXiv:2403.10444), independently
// implemented from the published recurrence. Full FP64 p/q and residual mass;
// exact target sampling settings are unchanged. A final non-draft row is required.
extern "C" __global__ void ca_block_weights(const double* p,const double* q,
 const int32_t* proposals,double* w,uint32_t* errors,int drafts) {
  if(threadIdx.x)return;
  for(int i=0;i<=drafts;++i)w[i]=0.;
  w[0]=1.;
  for(int i=0;i<drafts;++i){int id=proposals[i];
    if(id<0||id>=129280){errors[0]|=1u;return;}
    double a=p[(size_t)i*129280+id],b=q[(size_t)i*129280+id];
    if(!isfinite(a)||!isfinite(b)||a<0.||b<=0.){errors[0]|=2u;return;}
    // Divide only after comparison, avoiding overflow when q is tiny.
    double numerator=w[i]*a;
    w[i+1]=numerator>=b?1.:numerator/b;
  }
}
extern "C" __global__ void ca_block_partials(const double* p,const double* q,
 const double* w,double* sums,uint32_t* errors,int drafts) {
  int row=blockIdx.y,tile=blockIdx.x,t=threadIdx.x;
  double sum=0.;
  for(int off=t;off<1024;off+=256){int id=tile*1024+off;if(id>=129280)continue;
    double a=p[(size_t)row*129280+id],b=row<drafts?q[(size_t)row*129280+id]:0.;
    if(!isfinite(a)||!isfinite(b)||a<0.||b<0.)atomicOr(errors+row,4u);
    else sum+=fmax(w[row]*a-b,0.);
  }
  sum=ca_categorical_sum(sum);if(t==0)sums[row*127+tile]=sum;
}
extern "C" __global__ void ca_block_decide(const double* w,const double* sums,
 const int32_t* proposals,const double* uniforms,const int32_t* bonus,
 uint32_t* state,int32_t* out,uint32_t* errors,int drafts) {
  if(threadIdx.x)return;
  int tau=0;
  for(int i=1;i<=drafts;++i){
    double mass=0.;for(int t=0;t<127;++t)mass+=sums[i*127+t];
    // Preserve the residual mass when w==1; (mass+1)-1 can round below
    // mass and produce h>1 for a mathematically saturated acceptance.
    double denominator=mass+(1.-w[i]);
    double h=i==drafts?w[i]:(denominator>0.?mass/denominator:0.);
    double u=uniforms[i-1];
    if(!isfinite(h)||h<0.||h>1.||!isfinite(u)||u<0.||u>=1.)errors[i]|=8u;
    if(u<h)tau=i;
  }
  for(int i=0;i<=drafts;++i){
    state[i]=i<tau?1u:(i==tau&&tau<drafts?0u:2u);
    out[i]=i<tau?proposals[i]:bonus[i];
  }
}
extern "C" __global__ void ca_block_finish(const double* p,const double* q,
 const double* w,const uint32_t* state,const int32_t* tiles,
 const double* fractions,int32_t* out,uint32_t* errors) {
  int row=blockIdx.x,t=threadIdx.x;if(state[row]!=0||errors[row])return;
  int tile=tiles[row];if(tile<0||tile>=127){if(t==0)errors[row]|=32u;return;}
  __shared__ double mass[1024];
  for(int off=t;off<1024;off+=256){int id=tile*1024+off;
    mass[off]=id<129280?fmax(w[row]*p[(size_t)row*129280+id]-q[(size_t)row*129280+id],0.):0.;}
  __syncthreads();if(t==0){double sum=0.;for(int i=0;i<1024;++i)sum+=mass[i];
    if(!isfinite(sum)||sum<=0.){errors[row]|=64u;return;}
    double target=fmin(fractions[row]*sum,nextafter(sum,0.)),before=0.;
    for(int i=0;i<1024;++i){double after=before+mass[i];if(target<after){out[row]=tile*1024+i;return;}before=after;}
    errors[row]|=128u;
  }
}
