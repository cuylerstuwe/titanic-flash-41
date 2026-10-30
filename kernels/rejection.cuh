// Canonical speculative rejection: accept with min(1,p(y)/q(y)); otherwise
// draw from max(p-q,0). All probabilities and residual CDFs remain on GPU.
extern "C" __global__ void ca_reject_decide(const double* p,const double* q,
 const int32_t* proposals,const double* uniforms,const int32_t* bonus,
 uint32_t* state,int32_t* out,uint32_t* errors,int drafts) {
  int row=blockIdx.x;if(threadIdx.x)return;
  if(row>=drafts){state[row]=2;out[row]=bonus[row];return;}
  int id=proposals[row];double u=uniforms[row];state[row]=0;out[row]=-1;
  if(id<0||id>=129280||!isfinite(u)||u<0.||u>=1.){errors[row]|=1u;return;}
  double a=p[(size_t)row*129280+id],b=q[(size_t)row*129280+id];
  if(!isfinite(a)||!isfinite(b)||a<0.||b<=0.){errors[row]|=2u;return;}
  if(a>=b||u*b<a){state[row]=1;out[row]=id;}
}
extern "C" __global__ void ca_reject_partials(const double* p,const double* q,
 const uint32_t* state,double* sums,uint32_t* errors) {
  int row=blockIdx.y,tile=blockIdx.x,t=threadIdx.x;
  if(state[row]!=0){if(t==0)sums[row*127+tile]=0.;return;}
  double sum=0.;for(int off=t;off<1024;off+=256){
    int id=tile*1024+off;if(id>=129280)continue;
    double a=p[(size_t)row*129280+id],b=q[(size_t)row*129280+id];
    if(!isfinite(a)||!isfinite(b)||a<0.||b<0.)atomicOr(errors+row,4u);
    else sum+=fmax(a-b,0.);
  }
  sum=ca_categorical_sum(sum);if(t==0)sums[row*127+tile]=sum;
}
extern "C" __global__ void ca_reject_tile(const double* sums,const double* uniforms,
 const uint32_t* state,int32_t* tiles,double* fractions,uint32_t* errors) {
  int row=blockIdx.x;if(threadIdx.x)return;tiles[row]=-1;fractions[row]=0.;
  if(state[row]!=0||errors[row])return;
  double sum=0.;for(int i=0;i<127;++i)sum+=sums[row*127+i];
  double u=uniforms[row];if(!isfinite(sum)||sum<=0.||!isfinite(u)||u<0.||u>=1.){errors[row]|=8u;return;}
  double target=fmin(u*sum,nextafter(sum,0.)),before=0.;
  for(int i=0;i<127;++i){double mass=sums[row*127+i],after=before+mass;
    if(target<after){tiles[row]=i;fractions[row]=fmin(fmax((target-before)/mass,0.),nextafter(1.,0.));return;}
    before=after;}
  errors[row]|=16u;
}
extern "C" __global__ void ca_reject_finish(const double* p,const double* q,
 const uint32_t* state,const int32_t* tiles,const double* fractions,int32_t* out,uint32_t* errors) {
  int row=blockIdx.x,t=threadIdx.x;if(state[row]!=0||errors[row])return;
  int tile=tiles[row];if(tile<0||tile>=127){if(t==0)errors[row]|=32u;return;}
  __shared__ double mass[1024];
  for(int off=t;off<1024;off+=256){int id=tile*1024+off;
    mass[off]=id<129280?fmax(p[(size_t)row*129280+id]-q[(size_t)row*129280+id],0.):0.;}
  __syncthreads();if(t==0){double sum=0.;for(int i=0;i<1024;++i)sum+=mass[i];
    if(!isfinite(sum)||sum<=0.){errors[row]|=64u;return;}
    double target=fmin(fractions[row]*sum,nextafter(sum,0.)),before=0.;
    for(int i=0;i<1024;++i){double after=before+mass[i];if(target<after){out[row]=tile*1024+i;return;}before=after;}
    errors[row]|=128u;
  }
}
