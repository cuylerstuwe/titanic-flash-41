// Native tiled categorical sampling at temperature T, top_p=1. The host
// supplies independent counter-addressed uniforms; all logit arithmetic stays
// on GPU. Float exponentials, double mass/CDF accumulation, no quantization.
__device__ __forceinline__ double ca_categorical_sum(double x) {
  __shared__ double warp[8];
  int lane=threadIdx.x&31, w=threadIdx.x>>5;
  for(int d=16;d;d>>=1)x+=__shfl_down_sync(0xffffffff,x,d);
  if(lane==0)warp[w]=x;
  __syncthreads();
  x=(w==0&&lane<8)?warp[lane]:0.;
  if(w==0)for(int d=16;d;d>>=1)x+=__shfl_down_sync(0xffffffff,x,d);
  return x;
}
extern "C" __global__ void ca_categorical_partials(
 const float* logits,float* maxima,double* sums,uint32_t* errors,float temperature) {
  int row=blockIdx.y,tile=blockIdx.x,t=threadIdx.x;
  CaGreedyBest best={-INFINITY,2147483647};
  for(int off=t;off<1024;off+=256) {
    int id=tile*1024+off;if(id>=129280)continue;
    float x=logits[(size_t)row*129280+id];
    if(isnan(x)||x==INFINITY) {atomicOr(errors+row,1u);continue;}
    best=ca_greedy_better(best,CaGreedyBest{x,id});
  }
  best=ca_greedy_block(best);
  __shared__ float m;
  if(t==0) {m=best.value;maxima[row*127+tile]=best.value;}
  __syncthreads();
  double sum=0.;
  if(isfinite(m))for(int off=t;off<1024;off+=256) {
    int id=tile*1024+off;if(id>=129280)continue;
    float x=logits[(size_t)row*129280+id];
    if(isfinite(x))sum+=(double)expf((x-m)/temperature);
  }
  sum=ca_categorical_sum(sum);
  if(t==0)sums[row*127+tile]=sum;
}
extern "C" __global__ void ca_categorical_tile(
 const float* maxima,const double* sums,const double* uniforms,
 int32_t* selected,double* fractions,double* normalization,uint32_t* errors,float temperature) {
  int row=blockIdx.x,t=threadIdx.x;
  CaGreedyBest best=t<127?CaGreedyBest{maxima[row*127+t],t}:CaGreedyBest{-INFINITY,2147483647};
  best=ca_greedy_block(best);
  __shared__ float m;
  __shared__ double mass[127];
  if(t==0)m=best.value;
  __syncthreads();
  if(t<127)mass[t]=(isfinite(m)&&isfinite(maxima[row*127+t]))?
      sums[row*127+t]*exp((double(maxima[row*127+t])-double(m))/double(temperature)):0.;
  __syncthreads();
  if(t==0) {
    selected[row]=-1;fractions[row]=0.;
    double sum=0.;for(int i=0;i<127;++i)sum+=mass[i];
    normalization[row*2]=double(m);normalization[row*2+1]=sum;
    double u=uniforms[row];
    if(!isfinite(sum)||sum<=0.)errors[row]|=2u;
    if(!isfinite(u)||u<0.||u>=1.)errors[row]|=4u;
    if(errors[row])return;
    double target=fmin(u*sum,nextafter(sum,0.)),before=0.;
    for(int i=0;i<127;++i) {
      double after=before+mass[i];
      if(target<after) {
        selected[row]=i;
        fractions[row]=fmin(fmax((target-before)/mass[i],0.),nextafter(1.,0.));
        return;
      }
      before=after;
    }
    errors[row]|=8u;
  }
}
// Cache the very same FP64 tile factor used for the CDF. Reuse it only
// after this kernel completes on the ordered sampling stream. Neither the
// per-logit expf nor the multiply-then-divide probability order changes.
extern "C" __global__ void ca_categorical_tile_cached(
 const float* maxima,const double* sums,const double* uniforms,
 int32_t* selected,double* fractions,double* normalization,double* factors,uint32_t* errors,float temperature) {
  int row=blockIdx.x,t=threadIdx.x;
  CaGreedyBest best=t<127?CaGreedyBest{maxima[row*127+t],t}:CaGreedyBest{-INFINITY,2147483647};
  best=ca_greedy_block(best);
  __shared__ float m;
  __shared__ double mass[127];
  if(t==0)m=best.value;
  __syncthreads();
  if(t<127) {
    double factor=(isfinite(m)&&isfinite(maxima[row*127+t]))?
      exp((double(maxima[row*127+t])-double(m))/double(temperature)):0.;
    factors[row*127+t]=factor;
    mass[t]=sums[row*127+t]*factor;
  }
  __syncthreads();
  if(t==0) {
    selected[row]=-1;fractions[row]=0.;
    double sum=0.;for(int i=0;i<127;++i)sum+=mass[i];
    normalization[row*2]=double(m);normalization[row*2+1]=sum;
    double u=uniforms[row];
    if(!isfinite(sum)||sum<=0.)errors[row]|=2u;
    if(!isfinite(u)||u<0.||u>=1.)errors[row]|=4u;
    if(errors[row])return;
    double target=fmin(u*sum,nextafter(sum,0.)),before=0.;
    for(int i=0;i<127;++i) {
      double after=before+mass[i];
      if(target<after) {
        selected[row]=i;
        fractions[row]=fmin(fmax((target-before)/mass[i],0.),nextafter(1.,0.));
        return;
      }
      before=after;
    }
    errors[row]|=8u;
  }
}
// Expose precisely the tiled exponential weights used to draw q. Rejection
// sampling must use the actual proposal law, not independently rounded logits.
extern "C" __global__ void ca_categorical_probabilities(
 const float* logits,const float* maxima,const double* normalization,
 double* probabilities,const uint32_t* errors,float temperature) {
  int row=blockIdx.y,id=blockIdx.x*256+threadIdx.x;
  if(id>=129280)return;
  float x=logits[(size_t)row*129280+id],local=maxima[row*127+id/1024];
  double global=normalization[row*2],total=normalization[row*2+1];
  probabilities[(size_t)row*129280+id]=errors[row]?NAN:
    (isfinite(x)?double(expf((x-local)/temperature))*
     exp((double(local)-global)/double(temperature))/total:0.);
}
// Same probability expression, with one shared double exponential per256-lane
// block. Every such block lies within one1024-element tile. In particular, do
// not pre-divide the shared factor: multiply followed by divide is unchanged.
extern "C" __global__ void ca_categorical_probabilities_shared(
 const float* logits,const float* maxima,const double* normalization,
 double* probabilities,const uint32_t* errors,float temperature) {
  int row=blockIdx.y,id=blockIdx.x*256+threadIdx.x,tile=blockIdx.x/4;
  if(id>=129280)return;
  if(errors[row]){probabilities[(size_t)row*129280+id]=NAN;return;}
  float local=maxima[row*127+tile];
  // A valid all-masked tile has zero mass. Row-level NaN/+infinity/empty
  // errors were handled above, uniformly across this entire block.
  if(!isfinite(local)){probabilities[(size_t)row*129280+id]=0.;return;}
  double global=normalization[row*2],total=normalization[row*2+1];
  __shared__ double factor;
  if(threadIdx.x==0)factor=exp((double(local)-global)/double(temperature));
  __syncthreads();
  float x=logits[(size_t)row*129280+id];
  probabilities[(size_t)row*129280+id]=isfinite(x)?
    double(expf((x-local)/temperature))*factor/total:0.;
}
extern "C" __global__ void ca_categorical_probabilities_cached(
 const float* logits,const float* maxima,const double* normalization,
 const double* factors,double* probabilities,const uint32_t* errors,float temperature) {
  int row=blockIdx.y,id=blockIdx.x*256+threadIdx.x,tile=blockIdx.x/4;
  if(id>=129280)return;
  if(errors[row]){probabilities[(size_t)row*129280+id]=NAN;return;}
  float local=maxima[row*127+tile];
  // A valid all-masked tile has zero mass. Row-level NaN/+infinity/empty
  // errors were handled above, uniformly across this entire block.
  if(!isfinite(local)){probabilities[(size_t)row*129280+id]=0.;return;}
  double total=normalization[row*2+1],factor=factors[row*127+tile];
  float x=logits[(size_t)row*129280+id];
  probabilities[(size_t)row*129280+id]=isfinite(x)?
    double(expf((x-local)/temperature))*factor/total:0.;
}
extern "C" __global__ void ca_categorical_finish(
 const float* logits,const float* maxima,const int32_t* selected,
 const double* fractions,uint32_t* errors,int32_t* output,float temperature) {
  int row=blockIdx.x,t=threadIdx.x,tile=selected[row];
  if(errors[row]||tile<0||tile>=127) {if(t==0)output[row]=-1;return;}
  __shared__ double mass[1024];
  float m=maxima[row*127+tile];
  for(int off=t;off<1024;off+=256) {
    int id=tile*1024+off;
    float x=id<129280?logits[(size_t)row*129280+id]:-INFINITY;
    mass[off]=isfinite(x)?(double)expf((x-m)/temperature):0.;
  }
  __syncthreads();
  if(t==0) {
    output[row]=-1;
    double sum=0.;for(int i=0;i<1024;++i)sum+=mass[i];
    if(!isfinite(sum)||sum<=0.) {errors[row]|=16u;return;}
    double target=fmin(fractions[row]*sum,nextafter(sum,0.)),before=0.;
    for(int i=0;i<1024;++i) {
      double after=before+mass[i];
      if(target<after) {output[row]=tile*1024+i;return;}
      before=after;
    }
    errors[row]|=32u;
  }
}
