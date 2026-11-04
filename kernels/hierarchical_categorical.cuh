// Conditional-shard draft sampler. Same expf/FP64 tile law as categorical.cuh.
// Canonical target arithmetic is unchanged; actual-q is retained in FP64.
extern "C" __global__ void ca_hierarchy_partials(
 const float* logits,float* maxima,double* sums,uint32_t* errors,float temperature) {
  int row=blockIdx.y,tile=blockIdx.x,t=threadIdx.x;
  CaGreedyBest best={-INFINITY,2147483647};
  for(int off=t;off<1024;off+=256) {
    int id=tile*1024+off;if(id>=32320)continue;
    float x=logits[(size_t)row*32320+id];
    if(isnan(x)||x==INFINITY) {atomicOr(errors+row,1u);continue;}
    best=ca_greedy_better(best,CaGreedyBest{x,id});
  }
  best=ca_greedy_block(best);
  __shared__ float m;
  if(t==0) {m=best.value;maxima[row*32+tile]=best.value;}
  __syncthreads();
  double sum=0.;
  if(isfinite(m))for(int off=t;off<1024;off+=256) {
    int id=tile*1024+off;if(id>=32320)continue;
    float x=logits[(size_t)row*32320+id];
    if(isfinite(x))sum+=(double)expf((x-m)/temperature);
  }
  sum=ca_categorical_sum(sum);
  if(t==0)sums[row*32+tile]=sum;
}

extern "C" __global__ void ca_hierarchy_tile(
 const float* maxima,const double* sums,const double* uniforms,
 int32_t* selected,double* fractions,double* normalization,double* factors,uint32_t* errors,float temperature) {
  int row=blockIdx.x,t=threadIdx.x;
  CaGreedyBest best=t<32?CaGreedyBest{maxima[row*32+t],t}:CaGreedyBest{-INFINITY,2147483647};
  best=ca_greedy_block(best);
  __shared__ float m;
  __shared__ double mass[32];
  if(t==0)m=best.value;
  __syncthreads();
  if(t<32) {
    double factor=(isfinite(m)&&isfinite(maxima[row*32+t]))?
      exp((double(maxima[row*32+t])-double(m))/double(temperature)):0.;
    factors[row*32+t]=factor;
    mass[t]=sums[row*32+t]*factor;
  }
  __syncthreads();
  if(t==0) {
    selected[row]=-1;fractions[row]=0.;
    double sum=0.;for(int i=0;i<32;++i)sum+=mass[i];
    normalization[row*2]=double(m);normalization[row*2+1]=sum;
    double u=uniforms[row];
    if(!isfinite(sum)||sum<=0.)errors[row]|=2u;
    if(!isfinite(u)||u<0.||u>=1.)errors[row]|=4u;
    if(errors[row])return;
    double target=fmin(u*sum,nextafter(sum,0.)),before=0.;
    for(int i=0;i<32;++i) {
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

template<cub::BlockScanAlgorithm Algorithm, bool Diagnostic>
__device__ __forceinline__ void ca_hierarchy_local_cdf_body(
 const float* logits,const float* maxima,const int32_t* selected,
 const double* fractions,uint32_t* errors,int32_t* output,float temperature,
 double* diagnostic) {
  const int row=blockIdx.x,t=threadIdx.x,tile=selected[row];
  if(errors[row]||tile<0||tile>=32) {if(t==0)output[row]=-1;return;}
  using Scan=cub::BlockScan<double,256,Algorithm>;
  __shared__ typename Scan::TempStorage scratch;
  __shared__ double total;
  __shared__ int chosen;
  __shared__ int last_positive;
  double cdf[4];
  bool positive[4];
  const float m=maxima[row*32+tile];
  #pragma unroll
  for(int i=0;i<4;++i) {
    const int id=tile*1024+t*4+i;
    const float x=id<32320?logits[(size_t)row*32320+id]:-INFINITY;
    cdf[i]=isfinite(x)?double(expf((x-m)/temperature)):0.;
    positive[i]=cdf[i]>0.;
  }
  Scan(scratch).InclusiveSum(cdf,cdf);
  if(t==255)total=cdf[3];
  if(t==0) {chosen=32320;last_positive=-1;}
  __syncthreads();
  if(!isfinite(total)||total<=0.) {
    if(t==0) {errors[row]|=16u;output[row]=-1;}
    return;
  }
  const double target=fmin(fractions[row]*total,nextafter(total,0.));
  int candidate=32320;
  int last=-1;
  #pragma unroll
  for(int i=0;i<4;++i) {
    if(Diagnostic)diagnostic[(size_t)row*1024+t*4+i]=positive[i]?cdf[i]/total:0.;
    const int id=tile*1024+t*4+i;
    if(positive[i]&&id<32320) {
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
    output[row]=chosen<32320?chosen:last_positive;
    if(last_positive<0)errors[row]|=32u;
    if(Diagnostic&&last_positive>=0)
      diagnostic[(size_t)row*1024+last_positive-tile*1024]=1.;
  }
}

extern "C" __global__ void ca_hierarchy_local_finish(
 const float* x,const float* m,const int32_t* tiles,const double* u,
 uint32_t* errors,int32_t* out,float temperature) {
  ca_hierarchy_local_cdf_body<cub::BLOCK_SCAN_RAKING,false>(x,m,tiles,u,errors,out,temperature,nullptr);
}

struct CaHierarchyDraw { double maximum, mass; int id; uint32_t error; double conditional_q; };
static_assert(sizeof(CaHierarchyDraw)==32,"wire geometry");
extern "C" __global__ void ca_hierarchy_pack(const double* norm,const int* id,
 const uint32_t* error,CaHierarchyDraw* packed,const float* logits,
 const float* maxima,const double* factors,float temperature) {
 if(threadIdx.x)return;
 double q=0.;int token=id[0];
 if(!error[0]&&token>=0&&token<32320)
  q=double(expf((logits[token]-maxima[token/1024])/temperature))*(factors[token/1024]/norm[1]);
 packed[0]={norm[0],norm[1],token,error[0],q};
}
extern "C" __global__ void ca_hierarchy_finish(const CaHierarchyDraw* packed,const double* uniform,
 double* fractions,int* output,uint32_t* error,double* selected_q,float temperature) {
 if(threadIdx.x)return;
 double maximum=-INFINITY;uint32_t bad=0;
 for(int r=0;r<4;r++) {
   bad|=packed[r].error&~2u; // An empty shard is legal; an empty vocabulary isn't.
   if(packed[r].mass>0)maximum=fmax(maximum,packed[r].maximum);
 }
 double masses[4],total=0;
 for(int r=0;r<4;r++) {
   masses[r]=isfinite(maximum)&&isfinite(packed[r].maximum)?
       packed[r].mass*exp((packed[r].maximum-maximum)/double(temperature)):0.;
   total+=masses[r];
 }
 if(!isfinite(total)||total<=0)bad|=2u;
 double u=uniform[0];if(!isfinite(u)||u<0||u>=1)bad|=4u;
 for(int r=0;r<4;r++)fractions[r]=bad?NAN:masses[r]/total;
 int chosen=-1;
 if(!bad) {
   double target=fmin(u*total,nextafter(total,0.)),before=0.;
   for(int r=0;r<4;r++) {
     double after=before+masses[r];
     if(target<after) {chosen=r;break;}
     before=after;
   }
   if(chosen<0||packed[chosen].id<0||packed[chosen].id>=32320)bad|=8u;
 }
 double q=bad?NAN:packed[chosen].conditional_q*fractions[chosen];
 if(!bad&&(!isfinite(q)||q<=0.))bad|=16u;
 output[0]=bad?-1:chosen*32320+packed[chosen].id;error[0]=bad;
 selected_q[0]=bad?NAN:q;
}
extern "C" __global__ void ca_hierarchy_q(const float* logits,const float* maxima,
 const double* norm,const double* factors,const double* fractions,
 double* q,const uint32_t* error,int rank,float temperature) {
 int id=blockIdx.x*256+threadIdx.x;if(id>=32320)return;
 if(error[0]){q[id]=NAN;return;}
 const float x=logits[id],m=maxima[id/1024];
 if(!isfinite(x)||!isfinite(m)){q[id]=0.;return;}
 // Preserve the conditional tile law, then multiply by the shard's mass.
 const double local=double(expf((x-m)/temperature))*(factors[id/1024]/norm[1]);
 q[id]=local*fractions[rank];
}

// Each aligned warp lies inside one1024-token tile. Compute its invariant
// FP64 quotient once, preserving the two original multiplication boundaries.
//32320 is a multiple of32, so the final active warp is also complete. Form
// the quotient before any per-logit return so every active lane participates.
extern "C" __global__ void ca_hierarchy_q_warp(const float* logits,const float* maxima,
 const double* norm,const double* factors,const double* fractions,
 double* q,const uint32_t* error,int rank,float temperature) {
 int id=blockIdx.x*256+threadIdx.x;if(id>=32320)return;
 if(error[0]){q[id]=NAN;return;}
 double coefficient=0.;
 if((threadIdx.x&31)==0)coefficient=factors[id/1024]/norm[1];
 coefficient=__shfl_sync(0xffffffffu,coefficient,0);
 const float x=logits[id],m=maxima[id/1024];
 if(!isfinite(x)||!isfinite(m)){q[id]=0.;return;}
 const double local=double(expf((x-m)/temperature))*coefficient;
 q[id]=local*fractions[rank];
}
