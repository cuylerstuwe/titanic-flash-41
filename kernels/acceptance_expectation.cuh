// Diagnostic conditional expected committed lengths on the actual draft path.
// Token rejection: 1 + sum(prefix products of min(1,p(x)/q(x))).
// Block rejection: 1 + sum(Pr[tau>=i]), using the same Sun et al. recurrence
// as block_rejection.cuh. No random draws or sampled output are modified.
// Prefix-length alternatives are offline diagnostics, not permission to pick
// a proposal length using future sampled tokens or their target probabilities.
extern "C" __global__ void ca_acceptance_expected_lengths(
 const double* p,const double* q,const int32_t* proposals,const double* w,
 const double* sums,double* expected,uint32_t* errors,int drafts) {
  if(threadIdx.x)return;
  if(drafts<0||drafts>127){errors[0]|=256u;return;}
  expected[0]=expected[1]=1.;
  double survival=1.,token_length=1.;
  for(int n=1;n<=drafts;++n){
    int id=proposals[n-1];
    if(id<0||id>=129280){errors[0]|=256u;return;}
    double a=p[(size_t)(n-1)*129280+id],b=q[(size_t)(n-1)*129280+id];
    if(!isfinite(a)||!isfinite(b)||a<0.||b<=0.){errors[0]|=256u;return;}
    survival*=a>=b?1.:a/b;
    token_length+=survival;
    double none_later=1.,block_length=1.;
    for(int i=n;i>=1;--i){
      double mass=0.;for(int t=0;t<127;++t)mass+=sums[i*127+t];
      double denominator=mass+(1.-w[i]);
      double h=i==n?w[i]:(denominator>0.?mass/denominator:0.);
      if(!isfinite(h)||h<0.||h>1.){errors[0]|=256u;return;}
      none_later*=1.-h;
      block_length+=1.-none_later;
    }
    expected[2*n]=token_length;expected[2*n+1]=block_length;
  }
  // Row zero has w[0]=1, so its already-computed positive residual
  // mass is TV(p,q). This marginal overlap integrates over every possible
  // first proposal and is independent of the particular sampled draft token.
  // It is diagnostic metadata, never a same-round draft-selection rule.
  double residual=0.;for(int t=0;t<127;++t)residual+=sums[t];
  double overlap=1.-residual;
  if(!isfinite(overlap)||overlap < -1.e-12||overlap > 1.+1.e-12){errors[0]|=256u;return;}
  expected[2*(drafts+1)]=fmin(1.,fmax(0.,overlap));
}

// Separate shadow pass: integrate each next-token overlap at its sampled
// ancestor prefix. These rows are path-dependent; their product is a heuristic,
// not the block verifier's expected accepted length or a same-round policy.
extern "C" __global__ void ca_acceptance_local_partials(
 const double* p,const double* q,double* sums,uint32_t* errors,int drafts) {
  int row=blockIdx.y,tile=blockIdx.x,t=threadIdx.x;if(row>=drafts)return;
  double sum=0.;
  for(int off=t;off<1024;off+=256){int id=tile*1024+off;if(id>=129280)continue;
    double a=p[(size_t)row*129280+id],b=q[(size_t)row*129280+id];
    if(!isfinite(a)||!isfinite(b)||a<0.||b<0.)atomicOr(errors+row,512u);
    else sum+=fmin(a,b);
  }
  sum=ca_categorical_sum(sum);if(t==0)sums[row*127+tile]=sum;
}
extern "C" __global__ void ca_acceptance_local_finish(
 const double* p,const double* q,const int32_t* proposals,const double* sums,
 double* expected,uint32_t* errors,int drafts) {
  if(threadIdx.x)return;
  const int base=2*(drafts+1)+1;
  double survival=1.,rb_length=1.,product=1.,product_score=1.;
  for(int row=0;row<drafts;++row){
    double overlap=0.;for(int tile=0;tile<127;++tile)overlap+=sums[row*127+tile];
    if(!isfinite(overlap)||overlap < -1.e-12||overlap > 1.+1.e-12){errors[0]|=512u;return;}
    overlap=fmin(1.,fmax(0.,overlap));expected[base+row]=overlap;
    // Integrate the final proposed token of each sequential-rejection prefix,
    // while retaining sampled ancestors. Unbiased for that sequential rule
    // over independent q draws; not an unbiased block-acceptance estimator.
    rb_length+=survival*overlap;
    product*=overlap;product_score+=product;
    int id=proposals[row];
    if(id<0||id>=129280){errors[0]|=512u;return;}
    double a=p[(size_t)row*129280+id],b=q[(size_t)row*129280+id];
    if(!isfinite(a)||!isfinite(b)||a<0.||b<=0.){errors[0]|=512u;return;}
    survival*=a>=b?1.:a/b;
  }
  expected[base+drafts]=rb_length;
  expected[base+drafts+1]=product_score;
}
