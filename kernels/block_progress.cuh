// Optional progress telemetry fused into the ordinary block decision.
// Same Sun et al. recurrence and draw as block_rejection.cuh. The observed
// draft path is held fixed: expectation integrates only the verifier's
// acceptance uniforms. It is NOT a marginal overlap product, or permission to
// choose this round's length after seeing its target probabilities.
// A future-round controller may consume these values after commit.
extern "C" __global__ void ca_block_decide_progress(
 const double* w,const double* sums,const int32_t* proposals,
 const double* uniforms,const int32_t* bonus,uint32_t* state,int32_t* out,
 uint32_t* errors,int drafts,double* expected) {
  if(threadIdx.x)return;
  if(drafts<0||drafts>5){errors[0]|=256u;return;}
  double h[5];int tau=0;
  for(int i=1;i<=drafts;++i){
    double mass=0.;for(int t=0;t<127;++t)mass+=sums[i*127+t];
    double denominator=mass+(1.-w[i]);
    double value=i==drafts?w[i]:(denominator>0.?mass/denominator:0.);
    double u=uniforms[i-1];h[i-1]=value;
    if(!isfinite(value)||value<0.||value>1.||!isfinite(u)||u<0.||u>=1.)errors[i]|=8u;
    if(u<value)tau=i;
  }
  for(int i=0;i<=drafts;++i){
    state[i]=i<tau?1u:(i==tau&&tau<drafts?0u:2u);
    out[i]=i<tau?proposals[i]:bonus[i];
  }
  expected[0]=1.;
  for(int n=1;n<=drafts;++n){
    double none_later=1.,length=1.;
    for(int i=n;i>=1;--i){
      // The last row of a shorter prefix uses w[n], not the full block's h[n].
      double value=i==n?w[n]:h[i-1];
      if(!isfinite(value)||value<0.||value>1.){errors[0]|=256u;return;}
      none_later*=1.-value;length+=1.-none_later;
    }
    expected[n]=length;
  }
}

// Isolated NEXT-round policy. Probability-derived estimates stay on the GPU;
// the CPU supplies only the elapsed time of the completed committed step.
// state: expected lengths[3,6], elapsed seconds[3,6]. counts: observations per
// arm, total observations, last selected rows. Readout has four u32 words,
// then six current expectations and four state doubles.
extern "C" __global__ void ca_block_progress_update(const double* expected,
 double* state,uint32_t* counts,uint32_t* readout,int rows,double seconds) {
  if(threadIdx.x)return;
  readout[0]=0;
  if((rows!=3&&rows!=6)||!isfinite(seconds)||seconds<=0.){readout[0]=1;return;}
  int arm=rows==6;const double alpha=.25;
  for(int i=0;i<=arm;++i){
    double e=expected[i?5:2];
    if(!isfinite(e)||e<1.||e>(i?6.:3.)){readout[0]=2;return;}
    state[i]=state[i]>0.?state[i]+alpha*(e-state[i]):e;
  }
  state[2+arm]=counts[arm]?state[2+arm]+alpha*(seconds-state[2+arm]):seconds;
  counts[arm]++;counts[2]++;
  int next=rows;
  if(counts[0]<2||counts[1]<2)next=rows==3?6:3;
  else {
    double short_rate=state[0]/state[2],long_rate=state[1]/state[3];
    if(rows==3&&long_rate>short_rate*1.03)next=6;
    if(rows==6&&short_rate>long_rate*1.03)next=3;
    // Periodically refresh an unobserved arm, with a bounded one-round probe.
    if(counts[2]%8==0)next=rows==3?6:3;
  }
  counts[3]=next;readout[1]=next;readout[2]=counts[2];readout[3]=rows;
  double* values=reinterpret_cast<double*>(readout+4);
  for(int i=0;i<6;++i)values[i]=i<rows?expected[i]:0.;
  for(int i=0;i<4;++i)values[6+i]=state[i];
}

extern "C" __global__ void ca_block_progress_compare(const uint32_t* control_state,
 const int32_t* control_ids,const uint32_t* state,const int32_t* ids,uint32_t* errors,int live) {
  int i=threadIdx.x;if(i<live&&(control_state[i]!=state[i]||control_ids[i]!=ids[i]))atomicOr(errors,1024u);
}
