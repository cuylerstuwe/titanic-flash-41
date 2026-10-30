// Isolated next-round draft policy. Inputs are complete root p/q overlaps;
// no sampled token or RNG enters this decision, and the current draw is final.
extern "C" __global__ void ca_root_feedback_finish(const double* sums,double* metrics,
 uint32_t* choice,uint32_t* errors) {
  if(threadIdx.x)return;
  double cold=0.,hot=0.;
  for(int tile=0;tile<127;++tile){cold+=sums[tile];hot+=sums[127+tile];}
  if(errors[0]||!isfinite(cold)||!isfinite(hot)||cold < -1.e-10||cold > 1.+1.e-10
      ||hot < -1.e-10||hot > 1.+1.e-10||choice[0]>1){errors[0]|=1024u;return;}
  double delta=hot-cold;
  double ema=choice[1] ? .5*metrics[2]+.5*delta : delta;
  metrics[0]=cold;metrics[1]=hot;metrics[2]=ema;
  if(ema>.01)choice[0]=1;
  else if(ema<-.01)choice[0]=0;
  ++choice[1];
}
