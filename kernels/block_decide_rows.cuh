// Independent block rows have no arithmetic dependencies. Use one warp lane
// per row while retaining each row's original127-term FP64 accumulation order.
// The original serial kernel remains the explicit control and supports wider blocks.
extern "C" __global__ void ca_block_decide_rows(const double* w,const double* sums,
 const int32_t* proposals,const double* uniforms,const int32_t* bonus,
 uint32_t* state,int32_t* out,uint32_t* errors,int drafts) {
  const int row=threadIdx.x;
  if(blockDim.x!=32||drafts<0||drafts>=32){if(row==0)errors[0]|=256u;return;}
  bool accepted=false;
  if(row>0&&row<=drafts){
    double h;
    if(row==drafts)h=w[row];
    else {
      double mass=0.;for(int t=0;t<127;++t)mass+=sums[row*127+t];
      double denominator=mass+(1.-w[row]);
      h=denominator>0.?mass/denominator:0.;
    }
    const double u=uniforms[row-1];
    if(!isfinite(h)||h<0.||h>1.||!isfinite(u)||u<0.||u>=1.)errors[row]|=8u;
    accepted=u<h;
  }
  const unsigned mask=__ballot_sync(0xffffffffu,accepted);
  const int tau=mask?31-__clz(mask):0;
  if(row<=drafts){
    state[row]=row<tau?1u:(row==tau&&tau<drafts?0u:2u);
    out[row]=row<tau?proposals[row]:bonus[row];
  }
}
