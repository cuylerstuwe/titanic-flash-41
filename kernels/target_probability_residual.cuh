// Preserve canonical p and the residual reduction tree, but generate each
// probability in the consuming partial's registers instead of rereading it.
// The selected proposal p was published by ca_target_tile_factors_fused before
// block weights are calculated; this writes the same full local p as before.
extern "C" __global__ void ca_target_probability_residual(
 const float* x,const float* maxima,const double* normalized,const uint32_t* target_error,
 double* p,const double* q,const double* w,double* sums,uint32_t* errors,int drafts,int rank) {
 int row=blockIdx.y,tile=blockIdx.x,t=threadIdx.x;double sum=0.;
 for(int off=t;off<1024;off+=256) {
  int id=tile*1024+off;if(id>=32320)continue;
  int global=rank*32320+id;
  float m=maxima[row*127+global/1024],v=x[(size_t)row*32320+id];
  double a=target_error[row]?NAN:isfinite(m)&&isfinite(v)?
    __dmul_rn(double(expf((v-m)/1.f)),normalized[row*127+global/1024]):0.;
  p[(size_t)row*129280+global]=a;
  double b=row<drafts?q[(size_t)row*32320+id]:0.;
  if(!isfinite(a)||!isfinite(b)||a<0||b<0)atomicOr(errors+row,4u);
  else sum+=fmax(w[row]*a-b,0.);
 }
 sum=ca_categorical_sum(sum);if(t==0)sums[row*32+tile]=sum;
}
