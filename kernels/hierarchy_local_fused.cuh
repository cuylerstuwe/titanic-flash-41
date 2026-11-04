// Same conditional tile law, local CDF and wire packing as the three original
// kernels. All CTA threads reach both barriers, including invalid/empty shards.
extern "C" __global__ void ca_hierarchy_local_fused(
 const float* logits,const float* maxima,const double* sums,const double* uniforms,
 int32_t* selected,double* fractions,double* normalization,double* factors,
 uint32_t* errors,int32_t* output,CaHierarchyDraw* packed,float temperature) {
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
    factors[row*32+t]=factor;mass[t]=sums[row*32+t]*factor;
  }
  __syncthreads();
  if(t==0) {
    selected[row]=-1;fractions[row]=0.;
    double sum=0.;for(int i=0;i<32;++i)sum+=mass[i];
    normalization[row*2]=double(m);normalization[row*2+1]=sum;
    double u=uniforms[row];
    if(!isfinite(sum)||sum<=0.)errors[row]|=2u;
    if(!isfinite(u)||u<0.||u>=1.)errors[row]|=4u;
    if(!errors[row]) {
      double target=fmin(u*sum,nextafter(sum,0.)),before=0.;
      for(int i=0;i<32;++i) {
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
  ca_hierarchy_local_cdf_body<cub::BLOCK_SCAN_RAKING,false>(
    logits,maxima,selected,fractions,errors,output,temperature,nullptr);
  __syncthreads();
  if(t==0) {
    double q=0.;int token=output[row];
    if(!errors[row]&&token>=0&&token<32320)
      q=double(expf((logits[(size_t)row*32320+token]-maxima[row*32+token/1024])/temperature))*
        (factors[row*32+token/1024]/normalization[row*2+1]);
    packed[row]={normalization[row*2],normalization[row*2+1],token,errors[row],q};
  }
}
