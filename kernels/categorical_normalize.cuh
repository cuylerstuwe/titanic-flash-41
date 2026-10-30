// Component candidate: amortize the common FP64 normalization division.
// No logit quantization, probability truncation or FP32 probability storage.
template<int Mode>
__device__ __forceinline__ void ca_categorical_probabilities_amortized_body(
 const float* logits,const float* maxima,const double* normalization,
 const double* factors,double* probabilities,const uint32_t* errors,float temperature) {
  const int row=blockIdx.y,id=blockIdx.x*256+threadIdx.x,tile=blockIdx.x/4;
  if(id>=129280)return;
  if(errors[row]){probabilities[(size_t)row*129280+id]=NAN;return;}
  const float local=maxima[row*127+tile];
  if(!isfinite(local)){probabilities[(size_t)row*129280+id]=0.;return;}
  const double total=normalization[row*2+1],factor=factors[row*127+tile];
  __shared__ double shared;
  if(threadIdx.x==0)shared=Mode==1?factor/total:1./total;
  __syncthreads();
  const float x=logits[(size_t)row*129280+id];
  const double mass=isfinite(x)?double(expf((x-local)/temperature)):0.;
  double q;
  if constexpr(Mode==1)q=__dmul_rn(mass,shared);
  else {
    // Retain the original rounded numerator, then correct its reciprocal
    // product with a fused remainder. Compare against ordinary division.
    const double weighted=__dmul_rn(mass,factor);
    const double initial=__dmul_rn(weighted,shared);
    const double residual=__fma_rn(-initial,total,weighted);
    q=__fma_rn(residual,shared,initial);
  }
  probabilities[(size_t)row*129280+id]=q;
}
template<int Mode>
__global__ void ca_categorical_probabilities_amortized(
 const float* x,const float* m,const double* n,const double* f,
 double* q,const uint32_t* e,float t) {
  ca_categorical_probabilities_amortized_body<Mode>(x,m,n,f,q,e,t);
}
extern "C" __global__ void ca_categorical_probabilities_normalized(
 const float* x,const float* m,const double* n,const double* f,
 double* q,const uint32_t* e,float t) {
  ca_categorical_probabilities_amortized_body<1>(x,m,n,f,q,e,t);
}
