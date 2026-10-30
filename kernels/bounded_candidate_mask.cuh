// Exact candidate masking over live8-key blocks. GPU component screen s355.
// Preserve unaligned windows and partial blocks; no score arithmetic changes.
extern "C" __global__ void ca_apply_candidate_mask_bounded(float* scores,int stride,const int32_t* starts,
 const int32_t* lengths,const uint8_t* mask,int mask_stride,int mask_rows){
 int r=blockIdx.y,m=mask_rows+r,n=lengths[m],start=starts[m];
 for(int b=blockIdx.x*blockDim.x+threadIdx.x;b<(n+7)/8;b+=gridDim.x*blockDim.x){
  if(mask[(size_t)m*mask_stride+b])continue;
  int p=8*b;float* dst=scores+(size_t)r*stride+start+p;
  if(p+8<=n&&((uintptr_t)dst&15)==0){
   float4 value=make_float4(-INFINITY,-INFINITY,-INFINITY,-INFINITY);
   reinterpret_cast<float4*>(dst)[0]=value;reinterpret_cast<float4*>(dst)[1]=value;
  }else for(int k=0;k<8&&p+k<n;k++)dst[k]=-INFINITY;
 }
}
