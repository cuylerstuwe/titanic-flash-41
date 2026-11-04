// The first word of the left-boundary area is unused on every TP4 rank:
// rank0 has no left boundary; the other lower bounds have nonzero offsets
// 576,128,704. Original global tiles and reduction arithmetic stay intact.
extern "C" __global__ void ca_target_proposal_pack(const float* x,const int* props,float* packed,int rank,int rows) {
 int row=threadIdx.x;if(row>=rows)return;int id=props[row];
 packed[row*CA_SHARD_WORDS+128]=id>=rank*32320&&id<(rank+1)*32320?x[row*32320+id-rank*32320]:-INFINITY;
}
extern "C" __global__ void ca_target_local_factors(const float* gathered,const float* maxima,
 const double* norm,const double* factors,const int* props,const uint32_t* error,
 double* normalized,double* p,double* selected,float temperature) {
 int row=blockIdx.x,t=threadIdx.x;
 if(t<127)normalized[row*127+t]=factors[row*127+t]/norm[row*2+1];
 __syncthreads();
 if(t==0) {
  int id=props[row];double value=NAN;
  if(!error[row]&&id>=0&&id<129280) {
   float x=gathered[((size_t)row*4+id/32320)*CA_SHARD_WORDS+128],m=maxima[row*127+id/1024];
   value=isfinite(x)&&isfinite(m)?__dmul_rn(double(expf((x-m)/temperature)),normalized[row*127+id/1024]):0.;
  }
  selected[row]=value;if(id>=0&&id<129280)p[(size_t)row*129280+id]=value;
 }
}
extern "C" __global__ void ca_target_local_probabilities(const float* x,const float* maxima,
 const double* normalized,const uint32_t* error,double* p,int rank,int rows,float temperature) {
 int id=blockIdx.x*256+threadIdx.x,row=blockIdx.y;if(id>=32320||row>=rows)return;
 int global=rank*32320+id;float m=maxima[row*127+global/1024],v=x[(size_t)row*32320+id];
 p[(size_t)row*129280+global]=error[row]?NAN:
   isfinite(m)&&isfinite(v)?__dmul_rn(double(expf((v-m)/temperature)),normalized[row*127+global/1024]):0.;
}
