// Existing canonical reduction, plus an exact live-window schedule.
extern "C" __global__ void ca_candidate_block_scores(const float* scores,int stride,
 const int32_t* starts,const int32_t* lengths,float* blocks,int block_stride,
 int row0,int rows,int nblocks){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*nblocks)return;
 int r=i/nblocks,b=i%nblocks,g=row0+r,len=lengths[g];float best=-INFINITY;
 if(len>0){int last=(len-1)/8;
  if(b==last)best=INFINITY;
  else if(b<last){const float* row=scores+(size_t)r*stride+starts[g]+b*8;
   for(int j=0;j<8;j++){float v=row[j];best=(v>best||v!=v)?v:best;}}}
 blocks[(size_t)g*block_stride+b]=best;
}
// Only reachable block maxima are consumed by ragged top-k. Leave padding
// untouched, including after a shorter replay; the published mask is cleared
// independently. The order of all eight comparisons stays canonical.
extern "C" __global__ void ca_candidate_block_scores_bounded(const float* scores,int stride,
 const int32_t* starts,const int32_t* lengths,float* blocks,int block_stride,
 int row0,int rows,int nblocks){
 int r=blockIdx.y;if(r>=rows)return;
 int g=row0+r,len=lengths[g],live=min(nblocks,(len+7)/8),last=(len-1)/8;
 for(int b=blockIdx.x*blockDim.x+threadIdx.x;b<live;b+=gridDim.x*blockDim.x){
  float best=-INFINITY;
  if(b==last)best=INFINITY;
  else{
   const float* row=scores+(size_t)r*stride+starts[g]+b*8;
   if(((uintptr_t)row&15)==0){
    float4 a=*(const float4*)row,c=*(const float4*)(row+4);
    float v[8]={a.x,a.y,a.z,a.w,c.x,c.y,c.z,c.w};
    #pragma unroll
    for(int j=0;j<8;j++)best=(v[j]>best||v[j]!=v[j])?v[j]:best;
   }else{
    #pragma unroll
    for(int j=0;j<8;j++){float v=row[j];best=(v>best||v!=v)?v:best;}
   }
  }
  blocks[(size_t)g*block_stride+b]=best;
 }
}
