// Experimental wrapper only; native serving calls the same device kernels.
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "candidate_block_scores.cuh"
extern "C" int ca_candidate_probe(int tiles,const float* scores,int stride,
 const int32_t* starts,const int32_t* lengths,float* blocks,int block_stride,
 int row0,int rows,int nblocks,cudaStream_t stream){
 if(tiles==0)ca_candidate_block_scores<<<(rows*nblocks+255)/256,256,0,stream>>>(scores,stride,starts,lengths,blocks,block_stride,row0,rows,nblocks);
 else ca_candidate_block_scores_bounded<<<dim3(tiles,rows),256,0,stream>>>(scores,stride,starts,lengths,blocks,block_stride,row0,rows,nblocks);
 return(int)cudaGetLastError();
}
