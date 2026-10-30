// Reversible GPU-only layout permutation. Canonical checkpoint bytes stay intact.
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void permute(uint32_t* words, uint64_t count, bool inverse) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=uint64_t(gridDim.x)*blockDim.x) {
        uint32_t x=words[i],y=0;
        #pragma unroll
        for(int j=0;j<8;++j) {
            int source=(j&1)*4+(j>>1);
            if(inverse)y|=((x>>(j*4))&15u)<<(source*4);
            else y|=((x>>(source*4))&15u)<<(j*4);
        }
        words[i]=y;
    }
}
extern "C" int ca_moe_nibble_permute(void* data,uint64_t bytes,int inverse,void* stream) {
    if(!data || bytes%4 || (inverse!=0 && inverse!=1))return -1;
    permute<<<256,256,0,(cudaStream_t)stream>>>((uint32_t*)data,bytes/4,inverse!=0);
    return int(cudaGetLastError());
}
