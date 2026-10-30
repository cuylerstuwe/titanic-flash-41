// Original integer-only sampler result transport. Run after categorical and
// rejection writes on the same stream, then synchronize before host access.
extern "C" __global__ void ca_sampler_readout(const uint32_t* target_errors,
 const int32_t* target_ids,const uint32_t* rejection_errors,const int32_t* ids,
 const uint32_t* state,uint32_t* packed,int capacity) {
 int i=threadIdx.x+blockIdx.x*blockDim.x;
 if(i>=capacity)return;
 packed[i*5]=target_errors[i];packed[i*5+1]=(uint32_t)target_ids[i];
 packed[i*5+2]=rejection_errors[i];packed[i*5+3]=(uint32_t)ids[i];
 packed[i*5+4]=state[i];
}
