// Original diagnostic using the installed NCCL2.30.7 public C API.
// API reference: NVIDIA/nccl v2.30.7-1 docs/userguide/source/usage/bufferreg.rst.
// No model arithmetic: generate and check transport bytes entirely on GPU.
#include <nccl.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
extern "C" int zg_init(void** output, const void* id, int rank, int zero) {
  ncclUniqueId uid; std::memcpy(&uid,id,sizeof(uid));
  ncclConfig_t config=NCCL_CONFIG_INITIALIZER;
  config.CTAPolicy=zero?NCCL_CTA_POLICY_ZERO:NCCL_CTA_POLICY_DEFAULT;
  return ncclCommInitRankConfig(reinterpret_cast<ncclComm_t*>(output),4,uid,rank,&config);
}
__device__ unsigned char value(size_t i,int rank,int pattern) {
  return static_cast<unsigned char>((i*17+(i>>8)*13+rank*61+pattern*37)&255);
}
__global__ void fill_bytes(unsigned char* p,size_t n,int rank,int pattern) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)p[i]=value(i,rank,pattern);
}
__global__ void check_bytes(const unsigned char* p,size_t n,int pattern,unsigned* bad) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<4*n;i+=gridDim.x*blockDim.x)
    if(p[i]!=value(i%n,i/n,pattern))atomicExch(bad,1u);
}
extern "C" int zg_fill(void* p,size_t n,int rank,int pattern,void* stream) {
  fill_bytes<<<128,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<unsigned char*>(p),n,rank,pattern);
  return cudaGetLastError();
}
extern "C" int zg_check(const void* p,size_t n,int pattern,unsigned* bad,void* stream) {
  cudaError_t e=cudaMemsetAsync(bad,0,4,static_cast<cudaStream_t>(stream));if(e)return e;
  check_bytes<<<128,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const unsigned char*>(p),n,pattern,bad);
  return cudaGetLastError();
}
// Positive errors are CUDA codes; negative errors are NCCL codes.
extern "C" int zg_capture(void* comm,const void* src,void* dst,size_t n,void* stream,
    void** graph,void** exec,unsigned* node_counts) {
  auto s=static_cast<cudaStream_t>(stream);
  cudaError_t e=cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal);if(e)return e;
  auto status=ncclAllGather(src,dst,n,ncclUint8,static_cast<ncclComm_t>(comm),s);
  if(status!=ncclSuccess)return -static_cast<int>(status);
  cudaGraph_t g; e=cudaStreamEndCapture(s,&g);if(e)return e;
  size_t n_nodes=0;e=cudaGraphGetNodes(g,nullptr,&n_nodes);if(e)return e;
  std::vector<cudaGraphNode_t> nodes(n_nodes);e=cudaGraphGetNodes(g,nodes.data(),&n_nodes);if(e)return e;
  std::memset(node_counts,0,4*sizeof(unsigned));node_counts[0]=n_nodes;
  for(auto node:nodes){cudaGraphNodeType type;e=cudaGraphNodeGetType(node,&type);if(e)return e;
    node_counts[1]+=(type==cudaGraphNodeTypeKernel);node_counts[2]+=(type==cudaGraphNodeTypeHost);
    node_counts[3]+=(type==cudaGraphNodeTypeMemcpy);}
  cudaGraphExec_t executable;e=cudaGraphInstantiate(&executable,g,0);if(e)return e;
  *graph=g;*exec=executable;return 0;
}
