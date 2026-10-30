// Isolated GPU route-selection experiment; no serving integration.
// CUDA conditional-node API reference:
// https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
#include <cuda_runtime.h>
#include <cstdint>
#include <new>

__global__ void choose_moe(cudaGraphConditionalHandle handle, const int* ids,
                           unsigned* decisions, int call, int force) {
    const int t=threadIdx.x;
    const int id=t<36 ? ids[t] : -1;
    bool first=id>=0;
    #pragma unroll
    for(int i=0;i<36;++i) if(i<t && id>=0 && ids[i]==id) first=false;
    const unsigned distinct=__popc(__ballot_sync(0xffffffff,first));
    const unsigned active=__popc(__ballot_sync(0xffffffff,id>=0));
    __shared__ unsigned counts[4];
    if((t&31)==0){counts[(t>>5)*2]=distinct;counts[(t>>5)*2+1]=active;}
    __syncthreads();
    if(t==0){
        const unsigned unique=counts[0]+counts[2],rows=(counts[1]+counts[3])/6;
        const bool micro=force==2 || (force==0 && (rows<=1 || unique>=2*rows+4));
        decisions[call]=(unsigned)micro | (unique<<8) | (rows<<16);
        cudaGraphSetConditional(handle,micro?1:0);
    }
}
struct Holder {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    ~Holder(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}
};
#define TRY(expr) do {cudaError_t rc=(expr);if(rc!=cudaSuccess){delete p;return (int)rc;}} while(0)
extern "C" int ca_moe_cond_create(void* grouped,void* micro,int* ids,
                                  const int* rotating,unsigned* decisions,
                                  int force,void** output){
    if(!grouped || !micro || !ids || !rotating || !decisions || !output || force<0 || force>2)return -1;
    *output=nullptr;
    auto* p=new(std::nothrow) Holder;if(!p)return -2;
    TRY(cudaGraphCreate(&p->graph,0));
    cudaGraphNode_t previous=nullptr;
    for(int call=0;call<16;++call){
        cudaGraphNode_t copy,select,condition,child;
        TRY(cudaGraphAddMemcpyNode1D(&copy,p->graph,previous?&previous:nullptr,previous?1:0,
            ids,rotating+(call%8)*36,36*sizeof(int),cudaMemcpyDeviceToDevice));
        cudaGraphConditionalHandle handle;
        TRY(cudaGraphConditionalHandleCreate(&handle,p->graph,0,cudaGraphCondAssignDefault));
        void* args[]={&handle,&ids,&decisions,&call,&force};
        cudaKernelNodeParams k{};k.func=(void*)choose_moe;k.gridDim=dim3(1);k.blockDim=dim3(64);k.kernelParams=args;
        TRY(cudaGraphAddKernelNode(&select,p->graph,&copy,1,&k));
        cudaGraphNodeParams c{};c.type=cudaGraphNodeTypeConditional;
        c.conditional.handle=handle;c.conditional.type=cudaGraphCondTypeIf;c.conditional.size=2;
        TRY(cudaGraphAddNode(&condition,p->graph,&select,nullptr,1,&c));
        TRY(cudaGraphAddChildGraphNode(&child,c.conditional.phGraph_out[0],nullptr,0,(cudaGraph_t)micro));
        TRY(cudaGraphAddChildGraphNode(&child,c.conditional.phGraph_out[1],nullptr,0,(cudaGraph_t)grouped));
        previous=condition;
    }
    TRY(cudaGraphInstantiate(&p->exec,p->graph,0));
    *output=p;return 0;
}
extern "C" int ca_moe_cond_launch(void* handle,void* stream){
    if(!handle)return -1;
    return (int)cudaGraphLaunch(((Holder*)handle)->exec,(cudaStream_t)stream);
}
extern "C" void ca_moe_cond_close(void* handle){delete (Holder*)handle;}
