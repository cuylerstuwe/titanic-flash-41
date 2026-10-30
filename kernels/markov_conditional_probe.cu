// Isolated draft-work control; no target or proposal arithmetic changes.
// NVIDIA CUDA Graph IF API, same design as our moe_conditional_probe.cu.
// https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
#include <cuda_runtime.h>
#include <cmath>
#include <new>

__global__ void choose_tail(cudaGraphConditionalHandle handle,const float* consensus,
                            unsigned* result,int force,float threshold) {
    if(threadIdx.x==0){
        // consensus[0..2] is rank0's gathered confidence, on every rank.
        bool valid=true;float cumulative=1.f;
        for(int i=0;i<3;++i){float p=consensus[i];valid=valid&&isfinite(p)&&p>=0.f&&p<=1.f;
            cumulative=__fmul_rn(cumulative,p);}
        bool take=valid&&(force==2||(force==0&&cumulative>=threshold));
        result[0]=take?1u:0u;result[1]=valid?0u:1u;
        result[2]=cumulative<threshold?1u:0u;result[3]=0x4d434f4eu;
        cudaGraphSetConditional(handle,take?1:0);
    }
}
__global__ void finish_readout(const unsigned* source,const unsigned* decision,unsigned* output){
    unsigned i=threadIdx.x;
    if(i<32)output[i]=source[i];
    else if(i<36)output[i]=decision[i-32];
}
struct Holder {
    cudaGraph_t graph{};cudaGraphExec_t exec{};
    ~Holder(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}
};
#define TRY(expr) do{cudaError_t rc=(expr);if(rc!=cudaSuccess){delete p;return(int)rc;}}while(0)
extern "C" int ca_markov_capture_begin(void* stream){
    return(int)cudaStreamBeginCapture((cudaStream_t)stream,cudaStreamCaptureModeGlobal);
}
extern "C" int ca_markov_capture_end(void* stream,void** output){
    return(int)cudaStreamEndCapture((cudaStream_t)stream,(cudaGraph_t*)output);
}
extern "C" void ca_markov_capture_close(void* graph){if(graph)cudaGraphDestroy((cudaGraph_t)graph);}
static int create_graph(void* prefix,void* tail,const float* consensus,
    unsigned* result,int force,float threshold,const unsigned* readout,unsigned* combined,void** output){
    if(!prefix||!tail||!consensus||!result||!output||force<0||force>2
        ||!std::isfinite(threshold)||threshold<0.f||threshold>1.f||bool(readout)!=bool(combined))return -1;
    *output=nullptr;auto* p=new(std::nothrow) Holder;if(!p)return -2;
    TRY(cudaGraphCreate(&p->graph,0));
    cudaGraphNode_t first,select,condition,child;
    TRY(cudaGraphAddChildGraphNode(&first,p->graph,nullptr,0,(cudaGraph_t)prefix));
    cudaGraphConditionalHandle handle;
    TRY(cudaGraphConditionalHandleCreate(&handle,p->graph,0,cudaGraphCondAssignDefault));
    void* args[]={&handle,&consensus,&result,&force,&threshold};
    cudaKernelNodeParams k{};k.func=(void*)choose_tail;k.gridDim=dim3(1);k.blockDim=dim3(32);k.kernelParams=args;
    TRY(cudaGraphAddKernelNode(&select,p->graph,&first,1,&k));
    cudaGraphNodeParams c{};c.type=cudaGraphNodeTypeConditional;
    c.conditional.handle=handle;c.conditional.type=cudaGraphCondTypeIf;c.conditional.size=1;
    TRY(cudaGraphAddNode(&condition,p->graph,&select,nullptr,1,&c));
    TRY(cudaGraphAddChildGraphNode(&child,c.conditional.phGraph_out[0],nullptr,0,(cudaGraph_t)tail));
    if(combined){
        cudaGraphNode_t finish;void* args[]={&readout,&result,&combined};
        cudaKernelNodeParams f{};f.func=(void*)finish_readout;f.gridDim=dim3(1);f.blockDim=dim3(64);f.kernelParams=args;
        TRY(cudaGraphAddKernelNode(&finish,p->graph,&condition,1,&f));
    }
    TRY(cudaGraphInstantiate(&p->exec,p->graph,0));
    *output=p;return 0;
}
extern "C" int ca_markov_cond_create(void* prefix,void* tail,const float* consensus,
                                     unsigned* result,int force,void** output){
    return create_graph(prefix,tail,consensus,result,force,0.2f,nullptr,nullptr,output);
}
extern "C" int ca_markov_cond_create_readout(void* prefix,void* tail,const float* consensus,
    unsigned* result,float threshold,const unsigned* readout,unsigned* combined,void** output){
    return create_graph(prefix,tail,consensus,result,0,threshold,readout,combined,output);
}
extern "C" int ca_markov_cond_launch(void* handle,void* stream){
    if(!handle)return -1;
    return(int)cudaGraphLaunch(((Holder*)handle)->exec,(cudaStream_t)stream);
}
extern "C" void ca_markov_cond_close(void* handle){delete(Holder*)handle;}
