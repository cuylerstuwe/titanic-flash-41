// Isolated FFN output-chunk experiment. No serving path uses this bridge.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstring>
#include <new>
#include <vector>

// Match the existing six-route FP32 sum, BF16 boundary, then shared BF16 add.
// Contiguous half buffers avoid communicating a strided [6,2560] view.
__global__ void pack_ffn_half(const __nv_bfloat16* routes,
                             const __nv_bfloat16* shared,
                             __nv_bfloat16* packed,int half) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=6*2560)return;
    int row=i/2560,col=i%2560+half*2560;
    float sum=0.f;
    #pragma unroll
    for(int route=0;route<6;++route)
        sum=__fadd_rn(sum,__bfloat162float(routes[(row*6+route)*5120+col]));
    float routed=__bfloat162float(__float2bfloat16_rn(sum));
    packed[i]=__float2bfloat16_rn(__fadd_rn(routed,__bfloat162float(shared[row*5120+col])));
}
__global__ void join_ffn_halves(const __nv_bfloat16* low,const __nv_bfloat16* high,
                              __nv_bfloat16* output) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=6*5120)return;
    int row=i/5120,col=i%5120;
    output[i]=col<2560?low[row*2560+col]:high[row*2560+col-2560];
}
extern "C" int ca_ffn_chunk_pack(const void* routes,const void* shared,void* packed,
                                  int half,void* stream) {
    if(!routes||!shared||!packed||(half!=0&&half!=1))return -1;
    pack_ffn_half<<<60,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)routes,
        (const __nv_bfloat16*)shared,(__nv_bfloat16*)packed,half);
    return (int)cudaGetLastError();
}
extern "C" int ca_ffn_chunk_join(const void* low,const void* high,void* output,void* stream) {
    if(!low||!high||!output)return -1;
    join_ffn_halves<<<120,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)low,
        (const __nv_bfloat16*)high,(__nv_bfloat16*)output);
    return (int)cudaGetLastError();
}
extern "C" int ca_ffn_chunk_capture_begin(void* stream) {
    return (int)cudaStreamBeginCapture((cudaStream_t)stream,cudaStreamCaptureModeGlobal);
}
extern "C" int ca_ffn_chunk_capture_end(void* stream,void** graph) {
    return (int)cudaStreamEndCapture((cudaStream_t)stream,(cudaGraph_t*)graph);
}
extern "C" void ca_ffn_chunk_capture_close(void* graph) { if(graph)cudaGraphDestroy((cudaGraph_t)graph); }

struct Holder {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    cudaGraph_t parts[4]{};
    ~Holder(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);
        for(auto g:parts)if(g)cudaGraphDestroy(g);}
};
#define TRY(expr) do{int rc=(int)(expr);if(rc){delete p;return rc;}}while(0)
#define REQUIRE(expr,code) do{if(!(expr)){delete p;return code;}}while(0)

static bool reaches(CUgraphNode begin,CUgraphNode end,
                    const std::vector<CUgraphNode>& from,const std::vector<CUgraphNode>& to) {
    std::vector<CUgraphNode> pending{begin},seen;
    while(!pending.empty()) {
        auto n=pending.back();pending.pop_back();if(n==end)return true;
        if(std::find(seen.begin(),seen.end(),n)!=seen.end())continue;
        seen.push_back(n);
        for(size_t i=0;i<from.size();++i)if(from[i]==n)pending.push_back(to[i]);
    }
    return false;
}
extern "C" int ca_ffn_chunk_create(void* source,void* low_tail,void* high_tail,
    void* finish,int overlap,void** output,unsigned* counts) {
    if(!source||!low_tail||!high_tail||!finish||!output||!counts||(overlap<0||overlap>2))return -1;
    *output=nullptr;auto* p=new(std::nothrow) Holder;if(!p)return -2;
    size_t count=0;TRY(cuGraphGetNodes((CUgraph)source,nullptr,&count));
    std::vector<CUgraphNode> nodes(count),phases,excluded;
    TRY(cuGraphGetNodes((CUgraph)source,nodes.data(),&count));
    unsigned sums=0,adds=0;
    for(auto n:nodes) {
        CUgraphNodeType type;TRY(cuGraphNodeGetType(n,&type));
        REQUIRE(type==CU_GRAPH_NODE_TYPE_KERNEL||type==CU_GRAPH_NODE_TYPE_EMPTY
            ||type==CU_GRAPH_NODE_TYPE_MEMCPY||type==CU_GRAPH_NODE_TYPE_MEMSET,-3);
        if(type!=CU_GRAPH_NODE_TYPE_KERNEL)continue;
        CUDA_KERNEL_NODE_PARAMS params{};TRY(cuGraphKernelNodeGetParams(n,&params));
        const char* name=nullptr;TRY(cuFuncGetName(&name,params.func));
        if(std::strstr(name,"w4a8_phase2W4A8MaterializedPhase2Kernel")) {
            phases.push_back(n);excluded.push_back(n);
        }
        if(std::strstr(name,"W4A16TopKSumKernel")){++sums;excluded.push_back(n);}
        if(std::strstr(name,"add_outputs")){++adds;excluded.push_back(n);}
    }
    REQUIRE(phases.size()==2&&sums==1&&adds==1&&excluded.size()==4,-4);
    size_t edge_count=0;TRY(cuGraphGetEdges((CUgraph)source,nullptr,nullptr,nullptr,&edge_count));
    std::vector<CUgraphNode> from(edge_count),to(edge_count);
    TRY(cuGraphGetEdges((CUgraph)source,from.data(),to.data(),nullptr,&edge_count));
    bool forward=reaches(phases[0],phases[1],from,to),reverse=reaches(phases[1],phases[0],from,to);
    REQUIRE(forward!=reverse,-5);if(reverse)std::swap(phases[0],phases[1]);
    for(int part=0;part<3;++part) {
        TRY(cuGraphClone((CUgraph*)&p->parts[part],(CUgraph)source));
        for(auto n:nodes) {
            bool keep=part==0?std::find(excluded.begin(),excluded.end(),n)==excluded.end():n==phases[part-1];
            if(!keep){CUgraphNode copy;TRY(cuGraphNodeFindInClone(&copy,n,(CUgraph)p->parts[part]));TRY(cuGraphDestroyNode(copy));}
        }
    }
    TRY(cudaGraphCreate(&p->graph,0));
    cudaGraphNode_t trunk,low,high,low_reduce,high_reduce,joined;
    TRY(cudaGraphAddChildGraphNode(&trunk,p->graph,nullptr,0,p->parts[0]));
    TRY(cudaGraphAddChildGraphNode(&low,p->graph,&trunk,1,p->parts[1]));
    if(overlap==2) {
        // Let the four-CTA native transport obtain resources before the next
        // compute chunk. Priority is a scheduler hint, not a dependency change.
        // https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html
        TRY(cudaGraphClone(&p->parts[3],(cudaGraph_t)low_tail));
        size_t n=0;TRY(cudaGraphGetNodes(p->parts[3],nullptr,&n));
        std::vector<cudaGraphNode_t> tail_nodes(n);TRY(cudaGraphGetNodes(p->parts[3],tail_nodes.data(),&n));
        int least=0,greatest=0;TRY(cudaDeviceGetStreamPriorityRange(&least,&greatest));
        REQUIRE(greatest<least,-6);unsigned changed=0;
        for(auto node:tail_nodes) {
            cudaGraphNodeType type;TRY(cudaGraphNodeGetType(node,&type));
            REQUIRE(type==cudaGraphNodeTypeKernel||type==cudaGraphNodeTypeEmpty,-7);
            if(type==cudaGraphNodeTypeKernel) {
                cudaLaunchAttributeValue value{};value.priority=greatest;
                TRY(cudaGraphKernelNodeSetAttribute(node,cudaLaunchAttributePriority,&value));++changed;
            }
        }
        REQUIRE(changed==2,-8);
        TRY(cudaGraphAddChildGraphNode(&low_reduce,p->graph,&low,1,p->parts[3]));
        TRY(cudaGraphAddChildGraphNode(&high,p->graph,&low,1,p->parts[2]));
    } else {
        TRY(cudaGraphAddChildGraphNode(&high,p->graph,&low,1,p->parts[2]));
        auto low_dependency=overlap?low:high;
        TRY(cudaGraphAddChildGraphNode(&low_reduce,p->graph,&low_dependency,1,(cudaGraph_t)low_tail));
    }
    cudaGraphNode_t both[]={high,low_reduce};
    TRY(cudaGraphAddChildGraphNode(&high_reduce,p->graph,both,2,(cudaGraph_t)high_tail));
    TRY(cudaGraphAddChildGraphNode(&joined,p->graph,&high_reduce,1,(cudaGraph_t)finish));
    TRY(cudaGraphInstantiate(&p->exec,p->graph,overlap==2?cudaGraphInstantiateFlagUseNodePriority:0));
    counts[0]=(unsigned)count;counts[1]=2;counts[2]=sums;counts[3]=adds;
    *output=p;return 0;
}
extern "C" int ca_ffn_chunk_launch(void* handle,void* stream) {
    if(!handle)return -1;return (int)cudaGraphLaunch(((Holder*)handle)->exec,(cudaStream_t)stream);
}
extern "C" void ca_ffn_chunk_close(void* handle){delete (Holder*)handle;}
