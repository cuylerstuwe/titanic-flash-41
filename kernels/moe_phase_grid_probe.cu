// Isolated launch-geometry probe. Leave the cooperative routing front-end
// unchanged; only the two independent, grid-stride materialized kernels vary.
#include <cuda.h>
#include <cstring>
#include <vector>

extern "C" int ca_moe_regrid(void* graph, unsigned phase1, unsigned phase2,
                              unsigned* counts) {
    if (!graph || !counts || (phase1!=48 && phase1!=96 && phase1!=144 && phase1!=192)
        || (phase2!=48 && phase2!=96 && phase2!=144 && phase2!=192)) return -1;
    counts[0]=counts[1]=0;
    size_t size=0;
    CUresult rc=cuGraphGetNodes((CUgraph)graph,nullptr,&size);
    if(rc!=CUDA_SUCCESS)return (int)rc;
    std::vector<CUgraphNode> nodes(size);
    rc=cuGraphGetNodes((CUgraph)graph,nodes.data(),&size);
    if(rc!=CUDA_SUCCESS)return (int)rc;
    for(auto node:nodes) {
        CUgraphNodeType type;
        rc=cuGraphNodeGetType(node,&type);if(rc!=CUDA_SUCCESS)return (int)rc;
        if(type!=CU_GRAPH_NODE_TYPE_KERNEL)continue;
        CUDA_KERNEL_NODE_PARAMS params{};
        rc=cuGraphKernelNodeGetParams(node,&params);if(rc!=CUDA_SUCCESS)return (int)rc;
        const char* name=nullptr;
        rc=cuFuncGetName(&name,params.func);if(rc!=CUDA_SUCCESS)return (int)rc;
        int phase=-1;
        if(std::strstr(name,"w4a8_phase1W4A8MaterializedPhase1Kernel"))phase=0;
        if(std::strstr(name,"w4a8_phase2W4A8MaterializedPhase2Kernel"))phase=1;
        if(phase<0)continue;
        if(params.gridDimX!=1 || params.gridDimY!=1 || params.gridDimZ!=96
           || params.blockDimX!=128 || params.blockDimY!=1 || params.blockDimZ!=1)return -2;
        params.gridDimZ=phase==0?phase1:phase2;
        rc=cuGraphKernelNodeSetParams(node,&params);if(rc!=CUDA_SUCCESS)return (int)rc;
        ++counts[phase];
    }
    return 0;
}
