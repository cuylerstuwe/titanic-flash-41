// Isolated resource-allocation experiment; no kernel arithmetic or geometry changes.
// CUDA driver attributes are preferences, not measured cache sizes.
#include <cuda.h>
#include <cstring>
#include <vector>
#include <set>

extern "C" int ca_moe_carveout(void* graph, int preference, int* records) {
    if (!graph || !records || (preference != -2 && preference != -1 && preference != 0 && preference != 50 && preference != 100)) return -1;
    size_t count=0;
    CUresult rc=cuGraphGetNodes((CUgraph)graph,nullptr,&count);
    if(rc!=CUDA_SUCCESS)return (int)rc;
    std::vector<CUgraphNode> nodes(count);
    rc=cuGraphGetNodes((CUgraph)graph,nodes.data(),&count);
    if(rc!=CUDA_SUCCESS)return (int)rc;
    std::set<CUfunction> seen;
    int phases[2]={0,0};
    for(auto node:nodes) {
        CUgraphNodeType type;
        rc=cuGraphNodeGetType(node,&type);if(rc!=CUDA_SUCCESS)return (int)rc;
        if(type!=CU_GRAPH_NODE_TYPE_KERNEL)continue;
        CUDA_KERNEL_NODE_PARAMS p{};
        rc=cuGraphKernelNodeGetParams(node,&p);if(rc!=CUDA_SUCCESS)return (int)rc;
        const char* name=nullptr;
        rc=cuFuncGetName(&name,p.func);if(rc!=CUDA_SUCCESS)return (int)rc;
        int phase=-1;
        if(std::strstr(name,"w4a8_phase1W4A8MaterializedPhase1Kernel"))phase=0;
        if(std::strstr(name,"w4a8_phase2W4A8MaterializedPhase2Kernel"))phase=1;
        if(phase<0 || !seen.insert(p.func).second)continue;
        if(p.gridDimX!=1 || p.gridDimY!=1 || p.gridDimZ!=96 || p.blockDimX!=128 || p.blockDimY!=1 || p.blockDimZ!=1)return -2;
        int* row=records+phase*7;
        rc=cuFuncGetAttribute(&row[0],CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,p.func);if(rc!=CUDA_SUCCESS)return (int)rc;
        if(preference!=-1) {
            rc=cuFuncSetAttribute(p.func,CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,preference == -2 ? -1 : preference);if(rc!=CUDA_SUCCESS)return (int)rc;
        }
        rc=cuFuncGetAttribute(&row[1],CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,p.func);if(rc!=CUDA_SUCCESS)return (int)rc;
        rc=cuFuncGetAttribute(&row[2],CU_FUNC_ATTRIBUTE_NUM_REGS,p.func);if(rc!=CUDA_SUCCESS)return (int)rc;
        rc=cuFuncGetAttribute(&row[3],CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,p.func);if(rc!=CUDA_SUCCESS)return (int)rc;
        rc=cuFuncGetAttribute(&row[4],CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,p.func);if(rc!=CUDA_SUCCESS)return (int)rc;
        row[5]=p.sharedMemBytes;
        rc=cuOccupancyMaxActiveBlocksPerMultiprocessor(&row[6],p.func,128,p.sharedMemBytes);if(rc!=CUDA_SUCCESS)return (int)rc;
        ++phases[phase];
    }
    return phases[0]==1 && phases[1]==1 ? 0 : -3;
}
