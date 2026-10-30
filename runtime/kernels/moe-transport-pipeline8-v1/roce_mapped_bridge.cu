// Original experimental CUDA-graph adapter around the separately pinned verbs
// proxy. GPU kernels produce/validate bytes; the CPU handles metadata only.
// This is not a production transport or an inference backend.
#include "nccl_zero_cta_bridge.cu"
#include <cuda.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <thread>
#include <vector>

static void pg_rt(cudaError_t e,const char* where) {
    if(e!=cudaSuccess) {std::fprintf(stderr,"%s: %s\n",where,cudaGetErrorString(e));std::_Exit(41);}
}
static void pg_cu(CUresult e,const char* where) {
    if(e!=CUDA_SUCCESS) {const char* s=nullptr;cuGetErrorName(e,&s);std::fprintf(stderr,"%s: %s\n",where,s?s:"unknown");std::_Exit(42);}
}
__device__ static unsigned char pg_byte(size_t i,int rank,unsigned epoch) {
    unsigned x=unsigned(i)^unsigned(i>>19)^unsigned(rank*100003)^epoch*17239u;
    x^=x>>13;x*=0x85ebca6bu;x^=x>>16;return static_cast<unsigned char>(x);
}
__device__ static unsigned pg_epoch(const unsigned* address) {
    unsigned value;asm volatile("ld.acquire.sys.global.u32 %0,[%1];":"=r"(value):"l"(address):"memory");return value;
}
__global__ static void pg_fill(unsigned char* output,size_t count,int rank,const unsigned* epoch) {
    __shared__ unsigned e;if(threadIdx.x==0)e=pg_epoch(epoch);__syncthreads();
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x) output[i]=pg_byte(i,rank,e);
}
__global__ static void pg_check(const unsigned char* input,size_t count,const unsigned* epoch,unsigned* bad) {
    __shared__ unsigned e;if(threadIdx.x==0)e=pg_epoch(epoch);__syncthreads();
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<4*count;i+=gridDim.x*blockDim.x)
        if(input[i]!=pg_byte(i%count,int(i/count),e)){atomicOr(bad,1u);return;}
}
struct Pg {
    int rank=0,flush=0;size_t count=0;uint64_t layout[7]={};
    bool profile=false;cudaEvent_t events[2][2][5]={};
    void* library=nullptr;void* proxy=nullptr;void* region=nullptr;unsigned* words=nullptr;
    CUdeviceptr device_region=0,device_words=0,device_epoch=0;CUcontext context=nullptr;
    std::atomic<bool> stop{false};std::atomic<unsigned> completed{0};std::thread worker;
    int(*local_blob)(void*,void*,uint64_t)=nullptr;
    int(*connect)(void*,const void*,uint64_t)=nullptr;int(*start)(void*)=nullptr;
    int(*failed)(void*)=nullptr;const char*(*error)(void*)=nullptr;void(*destroy)(void*)=nullptr;
    uint64_t blob_bytes=0;
};
extern "C" void* pg_create(int rank,size_t count,const char* path) {
    if(rank<0||rank>=4||!count||count%4096||count>20971520) return nullptr;
    auto* p=new Pg();p->rank=rank;p->count=count;
    p->profile=std::getenv("CANADA_PROBE_PHASE_TIMING")!=nullptr;
    if(p->profile)for(auto& mode:p->events)for(auto& op:mode)for(auto& event:op)
        pg_rt(cudaEventCreate(&event),"phase event");
    p->library=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!p->library)return nullptr;
    auto symbol=[&](const char* n){void* f=dlsym(p->library,n);if(!f)std::_Exit(43);return f;};
    auto layout=(int(*)(int,uint64_t,uint64_t*))symbol("roce_layout");
    auto create=(void*(*)(int,int,const char*const*,int,int,void*,uint64_t,uint64_t,char*,uint64_t))symbol("roce_create");
    p->local_blob=(int(*)(void*,void*,uint64_t))symbol("roce_local_blob");
    p->connect=(int(*)(void*,const void*,uint64_t))symbol("roce_connect");
    p->start=(int(*)(void*))symbol("roce_start");p->failed=(int(*)(void*))symbol("roce_failed");
    p->error=(const char*(*)(void*))symbol("roce_error");p->destroy=(void(*)(void*))symbol("roce_destroy");
    p->blob_bytes=((uint64_t(*)())symbol("roce_blob_bytes"))();
    if(layout(4,count,p->layout))std::_Exit(44);
    pg_cu(cuCtxGetCurrent(&p->context),"current context");if(!p->context)std::_Exit(45);
    CUdevice device;pg_cu(cuCtxGetDevice(&device),"context device");
    pg_cu(cuDeviceGetAttribute(&p->flush,CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES,device),"remote flush capability");
    pg_cu(cuMemHostAlloc(&p->region,p->layout[4],CU_MEMHOSTALLOC_DEVICEMAP),"registered region");
    pg_cu(cuMemHostGetDevicePointer(&p->device_region,p->region,0),"region pointer");
    pg_cu(cuMemHostAlloc((void**)&p->words,4096,CU_MEMHOSTALLOC_DEVICEMAP),"control words");
    pg_cu(cuMemHostGetDevicePointer(&p->device_words,p->words,0),"control pointer");
    pg_cu(cuMemAlloc(&p->device_epoch,sizeof(unsigned)),"cached epoch allocation");
    std::memset(p->region,0,p->layout[4]);std::memset(p->words,0,4096);p->words[64]=1;
    const char* names[]={"rocep1s0f1","roceP2p1s0f0"};char error[1024]={};
    p->proxy=create(4,rank,names,2,3,p->region,p->layout[4],count,error,sizeof(error));
    if(!p->proxy){std::fprintf(stderr,"proxy create: %s\n",error);std::_Exit(46);}
    return p;
}
extern "C" size_t pg_blob_bytes(Pg* p){return p->blob_bytes;}
extern "C" int pg_blob(Pg* p,void* buffer){return p->local_blob(p->proxy,buffer,p->blob_bytes);}
extern "C" int pg_connect(Pg* p,const void* blobs) {
    int status=p->connect(p->proxy,blobs,4*p->blob_bytes);if(status)return status;
    status=p->start(p->proxy);if(status)return status;
    p->worker=std::thread([p] {
        unsigned seq=0;auto* ctrl=(unsigned*)((char*)p->region+p->layout[3]);
        while(!p->stop.load(std::memory_order_relaxed)) {
            if(__atomic_load_n(p->words,__ATOMIC_ACQUIRE)!=1)continue;
            ++seq;unsigned slot=seq&1u;
            if(seq>1000000 || __atomic_load_n(p->words+64,__ATOMIC_ACQUIRE)!=seq)std::_Exit(47);
            bool dynamic=__atomic_load_n(ctrl+14,__ATOMIC_ACQUIRE)!=0;
            unsigned parts=__atomic_load_n(ctrl+(dynamic?13:6+slot),__ATOMIC_ACQUIRE);
            if(parts!=0&&parts!=2&&parts!=4&&parts!=8)std::_Exit(50);
            if(dynamic)ctrl[6+slot]=parts;
            ctrl[4+slot]=unsigned(p->count);__atomic_store_n(ctrl,seq,__ATOMIC_RELEASE);
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(8);
            for(unsigned part=0;part<(parts?parts:1);++part) {
                for(int rank=0;rank<4;++rank)if(rank!=p->rank)for(int rail=0;rail<2;++rail) {
                    auto* flag=(unsigned*)((char*)p->region+p->layout[1]+((rank*2+slot)*2+rail)*128)+part;
                    while(__atomic_load_n(flag,__ATOMIC_ACQUIRE)!=seq) {
                        if(p->failed(p->proxy)||std::chrono::steady_clock::now()>deadline) {
                            std::fprintf(stderr,"receive timeout rank%d seq%u peer%d rail%d part%u: %s\n",p->rank,seq,rank,rail,part,p->error(p->proxy));std::_Exit(48);
                        }
                    }
                }
                if(parts)__atomic_store_n(p->words+128+32*part,1,__ATOMIC_RELEASE);
            }
            // Do not clear a producer's ready counter until our sender has
            // consumed it too: receiving peers' payloads alone is insufficient.
            while(__atomic_load_n(ctrl+10+slot,__ATOMIC_ACQUIRE)!=seq) {
                if(p->failed(p->proxy)||std::chrono::steady_clock::now()>deadline)std::_Exit(51);
            }
            __atomic_store_n(p->words+96,seq,__ATOMIC_RELEASE);
            __atomic_store_n(p->words+32,1,__ATOMIC_RELEASE);
            while(__atomic_load_n(p->words,__ATOMIC_ACQUIRE)!=0) {
                if(std::chrono::steady_clock::now()>deadline){std::fprintf(stderr,"consumer timeout\n");std::_Exit(49);}
            }
            __atomic_store_n(p->words+64,seq+1,__ATOMIC_RELEASE);
            __atomic_store_n(ctrl+6+slot,0,__ATOMIC_RELEASE);
            __atomic_store_n(ctrl+8+slot,0,__ATOMIC_RELEASE);
            for(unsigned i=12;i<=14;++i)__atomic_store_n(ctrl+i,0,__ATOMIC_RELEASE);
            for(unsigned part=0;part<8;++part)__atomic_store_n(p->words+128+32*part,0,__ATOMIC_RELEASE);
            p->completed.store(seq,std::memory_order_release);
            __atomic_store_n(p->words+32,0,__ATOMIC_RELEASE);
        }
    });return 0;
}
static void pg_memnode(Pg* p,cudaGraph_t graph,cudaGraphNode_t& previous,unsigned value) {
    CUstreamBatchMemOpParams ops[3]={};
    ops[0].operation=CU_STREAM_MEM_OP_WRITE_VALUE_32;
    ops[0].writeValue.address=p->device_words;ops[0].writeValue.value=value;
    ops[0].writeValue.flags=CU_STREAM_WRITE_VALUE_DEFAULT;
    ops[1].operation=CU_STREAM_MEM_OP_WAIT_VALUE_32;
    ops[1].waitValue.address=p->device_words+128;ops[1].waitValue.value=value;
    ops[1].waitValue.flags=CU_STREAM_WAIT_VALUE_EQ|(p->flush?CU_STREAM_WAIT_VALUE_FLUSH:0);
    ops[2].operation=CU_STREAM_MEM_OP_BARRIER;ops[2].memoryBarrier.flags=CU_STREAM_MEMORY_BARRIER_TYPE_SYS;
    CUDA_BATCH_MEM_OP_NODE_PARAMS params={};params.ctx=p->context;params.count=3;params.paramArray=ops;
    CUgraphNode node;
    pg_cu(cuGraphAddBatchMemOpNode(&node,(CUgraph)graph,(CUgraphNode*)&previous,1,&params),"control and system barrier node");
    previous=(cudaGraphNode_t)node;
}
// A publication does not wait: subsequent GPU production can overlap the NIC.
// A range wait imports every peer/rail's completion before a GPU consumer runs.
static void pg_wordnode(Pg* p,cudaGraph_t graph,cudaGraphNode_t& previous,
                        CUdeviceptr address,unsigned value,bool wait=false) {
    CUstreamBatchMemOpParams ops[2]={};
    ops[0].operation=wait?CU_STREAM_MEM_OP_WAIT_VALUE_32:CU_STREAM_MEM_OP_WRITE_VALUE_32;
    if(wait) {
        ops[0].waitValue.address=address;ops[0].waitValue.value=value;
        ops[0].waitValue.flags=CU_STREAM_WAIT_VALUE_EQ|(p->flush?CU_STREAM_WAIT_VALUE_FLUSH:0);
    } else {
        ops[0].writeValue.address=address;ops[0].writeValue.value=value;
        ops[0].writeValue.flags=CU_STREAM_WRITE_VALUE_DEFAULT;
    }
    ops[1].operation=CU_STREAM_MEM_OP_BARRIER;
    ops[1].memoryBarrier.flags=CU_STREAM_MEMORY_BARRIER_TYPE_SYS;
    CUDA_BATCH_MEM_OP_NODE_PARAMS params={};params.ctx=p->context;params.count=2;params.paramArray=ops;
    CUgraphNode node;
    pg_cu(cuGraphAddBatchMemOpNode(&node,(CUgraph)graph,(CUgraphNode*)&previous,1,&params),"pipeline control node");
    previous=(cudaGraphNode_t)node;
}
static void pg_eventnode(Pg* p,cudaGraph_t graph,cudaGraphNode_t& previous,int op,int phase) {
    if(!p->profile)return;
    CUgraphNode node;
    pg_cu(cuGraphAddEventRecordNode(&node,(CUgraph)graph,previous?(CUgraphNode*)&previous:nullptr,
        previous?1:0,(CUevent)p->events[1][op][phase]),"phase event node");
    previous=(cudaGraphNode_t)node;
}
static void pg_copy_node(Pg* p,cudaGraph_t graph,cudaGraphNode_t& previous,CUdeviceptr source,CUdeviceptr target,size_t bytes) {
    CUDA_MEMCPY3D copy={};copy.srcMemoryType=copy.dstMemoryType=CU_MEMORYTYPE_DEVICE;
    copy.srcDevice=source;copy.dstDevice=target;copy.WidthInBytes=bytes;copy.Height=copy.Depth=1;
    CUgraphNode node;pg_cu(cuGraphAddMemcpyNode(&node,(CUgraph)graph,(CUgraphNode*)&previous,1,&copy,p->context),"staged payload copy");
    previous=(cudaGraphNode_t)node;
}
static int pg_capture_impl(Pg* p,unsigned* bad,void** graph_out,void** exec_out,unsigned* counts,bool cached_epoch,
    unsigned char* staged_input=nullptr,unsigned char* staged_output=nullptr) {
    cudaGraph_t graph;pg_rt(cudaGraphCreate(&graph,0),"graph create");cudaGraphNode_t previous=nullptr;
    // Two slots make each replay advance an even number of epochs. All work
    // consuming slot N precedes publication of N+1; receipt of every N+1 flag
    // proves every peer finished N before anyone can overwrite it at N+2.
    for(int slot:{1,0}) {
        int op=1-slot;pg_eventnode(p,graph,previous,op,0);
        if(cached_epoch) {
            // One ordered four-byte control copy replaces an uncached host
            // acquire in every producer/consumer CTA. Payloads stay registered
            // and mapped. The same epoch remains valid through consumption.
            CUDA_MEMCPY3D copy={};copy.srcMemoryType=CU_MEMORYTYPE_HOST;copy.srcHost=p->words+64;
            copy.dstMemoryType=CU_MEMORYTYPE_DEVICE;copy.dstDevice=p->device_epoch;
            copy.WidthInBytes=sizeof(unsigned);copy.Height=copy.Depth=1;
            CUgraphNode node;
            pg_cu(cuGraphAddMemcpyNode(&node,(CUgraph)graph,previous?(CUgraphNode*)&previous:nullptr,
                previous?1:0,&copy,p->context),"cache next epoch node");previous=(cudaGraphNode_t)node;
        }
        auto* output=(unsigned char*)(p->device_region+slot*4*p->count);
        auto* input=output+p->rank*p->count;
        auto* next=(unsigned*)(cached_epoch?p->device_epoch:p->device_words+256);
        auto* current=(unsigned*)(cached_epoch?p->device_epoch:p->device_words+384);
        cudaKernelNodeParams k={};k.func=(void*)pg_fill;k.gridDim=dim3(1024);k.blockDim=dim3(256);
        auto* producer=staged_input?staged_input:input;
        void* fill_args[]={&producer,&p->count,&p->rank,&next};k.kernelParams=fill_args;
        cudaGraphNode_t node;
        pg_rt(cudaGraphAddKernelNode(&node,graph,previous?&previous:nullptr,previous?1:0,&k),"producer node");previous=node;
        if(staged_input)pg_copy_node(p,graph,previous,(CUdeviceptr)staged_input,(CUdeviceptr)input,p->count);
        pg_eventnode(p,graph,previous,op,1);
        pg_memnode(p,graph,previous,1);
        if(staged_output)pg_copy_node(p,graph,previous,(CUdeviceptr)output,(CUdeviceptr)staged_output,4*p->count);
        pg_eventnode(p,graph,previous,op,2);
        k={};k.func=(void*)pg_check;k.gridDim=dim3(1024);k.blockDim=dim3(256);
        auto* consumer=staged_output?staged_output:output;
        void* check_args[]={&consumer,&p->count,&current,&bad};k.kernelParams=check_args;
        pg_rt(cudaGraphAddKernelNode(&node,graph,&previous,1,&k),"consumer validation node");previous=node;
        pg_eventnode(p,graph,previous,op,3);
        pg_memnode(p,graph,previous,0);
        pg_eventnode(p,graph,previous,op,4);
    }
    cudaGraphExec_t exec;pg_rt(cudaGraphInstantiate(&exec,graph,0),"graph instantiate");
    // Runtime node-type introspection cannot represent driver-only batch memory
    // operations on this CUDA build. Query the mixed graph through the driver.
    size_t n=0;pg_cu(cuGraphGetNodes((CUgraph)graph,nullptr,&n),"graph node count");
    std::vector<CUgraphNode> nodes(n);pg_cu(cuGraphGetNodes((CUgraph)graph,nodes.data(),&n),"graph nodes");
    counts[0]=unsigned(n);counts[1]=0;counts[2]=unsigned(p->flush);
    for(auto node:nodes){CUgraphNodeType type;pg_cu(cuGraphNodeGetType(node,&type),"node type");counts[1]+=type==CU_GRAPH_NODE_TYPE_KERNEL;}
    *graph_out=(void*)graph;*exec_out=(void*)exec;return 0;
}
extern "C" int pg_capture(Pg* p,unsigned* bad,void** graph_out,void** exec_out,unsigned* counts) {
    return pg_capture_impl(p,bad,graph_out,exec_out,counts,false);
}
extern "C" int pg_capture_cached_epoch(Pg* p,unsigned* bad,void** graph_out,void** exec_out,unsigned* counts) {
    return pg_capture_impl(p,bad,graph_out,exec_out,counts,true);
}
extern "C" int pg_capture_staged(Pg* p,unsigned* bad,unsigned char* input,unsigned char* output,
    void** graph_out,void** exec_out,unsigned* counts) {
    if(!input||!output)return -1;
    return pg_capture_impl(p,bad,graph_out,exec_out,counts,true,input,output);
}
extern "C" int pg_nccl_capture(ncclComm_t comm,unsigned char* input,unsigned char* output,size_t count,
    int rank,unsigned* epoch,unsigned* bad,cudaStream_t stream,void** graph_out,void** exec_out,Pg* p) {
    pg_rt(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal),"NCCL capture begin");
    for(int i=0;i<2;++i) {
        auto event=[&](int phase){if(p->profile)pg_rt(cudaEventRecordWithFlags(
            p->events[0][i][phase],stream,cudaEventRecordExternal),"NCCL phase event");};
        event(0);
        pg_fill<<<1024,256,0,stream>>>(input,count,rank,epoch);
        event(1);
        ncclResult_t status=ncclAllGather(input,output,count,ncclUint8,comm,stream);if(status!=ncclSuccess)return -int(status);
        event(2);
        pg_check<<<1024,256,0,stream>>>(output,count,epoch,bad);
        event(3);event(4);
    }
    cudaGraph_t graph;pg_rt(cudaStreamEndCapture(stream,&graph),"NCCL capture end");
    cudaGraphExec_t exec;pg_rt(cudaGraphInstantiate(&exec,graph,0),"NCCL instantiate");
    *graph_out=(void*)graph;*exec_out=(void*)exec;return 0;
}
extern "C" int pg_phase_ms(Pg* p,int mode,float* values) {
    if(!p->profile||mode<0||mode>1)return -1;
    for(int phase=0;phase<4;++phase){values[phase]=0;for(int op=0;op<2;++op){float ms=0;
        pg_rt(cudaEventElapsedTime(&ms,p->events[mode][op][phase],p->events[mode][op][phase+1]),"phase elapsed");
        values[phase]+=ms*.5f;}}
    return 0;
}
extern "C" unsigned pg_epochs(Pg* p){return p->completed.load(std::memory_order_acquire);}
extern "C" void pg_destroy(Pg* p) {
    p->stop.store(true);if(p->worker.joinable())p->worker.join();p->destroy(p->proxy);
    if(p->profile)for(auto& mode:p->events)for(auto& op:mode)for(auto event:op)cudaEventDestroy(event);
    cuMemFree(p->device_epoch);cuMemFreeHost(p->words);cuMemFreeHost(p->region);dlclose(p->library);delete p;
}
