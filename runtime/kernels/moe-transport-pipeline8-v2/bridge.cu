// Original component diagnostic. Use the already-qualified codec binary without
// changing its arithmetic. Compare direct registered-memory IO with CUDA-memory
// IO around NCCL. Reference production and all comparisons execute on the GPU.
#include "roce_mapped_bridge.cu"
#include "roce_pipeline_inline.cuh"

struct Pm {
    Pg* transport;
    void* library;
    void* registered_library=nullptr;
    int rows;
    int(*pack)(const void*,void*,int,int,cudaStream_t);
    int(*decode)(const void*,void*,void*,void*,int,int,cudaStream_t);
    int(*pack_registered)(const void*,void*,int,int,cudaStream_t)=nullptr;
    int(*decode_registered)(const void*,void*,void*,void*,int,int,cudaStream_t)=nullptr;
    int(*decode_part)(const void*,void*,void*,void*,int,int,int,void*,cudaStream_t)=nullptr;
    bool normal_timing=false;
    bool metadata=false;
    Pp* pipeline=nullptr;
    unsigned* epoch;
    unsigned short *input,*reference_input;
    unsigned char *send,*gather,*values,*row_scales,*mma_scales;
    unsigned char *reference_packed,*reference_values,*reference_row,*reference_mma;
    unsigned *meta_input[2]={},*meta_output[2]={};
};
extern "C" void* pm_registered_slot(Pg* p,int slot) {
    if(!p||slot<0||slot>1)return nullptr;
    return (void*)(p->device_region+size_t(slot)*4*p->count);
}
__global__ void pm_advance(unsigned* epoch) { ++*epoch; }
__global__ void pm_fill(unsigned short* input,size_t local_values,int rank,
                        bool all_ranks,const unsigned* epoch,bool broad) {
    unsigned e=*epoch;
    size_t n=local_values*(all_ranks?4:1);
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
        size_t j=i%local_values;unsigned r=all_ranks?unsigned(i/local_values):unsigned(rank);
        unsigned group=unsigned(j/32)*65537u+r*100003u+e*17239u;
        group^=group>>13;
        unsigned x=unsigned(j)*0x85ebca6bu+group;x^=x>>16;
        // Finite BF16 fixtures include zeros and groups below the shared-scale
        // floor. Only the pinned codec performs quantization/model arithmetic.
        input[i]=group%23==0?0:static_cast<unsigned short>(
            ((x&1u)<<15)|(((broad?96u+group%39u:124u+group%8u))<<7)|((x>>1)&127u));
    }
}
__global__ void pm_equal(const unsigned char* actual,const unsigned char* expected,
                         size_t bytes,unsigned* bad,unsigned bit) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<bytes;i+=gridDim.x*blockDim.x)
        if(actual[i]!=expected[i]) {atomicOr(bad,bit);return;}
}
__device__ unsigned pm_metadata_word(size_t i,int rank,unsigned epoch,int tag) {
    return unsigned(i)*1664525u ^ (epoch+unsigned(tag))*1013904223u ^ unsigned(rank)*0x85ebca6bu;
}
__global__ void pm_metadata_values(unsigned* values,size_t count,int rank,const unsigned* epoch,int tag) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
        values[i]=pm_metadata_word(i,rank,*epoch,tag);
}
__global__ void pm_metadata_check(const unsigned* values,size_t local_count,const unsigned* epoch,
    int tag,unsigned* bad) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<4*local_count;i+=gridDim.x*blockDim.x)
        if(values[i]!=pm_metadata_word(i%local_count,int(i/local_count),*epoch,tag)){atomicOr(bad,32u<<tag);return;}
}
static void pm_metadata_produce(Pm* p,unsigned* epoch,cudaStream_t stream) {
    if(p->metadata)for(int tag=0;tag<2;++tag)
        pm_metadata_values<<<32,256,0,stream>>>(p->meta_input[tag],size_t(p->rows)*6,p->transport->rank,epoch,tag);
}
static int pm_metadata_gather(Pm* p,ncclComm_t comm,cudaStream_t stream) {
    if(p->metadata)for(int tag=0;tag<2;++tag) {
        auto status=ncclAllGather(p->meta_input[tag],p->meta_output[tag],size_t(p->rows)*6*4,ncclUint8,comm,stream);
        if(status!=ncclSuccess)return -int(status);
    }
    return 0;
}
static void pm_status(int status,const char* where) {
    if(status) {std::fprintf(stderr,"%s: codec status %d\n",where,status);std::_Exit(52);}
}
extern "C" void* pm_create(Pg* transport,const char* codec_path) {
    if(!transport||transport->count%5920)return nullptr;
    auto* p=new Pm{};p->transport=transport;p->rows=int(transport->count/5920);
    p->normal_timing=std::getenv("CANADA_PROBE_MOE_NORMAL_TIMING")!=nullptr;
    p->metadata=std::getenv("CANADA_PROBE_MOE_PIPELINE_METADATA")!=nullptr;
    if(p->rows%128) {delete p;return nullptr;}
    p->library=dlopen(codec_path,RTLD_NOW|RTLD_LOCAL);if(!p->library)return nullptr;
    p->pack=(decltype(p->pack))dlsym(p->library,"ca_pack_bf16");
    p->decode=(decltype(p->decode))dlsym(p->library,"ca_decode");
    if(!p->pack||!p->decode)std::_Exit(53);
    auto alloc=[](auto** address,size_t bytes) {pg_rt(cudaMalloc((void**)address,bytes),"codec probe allocation");};
    alloc(&p->epoch,4);pg_rt(cudaMemset(p->epoch,0,4),"epoch init");
    alloc(&p->input,size_t(p->rows)*5120*2);alloc(&p->reference_input,size_t(p->rows)*4*5120*2);
    alloc(&p->send,transport->count);alloc(&p->gather,4*transport->count);
    alloc(&p->reference_packed,4*transport->count);
    alloc(&p->values,size_t(p->rows)*4*5120);alloc(&p->reference_values,size_t(p->rows)*4*5120);
    for(auto** address:{&p->row_scales,&p->mma_scales,&p->reference_row,&p->reference_mma})
        alloc(address,size_t(p->rows)*4*160);
    if(p->metadata)for(int tag=0;tag<2;++tag) {
        alloc(&p->meta_input[tag],size_t(p->rows)*6*4);alloc(&p->meta_output[tag],size_t(p->rows)*4*6*4);
    }
    return p;
}
extern "C" int pm_enable_registered_codec(Pm* p,const char* path) {
    if(p->registered_library)return -1;
    p->registered_library=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!p->registered_library)return -2;
    p->pack_registered=(decltype(p->pack_registered))dlsym(p->registered_library,"ca_pack_registered");
    p->decode_registered=(decltype(p->decode_registered))dlsym(p->registered_library,"ca_decode_registered");
    p->decode_part=(decltype(p->decode_part))dlsym(p->registered_library,"ca_decode_registered_part");
    p->pipeline=(Pp*)pp_create(p->transport,path);
    return p->pack_registered&&p->decode_registered?0:-3;
}
static void pm_produce(Pm* p,unsigned char* packed,unsigned* epoch,cudaStream_t stream,bool registered=false,bool broad=true) {
    pm_fill<<<192,256,0,stream>>>(p->input,size_t(p->rows)*5120,p->transport->rank,false,epoch,broad);
    pm_metadata_produce(p,epoch,stream);
    auto pack=registered?p->pack_registered:p->pack;
    pm_status(pack(p->input,packed,p->rows,5120,stream),"pack into transport input");
}
static void pm_consume(Pm* p,unsigned char* packed,cudaStream_t stream,bool registered=false) {
    auto decode=registered?p->decode_registered:p->decode;
    pm_status(decode(packed,p->values,p->row_scales,p->mma_scales,p->rows*4,5120,stream),
        "decode directly from transport output");
}
static void pm_validate(Pm* p,unsigned char* packed,unsigned* epoch,unsigned* bad,cudaStream_t stream,bool broad=true) {
    pm_fill<<<192,256,0,stream>>>(p->reference_input,size_t(p->rows)*5120,0,true,epoch,broad);
    pm_status(p->pack(p->reference_input,p->reference_packed,p->rows*4,5120,stream),"reference pack");
    pm_status(p->decode(p->reference_packed,p->reference_values,p->reference_row,p->reference_mma,
        p->rows*4,5120,stream),"reference decode");
    pm_equal<<<192,256,0,stream>>>(packed,p->reference_packed,p->transport->count*4,bad,1);
    pm_equal<<<192,256,0,stream>>>(p->values,p->reference_values,size_t(p->rows)*4*5120,bad,2);
    pm_equal<<<192,256,0,stream>>>(p->row_scales,p->reference_row,size_t(p->rows)*4*160,bad,4);
    pm_equal<<<192,256,0,stream>>>(p->mma_scales,p->reference_mma,size_t(p->rows)*4*160,bad,8);
    if(p->metadata)for(int tag=0;tag<2;++tag)
        pm_metadata_check<<<32,256,0,stream>>>(p->meta_output[tag],size_t(p->rows)*6,epoch,tag,bad);
    pg_rt(cudaGetLastError(),"GPU reference comparison");
}
template<class F> static void pm_child(cudaGraph_t graph,cudaGraphNode_t& previous,
                                      cudaStream_t stream,F body) {
    pg_rt(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"codec child capture");
    body();cudaGraph_t child;
    pg_rt(cudaStreamEndCapture(stream,&child),"codec child capture end");
    cudaGraphNode_t node;
    pg_rt(cudaGraphAddChildGraphNode(&node,graph,previous?&previous:nullptr,previous?1:0,child),"codec child node");
    previous=node;pg_rt(cudaGraphDestroy(child),"captured child cleanup");
}
extern "C" int pm_capture(Pm* p,ncclComm_t comm,int mode,int validate,unsigned* bad,
                           cudaStream_t stream,void** graph_out,void** exec_out) {
    auto* t=p->transport;cudaGraph_t graph;
    if(mode<0||mode>12||((mode==2||mode>=4)&&!p->pack_registered))return -1;
    if(mode>=11&&!p->pipeline)return -1;
    if(p->metadata&&mode!=5&&mode!=11&&mode!=12)return -1;
    int parts=mode==6?2:(mode==7||mode==9)?4:(mode==8||mode==10)?8:0;
    bool copy_packed=mode>=9;
    if(parts&&(!p->decode_part||p->rows%(128*parts)))return -1;
    bool mapped=mode!=0&&mode!=5;
    bool broad=validate||!p->normal_timing;
    if(mapped) {
        pg_rt(cudaGraphCreate(&graph,0),"codec mapped graph");cudaGraphNode_t previous=nullptr;
        for(int slot:{1,0}) {
            int op=1-slot;pg_eventnode(t,graph,previous,op,0);
            CUDA_MEMCPY3D copy={};copy.srcMemoryType=CU_MEMORYTYPE_HOST;copy.srcHost=t->words+64;
            copy.dstMemoryType=CU_MEMORYTYPE_DEVICE;copy.dstDevice=t->device_epoch;
            copy.WidthInBytes=4;copy.Height=copy.Depth=1;CUgraphNode node;
            pg_cu(cuGraphAddMemcpyNode(&node,(CUgraph)graph,previous?(CUgraphNode*)&previous:nullptr,
                previous?1:0,&copy,t->context),"codec cached epoch");previous=(cudaGraphNode_t)node;
            auto* gathered=(unsigned char*)(t->device_region+slot*4*t->count);
            auto* epoch=(unsigned*)t->device_epoch;
            if(mode>=11) {
                // Exercise the exact stream-capture splice intended for Rust.
                pm_child(graph,previous,stream,[&]{
                    pm_fill<<<192,256,0,stream>>>(p->input,size_t(p->rows)*5120,t->rank,false,epoch,broad);
                    pm_metadata_produce(p,epoch,stream);
                    if(mode==12) {
                        pm_status(pp_begin(p->pipeline,p->input,stream),"split pipeline begin");
                        pm_status(pm_metadata_gather(p,comm,stream),"overlapped routing metadata");
                        pm_status(pp_finish(p->pipeline,p->gather,p->values,p->row_scales,p->mma_scales,stream),"split pipeline finish");
                    } else {
                        pm_status(pp_enqueue(p->pipeline,p->input,p->gather,p->values,p->row_scales,p->mma_scales,stream),"inline pipeline");
                        pm_status(pm_metadata_gather(p,comm,stream),"sequential routing metadata");
                    }
                    if(validate)pm_validate(p,p->gather,epoch,bad,stream,broad);
                });
                for(int phase=1;phase<5;++phase)pg_eventnode(t,graph,previous,op,phase);
                continue;
            }
            if(parts) {
                const auto ctrl=t->device_region+t->layout[3];
                pg_wordnode(t,graph,previous,ctrl+(6+slot)*4,parts);
                pm_child(graph,previous,stream,[&]{
                    pm_fill<<<192,256,0,stream>>>(p->input,size_t(p->rows)*5120,t->rank,false,epoch,broad);
                });
                for(int part=0;part<parts;++part) {
                    const int rows=p->rows/parts;
                    pm_child(graph,previous,stream,[&]{
                        pm_status(p->pack_registered(p->input+size_t(part)*rows*5120,
                            gathered+t->rank*t->count+size_t(part)*rows*5920,rows,5120,stream),"pipeline pack");
                    });
                    pg_wordnode(t,graph,previous,ctrl+(8+slot)*4,part+1);
                    if(part==0)pg_wordnode(t,graph,previous,t->device_words,1);
                }
            } else pm_child(graph,previous,stream,[&]{
                auto* target=gathered+t->rank*t->count;
                pm_produce(p,mode==3?p->send:target,epoch,stream,mode==2||mode==4,broad);
                if(mode==3)pg_rt(cudaMemcpyAsync(target,p->send,t->count,cudaMemcpyDeviceToDevice,stream),
                    "send-only staging copy");
            });
            pg_eventnode(t,graph,previous,op,1);
            if(!parts)pg_memnode(t,graph,previous,1);
            else pg_wordnode(t,graph,previous,t->device_words+128*4,1,true);
            pg_eventnode(t,graph,previous,op,2);
            if(parts) {
                for(int part=0;part<parts;++part) {
                    if(part)pg_wordnode(t,graph,previous,t->device_words+(128+32*part)*4,1,true);
                    pm_child(graph,previous,stream,[&]{
                        pm_status(p->decode_part(gathered,p->values,p->row_scales,p->mma_scales,
                            p->rows,parts,part,copy_packed?p->gather:nullptr,stream),"pipeline decode");
                    });
                }
                // Acknowledge only after the worker has published global
                // completion, including consumption of our ready counters.
                pg_wordnode(t,graph,previous,t->device_words+128,1,true);
            } else pm_child(graph,previous,stream,[&]{pm_consume(p,gathered,stream,mode==4);});
            pg_eventnode(t,graph,previous,op,3);
            if(validate)pm_child(graph,previous,stream,[&]{pm_validate(p,gathered,epoch,bad,stream,broad);});
            if(validate&&copy_packed)pm_child(graph,previous,stream,[&]{
                pm_equal<<<192,256,0,stream>>>(p->gather,p->reference_packed,t->count*4,bad,16);
            });
            pg_memnode(t,graph,previous,0);pg_eventnode(t,graph,previous,op,4);
        }
    } else {
        pg_rt(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal),"codec NCCL graph");
        for(int op=0;op<2;++op) {
            auto event=[&](int phase){if(t->profile)pg_rt(cudaEventRecordWithFlags(
                t->events[0][op][phase],stream,cudaEventRecordExternal),"codec NCCL event");};
            event(0);pm_advance<<<1,1,0,stream>>>(p->epoch);pm_produce(p,p->send,p->epoch,stream,mode==5,broad);event(1);
            auto status=ncclAllGather(p->send,p->gather,t->count,ncclUint8,comm,stream);
            if(status!=ncclSuccess)return -int(status);
            event(2);pm_consume(p,p->gather,stream,mode==5);event(3);
            pm_status(pm_metadata_gather(p,comm,stream),"NCCL routing metadata");
            if(validate)pm_validate(p,p->gather,p->epoch,bad,stream,broad);
            event(4);
        }
        pg_rt(cudaStreamEndCapture(stream,&graph),"codec NCCL capture end");
    }
    cudaGraphExec_t executable;pg_rt(cudaGraphInstantiate(&executable,graph,0),"codec graph instantiate");
    *graph_out=graph;*exec_out=executable;return 0;
}
extern "C" int pm_check(Pm* p,int mode,unsigned* bad,cudaStream_t stream) {
    // Every replay ends in slot zero; no next replay may begin before this check.
    bool mapped=mode!=0&&mode!=5;
    pm_validate(p,mapped&&mode<11?(unsigned char*)p->transport->device_region:p->gather,
        mapped?(unsigned*)p->transport->device_epoch:p->epoch,bad,stream,!p->normal_timing);
    if(mode>=9)pm_equal<<<192,256,0,stream>>>(p->gather,p->reference_packed,p->transport->count*4,bad,16);
    return int(cudaGetLastError());
}
extern "C" int pm_inline_once(Pm* p,unsigned* bad,cudaStream_t stream) {
    if(!p||!p->pipeline||p->metadata)return -1;
    auto* t=p->transport;
    pg_rt(cudaMemcpyAsync((void*)t->device_epoch,t->words+64,4,cudaMemcpyHostToDevice,stream),"inline fixture epoch");
    pm_fill<<<192,256,0,stream>>>(p->input,size_t(p->rows)*5120,t->rank,false,(unsigned*)t->device_epoch,true);
    int status=pp_enqueue(p->pipeline,p->input,p->gather,p->values,p->row_scales,p->mma_scales,stream);
    if(status)return status;
    pm_validate(p,p->gather,(unsigned*)t->device_epoch,bad,stream,true);
    return int(cudaGetLastError());
}
extern "C" int pm_warm(Pm* p,ncclComm_t comm,unsigned* bad,cudaStream_t stream) {
    pm_advance<<<1,1,0,stream>>>(p->epoch);pm_produce(p,p->send,p->epoch,stream);
    auto status=ncclAllGather(p->send,p->gather,p->transport->count,ncclUint8,comm,stream);
    if(status!=ncclSuccess)return -int(status);
    pm_consume(p,p->gather,stream);pm_status(pm_metadata_gather(p,comm,stream),"warm routing metadata");
    pm_validate(p,p->gather,p->epoch,bad,stream);return 0;
}
extern "C" void pm_destroy(Pm* p) {
    pp_destroy(p->pipeline);
    if(p->metadata)for(int tag=0;tag<2;++tag){cudaFree(p->meta_input[tag]);cudaFree(p->meta_output[tag]);}
    for(void* address:{(void*)p->epoch,(void*)p->input,(void*)p->reference_input,(void*)p->send,
        (void*)p->gather,(void*)p->reference_packed,(void*)p->values,(void*)p->reference_values,
        (void*)p->row_scales,(void*)p->mma_scales,(void*)p->reference_row,(void*)p->reference_mma})
        cudaFree(address);
    if(p->registered_library)dlclose(p->registered_library);
    dlclose(p->library);delete p;
}
