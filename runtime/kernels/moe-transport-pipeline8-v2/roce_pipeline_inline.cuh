// Original stream adapter for the experimental pipeline. No model state or
// scheduling policy lives here: Rust supplies operands and owns ordered use.
struct Pp {
    Pg* transport;
    void* library;
    int(*pack)(const void*,void*,const unsigned*,int,int,int,int,cudaStream_t);
    int(*decode)(const void*,void*,void*,void*,void*,const unsigned*,int,int,int,cudaStream_t);
};
static void pp_mem(Pg* p,cudaStream_t stream,CUstreamBatchMemOpParams* ops,unsigned count) {
    CUstreamCaptureStatus status;CUgraph graph;const CUgraphNode* deps=nullptr;size_t n=0;
    pg_cu(cuStreamGetCaptureInfo((CUstream)stream,&status,nullptr,&graph,&deps,nullptr,&n),"pipeline capture info");
    if(status==CU_STREAM_CAPTURE_STATUS_NONE) {
        pg_cu(cuStreamBatchMemOp((CUstream)stream,count,ops,0),"pipeline eager memops");
    } else {
        if(status!=CU_STREAM_CAPTURE_STATUS_ACTIVE)std::_Exit(54);
        CUDA_BATCH_MEM_OP_NODE_PARAMS params={};params.ctx=p->context;params.count=count;params.paramArray=ops;
        CUgraphNode node;
        pg_cu(cuGraphAddBatchMemOpNode(&node,graph,deps,n,&params),"pipeline captured memops");
        pg_cu(cuStreamUpdateCaptureDependencies((CUstream)stream,&node,nullptr,1,CU_STREAM_SET_CAPTURE_DEPENDENCIES),
            "pipeline capture dependencies");
    }
}
static void pp_word(Pg* p,cudaStream_t stream,CUdeviceptr address,unsigned value,bool wait=false) {
    CUstreamBatchMemOpParams ops[2]={};
    ops[0].operation=wait?CU_STREAM_MEM_OP_WAIT_VALUE_32:CU_STREAM_MEM_OP_WRITE_VALUE_32;
    if(wait) {
        ops[0].waitValue.address=address;ops[0].waitValue.value=value;
        ops[0].waitValue.flags=CU_STREAM_WAIT_VALUE_EQ|(p->flush?CU_STREAM_WAIT_VALUE_FLUSH:0);
    } else {
        ops[0].writeValue.address=address;ops[0].writeValue.value=value;
        ops[0].writeValue.flags=CU_STREAM_WRITE_VALUE_DEFAULT;
    }
    ops[1].operation=CU_STREAM_MEM_OP_BARRIER;ops[1].memoryBarrier.flags=CU_STREAM_MEMORY_BARRIER_TYPE_SYS;
    pp_mem(p,stream,ops,2);
}
extern "C" void* pp_create(Pg* transport,const char* path) {
    if(!transport||transport->count!=2048*5920)return nullptr;
    auto* p=new Pp{};p->transport=transport;p->library=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if(p->library) {
        p->pack=(decltype(p->pack))dlsym(p->library,"ca_pack_registered_slot");
        p->decode=(decltype(p->decode))dlsym(p->library,"ca_decode_registered_slot");
    }
    if(!p->pack||!p->decode) {if(p->library)dlclose(p->library);delete p;return nullptr;}
    return p;
}
extern "C" int pp_begin(Pp* p,const void* input,cudaStream_t stream) {
    if(!p||!input||!stream)return -1;
    auto* t=p->transport;
    // The final consumer acknowledgment always precedes this next epoch read.
    // Therefore every replay, including an odd number of invocations, chooses
    // the right slot independently of host capture or warmup call counts.
    pg_rt(cudaMemcpyAsync((void*)t->device_epoch,t->words+64,4,cudaMemcpyHostToDevice,stream),"inline cached epoch");
    const auto ctrl=t->device_region+t->layout[3];
    pp_word(t,stream,ctrl+14*4,1);pp_word(t,stream,ctrl+13*4,8);
    for(int part=0;part<8;++part) {
        int status=p->pack(input,(void*)t->device_region,(unsigned*)t->device_epoch,2048,8,part,t->rank,stream);
        if(status)return status;
        pp_word(t,stream,ctrl+12*4,part+1);
        if(part==0)pp_word(t,stream,t->device_words,1);
    }
    return 0;
}
extern "C" int pp_finish(Pp* p,void* packed,void* values,void* row_scales,
    void* mma_scales,cudaStream_t stream) {
    if(!p||!packed||!values||!row_scales||!mma_scales||!stream)return -1;
    auto* t=p->transport;
    for(int part=0;part<8;++part) {
        pp_word(t,stream,t->device_words+(128+32*part)*4,1,true);
        int status=p->decode((void*)t->device_region,values,row_scales,mma_scales,packed,
            (unsigned*)t->device_epoch,2048,8,part,stream);
        if(status)return status;
    }
    pp_word(t,stream,t->device_words+128,1,true);
    pp_word(t,stream,t->device_words,0);
    pp_word(t,stream,t->device_words+128,0,true);
    return 0;
}
extern "C" int pp_enqueue(Pp* p,const void* input,void* packed,void* values,
    void* row_scales,void* mma_scales,cudaStream_t stream) {
    int status=pp_begin(p,input,stream);if(status)return status;
    return pp_finish(p,packed,values,row_scales,mma_scales,stream);
}
extern "C" void pp_destroy(Pp* p) {if(p){dlclose(p->library);delete p;}}
extern "C" int pg_health(Pg* p) {return p?p->failed(p->proxy):-1;}
