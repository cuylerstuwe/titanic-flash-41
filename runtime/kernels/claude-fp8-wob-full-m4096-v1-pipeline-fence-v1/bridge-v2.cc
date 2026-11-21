#include "kernel_000.h"
#include "kernel_001.h"
#include <memory>
#include <cstring>
static float word_float(uint64_t word){uint32_t a=(uint32_t)word;float b;memcpy(&b,&a,4);return b;}
static double word_double(uint64_t word){double b;memcpy(&b,&word,8);return b;}
struct Context{
canada_kernel_000_Kernel_Module_t m0{};
canada_kernel_001_Kernel_Module_t m1{};
~Context(){if(m0.module)cudaLibraryUnload(m0.module);if(m1.module)cudaLibraryUnload(m1.module);}};
extern "C" int32_t ca_primitives_abi(){return 2;}
extern "C" void* ca_primitives_open(int32_t* status){*status=0;auto ctx=std::make_unique<Context>();
{cudaLibrary_t* p=&ctx->m0.module;cudaError_t ret=cudaSuccess;
        void* init[]={&p,&ret};_mlir_canada_kernel_000_cuda_init(init);if(ret){*status=ret;return nullptr;}
        int32_t device=0;void* load[]={&p,&device,&ret};_mlir_canada_kernel_000_cuda_load_to_device(load);
        if(ret){*status=ret;return nullptr;}}
{cudaLibrary_t* p=&ctx->m1.module;cudaError_t ret=cudaSuccess;
        void* init[]={&p,&ret};_mlir_canada_kernel_001_cuda_init(init);if(ret){*status=ret;return nullptr;}
        int32_t device=0;void* load[]={&p,&device,&ret};_mlir_canada_kernel_001_cuda_load_to_device(load);
        if(ret){*status=ret;return nullptr;}}return ctx.release();}
extern "C" void ca_primitives_close(void* ctx){delete (Context*)ctx;}
extern "C" int32_t ca_primitives_launch(void*,int32_t,const uint64_t*,size_t,const int32_t*,size_t,void*){return -5;}
extern "C" int32_t ca_primitives_launch2(void* context,int32_t index,const uint64_t* values,size_t count,void* stream){
 if(!context)return -1;auto ctx=(Context*)context;switch(index){
case 0:{if(count!=7)return -2;return cute_dsl_canada_kernel_000_wrapper(&ctx->m0,(void*)values[0],(void*)values[1],(void*)values[2],(void*)values[3],(int32_t)values[4],(int32_t)values[5],(int32_t)values[6],(cudaStream_t)stream);}
case 1:{if(count!=10)return -2;return cute_dsl_canada_kernel_001_wrapper(&ctx->m1,(void*)values[0],(void*)values[1],(void*)values[2],(void*)values[3],(void*)values[4],(void*)values[5],(void*)values[6],(void*)values[7],(void*)values[8],(int32_t)values[9],(cudaStream_t)stream);}default:return -3;}}
