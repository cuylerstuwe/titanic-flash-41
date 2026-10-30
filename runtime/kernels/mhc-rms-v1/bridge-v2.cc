#include "kernel_000.h"
#include <memory>
#include <cstring>
static float word_float(uint64_t word){uint32_t a=(uint32_t)word;float b;memcpy(&b,&a,4);return b;}
static double word_double(uint64_t word){double b;memcpy(&b,&word,8);return b;}
struct Context{
canada_kernel_000_Kernel_Module_t m0{};
~Context(){if(m0.module)cudaLibraryUnload(m0.module);}};
extern "C" int32_t ca_primitives_abi(){return 2;}
extern "C" void* ca_primitives_open(int32_t* status){*status=0;auto ctx=std::make_unique<Context>();
{cudaLibrary_t* p=&ctx->m0.module;cudaError_t ret=cudaSuccess;
        void* init[]={&p,&ret};_mlir_canada_kernel_000_cuda_init(init);if(ret){*status=ret;return nullptr;}
        int32_t device=0;void* load[]={&p,&device,&ret};_mlir_canada_kernel_000_cuda_load_to_device(load);
        if(ret){*status=ret;return nullptr;}}return ctx.release();}
extern "C" void ca_primitives_close(void* ctx){delete (Context*)ctx;}
extern "C" int32_t ca_primitives_launch(void*,int32_t,const uint64_t*,size_t,const int32_t*,size_t,void*){return -5;}
extern "C" int32_t ca_primitives_launch2(void* context,int32_t index,const uint64_t* values,size_t count,void* stream){
 if(!context)return -1;auto ctx=(Context*)context;switch(index){
case 0:{if(count!=5)return -2;static_assert(sizeof(canada_kernel_000_Tensor_x_t)==8);canada_kernel_000_Tensor_x_t t0{(void*)values[0]};static_assert(sizeof(canada_kernel_000_Tensor_w_t)==8);canada_kernel_000_Tensor_w_t t1{(void*)values[1]};static_assert(sizeof(canada_kernel_000_Tensor_y_t)==8);canada_kernel_000_Tensor_y_t t2{(void*)values[2]};return cute_dsl_canada_kernel_000_wrapper(&ctx->m0,&t0,&t1,&t2,(int64_t)values[3],word_float(values[4]),(cudaStream_t)stream);}default:return -3;}}
