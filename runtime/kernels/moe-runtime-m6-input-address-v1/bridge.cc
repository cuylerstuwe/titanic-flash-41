#include "kernel_000.h"
#include "kernel_001.h"
#include "kernel_002.h"
#include <memory>
struct Context {
canada_kernel_000_Kernel_Module_t m0{};
canada_kernel_001_Kernel_Module_t m1{};
canada_kernel_002_Kernel_Module_t m2{};
~Context(){
if(m0.module)cudaLibraryUnload(m0.module);
if(m1.module)cudaLibraryUnload(m1.module);
if(m2.module)cudaLibraryUnload(m2.module);
}};
extern "C" int32_t ca_primitives_abi(){return 1;}
extern "C" void* ca_primitives_open(int32_t* status){
 *status=0;auto ctx=std::make_unique<Context>();
{cudaLibrary_t* ptr=&ctx->m0.module;cudaError_t error=cudaSuccess;
        void* init_args[]={&ptr,&error};_mlir_canada_kernel_000_cuda_init(init_args);
        if(error!=cudaSuccess){*status=(int)error;return nullptr;}
        int32_t device=0;void* load_args[]={&ptr,&device,&error};
        _mlir_canada_kernel_000_cuda_load_to_device(load_args);
        if(error!=cudaSuccess){*status=(int)error;return nullptr;}}
{cudaLibrary_t* ptr=&ctx->m1.module;cudaError_t error=cudaSuccess;
        void* init_args[]={&ptr,&error};_mlir_canada_kernel_001_cuda_init(init_args);
        if(error!=cudaSuccess){*status=(int)error;return nullptr;}
        int32_t device=0;void* load_args[]={&ptr,&device,&error};
        _mlir_canada_kernel_001_cuda_load_to_device(load_args);
        if(error!=cudaSuccess){*status=(int)error;return nullptr;}}
{cudaLibrary_t* ptr=&ctx->m2.module;cudaError_t error=cudaSuccess;
        void* init_args[]={&ptr,&error};_mlir_canada_kernel_002_cuda_init(init_args);
        if(error!=cudaSuccess){*status=(int)error;return nullptr;}
        int32_t device=0;void* load_args[]={&ptr,&device,&error};
        _mlir_canada_kernel_002_cuda_load_to_device(load_args);
        if(error!=cudaSuccess){*status=(int)error;return nullptr;}}
 return ctx.release();
}
extern "C" void ca_primitives_close(void* context){delete (Context*)context;}
extern "C" int32_t ca_primitives_launch(void* context,int32_t index,const uint64_t* pointers,
 size_t np,const int32_t* scalars,size_t ns,void* stream){
 if(!context)return -1;auto ctx=(Context*)context;
 switch(index){
case 0: {if(np!=44||ns!=7)return -2;
        static_assert(sizeof(canada_kernel_000_Tensor_barrier_count_t)==sizeof(void*));canada_kernel_000_Tensor_barrier_count_t t8{(void*)pointers[8]};static_assert(sizeof(canada_kernel_000_Tensor_barrier_epoch_t)==sizeof(void*));canada_kernel_000_Tensor_barrier_epoch_t t9{(void*)pointers[9]};static_assert(sizeof(canada_kernel_000_Tensor_pair_head_t)==sizeof(void*));canada_kernel_000_Tensor_pair_head_t t10{(void*)pointers[10]};static_assert(sizeof(canada_kernel_000_Tensor_producers_done_count_t)==sizeof(void*));canada_kernel_000_Tensor_producers_done_count_t t11{(void*)pointers[11]};static_assert(sizeof(canada_kernel_000_Tensor_all_work_published_t)==sizeof(void*));canada_kernel_000_Tensor_all_work_published_t t12{(void*)pointers[12]};static_assert(sizeof(canada_kernel_000_Tensor_task_head_t)==sizeof(void*));canada_kernel_000_Tensor_task_head_t t13{(void*)pointers[13]};static_assert(sizeof(canada_kernel_000_Tensor_task_tail_t)==sizeof(void*));canada_kernel_000_Tensor_task_tail_t t14{(void*)pointers[14]};static_assert(sizeof(canada_kernel_000_Tensor_b_w13_t)==sizeof(void*));canada_kernel_000_Tensor_b_w13_t t22{(void*)pointers[22]};static_assert(sizeof(canada_kernel_000_Tensor_b_down_t)==sizeof(void*));canada_kernel_000_Tensor_b_down_t t24{(void*)pointers[24]};static_assert(sizeof(canada_kernel_000_Tensor_row_counts_t)==sizeof(void*));canada_kernel_000_Tensor_row_counts_t t34{(void*)pointers[34]};static_assert(sizeof(canada_kernel_000_Tensor_expert_write_rows_t)==sizeof(void*));canada_kernel_000_Tensor_expert_write_rows_t t35{(void*)pointers[35]};static_assert(sizeof(canada_kernel_000_Tensor_expert_tile_base_t)==sizeof(void*));canada_kernel_000_Tensor_expert_tile_base_t t36{(void*)pointers[36]};static_assert(sizeof(canada_kernel_000_Tensor_input_global_scale_t)==sizeof(void*));canada_kernel_000_Tensor_input_global_scale_t t37{(void*)pointers[37]};static_assert(sizeof(canada_kernel_000_Tensor_alpha_t)==sizeof(void*));canada_kernel_000_Tensor_alpha_t t38{(void*)pointers[38]};static_assert(sizeof(canada_kernel_000_Tensor_down_alpha_t)==sizeof(void*));canada_kernel_000_Tensor_down_alpha_t t39{(void*)pointers[39]};static_assert(sizeof(canada_kernel_000_Tensor_global_scale_t)==sizeof(void*));canada_kernel_000_Tensor_global_scale_t t40{(void*)pointers[40]};
        return cute_dsl_canada_kernel_000_wrapper(&ctx->m0,(void*)pointers[0],(void*)pointers[1],(void*)pointers[2],(void*)pointers[3],(void*)pointers[4],(void*)pointers[5],(void*)pointers[6],(void*)pointers[7],&t8,&t9,&t10,&t11,&t12,&t13,&t14,(void*)pointers[15],(void*)pointers[16],(void*)pointers[17],(void*)pointers[18],(void*)pointers[19],(void*)pointers[20],(void*)pointers[21],&t22,(void*)pointers[23],&t24,(void*)pointers[25],(void*)pointers[26],(void*)pointers[27],(void*)pointers[28],(void*)pointers[29],(void*)pointers[30],(void*)pointers[31],(void*)pointers[32],(void*)pointers[33],&t34,&t35,&t36,&t37,&t38,&t39,&t40,(void*)pointers[41],(void*)pointers[42],(void*)pointers[43],scalars[0],scalars[1],scalars[2],scalars[3],scalars[4],scalars[5],scalars[6],(cudaStream_t)stream);}
case 1: {if(np!=44||ns!=7)return -2;
        static_assert(sizeof(canada_kernel_001_Tensor_barrier_count_t)==sizeof(void*));canada_kernel_001_Tensor_barrier_count_t t8{(void*)pointers[8]};static_assert(sizeof(canada_kernel_001_Tensor_barrier_epoch_t)==sizeof(void*));canada_kernel_001_Tensor_barrier_epoch_t t9{(void*)pointers[9]};static_assert(sizeof(canada_kernel_001_Tensor_pair_head_t)==sizeof(void*));canada_kernel_001_Tensor_pair_head_t t10{(void*)pointers[10]};static_assert(sizeof(canada_kernel_001_Tensor_producers_done_count_t)==sizeof(void*));canada_kernel_001_Tensor_producers_done_count_t t11{(void*)pointers[11]};static_assert(sizeof(canada_kernel_001_Tensor_all_work_published_t)==sizeof(void*));canada_kernel_001_Tensor_all_work_published_t t12{(void*)pointers[12]};static_assert(sizeof(canada_kernel_001_Tensor_task_head_t)==sizeof(void*));canada_kernel_001_Tensor_task_head_t t13{(void*)pointers[13]};static_assert(sizeof(canada_kernel_001_Tensor_task_tail_t)==sizeof(void*));canada_kernel_001_Tensor_task_tail_t t14{(void*)pointers[14]};static_assert(sizeof(canada_kernel_001_Tensor_b_w13_t)==sizeof(void*));canada_kernel_001_Tensor_b_w13_t t22{(void*)pointers[22]};static_assert(sizeof(canada_kernel_001_Tensor_b_down_t)==sizeof(void*));canada_kernel_001_Tensor_b_down_t t24{(void*)pointers[24]};static_assert(sizeof(canada_kernel_001_Tensor_row_counts_t)==sizeof(void*));canada_kernel_001_Tensor_row_counts_t t34{(void*)pointers[34]};static_assert(sizeof(canada_kernel_001_Tensor_expert_write_rows_t)==sizeof(void*));canada_kernel_001_Tensor_expert_write_rows_t t35{(void*)pointers[35]};static_assert(sizeof(canada_kernel_001_Tensor_expert_tile_base_t)==sizeof(void*));canada_kernel_001_Tensor_expert_tile_base_t t36{(void*)pointers[36]};static_assert(sizeof(canada_kernel_001_Tensor_input_global_scale_t)==sizeof(void*));canada_kernel_001_Tensor_input_global_scale_t t37{(void*)pointers[37]};static_assert(sizeof(canada_kernel_001_Tensor_alpha_t)==sizeof(void*));canada_kernel_001_Tensor_alpha_t t38{(void*)pointers[38]};static_assert(sizeof(canada_kernel_001_Tensor_down_alpha_t)==sizeof(void*));canada_kernel_001_Tensor_down_alpha_t t39{(void*)pointers[39]};static_assert(sizeof(canada_kernel_001_Tensor_global_scale_t)==sizeof(void*));canada_kernel_001_Tensor_global_scale_t t40{(void*)pointers[40]};
        return cute_dsl_canada_kernel_001_wrapper(&ctx->m1,(void*)pointers[0],(void*)pointers[1],(void*)pointers[2],(void*)pointers[3],(void*)pointers[4],(void*)pointers[5],(void*)pointers[6],(void*)pointers[7],&t8,&t9,&t10,&t11,&t12,&t13,&t14,(void*)pointers[15],(void*)pointers[16],(void*)pointers[17],(void*)pointers[18],(void*)pointers[19],(void*)pointers[20],(void*)pointers[21],&t22,(void*)pointers[23],&t24,(void*)pointers[25],(void*)pointers[26],(void*)pointers[27],(void*)pointers[28],(void*)pointers[29],(void*)pointers[30],(void*)pointers[31],(void*)pointers[32],(void*)pointers[33],&t34,&t35,&t36,&t37,&t38,&t39,&t40,(void*)pointers[41],(void*)pointers[42],(void*)pointers[43],scalars[0],scalars[1],scalars[2],scalars[3],scalars[4],scalars[5],scalars[6],(cudaStream_t)stream);}
case 2: {if(np!=6||ns!=3)return -2;
        
        return cute_dsl_canada_kernel_002_wrapper(&ctx->m2,(void*)pointers[0],(void*)pointers[1],(void*)pointers[2],(void*)pointers[3],(void*)pointers[4],(void*)pointers[5],scalars[0],scalars[1],scalars[2],(cudaStream_t)stream);}
 default:return -3;
 }
}
