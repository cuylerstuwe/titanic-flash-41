
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>


// Macro to check for cuda errors.
#ifndef CUTE_DSL_CUDA_ERROR_CHECK
#define CUTE_DSL_CUDA_ERROR_CHECK(err) { \
    if ((err) != cudaSuccess) { \
        printf("Got Cuda Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
    } \
}

#endif

typedef struct {
    cudaLibrary_t module;
} canada_kernel_004_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_canada_kernel_004_cuda_init(void **);
void _mlir_canada_kernel_004_cuda_load_to_device(void **);
static inline void canada_kernel_004_Kernel_Module_Load(canada_kernel_004_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret = cudaSuccess;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_canada_kernel_004_cuda_init((void **)(&initArgs));
    CUTE_DSL_CUDA_ERROR_CHECK(ret);
    int32_t device_id = 0;
    struct {
        cudaLibrary_t **library;
        int32_t *device_id;
        cudaError_t *ret;
    } loadArgs = {&libraryPtr, &device_id, &ret};
    int32_t device_count;
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceCount(&device_count));
    for (int32_t i = 0; i < device_count; i++) {
        device_id = i;
        _mlir_canada_kernel_004_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void canada_kernel_004_Kernel_Module_Unload(canada_kernel_004_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
#endif
void _mlir_canada_kernel_004__mlir_ciface_cutlass___call___b12x_nextmoe_sharedkernelsw4a16kernelW4A16TopKSumKernel_object_at__Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem___1_CUstream0x0(void **args, int32_t num_args);

static inline int32_t cute_dsl_canada_kernel_004_wrapper(canada_kernel_004_Kernel_Module_t *module, void *fc2_ptr, void *output_ptr, void *topk_weights_ptr, void *route_expert_ids_ptr, void *expert_map_ptr, void *svh_ptr, int32_t weight_num_experts, int32_t route_num_experts, int32_t active_m, cudaStream_t stream) {
    int32_t ret = 0;
    void *args[11] = {
        &fc2_ptr, &output_ptr, &topk_weights_ptr, &route_expert_ids_ptr, &expert_map_ptr, &svh_ptr, &weight_num_experts, &route_num_experts, &active_m, &stream,
        &ret
    };
    _mlir_canada_kernel_004__mlir_ciface_cutlass___call___b12x_nextmoe_sharedkernelsw4a16kernelW4A16TopKSumKernel_object_at__Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem___1_CUstream0x0(args, 11);
    return ret;
}
