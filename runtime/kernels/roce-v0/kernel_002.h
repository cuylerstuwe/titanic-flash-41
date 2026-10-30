
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
} canada_kernel_002_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_canada_kernel_002_cuda_init(void **);
void _mlir_canada_kernel_002_cuda_load_to_device(void **);
static inline void canada_kernel_002_Kernel_Module_Load(canada_kernel_002_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret = cudaSuccess;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_canada_kernel_002_cuda_init((void **)(&initArgs));
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
        _mlir_canada_kernel_002_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void canada_kernel_002_Kernel_Module_Unload(canada_kernel_002_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
#endif
void _mlir_canada_kernel_002__mlir_ciface_cutlass___call___b12xcommroce_oneshot_cute_RoceOneshotLaunch_object_at__Ptrgmem_Ptrgmem_1_16_16_16_16_16_4096_16_16_16_16_1_1_CUstream0x0(void **args, int32_t num_args);

static inline int32_t cute_dsl_canada_kernel_002_wrapper(canada_kernel_002_Kernel_Module_t *module, void *input_ptr, void *output_ptr, int32_t size_packs, int32_t nbytes, int64_t recv_base, int64_t flag_base, int64_t send_base, int64_t ctrl_base, int64_t slot_bytes, int64_t epoch_ptr, int64_t stage_counter_ptr, int64_t tail_counter_ptr, int64_t poison_ptr, uint32_t spin_limit, int32_t grid_x, cudaStream_t stream) {
    int32_t ret = 0;
    void *args[17] = {
        &input_ptr, &output_ptr, &size_packs, &nbytes, &recv_base, &flag_base, &send_base, &ctrl_base, &slot_bytes, &epoch_ptr, &stage_counter_ptr, &tail_counter_ptr, &poison_ptr, &spin_limit, &grid_x, &stream,
        &ret
    };
    _mlir_canada_kernel_002__mlir_ciface_cutlass___call___b12xcommroce_oneshot_cute_RoceOneshotLaunch_object_at__Ptrgmem_Ptrgmem_1_16_16_16_16_16_4096_16_16_16_16_1_1_CUstream0x0(args, 17);
    return ret;
}
