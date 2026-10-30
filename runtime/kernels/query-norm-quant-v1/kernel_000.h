
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
} canada_kernel_000_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_canada_kernel_000_cuda_init(void **);
void _mlir_canada_kernel_000_cuda_load_to_device(void **);
static inline void canada_kernel_000_Kernel_Module_Load(canada_kernel_000_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret = cudaSuccess;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_canada_kernel_000_cuda_init((void **)(&initArgs));
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
        _mlir_canada_kernel_000_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void canada_kernel_000_Kernel_Module_Unload(canada_kernel_000_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

typedef struct {
    void *data;
} canada_kernel_000_Tensor_x_t;


typedef struct {
    void *data;
} canada_kernel_000_Tensor_w_t;


typedef struct {
    void *data;
} canada_kernel_000_Tensor_y_t;


typedef struct {
    void *data;
} canada_kernel_000_Tensor_q_t;


typedef struct {
    void *data;
} canada_kernel_000_Tensor_s_t;

#ifdef __cplusplus
extern "C"
#endif
void _mlir_canada_kernel_000__mlir_ciface_cutlass_entry_query_rms_quantQueryRMSQuantobjectat_FakeTensorBFloat1665536128012801_FakeTensorBFloat1612801_FakeTensorBFloat1665536128012801_FakeTensorUint64655361601601_FakeTensor(void **args, int32_t num_args);

static inline int32_t cute_dsl_canada_kernel_000_wrapper(canada_kernel_000_Kernel_Module_t *module, canada_kernel_000_Tensor_x_t *x, canada_kernel_000_Tensor_w_t *w, canada_kernel_000_Tensor_y_t *y, canada_kernel_000_Tensor_q_t *q, canada_kernel_000_Tensor_s_t *s, int64_t rows, float eps, cudaStream_t stream) {
    int32_t ret = 0;
    void *args[9] = {
        x, w, y, q, s, &rows, &eps, &stream,
        &ret
    };
    _mlir_canada_kernel_000__mlir_ciface_cutlass_entry_query_rms_quantQueryRMSQuantobjectat_FakeTensorBFloat1665536128012801_FakeTensorBFloat1612801_FakeTensorBFloat1665536128012801_FakeTensorUint64655361601601_FakeTensor(args, 9);
    return ret;
}
