
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
} canada_kernel_001_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_canada_kernel_001_cuda_init(void **);
void _mlir_canada_kernel_001_cuda_load_to_device(void **);
static inline void canada_kernel_001_Kernel_Module_Load(canada_kernel_001_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret = cudaSuccess;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_canada_kernel_001_cuda_init((void **)(&initArgs));
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
        _mlir_canada_kernel_001_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void canada_kernel_001_Kernel_Module_Unload(canada_kernel_001_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

typedef struct {
    void *data;
} canada_kernel_001_Tensor_barrier_count_t;


typedef struct {
    void *data;
} canada_kernel_001_Tensor_barrier_epoch_t;


typedef struct {
    void *data;
} canada_kernel_001_Tensor_pair_head_t;


typedef struct {
    void *data;
} canada_kernel_001_Tensor_producers_done_count_t;


typedef struct {
    void *data;
} canada_kernel_001_Tensor_all_work_published_t;


typedef struct {
    void *data;
} canada_kernel_001_Tensor_task_head_t;


typedef struct {
    void *data;
} canada_kernel_001_Tensor_task_tail_t;


typedef struct {
    void *data;
} canada_kernel_001_Tensor_b_w13_t;


typedef struct {
    void *data;
} canada_kernel_001_Tensor_b_down_t;


typedef struct {
    void *data;
} canada_kernel_001_Tensor_row_counts_t;


typedef struct {
    void *data;
} canada_kernel_001_Tensor_expert_write_rows_t;


typedef struct {
    void *data;
} canada_kernel_001_Tensor_expert_tile_base_t;


typedef struct {
    void *data;
} canada_kernel_001_Tensor_input_global_scale_t;


typedef struct {
    void *data;
} canada_kernel_001_Tensor_alpha_t;


typedef struct {
    void *data;
} canada_kernel_001_Tensor_down_alpha_t;


typedef struct {
    void *data;
} canada_kernel_001_Tensor_global_scale_t;

#ifdef __cplusplus
extern "C"
#endif
void _mlir_canada_kernel_001__mlir_ciface_cutlass___call___b12x_nextmoefused_moe_impl_DynamicMoEW4A8Launch_object_at__Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_FakeTensorInt3211_FakeTensorInt3211_Fake(void **args, int32_t num_args);

static inline int32_t cute_dsl_canada_kernel_001_wrapper(canada_kernel_001_Kernel_Module_t *module, void *a_ptr, void *topk_ids_ptr, void *topk_weights_ptr, void *packed_a_ptr, void *sfa_ptr, void *packed_a_storage_ptr, void *scale_storage_ptr, void *intermediate_ptr, canada_kernel_001_Tensor_barrier_count_t *barrier_count, canada_kernel_001_Tensor_barrier_epoch_t *barrier_epoch, canada_kernel_001_Tensor_pair_head_t *pair_head, canada_kernel_001_Tensor_producers_done_count_t *producers_done_count, canada_kernel_001_Tensor_all_work_published_t *all_work_published, canada_kernel_001_Tensor_task_head_t *task_head, canada_kernel_001_Tensor_task_tail_t *task_tail, void *task_ready_ptr, void *task_expert_ptr, void *task_m_tile_ptr, void *task_slice_begin_ptr, void *task_slice_count_ptr, void *task_valid_rows_ptr, void *tile_write_count_ptr, canada_kernel_001_Tensor_b_w13_t *b_w13, void *sfb_w13_ptr, canada_kernel_001_Tensor_b_down_t *b_down, void *sfb_down_ptr, void *sfb_w13_mx_ptr, void *sfb_down_mx_ptr, void *w13_residual_ptr, void *down_residual_ptr, void *w13_rp_ptr, void *w13_sfb_rp_ptr, void *down_rp_ptr, void *down_sfb_rp_ptr, canada_kernel_001_Tensor_row_counts_t *row_counts, canada_kernel_001_Tensor_expert_write_rows_t *expert_write_rows, canada_kernel_001_Tensor_expert_tile_base_t *expert_tile_base, canada_kernel_001_Tensor_input_global_scale_t *input_global_scale, canada_kernel_001_Tensor_alpha_t *alpha, canada_kernel_001_Tensor_down_alpha_t *down_alpha, canada_kernel_001_Tensor_global_scale_t *global_scale, void *scatter_ptr, void *token_map_ptr, void *token_weights_ptr, int32_t num_tokens, int32_t max_rows, int32_t scatter_rows, int32_t rows_padded, int32_t max_tasks, int32_t max_phys_tiles, int32_t max_active_clusters, cudaStream_t stream) {
    int32_t ret = 0;
    void *args[53] = {
        &a_ptr, &topk_ids_ptr, &topk_weights_ptr, &packed_a_ptr, &sfa_ptr, &packed_a_storage_ptr, &scale_storage_ptr, &intermediate_ptr, barrier_count, barrier_epoch, pair_head, producers_done_count, all_work_published, task_head, task_tail, &task_ready_ptr, &task_expert_ptr, &task_m_tile_ptr, &task_slice_begin_ptr, &task_slice_count_ptr, &task_valid_rows_ptr, &tile_write_count_ptr, b_w13, &sfb_w13_ptr, b_down, &sfb_down_ptr, &sfb_w13_mx_ptr, &sfb_down_mx_ptr, &w13_residual_ptr, &down_residual_ptr, &w13_rp_ptr, &w13_sfb_rp_ptr, &down_rp_ptr, &down_sfb_rp_ptr, row_counts, expert_write_rows, expert_tile_base, input_global_scale, alpha, down_alpha, global_scale, &scatter_ptr, &token_map_ptr, &token_weights_ptr, &num_tokens, &max_rows, &scatter_rows, &rows_padded, &max_tasks, &max_phys_tiles, &max_active_clusters, &stream,
        &ret
    };
    _mlir_canada_kernel_001__mlir_ciface_cutlass___call___b12x_nextmoefused_moe_impl_DynamicMoEW4A8Launch_object_at__Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_FakeTensorInt3211_FakeTensorInt3211_Fake(args, 53);
    return ret;
}
