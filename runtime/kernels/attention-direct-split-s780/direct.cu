// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Sparse-MLA SM120 prefill. Single raw-pointer entry point that dispatches:
//   - DSV3_2 / DSV4 model split
//   - SG (single-group, 16 heads/CTA) for num_heads <= 16
//   - MG (multi-group, 32 heads/CTA) for num_heads > 16
//   - Dual-cache MG variants (DSV4 only)
//
// Raw-pointer interface; framework-agnostic.

#include <cuda_runtime.h>
#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>

#include <flashinfer/attention/sparse_mla_sm120/arch/common.cuh>
#include <flashinfer/attention/sparse_mla_sm120/common/smem_layout.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>
#include <flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh>

namespace flashinfer::sparse_mla_sm120 {

namespace {

constexpr int kMaxCachedCudaDevices = 32;

template <typename Kernel>
void configure_dynamic_smem_per_device(Kernel kernel, size_t smem_bytes,
                                       bool (&configured)[kMaxCachedCudaDevices]) {
  if (smem_bytes <= 48 * 1024) return;

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  const bool cacheable_device = device >= 0 && device < kMaxCachedCudaDevices;
  if (cacheable_device && configured[device]) return;

  const cudaError_t rc = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(smem_bytes));
  if (rc == cudaSuccess) {
    if (cacheable_device) configured[device] = true;
    return;
  }
  CUDA_CHECK(rc);
}

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
void launch_prefill_sg(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                       const float* attn_sink, bf16* output, float* out_lse, float sm_scale,
                       int num_tokens, size_t stride_kv_block, const int* topk_length_ptr,
                       cudaStream_t stream) {
  constexpr size_t smem_bytes = SmemLayout<MT, CM>::TOTAL;
  // Ceil-div so NUM_HEADS < HPB (small-TP shards) still launches 1 CTA per token.
  constexpr int REPLICATE_H = (NUM_HEADS + HPB - 1) / HPB;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE>;
  static bool configured[kMaxCachedCudaDevices] = {};
  configure_dynamic_smem_per_device(kernel, smem_bytes, configured);

  // SG is single-cache only.
  PrefillColdParams cold{sm_scale,
                         num_tokens,
                         stride_kv_block,
                         /*stride_kv_block_extra=*/(size_t)0,
                         /*topk_extra=*/0,
                         attn_sink,
                         topk_length_ptr,
                         /*topk_length_extra=*/(const int*)nullptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,      (void*)&KV_cache, (void*)&indices, (void*)&attn_sink,
                  (void*)&output, (void*)&out_lse,  (void*)&cold};
  CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

// Single-cache MG dispatcher. MG_N_HG_T: 1 lets NUM_HEADS=8/16 use a 16-head
// CTA (NH=8 is internally padded); 2 is the default for NH >= 32.
template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE,
          int MG_N_HG_T = MG_N_HG_DEFAULT>
void launch_prefill_mg(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                       const float* attn_sink, bf16* output, float* out_lse, float sm_scale,
                       int num_tokens, size_t stride_kv_block, const int* topk_length_ptr,
                       cudaStream_t stream) {
  constexpr size_t smem_bytes = SmemLayoutMG<MT, CM>::TOTAL;
  constexpr int MG_HEADS_PER_CTA_LOCAL = MG_N_HG_T * HPB;
  static_assert(NUM_HEADS % MG_HEADS_PER_CTA_LOCAL == 0 || (MG_N_HG_T == 1 && NUM_HEADS < HPB),
                "NUM_HEADS must fill the MG head tile, except a padded NH=8 tile");
  constexpr int REPLICATE_H = (NUM_HEADS + MG_HEADS_PER_CTA_LOCAL - 1) / MG_HEADS_PER_CTA_LOCAL;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_mg_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE, MG_N_HG_T>;
  static bool configured[kMaxCachedCudaDevices] = {};
  configure_dynamic_smem_per_device(kernel, smem_bytes, configured);

  PrefillColdParams cold{sm_scale,
                         num_tokens,
                         stride_kv_block,
                         /*stride_kv_block_extra=*/(size_t)0,
                         /*topk_extra=*/0,
                         attn_sink,
                         topk_length_ptr,
                         /*topk_length_extra=*/(const int*)nullptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,       (void*)&KV_cache,  (void*)&indices, (void*)&output,
                  (void*)&out_lse, (void*)&attn_sink, (void*)&cold};
  CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

// Dual-cache MG dispatcher. `topk_extra` is runtime; PAGE_BLOCK_SIZE_EXTRA
// stays template because it changes the KV stride.
template <ModelType MT, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE, int PAGE_BLOCK_SIZE_EXTRA,
          int MG_N_HG_T = MG_N_HG_DEFAULT>
void launch_prefill_mg_dual_fulltile(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                                     const uint8_t* KV_cache_extra, const int32_t* indices_extra,
                                     const float* attn_sink, bf16* output, float* out_lse,
                                     float sm_scale, int num_tokens, int topk_extra,
                                     size_t stride_kv_block, size_t stride_kv_block_extra,
                                     cudaStream_t stream) {
  constexpr size_t smem_bytes = SmemLayoutMG<MT, ComputeMode::BF16>::TOTAL;
  constexpr int MG_HEADS_PER_CTA_LOCAL = MG_N_HG_T * HPB;
  static_assert(NUM_HEADS % MG_HEADS_PER_CTA_LOCAL == 0 || (MG_N_HG_T == 1 && NUM_HEADS < HPB),
                "NUM_HEADS must fill the MG head tile, except a padded NH=8 tile");
  constexpr int REPLICATE_H = (NUM_HEADS + MG_HEADS_PER_CTA_LOCAL - 1) / MG_HEADS_PER_CTA_LOCAL;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_mg_dual_fulltile_kernel<MT, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE,
                                                           PAGE_BLOCK_SIZE_EXTRA, MG_N_HG_T>;
  static bool configured[kMaxCachedCudaDevices] = {};
  configure_dynamic_smem_per_device(kernel, smem_bytes, configured);

  PrefillColdParams cold{sm_scale,
                         num_tokens,
                         stride_kv_block,
                         stride_kv_block_extra,
                         topk_extra,
                         attn_sink,
                         /*topk_length=*/(const int*)nullptr,
                         /*topk_length_extra=*/(const int*)nullptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,
                  (void*)&KV_cache,
                  (void*)&indices,
                  (void*)&KV_cache_extra,
                  (void*)&indices_extra,
                  (void*)&output,
                  (void*)&out_lse,
                  (void*)&attn_sink,
                  (void*)&cold};
  CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE,
          int PAGE_BLOCK_SIZE_EXTRA, int MG_N_HG_T = MG_N_HG_DEFAULT>
void launch_prefill_mg_dual(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                            const uint8_t* KV_cache_extra, const int32_t* indices_extra,
                            const float* attn_sink, bf16* output, float* out_lse, float sm_scale,
                            int num_tokens, int topk_extra, size_t stride_kv_block,
                            size_t stride_kv_block_extra, const int* topk_length_ptr,
                            const int* topk_length_extra_ptr, cudaStream_t stream) {
  constexpr size_t smem_bytes = SmemLayoutMG<MT, CM>::TOTAL;
  constexpr int MG_HEADS_PER_CTA_LOCAL = MG_N_HG_T * HPB;
  static_assert(NUM_HEADS % MG_HEADS_PER_CTA_LOCAL == 0 || (MG_N_HG_T == 1 && NUM_HEADS < HPB),
                "NUM_HEADS must fill the MG head tile, except a padded NH=8 tile");
  constexpr int REPLICATE_H = (NUM_HEADS + MG_HEADS_PER_CTA_LOCAL - 1) / MG_HEADS_PER_CTA_LOCAL;
  dim3 grid(num_tokens * CANADA_BF16_KEY_SPLITS * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_mg_dual_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE,
                                                  PAGE_BLOCK_SIZE_EXTRA, MG_N_HG_T>;
  static bool configured[kMaxCachedCudaDevices] = {};
  configure_dynamic_smem_per_device(kernel, smem_bytes, configured);

  PrefillColdParams cold{sm_scale,   num_tokens, stride_kv_block, stride_kv_block_extra,
                         topk_extra, attn_sink,  topk_length_ptr, topk_length_extra_ptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,
                  (void*)&KV_cache,
                  (void*)&indices,
                  (void*)&KV_cache_extra,
                  (void*)&indices_extra,
                  (void*)&output,
                  (void*)&out_lse,
                  (void*)&attn_sink,
                  (void*)&cold};
  CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}


} // anonymous namespace
extern "C" int ca_direct_split_abi() { return 1; }
extern "C" int ca_sparse_prefill(
 int rows,int extra_topk,int extra_pbs,float scale,const void* q,const void* kv,
 size_t stride,const void* indices,void* output,void* lse,const void* lengths,
 const void* sink,const void* extra_kv,size_t extra_stride,const void* extra_indices,
 const void* extra_lengths,cudaStream_t stream) {
 if(rows<1||rows>16||(extra_topk!=512&&extra_topk!=1024)||extra_pbs!=2||!(scale>0)
  ||!q||!kv||!indices||!output||!lse||!lengths||!sink||!extra_kv||!extra_indices||!extra_lengths)return -1;
 launch_prefill_mg_dual<ModelType::DSV4,ComputeMode::BF16,16,128,64,2,1>(
  (const bf16*)q,(const uint8_t*)kv,(const int32_t*)indices,(const uint8_t*)extra_kv,
  (const int32_t*)extra_indices,(const float*)sink,(bf16*)output,(float*)lse,
  scale,rows,extra_topk,stride,extra_stride,(const int*)lengths,(const int*)extra_lengths,stream);
 return (int)cudaPeekAtLastError();
}
} // namespace flashinfer::sparse_mla_sm120
