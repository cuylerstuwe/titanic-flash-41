// Titanic raw-pointer adapter around FlashInfer v0.7.3's SM120 sparse-MLA DeepSeek-V4.1 BF16 prefill
// kernel (include/flashinfer/attention/sparse_mla_sm120, Apache-2.0 / BSD-3-Clause; see manifest.json).
// No engine code. The FlashInfer source is unmodified; this file only fills its parameter struct.
//
// Caches (both footer-scale layouts, as SGLang's V4.1 store kernels write them):
//   main  (sliding window): 512 fp8 e4m3 + 16 ue8m0 scales (one per 32 values) = 528 B/token
//   extra (compressed):     512 e2m1 (two per byte) + 32 e4m3 scales (one per 16) = 288 B/token
// Numerics: both caches are dequantized exactly to bf16; QK and PV are bf16 MMAs with fp32 accumulation;
// the attention sink is folded into the normalization; the output LSE is log2-based, as before.
//
// ABI 2 keeps ABI 1's argument list (ca_sparse_prefill) so the engine binds it the same way.
#include <cuda_runtime.h>

#include <flashinfer/attention/sparse_mla_sm120/execution/prefill_launch.cuh>

namespace fi = flashinfer::sparse_mla_sm120;

extern "C" int ca_sparse_prefill_abi() { return 2; }

extern "C" int ca_sparse_prefill(int rows, int extra_topk, int extra_pbs, float scale, const void* q,
                                 const void* kv, size_t stride, const void* indices, void* output,
                                 void* lse, const void* lengths, const void* sink,
                                 const void* extra_kv, size_t extra_stride,
                                 const void* extra_indices, const void* extra_lengths,
                                 cudaStream_t stream) {
  constexpr int kHeads = 16, kMainTopk = 128, kMainPage = 64, kExtraPage = 64;
  if (rows < 1 || rows > 16384 || !(scale > 0) || extra_topk < 0 || extra_topk > 1024 ||
      extra_topk % 64 || !q || !kv || !indices || !output || !lse || !lengths || !sink ||
      stride == 0)
    return -1;
  fi::execution::AttentionParams p{};
  p.num_heads = kHeads;
  p.topk = kMainTopk;
  p.q = static_cast<const __nv_bfloat16*>(q);
  p.kv = static_cast<const uint8_t*>(kv);
  p.indices = static_cast<const int32_t*>(indices);
  p.topk_length = static_cast<const int*>(lengths);
  p.output = static_cast<__nv_bfloat16*>(output);
  p.out_lse = static_cast<float*>(lse);
  p.attn_sink = static_cast<const float*>(sink);
  p.num_tokens = rows;
  p.sm_scale = scale;
  p.page_size = kMainPage;
  p.page_stride_bytes = stride;
  p.indices_stride_elems = kMainTopk;
  p.out_lse_stride_elems = kHeads;
  p.lse_scale = 1.f;
  if (extra_topk == 0) {
    if (extra_kv || extra_indices || extra_lengths) return -2;
    const auto r = fi::launch_dsv41_bf16_prefill<kHeads, false, ::ModelType::DSV4_1>(p, stream);
    if (r.error != cudaSuccess) return (int)r.error;
  } else {
    if (!extra_kv || !extra_indices || !extra_lengths || extra_pbs != kExtraPage ||
        extra_stride == 0)
      return -3;
    p.extra_kv = static_cast<const uint8_t*>(extra_kv);
    p.extra_indices = static_cast<const int32_t*>(extra_indices);
    p.extra_topk_length = static_cast<const int*>(extra_lengths);
    p.extra_topk = extra_topk;
    p.extra_page_size = kExtraPage;
    p.extra_page_stride_bytes = extra_stride;
    p.extra_indices_stride_elems = (size_t)extra_topk;
    p.extra_fp4 = true;
    const auto r = fi::launch_dsv41_bf16_prefill<kHeads, true, ::ModelType::DSV4_1>(p, stream);
    if (r.error != cudaSuccess) return (int)r.error;
  }
  return (int)cudaPeekAtLastError();
}
