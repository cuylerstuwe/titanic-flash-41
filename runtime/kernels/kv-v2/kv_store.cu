// Device kernel adapted from SGLang, Apache-2.0.
// Pinned source and all dependency hashes are recorded in manifest.json.
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>
#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>
#include <sgl_kernel/deepseek_v4/kv_layout.cuh>
namespace sglang {
using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;
constexpr uint32_t kFusedKBlockSize = 256;
constexpr uint32_t kFusedKNumWarps = 8;
#define K_KERNEL __global__ __launch_bounds__(kFusedKBlockSize, 8)
struct FusedKNormRopeFlashMLAParams {
  const void* __restrict__ kv;          // (B, kHeadDim) DType
  const void* __restrict__ kv_weight;   // (kHeadDim,) DType
  const float* __restrict__ freqs_cis;  // (max_pos, kRopeDim) fp32
  const void* __restrict__ positions;   // (B,) PosT
  const int32_t* __restrict__ out_loc;  // (B,) int32 -> cache slot id
  uint8_t* __restrict__ kvcache;        // (npages, kPageBytes) uint8
  // Row stride for `kv` in elements. Required because the upstream caller often
  // passes `qkv_a[..., q_lora_rank:]`, a non-contiguous slice whose stride[0]
  // equals `q_lora_rank + kHeadDim` rather than `kHeadDim`.
  int64_t kv_stride_batch;
  uint32_t batch_size;
  float eps;
};

template <
    typename DType,
    int64_t kHeadDim,
    int64_t kRopeDim,
    typename PosT,
    int32_t kPageBits,
    deepseek_v4::KVLayout kLayout,
    bool kUsePDL>
K_KERNEL void fused_k_norm_rope_flashmla(const __grid_constant__ FusedKNormRopeFlashMLAParams params) {
  using namespace device;

  constexpr int64_t kVecSize = 2;
  constexpr uint32_t kRopeWarp = kFusedKNumWarps - 1;
  using Paged = deepseek_v4::PagedKV<kLayout, kPageBits>;
  static_assert(kHeadDim == kFusedKBlockSize * kVecSize);
  static_assert(kRopeDim == kWarpThreads * kVecSize);
  static_assert(kHeadDim - kRopeDim == kRopeWarp * kWarpThreads * kVecSize);
  using Storage = AlignedVector<DType, kVecSize>;
  using Float2 = AlignedVector<float, kVecSize>;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto work_id = blockIdx.x;
  if (work_id >= params.batch_size) return;

  const auto input_ptr = static_cast<const DType*>(params.kv) + work_id * params.kv_stride_batch;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[work_id]);
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  const auto out_loc = params.out_loc[work_id];
  Float2 data, freq;

  // part 1: norm. Each thread owns one 2-elem pack (the `tx`-th).
  // Sum-of-squares is reduced block-wide via per-warp partials.
  {
    __shared__ float partial_sums[kFusedKNumWarps];

    Storage input_vec, weight_vec;
    input_vec.load(input_ptr, tx);
    weight_vec.load(params.kv_weight, tx);
    if (warp_id == kRopeWarp) freq.load(freqs_cis, lane_id);

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto x = cast<float>(input_vec[i]);
      sum_of_squares += x * x;
    }
    const auto warp_sum = warp::reduce_sum(sum_of_squares);
    if (lane_id == 0) partial_sums[warp_id] = warp_sum;
    __syncthreads();
    // Replicate the per-warp partial sums onto all lanes of one warp and
    // reduce. Every group of `kBlockItemNumWarps` lanes ends up with the
    // global sum.
    sum_of_squares = warp::reduce_sum<kFusedKNumWarps>(partial_sums[lane_id % kFusedKNumWarps]);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto x = cast<float>(input_vec[i]);
      const auto w = cast<float>(weight_vec[i]);
      data[i] = x * norm_factor * w;
    }
  }

  // A negative out_loc marks a slot with no KV write target (e.g. the -1
  // sentinel from the full->SWA translation for out-of-window tokens or
  // padded rows); skip the row instead of writing out of bounds. Checked
  // here, not at the load, so the out_loc prefetch overlaps the norm above.
  if (out_loc < 0) return;

  const auto row = Paged::row(params.kvcache, out_loc);

  if constexpr (kLayout != deepseek_v4::KVLayout::V4) {
    // V4.1 layouts: every dim is quantized, with one scale per 32 (fp8) or 16 (fp4)
    // values. The reference quantizes the bf16 tensor kv_norm produces and rotates
    // the tail in bf16, so round the normed values to the storage dtype, rotate,
    // round again, then quantize the whole row.
    using Packed = packed_t<DType>;
    PDLTriggerSecondary<kUsePDL>();

    auto rounded = cast<fp32x2_t>(cast<Packed>(fp32x2_t{data[0], data[1]}));
    if (warp_id == kRopeWarp) {
      const auto x_real = rounded.x;
      const auto x_imag = rounded.y;
      const auto freq_real = freq[0];
      const auto freq_imag = freq[1];
      rounded = cast<fp32x2_t>(
          cast<Packed>(fp32x2_t{x_real * freq_real - x_imag * freq_imag, x_real * freq_imag + x_imag * freq_real}));
    }
    const float v[2] = {rounded.x, rounded.y};
    return deepseek_v4::v41::store_row<kLayout>(row.data, row.scale, tx, v);
  }

  const auto value_ptr = row.data;

  PDLTriggerSecondary<kUsePDL>();

  // part 2: rope on warp 7 (BF16 store), per-warp UE8M0 quant + store on warps 0..6.
  if (warp_id == kRopeWarp) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto freq_real = freq[0];
    const auto freq_imag = freq[1];
    data[0] = x_real * freq_real - x_imag * freq_imag;
    data[1] = x_real * freq_imag + x_imag * freq_real;
    const auto result = cast<bf16x2_t>(fp32x2_t{data[0], data[1]});
    const auto rope_ptr = value_ptr + 448;
    reinterpret_cast<bf16x2_t*>(rope_ptr)[lane_id] = result;
  } else {
    const auto x = data[0];
    const auto y = data[1];
    const auto abs_max = warp::reduce_max(fmaxf(fabs(x), fabs(y)));
    const auto scale_raw = fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX;
    const auto scale_ue8m0 = cast_to_ue8m0(scale_raw);
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
    const auto result = pack_fp8(x * inv_scale, y * inv_scale);
    const auto scale_ptr = row.scale;
    reinterpret_cast<fp8x2_e4m3_t*>(value_ptr)[tx] = result;
    if (lane_id == 0) static_cast<uint8_t*>(scale_ptr)[warp_id] = scale_ue8m0;
  }
}


}
extern "C" int ca_kv_abi() { return 1; }
extern "C" int ca_kv_store(int rows, int page_tokens, int64_t kv_stride,
 const void* kv, const void* weight, const float* freqs, const int32_t* positions,
 const int32_t* locations, uint8_t* cache, float eps, cudaStream_t stream) {
 if (rows < 1 || rows > 65536 || kv_stride < 512 || !(eps > 0)) return -1;
 sglang::FusedKNormRopeFlashMLAParams p{kv,weight,freqs,positions,locations,cache,kv_stride,(uint32_t)rows,eps};
#define CA_STORE(BITS) \
 sglang::fused_k_norm_rope_flashmla<__nv_bfloat16,512,64,int32_t,BITS,sglang::deepseek_v4::KVLayout::V4,false> \
   <<<rows,256,0,stream>>>(p)
 if (page_tokens == 64) { CA_STORE(6); }
 else if (page_tokens == 256) { CA_STORE(8); }
 else return -2;
#undef CA_STORE
 return (int)cudaGetLastError();
}
