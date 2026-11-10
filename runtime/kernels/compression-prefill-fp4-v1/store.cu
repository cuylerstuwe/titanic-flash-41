


#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp4_utils.cuh>
#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>
#include <sgl_kernel/deepseek_v4/kv_layout.cuh>



#include <bit>
#include <cstdint>

namespace sglang {

/// \brief Ratio-1 decode compressor: RMSNorm and the whole main-KV write.
///
/// The ratio-1 compressor pools nothing -- `project` is a single bf16 `wkv`
/// GEMM and the latent it produces stands for the token itself -- so the
/// kernel's input is the GEMM output and its RoPE position is `positions`, not
/// `positions - 1`. `kv_output` is the pre-RoPE latent, for the index-K
/// branch's `wk` projection.
struct Compress1DecodeParams {
  const bf16_t* __restrict__ kv_input;     // [num_tokens, kHeadDim] bf16
  bf16_t* __restrict__ kv_output;          // [num_tokens, kHeadDim] bf16, pre-RoPE
  const bf16_t* __restrict__ norm_weight;  // [kHeadDim] bf16
  const float* __restrict__ freqs_cis;     // [max_pos, kRopeDim] fp32, real/imag interleaved
  const void* __restrict__ positions;      // [num_tokens] PosT
  const void* __restrict__ out_loc;        // [num_tokens] LocT compressed slot; 0 marks a padded row
  uint8_t* __restrict__ kvcache;           // [npages, kPageBytes] uint8
  float eps;
};

/// Elements per thread; 256 threads per token was measured fastest on B200 decode batches.
/// At head_dim 512, (512 - 64) / 2 = 224 threads keeps the nope/rope split warp-aligned;
/// the fp8 amax reduction requires every lane in its full-warp mask to participate.
constexpr uint32_t kC1VecSize = 2;

/// \brief RMSNorm + RoPE tail + fp4 fake-quant + the FlashMLA store.
///
/// One CTA per token, `kHeadDim / kC1VecSize` threads over the row.
///
/// The three reductions have three different widths and are not
/// interchangeable: the RMSNorm statistic spans the row, an fp8 store scale
/// spans 64 elements, an fp4 block spans 16. All asserted below.
///
/// kLayout is the cache's page format. V4 (584 B/token) and V41 (528 B/token,
/// fp8 with per-32 scales) store the fake-quantized value; V41_FP4 (288 B/token)
/// stores the e2m1 codes and their e4m3 scales directly, so the fp4 rounding
/// happens once and no fp8 rounding follows it.
template <
    int64_t kHeadDim,
    int64_t kRopeDim,
    int32_t kPageBits,
    typename PosT,
    typename LocT,
    deepseek_v4::KVLayout kLayout,
    bool kUsePDL>
__global__ __launch_bounds__(kHeadDim / kC1VecSize) void flash_c1_decode_kernel(
    const __grid_constant__ Compress1DecodeParams params) {
  using namespace device;
  using deepseek_v4::KVLayout;
  using deepseek_v4::fp8::cast_to_ue8m0;
  using deepseek_v4::fp8::inv_scale_ue8m0;
  using deepseek_v4::fp8::pack_fp8;

  /// Threads over one token, and the leading ones of those that carry the fp8
  /// nope part; the rest carry the bf16 RoPE tail.
  constexpr uint32_t kVecSize = kC1VecSize;
  constexpr uint32_t kRowLanes = kHeadDim / kVecSize;
  constexpr uint32_t kNopeLanes = (kHeadDim - kRopeDim) / kVecSize;
  constexpr uint32_t kRowWarps = kRowLanes / kWarpThreads;
  constexpr uint32_t kFp8Lanes = 64 / kVecSize;
  constexpr uint32_t kFp4Lanes = deepseek_v4::fp4::kCompressedKVBlockSize / kVecSize;
  using Paged = deepseek_v4::PagedKV<kLayout, kPageBits>;
  static_assert(kHeadDim == 512 && kRopeDim == 64, "the FlashMLA layouts require (512, 64)");
  static_assert(kHeadDim % kVecSize == 0 && kVecSize % 2 == 0);
  static_assert(kRowLanes % kWarpThreads == 0, "a token owns a whole number of warps");
  static_assert(kNopeLanes % kFp8Lanes == 0, "the nope part must end on an fp8 scale block");
  static_assert(
      (kHeadDim - kRopeDim) % deepseek_v4::fp4::kCompressedKVBlockSize == 0,
      "no fp4 block may straddle the nope/rope seam");
  static_assert(kFp8Lanes <= kWarpThreads && kFp4Lanes <= kWarpThreads);

  using bf16_vec_t = AlignedVector<bf16x2_t, kVecSize / 2>;
  using fp8_vec_t = AlignedVector<fp8x2_e4m3_t, kVecSize / 2>;
  using freq_vec_t = AlignedVector<float, kVecSize>;

  const uint32_t tx = threadIdx.x;
  const uint32_t row = blockIdx.x;

  // `out_loc` and `positions` are step metadata, independent of the PDL producer.
  // Slots fit in int32; padded rows are suppressed at the cache store.
  const auto out_loc = static_cast<int32_t>(static_cast<const LocT*>(params.out_loc)[row]);
  const auto position = static_cast<int64_t>(static_cast<const PosT*>(params.positions)[row]);
  PDLWaitPrimary<kUsePDL>();

  float data[kVecSize];
  bf16_vec_t latent;
  latent.load(params.kv_input + row * kHeadDim, tx);
  PDLTriggerSecondary<kUsePDL>();

  // Match finish()'s bf16 rounding before the main-KV RoPE.
#pragma unroll
  for (uint32_t j = 0; j < kVecSize / 2; ++j) {
    const auto [x, y] = cast<fp32x2_t>(latent[j]);
    data[j * 2 + 0] = x;
    data[j * 2 + 1] = y;
  }

  if (tx >= kNopeLanes) {
    // `rope_tail` ends in `.to(x.dtype)`, so the rotated value is rounded back
    // to bf16 before the fake-quant widens it again.
    freq_vec_t freq;
    freq.load(params.freqs_cis + position * kRopeDim, tx - kNopeLanes);
#pragma unroll
    for (uint32_t j = 0; j < kVecSize / 2; ++j) {
      const auto k = j * 2;
      const auto x_real = data[k + 0];
      const auto x_imag = data[k + 1];
      const auto f_real = freq[k + 0];
      const auto f_imag = freq[k + 1];
      const auto rotated =
          cast<bf16x2_t>(fp32x2_t{x_real * f_real - x_imag * f_imag, x_real * f_imag + x_imag * f_real});
      const auto [r0, r1] = cast<fp32x2_t>(rotated);
      data[k + 0] = r0;
      data[k + 1] = r1;
    }
  }

  if constexpr (kLayout == KVLayout::V41_FP4) {
    // The fp4 cache takes the rotated bf16 value as is: its row quantizer is the
    // fake quantization, minus the dequantization.
    if (out_loc <= 0) return;
    const auto kv_row = Paged::row(params.kvcache, out_loc);
    return deepseek_v4::v41::store_row<kLayout>(kv_row.data, kv_row.scale, tx, data);
  }

  // FP4/E4M3 fake-quant over 16 elements, i.e. kFp4Lanes threads.
  {
    float amax = fabsf(data[0]);
#pragma unroll
    for (uint32_t i = 1; i < kVecSize; ++i) {
      amax = fmaxf(amax, fabsf(data[i]));
    }
    amax = warp::reduce_max<kFp4Lanes>(amax);
    const auto scale = deepseek_v4::fp4::compressed_kv_scale(amax);
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      const auto [x, y] = deepseek_v4::fp4::fake_quant_compressed_kv_x2({data[i * 2 + 0], data[i * 2 + 1]}, scale);
      data[i * 2 + 0] = x;
      data[i * 2 + 1] = y;
    }
  }

  // A padded CUDA-graph row carries `out_loc == 0`, the reserved dummy slot,
  // and must publish nothing: at ratio 1 the compressed slot *is* the FULL
  // slot, so there is nothing to divide and no other marker to read.
  if (out_loc <= 0) return;
  const auto kv_row = Paged::row(params.kvcache, out_loc);

  if constexpr (kLayout == KVLayout::V41) {
    // fp8 with one ue8m0 scale per 32 elements over the whole row, RoPE included.
    return deepseek_v4::v41::store_row<kLayout>(kv_row.data, kv_row.scale, tx, data);
  }

  const auto value_ptr = kv_row.data;

  if (tx >= kNopeLanes) {
    bf16_vec_t rope_out;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize / 2; ++j) {
      rope_out[j] = cast<bf16x2_t>(fp32x2_t{data[j * 2 + 0], data[j * 2 + 1]});
    }
    rope_out.store(value_ptr + (kHeadDim - kRopeDim), tx - kNopeLanes);
  } else {
    // fp8 e4m3 with one ue8m0 scale per 64 elements.
    float abs_max = fabsf(data[0]);
#pragma unroll
    for (uint32_t i = 1; i < kVecSize; ++i) {
      abs_max = fmaxf(abs_max, fabsf(data[i]));
    }
    abs_max = warp::reduce_max<kFp8Lanes>(abs_max);
    const auto scale_ue8m0 = cast_to_ue8m0(fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX);
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
    fp8_vec_t nope_out;
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      nope_out[i] = pack_fp8(data[i * 2 + 0] * inv_scale, data[i * 2 + 1] * inv_scale);
    }
    nope_out.store(value_ptr, tx);
    kv_row.scale[tx / kFp8Lanes] = scale_ue8m0;
  }
}

/// \brief Host side of `flash_c1_decode_kernel`.

} // namespace sglang
using namespace sglang;
template<int PBS> int store_launch(int rows,const void* input,const int32_t* pos,const int64_t* loc,
 const float* freq,uint8_t* cache,cudaStream_t stream){
 Compress1DecodeParams p{(const bf16_t*)input,nullptr,nullptr,freq,pos,loc,cache,0.f};
 constexpr int bits=std::bit_width((unsigned)PBS)-1;
 auto kernel=flash_c1_decode_kernel<512,64,bits,int32_t,int64_t,deepseek_v4::KVLayout::V41_FP4,false>;
 void* args[]={&p};return (int)cudaLaunchKernel((void*)kernel,dim3(rows),dim3(256),args,0,stream);
}
extern "C" int ca_prefill_store_abi(){return 2;}
extern "C" int ca_prefill_store(int rows,int page,const void* input,const int32_t* pos,const int64_t* loc,
 const float* freq,uint8_t* cache,cudaStream_t stream){
 if(rows<1||rows>4096)return -1;
 #define STORE(P) case P:return store_launch<P>(rows,input,pos,loc,freq,cache,stream);
 switch(page){STORE(32) STORE(64) STORE(128) STORE(256) default:return -2;}
 #undef STORE
}
