// Selected SGLang CUDA primitives, Apache-2.0; provenance in native.json.



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
  {
    bf16_vec_t input, weight;
    input.load(params.kv_input + row * kHeadDim, tx);
    weight.load(params.norm_weight, tx);

    // `project` already returns bf16 at ratio 1, so `finish`'s `.to(bfloat16)`
    // is a no-op and the statistic is taken over the loaded values as they are.
    float local_sqrsum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize / 2; ++j) {
      const auto [x, y] = cast<fp32x2_t>(input[j]);
      local_sqrsum += x * x;
      local_sqrsum += y * y;
      data[j * 2 + 0] = x;
      data[j * 2 + 1] = y;
    }

    __shared__ float s_warp_sum[kRowWarps];
    s_warp_sum[tx / kWarpThreads] = warp::reduce_sum(local_sqrsum);
    __syncthreads();
    float sqrsum = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kRowWarps; ++i) {
      sqrsum += s_warp_sum[i];
    }
    constexpr float kInvHeadDim = 1.0f / static_cast<float>(kHeadDim);
    const auto norm_factor = math::rsqrt(sqrsum * kInvHeadDim + params.eps);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize / 2; ++j) {
      const auto [wx, wy] = cast<fp32x2_t>(weight[j]);
      const auto x = data[j * 2 + 0] * norm_factor * wx;
      const auto y = data[j * 2 + 1] * norm_factor * wy;
      latent[j] = cast<bf16x2_t>(fp32x2_t{x, y});
    }
  }

  latent.store(params.kv_output + row * kHeadDim, tx);
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

/// \brief Ratio-2 decode compressor: pair-pool, RMSNorm and the main-KV write.
///
/// `kv_input` and `kv_state` rows are `2 * kHeadDim` floats, kv then score.
/// `kv_output` is the pre-RoPE latent, for the index-K branch's `wk`.
struct Compress2DecodeParams {
  const float* __restrict__ kv_input;  // [num_tokens, 2 * kHeadDim] fp32
  /// `CompressStatePool`'s flat `KVAndScore` buffer, `[size, 2 * kHeadDim]`
  /// fp32 with kv in the low half and score in the high half. A request's
  /// pending pair lives at `req * ring_size + pos % ring_size`.
  float* __restrict__ kv_state;
  bf16_t* __restrict__ kv_output;          // [num_tokens, kHeadDim] bf16, pre-RoPE
  const bf16_t* __restrict__ norm_weight;  // [kHeadDim] bf16
  const float* __restrict__ freqs_cis;     // [max_pos, kRopeDim] fp32, real/imag interleaved
  const void* __restrict__ positions;      // [num_tokens] PosT
  const int64_t* __restrict__ req;         // [num_tokens], req_pool_idx per token
  const void* __restrict__ raw_out_loc;    // [num_tokens] LocT, the FULL slot; 0 marks a padded row
  uint8_t* __restrict__ kvcache;           // [npages, kPageBytes] uint8
  /// Positions per request slot in the pair-state ring.
  uint32_t ring_size;
  float eps;
};

/// Elements per thread; 256 threads per token was measured fastest on B200 decode batches.
/// The launcher and launch bounds share this value. At head_dim 512, the nope/rope split
/// is (512 - 64) / 2 = 224 threads, keeping the fp8 amax reduction's full-warp mask valid.
constexpr uint32_t kC2VecSize = 2;

/// \brief grid = num_tokens, block = kHeadDim / kC2VecSize.
///
/// An odd position completes a group with its even predecessor; an even one
/// parks itself in the state.
///
/// Target-verify runs that same schedule with `draft_len` consecutive positions
/// per request instead of one, so a row's partner is usually the row before it
/// in `kv_input` rather than the ring. That is the whole difference, and a 2D
/// grid answers it without arithmetic: `blockIdx.x` is the position inside the
/// block, `blockIdx.y` the request.
///
/// The three reductions below have three different widths and are not
/// interchangeable: the RMSNorm statistic spans the row, an fp8 store scale
/// spans 64 elements, an fp4 block spans 16. All asserted.
///
/// kLayout is the cache's page format. V4 (584 B/token) and V41 (528 B/token,
/// fp8 with per-32 scales) store the fake-quantized value; V41_FP4 (288 B/token)
/// stores the e2m1 codes and their e4m3 scales directly, so the fp4 rounding
/// happens once and no fp8 rounding follows it.
template <
    bool kVerify,
    int64_t kHeadDim,
    int64_t kRopeDim,
    int32_t kPageBits,
    typename PosT,
    typename LocT,
    deepseek_v4::KVLayout kLayout,
    bool kUsePDL>
__global__ __launch_bounds__(kHeadDim / kC2VecSize) void flash_c2_decode_kernel(const Compress2DecodeParams params) {
  using namespace device;
  using deepseek_v4::KVLayout;
  using deepseek_v4::fp8::cast_to_ue8m0;
  using deepseek_v4::fp8::inv_scale_ue8m0;
  using deepseek_v4::fp8::pack_fp8;

  constexpr uint32_t kVecSize = kC2VecSize;
  constexpr uint32_t kCTASize = kHeadDim / kVecSize;
  constexpr int64_t kStride = kHeadDim * 2;
  /// Threads covering the fp8 nope part; the rest carry the bf16 RoPE tail.
  constexpr uint32_t kNopeThreads = (kHeadDim - kRopeDim) / kVecSize;
  constexpr uint32_t kFp8Lanes = 64 / kVecSize;
  constexpr uint32_t kFp4Lanes = deepseek_v4::fp4::kCompressedKVBlockSize / kVecSize;
  using Paged = deepseek_v4::PagedKV<kLayout, kPageBits>;

  static_assert(kHeadDim == (kVecSize * kCTASize));
  static_assert(kCTASize % kWarpThreads == 0);
  static_assert(kNopeThreads % kFp8Lanes == 0, "the nope part must end on an fp8 scale block");
  static_assert(kWarpThreads % kFp8Lanes == 0 && kWarpThreads % kFp4Lanes == 0);
  static_assert(kHeadDim == 512 && kRopeDim == 64, "the FlashMLA layouts require (512, 64)");
  using fp32_vec_t = AlignedVector<float, kVecSize>;
  using bf16_vec_t = AlignedVector<bf16x2_t, kVecSize / 2>;

  const auto tx = threadIdx.x;
  // Verify gives each request a CTA column; decode a flat grid of one row each.
  const auto row = kVerify ? blockIdx.y * gridDim.x + blockIdx.x : blockIdx.x;
  PDLWaitPrimary<kUsePDL>();
  // Slots fit in int32 whatever width the scheduler hands them in.
  const auto raw_out_loc = static_cast<int32_t>(static_cast<const LocT*>(params.raw_out_loc)[row]);
  // CUDA graph padding is a completely inert row: do not read schedule, input,
  // state, or RoPE data, and do not publish output or update the cache.
  if (raw_out_loc == 0) return PDLTriggerSecondary<kUsePDL>();
  const auto pos = static_cast<const PosT*>(params.positions)[row];
  // A completing row reads the slot left by `pos - 1`;
  // a pending row writes its own slot, so reads and writes stay disjoint.
  const auto rid = params.req[row];

  fp32_vec_t kv_new, score_new;
  kv_new.load(params.kv_input + row * kStride, tx);
  score_new.load(params.kv_input + row * kStride, tx + kCTASize);

  const auto ring = static_cast<int64_t>(rid) * params.ring_size;
  const auto read_row = ring + (pos - 1 + params.ring_size) % params.ring_size;
  const auto write_row = ring + pos % params.ring_size;

  fp32_vec_t kv_old, score_old;
  const float* partner = params.kv_state + read_row * kStride;
  if constexpr (kVerify) {
    if (blockIdx.x != 0) partner = params.kv_input + static_cast<int64_t>(row - 1) * kStride;
  }
  kv_old.load(partner, tx);
  score_old.load(partner, tx + kCTASize);

  if ((pos & 1) == 0) {
    kv_new.store(params.kv_state + write_row * kStride, tx);
    score_new.store(params.kv_state + write_row * kStride, tx + kCTASize);
    return PDLTriggerSecondary<kUsePDL>();
  }

  constexpr uint32_t kNumWarps = kCTASize / kWarpThreads;
  __shared__ float s_warp_sum[kNumWarps];
  fp32_vec_t staged, freq;
  bf16_vec_t weight, out;
  weight.load(params.norm_weight, tx);
  if (tx >= kNopeThreads) freq.load(params.freqs_cis + (pos - 1) * kRopeDim, tx - kNopeThreads);

  // With two scores `exp(-|s0 - s1|)` is the whole softmax: one exp, argument
  // always <= 0, so no max-subtraction pass and no overflow.
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    const auto delta = score_old[i] - score_new[i];
    const auto scale = expf(-fabsf(delta));
    const auto scale_0 = delta > 0 ? 1.0f : scale;
    const auto scale_1 = delta > 0 ? scale : 1.0f;
    staged[i] = (kv_old[i] * scale_0 + kv_new[i] * scale_1) / (1.0f + scale);
  }

  // `finish` casts to bf16 before the norm, so the sum of squares has to see
  // the rounded values.
  float local_sqrsum = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kVecSize / 2; ++i) {
    const auto packed = fp32x2_t{staged[i * 2 + 0], staged[i * 2 + 1]};
    const auto [x, y] = cast<fp32x2_t>(cast<bf16x2_t>(packed));
    local_sqrsum += x * x;
    local_sqrsum += y * y;
    staged[i * 2 + 0] = x;
    staged[i * 2 + 1] = y;
  }
  const auto warp_sum = warp::reduce_sum(local_sqrsum);
  s_warp_sum[tx / kWarpThreads] = warp_sum;
  __syncthreads();

  float sqrsum = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kNumWarps; ++i) {
    sqrsum += s_warp_sum[i];
  }
  constexpr float kInvScale = 1.0f / static_cast<float>(kHeadDim);
  const auto norm_factor = math::rsqrt(sqrsum * kInvScale + params.eps);

#pragma unroll
  for (uint32_t i = 0; i < kVecSize / 2; ++i) {
    const auto [wx, wy] = cast<fp32x2_t>(weight[i]);
    const auto x = staged[i * 2 + 0] * norm_factor * wx;
    const auto y = staged[i * 2 + 1] * norm_factor * wy;
    out[i] = cast<bf16x2_t>(fp32x2_t{x, y});
  }
  // The pre-RoPE latent, for the index-K branch's `wk` projection. Published
  // before the trigger because that GEMM is the successor that reads it.
  out.store(params.kv_output, static_cast<int64_t>(row) * kCTASize + tx);
  PDLTriggerSecondary<kUsePDL>();

  // ---- main-KV branch: RoPE tail, fp4 fake-quant, 584-byte store ----
  // Match finish()'s bf16 rounding before RoPE.
#pragma unroll
  for (uint32_t i = 0; i < kVecSize / 2; ++i) {
    const auto [x, y] = cast<fp32x2_t>(out[i]);
    staged[i * 2 + 0] = x;
    staged[i * 2 + 1] = y;
  }

  if (tx >= kNopeThreads) {
    // Match rope_tail()'s bf16 rounding before fake quantization.
    // Only odd positions reach here; the latent represents `pos - 1`.
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      const auto x_real = staged[i * 2 + 0];
      const auto x_imag = staged[i * 2 + 1];
      const auto f_real = x_real * freq[i * 2 + 0] - x_imag * freq[i * 2 + 1];
      const auto f_imag = x_real * freq[i * 2 + 1] + x_imag * freq[i * 2 + 0];
      const auto rotated = cast<bf16x2_t>(fp32x2_t{f_real, f_imag});
      const auto [r0, r1] = cast<fp32x2_t>(rotated);
      staged[i * 2 + 0] = r0;
      staged[i * 2 + 1] = r1;
    }
  }

  if constexpr (kLayout == KVLayout::V41_FP4) {
    // The fp4 cache takes the rotated bf16 value as is: its row quantizer is the
    // fake quantization, minus the dequantization.
    const int32_t out_loc = raw_out_loc >> 1;
    const auto kv_row = Paged::row(params.kvcache, out_loc);
    return deepseek_v4::v41::store_row<kLayout>(kv_row.data, kv_row.scale, tx, staged);
  }

  // FP4/E4M3 fake-quant over 16 elements, i.e. kFp4Lanes threads.
  {
    float amax = fabsf(staged[0]);
#pragma unroll
    for (uint32_t i = 1; i < kVecSize; ++i) {
      amax = fmaxf(amax, fabsf(staged[i]));
    }
    amax = warp::reduce_max<kFp4Lanes>(amax);
    const auto scale = deepseek_v4::fp4::compressed_kv_scale(amax);
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      const auto [x, y] = deepseek_v4::fp4::fake_quant_compressed_kv_x2({staged[i * 2 + 0], staged[i * 2 + 1]}, scale);
      staged[i * 2 + 0] = x;
      staged[i * 2 + 1] = y;
    }
  }

  // `raw_out_loc / ratio`; ratio 2 makes it a shift.
  const int32_t out_loc = raw_out_loc >> 1;
  const auto kv_row = Paged::row(params.kvcache, out_loc);

  if constexpr (kLayout == KVLayout::V41) {
    // fp8 with one ue8m0 scale per 32 elements over the whole row, RoPE included.
    return deepseek_v4::v41::store_row<kLayout>(kv_row.data, kv_row.scale, tx, staged);
  }

  const auto value_ptr = kv_row.data;

  if (tx >= kNopeThreads) {
    bf16_vec_t rope_out;
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      rope_out[i] = cast<bf16x2_t>(fp32x2_t{staged[i * 2 + 0], staged[i * 2 + 1]});
    }
    rope_out.store(value_ptr + (kHeadDim - kRopeDim), tx - kNopeThreads);
  } else {
    // fp8 e4m3 with one ue8m0 scale per 64 elements.
    auto abs_max = fabsf(staged[0]);
#pragma unroll
    for (uint32_t i = 1; i < kVecSize; ++i) {
      abs_max = fmaxf(abs_max, fabsf(staged[i]));
    }
    abs_max = warp::reduce_max<kFp8Lanes>(abs_max);
    const auto scale_ue8m0 = cast_to_ue8m0(fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX);
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      reinterpret_cast<fp8x2_e4m3_t*>(value_ptr)[tx * (kVecSize / 2) + i] =
          pack_fp8(staged[i * 2 + 0] * inv_scale, staged[i * 2 + 1] * inv_scale);
    }
    kv_row.scale[tx / kFp8Lanes] = scale_ue8m0;
  }
}


} // namespace sglang

using namespace sglang;
template<int PBS> int c1_launch(int rows,const void* input,const void* weight,const int32_t* pos,
 const int64_t* loc,const float* freq,uint8_t* cache,void* latent,cudaStream_t stream){
 Compress1DecodeParams p{(const bf16_t*)input,(bf16_t*)latent,(const bf16_t*)weight,freq,pos,loc,cache,1e-20f};
 constexpr int bits=std::bit_width((unsigned)PBS)-1;
 auto kernel=flash_c1_decode_kernel<512,64,bits,int32_t,int64_t,deepseek_v4::KVLayout::V41_FP4,false>;
 void* args[]={&p};return (int)cudaLaunchKernel((void*)kernel,dim3(rows),dim3(256),args,0,stream);
}
template<int PBS,bool VERIFY> int c2_launch(int rows,int ring,int draft,const float* input,float* state,
 const void* weight,const int32_t* pos,const int64_t* req,const int64_t* loc,
 const float* freq,uint8_t* cache,void* latent,cudaStream_t stream){
 Compress2DecodeParams p{input,state,(bf16_t*)latent,(const bf16_t*)weight,freq,pos,req,loc,cache,(uint32_t)ring,1e-20f};
 constexpr int bits=std::bit_width((unsigned)PBS)-1;
 auto kernel=flash_c2_decode_kernel<VERIFY,512,64,bits,int32_t,int64_t,deepseek_v4::KVLayout::V41_FP4,false>;
 void* args[]={&p};dim3 grid=VERIFY?dim3(draft,rows/draft):dim3(rows);
 return (int)cudaLaunchKernel((void*)kernel,grid,dim3(256),args,0,stream);
}
extern "C" int ca_compression_abi(){return 2;}
extern "C" int ca_compress1(int rows,int page,const void* input,const void* weight,const int32_t* pos,
 const int64_t* loc,const float* freq,uint8_t* cache,void* latent,cudaStream_t stream){
 if(rows<1||rows>4096)return -1;
 #define C1(P) case P:return c1_launch<P>(rows,input,weight,pos,loc,freq,cache,latent,stream);
 switch(page){C1(32) C1(64) C1(128) C1(256) default:return -2;}
 #undef C1
}
extern "C" int ca_compress2(int rows,int page,int ring,int draft,const float* input,float* state,
 const void* weight,const int32_t* pos,const int64_t* req,const int64_t* loc,
 const float* freq,uint8_t* cache,void* latent,cudaStream_t stream){
 if(rows<1||rows>4096||ring<2||ring>64||draft<1||draft>16||rows%draft||(draft>1&&ring<=draft))return -1;
 #define C2(P) case P:return draft>1?c2_launch<P,true>(rows,ring,draft,input,state,weight,pos,req,loc,freq,cache,latent,stream):c2_launch<P,false>(rows,ring,draft,input,state,weight,pos,req,loc,freq,cache,latent,stream);
 switch(page){C2(32) C2(64) C2(128) C2(256) default:return -2;}
 #undef C2
}
