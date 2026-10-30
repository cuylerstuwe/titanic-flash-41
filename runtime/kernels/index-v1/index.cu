// Selected SGLang device primitives, Apache-2.0; provenance in native.json
/// \file small_gemm_bf16.cuh
/// \brief Small bf16 GEMMs for two fixed shapes, `out[m, n] = sum_k a[m, k] * b[n, k]`
/// with N = 128, K = 512 (one warp per output column) and N = 32, K = 5120 (one CTA
/// per output column).
///
/// The thread-to-K mapping and reduction order are those of tiny_gemm's N-variant,
/// so results are row-invariant across M and bitwise equal to tiny_gemm where both
/// apply; cuBLAS uses a different reduction order.




#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/gemm/dot_product.cuh>



#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace sglang {

struct SmallGemmParams {
  bf16_t* __restrict__ out;
  const bf16_t* __restrict__ a;
  const bf16_t* __restrict__ b;
  int64_t stride_a;  // row stride of `a`, in elements
  uint32_t m;        // batch size
};

template <uint32_t M_SPLIT_>
struct N128K512Trait {
  static constexpr uint32_t N = 128;
  static constexpr uint32_t K = 512;
  static constexpr uint32_t kMaxMSplit = 8;
  static constexpr uint32_t M_SPLIT = M_SPLIT_;  // rows of `a` one warp handles
  static_assert(M_SPLIT >= 1 && M_SPLIT <= kMaxMSplit, "M_SPLIT must be in [1, 8]");
  static constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(bf16_t);
  static constexpr uint32_t kNumVecs = K / (kVecSize * device::kWarpThreads);
  static_assert(K % (kVecSize * device::kWarpThreads) == 0, "K must be a whole number of warp-wide vectors");
  using vec_t = device::AlignedVector<bf16x2_t, kVecSize / 2>;
  static constexpr uint32_t kBlockSize = 128;
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  static_assert(N % kNumWarps == 0, "every warp of a block owns one output column");
  static constexpr uint32_t kGridX = N / kNumWarps;
};

template <typename T, bool kUsePDL>
__global__ __launch_bounds__(T::kBlockSize, 1) void n128k512_kernel(const SmallGemmParams params) {
  using namespace device;
  constexpr uint32_t kNumVecs = T::kNumVecs;
  constexpr uint32_t M_SPLIT = T::M_SPLIT;
  constexpr uint32_t K = T::K;
  constexpr uint32_t N = T::N;
  using vec_t = typename T::vec_t;

  const uint32_t M = params.m;
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t n = blockIdx.x * T::kNumWarps + warp_id;
  const uint32_t m_start = blockIdx.y * M_SPLIT;
  const auto gmem = tile::Memory<vec_t>::warp();

  // The weight row does not depend on the previous kernel: load it before the PDL wait.
  vec_t b[kNumVecs];
#pragma unroll
  for (uint32_t j = 0; j < kNumVecs; ++j) {
    b[j] = gmem.load(params.b + n * K, j);
  }

  PDLWaitPrimary<kUsePDL>();

  // Clamp padded loads to the last valid row to keep vector loads branch-free;
  // only stores are guarded.
  vec_t a[M_SPLIT][kNumVecs];
#pragma unroll
  for (uint32_t i = 0; i < M_SPLIT; ++i) {
    const uint32_t m = min(m_start + i, M - 1);
#pragma unroll
    for (uint32_t j = 0; j < kNumVecs; ++j) {
      a[i][j] = gmem.load(params.a + m * params.stride_a, j);
    }
  }

#pragma unroll
  for (uint32_t i = 0; i < M_SPLIT; ++i) {
    float acc = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kNumVecs; ++j) {
      dot_product_vec(a[i][j], b[j], acc);
    }
    acc = warp::reduce_sum(acc);
    const uint32_t m = m_start + i;
    if (m < M) {
      params.out[m * N + n] = cast<bf16_t>(acc);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <uint32_t M_SPLIT_>
struct N32K5120Trait {
  static constexpr uint32_t N = 32;
  static constexpr uint32_t K = 5120;
  static constexpr uint32_t kMaxMSplit = 8;
  static constexpr uint32_t M_SPLIT = M_SPLIT_;  // rows of `a` one CTA handles
  static_assert(M_SPLIT >= 1 && M_SPLIT <= kMaxMSplit, "M_SPLIT must be in [1, 8]");
  static constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(bf16_t);
  // Ten warps preserve tiny_gemm's mapping: thread t owns [16t, 16t + 16).
  static constexpr uint32_t kBlockSize = 320;
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  static constexpr uint32_t kNumVecs = K / (kVecSize * kBlockSize);  // vectors per thread
  static_assert(K % (kVecSize * kBlockSize) == 0, "K must be a whole number of CTA-wide vectors");
  static_assert(kBlockSize % device::kWarpThreads == 0, "the reduction needs whole warps");
  static_assert(M_SPLIT <= kBlockSize, "one thread finishes one row");
  using vec_t = device::AlignedVector<bf16x2_t, kVecSize / 2>;
  static constexpr uint32_t kGridX = N;  // one CTA per output column
};

template <typename T, bool kUsePDL>
__global__ __launch_bounds__(T::kBlockSize, 1) void n32k5120_kernel(const SmallGemmParams params) {
  using namespace device;
  constexpr uint32_t kNumVecs = T::kNumVecs;
  constexpr uint32_t kNumWarps = T::kNumWarps;
  constexpr uint32_t M_SPLIT = T::M_SPLIT;
  constexpr uint32_t K = T::K;
  constexpr uint32_t N = T::N;
  using vec_t = typename T::vec_t;

  const uint32_t M = params.m;
  const uint32_t tx = threadIdx.x;
  const uint32_t warp_id = tx / kWarpThreads;
  const uint32_t n = blockIdx.x;
  const uint32_t m_start = blockIdx.y * M_SPLIT;
  const auto gmem = tile::Memory<vec_t>::cta(T::kBlockSize);

  // The weight column does not depend on the previous kernel: load it before the PDL wait.
  vec_t b[kNumVecs];
#pragma unroll
  for (uint32_t j = 0; j < kNumVecs; ++j) {
    b[j] = gmem.load(params.b + n * K, j);
  }

  PDLWaitPrimary<kUsePDL>();

  // Clamp padded loads to the last valid row to keep vector loads branch-free;
  // only stores are guarded.
  vec_t a[M_SPLIT][kNumVecs];
#pragma unroll
  for (uint32_t i = 0; i < M_SPLIT; ++i) {
    const uint32_t m = min(m_start + i, M - 1);
#pragma unroll
    for (uint32_t j = 0; j < kNumVecs; ++j) {
      a[i][j] = gmem.load(params.a + m * params.stride_a, j);
    }
  }

  __shared__ float s_acc[kNumWarps][M_SPLIT];
#pragma unroll
  for (uint32_t i = 0; i < M_SPLIT; ++i) {
    float acc = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kNumVecs; ++j) {
      dot_product_vec(a[i][j], b[j], acc);
    }
    acc = warp::reduce_sum(acc);
    if (tx % kWarpThreads == 0) s_acc[warp_id][i] = acc;
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();

  // One thread per row sums the warps in a fixed order, so a row's result does
  // not depend on which rows share its CTA.
  if (tx < M_SPLIT) {
    float acc = s_acc[0][tx];
#pragma unroll
    for (uint32_t w = 1; w < kNumWarps; ++w) {
      acc += s_acc[w][tx];
    }
    const uint32_t m = m_start + tx;
    if (m < M) {
      params.out[m * N + n] = cast<bf16_t>(acc);
    }
  }
}


} // namespace sglang



#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp4_utils.cuh>



#include <bit>
#include <cstdint>
#include <optional>

namespace sglang {

/// \brief RMSNorm, RoPE, two FP4 stages and a 68-byte index-K cache store.
///
/// `input` is `wk(latent)`, before `k_norm`. A ratio-r group's latent uses
/// its first position, `positions & ~(r - 1)`, for power-of-two ratios.
struct IndexKParams {
  const bf16_t* __restrict__ input;        // [num_tokens, kHeadDim] bf16, pre-norm
  const bf16_t* __restrict__ norm_weight;  // [kHeadDim] bf16
  const float* __restrict__ freqs_cis;     // [max_pos, kRopeDim] fp32, real/imag interleaved
  const void* __restrict__ positions;      // [num_tokens] PosT
  const int64_t* __restrict__ loc;         // [num_tokens] index-K slot; 0 publishes nothing
  uint8_t* __restrict__ cache;             // [npages, kPageSize * 68] uint8
  uint32_t num_tokens;
  float eps;
};

/// \brief RoPE and two-stage FP4 packing for index-Q, after `wq_b`.
///
/// Each (token, head) row uses its token's own position, without RMSNorm
/// or a ratio mask. Payload and scales are contiguous outputs, not a paged cache.
struct IndexQParams {
  const bf16_t* __restrict__ input;     // [num_tokens, heads, kHeadDim] bf16
  const float* __restrict__ freqs_cis;  // [max_pos, kRopeDim] fp32, real/imag interleaved
  const void* __restrict__ positions;   // [num_tokens] PosT
  int8_t* __restrict__ payload;         // [num_tokens * heads, kHeadDim / 2] int8
  int32_t* __restrict__ scale;          // [num_tokens * heads] int32, four ue8m0 bytes
  // Optional head-weight epilogue (kWeights): the raw `weights_proj` output for
  // the same (token, head) rows, and where `float(bf16(w * weight_scale))` goes.
  const bf16_t* __restrict__ head_weights;  // [num_tokens * heads] bf16, or nullptr
  float* __restrict__ weights_out;          // [num_tokens * heads] fp32, or nullptr
  float weight_scale;
  uint32_t num_rows;
  uint32_t heads;
};

/// Warps per CTA; one warp owns one row. Chosen from B200 decode measurements,
/// where occupancy has little effect; multiple warps avoid starving larger batches.
constexpr uint32_t kFp4RopeWarpsPerCTA = 4;

/// \brief Indexer packer scale: `_ceil_ue8m0_exp(max(amax / 6, 1e-4))`.
///
/// Unlike fake quantization, the floor follows the divide, division is not
/// replaced by multiplication by 1/6, and the exponent is clamped.
/// The two stages therefore require separate scales.
SGL_DEVICE uint32_t index_pack_exponent(float amax) {
  const auto bits = __float_as_uint(fmaxf(amax / 6.0f, 1.0e-4f));
  const auto exponent = static_cast<int32_t>((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0);
  // Neither bound is reachable for finite fp32 inputs: the 1e-4 floor keeps
  // the exponent above 1, and reaching 254 requires absmax > 6 * 2^126.
  return static_cast<uint32_t>(min(max(exponent, 1), 254));
}

/// \brief Clear the sign of every packed nibble whose magnitude rounded to zero.
///
/// `cvt.rn.satfinite.e2m1x2.f32` keeps the sign of a small negative, giving the
/// `-0` code `0x8`; the reference packer drops it (`sign = (x < 0) & (idx != 0)`).
/// Both dequantize to zero, but the stored byte differs, so match the reference.
/// Branchless and independent of how many nibbles the word holds: bit 4k+3 of
/// each nibble survives only if one of bits 4k..4k+2 is set.
SGL_DEVICE uint32_t clear_negative_zero(uint32_t packed) {
  const auto any_magnitude = (packed | (packed >> 1) | (packed >> 2)) & 0x11111111u;
  return packed & ((any_magnitude << 3) | 0x77777777u);
}

/// One lane's share of a packed 128-element row: two payload bytes and the two
/// block exponents its half of the warp owns.
struct IndexPacked {
  uint32_t payload[2];   // the head pair's byte, then the tail pair's
  uint32_t exponent[2];  // blocks {0, 1} then {2, 3}, by half of the warp
};

/// \brief The whole shared body of both directions: RoPE tail, both fp4 stages,
///        and the indexer's pack.
///
/// `head` and `tail` are the lane's two bf16 pairs, widened and already rounded
/// to bf16 by whatever produced them (the caller's RMSNorm, or the load itself).
/// `tail` is pre-rotation and `freq` is its matching `(real, imag)`.
///
/// A 32-element fp4 block is 16 lanes of *one* half -- blocks 0/1 are the head
/// on lanes 0-15 / 16-31 and blocks 2/3 the tail -- so each quantization stage
/// costs two `reduce_max<16>`, not four, and nothing here spans the row.
SGL_DEVICE IndexPacked index_rope_quant_pack(fp32x2_t head, fp32x2_t tail, fp32x2_t freq) {
  using namespace device;
  namespace fp4 = deepseek_v4::fp4;

  constexpr uint32_t kHalfLanes = kWarpThreads / 2;
  static_assert(fp4::kBlockSize == kHalfLanes * 2, "an fp4 block must be half a warp of one half");

  float data[4];
  data[0] = head.x;
  data[1] = head.y;
  // `rope_tail` ends in `.to(x.dtype)`, so the rotated pair is rounded again.
  const auto rotated =
      cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{tail.x * freq.x - tail.y * freq.y, tail.x * freq.y + tail.y * freq.x}));
  data[2] = rotated.x;
  data[3] = rotated.y;

  // The fake-quant result is already exact in bf16: e2m1 needs at most three
  // significant bits, and the power-of-two scales admitted by the amax floor fit bf16.
#pragma unroll
  for (uint32_t half = 0; half < 2; ++half) {
    const auto amax = warp::reduce_max<kHalfLanes>(fmaxf(fabsf(data[half * 2]), fabsf(data[half * 2 + 1])));
    const auto [scale, inv_scale] = fp4::block_scale(amax);
    const auto q = fp4::fake_quant_x2({data[half * 2], data[half * 2 + 1]}, scale, inv_scale);
    data[half * 2 + 0] = q.x;
    data[half * 2 + 1] = q.y;
  }

  // The packer's scale floor differs from fake quantization; keep both stages.
  // Each packed byte puts `.x` in the low nibble.
  IndexPacked out;
#pragma unroll
  for (uint32_t half = 0; half < 2; ++half) {
    const auto amax = warp::reduce_max<kHalfLanes>(fmaxf(fabsf(data[half * 2]), fabsf(data[half * 2 + 1])));
    out.exponent[half] = index_pack_exponent(amax);
    // `inv_scale_ue8m0` instead of the reference's division: the scale is a
    // power of two so both are exact, except at exponent 254, which needs a
    // block absmax above `6 * 2^126` and so cannot come from a finite float.
    const auto inv_scale = deepseek_v4::fp8::inv_scale_ue8m0(static_cast<int32_t>(out.exponent[half]));
    const auto code = __nv_cvt_float2_to_fp4x2(
        fp32x2_t{data[half * 2] * inv_scale, data[half * 2 + 1] * inv_scale}, __NV_E2M1, cudaRoundNearest);
    out.payload[half] = clear_negative_zero(static_cast<uint32_t>(code));
  }
  return out;
}

/// \brief The row's four block exponents, packed little-endian into one word.
///
/// They live on two lanes -- 0 holds blocks 0 and 2, 16 holds 1 and 3 -- so this
/// costs two shuffles, and the result is the word only for the lower half of
/// the warp. Every lane must reach it: the shuffles are warp-wide.
SGL_DEVICE uint32_t index_scale_word(const uint32_t (&exponent)[2]) {
  using namespace device;
  const auto exp_1 = __shfl_sync(warp::kFullMask, exponent[0], kWarpThreads / 2);
  const auto exp_3 = __shfl_sync(warp::kFullMask, exponent[1], kWarpThreads / 2);
  return exponent[0] | (exp_1 << 8) | (exponent[1] << 16) | (exp_3 << 24);
}

/// \brief One warp per token; grid = ceil(num_tokens / kFp4RopeWarpsPerCTA).
///
/// Lane L owns head elements {2L, 2L+1} and tail elements {64+2L, 64+2L+1},
/// so each lane carries one complex RoPE pair. Only RMSNorm spans the full
/// row; the remaining reductions use the FP4 block layout.
template <bool kUsePDL, int64_t kHeadDim, int64_t kRopeDim, uint32_t kPageSize, uint32_t kRatio, typename PosT>
__global__
__launch_bounds__(kFp4RopeWarpsPerCTA* device::kWarpThreads) void flash_index_k_kernel(const IndexKParams params) {
  using namespace device;
  namespace fp4 = deepseek_v4::fp4;

  constexpr uint32_t kPayloadBytes = kHeadDim / 2;
  constexpr uint32_t kScaleBytes = kHeadDim / fp4::kBlockSize;
  constexpr uint32_t kSlotBytes = kPayloadBytes + kScaleBytes;

  static_assert(
      kHeadDim == 128 && kRopeDim == 64,
      "the one-warp tiling is specific to a 128-wide row whose second half is the RoPE tail");
  static_assert(kHeadDim == 2 * kWarpThreads * 2, "a lane owns one bf16x2 of each half");
  static_assert(kScaleBytes == 4, "the four block exponents are packed into one uint32 store");
  static_assert(std::has_single_bit(kRatio), "group_pos is derived by masking, so the ratio must be a power of two");

  using bf16_vec_t = AlignedVector<bf16x2_t, 1>;
  using fp32_vec_t = AlignedVector<float, 2>;

  const auto lane = threadIdx.x % kWarpThreads;
  const auto row = blockIdx.x * kFp4RopeWarpsPerCTA + threadIdx.x / kWarpThreads;
  // Warp-uniform, so the reductions below still see a full warp.
  if (row >= params.num_tokens) return;

  // Both come from the step's metadata rather than the predecessor, so
  // prefetching them ahead of the PDL gate overlaps with the `wk` GEMM's tail.
  // A row that publishes nothing is dropped at the store, not here.
  const auto slot_id = params.loc[row];
  const auto position = static_cast<int64_t>(static_cast<const PosT*>(params.positions)[row]);
  PDLWaitPrimary<kUsePDL>();

  bf16_vec_t head_in, tail_in, head_w, tail_w;
  head_in.load(params.input + row * kHeadDim, lane);
  tail_in.load(params.input + row * kHeadDim, lane + kWarpThreads);
  head_w.load(params.norm_weight, lane);
  tail_w.load(params.norm_weight, lane + kWarpThreads);
  fp32_vec_t freq;
  freq.load(params.freqs_cis + (position & ~static_cast<int64_t>(kRatio - 1)) * kRopeDim, lane);

  fp32x2_t head, tail;
  {
    const auto [h0, h1] = cast<fp32x2_t>(head_in[0]);
    const auto [t0, t1] = cast<fp32x2_t>(tail_in[0]);
    const auto sqrsum = warp::reduce_sum(h0 * h0 + h1 * h1 + t0 * t0 + t1 * t1);
    const auto inv_rms = math::rsqrt(sqrsum * (1.0f / static_cast<float>(kHeadDim)) + params.eps);
    const auto [wh0, wh1] = cast<fp32x2_t>(head_w[0]);
    const auto [wt0, wt1] = cast<fp32x2_t>(tail_w[0]);
    // `k_norm` materializes a bf16 tensor, so the norm result is rounded before
    // anything downstream sees it -- the RoPE below included.
    head = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{wh0 * (h0 * inv_rms), wh1 * (h1 * inv_rms)}));
    tail = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{wt0 * (t0 * inv_rms), wt1 * (t1 * inv_rms)}));
  }

  const auto packed = index_rope_quant_pack(head, tail, fp32x2_t{freq[0], freq[1]});
  const auto scale_word = index_scale_word(packed.exponent);

  // A padded graph row, and at ratio > 1 a row completing no group, carry the
  // reserved slot 0 and must publish nothing.
  if (slot_id <= 0) return;
  const auto page = slot_id / kPageSize;
  const auto slot = slot_id % kPageSize;
  const auto page_ptr = params.cache + page * (kPageSize * kSlotBytes);

  // Byte i of the payload covers elements (2i, 2i+1), so a lane's head pair is
  // byte `lane` and its tail pair byte `lane + 32`: two coalesced 32-byte runs.
  const auto payload_ptr = page_ptr + slot * kPayloadBytes;
  payload_ptr[lane] = static_cast<uint8_t>(packed.payload[0]);
  payload_ptr[lane + kWarpThreads] = static_cast<uint8_t>(packed.payload[1]);

  if (lane == 0) {
    *reinterpret_cast<uint32_t*>(page_ptr + kPageSize * kPayloadBytes + slot * kScaleBytes) = scale_word;
  }
}

/// \brief One warp per (token, head); grid = ceil(num_rows / kFp4RopeWarpsPerCTA).
///
/// Input is contiguous [num_tokens, heads, kHeadDim]; row r uses token r / heads
/// and head r % heads. Queries use their own position, without a ratio mask.
template <bool kUsePDL, int64_t kHeadDim, int64_t kRopeDim, typename PosT, bool kWeights>
__global__
__launch_bounds__(kFp4RopeWarpsPerCTA* device::kWarpThreads) void flash_index_q_kernel(const IndexQParams params) {
  using namespace device;

  constexpr uint32_t kPayloadBytes = kHeadDim / 2;

  static_assert(
      kHeadDim == 128 && kRopeDim == 64,
      "the one-warp tiling is specific to a 128-wide row whose second half is the RoPE tail");
  static_assert(kHeadDim == 2 * kWarpThreads * 2, "a lane owns one bf16x2 of each half");

  using bf16_vec_t = AlignedVector<bf16x2_t, 1>;
  using fp32_vec_t = AlignedVector<float, 2>;

  const auto lane = threadIdx.x % kWarpThreads;
  const auto row = blockIdx.x * kFp4RopeWarpsPerCTA + threadIdx.x / kWarpThreads;
  // Warp-uniform, so the reductions below still see a full warp.
  if (row >= params.num_rows) return;

  // The position lookup is independent of the PDL producer.
  const auto position = static_cast<int64_t>(static_cast<const PosT*>(params.positions)[row / params.heads]);
  PDLWaitPrimary<kUsePDL>();

  bf16_vec_t head_in, tail_in;
  head_in.load(params.input + row * kHeadDim, lane);
  tail_in.load(params.input + row * kHeadDim, lane + kWarpThreads);
  fp32_vec_t freq;
  freq.load(params.freqs_cis + position * kRopeDim, lane);

  const auto packed =
      index_rope_quant_pack(cast<fp32x2_t>(head_in[0]), cast<fp32x2_t>(tail_in[0]), fp32x2_t{freq[0], freq[1]});
  const auto scale_word = index_scale_word(packed.exponent);

  const auto payload_ptr = params.payload + row * kPayloadBytes;
  payload_ptr[lane] = static_cast<int8_t>(packed.payload[0]);
  payload_ptr[lane + kWarpThreads] = static_cast<int8_t>(packed.payload[1]);

  if (lane == 0) params.scale[row] = static_cast<int32_t>(scale_word);

  if constexpr (kWeights) {
    // Match head_weights(x).float(): multiply by the fp32-rounded scale,
    // round to nearest-even bf16, then widen to fp32.
    if (lane == 0) {
      const auto w = cast<float>(params.head_weights[row]) * params.weight_scale;
      params.weights_out[row] = cast<float>(cast<bf16_t>(w));
    }
  }
}

/// \brief Host side of `flash_index_k_kernel`.

} // namespace sglang

using namespace sglang;
template<int SPLIT> int small_launch(int shape,int rows,int64_t stride,const void* x,const void* w,void* y,cudaStream_t stream){
 SmallGemmParams p{(bf16_t*)y,(const bf16_t*)x,(const bf16_t*)w,stride,(uint32_t)rows};
 void* args[]={&p};int gy=(rows+7)/8;
 if(shape==128)return (int)cudaLaunchKernel((void*)n128k512_kernel<N128K512Trait<SPLIT>,false>,dim3(32,gy),dim3(128),args,0,stream);
 return (int)cudaLaunchKernel((void*)n32k5120_kernel<N32K5120Trait<SPLIT>,false>,dim3(32,gy),dim3(320),args,0,stream);
}
extern "C" int ca_index_abi(){return 1;}
extern "C" int ca_index_small(int shape,int rows,int64_t stride,const void* x,const void* w,void* y,cudaStream_t stream){
 if(rows<1||rows>32||(shape!=32&&shape!=128)||stride<(shape==128?512:5120)||stride%16)return -1;
 int split=(rows+((rows+7)/8)-1)/((rows+7)/8);
 #define S(I) case I:return small_launch<I>(shape,rows,stride,x,w,y,stream);
 switch(split){S(1) S(2) S(3) S(4) S(5) S(6) S(7) S(8) default:return -2;}
 #undef S
}
extern "C" int ca_index_k(int rows,int ratio,const void* input,const void* weight,const float* freq,
 const int32_t* pos,const int64_t* loc,uint8_t* cache,cudaStream_t stream){
 if(rows<1||rows>4096||(ratio!=1&&ratio!=2))return -1;
 IndexKParams p{(const bf16_t*)input,(const bf16_t*)weight,freq,pos,loc,cache,(uint32_t)rows,1e-20f};
 void* args[]={&p};
 if(ratio==1)return (int)cudaLaunchKernel((void*)flash_index_k_kernel<false,128,64,64,1,int32_t>,dim3((rows+3)/4),dim3(128),args,0,stream);
 return (int)cudaLaunchKernel((void*)flash_index_k_kernel<false,128,64,64,2,int32_t>,dim3((rows+3)/4),dim3(128),args,0,stream);
}
extern "C" int ca_index_q(int rows,const void* input,const float* freq,const int32_t* pos,
 const void* head_weights,int8_t* payload,int32_t* scale,float* weights,cudaStream_t stream){
 if(rows<1||rows>4096)return -1;
 IndexQParams p{(const bf16_t*)input,freq,pos,payload,scale,(const bf16_t*)head_weights,weights,0.015625f,(uint32_t)rows*32,32};
 void* args[]={&p};return (int)cudaLaunchKernel((void*)flash_index_q_kernel<false,128,64,int32_t,true>,dim3(rows*8),dim3(128),args,0,stream);
}
