// Attributed SGLang CUDA primitives; Apache-2.0. See native.json.
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>
#include <sgl_kernel/gemm/dot_product.cuh>
#include <array>
#include <type_traits>
namespace sglang {
constexpr uint32_t kMaxBlockThreads = 1024;

template <uint32_t N_, uint32_t K_, uint32_t N_SPLIT_, uint32_t kBytes_>
struct GEMMTraitN {
  static constexpr uint32_t N = N_;
  static constexpr uint32_t K = K_;
  static constexpr uint32_t N_SPLIT = N_SPLIT_;  // per block n size
  static constexpr uint32_t kBytes = kBytes_;

  static_assert(N % N_SPLIT == 0, "N must be divisible by n_split");
  static_assert((K * sizeof(bf16_t)) % kBytes == 0, "K must be divisible by kBytes");
  static_assert(kBytes % device::kMaxVecBytes == 0);
  static constexpr uint32_t kNumBlocks = N / N_SPLIT;
  static constexpr uint32_t kBlockSize = (K * sizeof(bf16_t)) / kBytes;
  // always use the largest possible vector
  static constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(bf16_t);
  // may reduce block size for less thread usage
  static constexpr uint32_t kUnroll = kBytes / device::kMaxVecBytes;
  static_assert(kBlockSize % device::kWarpThreads == 0, "block size must be divisible by warp size");
  static_assert(kBlockSize <= kMaxBlockThreads, "block size exceeds the maximum block size");
};

template <uint32_t N_, uint32_t K_, uint32_t N_SPLIT_, uint32_t kBytes_>
struct GEMMTraitK {
  static constexpr uint32_t N = N_;
  static constexpr uint32_t K = K_;
  static constexpr uint32_t N_SPLIT = N_SPLIT_;  // per block n size
  static constexpr uint32_t kBytes = kBytes_;

  static_assert(N % N_SPLIT == 0, "N must be divisible by n_split");
  static_assert((K * sizeof(bf16_t)) % kBytes == 0, "K must be divisible by kBytes");
  static_assert(device::kMaxVecBytes % kBytes == 0);
  static constexpr uint32_t kNumBlocks = N / N_SPLIT;
  static constexpr uint32_t kNumKLanes = (K * sizeof(bf16_t)) / kBytes;
  static constexpr uint32_t kVecSize = kBytes / sizeof(bf16_t);
  static constexpr uint32_t kBlockSize = N_SPLIT * K / kVecSize;
  static_assert(device::kWarpThreads % kNumKLanes == 0, "K reduction must fit in a warp");
  // A partial warp would leave reduce_sum's kFullMask naming absent lanes.
  static_assert(kBlockSize % device::kWarpThreads == 0, "block size must be divisible by warp size");
  static_assert(kBlockSize <= kMaxBlockThreads, "block size exceeds the maximum block size");
};

#define TINY_GEMM_KERNEL __global__ __launch_bounds__(Trait::kBlockSize, 1)  // grid: 1 block per SM

struct TinyGEMMParams {
  void* __restrict__ out;
  const bf16_t* __restrict__ x;
  const bf16_t* __restrict__ w;
  int64_t stride_x;
};

template <typename Trait, uint32_t M, typename Out, bool kUsePDL>
TINY_GEMM_KERNEL void tiny_n_gemm_kernel(const TinyGEMMParams params) {
  using namespace device;
  constexpr uint32_t N = Trait::N;
  constexpr uint32_t K = Trait::K;
  constexpr uint32_t N_SPLIT = Trait::N_SPLIT;
  constexpr uint32_t kVecSize = Trait::kVecSize;
  constexpr uint32_t kUnroll = Trait::kUnroll;
  constexpr uint32_t kBlockSize = Trait::kBlockSize;
  constexpr uint32_t kNumWarps = kBlockSize / kWarpThreads;
  using vec_t = AlignedVector<bf16x2_t, kVecSize / 2>;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const bf16_t* w_tile = params.w + bx * (N_SPLIT * K);

  // prefetch weight before PDL
  vec_t wv[N_SPLIT][kUnroll];
#pragma unroll
  for (uint32_t n = 0; n < N_SPLIT; ++n) {
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      wv[n][u].load(w_tile + n * K, tx + u * kBlockSize);
    }
  }

  PDLWaitPrimary<kUsePDL>();

  vec_t xv[M][kUnroll];
#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      xv[m][u].load(params.x + m * params.stride_x, tx + u * kBlockSize);
    }
  }

  __shared__ float s_acc[kNumWarps][M * N_SPLIT];
  const uint32_t warp_id = tx / kWarpThreads;

#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
    for (uint32_t n = 0; n < N_SPLIT; ++n) {
      float acc = 0.0f;
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        dot_product_vec(xv[m][u], wv[n][u], acc);
      }
      s_acc[warp_id][m * N_SPLIT + n] = warp::reduce_sum(acc);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();

  static_assert(M * N_SPLIT <= kBlockSize);
  if (tx < M * N_SPLIT) {
    float acc[kNumWarps];
#pragma unroll
    for (uint32_t i = 0; i < kNumWarps; ++i) {
      acc[i] = s_acc[i][tx];
    }
#pragma unroll
    for (uint32_t i = 1; i < kNumWarps; ++i) {
      acc[0] += acc[i];
    }
    const uint32_t m = tx / N_SPLIT;
    const uint32_t n = tx % N_SPLIT;
    static_cast<Out*>(params.out)[m * N + bx * N_SPLIT + n] = cast<Out>(acc[0]);
  }
}

template <typename Trait, uint32_t M, typename Out, bool kUsePDL>
TINY_GEMM_KERNEL void tiny_k_gemm_kernel(const TinyGEMMParams params) {
  using namespace device;
  constexpr uint32_t N = Trait::N;
  constexpr uint32_t K = Trait::K;
  constexpr uint32_t N_SPLIT = Trait::N_SPLIT;
  constexpr uint32_t kVecSize = Trait::kVecSize;
  constexpr uint32_t kNumKLanes = Trait::kNumKLanes;
  using vec_t = AlignedVector<bf16x2_t, kVecSize / 2>;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t n_idx = bx * N_SPLIT + tx / kNumKLanes;
  const uint32_t work_id = tx % kNumKLanes;
  const bf16_t* w_tile = params.w + n_idx * K;

  // Weight prefetch: address is input-independent, load before the PDL wait.
  vec_t wv;
  wv.load(w_tile, work_id);

  PDLWaitPrimary<kUsePDL>();
  vec_t xv[M];
#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
    xv[m].load(params.x + m * params.stride_x, work_id);
  }
#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
    float acc = 0.0f;
    dot_product_vec(xv[m], wv, acc);
    // Broadcast store: every lane of the group holds the reduced sum.
    const auto sum = warp::reduce_sum<kNumKLanes>(acc);
    static_cast<Out*>(params.out)[m * N + n_idx] = cast<Out>(sum);
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <bool kApplySwigluLimit, bool kPrecise = true, typename DType2>
SGL_DEVICE fp32x2_t silu_and_mul(DType2 gate, DType2 up, float limit) {
  using namespace device;
  // refer to as implementation. TL;DR: must clamp in bf16
  // https://github.com/deepseek-ai/DeepGEMM/blob/7f2a703ed51ac1f7af07f5e1453b2d3267d37d50/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L984-L997
  if constexpr (kApplySwigluLimit) {
    static_assert(std::is_same_v<DType2, bf16x2_t>);
    gate = __hmin2(gate, {limit, limit});
    up = __hmax2(up, {-limit, -limit});
    up = __hmin2(up, {limit, limit});
  }
  const auto [g0, g1] = cast<fp32x2_t>(gate);
  const auto [u0, u1] = cast<fp32x2_t>(up);
  const auto silu0 = g0 / (1.0f + __expf(-g0));
  const auto silu1 = g1 / (1.0f + __expf(-g1));
  const float val0 = silu0 * u0;
  const float val1 = silu1 * u1;
  if constexpr (kPrecise) {  // I don't know if we should enable this?
    return {val0, val1};
  } else {
    return cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{val0, val1}));
  }
}

struct SiluAndMulClampParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  float swiglu_limit;
  uint32_t out_vecs;
  uint32_t blocks_per_row;
};

template <typename DType2>
SGL_DEVICE bf16x2_t to_bf16x2(DType2 value) {
  if constexpr (std::is_same_v<DType2, bf16x2_t>) {
    return value;
  } else {
    return device::cast<bf16x2_t>(device::cast<fp32x2_t>(value));
  }
}

template <typename DType, bool kUsePDL>
__global__ __launch_bounds__(1024, 2) void  // maximize occupancy
    silu_mul_clamp_kernel(const SiluAndMulClampParams __grid_constant__ params) {
  using namespace device;
  static_assert(sizeof(DType) == 2, "only fp16/bf16 supported");
  using DType2 = packed_t<DType>;
  constexpr auto kVecSize = 16 / sizeof(DType);
  static_assert(kVecSize % 2 == 0 && kVecSize > 0);
  using Vec = AlignedVector<DType2, kVecSize / 2>;
  const auto row = blockIdx.x / params.blocks_per_row;
  const auto block_in_row = blockIdx.x % params.blocks_per_row;
  const auto vec_id = block_in_row * blockDim.x + threadIdx.x;
  const float limit = params.swiglu_limit;

  PDLWaitPrimary<kUsePDL>();
  if (vec_id < params.out_vecs) {
    const auto input = static_cast<const Vec*>(params.input);
    auto output = static_cast<Vec*>(params.output);
    const auto input_row = row * 2 * params.out_vecs;
    const auto gate = input[input_row + vec_id];
    const auto up = input[input_row + params.out_vecs + vec_id];
    Vec out;

#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      out[i] = cast<DType2>(silu_and_mul<true>(to_bf16x2(gate[i]), to_bf16x2(up[i]), limit));
    }

    output[row * params.out_vecs + vec_id] = out;
  }
  PDLTriggerSecondary<kUsePDL>();
}


using RouterTrait = GEMMTraitN<384,5120,8,32>;
using RouterFn = void (*)(TinyGEMMParams);
template <size_t... I> constexpr auto router_table(std::index_sequence<I...>) {
 return std::array<RouterFn,17>{nullptr,tiny_n_gemm_kernel<RouterTrait,I+1,float,false>...};
}
constexpr auto router_kernels=router_table(std::make_index_sequence<16>{});
}
// Independent elementwise addition: exact BF16 inputs, FP32 addition, BF16 RN.
__global__ void add_outputs(const __nv_bfloat16* a,const __nv_bfloat16* b,__nv_bfloat16* out,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n) out[i]=__float2bfloat16_rn(__bfloat162float(a[i])+__bfloat162float(b[i]));
}
extern "C" int ca_ffn_abi() { return 1; }
extern "C" int ca_router_logits(int rows,const void* x,const void* w,void* out,cudaStream_t stream) {
 if(rows<1||rows>16) return -1;
 sglang::TinyGEMMParams p{out,(const sglang::bf16_t*)x,(const sglang::bf16_t*)w,5120};
 auto fn=sglang::router_kernels[rows];void* args[]={&p};
 return (int)cudaLaunchKernel((void*)fn,dim3(48),dim3(320),args,0,stream);
}
extern "C" int ca_shared_swiglu(int rows,const void* input,void* output,cudaStream_t stream) {
 if(rows<1||rows>65536) return -1;
 sglang::SiluAndMulClampParams p{input,output,10.f,72,1};
 sglang::silu_mul_clamp_kernel<sglang::bf16_t,false><<<rows,72,0,stream>>>(p);
 return (int)cudaGetLastError();
}
extern "C" int ca_ffn_add(int rows,const void* routed,const void* shared,void* out,cudaStream_t stream) {
 if(rows<1||rows>65536) return -1;
 add_outputs<<<(rows*5120+255)/256,256,0,stream>>>((const __nv_bfloat16*)routed,(const __nv_bfloat16*)shared,(__nv_bfloat16*)out,rows*5120);
 return (int)cudaGetLastError();
}
