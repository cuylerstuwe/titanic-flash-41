// SGLang device RoPE kernel, Apache-2.0; source hashes in native.json.
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <flashinfer/norm.cuh>
namespace sglang {
using DType = bf16_t;
constexpr int64_t kRopeDim = 64;
constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;

struct FusedQKRopeParams {
  void* __restrict__ q;
  void* __restrict__ k;
  const float* __restrict__ freqs_cis;
  const void* __restrict__ positions;
  int64_t q_stride_batch;
  int64_t k_stride_batch;
  int64_t q_stride_head;
  int64_t k_stride_head;
  uint32_t num_q_heads;
  uint32_t num_k_heads;
  uint32_t batch_size;
};

template <bool kUsePDL, bool kInverse, typename IndexType>
__global__ __launch_bounds__(kBlockSize, 16)  //
    void deepseek_rope_kernel(const __grid_constant__ FusedQKRopeParams param) {
  using namespace device;
  using DType2 = packed_t<DType>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto global_warp_id = blockIdx.x * kNumWarps + warp_id;

  const auto& [
    q, k, freqs_cis, positions, //
    q_stride_batch, k_stride_batch, q_stride_head, k_stride_head, //
    num_q_heads, num_k_heads, batch_size
  ] = param;

  const auto num_total_heads = num_q_heads + num_k_heads;
  const auto head_id = global_warp_id % num_total_heads;
  const auto batch_id = global_warp_id / num_total_heads;
  if (batch_id >= batch_size) return;

  const auto position = static_cast<const IndexType*>(positions)[batch_id];
  const auto is_q = head_id < num_q_heads;
  const auto local_head = is_q ? head_id : (head_id - num_q_heads);
  const auto stride_batch = is_q ? q_stride_batch : k_stride_batch;
  const auto stride_head = is_q ? q_stride_head : k_stride_head;
  const auto base_ptr = is_q ? q : k;
  const auto input = static_cast<DType2*>(pointer::offset(base_ptr, batch_id * stride_batch, local_head * stride_head));

  const auto freq_ptr = reinterpret_cast<const fp32x2_t*>(freqs_cis + position * kRopeDim);
  const auto [f_real, f_imag] = freq_ptr[lane_id];
  PDLWaitPrimary<kUsePDL>();

  const auto data = input[lane_id];
  const auto [x_real, x_imag] = cast<fp32x2_t>(data);
  fp32x2_t output;
  if constexpr (kInverse) {
    // (a + bi) * (c - di) = (ac + bd) + (bc - ad)i
    output = {
        x_real * f_real + x_imag * f_imag,
        x_imag * f_real - x_real * f_imag,
    };
  } else {
    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    output = {
        x_real * f_real - x_imag * f_imag,
        x_real * f_imag + x_imag * f_real,
    };
  }
  input[lane_id] = cast<DType2>(output);

  PDLTriggerSecondary<kUsePDL>();
}


}
extern "C" int ca_prelude_abi() { return 1; }
extern "C" int ca_rmsnorm(int rows, int width, int stride, const void* input,
 const void* weight, void* output, cudaStream_t stream) {
 if (rows<1 || rows>65536 || (width!=512 && width!=1280 && width!=5120) || stride<width) return -1;
 return (int)flashinfer::norm::RMSNorm((__nv_bfloat16*)input,(__nv_bfloat16*)weight,
   (__nv_bfloat16*)output,rows,width,stride,width,1e-20f,false,stream);
}
extern "C" int ca_rope(int rows, int heads, int inverse, void* input,
 const float* frequencies, const int32_t* positions, cudaStream_t stream) {
 if (rows<1 || rows>65536 || heads!=16 || (inverse!=0 && inverse!=1)) return -1;
 sglang::FusedQKRopeParams p{(uint8_t*)input+448*2,nullptr,frequencies,positions,
   (int64_t)heads*512*2,0,512*2,0,(uint32_t)heads,0,(uint32_t)rows};
 int blocks=(rows*heads+3)/4;
 if (inverse) sglang::deepseek_rope_kernel<false,true,int32_t><<<blocks,128,0,stream>>>(p);
 else sglang::deepseek_rope_kernel<false,false,int32_t><<<blocks,128,0,stream>>>(p);
 return (int)cudaGetLastError();
}
