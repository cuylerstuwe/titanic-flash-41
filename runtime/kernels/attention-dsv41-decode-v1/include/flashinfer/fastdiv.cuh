/*
 * Titanic local replacement for FlashInfer v0.7.3 include/flashinfer/fastdiv.cuh (Apache-2.0).
 * Upstream wraps cuda::fast_mod_div<uint32_t>, which the pinned toolchain (CUDA 13.0.88, CCCL 3.0) does not provide.
 * Same API (default constructor, implicit conversion, divmod, operator/ and operator%) with exact integer division,
 * so results are identical to upstream; only the instruction sequence differs.
 */
#ifndef FLASHINFER_FASTDIV_CUH_
#define FLASHINFER_FASTDIV_CUH_
#include <cstdint>

namespace flashinfer {

struct uint_fastdiv {
  __host__ __device__ uint_fastdiv() : div_(1), d_(0) {}

  __host__ uint_fastdiv(uint32_t d) : div_(d ? d : 1), d_(d) {}

  __host__ __device__ __forceinline__ operator unsigned int() const { return d_; }

  __host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
    q = n / div_;
    r = n - q * d_;
  }

 private:
  uint32_t div_;
  uint32_t d_;
};

__host__ __device__ __forceinline__ uint32_t operator/(const uint32_t n, const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return q;
}

__host__ __device__ __forceinline__ uint32_t operator%(const uint32_t n, const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return r;
}

}  // namespace flashinfer

#endif  // FLASHINFER_FASTDIV_CUH_
