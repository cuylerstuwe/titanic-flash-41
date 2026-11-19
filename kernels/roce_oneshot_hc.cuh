// Verify-layer stage 1 (b/c): the RoCE one-shot sum (roce_oneshot_twin.cuh's protocol and arithmetic, unchanged) with
// the mHC boundary in its reduce epilogue, so the reduced sublayer output never leaves registers:
//   y_c   = hc_post: the new residual stream, four copies (the exported mhc-post-tiles-v1 kernel's arithmetic: Triton,
//           from SGLang f80c91a4 mhc_post_split_h.py, Apache-2.0):
//             z = fma(post_c, a, comb_c * r_0); z = fma(comb_{4+c}, r_1, z); z = fma(comb_{8+c}, r_2, z);
//             z = fma(comb_{12+c}, r_3, z); y_c = bf16(z)
//           with a the reduced output rounded to bf16 and r_k the input stream's copies;
//   value = the next sublayer's collapse with this sublayer's pre (the model's pre-from-previous-sublayer form, as
//           the exported mhc-norm-quant-v1 kernel computes it before its RMS: four sequential fp32 FMAs from zero,
//           bf16): value = bf16(fma(y_3, p_3, fma(y_2, p_2, fma(y_1, p_1, fma(y_0, p_0, 0)))))
// The RMS norm and MXFP8 quantization stay in the norm-quant kernel (its RMS reduction tree is Triton's), re-exported
// to read `value` instead of the four copies. post / comb / pre are this sublayer's coefficients, joined before the sum
// as today. Every value matches the separate kernels' bits (checked by the collective probe and the end-to-end MXFP8
// comparison). Rows of 5,120 columns: a 16-byte pack is eight columns of one row. The sum covers every row of the
// payload; the epilogue only the first `epilogue_rows` (the boundary's live rows, as the separate post-mix).
#pragma once
#include "roce_oneshot_twin.cuh"

namespace roce_hc {
__device__ __forceinline__ uint32_t ld_acquire_gpu(uint64_t a) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
constexpr int kPacksPerRow = 5120 / 8;
// The boundary epilogue for one 16-byte pack (eight columns of one row) of the reduced sum.
__device__ __forceinline__ void epilogue(const uint4 summed, int i, const uint16_t* __restrict__ residual,
                                         const float* __restrict__ post, const float* __restrict__ comb,
                                         const float* __restrict__ pre, uint16_t* __restrict__ stream_out,
                                         uint16_t* __restrict__ value_out) {
  const int row = i / kPacksPerRow, col = (i - row * kPacksPerRow) * 8;
  float po[4], pr[4], cb[16];
#pragma unroll
  for (int j = 0; j < 4; ++j) po[j] = post[row * 4 + j], pr[j] = pre[row * 4 + j];
#pragma unroll
  for (int j = 0; j < 16; ++j) cb[j] = comb[row * 16 + j];
  uint4 xin[4];
#pragma unroll
  for (int k = 0; k < 4; ++k)
    xin[k] = *reinterpret_cast<const uint4*>(residual + (size_t)row * 20480 + (size_t)k * 5120 + col);
  const uint32_t sw[4] = {summed.x, summed.y, summed.z, summed.w};
  uint32_t yw[4][4], vw[4];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float a = (e & 1) ? __uint_as_float(sw[e >> 1] & 0xffff0000u) : __uint_as_float(sw[e >> 1] << 16);
    float r[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint32_t xw = (&xin[k].x)[e >> 1];
      r[k] = (e & 1) ? __uint_as_float(xw & 0xffff0000u) : __uint_as_float(xw << 16);
    }
    float yb[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      float z = __fmaf_rn(po[c], a, __fmul_rn(cb[c], r[0]));
      z = __fmaf_rn(cb[4 + c], r[1], z);
      z = __fmaf_rn(cb[8 + c], r[2], z);
      z = __fmaf_rn(cb[12 + c], r[3], z);
      const uint32_t bits = __bfloat16_as_ushort(__float2bfloat16_rn(z));
      yb[c] = __uint_as_float(bits << 16);
      if (e & 1) yw[c][e >> 1] |= bits << 16;
      else yw[c][e >> 1] = bits;
    }
    float v = __fmaf_rn(yb[0], pr[0], 0.0f);
    v = __fmaf_rn(yb[1], pr[1], v);
    v = __fmaf_rn(yb[2], pr[2], v);
    v = __fmaf_rn(yb[3], pr[3], v);
    const uint32_t vb = __bfloat16_as_ushort(__float2bfloat16_rn(v));
    if (e & 1) vw[e >> 1] |= vb << 16;
    else vw[e >> 1] = vb;
  }
#pragma unroll
  for (int c = 0; c < 4; ++c)
    *reinterpret_cast<uint4*>(stream_out + (size_t)row * 20480 + (size_t)c * 5120 + col) =
        make_uint4(yw[c][0], yw[c][1], yw[c][2], yw[c][3]);
  *reinterpret_cast<uint4*>(value_out + (size_t)row * 5120 + col) = make_uint4(vw[0], vw[1], vw[2], vw[3]);
}
// Steps 1-2 of the protocol (stage into send[seq & 1], the last staging block rings the doorbell).
__device__ __forceinline__ void stage_and_ring(uint64_t input, int size_packs, int nbytes, uint64_t send_slot,
                                               uint64_t ctrl_base, uint64_t slot, uint32_t seq,
                                               uint64_t stage_counter_ptr, int index, int stride, int gdim) {
  using namespace roce_twin;
  for (int i = index; i < size_packs; i += stride)
    st_global_v4(send_slot + (uint64_t)i * 16u, ld_global_v4(input + (uint64_t)i * 16u));
  __syncthreads();
  if (threadIdx.x == 0) {
    fence_sc_sys();
    const uint32_t prior = atom_add_relaxed_gpu(stage_counter_ptr, 1u);
    if ((prior + 1u) % (uint32_t)gdim == 0u) {
      st_relaxed_sys(ctrl_base + 4, (uint32_t)nbytes);
      st_relaxed_sys(ctrl_base + 16 + slot * 4, (uint32_t)nbytes);
      fence_sc_sys();
      st_relaxed_sys(ctrl_base, seq);
    }
  }
}
// Step 3's flag wait for one block: every peer x HCA stripe flag; a timeout records the peer / HCA and poisons.
__device__ __forceinline__ void wait_flags(uint64_t flag_base, uint64_t ctrl_base, uint64_t slot, uint32_t seq,
                                           uint32_t spin_limit, int rank, uint64_t poison_ptr) {
  using namespace roce_twin;
  const int tid = threadIdx.x;
  if (tid < kWorld * kHcas) {
    const int peer = tid / kHcas, hca = tid - peer * kHcas;
    if (peer != rank) {
      const uint64_t flag = flag_base + (((uint64_t)peer * kSlots + slot) * kHcas + (uint64_t)hca) * kFlagStride;
      if (spin_until_eq_acquire_sys(flag, seq, spin_limit)) {
        st_relaxed_sys(ctrl_base + 12, (uint32_t)peer);
        st_relaxed_sys(ctrl_base + 24, (uint32_t)hca);
        st_relaxed_sys(ctrl_base + 8, seq);
        st_release_gpu(poison_ptr, seq);
      }
    }
  }
}
// Steps 4 (reduce in rank order + the boundary epilogue on the live rows) and 5 (the epoch).
__device__ __forceinline__ void reduce_and_finish(uint64_t input, uint64_t output, int size_packs, uint64_t recv_base,
                                                  uint64_t ctrl_base, uint64_t slot_bytes, uint64_t slot, uint32_t seq,
                                                  uint64_t epoch_ptr, uint64_t tail_counter_ptr, uint64_t poison_ptr,
                                                  int rank, const uint16_t* __restrict__ residual,
                                                  const float* __restrict__ post, const float* __restrict__ comb,
                                                  const float* __restrict__ pre, uint16_t* __restrict__ stream_out,
                                                  uint16_t* __restrict__ value_out, int epilogue_rows, int index,
                                                  int stride, int gdim) {
  using namespace roce_twin;
  if (ld_relaxed_gpu(poison_ptr) == 0u) {
    for (int i = index; i < size_packs; i += stride) {
      const uint64_t offset = (uint64_t)i * 16u;
      float acc[8];
#pragma unroll
      for (int source = 0; source < kWorld; ++source) {
        const uint4 w = source == rank
            ? ld_global_v4(input + offset)
            : ld_relaxed_sys_v4(recv_base + ((uint64_t)source * kSlots + slot) * slot_bytes + offset);
        accumulate(acc, w, source == 0);
      }
      const uint4 summed = pack(acc);
      st_global_v4(output + offset, summed);
      // The post-mix runs on the boundary's live rows only (rows past them keep their contents, as with post_op).
      if (i / kPacksPerRow >= epilogue_rows) continue;
      epilogue(summed, i, residual, post, comb, pre, stream_out, value_out);
    }
  }
  fence_sc_gpu();
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t prior = atom_add_relaxed_gpu(tail_counter_ptr, 1u);
    if ((prior + 1u) % (uint32_t)gdim == 0u) {
      fence_sc_gpu();
      if (ld_relaxed_sys(ctrl_base + 8) == 0u) st_release_gpu(epoch_ptr, seq);
    }
  }
}
}  // namespace roce_hc

extern "C" __global__ void __launch_bounds__(256)
    ca_roce_oneshot_hc_bf16(uint64_t input, uint64_t output, int size_packs, int nbytes, uint64_t recv_base,
                            uint64_t flag_base, uint64_t send_base, uint64_t ctrl_base, uint64_t slot_bytes,
                            uint64_t epoch_ptr, uint64_t stage_counter_ptr, uint64_t tail_counter_ptr,
                            uint64_t poison_ptr, uint32_t spin_limit, int rank, const uint16_t* __restrict__ residual,
                            const float* __restrict__ post, const float* __restrict__ comb,
                            const float* __restrict__ pre, uint16_t* __restrict__ stream_out,
                            uint16_t* __restrict__ value_out, int epilogue_rows) {
  using namespace roce_twin;
  const int tid = threadIdx.x, gdim = gridDim.x;
  const uint32_t seq = ld_relaxed_gpu(epoch_ptr) + 1u;
  const uint64_t slot = seq & 1u;
  const int index = blockIdx.x * kThreads + tid, stride = gdim * kThreads;
  if (ld_relaxed_gpu(poison_ptr) != 0u) return;
  roce_hc::stage_and_ring(input, size_packs, nbytes, send_base + slot * slot_bytes, ctrl_base, slot, seq,
                          stage_counter_ptr, index, stride, gdim);
  // 3. every block waits for the peers' stripe flags (as the exported kernel)
  roce_hc::wait_flags(flag_base, ctrl_base, slot, seq, spin_limit, rank, poison_ptr);
  __syncthreads();
  roce_hc::reduce_and_finish(input, output, size_packs, recv_base, ctrl_base, slot_bytes, slot, seq, epoch_ptr,
                             tail_counter_ptr, poison_ptr, rank, residual, post, comb, pre, stream_out, value_out,
                             epilogue_rows, index, stride, gdim);
}

// The same with one polling block: block 0 waits for the peers' flags in host memory and releases the other blocks
// through a device word (`ready_ptr` = seq, release / acquire at GPU scope, transitive with block 0's system-scope
// acquire of the flags), so a wide grid spreads the reduce and epilogue without multiplying the host-memory polling.
// A block that waits past spin_limit records a timeout and poisons, like a flag wait.
extern "C" __global__ void __launch_bounds__(256)
    ca_roce_oneshot_hc2_bf16(uint64_t input, uint64_t output, int size_packs, int nbytes, uint64_t recv_base,
                             uint64_t flag_base, uint64_t send_base, uint64_t ctrl_base, uint64_t slot_bytes,
                             uint64_t epoch_ptr, uint64_t stage_counter_ptr, uint64_t tail_counter_ptr,
                             uint64_t poison_ptr, uint32_t spin_limit, int rank, const uint16_t* __restrict__ residual,
                             const float* __restrict__ post, const float* __restrict__ comb,
                             const float* __restrict__ pre, uint16_t* __restrict__ stream_out,
                             uint16_t* __restrict__ value_out, int epilogue_rows, uint64_t ready_ptr) {
  using namespace roce_twin;
  const int tid = threadIdx.x, gdim = gridDim.x;
  const uint32_t seq = ld_relaxed_gpu(epoch_ptr) + 1u;
  const uint64_t slot = seq & 1u;
  const int index = blockIdx.x * kThreads + tid, stride = gdim * kThreads;
  if (ld_relaxed_gpu(poison_ptr) != 0u) return;
  roce_hc::stage_and_ring(input, size_packs, nbytes, send_base + slot * slot_bytes, ctrl_base, slot, seq,
                          stage_counter_ptr, index, stride, gdim);
  if (blockIdx.x == 0) {
    roce_hc::wait_flags(flag_base, ctrl_base, slot, seq, spin_limit, rank, poison_ptr);
    __syncthreads();
    if (tid == 0 && ld_relaxed_gpu(poison_ptr) == 0u) st_release_gpu(ready_ptr, seq);
  } else if (tid == 0) {
    bool ready = false;
    for (uint32_t n = 0; n < spin_limit; ++n) {
      if (roce_hc::ld_acquire_gpu(ready_ptr) == seq) { ready = true; break; }
      if (ld_relaxed_gpu(poison_ptr) != 0u) break;
    }
    if (!ready && ld_relaxed_gpu(poison_ptr) == 0u) {
      st_relaxed_sys(ctrl_base + 8, seq);
      st_release_gpu(poison_ptr, seq);
    }
  }
  __syncthreads();
  roce_hc::reduce_and_finish(input, output, size_packs, recv_base, ctrl_base, slot_bytes, slot, seq, epoch_ptr,
                             tail_counter_ptr, poison_ptr, rank, residual, post, comb, pre, stream_out, value_out,
                             epilogue_rows, index, stride, gdim);
}
