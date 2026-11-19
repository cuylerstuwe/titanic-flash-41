// CUDA C++ twin of b12x's RoCE one-shot all-reduce kernel (SG17 release, Apache-2.0; the DSL source is
// primitive-exports/roce-notify-stage-v1/_notify.py, a variant of it), for the pico-style verify layer's fused sum
// boundary (verify-layer stage 1 (a)). Same protocol on the same proxy, counters and
// slots, same arithmetic: it must store the same bits as the exported kernel (checked by collective_probe).
//
// One launch performs one all-reduce of `size_packs` 16-byte packs of bf16:
//   1. stage: copy the device input into send[seq & 1] in pinned host memory;
//   2. doorbell: the last block to finish staging publishes nbytes (also in the per-slot word nbytes[seq & 1]) and
//      then seq in the control record the proxy polls;
//   3. wait: spin on flag[peer][seq & 1][hca] == seq for every peer and HCA; a wait past spin_limit records the peer,
//      the HCA and seq in the control record and poisons the device word;
//   4. reduce: the local input and every peer slot in fixed rank order, fp32 accumulation, one bf16 rounding;
//   5. epoch: the last block to finish advances the device-resident epoch (unless a wait failed).
// Layout constants as roce-v0's manifest: world 4, 2 HCAs, 2 slots, flag stride 128, 256 threads a block.
#pragma once
#include <stdint.h>

namespace roce_twin {
constexpr int kWorld = 4, kHcas = 2, kSlots = 2, kFlagStride = 128, kThreads = 256;

__device__ __forceinline__ uint32_t ld_relaxed_gpu(uint64_t a) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_relaxed_sys(uint64_t a) {
  uint32_t v;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_acquire_sys(uint64_t a) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_sys(uint64_t a, uint32_t v) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release_gpu(uint64_t a, uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t atom_add_relaxed_gpu(uint64_t a, uint32_t v) {
  uint32_t old;
  asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(a), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ void fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_sc_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }
__device__ __forceinline__ uint4 ld_global_v4(uint64_t a) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_relaxed_sys_v4(uint64_t a) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ void st_global_v4(uint64_t a, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
// Spin until *a == want (acquire, system scope) or `limit` polls: returns true on a timeout.
__device__ __forceinline__ bool spin_until_eq_acquire_sys(uint64_t a, uint32_t want, uint32_t limit) {
  for (uint32_t i = 0; i < limit; ++i)
    if (ld_acquire_sys(a) == want) return false;
  return true;
}
__device__ __forceinline__ void accumulate(float (&acc)[8], uint4 w, bool init) {
  const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float lo = __uint_as_float(words[j] << 16), hi = __uint_as_float(words[j] & 0xffff0000u);  // cvt.f32.bf16
    if (init) {
      acc[2 * j] = lo;
      acc[2 * j + 1] = hi;
    } else {
      acc[2 * j] = __fadd_rn(acc[2 * j], lo);
      acc[2 * j + 1] = __fadd_rn(acc[2 * j + 1], hi);
    }
  }
}
__device__ __forceinline__ uint4 pack(const float (&acc)[8]) {
  uint32_t words[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(acc[2 * j]));
    const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(acc[2 * j + 1]));
    words[j] = lo | hi << 16;
  }
  return make_uint4(words[0], words[1], words[2], words[3]);
}
}  // namespace roce_twin

extern "C" __global__ void __launch_bounds__(256)
    ca_roce_oneshot_bf16(uint64_t input, uint64_t output, int size_packs, int nbytes, uint64_t recv_base,
                         uint64_t flag_base, uint64_t send_base, uint64_t ctrl_base, uint64_t slot_bytes,
                         uint64_t epoch_ptr, uint64_t stage_counter_ptr, uint64_t tail_counter_ptr, uint64_t poison_ptr,
                         uint32_t spin_limit, int rank) {
  using namespace roce_twin;
  const int tid = threadIdx.x, gdim = gridDim.x;
  // Every block reads the epoch before any block can advance it: the advance happens only after all blocks arrived
  // at the tail counter.
  const uint32_t seq = ld_relaxed_gpu(epoch_ptr) + 1u;
  const uint64_t slot = seq & 1u;
  const uint64_t send_slot = send_base + slot * slot_bytes;
  const int index = blockIdx.x * kThreads + tid, stride = gdim * kThreads;
  // A recorded timeout poisons the runtime: later launches do nothing.
  if (ld_relaxed_gpu(poison_ptr) != 0u) return;
  // 1. stage the input into the pinned send slot
  for (int i = index; i < size_packs; i += stride)
    st_global_v4(send_slot + (uint64_t)i * 16u, ld_global_v4(input + (uint64_t)i * 16u));
  __syncthreads();
  // 2. the last block to finish staging rings the proxy doorbell
  if (tid == 0) {
    fence_sc_sys();
    const uint32_t prior = atom_add_relaxed_gpu(stage_counter_ptr, 1u);
    if ((prior + 1u) % (uint32_t)gdim == 0u) {
      st_relaxed_sys(ctrl_base + 4, (uint32_t)nbytes);
      st_relaxed_sys(ctrl_base + 16 + slot * 4, (uint32_t)nbytes);
      fence_sc_sys();
      st_relaxed_sys(ctrl_base, seq);
    }
  }
  // 3. wait for every peer's payload-stripe flags
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
  __syncthreads();
  // A wait that timed out leaves a peer slot unreliable: skip the data phase.
  if (ld_relaxed_gpu(poison_ptr) == 0u) {
    // 4. reduce in fixed rank order so every rank stores identical bits
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
      st_global_v4(output + offset, pack(acc));
    }
  }
  // 5. the last block to finish reduction publishes the next epoch
  fence_sc_gpu();
  __syncthreads();
  if (tid == 0) {
    const uint32_t prior = atom_add_relaxed_gpu(tail_counter_ptr, 1u);
    if ((prior + 1u) % (uint32_t)gdim == 0u) {
      fence_sc_gpu();
      // Every block's timeout store precedes its tail arrival, so the error word is final here; a failed sequence
      // keeps the epoch, which makes every later launch a no-op until the host raises.
      if (ld_relaxed_sys(ctrl_base + 8) == 0u) st_release_gpu(epoch_ptr, seq);
    }
  }
}
