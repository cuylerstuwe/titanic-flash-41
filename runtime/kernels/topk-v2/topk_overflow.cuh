  // Exact refinement when a coarse FP16 bucket contains more distinct FP32
  // candidates than the fixed shared-memory tie array can retain. The original
  // coarse winners [0, base) remain valid; rescan the entire threshold bucket
  // using ordered FP32 bits instead of selecting from a truncated prefix.
  SGL_DEVICE static void exact_overflow(const TopKProblem& problem, Smem* smem,
      uint32_t base, uint32_t need, float lo, float hi) {
    const auto tx = threadIdx.x;
    uint32_t prefix = 0, prefix_mask = 0, remaining = need;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (tx < 256) smem->histogram[tx] = 0;
      __syncthreads();
      for_each_input(problem.in, problem.seq_len, [&](float value, uint32_t) {
        if (value >= lo && value < hi) {
          const uint32_t key = extract_exact_bin(value);
          if ((key & prefix_mask) == prefix) {
            atomicAdd(&smem->histogram[(key >> shift) & 255u], 1u);
          }
        }
      });
      __syncthreads();
      if (tx == 0) {
        uint32_t above = 0; bool found = false;
        for (int bin = 255; bin >= 0; --bin) {
          const uint32_t count = smem->histogram[bin];
          if (above + count >= remaining) {
            smem->warp_sum[0] = static_cast<uint32_t>(bin);
            smem->warp_sum[1] = above;
            found = true;
            break;
          }
          above += count;
        }
        if (!found) asm volatile("trap;");
      }
      __syncthreads();
      prefix |= smem->warp_sum[0] << shift;
      prefix_mask |= 255u << shift;
      remaining -= smem->warp_sum[1];
      __syncthreads();
    }
    const uint32_t greater = need - remaining;
    if (tx == 0) smem->count_gt = smem->count_eq = 0;
    __syncthreads();
    for_each_input(problem.in, problem.seq_len, [&](float value, uint32_t idx) {
      if (value >= lo && value < hi) {
        const uint32_t key = extract_exact_bin(value);
        if (key > prefix) {
          const uint32_t pos = atomicAdd(&smem->count_gt, 1u);
          if (pos >= greater) asm volatile("trap;");
          problem.emit(base + pos, idx);
        } else if (key == prefix) {
          const uint32_t pos = atomicAdd(&smem->count_eq, 1u);
          if (pos < remaining) problem.emit(base + greater + pos, idx);
        }
      }
    });
    __syncthreads();
    if (tx == 0 && (smem->count_gt != greater || smem->count_eq < remaining)) {
      asm volatile("trap;");
    }
  }
