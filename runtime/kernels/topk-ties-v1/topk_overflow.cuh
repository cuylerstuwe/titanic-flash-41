  // Experimental exact refinement with lowest-input-index membership at an
  // equal-score cutoff. Output order within the selected set remains free.
  // The index passes execute only if the final FP32 score is oversubscribed.
  SGL_DEVICE static void exact_overflow(const TopKProblem& problem, Smem* smem,
      uint32_t base, uint32_t need, float lo, float hi, bool include_hi = false) {
    const auto tx = threadIdx.x;
    uint32_t prefix = 0, prefix_mask = 0, remaining = need;
    uint32_t score_cutoff = 0, index_cutoff = 0xFFFFFFFFu, tie_need = 0;
    for (int round = 0; round < 8; ++round) {
      const int shift = 24 - (round % 4) * 8;
      if (tx < 256) smem->histogram[tx] = 0;
      __syncthreads();
      for_each_input(problem.in, problem.seq_len, [&](float value, uint32_t idx) {
        if (value >= lo && (value < hi || (include_hi && value == hi))) {
          const uint32_t score = extract_exact_bin(value);
          if (round < 4 || score == score_cutoff) {
            const uint32_t key = round < 4 ? score : ~idx;
            if ((key & prefix_mask) == prefix) {
              atomicAdd(&smem->histogram[(key >> shift) & 255u], 1u);
            }
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
            smem->warp_sum[2] = count;
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
      const uint32_t equal_count = smem->warp_sum[2];
      __syncthreads();
      if (round == 3) {
        score_cutoff = prefix;
        tie_need = remaining;
        if (equal_count == remaining) break;
        prefix = prefix_mask = 0;
      } else if (round == 7) {
        index_cutoff = ~prefix;
      }
    }
    const uint32_t greater = need - tie_need;
    if (tx == 0) smem->count_gt = smem->count_eq = 0;
    __syncthreads();
    for_each_input(problem.in, problem.seq_len, [&](float value, uint32_t idx) {
      if (value >= lo && (value < hi || (include_hi && value == hi))) {
        const uint32_t key = extract_exact_bin(value);
        if (key > score_cutoff) {
          const uint32_t pos = atomicAdd(&smem->count_gt, 1u);
          if (pos >= greater) asm volatile("trap;");
          problem.emit(base + pos, idx);
        } else if (key == score_cutoff && idx <= index_cutoff) {
          const uint32_t pos = atomicAdd(&smem->count_eq, 1u);
          if (pos >= tie_need) asm volatile("trap;");
          problem.emit(base + greater + pos, idx);
        }
      }
    });
    __syncthreads();
    if (tx == 0 && (smem->count_gt != greater || smem->count_eq != tie_need)) {
      asm volatile("trap;");
    }
  }
