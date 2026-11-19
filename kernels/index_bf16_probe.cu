// Standalone probe of index_bf16.cu: selections against a CPU reference of the same bf16 arithmetic (the order
// of every rounding and addition as the kernel does it), then timings. Not part of the engine.
//   nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 -Xcompiler -ffp-contract=off kernels/index_bf16_probe.cu -o idx16probe
//   ./idx16probe            correctness on random data (several shapes and windows), then timings
#include "index_bf16.cu"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)

static float h_e2m1(uint32_t c) {
  static const float m[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = m[c & 7];
  return (c & 8) ? -v : v;
}
static uint32_t f_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}
static float bits_f(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static uint16_t h_f2bf(float f) {  // round to nearest even (no NaN in this probe)
  uint32_t u = f_bits(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
static float h_bfround(float f) { return bits_f((uint32_t)h_f2bf(f) << 16); }
static uint32_t h_key(uint16_t b) { return (b & 0x8000) ? (~b & 0xFFFF) : (b | 0x8000); }

// exact dot product of a query (64 B + scales) with a key (64 B + scales), as the MMA computes it when exact
static double h_dot(const uint8_t* q, uint32_t qs, const uint8_t* k, uint32_t ks) {
  double s = 0;
  for (int b = 0; b < 4; ++b) {  // a block of 32 elements shares one scale on each side
    double sb = 0;
    for (int j = 32 * b; j < 32 * b + 32; ++j) {
      uint32_t qc = (q[j / 2] >> (4 * (j & 1))) & 15, kc = (k[j / 2] >> (4 * (j & 1))) & 15;
      sb += (double)h_e2m1(qc) * h_e2m1(kc);
    }
    int qe = (int)((qs >> (8 * b)) & 255) - 127, ke = (int)((ks >> (8 * b)) & 255) - 127;
    s += sb * ldexp(1.0, qe + ke);
  }
  return s;
}

struct Case {
  int rows, columns, kmax;
  std::vector<int> starts, lengths, offsets;
  bool equal_keys = false;  // every key identical: one threshold bin holds every key
};

// The reference key of (row r, column c): per lane t, heads nt*8+2t and +1 over nt = 0..3, then (a0+a1)+(a2+a3).
static uint32_t h_score_key(const std::vector<uint8_t>& q, const std::vector<uint32_t>& qs, const std::vector<float>& w,
                            const std::vector<uint8_t>& k, const std::vector<uint32_t>& ks, int r, int c) {
  float lane[4];
  for (int t = 0; t < 4; ++t) {
    float acc = 0.f;
    for (int nt = 0; nt < 4; ++nt)
      for (int e = 0; e < 2; ++e) {
        int h = nt * 8 + 2 * t + e, qi = r * 32 + h;
        float d = (float)h_dot(&q[(size_t)qi * 64], qs[qi], &k[(size_t)c * 64], ks[c]);
        float x = h_bfround(d);
        x = x > 0.f ? x : 0.f;
        volatile float p = x * w[qi];
        acc = acc + h_bfround(p);
      }
    lane[t] = acc;
  }
  volatile float a01 = lane[0] + lane[1], a23 = lane[2] + lane[3];
  volatile float s = a01 + a23;
  return h_key(h_f2bf(s));
}

static int run_case(const Case& cs, std::mt19937& rng, int scale_spread, bool verbose) {
  const int R = cs.rows, C = cs.columns, kmax = cs.kmax;
  std::vector<uint8_t> q((size_t)R * 32 * 64), k((size_t)C * 64);
  std::vector<uint32_t> qs((size_t)R * 32), ks(C);
  std::vector<float> w((size_t)R * 32);
  std::uniform_int_distribution<int> byte(0, 255), ex(127 - scale_spread, 127 + scale_spread);
  std::uniform_real_distribution<float> wd(-1.f, 1.f);
  for (auto& b : q) b = (uint8_t)byte(rng);
  for (auto& b : k) b = (uint8_t)byte(rng);
  if (cs.equal_keys)
    for (int c = 1; c < C; ++c) memcpy(&k[(size_t)c * 64], &k[0], 64);
  auto sc = [&]() { return (uint32_t)ex(rng) | (uint32_t)ex(rng) << 8 | (uint32_t)ex(rng) << 16 | (uint32_t)ex(rng) << 24; };
  for (auto& s : qs) s = sc();
  for (auto& s : ks) s = sc();
  if (cs.equal_keys)
    for (int c = 1; c < C; ++c) ks[c] = ks[0];
  for (auto& x : w) x = ldexp((float)h_bfround(wd(rng)), -6);  // bf16 projection times 2^-6, as the engine's
  // Ties: copy some keys so equal scores occur (the cut's tie rule is then exercised).
  for (int c = 64; c < C; c += 97) memcpy(&k[(size_t)c * 64], &k[(size_t)(c - 64) * 64], 64), ks[c] = ks[c - 64];

  const int stride = (C + 15) / 16 * 16;
  uint8_t *dq, *dk;
  uint32_t *dqs, *dks;
  float* dw;
  int *dst, *dlen, *doff, *dout;
  uint16_t* dkeys;
  CK(cudaMalloc(&dq, q.size()));
  CK(cudaMalloc(&dk, k.size()));
  CK(cudaMalloc(&dqs, qs.size() * 4));
  CK(cudaMalloc(&dks, ks.size() * 4));
  CK(cudaMalloc(&dw, w.size() * 4));
  CK(cudaMalloc(&dst, R * 4));
  CK(cudaMalloc(&dlen, R * 4));
  CK(cudaMalloc(&doff, R * 4));
  CK(cudaMalloc(&dout, (size_t)R * kmax * 4));
  CK(cudaMalloc(&dkeys, (size_t)R * stride * 2));
  CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dk, k.data(), k.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dqs, qs.data(), qs.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dks, ks.data(), ks.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dst, cs.starts.data(), R * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dlen, cs.lengths.data(), R * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(doff, cs.offsets.data(), R * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(dkeys, 0xAB, (size_t)R * stride * 2));  // stale data outside windows must not matter
  dim3 sg(96, (R + 7) / 8);
  ca_idx16_score<<<sg, 128>>>(dq, dqs, dw, dk, dks, dst, dlen, R, C, dkeys, stride, nullptr, 0);
  ca_idx16_select<<<R, 1024>>>(dkeys, stride, dst, dlen, doff, kmax, dout, kmax);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<uint16_t> keys((size_t)R * stride);
  std::vector<int> out((size_t)R * kmax);
  CK(cudaMemcpy(keys.data(), dkeys, keys.size() * 2, cudaMemcpyDeviceToHost));
  // f32 mode: the same scores as f32 values (every window column), compared below with the keys' bf16 values
  float* dfs;
  CK(cudaMalloc(&dfs, (size_t)R * stride * 4));
  CK(cudaMemset(dfs, 0, (size_t)R * stride * 4));
  ca_idx16_score<<<sg, 128>>>(dq, dqs, dw, dk, dks, dst, dlen, R, C, nullptr, 0, dfs, stride);
  CK(cudaDeviceSynchronize());
  std::vector<float> fs((size_t)R * stride);
  CK(cudaMemcpy(fs.data(), dfs, fs.size() * 4, cudaMemcpyDeviceToHost));
  cudaFree(dfs);
  int bad_f32 = 0;
  for (int r = 0; r < R; ++r)
    for (int c = cs.starts[r]; c < cs.starts[r] + cs.lengths[r]; ++c) {
      const size_t j = (size_t)r * stride + (c - cs.starts[r]);
      uint16_t b = (uint16_t)(f_bits(fs[j]) >> 16);
      if ((f_bits(fs[j]) & 0xffff) || h_key(b) != keys[j]) ++bad_f32;
    }
  CK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
  int bad_keys = 0, bad_rows = 0;
  for (int r = 0; r < R; ++r) {
    const int lo = cs.starts[r], hi = lo + cs.lengths[r];
    std::vector<std::pair<uint32_t, int>> ref;
    // score check on a sample of columns (every column when small)
    const int step = (hi - lo) > 4096 ? 7 : 1;
    for (int c = lo; c < hi; ++c) {
      uint32_t kk = h_score_key(q, qs, w, k, ks, r, c);
      if ((c - lo) % step == 0 && keys[(size_t)r * stride + (c - lo)] != kk) {
        if (bad_keys < 5 && verbose)
          fprintf(stderr, "  key r=%d c=%d gpu=%04x ref=%04x\n", r, c, keys[(size_t)r * stride + (c - lo)], kk);
        ++bad_keys;
      }
      ref.push_back({kk, c});
    }
    std::stable_sort(ref.begin(), ref.end(), [](auto& a, auto& b) { return a.first > b.first; });
    const int K = std::min(kmax, (int)ref.size());
    std::vector<int> sel;
    for (int i = 0; i < K; ++i) sel.push_back(ref[i].second);
    std::sort(sel.begin(), sel.end());
    bool ok = true;
    for (int i = 0; i < kmax; ++i) {
      int want = i < K ? cs.offsets[r] + (sel[i] - lo) : -1;
      if (out[(size_t)r * kmax + i] != want) ok = false;
    }
    if (!ok) {
      if (bad_rows < 3 && verbose) fprintf(stderr, "  selection differs: row %d (window %d..%d)\n", r, lo, hi);
      ++bad_rows;
    }
  }
  // The fused launch (ca_idx16_dense, up to kDenseRows rows) must write the same keys and the same selection.
  int dense_bad = -1;
  if (R <= kDenseRows) {
    uint16_t* dkeys2;
    int* dout2;
    uint32_t* dscr;
    CK(cudaMalloc(&dkeys2, (size_t)R * stride * 2));
    CK(cudaMalloc(&dout2, (size_t)R * kmax * 4));
    CK(cudaMalloc(&dscr, (2 + (size_t)R * 2048) * 4));
    CK(cudaMemset(dkeys2, 0xAB, (size_t)R * stride * 2));
    CK(cudaMemset(dout2, 0x5A, (size_t)R * kmax * 4));
    CK(cudaMemset(dscr, 0, (2 + (size_t)R * 2048) * 4));
    CK(cudaFuncSetAttribute(ca_idx16_dense, cudaFuncAttributeMaxDynamicSharedMemorySize, kDenseSmem));
    dense_bad = 0;
    for (int rep = 0; rep < 2; ++rep) {  // twice: the scratch must be left zero for the next launch
      ca_idx16_dense<<<48, 1024, kDenseSmem>>>(dq, dqs, dw, dk, dks, dst, dlen, doff, R, C, dkeys2, stride, kmax, dout2,
                                               kmax, dscr);
      CK(cudaGetLastError());
      CK(cudaDeviceSynchronize());
      std::vector<uint16_t> keys2((size_t)R * stride);
      std::vector<int> out2((size_t)R * kmax);
      CK(cudaMemcpy(keys2.data(), dkeys2, keys2.size() * 2, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(out2.data(), dout2, out2.size() * 4, cudaMemcpyDeviceToHost));
      for (int r = 0; r < R; ++r) {
        bool row_ok = true;
        for (int j = 0; j < cs.lengths[r]; ++j)
          row_ok &= keys2[(size_t)r * stride + j] == keys[(size_t)r * stride + j];
        for (int i = 0; i < kmax; ++i) row_ok &= out2[(size_t)r * kmax + i] == out[(size_t)r * kmax + i];
        if (!row_ok) {
          if (dense_bad < 3 && verbose) fprintf(stderr, "  fused launch differs: row %d (launch %d)\n", r, rep);
          ++dense_bad;
        }
      }
    }
    std::vector<uint32_t> scr(2 + (size_t)R * 2048);
    CK(cudaMemcpy(scr.data(), dscr, scr.size() * 4, cudaMemcpyDeviceToHost));
    for (auto v : scr) dense_bad += v != 0;  // left zero
    cudaFree(dkeys2), cudaFree(dout2), cudaFree(dscr);
  }
  cudaFree(dq), cudaFree(dk), cudaFree(dqs), cudaFree(dks), cudaFree(dw), cudaFree(dst), cudaFree(dlen),
      cudaFree(doff), cudaFree(dout), cudaFree(dkeys);
  printf("{\"case\": \"rows %d columns %d kmax %d spread %d\", \"bad_keys\": %d, \"bad_rows\": %d, \"bad_f32\": %d, \"dense_bad\": %d}\n",
         R, C, kmax, scale_spread, bad_keys, bad_rows, bad_f32, dense_bad);
  return bad_keys + bad_rows + bad_f32 + (dense_bad > 0 ? dense_bad : 0);
}

static Case causal(int rows, int columns, int kmax) {
  Case c{rows, columns, kmax, {}, {}, {}, false};
  for (int r = 0; r < rows; ++r) {
    c.starts.push_back(0);
    c.lengths.push_back(columns - (rows - 1 - r));
    c.offsets.push_back(0);
  }
  return c;
}

static void timing(int rows, int columns) {
  const int kmax = 512, stride = (columns + 15) / 16 * 16;
  uint8_t *dq, *dk;
  uint32_t *dqs, *dks;
  float* dw;
  int *dst, *dlen, *doff, *dout;
  uint16_t* dkeys;
  std::mt19937 rng(7);
  std::vector<uint8_t> hq((size_t)rows * 32 * 64), hk((size_t)columns * 64);
  for (auto& b : hq) b = rng() & 255;
  for (auto& b : hk) b = rng() & 255;
  std::vector<uint32_t> hqs((size_t)rows * 32, 0x7f7f7f7fu), hks(columns, 0x7e7f807fu);
  std::vector<float> hw((size_t)rows * 32, 0x1p-6f);
  Case c = causal(rows, columns, kmax);
  CK(cudaMalloc(&dq, hq.size()));
  CK(cudaMalloc(&dk, hk.size()));
  CK(cudaMalloc(&dqs, hqs.size() * 4));
  CK(cudaMalloc(&dks, hks.size() * 4));
  CK(cudaMalloc(&dw, hw.size() * 4));
  CK(cudaMalloc(&dst, rows * 4));
  CK(cudaMalloc(&dlen, rows * 4));
  CK(cudaMalloc(&doff, rows * 4));
  CK(cudaMalloc(&dout, (size_t)rows * kmax * 4));
  CK(cudaMalloc(&dkeys, (size_t)rows * stride * 2));
  CK(cudaMemcpy(dq, hq.data(), hq.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dk, hk.data(), hk.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dqs, hqs.data(), hqs.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dks, hks.data(), hks.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dw, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dst, c.starts.data(), rows * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dlen, c.lengths.data(), rows * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(doff, c.offsets.data(), rows * 4, cudaMemcpyHostToDevice));
  dim3 sg(96, (rows + 7) / 8);
  cudaEvent_t e0, e1, e2;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventCreate(&e2));
  for (int i = 0; i < 5; ++i) {
    ca_idx16_score<<<sg, 128>>>(dq, dqs, dw, dk, dks, dst, dlen, rows, columns, dkeys, stride, nullptr, 0);
    ca_idx16_select<<<rows, 1024>>>(dkeys, stride, dst, dlen, doff, kmax, dout, kmax);
  }
  const int n = 50;
  float ts = 0, tt = 0;
  for (int i = 0; i < n; ++i) {
    float a, b;
    CK(cudaEventRecord(e0));
    ca_idx16_score<<<sg, 128>>>(dq, dqs, dw, dk, dks, dst, dlen, rows, columns, dkeys, stride, nullptr, 0);
    CK(cudaEventRecord(e1));
    ca_idx16_select<<<rows, 1024>>>(dkeys, stride, dst, dlen, doff, kmax, dout, kmax);
    CK(cudaEventRecord(e2));
    CK(cudaEventSynchronize(e2));
    CK(cudaEventElapsedTime(&a, e0, e1));
    CK(cudaEventElapsedTime(&b, e1, e2));
    ts += a, tt += b;
  }
  CK(cudaGetLastError());
  float td = -1.f;
  if (rows <= kDenseRows) {
    uint32_t* dscr;
    CK(cudaMalloc(&dscr, (2 + (size_t)rows * 2048) * 4));
    CK(cudaMemset(dscr, 0, (2 + (size_t)rows * 2048) * 4));
    CK(cudaFuncSetAttribute(ca_idx16_dense, cudaFuncAttributeMaxDynamicSharedMemorySize, kDenseSmem));
    for (int i = 0; i < 5; ++i)
      ca_idx16_dense<<<48, 1024, kDenseSmem>>>(dq, dqs, dw, dk, dks, dst, dlen, doff, rows, columns, dkeys, stride, kmax,
                                               dout, kmax, dscr);
    td = 0;
    for (int i = 0; i < n; ++i) {
      float a;
      CK(cudaEventRecord(e0));
      ca_idx16_dense<<<48, 1024, kDenseSmem>>>(dq, dqs, dw, dk, dks, dst, dlen, doff, rows, columns, dkeys, stride, kmax,
                                               dout, kmax, dscr);
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      CK(cudaEventElapsedTime(&a, e0, e1));
      td += a;
    }
    CK(cudaGetLastError());
    td = 1000.f * td / n;
    cudaFree(dscr);
  }
  printf("{\"timing\": \"rows %d columns %d\", \"score_us\": %.1f, \"select_us\": %.1f, \"dense_us\": %.1f}\n", rows,
         columns, 1000.f * ts / n, 1000.f * tt / n, td);
  cudaFree(dq), cudaFree(dk), cudaFree(dqs), cudaFree(dks), cudaFree(dw), cudaFree(dst), cudaFree(dlen),
      cudaFree(doff), cudaFree(dout), cudaFree(dkeys);
}

int main(int argc, char** argv) {
  std::mt19937 rng(12345);
  int bad = 0;
  bad += run_case(causal(1, 300, 512), rng, 2, true);    // shorter than k: every key, in order
  bad += run_case(causal(1, 2048, 512), rng, 2, true);
  bad += run_case(causal(6, 5000, 512), rng, 2, true);
  bad += run_case(causal(16, 8192, 512), rng, 2, true);  // two row groups
  {
    Case c{4, 9000, 512, {0, 1234, 4000, 8999}, {3000, 4000, 5000, 1}, {0, 70000, 5, 9}};  // windows and offsets
    bad += run_case(c, rng, 2, true);
  }
  bad += run_case(causal(2, 40000, 512), rng, 2, true);
  {  // the fused launch's fallback: more keys in the threshold bin than its shared candidates hold
    Case e = causal(3, 20000, 512);
    e.equal_keys = true;
    bad += run_case(e, rng, 2, true);
  }
  bad += run_case(causal(8, 131072, 512), rng, 3, true);  // a full group of eight rows at 131K
  printf("{\"correctness_failures\": %d}\n", bad);
  if (argc > 1 && !strcmp(argv[1], "--no-timing")) return bad ? 1 : 0;
  for (int rows : {1, 2, 6, 16})
    for (int columns : {8192, 16384, 65536, 131072, 262144}) timing(rows, columns);
  return bad ? 1 : 0;
}
