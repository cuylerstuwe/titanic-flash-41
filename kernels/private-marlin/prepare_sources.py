"""Make a narrow C ABI adapter for pinned upstream Marlin CUDA kernels.
Original sources/notices stay intact under upstream/. No PyTorch/vLLM runtime.
"""
import hashlib,json,pathlib
p=pathlib.Path(__file__).resolve().parent;u=p/'upstream';b=p/'build';b.mkdir(exist_ok=True)
# Only the host assertion dependency changes; the kernel arithmetic is intact.
core=(u/'core/scalar_type.hpp').read_text()
core=core.replace('#include <torch/headeronly/util/Exception.h>','#include <stdexcept>\n#define STD_TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("Marlin scalar check: " #c); } while (0)')
(b/'core').mkdir(exist_ok=True);(b/'core/scalar_type.hpp').write_text(core)
for name in ['marlin.cuh','kernel.h','marlin_dtypes.cuh','marlin_template.h','dequant.h','marlin_mma.h']:
    (b/name).write_bytes((u/name).read_bytes())
repack=(u/'gptq_marlin_repack.cu').read_text()
repack='#include "marlin.cuh"\n'+repack[repack.index('namespace marlin {'):repack.index('#define CALL_IF')]
(b/'repack.cuh').write_text('// Adapted only by removing the Torch host wrapper; see ../upstream/LICENSE.\n'+repack)
wrapper=r'''
// Component-only C ABI adapter. Upstream Marlin kernels pinned in ../upstream.
// Copyright contributors to the vLLM project / Marlin original authors.
// SPDX-License-Identifier: Apache-2.0
#include "kernel.h"
#include "marlin_template.h"
#include "repack.cuh"
static int probe_sms=0, probe_smem=0;
template<bool Small> auto probe_kernel() {
  // Same third small-batch shape selected by upstream for N=32320, K=5120.
  return marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
      vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 128, 1, 4, 8,
      Small, 4, 1, false>;
}
extern "C" int probe_init() {
  int dev=0;auto e=cudaGetDevice(&dev);if(e)return e;
  if((e=cudaDeviceGetAttribute(&probe_sms,cudaDevAttrMultiProcessorCount,dev)))return e;
  if((e=cudaDeviceGetAttribute(&probe_smem,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev)))return e;
  if((e=cudaFuncSetAttribute(probe_kernel<true>(),cudaFuncAttributeMaxDynamicSharedMemorySize,probe_smem)))return e;
  if((e=cudaFuncSetAttribute(probe_kernel<false>(),cudaFuncAttributeMaxDynamicSharedMemorySize,probe_smem)))return e;
  return cudaFuncSetAttribute(marlin::gptq_marlin_repack_kernel<marlin::repack_threads,4,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,probe_smem);
}
extern "C" int probe_get_sms(){return probe_sms;}
extern "C" int probe_get_smem(){return probe_smem;}
extern "C" int probe_repack(const void *q,void *out,int n,int k,void *s) {
  if(!probe_sms||n%64||k%16)return cudaErrorInvalidValue;
  marlin::gptq_marlin_repack_kernel<marlin::repack_threads,4,false>
    <<<probe_sms,marlin::repack_threads,probe_smem,(cudaStream_t)s>>>
    ((const uint32_t*)q,(uint32_t*)out,k,n);
  return cudaGetLastError();
}
extern "C" int probe_gemm(const void *a,const void *w,void *c,void *tmp,
    const void *scales,const void *global,void *locks,int m,int n,int k,void *s) {
  if(!probe_sms||n%64||k%128||m<1||m>16)return cudaErrorInvalidValue;
  auto kernel=m<=8?probe_kernel<true>():probe_kernel<false>();
  kernel<<<probe_sms,128,probe_smem,(cudaStream_t)s>>>(
      (const int4*)a,(const int4*)w,(int4*)c,(int4*)tmp,
      nullptr,nullptr,(const int4*)scales,(const float*)global,nullptr,
      m,n,k,k,(int*)locks,false,false,true,probe_smem);
  return cudaGetLastError();
}
'''
(b/'adapter.cu').write_text(wrapper)
manifest=dict(repository='vllm-project/vllm',revision='784d1fff0c491b9add10ecd732de8f707a7e7dbe',license='Apache-2.0',
    purpose='Private drafter-head component only, canonical BF16 target unchanged. Raw C ABI avoids adding an existing inference engine.',
    numerical_contract='NVFP4 E2M1 weights, positive E4M3 scale/16 and FP32 global scale; BF16 activation and output; FP32 global reduction, no atomic add. Output converted to FP32 for existing q path.',
    changes=['Remove Torch host wrappers, retain CUDA arithmetic.','Replace scalar host assertion include with std::runtime_error.','Instantiate only upstream 128-thread N64 K128 M8/M16 BF16 NVFP4 kernel; upstream selector picks this for our unpadded shard.'],
    upstream_sha256={str(f.relative_to(u)):hashlib.sha256(f.read_bytes()).hexdigest()for f in sorted(u.rglob('*'))if f.is_file()})
(p/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

