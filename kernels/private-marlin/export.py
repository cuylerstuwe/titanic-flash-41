"""Export the pinned CUDA kernels through the driver ABI used by Rust."""
import hashlib,json,re,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;b=p/'build';out=p/'native-artifact';assert not out.exists();out.mkdir()
text='''// Instantiation only. Original upstream kernel and notices: ../upstream/.\n#include "kernel.h"\n#include "marlin_template.h"\n#include "repack.cuh"\n#include "../native_quantize.cu"\nnamespace marlin {\ntemplate __global__ void Marlin<vllm::kBFloat16.id(),vllm::kFE2M1f.id(),vllm::kBFloat16.id(),vllm::kFE4M3fn.id(),128,1,4,8,true,4,1,false>(MARLIN_KERNEL_PARAMS);\ntemplate __global__ void gptq_marlin_repack_kernel<repack_threads,4,false>(uint32_t const* __restrict__,uint32_t* __restrict__,int,int);\n}\n'''
src=b/'native.cu';src.write_text(text)
base=['/usr/local/cuda/bin/nvcc','-std=c++17','--expt-relaxed-constexpr','-O3','-arch=sm_121','-I'+str(b),str(src)]
for mode,suffix in [('--cubin','cubin'),('--ptx','ptx')]:subprocess.run(base+[mode,'-o',str(out/f'head.{suffix}')],check=True,timeout=120)
ptx=(out/'head.ptx').read_text()
functions=[]
for match in re.finditer(r'\.visible \.entry\s+(\S+)\s*\((.*?)\)',ptx,re.S):
    name,params=match.groups()
    params=[re.search(r'\.param\s+\.(\w+)',x).group(1)for x in params.split(',')]
    key='projection'if 'MarlinI'in name else'repack'if 'gptq_marlin_repack_kernel'in name else name.removeprefix('nvfp4_head_')
    functions.append(dict(name=key,symbol=name,parameters=params))
assert {x['name']for x in functions}=={'projection','repack','amax','quantize','publish'},functions
manifest=dict(shape=[5,32320,5120],cuda_arch=121,sm_count=48,projection_threads=128,repack_threads=256,shared_bytes=101376,
    artifact='head.cubin',artifact_sha256=hashlib.sha256((out/'head.cubin').read_bytes()).hexdigest(),
    functions=functions,source_revision='784d1fff0c491b9add10ecd732de8f707a7e7dbe',
    upstream_license='Apache-2.0',source_manifest_sha256=hashlib.sha256((p/'source-manifest.json').read_bytes()).hexdigest(),
    quantizer_sha256=hashlib.sha256((p/'native_quantize.cu').read_bytes()).hexdigest(),
    contract='Private proposal head only: GPU E2M1 RNE, E4M3/16 positive scales, global FP32. BF16 inputs/output, FP32 Marlin reduction, no atomic. Exact FP32 conversion of BF16 output for actual q. Target unchanged.')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest),flush=True)

