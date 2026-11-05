#!/usr/bin/env python3
"""AOT compact canonical-weight WO-A for M16/M32; BF16 operands, FP32 sums.

Kernel adapted from Mia e642f230de3d... wo_a_w8.py (AGPL-3.0), same
operand reconstruction as the already qualified small-row path. The final
reduction retains FP32 partials until one BF16 output rounding.
"""
import ast,hashlib,json,os,re,sys,types
from pathlib import Path
os.environ['TRITON_PTXAS_BLACKWELL_PATH']='/usr/local/cuda/bin/ptxas'
import triton,triton.language as tl
assert triton.__version__=='3.7.1', 'use the pinned Triton compiler'
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

@triton.jit
def reduce_partials(P,Y,E:tl.constexpr,SK:tl.constexpr):
    i=tl.program_id(0)*256+tl.arange(0,256)
    k=tl.arange(0,SK)
    v=tl.load(P+k[:,None]*E+i[None,:],i[None,:]<E,0.)
    tl.store(Y+i,tl.sum(v,0).to(tl.bfloat16),i<E)

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
assert not(out/'manifest.json').exists()
os.environ['TRITON_CACHE_DIR']=str(out/'compile-cache')
source=Path('/data/third-party/mia/adapter/wo_a_w8.py')
sha=hashlib.sha256(source.read_bytes()).hexdigest()
assert sha=='e642f230de3d2f2da94ff19e8d6f0b5565a77450fb012851be1c61931cd53901'
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_wo_a_mid_kernel']
assert len(nodes)==1
mod=types.ModuleType('native_mid_woa');mod.__file__=str(source)
mod.__dict__.update(triton=triton,tl=tl);sys.modules[mod.__name__]=mod
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),mod.__dict__)
entries=[]
for rows,sk,bm,bn,warps in [(16,4,16,64,8),(32,2,32,32,4)]:
    specs=[(f'woa_mid_partial_m{rows}',mod._wo_a_mid_kernel,
        {'X':'*bf16','W8':'*fp8e4nv','S':'*u8','Y':'*fp32'},
        dict(M=rows,SXM=8192,SPLIT_K=sk,BM=bm,BN=bn,BK=128,G=2),
        dict(rows_div=1,x_mul=0,x_add=1024//bn,y=2,z=sk),warps,3),
        (f'woa_mid_reduce_m{rows}',reduce_partials,{'P':'*fp32','Y':'*bf16'},
        dict(E=rows*2048,SK=sk),dict(rows_div=1,x_mul=8,x_add=0,y=1,z=1),4,1)]
    for name,fn,signature,constexprs,grid,nw,ns in specs:
        attrs={(fn.arg_names.index(k),):[['tt.divisibility',16]]for k in signature}
        kernel=triton.compile(ASTSource(fn=fn,signature=signature,constexprs=constexprs,attrs=attrs),
            target=GPUTarget('cuda',121,32),options=dict(num_warps=nw,num_stages=ns))
        meta=kernel.metadata._asdict();cubin=kernel.asm['cubin'];ptx=kernel.asm['ptx']
        assert meta.get('global_scratch_size',0)==0 and meta.get('profile_scratch_size',0)==0
        params=[];decl=re.search(r'\.entry\s+\w+\s*\((.*?)\)',ptx,re.S);assert decl
        for p in decl[1].split(','):
            kind=re.search(r'\.param\s+\.(u\d+|s\d+|f\d+|b\d+)\b',p)
            ident=re.search(r'([A-Za-z0-9_]+)\s*$',p);assert kind and ident
            params.append((kind[1],ident[1]))
        args=[dict(name=n,kind=signature[n])for n in fn.arg_names if n in signature]
        assert len(params)==len(args)+2 and all(t=='u64'for t,_ in params)
        (out/(name+'.cubin')).write_bytes(cubin);(out/(name+'.ptx')).write_text(ptx)
        entries.append(dict(name=name,symbol=meta['name'],architecture=121,physical_sms=48,arguments=args,
            metadata=meta,grid=grid,ptx_parameters=params,constexprs=constexprs,source_sha256=sha,
            artifact_sha256=hashlib.sha256(cubin).hexdigest()))
        print(json.dumps(dict(compiled=name,bytes=len(cubin))),flush=True)
(out/'manifest.json').write_text(json.dumps(dict(entries=entries,
    source='Mia compact WO-A low-level kernel; Rust-owned shared scratch and execution',
    source_sha256=sha,license='AGPL-3.0',
    source_url='https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks',
    numerical_contract='Canonical FP8/E8M0 reconstructed exactly to BF16 operands; BF16 MMA, FP32 split partials and reduction; one BF16 output rounding.',
    workspace=dict(m16_bytes=4*16*2048*4,m32_bytes=2*32*2048*4)),indent=2,default=str)+'\n')
