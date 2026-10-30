#!/usr/bin/env python3
"""AOT selected low-level lossless W8 grouped WO-A and its MXFP8 epilogue."""
import ast,hashlib,json,os,re,sys,types
from pathlib import Path
W=Path('/data/build');REF=Path('/data/third-party');OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'manifest.json').exists()
os.environ['TRITON_CACHE_DIR']=str(W/'triton-build-cache')
os.environ['TRITON_PTXAS_BLACKWELL_PATH']='/usr/local/cuda/bin/ptxas'
import triton,triton.language as tl
assert triton.__version__ == '3.7.1', 'use the pinned Triton compiler'
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

def functions(path,name,globals=None):
    s=path.read_text();nodes=[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and any(isinstance(d,ast.Attribute)and isinstance(d.value,ast.Name)and d.value.id=='triton'and d.attr=='jit'for d in n.decorator_list)]
    mod=types.ModuleType(name);mod.__file__=str(path);mod.__dict__.update(triton=triton,tl=tl,**(globals or {}));sys.modules[name]=mod
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),mod.__dict__);return mod,hashlib.sha256(s.encode()).hexdigest()
norm,_=functions(REF/'sglang/kernels/ops/layernorm/mxfp8_epilogue.py','canada_woa_quant')
base,bsha=functions(REF/'sglang/kernels/ops/attention/dsv4/wo_a_bf16.py','canada_woa_base',{'ue8m0_scale':norm.ue8m0_scale})
w8,wsha=functions(REF/'mia/adapter/wo_a_w8.py','canada_woa_w8')
entries=[]
for rows in range(2,9):
    specs=[(f'woa_w8_partial_m{rows}',w8._wo_a_partial_w8,{'X':'*bf16','W8':'*fp8e4nv','S':'*u8','P':'*fp32'},
       {'M':rows,'SX':8192,'G':2},{'rows_div':1,'x_mul':0,'x_add':16,'y':2,'z':8},wsha),
      (f'woa_reduce_quant_m{rows}',base._wo_a_reduce_quant,{'P':'*fp32','Q':'*fp8e4nv','S':'*u8'},
       {'M':rows},{'rows_div':1,'y':8,'z':1},bsha)]
    for name,fn,signature,constexprs,grid,source_sha in specs:
        attrs={(fn.arg_names.index(k),):[['tt.divisibility',16]]for k,v in signature.items()if v.startswith('*')}
        options={'num_warps':4}
        if 'partial'in name:options['num_stages']=3
        k=triton.compile(ASTSource(fn=fn,signature=signature,constexprs=constexprs,attrs=attrs),target=GPUTarget('cuda',121,32),options=options)
        meta=k.metadata._asdict();cubin=k.asm['cubin'];ptx=k.asm['ptx'];assert meta.get('global_scratch_size',0)==0 and meta.get('profile_scratch_size',0)==0
        (OUT/(name+'.cubin')).write_bytes(cubin);(OUT/(name+'.ptx')).write_text(ptx)
        d=re.search(r'\.entry\s+\w+\s*\((.*?)\)',ptx,re.S);assert d
        params=[]
        for p in d[1].split(','):
            kind=re.search(r'\.param\s+\.(u\d+|s\d+|f\d+|b\d+)\b',p);ident=re.search(r'([A-Za-z0-9_]+)\s*$',p);assert kind and ident;params.append((kind[1],ident[1]))
        args=[{'name':n,'kind':signature[n]}for n in fn.arg_names if n in signature]
        assert len(params)==len(args)+2 and all(t=='u64'for t,_ in params[-2:])
        entries.append({'name':name,'symbol':meta['name'],'architecture':121,'physical_sms':48,'arguments':args,'metadata':meta,'grid':grid,
          'ptx_parameters':params,'source_sha256':source_sha,'constexprs':constexprs,'artifact_sha256':hashlib.sha256(cubin).hexdigest()})
        print(json.dumps({'compiled':name,'bytes':len(cubin)}),flush=True)
(OUT/'manifest.json').write_text(json.dumps({'source':'Mia low-level lossless W8 WO-A plus SGLang MXFP8 epilogue',
 'recipe_source':'https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks','license':'AGPL-3.0 AND Apache-2.0',
 'mia_license':'AGPL-3.0','sglang_license':'Apache-2.0','mia_license_source':'https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/64b02dea8681fb1d8e3a659ea432a8e9e9ba3995/LICENSE',
 'mia_source_sha256':wsha,'sglang_source_sha256':bsha,'entries':entries},indent=2,default=str)+'\n')
