"""Actual M6 coefficient stats -> reduction chain, with PDL scheduling only.

Dependency contract: NVIDIA CUDA Programmatic Dependent Launch guide.
The existing Apache-2.0-attributed AOT primitives remain the arithmetic source.
Only PTX dependency-control instructions are inserted; no model engine runs.
"""
import ctypes as C,hashlib,json,re,statistics,subprocess
from pathlib import Path
OUT=Path('/probe');assert not(OUT/'results.json').exists()
ROOT=Path('/scratch/primitive-exports');CUDA=Path('/usr/local/cuda')
subprocess.run(['g++','-std=c++17','-shared','-fPIC','-O2','-I'+str(CUDA/'include'),
    str(OUT/'launch.cpp'),'-L'+str(CUDA/'lib64/stubs'),'-lcuda','-o',str(OUT/'launch.so')],check=True)
import triton
assembler_candidates=[CUDA/'bin/ptxas',Path(triton.__file__).parent/'backends/nvidia/bin/ptxas',Path(triton.__file__).parent/'backends/nvidia/bin/ptxas-blackwell']
versions=[]
for path in assembler_candidates:
    if path.exists():
        version=subprocess.check_output([str(path),'--version'],text=True)
        match=re.search(r'release (\d+)\.(\d+)',version);assert match
        versions.append((tuple(map(int,match.groups())),path,version))
_,ASSEMBLER,compiler=max(versions,key=lambda v:v[0])
assert max(v[0]for v in versions)>=(13,1),'PTX9.1 needs the matching bundled assembler'

specs=[('post','triton-hc-v2','hc_stats_mb8_s2'),
       ('norm','triton-hc-v2','hc_reduce80')]
entries={};artifacts={};provenance={}
for op,directory,name in specs:
    root=ROOT/directory;manifest=json.loads((root/'manifest.json').read_text())
    entry=next(e for e in manifest['entries']if e['name']==name);entries[op]=entry
    assert entry['architecture']==121 and entry['metadata']['num_ctas']==1
    raw=(root/(name+'.cubin')).read_bytes();assert hashlib.sha256(raw).hexdigest()==entry['artifact_sha256']
    ptx=(root/(name+'.ptx')).read_text();assert 'griddepcontrol'not in ptx
    assert len(re.findall(r'\.entry\s+',ptx))==1
    provenance[op]=dict(source=manifest.get('source'),source_commit=manifest.get('source_commit'),
        license=manifest.get('license'),original_cubin=entry['artifact_sha256'],
        original_ptx=hashlib.sha256(ptx.encode()).hexdigest(),entry=entry)
    artifacts[op,'native']=raw
    for mode in ['plain','early','late']:
        text=ptx
        if mode!='plain':
            insert='\n\tgriddepcontrol.wait;\n'
            if mode=='early':insert+='\tgriddepcontrol.launch_dependents;\n'
            text,count=re.subn(r'(?m)^[ \t]*// %bb\.0:[^\n]*$',lambda m:m[0]+insert,text,count=1)
            assert count==1,'unrecognized PTX entry block'
            if mode=='late':
                text,count=re.subn(r'(?m)^(\s*)ret;',r'\1griddepcontrol.launch_dependents;\n\1ret;',text)
                assert count>0,'missing PTX return'
        path=OUT/f'{op}-{mode}.ptx';path.write_text(text)
        cubin=path.with_suffix('.cubin')
        subprocess.run([str(ASSEMBLER),'-arch=sm_121a','-O3',str(path),'-o',str(cubin)],check=True)
        artifacts[op,mode]=cubin.read_bytes()

import torch
torch.set_num_threads(2);torch.manual_seed(20261031)
assert torch.cuda.get_device_capability()==(12,1)
owner=torch.empty(1,device='cuda');stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
P=C.c_void_p;driver=C.CDLL('libcuda.so.1');shim=C.CDLL(str(OUT/'launch.so'))
def api(name,args):
    f=getattr(driver,name);f.argtypes=args;f.restype=C.c_int;return f
load=api('cuModuleLoadData',[C.POINTER(P),P]);get=api('cuModuleGetFunction',[C.POINTER(P),P,C.c_char_p])
unload=api('cuModuleUnload',[P]);attribute=api('cuFuncSetAttribute',[P,C.c_int,C.c_int])
launch=shim.launch_pair_op;launch.argtypes=[P,C.c_uint,C.c_uint,C.c_uint,C.c_uint,P,C.POINTER(P),C.c_int];launch.restype=C.c_int
modules=[];ops={}
for key,raw in artifacts.items():
    mod=P();fn=P();blob=C.create_string_buffer(raw);assert load(C.byref(mod),C.cast(blob,P))==0
    e=entries[key[0]];assert get(C.byref(fn),mod,e['symbol'].encode())==0
    if e['metadata']['shared']>48*1024:assert attribute(fn,8,e['metadata']['shared'])==0
    modules.append(mod);ops[key]=fn
variants=[('native','native',0),('plain','plain',0),('early','early',1),('late','early',1)]
checks=[];timings=[];guard=64
for rows in [3,6]:
    x=torch.randn(40,rows,20480,device='cuda').bfloat16()
    weight=torch.randn(40,24,20480,device='cuda')*.01
    scale=torch.tensor([.1,.1,.1],device='cuda');base=torch.randn(24,device='cuda')*.01
    storage=[[torch.full((rows*n+2*guard,),1.75,device='cuda') for n in [80*24,80,4,4,16]]for _ in variants]
    def call(op,mode,pdl,values):
        e=entries[op];words=[C.c_uint64(v)for v in values+[0,0]]
        assert len(values)==len(e['arguments'])
        args=(P*len(words))(*(C.cast(C.byref(v),P)for v in words))
        g=e['grid'];gx=(rows*g.get('x_mul',1)+g['rows_div']-1)//g['rows_div']+g.get('x_add',0)
        assert g['z']==1 and launch(ops[op,mode],gx,g['y'],32*e['metadata']['num_warps'],
            e['metadata']['shared'],stream.cuda_stream,args,pdl)==0
    f32=lambda v:C.cast(C.pointer(C.c_float(v)),C.POINTER(C.c_uint32)).contents.value
    def run(v,index=0):
        pm,ps,pre,post,comb=[t[guard:-guard]for t in storage[v]];sm,rm,pdl=variants[v]
        call('post',sm,pdl,[t.data_ptr()for t in [x[index],weight[index],pm,ps]]+[rows,20480,20480,20480])
        call('norm',rm,pdl,[t.data_ptr()for t in [pm,ps,scale,base,pre,post,comb]]+[rows,f32(1./20480),f32(1e-20)])
    graphs=[]
    for v in range(len(variants)):
        run(v);stream.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):run(v)
        graphs.append(g)
    for case in ['normal','changed','zero','tiny','near_floor','mixed_scales','large']:
        x.normal_();weight.normal_().mul_(.01);base.normal_().mul_(.01)
        if case=='changed':x.mul_(3.25).sub_(.5)
        elif case=='zero':x.zero_();x[:,:,1::2]=-0.
        elif case=='tiny':x.mul_(1e-30)
        elif case=='near_floor':x.mul_(1e-10)
        elif case=='mixed_scales':weight.mul_(torch.pow(2.,torch.randint(-12,12,(40,24,1),device='cuda').float()))
        elif case=='large':x.mul_(1e10)
        for g in graphs:g.replay()
        stream.synchronize()
        diffs=[[int(a.view(torch.uint8).ne(b.view(torch.uint8)).sum())for a,b in zip(storage[0],s)]for s in storage[1:]]
        assert not any(d for row in diffs for d in row),(rows,case,diffs)
        for s in storage:
            for t in s:
                assert bool((t[:guard]==1.75).all())and bool((t[-guard:]==1.75).all())
                assert bool(torch.isfinite(t).all())
        checks.append(dict(rows=rows,case=case,different_bytes=diffs))
    x.normal_();weight.normal_().mul_(.01)
    for locality in ['hot','rotating40']:
        timed=[]
        for v in range(len(variants)):
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):
                for i in range(40):run(v,i if locality=='rotating40'else 0)
            timed.append(g)
        for g in timed:g.replay()
        stream.synchronize();samples=[[]for _ in variants]
        for repeat in range(6):
            for v in range(4)if repeat%2==0 else reversed(range(4)):
                start,end=[torch.cuda.Event(enable_timing=True)for _ in range(2)]
                start.record(stream);timed[v].replay();end.record(stream);end.synchronize()
                samples[v].append(start.elapsed_time(end)*1000/40)
        timings.append(dict(rows=rows,locality=locality,median_us=[statistics.median(v)for v in samples],samples_us=samples))
        for g in timed:g.reset()
    for g in graphs:g.reset()
stream.synchronize()
for mod in modules:assert unload(mod)==0
result=dict(all_exact=True,checks=checks,timings=timings,variants=variants,compiler=compiler,provenance=provenance,
    artifacts={f'{op}-{mode}':hashlib.sha256(raw).hexdigest()for(op,mode),raw in artifacts.items()},
    scope='M3/M6 actual immutable80-slice coefficient stats/reduce. Rebuilt plain assembler control plus early/latePDL, unchanged PTX arithmetic.14changed-input/guard checks and6reversed40-pair graph timings,hot androtating40coefficient matrices. Small component screen, no whole-model or auxiliary-overlap claim.')
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(all_exact=True,timings=timings)),flush=True)
