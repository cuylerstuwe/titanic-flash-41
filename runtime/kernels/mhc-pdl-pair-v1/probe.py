"""Actual M6 residual post -> norm/FP8-quant chain, with PDL scheduling only.

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
compiler=subprocess.check_output([str(CUDA/'bin/ptxas'),'--version'],text=True)
specs=[('post','mhc-post-tiles-v1','hc_post_b128_w1'),
       ('norm','mhc-norm-quant-layout8-v1','hc_norm_quant_p2')]
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
        subprocess.run([str(CUDA/'bin/ptxas'),'-arch=sm_121a','-O3',str(path),'-o',str(cubin)],check=True)
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
    a=torch.randn(rows,5120,device='cuda').bfloat16();x=torch.randn(rows,20480,device='cuda').bfloat16()
    post=torch.randn(rows,4,device='cuda');comb=torch.randn(rows,16,device='cuda')
    pre=torch.randn(rows,4,device='cuda');weight=torch.randn(5120,device='cuda').bfloat16()
    storage=[]
    for _ in variants:
        storage.append([torch.full((n+2*guard,),value,dtype=dtype,device='cuda') for n,value,dtype in [
            (rows*20480,1.75,torch.bfloat16),(rows*5120,1.75,torch.bfloat16),
            (rows*5120,91,torch.uint8),(rows*160,91,torch.uint8),(20480,91,torch.uint8)]])
    def call(op,mode,pdl,values):
        e=entries[op];words=[C.c_uint64(v)for v in values+[0,0]]
        args=(P*len(words))(*(C.cast(C.byref(v),P)for v in words))
        g=e['grid'];gx=(rows*g.get('x_mul',1)+g['rows_div']-1)//g['rows_div']+g.get('x_add',0)
        assert g['z']==1 and launch(ops[op,mode],gx,g['y'],32*e['metadata']['num_warps'],
            e['metadata']['shared'],stream.cuda_stream,args,pdl)==0
    def run(v):
        z,y,q,sr,sm=[t[guard:-guard]for t in storage[v]];pm,nm,pdl=variants[v]
        call('post',pm,pdl,[t.data_ptr()for t in [a,x,post,comb,z]])
        call('norm',nm,pdl,[t.data_ptr()for t in [z,pre,weight,y,q,sr,sm]])
    graphs=[]
    for v in range(len(variants)):
        run(v);stream.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):run(v)
        graphs.append(g)
    for case in ['normal','changed','zero','tiny','near_floor','mixed_scales','large']:
        a.normal_();x.normal_();post.normal_();comb.normal_();pre.normal_();weight.normal_()
        if case=='changed':x.mul_(3.25).sub_(.5)
        elif case=='zero':x.zero_();x[:,1::2]=-0.;a.zero_()
        elif case=='tiny':weight.mul_(1e-30)
        elif case=='near_floor':weight.mul_(3e-5)
        elif case=='mixed_scales':weight.copy_(torch.pow(2.,torch.randint(-100,100,(5120,),device='cuda').float()).bfloat16())
        elif case=='large':x.mul_(1e10);weight.mul_(1e20)
        for g in graphs:g.replay()
        stream.synchronize()
        diffs=[[int(a.view(torch.uint8).ne(b.view(torch.uint8)).sum())for a,b in zip(storage[0],s)]for s in storage[1:]]
        assert not any(d for row in diffs for d in row),(rows,case,diffs)
        for s in storage:
            for i,t in enumerate(s):
                value=1.75 if i<2 else 91
                assert bool((t[:guard]==value).all())and bool((t[-guard:]==value).all())
        checks.append(dict(rows=rows,case=case,different_bytes=diffs))
    x.normal_();a.normal_();weight.normal_()
    timed=[]
    for v in range(len(variants)):
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):
            for _ in range(64):run(v)
        timed.append(g)
    for g in timed:g.replay()
    stream.synchronize();samples=[[]for _ in variants]
    for repeat in range(6):
        for v in range(4)if repeat%2==0 else reversed(range(4)):
            start,end=[torch.cuda.Event(enable_timing=True)for _ in range(2)]
            start.record(stream);timed[v].replay();end.record(stream);end.synchronize()
            samples[v].append(start.elapsed_time(end)*1000/64)
    timings.append(dict(rows=rows,median_us=[statistics.median(v)for v in samples],samples_us=samples))
    for g in timed+graphs:g.reset()
stream.synchronize()
for mod in modules:assert unload(mod)==0
result=dict(all_exact=True,checks=checks,timings=timings,variants=variants,compiler=compiler,provenance=provenance,
    artifacts={f'{op}-{mode}':hashlib.sha256(raw).hexdigest()for(op,mode),raw in artifacts.items()},
    scope='M3/M6 actual immutable residual post and layout8norm/quant. Rebuilt plain assembler control plus early/latePDL, unchanged PTX arithmetic.14changed-input/guard checks and6reversed64-pair graph timings. Small component screen, no whole-model claim.')
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(all_exact=True,timings=timings)),flush=True)
