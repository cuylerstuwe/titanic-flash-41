#!/usr/bin/env python3
"""Export the s824 invariant FC1 input-address patch with the GPU packer.
Build-time dependency: b12x a7d7d29, Apache-2.0. Canonical layer0/rank3
weights; metadata-only ABI capture. No persisted weights or Python runtime.
"""
from moe_fc1_input_address_patch import patch as input_address_patch
from moe_fc_pdl_patch import patch as pdl_patch
import ast, contextlib, ctypes, faulthandler, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
faulthandler.enable(all_threads=True)
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=False)
PIPELINE=(2,2)
REUSE_FC1_SCALES=False
LIB = Path('/kernel-reference')
PIN = 'a7d7d29b2ef8869086e0ceaa787321f17544e3c9'
assert (LIB / 'b12x_next/SOURCE_COMMIT').read_text().strip() == PIN
pipeline_overlay = None
if PIPELINE is not None:
    isolated = OUT.parent / 'kernel-reference-pipeline'
    assert not isolated.exists()
    shutil.copytree(LIB / 'b12x_next', isolated / 'b12x_next',
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    folder = isolated / 'b12x_next/moe/_shared/kernels'
    originals = {
        'w4a8_phase1.py': 'fc9f78557b85fd415ebf0103e8bd6f345fbd1d166e033a38aaff5c4cfc343948',
        'w4a8_phase2.py': '0c9fce39b243676e6754ea1fe111f765b5a993b6d6946ac410301a4663d1c5c3',
    }
    pipeline_overlay = dict(base_commit=PIN, license='Apache-2.0',
        source='https://github.com/local-inference-lab/b12x', stages=list(PIPELINE),
        reuse_fc1_weight_scales=False, output_major_tasks=False, hoisted_input=True, fc_pdl=True, files={})
    saved = OUT.parent / 'pipeline-source'; saved.mkdir()
    for phase, (name, expected) in enumerate(originals.items()):
        path = folder / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        source = path.read_text()
        before = '        self.stages = 2\n' if phase == 0 else '        self.stages = 4 if self.direct_routes else 2\n'
        after = f'        self.stages = {PIPELINE[phase]}\n' if phase == 0 else f'        self.stages = 4 if self.direct_routes else {PIPELINE[phase]}\n'
        assert source.count(before) == 1
        source = source.replace(before, after)
        if phase == 0:
            source = input_address_patch(source)
        source = pdl_patch(source, phase + 1)
        path.write_text(source)
        shutil.copy2(path, saved / name)
        pipeline_overlay['files'][name] = dict(original_sha256=expected,
            patched_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT.parent / 'pipeline-overlay.json').write_text(json.dumps(pipeline_overlay, indent=2)+'\n')
    LIB = isolated
os.environ['B12X_NEXT_COMPILE_CACHE_DIR']=str(OUT.parent/'compile-cache')
os.environ['B12X_NEXT_COMPILE_DISK_CACHE']='0'
os.environ['B12X_NEXT_COMPILE_MEMORY_CACHE']='0'
os.environ['TRITON_PTXAS_BLACKWELL_PATH']='/usr/local/cuda/bin/ptxas'
sys.path.insert(0,str(LIB))
import torch
from safetensors import safe_open
from b12x_next import preparation as prep
from b12x_next.moe import fused_moe as fm

torch.set_num_threads(4)
torch.manual_seed(1017)
device = torch.device('cuda')
props = torch.cuda.get_device_properties(device)
assert (props.major, props.minor, props.multi_processor_count) == (12, 1, 48)
assert torch.cuda.mem_get_info()[0] > 16 * 2**30
import canonical_weights as canonical
E,K,N,TOPK,M,RANK,LAYER = canonical.E,canonical.K,canonical.N,canonical.TOPK,canonical.M,canonical.RANK,canonical.LAYER
experts=canonical.experts
log=canonical.log

from b12x_next._lib import compiler
exports=[];compiled_entries={};patched_types=set();capture=False;nodes=[];roles=[]
original_compile=compiler._call_cute_compile;original_run=compiler.run_compiled
def add_role(role,t):
    assert t.is_contiguous(),role
    roles.append((t.data_ptr(),t.numel()*t.element_size(),role))
def pointer_role(value):
    matches={(role,value-base) for base,size,role in roles if base<=value<base+size}
    assert len(matches)==1,('unknown or ambiguous runtime operand',value,matches)
    role,offset=next(iter(matches));return dict(role=role,offset=offset)
def capture_run(compiled,args):
    if not capture:return original_run(compiled,args)
    entry=compiled_entries[id(compiled)];header=(OUT/(exports[entry]['name']+'.h')).read_text()
    params=re.search(r'static inline int32_t cute_dsl_\w+_wrapper\((.*?)\)\s*\{',header,re.S)[1].split(',')[1:]
    execution_args,adapted=compiled.generate_execution_args(*args)
    addresses=list(execution_args);assert len(params)==len(addresses)
    pointers={};scalars={}
    for param,address in zip(params,addresses):
        param=param.strip();address=address.value if isinstance(address,ctypes.c_void_p) else int(address)
        if param=='cudaStream_t stream':continue
        name=param.split()[-1].lstrip('*')
        if param.startswith('int32_t '):scalars[name]=ctypes.c_int32.from_address(address).value
        else:
            assert '*' in param,param
            pointers[name]=pointer_role(ctypes.c_uint64.from_address(address).value)
    nodes.append(dict(entry=entry,kernel_id=exports[entry]['kernel_id'],pointers=pointers,scalars=scalars))
    return original_run(compiled,args)
def exporting(*args,**kw):
    compiled=original_compile(*args,**kw);name=f'kernel_{len(exports):03d}'
    compiled.export_to_c(str(OUT),name,'canada_'+name);spec=kw.get('compile_spec')
    exports.append(dict(name=name,kernel_id=str(getattr(spec,'kernel_id','unknown')),
        header_sha256=hashlib.sha256((OUT/(name+'.h')).read_bytes()).hexdigest(),
        object_sha256=hashlib.sha256((OUT/(name+'.o')).read_bytes()).hexdigest()))
    compiled_entries[id(compiled)]=len(exports)-1
    if type(compiled) not in patched_types:
        typ=type(compiled);patched_types.add(typ);prior=typ.__call__
        def intercepted(self,*args,_prior=prior,**kw):
            if not capture:return _prior(self,*args,**kw)
            assert not kw;return capture_run(self,args)
        typ.__call__=intercepted
    log(phase='exported',kernel=exports[-1]);return compiled
compiler._call_cute_compile=exporting;compiler.run_compiled=capture_run
plan=fm.plan_execution(experts=experts,routing=fm.RoutingSpec(deterministic_output=True),
    capacity=fm.ExecutionCapacity(max_tokens=M,top_k=TOPK),
    override=fm.MoeDecodeConfig(backend='dynamic',route_planner='internal',max_active_clusters=48,dynamic_tile_m=16,dynamic_route_mode='grouped'))
session=prep.PreparationSession(device=device,autotune=False,compile_workers=0,cache_only=False)
session.prepare((plan.request(name='canada-fc1-input-address',prepare_call=lambda state:prep.PreparedCall(run=lambda:None)),),autotune=False)
specs=tuple(plan.scratch_specs());assert len(specs)==1
scratch=tuple(torch.zeros(tuple(s.shape),dtype=s.dtype,device=device) for s in specs)
x=(torch.randn(M,K,device=device)*.5).bfloat16();ids=torch.arange(M*TOPK,device=device,dtype=torch.int32).reshape(M,TOPK)
weights=torch.full((M,TOPK),.25,device=device);output=torch.zeros_like(x)
for role,t in [('input',x),('ids',ids),('routing_weights',weights),('scratch',scratch[0]),('output',output),
    ('w13',experts._impl.w1_fp4),('w2',experts._impl.w2_fp4),('s13',experts._impl.w1_blockscale),('s2',experts._impl.w2_blockscale)]:add_role(role,t)
for t in [experts._impl.a1_gscale,experts._impl.a2_gscale,experts._impl.w1_alphas,experts._impl.w2_alphas]:
    assert t.dtype==torch.float32 and t.numel()<=E and bool((t==1).all());add_role('ones',t)
binding=fm.bind(plan,scratch=scratch,a=x,topk_ids=ids,topk_weights=weights,output=output,unit_scale_contract=False)
capture=True
fm.run(binding=binding);torch.cuda.synchronize();capture=False
assert torch.isfinite(output).all() and nodes
manifest=dict(source='https://github.com/local-inference-lab/b12x',source_commit=PIN,license='Apache-2.0',
    architecture=121,physical_sms=48,kernels=exports,geometry=dict(experts=E,hidden=K,intermediate=N,topk=TOPK,rows=M),
    precision='Canonical MXFP4 E8M0 K32;W4A8;FP32 FC1 through SwiGLU;BF16 activation boundary and output;deterministic topk sum',
    fc1_input_address_overlay=pipeline_overlay,fc_pdl=True,atlas_sha256=canonical.meta['atlas_sha256'])
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
subprocess.run([sys.executable,str(Path(__file__).with_name('build-primitive-library.py')),str(OUT)],check=True)
bindings=json.loads((OUT/'bindings.json').read_text());assert bindings['abi']==1
for node in nodes:
    e=bindings['entries'][node['entry']]
    assert e['kernel_id']==node['kernel_id'] and set(e['pointers'])==set(node['pointers']) and set(e['i32_scalars'])==set(node['scalars'])
contract=dict(abi=1,rows=M,experts=E,hidden=K,intermediate=N,topk=TOPK,deterministic=True,swiglu_limit=10.,
    scratch_bytes=scratch[0].numel()*scratch[0].element_size(),scratch_initializer='zero',kernel_library_sha256=bindings['library_sha256'],
    source=manifest['source'],source_commit=PIN,license='Apache-2.0',nodes=nodes,
    fc1_input_address_overlay=pipeline_overlay,fc_pdl=True)
(OUT/'runtime-contract.json').write_text(json.dumps(contract,indent=2)+'\n')
# Check original and candidate C ABIs against the patched GPU launcher.
# Physical M6 scratch and offsets stay fixed; only validated live scalars vary.
old_root=Path('/native/moe-runtime-m6-input-address-v1')
old_contract=json.loads((old_root/'runtime-contract.json').read_text())
assert old_contract['kernel_library_sha256']=='c76ab738eccf40f86106ad2651f883a5e79e66492be61fe09a68a21bf54cc761'
for key in ['abi','rows','experts','hidden','intermediate','topk','deterministic','swiglu_limit','scratch_bytes','scratch_initializer']:
    assert contract[key]==old_contract[key],key
assert [{k:v for k,v in n.items()if k!='entry'}for n in nodes]==[{k:v for k,v in n.items()if k!='entry'}for n in old_contract['nodes']]
base_by_role={role:base for base,size,role in roles}
opened=[]
def open_native(root,contract):
    lib=ctypes.CDLL(str(root/'libcanada_moe_primitives.so'))
    assert hashlib.sha256((root/'libcanada_moe_primitives.so').read_bytes()).hexdigest()==contract['kernel_library_sha256']
    bindings=json.loads((root/'bindings.json').read_text())
    lib.ca_primitives_open.argtypes=[ctypes.POINTER(ctypes.c_int32)];lib.ca_primitives_open.restype=ctypes.c_void_p
    lib.ca_primitives_launch.argtypes=[ctypes.c_void_p,ctypes.c_int32,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t,ctypes.POINTER(ctypes.c_int32),ctypes.c_size_t,ctypes.c_void_p]
    lib.ca_primitives_launch.restype=ctypes.c_int32;lib.ca_primitives_close.argtypes=[ctypes.c_void_p]
    status=ctypes.c_int32();handle=lib.ca_primitives_open(ctypes.byref(status));assert status.value==0 and handle
    opened.append((lib,handle))
    def run(live=None):
        for node in contract['nodes']:
            entry=bindings['entries'][node['entry']]
            ptr=[base_by_role[node['pointers'][n]['role']]+node['pointers'][n]['offset']for n in entry['pointers']]
            scalar=dict(node['scalars'])
            if live:
                if node['kernel_id']=='integration.tp_moe.dynamic':scalar['num_tokens']=live
                elif node['kernel_id']=='moe.w4a16.topk_sum':scalar['active_m']=live
                else:raise AssertionError(node['kernel_id'])
            vals=[scalar[n]for n in entry['i32_scalars']]
            rc=lib.ca_primitives_launch(handle,node['entry'],(ctypes.c_uint64*len(ptr))(*ptr),len(ptr),
                (ctypes.c_int32*len(vals))(*vals),len(vals),ctypes.c_void_p(torch.cuda.current_stream().cuda_stream));assert rc==0,rc
    return run
control= open_native(old_root,old_contract)
candidate=open_native(OUT,contract)
records=[]
try:
    cases=[(live,route)for live in [1,2,3,4,5,6]for route in ['shared','overlap','disjoint']]+[(0,'masked'),(6,'zero')]
    for idx,(live,route)in enumerate(cases):
        x.copy_((torch.randn(M,K,device=device)*.5).bfloat16())
        row=torch.arange(M,device=device,dtype=torch.int32).reshape(M,1)
        col=torch.arange(TOPK,device=device,dtype=torch.int32).reshape(1,TOPK)
        step=0 if route in ['shared','zero','masked']else 2 if route=='overlap'else TOPK
        ids.copy_((row*step+col+idx*17)%E);ids[live:].fill_(-1)
        weights.fill_(.25);weights[live:].zero_()
        if route=='zero':weights.zero_()
        # Equal full-capacity control, full-capacity candidate, live candidate,
        # repeated captured launches and independently perturbed padded inputs.
        fm.run(binding=binding);torch.cuda.synchronize();expected=output[:live].clone()
        assert bool(torch.isfinite(expected).all())
        def check():
            torch.cuda.synchronize();assert torch.equal(output[:live].view(torch.int16),expected.view(torch.int16)),(live,route)
        output.fill_(19.);control();check()
        output.fill_(19.);candidate();check()
        output.fill_(19.);candidate(live);check()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidate(live)
        for _ in range(3):graph.replay();check()
        x[live:].fill_(77.);graph.replay();check()
        del graph
        records.append(dict(live_rows=live,routing=route,exact=True,padding_independent=True,graph_repeats=3,
            sha256=hashlib.sha256(expected.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()))
finally:
    torch.cuda.synchronize()
    for lib,handle in opened:lib.ca_primitives_close(handle)
(OUT/'qualification.json').write_text(json.dumps(dict(all_live_exact=True,cases=records,nodes=[n['kernel_id']for n in nodes],
    canonical_layer=LAYER,tensor_parallel_rank=RANK,scratch_bytes=contract['scratch_bytes'],
    control_library_sha256=old_contract['kernel_library_sha256'],runtime_python_required=False),indent=2)+'\n')
log(phase='complete',nodes=[n['kernel_id']for n in nodes],scratch_bytes=contract['scratch_bytes'])

# Same process: retain canonical pack and compare current/candidate AOT on the same compressed mapping.
import runpy
runpy.run_path('/probe/probe.py', run_name='__main__')
