"""Container/control validation with canned replies, never a model quality run."""
from pathlib import Path
import json,sys,time
import run as h
W=Path(__file__).resolve().parents[3]
out=W/'evidence/agentic-outcome-panel-v1-selfcheck';out.mkdir(exist_ok=False)
case=out/'files';case.mkdir()
for name in h.FILES:(case/name).write_bytes((h.ROOT/('selfcheck_'+name)).read_bytes())
deps=W/'evidence/evaluation-dependencies-v1/python'
checks={};started=time.time()
code="""import os,socket,pathlib
assert os.geteuid()==65534
assert not pathlib.Path('/dev/nvidia0').exists()
s=pathlib.Path('/proc/self/status').read_text()
assert int(s.split('CapEff:\\t')[1].splitlines()[0],16)==0
assert all(name=='lo' for _,name in socket.if_nameindex())
try:
 with open('/case/client.py','a') as f:f.write('should not be allowed')
except PermissionError:pass
else:raise AssertionError('read-only source mount was writable')
print('isolation_pass')
"""
checks['isolation']=h.sandbox(case,deps,['-c',code],'harness-isolation')
assert checks['isolation']['returncode']==0,checks['isolation']
checks['bounded_output']=h.sandbox(case,deps,['-c',"print('x'*100000)"],'harness-output')
assert checks['bounded_output']['returncode']==0 and checks['bounded_output']['output_truncated'] and len(checks['bounded_output']['output'])==24000
checks['timeout']=h.sandbox(case,deps,['-c','import time;time.sleep(60)'],'harness-timeout',seconds=1)
assert checks['timeout']['returncode']!=0 and checks['timeout']['seconds']<8,checks['timeout']
source={n:(case/n).read_text()for n in h.FILES}
def call(id,name,args):return dict(id=id,type='function',function=dict(name=name,arguments=json.dumps(args)))
responses=[
 dict(role='assistant',content=None,reasoning_content='Canned boundary exercise.',tool_calls=[
  call('escape','write_file',dict(path='../escape.txt',content='blocked')),
  call('unknown','bash',dict(command='must not execute')),
  call('read','read_file',dict(path='client.py'))]),
 dict(role='assistant',content=None,tool_calls=[
  call('client','write_file',dict(path='client.py',content=source['client.py'])),
  call('tests','write_file',dict(path='test_client.py',content=source['test_client.py'])),
  call('run','run_tests',{})]),
 dict(role='assistant',content='Canned protocol completion; not a model answer.'),
]
def fake_chat(endpoint,key,body,seconds):
 assert key=='not-a-real-key' and body['temperature']==1 and body['top_p']==1
 assistant=responses.pop(0)
 return dict(choices=[dict(message=assistant,finish_reason='tool_calls'if assistant.get('tool_calls')else'stop')],usage=dict(prompt_tokens=0,completion_tokens=0)),0.
h.chat=fake_chat
key=out/'fake.key';key.write_text('not-a-real-key\n')
sys.argv=['run.py','--url','http://127.0.0.1:1','--model','canned-control-selfcheck','--key-file',str(key),'--deps',str(deps),'--output',str(out/'protocol'),'--seed','3','--tag','harnessprotocol']
h.main()
summary=json.loads((out/'protocol/summary.json').read_text());r=summary['results'][0]
assert not responses and summary['successful']==1 and r['tool_errors']==2 and r['test_runs']==1
assert not(out/'protocol/r0/escape.txt').exists()
assert r['final_files']==h.file_hashes(case)
checks['protocol']=dict(successful=True,rejected_tool_actions=2,model_called=False,final_files_match=True,held_out=r['held_out']['passed_required'])
report=dict(all_pass=True,started=started,finished=time.time(),checks=checks,model_called=False,scope='Canned controller boundary checks, unchanged known-good generated code in the bounded container, readonly/nonroot/no-network/no-GPU contract, bounded stdout and timeout. No inference or quality/performance sample.')
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='checks'}))
