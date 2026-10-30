"""A finite public-fixture coding loop. Generated code runs only in Docker.

No test or candidate Python is imported by this controller. It writes only the
 two named task files, never executes a model-supplied command, and never sends
 fixture data outside the explicitly supplied localhost inference endpoint.
"""
from pathlib import Path
import argparse,hashlib,http.client,json,os,selectors,subprocess,time,urllib.parse

IMAGE='sha256:82bb585723d16d25423646dd9b1300c51f5913f28c5e3900d5cc8eaf52f8aa4d'
FILES=['client.py','test_client.py']
BASE=Path(__file__).resolve().parent
ROOT=BASE

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def file_hashes(case):return {name:digest(case/name)for name in FILES}
def tool(name,description,properties,required):
    return dict(type='function',function=dict(name=name,description=description,parameters=dict(type='object',properties=properties,required=required,additionalProperties=False)))
TOOLS=[
    tool('read_file','Read one of the two task files.',{'path':dict(type='string',enum=FILES)},['path']),
    tool('write_file','Replace one complete task file. Files are client.py and test_client.py only.',{'path':dict(type='string',enum=FILES),'content':dict(type='string')},['path','content']),
    tool('run_tests','Run pytest against the current task files in an isolated Python environment. No network access.',{},[]),
]

def sandbox(case,deps,command,label,seconds=12):
    deps=deps.resolve()
    name='canada-scratch-agent-eval-'+label
    args=['docker','run','--rm','--name',name,'--label','owner=canada-scratch-20261022',
          '--network','none','--cpus','1','--memory','512m','--memory-swap','512m','--pids-limit','64',
          '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--user','65534:65534',
          '--tmpfs','/tmp:rw,nosuid,nodev,noexec,size=64m','-v',str(case)+':/case:ro',
          '-v',str(deps)+':/deps:ro',
          *(['-v',str(ROOT/'evaluate.py')+':/grader.py:ro']if command[0]=='/grader.py'else[]),
          '-e','PYTHONPATH=/deps:/case','-e','PYTHONDONTWRITEBYTECODE=1',
          '-e','PYTEST_DISABLE_PLUGIN_AUTOLOAD=1','--workdir','/case','--entrypoint','/usr/bin/timeout',
          IMAGE,'--signal=KILL',str(seconds),'/opt/sglang/bin/python3',*command]
    began=time.monotonic()
    try:
        process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output=bytearray();truncated=False;timed_out=False
        with selectors.DefaultSelector()as selector:
            selector.register(process.stdout,selectors.EVENT_READ)
            while True:
                if time.monotonic()-began>seconds+10:
                    timed_out=True;process.kill();break
                if selector.select(.25):
                    chunk=os.read(process.stdout.fileno(),16384)
                    if not chunk:break
                    retained=min(len(chunk),24000-len(output));output.extend(chunk[:retained]);truncated|=retained<len(chunk)
                elif process.poll()is not None:break
        code=process.wait(timeout=5);process.stdout.close()
        return dict(returncode=124 if timed_out else code,output=output.decode('utf8',errors='replace'),output_truncated=truncated,controller_timeout=timed_out,seconds=time.monotonic()-began)
    finally:
        subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15,check=False)

def chat(endpoint,key,body,seconds):
    c=http.client.HTTPConnection(endpoint.hostname,endpoint.port,timeout=seconds)
    began=time.monotonic()
    try:
        c.request('POST','/v1/chat/completions',json.dumps(body),{'Content-Type':'application/json','Authorization':'Bearer '+key})
        r=c.getresponse();raw=r.read(16*1024*1024+1)
        assert len(raw)<=16*1024*1024,'response bound exceeded'
        value=json.loads(raw)
        assert r.status==200,dict(status=r.status,error=value)
        return value,time.monotonic()-began
    finally:c.close()

def main():
    global ROOT
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--model',required=True)
    p.add_argument('--key-file',type=Path,required=True);p.add_argument('--deps',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,action='append',required=True)
    p.add_argument('--case',choices=['intervals','expiring-cache','atomic-batch','layered-config'],required=True)
    p.add_argument('--tag',required=True);p.add_argument('--sglang-metadata',action='store_true');a=p.parse_args()
    ROOT=BASE/a.case
    assert a.tag.isalnum()and len(a.tag)<32
    u=urllib.parse.urlsplit(a.url);assert u.scheme=='http'and u.hostname in ['127.0.0.1','localhost']and not u.username and u.path in ['','/']
    assert not a.sglang_metadata or u.port==18042
    assert all(0<=s<2**63-24 for s in a.seed)and len(set(a.seed))==len(a.seed)<=3
    spec=json.loads((ROOT/'task.json').read_text());manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,h in manifest['files'].items():assert digest(ROOT/name)==h,name
    assert digest(ROOT/'evaluate.py')==spec['held_out_grader_sha256']
    assert a.deps.is_dir()and not a.output.exists();a.output.mkdir(parents=True)
    key=a.key_file.read_text().strip();reports=[]
    for case_index,seed in enumerate(a.seed):
        out=(a.output/f'r{case_index}').resolve();out.mkdir();case=out/'files';case.mkdir()
        for dst,src in [('client.py','initial_client.py'),('test_client.py','initial_test_client.py')]:
            (case/dst).write_bytes((ROOT/src).read_bytes())
        messages=[dict(role='system',content=spec['system']),dict(role='user',content=spec['instructions'])]
        began=time.monotonic();deadline=began+spec['max_case_seconds'];last_test_hashes=None;last_test_pass=False
        tool_actions=tool_errors=test_runs=model_calls=completion_tokens=prompt_tokens=0
        model_seconds=tool_seconds=0.;terminal=False;fail_reason=None
        log=(out/'events.jsonl').open('x')
        def record(**v):log.write(json.dumps(v)+'\n');log.flush()
        record(kind='start',seed=seed,initial_files=file_hashes(case),manifest_sha256=digest(ROOT/'manifest.json'))
        try:
            for turn in range(spec['max_calls']):
                assert time.monotonic()<deadline,'case wall-clock budget'
                body=dict(model=a.model,messages=messages,tools=TOOLS,temperature=spec['temperature'],top_p=spec['top_p'],top_k=-1,
                          max_tokens=spec['max_tokens'],reasoning_effort=spec['reasoning_effort'],seed=seed+turn,stream=False)
                if a.sglang_metadata:body.update(return_spec_tokens_details=True,return_cached_tokens_details=True,return_meta_info=True)
                value,elapsed=chat(u,key,body,min(300,max(1,deadline-time.monotonic())))
                model_calls+=1;model_seconds+=elapsed
                usage=value.get('usage',{});completion_tokens+=usage.get('completion_tokens',0);prompt_tokens+=usage.get('prompt_tokens',0)
                record(kind='model_response',turn=turn,seconds=elapsed,response=value)
                assert len(value.get('choices',[]))==1,'expected one choice'
                choice=value['choices'][0]
                assistant={k:v for k,v in choice['message'].items()if k in ['role','content','reasoning_content','tool_calls']}
                assert assistant.get('role')=='assistant'
                if assistant.get('tool_calls'):
                    assistant['tool_calls']=[{k:v for k,v in c.items()if k!='index'}for c in assistant['tool_calls']]
                messages.append(assistant);calls=assistant.get('tool_calls')or[]
                if not calls:
                    terminal=choice.get('finish_reason')=='stop'
                    if not terminal:fail_reason='Nonterminal answer: '+str(choice.get('finish_reason'))
                    break
                assert choice.get('finish_reason')=='tool_calls','tool-call finish mismatch'
                assert all(isinstance(c.get('id'),str)and c['id']for c in calls)and len({c['id']for c in calls})==len(calls),'invalid tool call identities'
                for call in calls:
                    tool_actions+=1;assert tool_actions<=64 and time.monotonic()<deadline,'tool budget'
                    started=time.monotonic();name=call.get('function',{}).get('name');is_error=False
                    try:
                        args=json.loads(call['function']['arguments']);assert isinstance(args,dict),'arguments must be an object'
                        if name in ['read_file','write_file']:
                            expected={'path'}if name=='read_file'else{'path','content'}
                            assert set(args)==expected and args['path']in FILES,'unsupported fields or file'
                            if name=='read_file':result=dict(path=args['path'],content=(case/args['path']).read_text())
                            else:
                                content=args['content'];assert isinstance(content,str)and '\0'not in content and len(content.encode())<=65536,'file content bound'
                                (case/args['path']).write_text(content);result=dict(written=args['path'],sha256=digest(case/args['path']))
                        elif name=='run_tests':
                            assert not args,'run_tests takes no arguments'
                            test_runs+=1;last_test_hashes=file_hashes(case)
                            result=sandbox(case,a.deps,['-m','pytest','-q','-p','no:cacheprovider','-p','pytest_asyncio.plugin','test_client.py'],f'{a.tag}-{case_index}-{tool_actions}')
                            last_test_pass=result['returncode']==0
                        else:raise ValueError('Unknown tool. Available: read_file, write_file, run_tests.')
                    except (AssertionError,ValueError,KeyError,TypeError)as error:
                        is_error=True;tool_errors+=1;result=dict(error=str(error))
                    elapsed=time.monotonic()-started;tool_seconds+=elapsed
                    record(kind='tool_result',name=name,call_id=call['id'],is_error=is_error,seconds=elapsed,result=result)
                    messages.append(dict(role='tool',tool_call_id=call['id'],content=json.dumps(dict(is_error=is_error,**result))))
                if turn+1==spec['max_calls']:fail_reason='model call budget exhausted'
        except Exception as error:
            fail_reason=f'{type(error).__name__}: {error}'
        complete_seconds=time.monotonic()-began
        # Qualification is outside the measured repair loop and never exposed
        # to the model. Re-run supplied tests on exactly the final read-only files.
        held=sandbox(case,a.deps,['/grader.py','/case/client.py'],f'{a.tag}-{case_index}-held')
        supplied=sandbox(case,a.deps,['-m','pytest','-q','-p','no:cacheprovider','-p','pytest_asyncio.plugin','test_client.py'],f'{a.tag}-{case_index}-final')
        marker=held['output'].split('CANADA_OUTCOME=',1)
        try:grade=json.loads(marker[1])if len(marker)==2 else dict(task_checks_pass=False,grader_error=held)
        except json.JSONDecodeError:grade=dict(task_checks_pass=False,grader_error=held)
        ran_final_tests=last_test_hashes==file_hashes(case)and last_test_pass
        success=terminal and fail_reason is None and grade.get('task_checks_pass')is True and supplied['returncode']==0 and ran_final_tests
        report=dict(seed=seed,successful_outcome=success,terminal=terminal,failure=fail_reason,complete_seconds=complete_seconds,
                    model_seconds=model_seconds,tool_seconds=tool_seconds,model_calls=model_calls,tool_actions=tool_actions,
                    tool_errors=tool_errors,test_runs=test_runs,ran_passing_tests_on_final_files=ran_final_tests,
                    completion_tokens=completion_tokens,prompt_tokens=prompt_tokens,held_out=grade,supplied=supplied,final_files=file_hashes(case))
        record(kind='qualification',report=report);log.close();reports.append(report)
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items()if k not in ['held_out','supplied','final_files']}),flush=True)
        if fail_reason and ('HTTP' in fail_reason or 'Timeout' in fail_reason):break
    summary=dict(task=spec['id'],results=reports,successful=sum(r['successful_outcome']for r in reports),attempted=len(reports),
                 scope='Small tool-driven coding pilot. Timing includes all model calls and tool execution through final answer. Held-out grading runs afterwards. Failed, capped or incomplete tasks are not successful latency samples; no broad quality claim.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
