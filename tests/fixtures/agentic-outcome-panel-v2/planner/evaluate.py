"""Held-out checks; executed only inside the bounded evaluation container."""
import copy,importlib.util,json,sys

def load(path):
    spec=importlib.util.spec_from_file_location('candidate_client',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

def evaluate(module):
    rows=[]
    def check(name,fn):
        try:fn();rows.append(dict(name=name,passed=True))
        except BaseException as error:rows.append(dict(name=name,passed=False,error=f'{type(error).__name__}: {error}'))
    def run(targets,deps,done,expected):
        before=copy.deepcopy((targets,deps,done))
        got=module.plan(targets,deps,done)
        assert isinstance(got,list)and got==expected,(got,expected)
        assert(targets,deps,done)==before,'input mutated'
    cases=[
        ('empty',[],{'x':['x']},[],[]),
        ('unknown_target',['x'],{},[],['x']),
        ('lexical_targets',['z','a','m'],{},[],['a','m','z']),
        ('chain',['a'],{'a':['b'],'b':['c']},[],['c','b','a']),
        ('newly_ready_priority',['a','z'],{'a':['b']},[],['b','a','z']),
        ('shared_dependency',['z','a'],{'z':['x'],'a':['x']},[],['x','a','z']),
        ('duplicates',['z','a','a'],{'z':['x','x'],'a':['x']},[],['x','a','z']),
        ('disconnected_cycle',['a'],{'u':['v'],'v':['u']},[],['a']),
        ('completed_target_prunes',['a'],{'a':['b'],'b':['b']},['a'],[]),
        ('completed_dependency_prunes',['a'],{'a':['b'],'b':['c'],'c':['c']},['b'],['a']),
        ('completed_breaks_cycle',['a'],{'a':['b'],'b':['a']},['b'],['a']),
        ('mixed_completed',['z','a'],{'a':['b','c'],'z':['c']},['c'],['b','a','z']),
        ('unknown_dependency',['z'],{'z':['a']},[],['a','z']),
        ('unused_work_excluded',['a'],{'x':['y']},[],['a']),
        ('case_sensitive_order',['b','A','a'],{},[],['A','a','b']),
    ]
    for name,targets,deps,done,expected in cases:
        check(name,lambda targets=targets,deps=deps,done=done,expected=expected:run(targets,deps,done,expected))
    def cycle(deps):
        before=copy.deepcopy(deps)
        try:module.plan(['a'],deps)
        except ValueError as error:assert 'cycle'in str(error).lower()
        else:raise AssertionError('cycle accepted')
        assert deps==before
    for name,deps in [('self',{'a':['a']}),('pair',{'a':['b'],'b':['a']}),
        ('reachable_tail',{'a':['b','z'],'b':['c'],'c':['b']})]:
        check('cycle_'+name,lambda deps=deps:cycle(deps))
    def deep_chain():
        names=[f'n{i:04}'for i in range(3000)]
        deps={names[i]:[names[i+1]]for i in range(len(names)-1)}
        run([names[0]],deps,[],list(reversed(names)))
    check('deep_chain_without_recursion_limit',deep_chain)
    def order_invariant():
        expected=['d','b','c','a','z']
        for targets in [['z','a'],['a','z'],['a','z','a']]:
            for deps in [{'a':['b','c'],'c':['d'],'b':['d']},
                {'b':['d','d'],'c':['d'],'a':['c','b']}]:run(targets,deps,[],expected)
    check('input_order_independent',order_invariant)
    return dict(task='deterministic-build-plan-v1',tests=rows,passed_required=sum(r['passed']for r in rows),
        required_checks=len(rows),task_checks_pass=all(r['passed']for r in rows))

if __name__=='__main__':
    try:result=evaluate(load(sys.argv[1]))
    except BaseException as error:result=dict(task_checks_pass=False,grader_error=f'{type(error).__name__}: {error}')
    print('CANADA_OUTCOME='+json.dumps(result))
