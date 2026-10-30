import copy,importlib.util,json,random,sys
spec=importlib.util.spec_from_file_location("candidate",sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
try:
    f=m.apply_batch;s={'a':1};identity=id(s);r=f(s,iter([('add','a',2),('check','a',3),('set','b',4),('delete','a')]));assert s==r=={'b':4}and id(s)==identity and r is not s;r['b']=9;assert s['b']==4
    for bad,kind in [([('add','missing',1)],KeyError),([('delete','missing')],KeyError),([('check','missing',0)],ValueError),([('check','a',2)],ValueError),([('other','a')],ValueError),([('set','a')],ValueError),([('delete','a',1)],ValueError)]:
     s={'a':1};before=s.copy()
     try:f(s,iter([('set','new',9)]+bad))
     except kind:pass
     else:raise AssertionError('operation did not fail as specified')
     assert s==before
    class IterationFault(Exception):pass
    def broken():
     yield ('set','new',9)
     raise IterationFault()
    s={'a':1}
    try:f(s,broken())
    except IterationFault:pass
    else:raise AssertionError('iterator failure swallowed')
    assert s=={'a':1};r=f(s,[]);assert r==s and r is not s
except Exception as error:
    print("CANADA_OUTCOME="+json.dumps(dict(task_checks_pass=False,error=repr(error))))
else:
    print("CANADA_OUTCOME="+json.dumps(dict(task_checks_pass=True)))
