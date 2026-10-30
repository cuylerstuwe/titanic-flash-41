import copy,importlib.util,json,random,sys
spec=importlib.util.spec_from_file_location("candidate",sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
try:
    layers=[{'a':{'x':1,'y':[1,{'k':2}]},'b':2,'c':[None,{'d':None}]},{'a':{'x':None,'z':3},'b':{},'missing':None}];before=copy.deepcopy(layers);r=m.merge_layers(iter(layers));assert r=={'a':{'y':[1,{'k':2}],'z':3},'b':{},'c':[None,{'d':None}]}and layers==before;r['a']['y'][1]['k']=9;r['c'][1]['d']=3;assert layers==before
    assert m.merge_layers([{'x':{'a':1}},{'x':{} }])=={'x':{'a':1}}
    assert m.merge_layers([{'x':{'a':1}},{'x':None},{'x':{'b':2}}])=={'x':{'b':2}}
    assert m.merge_layers([{'x':[1,2]},{'x':[3,None]}])=={'x':[3,None]}
    assert m.merge_layers([{'x':1},{'x':{'gone':None}}])=={'x':{}}
    assert m.merge_layers(iter([]))=={}
except Exception as error:
    print("CANADA_OUTCOME="+json.dumps(dict(task_checks_pass=False,error=repr(error))))
else:
    print("CANADA_OUTCOME="+json.dumps(dict(task_checks_pass=True)))
