import copy,importlib.util,json,random,sys
spec=importlib.util.spec_from_file_location("candidate",sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
try:
    c=m.ExpiringLRU(2);a=object();c.put('a',a,10,0);c.put('b',None,2,0);assert c.keys(0)==['a','b'];assert c.get('a',1)is a;assert c.keys(1)==['b','a'];sentinel=object();assert c.get('b',1,sentinel)is None;assert c.keys(2)==['a'];assert c.get('b',2,sentinel)is sentinel
    c.put('b',2,10,2);c.put('c',3,10,2);assert c.keys(2)==['b','c'];c.get('missing',2);assert c.keys(2)==['b','c'];c.put('b',4,0,2);assert c.keys(2)==['c'];c.put('c',5,1,2);assert c.get('c',3,sentinel)is sentinel
    c=m.ExpiringLRU(3);c.put('long',1,100,0);c.put('short',2,.5,0);c.put('medium',3,10,0);c.get('short',.25);assert c.keys(.5)==['long','medium'];copy_keys=c.keys(1);copy_keys.clear();assert c.keys(1)==['long','medium'];c.put('long',4,.25,1);assert c.keys(1)==['medium','long'];assert c.keys(1.25)==['medium']
except Exception as error:
    print("CANADA_OUTCOME="+json.dumps(dict(task_checks_pass=False,error=repr(error))))
else:
    print("CANADA_OUTCOME="+json.dumps(dict(task_checks_pass=True)))
