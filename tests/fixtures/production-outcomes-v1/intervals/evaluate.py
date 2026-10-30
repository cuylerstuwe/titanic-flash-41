import copy,importlib.util,json,random,sys
spec=importlib.util.spec_from_file_location("candidate",sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
try:
    fn=m.subtract_intervals
    samples=[([],[],[]), ([(0,10)],[(3,7)],[(0,3),(7,10)]),
     ([(5,10),(0,5),(2,8),(0,5)],[(10,12),(-2,0)],[(0,10)]),
     ([(-10,10)], [(-20,-4),(-4,0),(2,4),(3,5)],[(0,2),(5,10)]),
     ([(0,10)],[(0,10)],[]), ([(4,4)],[(2,2)],[]),
     ([(10**40,10**40+100)],[(10**40+1,10**40+99)],[(10**40,10**40+1),(10**40+99,10**40+100)])]
    for a,b,want in samples:
     before=copy.deepcopy((a,b));assert fn(a,b)==want;assert (a,b)==before
    for a,b in [([(2,1)],[]),([],[(2,1)])]:
     try:fn(a,b)
     except ValueError:pass
     else:raise AssertionError('reversed interval was accepted')
    # Small exhaustive coordinate oracle, kept entirely inside the code sandbox.
    for seed in range(80):
     rng=random.Random(seed);a=[tuple(sorted([rng.randrange(-8,9),rng.randrange(-8,9)]))for _ in range(8)];b=[tuple(sorted([rng.randrange(-8,9),rng.randrange(-8,9)]))for _ in range(5)]
     expected={x for l,h in a for x in range(l,h)}-{x for l,h in b for x in range(l,h)}
     got=fn(a,b);assert got==sorted(got)and all(l<h for l,h in got)and all(got[i][1]<got[i+1][0]for i in range(len(got)-1));assert {x for l,h in got for x in range(l,h)}==expected
except Exception as error:
    print("CANADA_OUTCOME="+json.dumps(dict(task_checks_pass=False,error=repr(error))))
else:
    print("CANADA_OUTCOME="+json.dumps(dict(task_checks_pass=True)))
