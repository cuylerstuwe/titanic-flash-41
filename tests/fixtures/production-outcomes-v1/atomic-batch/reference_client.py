def apply_batch(state,operations):
    work=state.copy()
    for op in operations:
        if not isinstance(op,tuple) or not op:raise ValueError('operation')
        kind=op[0]
        if kind in ('set','add','check'):
            if len(op)!=3:raise ValueError('arity')
            _,key,value=op
            if kind=='set':work[key]=value
            elif kind=='add':work[key]+=value
            elif key not in work or work[key]!=value:raise ValueError('check')
        elif kind=='delete':
            if len(op)!=2:raise ValueError('arity')
            del work[op[1]]
        else:raise ValueError('operation')
    state.clear();state.update(work);return work.copy()
