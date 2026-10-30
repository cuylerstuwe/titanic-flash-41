def apply_batch(state, operations):
    for op in operations:
        if op[0]=='set':state[op[1]]=op[2]
    return state
