def subtract_intervals(base, blocked):
    def union(items):
        pairs = []
        for a, b in items:
            if a > b: raise ValueError('reversed interval')
            if a < b: pairs.append((a,b))
        out = []
        for a,b in sorted(pairs):
            if out and a <= out[-1][1]: out[-1] = (out[-1][0],max(out[-1][1],b))
            else: out.append((a,b))
        return out
    source, holes = union(base), union(blocked)
    out=[];j=0
    for a,b in source:
        while j<len(holes) and holes[j][1]<=a: j+=1
        k=j;at=a
        while k<len(holes) and holes[k][0]<b:
            c,d=holes[k]
            if c>at: out.append((at,min(c,b)))
            at=max(at,d)
            if at>=b: break
            k+=1
        if at<b: out.append((at,b))
    return out
