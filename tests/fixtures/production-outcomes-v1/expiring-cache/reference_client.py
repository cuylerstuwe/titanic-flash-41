from collections import OrderedDict
class ExpiringLRU:
    def __init__(self,capacity): self.capacity=capacity;self.data=OrderedDict()
    def _expire(self,now):
        for key,(_,end) in list(self.data.items()):
            if end<=now:del self.data[key]
    def put(self,key,value,ttl,now):
        self._expire(now);self.data.pop(key,None)
        if ttl>0:self.data[key]=(value,now+ttl)
        while len(self.data)>self.capacity:self.data.popitem(last=False)
    def get(self,key,now,default=None):
        self._expire(now)
        if key not in self.data:return default
        self.data.move_to_end(key);return self.data[key][0]
    def keys(self,now):self._expire(now);return list(self.data)
