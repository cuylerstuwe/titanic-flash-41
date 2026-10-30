class ExpiringLRU:
    def __init__(self, capacity): self.capacity=capacity; self.data={}
    def put(self,key,value,ttl,now): self.data[key]=value
    def get(self,key,now,default=None): return self.data.get(key,default)
    def keys(self,now): return list(self.data)
