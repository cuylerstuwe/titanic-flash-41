import copy
def merge_layers(layers):
    def overlay(dst,src):
        for key,value in src.items():
            if value is None:dst.pop(key,None)
            elif isinstance(value,dict):
                old=dst.get(key)
                if not isinstance(old,dict):old={};dst[key]=old
                overlay(old,value)
            else:dst[key]=copy.deepcopy(value)
    out={}
    for layer in layers:overlay(out,layer)
    return out
