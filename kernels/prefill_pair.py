# Pair-pooling math adapted from SGLang pair_pool_decode.py (Apache-2.0),
# f80c91a4b97e52e17fbcff088aa6b47b6270f6b9. Scheduling and two-pass tail
# publication are independent: all carried reads finish before wrapping writes.
# Build with enable_reflect_ftz=False: softmax weights can be subnormal, and
# flushing libdevice exp would lose BF16 subnormals at the required boundary.
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def pool(INPUT, STATE, POS, REQ, PREVIOUS, OUTPUT, RING: tl.constexpr, D: tl.constexpr):
    row=tl.program_id(0)
    offs=tl.arange(0,D)
    previous=tl.load(PREVIOUS+row)
    pos=tl.load(POS+row)
    req=tl.load(REQ+row)
    carry=req*RING+(pos-1+RING)%RING
    old_k=tl.load(STATE+carry*1024+offs,mask=previous==-1,other=0.)
    old_s=tl.load(STATE+carry*1024+512+offs,mask=previous==-1,other=0.)
    input_k=tl.load(INPUT+tl.maximum(previous,0)*1024+offs,mask=previous>=0,other=0.)
    input_s=tl.load(INPUT+tl.maximum(previous,0)*1024+512+offs,mask=previous>=0,other=0.)
    old_k=tl.where(previous>=0,input_k,old_k)
    old_s=tl.where(previous>=0,input_s,old_s)
    new_k=tl.load(INPUT+row*1024+offs)
    new_s=tl.load(INPUT+row*1024+512+offs)
    maximum=tl.maximum(old_s,new_s)
    e0=libdevice.exp(old_s-maximum)
    e1=libdevice.exp(new_s-maximum)
    denominator=e0+e1
    t0=old_k*libdevice.div_rn(e0,denominator)
    t1=new_k*libdevice.div_rn(e1,denominator)
    t0=t0+0.0
    t1=t1+0.0
    tl.store(OUTPUT+row*512+offs,t0+t1)

@triton.jit
def publish(INPUT, STATE, POS, REQ, KEEP, RING: tl.constexpr, D: tl.constexpr):
    row=tl.program_id(0)
    keep=tl.load(KEEP+row)!=0
    req=tl.load(REQ+row)
    pos=tl.load(POS+row)
    offs=tl.arange(0,D)
    values=tl.load(INPUT+row*1024+offs,mask=keep,other=0.)
    tl.store(STATE+(req*RING+pos%RING)*1024+offs,values,mask=keep)
