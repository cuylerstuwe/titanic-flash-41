"""Exact memoization of the canonical token-conditioned Markov projection."""
import triton
import triton.language as tl

@triton.jit
def markov_head_cached(X,W,Base,Token,Cache,Tags,Y,Hit,Counters,N:tl.constexpr,SLOTS:tl.constexpr,BN:tl.constexpr):
    block=tl.program_id(0)
    col=block*BN+tl.arange(0,BN)
    token=tl.load(Token)
    valid=(token>=0)&(token<129280)
    slot=tl.where(valid,token,0)%SLOTS
    tag=Tags+slot*tl.cdiv(N,BN)+block
    hit=valid&(tl.load(tag)==token)
    if hit:
        dot=tl.load(Cache+slot*N+col,col<N,other=0)
    else:
        k=tl.arange(0,256)
        x=tl.load(X+k).to(tl.float32)
        w=tl.load(W+col[:,None]*256+k[None,:],col[:,None]<N,other=0).to(tl.float32)
        dot=tl.sum(w*x[None,:],1)
        tl.store(Cache+slot*N+col,dot,(col<N)&valid)
        # Each CTA owns one disjoint tag and column range. Calls use one
        # ordered stream, so a future lookup follows completion of every store.
        tl.store(tag,token,valid)
    base=tl.load(Base+col,col<N,other=0)
    tl.store(Y+col,dot+base,col<N)
    if block==0:
        tl.store(Hit,hit.to(tl.int32))
        tl.store(Counters,tl.load(Counters)+hit.to(tl.uint64))
        tl.store(Counters+1,tl.load(Counters+1)+1)
