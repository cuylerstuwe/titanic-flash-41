"""Original GPU-only private-drafter integer copies; target weights unchanged.

INT4 uses per-row groups of32 and FP32 scales. INT8 uses groups of128.
Both preserve BF16 input and dequantized operands, FP32 accumulation/output.
These approximate proposal weights require actual-q rejection, not target use.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def quantize(W,Q,S,N:tl.constexpr,K:tl.constexpr,BITS:tl.constexpr,G:tl.constexpr):
    row=tl.program_id(0);group=tl.program_id(1)
    i=tl.arange(0,G//2);k=group*G+2*i
    x0=tl.load(W+row*K+k).to(tl.float32)
    x1=tl.load(W+row*K+k+1).to(tl.float32)
    amax=tl.max(tl.maximum(tl.abs(x0),tl.abs(x1)),0)
    qmax:tl.constexpr=(1<<(BITS-1))-1
    scale=tl.where(amax>0,tl.div_rn(amax,tl.full((),qmax,tl.float32)),1.)
    a=libdevice.nearbyint(tl.div_rn(x0,scale)).to(tl.int32)+(1<<(BITS-1))
    b=libdevice.nearbyint(tl.div_rn(x1,scale)).to(tl.int32)+(1<<(BITS-1))
    if BITS==4:
        tl.store(Q+row*(K//2)+k//2,(a|(b<<4)).to(tl.uint8))
    else:
        tl.store(Q+row*K+k,a.to(tl.uint8));tl.store(Q+row*K+k+1,b.to(tl.uint8))
    tl.store(S+row*(K//G)+group,scale)

@triton.jit
def decode(Q,S,n,k,N:tl.constexpr,K:tl.constexpr,BITS:tl.constexpr,G:tl.constexpr):
    valid=(n<N)&(k<K)
    if BITS==4:
        raw=tl.load(Q+n*(K//2)+k//2,valid,other=0).to(tl.int32)
        code=(raw>>((k%2)*4))&15
    else:code=tl.load(Q+n*K+k,valid,other=0).to(tl.int32)
    scale=tl.load(S+n*(K//G)+k//G,valid,other=0)
    return ((code-(1<<(BITS-1))).to(tl.float32)*scale).to(tl.bfloat16)

@triton.jit
def gemv(X,Q,S,Y,N:tl.constexpr,K:tl.constexpr,BITS:tl.constexpr,G:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN);inner=tl.arange(0,BK)
    a0=tl.zeros((BN,BK),tl.float32);a1=tl.zeros((BN,BK),tl.float32)
    a2=tl.zeros((BN,BK),tl.float32);a3=tl.zeros((BN,BK),tl.float32);a4=tl.zeros((BN,BK),tl.float32)
    for block in range(tl.cdiv(K,BK)):
        k=inner+block*BK
        w=decode(Q,S,n[:,None],k[None,:],N,K,BITS,G).to(tl.float32)
        x0=tl.load(X+k,k<K,other=0).to(tl.float32)
        x1=tl.load(X+K+k,k<K,other=0).to(tl.float32)
        x2=tl.load(X+2*K+k,k<K,other=0).to(tl.float32)
        x3=tl.load(X+3*K+k,k<K,other=0).to(tl.float32)
        x4=tl.load(X+4*K+k,k<K,other=0).to(tl.float32)
        a0=tl.fma(w,x0[None,:],a0);a1=tl.fma(w,x1[None,:],a1)
        a2=tl.fma(w,x2[None,:],a2);a3=tl.fma(w,x3[None,:],a3);a4=tl.fma(w,x4[None,:],a4)
    tl.store(Y+n,tl.sum(a0,1),n<N);tl.store(Y+N+n,tl.sum(a1,1),n<N)
    tl.store(Y+2*N+n,tl.sum(a2,1),n<N);tl.store(Y+3*N+n,tl.sum(a3,1),n<N)
    tl.store(Y+4*N+n,tl.sum(a4,1),n<N)

@triton.jit
def tiled(X,Q,S,Y,N:tl.constexpr,K:tl.constexpr,BITS:tl.constexpr,G:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN);r=tl.arange(0,16);inner=tl.arange(0,BK)
    acc=tl.zeros((16,BN),tl.float32)
    for block in range(tl.cdiv(K,BK)):
        k=inner+block*BK
        x=tl.load(X+r[:,None]*K+k[None,:],(r[:,None]<5)&(k[None,:]<K),other=0)
        w=decode(Q,S,n[None,:],k[:,None],N,K,BITS,G)
        acc+=tl.dot(x,w)
    tl.store(Y+r[:,None]*N+n[None,:],acc,(r[:,None]<5)&(n[None,:]<N))
