"""Residual-update tiling screen, preserving explicit FP32 FMA/BF16 boundaries.

Adapted from SGLang f80c91a4b97e52e17fbcff088aa6b47b6270f6b9,
python/sglang/kernels/ops/layernorm/mhc_post_split_h.py (Apache-2.0).
No coefficient reduction is moved from its existing overlapping stream.
"""
import triton
import triton.language as tl

@triton.jit
def hc_post_tiles(A,R,POST,C,Y,B:tl.constexpr):
    row,tile=tl.program_id(0),tl.program_id(1)
    h=tile*B+tl.arange(0,B)
    a=tl.load(A+row*5120+h,h<5120,0).to(tl.float32)
    r0=tl.load(R+row*20480+h,h<5120,0).to(tl.float32)
    r1=tl.load(R+row*20480+5120+h,h<5120,0).to(tl.float32)
    r2=tl.load(R+row*20480+10240+h,h<5120,0).to(tl.float32)
    r3=tl.load(R+row*20480+15360+h,h<5120,0).to(tl.float32)
    for c in tl.static_range(4):
        post=tl.load(POST+row*4+c)
        c0=tl.load(C+row*16+c)
        z=tl.fma(post,a,c0*r0)
        z=tl.fma(tl.load(C+row*16+4+c),r1,z)
        z=tl.fma(tl.load(C+row*16+8+c),r2,z)
        z=tl.fma(tl.load(C+row*16+12+c),r3,z)
        tl.store(Y+row*20480+c*5120+h,z.to(tl.bfloat16),h<5120)
