"""Small-row HC combine/RMS plus existing MXFP8 input preparation.

Adapted from SGLang _hc_combine_norm at f80c91a4b97e52e17fbcff088aa6b47b6270f6b9
and b12x a7d7d29b2ef8869086e0ceaa787321f17544e3c9 quant/mxfp8_rows.py and
intrinsics.py. Both Apache-2.0; https://www.apache.org/licenses/LICENSE-2.0.
Preserves the four sequential FP32 FMAs, BF16 combine boundary, RMS reduction,
BF16 normalized boundary, minimum amax1e-4, UE8M0 ceiling, zero-scale handling and native MMA
scale addressing. Writes the BF16 values as well for routing and statistics.
No model weights or target precision changes. Component prototype.
"""
import triton
import triton.language as tl

@triton.jit
def hc_combine_norm_quant(X,P,W,Y,Q,SR,SM,PARTS:tl.constexpr):
    row,part=tl.program_id(0),tl.program_id(1)
    h=tl.arange(0,8192)
    value=tl.full((8192,),0,tl.float32)
    for c in tl.static_range(4):
        pre=tl.load(P+row*4+c).to(tl.float32)
        x=tl.load(X+row*20480+c*5120+h,h<5120,0).to(tl.float32)
        value+=x*pre
    value=value.to(tl.bfloat16).to(tl.float32)
    inv_rms=tl.rsqrt(tl.sum(value*value,0)/5120+1.e-20)
    mask=(h>=part*(5120//PARTS))&(h<(part+1)*(5120//PARTS))
    weight=tl.load(W+h,mask,0).to(tl.float32)
    y=(value*inv_rms*weight).to(tl.bfloat16)
    tl.store(Y+row*5120+h,y,mask)
    blocks=tl.reshape(y.to(tl.float32),(256,32))
    amax=tl.maximum(tl.max(tl.abs(blocks),1),1.e-4)
    bits=(amax*(1./448.)).to(tl.uint32,bitcast=True)
    rounded=tl.where((bits&0x007fffff)!=0,(bits+0x00800000)&0x7f800000,bits)
    sf=(rounded>>23)&255
    # Same exponent intrinsic as the current native eight-lane quantizer.
    inv=tl.inline_asm_elementwise(
        "{.reg .pred p; .reg .s32 e; .reg .f32 f,s; setp.eq.u32 p,$1,0; sub.s32 e,127,$1; cvt.rn.f32.s32 f,e; ex2.approx.f32 s,f; selp.f32 $0,0f00000000,s,p;}",
        constraints="=f,r",args=[sf],dtype=tl.float32,is_pure=True,pack=1)
    q=tl.reshape((blocks*inv[:,None]).to(tl.float8e4nv),(8192,))
    tl.store(Q+row*5120+h,q,mask)
    group=tl.arange(0,256)
    selected=(group>=part*(160//PARTS))&(group<(part+1)*(160//PARTS))
    tl.store(SR+row*160+group,sf.to(tl.uint8),selected)
    offset=(row%32)*16+((row//32)%4)*4+(row//128)*20480+(group//4)*512+(group%4)
    tl.store(SM+offset,sf.to(tl.uint8),selected)
