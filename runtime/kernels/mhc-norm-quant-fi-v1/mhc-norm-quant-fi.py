"""HC combine/RMS plus the actual small-row FlashInfer MXFP8 quantizer.

HC math adapted from SGLang f80c91a4b97e52e17fbcff088aa6b47b6270f6b9
_hc_combine_norm. Quantization adapted from FlashInfer's mxfp8_quantize.py
and quantization_cute_dsl_utils.py in the pinned deployment image
sha256:82bb585723d16d25423646dd9b1300c51f5913f28c5e3900d5cc8eaf52f8aa4d.
Both Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0.
Preserves ordered FP32 FMAs, BF16 combine, RMS and BF16 normalized boundaries.
Uses FlashInfer's no-floor UE8M0 ceiling, zero handling and integer inverse
scale construction. Writes normalized BF16 for routing/indexing as before.
Component prototype; no model-weight or target-precision changes.
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
    amax=tl.max(tl.abs(blocks),1)
    normalized_max=amax*(1./448.)
    bits=normalized_max.to(tl.uint32,bitcast=True)
    exp=(bits>>23)&255
    mantissa=bits&0x7fffff
    bump=tl.where((exp==0)&(mantissa<=0x400000),0,(mantissa!=0).to(tl.uint32))
    sf=tl.where(normalized_max<=0,0,tl.minimum(exp+bump,254)).to(tl.uint32)
    # Match FlashInfer's exact exponent construction, including sf==0.
    inv=tl.where(sf==0,0,(tl.maximum(254-sf.to(tl.int32),0).to(tl.uint32)<<23)).to(tl.float32,bitcast=True)
    q=tl.reshape((blocks*inv[:,None]).to(tl.float8e4nv),(8192,))
    tl.store(Q+row*5120+h,q,mask)
    group=tl.arange(0,256)
    selected=(group>=part*(160//PARTS))&(group<(part+1)*(160//PARTS))
    tl.store(SR+row*160+group,sf.to(tl.uint8),selected)
    offset=(row%32)*16+((row//32)%4)*4+(row//128)*20480+(group//4)*512+(group%4)
    tl.store(SM+offset,sf.to(tl.uint8),selected)
