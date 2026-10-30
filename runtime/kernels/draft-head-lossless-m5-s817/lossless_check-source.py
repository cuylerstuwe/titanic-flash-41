@triton.jit
def check_head(W, SM, Codes, Base, Errors, SIZE: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    base = tl.load(Base + i // 32, i < SIZE, other=255).to(tl.uint32)
    sm = tl.load(SM + i, i < SIZE, other=0).to(tl.uint32)
    code = tl.load(Codes + i // 8, i < SIZE, other=0) >> i % 8 * 4 & 15
    bits = sm & 127 | (sm & 128) << 8 | base + code << 7
    original = tl.load(W + i, i < SIZE, other=0).to(tl.uint16, bitcast=True)
    decoded = tl.where(base == 255, original, bits).to(tl.uint16)
    tl.store(Errors + tl.program_id(0), tl.sum(((decoded != original) & (i < SIZE)).to(tl.uint32), 0))
