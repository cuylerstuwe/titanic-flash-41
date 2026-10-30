@triton.jit
def pack_head(W, SM, Codes, Base, GROUPS: tl.constexpr, BG: tl.constexpr):
    group = tl.program_id(0) * BG + tl.arange(0, BG)
    lane = tl.arange(0, 32)
    bits = tl.load(W + group[:, None] * 32 + lane[None, :], group[:, None] < GROUPS, other=0).to(tl.uint16, bitcast=True).to(tl.uint32)
    exponent = bits >> 7 & 255
    low = tl.min(exponent, 1)
    high = tl.max(exponent, 1)
    escape = (high - low > 15) | (high == 255)
    tl.store(Base + group, tl.where(escape, 255, low), group < GROUPS)
    tl.store(SM + group[:, None] * 32 + lane[None, :], bits & 127 | bits >> 8 & 128, group[:, None] < GROUPS)
    delta = exponent - low[:, None] & 15
    for word in tl.static_range(4):
        fragment = tl.where(lane[None, :] // 8 == word, delta << lane[None, :] % 8 * 4, 0)
        tl.store(Codes + group * 4 + word, tl.sum(fragment, 1), group < GROUPS)
