@triton.jit
def markov_head_scaled(X, W, Base, Y, N: tl.constexpr, BN: tl.constexpr, STRENGTH: tl.constexpr):
    col = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, 256)
    x = tl.load(X + k).to(tl.float32)
    w = tl.load(W + col[:, None] * 256 + k[None, :], col[:, None] < N, other=0).to(tl.float32)
    dot = tl.sum(w * x[None, :], 1)
    base = tl.load(Base + col, col < N, other=0)
    tl.store(Y + col, STRENGTH * dot + base, col < N)
