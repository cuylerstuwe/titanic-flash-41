@triton.jit
def lossless_gemv(X, W, SM, Codes, Base, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    col = tl.program_id(0) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    a0 = tl.zeros((BN, BK), tl.float32)
    a1 = tl.zeros((BN, BK), tl.float32)
    a2 = tl.zeros((BN, BK), tl.float32)
    a3 = tl.zeros((BN, BK), tl.float32)
    a4 = tl.zeros((BN, BK), tl.float32)
    a5 = tl.zeros((BN, BK), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = inner + block * BK
        offset = col[:, None] * K + k[None, :]
        valid = (col[:, None] < N) & (k[None, :] < K)
        base = tl.load(Base + offset // 32, valid, other=255).to(tl.uint32)
        sm = tl.load(SM + offset, valid, other=0, cache_modifier='.cg' if M == 6 else '').to(tl.uint32)
        code = tl.load(Codes + offset // 8, valid, other=0) >> offset % 8 * 4 & 15
        bits = sm & 127 | (sm & 128) << 8 | base + code << 7
        fallback = tl.load(W + offset, valid & (base == 255), other=0, cache_modifier='.cg' if M == 6 else '').to(tl.uint16, bitcast=True)
        bits = tl.where(base == 255, fallback, bits).to(tl.uint16)
        w = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        x = tl.load(X + k, k < K, other=0).to(tl.float32)
        a0 = tl.fma(w, x[None, :], a0)
        if M > 1:
            x = tl.load(X + K + k, k < K, other=0).to(tl.float32)
            a1 = tl.fma(w, x[None, :], a1)
        if M > 2:
            x = tl.load(X + 2 * K + k, k < K, other=0).to(tl.float32)
            a2 = tl.fma(w, x[None, :], a2)
        if M > 3:
            x = tl.load(X + 3 * K + k, k < K, other=0).to(tl.float32)
            a3 = tl.fma(w, x[None, :], a3)
        if M > 4:
            x = tl.load(X + 4 * K + k, k < K, other=0).to(tl.float32)
            a4 = tl.fma(w, x[None, :], a4)
        if M > 5:
            x = tl.load(X + 5 * K + k, k < K, other=0).to(tl.float32)
            a5 = tl.fma(w, x[None, :], a5)
    tl.store(Y + col, tl.sum(a0, 1), col < N)
    if M > 1:
        tl.store(Y + N + col, tl.sum(a1, 1), col < N)
    if M > 2:
        tl.store(Y + 2 * N + col, tl.sum(a2, 1), col < N)
    if M > 3:
        tl.store(Y + 3 * N + col, tl.sum(a3, 1), col < N)
    if M > 4:
        tl.store(Y + 4 * N + col, tl.sum(a4, 1), col < N)
    if M > 5:
        tl.store(Y + 5 * N + col, tl.sum(a5, 1), col < N)
