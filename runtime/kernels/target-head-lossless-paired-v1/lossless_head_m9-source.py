@triton.jit
def lossless_simt16(X, W, SM, Codes, Base, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    col = tl.program_id(0) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    a0 = tl.zeros((BN, BK), tl.float32)
    a1 = tl.zeros((BN, BK), tl.float32)
    a2 = tl.zeros((BN, BK), tl.float32)
    a3 = tl.zeros((BN, BK), tl.float32)
    a4 = tl.zeros((BN, BK), tl.float32)
    a5 = tl.zeros((BN, BK), tl.float32)
    a6 = tl.zeros((BN, BK), tl.float32)
    a7 = tl.zeros((BN, BK), tl.float32)
    a8 = tl.zeros((BN, BK), tl.float32)
    a9 = tl.zeros((BN, BK), tl.float32)
    a10 = tl.zeros((BN, BK), tl.float32)
    a11 = tl.zeros((BN, BK), tl.float32)
    a12 = tl.zeros((BN, BK), tl.float32)
    a13 = tl.zeros((BN, BK), tl.float32)
    a14 = tl.zeros((BN, BK), tl.float32)
    a15 = tl.zeros((BN, BK), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = inner + block * BK
        offset = col[:, None] * K + k[None, :]
        valid = (col[:, None] < N) & (k[None, :] < K)
        base = tl.load(Base + offset // 32, valid, other=255).to(tl.uint32)
        sm = tl.load(SM + offset, valid, other=0).to(tl.uint32)
        code = tl.load(Codes + offset // 8, valid, other=0) >> offset % 8 * 4 & 15
        bits = sm & 127 | (sm & 128) << 8 | base + code << 7
        fallback = tl.load(W + offset, valid & (base == 255), other=0).to(tl.uint16, bitcast=True)
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
        if M > 6:
            x = tl.load(X + 6 * K + k, k < K, other=0).to(tl.float32)
            a6 = tl.fma(w, x[None, :], a6)
        if M > 7:
            x = tl.load(X + 7 * K + k, k < K, other=0).to(tl.float32)
            a7 = tl.fma(w, x[None, :], a7)
        if M > 8:
            x = tl.load(X + 8 * K + k, k < K, other=0).to(tl.float32)
            a8 = tl.fma(w, x[None, :], a8)
        if M > 9:
            x = tl.load(X + 9 * K + k, k < K, other=0).to(tl.float32)
            a9 = tl.fma(w, x[None, :], a9)
        if M > 10:
            x = tl.load(X + 10 * K + k, k < K, other=0).to(tl.float32)
            a10 = tl.fma(w, x[None, :], a10)
        if M > 11:
            x = tl.load(X + 11 * K + k, k < K, other=0).to(tl.float32)
            a11 = tl.fma(w, x[None, :], a11)
        if M > 12:
            x = tl.load(X + 12 * K + k, k < K, other=0).to(tl.float32)
            a12 = tl.fma(w, x[None, :], a12)
        if M > 13:
            x = tl.load(X + 13 * K + k, k < K, other=0).to(tl.float32)
            a13 = tl.fma(w, x[None, :], a13)
        if M > 14:
            x = tl.load(X + 14 * K + k, k < K, other=0).to(tl.float32)
            a14 = tl.fma(w, x[None, :], a14)
        if M > 15:
            x = tl.load(X + 15 * K + k, k < K, other=0).to(tl.float32)
            a15 = tl.fma(w, x[None, :], a15)
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
    if M > 6:
        tl.store(Y + 6 * N + col, tl.sum(a6, 1), col < N)
    if M > 7:
        tl.store(Y + 7 * N + col, tl.sum(a7, 1), col < N)
    if M > 8:
        tl.store(Y + 8 * N + col, tl.sum(a8, 1), col < N)
    if M > 9:
        tl.store(Y + 9 * N + col, tl.sum(a9, 1), col < N)
    if M > 10:
        tl.store(Y + 10 * N + col, tl.sum(a10, 1), col < N)
    if M > 11:
        tl.store(Y + 11 * N + col, tl.sum(a11, 1), col < N)
    if M > 12:
        tl.store(Y + 12 * N + col, tl.sum(a12, 1), col < N)
    if M > 13:
        tl.store(Y + 13 * N + col, tl.sum(a13, 1), col < N)
    if M > 14:
        tl.store(Y + 14 * N + col, tl.sum(a14, 1), col < N)
    if M > 15:
        tl.store(Y + 15 * N + col, tl.sum(a15, 1), col < N)
