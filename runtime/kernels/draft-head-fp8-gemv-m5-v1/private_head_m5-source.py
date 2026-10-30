@triton.jit
def fp8_batched_gemv(X, W8, S, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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
        valid = (col[:, None] < N) & (k[None, :] < K)
        raw = tl.load(W8 + col[:, None] * K + k[None, :], valid, other=0.0).to(tl.float32)
        exponent = tl.load(S + col[:, None] // 32 * (K // 32) + k[None, :] // 32, valid, other=127).to(tl.float32)
        w = (raw * tl.exp2(exponent - 127.0)).to(tl.bfloat16).to(tl.float32)
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
