@triton.jit
def tiled_head(X, W, Y, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    row = tl.arange(0, 16)
    col = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    span: tl.constexpr = tl.cdiv(K, SPLIT)
    inner = split * span + tl.arange(0, BK)
    acc = tl.zeros((16, BN), tl.float32)
    for block in range(tl.cdiv(span, BK)):
        k = inner + block * BK
        valid_k = (k < K) & (k < (split + 1) * span)
        x = tl.load(X + row[:, None] * K + k[None, :], (row[:, None] < M) & valid_k[None, :], other=0)
        w = tl.load(W + col[None, :] * K + k[:, None], (col[None, :] < N) & valid_k[:, None], other=0)
        acc = tl.dot(x, w, acc)
    offset = row[:, None] * N + col[None, :]
    valid = (row[:, None] < M) & (col[None, :] < N)
    if SPLIT == 1:
        tl.store(Y + offset, acc, valid)
    else:
        tl.store(P + split * M * N + offset, acc, valid)
