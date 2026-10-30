@triton.jit
def quantize_draft_head(W, Q, S, N: tl.constexpr, K: tl.constexpr):
    block_n = tl.program_id(0)
    block_k = tl.program_id(1)
    i = tl.arange(0, 1024)
    n = block_n * 32 + i // 32
    k = block_k * 32 + i % 32
    values = tl.load(W + n * K + k, n < N, 0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), 0), 2.0 ** (-126))
    bits = amax.to(tl.int32, bitcast=True)
    exponent = (bits >> 23 & 255) - 135 + (bits & 8388607 > 6291456).to(tl.int32)
    exponent = tl.minimum(tl.maximum(exponent, -127), 127)
    scale_bits = tl.where(exponent == -127, 4194304, exponent + 127 << 23)
    scale = scale_bits.to(tl.float32, bitcast=True)
    quantized = tl.div_rn(values, scale).to(tl.float8e4nv)
    tl.store(Q + n * K + k, quantized, n < N)
    tl.store(S + block_n * (K // 32) + block_k, exponent + 127)
