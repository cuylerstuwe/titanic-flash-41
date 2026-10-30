"""GPU-only preparation of a private, lossy DSpark proposal-head copy.

Canonical target weights are read-only. This follows the qualified adapter's
32x32 E4M3/power-of-two format. The exponent calculation is exact for finite
BF16 input values, including subnormals and powers on the 448 boundary.
"""
import triton
import triton.language as tl


@triton.jit
def quantize_draft_head(W, Q, S, N: tl.constexpr, K: tl.constexpr):
    block_n = tl.program_id(0)
    block_k = tl.program_id(1)
    i = tl.arange(0, 1024)
    n = block_n * 32 + i // 32
    k = block_k * 32 + i % 32
    values = tl.load(W + n * K + k, n < N, 0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), 0), 2.0 ** -126)
    bits = amax.to(tl.int32, bitcast=True)
    # ceil(log2(amax / 448)), where 448 = 1.75 * 2**8.
    exponent = ((bits >> 23) & 255) - 135 + ((bits & 0x7fffff) > 0x600000).to(tl.int32)
    exponent = tl.minimum(tl.maximum(exponent, -127), 127)
    scale_bits = tl.where(exponent == -127, 0x00400000, (exponent + 127) << 23)
    scale = scale_bits.to(tl.float32, bitcast=True)
    # Rounded division also handles the smallest admitted scale, whose inverse
    # would overflow FP32. Do not replace it with a reciprocal multiplication.
    quantized = tl.div_rn(values, scale).to(tl.float8e4nv)
    tl.store(Q + n * K + k, quantized, n < N)
    tl.store(S + block_n * (K // 32) + block_k, exponent + 127)
