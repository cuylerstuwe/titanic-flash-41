"""Bounded PDL experiment for the pinned b12x W4A8 FC1 -> FC2 chain.

Source: b12x a7d7d29b2ef8869086e0ceaa787321f17544e3c9, Apache-2.0.
https://github.com/local-inference-lab/b12x

FC1 retains its ordinary stream dependency on routing/input preparation. Every
FC1 CTA signals that FC2 may start. FC2 opts into PDL and waits before its first
global read; thus only its register/shared-memory setup may overlap FC1. The
following top-k reduction retains its ordinary dependency on FC2 completion.
No arithmetic, routing, shared-memory layout, or output format changes.

CUDA contract:
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
"""
import ast


def patch(source, phase):
    assert phase in (1, 2)
    assert 'griddepcontrol_' not in source and 'use_pdl=' not in source
    start = source.index('    @cute.kernel\n    def kernel(')
    prefix, kernel = source[:start], source[start:]
    if phase == 1:
        anchor = '        tidx, _, _ = cute.arch.thread_idx()\n'
        assert kernel.count(anchor) == 1
        kernel = kernel.replace(anchor,
            '        # Routing/input preparation is an ordinary predecessor.\n'
            '        cute.arch.griddepcontrol_launch_dependents()\n' + anchor)
    else:
        anchor = '            min_blocks_per_mp=2,\n            stream=stream,\n'
        assert prefix.count(anchor) == 1
        prefix = prefix.replace(anchor, anchor + '            use_pdl=True,\n')
        anchor = '        smem_base = shared_ptr_to_u32(storage.words.data_ptr())\n'
        assert kernel.count(anchor) == 1
        kernel = kernel.replace(anchor, anchor +
            '        # No global reads precede this wait. FC1 must finish and\n'
            '        # flush its intermediate before any FC2 operand is read.\n'
            '        cute.arch.griddepcontrol_wait()\n')
    result = prefix + kernel
    ast.parse(result)
    return result
