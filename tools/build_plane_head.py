#!/usr/bin/env python3
"""Build primitive-exports/target-head-lossless-compact-planes-v1 from target-head-lossless-weight-cg-v1.

The six-row lossless head kernel with two changes to its weight unpack; thread layout, FP32 FMA order, reduction
and output are untouched:

1. Bit planes (the research exact_plane_ptx). The same 12 bits per
   weight in another arrangement. The SM argument carries a "high" byte per weight:
       sign << 7 | ((exponent - base) & 15) << 3 | mantissa >> 4
   and the Codes argument carries the low four mantissa bits, eight nibbles per u32 (weight i of a slice in bits
   4i..4i+3). The decoded BF16 word is
       ((high & 127) << 4) + (base << 7) | nibble | (high & 128) << 8
   so one byte gives sign, exponent delta and the top of the mantissa with a single shift and add, where the
   stock unpack pairs a mantissa byte with a separately shifted exponent nibble.
2. Compact exceptions (tools/build_compact_head.py). An escaped slice (base 255) reads its eight original BF16
   words from W + offset, where offset is the u32 in the first four bytes of that slice's high-plane bytes
   (unused otherwise) and W is the compact table of escaped groups.

Deterministic: the source manifest and PTX are pinned by sha256, the assembler by version. Run on a GB10 (CPU
only; ptxas must be V13.0.88). The engine pins the manifest sha this prints (src/lossless_head.rs).

Usage: build_plane_head.py PRIMITIVE_EXPORTS_DIR
       build_plane_head.py --ptx SOURCE.ptx OUT.ptx     (the PTX rewrite alone; needs no assembler)
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

SOURCE_MANIFEST = 'd0ff14ca7341e5489a1a252dd8f9e24205fe583118ead1810e501909ba244274'
SOURCE_PTX = 'b068d465252caaa40d3e16a627d72c767f1d951afa5d4b126a34219c73d86f4d'
ESCAPED_LOAD = '@%p2 ld.global.cg.v4.b32 { %r20, %r21, %r22, %r23 }, [ %rd13 + 0 ];'
RELATIVE_LOAD = '''{
.reg .b64 addr,offset;
cvt.u64.u32 offset,%r10;
add.u64 addr,%rd4,offset;
@%p2 ld.global.cg.v4.b32 { %r20, %r21, %r22, %r23 }, [addr];
}'''
# Registers of the weight-cg M6 kernel, in weight order within a thread's eight-weight slice: the byte of the
# SM argument, and the decoded BF16 word the kernel selects against the escaped word.
HIGH_BYTE = (74, 79, 80, 81, 77, 78, 76, 75)
DECODED = (173, 168, 167, 166, 170, 169, 171, 172)

sha = lambda b: hashlib.sha256(b).hexdigest()


def plane_ptx(source):
    """The plane unpack, inserted after the stock unpack so it overrides the eight decoded words."""
    lines = ['.reg .b32 %plane_delta, %plane_nibble, %plane_base, %plane_sign;',
             'shl.b32 %plane_base,%r83,7;']
    for i, (high, dest) in enumerate(zip(HIGH_BYTE, DECODED)):
        lines.extend((f'and.b32 %plane_delta,%r{high},127;',
                      'shl.b32 %plane_delta,%plane_delta,4;',
                      'add.u32 %plane_delta,%plane_delta,%plane_base;',
                      f'bfe.u32 %plane_nibble,%r12,{i * 4},4;',
                      'or.b32 %plane_delta,%plane_delta,%plane_nibble;',
                      f'and.b32 %plane_sign,%r{high},128;',
                      'shl.b32 %plane_sign,%plane_sign,8;',
                      f'or.b32 %r{dest},%plane_delta,%plane_sign;'))
    text, n = re.subn(r'(?=shr.u32\s+%r174, %r21, 16;)', lambda _: '\n'.join(lines) + '\n', source)
    assert n == 1, 'insertion point after the stock unpack not found exactly once'
    return text


def compact_plane_ptx(source):
    assert sha(source.encode()) == SOURCE_PTX, 'weight-cg M6 PTX changed: the register names above are unverified'
    text = plane_ptx(source)
    assert text.count(ESCAPED_LOAD) == 1, 'escaped-slice load not found exactly once'
    return text.replace(ESCAPED_LOAD, RELATIVE_LOAD)


def main():
    if sys.argv[1] == '--ptx':
        Path(sys.argv[3]).write_text(compact_plane_ptx(Path(sys.argv[2]).read_text()))
        return
    root = Path(sys.argv[1])
    src = root / 'target-head-lossless-weight-cg-v1'
    out = root / 'target-head-lossless-compact-planes-v1'
    raw = (src / 'manifest.json').read_bytes()
    assert sha(raw) == SOURCE_MANIFEST, 'weight-cg manifest changed'
    ptxas = '/usr/local/cuda/bin/ptxas'
    version = subprocess.run([ptxas, '--version'], capture_output=True, text=True, check=True).stdout
    assert 'V13.0.88' in version, version
    source = (src / 'lossless_head_m6.ptx').read_text()
    text = compact_plane_ptx(source)
    out.mkdir(exist_ok=False)
    (out / 'lossless_head_m6.ptx').write_text(text)
    subprocess.run([ptxas, '-arch=sm_121a', out / 'lossless_head_m6.ptx', '-o', out / 'lossless_head_m6.cubin'], check=True)
    cubin = (out / 'lossless_head_m6.cubin').read_bytes()
    entry = next(e for e in json.loads(raw)['entries'] if e['name'] == 'lossless_head_m6')
    entry['artifact_sha256'] = sha(cubin)
    manifest = (json.dumps({'entries': [entry]}, sort_keys=True, separators=(',', ':')) + '\n').encode()
    (out / 'manifest.json').write_bytes(manifest)
    (out / 'provenance.json').write_text(json.dumps(dict(
        source_manifest_sha256=sha(raw), source_ptx_sha256=sha(source.encode()), ptxas=version.strip().splitlines()[-1],
        plane_unpack='inserted before "shr.u32 %r174, %r21, 16;"', patched_load=ESCAPED_LOAD,
        arguments='X, compact escape table, high plane (u8 per weight; escaped slices hold table offsets), '
                  'low plane (u32 per eight weights), bases, Y',
        ptx_sha256=sha(text.encode()), cubin_sha256=sha(cubin), manifest_sha256=sha(manifest)), indent=1) + '\n')
    print(json.dumps(dict(out=str(out), ptx_sha256=sha(text.encode()), cubin_sha256=sha(cubin), manifest_sha256=sha(manifest))))


if __name__ == '__main__':
    main()
