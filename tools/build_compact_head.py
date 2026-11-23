#!/usr/bin/env python3
"""Build primitive-exports/target-head-lossless-compact-v1 from target-head-lossless-weight-cg-v1.

The six-row lossless head reads an escaped 32-weight group's original BF16 words with one 16-byte load per
eight-weight slice. This bundle changes only that load: the address becomes W + a relative byte offset that the
existing payload load already holds for escaped slices (written at startup by kernels/compact_head_escapes.ptx), so
W can be a ~0.4 MiB compact table instead of the 315 MiB canonical backing. Every other instruction, the ABI and the
launch geometry are the weight-cg kernel's. Deterministic: the same input PTX and ptxas give the same cubin.
Measured on Nov 22: 610/610 exact prediction points, +0.27 % / +0.59 % decode.

Usage: build_compact_head.py PRIMITIVE_EXPORTS_DIR
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
src = root / 'target-head-lossless-weight-cg-v1'
out = root / 'target-head-lossless-compact-v1'
sha = lambda b: hashlib.sha256(b).hexdigest()
raw = (src / 'manifest.json').read_bytes()
assert sha(raw) == 'd0ff14ca7341e5489a1a252dd8f9e24205fe583118ead1810e501909ba244274', 'weight-cg manifest changed'
ptxas = '/usr/local/cuda/bin/ptxas'
version = subprocess.run([ptxas, '--version'], capture_output=True, text=True, check=True).stdout
assert 'V13.0.88' in version, version
ptx = (src / 'lossless_head_m6.ptx').read_text()
match = '@%p2 ld.global.cg.v4.b32 { %r20, %r21, %r22, %r23 }, [ %rd13 + 0 ];'
assert ptx.count(match) == 1, 'escaped-slice load not found exactly once'
patch = '''{
.reg .b64 addr,offset;
cvt.u64.u32 offset,%r10;
add.u64 addr,%rd4,offset;
@%p2 ld.global.cg.v4.b32 { %r20, %r21, %r22, %r23 }, [addr];
}'''
out.mkdir(exist_ok=False)
(out / 'lossless_head_m6.ptx').write_text(ptx.replace(match, patch))
subprocess.run([ptxas, '-arch=sm_121a', out / 'lossless_head_m6.ptx', '-o', out / 'lossless_head_m6.cubin'], check=True)
cubin = (out / 'lossless_head_m6.cubin').read_bytes()
entry = next(e for e in json.loads(raw)['entries'] if e['name'] == 'lossless_head_m6')
entry['artifact_sha256'] = sha(cubin)
manifest = (json.dumps({'entries': [entry]}, sort_keys=True, separators=(',', ':')) + '\n').encode()
(out / 'manifest.json').write_bytes(manifest)
(out / 'provenance.json').write_text(json.dumps(dict(
    source_manifest_sha256=sha(raw), source_ptx_sha256=sha(ptx.encode()), ptxas=version.strip().splitlines()[-1],
    patched_load=match, cubin_sha256=sha(cubin), manifest_sha256=sha(manifest)), indent=1) + '\n')
print(json.dumps(dict(out=str(out), cubin_sha256=sha(cubin), manifest_sha256=sha(manifest))))
