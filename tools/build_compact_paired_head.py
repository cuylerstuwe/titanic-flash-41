#!/usr/bin/env python3
"""Build primitive-exports/target-head-lossless-paired-compact-v1 from target-head-lossless-paired-v1.

Patches the single escaped-group BF16 load in lossless_head_m9.ptx and lossless_head_m12.ptx so that
escaped slices read from the compact ~0.4 MiB table (W + offset from %r10, written at startup by
kernels/compact_head_escapes.ptx) instead of the 315 MiB canonical head.weight backing.
Matches tools/build_compact_head.py (M6) so that M6, M9, and M12 all operate exclusively on the
encoded payload + compact escape table and allow dropping canonical head.weight (+315 MiB/rank UMA).

Usage: build_compact_paired_head.py PRIMITIVE_EXPORTS_DIR
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
src = root / 'target-head-lossless-paired-v1'
out = root / 'target-head-lossless-paired-compact-v1'
sha = lambda b: hashlib.sha256(b).hexdigest()
raw = (src / 'manifest.json').read_bytes()
assert sha(raw) == 'f70610b580a96a550b9120a8780867502b2c830c3147528a3fe9a52a9e3b6b97', 'paired lossless manifest changed'
ptxas = '/usr/local/cuda/bin/ptxas'
version = subprocess.run([ptxas, '--version'], capture_output=True, text=True, check=True).stdout
assert 'V13.0.88' in version, version

match = '@%p2 ld.global.v4.b32 { %r20, %r21, %r22, %r23 }, [ %rd13 + 0 ];'
patch = '''{
.reg .b64 addr,offset;
cvt.u64.u32 offset,%r10;
add.u64 addr,%rd4,offset;
@%p2 ld.global.cg.v4.b32 { %r20, %r21, %r22, %r23 }, [addr];
}'''

out.mkdir(exist_ok=False)
src_entries = {e['name']: e for e in json.loads(raw)['entries']}
out_entries = []
prov_items = {}
for name in ('lossless_head_m9', 'lossless_head_m12'):
    ptx = (src / f'{name}.ptx').read_text()
    assert ptx.count(match) == 1, f'{name}: escaped-slice load not found exactly once'
    assert 'ld.param.b64 \t%rd4, [lossless_simt16_param_1];' in ptx, f'{name}: W param register mismatch'
    assert '@%p1 ld.global.v2.b32 { %r10, %r11 }, [ %rd11 + 0 ];' in ptx, f'{name}: SM payload register mismatch'
    (out / f'{name}.ptx').write_text(ptx.replace(match, patch))
    subprocess.run([ptxas, '-arch=sm_121a', out / f'{name}.ptx', '-o', out / f'{name}.cubin'], check=True)
    cubin = (out / f'{name}.cubin').read_bytes()
    entry = dict(src_entries[name])
    entry['artifact_sha256'] = sha(cubin)
    out_entries.append(entry)
    prov_items[name] = dict(source_ptx_sha256=sha(ptx.encode()), cubin_sha256=sha(cubin))

manifest = (json.dumps({'entries': out_entries}, sort_keys=True, separators=(',', ':')) + '\n').encode()
(out / 'manifest.json').write_bytes(manifest)
(out / 'provenance.json').write_text(json.dumps(dict(
    source_manifest_sha256=sha(raw), ptxas=version.strip().splitlines()[-1],
    patched_load=match, kernels=prov_items, manifest_sha256=sha(manifest)), indent=1) + '\n')
print(json.dumps(dict(out=str(out), manifest_sha256=sha(manifest), kernels=prov_items)))
