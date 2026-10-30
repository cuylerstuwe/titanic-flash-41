#!/usr/bin/env python3
"""CPU-only integrity and dynamic-dependency checks for the prepared runtime."""
import argparse
import ctypes
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data', type=Path, default=Path('/data'), nargs='?')
    a = ap.parse_args()
    expected = json.loads((ROOT/'runtime/artifacts.json').read_text())['files']
    for name, digest in expected.items():
        assert sha(a.data/name) == digest, name
    external = a.data/'kernels/woa-v1'
    manifest = json.loads((external/'manifest.json').read_text())
    assert len(manifest['entries']) == 14
    for entry in manifest['entries']:
        assert sha(external/(entry['name']+'.cubin')) == entry['artifact_sha256']
    provenance = json.loads((a.data/'runtime/provenance.json').read_text())
    assert provenance['serving'] and provenance['vision_protocol'] and not provenance['storage_fault_injection']
    current = subprocess.check_output(['python3', str(ROOT/'tools/source_identity.py')], text=True).strip()
    assert provenance['source_manifest_sha256'] == current, 'compiled source differs from this checkout'
    # libcuda.so.1 comes from the host driver when the serving container is
    # started with --gpus all; it is deliberately absent from a CPU build.
    libraries = sorted((a.data/'kernels').rglob('*.so'))
    for path in [a.data/'runtime/titanic-flash', *libraries]:
        result = subprocess.run(['ldd', str(path)], capture_output=True, text=True, check=True)
        missing = [line.strip() for line in result.stdout.splitlines() if 'not found' in line]
        assert all(line.startswith('libcuda.so.1 ') for line in missing), (str(path), missing)
    nccl = ctypes.CDLL('/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2')
    version = ctypes.c_int()
    assert nccl.ncclGetVersion(ctypes.byref(version)) == 0
    assert version.value == 23007, 'use the pinned NCCL 2.30.7 runtime'
    print(json.dumps(dict(passed=True, checked_artifacts=len(expected), external_operators=14,
                         dynamic_libraries=len(libraries), nccl_version=version.value,
                         source_sha256=current, scope='CPU integrity/ABI dependencies; no GPU inference')))

if __name__ == '__main__':
    main()
