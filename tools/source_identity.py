#!/usr/bin/env python3
"""Stable identity of the compiled source, independent of checkout location."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def identity():
    files = {}
    for name in ['Cargo.toml', 'Cargo.lock', 'src', 'kernels', 'tests', 'examples']:
        p = ROOT / name
        paths = [p] if p.is_file() else sorted(p.rglob('*'))
        for f in paths:
            if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc':
                files[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    payload = json.dumps(dict(features='serving', files=files), sort_keys=True,
                         separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()

if __name__ == '__main__':
    print(identity())
