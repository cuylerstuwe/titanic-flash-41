#!/usr/bin/env python3
"""Install the checked kernel bundle and build the separately fetched WO-A component."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SGLANG = 'f80c91a4b97e52e17fbcff088aa6b47b6270f6b9'
MIA = '64b02dea8681fb1d8e3a659ea432a8e9e9ba3995'

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def fetch(url, path, expected=None):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=90) as r:
            raw = r.read()
        if expected:
            assert hashlib.sha256(raw).hexdigest() == expected, 'upstream source identity mismatch'
        temp = path.with_suffix(path.suffix+'.part')
        temp.write_bytes(raw); temp.replace(path)
    if expected:
        assert digest(path) == expected, 'existing upstream source identity mismatch'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data', type=Path)
    ap.add_argument('--with-external-woa', action='store_true',
                    help='fetch/build the required separately licensed AGPL-3.0 operator')
    a = ap.parse_args()
    if not a.with_external_woa:
        ap.error('read THIRD_PARTY_NOTICES.md, then use --with-external-woa; this component is required')
    inventory = json.loads((ROOT/'runtime/artifacts.json').read_text())
    for name, expected in inventory['files'].items():
        src = ROOT/'runtime'/name
        assert digest(src) == expected, f'bundled artifact mismatch: {name}'
        target = a.data/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert digest(target) == expected, f'preserving different existing artifact: {name}'
        else:
            shutil.copy2(src, target)
    external = a.data/'third-party'
    fetch(f'https://raw.githubusercontent.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/{MIA}/adapter/wo_a_w8.py',
          external/'mia/adapter/wo_a_w8.py', 'e642f230de3d2f2da94ff19e8d6f0b5565a77450fb012851be1c61931cd53901')
    fetch(f'https://raw.githubusercontent.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/{MIA}/LICENSE', external/'mia/LICENSE')
    for file in ['kernels/ops/attention/dsv4/wo_a_bf16.py', 'kernels/ops/layernorm/mxfp8_epilogue.py']:
        fetch(f'https://raw.githubusercontent.com/sgl-project/sglang/{SGLANG}/python/sglang/{file}', external/'sglang'/file)
    fetch(f'https://raw.githubusercontent.com/sgl-project/sglang/{SGLANG}/LICENSE', external/'sglang/LICENSE')
    # The builder has fixed container paths so its ABI/source identity is stable.
    assert a.data.resolve() == Path('/data'), 'run the setup command in the documented container at /data'
    out = a.data/'kernels/woa-v1'
    if (out/'manifest.json').exists():
        manifest = json.loads((out/'manifest.json').read_text())
        assert manifest['mia_source_sha256'] == digest(external/'mia/adapter/wo_a_w8.py')
        assert manifest['sglang_source_sha256'] == digest(external/'sglang/kernels/ops/attention/dsv4/wo_a_bf16.py')
        for e in manifest['entries']:
            assert digest(out/(e['name']+'.cubin')) == e['artifact_sha256']
    else:
        assert not out.exists(), 'partial external export preserved; inspect it before retrying'
        (a.data/'build').mkdir(exist_ok=True)
        subprocess.run([sys.executable, str(ROOT/'tools/build_external_woa.py'), str(out)], check=True)
    print('Kernel and model metadata setup verified. Model weights are staged separately.')

if __name__ == '__main__':
    main()
