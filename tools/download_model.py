#!/usr/bin/env python3
"""Download the pinned canonical checkpoint, verifying every remote file.

No model code is imported. LFS SHA256 / Git blob SHA1 identities come from the
immutable revision's metadata. Incomplete downloads use resumable .part files.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

REPO = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'
FLOOR = 35 * 2**30

def request(url, headers=None):
    # The checkpoint is public. No credential is needed or forwarded to a CDN.
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120)

def safe_name(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('unsafe checkpoint filename')
    return p

def verify(path, record):
    if not path.is_file() or path.stat().st_size != record['size']:
        return False
    lfs = record.get('lfs')
    h = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        h.update(f"blob {record['size']}\0".encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest() == (lfs['sha256'] if lfs else record['blobId'])

def download(root, record):
    path = root / safe_name(record['rfilename'])
    if path.exists():
        if not verify(path, record):
            raise RuntimeError(f'existing file failed verification; preserve and inspect: {path.name}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    if part.exists() and part.stat().st_size > record['size']:
        raise RuntimeError(f'oversized partial file: {part.name}')
    for attempt in range(5):
        offset = part.stat().st_size if part.exists() else 0
        if offset == record['size']:
            break
        if shutil.disk_usage(root).free < record['size'] - offset + FLOOR:
            raise RuntimeError('insufficient free storage with 35 GiB reserve')
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{record['rfilename']}"
        try:
            with request(url, {'Range': f'bytes={offset}-'} if offset else {}) as response:
                if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                    raise RuntimeError('server did not honor resume offset; partial file preserved')
                with part.open('ab' if offset else 'wb') as out:
                    shutil.copyfileobj(response, out, 8 << 20)
                    out.flush()
                    os.fsync(out.fileno())
            break
        except (OSError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 10))
    if not verify(part, record):
        raise RuntimeError(f'download hash or size mismatch: {path.name}; .part preserved')
    part.replace(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    ap.add_argument('--metadata-only', action='store_true')
    a = ap.parse_args()
    a.directory.mkdir(parents=True, exist_ok=True)
    with request(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true') as r:
        info = json.load(r)
    assert info['sha'] == REVISION
    records = [r for r in info['siblings'] if r['rfilename'].endswith('.safetensors')
               or r['rfilename'] in ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                                      'model.safetensors.index.json', 'generation_config.json', 'LICENSE']]
    assert sum(r['rfilename'].endswith('.safetensors') for r in records) == 48
    assert all('size' in r and ('blobId' in r or 'lfs' in r) for r in records)
    receipt = dict(repository=REPO, revision=REVISION, files=records)
    dest = a.directory / 'download-manifest.json'
    if dest.exists():
        assert json.loads(dest.read_text()) == receipt, 'immutable metadata changed'
    else:
        dest.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(files=len(records), bytes=sum(r['size'] for r in records), revision=REVISION)), flush=True)
    if a.metadata_only:
        return
    for i, record in enumerate(records):
        download(a.directory, record)
        print(json.dumps(dict(verified=i+1, total=len(records), file=record['rfilename'])), flush=True)

if __name__ == '__main__':
    main()
