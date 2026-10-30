#!/usr/bin/env python3
"""Lossless byte-layout conversion for rank-local Engram SSD backing.

The FP8 row and eight E8M0 scale bytes are copied without dequantization or
numeric conversion. NumPy only performs byte copies; no CPU model arithmetic.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
import numpy as np

MAGIC = 0x31344E4531565344
ROWS = {1: 384006168, 14: 384016682}
CHUNK = 1 << 18

def tensor(model, index, name, columns, rows):
    path = model / index[name]
    assert path.parent == model and path.suffix == '.safetensors'
    f = path.open('rb')
    length = struct.unpack('<Q', f.read(8))[0]
    assert length < 100 << 20
    desc = json.loads(f.read(length))[name]
    assert desc['shape'] == [rows, columns]
    assert desc['dtype'] == ('F8_E4M3' if columns == 256 else 'F8_E8M0')
    start, end = desc['data_offsets']
    assert end - start == rows * columns and 8 + length + end <= path.stat().st_size
    return f, 8 + length + start

def interleave(weights, scales, count):
    assert len(weights) == count * 256 and len(scales) == count * 8
    out = np.empty((count, 264), dtype=np.uint8)
    out[:, :256] = np.frombuffer(weights, np.uint8).reshape(count, 256)
    out[:, 256:] = np.frombuffer(scales, np.uint8).reshape(count, 8)
    return out.tobytes()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--rank', type=int, required=True, choices=range(4))
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    index = json.loads((a.model / 'model.safetensors.index.json').read_text())['weight_map']
    for layer, rows in ROWS.items():
        low, high = rows * a.rank // 4, rows * (a.rank+1) // 4
        size = 4096 + (high - low) * 264
        dest = a.output / f'engram-l{layer}-r{a.rank}of4.bin'
        receipt = dest.with_suffix('.json')
        if dest.exists():
            assert receipt.exists() and dest.stat().st_size == size, 'incomplete existing Engram shard'
            old = json.loads(receipt.read_text())
            with dest.open('rb') as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == old['sha256']
            continue
        part = dest.with_suffix('.bin.part')
        assert not part.exists(), 'partial shard preserved: inspect/remove it explicitly before retrying'
        assert shutil.disk_usage(a.output).free > size + 35*2**30, 'Engram staging disk reserve'
        prefix = f'layers.{layer}.engram.embed'
        w, wo = tensor(a.model, index, prefix+'.weight', 256, rows)
        s, so = tensor(a.model, index, prefix+'.scale', 8, rows)
        h = hashlib.sha256()
        with w, s, part.open('xb') as out:
            header = struct.pack('<6Q', MAGIC, layer, low, high, rows, 264).ljust(4096, b'\0')
            out.write(header); h.update(header)
            for begin in range(low, high, CHUNK):
                count = min(CHUNK, high-begin)
                w.seek(wo + begin*256); s.seek(so + begin*8)
                raw = interleave(w.read(count*256), s.read(count*8), count)
                out.write(raw); h.update(raw)
            out.flush(); os.fsync(out.fileno())
        assert part.stat().st_size == size
        record = dict(rank=a.rank, layer=layer, world=4, rows=[low, high],
                      bytes=size, sha256=h.hexdigest(), encoding='canonical FP8 + E8M0 bytes')
        temp = receipt.with_suffix('.json.part')
        temp.write_text(json.dumps(record, indent=2)+'\n')
        part.replace(dest); temp.replace(receipt)
        print(json.dumps(record), flush=True)

if __name__ == '__main__':
    main()
