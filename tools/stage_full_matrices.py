#!/usr/bin/env python3
"""Derive the full projection matrices used by the s1387 sequence-parallel prefill options.

Those options run a few projections on each rank's own rows, so every rank needs the whole matrix
instead of its TP4 shard. This tool rebuilds each matrix from the staged per-rank atlases by placing
the four shards in rank order. It copies bytes only: FP8 E4M3 values and canonical 32x32 UE8M0
scales are never converted. At load, each rank also checks that its own part of every matrix equals
the shard it serves, so a wrong or stale file stops startup instead of changing outputs.

On the head, after `stage_weights.py --rank all` and before `sync_fleet.py`:

    stage_full_matrices.py --data /data

This writes /data/kernels/claude-{wob,shared,engram-wkv}-full-v1 (about 1.9 GB; sync_fleet.py copies
the same files to every rank). If each node holds only its own rank's atlases, run
`stage_full_matrices.py --data /data extract --rank R --parts DIR` on each node, gather the four
DIR/rR folders on one host, and run `stage_full_matrices.py --data /data assemble --parts DIR` there.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ENCODER = tuple(range(21))  # sequence-parallel prefill layers
ENGRAM = (1, 14)
# (artifact, file stem, layers, source tensors, split, per-rank weight shape)
# rows:    rank r's tensors, concatenated in order, form row block r of the full matrix.
# columns: rank r's tensor forms column block r of every row (the four K shards side by side).
MATRICES = (
    ('claude-wob-full-v1', '', ENCODER, ('attn.wo_b',), 'columns', (5120, 2048)),
    ('claude-shared-full-v1', 'w13', ENCODER,
     ('ffn.shared_experts.w1', 'ffn.shared_experts.w3'), 'rows', (576, 5120)),
    ('claude-shared-full-v1', 'w2', ENCODER, ('ffn.shared_experts.w2',), 'columns', (5120, 576)),
    ('claude-engram-wkv-full-v1', '', ENGRAM, ('engram.wkv',), 'rows', (6400, 6144)),
)
ARTIFACTS = tuple(dict.fromkeys(m[0] for m in MATRICES))
FORMAT = ('FP8 E4M3 values and canonical 32x32 UE8M0 scales of the four staged TP4 shards '
          'in rank order; no numerical conversion')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def needed():
    """layer -> {tensor: per-rank weight shape} for every tensor the matrices read."""
    out = {}
    for _, _, layers, names, _, shape in MATRICES:
        for layer in layers:
            for name in names:
                out.setdefault(layer, {})[name] = shape
    return dict(sorted(out.items()))


def extract(data, rank, parts):
    """Copy one rank's shard bytes for every derived matrix out of its staged atlases."""
    out = parts/f'r{rank}'
    tmp = out.with_name(out.name+'.part')
    assert not out.exists() and not tmp.exists(), f'existing part directory preserved: {out}'
    tmp.mkdir(parents=True)
    record = dict(rank=rank, world=4, checkpoint_sha256=None, atlases={}, parts={})
    for layer, tensors in needed().items():
        stage = data/'staged'/f'layer{layer}-v1'/f'r{rank}'
        meta = json.loads((stage/'atlas.json').read_text())
        assert meta['rank'] == rank and meta['world'] == 4, f'{stage}: atlas rank or world'
        assert meta['prefixes'] == [f'layers.{layer}.'], f'{stage}: atlas selection'
        assert (stage/'atlas.bin').stat().st_size == meta['bytes'], f'{stage}: incomplete atlas'
        record['checkpoint_sha256'] = record['checkpoint_sha256'] or meta['checkpoint_sha256']
        assert meta['checkpoint_sha256'] == record['checkpoint_sha256'], 'atlases come from different checkpoints'
        record['atlases'][str(layer)] = meta['atlas_sha256']
        with (stage/'atlas.bin').open('rb') as atlas:
            for name, (rows, cols) in tensors.items():
                for kind, shape, dtype in (('weight', [rows, cols], 'F8_E4M3'),
                                           ('scale', [rows//32, cols//32], 'F8_E8M0')):
                    view = meta['views'][f'layers.{layer}.{name}.{kind}']
                    assert view['shape'] == shape and view['dtype'] == dtype \
                        and view['bytes'] == shape[0]*shape[1], f'unexpected staged layers.{layer}.{name}.{kind}'
                    atlas.seek(view['offset'])
                    raw = atlas.read(view['bytes'])
                    assert len(raw) == view['bytes'], f'{stage}: short atlas read'
                    file = f'l{layer}-{name}.{kind}.bin'
                    (tmp/file).write_bytes(raw)
                    record['parts'][file] = sha(raw)
    (tmp/'sources.json').write_text(json.dumps(record, indent=2)+'\n')
    tmp.replace(out)
    return record


def verify(out, manifest):
    """An existing output must come from the same atlases and match its recorded digests."""
    existing = json.loads((out/'manifest.json').read_text())
    assert {k: existing.get(k) for k in ('schema', 'checkpoint_sha256', 'sources')} == \
        {k: manifest[k] for k in ('schema', 'checkpoint_sha256', 'sources')}, \
        f'{out} was derived from different staged atlases; move it aside and rerun'
    for layer, entry in existing['layers'].items():
        for key, digest in entry.items():
            if key.endswith('_sha256'):
                path = out/f'l{layer}-{key.removesuffix("_sha256")}.bin'
                assert file_sha(path) == digest, f'derived file differs from its manifest: {path}'


def assemble(data, parts):
    """Build each full matrix from the four ranks' parts, in rank order."""
    records = [json.loads((parts/f'r{rank}'/'sources.json').read_text()) for rank in range(4)]
    assert [(r['rank'], r['world']) for r in records] == [(rank, 4) for rank in range(4)], 'part rank layout'
    checkpoint = records[0]['checkpoint_sha256']
    assert all(r['checkpoint_sha256'] == checkpoint for r in records), 'ranks were staged from different checkpoints'

    def shard(rank, layer, names, kind):
        raw = b''
        for name in names:
            file = f'l{layer}-{name}.{kind}.bin'
            piece = (parts/f'r{rank}'/file).read_bytes()
            assert sha(piece) == records[rank]['parts'][file], f'part digest mismatch: r{rank}/{file}'
            raw += piece
        return raw

    assert shutil.disk_usage(data).free > 3*2**30, 'derived matrix disk reserve'
    for artifact in ARTIFACTS:
        specs = [m for m in MATRICES if m[0] == artifact]
        layers = sorted({layer for m in specs for layer in m[2]})
        manifest = dict(schema=artifact, format=FORMAT, checkpoint_sha256=checkpoint, matrices={}, layers={},
                        sources={str(rank): {str(l): records[rank]['atlases'][str(l)] for l in layers}
                                 for rank in range(4)})
        out = data/'kernels'/artifact
        if out.exists():
            verify(out, manifest)
            print(json.dumps(dict(artifact=artifact, verified_existing=True)), flush=True)
            continue
        tmp = out.with_name(out.name+'.part')
        assert not tmp.exists(), f'partial output preserved: {tmp}'
        tmp.mkdir(parents=True)
        for _, stem, layers_, names, split, (rows, cols) in specs:
            n, k = (rows, 4*cols) if split == 'columns' else (4*len(names)*rows, cols)
            manifest['matrices'][stem or 'matrix'] = dict(tensors=list(names), split=split, n=n, k=k)
            tag = stem+'-' if stem else ''
            for layer in layers_:
                entry = manifest['layers'].setdefault(str(layer), {})
                for kind, (r, c) in (('weight', (rows, cols)), ('scale', (rows//32, cols//32))):
                    shards = [shard(rank, layer, names, kind) for rank in range(4)]
                    if split == 'columns':
                        full = b''.join(shards[rank][i*c:(i+1)*c] for i in range(r) for rank in range(4))
                    else:
                        full = b''.join(shards)
                    assert len(full) == 4*len(names)*r*c
                    (tmp/f'l{layer}-{tag}{kind}.bin').write_bytes(full)
                    entry[f'{tag}{kind}_sha256'] = sha(full)
                    entry[f'{tag}{kind}_bytes'] = len(full)
        (tmp/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        tmp.replace(out)
        print(json.dumps(dict(artifact=artifact, bytes=sum(p.stat().st_size for p in out.iterdir()))), flush=True)


def main():
    ap = argparse.ArgumentParser(description='Derive the full matrices for the s1387 prefill options.')
    ap.add_argument('--data', type=Path, default=Path('/data'))
    sub = ap.add_subparsers(dest='command')
    one = sub.add_parser('extract', help="copy one rank's shard bytes from its staged atlases")
    one.add_argument('--rank', type=int, choices=range(4), required=True)
    one.add_argument('--parts', type=Path, required=True)
    sub.add_parser('assemble', help='build the full matrices from four ranks of parts').add_argument(
        '--parts', type=Path, required=True)
    a = ap.parse_args()
    if a.command == 'extract':
        record = extract(a.data, a.rank, a.parts)
        print(json.dumps(dict(rank=a.rank, parts=len(record['parts']))), flush=True)
    elif a.command == 'assemble':
        assemble(a.data, a.parts)
    else:
        parts = a.data/'build/full-matrix-parts'
        assert not parts.exists(), f'previous parts preserved for inspection: {parts}'
        assert shutil.disk_usage(a.data).free > 6*2**30, 'derived matrix disk reserve'
        for rank in range(4):
            extract(a.data, rank, parts)
        assemble(a.data, parts)
        shutil.rmtree(parts)
    print('Full prefill matrices ready.' if a.command != 'extract' else 'Parts ready.', flush=True)


if __name__ == '__main__':
    main()
