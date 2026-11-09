#!/usr/bin/env python3
"""CPU-only checks that derived full matrices place every shard byte where the loader expects it."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stage_full_matrices', ROOT/'tools/stage_full_matrices.py')
stage = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage)

# Same tensors, splits and artifact names, at a small geometry (multiples of the 32x32 scale block).
SMALL = (
    ('claude-wob-full-v1', '', (0, 1, 2), ('attn.wo_b',), 'columns', (64, 32)),
    ('claude-shared-full-v1', 'w13', (0, 1, 2), ('ffn.shared_experts.w1', 'ffn.shared_experts.w3'), 'rows', (32, 64)),
    ('claude-shared-full-v1', 'w2', (0, 1, 2), ('ffn.shared_experts.w2',), 'columns', (64, 32)),
    ('claude-engram-wkv-full-v1', '', (1,), ('engram.wkv',), 'rows', (96, 64)),
)


def write_atlases(data, checkpoint='c'*64):
    """Random staged atlases with only the views the tool reads; returns the qualified shard bytes."""
    shards = {}
    for rank in range(4):
        for layer, tensors in stage.needed().items():
            views, blob = {}, b''
            for name, (rows, cols) in tensors.items():
                for kind, shape, dtype in (('weight', [rows, cols], 'F8_E4M3'),
                                           ('scale', [rows//32, cols//32], 'F8_E8M0')):
                    raw = os.urandom(shape[0]*shape[1])
                    views[f'layers.{layer}.{name}.{kind}'] = dict(offset=len(blob), bytes=len(raw), shape=shape, dtype=dtype)
                    shards[(rank, layer, name, kind)] = raw
                    blob += raw
            out = data/'staged'/f'layer{layer}-v1'/f'r{rank}'
            out.mkdir(parents=True)
            (out/'atlas.bin').write_bytes(blob)
            (out/'atlas.json').write_text(json.dumps(dict(
                version=1, rank=rank, world=4, prefixes=[f'layers.{layer}.'], checkpoint_sha256=checkpoint,
                atlas_sha256=stage.sha(blob), bytes=len(blob), views=views)))
    return shards


class StageFullMatrices(unittest.TestCase):
    def setUp(self):
        self.saved = stage.MATRICES, stage.ARTIFACTS
        stage.MATRICES = SMALL
        stage.ARTIFACTS = tuple(dict.fromkeys(m[0] for m in SMALL))
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        (self.data/'kernels').mkdir()

    def tearDown(self):
        stage.MATRICES, stage.ARTIFACTS = self.saved
        self.tmp.cleanup()

    def build(self):
        parts = self.data/'parts'
        for rank in range(4):
            stage.extract(self.data, rank, parts)
        stage.assemble(self.data, parts)
        return parts

    def test_each_rank_finds_its_own_shard_like_the_loader(self):
        shards = write_atlases(self.data)
        self.build()
        for artifact, stem, layers, names, split, (rows, cols) in SMALL:
            manifest = json.loads((self.data/'kernels'/artifact/'manifest.json').read_text())
            tag = stem+'-' if stem else ''
            for layer in layers:
                for kind, (r, c) in (('weight', (rows, cols)), ('scale', (rows//32, cols//32))):
                    full = (self.data/'kernels'/artifact/f'l{layer}-{tag}{kind}.bin').read_bytes()
                    self.assertEqual(stage.sha(full), manifest['layers'][str(layer)][f'{tag}{kind}_sha256'])
                    for rank in range(4):
                        own = b''.join(shards[(rank, layer, name, kind)] for name in names)
                        if split == 'rows':  # staged_full::load_parts, Split::Rows
                            block = len(full)//4
                            self.assertEqual(full[rank*block:(rank+1)*block], own)
                        else:                # Split::Columns: column block `rank` of every row
                            k = 4*c
                            got = b''.join(full[i*k+rank*c:i*k+(rank+1)*c] for i in range(r))
                            self.assertEqual(got, own)

    def test_rerun_verifies_and_refuses_stale_or_altered_outputs(self):
        write_atlases(self.data)
        parts = self.build()
        stage.assemble(self.data, parts)  # unchanged inputs: verified, not rewritten
        wob = self.data/'kernels/claude-wob-full-v1/l0-weight.bin'
        raw = bytearray(wob.read_bytes()); raw[0] ^= 1; wob.write_bytes(bytes(raw))
        with self.assertRaisesRegex(AssertionError, 'differs from its manifest'):
            stage.assemble(self.data, parts)
        raw[0] ^= 1; wob.write_bytes(bytes(raw))
        record = json.loads((parts/'r2/sources.json').read_text())
        record['atlases']['1'] = '0'*64
        (parts/'r2/sources.json').write_text(json.dumps(record))
        with self.assertRaisesRegex(AssertionError, 'different staged atlases'):
            stage.assemble(self.data, parts)

    def test_rejects_mixed_checkpoints_and_unexpected_geometry(self):
        write_atlases(self.data)
        meta_path = self.data/'staged/layer1-v1/r3/atlas.json'
        meta = json.loads(meta_path.read_text())
        meta['checkpoint_sha256'] = 'd'*64
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(AssertionError, 'different checkpoints'):
            stage.extract(self.data, 3, self.data/'parts')
        meta['checkpoint_sha256'] = 'c'*64
        meta['views']['layers.1.engram.wkv.weight']['shape'] = [96, 32]
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(AssertionError, 'unexpected staged'):
            stage.extract(self.data, 3, self.data/'parts-2')

    def test_production_geometry_matches_the_loader(self):
        geometry = {(a, s): (split, rows, cols, len(names)) for a, s, _, names, split, (rows, cols) in self.saved[0]}
        full = {key: ((rows, 4*cols) if split == 'columns' else (4*n*rows, cols))
                for key, (split, rows, cols, n) in geometry.items()}
        # n x k passed to staged_full::load / load_parts in decoder.rs, ffn.rs and engram_gpu.rs.
        self.assertEqual(full, {('claude-wob-full-v1', ''): (5120, 8192),
                                ('claude-shared-full-v1', 'w13'): (4608, 5120),
                                ('claude-shared-full-v1', 'w2'): (5120, 2304),
                                ('claude-engram-wkv-full-v1', ''): (25600, 6144)})


if __name__ == '__main__':
    unittest.main()
