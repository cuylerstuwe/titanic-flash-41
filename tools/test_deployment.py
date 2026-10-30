#!/usr/bin/env python3
"""CPU-only checks for deployment boundaries and lossless backing layout."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
fleet = module('fleet', ROOT/'deploy/fleet.py')
pack = module('pack_engram', ROOT/'tools/pack_engram.py')
download = module('download_model', ROOT/'tools/download_model.py')

class DeploymentChecks(unittest.TestCase):
    def test_inventory_rejects_wrong_rank_count_or_ambiguous_rails(self):
        original = json.loads((ROOT/'deploy/fleet.example.json').read_text())
        fleet.validate(original)
        for alter in [lambda x: x['nodes'].pop(),
                      lambda x: x['nodes'][1].update(ssh='-oProxyCommand=bad'),
                      lambda x: x['nodes'][1].update(fabric_ipv4=x['nodes'][0]['fabric_ipv4']),
                      lambda x: x.update(root='/'),
                      lambda x: x.update(root='/srv/path with spaces'),
                      lambda x: x.update(http_bind='0.0.0.0:8000', allow_remote=True),
                      lambda x: x.update(http_bind='127.0.0.1:18043'),
                      lambda x: x['nodes'][2].update(hcas=['same', 'same'])]:
            changed=copy.deepcopy(original); alter(changed)
            with self.assertRaises((AssertionError, ValueError)):
                fleet.validate(changed)

    def test_engram_layout_preserves_each_byte_at_row_boundaries(self):
        for count in [1, 2, 17, 257]:
            weights = bytes((i*7+3) % 256 for i in range(count*256))
            scales = bytes((i*11+129) % 256 for i in range(count*8))
            out = pack.interleave(weights, scales, count)
            self.assertEqual(len(out), count*264)
            for row in range(count):
                self.assertEqual(out[row*264:row*264+256], weights[row*256:(row+1)*256])
                self.assertEqual(out[row*264+256:(row+1)*264], scales[row*8:(row+1)*8])

    def test_download_cannot_escape_repository_root(self):
        for name in ['../secret', '/etc/passwd', 'a/../../secret']:
            with self.assertRaises(ValueError): download.safe_name(name)
        self.assertEqual(download.safe_name('model-00001-of-00048.safetensors').name,
                         'model-00001-of-00048.safetensors')

    def test_model_files_accept_only_matching_hash_and_length(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model'; path.write_bytes(b'raw weight bytes')
            record=dict(size=16, lfs=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            self.assertTrue(download.verify(path, record))
            path.write_bytes(b'raw weight bytex')
            self.assertFalse(download.verify(path, record))
            path.write_bytes(b'raw')
            self.assertFalse(download.verify(path, record))

    def test_stop_during_startup_exits_cleanly_instead_of_restarting(self):
        import signal
        with tempfile.TemporaryDirectory() as directory:
            cfg = json.loads((ROOT/'deploy/fleet.example.json').read_text())
            cfg['root'] = directory
            inventory = Path(directory)/'fleet.json'
            inventory.write_text(json.dumps(cfg))
            class Stopping(fleet.Fleet):
                def launch(self):
                    self.stop_file.touch()
                    raise InterruptedError('operator stop')
                def stop(self, state):
                    self.cleaned_up = True
            probe = Stopping(inventory)
            handlers = {s: signal.getsignal(s) for s in [signal.SIGINT, signal.SIGTERM]}
            try:
                probe.serve()
                self.assertTrue(probe.cleaned_up)
            finally:
                for s, handler in handlers.items(): signal.signal(s, handler)

if __name__ == '__main__':
    unittest.main()
