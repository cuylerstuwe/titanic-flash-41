"""Portable controller recovery boundaries; no SSH, systemd or GPU actions."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'deploy'))
import fleet


class HostControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        cfg = json.loads((ROOT/'deploy/fleet.example.json').read_text())
        cfg['root'] = str(self.root)
        inventory = self.root/'fleet.json'
        inventory.write_text(json.dumps(cfg))
        self.fleet = fleet.Fleet(inventory)
        self.policy = dict(code=str(self.fleet.state_dir/'host-policy-code'/('a'*64)),
                           source_sha256='a'*64, cap_mhz=2350, apst_ranks=[0])
        self.state = dict(generation='b'*32, inventory_sha256=self.fleet.digest,
                          containers=['one', 'two', 'three', 'four'], stopped=False,
                          host_policy=self.policy)
        fleet.atomic(self.fleet.state_file, self.state)
        self.calls = []
        self.fail = set()
        self.patches = [patch.object(self.fleet, 'run', side_effect=self.fake_run),
                        patch.object(self.fleet, 'inspect', return_value=None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def fake_run(self, args, rank=0, **kwargs):
        self.assertEqual(args[0], 'python3')
        kind = Path(args[1]).stem
        action = args[4]
        call = (rank, kind, action)
        self.calls.append(call)
        if call in self.fail:
            raise subprocess.TimeoutExpired('node management', 90)
        return subprocess.CompletedProcess(args, 0, json.dumps(dict(
            action={'arm':'armed', 'ensure':'renewed', 'restore':'restored'}[action],
            cap_mhz=2350)), '')

    def test_failed_nvme_restore_still_attempts_every_clock_and_preserves_recovery_state(self):
        self.fail.add((0, 'host_apst', 'restore'))
        with self.assertRaisesRegex(RuntimeError, 'stop incomplete'):
            self.fleet.stop(self.state)
        self.assertEqual([c[0] for c in self.calls if c[1] == 'host_clock'], [0, 1, 2, 3])
        self.assertFalse(self.fleet.state()['stopped'])
        self.fail.clear()
        self.fleet.stop(self.state)
        self.assertTrue(self.fleet.state()['stopped'])

    def test_partial_arm_failure_restores_all_nodes_before_any_model_launch(self):
        self.state['stopped'] = True
        fleet.atomic(self.fleet.state_file, self.state)
        self.fail.add((1, 'host_clock', 'arm'))
        with patch.object(self.fleet, 'preflight', return_value='image-id'), \
             patch.object(self.fleet, 'prepare_host_policy', return_value=self.policy):
            with self.assertRaisesRegex(RuntimeError, 'arm incomplete'):
                self.fleet.launch()
        self.assertEqual([c[0] for c in self.calls if c[1:] == ('host_clock', 'restore')], [0, 1, 2, 3])
        self.assertTrue(self.fleet.state()['stopped'])

    def test_transient_renewal_failure_keeps_model_and_recovers_without_repeated_disk_writes(self):
        self.fail.add((1, 'host_clock', 'ensure'))
        with patch.object(self.fleet, 'stop') as stop:
            self.fleet.maintain_host_policy(self.state)
            self.assertEqual(self.state['host_policy_status'], 'degraded')
            stop.assert_not_called()
        self.fail.clear()
        self.fleet.next_policy_attempt = 0
        self.fleet.maintain_host_policy(self.state)
        self.assertEqual(self.state['host_policy_status'], 'active')
        with patch.object(fleet, 'atomic') as write:
            self.fleet.next_policy_attempt = 0
            self.fleet.maintain_host_policy(self.state)
            write.assert_not_called()

    def test_explicit_stock_and_no_nvme_policy_make_no_host_changes(self):
        self.fleet.cfg['gpu_clock_cap_mhz'] = 0
        for node in self.fleet.cfg['nodes']:
            node['disable_nvme0_apst'] = False
        self.assertIsNone(self.fleet.prepare_host_policy())
        self.state['host_policy'] = None
        self.fleet.host_policies('arm', self.state)
        self.fleet.maintain_host_policy(self.state)
        self.fleet.stop(self.state)
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
