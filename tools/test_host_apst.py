#!/usr/bin/env python3
"""CPU APST state-machine tests; NVMe, systemd and mount observations are fakes."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import host_apst as apst


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.policy = apst.Policy(self.root, 'a' * 32)
        self.policy.runtime_base = self.root / 'ram-fixture'
        self.policy.mountinfo_file = self.root / 'mountinfo'
        self.policy.mountinfo_file.write_text(f'1 0 0:1 / {self.root} rw - tmpfs tmpfs rw\n')
        self.policy.boot_file = self.root / 'boot-id'
        self.policy.boot_file.write_text('first-boot')
        self.raw = bytes(range(256))
        self.value, self.table = 1, self.raw
        self.now, self.epoch = 100., 1000.
        self.guard, self.timer_loaded, self.guard_allowed = False, False, True
        self.actions = []
        self.patches = [patch.object(apst, 'feature', side_effect=lambda: (self.value, self.table)),
                        patch.object(apst, 'apply', side_effect=self.apply),
                        patch.object(apst, 'run', side_effect=self.mock_run),
                        patch.object(apst.time, 'monotonic', side_effect=lambda: self.now),
                        patch.object(apst.time, 'time', side_effect=lambda: self.epoch)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def apply(self, path, value):
        self.actions.append(('apply', value, self.guard))
        self.value, self.table = value, path.read_bytes()

    def mock_run(self, args, check=True):
        if args[:3] == ['systemctl', 'is-active', '--quiet']:
            return subprocess.CompletedProcess(args, 0 if self.guard else 3)
        if args[:4] == ['sudo', '-n', 'systemctl', 'start']:
            if self.timer_loaded and self.guard_allowed:
                self.guard = True
                self.actions.append(('guard-start',))
                return subprocess.CompletedProcess(args, 0)
            return subprocess.CompletedProcess(args, 5)
        if args[:3] == ['sudo', '-n', 'systemd-run']:
            if not self.guard_allowed:
                raise subprocess.CalledProcessError(1, args)
            self.assertIn('--property=StandardOutput=null', args)
            self.guard = self.timer_loaded = True
            self.actions.append(('guard-create',))
            return subprocess.CompletedProcess(args, 0)
        if args[:4] == ['sudo', '-n', 'systemctl', 'stop']:
            self.guard = False
            self.actions.append(('guard-stop',))
            return subprocess.CompletedProcess(args, 0)
        raise AssertionError('unexpected external command: ' + str(args))

    def arm(self):
        self.assertEqual(self.policy.execute('arm')['action'], 'armed')
        self.assertEqual(self.value, 0)
        self.assertTrue(self.guard)

    def state(self):
        return json.loads(self.policy.control.read_text())

    def test_arm_persists_exact_original_before_guarded_feature_change(self):
        self.arm()
        self.assertEqual(self.policy.table.read_bytes(), self.raw)
        self.assertEqual(self.state()['original_value'], 1)
        self.assertEqual(self.actions, [('guard-create',), ('apply', 0, True)])
        self.assertFalse(self.state()['restored'])

    def test_normal_renewal_and_guard_checks_do_not_write_durable_state(self):
        self.arm()
        original = self.policy.control.read_bytes()
        files = sorted(path.name for path in self.policy.directory.iterdir())
        with patch.object(apst.os, 'fsync') as fsync:
            for _ in range(24):
                self.now += 30
                self.epoch += 30
                self.assertEqual(self.policy.execute('ensure')['action'], 'renewed')
                self.assertEqual(self.policy.execute('check')['action'], 'waiting')
            self.assertEqual(fsync.call_count, 0)
        self.assertEqual(self.policy.control.read_bytes(), original)
        self.assertEqual(self.policy.table.read_bytes(), self.raw)
        self.assertEqual(sorted(path.name for path in self.policy.directory.iterdir()), files)
        self.assertFalse((self.policy.directory / 'actions.jsonl').exists())
        self.assertEqual(self.actions, [('guard-create',), ('apply', 0, True)])

    def test_expired_lease_restores_exact_value_and_table(self):
        self.arm()
        self.now += apst.LEASE_SECONDS
        result = self.policy.execute('check')
        self.assertTrue(result['original_table_exact'])
        self.assertEqual((self.value, self.table), (1, self.raw))
        self.assertFalse(self.guard)
        self.assertTrue(self.state()['restored'])
        self.assertFalse(self.policy.lease_file.exists())

    def test_missing_or_corrupt_ram_lease_restores_instead_of_extending(self):
        self.arm()
        self.policy.lease_file.unlink()
        self.policy.execute('check')
        self.assertEqual(self.value, 1)
        self.policy.execute('ensure')
        self.policy.lease_file.write_text('{incomplete')
        self.policy.execute('check')
        self.assertEqual(self.value, 1)
        self.assertTrue(self.state()['restored'])

    def test_rearm_uses_saved_original_instead_of_current_disabled_value(self):
        self.arm()
        self.now += 121
        self.assertEqual(self.value, 0, 'guard has not yet had its scheduled check')
        result = self.policy.execute('ensure')
        self.assertEqual(result['action'], 'rearmed')
        self.assertEqual(self.state()['original_value'], 1)
        self.assertEqual(self.actions[-2:], [('apply', 1, True), ('apply', 0, True)])
        self.policy.execute('restore')
        self.assertEqual((self.value, self.table), (1, self.raw))

    def test_failed_guard_rearm_leaves_original_policy_in_force(self):
        self.arm()
        self.guard = self.guard_allowed = False
        with self.assertRaises(subprocess.CalledProcessError):
            self.policy.execute('ensure')
        self.assertEqual(self.value, 1)
        self.assertTrue(self.state()['restored'])
        self.assertEqual(self.state()['original_value'], 1)
        self.assertFalse(self.guard)

    def test_failed_initial_guard_never_disables_apst(self):
        self.guard_allowed = False
        with self.assertRaises(subprocess.CalledProcessError):
            self.policy.execute('arm')
        self.assertEqual(self.value, 1)
        self.assertEqual(self.actions, [])
        self.assertEqual(self.policy.table.read_bytes(), self.raw)
        self.assertTrue(self.state()['restored'])
        self.assertLessEqual(self.state()['expires_epoch'], self.epoch)

    def test_interrupted_activation_is_recoverable_from_durable_intent(self):
        with patch.object(apst, 'apply', side_effect=OSError('interrupted before apply')):
            with self.assertRaises(OSError):
                self.policy.execute('arm')
        self.assertFalse(self.state()['restored'])
        self.assertEqual(self.value, 1)
        self.assertTrue(self.guard)
        self.assertEqual(self.policy.execute('ensure')['action'], 'rearmed')
        self.assertEqual(self.state()['original_value'], 1)

    def test_boot_identity_invalidates_lease_but_wall_clock_changes_do_not(self):
        self.arm()
        self.epoch -= 1000000
        self.assertEqual(self.policy.execute('check')['action'], 'waiting')
        self.policy.boot_file.write_text('second-boot')
        # A real reboot also discards the owner and lease stored in tmpfs.
        self.policy.owner_file.unlink()
        self.policy.lease_file.unlink()
        self.policy.execute('check')
        self.assertEqual(self.value, 1)

    def test_original_disabled_policy_is_preserved(self):
        self.value = 0
        self.arm()
        self.now += 121
        self.policy.execute('check')
        self.assertEqual(self.state()['original_value'], 0)
        self.assertEqual((self.value, self.table), (0, self.raw))

    def test_legacy_restore_remains_available_without_adopting_a_new_baseline(self):
        self.policy.directory.mkdir(parents=True)
        self.policy.table.write_bytes(self.raw)
        apst.atomic(self.policy.control, dict(original_value=1, table_sha256=hashlib.sha256(self.raw).hexdigest(),
                                            restored=False, expires_epoch=2000))
        self.value, self.guard = 0, True
        with self.assertRaisesRegex(AssertionError, 'legacy'):
            self.policy.execute('ensure')
        self.assertEqual(self.policy.execute('check')['action'], 'waiting')
        self.policy.execute('restore')
        self.assertEqual((self.value, self.table), (1, self.raw))

    def test_invalid_disk_mount_cannot_store_a_renewable_lease(self):
        self.policy.mountinfo_file.write_text(f'1 0 0:1 / {self.root} rw - ext4 disk rw\n')
        with self.assertRaisesRegex(AssertionError, 'tmpfs'):
            self.policy.execute('arm')
        self.assertEqual(self.value, 1)
        self.assertFalse(self.policy.lease_file.exists())

    def test_damaged_baseline_is_not_used_for_restoration(self):
        self.arm()
        self.policy.table.write_bytes(b'X' * 256)
        with self.assertRaisesRegex(AssertionError, 'baseline table changed'):
            self.policy.execute('restore')
        self.assertEqual(self.actions, [('guard-create',), ('apply', 0, True)])

    def test_expiry_check_and_rearm_serialize_to_a_guarded_policy(self):
        self.arm()
        self.now += 121
        barrier = threading.Barrier(2)
        def execute(action):
            barrier.wait(timeout=2)
            return self.policy.execute(action)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            outcomes = list(pool.map(execute, ['check', 'ensure']))
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(self.value, 0)
        self.assertTrue(self.guard)
        self.assertFalse(self.state()['restored'])
        self.assertEqual(self.state()['original_value'], 1)

    def test_transition_history_is_bounded_and_repeated_restore_is_read_only(self):
        self.arm()
        for _ in range(6):
            self.now += 121
            self.policy.execute('ensure')
        self.assertEqual(len(self.state()['transitions']), 8)
        self.policy.execute('restore')
        before = self.policy.control.read_bytes()
        with patch.object(apst.os, 'fsync') as fsync:
            self.policy.execute('restore')
            self.assertEqual(fsync.call_count, 0)
        self.assertEqual(self.policy.control.read_bytes(), before)

    def test_previous_helper_retains_a_readable_conservative_restore_deadline(self):
        self.arm()
        deadline = self.state()['expires_epoch']
        for _ in range(5):
            self.now += 30
            self.epoch += 30
            self.policy.execute('ensure')
        self.assertEqual(self.state()['expires_epoch'], deadline)
        self.assertLess(deadline, self.epoch)
        self.assertIsNotNone(self.policy.lease(self.state()))
        self.assertEqual(self.state()['original_value'], 1)

    def test_partial_group_arm_can_restore_an_unreached_rank(self):
        self.assertEqual(self.policy.execute('restore')['action'], 'not-armed')
        self.assertEqual(self.actions, [])

    def test_new_owner_is_immune_to_delayed_old_checks(self):
        self.arm()
        self.policy.execute('restore')
        newer = apst.Policy(self.root, 'b' * 32)
        newer.runtime_base = self.policy.runtime_base
        newer.mountinfo_file = self.policy.mountinfo_file
        newer.boot_file = self.policy.boot_file
        newer.execute('arm')
        before = self.actions[:]
        for action in ('check', 'restore', 'ensure'):
            self.assertEqual(self.policy.execute(action)['action'], 'superseded')
            self.assertEqual(self.value, 0)
        self.assertEqual([x for x in self.actions[len(before):] if x[0] == 'apply'], [])
        with self.assertRaisesRegex(AssertionError, 'previous APST owner'):
            self.policy.execute('arm')
        newer.execute('restore')
        self.assertEqual((self.value, self.table), (1, self.raw))


if __name__ == '__main__':
    unittest.main()
