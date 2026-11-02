"""Clock ownership/expiry failure cases; no GPU or systemd changes in these tests."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import host_clock as clock


class ClockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.now = 100.
        self.guards = set()
        self.allow_guard = True
        self.idle = True
        self.cap = None
        self.changes = []
        self.policy = self.make_policy('a' * 32)
        self.patches = [patch.object(clock, 'run', side_effect=self.mock_run),
                        patch.object(clock.time, 'monotonic', side_effect=lambda: self.now)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def make_policy(self, session):
        p = clock.Policy(self.root, session)
        p.runtime = self.root / 'ram'
        p.mountinfo_file = self.root / 'mounts'
        p.mountinfo_file.write_text(f'1 0 0:1 / {self.root} rw - tmpfs tmpfs rw\n')
        p.boot_file = self.root / 'boot'
        if not p.boot_file.exists():
            p.boot_file.write_text('boot1')
        return p

    def mock_run(self, args, check=True):
        code, stdout = 0, ''
        if args[:3] == ['systemctl', 'is-active', '--quiet']:
            code = 0 if args[-1] in self.guards else 3
        elif args[:4] == ['sudo', '-n', 'systemctl', 'start']:
            code = 5
        elif args[:3] == ['sudo', '-n', 'systemd-run']:
            if not self.allow_guard:
                raise subprocess.CalledProcessError(1, args)
            unit = next(x.split('=', 1)[1] for x in args if x.startswith('--unit='))
            self.guards.add(unit + '.timer')
        elif args[:4] == ['sudo', '-n', 'systemctl', 'stop']:
            self.guards.discard(args[-1])
        elif args[:2] == ['nvidia-smi', '--query-compute-apps=pid']:
            stdout = '' if self.idle else '12345\n'
        elif args[:3] == ['sudo', '-n', 'nvidia-smi']:
            if args[3] == '-lgc':
                self.assertTrue(self.guards, 'cap applied without independent reset')
                self.cap = int(args[4].split(',')[1])
            else:
                self.assertEqual(args[3], '-rgc')
                self.cap = None
            self.changes.append(self.cap)
        else:
            raise AssertionError(args)
        if check and code:
            raise subprocess.CalledProcessError(code, args)
        return subprocess.CompletedProcess(args, code, stdout, '')

    def arm(self, cap=2350):
        self.assertEqual(self.policy.execute('arm', cap)['action'], 'armed')
        self.assertEqual(self.cap, cap)

    def test_renewals_leave_disk_and_clocks_untouched(self):
        self.arm()
        original = self.policy.control.read_bytes()
        with patch.object(clock.os, 'fsync') as fsync:
            for _ in range(10):
                self.now += 30
                self.assertEqual(self.policy.execute('ensure')['action'], 'renewed')
                self.assertEqual(self.policy.execute('check')['action'], 'waiting')
            self.assertEqual(fsync.call_count, 0)
        self.assertEqual(self.policy.control.read_bytes(), original)
        self.assertEqual(self.changes, [2350])

    def test_expiry_and_missing_lease_restore_stock(self):
        self.arm(2000)
        self.now += clock.LEASE_SECONDS
        self.assertTrue(self.policy.execute('check')['stock_reset'])
        self.assertIsNone(self.cap)
        self.assertFalse(self.guards)
        self.policy.execute('ensure')
        self.policy.lease_file.unlink()
        self.policy.execute('check')
        self.assertIsNone(self.cap)

    def test_new_generation_is_immune_to_stale_restore_and_ensure(self):
        self.arm()
        self.policy.execute('restore')
        newer = self.make_policy('b' * 32)
        newer.execute('arm', 2000)
        before = self.changes[:]
        for action in ('check', 'restore', 'ensure'):
            self.assertEqual(self.policy.execute(action)['action'], 'superseded')
            self.assertEqual(self.cap, 2000)
            self.assertTrue(newer.guard_active())
        self.assertEqual(self.changes, before)

    def test_other_generation_cannot_arm_over_current_owner(self):
        self.arm()
        with self.assertRaisesRegex(AssertionError, 'previous clock owner'):
            self.make_policy('b' * 32).execute('arm', 2000)
        self.assertEqual(self.changes, [2350])

    def test_missing_guard_resets_before_rearm_and_failure_stays_stock(self):
        self.arm()
        self.guards.clear()
        self.assertEqual(self.policy.execute('ensure')['action'], 'rearmed')
        self.assertEqual(self.changes, [2350, None, 2350])
        self.guards.clear()
        self.allow_guard = False
        with self.assertRaises(subprocess.CalledProcessError):
            self.policy.execute('ensure')
        self.assertIsNone(self.cap)
        self.policy.execute('restore')
        self.assertIsNone(self.policy.owner())

    def test_failed_initial_guard_does_not_apply_cap(self):
        self.allow_guard = False
        with self.assertRaises(subprocess.CalledProcessError):
            self.policy.execute('arm', 2350)
        self.assertEqual(self.changes, [])
        self.policy.execute('restore')
        self.assertIsNone(self.policy.owner())

    def test_interrupted_clock_command_leaves_reset_intent(self):
        underlying = self.mock_run
        def interrupted(args, check=True):
            result = underlying(args, check)
            if args[:4] == ['sudo', '-n', 'nvidia-smi', '-lgc']:
                raise OSError('caller disappeared after driver accepted cap')
            return result
        with patch.object(clock, 'run', side_effect=interrupted):
            with self.assertRaises(OSError):
                self.policy.execute('arm', 2350)
        self.assertEqual(self.cap, 2350)
        self.now += clock.LEASE_SECONDS
        self.policy.execute('check')
        self.assertIsNone(self.cap)

    def test_idle_and_tmpfs_admission_precede_any_change(self):
        self.idle = False
        with self.assertRaisesRegex(AssertionError, 'idle GPU'):
            self.policy.execute('arm', 2350)
        self.assertEqual(self.changes, [])
        self.idle = True
        self.policy.mountinfo_file.write_text(f'1 0 0:1 / {self.root} rw - ext4 disk rw\n')
        with self.assertRaisesRegex(AssertionError, 'tmpfs'):
            self.policy.execute('arm', 2350)
        self.assertEqual(self.changes, [])

    def test_boot_invalidates_lease_and_requires_new_generation(self):
        self.arm()
        self.policy.boot_file.write_text('boot2')
        # Real reboot loses tmpfs as well as the device lock.
        self.policy.owner_file.unlink()
        self.policy.execute('check')
        self.assertIsNone(self.cap)
        with self.assertRaisesRegex(AssertionError, 'new boot'):
            self.policy.execute('ensure')

    def test_partial_group_arm_can_restore_an_unreached_rank(self):
        self.assertEqual(self.policy.execute('restore')['action'], 'not-armed')
        self.assertEqual(self.changes, [])


if __name__ == '__main__':
    unittest.main()
