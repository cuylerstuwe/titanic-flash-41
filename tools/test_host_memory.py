import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import host_memory as policy


class MemoryOwnershipTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.knob = base / "sysctl"
        self.swap = base / "swappiness"
        self.boot = base / "boot"
        self.boot.write_text("first")
        self.marker = base / "owner.json"
        for p in (patch.object(policy, "DIRECTORY", base), patch.object(policy, "BOOT", self.boot),
                  patch.object(policy, "KNOB", self.knob), patch.object(policy, "SWAP_KNOB", self.swap),
                  patch.object(policy, "require_volatile_directory"),
                  patch.object(policy, "set_sysctl", side_effect=lambda name,path,n: path.write_text(str(n)))):
            p.start()
            self.addCleanup(p.stop)

    def test_restore_arbitrary_value_or_existing_zero(self):
        for value in (0, 20, 87):
            with self.subTest(value=value):
                self.knob.write_text(str(value))
                policy.ownership("a", True)
                self.assertEqual(self.knob.read_text(), "0")
                self.assertEqual(json.loads(self.marker.read_bytes())["original"], value)
                policy.ownership("a", False)
                policy.ownership("a", False)
                self.assertEqual(self.knob.read_text(), str(value))
                self.assertFalse(self.marker.exists())

    def test_partial_start_recovers_from_recorded_intent(self):
        self.knob.write_text("42")
        with patch.object(policy, "set_sysctl", side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                policy.ownership("a", True)
        self.assertTrue(self.marker.exists())
        policy.ownership("a", False)
        self.assertEqual(self.knob.read_text(), "42")
        self.assertFalse(self.marker.exists())

    def test_failed_retire_can_retry_exact_prior_value(self):
        self.knob.write_text("20")
        policy.ownership("a", True)
        with patch.object(policy, "set_sysctl", side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                policy.ownership("a", False)
        self.assertTrue(self.marker.exists())
        policy.ownership("a", False)
        self.assertEqual(self.knob.read_text(), "20")

    def test_late_unit_and_competing_arm_do_not_touch_live_owner(self):
        self.knob.write_text("20")
        policy.ownership("a", True)
        with self.assertRaises(RuntimeError):
            policy.ownership("b", True)
        policy.ownership("b", False)
        self.assertEqual(self.knob.read_text(), "0")
        self.assertEqual(json.loads(self.marker.read_bytes())["session"], "a")
        self.knob.write_text("13")
        policy.ownership("a", False)
        self.assertEqual(self.knob.read_text(), "13")
        self.assertFalse(self.marker.exists())

    def test_stale_boot_cannot_restore(self):
        self.knob.write_text("20")
        policy.ownership("a", True)
        self.boot.write_text("second")
        policy.ownership("a", False)
        self.assertEqual(self.knob.read_text(), "0")

    def test_swap_zero_opt_in_preserves_original_and_operator_override(self):
        for initial,override in ((0,None),(1,None),(145,None),(1,12)):
            with self.subTest(initial=initial,override=override):
                self.knob.write_text("20")
                self.swap.write_text(str(initial))
                policy.ownership("a", True, True)
                self.assertEqual((self.knob.read_text(),self.swap.read_text()),("0","0"))
                self.assertEqual(json.loads(self.marker.read_bytes())["swappiness"],initial)
                if override is not None:
                    self.swap.write_text(str(override))
                policy.ownership("a", False)
                self.assertEqual((self.knob.read_text(),self.swap.read_text()),
                                 ("20",str(initial if override is None else override)))
                self.assertFalse(self.marker.exists())

    def test_interrupted_second_write_and_failed_first_restore(self):
        self.knob.write_text("20")
        self.swap.write_text("1")
        def fail_swap(name,path,new):
            if path == self.swap:
                raise OSError("swap write failed")
            path.write_text(str(new))
        with patch.object(policy,"set_sysctl",side_effect=fail_swap), self.assertRaises(OSError):
            policy.ownership("a",True,True)
        self.assertEqual((self.knob.read_text(),self.swap.read_text()),("0","1"))
        policy.ownership("a",False)
        self.assertEqual((self.knob.read_text(),self.swap.read_text()),("20","1"))
        policy.ownership("a",True,True)
        def fail_compaction(name,path,new):
            if path == self.knob:
                raise OSError("compaction restore failed")
            path.write_text(str(new))
        with patch.object(policy,"set_sysctl",side_effect=fail_compaction), self.assertRaises(OSError):
            policy.ownership("a",False)
        self.assertTrue(self.marker.exists())
        self.assertEqual((self.knob.read_text(),self.swap.read_text()),("0","1"))
        policy.ownership("a",False)
        self.assertEqual((self.knob.read_text(),self.swap.read_text()),("20","1"))
        self.assertFalse(self.marker.exists())


class WorkerLifetimeTests(unittest.TestCase):
    def test_container_identity_is_required(self):
        with patch.object(policy, "run", return_value=subprocess.CompletedProcess([], 0, 'id 1 running ["other"]', '')):
            with self.assertRaisesRegex(RuntimeError, "different serving generation"):
                policy.worker("target", "expected")

    def test_pid_reuse_between_observations(self):
        states = [("orig", 7, "running"), ("new", 8, "running"), ("new", 8, "running"), ("new", 8, "running")]
        poller = Mock()
        with patch.object(policy, "worker", side_effect=states), patch.object(policy.os, "pidfd_open", create=True,
                side_effect=[71, 81]), patch.object(policy.os, "close") as close, patch.object(policy.select, "poll",
                return_value=poller), patch.object(policy.time, "sleep"):
            policy.wait_for_worker("container", "config")
        self.assertEqual([c.args[0] for c in close.call_args_list], [71, 81])
        poller.register.assert_called_once_with(81, policy.select.POLLIN)
        poller.poll.assert_called_once_with()

    def test_absent_then_exited_is_a_finished_worker(self):
        with patch.object(policy, "worker", side_effect=[None, ("id", 0, "exited")]), patch.object(policy.time, "sleep"):
            policy.wait_for_worker("c", "config")

    def test_abandoned_launch_expires(self):
        with patch.object(policy.time, "monotonic", side_effect=[0, 601]), patch.object(policy, "worker") as worker:
            with self.assertRaises(TimeoutError):
                policy.wait_for_worker("c", "config")
        worker.assert_not_called()
