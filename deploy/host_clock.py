#!/usr/bin/env python3
"""Temporary GB10 graphics-clock caps with node-local expiry and stock reset.

The fleet lifecycle owns an otherwise idle GPU before arming this policy. Its
baseline is explicitly stock clocks, not an arbitrary administrator's lock.
Only startup/failure/stop transitions are durable; renewals live in tmpfs.
This helper is opt-in and is not installed by adding it to the source tree.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from host_state import atomic, require_volatile_directory

LEASE_SECONDS = 120
CAPS = (2000, 2350)


def run(args, check=True):
    return subprocess.run(args, capture_output=True, text=True, check=check, timeout=30)


class Policy:
    def __init__(self, root, session):
        self.root, self.session = root, session
        self.identity = dict(root_sha256=hashlib.sha256(str(root).encode()).hexdigest(),
                             session=session)
        self.control = root / 'gpu-clock' / (session + '.json')
        # logind may remove a non-lingering user's /dev/shm objects when the
        # last SSH session closes, even while this system service is alive.
        # A dedicated /run directory keeps ownership tied to the serving
        # generation and reboot, not to an administrator's login session.
        self.runtime = Path('/run') / ('titanic-clock-' + str(os.getuid()))
        self.mountinfo_file = Path('/proc/self/mountinfo')
        self.boot_file = Path('/proc/sys/kernel/random/boot_id')
        self.timer = 'titanic-clock-' + session

    @property
    def owner_file(self):
        return self.runtime / 'owner.json'

    @property
    def lease_file(self):
        return self.runtime / (self.identity['root_sha256'][:16] + '-' + self.session + '.json')

    def boot_id(self):
        return self.boot_file.read_text().strip()

    def prepare_runtime(self):
        try:
            self.runtime.mkdir(mode=0o700, exist_ok=True)
        except PermissionError:
            # /run is root-owned. Only initial directory provisioning needs
            # privilege; leases, locks and renewals stay unprivileged in RAM.
            assert self.runtime.parent == Path('/run') and not self.runtime.is_symlink()
            assert not self.runtime.exists(), 'refuse to take over an existing runtime directory'
            run(['sudo', '-n', 'install', '-d', '-m', '0700', '-o', str(os.getuid()),
                 '-g', str(os.getgid()), str(self.runtime)])
        require_volatile_directory(self.runtime, self.mountinfo_file)

    def owner(self):
        return json.loads(self.owner_file.read_text()) if self.owner_file.exists() else None

    def mine(self, owner):
        return owner == dict(self.identity, boot_id=self.boot_id())

    def state(self):
        state = json.loads(self.control.read_text())
        assert state['schema'] == 1 and state['cap_mhz'] in CAPS
        assert all(state[k] == v for k, v in self.identity.items()), 'clock session identity changed'
        return state

    def save(self, state, action, **fields):
        state.update(fields, action=action, epoch=time.time())
        atomic(self.control, state)

    def lease(self, state):
        try:
            value = json.loads(self.lease_file.read_text())
            now = time.monotonic()
            if (all(value[k] == v for k, v in self.identity.items())
                    and value['boot_id'] == self.boot_id() == state['boot_id']
                    and value['cap_mhz'] == state['cap_mhz']
                    and value['renewed_monotonic'] <= now
                    and now < value['expires_monotonic'] <= now + LEASE_SECONDS):
                return value
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def renew(self, state):
        now = time.monotonic()
        lease = dict(self.identity, boot_id=self.boot_id(), cap_mhz=state['cap_mhz'],
                     renewed_monotonic=now, expires_monotonic=now + LEASE_SECONDS)
        atomic(self.lease_file, lease, durable=False)
        return lease

    def guard_active(self):
        return run(['systemctl', 'is-active', '--quiet', self.timer + '.timer'], check=False).returncode == 0

    def ensure_guard(self):
        if self.guard_active():
            return
        started = run(['sudo', '-n', 'systemctl', 'start', self.timer + '.timer'], check=False)
        if started.returncode:
            run(['sudo', '-n', 'systemd-run', '--unit=' + self.timer, '--on-active=30s',
                 '--on-unit-active=30s', '--timer-property=AccuracySec=1s',
                 '--property=User=' + str(os.getuid()), '--property=TimeoutStartSec=45',
                 '--property=StandardOutput=null', '/usr/bin/python3',
                 str(Path(__file__).resolve()), str(self.root), self.session, 'check'])
        assert self.guard_active(), 'independent clock reset guard unavailable'

    def stop_guard(self):
        stopped = run(['sudo', '-n', 'systemctl', 'stop', self.timer + '.timer'], check=False)
        assert stopped.returncode in (0, 5)

    def restore(self, state, reason, stop_guard=True):
        owner = self.owner()
        # A delayed old check must never reset a newer generation's cap.
        if owner is not None and not self.mine(owner):
            self.lease_file.unlink(missing_ok=True)
            if stop_guard:
                self.stop_guard()
            return dict(action='superseded', stock_reset=False)
        if not state['restored']:
            run(['sudo', '-n', 'nvidia-smi', '-rgc'])
            self.save(state, reason, restored=True)
        self.lease_file.unlink(missing_ok=True)
        if self.mine(owner):
            self.owner_file.unlink()
        if stop_guard:
            self.stop_guard()
        return dict(action='restored', stock_reset=True)

    def activate(self, state, action):
        assert self.owner() is None or self.mine(self.owner()), 'another clock generation owns the GPU'
        atomic(self.owner_file, dict(self.identity, boot_id=self.boot_id()), durable=False)
        self.ensure_guard()
        lease = self.renew(state)
        # Publish recoverable intent before touching clocks. The independent
        # timer can reset stock even if the caller dies during nvidia-smi.
        self.save(state, action, restored=False)
        run(['sudo', '-n', 'nvidia-smi', '-lgc', '300,' + str(state['cap_mhz'])])
        return dict(action=action, cap_mhz=state['cap_mhz'],
                    expires_monotonic=lease['expires_monotonic'])

    def waiting(self, state):
        return dict(action='waiting', cap_mhz=state['cap_mhz'])

    def execute(self, action, cap_mhz=None):
        self.prepare_runtime()
        # One lock covers every generation and deployment root on this node.
        with (self.runtime / 'control.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if action == 'restore' and not self.control.exists():
                # A four-node arm can fail before it reaches this node. No
                # intent file means this helper never issued a cap here.
                assert not self.mine(self.owner()), 'clock owner lost its restoration intent'
                return dict(action='not-armed', stock_reset=False)
            if action == 'arm':
                assert cap_mhz in CAPS and not self.control.exists()
                assert self.owner() is None, 'restore the previous clock owner before arming'
                idle = run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
                assert not idle.stdout.strip(), 'clock admission requires an idle GPU'
                self.control.parent.mkdir(mode=0o700, exist_ok=True)
                state = dict(self.identity, schema=1, boot_id=self.boot_id(),
                             cap_mhz=cap_mhz, restored=True, baseline='stock')
                self.save(state, 'prepared')
                return self.activate(state, 'armed')
            state = self.state()
            if action == 'restore':
                return self.restore(state, 'restored')
            assert action in ('check', 'ensure')
            owner = self.owner()
            if owner is not None and not self.mine(owner):
                return self.restore(state, 'superseded')
            lease = self.lease(state)
            if action == 'check':
                if state['restored'] or lease is None or not self.mine(owner):
                    return self.restore(state, 'expired')
                return self.waiting(state)
            assert state['boot_id'] == self.boot_id(), 'new boot requires a new serving generation'
            if not state['restored'] and lease is not None and self.mine(owner) and self.guard_active():
                self.renew(state)
                return dict(action='renewed', cap_mhz=state['cap_mhz'])
            self.restore(state, 'reset-before-rearm', stop_guard=False)
            return self.activate(state, 'rearmed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('session')
    parser.add_argument('action', choices=['arm', 'ensure', 'check', 'restore'])
    parser.add_argument('--cap-mhz', type=int, choices=CAPS)
    args = parser.parse_args()
    assert re.fullmatch(r'[0-9a-f]{32}', args.session)
    root = args.root.resolve()
    assert root.is_absolute() and len(root.parts) >= 3 and root.is_dir()
    assert root.stat().st_uid == os.getuid(), 'policy root must belong to the deployment account'
    assert (args.action == 'arm') == (args.cap_mhz is not None)
    result = Policy(root, args.session).execute(args.action, args.cap_mhz)
    if args.action != 'check' or result['action'] != 'waiting':
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
