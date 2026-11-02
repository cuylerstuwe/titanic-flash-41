#!/usr/bin/env python3
"""Renewable, volatile Spark APST policy with independent exact restoration.

The head renews a short lease while native serving is active. A Spark-local
systemd timer restores the original value/table if the head or monitor stops.
This changes neither persistent NVMe settings nor model/runtime allocations.
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
from host_state import atomic, atomic_bytes, require_volatile_directory

LEASE_SECONDS = 120


def run(args, check=True):
    return subprocess.run(args, capture_output=True, check=check, timeout=30)


def feature():
    text = run(['sudo', '-n', 'nvme', 'get-feature', '/dev/nvme0', '-f', '12', '-H']).stdout.decode()
    value = int(re.search(r'Current value:\s*(?:0x)?([0-9a-fA-F]+)', text).group(1), 16)
    table = run(['sudo', '-n', 'nvme', 'get-feature', '/dev/nvme0', '-f', '12', '-b', '-l', '256']).stdout
    assert len(table) == 256
    return value, table


def apply(table, value):
    run(['sudo', '-n', 'nvme', 'set-feature', '/dev/nvme0', '--feature-id=12',
         '--value=' + str(value), '--data-len=256', '--data=' + str(table)])


class Policy:
    def __init__(self, root, session):
        self.root, self.session = root, session
        self.directory = root / 'apst' / session
        self.control = self.directory / 'control.json'
        self.table = self.directory / 'original-table.bin'
        self.root_digest = hashlib.sha256(str(root).encode()).hexdigest()
        # A last SSH logout may remove user-owned /dev/shm objects under
        # logind RemoveIPC, independently of this system service's lifetime.
        # Keep the renewable lease in private tmpfs tied to the node's boot.
        self.runtime_base = Path('/run') / ('titanic-apst-' + str(os.getuid()))
        self.mountinfo_file = Path('/proc/self/mountinfo')
        self.boot_file = Path('/proc/sys/kernel/random/boot_id')
        self.timer = 'titanic-apst-' + session

    @property
    def owner_file(self):
        return self.runtime_base / 'nvme0-owner.json'

    def owner(self):
        return json.loads(self.owner_file.read_text()) if self.owner_file.exists() else None

    def mine(self, owner):
        return owner == dict(root_sha256=self.root_digest, session=self.session,
                             boot_id=self.boot_id())

    @property
    def runtime(self):
        return self.runtime_base / self.root_digest[:16]

    @property
    def lease_file(self):
        return self.runtime / (self.session + '.json')

    def prepare_runtime(self):
        try:
            self.runtime_base.mkdir(mode=0o700, exist_ok=True)
        except PermissionError:
            assert self.runtime_base.parent == Path('/run') and not self.runtime_base.is_symlink()
            assert not self.runtime_base.exists(), 'refuse to take over an existing runtime directory'
            run(['sudo', '-n', 'install', '-d', '-m', '0700', '-o', str(os.getuid()),
                 '-g', str(os.getgid()), str(self.runtime_base)])
        require_volatile_directory(self.runtime_base, self.mountinfo_file)
        self.runtime.mkdir(mode=0o700, exist_ok=True)
        require_volatile_directory(self.runtime, self.mountinfo_file)

    def boot_id(self):
        return self.boot_file.read_text().strip()

    def lease(self, state):
        try:
            value = json.loads(self.lease_file.read_text())
            now = time.monotonic()
            if (value['session'] == self.session and value['root_sha256'] == self.root_digest
                    and value['table_sha256'] == state['table_sha256']
                    and value['boot_id'] == self.boot_id()
                    and value['renewed_monotonic'] <= now
                    and now < value['expires_monotonic'] <= now + LEASE_SECONDS):
                return value
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def renew_lease(self, state):
        self.prepare_runtime()
        now = time.monotonic()
        lease = dict(session=self.session, root_sha256=self.root_digest,
                     table_sha256=state['table_sha256'], boot_id=self.boot_id(),
                     renewed_monotonic=now, expires_monotonic=now + LEASE_SECONDS,
                     expires_epoch=time.time() + LEASE_SECONDS)
        atomic(self.lease_file, lease, durable=False)
        return lease

    def transition(self, state, action, **fields):
        state.update(fields)
        state['transitions'] = (state.get('transitions', []) + [dict(action=action, epoch=time.time())])[-8:]
        atomic(self.control, state)

    def guard_active(self):
        return run(['systemctl', 'is-active', '--quiet', self.timer + '.timer'], check=False).returncode == 0

    def ensure_guard(self):
        if self.guard_active():
            return
        # A just-expired timer may still be loaded. Restart it if possible;
        # otherwise create the transient timer again. No feature change may
        # precede a positively active independent guard.
        started = run(['sudo', '-n', 'systemctl', 'start', self.timer + '.timer'], check=False)
        if started.returncode:
            run(['sudo', '-n', 'systemd-run', '--unit=' + self.timer, '--on-active=30s',
                 '--on-unit-active=30s', '--timer-property=AccuracySec=1s',
                 '--property=User=' + str(os.getuid()), '--property=TimeoutStartSec=45',
                 '--property=StandardOutput=null',
                 '/usr/bin/python3', str(Path(__file__).resolve()), str(self.root), self.session, 'check'])
        assert self.guard_active(), 'independent APST guard unavailable'

    def restore(self, state, raw, reason, stop_guard=True):
        owner = self.owner()
        if owner is not None and not self.mine(owner):
            # A delayed check from an old deployment must not undo the current
            # owner's policy, even when it used a different deployment root.
            self.lease_file.unlink(missing_ok=True)
            if stop_guard:
                stopped = run(['sudo', '-n', 'systemctl', 'stop', self.timer + '.timer'], check=False)
                assert stopped.returncode in [0, 5]
            return dict(action='superseded', original_table_exact=False)
        actual, actual_table = feature()
        changed = actual != state['original_value'] or actual_table != raw
        if changed:
            apply(self.table, state['original_value'])
            actual, actual_table = feature()
        assert actual == state['original_value'] and actual_table == raw, 'APST restoration mismatch'
        if changed or not state['restored']:
            self.transition(state, reason, restored=True, restored_epoch=time.time())
        self.lease_file.unlink(missing_ok=True)
        if self.mine(owner):
            self.owner_file.unlink()
        if stop_guard:
            stopped = run(['sudo', '-n', 'systemctl', 'stop', self.timer + '.timer'], check=False)
            assert stopped.returncode in [0, 5]
        return dict(action='restored', value=actual, original_table_exact=True)

    def activate(self, state, raw, action):
        self.prepare_runtime()
        assert self.owner() is None or self.mine(self.owner()), 'another generation owns NVMe APST'
        atomic(self.owner_file, dict(root_sha256=self.root_digest, session=self.session,
                                    boot_id=self.boot_id()), durable=False)
        self.ensure_guard()
        lease = self.renew_lease(state)
        # Durable intent precedes the feature change. Even a killed helper
        # leaves the independent timer enough information to restore exactly.
        # Keep a conservative deadline readable by the previous helper. Normal
        # v2 renewals never update it: an interrupted rollback can still restore
        # this baseline rather than failing to parse the newer state file.
        self.transition(state, action, restored=False, activated_epoch=time.time(),
                        expires_epoch=lease['expires_epoch'])
        apply(self.table, 0)
        actual, actual_table = feature()
        assert actual == 0 and actual_table == raw, 'APST activation mismatch'
        return dict(action=action, value=actual, expires_epoch=lease['expires_epoch'])

    def execute(self, action):
        self.prepare_runtime()
        # Serialize all deployment roots and generations targeting /dev/nvme0.
        with (self.runtime_base / 'nvme0-control.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if action == 'restore' and not self.control.exists():
                assert not self.mine(self.owner()), 'APST owner lost its restoration intent'
                return dict(action='not-armed', original_table_exact=False)
            if action == 'arm':
                assert self.owner() is None, 'restore the previous APST owner before arming'
            elif self.owner() is not None and not self.mine(self.owner()):
                return self.restore({}, b'', 'superseded')
            return self.execute_owned(action)

    def execute_owned(self, action):
        if action == 'arm':
            self.directory.parent.mkdir(mode=0o700, exist_ok=True)
            self.directory.mkdir(mode=0o700, exist_ok=False)
            # Persist the newly created directory links too; fsync of a file
            # and its immediate directory alone does not publish all parents.
            for parent in [self.directory.parent, self.root]:
                descriptor = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        with (self.directory / 'control.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if action == 'arm':
                value, raw = feature()
                assert value in [0, 1]
                # Baseline bytes and containing directory are durable before
                # creating the guard or applying the temporary policy.
                atomic_bytes(self.table, raw)
                state = dict(schema=2, session=self.session, root_sha256=self.root_digest,
                             original_value=value, table_sha256=hashlib.sha256(raw).hexdigest(),
                             captured_epoch=time.time(), expires_epoch=time.time(), restored=True)
                atomic(self.control, state)
                return self.activate(state, raw, 'armed')

            state = json.loads(self.control.read_text())
            raw = self.table.read_bytes()
            assert len(raw) == 256 and state['original_value'] in [0, 1], 'invalid APST baseline'
            assert hashlib.sha256(raw).hexdigest() == state['table_sha256'], 'APST baseline table changed'
            if state.get('schema') != 2:
                assert state.get('schema', 1) == 1, 'unknown APST state schema'
                # A coordinated upgrade restores the old lease first. Retain
                # its check/restore path for interrupted upgrades; never adopt
                # the current disabled setting as a replacement baseline.
                assert action in ['check', 'restore'], 'legacy APST session must be restored before upgrade'
                if action == 'restore' or state['restored'] or time.time() >= state['expires_epoch']:
                    return self.restore(state, raw, 'legacy-restored')
                return dict(action='waiting', expires_epoch=state['expires_epoch'])
            assert state['session'] == self.session and state['root_sha256'] == self.root_digest, 'APST session identity changed'
            lease = self.lease(state)
            if action == 'restore' or (action == 'check' and (state['restored'] or lease is None)):
                return self.restore(state, raw, 'restored')
            if action == 'check':
                return dict(action='waiting', expires_epoch=lease['expires_epoch'])
            assert action in ['renew', 'ensure']
            if action == 'renew':
                assert not state['restored'] and lease is not None, 'APST lease expired; use baseline-preserving ensure'
            if not state['restored'] and lease is not None and self.guard_active():
                actual, actual_table = feature()
                assert actual_table == raw and actual in [0, state['original_value']], 'active APST policy drift'
                if actual == 0:
                    lease = self.renew_lease(state)
                    return dict(action='renewed', value=0, expires_epoch=lease['expires_epoch'])
            # Missing/expired RAM, an interrupted activation or a lost guard
            # restores the original baseline before any re-arm is attempted.
            self.restore(state, raw, 'restored-before-rearm', stop_guard=False)
            if action == 'renew':
                raise RuntimeError('APST policy restored; ensure required to re-arm')
            return self.activate(state, raw, 'rearmed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('session')
    parser.add_argument('action', choices=['arm', 'renew', 'ensure', 'check', 'restore'])
    args = parser.parse_args()
    assert re.fullmatch(r'[0-9a-f]{32}', args.session)
    root = args.root.resolve()
    assert root.is_absolute() and len(root.parts) >= 3 and root.is_dir()
    assert root.stat().st_uid == os.getuid(), 'policy root must belong to the deployment account'
    result = Policy(root, args.session).execute(args.action)
    result['epoch'] = time.time()
    # Periodic successful guard checks need no journal payload either. Failures
    # still exit nonzero on stderr, while callers receive all requested actions.
    if args.action != 'check' or result['action'] != 'waiting':
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
