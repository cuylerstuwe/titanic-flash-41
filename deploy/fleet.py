#!/usr/bin/env python3
"""Explicit four-rank lifecycle, run on the head. Never stops unrelated services.

The resident Rust server owns scheduling, cache, cancellation and priorities.
This controller admits one cohort and fails the whole cohort if a rank dies.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import ipaddress
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from host_state import atomic, atomic_bytes

PROJECT = 'titanic-flash-41'
SOURCE = Path(__file__).resolve().parents[1]

def vision_encoder_rank(profile):
    value = profile['CANADA_SCRATCH_VISION_ENCODER_RANK']
    assert value in ('0', '1', '2', '3'), 'vision encoder rank must be an explicit TP4 rank'
    return int(value)

def rank_data_paths(rank, vision_rank):
    """Relative immutable shard paths; atlas rank is distinct from its host."""
    assert type(rank) is int and 0 <= rank < 4
    assert type(vision_rank) is int and 0 <= vision_rank < 4
    paths = [f'staged/layer{layer}-v1/r{rank}/' for layer in range(40)]
    paths += [f'staged/model-io-v1/r{rank}/', f'staged/dspark-mtp-v1/r{rank}/']
    paths += [f'engram/engram-l{layer}-r{rank}of4.{ext}'
              for layer in (1, 14) for ext in ('bin', 'json')]
    if rank == vision_rank:
        # Vision is replicated, not TP-sharded: preserve the canonical r0 atlas
        # bytes/metadata even when another device owns image encoding.
        paths.append('staged/vision-v1/r0/')
    return paths

def validate(cfg):
    assert len(cfg['nodes']) == 4 and cfg['nodes'][0]['ssh'] is None
    root = Path(cfg['root'])
    assert root.is_absolute() and len(root.parts) >= 3 and '..' not in root.parts
    assert re.fullmatch(r'/[A-Za-z0-9_./-]+', str(root)), 'use a root without whitespace or shell metacharacters'
    assert re.fullmatch(r'[A-Za-z0-9_./:@-]+', cfg['image'])
    assert type(cfg.get('gpu_clock_cap_mhz')) is int and cfg['gpu_clock_cap_mhz'] in (0, 2350), \
        'choose the qualified 2350 MHz cap or explicit stock-clock override 0'
    assert 1024 <= cfg['rendezvous_port'] <= 65530
    host, port = cfg['http_bind'].rsplit(':', 1)
    ip = ipaddress.ip_address(host)
    assert ip.version == 4 and 1024 <= int(port) <= 65535
    assert int(port) not in {cfg['rendezvous_port']+i for i in range(5)}
    assert ip.is_loopback or (cfg['allow_remote'] and ip.is_private and not ip.is_unspecified)
    addresses = []
    for rank, node in enumerate(cfg['nodes']):
        assert type(node.get('disable_nvme0_apst')) is bool, 'select the NVMe policy explicitly for every node'
        assert re.fullmatch(r'[A-Za-z0-9_.-]+', node['hostname'])
        if rank:
            assert isinstance(node['ssh'], str) and re.fullmatch(r'[A-Za-z0-9_@.-]+', node['ssh'])
            assert not node['ssh'].startswith('-')
        assert len(node['hcas']) == len(node['fabric_ipv4']) == 2
        assert len(set(node['hcas'])) == 2
        assert all(re.fullmatch(r'[A-Za-z0-9_-]+', x) for x in node['hcas'])
        assert re.fullmatch(r'[A-Za-z0-9_.-]+', node['socket_interface'])
        assert 0 <= node['gid_index'] <= 255
        for value in node['fabric_ipv4']:
            address = ipaddress.IPv4Address(value)
            assert not (address.is_loopback or address.is_unspecified or address.is_multicast)
            addresses.append(value)
    assert len(set(addresses)) == 8 and len({n['hostname'] for n in cfg['nodes']}) == 4
    return cfg

class Fleet:
    def __init__(self, inventory):
        self.cfg = validate(json.loads(inventory.read_text()))
        self.profile = json.loads((SOURCE/'deploy/profile.json').read_text())
        self.vision_rank = vision_encoder_rank(self.profile)
        self.root = Path(self.cfg['root'])
        self.state_dir = self.root/'state'
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_dir.chmod(0o700)
        self.state_file = self.state_dir/'cohort.json'
        self.stop_file = self.state_dir/'stop-requested'
        self.digest = hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest()
        self.next_policy_attempt = 0
        self.policy_retry_delay = 5

    def prepare_host_policy(self):
        cap = self.cfg['gpu_clock_cap_mhz']
        apst = [rank for rank, node in enumerate(self.cfg['nodes']) if node['disable_nvme0_apst']]
        if not cap and not apst:
            return None
        # Timer commands point to immutable code, so updating the checkout
        # cannot change how an already armed generation restores its settings.
        files = {name:(SOURCE/'deploy'/name).read_bytes()
                 for name in ['host_state.py', 'host_clock.py', 'host_apst.py']}
        fingerprint = hashlib.sha256(b''.join(name.encode()+b'\0'+raw
                                    for name, raw in sorted(files.items()))).hexdigest()
        directory = self.state_dir/'host-policy-code'/fingerprint
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name, raw in files.items():
            path = directory/name
            if path.exists():
                assert path.read_bytes() == raw, 'immutable host-policy source changed'
            else:
                atomic_bytes(path, raw)
        for rank, node in enumerate(self.cfg['nodes']):
            if rank:
                self.run(['mkdir', '-p', directory], rank)
                self.run(['chmod', '700', directory], rank)
                subprocess.run(['rsync', '-rt', '--ignore-existing', '--chmod=F600,D700',
                    str(directory)+'/', node['ssh']+':'+str(directory)+'/'], check=True, timeout=60)
            observed = self.run(['sha256sum', *[directory/name for name in files]], rank).stdout.splitlines()
            assert [line.split()[0] for line in observed] == [hashlib.sha256(raw).hexdigest() for raw in files.values()], \
                'host-policy source differs on a node'
        return dict(code=str(directory), source_sha256=fingerprint, cap_mhz=cap, apst_ranks=apst)

    def host_policies(self, action, state):
        policy = state.get('host_policy')
        if not policy:
            return []
        results, errors = [], []
        directory = Path(policy['code'])
        assert directory == self.state_dir/'host-policy-code'/policy['source_sha256']
        for rank in range(4):
            kinds = (['clock'] if policy['cap_mhz'] else []) + (['apst'] if rank in policy['apst_ranks'] else [])
            for kind in kinds:
                cmd = ['python3', directory/f'host_{kind}.py', self.state_dir, state['generation'], action]
                if kind == 'clock' and action == 'arm':
                    cmd += ['--cap-mhz', str(policy['cap_mhz'])]
                try:
                    result = json.loads(self.run(cmd, rank, timeout=90).stdout)
                    allowed = ('armed',) if action == 'arm' else ('renewed', 'rearmed') if action == 'ensure' \
                        else ('restored', 'not-armed', 'superseded')
                    assert result['action'] in allowed, 'host policy did not retain this generation'
                    results.append(dict(rank=rank, kind=kind, result=result))
                except Exception as e:
                    errors.append(f'rank {rank} {kind}: {type(e).__name__}: {e}')
        if errors:
            raise RuntimeError('host policy '+action+' incomplete: '+str(errors))
        return results

    def maintain_host_policy(self, state):
        if not state.get('host_policy') or time.monotonic() < self.next_policy_attempt:
            return
        try:
            self.host_policies('ensure', state)
        except Exception as e:
            status, error = 'degraded', str(e)
            self.next_policy_attempt = time.monotonic() + self.policy_retry_delay
            self.policy_retry_delay = min(30, self.policy_retry_delay * 2)
        else:
            status, error = 'active', None
            self.next_policy_attempt = time.monotonic() + 30
            self.policy_retry_delay = 5
        # Node-local timers restore stock settings after a lost controller.
        # A transient management outage need not retire a healthy model. Write
        # only status transitions; ordinary renewal remains entirely in tmpfs.
        if state.get('host_policy_status') != status or state.get('host_policy_error') != error:
            state.update(host_policy_status=status, host_policy_error=error)
            atomic(self.state_file, state)

    def run(self, args, rank=0, check=True, timeout=40):
        args = list(map(str, args))
        host = self.cfg['nodes'][rank]['ssh']
        if host:
            args = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, shlex.join(args)]
        return subprocess.run(args, check=check, capture_output=True, text=True, timeout=timeout)

    def state(self):
        if not self.state_file.exists():
            return None
        s = json.loads(self.state_file.read_text())
        assert s['inventory_sha256'] == self.digest, 'inventory changed while a cohort exists; restore the saved inventory to stop it'
        return s

    def inspect(self, rank, name):
        r = self.run(['docker', 'inspect', name], rank, check=False)
        if r.returncode:
            names = self.run(['docker', 'ps', '-a', '--format', '{{.Names}}'], rank).stdout.splitlines()
            assert name not in names, 'container inspection failed'
            return None
        return json.loads(r.stdout)[0]

    def health(self, state):
        host, port = self.cfg['http_bind'].rsplit(':', 1)
        key = (self.root/'state/keys/http.key').read_text().strip()
        request = urllib.request.Request(f'http://{host}:{port}/health', headers={'Authorization': 'Bearer '+key})
        try:
            with urllib.request.urlopen(request, timeout=5) as r:
                return json.load(r)
        except (OSError, ValueError):
            return None

    def stop(self, state):
        if not state or state.get('stopped'):
            return
        errors = []
        def stop_rank(rank, name):
            try:
                c = self.inspect(rank, name)
                if c is None:
                    return None
                labels = c['Config']['Labels']
                assert labels.get('project') == PROJECT and labels.get('generation') == state['generation']
                assert c['Config']['Entrypoint'] == ['/data/runtime/titanic-flash']
                # Stop peers together: a blocked collective must not keep a
                # later rank alive while an earlier GPU context is torn down.
                # CUDA teardown can outlast Docker's signal grace period.
                self.run(['docker', 'stop', '--time', '15', c['Id']], rank, timeout=90)
                self.run(['docker', 'rm', c['Id']], rank)
            except Exception as e:
                return f'rank {rank}: {e}'
            return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            pending = [pool.submit(stop_rank, rank, name)
                       for rank, name in enumerate(state['containers'])]
            errors.extend(error for future in pending if (error := future.result()))
        try:
            self.host_policies('restore', state)
        except Exception as e:
            errors.append(str(e))
        if errors:
            raise RuntimeError('coordinated stop incomplete: '+str(errors))
        state.update(stopped=True, stopped_epoch=time.time())
        atomic(self.state_file, state)

    def preflight(self):
        for rank, node in enumerate(self.cfg['nodes']):
            assert self.run(['hostname'], rank).stdout.strip() == node['hostname'], 'SSH reached a different host'
            assert self.run(['uname', '-m'], rank).stdout.strip() == 'aarch64'
            assert not self.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], rank).stdout.strip(), 'GPU busy; stop its owner explicitly'
            mem = self.run(['cat', '/proc/meminfo'], rank).stdout
            available = int(next(x.split()[1] for x in mem.splitlines() if x.startswith('MemAvailable:')))*1024
            assert available >= 110*2**30, 'less than 110 GiB available before model load'
            assert int(self.run(['df', '-B1', '--output=avail', self.root], rank).stdout.split()[-1]) >= 35*2**30
            image = json.loads(self.run(['docker', 'image', 'inspect', self.cfg['image']], rank).stdout)[0]
            if rank == 0:
                image_id = image['Id']
            assert image['Id'] == image_id, 'different image on a peer'
            for name in ['titanic-flash', 'core.cubin', 'vision.cubin', 'provenance.json']:
                local = hashlib.sha256((self.root/'data/runtime'/name).read_bytes()).hexdigest()
                assert self.run(['sha256sum', self.root/'data/runtime'/name], rank).stdout.split()[0] == local
            paths = []
            for name in rank_data_paths(rank, self.vision_rank):
                base = self.root/'data'/name
                paths.extend([base/'atlas.json', base/'atlas.bin'] if name.endswith('/') else [base])
            self.run(['python3', '-c', 'from pathlib import Path; import sys; '
                      'missing=[p for p in sys.argv[1:] if not Path(p).is_file()]; '
                      'assert not missing, missing', *paths], rank)
        provenance = json.loads((self.root/'data/runtime/provenance.json').read_text())
        assert provenance['serving'] and not provenance['storage_fault_injection']
        return image_id

    def config(self, rank):
        return dict(config='/data/model/config.json', checkpoint='/data/model/checkpoint.json', rank=rank,
                    core='/data/runtime/core.cubin', kernels='/data/kernels',
                    stages=[f'/data/staged/layer{layer}-v1/r{rank}' for layer in range(40)],
                    io_stage=f'/data/staged/model-io-v1/r{rank}', backing='/data/engram',
                    hash_metadata='/data/model/engram-metadata.json',
                    dspark_staged=f'/data/staged/dspark-mtp-v1/r{rank}',
                    endpoint=f"{self.cfg['nodes'][0]['fabric_ipv4'][0]}:{self.cfg['rendezvous_port']}",
                    ticket='/state/rendezvous.ticket', context=1048576,
                    wide_widths=[512, 1024, 2048, 4096, 8192], events=f'/state/events-r{rank}.jsonl', requests=None,
                    vision=dict(checkpoint='/data/model/checkpoint.json', stage='/data/staged/vision-v1/r0',
                                kernel='/data/runtime/vision.cubin', attention='/data/kernels/vision-attention-v1',
                                weights_sha256='36f12e6a6f4e5ec63bbf0a8e27d22cadbd582856e4d1ac77414647def761b259',
                                cache_bytes=268435456),
                    http=dict(bind=self.cfg['http_bind'], allow_remote=self.cfg['allow_remote'],
                              tokenizer='/data/model/tokenizer.json', bearer_file='/state/http.key',
                              priority_file='/state/priority.key', output_stall_seconds=60, context_cache_mib=64) if rank == 0 else None)

    def launch(self):
        self.stop(self.state())
        image = self.preflight()
        host_policy = self.prepare_host_policy()
        generation = secrets.token_hex(16)
        run_dir = self.state_dir/generation
        run_dir.mkdir(mode=0o700)
        keys = self.state_dir/'keys'
        keys.mkdir(exist_ok=True, mode=0o700)
        for name in ['http.key', 'priority.key']:
            key = keys/name
            if not key.exists():
                with key.open('x') as f:
                    f.write(secrets.token_urlsafe(48)+'\n')
                key.chmod(0o600)
            assert key.stat().st_mode & 0o077 == 0, 'key permissions must be private'
            (run_dir/name).write_bytes(key.read_bytes()); (run_dir/name).chmod(0o600)
        (run_dir/'rendezvous.ticket').write_bytes(secrets.token_bytes(32))
        (run_dir/'rendezvous.ticket').chmod(0o600)
        containers = [f'titanic-{generation}-r{r}' for r in range(4)]
        state = dict(generation=generation, inventory_sha256=self.digest, image=image,
                     containers=containers, started_epoch=time.time(), stopped=False,
                     host_policy=host_policy)
        atomic(self.state_file, state)
        profile = self.profile
        try:
            self.host_policies('arm', state)
            for rank, node in enumerate(self.cfg['nodes']):
                config = run_dir/f'rank{rank}.json'; atomic(config, self.config(rank))
                if rank:
                    self.run(['mkdir', '-p', run_dir], rank)
                    self.run(['chmod', '700', run_dir], rank)
                    subprocess.run(['rsync', '-rt', '--chmod=F600,D700', str(run_dir)+'/',
                                    node['ssh']+':'+str(run_dir)+'/'], check=True, timeout=60)
                env = dict(profile, TITANIC_ROCE_HCAS=','.join(node['hcas']),
                           TITANIC_ROCE_IPV4=','.join(node['fabric_ipv4']),
                           TITANIC_ROCE_GID_INDEX=str(node['gid_index']), NCCL_IB_GID_INDEX=str(node['gid_index']),
                           NCCL_IB_HCA=','.join(node['hcas']), NCCL_SOCKET_IFNAME=node['socket_interface'])
                # Rust does not install a PID-1 signal handler. Docker's init
                # forwards SIGTERM and reaps the worker during coordinated stop.
                cmd = ['docker', 'run', '-d', '--init', '--no-healthcheck', '--name', containers[rank],
                       '--label', 'project='+PROJECT, '--label', 'generation='+generation,
                       '--log-driver', 'none', '--gpus', 'all', '--network', 'host', '--ipc', 'host',
                       '--device', '/dev/infiniband', '--cap-add', 'IPC_LOCK', '--ulimit', 'memlock=-1',
                       '--security-opt', 'seccomp=unconfined', '--memory', '96g', '--memory-swap', '96g',
                       '-v', str(self.root/'data')+':/data:ro', '-v', str(run_dir)+':/state',
                       '--entrypoint', '/data/runtime/titanic-flash']
                for k, v in env.items():
                    cmd.extend(['-e', k+'='+v])
                cmd += [image, 'resident-http-peer', f'/state/rank{rank}.json']
                self.run(cmd, rank, timeout=60)
            until = time.monotonic()+900
            while time.monotonic() < until:
                if self.stop_file.exists():
                    raise InterruptedError('stop requested during startup')
                self.maintain_host_policy(state)
                for rank, name in enumerate(containers):
                    c = self.inspect(rank, name)
                    assert c and c['State']['Running'], f'rank {rank} exited during loading'
                h = self.health(state)
                if h and h.get('status') == 'ready':
                    state['ready_epoch'] = time.time(); atomic(self.state_file, state)
                    return state
                time.sleep(3)
            raise TimeoutError('startup exceeded 900 seconds')
        except BaseException:
            self.stop(state)
            raise

    def serve(self):
        with (self.state_dir/'lifecycle.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.stop_file.unlink(missing_ok=True)
            def stopping(*_):
                self.stop_file.touch(mode=0o600)
            signal.signal(signal.SIGTERM, stopping)
            signal.signal(signal.SIGINT, stopping)
            state = None
            try:
                state = self.launch()
                print(json.dumps(dict(status='ready', generation=state['generation'])), flush=True)
                failures = 0
                while not self.stop_file.exists():
                    self.maintain_host_policy(state)
                    h = self.health(state)
                    failures = 0 if h and h.get('status') == 'ready' else failures+1
                    assert failures < 3, 'head health failed repeatedly'
                    for rank, name in enumerate(state['containers']):
                        c = self.inspect(rank, name)
                        assert c and c['State']['Running'], f'rank {rank} exited'
                    time.sleep(5)
            except InterruptedError:
                if not self.stop_file.exists():
                    raise
            finally:
                self.stop(state or self.state())

def main():
    os.umask(0o077)
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['validate', 'serve', 'status', 'stop'])
    ap.add_argument('inventory', type=Path)
    args = ap.parse_args()
    if args.action == 'validate':
        validate(json.loads(args.inventory.read_text())); print('Inventory syntax valid; replace documentation addresses before use.'); return
    fleet = Fleet(args.inventory)
    if args.action == 'serve':
        fleet.serve()
    elif args.action == 'status':
        state = fleet.state()
        print(json.dumps(dict(cohort=state, health=fleet.health(state) if state and not state.get('stopped') else None), indent=2))
    else:
        fleet.stop_file.touch(mode=0o600)
        # A live supervisor owns cleanup. If it is absent, reclaim only our
        # recorded generation under the lifecycle lock.
        with (fleet.state_dir/'lifecycle.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                until=time.monotonic()+150
                while time.monotonic()<until:
                    if (fleet.state() or {}).get('stopped'):
                        print('All four ranks stopped.'); return
                    time.sleep(2)
                raise TimeoutError('supervisor did not finish stopping; inspect it before retrying')
            fleet.stop(fleet.state())

if __name__ == '__main__':
    main()
