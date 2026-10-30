#!/usr/bin/env python3
"""Explicit four-rank lifecycle, run on the head. Never stops unrelated services.

The resident Rust server owns scheduling, cache, cancellation and priorities.
This controller admits one cohort and fails the whole cohort if a rank dies.
"""
import argparse
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

PROJECT = 'titanic-flash-41'
SOURCE = Path(__file__).resolve().parents[1]

def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.chmod(0o600)
    temp.replace(path)

def validate(cfg):
    assert len(cfg['nodes']) == 4 and cfg['nodes'][0]['ssh'] is None
    root = Path(cfg['root'])
    assert root.is_absolute() and len(root.parts) >= 3 and '..' not in root.parts
    assert re.fullmatch(r'/[A-Za-z0-9_./-]+', str(root)), 'use a root without whitespace or shell metacharacters'
    assert re.fullmatch(r'[A-Za-z0-9_./:@-]+', cfg['image'])
    assert 1024 <= cfg['rendezvous_port'] <= 65530
    host, port = cfg['http_bind'].rsplit(':', 1)
    ip = ipaddress.ip_address(host)
    assert ip.version == 4 and 1024 <= int(port) <= 65535
    assert int(port) not in {cfg['rendezvous_port']+i for i in range(5)}
    assert ip.is_loopback or (cfg['allow_remote'] and ip.is_private and not ip.is_unspecified)
    addresses = []
    for rank, node in enumerate(cfg['nodes']):
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
        self.root = Path(self.cfg['root'])
        self.state_dir = self.root/'state'
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_dir.chmod(0o700)
        self.state_file = self.state_dir/'cohort.json'
        self.stop_file = self.state_dir/'stop-requested'
        self.digest = hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest()

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
        for rank, name in enumerate(state['containers']):
            try:
                c = self.inspect(rank, name)
                if c is None:
                    continue
                labels = c['Config']['Labels']
                assert labels.get('project') == PROJECT and labels.get('generation') == state['generation']
                assert c['Config']['Entrypoint'] == ['/data/runtime/titanic-flash']
                self.run(['docker', 'stop', '--time', '15', c['Id']], rank, timeout=25)
                self.run(['docker', 'rm', c['Id']], rank)
            except Exception as e:
                errors.append(f'rank {rank}: {e}')
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
            paths = [self.root/f'data/staged/layer{layer}-v1/r{rank}/atlas.json' for layer in range(40)]
            paths += [self.root/f'data/staged/{kind}/r{rank}/atlas.json' for kind in ['model-io-v1', 'dspark-mtp-v1']]
            paths += [self.root/f'data/engram/engram-l{layer}-r{rank}of4.bin' for layer in [1, 14]]
            if rank == 0:
                paths.append(self.root/'data/staged/vision-v1/r0/atlas.json')
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
                              priority_file='/state/priority.key', output_stall_seconds=60) if rank == 0 else None)

    def launch(self):
        self.stop(self.state())
        image = self.preflight()
        generation = secrets.token_hex(12)
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
                     containers=containers, started_epoch=time.time(), stopped=False)
        atomic(self.state_file, state)
        profile = json.loads((SOURCE/'deploy/profile.json').read_text())
        try:
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
                cmd = ['docker', 'run', '-d', '--no-healthcheck', '--name', containers[rank],
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
