#!/usr/bin/env python3
"""Copy only runtime assets and each peer's TP4 shards. Run on the head."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'deploy'))
from fleet import validate, vision_encoder_rank, rank_data_paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inventory', type=Path)
    a = ap.parse_args()
    cfg = validate(json.loads(a.inventory.read_text()))
    profile = json.loads((Path(__file__).resolve().parents[1]/'deploy/profile.json').read_text())
    vision_rank = vision_encoder_rank(profile)
    root = Path(cfg['root'])
    for rank, node in enumerate(cfg['nodes'][1:], 1):
        host = node['ssh']
        assert subprocess.check_output(['ssh', '-o', 'BatchMode=yes', host, 'hostname'], text=True).strip() == node['hostname']
        # All arguments cross SSH as a quoted command, never executable inventory text.
        import shlex
        subprocess.run(['ssh', host, shlex.join(['mkdir', '-p', str(root/'data')])], check=True)
        paths = ['runtime/', 'model/', 'kernels/']
        paths += rank_data_paths(rank, vision_rank)
        assert all((root/'data'/p).exists() for p in paths), f'missing staged data for rank {rank}'
        with tempfile.NamedTemporaryFile(mode='w') as listing:
            listing.write('\n'.join(paths)+'\n'); listing.flush()
            subprocess.run(['rsync', '-rlt', '--partial', '--partial-dir=.rsync-partial',
                            '--files-from='+listing.name, str(root/'data')+'/',
                            host+':'+str(root/'data')+'/'], check=True)
        # Files are immutable after staging; native load independently hashes every atlas.
        print(f'Rank {rank} runtime and weight shards copied.', flush=True)

if __name__ == '__main__':
    main()
