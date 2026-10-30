#!/usr/bin/env python3
"""Stage exact TP4 weight atlases through the Rust GPU loader, one layer at a time."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('/data'))
    ap.add_argument('--model', type=Path, default=Path('/model'))
    ap.add_argument('--rank', choices=['0', '1', '2', '3', 'all'], default='all')
    a = ap.parse_args()
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    assert not active.strip(), 'stop serving before GPU staging; another GPU process is active'
    binary = a.data/'runtime/titanic-flash'
    config = a.data/'model/config.json'
    assert json.loads(config.read_text()) == json.loads((a.model/'config.json').read_text()), 'canonical model configuration differs'
    checkpoint = a.data/'model/checkpoint.json'
    if not checkpoint.exists():
        temporary = checkpoint.with_suffix('.json.part')
        assert not temporary.exists(), 'partial checkpoint manifest preserved'
        subprocess.run([str(binary), 'scan', str(a.model), str(temporary)], check=True)
        temporary.replace(checkpoint)
    checkpoint_hash = sha(checkpoint)
    ranks = range(4) if a.rank == 'all' else [int(a.rank)]
    for rank in ranks:
        selections = [(f'layer{layer}-v1/r{rank}', 'gpu-stage-layer', str(layer), [f'layers.{layer}.']) for layer in range(40)]
        selections += [(f'model-io-v1/r{rank}', 'gpu-stage-prefix', 'model-io', ['embed.', 'head.', 'norm.']),
                       (f'dspark-mtp-v1/r{rank}', 'gpu-stage-prefix', 'mtp.', ['mtp.'])]
        if rank == 0:
            selections.append(('vision-v1/r0', 'gpu-stage-prefix', 'vision',
                               ['vision.', 'aligner.', 'image_start', 'image_newline', 'image_end']))
        for name, command, prefix, prefixes in selections:
            out = a.data/'staged'/name
            if out.exists():
                metadata = json.loads((out/'atlas.json').read_text())
                assert metadata['rank'] == rank and metadata['world'] == 4
                assert metadata['prefixes'] == prefixes and metadata['checkpoint_sha256'] == checkpoint_hash
                assert (out/'atlas.bin').stat().st_size == metadata['bytes']
                assert sha(out/'atlas.bin') == metadata['atlas_sha256'], f'existing atlas mismatch: {name}'
                continue
            assert shutil.disk_usage(a.data).free > 45*2**30, 'weight staging disk reserve'
            out.parent.mkdir(parents=True, exist_ok=True)
            temporary = out.with_name(out.name+'.part')
            assert not temporary.exists(), f'partial stage preserved: {temporary}'
            subprocess.run([str(binary), command, str(config), str(checkpoint), str(a.model),
                            str(rank), prefix, str(temporary)], check=True)
            metadata = json.loads((temporary/'atlas.json').read_text())
            assert sha(temporary/'atlas.bin') == metadata['atlas_sha256']
            temporary.replace(out)
            print(json.dumps(dict(rank=rank, staged=name, bytes=metadata['bytes'])), flush=True)

if __name__ == '__main__':
    main()
