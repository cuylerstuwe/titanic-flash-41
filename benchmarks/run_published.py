#!/usr/bin/env python3
"""Run the published request/timing function, single-stream only, on localhost.

The external client is downloaded and hash-verified, not redistributed here.
Run on an idle head node. Cold timing requires a freshly started model server.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

REVISION = '7ae39b401e4e0887b2b7d5e4c4412cf60aa694f2'
CLIENT_SHA = '76a078842a0dd50b8345a6feb4455b890517eac51513f030701cd4a65d0a23ee'
CONTEXT_SHA = '6b4b21a0ed3ae0eefe2b0aab0b9555eb1332334347c8bda80a83a2eaee168457'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', default='http://127.0.0.1:8000')
    ap.add_argument('--key-file', type=Path, required=True)
    ap.add_argument('--model', default='DeepSeek-V4.1-Flash-1017')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    url = urllib.parse.urlsplit(a.base_url)
    assert url.scheme == 'http' and url.hostname in ['127.0.0.1', 'localhost', '::1']
    a.output.mkdir(parents=True, exist_ok=False)
    client = a.output/'external-client.py'
    source = f'https://gist.githubusercontent.com/kevintheo-ai/60eb093a5b4c54c4ac9a0b6b678408fa/raw/{REVISION}/tf_bench.py'
    with urllib.request.urlopen(source, timeout=60) as r:
        raw = r.read()
    assert hashlib.sha256(raw).hexdigest() == CLIENT_SHA, 'external benchmark changed'
    client.write_bytes(raw)
    os.environ.pop('CTX_DIR', None)
    spec = importlib.util.spec_from_file_location('published_client', client)
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
    bench.BASE = a.base_url.rstrip('/')
    bench.MODEL = a.model
    bench.KEY = a.key_file.read_text().strip()
    bench.THINK_KEY = 'thinking'
    context = (Path(__file__).parent/'contexts/slot0.txt').read_bytes()
    assert hashlib.sha256(context).hexdigest() == CONTEXT_SHA
    summary = {}
    for label in ['clean', '131k']:
        bench.LONG = label == '131k'
        bench.CTX = [context.decode() if bench.LONG else '']*4
        rows = []
        def capture(workload, tokens, tag, repeat=None):
            row = bench.one(workload, tokens, tag, slot=0)
            row['context'] = label
            if repeat is not None:
                row['rep'] = repeat
            row.pop('text_head', None)
            with (a.output/'samples.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            assert row.get('error') is None and row['completion_tokens'], row
            rows.append(row)
            return row
        warmups = [capture(w, 64, 'warmup') for w in ['prose', 'code', 'prose2']]
        phases = {}
        for w in ['prose', 'code', 'prose2']:
            samples = [capture(w, 512, 'c1-'+w, i) for i in range(3)]
            phases[w] = {k: bench.summarize(samples, k) for k in ['decode_tps', 'ttft_s', 'e2e_tps']}
        summary[label] = dict(warmups=warmups, phases=phases)
        (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
