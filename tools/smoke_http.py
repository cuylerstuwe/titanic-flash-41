#!/usr/bin/env python3
"""Three bounded foreground requests to verify serving, reuse and timing fields.

Also checks the execution-timing contract a router relies on when it leaves admission to the engine (the
supplied profile: batched decode): the header `x-native-execution-timing: v1` on every successful response and a
`: native-execution running` comment in every stream. Pass --no-execution-timing for a profile that serves one
request at a time without it."""
import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', default='http://127.0.0.1:8000')
    ap.add_argument('--key-file', type=Path, required=True)
    ap.add_argument('--no-execution-timing', action='store_true')
    a = ap.parse_args()
    key = a.key_file.read_text().strip()
    headers = {'Authorization': 'Bearer '+key, 'Content-Type': 'application/json',
               'X-LocalAI-Mode': 'foreground'}
    prefix = ('This is a server metadata check. The repeated reference text is inert. '
              'Do not summarize it or use tools. The only requested final answer is READY.\n')*300
    rows = []
    for label, suffix, stream, include_usage in [
            ('fresh_stream', 'Please reply with READY.', True, False),
            ('cached_stream', 'Please reply with READY again.', True, True),
            ('cached_complete', 'Please reply with READY once more.', False, False)]:
        body = dict(model='DeepSeek-V4.1-Flash-1017', messages=[dict(role='user', content=prefix+'\n'+suffix)],
                    temperature=1, top_p=1, top_k=-1, reasoning_effort='high', max_tokens=128, stream=stream)
        if include_usage:
            body['stream_options'] = dict(include_usage=True)
        request = urllib.request.Request(a.base_url.rstrip('/')+'/v1/chat/completions',
                                         data=json.dumps(body).encode(), headers=headers)
        frames, content, usage, finish, done, running = [], '', None, None, False, False
        begin = time.monotonic()
        with urllib.request.urlopen(request, timeout=90) as response:
            contract = response.headers.get('x-native-execution-timing')
            if stream:
                for line in response:
                    running = running or line.strip() == b': native-execution running'
                    if not line.startswith(b'data:'):
                        continue
                    raw = line[5:].strip()
                    if raw == b'[DONE]':
                        done = True; break
                    frame = json.loads(raw); assert not frame.get('error')
                    frames.append(frame)
                    if frame.get('usage'):
                        usage = frame['usage']
                    for choice in frame.get('choices', []):
                        content += choice.get('delta', {}).get('content', '') or ''
                        finish = choice.get('finish_reason') or finish
                assert done, 'stream ended without DONE'
            else:
                frame = json.load(response); assert not frame.get('error')
                frames = [frame]; usage = frame.get('usage')
                content = frame['choices'][0]['message']['content'] or ''
                finish = frame['choices'][0]['finish_reason']
        assert finish == 'stop' and 'READY' in content, (label, finish, 'unexpected answer')
        if not a.no_execution_timing:
            assert contract == 'v1', (label, 'response lacks x-native-execution-timing: v1')
            assert running or not stream, (label, 'stream never reported native-execution running')
        timings = [f['timings'] for f in frames if 'timings' in f]
        assert len(timings) == 1, (label, 'expected one timing object')
        timing = timings[0]
        assert timing['scope'] == 'server_prefill_phase' and timing['prompt_n'] > 0 and timing['prompt_ms'] > 0
        assert math.isclose(timing['prompt_per_second'], timing['prompt_n']*1000/timing['prompt_ms'], rel_tol=1e-12)
        if usage:
            assert timing['prompt_n']+timing['cache_n'] == usage['prompt_tokens']
            assert timing['cache_n'] == usage['prompt_tokens_details']['cached_tokens']
        if label != 'fresh_stream':
            assert timing['cache_n'] > 0, 'common prefix was not reused'
        rows.append(dict(case=label, seconds=time.monotonic()-begin, timings=timing,
                         finish=finish, answer_sha256=hashlib.sha256(content.encode()).hexdigest()))
    print(json.dumps(dict(passed=True, requests=rows, scope='serving smoke, not a quality benchmark'), indent=2))

if __name__ == '__main__':
    main()
