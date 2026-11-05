#!/usr/bin/env python3
"""Small opt-in context-transport check; never logs credentials or response text."""
import argparse
import hashlib
import json
import secrets
import urllib.error
import urllib.request
from pathlib import Path


def chain(messages):
    key = hashlib.sha256(b'titanic-context-v1/root').digest()
    for raw in messages:
        digest = hashlib.sha256(b'titanic-context-v1/message\0' + raw.encode()).digest()
        key = hashlib.sha256(b'titanic-context-v1/chain\0' + key + digest).digest()
    return key.hex()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--key-file', type=Path, required=True)
    a = p.parse_args()
    key = a.key_file.read_text().strip()
    namespace = secrets.token_hex(16)
    controls = dict(model='DeepSeek-V4.1-Flash-1017', temperature=1, top_p=1,
                    top_k=-1, reasoning_effort='high', max_tokens=32, stream=False)
    raw = lambda role, content: json.dumps(dict(role=role, content=content), separators=(',', ':'))
    initial = [raw('system', 'Answer simple factual questions.'), raw('user', 'Name a fruit.')]
    messages = initial
    previous = None
    records = []
    for label, keep, append in [
        ('full', 0, initial),
        ('extend', 2, [raw('assistant', 'An apple.'), raw('user', 'Name a different fruit.')]),
        ('rewind', 2, [raw('assistant', 'An apple.'), raw('user', 'Name a vegetable.')]),
    ]:
        messages = messages[:keep] + append
        identifier = secrets.token_hex(16)
        envelope = dict(version=1, namespace=namespace, id=identifier,
                        base=None if previous is None else dict(key=previous, keep=keep),
                        append=append, key=chain(messages), request=controls)
        request = urllib.request.Request(a.base_url.rstrip('/')+'/v1/context/chat/completions',
            data=json.dumps(envelope).encode(), headers={'Authorization':'Bearer '+key,
                'Content-Type':'application/json', 'X-Titanic-Context':'v1'})
        with urllib.request.urlopen(request, timeout=90) as response:
            assert response.headers['x-titanic-context-ack'] == identifier
            assert response.headers['x-titanic-context-key'] == envelope['key']
            assert response.headers['x-titanic-context-stored'] == '1', 'context was not retained for this check'
            result = json.load(response)
        assert result['usage']['completion_tokens'] > 0
        assert result['choices'][0]['finish_reason'] in ['stop', 'length']
        records.append(dict(case=label, messages=len(messages), appended=len(append),
                            usage=result['usage'], timings=result.get('timings')))
        previous = envelope['key']
    envelope.update(id=secrets.token_hex(16), base=dict(key='0'*64, keep=2))
    request = urllib.request.Request(a.base_url.rstrip('/')+'/v1/context/chat/completions',
        data=json.dumps(envelope).encode(), headers={'Authorization':'Bearer '+key,
            'Content-Type':'application/json', 'X-Titanic-Context':'v1'})
    try:
        urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as error:
        assert error.code == 409
        assert error.headers.get('x-titanic-context-ack') is None
        assert json.load(error)['error']['code'] == 'context_miss'
    else:
        raise AssertionError('unknown context was silently accepted')
    print(json.dumps(dict(passed=True, requests=4, records=records, missing_context_refused=True)))


if __name__ == '__main__':
    main()
