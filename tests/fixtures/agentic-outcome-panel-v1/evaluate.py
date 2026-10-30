#!/usr/bin/env python3
"""Held-out outcome checks for the frozen python-http-retry-fix task.

Run generated candidate code only in the bounded, network-free evaluation
container. This script does not launch a container or modify candidate code.
Hook-name adapters are reported; unfamiliar APIs require review, not a guessed
failure. Results concern this task, not general model intelligence.
"""
import argparse
import asyncio
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import time

HOOKS = {
    'sleep': ('sleep', 'sleep_fn', 'sleep_func', 'sleeper'),
    'jitter': ('jitter', 'jitter_fn', 'jitter_func'),
}


def load(path):
    spec = importlib.util.spec_from_file_location('candidate_client', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Transport:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.calls = 0

    async def get(self, url):
        assert url == 'https://example.invalid/test'
        index = self.calls
        self.calls += 1
        if index >= len(self.sequence):
            raise AssertionError('candidate made an extra transport call')
        item = self.sequence[index]
        if isinstance(item, BaseException):
            raise item
        return item


class Hooks:
    def __init__(self, cancel_sleep=False, jitter_value=0.):
        self.delays = []
        self.jitter_calls = []
        self.cancel_sleep = cancel_sleep
        self.jitter_value = jitter_value

    async def sleep(self, delay):
        assert isinstance(delay, (int, float)) and math.isfinite(delay)
        self.delays.append(float(delay))
        if self.cancel_sleep:
            raise asyncio.CancelledError('cancelled during backoff')

    def jitter(self, *args, **kwargs):
        self.jitter_calls.append((args, kwargs))
        return self.jitter_value


def evaluate(module):
    signature = inspect.signature(module.Client)
    selected = {kind: next((n for n in names if n in signature.parameters), None)
                for kind, names in HOOKS.items()}
    report = dict(task='python-http-retry-fix', hook_parameters=selected,
                  adapter_review_required=not all(selected.values()), tests=[])
    if report['adapter_review_required']:
        report['reason'] = 'Recognize the candidate hook API before assigning a verdict; no code repair is allowed.'
        return report
    reply = lambda status=200, retry_after=None: module.Reply(status, b'expected body', retry_after)

    async def scenario(sequence, *, calls, sleeps, raises=False, cancel_sleep=False,
                       min_delay=None, max_delay=2., attempts=4, require_jitter=False, jitter_value=0.):
        transport = Transport(sequence)
        hooks = Hooks(cancel_sleep, jitter_value)
        client = module.Client(transport, attempts=attempts, base_delay=.1,
                               max_delay=max_delay,
                               **{selected['sleep']: hooks.sleep, selected['jitter']: hooks.jitter})
        caught = None
        result = None
        try:
            result = await client.fetch('https://example.invalid/test')
        except BaseException as error:
            caught = error
        assert transport.calls == calls, f'transport calls: {transport.calls}, expected {calls}'
        assert len(hooks.delays) == sleeps, f'sleeps: {len(hooks.delays)}, expected {sleeps}'
        assert all(0 <= d <= max_delay for d in hooks.delays), f'unbounded delays: {hooks.delays}'
        if min_delay is not None:
            assert all(d >= min_delay for d in hooks.delays), f'Retry-After ignored: {hooks.delays}'
        if raises is asyncio.CancelledError:
            assert isinstance(caught, asyncio.CancelledError), f'cancellation became {caught!r}'
        elif raises:
            assert isinstance(caught, Exception), f'expected failure, got {result!r}, {caught!r}'
            assert not isinstance(caught, AssertionError), str(caught)
        else:
            assert caught is None, f'unexpected {type(caught).__name__}: {caught}'
            assert result == b'expected body', f'success body changed: {result!r}'
        if require_jitter:
            assert hooks.jitter_calls, 'injected jitter was never used'
        return dict(transport_calls=transport.calls, sleeps=hooks.delays, jitter_calls=len(hooks.jitter_calls))

    def check(name, factory, required=True):
        started = time.perf_counter()
        try:
            details = asyncio.run(factory())
            row = dict(name=name, passed=True, details=details)
        except BaseException as error:
            row = dict(name=name, passed=False, error=f'{type(error).__name__}: {error}')
        row.update(required=required, evaluation_seconds=time.perf_counter()-started)
        report['tests'].append(row)

    # The supplied helper returns the body for every status below400, including
    # redirect responses returned directly by its injected transport.
    for status in [200, 201, 204, 301, 302, 304]:
        check(f'success_{status}', lambda status=status: scenario([reply(status)], calls=1, sleeps=0))
    for status in [400, 401, 403, 404, 409, 422]:
        check(f'non_retryable_http_{status}', lambda status=status: scenario([reply(status)], calls=1, sleeps=0, raises=True))
    for status in [408, 429, 500, 502, 503, 599]:
        check(f'retryable_http_{status}', lambda status=status: scenario([reply(status), reply()], calls=2, sleeps=1))
    for kind in [ConnectionError, TimeoutError, OSError]:
        check(f'transport_{kind.__name__}', lambda kind=kind: scenario([kind('network'), reply()], calls=2, sleeps=1))
    check('injected_jitter_used', lambda: scenario([ConnectionError('network'),reply()], calls=2, sleeps=1, require_jitter=True))
    async def jitter_comparison(repeat):
        values = [.25, .25] if repeat else [0., .25]
        results = [await scenario([ConnectionError('network'), ConnectionError('network'), reply()],
                   calls=3, sleeps=2, require_jitter=True, jitter_value=v) for v in values]
        same = results[0]['sleeps'] == results[1]['sleeps']
        assert same == repeat, ('injected hooks are not deterministic' if repeat else 'injected jitter return was ignored')
        return dict(injected_values=values, delays=[r['sleeps'] for r in results])
    check('injected_jitter_controls_delay', lambda: jitter_comparison(False))
    check('injected_hooks_are_deterministic', lambda: jitter_comparison(True))
    check('programming_error_not_retried', lambda: scenario([ValueError('program bug')], calls=1, sleeps=0, raises=True))
    check('transport_exhausted_no_final_sleep', lambda: scenario([ConnectionError('network') for _ in range(4)], calls=4, sleeps=3, raises=True))
    check('http_exhausted_no_final_sleep', lambda: scenario([reply(503) for _ in range(4)], calls=4, sleeps=3, raises=True))
    check('one_attempt_never_sleeps', lambda: scenario([ConnectionError('network')], calls=1, sleeps=0, raises=True, attempts=1))
    for retry_after in [0., 1., 100.]:
        check(f'retry_after_{retry_after}', lambda retry_after=retry_after: scenario([reply(429,retry_after),reply()], calls=2,sleeps=1,min_delay=min(retry_after,2.)))
    check('negative_retry_after_still_bounded', lambda: scenario([reply(429,-2.),reply()],calls=2,sleeps=1))
    check('cancel_transport', lambda: scenario([asyncio.CancelledError('cancelled')],calls=1,sleeps=0,raises=asyncio.CancelledError))
    check('cancel_backoff', lambda: scenario([reply(503)],calls=1,sleeps=1,raises=asyncio.CancelledError,cancel_sleep=True))
    required = [r for r in report['tests'] if r['required']]
    report.update(required_checks=len(required), passed_required=sum(r['passed'] for r in required),
                  task_checks_pass=all(r['passed'] for r in required),
                  scope='Held-out behavior checks for the supplied task. Supplied pytest quality, explanatory accuracy and ambiguous API choices need separate review. No performance score for a failed or unreviewed result.')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        report = evaluate(load(args.candidate))
    except BaseException as error:
        report = dict(task='python-http-retry-fix', task_checks_pass=False,
                      load_or_api_error=f'{type(error).__name__}: {error}')
    report['evaluation_seconds'] = time.perf_counter()-started
    print('CANADA_OUTCOME='+json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
