import asyncio
import pytest
from client import Client, Reply

class Transport:
    def __init__(self, items):
        self.items = list(items)
        self.calls = 0
    async def get(self, url):
        self.calls += 1
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

def test_success_including_redirect_keeps_body():
    for status in [200, 302]:
        t = Transport([Reply(status, b'ok')])
        assert asyncio.run(Client(t, base_delay=0).fetch('url')) == b'ok'
        assert t.calls == 1

def test_http_404_is_not_retried():
    t = Transport([Reply(404, b'bad')] * 4)
    with pytest.raises(RuntimeError):
        asyncio.run(Client(t, base_delay=0).fetch('url'))
    assert t.calls == 1

def test_programming_error_propagates_without_retry():
    error = ValueError('programming defect')
    t = Transport([error] * 4)
    with pytest.raises(ValueError) as caught:
        asyncio.run(Client(t, base_delay=0).fetch('url'))
    assert caught.value is error
    assert t.calls == 1

def test_cancellation_propagates():
    t = Transport([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(Client(t, base_delay=0).fetch('url'))
    assert t.calls == 1

def test_exhausted_http_never_sleeps_after_final_attempt():
    calls = []
    async def sleep(delay):
        calls.append(delay)
    t = Transport([Reply(503, b'bad')] * 4)
    with pytest.raises(RuntimeError, match='HTTP 503'):
        asyncio.run(Client(t, sleep=sleep, jitter=lambda: 0).fetch('url'))
    assert t.calls == 4
    assert len(calls) == 3

def test_retry_after_is_bounded():
    delays = []
    async def sleep(delay):
        delays.append(delay)
    t = Transport([Reply(429, b'bad', 100.0), Reply(200, b'ok')])
    assert asyncio.run(Client(t, max_delay=5, sleep=sleep, jitter=lambda: 0).fetch('url')) == b'ok'
    assert delays == [5.0]
