import asyncio

import pytest

from client import Client, Reply


class RecordingSleep:
    def __init__(self):
        self.delays = []

    async def __call__(self, delay):
        self.delays.append(delay)


class Transport:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def get(self, url):
        self.calls.append(url)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def test_successful_response_is_returned_without_sleeping():
    async def scenario():
        transport = Transport([Reply(200, b"ok")])
        sleep = RecordingSleep()
        client = Client(transport, sleep=sleep, jitter=lambda: 0.0)

        assert await client.fetch("https://example.test") == b"ok"
        assert transport.calls == ["https://example.test"]
        assert sleep.delays == []

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [408, 429, 500, 503, 599])
def test_retryable_http_statuses_are_retried(status):
    async def scenario():
        transport = Transport([Reply(status, b"retry"), Reply(200, b"ok")])
        sleep = RecordingSleep()
        client = Client(
            transport,
            attempts=2,
            base_delay=0.1,
            sleep=sleep,
            jitter=lambda: 0.0,
        )

        assert await client.fetch("u") == b"ok"
        assert len(transport.calls) == 2
        assert sleep.delays == pytest.approx([0.1])

    asyncio.run(scenario())


@pytest.mark.parametrize("exc_type", [OSError, ConnectionError, TimeoutError])
def test_oserror_transport_failures_are_retried(exc_type):
    async def scenario():
        transport = Transport([exc_type("temporary"), Reply(200, b"ok")])
        sleep = RecordingSleep()
        client = Client(transport, attempts=2, sleep=sleep, jitter=lambda: 0.0)

        assert await client.fetch("u") == b"ok"
        assert len(transport.calls) == 2
        assert sleep.delays == pytest.approx([0.1])

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "exc_type",
    [ValueError, AssertionError, RuntimeError, KeyError],
)
def test_non_oserror_transport_failures_propagate_immediately(exc_type):
    async def scenario():
        transport = Transport([exc_type("programming error")])
        sleep = RecordingSleep()
        client = Client(transport, attempts=4, sleep=sleep, jitter=lambda: 0.0)

        with pytest.raises(exc_type, match="programming error"):
            await client.fetch("u")

        assert len(transport.calls) == 1
        assert sleep.delays == []

    asyncio.run(scenario())


def test_non_retryable_http_status_is_not_retried():
    async def scenario():
        transport = Transport([Reply(404, b"missing")])
        sleep = RecordingSleep()
        client = Client(transport, sleep=sleep, jitter=lambda: 0.0)

        with pytest.raises(RuntimeError, match="HTTP 404"):
            await client.fetch("u")

        assert len(transport.calls) == 1
        assert sleep.delays == []

    asyncio.run(scenario())


def test_no_sleep_after_final_attempt():
    async def scenario():
        transport = Transport(
            [
                OSError("one"),
                OSError("two"),
                OSError("three"),
                OSError("four"),
            ]
        )
        sleep = RecordingSleep()
        client = Client(
            transport,
            attempts=4,
            base_delay=0.1,
            max_delay=5.0,
            sleep=sleep,
            jitter=lambda: 0.0,
        )

        with pytest.raises(OSError, match="four"):
            await client.fetch("u")

        assert len(transport.calls) == 4
        assert sleep.delays == pytest.approx([0.1, 0.2, 0.4])

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("retry_after", "expected_delay"),
    [(0.25, 0.25), (60.0, 5.0)],
)
def test_retry_after_is_respected_and_bounded(retry_after, expected_delay):
    async def scenario():
        transport = Transport(
            [
                Reply(429, b"slow down", retry_after=retry_after),
                Reply(200, b"ok"),
            ]
        )
        sleep = RecordingSleep()
        client = Client(
            transport,
            max_delay=5.0,
            sleep=sleep,
            jitter=lambda: 0.0,
        )

        assert await client.fetch("u") == b"ok"
        assert len(transport.calls) == 2
        assert sleep.delays == pytest.approx([expected_delay])

    asyncio.run(scenario())


def test_jittered_backoff_never_exceeds_max_delay():
    async def scenario():
        transport = Transport([OSError("one"), OSError("two"), Reply(200, b"ok")])
        sleep = RecordingSleep()
        client = Client(
            transport,
            attempts=3,
            base_delay=4.0,
            max_delay=5.0,
            sleep=sleep,
            jitter=lambda: 1.0,
        )

        assert await client.fetch("u") == b"ok"
        assert sleep.delays == pytest.approx([5.0, 5.0])
        assert all(delay <= 5.0 for delay in sleep.delays)

    asyncio.run(scenario())


def test_sleep_and_jitter_are_injectable_and_deterministic():
    async def scenario():
        transport = Transport([OSError("one"), OSError("two"), Reply(200, b"ok")])
        sleep = RecordingSleep()
        client = Client(
            transport,
            attempts=3,
            base_delay=1.0,
            max_delay=10.0,
            sleep=sleep,
            jitter=lambda: 0.5,
        )

        assert await client.fetch("u") == b"ok"
        assert sleep.delays == pytest.approx([1.5, 3.0])

    asyncio.run(scenario())


def test_cancellation_from_transport_propagates_without_retry_or_sleep():
    async def scenario():
        transport = Transport([asyncio.CancelledError()])
        sleep = RecordingSleep()
        client = Client(transport, sleep=sleep, jitter=lambda: 0.0)

        with pytest.raises(asyncio.CancelledError):
            await client.fetch("u")

        assert len(transport.calls) == 1
        assert sleep.delays == []

    asyncio.run(scenario())


def test_cancellation_during_sleep_propagates():
    class CancellingSleep:
        async def __call__(self, delay):
            raise asyncio.CancelledError()

    async def scenario():
        transport = Transport([OSError("temporary"), Reply(200, b"ok")])
        client = Client(
            transport,
            attempts=3,
            sleep=CancellingSleep(),
            jitter=lambda: 0.0,
        )

        with pytest.raises(asyncio.CancelledError):
            await client.fetch("u")

        assert len(transport.calls) == 1

    asyncio.run(scenario())
