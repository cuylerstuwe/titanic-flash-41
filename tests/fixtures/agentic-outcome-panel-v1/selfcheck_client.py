from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable

Sleep = Callable[[float], Awaitable[None]]
Jitter = Callable[[], float]


@dataclass
class Reply:
    status: int
    body: bytes
    retry_after: float | None = None


class Client:
    def __init__(
        self,
        transport,
        *,
        attempts: int = 4,
        base_delay: float = 0.1,
        max_delay: float = 5.0,
        sleep: Sleep | None = None,
        jitter: Jitter | None = None,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be at least 1")

        self.transport = transport
        self.attempts = attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._jitter = jitter if jitter is not None else random.random

    @staticmethod
    def _is_retryable_status(status: int) -> bool:
        return status in (408, 429) or 500 <= status <= 599

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return max(0.0, min(self.max_delay, retry_after))

        backoff = max(0.0, min(self.max_delay, self.base_delay * (2 ** attempt)))
        delay = backoff + self._jitter() * backoff
        return max(0.0, min(self.max_delay, delay))

    async def fetch(self, url):
        last_error: Exception | None = None

        for attempt in range(self.attempts):
            retry_after: float | None = None

            try:
                reply = await self.transport.get(url)
            except asyncio.CancelledError:
                raise
            except OSError as exc:
                # Only OSError and subclasses are transport failures.
                last_error = exc
            else:
                if reply.status < 400:
                    return reply.body

                error = RuntimeError(f"HTTP {reply.status}")

                if not self._is_retryable_status(reply.status):
                    raise error

                last_error = error
                retry_after = reply.retry_after

            if attempt + 1 >= self.attempts:
                break

            await self._sleep(self._delay(attempt, retry_after))

        if last_error is None:
            raise RuntimeError("retry loop ended without an error")

        raise last_error
