import asyncio
import random
from dataclasses import dataclass

@dataclass
class Reply:
    status: int
    body: bytes
    retry_after: float | None = None

class Client:
    def __init__(self, transport, *, attempts=4, base_delay=0.1, max_delay=5.0):
        self.transport = transport
        self.attempts = attempts
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def fetch(self, url):
        last = None
        for attempt in range(self.attempts):
            try:
                reply = await self.transport.get(url)
                if reply.status >= 400:
                    raise RuntimeError(f'HTTP {reply.status}')
                return reply.body
            except Exception as exc:
                last = exc
                delay = min(self.max_delay, self.base_delay * 2 ** attempt)
                await asyncio.sleep(delay + random.random() * delay)
        raise last
