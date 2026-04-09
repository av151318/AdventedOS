"""Backend stream reads with per-chunk idle timeout (stall detection)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from aiohttp import StreamReader


class StreamIdleTimeoutError(TimeoutError):
    """No bytes from upstream within ``idle_seconds``."""


async def iter_chunked_with_idle(
    content: StreamReader,
    chunk_size: int,
    idle_seconds: float,
) -> AsyncIterator[bytes]:
    while True:
        try:
            chunk = await asyncio.wait_for(content.read(chunk_size), timeout=idle_seconds)
        except asyncio.TimeoutError as e:
            raise StreamIdleTimeoutError("upstream idle timeout") from e
        if not chunk:
            break
        yield chunk
