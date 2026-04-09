"""Client-facing stream writes with disconnect detection."""

from __future__ import annotations

import asyncio
from typing import Literal

from aiohttp.web import StreamResponse

from .crash_classify import is_client_disconnect_message


async def sse_write(
    resp: StreamResponse,
    data: bytes,
) -> Literal["ok", "client_disconnect"]:
    try:
        await resp.write(data)
        return "ok"
    except asyncio.CancelledError:
        raise
    except Exception as e:
        if is_client_disconnect_message(str(e)):
            return "client_disconnect"
        raise
