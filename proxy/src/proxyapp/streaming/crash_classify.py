"""Classify upstream vs client errors for streaming and admission gating."""

from __future__ import annotations

import os
import re
from typing import Literal

TerminalCodeStr = Literal[
    "upstream_reset",
    "upstream_stall",
    "upstream_eof",
    "client_disconnect",
    "tool_boundary_crossed",
]


def stream_idle_timeout_seconds() -> float:
    raw = (os.environ.get("PROXY_STREAM_IDLE_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return 45.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 45.0


def _norm_msg(exc: BaseException | None, fallback: str = "") -> str:
    parts: list[str] = []
    if fallback:
        parts.append(fallback.lower())
    if exc is not None:
        parts.append(str(exc).lower())
        if getattr(exc, "__cause__", None) is not None:
            parts.append(str(exc.__cause__).lower())
    return " ".join(parts)


_RE_RESET = re.compile(
    r"(connection reset|connection aborted|broken pipe|errno 104|econnreset|remote end closed)",
    re.IGNORECASE,
)


def is_client_disconnect_message(msg: str) -> bool:
    m = (msg or "").lower()
    if "cannot write to closing transport" in m:
        return True
    if "cannot write" in m and "closing transport" in m:
        return True
    return False


def classify_stream_termination(
    *,
    exc: BaseException | None,
    message: str = "",
    saw_done: bool,
    mid_stream: bool,
    idle_timeout: bool = False,
) -> TerminalCodeStr:
    """Best-effort classification for gate + terminal SSE."""
    blob = _norm_msg(exc, message)
    if is_client_disconnect_message(blob):
        return "client_disconnect"
    if idle_timeout and mid_stream:
        return "upstream_stall"
    if _RE_RESET.search(blob):
        return "upstream_reset"
    if exc is not None and mid_stream:
        if idle_timeout:
            return "upstream_stall"
        low = blob
        if "eof" in low or "incomplete" in low:
            return "upstream_eof"
        return "upstream_reset"
    if not saw_done and not mid_stream:
        return "upstream_eof"
    return "upstream_eof"


def should_trip_admission_on_classify(code: TerminalCodeStr) -> bool:
    return code in ("upstream_reset", "upstream_stall", "upstream_eof")
