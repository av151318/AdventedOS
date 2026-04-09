"""Committed output ledger for retry policy and terminal SSE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class StreamCommittedLedger:
    request_id: str
    first_byte_sent: bool = False
    committed_text_offset: int = 0
    committed_reasoning_offset: int = 0
    last_tool_event: Optional[dict[str, Any]] = None
    terminal_classifier: str = ""
    idempotency_key: Optional[str] = None

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "first_byte_sent": self.first_byte_sent,
            "committed_text_offset": self.committed_text_offset,
            "committed_reasoning_offset": self.committed_reasoning_offset,
            "last_tool_event": self.last_tool_event,
            "terminal_classifier": self.terminal_classifier,
            "idempotency_key": self.idempotency_key,
        }


def extract_idempotency_key(body: Mapping[str, Any]) -> Optional[str]:
    raw = body.get("idempotency_key")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    meta = body.get("metadata")
    if isinstance(meta, dict):
        m = meta.get("idempotency_key")
        if isinstance(m, str) and m.strip():
            return m.strip()
    return None


def _tools_absent(body: Mapping[str, Any]) -> bool:
    t = body.get("tools")
    return t is None or t == []


def is_retry_eligible(ledger: StreamCommittedLedger, request_body: Mapping[str, Any]) -> bool:
    """Read-only safe path: no tool defs in request and no tool events flushed to client."""
    return _tools_absent(request_body) and ledger.last_tool_event is None
