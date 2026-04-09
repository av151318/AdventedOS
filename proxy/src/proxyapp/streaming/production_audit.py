"""Single-request production audit record: inbound → backend → stream → root cause."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any

from ..audit_ring import store_record
from ..tracing import bound_for_contract, emit_contract_event


def sse_audit_comment_enabled() -> bool:
    return os.environ.get("PROXY_SSE_AUDIT_COMMENT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def normalization_delta(raw: dict[str, Any] | None, normalized: dict[str, Any] | None) -> dict[str, Any]:
    """Fields present in client JSON but not forwarded after normalize_openai_chat_request."""
    if not isinstance(raw, dict) or not isinstance(normalized, dict):
        return {"error": "missing_raw_or_normalized"}
    raw_keys = set(raw.keys())
    norm_keys = set(normalized.keys())
    dropped = sorted(raw_keys - norm_keys)
    added_only_in_norm = sorted(norm_keys - raw_keys)
    return {
        "client_key_count": len(raw_keys),
        "normalized_key_count": len(norm_keys),
        "dropped_from_forwarding": dropped,
        "keys_only_normalized": added_only_in_norm,
    }


def compute_tool_turn_root_cause(
    *,
    wants_tools: bool,
    had_backend_tool_delta: bool,
    proxy_emitted_structured_tool: bool,
    content_tool_like: bool,
    reasoning_tool_like: bool,
    responses_endpoint_used: bool,
    responses_tools_forwarded: bool | None,
    http_stream_status: int,
    admission_failure_class: str | None,
) -> tuple[str, str]:
    """Return (machine_code, human_summary)."""
    if admission_failure_class:
        if admission_failure_class in (
            "admission_not_ready",
            "admission_circuit_open",
            "admission_prompt_too_large",
        ):
            return (
                admission_failure_class,
                f"Request rejected before backend: {admission_failure_class}",
            )
    if http_stream_status >= 500:
        return (
            "engine_degraded_transport",
            f"Backend HTTP {http_stream_status}; diagnose readiness/container separately from tool contract",
        )
    if responses_endpoint_used and responses_tools_forwarded is False:
        return (
            "responses_tools_dropped",
            "/v1/responses carried tools/tool_choice but mapper did not forward them to chat (contract bug)",
        )
    if not wants_tools:
        return ("ok_no_tools_requested", "Client did not request tools")
    if had_backend_tool_delta or proxy_emitted_structured_tool:
        return ("ok_structured_tools", "Structured tool deltas observed on the wire")
    if content_tool_like or reasoning_tool_like:
        return (
            "content_backfilled_tool_intent_without_structured_tool_call",
            "Tool-like markup/text in content or reasoning but no choices[].delta.tool_calls from backend",
        )
    return (
        "backend_missing_structured_tools",
        "Client sent tools but backend stream had no structured delta.tool_calls",
    )


@dataclass
class ProductionAuditSession:
    request_id: str
    route: str
    api: str
    inbound_raw: dict[str, Any] | None = None
    normalized_client: dict[str, Any] | None = None
    normalization_delta: dict[str, Any] = field(default_factory=dict)
    backend_payload: dict[str, Any] | None = None
    backend_url: str | None = None
    wants_tools: bool = False
    had_backend_tool_delta: bool = False
    proxy_emitted_structured_tool: bool = False
    content_had_tool_like_markup: bool = False
    reasoning_had_tool_like_markup: bool = False
    responses_endpoint_used: bool = False
    chat_endpoint_used: bool = False
    responses_tools_forwarded: bool | None = None
    finish_reason: str | None = None
    backend_health_state: str | None = None
    admission_decision: str | None = None
    admission_failure_class: str | None = None
    sanitizer_fragments: list[dict[str, Any]] = field(default_factory=list)
    http_final_stream_status: int = 200
    tool_audit_snapshot: dict[str, Any] | None = None
    normalization_anthropic_stripped: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.sanitizer_fragments, list):
            self.sanitizer_fragments = []

    def merge_sanitizer_audit(self, fragments: list[dict[str, Any]] | None) -> None:
        if not fragments:
            return
        self.sanitizer_fragments.extend(fragments)

    def to_record(
        self,
        *,
        root_cause_code: str,
        root_cause_summary: str,
    ) -> dict[str, Any]:
        return {
            "v": 1,
            "request_id": self.request_id,
            "route": self.route,
            "api": self.api,
            "root_cause_code": root_cause_code,
            "root_cause_summary": root_cause_summary,
            "wants_tools": self.wants_tools,
            "had_backend_tool_delta": self.had_backend_tool_delta,
            "proxy_emitted_structured_tool": self.proxy_emitted_structured_tool,
            "content_had_tool_like_markup": self.content_had_tool_like_markup,
            "reasoning_had_tool_like_markup": self.reasoning_had_tool_like_markup,
            "responses_endpoint_used": self.responses_endpoint_used,
            "chat_endpoint_used": self.chat_endpoint_used,
            "responses_tools_forwarded": self.responses_tools_forwarded,
            "finish_reason": self.finish_reason,
            "backend_health_state": self.backend_health_state,
            "admission_decision": self.admission_decision,
            "admission_failure_class": self.admission_failure_class,
            "http_final_stream_status": self.http_final_stream_status,
            "normalization_delta": self.normalization_delta,
            "normalization_anthropic_stripped": self.normalization_anthropic_stripped,
            "inbound_raw": bound_for_contract(copy.deepcopy(self.inbound_raw))
            if self.inbound_raw
            else None,
            "normalized_client": bound_for_contract(copy.deepcopy(self.normalized_client))
            if self.normalized_client
            else None,
            "backend_payload": bound_for_contract(copy.deepcopy(self.backend_payload))
            if self.backend_payload
            else None,
            "backend_url": self.backend_url,
            "sanitizer_fragments": bound_for_contract(self.sanitizer_fragments[-200:]),
            "tool_audit": bound_for_contract(self.tool_audit_snapshot) if self.tool_audit_snapshot else None,
        }

    def emit(
        self,
        *,
        root_cause_code: str,
        root_cause_summary: str,
        failure_taxonomy_extra: str | None = None,
    ) -> dict[str, Any]:
        rec = self.to_record(root_cause_code=root_cause_code, root_cause_summary=root_cause_summary)
        emit_contract_event(
            "production_audit",
            self.request_id,
            self.api,
            self.route,
            failure_taxonomy_extra=failure_taxonomy_extra,
            audit=rec,
        )
        store_record(
            request_id=self.request_id,
            route=self.route,
            production_audit=rec,
            inbound_raw=self.inbound_raw,
            inbound=self.normalized_client,
            backend_request=self.backend_payload,
        )
        return rec

    def sse_comment_line(self, rec: dict[str, Any]) -> str:
        """ASCII-only JSON for SSE comment line (ignored by most parsers)."""
        blob = {
            "proxy_audit_v1": {
                "request_id": rec["request_id"],
                "root_cause_code": rec["root_cause_code"],
                "root_cause_summary": rec["root_cause_summary"],
                "wants_tools": rec["wants_tools"],
                "had_backend_tool_delta": rec["had_backend_tool_delta"],
            }
        }
        return json.dumps(blob, ensure_ascii=True, separators=(",", ":"))
