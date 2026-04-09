"""Per-request aggregates for [trace] request_tool_audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..parsers.nemotron_content_tools import raw_visible_suggests_tool_calls
from ..policy import content_tool_fallback_enabled, fallback_may_run, strict_oai_tools


@dataclass
class ToolStreamAudit:
    api: str
    backend_chunks_with_tool_calls: int = 0
    backend_tool_arg_chars: int = 0
    proxy_emitted_chat_tool_chunks: int = 0
    proxy_emitted_chat_tool_arg_chars: int = 0
    proxy_response_function_arg_deltas: int = 0
    proxy_response_function_arg_chars: int = 0
    client_reasoning_chars: int = 0
    client_visible_chars: int = 0
    accumulated_raw_content: str = ""
    fallback_synthetic_tools: bool = False
    fallback_synthetic_count: int = 0
    sanitizer_eof: dict[str, Any] = field(default_factory=dict)

    def note_backend_delta(self, delta_obj: dict) -> None:
        tcs = delta_obj.get("tool_calls") or []
        if not tcs:
            return
        self.backend_chunks_with_tool_calls += 1
        for tc in tcs:
            fn = tc.get("function") or {}
            self.backend_tool_arg_chars += len(fn.get("arguments") or "")

    def note_raw_content(self, chunk: str) -> None:
        if chunk:
            self.accumulated_raw_content += chunk

    def note_proxy_chat_out(self, event: dict) -> None:
        ch0 = (event.get("choices") or [{}])[0]
        delta = ch0.get("delta") or {}
        tcs = delta.get("tool_calls") or []
        if not tcs:
            return
        self.proxy_emitted_chat_tool_chunks += 1
        for tc in tcs:
            fn = tc.get("function") or {}
            self.proxy_emitted_chat_tool_arg_chars += len(fn.get("arguments") or "")

    def note_proxy_responses_out(self, event: dict) -> None:
        et = event.get("type") or ""
        if "function_call_arguments.delta" in et:
            self.proxy_response_function_arg_deltas += 1
            self.proxy_response_function_arg_chars += len(event.get("delta") or "")

    def note_client_reasoning(self, n: int) -> None:
        self.client_reasoning_chars += n

    def note_client_visible(self, n: int) -> None:
        self.client_visible_chars += n

    def build_record(
        self,
        request_id: str,
        *,
        content_had_tool_like_markup: bool | None = None,
        reasoning_had_tool_like_markup: bool | None = None,
    ) -> dict[str, Any]:
        backend_had = self.backend_chunks_with_tool_calls > 0
        down_chat = self.proxy_emitted_chat_tool_chunks > 0
        down_res = self.proxy_response_function_arg_deltas > 0
        synthetic = self.fallback_synthetic_tools

        flags: list[str] = []
        if not backend_had:
            flags.append("STRUCTURED_TOOLS_ABSENT")
        raw = self.accumulated_raw_content
        raw_suggests = not backend_had and raw_visible_suggests_tool_calls(raw)
        if raw_suggests:
            flags.append("TOOL_LIKE_PATTERN_IN_RAW_VISIBLE")
        if synthetic:
            flags.append("FALLBACK_EMITTED")
        if content_tool_fallback_enabled() and strict_oai_tools():
            flags.append("fallback_skipped_strict_mode")

        c_tool = bool(content_had_tool_like_markup if content_had_tool_like_markup is not None else raw_suggests)
        r_tool = bool(reasoning_had_tool_like_markup) if reasoning_had_tool_like_markup is not None else False
        if c_tool:
            flags.append("CONTENT_TOOL_LIKE_MARKUP")
        if r_tool:
            flags.append("REASONING_TOOL_LIKE_MARKUP")

        return {
            "request_id": request_id,
            "api": self.api,
            "strict_oai_tools": strict_oai_tools(),
            "content_tool_fallback_env": content_tool_fallback_enabled(),
            "fallback_may_run": fallback_may_run(),
            "backend_chunks_with_tool_calls": self.backend_chunks_with_tool_calls,
            "backend_had_structured_tool_deltas": backend_had,
            "backend_tool_arg_chars": self.backend_tool_arg_chars,
            "proxy_emitted_chat_tool_chunks": self.proxy_emitted_chat_tool_chunks,
            "proxy_emitted_chat_tool_arg_chars": self.proxy_emitted_chat_tool_arg_chars,
            "proxy_response_function_arg_deltas": self.proxy_response_function_arg_deltas,
            "proxy_response_function_arg_chars": self.proxy_response_function_arg_chars,
            "downstream_had_structured_tool_events": down_chat or down_res or synthetic,
            "client_reasoning_chars": self.client_reasoning_chars,
            "client_visible_chars": self.client_visible_chars,
            "content_had_tool_like_markup": c_tool,
            "reasoning_had_tool_like_markup": r_tool,
            "fallback_synthetic_tools": synthetic,
            "fallback_synthetic_count": self.fallback_synthetic_count,
            "sanitizer_eof": self.sanitizer_eof,
            "flags": flags,
        }
