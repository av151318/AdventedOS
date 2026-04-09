"""Production audit trail: normalization, root-cause codes, Responses tool forwarding."""

from __future__ import annotations

import pytest

from proxyapp.proxy import (
    UnifiedProxy,
    _responses_reasoning_commit_index,
    normalize_openai_chat_request,
)
from proxyapp.tracing import get_or_create_request_id
from proxyapp.sanitizers.reasoning_xml import ReasoningXmlSanitizer
from proxyapp.streaming.production_audit import (
    compute_tool_turn_root_cause,
    normalization_delta,
)


def test_normalization_delta_reports_dropped_keys():
    raw = {"model": "x", "messages": [], "thinking": {"type": "enabled"}, "foo_unknown": 1}
    meta: dict = {}
    norm = normalize_openai_chat_request(raw, meta)
    delta = normalization_delta(raw, norm)
    assert "thinking" in delta["dropped_from_forwarding"]
    assert "foo_unknown" in delta["dropped_from_forwarding"]
    assert "anthropic_stripped" in meta


def test_max_completion_tokens_maps_to_max_tokens():
    raw = {"model": "x", "messages": [], "max_completion_tokens": 4096}
    meta: dict = {}
    norm = normalize_openai_chat_request(raw, meta)
    assert norm.get("max_tokens") == 4096
    assert "max_completion_tokens" not in norm
    assert "max_completion_tokens" not in meta.get("unknown_ignored", [])


def test_max_completion_tokens_overrides_max_tokens():
    raw = {"model": "x", "messages": [], "max_tokens": 100, "max_completion_tokens": 999}
    norm = normalize_openai_chat_request(raw)
    assert norm.get("max_tokens") == 999


def test_compute_root_cause_content_backfill():
    code, _ = compute_tool_turn_root_cause(
        wants_tools=True,
        had_backend_tool_delta=False,
        proxy_emitted_structured_tool=False,
        content_tool_like=True,
        reasoning_tool_like=False,
        responses_endpoint_used=False,
        responses_tools_forwarded=None,
        http_stream_status=200,
        admission_failure_class=None,
    )
    assert code == "content_backfilled_tool_intent_without_structured_tool_call"


def test_compute_root_cause_responses_tools_dropped():
    code, _ = compute_tool_turn_root_cause(
        wants_tools=True,
        had_backend_tool_delta=False,
        proxy_emitted_structured_tool=False,
        content_tool_like=False,
        reasoning_tool_like=False,
        responses_endpoint_used=True,
        responses_tools_forwarded=False,
        http_stream_status=200,
        admission_failure_class=None,
    )
    assert code == "responses_tools_dropped"


def test_map_responses_to_chat_forwards_tools():
    """Regression: /v1/responses must not drop tools/tool_choice before vLLM."""
    p = UnifiedProxy.__new__(UnifiedProxy)
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    req = {
        "model": "cascade-test",
        "input": "hello",
        "stream": True,
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "metadata": {"k": "v"},
        "verbosity": "low",
        "service_tier": "default",
    }
    chat = UnifiedProxy._map_responses_to_chat(p, req)
    assert chat.get("tools") == tools
    assert chat.get("tool_choice") == "auto"
    assert chat.get("parallel_tool_calls") is True
    assert chat.get("metadata") == {"k": "v"}
    assert chat.get("verbosity") == "low"
    assert chat.get("service_tier") == "default"


def test_normalize_forwards_oai_extended_fields():
    raw = {
        "model": "x",
        "messages": [],
        "metadata": {"run": "1"},
        "modalities": ["text"],
        "verbosity": "medium",
        "store": False,
        "prompt_cache_key": "pk-1",
        "safety_identifier": "sid-1",
    }
    norm = normalize_openai_chat_request(raw)
    assert norm.get("metadata") == {"run": "1"}
    assert norm.get("modalities") == ["text"]
    assert norm.get("verbosity") == "medium"
    assert norm.get("store") is False
    assert norm.get("prompt_cache_key") == "pk-1"
    assert norm.get("safety_identifier") == "sid-1"


def test_get_or_create_request_id_accepts_dict():
    rid = get_or_create_request_id({"not": "a request"})
    assert rid.startswith("req_")
    assert len(rid) > 8


def test_responses_reasoning_commit_index_dangling_bracket():
    assert _responses_reasoning_commit_index("hello") == 5
    assert _responses_reasoning_commit_index("hello<") == 5
    assert _responses_reasoning_commit_index("<") == 0


def test_responses_reasoning_commit_index_redacted_unclosed():
    s = "prep<redacted_thinking>inside"
    idx = _responses_reasoning_commit_index(s)
    assert s[:idx] == "prep"
    assert s[idx:].startswith("<redacted_thinking>")


def test_sanitizer_records_tool_strip_audit():
    san = ReasoningXmlSanitizer()
    san.feed_content("<tool_call>x</tool_call> visible")
    frag = san.take_audit_events()
    assert any(x.get("kind") == "tool_region_stripped" for x in frag)
    r, v = san.finalize()
    assert v is None or "visible" in (v or "")
