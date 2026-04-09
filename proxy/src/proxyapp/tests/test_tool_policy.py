import pytest

from proxyapp.parsers.nemotron_content_tools import extract_openai_tool_calls_from_raw, raw_visible_suggests_tool_calls
from proxyapp.policy import fallback_may_run, should_try_xml_tool_promotion, strict_oai_tools
from proxyapp.streaming.tool_audit import ToolStreamAudit


def test_strict_oai_default(monkeypatch):
    monkeypatch.delenv("PROXY_STRICT_OAI_TOOLS", raising=False)
    assert strict_oai_tools() is True


def test_fallback_only_when_strict_off(monkeypatch):
    monkeypatch.setenv("PROXY_STRICT_OAI_TOOLS", "0")
    monkeypatch.setenv("PROXY_CONTENT_TOOL_FALLBACK", "1")
    assert fallback_may_run() is True
    monkeypatch.setenv("PROXY_STRICT_OAI_TOOLS", "1")
    assert fallback_may_run() is False


def test_extract_nemotron_tool_call():
    raw = (
        "prefix "
        + "<tool_call>\n<function=do_ping>\n<parameter=x>1</parameter>\n</function>\n</tool_call>\n"
        + "suffix"
    )
    tools, pfx = extract_openai_tool_calls_from_raw(raw)
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "do_ping"
    assert "x" in tools[0]["function"]["arguments"]
    assert "prefix" in pfx


def test_raw_suggests_tools():
    assert raw_visible_suggests_tool_calls("<tool_call>") is True
    assert raw_visible_suggests_tool_calls("hello") is False


def test_tool_audit_flags():
    a = ToolStreamAudit(api="v1/chat/completions")
    a.note_raw_content("plain <tool_call>x</tool_call>")
    rec = a.build_record("rid1")
    assert "STRUCTURED_TOOLS_ABSENT" in rec["flags"]
    assert "TOOL_LIKE_PATTERN_IN_RAW_VISIBLE" in rec["flags"]
    assert rec["strict_oai_tools"] is True


def test_fallback_skipped_flag_when_both_env_on(monkeypatch):
    monkeypatch.setenv("PROXY_STRICT_OAI_TOOLS", "1")
    monkeypatch.setenv("PROXY_CONTENT_TOOL_FALLBACK", "1")
    a = ToolStreamAudit(api="v1/chat/completions")
    r = a.build_record("r2")
    assert "fallback_skipped_strict_mode" in r["flags"]


def test_xml_promotion_when_tools_requested_and_strict(monkeypatch):
    monkeypatch.setenv("PROXY_STRICT_OAI_TOOLS", "1")
    monkeypatch.delenv("PROXY_CONTENT_TOOL_FALLBACK", raising=False)
    assert should_try_xml_tool_promotion({"tools": [{"type": "function", "function": {"name": "x"}}]}) is True


def test_xml_promotion_off_when_env_disabled(monkeypatch):
    monkeypatch.setenv("PROXY_STRICT_OAI_TOOLS", "1")
    monkeypatch.setenv("PROXY_PROMOTE_CONTENT_TOOLS_WHEN_REQUESTED", "0")
    assert should_try_xml_tool_promotion({"tools": [{"type": "function"}]}) is False
