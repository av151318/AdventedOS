"""Explicit product policy for tool execution vs assistant content."""

from __future__ import annotations

import os


def strict_oai_tools() -> bool:
    """Default True: only structured backend tool deltas may produce executable tool streams."""
    v = os.environ.get("PROXY_STRICT_OAI_TOOLS", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def content_tool_fallback_enabled() -> bool:
    v = os.environ.get("PROXY_CONTENT_TOOL_FALLBACK", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def fallback_may_run() -> bool:
    """Nemotron content parser may synthesize tool_calls only when strict mode is off."""
    return (not strict_oai_tools()) and content_tool_fallback_enabled()


def should_try_xml_tool_promotion(openai_chat_request: dict) -> bool:
    """Lift Nemotron ``<tool_call>`` from streamed/raw content when tools were requested.

    Runs legacy ``fallback_may_run()`` first; otherwise (default) promotes XML tools whenever
    the client sent a non-empty ``tools`` list. Disable with PROXY_PROMOTE_CONTENT_TOOLS_WHEN_REQUESTED=0.
    """
    tools = openai_chat_request.get("tools")
    if not tools or not isinstance(tools, list):
        return False
    if fallback_may_run():
        return True
    v = os.environ.get("PROXY_PROMOTE_CONTENT_TOOLS_WHEN_REQUESTED", "1").strip().lower()
    return v not in ("0", "false", "no", "off")
