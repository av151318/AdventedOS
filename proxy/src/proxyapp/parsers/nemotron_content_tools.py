"""
Narrow, documented fallback: Nemotron-style ``<tool_call>...</tool_call>`` in raw assistant text.

Also used when the client sent ``tools`` and ``PROXY_PROMOTE_CONTENT_TOOLS_WHEN_REQUESTED`` is on (default),
or when ``PROXY_CONTENT_TOOL_FALLBACK=1`` with strict OAI tools off.
Does not import vLLM (cf. repo root ``proxy/nemotron_tool_parser.py``).
"""

from __future__ import annotations

import json
import re
import uuid

TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
FUNCTION_NAME_RE = re.compile(r"<function=([^>\s]+)>", re.IGNORECASE)
PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL | re.IGNORECASE)


def _parse_nemotron_tool_block(block: str) -> tuple[str, dict] | None:
    block = block.strip()
    func_match = FUNCTION_NAME_RE.search(block)
    if not func_match:
        return None
    func_name = func_match.group(1).strip()
    args = {}
    for param_match in PARAM_RE.finditer(block):
        param_name = param_match.group(1).strip()
        param_val = param_match.group(2).strip()
        args[param_name] = param_val
    return (func_name, args)


def raw_visible_suggests_tool_calls(raw: str) -> bool:
    if not raw:
        return False
    low = raw.lower()
    return (
        "<tool_call" in low
        or "<invoke" in low
        or "tool_code" in low
        or "<function=" in low
    )


def extract_openai_tool_calls_from_raw(raw: str) -> tuple[list[dict], str]:
    """
    Parse complete ``<tool_call>...</tool_call>`` blocks from raw model content.

    Returns (list of chat delta-shaped tool_call objects with index/id/type/function)
    and visible_prefix (text before first tool_call) for audit only.
    """
    if not raw or "<tool_call" not in raw.lower():
        return [], raw
    low = raw.lower()
    idx = low.find("<tool_call")
    prefix = raw[:idx] if idx >= 0 else raw
    tool_calls: list[dict] = []
    for block in TOOL_CALL_BLOCK_RE.findall(raw):
        parsed = _parse_nemotron_tool_block(block)
        if not parsed:
            continue
        func_name, args = parsed
        i = len(tool_calls)
        tool_calls.append(
            {
                "index": i,
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )
    return tool_calls, prefix


def extract_promotable_tool_calls_from_raw(raw: str) -> tuple[list[dict], str]:
    """Nemotron ``<tool_call>`` first, then Factory/Droid ``<function=`` blocks."""
    t, p = extract_openai_tool_calls_from_raw(raw)
    if t:
        return t, p
    from .factory_content_tools import extract_factory_tool_calls_from_raw

    return extract_factory_tool_calls_from_raw(raw)
