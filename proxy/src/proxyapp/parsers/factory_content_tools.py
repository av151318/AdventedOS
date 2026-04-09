"""
Factory CLI / Droid-style tool markup in assistant ``content`` (not OpenAI ``tool_calls``).

Parses blocks like::
    <function=TodoWrite>
    <parameter=todos>1. [in_progress] ...</parameter>
    </function=TodoWrite>

Used by ``extract_promotable_tool_calls_from_raw`` when Nemotron ``<tool_call>`` blocks are absent.
"""

from __future__ import annotations

import json
import re
import uuid

PARAM_RE = re.compile(r"<parameter\s*=\s*([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL | re.IGNORECASE)
FACTORY_OPEN_RE = re.compile(r"<\s*function\s*=\s*([^\s>]+)\s*>", re.IGNORECASE)


def extract_factory_tool_calls_from_raw(raw: str) -> tuple[list[dict], str]:
    """Extract OpenAI-shaped ``tool_calls`` from Factory XML; return (calls, visible_prefix)."""
    if not raw or "<function=" not in raw.lower():
        return [], raw
    low_full = raw.lower()
    idx = low_full.find("<function=")
    if idx < 0:
        return [], raw
    prefix = raw[:idx]
    tool_calls: list[dict] = []
    pos = idx
    while pos < len(raw):
        m = FACTORY_OPEN_RE.search(raw, pos)
        if not m:
            break
        name = m.group(1).strip()
        name_l = name.lower()
        start = m.end()
        low = raw.lower()
        candidates = [i for i in (low.find(f"</function={name_l}>", start), low.find(f"</function={name}>", start)) if i >= 0]
        p_generic = low.find("</function>", start)
        p_named = min(candidates) if candidates else -1
        if p_named >= 0 and (p_generic < 0 or p_named <= p_generic):
            close_start = p_named
        elif p_generic >= 0:
            close_start = p_generic
        else:
            break
        gt = raw.find(">", close_start)
        if gt < 0:
            break
        inner = raw[start:close_start]
        end = gt + 1
        args: dict = {}
        for pm in PARAM_RE.finditer(inner):
            args[pm.group(1).strip()] = pm.group(2).strip()
        i = len(tool_calls)
        tool_calls.append(
            {
                "index": i,
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False) if args else "{}",
                },
            }
        )
        pos = end
    return tool_calls, prefix
