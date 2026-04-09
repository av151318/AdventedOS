"""
Incremental lexer for assistant streams: reasoning/thinking XML vs visible text vs tool-like XML.

Tool execution semantics MUST NOT be derived here; structured ``delta.tool_calls`` is authoritative.
This module only splits and strips text channels for client display.

Inbound client requests (Droid system prompts, ``tools`` schema JSON, user messages) are never passed
through this sanitizer—only assistant ``delta.content`` chunks from the backend SSE path (and related
response post-processing). If tool definitions are missing from logs, check contract truncation settings
before blaming this module.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal


def _sanitizer_audit_limits() -> tuple[int, int]:
    try:
        max_ev = int(os.environ.get("PROXY_SANITIZER_AUDIT_MAX_EVENTS", "120"))
    except ValueError:
        max_ev = 120
    try:
        max_b = int(os.environ.get("PROXY_SANITIZER_AUDIT_MAX_BYTES_PER", "65536"))
    except ValueError:
        max_b = 65536
    return max(0, min(500, max_ev)), max(256, min(262144, max_b))

Mode = Literal["visible", "reasoning", "tool"]


# Open tag names (after '<') for model "thinking" regions — compared case-insensitively.
_REASONING_TAG_NAMES: tuple[str, ...] = (
    "redacted_thinking",
    "think",
    "thinking",
    "reasoning",
)

# Tool-like blobs to strip from visible content only (never promoted to tool_calls).
_TOOL_OPEN_FRAGMENTS: tuple[str, ...] = (
    "invoke name=",
    "invoke ",
    "function=",
    "tool_code",
)

# Closing tags: allow optional whitespace before ``>`` (models often emit ``</tool_call >``).
_FLEX_TOOL_CLOSE_REGEX: dict[str, str] = {
    "</tool_call>": r"</tool_call\s*>",
    "</invoke>": r"</invoke\s*>",
    "</tool_code>": r"</tool_code\s*>",
}

_INCOMPLETE_FLEX_TOOL_TAIL: dict[str, str] = {
    "</tool_call>": r"</tool_call\s*$",
    "</invoke>": r"</invoke\s*$",
    "</tool_code>": r"</tool_code\s*$",
}


@dataclass
class ReasoningXmlSanitizer:
    _buf: str = ""
    _mode: Mode = "visible"
    _reasoning_close_lower: str | None = None  # e.g. "</think>" lowercased without needing full string
    _reasoning_tag: str | None = None
    _tool_close: str | None = None
    _factory_function: bool = False
    _audit_events: list[dict[str, Any]] = field(default_factory=list)

    def _audit_append(self, kind: str, *, pre: str, post: str | None = None, phase: str | None = None) -> None:
        max_ev, max_b = _sanitizer_audit_limits()
        if max_ev <= 0 or len(self._audit_events) >= max_ev:
            return

        def clip(x: str) -> str:
            if len(x) > max_b:
                return x[:max_b] + f"…[truncated {len(x)} bytes]"
            return x

        row: dict[str, Any] = {"kind": kind, "pre_utf8": clip(pre)}
        if post is not None:
            row["post_utf8"] = clip(post)
        if phase:
            row["phase"] = phase
        self._audit_events.append(row)

    def take_audit_events(self) -> list[dict[str, Any]]:
        out = list(self._audit_events)
        self._audit_events.clear()
        return out

    def feed(self, channel: str, chunk: str) -> tuple[str | None, str | None]:
        """Backward-compatible API.

        ``channel == "reasoning"``: pass-through for backend-native reasoning field (not HTML/XML split).
        ``channel == "content"``: run incremental lexer → (reasoning_delta, visible_delta).
        """
        if channel == "reasoning":
            return (chunk if chunk else None, None)
        return self.feed_content(chunk)

    def feed_content(self, chunk: str) -> tuple[str | None, str | None]:
        if not chunk:
            return (None, None)
        self._buf += chunk
        reasoning_parts: list[str] = []
        visible_parts: list[str] = []

        while self._buf:
            if self._mode == "visible":
                consumed = self._scan_visible(reasoning_parts, visible_parts)
                if consumed == 0:
                    break
            elif self._mode == "reasoning":
                consumed = self._scan_reasoning(reasoning_parts)
                if consumed == 0:
                    break
            else:
                consumed = self._scan_tool(visible_parts)
                if consumed == 0:
                    break

        r_out = "".join(reasoning_parts) if reasoning_parts else None
        v_out = "".join(visible_parts) if visible_parts else None
        if (
            os.environ.get("PROXY_SANITIZER_AUDIT_CHUNK_DELTAS", "").strip().lower()
            in ("1", "true", "yes", "on")
            and chunk
        ):
            rp = r_out or ""
            vp = v_out or ""
            if rp != chunk or vp:
                self._audit_append(
                    "content_chunk_split",
                    pre=chunk,
                    post=json.dumps({"reasoning": rp, "visible": vp}, ensure_ascii=False),
                    phase="feed_content",
                )
        return (r_out, v_out)

    def _emit_visible(self, visible_parts: list[str], text: str) -> None:
        if text:
            visible_parts.append(text)

    def _scan_visible(self, reasoning_parts: list[str], visible_parts: list[str]) -> int:
        assert self._mode == "visible"
        low = self._buf.lower()
        lt = low.find("<")
        if lt < 0:
            self._emit_visible(visible_parts, self._buf)
            self._buf = ""
            return 1
        if lt > 0:
            self._emit_visible(visible_parts, self._buf[:lt])
            self._buf = self._buf[lt:]
            return 1

        # self._buf starts with '<'
        if len(self._buf) == 1:
            return 0

        # Not a tag: e.g. "< 5", "<=", "<?"
        ch2 = self._buf[1]
        if ch2 in " \t\n=!?0123456789" or ch2 == "/":
            # '</' might be closing tag in wrong mode — handled elsewhere; here if '</' but we're visible, could be malformed
            if low.startswith("</"):
                gt = low.find(">", 1)
                if gt < 0 and len(self._buf) < 256:
                    return 0
                if gt < 0:
                    self._emit_visible(visible_parts, self._buf[:1])
                    self._buf = self._buf[1:]
                    return 1
            else:
                self._emit_visible(visible_parts, self._buf[:1])
                self._buf = self._buf[1:]
                return 1

        # Reasoning open?
        tried = self._try_consume_reasoning_open()
        if tried == "incomplete":
            return 0
        if tried == "consumed":
            return 1

        # Tool strip open?
        tried_t = self._try_consume_tool_open()
        if tried_t == "incomplete":
            return 0
        if tried_t == "consumed":
            return 1

        # Bare '<' with letters but unknown tag: emit '<' and continue (avoid deadlock)
        self._emit_visible(visible_parts, self._buf[:1])
        self._buf = self._buf[1:]
        return 1

    def _try_consume_reasoning_open(self) -> Literal["incomplete", "consumed", "no"]:
        raw = self._buf
        low = raw.lower()
        if not low.startswith("<"):
            return "no"
        # Need '>'
        gt = raw.find(">", 1)
        if gt < 0:
            # Unclosed — if looks like could still complete reasoning tag, wait
            tail = low[1:50]
            for name in _REASONING_TAG_NAMES:
                full = "<" + name
                if full.startswith(low[: len(full)]) or low.startswith(full[: min(len(low), len(full))]):
                    if len(low) <= len(full) or (len(low) > len(full) and low[len(full)] in " \t\n>/"):
                        return "incomplete"
            if len(self._buf) > 256:
                return "no"
            return "incomplete"

        inner = low[1:gt].strip()
        first_token = inner.split()[0] if inner else ""
        for name in _REASONING_TAG_NAMES:
            if first_token == name:
                self._mode = "reasoning"
                self._reasoning_tag = name
                self._reasoning_close_lower = "</" + name + ">"
                self._buf = raw[gt + 1 :]
                return "consumed"
        return "no"

    def _try_consume_tool_open(self) -> Literal["incomplete", "consumed", "no"]:
        raw = self._buf
        low = raw.lower()
        if not low.startswith("<"):
            return "no"
        gt = raw.find(">", 1)
        if gt < 0:
            if low.startswith("<tool_call") or low.startswith("<tool_call>"):
                return "incomplete"
            if low.startswith("<function"):
                return "incomplete"
            for frag in _TOOL_OPEN_FRAGMENTS:
                pref = "<" + frag[: max(1, len(frag))]
                plen = min(len(low), len(pref))
                if plen and pref[:plen] == low[:plen]:
                    return "incomplete"
            if len(self._buf) > 256:
                return "no"
            return "incomplete"

        inner = low[1:gt].strip()
        first_token = inner.split()[0] if inner else ""
        # Nemotron: outer <tool_call> must close with </tool_call> before inner <function=...> logic.
        if first_token == "tool_call":
            self._mode = "tool"
            self._tool_close = "</tool_call>"
            self._factory_function = False
            self._buf = raw[gt + 1 :]
            return "consumed"

        if first_token.lower().startswith("function="):
            self._mode = "tool"
            self._tool_close = ""  # unused; _factory_function drives _scan_tool
            self._factory_function = True
            self._buf = raw[gt + 1 :]
            return "consumed"

        segment_lower = low[1:gt]
        for frag in _TOOL_OPEN_FRAGMENTS:
            if frag in segment_lower or segment_lower.startswith(frag.rstrip("=")):
                closer = self._pick_tool_close(segment_lower)
                if closer:
                    self._mode = "tool"
                    self._tool_close = closer
                    self._factory_function = False
                    self._buf = raw[gt + 1 :]
                    return "consumed"
        return "no"

    def _pick_tool_close(self, segment_lower: str) -> str | None:
        """Match only unambiguous strip regions; do not use bare 'function' (Nemotron nests ``<function=`` inside tool_call)."""
        sl = segment_lower.lower()
        if "invoke" in sl:
            return "</invoke>"
        if "tool_code" in sl:
            return "</tool_code>"
        return None

    def _scan_reasoning(self, reasoning_parts: list[str]) -> int:
        assert self._mode == "reasoning" and self._reasoning_close_lower is not None
        low = self._buf.lower()
        close = self._reasoning_close_lower
        idx = low.find(close)
        if idx >= 0:
            reasoning_parts.append(self._buf[:idx])
            self._buf = self._buf[idx + len(close) :]
            self._mode = "visible"
            self._reasoning_close_lower = None
            self._reasoning_tag = None
            return 1

        hold = 0
        max_k = min(len(close) - 1, len(self._buf))
        for k in range(max_k, 0, -1):
            if close.startswith(low[-k:]):
                hold = k
                break
        if hold:
            if hold < len(self._buf):
                reasoning_parts.append(self._buf[:-hold])
                self._buf = self._buf[-hold:]
            return 1 if hold < len(self._buf) else 0
        reasoning_parts.append(self._buf)
        self._buf = ""
        return 1

    def _scan_factory_function(self) -> int:
        low = self._buf.lower()
        p_named = -1
        p_generic = low.find("</function>", 0)
        # prefer earliest valid closing tag
        search_hi = min(len(self._buf), 262144)
        segment = low[:search_hi]
        i = 0
        while True:
            j = segment.find("</function=", i)
            if j < 0:
                break
            gt = self._buf.find(">", j)
            if gt < 0:
                return 0
            if p_named < 0 or j < p_named:
                p_named = j
            i = j + 1
        candidates = [p for p in (p_named, p_generic) if p >= 0]
        if not candidates:
            if len(self._buf) > 65536:
                self._audit_append(
                    "factory_function_aborted",
                    pre=self._buf,
                    post="",
                    phase="overflow_reset",
                )
                self._buf = ""
                self._mode = "visible"
                self._factory_function = False
                self._tool_close = None
                return 1
            return 0
        close_start = min(candidates)
        gt = self._buf.find(">", close_start)
        if gt < 0:
            return 0
        consumed_region = self._buf[: gt + 1]
        self._audit_append(
            "factory_function_stripped",
            pre=consumed_region,
            post="",
            phase="close",
        )
        self._buf = self._buf[gt + 1 :]
        self._mode = "visible"
        self._factory_function = False
        self._tool_close = None
        return 1

    def _scan_tool(self, visible_parts: list[str]) -> int:
        if self._factory_function:
            return self._scan_factory_function()
        assert self._mode == "tool" and self._tool_close is not None
        low = self._buf.lower()
        tc = self._tool_close
        flex = _FLEX_TOOL_CLOSE_REGEX.get(tc)
        if flex:
            m = re.search(flex, low)
            if m:
                inner = self._buf[: m.start()]
                self._audit_append(
                    "tool_region_stripped",
                    pre=inner,
                    post="",
                    phase="flex_close",
                )
                self._buf = self._buf[m.end() :]
                self._mode = "visible"
                self._tool_close = None
                self._factory_function = False
                return 1
            inc = _INCOMPLETE_FLEX_TOOL_TAIL.get(tc)
            if inc and re.search(inc, low):
                return 0

        close_l = tc.lower()
        idx = low.find(close_l)
        if idx >= 0:
            inner = self._buf[:idx]
            self._audit_append(
                "tool_region_stripped",
                pre=inner,
                post="",
                phase="literal_close",
            )
            self._buf = self._buf[idx + len(tc) :]
            self._mode = "visible"
            self._tool_close = None
            self._factory_function = False
            return 1

        hold = 0
        max_k = min(len(close_l) - 1, len(self._buf))
        for k in range(max_k, 0, -1):
            if close_l.startswith(low[-k:]):
                hold = k
                break
        if hold == len(self._buf):
            return 0
        if hold:
            self._buf = self._buf[-hold:]
        else:
            self._buf = ""
        return 1

    def drain(self) -> str | None:
        """Flush safe visible tail in ``visible`` mode; drop unclosed tool/reasoning markup tails."""
        if self._mode != "visible":
            return None
        if not self._buf:
            return None
        low = self._buf.lower()
        if self._buf.startswith("<") and ">" not in self._buf:
            if "tool_call" in low or "invoke" in low or "tool_code" in low or "function=" in low:
                self._audit_append(
                    "drain_drop_toolish_partial",
                    pre=self._buf,
                    post="",
                    phase="drain",
                )
                self._buf = ""
                return None
            for name in _REASONING_TAG_NAMES:
                pref = "<" + name
                if low.startswith(pref[: min(len(low), len(pref))]):
                    self._audit_append(
                        "drain_drop_reasoning_partial",
                        pre=self._buf,
                        post="",
                        phase="drain",
                    )
                    self._buf = ""
                    return None
        if self._buf.startswith("<"):
            for name in _REASONING_TAG_NAMES:
                pref = "<" + name
                if low.startswith(pref[: min(len(low), len(pref))]) and len(low) < len(pref) + 2:
                    return None
        out = self._buf
        self._buf = ""
        return out if out.strip() else None

    def finalize(self) -> tuple[str | None, str | None]:
        """End-of-stream flush: unclosed reasoning becomes reasoning tail; visible tail after ``drain()``."""
        reasoning_tail: list[str] = []
        visible_tail: list[str] = []
        if self._mode == "reasoning":
            if self._buf:
                reasoning_tail.append(self._buf)
            self._buf = ""
            self._mode = "visible"
            self._reasoning_close_lower = None
            self._reasoning_tag = None
        elif self._mode == "tool":
            if self._buf:
                self._audit_append(
                    "tool_finalize_drop",
                    pre=self._buf,
                    post=None,
                    phase="eof",
                )
            self._buf = ""
            self._mode = "visible"
            self._tool_close = None
            self._factory_function = False
        dv = self.drain()
        if dv:
            visible_tail.append(dv)
        if self._mode == "visible" and self._buf:
            self._buf = ""
        r = "".join(reasoning_tail) if reasoning_tail else None
        v = "".join(visible_tail) if visible_tail else None
        return (r, v)

    def trace_state(self) -> dict:
        """For structured logging at EOF."""
        return {
            "mode": self._mode,
            "buf_len": len(self._buf),
            "buf_tail": (self._buf[-48:] if self._buf else ""),
            "reasoning_tag": self._reasoning_tag,
            "tool_close": self._tool_close,
            "factory_function": self._factory_function,
            "partial_tag_remainder": bool(self._buf and self._buf.startswith("<")),
        }
