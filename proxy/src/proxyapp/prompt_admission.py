"""Rough prompt token estimation and first-turn detection for proxy admission.

Estimates are intentionally conservative-ish (char heuristic). They are not a
replacement for the server tokenizer; they exist to reject absurd prompts early
and to mirror vLLM long-prefill classification at the proxy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


def _len_as_tokens(s: str) -> int:
    if not s:
        return 0
    # ~4 UTF-8 chars/token for English-ish text; tools/json skew high, which is OK for a ceiling check.
    return max(1, (len(s) + 3) // 4)


def _content_tokens(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return _len_as_tokens(content)
    if isinstance(content, list):
        t = 0
        for part in content:
            if not isinstance(part, dict):
                t += _len_as_tokens(str(part))
                continue
            typ = part.get("type")
            if typ == "text" and isinstance(part.get("text"), str):
                t += _len_as_tokens(part["text"])
            else:
                t += _len_as_tokens(json.dumps(part, default=str))
        return t
    return _len_as_tokens(str(content))


def estimate_chat_prompt_tokens(messages: Any, tools: Any) -> int:
    total = 0
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                total += _len_as_tokens(str(msg))
                continue
            total += 4  # role / structure overhead
            total += _content_tokens(msg.get("content"))
            for k in ("name", "tool_call_id", "function_call"):
                v = msg.get(k)
                if isinstance(v, str):
                    total += _len_as_tokens(v)
                elif v is not None:
                    total += _len_as_tokens(json.dumps(v, default=str))
            tcs = msg.get("tool_calls")
            if isinstance(tcs, list):
                for tc in tcs:
                    total += _len_as_tokens(json.dumps(tc, default=str))
    if isinstance(tools, list):
        total += _len_as_tokens(json.dumps(tools, default=str))
    elif tools is not None:
        total += _len_as_tokens(json.dumps(tools, default=str))
    return int(total)


def is_first_chat_turn(messages: Any) -> bool:
    if not isinstance(messages, list):
        return True
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            return False
    return True


@dataclass(frozen=True)
class AdmissionThresholds:
    short_lt: int  # [0, short_lt) => short
    medium_lt: int  # [short_lt, medium_lt) => medium
    long_lt: int  # [medium_lt, long_lt] => long; above => extreme


def admission_thresholds_from_env() -> AdmissionThresholds:
    def _i(name: str, default: int) -> int:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return max(1, int(raw))
        except ValueError:
            return default

    return AdmissionThresholds(
        short_lt=_i("PROXY_ADMISSION_SHORT_LT", 8192),
        medium_lt=_i("PROXY_ADMISSION_MEDIUM_LT", 32768),
        long_lt=_i("PROXY_ADMISSION_LONG_LT", 98304),
    )


def classify_admission_class(est: int, th: AdmissionThresholds | None = None) -> str:
    """Label for logging/operators: short | medium | long | extreme."""
    t = th or admission_thresholds_from_env()
    if est < t.short_lt:
        return "short"
    if est < t.medium_lt:
        return "medium"
    if est <= t.long_lt:
        return "long"
    return "extreme"


def admission_class_slot(est: int, th: AdmissionThresholds | None = None) -> str:
    """Semaphore bucket: short | medium | heavy (long+extreme share one cap)."""
    c = classify_admission_class(est, th)
    return "heavy" if c in ("long", "extreme") else c


def admission_slot_limits_from_env() -> dict[str, int]:
    def _i(name: str, default: int) -> int:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return max(1, int(raw))
        except ValueError:
            return default

    return {
        "short": _i("PROXY_ADMISSION_SHORT_MAX", 6),
        "medium": _i("PROXY_ADMISSION_MEDIUM_MAX", 3),
        "heavy": _i("PROXY_ADMISSION_HEAVY_MAX", 1),
    }


def strict_health_required_for_estimate(est: int, th: AdmissionThresholds | None = None) -> bool:
    """Heavy lane (>= medium_lt): require /health == 200, not only /v1/models."""
    t = th or admission_thresholds_from_env()
    return est >= t.medium_lt


def circuit_block_heavy_fail_streak_default() -> int:
    raw = (os.environ.get("PROXY_CIRCUIT_MIN_FAILS") or "").strip()
    if not raw:
        raw = (os.environ.get("PROXY_CIRCUIT_HEAVY_FAIL_STREAK") or "").strip()
    if not raw:
        return 2
    try:
        return max(1, int(raw))
    except ValueError:
        return 2


def circuit_debounce_seconds_default() -> float:
    raw = (os.environ.get("PROXY_CIRCUIT_DEBOUNCE_SECONDS") or "").strip()
    if not raw:
        return 35.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 35.0


def admission_classes_enabled() -> bool:
    return (os.environ.get("PROXY_STRICT_ADMISSION") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
