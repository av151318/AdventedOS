"""Per-request tracing for proxy streaming paths."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("proxyapp.trace")

_REQUEST_ID_ATTR = "_proxy_request_id"


# Redact common secret-like keys from logged JSON summaries
def get_or_create_request_id(request=None) -> str:
    if request is None:
        return f"req_{uuid.uuid4().hex[:24]}"
    # Internal call sites may pass a plain dict (e.g. keep-alive); never setattr on that.
    if isinstance(request, dict):
        return f"req_{uuid.uuid4().hex[:24]}"
    cached = getattr(request, _REQUEST_ID_ATTR, None)
    if cached:
        return str(cached)
    if hasattr(request, "headers"):
        h = request.headers
        for key in ("X-Request-Id", "X-Correlation-Id", "OpenAI-Request-Id", "x-request-id"):
            if key in h:
                rid = str(h[key]).strip()[:128]
                setattr(request, _REQUEST_ID_ATTR, rid)
                return rid
    rid = f"req_{uuid.uuid4().hex[:24]}"
    setattr(request, _REQUEST_ID_ATTR, rid)
    return rid


def summarize_body(body: dict, max_len: int = 2000) -> str:
    try:
        s = json.dumps(body, ensure_ascii=False, default=str)
    except Exception:
        s = str(body)[:max_len]
    if len(s) > max_len:
        s = s[:max_len] + "…[truncated]"
    return s


def redact_summary(text: str) -> str:
    import re as _re

    return _re.sub(
        r'("(api[_-]?key|authorization|password|token)"\s*:\s*)"[^"]*(")',
        r'\1[REDACTED]\3',
        text,
        flags=_re.I,
    )


def summarize_chat_request(data: dict) -> dict:
    tools = data.get("tools") or []
    return {
        "model": data.get("model"),
        "stream": data.get("stream"),
        "n_messages": len(data.get("messages") or []),
        "tool_count": len(tools) if isinstance(tools, list) else 0,
        "max_tokens": data.get("max_tokens"),
    }


def summarize_backend_chat_payload(data: dict) -> dict:
    return summarize_chat_request(data)


def summarize_responses_request(body: dict) -> dict:
    inp = body.get("input")
    n_items = len(inp) if isinstance(inp, list) else (1 if inp else 0)
    tools = body.get("tools") or []
    return {
        "model": body.get("model"),
        "stream_implied": True,
        "input_items": n_items,
        "tool_count": len(tools) if isinstance(tools, list) else 0,
        "max_output_tokens": body.get("max_output_tokens"),
    }


def log_inbound_chat(request_id: str, route: str, summary: dict, client_json_utf8_bytes: int) -> None:
    """One line per request; full messages/tools/system live in GET /v1/proxy/audit/*."""
    logger.info(
        "[trace] inbound request_id=%s route=%s summary=%s client_json_utf8_bytes=%s "
        "(full payload: GET /v1/proxy/audit/%s)",
        request_id,
        route,
        summary,
        client_json_utf8_bytes,
        request_id,
    )


def log_inbound_responses(request_id: str, route: str, summary: dict, client_json_utf8_bytes: int) -> None:
    log_inbound_chat(request_id, route, summary, client_json_utf8_bytes)


def trace_stream_chunks_enabled() -> bool:
    return os.environ.get("PROXY_TRACE_STREAM_CHUNKS", "").strip().lower() in ("1", "true", "yes", "on")


def chat_payload_logging_enabled() -> bool:
    return os.environ.get("PROXY_LOG_CHAT_PAYLOADS", "1").strip().lower() not in ("0", "false", "no", "off")


def chat_payload_max_bytes() -> int:
    try:
        return max(1024, min(2 * 1024 * 1024, int(os.environ.get("PROXY_LOG_PAYLOAD_MAX_BYTES", "65536"))))
    except ValueError:
        return 65536


def structured_trace_enabled() -> bool:
    return os.environ.get("PROXY_STRUCTURED_TRACE", "1").strip().lower() not in ("0", "false", "no", "off")


def contract_human_log_enabled() -> bool:
    """Pretty-print full contract payloads to proxy.log (same root handler as proxyapp.trace). Opt out: PROXY_CONTRACT_HUMAN_LOG=0."""
    return os.environ.get("PROXY_CONTRACT_HUMAN_LOG", "1").strip().lower() not in ("0", "false", "no", "off")


def structured_log_path() -> Path:
    raw = (os.environ.get("PROXY_STRUCTURED_LOG", "logs/proxy_trace.jsonl") or "logs/proxy_trace.jsonl").strip()
    p = Path(raw)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def contract_log_ensure_ascii() -> bool:
    """If true (default), contract NDJSON and ``[CONTRACT]`` blocks use JSON ``\\uXXXX`` escapes instead of raw UTF-8 in log files. Set ``PROXY_CONTRACT_JSON_ASCII=0`` for readable non-Latin text in logs."""
    return os.environ.get("PROXY_CONTRACT_JSON_ASCII", "1").strip().lower() not in ("0", "false", "no", "off")


def _append_ndjson(rec: dict[str, Any]) -> None:
    if not structured_trace_enabled():
        return
    line = json.dumps(rec, ensure_ascii=contract_log_ensure_ascii(), default=str, separators=(",", ":"))
    try:
        p = structured_log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def contract_max_string() -> int:
    """Per-string cap inside contract payloads (system prompts + tool JSON). Default high enough for Droid."""
    try:
        return max(4096, min(16 * 1024 * 1024, int(os.environ.get("PROXY_CONTRACT_MAX_STRING", "2000000"))))
    except ValueError:
        return 2_000_000


def contract_max_list_items() -> int:
    try:
        return max(256, min(200_000, int(os.environ.get("PROXY_CONTRACT_MAX_LIST_ITEMS", "50000"))))
    except ValueError:
        return 50_000


def contract_diagnostic_enabled() -> bool:
    """Optional bump above ``PROXY_CONTRACT_MAX_STRING`` / ``PROXY_CONTRACT_MAX_LIST_ITEMS``."""
    return os.environ.get("PROXY_CONTRACT_DIAGNOSTIC", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def contract_bounds() -> tuple[int, int]:
    ms, ml = contract_max_string(), contract_max_list_items()
    if contract_diagnostic_enabled():
        try:
            ms = max(ms, min(32 * 1024 * 1024, int(os.environ.get("PROXY_CONTRACT_DIAGNOSTIC_MAX_STRING", str(ms)))))
        except ValueError:
            pass
        try:
            ml = max(ml, min(200_000, int(os.environ.get("PROXY_CONTRACT_DIAGNOSTIC_MAX_LIST_ITEMS", str(ml)))))
        except ValueError:
            pass
    return ms, ml


def bound_for_contract(obj: Any) -> Any:
    ms, ml = contract_bounds()
    return _trim_strings(redact_nested(obj), ms, ml)


# Per-chunk SSE events:same NDJSON stream, but omit multi-line [CONTRACT] blocks by default (proxy.log noise).
_HUMAN_CONTRACT_BLOCK_SKIP = frozenset(
    {
        "backend_sse_delta",
        "proxy_outbound_delta",
        "proxy_sanitizer_transition",
        "prefill_admission",
    }
)


def contract_human_prints_multiline_block(event: str) -> bool:
    if not contract_human_log_enabled():
        return False
    if os.environ.get("PROXY_CONTRACT_HUMAN_SSE_DELTAS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    return event not in _HUMAN_CONTRACT_BLOCK_SKIP


def canonical_backend_chat_sse(obj: dict[str, Any]) -> dict[str, Any]:
    chs: list[dict[str, Any]] = []
    for i, ch in enumerate((obj.get("choices") or [])[:16]):
        if not isinstance(ch, dict):
            continue
        d = dict(ch.get("delta") or {})
        slim_delta: dict[str, Any] = {}
        for k in ("role", "content", "reasoning", "reasoning_content", "tool_calls"):
            if k in d:
                slim_delta[k] = d.get(k)
        chs.append({"index": ch.get("index", i), "delta": slim_delta, "finish_reason": ch.get("finish_reason")})
    return {
        "id": obj.get("id"),
        "object": obj.get("object"),
        "model": obj.get("model"),
        "choices": chs,
    }


def projection_backend_delta(delta: dict[str, Any], finish_reason: Any) -> dict[str, Any]:
    r = delta.get("reasoning") or delta.get("reasoning_content") or ""
    c = delta.get("content") or ""
    if not isinstance(r, str):
        r = ""
    if not isinstance(c, str):
        c = ""
    tcs = delta.get("tool_calls") or []
    if not isinstance(tcs, list):
        tcs = []
    ta = 0
    for tc in tcs:
        fn = (tc or {}).get("function") or {}
        ta += len(fn.get("arguments") or "")
    return {
        "reasoning_chars": len(r),
        "content_chars": len(c),
        "tool_calls_count": len(tcs),
        "tool_arg_chars": ta,
        "finish_reason": finish_reason,
    }


def projection_chat_outbound_event(event: dict[str, Any]) -> dict[str, Any]:
    ch0 = (event.get("choices") or [{}])[0]
    if not isinstance(ch0, dict):
        return {"reasoning_chars": 0, "content_chars": 0, "tool_calls_count": 0, "tool_arg_chars": 0, "finish_reason": None}
    return projection_backend_delta(ch0.get("delta") or {}, ch0.get("finish_reason"))


def projection_responses_outbound_event(event: dict[str, Any]) -> dict[str, Any]:
    r, c, ta = responses_event_delta_lengths(event)
    return {
        "reasoning_chars": r,
        "content_chars": c,
        "tool_arg_chars": ta,
        "event_type": event.get("type") or "",
        "sequence_number": event.get("sequence_number"),
    }


def _emit_contract_to_proxy_log(rec: dict[str, Any]) -> None:
    """One grep-friendly block per event in proxy.log (not a separate analyzer)."""
    if not contract_human_log_enabled():
        return
    try:
        cap = int(os.environ.get("PROXY_CONTRACT_HUMAN_DUMP_BYTES", "8388608"))
    except ValueError:
        cap = 8_388_608
    cap = max(4096, min(32 * 1024 * 1024, cap))
    header = (
        "[CONTRACT] "
        f"event={rec.get('event')} "
        f"request_id={rec.get('request_id')} "
        f"api={rec.get('api')} "
        f"route={rec.get('route')} "
        f"sequence={rec.get('sequence', '-')}"
    )
    try:
        body = json.dumps(rec, indent=2, ensure_ascii=contract_log_ensure_ascii(), default=str)
    except Exception:
        body = str(rec)
    if len(body) > cap:
        body = body[:cap] + "\n… [truncated for PROXY_CONTRACT_HUMAN_DUMP_BYTES]"
    logger.info("%s\n%s", header, body)


def emit_contract_event(
    event: str,
    request_id: str,
    api: str,
    route: str,
    *,
    sequence: int | None = None,
    **fields: Any,
) -> None:
    rec: dict[str, Any] = {
        "ts": _utc_iso(),
        "logger": "proxyapp.trace",
        "event": event,
        "request_id": request_id,
        "api": api,
        "route": route,
    }
    if sequence is not None:
        rec["sequence"] = sequence
    for k, v in fields.items():
        if v is not None:
            rec[k] = v
    _append_ndjson(rec)
    if contract_human_prints_multiline_block(event):
        _emit_contract_to_proxy_log(rec)


def log_backend_request_payload(request_id: str, api: str, route: str, backend_url: str, payload: dict[str, Any]) -> None:
    emit_contract_event(
        "backend_request_payload",
        request_id,
        api,
        route,
        backend_url=backend_url,
        payload=bound_for_contract(payload),
    )


def log_normalized_client_payload(
    request_id: str,
    api: str,
    route: str,
    normalized_body: dict[str, Any],
    normalization_notes: dict[str, Any] | None = None,
) -> None:
    """Post-``normalize_openai_chat_request`` body (client model id, before backend id remap)."""
    emit_contract_event(
        "normalized_client_payload",
        request_id,
        api,
        route,
        normalization_notes=bound_for_contract(normalization_notes) if normalization_notes else None,
        body=bound_for_contract(normalized_body),
    )


def log_backend_sse_delta(
    request_id: str,
    api: str,
    route: str,
    *,
    sequence: int,
    backend_url: str,
    raw_sse_bytes: int,
    parsed: dict[str, Any],
    projection: dict[str, Any],
) -> None:
    emit_contract_event(
        "backend_sse_delta",
        request_id,
        api,
        route,
        sequence=sequence,
        backend_url=backend_url,
        raw_sse_bytes=raw_sse_bytes,
        parsed=bound_for_contract(parsed),
        projection=projection,
    )


def log_proxy_outbound_delta(
    request_id: str,
    api: str,
    route: str,
    *,
    sequence: int,
    surface: str,
    emitted: dict[str, Any],
    projection: dict[str, Any],
) -> None:
    emit_contract_event(
        "proxy_outbound_delta",
        request_id,
        api,
        route,
        sequence=sequence,
        surface=surface,
        emitted=bound_for_contract(emitted),
        projection=projection,
    )


def log_proxy_sanitizer_transition(
    request_id: str,
    api: str,
    route: str,
    *,
    sequence: int,
    before: dict[str, Any],
    after: dict[str, Any],
    phase: str | None = None,
) -> None:
    emit_contract_event(
        "proxy_sanitizer_transition",
        request_id,
        api,
        route,
        sequence=sequence,
        phase=phase,
        before=bound_for_contract(before),
        after=bound_for_contract(after),
    )


def log_stream_terminal(
    request_id: str,
    *,
    api: str,
    route: str,
    backend_delta_rows: int,
    outbound_delta_rows: int,
    finish_reason: str | None,
    content_chars: int,
    reasoning_chars: int,
    n_tool_calls: int,
    promoted_xml_tools: bool,
    proxy_aggregated_outcome: dict[str, Any] | None = None,
) -> None:
    logger.info(
        "[trace] stream_terminal request_id=%s backend_deltas=%s outbound_deltas=%s finish_reason=%s",
        request_id,
        backend_delta_rows,
        outbound_delta_rows,
        finish_reason,
    )
    emit_contract_event(
        "stream_terminal",
        request_id,
        api,
        route,
        backend_delta_rows=backend_delta_rows,
        outbound_delta_rows=outbound_delta_rows,
        finish_reason=finish_reason,
        content_chars=content_chars,
        reasoning_chars=reasoning_chars,
        n_tool_calls=n_tool_calls,
        promoted_xml_tools=promoted_xml_tools,
        proxy_aggregated_outcome=bound_for_contract(proxy_aggregated_outcome) if proxy_aggregated_outcome else None,
    )


def _trim_strings(obj: Any, max_len: int, max_list: int = 2000, depth: int = 0) -> Any:
    if depth > 24:
        return "[MAX_DEPTH]"
    if isinstance(obj, str):
        return obj if len(obj) <= max_len else obj[:max_len] + "…[truncated]"
    if isinstance(obj, dict):
        return {str(k): _trim_strings(v, max_len, max_list, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        cap = max_list if max_list > 0 else len(obj)
        return [_trim_strings(x, max_len, max_list, depth + 1) for x in obj[:cap]]
    return obj


def sse_trace_line_cap() -> int:
    """Cap per raw SSE line appended to PROXY_SSE_TRACE_PATH."""
    try:
        default = "16777216" if contract_diagnostic_enabled() else "1048576"
        return max(256, min(32 * 1024 * 1024, int(os.environ.get("PROXY_SSE_TRACE_MAX_LINE", default))))
    except ValueError:
        return 16_777_216 if contract_diagnostic_enabled() else 1_048_576


def redact_nested(obj: Any, depth: int = 0) -> Any:
    if depth > 24:
        return obj
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            lk = str(k).lower().replace("-", "_")
            if lk in (
                "authorization",
                "api_key",
                "password",
                "token",
                "openai_api_key",
                "proxyapp_api_key",
            ):
                out[k] = "[REDACTED]"
            else:
                out[k] = redact_nested(v, depth + 1)
        return out
    if isinstance(obj, list):
        return [redact_nested(x, depth + 1) for x in obj]
    if isinstance(obj, str):
        return redact_summary(obj)
    return obj


def log_structured_record(request_id: str, event: str, **data: Any) -> None:
    """Legacy NDJSON row (no api/route). Prefer ``emit_contract_event``."""
    rec: dict[str, Any] = {"v": 1, "ts": _utc_iso(), "logger": "proxyapp.trace", "request_id": request_id, "event": event}
    for k, v in data.items():
        if v is not None:
            rec[k] = v
    if structured_trace_enabled():
        _append_ndjson(rec)
    if rec.get("api") is None:
        rec["api"] = "legacy"
    if rec.get("route") is None:
        rec["route"] = ""
    if contract_human_prints_multiline_block(event):
        _emit_contract_to_proxy_log(rec)


def _json_char_len(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False, default=str))
    except Exception:
        return -1


def log_api_inbound_evidence(request_id: str, route: str, flavor: str, body: dict[str, Any]) -> None:
    """Single INFO line: proves system prompt + tool definitions sizes on the wire (sanitizer does not touch this)."""
    if not isinstance(body, dict):
        return
    tools = body.get("tools")
    n_tools = len(tools) if isinstance(tools, list) else 0
    tools_chars = _json_char_len(tools) if tools is not None else 0
    model = body.get("model")
    stream = body.get("stream")
    n_msg = 0
    n_sys = 0
    sys_chars = 0
    if flavor == "chat":
        n_msg = len(body.get("messages") or [])
        for m in body.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "system":
                n_sys += 1
                sys_chars += _json_char_len(m.get("content"))
    elif flavor == "responses":
        inp = body.get("input")
        if isinstance(inp, list):
            n_msg = len(inp)
            for it in inp:
                if isinstance(it, dict) and it.get("role") == "system":
                    n_sys += 1
                    sys_chars += _json_char_len(it.get("content"))
        elif isinstance(inp, dict):
            n_msg = 1
            if inp.get("role") == "system":
                n_sys = 1
                sys_chars = _json_char_len(inp.get("content"))
        elif inp is not None:
            n_msg = 1
    logger.info(
        "[api_in] request_id=%s route=%s flavor=%s model=%r stream=%s n_messages=%s n_system=%s "
        "system_json_chars=%s tool_defs=%s tools_json_chars=%s body_json_chars=%s",
        request_id,
        route,
        flavor,
        model,
        stream,
        n_msg,
        n_sys,
        sys_chars,
        n_tools,
        tools_chars,
        _json_char_len(body),
    )


def log_chat_json_payload(request_id: str, direction: str, obj: Any) -> None:
    """Emit ``inbound_client_payload`` contract row; optional human summary."""
    if not chat_payload_logging_enabled():
        return
    route_api = {
        "inbound_chat_completions": ("POST /v1/chat/completions", "chat.completions"),
        "inbound_responses": ("POST /v1/responses", "v1/responses"),
    }.get(direction)
    if not route_api:
        return
    route, api = route_api
    meta: dict[str, Any] = {}
    if isinstance(obj, dict):
        meta["model"] = obj.get("model")
        meta["stream"] = obj.get("stream")
        meta["n_messages"] = len(obj.get("messages") or [])
        tools = obj.get("tools")
        meta["tool_count"] = len(tools) if isinstance(tools, list) else 0
        if direction == "inbound_responses":
            inp = obj.get("input")
            meta["input_items"] = len(inp) if isinstance(inp, list) else (1 if inp else 0)
    try:
        raw_len = len(json.dumps(obj, ensure_ascii=False, default=str))
    except Exception:
        raw_len = -1
    logger.info(
        "[trace] payload_summary request_id=%s direction=%s json_chars=%s meta=%s",
        request_id,
        direction,
        raw_len,
        meta,
    )
    emit_contract_event(
        "inbound_client_payload",
        request_id,
        api,
        route,
        json_chars=raw_len,
        meta=meta,
        body=bound_for_contract(obj),
    )


def log_stream_text_preview(request_id: str, label: str, text: str) -> None:
    """Off by default — previews are derived, not wire-contract evidence."""
    if os.environ.get("PROXY_STREAM_TEXT_PREVIEW", "").strip().lower() not in ("1", "true", "yes", "on"):
        return
    if not chat_payload_logging_enabled():
        return
    try:
        cap = int(os.environ.get("PROXY_LOG_STREAM_TEXT_PREVIEW_CHARS", "6000"))
    except ValueError:
        cap = 6000
    if cap <= 0 or not (text or "").strip():
        return
    t = text if len(text) <= cap else text[:cap] + "…[truncated]"
    t = redact_summary(t)
    logger.info("[trace] stream_text_preview request_id=%s label=%s text=%s", request_id, label, t)


def log_chat_stream_summary(
    request_id: str,
    *,
    content_chars: int,
    reasoning_chars: int,
    n_tool_calls: int,
    finish_reason: str | None,
    promoted_xml_tools: bool,
) -> None:
    """Deprecated: use ``log_stream_terminal`` with delta counts."""
    logger.info(
        "[trace] stream_summary request_id=%s client_visible_chars=%s reasoning_chars=%s "
        "n_tool_calls=%s finish_reason=%s promoted_xml_tools=%s",
        request_id,
        content_chars,
        reasoning_chars,
        n_tool_calls,
        finish_reason,
        promoted_xml_tools,
    )


def log_backend_start(request_id: str, backend_url: str, payload_summary: dict) -> None:
    logger.info(
        "[trace] backend_start request_id=%s url=%s payload_summary=%s",
        request_id,
        backend_url,
        payload_summary,
    )


def log_backend_first_byte(request_id: str, elapsed_ms: float, status: int) -> None:
    logger.info(
        "[trace] backend_first_byte request_id=%s elapsed_ms=%.2f status=%s",
        request_id,
        elapsed_ms,
        status,
    )


def log_outbound_event(
    request_id: str,
    api: str,
    event: dict[str, Any],
    *,
    reasoning_len: int = 0,
    content_len: int = 0,
    tool_arg_delta_len: int = 0,
    finish_reason: str | None = None,
) -> None:
    if not trace_stream_chunks_enabled():
        return
    et = event.get("type") or event.get("object") or "unknown"
    seq = event.get("sequence_number", "")
    item_id = event.get("item_id", "")
    out_idx = event.get("output_index", "")
    ci = event.get("content_index", "")
    extra = ""
    if finish_reason is not None:
        extra = f" finish_reason={finish_reason!r}"
    logger.info(
        "[trace] outbound request_id=%s api=%s type=%s seq=%s item_id=%s output_index=%s content_index=%s "
        "reasoning_len=%s content_len=%s tool_arg_delta_len=%s%s",
        request_id,
        api,
        et,
        seq,
        item_id,
        out_idx,
        ci,
        reasoning_len,
        content_len,
        tool_arg_delta_len,
        extra,
    )


def log_sanitizer_eof(request_id: str, state: dict[str, Any]) -> None:
    logger.info("[trace] sanitizer_eof request_id=%s %s", request_id, state)


def log_failure_taxonomy(
    request_id: str,
    route: str,
    failure_class: str,
    *,
    api: str = "proxy",
    **fields: Any,
) -> None:
    """Operator-facing classification: admission vs backend contract vs tool runtime (not HTTP transport)."""
    emit_contract_event(
        "failure_taxonomy",
        request_id,
        api,
        route,
        failure_class=failure_class,
        **fields,
    )


def log_prefill_admission(
    request_id: str,
    route: str,
    *,
    api: str = "v1/chat_completions",
    **fields: Any,
) -> None:
    emit_contract_event(
        "prefill_admission",
        request_id,
        api,
        route,
        **fields,
    )


def log_request_tool_audit(
    request_id: str, record: dict[str, Any], *, api: str, route: str
) -> None:
    s = json.dumps(record, ensure_ascii=contract_log_ensure_ascii(), default=str)
    if len(s) > 8000:
        s = s[:8000] + "…[truncated; full tool_audit in GET /v1/proxy/audit/{id}]"
    logger.info("[trace] request_tool_audit request_id=%s %s", request_id, s)
    emit_contract_event(
        "request_tool_audit",
        request_id,
        api,
        route,
        audit=bound_for_contract(record),
    )


def log_edge_request(request_id: str, method: str, path: str, remote: str, user_agent: str) -> None:
    logger.info(
        "[trace] edge request_id=%s %s %s remote=%s ua=%s",
        request_id,
        method,
        path,
        remote or "-",
        (user_agent or "-")[:300],
    )
    emit_contract_event(
        "edge_request",
        request_id,
        "edge",
        path,
        method=method,
        remote=remote or "-",
        user_agent=(user_agent or "-")[:400],
    )


def log_auth_outcome(request_id: str, path: str, outcome: str) -> None:
    logger.info("[trace] auth request_id=%s path=%s outcome=%s", request_id, path, outcome)


def summarize_chat_completion_response(body: dict) -> dict[str, Any]:
    try:
        ch = (body.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        tc = msg.get("tool_calls") or []
        return {
            "finish_reason": ch.get("finish_reason"),
            "content_len": len(msg.get("content") or ""),
            "reasoning_len": len(msg.get("reasoning_content") or msg.get("reasoning") or ""),
            "n_tool_calls": len(tc) if isinstance(tc, list) else 0,
        }
    except Exception:
        return {"error": "unparseable"}


def log_nonstream_completion(request_id: str, status: int, summary: dict[str, Any]) -> None:
    logger.info("[trace] nonstream_done request_id=%s http_status=%s summary=%s", request_id, status, summary)


def chat_chunk_delta_lengths(event: dict) -> tuple[int, int, int]:
    ch0 = (event.get("choices") or [{}])[0]
    delta = ch0.get("delta") or {}
    r = len(delta.get("reasoning_content") or delta.get("reasoning") or "")
    c = len(delta.get("content") or "")
    ta = 0
    for tc in delta.get("tool_calls") or []:
        fn = tc.get("function") or {}
        ta += len(fn.get("arguments") or "")
    return r, c, ta


def responses_event_delta_lengths(event: dict) -> tuple[int, int, int]:
    et = event.get("type") or ""
    if "reasoning_text.delta" in et:
        return len(event.get("delta") or ""), 0, 0
    if "output_text.delta" in et:
        return 0, len(event.get("delta") or ""), 0
    if "function_call_arguments.delta" in et:
        return 0, 0, len(event.get("delta") or "")
    return 0, 0, 0


def trace_sse_enabled() -> bool:
    return os.environ.get("PROXY_SSE_TRACE", "").strip().lower() in ("1", "true", "yes", "on")


def trace_sse_path() -> str:
    return os.environ.get("PROXY_SSE_TRACE_PATH", "logs/proxy_sse_trace.log").strip() or "logs/proxy_sse_trace.log"


def trace_sse_max_bytes() -> int:
    try:
        return int(os.environ.get("PROXY_SSE_TRACE_MAX_BYTES", "10485760"))
    except ValueError:
        return 10 * 1024 * 1024


class SseTraceWriter:
    """Append-only SSE trace with naive size-based rotation."""

    def __init__(self):
        self._path = trace_sse_path()
        self._max = trace_sse_max_bytes()

    def _maybe_rotate(self) -> None:
        from pathlib import Path

        p = Path(self._path)
        if not p.exists():
            return
        try:
            if p.stat().st_size >= self._max:
                alt = p.with_suffix(".1.log")
                if alt.exists():
                    alt.unlink()
                p.rename(alt)
        except OSError:
            pass

    def write(self, request_id: str, line: str, *, ts: float | None = None) -> None:
        if not trace_sse_enabled():
            return
        from pathlib import Path
        import time as _time

        p = Path(self._path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._maybe_rotate()
        t = ts if ts is not None else _time.time()
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(f"{t:.6f}\t{request_id}\t{line}\n")
        except OSError:
            pass


_sse_writer: SseTraceWriter | None = None


def get_sse_trace_writer() -> SseTraceWriter:
    global _sse_writer
    if _sse_writer is None:
        _sse_writer = SseTraceWriter()
    return _sse_writer
