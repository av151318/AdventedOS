"""
Unified API Proxy/Gateway
Aggregates models from vLLM and llama.cpp backends
Routes requests to appropriate backend
Manages model lifecycle and request queueing
"""

import asyncio
import logging
import json
import time
import os
import traceback
from collections import defaultdict
from threading import Lock
from pathlib import Path
from typing import Dict, Optional, List, Any, Callable, NamedTuple
from aiohttp import web, ClientSession, ClientError
from aiohttp.web import Request, Response

# Resolve repo root more robustly for both host and container environments
_current_file = Path(__file__).resolve()
# Try container path first (/app), then fall back to host path resolution
if _current_file.parents[1].name == "proxyapp" and _current_file.parents[2].name == "src":
    # We're in the container structure
    REPO_ROOT = Path("/app")
else:
    # We're in the host structure - go up 3 levels from proxyapp/proxy.py
    try:
        REPO_ROOT = _current_file.parents[3]
    except IndexError:
        # Fallback if path resolution fails
        REPO_ROOT = Path.cwd()

from .model_manager import ModelManager, ModelBackend
from .memory_manager import MemoryManager
from .request_queue import RequestQueue
from .chat_history_db import ChatHistoryDB
from .sanitizers.reasoning_xml import ReasoningXmlSanitizer
from .prompt_admission import (
    admission_class_slot,
    admission_classes_enabled,
    admission_slot_limits_from_env,
    circuit_block_heavy_fail_streak_default,
    circuit_debounce_seconds_default,
    classify_admission_class,
    estimate_chat_prompt_tokens,
    is_first_chat_turn,
    strict_health_required_for_estimate,
)
from .policy import should_try_xml_tool_promotion
from .parsers.nemotron_content_tools import (
    extract_promotable_tool_calls_from_raw,
    raw_visible_suggests_tool_calls,
)
from .streaming.committed_ledger import (
    StreamCommittedLedger,
    extract_idempotency_key,
    is_retry_eligible,
)
from .streaming.crash_classify import (
    classify_stream_termination,
    should_trip_admission_on_classify,
    stream_idle_timeout_seconds,
)
from .streaming.stream_chunk_iter import StreamIdleTimeoutError, iter_chunked_with_idle
from .streaming.stream_io import sse_write
from .streaming.tool_audit import ToolStreamAudit
from .streaming.production_audit import (
    ProductionAuditSession,
    compute_tool_turn_root_cause,
    normalization_delta,
    sse_audit_comment_enabled,
)
from .tracing import (
    get_or_create_request_id,
    log_backend_request_payload,
    log_backend_sse_delta,
    log_failure_taxonomy,
    log_prefill_admission,
    log_proxy_outbound_delta,
    log_proxy_sanitizer_transition,
    log_request_tool_audit,
    log_sanitizer_eof,
    log_chat_json_payload,
    log_api_inbound_evidence,
    log_normalized_client_payload,
    canonical_backend_chat_sse,
    projection_backend_delta,
    projection_chat_outbound_event,
    projection_responses_outbound_event,
    sse_trace_line_cap,
    get_sse_trace_writer,
)

logger = logging.getLogger(__name__)


class _ReadinessProbeResult(NamedTuple):
    ok: bool
    probe_class: str
    health_status: Optional[int]
    health_snip: str
    models_status: Optional[int]
    models_snip: str


def normalize_openai_chat_request(body: dict, audit_meta: Optional[dict] = None) -> dict:
    """Normalize OpenAI chat completion requests to handle extra fields gracefully.

    This function ensures backward compatibility while allowing modern clients
    to send additional fields like tools, tool_choice, etc.
    
    Also strips Anthropic-specific parameters that OMO may send even when using OpenAI provider.

    When ``audit_meta`` is a dict, normalization drops are recorded under keys
    ``ignorable_dropped``, ``anthropic_stripped``, and ``unknown_ignored``.
    """
    # Fields we know and support (OpenAI Chat Completions; forward for Droid/OAI-compatible clients).
    # Backends may ignore keys they do not implement; dropping here hides real client intent in logs + vLLM.
    known_fields = {
        'model', 'messages', 'temperature', 'max_tokens', 'top_p', 'n',
        'stream', 'stop', 'presence_penalty', 'frequency_penalty',
        'logit_bias', 'user', 'functions', 'function_call',
        'tools', 'tool_choice', 'stream_options',
        'response_format', 'seed', 'logprobs', 'top_logprobs', 'parallel_tool_calls',
        'reasoning_effort', 'chat_template_kwargs',
        'metadata',
        'modalities', 'verbosity', 'service_tier', 'store',
        'prediction', 'web_search_options', 'audio',
        'prompt_cache_key', 'safety_identifier',
    }

    ignorable_fields: set = set()

    # Anthropic-specific parameters that OMO may send (strip these)
    anthropic_params = {
        'thinking', 'extended_thinking', 'budget_tokens',
        'thinking_blocks', 'interleaved_thinking', 'thinking_budget',
        'max_tokens_to_sample'  # Anthropic uses this instead of max_tokens
    }

    normalized = {}

    for key, value in body.items():
        if key in known_fields:
            normalized[key] = value
        elif key == "max_completion_tokens":
            # Applied after the loop (maps to max_tokens for OpenAI-compatible backends).
            continue
        elif key in ignorable_fields:
            logger.debug(f"normalize_openai_chat_request: ignoring field '{key}' = {value}")
            if audit_meta is not None:
                audit_meta.setdefault("ignorable_dropped", []).append(key)
        elif key in anthropic_params:
            logger.info(f"normalize_openai_chat_request: stripping Anthropic param '{key}' = {value}")
            if audit_meta is not None:
                audit_meta.setdefault("anthropic_stripped", []).append(key)
            # Don't include in normalized request
        else:
            # Unknown field - log and ignore to be safe
            logger.info(f"normalize_openai_chat_request: unknown field '{key}' ignored for compatibility")
            if audit_meta is not None:
                audit_meta.setdefault("unknown_ignored", []).append(key)

    if "max_completion_tokens" in body and body.get("max_completion_tokens") is not None:
        mct = body["max_completion_tokens"]
        try:
            normalized["max_tokens"] = int(mct)
        except (TypeError, ValueError):
            normalized["max_tokens"] = mct

    return normalized


def normalize_vllm_chat_completion_response(body: dict) -> dict:
    """If the assistant message has empty ``content`` but reasoning text, copy reasoning into ``content``."""
    out = json.loads(json.dumps(body))
    for ch in out.get("choices") or []:
        if not isinstance(ch, dict):
            continue
        msg = ch.get("message")
        if not isinstance(msg, dict):
            continue
        c = msg.get("content")
        rc = msg.get("reasoning_content")
        if rc is None:
            rc = msg.get("reasoning")
        if c is None or (isinstance(c, str) and c == ""):
            if isinstance(rc, str) and rc:
                msg["content"] = rc
    return out


def maybe_promote_nemotron_tools_in_completion(body: dict, openai_request: dict) -> dict:
    """Non-stream chat completion: promote Nemotron/Factory tool markup from ``content`` into ``tool_calls``."""
    out = json.loads(json.dumps(body))
    if not should_try_xml_tool_promotion(openai_request):
        return out
    choices = out.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return out
    ch = choices[0]
    msg = ch.get("message")
    if not isinstance(msg, dict):
        return out
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        return out
    tool_calls, prefix = extract_promotable_tool_calls_from_raw(content)
    if not tool_calls:
        return out
    msg["tool_calls"] = tool_calls
    stripped = (prefix or "").strip()
    msg["content"] = stripped if stripped else None
    ch["finish_reason"] = "tool_calls"
    return out


def _responses_reasoning_commit_index(buf: str) -> int:
    """Longest safe prefix length for ``response.reasoning_text.delta`` emission.

    Suffix ``buf[idx:]`` may be an incomplete tag, enclosure, or tool-like fragment that
    should wait for more streamed bytes (or EOF handling).
    """
    if not buf:
        return 0
    low = buf.lower()
    n = len(buf)

    o_tag = "<redacted_thinking>"
    c_tag = "</redacted_thinking>"
    if low.count(o_tag) > low.count(c_tag):
        ri = low.rfind(o_tag)
        if ri != -1:
            return ri

    last_lt = buf.rfind("<")
    if last_lt != -1:
        seg = buf[last_lt:]
        if ">" not in seg:
            return last_lt

    rclose = buf.rfind("</")
    if rclose != -1 and ">" not in buf[rclose:]:
        return rclose

    tail = buf[-400:] if n > 400 else buf
    tlow = tail.lower()
    t_off = n - len(tail)
    needles = (
        "<tool_call",
        "</tool_call",
        "<invoke",
        "</invoke",
        "<tool_code",
        "</tool_code",
        "<think",
        "</think",
        "<reasoning",
        "</reasoning",
        "<function_calls",
        "<redacted_thinking",
    )
    best = n
    for nd in needles:
        p = tlow.rfind(nd)
        if p == -1:
            continue
        abs_p = t_off + p
        frag = buf[abs_p:]
        lim = min(500, len(frag))
        if ">" not in frag[:lim]:
            best = min(best, abs_p)
    if best < n:
        return best
    return n


_LIST_MODELS_LOG_INTERVAL_S = 180.0


class UnifiedProxy:
    """Unified API proxy for vLLM and llama.cpp models"""
    
    def __init__(
        self,
        proxy_port: int = 52415,
        memory_threshold: float = 0.90,
        db_path: Optional[str] = None,
        docker_cmd: str = "docker",
        config_path: Optional[str] = None
    ):
        """
        Initialize unified proxy
        
        Args:
            proxy_port: Port for proxy server
            memory_threshold: GPU memory threshold (default: 0.90)
            db_path: Path to chat history database
            docker_cmd: Docker command to use (default: "docker", can be "sudo docker")
            config_path: Path to models_config.yaml file
        """
        self.proxy_port = proxy_port
        self.model_manager = ModelManager(docker_cmd=docker_cmd, config_path=config_path)
        # CRITICAL: Ensure models are initialized immediately
        if not self.model_manager.models:
            logger.warning("No models found in ModelManager - initializing defaults")
            self.model_manager._init_default_models()
        logger.info(f"Proxy initialized with {len(self.model_manager.models)} models: {list(self.model_manager.models.keys())}")
        # Track in-flight requests per model so the memory evictor doesn't unload an active model.
        self._active_model_requests = defaultdict(int)  # model_id -> count
        self._readiness_locks: Dict[str, Lock] = defaultdict(Lock)
        self._engine_recreate_locks: Dict[str, Lock] = defaultdict(Lock)
        self._engine_recreate_task: Dict[str, asyncio.Task] = {}
        self._pinned_models = self._load_pinned_models(config_path) if config_path else {"qwen3-14b"}

        # Memory monitoring can evict unused models when RAM pressure is high.
        # This avoids system-wide thrash that can cause long-latency or timeouts in n8n Chat Hub.
        self.memory_manager = MemoryManager(
            threshold_percent=memory_threshold * 100,
            evict_callback=self._evict_model_if_idle,
            min_idle_seconds=120,
        )
        self.request_queue = RequestQueue()
        self._vllm_admission_sems: Dict[str, asyncio.Semaphore] = {}
        self._vllm_long_prefill_sems: Dict[str, asyncio.Semaphore] = {}
        self._long_prefill_active: Dict[str, int] = defaultdict(int)
        self._vllm_class_sems: Dict[str, Dict[str, asyncio.Semaphore]] = {}
        # Set default db_path if not provided
        if db_path is None:
            db_path = "data/history.db"
        self.chat_history = ChatHistoryDB(db_path=db_path)
        
        # Start background tasks
        self._monitoring_task = None
        self._queue_processing_task = None
        self._keep_alive_task = None

        # Request tracking for watchdog monitoring
        self.active_requests = {}  # request_id -> start_time
        self.request_counter = 0
        # Rate-limit noisy /v1/models DIAG INFO logs (see list_models).
        self._list_models_log_deadline: float = 0.0

    def _evict_model_if_idle(self, model_id: str) -> bool:
        """Evict a model only if it has no in-flight requests."""
        try:
            if model_id in self._pinned_models:
                logger.info(f"[MEM] Skip eviction: {model_id} is pinned")
                return False
            if self._active_model_requests.get(model_id, 0) > 0:
                logger.warning(f"[MEM] Skip eviction: {model_id} has active requests")
                return False
        except Exception:
            # If we can't determine activity, prefer safety (don't evict).
            return False
        return self.model_manager.unload_model(model_id)

    def _load_pinned_models(self, config_path: str) -> set:
        """Load pinned model IDs from the models config.

        We pin `preload: true` models so they stay warm; n8n Chat Hub has a ~60s request timeout,
        and vLLM cold starts can exceed that.
        """
        pinned = set()
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            for m in cfg.get("models", []) or []:
                if m.get("preload") is True and m.get("model_id"):
                    pinned.add(m["model_id"])
        except Exception as e:
            logger.warning(f"[MEM] Could not load pinned models from config: {e}")
        # Always pin Qwen3-14B for Chat Hub usability
        pinned.add("qwen3-14b")
        return pinned

    def _get_max_num_seqs(self, model_id: str) -> int:
        model_info = self.model_manager.models.get(model_id)
        if not model_info or not getattr(model_info, "config_path", None):
            return 8
        try:
            import yaml
            with open(model_info.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return int(cfg.get("max_num_seqs", 8))
        except Exception:
            return 8

    def _get_vllm_concurrency_limit(self, model_id: str) -> int:
        """Concurrent vLLM chats to allow (default ``max_num_seqs - headroom``). CLI harness parallel calls should block here instead of overfilling vLLM."""
        seq = self._get_max_num_seqs(model_id)
        try:
            headroom = int(os.environ.get("PROXY_VLLM_SEQ_HEADROOM", "1"))
        except ValueError:
            headroom = 1
        headroom = max(0, headroom)
        lim = max(1, seq - headroom)
        raw = (os.environ.get("PROXY_VLLM_MAX_CONCURRENT") or "").strip()
        if raw:
            try:
                override = int(raw)
                lim = max(1, min(override, seq))
            except ValueError:
                pass
        return lim

    def _ensure_vllm_admission_sem(self, model_id: str) -> asyncio.Semaphore:
        if model_id not in self._vllm_admission_sems:
            n = self._get_vllm_concurrency_limit(model_id)
            self._vllm_admission_sems[model_id] = asyncio.Semaphore(n)
            logger.info(
                "[CAP] vLLM admission model=%s max_concurrent=%s (max_num_seqs=%s); override PROXY_VLLM_MAX_CONCURRENT / PROXY_VLLM_SEQ_HEADROOM",
                model_id,
                n,
                self._get_max_num_seqs(model_id),
            )
        return self._vllm_admission_sems[model_id]

    def _resolve_model_config_path(self, model_info) -> Optional[Path]:
        raw = getattr(model_info, "config_path", None)
        if not raw:
            return None
        p = Path(raw)
        if p.is_absolute():
            return p
        root = Path(os.environ.get("EXO_WORKSPACE", REPO_ROOT))
        return root / p

    def _get_vllm_prefill_yaml(self, model_id: str) -> Dict[str, Any]:
        model_info = self.model_manager.models.get(model_id)
        if not model_info:
            return {}
        path = self._resolve_model_config_path(model_info)
        if not path or not path.exists():
            return {}
        try:
            import yaml

            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.debug("prefill yaml read %s: %s", path, e)
            return {}

    def _prompt_completion_reserve(self, data: Dict[str, Any]) -> int:
        mt = data.get("max_tokens")
        if mt is None:
            mt = data.get("max_completion_tokens")
        if mt is not None:
            return max(1, int(mt))
        try:
            return max(256, int(os.environ.get("PROXY_PROMPT_COMPLETION_RESERVE_DEFAULT", "8192")))
        except ValueError:
            return 8192

    def _prompt_budget_margin(self) -> int:
        try:
            return max(0, int(os.environ.get("PROXY_PROMPT_BUDGET_MARGIN", "2048")))
        except ValueError:
            return 2048

    def _long_prefill_threshold_tokens(self, model_id: str) -> int:
        raw = (os.environ.get("PROXY_LONG_PREFILL_THRESHOLD") or "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
        cfg = self._get_vllm_prefill_yaml(model_id)
        return max(1, int(cfg.get("long_prefill_token_threshold", 32768)))

    def _ensure_long_prefill_sem(self, model_id: str) -> Optional[asyncio.Semaphore]:
        cfg = self._get_vllm_prefill_yaml(model_id)
        raw = cfg.get("max_long_partial_prefills", cfg.get("max_long_prefills"))
        if raw is not None:
            n = int(raw)
        elif cfg.get("enable_chunked_prefill"):
            try:
                n = int(os.environ.get("PROXY_LONG_PREFILL_MAX_SLOTS", "2"))
            except ValueError:
                n = 2
        else:
            n = 0
        if n <= 0:
            return None
        if model_id not in self._vllm_long_prefill_sems:
            self._vllm_long_prefill_sems[model_id] = asyncio.Semaphore(n)
            logger.info(
                "[CAP] long-prefill model=%s concurrent_slots=%s "
                "(yaml max_long_* or PROXY_LONG_PREFILL_MAX_SLOTS with enable_chunked_prefill)",
                model_id,
                n,
            )
        return self._vllm_long_prefill_sems[model_id]

    def _ensure_vllm_class_sem(self, model_id: str, slot: str) -> asyncio.Semaphore:
        """short | medium | heavy concurrent caps (production admission shaping)."""
        if model_id not in self._vllm_class_sems:
            limits = admission_slot_limits_from_env()
            self._vllm_class_sems[model_id] = {
                "short": asyncio.Semaphore(limits["short"]),
                "medium": asyncio.Semaphore(limits["medium"]),
                "heavy": asyncio.Semaphore(limits["heavy"]),
            }
            logger.info(
                "[CAP] admission classes model=%s short_max=%s medium_max=%s heavy_max=%s "
                "(PROXY_ADMISSION_{SHORT,MEDIUM,HEAVY}_MAX; disable PROXY_STRICT_ADMISSION=0)",
                model_id,
                limits["short"],
                limits["medium"],
                limits["heavy"],
            )
        return self._vllm_class_sems[model_id][slot]

    def _probe_backend_readiness(self, model_info) -> _ReadinessProbeResult:
        import requests

        port = model_info.port

        def _snip(txt: Optional[str]) -> str:
            if not txt:
                return ""
            t = txt.replace("\n", " ").strip()
            return t[:400]

        health_status: Optional[int] = None
        health_snip = ""
        models_status: Optional[int] = None
        models_snip = ""

        for url, label in (
            (f"http://127.0.0.1:{port}/health", "health"),
            (f"http://127.0.0.1:{port}/v1/models", "models"),
        ):
            try:
                r = requests.get(url, timeout=1.5)
                code = r.status_code
                sn = _snip(r.text)
                if label == "health":
                    health_status, health_snip = code, sn
                else:
                    models_status, models_snip = code, sn
            except requests.exceptions.ConnectionError as e:
                msg = _snip(str(e))
                if label == "health":
                    health_status, health_snip = None, f"connection_error:{msg}"
                else:
                    models_status, models_snip = None, f"connection_error:{msg}"
            except Exception as e:
                msg = _snip(str(e))
                if label == "health":
                    health_status, health_snip = None, msg
                else:
                    models_status, models_snip = None, msg

        models_ok = models_status == 200
        health_ok = health_status == 200
        if models_ok:
            cls = "ready" if health_ok else "models_ok_health_non_200"
            return _ReadinessProbeResult(True, cls, health_status, health_snip, models_status, models_snip)

        parts: list[str] = []
        if models_status is None:
            parts.append("models_unreachable")
        else:
            parts.append(f"models_http_{models_status}")
        if health_status is None:
            parts.append("health_unreachable")
        elif not health_ok:
            parts.append(f"health_http_{health_status}")
        return _ReadinessProbeResult(False, "|".join(parts), health_status, health_snip, models_status, models_snip)

    def _apply_readiness_probe(self, model_id: str, model_info, result: _ReadinessProbeResult) -> bool:
        now = time.time()
        bucket = 3.0
        with self._readiness_locks[model_id]:
            if result.ok:
                model_info.consecutive_readiness_failures = 0
                model_info.first_not_ready_at = 0.0
                model_info.readiness_fail_bucket_ts = 0.0
                model_info.last_ready_at = now
                model_info.last_readiness_probe_class = result.probe_class
                if model_info.status == "loaded":
                    model_info.readiness_ever_served = True
                return True

            if model_info.first_not_ready_at <= 0:
                model_info.first_not_ready_at = now
            if now - model_info.readiness_fail_bucket_ts >= bucket:
                model_info.consecutive_readiness_failures += 1
                model_info.readiness_fail_bucket_ts = now
            model_info.last_readiness_probe_class = result.probe_class
            if (
                model_info.backend == ModelBackend.VLLM
                and model_info.readiness_ever_served
                and model_info.consecutive_readiness_failures
                >= circuit_block_heavy_fail_streak_default()
            ):
                model_info.heavy_circuit_tripped_at = now
                model_info.heavy_circuit_recovery_probes = 0
            return False

    def _heavy_circuit_blocks_heavy_admission(self, model_info) -> bool:
        if model_info.backend != ModelBackend.VLLM:
            return False
        t = float(getattr(model_info, "heavy_circuit_tripped_at", 0.0) or 0.0)
        if t <= 0:
            return False
        now = time.time()
        debounce = circuit_debounce_seconds_default()
        if now < t + debounce:
            return True
        probes = int(getattr(model_info, "heavy_circuit_recovery_probes", 0))
        return probes < 2

    def _trip_heavy_admission_circuit(self, model_id: str) -> None:
        model_info = self.model_manager.models.get(model_id)
        if not model_info or model_info.backend != ModelBackend.VLLM:
            return
        if not getattr(model_info, "readiness_ever_served", False):
            return
        model_info.heavy_circuit_tripped_at = time.time()
        model_info.heavy_circuit_recovery_probes = 0
        logger.warning(
            "[ADMISSION] heavy circuit tripped model=%s (stream/upstream fault)",
            model_id,
        )

    async def _emit_abnormal_stream_terminal(
        self,
        stream_response: web.StreamResponse,
        ledger: StreamCommittedLedger,
        term: str,
        *,
        request_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        if term == "client_disconnect":
            return
        ledger.terminal_classifier = term
        if request_json is not None:
            logger.info(
                "[stream_terminal] request_id=%s term=%s ledger=%s retry_eligible_read_only=%s",
                ledger.request_id,
                term,
                ledger.to_audit_dict(),
                is_retry_eligible(ledger, request_json),
            )
        else:
            logger.info(
                "[stream_terminal] request_id=%s term=%s ledger=%s",
                ledger.request_id,
                term,
                ledger.to_audit_dict(),
            )
        lt = ledger.last_tool_event
        if lt and lt.get("partial") is False:
            payload = {
                "type": "response.failed",
                "code": "tool_boundary_crossed",
                "last_tool_event": {
                    "index": int(lt.get("index", 0)),
                    "partial": False,
                },
            }
            line = (
                "event: response.failed\ndata: "
                + json.dumps(payload, ensure_ascii=False)
                + "\n\n"
            )
            await sse_write(stream_response, line.encode("utf-8"))
            return
        code = (
            term
            if term in ("upstream_reset", "upstream_stall", "upstream_eof")
            else "upstream_eof"
        )
        payload = {
            "type": "response.incomplete",
            "code": code,
            "committed_text_offset": ledger.committed_text_offset,
            "committed_reasoning_offset": ledger.committed_reasoning_offset,
        }
        line = (
            "event: response.incomplete\ndata: "
            + json.dumps(payload, ensure_ascii=False)
            + "\n\n"
        )
        await sse_write(stream_response, line.encode("utf-8"))

    def _readiness_log_failure(self, model_id: str, model_info, result: _ReadinessProbeResult) -> None:
        logger.warning(
            "[READINESS] model=%s ok=False class=%s health=(%s,%r) models=(%s,%r) streak=%s first_not_ready=%s last_ready=%s",
            model_id,
            result.probe_class,
            result.health_status,
            result.health_snip[:200],
            result.models_status,
            result.models_snip[:200],
            model_info.consecutive_readiness_failures,
            model_info.first_not_ready_at,
            model_info.last_ready_at,
        )

    def _refresh_readiness(
        self, model_id: str, model_info, *, require_full_health: bool = False
    ) -> tuple[bool, Optional[str]]:
        """Return (ok, deny_reason). ``heavy_requires_health_200`` when long-context admission needs
        ``/health`` 200 but only models list is healthy.
        """
        if model_info.backend == ModelBackend.LLAMACPP:
            import requests

            try:
                r = requests.get(f"http://127.0.0.1:{model_info.port}/v1/models", timeout=1.5)
                ok = r.status_code == 200
                faux = _ReadinessProbeResult(
                    ok,
                    "llamacpp_models_200" if ok else f"llamacpp_models_http_{r.status_code}",
                    None,
                    "",
                    r.status_code,
                    (r.text or "").replace("\n", " ")[:400],
                )
                res = self._apply_readiness_probe(model_id, model_info, faux)
                if not res:
                    self._readiness_log_failure(model_id, model_info, faux)
                return (res, None)
            except Exception as e:
                faux = _ReadinessProbeResult(False, "llamacpp_unreachable", None, str(e)[:200], None, "")
                self._apply_readiness_probe(model_id, model_info, faux)
                self._readiness_log_failure(model_id, model_info, faux)
                return (False, None)

        result = self._probe_backend_readiness(model_info)
        if require_full_health:
            if result.models_status != 200 or result.models_status is None:
                ok = self._apply_readiness_probe(model_id, model_info, result)
                if not ok:
                    self._readiness_log_failure(model_id, model_info, result)
                return (ok, None)
            if result.health_status != 200:
                logger.warning(
                    "[READINESS] model=%s heavy admission blocked: /v1/models ok but /health=%s snippet=%r",
                    model_id,
                    result.health_status,
                    (result.health_snip or "")[:200],
                )
                return (False, "heavy_requires_health_200")

        ok = self._apply_readiness_probe(model_id, model_info, result)
        if not ok:
            self._readiness_log_failure(model_id, model_info, result)
        elif result.probe_class == "models_ok_health_non_200":
            logger.warning(
                "[READINESS] model=%s serving /v1/models but /health != 200 (health=%s snippet=%r)",
                model_id,
                result.health_status,
                result.health_snip[:200],
            )
        return (ok, None)

    def _emit_admission_failure_trace(
        self,
        request: Optional[Request],
        route: str,
        *,
        failure_class: str,
        model_id: str,
        admission_kind: str,
        model_info: Any,
        container_running: bool,
    ) -> None:
        rid = get_or_create_request_id(request)
        log_failure_taxonomy(
            rid,
            route,
            failure_class,
            api="v1/chat_completions",
            model_id=model_id,
            admission_kind=admission_kind,
            container_running=container_running,
            model_status=model_info.status,
            consecutive_readiness_failures=model_info.consecutive_readiness_failures,
            last_readiness_probe_class=model_info.last_readiness_probe_class,
            last_ready_at=model_info.last_ready_at,
            first_not_ready_at=model_info.first_not_ready_at,
        )

    def _json_model_not_ready(
        self,
        model_id: str,
        model_info,
        *,
        container_running: bool,
        degraded: bool,
    ) -> Dict[str, Any]:
        if degraded:
            return {
                "error": {
                    "message": f"Model {model_id} engine is unhealthy (lost readiness after it was serving).",
                    "type": "engine_unhealthy",
                    "code": "engine_unhealthy",
                },
                "status": "degraded",
                "model_id": model_id,
                "message": "Backend container is running but readiness checks fail. See proxy logs for /health and /v1/models details.",
                "readiness": {
                    "last_ready_at": model_info.last_ready_at,
                    "first_not_ready_at": model_info.first_not_ready_at,
                    "consecutive_readiness_failures": model_info.consecutive_readiness_failures,
                    "last_probe_class": model_info.last_readiness_probe_class,
                },
                "output": [],
                "choices": [],
            }
        return {
            "error": {
                "message": f"Model {model_id} is still initializing. This typically takes 1-2 minutes for first load.",
                "type": "model_loading",
                "code": "model_not_ready",
            },
            "status": "loading",
            "model_id": model_id,
            "message": "Model container is running but not ready yet. Please wait and retry in 30 seconds.",
            "readiness": {
                "last_probe_class": model_info.last_readiness_probe_class,
            },
            "output": [],
            "choices": [],
        }

    def _sanitize_assistant_message_inplace(self, msg: Dict[str, Any]) -> None:
        if not isinstance(msg, dict):
            return
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            return
        san = ReasoningXmlSanitizer()
        r1, v1 = san.feed_content(content)
        r2, v2 = san.finalize()
        reasoning_chunks = [x for x in (r1, r2) if x]
        vis_chunks = [x for x in (v1, v2) if x]
        if reasoning_chunks:
            prev = msg.get("reasoning_content")
            extra = "".join(reasoning_chunks)
            if isinstance(prev, str) and prev:
                msg["reasoning_content"] = prev + extra
            else:
                msg["reasoning_content"] = extra
        if vis_chunks:
            msg["content"] = "".join(vis_chunks)
        else:
            msg["content"] = ""

    def _note_tool_runtime_from_messages(self, request: Optional[Request], messages: Any, route: str) -> None:
        if not isinstance(messages, list) or not messages:
            return
        last = messages[-1]
        if not isinstance(last, dict) or last.get("role") != "tool":
            return
        content = last.get("content")
        if not isinstance(content, str):
            return
        c = content.strip()
        if not c:
            return
        low = c.lower()
        if "error" not in low[:500] and not c.startswith("Error"):
            return
        log_failure_taxonomy(
            get_or_create_request_id(request),
            route,
            "tool_runtime_error",
            api="v1/chat_completions",
            tool_message_snip=c[:300],
        )

    async def _finalize_chat_stream_production_audit(
        self,
        *,
        pa: ProductionAuditSession,
        tool_audit: ToolStreamAudit,
        sanitizer: ReasoningXmlSanitizer,
        wants_tools: bool,
        had_backend_tool_delta: bool,
        http_status: int,
        stream_response: web.StreamResponse,
        route: str,
        log_api: str = "v1/chat_completions",
    ) -> None:
        rid = pa.request_id
        pa.merge_sanitizer_audit(sanitizer.take_audit_events())
        pa.had_backend_tool_delta = had_backend_tool_delta
        pa.http_final_stream_status = http_status
        rec_tool = tool_audit.build_record(
            rid,
            content_had_tool_like_markup=pa.content_had_tool_like_markup,
            reasoning_had_tool_like_markup=pa.reasoning_had_tool_like_markup,
        )
        pa.tool_audit_snapshot = rec_tool
        log_request_tool_audit(rid, rec_tool, api=log_api, route=route)
        code, summary = compute_tool_turn_root_cause(
            wants_tools=wants_tools,
            had_backend_tool_delta=had_backend_tool_delta,
            proxy_emitted_structured_tool=pa.proxy_emitted_structured_tool,
            content_tool_like=pa.content_had_tool_like_markup,
            reasoning_tool_like=pa.reasoning_had_tool_like_markup,
            responses_endpoint_used=pa.responses_endpoint_used,
            responses_tools_forwarded=pa.responses_tools_forwarded,
            http_stream_status=http_status,
            admission_failure_class=pa.admission_failure_class,
        )
        aud = pa.emit(root_cause_code=code, root_cause_summary=summary)
        if wants_tools and code not in ("ok_structured_tools", "ok_no_tools_requested"):
            log_failure_taxonomy(
                rid,
                route,
                code,
                api=log_api,
                stream=True,
            )
        if sse_audit_comment_enabled() and http_status < 400:
            await stream_response.write(
                (f": {pa.sse_comment_line(aud)}\n\n").encode("utf-8")
            )

    async def start(self):
        """Start proxy server and background tasks"""
        logger.info("[DIAG] [PROXY.START] ========== Starting UnifiedProxy ==========")
        
        try:
            logger.info("[DIAG] [PROXY.START] Creating web.Application...")
            app = web.Application()
            
            # Add CORS middleware for OpenWebUI browser access
            @web.middleware
            async def cors_middleware(request, handler):
                response = await handler(request)
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, DELETE'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
                return response
            
            app.middlewares.append(cors_middleware)
            
            # Handle OPTIONS requests for CORS preflight
            async def options_handler(request):
                return web.Response(headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, DELETE',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                })
            
            app.router.add_options('/{path:.*}', options_handler)

            logger.info("[DIAG] [PROXY.START] ✓ web.Application created with CORS middleware")

            # Authentication middleware for internal API calls
            expected_api_key = os.getenv("PROXYAPP_API_KEY")

            @web.middleware
            async def auth_middleware(request: web.Request, handler):
                # Skip auth for CORS preflight + public endpoints (health/model discovery).
                # n8n Chat Hub performs model discovery probes and may not attach auth on every request.
                if request.method == "OPTIONS":
                    return await handler(request)

                # Skip auth for local requests (OpenWebUI, development)
                client_ip = request.remote
                if client_ip in ["127.0.0.1", "localhost", "::1"] or client_ip.startswith("172.17."):
                    return await handler(request)

                public_paths = [
                    "/healthcheck",
                    "/watchdog/status",
                    "/v1/models",
                    "/v1/chat/completions/models",
                    "/v1/responses/models",
                ]
                if any(request.path.startswith(path) for path in public_paths):
                    return await handler(request)

                # Check for Authorization header OR api_key query parameter (OpenWebUI compatibility)
                auth_header = request.headers.get("Authorization")
                api_key_param = request.query.get("api_key")

                if not auth_header and not api_key_param:
                    return web.json_response(
                        {
                            "error": {
                                "message": "Authorization header or api_key parameter required",
                                "type": "authentication_error",
                                "code": "missing_authorization",
                            },
                            # Compatibility: some n8n Chat Hub internals assume output is iterable
                            "output": [],
                            "choices": [],
                        },
                        status=401
                    )

                # Get token from Authorization header or api_key parameter
                if auth_header:
                    # For debugging, accept both Bearer format and direct API key
                    if auth_header.startswith("Bearer "):
                        token = auth_header[7:]  # Remove "Bearer " prefix
                    else:
                        token = auth_header  # Direct API key
                else:
                    token = api_key_param  # Use api_key from query parameter

                if not expected_api_key or token != expected_api_key:
                    return web.json_response(
                        {
                            "error": {
                                "message": "Invalid API key",
                                "type": "authentication_error",
                                "code": "invalid_api_key",
                            },
                            "output": [],
                            "choices": [],
                        },
                        status=401
                    )

                return await handler(request)

            app.middlewares.append(auth_middleware)
            logger.info("[DIAG] [PROXY.START] ✓ Authentication middleware added")

            # API endpoints
            logger.info("[DIAG] [PROXY.START] Registering API endpoints...")
            app.router.add_get("/healthcheck", self.healthcheck)
            app.router.add_get("/watchdog/status", self.watchdog_status)
            app.router.add_get("/v1/models", self.list_models)

            # Model discovery endpoints for n8n Chat Hub compatibility
            # n8n expects to discover models available for each API endpoint
            app.router.add_get("/v1/chat/completions/models", self.list_models)
            app.router.add_get("/v1/responses/models", self.list_models)

            app.router.add_post("/v1/chat/completions", self.chat_completions)

            # Handle /v1/responses - OpenAI Responses API compatibility layer
            app.router.add_post("/v1/responses", self.responses_api)
            logger.info("[DIAG] [PROXY.START] ✓ Core API endpoints registered")
            
            # Chat history endpoints
            app.router.add_post("/v1/chat/history", self.save_history)
            app.router.add_get("/v1/chat/history", self.get_history)
            app.router.add_get("/v1/chat/sessions", self.list_sessions)
            app.router.add_delete("/v1/chat/history/{session_id}", self.delete_session)
            logger.info("[DIAG] [PROXY.START] ✓ Chat history endpoints registered")
            
            # Model management endpoints
            app.router.add_post("/v1/models/{model_id}/load", self.load_model)
            app.router.add_post("/v1/models/{model_id}/unload", self.unload_model)
            app.router.add_get("/v1/models/{model_id}/status", self.model_status)
            logger.info("[DIAG] [PROXY.START] ✓ Model management endpoints registered")
            
            # Start background monitoring
            logger.info("[DIAG] [PROXY.START] Starting background tasks...")
            self.memory_manager.start_monitoring()
            self._queue_processing_task = asyncio.create_task(self._process_queues())
            self._health_check_task = asyncio.create_task(self._background_health_check())
            self._keep_alive_task = asyncio.create_task(self._llama_keep_alive())
            logger.info("[DIAG] [PROXY.START] ✓ Background tasks started (including llama-3.1 keep-alive)")
            
            # Start server
            logger.info("[DIAG] [PROXY.START] Setting up AppRunner...")
            runner = web.AppRunner(app)
            await runner.setup()
            logger.info("[DIAG] [PROXY.START] ✓ AppRunner setup complete")
            
            logger.info(f"[DIAG] [PROXY.START] Starting TCPSite on 0.0.0.0:{self.proxy_port}...")
            site = web.TCPSite(runner, "0.0.0.0", self.proxy_port)
            await site.start()
            logger.info(f"[DIAG] [PROXY.START] ✓ TCPSite started successfully")
            
            logger.info(f"[DIAG] [PROXY.START] ========== Unified proxy server started on port {self.proxy_port} ==========")
        except Exception as e:
            logger.error(f"[DIAG] [PROXY.START] ✗ EXCEPTION in start(): {e}", exc_info=True)
            raise
    
    async def healthcheck(self, request: Request) -> Response:
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "proxy_port": self.proxy_port,
            "models": len(self.model_manager.models),
            "loaded_models": sum(
                1 for m in self.model_manager.models.values()
                if m.status == "loaded"
            )
        })
    
    async def list_models(self, request: Request) -> Response:
        """List all available models (aggregated from all backends)"""
        now = time.time()
        diag = os.environ.get("PROXY_DIAG_LIST_MODELS", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        throttle_ok = now >= self._list_models_log_deadline
        if diag:
            log_lm = logger.info
        elif throttle_ok:
            self._list_models_log_deadline = now + _LIST_MODELS_LOG_INTERVAL_S
            log_lm = logger.info
        else:
            log_lm = logger.debug
        log_lm("[DIAG] [LIST_MODELS] ========== /v1/models endpoint called ==========")
        models = []

        log_lm("[DIAG] [LIST_MODELS] Total models in manager: %s", len(self.model_manager.models))
        if not self.model_manager.models:
            logger.error("[DIAG] [LIST_MODELS] ✗ CRITICAL: No models found in ModelManager!")
            logger.error("[DIAG] [LIST_MODELS] This should never happen - models should be initialized in __init__")

        for model_id, model_info in self.model_manager.models.items():
            log_lm(
                "[DIAG] [LIST_MODELS] Processing model: %s, status: %s, port: %s",
                model_id,
                model_info.status,
                model_info.port,
            )
            
            # Determine availability based on status
            # OpenWebUI filters out models with available=false, so we need available=true for all models
            is_available = model_info.status in ["loaded", "loading", "unloaded"]
            
            # OpenWebUI expects OpenAI-compatible format
            # OpenAI format: id, object, created, owned_by (no "available" field)
            # OpenWebUI will show all models returned - don't filter with "available"
            model_dict = {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": model_info.backend.value
            }

            # Add OpenAPI extra fields if available (needed for n8n/OpenCode compatibility)
            if model_info.openapi_extra:
                model_dict.update(model_info.openapi_extra)
            # Don't add "available" field - OpenWebUI shows all models by default
            
            models.append(model_dict)
            log_lm(
                "[DIAG] [LIST_MODELS] Added model: %s, available=%s, status=%s",
                model_id,
                is_available,
                model_info.status,
            )

        log_lm("[DIAG] [LIST_MODELS] Returning %s models", len(models))
        log_lm("[DIAG] [LIST_MODELS] ========== /v1/models response ready ==========")
        return web.json_response({
            "object": "list",
            "data": models
        })

    async def watchdog_status(self, request: Request) -> Response:
        """Watchdog monitoring endpoint - provides detailed system status"""
        current_time = asyncio.get_event_loop().time()

        # Get active request details
        active_requests_info = []
        long_running_requests = []

        for req_id, start_time in self.active_requests.items():
            duration = current_time - start_time
            req_info = {
                "request_id": req_id,
                "start_time": start_time,
                "duration_seconds": duration,
                "duration_human": f"{duration:.1f}s"
            }
            active_requests_info.append(req_info)

            # Check for long-running requests (15+ minutes)
            if duration > 900:  # 15 minutes
                long_running_requests.append(req_info)

        # Get model status
        model_status = {}
        for model_id, model_info in self.model_manager.models.items():
            model_status[model_id] = {
                "status": model_info.status,
                "backend": model_info.backend.value if hasattr(model_info.backend, 'value') else str(model_info.backend),
                "port": model_info.port,
                "container_name": getattr(model_info, 'container_name', None)
            }

        # Get queue status
        queue_status = {
            "queue_size": len(self.request_queue.queue),
            "processing": self.request_queue.processing
        }

        return web.json_response({
            "timestamp": current_time,
            "active_requests": {
                "count": len(active_requests_info),
                "details": active_requests_info,
                "long_running": long_running_requests
            },
            "models": model_status,
            "queue": queue_status,
            "memory": {
                "threshold_percent": self.memory_manager.threshold_percent,
                "last_check": getattr(self.memory_manager, 'last_check_time', None)
            }
        })
    
    async def chat_completions(self, request) -> Response:
        """Handle chat completion requests"""
        start_time = asyncio.get_event_loop().time()
        self._current_request = request
        try:
            # Handle both Request objects and dicts
            if hasattr(request, 'json'):
                client_body = await request.json()
            else:
                client_body = request

            inbound_raw = json.loads(json.dumps(client_body))
            norm_meta: Dict[str, Any] = {}
            data = normalize_openai_chat_request(client_body, norm_meta)
            rid = get_or_create_request_id(request)
            self.active_requests[rid] = start_time

            normalization_notes = {**normalization_delta(inbound_raw, data), **norm_meta}
            log_chat_json_payload(rid, "inbound_chat_completions", inbound_raw)
            log_api_inbound_evidence(rid, "POST /v1/chat/completions", "chat", inbound_raw)
            # Note: dict path loses headers; UA-based debug kept on normalized payload
            if "model" in data and any(keyword in data.get("model", "").lower() for keyword in ["qwen", "deepseek", "nous", "llama"]):
                logger.info(f"[OPENCODE-DEBUG] Raw request body: {json.dumps(data, indent=2)}")
                logger.info(f"[OPENCODE-DEBUG] Processing OpenCode request")

            logger.debug("[chat] normalized model=%s", data.get("model"))
            model_id = data.get("model")

            if not model_id:
                return web.json_response(
                    {"error": "model parameter required", "output": [], "choices": []},
                    status=400
                )
            
            # Get model info
            if model_id not in self.model_manager.models:
                hint = ""
                if isinstance(model_id, str) and model_id.startswith("custom:"):
                    hint = (
                        " Use a model `id` exactly as returned by GET /v1/models on this proxy "
                        "(Droid/Factory `custom:` aliases are not mapped automatically)."
                    )
                return web.json_response(
                    {
                        "error": f"Model {model_id} not found.{hint}",
                        "output": [],
                        "choices": [],
                    },
                    status=404,
                )
            
            model_info = self.model_manager.models[model_id]
            
            # Check if model is loaded
            if model_info.status != "loaded":
                # Queue request if model is loading
                if model_info.status == "loading":
                    # Check container status for better UX
                    container_running = self._check_container_running(model_info)
                    container_ready = (
                        self._refresh_readiness(model_id, model_info)[0]
                        if container_running
                        else False
                    )

                    # Return clear loading status with progress info
                    if not container_running:
                        message = f"Model {model_id} container is starting..."
                    elif not container_ready:
                        message = f"Model {model_id} is initializing (this may take 1-2 minutes)..."
                    else:
                        message = f"Model {model_id} is almost ready..."

                    self._emit_admission_failure_trace(
                        request,
                        "POST /v1/chat/completions",
                        failure_class="admission_not_ready",
                        model_id=model_id,
                        admission_kind="initializing",
                        model_info=model_info,
                        container_running=container_running,
                    )
                    return web.json_response({
                        "error": {
                            "message": message,
                            "type": "model_loading",
                            "code": "model_not_ready"
                        },
                        "status": "loading",
                        "model_id": model_id,
                        "container_running": container_running,
                        "container_ready": container_ready,
                        "queue_size": self.request_queue.get_queue_size(model_id),
                        "message": "Please wait and retry in 30 seconds.",
                        "readiness": {"last_probe_class": model_info.last_readiness_probe_class},
                        "output": [],
                        "choices": [],
                    }, status=503)
                
                # Start loading model
                logger.info(f"Loading model {model_id} for request")
                loading_started = self.model_manager.load_model(model_id)
                
                if not loading_started:
                    return web.json_response(
                        {"error": f"Failed to load model {model_id}", "output": [], "choices": []},
                        status=500
                    )
                
                # Queue request - create async callback
                async def process_queued_request():
                    return await self._forward_request(
                        model_id,
                        data,
                        request,
                        inbound_raw=inbound_raw,
                        normalization_notes=normalization_notes,
                        route_label="POST /v1/chat/completions",
                    )

                request_id = self.request_queue.enqueue(
                    model_id=model_id,
                    request_data=data,
                    callback=process_queued_request
                )
                
                return web.json_response({
                    "status": "loading",
                    "message": f"Model {model_id} is being loaded. Request queued.",
                    "request_id": request_id,
                    "model_id": model_id,
                    "output": [],
                    "choices": [],
                }, status=503)
            
            # Forward request to backend
            response = await self._forward_request(
                model_id,
                data,
                request,
                inbound_raw=inbound_raw,
                normalization_notes=normalization_notes,
                route_label="POST /v1/chat/completions",
            )

            # Clean up request tracking on completion
            if rid in self.active_requests:
                del self.active_requests[rid]

            return response

        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON"},
                status=400
            )
        except Exception as e:
            logger.error(f"Error handling chat completion: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.error(f"Request data at error: {locals().get('data')}")
            rlocal = locals().get("rid")
            if rlocal and rlocal in self.active_requests:
                del self.active_requests[rlocal]
            return web.json_response(
                {"error": str(e)},
                status=500
            )

    async def responses_api(self, request: Request) -> Response:
        """Handle /v1/responses requests - OpenAI Responses API compatibility layer.

        This implements a compatibility layer that maps Responses API requests
        to internal chat completions calls, allowing modern clients to work
        while keeping the proxy reliable.
        """
        try:
            # Parse request body
            raw_data = await request.json()
            rid = get_or_create_request_id(request)
            log_chat_json_payload(rid, "inbound_responses", raw_data)
            log_api_inbound_evidence(rid, "POST /v1/responses", "responses", raw_data)

            # DEBUG: Log the incoming request for troubleshooting
            logger.info(f"[RESPONSES] Incoming request: {json.dumps(raw_data, indent=2)}")

            want_stream = bool(raw_data.get("stream", False))

            # Map Responses API request to chat completions format
            try:
                chat_request = self._map_responses_to_chat(raw_data)
                logger.info(f"[RESPONSES] Mapped to chat request: {json.dumps(chat_request, indent=2)}")
            except ValueError as e:
                logger.error(f"[RESPONSES] Mapping error: {e}")
                return web.json_response(
                    {
                        "error": {
                            "message": str(e),
                            "type": "not_implemented",
                            "code": "responses_feature_not_supported"
                        },
                        "output": [],
                        "output_text": "",
                        "choices": [],
                    },
                    status=400
                )

            # IMPORTANT: /v1/responses must return a real output payload (not a queued/loading placeholder)
            # because n8n Chat Hub internal workflows will fail if response.output isn't a proper array of items.
            #
            # Additionally, n8n's LangChain OpenAI integration often sets `stream: true` and expects
            # SSE events in the OpenAI Responses streaming format. If we return plain JSON while
            # the client expects a stream, LangChain yields empty generations and the agent node crashes.
            if want_stream:
                return await self._stream_responses_from_chat_backend(request, raw_data, chat_request)

            chat_data = await self._process_chat_request_blocking(chat_request, request)
            logger.info(f"[RESPONSES] Chat response: {json.dumps(chat_data, indent=2)}")

            # Map chat completions response to Responses API format (or Chat Completions for langchain)
            responses_data = self._map_chat_to_responses(raw_data, chat_data, request)
            logger.info(f"[RESPONSES] Final response: {json.dumps(responses_data, indent=2)}")

            return web.json_response(responses_data)

        except TimeoutError as e:
            # Model still loading / backend not ready in time
            logger.warning(f"Timeout in responses_api: {e}")
            return web.json_response(
                {
                    "error": {
                        "message": str(e),
                        "type": "model_loading",
                        "code": "model_not_ready",
                    },
                    "output": [],
                    "output_text": "",
                    "choices": [],
                },
                status=503,
            )
        except Exception as e:
            logger.error(f"Error in responses_api: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return web.json_response(
                {
                    "error": {
                        "message": f"Error processing responses request: {str(e)}",
                        "type": "internal_error",
                        "code": "responses_processing_error"
                    },
                    "output": [],
                    "output_text": "",
                    "choices": [],
                },
                status=500
            )

    async def _stream_responses_sse(self, request: Request, responses_data: Dict) -> Response:
        """Stream an OpenAI Responses API style SSE stream.

        n8n (via @langchain/openai ChatOpenAIResponses) expects events like:
        - response.created
        - response.output_item.added
        - response.output_text.delta
        - response.completed
        followed by [DONE]
        """
        resp = web.StreamResponse(status=200)
        resp.content_type = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Connection"] = "keep-alive"
        await resp.prepare(request)

        response_id = responses_data.get("id", "resp_unknown")
        model = responses_data.get("model", "unknown")
        output_items = responses_data.get("output") if isinstance(responses_data.get("output"), list) else []
        msg_item = output_items[0] if output_items else None
        msg_id = msg_item.get("id") if isinstance(msg_item, dict) else f"msg_{response_id}"
        output_text = responses_data.get("output_text", "")

        async def send_event(obj: Dict):
            payload = json.dumps(obj, ensure_ascii=False)
            await resp.write(f"data: {payload}\n\n".encode("utf-8"))

        # response.created
        await send_event(
            {
                "type": "response.created",
                "response": {"id": response_id, "model": model, "object": "response"},
            }
        )

        # response.output_item.added (message)
        await send_event(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"id": msg_id, "type": "message", "role": "assistant", "content": []},
            }
        )

        # response.output_text.delta (chunked)
        if isinstance(output_text, str) and output_text:
            chunk_size = 120
            for i in range(0, len(output_text), chunk_size):
                await send_event(
                    {
                        "type": "response.output_text.delta",
                        "delta": output_text[i : i + chunk_size],
                        "content_index": 0,
                        "output_index": 0,
                    }
                )
                await asyncio.sleep(0)  # yield control

        # Ensure completed response contains required fields expected by LangChain converters
        completed_response = dict(responses_data)
        # OpenAI Responses API uses created_at + status; keep existing fields too
        completed_response.setdefault("created_at", completed_response.get("created"))
        completed_response.setdefault("status", "completed")

        await send_event({"type": "response.completed", "response": completed_response})
        await resp.write(b"data: [DONE]\n\n")
        return resp

    def _split_think_block(self, text: str) -> tuple[str, str]:
        """Split '<think>...</think>' blocks from text.

        Many vLLM/chat templates (e.g., Qwen) return reasoning inside the assistant content as:
          <think> ... </think>\n\nFINAL_ANSWER
        n8n Chat Hub should display only the final answer. We strip the think block and return it
        separately for optional debugging.
        """
        if not isinstance(text, str) or "<think>" not in text:
            return ("", text if isinstance(text, str) else "")

        start = text.find("<think>")
        end = text.find("</think>", start + 7)
        if start != -1 and end != -1:
            reasoning = text[start + 7 : end]
            visible = (text[:start] + text[end + 8 :]).strip()
            return (reasoning.strip(), visible)
        return ("", text)

    async def _stream_responses_from_chat_backend(self, request: Request, original_request: Dict, chat_request: Dict) -> Response:
        """Proxy streaming from /v1/chat/completions into Responses SSE events.

        This keeps n8n's 60s timeout happy by sending data immediately and then streaming deltas.
        It also strips <think> blocks from the streamed content (Qwen-style).
        """
        import uuid
        import time as _time

        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        msg_id = f"msg_{response_id}"
        model_id = chat_request.get("model", "unknown")

        resp = web.StreamResponse(status=200)
        resp.content_type = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Connection"] = "keep-alive"

        # Track in-flight request so memory eviction doesn't unload this model mid-stream.
        self._active_model_requests[model_id] += 1

        # Ensure model is loaded/ready before we start backend streaming
        try:

            norm_meta_resp: Dict[str, Any] = {}
            mapped_snap = json.loads(json.dumps(chat_request))
            normalized = normalize_openai_chat_request(dict(chat_request), norm_meta_resp)
            rid = get_or_create_request_id(request)
            ledger_r = StreamCommittedLedger(
                request_id=rid,
                idempotency_key=extract_idempotency_key(original_request),
            )
            client_write_closed_r = False

            async def send_event(obj: Dict) -> bool:
                nonlocal client_write_closed_r
                payload = json.dumps(obj, ensure_ascii=False)
                st = await sse_write(resp, f"data: {payload}\n\n".encode("utf-8"))
                if st == "client_disconnect":
                    client_write_closed_r = True
                    return False
                ledger_r.first_byte_sent = True
                return True

            async def send_comment(comment: str = "keep-alive") -> bool:
                st = await sse_write(resp, f": {comment}\n\n".encode("utf-8"))
                if st == "client_disconnect":
                    return False
                return True

            req_tools = bool(
                (isinstance(original_request.get("tools"), list) and len(original_request.get("tools") or []) > 0)
                or (original_request.get("tool_choice") is not None)
            )
            fwd_ok = bool(normalized.get("tools")) or (normalized.get("tool_choice") is not None)
            responses_tools_forwarded = req_tools and fwd_ok
            if req_tools and not fwd_ok:
                logger.error("[RESPONSES] tools/tool_choice were present on the client request but absent after map/normalize")

            model_id = normalized.get("model", model_id)
            if model_id not in self.model_manager.models:
                raise ValueError(f"Model '{model_id}' not available")
            model_info = self.model_manager.models[model_id]

            await resp.prepare(request)

            # Send created event ASAP (prevents "no response" timeouts)
            await send_event(
                {
                    "type": "response.created",
                    "response": {"id": response_id, "model": model_id, "object": "response"},
                }
            )
            await send_event(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"id": msg_id, "type": "message", "role": "assistant", "content": []},
                }
            )

            # Keep access times fresh so memory eviction doesn't target an in-flight stream.
            try:
                self.memory_manager.update_access_time(model_id)
                self.model_manager.update_access_time(model_id)
            except Exception:
                pass

            # IMPORTANT: "status=loaded" can drift from reality (e.g., vLLM container exits).
            # Re-check liveness and restart if needed.
            is_running = self._check_container_running(model_info)
            is_ready = (
                self._refresh_readiness(model_id, model_info)[0] if is_running else False
            )
            if not is_running or not is_ready:
                logger.warning(
                    f"[RESPONSES] Backend not ready for {model_id} (running={is_running}, ready={is_ready}); reloading"
                )
                model_info.status = "unloaded"

            if model_info.status != "loaded":
                ok = await asyncio.to_thread(self.model_manager.load_model, model_id)
                if not ok:
                    raise RuntimeError(f"Failed to load model '{model_id}'")
                # While waiting, send keep-alives so clients (n8n) don't hit read timeouts.
                start_wait = _time.time()
                while _time.time() - start_wait < 240.0:
                    if await self._wait_for_model_ready(model_id, timeout_s=0.5):
                        break
                    # Keep-alive + touch access time so eviction doesn't treat this as idle.
                    try:
                        self.memory_manager.update_access_time(model_id)
                        self.model_manager.update_access_time(model_id)
                    except Exception:
                        pass
                    await send_comment("warming-up")
                    await asyncio.sleep(2.0)
                else:
                    raise TimeoutError(f"Model '{model_id}' is still loading")

            # Remap/strip fields like _forward_request does
            data = normalized.copy()
            if model_info.backend == ModelBackend.VLLM:
                data["model"] = "/app/model"
            elif model_info.backend == ModelBackend.LLAMACPP:
                for key in ("tools", "tool_choice", "stream_options"):
                    data.pop(key, None)

                workspace_dir = Path(os.environ.get("EXO_WORKSPACE", REPO_ROOT))
                model_path = Path(workspace_dir) / model_info.model_path
                if model_path.is_dir():
                    gguf_files = list(model_path.glob("*.gguf"))
                    if gguf_files:
                        preferred_order = ["Q8_0", "Q6_K", "Q4_K_M", "Q4_K"]
                        for pref in preferred_order:
                            for gguf_file in gguf_files:
                                if pref in gguf_file.name:
                                    model_path = gguf_file
                                    break
                            if model_path.suffix == ".gguf":
                                break
                        if model_path.suffix != ".gguf":
                            model_path = gguf_files[0]
                data["model"] = str(model_path)

            data["stream"] = True
            backend_url = f"http://localhost:{model_info.port}/v1/chat/completions"
            backend_snapshot = json.loads(json.dumps(data))
            pa = ProductionAuditSession(
                request_id=rid,
                route="POST /v1/responses",
                api="v1/responses",
                inbound_raw=original_request,
                normalized_client=mapped_snap,
                normalization_delta={**normalization_delta(mapped_snap, normalized), **norm_meta_resp},
                backend_payload=backend_snapshot,
                backend_url=backend_url,
                wants_tools=bool(data.get("tools")),
                responses_endpoint_used=True,
                chat_endpoint_used=False,
                responses_tools_forwarded=responses_tools_forwarded,
                backend_health_state=getattr(model_info, "last_readiness_probe_class", None)
                or model_info.status,
                admission_decision="forwarded",
            )
            pa.normalization_anthropic_stripped = list(norm_meta_resp.get("anthropic_stripped", []))
            log_normalized_client_payload(
                rid,
                "v1/responses",
                "POST /v1/responses",
                normalized,
                {**normalization_delta(mapped_snap, normalized), **norm_meta_resp},
            )
            log_backend_request_payload(
                rid, "v1/responses", "POST /v1/responses", backend_url, backend_snapshot
            )

            visible_accum: list[str] = []
            last_write = _time.time()
            rs_sanitizer = ReasoningXmlSanitizer()
            tool_audit_r = ToolStreamAudit(api="v1/responses")
            tool_calls_accum: Dict[str, Any] = {}
            had_backend_tool_delta = False
            wants_tools_r = bool(data.get("tools"))
            seq_b = 0
            seq_o = 0
            finish_reason_r: Optional[str] = None
            reasoning_commit_buf = ""

            async def emit_reasoning_committed(piece: str) -> None:
                nonlocal seq_o
                if not piece:
                    return
                tool_audit_r.note_client_reasoning(len(piece))
                if raw_visible_suggests_tool_calls(piece):
                    pa.reasoning_had_tool_like_markup = True
                evr = {
                    "type": "response.reasoning_text.delta",
                    "item_id": msg_id,
                    "delta": piece,
                }
                await send_event(evr)
                ledger_r.committed_reasoning_offset += len(piece)
                seq_o += 1
                log_proxy_outbound_delta(
                    rid,
                    "v1/responses",
                    "POST /v1/responses",
                    sequence=seq_o,
                    surface="response.sse",
                    emitted=evr,
                    projection=projection_responses_outbound_event(evr),
                )

            async def flush_reasoning_commits() -> None:
                nonlocal reasoning_commit_buf
                while True:
                    idx = _responses_reasoning_commit_index(reasoning_commit_buf)
                    if idx <= 0:
                        break
                    part = reasoning_commit_buf[:idx]
                    reasoning_commit_buf = reasoning_commit_buf[idx:]
                    await emit_reasoning_committed(part)

            async def finalize_reasoning_at_eof() -> None:
                nonlocal reasoning_commit_buf
                while True:
                    idx = _responses_reasoning_commit_index(reasoning_commit_buf)
                    if idx <= 0:
                        break
                    part = reasoning_commit_buf[:idx]
                    reasoning_commit_buf = reasoning_commit_buf[idx:]
                    await emit_reasoning_committed(part)
                if reasoning_commit_buf:
                    logger.debug(
                        "[RESPONSES] dropped ambiguous reasoning tail at EOF len=%s",
                        len(reasoning_commit_buf),
                    )
                    reasoning_commit_buf = ""

            saw_done_r = False
            stream_exc_r: Optional[BaseException] = None
            idle_timeout_r = False
            idle_resp_to = stream_idle_timeout_seconds()
            async with ClientSession() as session:
                async with session.post(backend_url, json=data, timeout=300) as backend_resp:
                    if backend_resp.status >= 400:
                        body = await backend_resp.text()
                        self._trip_heavy_admission_circuit(model_id)
                        await self._emit_abnormal_stream_terminal(
                            resp,
                            ledger_r,
                            "upstream_eof",
                            request_json=original_request,
                        )
                        rs_sanitizer.finalize()
                        tool_audit_r.sanitizer_eof = rs_sanitizer.trace_state()
                        pa.finish_reason = finish_reason_r
                        await self._finalize_chat_stream_production_audit(
                            pa=pa,
                            tool_audit=tool_audit_r,
                            sanitizer=rs_sanitizer,
                            wants_tools=wants_tools_r,
                            had_backend_tool_delta=had_backend_tool_delta,
                            http_status=502,
                            stream_response=resp,
                            route="POST /v1/responses",
                            log_api="v1/responses",
                        )
                        return resp

                    buffer = b""
                    try:
                        async for chunk in iter_chunked_with_idle(
                            backend_resp.content, 8192, idle_resp_to
                        ):
                            if not chunk:
                                continue
                            buffer += chunk
                            while b"\n" in buffer:
                                line, buffer = buffer.split(b"\n", 1)
                                line = line.strip()
                                if not line:
                                    continue
                                if line.startswith(b"data:"):
                                    payload = line[5:].strip()
                                    if payload == b"[DONE]":
                                        saw_done_r = True
                                        buffer = b""
                                        break
                                    try:
                                        obj = json.loads(payload.decode("utf-8"))
                                    except Exception:
                                        continue
                                    try:
                                        choice0 = (obj.get("choices") or [])[0] or {}
                                    except Exception:
                                        continue
                                    delta_obj = choice0.get("delta") or {}
                                    fr0 = choice0.get("finish_reason")
                                    if fr0:
                                        finish_reason_r = fr0
                                    seq_b += 1
                                    log_backend_sse_delta(
                                        rid,
                                        "v1/responses",
                                        "POST /v1/responses",
                                        sequence=seq_b,
                                        backend_url=backend_url,
                                        raw_sse_bytes=len(line),
                                        parsed=canonical_backend_chat_sse(obj),
                                        projection=projection_backend_delta(
                                            delta_obj,
                                            choice0.get("finish_reason"),
                                        ),
                                    )
                                    reasoning_native = (
                                        delta_obj.get("reasoning") or delta_obj.get("reasoning_content") or ""
                                    )
                                    if isinstance(reasoning_native, str) and reasoning_native:
                                        reasoning_commit_buf += reasoning_native

                                    tool_calls = delta_obj.get("tool_calls") or []
                                    await flush_reasoning_commits()
                                    if tool_calls:
                                        had_backend_tool_delta = True
                                        tool_audit_r.note_backend_delta(delta_obj)
                                    for tc in tool_calls:
                                        cid = str(tc.get("id") if tc.get("id") is not None else tc.get("index", ""))
                                        fn = tc.get("function") or {}
                                        argfrag = fn.get("arguments") or ""
                                        if cid not in tool_calls_accum:
                                            tool_calls_accum[cid] = {
                                                "id": cid,
                                                "type": "function",
                                                "function": {
                                                    "name": fn.get("name") or "",
                                                    "arguments": "",
                                                },
                                            }
                                            ev_add = {
                                                "type": "response.output_item.added",
                                                "output_index": len(tool_calls_accum),
                                                "item": {
                                                    "id": cid,
                                                    "type": "function_call",
                                                    "call_id": cid,
                                                    "name": fn.get("name") or "",
                                                    "status": "in_progress",
                                                },
                                            }
                                            await send_event(ev_add)
                                            seq_o += 1
                                            log_proxy_outbound_delta(
                                                rid,
                                                "v1/responses",
                                                "POST /v1/responses",
                                                sequence=seq_o,
                                                surface="response.sse",
                                                emitted=ev_add,
                                                projection=projection_responses_outbound_event(ev_add),
                                            )
                                            pa.proxy_emitted_structured_tool = True
                                        tool_calls_accum[cid]["function"]["arguments"] += argfrag
                                        ev_fc = {
                                            "type": "response.function_call_arguments.delta",
                                            "call_id": cid,
                                            "delta": argfrag,
                                        }
                                        await send_event(ev_fc)
                                        try:
                                            tidx = int(
                                                tc.get("index", len(tool_calls_accum) - 1)
                                            )
                                        except (TypeError, ValueError):
                                            tidx = len(tool_calls_accum) - 1
                                        ledger_r.last_tool_event = {
                                            "index": tidx,
                                            "partial": True,
                                        }
                                        tool_audit_r.note_proxy_responses_out(ev_fc)
                                        seq_o += 1
                                        log_proxy_outbound_delta(
                                            rid,
                                            "v1/responses",
                                            "POST /v1/responses",
                                            sequence=seq_o,
                                            surface="response.sse",
                                            emitted=ev_fc,
                                            projection=projection_responses_outbound_event(ev_fc),
                                        )
                                        try:
                                            self.memory_manager.update_access_time(model_id)
                                            self.model_manager.update_access_time(model_id)
                                        except Exception:
                                            pass
                                        last_write = _time.time()

                                    delta = delta_obj.get("content") or ""
                                    r_part, visible_delta = (
                                        rs_sanitizer.feed_content(delta)
                                        if isinstance(delta, str) and delta
                                        else (None, None)
                                    )
                                    pa.merge_sanitizer_audit(rs_sanitizer.take_audit_events())
                                    if isinstance(r_part, str) and r_part:
                                        reasoning_commit_buf += r_part
                                    await flush_reasoning_commits()
                                    if visible_delta:
                                        if raw_visible_suggests_tool_calls(visible_delta):
                                            pa.content_had_tool_like_markup = True
                                        tool_audit_r.note_client_visible(len(visible_delta))
                                    if visible_delta:
                                        visible_accum.append(visible_delta)
                                        ev_txt = {
                                            "type": "response.output_text.delta",
                                            "delta": visible_delta,
                                            "content_index": 0,
                                            "output_index": 0,
                                        }
                                        await send_event(ev_txt)
                                        ledger_r.committed_text_offset += len(visible_delta)
                                        try:
                                            self.memory_manager.update_access_time(model_id)
                                            self.model_manager.update_access_time(model_id)
                                        except Exception:
                                            pass
                                        last_write = _time.time()
                                        seq_o += 1
                                        log_proxy_outbound_delta(
                                            rid,
                                            "v1/responses",
                                            "POST /v1/responses",
                                            sequence=seq_o,
                                            surface="response.sse",
                                            emitted=ev_txt,
                                            projection=projection_responses_outbound_event(ev_txt),
                                        )
                                    else:
                                        if _time.time() - last_write > 5:
                                            try:
                                                self.memory_manager.update_access_time(model_id)
                                                self.model_manager.update_access_time(model_id)
                                            except Exception:
                                                pass
                                            await send_comment()
                                            last_write = _time.time()

                            if saw_done_r:
                                break

                    except StreamIdleTimeoutError as e:
                        stream_exc_r = e
                        idle_timeout_r = True
                    except ClientError as e:
                        stream_exc_r = e

                    if client_write_closed_r:
                        rs_sanitizer.finalize()
                        tool_audit_r.sanitizer_eof = rs_sanitizer.trace_state()
                        pa.finish_reason = finish_reason_r
                        await self._finalize_chat_stream_production_audit(
                            pa=pa,
                            tool_audit=tool_audit_r,
                            sanitizer=rs_sanitizer,
                            wants_tools=wants_tools_r,
                            had_backend_tool_delta=had_backend_tool_delta,
                            http_status=200,
                            stream_response=resp,
                            route="POST /v1/responses",
                            log_api="v1/responses",
                        )
                        return resp

                    if not saw_done_r:
                        term_r = classify_stream_termination(
                            exc=stream_exc_r,
                            saw_done=False,
                            mid_stream=ledger_r.first_byte_sent,
                            idle_timeout=idle_timeout_r,
                        )
                        if term_r == "client_disconnect":
                            rs_sanitizer.finalize()
                            tool_audit_r.sanitizer_eof = rs_sanitizer.trace_state()
                            pa.finish_reason = finish_reason_r
                            await self._finalize_chat_stream_production_audit(
                                pa=pa,
                                tool_audit=tool_audit_r,
                                sanitizer=rs_sanitizer,
                                wants_tools=wants_tools_r,
                                had_backend_tool_delta=had_backend_tool_delta,
                                http_status=200,
                                stream_response=resp,
                                route="POST /v1/responses",
                                log_api="v1/responses",
                            )
                            return resp
                        if should_trip_admission_on_classify(term_r):
                            self._trip_heavy_admission_circuit(model_id)
                        await self._emit_abnormal_stream_terminal(
                            resp, ledger_r, term_r, request_json=original_request
                        )
                        rs_sanitizer.finalize()
                        tool_audit_r.sanitizer_eof = rs_sanitizer.trace_state()
                        pa.finish_reason = finish_reason_r
                        await self._finalize_chat_stream_production_audit(
                            pa=pa,
                            tool_audit=tool_audit_r,
                            sanitizer=rs_sanitizer,
                            wants_tools=wants_tools_r,
                            had_backend_tool_delta=had_backend_tool_delta,
                            http_status=502,
                            stream_response=resp,
                            route="POST /v1/responses",
                            log_api="v1/responses",
                        )
                        return resp

                    r_fin, v_fin = rs_sanitizer.finalize()
                    pa.merge_sanitizer_audit(rs_sanitizer.take_audit_events())
                    tool_audit_r.sanitizer_eof = rs_sanitizer.trace_state()
                    if v_fin and v_fin.strip():
                        visible_accum.append(v_fin)
                        ev_tail = {
                            "type": "response.output_text.delta",
                            "delta": v_fin,
                            "content_index": 0,
                            "output_index": 0,
                        }
                        await send_event(ev_tail)
                        ledger_r.committed_text_offset += len(v_fin)
                        seq_o += 1
                        log_proxy_outbound_delta(
                            rid,
                            "v1/responses",
                            "POST /v1/responses",
                            sequence=seq_o,
                            surface="response.sse",
                            emitted=ev_tail,
                            projection=projection_responses_outbound_event(ev_tail),
                        )
                    if isinstance(r_fin, str) and r_fin.strip():
                        reasoning_commit_buf += r_fin
                    await finalize_reasoning_at_eof()

                    if tool_calls_accum:
                        for call_id, tc in tool_calls_accum.items():
                            ev_done = {
                                "type": "response.function_call_arguments.done",
                                "call_id": call_id,
                                "name": (tc.get("function") or {}).get("name") or "",
                                "arguments": (tc.get("function") or {}).get("arguments") or "",
                            }
                            await send_event(ev_done)
                            try:
                                didx = int(call_id)
                            except (TypeError, ValueError):
                                didx = 0
                            ledger_r.last_tool_event = {"index": didx, "partial": False}
                            seq_o += 1
                            log_proxy_outbound_delta(
                                rid,
                                "v1/responses",
                                "POST /v1/responses",
                                sequence=seq_o,
                                surface="response.sse",
                                emitted=ev_done,
                                projection=projection_responses_outbound_event(ev_done),
                            )

            visible_text = "".join(visible_accum).strip()
            completed_output: list[dict[str, Any]] = []
            if visible_text:
                completed_output.append(
                    {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": visible_text}],
                    }
                )
            if tool_calls_accum:
                for tc in tool_calls_accum.values():
                    fn = tc.get("function") or {}
                    completed_output.append(
                        {
                            "type": "function_call",
                            "call_id": tc["id"],
                            "name": fn.get("name") or "",
                            "arguments": fn.get("arguments") or "",
                        }
                    )

            completed_response = {
                "id": response_id,
                "object": "response",
                "created": int(_time.time()),
                "created_at": int(_time.time()),
                "status": "completed",
                "model": model_id,
                "output": completed_output,
                "output_text": visible_text,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "choices": [],
            }
            await send_event({"type": "response.completed", "response": completed_response})
            pa.finish_reason = finish_reason_r
            await self._finalize_chat_stream_production_audit(
                pa=pa,
                tool_audit=tool_audit_r,
                sanitizer=rs_sanitizer,
                wants_tools=wants_tools_r,
                had_backend_tool_delta=had_backend_tool_delta,
                http_status=200,
                stream_response=resp,
                route="POST /v1/responses",
                log_api="v1/responses",
            )
            await sse_write(resp, b"data: [DONE]\n\n")
            return resp

        except Exception as e:
            logger.error(f"[RESPONSES] Streaming failed: {e}", exc_info=True)
            lr = locals().get("ledger_r")
            mid = bool(lr and lr.first_byte_sent)
            term_x = classify_stream_termination(
                exc=e, saw_done=False, mid_stream=mid, idle_timeout=False
            )
            if lr is not None and term_x != "client_disconnect":
                if should_trip_admission_on_classify(term_x):
                    try:
                        self._trip_heavy_admission_circuit(model_id)
                    except Exception:
                        pass
                try:
                    await self._emit_abnormal_stream_terminal(
                        resp, lr, term_x, request_json=original_request
                    )
                except Exception:
                    pass
            try:
                if locals().get("pa") and locals().get("tool_audit_r") and locals().get("rs_sanitizer"):
                    await self._finalize_chat_stream_production_audit(
                        pa=locals()["pa"],
                        tool_audit=locals()["tool_audit_r"],
                        sanitizer=locals()["rs_sanitizer"],
                        wants_tools=locals().get("wants_tools_r", False),
                        had_backend_tool_delta=locals().get("had_backend_tool_delta", False),
                        http_status=502,
                        stream_response=resp,
                        route="POST /v1/responses",
                        log_api="v1/responses",
                    )
            except Exception:
                pass
            return resp
        finally:
            try:
                self._active_model_requests[model_id] -= 1
                if self._active_model_requests[model_id] <= 0:
                    self._active_model_requests.pop(model_id, None)
            except Exception:
                pass

    async def _process_chat_request_blocking(self, data: Dict, request: Request) -> Dict:
        """Process a chat request without queueing.

        /v1/responses is used by n8n Chat Hub internal workflows. Returning a 200 response
        with empty output while a model is loading causes those workflows to fail.
        This helper loads the model synchronously and waits until it is ready before forwarding.
        """
        data = normalize_openai_chat_request(data)

        model_id = data.get("model", "llama-3.1-8b-q4k-q4_k")
        if model_id not in self.model_manager.models:
            available_models = list(self.model_manager.models.keys())
            raise ValueError(f"Model '{model_id}' not available. Available models: {available_models}")

        model_info = self.model_manager.models[model_id]
        if model_info.status != "loaded":
            # Start/load synchronously (may take time; keeps /responses semantics correct)
            # IMPORTANT: Model loading does blocking IO (docker, requests). Run it off the event loop.
            ok = await asyncio.to_thread(self.model_manager.load_model, model_id)
            if not ok:
                raise RuntimeError(f"Failed to load model '{model_id}'")

            ready = await self._wait_for_model_ready(model_id, timeout_s=180.0)
            if not ready:
                raise TimeoutError(f"Model '{model_id}' is still loading")

        backend_resp = await self._forward_request(model_id, data, request)
        if hasattr(backend_resp, "body"):
            return json.loads(backend_resp.body.decode("utf-8"))
        return backend_resp

    async def _wait_for_model_ready(self, model_id: str, timeout_s: float = 60.0) -> bool:
        """Wait until a model backend is ready to accept requests."""
        start = time.time()
        while time.time() - start < timeout_s:
            model_info = self.model_manager.models.get(model_id)
            if (
                model_info
                and self._check_container_running(model_info)
                and self._refresh_readiness(model_id, model_info)[0]
            ):
                # Status can drift (e.g., vLLM container becomes ready after initial load_model prompt probe fails).
                if model_info.status != "loaded":
                    model_info.status = "loaded"
                    try:
                        self.model_manager.update_access_time(model_id)
                    except Exception:
                        pass
                return True
            await asyncio.sleep(0.25)
        return False

    async def _process_chat_request(self, data: Dict, request: Request) -> Dict:
        """Process a chat request internally, returning the response data.

        This is the core logic from chat_completions but returns data instead of HTTP response.
        """
        # Normalize request to handle extra fields gracefully
        data = normalize_openai_chat_request(data)

        # Enhanced logging for OpenCode/ai-sdk requests
        user_agent = request.headers.get("User-Agent", "")
        if "ai-sdk" in user_agent or "opencode" in user_agent.lower():
            logger.info(f"[OPENCODE-DEBUG] Processing chat request")
            # Note: Removed json.dumps logging to avoid scoping issues

        # Track request for watchdog monitoring
        self.request_counter += 1
        request_id = f"req_{self.request_counter}_{int(asyncio.get_event_loop().time())}"
        start_time = asyncio.get_event_loop().time()
        self.active_requests[request_id] = start_time

        # Store request for streaming response
        self._current_request = request

        try:
            # Validate model
            model_id = data.get("model", "llama-3.1-8b-q4k-q4_k")
            if model_id not in self.model_manager.models:
                available_models = list(self.model_manager.models.keys())
                raise ValueError(f"Model '{model_id}' not available. Available models: {available_models}")

            model_info = self.model_manager.models[model_id]

            # Check if model is loaded
            if model_info.status != "loaded":
                # Model is not loaded, check if it's loading or needs to be loaded
                if model_info.status == "loading":
                    # Model is already being loaded, queue the request
                    request_id = self.request_queue.enqueue(
                        model_id=model_id,
                        request_data=data,
                        callback=lambda: self._forward_request(model_id, data, request)
                    )

                    return {
                        "status": "loading",
                        "message": f"Model {model_id} is being loaded. Request queued.",
                        "request_id": request_id,
                        "model_id": model_id
                    }
                else:
                    # Model needs to be loaded
                    # Queue the request and start loading in background
                    request_id = self.request_queue.enqueue(
                        model_id=model_id,
                        request_data=data,
                        callback=lambda: self._forward_request(model_id, data, request)
                    )

                    # Start background loading
                    self.model_manager.load_model(model_id)

                    return {
                        "status": "loading",
                        "message": f"Model {model_id} is being loaded. Request queued.",
                        "request_id": request_id,
                        "model_id": model_id
                    }

            # Forward request to backend
            response = await self._forward_request(model_id, data, request)

            # Clean up request tracking on completion
            if request_id in self.active_requests:
                del self.active_requests[request_id]

            # Extract JSON from response
            if hasattr(response, 'body'):
                return json.loads(response.body.decode('utf-8'))
            else:
                return response

        except ValueError as e:
            if "JSON" in str(e):
                raise ValueError("Invalid JSON in request")
            else:
                raise
        except Exception as e:
            logger.error(f"Error processing chat request: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.error(f"Request data at error: {data}")
            # Clean up request tracking on error
            if 'request_id' in locals() and request_id in self.active_requests:
                del self.active_requests[request_id]
            raise

    def _map_responses_to_chat(self, responses_data: Dict) -> Dict:
        """Map OpenAI Responses API request to chat completions format.

        v0: single-shot mapping; forwards ``tools``, ``tool_choice``, and related
        chat fields so vLLM receives the same tool contract as chat completions.
        Raises exception for complex requests that can't be handled.
        """
        # Check for unsupported complex features
        if responses_data.get("response_format") and responses_data["response_format"] != "text":
            raise ValueError("Structured output formats not yet supported in Responses API v0")

        if responses_data.get("max_steps") and responses_data["max_steps"] > 1:
            raise ValueError("Multi-step responses not yet supported in Responses API v0")

        chat_request = {
            "model": responses_data.get("model", "llama-3.1-8b-q4k-q4_k"),
            "messages": [],
            "stream": responses_data.get("stream", False)
        }

        # Handle input field -> messages
        input_data = responses_data.get("input")
        if input_data:
            if isinstance(input_data, str):
                # Simple string input
                chat_request["messages"].append({
                    "role": "user",
                    "content": input_data
                })
            elif isinstance(input_data, list):
                # Array input - for v0, treat as multiple user messages
                for item in input_data:
                    if isinstance(item, str):
                        chat_request["messages"].append({
                            "role": "user",
                            "content": item
                        })
                    elif isinstance(item, dict) and "content" in item:
                        # Sanitize assistant history: strip <think> blocks so prompts don't balloon.
                        try:
                            role = item.get("role")
                            content = item.get("content")
                            if role == "assistant" and isinstance(content, str) and "<think>" in content:
                                _, visible = self._split_think_block(content)
                                item = dict(item)
                                item["content"] = visible
                        except Exception:
                            pass
                        # Already in message format
                        chat_request["messages"].append(item)

        # Handle instructions -> system message
        instructions = responses_data.get("instructions")
        if instructions:
            chat_request["messages"].insert(0, {
                "role": "system",
                "content": instructions
            })

        # Qwen models are prone to verbose <think> outputs which slow requests and exceed n8n's timeout.
        # Add a system constraint to encourage direct answers.
        model_id = chat_request.get("model", "")
        if isinstance(model_id, str) and model_id.startswith("qwen"):
            constraint = (
                "Respond with the final answer only. Do not include <think> tags or internal reasoning. "
                "Be concise."
            )
            if chat_request["messages"] and chat_request["messages"][0].get("role") == "system":
                chat_request["messages"][0]["content"] = f"{chat_request['messages'][0].get('content','')}\n\n{constraint}"
            else:
                chat_request["messages"].insert(0, {"role": "system", "content": constraint})

        if "max_output_tokens" in responses_data:
            chat_request["max_tokens"] = responses_data["max_output_tokens"]
        elif "max_completion_tokens" in responses_data:
            chat_request["max_tokens"] = responses_data["max_completion_tokens"]

        for key in (
            "temperature",
            "reasoning_effort",
            "chat_template_kwargs",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "stream_options",
            "metadata",
            "modalities",
            "verbosity",
            "service_tier",
            "store",
            "prediction",
            "web_search_options",
            "audio",
            "seed",
            "user",
            "frequency_penalty",
            "presence_penalty",
            "top_p",
            "logprobs",
            "top_logprobs",
            "n",
            "stop",
            "response_format",
            "prompt_cache_key",
            "safety_identifier",
        ):
            if key in responses_data:
                chat_request[key] = responses_data[key]

        return chat_request

    def _map_chat_to_responses(self, original_request: Dict, chat_response: Dict, request=None) -> Dict:
        """Map chat completions response into a Responses API shaped payload.

        n8n Chat Hub uses the OpenAI Responses API internally and expects:
        - `output` to be an iterable list
        - output items to contain `content` blocks with `text`

        We keep a compatibility `choices` field as well (hybrid), so any components
        still expecting Chat Completions style can continue working.
        """
        import time

        # Preserve a few useful non-standard fields from internal processing (e.g., "loading" status)
        passthrough = {}
        if isinstance(chat_response, dict):
            for key in ("status", "message", "request_id", "model_id"):
                if key in chat_response:
                    passthrough[key] = chat_response[key]

        # Extract assistant visible text. Do NOT include reasoning/thinking in output_text because
        # n8n Chat Hub should display only the assistant response.
        assistant_text = ""
        reasoning_text = ""
        choices = []
        usage = {}

        if isinstance(chat_response, dict):
            choices = chat_response.get("choices") or []
            usage = chat_response.get("usage") or {}

        if isinstance(choices, list) and choices:
            msg = (choices[0] or {}).get("message") or {}
            content_raw = (msg.get("content") or "")
            reasoning_raw = (msg.get("reasoning_content") or "")

            # Qwen/vLLM style: <think>...</think> inside content
            if isinstance(content_raw, str):
                q_reasoning, q_visible = self._split_think_block(content_raw)
                if q_reasoning:
                    reasoning_text = q_reasoning
                content_clean = q_visible
            else:
                content_clean = ""

            # DeepSeek/llama.cpp style: reasoning_content separate
            if isinstance(reasoning_raw, str) and reasoning_raw.strip():
                reasoning_text = reasoning_raw.strip()

            # Prefer visible content; fall back to reasoning only if content is empty
            if isinstance(content_clean, str) and content_clean.strip():
                assistant_text = content_clean.strip()
            elif reasoning_text:
                assistant_text = reasoning_text

        # Build Responses API output items
        output_items = []
        if assistant_text:
            output_items = [
                {
                    "id": f"msg_{chat_response.get('id', 'unknown')}",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": assistant_text,
                        }
                    ],
                }
            ]

        # Map usage to Responses API token naming
        responses_usage = {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

        response_data = {
            **passthrough,
            "id": f"resp_{(chat_response or {}).get('id', 'unknown')}",
            "object": "response",
            "created": (chat_response or {}).get("created", int(time.time())),
            # Prefer the client-facing model id (what n8n selected) over backend internal model path.
            "model": original_request.get("model", (chat_response or {}).get("model", "unknown")),
            "output": output_items,
            "output_text": assistant_text,
            "usage": responses_usage,
        }

        return response_data

    async def _forward_request(
        self,
        model_id: str,
        data: Dict,
        request: Request,
        *,
        inbound_raw: Optional[Dict] = None,
        normalization_notes: Optional[Dict] = None,
        route_label: str = "POST /v1/chat/completions",
    ) -> Response:
        """Forward request to backend and track memory usage"""
        # Track model access for memory management
        self.memory_manager.update_access_time(model_id)
        model_info = self.model_manager.models[model_id]
        self._active_model_requests[model_id] += 1
        admission_sem: Optional[asyncio.Semaphore] = None
        long_prefill_sem: Optional[asyncio.Semaphore] = None
        long_prefill_acquired = False
        class_sem: Optional[asyncio.Semaphore] = None
        class_acquired = False
        est_vllm: Optional[int] = None
        admission_slot: Optional[str] = None

        # Remap model ID for different backends
        client_normalized = dict(data)
        data = data.copy()  # Don't modify original
        if model_info.backend == ModelBackend.VLLM:
            # vLLM uses "/app/model" as the model ID when mounting local directories
            data["model"] = "/app/model"
        elif model_info.backend == ModelBackend.LLAMACPP:
            # OpenCode/AI-SDK sends tool-calling fields that llama.cpp rejects unless started with --jinja.
            # For now, strip these fields so OpenCode can run against llama.cpp without requiring server flags.
            # (This enables basic chat/code responses; full tool-calling can be enabled later by starting llama-server with --jinja.)
            for key in ("tools", "tool_choice", "stream_options"):
                if key in data:
                    data.pop(key, None)

            # llama.cpp uses the full path to the GGUF file as model ID
            workspace_dir = Path(os.environ.get("EXO_WORKSPACE", REPO_ROOT))
            model_path = Path(workspace_dir) / model_info.model_path
            # For GGUF models, find the actual .gguf file if path is a directory
            if model_path.is_dir():
                gguf_files = list(model_path.glob("*.gguf"))
                if gguf_files:
                    # Use the same logic as in model_manager
                    preferred_order = ["Q8_0", "Q6_K", "Q4_K_M", "Q4_K"]
                    for pref in preferred_order:
                        for gguf_file in gguf_files:
                            if pref in gguf_file.name:
                                model_path = gguf_file
                                break
                        if model_path.suffix == ".gguf":
                            break
                    if model_path.suffix != ".gguf":
                        model_path = gguf_files[0]
            data["model"] = str(model_path)

        # Forward to backend
        backend_url = f"http://localhost:{model_info.port}/v1/chat/completions"
        
        # Check if container is actually running and ready before forwarding
        if not self._check_container_running(model_info):
            logger.warning(f"Container {model_info.container_name} is not running")
            model_info.status = "unloaded"
            self._emit_admission_failure_trace(
                request,
                "POST /v1/chat/completions",
                failure_class="admission_not_ready",
                model_id=model_id,
                admission_kind="container_down",
                model_info=model_info,
                container_running=False,
            )
            return web.json_response(
                {
                    "error": f"Model {model_id} container is not running.",
                    "status": "unloaded",
                    "message": "Please wait for the model to load, then retry.",
                    "output": [],
                    "choices": [],
                },
                status=503
            )

        if model_info.backend == ModelBackend.VLLM:
            est_vllm = estimate_chat_prompt_tokens(data.get("messages"), data.get("tools"))
            admission_slot = admission_class_slot(est_vllm)
            if admission_slot == "heavy":
                if getattr(model_info, "readiness_ever_served", False):
                    streak_need = circuit_block_heavy_fail_streak_default()
                    if (
                        model_info.consecutive_readiness_failures >= streak_need
                        or self._heavy_circuit_blocks_heavy_admission(model_info)
                    ):
                        rid = get_or_create_request_id(request)
                        log_failure_taxonomy(
                            rid,
                            "POST /v1/chat_completions",
                            "admission_circuit_open",
                            api="v1/chat_completions",
                            model_id=model_id,
                            admission_class=classify_admission_class(est_vllm),
                            consecutive_readiness_failures=model_info.consecutive_readiness_failures,
                        )
                        self._emit_admission_failure_trace(
                            request,
                            "POST /v1/chat/completions",
                            failure_class="admission_circuit_open",
                            model_id=model_id,
                            admission_kind="readiness_streak",
                            model_info=model_info,
                            container_running=True,
                        )
                        try:
                            self._active_model_requests[model_id] -= 1
                            if self._active_model_requests[model_id] <= 0:
                                self._active_model_requests.pop(model_id, None)
                        except Exception:
                            pass
                        return web.json_response(
                            {
                                "error": {
                                    "message": (
                                        f"Model {model_id} recently failed readiness; long-context "
                                        "admissions are paused until the engine recovers."
                                    ),
                                    "type": "engine_unhealthy",
                                    "code": "admission_circuit_open",
                                },
                                "output": [],
                                "choices": [],
                            },
                            status=503,
                        )

        need_strict_health = (
            model_info.backend == ModelBackend.VLLM
            and est_vllm is not None
            and strict_health_required_for_estimate(est_vllm)
        )
        ready, deny_reason = self._refresh_readiness(
            model_id, model_info, require_full_health=need_strict_health
        )
        if not ready:
            degraded = model_info.status == "loaded"
            self._emit_admission_failure_trace(
                request,
                "POST /v1/chat/completions",
                failure_class="admission_not_ready",
                model_id=model_id,
                admission_kind=(
                    "heavy_requires_health"
                    if deny_reason == "heavy_requires_health_200"
                    else ("engine_degraded" if degraded else "initializing")
                ),
                model_info=model_info,
                container_running=True,
            )
            body = self._json_model_not_ready(
                model_id, model_info, container_running=True, degraded=degraded
            )
            if deny_reason == "heavy_requires_health_200" and isinstance(body.get("error"), dict):
                em = str(body["error"].get("message") or "")
                body["error"]["message"] = (
                    em
                    + " Estimated long prompt requires a healthy /health endpoint; "
                    "the backend is returning non-200 on /health."
                ).strip()
                body["error"]["code"] = "heavy_admission_blocked"
            return web.json_response(body, status=503)

        self._note_tool_runtime_from_messages(request, data.get("messages"), "POST /v1/chat_completions")

        if model_info.backend == ModelBackend.VLLM:
            vllm_cfg = self._get_vllm_prefill_yaml(model_id)
            max_ml = int(vllm_cfg.get("max_model_len", 0) or 0)
            est = est_vllm if est_vllm is not None else estimate_chat_prompt_tokens(
                data.get("messages"), data.get("tools")
            )
            first_turn = is_first_chat_turn(data.get("messages"))
            thr = self._long_prefill_threshold_tokens(model_id)
            is_long_prefill = est >= thr
            reserve = self._prompt_completion_reserve(data)
            margin = self._prompt_budget_margin()
            if max_ml > 0:
                budget = max_ml - reserve - margin
                if budget < 1:
                    budget = 1
                if est > budget:
                    rid = get_or_create_request_id(request)
                    log_failure_taxonomy(
                        rid,
                        "POST /v1/chat/completions",
                        "admission_prompt_too_large",
                        api="v1/chat_completions",
                        model_id=model_id,
                        estimated_prompt_tokens=est,
                        max_model_len=max_ml,
                        completion_reserve=reserve,
                        budget_margin=margin,
                        allowed_prompt_budget=budget,
                    )
                    try:
                        self._active_model_requests[model_id] -= 1
                        if self._active_model_requests[model_id] <= 0:
                            self._active_model_requests.pop(model_id, None)
                    except Exception:
                        pass
                    return web.json_response(
                        {
                            "error": {
                                "message": (
                                    f"Estimated prompt tokens ({est}) exceed allowed budget ({budget}) "
                                    f"for max_model_len={max_ml} with completion reserve={reserve}."
                                ),
                                "type": "invalid_request_error",
                                "code": "context_length_exceeded",
                            },
                            "output": [],
                            "choices": [],
                        },
                        status=400,
                    )
            if admission_classes_enabled() and admission_slot is not None:
                class_sem = self._ensure_vllm_class_sem(model_id, admission_slot)
                await class_sem.acquire()
                class_acquired = True
            t_adm_set = time.monotonic()
            long_prefill_sem = self._ensure_long_prefill_sem(model_id)
            if is_long_prefill and long_prefill_sem is not None:
                await long_prefill_sem.acquire()
                long_prefill_acquired = True
                self._long_prefill_active[model_id] += 1
            admission_sem = self._ensure_vllm_admission_sem(model_id)
            await admission_sem.acquire()
            queue_wait_ms = (time.monotonic() - t_adm_set) * 1000.0
            log_prefill_admission(
                get_or_create_request_id(request),
                "POST /v1/chat/completions",
                estimated_prompt_tokens=est,
                is_first_turn=first_turn,
                is_long_prefill=is_long_prefill,
                long_prefill_threshold=thr,
                queue_wait_ms=round(queue_wait_ms, 3),
                long_prefill_slots_in_use=int(self._long_prefill_active[model_id]),
                model_id=model_id,
                health_failures=model_info.consecutive_readiness_failures,
                restart_reason=model_info.last_engine_recreate_reason or None,
                admission_class=classify_admission_class(est),
                admission_slot=admission_slot or "-",
            )

        backend_snapshot = json.loads(json.dumps(data))
        rid_fw = get_or_create_request_id(request)
        log_normalized_client_payload(
            rid_fw,
            "v1/chat_completions",
            route_label,
            client_normalized,
            dict(normalization_notes or {}),
        )
        log_backend_request_payload(
            rid_fw, "v1/chat_completions", route_label, backend_url, backend_snapshot
        )

        def _tool_def_count(d: Dict) -> int:
            t = d.get("tools")
            return len(t) if isinstance(t, list) else 0

        _tc = _tool_def_count(client_normalized)
        _tb = _tool_def_count(backend_snapshot)
        if _tc or _tb:
            logger.info(
                "[tools_forward] request_id=%s client_tool_defs=%s backend_tool_defs=%s ok=%s",
                rid_fw,
                _tc,
                _tb,
                _tc == _tb,
            )

        try:
            async with ClientSession() as session:
                # Check if streaming is requested
                stream = data.get("stream", False)
                
                if stream:
                    async with session.post(
                        backend_url,
                        json=data,
                        timeout=300
                    ) as resp:
                        if resp.status >= 400:
                            err_body = await resp.text()
                            return web.json_response(
                                {
                                    "error": f"Backend error: {err_body[:1200]}",
                                    "output": [],
                                    "choices": [],
                                },
                                status=502,
                            )

                        rid = get_or_create_request_id(request)
                        ledger = StreamCommittedLedger(
                            request_id=rid,
                            idempotency_key=extract_idempotency_key(client_normalized),
                        )
                        response = web.StreamResponse(status=200)
                        response.content_type = resp.content_type or "text/event-stream"
                        for header_name, header_value in resp.headers.items():
                            if header_name.lower() not in ["content-length", "transfer-encoding"]:
                                response.headers[header_name] = header_value
                        stream_prepared = False

                        async def ensure_prepared() -> bool:
                            nonlocal stream_prepared
                            if stream_prepared:
                                return True
                            await response.prepare(request)
                            stream_prepared = True
                            return True

                        pa = ProductionAuditSession(
                            request_id=rid,
                            route=route_label,
                            api="v1/chat_completions",
                            inbound_raw=inbound_raw,
                            normalized_client=client_normalized,
                            normalization_delta=dict(normalization_notes or {}),
                            backend_payload=backend_snapshot,
                            backend_url=backend_url,
                            wants_tools=bool(data.get("tools")),
                            chat_endpoint_used=True,
                            responses_endpoint_used=False,
                            responses_tools_forwarded=None,
                            backend_health_state=getattr(
                                model_info, "last_readiness_probe_class", None
                            )
                            or model_info.status,
                            admission_decision="forwarded",
                        )
                        pa.normalization_anthropic_stripped = list(
                            (normalization_notes or {}).get("anthropic_stripped", [])
                        )
                        tool_audit = ToolStreamAudit(api="v1/chat_completions")
                        sanitizer = ReasoningXmlSanitizer()
                        had_backend_tool_delta = False
                        wants_tools = bool(data.get("tools"))
                        buf = b""
                        seq_b = 0
                        seq_o = 0
                        sse_tw = get_sse_trace_writer()
                        finish_reason: Optional[str] = None

                        async def write_to_client(blob: bytes) -> bool:
                            await ensure_prepared()
                            st = await sse_write(response, blob)
                            if st == "client_disconnect":
                                return False
                            if blob.startswith(b"data:") and b"[DONE]" not in blob:
                                ledger.first_byte_sent = True
                            return True

                        def note_tool_ledger(delta_obj: Dict[str, Any], fr: Optional[str]) -> None:
                            tcalls = delta_obj.get("tool_calls")
                            if isinstance(tcalls, list) and tcalls:
                                for tc in tcalls:
                                    idx = tc.get("index")
                                    ledger.last_tool_event = {
                                        "index": int(idx) if idx is not None else 0,
                                        "partial": True,
                                    }
                            if fr == "tool_calls" and ledger.last_tool_event:
                                ledger.last_tool_event = {
                                    **ledger.last_tool_event,
                                    "partial": False,
                                }

                        async def close_stream_tail(*, emit_done: bool = True) -> bool:
                            nonlocal seq_o
                            r_tail, v_tail = sanitizer.finalize()
                            log_sanitizer_eof(rid, sanitizer.trace_state())
                            tool_audit.sanitizer_eof = sanitizer.trace_state()
                            extra_obj = None
                            if (r_tail and r_tail.strip()) or (v_tail and v_tail.strip()):
                                extra_obj = {
                                    "id": "",
                                    "object": "chat.completion.chunk",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                dlt2: Dict[str, Any] = {}
                                if v_tail and v_tail.strip():
                                    dlt2["content"] = v_tail
                                    ledger.committed_text_offset += len(v_tail)
                                if r_tail and r_tail.strip():
                                    dlt2["reasoning_content"] = r_tail
                                    ledger.committed_reasoning_offset += len(r_tail)
                                extra_obj["choices"][0]["delta"] = dlt2
                            if extra_obj:
                                out = (
                                    b"data: "
                                    + json.dumps(extra_obj, ensure_ascii=False).encode("utf-8")
                                    + b"\n\n"
                                )
                                if not await write_to_client(out):
                                    return False
                                seq_o += 1
                                tool_audit.note_proxy_chat_out(extra_obj)
                                log_proxy_outbound_delta(
                                    rid,
                                    "v1/chat_completions",
                                    route_label,
                                    sequence=seq_o,
                                    surface="chat.completion.chunk",
                                    emitted=canonical_backend_chat_sse(extra_obj),
                                    projection=projection_chat_outbound_event(extra_obj),
                                )
                            pa.finish_reason = finish_reason
                            await self._finalize_chat_stream_production_audit(
                                pa=pa,
                                tool_audit=tool_audit,
                                sanitizer=sanitizer,
                                wants_tools=wants_tools,
                                had_backend_tool_delta=had_backend_tool_delta,
                                http_status=resp.status,
                                stream_response=response,
                                route=route_label,
                            )
                            if emit_done:
                                if not await write_to_client(b"data: [DONE]\n\n"):
                                    return False
                            return True

                        saw_done = False
                        client_write_closed = False
                        idle_to = stream_idle_timeout_seconds()
                        stream_had_exc: Optional[BaseException] = None
                        idle_timeout = False
                        try:
                            async for chunk in iter_chunked_with_idle(
                                resp.content, 8192, idle_to
                            ):
                                if not chunk:
                                    continue
                                buf += chunk
                                while b"\n" in buf:
                                    raw_line, buf = buf.split(b"\n", 1)
                                    line = raw_line.strip()
                                    if not line:
                                        continue
                                    if not line.startswith(b"data:"):
                                        if not await write_to_client(raw_line + b"\n"):
                                            client_write_closed = True
                                            break
                                        continue
                                    payload = line[5:].strip()
                                    if payload == b"[DONE]":
                                        saw_done = True
                                        if await close_stream_tail(emit_done=True):
                                            return response
                                        return response
                                    try:
                                        obj = json.loads(payload.decode("utf-8"))
                                    except Exception:
                                        if not await write_to_client(line + b"\n\n"):
                                            client_write_closed = True
                                            break
                                        continue
                                    try:
                                        choice0 = (obj.get("choices") or [])[0] or {}
                                        delta_obj = choice0.get("delta") or {}
                                        fr = choice0.get("finish_reason")
                                        if fr:
                                            finish_reason = fr
                                        if delta_obj.get("tool_calls"):
                                            had_backend_tool_delta = True
                                            tool_audit.note_backend_delta(delta_obj)
                                        for rk in ("reasoning_content", "reasoning"):
                                            rr = delta_obj.get(rk)
                                            if isinstance(rr, str) and raw_visible_suggests_tool_calls(rr):
                                                pa.reasoning_had_tool_like_markup = True
                                        seq_b += 1
                                        try:
                                            trace_line = raw_line.decode("utf-8", errors="replace").strip()
                                        except Exception:
                                            trace_line = ""
                                        if trace_line:
                                            sse_tw.write(rid, trace_line[: sse_trace_line_cap()])
                                        log_backend_sse_delta(
                                            rid,
                                            "v1/chat_completions",
                                            route_label,
                                            sequence=seq_b,
                                            backend_url=backend_url,
                                            raw_sse_bytes=len(raw_line),
                                            parsed=canonical_backend_chat_sse(obj),
                                            projection=projection_backend_delta(
                                                delta_obj,
                                                choice0.get("finish_reason"),
                                            ),
                                        )
                                        dc = delta_obj.get("content")
                                        if isinstance(dc, str) and dc:
                                            tool_audit.note_raw_content(dc)
                                            r_part, v_part = sanitizer.feed_content(dc)
                                            pa.merge_sanitizer_audit(sanitizer.take_audit_events())
                                            if r_part and raw_visible_suggests_tool_calls(r_part):
                                                pa.reasoning_had_tool_like_markup = True
                                            if v_part and raw_visible_suggests_tool_calls(v_part):
                                                pa.content_had_tool_like_markup = True
                                            if r_part or (v_part is not None and v_part != dc):
                                                log_proxy_sanitizer_transition(
                                                    rid,
                                                    "v1/chat_completions",
                                                    route_label,
                                                    sequence=seq_b,
                                                    before={"delta_content": dc},
                                                    after={
                                                        "reasoning_delta": r_part,
                                                        "visible_delta": v_part,
                                                    },
                                                    phase="reasoning_xml",
                                                )
                                            new_delta = dict(delta_obj)
                                            if r_part:
                                                prev_rc = new_delta.get("reasoning_content")
                                                new_delta["reasoning_content"] = (
                                                    (prev_rc + r_part)
                                                    if isinstance(prev_rc, str)
                                                    else r_part
                                                )
                                                ledger.committed_reasoning_offset += len(r_part)
                                            if v_part is not None:
                                                new_delta["content"] = v_part
                                                if v_part:
                                                    ledger.committed_text_offset += len(v_part)
                                            else:
                                                new_delta["content"] = ""
                                            keep = bool(
                                                new_delta.get("tool_calls")
                                                or new_delta.get("content")
                                                or new_delta.get("reasoning_content")
                                                or new_delta.get("function_call")
                                            )
                                            note_tool_ledger(new_delta, fr)
                                            if keep:
                                                obj_out = dict(obj)
                                                choices = list(obj_out.get("choices") or [])
                                                if choices:
                                                    nc = dict(choice0)
                                                    nc["delta"] = new_delta
                                                    choices[0] = nc
                                                    obj_out["choices"] = choices
                                                out = (
                                                    b"data: "
                                                    + json.dumps(
                                                        obj_out, ensure_ascii=False
                                                    ).encode("utf-8")
                                                    + b"\n\n"
                                                )
                                                if not await write_to_client(out):
                                                    client_write_closed = True
                                                    break
                                                seq_o += 1
                                                tool_audit.note_proxy_chat_out(obj_out)
                                                if new_delta.get("tool_calls"):
                                                    pa.proxy_emitted_structured_tool = True
                                                log_proxy_outbound_delta(
                                                    rid,
                                                    "v1/chat_completions",
                                                    route_label,
                                                    sequence=seq_o,
                                                    surface="chat.completion.chunk",
                                                    emitted=canonical_backend_chat_sse(obj_out),
                                                    projection=projection_chat_outbound_event(obj_out),
                                                )
                                        else:
                                            note_tool_ledger(delta_obj, fr)
                                            rc = delta_obj.get("reasoning_content")
                                            rr = delta_obj.get("reasoning")
                                            if isinstance(rc, str) and rc:
                                                ledger.committed_reasoning_offset += len(rc)
                                            elif isinstance(rr, str) and rr:
                                                ledger.committed_reasoning_offset += len(rr)
                                            dc2 = delta_obj.get("content")
                                            if isinstance(dc2, str) and dc2:
                                                ledger.committed_text_offset += len(dc2)
                                            out_raw = (
                                                b"data: "
                                                + json.dumps(obj, ensure_ascii=False).encode("utf-8")
                                                + b"\n\n"
                                            )
                                            if not await write_to_client(out_raw):
                                                client_write_closed = True
                                                break
                                            seq_o += 1
                                            tool_audit.note_proxy_chat_out(obj)
                                            if delta_obj.get("tool_calls"):
                                                pa.proxy_emitted_structured_tool = True
                                            log_proxy_outbound_delta(
                                                rid,
                                                "v1/chat_completions",
                                                route_label,
                                                sequence=seq_o,
                                                surface="chat.completion.chunk",
                                                emitted=canonical_backend_chat_sse(obj),
                                                projection=projection_chat_outbound_event(obj),
                                            )
                                    except Exception:
                                        if not await write_to_client(
                                            b"data: "
                                            + json.dumps(obj, ensure_ascii=False).encode("utf-8")
                                            + b"\n\n"
                                        ):
                                            client_write_closed = True
                                            break
                                if client_write_closed:
                                    break
                        except StreamIdleTimeoutError as e:
                            stream_had_exc = e
                            idle_timeout = True
                        except ClientError as e:
                            stream_had_exc = e

                        if saw_done:
                            return response

                        if client_write_closed:
                            sanitizer.finalize()
                            tool_audit.sanitizer_eof = sanitizer.trace_state()
                            pa.finish_reason = finish_reason
                            await self._finalize_chat_stream_production_audit(
                                pa=pa,
                                tool_audit=tool_audit,
                                sanitizer=sanitizer,
                                wants_tools=wants_tools,
                                had_backend_tool_delta=had_backend_tool_delta,
                                http_status=resp.status,
                                stream_response=response,
                                route=route_label,
                            )
                            return response

                        if not stream_prepared and not ledger.first_byte_sent:
                            sanitizer.finalize()
                            tool_audit.sanitizer_eof = sanitizer.trace_state()
                            pa.finish_reason = finish_reason
                            await self._finalize_chat_stream_production_audit(
                                pa=pa,
                                tool_audit=tool_audit,
                                sanitizer=sanitizer,
                                wants_tools=wants_tools,
                                had_backend_tool_delta=had_backend_tool_delta,
                                http_status=502,
                                stream_response=response,
                                route=route_label,
                            )
                            return web.json_response(
                                {
                                    "error": "Upstream closed before streaming began or idle timeout.",
                                    "output": [],
                                    "choices": [],
                                    "stream_ledger": ledger.to_audit_dict(),
                                },
                                status=502,
                            )

                        term = classify_stream_termination(
                            exc=stream_had_exc,
                            saw_done=False,
                            mid_stream=ledger.first_byte_sent,
                            idle_timeout=idle_timeout,
                        )
                        if term == "client_disconnect":
                            r_tail, v_tail = sanitizer.finalize()
                            tool_audit.sanitizer_eof = sanitizer.trace_state()
                            pa.finish_reason = finish_reason
                            await self._finalize_chat_stream_production_audit(
                                pa=pa,
                                tool_audit=tool_audit,
                                sanitizer=sanitizer,
                                wants_tools=wants_tools,
                                had_backend_tool_delta=had_backend_tool_delta,
                                http_status=resp.status,
                                stream_response=response,
                                route=route_label,
                            )
                            return response
                        if should_trip_admission_on_classify(term):
                            self._trip_heavy_admission_circuit(model_id)
                        await self._emit_abnormal_stream_terminal(
                            response, ledger, term, request_json=client_normalized
                        )
                        sanitizer.finalize()
                        tool_audit.sanitizer_eof = sanitizer.trace_state()
                        pa.finish_reason = finish_reason
                        await self._finalize_chat_stream_production_audit(
                            pa=pa,
                            tool_audit=tool_audit,
                            sanitizer=sanitizer,
                            wants_tools=wants_tools,
                            had_backend_tool_delta=had_backend_tool_delta,
                            http_status=502,
                            stream_response=response,
                            route=route_label,
                        )
                        return response
                else:
                    async with session.post(
                        backend_url,
                        json=data,
                        timeout=300
                    ) as resp:
                        response_data = await resp.json()
                        response_data.pop("output", None)
                        choices = response_data.get("choices") or []
                        wants_tools_ns = bool(data.get("tools"))
                        rid_ns = get_or_create_request_id(request)
                        had_tc_ns = False
                        content_like_ns = False
                        reasoning_like_ns = False
                        finish_fr_ns: Optional[str] = None
                        if choices and isinstance(choices[0], dict):
                            ch0 = choices[0] or {}
                            finish_fr_ns = ch0.get("finish_reason")
                            msg = ch0.get("message")
                            if isinstance(msg, dict):
                                raw_c = msg.get("content")
                                raw_r = msg.get("reasoning_content") or msg.get("reasoning")
                                if isinstance(raw_c, str) and raw_visible_suggests_tool_calls(raw_c):
                                    content_like_ns = True
                                if isinstance(raw_r, str) and raw_visible_suggests_tool_calls(raw_r):
                                    reasoning_like_ns = True
                                self._sanitize_assistant_message_inplace(msg)
                                tcl = msg.get("tool_calls") or []
                                had_tc_ns = bool(tcl)
                        ta_ns = ToolStreamAudit(api="v1/chat_completions")
                        if had_tc_ns:
                            ta_ns.backend_chunks_with_tool_calls = 1
                            ta_ns.proxy_emitted_chat_tool_chunks = 1
                        rec_ns = ta_ns.build_record(
                            rid_ns,
                            content_had_tool_like_markup=content_like_ns,
                            reasoning_had_tool_like_markup=reasoning_like_ns,
                        )
                        pa_ns = ProductionAuditSession(
                            request_id=rid_ns,
                            route=route_label,
                            api="v1/chat_completions",
                            inbound_raw=inbound_raw,
                            normalized_client=client_normalized,
                            normalization_delta=dict(normalization_notes or {}),
                            backend_payload=backend_snapshot,
                            backend_url=backend_url,
                            wants_tools=wants_tools_ns,
                            had_backend_tool_delta=had_tc_ns,
                            proxy_emitted_structured_tool=had_tc_ns,
                            content_had_tool_like_markup=content_like_ns,
                            reasoning_had_tool_like_markup=reasoning_like_ns,
                            chat_endpoint_used=True,
                            responses_endpoint_used=False,
                            responses_tools_forwarded=None,
                            finish_reason=finish_fr_ns,
                            backend_health_state=getattr(
                                model_info, "last_readiness_probe_class", None
                            )
                            or model_info.status,
                            admission_decision="forwarded",
                            http_final_stream_status=resp.status,
                        )
                        pa_ns.normalization_anthropic_stripped = list(
                            (normalization_notes or {}).get("anthropic_stripped", [])
                        )
                        pa_ns.tool_audit_snapshot = rec_ns
                        log_request_tool_audit(rid_ns, rec_ns, api="v1/chat_completions", route=route_label)
                        code_ns, summary_ns = compute_tool_turn_root_cause(
                            wants_tools=wants_tools_ns,
                            had_backend_tool_delta=had_tc_ns,
                            proxy_emitted_structured_tool=had_tc_ns,
                            content_tool_like=content_like_ns,
                            reasoning_tool_like=reasoning_like_ns,
                            responses_endpoint_used=False,
                            responses_tools_forwarded=None,
                            http_stream_status=resp.status,
                            admission_failure_class=None,
                        )
                        pa_ns.emit(root_cause_code=code_ns, root_cause_summary=summary_ns)
                        if wants_tools_ns and code_ns not in (
                            "ok_structured_tools",
                            "ok_no_tools_requested",
                        ):
                            log_failure_taxonomy(
                                rid_ns,
                                route_label,
                                code_ns,
                                api="v1/chat_completions",
                                stream=False,
                            )
                        return web.json_response(response_data, status=resp.status)
        except ClientError as e:
            logger.error(f"Error forwarding request to {backend_url}: {e}")
            return web.json_response(
                {"error": f"Backend error: {str(e)}", "output": [], "choices": []},
                status=502
            )
        finally:
            if admission_sem is not None:
                admission_sem.release()
            if long_prefill_acquired and long_prefill_sem is not None:
                self._long_prefill_active[model_id] = max(0, self._long_prefill_active[model_id] - 1)
                long_prefill_sem.release()
            if class_acquired and class_sem is not None:
                class_sem.release()
            try:
                self._active_model_requests[model_id] -= 1
                if self._active_model_requests[model_id] <= 0:
                    self._active_model_requests.pop(model_id, None)
            except Exception:
                pass

    def _check_container_running(self, model_info) -> bool:
        """Check if container/process is actually running"""
        import subprocess
        import socket

        try:
            # For vLLM models, check Docker containers
            if model_info.backend == ModelBackend.VLLM:
                result = subprocess.run(
                    self.model_manager.docker_cmd + ["ps", "--filter", f"name={model_info.container_name}", "--format", "{{.Status}}"],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                return "Up" in result.stdout
            # For llama.cpp models, check if native process is listening on port
            elif model_info.backend == ModelBackend.LLAMACPP:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('127.0.0.1', model_info.port))
                sock.close()
                return result == 0  # 0 means connection successful (port is open)
            else:
                return False
        except:
            return False
    
    async def save_history(self, request: Request) -> Response:
        """Save chat message to history"""
        try:
            data = await request.json()
            
            message_id = self.chat_history.save_message(
                session_id=data.get("session_id"),
                message=data.get("message"),
                role=data.get("role"),
                user_id=data.get("user_id", "localuser"),
                context_ref=data.get("context_ref"),
                metadata=data.get("metadata")
            )
            
            return web.json_response({
                "message_id": message_id,
                "status": "saved"
            })
        except Exception as e:
            logger.error(f"Error saving history: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def get_history(self, request: Request) -> Response:
        """Get chat history for a session"""
        try:
            session_id = request.query.get("session_id")
            if not session_id:
                return web.json_response(
                    {"error": "session_id parameter required"},
                    status=400
                )
            
            limit = request.query.get("limit")
            limit = int(limit) if limit else None
            
            history = self.chat_history.get_history(session_id, limit=limit)
            
            return web.json_response({
                "session_id": session_id,
                "messages": history
            })
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def list_sessions(self, request: Request) -> Response:
        """List all chat sessions"""
        try:
            user_id = request.query.get("user_id")
            sessions = self.chat_history.list_sessions(user_id=user_id)
            
            return web.json_response({
                "sessions": sessions
            })
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def delete_session(self, request: Request) -> Response:
        """Delete a chat session"""
        try:
            session_id = request.match_info["session_id"]
            deleted = self.chat_history.delete_session(session_id)
            
            if deleted:
                return web.json_response({"status": "deleted"})
            else:
                return web.json_response(
                    {"error": "Session not found"},
                    status=404
                )
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def load_model(self, request: Request) -> Response:
        """Load a model"""
        try:
            model_id = request.match_info["model_id"]
            success = self.model_manager.load_model(model_id)
            
            if success:
                return web.json_response({
                    "status": "loading",
                    "model_id": model_id,
                    "message": "Model loading started"
                })
            else:
                return web.json_response(
                    {"error": f"Failed to load model {model_id}"},
                    status=500
                )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def unload_model(self, request: Request) -> Response:
        """Unload a model"""
        try:
            model_id = request.match_info["model_id"]
            success = self.model_manager.unload_model(model_id)
            
            if success:
                return web.json_response({
                    "status": "unloaded",
                    "model_id": model_id
                })
            else:
                return web.json_response(
                    {"error": f"Failed to unload model {model_id}"},
                    status=500
                )
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def model_status(self, request: Request) -> Response:
        """Get model status with detailed health information"""
        try:
            model_id = request.match_info["model_id"]
            
            if model_id not in self.model_manager.models:
                return web.json_response(
                    {"error": f"Model {model_id} not found"},
                    status=404
                )
            
            model_info = self.model_manager.models[model_id]
            
            # Check container health
            container_running = self._check_container_running(model_info)
            container_ready = (
                self._refresh_readiness(model_id, model_info)[0]
                if container_running
                else False
            )

            # Determine detailed status
            if model_info.status == "loading":
                if not container_running:
                    detailed_status = "container_starting"
                    message = "Container is starting..."
                elif not container_ready:
                    detailed_status = "model_initializing"
                    message = "Model is initializing (this may take 1-2 minutes)..."
                else:
                    detailed_status = "ready"
                    message = "Model is ready"
            elif model_info.status == "loaded":
                detailed_status = "ready" if container_ready else "degraded"
                message = "Model is ready" if container_ready else "Model loaded but container may be restarting"
            else:
                detailed_status = model_info.status
                message = f"Model is {model_info.status}"
            
            return web.json_response({
                "model_id": model_id,
                "status": model_info.status,
                "detailed_status": detailed_status,
                "message": message,
                "container_running": container_running,
                "container_ready": container_ready,
                "readiness": {
                    "last_ready_at": model_info.last_ready_at,
                    "first_not_ready_at": model_info.first_not_ready_at,
                    "consecutive_readiness_failures": model_info.consecutive_readiness_failures,
                    "last_probe_class": model_info.last_readiness_probe_class,
                    "readiness_ever_served": model_info.readiness_ever_served,
                    "last_engine_recreate_at": model_info.last_engine_recreate_at,
                    "last_engine_recreate_reason": model_info.last_engine_recreate_reason,
                },
                "port": model_info.port,
                "backend": model_info.backend.value,
                "last_access": model_info.last_access_time
            })
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _monitor_memory(self):
        """Background task to monitor GPU memory"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                if self.memory_monitor.check_memory_threshold():
                    # Memory threshold exceeded, unload least recently used model
                    lru_model_id = self.model_manager.get_least_recently_used_model()
                    
                    if lru_model_id:
                        logger.warning(
                            f"Memory threshold exceeded. Unloading least recently used model: {lru_model_id}"
                        )
                        self.model_manager.unload_model(lru_model_id)
                        
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}", exc_info=True)
                # Don't crash - keep monitoring
                await asyncio.sleep(10)
    
    async def _process_queues(self):
        """Background task to process request queues"""
        while True:
            try:
                await asyncio.sleep(2)  # Check every 2 seconds
                
                # Process queues for all models
                for model_id in self.model_manager.models.keys():
                    if self.request_queue.is_processing(model_id):
                        continue
                    
                    model_info = self.model_manager.models[model_id]
                    
                    # Only process if model is loaded
                    if model_info.status != "loaded":
                        continue
                    
                    # Process queued requests
                    while True:
                        queued_request = self.request_queue.dequeue(model_id)
                        if not queued_request:
                            break
                        
                        # Check timeout
                        if queued_request.timeout:
                            elapsed = time.time() - queued_request.created_at
                            if elapsed > queued_request.timeout:
                                logger.warning(
                                    f"Request {queued_request.request_id} timed out"
                                )
                                continue
                        
                        # Process request
                        try:
                            self.request_queue.process_queue(model_id)
                            await queued_request.callback()
                        except Exception as e:
                            logger.error(
                                f"Error processing queued request {queued_request.request_id}: {e}",
                                exc_info=True
                            )
                        finally:
                            self.request_queue.finish_processing(model_id)
                            
            except Exception as e:
                logger.error(f"Error in queue processing: {e}", exc_info=True)
                # Don't crash - wait and retry
                await asyncio.sleep(2)
    
    async def _maybe_auto_recreate_engine(self, model_id: str) -> None:
        raw = (os.environ.get("PROXY_ENGINE_AUTO_RECREATE") or "").strip().lower()
        if raw not in ("1", "true", "yes", "on"):
            return
        try:
            debounce = float(os.environ.get("PROXY_ENGINE_RECREATE_DEBOUNCE_S", "120"))
        except ValueError:
            debounce = 120.0
        try:
            min_fails = int(os.environ.get("PROXY_ENGINE_RECREATE_MIN_FAILS", "8"))
        except ValueError:
            min_fails = 8
        model_info = self.model_manager.models.get(model_id)
        if not model_info or model_info.backend != ModelBackend.VLLM:
            return
        now = time.time()
        if model_info.consecutive_readiness_failures < min_fails:
            return
        if now - model_info.last_engine_recreate_at < debounce:
            return
        with self._engine_recreate_locks[model_id]:
            t = self._engine_recreate_task.get(model_id)
            if t is not None and not t.done():
                return
            reason = (
                f"readiness_streak={model_info.consecutive_readiness_failures} "
                f"class={model_info.last_readiness_probe_class} "
                f"first_not_ready={model_info.first_not_ready_at} "
                f"last_ready={model_info.last_ready_at}"
            )
            model_info.last_engine_recreate_at = now
            model_info.last_engine_recreate_reason = reason
            logger.error(
                "[ENGINE_RECREATE] model=%s debounce=%ss min_fails=%s %s",
                model_id,
                debounce,
                min_fails,
                reason,
            )

            async def _run() -> None:
                try:
                    await asyncio.to_thread(self.model_manager.unload_model, model_id)
                    await asyncio.sleep(2.0)
                    await asyncio.to_thread(self.model_manager.load_model, model_id)
                except Exception as e:
                    logger.error("[ENGINE_RECREATE] failed model=%s err=%s", model_id, e, exc_info=True)
                finally:
                    self._engine_recreate_task.pop(model_id, None)

            self._engine_recreate_task[model_id] = asyncio.create_task(_run())

    async def _background_health_check(self):
        """Background task to periodically check container health and update model status."""
        while True:
            try:
                await asyncio.sleep(5)
                for model_id, model_info in self.model_manager.models.items():
                    if model_info.status not in ("loading", "loaded"):
                        continue
                    if not self._check_container_running(model_info):
                        continue
                    ready = self._refresh_readiness(model_id, model_info)[0]
                    if ready and model_info.status == "loading":
                        logger.info(
                            f"Model {model_id} container is now ready - updating status to loaded"
                        )
                        model_info.status = "loaded"
                        model_info.last_access_time = time.time()
                    elif (
                        not ready
                        and model_info.status == "loaded"
                        and model_info.readiness_ever_served
                    ):
                        await self._maybe_auto_recreate_engine(model_id)
                    if (
                        model_info.backend == ModelBackend.VLLM
                        and model_info.readiness_ever_served
                        and model_info.heavy_circuit_tripped_at > 0
                    ):
                        prob = self._probe_backend_readiness(model_info)
                        clean = prob.health_status == 200 and prob.models_status == 200
                        now = time.time()
                        trip = model_info.heavy_circuit_tripped_at
                        debounce = circuit_debounce_seconds_default()
                        if not clean:
                            model_info.heavy_circuit_recovery_probes = 0
                        elif now >= trip + debounce:
                            model_info.heavy_circuit_recovery_probes += 1
                            if model_info.heavy_circuit_recovery_probes >= 2:
                                model_info.heavy_circuit_tripped_at = 0.0
                                model_info.heavy_circuit_recovery_probes = 0
                                logger.info(
                                    "[ADMISSION] heavy circuit closed model=%s (debounce + 2 clean /health)",
                                    model_id,
                                )
            except Exception as e:
                logger.error(f"Error in background health check: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _llama_keep_alive(self):
        """Background task to keep llama-3.1-8b-q4k-q4_k warm and prevent unloading."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Small keep-alive request to llama-3.1
                test_request = {
                    "model": "llama-3.1-8b-q4k-q4_k",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 3,
                    "temperature": 0.1
                }

                start_time = time.time()
                response = await self.chat_completions(test_request)
                end_time = time.time()

                latency = (end_time - start_time) * 1000
                logger.debug(f"[KEEP_ALIVE] ✓ Llama-3.1 keep-alive ping completed in {latency:.1f}ms")

            except Exception as e:
                logger.warning(f"[KEEP_ALIVE] ⚠ Llama-3.1 keep-alive failed: {e}")
                # Continue the loop - don't crash on keep-alive failures