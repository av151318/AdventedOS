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
from pathlib import Path
from typing import Dict, Optional, List, Any, Callable
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

logger = logging.getLogger(__name__)

def normalize_openai_chat_request(body: dict) -> dict:
    """Normalize OpenAI chat completion requests to handle extra fields gracefully.

    This function ensures backward compatibility while allowing modern clients
    to send additional fields like tools, tool_choice, etc.
    
    Also strips Anthropic-specific parameters that OMO may send even when using OpenAI provider.
    """
    # Fields we know and support
    known_fields = {
        'model', 'messages', 'temperature', 'max_tokens', 'top_p', 'n',
        'stream', 'stop', 'presence_penalty', 'frequency_penalty',
        'logit_bias', 'user', 'functions', 'function_call'  # Legacy OpenAI fields
    }

    # Fields to explicitly ignore (log but don't crash)
    ignorable_fields = {
        'tools', 'tool_choice', 'stream_options', 'metadata',
        'response_format', 'seed', 'logprobs', 'top_logprobs'
    }

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
        elif key in ignorable_fields:
            logger.debug(f"normalize_openai_chat_request: ignoring field '{key}' = {value}")
        elif key in anthropic_params:
            logger.info(f"normalize_openai_chat_request: stripping Anthropic param '{key}' = {value}")
            # Don't include in normalized request
        else:
            # Unknown field - log and ignore to be safe
            logger.info(f"normalize_openai_chat_request: unknown field '{key}' ignored for compatibility")

    return normalized

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
        from collections import defaultdict
        self._active_model_requests = defaultdict(int)  # model_id -> count
        self._pinned_models = self._load_pinned_models(config_path) if config_path else {"qwen3-14b"}

        # Memory monitoring can evict unused models when RAM pressure is high.
        # This avoids system-wide thrash that can cause long-latency or timeouts in n8n Chat Hub.
        self.memory_manager = MemoryManager(
            threshold_percent=memory_threshold * 100,
            evict_callback=self._evict_model_if_idle,
            min_idle_seconds=120,
        )
        self.request_queue = RequestQueue()
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
        logger.info("[DIAG] [LIST_MODELS] ========== /v1/models endpoint called ==========")
        models = []
        
        logger.info(f"[DIAG] [LIST_MODELS] Total models in manager: {len(self.model_manager.models)}")
        if not self.model_manager.models:
            logger.error("[DIAG] [LIST_MODELS] ✗ CRITICAL: No models found in ModelManager!")
            logger.error("[DIAG] [LIST_MODELS] This should never happen - models should be initialized in __init__")
        
        for model_id, model_info in self.model_manager.models.items():
            logger.info(f"[DIAG] [LIST_MODELS] Processing model: {model_id}, status: {model_info.status}, port: {model_info.port}")
            
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
            logger.info(f"[DIAG] [LIST_MODELS] Added model: {model_id}, available={is_available}, status={model_info.status}")
        
        logger.info(f"[DIAG] [LIST_MODELS] Returning {len(models)} models")
        logger.info(f"[DIAG] [LIST_MODELS] ========== /v1/models response ready ==========")
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
        # Track request for watchdog monitoring
        self.request_counter += 1
        request_id = f"req_{self.request_counter}_{int(asyncio.get_event_loop().time())}"
        start_time = asyncio.get_event_loop().time()
        self.active_requests[request_id] = start_time

        # Store request for streaming response
        self._current_request = request
        try:
            # Handle both Request objects and dicts
            if hasattr(request, 'json'):
                raw_data = await request.json()
            else:
                raw_data = request

            # Normalize request to handle extra fields gracefully
            raw_data = normalize_openai_chat_request(raw_data)
            # Note: request is now a dict, so we can't access headers directly
            user_agent = "unknown"  # Headers not available in this context
            if "model" in raw_data and any(keyword in raw_data.get("model", "").lower() for keyword in ["qwen", "deepseek", "nous", "llama"]):
                logger.info(f"[OPENCODE-DEBUG] Raw request body: {json.dumps(raw_data, indent=2)}")
                logger.info(f"[OPENCODE-DEBUG] Processing OpenCode request")

            # Normalize the request to handle OpenCode/ai-sdk specific fields
            data = normalize_openai_chat_request(raw_data)
            logger.info(f"[DEBUG] raw_data model: {raw_data.get('model')}")
            logger.info(f"[DEBUG] normalized data model: {data.get('model')}")
            model_id = data.get("model")

            if not model_id:
                return web.json_response(
                    {"error": "model parameter required", "output": [], "choices": []},
                    status=400
                )
            
            # Get model info
            if model_id not in self.model_manager.models:
                return web.json_response(
                    {"error": f"Model {model_id} not found", "output": [], "choices": []},
                    status=404
                )
            
            model_info = self.model_manager.models[model_id]
            
            # Check if model is loaded
            if model_info.status != "loaded":
                # Queue request if model is loading
                if model_info.status == "loading":
                    # Check container status for better UX
                    container_running = self._check_container_running(model_info)
                    container_ready = self._check_container_ready(model_info) if container_running else False
                    
                    # Return clear loading status with progress info
                    if not container_running:
                        message = f"Model {model_id} container is starting..."
                    elif not container_ready:
                        message = f"Model {model_id} is initializing (this may take 1-2 minutes)..."
                    else:
                        message = f"Model {model_id} is almost ready..."
                    
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
                    return await self._forward_request(model_id, data, request)
                
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
            response = await self._forward_request(model_id, data, request)

            # Clean up request tracking on completion
            if request_id in self.active_requests:
                del self.active_requests[request_id]

            return response

        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON"},
                status=400
            )
        except Exception as e:
            logger.error(f"Error handling chat completion: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.error(f"Request data at error: {data}")
            # Clean up request tracking on error
            if 'request_id' in locals() and request_id in self.active_requests:
                del self.active_requests[request_id]
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
        await resp.prepare(request)

        async def send_event(obj: Dict):
            payload = json.dumps(obj, ensure_ascii=False)
            await resp.write(f"data: {payload}\n\n".encode("utf-8"))

        async def send_comment(comment: str = "keep-alive"):
            # SSE comment line - ignored by most clients/parsers
            await resp.write(f": {comment}\n\n".encode("utf-8"))

        # Track in-flight request so memory eviction doesn't unload this model mid-stream.
        self._active_model_requests[model_id] += 1

        # Send created event ASAP (prevents "no response" timeouts)
        await send_event({"type": "response.created", "response": {"id": response_id, "model": model_id, "object": "response"}})
        await send_event({"type": "response.output_item.added", "output_index": 0, "item": {"id": msg_id, "type": "message", "role": "assistant", "content": []}})

        # Ensure model is loaded/ready before we start backend streaming
        try:
            normalized = normalize_openai_chat_request(chat_request)
            model_id = normalized.get("model", model_id)
            if model_id not in self.model_manager.models:
                raise ValueError(f"Model '{model_id}' not available")
            model_info = self.model_manager.models[model_id]

            # Keep access times fresh so memory eviction doesn't target an in-flight stream.
            try:
                self.memory_manager.update_access_time(model_id)
                self.model_manager.update_access_time(model_id)
            except Exception:
                pass

            # IMPORTANT: "status=loaded" can drift from reality (e.g., vLLM container exits).
            # Re-check liveness and restart if needed.
            is_running = self._check_container_running(model_info)
            is_ready = self._check_container_ready(model_info) if is_running else False
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

            visible_accum: list[str] = []
            in_think = False
            pending = ""
            last_write = _time.time()

            def _strip_think_delta(delta: str) -> str:
                nonlocal in_think, pending
                if not isinstance(delta, str) or not delta:
                    return ""

                # Carry over potential partial tag fragments
                text = pending + delta
                pending = ""

                if not in_think:
                    # Hold suffix that could be start of "<think>"
                    tag = "<think>"
                    for l in range(len(tag) - 1, 0, -1):
                        if text.endswith(tag[:l]):
                            pending = text[-l:]
                            text = text[:-l]
                            break
                else:
                    # Hold suffix that could be start of "</think>"
                    tag = "</think>"
                    for l in range(len(tag) - 1, 0, -1):
                        if text.endswith(tag[:l]):
                            pending = text[-l:]
                            text = text[:-l]
                            break

                out: list[str] = []
                i = 0
                while i < len(text):
                    if not in_think:
                        j = text.find("<think>", i)
                        if j == -1:
                            out.append(text[i:])
                            break
                        out.append(text[i:j])
                        i = j + 7
                        in_think = True
                    else:
                        j = text.find("</think>", i)
                        if j == -1:
                            break
                        i = j + 8
                        in_think = False
                return "".join(out)

            async with ClientSession() as session:
                async with session.post(backend_url, json=data, timeout=300) as backend_resp:
                    if backend_resp.status >= 400:
                        body = await backend_resp.text()
                        raise RuntimeError(f"Backend returned {backend_resp.status}: {body[:300]}")

                    buffer = b""
                    async for chunk in backend_resp.content.iter_chunked(8192):
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
                                    buffer = b""
                                    break
                                try:
                                    obj = json.loads(payload.decode("utf-8"))
                                except Exception:
                                    continue

                                # Extract chat-completions streaming delta content
                                delta = ""
                                try:
                                    choice0 = (obj.get("choices") or [])[0] or {}
                                    delta_obj = choice0.get("delta") or {}
                                    delta = delta_obj.get("content") or ""
                                except Exception:
                                    delta = ""

                                visible_delta = _strip_think_delta(delta)
                                if visible_delta:
                                    visible_accum.append(visible_delta)
                                    await send_event({"type": "response.output_text.delta", "delta": visible_delta, "content_index": 0, "output_index": 0})
                                    try:
                                        self.memory_manager.update_access_time(model_id)
                                        self.model_manager.update_access_time(model_id)
                                    except Exception:
                                        pass
                                    last_write = _time.time()
                                else:
                                    # Keep connection alive even if we filtered content (e.g., inside <think>)
                                    if _time.time() - last_write > 5:
                                        try:
                                            self.memory_manager.update_access_time(model_id)
                                            self.model_manager.update_access_time(model_id)
                                        except Exception:
                                            pass
                                        await send_comment()
                                        last_write = _time.time()

                    # Flush any trailing pending text after stream ends
                    if pending and not in_think:
                        visible_accum.append(pending)
                        await send_event({"type": "response.output_text.delta", "delta": pending, "content_index": 0, "output_index": 0})

            visible_text = "".join(visible_accum).strip()
            completed_response = {
                "id": response_id,
                "object": "response",
                "created": int(_time.time()),
                "created_at": int(_time.time()),
                "status": "completed",
                "model": model_id,
                "output": [
                    {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": visible_text}],
                    }
                ] if visible_text else [],
                "output_text": visible_text,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "choices": [],
            }
            await send_event({"type": "response.completed", "response": completed_response})
            await resp.write(b"data: [DONE]\n\n")
            return resp

        except Exception as e:
            logger.error(f"[RESPONSES] Streaming failed: {e}")
            error_response = {
                "id": response_id,
                "object": "response",
                "created": int(_time.time()),
                "created_at": int(_time.time()),
                "status": "failed",
                "model": model_id,
                "error": {"message": str(e), "type": "internal_error", "code": "responses_stream_error"},
                "output": [],
                "output_text": "",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "choices": [],
            }
            await send_event({"type": "response.completed", "response": error_response})
            await resp.write(b"data: [DONE]\n\n")
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
            if model_info and self._check_container_running(model_info) and self._check_container_ready(model_info):
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

        Handles the basic mapping for v0: single-shot text responses.
        Ignores tools, tool_choice, and advanced features for now.
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

        # Copy supported scalar parameters
        if "temperature" in responses_data:
            chat_request["temperature"] = responses_data["temperature"]
        if "max_output_tokens" in responses_data:
            chat_request["max_tokens"] = responses_data["max_output_tokens"]

        # Log that we're ignoring tools for v0
        if "tools" in responses_data or "tool_choice" in responses_data:
            logger.info("Responses API v0: ignoring tools/tool_choice fields - not yet implemented")

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

    async def _forward_request(self, model_id: str, data: Dict, request: Request) -> Response:
        """Forward request to backend and track memory usage"""
        # Track model access for memory management
        self.memory_manager.update_access_time(model_id)
        model_info = self.model_manager.models[model_id]
        self._active_model_requests[model_id] += 1

        # Remap model ID for different backends
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
            # Update model status to unloaded
            model_info.status = "unloaded"
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
        
        # Check if container is ready (health check)
        if not self._check_container_ready(model_info):
            logger.info(f"Container {model_info.container_name} is running but not ready yet - model still initializing")
            # Keep status as loading - return clear message that it's loading, not failed
            return web.json_response(
                {
                    "error": {
                        "message": f"Model {model_id} is still initializing. This typically takes 1-2 minutes for first load.",
                        "type": "model_loading",
                        "code": "model_not_ready"
                    },
                    "status": "loading",
                    "model_id": model_id,
                    "message": "Model container is running but not ready yet. Please wait and retry in 30 seconds.",
                    "output": [],
                    "choices": [],
                },
                status=503
            )
        
        try:
            async with ClientSession() as session:
                # Check if streaming is requested
                stream = data.get("stream", False)
                
                if stream:
                    # Handle streaming response
                    async with session.post(
                        backend_url,
                        json=data,
                        timeout=300
                    ) as resp:
                        # Create streaming response
                        response = web.StreamResponse(status=resp.status)
                        response.content_type = resp.content_type or 'text/event-stream'
                        
                        # Copy headers
                        for header_name, header_value in resp.headers.items():
                            if header_name.lower() not in ['content-length', 'transfer-encoding']:
                                response.headers[header_name] = header_value
                        
                        await response.prepare(request)
                        
                        # Stream chunks
                        async for chunk in resp.content.iter_chunked(8192):
                            await response.write(chunk)
                        
                        return response
                else:
                    # Non-streaming response
                    async with session.post(
                        backend_url,
                        json=data,
                        timeout=300
                    ) as resp:
                        response_data = await resp.json()
                        # Remove non-standard fields for OpenAI API compliance
                        # Chat Completions API should only return standard fields
                        response_data.pop("output", None)
                        return web.json_response(response_data, status=resp.status)
        except ClientError as e:
            logger.error(f"Error forwarding request to {backend_url}: {e}")
            return web.json_response(
                {"error": f"Backend error: {str(e)}", "output": [], "choices": []},
                status=502
            )
        finally:
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
    
    def _check_container_ready(self, model_info) -> bool:
        """Quick check if container is ready (health endpoint responds)"""
        import requests
        try:
            # Quick check - 1 second timeout
            response = requests.get(f"http://localhost:{model_info.port}/v1/models", timeout=1)
            return response.status_code == 200
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
            container_ready = self._check_container_ready(model_info) if container_running else False
            
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
    
    async def _background_health_check(self):
        """Background task to periodically check container health and update model status"""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                for model_id, model_info in self.model_manager.models.items():
                    # Only check models that are marked as "loading"
                    if model_info.status != "loading":
                        continue
                    
                    # Check if container is running
                    if not self._check_container_running(model_info):
                        logger.debug(f"Model {model_id} container not running yet")
                        continue
                    
                    # Check if container is ready
                    if self._check_container_ready(model_info):
                        # Container is ready - mark model as loaded
                        logger.info(f"Model {model_id} container is now ready - updating status to loaded")
                        model_info.status = "loaded"
                        model_info.last_access_time = time.time()
                    else:
                        logger.debug(f"Model {model_id} container running but not ready yet")
                        
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