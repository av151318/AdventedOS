# AdventedOS Architecture

## Overview

AdventedOS is a unified AI proxy platform that provides a single OpenAI-compatible API endpoint for locally hosted language models. It abstracts multiple LLM inference backends (vLLM, llama.cpp) behind a consistent interface, enabling OpenWebUI and other OpenAI-compatible clients to work seamlessly with self-hosted models.

## System Components

### 1. Proxy Platform (`proxy/`)

The core of AdventedOS. A native Python service built on aiohttp, running on port 52415.

**Modules:**

| Module | Responsibility |
|--------|---------------|
| `proxy.py` | Main request routing, OpenAI API compatibility layer |
| `proxy_server.py` | aiohttp server entry point, route registration |
| `model_manager.py` | Model lifecycle: load, unload, health checks |
| `memory_manager.py` | GPU VRAM management, model eviction policy |
| `memory_monitor.py` | Real-time GPU memory monitoring via `nvidia-smi` |
| `diagnostics.py` | `/diagnostics` endpoint, system health reporting |
| `chat_history_db.py` | SQLite-backed chat history persistence |
| `request_queue.py` | Async request queuing, concurrency control |

**API Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |
| `GET` | `/v1/models` | List all available models (aggregated) |
| `GET` | `/v1/models/{id}/status` | Model loading status |
| `POST` | `/v1/models/{id}/load` | Load a model on demand |
| `POST` | `/v1/models/{id}/unload` | Unload a model |
| `GET` | `/healthcheck` | Proxy health status |
| `GET` | `/diagnostics` | Detailed system diagnostics |
| `POST` | `/v1/chat/history` | Save chat message |
| `GET` | `/v1/chat/history` | Retrieve chat history by session |
| `GET` | `/v1/chat/sessions` | List all chat sessions |

### 2. OpenWebUI (Docker)

A web-based chat interface that connects to the proxy via the OpenAI-compatible API.

- **Image**: `ghcr.io/open-webui/open-webui:latest`
- **Port**: 3000 (configurable via `WEBUI_PORT`)
- **Connection**: Points `OPENAI_API_BASE_URL` to the proxy's `/v1` endpoint
- **Managed by**: `docker-compose.yml`

### 3. LLM Backends

The proxy manages two inference backend types:

**vLLM (GPU-accelerated)**
- Runs as a Docker container (NGC image)
- Managed by `model_manager.py` — started/stopped on demand
- Optimal for large models requiring full GPU acceleration
- Configuration via YAML files in `vllm_configs/`

**llama.cpp (CPU/GPU hybrid)**
- Runs as a native process
- Managed by `model_manager.py`
- Supports GGUF quantized model format
- Suitable for smaller models or mixed CPU/GPU deployments

### 4. Watchdog (`watchdog/`)

A process monitor that automatically restarts the proxy platform if it crashes or becomes unresponsive.

## Data Flow

```
User
  │
  ▼  HTTP (port 3000)
OpenWebUI (Docker)
  │
  ▼  OpenAI API (port 52415)
Proxy Platform (native, aiohttp)
  │
  ├──► Request Queue
  │       │
  │       ▼
  │    Model Manager ──► vLLM (Docker, GPU)
  │                  └──► llama.cpp (native, CPU/GPU)
  │
  ▼  Streamed response
OpenWebUI → User
```

**Request lifecycle:**

1. OpenWebUI sends `POST /v1/chat/completions` to proxy
2. Proxy authenticates via `PROXYAPP_API_KEY`
3. Request enters the async queue (`request_queue.py`)
4. Model manager checks if the target model is loaded
   - If unloaded: triggers on-demand loading, queues the request
   - If loading: request waits in queue automatically
   - If loaded: request proceeds immediately
5. Request is forwarded to the appropriate backend (vLLM or llama.cpp)
6. Response is streamed back through the proxy to OpenWebUI
7. Chat history is persisted to SQLite

## Model Lifecycle

### Startup Strategy

On startup, the proxy preloads the two smallest models (one per backend) to balance immediate availability with resource efficiency. All other models load on demand when users select them.

### On-Demand Loading

When a user selects an unloaded model:
1. Proxy detects the model is not running
2. Starts the appropriate backend container/process
3. Queues incoming requests during loading (30s–2min typical)
4. Routes queued requests once the model is ready
5. User experience is seamless — requests queue transparently

### Memory Management

- **Monitoring**: GPU memory checked via `nvidia-smi` every 10 seconds
- **Threshold**: 90% VRAM utilization (configurable)
- **Eviction**: Least-recently-used model is unloaded when threshold is exceeded
- **Tracking**: Per-model last-access timestamps for LRU decisions

### Model States

| State | Description |
|-------|-------------|
| `unloaded` | Model not running |
| `loading` | Backend starting up |
| `loaded` | Ready for requests |
| `unloading` | Shutting down |

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `PROXYAPP_API_KEY` | Authentication key for proxy API |
| `PROXYAPP_URL` | Proxy base URL (default: `http://localhost:52415`) |
| `OPENAI_API_BASE_URL` | OpenAI-compatible URL for clients |
| `WEBUI_PORT` | OpenWebUI port (default: `3000`) |
| `WEBUI_SECRET_KEY` | OpenWebUI session secret |
| `PROXY_PORT` | Proxy server port (default: `52415`) |
| `MEMORY_THRESHOLD` | GPU memory threshold (default: `0.90`) |
| `LOAD_INITIAL` | Preload startup models (default: `true`) |

Copy `env.example` to `.env` and configure before starting.

### Model Configuration (`vllm_configs/`)

Each model is defined in a YAML configuration file:

```yaml
# vllm_configs/example-model.yaml
name: my-model
backend: vllm          # or: llama_cpp
model_path: models/my-model
context_length: 8192
gpu_memory_utilization: 0.85
```

The model manager reads these configs to determine available models and loading parameters.

## Deployment

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU + CUDA drivers
- `nvidia-smi` in PATH

### Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/av151318/AdventedOS.git
cd AdventedOS
cp env.example .env
# Edit .env with your configuration

# 2. Start the platform
./start.sh
# Starts proxy (native) + OpenWebUI (Docker)

# 3. Access
# OpenWebUI: http://localhost:3000
# Proxy API: http://localhost:52415/v1/models
```

### Docker Compose

`docker-compose.yml` manages OpenWebUI. The proxy runs natively (not in Docker).

For local-only services:
```bash
cp docker-compose.override.yml.example docker-compose.override.yml
# Add local services — Docker Compose auto-merges both files
```

`docker-compose.override.yml` is gitignored so local additions stay local.

## Directory Structure

```
AdventedOS/
├── proxy/                    # Proxy platform source
│   ├── src/proxyapp/         # Core Python modules
│   │   ├── proxy.py          # Request routing
│   │   ├── proxy_server.py   # Server entry point
│   │   ├── model_manager.py  # Model lifecycle
│   │   ├── memory_manager.py # VRAM management
│   │   ├── memory_monitor.py # GPU monitoring
│   │   ├── diagnostics.py    # Health endpoints
│   │   ├── chat_history_db.py# Chat persistence
│   │   └── request_queue.py  # Request queuing
│   └── requirements.txt
├── vllm_configs/             # Model YAML configs
├── vllm_models/              # Model weights (gitignored)
├── watchdog/                 # Process watchdog
├── scripts/                  # Utility scripts
├── docs/                     # Documentation
├── docker-compose.yml        # OpenWebUI service
├── docker-compose.override.yml.example
├── start.sh                  # Platform startup
├── env.example               # Environment template
└── .env                      # Local config (gitignored)
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Model not found | Returns 404 |
| Model loading | Returns 503 with status; request queued |
| Backend error | Returns 502 |
| Memory threshold exceeded | Auto-evicts LRU model |
| Queue timeout | Request fails after configured period |

## Security Notes

- API authentication via `PROXYAPP_API_KEY` (stored in `.env`, never committed)
- `.env` is gitignored — use `env.example` as the configuration template
- Model weights are never committed to the repository
- `docker-compose.override.yml` is gitignored for local service isolation
- All endpoints require valid API key authentication
