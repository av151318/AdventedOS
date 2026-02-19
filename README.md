# AdventedOS

> Unified AI proxy platform — route OpenWebUI to local LLMs (vLLM, llama.cpp) via a single OpenAI-compatible API endpoint.

## Overview

AdventedOS is a self-hosted AI proxy platform that provides a unified OpenAI-compatible API endpoint for local language models. It routes requests from OpenWebUI to your choice of vLLM (GPU-accelerated) or llama.cpp (CPU/GPU) backends, with automatic model lifecycle management, GPU memory optimization, and request queuing.

## Architecture

```
OpenWebUI (port 3000)
    │
    ▼ OpenAI-compatible API
Proxy Platform (port 52415)  ← native Python process
    │
    ├── vLLM backend (Docker, GPU)
    └── llama.cpp backend (native, CPU/GPU)
```

The proxy platform runs natively on the host. OpenWebUI runs in Docker and connects to the proxy via `host.docker.internal`.

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU with CUDA (for vLLM) or CPU (for llama.cpp)
- NGC account (for vLLM Docker image)

## Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/av151318/AdventedOS.git
   cd AdventedOS
   ```

2. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your values
   ```

3. **Start the platform**
   ```bash
   ./start.sh
   ```

4. **Open OpenWebUI** — navigate to `http://localhost:3000`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROXYAPP_API_KEY` | API key for proxy authentication | — |
| `PROXYAPP_URL` | Proxy base URL | `http://localhost:52415` |
| `WEBUI_PORT` | OpenWebUI port | `3000` |
| `VLLM_BASE_URL` | vLLM base URL | `http://localhost:52415/v1` |
| `OPENAI_API_BASE_URL` | OpenAI-compatible URL (points to proxy) | — |

Copy `env.example` to `.env` and fill in your values.

### Model Configuration

Model definitions live in `configs/`. Each YAML file defines a model endpoint, backend type, and parameters. The proxy's model manager reads these configs to load/unload models on demand.

## Services

| Service | Port | Type | Description |
|---------|------|------|-------------|
| Proxy Platform | 52415 | Native | OpenAI-compatible API, model lifecycle, GPU memory optimization, request queuing |
| OpenWebUI | 3000 | Docker | Chat UI, connects to proxy via `OPENAI_API_BASE_URL` |

## Project Structure

```
AdventedOS/
├── proxy/                    # Proxy platform (Python)
│   ├── src/proxyapp/         # Core proxy modules
│   │   ├── proxy.py          # Main proxy logic
│   │   ├── proxy_server.py   # aiohttp server entry point
│   │   ├── model_manager.py  # Model lifecycle management
│   │   ├── memory_manager.py # GPU memory management
│   │   ├── memory_monitor.py # Memory monitoring
│   │   ├── diagnostics.py    # Diagnostics endpoints
│   │   ├── chat_history_db.py# SQLite chat history
│   │   └── request_queue.py  # Request queuing
│   ├── start_proxy.sh        # Proxy startup script
│   └── requirements.txt      # Python dependencies
├── configs/                  # Model configuration files (YAML)
├── watchdog/                 # Process watchdog service
├── scripts/                  # Startup and utility scripts
├── docs/                     # Documentation
│   └── ARCHITECTURE.md       # Architecture reference
├── docker-compose.yml        # OpenWebUI Docker service
├── docker-compose.override.yml.example
├── start.sh                  # Platform startup script
├── demo_proxy_platform.sh    # Demo / all-in-one launcher
└── env.example               # Environment variable template
```

## Building llama.cpp

If using llama.cpp backends, build once from the repo root:

```bash
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

Omit `-DGGML_CUDA=ON` for CPU-only builds. The binary at `llama.cpp/build/bin/llama-server` is resolved relative to the repo root.

## Local Development

To add local-only Docker services without committing them:

```bash
cp docker-compose.override.yml.example docker-compose.override.yml
# Edit with your local services
docker compose up -d  # Docker Compose auto-merges both files
```

The override file is gitignored — your local additions stay local.

## Operations

| Action | Command |
|--------|---------|
| Start platform | `./start.sh` or `./demo_proxy_platform.sh` |
| Health check | `curl http://localhost:52415/healthcheck` |
| List models | `curl http://localhost:52415/v1/models` |
| Stop | Ctrl+C in launcher; containers cleaned automatically |

### Watchdog

The `watchdog/` directory contains a service monitor that automatically restarts the proxy if it crashes. Install via the included systemd unit. Control with `watchdog/watchdog-control.sh`.

## License

See [LICENSE](LICENSE) for details.
