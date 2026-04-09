#!/bin/bash

# Unified API Proxy Startup Script
# Starts the unified proxy server for vLLM and llama.cpp models

# Don't use set -e in background mode (allows PID capture)
if [ "${BACKGROUND:-false}" != "true" ]; then
    set -e
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Resolve paths for new layout
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load .env if present (API key is required for authenticated endpoints)
if [ -f "$REPO_ROOT/.env" ]; then
    # shellcheck disable=SC1091
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

# Same secret is often only in OPENAI_API_KEY (OpenWebUI / BYOK); use it if PROXYAPP_* not set.
PROXYAPP_API_KEY="${PROXYAPP_API_KEY:-${OPENAI_API_KEY:-}}"
export PROXYAPP_API_KEY

if [ -z "${PROXYAPP_API_KEY}" ] && [ "${PROXY_ALLOW_NO_API_KEY:-}" != "1" ]; then
    echo -e "${RED}Error: PROXYAPP_API_KEY is not set after loading ${REPO_ROOT}/.env${NC}"
    echo -e "${YELLOW}Remote clients (Factory CLI, Droid, Tailscale) will get 401 until the proxy and client use the same key.${NC}"
    echo -e "${YELLOW}Set in .env: PROXYAPP_API_KEY=<secret> (or OPENAI_API_KEY=… — start_proxy copies it across).${NC}"
    echo -e "${YELLOW}See: ${REPO_ROOT}/env.example${NC}"
    echo -e "${YELLOW}Local-only escape hatch: PROXY_ALLOW_NO_API_KEY=1 (not for production).${NC}"
    exit 1
fi

# Configuration
PROXY_PORT=${PROXY_PORT:-52415}
MEMORY_THRESHOLD=${MEMORY_THRESHOLD:-0.90}
LOAD_INITIAL=${LOAD_INITIAL:-true}
MODELS_CONFIG="${MODELS_CONFIG:-$REPO_ROOT/configs/models_config.yaml}"

# Determine PID/log locations at repo root
PID_DIR="$REPO_ROOT/pids"
mkdir -p "$PID_DIR"
PID_FILE="$PID_DIR/proxy.pid"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/proxy.log"

# Ensure/activate venv for proxy deps
PYTHON_CMD="python3"
if [ ! -d "$REPO_ROOT/vllm_env" ]; then
    echo -e "${YELLOW}Creating vllm_env virtualenv...${NC}"
    python3 -m venv "$REPO_ROOT/vllm_env"
fi

if [ -d "$REPO_ROOT/vllm_env" ]; then
    echo -e "${GREEN}Activating vllm_env...${NC}"
    # shellcheck disable=SC1091
    source "$REPO_ROOT/vllm_env/bin/activate"
    PYTHON_CMD="$REPO_ROOT/vllm_env/bin/python"
fi

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. GPU memory monitoring will not work.${NC}"
    exit 1
fi

# Install/check required Python packages
if [ -f "$REPO_ROOT/proxy/requirements.txt" ]; then
    echo -e "${YELLOW}Installing proxy Python dependencies...${NC}"
    "$PYTHON_CMD" -m pip install --quiet -r "$REPO_ROOT/proxy/requirements.txt"
fi

echo -e "${YELLOW}Checking dependencies...${NC}"
PYTHONPATH="$SCRIPT_DIR/src" "$PYTHON_CMD" -c "import aiohttp" 2>/dev/null || {
    echo -e "${RED}Error: aiohttp not installed (expected via requirements.txt).${NC}"
    exit 1
}

PYTHONPATH="$SCRIPT_DIR/src" "$PYTHON_CMD" -c "import yaml" 2>/dev/null || {
    echo -e "${YELLOW}Warning: PyYAML not installed (expected via requirements.txt).${NC}"
}

# Start proxy server
echo -e "${GREEN}Starting unified proxy server...${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Proxy Port: $PROXY_PORT"
echo -e "  Memory Threshold: $MEMORY_THRESHOLD"
echo -e "  Load Initial Models: $LOAD_INITIAL"
echo -e "  Models Config: $MODELS_CONFIG"
if [ -n "${PROXYAPP_API_KEY}" ]; then
    echo -e "  ${GREEN}PROXYAPP_API_KEY: set (${#PROXYAPP_API_KEY} chars)${NC}"
else
    echo -e "  ${YELLOW}PROXYAPP_API_KEY: empty (remote auth off for non-loopback)${NC}"
fi
echo ""

# Detect Docker command if not provided (for background mode)
if [ -z "$DOCKER_CMD" ]; then
    DOCKER_CMD="docker"
    if ! docker ps &>/dev/null; then
        if sudo docker ps &>/dev/null; then
            DOCKER_CMD="sudo docker"
        fi
    fi
fi

# Ensure we run from repo root so relative paths (models/logs) align
cd "$REPO_ROOT"

# Base command
BASE_CMD=("$PYTHON_CMD" -m proxyapp.proxy_server \
    --port "$PROXY_PORT" \
    --memory-threshold "$MEMORY_THRESHOLD" \
    --config "$MODELS_CONFIG" \
    --log-file "$LOG_FILE")

if [ "$LOAD_INITIAL" = "true" ]; then
    BASE_CMD+=(--load-initial)
fi

# Run proxy server
if [ "${BACKGROUND:-false}" = "true" ]; then
    # Python logs to LOG_FILE via FileHandler; do not redirect process stdout to the same path
    # (truncation / double-writer races). Capture shell-level noise separately.
    STDIO_LOG="$LOG_DIR/proxy_stdio.log"
    # Pass API key explicitly: some nohup/setsid/cron paths do not inherit a sourced .env reliably.
    nohup setsid env \
        DOCKER_CMD="$DOCKER_CMD" \
        PYTHONPATH="$SCRIPT_DIR/src" \
        PROXYAPP_API_KEY="$PROXYAPP_API_KEY" \
        "${BASE_CMD[@]}" >> "$STDIO_LOG" 2>&1 &
    proxy_pid=$!
    echo $proxy_pid > "$PID_FILE"
    echo "Proxy server started in background (PID: $proxy_pid)"
    echo "Logs: $LOG_FILE"
    echo "Python stderr/uncaught (if any): $STDIO_LOG"
else
    env DOCKER_CMD="$DOCKER_CMD" PYTHONPATH="$SCRIPT_DIR/src" PROXYAPP_API_KEY="$PROXYAPP_API_KEY" "${BASE_CMD[@]}" 2>&1 | tee "$LOG_FILE"
fi
