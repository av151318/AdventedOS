#!/bin/bash
# AdventedOS — Start Proxy Platform + OpenWebUI
# Usage: ./start.sh

set -e

# Colors
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PROXYAPP_URL="${PROXYAPP_URL:-http://localhost:52415}"

echo -e "${BLUE}Starting AdventedOS platform...${NC}"
echo ""

# ── 1. Proxy Platform (native) ──────────────────────────────────────
echo -e "${YELLOW}[1/2] Starting Proxy Platform...${NC}"
cd "$REPO_ROOT/proxy"
BACKGROUND=true ./start_proxy.sh
cd "$REPO_ROOT"
sleep 3

if [ -f "$REPO_ROOT/pids/proxy.pid" ] && kill -0 "$(cat "$REPO_ROOT/pids/proxy.pid")" 2>/dev/null; then
    echo -e "${GREEN}  Proxy Platform running (PID: $(cat "$REPO_ROOT/pids/proxy.pid"))${NC}"
else
    echo -e "${RED}  Proxy Platform failed to start. Check logs/proxy.log${NC}"
    exit 1
fi

# ── 2. OpenWebUI (Docker) ───────────────────────────────────────────
echo -e "${YELLOW}[2/2] Starting OpenWebUI...${NC}"
docker compose up -d
sleep 3

if docker ps --format '{{.Names}}' | grep -q open-webui; then
    echo -e "${GREEN}  OpenWebUI container running${NC}"
else
    echo -e "${RED}  OpenWebUI failed to start. Check: docker compose logs${NC}"
fi

# ── Health summary ───────────────────────────────────────────────────
echo ""
echo -e "${BLUE}Health check URLs:${NC}"
echo -e "  Proxy Platform : ${PROXYAPP_URL}/healthcheck"
echo -e "  OpenWebUI      : http://localhost:3000"
echo ""
echo -e "${GREEN}Platform started.${NC}"
