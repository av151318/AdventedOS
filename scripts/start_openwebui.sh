#!/bin/bash

# OpenWebUI Setup for vLLM Integration
# Provides web chat interface for vLLM models

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}OpenWebUI Setup for vLLM${NC}"
echo -e "${BLUE}=========================${NC}"
echo ""

# Configuration
WEBUI_PORT=${WEBUI_PORT:-3000}
VLLM_BASE_URL=${OPENAI_API_BASE_URL:-${VLLM_BASE_URL:-"http://host.docker.internal:52415/v1"}}
OPENAI_API_KEY=${PROXYAPP_API_KEY:-"test-api-key-123"}

echo -e "${YELLOW}Configuration:${NC}"
echo -e "  WebUI Port: $WEBUI_PORT"
echo -e "  vLLM Base URL: $VLLM_BASE_URL"
echo -e "  API Key: $OPENAI_API_KEY"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Use DOCKER_CMD if provided (for sudo support), otherwise default to docker
DOCKER_CMD=${DOCKER_CMD:-docker}

# Check if OpenWebUI container already exists
if $DOCKER_CMD ps -a --format 'table {{.Names}}' | grep -q "^open-webui$"; then
    echo -e "${YELLOW}OpenWebUI container already exists. Removing old container...${NC}"
    $DOCKER_CMD rm -f open-webui
fi

echo -e "${GREEN}Starting OpenWebUI container...${NC}"

# Start OpenWebUI with vLLM integration
# Use host.docker.internal to reach host services from container
# On Linux, --add-host=host.docker.internal:host-gateway makes host accessible
# Replace localhost/127.0.0.1 with host.docker.internal so container can reach host proxy
CONTAINER_API_URL=$(echo "$VLLM_BASE_URL" | sed "s|localhost|host.docker.internal|g" | sed "s|127.0.0.1|host.docker.internal|g")

echo -e "${YELLOW}OpenWebUI will connect to proxy at: $CONTAINER_API_URL${NC}"
echo -e "${YELLOW}Make sure proxy is running on host at port ${PROXY_PORT:-52415}${NC}"

$DOCKER_CMD run -d \
    --name open-webui \
    -p $WEBUI_PORT:8080 \
    --add-host=host.docker.internal:host-gateway \
    -e ENABLE_OPENAI_API=true \
    -e WEBUI_AUTH=true \
    -e ENABLE_SIGNUP=true \
    -e ENABLE_LOGIN_FORM=true \
    -e OPENAI_API_BASE_URL="$CONTAINER_API_URL" \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -e WEBUI_SECRET_KEY="${OPENAI_API_KEY}" \
    -v open-webui:/app/backend/data \
    --restart unless-stopped \
    ghcr.io/open-webui/open-webui:latest

echo -e "${GREEN}✓ OpenWebUI started successfully!${NC}"
echo ""
echo -e "${BLUE}Access OpenWebUI at: http://localhost:$WEBUI_PORT${NC}"
echo ""
echo -e "${YELLOW}Container logs:${NC}"
echo -e "  $DOCKER_CMD logs -f open-webui"
echo ""
echo -e "${YELLOW}Stop OpenWebUI:${NC}"
echo -e "  $DOCKER_CMD stop open-webui"
echo ""
echo -e "${YELLOW}Remove OpenWebUI:${NC}"
echo -e "  $DOCKER_CMD rm -f open-webui"

