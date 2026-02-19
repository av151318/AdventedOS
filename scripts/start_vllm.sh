#!/bin/bash

# DGX Spark vLLM NGC Container Launcher Script
# Start vLLM servers using NVIDIA NGC containers for CUDA 13.0 compatibility

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}DGX Spark vLLM NGC Platform${NC}"
echo -e "${BLUE}============================${NC}"
echo -e "${YELLOW}Using NVIDIA NGC vLLM 25.09+ for CUDA 13.0 compatibility${NC}"
echo ""

# NGC Container configuration
VLLM_IMAGE="nvcr.io/nvidia/vllm:25.09-py3"
WORKSPACE_DIR="$(pwd)"

# Function to start a vLLM server using NGC container
start_vllm_server() {
    local model_name=$1
    local config_file=$2
    local port=$3

    echo -e "${GREEN}Starting vLLM NGC container for ${model_name}...${NC}"

    # Check if config file exists
    if [ ! -f "$config_file" ]; then
        echo -e "${RED}Error: Config file $config_file not found!${NC}"
        return 1
    fi

    # Check if model directory exists
    local model_path="vllm_models/$model_name"
    if [ ! -d "$model_path" ]; then
        echo -e "${YELLOW}Warning: Model directory $model_path not found, skipping $model_name${NC}"
        return 1
    fi

    # Container name for this model
    local container_name="vllm-${model_name,,}"

    # Start NGC container in background
    docker run --user root -d \
        --name "$container_name" \
        --gpus all \
        --shm-size=16GB \
        -p "$port:$port" \
        -v "$WORKSPACE_DIR/vllm_configs:/app/configs:ro" \
        -v "$WORKSPACE_DIR/logs:/app/logs" \
        -v "$WORKSPACE_DIR/pids:/app/pids" \
        --env PYTHONPATH=/app \
        "$VLLM_IMAGE" \
        python -m vllm.entrypoints.openai.api_server \
        --config "/app/configs/$(basename "$config_file")" \
        --host 0.0.0.0 \
        --port "$port"

    echo -e "${GREEN}✓ ${model_name} NGC container started (Port: $port)${NC}"
    echo -e "${YELLOW}  Container: $container_name${NC}"
    echo -e "${YELLOW}  API: http://localhost:$port/v1/chat/completions${NC}"
    echo ""
}

# Create directories
mkdir -p logs pids

# Available models
declare -A models=(
    ["Qwen3-14B"]="vllm_configs/qwen3-14b.yaml"
    ["gpt-oss-20b"]="vllm_configs/gpt-oss-20b.yaml"
    ["Qwen3-32B"]="vllm_configs/qwen3-32b.yaml"
)

# Start servers for available models
echo -e "${YELLOW}Starting vLLM servers...${NC}"
echo ""

for model in "${!models[@]}"; do
    config_file="${models[$model]}"

    # Extract port from config file
    port=$(grep -E "^port:" "$config_file" | awk '{print $2}' | tr -d ' ')

    if [ -z "$port" ]; then
        echo -e "${RED}Error: Could not find port in $config_file${NC}"
        continue
    fi

    start_vllm_server "$model" "$config_file" "$port"
done

echo -e "${GREEN}All vLLM NGC containers started successfully!${NC}"
echo ""
echo -e "${BLUE}Container Status:${NC}"
echo -e "  Qwen3-14B: http://localhost:8000 (if available)"
echo -e "  GPT-OSS-20B: http://localhost:8004 (if available)"
echo ""
echo -e "${BLUE}Web UI Integration:${NC}"
echo -e "  OpenWebUI: docker run --user root -p 3000:8080 -e OPENAI_API_BASE_URL=http://localhost:8000/v1 ..."
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all containers${NC}"

# Function to stop all vLLM containers
stop_all_containers() {
    echo ""
    echo -e "${YELLOW}Stopping all vLLM NGC containers...${NC}"

    # Stop containers
    sudo docker stop $(sudo docker ps -q --filter "name=vllm-") 2>/dev/null || true

    # Remove containers
    docker rm $(sudo docker ps -aq --filter "name=vllm-") 2>/dev/null || true

    echo -e "${GREEN}All containers stopped and removed${NC}"
    exit 0
}

# Wait for user interrupt
trap stop_all_containers INT

# Keep script running
while true; do
    sleep 1
done
