#!/bin/bash

# vLLM Health Check Script
# Simple health check for vLLM servers - used in deployment and monitoring

set -e

# Default configuration
HOST=${VLLM_HOST:-localhost}
PORT=${VLLM_PORT:-8000}
TIMEOUT=${HEALTH_TIMEOUT:-10}
QUIET=${QUIET:-false}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2

    case $status in
        "success")
            echo -e "${GREEN}✓${NC} $message"
            ;;
        "warning")
            echo -e "${YELLOW}⚠${NC} $message"
            ;;
        "error")
            echo -e "${RED}✗${NC} $message"
            ;;
        "info")
            echo -e "$message"
            ;;
    esac
}

# Function to check if port is open
check_port() {
    if timeout $TIMEOUT bash -c "</dev/tcp/$HOST/$PORT" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to check vLLM health endpoint
check_health_endpoint() {
    local response
    response=$(timeout $TIMEOUT curl -s -w "%{http_code}" -o /dev/null "http://$HOST:$PORT/health" 2>/dev/null || echo "000")

    if [ "$response" = "200" ]; then
        return 0
    else
        return 1
    fi
}

# Function to check vLLM API endpoints
check_api_endpoints() {
    local endpoints=("/v1/models" "/docs")
    local failed=0

    for endpoint in "${endpoints[@]}"; do
        local response
        response=$(timeout $TIMEOUT curl -s -w "%{http_code}" -o /dev/null "http://$HOST:$PORT$endpoint" 2>/dev/null || echo "000")

        if [ "$response" != "200" ]; then
            failed=$((failed + 1))
        fi
    done

    if [ $failed -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# Main health check
main() {
    if [ "$QUIET" != "true" ]; then
        echo "Checking vLLM health at $HOST:$PORT..."
    fi

    # Check 1: Port connectivity
    if ! check_port; then
        if [ "$QUIET" != "true" ]; then
            print_status "error" "Cannot connect to $HOST:$PORT"
        fi
        exit 1
    elif [ "$QUIET" != "true" ]; then
        print_status "success" "Port $PORT is accessible"
    fi

    # Check 2: Health endpoint
    if ! check_health_endpoint; then
        if [ "$QUIET" != "true" ]; then
            print_status "error" "Health endpoint not responding"
        fi
        exit 1
    elif [ "$QUIET" != "true" ]; then
        print_status "success" "Health endpoint responding"
    fi

    # Check 3: API endpoints
    if ! check_api_endpoints; then
        if [ "$QUIET" != "true" ]; then
            print_status "warning" "Some API endpoints not responding"
        fi
        exit 1
    elif [ "$QUIET" != "true" ]; then
        print_status "success" "API endpoints responding"
    fi

    # All checks passed
    if [ "$QUIET" != "true" ]; then
        print_status "success" "vLLM server is healthy"
    fi
    exit 0
}

# Show usage if requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Check vLLM server health"
    echo ""
    echo "Options:"
    echo "  VLLM_HOST=host        Server host (default: localhost)"
    echo "  VLLM_PORT=port        Server port (default: 8000)"
    echo "  HEALTH_TIMEOUT=secs   Timeout in seconds (default: 10)"
    echo "  QUIET=true           Suppress output, only return exit code"
    echo ""
    echo "Exit codes:"
    echo "  0 - Server is healthy"
    echo "  1 - Server is not healthy"
    exit 0
fi

# Run main function
main
